"""Two additions to upstream ``qiskit-addon-dice-solver``, applied on import.

Upstream hard-codes both the location of the ``Dice`` executable (the ``bin/Dice``
shipped inside the installed package) and its Davidson tolerance (``davidsonTol 1e-5``).
This module adds, without forking upstream:

* :func:`resolve_dice_binary` -- find ``Dice`` via ``DICE_BINARY``, ``DICE_BIN_DIR`` or
  ``PATH`` before falling back to the packaged binary, so a separately built ``Dice``
  (``scripts/energy_solvers/build_dice.sh``) can be used;
* :func:`solve_hci` -- upstream's ``solve_hci`` plus ``davidson_tol`` /
  ``davidson_tol_loose``, written to Dice's ``input.dat`` as ``davidsonTol`` /
  ``davidsonTolLoose``.

Both work by replacing two **private** upstream functions, ``_call_dice`` and
``_write_input_files``, which upstream's ``solve_hci`` looks up as module globals at
call time, so it relies on the exact upstream commit pinned in ``pyproject.toml``.
With the defaults (no env vars, ``davidson_tol=1e-5``,
``davidson_tol_loose=None``), upstream's behavior is unchanged.

Notes
-----
``_call_dice`` below is upstream's own, altered only to call :func:`resolve_dice_binary`.
That upstream code is (C) Copyright IBM 2024, a Qiskit project licensed under the
Apache License, Version 2.0.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os
import shutil
import subprocess
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from qiskit_addon_dice_solver import dice_solver as _upstream

# Upstream writes exactly this line; the writer below swaps it for the requested value.
_UPSTREAM_DAVIDSON_TOL_LINE = "davidsonTol 1e-5\n"

# (davidson_tol, davidson_tol_loose) for the solve_hci call in progress. The default is
# upstream's hard-coded value, so it leaves upstream's input.dat untouched.
_DEFAULT_TOLERANCES: tuple[float, float | None] = (1e-5, None)
_TOLERANCES: ContextVar[tuple[float, float | None]] = ContextVar(
    "dice_davidson_tolerances", default=_DEFAULT_TOLERANCES
)

_upstream_write_input_files = _upstream._write_input_files


def resolve_dice_binary() -> str:
    """Return the path to the ``Dice`` executable that :func:`solve_hci` will run.

    Resolution order (first hit wins):

    1. ``DICE_BINARY`` env var -- an explicit path to the ``Dice`` executable.
    2. ``DICE_BIN_DIR`` env var -- a directory expected to contain ``Dice``.
    3. ``Dice`` on ``PATH``.
    4. The ``bin/Dice`` shipped inside ``qiskit_addon_dice_solver`` (upstream's only
       location).

    The path is not checked for existence; a missing binary surfaces as upstream's
    ``DiceExecutionError`` when Dice is run.
    """
    env_binary = os.environ.get("DICE_BINARY")
    if env_binary:
        return env_binary
    env_bin_dir = os.environ.get("DICE_BIN_DIR")
    if env_bin_dir:
        return str(Path(env_bin_dir) / "Dice")
    on_path = shutil.which("Dice")
    if on_path is not None:
        return on_path
    return str(Path(_upstream.DICE_BIN) / "Dice")


def _call_dice(dice_dir: Path, mpirun_options: Sequence[str] | str | None) -> None:
    """Upstream's ``_call_dice``, running the binary from :func:`resolve_dice_binary`."""
    dice_path = resolve_dice_binary()
    dice_log_path = os.path.join(dice_dir, "dice_solver_logfile.log")  # noqa: PTH118
    if mpirun_options:
        if isinstance(mpirun_options, str):
            mpirun_options = mpirun_options.split()
        dice_call = ["mpirun", *list(mpirun_options), dice_path]
    else:
        dice_call = ["mpirun", dice_path]

    with open(dice_log_path, "w") as logfile:  # noqa: PTH123
        process = subprocess.run(dice_call, cwd=dice_dir, stdout=logfile, stderr=logfile)
    # Dice returns non-zero codes on successful runs, so success is judged by its output.
    if not (Path(dice_dir) / "spin1RDM.0.0.txt").exists():
        raise _upstream.DiceExecutionError(
            command=dice_call,
            returncode=process.returncode,
            log_path=dice_log_path,
        )


def _write_input_files(*args: Any, dice_dir: str | Path, **kwargs: Any) -> None:
    """Upstream's writer, then set ``davidsonTol`` / ``davidsonTolLoose`` in ``input.dat``."""
    _upstream_write_input_files(*args, dice_dir=dice_dir, **kwargs)
    davidson_tol, davidson_tol_loose = _TOLERANCES.get()
    if (davidson_tol, davidson_tol_loose) == _DEFAULT_TOLERANCES:
        return  # keep upstream's input.dat byte for byte
    path = Path(dice_dir) / "input.dat"
    text = path.read_text()
    if _UPSTREAM_DAVIDSON_TOL_LINE not in text:
        raise RuntimeError(
            f"upstream qiskit-addon-dice-solver no longer writes {_UPSTREAM_DAVIDSON_TOL_LINE!r};"
            " re-check sci_solvers.energy_solvers._dice_compat against the pinned version"
        )
    lines = f"davidsonTol {davidson_tol}\n"
    if davidson_tol_loose is not None:
        lines += f"davidsonTolLoose {davidson_tol_loose}\n"
    path.write_text(text.replace(_UPSTREAM_DAVIDSON_TOL_LINE, lines, 1))


def solve_hci(
    *args: Any,
    davidson_tol: float = 1e-5,
    davidson_tol_loose: float | None = None,
    **kwargs: Any,
) -> Any:
    """Upstream ``solve_hci`` with Dice's two Davidson tolerances exposed.

    Args:
        *args: Forwarded to upstream ``solve_hci``.
        davidson_tol: Residual tolerance for Dice's final (tight) Davidson solve
            (``davidsonTol``). The default, ``1e-5``, is upstream's hard-coded value.
        davidson_tol_loose: Residual tolerance for the preliminary Davidson solve of
            each HCI macro-iteration (``davidsonTolLoose``). ``None`` omits the keyword,
            so Dice keeps its own default.
        **kwargs: Forwarded to upstream ``solve_hci``.

    Returns:
        Whatever upstream ``solve_hci`` returns.
    """
    token = _TOLERANCES.set((davidson_tol, davidson_tol_loose))
    try:
        return _upstream.solve_hci(*args, **kwargs)
    finally:
        _TOLERANCES.reset(token)


_upstream._call_dice = _call_dice
_upstream._write_input_files = _write_input_files
