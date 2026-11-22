"""Sanity-suite cases from YAML: one case file, a suite file, or a directory of either.

A **case** names a Hamiltonian and a determinant space::

    name: my_molecule                  # optional; default: the file stem
    hamiltonian:
      fcidump: path/to/molecule.fcidump  # or h5: <scalar_mo .h5>, or builtin: <name>
      nelec: [5, 5]                    # optional override of the file's (n_alpha, n_beta)
    determinants:                      # optional; default: the full CI space
      alphadets: path/to/alpha_dets.txt  # one bitstring per line, orbital 0 rightmost
      betadets: ...                    # optional; default: the alpha strings
      n_dets: 60                       # keep the first n_dets of each
    solvers: [arpack, arpack@fulqrum:csr_fast]  # optional; <name>[@<backend>[:<fulqrum_mode>]]
    seeds: [0, 1, 2]                   # optional ARPACK start-vector seeds; default [0]
    expected: fail                     # optional: a FAIL reports XFAIL, a PASS XPASS

A **suite** file has a ``datasets:`` list of cases (inline, or ``{case: <path>}``), plus
optional ``defaults:`` (``solvers``, ``seeds``, ``determinants``) and ``datasets_dir:`` (every
``*.yaml`` there is a case). Datasets are unioned by name; an explicit entry overrides a
discovered case of the same name, key by key. Relative paths resolve against the file
that names them.

``builtin`` names the PySCF Hamiltonians of :mod:`gdb_physical`: ``real_molecule``,
``rotated_radical`` and ``twisted_hydrogen_chain`` (which takes ``nelec``).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gdb_physical import (
    PhysicalHamiltonian,
    real_molecule,
    rotated_radical,
    twisted_hydrogen_chain,
)

from sci_solvers.hamiltonian import HamiltonianData

DEFAULT_SOLVERS = ["arpack"]
_BUILTINS = {
    "real_molecule": real_molecule,
    "rotated_radical": rotated_radical,
    "twisted_hydrogen_chain": twisted_hydrogen_chain,
}


@dataclass
class Case:
    """One dataset as read from YAML, not yet loaded.

    Attributes:
        name: Report label.
        spec: The case mapping, with every relative path already made absolute.
        solvers: Solvers to run it on.
        expected_fail: Whether a FAIL is expected (reported XFAIL).
    """

    name: str
    spec: dict[str, Any]
    solvers: list[str]
    expected_fail: bool = False


class SkipCase(Exception):
    """Raised by a loader when a case can't run here (e.g. an optional package is missing)."""


def _absolute(spec: dict[str, Any], base: Path) -> dict[str, Any]:
    """Copy of ``spec`` with its path-valued keys resolved against ``base``."""
    out = dict(spec)
    for section, keys in (
        ("hamiltonian", ("fcidump", "h5")),
        ("determinants", ("alphadets", "betadets")),
    ):
        if isinstance(out.get(section), dict):
            sub = dict(out[section])
            for k in keys:
                if k in sub:
                    sub[k] = str((base / sub[k]).resolve())
            out[section] = sub
    return out


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``over`` on top of ``base``, merging one level of nested mappings."""
    out = dict(base)
    for k, v in over.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _case_entry(spec: dict[str, Any], base: Path, default_name: str) -> dict[str, Any]:
    """One case mapping with absolute paths and a name."""
    if "case" in spec:
        path = (base / spec["case"]).resolve()
        rest = {k: v for k, v in spec.items() if k != "case"}
        return _merge(_case_entry(_read(path), path.parent, path.stem), rest)
    return {"name": default_name, **_absolute(spec, base)}


def _suite_entries(path: Path) -> list[dict[str, Any]]:
    """The case mappings a case file, suite file or directory stands for."""
    if path.is_dir():
        return [e for p in sorted(path.glob("*.yaml")) for e in _suite_entries(p)]
    data = _read(path)
    base = path.parent
    if "datasets" not in data and "datasets_dir" not in data:
        return [_case_entry(data, base, path.stem)]
    defaults = data.get("defaults", {})
    by_name: dict[str, dict[str, Any]] = {}
    if "datasets_dir" in data:
        for e in _suite_entries((base / data["datasets_dir"]).resolve()):
            by_name[e["name"]] = e
    for i, spec in enumerate(data.get("datasets", [])):
        e = _case_entry(spec, base, f"{path.stem}[{i}]")
        by_name[e["name"]] = _merge(by_name.get(e["name"], {}), e)
    return [_merge(defaults, e) for e in by_name.values()]


def discover(paths: list[Path], solvers: list[str] | None = None) -> list[Case]:
    """Cases from every path (case file, suite file or directory), in order.

    ``solvers`` overrides every case's own solver list.
    """
    cases = []
    for path in paths:
        for e in _suite_entries(path.resolve()):
            cases.append(
                Case(
                    name=e["name"],
                    spec=e,
                    solvers=solvers or e.get("solvers") or DEFAULT_SOLVERS,
                    expected_fail=e.get("expected") == "fail",
                )
            )
    return cases


def builtin_cases(solvers: list[str] | None = None) -> list[Case]:
    """The suite's own PySCF cases, used when no path is given."""
    specs = [
        ("n2", {"builtin": "real_molecule"}),
        ("nh2_rotated", {"builtin": "rotated_radical"}),
        *(
            (f"h4_twisted_{a}{b}", {"builtin": "twisted_hydrogen_chain", "nelec": [a, b]})
            for a, b in [(2, 2), (2, 1), (3, 1)]
        ),
    ]
    return [
        Case(name, {"name": name, "hamiltonian": h}, solvers or DEFAULT_SOLVERS)
        for name, h in specs
    ]


def _bitstrings(path: str, n_dets: int | None) -> np.ndarray:
    """Occupation integers from a file of bitstrings, one per line, orbital 0 rightmost."""
    lines = Path(path).read_text().split()
    if n_dets is not None:
        lines = lines[:n_dets]
    return np.array(sorted({int(x, 2) for x in lines}), dtype=np.int64)


def _hamiltonian(name: str, spec: dict[str, Any]) -> PhysicalHamiltonian:
    nelec = tuple(spec["nelec"]) if "nelec" in spec else None
    if "builtin" in spec:
        fn: Any = _BUILTINS[spec["builtin"]]
        return fn(nelec) if nelec is not None else fn()
    if "fcidump" in spec:
        data = HamiltonianData.from_fcidump(spec["fcidump"])
        hcore, eri, norb = data.hcore, data.eri, data.num_orbitals
        file_nelec = (data.num_elec_a, data.num_elec_b)
    elif "h5" in spec:
        import h5py

        with h5py.File(spec["h5"], "r") as f:
            attrs = dict(f.attrs)
            if bool(attrs.get("spinless", attrs.get("basis_repr") == "spinor_mo")):
                raise SkipCase("spinless (spinor_mo) h5: GDB needs (n_alpha, n_beta)")
            hcore, eri = np.asarray(f["hcore"][:]), np.asarray(f["eri"][:])
        norb = hcore.shape[0]
        if "n_alpha_at_k" in attrs and "n_beta_at_k" in attrs:
            file_nelec = (int(attrs["n_alpha_at_k"]), int(attrs["n_beta_at_k"]))
        else:
            n, spin = int(attrs["nelectron"]), int(attrs.get("spin", 0))
            file_nelec = ((n + spin) // 2, (n - spin) // 2)
    else:
        raise ValueError(f"{name}: hamiltonian needs fcidump, h5 or builtin")
    return PhysicalHamiltonian(
        name=name,
        hcore=hcore,
        eri=eri,
        norb=norb,
        nelec=nelec or file_nelec,  # type: ignore[arg-type]
    )


def load(case: Case) -> tuple[PhysicalHamiltonian, tuple[np.ndarray, np.ndarray] | None]:
    """The Hamiltonian and determinant product of ``case`` (``None``: full CI space).

    Raises:
        SkipCase: the case can't run here.
    """
    spec = case.spec
    ham = _hamiltonian(case.name, spec["hamiltonian"])
    ham.name = case.name
    dets = spec.get("determinants")
    if not dets or dets.get("full_ci"):
        return ham, None
    n = dets.get("n_dets")
    alpha = _bitstrings(dets["alphadets"], n)
    beta = _bitstrings(dets["betadets"], n) if "betadets" in dets else alpha
    return ham, (alpha, beta)
