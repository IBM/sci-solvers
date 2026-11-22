"""Resolves filesystem paths for the bundled test molecules.

The package ships, as data, the FCIDUMP and a sample of SqDRIFT quantum-hardware shots for
two molecules from the SQD molecules dataset (see ``data/README.md``):

- ``"methane"`` (dataset ID 001, 10 electrons in 10 orbitals): small enough for exact full CI;
- ``"benzoic_acid"`` (dataset ID 182, 36 electrons in 30 orbitals): a larger active space.

Tests and scripts get them through :func:`fcidump_path` and :func:`shots_path`. Environment
variables override the bundled files:

    export TORCH_SCI_SOLVER_FCIDUMP=/path/to/molecule.fcidump
    export TORCH_SCI_SOLVER_SHOTS=/path/to/shots.npz
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple, Optional


class Molecule(NamedTuple):
    """A bundled test molecule: its data-file stem and active space."""

    stem: str
    norb: int
    nelec_a: int
    nelec_b: int


MOLECULES = {
    "methane": Molecule("methane_001", norb=10, nelec_a=5, nelec_b=5),
    "benzoic_acid": Molecule("benzoic_acid_182", norb=30, nelec_a=18, nelec_b=18),
}

# The molecule tests use when they need one realistic, larger active space.
DEFAULT_MOLECULE = "benzoic_acid"


def _get_env_path(name: str) -> Optional[Path]:
    """Get env path."""
    val = os.environ.get(name)
    if val is None or val == "":
        return None
    p = Path(val).expanduser()
    return p


def _bundled(filename: str) -> Optional[Path]:
    path = Path(str(files("torch_sci_solver") / "data" / filename))
    return path if path.exists() else None


def fcidump_path(molecule: str = DEFAULT_MOLECULE) -> Optional[Path]:
    """FCIDUMP of a bundled test molecule.

    Resolution order:
      1. ``TORCH_SCI_SOLVER_FCIDUMP`` env var.
      2. The molecule's FCIDUMP bundled in ``torch_sci_solver/data/``.
      3. None — caller decides what to do (skip test, raise, etc).
    """
    p = _get_env_path("TORCH_SCI_SOLVER_FCIDUMP")
    if p is not None and p.exists():
        return p
    return _bundled(f"{MOLECULES[molecule].stem}.fcidump")


def shots_path(molecule: str = DEFAULT_MOLECULE) -> Optional[Path]:
    """SqDRIFT shots sample of a bundled test molecule.

    Resolution order:
      1. ``TORCH_SCI_SOLVER_SHOTS`` env var.
      2. The molecule's shots bundled in ``torch_sci_solver/data/``.
      3. None — caller decides what to do (skip test, raise, etc).
    """
    p = _get_env_path("TORCH_SCI_SOLVER_SHOTS")
    if p is not None and p.exists():
        return p
    return _bundled(f"{MOLECULES[molecule].stem}_sqdrift_shots_10k.npz")


def hf_and_singles(norb: int, nelec: int) -> list[int]:
    """Hartree-Fock string plus all its single excitations, sorted ascending.

    A deterministic basis with the Hartree-Fock determinant first (lowest integer) that
    exercises every matvec task.
    """
    hf = (1 << nelec) - 1
    return sorted(
        {hf} | {hf ^ (1 << i) ^ (1 << a) for i in range(nelec) for a in range(nelec, norb)}
    )
