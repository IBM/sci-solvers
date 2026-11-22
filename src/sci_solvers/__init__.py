"""
Physics module for quantum chemistry calculations.

This module contains physics-related functionality for quantum chemistry,
including energy calculations from determinants using various solvers.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import TYPE_CHECKING, Any

# Make a subsequent bare `import sbd` safe on an HPC login node: sbd runs a
# singleton MPI_Init at import that otherwise hangs probing the cluster fabric.
# Idempotent and a no-op under mpirun/srun, so this costs nothing when sbd is
# never imported or when running multi-node. SBDSolver.__init__ also calls it,
# so the supported solver path is safe even without importing this package.
from sci_solvers._mpi_singleton import configure_singleton_mpi as _configure_singleton_mpi

_configure_singleton_mpi()

__all__ = [
    "EnergyCalculatorBase",
]

if TYPE_CHECKING:
    from sci_solvers.energy_calculation_base import (
        EnergyCalculatorBase,
    )

_LAZY_IMPORTS = {
    "EnergyCalculatorBase": "sci_solvers.energy_calculation_base",
}


def __getattr__(name: str) -> Any:
    """Lazy import attributes using PEP 562."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
