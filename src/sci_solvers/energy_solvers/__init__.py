"""
Energy solver implementations.

This module contains concrete implementations of EnergyCalculatorBase
using different backends:
- PySCF (CPU, single-node)
- torch_sci_solver (GPU, CPU, single-node)
- DICE (CPU, MPI-based)
- sbd (GPU, RIKEN's accelerated solver, MPI-based)
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import TYPE_CHECKING, Any

__all__ = [
    "PySCFSolver",
    "PySCFSolverSpinFilterMixin",
    "DICESolver",
    "DICESolverSpinFilterMixin",
    "DavidsonSolver",
    "DavidsonSolverSpinFilterMixin",
    "SBDSolver",
    "SolverType",
    "AVAILABLE_SOLVERS",
    "create_solver",
    "solver_class",
    "is_complex_capable",
    "supports_multiroot",
    "is_general_algebra",
    "supports_spinless",
    # general-algebra solvers (diagonalize via the EigenProblem reduction)
    "GeneralAlgebraSolverBase",
    "SpinFilterMixin",
    "ArpackSolver",
    "MultiRootDavidsonSolver",
    "LobpcgSolver",
    "PrimmeSolver",
    "SlepcSolver",
    "EvslSolver",
    "PyscfDavidsonSolver",
]

if TYPE_CHECKING:
    from sci_solvers.energy_solvers.davidson_solver import (
        DavidsonSolver,
        DavidsonSolverSpinFilterMixin,
    )
    from sci_solvers.energy_solvers.dice_solver import DICESolver, DICESolverSpinFilterMixin
    from sci_solvers.energy_solvers.factory import (
        AVAILABLE_SOLVERS,
        SolverType,
        create_solver,
        is_complex_capable,
        is_general_algebra,
        solver_class,
        supports_multiroot,
        supports_spinless,
    )
    from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase
    from sci_solvers.energy_solvers.general_algebra_solvers import (
        ArpackSolver,
        EvslSolver,
        LobpcgSolver,
        MultiRootDavidsonSolver,
        PrimmeSolver,
        PyscfDavidsonSolver,
        SlepcSolver,
    )
    from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver, PySCFSolverSpinFilterMixin
    from sci_solvers.energy_solvers.sbd_solver import SBDSolver
    from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin

_LAZY_IMPORTS = {
    "DICESolver": "sci_solvers.energy_solvers.dice_solver",
    "DICESolverSpinFilterMixin": "sci_solvers.energy_solvers.dice_solver",
    "DavidsonSolver": "sci_solvers.energy_solvers.davidson_solver",
    "DavidsonSolverSpinFilterMixin": "sci_solvers.energy_solvers.davidson_solver",
    "PySCFSolver": "sci_solvers.energy_solvers.pyscf_solver",
    "PySCFSolverSpinFilterMixin": "sci_solvers.energy_solvers.pyscf_solver",
    "SBDSolver": "sci_solvers.energy_solvers.sbd_solver",
    "SolverType": "sci_solvers.energy_solvers.factory",
    "AVAILABLE_SOLVERS": "sci_solvers.energy_solvers.factory",
    "create_solver": "sci_solvers.energy_solvers.factory",
    "solver_class": "sci_solvers.energy_solvers.factory",
    "is_complex_capable": "sci_solvers.energy_solvers.factory",
    "supports_multiroot": "sci_solvers.energy_solvers.factory",
    "is_general_algebra": "sci_solvers.energy_solvers.factory",
    "supports_spinless": "sci_solvers.energy_solvers.factory",
    "GeneralAlgebraSolverBase": "sci_solvers.energy_solvers.general_algebra_base",
    "SpinFilterMixin": "sci_solvers.energy_solvers.spin_filter_mixin",
    "ArpackSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "MultiRootDavidsonSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "LobpcgSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "PrimmeSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "SlepcSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "EvslSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
    "PyscfDavidsonSolver": "sci_solvers.energy_solvers.general_algebra_solvers",
}


def __getattr__(name: str) -> Any:
    """Lazy import attributes using PEP 562."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
