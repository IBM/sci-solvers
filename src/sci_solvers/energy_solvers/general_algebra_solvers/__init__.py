"""General-algebra selected-CI solvers, one per module.

Each backend hands the projected CI subspace to a general-purpose eigensolver and
is a small subclass of
:class:`~sci_solvers.energy_solvers.general_algebra_base.GeneralAlgebraSolverBase`
implementing only ``_diagonalize``; spin projection comes from
:class:`~sci_solvers.energy_solvers.spin_filter_mixin.SpinFilterMixin` (except
:class:`PyscfDavidsonSolver`, which projects natively via PySCF's ``fix_spin_``).

Re-exported here so ``from sci_solvers.energy_solvers.general_algebra_solvers
import ArpackSolver`` keeps working after the split into per-solver files.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from sci_solvers.energy_solvers.general_algebra_solvers.arpack import ArpackSolver
from sci_solvers.energy_solvers.general_algebra_solvers.davidson import MultiRootDavidsonSolver
from sci_solvers.energy_solvers.general_algebra_solvers.evsl import EvslSolver
from sci_solvers.energy_solvers.general_algebra_solvers.lobpcg import LobpcgSolver
from sci_solvers.energy_solvers.general_algebra_solvers.primme import PrimmeSolver
from sci_solvers.energy_solvers.general_algebra_solvers.pyscf_davidson import PyscfDavidsonSolver
from sci_solvers.energy_solvers.general_algebra_solvers.slepc import SlepcSolver

__all__ = [
    "ArpackSolver",
    "MultiRootDavidsonSolver",
    "LobpcgSolver",
    "PrimmeSolver",
    "SlepcSolver",
    "EvslSolver",
    "PyscfDavidsonSolver",
]
