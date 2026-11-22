"""PySCF fixed-space Davidson selected-CI solver (native spin projection)."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_calculation_base import davidson_block_size_to_max_space
from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase


class PyscfDavidsonSolver(GeneralAlgebraSolverBase):
    """PySCF fixed-space Davidson solver — reads raw integrals, projects spin natively.

    Does NOT mix in :class:`~sci_solvers.energy_solvers.spin_filter_mixin.SpinFilterMixin`:
    it targets a spin sector directly via PySCF's ``fix_spin_`` penalty
    (``spin_sq``). It bypasses the matrix-free operator (and thus the packed sector
    view), always solving the full space.

    It cannot represent a non-product determinant basis — ``kernel_fixed_space``
    builds a PySCF ``SCIvector`` over the Cartesian product of ``strs_a x strs_b`` —
    so ``mode="gdb"`` is rejected (``supports_gdb = False``). For the same reason
    ``hamiltonian_backend="fulqrum"`` is rejected (``supports_fulqrum_backend = False``):
    it would build an operator this solver never reads.
    """

    supports_gdb: bool = False
    supports_fulqrum_backend: bool = False

    def __init__(
        self,
        *,
        spin_sq: float | None = None,
        davidson_block_size: int | None = None,
        **kwargs: Any,
    ):
        """``spin_sq``: target total spin squared, projected via PySCF ``fix_spin_``.

        ``davidson_block_size``: Davidson subspace size, the number of vectors kept
        before a restart -- same meaning as on ``PySCFSolver``, converted to PySCF's
        ``max_space`` by :func:`davidson_block_size_to_max_space`. ``None`` (default)
        leaves ``max_space`` unset, i.e. PySCF's own default, as before this
        parameter existed.
        """
        super().__init__(**kwargs)
        self.spin_sq = spin_sq
        self.davidson_block_size = davidson_block_size

    def _solve_batch(self, problem: EigenProblem, k: int) -> EigenResult:
        """Always solve the full space; PySCF Davidson cannot use the packed view."""
        return self._diagonalize(problem, k).sorted()

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via PySCF ``kernel_fixed_space``.

        ``kernel_fixed_space`` calls ``myci.eig`` internally, which sets
        ``myci.converged`` as a side effect (PySCF's own Davidson convergence
        flag -- a scalar bool for ``nroots == 1``, else a per-root
        array/list) but never returns it. We read it back here so
        :class:`~sci_solvers.energy_solvers.general_algebra_base.GeneralAlgebraSolverBase`
        can report real convergence instead of the "unknown" default.

        PySCF's ``selected_ci`` needs strictly ascending (sorted, unique) strings;
        the base class rejects any other TPB batch before this is called.
        """
        from pyscf.fci import addons, selected_ci

        myci = selected_ci.SelectedCI()
        if self.spin_sq is not None:
            myci = addons.fix_spin_(myci, ss=self.spin_sq)
        extra: dict[str, Any] = (
            {}
            if self.davidson_block_size is None
            else {"max_space": davidson_block_size_to_max_space(self.davidson_block_size, nroots)}
        )
        e, c = selected_ci.kernel_fixed_space(
            myci,
            problem.hcore,
            problem.eri,
            problem.norb,
            problem.nelec,
            ci_strs=(problem.strs_a, problem.strs_b),
            nroots=nroots,
            tol=self.tol,
            max_cycle=self.maxiter,
            **extra,
        )
        energies = np.atleast_1d(np.asarray(e, dtype=np.float64))
        civecs = c if isinstance(c, (list, tuple)) else [c]
        vectors = np.column_stack([np.asarray(ci, dtype=np.float64).ravel() for ci in civecs])
        converged_raw = getattr(myci, "converged", None)
        converged = (
            None
            if converged_raw is None
            else np.atleast_1d(np.asarray(converged_raw, dtype=bool))[:nroots]
        )
        return EigenResult(energies[:nroots], vectors[:, :nroots], converged).sorted()
