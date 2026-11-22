"""In-house Davidson selected-CI solver (matrix-free, real symmetric or complex Hermitian).

Wraps the conjugate-aware :func:`~sci_solvers.energy_solvers.davidson_solver.davidson1`
core, which forms the effective subspace matrix with conjugate inner products
(``_fill_heff_hermitian``) and infers its dtype from the trial vectors — so the
same iteration serves the real and complex Hermitian cases. Dependency-free
(no scipy/PETSc/primme) and the reference complex-capable general-algebra solver.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_calculation_base import davidson_block_size_to_max_space
from sci_solvers.energy_solvers.general_algebra_base import (
    GeneralAlgebraSolverBase,
    _numpy_dtype,
    dense_fallback,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin


class MultiRootDavidsonSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """In-house matrix-free multi-root Davidson over the ``EigenProblem`` / ``HermitianOperator`` reduction.

    Uses :func:`sci_solvers.energy_solvers.davidson_solver.davidson1`, which is
    dtype-generic and Hermitian-correct (conjugate inner products throughout), so
    both real symmetric and complex Hermitian operators are handled by the same
    iteration, for any number of roots. Falls back to an exact dense ``eigh`` for
    small ``n``.

    Distinct from :class:`~sci_solvers.energy_solvers.davidson_solver.DavidsonSolver`,
    the native single-root fixed-space Davidson (real only, exposes ``shift_mode``);
    this one is matrix-free over the reduction, multi-root, and complex-capable.
    """

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)
    complex_capable = True  # davidson1 is conjugate-aware and dtype-generic
    supports_spinless = (
        True  # inherited base dispatch (_is_spinless_request) on a complex-capable solver
    )

    # Fallback used ONLY when ``davidson_block_size`` is None. It is handed to
    # davidson1 as ``max_space`` verbatim, with no widening correction, so davidson1
    # widens it by ``(nroots-1)*4`` itself and the subspace actually used grows with
    # the root count (30, 38, 58, ... for nroots = 1, 3, 8). When
    # ``davidson_block_size`` IS given, this constant is unused: the widening is
    # subtracted instead, so the subspace equals the requested size at every
    # ``nroots``. The two therefore differ for nroots > 1 — leaving the default is
    # not the same as passing this number explicitly.
    #
    # Applied in ``_diagonalize`` rather than ``__init__`` because the widening
    # correction needs ``nroots``, which is a solve-time argument here (the
    # spin-filter mixin may raise it per attempt while oversampling).
    _DEFAULT_MAX_SPACE = 30

    def __init__(
        self, *, davidson_block_size: int | None = None, max_cycle: int | None = None, **kwargs: Any
    ):
        """Initialize the subspace size and iteration budget.

        Args:
            davidson_block_size: Davidson subspace size — vectors accumulated
                before a restart, carrying the same meaning as on
                ``TorchSCISolver`` / ``SBDSolver`` / ``DavidsonSolver``.
                ``davidson1`` widens its ``max_space`` by ``(nroots-1)*4``, so
                that widening is subtracted in :meth:`_diagonalize` (where
                ``nroots`` is known) and the subspace used equals what you
                request, at any root count. ``None`` (default) instead yields
                ``_DEFAULT_MAX_SPACE + (nroots-1)*4`` — a subspace that scales
                with the root count rather than a fixed size, since that
                constant is passed through uncorrected.
            max_cycle: Flat Davidson iteration budget — total matvec cycles
                across all restarts, handed to ``davidson1`` as its ``max_cycle``.
                This is the only accepted spelling on this solver: the base class
                calls the same quantity ``maxiter``, and passing that here raises
                rather than silently offering two names for one knob. ``maxiter``
                remains correct on the sibling backends, where it does *not* denote
                a flat matvec budget (ARPACK and PRIMME scale it by ``n``, LOBPCG
                does not). ``None`` (default) uses the base's own default value.

        Raises:
            TypeError: If ``maxiter`` is passed — pass ``max_cycle`` instead.
        """
        # Checked before super().__init__ consumes kwargs: the base always sets
        # self.maxiter from its own default, so afterwards there is nothing left to
        # tell an explicitly-passed maxiter from that default.
        if "maxiter" in kwargs:
            raise TypeError(
                f"maxiter is not accepted by {type(self).__name__}; pass "
                f"max_cycle={kwargs['maxiter']} instead. It names the same quantity — "
                "the flat total-matvec budget this solver hands davidson1 — so only the "
                "standardized name is exposed here. Note `maxiter` stays valid on the "
                "other general-algebra backends, where it does not mean a flat matvec "
                "budget (ARPACK and PRIMME scale it by the problem dimension)."
            )
        super().__init__(**kwargs)
        self.davidson_block_size = davidson_block_size
        self.max_cycle = max_cycle if max_cycle is not None else self.maxiter

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via the in-house Davidson, dense fallback for small ``n``."""
        # Import lazily: davidson_solver pulls in pyscf, and this solver is meant to
        # be usable for the general (non-chemistry) complex path too.
        from sci_solvers.energy_solvers.davidson_solver import davidson1, make_diag_precond

        n = problem.dim
        if n < max(2 * nroots + 1, 4):
            return dense_fallback(problem, nroots)

        # Seed one unit vector per lowest-diagonal index in the operator dtype, so the
        # whole Krylov space (and davidson1's inferred dtype) is complex for a complex
        # operator and real otherwise; hdiag is real for a Hermitian operator.
        dtype = _numpy_dtype(problem)
        hd = np.asarray(problem.hdiag())
        x0 = []
        for idx in np.argsort(hd)[:nroots]:
            e = np.zeros(n, dtype=dtype)
            e[idx] = 1.0
            x0.append(e)

        # matvec may return a torch tensor for some problems; coerce to ndarray and
        # enforce the trial dtype so davidson1's dtype inference is stable.
        def aop(x: np.ndarray) -> np.ndarray:
            y = problem.matvec(np.asarray(x, dtype=dtype))
            return np.asarray(y, dtype=dtype).reshape(-1)

        precond = make_diag_precond(hd.astype(np.result_type(dtype, np.float64)))
        max_space = davidson_block_size_to_max_space(
            self.davidson_block_size, nroots, default_max_space=self._DEFAULT_MAX_SPACE
        )
        _conv, e, c = davidson1(
            lambda xs: [aop(x) for x in xs],
            x0,
            precond,
            tol=self.tol,
            max_cycle=self.max_cycle,
            max_space=max_space,
            nroots=nroots,
        )
        energies = np.asarray(e)
        vectors = np.column_stack([np.asarray(ci, dtype=dtype).reshape(-1) for ci in c])
        return EigenResult(energies, vectors).sorted()
