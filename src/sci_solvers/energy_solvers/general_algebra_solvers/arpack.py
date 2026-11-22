"""ARPACK (SciPy ``eigsh``) selected-CI solver."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import warnings
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from scipy.linalg.lapack import HAS_ILP64

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase, dense_fallback
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin


def _check_int32_workspace(n: int, nroots: int, ncv: int | None) -> None:
    """Raise before ``eigsh`` would overflow ARPACK's 32-bit workspace offsets.

    Notes
    -----
    Unless SciPy is built with ILP64 LAPACK, ARPACK indexes its arrays with 32-bit
    integers. The ``n x ncv`` Lanczos block is addressed through offsets up to
    ``(ncv - 1) * n`` and the work array ``workd`` through offsets up to ``3 * n``, so
    ``eigsh`` segfaults (or returns a mis-sized slice) once

    .. math::

        \\max(n_{cv} - 1, 3)\\, n > 2^{31} - 1,

    with ``ncv`` defaulting, as in ``eigsh``, to ``min(max(2 k + 1, 20), n)``.
    """
    if HAS_ILP64:
        return
    ncv_eff = min(ncv if ncv is not None else max(2 * nroots + 1, 20), n)
    if max(ncv_eff - 1, 3) * n > 2**31 - 1:
        raise ValueError(
            f"ARPACK (scipy eigsh) cannot handle dimension n={n} with ncv={ncv_eff}: "
            f"max(ncv - 1, 3) * n = {max(ncv_eff - 1, 3) * n} exceeds its 32-bit index "
            f"limit 2**31 - 1 and would crash. Use a smaller ncv or another solver "
            f"(e.g. lobpcg or davidson)."
        )


class ArpackSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """SciPy ARPACK (``eigsh``) solver over the ``EigenProblem`` reduction."""

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)
    complex_capable = True  # eigsh solves complex Hermitian ops via the dtype-carrying operator
    supports_spinless = (
        True  # inherited base dispatch (_is_spinless_request) on a complex-capable solver
    )

    def __init__(
        self,
        *,
        use_operator: bool = True,
        ncv: int | None = None,
        max_restarts: int | None = None,
        **kwargs: Any,
    ):
        """``use_operator``: drive the matrix-free ``LinearOperator`` (else the sparse matrix).

        ``max_restarts``: number of Arnoldi restart cycles, forwarded to
        ``eigsh``'s ``maxiter`` directly (no ``n`` scaling). ``None`` (default)
        preserves the prior behavior — ``self.maxiter * n`` in
        :meth:`_diagonalize`, matching SciPy's own size-scaled default
        (``n * 10``) in spirit — so existing callers see no change. Set this
        explicitly for a flat, size-independent iteration budget, e.g. to
        match a Davidson-family solver's ``num_block * max_iteration`` total
        matvec budget: each restart cycle costs ``ncv - nroots`` new matvecs
        (the implicit-restart shift/purge step itself is a cheap dense
        operation on the ``ncv``-sized projected matrix, not a matvec), so
        ``max_restarts ~= total_matvec_budget / (ncv - nroots)``.
        """
        super().__init__(**kwargs)
        self.use_operator = use_operator
        self.ncv = ncv
        self.max_restarts = max_restarts
        self._check_fulqrum_mode()

    def _needs_assembled_matrix(self) -> bool:
        """``use_operator=False`` drives ARPACK with ``problem.sparse()``."""
        return not self.use_operator

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via ARPACK, dense fallback when ``k`` ~ ``n``."""
        n = problem.dim
        if nroots >= n - 1 or n < 50:  # ARPACK needs k << n
            return dense_fallback(problem, nroots)
        operator = problem.linear_operator() if self.use_operator else problem.sparse()
        _raw = self.maxiter * n if self.max_restarts is None else self.max_restarts
        # ARPACK's Fortran backend stores maxiter in a 32-bit integer; clamp to
        # avoid silent overflow when n is large (self.maxiter * n > 2^31 - 1).
        maxiter = min(_raw, 2**31 - 1)
        _check_int32_workspace(n, nroots, self.ncv)
        try:
            w, v = spla.eigsh(
                operator, k=nroots, which="SA", tol=self.tol, maxiter=maxiter, ncv=self.ncv
            )
            converged = np.full(len(w), True)
        except spla.ArpackNoConvergence as exc:
            warnings.warn(
                f"ARPACK did not converge ({exc}); returning {len(exc.eigenvalues)} "
                f"partially-converged eigenpair(s) out of {nroots} requested.",
                stacklevel=2,
            )
            w, v = exc.eigenvalues, exc.eigenvectors
            # scipy's own docs describe these as "the currently converged
            # eigenvalues", so each individual one may be numerically fine --
            # but the requested solve as a whole did not succeed (fewer roots
            # came back than nroots asked for). Mark the whole batch
            # not-converged so this is visible in the benchmark output,
            # consistent with how e.g. TorchSCISolver treats hitting its
            # iteration cap as "not_converged" rather than judging each
            # value's own accuracy.
            converged = np.full(len(w), False)
        return EigenResult(w, v, converged).sorted()
