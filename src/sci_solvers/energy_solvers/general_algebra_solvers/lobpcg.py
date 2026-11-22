"""LOBPCG (SciPy) selected-CI solver with a Jacobi preconditioner."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np
import scipy.sparse.linalg as spla

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_solvers.general_algebra_base import (
    GeneralAlgebraSolverBase,
    _numpy_dtype,
    dense_fallback,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin


class LobpcgSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """SciPy LOBPCG solver with a Jacobi (diagonal) preconditioner."""

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)
    complex_capable = True  # complex-safe trial block + preconditioner (dtype follows the operator)
    supports_spinless = (
        True  # inherited base dispatch (_is_spinless_request) on a complex-capable solver
    )

    def __init__(self, *, seed: int = 0, **kwargs: Any):
        """``seed``: RNG seed for the randomized initial guess."""
        super().__init__(**kwargs)
        self.seed = seed

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via LOBPCG, dense fallback for small ``n``."""
        n = problem.dim
        if n < max(200, 6 * nroots):
            return dense_fallback(problem, nroots)
        rng = np.random.default_rng(self.seed)
        # The trial block and preconditioner follow the operator dtype so LOBPCG's
        # complex code path engages for a complex Hermitian operator; hdiag is real.
        dtype = _numpy_dtype(problem)
        hd = problem.hdiag()
        guess = np.zeros((n, nroots), dtype=dtype)
        for j, idx in enumerate(np.argsort(hd)[:nroots]):
            guess[idx, j] = 1.0
        noise = np.asarray(rng.standard_normal(guess.shape))
        if np.iscomplexobj(guess):
            noise = noise + 1j * rng.standard_normal(guess.shape)
        guess += 1e-3 * noise
        shift = hd.min() - 1.0
        dinv = 1.0 / (hd - shift)

        def _prec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return (x.reshape(n, -1) * dinv[:, None]).reshape(x.shape)

        precond = spla.LinearOperator((n, n), matvec=_prec, matmat=_prec, dtype=dtype)
        w, v = spla.lobpcg(
            problem.linear_operator(),
            guess,
            M=precond,
            tol=self.tol,
            maxiter=self.maxiter,
            largest=False,
        )
        w = np.asarray(w)
        v = np.asarray(v)
        # scipy's lobpcg never raises and never reports whether `tol` was
        # actually reached (only a UserWarning printed to stderr on failure,
        # not a programmatically-readable signal) -- and per its own docs it
        # returns "the block of iterative vectors with the best accuracy
        # rather than the last one iterated", so even
        # `retResidualNormsHistory`'s last entry isn't guaranteed to match
        # the returned (w, v). Verify directly against what was actually
        # returned: recompute the residual ||A v - w v|| ourselves via the
        # same operator lobpcg was given.
        residual_norms = np.linalg.norm(problem.matvec(v) - v * w[None, :], axis=0)
        converged = residual_norms <= self.tol
        return EigenResult(w, v, converged).sorted()
