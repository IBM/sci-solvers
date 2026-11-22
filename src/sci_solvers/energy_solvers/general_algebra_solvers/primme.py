"""PRIMME selected-CI solver (needs the ``primme`` extra)."""

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


class PrimmeSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """PRIMME (``eigsh``) solver with a Jacobi preconditioner (needs the ``primme`` extra)."""

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)
    complex_capable = True  # primme.eigsh handles complex Hermitian ops natively
    supports_spinless = (
        True  # inherited base dispatch (_is_spinless_request) on a complex-capable solver
    )

    def __init__(
        self, *, max_restarts: int | None = None, a_norm: float | None = 1.0, **kwargs: Any
    ):
        """``max_restarts``: forwarded to ``primme.eigsh``'s ``maxiter`` directly (no ``n`` scaling).

        ``None`` (default) preserves the prior behavior -- ``self.maxiter * n`` in
        :meth:`_diagonalize` -- so existing callers see no change. Set this explicitly for a
        flat, size-independent iteration budget, mirroring :class:`ArpackSolver`'s identical
        ``max_restarts`` knob (added there for the same ``maxiter * n`` scaling issue).

        ``a_norm``: forwarded to ``primme.eigsh`` as ``aNorm``. PRIMME's default stopping test is
        ``||A v - lambda v|| < tol * ||A||`` with ``||A||`` an internal running estimate of the
        spectral norm, so ``tol`` is relative and its absolute meaning varies per molecule. When
        ``a_norm`` is set, PRIMME uses it verbatim as ``||A||`` in that test, so the default
        ``a_norm=1.0`` makes ``tol`` an absolute residual bound, the same quantity the
        Davidson-family solvers use. Pass ``None`` to keep PRIMME's own ``||A||`` estimate (the
        relative behavior this solver had before ``a_norm`` existed).
        """
        super().__init__(**kwargs)
        self.max_restarts = max_restarts
        self.a_norm = a_norm

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via PRIMME, dense fallback for small ``n``."""
        try:
            import primme
        except ImportError as exc:
            raise ImportError(
                "PrimmeSolver requires primme. Install the optional extra: "
                "`uv sync --extra primme` (or --extra external-solvers)."
            ) from exc

        n = problem.dim
        if n < 50:
            return dense_fallback(problem, nroots)
        # primme.eigsh handles complex Hermitian operators natively, so we pass the
        # complex operator through rather than rejecting it. Only the Jacobi
        # preconditioner dtype follows the operator; hdiag is real for a Hermitian op.
        dtype = _numpy_dtype(problem)
        hd = problem.hdiag()
        shift = hd.min() - 1.0
        dinv = 1.0 / (hd - shift)

        def _prec(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return (x.reshape(n, -1) * dinv[:, None]).reshape(x.shape)

        precond = spla.LinearOperator((n, n), matvec=_prec, matmat=_prec, dtype=dtype)
        _raw = self.maxiter * n if self.max_restarts is None else self.max_restarts
        # primme's C backend stores maxiter in a 32-bit integer; clamp to avoid silent
        # overflow when n is large (self.maxiter * n > 2^31 - 1), same risk ARPACK had.
        maxiter = min(_raw, 2**31 - 1)
        # raise_for_unconverged=True (primme's own default) makes primme.eigsh
        # raise PrimmeError rather than returning a partial result -- unlike
        # arpack's eigsh, which started returning partial results on failure
        # after a scipy update. Pinned explicitly so a future primme default
        # change can't silently reintroduce that same gap here.
        extra: dict[str, Any] = {} if self.a_norm is None else {"aNorm": self.a_norm}
        w, v = primme.eigsh(
            problem.linear_operator(),
            k=nroots,
            which="SA",
            tol=self.tol,
            maxiter=maxiter,
            OPinv=precond,
            raise_for_unconverged=True,
            return_unconverged=False,
            **extra,
        )
        return EigenResult(w, v).sorted()
