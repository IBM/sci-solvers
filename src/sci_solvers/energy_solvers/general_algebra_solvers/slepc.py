"""SLEPc/PETSc selected-CI solver (needs the ``slepc`` extra)."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_solvers.general_algebra_base import (
    GeneralAlgebraSolverBase,
    _is_complex_problem,
    dense_fallback,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin


class SlepcSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """SLEPc/PETSc solver, matrix-free (``storage="shell"``) or assembled (``storage="aij"``).

    Needs the ``slepc`` extra (``petsc4py`` / ``slepc4py``).

    ``storage`` is an **operator-storage** choice and is deliberately distinct from the
    family-wide ``mode`` (``"tpb"``/``"gdb"``, the CI-basis contract, owned by the base
    class): the two are orthogonal, so a GDB pair basis can be solved with either storage.
    """

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)
    complex_capable = True  # HEP type solves complex Hermitian — if PETSc has complex scalars
    supports_spinless = True  # inherited base dispatch (_is_spinless_request); same PETSc caveat

    def __init__(
        self,
        *,
        storage: str = "shell",
        eps_type: str | None = None,
        comm: Any = None,
        **kwargs: Any,
    ):
        """``storage``: ``"shell"`` (matrix-free) or ``"aij"`` (assembled CSR)."""
        if storage not in ("shell", "aij"):
            raise ValueError(f"storage must be 'shell' or 'aij', got {storage!r}")
        super().__init__(**kwargs)
        self.storage = storage
        self.eps_type = eps_type
        self.comm = comm
        self._check_fulqrum_mode()

    def _needs_assembled_matrix(self) -> bool:
        """``storage="aij"`` copies ``problem.sparse()`` into a PETSc AIJ matrix."""
        return self.storage == "aij"

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via SLEPc, dense fallback for small ``n``."""
        try:
            from petsc4py import PETSc
            from slepc4py import SLEPc
        except ImportError as exc:
            raise ImportError(
                "SlepcSolver requires slepc4py. Install the optional extra: "
                "`uv sync --extra slepc` (or --extra external-solvers)."
            ) from exc

        comm = self.comm or PETSc.COMM_SELF
        n = problem.dim
        if n < 50 or nroots >= n:
            return dense_fallback(problem, nroots)

        # Best effort for complex Hermitian: SLEPc's HEP problem type IS the
        # Hermitian eigenproblem, so it handles complex Hermitian operators — but
        # only when PETSc/SLEPc was built with complex scalars. PETSc has a single
        # global scalar type; if it is real, a complex operator cannot be solved
        # here, so fail with a clear, actionable error rather than silently
        # discarding the imaginary part.
        is_complex = _is_complex_problem(problem)
        petsc_complex = np.iscomplexobj(np.empty(0, dtype=PETSc.ScalarType))
        if is_complex and not petsc_complex:
            from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, is_complex_capable

            others = ", ".join(
                repr(n) for n in AVAILABLE_SOLVERS if is_complex_capable(n) and n != "slepc"
            )
            raise NotImplementedError(
                "SlepcSolver received a complex Hermitian operator but PETSc/SLEPc "
                "was built with real scalars (PETSc.ScalarType is real). Rebuild "
                f"PETSc/SLEPc with complex scalar support, or use one of: {others}."
            )
        result_dtype = PETSc.ScalarType if is_complex else np.float64

        if self.storage == "aij":
            h_csr = problem.sparse()
            mat = PETSc.Mat().createAIJ(
                size=(n, n),
                csr=(
                    h_csr.indptr.astype(PETSc.IntType),
                    h_csr.indices.astype(PETSc.IntType),
                    h_csr.data,
                ),
                comm=comm,
            )
        else:  # shell / matrix-free
            prob = problem

            class _Shell:
                def mult(self, mat, x, y):  # noqa: ARG002
                    y.setArray(prob.matvec(x.getArray(readonly=True).copy()))

                def multTranspose(self, mat, x, y):  # A^T = conj(A) for Hermitian A  # noqa: ARG002
                    # SLEPc only asks for the (plain) transpose; for a Hermitian
                    # operator A^T == conj(A^H) == conj(A), so conjugate the input,
                    # apply A, and conjugate the result. For a real operator both
                    # conjugations are no-ops and this reduces to the symmetric case.
                    xin = np.conj(x.getArray(readonly=True).copy())
                    y.setArray(np.conj(prob.matvec(xin)))

            mat = PETSc.Mat().createPython((n, n), _Shell(), comm=comm)
        mat.setUp()

        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(mat)
        eps.setProblemType(SLEPc.EPS.ProblemType.HEP)
        if self.eps_type:
            eps.setType(self.eps_type)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        eps.setDimensions(nev=nroots)
        eps.setTolerances(tol=self.tol, max_it=self.maxiter)
        eps.setFromOptions()
        eps.solve()

        nconv = eps.getConverged()
        if nconv < nroots:
            raise RuntimeError(
                f"SLEPc converged {nconv}/{nroots} eigenpairs; "
                "increase maxiter or subspace (-eps_ncv)."
            )
        xr, _ = mat.createVecs()  # type: ignore[misc]  # returns (right, left) at runtime
        # Eigenvalues of a Hermitian operator are real, so energies stay real; the
        # eigenvectors carry the operator dtype (complex when the operator is).
        energies = np.empty(nroots)
        vectors = np.empty((n, nroots), dtype=result_dtype)
        for j in range(nroots):
            energies[j] = np.real(eps.getEigenpair(j, xr))
            arr = xr.getArray()
            vectors[:, j] = arr if is_complex else np.real(arr)
        eps.destroy()
        mat.destroy()
        return EigenResult(energies, vectors).sorted()
