"""Tests for the complex Hermitian eigenvalue path.

These exercise the *new* complex Hermitian support without touching the real
symmetric behavior:

* :class:`~sci_solvers.eigen_problem.HermitianOperator` as a general operator
  (dense / sparse / matrix-free), including its Hermiticity guard.
* The complex-capable adapters (``dense`` fallback, ``arpack``, ``lobpcg``,
  ``davidson``) agreeing with ``numpy.linalg.eigvalsh`` on a complex Hermitian
  matrix, and reproducing the real path on a real symmetric matrix.
* The real-only general-algebra adapters (``pyscf_davidson``, ``evsl``,
  ``dice``, ``truncated_davidson``) failing loudly on a complex operator.
* ``torch_sci``'s own complex contract (its own ``dtype=`` opt-in, not a
  general-algebra solver so it is tested via its native SCI-solver entry
  point rather than through ``_ga_solve``/``HermitianOperator``).
* The complex chemistry path (:func:`build_complex_ci_problem`) matching a dense
  reference, plus the real-input rejection.
* :class:`~sci_solvers.hamiltonian.HamiltonianData` complex handling and the
  complex ``to_fcidump`` rejection.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest
import scipy.sparse as sps

from sci_solvers.complex_ci_problem import (
    build_complex_ci_problem,
    build_complex_ci_problem_from_hamiltonian,
    build_selected_complex_ci_problem,
)
from sci_solvers.eigen_problem import EigenResult, HermitianOperator
from sci_solvers.energy_solvers.factory import (
    AVAILABLE_SOLVERS,
    create_solver,
    is_complex_capable,
    is_general_algebra,
)
from sci_solvers.energy_solvers.general_algebra_base import dense_fallback
from sci_solvers.hamiltonian import HamiltonianData


# The complex-capable general-algebra solvers replace the former adapters. Their
# per-problem hook is ``_diagonalize`` (no spin symmetrization / RDM packing), which
# is the right entry point for diagonalizing a bare ``HermitianOperator``.
def _petsc_is_complex() -> bool:
    """Whether the installed PETSc was built with complex scalars.

    PETSc has a single global scalar type fixed at build time; a real-scalar
    build cannot diagonalize a complex operator regardless of any Python-side
    configuration. This is a property of the compiled ``libpetsc`` binary, not
    something the project's dependency set can change.
    """
    from petsc4py import PETSc

    return np.iscomplexobj(np.empty(0, dtype=PETSc.ScalarType))


def _ga_solve(prob, k, *, name="davidson", tol=1e-10, maxiter=500):
    """Lowest ``k`` eigenpairs of ``prob`` via a general-algebra solver's ``_diagonalize``."""
    if name == "dense":
        return dense_fallback(prob, k)
    if name == "primme":
        pytest.importorskip("primme")
    if name == "slepc":
        pytest.importorskip("slepc4py")
        # slepc4py present but the operator is complex and PETSc is real-scalar:
        # the complex path is physically unsupported by this PETSc build, so skip
        # (mirrors the missing-backend skip above) rather than failing.
        if np.issubdtype(prob.dtype, np.complexfloating) and not _petsc_is_complex():
            pytest.skip("PETSc built with real scalars; complex path unsupported")
    # `davidson` exposes the flat iteration budget as `max_cycle` and rejects the
    # base's `maxiter` spelling; the other backends read `maxiter` with their own
    # per-solver meaning, so the knob name is chosen per solver here.
    budget = {"max_cycle": maxiter} if name == "davidson" else {"maxiter": maxiter}
    solver = create_solver(name, symmetrize_spin=False, tol=tol, **budget)
    return solver._diagonalize(prob, k)


# Derived from the capability flags, never hand-listed: the exact "dense" fallback
# plus every complex-capable *general-algebra* solver (primme/slepc skip in
# _ga_solve when their optional backend is missing). This is exactly what would
# drift by hand — e.g. it now includes primme/slepc, which the old literal
# omitted. Restricted to is_general_algebra(n): _ga_solve drives a solver via
# its GeneralAlgebraSolverBase._diagonalize(prob, k) on a bare HermitianOperator,
# which native SCI solvers like torch_sci (complex_capable but not a
# general-algebra solver — no _diagonalize at all) cannot answer; torch_sci's
# own complex contract is exercised separately below via its SCI-solver
# entry point (sci_solver_callable_eigen_states), not through this harness.
COMPLEX_SOLVERS = [
    "dense",
    *(n for n in AVAILABLE_SOLVERS if is_general_algebra(n) and is_complex_capable(n)),
]
# The complement: general-algebra solvers that are NOT complex-capable must reject
# complex integrals.
REAL_ONLY_GENERAL_ALGEBRA_SOLVERS = [
    n for n in AVAILABLE_SOLVERS if is_general_algebra(n) and not is_complex_capable(n)
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_hermitian(n: int, seed: int = 0) -> np.ndarray:
    """A random complex Hermitian ``(n, n)`` matrix (real eigenvalues)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return a + a.conj().T


def _random_symmetric(n: int, seed: int = 0) -> np.ndarray:
    """A random real symmetric ``(n, n)`` matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a + a.T


def _real_ci_integrals(norb: int, seed: int = 0):
    """Real symmetric ``hcore`` and a real ERI with full permutational symmetry.

    Built by density fitting from a symmetric factor ``L[P] == L[P].T`` so the
    ERI ``(pq|rs) = sum_P L[P,p,q] L[P,r,s]`` has both ``(pq|rs) == (qp|sr)`` and
    ``(pq|rs) == (rs|pq)`` — the real limit of the complex-Hermitian symmetries.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    hcore = h + h.T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb))
    for _ in range(naux):
        g = rng.standard_normal((norb, norb))
        L = g + g.T
        eri += np.einsum("pq,rs->pqrs", L, L)
    return hcore, eri


def _random_complex_hermitian_integrals(norb: int, seed: int = 0):
    """Complex ``hcore`` and a physically valid complex chemist-order ERI.

    The ERI is built as ``(pq|rs) = sum_P L[P,p,q] * conj(L[P,r,s])`` from a
    random complex Cholesky-like factor ``L``. This guarantees *both* Hermitian
    symmetries the CI operator needs — ``(pq|rs) == conj((qp|sr))`` and
    ``(pq|rs) == (rs|pq)`` — because ``L[P,p,q]`` is itself built Hermitian in
    ``(p, q)`` (``L[P] == L[P]^H``), so ``build_complex_ci_problem`` accepts it.
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T  # Hermitian factor: L == L^H
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    # The density-fit form above gives sym1 ((pq|rs) == conj((qp|sr))) but not
    # the particle-exchange sym2 ((pq|rs) == (rs|pq)); symmetrizing over the
    # (pq)<->(rs) swap enforces sym2 while preserving sym1, so the CI operator
    # is Hermitian — exactly the pair of symmetries a physical complex ERI has.
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


# ---------------------------------------------------------------------------
# HermitianOperator — construction and guards
# ---------------------------------------------------------------------------
def test_hermitian_problem_dense_complex_dtype_and_interface():
    """A complex matrix yields a complex128 problem with a true adjoint."""
    A = _random_hermitian(6, seed=1)
    prob = HermitianOperator(A)
    assert prob.dtype == np.complex128
    assert prob.dim == 6
    x = np.random.default_rng(0).standard_normal(6) + 1j * np.random.default_rng(1).standard_normal(
        6
    )
    np.testing.assert_allclose(prob.matvec(x), A @ x)
    # rmatvec is the conjugate transpose (== matvec for Hermitian A).
    np.testing.assert_allclose(prob.rmatvec(x), A.conj().T @ x)
    # hdiag is real.
    assert np.isrealobj(prob.hdiag())
    np.testing.assert_allclose(prob.hdiag(), np.real(np.diagonal(A)))


def test_hermitian_problem_real_stays_real():
    """A real matrix yields a float64 problem — the real path is unchanged."""
    A = _random_symmetric(6, seed=2)
    prob = HermitianOperator(A)
    assert prob.dtype == np.float64
    assert prob.linear_operator().dtype == np.float64


def test_hermitian_problem_rejects_non_hermitian():
    """A non-Hermitian matrix is rejected with a clear ValueError."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))  # not Hermitian
    with pytest.raises(ValueError, match="not Hermitian"):
        HermitianOperator(A)
    # ...but the check can be bypassed.
    HermitianOperator(A, check_hermitian=False)


def test_hermitian_problem_rejects_non_square():
    """A non-square matrix is rejected."""
    with pytest.raises(ValueError, match="square"):
        HermitianOperator(np.zeros((3, 4), dtype=np.complex128))


def test_hermitian_problem_sparse_complex():
    """A complex sparse Hermitian matrix is accepted and preserved."""
    A = _random_hermitian(8, seed=4)
    prob = HermitianOperator(sps.csr_matrix(A))
    assert prob.dtype == np.complex128
    np.testing.assert_allclose(prob.sparse().toarray(), A, atol=1e-12)


def test_hermitian_problem_sparse_rejects_non_hermitian():
    """A non-Hermitian sparse matrix is rejected."""
    rng = np.random.default_rng(5)
    A = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    with pytest.raises(ValueError, match="not Hermitian"):
        HermitianOperator(sps.csr_matrix(A))


def test_hermitian_problem_from_matvec():
    """The matrix-free constructor applies the operator and reports its dtype."""
    A = _random_hermitian(7, seed=6)
    prob = HermitianOperator.from_matvec(lambda x: A @ x, np.diagonal(A), 7, dtype=np.complex128)
    assert prob.dtype == np.complex128
    x = np.random.default_rng(7).standard_normal(7).astype(np.complex128)
    np.testing.assert_allclose(prob.matvec(x), A @ x)
    # sparse() materializes A from the matvec.
    np.testing.assert_allclose(prob.sparse().toarray(), A, atol=1e-12)


# ---------------------------------------------------------------------------
# EigenResult.sorted — complex energies sort on the real part
# ---------------------------------------------------------------------------
def test_eigenresult_sorted_on_real_part():
    """Sorting orders by the real part and keeps columns aligned."""
    energies = np.array([2.0 + 1e-12j, -1.0 - 1e-12j, 0.5])
    vectors = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex128)
    out = EigenResult(energies, vectors).sorted()
    np.testing.assert_allclose(np.real(out.energies), [-1.0, 0.5, 2.0])
    np.testing.assert_array_equal(np.real(out.vectors[0]), [2, 3, 1])


# ---------------------------------------------------------------------------
# complex-capable adapters agree with eigvalsh
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", COMPLEX_SOLVERS)
def test_complex_adapter_matches_eigvalsh(name):
    """Each complex-capable solver recovers the lowest eigenvalues of a Hermitian matrix."""
    n, nroots = 200, 3
    A = _random_hermitian(n, seed=42)
    ref = np.sort(np.linalg.eigvalsh(A))[:nroots]
    prob = HermitianOperator(A)
    res = _ga_solve(prob, nroots, name=name).sorted()
    np.testing.assert_allclose(np.real(res.energies[:nroots]), ref, atol=1e-6)
    # Returned eigenvectors are complex and satisfy A v = e v.
    for j in range(nroots):
        v = res.vectors[:, j]
        np.testing.assert_allclose(A @ v, res.energies[j] * v, atol=1e-5)


@pytest.mark.parametrize("name", COMPLEX_SOLVERS)
def test_real_symmetric_regression(name):
    """The same adapters reproduce eigvalsh on a real symmetric matrix (path unchanged)."""
    n, nroots = 200, 3
    A = _random_symmetric(n, seed=7)
    ref = np.sort(np.linalg.eigvalsh(A))[:nroots]
    prob = HermitianOperator(A)
    assert prob.dtype == np.float64
    res = _ga_solve(prob, nroots, name=name).sorted()
    np.testing.assert_allclose(np.real(res.energies[:nroots]), ref, atol=1e-6)


# ---------------------------------------------------------------------------
# real and complex paths agree on the same operator
# ---------------------------------------------------------------------------
# A real symmetric matrix is *also* Hermitian, so the same physical operator can
# be driven down the real path (float64) or the complex path (complex128 with a
# zero imaginary part). Every supporting adapter must return the same
# eigenvalues either way — this is the direct "real path == complex path" check
# and guards against the two dtype branches drifting apart.
@pytest.mark.parametrize("name", COMPLEX_SOLVERS)
def test_real_and_complex_paths_agree_explicit_matrix(name):
    """For a real symmetric matrix, the float64 and complex128 problems match."""
    n, nroots = 200, 4
    A = _random_symmetric(n, seed=101)

    prob_real = HermitianOperator(A)  # float64 branch
    prob_cplx = HermitianOperator(A.astype(np.complex128))  # complex128 branch
    assert prob_real.dtype == np.float64
    assert prob_cplx.dtype == np.complex128

    res_real = _ga_solve(prob_real, nroots, name=name).sorted()
    res_cplx = _ga_solve(prob_cplx, nroots, name=name).sorted()

    # Same eigenvalues from both dtype branches.
    np.testing.assert_allclose(
        np.real(res_cplx.energies[:nroots]),
        np.real(res_real.energies[:nroots]),
        atol=1e-6,
    )
    # And both match the dense reference.
    ref = np.sort(np.linalg.eigvalsh(A))[:nroots]
    np.testing.assert_allclose(np.real(res_real.energies[:nroots]), ref, atol=1e-6)
    # The complex-branch eigenvectors carry the complex dtype but a negligible
    # imaginary part (the operator is really real).
    assert np.iscomplexobj(res_cplx.vectors)


@pytest.mark.parametrize("name", COMPLEX_SOLVERS)
def test_real_and_complex_ci_paths_agree(name):
    """Real integrals via the dense CI operator match the same integrals cast complex.

    The real chemistry CI operator (built here with the real-capable
    ``direct_nosym`` contraction) and :func:`build_complex_ci_problem` on the
    *same integrals cast to complex* describe one physical Hamiltonian, so their
    lowest eigenvalues must coincide for every supporting adapter.
    """
    from pyscf.fci import cistring, direct_nosym

    norb, nelec, nroots = 4, (2, 2), 3
    hcore, eri = _real_ci_integrals(norb, seed=202)

    # Real reference: assemble the dense real CI Hamiltonian directly.
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dim = na * nb
    h2e = direct_nosym.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    H_real = np.empty((dim, dim))
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H_real[:, i] = direct_nosym.contract_2e(h2e, e.reshape(na, nb), norb, nelec).reshape(-1)
    ref = np.sort(np.linalg.eigvalsh(H_real))[:nroots]

    # Complex path: same integrals, cast to complex, through the complex builder.
    prob_cplx = build_complex_ci_problem(
        hcore.astype(np.complex128), eri.astype(np.complex128), norb, nelec
    )
    assert prob_cplx.dtype == np.complex128
    res = _ga_solve(prob_cplx, nroots, name=name).sorted()
    np.testing.assert_allclose(np.real(res.energies[:nroots]), ref, atol=1e-7)


# ---------------------------------------------------------------------------
# real-only adapters fail loudly on a complex operator
# ---------------------------------------------------------------------------
# The complex-capability gate lives on the general-algebra reduction: solvers whose
# kernels read the real integrals directly (pyscf_davidson) or slice a real spectrum
# (evsl) must reject complex integrals rather than drop the imaginary part.
# REAL_ONLY_GENERAL_ALGEBRA_SOLVERS is derived from the capability flags near the top.


@pytest.mark.parametrize("name", REAL_ONLY_GENERAL_ALGEBRA_SOLVERS)
def test_real_only_solver_rejects_complex(name):
    """Real-only general-algebra solvers raise a clear NotImplementedError on complex integrals."""
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=8)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    solver = create_solver(name, symmetrize_spin=False)
    with pytest.raises(NotImplementedError, match="complex"):
        solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)


# ---------------------------------------------------------------------------
# torch_sci: its own complex contract (SCI-native, not general-algebra)
# ---------------------------------------------------------------------------
# torch_sci is complex_capable but is not a general-algebra solver (no
# _diagonalize / HermitianOperator entry point — see the COMPLEX_SOLVERS
# comment above), so it is exercised directly via its own
# sci_solver_callable_eigen_states contract: dtype=torch.complex128 opt-in at
# construction, real-default construction rejecting complex integrals, and
# agreement with the real path on the same physical Hamiltonian.
def test_torch_sci_rejects_complex_without_complex_dtype():
    """Constructed with the real default dtype, torch_sci rejects complex integrals."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sci_solver")
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=9)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu")
    assert solver._solver.dtype == torch.float64
    with pytest.raises(NotImplementedError, match="complex"):
        solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)


def test_torch_sci_complex_dtype_matches_real_path():
    """torch_sci with dtype=torch.complex128 reproduces the real-path ground energy.

    Same physical Hamiltonian (real integrals, cast to complex128 with a zero
    imaginary part) solved once through the real float64 path and once through
    the complex128 opt-in path; both must agree with each other and with the
    PySCF dense reference.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sci_solver")
    from pyscf.fci import cistring, direct_nosym

    # seed=0 is a known-converging case for torch_sci's default Davidson knobs
    # at this tiny (norb=4, dim=36) basis — the point of this test is the
    # real-vs-complex dtype agreement, not torch_sci's small-basis Davidson
    # convergence behavior (a separate, pre-existing concern independent of
    # complex support).
    norb, nelec = 4, (2, 2)
    hcore, eri = _real_ci_integrals(norb, seed=0)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dim = na * nb
    h2e = direct_nosym.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    H_real = np.empty((dim, dim))
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H_real[:, i] = direct_nosym.contract_2e(h2e, e.reshape(na, nb), norb, nelec).reshape(-1)
    ref = np.sort(np.linalg.eigvalsh(H_real))[0]

    real_solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu")
    e_real = real_solver.sci_solver_callable([(strs_a, strs_b)], hcore, eri, norb, nelec)[0].energy

    cplx_solver = create_solver(
        "torch_sci", symmetrize_spin=False, device="cpu", dtype=torch.complex128
    )
    assert cplx_solver._solver.dtype == torch.complex128
    e_cplx = cplx_solver.sci_solver_callable(
        [(strs_a, strs_b)], hcore.astype(np.complex128), eri.astype(np.complex128), norb, nelec
    )[0].energy

    np.testing.assert_allclose(e_real, ref, atol=1e-7)
    np.testing.assert_allclose(np.real(e_cplx), ref, atol=1e-7)
    np.testing.assert_allclose(np.real(e_cplx), e_real, atol=1e-7)


# ---------------------------------------------------------------------------
# complex chemistry path
# ---------------------------------------------------------------------------
def test_build_complex_ci_problem_matches_dense():
    """The full-CAS complex CI matvec agrees with a dense eigvalsh reference."""
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=11)
    prob = build_complex_ci_problem(hcore, eri, norb, nelec)
    assert prob.dtype == np.complex128
    H = prob.sparse().toarray()
    # The assembled operator must itself be Hermitian.
    np.testing.assert_allclose(H, H.conj().T, atol=1e-9)
    ref = np.sort(np.linalg.eigvalsh(H))[:3]
    res = _ga_solve(prob, 3).sorted()
    np.testing.assert_allclose(np.real(res.energies[:3]), ref, atol=1e-7)


def test_build_complex_ci_problem_rejects_real():
    """A real ERI is rejected — that is the real EigenProblem path's job."""
    norb = 4
    hcore = _random_symmetric(norb, seed=12)
    eri = np.zeros((norb, norb, norb, norb))
    with pytest.raises(ValueError, match="complex"):
        build_complex_ci_problem(hcore, eri, norb, (2, 2))


def test_build_complex_ci_problem_rejects_non_hermitian_eri():
    """An ERI lacking the required Hermitian symmetry is rejected."""
    norb = 4
    rng = np.random.default_rng(13)
    hcore = _random_hermitian(norb, seed=13)
    eri = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    with pytest.raises(ValueError, match="Hermitian|conj|rs"):
        build_complex_ci_problem(hcore, eri, norb, (2, 2))


def test_build_complex_ci_problem_from_hamiltonian():
    """The HamiltonianData wrapper builds the same operator."""
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=14)
    ham = HamiltonianData(hcore, eri, 0.0, norb, nelec[0], nelec[1])
    prob = build_complex_ci_problem_from_hamiltonian(ham)
    ref = build_complex_ci_problem(hcore, eri, norb, nelec)
    np.testing.assert_allclose(prob.sparse().toarray(), ref.sparse().toarray(), atol=1e-10)


# ---------------------------------------------------------------------------
# HamiltonianData complex handling
# ---------------------------------------------------------------------------
def test_hamiltonian_data_complex_keeps_full_eri():
    """Complex integrals skip ao2mo.restore and keep the full 4-index ERI."""
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=15)
    ham = HamiltonianData(hcore, eri, 0.0, norb, nelec[0], nelec[1])
    assert ham.is_complex
    assert ham.eri.shape == (norb, norb, norb, norb)
    assert np.iscomplexobj(ham.eri)


def test_hamiltonian_data_complex_requires_full_eri():
    """A complex ERI that is not the full dense tensor is rejected."""
    norb = 4
    hcore = _random_hermitian(norb, seed=16)
    eri = np.zeros((norb, norb), dtype=np.complex128)  # wrong shape for complex
    with pytest.raises(ValueError):
        HamiltonianData(hcore, eri, 0.0, norb, 2, 2)


def test_hamiltonian_data_real_unchanged():
    """A real Hamiltonian is not flagged complex (real path preserved)."""
    norb = 4
    hcore = _random_symmetric(norb, seed=17)
    eri = np.zeros((norb, norb, norb, norb))
    ham = HamiltonianData(hcore, eri, 0.0, norb, 2, 2)
    assert not ham.is_complex


def test_to_fcidump_rejects_complex():
    """FCIDUMP export is real-only; a complex Hamiltonian is rejected."""
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=18)
    ham = HamiltonianData(hcore, eri, 0.0, norb, nelec[0], nelec[1])
    with pytest.raises(NotImplementedError):
        ham.to_fcidump("/tmp/should_not_be_written.fcidump")


# ---------------------------------------------------------------------------
# selected complex CI path (torch kernel)
# ---------------------------------------------------------------------------
def _dense_projected_ci(hcore, eri, norb, nelec, idx=None):
    """Dense complex CI Hamiltonian (optionally projected onto ``idx``).

    Assembles the full-CAS complex CI operator with the complex-capable
    ``direct_nosym`` contraction — an independent oracle from the torch kernel —
    and returns it, restricted to the flat product-basis indices ``idx`` when a
    selected subspace is requested.
    """
    from pyscf.fci import cistring, direct_nosym

    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dim = na * nb
    h2e = direct_nosym.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    H = np.empty((dim, dim), dtype=np.complex128)
    for c in range(dim):
        e = np.zeros(dim, dtype=np.complex128)
        e[c] = 1.0
        H[:, c] = direct_nosym.contract_2e(h2e, e.reshape(na, nb), norb, nelec).reshape(-1)
    if idx is not None:
        H = H[np.ix_(idx, idx)]
    return H


def test_selected_complex_ci_full_space_matches_oracle():
    """On the full product basis the torch kernel matches the dense oracle."""
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=21)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    prob = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)
    assert prob.dtype == np.complex128
    assert prob.dim == len(strs_a) * len(strs_b)
    H = prob.sparse().toarray()
    # The assembled selected-CI operator is itself Hermitian.
    np.testing.assert_allclose(H, H.conj().T, atol=1e-9)

    ref = np.sort(np.linalg.eigvalsh(_dense_projected_ci(hcore, eri, norb, nelec)))
    got = np.sort(np.linalg.eigvalsh(H))
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_selected_complex_ci_subspace_matches_projected_oracle():
    """On a genuinely selected subspace the kernel matches the projected oracle.

    A strict subset of the alpha/beta strings defines a product basis smaller than
    the full CAS; the torch kernel's operator on that basis must equal the dense
    CI Hamiltonian projected onto the same flat product-basis indices.
    """
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=22)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    nb = len(strs_b)

    # Pick a strict subset of each spin's strings.
    sel_a_pos, sel_b_pos = [0, 1, 3], [0, 2, 4, 5]
    sel_a, sel_b = strs_a[sel_a_pos], strs_b[sel_b_pos]
    idx = np.array([i * nb + j for i in sel_a_pos for j in sel_b_pos])

    prob = build_selected_complex_ci_problem(sel_a, sel_b, hcore, eri, norb, nelec)
    assert prob.dim == len(sel_a) * len(sel_b)
    H = prob.sparse().toarray()
    np.testing.assert_allclose(H, H.conj().T, atol=1e-9)

    ref = np.sort(np.linalg.eigvalsh(_dense_projected_ci(hcore, eri, norb, nelec, idx=idx)))
    got = np.sort(np.linalg.eigvalsh(H))
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_selected_complex_ci_diagonal_matches():
    """The problem's ``hdiag`` equals the real diagonal of the assembled operator."""
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=23)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    prob = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)
    H = prob.sparse().toarray()
    assert np.isrealobj(prob.hdiag())
    np.testing.assert_allclose(prob.hdiag(), np.real(np.diagonal(H)), atol=1e-9)


def test_selected_complex_ci_davidson_recovers_ground_state():
    """The complex-capable Davidson adapter recovers the lowest eigenvalues."""
    from pyscf.fci import cistring

    norb, nelec, nroots = 4, (2, 2), 3
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=24)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    prob = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)
    ref = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))[:nroots]
    res = _ga_solve(prob, nroots).sorted()
    np.testing.assert_allclose(np.real(res.energies[:nroots]), ref, atol=1e-7)


def test_selected_complex_ci_matches_full_cas_builder():
    """On the full space the torch and direct_nosym complex builders agree.

    ``build_selected_complex_ci_problem`` (torch kernel) and
    ``build_complex_ci_problem`` (pyscf ``direct_nosym``) are two independent
    complex-Hermitian implementations of the same operator; on the full product
    basis their spectra must coincide.
    """
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=25)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    prob_torch = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)
    prob_nosym = build_complex_ci_problem(hcore, eri, norb, nelec)
    e_torch = np.sort(np.linalg.eigvalsh(prob_torch.sparse().toarray()))
    e_nosym = np.sort(np.linalg.eigvalsh(prob_nosym.sparse().toarray()))
    np.testing.assert_allclose(e_torch, e_nosym, atol=1e-8)


def test_selected_complex_ci_rejects_real():
    """A real ERI is rejected — that is the real EigenProblem path's job."""
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    hcore, eri = _real_ci_integrals(norb, seed=26)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    with pytest.raises(ValueError, match="complex"):
        build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)


def test_selected_complex_ci_rejects_non_hermitian_eri():
    """An ERI lacking the required Hermitian symmetry is rejected."""
    from pyscf.fci import cistring

    norb, nelec = 4, (2, 2)
    rng = np.random.default_rng(27)
    hcore = _random_hermitian(norb, seed=27)
    eri = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    with pytest.raises(ValueError, match="Hermitian|conj|rs"):
        build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, norb, nelec)


# ---------------------------------------------------------------------------
# General-algebra solver wired for complex integrals (RDMs, occupancies, spin,
# symmetrization) — the full-parity path
# ---------------------------------------------------------------------------
def _full_strings(norb, nelec):
    """Full-CAS alpha/beta CI-string arrays for ``(norb, nelec)``."""
    from pyscf.fci import cistring

    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))
    return strs_a, strs_b


def test_complex_rdm_energy_matches_dense_oracle():
    """Energy rebuilt from the complex 1-/2-RDM equals the dense eigenvalue.

    The complex-correct ``complex_rdm`` builders (bra conjugated) must reproduce
    the ground-state energy of the dense ``direct_nosym`` operator via the same
    ``<dm1, h> + 1/2 <dm2, eri>`` contraction the solver uses.
    """
    from sci_solvers.complex_rdm import energy_from_rdm, make_rdm1_complex, make_rdm12_complex

    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=41)
    H = _dense_projected_ci(hcore, eri, norb, nelec)
    w, v = np.linalg.eigh(0.5 * (H + H.conj().T))
    strs_a, strs_b = _full_strings(norb, nelec)
    C = v[:, 0].reshape(len(strs_a), len(strs_b))

    dm1, (dma, dmb) = make_rdm1_complex(C, norb, nelec)
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-9)  # Hermitian
    assert np.isclose(dm1.trace().real, sum(nelec), atol=1e-8)
    np.testing.assert_allclose(np.diagonal(dma).imag, 0.0, atol=1e-9)  # real occupancies

    d1, dm2, _ = make_rdm12_complex(C, norb, nelec)
    e = energy_from_rdm(d1, dm2, hcore, eri)
    assert abs(e.imag) < 1e-8
    assert np.isclose(e.real, w[0], atol=1e-8)


def test_complex_s_squared_matches_pyscf_and_is_phase_invariant():
    """``s_squared_complex`` matches pyscf on a real state and ignores a phase."""
    from pyscf.fci import selected_ci

    from sci_solvers.complex_rdm import s_squared_complex

    norb, nelec = 4, (2, 2)
    hcore, eri = _real_ci_integrals(norb, seed=42)
    H = _dense_projected_ci(hcore.astype(complex), eri.astype(complex), norb, nelec)
    w, v = np.linalg.eigh(0.5 * (H + H.conj().T))
    strs_a, strs_b = _full_strings(norb, nelec)
    C = v[:, 0].reshape(len(strs_a), len(strs_b)).real

    ref = selected_ci.spin_square(
        selected_ci._as_SCIvector(np.ascontiguousarray(C), (strs_a, strs_b)), norb, nelec
    )[0]
    got = s_squared_complex(C.astype(complex), norb, nelec)
    assert np.isclose(ref, got, atol=1e-8)
    # A global phase must not change <S^2>.
    got_phase = s_squared_complex(C.astype(complex) * np.exp(1j * 0.7), norb, nelec)
    assert np.isclose(ref, got_phase, atol=1e-8)


def test_multiroot_complex_matches_dense_oracle():
    """A general-algebra solver on complex integrals returns the dense ground energy.

    End-to-end: complex ``hcore``/``eri`` auto-route through the complex builder,
    and the reported energy (rebuilt from the complex RDMs) matches the dense
    oracle; the state amplitudes stay complex and the 1-RDM is Hermitian with the
    correct trace.
    """
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=43)
    H = _dense_projected_ci(hcore, eri, norb, nelec)
    ref = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    strs_a, strs_b = _full_strings(norb, nelec)

    solver = create_solver("davidson", nroots=1, compute_rdm2=True, symmetrize_spin=False)
    out = solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)
    r = out[0].states[0]
    assert np.isclose(r.energy, ref[0], atol=1e-6)
    assert np.iscomplexobj(r.sci_state.amplitudes)
    np.testing.assert_allclose(r.rdm1, r.rdm1.conj().T, atol=1e-8)
    assert np.isclose(r.rdm1.trace().real, sum(nelec), atol=1e-8)
    occ = sum(np.asarray(o).sum() for o in r.orbital_occupancies)
    assert np.isclose(np.real(occ), sum(nelec), atol=1e-8)


def test_multiroot_real_vs_complex_paths_agree():
    """Real integrals and their zero-imaginary complex cast give identical results.

    Casting a real problem to complex must route through the new complex path yet
    return the same energy, occupancies, 1-RDM and 2-RDM as the untouched real
    ``float64`` path — the disentanglement guarantee.
    """
    norb, nelec = 4, (2, 2)
    hcore, eri = _real_ci_integrals(norb, seed=44)
    strs_a, strs_b = _full_strings(norb, nelec)

    def run(hc, er):
        solver = create_solver("davidson", nroots=1, compute_rdm2=True, symmetrize_spin=False)
        return solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hc, er, norb, nelec)[
            0
        ].states[0]

    r_real = run(hcore, eri)
    r_cplx = run(hcore.astype(np.complex128), eri.astype(np.complex128))

    assert np.isclose(r_real.energy, r_cplx.energy, atol=1e-8)
    np.testing.assert_allclose(np.real(r_cplx.rdm1), r_real.rdm1, atol=1e-7)
    np.testing.assert_allclose(np.imag(r_cplx.rdm1), 0.0, atol=1e-7)
    np.testing.assert_allclose(r_cplx.rdm2.real, r_real.rdm2, atol=1e-7)
    for oc, orl in zip(r_cplx.orbital_occupancies, r_real.orbital_occupancies):
        np.testing.assert_allclose(np.real(oc), orl, atol=1e-7)


def test_multiroot_complex_symmetrized_matches_full_space():
    """The symmetrized (both-sector) complex solve matches the full-space spectrum.

    With ``symmetrize_spin=True`` and ``nroots>1`` both parity sectors are solved
    on the complex packed :class:`SymmetrizedEigenProblem` and merged; the lowest
    roots must equal a plain full-space complex solve (and the dense oracle).
    """
    norb, nelec, nroots = 4, (2, 2), 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=45)
    H = _dense_projected_ci(hcore, eri, norb, nelec)
    ref = np.sort(np.linalg.eigvalsh(0.5 * (H + H.conj().T)))[:nroots]
    strs_a, strs_b = _full_strings(norb, nelec)

    def energies(symmetrize):
        solver = create_solver(
            "davidson",
            nroots=nroots,
            compute_rdm2=True,
            symmetrize_spin=symmetrize,
        )
        out = solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)
        return np.sort([r.energy for r in out[0].states])

    sym = energies(True)
    full = energies(False)
    np.testing.assert_allclose(sym, ref, atol=1e-6)
    np.testing.assert_allclose(full, ref, atol=1e-6)
    np.testing.assert_allclose(sym, full, atol=1e-6)


@pytest.mark.parametrize("adapter", REAL_ONLY_GENERAL_ALGEBRA_SOLVERS)
def test_multiroot_complex_rejects_real_only_adapter(adapter):
    """A real-only adapter + complex integrals raises a clear NotImplementedError."""
    norb, nelec = 4, (2, 2)
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=46)
    strs_a, strs_b = _full_strings(norb, nelec)
    solver = create_solver(adapter, nroots=1, compute_rdm2=False, symmetrize_spin=False)
    with pytest.raises(NotImplementedError, match="complex|davidson|arpack|lobpcg"):
        solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)


# ---------------------------------------------------------------------------
# SpinFilterMixin on the complex spatial path (spin projection via s_squared_complex)
# ---------------------------------------------------------------------------
def test_complex_spin_filter_projects_singlet():
    """A complex-capable solver with ``spin_sq=0`` returns only singlet roots.

    Real integrals cast to complex128 carry a well-defined spin structure but route
    through the complex path (``build_selected_complex_ci_problem``). The
    ``SpinFilterMixin`` must classify ``<S^2>`` via the complex-correct
    ``s_squared_complex`` and keep only the target-spin roots.
    """
    from pyscf.fci import cistring

    from sci_solvers.complex_rdm import s_squared_complex

    norb, nelec = 4, (2, 2)
    hcore, eri = _real_ci_integrals(norb, seed=51)  # real -> well-defined spin
    hcore = hcore.astype(np.complex128)  # cast routes through the complex builder
    eri = eri.astype(np.complex128)
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]))
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]))

    solver = create_solver(
        "arpack",
        symmetrize_spin=False,
        spin_sq=0.0,
        nroots=2,
        tol=1e-10,
        maxiter=500,
        oversample=4,
        max_attempts=5,
    )
    out = solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)
    states = out[0].states
    assert len(states) == 2
    for s in states:
        c = np.ascontiguousarray(s.sci_state.amplitudes, dtype=np.complex128)
        assert np.iscomplexobj(c)
        assert abs(s_squared_complex(c, norb, nelec)) < 0.1  # singlet (<S^2> ~ 0)
