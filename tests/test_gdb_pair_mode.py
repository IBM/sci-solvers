"""Tests for the non-product (GDB) determinant-pair CI basis on the general-algebra solvers.

Real and complex ``mode="gdb"`` support: every
general-algebra solver accepts an explicit parallel ``(alpha_i, beta_i)`` pair
list instead of a Cartesian product, backed by
:class:`~sci_solvers.eigen_problem.PairCIProblem` and the vendored
``torch_sci_solver`` pair engine.

The correctness anchors are:

* **product-vs-pairs equivalence** — feeding a full ``alpha x beta`` product as a
  pair list under ``mode="gdb"`` must reproduce ``mode="tpb"`` (energies + RDMs),
  for both real and complex Hermitian integrals. This is the upstream pair
  engine's own correctness anchor and is what the complex bra-conjugation fix in
  ``rdm_pairs.py`` is validated against (without it the complex 1-RDM is
  non-Hermitian and the energy-from-RDM is wrong);
* **genuinely non-product subset** — a pair list that is *not* a product must
  match a dense ``eigh`` over exactly that pair basis;
* **spin-summed 2-RDM** — the GDB ``rdm2`` must equal a brute-force Jordan-Wigner
  2-RDM, for real and genuinely complex integrals and for open shells;
* **complex gating** — a complex GDB solve on a non-complex-capable solver must be
  rejected loudly (it goes through the same complex gate as the product path).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from itertools import combinations

import numpy as np
import pytest

from sci_solvers.eigen_problem import PairCIProblem
from sci_solvers.energy_solvers.factory import (
    AVAILABLE_SOLVERS,
    create_solver,
    is_complex_capable,
    is_general_algebra,
)
from sci_solvers.hamiltonian import HamiltonianData

# Solver lists derived from the capability flags (never hand-listed, so they never
# drift). ``evsl`` carries a costly DOS pre-pass; it is real-only and covered by the
# real sweeps here like any other matrix-free general-algebra solver.
GENERAL_ALGEBRA_SOLVERS = [n for n in AVAILABLE_SOLVERS if is_general_algebra(n)]
# ``pyscf_davidson`` solves through PySCF's product ``kernel_fixed_space`` and cannot
# represent a non-product basis, so it declines ``mode="gdb"`` (asserted separately in
# ``test_gdb_rejected_on_pyscf_davidson``). Every other general-algebra solver is
# matrix-free over the pair operator and takes the GDB path.
GDB_SOLVERS = [n for n in GENERAL_ALGEBRA_SOLVERS if n != "pyscf_davidson"]
COMPLEX_CAPABLE_GDB_SOLVERS = [n for n in GDB_SOLVERS if is_complex_capable(n)]
REAL_ONLY_GDB_SOLVERS = [n for n in GDB_SOLVERS if not is_complex_capable(n)]

ATOL = 1e-7
# The iterative solvers must converge tightly for the *2-RDM* (quadratic in the
# eigenvector, so it amplifies the residual) to match across bases to 1e-7; ARPACK
# already converges to ~1e-13, but the in-house Davidson needs the tighter tol.
_MAXITER = 2000


def _solve_kw(name: str) -> dict:
    """Shared solver kwargs, with the iteration budget under each solver's own name.

    ``davidson`` exposes the flat budget as ``max_cycle`` and rejects the base's
    ``maxiter`` spelling; the other backends read ``maxiter`` with their own
    per-solver meaning.
    """
    budget = "max_cycle" if name == "davidson" else "maxiter"
    return {"tol": 1e-12, budget: _MAXITER}


def _skip_if_unavailable(name: str) -> None:
    """Skip when a solver's optional backend (or complex PETSc build) is missing."""
    if name == "primme":
        pytest.importorskip("primme")
    elif name == "slepc":
        pytest.importorskip("slepc4py")


def _skip_slepc_real_scalar(name: str, complex_problem: bool) -> None:
    """Skip complex GDB on a real-scalar PETSc build (unsupported by the binary)."""
    if name == "slepc" and complex_problem:
        from petsc4py import PETSc

        if not np.iscomplexobj(np.empty(0, dtype=PETSc.ScalarType)):
            pytest.skip("PETSc built with real scalars; complex path unsupported")


def _all_strings(norb: int, nelec: int) -> np.ndarray:
    return np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(norb), nelec)),
        dtype=np.int64,
    )


def _real_integrals(norb: int, seed: int = 0):
    """Real symmetric hcore and a full-permutational-symmetry real ERI."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    hcore = h + h.T
    eri = np.zeros((norb, norb, norb, norb))
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb))
        L = g + g.T
        eri += np.einsum("pq,rs->pqrs", L, L)
    return hcore, eri


def _complex_integrals(norb: int, seed: int = 0):
    """Complex Hermitian hcore and a physically valid complex chemist-order ERI.

    Same construction as ``test_complex_hermitian`` — a Hermitian Cholesky-like
    factor gives sym1, and the ``(pq)<->(rs)`` symmetrization enforces sym2, so the
    CI operator is Hermitian.
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _product_as_pairs(strs_a: np.ndarray, strs_b: np.ndarray):
    """Flatten a full ``alpha x beta`` product into parallel pair lists."""
    A, B = np.meshgrid(strs_a, strs_b, indexing="ij")
    return A.ravel(), B.ravel()


# ---------------------------------------------------------------------------
# mode validation
# ---------------------------------------------------------------------------
def test_invalid_mode_rejected():
    """An unknown ``mode`` is rejected at construction, not silently ignored."""
    with pytest.raises(ValueError, match="mode must be 'tpb' or 'gdb'"):
        create_solver("arpack", symmetrize_spin=False, mode="nonsense")


def test_default_mode_is_tpb():
    """The Cartesian-product basis stays the default when ``mode`` is unset."""
    solver = create_solver("arpack", symmetrize_spin=False)
    assert solver.mode == "tpb"


# ---------------------------------------------------------------------------
# Real product-vs-pairs equivalence, across every general-algebra solver
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GDB_SOLVERS)
def test_real_gdb_matches_tpb_on_full_product(name):
    """A full product fed as a GDB pair list reproduces TPB energy and RDMs."""
    _skip_if_unavailable(name)
    norb, na, nb = 4, 2, 2
    hcore, eri = _real_integrals(norb, seed=7)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)

    tpb = create_solver(
        name, symmetrize_spin=False, mode="tpb", compute_rdm2=True, **_solve_kw(name)
    )
    r_tpb = tpb.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, (na, nb))[0]

    pa, pb = _product_as_pairs(sa, sb)
    gdb = create_solver(
        name, symmetrize_spin=False, mode="gdb", compute_rdm2=True, **_solve_kw(name)
    )
    r_gdb = gdb.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, (na, nb))[0]

    assert abs(r_tpb.states[0].energy - r_gdb.states[0].energy) < 1e-8
    np.testing.assert_allclose(r_tpb.states[0].rdm1, r_gdb.states[0].rdm1, atol=ATOL)
    np.testing.assert_allclose(r_tpb.states[0].rdm2, r_gdb.states[0].rdm2, atol=ATOL)


@pytest.mark.parametrize("name", GDB_SOLVERS)
def test_real_gdb_nonproduct_subset_matches_dense(name):
    """A genuinely non-product real pair subset matches dense eigh over that basis."""
    _skip_if_unavailable(name)
    norb, na, nb = 4, 2, 2
    hcore, eri = _real_integrals(norb, seed=11)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)
    # A non-product selection (no full row/column of the product grid).
    pa = np.array([sa[0], sa[1], sa[3], sa[2], sa[0]], dtype=np.int64)
    pb = np.array([sb[0], sb[2], sb[1], sb[3], sb[3]], dtype=np.int64)

    gdb = create_solver(name, symmetrize_spin=False, mode="gdb", **_solve_kw(name))
    res = gdb.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, (na, nb))[0]

    prob = PairCIProblem(pa, pb, hcore, eri, norb, (na, nb))
    w = np.linalg.eigvalsh(prob.sparse().toarray())
    assert abs(res.states[0].energy - w[0]) < 1e-8


@pytest.mark.parametrize("name", ["arpack", "torch_sci"])
def test_compute_energy_leaves_gdb_pair_lists_untouched(name):
    """``compute_energy`` must not sort a GDB batch's two lists independently.

    ``strs_a[i]`` and ``strs_b[i]`` form one ``(alpha_i, beta_i)`` pair, so the TPB
    normalization (``np.unique`` per list) would destroy the pairing: here ``strs_a``
    has two distinct values and ``strs_b`` two, so lengths would still match and the
    batch would be solved on the wrong pair set -- silently.
    """
    norb, na, nb = 4, 2, 1
    hcore, eri = _real_integrals(norb, seed=21)
    ham = HamiltonianData(
        hcore=hcore, eri=eri, e_nuc=0.0, num_orbitals=norb, num_elec_a=na, num_elec_b=nb
    )
    solver = create_solver(
        name,
        symmetrize_spin=False,
        mode="gdb",
        hamiltonian=ham,
        **({"device": "cpu"} if name == "torch_sci" else {}),
    )
    # alpha=3 pairs with two different betas; beta=1 also pairs with alpha=5.
    pa = np.array([3, 3, 5], dtype=np.int64)
    pb = np.array([1, 2, 1], dtype=np.int64)

    (result,) = solver.compute_energy(ci_strs_batches=[(pa, pb)])

    prob = PairCIProblem(pa, pb, hcore, eri, norb, (na, nb))
    w = np.linalg.eigvalsh(prob.sparse().toarray())
    assert result.energy == pytest.approx(w[0], abs=1e-8)


# ---------------------------------------------------------------------------
# Complex product-vs-pairs equivalence + Hermitian/real-energy checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", COMPLEX_CAPABLE_GDB_SOLVERS)
def test_complex_gdb_matches_tpb_on_full_product(name):
    """Complex full product as a GDB pair list reproduces complex TPB.

    This is the correctness anchor for the guarded bra-conjugation fix in
    ``rdm_pairs.py``: the complex 1-RDM must be Hermitian and the RDMs must match
    the (independently conjugate-correct) complex product path.
    """
    _skip_if_unavailable(name)
    _skip_slepc_real_scalar(name, complex_problem=True)
    norb, na, nb = 4, 2, 2
    hcore, eri = _complex_integrals(norb, seed=13)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)

    tpb = create_solver(
        name, symmetrize_spin=False, mode="tpb", compute_rdm2=True, **_solve_kw(name)
    )
    r_tpb = tpb.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, (na, nb))[0]

    pa, pb = _product_as_pairs(sa, sb)
    gdb = create_solver(
        name, symmetrize_spin=False, mode="gdb", compute_rdm2=True, **_solve_kw(name)
    )
    r_gdb = gdb.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, (na, nb))[0]

    assert abs(r_tpb.states[0].energy - r_gdb.states[0].energy) < 1e-7
    dm1 = r_gdb.states[0].rdm1
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=ATOL)  # Hermitian
    np.testing.assert_allclose(r_tpb.states[0].rdm1, dm1, atol=ATOL)
    np.testing.assert_allclose(r_tpb.states[0].rdm2, r_gdb.states[0].rdm2, atol=ATOL)


# ---------------------------------------------------------------------------
# Spin-summed 2-RDM against a brute-force reference
# ---------------------------------------------------------------------------
def _exact_spin_summed_rdm2(
    hcore: np.ndarray, eri: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Exact ``dm2[p,q,r,s] = sum_spins <a+_p a+_r a_s a_q>`` of the full-CI ground state.

    Brute force: the ground state of the full-space complex CI problem, embedded in
    the ``2 norb``-mode Fock space (alpha modes ``0..norb-1``, beta ``norb..2norb-1``),
    with the ladder operators built by Jordan-Wigner. Independent of every RDM builder.

    Notes
    -----
    The Jordan-Wigner matrices are real, so ``a+_p = a_p^T`` and
    ``<a+_p a+_r a_s a_q> = (a_r a_p psi)^H (a_s a_q psi)``: only the pair-annihilated
    vectors ``v[i, j] = a_i a_j psi`` are needed.
    """
    from sci_solvers.complex_ci_problem import build_complex_ci_problem

    n_modes = 2 * norb
    dim = 2**n_modes
    ann = []
    for j in range(n_modes):
        a = np.zeros((dim, dim))
        for x in range(dim):
            if x >> j & 1:
                a[x ^ (1 << j), x] = (-1) ** bin(x & ((1 << j) - 1)).count("1")
        ann.append(a)
    sa = _all_strings(norb, nelec[0])
    sb = _all_strings(norb, nelec[1])
    h_full = build_complex_ci_problem(
        np.asarray(hcore, dtype=np.complex128), np.asarray(eri, dtype=np.complex128), norb, nelec
    )
    coeffs = np.linalg.eigh(h_full.sparse().toarray())[1][:, 0]
    psi = np.zeros(dim, dtype=np.complex128)
    psi[[int(b) << norb | int(a) for a in sa for b in sb]] = coeffs
    v = np.array([[ann[i] @ (ann[j] @ psi) for j in range(n_modes)] for i in range(n_modes)])
    dm2 = np.zeros((norb,) * 4, dtype=np.complex128)
    # (p, q) carry spin s1 and (r, s) spin s2; v[s2 + r, s1 + p] = a_r a_p psi.
    for s1, s2 in ((0, 0), (0, norb), (norb, 0), (norb, norb)):
        blk = v[s2 : s2 + norb, s1 : s1 + norb]
        dm2 += np.einsum("rpx,sqx->pqrs", blk.conj(), blk)
    return dm2


@pytest.mark.parametrize("nelec", [(2, 1), (1, 2), (3, 1), (2, 2)])
@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("name", COMPLEX_CAPABLE_GDB_SOLVERS)
def test_gdb_rdm2_matches_exact(name, kind, nelec):
    """The GDB spin-summed 2-RDM equals the exact one, open shells included.

    The mixed-spin blocks must be summed as ``Gab + Gba`` with ``Gba`` the
    particle-exchanged ``Gab``, not as ``2 Gab``. The two agree for real integrals and for
    ``n_a == n_b``, but not for a complex open shell, which needs an ERI with a nonzero
    imaginary part to show it. A ``2 Gab`` error is antisymmetric under the exchange and
    cancels in the energy, so only the 2-RDM itself is checked here.
    """
    from test_complex_tpb_double_excitations import _genuinely_complex_integrals

    _skip_if_unavailable(name)
    _skip_slepc_real_scalar(name, complex_problem=kind == "complex")
    norb = 4
    if kind == "real":
        hcore, eri = _real_integrals(norb, seed=3)
    else:
        hcore, eri = _genuinely_complex_integrals(norb, seed=3)
        assert np.abs(eri.imag).max() > 0.1
    pa, pb = _product_as_pairs(_all_strings(norb, nelec[0]), _all_strings(norb, nelec[1]))
    solver = create_solver(
        name, symmetrize_spin=False, mode="gdb", compute_rdm2=True, **_solve_kw(name)
    )
    state = solver.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, nelec)[0]
    np.testing.assert_allclose(
        state.states[0].rdm2, _exact_spin_summed_rdm2(hcore, eri, norb, nelec), atol=ATOL
    )


# ---------------------------------------------------------------------------
# complex gating on the GDB path
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_ONLY_GDB_SOLVERS, reason="no real-only GDB-capable solver")
@pytest.mark.parametrize("name", REAL_ONLY_GDB_SOLVERS)
def test_complex_gdb_rejected_on_real_only_solver(name):
    """A complex GDB solve on a non-complex-capable solver raises, not silently drops Im."""
    _skip_if_unavailable(name)
    norb, na, nb = 4, 2, 2
    hcore, eri = _complex_integrals(norb, seed=17)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)
    pa, pb = _product_as_pairs(sa, sb)

    solver = create_solver(name, symmetrize_spin=False, mode="gdb")
    with pytest.raises(NotImplementedError):
        solver.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, (na, nb))


def test_slepc_storage_knob_does_not_collide_with_basis_mode():
    """``slepc`` has an orthogonal ``storage`` knob; it must not shadow basis ``mode``.

    Construction-only (no ``slepc4py`` needed): the family-wide ``mode="gdb"`` must survive
    alongside ``storage="aij"`` so ``slepc`` can actually reach the GDB dispatch.
    """
    s = create_solver("slepc", symmetrize_spin=False, mode="gdb", storage="aij")
    assert s.mode == "gdb"
    assert s.storage == "aij"

    default = create_solver("slepc", symmetrize_spin=False)
    assert default.mode == "tpb"
    assert default.storage == "shell"

    with pytest.raises(ValueError, match="storage must be 'shell' or 'aij'"):
        create_solver("slepc", symmetrize_spin=False, storage="bogus")


def test_gdb_rejected_on_pyscf_davidson():
    """``pyscf_davidson`` cannot represent a non-product basis, so it declines GDB."""
    norb, na, nb = 4, 2, 2
    hcore, eri = _real_integrals(norb, seed=5)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)
    pa, pb = _product_as_pairs(sa, sb)

    solver = create_solver("pyscf_davidson", symmetrize_spin=False, mode="gdb")
    assert solver.supports_gdb is False
    with pytest.raises(NotImplementedError, match="does not support mode='gdb'"):
        solver.sci_solver_callable_eigen_states([(pa, pb)], hcore, eri, norb, (na, nb))


# ---------------------------------------------------------------------------
# PairCIProblem shape / dedup invariants
# ---------------------------------------------------------------------------
def test_pair_problem_dedups_and_has_no_product_shape():
    """PairCIProblem dim is the number of unique pairs; no Na/Nb attributes."""
    norb, na, nb = 4, 2, 2
    hcore, eri = _real_integrals(norb, seed=1)
    sa = _all_strings(norb, na)
    sb = _all_strings(norb, nb)
    # Duplicate one pair; PairBasis should collapse it.
    pa = np.array([sa[0], sa[1], sa[0]], dtype=np.int64)
    pb = np.array([sb[0], sb[1], sb[0]], dtype=np.int64)
    prob = PairCIProblem(pa, pb, hcore, eri, norb, (na, nb))
    assert prob.dim == 2  # the duplicate collapsed
    assert not hasattr(prob, "Na")
    assert not hasattr(prob, "Nb")
    # strs_a / strs_b are read back from the (lex-sorted) basis, parallel and equal length.
    assert prob.strs_a.shape == prob.strs_b.shape == (2,)


def test_pair_problem_rejects_mismatched_lengths():
    """Alpha and beta lists must be parallel; unequal lengths are rejected."""
    norb, na, nb = 4, 2, 2
    hcore, eri = _real_integrals(norb, seed=1)
    with pytest.raises(ValueError, match="parallel pair lists"):
        PairCIProblem(
            np.array([3, 5], dtype=np.int64),
            np.array([3], dtype=np.int64),
            hcore,
            eri,
            norb,
            (na, nb),
        )
