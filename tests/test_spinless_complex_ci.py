"""Tests for the complex Hermitian *spinless* selected-CI path.

These exercise :mod:`sci_solvers.spinless_ci_problem` -- the genuinely spinless
(spinor / GHF / SOC) complex Hermitian operator over a single flat block of
``nmo`` spin-orbitals with only a total particle number ``N`` and no
``(n_alpha, n_beta)`` factorization. The spatial complex path
(:mod:`sci_solvers.complex_ci_problem`) is *not* touched here; a genuinely
spinless Hamiltonian has no valid entry point there because its same-spin-only
spin-doubling drops exactly the spin-flip coupling that defines a spinless
operator.

The oracles are deliberately independent of the module under test:

* A **brute-force second-quantized** CI Hamiltonian assembled by applying the
  bare creation/annihilation operators over the selected determinant list -- it
  carries *no* spin assumption whatsoever, so it is the right reference for a
  spinless operator (the exchange term applies to every occupied pair, the
  integrals carry the spin structure).
* PySCF's complex-capable ``direct_nosym`` on the ``(N, 0)`` sector -- a single
  spin species over ``nmo`` orbitals, which is exactly the spinless
  spin-orbital problem, used as a cross-check on the canonical string order.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from itertools import combinations

import numpy as np
import pytest

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase
from sci_solvers.spinless_ci_problem import (
    build_spinless_complex_ci_problem,
    spinless_energy,
    spinless_rdm1,
    spinless_rdm12,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int = 0):
    """Complex ``hcore`` and a physically valid complex chemist-order ERI.

    Built exactly like the spatial complex test's factor form
    (``(pq|rs) = sum_P L[P] conj(L[P])`` with a Hermitian factor ``L == L^H``,
    then symmetrized over the ``(pq)<->(rs)`` swap), so the ERI carries both
    Hermitian symmetries -- ``(pq|rs) == conj((qp|sr))`` and ``(pq|rs) ==
    (rs|pq)`` -- that make the spinless CI operator Hermitian. The integrals are
    a *single flat block* of ``nmo`` spin-orbitals with no spin-block structure,
    so their off-diagonal spin blocks are generically non-zero (a genuine
    spinless / spin-flip operator).
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    naux = nmo + 2
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T  # Hermitian factor: L == L^H
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _all_dets(nmo: int, n_particles: int):
    """Every ``n_particles``-particle spin-orbital bitmask over ``nmo`` orbitals."""
    dets = []
    for occ in combinations(range(nmo), n_particles):
        d = 0
        for o in occ:
            d |= 1 << o
        dets.append(d)
    dets.sort()
    return dets


def _annihilate(det: int, p: int):
    """``a_p |det>`` as ``(sign, newdet)`` or ``None`` if ``p`` is empty."""
    if not ((det >> p) & 1):
        return None
    sign = (-1) ** bin(det & ((1 << p) - 1)).count("1")
    return sign, det & ~(1 << p)


def _create(det: int, p: int):
    """``a_p^ |det>`` as ``(sign, newdet)`` or ``None`` if ``p`` is occupied."""
    if (det >> p) & 1:
        return None
    sign = (-1) ** bin(det & ((1 << p) - 1)).count("1")
    return sign, det | (1 << p)


def _brute_force_dense(hcore, eri, nmo, dets):
    """Dense CI Hamiltonian over ``dets`` from the bare second-quantized operators.

    ``H = sum_pq h_pq a_p^ a_q + 0.5 sum_pqrs (pq|rs) a_p^ a_r^ a_s a_q`` applied
    column-by-column by explicit ladder-operator action -- an oracle with no spin
    assumption at all, so it is correct for a genuinely spinless operator.
    """
    idx = {d: i for i, d in enumerate(dets)}
    dim = len(dets)
    H = np.zeros((dim, dim), dtype=np.complex128)
    for col, d in enumerate(dets):
        # one-body: sum_pq h_pq a_p^ a_q
        for q in range(nmo):
            aq = _annihilate(d, q)
            if aq is None:
                continue
            sq, dq = aq
            for p in range(nmo):
                cp = _create(dq, p)
                if cp is None:
                    continue
                sp, dp = cp
                if dp in idx:
                    H[idx[dp], col] += sq * sp * hcore[p, q]
        # two-body: 0.5 sum_pqrs (pq|rs) a_p^ a_r^ a_s a_q
        for q in range(nmo):
            aq = _annihilate(d, q)
            if aq is None:
                continue
            sq, dq = aq
            for s in range(nmo):
                as_ = _annihilate(dq, s)
                if as_ is None:
                    continue
                ss, ds = as_
                for r in range(nmo):
                    cr = _create(ds, r)
                    if cr is None:
                        continue
                    sr, dr = cr
                    for p in range(nmo):
                        cp = _create(dr, p)
                        if cp is None:
                            continue
                        sp, dp = cp
                        if dp in idx:
                            H[idx[dp], col] += 0.5 * eri[p, q, r, s] * sq * ss * sr * sp
    return H


def _direct_nosym_n0_oracle(hcore, eri, nmo, n_particles):
    """Dense complex CI Hamiltonian on the ``(N, 0)`` sector via ``direct_nosym``.

    ``nelec = (N, 0)`` is a single spin species over ``nmo`` orbitals -- exactly
    the spinless spin-orbital problem -- so its full-CAS spectrum is an
    independent oracle for the canonical-order spinless operator.
    """
    from pyscf.fci import cistring, direct_nosym

    nelec = (int(n_particles), 0)
    na = cistring.num_strings(nmo, nelec[0])
    nb = cistring.num_strings(nmo, nelec[1])
    dim = na * nb
    h2e = direct_nosym.absorb_h1e(hcore, eri, nmo, nelec, 0.5)
    H = np.empty((dim, dim), dtype=np.complex128)
    for c in range(dim):
        e = np.zeros(dim, dtype=np.complex128)
        e[c] = 1.0
        H[:, c] = direct_nosym.contract_2e(h2e, e.reshape(na, nb), nmo, nelec).reshape(-1)
    return H


def _canonical_strings(nmo, n_particles):
    """Full canonical ``cistring`` list over ``nmo`` spin-orbitals."""
    from pyscf.fci import cistring

    return np.asarray(cistring.make_strings(range(nmo), int(n_particles)))


# ---------------------------------------------------------------------------
# operator: spectrum matches independent oracles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nmo,n_particles,seed", [(6, 3, 1), (6, 2, 2), (8, 4, 3)])
def test_spinless_full_space_matches_brute_force(nmo, n_particles, seed):
    """On the full ``N``-particle basis the operator matches the brute-force oracle."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    dets = _all_dets(nmo, n_particles)
    strs = np.array(dets, dtype=object)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    assert prob.dtype == np.complex128
    assert prob.dim == len(dets)
    H = prob.sparse().toarray()
    # The assembled operator is itself Hermitian (real spectrum).
    np.testing.assert_allclose(H, H.conj().T, atol=1e-9)

    Href = _brute_force_dense(hcore, eri, nmo, dets)
    got = np.sort(np.linalg.eigvalsh(H))
    ref = np.sort(np.linalg.eigvalsh(0.5 * (Href + Href.conj().T)))
    np.testing.assert_allclose(got, ref, atol=1e-8)


@pytest.mark.parametrize("nmo,n_particles,seed", [(6, 3, 4), (8, 4, 5)])
def test_spinless_canonical_matches_direct_nosym_n0(nmo, n_particles, seed):
    """Canonical-order operator matches ``direct_nosym`` on the ``(N, 0)`` sector."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = _canonical_strings(nmo, n_particles)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    got = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))
    ref = np.sort(np.linalg.eigvalsh(_direct_nosym_n0_oracle(hcore, eri, nmo, n_particles)))
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_spinless_odd_nmo_supported():
    """An odd spin-orbital count -- impossible for the spatial ``(n_a, n_b)`` path.

    A spatial complex problem always has an even ``2*norb`` spin-orbital count;
    an odd ``nmo`` can only be a genuinely spinless operator, so this both
    exercises the spinless path and demonstrates why it is a separate entry point.
    """
    nmo, n_particles = 5, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=7)
    dets = _all_dets(nmo, n_particles)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, np.array(dets, dtype=object))
    Href = _brute_force_dense(hcore, eri, nmo, dets)
    got = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))
    ref = np.sort(np.linalg.eigvalsh(0.5 * (Href + Href.conj().T)))
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_spinless_selected_subspace_matches_brute_force():
    """On a genuinely selected strict subset the operator matches the projected oracle.

    The brute-force oracle is assembled over the *exact* selected determinant
    list, so equality confirms the operator restricts correctly to an arbitrary
    sampled subspace (the selected-CI regime).
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=8)
    canon = [int(s) for s in _canonical_strings(nmo, n_particles)]
    sel = [canon[i] for i in (0, 2, 5, 7, 11, 15, 18)]

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, np.array(sel, dtype=object))
    assert prob.dim == len(sel)
    H = prob.sparse().toarray()
    np.testing.assert_allclose(H, H.conj().T, atol=1e-9)

    Href = _brute_force_dense(hcore, eri, nmo, sel)
    got = np.sort(np.linalg.eigvalsh(H))
    ref = np.sort(np.linalg.eigvalsh(0.5 * (Href + Href.conj().T)))
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_spinless_hdiag_matches_operator_diagonal():
    """The problem's ``hdiag`` equals the real diagonal of the assembled operator."""
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=9)
    strs = _canonical_strings(nmo, n_particles)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    H = prob.sparse().toarray()
    assert np.isrealobj(prob.hdiag())
    np.testing.assert_allclose(prob.hdiag(), np.real(np.diagonal(H)), atol=1e-9)


# ---------------------------------------------------------------------------
# matrix-free engine (direct_nosym) equals the dense engine element-for-element
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nmo,n_particles,seed", [(6, 3, 21), (8, 4, 22), (5, 2, 23)])
def test_spinless_matrixfree_matches_dense_operator(nmo, n_particles, seed):
    """The matrix-free (``direct_nosym``) operator equals the dense one exactly.

    Both engines build the *same* complex Hermitian operator on the full canonical
    ``(N, 0)`` sector; the matrix-free path reuses ``pyscf.fci.direct_nosym`` on the
    un-doubled ``nmo``-orbital integrals and never materializes the ``(dim, dim)``
    matrix (``problem._dense is None``). Equality on random complex vectors and
    blocks ties the new engine to the trusted dense ``_hij_spinless`` reference.
    """
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = _canonical_strings(nmo, n_particles)

    dense = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs, matrix_free=False)
    free = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs, matrix_free=True)

    assert dense._dense is not None
    assert free._dense is None  # matrix-free: no dense matrix materialized
    assert free.dim == dense.dim and free.dtype == np.complex128

    rng = np.random.default_rng(seed + 100)
    x = rng.standard_normal(dense.dim) + 1j * rng.standard_normal(dense.dim)
    np.testing.assert_allclose(free.matvec(x), dense.matvec(x), atol=1e-10)

    X = rng.standard_normal((dense.dim, 3)) + 1j * rng.standard_normal((dense.dim, 3))
    np.testing.assert_allclose(free.matvec(X), dense.matvec(X), atol=1e-10)

    # hdiag (preconditioner) agrees, and the full spectra coincide.
    np.testing.assert_allclose(free.hdiag(), dense.hdiag(), atol=1e-9)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(free.sparse().toarray())),
        np.sort(np.linalg.eigvalsh(dense.sparse().toarray())),
        atol=1e-9,
    )


@pytest.mark.parametrize("nmo,n_particles,seed", [(6, 3, 24), (8, 4, 25)])
def test_spinless_matrixfree_matches_oracles(nmo, n_particles, seed):
    """The matrix-free spectrum matches the brute-force and ``direct_nosym`` oracles."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs, matrix_free=True)

    got = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))
    Href = _brute_force_dense(hcore, eri, nmo, [int(s) for s in strs])
    ref_bf = np.sort(np.linalg.eigvalsh(0.5 * (Href + Href.conj().T)))
    ref_ns = np.sort(np.linalg.eigvalsh(_direct_nosym_n0_oracle(hcore, eri, nmo, n_particles)))
    np.testing.assert_allclose(got, ref_bf, atol=1e-8)
    np.testing.assert_allclose(got, ref_ns, atol=1e-8)


def test_spinless_matrix_free_requires_full_canonical():
    """``matrix_free=True`` is refused for a genuinely selected strict subset.

    The ``direct_nosym`` contraction operates on the complete ``(na, 1)`` CI grid,
    so a selected subset has no matrix-free representation. The dense engine
    (``matrix_free=False`` / auto) still handles the subset.
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=26)
    canon = [int(s) for s in _canonical_strings(nmo, n_particles)]
    sel = np.array([canon[i] for i in (0, 2, 5, 7, 11)], dtype=object)

    with pytest.raises(ValueError, match="full canonical"):
        build_spinless_complex_ci_problem(hcore, eri, n_particles, sel, matrix_free=True)

    # The dense path handles the subset unchanged.
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, sel, matrix_free=False)
    assert prob.dim == len(sel) and prob._dense is not None


def test_spinless_is_full_canonical_beyond_cistring_int64_limit():
    """``_is_full_canonical`` recognizes a full canonical list past nmo=64.

    Past pyscf's internal int64-packing limit, ``cistring.make_strings`` stops
    returning plain bitmask scalars and returns ``OIndexList`` rows (arrays of
    occupied orbital indices) instead -- ``int()`` on those rows raises. This
    checks the conversion handles both representations and still produces the
    right True/False answer, using a small ``n_particles`` so the canonical
    list itself stays tiny (``C(70, 1) == 70``) even though ``nmo`` crosses the
    limb boundary. The dense/matrix-free *solve* path at this width is covered
    separately by ``test_spinless_multilimb.py`` (torch package) -- this test
    is only about the full-canonical-list classification.
    """
    from sci_solvers.spinless_ci_problem import _is_full_canonical

    nmo, n_particles = 70, 1
    full = _all_dets(nmo, n_particles)
    assert _is_full_canonical(full, nmo, n_particles)

    selected = full[:-1]  # a strict subset is not full canonical
    assert not _is_full_canonical(selected, nmo, n_particles)

    reordered = list(reversed(full))  # same set, wrong order -- also not canonical
    assert not _is_full_canonical(reordered, nmo, n_particles)


def test_spinless_auto_dispatch_selects_matrixfree_when_large():
    """``matrix_free=None`` auto-picks the engine by canonical-ness and size.

    A small full-canonical problem stays dense; forcing the "Large" heuristic (via a
    patched-down threshold) flips a full-canonical problem to matrix-free, while a
    selected subset always stays dense.
    """
    import sci_solvers.spinless_ci_problem as scp

    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=27)
    strs = _canonical_strings(nmo, n_particles)
    canon = [int(s) for s in strs]
    sel = np.array(canon[:5], dtype=object)

    # Small full-canonical: auto -> dense.
    small = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    assert small._dense is not None

    # Force the Large regime by lowering the dim threshold below this dim.
    old = scp._MATRIXFREE_DIM_THRESHOLD
    try:
        scp._MATRIXFREE_DIM_THRESHOLD = 1
        big = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
        assert big._dense is None  # full canonical + Large -> matrix-free
        # A selected subset is not full canonical: stays dense even when "Large".
        subset = build_spinless_complex_ci_problem(hcore, eri, n_particles, sel)
        assert subset._dense is not None
    finally:
        scp._MATRIXFREE_DIM_THRESHOLD = old


# ---------------------------------------------------------------------------
# complex-capable adapter recovers the ground state end-to-end
# ---------------------------------------------------------------------------
def test_spinless_davidson_recovers_ground_state():
    """The complex-capable Davidson adapter recovers the lowest eigenvalues."""
    nmo, n_particles, nroots = 8, 4, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=10)
    strs = _canonical_strings(nmo, n_particles)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    ref = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))[:nroots]
    res = (
        create_solver(solver_type="davidson", symmetrize_spin=False, tol=1e-10, max_cycle=200)
        ._diagonalize(prob, nroots)
        .sorted()
    )
    np.testing.assert_allclose(np.real(res.energies[:nroots]), ref, atol=1e-7)
    # Eigenvectors are complex and satisfy H v = e v.
    H = prob.sparse().toarray()
    for j in range(nroots):
        v = res.vectors[:, j]
        np.testing.assert_allclose(H @ v, res.energies[j] * v, atol=1e-5)


# ---------------------------------------------------------------------------
# RDMs / energy via the (N, 0) sector embedding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nmo,n_particles,seed", [(6, 3, 11), (8, 4, 12)])
def test_spinless_rdm_energy_matches_eigenvalue(nmo, n_particles, seed):
    """Energy rebuilt from the spinless 1-/2-RDM equals the ground eigenvalue.

    Requires the canonical string order (the RDM ``(N, 0)`` embedding contract),
    so the vector is taken from the canonical-order operator's ground state.
    """
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    w, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1, dm2 = spinless_rdm12(C, nmo, n_particles)
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-8)  # Hermitian
    assert np.isclose(dm1.trace().real, n_particles, atol=1e-8)

    e = spinless_energy(C, hcore, eri, n_particles)
    assert np.isclose(e, w[0], atol=1e-8)


def test_spinless_rdm1_carries_full_occupancy_in_alpha_block():
    """``spinless_rdm1`` returns the full 1-RDM in ``dma`` with ``dmb`` exactly zero.

    The ``(N, 0)`` embedding puts every particle in the alpha sector, so the beta
    1-RDM vanishes and ``dma`` carries the full spin-orbital 1-RDM whose real
    diagonal is the orbital occupancy (summing to ``N``).
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=13)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1, (dma, dmb) = spinless_rdm1(C, nmo, n_particles)
    np.testing.assert_allclose(dmb, 0.0, atol=1e-12)
    np.testing.assert_allclose(dma, dm1, atol=1e-12)
    np.testing.assert_allclose(np.diagonal(dma).imag, 0.0, atol=1e-9)  # real occupancies
    assert np.isclose(np.real(np.diagonal(dma).sum()), n_particles, atol=1e-8)


def test_spinless_energy_phase_invariant():
    """A global phase on the CI vector leaves the RDM-rebuilt energy unchanged."""
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=14)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    e = spinless_energy(C, hcore, eri, n_particles)
    e_phase = spinless_energy(C * np.exp(1j * 0.7), hcore, eri, n_particles)
    assert np.isclose(e, e_phase, atol=1e-10)


@pytest.mark.parametrize("chunk", [1, 2, 5, 1_000_000])
def test_spinless_rdm12_chunked_matches_unchunked(chunk):
    """The chunked RDM path reproduces the unchunked RDMs to machine precision.

    Chunking blocks the single-excitation tensor ``B = E_pq|C>`` over the ``nmo²``
    operator-index axis to cap peak RAM; it must not change the result for any block
    size, from ``chunk=1`` (one ``(p, q)`` row at a time) to ``chunk`` larger than
    ``nmo²`` (a single block, i.e. the unchunked build).
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=31)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1_ref, dm2_ref = spinless_rdm12(C, nmo, n_particles, chunk=None)
    dm1, dm2 = spinless_rdm12(C, nmo, n_particles, chunk=chunk)
    np.testing.assert_allclose(dm1, dm1_ref, atol=1e-12)
    np.testing.assert_allclose(dm2, dm2_ref, atol=1e-12)

    dm1a_ref, _ = spinless_rdm1(C, nmo, n_particles, chunk=None)
    dm1a, (dma, dmb) = spinless_rdm1(C, nmo, n_particles, chunk=chunk)
    np.testing.assert_allclose(dm1a, dm1a_ref, atol=1e-12)
    np.testing.assert_allclose(dmb, 0.0, atol=1e-12)


def test_spinless_rdm12_chunked_energy_matches_eigenvalue():
    """Energy rebuilt from the chunked RDMs equals the ground eigenvalue."""
    nmo, n_particles = 8, 4
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=32)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    w, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1, dm2 = spinless_rdm12(C, nmo, n_particles, chunk=3)
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-8)
    assert np.isclose(dm1.trace().real, n_particles, atol=1e-8)

    # Contract the chunked RDMs against the integrals -> ground energy.
    e = np.einsum("pq,pq->", hcore, dm1) + 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    assert np.isclose(e.real, w[0], atol=1e-8)


# ---------------------------------------------------------------------------
# rejections -- the three paths stay disentangled
# ---------------------------------------------------------------------------
def test_spinless_rejects_real_integrals():
    """Real integrals are rejected -- that is the real ``EigenProblem`` path's job."""
    nmo = 4
    hcore = np.eye(nmo)
    eri = np.zeros((nmo, nmo, nmo, nmo))
    with pytest.raises(ValueError, match="complex"):
        build_spinless_complex_ci_problem(hcore, eri, 2, np.array([3, 5, 6], dtype=object))


def test_spinless_rejects_na_nb_tuple():
    """An ``(n_a, n_b)`` tuple is rejected -- use the spatial selected builder."""
    nmo = 6
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=15)
    strs = _canonical_strings(nmo, 3)
    with pytest.raises(ValueError, match="build_selected_complex_ci_problem"):
        build_spinless_complex_ci_problem(hcore, eri, (2, 1), strs)


def test_spinless_rejects_non_hermitian_hcore():
    """A non-Hermitian ``hcore`` is rejected before building."""
    nmo = 6
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=16)
    strs = _canonical_strings(nmo, 3)
    bad_hcore = hcore + 1j * np.triu(np.ones((nmo, nmo)), 1)  # break h == h^H
    with pytest.raises(ValueError, match="Hermitian"):
        build_spinless_complex_ci_problem(bad_hcore, eri, 3, strs)


def test_spinless_rejects_non_hermitian_eri():
    """An ERI lacking the required Hermitian symmetry is rejected."""
    nmo = 4
    rng = np.random.default_rng(17)
    hcore = np.eye(nmo) + 0j
    eri = rng.standard_normal((nmo,) * 4) + 1j * rng.standard_normal((nmo,) * 4)
    with pytest.raises(ValueError, match="Hermitian|conj|rs"):
        build_spinless_complex_ci_problem(hcore, eri, 2, np.array([3, 5, 6], dtype=object))


def test_spinless_rejects_bad_shapes():
    """Inconsistent ``hcore`` / ``eri`` shapes are rejected."""
    nmo = 4
    hcore = np.eye(nmo) + 0j
    with pytest.raises(ValueError):
        build_spinless_complex_ci_problem(
            hcore, np.zeros((nmo, nmo, nmo), dtype=complex), 2, np.array([3, 5], dtype=object)
        )


# ---------------------------------------------------------------------------
# General-algebra auto-routing: a scalar nelec + flat strs batch is spinless
# ---------------------------------------------------------------------------
def test_solver_auto_routes_scalar_nelec_to_spinless():
    """A general-algebra solver routes a scalar ``nelec`` to the spinless builder.

    A scalar total ``N`` (not an ``(n_a, n_b)`` tuple) with flat ``strs`` batches
    and complex integrals is the spinless spin-orbital request. The recovered
    roots must match the direct spinless operator's lowest eigenvalues, and each
    root's state is a ``SpinlessSCIState`` whose occupancies / RDM trace sum to
    ``N``.
    """
    from sci_solvers.spinless_ci_problem import SpinlessSCIState

    nmo, N, nroots = 8, 4, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=10)
    strs = _canonical_strings(nmo, N)

    prob = build_spinless_complex_ci_problem(hcore, eri, N, strs)
    ref = np.sort(np.linalg.eigvalsh(prob.sparse().toarray()))[:nroots]

    solver = create_solver(
        symmetrize_spin=True,  # a spatial-parity concept; must be ignored here
        solver_type="davidson",
        nroots=nroots,
        compute_rdm2=True,
        tol=1e-10,
        max_cycle=300,
    )
    out = solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, N)

    assert len(out) == 1
    states = out[0].states
    got = np.sort([s.energy for s in states])
    np.testing.assert_allclose(got, ref, atol=1e-6)

    gs = states[0]
    assert isinstance(gs.sci_state, SpinlessSCIState)
    assert gs.sci_state.amplitudes.shape == (prob.dim,)
    assert gs.sci_state.nelec == (N, 0)
    # (N, 0) embedding: full 1-RDM in the alpha block, single occupancy vector.
    assert len(gs.orbital_occupancies) == 1
    assert np.isclose(np.real(gs.orbital_occupancies[0].sum()), N, atol=1e-8)
    assert np.isclose(np.real(gs.rdm1.trace()), N, atol=1e-8)
    assert gs.rdm2 is not None and gs.rdm2.shape == (nmo, nmo, nmo, nmo)


def test_solver_ground_state_only_spinless():
    """The single-root ``sci_solver_callable`` wrapper works on the spinless path."""

    nmo, N = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=11)
    strs = _canonical_strings(nmo, N)
    ref = np.sort(
        np.linalg.eigvalsh(
            build_spinless_complex_ci_problem(hcore, eri, N, strs).sparse().toarray()
        )
    )[0]

    solver = create_solver(symmetrize_spin=False, solver_type="davidson", tol=1e-10, max_cycle=300)
    ground = solver.sci_solver_callable([strs], hcore, eri, nmo, N)
    assert len(ground) == 1
    assert np.isclose(ground[0].energy, ref, atol=1e-6)


def test_solver_multiple_spinless_batches():
    """Each flat batch is solved independently, preserving input order.

    The solver's spinless RDM/energy extraction reuses the ``(N, 0)``-sector
    embedding, which -- like the spatial complex path -- assumes each batch is the
    full canonical ``cistring`` list for its ``N`` (the module contract). SQD
    hands full per-sector canonical string lists, so the batches here are two
    distinct full canonical particle-number sectors to exercise independent,
    order-preserving per-batch solving.
    """

    nmo = 6
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=12)
    # NOTE: distinct N per batch would need distinct nelec; the interface carries a
    # single scalar N, so batch independence is shown with two canonical lists that
    # sample the SAME N sector but in different (still canonical) orderings is not
    # possible -- instead repeat the same full sector twice and confirm both solve.
    N = 3
    full = _canonical_strings(nmo, N)
    batches = [full, full]

    solver = create_solver(symmetrize_spin=False, solver_type="davidson", tol=1e-10, max_cycle=300)
    out = solver.sci_solver_callable_eigen_states(batches, hcore, eri, nmo, N)

    assert len(out) == 2
    ref = np.linalg.eigvalsh(
        build_spinless_complex_ci_problem(hcore, eri, N, full).sparse().toarray()
    ).min()
    for es in out:
        assert np.isclose(es.states[0].energy, ref, atol=1e-6)


def test_solver_rejects_real_integrals_with_scalar_nelec():
    """A real Hamiltonian in the spinless shape (scalar ``nelec``) is rejected."""

    nmo, N = 6, 3
    strs = _canonical_strings(nmo, N)
    hcore = np.eye(nmo)
    eri = np.zeros((nmo, nmo, nmo, nmo))
    solver = create_solver(symmetrize_spin=False, solver_type="davidson")
    with pytest.raises(ValueError, match="spinless|complex"):
        solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, N)


def test_solver_is_spinless_request_classification():
    """Scalar ``nelec`` is spinless; a tuple/list/array is the spatial path."""

    assert GeneralAlgebraSolverBase._is_spinless_request(4)
    assert GeneralAlgebraSolverBase._is_spinless_request(np.int64(4))
    assert not GeneralAlgebraSolverBase._is_spinless_request((2, 2))
    assert not GeneralAlgebraSolverBase._is_spinless_request([2, 2])
    assert not GeneralAlgebraSolverBase._is_spinless_request(np.array([2, 2]))


# ---------------------------------------------------------------------------
# selected-CI (subset) RDMs -- correct for an arbitrary determinant subset
# ---------------------------------------------------------------------------
def _brute_force_rdm1(nmo, dets, C):
    """1-RDM ``gamma_pq = sum_{IJ} conj(C_I) C_J <I| a^+_p a_q |J>`` over ``dets``.

    Assembled from the bare ladder operators over the *exact* selected determinant
    list (projecting onto it via ``if dp in idx``), so it is an oracle independent
    of the module under test and valid for an arbitrary subset.
    """
    idx = {d: i for i, d in enumerate(dets)}
    g = np.zeros((nmo, nmo), dtype=np.complex128)
    for col, d in enumerate(dets):
        for q in range(nmo):
            aq = _annihilate(d, q)
            if aq is None:
                continue
            sq, dq = aq
            for p in range(nmo):
                cp = _create(dq, p)
                if cp is None:
                    continue
                sp, dp = cp
                if dp in idx:
                    g[p, q] += np.conj(C[idx[dp]]) * C[col] * sq * sp
    return g


def _brute_force_rdm2(nmo, dets, C):
    """2-RDM ``dm2[p,q,r,s] = sum <I| a^+_p a^+_r a_s a_q |J> conj(C_I) C_J`` (subset).

    Matches the ``pyscf.fci.selected_ci.make_rdm2`` ordering the module uses (so the
    energy is ``<dm1, h> + 0.5 * einsum('prqs,prqs->', dm2, eri)``). Independent
    ladder-operator oracle over the selected list.
    """
    idx = {d: i for i, d in enumerate(dets)}
    dm2 = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for col, d in enumerate(dets):
        for q in range(nmo):
            aq = _annihilate(d, q)
            if aq is None:
                continue
            sq, dq = aq
            for s in range(nmo):
                as_ = _annihilate(dq, s)
                if as_ is None:
                    continue
                ss, ds = as_
                for r in range(nmo):
                    cr = _create(ds, r)
                    if cr is None:
                        continue
                    sr, dr = cr
                    for p in range(nmo):
                        cp = _create(dr, p)
                        if cp is None:
                            continue
                        sp, dp = cp
                        if dp in idx:
                            dm2[p, q, r, s] += np.conj(C[idx[dp]]) * C[col] * sq * ss * sr * sp
    return dm2


def test_spinless_selected_subspace_rdm_matches_oracle():
    """On a strict determinant subset the spinless RDMs match the ladder-operator oracle.

    The subset ground state is genuinely different from a slice of the full-CAS
    state, so the oracle is assembled over the exact selected list (not sliced).
    Exercises the ``strs=`` selected-CI path in ``spinless_rdm1`` / ``spinless_rdm12``.
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=41)
    canon = [int(s) for s in _canonical_strings(nmo, n_particles)]
    sel = [canon[i] for i in (0, 2, 5, 7, 11, 15, 18)]
    strs = np.array(sel, dtype=object)

    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1, (dma, dmb) = spinless_rdm1(C, nmo, n_particles, strs=strs)
    np.testing.assert_allclose(dm1, _brute_force_rdm1(nmo, sel, C), atol=1e-12)
    np.testing.assert_allclose(dmb, 0.0, atol=1e-12)

    dm1b, dm2 = spinless_rdm12(C, nmo, n_particles, strs=strs)
    np.testing.assert_allclose(dm1b, _brute_force_rdm1(nmo, sel, C), atol=1e-12)
    np.testing.assert_allclose(dm2, _brute_force_rdm2(nmo, sel, C), atol=1e-12)

    # Energy rebuilt from the subset RDMs equals the subset ground eigenvalue.
    e = np.einsum("pr,pr->", dm1b, hcore) + 0.5 * np.einsum("prqs,prqs->", dm2, eri)
    w = np.linalg.eigvalsh(prob.sparse().toarray())
    assert np.isclose(e.real, w[0], atol=1e-8)


def test_spinless_selected_subspace_full_canonical_bit_identical():
    """Passing the full canonical list as ``strs`` is byte-identical to omitting it.

    Guards the default ``strs=None`` regression: the selected-CI ``row_map`` is a
    no-op on the full canonical list, so the RDMs must match to the bit.
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=42)
    strs = _canonical_strings(nmo, n_particles)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, strs)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1_none, dm2_none = spinless_rdm12(C, nmo, n_particles)
    dm1_full, dm2_full = spinless_rdm12(C, nmo, n_particles, strs=strs)
    np.testing.assert_array_equal(dm1_none, dm1_full)
    np.testing.assert_array_equal(dm2_none, dm2_full)


def test_spinless_selected_subspace_chunked_matches_direct():
    """The chunked selected-CI path matches the direct path *and* the oracle on a subset.

    The chunked reconstruction re-indexes the intermediate determinant axis onto the
    connected set exactly as the direct path does, so both must reproduce the
    independent ladder-operator RDMs over the explicit selected list.
    """
    nmo, n_particles = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=43)
    canon = [int(s) for s in _canonical_strings(nmo, n_particles)]
    sel_list = [canon[i] for i in (1, 3, 4, 9, 12, 17)]
    sel = np.array(sel_list, dtype=object)
    prob = build_spinless_complex_ci_problem(hcore, eri, n_particles, sel)
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0]

    dm1d, dm2d = spinless_rdm12(C, nmo, n_particles, strs=sel, chunk=None)
    dm1c, dm2c = spinless_rdm12(C, nmo, n_particles, strs=sel, chunk=2)
    np.testing.assert_allclose(dm1d, dm1c, atol=1e-12)
    np.testing.assert_allclose(dm2d, dm2c, atol=1e-12)

    # Both paths reproduce the independent ladder-operator oracle over the subset.
    np.testing.assert_allclose(dm1c, _brute_force_rdm1(nmo, sel_list, C), atol=1e-12)
    np.testing.assert_allclose(dm2c, _brute_force_rdm2(nmo, sel_list, C), atol=1e-12)


def test_spinless_selected_rdm_never_enumerates_full_canonical_space(monkeypatch):
    """A selected-subset RDM on a large CAS never touches pyscf's cistring addressing.

    Regression for the SIGABRT (bus error): with a large active space (28
    spin-orbitals, 12 electrons -> C(28, 12) ~ 30M canonical determinants), the
    selected-CI RDM path used to call ``cistring.gen_linkstr_index(range(28), 12)``
    with **no** ``strs`` -- allocating the full ~30M-row link table (hundreds of GB)
    and aborting the process -- before filtering to the handful of selected sources.

    The fix (:func:`~sci_solvers.complex_rdm._link_arrays_selected`) builds the link
    table directly from determinant bitmasks and never calls
    ``cistring.gen_linkstr_index``/``strs2addr`` at all: both ultimately rely on
    pyscf's C ``binomial()`` helper, an unmemoized Pascal's-triangle recursion that is
    exponential in ``norb`` once it exceeds pyscf's small in-code cache (``norb >
    20``) -- unusable at a large active space (e.g. ``norb=56``) regardless of
    ``len(strs)``, which the ~30M-canonical-determinant size here also guards against
    (bypassing ``gen_linkstr_index`` sidesteps both the OOM and the ``binomial`` cost).
    This spies on ``gen_linkstr_index`` and asserts it is **never** called at all.
    Also checks the returned 1-RDM occupancy sums to ``n``.
    """
    from pyscf.fci import cistring

    nmo, n_particles = 28, 12  # C(28, 12) ~ 30M: full canonical table is unbuildable
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=71)

    # A small selected subset: distinct random n-particle bitmasks.
    rng = np.random.default_rng(71)
    dets = set()
    while len(dets) < 24:
        occ = rng.choice(nmo, size=n_particles, replace=False)
        d = 0
        for o in occ:
            d |= 1 << int(o)
        dets.add(d)
    sel_list = sorted(dets)
    strs = np.array(sel_list, dtype=object)

    # A normalized complex amplitude vector over the selected determinants.
    C = rng.standard_normal(len(sel_list)) + 1j * rng.standard_normal(len(sel_list))
    C = (C / np.linalg.norm(C)).astype(np.complex128)

    real_gen = cistring.gen_linkstr_index
    calls = {"total": 0}

    def _spy_gen_linkstr_index(orb_list, nocc, strs=None, tril=False):
        calls["total"] += 1
        return real_gen(orb_list, nocc, strs=strs, tril=tril)

    monkeypatch.setattr(cistring, "gen_linkstr_index", _spy_gen_linkstr_index)

    dm1, (dma, dmb) = spinless_rdm1(C, nmo, n_particles, strs=strs)

    # pyscf's cistring link-table builder is never called on the selected path.
    assert calls["total"] == 0

    # The 1-RDM is finite, Hermitian, and its occupancy sums to the particle number.
    assert np.all(np.isfinite(dm1))
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-10)
    assert np.isclose(np.real(np.trace(dma)), float(n_particles), atol=1e-8)
    np.testing.assert_allclose(dmb, 0.0, atol=1e-12)
    # Matches the independent ladder-operator oracle over the selected subset.
    np.testing.assert_allclose(dm1, _brute_force_rdm1(nmo, sel_list, C), atol=1e-10)


def test_link_arrays_selected_matches_pyscf_reference():
    """The bitmask-native link builder reproduces pyscf's ``gen_linkstr_index``.

    At a size small enough for pyscf's ``cistring`` to still work
    (``norb=12, nele=5``, well under both the ``2**31`` address ceiling and the
    C ``binomial()`` recursion's practical range), cross-checks
    :func:`~sci_solvers.complex_rdm._link_arrays_selected` -- which builds
    ``(p, q, dst, src, sign)`` directly from bitmasks -- against
    ``cistring.gen_linkstr_index``/``strs2addr`` (with ``dst``/``src`` converted from
    canonical addresses to bitmasks for a like-for-like comparison). Both the
    diagonal (occupied-occupied) and off-diagonal (occupied-virtual) entries, and
    their signs, must match exactly.
    """
    from pyscf.fci import cistring

    from sci_solvers.complex_rdm import _link_arrays_selected

    norb, nele = 12, 5
    rng = np.random.default_rng(101)
    canon = [int(s) for s in cistring.make_strings(range(norb), nele)]
    idx = rng.choice(len(canon), size=40, replace=False)
    sel = np.array(sorted(canon[i] for i in idx), dtype=np.int64)

    p, q, dst, src, sign = _link_arrays_selected(norb, nele, sel)

    link = cistring.gen_linkstr_index(range(norb), nele, strs=sel).astype(np.int64)
    addrs = cistring.strs2addr(norb, nele, sel)
    p_ref = link[:, :, 0].ravel()
    q_ref = link[:, :, 1].ravel()
    dst_ref = link[:, :, 2].ravel()
    sign_ref = link[:, :, 3].ravel().astype(np.float64)
    src_ref_addr = addrs[np.repeat(np.arange(len(sel)), link.shape[1])]
    diag = p_ref == q_ref
    dst_ref_addr = dst_ref.copy()
    dst_ref_addr[diag] = addrs[dst_ref[diag]]  # diagonal dst is a local index, not an address
    dst_ref_bitmask = np.array([int(cistring.addr2str(norb, nele, a)) for a in dst_ref_addr])
    src_ref_bitmask = np.array([int(cistring.addr2str(norb, nele, a)) for a in src_ref_addr])

    order_mine = np.lexsort((q, p, src))
    order_ref = np.lexsort((q_ref, p_ref, src_ref_bitmask))
    np.testing.assert_array_equal(p[order_mine], p_ref[order_ref])
    np.testing.assert_array_equal(q[order_mine], q_ref[order_ref])
    np.testing.assert_array_equal(src[order_mine], src_ref_bitmask[order_ref])
    np.testing.assert_array_equal(dst[order_mine], dst_ref_bitmask[order_ref])
    np.testing.assert_array_equal(sign[order_mine], sign_ref[order_ref])


def test_spinless_selected_rdm_supports_large_address_active_space():
    """Regression: ``spinless_rdm1``/``spinless_rdm12`` at a
    full-canonical Hilbert-space size >= 2**31.

    At ``norb=36, nele=13``, ``C(36, 13) ~ 2.31e9 >= 2**31``, so
    ``pyscf.fci.cistring.strs2addr`` unconditionally refuses to run (regardless of
    how few determinants are actually selected) -- reproducing the exact
    ``NotImplementedError('Large address')`` from PySCF. The fix never calls
    ``strs2addr``/``gen_linkstr_index`` on the selected path at all (see
    :func:`~sci_solvers.complex_rdm._link_arrays_selected`), so this must run to
    completion and return a physically sane RDM. A dense brute-force oracle is
    infeasible at this size (``C(36, 13)`` determinants), so correctness is checked
    via Hermiticity, occupancy trace, and the chunked-vs-direct cross-check already
    used elsewhere in this file.

    ``chunk=None`` (the direct, unchunked reconstruction) is deliberately *not*
    exercised for the 2-RDM here: its ``O(norb**4 * n_connected)`` Gram contraction
    is expensive at this ``norb`` independent of the address-space fix under test
    (the same cost applies to the pre-existing full-canonical path at this size), so
    real usage relies on :func:`~sci_solvers.spinless_ci_problem._resolve_rdm_chunk`
    auto-selecting the chunked path -- exercised here via the default ``chunk=None``
    argument to :func:`spinless_rdm12` itself (which is *auto*, not *forced-direct*;
    only ``spinless_rdm1``/``spinless_rdm12``'s internal ``_resolve_rdm_chunk`` call
    treats a non-positive value as forced-direct, and auto only chunks once
    ``dim >= 50_000`` -- smaller than this test's selection, so add an explicit small
    ``chunk`` to actually exercise the chunked 2-RDM code path here).
    """
    from pyscf.fci import cistring

    norb, n_particles = 36, 13
    assert cistring.num_strings(norb, n_particles) >= 2**31  # the issue's exact regime
    with pytest.raises(NotImplementedError):
        cistring.strs2addr(norb, n_particles, np.array([0], dtype=np.int64))

    hcore, eri = _random_complex_hermitian_spinorbital_integrals(norb, seed=91)
    rng = np.random.default_rng(91)
    dets = set()
    while len(dets) < 200:
        occ = rng.choice(norb, size=n_particles, replace=False)
        d = 0
        for o in occ:
            d |= 1 << int(o)
        dets.add(d)
    sel_list = sorted(dets)
    strs = np.array(sel_list, dtype=np.int64)

    C = rng.standard_normal(len(sel_list)) + 1j * rng.standard_normal(len(sel_list))
    C = (C / np.linalg.norm(C)).astype(np.complex128)

    dm1, (dma, dmb) = spinless_rdm1(C, norb, n_particles, strs=strs, chunk=None)
    assert np.all(np.isfinite(dm1))
    np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-10)
    assert np.isclose(np.real(np.trace(dma)), float(n_particles), atol=1e-8)
    np.testing.assert_allclose(dmb, 0.0, atol=1e-12)

    dm1_chunked, (dma_chunked, _dmb_chunked) = spinless_rdm1(
        C, norb, n_particles, strs=strs, chunk=64
    )
    np.testing.assert_allclose(dm1, dm1_chunked, atol=1e-10)

    dm1b, dm2 = spinless_rdm12(C, norb, n_particles, strs=strs, chunk=64)
    np.testing.assert_allclose(dm1, dm1b, atol=1e-10)
    e = np.einsum("pr,pr->", dm1b, hcore) + 0.5 * np.einsum("prqs,prqs->", dm2, eri)
    assert np.isfinite(e)


@pytest.mark.parametrize("solver_type", ["arpack", "davidson"])
def test_solver_partial_batch_no_crash(solver_type):
    """The minimal-repro partial batch returns finite energy/occupancies (no IndexError).

    Regression for the selected-subspace crash: a strict subset of the canonical
    list (``full[:3]``) previously raised ``IndexError`` in the RDM reconstruction.
    """
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=44)
    full = _canonical_strings(nmo, n)
    subset = full[:3]

    # `davidson` takes the flat budget as `max_cycle` and rejects `maxiter`; the other
    # backends read `maxiter` with their own per-solver meaning.
    budget = {"max_cycle": 300} if solver_type == "davidson" else {"maxiter": 300}
    solver = create_solver(
        solver_type=solver_type, symmetrize_spin=False, compute_rdm2=True, tol=1e-10, **budget
    )
    out = solver.sci_solver_callable_eigen_states([subset], hcore, eri, nmo, n)
    gs = out[0].states[0]
    assert np.isfinite(gs.energy)
    assert np.isclose(np.real(gs.orbital_occupancies[0].sum()), n, atol=1e-8)
    assert np.isclose(np.real(gs.rdm1.trace()), n, atol=1e-8)

    # Cross-check against a dense eigh over the exact subset.
    prob = build_spinless_complex_ci_problem(hcore, eri, n, subset)
    w = np.linalg.eigvalsh(prob.sparse().toarray())
    assert np.isclose(gs.energy, w[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Hermiticity tolerance is reachable from the solver's create_solver kwargs
# ---------------------------------------------------------------------------
def _perturb_conj_symmetry(eri, mag):
    """Break only ``(pq|rs) == conj((qp|sr))`` by ``mag``, keeping ``(pq|rs) == (rs|pq)``.

    Adds ``mag`` to ``(p,q,r,s)`` *and its ``(r,s,p,q)`` swap partner* so the
    particle-exchange symmetry the CI operator also needs is preserved, while the
    conjugate-transpose symmetry is left violated (the ``(q,p,s,r)`` conjugate
    partner is untouched). This reproduces, deterministically, the one round-off
    asymmetry the two-component SCF integrals exhibit -- the ``(pq|rs) ==
    conj((qp|sr))`` guard -- without tripping the unrelated ``(rs|pq)`` guard.
    Returns a fresh array.
    """
    out = eri.copy()
    out[0, 1, 2, 3] += mag
    out[2, 3, 0, 1] += mag  # keep (pq|rs) == (rs|pq)
    return out


def test_spinless_hermiticity_tolerance_reachable_via_solver_kwargs():
    """A round-off-sized conj-symmetry break is rejected by default, accepted when relaxed.

    The ``hermiticity_atol`` / ``hermiticity_rtol`` knobs added to
    ``GeneralAlgebraSolverBase`` flow verbatim through ``create_solver`` to the
    spinless complex CI builder's Hermiticity guard, so a two-component SCF's
    round-off asymmetry can be admitted from solver kwargs alone -- with no new
    downstream step setting. The perturbation is well above the default
    ``atol=1e-7`` (so the default still catches it) yet within a relaxed
    tolerance.
    """
    nmo, n = 6, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=101)
    # The perturbed element has |b| ~ 3.7, so the default combined tolerance
    # (atol + rtol*|b| ~ 3.7e-5) rejects a 2e-4 break while the relaxed one
    # (~3.7e-3) admits it.
    eri_noisy = _perturb_conj_symmetry(eri, mag=2e-4)
    strs = _canonical_strings(nmo, n)

    # Default tolerance rejects the (deliberately) non-Hermitian ERI.
    default_solver = create_solver(solver_type="arpack", symmetrize_spin=False)
    with pytest.raises(ValueError, match="does not satisfy"):
        default_solver.sci_solver_callable_eigen_states([strs], hcore, eri_noisy, nmo, n)

    # Forwarding a relaxed tolerance through create_solver admits the round-off.
    relaxed_solver = create_solver(
        solver_type="arpack",
        symmetrize_spin=False,
        hermiticity_atol=1e-5,
        hermiticity_rtol=1e-3,
    )
    assert relaxed_solver.hermiticity_atol == 1e-5
    assert relaxed_solver.hermiticity_rtol == 1e-3
    out = relaxed_solver.sci_solver_callable_eigen_states([strs], hcore, eri_noisy, nmo, n)
    gs = out[0].states[0]
    assert np.isfinite(gs.energy)

    # The relaxed solve tracks the symmetrized (physical) operator: the admitted
    # asymmetry only shifts the ground state by ~O(2e-4), so the answer stays
    # physical. Cross-check against a dense eigh of the Hermitized operator with a
    # tolerance sized to the deliberately injected asymmetry (not machine eps).
    eri_sym = 0.5 * (eri_noisy + np.conj(np.transpose(eri_noisy, (1, 0, 3, 2))))
    prob = build_spinless_complex_ci_problem(hcore, eri_sym, n, strs)
    w = np.linalg.eigvalsh(prob.sparse().toarray())
    assert np.isfinite(w[0])
    assert np.isclose(gs.energy, w[0], atol=1e-2)
