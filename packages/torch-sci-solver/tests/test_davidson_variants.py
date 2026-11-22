"""Tests for the opt-in Davidson variants: Olsen correction, restart reuse, per-root start.

``davidson_correction="olsen"``, ``davidson_restart="reuse"``/``"thick"`` and a
``(k, len)`` ``init_vectors`` entry with ``n_roots > 1`` must each converge to the
same eigenpairs as the default solve. The defaults are covered by every other test
file. H6/STO-3G full CI (400 determinants) with ``davidson_block_size=6`` makes
Davidson restart several times, so the restart modes are exercised.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

pyscf = pytest.importorskip("pyscf")

from pyscf import ao2mo, gto, scf  # noqa: E402
from torch_sci_solver.torch_impl.davidson import davidson_torch  # noqa: E402
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver  # noqa: E402

KW = dict(device="cpu", davidson_block_size=6, max_davidson_iters=200, davidson_tolerance=1e-8)


def _h6():
    mol = gto.M(atom="; ".join(f"H 0 0 {i * 1.0}" for i in range(6)), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return h1, eri, norb, mol.nelec


H1, ERI, NORB, NELEC = _h6()
STRS = np.array(
    sorted(sum(1 << o for o in c) for c in itertools.combinations(range(NORB), NELEC[0])),
    dtype=np.int64,
)


def _batch(mode: str):
    if mode == "tpb":
        return (STRS, STRS)
    return np.repeat(STRS, len(STRS)), np.tile(STRS, len(STRS))


def _solve(solver, mode: str, **kwargs):
    res = solver([_batch(mode)], H1, ERI, NORB, NELEC, **kwargs)[0].states
    return res, solver.stats["per_call_log"][-1]["subspaces"][0]


@pytest.fixture(scope="module", params=["tpb", "gdb"])
def mode(request) -> str:
    """Basis layout under test."""
    return request.param


@pytest.fixture(scope="module")
def reference(mode):
    """Default single-root solve: (energy, n_matvecs)."""
    res, perf = _solve(make_torch_sci_solver(mode=mode, **KW), mode)
    return res[0].energy, perf["n_matvecs"]


@pytest.mark.parametrize("restart", ["recompute", "reuse", "thick"])
@pytest.mark.parametrize("correction", ["davidson", "olsen"])
def test_variant_matches_default_energy(mode, reference, correction, restart) -> None:
    """Every correction x restart combination converges to the default energy."""
    solver = make_torch_sci_solver(
        mode=mode, davidson_correction=correction, davidson_restart=restart, **KW
    )
    res, _ = _solve(solver, mode)
    assert res[0].state == "converged"
    assert res[0].energy == pytest.approx(reference[0], abs=1e-9)


def test_reuse_saves_the_restart_matvecs(mode, reference) -> None:
    """``reuse`` skips the H x of each restart; it costs one confirming matvec at the end."""
    solver = make_torch_sci_solver(mode=mode, davidson_restart="reuse", **KW)
    _, perf = _solve(solver, mode)
    assert perf["n_matvecs"] < reference[1]


@pytest.mark.parametrize("correction", ["davidson", "olsen"])
def test_multiroot_correction_matches_default(mode, correction) -> None:
    """Olsen in the multi-root path finds the same three lowest roots."""
    base = make_torch_sci_solver(mode=mode, n_roots=3, **{**KW, "davidson_block_size": 12})
    want, _ = _solve(base, mode)
    solver = make_torch_sci_solver(
        mode=mode, n_roots=3, davidson_correction=correction, **{**KW, "davidson_block_size": 12}
    )
    got, _ = _solve(solver, mode)
    for g, w in zip(got, want):
        assert g.energy == pytest.approx(w.energy, abs=1e-8)


def test_per_root_start_vectors_converge_fast(mode) -> None:
    """Every converged root passed back: all roots match, in fewer matvecs than cold."""
    solver = make_torch_sci_solver(mode=mode, n_roots=3, **{**KW, "davidson_block_size": 12})
    cold, perf_cold = _solve(solver, mode)
    rows = np.stack([np.asarray(s.sci_state.amplitudes).ravel() for s in cold])
    warm, perf = _solve(solver, mode, init_vectors=[rows])
    assert perf["warm"] is True
    assert perf["n_matvecs"] < perf_cold["n_matvecs"]
    for c, w in zip(cold, warm):
        assert w.energy == pytest.approx(c.energy, abs=1e-8)


# A closed-shell N2/6-31G CAS(10e,10o) SQD batch (40 alpha = beta strings, 1600 determinants),
# with the integrals stored in fixtures/n2_cas10_sqd_batch.npz. Exact lowest roots there:
# -28.748943 (singlet), -28.642860 (triplet), -28.330502 (triplet).
N2_BATCH = np.array(
    [
        31,
        87,
        115,
        117,
        122,
        157,
        173,
        188,
        206,
        218,
        227,
        233,
        242,
        244,
        248,
        301,
        372,
        403,
        405,
        410,
        419,
        453,
        465,
        466,
        496,
        535,
        551,
        563,
        565,
        569,
        570,
        590,
        598,
        613,
        618,
        707,
        724,
        781,
        932,
        962,
    ],
    dtype=np.int64,
)


def _n2_cas():
    """The stored N2 CAS(10e,10o) integrals.

    Stored rather than rebuilt: N2 has three pairs of degenerate MOs, whose rotation PySCF
    leaves to the BLAS build, and a 40-string batch is not invariant under it (its energies
    move by up to 53 mHa), so a rebuilt Hamiltonian differs between machines. Made with
    ``mcscf.CASCI(scf.RHF(N 0 0 0; N 0 0 1.4, 6-31g), 10, 10)``; ``eri8`` is 8-fold reduced.
    """
    data = np.load(Path(__file__).parent / "fixtures" / "n2_cas10_sqd_batch.npz")
    return data["h1"], ao2mo.restore(1, data["eri8"], 10)


@pytest.mark.parametrize("check", ["auto", False])
def test_warm_ground_state_still_finds_the_triplet(check) -> None:
    """Warm from the exact singlet ground state, the lowest triplet must still be found.

    Without the missed-root check the solve converges to the second singlet (-28.1966) for
    root 1 instead of the triplet (-28.6429), every residual small: the warm seed never builds
    a direction with triplet character. ``"auto"`` checks a warm multi-root solve.
    """
    h1, eri = _n2_cas()
    kw = dict(
        device="cpu",
        davidson_block_size=20,
        max_davidson_iters=20,
        davidson_tolerance=1e-8,
        apply_ab_projection=False,
        n_roots=2,
    )
    solver = make_torch_sci_solver(davidson_missed_root_check=check, **kw)
    batch = [(N2_BATCH, N2_BATCH)]
    cold = solver(batch, h1, eri, 10, (5, 5))[0].states
    warm = solver(
        batch, h1, eri, 10, (5, 5), init_vectors=[np.asarray(cold[0].sci_state.amplitudes).ravel()]
    )[0].states
    want = [s.energy for s in cold]
    np.testing.assert_allclose(want, [-28.748943, -28.642860], atol=1e-6)
    got = [s.energy for s in warm]
    if check == "auto":
        np.testing.assert_allclose(got, want, atol=1e-8)
    else:  # documents the failure the check exists for: root 1 is the second singlet
        assert got[0] == pytest.approx(want[0], abs=1e-8)
        assert got[1] > want[1] + 0.1


def test_per_root_start_does_not_skip_a_root() -> None:
    """Start vectors all in one alpha-beta swap sector must not hide the other sector's roots.

    H6 full CI without the alpha-beta projection (``apply_ab_projection=False``, the
    sci_solvers default): roots 1 and 2 are swap-odd, roots 0, 3 and 4 swap-even. Seeding
    every root from the even ones returned [E0, E3, E4] as converged in 3 matvecs (Davidson
    never left the even sector); the missed-root check brings back the lowest three.
    """
    kw = dict(
        device="cpu", davidson_block_size=16, max_davidson_iters=200, apply_ab_projection=False
    )
    exact = make_torch_sci_solver(n_roots=8, davidson_tolerance=1e-10, **kw)
    roots = exact([(STRS, STRS)], H1, ERI, NORB, NELEC)[0].states
    parity = []
    for r in roots:
        c = np.asarray(r.sci_state.amplitudes).reshape(len(STRS), len(STRS))
        parity.append(np.sign(np.vdot(c, c.T).real))
    even = [i for i, p in enumerate(parity) if p > 0][:3]
    assert even != [0, 1, 2]  # the trap: some of the lowest three are odd
    rows = np.stack([np.asarray(roots[i].sci_state.amplitudes).ravel() for i in even])
    for tol in (1e-8, 1e-4):
        solver = make_torch_sci_solver(n_roots=3, davidson_tolerance=tol, **kw)
        got = solver([(STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[rows])[0].states
        for g, want in zip(got, roots[:3]):
            assert g.energy == pytest.approx(want.energy, abs=1e-6)


def test_missed_root_check_value_is_validated() -> None:
    """``davidson_missed_root_check`` takes "auto", True or False only."""
    with pytest.raises(ValueError, match="davidson_missed_root_check"):
        make_torch_sci_solver(davidson_missed_root_check="always")


def test_per_root_rows_must_not_exceed_n_roots(mode) -> None:
    """A ``(k, len)`` entry needs ``k <= n_roots``, and only when ``n_roots > 1``."""
    n = len(_batch(mode)[0]) * (len(STRS) if mode == "tpb" else 1)
    rows = np.ones((3, n))
    with pytest.raises(ValueError, match=r"init_vectors\[0\] has shape"):
        _solve(make_torch_sci_solver(mode=mode, n_roots=2, **KW), mode, init_vectors=[rows])
    with pytest.raises(ValueError, match=r"init_vectors\[0\] has shape"):
        _solve(make_torch_sci_solver(mode=mode, **KW), mode, init_vectors=[rows[:1]])


def test_spinless_variants_match_default() -> None:
    """The spinless path forwards both knobs and per-root rows (complex, 56 determinants)."""
    from pyscf.fci import cistring
    from test_spinless_davidson import _random_complex_hermitian_spinorbital_integrals

    nmo, n = 8, 3
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, 30)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    kw = dict(device="cpu", dtype=torch.complex128, davidson_block_size=12, davidson_tolerance=1e-8)
    want = make_torch_sci_solver(n_roots=2, **kw)([strs], hcore, eri, nmo, n)[0].states
    solver = make_torch_sci_solver(
        n_roots=2, davidson_correction="olsen", davidson_restart="reuse", **kw
    )
    rows = np.stack([np.asarray(s.sci_state.amplitudes).ravel() for s in want])
    for init in (None, [rows]):
        got = solver([strs], hcore, eri, nmo, n, init_vectors=init)[0].states
        for g, w in zip(got, want):
            assert g.energy == pytest.approx(w.energy, abs=1e-8)


@pytest.mark.parametrize(
    ("kwarg", "value"), [("davidson_correction", "jacobi"), ("davidson_restart", "keep")]
)
def test_invalid_values_raise(kwarg, value) -> None:
    """An unknown correction or restart mode is refused at construction."""
    with pytest.raises(ValueError, match=kwarg):
        make_torch_sci_solver(**{kwarg: value})


def test_olsen_correction_is_orthogonal_to_the_ritz_vector() -> None:
    """The Olsen step itself: the correction has no component along x."""
    from torch_sci_solver.torch_impl.davidson import _olsen

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(50, generator=gen, dtype=torch.float64)
    x = x / torch.linalg.norm(x)
    r = torch.randn(50, generator=gen, dtype=torch.float64)
    denom = -1.0 - torch.rand(50, generator=gen, dtype=torch.float64)
    delta = _olsen(r / denom, x, denom, torch.finfo(torch.float64).tiny)
    assert abs(float(torch.dot(x, delta))) < 1e-12


def test_davidson_torch_rejects_2d_start_for_single_root() -> None:
    """A per-root block is a multi-root input."""
    hii = torch.arange(10, dtype=torch.float64)
    with pytest.raises(ValueError, match="1-D"):
        davidson_torch(hii, torch.ones(2, 10), lambda v: hii * v, 5, 4, 1e-8)


# --------------------------------------------------------------------------- #
# Breakdown: a zero correction column must never become the Ritz vector       #
# --------------------------------------------------------------------------- #
def _isolated_spinless():
    """A positive-spectrum spinless Hamiltonian of mostly isolated determinants.

    Every determinant but the HF block shares <= 1 mode with all others, so the
    corrections collapse onto the subspace (breakdown) before a tight tolerance is
    reached. Before the fix, the zero column a breakdown leaves was diagonalized as
    an exact Ritz pair of value 0, below every root here (E0 = 11.8), and returned.
    """
    from test_spinless_multilimb_davidson_seed import (
        N_PARTICLES,
        NMO,
        STRS,
        _small_random_complex_eri,
    )
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
    from torch_sci_solver.torch_impl.mult_spinless import (
        make_diag_terms_spinless_torch,
        mult_spinless_torch,
    )
    from torch_sci_solver.torch_impl.spinless_helpers import (
        basis_init_vector_spinless_torch,
        generate_spinless_helpers,
        spinless_basis_from_array,
    )

    hcore = np.diag(2.0 * np.arange(NMO, dtype=np.float64))
    eri = _small_random_complex_eri(NMO, seed=99, scale=0.01)
    basis = spinless_basis_from_array(STRS, NMO, N_PARTICLES)
    helpers = generate_spinless_helpers(basis)
    ints = to_torch_integrals_spinless(hcore, eri, dtype=torch.complex128)
    hii = make_diag_terms_spinless_torch(basis, ints)

    def matvec(v):
        return mult_spinless_torch(hii, v, basis, helpers, ints)

    dense = matvec(torch.eye(basis.dim, dtype=torch.complex128)).numpy()
    e_ref = np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))
    seed = int(torch.argmin(hii))
    W = basis_init_vector_spinless_torch(basis.dim, init=0, seed_index=seed, dtype=torch.complex128)
    return hii, W, matvec, e_ref


def test_single_root_breakdown_does_not_return_zero() -> None:
    """A tight tolerance forces breakdown; the ground state, not 0.0, comes back."""
    hii, W, matvec, e_ref = _isolated_spinless()
    assert e_ref[0] > 1.0  # the spectrum lies above 0, so a zero Ritz pair would win
    energy, _, _ = davidson_torch(hii, W, matvec, 20, 8, 1e-14)
    assert energy == pytest.approx(e_ref[0], abs=1e-8)


@pytest.mark.parametrize("block", [4, 6, 8])
def test_multiroot_breakdown_keeps_the_true_roots(block) -> None:
    """Multi-root used to crash in eigh, blow up to -1e14 or return [0, 0] here."""
    hii, W, matvec, e_ref = _isolated_spinless()
    energies, _, _ = davidson_torch(hii, W, matvec, 20, block, 1e-8, n_roots=2)
    np.testing.assert_allclose(energies.numpy(), e_ref[:2], atol=1e-7)


def _stagnant(pairs: int, n: int = 14):
    """A Hamiltonian on which the Davidson correction collapses at every step.

    ``H = D + V`` with ``D = diag(H)`` and ``V x_k = 0`` for ``x_k = (e_2k + e_2k+1)/sqrt2``.
    From ``span{x_k}``, ``(H - theta) x = (D - theta) x``, so the correction
    ``(D - theta)^-1 r`` lies in that span again and is removed by Gram-Schmidt: the
    textbook Davidson stagnation, exactly. Returns ``hii``, the ``(pairs, n)`` start rows,
    a matvec and the exact eigenvalues.
    """
    rng = np.random.default_rng(0)
    m = 2 * pairs
    H = np.diag(np.arange(n, dtype=np.float64))
    B = rng.normal(scale=0.3, size=(n - m, n - m))
    B = B + B.T
    np.fill_diagonal(B, 0.0)
    H[m:, m:] += B
    for k in range(pairs):  # the rest of the space couples to e_2k - e_2k+1 only
        c = rng.normal(scale=0.3, size=n - m)
        H[m:, 2 * k] += c
        H[m:, 2 * k + 1] -= c
        H[2 * k, m:] += c
        H[2 * k + 1, m:] -= c
    X = np.zeros((pairs, n))
    for k in range(pairs):
        X[k, 2 * k : 2 * k + 2] = 1.0 / np.sqrt(2.0)
    H_t = torch.tensor(H)
    return (
        torch.tensor(np.diag(H).copy()),
        torch.tensor(X),
        lambda v: H_t @ v,
        np.linalg.eigvalsh(H),
    )


def test_single_root_stagnant_correction_falls_back_to_the_residual() -> None:
    """Before the fallback, every restart repeated the collapsed step: E stayed at 0.5."""
    hii, X, matvec, e_ref = _stagnant(pairs=1)
    energy, _, info = davidson_torch(hii, X[0], matvec, 30, 6, 1e-10)
    assert info["converged"] is True
    assert energy == pytest.approx(e_ref[0], abs=1e-10)


@pytest.mark.parametrize("block", [6, 8])
def test_multiroot_stagnant_corrections_fall_back_to_the_residual(block) -> None:
    """Both roots' corrections collapse; the solve must still move and find both.

    This is the failure seen on an x86 MKL host in the fp32 H6 2-root solve
    (test_davidson_dtype.py::test_fp32_subspace_multiroot): one correction per restart
    collapsed, the restart repeated it until max_davidson_iters, and the unconverged
    root 0 came back 7.6 mHa high. Here it is exact, so it does not depend on rounding:
    before the fallback this returned the start values [0.5, 2.5] after 2 matvecs.
    """
    hii, X, matvec, e_ref = _stagnant(pairs=2)
    energies, _, info = davidson_torch(hii, X, matvec, 30, block, 1e-10, n_roots=2)
    assert info["converged"] is True
    np.testing.assert_allclose(energies.numpy(), e_ref[:2], atol=1e-10)


def test_olsen_converges_where_the_plain_multiroot_correction_stalls() -> None:
    """The excited root's plain correction collapses at ~4e-8; Olsen's reaches 1e-8."""
    hii, W, matvec, e_ref = _isolated_spinless()
    _, _, plain = davidson_torch(hii, W, matvec, 20, 8, 1e-8, n_roots=2)
    energies, _, olsen = davidson_torch(hii, W, matvec, 20, 8, 1e-8, n_roots=2, correction="olsen")
    assert plain["converged"] is False
    assert olsen["converged"] is True
    np.testing.assert_allclose(energies.numpy(), e_ref[:2], atol=1e-8)
