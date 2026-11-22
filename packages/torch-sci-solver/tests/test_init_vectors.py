"""Tests for the per-call ``init_vectors`` Davidson start vector.

Every test uses H6/STO-3G full CI (20 alpha x 20 beta = 400 determinants) with
``davidson_block_size=8``, so Davidson runs its Krylov loop rather than the dense
small-basis bypass, which never reads the start vector. ``tol=1e-8`` makes the
matvec count a clean signal: a cold solve takes tens of matvecs, an exact start
vector converges on the first residual check (1 matvec).

Covers, in product (TPB) and pair (GDB) mode:
  1. exact vector on an identical basis converges on the first residual check;
  2. a basis that loses and gains strings (projected vector): same energy as
     cold, fewer matvecs;
  3. zero-overlap vector falls back to the cold start;
  4. ``seed_strategy="connected"`` gives the same result as ``"smallest_int"``;
  5. multi-GPU (``devices=[...]``) routes each vector to its subspace;
  6. wrong length raises ``ValueError``.
Plus: shuffled and duplicated caller strings, the alpha-beta projection erasing
an antisymmetric vector, multi-root, ``spin_sq``, spinless and complex dtypes,
and the ``probe_state`` exclusion.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf import ao2mo, gto, scf  # noqa: E402
from torch_sci_solver.multi_gpu import run_in_spawned_process  # noqa: E402
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver  # noqa: E402

SOLVER_KWARGS = dict(
    device="cpu", davidson_block_size=8, max_davidson_iters=60, davidson_tolerance=1e-8
)


def _h6():
    mol = gto.M(atom="; ".join(f"H 0 0 {i * 1.0}" for i in range(6)), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return h1, eri, norb, mol.nelec


H1, ERI, NORB, NELEC = _h6()


def _strings(norb: int, n_elec: int) -> np.ndarray:
    return np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), n_elec)),
        dtype=np.int64,
    )


STRS = _strings(NORB, NELEC[0])  # alpha == beta (closed shell)


def _pairs(strs_a: np.ndarray, strs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The GDB pair list covering the full product of ``strs_a`` x ``strs_b``."""
    return np.repeat(strs_a, len(strs_b)), np.tile(strs_b, len(strs_a))


def _batch(mode: str, strs_a: np.ndarray, strs_b: np.ndarray):
    return (strs_a, strs_b) if mode == "tpb" else _pairs(strs_a, strs_b)


def _solve(solver, batches, **kwargs):
    """Solve and return (ground-state results, per-subspace perf dicts) of this call."""
    results = solver(batches, H1, ERI, NORB, NELEC, **kwargs)
    return [r.states[0] for r in results], solver.stats["per_call_log"][-1]["subspaces"]


def _project(amps: np.ndarray, old_a, old_b, new_a, new_b) -> np.ndarray:
    """Copy TPB amplitudes from the (old_a, old_b) grid onto (new_a, new_b); new = 0."""
    old = amps.reshape(len(old_a), len(old_b))
    out = np.zeros((len(new_a), len(new_b)), dtype=old.dtype)
    ia = {int(s): i for i, s in enumerate(old_a)}
    ib = {int(s): i for i, s in enumerate(old_b)}
    for i, s in enumerate(new_a):
        for j, t in enumerate(new_b):
            if int(s) in ia and int(t) in ib:
                out[i, j] = old[ia[int(s)], ib[int(t)]]
    return out.ravel()


def _project_pairs(amps: np.ndarray, old_pairs, new_pairs) -> np.ndarray:
    """Copy GDB pair amplitudes from ``old_pairs`` onto ``new_pairs``; new = 0."""
    k = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(*old_pairs))}
    return np.array([
        amps[k[(int(a), int(b))]] if (int(a), int(b)) in k else 0.0 for a, b in zip(*new_pairs)
    ])


@pytest.fixture(scope="module", params=["tpb", "gdb"])
def mode(request) -> str:
    """Basis layout under test: product (TPB) or pair (GDB)."""
    return request.param


@pytest.fixture(scope="module")
def cold(mode):
    """Cold full-CI solve: (solver, ground-state result, n_matvecs)."""
    solver = make_torch_sci_solver(mode=mode, **SOLVER_KWARGS)
    (gs,), (perf,) = _solve(solver, [_batch(mode, STRS, STRS)])
    assert perf["warm"] is False
    return solver, gs, perf["n_matvecs"]


# --------------------------------------------------------------------------- #
# 1. Exact vector on an identical basis                                        #
# --------------------------------------------------------------------------- #
def test_exact_vector_converges_on_first_check(mode, cold) -> None:
    """The converged state passed back is already an eigenvector: one matvec, same energy."""
    solver, gs, mv_cold = cold
    (warm,), (perf,) = _solve(
        solver, [_batch(mode, STRS, STRS)], init_vectors=[gs.sci_state.amplitudes.ravel()]
    )
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1 < mv_cold
    assert warm.energy == pytest.approx(gs.energy, abs=1e-9)


def test_shuffled_and_duplicated_caller_strings(mode, cold) -> None:
    """The vector follows the caller's string order; duplicates take the first entry."""
    solver, gs, _ = cold
    rng = np.random.default_rng(0)
    amps = gs.sci_state.amplitudes.ravel()
    if mode == "tpb":
        pa, pb = rng.permutation(len(STRS)), rng.permutation(len(STRS))
        strs_a = np.concatenate([STRS[pa], STRS[pa[:3]]])  # 3 duplicated alpha strings
        strs_b = STRS[pb]
        grid = amps.reshape(len(STRS), len(STRS))[np.ix_(pa, pb)]
        vec = np.concatenate([grid, 99.0 * np.ones((3, len(STRS)))]).ravel()  # dupes ignored
        batch = (strs_a, strs_b)
    else:
        pa, pb = _pairs(STRS, STRS)
        perm = rng.permutation(len(pa))
        batch = (np.concatenate([pa[perm], pa[perm[:5]]]), np.concatenate([pb[perm], pb[perm[:5]]]))
        vec = np.concatenate([amps[perm], 99.0 * np.ones(5)])
    (warm,), (perf,) = _solve(solver, [batch], init_vectors=[vec])
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert warm.energy == pytest.approx(gs.energy, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Basis that loses and gains strings                                        #
# --------------------------------------------------------------------------- #
def test_projected_vector_on_changed_basis(mode, cold) -> None:
    """The SQD use case: previous solve projected onto a grid that drops and adds strings.

    Run at the loose residual the issue pairs with warm starting (3e-3): a tight
    residual hides most of the saving on a basis this small, since the cold solve
    already converges in a few restarts.
    """
    solver, _, _ = cold
    old = STRS[:19]
    new = np.concatenate([STRS[:17], STRS[19:]])  # drops STRS[17:19], adds STRS[19]
    (prev,), _ = _solve(solver, [_batch(mode, old, old)])
    if mode == "tpb":
        vec = _project(prev.sci_state.amplitudes.ravel(), old, old, new, new)
    else:
        vec = _project_pairs(prev.sci_state.amplitudes, _pairs(old, old), _pairs(new, new))
    (exact,), _ = _solve(solver, [_batch(mode, new, new)])
    (cold_new,), (perf_cold,) = _solve(solver, [_batch(mode, new, new)], tol=3e-3)
    (warm_new,), (perf_warm,) = _solve(
        solver, [_batch(mode, new, new)], tol=3e-3, init_vectors=[vec]
    )
    assert perf_warm["warm"] is True
    assert warm_new.energy == pytest.approx(exact.energy, abs=1e-5)  # 0.01 mHa
    assert perf_warm["n_matvecs"] <= perf_cold["n_matvecs"] // 2


# --------------------------------------------------------------------------- #
# 3. Zero overlap                                                              #
# --------------------------------------------------------------------------- #
def test_zero_vector_falls_back_to_cold(mode, cold) -> None:
    """No overlap with the basis: bit-identical to the cold solve, recorded as not warm."""
    solver, gs, mv_cold = cold
    n = len(STRS) ** 2
    (res,), (perf,) = _solve(solver, [_batch(mode, STRS, STRS)], init_vectors=[np.zeros(n)])
    assert perf["warm"] is False
    assert perf["n_matvecs"] == mv_cold
    assert res.energy == gs.energy


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_overall_scale_does_not_cause_a_cold_start(mode, cold, scale) -> None:
    """The fallback is relative to the caller's own norm, so scale is irrelevant.

    An absolute 1e-8 cut-off cold-started a correctly shaped vector scaled by 1e-12;
    a vector that genuinely has no overlap still falls back (the zero-vector test).
    """
    solver, gs, _ = cold
    vec = gs.sci_state.amplitudes.ravel() * scale
    (res,), (perf,) = _solve(solver, [_batch(mode, STRS, STRS)], init_vectors=[vec])
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert res.energy == pytest.approx(gs.energy, abs=1e-9)


def test_ab_projection_erasing_vector_falls_back(mode) -> None:
    """An alpha-beta-antisymmetric vector projects to 0; it must fall back, not give E = 0."""
    solver = make_torch_sci_solver(mode=mode, apply_ab_projection=True, **SOLVER_KWARGS)
    (gs,), _ = _solve(solver, [_batch(mode, STRS, STRS)])
    # The projector keeps the transpose-symmetric spin-block part (the closed-shell
    # singlet sector), so the transpose-antisymmetric part is erased entirely.
    rng = np.random.default_rng(1)
    m = rng.standard_normal((len(STRS), len(STRS)))
    block = m - m.T
    if mode == "tpb":
        vec = block.ravel()
    else:
        from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

        pa, pb = _pairs(STRS, STRS)
        sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), NORB).numpy()
        vec = block.ravel() * sigma  # GDB takes interleaved-Wick signs
    (res,), (perf,) = _solve(solver, [_batch(mode, STRS, STRS)], init_vectors=[vec])
    assert perf["warm"] is False
    assert res.energy == pytest.approx(gs.energy, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. seed_strategy="connected"                                                 #
# --------------------------------------------------------------------------- #
def test_connected_seed_strategy_matches_smallest_int(mode, cold) -> None:
    """The connected reorder changes the internal order; the vector must follow it."""
    solver, gs, _ = cold
    vec = gs.sci_state.amplitudes.ravel()
    connected = make_torch_sci_solver(mode=mode, seed_strategy="connected", **SOLVER_KWARGS)
    (res,), (perf,) = _solve(connected, [_batch(mode, STRS, STRS)], init_vectors=[vec])
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert res.energy == pytest.approx(gs.energy, abs=1e-9)


# --------------------------------------------------------------------------- #
# 6. Validation                                                                #
# --------------------------------------------------------------------------- #
def test_wrong_vector_length_raises(mode, cold) -> None:
    """A vector for a different basis is a caller bug, not something to pad or truncate."""
    solver, _, _ = cold
    with pytest.raises(ValueError, match=r"init_vectors\[0\] has shape"):
        solver([_batch(mode, STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[np.ones(7)])


def test_wrong_list_length_raises(mode, cold) -> None:
    """One entry per subspace; a mismatch would shift every vector onto the wrong basis."""
    solver, _, _ = cold
    with pytest.raises(ValueError, match="init_vectors has length 2"):
        solver([_batch(mode, STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[None, None])


def test_init_vectors_and_probe_state_are_exclusive(mode, cold) -> None:
    """Both set the start vector, so accepting both would silently drop one."""
    solver, _, _ = cold
    with pytest.raises(ValueError, match="mutually exclusive"):
        solver(
            [_batch(mode, STRS, STRS)],
            H1,
            ERI,
            NORB,
            NELEC,
            init_vectors=[None],
            probe_method="warm_start",
            probe_state=[object()],
        )


def test_complex_vector_on_real_solver_raises(mode, cold) -> None:
    """A real solver must not drop the imaginary part of a complex start vector."""
    solver, _, _ = cold
    vec = np.ones(len(STRS) ** 2, dtype=np.complex128)
    with pytest.raises(TypeError, match="is complex"):
        solver([_batch(mode, STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[vec])


def test_none_entries_mix_warm_and_cold(mode, cold) -> None:
    """A ``None`` entry cold starts its own subspace only."""
    solver, gs, mv_cold = cold
    batch = _batch(mode, STRS, STRS)
    results, perfs = _solve(
        solver, [batch, batch], init_vectors=[None, gs.sci_state.amplitudes.ravel()]
    )
    assert [p["warm"] for p in perfs] == [False, True]
    assert [p["n_matvecs"] for p in perfs] == [mv_cold, 1]
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-9)


# --------------------------------------------------------------------------- #
# Variants: multi-root, spin_sq                                                #
# --------------------------------------------------------------------------- #
def test_multiroot_seeds_root_zero(mode) -> None:
    """With ``n_roots > 1`` a 1-D vector seeds root 0; every root still converges.

    The missed-root check is off here, so this measures the seeding alone: with it on
    (the default for a warm multi-root start) the search costs more matvecs than this
    400-determinant solve saves. ``test_davidson_variants.py`` covers the checked path.
    """
    solver = make_torch_sci_solver(
        mode=mode, n_roots=2, davidson_missed_root_check=False, **SOLVER_KWARGS
    )
    batch = _batch(mode, STRS, STRS)
    cold_res = solver([batch], H1, ERI, NORB, NELEC)[0].states
    mv_cold = solver.stats["per_call_log"][-1]["subspaces"][0]["n_matvecs"]
    warm_res = solver(
        [batch], H1, ERI, NORB, NELEC, init_vectors=[cold_res[0].sci_state.amplitudes.ravel()]
    )[0].states
    perf = solver.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["warm"] is True
    assert perf["n_matvecs"] < mv_cold
    for c, w in zip(cold_res, warm_res):
        assert w.energy == pytest.approx(c.energy, abs=1e-7)


def test_spin_sq_penalty(mode, cold) -> None:
    """Warm start under the S^2 penalty: the start vector seeds H + lambda (S^2 - target)."""
    _, gs, _ = cold
    solver = make_torch_sci_solver(mode=mode, **SOLVER_KWARGS)
    batch = _batch(mode, STRS, STRS)
    (_,), (perf_cold,) = _solve(solver, [batch], spin_sq=0.0)
    (res,), (perf,) = _solve(
        solver, [batch], spin_sq=0.0, init_vectors=[gs.sci_state.amplitudes.ravel()]
    )
    assert perf["warm"] is True
    # Not exactly 1: round-off in the S^2 term leaves the exact H eigenvector a
    # residual just above tol, so Davidson takes one more step.
    assert perf["n_matvecs"] <= 3 < perf_cold["n_matvecs"]
    assert res.energy == pytest.approx(gs.energy, abs=1e-9)


# --------------------------------------------------------------------------- #
# Complex dtype (spatial) and the spinless path                                #
# --------------------------------------------------------------------------- #
def test_complex_dtype_spatial(mode) -> None:
    """Complex amplitudes (any global phase) on the complex-dtype path."""
    solver = make_torch_sci_solver(mode=mode, dtype=torch.complex128, **SOLVER_KWARGS)
    batch = _batch(mode, STRS, STRS)
    h1, eri = H1.astype(np.complex128), ERI.astype(np.complex128)
    cold_gs = solver([batch], h1, eri, NORB, NELEC)[0].states[0]
    vec = cold_gs.sci_state.amplitudes.ravel() * np.exp(0.3j)  # global phase is irrelevant
    warm_gs = solver([batch], h1, eri, NORB, NELEC, init_vectors=[vec])[0].states[0]
    perf = solver.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert warm_gs.energy == pytest.approx(cold_gs.energy, abs=1e-9)


def _spinless_problem():
    """Complex Hermitian spin-orbital integrals on nmo=8, N=3 (56 determinants)."""
    nmo, n = 8, 3
    rng = np.random.default_rng(3)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(nmo + 2):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri, nmo, n, _strings(nmo, n)


def test_spinless_path() -> None:
    """Scalar ``nelec``: one amplitude per caller string, shuffled and duplicated."""
    hcore, eri, nmo, n, strs = _spinless_problem()
    solver = make_torch_sci_solver(dtype=torch.complex128, **SOLVER_KWARGS)
    cold_gs = solver([strs], hcore, eri, nmo, n)[0].states[0]
    # The engine sorts its basis; the caller passes the strings shuffled + one duplicate.
    perm = np.random.default_rng(4).permutation(len(strs))
    order = {int(s): i for i, s in enumerate(cold_gs.sci_state.ci_strs)}
    amps = np.array([cold_gs.sci_state.amplitudes[order[int(s)]] for s in strs[perm]])
    caller = np.concatenate([strs[perm], strs[perm[:1]]])
    vec = np.concatenate([amps, [5.0]])
    warm_gs = solver([caller], hcore, eri, nmo, n, init_vectors=[vec])[0].states[0]
    assert warm_gs.energy == pytest.approx(cold_gs.energy, abs=1e-9)
    with pytest.raises(ValueError, match=r"init_vectors\[0\] has shape"):
        solver([strs], hcore, eri, nmo, n, init_vectors=[np.ones(3)])


def test_spinless_init_vectors_and_probe_state_are_exclusive() -> None:
    """Both set the spinless Davidson start vector, so one must be dropped."""
    hcore, eri, nmo, n, strs = _spinless_problem()
    solver = make_torch_sci_solver(dtype=torch.complex128, **SOLVER_KWARGS)
    with pytest.raises(ValueError, match="mutually exclusive"):
        solver(
            [strs],
            hcore,
            eri,
            nmo,
            n,
            init_vectors=[np.ones(len(strs))],
            probe_method="warm_start",
            probe_state=[object()],
        )


# --------------------------------------------------------------------------- #
# 5. Multi-GPU                                                                 #
# --------------------------------------------------------------------------- #
def _multi_gpu_body(mode: str):
    """Child process: two distinct subspaces, only the second warm started."""
    solver = make_torch_sci_solver(
        devices=["cuda:0", "cuda:1"],
        mode=mode,
        **{k: v for k, v in SOLVER_KWARGS.items() if k != "device"},
    )
    small = STRS[:16]
    batches = [_batch(mode, small, small), _batch(mode, STRS, STRS)]
    cold_res = solver(batches, H1, ERI, NORB, NELEC)
    vec = cold_res[1].states[0].sci_state.amplitudes.ravel()
    warm_res = solver(batches, H1, ERI, NORB, NELEC, init_vectors=[None, vec])
    perfs = solver.stats["per_call_log"][-1]["subspaces"]
    out = (
        [r.states[0].energy for r in cold_res],
        [r.states[0].energy for r in warm_res],
        [p["warm"] for p in perfs],
        [p["n_matvecs"] for p in perfs],
    )
    solver.close()
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-GPU tests require >=2 CUDA devices",
)
def test_multi_gpu_routes_vectors_to_their_subspace(mode) -> None:
    """Each vector reaches the worker that solves its subspace, not a neighbour."""
    cold_e, warm_e, warm_flags, matvecs = run_in_spawned_process(_multi_gpu_body, mode)
    assert warm_flags == [False, True]
    assert matvecs[1] == 1
    np.testing.assert_allclose(warm_e, cold_e, atol=1e-9)


# --------------------------------------------------------------------------- #
# Key conversion edges                                                         #
# --------------------------------------------------------------------------- #
def test_signed_int64_keys_matches_engine_mapping_for_every_input_type() -> None:
    """The caller's strings must map to the exact int64 the engine stores, whatever
    container they come in -- a Python list holding a value >= 2**63 would otherwise
    infer float64 and round the string's bits."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _signed_int64_keys
    from torch_sci_solver.torch_impl.sbdiag import _to_int64_python

    u = np.random.default_rng(1).integers(0, 2**64, size=1000, dtype=np.uint64)
    expected = np.asarray(_to_int64_python(u.tolist()), dtype=np.int64)
    for strs in (u, u.view(np.int64), u.astype(object), u.tolist()):
        np.testing.assert_array_equal(_signed_int64_keys(strs), expected)


@pytest.mark.parametrize("value", [1 << 64, -(1 << 63) - 1])
def test_signed_int64_keys_rejects_strings_outside_64_bits(value) -> None:
    """A string wider than 64 bits must raise, not wrap to a key that matches a
    different determinant (TPB and GDB support at most 64 spatial orbitals).

    A negative int down to ``-2**63`` is a signed int64 bit pattern (the form
    int64 arrays and the engine's cached pair states carry), so ``-1`` is the
    all-ones 64-bit string rather than an error."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _signed_int64_keys

    with pytest.raises(ValueError, match="64 bits"):
        _signed_int64_keys([value])
    assert _signed_int64_keys([-1]).view(np.uint64).tolist() == [(1 << 64) - 1]


def test_spinless_fallback_uses_the_argmin_seed_on_a_multilimb_basis() -> None:
    """A spinless warm start that falls back must seed at argmin(hii), not index 0.

    On this multi-limb basis (68 spin-orbitals) index 0 is an isolated high-energy
    determinant, so a Davidson seeded there converges to the wrong eigenpair. The
    warm path's fallback reuses the cold seed, so a no-overlap init_vectors entry
    must still land on the true ground state. Checked against the dense spectrum,
    not against the cold solve: a seed bug makes both equally wrong.
    """
    from test_spinless_multilimb_davidson_seed import N_PARTICLES as N_MULTI
    from test_spinless_multilimb_davidson_seed import NMO as NMO_MULTI
    from test_spinless_multilimb_davidson_seed import STRS as STRS_MULTI
    from test_spinless_multilimb_davidson_seed import _small_random_complex_eri
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
    from torch_sci_solver.torch_impl.mult_spinless import (
        make_diag_terms_spinless_torch,
        mult_spinless_torch,
    )
    from torch_sci_solver.torch_impl.spinless_helpers import (
        generate_spinless_helpers,
        spinless_basis_from_array,
    )

    hcore = np.diag(2.0 * np.arange(NMO_MULTI, dtype=np.float64))
    eri = _small_random_complex_eri(NMO_MULTI, seed=99, scale=0.01)

    # Dense reference, independent of any Davidson seed.
    basis = spinless_basis_from_array(STRS_MULTI, NMO_MULTI, N_MULTI)
    integrals = to_torch_integrals_spinless(hcore, eri, dtype=torch.complex128)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    eye = torch.eye(basis.dim, dtype=torch.complex128)
    dense = mult_spinless_torch(hii, eye, basis, generate_spinless_helpers(basis), integrals)
    dense = dense.numpy()
    e_ref = float(np.linalg.eigvalsh(0.5 * (dense + dense.conj().T))[0])

    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128)
    no_overlap = [np.zeros(len(STRS_MULTI), dtype=np.complex128)]
    fallback = solver([STRS_MULTI], hcore, eri, NMO_MULTI, N_MULTI, init_vectors=no_overlap)
    assert fallback[0].states[0].energy == pytest.approx(e_ref, abs=1e-6)
