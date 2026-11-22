"""Tests for the fp32 Davidson subspace over the fp64 matvec (``davidson_dtype``).

Every test uses H6/STO-3G full CI (400 determinants) with ``davidson_block_size=8``, so the
Krylov loop runs rather than the dense small-basis bypass (which stays fp64). The reference
is the default fp64 subspace at the same settings.

Covers, in product (TPB) and pair (GDB) mode:
  1. energy parity, cold and warm (``init_vectors``), at ``tol`` 1e-4 and 3e-3;
  2. ``davidson_fp32_shift=None`` still completes;
  3. returned dtypes, ``davidson_fp32_energy`` "rq64" vs "ritz", and "auto" switching
     to "ritz" at ``tol >= 1e-3``;
  4. ``spin_sq``, ``n_roots=2`` and ``apply_ab_projection``;
  5. the narrow complex64 subspace over complex128, and the spinless path's
     fp32/complex64 narrow subspaces;
  6. validation of the dtype combination;
  7. ``_pick_num_block`` sizes the auto block for the fp32 element size;
  8. the knobs reach the multi-GPU workers (worker config completeness).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest
import torch

pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf.fci import cistring  # noqa: E402
from test_init_vectors import (  # noqa: E402
    ERI,
    H1,
    NELEC,
    NORB,
    SOLVER_KWARGS,
    STRS,
    _batch,
)
from test_spinless_solver import (  # noqa: E402
    _random_complex_hermitian_spinorbital_integrals,
)
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver  # noqa: E402

FP32 = {"davidson_dtype": torch.float32}


def _real_spinless_integrals(nmo: int, seed: int):
    """Real Hermitian spin-orbital integrals for the flat spinless path."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo))
    hcore = hc + hc.T
    eri = np.zeros((nmo, nmo, nmo, nmo))
    for _ in range(nmo + 2):
        L = rng.standard_normal((nmo, nmo))
        L = L + L.T
        eri += np.einsum("pq,rs->pqrs", L, L)
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _solve(mode: str, n_roots: int = 1, **kwargs):
    """One call on the full H6 grid: (EigenStates.states, perf dict of the subspace)."""
    call = {k: kwargs.pop(k) for k in ("tol", "init_vectors", "spin_sq") if k in kwargs}
    solver = make_torch_sci_solver(mode=mode, n_roots=n_roots, **SOLVER_KWARGS, **kwargs)
    states = solver([_batch(mode, STRS, STRS)], H1, ERI, NORB, NELEC, **call)[0].states
    return states, solver.stats["per_call_log"][-1]["subspaces"][0]


def _solve_on(mode: str, h1, eri, n_roots: int = 1, **kwargs):
    """Like :func:`_solve`, on a caller-supplied integral set."""
    call = {k: kwargs.pop(k) for k in ("tol", "init_vectors", "spin_sq") if k in kwargs}
    solver = make_torch_sci_solver(mode=mode, n_roots=n_roots, **SOLVER_KWARGS, **kwargs)
    states = solver([_batch(mode, STRS, STRS)], h1, eri, NORB, NELEC, **call)[0].states
    return states, solver.stats["per_call_log"][-1]["subspaces"][0]


@pytest.fixture(scope="module", params=["tpb", "gdb"])
def mode(request) -> str:
    """Basis layout under test: product (TPB) or pair (GDB)."""
    return request.param


# --------------------------------------------------------------------------- #
# 1. Energy parity                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tol", [1e-4, 3e-3])
def test_fp32_subspace_matches_fp64_energy(mode, tol) -> None:
    """The default fp32 subspace matches fp64 to well below the residual's own error.

    At ``tol=1e-4`` "auto" resolves to "rq64": the fp64 Rayleigh quotient matches to
    1e-8 Ha for one extra matvec. At 3e-3 it resolves to "ritz": no extra matvec, and
    the fp32 Ritz error (~4e-7 Ha here) is below what fp64 itself is off by at that
    tol (2.9e-7 Ha), so both stay within 1e-6 Ha of the exact energy.
    """
    (exact,), _ = _solve(mode)
    (ref,), perf64 = _solve(mode, tol=tol)
    (got,), perf32 = _solve(mode, tol=tol, **FP32)
    assert perf32["davidson_dtype"] == "torch.float32"
    assert perf32["davidson_fp32_shift"] == pytest.approx(-7.7394, abs=1e-4)  # min(hii), HF-like
    if tol < 1e-3:
        assert perf32["davidson_fp32_energy"] == "rq64"
        assert got.energy == pytest.approx(ref.energy, abs=1e-8)
        extra = 1  # the fp64 Rayleigh quotient
    else:
        assert perf32["davidson_fp32_energy"] == "ritz"
        assert got.energy == pytest.approx(exact.energy, abs=1e-6)
        extra = 0
    # +-1: fp32 rounding may shift one restart decision.
    assert abs(perf32["n_matvecs"] - (perf64["n_matvecs"] + extra)) <= 1


@pytest.mark.parametrize("tol", [1e-4, 3e-3])
def test_fp32_subspace_warm_start(mode, tol) -> None:
    """``init_vectors`` and the fp32 subspace compose: the warm vector is cast once."""
    (exact,), _ = _solve(mode)
    vec = exact.sci_state.amplitudes.ravel()
    (ref,), perf64 = _solve(mode, tol=tol, init_vectors=[vec])
    (got,), perf32 = _solve(mode, tol=tol, init_vectors=[vec], **FP32)
    assert perf32["warm"] is True
    assert got.energy == pytest.approx(exact.energy, abs=1e-6)
    assert perf32["n_matvecs"] <= perf64["n_matvecs"] + 2


# --------------------------------------------------------------------------- #
# 2. Without the shift                                                         #
# --------------------------------------------------------------------------- #
def test_fp32_subspace_without_shift_completes(mode) -> None:
    """``davidson_fp32_shift=None`` is documented as unsafe, but it must still run."""
    (ref,), _ = _solve(mode, tol=1e-4)
    (got,), perf = _solve(mode, tol=1e-4, davidson_fp32_shift=None, **FP32)
    assert perf["davidson_fp32_shift"] == 0.0
    assert got.energy == pytest.approx(ref.energy, abs=1e-6)


# --------------------------------------------------------------------------- #
# 3. Returned dtypes and davidson_fp32_energy                                       #
# --------------------------------------------------------------------------- #
def test_fp32_subspace_returns_fp64(mode) -> None:
    """The vector and everything built from it stay fp64, so callers see no change."""
    (got,), _ = _solve(mode, tol=1e-4, compute_full_rdm=True, **FP32)
    assert got.sci_state.amplitudes.dtype == np.float64
    assert all(o.dtype == np.float64 for o in got.orbital_occupancies)
    assert got.rdm1.dtype == np.float64 and got.rdm2.dtype == np.float64
    assert np.linalg.norm(got.sci_state.amplitudes) == pytest.approx(1.0, abs=1e-12)


def test_ritz_energy_saves_one_matvec(mode) -> None:
    """ "ritz" skips the fp64 RQ; its energy is that of an fp32-rounded vector.

    The bound is 1e-6 Ha, not tighter: rounding the *exact* eigenvector to fp32 moves
    its Rayleigh quotient by ~3e-7 Ha on this Hamiltonian (diagonal spread ~4.4 Ha
    around a ~-0.1 Ha shifted Ritz value), independent of tol. "rq64" removes it.
    """
    (exact,), _ = _solve(mode)
    (rq,), perf_rq = _solve(mode, tol=1e-6, davidson_fp32_energy="rq64", **FP32)
    (ritz,), perf_ritz = _solve(mode, tol=1e-6, davidson_fp32_energy="ritz", **FP32)
    assert rq.energy == pytest.approx(exact.energy, abs=1e-10)
    assert ritz.energy == pytest.approx(exact.energy, abs=1e-6)
    assert perf_ritz["n_matvecs"] == perf_rq["n_matvecs"] - 1


@pytest.mark.parametrize(
    "tol, expected",
    [(1e-2, "ritz"), (3e-3, "ritz"), (1e-3, "ritz"), (9e-4, "rq64"), (1e-6, "rq64")],
)
def test_auto_energy_follows_the_call_tol(mode, tol, expected) -> None:
    """ "auto" (the default) picks per call: "ritz" at tol >= 1e-3, "rq64" tighter.

    The same solver can serve the loose SQD warm loop and a tight final solve, so the
    choice follows the per-call ``tol``, not a construction-time setting.
    """
    (_,), perf = _solve(mode, tol=tol, **FP32)
    assert perf["davidson_fp32_energy"] == expected


def test_auto_energy_is_none_without_fp32_subspace(mode) -> None:
    """The default fp64 subspace never resolves an fp32 energy mode."""
    (_,), perf = _solve(mode, tol=3e-3)
    assert perf["davidson_fp32_energy"] is None


# --------------------------------------------------------------------------- #
# 4. Variants: spin_sq, multi-root, alpha-beta projection                      #
# --------------------------------------------------------------------------- #
def test_fp32_subspace_spin_sq(mode) -> None:
    """The penalty strip still applies: the RQ is of the operator Davidson saw."""
    (ref,), _ = _solve(mode, tol=1e-6, spin_sq=0.0)
    (got,), _ = _solve(mode, tol=1e-6, spin_sq=0.0, **FP32)
    assert got.energy == pytest.approx(ref.energy, abs=1e-8)


@pytest.mark.parametrize("apply_ab_projection", [True, False])
def test_fp32_subspace_multiroot(mode, apply_ab_projection) -> None:
    """Each root gets its own fp64 RQ, with and without the alpha-beta projection.

    With the projection (the production default) both precisions must return the two
    lowest singlets, E0 and E3; before the thick-restart buffer, multi-root Davidson
    returned E4 for root 1 on this grid, which is pinned by test_n_roots.py.
    """
    ref, _ = _solve(mode, n_roots=2, tol=1e-6, apply_ab_projection=apply_ab_projection)
    got, perf = _solve(mode, n_roots=2, tol=1e-6, apply_ab_projection=apply_ab_projection, **FP32)
    assert perf["davidson_dtype"] == "torch.float32"
    for r, g in zip(ref, got):
        assert g.energy == pytest.approx(r.energy, abs=1e-8)
        assert g.sci_state.amplitudes.dtype == np.float64
    if apply_ab_projection:
        assert got[1].energy == pytest.approx(-7.448971, abs=1e-6)  # E3, not E4


def test_fp32_subspace_ab_projection(mode) -> None:
    """The projector (built in fp64) acts on the fp32 vectors through type promotion."""
    (ref,), _ = _solve(mode, tol=1e-6, apply_ab_projection=True)
    (got,), _ = _solve(mode, tol=1e-6, apply_ab_projection=True, **FP32)
    assert got.energy == pytest.approx(ref.energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# 5-6. Spinless and validation                                                 #
# --------------------------------------------------------------------------- #
def test_complex64_subspace_is_accepted() -> None:
    """A complex64 subspace over complex128 is the complex analogue of fp32 over fp64."""
    solver = make_torch_sci_solver(
        device="cpu", dtype=torch.complex128, davidson_dtype=torch.complex64
    )
    assert solver.davidson_dtype is torch.complex64


def test_complex64_subspace_matches_complex128(mode) -> None:
    """The complex64 subspace keeps the full-precision matvec and shifts like fp32."""
    complex_h1 = H1.astype(np.complex128)
    complex_eri = ERI.astype(np.complex128)
    (ref,), _ = _solve_on(mode, complex_h1, complex_eri, dtype=torch.complex128, tol=1e-8)
    (got,), perf = _solve_on(
        mode,
        complex_h1,
        complex_eri,
        dtype=torch.complex128,
        davidson_dtype=torch.complex64,
        tol=1e-4,
    )
    assert perf["davidson_dtype"] == "torch.complex64"
    assert perf["davidson_fp32_shift"] == pytest.approx(-7.7394, abs=1e-4)
    assert got.energy == pytest.approx(ref.energy, abs=1e-6)
    assert np.iscomplexobj(got.sci_state.amplitudes)
    assert got.sci_state.amplitudes.dtype == np.complex128


def test_spinless_fp32_subspace_matches_fp64() -> None:
    """The narrow subspace also reaches the spinless path (real integrals, fp64 dtype)."""
    nmo, n, seed = 5, 2, 42
    hcore, eri = _real_spinless_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    common = dict(device="cpu", davidson_block_size=6, max_davidson_iters=100)
    ref = make_torch_sci_solver(davidson_tolerance=1e-8, **common)
    e_ref = ref([strs], hcore, eri, nmo, n)[0].states[0]
    got = make_torch_sci_solver(davidson_dtype=torch.float32, davidson_tolerance=1e-4, **common)
    state = got([strs], hcore, eri, nmo, n)[0].states[0]
    assert state.energy == pytest.approx(float(e_ref.energy), abs=1e-6)
    assert state.sci_state.amplitudes.dtype == np.float64


def test_spinless_complex64_subspace_matches_complex128() -> None:
    """The spinless path's complex64 subspace keeps the complex128 result."""
    nmo, n, seed = 5, 2, 42
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    common = dict(
        device="cpu", dtype=torch.complex128, davidson_block_size=6, max_davidson_iters=100
    )
    ref = make_torch_sci_solver(davidson_tolerance=1e-8, **common)
    e_ref = ref([strs], hcore, eri, nmo, n)[0].states[0]
    got = make_torch_sci_solver(davidson_dtype=torch.complex64, davidson_tolerance=1e-4, **common)
    state = got([strs], hcore, eri, nmo, n)[0].states[0]
    assert state.energy == pytest.approx(float(e_ref.energy), abs=1e-6)
    assert state.sci_state.amplitudes.dtype == np.complex128


@pytest.mark.parametrize(
    "dtype, davidson_dtype",
    [
        (torch.float64, torch.float16),
        (torch.float64, torch.bfloat16),
        (torch.float32, torch.float64),
        (torch.complex128, torch.float32),
    ],
)
def test_invalid_dtype_combination_raises(dtype, davidson_dtype) -> None:
    """Only an fp32 subspace over an fp64 matvec is a supported mixed combination."""
    with pytest.raises(ValueError, match="davidson_dtype"):
        make_torch_sci_solver(device="cpu", dtype=dtype, davidson_dtype=davidson_dtype)


def test_invalid_davidson_energy_raises() -> None:
    """A typo in davidson_fp32_energy must not silently pick one of the two modes."""
    with pytest.raises(ValueError, match="davidson_fp32_energy"):
        make_torch_sci_solver(device="cpu", davidson_fp32_energy="rq32")


def test_fp32_same_as_dtype_is_a_noop() -> None:
    """``davidson_dtype`` equal to ``dtype`` is the default path, not an error."""
    make_torch_sci_solver(device="cpu", dtype=torch.float64, davidson_dtype=torch.float64)


# --------------------------------------------------------------------------- #
# 7-8. Block sizing and multi-GPU config                                       #
# --------------------------------------------------------------------------- #
def test_pick_num_block_uses_fp32_element_size(monkeypatch) -> None:
    """With davidson_block_size=None the fp32 subspace fits about twice the block.

    Only C/HC shrink: the overhead stays fp64-sized, since the matvec output is fp64
    even with an fp32 subspace, so the fp32 block is a little under twice fp64's.
    """
    sizes = {}
    for name, kw in (("fp64", {}), ("fp32", FP32)):
        solver = make_torch_sci_solver(device="cpu", **kw)
        # 1000x1000 grid; a budget that fits a block of 6 in fp64 (below the cap of 30).
        n = 1000 * 1000
        monkeypatch.setattr(solver, "_free_bytes", lambda device=None, n=n: (16 * n * 8) / 0.65)
        sizes[name] = solver._pick_num_block(1000, 1000)
    assert sizes["fp64"] == 6
    assert sizes["fp32"] == 12


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, 6),
        (FP32, 12),
        ({"dtype": torch.complex128}, 2),
        ({"dtype": torch.complex128, "davidson_dtype": torch.complex64}, 4),
        ({"dtype": torch.float32}, 14),
    ],
    ids=["fp64", "fp32-subspace", "complex128", "complex64-subspace", "float32"],
)
def test_pick_num_block_counts_real_element_sizes(monkeypatch, kw, expected) -> None:
    """The auto block budgets every dtype by its real element size.

    With ``16 N 8`` bytes free: fp64 fits ``(128 - 4·8) / (2·8) = 6`` columns. A
    complex128 element is 16 bytes in the block and in the four work vectors, so
    only 2 fit; complex64 over complex128 keeps 16-byte work vectors (4); a pure
    float32 solve halves both (14). Charging complex at 8 bytes would overshoot
    a tight budget by about 2x.
    """
    solver = make_torch_sci_solver(device="cpu", **kw)
    n = 1000 * 1000
    monkeypatch.setattr(solver, "_free_bytes", lambda device=None, n=n: (16 * n * 8) / 0.65)
    assert solver._pick_num_block(1000, 1000) == expected


def test_worker_config_forwards_davidson_dtype() -> None:
    """Multi-GPU workers are rebuilt from this dict; a missing knob would revert to fp64."""
    solver = make_torch_sci_solver(
        device="cpu",
        davidson_dtype=torch.float32,
        davidson_fp32_shift=-1.5,
        davidson_fp32_energy="ritz",
    )
    cfg = solver._worker_cfg()  # also asserts completeness against __init__
    assert cfg["davidson_dtype"] == torch.float32
    assert cfg["davidson_fp32_shift"] == -1.5
    assert cfg["davidson_fp32_energy"] == "ritz"


def test_fp32_multiroot_survives_eigh_convergence_failure(mode, monkeypatch) -> None:
    """An fp32 subspace eigh that fails to converge is retried in double precision.

    On one A100 node, LAPACK's eigh failed to converge on the near-degenerate fp32
    subspace of this exact solve and crashed it; the same solve passes elsewhere, so
    the failure is forced here. Every fp32 eigh call raises, and the solve must still
    return the two lowest singlets, matching the fp64 reference.
    """
    ref, _ = _solve(mode, n_roots=2, tol=1e-6, apply_ab_projection=True)
    real_eigh = torch.linalg.eigh

    def failing_in_fp32(a: torch.Tensor, *args, **kwargs):  # noqa: ANN202
        if a.dtype == torch.float32:
            raise torch.linalg.LinAlgError("linalg.eigh: failed to converge (forced)")
        return real_eigh(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", failing_in_fp32)
    got, _ = _solve(mode, n_roots=2, tol=1e-6, apply_ab_projection=True, **FP32)
    for r, g in zip(ref, got):
        assert g.state == "converged"
        assert g.energy == pytest.approx(r.energy, abs=1e-8)


def test_fp32_basis_stays_orthonormal_below_its_precision_floor() -> None:
    """A float32 single-root Davidson asked for an unreachable residual stays exact.

    Below the float32 floor every correction is mostly rounding noise; with the
    conditional second Gram-Schmidt pass the leftover overlap compounded over a
    30-column block until the Ritz value left the spectrum and the returned
    vector drifted (2.4e-4 Ha here, 1.8 Ha at 1225 determinants). Reorthogonalizing
    every correction keeps the basis orthonormal, so the fp64 Rayleigh quotient
    of the returned vector stays the ground energy.
    """
    import itertools

    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch
    from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals
    from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
    from torch_sci_solver.torch_impl.mult import make_diag_terms_torch

    norb, nocc = 6, 3
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = sum(raw.transpose(p) for p in perms) / 8.0
    strings = sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), nocc))
    integrals = to_torch_integrals(*setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0))
    dets = torch.tensor(strings)
    helpers = generate_all_torch_helpers(strings, strings, norb, device="cpu")
    hii = make_diag_terms_torch(dets, dets, norb, integrals)
    matvec64 = build_matvec(
        "batched", hii, dets, dets, norb, integrals, helpers, chunk_size=1 << 20
    )
    n = hii.numel()
    h = torch.stack([matvec64(column) for column in torch.eye(n, dtype=torch.float64)], 1)
    exact = float(torch.linalg.eigvalsh(h)[0])
    shift = float(hii.min())
    shifted = h - shift * torch.eye(n, dtype=torch.float64)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return (shifted @ v.to(torch.float64)).to(torch.float32)

    start = basis_init_vector_torch(n, 1, 0, dtype=torch.float64).to(torch.float32)
    energy, vec, info = davidson_torch(
        (hii - shift).to(torch.float32), start, matvec, 100, 30, 1e-10, n_roots=1
    )
    assert not info["converged"]  # the float32 floor is far above 1e-10
    v = vec.to(torch.float64)
    v = v / v.norm()
    assert float(v @ h @ v) == pytest.approx(exact, abs=1e-6)
    assert energy + shift == pytest.approx(exact, abs=1e-4)


def test_sub_float64_solve_returns_at_the_precision_floor() -> None:
    """A sub-float64 solve stops at its residual floor instead of the full budget.

    The residual cannot go below roughly ``sqrt(nnz)·‖H‖·eps(float32)``, so at
    ``tol=1e-12`` it plateaus and every remaining restart is wasted — and the
    missed-root check re-solves on top of that. The stop is reported as
    ``stagnated``; the same solve in float64 still converges, and a reachable
    float32 tolerance converges without the flag. Both Davidson loops (multi-
    and single-root, the latter also used by each missed-root search) must stop.
    """
    from torch_sci_solver.torch_impl.davidson import davidson_torch

    n = 400
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n))
    h32 = torch.tensor(a + a.T, dtype=torch.float32)
    h64 = h32.double()
    start = torch.zeros(n, dtype=torch.float32)
    start[0] = 1.0

    _, _, info_multi = davidson_torch(
        h32.diagonal().clone(), start, lambda v: h32 @ v, 80, 10, 1e-12, n_roots=3
    )
    assert info_multi["stagnated"] is True and not info_multi["converged"]
    assert info_multi["n_outer"] < 80

    _, _, info_single = davidson_torch(
        h32.diagonal().clone(), start, lambda v: h32 @ v, 80, 10, 1e-12
    )
    assert info_single["stagnated"] is True and info_single["n_outer"] < 80

    _, _, info_fp64 = davidson_torch(
        h64.diagonal().clone(), start.double(), lambda v: h64 @ v, 80, 10, 1e-12, n_roots=3
    )
    assert info_fp64["converged"] and "stagnated" not in info_fp64

    _, _, info_reachable = davidson_torch(
        h32.diagonal().clone(), start, lambda v: h32 @ v, 80, 10, 1e-3, n_roots=3
    )
    assert info_reachable["converged"] and "stagnated" not in info_reachable

    # The fp64 guard has to be observable on a solve that does not converge
    # early: at 60 restarts float32 stops at ~50 with the flag, while float64
    # runs the full budget without it.
    _, _, info32_budget = davidson_torch(
        h32.diagonal().clone(), start, lambda v: h32 @ v, 60, 10, 1e-12, n_roots=3
    )
    assert info32_budget["stagnated"] is True and info32_budget["n_outer"] < 60
    _, _, info64_budget = davidson_torch(
        h64.diagonal().clone(), start.double(), lambda v: h64 @ v, 60, 10, 1e-12, n_roots=3
    )
    assert info64_budget["n_outer"] == 60 and "stagnated" not in info64_budget
