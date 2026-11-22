"""Tests for torch_sci_solver.torch_impl.determinants.

Every batched torch function must produce, element-by-element, the
same result as the scalar Python reference (within float64 precision).
We test on:
  - Random 2L=8 bitmasks (L=4) — low-limb only, simplest.
  - Random 2L=72 bitmasks (L=36) — uses both limbs.
  - Real H2, H4, LiH integrals from PySCF — covers realistic input distributions.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import random

import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref import (
    det_from_alpha_beta,
    one_excite,
    # scalar reference functions
    parity,
    setup_integrals_from_pyscf_h1_eri,
    two_excite,
    zero_excite,
)
from torch_sci_solver.torch_impl import (
    det_from_alpha_beta_batch,
    getocc_batch,
    one_excite_batch,
    parity_batch,
    two_excite_batch,
    zero_excite_batch,
)
from torch_sci_solver.torch_impl.determinants import (
    _bit64_mask_below,
    _popcount64,
)

# Force CPU; these tests do not need device dispatch yet.
# Every tensor constructor below passes dtype= explicitly, so no
# torch.set_default_dtype() — that mutates global state and breaks
# unrelated tests that run after this file in CI.
DEVICE = torch.device("cpu")


def assert_close(actual, expected, rtol=1e-12, atol=1e-13, label=""):
    """Assert two arrays agree within a combined relative/absolute tolerance."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    diff = np.abs(a - e).max() if a.size > 0 else 0.0
    scale = max(np.abs(e).max() if e.size > 0 else 1.0, 1.0)
    if diff > atol + rtol * scale:
        raise AssertionError(f"{label}: max |a-b| = {diff:.3e}, scale = {scale:.3e}")


def _to_int64(values):
    """Convert iterable of Python ints in [0, 2^64) to signed int64
    bit pattern (subtract 2^64 if the high bit is set).
    """
    INT64_OFFSET = 1 << 64
    INT64_MAX_UNSIGNED = 1 << 63
    return [int(v) if int(v) < INT64_MAX_UNSIGNED else int(v) - INT64_OFFSET for v in values]


def _det_to_lo_hi(dets_py):
    """Split Python-int dets into (lo, hi) int64 tensors."""
    LO_MASK = (1 << 64) - 1
    lo_py = [d & LO_MASK for d in dets_py]
    hi_py = [d >> 64 for d in dets_py]
    return (
        torch.tensor(_to_int64(lo_py), dtype=torch.int64),
        torch.tensor(_to_int64(hi_py), dtype=torch.int64),
    )


# ============================================================================
# 1. Popcount + range mask basics
# ============================================================================


def test_popcount_basic():
    """_popcount64 matches Python's bit_count on a handful of fixed bit patterns."""
    xs = [0, 1, 0xFF, 0xFFFFFFFFFFFFFFFF, 0x5555555555555555, 0xAAAAAAAAAAAAAAAA]
    # Avoid the high bit (would be negative in int64); we'll test it via mask.
    safe_xs = [x for x in xs if x < (1 << 63)]
    t = torch.tensor(safe_xs, dtype=torch.int64)
    pop = _popcount64(t)
    expected = [bin(x).count("1") for x in safe_xs]
    print(f"  inputs:   {safe_xs}")
    print(f"  popcount: {pop.tolist()}")
    print(f"  expected: {expected}")
    assert pop.tolist() == expected
    print("test_popcount_basic passed")


def test_popcount_random_64bit():
    """_popcount64 matches Python's bit_count on random 64-bit values."""
    rng = random.Random(0)
    samples = [rng.randint(0, (1 << 63) - 1) for _ in range(500)]
    t = torch.tensor(samples, dtype=torch.int64)
    pop = _popcount64(t)
    expected = [x.bit_count() for x in samples]
    assert pop.tolist() == expected
    print("test_popcount_random_64bit passed")


def test_mask_below():
    """_bit64_mask_below produces the expected low-bit mask for each bit count."""
    n = torch.tensor([0, 1, 2, 8, 32, 63, 64], dtype=torch.int64)
    got = _bit64_mask_below(n)
    expected = [
        0,  # below 0 bits = 0
        1,  # 1 bit
        3,  # 2 bits
        0xFF,
        0xFFFFFFFF,
        0x7FFFFFFFFFFFFFFF,
        -1,  # all 64 bits set = -1 in two's complement
    ]
    assert got.tolist() == expected, f"got {got.tolist()}, expected {expected}"
    print("test_mask_below passed")


# ============================================================================
# 2. getocc_batch matches scalar
# ============================================================================


def test_getocc_batch_random():
    """getocc_batch matches the scalar getocc bit-check, incl. per-batch indices."""
    rng = random.Random(1)
    n_bits = 72
    n_dets = 50
    dets_py = [rng.randint(0, (1 << n_bits) - 1) for _ in range(n_dets)]
    det_lo, det_hi = _det_to_lo_hi(dets_py)
    dets = torch.stack((det_lo, det_hi), dim=-1)
    for idx in range(n_bits):
        got = getocc_batch(dets, None, idx).tolist()
        expected = [bool((d >> idx) & 1) for d in dets_py]
        assert got == expected, f"idx={idx}, mismatch"

    # Test per-batch indices
    per_batch_idx = torch.tensor(
        [rng.randint(0, n_bits - 1) for _ in range(n_dets)], dtype=torch.int64
    )
    got = getocc_batch(dets, None, per_batch_idx).tolist()
    expected = [bool((d >> int(idx)) & 1) for d, idx in zip(dets_py, per_batch_idx)]
    assert got == expected
    print("test_getocc_batch_random passed")


# ============================================================================
# 3. parity_batch matches scalar (the trickiest one)
# ============================================================================


def test_parity_batch_random():
    """parity_batch matches the scalar parity() over random (det, start, end) triples."""
    rng = random.Random(2)
    n_bits = 72
    n = 200
    dets_py = [rng.randint(0, (1 << n_bits) - 1) for _ in range(n)]
    # Pick random (start, end) pairs with start <= end <= n_bits
    pairs = [tuple(sorted((rng.randint(0, n_bits), rng.randint(0, n_bits)))) for _ in range(n)]
    starts_py = [s for s, _ in pairs]
    ends_py = [e for _, e in pairs]

    expected = [parity(d, s, e, 1.0) for d, s, e in zip(dets_py, starts_py, ends_py)]

    det_lo, det_hi = _det_to_lo_hi(dets_py)
    dets = torch.stack((det_lo, det_hi), dim=-1)
    starts = torch.tensor(starts_py, dtype=torch.int64)
    ends = torch.tensor(ends_py, dtype=torch.int64)
    sgn0 = torch.ones(n, dtype=torch.float64)

    got = parity_batch(dets, None, starts, ends, sgn0).tolist()
    diff_count = sum(1 for g, e in zip(got, expected) if g != e)
    if diff_count > 0:
        # Show first mismatch
        for i, (g, e_) in enumerate(zip(got, expected)):
            if g != e_:
                print(
                    f"  mismatch #{i}: det={dets_py[i]:#x}, start={starts_py[i]}, end={ends_py[i]}"
                )
                print(f"    scalar: {e_}, batched: {g}")
                break
    assert diff_count == 0, f"{diff_count}/{n} mismatches"
    print(f"test_parity_batch_random passed (n={n}, all match)")


# ============================================================================
# 4. det_from_alpha_beta_batch matches scalar
# ============================================================================


def test_det_from_alpha_beta_batch():
    """det_from_alpha_beta_batch matches the scalar det_from_alpha_beta on random strings."""
    rng = random.Random(3)
    L = 36
    n = 50
    alpha_py = [rng.randint(0, (1 << L) - 1) for _ in range(n)]
    beta_py = [rng.randint(0, (1 << L) - 1) for _ in range(n)]
    expected = [det_from_alpha_beta(a, b, L) for a, b in zip(alpha_py, beta_py)]

    alpha = torch.tensor(_to_int64(alpha_py), dtype=torch.int64)
    beta = torch.tensor(_to_int64(beta_py), dtype=torch.int64)
    det = det_from_alpha_beta_batch(alpha, beta, L)
    got_unsigned = [g & ((1 << (2 * L)) - 1) for g in _limbs_to_int(det)]
    assert got_unsigned == expected, (
        f"first mismatch at idx {next(i for i, (g, e) in enumerate(zip(got_unsigned, expected)) if g != e)}"
    )
    print(f"test_det_from_alpha_beta_batch passed (n={n}, L={L})")


# ============================================================================
# 5. zero_excite_batch matches scalar
# ============================================================================


def test_zero_excite_batch_h4():
    """H4 minimal: enumerate all dets in the C(4,2)×C(4,2) basis,
    compare scalar zero_excite to batched."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    # Enumerate dets
    adets = []
    for combo in itertools.combinations(range(L), n_a):
        mask = 0
        for bit in combo:
            mask |= 1 << bit
        adets.append(mask)
    bdets = adets[:]  # singlet
    adets.sort()
    bdets.sort()

    # All product dets
    L_full = 2 * L
    dets_py = [det_from_alpha_beta(a, b, L) for a in adets for b in bdets]
    expected = np.array([zero_excite(d, L_full, I0, I1, I2) for d in dets_py])

    # Batched
    det_lo, det_hi = _det_to_lo_hi(dets_py)
    dets = torch.stack((det_lo, det_hi), dim=-1)
    I1_diag = torch.tensor([I1.value(i, i) for i in range(L_full)], dtype=torch.float64)
    I2_dir = torch.tensor(I2._direct, dtype=torch.float64)
    I2_exc = torch.tensor(I2._exchange, dtype=torch.float64)

    got = zero_excite_batch(dets, None, L_full, I0, I1_diag, I2_dir, I2_exc).numpy()
    diff = np.abs(got - expected).max()
    print(f"  H4: {len(dets_py)} dets, max |torch - scalar| = {diff:.3e}")
    assert_close(got, expected, rtol=1e-12, label="zero_excite_batch H4")
    print("test_zero_excite_batch_h4 passed")


# ============================================================================
# 6. one_excite_batch matches scalar
# ============================================================================


def test_one_excite_batch_h4():
    """one_excite_batch matches the scalar one_excite on random H4 (det, i, a) triples."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    L_full = 2 * L

    # Build random (det, i, a) triples where i,a are spin-orbital indices
    rng = random.Random(4)
    samples = []
    n_samples = 200
    for _ in range(n_samples):
        # Random det with random electron count
        det = rng.randint(0, (1 << L_full) - 1)
        i_ = rng.randint(0, L_full - 1)
        a_ = rng.randint(0, L_full - 1)
        if i_ == a_:
            a_ = (a_ + 1) % L_full  # i ≠ a
        samples.append((det, i_, a_))

    expected = np.array([one_excite(d, i_, a_, I1, I2, L_full) for d, i_, a_ in samples])

    det_lo, det_hi = _det_to_lo_hi([d for d, _, _ in samples])
    dets = torch.stack((det_lo, det_hi), dim=-1)
    i_b = torch.tensor([i_ for _, i_, _ in samples], dtype=torch.int64)
    a_b = torch.tensor([a_ for _, _, a_ in samples], dtype=torch.int64)
    I1_t = torch.tensor(I1._h, dtype=torch.float64)
    I2_t = torch.tensor(I2._eri, dtype=torch.float64)

    got = one_excite_batch(dets, None, i_b, a_b, I1_t, I2_t, L_full).numpy()
    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1.0)
    print(f"  H4 random (det,i,a): n={n_samples}, max diff = {diff:.3e}, scale = {scale:.3e}")
    assert_close(got, expected, rtol=1e-12, label="one_excite_batch H4")
    print("test_one_excite_batch_h4 passed")


# ============================================================================
# 7. two_excite_batch matches scalar
# ============================================================================


def test_two_excite_batch_h4():
    """two_excite_batch matches the scalar two_excite on random H4 double excitations."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    L_full = 2 * L

    rng = random.Random(5)
    samples = []
    n_samples = 200
    for _ in range(n_samples):
        det = rng.randint(0, (1 << L_full) - 1)
        # Pick 4 distinct orbitals
        orbs = rng.sample(range(L_full), 4)
        i_, j_, a_, b_ = orbs
        samples.append((det, i_, j_, a_, b_))

    expected = np.array([two_excite(d, i_, j_, a_, b_, I2) for d, i_, j_, a_, b_ in samples])

    det_lo, det_hi = _det_to_lo_hi([d for d, *_ in samples])
    dets = torch.stack((det_lo, det_hi), dim=-1)
    i_t = torch.tensor([i_ for _, i_, _, _, _ in samples], dtype=torch.int64)
    j_t = torch.tensor([j_ for _, _, j_, _, _ in samples], dtype=torch.int64)
    a_t = torch.tensor([a_ for _, _, _, a_, _ in samples], dtype=torch.int64)
    b_t = torch.tensor([b_ for _, _, _, _, b_ in samples], dtype=torch.int64)
    I2_t = torch.tensor(I2._eri, dtype=torch.float64)

    got = two_excite_batch(dets, None, i_t, j_t, a_t, b_t, I2_t).numpy()
    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1.0)
    print(f"  H4 random (det,i,j,a,b): n={n_samples}, max diff = {diff:.3e}, scale = {scale:.3e}")
    assert_close(got, expected, rtol=1e-12, label="two_excite_batch H4")
    print("test_two_excite_batch_h4 passed")


# ============================================================================
# 8. 36-orbital test (uses high limb)
# ============================================================================


def test_36_orbital_kernels():
    """Smoke test: 72-bit dets (L=36) — make sure the high limb is exercised."""
    L = 36
    L_full = 2 * L
    rng = random.Random(6)

    # Random dets
    n_dets = 100
    dets_py = [rng.randint(0, (1 << L_full) - 1) for _ in range(n_dets)]

    # Synthetic integrals (random small values)
    np.random.seed(0)
    I1_arr = np.random.randn(L_full, L_full) * 0.1
    I1_arr = (I1_arr + I1_arr.T) / 2
    # Diagonal of I1 — simulate a OneInt
    I1_diag = np.diag(I1_arr).copy()
    # Synthetic ERI with spin selection enforced
    I2_arr = np.zeros((L_full, L_full, L_full, L_full))
    for p in range(L_full):
        for q in range(L_full):
            for r in range(L_full):
                for s in range(L_full):
                    if (p % 2) != (q % 2) or (r % 2) != (s % 2):
                        continue
                    I2_arr[p, q, r, s] = np.random.randn() * 0.05
    I2_arr = (I2_arr + I2_arr.transpose(1, 0, 2, 3)) / 2  # mild symmetry
    I2_direct = np.array([[I2_arr[p, p, q, q] for q in range(L_full)] for p in range(L_full)])
    I2_exchange = np.array([[I2_arr[p, q, q, p] for q in range(L_full)] for p in range(L_full)])

    # Wrap into scalar-compatible objects
    class _I1:
        pass

    I1 = _I1()
    I1.value = lambda i, j: I1_arr[i, j]

    class _I2:
        def value(self, p, q, r, s):
            if (p % 2) != (q % 2) or (r % 2) != (s % 2):
                return 0.0
            return I2_arr[p, q, r, s]

        def direct_value(self, i, j):
            return I2_direct[i, j]

        def exchange_value(self, i, j):
            return I2_exchange[i, j]

    I2_obj = _I2()
    I0 = 1.234

    # Scalar zero_excite
    expected = np.array([zero_excite(d, L_full, I0, I1, I2_obj) for d in dets_py])

    # Batched zero_excite
    det_lo, det_hi = _det_to_lo_hi(dets_py)
    dets = torch.stack((det_lo, det_hi), dim=-1)
    I1d = torch.tensor(I1_diag, dtype=torch.float64)
    I2_dir = torch.tensor(I2_direct, dtype=torch.float64)
    I2_exc = torch.tensor(I2_exchange, dtype=torch.float64)
    got = zero_excite_batch(dets, None, L_full, I0, I1d, I2_dir, I2_exc).numpy()
    diff = np.abs(got - expected).max()
    scale = max(np.abs(expected).max(), 1.0)
    print(f"  36-orbital zero_excite: n={n_dets}, max diff = {diff:.3e}, scale = {scale:.3e}")
    assert_close(got, expected, rtol=1e-11, label="zero_excite 36 orbitals")

    # one_excite on the same dets, random (i, a)
    n_pairs = 200
    i_py = [rng.randint(0, L_full - 1) for _ in range(n_pairs)]
    a_py = [(j + rng.randint(1, L_full - 1)) % L_full for j in i_py]
    det_chosen = [dets_py[k % n_dets] for k in range(n_pairs)]
    expected1 = np.array([
        one_excite(d, i_, a_, I1, I2_obj, L_full) for d, i_, a_ in zip(det_chosen, i_py, a_py)
    ])

    det_lo1, det_hi1 = _det_to_lo_hi(det_chosen)
    dets1 = torch.stack((det_lo1, det_hi1), dim=-1)
    i_t = torch.tensor(i_py, dtype=torch.int64)
    a_t = torch.tensor(a_py, dtype=torch.int64)
    I1_t = torch.tensor(I1_arr, dtype=torch.float64)
    I2_t = torch.tensor(I2_arr, dtype=torch.float64)
    got1 = one_excite_batch(dets1, None, i_t, a_t, I1_t, I2_t, L_full).numpy()
    diff1 = np.abs(got1 - expected1).max()
    scale1 = max(np.abs(expected1).max(), 1.0)
    print(f"  36-orbital one_excite:  n={n_pairs}, max diff = {diff1:.3e}, scale = {scale1:.3e}")
    assert_close(got1, expected1, rtol=1e-11, label="one_excite 36 orbitals")

    print("test_36_orbital_kernels passed")


# ============================================================================
# N-limb characterization: >128-bit determinants
#
# These pin the target behaviour for the n-limb generalization: determinants
# stored as a stacked ``(batch, n_limbs)`` int64 tensor rather than a fixed
# ``(det_lo, det_hi)`` pair, so ``n_spinorb`` is no longer capped at 128.
#
# They are ``xfail(strict=True)`` until that generalization lands: strict so
# they flip to a hard failure the moment the primitives start working, rather
# than silently passing as xpass.
# ============================================================================

# Bit widths crossing 2, 2, 3 and 4 limbs respectively. 65 and 129 sit one bit
# past a limb boundary — the case where the top limb is almost entirely unused
# and an off-by-one in limb-count arithmetic shows up.
_NLIMB_WIDTHS = [65, 100, 129, 256]


def _n_limbs_for(n_bits: int) -> int:
    """Number of int64 limbs needed to hold an ``n_bits``-wide bitmask."""
    return (n_bits + 63) // 64


def _det_to_limbs(dets_py, n_limbs: int) -> torch.Tensor:
    """Pack Python-int dets into a ``(batch, n_limbs)`` signed-int64 tensor.

    N-limb generalization of :func:`_det_to_lo_hi`; limb ``k`` holds bits
    ``[64k, 64k+64)``, low limb first (same ordering convention as the
    ``(lo, hi)`` pair it replaces).
    """
    LO_MASK = (1 << 64) - 1
    rows = [_to_int64([(d >> (64 * k)) & LO_MASK for k in range(n_limbs)]) for d in dets_py]
    return torch.tensor(rows, dtype=torch.int64)


def _limbs_to_int(dets: torch.Tensor) -> list[int]:
    """Reassemble a ``(batch, n_limbs)`` int64 tensor into unsigned Python ints."""
    INT64_OFFSET = 1 << 64
    out = []
    for row in dets.tolist():
        v = 0
        for k, limb in enumerate(row):
            v |= ((limb + INT64_OFFSET) & ((1 << 64) - 1)) << (64 * k)
        out.append(v)
    return out


def test_limb_pack_roundtrip():
    """``_det_to_limbs``/``_limbs_to_int`` round-trip, and agree with the (lo, hi) helpers.

    Not xfailed: this validates the *test harness* itself, independently of the
    production generalization, so a later failure here means the helpers are
    wrong rather than the kernels.
    """
    rng = random.Random(10)
    for n_bits in _NLIMB_WIDTHS:
        n_limbs = _n_limbs_for(n_bits)
        dets_py = [rng.randrange(0, 1 << n_bits) for _ in range(32)]
        packed = _det_to_limbs(dets_py, n_limbs)
        assert packed.shape == (len(dets_py), n_limbs)
        assert _limbs_to_int(packed) == dets_py, f"round-trip failed at n_bits={n_bits}"

    # Cross-check against the existing 2-limb helpers on a 128-bit-safe width.
    dets_py = [rng.randrange(0, 1 << 120) for _ in range(16)]
    lo, hi = _det_to_lo_hi(dets_py)
    packed = _det_to_limbs(dets_py, 2)
    assert torch.equal(packed[:, 0], lo)
    assert torch.equal(packed[:, 1], hi)
    print("test_limb_pack_roundtrip passed")


@pytest.mark.parametrize("n_bits", _NLIMB_WIDTHS)
def test_getocc_batch_nlimb(n_bits):
    """getocc_batch reads any bit of an n-limb det, matching the scalar bit-check."""
    rng = random.Random(11 + n_bits)
    n_dets = 40
    n_limbs = _n_limbs_for(n_bits)
    dets_py = [rng.randrange(0, 1 << n_bits) for _ in range(n_dets)]
    dets = _det_to_limbs(dets_py, n_limbs)

    # Per-batch bit indices spanning the full width, including the top limb.
    idx_py = [rng.randrange(0, n_bits) for _ in range(n_dets)]
    idx = torch.tensor(idx_py, dtype=torch.int64)
    got = getocc_batch(dets, None, idx).tolist()
    expected = [bool((d >> i) & 1) for d, i in zip(dets_py, idx_py)]
    assert got == expected, f"n_bits={n_bits}: per-batch index mismatch"

    # Every bit position, scalar index.
    for i in range(n_bits):
        got_i = getocc_batch(dets, None, i).tolist()
        expected_i = [bool((d >> i) & 1) for d in dets_py]
        assert got_i == expected_i, f"n_bits={n_bits}, idx={i}"


@pytest.mark.parametrize("n_bits", _NLIMB_WIDTHS)
def test_parity_batch_nlimb(n_bits):
    """parity_batch matches the scalar ``parity()`` for n-limb dets.

    The fermionic sign is the operation most sensitive to limb splitting: it
    needs a popcount over an arbitrary ``[start, end)`` window that may span
    several limbs, so an error in the per-limb range clamping shows up here as
    a wrong sign rather than a wrong occupancy.
    """
    rng = random.Random(12 + n_bits)
    n = 150
    n_limbs = _n_limbs_for(n_bits)
    dets_py = [rng.randrange(0, 1 << n_bits) for _ in range(n)]
    pairs = [tuple(sorted((rng.randint(0, n_bits), rng.randint(0, n_bits)))) for _ in range(n)]
    starts_py = [s for s, _ in pairs]
    ends_py = [e for _, e in pairs]

    expected = [parity(d, s, e, 1.0) for d, s, e in zip(dets_py, starts_py, ends_py)]

    dets = _det_to_limbs(dets_py, n_limbs)
    got = parity_batch(
        dets,
        None,
        torch.tensor(starts_py, dtype=torch.int64),
        torch.tensor(ends_py, dtype=torch.int64),
        torch.ones(n, dtype=torch.float64),
    ).tolist()
    assert got == expected, f"n_bits={n_bits}: parity mismatch"


@pytest.mark.parametrize("n_bits", _NLIMB_WIDTHS)
def test_popcount_nlimb(n_bits):
    """Total popcount over all limbs matches ``int.bit_count()``."""
    from torch_sci_solver.torch_impl.determinants import _popcount

    rng = random.Random(13 + n_bits)
    n_limbs = _n_limbs_for(n_bits)
    dets_py = [rng.randrange(0, 1 << n_bits) for _ in range(64)]
    # Include the all-ones pattern: every limb fully packed, so every limb's
    # sign bit is set — the case a naive signed-shift popcount gets wrong.
    dets_py.append((1 << n_bits) - 1)
    dets = _det_to_limbs(dets_py, n_limbs)
    got = _popcount(dets).tolist()
    assert got == [d.bit_count() for d in dets_py], f"n_bits={n_bits}"


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    test_popcount_basic()
    test_popcount_random_64bit()
    test_mask_below()
    test_getocc_batch_random()
    test_parity_batch_random()
    test_det_from_alpha_beta_batch()
    test_zero_excite_batch_h4()
    test_one_excite_batch_h4()
    test_two_excite_batch_h4()
    test_36_orbital_kernels()
    print("\nALL TESTS PASSED")
