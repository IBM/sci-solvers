"""Batched PyTorch port of the SBD Slater-Condon kernels.

Mirrors ``torch_sci_solver.sbd_ref.determinants`` (scalar Python). All
functions here are vectorized over a batch dimension.

Bitmask representation
----------------------
A determinant over ``n_bits`` spin-orbitals is stored as a stacked
``(batch, n_limbs)`` int64 tensor, where ``n_limbs = ceil(n_bits / 64)``
and limb ``k`` holds bits ``[64k, 64k+64)`` (limb 0 is the lowest 64
bits). This supports any ``n_bits`` the caller needs — not just the
``2L ≤ 128`` (two-limb) case this module originally covered.

Spin-orbital convention (same as scalar reference):
    bit 2k     ↔ spatial orbital k, α-spin
    bit 2k + 1 ↔ spatial orbital k, β-spin

Sign convention for torch
--------------------------
Torch uses *signed* int64 for bitwise ops. A fully-packed limb (every
bit set, including bit 63) is represented in two's complement, but
every bitwise op used here (``&``, ``|``, ``^``, right-shift followed
by ``& 1`` or ``& mask``) gives the same result on the signed bit
pattern as it would on the equivalent unsigned value, so no limb is
required to leave its high bit clear.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

Derived from SBD (https://github.com/r-ccs-cms/sbd), licensed under the Apache
License 2.0: Copyright 2025 RIKEN and its contributors, and the other SBD
copyright holders listed in NOTICE. This file is a modified Python version of
the original C++ code.

SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import Tensor


def _as_limbs(det: Tensor, det_hi: None) -> Tensor:
    """Normalize a determinant argument to a stacked ``(batch, n_limbs)`` tensor.

    Every caller passes ``det_hi=None``: a ``(batch, n_limbs)`` tensor
    (``det.ndim >= 2``) is returned unchanged, and a bare single-limb
    ``(batch,)`` tensor gets an ``n_limbs=1`` axis added. ``det_hi`` only
    exists to flag a caller still on the old ``(det_lo, det_hi)`` pair
    convention — no such caller remains.

    Raises:
        ValueError: ``det_hi`` is not ``None`` (a leftover ``(lo, hi)`` pair
            call, which every primitive in this module has stopped supporting).
    """
    if det_hi is not None:
        raise ValueError(
            "the (det_lo, det_hi) pair calling convention is no longer supported; "
            "pass a single (batch, n_limbs) (or bare (batch,)) tensor as `det` "
            "with det_hi=None."
        )
    if det.ndim >= 2:
        return det
    return det.unsqueeze(-1)


# ============================================================================
# Limb math
# ============================================================================


def _bit64_mask_below(n: Tensor) -> Tensor:
    """Mask of low `n` bits of an int64 word, vectorized.

    Equivalent to ``(1 << n) - 1`` per-element, but careful for n == 64:
    ``1 << 64`` is undefined in fixed-width arithmetic. We special-case
    n == 64 to return -1 (= all bits set in two's complement).

    Args:
        n: int64 tensor with values in [0, 64].

    Returns:
        int64 tensor of bitmasks, same shape as ``n``.

    Notes:
        The masks are 64-bit words, so an int32 ``n`` (a caller's narrowed
        orbital-index table) is widened first; shifting an int64 ``1`` by an
        int32 amount would otherwise yield an int32 result and lose the bits
        above position 31.
    """
    n = n.to(torch.int64)
    # Clamp n to [0, 64] and treat 64 specially. A negative n (an owned limb
    # window that starts past the orbital) masks nothing.
    n64 = torch.where(n >= 64, torch.full_like(n, 64), n.clamp_min(0))
    # For n < 64: mask = (1 << n) - 1. For n == 64: -1 (all ones).
    all_ones = torch.tensor(-1, dtype=torch.int64, device=n.device)
    mask_lt64 = (torch.tensor(1, dtype=torch.int64, device=n.device) << n64) - 1
    return torch.where(n >= 64, all_ones, mask_lt64)


def _popcount64(x: Tensor) -> Tensor:
    """SWAR popcount for int64 tensors.

    Implements the standard "bit-twiddling hack" parallel popcount on
    each int64 word. Works on any device (CPU, CUDA). Roughly 5x faster
    than a Python loop and fully vectorized across the batch.

    Verified against scalar ``int.bit_count()`` for all 64-bit inputs in
    the test suite (sample, not exhaustive).

    Args:
        x: int64 tensor (any shape).
    Returns:
        int64 tensor of popcounts, same shape.
    """
    # Constants. Torch lacks unsigned int64, but signed-bit-pattern
    # constants work the same under bitwise ops.
    K1 = 0x5555555555555555  # 01010101...
    K2 = 0x3333333333333333  # 00110011...
    K4 = 0x0F0F0F0F0F0F0F0F  # 0000111100001111...
    KM = 0x0101010101010101  # multiplier for sum-of-bytes

    # Use logical right-shift semantics for negative inputs by masking.
    # Since all our bitmasks are non-negative (high bit of limb is zero
    # for our problem sizes), arithmetic shift gives the same result —
    # but we mask defensively in case any caller passes a value with
    # sign bit set.
    x = x.to(torch.int64)
    x = x - ((x >> 1) & K1)
    x = (x & K2) + ((x >> 2) & K2)
    x = (x + (x >> 4)) & K4
    # Multiply by KM, take high byte = sum of all bytes (popcount).
    return (x * KM) >> 56


_INT64_MASK = (1 << 64) - 1


def _bra_from_offsets(offsets: Tensor, n_strings: int, device: torch.device) -> Tensor:
    """Inverse of cumulative offsets — repeat_interleave bra indices by
    per-bra excitation counts."""
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(n_strings, device=device, dtype=torch.int64), counts
    )


def _string_limbs(strings: Tensor) -> Tensor:
    """``(M, n_limbs)`` word matrix of a string tensor.

    A ``(M,)`` tensor is one word (``n_limbs == 1``); a ``(M, n_limbs)`` tensor
    already carries its limbs, limb ``k`` holding bits ``[64k, 64k + 64)``.
    """
    if strings.ndim == 1:
        return strings.unsqueeze(-1)
    if strings.ndim != 2:
        raise ValueError(f"string tensors must be 1-D or 2-D, got ndim={strings.ndim}")
    return strings


def _occupancy_bits(strings: Tensor, n_orb: int) -> Tensor:
    """``(M, n_orb)`` int64 0/1 occupancy of every string, summed over limbs.

    Bit ``p`` of a string lives in limb ``p // 64`` at position ``p % 64``;
    limbs whose window does not contain ``p`` contribute zero.
    """
    limbs = _string_limbs(strings)
    device = strings.device
    p_range = torch.arange(n_orb, device=device, dtype=torch.int64)
    bits = torch.zeros((limbs.shape[0], n_orb), dtype=torch.int64, device=device)
    for k in range(limbs.shape[1]):
        shift = p_range - 64 * k
        valid = (shift >= 0) & (shift < 64)
        word_bits = (limbs[:, k].unsqueeze(1) >> shift.clamp(0, 63).unsqueeze(0)) & 1
        bits += word_bits * valid.to(torch.int64)
    return bits


def _signed_int64(value: int) -> int:
    """Wrap a Python integer into torch's signed int64 range."""
    value &= _INT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


def _popcount(det_lo: Tensor, det_hi: None = None) -> Tensor:
    """Total popcount over every limb.

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
    """
    det = _as_limbs(det_lo, det_hi)
    total = _popcount64(det[..., 0])
    for k in range(1, det.shape[-1]):
        total = total + _popcount64(det[..., k])
    return total


# ============================================================================
# Bit accessors
# ============================================================================


def getocc_batch(det_lo: Tensor, det_hi: None, idx: Tensor | int) -> Tensor:
    """Get the bit at position ``idx`` of each det.

    SBD-scalar equivalent: ``getocc(det, idx)``.

    Args:
        det_lo: an ``(*idx.shape, n_limbs)`` tensor — already broadcast
            against ``idx`` on every dimension but the trailing (limb) one.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        idx:    Either an int (same bit position for all dets) or an
                int64 tensor of shape ``det_lo.shape[:-1]`` (per-element bit
                position). Must satisfy ``0 ≤ idx < 64 * n_limbs``.

    Returns:
        Bool tensor of shape ``det_lo.shape[:-1]``: True iff bit ``idx`` is
        set in the det.
    """
    det = _as_limbs(det_lo, det_hi)
    if isinstance(idx, int):
        idx = torch.tensor(idx, dtype=torch.int64, device=det.device)
        idx = idx.expand(det.shape[:-1])
    limb_idx = idx >> 6
    bit_pos = idx & 63
    limb = torch.gather(det, -1, limb_idx.unsqueeze(-1)).squeeze(-1)
    bit = (limb >> bit_pos) & 1
    return bit.to(torch.bool)


# ============================================================================
# Popcount in a range
# ============================================================================


def _popcount_in_range(det_lo: Tensor, det_hi: None, start: Tensor, end: Tensor) -> Tensor:
    """Popcount of bits in positions [start, end) of each det.

    Per-element start/end. ``start`` must be ≤ ``end``; caller's
    responsibility.

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        start, end: (batch,) int64 bit positions, ``0 ≤ start ≤ end ≤ 64 * n_limbs``.
    Returns:
        (batch,) int64 popcounts.
    """
    # Build a "range mask" that has 1s only in positions [start, end),
    # one limb at a time: limb k covers bits [64k, 64k+64), so its share
    # of [start, end) is [start, end) clamped to that window and shifted
    # down by 64k before building the mask.
    det = _as_limbs(det_lo, det_hi)
    zero = torch.zeros_like(start)
    sixty_four = torch.full_like(start, 64)

    total = torch.zeros_like(start)
    for k in range(det.shape[-1]):
        offset = 64 * k
        limb_start = torch.clamp(start - offset, zero, sixty_four)
        limb_end = torch.clamp(end - offset, zero, sixty_four)
        limb_mask = _bit64_mask_below(limb_end) ^ _bit64_mask_below(limb_start)
        total = total + _popcount64(det[..., k] & limb_mask)
    return total


# ============================================================================
# Parity (fermion sign for moving electron from `start` to `end`)
# ============================================================================


def parity_batch(det_lo: Tensor, det_hi: None, start: Tensor, end: Tensor, sgn: Tensor) -> Tensor:
    """Update fermion sign for moving an electron through ``[start, end)``
    of each det. Vectorized version of scalar ``parity()``.

    SBD-scalar equivalent: ``parity(det, start, end, sgn)`` returns the
    updated sgn. The scalar version:
        nonzero = popcount of bits of det in [start, end)
        sgn *= (-1)^nonzero
        if det.bit(start): sgn *= -1

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        start, end:      (batch,) int64 with 0 ≤ start ≤ end ≤ 64 * n_limbs.
        sgn:             (batch,) float64 — sign before this call (typically ±1).
    Returns:
        (batch,) float64 — sign after this call.
    """
    nonzero = _popcount_in_range(det_lo, det_hi, start, end)
    # Multiply by (-1)^nonzero. Vectorized via 1 - 2*(nonzero & 1).
    sign_count = 1.0 - 2.0 * (nonzero & 1).to(sgn.dtype)
    out = sgn * sign_count

    # Extra flip if bit at `start` is set.
    bit_at_start = getocc_batch(det_lo, det_hi, start).to(sgn.dtype)
    extra = 1.0 - 2.0 * bit_at_start  # +1 if 0, -1 if 1
    return out * extra


# ============================================================================
# det_from_alpha_beta_batch
# ============================================================================


def det_from_alpha_beta_batch(alpha: Tensor, beta: Tensor, L: int) -> Tensor:
    """Interleave α and β half-bitmasks into full 2L-bit det bitmasks.

    SBD-scalar equivalent: ``det_from_alpha_beta(a, b, L)``.

    The interleaving: bit ``2k`` of det = bit ``k`` of α, bit ``2k+1`` of
    det = bit ``k`` of β. α and β may each be a single word (``L <= 64``) or
    a ``(batch, n_limbs)`` limb matrix, so the same call serves strings wider
    than 64 bits.

    Args:
        alpha, beta: (batch,) int64 half-bitmasks over L bits each, or
            (batch, ceil(L/64)) limb matrices (limb ``k`` holds bits
            ``[64k, 64k + 64)``).
        L:           number of spatial orbitals.

    Returns:
        ``(batch, n_limbs)`` int64, ``n_limbs = ceil(2L / 64)``. Limb ``j``
        holds bits ``[64j, 64j+64)`` of the interleaved det.

    Raises:
        ValueError: a limb matrix with fewer than ``ceil(L / 64)`` limbs.
    """
    device = alpha.device
    dtype = torch.int64
    batch_shape = alpha.shape if alpha.dim() == 1 else alpha.shape[:-1]
    n_alpha_limbs = (L + 63) // 64
    a = _as_limbs(alpha, None)
    b = _as_limbs(beta, None)
    if a.shape[-1] < n_alpha_limbs or b.shape[-1] < n_alpha_limbs:
        raise ValueError(
            f"alpha/beta need {n_alpha_limbs} limb(s) for L={L}; got "
            f"{a.shape[-1]} and {b.shape[-1]}"
        )
    n_limbs = (2 * L + 63) // 64
    limbs = [torch.zeros(batch_shape, dtype=dtype, device=device) for _ in range(n_limbs)]

    # For each spatial orbital k in [0, L):
    #   bit 2k of det ← bit k of α  → limb (2k) // 64, bit (2k) % 64
    #   bit 2k+1 of det ← bit k of β → limb (2k+1) // 64, bit (2k+1) % 64
    # Limb/bit selection is a Python int (pos is a loop variable, not a
    # tensor), so this costs no more than the fixed 2-limb version did.
    for k in range(L):
        alpha_bit_k = (a[..., k // 64] >> (k % 64)) & 1
        beta_bit_k = (b[..., k // 64] >> (k % 64)) & 1
        pos_a = 2 * k
        pos_b = 2 * k + 1
        limb_a, bit_a = pos_a // 64, pos_a % 64
        limb_b, bit_b = pos_b // 64, pos_b % 64
        limbs[limb_a] = limbs[limb_a] | (alpha_bit_k << bit_a)
        limbs[limb_b] = limbs[limb_b] | (beta_bit_k << bit_b)

    return torch.stack(limbs, dim=-1)


# ============================================================================
# ZeroExcite (diagonal element)
# ============================================================================


def zero_excite_batch(
    det_lo: Tensor,
    det_hi: None,
    n_spinorb: int,
    I0: float,
    I1_diag: Tensor,  # (2L,)   diagonal of one-electron integrals
    I2_direct: Tensor,  # (2L,2L) (ii|jj) chemist
    I2_exchange: Tensor,  # (2L,2L) (ij|ji) chemist
) -> Tensor:
    """Vectorized diagonal matrix element ⟨D|H|D⟩ over a batch of dets.

    SBD-scalar equivalent: ``zero_excite(det, n_spinorb, I0, I1, I2)``.

    Formula:
        energy = I0 + sum_{i ∈ D} I1(i, i)
                 + sum_{i < j ∈ D} [I2.direct(i,j) - δ_{spin(i)=spin(j)} I2.exchange(i,j)]

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        n_spinorb:                2L, the total number of spin-orbitals.
        I0:                       float, the core energy.
        I1_diag:                  (2L,) float64 — diagonal of the one-electron
                                  Hamiltonian: ``I1_diag[i] = I1.value(i, i)``.
        I2_direct, I2_exchange:   (2L, 2L) float64 — full direct and exchange
                                  matrices. We expect the caller to pre-mask
                                  exchange for same-spin only (since this
                                  function applies the spin-equality check
                                  separately if needed).

    Returns:
        (batch,) float64 of diagonal energies.

    Implementation: build a (batch, 2L) bool occupancy tensor, then
    compute the sums via einsum / batched matrix multiplications.
    """
    det = _as_limbs(det_lo, det_hi)
    batch_shape = det.shape[:-1]
    n_limbs = det.shape[-1]
    device = det.device
    dtype = I1_diag.dtype

    # Build occupancy: occ[b, i] = 1.0 if det b has bit i set, else 0.
    idx_range = torch.arange(n_spinorb, dtype=torch.int64, device=device)
    # Broadcast for each (b, i)
    occ = getocc_batch(
        det.unsqueeze(-2).expand(*batch_shape, n_spinorb, n_limbs),
        None,
        idx_range.expand(*batch_shape, n_spinorb),
    ).to(dtype)  # (batch, 2L)

    # One-electron: sum_i occ_i * I1_diag[i]
    e1 = occ @ I1_diag  # (batch,)

    # Two-electron pair sum:
    # Build pair-coefficient matrix once:
    #   pair[i, j] = (I2.direct(i, j) - δ_{i%2==j%2} I2.exchange(i, j))  for i < j
    #   pair[i, j] = 0 otherwise
    # Note we need same-spin check. spin(i) = i%2.
    # Caller passes I2_direct and I2_exchange as full (2L, 2L); we form
    # the mask here.
    spin_idx = idx_range % 2
    same_spin = (spin_idx.unsqueeze(-1) == spin_idx.unsqueeze(0)).to(dtype)  # (2L, 2L)
    pair_full = I2_direct - same_spin * I2_exchange  # (2L, 2L)
    # Upper triangle (i < j) — diagonal not included
    upper = torch.triu(torch.ones_like(pair_full), diagonal=1)
    pair_ut = pair_full * upper  # (2L, 2L); only i<j entries nonzero

    # Two-electron contribution per det:
    #   e2[b] = sum_{i<j} occ[b,i] * occ[b,j] * pair_ut[i,j]
    #         = occ @ pair_ut @ occ
    # Vectorize over batch via einsum.
    e2 = torch.einsum("bi,ij,bj->b", occ, pair_ut, occ)

    return I0 + e1 + e2


def zero_excite_batch_spinless(
    det_lo: Tensor,
    det_hi: None,
    n_spinorb: int,
    I0: float,
    I1_diag: Tensor,  # (nmo,)   diagonal of one-electron integrals
    I2_direct: Tensor,  # (nmo,nmo) (ii|jj) chemist
    I2_exchange: Tensor,  # (nmo,nmo) (ij|ji) chemist
) -> Tensor:
    """Vectorized diagonal matrix element ⟨D|H|D⟩ for a flat spinless operator.

    Identical to :func:`zero_excite_batch` except it drops the
    ``δ_{spin(i)=spin(j)}`` exchange guard: for a genuinely spinless operator
    (a single flat block of ``nmo`` spin-orbitals, no α/β factorization) the
    exchange term applies to *every* occupied pair — the spin structure lives
    in the integrals themselves, not in an orbital-index parity rule. Mirrors
    ``sci_solvers.spinless_ci_problem._zero_excite_spinless``.

    Formula:
        energy = I0 + sum_{i ∈ D} I1(i, i) + sum_{i < j ∈ D} [I2.direct(i,j) - I2.exchange(i,j)]

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor over ``n_spinorb`` bits, or a
            bare ``(batch,)`` tensor for a single-limb det. ``det_hi`` must
            be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        n_spinorb:                nmo, the total number of spin-orbitals.
        I0:                       float, the core energy.
        I1_diag:                  (nmo,) — diagonal of the one-electron Hamiltonian.
        I2_direct, I2_exchange:   (nmo, nmo) — full direct and exchange matrices,
                                  no spin masking applied by the caller.

    Returns:
        (batch,) of diagonal energies, same dtype as ``I1_diag``.
    """
    det = _as_limbs(det_lo, det_hi)
    batch_shape = det.shape[:-1]
    n_limbs = det.shape[-1]
    device = det.device
    dtype = I1_diag.dtype

    idx_range = torch.arange(n_spinorb, dtype=torch.int64, device=device)
    occ = getocc_batch(
        det.unsqueeze(-2).expand(*batch_shape, n_spinorb, n_limbs),
        None,
        idx_range.expand(*batch_shape, n_spinorb),
    ).to(dtype)  # (batch, nmo)

    e1 = occ @ I1_diag  # (batch,)

    pair_full = I2_direct - I2_exchange  # (nmo, nmo); no same-spin mask
    upper = torch.triu(torch.ones_like(pair_full), diagonal=1)
    pair_ut = pair_full * upper  # only i<j entries nonzero

    e2 = torch.einsum("bi,ij,bj->b", occ, pair_ut, occ)

    return I0 + e1 + e2


# ============================================================================
# OneExcite (single excitation matrix element)
# ============================================================================


def one_excite_batch(
    det_lo: Tensor,
    det_hi: None,
    i: Tensor,
    a: Tensor,
    I1: Tensor,
    I2_eri: Tensor,
    n_spinorb: int,
) -> Tensor:
    """Vectorized single-excitation matrix element.

    SBD-scalar equivalent: ``one_excite(det, i, a, I1, I2, n_spinorb)``.

    Formula:
        sgn = parity(det, min(i, a), max(i, a))
        e = I1[a, i] + sum_{j ∈ det} (I2(a, i, j, j) - I2(a, j, j, i))
        return sgn * e

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        i, a:            (batch,) int64 spin-orbital indices.
        I1:              (2L, 2L) float64 one-electron integrals h(p, q).
        I2_eri:          (2L, 2L, 2L, 2L) float64 (pq|rs) chemist integrals
                         with spin-selection rule already enforced by the
                         storage (or zeroed by caller; we apply it again
                         here lazily — see notes).
        n_spinorb:       2L.

    Returns:
        (batch,) float64.

    Spin selection rule: (pq|rs) is zero if spin(p) != spin(q) or
    spin(r) != spin(s). Our caller (``setup_integrals_from_pyscf_h1_eri``)
    already places zeros for spin-forbidden entries, so I2_eri[a, i, j, j]
    is correct directly (no extra masking needed here).
    """
    det = _as_limbs(det_lo, det_hi)
    batch_shape = det.shape[:-1]
    n_limbs = det.shape[-1]
    device = det.device
    dtype = I1.dtype

    # Parity from start=min(i,a) to end=max(i,a)
    start = torch.minimum(i, a)
    end = torch.maximum(i, a)
    sgn = torch.ones(batch_shape, dtype=dtype, device=device)
    sgn = parity_batch(det, None, start, end, sgn)

    # Term 1: I1[a, i]  (per-element gather)
    e1 = I1[a, i]  # (batch,)

    # Term 2: sum over j ∈ det of [I2(a, i, j, j) - I2(a, j, j, i)]
    # We need occ[b, j] for all j, plus the j-conditional integrals at (a, i, j, j) and (a, j, j, i).
    idx_j = torch.arange(n_spinorb, dtype=torch.int64, device=device)
    # occ: (batch, 2L)
    occ = getocc_batch(
        det.unsqueeze(-2).expand(*batch_shape, n_spinorb, n_limbs),
        None,
        idx_j.expand(*batch_shape, n_spinorb),
    ).to(dtype)

    # I2[a, i, j, j] for each batch element b and each j:
    #   shape (batch, 2L) — gather across the 4D tensor
    # Use advanced indexing.
    # I2_aijj[b, j] = I2_eri[a[b], i[b], j, j]
    a_b = a.unsqueeze(-1)  # (batch, 1)
    i_b = i.unsqueeze(-1)  # (batch, 1)
    j_idx = idx_j.unsqueeze(0)  # (1, 2L)
    I2_aijj = I2_eri[a_b, i_b, j_idx, j_idx]  # (batch, 2L)
    I2_ajji = I2_eri[a_b, j_idx, j_idx, i_b]  # (batch, 2L)

    # Summed over occupied j
    e2 = ((I2_aijj - I2_ajji) * occ).sum(dim=-1)  # (batch,)

    return sgn * (e1 + e2)


# ============================================================================
# TwoExcite (double excitation matrix element)
# ============================================================================


def two_excite_batch(
    det_lo: Tensor,
    det_hi: None,
    i: Tensor,
    j: Tensor,
    a: Tensor,
    b: Tensor,
    I2_eri: Tensor,
) -> Tensor:
    """Vectorized double-excitation matrix element.

    SBD-scalar equivalent: ``two_excite(det, i, j, a, b, I2)``.

    Formula (per SBD lines 483-501):
        I = min(i, j); J = max(i, j)
        A = min(a, b); B = max(a, b)
        sgn = parity(det, min(I, A), max(I, A))
        sgn = parity(det, min(J, B), max(J, B), sgn)
        if A > J or B < I: sgn *= -1
        return sgn * (I2.value(A, I, B, J) - I2.value(A, J, B, I))

    Args:
        det_lo: a ``(batch, n_limbs)`` tensor, or a bare ``(batch,)`` tensor
            for a single-limb det. ``det_hi`` must be ``None``.
        det_hi: always ``None`` — kept so a stray ``(det_lo, det_hi)`` pair
            call raises instead of silently doing the wrong thing.
        i, j, a, b:      (batch,) int64 spin-orbital indices.
        I2_eri:          (2L, 2L, 2L, 2L) float64.

    Returns:
        (batch,) float64.
    """
    det = _as_limbs(det_lo, det_hi)
    batch_shape = det.shape[:-1]
    dtype = I2_eri.dtype

    i_lo = torch.minimum(i, j)
    j_hi = torch.maximum(i, j)
    a_lo = torch.minimum(a, b)
    b_hi = torch.maximum(a, b)

    sgn = torch.ones(batch_shape, dtype=dtype, device=det.device)
    sgn = parity_batch(det, None, torch.minimum(i_lo, a_lo), torch.maximum(i_lo, a_lo), sgn)
    sgn = parity_batch(det, None, torch.minimum(j_hi, b_hi), torch.maximum(j_hi, b_hi), sgn)
    # Sign correction when the (i_lo, j_hi) and (a_lo, b_hi) intervals "skip"
    # past each other rather than interleave.
    skip = (a_lo > j_hi) | (b_hi < i_lo)
    sgn = torch.where(skip, -sgn, sgn)

    # Two integrals
    val_aibj = I2_eri[a_lo, i_lo, b_hi, j_hi]
    val_ajbi = I2_eri[a_lo, j_hi, b_hi, i_lo]
    return sgn * (val_aibj - val_ajbi)
