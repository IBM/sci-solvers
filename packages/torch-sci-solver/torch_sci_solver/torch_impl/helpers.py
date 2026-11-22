"""Flat torch tensor helpers for the batched matvec.

The scalar reference stores excitation tables as Python lists of lists:
``singles_from_alpha[ia] = [ja_0, ja_1, ...]``. For vectorized matvec we
need contiguous tensors. We use a CSR-style layout: a single flat tensor
of excitation targets/orbital-indices, plus a length-(M+1) offset array
that demarcates the per-ia ranges.

For each task type (0, 1, 2) we expose a ``TorchHelper`` dataclass
holding *only* the tables that task type uses. The construction is done
once on CPU and the result is shipped to GPU.

Helper tables (for α; β is symmetric with odd CrAn indices):

  singles_from_alpha_offsets: (M_a+1,) int64
      offsets[ia]..offsets[ia+1] is the slice in the flat arrays for
      bra α-string ``ia``.
  singles_from_alpha_to:    (n_α_singles,) int64
      flat array of ket α-string indices.
  singles_alpha_cr:         (n_α_singles,) int64
      flat array of creation spin-orbital indices (2*spatial for α).
  singles_alpha_an:         (n_α_singles,) int64
      flat array of annihilation spin-orbital indices.

Doubles use the same pattern with 4 orbital arrays (cr0, cr1, an0, an1).
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


from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.sbd_ref.helpers import TaskHelper as ScalarTaskHelper

# Largest index an int32 table can hold. String indices stay far below this
# (a basis with 2**31 half-strings would not fit in memory); orbital indices
# span ``2 * n_orb``, comfortably inside the bound for any physical basis.
_MAX_TABLE_INDEX = (1 << 31) - 1


def _narrow_table_indices(values: np.ndarray, table_dtype: torch.dtype, what: str) -> np.ndarray:
    """Cast a flat index array to ``table_dtype`` after checking the int32 bound.

    Offsets and bit masks are never narrowed (entry counts can exceed int32);
    only the ``to``/``cr``/``an`` payload arrays go through here.

    Args:
        values: Integer index array.
        table_dtype: ``torch.int32`` or ``torch.int64``.
        what: Name used in the error message.

    Returns:
        A contiguous array of ``table_dtype``.

    Raises:
        ValueError: unsupported dtype.
        OverflowError: an index does not fit int32.
    """
    if table_dtype == torch.int64:
        return np.ascontiguousarray(values, dtype=np.int64)
    if table_dtype != torch.int32:
        raise ValueError(f"unsupported table dtype {table_dtype}; expected int32 or int64")
    if values.size:
        high = int(values.max())
        low = int(values.min())
        if high > _MAX_TABLE_INDEX or low < -_MAX_TABLE_INDEX - 1:
            raise OverflowError(
                f"{what} index {high if high > _MAX_TABLE_INDEX else low} does not fit "
                "int32; pass table_dtype=torch.int64"
            )
    return np.ascontiguousarray(values, dtype=np.int32)


@dataclass
class TorchHelper:
    """Per-task flat-tensor helper tables.

    Fields not used by ``task_type`` are left as None.
    """

    task_type: int
    n_alpha: int
    n_beta: int
    device: torch.device

    # --- α singles --- (task_type ∈ {0, 2})
    a_singles_offsets: Optional[Tensor] = None  # (M_a+1,) int64
    a_singles_to: Optional[Tensor] = None  # (n_α_singles,) int64
    a_singles_cr: Optional[Tensor] = None  # (n_α_singles,) int64 — spin-orbital
    a_singles_an: Optional[Tensor] = None  # (n_α_singles,) int64

    # --- α doubles --- (task_type == 2)
    a_doubles_offsets: Optional[Tensor] = None  # (M_a+1,) int64
    a_doubles_to: Optional[Tensor] = None  # (n_α_doubles,) int64
    a_doubles_cr0: Optional[Tensor] = None  # (n_α_doubles,) int64
    a_doubles_cr1: Optional[Tensor] = None
    a_doubles_an0: Optional[Tensor] = None
    a_doubles_an1: Optional[Tensor] = None

    # --- β singles --- (task_type ∈ {0, 1})
    b_singles_offsets: Optional[Tensor] = None  # (M_b+1,) int64
    b_singles_to: Optional[Tensor] = None
    b_singles_cr: Optional[Tensor] = None
    b_singles_an: Optional[Tensor] = None

    # --- β doubles --- (task_type == 1)
    b_doubles_offsets: Optional[Tensor] = None
    b_doubles_to: Optional[Tensor] = None
    b_doubles_cr0: Optional[Tensor] = None
    b_doubles_cr1: Optional[Tensor] = None
    b_doubles_an0: Optional[Tensor] = None
    b_doubles_an1: Optional[Tensor] = None

    # --- locality flags ---
    # True when the α (β) bra rows are a local block: the CSR offsets span the
    # owned rows only while ``to`` keeps global ket indices.
    alpha_is_local: bool = False
    beta_is_local: bool = False


def _flatten_singles(
    per_ia_to: List[List[int]],
    per_ia_cran: List[List[tuple]],
    device: torch.device,
    table_dtype: torch.dtype = torch.int64,
):
    """Flatten ragged singles tables to CSR.

    Args:
        per_ia_to:   list-of-lists; per_ia_to[ia] is list of ja indices.
        per_ia_cran: list-of-lists; per_ia_cran[ia][k] is (cr, an) tuple.
        device:      target torch device.
        table_dtype: dtype of the ``to``/``cr``/``an`` payloads (offsets stay int64).

    Returns:
        (offsets, flat_to, flat_cr, flat_an) where offsets is int64 and the
        payloads are ``table_dtype``.
    """
    M = len(per_ia_to)
    lengths = np.fromiter((len(x) for x in per_ia_to), dtype=np.int64, count=M)
    offsets = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    N = int(offsets[-1])

    flat_to = np.empty(N, dtype=np.int64)
    flat_cr = np.empty(N, dtype=np.int64)
    flat_an = np.empty(N, dtype=np.int64)
    pos = 0
    for ia in range(M):
        to_ia = per_ia_to[ia]
        cran_ia = per_ia_cran[ia]
        n = len(to_ia)
        flat_to[pos : pos + n] = to_ia
        for k in range(n):
            flat_cr[pos + k] = cran_ia[k][0]
            flat_an[pos + k] = cran_ia[k][1]
        pos += n

    return (
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(_narrow_table_indices(flat_to, table_dtype, "to")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_cr, table_dtype, "cr")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_an, table_dtype, "an")).to(device),
    )


def _flatten_doubles(
    per_ia_to: List[List[int]],
    per_ia_cran: List[List[tuple]],
    device: torch.device,
    table_dtype: torch.dtype = torch.int64,
):
    """Flatten ragged doubles tables to CSR."""
    M = len(per_ia_to)
    lengths = np.fromiter((len(x) for x in per_ia_to), dtype=np.int64, count=M)
    offsets = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    N = int(offsets[-1])

    flat_to = np.empty(N, dtype=np.int64)
    flat_cr0 = np.empty(N, dtype=np.int64)
    flat_cr1 = np.empty(N, dtype=np.int64)
    flat_an0 = np.empty(N, dtype=np.int64)
    flat_an1 = np.empty(N, dtype=np.int64)
    pos = 0
    for ia in range(M):
        to_ia = per_ia_to[ia]
        cran_ia = per_ia_cran[ia]
        n = len(to_ia)
        flat_to[pos : pos + n] = to_ia
        for k in range(n):
            cr0, cr1, an0, an1 = cran_ia[k]
            flat_cr0[pos + k] = cr0
            flat_cr1[pos + k] = cr1
            flat_an0[pos + k] = an0
            flat_an1[pos + k] = an1
        pos += n

    return (
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(_narrow_table_indices(flat_to, table_dtype, "to")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_cr0, table_dtype, "cr0")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_cr1, table_dtype, "cr1")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_an0, table_dtype, "an0")).to(device),
        torch.from_numpy(_narrow_table_indices(flat_an1, table_dtype, "an1")).to(device),
    )


def torch_helper_from_scalar(
    scalar_helper: ScalarTaskHelper,
    device: torch.device | str = "cpu",
    *,
    table_dtype: torch.dtype = torch.int64,
) -> TorchHelper:
    """Convert a scalar TaskHelper to a flat TorchHelper bundle.

    The scalar helper is built once via ``generate_excitation``; this
    just reshapes its Python lists into contiguous tensors. We never
    touch this on the matvec hot path.

    Args:
        scalar_helper: a fully populated ScalarTaskHelper.
        device:        target torch device for the flat tensors.
        table_dtype:   dtype of the ``to``/``cr``/``an`` payloads; offsets
                       stay int64. Defaults to int64 (the reference build).

    Returns:
        TorchHelper with all relevant CSR fields populated.
    """
    if isinstance(device, str):
        device = torch.device(device)

    out = TorchHelper(
        task_type=scalar_helper.task_type,
        n_alpha=scalar_helper.n_alpha,
        n_beta=scalar_helper.n_beta,
        device=device,
    )

    # α singles for task types 0 and 2
    if scalar_helper.task_type in (0, 2):
        off, to, cr, an = _flatten_singles(
            scalar_helper.singles_from_alpha,
            scalar_helper.singles_alpha_cran,
            device,
            table_dtype=table_dtype,
        )
        out.a_singles_offsets = off
        out.a_singles_to = to
        out.a_singles_cr = cr
        out.a_singles_an = an

    # α doubles only for task type 2
    if scalar_helper.task_type == 2:
        off, to, cr0, cr1, an0, an1 = _flatten_doubles(
            scalar_helper.doubles_from_alpha,
            scalar_helper.doubles_alpha_cran,
            device,
            table_dtype=table_dtype,
        )
        out.a_doubles_offsets = off
        out.a_doubles_to = to
        out.a_doubles_cr0 = cr0
        out.a_doubles_cr1 = cr1
        out.a_doubles_an0 = an0
        out.a_doubles_an1 = an1

    # β singles for task types 0 and 1
    if scalar_helper.task_type in (0, 1):
        off, to, cr, an = _flatten_singles(
            scalar_helper.singles_from_beta,
            scalar_helper.singles_beta_cran,
            device,
            table_dtype=table_dtype,
        )
        out.b_singles_offsets = off
        out.b_singles_to = to
        out.b_singles_cr = cr
        out.b_singles_an = an

    # β doubles only for task type 1
    if scalar_helper.task_type == 1:
        off, to, cr0, cr1, an0, an1 = _flatten_doubles(
            scalar_helper.doubles_from_beta,
            scalar_helper.doubles_beta_cran,
            device,
            table_dtype=table_dtype,
        )
        out.b_doubles_offsets = off
        out.b_doubles_to = to
        out.b_doubles_cr0 = cr0
        out.b_doubles_cr1 = cr1
        out.b_doubles_an0 = an0
        out.b_doubles_an1 = an1

    return out


# ---------------------------------------------------------------------------
# Vectorized (numpy) helper builder
# ---------------------------------------------------------------------------
#
# Builds α/β singles + doubles CSR tables from a per-spin determinant list
# in one vectorized numpy pass:
#
#   1. XOR bra rows in chunks of ``chunk_rows`` against every ket string into
#      a (chunk, M) uint64 matrix — reachable-by-single-or-double filter
#      reads its popcount. The full (M, M) matrix is never materialized.
#   2. SWAR popcount → (chunk, M) int64 Hamming-distance matrix.
#   3. ``np.where(diff == 2)`` extracts the (ia, ja) pairs reachable by a
#      single excitation in row-major (ia-ascending, ja-ascending) order,
#      which is directly the CSR order — no per-ia bucket + sort needed.
#      Chunks are concatenated in row order, preserving that ordering.
#   4. Analogous ``diff == 4`` for doubles.
#   5. For each hit, the excitation orbital positions come from splitting
#      ``a_only = a & ~b`` (bits in bra but not ket) and its counterpart
#      ``b_only`` into their set-bit positions. For singles that's one bit
#      each. For doubles it's two, extracted as (low, high) via the standard
#      ``x & (~x + 1)`` low-bit isolation to get ascending orbital indices.
#
# Bit-identical to the scalar reference in ``torch_sci_solver.sbd_ref.helpers.
# generate_excitation`` — checked at M=1000+ (dim ~ 1M) in
# ``tests/test_torch_helpers.py``.


# --- Utility bitops ------------------------------------------------------- #


# SWAR popcount masks (Wikipedia). All uint64 for numpy elementwise on int64
# XOR matrices reinterpreted as unsigned.
_SWAR_M1 = np.uint64(0x5555555555555555)
_SWAR_M2 = np.uint64(0x3333333333333333)
_SWAR_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_SWAR_H01 = np.uint64(0x0101010101010101)
_SWAR_1 = np.uint64(1)
_SWAR_2 = np.uint64(2)
_SWAR_4 = np.uint64(4)
_SWAR_56 = np.uint64(56)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """Vectorized popcount for a uint64 numpy array (any shape).

    Classical SWAR (SIMD-Within-A-Register) implementation — same instruction
    sequence a C compiler emits before POPCNT. Fully vectorized across the
    input array; no lookup table, no Python loop over elements.

    Args:
        x: uint64 array (or view thereof). Not mutated.

    Returns:
        int64 array of the same shape, with values in ``[0, 64]``.
    """
    x = x.astype(np.uint64, copy=True)
    x -= (x >> _SWAR_1) & _SWAR_M1
    x = (x & _SWAR_M2) + ((x >> _SWAR_2) & _SWAR_M2)
    x = (x + (x >> _SWAR_4)) & _SWAR_M4
    return ((x * _SWAR_H01) >> _SWAR_56).astype(np.int64)


def _log2_single_bit_u64(x: np.ndarray) -> np.ndarray:
    """Position (0..63) of the single set bit in each element of ``x``.

    Requires every element of ``x`` to be a power of two (exactly one bit
    set). Uses float64 log2: an exact power of two is exactly representable,
    and its log2 is the integer exponent, so the result is exact for every
    bit position a limb can hold.

    Args:
        x: uint64 array, each entry a power of two.

    Returns:
        int64 array of bit positions, same shape as ``x``.
    """
    return np.log2(x.astype(np.float64)).astype(np.int64)


# --- Multi-limb string representation -------------------------------------- #

_LIMB_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _n_limbs(n_bits: int) -> int:
    """Number of 64-bit limbs needed to hold an ``n_bits``-bit string."""
    return max(1, (int(n_bits) + 63) // 64)


def _string_limb_mask(n_orb: int, n_limbs: int) -> np.ndarray:
    """``(n_limbs,)`` uint64 mask keeping only the ``n_orb`` valid bits."""
    mask = np.full(n_limbs, _LIMB_MASK, dtype=np.uint64)
    bits = int(n_orb) - 64 * (n_limbs - 1)
    if bits <= 0:
        mask[-1] = np.uint64(0)
    elif bits < 64:
        mask[-1] = np.uint64((1 << bits) - 1)
    return mask


def _as_string_limbs(strings, n_orb: int) -> np.ndarray:
    """Normalize determinant strings to a ``(M, n_limbs)`` uint64 limb matrix.

    Determinant strings wider than 64 bits (``n_orb > 64``) are carried as
    limbs: limb ``k`` holds bits ``[64k, 64k + 64)`` of the string, limb 0 the
    lowest. A caller may pass

    - a sequence of Python ints of any width (split here),
    - a 1-D integer array (its bit pattern; bits above ``n_orb`` are masked),
    - an already-split ``(M, n_limbs)`` integer array.

    Args:
        strings: the determinant strings, in any of the forms above.
        n_orb: number of spatial orbitals (the valid bit count).

    Returns:
        ``(M, n_limbs)`` uint64 array with every bit above ``n_orb`` cleared.

    Raises:
        ValueError: an unexpected shape, a negative Python integer, or a
            limb matrix with the wrong number of limbs.
    """
    n_limbs = _n_limbs(n_orb)
    arr = np.asarray(strings)
    if arr.dtype.kind == "f" and not isinstance(strings, np.ndarray):
        # A list holding a value >= 2**63 infers float64; keep the exact ints
        # (a list of genuine floats still fails the integer check below).
        exact = np.asarray(strings, dtype=object)
        if all(isinstance(v, (int, np.integer)) for v in exact.ravel().tolist()):
            arr = exact
    if arr.size == 0:
        return np.zeros((0, n_limbs), dtype=np.uint64)

    if arr.ndim == 2:
        if arr.shape[1] != n_limbs:
            raise ValueError(
                f"limb matrix has {arr.shape[1]} limbs, expected {n_limbs} for n_orb={n_orb}"
            )
        if arr.dtype.kind not in "iu":
            raise TypeError(f"determinant strings must be integers, got {arr.dtype}")
        # An unsigned cast of a signed array keeps the bit pattern, so the
        # existing signed-int64 convention (top limb bit set → negative) works.
        limbs = np.array(arr, dtype=np.uint64, copy=True)
    elif arr.ndim == 1:
        if arr.dtype == object:
            values = [int(v) for v in arr]
            if any(v < 0 for v in values):
                raise ValueError("determinant strings must be non-negative")
            limbs = np.zeros((len(values), n_limbs), dtype=np.uint64)
            for k in range(n_limbs):
                limbs[:, k] = [(v >> (64 * k)) & int(_LIMB_MASK) for v in values]
        elif arr.dtype.kind in "iu":
            # A 1-D input carries only the low limb; bits above n_orb are
            # masked below, so an n_orb > 64 basis with high bits clear is fine.
            limbs = np.array(arr, dtype=np.uint64, copy=True).reshape(-1, 1)
        else:
            raise TypeError(f"determinant strings must be integers, got {arr.dtype}")
    else:
        raise ValueError(f"determinant strings must be 1-D or 2-D, got ndim={arr.ndim}")
    limbs &= _string_limb_mask(n_orb, limbs.shape[1])
    return limbs


def _single_bit_positions(limbs: np.ndarray) -> np.ndarray:
    """Absolute bit position of the single set bit in each row of ``(n, limbs)``."""
    rows = np.arange(limbs.shape[0])
    nonzero = limbs != 0
    limb_idx = nonzero.argmax(axis=1)
    word = limbs[rows, limb_idx]
    return limb_idx.astype(np.int64) * 64 + _log2_single_bit_u64(word)


def _two_bit_positions(limbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ascending absolute positions of the two set bits in each row."""
    rows = np.arange(limbs.shape[0])
    nonzero = limbs != 0
    limb0 = nonzero.argmax(axis=1)
    word0 = limbs[rows, limb0]
    low0 = word0 & ((~word0) + np.uint64(1))  # (~x) + 1 == -x on uint64
    rest = limbs.copy()
    rest[rows, limb0] = word0 ^ low0
    limb1 = (rest != 0).argmax(axis=1)
    pos0 = limb0.astype(np.int64) * 64 + _log2_single_bit_u64(low0)
    pos1 = limb1.astype(np.int64) * 64 + _log2_single_bit_u64(rest[rows, limb1])
    return pos0, pos1


# --- Vectorized per-spin table builders ----------------------------------- #


def _pairwise_diff(
    bra_limbs: np.ndarray, n_orb: int, kets_limbs: np.ndarray | None = None
) -> np.ndarray:
    """Return the pairwise int64 Hamming-distance matrix over the low ``n_orb`` bits.

    Strings are ``(rows, n_limbs)`` limb matrices; the distance is the
    limb-wise XOR popcount summed over limbs, so the same code serves one
    limb (``n_orb <= 64``) and any wider string.

    With ``kets_limbs=None`` this is the full ``(M, M)`` matrix. Passing an
    explicit ket set computes a ``(K, N)`` bra block against all strings,
    which is how the row-chunked builders avoid materializing ``(M, M)``.

    Args:
        bra_limbs: (M, n_limbs) or (K, n_limbs) uint64 bra limb matrix.
        n_orb: number of spatial orbitals — only bits ``[0, n_orb)`` count.
        kets_limbs: optional (N, n_limbs) uint64 ket limb matrix;
            defaults to ``bra_limbs`` (the symmetric full-matrix case).

    Returns:
        (M, M) or (K, N) int64 matrix of pairwise Hamming distances, values
        in ``[0, 2 * n_orb]``. Zero on the diagonal of the symmetric case.
    """
    key = bra_limbs if kets_limbs is None else kets_limbs
    if key.shape[1] != bra_limbs.shape[1]:
        raise ValueError(f"bra and ket limb counts differ: {bra_limbs.shape[1]} vs {key.shape[1]}")
    mask = _string_limb_mask(n_orb, bra_limbs.shape[1])
    xor = (bra_limbs[:, None, :] ^ key[None, :, :]) & mask  # (K, N, n_limbs)
    return _popcount_u64(xor).sum(axis=-1)


def _chunked_hits(
    bra_limbs: np.ndarray, ket_limbs: np.ndarray, n_orb: int, chunk_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-chunked extraction of all singles/doubles ``(ia, ja)`` hits.

    Process ``chunk_rows`` bra rows at a time against the full ket set, so
    the only Hamming-distance temporary is ``(chunk_rows, len(ket), n_limbs)``.
    Hits are concatenated in ascending ``(ia, ja)`` order — row-major, which
    is directly the CSR order the scalar path appends in.

    Args:
        bra_limbs: (K, n_limbs) uint64 bra limb matrix.
        ket_limbs: (N, n_limbs) uint64 ket limb matrix.
        n_orb: number of spatial orbitals.
        chunk_rows: number of bra rows per diff chunk.

    Returns:
        ``(singles_ia, singles_ja, doubles_ia, doubles_ja)`` int64 arrays.
        ``ia`` indexes ``bra_limbs``; ``ja`` indexes ``ket_limbs``.
    """
    K = int(bra_limbs.shape[0])
    # Seed with empty int64 arrays so the concatenation is well-defined for
    # an empty bra block (K == 0) and keeps the dtype fixed.
    s_ia: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    s_ja: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    d_ia: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    d_ja: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    for c0 in range(0, K, chunk_rows):
        c1 = min(c0 + chunk_rows, K)
        diff = _pairwise_diff(bra_limbs[c0:c1], n_orb, kets_limbs=ket_limbs)
        ia_s, ja_s = np.where(diff == 2)
        s_ia.append(ia_s + np.int64(c0))
        s_ja.append(ja_s)
        ia_d, ja_d = np.where(diff == 4)
        d_ia.append(ia_d + np.int64(c0))
        d_ja.append(ja_d)
    return (
        np.concatenate(s_ia).astype(np.int64, copy=False),
        np.concatenate(s_ja).astype(np.int64, copy=False),
        np.concatenate(d_ia).astype(np.int64, copy=False),
        np.concatenate(d_ja).astype(np.int64, copy=False),
    )


def _build_singles_flat_vec(
    bra_limbs: np.ndarray,
    ket_limbs: np.ndarray,
    ia: np.ndarray,
    ja: np.ndarray,
    n_orb: int,
    spin_shift: int,
    device: torch.device,
    *,
    table_dtype: torch.dtype = torch.int64,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build flat CSR tensors for the singles table of one spin sector.

    Args:
        bra_limbs: (K, n_limbs) uint64 bra limb matrix.
        ket_limbs: (N, n_limbs) uint64 ket limb matrix.
        ia: (n_hits,) int64 bra indices of the Δ=2 hits, ascending.
        ja: (n_hits,) int64 ket indices of the Δ=2 hits, ascending per row.
        n_orb: number of spatial orbitals; only bits ``0..n_orb-1`` are
            considered (matches ``difference``'s mask semantics).
        spin_shift: 0 for α (spin-orbital = ``2 * spatial``), 1 for β
            (``2 * spatial + 1``). Matches the ``2 * cr[0]`` / ``2 * cr[0] + 1``
            constants in the scalar path.
        device: target torch device for the returned tensors.
        table_dtype: dtype of the ``to``/``cr``/``an`` payloads; offsets stay
            int64. Defaults to int64 (the reference build).

    Returns:
        ``(offsets, flat_to, flat_cr, flat_an)`` matching the scalar path's
        CSR layout; offsets are int64 and the payloads ``table_dtype``.
        ``offsets`` has shape ``(K + 1,)``, so the same builder serves the
        full table (``bra_limbs is ket_limbs``) and a row-local α block.
    """
    M = int(bra_limbs.shape[0])
    mask = _string_limb_mask(n_orb, bra_limbs.shape[1])

    if ia.size == 0:
        empty = torch.zeros(0, dtype=table_dtype, device=device)
        return (
            torch.zeros(M + 1, dtype=torch.int64, device=device),
            empty,
            empty,
            empty,
        )

    a_only = (bra_limbs[ia] & (~ket_limbs[ja])) & mask  # bits set in bra not ket
    b_only = (ket_limbs[ja] & (~bra_limbs[ia])) & mask  # bits set in ket not bra
    # Both single-bit (across all limbs) for a single excitation.
    cr_pos = _single_bit_positions(a_only)  # spatial orbital index in [0, norb)
    an_pos = _single_bit_positions(b_only)
    # α: spin-orbital = 2 * spatial; β: 2 * spatial + 1.
    cr_so = (cr_pos << np.int64(1)) | np.int64(spin_shift)
    an_so = (an_pos << np.int64(1)) | np.int64(spin_shift)

    # CSR offsets: cumsum of the per-bra count.
    counts = np.bincount(ia, minlength=M).astype(np.int64)  # (M,)
    offsets = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return (
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(_narrow_table_indices(ja, table_dtype, "to")).to(device),
        torch.from_numpy(_narrow_table_indices(cr_so, table_dtype, "cr")).to(device),
        torch.from_numpy(_narrow_table_indices(an_so, table_dtype, "an")).to(device),
    )


def _build_doubles_flat_vec(
    bra_limbs: np.ndarray,
    ket_limbs: np.ndarray,
    ia: np.ndarray,
    ja: np.ndarray,
    n_orb: int,
    spin_shift: int,
    device: torch.device,
    *,
    table_dtype: torch.dtype = torch.int64,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build flat CSR tensors for the doubles table of one spin sector.

    Same structure as :func:`_build_singles_flat_vec`, but selects pairs
    with Hamming distance 4 and returns two cr/an columns per pair. The
    per-pair orbital positions are extracted in ascending order (low-bit
    isolation per limb, then the next nonzero limb), matching
    ``orbital_difference``'s ascending-index convention.

    Args:
        bra_limbs: (K, n_limbs) uint64 bra limb matrix.
        ket_limbs: (N, n_limbs) uint64 ket limb matrix.
        ia: (n_hits,) int64 bra indices of the Δ=4 hits, ascending.
        ja: (n_hits,) int64 ket indices of the Δ=4 hits, ascending per row.
        n_orb: number of spatial orbitals.
        spin_shift: 0 for α, 1 for β.
        device: target torch device.

    Returns:
        ``(offsets, flat_to, flat_cr0, flat_cr1, flat_an0, flat_an1)`` — int64
        tensors matching the scalar doubles CSR layout. ``cr0 < cr1`` and
        ``an0 < an1`` per row, and ``offsets`` has shape ``(K + 1,)``.
    """
    M = int(bra_limbs.shape[0])
    mask = _string_limb_mask(n_orb, bra_limbs.shape[1])

    if ia.size == 0:
        empty = torch.zeros(0, dtype=table_dtype, device=device)
        return (
            torch.zeros(M + 1, dtype=torch.int64, device=device),
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    a_only = (bra_limbs[ia] & (~ket_limbs[ja])) & mask  # 2 bits set
    b_only = (ket_limbs[ja] & (~bra_limbs[ia])) & mask  # 2 bits set
    cr0_pos, cr1_pos = _two_bit_positions(a_only)
    an0_pos, an1_pos = _two_bit_positions(b_only)
    cr0_so = (cr0_pos << np.int64(1)) | np.int64(spin_shift)
    cr1_so = (cr1_pos << np.int64(1)) | np.int64(spin_shift)
    an0_so = (an0_pos << np.int64(1)) | np.int64(spin_shift)
    an1_so = (an1_pos << np.int64(1)) | np.int64(spin_shift)

    counts = np.bincount(ia, minlength=M).astype(np.int64)
    offsets = np.zeros(M + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return (
        torch.from_numpy(offsets).to(device),
        torch.from_numpy(_narrow_table_indices(ja, table_dtype, "to")).to(device),
        torch.from_numpy(_narrow_table_indices(cr0_so, table_dtype, "cr0")).to(device),
        torch.from_numpy(_narrow_table_indices(cr1_so, table_dtype, "cr1")).to(device),
        torch.from_numpy(_narrow_table_indices(an0_so, table_dtype, "an0")).to(device),
        torch.from_numpy(_narrow_table_indices(an1_so, table_dtype, "an1")).to(device),
    )


def _build_spin_tables_vec(
    dets_limbs: np.ndarray,
    n_orb: int,
    spin_shift: int,
    device: torch.device,
    chunk_rows: int,
    *,
    row_start: int = 0,
    row_stop: Optional[int] = None,
    table_dtype: torch.dtype = torch.int64,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
]:
    """Build the singles and doubles CSR tables of one spin sector.

    The bra rows are ``dets_limbs[row_start:row_stop]`` and the ket set is
    the full ``dets_limbs``, so a local bra block keeps **global** ket
    indices while its CSR offsets span the owned rows only. The default range
    builds the full table. Rows are diffed chunk-wise by
    :func:`_chunked_hits`.

    Args:
        dets_limbs: (M, n_limbs) uint64 limb matrix (see
            :func:`_as_string_limbs`).
        n_orb: number of spatial orbitals.
        spin_shift: 0 for α, 1 for β.
        device: target torch device.
        chunk_rows: number of bra rows per diff chunk.
        row_start: first bra row (inclusive).
        row_stop: one past the last bra row; ``None`` means ``M``.
        table_dtype: dtype of the ``to``/``cr``/``an`` payloads; offsets stay
            int64. Defaults to int64 (the reference build).

    Returns:
        ``(singles, doubles)`` flat CSR tuples, as returned by
        :func:`_build_singles_flat_vec` / :func:`_build_doubles_flat_vec`.
    """
    stop = int(dets_limbs.shape[0]) if row_stop is None else int(row_stop)
    bra_limbs = dets_limbs[row_start:stop]
    s_ia, s_ja, d_ia, d_ja = _chunked_hits(bra_limbs, dets_limbs, n_orb, chunk_rows)
    singles = _build_singles_flat_vec(
        bra_limbs,
        dets_limbs,
        s_ia,
        s_ja,
        n_orb,
        spin_shift=spin_shift,
        device=device,
        table_dtype=table_dtype,
    )
    del s_ia, s_ja
    doubles = _build_doubles_flat_vec(
        bra_limbs,
        dets_limbs,
        d_ia,
        d_ja,
        n_orb,
        spin_shift=spin_shift,
        device=device,
        table_dtype=table_dtype,
    )
    del d_ia, d_ja
    return singles, doubles


def _make_task_helpers(
    a_singles: tuple[Tensor, Tensor, Tensor, Tensor],
    a_doubles: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    b_singles: tuple[Tensor, Tensor, Tensor, Tensor],
    b_doubles: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    *,
    n_alpha: int,
    n_alpha_task1: int,
    n_beta: int,
    device: torch.device,
    alpha_is_local: bool = False,
    beta_is_local: bool = False,
) -> tuple[TorchHelper, TorchHelper, TorchHelper]:
    """Assemble the (task 2, task 1, task 0) helpers from flat CSR tuples.

    ``n_alpha`` is the row count the α singles/doubles offsets span (the
    local row count for :func:`generate_local_torch_helpers`); task 1 reports
    ``n_alpha_task1`` because its α axis is not tabulated. ``alpha_is_local``
    / ``beta_is_local`` record whether the respective bra rows are a local
    block (offsets local, ``to`` global).
    """
    a_s_off, a_s_to, a_s_cr, a_s_an = a_singles
    a_d_off, a_d_to, a_d_cr0, a_d_cr1, a_d_an0, a_d_an1 = a_doubles
    b_s_off, b_s_to, b_s_cr, b_s_an = b_singles
    b_d_off, b_d_to, b_d_cr0, b_d_cr1, b_d_an0, b_d_an1 = b_doubles

    task2 = TorchHelper(
        task_type=2,
        n_alpha=n_alpha,
        n_beta=n_beta,
        device=device,
        alpha_is_local=alpha_is_local,
        beta_is_local=beta_is_local,
        a_singles_offsets=a_s_off,
        a_singles_to=a_s_to,
        a_singles_cr=a_s_cr,
        a_singles_an=a_s_an,
        a_doubles_offsets=a_d_off,
        a_doubles_to=a_d_to,
        a_doubles_cr0=a_d_cr0,
        a_doubles_cr1=a_d_cr1,
        a_doubles_an0=a_d_an0,
        a_doubles_an1=a_d_an1,
    )
    task1 = TorchHelper(
        task_type=1,
        n_alpha=n_alpha_task1,
        n_beta=n_beta,
        device=device,
        beta_is_local=beta_is_local,
        b_singles_offsets=b_s_off,
        b_singles_to=b_s_to,
        b_singles_cr=b_s_cr,
        b_singles_an=b_s_an,
        b_doubles_offsets=b_d_off,
        b_doubles_to=b_d_to,
        b_doubles_cr0=b_d_cr0,
        b_doubles_cr1=b_d_cr1,
        b_doubles_an0=b_d_an0,
        b_doubles_an1=b_d_an1,
    )
    task0 = TorchHelper(
        task_type=0,
        n_alpha=n_alpha,
        n_beta=n_beta,
        device=device,
        alpha_is_local=alpha_is_local,
        beta_is_local=beta_is_local,
        a_singles_offsets=a_s_off,
        a_singles_to=a_s_to,
        a_singles_cr=a_s_cr,
        a_singles_an=a_s_an,
        b_singles_offsets=b_s_off,
        b_singles_to=b_s_to,
        b_singles_cr=b_s_cr,
        b_singles_an=b_s_an,
    )
    return task2, task1, task0


def generate_all_torch_helpers(
    adets,
    bdets,
    n_orb: int,
    device: torch.device | str = "cpu",
    *,
    chunk_rows: int = 1024,
    table_dtype: torch.dtype = torch.int64,
) -> tuple[TorchHelper, TorchHelper, TorchHelper]:
    """Build all three task helpers as TorchHelpers in one call.

    Returns helpers in SBD scheduling order: (task 2, task 1, task 0).

    Builds α-singles, α-doubles, β-singles, β-doubles once each — the α/β
    singles arrays are shared between task 2 / task 0 (α) and task 1 /
    task 0 (β), matching SBD's own reuse pattern across task_types.

    Numpy-vectorized (see the module-level notes above), processing bra rows
    in chunks of ``chunk_rows`` so the ``(M, M)`` Hamming-distance matrix is
    never materialized: the peak host temporary is O(chunk_rows * M). The
    output is bit-identical to the unchunked builder for every chunk size
    (``chunk_rows >= M`` is a single chunk — the old full-matrix case).
    ``table_dtype`` selects the index payload dtype (int32 halves the table
    memory; the bounds are asserted, and offsets stay int64). The strings may
    be Python ints of any width, 1-D bit patterns, or ``(M, n_limbs)`` limb
    matrices (see :func:`_as_string_limbs`), so one call serves ``n_orb <= 64``
    and wider strings alike.
    Bit-identical to the scalar reference in :func:`torch_sci_solver.sbd_ref.
    helpers.generate_excitation`; checked at M=252 in
    ``tests/test_torch_helpers.py``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    adets_np = _as_string_limbs(adets, n_orb)
    bdets_np = _as_string_limbs(bdets, n_orb)
    M_a = int(adets_np.shape[0])
    M_b = int(bdets_np.shape[0])

    # α-side: singles used by tasks 0 and 2; doubles used by task 2 only.
    a_singles, a_doubles = _build_spin_tables_vec(
        adets_np, n_orb, 0, device, chunk_rows, table_dtype=table_dtype
    )
    # β-side: singles used by tasks 0 and 1; doubles used by task 1 only.
    b_singles, b_doubles = _build_spin_tables_vec(
        bdets_np, n_orb, 1, device, chunk_rows, table_dtype=table_dtype
    )

    return _make_task_helpers(
        a_singles,
        a_doubles,
        b_singles,
        b_doubles,
        n_alpha=M_a,
        n_alpha_task1=M_a,
        n_beta=M_b,
        device=device,
    )


def generate_local_torch_helpers(
    adets,
    bdets,
    n_orb: int,
    row_start: int,
    row_stop: int,
    device: torch.device | str = "cpu",
    *,
    chunk_rows: int = 1024,
    col_start: Optional[int] = None,
    col_stop: Optional[int] = None,
    table_dtype: torch.dtype = torch.int64,
) -> tuple[TorchHelper, TorchHelper, TorchHelper]:
    """Helpers for the owner-computes row block, in (task 2, task 1, task 0) order.

    The α singles/doubles tables contain only the owned bra rows
    ``[row_start, row_stop)``. Their CSR offsets are local — length
    ``local_rows + 1`` with bra indices ``0..local_rows-1`` — while ``to``
    keeps global α-ket indices. The β tables are full by default (a 1-D
    partition owns all β columns); passing ``col_start``/``col_stop``
    restricts them to the owned β rows in the same way (local offsets,
    global ``to``), which is what a 2-D tile partition needs. ``n_alpha`` is
    the local row count on task 2 / task 0 and the global ``M_a`` on task 1
    (which tabulates no α excitations); ``n_beta`` is the tabulated β row
    count.

    Both sides are diffed chunk-wise — local bra rows against all strings —
    so the full ``(M, M)`` matrix is never materialized. The strings accept
    the same forms as :func:`generate_all_torch_helpers` (Python ints, 1-D
    bit patterns or ``(M, n_limbs)`` limb matrices).

    Args:
        adets: full α-string list (same ordering on every rank).
        bdets: full β-string list.
        n_orb: number of spatial orbitals.
        row_start: first owned α row (inclusive).
        row_stop: one past the last owned α row.
        device: target torch device.
        chunk_rows: bra rows per diff chunk.
        col_start: first owned β row (inclusive); ``None`` keeps the full table.
        col_stop: one past the last owned β row; ``None`` keeps the full table.
        table_dtype: dtype of the ``to``/``cr``/``an`` payloads; offsets stay
            int64. Defaults to int64 (the reference build).

    Returns:
        ``(task2, task1, task0)`` helpers. Slicing the full builder's α CSR
        by ``[offset[row_start], offset[row_stop])`` reproduces these α
        tables; the β tables equal the full builder's when no column range is
        given, and its ``[col_start, col_stop)`` slice otherwise.
    """
    if isinstance(device, str):
        device = torch.device(device)

    adets_np = _as_string_limbs(adets, n_orb)
    bdets_np = _as_string_limbs(bdets, n_orb)
    M_a = int(adets_np.shape[0])
    M_b = int(bdets_np.shape[0])

    local_adets = adets_np[row_start:row_stop]
    a_s_ia, a_s_ja, a_d_ia, a_d_ja = _chunked_hits(local_adets, adets_np, n_orb, chunk_rows)
    a_singles = _build_singles_flat_vec(
        local_adets,
        adets_np,
        a_s_ia,
        a_s_ja,
        n_orb,
        spin_shift=0,
        device=device,
        table_dtype=table_dtype,
    )
    del a_s_ia, a_s_ja
    a_doubles = _build_doubles_flat_vec(
        local_adets,
        adets_np,
        a_d_ia,
        a_d_ja,
        n_orb,
        spin_shift=0,
        device=device,
        table_dtype=table_dtype,
    )
    del a_d_ia, a_d_ja

    beta_is_local = col_start is not None or col_stop is not None
    if beta_is_local:
        b_start = 0 if col_start is None else int(col_start)
        b_stop = M_b if col_stop is None else int(col_stop)
    else:
        b_start, b_stop = 0, M_b
    b_singles, b_doubles = _build_spin_tables_vec(
        bdets_np,
        n_orb,
        1,
        device,
        chunk_rows,
        row_start=b_start,
        row_stop=b_stop,
        table_dtype=table_dtype,
    )

    return _make_task_helpers(
        a_singles,
        a_doubles,
        b_singles,
        b_doubles,
        n_alpha=row_stop - row_start,
        n_alpha_task1=M_a,
        n_beta=b_stop - b_start,
        device=device,
        alpha_is_local=True,
        beta_is_local=beta_is_local,
    )
