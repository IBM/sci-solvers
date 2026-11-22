"""Entry-parallel Triton kernel for the streamed doubles matvec passes.

Two matvec paths stream flat entry tables instead of grouped excitation
slices: the flat spin-orbital ("spinless") matvec
(:mod:`~torch_sci_solver.torch_impl.mult_spinless`) and the streamed
(non-block) GDB pair matvec (:mod:`~torch_sci_solver.torch_impl.mult_pairs`).
Both walk rows of ``(k_bra, k_ket, cr0, cr1, an0, an1)``, evaluate
:func:`~torch_sci_solver.torch_impl.determinants.two_excite_batch`, multiply by
``T[k_ket]`` and scatter with ``index_add_``. The output-stationary kernels of
:mod:`~torch_sci_solver.torch_impl.mult_triton` own a block of outputs and walk
their excitation lists, so they cannot serve a table whose rows share no
destination grouping.

This module fuses the excitation matrix element with the
gather-scale-scatter in one entry-parallel kernel: one lane per table entry,
one ``atomic_add`` into ``Wb[k_bra]``. Both passes are tiled: the doubles
evaluate
:func:`~torch_sci_solver.torch_impl.determinants.two_excite_batch`, and the
singles evaluate
:func:`~torch_sci_solver.torch_impl.determinants.one_excite_batch`, whose
occupied-set sum runs over 64-orbital blocks inside the kernel.

The kernel accepts the same storage layouts as the batched scatters it
replaces:

* **ERI.** A dense chemist-order ``(2L, 2L, 2L, 2L)`` tensor, or the compact
  spatial ``(L, L, L, L)`` store of
  :class:`~torch_sci_solver.torch_impl.integrals.CompactEri` (the
  ``to_torch_integrals(..., compact_eri=True)`` form). For the compact store
  every spin-orbital index decodes into ``(spatial, spin)``, the element is
  the spatial entry selected by the spin pattern, and a chemist pair whose
  two spins differ contributes zero --- exactly the wrapper's indexing rule.
* **Determinant strings.** Single-word ``(M,)`` tensors or ``(M, n_limbs)``
  int64 limb matrices (limb ``k`` holds bits ``[64k, 64k + 64)``), with the
  range-popcount/parity decomposition of
  :mod:`~torch_sci_solver.torch_impl.mult_triton`: flat spin-orbital ranges
  for the spinless mode and the α-even/β-odd split for the pair mode.
* **Input block.** A 1-D ``(M,)`` vector or an ``(M, K)`` block for ``T``
  (and ``Wb``); the block path loops the columns and issues one atomic add
  per column. The batched spinless scatter takes blocks too (its ``_index_add``
  lifts the element across columns), so the kernel replaces it directly; the
  streamed pair scatter is vector-only and the block form extends coverage.

These forms compose: a complex compact store is read through its interleaved
real view and the string tables stay int64, so one launch covers every
combination. No combination falls back to a silently different element
evaluation --- a layout the kernel cannot evaluate is rejected by the
predicate before launching.

Dispatch is a predicate, not an error: :func:`apply_double_entries` and
:func:`apply_single_entries` return ``False`` when Triton is missing, the
device is not CUDA, the dtype is outside the supported set, or the
tables/strings/ERI have a layout the kernel cannot represent, so the callers
keep their batched scatter and stay correct.
The kernel is written from the :func:`two_excite_batch` formula (see its
docstring), evaluated on the determinant carried by the entry table (the ket
det for the spinless mode, the bra pair's strings for the pair mode).

Complex dtypes (``complex64``/``complex128``) run through the same kernel.
Triton has no complex scalar type, so every complex operand is passed as its
interleaved ``(re, im)`` real view --- exactly torch's complex memory layout
--- and an ``IS_COMPLEX`` constexpr selects the pair arithmetic, as in
:mod:`~torch_sci_solver.torch_impl.mult_triton`. Arithmetic stays in
``float64`` for both complex precisions; only the loads and the atomic store
use the storage precision. The element is evaluated in the same index order as
the batched ``two_excite_batch`` call (creation index first), so a complex
Hermitian ERI needs no conjugation correction.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import Tensor

from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_triton import _kernel_complex_args

try:  # Triton is optional and CUDA-only; the package must import without it.
    import triton
    import triton.language as tl

    from torch_sci_solver.torch_impl.mult_triton import _range_parity

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

# Source-of-truth descriptor of which passes this build runs entry-tiled for
# the two streamed matvec layouts. Lives next to the kernel so a result file
# can self-identify its backend split and the table cannot drift. Both kernels
# cover dense and compact ERIs, single-word and limb-matrix strings, and 1-D or
# ``(M, K)`` block inputs, so these entries do not depend on the storage
# layout the caller chose.
TRITON_ENTRY_PASS_BACKENDS = {
    "spinless_singles": "tiled",
    "spinless_doubles": "tiled",
    "gdb_streamed_singles": "tiled",
    "gdb_streamed_same_spin_doubles": "tiled",
    "gdb_streamed_alphabeta_doubles": "tiled",
}

# Entries per program. Entries are independent (unlike the output-stationary
# kernels' rows), so a plain lane-block is enough.
_BLOCK = 128

# Spin-orbitals per block of the singles kernel's occupied-set sum. One block
# spans one int64 limb of a flat string, or 32 α plus 32 β bits of an
# interleaved α/β string.
_J_BLOCK = 64

_MODE_CODES = {"spinless": 0, "pair": 1}

_SUPPORTED_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


if _HAVE_TRITON:

    @triton.jit
    def _popc64(x):  # SWAR popcount, vectorized over the lane block
        """Popc64."""
        m1 = tl.full((1,), 0x5555555555555555, tl.int64)
        m2 = tl.full((1,), 0x3333333333333333, tl.int64)
        m4 = tl.full((1,), 0x0F0F0F0F0F0F0F0F, tl.int64)
        h01 = tl.full((1,), 0x0101010101010101, tl.int64)
        x = x - ((x >> 1) & m1)
        x = (x & m2) + ((x >> 2) & m2)
        x = (x + (x >> 4)) & m4
        return (x * h01) >> 56

    @triton.jit
    def _mask_below64(n):
        """Low-``n`` bits of one int64 word, all ones for ``n >= 64``."""
        one = tl.full((1,), 1, tl.int64)
        n = tl.minimum(tl.maximum(n, 0), 64)
        return tl.where(n >= 64, -one, (one << n) - one)

    @triton.jit
    def _flat_range_parity(det, lo, hi):
        """Parity of the flat spin-orbital range ``[lo, hi)`` of one det word.

        Mirrors ``parity_batch``: the popcount of the bits in ``[lo, hi)`` plus
        the bit at ``lo`` itself. Spinless determinants have no α/β split, so
        ``lo``/``hi`` are spin-orbital positions directly.
        """
        mask = _mask_below64(hi) ^ _mask_below64(lo)
        pop = _popc64(det & mask)
        return pop + ((det >> lo) & 1)

    @triton.jit
    def _flat_range_parity_limbs(dets_ptr, rows, lo, hi, active, N_LIMBS: tl.constexpr):
        """Parity of the flat spin-orbital range ``[lo, hi)`` of a limb-matrix det.

        Limb ``k`` holds bits ``[64k, 64k + 64)``, so it contributes the part
        of the range clipped to its window; the bit-at-start term comes from
        the one limb holding the global position ``lo``. This is the
        limb-matrix form of :func:`_flat_range_parity`, matching
        ``parity_batch``'s per-limb popcount plus ``getocc`` at ``lo``.
        """
        total = lo * 0
        bit = lo & 63
        for k in tl.static_range(0, N_LIMBS):
            off = k * 64
            word = tl.load(dets_ptr + rows * N_LIMBS + k, mask=active, other=0)
            mask = _mask_below64(hi - off) ^ _mask_below64(lo - off)
            total += _popc64(word & mask)
            total += tl.where((lo >> 6) == k, (word >> bit) & 1, 0)
        return total

    @triton.jit
    def _range_parity_word(lo, hi, a_str, b_str):
        """Parity of the spin-orbital range ``[lo, hi)`` of an interleaved det.

        Same decomposition as ``mult_triton._range_parity`` for single-word
        strings: α sits at even spin-orbital positions and β at odd ones, so
        the α range is ``[(lo+1)//2, (hi+1)//2)`` and the β range
        ``[lo//2, hi//2)``. The bit-at-start term reads the α string for an
        even ``lo`` and the β string for an odd one.
        """
        a_lo = (lo + 1) // 2
        a_hi = (hi + 1) // 2
        b_lo = lo // 2
        b_hi = hi // 2
        shift = lo // 2
        a_mask = _mask_below64(a_hi) ^ _mask_below64(a_lo)
        b_mask = _mask_below64(b_hi) ^ _mask_below64(b_lo)
        pa = _popc64(a_str & a_mask)
        pb = _popc64(b_str & b_mask)
        a_bit = (a_str >> shift) & 1
        b_bit = (b_str >> shift) & 1
        return pa + pb + tl.where((lo & 1) == 0, a_bit, b_bit)

    @triton.jit
    def _double_entries_kernel(
        I2_ptr,
        T_ptr,
        Wb_ptr,
        k_bra_ptr,
        k_ket_ptr,
        cr0_ptr,
        cr1_ptr,
        an0_ptr,
        an1_ptr,
        dets_ptr,
        alpha_ptr,
        beta_ptr,
        n_spinorb,
        n_spatial,
        n,
        offset,
        n_cols,
        T_stride,
        MODE: tl.constexpr,
        IS_COMPLEX: tl.constexpr,
        COMPACT: tl.constexpr,
        N_LIMBS: tl.constexpr,
        IS_BLOCK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One lane per table entry; MODE selects the determinant the parity
        # sign is read from (0: spinless ket det, 1: pair bra strings).
        """Accumulate one streamed doubles table, one lane per entry.

        ``MODE == 0`` reads the ket determinant from ``dets_ptr`` (the
        spinless table stores flat spin-orbital strings). ``MODE == 1`` reads
        the bra pair's α/β strings from ``alpha_ptr``/``beta_ptr``. ``n`` is
        the table's entry count and ``offset`` the owner-computes start of the
        output block (``Wb[k_bra - offset]``; zero for a full vector).

        ``N_LIMBS`` sizes the per-string limb axis (``1`` for a 1-D string
        tensor); each limb holds bits ``[64k, 64k + 64)`` of its string.
        ``COMPACT`` reads ``I2_ptr`` as the dense spatial ``(L, L, L, L)``
        block of a compact store and decodes each spin-orbital index into
        ``(index // 2, index & 1)``, zeroing spin-mismatched chemist pairs.
        ``IS_BLOCK`` loops the ``n_cols`` columns of an ``(M, K)`` ``T``/``Wb``
        pair, whose row stride (``T_stride``, in elements of the possibly
        interleaved real view) is ``K`` for a real block and ``2K`` for a
        complex one.

        Notes:
            Per entry the kernel evaluates
            :func:`~torch_sci_solver.torch_impl.determinants.two_excite_batch`
            on the bra determinant::

                i_lo = min(i, j); j_hi = max(i, j)
                a_lo = min(a, b); b_hi = max(a, b)
                sgn = parity(det, min(i_lo, a_lo), max(i_lo, a_lo))
                sgn *= parity(det, min(j_hi, b_hi), max(j_hi, b_hi))
                sgn *= -1 if a_lo > j_hi or b_hi < i_lo
                element = sgn * (I2[a_lo, i_lo, b_hi, j_hi]
                                 - I2[a_lo, j_hi, b_hi, i_lo])

            with ``i = an0, j = an1, a = cr0, b = cr1`` (the caller's threading
            of the annihilation and creation orbitals), and adds
            ``element * T[k_ket]`` into ``Wb[k_bra]`` (per column for a block).
        """
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        active = idx < n

        kb = tl.load(k_bra_ptr + idx, mask=active, other=0).to(tl.int64)
        kk = tl.load(k_ket_ptr + idx, mask=active, other=0).to(tl.int64)
        cr0 = tl.load(cr0_ptr + idx, mask=active, other=0).to(tl.int64)
        cr1 = tl.load(cr1_ptr + idx, mask=active, other=0).to(tl.int64)
        an0 = tl.load(an0_ptr + idx, mask=active, other=0).to(tl.int64)
        an1 = tl.load(an1_ptr + idx, mask=active, other=0).to(tl.int64)

        # ``two_excite_batch(det, None, i, j, a, b, I2)`` is evaluated with
        # the caller's argument order, which differs per path: the spinless
        # scatter threads the creations as (i, j) and the annihilations as
        # (a, b) because its ket is the excited determinant, while the pair
        # scatter threads (an, an, cr, cr). The swap is not cosmetic for a
        # complex Hermitian ERI: it is the conjugate element, which only
        # coincides with the correct one under the real 8-fold symmetry.
        if MODE == 0:
            i_lo = tl.minimum(cr0, cr1)
            j_hi = tl.maximum(cr0, cr1)
            a_lo = tl.minimum(an0, an1)
            b_hi = tl.maximum(an0, an1)
        else:
            i_lo = tl.minimum(an0, an1)
            j_hi = tl.maximum(an0, an1)
            a_lo = tl.minimum(cr0, cr1)
            b_hi = tl.maximum(cr0, cr1)

        lo1 = tl.minimum(i_lo, a_lo)
        hi1 = tl.maximum(i_lo, a_lo)
        lo2 = tl.minimum(j_hi, b_hi)
        hi2 = tl.maximum(j_hi, b_hi)
        if MODE == 0:
            # Flat spin-orbital ranges: the det word carries no α/β split.
            if N_LIMBS == 1:
                det = tl.load(dets_ptr + kk, mask=active, other=0)
                p = _flat_range_parity(det, lo1, hi1)
                p += _flat_range_parity(det, lo2, hi2)
            else:
                p = _flat_range_parity_limbs(dets_ptr, kk, lo1, hi1, active, N_LIMBS)
                p += _flat_range_parity_limbs(dets_ptr, kk, lo2, hi2, active, N_LIMBS)
        else:
            # Interleaved α/β strings: α sits at even spin-orbital positions.
            if N_LIMBS == 1:
                a_str = tl.load(alpha_ptr + kb, mask=active, other=0)
                b_str = tl.load(beta_ptr + kb, mask=active, other=0)
                p = _range_parity_word(lo1, hi1, a_str, b_str)
                p += _range_parity_word(lo2, hi2, a_str, b_str)
            else:
                p = _range_parity(lo1, hi1, kb, kb, alpha_ptr, kb, beta_ptr, kb, active, N_LIMBS)
                p += _range_parity(lo2, hi2, kb, kb, alpha_ptr, kb, beta_ptr, kb, active, N_LIMBS)
        sgn = 1 - 2 * (p & 1)
        sgn = tl.where((a_lo > j_hi) | (b_hi < i_lo), -sgn, sgn).to(tl.float64)

        elem_ty = Wb_ptr.dtype.element_ty
        w_index = kb - offset
        if COMPACT:
            # Dense entries decode to the spatial block; a chemist pair whose
            # two spins differ is zero, as in ``CompactEri.__getitem__``.
            a_sp = a_lo // 2
            i_sp = i_lo // 2
            b_sp = b_hi // 2
            j_sp = j_hi // 2
            off_aibj = ((a_sp * n_spatial + i_sp) * n_spatial + b_sp) * n_spatial + j_sp
            off_ajbi = ((a_sp * n_spatial + j_sp) * n_spatial + b_sp) * n_spatial + i_sp
            keep_aibj = (((a_lo ^ i_lo) & 1) == 0) & (((b_hi ^ j_hi) & 1) == 0)
            keep_ajbi = (((a_lo ^ j_hi) & 1) == 0) & (((b_hi ^ i_lo) & 1) == 0)
        else:
            off_aibj = ((a_lo * n_spinorb + i_lo) * n_spinorb + b_hi) * n_spinorb + j_hi
            off_ajbi = ((a_lo * n_spinorb + j_hi) * n_spinorb + b_hi) * n_spinorb + i_lo

        if IS_COMPLEX:
            # ``off_*`` are complex-element offsets into the (re, im) view.
            off_aibj2 = off_aibj * 2
            off_ajbi2 = off_ajbi * 2
            va_re = tl.load(I2_ptr + off_aibj2, mask=active, other=0.0)
            vb_re = tl.load(I2_ptr + off_ajbi2, mask=active, other=0.0)
            va_im = tl.load(I2_ptr + off_aibj2 + 1, mask=active, other=0.0)
            vb_im = tl.load(I2_ptr + off_ajbi2 + 1, mask=active, other=0.0)
            if COMPACT:
                va_re = tl.where(keep_aibj, va_re, 0.0)
                vb_re = tl.where(keep_ajbi, vb_re, 0.0)
                va_im = tl.where(keep_aibj, va_im, 0.0)
                vb_im = tl.where(keep_ajbi, vb_im, 0.0)
            e_re = (va_re - vb_re).to(tl.float64) * sgn
            e_im = (va_im - vb_im).to(tl.float64) * sgn
            if IS_BLOCK:
                for col in range(0, n_cols):
                    t_off = kk * T_stride + 2 * col
                    w_off = w_index * T_stride + 2 * col
                    t_re = tl.load(T_ptr + t_off, mask=active, other=0.0).to(tl.float64)
                    t_im = tl.load(T_ptr + t_off + 1, mask=active, other=0.0).to(tl.float64)
                    tl.atomic_add(
                        Wb_ptr + w_off,
                        (e_re * t_re - e_im * t_im).to(elem_ty),
                        mask=active,
                    )
                    tl.atomic_add(
                        Wb_ptr + w_off + 1,
                        (e_re * t_im + e_im * t_re).to(elem_ty),
                        mask=active,
                    )
            else:
                t2 = kk * 2
                t_re = tl.load(T_ptr + t2, mask=active, other=0.0).to(tl.float64)
                t_im = tl.load(T_ptr + t2 + 1, mask=active, other=0.0).to(tl.float64)
                tl.atomic_add(
                    Wb_ptr + w_index * 2, (e_re * t_re - e_im * t_im).to(elem_ty), mask=active
                )
                tl.atomic_add(
                    Wb_ptr + w_index * 2 + 1, (e_re * t_im + e_im * t_re).to(elem_ty), mask=active
                )
        else:
            va = tl.load(I2_ptr + off_aibj, mask=active, other=0.0)
            vb = tl.load(I2_ptr + off_ajbi, mask=active, other=0.0)
            if COMPACT:
                va = tl.where(keep_aibj, va, 0.0)
                vb = tl.where(keep_ajbi, vb, 0.0)
            e = (va - vb).to(tl.float64) * sgn
            if IS_BLOCK:
                for col in range(0, n_cols):
                    t_val = tl.load(T_ptr + kk * T_stride + col, mask=active, other=0.0).to(
                        tl.float64
                    )
                    tl.atomic_add(
                        Wb_ptr + w_index * T_stride + col, (e * t_val).to(elem_ty), mask=active
                    )
            else:
                t_val = tl.load(T_ptr + kk, mask=active, other=0.0).to(tl.float64)
                tl.atomic_add(Wb_ptr + w_index, (e * t_val).to(elem_ty), mask=active)

    @triton.jit
    def _single_entries_kernel(
        I1_ptr,
        I2_ptr,
        T_ptr,
        Wb_ptr,
        k_bra_ptr,
        k_ket_ptr,
        cr_ptr,
        an_ptr,
        dets_ptr,
        alpha_ptr,
        beta_ptr,
        n_spinorb,
        n_spatial,
        n,
        offset,
        n_cols,
        T_stride,
        MODE: tl.constexpr,
        IS_COMPLEX: tl.constexpr,
        COMPACT: tl.constexpr,
        N_LIMBS: tl.constexpr,
        N_J_BLOCKS: tl.constexpr,
        J_BLOCK: tl.constexpr,
        IS_BLOCK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One lane per table entry; MODE selects the determinant the parity
        # sign and the occupied set are read from (0: spinless ket det,
        # 1: pair bra strings).
        """Accumulate one streamed singles table, one lane per entry.

        ``MODE == 0`` reads the ket determinant from ``dets_ptr`` (the
        spinless table stores flat spin-orbital strings) and threads the
        excitation as ``i = cr``, ``a = an``. ``MODE == 1`` reads the bra
        pair's α/β strings from ``alpha_ptr``/``beta_ptr`` and threads it as
        ``i = an``, ``a = cr``. ``n`` is the table's entry count and
        ``offset`` the owner-computes start of the output block
        (``Wb[k_bra - offset]``; zero for a full vector).

        ``N_LIMBS`` sizes the per-string limb axis (``1`` for a 1-D string
        tensor); each limb holds bits ``[64k, 64k + 64)`` of its string.
        ``N_J_BLOCKS = ceil(n_spinorb / J_BLOCK)`` blocks of ``J_BLOCK``
        spin-orbitals cover the occupied-set sum; block ``k`` reads limb ``k``
        of a flat string, or half-limb ``k // 2`` of an interleaved α/β
        string (``J_BLOCK == 64`` interleaved positions encode 32 α and 32 β
        bits).
        ``COMPACT`` reads ``I2_ptr`` as the dense spatial ``(L, L, L, L)``
        block of a compact store and decodes each spin-orbital index into
        ``(index // 2, index & 1)``, zeroing spin-mismatched entries.
        ``IS_BLOCK`` loops the ``n_cols`` columns of an ``(M, K)`` ``T``/``Wb``
        pair, whose row stride (``T_stride``, in elements of the possibly
        interleaved real view) is ``K`` for a real block and ``2K`` for a
        complex one. ``I1_ptr`` is always the dense ``(2L, 2L)`` one-body
        tensor.

        Notes:
            Per entry the kernel evaluates
            :func:`~torch_sci_solver.torch_impl.determinants.one_excite_batch`
            on the determinant the mode reads::

                sgn = parity(det, min(i, a), max(i, a))
                element = sgn * (I1[a, i]
                                 + sum_j occ(det, j)
                                     * (I2[a, i, j, j] - I2[a, j, j, i]))

            with the sum over ``j`` in ``[0, n_spinorb)``, and adds
            ``element * T[k_ket]`` into ``Wb[k_bra]`` (per column for a
            block).
        """
        idx = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        active = idx < n

        kb = tl.load(k_bra_ptr + idx, mask=active, other=0).to(tl.int64)
        kk = tl.load(k_ket_ptr + idx, mask=active, other=0).to(tl.int64)
        cr = tl.load(cr_ptr + idx, mask=active, other=0).to(tl.int64)
        an = tl.load(an_ptr + idx, mask=active, other=0).to(tl.int64)

        # ``one_excite_batch``'s calling order: the spinless scatter passes the
        # ket-only orbital as ``i`` and the bra-only one as ``a``; the pair
        # scatter passes them the other way round. ``I1[a, i]`` differs between
        # the two (it is the conjugate for a complex Hermitian I1), so the
        # distinction is not cosmetic.
        if MODE == 0:
            i_orb = cr
            a_orb = an
        else:
            i_orb = an
            a_orb = cr

        lo = tl.minimum(i_orb, a_orb)
        hi = tl.maximum(i_orb, a_orb)
        if MODE == 0:
            # Flat spin-orbital ranges: the det word carries no α/β split.
            if N_LIMBS == 1:
                det = tl.load(dets_ptr + kk, mask=active, other=0)
                p = _flat_range_parity(det, lo, hi)
            else:
                p = _flat_range_parity_limbs(dets_ptr, kk, lo, hi, active, N_LIMBS)
        else:
            # Interleaved α/β strings: α sits at even spin-orbital positions.
            if N_LIMBS == 1:
                a_str = tl.load(alpha_ptr + kb, mask=active, other=0)
                b_str = tl.load(beta_ptr + kb, mask=active, other=0)
                p = _range_parity_word(lo, hi, a_str, b_str)
            else:
                p = _range_parity(lo, hi, kb, kb, alpha_ptr, kb, beta_ptr, kb, active, N_LIMBS)
        sgn = (1 - 2 * (p & 1)).to(tl.float64)

        # ``sum_j occ(det, j) * (I2[a, i, j, j] - I2[a, j, j, i])`` over blocks
        # of 64 spin-orbitals, accumulated in float64 for every storage dtype.
        acc_re = tl.zeros((BLOCK,), tl.float64)
        acc_im = tl.zeros((BLOCK,), tl.float64)
        q = tl.arange(0, J_BLOCK).to(tl.int64)
        for k in range(0, N_J_BLOCKS):
            jv = (k * J_BLOCK + q).to(tl.int64)
            jmask = jv < n_spinorb
            if MODE == 0:
                word = tl.load(dets_ptr + kk * N_LIMBS + k, mask=active, other=0)
                occ = (word[:, None] >> q[None, :]) & 1
            else:
                # 64 interleaved positions = 32 α bits then 32 β bits, all
                # inside half-limb ``k // 2`` at offset ``32 * (k % 2)``.
                half = k // 2
                t_shift = 32 * (k % 2) + q // 2
                a_word = tl.load(alpha_ptr + kb * N_LIMBS + half, mask=active, other=0)
                b_word = tl.load(beta_ptr + kb * N_LIMBS + half, mask=active, other=0)
                a_occ = (a_word[:, None] >> t_shift[None, :]) & 1
                b_occ = (b_word[:, None] >> t_shift[None, :]) & 1
                occ = tl.where((q[None, :] & 1) == 0, a_occ, b_occ)
            weight = occ.to(tl.float64)
            jj = jv[None, :]
            if COMPACT:
                # Dense entries decode to the spatial block; an index pair
                # whose two spins differ is zero, as in ``CompactEri``.
                a_sp = a_orb[:, None] // 2
                i_sp = i_orb[:, None] // 2
                j_sp = jv[None, :] // 2
                off_aijj = ((a_sp * n_spatial + i_sp) * n_spatial + j_sp) * n_spatial + j_sp
                off_ajji = ((a_sp * n_spatial + j_sp) * n_spatial + j_sp) * n_spatial + i_sp
                keep_aijj = (((a_orb[:, None] ^ i_orb[:, None]) & 1) == 0) & jmask[None, :]
                keep_ajji = (((a_orb[:, None] ^ jj) & 1) == 0) & (((jj ^ i_orb[:, None]) & 1) == 0)
            else:
                off_aijj = (
                    (a_orb[:, None] * n_spinorb + i_orb[:, None]) * n_spinorb + jj
                ) * n_spinorb + jj
                off_ajji = ((a_orb[:, None] * n_spinorb + jj) * n_spinorb + jj) * n_spinorb + i_orb[
                    :, None
                ]
            mask = active[:, None] & jmask[None, :]
            if IS_COMPLEX:
                # ``off_*`` are complex-element offsets into the (re, im) view.
                off_aijj2 = off_aijj * 2
                off_ajji2 = off_ajji * 2
                va_re = tl.load(I2_ptr + off_aijj2, mask=mask, other=0.0)
                vb_re = tl.load(I2_ptr + off_ajji2, mask=mask, other=0.0)
                va_im = tl.load(I2_ptr + off_aijj2 + 1, mask=mask, other=0.0)
                vb_im = tl.load(I2_ptr + off_ajji2 + 1, mask=mask, other=0.0)
                if COMPACT:
                    va_re = tl.where(keep_aijj, va_re, 0.0)
                    vb_re = tl.where(keep_ajji, vb_re, 0.0)
                    va_im = tl.where(keep_aijj, va_im, 0.0)
                    vb_im = tl.where(keep_ajji, vb_im, 0.0)
                acc_re += tl.sum((va_re.to(tl.float64) - vb_re.to(tl.float64)) * weight, axis=1)
                acc_im += tl.sum((va_im.to(tl.float64) - vb_im.to(tl.float64)) * weight, axis=1)
            else:
                va = tl.load(I2_ptr + off_aijj, mask=mask, other=0.0)
                vb = tl.load(I2_ptr + off_ajji, mask=mask, other=0.0)
                if COMPACT:
                    va = tl.where(keep_aijj, va, 0.0)
                    vb = tl.where(keep_ajji, vb, 0.0)
                acc_re += tl.sum((va.to(tl.float64) - vb.to(tl.float64)) * weight, axis=1)

        elem_ty = Wb_ptr.dtype.element_ty
        w_index = kb - offset
        if IS_COMPLEX:
            i1_off = (a_orb * n_spinorb + i_orb) * 2
            e1_re = tl.load(I1_ptr + i1_off, mask=active, other=0.0).to(tl.float64)
            e1_im = tl.load(I1_ptr + i1_off + 1, mask=active, other=0.0).to(tl.float64)
            e_re = (e1_re + acc_re) * sgn
            e_im = (e1_im + acc_im) * sgn
            if IS_BLOCK:
                for col in range(0, n_cols):
                    t_off = kk * T_stride + 2 * col
                    w_off = w_index * T_stride + 2 * col
                    t_re = tl.load(T_ptr + t_off, mask=active, other=0.0).to(tl.float64)
                    t_im = tl.load(T_ptr + t_off + 1, mask=active, other=0.0).to(tl.float64)
                    tl.atomic_add(
                        Wb_ptr + w_off,
                        (e_re * t_re - e_im * t_im).to(elem_ty),
                        mask=active,
                    )
                    tl.atomic_add(
                        Wb_ptr + w_off + 1,
                        (e_re * t_im + e_im * t_re).to(elem_ty),
                        mask=active,
                    )
            else:
                t2 = kk * 2
                t_re = tl.load(T_ptr + t2, mask=active, other=0.0).to(tl.float64)
                t_im = tl.load(T_ptr + t2 + 1, mask=active, other=0.0).to(tl.float64)
                tl.atomic_add(
                    Wb_ptr + w_index * 2, (e_re * t_re - e_im * t_im).to(elem_ty), mask=active
                )
                tl.atomic_add(
                    Wb_ptr + w_index * 2 + 1, (e_re * t_im + e_im * t_re).to(elem_ty), mask=active
                )
        else:
            e1 = tl.load(I1_ptr + a_orb * n_spinorb + i_orb, mask=active, other=0.0).to(tl.float64)
            e = (e1 + acc_re) * sgn
            if IS_BLOCK:
                for col in range(0, n_cols):
                    t_val = tl.load(T_ptr + kk * T_stride + col, mask=active, other=0.0).to(
                        tl.float64
                    )
                    tl.atomic_add(
                        Wb_ptr + w_index * T_stride + col, (e * t_val).to(elem_ty), mask=active
                    )
            else:
                t_val = tl.load(T_ptr + kk, mask=active, other=0.0).to(tl.float64)
                tl.atomic_add(Wb_ptr + w_index, (e * t_val).to(elem_ty), mask=active)


def _table_limbs(strings: Tensor | None, n_bits: int) -> int | None:
    """Limb count of a contiguous int64 determinant-string table.

    A ``(M,)`` tensor is one word per row; an ``(M, n_limbs)`` matrix stores
    limb ``k`` (bits ``[64k, 64k + 64)``) in column ``k``, and the row stride
    is then ``n_limbs``. Returns ``None`` when the table is not a contiguous
    int64 vector/matrix or cannot represent ``n_bits``-wide strings.
    """
    if not isinstance(strings, Tensor) or strings.dtype != torch.int64:
        return None
    if not strings.is_contiguous() or strings.ndim not in (1, 2):
        return None
    n_limbs = 1 if strings.ndim == 1 else int(strings.shape[1])
    if n_limbs < 1 or 64 * n_limbs < n_bits:
        return None
    return n_limbs


def _eri_buffer(integrals: TorchIntegrals) -> Tensor | None:
    """Dense ERI buffer the kernel reads for this bundle, or ``None``.

    A dense bundle exposes its chemist-order ``(2L, 2L, 2L, 2L)`` tensor
    directly. A compact bundle (``eri_compact``) exposes the spatial
    ``(L, L, L, L)`` tensor held in the
    :class:`~torch_sci_solver.torch_impl.integrals.CompactEri` wrapper's
    ``_spatial`` slot; the kernel then decodes every spin-orbital index and
    applies the spin pattern itself. Sharded or otherwise non-tensor stores
    have no dense buffer and are rejected.
    """
    I2 = integrals.I2_eri
    n_spinorb = int(integrals.n_spinorb)
    if integrals.eri_compact:
        if n_spinorb % 2:
            return None
        spatial = getattr(I2, "_spatial", None)
        if not isinstance(spatial, Tensor) or not spatial.is_contiguous():
            return None
        if tuple(spatial.shape) != (n_spinorb // 2,) * 4:
            return None
        return spatial
    if not isinstance(I2, Tensor) or not I2.is_contiguous():
        return None
    if tuple(I2.shape) != (n_spinorb,) * 4:
        return None
    return I2


def _entry_layout_supported(
    Wb: Tensor,
    T: Tensor,
    integrals: TorchIntegrals,
    tables: tuple[Tensor, ...],
    mode_code: int,
    dets: Tensor | None,
    alpha_of_pair: Tensor | None,
    beta_of_pair: Tensor | None,
) -> bool:
    """Whether tensors, dtypes and storage layouts fit the entry kernel.

    Every check here is device-agnostic apart from requiring all operands to
    share one device, so a CPU caller can exercise the accept/reject matrix
    without a GPU; :func:`apply_double_entries` adds the Triton and CUDA gates
    on top. A rejected layout always returns ``False``, never raises.
    """
    n_entries = int(tables[0].shape[0])
    if any(
        t.shape != (n_entries,) or t.dtype != torch.int64 or not t.is_contiguous() for t in tables
    ):
        return False
    if T.ndim not in (1, 2) or Wb.ndim != T.ndim:
        return False
    if T.ndim == 2 and (T.shape[1] < 1 or Wb.shape[1] != T.shape[1]):
        return False
    if not T.is_contiguous() or not Wb.is_contiguous():
        return False

    eri = _eri_buffer(integrals)
    if eri is None:
        return False
    if Wb.dtype not in _SUPPORTED_DTYPES or T.dtype != Wb.dtype or eri.dtype != Wb.dtype:
        return False
    if any(t.device != Wb.device for t in (*tables, T, eri)):
        return False

    n_spinorb = int(integrals.n_spinorb)
    if mode_code == 0:
        n_limbs = _table_limbs(dets, n_spinorb)
        if n_limbs is None or dets is None:
            return False
        if dets.device != Wb.device or dets.shape[0] != T.shape[0]:
            return False
    else:
        if n_spinorb % 2:
            return False
        n_limbs = _table_limbs(alpha_of_pair, n_spinorb // 2)
        if n_limbs is None or alpha_of_pair is None:
            return False
        if _table_limbs(beta_of_pair, n_spinorb // 2) != n_limbs or beta_of_pair is None:
            return False
        if alpha_of_pair.device != Wb.device or beta_of_pair.device != Wb.device:
            return False
        if alpha_of_pair.shape[0] != T.shape[0] or beta_of_pair.shape[0] != T.shape[0]:
            return False
    return True


def _single_entry_layout_supported(
    Wb: Tensor,
    T: Tensor,
    integrals: TorchIntegrals,
    tables: tuple[Tensor, ...],
    mode_code: int,
    dets: Tensor | None,
    alpha_of_pair: Tensor | None,
    beta_of_pair: Tensor | None,
) -> bool:
    """Whether tensors, dtypes and storage layouts fit the singles kernel.

    Extends :func:`_entry_layout_supported` (table, ``T``/``Wb`` and ERI
    checks, shared with the doubles kernel) with the one-body tensor the
    singles additionally read: a contiguous dense ``(n_spinorb, n_spinorb)``
    ``I1`` in the output's dtype and device. Like the doubles predicate it is
    device-agnostic apart from requiring one device, and a rejected layout
    returns ``False``, never raises.
    """
    if not _entry_layout_supported(
        Wb, T, integrals, tables, mode_code, dets, alpha_of_pair, beta_of_pair
    ):
        return False
    I1 = integrals.I1
    if not isinstance(I1, Tensor) or not I1.is_contiguous():
        return False
    n_spinorb = int(integrals.n_spinorb)
    if tuple(I1.shape) != (n_spinorb, n_spinorb):
        return False
    return I1.dtype == Wb.dtype and I1.device == Wb.device


def apply_double_entries(
    Wb: Tensor,
    T: Tensor,
    integrals: TorchIntegrals,
    *,
    k_bra: Tensor,
    k_ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    mode: str,
    dets: Tensor | None = None,
    alpha_of_pair: Tensor | None = None,
    beta_of_pair: Tensor | None = None,
    offset: int = 0,
    BLOCK: int = _BLOCK,
) -> bool:
    """Add one streamed doubles table with the entry-parallel Triton kernel.

    The entry table threads the annihilation orbitals as ``an0, an1`` and the
    creation orbitals as ``cr0, cr1``, exactly like the batched
    :func:`~torch_sci_solver.torch_impl.determinants.two_excite_batch` calls in
    :func:`~torch_sci_solver.torch_impl.mult_spinless._scatter_double_spinless`
    and :func:`~torch_sci_solver.torch_impl.mult_pairs._scatter_double_pair`.

    Args:
        Wb: output vector (or owner-computes output block) accumulated in
            place; a ``(M, K)`` block when ``T`` is one.
        T: input vector ``(M,)`` or block ``(M, K)``; the block path loops the
            ``K`` columns inside the kernel.
        integrals: integral bundle holding the ERI the kernel reads: either a
            dense chemist-order ``(2L, 2L, 2L, 2L)`` tensor, or the compact
            spatial store of
            :class:`~torch_sci_solver.torch_impl.integrals.CompactEri` with
            ``eri_compact`` set. Its ``n_spinorb`` is the flat ERI stride.
        k_bra: ``(n,)`` int64 destination index of each entry.
        k_ket: ``(n,)`` int64 source index of each entry.
        cr0, cr1: ``(n,)`` int64 creation (ket-only) spin-orbital indices.
        an0, an1: ``(n,)`` int64 annihilation (bra-only) spin-orbital indices.
        mode: ``"spinless"`` (flat determinant table) or ``"pair"`` (GDB pair
            table with α/β bra strings).
        dets: spinless mode only: the ``(M,)`` int64 ket determinant words
            ``k_ket`` indexes, or an ``(M, n_limbs)`` limb matrix covering
            ``n_spinorb`` bits.
        alpha_of_pair: pair mode only: the ``(M,)`` int64 bra α-string words
            ``k_bra`` indexes, or an ``(M, n_limbs)`` limb matrix covering
            ``n_spinorb // 2`` bits.
        beta_of_pair: pair mode only: the matching bra β-string words, in the
            same representation as ``alpha_of_pair``.
        offset: owner-computes start of the output block; the kernel writes
            ``Wb[k_bra - offset]``. Zero for a full replicated vector.
        BLOCK: entries per program.

    Returns:
        ``True`` when the kernel ran; ``False`` when the call is unsupported
        and the caller must keep its batched scatter. The unsupported cases
        are: Triton missing, tensors not on one CUDA device, a dtype outside
        ``float32``/``float64``/``complex64``/``complex128`` (or a mix across
        ``Wb``/``T``/ERI), a non-int64 or non-contiguous table, a
        non-contiguous input or output, an ERI that is neither a dense
        ``(2L, 2L, 2L, 2L)`` tensor nor a compact spatial store, a string
        table that cannot represent its determinant width, a ``T``/``Wb``
        pair with different ranks or column counts, and an empty table. These
        never raise; ``mode`` is validated as a programming error.
    """
    if not _HAVE_TRITON or torch.are_deterministic_algorithms_enabled():
        # The entry kernels accumulate with atomics, whose order varies run to
        # run; under deterministic algorithms the caller's batched index_add
        # (deterministic in that mode) takes the table instead.
        return False
    mode_code = _MODE_CODES.get(mode)
    if mode_code is None:
        raise ValueError(f"mode must be 'spinless' or 'pair', got {mode!r}")

    n_entries = int(k_bra.shape[0])
    if n_entries == 0:
        return False
    tables = (k_bra, k_ket, cr0, cr1, an0, an1)
    if not _entry_layout_supported(
        Wb, T, integrals, tables, mode_code, dets, alpha_of_pair, beta_of_pair
    ):
        return False

    eri = _eri_buffer(integrals)
    assert eri is not None  # narrowed by _entry_layout_supported
    if not (Wb.is_cuda and T.is_cuda and eri.is_cuda):
        return False

    n_spinorb = int(integrals.n_spinorb)
    compact = bool(integrals.eri_compact)
    is_block = T.ndim == 2
    n_cols = int(T.shape[1]) if is_block else 1
    if mode_code == 0:
        assert dets is not None  # narrowed by _entry_layout_supported
        n_limbs = _table_limbs(dets, n_spinorb)
    else:
        assert alpha_of_pair is not None  # narrowed by _entry_layout_supported
        n_limbs = _table_limbs(alpha_of_pair, n_spinorb // 2)
    assert n_limbs is not None  # narrowed by _entry_layout_supported

    eri_flat = eri.reshape(-1)
    is_complex, (I2_arg, T_arg, Wb_arg) = _kernel_complex_args(eri_flat, T, Wb)
    # Row stride of the (possibly interleaved real-view) block buffers: the
    # complex view holds twice the columns.
    T_stride = 2 * n_cols if is_complex else n_cols
    # The branch the mode does not read still needs a pointer argument; any
    # int64 table works because the unused load is eliminated at compile time.
    dets_arg = dets if mode_code == 0 else k_bra
    alpha_arg = alpha_of_pair if mode_code == 1 else k_bra
    beta_arg = beta_of_pair if mode_code == 1 else k_bra
    grid = (triton.cdiv(n_entries, int(BLOCK)),)
    _double_entries_kernel[grid](
        I2_arg,
        T_arg,
        Wb_arg,
        k_bra,
        k_ket,
        cr0,
        cr1,
        an0,
        an1,
        dets_arg,
        alpha_arg,
        beta_arg,
        n_spinorb,
        n_spinorb // 2,
        n_entries,
        int(offset),
        n_cols,
        T_stride,
        MODE=mode_code,
        IS_COMPLEX=1 if is_complex else 0,
        COMPACT=1 if compact else 0,
        N_LIMBS=n_limbs,
        IS_BLOCK=1 if is_block else 0,
        BLOCK=int(BLOCK),
    )
    return True


def apply_single_entries(
    Wb: Tensor,
    T: Tensor,
    integrals: TorchIntegrals,
    *,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    mode: str,
    dets: Tensor | None = None,
    alpha_of_pair: Tensor | None = None,
    beta_of_pair: Tensor | None = None,
    offset: int = 0,
    BLOCK: int = _BLOCK,
) -> bool:
    """Add one streamed singles table with the entry-parallel Triton kernel.

    The table threads one creation orbital (``cr``) and one annihilation
    orbital (``an``) per entry, exactly like the batched
    :func:`~torch_sci_solver.torch_impl.determinants.one_excite_batch` calls in
    :func:`~torch_sci_solver.torch_impl.mult_spinless._scatter_single_spinless`
    and :func:`~torch_sci_solver.torch_impl.mult_pairs._scatter_single_pair`.
    The two callers pass those orbitals in opposite roles (the spinless mode
    evaluates the ket determinant, the pair mode the bra), which the kernel
    preserves.

    Args:
        Wb: output vector (or owner-computes output block) accumulated in
            place; a ``(M, K)`` block when ``T`` is one.
        T: input vector ``(M,)`` or block ``(M, K)``; the block path loops the
            ``K`` columns inside the kernel.
        integrals: integral bundle holding the one-body tensor and the ERI the
            kernel reads: either a dense chemist-order ``(2L, 2L, 2L, 2L)``
            tensor, or the compact spatial store of
            :class:`~torch_sci_solver.torch_impl.integrals.CompactEri` with
            ``eri_compact`` set. Its ``I1`` is read as the dense
            ``(2L, 2L)`` one-body tensor and its ``n_spinorb`` is the flat ERI
            stride.
        k_bra: ``(n,)`` int64 destination index of each entry.
        k_ket: ``(n,)`` int64 source index of each entry.
        cr: ``(n,)`` int64 creation (ket-only) spin-orbital indices.
        an: ``(n,)`` int64 annihilation (bra-only) spin-orbital indices.
        mode: ``"spinless"`` (flat determinant table) or ``"pair"`` (GDB pair
            table with α/β bra strings).
        dets: spinless mode only: the ``(M,)`` int64 ket determinant words
            ``k_ket`` indexes, or an ``(M, n_limbs)`` limb matrix covering
            ``n_spinorb`` bits.
        alpha_of_pair: pair mode only: the ``(M,)`` int64 bra α-string words
            ``k_bra`` indexes, or an ``(M, n_limbs)`` limb matrix covering
            ``n_spinorb // 2`` bits.
        beta_of_pair: pair mode only: the matching bra β-string words, in the
            same representation as ``alpha_of_pair``.
        offset: owner-computes start of the output block; the kernel writes
            ``Wb[k_bra - offset]``. Zero for a full replicated vector.
        BLOCK: entries per program.

    Returns:
        ``True`` when the kernel ran; ``False`` when the call is unsupported
        and the caller must keep its batched scatter. The unsupported cases
        are: Triton missing, tensors not on one CUDA device, a dtype outside
        ``float32``/``float64``/``complex64``/``complex128`` (or a mix across
        ``Wb``/``T``/``I1``/ERI), a non-int64 or non-contiguous table, a
        non-contiguous input or output, a one-body tensor that is not a
        contiguous dense ``(2L, 2L)`` tensor in the output's dtype, an ERI
        that is neither a dense ``(2L, 2L, 2L, 2L)`` tensor nor a compact
        spatial store, a string table that cannot represent its determinant
        width, a ``T``/``Wb`` pair with different ranks or column counts, and
        an empty table. These never raise; ``mode`` is validated as a
        programming error.
    """
    if not _HAVE_TRITON or torch.are_deterministic_algorithms_enabled():
        # The entry kernels accumulate with atomics, whose order varies run to
        # run; under deterministic algorithms the caller's batched index_add
        # (deterministic in that mode) takes the table instead.
        return False
    mode_code = _MODE_CODES.get(mode)
    if mode_code is None:
        raise ValueError(f"mode must be 'spinless' or 'pair', got {mode!r}")

    n_entries = int(k_bra.shape[0])
    if n_entries == 0:
        return False
    tables = (k_bra, k_ket, cr, an)
    if not _single_entry_layout_supported(
        Wb, T, integrals, tables, mode_code, dets, alpha_of_pair, beta_of_pair
    ):
        return False

    eri = _eri_buffer(integrals)
    I1 = integrals.I1
    assert eri is not None and isinstance(I1, Tensor)  # narrowed by the predicate
    if not (Wb.is_cuda and T.is_cuda and eri.is_cuda and I1.is_cuda):
        return False

    n_spinorb = int(integrals.n_spinorb)
    compact = bool(integrals.eri_compact)
    is_block = T.ndim == 2
    n_cols = int(T.shape[1]) if is_block else 1
    if mode_code == 0:
        assert dets is not None  # narrowed by the predicate
        n_limbs = _table_limbs(dets, n_spinorb)
    else:
        assert alpha_of_pair is not None  # narrowed by the predicate
        n_limbs = _table_limbs(alpha_of_pair, n_spinorb // 2)
    assert n_limbs is not None  # narrowed by the predicate

    eri_flat = eri.reshape(-1)
    # ``I1`` participates so a complex launch hands every operand to the kernel
    # as its interleaved real view under one IS_COMPLEX flag.
    is_complex, (I1_arg, I2_arg, T_arg, Wb_arg) = _kernel_complex_args(I1, eri_flat, T, Wb)
    # Row stride of the (possibly interleaved real-view) block buffers: the
    # complex view holds twice the columns.
    T_stride = 2 * n_cols if is_complex else n_cols
    # The branch the mode does not read still needs a pointer argument; any
    # int64 table works because the unused load is eliminated at compile time.
    dets_arg = dets if mode_code == 0 else k_bra
    alpha_arg = alpha_of_pair if mode_code == 1 else k_bra
    beta_arg = beta_of_pair if mode_code == 1 else k_bra
    grid = (triton.cdiv(n_entries, int(BLOCK)),)
    _single_entries_kernel[grid](
        I1_arg,
        I2_arg,
        T_arg,
        Wb_arg,
        k_bra,
        k_ket,
        cr,
        an,
        dets_arg,
        alpha_arg,
        beta_arg,
        n_spinorb,
        n_spinorb // 2,
        n_entries,
        int(offset),
        n_cols,
        T_stride,
        MODE=mode_code,
        IS_COMPLEX=1 if is_complex else 0,
        COMPACT=1 if compact else 0,
        N_LIMBS=n_limbs,
        N_J_BLOCKS=(n_spinorb + _J_BLOCK - 1) // _J_BLOCK,
        J_BLOCK=_J_BLOCK,
        IS_BLOCK=1 if is_block else 0,
        BLOCK=int(BLOCK),
    )
    return True
