"""Experimental output-stationary Triton matvec (``matvec_impl="triton"``).

Tiled, register-accumulating kernels for the excitation passes:

  * αβ-doubles        (``task_type == 0``)   -> :func:`_alphabeta_kernel`
  * same-spin doubles (``task_type == 1/2``) -> :func:`_samespin_double_kernel`
  * same-spin singles (``task_type == 1/2``) -> :func:`_singles_kernel`

The diagonal stays on the batched path. A singles launch whose string/table
layout the kernel cannot address keeps the batched fast path (see
:func:`_singles_layout_supported`), so the two backends always agree.

Each program owns ``BLOCK`` consecutive outputs, **one lane per output**, and
accumulates its excitations in a register --- no ``(csize x M)``
materialization, no ``index_add_`` scatter. That is the structure of SBD's
one-thread-per-output kernel, and it is what lets the slope flatten toward
SBD's instead of merely lowering the batched prefactor.

Matrix elements:

  * αβ pass: transcribed inline from ``mult._apply_alphabeta_chunk_fast``.
  * same-spin doubles: the integral x same-spin-sign factor ``eij`` is
    **precomputed on the host** with the *same* ``two_excite_batch`` the batched
    path uses (``mult._apply_alpha_double_chunk_fast``), so the kernel only
    evaluates the cheap spectator-string parity. The α-double's sign depends on
    the β-string over β-orbital ranges ``[min(I,A), max(I,A))`` /
    ``[min(J,B), max(J,B))``; the β-double mirrors it on the α-string with a
    ``+1`` orbital-range offset (interleaved-Wick encoding).
  * same-spin singles: the source-spin half of the ``one_excite_batch``
    element — ``I1[cr, an]`` plus the occupied-orbital ``I2`` contraction over
    the source string — is **precomputed on the host** with the same
    expressions the batched fast path uses (:func:`_singles_source_factor`),
    so the kernel evaluates the full-det parity and the opposite-spin direct
    ``I2`` contraction. The opposite-spin *exchange* term is exactly zero
    because the dense TPB ERI has no mixed-spin quadrant.

Validate against the batched kernel with
``torch_sci_solver.torch_impl.validate_matvec.compare_matvec_impls`` before
trusting any timing.

String tensors are ``(M,)`` single words or ``(M, n_limbs)`` int64 limb
matrices (limb ``k`` holds bits ``[64k, 64k + 64)``). The limb count is a
``tl.constexpr`` (``N_LIMBS``): each lane loads its own limbs, range masks
are clamped to each limb's window and popcounts are summed across limbs, so
``N_LIMBS == 1`` compiles to the original single-word arithmetic.

Complex dtypes (``complex64``/``complex128``) run through the same kernels.
Triton has no complex scalar type, so every complex operand is passed as its
interleaved ``(re, im)`` real view --- which is exactly torch's complex
memory layout --- and an ``IS_COMPLEX`` constexpr selects the pair
arithmetic: the two parts are loaded separately and multiplied with the
usual four-multiply rule. Arithmetic and accumulation stay in ``float64``
for both complex precisions, like the real kernels' unconditional
``float64`` sign factor; only the loads and the final store use the storage
precision. This keeps the tiled traffic advantage over the batched complex
path, which materializes ``(csize x M)`` complex tiles and scatters them.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os

import torch

from torch_sci_solver.torch_impl.determinants import det_from_alpha_beta_batch, two_excite_batch
from torch_sci_solver.torch_impl.mult import (  # noqa: F401
    _apply_alpha_single_chunk_fast,
    _apply_beta_single_chunk_fast,
    _bra_from_offsets,
    _string_occ_matrix,
    build_beta_pass_cache,
)

# Source-of-truth descriptor of which matvec passes this build runs
# output-stationary ("tiled") vs. on the batched lift-and-scatter path.
# Recorded into benchmark metadata so a result file self-identifies which
# kernel improvements it contains (e.g. whether the same-spin alpha/beta
# singles and doubles are tiled). Lives next to the kernels so it can't drift
# out of sync.
TRITON_PASS_BACKENDS = {
    "diagonal": "batched",
    "alpha_singles": "tiled",
    "beta_singles": "tiled",
    "alpha_doubles": "tiled",
    "beta_doubles": "tiled",
    "alphabeta_doubles": "tiled",
}

try:  # Triton is optional and CUDA-only; the package must import without it.
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _popc64_swar(x):  # portable SWAR popcount, vectorized over the lane block
        """Count all 64 bits of each limb with portable integer arithmetic."""
        m1 = tl.full((1,), 0x5555555555555555, tl.int64)
        m2 = tl.full((1,), 0x3333333333333333, tl.int64)
        m4 = tl.full((1,), 0x0F0F0F0F0F0F0F0F, tl.int64)
        h01 = tl.full((1,), 0x0101010101010101, tl.int64)
        x = x - ((x >> 1) & m1)
        x = (x & m2) + ((x >> 2) & m2)
        x = (x + (x >> 4)) & m4
        return (x * h01) >> 56

    @triton.jit
    def _popc64_native(x):
        """Count all 64 bits of each limb with NVIDIA's native population count."""
        # Unsigned conversion includes bit 63 of signed int64 determinant limbs.
        return tl.inline_asm_elementwise(
            "popc.b64 $0, $1;",
            constraints="=r,l",
            args=[x.to(tl.uint64)],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        ).to(tl.int64)

    # The PTX instruction only exists on the NVIDIA backend; ROCm builds and the Triton
    # interpreter (which cannot run inline asm) keep the SWAR implementation.
    _popc64 = (
        _popc64_native
        if (
            torch.version.cuda is not None
            and torch.version.hip is None
            and os.environ.get("TRITON_INTERPRET", "0") != "1"
        )
        else _popc64_swar
    )

    @triton.jit
    def _mask_below64(n):
        """Low-``n`` bits of one int64 word, all ones for ``n >= 64``.

        ``lo``/``hi`` below are global bit positions that may fall outside a
        limb; clamping to ``[0, 64]`` keeps the shift well-defined and the
        ``n == 64`` case (the whole word) is the two's-complement ``-1``.
        """
        one = tl.full((1,), 1, tl.int64)
        n = tl.minimum(tl.maximum(n, 0), 64)
        return tl.where(n >= 64, -one, (one << n) - one)

    @triton.jit
    def _string_range_popcount(strs, ptr, rows, lo, hi, active, N_LIMBS: tl.constexpr):
        """Popcount of the string bits in the global spatial range ``[lo, hi)``.

        ``strs`` is the caller-loaded single-word string (used when
        ``N_LIMBS == 1``); for a limb matrix ``ptr``/``rows`` address the
        row, and limb ``k`` covers bits ``[64k, 64k + 64)``, so each limb
        contributes the part of the range that overlaps its own window.
        """
        one = tl.full((1,), 1, tl.int64)
        if N_LIMBS == 1:
            return _popc64(strs & ((one << hi) - (one << lo)))
        total = lo * 0
        for k in tl.static_range(0, N_LIMBS):
            off = k * 64
            s = tl.load(ptr + rows * N_LIMBS + k, mask=active, other=0)
            mask = _mask_below64(hi - off) ^ _mask_below64(lo - off)
            total += _popc64(s & mask)
        return total

    @triton.jit
    def _range_parity(
        lo,
        hi,
        a_str,
        b_str,
        a_ptr,
        a_rows,
        b_ptr,
        b_rows,
        active,
        N_LIMBS: tl.constexpr,
    ):
        """Range parity over a spin-orbital range ``[lo, hi)``.

        ``lo``/``hi`` are spin-orbital indices; α sits at even positions and β
        at odd ones, so the α-orbital range is ``[(lo+1)//2, (hi+1)//2)`` and
        the β-orbital range ``[lo//2, hi//2)``. The extra bit-at-start term
        reads the α string for an even ``lo`` and the β string for an odd one.
        ``a_str``/``b_str`` are the caller-loaded single-word strings (used
        when ``N_LIMBS == 1``); a limb matrix is read through the pointers.
        """
        one = tl.full((1,), 1, tl.int64)
        a_lo = (lo + 1) // 2
        a_hi = (hi + 1) // 2
        b_lo = lo // 2
        b_hi = hi // 2
        shift = lo // 2
        if N_LIMBS == 1:
            a_mask = (one << a_hi) - (one << a_lo)
            b_mask = (one << b_hi) - (one << b_lo)
            pa = _popc64(a_str & a_mask)
            pb = _popc64(b_str & b_mask)
            bit_mask = one << shift
            a_bit = ((a_str & bit_mask) != 0).to(tl.int64)
            b_bit = ((b_str & bit_mask) != 0).to(tl.int64)
            start_bit = tl.where((lo & 1) == 0, a_bit, b_bit)
            return pa + pb + start_bit

        total = lo * 0
        for k in tl.static_range(0, N_LIMBS):
            off = k * 64
            a_word = tl.load(a_ptr + a_rows * N_LIMBS + k, mask=active, other=0)
            b_word = tl.load(b_ptr + b_rows * N_LIMBS + k, mask=active, other=0)
            a_mask = _mask_below64(a_hi - off) ^ _mask_below64(a_lo - off)
            b_mask = _mask_below64(b_hi - off) ^ _mask_below64(b_lo - off)
            total += _popc64(a_word & a_mask) + _popc64(b_word & b_mask)
            # Bit at the global position ``shift``, taken from its own limb.
            bit = shift & 63
            a_bit = (a_word >> bit) & 1
            b_bit = (b_word >> bit) & 1
            start = tl.where((lo & 1) == 0, a_bit, b_bit)
            total += tl.where((shift >> 6) == k, start, 0)
        return total

    @triton.jit
    def _alphabeta_kernel(
        adets_ptr,
        bdets_ptr,
        a_off_ptr,
        a_cr_ptr,
        a_an_ptr,
        a_to_ptr,
        b_off_ptr,
        b_cr_ptr,
        b_an_ptr,
        b_to_ptr,
        I2_ptr,
        T_ptr,
        Wb_ptr,
        M_b,
        n,
        N,
        T_stride,
        IS_COMPLEX: tl.constexpr,
        N_LIMBS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # This program owns BLOCK consecutive outputs, one per lane.
        """Alphabeta kernel.

        ``M_b`` is the owned output width; ``T_stride`` is the row width of the
        ``T`` buffer, which differs from ``M_b`` when the buffer belongs to a
        2-D tile stage (it then holds only that stage's ket columns).
        ``N_LIMBS`` sizes the per-string limb axis (``1`` for a 1-D string
        tensor). ``IS_COMPLEX`` reads ``I2``/``T``/``Wb`` as interleaved
        ``(re, im)`` real pairs and accumulates the pair arithmetic.
        """
        # The flat output index exceeds int32 past 2^31 determinants, and
        # it drives the output, Wb and string offsets, so widen it.
        out = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        active = out < N
        ia = out // M_b
        ib = out % M_b

        # Single-word strings are loaded once per program, as before; a limb
        # matrix is read inside the parity helper, one limb at a time.
        if N_LIMBS == 1:
            a_str = tl.load(adets_ptr + ia, mask=active, other=0)
            b_str = tl.load(bdets_ptr + ib, mask=active, other=0)
        else:
            a_str = ia
            b_str = ib

        a_lo = tl.load(a_off_ptr + ia, mask=active, other=0)
        a_hi = tl.load(a_off_ptr + ia + 1, mask=active, other=0)
        b_lo = tl.load(b_off_ptr + ib, mask=active, other=0)
        b_hi = tl.load(b_off_ptr + ib + 1, mask=active, other=0)

        a_cnt = tl.where(active, a_hi - a_lo, 0)
        b_cnt = tl.where(active, b_hi - b_lo, 0)
        max_a = tl.max(a_cnt)
        max_b = tl.max(b_cnt)

        if IS_COMPLEX:
            # Complex operands arrive as interleaved (re, im) real pairs.
            out2 = out.to(tl.int64) * 2
            acc_re = tl.load(Wb_ptr + out2, mask=active, other=0.0).to(tl.float64)
            acc_im = tl.load(Wb_ptr + out2 + 1, mask=active, other=0.0).to(tl.float64)
        else:
            acc = tl.load(Wb_ptr + out, mask=active, other=0.0)  # (BLOCK,) fp64

        for jj in range(0, max_a):
            a_on = active & (jj < a_cnt)
            a_idx = tl.where(a_on, a_lo + jj, 0)
            cra = tl.load(a_cr_ptr + a_idx, mask=a_on, other=0)
            ana = tl.load(a_an_ptr + a_idx, mask=a_on, other=0)
            # int64: table payloads may be narrow (int32), but ket * T_stride
            # can exceed 2**31 on a large local tile.
            keta = tl.load(a_to_ptr + a_idx, mask=a_on, other=0).to(tl.int64)
            for kk in range(0, max_b):
                b_on = a_on & (kk < b_cnt)
                b_idx = tl.where(b_on, b_lo + kk, 0)
                crb = tl.load(b_cr_ptr + b_idx, mask=b_on, other=0)
                anb = tl.load(b_an_ptr + b_idx, mask=b_on, other=0)
                ketb = tl.load(b_to_ptr + b_idx, mask=b_on, other=0).to(tl.int64)

                # NOTE: the annihilator lo/hi temporaries MUST NOT be named a_lo/b_hi:
                # those names are already bound above as this row's a-/b-excitation list
                # offsets and are read again by the enclosing jj-loop (a_idx = a_lo + jj).
                # Shadowing them here corrupted the offset after the first jj iteration
                # (silent wrong-excitation reads for max_a >= 2). See a_an_lo/b_an_hi.
                i_lo = tl.minimum(cra, crb)
                j_hi = tl.maximum(cra, crb)
                a_an_lo = tl.minimum(ana, anb)
                b_an_hi = tl.maximum(ana, anb)
                off_aibj = ((a_an_lo * n + i_lo) * n + b_an_hi) * n + j_hi
                off_ajbi = ((a_an_lo * n + j_hi) * n + b_an_hi) * n + i_lo
                if IS_COMPLEX:
                    # off_* are complex-element offsets into the (re, im) view.
                    # The kernel's index order is (A I | B J) - (A J | B I),
                    # the conjugate of the batched path's
                    # (I A | J B) - (J A | I B) for a complex Hermitian ERI
                    # ((pq|rs) = conj((qp|sr))). Real ERIs have the 8-fold
                    # symmetry that makes the two orders equal, so only the
                    # imaginary part of the difference changes sign.
                    off_aibj2 = off_aibj.to(tl.int64) * 2
                    off_ajbi2 = off_ajbi.to(tl.int64) * 2
                    e_re = (
                        tl.load(I2_ptr + off_aibj2, mask=b_on, other=0.0)
                        - tl.load(I2_ptr + off_ajbi2, mask=b_on, other=0.0)
                    ).to(tl.float64)
                    e_im = (
                        tl.load(I2_ptr + off_ajbi2 + 1, mask=b_on, other=0.0)
                        - tl.load(I2_ptr + off_aibj2 + 1, mask=b_on, other=0.0)
                    ).to(tl.float64)
                    flip = (a_an_lo > j_hi) | (b_an_hi < i_lo)
                    e_re = tl.where(flip, -e_re, e_re)
                    e_im = tl.where(flip, -e_im, e_im)
                else:
                    e = tl.load(I2_ptr + off_aibj, mask=b_on, other=0.0) - tl.load(
                        I2_ptr + off_ajbi, mask=b_on, other=0.0
                    )
                    e = tl.where((a_an_lo > j_hi) | (b_an_hi < i_lo), -e, e)
                p = _range_parity(
                    tl.minimum(i_lo, a_an_lo),
                    tl.maximum(i_lo, a_an_lo),
                    a_str,
                    b_str,
                    adets_ptr,
                    ia,
                    bdets_ptr,
                    ib,
                    active,
                    N_LIMBS,
                )
                p += _range_parity(
                    tl.minimum(j_hi, b_an_hi),
                    tl.maximum(j_hi, b_an_hi),
                    a_str,
                    b_str,
                    adets_ptr,
                    ia,
                    bdets_ptr,
                    ib,
                    active,
                    N_LIMBS,
                )
                sgn = (1 - 2 * (p & 1)).to(tl.float64)
                if IS_COMPLEX:
                    t2 = (keta * T_stride + ketb) * 2
                    t_re = tl.load(T_ptr + t2, mask=b_on, other=0.0).to(tl.float64)
                    t_im = tl.load(T_ptr + t2 + 1, mask=b_on, other=0.0).to(tl.float64)
                    acc_re += tl.where(b_on, sgn * (e_re * t_re - e_im * t_im), 0.0)
                    acc_im += tl.where(b_on, sgn * (e_re * t_im + e_im * t_re), 0.0)
                else:
                    t_val = tl.load(T_ptr + keta * T_stride + ketb, mask=b_on, other=0.0)
                    acc += tl.where(b_on, e * sgn * t_val, 0.0)

        if IS_COMPLEX:
            tl.store(Wb_ptr + out2, acc_re.to(Wb_ptr.dtype.element_ty), mask=active)
            tl.store(Wb_ptr + out2 + 1, acc_im.to(Wb_ptr.dtype.element_ty), mask=active)
        else:
            tl.store(Wb_ptr + out, acc, mask=active)

    @triton.jit
    def _samespin_double_kernel(
        adets_ptr,
        bdets_ptr,
        off_ptr,
        to_ptr,
        cr0_ptr,
        cr1_ptr,
        an0_ptr,
        an1_ptr,
        eij_ptr,
        T_ptr,
        Wb_ptr,
        M_b,
        N,
        T_stride,
        IS_ALPHA: tl.constexpr,
        RANGE_OFF: tl.constexpr,
        IS_COMPLEX: tl.constexpr,
        N_LIMBS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One lane per output (ia, ib). For α-doubles the β-string is the
        # spectator that carries the sign; for β-doubles it is the α-string.
        """Samespin double kernel.

        ``M_b`` is the owned output width; ``T_stride`` is the row width of the
        ``T`` buffer, which differs from ``M_b`` for a 2-D tile stage buffer
        (a remote column slab holds only the ket columns that stage needs).
        ``N_LIMBS`` sizes the per-string limb axis (``1`` for a 1-D string
        tensor). ``IS_COMPLEX`` reads ``eij``/``T``/``Wb`` as interleaved
        ``(re, im)`` real pairs and accumulates the pair arithmetic.
        """
        # The flat output index exceeds int32 past 2^31 determinants, and
        # it drives the output, Wb and string offsets, so widen it.
        out = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        active = out < N
        ia = out // M_b
        ib = out % M_b

        if IS_ALPHA:
            key = ia  # walk excitations whose α-bra is ia
            sign_ptr = bdets_ptr  # β-string carries the parity sign
            sign_rows = ib
        else:
            key = ib  # walk excitations whose β-bra is ib
            sign_ptr = adets_ptr  # α-string carries the parity sign
            sign_rows = ia

        # Single-word spectator loaded once per program, as before; a limb
        # matrix is read inside the parity helper, one limb at a time.
        if N_LIMBS == 1:
            sign_str = tl.load(sign_ptr + sign_rows, mask=active, other=0)
        else:
            sign_str = sign_rows

        lo = tl.load(off_ptr + key, mask=active, other=0)
        hi = tl.load(off_ptr + key + 1, mask=active, other=0)
        cnt = tl.where(active, hi - lo, 0)
        max_c = tl.max(cnt)

        if IS_COMPLEX:
            # Complex operands arrive as interleaved (re, im) real pairs.
            out2 = out.to(tl.int64) * 2
            acc_re = tl.load(Wb_ptr + out2, mask=active, other=0.0).to(tl.float64)
            acc_im = tl.load(Wb_ptr + out2 + 1, mask=active, other=0.0).to(tl.float64)
        else:
            acc = tl.load(Wb_ptr + out, mask=active, other=0.0)  # (BLOCK,) fp64

        for jj in range(0, max_c):
            on = active & (jj < cnt)
            idx = tl.where(on, lo + jj, 0)
            ket = tl.load(to_ptr + idx, mask=on, other=0).to(tl.int64)
            cr0 = tl.load(cr0_ptr + idx, mask=on, other=0)
            cr1 = tl.load(cr1_ptr + idx, mask=on, other=0)
            an0 = tl.load(an0_ptr + idx, mask=on, other=0)
            an1 = tl.load(an1_ptr + idx, mask=on, other=0)
            if IS_COMPLEX:
                # idx is a complex-element offset into the (re, im) view.
                idx2 = idx.to(tl.int64) * 2
                e_re = tl.load(eij_ptr + idx2, mask=on, other=0.0).to(tl.float64)
                e_im = tl.load(eij_ptr + idx2 + 1, mask=on, other=0.0).to(tl.float64)
            else:
                eij = tl.load(eij_ptr + idx, mask=on, other=0.0)

            # spectator parity over two orbital ranges (see module docstring).
            i_lo = tl.minimum(an0, an1) // 2
            j_hi = tl.maximum(an0, an1) // 2
            a_lo = tl.minimum(cr0, cr1) // 2
            b_hi = tl.maximum(cr0, cr1) // 2
            r1_lo = tl.minimum(i_lo, a_lo) + RANGE_OFF
            r1_hi = tl.maximum(i_lo, a_lo) + RANGE_OFF
            r2_lo = tl.minimum(j_hi, b_hi) + RANGE_OFF
            r2_hi = tl.maximum(j_hi, b_hi) + RANGE_OFF
            pop = _string_range_popcount(
                sign_str, sign_ptr, sign_rows, r1_lo, r1_hi, active, N_LIMBS
            ) + _string_range_popcount(sign_str, sign_ptr, sign_rows, r2_lo, r2_hi, active, N_LIMBS)
            sgn = (1 - 2 * (pop & 1)).to(tl.float64)

            if IS_COMPLEX:
                if IS_ALPHA:
                    t2 = (ket * T_stride + ib.to(tl.int64)) * 2
                else:
                    t2 = (ia.to(tl.int64) * T_stride + ket) * 2
                t_re = tl.load(T_ptr + t2, mask=on, other=0.0).to(tl.float64)
                t_im = tl.load(T_ptr + t2 + 1, mask=on, other=0.0).to(tl.float64)
                acc_re += tl.where(on, sgn * (e_re * t_re - e_im * t_im), 0.0)
                acc_im += tl.where(on, sgn * (e_re * t_im + e_im * t_re), 0.0)
            else:
                if IS_ALPHA:
                    t_val = tl.load(T_ptr + ket * T_stride + ib, mask=on, other=0.0)
                else:
                    t_val = tl.load(T_ptr + ia.to(tl.int64) * T_stride + ket, mask=on, other=0.0)
                acc += tl.where(on, eij * sgn * t_val, 0.0)

        if IS_COMPLEX:
            tl.store(Wb_ptr + out2, acc_re.to(Wb_ptr.dtype.element_ty), mask=active)
            tl.store(Wb_ptr + out2 + 1, acc_im.to(Wb_ptr.dtype.element_ty), mask=active)
        else:
            tl.store(Wb_ptr + out, acc, mask=active)

    @triton.jit
    def _singles_kernel(
        adets_ptr,
        bdets_ptr,
        off_ptr,
        to_ptr,
        cr_ptr,
        an_ptr,
        e0_ptr,
        I2_ptr,
        T_ptr,
        Wb_ptr,
        M_b,
        N,
        T_stride,
        n_spinorb,
        IS_ALPHA: tl.constexpr,
        IS_COMPLEX: tl.constexpr,
        N_LIMBS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One lane per output (ia, ib). For α singles the excitation list is
        # keyed by ia and the β string is the spectator; for β singles the
        # list is keyed by ib and the α string is the spectator.
        """Same-spin singles kernel.

        ``M_b`` is the owned output width; ``T_stride`` is the row width of the
        ``T`` buffer, which differs from ``M_b`` for a 2-D tile stage buffer
        (a remote column slab holds only the ket columns that stage needs).
        ``n_spinorb`` is the flat ERI stride. ``N_LIMBS`` sizes the per-string
        limb axis (``1`` for a 1-D string tensor). ``IS_COMPLEX`` reads
        ``e0``/``I2``/``T``/``Wb`` as interleaved ``(re, im)`` real pairs and
        accumulates the pair arithmetic.

        Notes:
            Per lane the excitation is one row of the source-spin CSR table and
            the matrix element is::

                element = (-1) ** p * (e0[idx] + sum_{j in occ(spectator)}
                                        I2[cr, an, p_j, p_j])

            with ``p = parity(bra det, min(cr, an), max(cr, an))`` the
            interleaved parity of the bra determinant, ``p_j = 2j + 1`` for
            α singles and ``2j`` for β singles, and ``e0`` the source-spin
            ``I1 + sum_j I2`` contraction precomputed on the host with the
            batched path's expressions (:func:`_singles_source_factor`). The
            opposite-spin exchange term ``I2[cr, p_j, p_j, an]`` is exactly
            zero: the dense TPB ERI has no mixed-spin quadrant, so the
            spectator sum keeps only the direct term. This is the
            ``one_excite_batch`` element of the bra determinant.
        """
        # The flat output index exceeds int32 past 2^31 determinants, and
        # it drives the output, Wb and string offsets, so widen it.
        out = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        active = out < N
        ia = out // M_b
        ib = out % M_b

        if IS_ALPHA:
            key = ia  # walk excitations whose α-bra is ia
            spec_ptr = bdets_ptr  # β-string is the spectator
            spec_rows = ib
        else:
            key = ib  # walk excitations whose β-bra is ib
            spec_ptr = adets_ptr  # α-string is the spectator
            spec_rows = ia

        # Both strings enter the full-det parity; the spectator one also
        # drives the opposite-spin direct contraction. Single-word strings are
        # loaded once per program, a limb matrix is read limb by limb.
        if N_LIMBS == 1:
            a_str = tl.load(adets_ptr + ia, mask=active, other=0)
            b_str = tl.load(bdets_ptr + ib, mask=active, other=0)
            if IS_ALPHA:
                spec_word = b_str
            else:
                spec_word = a_str
        else:
            a_str = ia
            b_str = ib
            spec_word = spec_rows

        lo = tl.load(off_ptr + key, mask=active, other=0)
        hi = tl.load(off_ptr + key + 1, mask=active, other=0)
        cnt = tl.where(active, hi - lo, 0)
        max_c = tl.max(cnt)

        if IS_COMPLEX:
            # Complex operands arrive as interleaved (re, im) real pairs.
            out2 = out.to(tl.int64) * 2
            acc_re = tl.load(Wb_ptr + out2, mask=active, other=0.0).to(tl.float64)
            acc_im = tl.load(Wb_ptr + out2 + 1, mask=active, other=0.0).to(tl.float64)
        else:
            acc = tl.load(Wb_ptr + out, mask=active, other=0.0).to(tl.float64)

        n_orb = n_spinorb // 2
        for jj in range(0, max_c):
            on = active & (jj < cnt)
            idx = tl.where(on, lo + jj, 0).to(tl.int64)
            ket = tl.load(to_ptr + idx, mask=on, other=0).to(tl.int64)
            cr = tl.load(cr_ptr + idx, mask=on, other=0).to(tl.int64)
            an = tl.load(an_ptr + idx, mask=on, other=0).to(tl.int64)

            p = _range_parity(
                tl.minimum(cr, an),
                tl.maximum(cr, an),
                a_str,
                b_str,
                adets_ptr,
                ia,
                bdets_ptr,
                ib,
                active,
                N_LIMBS,
            )
            sgn = (1 - 2 * (p & 1)).to(tl.float64)

            # Opposite-spin direct contraction: I2[cr, an, p_j, p_j] over the
            # occupied spectator orbitals. ``off0`` is the j = 0 offset and
            # ``step`` the j stride (p_j advances by two spin orbitals, the
            # row stride by n_spinorb, so the flat offset grows by
            # 2 * (n_spinorb + 1)).
            off0 = (cr * n_spinorb + an) * n_spinorb * n_spinorb
            step = 2 * (n_spinorb.to(tl.int64) + 1)
            if IS_ALPHA:
                off0 += n_spinorb + 1  # p_j = 1 at j = 0
            if IS_COMPLEX:
                dot_re = tl.zeros_like(acc_re)
                dot_im = tl.zeros_like(acc_im)
            else:
                dot = tl.zeros_like(acc)
            for k in tl.static_range(0, N_LIMBS):
                if N_LIMBS == 1:
                    word = spec_word
                else:
                    word = tl.load(spec_ptr + spec_rows * N_LIMBS + k, mask=active, other=0)
                base = k * 64
                n_bits = tl.maximum(0, tl.minimum(64, n_orb - base))
                for b in range(0, n_bits):
                    off = off0 + (base + b) * step
                    occ = ((word >> b) & 1).to(tl.float64)
                    if IS_COMPLEX:
                        off2 = off * 2
                        v_re = tl.load(I2_ptr + off2, mask=on, other=0.0).to(tl.float64)
                        v_im = tl.load(I2_ptr + off2 + 1, mask=on, other=0.0).to(tl.float64)
                        dot_re += v_re * occ
                        dot_im += v_im * occ
                    else:
                        v = tl.load(I2_ptr + off, mask=on, other=0.0).to(tl.float64)
                        dot += v * occ

            if IS_COMPLEX:
                idx2 = idx.to(tl.int64) * 2
                e_re = tl.load(e0_ptr + idx2, mask=on, other=0.0).to(tl.float64)
                e_im = tl.load(e0_ptr + idx2 + 1, mask=on, other=0.0).to(tl.float64)
                if IS_ALPHA:
                    t2 = (ket * T_stride + ib) * 2
                else:
                    t2 = (ia * T_stride + ket) * 2
                t_re = tl.load(T_ptr + t2, mask=on, other=0.0).to(tl.float64)
                t_im = tl.load(T_ptr + t2 + 1, mask=on, other=0.0).to(tl.float64)
                elem_re = e_re + dot_re
                elem_im = e_im + dot_im
                acc_re += tl.where(on, sgn * (elem_re * t_re - elem_im * t_im), 0.0)
                acc_im += tl.where(on, sgn * (elem_re * t_im + elem_im * t_re), 0.0)
            else:
                e0 = tl.load(e0_ptr + idx, mask=on, other=0.0).to(tl.float64)
                if IS_ALPHA:
                    t_val = tl.load(T_ptr + ket * T_stride + ib, mask=on, other=0.0)
                else:
                    t_val = tl.load(T_ptr + ia * T_stride + ket, mask=on, other=0.0)
                acc += tl.where(on, sgn * (e0 + dot) * t_val.to(tl.float64), 0.0)

        if IS_COMPLEX:
            tl.store(Wb_ptr + out2, acc_re.to(Wb_ptr.dtype.element_ty), mask=active)
            tl.store(Wb_ptr + out2 + 1, acc_im.to(Wb_ptr.dtype.element_ty), mask=active)
        else:
            tl.store(Wb_ptr + out, acc.to(Wb_ptr.dtype.element_ty), mask=active)


def _samespin_eij(
    strings,
    offsets,
    cr0,
    cr1,
    an0,
    an1,
    n_orb,
    integrals,
    *,
    is_alpha,
):
    """Integral x same-spin-sign factor for one same-spin doubles table.

    ``strings`` is the α (``is_alpha``) or β string set the table's bra
    indices refer to. This is the expression
    ``_apply_samespin_doubles_triton`` used to inline; extracting it lets the
    factor be built once per subspace.

    Notes:
        The sign comes from ``two_excite_batch`` on the bra determinant of the
        same spin only; the spectator-string parity is evaluated in the kernel.

        ``two_excite_batch`` takes the annihilation indices as ``i, j`` and the
        creation indices as ``a, b``, so the creation index leads each chemist
        pair; the batched same-spin kernels call it in exactly this order. The
        swapped order would conjugate every element of a complex Hermitian
        ERI (``I2[p, q, r, s] = conj(I2[q, p, s, r])``), so it is not
        interchangeable here.
    """
    device = strings.device
    bra = _bra_from_offsets(offsets, int(strings.shape[0]), device)
    bra_bm = strings[bra]
    zero = torch.zeros_like(bra_bm)
    if is_alpha:
        bra_det = det_from_alpha_beta_batch(bra_bm, zero, n_orb)
    else:
        bra_det = det_from_alpha_beta_batch(zero, bra_bm, n_orb)
    element = two_excite_batch(bra_det, None, an0, an1, cr0, cr1, integrals.I2_eri)
    if element.is_complex():
        # Keep the complex element as-is; the kernel does the pair arithmetic.
        return element.contiguous()
    return element.to(torch.float64).contiguous()


def _singles_source_factor(strings, offsets, cr, an, n_orb, integrals, *, is_alpha):
    """Source-spin factor of one singles table: the ``one_excite_batch`` sum.

    Args:
        strings: the α (``is_alpha``) or β string set the table's bra indices
            refer to, ``(M,)`` words or ``(M, n_limbs)`` limb matrices.
        offsets: ``(M+1,)`` CSR offsets of the table.
        cr, an: creation / annihilation spin-orbital indices of each entry.
        n_orb: spatial orbital count.
        integrals: the matvec :class:`TorchIntegrals` bundle.
        is_alpha: select the α- or β-spin table.

    Returns:
        ``(n_entries,)`` tensor in the ERI dtype holding
        ``I1[cr, an] + sum_j occ_source(j) * (I2[cr, an, p_j, p_j]
        - I2[cr, p_j, p_j, an])`` with ``p_j = 2j`` for α and ``2j + 1`` for β.
        The parity sign is not included; the kernel applies the full-det
        parity.

    Notes:
        This is the source-spin half of the ``one_excite_batch`` element,
        built with the batched fast kernels' expressions (``mult.
        _apply_alpha_single_chunk_fast`` / ``_beta_single_chunk_fast``). Only
        the source-spin occupancy enters: the opposite-spin exchange term
        vanishes in a dense TPB ERI (no mixed-spin quadrant), while the
        opposite-spin direct term is evaluated in the kernel against the
        spectator string.
    """
    device = strings.device
    dtype = integrals.I2_eri.dtype
    bra = _bra_from_offsets(offsets, int(strings.shape[0]), device)
    bra_bm = strings[bra]

    j = torch.arange(n_orb, dtype=torch.int64, device=device)
    j_spin = (2 * j + (0 if is_alpha else 1)).unsqueeze(0)  # (1, L)
    cr_b = cr.unsqueeze(-1)  # (n_exc, 1)
    an_b = an.unsqueeze(-1)
    int_vec = (
        integrals.I2_eri[cr_b, an_b, j_spin, j_spin] - integrals.I2_eri[cr_b, j_spin, j_spin, an_b]
    )
    occ = _string_occ_matrix(bra_bm, n_orb).to(dtype)
    element = integrals.I1[cr, an] + (int_vec * occ).sum(dim=-1)
    del int_vec, occ
    return element.contiguous()


def _string_n_limbs(adets_t, bdets_t, n_limbs: int = 1) -> int:
    """Resolve and cross-check the per-string limb count of two string tensors.

    A 1-D ``(M,)`` tensor carries one limb (``n_limbs == 1``); a 2-D
    ``(M, n_limbs)`` tensor carries one int64 word per limb, limb ``k`` holding
    bits ``[64k, 64k + 64)``. ``n_limbs`` defaults to the single-word value, so
    every existing call site keeps working: a 2-D tensor makes the width
    authoritative, and an explicit ``n_limbs > 1`` that disagrees is rejected.
    """
    widths = set()
    for tensor in (adets_t, bdets_t):
        if tensor.ndim == 2:
            widths.add(int(tensor.shape[1]))
        elif tensor.ndim != 1:
            raise ValueError(f"string tensors must be 1-D or 2-D, got ndim={tensor.ndim}")
    if adets_t.ndim != bdets_t.ndim:
        raise ValueError(
            "string tensors must share one representation: both 1-D single words or both "
            f"(M, n_limbs) matrices; got ndim={adets_t.ndim} and ndim={bdets_t.ndim}"
        )
    if len(widths) > 1:
        raise ValueError(f"string tensors disagree on the limb count: {sorted(widths)}")
    width = widths.pop() if widths else 1
    if int(n_limbs) > 1 and int(n_limbs) != width:
        raise ValueError(
            f"n_limbs={n_limbs} does not match the string tensor width {width}; pass "
            "n_limbs equal to the trailing axis of the (M, n_limbs) string matrices"
        )
    return width


def _singles_layout_supported(
    adets_t,
    bdets_t,
    offsets,
    to,
    cr,
    an,
    T,
    Wb,
    *,
    M,
    n_orb,
    factor_dtype,
) -> bool:
    """Whether strings, tables and buffers fit the tiled singles kernel.

    The kernel loads every table through a typed int64 pointer and derives the
    spectator range from the string bit positions, so it needs contiguous int64
    tables and int64 contiguous 1-D strings or limb matrices wide enough for
    ``n_orb``. ``M`` is the source-spin row count (``M_a`` for the α pass,
    ``M_b`` for the β pass) and ``factor_dtype`` the dtype
    :func:`_singles_source_factor` would produce; a real/complex mismatch with
    ``T``/``Wb`` cannot share one ``IS_COMPLEX`` flag. A rejected layout never
    raises: the caller keeps the batched fast path.
    """
    tables = (offsets, to, cr, an)
    if any(
        not isinstance(t, torch.Tensor)
        or t.dtype != torch.int64
        or not t.is_contiguous()
        or t.ndim != 1
        for t in tables
    ):
        return False
    if offsets.numel() != int(M) + 1 or to.numel() != cr.numel() or cr.numel() != an.numel():
        return False

    widths = set()
    for strings in (adets_t, bdets_t):
        if (
            not isinstance(strings, torch.Tensor)
            or strings.dtype != torch.int64
            or not strings.is_contiguous()
            or strings.ndim not in (1, 2)
        ):
            return False
        width = 1 if strings.ndim == 1 else int(strings.shape[1])
        if width < 1 or 64 * width < int(n_orb):
            return False
        widths.add(width)
    if len(widths) != 1:
        return False

    if not isinstance(T, torch.Tensor) or not isinstance(Wb, torch.Tensor):
        return False
    if T.ndim != 1 or Wb.ndim != 1 or not T.is_contiguous() or not Wb.is_contiguous():
        return False
    if bool(factor_dtype.is_complex) != bool(T.dtype.is_complex):
        return False
    if bool(Wb.dtype.is_complex) != bool(T.dtype.is_complex):
        return False
    if T.device != Wb.device or T.device != adets_t.device or T.device != bdets_t.device:
        return False
    return True


def _kernel_complex_args(*tensors):
    """Return ``(is_complex, kernel_args)`` for one matvec launch.

    Triton has no complex scalar type, so a complex tensor is handed to the
    kernel as its interleaved ``(re, im)`` real view (``view_as_real``, which
    preserves torch's complex memory layout) and the kernel's ``IS_COMPLEX``
    branch does the pair arithmetic. Real tensors pass through untouched. A
    launch mixing complex and real operands is rejected: the single flag
    governs every load and store of the kernel.
    """
    flags = {bool(tensor.dtype.is_complex) for tensor in tensors}
    if len(flags) != 1:
        raise ValueError(
            "a Triton matvec launch cannot mix complex and real numeric operands; got dtypes "
            + ", ".join(str(tensor.dtype) for tensor in tensors)
        )
    is_complex = flags.pop()
    if is_complex:
        return True, tuple(torch.view_as_real(tensor) for tensor in tensors)
    return False, tensors


def _launch_alphabeta(
    adets_t,
    bdets_t,
    a_offsets,
    a_cr,
    a_an,
    a_to,
    b_offsets,
    b_cr,
    b_an,
    b_to,
    I2_flat,
    T,
    Wb,
    M_b,
    n_spinorb,
    N,
    *,
    T_stride=None,
    n_limbs=1,
    BLOCK=128,
):
    """Launch the tiled αβ-doubles kernel on explicit tables.

    Thin wrapper around the kernel: every table may be a view or a stage-local
    remap, which is what the distributed owner-computes operator passes.
    ``N`` is the number of outputs this launch owns and ``adets_t`` indexes
    their α rows. ``T_stride`` is the row width of ``T`` (default: ``M_b``);
    a 2-D tile stage buffer is narrower than the owned output width.
    ``n_limbs`` is the trailing width of the string tensors (``1`` for a 1-D
    ``(M,)`` tensor). A complex ``I2_flat``/``T``/``Wb`` triple is launched
    through its interleaved real views with ``IS_COMPLEX`` set.
    """
    stride = M_b if T_stride is None else int(T_stride)
    n_limbs = _string_n_limbs(adets_t, bdets_t, n_limbs)
    is_complex, (I2_arg, T_arg, Wb_arg) = _kernel_complex_args(I2_flat, T, Wb)
    grid = (triton.cdiv(N, BLOCK),)
    _alphabeta_kernel[grid](
        adets_t,
        bdets_t,
        a_offsets,
        a_cr,
        a_an,
        a_to,
        b_offsets,
        b_cr,
        b_an,
        b_to,
        I2_arg,
        T_arg,
        Wb_arg,
        M_b,
        n_spinorb,
        N,
        stride,
        IS_COMPLEX=1 if is_complex else 0,
        N_LIMBS=n_limbs,
        BLOCK=BLOCK,
    )


def _launch_samespin_double(
    adets_t,
    bdets_t,
    offsets,
    to,
    cr0,
    cr1,
    an0,
    an1,
    eij,
    T,
    Wb,
    M_b,
    N,
    *,
    IS_ALPHA,
    RANGE_OFF,
    T_stride=None,
    n_limbs=1,
    BLOCK=128,
):
    """Launch the tiled same-spin doubles kernel on explicit tables.

    ``eij`` is the precomputed integral x same-spin-sign factor
    (:func:`_samespin_eij`); ``IS_ALPHA`` / ``RANGE_OFF`` are the kernel's
    constexpr selectors (α: ``1`` / ``0``, β: ``0`` / ``1``). ``T_stride`` is
    the row width of ``T`` (default: ``M_b``); a 2-D tile stage buffer holds
    only that stage's ket columns and is narrower than the output width.
    ``n_limbs`` is the trailing width of the string tensors (``1`` for a 1-D
    ``(M,)`` tensor). A complex ``eij``/``T``/``Wb`` triple is launched
    through its interleaved real views with ``IS_COMPLEX`` set.
    """
    stride = M_b if T_stride is None else int(T_stride)
    n_limbs = _string_n_limbs(adets_t, bdets_t, n_limbs)
    is_complex, (eij_arg, T_arg, Wb_arg) = _kernel_complex_args(eij, T, Wb)
    grid = (triton.cdiv(N, BLOCK),)
    _samespin_double_kernel[grid](
        adets_t,
        bdets_t,
        offsets,
        to,
        cr0,
        cr1,
        an0,
        an1,
        eij_arg,
        T_arg,
        Wb_arg,
        M_b,
        N,
        stride,
        IS_ALPHA=IS_ALPHA,
        RANGE_OFF=RANGE_OFF,
        IS_COMPLEX=1 if is_complex else 0,
        N_LIMBS=n_limbs,
        BLOCK=BLOCK,
    )


def _launch_singles(
    adets_t,
    bdets_t,
    offsets,
    to,
    cr,
    an,
    e0,
    I2_flat,
    T,
    Wb,
    M_b,
    N,
    n_spinorb,
    *,
    IS_ALPHA,
    T_stride=None,
    n_limbs=1,
    BLOCK=128,
):
    """Launch the tiled same-spin singles kernel on explicit tables.

    ``e0`` is the precomputed source-spin factor
    (:func:`_singles_source_factor`) and ``IS_ALPHA`` selects the spin channel
    (α: the table is keyed by ``ia``, the β string is the spectator; β: keyed
    by ``ib``, the α string is the spectator). ``T_stride`` is the row width of
    ``T`` (default: ``M_b``); a 2-D tile stage buffer holds only that stage's
    ket columns and can be narrower than the owned output width. ``n_limbs`` is
    the trailing width of the string tensors (``1`` for a 1-D ``(M,)`` tensor).
    A complex ``e0``/``I2_flat``/``T``/``Wb`` set is launched through its
    interleaved real views with ``IS_COMPLEX`` set.
    """
    stride = M_b if T_stride is None else int(T_stride)
    n_limbs = _string_n_limbs(adets_t, bdets_t, n_limbs)
    is_complex, (e0_arg, I2_arg, T_arg, Wb_arg) = _kernel_complex_args(e0, I2_flat, T, Wb)
    grid = (triton.cdiv(N, BLOCK),)
    _singles_kernel[grid](
        adets_t,
        bdets_t,
        offsets,
        to,
        cr,
        an,
        e0_arg,
        I2_arg,
        T_arg,
        Wb_arg,
        M_b,
        N,
        stride,
        n_spinorb,
        IS_ALPHA=IS_ALPHA,
        IS_COMPLEX=1 if is_complex else 0,
        N_LIMBS=n_limbs,
        BLOCK=BLOCK,
    )


def build_samespin_cache(adets_t, bdets_t, n_orb, integrals, helpers) -> dict:
    """Precompute the Triton path's ``T``-independent setup.

    Args:
        adets_t, bdets_t: full α/β string bitmasks.
        n_orb: spatial orbital count.
        integrals: the matvec :class:`TorchIntegrals` bundle.
        helpers: iterable of TorchHelpers.

    Returns:
        ``alpha_eij`` / ``beta_eij`` (same-spin doubles factors),
        ``alpha_singles_e0`` / ``beta_singles_e0`` (same-spin singles
        source-spin factors), each present only when the corresponding table
        is non-empty, and ``beta_pass``
        (:func:`~torch_sci_solver.torch_impl.mult.build_beta_pass_cache` output
        for the batched β passes).

    Notes:
        The same-spin ``eij`` and singles ``e0`` depend only on the bra strings
        and orbital indices, so rebuilding them on every matvec was pure
        overhead. The cached values come from the exact expressions the
        batched path uses, so results stay bit-identical.
    """
    helpers = tuple(helpers)
    cache: dict = {}
    for h in helpers:
        if h.task_type == 2:
            if h.a_doubles_to is not None and h.a_doubles_to.numel() > 0:
                cache["alpha_eij"] = _samespin_eij(
                    adets_t,
                    h.a_doubles_offsets,
                    h.a_doubles_cr0,
                    h.a_doubles_cr1,
                    h.a_doubles_an0,
                    h.a_doubles_an1,
                    n_orb,
                    integrals,
                    is_alpha=True,
                )
            if h.a_singles_to is not None and h.a_singles_to.numel() > 0:
                cache["alpha_singles_e0"] = _singles_source_factor(
                    adets_t,
                    h.a_singles_offsets,
                    h.a_singles_cr,
                    h.a_singles_an,
                    n_orb,
                    integrals,
                    is_alpha=True,
                )
        elif h.task_type == 1:
            if h.b_doubles_to is not None and h.b_doubles_to.numel() > 0:
                cache["beta_eij"] = _samespin_eij(
                    bdets_t,
                    h.b_doubles_offsets,
                    h.b_doubles_cr0,
                    h.b_doubles_cr1,
                    h.b_doubles_an0,
                    h.b_doubles_an1,
                    n_orb,
                    integrals,
                    is_alpha=False,
                )
            if h.b_singles_to is not None and h.b_singles_to.numel() > 0:
                cache["beta_singles_e0"] = _singles_source_factor(
                    bdets_t,
                    h.b_singles_offsets,
                    h.b_singles_cr,
                    h.b_singles_an,
                    n_orb,
                    integrals,
                    is_alpha=False,
                )
    cache["beta_pass"] = build_beta_pass_cache(adets_t, bdets_t, n_orb, integrals, helpers)
    return cache


def _apply_samespin_doubles_triton(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    h,
    M_a,
    M_b,
    N,
    *,
    is_alpha,
    BLOCK,
    eij=None,
    n_limbs=1,
):
    """Tiled output-stationary same-spin doubles pass.

    ``eij`` (integral x same-spin sign) is precomputed on the host with the
    same ``two_excite_batch`` the batched path uses, so only the cheap
    spectator-string parity is evaluated in the kernel. Pass the tensor cached
    by :func:`build_samespin_cache` to skip the rebuild; ``None`` computes it
    here exactly as before. ``n_limbs`` is the trailing width of the string
    tensors.
    """
    _ = M_a  # retained for call-site signature compatibility
    if is_alpha:
        off, to = h.a_doubles_offsets, h.a_doubles_to
        cr0, cr1 = h.a_doubles_cr0, h.a_doubles_cr1
        an0, an1 = h.a_doubles_an0, h.a_doubles_an1
        range_off = 0
    else:
        off, to = h.b_doubles_offsets, h.b_doubles_to
        cr0, cr1 = h.b_doubles_cr0, h.b_doubles_cr1
        an0, an1 = h.b_doubles_an0, h.b_doubles_an1
        range_off = 1

    if to is None or to.shape[0] == 0:
        return

    if eij is None:
        eij = _samespin_eij(
            adets_t if is_alpha else bdets_t,
            off,
            cr0,
            cr1,
            an0,
            an1,
            n_orb,
            integrals,
            is_alpha=is_alpha,
        )
    _launch_samespin_double(
        adets_t,
        bdets_t,
        off,
        to,
        cr0,
        cr1,
        an0,
        an1,
        eij,
        T,
        Wb,
        M_b,
        N,
        IS_ALPHA=1 if is_alpha else 0,
        RANGE_OFF=range_off,
        n_limbs=n_limbs,
        BLOCK=BLOCK,
    )


def _apply_singles_triton(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    h,
    M_a,
    M_b,
    N,
    *,
    is_alpha,
    BLOCK,
    chunk_size,
    I2_flat,
    e0=None,
    beta_pass=None,
    n_limbs=1,
):
    """Tiled output-stationary same-spin singles pass.

    The source-spin factor ``e0`` is precomputed on the host
    (:func:`_singles_source_factor`); pass the tensor cached by
    :func:`build_samespin_cache` to skip the rebuild. A table/string/buffer
    layout outside :func:`_singles_layout_supported` falls back to the batched
    fast kernel (``beta_pass`` is its β-run cache), so the result is unchanged.
    """
    if is_alpha:
        off, to = h.a_singles_offsets, h.a_singles_to
        cr, an = h.a_singles_cr, h.a_singles_an
        rows, strings = M_a, adets_t
    else:
        off, to = h.b_singles_offsets, h.b_singles_to
        cr, an = h.b_singles_cr, h.b_singles_an
        rows, strings = M_b, bdets_t
    if to is None or to.shape[0] == 0:
        return

    if _singles_layout_supported(
        adets_t,
        bdets_t,
        off,
        to,
        cr,
        an,
        T,
        Wb,
        M=rows,
        n_orb=n_orb,
        factor_dtype=integrals.I2_eri.dtype,
    ):
        if e0 is None:
            e0 = _singles_source_factor(strings, off, cr, an, n_orb, integrals, is_alpha=is_alpha)
        _launch_singles(
            adets_t,
            bdets_t,
            off,
            to,
            cr,
            an,
            e0,
            I2_flat,
            T,
            Wb,
            M_b,
            N,
            int(integrals.n_spinorb),
            IS_ALPHA=1 if is_alpha else 0,
            n_limbs=n_limbs,
            BLOCK=BLOCK,
        )
        return

    bra = _bra_from_offsets(off, rows, strings.device)
    if is_alpha:
        _apply_alpha_single_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_alpha_per_exc=bra,
            ket_alpha_per_exc=to,
            cr=cr,
            an=an,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
        )
    else:
        _apply_beta_single_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_beta_per_exc=bra,
            ket_beta_per_exc=to,
            cr=cr,
            an=an,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
            cache=beta_pass,
        )


def mult_triton(
    hii, T, adets_t, bdets_t, n_orb, integrals, helpers, *, chunk_size=2_000_000, cache=None
):
    """``Wb = H @ T`` with the αβ, same-spin doubles and singles passes tiled.

    ``cache`` is the output of :func:`build_samespin_cache`; ``None`` (default)
    recomputes the same-spin ``eij``, the singles source factors and the
    β-singles setup on every call, exactly as before. A singles table whose
    layout the kernel cannot address keeps the batched fast path.
    """
    device = adets_t.device
    if device.type != "cuda":
        raise RuntimeError(
            f"matvec_impl='triton' requires a CUDA device; got device={device}. "
            "Use matvec_impl='batched' on CPU."
        )
    if not _HAVE_TRITON:
        raise RuntimeError(
            "matvec_impl='triton' requires Triton (pip install triton). "
            "Use matvec_impl='batched' otherwise."
        )
    # The kernels load the CSR payloads through typed int64 pointers; a
    # narrowed (int32) table would be read at the wrong stride.
    payload_fields = (
        "a_singles_to",
        "a_singles_cr",
        "a_singles_an",
        "a_doubles_to",
        "a_doubles_cr0",
        "a_doubles_cr1",
        "a_doubles_an0",
        "a_doubles_an1",
        "b_singles_to",
        "b_singles_cr",
        "b_singles_an",
        "b_doubles_to",
        "b_doubles_cr0",
        "b_doubles_cr1",
        "b_doubles_an0",
        "b_doubles_an1",
    )
    for h in helpers:
        for field in payload_fields:
            value = getattr(h, field, None)
            if value is not None and value.dtype != torch.int64:
                raise NotImplementedError(
                    "matvec_impl='triton' requires int64 table payloads; build the "
                    "helpers with table_dtype=torch.int64 or use matvec_impl='batched'"
                )

    cache = {} if cache is None else cache
    M_a, M_b = len(adets_t), len(bdets_t)
    N = M_a * M_b
    n_limbs = _string_n_limbs(adets_t, bdets_t)
    Wb = hii * T
    I2_flat = integrals.I2_eri.reshape(-1).contiguous()
    n_spinorb = int(integrals.n_spinorb)
    BLOCK = 128

    for h in helpers:
        if h.task_type == 2:  # α-only: singles and doubles tiled
            _apply_singles_triton(
                Wb,
                T,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                h,
                M_a,
                M_b,
                N,
                is_alpha=True,
                BLOCK=BLOCK,
                chunk_size=chunk_size,
                I2_flat=I2_flat,
                e0=cache.get("alpha_singles_e0"),
                n_limbs=n_limbs,
            )
            _apply_samespin_doubles_triton(
                Wb,
                T,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                h,
                M_a,
                M_b,
                N,
                is_alpha=True,
                BLOCK=BLOCK,
                eij=cache.get("alpha_eij"),
                n_limbs=n_limbs,
            )
        elif h.task_type == 1:  # β-only: singles and doubles tiled
            _apply_singles_triton(
                Wb,
                T,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                h,
                M_a,
                M_b,
                N,
                is_alpha=False,
                BLOCK=BLOCK,
                chunk_size=chunk_size,
                I2_flat=I2_flat,
                e0=cache.get("beta_singles_e0"),
                beta_pass=cache.get("beta_pass"),
                n_limbs=n_limbs,
            )
            _apply_samespin_doubles_triton(
                Wb,
                T,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                h,
                M_a,
                M_b,
                N,
                is_alpha=False,
                BLOCK=BLOCK,
                eij=cache.get("beta_eij"),
                n_limbs=n_limbs,
            )
        elif h.task_type == 0:  # αβ-doubles: tiled inline element
            if h.a_singles_to is None or h.b_singles_to is None:
                continue
            if h.a_singles_to.shape[0] == 0 or h.b_singles_to.shape[0] == 0:
                continue
            _launch_alphabeta(
                adets_t,
                bdets_t,
                h.a_singles_offsets,
                h.a_singles_cr,
                h.a_singles_an,
                h.a_singles_to,
                h.b_singles_offsets,
                h.b_singles_cr,
                h.b_singles_an,
                h.b_singles_to,
                I2_flat,
                T,
                Wb,
                M_b,
                n_spinorb,
                N,
                n_limbs=n_limbs,
                BLOCK=BLOCK,
            )
        else:
            raise ValueError(f"unknown task_type {h.task_type}")
    return Wb
