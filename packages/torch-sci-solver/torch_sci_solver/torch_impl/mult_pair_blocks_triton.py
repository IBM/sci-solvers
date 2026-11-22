"""Output-stationary Triton kernel for the GDB same-spin doubles matvec.

The batched GDB kernels of :mod:`~torch_sci_solver.torch_impl.mult_pair_blocks`
lift every surviving move over the group's spectator axis and materialise a
``(excitation x spectator)`` grid per survivor slice, then scatter it with
``index_add_``. This module replaces that lift for the two same-spin doubles
channels with one lane per output, accumulating its own move list in a
register --- the structure of the product-mode output-stationary kernels of
:mod:`~torch_sci_solver.torch_impl.mult_triton`, applied to an explicit move
table instead of a CSR excitation table.

Only the same-spin **doubles** channels are tiled. Their matrix element
factorises into an integral times same-spin sign factor per move ---
computable on the host with ordinary :class:`TorchIntegrals` indexing, exactly
as the batched product kernel computes it --- times a spectator-string parity
evaluated in the kernel; the ERI is therefore never read inside the kernel.
The cross-spin (:math:`\\alpha\\beta`) element couples the two moves through
the integral indices and has no per-move factor, so it stays batched, as do
the singles channels.

Complex dtypes (``complex64``/``complex128``) run through the same kernel.
Triton has no complex scalar type, so the kernel's complex operands are
passed as their interleaved ``(re, im)`` real views --- exactly torch's
complex memory layout --- and an ``IS_COMPLEX`` constexpr selects the pair
arithmetic, as in :mod:`~torch_sci_solver.torch_impl.mult_triton`: the two
parts are loaded separately and multiplied with the usual four-multiply rule,
arithmetic accumulates in ``float64`` for both complex precisions, and only
the loads and the final store use the storage precision.

Dispatch is explicit: :func:`same_spin_doubles_available` gates on Triton,
CUDA and a ``float32``/``float64``/``complex64``/``complex128`` dtype shared
by the output, input and ERI, and :func:`apply_same_spin_doubles` raises when
it cannot run rather than silently falling back.

Notes
-----
A same-spin double move of one spin takes a source string to a target (bra)
string. With ``I = min(an0, an1) // 2``, ``J = max(an0, an1) // 2``,
``A = min(cr0, cr1) // 2`` and ``B = max(cr0, cr1) // 2`` in spectator-orbital
units, and with :math:`\\sigma` the spectator string, the move's element is

.. math::

    H_m = e_m \\, (-1)^{p_{[\\min(I, A), \\max(I, A))}(\\sigma)
        + p_{[\\min(J, B), \\max(J, B))}(\\sigma)},

where :math:`e_m` is the integral times same-spin-sign factor evaluated on
the bra determinant and :math:`p` is a string popcount. For α doubles
:math:`\\sigma` is the β string at the output column and the ranges are not
shifted; for β doubles :math:`\\sigma` is the α string at the output row and
the ranges are shifted by one orbital (the interleaved-Wick encoding). The
output update is

.. math::

    W[\\mathrm{bra}, c] \\mathrel{+}= H_m T[\\mathrm{ket}, c]

for α doubles and the mirrored ``W[r, bra] += H_m T[r, ket]`` for β doubles.

The enumerators emit moves grouped by their first output position. The
inverse (owner-computes) enumerators walk bra positions, so their table is
already grouped by destination and one ``bincount`` + ``cumsum`` yields the
CSR offsets the kernel walks; the forward full-vector enumerators group by
ket position instead, and the launcher stable-sorts those slices first.
Nothing in the kernel depends on the ERI's storage layout, so a compact ERI
is accepted unchanged.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import det_from_alpha_beta_batch, two_excite_batch
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_triton import _kernel_complex_args

try:  # Triton is optional and CUDA-only; the package must import without it.
    import triton
    import triton.language as tl

    from torch_sci_solver.torch_impl.mult_triton import (
        _string_n_limbs,
        _string_range_popcount,
    )

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    _HAVE_TRITON = False

# Outputs per program. A block of consecutive outputs usually shares one
# destination row (α) or spans two output rows (β), so the per-move loads on
# its lanes are broadcast or nearly so.
_BLOCK = 128


def same_spin_doubles_available(out: Tensor, T: Tensor, integrals: TorchIntegrals) -> bool:
    """Whether the Triton same-spin doubles kernel can run this call.

    Args:
        out: fragment-local output block the kernel would accumulate into.
        T: staged input buffer the kernel would read.
        integrals: integral bundle holding the ERI the host-side factor reads.

    Returns:
        ``True`` when Triton is importable, both tensors and the ERI live on
        one CUDA device, and the arithmetic dtype is one of ``float32``,
        ``float64``, ``complex64`` or ``complex128`` shared by ``out``, ``T``
        and the ERI.
    """
    if not _HAVE_TRITON:
        return False
    if not (out.is_cuda and T.is_cuda):
        return False
    if out.device != T.device or integrals.I2_eri.device != out.device:
        return False
    if out.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return False
    return T.dtype == out.dtype and integrals.I2_eri.dtype == out.dtype


if _HAVE_TRITON:

    @triton.jit
    def _same_spin_doubles_kernel(
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
        IS_FP64: tl.constexpr,
        IS_COMPLEX: tl.constexpr,
        N_LIMBS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Accumulate one survivor slice, one lane per output.

        ``M_b`` is the output width; ``T_stride`` is the row width of the ``T``
        buffer, which differs from ``M_b`` for a β slab (the buffer holds only
        the source columns) and for an owner-computes α slice (the buffer holds
        the whole ket group). ``off_ptr`` holds the CSR offsets of the move
        table over destination positions. ``N_LIMBS`` sizes the per-string limb
        axis (``1`` for a 1-D string tensor). ``IS_FP64`` selects the
        ``float64`` accumulator; ``IS_COMPLEX`` reads ``eij``/``T``/``Wb`` as
        interleaved ``(re, im)`` real pairs and accumulates the pair
        arithmetic.
        """
        out = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        active = out < N
        ia = out // M_b
        ib = out % M_b

        if IS_ALPHA:
            key = ia  # walk the moves whose α-bra is this output row
            sign_ptr = bdets_ptr  # the β string carries the parity sign
            sign_rows = ib
        else:
            key = ib  # walk the moves whose β-bra is this output column
            sign_ptr = adets_ptr  # the α string carries the parity sign
            sign_rows = ia

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
            out2 = out * 2
            acc_re = tl.load(Wb_ptr + out2, mask=active, other=0.0).to(tl.float64)
            acc_im = tl.load(Wb_ptr + out2 + 1, mask=active, other=0.0).to(tl.float64)
        else:
            acc = tl.load(Wb_ptr + out, mask=active, other=0.0)

        for jj in range(0, max_c):
            on = active & (jj < cnt)
            idx = tl.where(on, lo + jj, 0)
            ket = tl.load(to_ptr + idx, mask=on, other=0).to(tl.int64)
            cr0 = tl.load(cr0_ptr + idx, mask=on, other=0).to(tl.int64)
            cr1 = tl.load(cr1_ptr + idx, mask=on, other=0).to(tl.int64)
            an0 = tl.load(an0_ptr + idx, mask=on, other=0).to(tl.int64)
            an1 = tl.load(an1_ptr + idx, mask=on, other=0).to(tl.int64)
            if IS_COMPLEX:
                # idx is a complex-element offset into the (re, im) view.
                idx2 = idx.to(tl.int64) * 2
                e_re = tl.load(eij_ptr + idx2, mask=on, other=0.0).to(tl.float64)
                e_im = tl.load(eij_ptr + idx2 + 1, mask=on, other=0.0).to(tl.float64)
            else:
                eij = tl.load(eij_ptr + idx, mask=on, other=0.0)

            # Spectator parity over the two orbital ranges of the move.
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
            if IS_FP64:
                sgn = (1 - 2 * (pop & 1)).to(tl.float64)
            else:
                sgn = (1 - 2 * (pop & 1)).to(tl.float32)

            if IS_COMPLEX:
                if IS_ALPHA:
                    t2 = (ket * T_stride + ib) * 2
                else:
                    t2 = (ia * T_stride + ket) * 2
                t_re = tl.load(T_ptr + t2, mask=on, other=0.0).to(tl.float64)
                t_im = tl.load(T_ptr + t2 + 1, mask=on, other=0.0).to(tl.float64)
                acc_re += tl.where(on, sgn * (e_re * t_re - e_im * t_im), 0.0)
                acc_im += tl.where(on, sgn * (e_re * t_im + e_im * t_re), 0.0)
            else:
                if IS_ALPHA:
                    t_val = tl.load(T_ptr + ket * T_stride + ib, mask=on, other=0.0)
                else:
                    t_val = tl.load(T_ptr + ia * T_stride + ket, mask=on, other=0.0)
                acc += tl.where(on, eij * sgn * t_val, 0.0)

        if IS_COMPLEX:
            tl.store(Wb_ptr + out2, acc_re.to(Wb_ptr.dtype.element_ty), mask=active)
            tl.store(Wb_ptr + out2 + 1, acc_im.to(Wb_ptr.dtype.element_ty), mask=active)
        else:
            tl.store(Wb_ptr + out, acc, mask=active)


def _launch_same_spin_doubles(
    adets: Tensor,
    bdets: Tensor,
    offsets: Tensor,
    ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    eij: Tensor,
    T: Tensor,
    out: Tensor,
    M_b: int,
    N: int,
    T_stride: int,
    *,
    is_alpha: bool,
    n_limbs: int,
    BLOCK: int,
) -> None:
    """Launch the same-spin doubles kernel on explicit tables.

    ``eij`` is the host-side integral times same-spin-sign factor,
    ``IS_ALPHA`` / ``RANGE_OFF`` are the kernel's constexpr selectors (α:
    ``1`` / ``0``, β: ``0`` / ``1``), and ``N`` is the number of outputs this
    launch owns. A complex ``eij``/``T``/``out`` triple is launched through
    its interleaved real views with ``IS_COMPLEX`` set; any mix of complex and
    real operands is rejected because the single flag governs every load and
    store. ``IS_FP64`` selects the ``float64`` accumulator --- always on for
    complex, whose real parts are upcast before the pair arithmetic.
    """
    is_complex, (eij_arg, T_arg, out_arg) = _kernel_complex_args(eij, T, out)
    grid = (triton.cdiv(N, int(BLOCK)),)
    _same_spin_doubles_kernel[grid](
        adets,
        bdets,
        offsets,
        ket,
        cr0,
        cr1,
        an0,
        an1,
        eij_arg,
        T_arg,
        out_arg,
        M_b,
        N,
        T_stride,
        IS_ALPHA=1 if is_alpha else 0,
        RANGE_OFF=0 if is_alpha else 1,
        IS_FP64=1 if (is_complex or out.dtype == torch.float64) else 0,
        IS_COMPLEX=1 if is_complex else 0,
        N_LIMBS=n_limbs,
        BLOCK=int(BLOCK),
    )


def _move_element_factor(
    bra: Tensor,
    an0: Tensor,
    an1: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    strings: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    is_alpha: bool,
) -> Tensor:
    """Integral times same-spin-sign factor for every move of one slice.

    This is the expression the batched same-spin doubles product kernels
    evaluate (``two_excite_batch`` on the bra determinant with the move's
    annihilation and creation orbitals), specialised to the kernel's
    spectator-parity split. The bra determinant is built from the bra string
    of the destination position, so the sign factor is exact for both the
    full-vector and the inverse (owner-computes) enumeration directions.
    """
    bra_strings = strings[bra]
    zero = torch.zeros_like(bra_strings)
    if is_alpha:
        bra_det = det_from_alpha_beta_batch(bra_strings, zero, n_orb)
    else:
        bra_det = det_from_alpha_beta_batch(zero, bra_strings, n_orb)
    return two_excite_batch(bra_det, None, an0, an1, cr0, cr1, integrals.I2_eri)


def apply_same_spin_doubles(
    out: Tensor,
    T: Tensor,
    adets: Tensor,
    bdets: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    bra: Tensor,
    ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    M_a: int,
    M_b: int,
    is_alpha: bool,
    T_stride: int,
    bra_sorted: bool = True,
    BLOCK: int = _BLOCK,
) -> None:
    """Add one same-spin doubles survivor slice with the Triton kernel.

    The output block is ``(M_a, M_b)`` row-major; ``bra`` holds each move's
    destination position --- a row for α doubles, a column for β doubles ---
    and ``ket`` holds its source position (a ket row for α, a ket column for
    β). ``T`` is the input buffer with row width ``T_stride``: an α move reads
    ``T[ket_row, column]`` and a β move reads ``T[row, ket_column]``. The
    spectator-string parity is evaluated on ``bdets`` (α doubles) or ``adets``
    (β doubles). The inverse (owner-computes) enumerators emit moves grouped
    by destination position, so ``bra`` is already non-decreasing there;
    ``bra_sorted=False`` stable-sorts the slice first, which the forward
    full-vector enumerators need because they emit moves grouped by source
    position.

    Args:
        out: ``(M_a * M_b,)`` output block, accumulated in place. May be
            complex.
        T: input buffer, 1-D row-major with width ``T_stride``. May be
            complex.
        adets: α strings of the output rows, ``(M_a,)`` or ``(M_a, n_limbs)``.
        bdets: β strings of the output columns, ``(M_b,)`` or ``(M_b, n_limbs)``.
        n_orb: spatial orbital count.
        integrals: integral bundle on ``out``'s device.
        bra: ``(m,)`` destination positions.
        ket: ``(m,)`` source positions along the moving spin's axis.
        cr0, cr1, an0, an1: ``(m,)`` creation and annihilation spin-orbital
            indices of each move.
        M_a, M_b: output block dimensions.
        is_alpha: ``True`` for an α-doubles move (β spectator), ``False``
            for β doubles.
        T_stride: row width of ``T``.
        bra_sorted: whether ``bra`` is already non-decreasing.
        BLOCK: outputs per Triton program.

    Raises:
        RuntimeError: Triton is unavailable, the tensors are not on one CUDA
            device, or the arithmetic dtype is not a shared
            ``float32``/``float64``/``complex64``/``complex128``.
        ValueError: the spectator string tensors do not share one limb layout.
    """
    if not same_spin_doubles_available(out, T, integrals):
        raise RuntimeError(
            "same-spin Triton matvec requires Triton, one CUDA device, and one "
            "float32/float64/complex64/complex128 dtype shared by out/T/ERI; "
            f"got out={out.dtype} on {out.device}, T={T.dtype} on {T.device}, "
            f"ERI={integrals.I2_eri.dtype} on {integrals.I2_eri.device}"
        )
    if int(bra.numel()) == 0:
        return

    device = out.device
    bra = bra.to(device=device, dtype=torch.int64).contiguous()
    ket = ket.to(device=device, dtype=torch.int64).contiguous()
    cr0 = cr0.to(device=device, dtype=torch.int64).contiguous()
    cr1 = cr1.to(device=device, dtype=torch.int64).contiguous()
    an0 = an0.to(device=device, dtype=torch.int64).contiguous()
    an1 = an1.to(device=device, dtype=torch.int64).contiguous()
    adets = adets.contiguous()
    bdets = bdets.contiguous()
    n_limbs = _string_n_limbs(adets, bdets)

    if not bra_sorted:
        order = torch.argsort(bra, stable=True)
        bra = bra[order]
        ket = ket[order]
        cr0 = cr0[order]
        cr1 = cr1[order]
        an0 = an0[order]
        an1 = an1[order]

    n_dest = int(M_a) if is_alpha else int(M_b)
    counts = torch.bincount(bra, minlength=n_dest)
    offsets = torch.empty(int(counts.numel()) + 1, dtype=torch.int64, device=device)
    offsets[0] = 0
    torch.cumsum(counts, 0, out=offsets[1:])

    eij = _move_element_factor(
        bra,
        an0,
        an1,
        cr0,
        cr1,
        adets if is_alpha else bdets,
        int(n_orb),
        integrals,
        is_alpha=is_alpha,
    ).contiguous()
    T = T.contiguous()

    n_outputs = int(M_a) * int(M_b)
    if n_outputs == 0:
        return
    _launch_same_spin_doubles(
        adets,
        bdets,
        offsets,
        ket,
        cr0,
        cr1,
        an0,
        an1,
        eij,
        T,
        out,
        int(M_b),
        n_outputs,
        int(T_stride),
        is_alpha=is_alpha,
        n_limbs=n_limbs,
        BLOCK=BLOCK,
    )
