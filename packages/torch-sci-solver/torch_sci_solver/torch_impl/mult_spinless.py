"""Flat spin-orbital ("spinless") diagonal + matvec — no product-space, no α/β split.

Structurally identical to :mod:`torch_sci_solver.torch_impl.mult_pairs`'s
generic scatter pattern, simplified further: there is no
``det_from_alpha_beta_batch`` reconstruction step (the basis already stores
flat determinants directly) and only one excitation-table kind per order
(no same-spin-α / same-spin-β / cross-spin-αβ split — a flat double
excitation is just "two creations, two annihilations over the whole
``n_spinorb`` space").

    for each row (k_bra, k_ket, cr, an) in the singles/doubles table:
        matelem = ⟨D_{k_bra} | H_{single/double} | D_{k_ket}⟩   # Slater-Condon
        Wb[k_bra] += matelem * T[k_ket]

Uses the flat-det Slater-Condon primitives from
:mod:`torch_sci_solver.torch_impl.determinants`
(``one_excite_batch``/``two_excite_batch``, unchanged — already spin-orbital-
index-generic) plus the spinless diagonal variant
(``zero_excite_batch_spinless``, which drops the same-spin exchange mask
``one_excite_batch``/``two_excite_batch`` never had in the first place).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from torch import Tensor

from torch_sci_solver.torch_impl.determinants import (
    one_excite_batch,
    two_excite_batch,
    zero_excite_batch_spinless,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_entries_triton import (
    apply_double_entries,
    apply_single_entries,
)
from torch_sci_solver.torch_impl.mult_pairs import DEFAULT_CHUNK_SIZE, _entry_chunk
from torch_sci_solver.torch_impl.spinless_helpers import SpinlessBasis, SpinlessHelpers

# Slater-Condon workspace per lifted entry, same rationale as
# mult_pairs._BYTES_PER_SINGLE_ENTRY / _BYTES_PER_DOUBLE_ENTRY (calibration
# numbers, not derived) — no det_from_alpha_beta_batch reconstruction here, so
# lean slightly lower. _entry_chunk itself is pure chunk-size arithmetic with
# no pair/spinless distinction, so it is imported rather than redefined.
_BYTES_PER_SINGLE_ENTRY = 3000
_BYTES_PER_DOUBLE_ENTRY = 48


def make_diag_terms_spinless_torch(
    basis: SpinlessBasis,
    integrals: TorchIntegrals,
) -> Tensor:
    """Diagonal of ``H`` over a flat spinless basis: ``hii[k] = ⟨D_k|H|D_k⟩``.

    Spinless analog of
    :func:`torch_sci_solver.torch_impl.mult_pairs.make_diag_terms_pair_torch`.

    Always returns a REAL tensor, even for complex integrals: the diagonal
    expectation value of a Hermitian operator is mathematically real (the
    complex einsum in ``zero_excite_batch_spinless`` carries only round-off
    imaginary noise around zero), and ``davidson_torch``'s multi-root path
    (``_make_seed_block``) sorts ``hii`` via ``torch.argsort``, which torch
    does not support for a complex dtype on CPU -- mirrors the identical fix
    already applied to the product-mode ``make_diag_terms_torch``.
    """
    device = integrals.device
    dtype = integrals.dtype
    hii_dtype = dtype.to_real() if dtype.is_complex else dtype
    dets = basis.dets.to(device)
    hii = zero_excite_batch_spinless(
        dets,
        None,
        integrals.n_spinorb,
        integrals.I0,
        integrals.I1_diag,
        integrals.I2_direct,
        integrals.I2_exchange,
    )
    return hii.real.to(hii_dtype) if dtype.is_complex else hii.to(hii_dtype)


def mult_spinless_torch(
    hii: Tensor,
    T: Tensor,
    basis: SpinlessBasis,
    helpers: SpinlessHelpers,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
) -> Tensor:
    """``Wb = H @ T`` over a flat spinless basis.

    Streams the singles and doubles excitation tables and scatters each
    entry's contribution into ``Wb``. The diagonal pass is a single
    length-N elementwise multiply.

    Args:
        hii: (N,) diagonal of H (from :func:`make_diag_terms_spinless_torch`).
        T: (N,) or (N, k) input vector/block.
        basis: the flat determinant list.
        helpers: excitation-table streamer; its rows are re-derived on each
            call rather than held in memory.
        integrals: integrals bundle (see
            :func:`torch_sci_solver.torch_impl.integrals.to_torch_integrals_spinless`)
            on the basis's device.
        chunk_size: max entries per Slater-Condon batch, applied within each
            streamed chunk.
        triton_matvec: route the singles and doubles tables to the
            entry-parallel Triton kernels
            (:func:`~torch_sci_solver.torch_impl.mult_entries_triton.
            apply_single_entries`, :func:`~torch_sci_solver.torch_impl.
            mult_entries_triton.apply_double_entries`). An unsupported
            device/layout falls back to the batched scatter, so the result
            never depends on the flag.

    Returns:
        Same shape as ``T``, on the same device as ``hii``.
    """
    n_dets = basis.dim
    if T.shape[0] != n_dets:
        raise ValueError(f"spinless T shape {T.shape} != ({n_dets}, ...)")

    Wb = hii.unsqueeze(-1) * T if T.ndim == 2 else hii * T
    device = Wb.device
    dets = basis.dets.to(device)
    budget = helpers.budget_bytes
    single_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_SINGLE_ENTRY)
    double_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_DOUBLE_ENTRY)

    for t in helpers.singles():
        if triton_matvec and apply_single_entries(
            Wb,
            T,
            integrals,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            mode="spinless",
            dets=dets,
        ):
            continue
        _scatter_single_spinless(
            Wb=Wb,
            T=T,
            dets=dets,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            integrals=integrals,
            chunk_size=single_chunk,
        )

    for t in helpers.doubles():
        if triton_matvec and apply_double_entries(
            Wb,
            T,
            integrals,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr0=t["cr0"],
            cr1=t["cr1"],
            an0=t["an0"],
            an1=t["an1"],
            mode="spinless",
            dets=dets,
        ):
            continue
        _scatter_double_spinless(
            Wb=Wb,
            T=T,
            dets=dets,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr0=t["cr0"],
            cr1=t["cr1"],
            an0=t["an0"],
            an1=t["an1"],
            integrals=integrals,
            chunk_size=double_chunk,
        )
    return Wb


def _scatter_single_spinless(
    *,
    Wb: Tensor,
    T: Tensor,
    dets: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    integrals: TorchIntegrals,
    chunk_size: int,
) -> None:
    """Scatter ``one_excite_batch`` contributions for the flat singles table.

    ``k_bra`` indexes the excitation's *source* determinant (occupied at
    ``an``, empty at ``cr``); the destination/excited determinant
    (``k_ket``, occupied at ``cr``, empty at ``an``) is ``H``'s *ket* here —
    mirrors ``sci_solvers.spinless_ci_problem._hij_spinless``'s
    ``one_excite(det_b, d[0], c[0], ...)`` convention exactly: the ket goes
    in as ``det``, its ket-only orbital (``cr``) as ``i``, the bra-only
    orbital (``an``) as ``a``. Getting this backwards produces the complex
    conjugate of the correct matrix element (indistinguishable from correct
    on real integrals, where a Hermitian matrix element is its own
    conjugate — caught only by a complex Hermitian cross-check).
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        ket = dets[kk]
        matelem = one_excite_batch(
            ket,
            None,
            cr[start:end],
            an[start:end],
            integrals.I1,
            integrals.I2_eri,
            integrals.n_spinorb,
        )
        _index_add(Wb, kb, matelem, T, kk)


def _scatter_double_spinless(
    *,
    Wb: Tensor,
    T: Tensor,
    dets: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    integrals: TorchIntegrals,
    chunk_size: int,
) -> None:
    """Scatter ``two_excite_batch`` contributions for the flat doubles table.

    Same bra/ket convention as :func:`_scatter_single_spinless` (see its
    docstring): the ket (``k_ket``, occupied at ``cr0, cr1``) is passed as
    ``det``, its ket-only orbitals (``cr0, cr1``) as ``(i, j)``, the bra-only
    orbitals (``an0, an1``) as ``(a, b)`` — mirrors
    ``two_excite(det_b, d[0], d[1], c[0], c[1], I2)`` in
    ``sci_solvers.spinless_ci_problem._hij_spinless``.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        ket = dets[kk]
        matelem = two_excite_batch(
            ket,
            None,
            cr0[start:end],
            cr1[start:end],
            an0[start:end],
            an1[start:end],
            integrals.I2_eri,
        )
        _index_add(Wb, kb, matelem, T, kk)


def _index_add(Wb: Tensor, kb: Tensor, matelem: Tensor, T: Tensor, kk: Tensor) -> None:
    """``Wb[kb] += matelem * T[kk]``, handling both a (N,) vector and (N, k) block."""
    if T.ndim == 2:
        Wb.index_add_(0, kb, matelem.unsqueeze(-1) * T[kk])
    else:
        Wb.index_add_(0, kb, matelem * T[kk])
