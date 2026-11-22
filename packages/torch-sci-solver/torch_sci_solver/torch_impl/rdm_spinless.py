"""Flat spin-orbital ("spinless") 1- and 2-RDM construction.

Batched-torch port of ``sci_solvers.spinless_ci_problem.spinless_rdm12``'s
Gram-sum algorithm — build the single-excitation tensor ``B[p,q,K] =
<D_K| a_p^ a_q |C>`` from the singles excitation table (scatter, no
Slater-Condon integral evaluation needed here: RDM construction is pure
amplitude/sign bookkeeping), then ``dm1[p,q] = B[p,q,:] @ conj(C)`` and
``dm2`` via the Gram matrix ``T[p,q,r,s] = sum_K conj(B[q,p,K]) B[r,s,K]``.

Unlike :mod:`torch_sci_solver.torch_impl.rdm_pairs` (its α/β-block
counterpart), there is no interleaved-Wick <-> spin-block sign bridge here:
a flat spin-orbital determinant list has one natural ordering, so the
matvec's own fermion parity (:func:`torch_sci_solver.torch_impl.
determinants.parity_batch`) is the only convention in play.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Iterator, Tuple

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import getocc_batch, parity_batch
from torch_sci_solver.torch_impl.pair_helpers import _found_in_multilimb
from torch_sci_solver.torch_impl.spinless_helpers import SpinlessBasis, SpinlessHelpers

DEFAULT_CHUNK_SIZE = 2_000_000


def _singles_sign_and_indices(
    basis: SpinlessBasis, t: dict[str, Tensor]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per-entry ``(sign, k_bra, k_ket, cr, an)`` for one streamed singles chunk.

    ``sign`` is the fermion parity for the excitation ``k_ket -> k_bra``
    (annihilate ``an``, create ``cr``), computed from the ket determinant —
    same bra/ket convention validated for the matvec in
    :mod:`torch_sci_solver.torch_impl.mult_spinless` (the ket goes into
    ``parity_batch`` as ``det``, mirroring
    ``sci_solvers.spinless_ci_problem._hij_spinless``'s
    ``one_excite(det_b, d[0], c[0], ...)``: ``d`` = ket-only orbital = ``cr``
    here, is the higher-priority operand and enters first).
    """
    k_bra = t["k_bra"]
    k_ket = t["k_ket"]
    cr = t["cr0"]
    an = t["an0"]
    ket = basis.dets[k_ket]
    start = torch.minimum(cr, an)
    end = torch.maximum(cr, an)
    # MPS has no float64; ±1 signs are exact in float32 as well.
    sign_dtype = torch.float32 if basis.device.type == "mps" else torch.float64
    sgn = torch.ones(k_bra.shape[0], dtype=sign_dtype, device=basis.device)
    sgn = parity_batch(ket, None, start, end, sgn)
    return sgn, k_bra, k_ket, cr, an


def build_b_tensor_spinless(
    basis: SpinlessBasis,
    helpers: SpinlessHelpers,
    amplitudes: Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    connected: "tuple[Tensor, Iterator[dict[str, Tensor]]] | None" = None,
) -> Tensor:
    """Full ``B[p, q, :] = <.| a_p^ a_q |C>`` tensor.

    Args:
        basis: the flat determinant list.
        helpers: excitation-table streamer.
        amplitudes: ``(dim,)`` CI amplitude vector ``C``.
        chunk_size: max entries scattered per pass (memory bound).
        connected: when given, the ``(connected, chunks)`` pair from
            :meth:`~torch_sci_solver.torch_impl.spinless_helpers.
            SpinlessHelpers.connected_singles` -- ``B``'s intermediate axis
            is then sized to ``len(connected)`` (``basis.dets`` union every
            single-excitation reachable destination) instead of
            ``basis.dim``, scattering ``chunks`` in place of
            ``helpers.singles()``. Required for a correct 2-RDM Gram term on
            a truncated (selected-CI) basis -- see :func:`spinless_rdm12_torch`.
            ``None`` (default) keeps the original ``basis.dets``-sized
            behaviour, exact whenever ``basis`` is already closed under
            single excitations (e.g. the full canonical determinant list).

    Returns:
        ``(nmo, nmo, dim)`` complex tensor (``dim`` = ``basis.dim`` or
        ``len(connected)``). Use :func:`build_b_block_spinless` instead for a
        memory-bounded block.
    """
    nmo = basis.n_spinorb
    dtype = amplitudes.dtype
    device = amplitudes.device
    if connected is None:
        dim = basis.dim
        diag_rows = torch.arange(dim, dtype=torch.int64, device=device)
        singles_stream = helpers.singles()
    else:
        conn_dets, singles_stream = connected
        dim = int(conn_dets.shape[0])
        _found, diag_rows = _found_in_multilimb(conn_dets, basis.dets)
    # B[p, q, K] = <D_K| a_p^ a_q |C>, indexed (annihilation, creation, det) --
    # empirically verified against sci_solvers.spinless_ci_problem.spinless_rdm12
    # (agreement to ~1e-16): the excitation-table's "cr"/"an" name which
    # spin-orbital is newly occupied/vacated between k_bra and k_ket, not which
    # tensor axis they land on, so this is fixed by cross-check rather than
    # derived from the naming alone.
    B = torch.zeros((nmo, nmo, dim), dtype=dtype, device=device)

    # Diagonal (p == q) term: a_p^ a_p |K> = occ(K, p) |K> (string unchanged,
    # sign +1) -- the excitation-table streamer only enumerates genuine
    # excitations (cr != an, always a different determinant), so this
    # occupation-number contribution has no table row and must be added
    # directly. Mirrors sci_solvers.complex_rdm._link_arrays_selected's
    # explicit "Diagonal (p == q) ... sign +1" block. Scattered onto
    # diag_rows (the selected determinants' index into the B axis) rather
    # than written as a dense slice, since that axis may be the connected
    # set, in which case the selected determinants are a strict subset of it.
    p_range = torch.arange(nmo, dtype=torch.int64, device=device)
    n_limbs = basis.dets.shape[-1]
    occ = getocc_batch(
        basis.dets.unsqueeze(-2).expand(basis.dim, nmo, n_limbs),
        None,
        p_range.expand(basis.dim, nmo),
    ).to(dtype)  # (basis.dim, nmo)
    diag_contrib = occ * amplitudes.unsqueeze(-1)  # (basis.dim, nmo)
    B[p_range.unsqueeze(0), p_range.unsqueeze(0), diag_rows.unsqueeze(-1)] = diag_contrib

    chunk = max(1, int(chunk_size))
    for t in singles_stream:
        sgn, k_bra, k_ket, cr, an = _singles_sign_and_indices(basis, t)
        k_total = int(k_bra.shape[0])
        for start in range(0, k_total, chunk):
            end = min(start + chunk, k_total)
            contrib = sgn[start:end].to(dtype) * amplitudes[k_ket[start:end]]
            B.index_put_(
                (an[start:end], cr[start:end], k_bra[start:end]),
                contrib,
                accumulate=True,
            )
    return B


def spinless_rdm1_diag_torch(basis: SpinlessBasis, amplitudes: Tensor) -> Tensor:
    """Diagonal of the flat spin-orbital 1-RDM: occupancy per spin-orbital.

    ``dm1[p,p] = sum_K |C_K|^2 * occ(K, p)`` — the same occupation-number
    term already inlined in :func:`build_b_tensor_spinless` (its ``occ`` /
    ``diag_contrib`` computation for ``p == q``), but computed directly here
    without the singles excitation-table scatter or the ``dm2`` Gram matrix
    (``(nmo*nmo, nmo*nmo)``, O(nmo**4) memory) that :func:`spinless_rdm12_torch`
    builds and that a caller wanting only occupancies would otherwise throw
    away. Use :func:`spinless_rdm12_torch` when the full 1-/2-RDM (not just
    the diagonal) is actually needed.

    Args:
        basis: the flat determinant list.
        amplitudes: ``(dim,)`` CI amplitude vector ``C``.

    Returns:
        ``(nmo,)`` real tensor of orbital occupancies.
    """
    device = amplitudes.device
    dtype = amplitudes.dtype
    nmo = basis.n_spinorb

    # |C|^2, not C^2 — occupation numbers are real even for complex C.
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    C2 = amplitudes.abs().pow(2)

    p_range = torch.arange(nmo, dtype=torch.int64, device=device)
    n_limbs = basis.dets.shape[-1]
    occ = getocc_batch(
        basis.dets.unsqueeze(-2).expand(basis.dim, nmo, n_limbs),
        None,
        p_range.expand(basis.dim, nmo),
    ).to(real_dtype)  # (dim, nmo)
    return occ.T @ C2


def spinless_rdm12_torch(
    basis: SpinlessBasis,
    helpers: SpinlessHelpers,
    amplitudes: Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tuple[Tensor, Tensor]:
    """Complex-correct spin-orbital 1- and 2-RDM for a flat spinless CI vector.

    Batched-torch analogue of
    ``sci_solvers.spinless_ci_problem.spinless_rdm12`` (selected-CI path):
    builds the full ``B`` tensor via :func:`build_b_tensor_spinless`, then
    ``dm1[p,q] = sum_K conj(C_K) B[p,q,K]`` and the 2-RDM Gram term
    ``T[p,q,r,s] = sum_K conj(B[q,p,K]) B[r,s,K]``.

    ``B``'s intermediate axis ``K`` is sized to the *connected set* --
    ``basis.dets`` union every determinant one single excitation away --
    rather than ``basis.dets`` itself. The Gram term's
    intermediate ``K`` ranges over every determinant the single-excitation
    operators ``a_p^ a_q`` can reach from the selected set, not just the
    selected set, so restricting it to ``basis.dets`` silently drops terms
    on any basis not already closed under single excitations (the normal
    truncated/SQD case) -- see
    ``sci_solvers.complex_rdm._restrict_to_connected_set``. ``Cc`` is
    zero-embedded onto the same connected axis for ``dm1``, which agrees
    with the ``basis.dets``-only contraction either way (the extra rows
    contribute zero), so there is no need for a second, smaller ``B``.

    Args:
        basis: the flat determinant list.
        helpers: excitation-table streamer.
        amplitudes: ``(dim,)`` CI amplitude vector ``C``.
        chunk_size: forwarded to :func:`build_b_tensor_spinless`.

    Returns:
        ``(dm1, dm2)``: ``dm1`` is ``(nmo, nmo)`` Hermitian; ``dm2`` is
        ``(nmo, nmo, nmo, nmo)`` in the
        ``pyscf.fci.selected_ci.make_rdm2`` chemist convention (matches
        ``spinless_ci_problem.spinless_rdm12`` exactly).
    """
    nmo = basis.n_spinorb
    connected, chunks = helpers.connected_singles()
    B = build_b_tensor_spinless(
        basis, helpers, amplitudes, chunk_size=chunk_size, connected=(connected, chunks)
    )
    _found, sel_rows = _found_in_multilimb(connected, basis.dets)
    Cc_conn = torch.zeros(int(connected.shape[0]), dtype=amplitudes.dtype, device=amplitudes.device)
    Cc_conn[sel_rows] = amplitudes.conj()
    dm1 = torch.einsum("pqk,k->pq", B, Cc_conn)

    Bflat = B.reshape(nmo * nmo, -1)
    G = Bflat.conj() @ Bflat.T  # (nmo*nmo, nmo*nmo)
    T = G.reshape(nmo, nmo, nmo, nmo).permute(1, 0, 2, 3).contiguous()
    dm2 = T
    for qq in range(nmo):
        dm2[:, qq, qq, :] -= dm1
    return dm1, dm2


def spinless_energy_torch(
    basis: SpinlessBasis,
    helpers: SpinlessHelpers,
    amplitudes: Tensor,
    hcore: Tensor,
    eri: Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Electronic energy of a flat spinless CI vector, rebuilt from its RDMs.

    Cross-validation helper mirroring
    ``sci_solvers.spinless_ci_problem.spinless_energy``:
    ``E = <dm1, hcore> + 0.5 * einsum("prqs,prqs->", dm2, eri)``.
    """
    dm1, dm2 = spinless_rdm12_torch(basis, helpers, amplitudes, chunk_size=chunk_size)
    e = torch.einsum("pr,pr->", dm1, hcore) + 0.5 * torch.einsum("prqs,prqs->", dm2, eri)
    return float(e.real.item())
