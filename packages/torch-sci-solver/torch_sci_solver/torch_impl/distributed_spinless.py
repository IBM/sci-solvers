"""Owner-computes spinless matvec: replicated input, owned output.

Distributed-memory companion of
:func:`torch_sci_solver.torch_impl.mult_spinless.mult_spinless_torch` for the
flat spin-orbital ("spinless") basis. That basis is an explicit list of ``N``
determinants with no α/β product structure, so — unlike the TPB operator —
there is no row/column split to exploit: the excitation tables are streamed and
their entries couple arbitrary determinant pairs. Rank ``r`` therefore owns the
contiguous output block ``[r0, r1)``, streams only the excitation tables of the
determinants it owns, and reads the ket amplitudes from a full-length input
vector that every rank gathers (see the class notes).

Notes
-----
For a flat determinant list the matvec is

.. math::

    (H x)_k = h_{kk} x_k
      + \\sum_{(k, k') \\in \\mathcal{T}_1}
            \\langle D_k | H_1 | D_{k'} \\rangle x_{k'}
      + \\sum_{(k, k') \\in \\mathcal{T}_2}
            \\langle D_k | H_2 | D_{k'} \\rangle x_{k'},

where :math:`\\mathcal{T}_1` / :math:`\\mathcal{T}_2` are the streamed singles
and doubles tables (source/bra ``k``, excited/ket ``k'``, with the annihilated
and created spin-orbitals carried alongside). :func:`_stream_spinless`
enumerates **from** a source set and filters against the selected set, so
passing ``src_dets = basis.dets[r0:r1]`` with ``target_dets = basis.dets``
yields exactly the entries whose bra row this rank owns: ``k_bra`` indexes the
local output block and ``k_ket`` indexes the full input vector. The
Slater-Condon scatters (:func:`_scatter_single_spinless` /
:func:`_scatter_double_spinless`) then write into the local output while
reading the replicated input.

The input is replicated on purpose. The tables are streamed precisely because
their ``nnz`` does not fit in memory, so a per-rank ``nnz / P`` table cannot be
stored, and an index exchange would in the worst case ship nearly the whole
vector anyway. One length-``N`` input vector per rank is the deliberate cost
(the same vector the replicated fallback already holds); the output lives only
on its owner (``N / P``), and the caller's Davidson ``C`` / ``HC`` blocks are
distributed, which is the dominant memory item.

``__call__`` is collective: every rank performs the same all-gather in the same
order, so all ranks must enter it together.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Any, Optional

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import zero_excite_batch_spinless
from torch_sci_solver.torch_impl.distributed_davidson import (
    FlatGather,
    RankBounds,
    _resolve_rank_world,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_entries_triton import apply_double_entries
from torch_sci_solver.torch_impl.mult_pairs import (
    DEFAULT_CHUNK_SIZE,
    _entry_chunk,
    prefetch_while_pending,
)
from torch_sci_solver.torch_impl.mult_spinless import (
    _BYTES_PER_DOUBLE_ENTRY,
    _BYTES_PER_SINGLE_ENTRY,
    _scatter_double_spinless,
    _scatter_single_spinless,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
    SpinlessBasis,
    SpinlessHelpers,
    _stream_spinless,
)


def make_diag_terms_spinless_local(
    dets_local: Tensor,
    n_spinorb: int,
    n_particles: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Diagonal of ``H`` for one contiguous spinless determinant slice.

    Owner-computes companion of
    :func:`~torch_sci_solver.torch_impl.mult_spinless.make_diag_terms_spinless_torch`:
    the same formula and dtype (a real tensor even for complex integrals),
    evaluated in batches of at most ``chunk_size`` determinants so the
    ``(batch, n_spinorb)`` occupancy workspace stays bounded. A diagonal
    element is independent of the other rows in the batch, so chunking does
    not change the result.

    Args:
        dets_local: ``(N_local, n_limbs)`` int64 determinants owned by this rank.
        n_spinorb: nmo, the number of spin-orbitals.
        n_particles: total particle number; accepted for interface parity with
            :class:`~torch_sci_solver.torch_impl.spinless_helpers.SpinlessBasis`
            and checked against ``n_spinorb`` (the diagonal primitive reads the
            occupations from the bitmasks).
        integrals: spinless :class:`TorchIntegrals` bundle.
        chunk_size: max determinants per zero-excitation batch.

    Returns:
        ``(N_local,)`` real diagonal on the integrals device.

    Raises:
        ValueError: ``n_particles`` is outside ``[0, n_spinorb]``.
    """
    if not 0 <= int(n_particles) <= int(n_spinorb):
        raise ValueError(f"n_particles must be in [0, {n_spinorb}], got {n_particles}")

    device = integrals.device
    dtype = integrals.dtype
    hii_dtype = dtype.to_real() if dtype.is_complex else dtype
    dets = dets_local.to(device)
    n_dets = int(dets.shape[0])

    hii = torch.empty(n_dets, dtype=hii_dtype, device=device)
    chunk = max(1, int(chunk_size))
    for start in range(0, n_dets, chunk):
        end = min(start + chunk, n_dets)
        hii_chunk = zero_excite_batch_spinless(
            dets[start:end],
            None,
            integrals.n_spinorb,
            integrals.I0,
            integrals.I1_diag,
            integrals.I2_direct,
            integrals.I2_exchange,
        )
        hii[start:end] = hii_chunk.real if dtype.is_complex else hii_chunk
    return hii


class LocalSpinlessOperator:
    """Owner-computes spinless matvec: replicated input, owned output.

    Rank ``r`` owns the contiguous determinant block ``[r0, r1)`` and returns
    exactly ``(H x)[r0:r1]``. It gathers the local input block into a full
    length-``N`` vector, scales the local block by ``hii_local``, then streams
    the singles and doubles tables of the determinants it owns and scatters
    their contributions into the local output.

    Args:
        hii_local: ``(N_local,)`` diagonal the operator applies (real, from
            :func:`make_diag_terms_spinless_local`).
        basis: the global flat determinant list; its ``dets`` are indexed by
            the full-space ket indices.
        helpers: streaming excitation-table accessor; only its byte budget is
            read (the enumeration is restricted to the owned rows directly).
        integrals: the spinless :class:`TorchIntegrals` bundle.
        bounds: this rank's contiguous flat block. The blocks of all ranks
            must tile ``[0, basis.dim)`` in rank order; the input gather reads
            every rank's block size once and checks this.
        rank, world_size, group: process-group identity and communicator.
        chunk_size: per-Slater-Condon-batch entry cap (same meaning as in
            :func:`mult_spinless_torch`).

    Raises:
        ValueError: ``bounds`` is inconsistent with the basis and process
            group, or ``hii_local`` has the wrong shape.

    Notes
    -----
    The input is replicated because the tables are streamed: a per-rank
    ``nnz / P`` table cannot be stored, and an index exchange would in the
    worst case ship nearly the whole vector. Every rank therefore holds one
    length-``N`` input vector during the matvec, while the output and the
    caller's Davidson ``C`` / ``HC`` blocks are distributed.

    ``__call__`` is collective (one all-gather of the local input block), so
    every rank must enter it in the same order.

    The enumeration cost is ``1 / P`` of the full matvec: ``_stream_spinless``
    with ``src_dets = basis.dets[r0:r1]`` generates the chunks of exactly the
    owned source determinants, with ``k_bra`` local to ``[r0, r1)`` and
    ``k_ket`` global.
    """

    def __init__(
        self,
        hii_local: Tensor,
        basis: SpinlessBasis,
        helpers: SpinlessHelpers,
        integrals: TorchIntegrals,
        bounds: RankBounds,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        triton_matvec: bool = False,
    ) -> None:
        """Validate the local block and cache the streaming chunk bounds."""
        self._basis = basis
        self._integrals = integrals
        self._triton_matvec = bool(triton_matvec)
        self.bounds = bounds
        self.n = basis.dim
        self._rank, self._world_size = _resolve_rank_world(rank, world_size, group)
        if (
            bounds.n != self.n
            or bounds.rank != self._rank
            or bounds.world_size != self._world_size
            or not 0 <= bounds.start <= bounds.stop <= self.n
        ):
            raise ValueError(
                f"bounds {bounds} are inconsistent with n={self.n}, rank={self._rank}, "
                f"world_size={self._world_size}"
            )
        self.N_local = bounds.local_size
        if hii_local.shape != (self.N_local,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({self.N_local},)")
        self._hii_local = hii_local

        self._budget_bytes = helpers.budget_bytes
        self._single_chunk = _entry_chunk(
            int(chunk_size), self._budget_bytes, _BYTES_PER_SINGLE_ENTRY
        )
        self._double_chunk = _entry_chunk(
            int(chunk_size), self._budget_bytes, _BYTES_PER_DOUBLE_ENTRY
        )
        self._gather = FlatGather(bounds, group=group)

    def __call__(self, x_local: Tensor) -> Tensor:
        """Return this rank's owned output block ``(H @ x)[r0:r1]``.

        Args:
            x_local: ``(N_local,)`` owned input block.

        Returns:
            ``(N_local,)`` owned output block.

        Raises:
            ValueError: ``x_local`` has the wrong shape.
        """
        if x_local.shape != (self.N_local,):
            raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({self.N_local},)")

        # The gather runs while the diagonal and the first singles chunk (the
        # table streams never read the vector) are computed.
        pending = self._gather.start(x_local)
        y = self._hii_local * x_local

        device = y.device
        dets = self._basis.dets.to(device)
        stream = dict(
            dets=dets,
            n_spinorb=self._basis.n_spinorb,
            n_particles=self._basis.n_particles,
            device=device,
            budget_bytes=self._budget_bytes,
            src_dets=dets[self.bounds.start : self.bounds.stop],
            target_dets=dets,
        )
        singles = prefetch_while_pending(_stream_spinless(order=1, **stream), pending)
        T_full = pending.wait()
        for t in singles:
            _scatter_single_spinless(
                Wb=y,
                T=T_full,
                dets=dets,
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr=t["cr0"],
                an=t["an0"],
                integrals=self._integrals,
                chunk_size=self._single_chunk,
            )
        for t in _stream_spinless(order=2, **stream):
            if self._triton_matvec and apply_double_entries(
                y,
                T_full,
                self._integrals,
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
                Wb=y,
                T=T_full,
                dets=dets,
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr0=t["cr0"],
                cr1=t["cr1"],
                an0=t["an0"],
                an1=t["an1"],
                integrals=self._integrals,
                chunk_size=self._double_chunk,
            )
        return y
