"""Owner-computes spinless tile matvec: staged ket values, owned output.

Distributed-memory companion of
:func:`torch_sci_solver.torch_impl.mult_spinless.mult_spinless_torch` over a
:class:`~torch_sci_solver.torch_impl.spinless_tiles.SpinlessTileBounds`
rectangle. Rank ``r`` owns the scattered determinant set
``bounds.determinant_index`` of its ``row_uid × col_uid`` rectangle and returns
exactly ``(H x)[bounds.determinant_index]``. Unlike
:class:`~torch_sci_solver.torch_impl.distributed_spinless.LocalSpinlessOperator`
it never materialises a length-``N`` input vector: it requests only the remote
ket amplitudes its owned rows read, one ket-owner peer at a time, and
accumulates each stage straight into its owned output.

Notes
-----
For a flat spin-orbital basis the matvec is

.. math::

    (H x)_k = h_{kk} x_k
      + \\sum_{(k \\leftarrow j) \\in \\mathcal{T}_1}
            \\langle D_k | H_1 | D_j \\rangle x_j
      + \\sum_{(k \\leftarrow j) \\in \\mathcal{T}_2}
            \\langle D_k | H_2 | D_j \\rangle x_j,

with :math:`k` ranging over the owned rows. The operator runs the streaming
enumeration the other way round: :func:`_stream_spinless` gets the ket
determinants as its enumeration sources and the owned determinants as its
targets, so each yielded entry pairs a ket :math:`j` (``k_bra``, the position
in the source list) with the owned row :math:`k` (``k_ket``, the position in
the local row list). The enumeration's ``an`` orbitals are then the ket-only
orbitals and its ``cr`` orbitals the bra-only ones, so evaluating the same
Slater-Condon primitive on the source determinant with those roles (the
opposite argument order to the forward table) yields
:math:`\\langle D_k | H | D_j \\rangle` exactly, with no conjugation
assumption on the integrals. Scattering ``y[k] += matelem * x[j]`` over the
owned rows therefore reproduces the replicated matvec.

Stage 0 is the local ket block, read from ``x_local`` directly. Stage
``s`` in ``[1, world_size)`` carries the sorted global ket indices owned by
peer ``(rank + s) % world_size`` that this rank reads; the rank enumerates
that stage's sources as soon as the stage buffer arrives. Every rank also
sends whatever the peer it writes to at stage ``s`` needs from its own block.

Trade-offs
----------
The exchanged *volume* is not reduced (the measured needed-ket set — the
union of every ket a rank reads — is about ``1.00 N`` at two ranks and
``0.93-0.98 N`` on multi-stage grids of the same basis), so the win is
per-rank *peak memory*: the receive buffer is one stage, not the union.
Measured on a full Cartesian-product basis, the per-stage buffer is ``0.50 N``
at two ranks, ``0.27 N`` on a ``2 × 2`` grid, ``0.13 N`` on a ``2 × 4`` grid
and ``0.07 N`` on a ``4 × 4`` grid, where a single-buffer "compressed gather"
of the same needed set would hold ``1.00 / 1.00 / 0.97 / 0.91 N``. Because the
receive list is disjoint from the owned block (``|R| <= N - local_size``) and
the send list is a subset of it (``<= local_size``), the concurrent staging
never exceeds one full vector on any basis or grid, and it reaches that bound
only when a rank reads every remote ket and ships its whole block. A single
receive buffer is reused for every stage and is never double-buffered,
precisely so the peak stays at one stage; the cost is that communication and
compute do not overlap beyond prefetching stage 1 under the local stage.

The price is paid in two other places:

* the operator keeps each stage's sorted global source indices for its whole
  lifetime (``needed_size`` of them per rank, ``int32`` while every determinant
  position fits and ``int64`` otherwise). Those lists are setup metadata, not
  staging, but they count in the steady-state footprint of any design that
  reads a compressed needed set; and
* each matvec enumerates from ``local_size + needed_size`` source
  determinants instead of the 1-D operator's ``local_size``, because the
  per-stage split needs the ket-side enumeration. On full Cartesian-product
  bases the measured ratio is 2.0x at two ranks and 8.7x on a ``2 × 4`` grid.

The stage order is deterministic and the exchange is explicit point-to-point
over the world group (no subcommunicators); matvec stage ``s`` uses tag
``tag + s``, while the setup request exchange tags each directed message with
its sender rank (``tag + world_size + sender``) because the two directions of
one pair land on different stage numbers there. ``__call__`` is collective:
every rank must enter it in the same order.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Any, Optional

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import one_excite_batch, two_excite_batch
from torch_sci_solver.torch_impl.distributed_davidson import (
    TilePartition,
    _dist,
    _resolve_rank_world,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_entries_triton import apply_double_entries
from torch_sci_solver.torch_impl.mult_pairs import DEFAULT_CHUNK_SIZE, _entry_chunk
from torch_sci_solver.torch_impl.mult_spinless import (
    _BYTES_PER_DOUBLE_ENTRY,
    _BYTES_PER_SINGLE_ENTRY,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
    SpinlessBasis,
    SpinlessHelpers,
    _chunk_dets,
    _stream_spinless,
)
from torch_sci_solver.torch_impl.spinless_tiles import (
    SpinlessTileBounds,
    half_string_axes,
    half_string_uids_of,
)


def _source_chunk(n_spinorb: int, n_particles: int, order: int, budget_bytes: int) -> int:
    """Source determinants per chunk that keep one enumeration chunk in budget.

    The stage source determinants are gathered from the global basis, so the
    gather is bounded by the same determinant count the enumerator itself
    chunks at.

    Args:
        n_spinorb: Number of spin-orbitals.
        n_particles: Total particle number.
        order: ``1`` for singles, ``2`` for doubles.
        budget_bytes: Per-chunk workspace budget.

    Returns:
        A positive determinant count; ``1`` when the order has no candidates.
    """
    n_empty = int(n_spinorb) - int(n_particles)
    if order == 1:
        per_det = int(n_particles) * n_empty if n_particles >= 1 and n_empty >= 1 else 0
        kind = "singles"
    else:
        per_det = (
            (int(n_particles) * (int(n_particles) - 1) // 2) * (n_empty * (n_empty - 1) // 2)
            if n_particles >= 2 and n_empty >= 2
            else 0
        )
        kind = "doubles"
    if per_det <= 0:
        return 1
    return max(1, _chunk_dets(per_det, kind, int(budget_bytes)))


class LocalSpinlessTileOperator:
    """Owner-computes spinless matvec over one ``row_uid × col_uid`` tile.

    Rank ``r`` owns the scattered determinant set ``bounds.determinant_index``
    and returns exactly ``(H x)[bounds.determinant_index]``. The input is read
    from three places only: the owned input block (stage 0), one staged
    receive buffer per remote ket-owner stage, and the peers' staged sends.

    Args:
        hii_local: ``(N_local,)`` diagonal of the owned determinants, in
            ``determinant_index`` order (from
            :func:`~torch_sci_solver.torch_impl.distributed_spinless.
            make_diag_terms_spinless_local`).
        basis: the global flat determinant list.
        helpers: streaming excitation-table accessor; only its byte budget is
            read, since the enumeration is restricted to one stage's source
            determinants at a time.
        integrals: the spinless :class:`TorchIntegrals` bundle.
        bounds: this rank's tile: its uid slabs and sorted global determinant
            positions.
        partition: the :class:`TilePartition` the bounds were built from; it
            supplies every rank's uid slabs for the ket-owner routing.
        rank, world_size, group: process-group identity and communicator.
        chunk_size: per-Slater-Condon-batch entry cap (same meaning as in
            :func:`mult_spinless_torch`).
        triton_matvec: route the doubles tables to the entry-parallel Triton
            kernel (:func:`~torch_sci_solver.torch_impl.mult_entries_triton.
            apply_double_entries`); the singles stay batched. An unsupported
            device, dtype or layout falls back to the batched scatter, so the
            result never depends on the flag.
        tag: base P2P tag; matvec stage ``s`` uses ``tag + s`` and the setup
            request exchange uses ``tag + world_size + s``.

    Raises:
        ValueError: ``bounds`` / ``partition`` are inconsistent with the basis,
            the process group, or each other, or ``hii_local`` has the wrong
            shape.
        RuntimeError: a multi-rank operator has no initialized process group,
            or a peer requests a determinant this rank does not own.

    Attributes:
        n: Global determinant count ``N``.
        N_local: Number of owned determinants.
        bounds: The :class:`SpinlessTileBounds` this operator serves.
        partition: The :class:`TilePartition` over the uid axes.
        needed_size: Number of remote ket amplitudes this rank stages per
            matvec (the sum of the per-stage request counts).
        max_stage_size: Largest per-stage receive count, in values. The single
            receive buffer is this long, so the staging peak is bounded by it.
        max_send_size: Largest send payload, in values; ``0`` when every send
            aliases ``x_local``.
        stage_sizes: Per-stage receive counts, ``stage_sizes[s - 1]`` for stage
            ``s``.
        peak_staged_bytes: Concurrent exchange staging bytes, i.e. the receive
            buffer plus the largest send payload, for the current value dtype;
            a send of the whole local block aliases ``x_local`` and costs
            nothing.
    """

    def __init__(
        self,
        hii_local: Tensor,
        basis: SpinlessBasis,
        helpers: SpinlessHelpers,
        integrals: TorchIntegrals,
        bounds: SpinlessTileBounds,
        partition: TilePartition,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        triton_matvec: bool = False,
        tag: int = 0,
    ) -> None:
        """Validate the tile, discover the needed kets, and plan the exchange."""
        self._basis = basis
        self._integrals = integrals
        self._group = group
        self._tag = int(tag)
        self._triton_matvec = bool(triton_matvec)
        self.bounds = bounds
        self.partition = partition
        self.n = basis.dim
        self._rank, self._world_size = _resolve_rank_world(rank, world_size, group)
        if bounds.n_dets != self.n:
            raise ValueError(f"bounds.n_dets={bounds.n_dets} != basis dimension {self.n}")
        if bounds.rank != self._rank or bounds.world_size != self._world_size:
            raise ValueError(
                f"bounds rank/world_size ({bounds.rank}, {bounds.world_size}) != "
                f"({self._rank}, {self._world_size})"
            )
        if partition.world_size != self._world_size:
            raise ValueError(f"partition world_size={partition.world_size} != {self._world_size}")
        if partition.n_rows != bounds.n_row_uids or partition.n_cols != bounds.n_col_uids:
            raise ValueError(
                f"partition ({partition.n_rows}, {partition.n_cols}) does not match the "
                f"tile uid axes ({bounds.n_row_uids}, {bounds.n_col_uids})"
            )
        slab = partition.for_rank(self._rank)
        if (
            slab.row_slab,
            slab.col_slab,
            slab.row_start,
            slab.row_stop,
            slab.col_start,
            slab.col_stop,
        ) != (
            bounds.row_slab,
            bounds.col_slab,
            bounds.row_start,
            bounds.row_stop,
            bounds.col_start,
            bounds.col_stop,
        ):
            raise ValueError("bounds do not describe this rank's tile of the partition")

        self.N_local = bounds.local_size
        if hii_local.shape != (self.N_local,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({self.N_local},)")
        self._hii_local = hii_local

        self._device = integrals.device
        self._dets = basis.dets.to(self._device)
        self._unique_rows, self._unique_cols = half_string_axes(self._dets, bounds.split)
        if (
            int(self._unique_rows.shape[0]) != bounds.n_row_uids
            or int(self._unique_cols.shape[0]) != bounds.n_col_uids
        ):
            raise ValueError(
                "the determinant list's uid axes do not match the tile bounds; "
                "bounds must come from spinless_tile_partition on this basis"
            )
        self._local_index = bounds.determinant_index.to(self._device)
        self._local_dets = self._dets[self._local_index]
        # The per-stage source lists live for the operator's whole lifetime, so
        # they are the steadiest memory item after the basis itself; int32
        # halves them and is exact while every determinant position fits.
        self._index_dtype = torch.int32 if self.n < 2**31 else torch.int64

        self._budget_bytes = helpers.budget_bytes
        self._single_chunk = _entry_chunk(
            int(chunk_size), self._budget_bytes, _BYTES_PER_SINGLE_ENTRY
        )
        self._double_chunk = _entry_chunk(
            int(chunk_size), self._budget_bytes, _BYTES_PER_DOUBLE_ENTRY
        )
        n_spinorb = basis.n_spinorb
        n_particles = basis.n_particles
        self._single_source_chunk = _source_chunk(n_spinorb, n_particles, 1, self._budget_bytes)
        self._double_source_chunk = _source_chunk(n_spinorb, n_particles, 2, self._budget_bytes)
        self._index_chunk = max(1, int(chunk_size))

        needed = self._discover_needed()
        self._plan_stages(needed)
        self._plan_send_positions()
        self._max_recv = max((int(index.numel()) for index in self._recv_index.values()), default=0)
        self._max_send = max(
            (
                0 if positions is None else int(positions.numel())
                for positions in self._send_position.values()
            ),
            default=0,
        )
        self.max_stage_size = self._max_recv
        self.max_send_size = self._max_send
        self.stage_sizes = tuple(
            int(self._recv_index[stage].numel()) for stage in range(1, self._world_size)
        )
        self._buffers: Optional[Tensor] = None
        self._buffer_key: Optional[tuple[torch.dtype, torch.device]] = None
        self._peak_staged_bytes = (self._max_recv + self._max_send) * torch.empty(
            (), dtype=self._integrals.dtype
        ).element_size()

    def _discover_needed(self) -> Tensor:
        """Sorted global ket indices the owned rows read, local and remote.

        Runs one forward pass of the streaming table from the owned rows
        against the full basis. The ``N``-entry boolean accumulator is the only
        length-``N`` object this operator ever allocates; it is a dedup mask,
        not an amplitude vector, and is released when this method returns.

        Returns:
            ``(n_needed,)`` int64 global determinant positions.
        """
        needed = torch.zeros(self.n, dtype=torch.bool, device=self._device)
        stream = dict(
            dets=self._dets,
            n_spinorb=self._basis.n_spinorb,
            n_particles=self._basis.n_particles,
            device=self._device,
            budget_bytes=self._budget_bytes,
            src_dets=self._local_dets,
            target_dets=self._dets,
        )
        for order in (1, 2):
            for chunk in _stream_spinless(order=order, **stream):
                if int(chunk["k_ket"].numel()):
                    needed[chunk["k_ket"]] = True
        return torch.nonzero(needed, as_tuple=False).reshape(-1)

    def _owner_rank(self, index: Tensor) -> Tensor:
        """Owner rank of each global determinant, from its half-string uids.

        Args:
            index: ``(n,)`` global determinant positions.

        Returns:
            ``(n,)`` int64 rank owning each determinant.
        """
        row_uid, col_uid = half_string_uids_of(
            self._dets[index], self.bounds.split, self._unique_rows, self._unique_cols
        )
        grid = self.partition.grid
        row_starts = torch.tensor(
            [start for start, _ in grid.alpha_bounds], dtype=torch.int64, device=self._device
        )
        col_starts = torch.tensor(
            [start for start, _ in grid.beta_bounds], dtype=torch.int64, device=self._device
        )
        # right=True keeps the last slab among duplicate starts, so empty slabs
        # never claim an index (the same convention as TilePartition.owner_*).
        row_slab = (torch.searchsorted(row_starts, row_uid, right=True) - 1).clamp_(
            0, grid.p_alpha - 1
        )
        col_slab = (torch.searchsorted(col_starts, col_uid, right=True) - 1).clamp_(
            0, grid.p_beta - 1
        )
        return row_slab * grid.p_beta + col_slab

    def _plan_stages(self, needed: Tensor) -> None:
        """Split the needed kets by owner and record each stage's source list.

        Args:
            needed: ``(n_needed,)`` sorted global ket indices.
        """
        parts: list[list[Tensor]] = [[] for _ in range(self._world_size)]
        for start in range(0, int(needed.numel()), self._index_chunk):
            index = needed[start : start + self._index_chunk]
            owner = self._owner_rank(index)
            for peer in range(self._world_size):
                if peer == self._rank:
                    continue
                selected = index[owner == peer]
                if selected.numel():
                    parts[peer].append(selected)
        self._recv_index: dict[int, Tensor] = {}
        self.needed_size = 0
        for stage in range(1, self._world_size):
            peer = (self._rank + stage) % self._world_size
            if parts[peer]:
                merged = torch.sort(torch.cat(parts[peer])).values.to(self._index_dtype)
            else:
                merged = torch.empty(0, dtype=self._index_dtype, device=self._device)
            self._recv_index[stage] = merged
            self.needed_size += int(merged.numel())

    def _plan_send_positions(self) -> None:
        """Exchange the per-stage request lists and map them to send positions.

        Each rank broadcasts only the *counts* of its per-stage request lists
        (small), then posts one point-to-point message per peer carrying the
        list itself. The sender maps a request to positions in its sorted
        owned index list, so the values it ships are in the same order the
        requester's stage list records. A request covering the whole local
        block is recorded as ``None`` (send ``x_local`` itself, no copy).

        Raises:
            RuntimeError: a request names a determinant this rank does not own.
        """
        self._send_position: dict[int, Optional[Tensor]] = {
            stage: torch.empty(0, dtype=self._index_dtype, device=self._device)
            for stage in range(1, self._world_size)
        }
        if self._world_size == 1:
            return
        dist = _dist()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "a multi-rank LocalSpinlessTileOperator needs an initialized process group"
            )
        my_counts = [
            int(self._recv_index.get(stage, torch.empty(0)).numel())
            for stage in range(self._world_size)
        ]
        all_counts: list[Optional[list[int]]] = [None] * self._world_size
        dist.all_gather_object(all_counts, my_counts, group=self._group)

        setup_tag = self._tag + self._world_size
        ops: list[Any] = []
        keep: list[Tensor] = []
        received: dict[int, Tensor] = {}
        for stage in range(1, self._world_size):
            request = self._recv_index[stage]
            if request.numel():
                payload = request.contiguous()
                keep.append(payload)
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        payload,
                        (self._rank + stage) % self._world_size,
                        group=self._group,
                        tag=setup_tag + self._rank,
                    )
                )
            peer = (self._rank - stage) % self._world_size
            peer_counts = all_counts[peer]
            if peer_counts is None:
                raise RuntimeError("failed to gather every rank's stage counts")
            n_recv = int(peer_counts[stage])
            if n_recv:
                buffer = torch.empty(n_recv, dtype=self._index_dtype, device=self._device)
                received[stage] = buffer
                ops.append(
                    dist.P2POp(
                        dist.irecv,
                        buffer,
                        peer,
                        group=self._group,
                        tag=setup_tag + peer,
                    )
                )
        for request in dist.batch_isend_irecv(ops) if ops else []:
            request.wait()
        for stage, requested in received.items():
            if requested.numel() == self._local_index.numel() and torch.equal(
                requested, self._local_index
            ):
                self._send_position[stage] = None
                continue
            position = torch.searchsorted(self._local_index, requested).to(self._index_dtype)
            clipped = position.clamp(max=max(self._local_index.numel() - 1, 0))
            found = (position < self._local_index.numel()) & (
                self._local_index[clipped] == requested
            )
            if not bool(found.all()):
                raise RuntimeError("a peer requested spinless determinants this tile does not own")
            self._send_position[stage] = position

    def _ensure_buffers(self, reference: Tensor) -> None:
        """Allocate the single receive buffer for ``reference``'s dtype/device.

        Complex values travel as their interleaved ``(real, imag)`` real view
        (NCCL has no complex scalar type), so the buffer width counts both
        parts.
        """
        key = (reference.dtype, reference.device)
        if self._buffers is None or self._buffer_key != key:
            width = 2 if reference.dtype.is_complex else 1
            dtype = reference.dtype.to_real() if reference.dtype.is_complex else reference.dtype
            self._buffers = torch.empty(
                self._max_recv * width, dtype=dtype, device=reference.device
            )
            self._buffer_key = key
            self._peak_staged_bytes = (self._max_recv + self._max_send) * reference.element_size()

    def __call__(self, x_local: Tensor) -> Tensor:
        """Return this rank's owned output block ``(H @ x)[determinant_index]``.

        Args:
            x_local: ``(N_local,)`` owned input block, in
                ``bounds.determinant_index`` order.

        Returns:
            ``(N_local,)`` owned output block.

        Raises:
            ValueError: ``x_local`` has the wrong shape.
        """
        if x_local.shape != (self.N_local,):
            raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({self.N_local},)")
        y = self._hii_local * x_local
        if self.N_local == 0:
            # An empty tile reads nothing and owns nothing a peer can request.
            return y

        if self._world_size == 1:
            self._apply_stage(0, x_local, y)
            return y

        self._ensure_buffers(x_local)
        # Post stage 1 before the local stage runs: its buffer is not read
        # until the wait below, so the transfer overlaps the local enumeration.
        pending = self._post_stage(1, x_local)
        self._apply_stage(0, x_local, y)
        for stage in range(1, self._world_size):
            if stage > 1:
                pending = self._post_stage(stage, x_local)
            requests, recv, n_recv = pending
            for request in requests:
                request.wait()
            if n_recv:
                assert recv is not None
                if x_local.dtype.is_complex:
                    values = torch.view_as_complex(recv.view(n_recv, 2))
                else:
                    values = recv
                self._apply_stage(stage, values, y)
        return y

    def _post_stage(self, stage: int, x_local: Tensor) -> tuple[list[Any], Optional[Tensor], int]:
        """Issue stage ``stage``'s send and receive; keep the send alive."""
        dist = _dist()
        ops: list[Any] = []
        send_position = self._send_position.get(stage)
        if send_position is not None and send_position.numel():
            payload = x_local[send_position].contiguous()
        elif send_position is None and self.N_local:
            # The peer reads this whole block, so ship it without a copy.
            payload = x_local
        else:
            payload = None
        if payload is not None:
            packed = (
                torch.view_as_real(payload).reshape(-1)
                if payload.is_complex()
                else payload.reshape(-1)
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    packed,
                    (self._rank - stage) % self._world_size,
                    group=self._group,
                    tag=self._tag + stage,
                )
            )
        n_recv = int(self._recv_index[stage].numel())
        recv: Optional[Tensor] = None
        if n_recv:
            width = 2 if x_local.dtype.is_complex else 1
            assert self._buffers is not None
            recv = self._buffers[: n_recv * width]
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv,
                    (self._rank + stage) % self._world_size,
                    group=self._group,
                    tag=self._tag + stage,
                )
            )
        requests = dist.batch_isend_irecv(ops) if ops else []
        return requests, recv, n_recv

    def _apply_stage(self, stage: int, values: Tensor, y: Tensor) -> None:
        """Accumulate one ket-owner stage's contributions into ``y``.

        Args:
            stage: Stage index; ``0`` reads ``values`` as the local input.
            values: The stage's ket amplitudes, ordered like its source list.
            y: The owned output block.
        """
        if stage == 0:
            self._apply_order(1, self._local_dets, values, y)
            self._apply_order(2, self._local_dets, values, y)
            return
        index = self._recv_index[stage]
        for order, step in (
            (1, self._single_source_chunk),
            (2, self._double_source_chunk),
        ):
            for start in range(0, int(index.numel()), step):
                stop = min(start + step, int(index.numel()))
                source = self._dets[index[start:stop]]
                self._apply_order(order, source, values[start:stop], y)

    def _apply_order(self, order: int, source: Tensor, values: Tensor, y: Tensor) -> None:
        """Stream one order's table from ``source`` onto the owned rows.

        Doubles try the entry-parallel Triton kernel first when
        ``triton_matvec`` is set, and keep the batched scatter when it declines.

        Args:
            order: ``1`` for singles, ``2`` for doubles.
            source: ``(n_source, n_limbs)`` ket determinants to enumerate from.
            values: ``(n_source,)`` amplitudes, aligned with ``source``.
            y: The owned output block.
        """
        stream = dict(
            dets=self._dets,
            n_spinorb=self._basis.n_spinorb,
            n_particles=self._basis.n_particles,
            device=self._device,
            budget_bytes=self._budget_bytes,
            src_dets=source,
            target_dets=self._local_dets,
        )
        if order == 1:
            for table in _stream_spinless(order=1, **stream):
                self._scatter_single(
                    y=y,
                    source=source,
                    k_bra=table["k_bra"],
                    k_ket=table["k_ket"],
                    ket_only=table["an0"],
                    bra_only=table["cr0"],
                    values=values,
                    chunk_size=self._single_chunk,
                )
        else:
            for table in _stream_spinless(order=2, **stream):
                # The stage table enumerates from the ket side, so its k_bra is
                # the staged source index and its k_ket the owned row. The
                # kernel reads the determinant at its own k_ket and threads the
                # creations (i, j) before the annihilations (a, b), so the two
                # role pairs swap back onto the source determinant: the
                # ket-only orbitals are its creations and the bra-only ones
                # its annihilations. The owned row indexes ``y`` directly,
                # hence ``offset=0``.
                if self._triton_matvec and apply_double_entries(
                    y,
                    values,
                    self._integrals,
                    k_bra=table["k_ket"],
                    k_ket=table["k_bra"],
                    cr0=table["an0"],
                    cr1=table["an1"],
                    an0=table["cr0"],
                    an1=table["cr1"],
                    mode="spinless",
                    dets=source,
                    offset=0,
                ):
                    continue
                self._scatter_double(
                    y=y,
                    source=source,
                    k_bra=table["k_bra"],
                    k_ket=table["k_ket"],
                    ket_only0=table["an0"],
                    ket_only1=table["an1"],
                    bra_only0=table["cr0"],
                    bra_only1=table["cr1"],
                    values=values,
                    chunk_size=self._double_chunk,
                )

    def _scatter_single(
        self,
        *,
        y: Tensor,
        source: Tensor,
        k_bra: Tensor,
        k_ket: Tensor,
        ket_only: Tensor,
        bra_only: Tensor,
        values: Tensor,
        chunk_size: int,
    ) -> None:
        """Scatter ``⟨row|H_1|ket⟩ * x[ket]`` for the singles table of a stage.

        ``k_bra`` indexes the stage's source (ket) determinants, ``k_ket`` the
        owned rows. ``ket_only`` are the orbitals occupied only in the source
        and ``bra_only`` those occupied only in the owned row, which is exactly
        the ``(i, a)`` argument order of ``one_excite_batch`` when the source
        determinant is passed as the ket.
        """
        total = int(k_bra.numel())
        if total == 0:
            return
        chunk = max(1, int(chunk_size))
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            kb = k_bra[start:stop]
            kk = k_ket[start:stop]
            matelem = one_excite_batch(
                source[kb],
                None,
                ket_only[start:stop],
                bra_only[start:stop],
                self._integrals.I1,
                self._integrals.I2_eri,
                self._integrals.n_spinorb,
            )
            y.index_add_(0, kk, matelem * values[kb])

    def _scatter_double(
        self,
        *,
        y: Tensor,
        source: Tensor,
        k_bra: Tensor,
        k_ket: Tensor,
        ket_only0: Tensor,
        ket_only1: Tensor,
        bra_only0: Tensor,
        bra_only1: Tensor,
        values: Tensor,
        chunk_size: int,
    ) -> None:
        """Scatter ``⟨row|H_2|ket⟩ * x[ket]`` for the doubles table of a stage.

        Same role convention as :meth:`_scatter_single`: ``(ket_only0,
        ket_only1)`` are the ``(i, j)`` arguments and ``(bra_only0,
        bra_only1)`` the ``(a, b)`` arguments of ``two_excite_batch``.
        """
        total = int(k_bra.numel())
        if total == 0:
            return
        chunk = max(1, int(chunk_size))
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            kb = k_bra[start:stop]
            kk = k_ket[start:stop]
            matelem = two_excite_batch(
                source[kb],
                None,
                ket_only0[start:stop],
                ket_only1[start:stop],
                bra_only0[start:stop],
                bra_only1[start:stop],
                self._integrals.I2_eri,
            )
            y.index_add_(0, kk, matelem * values[kb])

    @property
    def peak_staged_bytes(self) -> int:
        """Concurrent exchange staging bytes for the current value dtype.

        The receive buffer (``max_stage_size`` values) and the largest send
        payload are alive together during a stage; a send that covers the
        whole local block aliases ``x_local`` and is not counted. Updated by
        the first call, which fixes the value dtype.
        """
        return self._peak_staged_bytes
