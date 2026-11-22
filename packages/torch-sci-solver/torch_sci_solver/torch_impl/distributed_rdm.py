"""Output-partitioned accumulation of distributed RDM tensors.

The serial RDM builders write every contribution into one dense ``(L, L, L, L)``
tensor, and the distributed tile builders replicate that tensor on every rank
before a single ``all_reduce(SUM)``. At large ``L`` the replicated output costs
``O(L^4)`` bytes per rank even though each rank only needs its own share of the
final result.

This module holds the shared machinery that turns such a sweep into a
memory-bounded one:

- :class:`OutputPartition` is the contiguous slice of the flattened output
  index space owned by one rank. It is a pure function of ``(n, world_size,
  rank)``, so every rank derives the same layout without communicating.
- :func:`plan_group_ranges` splits the owner slices into groups of ``groups``
  consecutive ranks. One *pass* processes one group: every rank runs its local
  contribution sweep filtered to the group's output range, the group's dense
  buffer is summed with one ``all_reduce(SUM)``, and each group member keeps its
  own sub-slice. Peak output memory per rank is therefore one group buffer
  plus the persistent local slice, i.e. ``O(L^4 / world_size)`` at
  ``groups=1``, at the cost of ``ceil(world_size / groups)`` sweeps.
- :class:`GroupedOutputAccumulator` owns the pass plan, the reusable group
  buffer, the persistent local slice and the contribution-filtering targets the
  existing accumulation helpers write through. A target mimics the flat tensor
  the helpers already call ``index_add_`` on, so the serial-style sweep code is
  reused unchanged.
- :class:`PartitionedRDM` is the handle returned when the caller asks to keep
  the result partitioned; :meth:`PartitionedRDM.gather` reassembles the full
  tensor on every rank and :meth:`PartitionedRDM.gather_to_root` only on rank 0.
- :func:`should_replicate` decides when the replicated accumulator is cheaper;
  callers fall back to it so the small-``L`` path stays bit-identical.

Collective order
----------------
For a fixed ``(n, world_size, groups)`` every rank executes the same
``ceil(world_size / groups)`` passes in the same order. Each pass performs
exactly one ``all_reduce(SUM)`` of a group buffer whose length is identical on
every rank. The sweep between two passes contains only the operator's own
staged send/recv exchanges, whose sequence is already rank-independent (every
rank pulls the same number of stages). No collective depends on a rank-local
table count, amplitude magnitude or ownership pattern, so the collective order
is deterministic and rank-consistent. Membership of a pass does not depend on
execution timing either: group ``g`` always covers owner ranks
``[g * groups, (g + 1) * groups)``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.distributed_davidson import GlobalSum, _dist, partition_vector

# A replicated ``(3, L, L, L, L)`` accumulator plus the serial builders' own
# temporaries fit comfortably below this for the basis sizes where the
# replicated path is faster; above it the partitioned path is preferred.
DEFAULT_MAX_REPLICATED_BYTES = 256 * 1024 * 1024


def tensor_itemsize(dtype: torch.dtype) -> int:
    """Bytes per element of a dense tensor of ``dtype``.

    Args:
        dtype: Element dtype (real or complex).

    Returns:
        The element size in bytes.
    """
    return int(torch.empty((), dtype=dtype).element_size())


def should_replicate(
    n: int,
    dtype: torch.dtype,
    *,
    world_size: int,
    max_replicated_bytes: int = DEFAULT_MAX_REPLICATED_BYTES,
) -> bool:
    """Whether the replicated accumulator is the right path for this output.

    Args:
        n: Number of elements of the full (replicated) output.
        dtype: Output dtype.
        world_size: Number of ranks in the group.
        max_replicated_bytes: Largest full output that may be replicated per
            rank. ``0`` always selects the partitioned path when ``world_size >
            1``.

    Returns:
        True for a single rank and for a full output that fits the byte bound,
        False otherwise.
    """
    if int(world_size) <= 1:
        return True
    return int(n) * tensor_itemsize(dtype) <= max(0, int(max_replicated_bytes))


@dataclass(frozen=True)
class OutputPartition:
    """Contiguous slice of a flattened output index space owned by one rank.

    Attributes:
        n: Total number of output elements.
        world_size: Number of ranks partitioning the output.
        rank: This slice's rank.
        start: First owned flat index (inclusive).
        stop: One past the last owned flat index (exclusive).
    """

    n: int
    world_size: int
    rank: int
    start: int
    stop: int

    @property
    def local_size(self) -> int:
        """Number of output elements owned by this rank."""
        return self.stop - self.start

    def for_rank(self, rank: int) -> "OutputPartition":
        """Return the slice of another rank.

        Args:
            rank: Rank in ``[0, world_size)``.
        """
        return build_output_partition(self.n, self.world_size, rank)


def build_output_partition(n: int, world_size: int, rank: int) -> OutputPartition:
    """Build one rank's contiguous output slice.

    Args:
        n: Total number of output elements.
        world_size: Number of ranks.
        rank: Rank in ``[0, world_size)``.

    Returns:
        The :class:`OutputPartition` of ``rank``; the bounds are exactly
        :func:`~torch_sci_solver.torch_impl.distributed_davidson.partition_vector`'s.
    """
    bounds = partition_vector(int(n), int(world_size), int(rank))
    return OutputPartition(
        n=int(n),
        world_size=int(world_size),
        rank=int(rank),
        start=int(bounds.start),
        stop=int(bounds.stop),
    )


def plan_group_ranges(n: int, world_size: int, groups: int) -> tuple[tuple[int, int], ...]:
    """Split the owner slices into consecutive groups, returning flat ranges.

    Args:
        n: Total number of output elements.
        world_size: Number of ranks.
        groups: Number of owner slices accumulated per pass. Clamped to
            ``[1, world_size]``.

    Returns:
        One ``(start, stop)`` flat range per pass, ascending and covering
        ``[0, n)``. The plan is a pure function of its arguments.
    """
    world = int(world_size)
    if world <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    size = max(1, min(int(groups), world))
    ranges: list[tuple[int, int]] = []
    for first in range(0, world, size):
        last = min(first + size, world)
        start = build_output_partition(n, world, first).start
        stop = build_output_partition(n, world, last - 1).stop
        ranges.append((int(start), int(stop)))
    return tuple(ranges)


class _FlatTarget:
    """``index_add_`` sink that keeps only a pass's output range.

    The accumulation helpers in the RDM sweeps call
    ``target.view(-1).index_add_(0, index, source)``. This object is the
    ``view(-1)`` result for the partitioned path: it shifts the block-local
    indices by ``base`` and forwards the entries inside the active pass range
    to the shared group buffer.
    """

    __slots__ = ("_accumulator", "_base")

    def __init__(self, accumulator: "GroupedOutputAccumulator", base: int) -> None:
        """Bind a sink to the accumulator and one packed block's base offset."""
        self._accumulator = accumulator
        self._base = int(base)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the group buffer."""
        return self._accumulator.dtype

    @property
    def device(self) -> torch.device:
        """Device of the group buffer."""
        return self._accumulator.device

    def index_add_(self, dim: int, index: Tensor, source: Tensor) -> "_FlatTarget":
        """Add ``source`` at the pass-local positions of ``index + base``.

        Args:
            dim: Must be ``0`` (the helpers only use the flat axis).
            index: ``(k,)`` flat block-local indices.
            source: ``(k,)`` values in the accumulator dtype.

        Returns:
            ``self``, mirroring :meth:`torch.Tensor.index_add_`.
        """
        if dim != 0:
            raise ValueError(f"a flat RDM target only supports index_add_(0, ...), got {dim}")
        self._accumulator.add(int(self._base), index, source)
        return self


class _BlockTarget:
    """One block of a packed multi-block RDM output, for the partitioned sweep.

    Attributes:
        dtype: Dtype of the group buffer.
        device: Device of the group buffer.
    """

    __slots__ = ("_accumulator", "_base")

    def __init__(self, accumulator: "GroupedOutputAccumulator", base: int) -> None:
        """Bind a block target to its packed block's base flat offset."""
        self._accumulator = accumulator
        self._base = int(base)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the group buffer."""
        return self._accumulator.dtype

    @property
    def device(self) -> torch.device:
        """Device of the group buffer."""
        return self._accumulator.device

    def view(self, *_shape: int) -> _FlatTarget:
        """Return the flat sink of this block (only ``view(-1)`` is used)."""
        return _FlatTarget(self._accumulator, self._base)

    def reshape(self, *_shape: int) -> _FlatTarget:
        """Return the flat sink of this block (only ``reshape(-1)`` is used)."""
        return _FlatTarget(self._accumulator, self._base)


class PackedTarget:
    """The packed ``(blocks, ...)`` RDM target of a partitioned sweep.

    Attributes:
        dtype: Dtype of the group buffer.
        device: Device of the group buffer.
    """

    __slots__ = ("_accumulator", "_block_numel", "_blocks")

    def __init__(
        self, accumulator: "GroupedOutputAccumulator", block_numel: int, blocks: int
    ) -> None:
        """Bind the packed target to its block geometry."""
        self._accumulator = accumulator
        self._block_numel = int(block_numel)
        self._blocks = int(blocks)

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the group buffer."""
        return self._accumulator.dtype

    @property
    def device(self) -> torch.device:
        """Device of the group buffer."""
        return self._accumulator.device

    def __getitem__(self, block: int) -> _BlockTarget:
        """Return block ``block``'s target."""
        if not 0 <= int(block) < self._blocks:
            raise IndexError(f"block {block} is outside [0, {self._blocks})")
        return _BlockTarget(self._accumulator, int(block) * self._block_numel)

    def view(self, *_shape: int) -> _FlatTarget:
        """Return the flat sink over every block (only ``view(-1)`` is used)."""
        return _FlatTarget(self._accumulator, 0)

    def reshape(self, *_shape: int) -> _FlatTarget:
        """Return the flat sink over every block (only ``reshape(-1)`` is used)."""
        return _FlatTarget(self._accumulator, 0)


class GroupedOutputAccumulator:
    """Rank-local output slice filled by group-reduced sweep passes.

    Attributes:
        partition: This rank's persistent output slice.
        ranges: One ``(start, stop)`` group range per pass.
        dtype: Output dtype.
        device: Output device.
        group: Process group of the reductions.
    """

    def __init__(
        self,
        n: int,
        *,
        world_size: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        group: Optional[Any] = None,
        groups: int = 1,
    ) -> None:
        """Allocate the persistent slice and the reusable group buffer."""
        self.partition = build_output_partition(n, world_size, rank)
        self.ranges = plan_group_ranges(n, world_size, groups)
        self.dtype = dtype
        self.device = device
        self.group = group
        self._local = torch.zeros(self.partition.local_size, dtype=dtype, device=device)
        max_group = max(stop - start for start, stop in self.ranges)
        self._buffer = torch.zeros(max_group, dtype=dtype, device=device)
        self._active = self._buffer[:0]
        self._lo = 0
        self._hi = 0

    @property
    def local_slice(self) -> Tensor:
        """The persistent ``(local_size,)`` slice owned by this rank."""
        return self._local

    def target(self, blocks: int, block_numel: int) -> PackedTarget:
        """Return a packed target writing into the current pass's buffer.

        Args:
            blocks: Number of packed blocks (2 for the 1-RDM, 3 for the 2-RDM).
            block_numel: Elements per block (``L^2`` or ``L^4``).
        """
        return PackedTarget(self, block_numel, blocks)

    def begin_pass(self, pass_index: int) -> None:
        """Select and clear pass ``pass_index``'s group buffer."""
        lo, hi = self.ranges[pass_index]
        self._lo, self._hi = int(lo), int(hi)
        self._active = self._buffer[: hi - lo]
        self._active.zero_()

    def add(self, base: int, index: Tensor, source: Tensor) -> None:
        """Forward the entries of one accumulation chunk that lie in the pass.

        Args:
            base: Flat offset of the packed block the chunk belongs to.
            index: ``(k,)`` block-local flat indices.
            source: ``(k,)`` values.
        """
        if index.numel() == 0:
            return
        shifted = index.reshape(-1).to(device=self.device, dtype=torch.int64) + int(base)
        keep = (shifted >= self._lo) & (shifted < self._hi)
        if not bool(keep.any()):
            return
        self._active.index_add_(0, shifted[keep] - self._lo, source.reshape(-1)[keep])

    def reduce_and_keep(self) -> None:
        """Sum the pass buffer across the group and keep this rank's slice."""
        GlobalSum(self.group).all_reduce(self._active)
        lo, hi = self._lo, self._hi
        start = max(lo, self.partition.start)
        stop = min(hi, self.partition.stop)
        if start < stop:
            self._local[start - self.partition.start : stop - self.partition.start].copy_(
                self._active[start - lo : stop - lo]
            )


@dataclass(frozen=True)
class PartitionedRDM:
    """A rank-local flat slice of a partitioned RDM output.

    Attributes:
        data: ``(local_size,)`` flat slice owned by this rank.
        partition: The output partition the slice belongs to.
        shape: Full tensor shape the flat index space reshapes to.
    """

    data: Tensor
    partition: OutputPartition
    shape: tuple[int, ...]

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the slice."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Device of the slice."""
        return self.data.device

    def gather(self, *, group: Optional[Any] = None) -> Tensor:
        """Return the full tensor, replicated on every rank.

        Args:
            group: Process group; defaults to the world group.

        Returns:
            The full ``shape`` tensor.
        """
        full = gather_flat_slices(self.data, self.partition, group=group)
        return full.reshape(self.shape)

    def gather_to_root(self, *, group: Optional[Any] = None, root: int = 0) -> Optional[Tensor]:
        """Return the full tensor on ``root`` and ``None`` elsewhere.

        Args:
            group: Process group; defaults to the world group.
            root: Group rank receiving the full tensor.
        """
        full = gather_flat_slices_to_root(self.data, self.partition, group=group, root=root)
        return None if full is None else full.reshape(self.shape)


def _reduce_partitioned(
    n_blocks: int,
    block_numel: int,
    shape: tuple[int, ...],
    accumulate: "Callable[[PackedTarget], None]",
    *,
    n: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    group: Optional[Any] = None,
    groups: int = 1,
) -> PartitionedRDM:
    """Run the output-partitioned sweep and return the packed local slice.

    Shared by the TPB and GDB/structural tiled RDM builders: allocate one
    :class:`GroupedOutputAccumulator`, then for every pass select the group
    buffer, let ``accumulate`` fill the packed target and reduce the pass.
    The collective sequence is one ``all_reduce(SUM)`` per pass.

    Args:
        n_blocks: Packed blocks per pass (2 for the 1-RDM, 3 for the 2-RDM).
        block_numel: Elements per packed block (``L**2`` or ``L**4``).
        shape: Logical output shape stored in the returned
            :class:`PartitionedRDM`.
        accumulate: Called as ``accumulate(target)`` once per pass with that
            pass's packed accumulation target.
        n: Flattened output length.
        world_size, rank, dtype, device: Accumulator settings.
        group: Process group of the reductions; defaults to the world group.
        groups: Owner slices accumulated per pass.

    Returns:
        The packed :class:`PartitionedRDM` holding this rank's slice.
    """
    accumulator = GroupedOutputAccumulator(
        n,
        world_size=world_size,
        rank=rank,
        dtype=dtype,
        device=device,
        group=group,
        groups=groups,
    )
    for pass_index in range(len(accumulator.ranges)):
        accumulator.begin_pass(pass_index)
        accumulate(accumulator.target(n_blocks, block_numel))
        accumulator.reduce_and_keep()
    return PartitionedRDM(
        data=accumulator.local_slice, partition=accumulator.partition, shape=shape
    )


def _can_partition_rdm(
    n_orb: int,
    world_size: int,
    valid: bool,
    *,
    order: int,
    dtype: Optional[torch.dtype],
    max_local_bytes: Optional[int],
    default_dtype: torch.dtype,
) -> bool:
    """Whether an output-partitioned RDM fits this operator layout.

    Shared by the TPB and GDB/structural predicates: both need a valid
    operator, a positive orbital count and world size, and an optional byte
    bound on the per-rank output slice.

    Args:
        n_orb: number of spatial orbitals.
        world_size: number of ranks.
        valid: the operator has the layout the partitioned sweep needs.
        order: ``1`` for the 1-RDM, ``2`` for the 2-RDM.
        dtype: output dtype; defaults to ``default_dtype``.
        max_local_bytes: optional largest acceptable per-rank output slice.
        default_dtype: the operator's diagonal dtype.

    Returns:
        True when the layout can build the requested RDM partitioned.

    Raises:
        ValueError: ``order`` is neither 1 nor 2.
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if not valid:
        return False
    n_orb = int(n_orb)
    world_size = int(world_size)
    if n_orb <= 0 or world_size <= 0:
        return False
    n = (2 * n_orb * n_orb) if order == 1 else (3 * n_orb**4)
    local = (n + world_size - 1) // world_size
    if max_local_bytes is not None:
        element = dtype if dtype is not None else default_dtype
        if local * tensor_itemsize(element) > int(max_local_bytes):
            return False
    return True


def gather_flat_slices(
    local: Tensor, partition: OutputPartition, *, group: Optional[Any] = None
) -> Tensor:
    """All-gather the per-rank slices into the full flat output.

    Complex payloads travel as their interleaved real view, the packing
    :meth:`~torch_sci_solver.torch_impl.distributed_davidson.DistributedVector.
    to_global` uses, because some backends have no complex scalar type.

    Args:
        local: ``(local_size,)`` slice owned by this rank.
        partition: The output partition.
        group: Process group; defaults to the world group.

    Returns:
        The ``(n,)`` full output on every rank.
    """
    world = partition.world_size
    if world == 1:
        return local.reshape(-1).clone()
    dist = _dist()
    device = local.device
    count = torch.tensor([int(local.numel())], dtype=torch.int64, device=device)
    counts = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world)]
    dist.all_gather(counts, count, group=group)
    sizes = [int(item.item()) for item in counts]
    max_size = max(sizes)
    padded = local.new_zeros(max_size)
    padded[: local.numel()] = local.reshape(-1)
    pieces: list[Tensor] = []
    if padded.is_complex():
        payload = torch.view_as_real(padded).reshape(-1)
        received = [
            torch.empty(payload.numel(), dtype=payload.dtype, device=device) for _ in range(world)
        ]
        dist.all_gather(received, payload, group=group)
        pieces = [
            torch.view_as_complex(item.view(-1, 2))[: sizes[rank]]
            for rank, item in enumerate(received)
        ]
    else:
        received = [torch.empty(max_size, dtype=local.dtype, device=device) for _ in range(world)]
        dist.all_gather(received, padded, group=group)
        pieces = [item[: sizes[rank]] for rank, item in enumerate(received)]
    full = torch.empty(partition.n, dtype=local.dtype, device=device)
    for rank, piece in enumerate(pieces):
        bounds = partition.for_rank(rank)
        full[bounds.start : bounds.stop] = piece.to(device=device, dtype=local.dtype)
    return full


def gather_flat_slices_to_root(
    local: Tensor,
    partition: OutputPartition,
    *,
    group: Optional[Any] = None,
    root: int = 0,
) -> Optional[Tensor]:
    """Gather the per-rank slices into the full flat output on ``root`` only.

    Args:
        local: ``(local_size,)`` slice owned by this rank.
        partition: The output partition.
        group: Process group; defaults to the world group.
        root: Group rank receiving the full tensor.

    Returns:
        The ``(n,)`` full output on ``root``, ``None`` on every other rank.
    """
    world = partition.world_size
    if world == 1:
        return local.reshape(-1).clone()
    dist = _dist()
    if not dist.is_initialized():
        raise RuntimeError("a multi-rank gather requires an initialized process group")
    global_root = int(dist.get_global_rank(group, int(root))) if group is not None else int(root)
    rank = int(dist.get_rank(group=group))
    gathered: Optional[list[Any]] = [None] * world if rank == int(root) else None
    dist.gather_object(local.detach().cpu(), gathered, dst=global_root, group=group)
    if rank != int(root):
        return None
    assert gathered is not None
    full = torch.empty(partition.n, dtype=local.dtype, device=local.device)
    for peer, piece in enumerate(gathered):
        bounds = partition.for_rank(peer)
        full[bounds.start : bounds.stop] = piece.to(device=local.device, dtype=local.dtype)
    return full
