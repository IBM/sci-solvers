"""Distributed-memory state-vector support for the SBD/torch_sci Davidson path.

This module backs the **opt-in** ``distributed=`` keyword of
:func:`~torch_sci_solver.torch_impl.qiskit_adapter.make_torch_sci_solver`.
When it is not requested (the default), nothing here is imported and the
existing single-process / batch-parallel multi-GPU behaviour is unchanged.
``torch.distributed`` is imported lazily inside the helpers, so importing this
module never requires a process group, a GPU, or a distributed build.

The scheme follows the SBD-G-style distributed selected-basis diagonalization
described in arXiv:2605.01138: the many-body state vector and the Davidson
subspace are partitioned across ranks, the Hamiltonian stays matrix-free (no
global ``Hxy`` is assembled), and the Davidson projection is formed with
all-rank reductions. The design note — partition strategy, row exchange,
memory accounting, integration points and risks — is
``docs/distributed_state_vector.md``.

Overview
--------
- :func:`partition_vector` / :func:`partition_rows` / :func:`owner_of_index`:
  contiguous, balanced determinant-index blocks (flat, or row-aligned for the
  TPB owner-computes matvec) and the matching index -> rank map.
- :class:`DistributedVector`: an ``N``-vector whose block ``[start, stop)``
  lives on each rank; global dot/norm via :class:`GlobalSum`; ``to_global``
  (all-gather) for the replicated matvec fallback and for reference checks.
- :class:`GlobalSum`: ``all_reduce(SUM)`` helper, a no-op when the group has a
  single rank.
- :func:`local_matvec`: the trivial hook ``hii_local * x_local + apply(x)``
  used by tests and callers that build a local operator by hand.
- :func:`projected_matrix` / :func:`rayleigh_quotient`: the Davidson
  projection ``H_proj[i, j] = <C_i | H C_j>`` as one fused all-reduce.
- :func:`davidson_distributed`: a block Davidson whose ``C``/``HC`` blocks are
  distributed and whose projections/residuals/orthogonalization use all-rank
  reductions. It accepts the same inputs and returns the same shapes as
  ``davidson_torch`` so the solver's post-processing is unchanged.
- :func:`davidson_distributed_local` / :class:`StagedRowExchange`: the
  owner-computes block Davidson and the row exchange used by the solver's TPB
  operator (see ``docs/distributed_state_vector.md``).

``davidson_distributed`` is the replicated fallback: the input vector is
all-gathered and the full-vector matvec runs on every rank, keeping only the
local slice. It is used for the ``spin_sq`` penalty, full-RDM/⟨S²⟩ calls,
an explicit ``matvec="replicated"`` and bases that fit in one block; the
owner-computes paths are ``davidson_distributed_local`` (TPB/GDB/spinless) and
their operators live in ``distributed_matvec.py`` / ``distributed_pairs.py`` /
``distributed_spinless.py``.

Notes
-----
Global determinant indexing matches the existing solver: in the TPB layout
``k = ia * M_b + ib``; in the GDB and spinless layouts ``k`` is the pair index.
A vector is partitioned into contiguous blocks

.. math::

    [s_r, s_{r+1}), \\qquad
    s_r = r \\lfloor N/P \\rfloor + \\min(r, N \\bmod P),

which is balanced to within one determinant and identical on every rank (no
communication is needed to agree on the layout). The Davidson projection is

.. math::

    H^{\\mathrm{proj}}_{ij} = \\langle C_i | H C_j \\rangle
    = \\sum_r \\sum_{k \\in r} \\overline{C_i[k]} (H C_j)[k],

one ``all_reduce`` of the replicated ``(d, d)`` partial matrix; the ``d x d``
eigendecomposition is then small and replicated on every rank, so all ranks
follow the same control flow.

Reference: arXiv:2605.01138 (SBD-G; the distributed selected-basis scheme this
implementation is derived from). The existing SBD GPU benchmark line is
arXiv:2601.16637.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import _signed_int64

__all__ = [
    "RankBounds",
    "DistributedContext",
    "partition_vector",
    "partition_vector_weighted",
    "VectorPartition",
    "partition_rows",
    "partition_rows_explicit",
    "partition_rows_weighted",
    "TileGrid",
    "TileBounds",
    "TilePartition",
    "partition_tiles",
    "resolve_tile_grid",
    "owner_of_index",
    "GlobalSum",
    "DistributedVector",
    "FlatGather",
    "local_matvec",
    "RowPartition",
    "StagedRowExchange",
    "StagedSlabExchange",
    "StagedCornerExchange",
    "hash_uniform",
    "global_smallest_hii",
    "projected_matrix",
    "rayleigh_quotient",
    "DavidsonCheckpoint",
    "davidson_distributed",
    "davidson_distributed_local",
]


def _dist() -> Any:
    """Return ``torch.distributed``, importing it lazily.

    Importing this module must not pull in ``torch.distributed`` (or require a
    process group) for the ordinary single-process solver path.
    """
    import torch.distributed as dist

    return dist


# ============================================================================
# Partition
# ============================================================================


@dataclass(frozen=True)
class RankBounds:
    """Contiguous determinant-index block owned by one rank.

    Attributes:
        n: Global number of determinants.
        world_size: Number of ranks the vector is partitioned over.
        rank: This block's rank (0 <= rank < world_size).
        start: First global index owned (inclusive).
        stop: One past the last global index owned (exclusive).
        max_local: Largest local block over all ranks, when the partition
            policy knows it (row-aligned blocks can be larger than
            ``ceil(n / world_size)``); ``None`` means ``ceil(n / world_size)``.
        n_rows: For a row-aligned TPB partition, the global α-row count;
            ``None`` for a flat partition.
        row_size: For a row-aligned TPB partition, the β-column count;
            ``None`` for a flat partition.
    """

    n: int
    world_size: int
    rank: int
    start: int
    stop: int
    max_local: Optional[int] = None
    n_rows: Optional[int] = None
    row_size: Optional[int] = None

    @property
    def local_size(self) -> int:
        """Number of determinants owned by this rank."""
        return self.stop - self.start

    @property
    def is_empty(self) -> bool:
        """Whether this rank owns no determinant (``n < world_size``)."""
        return self.local_size == 0

    def owns(self, index: int) -> bool:
        """Whether ``index`` belongs to this block."""
        return self.start <= index < self.stop

    def slice(self, tensor: Tensor) -> Tensor:
        """Return the ``[start, stop)`` slice of a global-length tensor."""
        return tensor[self.start : self.stop]

    def __iter__(self):
        """Unpack as ``(start, stop)``."""
        yield self.start
        yield self.stop


@dataclass(frozen=True)
class DistributedContext:
    """Identity of one rank in an initialized process group.

    Attributes:
        rank: Rank index in ``[0, world_size)``.
        world_size: Number of ranks.
        backend: ``"gloo"`` (CPU) or ``"nccl"`` (GPU).
        device: Torch device this rank's blocks live on.
        group: Process group for collectives; ``None`` uses the default group.
        tile: 2-D tile mode: ``None`` keeps the 1-D row partition, ``"auto"``
            picks the grid, ``(p_alpha, p_beta)`` fixes it.
        exchange_bytes: Optional element budget for one staged tile exchange;
            a larger αβ corner rectangle is split into row bands.
        hand_back_dir: Directory for the memmapped large-result hand-back. It
            must be visible from every rank, so multi-node runs point it at
            shared storage; ``None`` uses the node-local temporary directory.
    """

    rank: int
    world_size: int
    backend: str
    device: torch.device
    group: Optional[Any] = None
    matvec: str = "local"
    tile: Optional[Any] = None
    exchange_bytes: Optional[int] = None
    hand_back_dir: Optional[str] = None


@dataclass(frozen=True)
class DavidsonCheckpoint:
    """Iteration-boundary persistence hook for a distributed Davidson solve.

    The Davidson loops call :attr:`save` at the top of an iteration (when
    ``iteration > 0`` and ``iteration % every == 0``) and :attr:`load` once
    before seeding. The state passed to ``save`` is rank-local, so no rank ever
    materializes the full subspace; the hook writes it, and the solver's
    checkpoint layer supplies the manifest identity.

    A ``None`` hook is the zero-overhead path: both loops skip every
    checkpoint branch and never build a state dict.

    Attributes:
        every: Positive number of iterations between writes.
        load: ``() -> {"iteration", "layout_kind", "state"}`` or ``None``.
            Called by every rank, so it may use collectives.
        save: ``(iteration, state) -> None``. Called by every rank at the same
            iteration; ``state`` holds that rank's local blocks.
    """

    every: int
    load: Optional[Callable[[], Optional[dict]]] = None
    save: Optional[Callable[[int, dict], None]] = None

    def __post_init__(self) -> None:
        """Reject a non-positive write interval."""
        if int(self.every) <= 0:
            raise ValueError(f"DavidsonCheckpoint.every must be positive, got {self.every}")


def partition_vector(n: int, world_size: int, rank: int) -> RankBounds:
    """Return the contiguous index block owned by ``rank``.

    The first ``n % world_size`` ranks get one extra determinant; every rank
    computes the same bounds from ``(n, world_size)`` alone, so no
    communication is needed to agree on the layout.

    Args:
        n: Global vector length (determinant count).
        world_size: Number of ranks.
        rank: Rank whose block is requested.

    Returns:
        The :class:`RankBounds` for ``rank``.

    Raises:
        ValueError: ``n`` is negative, ``world_size <= 0``, or ``rank`` is out
            of range.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    base, remainder = divmod(n, world_size)
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return RankBounds(n=n, world_size=world_size, rank=rank, start=start, stop=stop)


@dataclass(frozen=True)
class VectorPartition:
    """Explicit contiguous per-rank partition of a flat length-``n`` vector.

    Unlike :func:`partition_vector`, the boundaries are stored for every rank,
    so an uneven (cost-balanced) split can be described and handed to the
    result gather without communication.

    Attributes:
        n: Global vector length.
        bounds: One contiguous :class:`RankBounds` per rank, ascending and
            covering ``[0, n)``.
    """

    n: int
    bounds: tuple[RankBounds, ...]

    def __post_init__(self) -> None:
        """Validate that the stored bounds form a contiguous covering."""
        if self.n < 0:
            raise ValueError(f"n must be >= 0, got {self.n}")
        if not self.bounds:
            raise ValueError("a VectorPartition needs at least one rank")
        previous_stop = 0
        for rank, bound in enumerate(self.bounds):
            if (
                bound.n != self.n
                or bound.world_size != len(self.bounds)
                or bound.rank != rank
                or bound.start != previous_stop
                or bound.stop < bound.start
            ):
                raise ValueError(f"bounds[{rank}]={bound} is not a contiguous covering of {self.n}")
            previous_stop = bound.stop
        if previous_stop != self.n:
            raise ValueError(f"bounds cover [0, {previous_stop}), expected [0, {self.n})")

    @property
    def max_local(self) -> int:
        """Largest per-rank block size (the gather's padding width)."""
        return max(bound.local_size for bound in self.bounds)

    def for_rank(self, rank: int) -> RankBounds:
        """The bounds of ``rank``."""
        return self.bounds[rank]


def partition_vector_weighted(n: int, world_size: int, weights: Tensor) -> VectorPartition:
    """Contiguous index blocks with approximately equal cumulative weight.

    The boundaries are cut on the prefix sum of the replicated non-negative
    per-index cost ``weights`` so every rank gets about ``1 / world_size`` of
    the total cost. The result is deterministic and identical on every rank.

    Args:
        n: Global vector length.
        world_size: Number of ranks.
        weights: ``(n,)`` non-negative per-index cost, replicated.

    Returns:
        The explicit flat partition.

    Raises:
        ValueError: invalid sizes, weights shape/dtype or negative weights.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if tuple(weights.shape) != (n,):
        raise ValueError(f"weights shape {tuple(weights.shape)} != ({n},)")
    if n == 0:
        return VectorPartition(
            0, tuple(partition_vector(0, world_size, rank) for rank in range(world_size))
        )
    cost = weights.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if cost.is_complex() or not bool(torch.isfinite(cost).all()):
        raise ValueError("weights must be finite real costs")
    if bool((cost < 0).any()):
        raise ValueError("weights must be non-negative")
    total = float(cost.sum())
    if total <= 0.0:
        return VectorPartition(
            n, tuple(partition_vector(n, world_size, rank) for rank in range(world_size))
        )
    prefix = torch.cumsum(cost, dim=0)
    cuts = [0]
    for rank in range(world_size - 1):
        target = float(prefix[-1]) * (rank + 1) / world_size
        cuts.append(int(torch.searchsorted(prefix, torch.tensor(target, dtype=torch.float64))))
    cuts.append(n)
    bounds = tuple(
        RankBounds(
            n=n,
            world_size=world_size,
            rank=rank,
            start=int(cuts[rank]),
            stop=int(cuts[rank + 1]),
        )
        for rank in range(world_size)
    )
    return VectorPartition(n, bounds)


def partition_rows(n_rows: int, row_size: int, world_size: int, rank: int) -> RankBounds:
    """Row-aligned partition of a ``(n_rows, row_size)`` coefficient matrix.

    Rank ``r`` owns the contiguous α-rows ``[r_start, r_stop)`` and *all*
    ``row_size`` columns, i.e. the flat block
    ``[r_start * row_size, r_stop * row_size)``. This is the layout the
    owner-computes TPB matvec needs (α-only/αβ restrictions and β-only
    locality), and it keeps the first ranks' blocks no smaller than the last
    ranks', which is what :meth:`DistributedVector.to_global`'s padding
    assumes.

    Args:
        n_rows: Number of rows (α-strings ``M_a``).
        row_size: Number of columns (β-strings ``M_b``).
        world_size: Number of ranks.
        rank: Rank whose block is requested.

    Returns:
        A :class:`RankBounds` over the flat ``n_rows * row_size`` vector.
    """
    if row_size <= 0:
        raise ValueError(f"row_size must be > 0, got {row_size}")
    rows = partition_vector(n_rows, world_size, rank)
    max_rows = (n_rows + world_size - 1) // world_size
    return RankBounds(
        n=n_rows * row_size,
        world_size=world_size,
        rank=rank,
        start=rows.start * row_size,
        stop=rows.stop * row_size,
        max_local=max_rows * row_size,
        n_rows=n_rows,
        row_size=row_size,
    )


def owner_of_index(index: Tensor | int, n: int, world_size: int) -> Tensor:
    """Map global determinant indices to their owning ranks.

    Vectorized inverse of :func:`partition_vector`: for every ``k`` in
    ``[0, n)`` the returned rank ``r`` satisfies
    ``partition_vector(n, world_size, r).owns(k)``. This is how a matvec maps
    an excitation destination (a global ket index stored in the helper tables)
    to the rank holding ``x[k]`` — no hash table and no communication.

    Args:
        index: Scalar or tensor of global indices in ``[0, n)``.
        n: Global vector length.
        world_size: Number of ranks.

    Returns:
        An int64 tensor of owning ranks, same shape/device as ``index``.
    """
    idx = torch.as_tensor(index, dtype=torch.int64)
    if n < 0 or world_size <= 0:
        raise ValueError(f"need n >= 0 and world_size > 0, got {n}, {world_size}")
    if idx.numel() == 0:
        return idx.clone()
    if int(idx.min()) < 0 or int(idx.max()) >= n:
        raise ValueError(f"index out of range [0, {n})")
    base, remainder = divmod(n, world_size)
    if base == 0:
        # n < world_size: ranks 0..n-1 own exactly one determinant each.
        return idx.clone()
    first_short = remainder * (base + 1)
    return torch.where(
        idx < first_short,
        idx // (base + 1),
        remainder + (idx - first_short) // base,
    )


# ============================================================================
# All-rank reductions
# ============================================================================


class GlobalSum:
    """All-rank sum, a no-op when the process group has a single rank.

    Every reduction goes through this helper so the same code runs (a)
    distributed under ``torch.distributed`` and (b) single-process with no
    process group — which is what keeps the math testable and lets the solver
    fall back to the existing path unchanged.
    """

    def __init__(self, group: Optional[Any] = None) -> None:
        """Wrap ``group`` (default: the default process group).

        Args:
            group: Process group to reduce over. ``None`` uses
                ``torch.distributed.group.WORLD``.
        """
        self.group = group
        dist = _dist()
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size(group)
        else:
            self.world_size = 1

    @property
    def enabled(self) -> bool:
        """Whether a real multi-rank collective is performed."""
        return self.world_size > 1

    def all_reduce(self, tensor: Tensor) -> Tensor:
        """Sum ``tensor`` across ranks in place and return it.

        Scalars (0-dim tensors) are reduced through a 1-element view, and
        complex tensors as an explicit ``(real, imag)`` pair — both are
        portable across gloo and NCCL builds.

        Args:
            tensor: Dense tensor to reduce. Must be identical in shape, dtype
                and device on every rank.

        Returns:
            The same tensor object, now holding the all-rank sum.
        """
        if not self.enabled:
            return tensor
        dist = _dist()
        if not dist.is_initialized():
            raise RuntimeError("GlobalSum has world_size > 1 but no process group is initialized")
        if tensor.is_complex():
            real = torch.view_as_real(tensor.contiguous())
            dist.all_reduce(real, op=dist.ReduceOp.SUM, group=self.group)
            tensor.copy_(torch.view_as_complex(real))
        elif tensor.dim() == 0:
            flat = tensor.reshape(1)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.group)
            tensor.copy_(flat[0])
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        return tensor

    def scalar(
        self,
        value: float,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> float:
        """Return the all-rank sum of a Python scalar.

        Args:
            value: This rank's contribution.
            device: Device for the temporary (needed for NCCL; CPU is fine for
                gloo).
            dtype: Real dtype of the temporary.
        """
        tensor = torch.tensor(float(value), dtype=dtype, device=device)
        self.all_reduce(tensor)
        return float(tensor)


# ============================================================================
# Distributed vector
# ============================================================================


def _resolve_rank_world(
    rank: Optional[int],
    world_size: Optional[int],
    group: Optional[Any],
) -> tuple[int, int]:
    """Resolve rank/world size from the process group and explicit overrides."""
    dist = _dist()
    initialized = dist.is_available() and dist.is_initialized()
    actual_world = dist.get_world_size(group) if initialized else 1
    if world_size is None:
        world = actual_world
    else:
        world = int(world_size)
        if initialized and world != actual_world:
            raise ValueError(
                f"explicit world_size={world} disagrees with the process group's "
                f"world_size={actual_world}"
            )
    if not initialized and world > 1:
        raise ValueError(
            "a multi-rank DistributedVector needs an initialized process group "
            "(torch.distributed.init_process_group)"
        )
    if rank is None:
        resolved = dist.get_rank(group) if initialized else 0
    else:
        resolved = int(rank)
    if not 0 <= resolved < world:
        raise ValueError(f"rank must be in [0, {world}), got {resolved}")
    return resolved, world


class DistributedVector:
    """An ``N``-vector whose contiguous block lives on each rank.

    The global vector is never materialized per rank except by an explicit
    :meth:`to_global` (the all-gather used by the replicated matvec fallback
    and by reference checks). ``dot``/``norm`` are global: they reduce the
    local partial sum with :class:`GlobalSum`. The TPB owner-computes matvec
    avoids the gather with :class:`StagedRowExchange`; the flat GDB/spinless
    operators deliberately all-gather once per matvec.
    """

    def __init__(
        self,
        local: Tensor,
        n: int,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        bounds: Optional[RankBounds] = None,
    ) -> None:
        """Wrap the local block of a global ``n``-vector.

        Args:
            local: This rank's contiguous block, shape
                ``(partition_vector(n, world_size, rank).local_size,)``.
            n: Global vector length.
            rank: Override the rank from the process group.
            world_size: Override the world size from the process group.
            group: Process group used by every collective.
            bounds: Optional explicit block (e.g. :func:`partition_rows` for
                the owner-computes TPB layout). Must be consistent with
                ``n``/``rank``/``world_size``.

        Raises:
            ValueError: rank/world-size resolution fails, ``local`` has the
                wrong shape, or ``bounds`` is inconsistent.
        """
        self.rank, self.world_size = _resolve_rank_world(rank, world_size, group)
        self.n = int(n)
        self.group = group
        if bounds is None:
            self.bounds = partition_vector(self.n, self.world_size, self.rank)
        else:
            if (
                bounds.n != self.n
                or bounds.world_size != self.world_size
                or bounds.rank != self.rank
                or not 0 <= bounds.start <= bounds.stop <= self.n
            ):
                raise ValueError(
                    f"bounds {bounds} are inconsistent with n={self.n}, "
                    f"rank={self.rank}, world_size={self.world_size}"
                )
            self.bounds = bounds
        if local.shape != (self.bounds.local_size,):
            raise ValueError(
                f"local block shape {tuple(local.shape)} != "
                f"({self.bounds.local_size},) for rank {self.rank}"
            )
        self.local = local
        self.global_sum = GlobalSum(group)

    @classmethod
    def zeros(
        cls,
        n: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device | str] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        bounds: Optional[RankBounds] = None,
    ) -> DistributedVector:
        """Return a zero vector partitioned like any other ``n``-vector."""
        resolved_rank, resolved_world = _resolve_rank_world(rank, world_size, group)
        if bounds is None:
            bounds = partition_vector(n, resolved_world, resolved_rank)
        local = torch.zeros(bounds.local_size, dtype=dtype, device=device)
        return cls(
            local,
            n,
            rank=resolved_rank,
            world_size=resolved_world,
            group=group,
            bounds=bounds,
        )

    @classmethod
    def from_global(
        cls,
        tensor: Tensor,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        bounds: Optional[RankBounds] = None,
    ) -> DistributedVector:
        """Slice a global-length tensor into this rank's block.

        Used for the initial guess (every rank may build the same global
        vector) and for reference checks. The block is cloned so the result
        does not keep the full tensor alive.
        """
        if tensor.dim() != 1:
            raise ValueError(f"expected a 1-D global vector, got shape {tuple(tensor.shape)}")
        resolved_rank, resolved_world = _resolve_rank_world(rank, world_size, group)
        if bounds is None:
            bounds = partition_vector(tensor.shape[0], resolved_world, resolved_rank)
        elif bounds.n != tensor.shape[0]:
            raise ValueError(f"bounds.n={bounds.n} != tensor length {tensor.shape[0]}")
        return cls(
            tensor[bounds.start : bounds.stop].clone(),
            tensor.shape[0],
            rank=resolved_rank,
            world_size=resolved_world,
            group=group,
            bounds=bounds,
        )

    def to_global(self) -> Tensor:
        """All-gather the local block into the full ``N``-vector on every rank.

        Blocks are padded to the largest block so the collective sees
        equal-sized tensors on every rank (gloo/NCCL requirement). The block
        offsets are read from the ranks themselves, so an explicit
        (cost-balanced) partition works as well as the canonical one; a small
        all-gather of the block sizes is used, on a path that already
        all-gathers the payload. Complex blocks are gathered as an explicit
        ``(real, imag)`` pair, like :meth:`GlobalSum.all_reduce`, because NCCL
        builds may reject a complex tensor directly. ``to_global`` is the
        replicated matvec's input path; the flat GDB/spinless operators gather
        through :class:`FlatGather` and the TPB owner-computes matvec avoids a
        full gather with :class:`StagedRowExchange` (see the design note).
        """
        if self.world_size == 1:
            return self.local.clone()
        dist = _dist()
        size = torch.tensor([self.bounds.local_size], dtype=torch.int64, device=self.local.device)
        gathered_sizes = [torch.empty_like(size) for _ in range(self.world_size)]
        dist.all_gather(gathered_sizes, size, group=self.group)
        rank_bounds = []
        start = 0
        for rank, chunk in enumerate(gathered_sizes):
            stop = start + int(chunk.item())
            rank_bounds.append(
                RankBounds(n=self.n, world_size=self.world_size, rank=rank, start=start, stop=stop)
            )
            start = stop
        if start != self.n or rank_bounds[self.rank].start != self.bounds.start:
            raise RuntimeError("gathered block sizes do not match this vector's partition")
        max_local = max(bound.local_size for bound in rank_bounds)
        if self.local.is_complex():
            padded = torch.zeros(
                max_local, 2, dtype=self.local.real.dtype, device=self.local.device
            )
            padded[: self.bounds.local_size] = torch.view_as_real(self.local.contiguous())
            gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
            dist.all_gather(gathered, padded, group=self.group)
            full_real = torch.zeros(
                self.n, 2, dtype=self.local.real.dtype, device=self.local.device
            )
            for rank, bound in enumerate(rank_bounds):
                full_real[bound.start : bound.stop] = gathered[rank][: bound.local_size]
            return torch.view_as_complex(full_real.contiguous())
        padded = torch.zeros(max_local, dtype=self.local.dtype, device=self.local.device)
        padded[: self.bounds.local_size] = self.local
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded, group=self.group)
        full = torch.zeros(self.n, dtype=self.local.dtype, device=self.local.device)
        for rank, bound in enumerate(rank_bounds):
            full[bound.start : bound.stop] = gathered[rank][: bound.local_size]
        return full

    def dot(self, other: DistributedVector) -> Tensor:
        """Global Hermitian inner product ``<self | other>`` (0-dim tensor).

        Every rank returns the same value (``all_reduce``, not
        ``reduce`` + broadcast), so all ranks can branch identically.
        """
        self._check_compatible(other)
        return self.global_sum.all_reduce(torch.vdot(self.local, other.local))

    def norm(self) -> Tensor:
        """Global Euclidean norm (0-dim real tensor), identical on every rank."""
        return torch.sqrt(self.dot(self).real.clamp_min(0.0))

    def scale(self, alpha: Tensor | float) -> DistributedVector:
        """Return ``alpha * self`` (block-local, no communication)."""
        return DistributedVector(
            self.local * alpha,
            self.n,
            rank=self.rank,
            world_size=self.world_size,
            group=self.group,
            bounds=self.bounds,
        )

    def add_scaled(self, other: DistributedVector, alpha: Tensor | float) -> DistributedVector:
        """Return ``self + alpha * other`` (block-local, no communication)."""
        self._check_compatible(other)
        return DistributedVector(
            self.local + alpha * other.local,
            self.n,
            rank=self.rank,
            world_size=self.world_size,
            group=self.group,
            bounds=self.bounds,
        )

    def clone(self) -> DistributedVector:
        """Return a copy of this distributed vector."""
        return DistributedVector(
            self.local.clone(),
            self.n,
            rank=self.rank,
            world_size=self.world_size,
            group=self.group,
            bounds=self.bounds,
        )

    def _check_compatible(self, other: DistributedVector) -> None:
        """Validate that two vectors can be combined elementwise."""
        if not isinstance(other, DistributedVector):
            raise TypeError(f"expected a DistributedVector, got {type(other).__name__}")
        if other.n != self.n or other.world_size != self.world_size:
            raise ValueError(
                f"incompatible vectors: n/world_size ({other.n}, {other.world_size}) "
                f"!= ({self.n}, {self.world_size})"
            )
        if other.bounds != self.bounds:
            raise ValueError(
                f"incompatible vectors: partition bounds differ ({other.bounds} != {self.bounds})"
            )
        if other.local.dtype != self.local.dtype or other.local.device != self.local.device:
            raise ValueError("incompatible vectors: dtype or device differs")

    def __len__(self) -> int:
        """Global vector length ``N`` (not the local block length)."""
        return self.n

    def __repr__(self) -> str:
        """Compact description including the owned global range."""
        return (
            f"DistributedVector(n={self.n}, rank={self.rank}/{self.world_size}, "
            f"range=[{self.bounds.start}, {self.bounds.stop}))"
        )


class FlatGather:
    """Asynchronous all-gather of a flat block-partitioned vector (collective).

    The flat GDB and spinless operators read their kets from the full input
    vector, so every matvec all-gathers the owned blocks. The block sizes are
    fixed for an operator's lifetime, so this plan exchanges them once, on the
    first :meth:`start` (one small collective), instead of once per matvec; each
    matvec is then a single ``all_gather_into_tensor`` posted with
    ``async_op=True``, and the caller does work that does not read the vector
    (the diagonal, the excitation-table enumeration) between :meth:`start` and
    :meth:`wait`.

    Notes
    -----
    Each rank sends its block padded to ``m = max_r |B_r|`` and receives the
    rank-major buffer ``R`` of ``P m`` entries; the full vector is
    ``x[B_r] = R[r m : r m + |B_r|]``. When every block has size ``m`` (the
    canonical even split) ``R`` *is* ``x``: the block is sent without a padded
    copy and :meth:`wait` returns ``R`` itself. Otherwise :meth:`wait` compacts
    into one length-``N`` tensor. Complex blocks travel as their ``(real, imag)``
    pairs, because NCCL builds may reject complex tensors. A backend without
    ``all_gather_into_tensor`` (gloo in older torch) takes the list
    ``all_gather`` instead, with the same result.

    Args:
        bounds: this rank's contiguous block.
        group: process group (``None`` for the default group).
    """

    def __init__(self, bounds: RankBounds, *, group: Optional[Any] = None) -> None:
        """Bind the plan to one partition; no collective runs until the first start."""
        self.bounds = bounds
        self.group = group
        self.world_size = int(bounds.world_size)
        self.n = int(bounds.n)
        self._sizes: Optional[list[int]] = None
        self._flat_collective = True
        self._pending: Optional[tuple[Any, Any, Tensor, bool]] = None

    def _gather(self, send: Tensor, *, async_op: bool) -> tuple[Any, Any]:
        """Post one gather of ``send``; returns ``(work, receive buffer or list)``."""
        dist = _dist()
        if self._flat_collective:
            recv = torch.empty(
                (self.world_size * send.shape[0], *send.shape[1:]),
                dtype=send.dtype,
                device=send.device,
            )
            try:
                return dist.all_gather_into_tensor(
                    recv, send, group=self.group, async_op=async_op
                ), recv
            except (AttributeError, RuntimeError, NotImplementedError):
                # Every rank fails the same local capability check here.
                self._flat_collective = False
        parts = [torch.empty_like(send) for _ in range(self.world_size)]
        return dist.all_gather(parts, send, group=self.group, async_op=async_op), parts

    def _block_sizes(self, device: torch.device) -> list[int]:
        """Every rank's block size, exchanged once (collective on first use)."""
        if self._sizes is None:
            size = torch.tensor([self.bounds.local_size], dtype=torch.int64, device=device)
            _, gathered = self._gather(size, async_op=False)
            if isinstance(gathered, list):
                gathered = torch.cat(gathered)
            sizes = [int(v) for v in gathered.tolist()]
            if sum(sizes) != self.n or sum(sizes[: self.bounds.rank]) != self.bounds.start:
                raise RuntimeError("gathered block sizes do not match this vector's partition")
            self._sizes = sizes
        return self._sizes

    def start(self, local: Tensor) -> "FlatGather":
        """Post the all-gather of ``local``; pair every call with :meth:`wait`."""
        if self._pending is not None:
            raise RuntimeError("FlatGather.start called twice without wait")
        if local.shape != (self.bounds.local_size,):
            raise ValueError(
                f"local block shape {tuple(local.shape)} != ({self.bounds.local_size},)"
            )
        if self.world_size == 1:
            if self.bounds.start != 0 or self.bounds.stop != self.n:
                raise ValueError(f"a single rank must own [0, {self.n}), got {self.bounds}")
            self._pending = (None, local, local, local.is_complex())
            return self
        sizes = self._block_sizes(local.device)
        width = max(sizes)
        is_complex = local.is_complex()
        payload = torch.view_as_real(local.contiguous()) if is_complex else local.contiguous()
        if self.bounds.local_size == width:
            send = payload
        else:
            send = torch.zeros(
                (width, *payload.shape[1:]), dtype=payload.dtype, device=local.device
            )
            send[: self.bounds.local_size] = payload
        work, recv = self._gather(send, async_op=True)
        # ``send`` stays referenced until wait, so its storage outlives the
        # in-flight collective on every backend.
        self._pending = (work, recv, send, is_complex)
        return self

    def done(self) -> bool:
        """Whether the posted gather has completed (never blocks)."""
        if self._pending is None:
            raise RuntimeError("FlatGather.done called without start")
        work = self._pending[0]
        return work is None or bool(work.is_completed())

    def wait(self) -> Tensor:
        """Wait for the posted gather and return the full length-``N`` vector."""
        if self._pending is None:
            raise RuntimeError("FlatGather.wait called without start")
        work, recv, _send, is_complex = self._pending
        self._pending = None
        if work is None:
            return recv.clone()
        work.wait()
        if isinstance(recv, list):
            recv = torch.cat(recv)
        sizes = self._sizes
        assert sizes is not None
        width = max(sizes)
        if all(size == width for size in sizes):
            full = recv
        else:
            full = torch.cat([
                recv[rank * width : rank * width + size] for rank, size in enumerate(sizes)
            ])
        return torch.view_as_complex(full.contiguous()) if is_complex else full


# ============================================================================
# Matvec hook and Davidson projection
# ============================================================================


LocalApply = Callable[[Tensor, Tensor, RankBounds], Tensor]


def local_matvec(
    x: DistributedVector,
    hii: Optional[Tensor] = None,
    local_apply: Optional[LocalApply] = None,
) -> DistributedVector:
    """Return the locally owned block of ``H @ x``.

    The operator is matrix-free: no global ``Hxy`` exists. Rank ``r`` writes
    only its owned output block,

    .. math::

        y[k] = h_{kk} x_k + \\texttt{local\\_apply}(x, k), \\quad
        k \\in [s_r, s_{r+1}),

    so ``hii`` is the local diagonal slice and ``local_apply`` carries the
    off-diagonal (excitation) terms.

    Args:
        x: Input distributed vector.
        hii: This rank's diagonal block, shape ``(bounds.local_size,)``. ``None``
            means the operator has no diagonal part.
        local_apply: Hook ``(x_local, x_full, bounds) -> y_local`` computing
            the off-diagonal contribution to this rank's output block. It
            receives the all-gathered full input vector so it can read remote
            ``x[j]``.
            It must return exactly ``(bounds.local_size,)`` entries and must
            not include the diagonal term.

    Returns:
        The distributed output vector ``H @ x``.

    Raises:
        ValueError: ``hii`` has the wrong shape, ``local_apply`` returns the
            wrong shape, or neither term is supplied.
    """
    if hii is None and local_apply is None:
        raise ValueError("at least one of hii / local_apply must be supplied")
    y_local = torch.zeros_like(x.local)
    if hii is not None:
        if hii.shape != (x.bounds.local_size,):
            raise ValueError(f"hii block shape {tuple(hii.shape)} != ({x.bounds.local_size},)")
        y_local = y_local + hii * x.local
    if local_apply is not None:
        term = local_apply(x.local, x.to_global(), x.bounds)
        if term.shape != (x.bounds.local_size,):
            raise ValueError(
                f"local_apply returned shape {tuple(term.shape)} != ({x.bounds.local_size},)"
            )
        y_local = y_local + term
    return DistributedVector(
        y_local,
        x.n,
        rank=x.rank,
        world_size=x.world_size,
        group=x.group,
        bounds=x.bounds,
    )


def projected_matrix(
    vectors: Iterable[DistributedVector],
    products: Iterable[DistributedVector],
    global_sum: Optional[GlobalSum] = None,
) -> Tensor:
    """Return the replicated Davidson projection ``H_proj[i, j] = <C_i | H C_j>``.

    Each rank forms the ``(d, d)`` partial matrix over its owned indices,
    ``V_local^H W_local``, and a **single fused** all-reduce sums the partials,
    so the cost is one collective per subspace build rather than ``d * d``
    scalar reductions. Every rank then diagonalizes the same small matrix
    locally; no broadcast of the Ritz coefficients is needed.

    Args:
        vectors: ``d`` subspace vectors ``C_i``.
        products: ``d`` Hamiltonian images ``H C_i``, in the same order.
        global_sum: Reduction helper; defaults to ``vectors[0].global_sum``.

    Returns:
        The ``(d, d)`` projection on every rank (same dtype/device as the
        blocks). For a Hermitian ``H`` it is Hermitian up to round-off.

    Raises:
        ValueError: empty input, mismatched lengths, or incompatible vectors.
    """
    vectors = list(vectors)
    products = list(products)
    if not vectors or len(vectors) != len(products):
        raise ValueError(
            f"need the same non-zero number of vectors and products, "
            f"got {len(vectors)} and {len(products)}"
        )
    for vector, product in zip(vectors, products):
        if not isinstance(product, DistributedVector):
            raise TypeError(f"expected a DistributedVector product, got {type(product).__name__}")
        if product.n != vector.n or product.world_size != vector.world_size:
            raise ValueError("vectors and products are not partitioned alike")
    local = torch.stack([v.local for v in vectors]).conj() @ torch.stack([
        w.local for w in products
    ]).transpose(0, 1)
    resolver = global_sum if global_sum is not None else vectors[0].global_sum
    return resolver.all_reduce(local)


def rayleigh_quotient(
    vector: DistributedVector,
    product: DistributedVector,
    global_sum: Optional[GlobalSum] = None,
) -> Tensor:
    """Return ``<v | H v> / <v | v>`` as a 0-dim real tensor, replicated.

    For a Hermitian operator the quotient is real up to round-off; the
    imaginary residue is dropped.
    """
    resolver = global_sum if global_sum is not None else vector.global_sum
    numerator = resolver.all_reduce(torch.vdot(vector.local, product.local))
    denominator = resolver.all_reduce(torch.vdot(vector.local, vector.local))
    if float(denominator.real) == 0.0:
        raise ValueError("Rayleigh quotient of a zero vector is undefined")
    return (numerator / denominator).real


# ============================================================================
# Distributed Davidson
# ============================================================================


def _format_davidson_result(
    energies: Tensor,
    vectors: Tensor,
    info: dict,
    n_roots: int,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Match ``davidson_torch``'s single-/multi-root return shapes."""
    if n_roots == 1:
        return float(energies[0]), vectors[0], info
    return energies, vectors, info


def _davidson_distributed_dense(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    eps: float,
    *,
    n_roots: int,
    project_fn: Optional[Callable[[Tensor], Tensor]],
    context: Optional[DistributedContext] = None,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Dense diagonalization when the whole CI space fits in one block.

    Mirrors ``davidson_torch``'s small-``N`` escape: every rank builds the
    same dense matrix (replicated, small) and returns the same eigenpairs.
    """
    N = hii.shape[0]
    dtype = W.dtype
    device = W.device
    eye = torch.eye(N, dtype=dtype, device=device)
    H_dense = torch.stack([matvec(eye[:, k]) for k in range(N)], dim=1)
    basis = None
    if project_fn is not None:
        projector = torch.stack([project_fn(eye[:, k]) for k in range(N)], dim=1)
        p_values, p_vectors = torch.linalg.eigh(projector)
        basis = p_vectors[:, p_values > 0.5]
        if n_roots > basis.shape[1]:
            raise ValueError(
                f"n_roots ({n_roots}) exceeds the projected CI dimension ({basis.shape[1]})"
            )
        H_dense = basis.conj().T @ H_dense @ basis
    E_all, U_all = torch.linalg.eigh(H_dense)
    if basis is not None:
        U_all = basis @ U_all
    vectors = U_all[:, :n_roots].T.contiguous()
    if context is not None and context.world_size > 1:
        # Same reason as the Krylov path: keep every rank's Ritz vectors
        # identical (sign included) under an external torchrun launch.
        _dist().broadcast(vectors, src=0, group=context.group)
    Hvecs = torch.stack([matvec(v) for v in vectors])
    if project_fn is not None:
        Hvecs = torch.stack([project_fn(v) for v in Hvecs])
    energies = E_all[:n_roots]
    residual_norms = (
        torch.linalg.norm(Hvecs - energies.unsqueeze(1) * vectors, dim=1).cpu().tolist()
    )
    max_residual = float(max(residual_norms))
    info = {
        "converged": max_residual < eps,
        "n_outer": 1,
        "n_matvecs": N + n_roots,
        "final_residual_norm": max_residual,
        "final_residual": max_residual,
        "final_residual_norms": residual_norms,
        "history": [(0, N - 1, energies.cpu().tolist(), residual_norms)],
        "distributed": True,
    }
    return _format_davidson_result(energies.cpu().to(torch.float64), vectors, info, n_roots)


def _davidson_distributed_krylov(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    max_iters: int,
    block_size: int,
    eps: float,
    *,
    n_roots: int,
    de_tol: Optional[float],
    project_fn: Optional[Callable[[Tensor], Tensor]],
    context: DistributedContext,
    verbose: bool,
    checkpoint: Optional[DavidsonCheckpoint] = None,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Block Davidson over distributed ``C``/``HC`` blocks."""
    from torch_sci_solver.torch_impl.davidson import _make_seed_block

    N = hii.shape[0]
    dtype = W.dtype
    device = W.device
    bounds = partition_vector(N, context.world_size, context.rank)
    hii_local = hii[bounds.start : bounds.stop]

    # The extra seed columns are chosen by diagonal order; a Hermitian H has a
    # real diagonal, and ``argsort`` has no complex ordering, so order the real
    # part when the caller passed ``hii`` in the Hamiltonian's dtype.
    seeds = _make_seed_block(
        hii.real if hii.is_complex() else hii, W, n_roots, project_fn, dtype, device
    )

    def _wrap(full: Tensor) -> DistributedVector:
        return DistributedVector.from_global(
            full,
            rank=context.rank,
            world_size=context.world_size,
            group=context.group,
            bounds=bounds,
        )

    def _matvec(vector: DistributedVector) -> DistributedVector:
        full = matvec(vector.to_global())
        if project_fn is not None:
            full = project_fn(full)
        if full.shape != (N,):
            raise ValueError(f"matvec returned shape {tuple(full.shape)} != ({N},)")
        return DistributedVector(
            full[bounds.start : bounds.stop].clone(),
            N,
            rank=context.rank,
            world_size=context.world_size,
            group=context.group,
            bounds=bounds,
        )

    def _wrap_local(local: Tensor) -> DistributedVector:
        return DistributedVector(
            local.to(device=device),
            N,
            rank=context.rank,
            world_size=context.world_size,
            group=context.group,
            bounds=bounds,
        )

    n_cycles = max(1, int(max_iters))
    # Each restart cycle can attempt at most block_size expansions.
    n_iters = n_cycles * block_size
    resume = None
    if checkpoint is not None and checkpoint.load is not None:
        loaded = checkpoint.load()
        if loaded is not None:
            resume = loaded["state"]

    if resume is None:
        C: list[DistributedVector] = [_wrap(seeds[i]) for i in range(seeds.shape[0])]
        HC: list[DistributedVector] = [_matvec(vector) for vector in C]
        n_matvecs = len(C)
        n_outer = 1
        start_it = 0
        energies = torch.zeros(n_roots, dtype=torch.float64, device=device)
        max_residual = float("inf")
    else:
        if resume.get("layout") != "krylov":
            raise RuntimeError(
                f"distributed checkpoint layout {resume.get('layout')!r} cannot resume "
                "a replicated Davidson solve"
            )
        if int(resume.get("block_size", -1)) != block_size:
            raise RuntimeError(
                f"distributed checkpoint block_size {resume.get('block_size')} != "
                f"current {block_size}"
            )
        if int(resume.get("n_roots", -1)) != n_roots or int(resume.get("n", -1)) != N:
            raise RuntimeError("distributed checkpoint basis does not match this solve")
        saved_C = resume.get("C_local")
        saved_HC = resume.get("HC_local")
        if (
            not isinstance(saved_C, Tensor)
            or not isinstance(saved_HC, Tensor)
            or saved_C.dim() != 2
            or saved_C.shape != saved_HC.shape
            or saved_C.shape[1] != bounds.local_size
        ):
            raise RuntimeError("distributed checkpoint has inconsistent local blocks")
        d = int(saved_C.shape[0])
        if d < n_roots:
            raise RuntimeError(
                f"distributed checkpoint has {d} subspace vectors, fewer than n_roots={n_roots}"
            )
        C = [_wrap_local(saved_C[i]) for i in range(d)]
        HC = [_wrap_local(saved_HC[i]) for i in range(d)]
        n_matvecs = int(resume.get("n_matvecs", d))
        n_outer = int(resume.get("n_outer", 1))
        start_it = min(max(0, int(resume.get("next_iter", 0))), n_iters)
        energies = resume.get("energies", torch.zeros(n_roots, dtype=torch.float64))
        energies = energies.to(device=device, dtype=torch.float64)
        max_residual = float("inf")
    history: list = list(resume.get("history") or []) if resume is not None else []
    prior_energies: Optional[Tensor] = None
    if resume is not None and resume.get("prior_energies") is not None:
        prior_energies = resume["prior_energies"].to(device=device, dtype=torch.float64)
    ritz_vectors: list[DistributedVector] = []
    ritz_products: list[DistributedVector] = []
    if (
        resume is not None
        and isinstance(resume.get("ritz_vectors"), Tensor)
        and resume["ritz_vectors"].shape[1] == bounds.local_size
    ):
        ritz_vectors = [_wrap_local(block) for block in resume["ritz_vectors"]]
        ritz_products = [_wrap_local(block) for block in resume["ritz_products"]]
    residual_norms: list[float] = []
    if resume is not None:
        residual_norms = [float(value) for value in resume.get("residual_norms") or []]
    if residual_norms:
        max_residual = max(residual_norms)
    if resume is not None and start_it >= n_iters and len(ritz_vectors) < n_roots:
        raise RuntimeError(
            "distributed checkpoint has no Ritz vectors to return for a solve "
            "that has already exhausted its iteration budget"
        )
    converged = False
    # See ``davidson._davidson_multi``: a sub-float64 subspace cannot push its
    # residual below its own precision floor, so stop once the worst root's
    # residual has plateaued instead of burning every remaining restart. The
    # window counts restarts (``n_outer``), not inner steps; fp64 is untouched.
    low_precision = torch.finfo(dtype).bits < 64
    stagnation_window = 15
    stagnation_rtol = 0.9
    best_residual = float("inf")
    best_history: list[float] = []
    stagnated = False

    def checkpoint_state(next_iter: int) -> dict:
        """Rank-local state written at an iteration boundary."""
        return {
            "layout": "krylov",
            "dtype": str(dtype),
            "n": int(N),
            "local_size": int(bounds.local_size),
            "block_size": int(block_size),
            "n_roots": int(n_roots),
            "d": len(C),
            "n_outer": int(n_outer),
            "n_matvecs": int(n_matvecs),
            "next_iter": int(next_iter),
            "C_local": torch.stack([vector.local.detach().cpu() for vector in C]),
            "HC_local": torch.stack([vector.local.detach().cpu() for vector in HC]),
            "energies": energies.detach().cpu().clone(),
            "residual_norms": list(residual_norms),
            "prior_energies": (
                None if prior_energies is None else prior_energies.detach().cpu().clone()
            ),
            "ritz_vectors": (
                torch.stack([vector.local.detach().cpu() for vector in ritz_vectors])
                if ritz_vectors
                else None
            ),
            "ritz_products": (
                torch.stack([vector.local.detach().cpu() for vector in ritz_products])
                if ritz_products
                else None
            ),
            "history": list(history),
        }

    for it in range(start_it, n_iters):
        if (
            checkpoint is not None
            and checkpoint.save is not None
            and it > 0
            and it % checkpoint.every == 0
        ):
            checkpoint.save(it, checkpoint_state(it))
        projection = projected_matrix(C, HC)
        energies, coefficients = torch.linalg.eigh(projection)
        # cuSOLVER returns column-major eigenvectors; NCCL needs contiguous.
        coefficients = coefficients.contiguous()
        # `H_proj` is all-reduced, but LAPACK/cuSOLVER eigenvectors can differ
        # in sign (or last-bit rotation) across ranks. Broadcast rank 0's Ritz
        # problem so every rank forms the *same* Ritz vectors; otherwise the
        # local blocks combined by a later all-gather are inconsistent.
        if context.world_size > 1:
            dist = _dist()
            dist.broadcast(energies, src=0, group=context.group)
            dist.broadcast(coefficients, src=0, group=context.group)
        n_keep = min(len(C), n_roots + 1)
        ritz_vectors = []
        ritz_products = []
        for j in range(n_keep):
            vector = C[0].scale(coefficients[0, j])
            product = HC[0].scale(coefficients[0, j])
            for i in range(1, len(C)):
                vector = vector.add_scaled(C[i], coefficients[i, j])
                product = product.add_scaled(HC[i], coefficients[i, j])
            ritz_vectors.append(vector)
            ritz_products.append(product)

        residual_norms = []
        for j in range(n_roots):
            residual = ritz_products[j].add_scaled(ritz_vectors[j], -energies[j])
            residual_norms.append(float(residual.norm()))
        max_residual = max(residual_norms)
        history.append((it, len(C) - 1, energies[:n_roots].cpu().tolist(), residual_norms))
        de_ok = True
        if de_tol is not None and prior_energies is not None:
            de_ok = float((energies[:n_roots] - prior_energies).abs().max()) < math.sqrt(de_tol)
        if max_residual < eps and de_ok:
            converged = True
            break
        if it == n_iters - 1 or (n_outer == n_cycles and len(C) >= block_size):
            break
        prior_energies = energies[:n_roots].clone()

        worst = max(range(n_roots), key=lambda j: residual_norms[j])
        residual = ritz_products[worst].add_scaled(ritz_vectors[worst], -energies[worst])
        denominator = energies[worst] - hii_local
        tiny = 1e-12
        # ``.real`` keeps the sign guard defined when ``hii_local`` is complex
        # (a Hermitian H has a real diagonal, but the caller may pass it in the
        # Hamiltonian's own dtype); it is the identity on a real tensor.
        denominator = torch.where(
            denominator.abs() < tiny,
            torch.where(denominator.real >= 0, tiny, -tiny).to(denominator.dtype),
            denominator,
        )
        correction = DistributedVector(
            residual.local / denominator,
            N,
            rank=context.rank,
            world_size=context.world_size,
            group=context.group,
            bounds=bounds,
        )
        if project_fn is not None:
            correction = _wrap(project_fn(correction.to_global()))
        for _pass in range(2):
            for vector in C:
                correction = correction.add_scaled(vector, -vector.dot(correction))
        correction_norm = correction.norm()
        if float(correction_norm) < tiny:
            # Deterministic random direction, orthogonalized again. The seed
            # is rank-dependent because each rank only fills its own block;
            # the gathered global vector is well defined either way.
            generator = torch.Generator(device=device)
            generator.manual_seed(0x5EED + it + context.rank * 1_000_003)
            random_local = torch.randn(
                bounds.local_size, generator=generator, dtype=dtype, device=device
            )
            correction = DistributedVector(
                random_local,
                N,
                rank=context.rank,
                world_size=context.world_size,
                group=context.group,
                bounds=bounds,
            )
            if project_fn is not None:
                correction = _wrap(project_fn(correction.to_global()))
            for _pass in range(2):
                for vector in C:
                    correction = correction.add_scaled(vector, -vector.dot(correction))
            correction_norm = correction.norm()
            if float(correction_norm) < tiny:
                continue
        correction = correction.scale(1.0 / correction_norm)
        correction_product = _matvec(correction)
        n_matvecs += 1
        C.append(correction)
        HC.append(correction_product)

        if len(C) >= block_size and n_outer < n_cycles:
            # Thick restart. Keep up to ``n_roots + 1`` Ritz vectors *and* the
            # correction just computed: dropping the correction (the previous
            # behaviour) can make a small block repeat the same step forever
            # (block_size <= n_roots + 2 never converged).
            keep_ritz = max(n_roots, min(n_roots + 1, block_size - 1))
            C = list(ritz_vectors[:keep_ritz]) + [correction]
            HC = list(ritz_products[:keep_ritz]) + [correction_product]
            n_outer += 1
            if low_precision:
                best_residual = min(best_residual, max_residual)
                best_history.append(best_residual)
                if (
                    len(best_history) > stagnation_window
                    and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
                ):
                    stagnated = True
                    break

    if verbose and context.rank == 0:
        print(
            f"  [distributed davidson] iters={len(history)} matvecs={n_matvecs} "
            f"residual={max_residual:.3e} converged={converged}"
        )

    final_energies = energies[:n_roots].cpu().to(torch.float64)
    final_vectors = torch.stack([vector.to_global() for vector in ritz_vectors[:n_roots]])
    info = {
        "converged": converged,
        "n_outer": n_outer,
        "n_matvecs": n_matvecs,
        "final_residual_norm": max_residual,
        "final_residual": max_residual,
        "final_residual_norms": residual_norms,
        "history": history,
        "distributed": True,
    }
    if stagnated:
        info["stagnated"] = True
    return _format_davidson_result(final_energies, final_vectors, info, n_roots)


def _validate_davidson_common(
    n_roots: int, block_size: int, *, restart: bool, context: str
) -> None:
    """Reject a root/block-size combination a Davidson driver cannot honor.

    Args:
        n_roots: number of requested roots.
        block_size: Davidson block size.
        restart: the driver may restart (``N > block_size``).
        context: driver name used in the restart message.

    Raises:
        ValueError: ``n_roots`` outside ``[1, block_size]``, or a restarting
            block that cannot retain the new correction.
    """
    if not 1 <= n_roots <= block_size:
        raise ValueError(f"need 1 <= n_roots ({n_roots}) <= block_size ({block_size})")
    if restart and block_size < n_roots + 1:
        raise ValueError(
            f"{context} needs block_size >= n_roots + 1 so a restart can retain the new correction"
        )


def _check_local_block(x: Tensor, n_local: int, name: str) -> None:
    """Reject a local vector block whose shape is not ``(n_local,)``.

    Args:
        x: the caller's local block.
        n_local: this rank's owned entry count.
        name: variable name used in the error message.

    Raises:
        ValueError: the block does not have the owned shape.
    """
    if x.shape != (n_local,):
        raise ValueError(f"{name} shape {tuple(x.shape)} != ({n_local},)")


def davidson_distributed(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    max_iters: int,
    block_size: int,
    eps: float,
    *,
    n_roots: int = 1,
    de_tol: Optional[float] = None,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    context: DistributedContext,
    verbose: bool = False,
    checkpoint: Optional[DavidsonCheckpoint] = None,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Distributed block Davidson, matching ``davidson_torch``'s interface.

    ``C`` and ``HC`` live as :class:`DistributedVector` blocks. The projection
    ``H_proj = C^H (HC)`` is one fused all-reduce; the ``d x d``
    eigendecomposition is replicated; Ritz vectors are local linear
    combinations; residual norms and orthogonalization coefficients are
    all-reduced (so every rank follows the same control flow). The returned
    eigenvectors are all-gathered to full vectors on every rank, which keeps
    the solver's existing post-processing (occupancies, RDMs, warm-start
    snapshots) unchanged.

    Args:
        hii: ``(N,)`` full Hamiltonian diagonal (identical on every rank). The
            diagonal of a Hermitian ``H`` is real; a complex tensor is accepted
            (its real part orders the extra seeds) so a caller may pass the
            diagonal in the Hamiltonian's own dtype.
        W: ``(N,)`` or ``(k, N)`` initial guess, full on every rank; real or
            complex. With a complex ``W`` every inner product is Hermitian
            (``vdot``), the projection is complex Hermitian and ``eigh``
            returns real Ritz values.
        matvec: full-vector ``v -> H v`` closure (the existing solver matvec).
        max_iters: maximum number of Davidson outer restart cycles.
        block_size: maximum number of subspace vectors kept.
        eps: residual-norm convergence threshold.
        n_roots: number of lowest eigenpairs to return.
        de_tol: opt-in eigenvalue-stability check (effective threshold
            ``sqrt(de_tol)``), matching ``davidson_torch``.
        project_fn: optional full-vector projector for the αβ-swap symmetry.
        context: rank/world-size/device of the initialized process group.
        verbose: print a one-line summary on rank 0.
        checkpoint: optional iteration-boundary persistence hook. It is only
            honored by the distributed Krylov loop; the single-rank and dense
            escapes have no iterations to resume. ``None`` is the
            zero-overhead path.

    Returns:
        ``(float, (N,), info)`` for ``n_roots == 1`` or
        ``((n_roots,), (n_roots, N), info)`` otherwise, exactly like
        ``davidson_torch``. ``info`` carries ``converged``, ``n_matvecs``,
        ``final_residual_norm`` and ``history``.

    Raises:
        ValueError: inconsistent shapes, ``n_roots`` outside
            ``[1, block_size]``, or ``N`` smaller than ``n_roots``.
    """
    if hii.dim() != 1:
        raise ValueError(f"hii must be 1-D, got shape {tuple(hii.shape)}")
    N = hii.shape[0]
    if W.shape[-1] != N:
        raise ValueError(f"W shape {tuple(W.shape)} does not end in N={N}")
    _validate_davidson_common(
        n_roots, block_size, restart=N > block_size, context="distributed Davidson"
    )
    if n_roots > N:
        raise ValueError(f"n_roots ({n_roots}) cannot exceed the CI dimension ({N})")
    if context.world_size <= 1:
        from torch_sci_solver.torch_impl.davidson import davidson_torch

        return davidson_torch(
            hii,
            W,
            matvec,
            max_iters,
            block_size,
            eps,
            n_roots=n_roots,
            de_tol=de_tol,
            project_fn=project_fn,
        )
    if N <= block_size:
        return _davidson_distributed_dense(
            hii,
            W,
            matvec,
            eps,
            n_roots=n_roots,
            project_fn=project_fn,
            context=context,
        )
    return _davidson_distributed_krylov(
        hii,
        W,
        matvec,
        max_iters,
        block_size,
        eps,
        n_roots=n_roots,
        de_tol=de_tol,
        project_fn=project_fn,
        context=context,
        verbose=verbose,
        checkpoint=checkpoint,
    )


# ============================================================================
# Explicit row partition (cost-balanced owner-computes TPB layout)
# ============================================================================


@dataclass(frozen=True)
class RowPartition:
    """Explicit per-rank α-row partition of a ``(n_rows, row_size)`` matrix.

    Unlike :func:`partition_rows`, the row boundaries are stored for every
    rank, so a cost-balanced split (uneven row counts) can be described and
    queried without communication. ``bounds[r].start // row_size`` is rank
    ``r``'s first α-row and ``bounds[r].stop // row_size`` its exclusive end.

    Attributes:
        n_rows: Global number of α-rows.
        row_size: Number of β-columns per row.
        bounds: One contiguous :class:`RankBounds` per rank, ascending,
            covering ``[0, n_rows)``.
    """

    n_rows: int
    row_size: int
    bounds: tuple[RankBounds, ...]

    def __post_init__(self) -> None:
        """Validate that the stored bounds form a contiguous covering."""
        if self.n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {self.n_rows}")
        if self.row_size <= 0:
            raise ValueError(f"row_size must be > 0, got {self.row_size}")
        if not self.bounds:
            raise ValueError("a RowPartition needs at least one rank")
        n = self.n_rows * self.row_size
        previous_stop = 0
        max_local = max(bound.local_size for bound in self.bounds)
        for rank, bound in enumerate(self.bounds):
            if (
                bound.n != n
                or bound.world_size != len(self.bounds)
                or bound.rank != rank
                or bound.n_rows != self.n_rows
                or bound.row_size != self.row_size
                or bound.start != previous_stop
                or bound.stop < bound.start
                or bound.max_local != max_local
            ):
                raise ValueError(
                    f"bounds[{rank}]={bound} is not a contiguous covering of "
                    f"({self.n_rows}, {self.row_size})"
                )
            previous_stop = bound.stop
        if previous_stop != n:
            raise ValueError(f"bounds cover [0, {previous_stop}) instead of [0, {n})")

    @property
    def world_size(self) -> int:
        """Number of ranks the rows are partitioned over."""
        return len(self.bounds)

    @property
    def n(self) -> int:
        """Global flat vector length ``n_rows * row_size``."""
        return self.n_rows * self.row_size

    @property
    def max_local(self) -> int:
        """Largest local flat block over all ranks."""
        return max(bound.local_size for bound in self.bounds)

    def for_rank(self, rank: int) -> RankBounds:
        """Return rank ``rank``'s bounds.

        Raises:
            ValueError: ``rank`` is outside ``[0, world_size)``.
        """
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")
        return self.bounds[rank]

    def rows_for_rank(self, rank: int) -> tuple[int, int]:
        """Return ``(start_row, stop_row)`` owned by ``rank``."""
        bound = self.for_rank(rank)
        return bound.start // self.row_size, bound.stop // self.row_size

    def owner_rows(self, rows: Tensor) -> Tensor:
        """Map global α-row indices to their owning rank (int64 tensor).

        Raises:
            ValueError: any row is outside ``[0, n_rows)``.
        """
        index = torch.as_tensor(rows, dtype=torch.int64)
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= self.n_rows):
            raise ValueError(f"rows outside [0, {self.n_rows})")
        starts = torch.tensor(
            [bound.start // self.row_size for bound in self.bounds], dtype=torch.int64
        )
        # right=True keeps the last rank among equal starts, so empty ranks
        # (duplicate boundaries) never claim a row.
        owner = torch.searchsorted(starts, index.cpu(), right=True) - 1
        return owner.clamp_(0, self.world_size - 1).to(index.device)

    def global_row(self, rank: int, local_row: Tensor) -> Tensor:
        """Map rank-local row indices to global α-row indices."""
        start, stop = self.rows_for_rank(rank)
        index = torch.as_tensor(local_row, dtype=torch.int64)
        if index.numel() and (int(index.min()) < 0 or int(index.max()) >= stop - start):
            raise ValueError(f"local rows outside [0, {stop - start}) for rank {rank}")
        return index + start


def partition_rows_explicit(
    n_rows: int, row_size: int, start_rows: Sequence[int], world_size: int
) -> RowPartition:
    """Build a :class:`RowPartition` from explicit row boundaries.

    Args:
        n_rows: Global number of α-rows.
        row_size: Number of β-columns per row.
        start_rows: ``world_size + 1`` boundaries; ``start_rows[r]`` is rank
            ``r``'s first row, ``start_rows[0] == 0`` and the last entry is
            ``n_rows``.
        world_size: Number of ranks.

    Returns:
        The explicit partition.

    Raises:
        ValueError: wrong number of boundaries, out of range, not monotone,
            or not covering ``[0, n_rows)``.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    if row_size <= 0:
        raise ValueError(f"row_size must be > 0, got {row_size}")
    starts = [int(value) for value in start_rows]
    if len(starts) != world_size + 1:
        raise ValueError(
            f"start_rows needs world_size + 1 = {world_size + 1} entries, got {len(starts)}"
        )
    if starts[0] != 0 or starts[-1] != n_rows:
        raise ValueError(
            f"start_rows must run from 0 to n_rows={n_rows}, got {starts[0]}..{starts[-1]}"
        )
    if any(right < left for left, right in zip(starts[:-1], starts[1:])):
        raise ValueError(f"start_rows must be non-decreasing, got {starts}")
    max_rows = max(right - left for left, right in zip(starts[:-1], starts[1:]))
    max_local = max_rows * row_size
    bounds = tuple(
        RankBounds(
            n=n_rows * row_size,
            world_size=world_size,
            rank=rank,
            start=starts[rank] * row_size,
            stop=starts[rank + 1] * row_size,
            max_local=max_local,
            n_rows=n_rows,
            row_size=row_size,
        )
        for rank in range(world_size)
    )
    return RowPartition(n_rows=int(n_rows), row_size=int(row_size), bounds=bounds)


def partition_rows_weighted(
    n_rows: int,
    row_size: int,
    weights: Tensor,
    world_size: int,
    *,
    max_imbalance: float = 1.25,
) -> RowPartition:
    """Cost-balanced row partition with a per-rank row cap.

    Rows are cut on the prefix sum of the (replicated) non-negative per-row
    cost ``weights`` so every rank gets about ``1 / world_size`` of the total
    cost, but never more than ``ceil(max_imbalance * n_rows / world_size)``
    rows (memory floor) and never fewer than zero. The result is deterministic
    and identical on every rank.

    Args:
        n_rows: Global number of α-rows.
        row_size: Number of β-columns per row.
        weights: ``(n_rows,)`` non-negative per-row cost, replicated.
        world_size: Number of ranks.
        max_imbalance: Rows per rank may exceed the mean by at most this
            factor (must be >= 1).

    Returns:
        The cost-balanced explicit partition.

    Raises:
        ValueError: invalid sizes, weights shape/dtype, negative weights, or
            ``max_imbalance < 1``.

    Notes
    -----
    With prefix costs :math:`P_b = \\sum_{j<b} w_j` and total :math:`W`, rank
    ``r`` starts at the first row whose prefix reaches its fair share of the
    remaining cost, clamped to ``[n - c (R - r - 1), prev + c]`` with
    ``c = \\lceil 1.25 n / R \\rceil``. A pure-Python walk is used because the
    result must be bit-identical on every rank and ``n_rows`` is bounded
    (``<= 1e5`` in the owner-computes layout).
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    if row_size <= 0:
        raise ValueError(f"row_size must be > 0, got {row_size}")
    starts = _weighted_boundaries(n_rows, weights, world_size, max_imbalance)
    return partition_rows_explicit(n_rows, row_size, starts, world_size)


def _weighted_boundaries(n: int, weights: Tensor, parts: int, max_imbalance: float) -> list[int]:
    """Cut ``[0, n)`` into ``parts`` cost-balanced contiguous ranges.

    Deterministic and identical on every rank: the walk uses CPU float64
    prefix sums only. Shared by the row and column axes of the 2-D tile
    partition, so both cuts follow the same policy.

    Args:
        n: Number of rows (or columns) to split.
        weights: ``(n,)`` non-negative per-row cost, replicated.
        parts: Number of ranges.
        max_imbalance: A range may exceed the mean row count by at most this
            factor (must be >= 1).

    Returns:
        ``parts + 1`` boundaries ``[0 = s_0, s_1, ..., s_parts = n]``.

    Raises:
        ValueError: invalid ``parts``/``max_imbalance``, or bad weights.
    """
    if parts <= 0:
        raise ValueError(f"parts must be > 0, got {parts}")
    if max_imbalance < 1.0:
        raise ValueError(f"max_imbalance must be >= 1, got {max_imbalance}")
    cost = torch.as_tensor(weights)
    if cost.shape != (n,):
        raise ValueError(f"weights shape {tuple(cost.shape)} != ({n},)")
    if cost.is_complex() or cost.dtype.is_floating_point and not torch.isfinite(cost).all():
        raise ValueError("weights must be finite real costs")
    cost = cost.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if n and bool((cost < 0).any()):
        raise ValueError("weights must be non-negative")

    cap = int(math.ceil(max_imbalance * n / parts))
    prefix = torch.cumsum(cost, dim=0)
    starts = [0]
    position = 0
    for part in range(parts - 1):
        remaining_parts = parts - part
        position_cost = float(prefix[position - 1]) if position > 0 else 0.0
        total_cost = float(prefix[-1]) if n else 0.0
        target = position_cost + (total_cost - position_cost) / remaining_parts
        # First boundary whose prefix reaches the fair share of the cost.
        candidate = int(torch.searchsorted(prefix, torch.tensor(target, dtype=torch.float64)))
        # Reserve enough rows so every later part stays under the row cap.
        lower = max(position, n - cap * (remaining_parts - 1))
        upper = min(position + cap, n)
        position = min(max(candidate, lower), upper)
        starts.append(position)
    starts.append(n)
    return starts


def _even_boundaries(n: int, parts: int) -> list[int]:
    """Cut ``[0, n)`` into ``parts`` even contiguous ranges (remainder first)."""
    if parts <= 0:
        raise ValueError(f"parts must be > 0, got {parts}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [(part * n) // parts for part in range(parts)] + [n]


# ============================================================================
# 2-D tile partition (alpha-row slab x beta-column slab)
# ============================================================================


def _check_ranges(bounds: Sequence[tuple[int, int]], n: int, axis: str) -> None:
    """Validate contiguous, ascending ranges that cover ``[0, n)``."""
    previous = 0
    for start, stop in bounds:
        if start != previous or stop < start:
            raise ValueError(f"{axis} bounds {tuple(bounds)} do not tile [0, {n}) contiguously")
        previous = stop
    if previous != n:
        raise ValueError(f"{axis} bounds cover [0, {previous}) instead of [0, {n})")


@dataclass(frozen=True)
class TileGrid:
    """Rank grid over an ``(n_rows, n_cols)`` matrix: ``p_alpha x p_beta``.

    Rank ``r = i * p_beta + j`` owns rows ``alpha_bounds[i]`` and columns
    ``beta_bounds[j]``. Both boundary tuples are contiguous, ascending and
    cover the full axis; empty slabs (a slab count larger than the axis
    length) are allowed so every world size is representable.

    Attributes:
        n_rows: Global number of α-rows.
        n_cols: Global number of β-columns.
        p_alpha: Number of row slabs.
        p_beta: Number of column slabs.
        alpha_bounds: ``p_alpha`` ``(start, stop)`` row ranges.
        beta_bounds: ``p_beta`` ``(start, stop)`` column ranges.
    """

    n_rows: int
    n_cols: int
    p_alpha: int
    p_beta: int
    alpha_bounds: tuple[tuple[int, int], ...]
    beta_bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        """Validate the grid shape and that both boundary lists tile their axis."""
        if self.n_rows < 0:
            raise ValueError(f"n_rows must be >= 0, got {self.n_rows}")
        if self.n_cols < 0:
            raise ValueError(f"n_cols must be >= 0, got {self.n_cols}")
        if self.p_alpha <= 0 or self.p_beta <= 0:
            raise ValueError(f"p_alpha and p_beta must be > 0, got {self.p_alpha} x {self.p_beta}")
        if len(self.alpha_bounds) != self.p_alpha:
            raise ValueError(
                f"alpha_bounds has {len(self.alpha_bounds)} entries, expected {self.p_alpha}"
            )
        if len(self.beta_bounds) != self.p_beta:
            raise ValueError(
                f"beta_bounds has {len(self.beta_bounds)} entries, expected {self.p_beta}"
            )
        _check_ranges(self.alpha_bounds, self.n_rows, "alpha")
        _check_ranges(self.beta_bounds, self.n_cols, "beta")

    @property
    def world_size(self) -> int:
        """Number of ranks, ``p_alpha * p_beta``."""
        return self.p_alpha * self.p_beta

    def _check_rank(self, rank: int) -> int:
        """Return ``rank`` after checking it is on the grid."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")
        return rank

    def row_slab(self, rank: int) -> int:
        """Row slab of ``rank`` (``rank // p_beta``)."""
        return self._check_rank(rank) // self.p_beta

    def col_slab(self, rank: int) -> int:
        """Column slab of ``rank`` (``rank % p_beta``)."""
        return self._check_rank(rank) % self.p_beta

    def rank_of(self, row_slab: int, col_slab: int) -> int:
        """Rank owning ``(row_slab, col_slab)`` (``row_slab * p_beta + col_slab``)."""
        if not 0 <= row_slab < self.p_alpha:
            raise ValueError(f"row_slab must be in [0, {self.p_alpha}), got {row_slab}")
        if not 0 <= col_slab < self.p_beta:
            raise ValueError(f"col_slab must be in [0, {self.p_beta}), got {col_slab}")
        return row_slab * self.p_beta + col_slab

    def rows(self, row_slab: int) -> tuple[int, int]:
        """``(start, stop)`` rows of ``row_slab``."""
        if not 0 <= row_slab < self.p_alpha:
            raise ValueError(f"row_slab must be in [0, {self.p_alpha}), got {row_slab}")
        return self.alpha_bounds[row_slab]

    def cols(self, col_slab: int) -> tuple[int, int]:
        """``(start, stop)`` columns of ``col_slab``."""
        if not 0 <= col_slab < self.p_beta:
            raise ValueError(f"col_slab must be in [0, {self.p_beta}), got {col_slab}")
        return self.beta_bounds[col_slab]


@dataclass(frozen=True)
class TileBounds:
    """Rectangular block of an ``(n_rows, n_cols)`` matrix owned by one rank.

    The local flat layout is row-major inside the tile:
    ``idx = (ia - row_start) * local_cols + (ib - col_start)``; the global
    flat index is ``k = ia * n_cols + ib``.

    Attributes:
        n_rows: Global α-row count.
        n_cols: Global β-column count.
        world_size: Number of ranks.
        rank: This tile's rank.
        row_slab: Index of the owned row slab.
        col_slab: Index of the owned column slab.
        row_start: First owned row (inclusive).
        row_stop: One past the last owned row (exclusive).
        col_start: First owned column (inclusive).
        col_stop: One past the last owned column (exclusive).
        max_rows: Largest tile row count over all ranks.
        max_cols: Largest tile column count over all ranks.
    """

    n_rows: int
    n_cols: int
    world_size: int
    rank: int
    row_slab: int
    col_slab: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    max_rows: int
    max_cols: int

    @property
    def local_rows(self) -> int:
        """Number of owned rows."""
        return self.row_stop - self.row_start

    @property
    def local_cols(self) -> int:
        """Number of owned columns."""
        return self.col_stop - self.col_start

    @property
    def local_size(self) -> int:
        """Number of owned determinants, ``local_rows * local_cols``."""
        return self.local_rows * self.local_cols

    @property
    def n(self) -> int:
        """Global flat vector length ``n_rows * n_cols``."""
        return self.n_rows * self.n_cols

    @property
    def is_empty(self) -> bool:
        """Whether this rank owns no determinant."""
        return self.local_size == 0

    def owns(self, index: int) -> bool:
        """Whether the global flat ``index`` lies in this tile."""
        row, col = divmod(index, self.n_cols)
        return self.row_start <= row < self.row_stop and self.col_start <= col < self.col_stop

    def local_to_global(self, index: Tensor) -> Tensor:
        """Map tile-local flat indices to global flat indices."""
        idx = torch.as_tensor(index, dtype=torch.int64)
        if self.local_size == 0:
            return idx.new_empty(0)
        rows = torch.div(idx, self.local_cols, rounding_mode="floor") + self.row_start
        cols = idx % self.local_cols + self.col_start
        return rows * self.n_cols + cols

    def global_to_local(self, index: Tensor) -> Tensor:
        """Map global flat indices inside this tile to tile-local indices.

        The caller is responsible for passing owned indices (see
        :meth:`owns`); the arithmetic is not masked.
        """
        idx = torch.as_tensor(index, dtype=torch.int64)
        rows = torch.div(idx, self.n_cols, rounding_mode="floor") - self.row_start
        cols = idx % self.n_cols - self.col_start
        return rows * self.local_cols + cols


@dataclass(frozen=True)
class TilePartition:
    """Explicit 2-D tile partition of an ``(n_rows, n_cols)`` matrix.

    Attributes:
        n_rows: Global α-row count.
        n_cols: Global β-column count.
        grid: The rank grid the tile bounds were derived from.
        bounds: One :class:`TileBounds` per rank, in rank order.
    """

    n_rows: int
    n_cols: int
    grid: TileGrid
    bounds: tuple[TileBounds, ...]

    def __post_init__(self) -> None:
        """Validate that every bound matches its slab of the grid."""
        if self.grid.n_rows != self.n_rows or self.grid.n_cols != self.n_cols:
            raise ValueError(
                f"grid covers ({self.grid.n_rows}, {self.grid.n_cols}) but the partition "
                f"is ({self.n_rows}, {self.n_cols})"
            )
        if len(self.bounds) != self.grid.world_size:
            raise ValueError(f"expected {self.grid.world_size} bounds, got {len(self.bounds)}")
        max_rows = max(bound.local_rows for bound in self.bounds)
        max_cols = max(bound.local_cols for bound in self.bounds)
        for rank, bound in enumerate(self.bounds):
            row_slab, col_slab = self.grid.row_slab(rank), self.grid.col_slab(rank)
            row_start, row_stop = self.grid.rows(row_slab)
            col_start, col_stop = self.grid.cols(col_slab)
            if (
                bound.n_rows != self.n_rows
                or bound.n_cols != self.n_cols
                or bound.world_size != self.grid.world_size
                or bound.rank != rank
                or bound.row_slab != row_slab
                or bound.col_slab != col_slab
                or bound.row_start != row_start
                or bound.row_stop != row_stop
                or bound.col_start != col_start
                or bound.col_stop != col_stop
                or bound.max_rows != max_rows
                or bound.max_cols != max_cols
            ):
                raise ValueError(f"bounds[{rank}]={bound} does not match the grid")
        covered = sum(bound.local_size for bound in self.bounds)
        if covered != self.n:
            raise ValueError(f"tiles cover {covered} determinants instead of {self.n}")

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return self.grid.world_size

    @property
    def n(self) -> int:
        """Global flat vector length ``n_rows * n_cols``."""
        return self.n_rows * self.n_cols

    @property
    def max_local(self) -> int:
        """Largest local tile over all ranks (buffer sizing)."""
        return max(bound.local_size for bound in self.bounds)

    def for_rank(self, rank: int) -> TileBounds:
        """Return rank ``rank``'s tile bounds."""
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {rank}")
        return self.bounds[rank]

    def rows_for_rank(self, rank: int) -> tuple[int, int]:
        """Return ``(start_row, stop_row)`` owned by ``rank``."""
        bound = self.for_rank(rank)
        return bound.row_start, bound.row_stop

    def cols_for_rank(self, rank: int) -> tuple[int, int]:
        """Return ``(start_col, stop_col)`` owned by ``rank``."""
        bound = self.for_rank(rank)
        return bound.col_start, bound.col_stop

    def _owner_slab(self, index: Tensor, bounds: Sequence[tuple[int, int]]) -> Tensor:
        """Map global axis indices to their slab id (int64 tensor)."""
        idx = torch.as_tensor(index, dtype=torch.int64)
        starts = torch.tensor([start for start, _ in bounds], dtype=torch.int64)
        # right=True keeps the last slab among duplicate starts, so empty
        # slabs never claim an index (same convention as RowPartition.owner_rows).
        owner = torch.searchsorted(starts, idx.cpu(), right=True) - 1
        return owner.clamp_(0, len(bounds) - 1).to(idx.device)

    def owner_row_slab(self, rows: Tensor) -> Tensor:
        """Map global α-row indices to their owning row slab."""
        return self._owner_slab(rows, self.grid.alpha_bounds)

    def owner_col_slab(self, cols: Tensor) -> Tensor:
        """Map global β-column indices to their owning column slab."""
        return self._owner_slab(cols, self.grid.beta_bounds)

    @classmethod
    def from_boundaries(
        cls,
        n_rows: int,
        n_cols: int,
        alpha_starts: Sequence[int],
        beta_starts: Sequence[int],
    ) -> "TilePartition":
        """Build a partition from the two boundary arrays.

        Args:
            n_rows: Global α-row count.
            n_cols: Global β-column count.
            alpha_starts: ``p_alpha + 1`` row boundaries from ``0`` to ``n_rows``.
            beta_starts: ``p_beta + 1`` column boundaries from ``0`` to ``n_cols``.

        Returns:
            The explicit tile partition.

        Raises:
            ValueError: boundaries that are out of range, not monotone, or do
                not cover their axis.
        """
        row_starts = [int(value) for value in alpha_starts]
        col_starts = [int(value) for value in beta_starts]
        _check_boundaries(row_starts, n_rows, "alpha")
        _check_boundaries(col_starts, n_cols, "beta")
        alpha_bounds = tuple(zip(row_starts[:-1], row_starts[1:]))
        beta_bounds = tuple(zip(col_starts[:-1], col_starts[1:]))
        grid = TileGrid(
            n_rows=int(n_rows),
            n_cols=int(n_cols),
            p_alpha=len(alpha_bounds),
            p_beta=len(beta_bounds),
            alpha_bounds=alpha_bounds,
            beta_bounds=beta_bounds,
        )
        max_rows = max(stop - start for start, stop in alpha_bounds)
        max_cols = max(stop - start for start, stop in beta_bounds)
        bounds = tuple(
            TileBounds(
                n_rows=int(n_rows),
                n_cols=int(n_cols),
                world_size=grid.world_size,
                rank=rank,
                row_slab=grid.row_slab(rank),
                col_slab=grid.col_slab(rank),
                row_start=grid.rows(grid.row_slab(rank))[0],
                row_stop=grid.rows(grid.row_slab(rank))[1],
                col_start=grid.cols(grid.col_slab(rank))[0],
                col_stop=grid.cols(grid.col_slab(rank))[1],
                max_rows=max_rows,
                max_cols=max_cols,
            )
            for rank in range(grid.world_size)
        )
        return cls(n_rows=int(n_rows), n_cols=int(n_cols), grid=grid, bounds=bounds)

    @classmethod
    def from_row_partition(cls, partition: RowPartition) -> "TilePartition":
        """Degenerate a 1-D :class:`RowPartition` to a ``p_beta == 1`` tile grid.

        Every rank keeps all columns, so the tile matvec reproduces the
        legacy owner-computes exchange pattern; this is the compatibility
        path for the existing flat layouts.
        """
        row_starts = [partition.rows_for_rank(rank)[0] for rank in range(partition.world_size)]
        row_starts.append(partition.n_rows)
        return cls.from_boundaries(
            partition.n_rows, partition.row_size, row_starts, [0, partition.row_size]
        )


def _check_boundaries(starts: Sequence[int], n: int, axis: str) -> None:
    """Validate a boundary array: sorted, in range, covering ``[0, n)``."""
    if len(starts) < 2:
        raise ValueError(f"{axis}_starts needs at least two entries, got {len(starts)}")
    if starts[0] != 0 or starts[-1] != n:
        raise ValueError(f"{axis}_starts must run from 0 to {n}, got {starts[0]}..{starts[-1]}")
    if any(right < left for left, right in zip(starts[:-1], starts[1:])):
        raise ValueError(f"{axis}_starts must be non-decreasing, got {tuple(starts)}")


def _table_candidates(n_orb: int, n_particles: int) -> int:
    """Candidate single-plus-double excitations of one spin string."""
    n_empty = int(n_orb) - int(n_particles)
    singles = int(n_particles) * n_empty
    doubles = 0
    if n_particles >= 2 and n_empty >= 2:
        doubles = (int(n_particles) * (int(n_particles) - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    return singles + doubles


def _device_memory_budget(device: torch.device) -> Optional[int]:
    """Usable device memory in bytes from the current free pool, or ``None`` off CUDA.

    The auto-tiling policies run after this rank has already allocated the
    replicated ERIs, basis and helper arrays, so a fraction of *total* memory
    would over-estimate what a new layout can still use. Ten percent of the
    free and of the total are both kept as headroom for transient allocations;
    a card whose live tensors already exceed that headroom is therefore not
    over-estimated.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return int(min(0.9 * free_bytes, 0.9 * total_bytes))


def resolve_auto_tile_grid(
    *,
    world_size: int,
    n_rows: int,
    n_cols: int,
    n_particles_a: int,
    n_particles_b: int,
    n_orb: int,
    budget_bytes: Optional[int],
    itemsize: int = 8,
    block_size: int = 8,
) -> Optional[tuple[int, int]]:
    """Best 2-D grid for the basis, or ``None`` when the 1-D split is better.

    Tiling buys memory, not speed: it removes the replicated α/β tables at the
    cost of the slab and corner exchanges (about a tenth of the runtime on one
    node). It is therefore chosen only when the replicated table term is a
    first-order part of the rank footprint, or when the 1-D footprint would
    not fit in the device budget at all.

    Args:
        world_size: Number of ranks.
        n_rows: Global α-row count.
        n_cols: Global β-column count.
        n_particles_a: α electron count.
        n_particles_b: β electron count.
        n_orb: Number of spatial orbitals.
        budget_bytes: Usable device memory per rank, or ``None`` when unknown
            (CPU runs); with no budget the faster 1-D split is kept.
        itemsize: Bytes per Davidson vector element.
        block_size: Nominal Davidson block size for the block-memory estimate.

    Returns:
        ``(p_alpha, p_beta)`` to tile, or ``None`` for the 1-D row split.

    Notes
    -----
    The 1-D owner-computes path builds α tables for its owned rows only, so
    the replicated 1-D term is the per-rank α slice plus the full β tables
    and a β-pass cache of the same order as the β tables:

    .. math::

        B_{1D} \\approx 3 \\cdot 8 \\left(\\lceil M_a/P \\rceil c_a
        + 2 M_b c_b\\right),

    where :math:`c_s` is :func:`_table_candidates` for spin ``s``. That bound
    overestimates the measured survivors, increasingly so at small ``M``,
    which biases the decision towards tiling (the memory-safe direction); on
    a 60-orbital benchmark basis it is about 1.5x high at ``M = 10^5`` and
    about 16x high at ``M = 10^4``. The Davidson block is modelled as
    :math:`(b + 9) N/P` elements. Tiling is chosen when the 1-D worst case
    would use more than half of the device budget; below that the 1-D split
    fits with headroom and keeps its ~10% faster matvec. A grid with a single
    column slab is rejected: it keeps the full β tables and the same α rows
    as the 1-D split, so it only adds exchange buffers.
    """
    if world_size <= 1 or n_rows == 0 or n_cols == 0:
        return None
    if budget_bytes is None:
        return None
    grid = resolve_tile_grid(world_size, n_rows, n_cols)
    if grid[1] == 1:
        # (P, 1) is the 1-D row split plus staged exchanges: no replicated
        # table shrinks, so it can never be the memory-better choice.
        return None
    c_a = _table_candidates(n_orb, n_particles_a)
    c_b = _table_candidates(n_orb, n_particles_b)
    rows_per_rank = -(-int(n_rows) // int(world_size))
    one_d = 3 * 8 * (rows_per_rank * c_a + 2 * int(n_cols) * c_b)
    block = (int(block_size) + 9) * (int(n_rows) * int(n_cols) // int(world_size)) * int(itemsize)
    if one_d + block <= int(0.5 * budget_bytes):
        return None
    return grid


def resolve_tile_grid(world_size: int, n_rows: int, n_cols: int) -> tuple[int, int]:
    """Pick the ``(p_alpha, p_beta)`` grid for an ``(n_rows, n_cols)`` basis.

    Minimises the largest per-rank *slab* ``max(n_rows/p_alpha, n_cols/p_beta)``
    (the two exchange volumes) among divisor pairs of ``world_size``, avoiding
    slabs with no rows or columns unless no divisor pair allows it. Ties prefer
    the pair whose row count is closest to ``sqrt(P * n_rows / n_cols)``, which
    equalises the two exchange volumes.

    Args:
        world_size: Number of ranks.
        n_rows: Global α-row count.
        n_cols: Global β-column count.

    Returns:
        ``(p_alpha, p_beta)`` with ``p_alpha * p_beta == world_size``.

    Raises:
        ValueError: non-positive ``world_size`` or negative axis lengths.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"n_rows/n_cols must be >= 0, got ({n_rows}, {n_cols})")
    rows = max(n_rows, 1)
    cols = max(n_cols, 1)
    ideal = math.sqrt(world_size * rows / cols)
    best: Optional[tuple[tuple[float, float, float], int, int]] = None
    for p_alpha in range(1, world_size + 1):
        if world_size % p_alpha:
            continue
        p_beta = world_size // p_alpha
        empty = 1.0 if p_alpha > rows or p_beta > cols else 0.0
        score = max(rows / p_alpha, cols / p_beta)
        key = (empty, score, abs(p_alpha - ideal))
        if best is None or key < best[0]:
            best = (key, p_alpha, p_beta)
    assert best is not None
    return best[1], best[2]


def partition_tiles(
    n_rows: int,
    n_cols: int,
    world_size: int,
    grid: Optional[tuple[int, int]] = None,
    *,
    row_weights: Optional[Tensor] = None,
    col_weights: Optional[Tensor] = None,
    max_imbalance: float = 1.25,
) -> TilePartition:
    """Build a 2-D tile partition, evenly or cost-balanced per axis.

    Args:
        n_rows: Global α-row count.
        n_cols: Global β-column count.
        world_size: Number of ranks.
        grid: Optional explicit ``(p_alpha, p_beta)``; defaults to
            :func:`resolve_tile_grid`.
        row_weights: Optional ``(n_rows,)`` per-row cost for a weighted cut.
        col_weights: Optional ``(n_cols,)`` per-column cost for a weighted cut.
        max_imbalance: Per-slab count cap factor for the weighted cuts.

    Returns:
        The tile partition.

    Raises:
        ValueError: invalid sizes, grid product, or weights.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if n_rows < 0 or n_cols < 0:
        raise ValueError(f"n_rows/n_cols must be >= 0, got ({n_rows}, {n_cols})")
    p_alpha, p_beta = grid if grid is not None else resolve_tile_grid(world_size, n_rows, n_cols)
    if p_alpha <= 0 or p_beta <= 0 or p_alpha * p_beta != world_size:
        raise ValueError(
            f"grid ({p_alpha}, {p_beta}) must have positive entries multiplying to "
            f"world_size={world_size}"
        )
    alpha_starts = (
        _weighted_boundaries(n_rows, row_weights, p_alpha, max_imbalance)
        if row_weights is not None
        else _even_boundaries(n_rows, p_alpha)
    )
    beta_starts = (
        _weighted_boundaries(n_cols, col_weights, p_beta, max_imbalance)
        if col_weights is not None
        else _even_boundaries(n_cols, p_beta)
    )
    return TilePartition.from_boundaries(n_rows, n_cols, alpha_starts, beta_starts)


# ============================================================================
# Staged pairwise row exchange (replaces the full-length halo)
# ============================================================================


class StagedRowExchange:
    """Cyclic staged peer exchange of the α rows a rank needs.

    Stage 0 is the local block. Stage ``s`` in ``[1, world_size)`` holds, in
    ascending global-row order, exactly the rows owned by rank
    ``(rank + s) % world_size`` that this rank needs. The generator returned
    by :meth:`stages` posts stage ``s + 1``'s send/recv before yielding stage
    ``s`` (double-buffered), so communication overlaps the stage ``s``
    kernels. Send payloads are kept alive until their wait.

    The setup exchanges each rank's needed row list with one
    ``all_gather_object``; each stage then issues one batched
    ``isend``/``irecv`` pair per rank.

    Complex rows are supported: P2P of a complex tensor is not portable
    (NCCL has no complex scalar type), so a complex payload is packed as its
    interleaved ``(real, imag)`` pairs, received into a real buffer and
    exposed as an ``(n_rows, row_size)`` complex view. Real payloads are sent
    and received exactly as before.
    """

    def __init__(
        self,
        partition: RowPartition,
        needed_rows: Tensor,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
    ) -> None:
        """Compute the per-stage send/receive row lists.

        Args:
            partition: Row layout of the coefficient matrix.
            needed_rows: Global α-row indices this rank needs remotely (owned
                rows are ignored). May contain duplicates.
            rank: Override the rank from the process group.
            world_size: Override the world size.
            group: Process group used by every collective and P2P op.

        Raises:
            ValueError: rows out of range or rank/world-size inconsistent
                with ``partition``.
        """
        self.partition = partition
        self.group = group
        self.rank, self.world_size = _resolve_rank_world(rank, world_size, group)
        if self.world_size != partition.world_size:
            raise ValueError(
                f"world_size={self.world_size} disagrees with the partition's "
                f"{partition.world_size}"
            )
        self.row_size = partition.row_size
        self.n_rows = partition.n_rows
        self._local_start, self._local_stop = partition.rows_for_rank(self.rank)

        needed = torch.as_tensor(needed_rows, dtype=torch.int64).reshape(-1).cpu()
        if needed.numel() and (int(needed.min()) < 0 or int(needed.max()) >= self.n_rows):
            raise ValueError(f"needed_rows outside [0, {self.n_rows})")
        needed = torch.unique(needed, sorted=True)
        needed = needed[(needed < self._local_start) | (needed >= self._local_stop)]

        all_needs: list[Optional[list[int]]] = [None] * self.world_size
        if self.world_size > 1:
            dist = _dist()
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "a multi-rank StagedRowExchange needs an initialized process group"
                )
            dist.all_gather_object(all_needs, needed.tolist(), group=group)
        else:
            all_needs[self.rank] = needed.tolist()

        my_rows = set(range(self._local_start, self._local_stop))
        self._send_rows: dict[int, Tensor] = {}
        self._recv_rows: dict[int, Tensor] = {}
        for stage in range(1, self.world_size):
            receiver = (self.rank + stage) % self.world_size
            peer_start, peer_stop = partition.rows_for_rank(receiver)
            recv = [int(row) for row in needed.tolist() if peer_start <= row < peer_stop]
            sender = (self.rank - stage) % self.world_size
            send = sorted(my_rows & set(all_needs[sender] or []))
            self._recv_rows[stage] = torch.tensor(recv, dtype=torch.int64)
            self._send_rows[stage] = torch.tensor(send, dtype=torch.int64)
        self._max_recv_rows = max((rows.numel() for rows in self._recv_rows.values()), default=0)
        self._buffers: Optional[list[Tensor]] = None
        self._buffer_key: Optional[tuple[torch.dtype, torch.device]] = None

    @property
    def n_stages(self) -> int:
        """Number of stages, including the local stage 0."""
        return self.world_size

    def rows(self, stage: int) -> Tensor:
        """Sorted global rows yielded at ``stage`` (stage 0: the local range)."""
        if not 0 <= stage < self.world_size:
            raise ValueError(f"stage must be in [0, {self.world_size}), got {stage}")
        if stage == 0:
            return torch.arange(self._local_start, self._local_stop, dtype=torch.int64)
        return self._recv_rows[stage].clone()

    def _ensure_buffers(self, reference: Tensor) -> None:
        """Allocate the two receive buffers once for this dtype/device.

        Complex payloads are received as interleaved ``(real, imag)`` pairs in
        a real buffer of width ``2 * row_size`` (complex P2P is not portable:
        NCCL has no complex scalar type); :meth:`stages` restores the complex
        dtype. The real path allocates exactly the same buffer as before.
        """
        key = (reference.dtype, reference.device)
        if self._buffers is None or self._buffer_key != key:
            if reference.dtype.is_complex:
                width = 2 * self.row_size
                dtype = reference.dtype.to_real()
            else:
                width = self.row_size
                dtype = reference.dtype
            self._buffers = [
                torch.empty(
                    (self._max_recv_rows, width),
                    dtype=dtype,
                    device=reference.device,
                )
                for _ in range(2)
            ]
            self._buffer_key = key

    def _post_stage(
        self, stage: int, x_local_2d: Tensor
    ) -> tuple[list[Any], dict[int, Tensor], dict[int, tuple[Tensor, int]]]:
        """Issue stage ``stage``'s batched send/recv; return requests and payloads."""
        dist = _dist()
        ops: list[Any] = []
        sends: dict[int, Tensor] = {}
        recvs: dict[int, tuple[Tensor, int]] = {}
        send_rows = self._send_rows[stage]
        if send_rows.numel():
            index = (send_rows - self._local_start).to(x_local_2d.device)
            payload = x_local_2d[index].reshape(-1).contiguous()
            if payload.is_complex():
                # Send the (real, imag) pairs as plain reals; the peer unpacks
                # them into its real receive buffer the same way.
                payload = torch.view_as_real(payload).reshape(-1)
            sends[stage] = payload
            ops.append(
                dist.P2POp(
                    dist.isend,
                    payload,
                    (self.rank - stage) % self.world_size,
                    group=self.group,
                )
            )
        n_recv = int(self._recv_rows[stage].numel())
        if n_recv:
            buffer = self._buffers[stage % 2]  # type: ignore[index]
            recvs[stage] = (buffer, n_recv)
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    buffer[:n_recv].reshape(-1),
                    (self.rank + stage) % self.world_size,
                    group=self.group,
                )
            )
        requests = dist.batch_isend_irecv(ops) if ops else []
        return requests, sends, recvs

    def stages(self, x_local_2d: Tensor) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(stage, buffer)`` for stage ``0 .. world_size - 1``.

        Stage 0 yields ``x_local_2d`` itself, shape
        ``(local_rows, row_size)``. Every later stage yields an
        ``(rows(stage).numel(), row_size)`` view of the double buffer.

        Args:
            x_local_2d: This rank's local rows, ``(local_rows, row_size)``.

        Raises:
            ValueError: ``x_local_2d`` has the wrong shape.
        """
        local_rows = self._local_stop - self._local_start
        if x_local_2d.dim() != 2 or x_local_2d.shape != (local_rows, self.row_size):
            raise ValueError(
                f"x_local_2d shape {tuple(x_local_2d.shape)} != ({local_rows}, {self.row_size})"
            )
        if self.world_size == 1:
            yield 0, x_local_2d
            return
        self._ensure_buffers(x_local_2d)
        pending: dict[int, tuple[list[Any], dict[int, Tensor], dict[int, tuple[Tensor, int]]]] = {}
        pending[1] = self._post_stage(1, x_local_2d)
        try:
            yield 0, x_local_2d
            for stage in range(1, self.world_size):
                requests, _sends, recvs = pending.pop(stage)
                for request in requests:
                    request.wait()
                # Post the next stage before yielding this one so it overlaps
                # the caller's kernels; its parity differs from this stage's.
                if stage + 1 < self.world_size:
                    pending[stage + 1] = self._post_stage(stage + 1, x_local_2d)
                if stage in recvs:
                    buffer, n_recv = recvs[stage]
                    if x_local_2d.dtype.is_complex:
                        # The real receive buffer holds interleaved (re, im).
                        yield (
                            stage,
                            torch.view_as_complex(buffer[:n_recv].view(n_recv, self.row_size, 2)),
                        )
                    else:
                        yield stage, buffer[:n_recv]
                else:
                    yield stage, x_local_2d.new_empty((0, self.row_size))
        finally:
            # Early close still owns posted requests; draining them keeps the
            # process group consistent for the caller's next collective.
            for requests, _sends, _recvs in pending.values():
                for request in requests:
                    request.wait()


class StagedSlabExchange:
    """Cyclic staged exchange of the α-row or β-column slabs a tile needs.

    The α stream runs along a tile's column group: rank ``(i, j)`` receives
    from ``((i + s) % p_alpha, j)`` the rows of that peer's tile that its own
    α tables read, keeping the full local column slab, so a stage buffer is
    ``(rows, local_cols)``. The β stream mirrors it along the row group and
    keeps the full local row slab, so its buffer is
    ``(local_rows, cols)``.

    Stage 0 is the caller's own tile; stage ``s > 0`` also ships to its peer
    ``(rank - s)`` whatever that peer reports needing of this tile. Both
    needed-index lists are exchanged once, in the constructor, with one
    ``all_gather_object``; the matvec then posts one batched
    ``isend``/``irecv`` pair per stage, double-buffered so stage ``s + 1`` is
    in flight while the kernels of stage ``s`` run.

    Buffers are flat so every stage's block is a contiguous prefix: a
    received block is exposed as its ``(rows, cols)`` view, whatever its
    exact shape. Complex tiles are packed as their interleaved ``(re, im)``
    real view (NCCL has no complex scalar type) and received into a real
    buffer.
    """

    def __init__(
        self,
        partition: TilePartition,
        axis: str,
        needed: Tensor,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        tag: int = 0,
    ) -> None:
        """Compute the per-stage send and receive index lists.

        Args:
            partition: The 2-D tile partition.
            axis: ``"alpha"`` (rows, column-group peers) or ``"beta"``
                (columns, row-group peers).
            needed: Global indices along ``axis`` this rank needs remotely
                (its own slab is ignored). May contain duplicates.
            rank: Override the rank from the process group.
            world_size: Override the world size.
            group: Process group used by every collective and P2P op.
            tag: Base P2P tag; stage ``s`` uses ``tag + s``.

        Raises:
            ValueError: bad axis, indices out of range, or a rank/world size
                inconsistent with ``partition``.
        """
        if axis not in ("alpha", "beta"):
            raise ValueError(f"axis must be 'alpha' or 'beta', got {axis!r}")
        self.partition = partition
        self.axis = axis
        self.group = group
        self._tag = int(tag)
        self.rank, self.world_size = _resolve_rank_world(rank, world_size, group)
        if self.world_size != partition.world_size:
            raise ValueError(
                f"world_size={self.world_size} disagrees with the partition's "
                f"{partition.world_size}"
            )
        self._grid = partition.grid
        self._bound = partition.for_rank(self.rank)
        self._height = self._bound.local_rows
        self._width = self._bound.local_cols
        if axis == "alpha":
            self._group_pos = self._bound.row_slab
            self._group_size = self._grid.p_alpha
            self._local_start, self._local_stop = self._bound.row_start, self._bound.row_stop
            self._n_axis = self._grid.n_rows
            self._axis_bounds = self._grid.alpha_bounds
        else:
            self._group_pos = self._bound.col_slab
            self._group_size = self._grid.p_beta
            self._local_start, self._local_stop = self._bound.col_start, self._bound.col_stop
            self._n_axis = self._grid.n_cols
            self._axis_bounds = self._grid.beta_bounds

        needed = torch.as_tensor(needed, dtype=torch.int64).reshape(-1).cpu()
        if needed.numel() and (int(needed.min()) < 0 or int(needed.max()) >= self._n_axis):
            raise ValueError(f"needed indices outside [0, {self._n_axis})")
        needed = torch.unique(needed, sorted=True)
        needed = needed[(needed < self._local_start) | (needed >= self._local_stop)]
        self.needed = needed

        all_needs: list[Optional[list[int]]] = [None] * self.world_size
        if self.world_size > 1:
            dist = _dist()
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "a multi-rank StagedSlabExchange needs an initialized process group"
                )
            dist.all_gather_object(all_needs, needed.tolist(), group=group)
        else:
            all_needs[self.rank] = needed.tolist()
        self.all_needed = [torch.tensor(rows, dtype=torch.int64) for rows in all_needs]

        self._recv_index: dict[int, Tensor] = {}
        self._send_index: dict[int, Tensor] = {}
        for stage in range(1, self._group_size):
            recv_pos = (self._group_pos + stage) % self._group_size
            peer_start, peer_stop = self._axis_bounds[recv_pos]
            lo = int(torch.searchsorted(needed, peer_start))
            hi = int(torch.searchsorted(needed, peer_stop))
            self._recv_index[stage] = needed[lo:hi].clone()
            send_pos = (self._group_pos - stage) % self._group_size
            self._send_index[stage] = self.send_positions(self._rank_at(send_pos))
        self._max_recv = max((int(idx.numel()) for idx in self._recv_index.values()), default=0)
        self._buffers: Optional[list[Tensor]] = None
        self._buffer_key: Optional[tuple[torch.dtype, torch.device]] = None

    def _rank_at(self, group_pos: int) -> int:
        """Rank owning ``group_pos`` on this axis, at this rank's other slab."""
        if self.axis == "alpha":
            return self._grid.rank_of(group_pos, self._bound.col_slab)
        return self._grid.rank_of(self._bound.row_slab, group_pos)

    def send_positions(self, peer_rank: int) -> Tensor:
        """Positions in this tile (along ``axis``) that ``peer_rank`` needs."""
        needs = self.all_needed[peer_rank]
        mine = needs[(needs >= self._local_start) & (needs < self._local_stop)]
        return mine - self._local_start

    def stage_indices(self, stage: int) -> Tensor:
        """Sorted global indices this stage's buffer holds (stage 0: local)."""
        if not 0 <= stage < self._group_size:
            raise ValueError(f"stage must be in [0, {self._group_size}), got {stage}")
        if stage == 0:
            return torch.arange(self._local_start, self._local_stop, dtype=torch.int64)
        return self._recv_index[stage].clone()

    def _block_shape(self, n_recv: int) -> tuple[int, int]:
        """Shape of a stage's received block."""
        return (n_recv, self._width) if self.axis == "alpha" else (self._height, n_recv)

    def _span(self) -> int:
        """Elements per received index: the block's other-axis length."""
        return self._width if self.axis == "alpha" else self._height

    def _ensure_buffers(self, reference: Tensor) -> None:
        """Allocate the two flat receive buffers for this dtype/device."""
        key = (reference.dtype, reference.device)
        if self._buffers is None or self._buffer_key != key:
            count = self._max_recv * self._span()
            if reference.dtype.is_complex:
                count *= 2
                dtype = reference.dtype.to_real()
            else:
                dtype = reference.dtype
            self._buffers = [
                torch.empty(count, dtype=dtype, device=reference.device) for _ in range(2)
            ]
            self._buffer_key = key

    def _decode(self, recv: Optional[Tensor], n_recv: int, reference: Tensor) -> Tensor:
        """Expose a received flat block as its ``(rows, cols)`` view."""
        shape = self._block_shape(n_recv)
        if recv is None:
            return reference.new_empty(shape)
        flat = torch.view_as_complex(recv.view(-1, 2)) if reference.dtype.is_complex else recv
        return flat.view(shape)

    def _post_stage(
        self, stage: int, x_tile: Tensor
    ) -> tuple[list[Any], dict[int, Tensor], Optional[Tensor], int]:
        """Issue stage ``stage``'s send and receive; return them with the count."""
        dist = _dist()
        ops: list[Any] = []
        sends: dict[int, Tensor] = {}
        recv: Optional[Tensor] = None
        recv_pos = (self._group_pos + stage) % self._group_size
        send_pos = (self._group_pos - stage) % self._group_size
        send_index = self._send_index[stage]
        n_recv = int(self._recv_index[stage].numel())
        if send_index.numel():
            index = send_index.to(x_tile.device)
            block = x_tile[index] if self.axis == "alpha" else x_tile[:, index]
            payload = block.contiguous()
            if payload.is_complex():
                payload = torch.view_as_real(payload).reshape(-1)
            else:
                payload = payload.reshape(-1)
            sends[stage] = payload
            ops.append(
                dist.P2POp(
                    dist.isend,
                    payload,
                    self._rank_at(send_pos),
                    group=self.group,
                    tag=self._tag + stage,
                )
            )
        if n_recv:
            count = n_recv * self._span()
            if self._buffer_key is not None and self._buffer_key[0].is_complex:
                count *= 2
            recv = self._buffers[stage % 2][:count]  # type: ignore[index]
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv,
                    self._rank_at(recv_pos),
                    group=self.group,
                    tag=self._tag + stage,
                )
            )
        requests = dist.batch_isend_irecv(ops) if ops else []
        return requests, sends, recv, n_recv

    def stages(self, x_tile: Tensor) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(stage, buffer)`` for stage ``0 .. group_size - 1``.

        Stage 0 yields ``x_tile`` itself; every later stage yields an
        ``(rows, local_cols)`` (α) or ``(local_rows, cols)`` (β) view of the
        double buffer.

        Args:
            x_tile: This rank's tile, ``(local_rows, local_cols)``.

        Raises:
            ValueError: ``x_tile`` has the wrong shape.
        """
        if x_tile.dim() != 2 or x_tile.shape != (self._height, self._width):
            raise ValueError(
                f"x_tile shape {tuple(x_tile.shape)} != ({self._height}, {self._width})"
            )
        if self._group_size == 1:
            yield 0, x_tile
            return
        self._ensure_buffers(x_tile)
        pending: dict[int, tuple[list[Any], dict[int, Tensor], Optional[Tensor], int]] = {}
        pending[1] = self._post_stage(1, x_tile)
        try:
            yield 0, x_tile
            for stage in range(1, self._group_size):
                requests, _sends, recv, n_recv = pending.pop(stage)
                for request in requests:
                    request.wait()
                # Post the next stage before yielding this one so it overlaps
                # the caller's kernels; its parity differs from this stage's.
                if stage + 1 < self._group_size:
                    pending[stage + 1] = self._post_stage(stage + 1, x_tile)
                yield stage, self._decode(recv, n_recv, x_tile)
        finally:
            # Early close still owns posted requests; draining them keeps the
            # process group consistent for the caller's next collective.
            for requests, _sends, _recv, _n in pending.values():
                for request in requests:
                    request.wait()


def _exchange_bands(
    n_rows: int, n_cols: int, max_elems: Optional[int]
) -> tuple[tuple[int, int], ...]:
    """Row bands of one staged rectangle under an element budget.

    The split depends only on the rectangle's shape, so the sender and the
    receiver of a stage derive the same bands from the sizes each of them
    knows; that is what keeps their sub-stage tags paired.
    """
    if n_rows == 0 or n_cols == 0:
        return ((0, n_rows),)
    if max_elems is None or n_rows * n_cols <= int(max_elems):
        return ((0, n_rows),)
    rows_per = max(1, int(max_elems) // n_cols)
    return tuple((lo, min(lo + rows_per, n_rows)) for lo in range(0, n_rows, rows_per))


class StagedCornerExchange:
    """Cyclic staged exchange of the αβ corner rectangles.

    Stage ``(s_a, s_b)`` (with ``s_a, s_b > 0``) partners the tile
    ``((i + s_a) % p_alpha, (j + s_b) % p_beta)``. For a destination tile the
    source rectangle is ``needed_alpha[i] ∩ A_src  ×  needed_beta[j] ∩ B_src``
    — contiguous slices of the two sorted needed lists — so both the outgoing
    slice of the local tile and the incoming shape follow from the lists the
    :class:`StagedSlabExchange` peers already broadcast, with no metadata at
    matvec time. The ``s_a == 0`` and ``s_b == 0`` corners are sub-blocks of
    the β and α stage buffers and are never shipped twice.

    Each pair index covers two independent messages: the receive rectangle
    from ``(i + s_a, j + s_b)`` and the send rectangle to
    ``(i - s_a, j - s_b)``. Their shapes differ in general (the needed
    counts in differently sized peer slabs), so each is banded from its own
    shape and the plan holds the union of the two band lists; a band present
    on only one of the two messages leaves the other side's slice empty. The sender and the
    receiver of one directed message do compute the same shape — the
    receiver's needed slice and the sender's send positions are the same set
    — and therefore the same bands. To pair their P2P operations without a
    running index (which a peer cannot reproduce) each sub-stage's tag
    addresses ``(pair, band)`` directly, as ``tag + n_pairs * band + pair``.

    Rectangles are packed flat, so a received block is exposed as its
    ``(rows, cols)`` view. Complex tiles are packed as their interleaved
    ``(re, im)`` real view.
    """

    def __init__(
        self,
        alpha: StagedSlabExchange,
        beta: StagedSlabExchange,
        *,
        tag: int = 0,
        max_elems: Optional[int] = None,
    ) -> None:
        """Compute the per-pair rectangle shapes from the two slab exchanges.

        Args:
            alpha: The α-axis slab exchange (row slabs, column-group peers).
            beta: The β-axis slab exchange (column slabs, row-group peers).
            tag: Base P2P tag; sub-stage ``(pair, band)`` uses
                ``tag + n_pairs * band + pair``.
            max_elems: Optional element budget for one directed rectangle. A
                rectangle larger than this is split into row bands, each shipped
                as its own sub-stage, so a single P2P buffer stays bounded.

        Raises:
            ValueError: the two exchanges disagree on the partition or rank.
        """
        if alpha.partition is not beta.partition or alpha.rank != beta.rank:
            raise ValueError("alpha and beta exchanges must share a partition and rank")
        self.partition = alpha.partition
        self.rank = alpha.rank
        self.world_size = alpha.world_size
        self.group = alpha.group
        self._tag = int(tag)
        self._grid = alpha.partition.grid
        bound = alpha.partition.for_rank(self.rank)
        self._i, self._j = bound.row_slab, bound.col_slab
        self._pairs = tuple(
            (s_a, s_b)
            for s_a in range(1, self._grid.p_alpha)
            for s_b in range(1, self._grid.p_beta)
        )
        recv_rows: dict[int, Tensor] = {}
        recv_cols: dict[int, Tensor] = {}
        send_rows: dict[int, Tensor] = {}
        send_cols: dict[int, Tensor] = {}
        for stage, (s_a, s_b) in enumerate(self._pairs):
            recv_rows[stage] = alpha.stage_indices(s_a)
            recv_cols[stage] = beta.stage_indices(s_b)
            send_rank = self._send_rank(s_a, s_b)
            send_rows[stage] = alpha.send_positions(send_rank)
            send_cols[stage] = beta.send_positions(send_rank)
        # Each pair index is two messages (receive from the ``+`` partner,
        # send to the ``-`` partner). Band them from their own shapes and emit
        # the union of the two band lists: on an uneven grid the two shapes
        # differ, while the two ends of one directed message agree on its
        # shape and so on its bands. The tag addresses ``(pair, band)`` rather
        # than the plan position, because the plan positions before a given
        # pair depend on the *other* message's band count, which the peer
        # cannot see; with the tag below both ends pair every band of every
        # directed message. An unchunked exchange has one band per message and
        # keeps the historical ``tag + pair`` numbering.
        self._plan: list[tuple[int, int, int, int]] = []
        self._recv_rows: dict[int, Tensor] = {}
        self._recv_cols: dict[int, Tensor] = {}
        self._send_rows: dict[int, Tensor] = {}
        self._send_cols: dict[int, Tensor] = {}
        self._tags: list[int] = []
        n_pairs = len(self._pairs)
        for pair, (s_a, s_b) in enumerate(self._pairs):
            recv_bands = _exchange_bands(
                int(recv_rows[pair].numel()), int(recv_cols[pair].numel()), max_elems
            )
            send_bands = _exchange_bands(
                int(send_rows[pair].numel()), int(send_cols[pair].numel()), max_elems
            )
            for band in range(max(len(recv_bands), len(send_bands))):
                row_lo, row_hi = recv_bands[band] if band < len(recv_bands) else (0, 0)
                send_lo, send_hi = send_bands[band] if band < len(send_bands) else (0, 0)
                sub = len(self._plan)
                self._plan.append((s_a, s_b, row_lo, row_hi))
                self._tags.append(self._tag + n_pairs * band + pair)
                self._recv_rows[sub] = recv_rows[pair][row_lo:row_hi]
                self._recv_cols[sub] = recv_cols[pair]
                self._send_rows[sub] = send_rows[pair][send_lo:send_hi]
                self._send_cols[sub] = send_cols[pair]
        self._max_elems = max(
            (
                int(self._recv_rows[sub].numel()) * int(self._recv_cols[sub].numel())
                for sub in range(len(self._plan))
            ),
            default=0,
        )
        self._buffers: Optional[list[Tensor]] = None
        self._buffer_key: Optional[tuple[torch.dtype, torch.device]] = None

    def _recv_rank(self, s_a: int, s_b: int) -> int:
        """Rank whose tile the ``(s_a, s_b)`` stage receives from."""
        return self._grid.rank_of(
            (self._i + s_a) % self._grid.p_alpha, (self._j + s_b) % self._grid.p_beta
        )

    def _send_rank(self, s_a: int, s_b: int) -> int:
        """Rank the ``(s_a, s_b)`` stage sends to."""
        return self._grid.rank_of(
            (self._i - s_a) % self._grid.p_alpha, (self._j - s_b) % self._grid.p_beta
        )

    def pairs(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(s_a, s_b, row_lo, row_hi)`` of every sub-stage, in stage order.

        ``row_lo``/``row_hi`` are positions inside the ``(s_a, s_b)`` receive
        rectangle; an unchunked stage is the single band ``(0, n_rows)``. A
        band that exists only on the outgoing message (its index exceeds the
        incoming message's band count) has ``row_lo == row_hi == 0`` and
        yields an empty buffer.
        """
        return tuple(self._plan)

    def _ensure_buffers(self, reference: Tensor) -> None:
        """Allocate the two flat receive buffers for this dtype/device."""
        key = (reference.dtype, reference.device)
        if self._buffers is None or self._buffer_key != key:
            count = self._max_elems * (2 if reference.dtype.is_complex else 1)
            dtype = reference.dtype.to_real() if reference.dtype.is_complex else reference.dtype
            self._buffers = [
                torch.empty(count, dtype=dtype, device=reference.device) for _ in range(2)
            ]
            self._buffer_key = key

    def _post_stage(
        self, stage: int, x_tile: Tensor
    ) -> tuple[list[Any], dict[int, Tensor], Optional[Tensor], tuple[int, int]]:
        """Issue one stage's send and receive; return them with the recv shape."""
        dist = _dist()
        s_a, s_b, _row_lo, _row_hi = self._plan[stage]
        tag = self._tags[stage]
        ops: list[Any] = []
        sends: dict[int, Tensor] = {}
        recv: Optional[Tensor] = None
        shape = (int(self._recv_rows[stage].numel()), int(self._recv_cols[stage].numel()))
        send_rows = self._send_rows[stage]
        send_cols = self._send_cols[stage]
        if send_rows.numel() and send_cols.numel():
            block = x_tile[send_rows.to(x_tile.device)][:, send_cols.to(x_tile.device)]
            payload = block.contiguous()
            if payload.is_complex():
                payload = torch.view_as_real(payload).reshape(-1)
            else:
                payload = payload.reshape(-1)
            sends[stage] = payload
            ops.append(
                dist.P2POp(
                    dist.isend,
                    payload,
                    self._send_rank(s_a, s_b),
                    group=self.group,
                    tag=tag,
                )
            )
        if shape[0] and shape[1]:
            count = shape[0] * shape[1]
            if self._buffer_key is not None and self._buffer_key[0].is_complex:
                count *= 2
            recv = self._buffers[stage % 2][:count]  # type: ignore[index]
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv,
                    self._recv_rank(s_a, s_b),
                    group=self.group,
                    tag=tag,
                )
            )
        requests = dist.batch_isend_irecv(ops) if ops else []
        return requests, sends, recv, shape

    def stages(self, x_tile: Tensor) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(stage, buffer)`` for every remote corner pair.

        Args:
            x_tile: This rank's tile, ``(local_rows, local_cols)``.

        Raises:
            ValueError: ``x_tile`` has the wrong shape.
        """
        bound = self.partition.for_rank(self.rank)
        if x_tile.dim() != 2 or x_tile.shape != (bound.local_rows, bound.local_cols):
            raise ValueError(
                f"x_tile shape {tuple(x_tile.shape)} != ({bound.local_rows}, {bound.local_cols})"
            )
        if not self._pairs:
            return
        self._ensure_buffers(x_tile)
        pending: dict[
            int, tuple[list[Any], dict[int, Tensor], Optional[Tensor], tuple[int, int]]
        ] = {}
        pending[0] = self._post_stage(0, x_tile)
        try:
            for stage in range(len(self._plan)):
                requests, _sends, recv, shape = pending.pop(stage)
                for request in requests:
                    request.wait()
                if stage + 1 < len(self._plan):
                    pending[stage + 1] = self._post_stage(stage + 1, x_tile)
                if recv is None:
                    yield stage, x_tile.new_empty(shape)
                    continue
                flat = torch.view_as_complex(recv.view(-1, 2)) if x_tile.dtype.is_complex else recv
                yield stage, flat.view(shape)
        finally:
            for requests, _sends, _recv, _shape in pending.values():
                for request in requests:
                    request.wait()


# ============================================================================
# Partition-invariant seed noise and global diagonal minima
# ============================================================================


def _unsigned_shift(value: Tensor, bits: int) -> Tensor:
    """Logical right shift on a signed int64 tensor (torch's ``>>`` is arithmetic)."""
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def hash_uniform(indices: Tensor, *, salt: int = 0) -> Tensor:
    """Partition-invariant pseudo-random values in ``[-1, 1)`` from global indices.

    Counter-based (splitmix64-style) hash, deterministic across ranks,
    devices and partitions. Used for the cold-start and extra-root seed noise,
    which must not depend on how the global vector is split.

    Args:
        indices: Integer tensor of global indices (any shape/device).
        salt: Integer stream selector; independent salts give independent
            value streams for the same indices.

    Returns:
        A float64 tensor in ``[-1, 1)`` with the shape/device of ``indices``.

    Notes
    -----
    splitmix64 mixes the counter with two 64-bit multiplications,

    .. math::

        z \\leftarrow (z \\oplus (z \\gg 30)) \\cdot c_1, \\quad
        z \\leftarrow (z \\oplus (z \\gg 27)) \\cdot c_2, \\quad
        z \\leftarrow z \\oplus (z \\gg 31),

    with :math:`c_1 = 0x\\mathrm{BF58476D1CE4E5B9}` and
    :math:`c_2 = 0x\\mathrm{94D049BB133111EB}`. The unsigned 53-bit mantissa
    ``(z >> 11)`` maps to ``2 * 2^-53 * bits - 1``. Torch int64 multiplication
    wraps, but ``>>`` is arithmetic, so every shift is masked to emulate the
    unsigned shift.
    """
    index = torch.as_tensor(indices)
    if index.is_floating_point() or index.is_complex():
        raise TypeError("hash_uniform takes integer global indices")
    z = index.to(torch.int64) + _signed_int64(salt)
    z = z + _signed_int64(0x9E3779B97F4A7C15)
    z = (z ^ _unsigned_shift(z, 30)) * _signed_int64(0xBF58476D1CE4E5B9)
    z = (z ^ _unsigned_shift(z, 27)) * _signed_int64(0x94D049BB133111EB)
    z = z ^ _unsigned_shift(z, 31)
    bits = _unsigned_shift(z, 11).to(torch.float64)
    return bits * (2.0**-52) - 1.0


@dataclass(frozen=True)
class _RowIndexLayout:
    """Flat contiguous-block layout: wraps a one-column :class:`RowPartition`.

    The local Davidson sees flat ``(N_local,)`` arrays; this view exposes the
    global determinant index of each local entry for a block that is the
    contiguous flat range ``[bound.start, bound.stop)``.
    """

    partition: RowPartition

    @property
    def n(self) -> int:
        """Global determinant count."""
        return self.partition.n

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return self.partition.world_size

    def local_size(self, rank: int) -> int:
        """Number of entries ``rank`` owns."""
        return self.partition.for_rank(rank).local_size

    def local_to_global(self, rank: int, index: Tensor) -> Tensor:
        """Map local flat indices of ``rank`` to global flat indices."""
        bound = self.partition.for_rank(rank)
        idx = torch.as_tensor(index, dtype=torch.int64)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= bound.local_size):
            raise ValueError(f"local indices outside [0, {bound.local_size})")
        return idx + bound.start

    def global_indices(self, rank: int) -> Tensor:
        """Global flat indices of ``rank``'s block, in local order."""
        bound = self.partition.for_rank(rank)
        return torch.arange(bound.start, bound.stop, dtype=torch.int64)


@dataclass(frozen=True)
class _TileIndexLayout:
    """2-D tile layout: wraps this rank's :class:`TileBounds`.

    Only the owning rank's tile is known here, which is all the seed,
    one-hot and missed-root helpers touch; the global 2-D mapping itself
    comes from :class:`TileBounds`.
    """

    bounds: TileBounds

    @property
    def n(self) -> int:
        """Global determinant count."""
        return self.bounds.n

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return self.bounds.world_size

    def _check_rank(self, rank: int) -> None:
        """Reject ranks other than the tile owner."""
        if rank != self.bounds.rank:
            raise ValueError(f"this layout holds rank {self.bounds.rank}'s tile, not rank {rank}")

    def local_size(self, rank: int) -> int:
        """Number of entries ``rank`` owns."""
        self._check_rank(rank)
        return self.bounds.local_size

    def local_to_global(self, rank: int, index: Tensor) -> Tensor:
        """Map tile-local flat indices of ``rank`` to global flat indices."""
        self._check_rank(rank)
        return self.bounds.local_to_global(index)

    def global_indices(self, rank: int) -> Tensor:
        """Global flat indices of the tile, in local (row-major) order."""
        self._check_rank(rank)
        local = torch.arange(self.bounds.local_size, dtype=torch.int64)
        return self.bounds.local_to_global(local)


@dataclass(frozen=True)
class _PairTileIndexLayout:
    """Scattered pair-tile layout: wraps a GDB tile bounds object.

    A GDB tile owns a sorted, *scattered* subset ``pair_index`` of the flat
    pair vector (one rectangle per basis block), so a local position maps to
    ``pair_index[position]`` instead of an affine offset. The bounds object is
    duck-typed — it must expose ``n``, ``local_size``, ``pair_index``, ``rank``
    and ``world_size`` — because the pair-tile module imports
    :class:`TilePartition` from this module and importing it back would create
    a cycle. Only the owning rank's index list is needed here.
    """

    bounds: Any

    @property
    def n(self) -> int:
        """Global pair count."""
        return int(self.bounds.n)

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return int(self.bounds.world_size)

    def _check_rank(self, rank: int) -> None:
        """Reject ranks other than the tile owner."""
        if rank != self.bounds.rank:
            raise ValueError(f"this layout holds rank {self.bounds.rank}'s tile, not rank {rank}")

    def local_size(self, rank: int) -> int:
        """Number of pairs ``rank`` owns."""
        self._check_rank(rank)
        return int(self.bounds.local_size)

    def local_to_global(self, rank: int, index: Tensor) -> Tensor:
        """Map tile-local positions of ``rank`` to global pair indices."""
        self._check_rank(rank)
        pair_index = self.bounds.pair_index
        idx = torch.as_tensor(index, dtype=torch.int64)
        if idx.numel():
            if int(idx.min()) < 0 or int(idx.max()) >= int(pair_index.numel()):
                raise ValueError(f"local indices outside [0, {int(pair_index.numel())})")
            idx = idx.to(pair_index.device)
        return pair_index[idx].detach().cpu()

    def global_indices(self, rank: int) -> Tensor:
        """Global pair indices of the tile, ascending (``pair_index`` order)."""
        self._check_rank(rank)
        return self.bounds.pair_index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)


_LocalIndexLayout = Union[_RowIndexLayout, _TileIndexLayout, _PairTileIndexLayout]


def _index_layout(
    n: int,
    bounds: Union[RankBounds, TileBounds, Any],
    context: DistributedContext,
) -> _LocalIndexLayout:
    """Build the global-index layout for this rank's local block.

    A :class:`RankBounds` reconstructs the flat covering of ``[0, n)`` from
    every rank's block (one gather), so one-hot seeds land on the right
    owner; a :class:`TileBounds` carries its own 2-D mapping and needs no
    communication; a GDB pair-tile bounds object carries its scattered
    ``pair_index`` list, which is already sorted. The last case is duck-typed
    to avoid importing the pair-tile module back (it imports
    :class:`TilePartition` from here).
    """
    if isinstance(bounds, TileBounds):
        if bounds.n != n or bounds.world_size != context.world_size or bounds.rank != context.rank:
            raise ValueError("`bounds` is inconsistent with the distributed context")
        return _TileIndexLayout(bounds)
    if isinstance(bounds, RankBounds):
        return _RowIndexLayout(_flat_row_partition(n, bounds, context))
    if getattr(bounds, "pair_index", None) is None:
        raise TypeError(
            "`bounds` must be a RankBounds, a TileBounds or a GDB pair-tile bounds "
            "object exposing `pair_index`"
        )
    if bounds.n != n or bounds.world_size != context.world_size or bounds.rank != context.rank:
        raise ValueError("`bounds` is inconsistent with the distributed context")
    return _PairTileIndexLayout(bounds)


def global_smallest_hii(
    hii_local: Tensor,
    layout: Union[RowPartition, _LocalIndexLayout],
    *,
    k: int,
    group: Optional[Any] = None,
) -> list[tuple[int, float]]:
    """The ``k`` globally smallest ``(index, value)`` pairs, replicated.

    Each rank takes its local ``topk`` (the global ``k`` smallest cannot
    miss a candidate this way), ``all_gather_object`` collects the
    candidates, and every rank sorts by ``(value, index)`` so ties break by
    global determinant index. Used by the local Davidson's multi-root seed
    block and the missed-root search.

    Args:
        hii_local: This rank's diagonal block, ``(local_size,)``.
        layout: Index layout the block belongs to: a :class:`RowPartition`
            (wrapped as a flat layout for backwards compatibility) or a
            :class:`_RowIndexLayout` / :class:`_TileIndexLayout`.
        k: Number of pairs to return (clamped to the global dimension).
        group: Process group used by the gather.

    Returns:
        Up to ``k`` ``(global_index, value)`` pairs, sorted ascending.

    Raises:
        ValueError: ``k <= 0`` or the local block has the wrong shape.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if isinstance(layout, RowPartition):
        layout = _RowIndexLayout(layout)
    rank, world_size = _resolve_rank_world(None, layout.world_size, group)
    local_size = layout.local_size(rank)
    if hii_local.shape != (local_size,):
        raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({local_size},)")
    take = min(k, local_size)
    if take == 0:
        values = torch.empty(0, dtype=torch.float64)
        rows = torch.empty(0, dtype=torch.int64)
    else:
        values, indices = torch.topk(hii_local.reshape(-1), take, largest=False, sorted=True)
        rows = layout.local_to_global(rank, indices.cpu())
        values = values.detach().cpu().to(torch.float64)
    local_pairs = list(zip(rows.tolist(), values.tolist()))
    all_pairs: list[Optional[list[tuple[int, float]]]] = [None] * world_size
    if world_size > 1:
        dist = _dist()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("a multi-rank global_smallest_hii needs a process group")
        dist.all_gather_object(all_pairs, local_pairs, group=group)
    else:
        all_pairs[rank] = local_pairs
    candidates = [pair for chunk in all_pairs if chunk is not None for pair in chunk]
    candidates.sort(key=lambda pair: (pair[1], pair[0]))
    return candidates[: min(k, layout.n)]


# ============================================================================
# Local-block Davidson internals
# ============================================================================


def _make_all_reduce(context: DistributedContext) -> Callable[[Tensor], Tensor]:
    """Return the in-place all-reduce closure used by the local Davidson."""
    dist = _dist()
    if context.world_size > 1 and not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "davidson_distributed_local with world_size > 1 needs an initialized process group"
        )

    def reduce(tensor: Tensor) -> Tensor:
        if context.world_size <= 1:
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=context.group)
        return tensor

    return reduce


def _reduced_dot(left: Tensor, right: Tensor, reduce: Callable[[Tensor], Tensor]) -> Tensor:
    """Global Hermitian inner product ``<left|right>`` as a 0-dim tensor.

    ``torch.vdot`` conjugates its first argument, which is the identity for
    real dtypes (it is bit-identical to ``torch.dot`` there).
    """
    value = torch.vdot(left.reshape(-1), right.reshape(-1))
    return reduce(value.reshape(1))[0]


def _reduced_norm(vector: Tensor, reduce: Callable[[Tensor], Tensor]) -> Tensor:
    """Global Euclidean norm as a 0-dim real tensor."""
    return torch.sqrt(_reduced_dot(vector, vector, reduce).real.clamp_min(0.0))


def _cgs2(vector: Tensor, basis: Tensor, reduce: Callable[[Tensor], Tensor]) -> Tensor:
    """Two-pass block classical Gram-Schmidt against ``basis`` rows.

    Each pass is one all-reduce of the ``(d,)`` coefficient vector, matching
    ``_davidson_multi``'s unconditional two passes.
    The coefficients are the Hermitian inner products ``<basis_i|vector>``;
    ``basis.conj()`` is a no-op for real dtypes.
    """
    for _pass in range(2):
        coefficients = basis.conj() @ vector
        reduce(coefficients)
        vector = vector - coefficients @ basis
    return vector


def _eigh_broadcast(projection: Tensor, context: DistributedContext) -> tuple[Tensor, Tensor]:
    """Diagonalize in fp64 and broadcast rank 0's eigenpairs.

    The projection is at most ``block_size`` wide, so the upcast is cheap; it
    also keeps fp32 subspaces from failing LAPACK's convergence checks (the
    same fallback ``_davidson_multi`` uses). A complex Hermitian projection
    is upcast to ``complex128``; ``eigh`` then returns real eigenvalues and
    complex eigenvectors, and both are broadcast. Broadcasting rank 0's result
    makes every rank form identical Ritz vectors even when cuSOLVER/LAPACK
    pick different signs or last-bit rotations.
    """
    work = projection.to(torch.complex128 if projection.is_complex() else torch.float64)
    energies, coefficients = torch.linalg.eigh(work)
    coefficients = coefficients.contiguous()
    if context.world_size > 1:
        dist = _dist()
        dist.broadcast(energies, src=0, group=context.group)
        dist.broadcast(coefficients, src=0, group=context.group)
    return energies, coefficients


def _flat_row_partition(n: int, bounds: RankBounds, context: DistributedContext) -> RowPartition:
    """One-column :class:`RowPartition` over global determinant indices.

    The local Davidson works on flat ``(N_local,)`` blocks but the seed block
    and missed-root search index determinants globally. Gathering every
    rank's ``(start, stop)`` once reconstructs any layout — flat, row-aligned
    or cost-balanced — so the one-hot seeds land on the right owner.
    """
    if bounds.n != n or bounds.world_size != context.world_size or bounds.rank != context.rank:
        raise ValueError("`bounds` is inconsistent with the distributed context")
    if context.world_size > 1:
        dist = _dist()
        gathered: list[Optional[tuple[int, int]]] = [None] * context.world_size
        dist.all_gather_object(gathered, (bounds.start, bounds.stop), group=context.group)
        chunks = [chunk for chunk in gathered if chunk is not None]
        if len(chunks) != context.world_size:
            raise RuntimeError("failed to gather every rank's bounds")
        starts = [chunk[0] for chunk in chunks]
        stops = [chunk[1] for chunk in chunks]
    else:
        starts, stops = [0], [n]
    if (
        starts[0] != 0
        or stops[-1] != n
        or any(stops[rank] != starts[rank + 1] for rank in range(context.world_size - 1))
        or any(stop < start for start, stop in zip(starts, stops))
    ):
        raise ValueError(f"rank bounds do not tile [0, {n}): {list(zip(starts, stops))}")
    if (bounds.start, bounds.stop) != (starts[context.rank], stops[context.rank]):
        raise ValueError("this rank's bounds disagree with the gathered layout")
    return partition_rows_explicit(n, 1, [*starts, n], context.world_size)


def _make_local_seed_block(
    hii_local: Tensor,
    W_local: Tensor,
    n_roots: int,
    project_fn: Optional[Callable[[Tensor], Tensor]],
    reduce: Callable[[Tensor], Tensor],
    layout: _LocalIndexLayout,
    context: DistributedContext,
    seed_salt: int,
) -> Tensor:
    """Build the ``(n_roots, N_local)`` orthonormal local seed block.

    Mirrors ``davidson_torch``'s ``_make_seed_block``: the caller's start
    vector(s) first, then one-hot vectors at the globally smallest ``hii``
    determinants, each perturbed by partition-invariant hashed noise
    (``SEED_NOISE_SCALE``, normalized to unit norm first). Every candidate is
    projected by ``project_fn`` (collective) and Gram-Schmidt deflated against
    the accepted columns with all-reduced inner products.

    Raises:
        ValueError: fewer than ``n_roots`` independent directions survive.
    """
    from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE

    dtype = W_local.dtype
    device = W_local.device
    local_size = hii_local.shape[0]
    tiny = torch.finfo(dtype).tiny
    accepted: list[Tensor] = []

    def deflate(vector: Tensor) -> Tensor:
        for previous in accepted:
            overlap = _reduced_dot(previous, vector, reduce)
            vector = vector - overlap * previous
        return vector

    if W_local.dim() == 2:
        for row in W_local.to(device=device, dtype=dtype):
            candidate = project_fn(row) if project_fn is not None else row.clone()
            reference = _reduced_norm(candidate, reduce)
            candidate = deflate(candidate)
            norm = _reduced_norm(candidate, reduce)
            if float(reference) > 0 and float(norm) > 1e-8 * float(reference):
                accepted.append(candidate / norm.clamp_min(tiny))
            if len(accepted) >= n_roots:
                break
    else:
        candidate = W_local.clone().to(device=device, dtype=dtype)
        if project_fn is not None:
            candidate = project_fn(candidate)
        norm = _reduced_norm(candidate, reduce)
        if float(norm) > 0:
            accepted.append(candidate / norm.clamp_min(tiny))

    if len(accepted) < n_roots:
        global_index_of_local = layout.global_indices(context.rank).to(device=device)
        candidates = global_smallest_hii(hii_local, layout, k=n_roots + 4, group=context.group)
        for global_index, _value in candidates:
            if len(accepted) >= n_roots:
                break
            candidate = torch.zeros(local_size, dtype=dtype, device=device)
            position = int(
                torch.searchsorted(
                    global_index_of_local, torch.tensor(global_index, device=device)
                ).item()
            )
            if position < local_size and int(global_index_of_local[position]) == global_index:
                candidate[position] = 1.0
            noise = hash_uniform(global_index_of_local, salt=seed_salt).to(
                device=device, dtype=dtype
            )
            noise_norm = _reduced_norm(noise, reduce)
            if float(noise_norm) > 0:
                candidate = candidate + SEED_NOISE_SCALE * (noise / noise_norm.clamp_min(tiny))
            if project_fn is not None:
                candidate = project_fn(candidate)
            candidate = deflate(candidate)
            norm = _reduced_norm(candidate, reduce)
            if float(norm) < 1e-10:
                continue  # degenerate after projection
            accepted.append(candidate / norm.clamp_min(tiny))
        if len(accepted) < n_roots:
            raise ValueError(
                f"n_roots ({n_roots}) exceeds the available projected CI dimension "
                f"({len(accepted)})."
            )
    return torch.stack(accepted, dim=0)


def _solve_local_core(
    hii_local: Tensor,
    seed_block: Optional[Tensor],
    matvec_local: Callable[[Tensor], Tensor],
    *,
    n_roots: int,
    block_size: int,
    max_iters: int,
    eps: float,
    de_tol: Optional[float],
    project_fn: Optional[Callable[[Tensor], Tensor]],
    reduce: Callable[[Tensor], Tensor],
    context: DistributedContext,
    verbose: bool,
    resume: Optional[dict] = None,
    checkpoint: Optional[DavidsonCheckpoint] = None,
) -> tuple[Tensor, Tensor, dict]:
    """Block Davidson over preallocated local ``C``/``HC`` blocks.

    See :func:`davidson_distributed_local`. ``seed_block`` is the already
    projected/orthonormal ``(n_roots, N_local)`` start; the returned vectors
    are the lowest ``n_roots`` Ritz vectors as local blocks. When ``resume``
    holds a checkpoint state, ``seed_block`` is ``None`` and ``C``/``HC``/
    ``H_proj`` and the Ritz bookkeeping are restored instead.

    The local block may be complex: rows are vectors, so every inner product
    is Hermitian (``C.conj() @ HC.T``, ``vdot``) and every squared norm is
    ``|x|^2``. For real dtypes ``conj``/``real`` are no-ops, so the real path
    is numerically identical. ``hii_local`` may stay real (the diagonal of a
    Hermitian ``H`` is real); it is cast to the subspace dtype only for the
    preconditioner.
    """
    if resume is not None:
        saved_C = resume.get("C")
        if not isinstance(saved_C, Tensor) or saved_C.dim() != 2:
            raise RuntimeError("distributed checkpoint has no valid local subspace block")
        dtype = saved_C.dtype
        N_local = int(saved_C.shape[1])
        device = hii_local.device
        if N_local != hii_local.shape[0]:
            raise RuntimeError(
                f"distributed checkpoint local block has {N_local} rows but this "
                f"rank owns {hii_local.shape[0]}"
            )
    else:
        if seed_block is None:
            raise ValueError("_solve_local_core needs a seed block or a checkpoint state")
        dtype = seed_block.dtype
        device = seed_block.device
        N_local = seed_block.shape[1]
    tiny = torch.finfo(dtype).tiny
    eps_reg = 1.0e-7
    breakdown_rtol = torch.finfo(dtype).eps ** 0.5
    de_tol_eff: Optional[float] = None if de_tol is None else math.sqrt(de_tol)
    diag = hii_local.to(dtype)

    C = torch.zeros((block_size, N_local), dtype=dtype, device=device)
    HC = torch.zeros_like(C)
    H_proj = torch.zeros((block_size, block_size), dtype=dtype, device=device)

    def apply_op(vector: Tensor) -> Tensor:
        product = matvec_local(vector)
        if product.shape != vector.shape:
            raise ValueError(
                f"matvec_local returned shape {tuple(product.shape)} != {tuple(vector.shape)}"
            )
        if project_fn is not None:
            product = project_fn(product)
            if product.shape != vector.shape:
                raise ValueError(
                    f"project_fn returned shape {tuple(product.shape)} != {tuple(vector.shape)}"
                )
        return product

    def recompute_projection(size: int) -> None:
        projection = C[:size].conj() @ HC[:size].T
        reduce(projection)
        H_proj[:size, :size] = torch.triu(projection) + torch.triu(projection, 1).mH

    keep = max(n_roots, min(n_roots + 1, block_size - 1))
    n_cycles = max(1, int(max_iters))
    n_iters = n_cycles * block_size
    converged = False
    history: list = []
    has_prev = False
    prev_energies = torch.zeros(n_roots, dtype=torch.float64)
    last_energies = torch.zeros(n_roots, dtype=torch.float64)
    last_resnorms = torch.full((n_roots,), float("inf"), dtype=torch.float64)
    # See ``davidson._davidson_multi``: a sub-float64 subspace cannot push its
    # residual below its own precision floor; stop at a plateau instead of
    # exhausting every restart. The window counts restarts (``n_outer``).
    low_precision = torch.finfo(dtype).bits < 64
    stagnation_window = 15
    stagnation_rtol = 0.9
    best_residual = float("inf")
    best_history: list[float] = []
    stagnated = False
    ritz_vectors = torch.zeros((n_roots, N_local), dtype=dtype, device=device)
    ritz_products = torch.zeros_like(ritz_vectors)
    start_it = 0
    max_residual = float("inf")

    if resume is None:
        assert seed_block is not None
        seed_size = seed_block.shape[0]
        C[:seed_size] = seed_block
        for index in range(seed_size):
            HC[index] = apply_op(C[index])
        n_matvecs = seed_size
        recompute_projection(seed_size)
        d = seed_size
        n_outer = 1
    else:
        if resume.get("layout") != "local":
            raise RuntimeError(
                f"distributed checkpoint layout {resume.get('layout')!r} cannot resume "
                "an owner-computes local Davidson solve"
            )
        if int(resume.get("block_size", -1)) != block_size:
            raise RuntimeError(
                f"distributed checkpoint block_size {resume.get('block_size')} != "
                f"current {block_size}"
            )
        if int(resume.get("n_roots", -1)) != n_roots:
            raise RuntimeError(
                f"distributed checkpoint n_roots {resume.get('n_roots')} != current {n_roots}"
            )
        d = int(resume["d"])
        if not n_roots <= d <= block_size:
            raise RuntimeError(
                f"distributed checkpoint has invalid subspace size {d} for "
                f"n_roots={n_roots}, block_size={block_size}"
            )
        saved_HC = resume.get("HC")
        saved_proj = resume.get("H_proj")
        if (
            not isinstance(saved_HC, Tensor)
            or saved_HC.shape != (d, N_local)
            or not isinstance(saved_proj, Tensor)
            or saved_proj.shape != (d, d)
        ):
            raise RuntimeError("distributed checkpoint has inconsistent local blocks")
        assert isinstance(saved_C, Tensor)
        assert isinstance(saved_HC, Tensor)
        assert isinstance(saved_proj, Tensor)
        C[:d] = saved_C.to(device=device)
        HC[:d] = saved_HC.to(device=device)
        H_proj[:d, :d] = saved_proj.to(device=device)
        n_matvecs = int(resume.get("n_matvecs", d))
        n_outer = int(resume.get("n_outer", 1))
        history = list(resume.get("history") or [])
        has_prev = bool(resume.get("has_prev", False))
        prev_energies = resume.get("prev_energies", prev_energies).to(torch.float64).cpu()
        last_energies = resume.get("last_energies", last_energies).to(torch.float64).cpu()
        last_resnorms = resume.get("last_resnorms", last_resnorms).to(torch.float64).cpu()
        saved_ritz = resume.get("ritz_vectors")
        saved_ritz_products = resume.get("ritz_products")
        if (
            isinstance(saved_ritz, Tensor)
            and saved_ritz.dim() == 2
            and saved_ritz.shape[0] >= n_roots
            and saved_ritz.shape[1] == N_local
        ):
            ritz_vectors = saved_ritz.to(device=device, dtype=dtype)
        if (
            isinstance(saved_ritz_products, Tensor)
            and isinstance(saved_ritz, Tensor)
            and saved_ritz_products.shape == saved_ritz.shape
        ):
            ritz_products = saved_ritz_products.to(device=device, dtype=dtype)
        start_it = min(max(0, int(resume.get("next_iter", 0))), n_iters)
        if start_it >= n_iters and ritz_vectors.shape[0] < n_roots:
            raise RuntimeError(
                "distributed checkpoint has no Ritz vectors to return for a solve "
                "that has already exhausted its iteration budget"
            )
        if last_resnorms.numel():
            max_residual = float(last_resnorms.max().item())

    def checkpoint_state(next_iter: int) -> dict:
        """Rank-local state written at an iteration boundary."""
        return {
            "layout": "local",
            "dtype": str(dtype),
            "n_local": int(N_local),
            "block_size": int(block_size),
            "n_roots": int(n_roots),
            "d": int(d),
            "n_outer": int(n_outer),
            "n_matvecs": int(n_matvecs),
            "next_iter": int(next_iter),
            "C": C[:d].detach().cpu().clone(),
            "HC": HC[:d].detach().cpu().clone(),
            "H_proj": H_proj[:d, :d].detach().cpu().clone(),
            "last_energies": last_energies.detach().cpu().clone(),
            "last_resnorms": last_resnorms.detach().cpu().clone(),
            "has_prev": bool(has_prev),
            "prev_energies": prev_energies.detach().cpu().clone(),
            "ritz_vectors": ritz_vectors.detach().cpu().clone(),
            "ritz_products": ritz_products.detach().cpu().clone(),
            "history": list(history),
        }

    for it in range(start_it, n_iters):
        if (
            checkpoint is not None
            and checkpoint.save is not None
            and it > 0
            and it % checkpoint.every == 0
        ):
            checkpoint.save(it, checkpoint_state(it))
        energies_full, coefficients_full = _eigh_broadcast(H_proj[:d, :d], context)
        coefficients = coefficients_full.to(dtype)
        n_avail = min(n_roots, d)
        n_kept = min(keep, d)
        ritz_vectors = coefficients[:, :n_kept].T @ C[:d]
        ritz_products = coefficients[:, :n_kept].T @ HC[:d]
        norm_sq = (ritz_vectors.conj() * ritz_vectors).real.sum(dim=1)
        reduce(norm_sq)
        norms_v = torch.sqrt(norm_sq.clamp_min(0.0)).reshape(-1, 1).clamp_min(tiny)
        ritz_vectors = ritz_vectors / norms_v
        ritz_products = ritz_products / norms_v
        residuals = ritz_products[:n_avail] - (
            energies_full[:n_avail].to(dtype).unsqueeze(1) * ritz_vectors[:n_avail]
        )
        residual_sq = (residuals.conj() * residuals).real.sum(dim=1)
        reduce(residual_sq)
        resnorms = torch.sqrt(residual_sq.clamp_min(0.0))

        last_energies[:n_avail] = energies_full[:n_avail].cpu().to(torch.float64)
        last_resnorms[:n_avail] = resnorms.cpu().to(torch.float64)
        history.append((it, d - 1, last_energies[:n_avail].tolist(), resnorms.cpu().tolist()))

        resid_ok = n_avail == n_roots and float(resnorms.max().item()) < eps
        if de_tol_eff is None:
            converged_now = resid_ok
        else:
            de_ok = has_prev and bool(
                (torch.abs(last_energies[:n_avail] - prev_energies[:n_avail]) < de_tol_eff).all()
            )
            converged_now = resid_ok and de_ok
            if not converged_now:
                prev_energies[:n_avail] = last_energies[:n_avail]
                has_prev = True

        if converged_now and n_outer > 1:
            # Products carried through a thick restart accumulate round-off:
            # confirm convergence with fresh matvecs, as ``_davidson_multi``.
            targets = ritz_vectors[:n_avail].clone()
            checked = torch.stack([apply_op(vector) for vector in targets])
            n_matvecs += n_avail
            checked_diff = checked - (energies_full[:n_avail].to(dtype).unsqueeze(1) * targets)
            checked_sq = (checked_diff.conj() * checked_diff).real.sum(dim=1)
            reduce(checked_sq)
            checked_norms = torch.sqrt(checked_sq.clamp_min(0.0))
            ritz_products[:n_avail] = checked
            last_resnorms[:n_avail] = checked_norms.cpu().to(torch.float64)
            residuals[:n_avail] = checked_diff
            resnorms = checked_norms
            if float(checked_norms.max().item()) >= eps:
                converged_now = False

        if converged_now:
            converged = True
            break

        if d >= block_size:
            # No slot left to expand: restart from the kept Ritz vectors only.
            keep_d = min(keep, d)
            C[:keep_d] = ritz_vectors[:keep_d]
            HC[:keep_d] = ritz_products[:keep_d]
            d = keep_d
            recompute_projection(d)
            n_outer += 1
            if low_precision:
                current = float(last_resnorms[:n_roots].max().item())
                best_residual = min(best_residual, current)
                best_history.append(best_residual)
                if (
                    len(best_history) > stagnation_window
                    and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
                ):
                    stagnated = True
                    break
            continue

        # Preconditioned correction for the unconverged root with the
        # largest residual norm.
        resnorms_cpu = last_resnorms[:n_avail].clone()
        unconverged = resnorms_cpu >= eps
        if bool(unconverged.any()):
            resnorms_cpu[~unconverged] = -1.0
            k_target = int(resnorms_cpu.argmax().item())
        else:
            k_target = 0
        residual_norm = resnorms[k_target]
        target = residuals[k_target]
        if float(residual_norm) > 0:
            target = target / residual_norm
        denominator = energies_full[k_target].to(dtype) - diag
        small = (denominator.abs() <= eps_reg).to(dtype)
        denominator = denominator - eps_reg * small
        next_vec = target / denominator

        basis = C[:d]
        candidate: Optional[Tensor] = None
        norm_c = torch.zeros((), dtype=dtype, device=device)
        breakdown = True
        for trial in (next_vec, target):
            norm_pre = _reduced_norm(trial, reduce)
            trial = _cgs2(trial, basis, reduce)
            if project_fn is not None:
                trial = project_fn(trial)
            norm_c = _reduced_norm(trial, reduce)
            if float(norm_c) >= breakdown_rtol * float(norm_pre):
                candidate = trial
                breakdown = False
                break
        if breakdown or candidate is None:
            keep_d = min(keep, d)
            C[:keep_d] = ritz_vectors[:keep_d]
            HC[:keep_d] = ritz_products[:keep_d]
            d = keep_d
            recompute_projection(d)
            n_outer += 1
            if low_precision:
                current = float(last_resnorms[:n_roots].max().item())
                best_residual = min(best_residual, current)
                best_history.append(best_residual)
                if (
                    len(best_history) > stagnation_window
                    and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
                ):
                    stagnated = True
                    break
            continue

        correction = candidate / norm_c.clamp_min(tiny)
        C[d] = correction
        HC[d] = apply_op(correction)
        n_matvecs += 1
        column = C[: d + 1].conj() @ HC[d]
        reduce(column)
        H_proj[: d + 1, d] = column
        H_proj[d, : d + 1] = column.conj()
        d += 1

        if d == block_size:
            # Thick restart: kept Ritz vectors plus the new correction.
            keep_d = min(keep, d - 1)
            C[:keep_d] = ritz_vectors[:keep_d]
            HC[:keep_d] = ritz_products[:keep_d]
            C[keep_d] = correction
            HC[keep_d] = HC[d - 1].clone()
            d = keep_d + 1
            recompute_projection(d)
            n_outer += 1
            if low_precision:
                current = float(last_resnorms[:n_roots].max().item())
                best_residual = min(best_residual, current)
                best_history.append(best_residual)
                if (
                    len(best_history) > stagnation_window
                    and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
                ):
                    stagnated = True
                    break

    final_energies = last_energies[:n_roots].to(torch.float64).cpu()
    final_vectors = ritz_vectors[:n_roots].detach()
    final_resnorms = last_resnorms[:n_roots].tolist()
    max_residual = float(last_resnorms[:n_roots].max().item())
    if verbose and context.rank == 0:
        print(
            f"  [distributed local davidson] iters={len(history)} "
            f"matvecs={n_matvecs} residual={max_residual:.3e} converged={converged}"
        )
    info = {
        "converged": converged,
        "n_outer": n_outer,
        "n_matvecs": n_matvecs,
        "final_residual_norm": max_residual,
        "final_residual_norms": final_resnorms,
        "history": history,
        "distributed": True,
    }
    if stagnated:
        info["stagnated"] = True
    return final_energies, final_vectors, info


def _davidson_single_local(
    hii_local: Tensor,
    W_local: Tensor,
    matvec_local: Callable[[Tensor], Tensor],
    max_iters: int,
    block_size: int,
    eps: float,
    *,
    project_fn: Optional[Callable[[Tensor], Tensor]],
    reduce: Callable[[Tensor], Tensor],
    context: DistributedContext,
    stop_below: Optional[float] = None,
) -> tuple[float, Tensor, dict]:
    """Single-root local Davidson used by the missed-root search.

    Mirrors ``_davidson_single`` with ``restart="recompute"``: every outer
    cycle starts again from the current Ritz vector, and every reduction is
    replaced by an all-reduce. ``stop_below`` stops as soon as a Ritz value
    falls below it (a Ritz value bounds the lowest eigenvalue from above, so
    that already proves a missed root exists). A complex local block is
    handled with the same Hermitian inner products as ``_solve_local_core``.
    """
    dtype = W_local.dtype
    device = W_local.device
    N_local = W_local.shape[0]
    tiny = torch.finfo(dtype).tiny
    eps_reg = 1.0e-12
    breakdown_rtol = torch.finfo(dtype).eps ** 0.5
    diag = hii_local.to(dtype)

    C = torch.zeros((block_size, N_local), dtype=dtype, device=device)
    HC = torch.zeros_like(C)
    H_proj = torch.zeros((block_size, block_size), dtype=dtype, device=device)
    W_cur = W_local.clone()
    if project_fn is not None:
        W_cur = project_fn(W_cur)
    W_cur = W_cur / _reduced_norm(W_cur, reduce).clamp_min(tiny)

    history: list = []
    n_matvecs = 0
    converged = False
    stopped_below = False
    last_energy = 0.0
    last_resnorm = float("inf")
    # See ``davidson._davidson_single``: stop a sub-float64 solve at its
    # precision floor rather than exhausting the restart budget. This also
    # bounds every distributed missed-root search.
    low_precision = torch.finfo(dtype).bits < 64
    stagnation_window = 15
    stagnation_rtol = 0.9
    best_resnorm = float("inf")
    best_history: list[float] = []
    stagnated = False
    for it in range(max(1, int(max_iters))):
        C[0] = W_cur
        HC[0] = matvec_local(C[0])
        n_matvecs += 1
        if project_fn is not None:
            HC[0] = project_fn(HC[0])
        column = _reduced_dot(C[0], HC[0], reduce)
        H_proj[0, 0] = column
        for ib in range(block_size):
            energies_full, coefficients_full = _eigh_broadcast(H_proj[: ib + 1, : ib + 1], context)
            energy = energies_full[0]
            coefficients = coefficients_full[:, 0].to(dtype)
            W_new = coefficients @ C[: ib + 1]
            HW_new = coefficients @ HC[: ib + 1]
            norm_W = _reduced_norm(W_new, reduce)
            W_new = W_new / norm_W.clamp_min(tiny)
            HW_new = HW_new / norm_W.clamp_min(tiny)
            residual = HW_new - energy.to(dtype) * W_new
            norm_R = _reduced_norm(residual, reduce)
            residual = residual / norm_R.clamp_min(tiny)
            W_cur = W_new
            last_energy = float(energy)
            last_resnorm = float(norm_R)
            history.append((it, ib, last_energy, last_resnorm))
            if stop_below is not None and last_energy < stop_below:
                stopped_below = True
                break
            if last_resnorm < eps:
                converged = True
                break
            if ib < block_size - 1:
                denominator = energy.to(dtype) - diag
                small = (denominator.abs() <= eps_reg).to(dtype)
                denominator = denominator - eps_reg * small
                next_vec = residual / denominator
                basis = C[: ib + 1]
                candidate: Optional[Tensor] = None
                breakdown = True
                for trial in (next_vec, residual):
                    norm_pre = _reduced_norm(trial, reduce)
                    trial = _cgs2(trial, basis, reduce)
                    if project_fn is not None:
                        trial = project_fn(trial)
                    norm_c = _reduced_norm(trial, reduce)
                    if float(norm_c) >= breakdown_rtol * float(norm_pre):
                        candidate = trial
                        breakdown = False
                        break
                if breakdown or candidate is None:
                    break
                C[ib + 1] = candidate / norm_c.clamp_min(tiny)
                HC[ib + 1] = matvec_local(C[ib + 1])
                n_matvecs += 1
                if project_fn is not None:
                    HC[ib + 1] = project_fn(HC[ib + 1])
                column = C[: ib + 2].conj() @ HC[ib + 1]
                reduce(column)
                H_proj[: ib + 2, ib + 1] = column
                H_proj[ib + 1, : ib + 2] = column.conj()
        if converged or stopped_below:
            break
        if low_precision:
            best_resnorm = min(best_resnorm, last_resnorm)
            best_history.append(best_resnorm)
            if (
                len(best_history) > stagnation_window
                and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
            ):
                stagnated = True
                break

    info = {
        "converged": converged,
        "n_outer": it + 1,
        "n_matvecs": n_matvecs,
        "final_residual_norm": last_resnorm,
        "history": history,
        "distributed": True,
    }
    if stop_below is not None:
        info["stopped_below"] = stopped_below
    if stagnated:
        info["stagnated"] = True
    return last_energy, W_cur, info


def _deflated_roots(roots: Tensor, reduce: Callable[[Tensor], Tensor]) -> Tensor:
    """Globally orthonormal local rows spanning the converged roots.

    The replicated Hermitian Gram matrix ``G = roots roots^H`` is all-reduced,
    and ``Q = L^{-1} roots`` with ``G = L L^H`` satisfies
    ``sum_r Q_r Q_r^H = I``: the rows of ``Q`` are globally orthonormal with
    respect to the Hermitian inner product. ``roots.mH``/``conj`` are no-ops
    for real dtypes.
    """
    width = roots.shape[0]
    work = roots.to(torch.complex128 if roots.is_complex() else torch.float64)
    gram = work @ work.mH
    reduce(gram)
    diagonal = float(gram.diagonal().abs().max().item()) if width else 1.0
    jitter = torch.finfo(torch.float64).eps * max(1.0, diagonal) * width
    factor = torch.linalg.cholesky(
        gram + jitter * torch.eye(width, dtype=work.dtype, device=gram.device)
    )
    return torch.linalg.solve_triangular(factor, work, upper=False).to(roots.dtype)


def _missed_root_local(
    hii_local: Tensor,
    matvec_local: Callable[[Tensor], Tensor],
    roots: Tensor,
    theta_max: float,
    eps: float,
    block_size: int,
    max_iters: int,
    *,
    project_fn: Optional[Callable[[Tensor], Tensor]],
    reduce: Callable[[Tensor], Tensor],
    context: DistributedContext,
    layout: _LocalIndexLayout,
    seed_salt: int,
    attempt: int,
    margin: float = 1e-6,
) -> tuple[bool, Tensor, float, int]:
    """Distributed analogue of ``_missed_root`` (see ``davidson_torch``).

    Runs a single-root local Davidson on the deflated operator
    ``P H P`` from the lowest-diagonal determinant not explained by the roots
    plus hashed noise, and reports ``(found, vector, energy, n_matvecs)``.
    ``found`` is also true when the search neither drops below
    ``theta_max - margin`` nor converges (undecided: conservative).
    """
    dtype = roots.dtype
    device = roots.device
    tiny = torch.finfo(dtype).tiny
    Q = _deflated_roots(roots, reduce)

    def deflate(vector: Tensor) -> Tensor:
        if project_fn is not None:
            vector = project_fn(vector)
        # Rows of ``Q`` are vectors, so the projector onto their (Hermitian)
        # span is ``sum_i q_i q_i^H``: the conj is inside the coefficient and
        # the transpose applies the rows. For real dtypes this is the previous
        # ``Q @ v`` / ``Q.T @ coefficients`` pair unchanged.
        coefficients = Q.conj() @ vector
        reduce(coefficients)
        return vector - Q.T @ coefficients

    global_index_of_local = layout.global_indices(context.rank).to(device=device)
    noise = hash_uniform(global_index_of_local, salt=seed_salt + 1 + attempt).to(
        device=device, dtype=dtype
    )
    noise = deflate(noise)
    noise = noise / _reduced_norm(noise, reduce).clamp_min(tiny)

    start = None
    candidates = global_smallest_hii(hii_local, layout, k=roots.shape[0] + 4, group=context.group)
    for global_index, _value in candidates:
        one_hot = torch.zeros(hii_local.shape[0], dtype=dtype, device=device)
        position = int(
            torch.searchsorted(
                global_index_of_local, torch.tensor(global_index, device=device)
            ).item()
        )
        if position < hii_local.shape[0] and int(global_index_of_local[position]) == global_index:
            one_hot[position] = 1.0
        one_hot = deflate(one_hot)
        norm = _reduced_norm(one_hot, reduce)
        if float(norm) > 0.5:  # mostly a new direction, not a leftover root
            start = one_hot / norm.clamp_min(tiny)
            break
    start = noise if start is None else start + 0.1 * noise
    start = start / _reduced_norm(start, reduce).clamp_min(tiny)

    def operator(vector: Tensor) -> Tensor:
        return deflate(matvec_local(deflate(vector)))

    energy, vector, info = _davidson_single_local(
        hii_local,
        start,
        operator,
        max_iters,
        block_size,
        eps,
        project_fn=deflate,
        reduce=reduce,
        context=context,
        stop_below=theta_max - margin,
    )
    found = bool(info.get("stopped_below")) or energy < theta_max - margin
    found = found or not info["converged"]
    return found, vector, float(energy), int(info["n_matvecs"])


def davidson_distributed_local(
    hii_local: Tensor,
    W_local: Tensor,
    matvec_local: Callable[[Tensor], Tensor],
    bounds: Union[RankBounds, TileBounds, Any],
    *,
    n: int,
    context: DistributedContext,
    max_iters: int,
    block_size: int,
    eps: float,
    n_roots: int = 1,
    de_tol: Optional[float] = None,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    missed_root_check: bool = False,
    verbose: bool = False,
    seed_salt: int = 0,
    checkpoint: Optional[DavidsonCheckpoint] = None,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Block Davidson on local ``(N_local,)`` blocks, all-reduced projections.

    This is the owner-computes variant of :func:`davidson_distributed`: no
    rank ever materializes a length-``n`` vector. ``C`` and ``HC`` are
    preallocated ``(block_size, N_local)`` blocks in ``W_local.dtype``; each
    appended column costs one ``(d + 1,)`` all-reduce for the projection, Ritz
    vectors/products are one GEMM, residual norms and the block CGS2
    coefficients are all-reduced, and ``eigh`` always runs in fp64 with a
    rank-0 broadcast of the eigenpairs so every rank forms identical Ritz
    vectors. The returned eigenvectors are the **local blocks** of the Ritz
    vectors, not gathered.

    ``W_local`` may be complex. Every inner product is then Hermitian
    (``C.conj() @ HC.T``, ``vdot``) and every squared norm is ``|x|^2``;
    ``hii_local`` may stay real because the diagonal of a Hermitian ``H`` is
    real.

    Args:
        hii_local: ``(N_local,)`` Hamiltonian diagonal block (already shifted
            when the subspace is fp32), real or in the subspace dtype.
        W_local: ``(N_local,)`` or ``(k <= n_roots, N_local)`` start vector(s)
            in the subspace dtype (real or complex).
        matvec_local: ``(N_local,) -> (N_local,)`` local operator closure in
            the subspace dtype; it performs its own row exchange.
        bounds: This rank's block: a flat :class:`RankBounds` (typically
            ``partition_vector``; row-aligned ``partition_rows`` is also
            supported), a 2-D :class:`TileBounds`, or a GDB pair-tile bounds
            object exposing ``n``, ``local_size``, ``pair_index``, ``rank``
            and ``world_size``.
        n: Global determinant count.
        context: Rank/world-size/device of the initialized process group.
        max_iters: Maximum number of outer restart cycles.
        block_size: Maximum subspace size.
        eps: Residual-norm convergence threshold.
        n_roots: Number of lowest eigenpairs to return.
        de_tol: Opt-in eigenvalue-stability check (effective threshold
            ``sqrt(de_tol)``), matching ``davidson_torch``.
        project_fn: Optional local projector (collective, it performs P2P
            inside). Applied to seeds, matvec products and corrections exactly
            like ``davidson_torch``.
        missed_root_check: Multi-root only. After convergence, search for a
            root below the highest one and re-seed with it (up to four
            attempts); results go to ``info["missed_root_checks"]``.
        verbose: Print a one-line summary on rank 0.
        seed_salt: Stream selector for the partition-invariant seed noise.
        checkpoint: Optional iteration-boundary persistence hook. When it
            yields a saved state, the seed block is skipped and ``C``/``HC``/
            ``H_proj`` continue from the saved iteration. ``None`` is the
            zero-overhead path.

    Returns:
        ``(float, (N_local,), info)`` for ``n_roots == 1`` or
        ``((n_roots,), (n_roots, N_local), info)`` otherwise. Energies are
        CPU float64; vectors are local blocks in the subspace dtype.

    Raises:
        ValueError: shapes/partition inconsistent, ``n <= block_size`` (use
            the replicated dense path), ``block_size < n_roots + 1``, or a
            complex ``hii_local`` with a real ``W_local``.

    Notes
    -----
    The projection is maintained incrementally,

    .. math::

        H^{\\mathrm{proj}}_{j,d} = \\sum_r \\overline{C_j[r]}\\,(HC_d)[r],
        \\qquad j \\le d,

    with one all-reduce per appended column; a thick restart keeps
    ``max(n_roots, min(n_roots + 1, block_size - 1))`` Ritz vectors plus the
    new correction and rebuilds the full ``(d, d)`` projection with one
    all-reduce. Every branch is taken from all-reduced quantities, so all
    ranks post the same collectives in the same order.

    ``n <= block_size`` is rejected because the diagonal preconditioner can
    stall on a near-degenerate block smaller than the space; the solver
    dispatch falls back to the replicated dense path there.
    """
    if bounds.n != n or bounds.world_size != context.world_size or bounds.rank != context.rank:
        raise ValueError("`bounds` is inconsistent with the distributed context")
    if n <= block_size:
        raise ValueError(
            f"davidson_distributed_local needs n ({n}) > block_size ({block_size}); "
            "use the replicated dense path for a basis that fits in one block"
        )
    _validate_davidson_common(
        n_roots, block_size, restart=True, context="davidson_distributed_local"
    )
    if hii_local.dim() != 1 or hii_local.shape[0] != bounds.local_size:
        raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({bounds.local_size},)")
    if W_local.dim() not in (1, 2) or W_local.shape[-1] != bounds.local_size:
        raise ValueError(
            f"W_local shape {tuple(W_local.shape)} does not end in N_local={bounds.local_size}"
        )
    if W_local.dim() == 2 and W_local.shape[0] > n_roots:
        raise ValueError(
            f"W_local has {W_local.shape[0]} start vectors, more than n_roots ({n_roots})"
        )
    if hii_local.dtype not in (W_local.dtype, W_local.dtype.to_real()):
        raise ValueError(
            f"hii_local dtype {hii_local.dtype} must be {W_local.dtype} or its "
            f"real part {W_local.dtype.to_real()}"
        )
    if W_local.device != hii_local.device:
        raise ValueError("W_local and hii_local must live on the same device")

    reduce = _make_all_reduce(context)
    layout = _index_layout(n, bounds, context)
    resume = None
    if checkpoint is not None and checkpoint.load is not None:
        loaded = checkpoint.load()
        if loaded is not None:
            resume = loaded["state"]
    seed_block = None
    if resume is None:
        seed_block = _make_local_seed_block(
            hii_local,
            W_local,
            n_roots,
            project_fn,
            reduce,
            layout,
            context,
            seed_salt,
        )
    energies, vectors, info = _solve_local_core(
        hii_local,
        seed_block,
        matvec_local,
        n_roots=n_roots,
        block_size=block_size,
        max_iters=max_iters,
        eps=eps,
        de_tol=de_tol,
        project_fn=project_fn,
        reduce=reduce,
        context=context,
        verbose=verbose,
        resume=resume,
        checkpoint=checkpoint,
    )

    n_matvecs = info["n_matvecs"]
    checks: list[dict] = []
    if missed_root_check and n_roots > 1:
        for attempt in range(4):
            found, missed, energy, matvecs = _missed_root_local(
                hii_local,
                matvec_local,
                vectors,
                float(energies.max()),
                eps,
                block_size,
                max_iters,
                project_fn=project_fn,
                reduce=reduce,
                context=context,
                layout=layout,
                seed_salt=seed_salt,
                attempt=attempt,
            )
            n_matvecs += matvecs
            checks.append({"found": found, "energy": energy, "n_matvecs": matvecs})
            if not found or attempt == 3:
                break
            # The missed vector first (it lies below the highest root), then
            # the roots found, as in ``davidson_torch``.
            seed = torch.cat([missed.unsqueeze(0).to(W_local.dtype), vectors[: n_roots - 1]])
            energies, vectors, info = _solve_local_core(
                hii_local,
                seed,
                matvec_local,
                n_roots=n_roots,
                block_size=block_size,
                max_iters=max_iters,
                eps=eps,
                de_tol=de_tol,
                project_fn=project_fn,
                reduce=reduce,
                context=context,
                verbose=verbose,
            )
            n_matvecs += info["n_matvecs"]
        info["n_matvecs"] = n_matvecs
        info["missed_root_checks"] = checks

    if n_roots == 1:
        return float(energies[0]), vectors[0], info
    return energies, vectors, info
