"""2-D tile bounds for a GDB (explicit pair) basis.

TPB carries a Cartesian ``α × β`` basis, so a tile is a slab product and its
flat vector is one contiguous rectangle. A GDB basis is an explicit pair list
that happens to be a disjoint union of Cartesian blocks ``U_g × B_g``
(:class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`), so the same
``α``-uid × ``β``-uid slabs select a *scattered* subset of the flat pair
vector: inside a block the owned pairs form a rectangle of that block's grid,
and the union over blocks is the tile.

This module supplies the foundation for the GDB owner-computes matvec over a
:class:`~torch_sci_solver.torch_impl.distributed_davidson.TilePartition`:

* :class:`PairTileBounds` — the uid slabs one rank owns, the sorted global
  pair indices it owns, and the local↔global pair map.
* :class:`PairFragment` / :func:`pair_tile_fragments` — the per-block
  rectangles a tile occupies, with their global pair indices in the block's
  row-major order, ready for the block kernels.
* :func:`pair_axis_weights` / :func:`pair_tile_partition` — per-uid pair
  counts and the :class:`TilePartition` they cost-balance.

Notes
-----
A pair ``k`` belongs to the tile when ``alpha_start ≤ α_uid(k) < alpha_stop``
and ``beta_start ≤ β_uid(k) < beta_stop``. Uid slabs are products of
contiguous *uid* ranges, so every pair belongs to exactly one tile of the
grid and the union over ranks is the whole basis. Inside a block, the sorted
uid arrays make each slab intersection a contiguous run of positions, so a
fragment is the rectangle ``[r0, r1) × [c0, c1)`` of the block's
``|U_g| × |B_g|`` grid, whose pair indices are the matching slice of the
block's row-major ``pair_idx`` array.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.distributed_davidson import TilePartition, partition_tiles
from torch_sci_solver.torch_impl.pair_blocks import SpinGroups, group_rect_indices
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis


@dataclass(frozen=True)
class PairTileBounds:
    """The ``α``-uid × ``β``-uid slab one rank owns of an explicit pair basis.

    Attributes:
        n_pairs: Global pair count.
        n_alpha_uids: Number of unique α strings (the row axis of the grid).
        n_beta_uids: Number of unique β strings (the column axis).
        rank: This tile's rank.
        world_size: Number of ranks.
        row_slab: Index of the owned α-uid slab.
        col_slab: Index of the owned β-uid slab.
        alpha_start: First owned α uid (inclusive).
        alpha_stop: One past the last owned α uid.
        beta_start: First owned β uid (inclusive).
        beta_stop: One past the last owned β uid.
        pair_index: ``(n_local,)`` sorted global pair indices this rank owns.
    """

    n_pairs: int
    n_alpha_uids: int
    n_beta_uids: int
    rank: int
    world_size: int
    row_slab: int
    col_slab: int
    alpha_start: int
    alpha_stop: int
    beta_start: int
    beta_stop: int
    pair_index: Tensor

    @property
    def n(self) -> int:
        """Global pair count (the flat vector length)."""
        return int(self.n_pairs)

    @property
    def local_size(self) -> int:
        """Number of owned pairs."""
        return int(self.pair_index.numel())

    @property
    def is_empty(self) -> bool:
        """Whether this rank owns no pair."""
        return self.local_size == 0

    def global_pair_index(self, local: Tensor) -> Tensor:
        """Map tile-local positions to global pair indices."""
        return self.pair_index[local]

    def local_of_global(self, global_idx: Tensor) -> Tensor:
        """Map owned global pair indices back to tile-local positions.

        Args:
            global_idx: Global pair indices contained in :attr:`pair_index`.

        Raises:
            ValueError: an index outside the tile.
        """
        index = torch.as_tensor(global_idx, dtype=torch.int64)
        local = torch.searchsorted(self.pair_index, index)
        if bool((local >= self.local_size).any()):
            raise ValueError("global pair index outside this tile")
        found = self.pair_index[local]
        if not torch.equal(found, index):
            raise ValueError("global pair index outside this tile")
        return local

    def owns(self, global_idx: int) -> bool:
        """Whether the global pair index belongs to this tile."""
        probe = torch.tensor([int(global_idx)], dtype=torch.int64, device=self.pair_index.device)
        local = torch.searchsorted(self.pair_index, probe)
        position = int(local[0])
        return position < self.local_size and int(self.pair_index[position]) == int(global_idx)


def pair_tile_bounds(
    tile: TilePartition,
    pair_basis: "PairBasis | StructuralPairBasis",
    rank: Optional[int] = None,
) -> PairTileBounds:
    """Build one rank's :class:`PairTileBounds` over the uid axes.

    Args:
        tile: A tile partition sized ``(M_α_unique, M_β_unique)``.
        pair_basis: The pair basis the partition indexes. A structural basis
            is intersected block by block, so only the owned indices are
            ever built.
        rank: Rank to describe; defaults to ``0``.

    Returns:
        The rank's tile bounds, with its owned global pair indices sorted.

    Raises:
        ValueError: the partition does not match the basis' uid axes.
    """
    n_alpha = int(pair_basis.unique_alphas.shape[0])
    n_beta = int(pair_basis.unique_betas.shape[0])
    if tile.n_rows != n_alpha or tile.n_cols != n_beta:
        raise ValueError(
            f"tile partition ({tile.n_rows}, {tile.n_cols}) does not match the uid axes "
            f"({n_alpha}, {n_beta})"
        )
    rank = 0 if rank is None else int(rank)
    bound = tile.for_rank(rank)
    device = pair_basis.device
    if isinstance(pair_basis, StructuralPairBasis):
        parts = []
        for block in pair_basis.blocks:
            row_start = max(bound.row_start, block.alpha_start)
            row_stop = min(bound.row_stop, block.alpha_stop)
            col_start = max(bound.col_start, block.beta_start)
            col_stop = min(bound.col_stop, block.beta_stop)
            if row_stop <= row_start or col_stop <= col_start:
                continue
            rows = torch.arange(row_start, row_stop, dtype=torch.int64, device=device)
            cols = torch.arange(col_start, col_stop, dtype=torch.int64, device=device)
            parts.append(
                block.pair_base
                + (rows - block.alpha_start).unsqueeze(1) * block.n_beta
                + (cols - block.beta_start).unsqueeze(0)
            )
        pair_index = (
            torch.sort(torch.cat([part.reshape(-1) for part in parts])).values
            if parts
            else torch.empty(0, dtype=torch.int64, device=device)
        )
    else:
        a_uid = pair_basis.alpha_uid_of_pair
        b_uid = pair_basis.beta_uid_of_pair
        mask = (
            (a_uid >= bound.row_start)
            & (a_uid < bound.row_stop)
            & (b_uid >= bound.col_start)
            & (b_uid < bound.col_stop)
        )
        pair_index = torch.nonzero(mask, as_tuple=False).reshape(-1).to(device)
    return PairTileBounds(
        n_pairs=pair_basis.n_pairs,
        n_alpha_uids=n_alpha,
        n_beta_uids=n_beta,
        rank=rank,
        world_size=tile.world_size,
        row_slab=bound.row_slab,
        col_slab=bound.col_slab,
        alpha_start=bound.row_start,
        alpha_stop=bound.row_stop,
        beta_start=bound.col_start,
        beta_stop=bound.col_stop,
        pair_index=pair_index,
    )


def pair_axis_weights(pair_helpers: PairHelpers) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Per-uid pair counts along the two tile axes.

    A pair count is a cheap proxy for the owner-computes block work: it tracks
    the number of owned rows and columns the same-spin channels enumerate and
    the αβ channel's owned-pair share, though the candidate-grid cost of one
    owned row still depends on the block density. The counts are exact in both
    representations — a materialised basis bin-counts its per-pair uid arrays,
    and a structural basis adds each Cartesian block's row/column extent to
    its uid ranges — so no pair list is ever built.

    Args:
        pair_helpers: Helpers for the basis being partitioned.

    Returns:
        ``(row_weights, col_weights)``: ``(M_α,)`` and ``(M_β,)`` float64
        non-negative pair counts on the helpers' device. An axis is ``None``
        when the helpers carry neither the per-pair uid array nor the
        structural block geometry needed to count it.
    """
    device = pair_helpers.device
    row_weights: Optional[Tensor]
    col_weights: Optional[Tensor]
    structural = pair_helpers.structural
    if structural is not None:
        row_weights = torch.zeros(pair_helpers.m_alpha, dtype=torch.float64, device=device)
        col_weights = torch.zeros(pair_helpers.m_beta, dtype=torch.float64, device=device)
        for block in structural.blocks:
            row_weights[block.alpha_start : block.alpha_stop] += float(block.n_beta)
            col_weights[block.beta_start : block.beta_stop] += float(block.n_alpha)
        return row_weights, col_weights
    alpha_uid = pair_helpers.alpha_uid_of_pair
    beta_uid = pair_helpers.beta_uid_of_pair
    row_weights = (
        None
        if alpha_uid is None
        else torch.bincount(alpha_uid, minlength=pair_helpers.m_alpha).to(torch.float64)
    )
    col_weights = (
        None
        if beta_uid is None
        else torch.bincount(beta_uid, minlength=pair_helpers.m_beta).to(torch.float64)
    )
    return row_weights, col_weights


def pair_replicated_bytes(pair_helpers: PairHelpers) -> int:
    """Bytes of the replicated pair arrays every rank holds before the solve.

    A GDB solve keeps the pair basis and its helpers on every rank whichever
    layout is chosen: the explicit basis' per-pair strings, uids and partners
    CSR, plus the helpers' derived per-pair keys and partner tables (nine
    length-``N`` int64 arrays), and for a structural basis the unique-string
    arrays and block-id tables. The estimate reads only tensor metadata, so it
    allocates nothing; shared tensor objects are counted once.

    Args:
        pair_helpers: Helpers for the basis being partitioned.

    Returns:
        Sum of ``numel() * element_size()`` over the replicated tensors.
    """
    tensors: list[Tensor] = []
    for field in fields(pair_helpers):
        value = getattr(pair_helpers, field.name)
        if isinstance(value, Tensor):
            tensors.append(value)
    structural = pair_helpers.structural
    if structural is not None:
        for field in fields(structural):
            value = getattr(structural, field.name)
            if isinstance(value, Tensor):
                tensors.append(value)
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += int(tensor.numel()) * int(tensor.element_size())
    return total


def resolve_auto_pair_tiling(
    *,
    world_size: int,
    n_pairs: int,
    block_size: int = 8,
    itemsize: int = 8,
    budget_bytes: Optional[int],
    replicated_bytes: int = 0,
) -> bool:
    """Whether the owner-computes GDB solve should use a tile grid.

    Tiling buys memory, not speed: at equal size the 1-D block matvec is
    measured roughly twice as fast as the tiled one, so a grid is chosen only
    when the structures it removes would not fit. The 1-D path all-gathers the
    full pair vector once per matvec and keeps the Davidson block terms; the
    tiled path holds only the owned tile plus one staged peer tile.

    Args:
        world_size: Number of ranks.
        n_pairs: Global pair count.
        block_size: Nominal Davidson block size.
        itemsize: Bytes per vector element.
        budget_bytes: Usable device memory per rank, or ``None`` when unknown
            (CPU runs).
        replicated_bytes: Bytes of the pair basis and helper arrays kept on
            every rank under either layout, from
            :func:`pair_replicated_bytes`. Defaults to 0 for callers that only
            model the vector working set.

    Returns:
        ``True`` when a tile grid should be used; ``False`` keeps the 1-D
        owner-computes block matvec.

    Notes
    -----
    The 1-D peak is modelled as one gathered input vector plus the Davidson
    block vectors:

    .. math::

        B_{1D} \\approx N \\cdot s + (b + 9) \\frac{N}{P} s .

    The replicated pair arrays are present under both layouts, so they cancel
    in a tile-versus-1-D comparison, but they still occupy the device when the
    fit is judged: they are counted against the budget here. Tiling is chosen
    when that combined footprint exceeds half the device budget, matching the
    TPB policy's headroom rule. With no byte budget (CPU runs) the faster 1-D
    block matvec is kept, since CPU memory is not the constraint; a single
    rank never tiles.
    """
    if world_size <= 1 or n_pairs <= 0:
        return False
    if budget_bytes is None:
        return False
    one_d_bytes = n_pairs * itemsize + (block_size + 9) * n_pairs * itemsize / world_size
    return one_d_bytes + int(replicated_bytes) > 0.5 * budget_bytes


def pair_tile_partition(
    pair_helpers: PairHelpers,
    world_size: int,
    grid: Optional[tuple[int, int]] = None,
) -> TilePartition:
    """Tile the pair basis' uid axes, cutting each axis by pair count.

    The cuts follow the weighted boundaries of ``partition_tiles`` over the
    weights of :func:`pair_axis_weights`. Balancing pair counts keeps the
    owned row/column counts, and hence the dominant per-rank matvec work,
    close to uniform, but the per-row candidate cost is block-density
    dependent, so the cut is an approximation rather than an exact work
    model. The per-slab row cap of the weighted cut still applies, so a very
    concentrated weight distribution is balanced only as far as that cap
    allows.

    Args:
        pair_helpers: Helpers for the basis being partitioned.
        world_size: Number of ranks (the grid product).
        grid: Optional explicit ``(p_alpha, p_beta)``; defaults to the
            even-tile grid choice of ``resolve_tile_grid``.

    Returns:
        The tile partition over the basis' ``M_α × M_β`` uid axes. When the
        helpers cannot supply weights for either axis, the unweighted even
        partition is returned instead of guessing.
    """
    row_weights, col_weights = pair_axis_weights(pair_helpers)
    if row_weights is None or col_weights is None:
        row_weights = col_weights = None
    return partition_tiles(
        int(pair_helpers.m_alpha),
        int(pair_helpers.m_beta),
        int(world_size),
        grid=grid,
        row_weights=row_weights,
        col_weights=col_weights,
    )


@dataclass(frozen=True)
class PairFragment:
    """One Cartesian block's part of a tile: a rectangle of the block grid.

    Attributes:
        group: Index of the block in its :class:`SpinGroups` tuple.
        row_start: First owned row of the block grid (α-uid position).
        row_stop: One past the last owned row.
        col_start: First owned column of the block grid (β-uid position).
        col_stop: One past the last owned column.
        pair_index: ``(rows * cols,)`` global pair indices of the rectangle in
            the block's row-major order.
    """

    group: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    pair_index: Tensor

    @property
    def n_rows(self) -> int:
        """Owned rows of the block grid."""
        return self.row_stop - self.row_start

    @property
    def n_cols(self) -> int:
        """Owned columns of the block grid."""
        return self.col_stop - self.col_start


def _lower_bound(values: Tensor, value: int) -> int:
    """Position of ``value`` in the sorted ``values`` (insertion point)."""
    probe = torch.tensor(int(value), dtype=torch.int64, device=values.device)
    return int(torch.searchsorted(values, probe).item())


def pair_tile_fragments(
    groups: SpinGroups,
    tile: PairTileBounds,
) -> tuple[PairFragment, ...]:
    """The per-block rectangles ``tile`` occupies in ``groups``.

    Every block of the decomposition is intersected with the tile's uid slabs;
    the intersection of two sorted uid arrays is a contiguous position range,
    so each surviving block contributes one rectangle whose pair indices are a
    slice of the block's row-major ``pair_idx`` grid.

    Args:
        groups: One axis of a :class:`PairBlocks` decomposition (``by_beta``
            for the α channels, ``by_alpha`` for the β channels).
        tile: The rank's tile bounds.

    Returns:
        The non-empty fragments, in block order. Their pair indices are
        disjoint and their union is :attr:`PairTileBounds.pair_index`.
    """
    fragments: list[PairFragment] = []
    for group_id, group in enumerate(groups.groups):
        if group.pair_idx is None:
            fragments.extend(_structural_fragments(group_id, group, tile))
            continue
        alpha_uids = group.alpha_uids
        beta_uids = group.beta_uids
        row_start = _lower_bound(alpha_uids, tile.alpha_start)
        row_stop = _lower_bound(alpha_uids, tile.alpha_stop)
        col_start = _lower_bound(beta_uids, tile.beta_start)
        col_stop = _lower_bound(beta_uids, tile.beta_stop)
        if row_stop <= row_start or col_stop <= col_start:
            continue
        grid = group.pair_idx.view(int(alpha_uids.numel()), int(beta_uids.numel()))
        pair_index = grid[row_start:row_stop, col_start:col_stop].reshape(-1).contiguous()
        fragments.append(
            PairFragment(
                group=group_id,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                pair_index=pair_index,
            )
        )
    return tuple(fragments)


def _structural_fragments(group_id: int, group, tile: PairTileBounds) -> list[PairFragment]:
    """Per-coarse-block fragments of a structural group intersecting a tile.

    The group's uid arrays inside a sub-block are ascending, so each uid slab
    intersection is one contiguous position range. The fragment's global pair
    indices come from the group's arithmetic layout.

    Args:
        group_id: Index of the group in its ``SpinGroups`` tuple.
        group: The structural group (``pair_idx is None``).
        tile: The rank's tile bounds.

    Returns:
        The non-empty fragments of this group.
    """
    fragments: list[PairFragment] = []
    for sub_block in group.sub_blocks:
        row_uids = group.alpha_uids[sub_block.row_start : sub_block.row_stop]
        col_uids = group.beta_uids[sub_block.col_start : sub_block.col_stop]
        device = row_uids.device
        row_lo = int(torch.searchsorted(row_uids, torch.tensor(tile.alpha_start, device=device)))
        row_hi = int(torch.searchsorted(row_uids, torch.tensor(tile.alpha_stop, device=device)))
        col_lo = int(torch.searchsorted(col_uids, torch.tensor(tile.beta_start, device=device)))
        col_hi = int(torch.searchsorted(col_uids, torch.tensor(tile.beta_stop, device=device)))
        if row_hi <= row_lo or col_hi <= col_lo:
            continue
        row_start = sub_block.row_start + row_lo
        row_stop = sub_block.row_start + row_hi
        col_start = sub_block.col_start + col_lo
        col_stop = sub_block.col_start + col_hi
        fragments.append(
            PairFragment(
                group=group_id,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                pair_index=group_rect_indices(
                    group, row_start, row_stop, col_start, col_stop
                ).contiguous(),
            )
        )
    return fragments
