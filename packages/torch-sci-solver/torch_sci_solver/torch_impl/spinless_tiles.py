"""2-D tile foundation for a flat spin-orbital ("spinless") selected-CI vector.

A spinless basis is an explicit, sorted list of ``N`` flat ``n_spinorb``-bit
determinants (:class:`~torch_sci_solver.torch_impl.spinless_helpers.
SpinlessBasis`), so unlike the TPB ``α × β`` product there is no row/column
factorization to partition directly. Splitting every determinant's bitmask at
a chosen position ``split`` recovers two "half strings": bits ``[0, split)``
(the column half) and bits ``[split, n_spinorb)`` (the row half). Determinants
with the same half string share a uid on that axis, so the basis becomes a
sparse set of ``(row_uid, col_uid)`` points in an
``n_row_uids × n_col_uids`` grid.

A :class:`~torch_sci_solver.torch_impl.distributed_davidson.TilePartition`
over that grid gives each rank a rectangle of *uid* space. The rank owns every
determinant whose ``(row_uid, col_uid)`` falls inside the rectangle: a
scattered, sorted subset of global determinant positions, because unlike a
Cartesian product the explicit basis does not lay those determinants out
contiguously. This module supplies the foundation for the spinless
owner-computes matvec:

* :func:`half_string_uids` — the uid of each determinant's row/column half and
  the sorted unique half strings.
* :class:`SpinlessTileBounds` — the uid slabs one rank owns, the sorted global
  determinant positions it owns, and the local↔global position map.
* :func:`spinless_tile_bounds` / :func:`spinless_tile_partition` — build one
  rank's bounds, or the whole partition with every rank's bounds.

Notes
-----
A determinant ``k`` belongs to a tile when
``row_start ≤ row_uid(k) < row_stop`` and
``col_start ≤ col_uid(k) < col_stop``. Slabs are products of contiguous *uid*
ranges, so every determinant belongs to exactly one tile of the grid and the
union over ranks is the whole basis.

Uids are the positions of the half strings in their sorted unique axis, which
keeps them int64 for arbitrarily wide spin-orbital spaces; the unique axes
themselves are ``(n_unique, n_limbs_half)`` limb matrices holding the patterns.
For a 64-aligned ``split`` each half lies inside whole limbs, so the halves are
plain limb slices; a non-aligned split masks and shifts the limb containing the
split bit.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.distributed_davidson import TilePartition, partition_tiles
from torch_sci_solver.torch_impl.pair_helpers import _found_in_multilimb

# Sign bit of an int64 limb. XOR-ing a limb with it turns signed comparison of
# the limb into unsigned comparison, which is the numeric order of a bit
# pattern stored in a two's-complement int64.
_INT64_SIGN_BIT = -(1 << 63)


def _split_limbs(dets: Tensor, split: int) -> tuple[Tensor, Tensor]:
    """Split a determinant limb matrix into row-half and column-half limbs.

    Args:
        dets: ``(N, n_limbs)`` int64 determinant limb matrix; limb ``k`` holds
            bits ``[64k, 64k + 64)``.
        split: Split bit position; bits below it are the column half.

    Returns:
        ``(row_limbs, col_limbs)`` with the same row count as ``dets``. An
        empty half is represented as a single zero limb so the unique axis is
        still one-dimensional.

    Raises:
        ValueError: ``dets`` is not 2-D, or ``split`` is outside
            ``[0, 64 * n_limbs]``.
    """
    if dets.dim() != 2:
        raise ValueError(f"dets must be a 2-D (N, n_limbs) tensor, got shape {tuple(dets.shape)}")
    n_limbs = int(dets.shape[1])
    split = int(split)
    if not 0 <= split <= 64 * n_limbs:
        raise ValueError(f"split must be in [0, {64 * n_limbs}], got {split}")
    whole, rem = divmod(split, 64)
    if rem == 0:
        row = dets[:, whole:].contiguous()
        col = dets[:, :whole].contiguous()
    else:
        # The split bit sits inside limb ``whole``: its low ``rem`` bits belong
        # to the column half, its high bits to the row half. The column half
        # keeps the original limb packing; the row half starts ``rem`` bits
        # into limb ``whole``, so every later limb must be re-aligned (shifted
        # left by ``64 - rem``, carrying its low bits into the previous output
        # limb). Masking after the arithmetic right shift keeps it logical.
        low_mask = (1 << rem) - 1
        high_mask = (1 << (64 - rem)) - 1
        col = torch.cat([dets[:, :whole], dets[:, whole : whole + 1] & low_mask], dim=1)
        n_row_limbs = n_limbs - whole
        row_limbs = []
        for j in range(n_row_limbs):
            source = whole + j
            # ``high_mask`` clears the arithmetic shift's sign extension; the
            # carry then fills those ``rem`` bits from the next limb's low bits.
            limb = (dets[:, source] >> rem) & high_mask
            if source + 1 < n_limbs:
                limb = limb | ((dets[:, source + 1] & low_mask) << (64 - rem))
            row_limbs.append(limb)
        row = torch.stack(row_limbs, dim=1)
        row = row.contiguous()
        col = col.contiguous()
    if col.shape[1] == 0:
        col = torch.zeros((dets.shape[0], 1), dtype=dets.dtype, device=dets.device)
    if row.shape[1] == 0:
        row = torch.zeros((dets.shape[0], 1), dtype=dets.dtype, device=dets.device)
    return row, col


def _sorted_unique_uids(halves: Tensor) -> tuple[Tensor, Tensor]:
    """Map half strings to uids in their sorted unique axis.

    Args:
        halves: ``(N, n_limbs_half)`` half-string limb matrix, low limb first.

    Returns:
        ``(uid_per_row, unique)`` where ``uid_per_row`` is an ``(N,)`` int64
        tensor of positions into ``unique`` and ``unique`` is the
        ``(n_unique, n_limbs_half)`` sorted, deduplicated half-string matrix.

    Notes:
        ``torch.unique(dim=0)`` compares rows lexicographically, so the limb
        columns are reversed (most-significant limb first) before deduplication
        and each limb is XOR-ed with its sign bit to compare as unsigned; the
        result is then the numeric order of the half string.
    """
    sign = torch.tensor(_INT64_SIGN_BIT, dtype=torch.int64, device=halves.device)
    key = halves.flip(1) ^ sign
    unique_key, inverse = torch.unique(key, dim=0, sorted=True, return_inverse=True)
    unique = (unique_key ^ sign).flip(1).contiguous()
    return inverse.to(torch.int64), unique


def _unique_axis(halves: Tensor) -> Tensor:
    """Sorted unique half strings of a limb matrix, in numeric order.

    Args:
        halves: ``(N, n_limbs_half)`` half-string limb matrix, low limb first.

    Returns:
        The ``(n_unique, n_limbs_half)`` sorted, deduplicated half-string
        matrix, in the same order :func:`_sorted_unique_uids` produces.

    Notes:
        Uses the same sort-key trick as :func:`_sorted_unique_uids`: the limb
        columns are flipped so the most significant limb compares first, and
        each limb is XOR-ed with its sign bit so the comparison is unsigned.
    """
    sign = torch.tensor(_INT64_SIGN_BIT, dtype=torch.int64, device=halves.device)
    key = halves.flip(1) ^ sign
    unique_key = torch.unique(key, dim=0, sorted=True)
    return (unique_key ^ sign).flip(1).contiguous()


def half_string_axes(dets: Tensor, split: int) -> tuple[Tensor, Tensor]:
    """Sorted unique row/column half strings of a determinant list.

    Companion of :func:`half_string_uids` for callers that need the uid axes
    but not the ``N``-long per-determinant uid tensors, e.g. to resolve the
    half strings of a subset of determinants later with
    :func:`half_string_uids_of`.

    Args:
        dets: ``(N, n_limbs)`` int64 determinant limb matrix.
        split: Split bit position; bits below it are the column half.

    Returns:
        ``(unique_rows, unique_cols)``, each ``(n_unique, n_limbs_half)`` and
        sorted in the numeric order :func:`half_string_uids` uses.

    Raises:
        ValueError: ``dets`` is not 2-D, or ``split`` is outside
            ``[0, 64 * n_limbs]``.
    """
    row_limbs, col_limbs = _split_limbs(dets, split)
    return _unique_axis(row_limbs), _unique_axis(col_limbs)


def _uid_in_axis(halves: Tensor, unique: Tensor) -> Tensor:
    """Positions of half-string rows in a precomputed sorted unique axis.

    Args:
        halves: ``(N, n_limbs_half)`` half-string limb matrix.
        unique: ``(M, n_limbs_half)`` sorted unique axis from
            :func:`_unique_axis`.

    Returns:
        ``(N,)`` int64 uid of every row.

    Raises:
        ValueError: a row is absent from ``unique``.

    Notes:
        Both sides are flipped (most significant limb first) and XOR-ed with
        the int64 sign bit so that
        :func:`~torch_sci_solver.torch_impl.pair_helpers._found_in_multilimb`
        compares the rows in the unsigned numeric order :func:`_unique_axis`
        sorts them in.
    """
    sign = torch.tensor(_INT64_SIGN_BIT, dtype=torch.int64, device=halves.device)
    found, uid = _found_in_multilimb(unique.flip(1) ^ sign, halves.flip(1) ^ sign)
    if not bool(found.all()):
        raise ValueError("half string is absent from the unique half-string axis")
    return uid


def half_string_uids_of(
    dets: Tensor, split: int, unique_rows: Tensor, unique_cols: Tensor
) -> tuple[Tensor, Tensor]:
    """Row/column uids of determinants resolved against precomputed axes.

    Args:
        dets: ``(N, n_limbs)`` int64 determinant limb matrix.
        split: Split bit position; bits below it are the column half.
        unique_rows: Sorted row half-string axis from :func:`half_string_axes`.
        unique_cols: Sorted column half-string axis from
            :func:`half_string_axes`.

    Returns:
        ``(row_uids, col_uids)``, each ``(N,)`` int64 positions in the
        corresponding axis.

    Raises:
        ValueError: ``dets`` is not 2-D, the split is out of range, or a half
            string is absent from its axis.
    """
    row_limbs, col_limbs = _split_limbs(dets, split)
    return _uid_in_axis(row_limbs, unique_rows), _uid_in_axis(col_limbs, unique_cols)


def half_string_uids(dets: Tensor, split: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split determinants into row/column half-string uids.

    Bits ``[0, split)`` form the column half and bits ``[split, n_spinorb)``
    the row half. Identical half strings share a uid, and a uid is the position
    of its half string in the sorted unique axis (so it stays int64 even when a
    half string itself needs several limbs).

    Args:
        dets: ``(N, n_limbs)`` int64 determinant limb matrix; limb ``k`` holds
            bits ``[64k, 64k + 64)``.
        split: Split bit position in ``[0, 64 * n_limbs]``. The caller should
            pass ``split <= n_spinorb``; a 64-aligned value keeps each half
            inside whole limbs.

    Returns:
        ``(row_uids_per_det, col_uids_per_det, unique_rows, unique_cols)``.
        The per-determinant uids are ``(N,)`` int64 position tensors;
        ``unique_rows`` / ``unique_cols`` are the sorted unique half-string
        limb matrices, of shape ``(n_unique, n_limbs_half)``.

    Raises:
        ValueError: ``dets`` is not 2-D, or ``split`` is outside
            ``[0, 64 * n_limbs]``.
    """
    row_limbs, col_limbs = _split_limbs(dets, split)
    row_uids, unique_rows = _sorted_unique_uids(row_limbs)
    col_uids, unique_cols = _sorted_unique_uids(col_limbs)
    return row_uids, col_uids, unique_rows, unique_cols


@dataclass(frozen=True)
class SpinlessTileBounds:
    """The ``row_uid × col_uid`` slab one rank owns of a spinless basis.

    Attributes:
        n_dets: Global determinant count.
        split: Bit position separating the column half ``[0, split)`` from the
            row half ``[split, n_spinorb)``.
        n_row_uids: Number of unique row half strings (the row axis of the grid).
        n_col_uids: Number of unique column half strings (the column axis).
        rank: This tile's rank.
        world_size: Number of ranks.
        row_slab: Index of the owned row-uid slab.
        col_slab: Index of the owned column-uid slab.
        row_start: First owned row uid (inclusive).
        row_stop: One past the last owned row uid.
        col_start: First owned column uid (inclusive).
        col_stop: One past the last owned column uid.
        determinant_index: ``(local_size,)`` sorted global determinant positions
            this rank owns.
    """

    n_dets: int
    split: int
    n_row_uids: int
    n_col_uids: int
    rank: int
    world_size: int
    row_slab: int
    col_slab: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    determinant_index: Tensor

    @property
    def n(self) -> int:
        """Global determinant count (alias of :attr:`n_dets`).

        The distributed Davidson driver accepts any bounds object exposing
        ``n``, ``world_size``, ``rank`` and ``local_size``; this alias lets a
        scattered tile feed it directly.
        """
        return self.n_dets

    @property
    def pair_index(self) -> Tensor:
        """Alias of :attr:`determinant_index` for the scattered flat layout.

        ``davidson_distributed_local`` duck-types a scattered 1-D block through
        the ``pair_index`` name GDB introduced; a spinless tile's owned global
        positions play exactly that role, so exposing them here lets the tile
        bounds drive the seed and one-hot helpers unchanged.
        """
        return self.determinant_index

    @property
    def local_size(self) -> int:
        """Number of owned determinants."""
        return int(self.determinant_index.numel())

    @property
    def is_empty(self) -> bool:
        """Whether this rank owns no determinant."""
        return self.local_size == 0

    def global_index(self, local: Tensor) -> Tensor:
        """Map tile-local positions to global determinant positions.

        Args:
            local: Tile-local positions in ``[0, local_size)``.

        Returns:
            The global determinant positions at those local offsets.
        """
        index = torch.as_tensor(local, dtype=torch.int64, device=self.determinant_index.device)
        return self.determinant_index[index]

    def local_of_global(self, global_index: Tensor) -> Tensor:
        """Map owned global determinant positions back to tile-local positions.

        Args:
            global_index: Global determinant positions contained in
                :attr:`determinant_index`.

        Returns:
            The tile-local offset of each position.

        Raises:
            ValueError: an index is outside this tile.
        """
        index = torch.as_tensor(
            global_index, dtype=torch.int64, device=self.determinant_index.device
        )
        local = torch.searchsorted(self.determinant_index, index)
        if bool((local >= self.local_size).any()):
            raise ValueError("global determinant index outside this tile")
        found = self.determinant_index[local]
        if not torch.equal(found, index):
            raise ValueError("global determinant index outside this tile")
        return local

    def owns(self, global_index: int) -> bool:
        """Whether a global determinant position belongs to this tile.

        Args:
            global_index: Global determinant position to test.

        Returns:
            ``True`` when this rank owns the determinant.
        """
        probe = torch.tensor(
            [int(global_index)], dtype=torch.int64, device=self.determinant_index.device
        )
        local = torch.searchsorted(self.determinant_index, probe)
        position = int(local[0])
        return position < self.local_size and int(self.determinant_index[position]) == int(
            global_index
        )


def _bounds_from_uids(
    tile: TilePartition,
    row_uids: Tensor,
    col_uids: Tensor,
    unique_rows: Tensor,
    unique_cols: Tensor,
    split: int,
    rank: int,
) -> SpinlessTileBounds:
    """Build one rank's bounds from an already computed uid decomposition.

    Args:
        tile: A tile partition sized ``(n_row_uids, n_col_uids)``.
        row_uids: ``(N,)`` row uid of every determinant.
        col_uids: ``(N,)`` column uid of every determinant.
        unique_rows: ``(n_row_uids, n_limbs_half)`` row half-string matrix.
        unique_cols: ``(n_col_uids, n_limbs_half)`` column half-string matrix.
        split: Split bit position.
        rank: Rank to describe.

    Returns:
        The rank's tile bounds.

    Raises:
        ValueError: the partition does not match the uid axes, or ``rank`` is
            outside ``[0, tile.world_size)``.
    """
    n_rows = int(unique_rows.shape[0])
    n_cols = int(unique_cols.shape[0])
    if tile.n_rows != n_rows or tile.n_cols != n_cols:
        raise ValueError(
            f"tile partition ({tile.n_rows}, {tile.n_cols}) does not match the uid axes "
            f"({n_rows}, {n_cols})"
        )
    rank = int(rank)
    bound = tile.for_rank(rank)
    mask = (
        (row_uids >= bound.row_start)
        & (row_uids < bound.row_stop)
        & (col_uids >= bound.col_start)
        & (col_uids < bound.col_stop)
    )
    determinant_index = torch.nonzero(mask, as_tuple=False).reshape(-1)
    return SpinlessTileBounds(
        n_dets=int(row_uids.shape[0]),
        split=int(split),
        n_row_uids=n_rows,
        n_col_uids=n_cols,
        rank=rank,
        world_size=tile.world_size,
        row_slab=bound.row_slab,
        col_slab=bound.col_slab,
        row_start=bound.row_start,
        row_stop=bound.row_stop,
        col_start=bound.col_start,
        col_stop=bound.col_stop,
        determinant_index=determinant_index,
    )


def spinless_tile_bounds(
    tile: TilePartition,
    dets: Tensor,
    split: int,
    rank: int = 0,
) -> SpinlessTileBounds:
    """Build one rank's :class:`SpinlessTileBounds` over the uid axes.

    Args:
        tile: A tile partition sized ``(n_row_uids, n_col_uids)``.
        dets: ``(N, n_limbs)`` determinant limb matrix the partition indexes.
        split: Split bit position used to derive the uid axes.
        rank: Rank to describe; defaults to ``0``.

    Returns:
        The rank's tile bounds, with its owned global positions sorted.

    Raises:
        ValueError: the partition does not match the uid axes, or ``rank`` is
            outside ``[0, tile.world_size)``.
    """
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    return _bounds_from_uids(tile, row_uids, col_uids, unique_rows, unique_cols, split, rank)


def spinless_tile_partition(
    dets: Tensor,
    split: int,
    world_size: int,
    grid: tuple[int, int] | None = None,
) -> tuple[TilePartition, tuple[SpinlessTileBounds, ...]]:
    """Partition a spinless basis over a ``(row_uid, col_uid)`` tile grid.

    Extracts the half-string uids once, cuts both uid axes into a
    ``world_size``-rank tile grid, and builds every rank's bounds from that one
    decomposition. Both axes are cut on the per-uid determinant counts rather
    than the uid counts: a selected basis concentrates many determinants on a
    few uids, so an even uid cut would leave ranks with very different
    determinant counts, while the auto-tiling model assumes a balanced ``N/P``
    per rank. The counts are read off the uid map already computed here.

    Args:
        dets: ``(N, n_limbs)`` determinant limb matrix.
        split: Split bit position used to derive the uid axes.
        world_size: Number of ranks.
        grid: Optional explicit ``(p_rows, p_cols)``; defaults to
            :func:`~torch_sci_solver.torch_impl.distributed_davidson.
            resolve_tile_grid`.

    Returns:
        ``(tile, bounds)`` where ``tile`` is the
        :class:`~torch_sci_solver.torch_impl.distributed_davidson.
        TilePartition` over the uid axes and ``bounds`` holds one
        :class:`SpinlessTileBounds` per rank, in rank order.

    Raises:
        ValueError: non-positive ``world_size``, an invalid ``grid`` whose
            entries do not multiply to ``world_size``, or an invalid split.
    """
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    n_rows = int(unique_rows.shape[0])
    n_cols = int(unique_cols.shape[0])
    row_weights = torch.bincount(row_uids, minlength=n_rows).to(torch.float64)
    col_weights = torch.bincount(col_uids, minlength=n_cols).to(torch.float64)
    tile = partition_tiles(
        n_rows,
        n_cols,
        world_size,
        grid=grid,
        row_weights=row_weights,
        col_weights=col_weights,
    )
    bounds = tuple(
        _bounds_from_uids(tile, row_uids, col_uids, unique_rows, unique_cols, split, rank)
        for rank in range(tile.world_size)
    )
    return tile, bounds


def resolve_auto_spinless_tiling(
    *,
    world_size: int,
    n_dets: int,
    block_size: int = 8,
    itemsize: int = 8,
    budget_bytes: int | None,
) -> bool:
    """Whether a spinless solve should use the tile grid under ``tile="auto"``.

    Tiling trades compute for peak memory: the 1-D owner-computes matvec
    all-gathers one full input vector per matvec, while the tile operator stages
    at most one peer's needed ket block but enumerates the transpose table
    (measured at 2-19x the local enumeration). The policy therefore keeps the
    1-D gather unless it does not fit.

    Args:
        world_size: Number of ranks.
        n_dets: Global determinant count.
        block_size: Nominal Davidson block size.
        itemsize: Bytes per vector element.
        budget_bytes: Usable device memory per rank, or ``None`` when unknown
            (CPU runs).

    Returns:
        ``True`` when a tile grid should be used; ``False`` keeps the 1-D
        owner-computes gather.

    Notes
    -----
    The 1-D peak is one gathered input vector plus the Davidson block terms;
    the tile peak is the block terms plus one staged ket block, measured at
    ``N/P`` for ``P >= 2``:

    .. math::

        B_{1D} = N s + (b + 2) \\frac{N}{P} s ,
        \\qquad B_{tile} = (b + 3) \\frac{N}{P} s .

    Tiling is chosen when the 1-D peak exceeds half the device budget,
    matching the TPB and GDB policies. With no byte budget (CPU runs) or a
    single rank the faster 1-D gather is kept. The tile partition cuts on the
    per-uid determinant counts, so its slabs carry a balanced share of the
    ``N/P`` determinants this model assumes.
    """
    if world_size <= 1 or n_dets <= 0:
        return False
    if budget_bytes is None:
        return False
    one_d_bytes = (1.0 + (block_size + 2.0) / world_size) * n_dets * itemsize
    return one_d_bytes > 0.5 * budget_bytes
