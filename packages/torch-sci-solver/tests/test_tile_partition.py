"""CPU checks for the 2-D tile partition (alpha-row slab x beta-column slab).

Pure arithmetic, no process group: grid selection, tile coverage of an
``(n_rows, n_cols)`` matrix, the local/global flat-index maps, the weighted and
even boundary cuts, and the ``p_beta == 1`` degeneration of a
:class:`RowPartition`.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pytest
import torch
from torch_sci_solver.torch_impl.distributed_davidson import (
    TileGrid,
    TilePartition,
    _device_memory_budget,
    _RowIndexLayout,
    _TileIndexLayout,
    global_smallest_hii,
    hash_uniform,
    partition_rows_explicit,
    partition_tiles,
    resolve_tile_grid,
)
from torch_sci_solver.torch_impl.distributed_matvec import (
    alpha_row_weights,
    beta_col_weights,
)


def test_tile_grid_rank_mapping_and_validation() -> None:
    """Rank <-> slab arithmetic round-trips and bad grids are rejected."""
    grid = TileGrid(
        n_rows=7,
        n_cols=5,
        p_alpha=2,
        p_beta=3,
        alpha_bounds=((0, 3), (3, 7)),
        beta_bounds=((0, 2), (2, 4), (4, 5)),
    )
    assert grid.world_size == 6
    for rank in range(grid.world_size):
        assert grid.rank_of(grid.row_slab(rank), grid.col_slab(rank)) == rank
    assert [grid.row_slab(rank) for rank in range(6)] == [0, 0, 0, 1, 1, 1]
    assert [grid.col_slab(rank) for rank in range(6)] == [0, 1, 2, 0, 1, 2]
    assert grid.rows(1) == (3, 7)
    assert grid.cols(2) == (4, 5)

    with pytest.raises(ValueError):
        TileGrid(7, 5, 2, 3, ((0, 3),), ((0, 2), (2, 4), (4, 5)))  # too few row bounds
    with pytest.raises(ValueError):
        TileGrid(7, 5, 2, 1, ((0, 3), (3, 6)), ((0, 5),))  # does not cover [0, 7)
    with pytest.raises(ValueError):
        TileGrid(7, 5, 2, 1, ((0, 3), (2, 7)), ((0, 5),))  # not contiguous
    with pytest.raises(ValueError):
        grid.rank_of(2, 0)
    with pytest.raises(ValueError):
        grid.row_slab(6)


def test_resolve_tile_grid_balances_slabs() -> None:
    """The chosen grid minimises the larger slab and prefers square-ish splits."""
    assert resolve_tile_grid(32, 100, 100) == (4, 8)
    assert resolve_tile_grid(4, 100, 100) == (2, 2)
    assert resolve_tile_grid(4, 1000, 10) == (4, 1)
    assert resolve_tile_grid(7, 100, 100) == (1, 7)
    for world_size in (1, 2, 3, 4, 6, 8, 12, 32):
        p_alpha, p_beta = resolve_tile_grid(world_size, 50, 30)
        assert p_alpha * p_beta == world_size
    with pytest.raises(ValueError):
        resolve_tile_grid(0, 10, 10)


def test_tile_partition_from_boundaries_covers_uneven_rectangles() -> None:
    """Every determinant belongs to exactly one uneven tile."""
    partition = partition_tiles(7, 5, 4, (2, 2))
    assert [partition.rows_for_rank(rank) for rank in range(4)] == [
        (0, 3),
        (0, 3),
        (3, 7),
        (3, 7),
    ]
    assert [partition.cols_for_rank(rank) for rank in range(4)] == [
        (0, 2),
        (2, 5),
        (0, 2),
        (2, 5),
    ]
    assert [partition.for_rank(rank).local_size for rank in range(4)] == [6, 9, 8, 12]
    assert partition.max_local == 12
    assert sum(partition.for_rank(rank).local_size for rank in range(4)) == 7 * 5

    owned = torch.zeros(7 * 5, dtype=torch.int64)
    for rank in range(4):
        bound = partition.for_rank(rank)
        for index in range(7 * 5):
            owned[index] += int(bound.owns(index))
        local = torch.arange(bound.local_size)
        glob = bound.local_to_global(local)
        assert torch.equal(bound.global_to_local(glob), local)
    assert torch.equal(owned, torch.ones(7 * 5, dtype=torch.int64))

    with pytest.raises(ValueError):
        TilePartition.from_boundaries(7, 5, [0, 3, 6], [0, 5])  # rows do not cover
    with pytest.raises(ValueError):
        partition.for_rank(4)


def test_tile_bounds_map_tile_local_and_global_indices() -> None:
    """``local_to_global``/``global_to_local`` agree with the row-major layout."""
    partition = partition_tiles(3, 4, 2, (1, 2))
    bound = partition.for_rank(1)
    assert (bound.row_start, bound.row_stop) == (0, 3)
    assert (bound.col_start, bound.col_stop) == (2, 4)
    assert (bound.local_rows, bound.local_cols) == (3, 2)
    local = torch.arange(6)
    glob = bound.local_to_global(local)
    # row-major inside the tile, stride n_cols in the global matrix
    assert glob.tolist() == [2, 3, 6, 7, 10, 11]
    assert bound.global_to_local(glob).tolist() == [0, 1, 2, 3, 4, 5]
    assert bound.owns(11) and not bound.owns(1) and not bound.owns(15)


def test_tile_partition_from_row_partition_degenerates() -> None:
    """``p_beta == 1`` reproduces the 1-D row partition exactly."""
    rows = partition_rows_explicit(7, 3, [0, 2, 5, 7], 3)
    tiles = TilePartition.from_row_partition(rows)
    assert tiles.grid.p_alpha == 3 and tiles.grid.p_beta == 1
    for rank in range(3):
        assert tiles.rows_for_rank(rank) == rows.rows_for_rank(rank)
        assert tiles.cols_for_rank(rank) == (0, 3)
        assert tiles.for_rank(rank).local_size == rows.for_rank(rank).local_size
    assert tiles.owner_col_slab(torch.arange(3)).tolist() == [0, 0, 0]
    assert tiles.owner_row_slab(torch.arange(7)).tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_partition_tiles_even_and_weighted() -> None:
    """Even and cost-weighted cuts are contiguous, capped and deterministic."""
    even = partition_tiles(10, 6, 2, (1, 2))
    assert even.rows_for_rank(0) == (0, 10)
    assert even.cols_for_rank(0) == (0, 3)
    assert even.cols_for_rank(1) == (3, 6)

    weights = torch.tensor([1.0] * 5 + [10.0] * 5)
    weighted = partition_tiles(10, 6, 2, (2, 1), row_weights=weights)
    # the cheap rows are followed by the cost-dense tail in the second slab
    assert weighted.rows_for_rank(0) == (0, 7)
    assert weighted.rows_for_rank(1) == (7, 10)
    assert weighted.cols_for_rank(0) == (0, 6)
    assert weighted == partition_tiles(10, 6, 2, (2, 1), row_weights=weights)

    with pytest.raises(ValueError):
        partition_tiles(10, 6, 2, (2, 2))  # product != world_size
    with pytest.raises(ValueError):
        partition_tiles(10, 6, 2, (1, 2), row_weights=-torch.ones(10))


def test_index_layouts_agree_with_the_flat_block() -> None:
    """A ``p_beta == 1`` tile exposes the same indices and hash as a flat block."""
    rows = partition_rows_explicit(12, 1, [0, 6, 12], 2)
    flat = _RowIndexLayout(rows)
    tiles = TilePartition.from_boundaries(6, 2, [0, 3, 6], [0, 2])
    tile = _TileIndexLayout(tiles.for_rank(1))
    assert flat.global_indices(1).tolist() == list(range(6, 12))
    assert tile.global_indices(1).tolist() == flat.global_indices(1).tolist()
    assert flat.local_to_global(1, torch.arange(6)).tolist() == list(range(6, 12))
    assert torch.equal(hash_uniform(flat.global_indices(1)), hash_uniform(tile.global_indices(1)))

    # The row layout rejects local indices outside the block and the tile
    # layout rejects ranks other than its owner.
    with pytest.raises(ValueError):
        flat.local_to_global(1, torch.tensor([6]))
    with pytest.raises(ValueError):
        tile.local_size(0)


def test_global_smallest_hii_accepts_a_tile_layout() -> None:
    """The global top-k is layout-independent for a one-rank tile."""
    hii = torch.tensor([5.0, 1.0, 9.0, 2.0, 8.0, 0.5, 7.0, 3.0, 6.0, 4.0, 10.0, 11.0])
    tiles = TilePartition.from_boundaries(3, 4, [0, 3], [0, 4])
    pairs = global_smallest_hii(hii, _TileIndexLayout(tiles.for_rank(0)), k=3)
    assert [index for index, _ in pairs] == [5, 1, 3]
    assert [value for _, value in pairs] == [0.5, 1.0, 2.0]
    rows = partition_rows_explicit(12, 1, [0, 12], 1)
    assert global_smallest_hii(hii, rows, k=3) == pairs


def test_empty_slabs_never_claim_an_index() -> None:
    """More slabs than rows leaves empty tiles that own nothing."""
    partition = partition_tiles(2, 2, 4, (4, 1))
    assert [partition.rows_for_rank(rank) for rank in range(4)] == [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
    ]
    assert [partition.for_rank(rank).local_size for rank in range(4)] == [0, 2, 0, 2]
    assert partition.owner_row_slab(torch.tensor([0, 1])).tolist() == [1, 3]
    empty = partition.for_rank(0)
    assert empty.is_empty
    assert empty.local_to_global(torch.arange(0)).numel() == 0
    assert sum(partition.for_rank(rank).local_size for rank in range(4)) == 4


def test_device_memory_budget_uses_the_free_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The auto budget follows free memory, not 90% of the card's nominal size."""
    gib = 2**30
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # A card that still has 2 of 8 GiB free must report 90% of 2 GiB, because
    # the auto policies run after the replicated ERIs/helpers are allocated.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (2 * gib, 8 * gib))
    assert _device_memory_budget(torch.device("cuda", 0)) == int(0.9 * 2 * gib)

    # An empty card is capped by its total as well.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (8 * gib, 8 * gib))
    assert _device_memory_budget(torch.device("cuda", 0)) == int(0.9 * 8 * gib)

    # Off CUDA there is no device budget.
    assert _device_memory_budget(torch.device("cpu")) is None


def test_beta_col_weights_mirror_the_row_model() -> None:
    """The column cost is the row model with the two spin axes swapped."""
    adets = torch.tensor([1, 2, 4], dtype=torch.int64)
    bdets = torch.tensor([0, 3, 5, 7], dtype=torch.int64)
    assert torch.equal(
        beta_col_weights(adets, bdets, n_orb=8), alpha_row_weights(bdets, adets, n_orb=8)
    )


def test_weighted_tile_cut_balances_a_skewed_column_basis() -> None:
    """Skewed column costs move the cut away from the even one and balance it.

    The synthetic β list mixes occupations, so the per-column cost is strongly
    non-uniform: the cheap strings sit first and the expensive ones last. The
    even cut splits the expensive tail, while the weighted walk keeps it in one
    slab and lowers the most expensive slab's cost.
    """
    adets = torch.tensor([1], dtype=torch.int64)
    bdets = torch.tensor([0, 1, 2, 4, 8, 15, 23], dtype=torch.int64)
    weights = beta_col_weights(adets, bdets, n_orb=8)
    even = partition_tiles(1, bdets.numel(), 2, (1, 2))
    weighted = partition_tiles(1, bdets.numel(), 2, (1, 2), col_weights=weights)
    assert weighted.grid.beta_bounds != even.grid.beta_bounds

    def max_slab_cost(partition: TilePartition) -> float:
        """Largest weighted column slab of a partition."""
        return max(float(weights[start:stop].sum()) for start, stop in partition.grid.beta_bounds)

    assert max_slab_cost(weighted) < max_slab_cost(even)
