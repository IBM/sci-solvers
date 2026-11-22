"""Tests for the 2-D tile bounds of a flat spin-orbital ("spinless") basis."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.spinless_helpers import (
    dets_to_python_ints,
    spinless_basis_from_array,
)
from torch_sci_solver.torch_impl.spinless_tiles import (
    half_string_uids,
    spinless_tile_bounds,
    spinless_tile_partition,
)

# (grid, world_size) pairs: a square grid and a tall one with a non-square shape.
_GRIDS = [((2, 2), 4), ((2, 3), 6)]


def _full_basis(n_spinorb: int, n_particles: int) -> Tensor:
    """All ``n_particles``-from-``n_spinorb`` determinant bitmasks as a limb matrix."""
    strs = [
        sum(1 << bit for bit in combo)
        for combo in itertools.combinations(range(n_spinorb), n_particles)
    ]
    return spinless_basis_from_array(strs, n_spinorb, n_particles).dets


def _sampled_basis(n_spinorb: int, n_particles: int, count: int, seed: int) -> Tensor:
    """A few hundred random ``n_particles``-from-``n_spinorb`` determinants."""
    rng = np.random.default_rng(seed)
    strs: set[int] = set()
    while len(strs) < count:
        combo = rng.choice(n_spinorb, n_particles, replace=False)
        strs.add(sum(1 << int(bit) for bit in combo))
    return spinless_basis_from_array(sorted(strs), n_spinorb, n_particles).dets


def _single_limb() -> tuple[Tensor, int]:
    """A full ten-orbital basis whose halves both fit in one limb."""
    return _full_basis(10, 3), 5


def _multi_limb() -> tuple[Tensor, int]:
    """A 130-orbital sample with a 64-aligned split (row half spans two limbs)."""
    return _sampled_basis(130, 5, 300, seed=11), 64


@pytest.fixture(params=[_single_limb, _multi_limb], ids=["single-limb", "multi-limb"])
def basis(request: pytest.FixtureRequest) -> tuple[Tensor, int]:
    """A synthetic spinless determinant list and its split point."""
    return request.param()


def _sorted_numbers(limbs: Tensor) -> list[int]:
    """Half-string values as Python ints, in the matrix's row order."""
    return [int(value) for value in dets_to_python_ints(limbs)]


def _assert_half_strings_match(
    dets: Tensor,
    split: int,
    row_uids: Tensor,
    col_uids: Tensor,
    unique_rows: Tensor,
    unique_cols: Tensor,
) -> None:
    """Check every uid resolves to its determinant's high/low half bit pattern."""
    dets_ints = dets_to_python_ints(dets)
    row_ints = dets_to_python_ints(unique_rows)
    col_ints = dets_to_python_ints(unique_cols)
    col_mask = (1 << split) - 1
    assert [int(col_ints[uid]) for uid in col_uids.tolist()] == [
        int(value) & col_mask for value in dets_ints
    ]
    assert [int(row_ints[uid]) for uid in row_uids.tolist()] == [
        int(value) >> split for value in dets_ints
    ]


def test_half_string_uids_match_brute_force(basis: tuple[Tensor, int]) -> None:
    """Uids resolve to the low/high halves, deduplicated and sorted."""
    dets, split = basis
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    n_dets = int(dets.shape[0])
    assert row_uids.shape == (n_dets,)
    assert col_uids.shape == (n_dets,)
    assert row_uids.dtype == torch.int64
    assert col_uids.dtype == torch.int64

    _assert_half_strings_match(dets, split, row_uids, col_uids, unique_rows, unique_cols)

    dets_ints = dets_to_python_ints(dets)
    col_mask = (1 << split) - 1
    assert int(unique_rows.shape[0]) == len({int(value) >> split for value in dets_ints})
    assert int(unique_cols.shape[0]) == len({int(value) & col_mask for value in dets_ints})
    for unique in (unique_rows, unique_cols):
        numbers = _sorted_numbers(unique)
        assert numbers == sorted(set(numbers))
    assert int(row_uids.min()) >= 0
    assert int(row_uids.max()) < int(unique_rows.shape[0])
    assert int(col_uids.min()) >= 0
    assert int(col_uids.max()) < int(unique_cols.shape[0])


@pytest.mark.parametrize(("grid", "world_size"), _GRIDS)
def test_tiles_cover_the_basis_disjointly(
    basis: tuple[Tensor, int],
    grid: tuple[int, int],
    world_size: int,
) -> None:
    """Ranks tile the whole basis exactly once, matching a brute-force uid mask."""
    dets, split = basis
    n_dets = int(dets.shape[0])
    row_uids, col_uids, _, _ = half_string_uids(dets, split)
    tile, bounds = spinless_tile_partition(dets, split, world_size, grid=grid)
    assert len(bounds) == world_size
    assert tile.world_size == world_size

    owned = []
    for rank, bound in enumerate(bounds):
        mask = (
            (row_uids >= bound.row_start)
            & (row_uids < bound.row_stop)
            & (col_uids >= bound.col_start)
            & (col_uids < bound.col_stop)
        )
        expected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        assert torch.equal(bound.determinant_index, expected)
        assert bound.local_size == int(expected.numel())
        assert bound.rank == rank
        assert bound.world_size == world_size
        assert bound.n_dets == n_dets
        assert bound.split == split
        assert bound.n_row_uids == tile.n_rows
        assert bound.n_col_uids == tile.n_cols

        single = spinless_tile_bounds(tile, dets, split, rank)
        assert torch.equal(single.determinant_index, bound.determinant_index)
        assert (single.row_start, single.row_stop) == (bound.row_start, bound.row_stop)
        assert (single.col_start, single.col_stop) == (bound.col_start, bound.col_stop)
        assert (single.row_slab, single.col_slab) == (bound.row_slab, bound.col_slab)
        owned.append(bound.determinant_index)

    default = spinless_tile_bounds(tile, dets, split)
    assert default.rank == 0
    assert torch.equal(default.determinant_index, bounds[0].determinant_index)

    union = torch.cat(owned)
    assert torch.equal(torch.sort(union).values, torch.arange(n_dets))
    assert torch.equal(torch.bincount(union, minlength=n_dets), torch.ones(n_dets))


@pytest.mark.parametrize(("grid", "world_size"), _GRIDS)
def test_tile_global_local_maps_round_trip(
    basis: tuple[Tensor, int],
    grid: tuple[int, int],
    world_size: int,
) -> None:
    """Local-to-global-to-local is the identity; foreign determinant positions raise."""
    dets, split = basis
    n_dets = int(dets.shape[0])
    _, bounds = spinless_tile_partition(dets, split, world_size, grid=grid)
    for bound in bounds:
        local = torch.arange(bound.local_size)
        assert torch.equal(bound.local_of_global(bound.global_index(local)), local)
        for index in bound.determinant_index.tolist():
            assert bound.owns(index)
        owned = torch.zeros(n_dets, dtype=torch.bool)
        owned[bound.determinant_index] = True
        foreign = torch.nonzero(~owned, as_tuple=False).reshape(-1)
        if int(foreign.numel()) == 0:
            continue  # every rank owns every determinant, so no foreign index exists
        assert not bound.owns(int(foreign[0]))
        with pytest.raises(ValueError, match="outside this tile"):
            bound.local_of_global(foreign[:1])


def test_empty_slabs_own_nothing() -> None:
    """A tall grid leaves ranks with no determinant, and ``owns`` says so."""
    dets = _full_basis(10, 1)
    split = 5
    _, bounds = spinless_tile_partition(dets, split, 8, grid=(8, 1))
    empty = [bound for bound in bounds if bound.is_empty]
    filled = [bound for bound in bounds if not bound.is_empty]
    assert empty
    assert filled
    for bound in empty:
        assert bound.local_size == 0
        assert int(bound.determinant_index.numel()) == 0
        with pytest.raises(ValueError, match="outside this tile"):
            bound.local_of_global(torch.tensor([0]))
    foreign = int(filled[0].determinant_index[0])
    for bound in empty:
        assert not bound.owns(foreign)
    union = torch.cat([bound.determinant_index for bound in bounds])
    assert torch.equal(torch.sort(union).values, torch.arange(int(dets.shape[0])))


def test_bit_at_or_above_64_lands_in_the_row_half() -> None:
    """With ``split = 64`` a bit at position >= 64 must be part of the row half."""
    one = (1 << 3) | (1 << 70)
    two = (1 << 5) | (1 << 70)
    dets = spinless_basis_from_array([one, two], 130, 2).dets
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, 64)
    assert [int(value) for value in dets_to_python_ints(unique_rows)] == [1 << (70 - 64)]
    assert [int(value) for value in dets_to_python_ints(unique_cols)] == [1 << 3, 1 << 5]
    assert row_uids.tolist() == [0, 0]
    assert col_uids.tolist() == [0, 1]


def test_unaligned_split_extracts_both_halves() -> None:
    """A split inside a limb still masks/shifts the cross-limb bits correctly."""
    strs = [
        (1 << 2) | (1 << 69) | (1 << 129),
        (1 << 63) | (1 << 70) | (1 << 128),
        # A set sign bit in the split limb and a zero carry from the next limb.
        (1 << 68) | (1 << 127),
    ]
    dets = spinless_basis_from_array(strs, 130, 3).dets
    split = 70
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    _assert_half_strings_match(dets, split, row_uids, col_uids, unique_rows, unique_cols)


def test_half_string_uids_rejects_bad_input() -> None:
    """Malformed limb matrices and split points are rejected."""
    dets = _full_basis(10, 3)
    with pytest.raises(ValueError, match="split"):
        half_string_uids(dets, -1)
    with pytest.raises(ValueError, match="split"):
        half_string_uids(dets, 65)
    with pytest.raises(ValueError, match="2-D"):
        half_string_uids(dets.reshape(-1), 5)


def test_tile_bounds_reject_mismatched_partition(basis: tuple[Tensor, int]) -> None:
    """A partition over other uid axes, or an out-of-range rank, is rejected."""
    dets, split = basis
    tile, _ = spinless_tile_partition(dets, split, 4, grid=(2, 2))
    wrong = partition_tiles(tile.n_rows + 1, tile.n_cols, 4, grid=(2, 2))
    with pytest.raises(ValueError, match="does not match the uid axes"):
        spinless_tile_bounds(wrong, dets, split)
    with pytest.raises(ValueError, match="rank"):
        spinless_tile_bounds(tile, dets, split, rank=4)
