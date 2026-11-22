"""Tests for the 2-D tile bounds of an explicit pair (GDB) basis."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import pytest
import torch
from test_distributed_pairs_tile import _FullVectorExchange  # noqa: E402
from test_mult_pair_blocks import _real_integrals  # noqa: E402
from torch_sci_solver.torch_impl import distributed_pairs_tile as dpt
from torch_sci_solver.torch_impl.distributed_davidson import TilePartition, partition_tiles
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch, mult_pair_torch
from torch_sci_solver.torch_impl.pair_blocks import PairBlocks, SpinGroups, build_pair_blocks
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiles import (
    PairTileBounds,
    pair_axis_weights,
    pair_replicated_bytes,
    pair_tile_bounds,
    pair_tile_fragments,
    pair_tile_partition,
    resolve_auto_pair_tiling,
)
from torch_sci_solver.torch_impl.pairs import (
    PairBasis,
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)

# (grid, world_size) pairs: a square grid and a tall one with a non-square shape.
_GRIDS = [((2, 2), 4), ((2, 3), 6)]


def _strings(norb: int, nelec: int, level: int) -> np.ndarray:
    """All ``nelec``-bit strings at exactly ``level`` excitations from HF."""
    hf = (1 << nelec) - 1
    out = []
    for holes in itertools.combinations(range(nelec), level):
        for parts in itertools.combinations(range(nelec, norb), level):
            mask = hf
            for h in holes:
                mask &= ~(1 << h)
            for p in parts:
                mask |= 1 << p
            out.append(mask)
    return np.array(sorted(out), dtype=np.int64)


def _full_product(norb: int, nelec: int) -> PairBasis:
    """The complete Cartesian product of two equal string sets."""
    strings = _strings(norb, nelec, 0)
    strings = np.sort(
        np.concatenate([_strings(norb, nelec, level) for level in range(norb - nelec + 1)])
    )
    return pair_basis_from_arrays(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))


def _tier_spec(
    norb: int, nelec: int, max_level: int, drop_in_beta: float = 0.0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The tier-block specification: one ``(α strings, β strings)`` product per block.

    Every block is ``tiers[a] × tiers[b]``; at ``drop_in_beta > 0`` each block's
    β side is subsampled independently, which splits β strings that would
    otherwise share one α-partner set.
    """
    tiers = [_strings(norb, nelec, level) for level in range(max_level + 1)]
    rng = np.random.default_rng(7)
    spec: list[tuple[np.ndarray, np.ndarray]] = []
    for level in range(max_level + 1):
        for a in range(level + 1):
            b = level - a
            sa = tiers[a]
            sb = tiers[b]
            if drop_in_beta:
                keep = max(1, int(len(sb) * (1.0 - drop_in_beta)))
                sb = np.sort(rng.choice(sb, keep, replace=False))
            spec.append((sa, sb))
    return spec


def _tier_blocks(norb: int, nelec: int, max_level: int, drop_in_beta: float = 0.0) -> PairBasis:
    """A tier-block basis mirroring the scaling benchmark's ``gdb_pairs``.

    Every block is ``tiers[a] × tiers[b]``; at ``drop_in_beta > 0`` each block's
    β side is subsampled independently, which splits β strings that would
    otherwise share one α-partner set.
    """
    spec = _tier_spec(norb, nelec, max_level, drop_in_beta)
    alphas = [np.repeat(alpha, len(beta)) for alpha, beta in spec]
    betas = [np.tile(beta, len(alpha)) for alpha, beta in spec]
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas))


def _tier_basis() -> PairBasis:
    """The tier-block basis with one β side subsampled per block."""
    return _tier_blocks(4, 2, 2, drop_in_beta=0.3)


def _product_basis() -> PairBasis:
    """A full six-by-six Cartesian product basis."""
    return _full_product(4, 2)


@pytest.fixture(params=[_tier_basis, _product_basis], ids=["tier-blocks", "full-product"])
def basis(request: pytest.FixtureRequest) -> PairBasis:
    """A block-decomposed pair basis, one per supported shape."""
    return request.param()


@pytest.fixture()
def blocks(basis: PairBasis) -> PairBlocks:
    """The exact Cartesian decomposition of the basis under test."""
    built = build_pair_blocks(basis, mode="on")
    assert built is not None
    return built


@pytest.mark.parametrize(("grid", "world_size"), _GRIDS)
def test_tiles_cover_the_basis_disjointly(
    basis: PairBasis,
    blocks: PairBlocks,
    grid: tuple[int, int],
    world_size: int,
) -> None:
    """Ranks tile the whole pair basis exactly once, matching a brute-force mask."""
    assert blocks.n_pairs == basis.n_pairs
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    owned = []
    for rank in range(world_size):
        tile = pair_tile_bounds(partition, basis, rank)
        mask = (
            (basis.alpha_uid_of_pair >= tile.alpha_start)
            & (basis.alpha_uid_of_pair < tile.alpha_stop)
            & (basis.beta_uid_of_pair >= tile.beta_start)
            & (basis.beta_uid_of_pair < tile.beta_stop)
        )
        expected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        assert torch.equal(tile.pair_index, expected)
        assert tile.local_size == int(expected.numel())
        owned.append(tile.pair_index)
    union = torch.cat(owned)
    assert torch.equal(torch.sort(union).values, torch.arange(basis.n_pairs))
    assert torch.equal(torch.bincount(union, minlength=basis.n_pairs), torch.ones(basis.n_pairs))


@pytest.mark.parametrize(("grid", "world_size"), _GRIDS)
@pytest.mark.parametrize("axis", ["by_beta", "by_alpha"])
def test_fragments_cover_each_tile(
    basis: PairBasis,
    blocks: PairBlocks,
    grid: tuple[int, int],
    world_size: int,
    axis: str,
) -> None:
    """Each block's fragment is its tile rectangle, and the fragments cover the tile."""
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    groups: SpinGroups = getattr(blocks, axis)
    for rank in range(world_size):
        tile = pair_tile_bounds(partition, basis, rank)
        fragments = pair_tile_fragments(groups, tile)
        if tile.is_empty:
            assert fragments == ()
            continue
        covered = torch.cat([fragment.pair_index for fragment in fragments])
        assert int(covered.numel()) == tile.local_size
        assert torch.equal(torch.sort(covered).values, tile.pair_index)
        assert torch.equal(
            torch.bincount(covered, minlength=basis.n_pairs).clamp(max=1),
            torch.bincount(tile.pair_index, minlength=basis.n_pairs),
        )
        for fragment in fragments:
            group = groups.groups[fragment.group]
            n_alpha = int(group.alpha_uids.numel())
            n_beta = int(group.beta_uids.numel())
            grid_pairs = group.pair_idx.view(n_alpha, n_beta)
            rectangle = grid_pairs[
                fragment.row_start : fragment.row_stop,
                fragment.col_start : fragment.col_stop,
            ].reshape(-1)
            assert torch.equal(fragment.pair_index, rectangle)
            assert fragment.pair_index.numel() == fragment.n_rows * fragment.n_cols
            rows = group.alpha_uids.unsqueeze(1).expand(n_alpha, n_beta)
            cols = group.beta_uids.unsqueeze(0).expand(n_alpha, n_beta)
            mask = (
                (rows >= tile.alpha_start)
                & (rows < tile.alpha_stop)
                & (cols >= tile.beta_start)
                & (cols < tile.beta_stop)
            )
            assert torch.equal(fragment.pair_index, grid_pairs[mask])


@pytest.mark.parametrize(("grid", "world_size"), _GRIDS)
def test_tile_global_local_maps_round_trip(
    basis: PairBasis,
    grid: tuple[int, int],
    world_size: int,
) -> None:
    """Local-to-global-to-local is the identity; foreign global pairs raise."""
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    assert pair_tile_bounds(partition, basis).rank == 0  # rank defaults to the first
    tiles: list[PairTileBounds] = [
        pair_tile_bounds(partition, basis, rank) for rank in range(world_size)
    ]
    for tile in tiles:
        local = torch.arange(tile.local_size)
        assert torch.equal(tile.local_of_global(tile.global_pair_index(local)), local)
        owned = torch.zeros(basis.n_pairs, dtype=torch.bool)
        owned[tile.pair_index] = True
        foreign = torch.nonzero(~owned, as_tuple=False).reshape(-1)
        if int(foreign.numel()) == 0:
            continue  # every rank owns every pair, so no foreign index exists
        with pytest.raises(ValueError, match="outside this tile"):
            tile.local_of_global(foreign[:1])


def test_empty_tiles_own_nothing(basis: PairBasis, blocks: PairBlocks) -> None:
    """A tall grid leaves ranks with no pair and no fragments, and ``owns`` says so."""
    world_size = 8
    partition = partition_tiles(
        basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=(8, 1)
    )
    tiles = [pair_tile_bounds(partition, basis, rank) for rank in range(world_size)]
    empty = [tile for tile in tiles if tile.is_empty]
    filled = [tile for tile in tiles if not tile.is_empty]
    assert empty
    assert filled
    for tile in empty:
        assert tile.local_size == 0
        assert int(tile.pair_index.numel()) == 0
        assert pair_tile_fragments(blocks.by_beta, tile) == ()
        assert pair_tile_fragments(blocks.by_alpha, tile) == ()
    foreign_pair = int(filled[0].pair_index[0])
    for tile in empty:
        assert not tile.owns(foreign_pair)


def _tile_pair_counts(partition: TilePartition, basis: PairBasis) -> list[int]:
    """Pair count of every rank's tile."""
    return [
        pair_tile_bounds(partition, basis, rank).local_size for rank in range(partition.world_size)
    ]


def _imbalance(counts: list[int]) -> float:
    """Largest tile pair count over the smallest."""
    return max(counts) / max(min(counts), 1)


def test_pair_count_weights_balance_a_skewed_even_split() -> None:
    """Pair-count weights turn a 2.2x uid split into an even one.

    The even split of this basis' seven β uids gives 22/10 pairs; the per-uid
    counts move the cut so both tiles own 16 pairs.
    """
    basis = _tier_blocks(5, 2, 2, drop_in_beta=0.5)
    helpers = generate_pair_helpers(basis, 5, device="cpu", block_matvec="off")
    even = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 2, grid=(1, 2))
    weighted = pair_tile_partition(helpers, 2, grid=(1, 2))
    even_counts = _tile_pair_counts(even, basis)
    weighted_counts = _tile_pair_counts(weighted, basis)
    assert sum(weighted_counts) == basis.n_pairs
    assert _imbalance(even_counts) >= 1.5
    assert _imbalance(weighted_counts) < 1.3


def test_uniform_weights_reproduce_the_even_cut() -> None:
    """A full product carries equal per-uid weight, so the cut stays even.

    The product has five strings — an odd count — which is where the weighted
    boundary walk lands exactly on the even boundary.
    """
    strings = np.sort(np.concatenate([_strings(4, 2, 0), _strings(4, 2, 1)]))
    basis = pair_basis_from_arrays(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    helpers = generate_pair_helpers(basis, 4, device="cpu", block_matvec="off")
    row_weights, col_weights = pair_axis_weights(helpers)
    assert row_weights is not None and col_weights is not None
    assert bool((row_weights == len(strings)).all())
    assert bool((col_weights == len(strings)).all())
    even = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=(2, 2))
    weighted = pair_tile_partition(helpers, 4, grid=(2, 2))
    assert weighted.grid.alpha_bounds == even.grid.alpha_bounds
    assert weighted.grid.beta_bounds == even.grid.beta_bounds


def test_structural_weights_match_the_materialised_basis() -> None:
    """Block-geometry counts equal the materialised uid bin counts, exactly."""
    spec = _tier_spec(5, 2, 2)
    structural = structural_pair_basis_from_blocks(spec)
    explicit = _tier_blocks(5, 2, 2)
    structural_helpers = generate_pair_helpers(structural, 5, device="cpu", block_matvec="on")
    explicit_helpers = generate_pair_helpers(explicit, 5, device="cpu", block_matvec="off")
    row_weights, col_weights = pair_axis_weights(structural_helpers)
    row_expected, col_expected = pair_axis_weights(explicit_helpers)
    assert row_weights is not None and col_weights is not None
    assert row_expected is not None and col_expected is not None
    alpha_at = torch.searchsorted(explicit.unique_alphas, structural.unique_alphas)
    beta_at = torch.searchsorted(explicit.unique_betas, structural.unique_betas)
    assert torch.equal(row_weights, row_expected[alpha_at])
    assert torch.equal(col_weights, col_expected[beta_at])
    assert row_weights.dtype == torch.float64 and col_weights.dtype == torch.float64
    assert row_weights.device == structural_helpers.device
    assert int(row_weights.sum()) == structural.n_pairs
    assert int(col_weights.sum()) == structural.n_pairs
    assert len(set(row_weights.tolist())) > 1, "the tiers must carry different pair counts"


def test_weighted_partition_tile_matvec_matches_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2x2 tile operators on the weighted partition reproduce the full matvec."""
    basis = _tier_blocks(5, 2, 2, drop_in_beta=0.5)
    norb = 5
    helpers = generate_pair_helpers(basis, norb, device="cpu", block_matvec="on")
    even = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=(2, 2))
    partition = pair_tile_partition(helpers, 4, grid=(2, 2))
    assert (partition.grid.alpha_bounds, partition.grid.beta_bounds) != (
        even.grid.alpha_bounds,
        even.grid.beta_bounds,
    )
    integrals = _real_integrals(norb)
    x = torch.randn(basis.n_pairs, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(4):
        bounds = pair_tile_bounds(partition, basis, rank)
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=4,
        )
        got = operator(x[bounds.pair_index])
        assert torch.allclose(got, expected[bounds.pair_index], atol=1e-12, rtol=0.0), rank


def test_tile_matvec_on_a_holed_product_refines_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered-style pair list tiles correctly once collisions are refined.

    Ten percent of a 70-string product's pairs are removed, so each string gets
    its own partner set and the linear fingerprints collide — the case that
    used to reject the decomposition and leave tiled GDB unreachable. The 2x2
    tile operators must reproduce the full matvec on the scattered owned
    subsets.
    """
    base = _full_product(8, 4)
    n_keep = int(round(0.9 * base.n_pairs))
    keep = np.sort(np.random.default_rng(0).permutation(base.n_pairs)[:n_keep])
    basis = pair_basis_from_arrays(
        base.alpha_of_pair.numpy()[keep], base.beta_of_pair.numpy()[keep], n_orb=8
    )
    norb = 8
    helpers = generate_pair_helpers(basis, norb, device="cpu", block_matvec="on")
    assert helpers.blocks is not None
    partition = pair_tile_partition(helpers, 4, grid=(2, 2))
    integrals = _real_integrals(norb)
    x = torch.randn(basis.n_pairs, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(4):
        bounds = pair_tile_bounds(partition, basis, rank)
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=4,
        )
        got = operator(x[bounds.pair_index])
        assert torch.allclose(got, expected[bounds.pair_index], atol=1e-12, rtol=0.0), rank


def test_resolve_auto_pair_tiling_keeps_one_d_when_it_fits() -> None:
    """The pair-tile auto policy only tiles when the 1-D working set is large."""
    big_budget = 80 * 2**30
    # A small basis fits the 1-D gather many times over: keep the faster path.
    assert not resolve_auto_pair_tiling(world_size=4, n_pairs=36, budget_bytes=big_budget)
    # No device budget (CPU runs) keeps 1-D too; one rank never tiles.
    assert not resolve_auto_pair_tiling(world_size=4, n_pairs=36, budget_bytes=None)
    assert not resolve_auto_pair_tiling(world_size=1, n_pairs=10**9, budget_bytes=1)
    # A basis whose gathered input plus block vectors exceed half the budget
    # has to tile.
    assert resolve_auto_pair_tiling(
        world_size=4, n_pairs=10**9, block_size=4, budget_bytes=8 * 2**30
    )


def test_pair_replicated_bytes_counts_the_basis_and_helper_arrays() -> None:
    """The estimate sums the nine per-pair arrays plus the uid/offset arrays."""
    basis = _full_product(4, 2)
    helpers = generate_pair_helpers(basis, 4, device="cpu", block_matvec="off")
    n_pairs = basis.n_pairs
    expected = 9 * n_pairs * 8 + 16 * (basis.n_unique_alphas + basis.n_unique_betas + 1)
    assert pair_replicated_bytes(helpers) == expected


def test_auto_pair_tiling_counts_the_replicated_structures() -> None:
    """Replicated basis/helper bytes can tip the fit decision into tiling."""
    basis = _full_product(4, 2)
    helpers = generate_pair_helpers(basis, 4, device="cpu", block_matvec="off")
    replicated = pair_replicated_bytes(helpers)
    # The vector working set alone fits half of this budget; adding the
    # replicated pair arrays breaks it, so the same call flips to a tile grid.
    assert not resolve_auto_pair_tiling(
        world_size=4,
        n_pairs=basis.n_pairs,
        block_size=8,
        budget_bytes=3200,
        replicated_bytes=0,
    )
    assert resolve_auto_pair_tiling(
        world_size=4,
        n_pairs=basis.n_pairs,
        block_size=8,
        budget_bytes=3200,
        replicated_bytes=replicated,
    )
