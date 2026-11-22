"""Tests for the tiled GDB (explicit pair) owner-computes matvec operator.

Covers:

1. Single process with a stand-in exchange that slices a full vector by the
   per-stage needed index lists: every rank of a 2x2 tile and of an uneven
   2x3 tile reproduces the full matvec restricted to its pair set, for real
   and complex integrals.
2. gloo, four ranks, 2x2: with the real staged P2P exchange, every rank's
   output matches the full-space matvec restricted to its pair set, for real
   and complex vectors.
3. The scattered-tile α↔β-swap projector: single process and gloo 2x2, real
   and complex, against the full-space ``(v + tau * v[k_swap]) / 2`` formula.
4. The scattered memmap hand-back: four gloo ranks write disjoint positions of
   one shared 1-D file.
5. The needed-set staging plan: against the whole-tile plan on dense, uneven,
   sparse and empty-slab layouts (single process and gloo 2/4 ranks); a sparse
   orbital-cluster basis proves it ships strictly fewer bytes; the whole-tile
   fallback is bit-identical to the whole-tile plan; the plan is reproducible
   from the replicated geometry alone; and the tiled 1-RDM, 2-RDM and ``S^2``
   sweeps return the same values with either staging plan.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch_sci_solver.torch_impl.distributed_pairs_tile as dpt
from test_distributed_pairs import _init_gloo, _run_spawned  # noqa: E402
from test_mult_pair_blocks import _complex_integrals, _real_integrals  # noqa: E402
from test_pair_blocks import _full_product, _tier_blocks  # noqa: E402
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import TilePartition, partition_tiles
from torch_sci_solver.torch_impl.mult_pairs import (
    make_diag_terms_pair_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiles import pair_tile_bounds
from torch_sci_solver.torch_impl.pairs import PairBasis, pair_basis_from_arrays

# A 2x2 rank grid over the uid axes, and an uneven 2x3 grid (6 ranks).
_GRID_2X2 = (2, 2)
_GRID_2X3 = (2, 3)


class _FullVectorExchange:
    """Stand-in for :class:`StagedPairExchange` that slices a full vector.

    Lets the operator serve any tile without a process group. Stage 0 yields
    the local block; stage ``s`` yields ``full_vector[needed[s]]``, exactly the
    values the real exchange would deliver in that order.
    """

    full_vector: torch.Tensor | None = None

    def __init__(
        self,
        local_index: Tensor,
        needed: tuple[Tensor, ...],
        *,
        rank: int,
        world_size: int,
        group: object = None,
        tag: int = 0,
    ) -> None:
        """Record the stage lists; the transport is the class-level vector."""
        _ = (local_index, group, tag)
        self.needed = tuple(needed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def stages(self, x_local: Tensor):
        """Yield ``(stage, buffer)`` for every stage."""
        yield 0, x_local
        full = _FullVectorExchange.full_vector
        assert full is not None
        for stage in range(1, self.world_size):
            yield stage, full[self.needed[stage]]


def _basis(name: str) -> tuple[PairBasis, int]:
    """One named small block-decomposed basis and its spatial orbital count."""
    if name == "full_product":
        return _full_product(4, 2), 4
    if name == "tier":
        return _tier_blocks(6, 3, max_level=2), 6
    raise ValueError(name)


def _helpers(basis: PairBasis, norb: int):
    """Pair helpers with the exact block decomposition forced on."""
    helpers = generate_pair_helpers(basis, norb, device="cpu", block_matvec="on")
    assert helpers.blocks is not None
    return helpers


def _random_vector(n: int, seed: int, *, complex_values: bool) -> Tensor:
    """Deterministic random real or complex vector."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if not complex_values:
        return real
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.complex(real, imag)


def _reference(basis: PairBasis, norb: int, integrals, helpers, x: Tensor) -> tuple[Tensor, Tensor]:
    """The full diagonal and the full-space matvec of ``x``."""
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    return hii, mult_pair_torch(hii, x, basis, helpers, norb, integrals)


def _assert_pair_slice(got: Tensor, expected: Tensor, label: str) -> None:
    """Assert a tile's output equals the reference's slice to ``atol=1e-12``."""
    if got.numel() == 0 and expected.numel() == 0:
        return
    diff = float((got - expected).abs().max())
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0), (label, diff)


# ============================================================================
# 1. Single process, stand-in exchange
# ============================================================================


@pytest.mark.parametrize(
    "basis_name,grid",
    [("full_product", _GRID_2X2), ("tier", _GRID_2X3)],
    ids=["2x2", "2x3"],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_tile_operator_single_process_matches_full(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank of the tile grid reproduces the full matvec's pair slice.

    The stand-in exchange slices a full vector, so no process group is needed;
    the 2x3 case splits 19 unique strings unevenly over three slabs.
    """
    basis, norb = _basis(basis_name)
    world_size = grid[0] * grid[1]
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    if grid == _GRID_2X3:
        lengths = {stop - start for start, stop in partition.grid.alpha_bounds}
        assert len(lengths) > 1, "the 2x3 fixture must split the α axis unevenly"
    x = _random_vector(basis.n_pairs, 11, complex_values=complex_values)
    hii, expected = _reference(basis, norb, integrals, helpers, x)

    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    for rank in range(world_size):
        bounds = pair_tile_bounds(partition, basis, rank)
        _FullVectorExchange.full_vector = x
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = operator(x[bounds.pair_index])
        _assert_pair_slice(
            got, expected[bounds.pair_index], f"{basis_name} rank={rank}/{world_size}"
        )


def test_tile_operator_rejects_helpers_without_blocks(monkeypatch) -> None:
    """The operator requires the exact block decomposition."""
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    off = generate_pair_helpers(basis, norb, device="cpu", block_matvec="off")
    on = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    bounds = pair_tile_bounds(partition, basis, 0)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    with pytest.raises(ValueError, match="blocks"):
        dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            off,
            norb,
            integrals,
            bounds,
            partition,
            rank=0,
            world_size=4,
        )
    # The same construction with the decomposition enabled succeeds.
    dpt.LocalPairTileOperator(
        hii[bounds.pair_index],
        basis,
        on,
        norb,
        integrals,
        bounds,
        partition,
        rank=0,
        world_size=4,
    )


def test_tile_operator_single_rank_matches_full() -> None:
    """world_size=1 with the real exchange: only the local stage runs."""
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    bounds = pair_tile_bounds(partition, basis, 0)
    x = _random_vector(basis.n_pairs, 13, complex_values=False)
    hii, expected = _reference(basis, norb, integrals, helpers, x)
    operator = dpt.LocalPairTileOperator(
        hii[bounds.pair_index],
        basis,
        helpers,
        norb,
        integrals,
        bounds,
        partition,
        rank=0,
        world_size=1,
    )
    _assert_pair_slice(operator(x), expected, "world_size=1")
    # A single rank exchanges nothing, so no needed plan is built.
    assert operator.staging_mode == "tile"


def test_needed_stage_lists_partition_the_basis(monkeypatch) -> None:
    """Stage ``s`` carries exactly the pair indices owned by peer ``rank + s``.

    The whole-tile plan is pinned with ``needed_staging="off"``: each remote
    stage list is the peer's sorted tile.
    """
    basis, norb = _basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    for rank in range(4):
        bounds = pair_tile_bounds(partition, basis, rank)
        operator = dpt.LocalPairTileOperator(
            torch.zeros(bounds.local_size, dtype=torch.float64),
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=4,
            needed_staging="off",
        )
        staged = torch.sort(
            torch.cat([bounds.pair_index.cpu(), *operator._needed_stages[1:]])
        ).values
        assert torch.equal(staged, torch.arange(basis.n_pairs)), rank
        for stage in range(1, 4):
            peer = pair_tile_bounds(partition, basis, (rank + stage) % 4)
            assert torch.equal(operator._needed_stages[stage], peer.pair_index.cpu())


def test_empty_tiles_produce_empty_blocks(monkeypatch) -> None:
    """A tile with no pair runs the exchange and returns an empty block."""
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    # 8 α slabs over 6 unique strings: two slabs (and their tiles) are empty.
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 16, grid=(8, 2))
    x = _random_vector(basis.n_pairs, 17, complex_values=False)
    hii, expected = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    empty_ranks = 0
    for rank in range(16):
        bounds = pair_tile_bounds(partition, basis, rank)
        empty_ranks += int(bounds.is_empty)
        _FullVectorExchange.full_vector = x
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=16,
        )
        got = operator(x[bounds.pair_index])
        _assert_pair_slice(got, expected[bounds.pair_index], f"empty-tile rank={rank}/16")
    assert empty_ranks > 0, "the fixture must contain an empty tile"


# ============================================================================
# 2. gloo, four ranks, 2x2, real staged P2P exchange
# ============================================================================


def _tile_operator_worker(
    rank: int,
    world_size: int,
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's tile output equals the full-space matvec's slice."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis, norb = _basis("tier")
        integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
        helpers = _helpers(basis, norb)
        partition = partition_tiles(
            basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=_GRID_2X2
        )
        bounds = pair_tile_bounds(partition, basis, rank)
        x = _random_vector(basis.n_pairs, 7, complex_values=complex_values)
        hii, expected = _reference(basis, norb, integrals, helpers, x)
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            norb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = operator(x[bounds.pair_index])
        _assert_pair_slice(got, expected[bounds.pair_index], f"gloo rank={rank}/{world_size}")
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_values", [False, True])
def test_tile_operator_gloo_two_by_two(complex_values: bool, tmp_path: Path) -> None:
    """gloo, 4 ranks, 2x2: the real exchange matches the full-space reference."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _tile_operator_worker,
        4,
        (complex_values,),
        tmp_path,
        f"pair_tile_2x2_{'complex' if complex_values else 'real'}",
    )


# ============================================================================
# 3. α↔β-swap projector over a scattered tile
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True])
def test_tile_projector_single_process_matches_full(complex_values: bool) -> None:
    """world_size=1: the tile projector equals the full-space pair formula."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _pair_ab_swap_permutation

    basis, norb = _basis("full_product")
    dtype = torch.complex128 if complex_values else torch.float64
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    bounds = pair_tile_bounds(partition, basis, 0)
    full_k_swap, full_tau = _pair_ab_swap_permutation(basis, norb, torch.device("cpu"), dtype)
    assert full_k_swap is not None and full_tau is not None
    k_swap, tau = _pair_ab_swap_permutation(
        basis, norb, torch.device("cpu"), dtype, pair_indices=bounds.pair_index
    )
    assert k_swap is not None and tau is not None
    assert torch.equal(k_swap, full_k_swap[bounds.pair_index])
    assert torch.equal(tau, full_tau[bounds.pair_index])

    v = _random_vector(basis.n_pairs, 21, complex_values=complex_values)
    expected = 0.5 * (v + full_tau * v[full_k_swap])
    projector = dpt.LocalPairTileSwapProjector(
        basis, k_swap, tau, bounds, partition, rank=0, world_size=1
    )
    _assert_pair_slice(
        projector(v[bounds.pair_index]), expected[bounds.pair_index], f"single {dtype}"
    )


def _tile_projector_worker(
    rank: int,
    world_size: int,
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's tile projector equals the full-space formula on its pairs."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _pair_ab_swap_permutation

    _init_gloo(rank, world_size, init_file)
    try:
        basis, norb = _basis("full_product")
        dtype = torch.complex128 if complex_values else torch.float64
        partition = partition_tiles(
            basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=_GRID_2X2
        )
        bounds = pair_tile_bounds(partition, basis, rank)
        full_k_swap, full_tau = _pair_ab_swap_permutation(basis, norb, torch.device("cpu"), dtype)
        assert full_k_swap is not None and full_tau is not None
        k_swap, tau = _pair_ab_swap_permutation(
            basis, norb, torch.device("cpu"), dtype, pair_indices=bounds.pair_index
        )
        assert k_swap is not None and tau is not None
        v = _random_vector(basis.n_pairs, 23, complex_values=complex_values)
        expected = 0.5 * (v + full_tau * v[full_k_swap])
        projector = dpt.LocalPairTileSwapProjector(
            basis, k_swap, tau, bounds, partition, rank=rank, world_size=world_size
        )
        got = projector(v[bounds.pair_index])
        _assert_pair_slice(got, expected[bounds.pair_index], f"gloo rank={rank}/{world_size}")
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_values", [False, True])
def test_tile_projector_gloo_two_by_two(complex_values: bool, tmp_path: Path) -> None:
    """gloo, 4 ranks, 2x2: the staged partner exchange matches the formula."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _tile_projector_worker,
        4,
        (complex_values,),
        tmp_path,
        f"pair_tile_proj_{'complex' if complex_values else 'real'}",
    )


# ============================================================================
# 4. Scattered memmap hand-back
# ============================================================================


def _pair_tile_memmap_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank writes its scattered pair slice into one shared 1-D file."""
    from types import SimpleNamespace

    import numpy as np
    from torch_sci_solver.torch_impl.distributed_solver import write_pair_tile_memmaps

    _init_gloo(rank, world_size, init_file)
    try:
        basis, _ = _basis("full_product")
        partition = partition_tiles(
            basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=_GRID_2X2
        )
        bounds = pair_tile_bounds(partition, basis, rank)
        x = _random_vector(basis.n_pairs, 31, complex_values=False)
        context = SimpleNamespace(rank=rank, world_size=world_size, group=None)
        marker = write_pair_tile_memmaps(
            {"amp": x[bounds.pair_index]},
            {"amp": (basis.n_pairs,)},
            pair_index=bounds.pair_index,
            context=context,
        )["amp"]
        if rank == 0:
            full = np.load(marker.path, mmap_mode="r")
            assert np.allclose(full, x.numpy(), atol=1e-12)
            Path(marker.path).unlink()
            Path(marker.path).parent.rmdir()
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_pair_tile_memmap_writes_each_scattered_slice(tmp_path: Path) -> None:
    """The scattered GDB hand-back fills every position of the shared file."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(_pair_tile_memmap_worker, 4, (), tmp_path, "pair_tile_memmap")


def _memo_operator(
    basis: PairBasis,
    norb: int,
    helpers,
    integrals,
    hii: Tensor,
    partition,
    rank: int,
    world_size: int,
    *,
    move_cache_bytes: int | None = None,
):
    """One tile operator with the stand-in exchange installed."""
    bounds = pair_tile_bounds(partition, basis, rank)
    return dpt.LocalPairTileOperator(
        hii[bounds.pair_index],
        basis,
        helpers,
        norb,
        integrals,
        bounds,
        partition,
        rank=rank,
        world_size=world_size,
        move_cache_bytes=move_cache_bytes,
    )


def test_tile_operator_move_cache_reuses_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second matvec reuses the same move tables and returns the same output.

    The staged slab kernels enumerate a fragment's moves once per source stage;
    the memo is what keeps that enumeration to once per fragment. The entries
    and their byte count must not grow on the second call, in either channel's
    cache.
    """
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    x = _random_vector(basis.n_pairs, 13, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    operator = _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4)
    ss_cache = operator._ss_move_cache
    ab_cache = operator._ab_move_cache
    assert ss_cache is not None and ab_cache is not None
    caches = (ss_cache, ab_cache)

    bounds = pair_tile_bounds(partition, basis, 0)
    first = operator(x[bounds.pair_index])
    snapshot = [(len(cache.entries), cache.used_bytes) for cache in caches]
    assert all(entries > 0 and used > 0 for entries, used in snapshot)
    second = operator(x[bounds.pair_index])
    assert torch.equal(first, second)
    assert [(len(cache.entries), cache.used_bytes) for cache in caches] == snapshot


@pytest.mark.parametrize("cache_bytes", [0, 1], ids=["off", "exhausted"])
def test_tile_operator_move_cache_bounded(
    monkeypatch: pytest.MonkeyPatch, cache_bytes: int
) -> None:
    """A disabled or exhausted memo changes neither the output nor correctness."""
    import dataclasses

    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    default_helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    x = _random_vector(basis.n_pairs, 17, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, default_helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    bounds = pair_tile_bounds(partition, basis, 0)

    reference = _memo_operator(basis, norb, default_helpers, integrals, hii, partition, 0, 4)(
        x[bounds.pair_index]
    )

    helpers = dataclasses.replace(default_helpers, string_cache_bytes=cache_bytes)
    operator = _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4)
    got = operator(x[bounds.pair_index])
    assert torch.equal(got, reference)
    if cache_bytes == 0:
        assert operator._ss_move_cache is None
        assert operator._ab_move_cache is None
    else:
        assert operator._ss_move_cache is not None
        assert operator._ss_move_cache.used_bytes == 0
        # The αβ share of a one-byte budget rounds away entirely.
        assert operator._ab_move_cache is None


def test_tile_operator_move_cache_bytes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit budget overrides the helpers' and keeps the matvec identical.

    ``move_cache_bytes`` is split into the documented same-spin/αβ shares, and
    a small override must not change a single output entry.
    """
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    x = _random_vector(basis.n_pairs, 23, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    bounds = pair_tile_bounds(partition, basis, 0)
    expected = _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4)(
        x[bounds.pair_index]
    )

    budget = 4096
    operator = _memo_operator(
        basis, norb, helpers, integrals, hii, partition, 0, 4, move_cache_bytes=budget
    )
    same_spin_bytes = max(1, int(budget * dpt._SAME_SPIN_CACHE_FRACTION))
    assert operator._ss_move_cache is not None
    assert operator._ss_move_cache.budget_bytes == same_spin_bytes
    assert operator._ab_move_cache is not None
    assert operator._ab_move_cache.budget_bytes == budget - same_spin_bytes
    assert torch.equal(operator(x[bounds.pair_index]), expected)

    with pytest.raises(ValueError, match="move_cache_bytes"):
        _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4, move_cache_bytes=0)
    with pytest.raises(ValueError, match="move_cache_bytes"):
        _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4, move_cache_bytes=-1)


def _entry_bytes(cache) -> list[int]:
    """Resident bytes of every cached move table, in insertion order."""
    return [
        0 if value is None else sum(int(t.numel()) * int(t.element_size()) for t in value)
        for value in cache.entries.values()
    ]


def test_tile_operator_move_cache_split_keeps_same_spin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget too small for αβ still retains the same-spin move tables.

    The same-spin channel owns the larger share, so a budget just large enough
    for one same-spin chunk leaves every αβ table over its own share: the
    same-spin memo keeps entries while the αβ memo retains nothing.
    """
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
    x = _random_vector(basis.n_pairs, 29, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    bounds = pair_tile_bounds(partition, basis, 0)

    # Measure the first table each channel memoises under an ample budget.
    probe = _memo_operator(basis, norb, helpers, integrals, hii, partition, 0, 4)
    probe(x[bounds.pair_index])
    assert probe._ss_move_cache is not None and probe._ab_move_cache is not None
    ss_first = next(size for size in _entry_bytes(probe._ss_move_cache) if size > 0)
    ab_first = next(size for size in _entry_bytes(probe._ab_move_cache) if size > 0)

    budget = math.ceil(ss_first / dpt._SAME_SPIN_CACHE_FRACTION)
    same_spin_bytes = max(1, int(budget * dpt._SAME_SPIN_CACHE_FRACTION))
    assert same_spin_bytes >= ss_first
    assert budget - same_spin_bytes < ab_first

    operator = _memo_operator(
        basis, norb, helpers, integrals, hii, partition, 0, 4, move_cache_bytes=budget
    )
    operator(x[bounds.pair_index])
    assert operator._ss_move_cache is not None
    assert operator._ab_move_cache is not None
    assert operator._ss_move_cache.used_bytes > 0
    assert len(operator._ss_move_cache.entries) > 0
    # An oversized first table no longer latches an empty cache, so the αβ
    # memo stays open for a smaller table while retaining none of these.
    assert not operator._ab_move_cache.full
    assert operator._ab_move_cache.used_bytes == 0
    assert not operator._ab_move_cache.entries


# ============================================================================
# 5. Needed-set staging plan
# ============================================================================


def _sparse_layout_basis() -> PairBasis:
    """A product basis of two orbital clusters that only couple through doubles.

    The α and β string sets are ``{0,1,2,3}`` and ``{3,4,5,6}`` three-subsets.
    A single excitation cannot cross the clusters (their intersection is at
    most one orbital), while a double excitation can, so the αβ channel of a
    cross-cluster source tile is empty and the same-spin channel reaches only
    the three strings that keep orbital 3.
    """
    strings = np.array(
        sorted(
            [sum(1 << i for i in combo) for combo in itertools.combinations(range(4), 3)]
            + [sum(1 << i for i in combo) for combo in itertools.combinations((3, 4, 5, 6), 3)]
        ),
        dtype=np.int64,
    )
    return pair_basis_from_arrays(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))


def _layout(name: str) -> tuple[PairBasis, int, TilePartition]:
    """One named fixture: basis, spatial orbital count and 2x2 tile partition."""
    if name == "full_product":
        basis, norb = _basis("full_product")
        partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
        return basis, norb, partition
    if name == "tier":
        basis, norb = _basis("tier")
        partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID_2X2)
        return basis, norb, partition
    if name == "sparse":
        basis = _sparse_layout_basis()
        starts = [0, 4, 8]
        partition = TilePartition.from_boundaries(
            basis.n_unique_alphas, basis.n_unique_betas, starts, starts
        )
        return basis, 8, partition
    raise ValueError(name)


def _worker_layout(name: str, grid: tuple[int, int]) -> tuple[PairBasis, int, TilePartition]:
    """Fixture of one gloo worker; ``grid`` is the rank grid for ``tier``."""
    if name == "tier":
        basis, norb = _basis("tier")
        partition = partition_tiles(
            basis.n_unique_alphas,
            basis.n_unique_betas,
            grid[0] * grid[1],
            grid=grid,
        )
        return basis, norb, partition
    if name == "sparse":
        basis = _sparse_layout_basis()
        starts = [0, 4, 8]
        return (
            basis,
            8,
            TilePartition.from_boundaries(
                basis.n_unique_alphas, basis.n_unique_betas, starts, starts
            ),
        )
    if name == "tier_empty":
        basis, norb = _basis("tier")
        # A 1x4 grid with one empty β slab, so two ranks own no pair.
        return (
            basis,
            norb,
            TilePartition.from_boundaries(
                basis.n_unique_alphas,
                basis.n_unique_betas,
                [0, basis.n_unique_alphas],
                [0, 10, 10, 15, basis.n_unique_betas],
            ),
        )
    raise ValueError(name)


def _mode_operator(
    basis: PairBasis,
    norb: int,
    helpers,
    integrals,
    hii: Tensor,
    partition: TilePartition,
    rank: int,
    mode: str,
):
    """One rank's tile operator with the requested staging mode."""
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
        world_size=partition.world_size,
        needed_staging=mode,
    )
    return operator, bounds


def _assert_match(got: Tensor, expected: Tensor, label: str) -> None:
    """Assert two tensors match to ``atol=1e-12``."""
    if got.numel() == 0 and expected.numel() == 0:
        return
    diff = float((got - expected).abs().max())
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0), (label, diff)


@pytest.mark.parametrize(
    "basis_name,grid",
    [
        ("full_product", _GRID_2X2),
        ("tier", (2, 1)),
        ("tier", (1, 2)),
        ("tier", _GRID_2X3),
    ],
    ids=["2x2", "2x1", "1x2", "2x3"],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_needed_staging_matches_whole_tile(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The needed plan reproduces the whole-tile matvec on every tile.

    Covers a dense basis, both one-dimensional grids and the uneven 2x3
    fixture, for real and complex amplitudes. Every needed list is a subset of
    the whole-tile list it replaces.
    """
    basis, norb = _basis(basis_name)
    world_size = grid[0] * grid[1]
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    x = _random_vector(basis.n_pairs, 11, complex_values=complex_values)
    hii, expected = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(world_size):
        whole, bounds = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "off")
        needed, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        x_local = x[bounds.pair_index]
        got_whole = whole(x_local)
        got_needed = needed(x_local)
        label = f"{basis_name} {grid} rank={rank}/{world_size}"
        _assert_pair_slice(got_whole, expected[bounds.pair_index], f"{label} whole")
        _assert_pair_slice(got_needed, expected[bounds.pair_index], f"{label} needed")
        _assert_pair_slice(got_needed, got_whole, f"{label} staged")
        assert needed.staging_mode == "needed", label
        assert needed.needed_size <= whole.needed_size, label
        for stage in range(1, world_size):
            assert bool(
                torch.isin(needed._needed_stages[stage], whole._needed_stages[stage]).all()
            ), (label, stage)


def test_needed_staging_auto_falls_back_to_whole_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dense layout resolves to the whole-tile plan and stays bit-identical.

    The full Cartesian product makes every source group's reachable hull the
    whole group rectangle, so "auto" keeps the whole-tile plan; its needed
    lists are the peer tiles and its matvec is bit-identical to "off".
    """
    basis, norb, partition = _layout("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, 13, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    world_size = partition.world_size
    for rank in range(world_size):
        auto, bounds = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "auto")
        whole, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "off")
        assert auto.staging_mode == "tile", rank
        assert auto.needed_size == auto.whole_tile_size, rank
        assert torch.equal(auto(x[bounds.pair_index]), whole(x[bounds.pair_index])), rank
        for stage in range(1, world_size):
            peer = pair_tile_bounds(partition, basis, (rank + stage) % world_size)
            assert torch.equal(auto._needed_stages[stage], peer.pair_index.cpu()), (rank, stage)


def test_needed_staging_on_overrides_the_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``needed_staging="on"`` uses the needed plan even when it is dense."""
    basis, norb, partition = _layout("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, 17, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(partition.world_size):
        needed, bounds = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        whole, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "off")
        assert needed.staging_mode == "needed", rank
        _assert_pair_slice(
            needed(x[bounds.pair_index]), whole(x[bounds.pair_index]), f"forced rank={rank}"
        )


@pytest.mark.parametrize("complex_values", [False, True])
def test_needed_staging_sparse_layout_ships_fewer_bytes(
    complex_values: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cluster-separated basis stages strictly fewer bytes than the tiles.

    Every remote source tile is 16 pairs. The αβ channel draws nothing across
    the clusters and the same-spin channel reaches three of the four source
    rows, so the needed plan ships 12 of each 16-value tile. The sparse audit
    is done in values and in bytes of the amplitude dtype.
    """
    basis, norb, partition = _layout("sparse")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, 19, complex_values=complex_values)
    hii, expected = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    total_needed = total_whole = 0
    for rank in range(partition.world_size):
        bounds = pair_tile_bounds(partition, basis, rank)
        x_local = x[bounds.pair_index]
        auto, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "auto")
        needed, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        whole, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "off")
        got_whole = whole(x_local)
        got_needed = needed(x_local)
        label = f"sparse rank={rank}/{partition.world_size}"
        _assert_pair_slice(got_whole, expected[bounds.pair_index], f"{label} whole")
        _assert_pair_slice(got_needed, got_whole, f"{label} staged")
        assert auto.staging_mode == "needed", label
        assert needed.staging_mode == "needed" and whole.staging_mode == "tile", label
        assert needed.needed_size < whole.needed_size, (label, needed.needed_size)
        itemsize = x.element_size()
        assert needed.needed_size * itemsize < whole.needed_size * itemsize, label
        total_needed += needed.needed_size
        total_whole += whole.needed_size
    assert total_needed * 4 <= total_whole * 3, (total_needed, total_whole)


@pytest.mark.parametrize("complex_values", [False, True])
def test_needed_staging_empty_slabs(complex_values: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Needed staging keeps the empty-tile behaviour on a partly empty grid."""
    basis, norb = _basis("full_product")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    # 8 α slabs over 6 unique strings: some tiles own no pair.
    world_size = 16
    partition = partition_tiles(
        basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=(8, 2)
    )
    x = _random_vector(basis.n_pairs, 23, complex_values=complex_values)
    hii, expected = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    empty_ranks = 0
    for rank in range(world_size):
        needed, bounds = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        _assert_pair_slice(
            needed(x[bounds.pair_index]),
            expected[bounds.pair_index],
            f"empty-slab rank={rank}/{world_size}",
        )
        empty_ranks += int(bounds.is_empty)
        if bounds.is_empty:
            # An empty destination reads nothing, so no plan is built and the
            # whole-tile path already stages nothing.
            assert needed.staging_mode == "tile", rank
            assert needed.needed_size == 0, rank
        else:
            assert needed.staging_mode == "needed", rank
            assert needed.needed_size <= needed.whole_tile_size, rank
    assert empty_ranks > 0, "the fixture must contain an empty tile"


@pytest.mark.parametrize("layout", ["tier", "sparse"])
def test_needed_plan_is_reproducible_from_replicated_geometry(
    layout: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any rank derives the same plan for a destination from published state.

    The plan function reads only the pair decomposition, the partition and the
    destination bounds, all of which every rank holds, so recomputing it in a
    fresh call reproduces the exact rectangles and index lists embedded in the
    destination's operator. Nothing here depends on the caller's own rank.
    """
    basis, norb, partition = _layout(layout)
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, 29, complex_values=False)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x

    def rect_fields(rect) -> tuple:
        return (
            rect.group,
            rect.row_start,
            rect.row_stop,
            rect.col_start,
            rect.col_stop,
            getattr(rect, "alpha_dests", ()),
            getattr(rect, "ab_dests", ()),
            getattr(rect, "beta_dests", ()),
        )

    def plan_fields(plan) -> tuple:
        return (
            tuple(tuple(rect_fields(rect) for rect in stage) for stage in plan.beta_rects),
            tuple(tuple(rect_fields(rect) for rect in stage) for stage in plan.alpha_rects),
        )

    for rank in range(partition.world_size):
        bounds = pair_tile_bounds(partition, basis, rank)
        first = dpt._needed_stage_plan(
            helpers,
            bounds,
            partition,
            dpt._tile_geometry(helpers.blocks, bounds, partition),
        )
        second = dpt._needed_stage_plan(
            helpers,
            bounds,
            partition,
            dpt._tile_geometry(helpers.blocks, bounds, partition),
        )
        assert plan_fields(first) == plan_fields(second), rank
        for left, right in zip(first.needed, second.needed):
            assert torch.equal(left, right), rank
        operator, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        for stage, expected in enumerate(first.needed):
            assert torch.equal(operator._needed_stages[stage], expected), (rank, stage)


def test_needed_staging_rejects_unknown_mode() -> None:
    """An unknown staging mode is rejected before any plan is built."""
    basis, norb, partition = _layout("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    with pytest.raises(ValueError, match="needed_staging"):
        _mode_operator(basis, norb, helpers, integrals, hii, partition, 0, "sometimes")


def _tiled_observables(operator, x_local: Tensor, helpers) -> tuple[Tensor, ...]:
    """Every tiled post-processing observable of one operator and block."""
    from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
        pair_tiled_one_rdm,
        pair_tiled_s2_apply,
        pair_tiled_spin_squared,
        pair_tiled_two_rdm,
    )

    gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
    gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
    spin_sq = pair_tiled_spin_squared(operator, x_local, helpers.n_a, helpers.n_b)
    s2_apply = pair_tiled_s2_apply(operator, x_local)
    return gamma_a, gamma_b, gamma2_aa, gamma2_ab, gamma2_bb, spin_sq, s2_apply


@pytest.mark.parametrize("layout", ["tier", "sparse"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_needed_staging_covers_tiled_postprocessing(
    layout: str, complex_values: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The needed plan stages every ket the RDM and S² sweeps read.

    Same-spin hulls include two-excitation reach and αβ hulls include the
    single α crossed with single β reach, which is exactly the set of ket rows
    the 1-RDM, 2-RDM and S² walkers look up in the exchange (the S² spin-swap
    source is one α single crossed with one β single). Both plans therefore
    return the same observables.
    """
    basis, norb, partition = _layout(layout)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, 31, complex_values=complex_values)
    hii, _ = _reference(basis, norb, integrals, helpers, x)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(partition.world_size):
        whole, bounds = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "off")
        needed, _ = _mode_operator(basis, norb, helpers, integrals, hii, partition, rank, "on")
        assert needed.staging_mode == "needed", rank
        x_local = x[bounds.pair_index]
        expected = _tiled_observables(whole, x_local, helpers)
        got = _tiled_observables(needed, x_local, helpers)
        for index, (left, right) in enumerate(zip(got, expected)):
            _assert_match(left, right, f"{layout} {complex_values} rank={rank} observable={index}")


def _needed_staging_worker(
    rank: int,
    world_size: int,
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Both staging plans agree on the matvec and every tiled observable."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis, norb, partition = _worker_layout(basis_name, grid)
        assert partition.world_size == world_size
        bounds = pair_tile_bounds(partition, basis, rank)
        integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
        helpers = _helpers(basis, norb)
        x = _random_vector(basis.n_pairs, 37, complex_values=complex_values)
        hii, expected = _reference(basis, norb, integrals, helpers, x)
        x_local = x[bounds.pair_index]
        outputs = {}
        operators = {}
        for mode in ("off", "on"):
            operator, _ = _mode_operator(
                basis, norb, helpers, integrals, hii, partition, rank, mode
            )
            operators[mode] = operator
            outputs[mode] = operator(x_local)
        label = f"{basis_name} {grid} rank={rank}/{world_size}"
        if bounds.is_empty:
            # An empty tile reads nothing and builds no plan.
            assert operators["on"].staging_mode == "tile", label
            assert operators["on"].needed_size == 0, label
        else:
            assert operators["on"].staging_mode == "needed", label
            assert operators["on"].needed_size <= operators["on"].whole_tile_size, label
        staged = operators["on"]
        assert staged.stage_sizes == staged._exchange.stage_sizes, label
        assert staged._exchange.max_stage_size == max(staged.stage_sizes, default=0), label
        assert 0 <= staged._exchange.max_send_size <= staged.N_local, label
        _assert_pair_slice(outputs["off"], expected[bounds.pair_index], f"{label} whole")
        _assert_pair_slice(outputs["on"], expected[bounds.pair_index], f"{label} needed")
        _assert_pair_slice(outputs["on"], outputs["off"], f"{label} staged")
        whole_obs = _tiled_observables(operators["off"], x_local, helpers)
        needed_obs = _tiled_observables(operators["on"], x_local, helpers)
        for index, (left, right) in enumerate(zip(needed_obs, whole_obs)):
            _assert_match(left, right, f"{label} observable={index}")
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "basis_name,grid,complex_values",
    [
        ("tier", (2, 1), False),
        ("tier", (2, 1), True),
        ("tier", (1, 2), False),
        ("tier", (1, 2), True),
    ],
    ids=["2x1-real", "2x1-complex", "1x2-real", "1x2-complex"],
)
def test_needed_staging_gloo_two_ranks(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    tmp_path: Path,
) -> None:
    """gloo, 2 ranks, 2x1 and 1x2 grids: needed staging matches the whole tile."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _needed_staging_worker,
        2,
        (basis_name, grid, complex_values),
        tmp_path,
        f"pair_needed_2_{grid[0]}x{grid[1]}_{'complex' if complex_values else 'real'}",
    )


@pytest.mark.parametrize(
    "basis_name,grid,complex_values",
    [
        ("tier", (2, 2), False),
        ("tier", (2, 2), True),
        ("sparse", (2, 2), False),
        ("sparse", (2, 2), True),
        ("tier_empty", (1, 4), False),
        ("tier_empty", (1, 4), True),
    ],
    ids=[
        "2x2-real",
        "2x2-complex",
        "sparse-real",
        "sparse-complex",
        "empty-real",
        "empty-complex",
    ],
)
def test_needed_staging_gloo_four_ranks(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    tmp_path: Path,
) -> None:
    """gloo, 4 ranks: 2x2, sparse and partly empty grids, real and complex."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _needed_staging_worker,
        4,
        (basis_name, grid, complex_values),
        tmp_path,
        f"pair_needed_4_{basis_name}_{'complex' if complex_values else 'real'}",
    )
