"""Parity tests for the source-slab GDB block kernels.

On small block-decomposed bases and a 2x2 tile grid, for every fragment and
every source-slab intersection of its block, this checks:

1. The α (``by_beta``) and β (``by_alpha``) slab channels summed over the
   source slabs equal the matching streamed same-spin channel restricted to
   the fragment, and the two channels scattered through ``local_of_global``
   reproduce ``add_same_spin_blocks`` on the whole tile.
2. The αβ slab channel summed over every source group's source slabs equals
   ``add_ab_blocks`` restricted to the fragment; the source rectangles are
   asserted to cover each source group exactly once.
3. Edge cases: a source slab with no rows or columns adds nothing, a
   single-row/column source matches a sparsely staged reference, and a source
   rectangle that only partly overlaps a fragment's move targets matches the
   same reference.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pytest
import torch
from test_mult_pair_blocks import _complex_integrals, _real_integrals  # noqa: E402
from test_pair_fragments import (  # noqa: E402
    _assert_close,
    _basis,
    _generic_same_spin_channels,
    _helpers,
    _random_vector,
)
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    _single_moves,
    add_ab_blocks,
    add_ab_slab,
    add_same_spin_blocks,
    add_same_spin_slab,
)
from torch_sci_solver.torch_impl.pair_blocks import SpinGroups, _Group
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers, generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiles import (
    PairFragment,
    PairTileBounds,
    pair_tile_bounds,
    pair_tile_fragments,
)
from torch_sci_solver.torch_impl.pairs import PairBasis

_GRID = (2, 2)
_N_RANKS = 4


def _tiles(partition, basis: PairBasis) -> list[PairTileBounds]:
    """The four tile bounds of one tile partition."""
    return [pair_tile_bounds(partition, basis, rank) for rank in range(_N_RANKS)]


def _slab_positions(sorted_uids: Tensor, start: int, stop: int) -> tuple[int, int]:
    """Positions of the uid range ``[start, stop)`` inside a sorted uid axis."""
    probe = torch.tensor([int(start), int(stop)], dtype=torch.int64)
    lo, hi = torch.searchsorted(sorted_uids.cpu(), probe).tolist()
    return int(lo), int(hi)


def _one_group(groups: SpinGroups, index: int) -> SpinGroups:
    """A grouping carrying only block ``index``, for a one-source-slab call."""
    return SpinGroups(
        axis=groups.axis,
        group_of_string=groups.group_of_string,
        groups=(groups.groups[index],),
        total_pairs=groups.total_pairs,
    )


def _grid(helpers: PairHelpers, axis: str, group_id: int, t: Tensor) -> Tensor:
    """One block's ``|U_g| x |B_g|`` grid of amplitudes."""
    blocks = helpers.blocks
    assert blocks is not None
    groups = blocks.by_beta if axis == "beta" else blocks.by_alpha
    group = groups.groups[group_id]
    n_rows = int(group.alpha_uids.numel())
    n_cols = int(group.beta_uids.numel())
    return t[group.pair_idx].view(n_rows, n_cols)


def _alpha_slab_buffer(
    grid: Tensor,
    group: _Group,
    frag: PairFragment,
    src_tile: PairTileBounds,
) -> tuple[Tensor, int]:
    """The α stage buffer of one source slab, or an empty buffer.

    An α move leaves β fixed, so the resident columns are the fragment's own β
    columns. A source slab whose β range misses them contributes nothing;
    fragments lie inside one β slab, so the intersection is either the whole
    fragment-column range or empty.

    Returns:
        ``(T, row_offset)`` where ``T`` is row-major over the source rows and
        the fragment's columns.
    """
    row_lo, row_hi = _slab_positions(group.alpha_uids, src_tile.alpha_start, src_tile.alpha_stop)
    col_lo, col_hi = _slab_positions(group.beta_uids, src_tile.beta_start, src_tile.beta_stop)
    disjoint = col_hi <= frag.col_start or frag.col_stop <= col_lo
    assert disjoint or (col_lo <= frag.col_start and frag.col_stop <= col_hi)
    if disjoint:
        return grid[0:0, 0:0].reshape(-1), row_lo
    return grid[row_lo:row_hi, frag.col_start : frag.col_stop].contiguous().reshape(-1), row_lo


def _beta_slab_buffer(
    grid: Tensor,
    group: _Group,
    frag: PairFragment,
    src_tile: PairTileBounds,
) -> tuple[Tensor, int]:
    """The β stage buffer of one source slab, or an empty buffer.

    A β move leaves α fixed, so the resident rows are the fragment's own α
    rows. A source slab whose α range misses them contributes nothing;
    fragments lie inside one α slab, so the intersection is either the whole
    fragment-row range or empty.

    Returns:
        ``(T, col_offset)`` where ``T`` is row-major over the fragment's rows
        and the source columns.
    """
    row_lo, row_hi = _slab_positions(group.alpha_uids, src_tile.alpha_start, src_tile.alpha_stop)
    col_lo, col_hi = _slab_positions(group.beta_uids, src_tile.beta_start, src_tile.beta_stop)
    disjoint = row_hi <= frag.row_start or frag.row_stop <= row_lo
    assert disjoint or (row_lo <= frag.row_start and frag.row_stop <= row_hi)
    if disjoint:
        return grid[0:0, 0:0].reshape(-1), col_lo
    return grid[frag.row_start : frag.row_stop, col_lo:col_hi].contiguous().reshape(-1), col_lo


@pytest.mark.parametrize("basis_name", ["full_product", "tier"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("budget_bytes", [None, 1024], ids=["default", "tiny"])
def test_same_spin_slab_matches_channel(
    basis_name: str, complex_values: bool, budget_bytes: int | None
) -> None:
    """α/β slab channels summed over all source slabs match the channel slice.

    Every fragment of both groupings consumes one staged slab per source tile;
    a source slab that cannot reach the fragment (a different column slab for
    α, a different row slab for β, or an empty intersection) is passed as an
    empty buffer. The scattered fragments also reproduce the whole-channel
    ``add_same_spin_blocks`` on the tile.
    """
    basis, norb = _basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb, budget_bytes)
    blocks = helpers.blocks
    assert blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, _N_RANKS, grid=_GRID)
    off = generate_pair_helpers(basis, norb, device="cpu", block_matvec="off")
    tiles = _tiles(partition, basis)
    seen_empty = False

    for seed in (41, 42):
        t = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
        generic = _generic_same_spin_channels(t, basis, off, norb, integrals)
        expected = torch.zeros_like(t)
        add_same_spin_blocks(expected, t, helpers, integrals)

        for tile in tiles:
            if tile.is_empty:
                continue
            got_tile = torch.zeros(tile.local_size, dtype=t.dtype)

            for frag in pair_tile_fragments(blocks.by_beta, tile):
                group = blocks.by_beta.groups[frag.group]
                grid = _grid(helpers, "beta", frag.group, t)
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                for src_tile in tiles:
                    buf, row_offset = _alpha_slab_buffer(grid, group, frag, src_tile)
                    seen_empty |= buf.numel() == 0
                    add_same_spin_slab(
                        got,
                        buf,
                        helpers,
                        integrals,
                        blocks.by_beta,
                        frag,
                        row_offset=row_offset,
                        col_offset=0,
                    )
                label = f"{basis_name} rank={tile.rank} α slab seed={seed}"
                _assert_close(got, generic["alpha"][frag.pair_index], label)
                got_tile.index_add_(0, tile.local_of_global(frag.pair_index), got)

            for frag in pair_tile_fragments(blocks.by_alpha, tile):
                group = blocks.by_alpha.groups[frag.group]
                grid = _grid(helpers, "alpha", frag.group, t)
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                for src_tile in tiles:
                    buf, col_offset = _beta_slab_buffer(grid, group, frag, src_tile)
                    seen_empty |= buf.numel() == 0
                    add_same_spin_slab(
                        got,
                        buf,
                        helpers,
                        integrals,
                        blocks.by_alpha,
                        frag,
                        row_offset=0,
                        col_offset=col_offset,
                    )
                label = f"{basis_name} rank={tile.rank} β slab seed={seed}"
                _assert_close(got, generic["beta"][frag.pair_index], label)
                got_tile.index_add_(0, tile.local_of_global(frag.pair_index), got)

            label = f"{basis_name} rank={tile.rank} slab same-spin tile seed={seed}"
            _assert_close(got_tile, expected[tile.pair_index], label)

    assert seen_empty, "the 2x2 grid must exercise an empty source-slab intersection"


@pytest.mark.parametrize("basis_name", ["full_product", "tier"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("budget_bytes", [None, 1024], ids=["default", "tiny"])
def test_ab_slab_matches_whole_channel(
    basis_name: str, complex_values: bool, budget_bytes: int | None
) -> None:
    """αβ slab channels summed over every source rectangle match the fragment.

    Every fragment consumes every source group's source slabs one at a time,
    and the four source slabs' rectangles are asserted to be disjoint and to
    cover each source group's grid exactly once. An empty rectangle must leave
    the output untouched.
    """
    basis, norb = _basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb, budget_bytes)
    blocks = helpers.blocks
    assert blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, _N_RANKS, grid=_GRID)
    tiles = _tiles(partition, basis)
    seen_empty = False

    for seed in (51, 52):
        t = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
        expected = torch.zeros_like(t)
        add_ab_blocks(expected, t, helpers, integrals)

        for tile in tiles:
            if tile.is_empty:
                continue
            for frag in pair_tile_fragments(blocks.by_beta, tile):
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                for group_id, source_group in enumerate(blocks.by_beta.groups):
                    n_rows = int(source_group.alpha_uids.numel())
                    n_cols = int(source_group.beta_uids.numel())
                    grid = _grid(helpers, "beta", group_id, t)
                    covered = torch.zeros(n_rows, n_cols, dtype=torch.bool)
                    src_groups = _one_group(blocks.by_beta, group_id)
                    for src_tile in tiles:
                        row_lo, row_hi = _slab_positions(
                            source_group.alpha_uids, src_tile.alpha_start, src_tile.alpha_stop
                        )
                        col_lo, col_hi = _slab_positions(
                            source_group.beta_uids, src_tile.beta_start, src_tile.beta_stop
                        )
                        rectangle = covered[row_lo:row_hi, col_lo:col_hi]
                        assert not bool(rectangle.any()), "source rectangles overlap"
                        rectangle.fill_(True)
                        buf = grid[row_lo:row_hi, col_lo:col_hi].contiguous().reshape(-1)
                        seen_empty |= buf.numel() == 0
                        before = got.clone()
                        add_ab_slab(
                            got,
                            buf,
                            helpers,
                            integrals,
                            src_groups,
                            frag,
                            src_row_offset=row_lo,
                            src_row_stop=row_hi,
                            src_col_offset=col_lo,
                            src_col_stop=col_hi,
                        )
                        if buf.numel() == 0:
                            assert torch.equal(got, before), "an empty slab changed the output"
                    assert bool(covered.all()), "source rectangles do not cover the group"
                label = f"{basis_name} rank={tile.rank} αβ slab seed={seed}"
                _assert_close(got, expected[frag.pair_index], label)

    if basis_name == "tier":
        # Its one-row and one-column blocks put a whole slab outside some
        # group's grid, so the sweep exercises the empty-rectangle path.
        assert seen_empty, "the tier fixture must exercise an empty source rectangle"


def _edge_setup() -> tuple[PairBasis, int, TorchIntegrals, PairHelpers, list[PairTileBounds]]:
    """The tier basis, real integrals, block helpers and the 2x2 tiles."""
    basis, norb = _basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    assert helpers.blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, _N_RANKS, grid=_GRID)
    return basis, norb, integrals, helpers, _tiles(partition, basis)


def test_empty_source_slabs_add_nothing() -> None:
    """A source slab with no rows or columns leaves the output untouched."""
    basis, _norb, integrals, helpers, tiles = _edge_setup()
    blocks = helpers.blocks
    assert blocks is not None
    t = _random_vector(basis.n_pairs, 61, complex_values=False)
    empty = torch.zeros(0, dtype=t.dtype)
    found = False

    for tile in tiles:
        for frag in pair_tile_fragments(blocks.by_beta, tile):
            group = blocks.by_beta.groups[frag.group]
            for src_tile in tiles:
                row_lo, row_hi = _slab_positions(
                    group.alpha_uids, src_tile.alpha_start, src_tile.alpha_stop
                )
                col_lo, col_hi = _slab_positions(
                    group.beta_uids, src_tile.beta_start, src_tile.beta_stop
                )
                if row_hi > row_lo and col_hi > col_lo:
                    continue
                found = True
                Wb = _random_vector(frag.n_rows * frag.n_cols, 1, complex_values=False)
                before = Wb.clone()
                add_ab_slab(
                    Wb,
                    empty,
                    helpers,
                    integrals,
                    _one_group(blocks.by_beta, frag.group),
                    frag,
                    src_row_offset=row_lo,
                    src_row_stop=row_hi,
                    src_col_offset=col_lo,
                    src_col_stop=col_hi,
                )
                assert torch.equal(Wb, before)
                same = before.clone()
                add_same_spin_slab(
                    same,
                    empty,
                    helpers,
                    integrals,
                    blocks.by_beta,
                    frag,
                    row_offset=row_lo,
                    col_offset=col_lo,
                )
                assert torch.equal(same, before)

        for frag in pair_tile_fragments(blocks.by_alpha, tile):
            group = blocks.by_alpha.groups[frag.group]
            for src_tile in tiles:
                row_lo, row_hi = _slab_positions(
                    group.alpha_uids, src_tile.alpha_start, src_tile.alpha_stop
                )
                col_lo, col_hi = _slab_positions(
                    group.beta_uids, src_tile.beta_start, src_tile.beta_stop
                )
                if row_hi > row_lo and col_hi > col_lo:
                    continue
                found = True
                Wb = _random_vector(frag.n_rows * frag.n_cols, 1, complex_values=False)
                before = Wb.clone()
                add_same_spin_slab(
                    Wb,
                    empty,
                    helpers,
                    integrals,
                    blocks.by_alpha,
                    frag,
                    row_offset=row_lo,
                    col_offset=col_lo,
                )
                assert torch.equal(Wb, before)

    assert found, "the tier fixture must have an empty source-slab intersection"


def _sparse_copy(t: Tensor, pair_idx: Tensor, grid: Tensor) -> Tensor:
    """A copy of ``t`` whose block slice is replaced by a sparse grid."""
    sparse = torch.zeros_like(t)
    sparse[pair_idx] = grid.reshape(-1)
    return sparse


@pytest.mark.parametrize("complex_values", [False, True])
def test_single_source_and_partial_rectangles(complex_values: bool) -> None:
    """Single-row/column slabs and partly overlapping rectangles match references.

    A one-row α source, a one-column β source and a one-pair or 2x2 αβ source
    rectangle are staged on their own; the matching whole-channel kernel on the
    same sparsely filled vector is the reference. These are the cases where
    only part of a fragment's inverse moves lands inside the source rectangle.
    """
    basis, norb = _basis("tier")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    blocks = helpers.blocks
    assert blocks is not None
    off = generate_pair_helpers(basis, norb, device="cpu", block_matvec="off")
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, _N_RANKS, grid=_GRID)
    tiles = _tiles(partition, basis)
    t = _random_vector(basis.n_pairs, 71, complex_values=complex_values)
    saw_nonzero = False

    for tile in tiles:
        if tile.is_empty:
            continue
        for frag in pair_tile_fragments(blocks.by_beta, tile):
            group = blocks.by_beta.groups[frag.group]
            n_rows = int(group.alpha_uids.numel())
            n_cols = int(group.beta_uids.numel())
            grid = _grid(helpers, "beta", frag.group, t)
            alpha_moves = _single_moves(
                group.alpha_uids[frag.row_start : frag.row_stop],
                group.alpha_uids,
                helpers.unique_alphas,
                n_orb=norb,
                n_particles=int(helpers.n_a),
                spin_offset=0,
            )
            beta_moves = _single_moves(
                group.beta_uids[frag.col_start : frag.col_stop],
                group.beta_uids,
                helpers.unique_betas,
                n_orb=norb,
                n_particles=int(helpers.n_b),
                spin_offset=1,
            )
            if alpha_moves is None or beta_moves is None:
                continue
            # Ket positions reached by an inverse single move from the
            # fragment, so the source slab really carries survivors.
            src_row = int(alpha_moves[1][0])
            src_col = int(beta_moves[1][0])

            # α: one source row.
            buf = grid[src_row, frag.col_start : frag.col_stop].contiguous()
            sparse_grid = torch.zeros(n_rows, n_cols, dtype=t.dtype)
            sparse_grid[src_row, frag.col_start : frag.col_stop] = buf
            sparse = _sparse_copy(t, group.pair_idx, sparse_grid)
            reference = _generic_same_spin_channels(sparse, basis, off, norb, integrals)
            got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
            add_same_spin_slab(
                got,
                buf,
                helpers,
                integrals,
                blocks.by_beta,
                frag,
                row_offset=src_row,
                col_offset=0,
            )
            _assert_close(got, reference["alpha"][frag.pair_index], "single α row slab")
            saw_nonzero |= float(reference["alpha"][frag.pair_index].abs().max()) > 0

            # αβ: one source pair and a 2x2 rectangle around it.
            rectangles = (
                (src_row, src_row + 1, src_col, src_col + 1),
                (src_row, min(src_row + 2, n_rows), src_col, min(src_col + 2, n_cols)),
            )
            for row_lo, row_hi, col_lo, col_hi in rectangles:
                buf = grid[row_lo:row_hi, col_lo:col_hi].contiguous().reshape(-1)
                sparse_grid = torch.zeros(n_rows, n_cols, dtype=t.dtype)
                sparse_grid[row_lo:row_hi, col_lo:col_hi] = grid[row_lo:row_hi, col_lo:col_hi]
                sparse = _sparse_copy(t, group.pair_idx, sparse_grid)
                expected = torch.zeros_like(t)
                add_ab_blocks(expected, sparse, helpers, integrals)
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                add_ab_slab(
                    got,
                    buf,
                    helpers,
                    integrals,
                    _one_group(blocks.by_beta, frag.group),
                    frag,
                    src_row_offset=row_lo,
                    src_row_stop=row_hi,
                    src_col_offset=col_lo,
                    src_col_stop=col_hi,
                )
                _assert_close(got, expected[frag.pair_index], "single/partial αβ rectangle")
                saw_nonzero |= float(expected[frag.pair_index].abs().max()) > 0

        for frag in pair_tile_fragments(blocks.by_alpha, tile):
            group = blocks.by_alpha.groups[frag.group]
            n_rows = int(group.alpha_uids.numel())
            n_cols = int(group.beta_uids.numel())
            grid = _grid(helpers, "alpha", frag.group, t)
            beta_moves = _single_moves(
                group.beta_uids[frag.col_start : frag.col_stop],
                group.beta_uids,
                helpers.unique_betas,
                n_orb=norb,
                n_particles=int(helpers.n_b),
                spin_offset=1,
            )
            if beta_moves is None:
                continue

            # β: one source column.
            src_col = int(beta_moves[1][0])
            buf = grid[frag.row_start : frag.row_stop, src_col].contiguous()
            sparse_grid = torch.zeros(n_rows, n_cols, dtype=t.dtype)
            sparse_grid[frag.row_start : frag.row_stop, src_col] = buf
            sparse = _sparse_copy(t, group.pair_idx, sparse_grid)
            reference = _generic_same_spin_channels(sparse, basis, off, norb, integrals)
            got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
            add_same_spin_slab(
                got,
                buf,
                helpers,
                integrals,
                blocks.by_alpha,
                frag,
                row_offset=0,
                col_offset=src_col,
            )
            _assert_close(got, reference["beta"][frag.pair_index], "single β column slab")
            saw_nonzero |= float(reference["beta"][frag.pair_index].abs().max()) > 0

    assert saw_nonzero, "the single/partial source checks must produce contributions"
