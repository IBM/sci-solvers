"""Parity tests for the fragment-generalized GDB block kernels.

For every fragment of a tiled block decomposition this checks:

1. The αβ channel on a fragment equals the whole-channel ``add_ab_blocks``
   output restricted to the fragment's global pair indices.
2. The α (``by_beta``) and β (``by_alpha``) same-spin channels on their
   fragments equal the matching generic streams restricted to the fragment,
   and the scatter-accumulated fragments equal the whole-channel
   ``add_same_spin_blocks`` output restricted to the tile.
3. Edge cases: a fragment with an interior column slab, a fragment with
   ``row_start > 0`` and a tile that owns no pair (an empty fragment set).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pytest
import torch
from test_mult_pair_blocks import _complex_integrals, _real_integrals  # noqa: E402
from test_pair_blocks import _full_product, _tier_blocks  # noqa: E402
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    add_ab_blocks,
    add_ab_fragment,
    add_same_spin_blocks,
    add_same_spin_fragment,
)
from torch_sci_solver.torch_impl.mult_pairs import (
    _scatter_double_pair,
    _scatter_single_pair,
)
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers, generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiles import (
    PairFragment,
    PairTileBounds,
    pair_tile_bounds,
    pair_tile_fragments,
)
from torch_sci_solver.torch_impl.pairs import PairBasis

# A 2x2 rank grid over the uid axes; each tile is one slab of each axis.
_GRID = (2, 2)


def _basis(name: str) -> tuple[PairBasis, int]:
    """One named small block-decomposed basis and its spatial orbital count."""
    if name == "full_product":
        return _full_product(4, 2), 4
    if name == "tier":
        return _tier_blocks(6, 3, max_level=2), 6
    raise ValueError(name)


def _helpers(basis: PairBasis, norb: int, budget_bytes: int | None = None) -> PairHelpers:
    """Pair helpers with the block decomposition forced on."""
    helpers = generate_pair_helpers(
        basis, norb, device="cpu", budget_bytes=budget_bytes, block_matvec="on"
    )
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


def _assert_close(got: Tensor, expected: Tensor, label: str) -> None:
    """Assert agreement to ``atol=1e-12`` with no relative tolerance."""
    if got.numel() == 0 and expected.numel() == 0:
        return
    diff = float((got - expected).abs().max())
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0), (label, diff)


def _stage_alpha(helpers: PairHelpers, t: Tensor, frag: PairFragment) -> Tensor:
    """The α stage buffer: full group rows by the fragment's β columns."""
    blocks = helpers.blocks
    assert blocks is not None
    group = blocks.by_beta.groups[frag.group]
    n_rows = int(group.alpha_uids.numel())
    n_cols = int(group.beta_uids.numel())
    grid = t[group.pair_idx].view(n_rows, n_cols)
    return grid[:, frag.col_start : frag.col_stop].contiguous().reshape(-1)


def _stage_beta(helpers: PairHelpers, t: Tensor, frag: PairFragment) -> Tensor:
    """The β stage buffer: the fragment's α rows at the full group width."""
    blocks = helpers.blocks
    assert blocks is not None
    group = blocks.by_alpha.groups[frag.group]
    n_rows = int(group.alpha_uids.numel())
    n_cols = int(group.beta_uids.numel())
    grid = t[group.pair_idx].view(n_rows, n_cols)
    return grid[frag.row_start : frag.row_stop].contiguous().reshape(-1)


def _generic_same_spin_channels(
    t: Tensor,
    basis: PairBasis,
    helpers: PairHelpers,
    norb: int,
    integrals: TorchIntegrals,
) -> dict[str, Tensor]:
    """The generic streamed α-only and β-only same-spin contributions."""
    dtype = torch.complex128 if t.is_complex() else torch.float64
    out = {
        "alpha": torch.zeros(basis.n_pairs, dtype=dtype),
        "beta": torch.zeros(basis.n_pairs, dtype=dtype),
    }
    common = dict(
        alpha_of_pair=basis.alpha_of_pair,
        beta_of_pair=basis.beta_of_pair,
        n_orb=norb,
        integrals=integrals,
    )
    channels = (
        ("alpha", helpers.asingles, helpers.adoubles),
        ("beta", helpers.bsingles, helpers.bdoubles),
    )
    for name, singles, doubles in channels:
        for table in singles():
            _scatter_single_pair(
                Wb=out[name],
                T=t,
                k_bra=table["k_bra"],
                k_ket=table["k_ket"],
                cr=table["cr0"],
                an=table["an0"],
                chunk_size=10_000,
                **common,
            )
        for table in doubles():
            _scatter_double_pair(
                Wb=out[name],
                T=t,
                k_bra=table["k_bra"],
                k_ket=table["k_ket"],
                cr0=table["cr0"],
                cr1=table["cr1"],
                an0=table["an0"],
                an1=table["an1"],
                chunk_size=10_000,
                **common,
            )
    return out


@pytest.mark.parametrize("basis_name", ["full_product", "tier"])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("budget_bytes", [None, 1024], ids=["default", "tiny"])
def test_fragments_match_whole_channel_kernels(
    basis_name: str, complex_values: bool, budget_bytes: int | None
) -> None:
    """Every fragment reproduces its slice of the whole-channel kernels.

    The αβ fragment is compared row-for-row with ``add_ab_blocks`` restricted
    to the fragment's global pair indices. Each same-spin fragment is compared
    with the matching generic stream, and the two channels scattered through
    ``local_of_global`` reproduce ``add_same_spin_blocks`` on the whole tile.
    A tiny byte budget forces the enumeration and survivor chunks below the
    group sizes, so the chunk offsets are guarded too.
    """
    basis, norb = _basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb, budget_bytes)
    blocks = helpers.blocks
    assert blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID)
    off = generate_pair_helpers(basis, norb, device="cpu", block_matvec="off")
    seen_row_offset = False
    seen_col_offset = False

    for seed in (11, 12):
        t = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
        expected_ab = torch.zeros_like(t)
        add_ab_blocks(expected_ab, t, helpers, integrals)
        expected_same_spin = torch.zeros_like(t)
        add_same_spin_blocks(expected_same_spin, t, helpers, integrals)
        generic = _generic_same_spin_channels(t, basis, off, norb, integrals)

        for rank in range(4):
            tile = pair_tile_bounds(partition, basis, rank)
            if tile.is_empty:
                continue
            got_same_spin = torch.zeros(tile.local_size, dtype=t.dtype)

            for frag in pair_tile_fragments(blocks.by_beta, tile):
                seen_row_offset |= frag.row_start > 0
                label = f"{basis_name} rank={rank} αβ seed={seed} {frag}"
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                add_ab_fragment(got, t, helpers, integrals, blocks.by_beta, frag)
                _assert_close(got, expected_ab[frag.pair_index], label)

                label = f"{basis_name} rank={rank} α seed={seed} {frag}"
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                add_same_spin_fragment(
                    got, _stage_alpha(helpers, t, frag), helpers, integrals, blocks.by_beta, frag
                )
                _assert_close(got, generic["alpha"][frag.pair_index], label)
                got_same_spin.index_add_(0, tile.local_of_global(frag.pair_index), got)

            for frag in pair_tile_fragments(blocks.by_alpha, tile):
                seen_col_offset |= frag.col_start > 0
                label = f"{basis_name} rank={rank} β seed={seed} {frag}"
                got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
                add_same_spin_fragment(
                    got, _stage_beta(helpers, t, frag), helpers, integrals, blocks.by_alpha, frag
                )
                _assert_close(got, generic["beta"][frag.pair_index], label)
                got_same_spin.index_add_(0, tile.local_of_global(frag.pair_index), got)

            label = f"{basis_name} rank={rank} same-spin tile seed={seed}"
            _assert_close(got_same_spin, expected_same_spin[tile.pair_index], label)

    # The 2x2 grid must actually exercise both non-zero offsets.
    assert seen_row_offset and seen_col_offset


@pytest.mark.parametrize("complex_values", [False, True])
def test_interior_column_fragment(complex_values: bool) -> None:
    """A fragment with ``col_start > 0`` and ``col_stop < n_cols`` matches the full channels."""
    basis, norb = _basis("full_product")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers(basis, norb)
    blocks = helpers.blocks
    assert blocks is not None
    t = _random_vector(basis.n_pairs, 21, complex_values=complex_values)
    expected_ab = torch.zeros_like(t)
    add_ab_blocks(expected_ab, t, helpers, integrals)
    generic = _generic_same_spin_channels(
        t,
        basis,
        generate_pair_helpers(basis, norb, device="cpu", block_matvec="off"),
        norb,
        integrals,
    )

    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 3, grid=(1, 3))
    tile = pair_tile_bounds(partition, basis, 1)  # middle column slab
    fragments = pair_tile_fragments(blocks.by_beta, tile)
    assert fragments
    for frag in fragments:
        group = blocks.by_beta.groups[frag.group]
        assert frag.col_start > 0 and frag.col_stop < int(group.beta_uids.numel())

        got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
        add_ab_fragment(got, t, helpers, integrals, blocks.by_beta, frag)
        _assert_close(got, expected_ab[frag.pair_index], "interior column αβ")

        got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
        add_same_spin_fragment(
            got, _stage_alpha(helpers, t, frag), helpers, integrals, blocks.by_beta, frag
        )
        _assert_close(got, generic["alpha"][frag.pair_index], "interior column α")


def test_fragment_with_row_offset() -> None:
    """A fragment with ``row_start > 0`` matches the full channels."""
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    helpers = _helpers(basis, norb)
    blocks = helpers.blocks
    assert blocks is not None
    t = _random_vector(basis.n_pairs, 23, complex_values=False)
    expected_ab = torch.zeros_like(t)
    add_ab_blocks(expected_ab, t, helpers, integrals)
    generic = _generic_same_spin_channels(
        t,
        basis,
        generate_pair_helpers(basis, norb, device="cpu", block_matvec="off"),
        norb,
        integrals,
    )

    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 3, grid=(3, 1))
    tile = pair_tile_bounds(partition, basis, 1)  # middle row slab
    fragments = pair_tile_fragments(blocks.by_beta, tile)
    assert fragments
    for frag in fragments:
        assert frag.row_start > 0

        got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
        add_ab_fragment(got, t, helpers, integrals, blocks.by_beta, frag)
        _assert_close(got, expected_ab[frag.pair_index], "row offset αβ")

        got = torch.zeros(frag.n_rows * frag.n_cols, dtype=t.dtype)
        add_same_spin_fragment(
            got, _stage_alpha(helpers, t, frag), helpers, integrals, blocks.by_beta, frag
        )
        _assert_close(got, generic["alpha"][frag.pair_index], "row offset α")


def test_empty_tiles_have_no_fragments() -> None:
    """A tile that owns no pair has an empty fragment set on both axes."""
    basis, norb = _basis("full_product")
    helpers = _helpers(basis, norb)
    blocks = helpers.blocks
    assert blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 8, grid=(8, 1))
    tiles: list[PairTileBounds] = [pair_tile_bounds(partition, basis, rank) for rank in range(8)]
    empty = [tile for tile in tiles if tile.is_empty]
    assert empty
    for tile in empty:
        assert int(tile.pair_index.numel()) == 0
        assert pair_tile_fragments(blocks.by_beta, tile) == ()
        assert pair_tile_fragments(blocks.by_alpha, tile) == ()


def test_fragment_kernels_require_pair_blocks() -> None:
    """The fragment entry points reject helpers built with ``block_matvec="off"``."""
    basis, norb = _basis("full_product")
    integrals = _real_integrals(norb)
    off = generate_pair_helpers(basis, norb, device="cpu", block_matvec="off")
    on = _helpers(basis, norb)
    blocks = on.blocks
    assert blocks is not None
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=_GRID)
    tile = pair_tile_bounds(partition, basis, 0)
    ab_frag = pair_tile_fragments(blocks.by_beta, tile)[0]
    ss_frag = pair_tile_fragments(blocks.by_alpha, tile)[0]
    t = _random_vector(basis.n_pairs, 0, complex_values=False)

    with pytest.raises(ValueError, match="blocks"):
        add_ab_fragment(torch.zeros(1), t, off, integrals, blocks.by_beta, ab_frag)
    with pytest.raises(ValueError, match="blocks"):
        add_same_spin_fragment(torch.zeros(1), t, off, integrals, blocks.by_alpha, ss_frag)
    with pytest.raises(ValueError, match="by_beta"):
        add_ab_fragment(torch.zeros(1), t, on, integrals, blocks.by_alpha, ss_frag)
