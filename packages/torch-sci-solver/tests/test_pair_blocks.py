"""Tests for the exact Cartesian-block decomposition of a pair basis."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.pair_blocks import build_pair_blocks
from torch_sci_solver.torch_impl.pairs import PairBasis, pair_basis_from_arrays


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


def _tier_blocks(norb: int, nelec: int, max_level: int, drop_in_beta: float = 0.0) -> PairBasis:
    """A tier-block basis mirroring the scaling benchmark's ``gdb_pairs``.

    Every block is ``tiers[a] × tiers[b]``; at ``drop_in_beta > 0`` each block's
    β side is subsampled independently, which splits β strings that would
    otherwise share one α-partner set.
    """
    tiers = [_strings(norb, nelec, level) for level in range(max_level + 1)]
    rng = np.random.default_rng(7)
    alphas: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    for level in range(max_level + 1):
        for a in range(level + 1):
            b = level - a
            sa = tiers[a]
            sb = tiers[b]
            if drop_in_beta:
                keep = max(1, int(len(sb) * (1.0 - drop_in_beta)))
                sb = np.sort(rng.choice(sb, keep, replace=False))
            alphas.append(np.repeat(sa, len(sb)))
            betas.append(np.tile(sb, len(sa)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas))


def _assert_exact(basis: PairBasis, groups, axis: str) -> None:
    """Every group is its α×β grid and the groups partition the basis."""
    n = basis.n_pairs
    alpha = basis.alpha_of_pair
    beta = basis.beta_of_pair
    seen = torch.cat([g.pair_idx for g in groups.groups])
    assert seen.numel() == n
    assert torch.equal(torch.sort(seen).values, torch.arange(n))
    for g in groups.groups:
        a, b = int(g.alpha_uids.numel()), int(g.beta_uids.numel())
        assert int(g.pair_idx.numel()) == a * b
        if g.pair_idx.numel() > 1:
            assert bool(torch.all(g.pair_idx[1:] > g.pair_idx[:-1]))
        got_alpha = alpha[g.pair_idx].view(a, b)
        got_beta = beta[g.pair_idx].view(a, b)
        expected_alpha = basis.unique_alphas[g.alpha_uids].unsqueeze(1).expand(a, b)
        expected_beta = basis.unique_betas[g.beta_uids].unsqueeze(0).expand(a, b)
        assert torch.equal(got_alpha, expected_alpha)
        assert torch.equal(got_beta, expected_beta)
    assert groups.axis == axis


def test_full_product_is_one_block() -> None:
    """A complete product collapses to exactly one group on each axis."""
    basis = _full_product(4, 2)
    blocks = build_pair_blocks(basis, mode="on")
    assert blocks is not None
    for groups in (blocks.by_beta, blocks.by_alpha):
        assert len(groups.groups) == 1
        _assert_exact(basis, groups, groups.axis)
    group = blocks.by_beta.groups[0]
    assert group.alpha_uids.numel() == 6
    assert group.beta_uids.numel() == 6


def test_tier_blocks_partition_exactly() -> None:
    """The benchmark-shaped basis decomposes into its β tiers, exactly."""
    basis = _tier_blocks(6, 3, max_level=2)
    blocks = build_pair_blocks(basis, mode="on")
    assert blocks is not None
    lengths = sorted(int(g.pair_idx.numel()) for g in blocks.by_beta.groups)
    assert lengths == [9, 19, 90]
    _assert_exact(basis, blocks.by_beta, "beta")
    _assert_exact(basis, blocks.by_alpha, "alpha")


def test_tier_blocks_with_subsampled_sides_stay_exact() -> None:
    """Independent per-block β subsampling splits groups but stays exact."""
    basis = _tier_blocks(6, 3, max_level=2, drop_in_beta=0.5)
    blocks = build_pair_blocks(basis, mode="on")
    assert blocks is not None
    assert len(blocks.by_beta.groups) > 3  # the cut splits equal-partner buckets
    _assert_exact(basis, blocks.by_beta, "beta")
    _assert_exact(basis, blocks.by_alpha, "alpha")


def test_irregular_subset_is_exact() -> None:
    """A random subset decomposes exactly (many small groups)."""
    basis = _full_product(4, 2)
    keep = np.sort(np.random.default_rng(3).choice(basis.n_pairs, 26, replace=False))
    subset = pair_basis_from_arrays(
        basis.alpha_of_pair.numpy()[keep], basis.beta_of_pair.numpy()[keep]
    )
    blocks = build_pair_blocks(subset, mode="on")
    assert blocks is not None
    _assert_exact(subset, blocks.by_beta, "beta")
    _assert_exact(subset, blocks.by_alpha, "alpha")


def test_coalesces_equal_partner_sets() -> None:
    """Disjoint β sets with the same α partners form one group."""
    a = np.array([0b0011, 0b0101], dtype=np.int64)
    b1 = np.array([0b1001, 0b1010], dtype=np.int64)
    b2 = np.array([0b1100, 0b0110], dtype=np.int64)
    basis = pair_basis_from_arrays(
        np.concatenate([np.repeat(a, len(b1)), np.repeat(a, len(b2))]),
        np.concatenate([np.tile(b1, len(a)), np.tile(b2, len(a))]),
    )
    blocks = build_pair_blocks(basis, mode="on")
    assert blocks is not None
    assert len(blocks.by_beta.groups) == 1
    group = blocks.by_beta.groups[0]
    assert group.beta_uids.numel() == 4
    _assert_exact(basis, blocks.by_beta, "beta")


def test_auto_gate_and_mode_validation() -> None:
    """Small bases are gated out by default; bad modes raise."""
    basis = _tier_blocks(6, 3, max_level=2)
    assert build_pair_blocks(basis, mode="auto") is None
    built = build_pair_blocks(basis, mode="auto", min_pairs=1, min_group_pairs=1)
    assert built is not None
    assert build_pair_blocks(basis, mode="off") is None
    with pytest.raises(ValueError, match="block_matvec"):
        build_pair_blocks(basis, mode="sometimes")


def test_decomposition_maps_are_linear_in_n() -> None:
    """The two pair_idx maps together hold exactly one index per pair."""
    basis = _tier_blocks(6, 3, max_level=2)
    blocks = build_pair_blocks(basis, mode="on")
    assert blocks is not None
    for groups in (blocks.by_beta, blocks.by_alpha):
        total = sum(int(g.pair_idx.numel()) for g in groups.groups)
        assert total == basis.n_pairs
    assert blocks.stats["block_bytes"] == 2 * basis.n_pairs * 8 + sum(
        (int(g.alpha_uids.numel()) + int(g.beta_uids.numel())) * 8
        for groups in (blocks.by_beta, blocks.by_alpha)
        for g in groups.groups
    )


def test_byte_cap_and_helper_plumbing() -> None:
    """A tiny decomposition cap disables the build through the helper knob."""
    from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers

    basis = _tier_blocks(6, 3, max_level=2)
    assert build_pair_blocks(basis, mode="on", byte_cap=1) is None
    helpers = generate_pair_helpers(basis, 6, device="cpu", block_matvec="on", block_table_bytes=1)
    assert helpers.blocks is None
    helpers = generate_pair_helpers(basis, 6, device="cpu", block_matvec="on")
    assert helpers.blocks is not None
