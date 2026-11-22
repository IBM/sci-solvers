"""Tests for the structural (block-described) pair basis.

Covers:

1. Construction from tier blocks: block-major pair order, member strings, and
   the absence of any length-N array on the basis or its groups.
2. Full and owner-computes block matvec parity against the materialised pair
   basis, real and complex, to 1e-12.
3. The tiled operator over structural fragments, single process with a
   full-vector stand-in exchange, 2x2 and 2x3 grids.
4. Refusal paths: the streamed tables, the full-diagonal builder, the S²
   operator and the per-pair work weights all reject a structural basis.
5. gloo, 2 ranks: the distributed GDB solver accepts ``structural_blocks``
   and reproduces the explicit-basis energy and roots, 1-D and tiled.
6. Multi-limb block specs (70 and 130 spatial orbitals): ``(M, n_limbs)``
   unique arrays, block-major ordering, ``pair_strings`` /
   ``pair_index_of_uids`` round-trips, full/owner-computes/tiled matvec
   parity against the explicit limb basis, and a 2-rank gloo tiled solve.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import functools
import pathlib

import numpy as np
import pytest
import torch
from test_distributed_pairs import _init_gloo, _run_spawned, _synthetic_h4  # noqa: E402
from test_distributed_pairs_tile import _FullVectorExchange  # noqa: E402
from test_mult_pair_blocks import _complex_integrals, _real_integrals  # noqa: E402
from test_pair_blocks import _strings  # noqa: E402
from test_rdm_pairs_multilimb import _rows_as_ints, _wide_strings  # noqa: E402
from torch import Tensor
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl import distributed_pairs_tile as dpt
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.mult_pair_blocks import pair_work_weights
from torch_sci_solver.torch_impl.mult_pairs import (
    make_diag_terms_pair_local,
    make_diag_terms_pair_torch,
    mult_pair_local_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiles import pair_tile_bounds
from torch_sci_solver.torch_impl.pairs import (
    StructuralPairBasis,
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

NORB = 5
NELEC = 2
NELEC_PAIR = (2, 2)
H4_NORB = 4
MAX_LEVEL = 2


def _tier_spec() -> list[tuple[np.ndarray, np.ndarray]]:
    """The benchmark-shaped tier blocks, as structural block specifications."""
    tiers = [_strings(NORB, NELEC, level) for level in range(MAX_LEVEL + 1)]
    return [
        (tiers[level - b], tiers[b]) for level in range(MAX_LEVEL + 1) for b in range(level + 1)
    ]


def _structural() -> StructuralPairBasis:
    """The small tier basis, described structurally."""
    return structural_pair_basis_from_blocks(_tier_spec())


def _explicit() -> object:
    """The same tier basis as an explicit, lex-sorted pair list."""
    alphas, betas = [], []
    for alpha, beta in _tier_spec():
        alphas.append(np.repeat(alpha, len(beta)))
        betas.append(np.tile(beta, len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas))


def _structural_to_explicit_perm(basis: StructuralPairBasis, explicit) -> Tensor:
    """Structural pair index -> explicit pair index, by (α, β) bitmask."""
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    lookup = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(
            zip(explicit.alpha_of_pair.tolist(), explicit.beta_of_pair.tolist())
        )
    }
    return torch.tensor([lookup[(int(a), int(b))] for a, b in zip(alpha, beta)])


def _wide_spec(n_orb: int) -> list[tuple[list[int], list[int]]]:
    """Three disjoint pair blocks whose strings straddle the limb boundary."""
    strings = _wide_strings(n_orb)
    return [(strings[:3], strings[:3]), (strings[:3], strings[3:]), (strings[3:], strings[:3])]


def _structural_wide(n_orb: int) -> StructuralPairBasis:
    """The wide structural basis of :func:`_wide_spec`."""
    return structural_pair_basis_from_blocks(_wide_spec(n_orb))


def _explicit_wide(n_orb: int):
    """The same wide block spec as an explicit lex-sorted limb pair list."""
    alphas, betas = [], []
    for alpha, beta in _wide_spec(n_orb):
        alphas.append(np.repeat(np.array(alpha, dtype=object), len(beta)))
        betas.append(np.tile(np.array(beta, dtype=object), len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas), n_orb=n_orb)


def _wide_perm(basis: StructuralPairBasis, explicit) -> Tensor:
    """Structural pair index -> explicit pair index, by arbitrary-precision string."""
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    lookup = {
        (a, b): index
        for index, (a, b) in enumerate(
            zip(
                _rows_as_ints(explicit.alpha_of_pair),
                _rows_as_ints(explicit.beta_of_pair),
            )
        )
    }
    return torch.tensor([lookup[(a, b)] for a, b in zip(_rows_as_ints(alpha), _rows_as_ints(beta))])


@functools.lru_cache(maxsize=None)
def _wide_integrals(n_orb: int, complex_values: bool):
    """The 70-orbital integral bundle, built once per dtype for the wide tests."""
    return _complex_integrals(n_orb) if complex_values else _real_integrals(n_orb)


def test_construction_is_block_major_and_has_no_pair_arrays() -> None:
    """Pair indices follow the block formula and no length-N array is stored."""
    basis = _structural()
    explicit = _explicit()
    assert basis.n_pairs == explicit.n_pairs

    # Block formula: pair k of a block is base + row * n_beta + col.
    for block in basis.blocks:
        row = block.n_alpha - 1
        col = block.n_beta - 1
        index = block.pair_base + row * block.n_beta + col
        alpha, beta = basis.pair_strings(torch.tensor([index]))
        assert int(alpha[0]) == int(basis.unique_alphas[block.alpha_start + row])
        assert int(beta[0]) == int(basis.unique_betas[block.beta_start + col])

    # Every stored tensor is O(M_unique) / O(n_blocks), never O(N).
    n = basis.n_pairs
    for value in vars(basis).values():
        if isinstance(value, Tensor):
            assert value.numel() < n
    assert basis.alpha_block_ids.shape[0] == basis.n_unique_alphas
    assert basis.beta_block_ids.shape[0] == basis.n_unique_betas

    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    assert helpers.structural is basis
    assert helpers.alpha_of_pair is None and helpers.beta_of_pair is None
    assert helpers.pair_keys is None
    assert helpers.partners_a_offsets is None and helpers.partners_b_offsets is None
    assert helpers.blocks is not None and helpers.blocks.structural
    for groups in (helpers.blocks.by_beta, helpers.blocks.by_alpha):
        assert all(group.pair_idx is None for group in groups.groups)
        assert sum(group.n_pairs for group in groups.groups) == n


def test_block_formula_matches_pair_index_of_uids() -> None:
    """``pair_index_of_uids`` inverts ``pair_uids`` on every pair."""
    basis = _structural()
    indices = torch.arange(basis.n_pairs)
    alpha_uid, beta_uid = basis.pair_uids(indices)
    found, located = basis.pair_index_of_uids(alpha_uid, beta_uid)
    assert bool(found.all())
    assert torch.equal(located, indices)


@pytest.mark.parametrize("complex_values", [False, True])
def test_full_matvec_matches_materialised(complex_values: bool) -> None:
    """The structural block matvec equals the explicit-basis matvec."""
    basis = _structural()
    explicit = _explicit()
    perm = _structural_to_explicit_perm(basis, explicit)
    integrals = _complex_integrals(NORB) if complex_values else _real_integrals(NORB)
    dtype = torch.complex128 if complex_values else torch.float64

    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, NORB, integrals)
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    hii_exp = make_diag_terms_pair_local(
        explicit.alpha_of_pair, explicit.beta_of_pair, NORB, integrals
    )
    helpers_exp = generate_pair_helpers(explicit, NORB, block_matvec="on")

    x = torch.randn(basis.n_pairs, dtype=dtype)
    got = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)
    # ``perm`` maps structural -> explicit, so feed the explicit basis the
    # inverse map and read the result back through ``perm``.
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)
    expected = mult_pair_torch(hii_exp, x[inverse], explicit, helpers_exp, NORB, integrals)
    torch.testing.assert_close(got, expected[perm], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("complex_values", [False, True])
def test_owner_computes_matvec_matches_full(complex_values: bool) -> None:
    """A rank's owned block matches the same slice of the full matvec."""
    basis = _structural()
    integrals = _complex_integrals(NORB) if complex_values else _real_integrals(NORB)
    dtype = torch.complex128 if complex_values else torch.float64
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, NORB, integrals)
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    x = torch.randn(basis.n_pairs, dtype=dtype)
    full = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)

    for start, stop in [(0, 13), (13, 30), (30, basis.n_pairs), (7, 8)]:
        local = mult_pair_local_torch(
            hii[start:stop],
            x[start:stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            start,
        )
        torch.testing.assert_close(local, full[start:stop], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("grid", [(2, 2), (2, 3)])
def test_tiled_operator_matches_full(
    monkeypatch, complex_values: bool, grid: tuple[int, int]
) -> None:
    """Structural tile fragments reproduce the full matvec on every rank."""
    basis = _structural()
    integrals = _complex_integrals(NORB) if complex_values else _real_integrals(NORB)
    dtype = torch.complex128 if complex_values else torch.float64
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, NORB, integrals)
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    x = torch.randn(basis.n_pairs, dtype=dtype)
    expected = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)

    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    for rank in range(world_size):
        bounds = pair_tile_bounds(partition, basis, rank)
        _FullVectorExchange.full_vector = x
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            NORB,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = operator(x[bounds.pair_index])
        torch.testing.assert_close(got, expected[bounds.pair_index], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_orb,expected_limbs", [(70, 2), (130, 3)])
def test_wide_construction_is_block_major_and_limb_correct(n_orb: int, expected_limbs: int) -> None:
    """A >64-bit block spec stores limb matrices and keeps the block formulas.

    The unique arrays are ``(M_unique, n_limbs)`` and every stored tensor stays
    O(M_unique) / O(n_blocks), never O(N); ``pair_strings`` and
    ``pair_index_of_uids`` round-trip in arbitrary-precision Python ints.
    """
    basis = _structural_wide(n_orb)
    explicit = _explicit_wide(n_orb)
    assert basis.n_limbs == expected_limbs
    assert basis.unique_alphas.dim() == 2
    assert basis.unique_alphas.shape[1] == expected_limbs
    assert basis.unique_betas.shape[1] == expected_limbs
    assert basis.n_pairs == explicit.n_pairs == 27

    # Block formula: pair k of a block is base + row * n_beta + col, and the
    # stored strings are the arbitrary-precision ones of the block spec.
    for block in basis.blocks:
        row, col = block.n_alpha - 1, block.n_beta - 1
        index = block.pair_base + row * block.n_beta + col
        alpha, beta = basis.pair_strings(torch.tensor([index]))
        want_alpha = _rows_as_ints(
            basis.unique_alphas[block.alpha_start + row : block.alpha_start + row + 1]
        )[0]
        want_beta = _rows_as_ints(
            basis.unique_betas[block.beta_start + col : block.beta_start + col + 1]
        )[0]
        assert _rows_as_ints(alpha)[0] == want_alpha
        assert _rows_as_ints(beta)[0] == want_beta

    indices = torch.arange(basis.n_pairs)
    found, located = basis.pair_index_of_uids(*basis.pair_uids(indices))
    assert bool(found.all())
    assert torch.equal(located, indices)

    # The same (α, β) pair set as the explicit, lex-sorted limb basis.
    alpha, beta = basis.pair_strings(indices)
    structural_pairs = set(zip(_rows_as_ints(alpha), _rows_as_ints(beta)))
    explicit_pairs = set(
        zip(
            _rows_as_ints(explicit.alpha_of_pair),
            _rows_as_ints(explicit.beta_of_pair),
        )
    )
    assert structural_pairs == explicit_pairs

    # Every stored tensor is O(M_unique) / O(n_blocks), never O(N).
    for value in vars(basis).values():
        if isinstance(value, Tensor):
            assert value.numel() < basis.n_pairs

    helpers = generate_pair_helpers(basis, n_orb, block_matvec="on")
    assert helpers.structural is basis
    assert helpers.n_a == 2 and helpers.n_b == 2
    assert helpers.blocks is not None and helpers.blocks.structural
    for groups in (helpers.blocks.by_beta, helpers.blocks.by_alpha):
        assert sum(group.n_pairs for group in groups.groups) == basis.n_pairs


def test_wide_construction_accepts_pre_split_limb_blocks() -> None:
    """A block spec passed as ``(M, n_limbs)`` matrices builds the same basis."""
    n_orb = 70
    mask = (1 << 64) - 1

    def _split(strings: list[int]) -> np.ndarray:
        return np.array([[s & mask, s >> 64] for s in strings], dtype=np.int64)

    spec = _wide_spec(n_orb)
    split_spec = [(_split(alpha), _split(beta)) for alpha, beta in spec]
    basis = structural_pair_basis_from_blocks(split_spec)
    expected = _structural_wide(n_orb)
    assert basis.n_limbs == 2
    assert torch.equal(basis.unique_alphas, expected.unique_alphas)
    assert torch.equal(basis.unique_betas, expected.unique_betas)
    assert basis.n_pairs == expected.n_pairs


def test_n_orb_forces_limb_storage_above_64() -> None:
    """``n_orb`` widens an all-narrow block spec to ceil(n_orb / 64) limbs."""
    basis = structural_pair_basis_from_blocks([([3, 5], [3, 5])], n_orb=70)
    assert basis.n_limbs == 2
    assert basis.unique_alphas.shape == (2, 2)
    assert torch.equal(basis.unique_alphas[:, 1], torch.zeros(2, dtype=torch.int64))
    helpers = generate_pair_helpers(basis, 70, block_matvec="on")
    assert helpers.n_a == 2 and helpers.n_b == 2


@pytest.mark.parametrize("complex_values", [False, True])
def test_wide_structural_matvec_matches_materialised(complex_values: bool) -> None:
    """At 70 orbitals the limb block matvec equals the explicit-limb matvec."""
    n_orb = 70
    basis = _structural_wide(n_orb)
    explicit = _explicit_wide(n_orb)
    perm = _wide_perm(basis, explicit)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)
    integrals = _wide_integrals(n_orb, complex_values)
    dtype = torch.complex128 if complex_values else torch.float64

    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, n_orb, integrals)
    helpers = generate_pair_helpers(basis, n_orb, block_matvec="on")
    hii_exp = make_diag_terms_pair_local(
        explicit.alpha_of_pair, explicit.beta_of_pair, n_orb, integrals
    )
    helpers_exp = generate_pair_helpers(explicit, n_orb, block_matvec="on")

    x = torch.randn(basis.n_pairs, dtype=dtype)
    got = mult_pair_torch(hii, x, basis, helpers, n_orb, integrals)
    expected = mult_pair_torch(hii_exp, x[inverse], explicit, helpers_exp, n_orb, integrals)
    torch.testing.assert_close(got, expected[perm], atol=1e-12, rtol=0.0)

    for start, stop in [(0, 13), (13, 27), (5, 6)]:
        local = mult_pair_local_torch(
            hii[start:stop],
            x[start:stop],
            x,
            basis,
            helpers,
            n_orb,
            integrals,
            start,
        )
        torch.testing.assert_close(local, got[start:stop], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("complex_values", [False, True])
def test_wide_structural_tiled_operator_matches_full(monkeypatch, complex_values: bool) -> None:
    """Limb tile fragments reproduce the full matvec at 70 orbitals."""
    n_orb = 70
    basis = _structural_wide(n_orb)
    integrals = _wide_integrals(n_orb, complex_values)
    dtype = torch.complex128 if complex_values else torch.float64
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, n_orb, integrals)
    helpers = generate_pair_helpers(basis, n_orb, block_matvec="on")
    x = torch.randn(basis.n_pairs, dtype=dtype)
    expected = mult_pair_torch(hii, x, basis, helpers, n_orb, integrals)

    grid = (2, 2)
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    for rank in range(world_size):
        bounds = pair_tile_bounds(partition, basis, rank)
        _FullVectorExchange.full_vector = x
        operator = dpt.LocalPairTileOperator(
            hii[bounds.pair_index],
            basis,
            helpers,
            n_orb,
            integrals,
            bounds,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = operator(x[bounds.pair_index])
        torch.testing.assert_close(got, expected[bounds.pair_index], atol=1e-12, rtol=0.0)


def test_structural_swap_permutation_matches_explicit() -> None:
    """The structural α↔β swap map and τ signs equal the explicit ones."""
    from torch_sci_solver.torch_impl.qiskit_adapter import (
        _pair_ab_swap_permutation,
        _structural_pair_ab_swap_permutation,
    )

    basis = _structural()
    explicit = _explicit()
    perm = _structural_to_explicit_perm(basis, explicit)
    structural_swap, structural_tau = _structural_pair_ab_swap_permutation(
        basis, NORB, torch.device("cpu"), torch.float64
    )
    explicit_swap, explicit_tau = _pair_ab_swap_permutation(
        explicit, NORB, torch.device("cpu"), torch.float64
    )
    assert structural_swap is not None and explicit_swap is not None
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)
    # Structural index of each structural pair's swapped partner.
    expected_swap = inverse[explicit_swap[perm]]
    torch.testing.assert_close(structural_swap, expected_swap)
    torch.testing.assert_close(structural_tau, explicit_tau[perm])


@pytest.mark.parametrize("n_orb", [70, 130])
def test_wide_structural_swap_permutation_matches_explicit(n_orb: int) -> None:
    """On a swap-closed limb basis the structural swap map equals the explicit one."""
    from torch_sci_solver.torch_impl.qiskit_adapter import (
        _pair_ab_swap_permutation,
        _structural_pair_ab_swap_permutation,
    )

    basis = _structural_wide(n_orb)
    explicit = _explicit_wide(n_orb)
    assert basis.n_limbs > 1
    perm = _wide_perm(basis, explicit)
    structural_swap, structural_tau = _structural_pair_ab_swap_permutation(
        basis, n_orb, torch.device("cpu"), torch.float64
    )
    explicit_swap, explicit_tau = _pair_ab_swap_permutation(
        explicit, n_orb, torch.device("cpu"), torch.float64
    )
    assert structural_swap is not None and explicit_swap is not None
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)
    torch.testing.assert_close(structural_swap, inverse[explicit_swap[perm]])
    torch.testing.assert_close(structural_tau, explicit_tau[perm])
    # Owned-subset queries return the same partners in the given order.
    owned = torch.tensor([basis.n_pairs - 1, 0, 2], dtype=torch.int64)
    sub_swap, sub_tau = _structural_pair_ab_swap_permutation(
        basis, n_orb, torch.device("cpu"), torch.float64, pair_indices=owned
    )
    torch.testing.assert_close(sub_swap, structural_swap[owned])
    torch.testing.assert_close(sub_tau, structural_tau[owned])


def test_streamed_paths_refuse_structural_basis() -> None:
    """Everything except the RDM builders needs the explicit pair list."""
    basis = _structural()
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    for stream in (helpers.asingles, helpers.adoubles, helpers.bsingles, helpers.bdoubles):
        with pytest.raises(ValueError, match="structural pair basis"):
            stream()
    with pytest.raises(ValueError, match="structural pair basis"):
        helpers.ab_doubles()
    with pytest.raises(ValueError, match="structural pair basis"):
        helpers.nnz()
    with pytest.raises(ValueError, match="structural pair basis"):
        make_diag_terms_pair_torch(basis, NORB, _real_integrals(NORB))
    with pytest.raises(ValueError, match="structural pair basis"):
        PairSpinSqOperator(basis, NORB, NELEC, NELEC)
    with pytest.raises(ValueError, match="structural pair basis"):
        pair_work_weights(helpers)
    with pytest.raises(ValueError, match="block matvec path"):
        generate_pair_helpers(basis, NORB, block_matvec="off")


def test_rdm_builders_accept_structural_basis() -> None:
    """The RDM builders walk the blocks; they no longer need the pair list."""
    from torch_sci_solver.torch_impl.rdm_pairs import make_one_rdm_pair_torch

    basis = _structural()
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    eigvec = torch.zeros(basis.n_pairs, dtype=torch.float64)
    gamma_a, gamma_b = make_one_rdm_pair_torch(eigvec, basis, helpers, NORB)
    assert gamma_a.shape == (NORB, NORB)
    assert gamma_b.shape == (NORB, NORB)


def test_construction_rejects_overlapping_blocks() -> None:
    """Blocks that share both an α and a β string overlap and are rejected."""
    with pytest.raises(ValueError, match="pair-disjoint"):
        structural_pair_basis_from_blocks([([1, 2], [4, 5]), ([2, 3], [5, 6])])
    with pytest.raises(ValueError, match="contiguous uid range"):
        # The last block's α strings sit at positions {0, 2}, with no new
        # string to append, so they cannot form one range.
        structural_pair_basis_from_blocks([([1], [4]), ([2], [5]), ([3], [6]), ([1, 3], [7])])


def test_gdb_kwargs_reject_structural_with_streamed_path() -> None:
    """``structural_blocks`` is GDB-only and needs the block matvec."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _validate_gdb_kwargs

    blocks = [[(np.array([3]), np.array([3]))]]
    with pytest.raises(ValueError, match="structural_blocks"):
        _validate_gdb_kwargs({"structural_blocks": blocks, "block_matvec": "off"}, mode="gdb")
    with pytest.raises(ValueError):
        _validate_gdb_kwargs({"structural_blocks": blocks}, mode="tpb")
    validated = _validate_gdb_kwargs({"structural_blocks": blocks}, mode="gdb")
    assert validated["structural_blocks"] is blocks


def _structural_solver_worker(
    rank: int,
    world_size: int,
    expected_energy: float,
    tile: str,
    init_file: str,
    out_file: str,
) -> None:
    """Solve the H4 product basis structurally and compare with the reference."""
    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri, strings = _synthetic_h4()
        distributed = {
            "world_size": world_size,
            "backend": "gloo",
            "timeout": 120,
            "persistent": False,
            "matvec": "local",
        }
        if tile:
            distributed["tile"] = tile
        solver = make_torch_sci_solver(
            device="cpu",
            matvec_impl="batched",
            davidson_block_size=4,
            max_davidson_iters=200,
            davidson_tolerance=1e-8,
            n_roots=2,
            davidson_missed_root_check=False,
            verbose=False,
            mode="gdb",
            gdb_kwargs={
                "block_matvec": "on",
                "structural_blocks": [[(strings, strings)]],
            },
            distributed=distributed,
        )
        placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
        result = solver(placeholder, h1, eri, H4_NORB, NELEC_PAIR)
        if rank == 0:
            energies = [float(state.energy) for state in result[0].states]
            assert abs(energies[0] - expected_energy) < 1e-10, energies
            pathlib.Path(out_file).open("w").write("ok")
        else:
            pathlib.Path(out_file).open("w").write("ok")
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()


def _explicit_gdb_energy() -> float:
    """Single-process explicit-basis ground energy for the H4 product basis."""
    h1, eri, strings = _synthetic_h4()
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=4,
        max_davidson_iters=200,
        davidson_tolerance=1e-8,
        n_roots=2,
        davidson_missed_root_check=False,
        verbose=False,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
    )
    expanded = (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    return float(solver([expanded], h1, eri, H4_NORB, NELEC_PAIR)[0].states[0].energy)


WIDE_NORB = 70


def _wide_h4_blocks() -> list[tuple[list[int], list[int]]]:
    """H4's six 2-electron strings plus three limb-boundary strings."""
    _, _, low = _synthetic_h4()
    low_strings = [int(s) for s in low]
    high = [(1 << 0) | (1 << 64), (1 << 1) | (1 << 65), (1 << 64) | (1 << 65)]
    return [
        (low_strings, low_strings),
        (low_strings, high),
        (high, low_strings),
    ]


def _wide_h4_integrals() -> tuple[np.ndarray, np.ndarray]:
    """The 4-orbital H4 integrals embedded in 70 orbitals.

    Orbitals 64/65 get a positive one-body energy and no couplings, so the
    ground state stays in the low-orbital H4 sector while the pair basis spans
    both limbs.
    """
    h1_low, eri_low, _ = _synthetic_h4()
    h1 = np.zeros((WIDE_NORB, WIDE_NORB))
    h1[:4, :4] = h1_low
    h1[64, 64] = h1[65, 65] = 5.0
    eri = np.zeros((WIDE_NORB,) * 4)
    eri[:4, :4, :4, :4] = eri_low
    return h1, eri


def _wide_explicit_gdb_energy() -> float:
    """Single-process explicit-limb ground energy of the wide H4 basis."""
    h1, eri = _wide_h4_integrals()
    alphas, betas = [], []
    for alpha, beta in _wide_h4_blocks():
        alphas.append(np.repeat(np.array(alpha, dtype=object), len(beta)))
        betas.append(np.tile(np.array(beta, dtype=object), len(alpha)))
    explicit = pair_basis_from_arrays(
        np.concatenate(alphas), np.concatenate(betas), n_orb=WIDE_NORB
    )
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=4,
        max_davidson_iters=300,
        davidson_tolerance=1e-9,
        n_roots=1,
        davidson_missed_root_check=False,
        verbose=False,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
    )
    state = solver(
        [(explicit.alpha_of_pair.numpy(), explicit.beta_of_pair.numpy())],
        h1,
        eri,
        WIDE_NORB,
        NELEC_PAIR,
    )[0].states[0]
    return float(state.energy)


def _wide_structural_solver_worker(
    rank: int,
    world_size: int,
    expected_energy: float,
    apply_ab_projection: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Solve the wide H4 basis structurally over a 2-rank tiled gloo group."""
    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri = _wide_h4_integrals()
        solver = make_torch_sci_solver(
            device="cpu",
            matvec_impl="batched",
            davidson_block_size=4,
            max_davidson_iters=300,
            davidson_tolerance=1e-9,
            n_roots=1,
            davidson_missed_root_check=False,
            verbose=False,
            apply_ab_projection=apply_ab_projection,
            mode="gdb",
            gdb_kwargs={
                "block_matvec": "on",
                "structural_blocks": [_wide_h4_blocks()],
            },
            distributed={
                "world_size": world_size,
                "backend": "gloo",
                "timeout": 180,
                "persistent": False,
                "matvec": "local",
                "tile": "force",
            },
        )
        placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
        result = solver(placeholder, h1, eri, WIDE_NORB, NELEC_PAIR)
        if rank == 0:
            energy = float(result[0].states[0].energy)
            assert abs(energy - expected_energy) < 1e-10, energy
    finally:
        import torch.distributed as dist

        dist.destroy_process_group()
    pathlib.Path(out_file).open("w").write("ok")


@pytest.mark.slow
@pytest.mark.parametrize("apply_ab_projection", [False, True])
def test_gloo_wide_structural_solver_matches_explicit(tmp_path, apply_ab_projection: bool) -> None:
    """gloo, 2 ranks: a limb structural tiled solve matches the explicit one.

    With the projection on, the swap-closed limb basis runs the structural
    α↔β-swap projector and the singlet ground energy is unchanged.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    expected = _wide_explicit_gdb_energy()
    _run_spawned(
        _wide_structural_solver_worker,
        2,
        (expected, apply_ab_projection),
        tmp_path,
        "pair_struct_wide",
    )


@pytest.mark.parametrize("tile", ["", "force"])
def test_gloo_structural_solver_matches_explicit(tmp_path, tile: str) -> None:
    """gloo, 2 ranks: structural GDB reproduces the explicit ground energy."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    expected = _explicit_gdb_energy()
    name = "pair_struct_tile" if tile else "pair_struct_1d"
    _run_spawned(
        _structural_solver_worker,
        2,
        (expected, tile),
        tmp_path,
        name,
    )


def test_structural_pair_superset_check_accepts_the_basis_and_rejects_outsiders() -> None:
    """The warm-start superset check reads pair membership from the blocks.

    Every pair of the basis passes; an α/β combination no block spans fails
    even though both strings occur in the basis, and a string outside the
    basis fails outright.
    """
    from torch_sci_solver.torch_impl.qiskit_adapter import _check_structural_pair_superset

    basis = _structural()
    device = torch.device("cpu")
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    _check_structural_pair_superset(basis, alpha.tolist(), beta.tolist(), device)

    tiers = [_strings(NORB, NELEC, level) for level in range(MAX_LEVEL + 1)]
    # No block is (tier 1) × (tier 2): that would be a level-3 block.
    with pytest.raises(ValueError, match="superset"):
        _check_structural_pair_superset(basis, [int(tiers[1][0])], [int(tiers[2][0])], device)
    # A single-occupancy string is not a two-electron string of the basis.
    with pytest.raises(ValueError, match="superset"):
        _check_structural_pair_superset(basis, [1], [int(tiers[0][0])], device)


def test_gloo_structural_warm_start_reproduces_the_cold_energy() -> None:
    """gloo, 2 ranks: a structural warm-extend reproduces its cold energy.

    The cached state of a structural cold-with-probe solve feeds a second
    structural solve of the same block spec; the cached amplitudes embed by
    pair lookup and Davidson starts from them, so the Ritz value matches.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    h1, eri, strings = _synthetic_h4()
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=4,
        max_davidson_iters=200,
        davidson_tolerance=1e-8,
        n_roots=2,
        davidson_missed_root_check=False,
        verbose=False,
        mode="gdb",
        gdb_kwargs={
            "block_matvec": "on",
            "structural_blocks": [[(strings, strings)]],
        },
        distributed={
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "persistent": False,
            "matvec": "local",
        },
    )
    placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
    cold_results, states = solver(
        placeholder, h1, eri, H4_NORB, NELEC_PAIR, probe_method="warm_start"
    )
    cached = states[0]
    assert cached is not None
    cold_energy = float(cold_results[0].states[0].energy)
    assert abs(cold_energy - _explicit_gdb_energy()) < 1e-10

    warm_results = solver(
        placeholder,
        h1,
        eri,
        H4_NORB,
        NELEC_PAIR,
        probe_method="warm_start",
        probe_state=[cached],
    )
    warm_energy = float(warm_results[0].states[0].energy)
    assert abs(warm_energy - cold_energy) < 1e-8


def _structural_h4_solver(tile: str):
    """A one-root, 2-rank gloo solver on the structural H4 product basis."""
    _, _, strings = _synthetic_h4()
    distributed = {
        "world_size": 2,
        "backend": "gloo",
        "timeout": 120,
        "persistent": False,
        "matvec": "local",
    }
    if tile:
        distributed["tile"] = tile
    return make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=4,
        max_davidson_iters=200,
        davidson_tolerance=1e-9,
        n_roots=1,
        verbose=False,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on", "structural_blocks": [[(strings, strings)]]},
        distributed=distributed,
    )


@pytest.mark.parametrize("tile", ["", "force"])
def test_gloo_structural_init_vector_is_indexed_by_structural_pair(tile: str) -> None:
    """gloo, 2 ranks: the returned structural amplitudes seed a converged start.

    A structural init vector is laid out like the amplitudes a structural solve
    returns (structural pair order); fed back, each rank gathers its owned
    entries and Davidson converges at the first check.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    h1, eri, _ = _synthetic_h4()
    solver = _structural_h4_solver(tile)
    placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
    cold = solver(placeholder, h1, eri, H4_NORB, NELEC_PAIR)[0].states[0]
    amplitudes = np.asarray(cold.sci_state.amplitudes).reshape(-1)
    seeded = solver(
        placeholder,
        h1,
        eri,
        H4_NORB,
        NELEC_PAIR,
        init_vectors=[amplitudes],
        max_davidson_iters=1,
    )[0].states[0]
    assert seeded.state == "converged"
    assert float(seeded.energy) == pytest.approx(float(cold.energy), abs=1e-9)
    assert solver.stats["per_call_log"][-1]["subspaces"][0]["warm"]


def test_structural_init_vector_length_is_the_structural_pair_count() -> None:
    """``preflight`` accepts an n_pairs-long vector and rejects any other length."""
    h1, eri, strings = _synthetic_h4()
    solver = _structural_h4_solver("")
    placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
    n_pairs = len(strings) ** 2
    solver.preflight(placeholder, NELEC_PAIR, norb=H4_NORB, init_vectors=[np.ones(n_pairs)])
    with pytest.raises(ValueError, match=rf"expected \({n_pairs},\)"):
        solver.preflight(placeholder, NELEC_PAIR, norb=H4_NORB, init_vectors=[np.ones(n_pairs - 1)])


@pytest.mark.parametrize("tile", ["", "force"])
def test_gloo_structural_init_vector_on_a_multi_block_basis(tile: str) -> None:
    """gloo, 2 ranks: block-major init vectors on tier blocks out of lex order.

    The tier blocks put every pair of a level before the next level, so the
    structural order differs from the explicit basis' lex order; a vector
    indexed the wrong way would not converge at the first check.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    explicit = _explicit()
    basis = _structural()
    perm = _structural_to_explicit_perm(basis, explicit)
    assert not torch.equal(perm, torch.arange(basis.n_pairs))  # block order != lex order
    rng = np.random.default_rng(7)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = sum(raw.transpose(p) for p in perms) / 8.0
    distributed = {
        "world_size": 2,
        "backend": "gloo",
        "timeout": 120,
        "persistent": False,
        "matvec": "local",
    }
    if tile:
        distributed["tile"] = tile
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=6,
        max_davidson_iters=200,
        davidson_tolerance=1e-9,
        n_roots=1,
        verbose=False,
        mode="gdb",
        apply_ab_projection=False,
        gdb_kwargs={"block_matvec": "on", "structural_blocks": [_tier_spec()]},
        distributed=distributed,
    )
    placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
    cold = solver(placeholder, h1, eri, NORB, (NELEC, NELEC))[0].states[0]
    amplitudes = np.asarray(cold.sci_state.amplitudes).reshape(-1)
    seeded = solver(
        placeholder,
        h1,
        eri,
        NORB,
        (NELEC, NELEC),
        init_vectors=[amplitudes],
        max_davidson_iters=1,
    )[0].states[0]
    assert seeded.state == "converged"
    assert float(seeded.energy) == pytest.approx(float(cold.energy), abs=1e-9)
