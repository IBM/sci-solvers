"""Multi-limb (strings wider than 64 bits) support in the tiled GDB post-processing.

The tiled builders in ``torch_impl/pair_tiled_postprocess.py`` used to do 1-D
bit operations (``(strings.unsqueeze(1) >> p_range) & 1``, ``1 << p``,
``(1 << n_orb) - 1``). They now accept ``(M, n_limbs)`` limb matrices like the
serial pair builders do.

1. An 8-orbital basis carried as two limbs (limb 0 holds every bit, limb 1 is
   zero padding) gives bit-identical 1-RDMs, αα/αβ/ββ 2-RDMs, ``S²`` apply and
   diagonal to the same basis in 1-D form, for real and complex amplitudes.

2. At 70 orbitals a selected pair basis whose strings straddle the limb
   boundary (occupied orbitals 0/1 and 64/65) reproduces the direct
   explicit-pair builders of ``rdm_pairs.py`` / ``spin_sq_pairs.py``.

3. A 2x2 simulated tile grid sums its staged partials back to the serial
   observables, so the limb path is exercised beyond stage 0.

4. The partition helpers the tiled path is built on — ``pair_tile_bounds``,
   ``pair_work_weights`` and ``pair_tile_partition`` — size their uid axes and
   weight vectors by string, not by limb word, so a wide basis partitions
   exactly like its single-word form.

5. A structural (block-described) limb basis at 70/130 orbitals: the tiled
   observables equal the explicit-limb builders, and a 2x2 staged grid sums
   its partials back to the same 1-RDMs and ``S²`` results.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import replace

import numpy as np
import pytest
import torch
import torch_sci_solver.torch_impl.distributed_pairs_tile as dpt
import torch_sci_solver.torch_impl.pair_tiled_postprocess as ptp
from test_distributed_pairs_tile import _FullVectorExchange  # noqa: E402
from test_rdm_pairs_multilimb import _rows_as_ints  # noqa: E402
from test_rdm_pairs_multilimb import _wide_strings as _limb_wide_strings  # noqa: E402
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.mult_pair_blocks import pair_work_weights
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
    pair_tiled_one_rdm,
    pair_tiled_s2_apply,
    pair_tiled_s2_diagonal,
    pair_tiled_spin_squared,
    pair_tiled_two_rdm,
)
from torch_sci_solver.torch_impl.pair_tiles import (
    PairTileBounds,
    pair_tile_bounds,
    pair_tile_partition,
)
from torch_sci_solver.torch_impl.pairs import (
    PairBasis,
    StructuralPairBasis,
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)
from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm
from torch_sci_solver.torch_impl.rdm_pairs import (
    _inter_to_block_signs_pair,
    make_one_rdm_pair_torch,
    make_two_rdm_ab_pair_torch,
    make_two_rdm_pair_torch,
)
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

# The tiled sweep and the serial builders differ only in summation order.
ATOL = 1e-10
# The S^2 application is a single scattered sweep, so it matches the serial
# operator to roundoff rather than to the builders' accumulation tolerance.
S2_ATOL = 1e-12

# Four 2-electron strings over the low orbitals {0, 1, 2, 3}: every 2-subset.
_SMALL_STRINGS = [
    (1 << 0) | (1 << 1),
    (1 << 0) | (1 << 2),
    (1 << 1) | (1 << 3),
    (1 << 2) | (1 << 3),
]


# ============================================================================
# Basis construction
# ============================================================================


def _wide_strings() -> list[int]:
    """Four 2-electron strings straddling the 64-bit limb boundary.

    The occupied orbitals are {0, 1, 64, 65}: two in limb 0 and two in limb 1,
    so every string carries a bit on both sides of the boundary.
    """
    return [
        (1 << 0) | (1 << 1),
        (1 << 0) | (1 << 64),
        (1 << 1) | (1 << 65),
        (1 << 64) | (1 << 65),
    ]


def _product_basis(strings: list[int], n_orb: int) -> PairBasis:
    """The full Cartesian product of one spin's string set with itself."""
    alphas = np.repeat(np.array(strings, dtype=object), len(strings))
    betas = np.tile(np.array(strings, dtype=object), len(strings))
    return pair_basis_from_arrays(alphas, betas, n_orb=n_orb)


def _two_limb(basis: PairBasis) -> PairBasis:
    """The same basis with every string field padded to ``(M, 2)`` limbs."""

    def pad(t: Tensor) -> Tensor:
        return torch.stack([t, torch.zeros_like(t)], dim=-1)

    return replace(
        basis,
        alpha_of_pair=pad(basis.alpha_of_pair),
        beta_of_pair=pad(basis.beta_of_pair),
        unique_alphas=pad(basis.unique_alphas),
        unique_betas=pad(basis.unique_betas),
    )


def _wide_spec(n_orb: int) -> list[tuple[list[int], list[int]]]:
    """Three disjoint pair blocks whose strings straddle the limb boundary."""
    strings = _limb_wide_strings(n_orb)
    return [(strings[:3], strings[:3]), (strings[:3], strings[3:]), (strings[3:], strings[:3])]


def _structural_wide(n_orb: int) -> StructuralPairBasis:
    """The limb structural basis of :func:`_wide_spec`."""
    return structural_pair_basis_from_blocks(_wide_spec(n_orb))


def _explicit_wide(n_orb: int) -> PairBasis:
    """The same limb block spec as an explicit lex-sorted pair list."""
    alphas, betas = [], []
    for alpha, beta in _wide_spec(n_orb):
        alphas.append(np.repeat(np.array(alpha, dtype=object), len(beta)))
        betas.append(np.tile(np.array(beta, dtype=object), len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas), n_orb=n_orb)


def _wide_perm(basis: StructuralPairBasis, explicit: PairBasis) -> Tensor:
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


def _bounds(partition, basis: PairBasis, rank: int) -> PairTileBounds:
    """One rank's tile bounds over the explicit pair list."""
    return pair_tile_bounds(partition, basis, rank)


def _helpers(basis: PairBasis, n_orb: int):
    """Pair helpers with the exact block decomposition forced on."""
    helpers = generate_pair_helpers(basis, n_orb, device="cpu", block_matvec="on")
    assert helpers.blocks is not None
    return helpers


def _tile_operator(basis: PairBasis, n_orb: int, helpers, partition, rank: int):
    """One rank's tile operator; the post-processing never reads integrals."""
    bounds = _bounds(partition, basis, rank)
    operator = dpt.LocalPairTileOperator(
        torch.zeros(bounds.local_size, dtype=torch.float64),
        basis,
        helpers,
        n_orb,
        None,
        bounds,
        partition,
        rank=rank,
        world_size=partition.world_size,
    )
    return operator, bounds


def _random_vector(n: int, seed: int, *, complex_values: bool) -> Tensor:
    """Deterministic random real or complex vector."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if not complex_values:
        return real
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.complex(real, imag)


# ============================================================================
# Direct reference
# ============================================================================


def _reference(basis: PairBasis, helpers, n_orb: int, x: Tensor) -> tuple:
    """Serial 1-RDM, 2-RDM, S^2 scalar/apply/diagonal of the full vector."""
    gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, n_orb)
    gamma2_aa, gamma2_ab, gamma2_bb = make_two_rdm_pair_torch(x, basis, helpers, n_orb)
    ab_for_spin = make_two_rdm_ab_pair_torch(x, basis, helpers, n_orb)
    spin_sq = s_squared_from_rdm(ab_for_spin, helpers.n_a, helpers.n_b)
    operator = PairSpinSqOperator(basis, n_orb, helpers.n_a, helpers.n_b)
    sigma = _inter_to_block_signs_pair(helpers.alpha_of_pair, helpers.beta_of_pair, n_orb).to(
        x.dtype
    )
    s2_apply = sigma * operator.apply(x * sigma)
    return (
        gamma_a,
        gamma_b,
        gamma2_aa,
        gamma2_ab,
        gamma2_bb,
        spin_sq,
        s2_apply,
        operator.diagonal(),
    )


def _assert_same(
    got: Tensor, expected: Tensor, *, exact: bool, label: str, atol: float = ATOL
) -> None:
    """Assert one observable matches its serial reference."""
    assert got.dtype == expected.dtype, (label, got.dtype, expected.dtype)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    if exact:
        assert torch.equal(got, expected), label
        return
    diff = float((got - expected).abs().max()) if got.numel() else 0.0
    assert torch.allclose(got, expected, atol=atol, rtol=0.0), (label, diff)


_OBSERVABLE_NAMES = ("γα", "γβ", "Γαα", "Γαβ", "Γββ", "<S^2>", "S^2 x", "S^2 diag")


def _observables(operator, bounds: PairTileBounds, x: Tensor, n_a: int, n_b: int) -> tuple:
    """Every tiled observable of one rank's tile, in the serial layout."""
    x_local = x[bounds.pair_index]
    gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
    gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
    spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
    s2_apply = pair_tiled_s2_apply(operator, x_local)
    s2_diag = pair_tiled_s2_diagonal(operator)
    return (gamma_a, gamma_b, gamma2_aa, gamma2_ab, gamma2_bb, spin_sq, s2_apply, s2_diag)


def _check(
    operator,
    bounds: PairTileBounds,
    x: Tensor,
    reference: tuple,
    n_a: int,
    n_b: int,
    *,
    exact: bool,
    label: str,
) -> tuple:
    """Assert every tiled observable matches the direct builders; return them."""
    got = _observables(operator, bounds, x, n_a, n_b)
    for name, value, expected in zip(_OBSERVABLE_NAMES, got, reference, strict=True):
        if name in ("S^2 x", "S^2 diag"):
            expected = expected[bounds.pair_index]
        atol = S2_ATOL if name == "S^2 x" else ATOL
        _assert_same(value, expected, exact=exact, label=f"{label} {name}", atol=atol)

    # The comparison must not be vacuous: singles, the αβ class-D rows and the
    # spin-swap rows all contribute.
    n_orb = got[0].shape[0]
    off_diagonal = ~torch.eye(n_orb, dtype=torch.bool)
    assert float(got[0][off_diagonal].abs().max()) > 1e-3, f"{label} γα off-diagonal is zero"
    class_d = off_diagonal[:, :, None, None] & off_diagonal[None, None, :, :]
    assert float(got[3].abs()[class_d].max()) > 1e-3, f"{label} Γαβ class D is zero"
    swapped = got[6] - got[7] * x[bounds.pair_index]
    assert float(swapped.abs().max()) > 1e-3, f"{label} spin-swap rows are zero"
    return got


class _NoReduceSum:
    """Stand-in ``GlobalSum`` that exposes each rank's partial unreduced."""

    def __init__(self, group: object = None) -> None:
        """Ignore the process group."""
        self.group = group

    def all_reduce(self, tensor: Tensor) -> Tensor:
        """Return the rank's partial unchanged."""
        return tensor


# ============================================================================
# 1. A padded two-limb basis matches the single-word basis bit for bit
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_embedded_two_limb_matches_single_word(complex_values: bool) -> None:
    """Padded limbs reproduce the 1-D builders exactly, real and complex."""
    n_orb = 8
    basis_1d = _product_basis(_SMALL_STRINGS, n_orb)
    basis_limb = _two_limb(basis_1d)
    assert basis_limb.n_limbs == 2

    helpers_1d = _helpers(basis_1d, n_orb)
    helpers_limb = _helpers(basis_limb, n_orb)
    assert helpers_1d.nnz() == helpers_limb.nnz()

    x = _random_vector(basis_1d.n_pairs, 7, complex_values=complex_values)
    reference_1d = _reference(basis_1d, helpers_1d, n_orb, x)
    reference_limb = _reference(basis_limb, helpers_limb, n_orb, x)
    for got, want in zip(reference_limb, reference_1d, strict=True):
        _assert_same(got, want, exact=True, label="serial reference")

    partition = partition_tiles(basis_1d.n_unique_alphas, basis_1d.n_unique_betas, 1)
    operator_1d, bounds_1d = _tile_operator(basis_1d, n_orb, helpers_1d, partition, 0)
    operator_limb, bounds_limb = _tile_operator(basis_limb, n_orb, helpers_limb, partition, 0)
    got_1d = _check(
        operator_1d,
        bounds_1d,
        x,
        reference_1d,
        helpers_1d.n_a,
        helpers_1d.n_b,
        exact=False,
        label="1-D",
    )
    got_limb = _check(
        operator_limb,
        bounds_limb,
        x,
        reference_limb,
        helpers_limb.n_a,
        helpers_limb.n_b,
        exact=False,
        label="2-limb",
    )
    for name, limb, single in zip(_OBSERVABLE_NAMES, got_limb, got_1d, strict=True):
        _assert_same(limb, single, exact=True, label=f"{name} 2-limb vs 1-D")


# ============================================================================
# 2. A 70-orbital basis matches the direct explicit-pair builders
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_wide_basis_matches_direct_builders(complex_values: bool) -> None:
    """Every tiled observable at n_orb=70 matches the direct pair builders."""
    n_orb = 70
    basis = _product_basis(_wide_strings(), n_orb)
    assert basis.n_limbs == 2 and basis.n_pairs == 16

    helpers = _helpers(basis, n_orb)
    x = _random_vector(basis.n_pairs, 11, complex_values=complex_values)
    reference = _reference(basis, helpers, n_orb, x)

    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1)
    operator, bounds = _tile_operator(basis, n_orb, helpers, partition, 0)
    _check(
        operator,
        bounds,
        x,
        reference,
        helpers.n_a,
        helpers.n_b,
        exact=False,
        label=f"70-orbital {complex_values}",
    )


# ============================================================================
# 3. A 2x2 simulated tile grid: staged partials sum to the serial observables
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_wide_basis_staged_partials(complex_values: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every rank's staged partial sums to the direct observable at n_orb=70."""
    n_orb = 70
    basis = _product_basis(_wide_strings(), n_orb)
    helpers = _helpers(basis, n_orb)
    x = _random_vector(basis.n_pairs, 13, complex_values=complex_values)
    (
        gamma_a_ref,
        gamma_b_ref,
        _,
        _,
        _,
        spin_sq_ref,
        s2_apply_ref,
        s2_diag_ref,
    ) = _reference(basis, helpers, n_orb, x)

    grid = (2, 2)
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    monkeypatch.setattr(ptp, "GlobalSum", _NoReduceSum)
    _FullVectorExchange.full_vector = x

    n_a, n_b = helpers.n_a, helpers.n_b
    s_z = (n_a - n_b) / 2.0
    constant = s_z * s_z - s_z + n_a
    gamma_a_total = gamma_b_total = None
    apply_total = torch.zeros_like(x)
    diag_total = torch.zeros(basis.n_pairs, dtype=s2_diag_ref.dtype, device=s2_diag_ref.device)
    spin_sq_cross_total = None
    for rank in range(world_size):
        operator, bounds = _tile_operator(basis, n_orb, helpers, partition, rank)
        x_local = x[bounds.pair_index]

        gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
        gamma_a_total = gamma_a if gamma_a_total is None else gamma_a_total + gamma_a
        gamma_b_total = gamma_b if gamma_b_total is None else gamma_b_total + gamma_b

        spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
        partial_cross = constant - spin_sq
        spin_sq_cross_total = (
            partial_cross if spin_sq_cross_total is None else spin_sq_cross_total + partial_cross
        )

        apply_total[bounds.pair_index] = pair_tiled_s2_apply(operator, x_local)
        diag_total[bounds.pair_index] = pair_tiled_s2_diagonal(operator)

    _assert_same(gamma_a_total, gamma_a_ref, exact=False, label="staged γα")
    _assert_same(gamma_b_total, gamma_b_ref, exact=False, label="staged γβ")
    _assert_same(apply_total, s2_apply_ref, exact=False, label="staged S^2 x", atol=S2_ATOL)
    _assert_same(diag_total, s2_diag_ref, exact=False, label="staged S^2 diag")
    spin_sq_total = constant - spin_sq_cross_total
    assert float(abs(spin_sq_total - spin_sq_ref)) <= ATOL


# ============================================================================
# 4. Partition helpers size by string, not by limb word
# ============================================================================


@pytest.mark.parametrize("world_size", [2, 4])
def test_partition_helpers_count_strings_not_limb_words(world_size: int) -> None:
    """A padded two-limb basis partitions exactly like its single-word form.

    ``pair_work_weights`` and ``pair_tile_bounds`` used ``numel()`` on the
    per-pair and unique string tensors, which counts limb words: the weight
    vector came out twice as long and the bounds rejected the partition.
    Both now size by string, so the padded basis partitions identically.
    """
    n_orb = 8
    basis_1d = _product_basis(_SMALL_STRINGS, n_orb)
    basis_limb = _two_limb(basis_1d)
    helpers_1d = _helpers(basis_1d, n_orb)
    helpers_limb = _helpers(basis_limb, n_orb)

    weights_1d = pair_work_weights(helpers_1d)
    weights_limb = pair_work_weights(helpers_limb)
    assert weights_limb.shape == (basis_1d.n_pairs,)
    assert torch.equal(weights_limb, weights_1d)

    partition_1d = pair_tile_partition(helpers_1d, world_size)
    partition_limb = pair_tile_partition(helpers_limb, world_size)
    assert (partition_limb.n_rows, partition_limb.n_cols) == (
        partition_1d.n_rows,
        partition_1d.n_cols,
    )

    covered = torch.zeros(basis_limb.n_pairs, dtype=torch.int64)
    for rank in range(world_size):
        bounds = pair_tile_bounds(partition_limb, basis_limb, rank)
        assert bounds.n_alpha_uids == basis_limb.n_unique_alphas
        assert bounds.n_beta_uids == basis_limb.n_unique_betas
        covered[bounds.pair_index] += 1
    # Every pair is in exactly one tile.
    assert torch.equal(covered, torch.ones_like(covered))


def test_wide_basis_partition_helpers_cover_the_pair_list() -> None:
    """At n_orb=70 the partition helpers accept the limb basis and cover it."""
    n_orb = 70
    basis = _product_basis(_wide_strings(), n_orb)
    assert basis.n_limbs == 2
    helpers = _helpers(basis, n_orb)
    weights = pair_work_weights(helpers)
    assert weights.shape == (basis.n_pairs,)
    assert bool(torch.isfinite(weights).all())

    partition = pair_tile_partition(helpers, 2, grid=(2, 1))
    covered = torch.zeros(basis.n_pairs, dtype=torch.int64)
    for rank in range(2):
        bounds = pair_tile_bounds(partition, basis, rank)
        covered[bounds.pair_index] += 1
    assert torch.equal(covered, torch.ones_like(covered))


# ============================================================================
# 5. A structural (block-described) limb basis matches explicit-limb builders
# ============================================================================


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_structural_wide_basis_matches_explicit_limb_builders(
    n_orb: int, complex_values: bool
) -> None:
    """Tiled structural limb observables equal the explicit-limb builders."""
    basis = _structural_wide(n_orb)
    explicit = _explicit_wide(n_orb)
    assert basis.n_limbs == explicit.n_limbs == (2 if n_orb == 70 else 3)
    perm = _wide_perm(basis, explicit)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)

    helpers = _helpers(basis, n_orb)
    helpers_explicit = _helpers(explicit, n_orb)
    x = _random_vector(basis.n_pairs, 31, complex_values=complex_values)
    x = x / torch.linalg.norm(x)
    x_explicit = x[inverse]

    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1)
    operator, bounds = _tile_operator(basis, n_orb, helpers, partition, 0)
    x_local = x[bounds.pair_index]

    gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
    expected_a, expected_b = make_one_rdm_pair_torch(x_explicit, explicit, helpers_explicit, n_orb)
    _assert_same(gamma_a, expected_a, exact=False, label=f"{n_orb} γα")
    _assert_same(gamma_b, expected_b, exact=False, label=f"{n_orb} γβ")

    operator_explicit = PairSpinSqOperator(explicit, n_orb, helpers.n_a, helpers.n_b)
    sigma = _inter_to_block_signs_pair(explicit.alpha_of_pair, explicit.beta_of_pair, n_orb).to(
        x.dtype
    )
    expected_apply = sigma * operator_explicit.apply(x_explicit * sigma)
    spin_sq = pair_tiled_spin_squared(operator, x_local, helpers.n_a, helpers.n_b)
    blocked = x_explicit * sigma
    assert (
        abs(float(spin_sq) - float(torch.vdot(blocked, operator_explicit.apply(blocked)).real))
        < ATOL
    )
    _assert_same(
        pair_tiled_s2_apply(operator, x_local),
        expected_apply[perm],
        exact=False,
        label=f"{n_orb} S^2 x",
        atol=S2_ATOL,
    )
    _assert_same(
        pair_tiled_s2_diagonal(operator),
        operator_explicit.diagonal()[perm],
        exact=False,
        label=f"{n_orb} S^2 diag",
    )

    if n_orb == 70:
        gamma2 = pair_tiled_two_rdm(operator, x_local)
        expected2 = make_two_rdm_pair_torch(x_explicit, explicit, helpers_explicit, n_orb)
        for got, want, label in zip(gamma2, expected2, ("Γαα", "Γαβ", "Γββ"), strict=True):
            _assert_same(got, want, exact=False, label=f"70 {label}")


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_structural_wide_staged_partials(
    complex_values: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2x2 grid's structural limb partials sum to the explicit observables."""
    n_orb = 70
    basis = _structural_wide(n_orb)
    explicit = _explicit_wide(n_orb)
    perm = _wide_perm(basis, explicit)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)

    helpers = _helpers(basis, n_orb)
    helpers_explicit = _helpers(explicit, n_orb)
    x = _random_vector(basis.n_pairs, 37, complex_values=complex_values)
    x = x / torch.linalg.norm(x)
    x_explicit = x[inverse]
    expected_a, expected_b = make_one_rdm_pair_torch(x_explicit, explicit, helpers_explicit, n_orb)
    operator_explicit = PairSpinSqOperator(explicit, n_orb, helpers.n_a, helpers.n_b)
    sigma = _inter_to_block_signs_pair(explicit.alpha_of_pair, explicit.beta_of_pair, n_orb).to(
        x.dtype
    )
    expected_apply = sigma * operator_explicit.apply(x_explicit * sigma)
    expected_diag = operator_explicit.diagonal()
    blocked = x_explicit * sigma
    expected_spin_sq = float(torch.vdot(blocked, operator_explicit.apply(blocked)).real)

    grid = (2, 2)
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    monkeypatch.setattr(ptp, "GlobalSum", _NoReduceSum)
    _FullVectorExchange.full_vector = x

    gamma_a_total = torch.zeros_like(expected_a)
    gamma_b_total = torch.zeros_like(expected_b)
    apply_total = torch.zeros_like(x)
    diag_total = torch.zeros_like(expected_diag)
    cross_total = None
    n_a, n_b = helpers.n_a, helpers.n_b
    s_z = (n_a - n_b) / 2.0
    constant = s_z * s_z - s_z + n_a
    for rank in range(world_size):
        operator, bounds = _tile_operator(basis, n_orb, helpers, partition, rank)
        x_local = x[bounds.pair_index]
        gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
        gamma_a_total += gamma_a
        gamma_b_total += gamma_b
        spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
        partial_cross = constant - spin_sq
        cross_total = partial_cross if cross_total is None else cross_total + partial_cross
        apply_total[bounds.pair_index] = pair_tiled_s2_apply(operator, x_local)
        diag_total[bounds.pair_index] = pair_tiled_s2_diagonal(operator)

    _assert_same(gamma_a_total, expected_a, exact=False, label="staged γα")
    _assert_same(gamma_b_total, expected_b, exact=False, label="staged γβ")
    _assert_same(apply_total, expected_apply[perm], exact=False, label="staged S^2 x", atol=S2_ATOL)
    _assert_same(diag_total, expected_diag[perm], exact=False, label="staged S^2 diag")
    assert abs(float(constant - cross_total) - expected_spin_sq) < ATOL
