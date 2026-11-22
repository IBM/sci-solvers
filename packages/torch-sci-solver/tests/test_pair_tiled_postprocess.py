"""Tests for the tiled GDB (explicit pair) post-processing.

Every entry point is compared against the serial pair builders on the gathered
vector at ``atol = 1e-10`` — real and complex amplitudes, over 1x1, 2x2 and 2x3
tile grids (the 2x3 case splits the 19 unique strings unevenly), plus a
degenerate 8x2 grid whose tiles are partly empty. A recording exchange asserts
that the tiled path reads only the owned block and stage-sized buffers, never
the full vector, and a four-rank gloo run exercises the real staged P2P
exchange.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch_sci_solver.torch_impl.distributed_pairs_tile as dpt
import torch_sci_solver.torch_impl.pair_tiled_postprocess as ptp
from test_distributed_pairs import _init_gloo, _run_spawned  # noqa: E402
from test_distributed_pairs_tile import (  # noqa: E402
    _basis,
    _FullVectorExchange,
    _helpers,
    _random_vector,
)
from test_mult_pair_blocks import _complex_integrals, _real_integrals  # noqa: E402
from torch import Tensor
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch
from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
    pair_tiled_one_rdm,
    pair_tiled_s2_apply,
    pair_tiled_spin_squared,
    pair_tiled_two_rdm,
)
from torch_sci_solver.torch_impl.pair_tiles import pair_tile_bounds
from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm
from torch_sci_solver.torch_impl.rdm_pairs import (
    _inter_to_block_signs_pair,
    make_one_rdm_pair_torch,
    make_two_rdm_ab_pair_torch,
    make_two_rdm_pair_torch,
)
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

# A 2x2 rank grid and an uneven 2x3 grid (6 ranks); the degenerate grid has
# more α slabs than unique α strings, so some tiles own no pair.
_GRID_2X2 = (2, 2)
_GRID_2X3 = (2, 3)
_GRID_EMPTY = (8, 2)
_EMPTY_RANKS = 16
# The tiled sweep and the serial builders differ only in summation order.
ATOL = 1e-10
# The S^2 application is a single scattered sweep, so it matches the serial
# operator to roundoff rather than to the builders' accumulation tolerance.
S2_ATOL = 1e-12


def _setup(name: str, complex_values: bool, seed: int = 11) -> tuple:
    """One named basis, its helpers and a deterministic full vector."""
    basis, norb = _basis(name)
    helpers = _helpers(basis, norb)
    x = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
    return basis, norb, helpers, x


def _reference(basis, helpers, norb: int, x: Tensor) -> tuple:
    """Serial 1-RDM, 2-RDM, <S^2> and S^2-apply of the full vector."""
    gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, norb)
    gamma2_aa, gamma2_ab, gamma2_bb = make_two_rdm_pair_torch(x, basis, helpers, norb)
    ab_for_spin = make_two_rdm_ab_pair_torch(x, basis, helpers, norb)
    spin_sq = s_squared_from_rdm(ab_for_spin, helpers.n_a, helpers.n_b)
    s2_apply = _s2_apply_reference(x, basis, helpers, norb)
    return gamma_a, gamma_b, gamma2_aa, gamma2_ab, gamma2_bb, spin_sq, s2_apply


def _s2_apply_reference(x: Tensor, basis, helpers, norb: int) -> Tensor:
    """Full ``S^2 x`` in interleaved-Wick convention from the serial operator.

    :class:`PairSpinSqOperator` is built in spin-block convention, so the input
    and the result are converted with the interleaved-to-block signs.
    """
    sigma = _inter_to_block_signs_pair(helpers.alpha_of_pair, helpers.beta_of_pair, norb).to(
        x.dtype
    )
    serial = PairSpinSqOperator(basis, norb, helpers.n_a, helpers.n_b)
    return sigma * serial.apply(x * sigma)


def _tile_operator(basis, norb: int, helpers, integrals, hii: Tensor, partition, rank: int):
    """One rank's tile operator, with whatever exchange is installed."""
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
    )
    return operator, bounds


def _assert_close(got: Tensor, expected: Tensor, label: str) -> None:
    """Assert one observable matches its serial reference to ``ATOL``."""
    assert got.dtype == expected.dtype, (label, got.dtype, expected.dtype)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    diff = float((got - expected).abs().max()) if got.numel() else 0.0
    assert torch.allclose(got, expected, atol=ATOL, rtol=0.0), (label, diff)


def _assert_populated(gamma_a: Tensor, gamma_b: Tensor, gamma_ab: Tensor, norb: int) -> None:
    """The comparison must not be vacuous: every term class is populated."""
    off_a = gamma_a - torch.diag(gamma_a.diagonal())
    off_b = gamma_b - torch.diag(gamma_b.diagonal())
    assert float(off_a.abs().max()) > 1e-3, "γα off-diagonal block is zero"
    assert float(off_b.abs().max()) > 1e-3, "γβ off-diagonal block is zero"

    off = ~torch.eye(norb, dtype=torch.bool)
    diag = ~off

    def category(alpha_diag: bool, beta_diag: bool) -> float:
        """Largest magnitude in one (α-side, β-side) diagonal category."""
        first = diag if alpha_diag else off
        second = diag if beta_diag else off
        mask = first[:, :, None, None] & second[None, None, :, :]
        return float(gamma_ab.abs()[mask].max())

    assert category(True, True) > 1e-3, "Γαβ class A is zero"
    assert category(True, False) > 1e-3, "Γαβ class B is zero"
    assert category(False, True) > 1e-3, "Γαβ class C is zero"
    assert category(False, False) > 1e-3, "Γαβ class D is zero"


def _check(operator, bounds, x: Tensor, reference: tuple, helpers, label: str) -> None:
    """Assert every tiled observable matches the serial builders on the tile."""
    (
        gamma_a_ref,
        gamma_b_ref,
        gamma2_aa_ref,
        gamma2_ab_ref,
        gamma2_bb_ref,
        spin_sq_ref,
        s2_apply_ref,
    ) = reference
    norb = gamma_a_ref.shape[0]
    n_a, n_b = helpers.n_a, helpers.n_b
    x_local = x[bounds.pair_index]

    gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
    _assert_close(gamma_a, gamma_a_ref, f"{label} γα")
    _assert_close(gamma_b, gamma_b_ref, f"{label} γβ")

    spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
    assert float(abs(spin_sq - spin_sq_ref)) <= ATOL, (label, float(spin_sq), float(spin_sq_ref))

    s2_apply = pair_tiled_s2_apply(operator, x_local)
    expected_apply = s2_apply_ref[bounds.pair_index]
    diff_apply = float((s2_apply - expected_apply).abs().max()) if s2_apply.numel() else 0.0
    assert s2_apply.dtype == x.dtype and s2_apply.shape == expected_apply.shape, label
    assert diff_apply <= S2_ATOL, (label, diff_apply)

    gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
    _assert_close(gamma2_aa, gamma2_aa_ref, f"{label} Γαα")
    _assert_close(gamma2_ab, gamma2_ab_ref, f"{label} Γαβ")
    _assert_close(gamma2_bb, gamma2_bb_ref, f"{label} Γββ")
    _assert_populated(gamma_a, gamma_b, gamma2_ab, norb)
    _assert_matched_class_d(spin_sq, x, helpers, n_a, n_b, label)

    # Caller-provided accumulation buffers receive the reduced observables.
    buffer_a = torch.zeros((norb, norb), dtype=x.dtype)
    buffer_b = torch.zeros((norb, norb), dtype=x.dtype)
    got_a, got_b = pair_tiled_one_rdm(operator, x_local, gamma_a=buffer_a, gamma_b=buffer_b)
    assert got_a is buffer_a and got_b is buffer_b
    _assert_close(got_a, gamma_a, f"{label} γα buffer")
    _assert_close(got_b, gamma_b, f"{label} γβ buffer")

    shape2 = (norb, norb, norb, norb)
    buf_aa = torch.zeros(shape2, dtype=x.dtype)
    buf_ab = torch.zeros(shape2, dtype=x.dtype)
    buf_bb = torch.zeros(shape2, dtype=x.dtype)
    got_aa, got_ab, got_bb = pair_tiled_two_rdm(
        operator, x_local, gamma2_a=buf_aa, gamma2_b=buf_bb, gamma2_ab=buf_ab
    )
    assert got_aa is buf_aa and got_ab is buf_ab and got_bb is buf_bb
    _assert_close(got_aa, gamma2_aa, f"{label} Γαα buffer")
    _assert_close(got_ab, gamma2_ab, f"{label} Γαβ buffer")
    _assert_close(got_bb, gamma2_bb, f"{label} Γββ buffer")


class _NoReduceSum:
    """Stand-in ``GlobalSum`` that exposes each rank's partial unreduced.

    A single-process tile grid has no process group, so the real reduction is a
    no-op there. The stand-in tests run every simulated rank and sum the
    partials themselves, which is exactly what the collective would deliver.
    """

    def __init__(self, group: object = None) -> None:
        """Ignore the process group."""
        self.group = group

    def all_reduce(self, tensor: Tensor) -> Tensor:
        """Return the rank's partial unchanged."""
        return tensor


def _add(total: Tensor | None, partial: Tensor) -> Tensor:
    """Add one rank's partial into a running sum."""
    return partial.clone() if total is None else total + partial


def _check_partials(operators, bounds, x: Tensor, reference: tuple, helpers, label: str) -> None:
    """Assert the sum of the per-rank partials matches the serial builders."""
    (
        gamma_a_ref,
        gamma_b_ref,
        gamma2_aa_ref,
        gamma2_ab_ref,
        gamma2_bb_ref,
        spin_sq_ref,
        _,
    ) = reference
    n_a, n_b = helpers.n_a, helpers.n_b
    # The collective reduces the Γαβ[p,q,q,p] cross term; the rank-local
    # S_z² − S_z + N_α constant is added after it, so the partial sum undoes
    # the constant once per rank.
    s_z = (n_a - n_b) / 2.0
    constant = s_z * s_z - s_z + n_a
    gamma_a_total = gamma_b_total = None
    gamma2_aa_total = gamma2_ab_total = gamma2_bb_total = None
    spin_sq_cross_total = None
    for operator, bound in zip(operators, bounds, strict=True):
        x_local = x[bound.pair_index]
        gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
        spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
        gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
        gamma_a_total = _add(gamma_a_total, gamma_a)
        gamma_b_total = _add(gamma_b_total, gamma_b)
        gamma2_aa_total = _add(gamma2_aa_total, gamma2_aa)
        gamma2_ab_total = _add(gamma2_ab_total, gamma2_ab)
        gamma2_bb_total = _add(gamma2_bb_total, gamma2_bb)
        spin_sq_cross_total = _add(spin_sq_cross_total, constant - spin_sq)
    spin_sq_total = constant - spin_sq_cross_total

    _assert_close(gamma_a_total, gamma_a_ref, f"{label} γα")
    _assert_close(gamma_b_total, gamma_b_ref, f"{label} γβ")
    _assert_close(gamma2_aa_total, gamma2_aa_ref, f"{label} Γαα")
    _assert_close(gamma2_ab_total, gamma2_ab_ref, f"{label} Γαβ")
    _assert_close(gamma2_bb_total, gamma2_bb_ref, f"{label} Γββ")
    assert float(abs(spin_sq_total - spin_sq_ref)) <= ATOL, (
        label,
        float(spin_sq_total),
        float(spin_sq_ref),
    )
    _assert_populated(gamma_a_total, gamma_b_total, gamma2_ab_total, gamma_a_ref.shape[0])
    _assert_matched_class_d(spin_sq_total, x, helpers, n_a, n_b, label)


def _assert_matched_class_d(spin_sq: Tensor, x: Tensor, helpers, n_a: int, n_b: int, label: str):
    """The matched class-D rows must contribute to <S^2>."""
    norb = int(helpers.n_orb)
    orbital = torch.arange(norb)
    real_dtype = spin_sq.dtype.to_real() if spin_sq.is_complex() else spin_sq.dtype
    alpha_occ = ((helpers.alpha_of_pair.unsqueeze(1) >> orbital) & 1).to(real_dtype)
    beta_occ = ((helpers.beta_of_pair.unsqueeze(1) >> orbital) & 1).to(real_dtype)
    weights = x.abs().pow(2)
    class_a = ((alpha_occ * weights.unsqueeze(1)).T @ beta_occ).diagonal().sum()
    s_z = (n_a - n_b) / 2.0
    class_a_only = s_z * s_z - s_z + n_a - class_a
    assert float(abs(spin_sq - class_a_only)) > 1e-6, f"{label} matched class-D term is zero"


# ============================================================================
# 1. Single process, stand-in exchange
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True])
def test_pair_tiled_postprocess_single_rank(complex_values: bool, monkeypatch) -> None:
    """A 1x1 tile reproduces all serial builders without a process group."""
    basis, norb, helpers, x = _setup("full_product", complex_values)
    reference = _reference(basis, helpers, norb, x)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    operator, bounds = _tile_operator(basis, norb, helpers, integrals, hii, partition, 0)
    _check(operator, bounds, x, reference, helpers, f"1x1 {complex_values}")


@pytest.mark.parametrize(
    "basis_name,grid",
    [("full_product", _GRID_2X2), ("tier", _GRID_2X3)],
    ids=["2x2", "2x3"],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_pair_tiled_postprocess_grid_matches_serial(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank's partial sums to the serial observables."""
    basis, norb, helpers, x = _setup(basis_name, complex_values)
    reference = _reference(basis, helpers, norb, x)
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    if grid == _GRID_2X3:
        lengths = {stop - start for start, stop in partition.grid.alpha_bounds}
        assert len(lengths) > 1, "the 2x3 fixture must split the α axis unevenly"
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    monkeypatch.setattr(ptp, "GlobalSum", _NoReduceSum)
    _FullVectorExchange.full_vector = x
    operators, bounds = [], []
    for rank in range(world_size):
        operator, bound = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        operators.append(operator)
        bounds.append(bound)
    _check_partials(operators, bounds, x, reference, helpers, f"{basis_name} {grid}")


@pytest.mark.parametrize(
    "basis_name,grid",
    [("full_product", (1, 1)), ("full_product", _GRID_2X2), ("tier", _GRID_2X3)],
    ids=["1x1", "2x2", "2x3"],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_pair_tiled_s2_apply_matches_serial(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank's ``S^2 x`` block matches the serial pair operator."""
    basis, norb, helpers, x = _setup(basis_name, complex_values)
    reference = _s2_apply_reference(x, basis, helpers, norb)
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    if grid == _GRID_2X3:
        lengths = {stop - start for start, stop in partition.grid.alpha_bounds}
        assert len(lengths) > 1, "the 2x3 fixture must split the α axis unevenly"
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    for rank in range(world_size):
        operator, bounds = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        got = pair_tiled_s2_apply(operator, x[bounds.pair_index])
        expected = reference[bounds.pair_index]
        assert got.dtype == x.dtype and got.shape == expected.shape
        diff = float((got - expected).abs().max()) if got.numel() else 0.0
        assert diff <= S2_ATOL, (basis_name, grid, complex_values, rank, diff)


def test_pair_tiled_s2_apply_is_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Complex linearity: ``apply(ax + by) == a apply(x) + b apply(y)``."""
    basis, norb, helpers, x = _setup("full_product", complex_values=True)
    y = _random_vector(basis.n_pairs, 29, complex_values=True)
    integrals = _complex_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    operator, _ = _tile_operator(basis, norb, helpers, integrals, hii, partition, 0)
    a = 0.75 - 0.5j
    b = -1.25 + 0.25j
    combined = pair_tiled_s2_apply(operator, a * x + b * y)
    expected = a * pair_tiled_s2_apply(operator, x) + b * pair_tiled_s2_apply(operator, y)
    diff = float((combined - expected).abs().max())
    assert diff <= S2_ATOL, diff


@pytest.mark.parametrize(
    "basis_name,grid",
    [("full_product", (1, 1)), ("full_product", _GRID_2X2), ("tier", _GRID_2X3)],
    ids=["1x1", "2x2", "2x3"],
)
@pytest.mark.parametrize("complex_values", [False, True])
def test_pair_tiled_s2_apply_rayleigh_matches_spin_squared(
    basis_name: str,
    grid: tuple[int, int],
    complex_values: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``⟨x|S^2|x⟩`` from the apply equals the reduced tiled ``<S^2>``.

    The RDM-based ``<S^2>`` uses the unit-norm convention, so the fixture is
    normalized. The per-rank partials are summed the same way the test harness
    reduces ``pair_tiled_spin_squared``: its post-reduction constant is added
    once, not once per rank.
    """
    basis, norb, helpers, x = _setup(basis_name, complex_values)
    x = x / x.norm()
    world_size = grid[0] * grid[1]
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    monkeypatch.setattr(ptp, "GlobalSum", _NoReduceSum)
    _FullVectorExchange.full_vector = x
    n_a, n_b = helpers.n_a, helpers.n_b
    s_z = (n_a - n_b) / 2.0
    constant = s_z * s_z - s_z + n_a
    inner_total = cross_total = None
    for rank in range(world_size):
        operator, bounds = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        x_local = x[bounds.pair_index]
        inner_total = _add(inner_total, torch.vdot(x_local, pair_tiled_s2_apply(operator, x_local)))
        spin_sq = pair_tiled_spin_squared(operator, x_local, n_a, n_b)
        cross_total = _add(cross_total, constant - spin_sq)
    spin_sq_total = constant - cross_total
    value = float(inner_total.real)
    assert abs(value - float(spin_sq_total)) <= S2_ATOL, (basis_name, grid, complex_values)
    if inner_total.is_complex():
        assert abs(float(inner_total.imag)) <= S2_ATOL


def test_pair_tiled_postprocess_empty_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiles with no pair contribute zeros to the reduction."""
    basis, norb, helpers, x = _setup("full_product", False)
    reference = _reference(basis, helpers, norb, x)
    partition = partition_tiles(
        basis.n_unique_alphas, basis.n_unique_betas, _EMPTY_RANKS, grid=_GRID_EMPTY
    )
    integrals = _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    monkeypatch.setattr(ptp, "GlobalSum", _NoReduceSum)
    _FullVectorExchange.full_vector = x
    operators, bounds = [], []
    empty_ranks = 0
    for rank in range(_EMPTY_RANKS):
        operator, bound = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        operators.append(operator)
        bounds.append(bound)
        empty_ranks += int(bound.is_empty)
    _check_partials(operators, bounds, x, reference, helpers, "empty tiles")
    assert empty_ranks > 0, "the fixture must contain an empty tile"


def test_pair_tiled_postprocess_rejects_bad_block() -> None:
    """A wrong block shape or a non-numeric dtype is rejected before any sweep."""
    basis, norb, helpers, _ = _setup("full_product", False)
    integrals = _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    bounds = pair_tile_bounds(partition, basis, 0)
    operator = dpt.LocalPairTileOperator(
        hii, basis, helpers, norb, integrals, bounds, partition, rank=0, world_size=1
    )
    with pytest.raises(ValueError, match="x_local shape"):
        pair_tiled_one_rdm(operator, torch.zeros(bounds.local_size + 1, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating or complex"):
        pair_tiled_spin_squared(
            operator, torch.zeros(bounds.local_size, dtype=torch.int64), helpers.n_a, helpers.n_b
        )
    with pytest.raises(ValueError, match="x_local shape"):
        pair_tiled_two_rdm(operator, torch.zeros(bounds.local_size + 1, dtype=torch.float64))
    with pytest.raises(ValueError, match="x_local shape"):
        pair_tiled_s2_apply(operator, torch.zeros(bounds.local_size + 1, dtype=torch.float64))
    with pytest.raises(TypeError, match="floating or complex"):
        pair_tiled_s2_apply(operator, torch.zeros(bounds.local_size, dtype=torch.int64))
    with pytest.raises(NotImplementedError, match="LocalPairTileOperator"):
        pair_tiled_s2_apply(object(), torch.zeros(bounds.local_size, dtype=torch.float64))


# ============================================================================
# 2. The tiled path never holds the full vector
# ============================================================================


class _RecordingExchange(_FullVectorExchange):
    """Stand-in exchange that records every buffer it hands to the sweep."""

    sizes: list[int] = []

    def stages(self, x_local: Tensor):
        """Record the size of each yielded buffer."""
        for stage, buffer in super().stages(x_local):
            _RecordingExchange.sizes.append(int(buffer.numel()))
            yield stage, buffer


def test_pair_tiled_postprocess_reads_only_tile_and_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every buffer read is one rank's tile; no buffer is ever the full vector."""
    basis, norb, helpers, x = _setup("tier", complex_values=False)
    world_size = _GRID_2X3[0] * _GRID_2X3[1]
    partition = partition_tiles(
        basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=_GRID_2X3
    )
    integrals = _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    monkeypatch.setattr(dpt, "StagedPairExchange", _RecordingExchange)
    _FullVectorExchange.full_vector = x
    tile_sizes = {pair_tile_bounds(partition, basis, rank).local_size for rank in range(world_size)}
    for rank in range(world_size):
        operator, bounds = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        x_local = x[bounds.pair_index]
        for observable in (
            lambda: pair_tiled_one_rdm(operator, x_local),
            lambda: pair_tiled_spin_squared(operator, x_local, helpers.n_a, helpers.n_b),
            lambda: pair_tiled_two_rdm(operator, x_local),
            lambda: pair_tiled_s2_apply(operator, x_local),
        ):
            _RecordingExchange.sizes = []
            observable()
            sizes = list(_RecordingExchange.sizes)
            # One buffer per stage, the first being the owned block itself.
            # A remote stage carries at most one peer tile: the whole tile or,
            # on a sparse layout, the needed-set subset of it.
            assert len(sizes) == world_size
            assert sizes[0] == bounds.local_size
            assert all(0 <= size <= max(tile_sizes) for size in sizes)
            assert max(sizes) < basis.n_pairs


# ============================================================================
# 3. gloo, four ranks, 2x2, real staged P2P exchange
# ============================================================================


def _postprocess_worker(
    rank: int,
    world_size: int,
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's tiled observables match the serial builders."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis, norb, helpers, x = _setup("tier", complex_values, seed=7)
        reference = _reference(basis, helpers, norb, x)
        integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
        hii = make_diag_terms_pair_torch(basis, norb, integrals)
        partition = partition_tiles(
            basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=_GRID_2X2
        )
        operator, bounds = _tile_operator(basis, norb, helpers, integrals, hii, partition, rank)
        _check(operator, bounds, x, reference, helpers, f"gloo rank={rank}/{world_size}")
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_values", [False, True])
def test_pair_tiled_postprocess_gloo_two_by_two(complex_values: bool, tmp_path: Path) -> None:
    """gloo, 4 ranks, 2x2: the real staged exchange matches the serial builders."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _postprocess_worker,
        4,
        (complex_values,),
        tmp_path,
        f"pair_tpost_{'complex' if complex_values else 'real'}",
    )
