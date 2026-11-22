"""Tests for the 2-D tile spinless distributed operator.

Covers:

1. ``world_size=1``: ``LocalSpinlessTileOperator`` matches
   ``mult_spinless_torch`` on real and complex vectors, and the operator never
   allocates a length-``N`` value vector (allocation guard).
2. gloo, 2 / 4 / 8 ranks on the matching tile grids, real and complex: every
   rank's tile output equals the corresponding slice of the full-space
   ``mult_spinless_torch``, the 1-D ``LocalSpinlessOperator`` still matches,
   each staged receive buffer is smaller than ``N``, and the operator reports
   its peak staged bytes.
3. Empty tiles and a tile with an empty needed set (a full-occupation basis
   has no excitations) at 2 and 4 ranks.
4. The uid-axis helpers used for ket-owner routing agree with
   ``half_string_uids``, including multi-limb and unaligned splits.
5. End-to-end solver wiring: ``tile="force"`` and an explicit ``(2, 1)`` grid
   match the 1-D owner-computes solve (energy and occupancies at 1e-8) at 2
   and 4 gloo ranks, ``tile="auto"`` resolves to 1-D on a CPU run, and
   ``compute_full_rdm`` returns both RDMs from the owner-computes solve.
6. CUDA (``@pytest.mark.gpu``): ``triton_matvec=True`` runs the tile
   operator's doubles through the entry-parallel Triton kernel; at two NCCL
   ranks each rank's partial matches the batched operator's at 1e-12 (real
   and complex), and summing the staged partials reproduces the serial
   matvec. On CPU the same flag falls back to the batched scatter bit-exactly.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl import distributed_spinless_tile
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles, partition_vector
from torch_sci_solver.torch_impl.distributed_spinless import LocalSpinlessOperator
from torch_sci_solver.torch_impl.distributed_spinless_tile import LocalSpinlessTileOperator
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver
from torch_sci_solver.torch_impl.spinless_helpers import (
    SpinlessBasis,
    generate_spinless_helpers,
    spinless_basis_from_array,
)
from torch_sci_solver.torch_impl.spinless_tiles import (
    half_string_axes,
    half_string_uids,
    half_string_uids_of,
    resolve_auto_spinless_tiling,
    spinless_tile_partition,
)

NMO = 12
N_ELEC = 6
SPLIT = 6
# One grid per world size; the 64 x 64 uid grid of the full C(12, 6) basis has
# enough slabs for all of them.
GRIDS = {2: (1, 2), 4: (2, 2), 8: (2, 4)}


def _real_spinorbital_integrals(nmo: int, seed: int):
    """Real symmetric hcore + an eri with the chemist 8-fold symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((nmo, nmo))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((nmo,) * 4)
    eri = np.zeros_like(raw)
    for perm in [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ]:
        eri += raw.transpose(perm)
    eri /= 8.0
    return h1, eri


def _complex_spinorbital_integrals(nmo: int, seed: int):
    """Complex hcore + a Hermitian complex chemist-order eri, flat spin-orbital."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(nmo + 2):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        herm = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", herm, herm.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _fixture(complex_case: bool, device: torch.device | str = "cpu"):
    """The full C(12, 6) basis, its integrals, diagonal and a random vector."""
    strs = [sum(1 << bit for bit in combo) for combo in itertools.combinations(range(NMO), N_ELEC)]
    basis = spinless_basis_from_array(strs, NMO, N_ELEC, device=device)
    helpers = generate_spinless_helpers(basis)
    if complex_case:
        h1, eri = _complex_spinorbital_integrals(NMO, seed=101)
        integrals = to_torch_integrals_spinless(h1, eri, device=device)
    else:
        h1, eri = _real_spinorbital_integrals(NMO, seed=101)
        integrals = to_torch_integrals_spinless(h1, eri, device=device, dtype=torch.float64)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    gen = torch.Generator().manual_seed(7)
    if complex_case:
        x = torch.complex(
            torch.randn(basis.dim, generator=gen, dtype=torch.float64),
            torch.randn(basis.dim, generator=gen, dtype=torch.float64),
        )
    else:
        x = torch.randn(basis.dim, generator=gen, dtype=torch.float64)
    return basis, helpers, integrals, hii, x.to(device)


def _record_length_n_allocations(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record every ``torch.zeros``/``torch.empty`` shape+dtype a call makes."""
    records: list[tuple] = []

    def wrap(original):
        def recording(*args, **kwargs):
            out = original(*args, **kwargs)
            records.append((tuple(out.shape), out.dtype))
            return out

        return recording

    monkeypatch.setattr(torch, "zeros", wrap(torch.zeros))
    monkeypatch.setattr(torch, "empty", wrap(torch.empty))
    return records


def _assert_no_length_n_value(records: list[tuple], n_dets: int, dtype: torch.dtype) -> None:
    """No recorded allocation is a length-``N`` tensor of the value dtype."""
    assert not any(shape == (n_dets,) and recorded == dtype for shape, recorded in records)


# ============================================================================
# 1. Single process
# ============================================================================


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_operator_world_size_one_matches_reference(complex_case: bool) -> None:
    """world_size=1: the tile operator reproduces ``mult_spinless_torch``."""
    basis, helpers, integrals, hii, x = _fixture(complex_case)
    tile, bounds = spinless_tile_partition(basis.dets, SPLIT, 1, grid=(1, 1))
    bound = bounds[0]
    op = LocalSpinlessTileOperator(
        hii[bound.determinant_index],
        basis,
        helpers,
        integrals,
        bound,
        tile,
        rank=0,
        world_size=1,
    )
    expected = mult_spinless_torch(hii, x, basis, helpers, integrals)
    got = op(x[bound.determinant_index])
    assert got.dtype == expected.dtype
    assert got.shape == (bound.local_size,)
    # The tile enumerates from the ket side, so each output row accumulates its
    # terms in a different (still deterministic) order than the forward table:
    # the remaining difference is summation round-off, bounded at 1e-12.
    assert torch.allclose(got, expected[bound.determinant_index], atol=1e-12, rtol=1e-12)
    assert op.needed_size == 0
    assert op.max_stage_size == 0


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_operator_never_allocates_a_length_n_value_vector(
    monkeypatch: pytest.MonkeyPatch, complex_case: bool
) -> None:
    """The matvec allocates no length-``N`` tensor of the value dtype."""
    basis, helpers, integrals, hii, x = _fixture(complex_case)
    tile, bounds = spinless_tile_partition(basis.dets, SPLIT, 1, grid=(1, 1))
    bound = bounds[0]
    op = LocalSpinlessTileOperator(
        hii[bound.determinant_index],
        basis,
        helpers,
        integrals,
        bound,
        tile,
        rank=0,
        world_size=1,
    )
    records = _record_length_n_allocations(monkeypatch)
    op(x)
    _assert_no_length_n_value(records, basis.dim, x.dtype)


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_operator_triton_flag_falls_back_to_the_batched_scatter(
    monkeypatch: pytest.MonkeyPatch, complex_case: bool
) -> None:
    """``triton_matvec=True`` on CPU declines into the batched scatter, bit-exactly."""
    basis, helpers, integrals, hii, x = _fixture(complex_case)
    tile, bounds = spinless_tile_partition(basis.dets, SPLIT, 1, grid=(1, 1))
    bound = bounds[0]
    batched = LocalSpinlessTileOperator(
        hii[bound.determinant_index],
        basis,
        helpers,
        integrals,
        bound,
        tile,
        rank=0,
        world_size=1,
    )
    want = batched(x[bound.determinant_index])

    real_apply = distributed_spinless_tile.apply_double_entries
    ran: list[bool] = []

    def recording(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        ran.append(result)
        return result

    monkeypatch.setattr(distributed_spinless_tile, "apply_double_entries", recording)
    tiled = LocalSpinlessTileOperator(
        hii[bound.determinant_index],
        basis,
        helpers,
        integrals,
        bound,
        tile,
        rank=0,
        world_size=1,
        triton_matvec=True,
    )
    got = tiled(x[bound.determinant_index])

    # The flag is an attempt, not a promise: every doubles table reached the
    # predicate, which declined off CUDA, and the scatter that ran is exactly
    # the un-flagged operator's.
    assert ran and not any(ran)
    assert torch.equal(got, want)


# ============================================================================
# 2. gloo: 2, 4 and 8 ranks
# ============================================================================


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _run_spawned(fn, world_size: int, args: tuple, tmp_path, name: str) -> None:
    """Spawn ``world_size`` ranks and assert the run wrote its OK marker."""
    init_file = tmp_path / f"{name}_init"
    out_file = tmp_path / f"{name}_out"
    mp.start_processes(
        fn,
        args=(world_size, *args, str(init_file), str(out_file)),
        nprocs=world_size,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


def _tile_worker(rank, world_size, complex_case, grid, init_file, out_file) -> None:
    """Every rank's tile output equals the full matvec's slice, staged in bounded buffers."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis, helpers, integrals, hii, x = _fixture(complex_case)
        expected = mult_spinless_torch(hii, x, basis, helpers, integrals)
        tile, all_bounds = spinless_tile_partition(basis.dets, SPLIT, world_size, grid=grid)
        bounds = all_bounds[rank]
        index = bounds.determinant_index
        op = LocalSpinlessTileOperator(
            hii[index],
            basis,
            helpers,
            integrals,
            bounds,
            tile,
            rank=rank,
            world_size=world_size,
        )

        zeros, empty = torch.zeros, torch.empty
        records: list[tuple] = []

        def recording_zeros(*args, **kwargs):
            out = zeros(*args, **kwargs)
            records.append((tuple(out.shape), out.dtype))
            return out

        def recording_empty(*args, **kwargs):
            out = empty(*args, **kwargs)
            records.append((tuple(out.shape), out.dtype))
            return out

        torch.zeros, torch.empty = recording_zeros, recording_empty
        try:
            got = op(x[index])
        finally:
            torch.zeros, torch.empty = zeros, empty

        ref = expected[index]
        assert got.shape == ref.shape
        # Ket-side enumeration reorders each row's accumulation; combined
        # atol/rtol 1e-12 bounds the resulting summation round-off.
        assert torch.allclose(got, ref, atol=1e-12, rtol=1e-12), (
            rank,
            world_size,
            float((got - ref).abs().max()),
        )
        _assert_no_length_n_value(records, basis.dim, x.dtype)

        # Each receive buffer is one stage, never the whole vector, and the
        # concurrent staging (receive + send payload) never exceeds one vector.
        assert op.max_stage_size < basis.dim
        assert op.max_stage_size == max(op.stage_sizes)
        assert op.needed_size == sum(op.stage_sizes)
        assert op.max_send_size <= bounds.local_size
        assert op.peak_staged_bytes >= op.max_stage_size * x.element_size()
        assert op.peak_staged_bytes <= basis.dim * x.element_size()
        if op.needed_size:
            assert op.max_stage_size > 0
            assert op.peak_staged_bytes > 0

        # The 1-D owner-computes operator over the canonical flat partition.
        flat = partition_vector(basis.dim, world_size, rank)
        op_flat = LocalSpinlessOperator(
            hii[flat.start : flat.stop],
            basis,
            helpers,
            integrals,
            flat,
            rank=rank,
            world_size=world_size,
        )
        got_flat = op_flat(x[flat.start : flat.stop])
        assert torch.allclose(got_flat, expected[flat.start : flat.stop], atol=1e-12, rtol=0.0)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_operator_gloo_matches_reference(tmp_path, complex_case: bool, world_size: int) -> None:
    """gloo at 2/4/8 ranks: tile and 1-D blocks match the full matvec."""
    _skip_without_gloo()
    _run_spawned(
        _tile_worker,
        world_size,
        (complex_case, GRIDS[world_size]),
        tmp_path,
        f"tile{world_size}{'c' if complex_case else 'r'}",
    )


# ============================================================================
# 3. Empty tiles and an empty needed set
# ============================================================================


def _empty_worker(rank, world_size, grid, init_file, out_file) -> None:
    """A full-occupation basis has no excitations: empty tiles and empty needs."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis = spinless_basis_from_array([0b11], 2, 2)
        helpers = generate_spinless_helpers(basis)
        h1, eri = _real_spinorbital_integrals(2, seed=17)
        integrals = to_torch_integrals_spinless(h1, eri, dtype=torch.float64)
        hii = make_diag_terms_spinless_torch(basis, integrals)
        x = torch.tensor([0.5], dtype=torch.float64)
        expected = mult_spinless_torch(hii, x, basis, helpers, integrals)

        tile, all_bounds = spinless_tile_partition(basis.dets, 1, world_size, grid=grid)
        bounds = all_bounds[rank]
        op = LocalSpinlessTileOperator(
            hii[bounds.determinant_index],
            basis,
            helpers,
            integrals,
            bounds,
            tile,
            rank=rank,
            world_size=world_size,
        )
        got = op(x[bounds.determinant_index])
        assert op.needed_size == 0
        assert op.max_stage_size == 0
        assert op.peak_staged_bytes == 0
        assert got.shape == (bounds.local_size,)
        if bounds.is_empty:
            assert op.N_local == 0
            assert got.numel() == 0
        else:
            assert torch.allclose(got, expected[bounds.determinant_index], atol=1e-12)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("world_size", "grid"), [(2, (2, 1)), (4, (2, 2))])
def test_operator_empty_tile_and_empty_needed_set(tmp_path, world_size: int, grid) -> None:
    """Empty tiles and an empty needed set run the collective path cleanly."""
    _skip_without_gloo()
    _run_spawned(_empty_worker, world_size, (grid,), tmp_path, f"empty{world_size}")


def _sampled_worker(rank, world_size, grid, init_file, out_file) -> None:
    """A sparse basis makes some peers request part of a block, not all of it."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis = _sampled_basis_object(NMO, N_ELEC, 200, seed=41)
        helpers = generate_spinless_helpers(basis)
        h1, eri = _real_spinorbital_integrals(NMO, seed=101)
        integrals = to_torch_integrals_spinless(h1, eri, dtype=torch.float64)
        hii = make_diag_terms_spinless_torch(basis, integrals)
        gen = torch.Generator().manual_seed(7)
        x = torch.randn(basis.dim, generator=gen, dtype=torch.float64)

        tile, all_bounds = spinless_tile_partition(basis.dets, SPLIT, world_size, grid=grid)
        bounds = all_bounds[rank]
        op = LocalSpinlessTileOperator(
            hii[bounds.determinant_index],
            basis,
            helpers,
            integrals,
            bounds,
            tile,
            rank=rank,
            world_size=world_size,
        )
        got = op(x[bounds.determinant_index])
        expected = mult_spinless_torch(hii, x, basis, helpers, integrals)
        assert torch.allclose(got, expected[bounds.determinant_index], atol=1e-12, rtol=1e-12)
        assert op.max_stage_size < basis.dim
        assert op.peak_staged_bytes <= basis.dim * x.element_size()
        flags: list[int] = [0] * world_size
        dist.all_gather_object(flags, int(op.max_send_size), group=None)
        if rank == 0:
            # At least one directed stage ships a strict subset of the sender's
            # block, which is the copy path (not the whole-block alias).
            assert any(flags)
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("world_size", "grid"), [(4, (2, 2)), (8, (2, 4))])
def test_operator_partial_send_staging(tmp_path, world_size: int, grid) -> None:
    """Sparse uid rectangles exercise the partial-send copy path."""
    _skip_without_gloo()
    _run_spawned(_sampled_worker, world_size, (grid,), tmp_path, f"sampled{world_size}")


# ============================================================================
# 4. Uid-axis helpers used for ket-owner routing
# ============================================================================


def _sampled_basis_object(n_spinorb: int, n_particles: int, count: int, seed: int) -> SpinlessBasis:
    """A few hundred random ``n_particles``-from-``n_spinorb`` determinants."""
    rng = np.random.default_rng(seed)
    strs: set[int] = set()
    while len(strs) < count:
        combo = rng.choice(n_spinorb, n_particles, replace=False)
        strs.add(sum(1 << int(bit) for bit in combo))
    return spinless_basis_from_array(sorted(strs), n_spinorb, n_particles)


def _sampled_basis(n_spinorb: int, n_particles: int, count: int, seed: int) -> torch.Tensor:
    """The determinant limb matrix of :func:`_sampled_basis_object`."""
    return _sampled_basis_object(n_spinorb, n_particles, count, seed).dets


@pytest.mark.parametrize("split", [64, 70])
def test_uid_axis_helpers_match_half_string_uids(split: int) -> None:
    """Precomputed axes and subset uid lookups agree with the full decomposition."""
    dets = _sampled_basis(130, 5, 250, seed=11)
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    axes_rows, axes_cols = half_string_axes(dets, split)
    assert torch.equal(axes_rows, unique_rows)
    assert torch.equal(axes_cols, unique_cols)

    subset = torch.tensor([0, 3, 7, 11, 249])
    got_rows, got_cols = half_string_uids_of(dets[subset], split, axes_rows, axes_cols)
    assert torch.equal(got_rows, row_uids[subset])
    assert torch.equal(got_cols, col_uids[subset])


@pytest.mark.parametrize("split", [32, 65])
def test_uid_axis_helpers_handle_single_and_cross_limb_splits(split: int) -> None:
    """A single-limb and an unaligned multi-limb split both resolve exactly."""
    dets = _sampled_basis(66, 3, 200, seed=23)
    row_uids, col_uids, unique_rows, unique_cols = half_string_uids(dets, split)
    axes_rows, axes_cols = half_string_axes(dets, split)
    assert torch.equal(axes_rows, unique_rows)
    assert torch.equal(axes_cols, unique_cols)
    got_rows, got_cols = half_string_uids_of(dets[:17], split, axes_rows, axes_cols)
    assert torch.equal(got_rows, row_uids[:17])
    assert torch.equal(got_cols, col_uids[:17])


def test_uid_axis_lookup_rejects_absent_half_strings() -> None:
    """A determinant outside the basis raises instead of returning a wrong uid."""
    dets = _sampled_basis(66, 3, 32, seed=29)
    axes_rows, axes_cols = half_string_axes(dets, 32)
    foreign = dets[:1].clone()
    foreign[0, 0] ^= 1 << 40
    with pytest.raises(ValueError, match="absent"):
        half_string_uids_of(foreign, 32, axes_rows, axes_cols)


def test_auto_policy_keeps_the_gather_when_it_fits() -> None:
    """``tile="auto"`` keeps 1-D for a basis the gather fits with headroom."""
    # 1e8 determinants fp64: ~0.9 GB of gathered vector against an 80 GiB budget.
    assert not resolve_auto_spinless_tiling(
        world_size=4,
        n_dets=100_000_000,
        block_size=30,
        itemsize=8,
        budget_bytes=80 << 30,
    )


def test_auto_policy_tiles_when_the_gather_does_not_fit() -> None:
    """The tile path is chosen only when the 1-D input vector blows the budget."""
    assert resolve_auto_spinless_tiling(
        world_size=4,
        n_dets=10_000_000_000,
        block_size=30,
        itemsize=8,
        budget_bytes=80 << 30,
    )


def test_auto_policy_without_a_budget_or_grid_keeps_1d() -> None:
    """CPU runs and single ranks never tile."""
    assert not resolve_auto_spinless_tiling(world_size=4, n_dets=10_000_000_000, budget_bytes=None)
    assert not resolve_auto_spinless_tiling(world_size=1, n_dets=10_000_000_000, budget_bytes=1)


def test_tile_partition_balances_determinant_counts() -> None:
    """A concentrated uid map is cut on determinant counts, not uid counts.

    Row uids 0 and 1 carry ten determinants each while uids 2 and 3 carry one,
    so the even uid cut gives slabs of 20 and 2 determinants. The weighted cut
    moves the boundary to the 10/12 determinant split the auto-tiling model
    assumes.
    """
    dets = torch.tensor(
        sorted(
            (row << 4) | col
            for row, count in ((0, 10), (1, 10), (2, 1), (3, 1))
            for col in range(count)
        ),
        dtype=torch.int64,
    ).reshape(-1, 1)
    even = partition_tiles(4, 10, 2, grid=(2, 1))
    assert [even.rows_for_rank(rank) for rank in range(2)] == [(0, 2), (2, 4)]
    tile, bounds = spinless_tile_partition(dets, 4, 2, grid=(2, 1))
    assert tile.grid.alpha_bounds == ((0, 1), (1, 4))
    assert [bound.local_size for bound in bounds] == [10, 12]
    covered = torch.cat([bound.determinant_index for bound in bounds]).sort().values
    assert torch.equal(covered, torch.arange(dets.shape[0]))


# ============================================================================
# 5. End-to-end solver wiring
# ============================================================================

_E2E_NMO = 10
_E2E_NELEC = 5
_E2E_TOL = 1e-8


def _e2e_strings() -> np.ndarray:
    """All ``C(10, 5) = 252`` determinants, above the local-path block cap."""
    return np.asarray(
        [
            sum(1 << bit for bit in combo)
            for combo in itertools.combinations(range(_E2E_NMO), _E2E_NELEC)
        ],
        dtype=np.int64,
    )


def _e2e_distributed_config(world_size: int, tile=None) -> dict:
    """gloo config for the public solver's spawned workers."""
    config: dict = {
        "world_size": int(world_size),
        "backend": "gloo",
        "timeout": 180,
        "persistent": False,
    }
    if tile is not None:
        config["tile"] = tile
    return config


def _solve_spinless_state(distributed, **overrides):
    """One spinless solve through the public solver; returns its first state."""
    kwargs = dict(
        device="cpu",
        dtype=torch.complex128,
        n_roots=1,
        davidson_tolerance=1e-9,
        max_davidson_iters=100,
        **overrides,
    )
    if distributed is not None:
        kwargs["distributed"] = distributed
    solver = make_torch_sci_solver(**kwargs)
    hcore, eri = _complex_spinorbital_integrals(_E2E_NMO, seed=101)
    strings = _e2e_strings()
    try:
        results = solver([strings], hcore, eri, _E2E_NMO, _E2E_NELEC)
        return results[0].states[0]
    finally:
        solver.close()


@pytest.mark.parametrize("world_size", [2, 4])
def test_solver_spinless_tile_force_matches_1d(world_size: int) -> None:
    """tile='force' matches the 1-D owner-computes solve at 1e-8."""
    _skip_without_gloo()
    flat = _solve_spinless_state(_e2e_distributed_config(world_size))
    tiled = _solve_spinless_state(_e2e_distributed_config(world_size, tile="force"))
    assert float(tiled.energy) == pytest.approx(float(flat.energy), abs=_E2E_TOL)
    assert np.allclose(
        np.asarray(tiled.orbital_occupancies[0]),
        np.asarray(flat.orbital_occupancies[0]),
        atol=_E2E_TOL,
        rtol=0.0,
    )


def test_solver_spinless_tile_auto_resolves_to_1d_on_cpu() -> None:
    """No device budget on CPU, so tile='auto' keeps the 1-D solve."""
    _skip_without_gloo()
    flat = _solve_spinless_state(_e2e_distributed_config(2))
    auto = _solve_spinless_state(_e2e_distributed_config(2, tile="auto"))
    assert float(auto.energy) == pytest.approx(float(flat.energy), abs=_E2E_TOL)
    assert np.allclose(
        np.asarray(auto.orbital_occupancies[0]),
        np.asarray(flat.orbital_occupancies[0]),
        atol=_E2E_TOL,
        rtol=0.0,
    )


def test_solver_spinless_explicit_tile_grid() -> None:
    """An explicit (2, 1) grid runs the tile operator and matches the 1-D solve."""
    _skip_without_gloo()
    flat = _solve_spinless_state(_e2e_distributed_config(2))
    grid = _solve_spinless_state(_e2e_distributed_config(2, tile=(2, 1)))
    assert float(grid.energy) == pytest.approx(float(flat.energy), abs=_E2E_TOL)
    assert np.allclose(
        np.asarray(grid.orbital_occupancies[0]),
        np.asarray(flat.orbital_occupancies[0]),
        atol=_E2E_TOL,
        rtol=0.0,
    )


def test_solver_spinless_full_rdm_matches_the_serial_solve() -> None:
    """compute_full_rdm runs on the owner-computes solve and still returns both RDMs.

    The RDM build gathers the eigenvector once on rank 0; the solve itself
    stays distributed (flat or tiled), so this checks the gathered-vector RDM
    against the serial solve.
    """
    _skip_without_gloo()
    reference = _solve_spinless_state(None, compute_full_rdm=True)
    tiled = _solve_spinless_state(_e2e_distributed_config(2, tile="force"), compute_full_rdm=True)
    assert float(tiled.energy) == pytest.approx(float(reference.energy), abs=_E2E_TOL)
    assert tiled.rdm1 is not None and tiled.rdm2 is not None
    assert np.allclose(tiled.rdm1, reference.rdm1, atol=_E2E_TOL, rtol=0.0)
    assert np.allclose(tiled.rdm2, reference.rdm2, atol=_E2E_TOL, rtol=0.0)
    flat = _solve_spinless_state(_e2e_distributed_config(2), compute_full_rdm=True)
    assert np.allclose(flat.rdm1, reference.rdm1, atol=_E2E_TOL, rtol=0.0)
    assert np.allclose(flat.rdm2, reference.rdm2, atol=_E2E_TOL, rtol=0.0)


# ============================================================================
# 6. CUDA: entry-tiled doubles in the tile operator
# ============================================================================


def _skip_without_nccl() -> None:
    """Skip when this host cannot run a two-rank NCCL process group."""
    if not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and dist.is_available()
        and dist.is_nccl_available()
    ):
        pytest.skip("two CUDA devices and NCCL are required")


def _triton_gpu_worker(rank, world_size, complex_case, grid, init_file, out_file) -> None:
    """Entry-tiled partials match the batched ones; their sum matches serial."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        basis, helpers, integrals, hii, x = _fixture(complex_case, device)
        expected = mult_spinless_torch(hii, x, basis, helpers, integrals)
        tile, all_bounds = spinless_tile_partition(basis.dets, SPLIT, world_size, grid=grid)
        bounds = all_bounds[rank]
        index = bounds.determinant_index
        common = dict(
            hii_local=hii[index],
            basis=basis,
            helpers=helpers,
            integrals=integrals,
            bounds=bounds,
            partition=tile,
            rank=rank,
            world_size=world_size,
        )
        got_batched = LocalSpinlessTileOperator(**common)(x[index])

        real_apply = distributed_spinless_tile.apply_double_entries
        ran: list[bool] = []

        def recording(*args, **kwargs):
            result = real_apply(*args, **kwargs)
            ran.append(result)
            return result

        distributed_spinless_tile.apply_double_entries = recording
        try:
            got_tiled = LocalSpinlessTileOperator(**common, triton_matvec=True)(x[index])
        finally:
            distributed_spinless_tile.apply_double_entries = real_apply

        # The kernel must actually run (no silent fallback), then agree with
        # the batched operator's partial; the ket-side enumeration reorders
        # each row's accumulation, so the bound is summation round-off.
        assert ran and all(ran)
        assert torch.allclose(got_tiled, got_batched, atol=1e-12, rtol=1e-12), (
            rank,
            float((got_tiled - got_batched).abs().max()),
        )

        # The tiles are disjoint and cover the basis, so summing the staged
        # partials reassembles the full matvec, which matches the serial one.
        full = torch.zeros(basis.dim, dtype=x.dtype, device=device)
        full[index] = got_tiled
        dist.all_reduce(full, op=dist.ReduceOp.SUM)
        assert torch.allclose(full, expected, atol=1e-12, rtol=1e-12), (
            rank,
            float((full - expected).abs().max()),
        )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_operator_triton_matvec_matches_batched_on_gpu(tmp_path, complex_case: bool) -> None:
    """NCCL at two ranks: the entry kernel reproduces the batched partials."""
    _skip_without_nccl()
    pytest.importorskip("triton")
    _run_spawned(
        _triton_gpu_worker,
        2,
        (complex_case, GRIDS[2]),
        tmp_path,
        f"tritongpu{'c' if complex_case else 'r'}",
    )
