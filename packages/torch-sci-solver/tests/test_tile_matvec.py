"""Tests for the 2-D tile owner-computes TPB matvec.

Covers:

1. Single process with a stand-in exchange that slices a full vector:
   every rank of a rectangular tile grid (including uneven and empty slabs)
   reproduces the corresponding tile of the full-space ``mult_torch``
   matvec, with full and with tile-local helpers.
2. ``p_beta == 1`` degenerates to the legacy row-partition operator: the two
   operators are bit-identical on every rank.
3. gloo multi-rank: real :class:`StagedSlabExchange` / :class:`StagedCornerExchange`
   over grids 2x1, 1x2, 2x2 and 2x3 (uneven rows) match the full-space matvec
   to ``1e-12``; the complex path is checked at 2x2.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl.distributed_davidson import (
    TilePartition,
    partition_rows_explicit,
    partition_tiles,
)
from torch_sci_solver.torch_impl.helpers import (
    generate_all_torch_helpers,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch

NORB = 4
# C(4, 2) = 6 strings per spin -> a 6x6 TPB basis (and a 5x6 rectangular one).
M_STRINGS = 6


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _synthetic_h4() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Deterministic 4-orbital / 4-electron integrals and full CI strings."""
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
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
    strings = [int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(NORB), 2)]
    return h1, eri, sorted(strings)


def _complex_hermitian_h4(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Complex Hermitian 4-orbital integrals (8-fold-symmetric ERIs) + strings."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
    hcore = hc + hc.conj().T
    eri = np.zeros((NORB, NORB, NORB, NORB), dtype=np.complex128)
    for _ in range(NORB + 2):
        g = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
        basis = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", basis, basis.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    strings = [int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(NORB), 2)]
    return hcore, eri, sorted(strings)


def _setup(n_alpha: int = M_STRINGS, complex_dtype: bool = False) -> tuple:
    """Build ``(hii, adets_t, bdets_t, integrals, helpers, x)`` for one basis."""
    if complex_dtype:
        h1, eri, strings = _complex_hermitian_h4(seed=0)
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
        integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
        adets = strings[:n_alpha]
        bdets = list(strings)
        real = torch.randn(len(adets) * len(bdets), generator=torch.Generator().manual_seed(7))
        imag = torch.randn(len(adets) * len(bdets), generator=torch.Generator().manual_seed(8))
        x = torch.complex(real, imag)
    else:
        h1, eri, strings = _synthetic_h4()
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
        integrals = to_torch_integrals(I0, I1, I2, device="cpu")
        adets = strings[:n_alpha]
        bdets = list(strings)
        x = torch.randn(
            len(adets) * len(bdets),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
        )
    adets_t = torch.tensor(adets, dtype=torch.int64)
    bdets_t = torch.tensor(bdets, dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(adets, bdets, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    return hii, adets_t, bdets_t, integrals, helpers, x


def _local_helpers(adets: list[int], bdets: list[int], bounds) -> list:
    """Tile-local helpers for one (row slab, column slab)."""
    return list(
        generate_local_torch_helpers(
            adets,
            bdets,
            NORB,
            bounds.row_start,
            bounds.row_stop,
            device="cpu",
            col_start=bounds.col_start,
            col_stop=bounds.col_stop,
        )
    )


def _tile_slice(tensor_flat: torch.Tensor, n_rows: int, n_cols: int, bounds) -> torch.Tensor:
    """The tile's block of a flat ``(n_rows, n_cols)`` vector, in local order."""
    return tensor_flat.view(n_rows, n_cols)[
        bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop
    ].reshape(-1)


class _FullTileExchange:
    """Stand-in for :class:`StagedSlabExchange` that slices a full vector.

    The stage index lists match the real exchange's (needed ∩ peer slab), so
    the stage tables the operator builds are the ones the real path builds;
    only the transfer itself is replaced by a slice of ``full_vector``.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, partition, axis, needed, *, rank=None, world_size=None, group=None, tag=0):
        """Record the partition, axis and needed indices."""
        _ = group, tag
        self.partition = partition
        self.axis = axis
        self.rank = rank
        self.world_size = world_size
        self._needed = torch.as_tensor(needed, dtype=torch.int64).reshape(-1).cpu()
        grid = partition.grid
        self._bound = partition.for_rank(rank)
        if axis == "alpha":
            self._group_pos, self._group_size = self._bound.row_slab, grid.p_alpha
            self._start, self._stop = self._bound.row_start, self._bound.row_stop
        else:
            self._group_pos, self._group_size = self._bound.col_slab, grid.p_beta
            self._start, self._stop = self._bound.col_start, self._bound.col_stop
        self.all_needed = [self._needed] * world_size

    def _peer_range(self, stage: int) -> tuple[int, int]:
        """Global index range of the stage's peer slab."""
        pos = (self._group_pos + stage) % self._group_size
        grid = self.partition.grid
        return grid.rows(pos) if self.axis == "alpha" else grid.cols(pos)

    def stage_indices(self, stage: int) -> torch.Tensor:
        """Sorted global indices the stage's buffer holds."""
        if stage == 0:
            return torch.arange(self._start, self._stop)
        lo, hi = self._peer_range(stage)
        needed = self._needed
        return needed[(needed >= lo) & (needed < hi)]

    def send_positions(self, peer_rank: int) -> torch.Tensor:
        """Positions this tile would ship (unused: the stand-in has no sends)."""
        _ = peer_rank
        return torch.empty(0, dtype=torch.int64)

    def stages(self, x_tile: torch.Tensor):
        """Yield ``(stage, block)`` for every stage."""
        full = _FullTileExchange.full_vector.view(self._bound.n_rows, self._bound.n_cols)
        rows = slice(self._bound.row_start, self._bound.row_stop)
        cols = slice(self._bound.col_start, self._bound.col_stop)
        for stage in range(self._group_size):
            if stage == 0:
                yield 0, x_tile
            elif self.axis == "alpha":
                yield stage, full[self.stage_indices(stage)][:, cols].contiguous()
            else:
                yield stage, full[rows][:, self.stage_indices(stage)].contiguous()


class _FullTileCornerExchange:
    """Stand-in for :class:`StagedCornerExchange` that slices a full vector.

    Mirrors the real exchange's sub-stage plan, including the row bands a
    ``max_elems`` budget produces, so the operator's banded α tables are
    exercised without a process group.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, alpha, beta, *, tag=0, max_elems=None):
        """Record the α/β stage lists and the banded sub-stage plan."""
        _ = tag
        self.partition = alpha.partition
        self.rank = alpha.rank
        self._alpha = alpha
        self._beta = beta
        grid = alpha.partition.grid
        pairs = tuple(
            (stage_a, stage_b)
            for stage_a in range(1, grid.p_alpha)
            for stage_b in range(1, grid.p_beta)
        )
        rows = {k: alpha.stage_indices(sa) for k, (sa, _) in enumerate(pairs)}
        cols = {k: beta.stage_indices(sb) for k, (_, sb) in enumerate(pairs)}
        self._plan: list[tuple[int, int, int, int]] = []
        self._rows: dict[int, torch.Tensor] = {}
        self._cols: dict[int, torch.Tensor] = {}
        for stage, (s_a, s_b) in enumerate(pairs):
            n_rows = int(rows[stage].numel())
            n_cols = int(cols[stage].numel())
            if max_elems is None or n_cols == 0 or n_rows * n_cols <= int(max_elems):
                bands = ((0, n_rows),)
            else:
                rows_per = max(1, int(max_elems) // n_cols)
                bands = tuple((lo, min(lo + rows_per, n_rows)) for lo in range(0, n_rows, rows_per))
            for row_lo, row_hi in bands:
                sub = len(self._plan)
                self._plan.append((s_a, s_b, row_lo, row_hi))
                self._rows[sub] = rows[stage][row_lo:row_hi]
                self._cols[sub] = cols[stage]

    def pairs(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(s_a, s_b, row_lo, row_hi)`` of every sub-stage, in stage order."""
        return tuple(self._plan)

    def stages(self, x_tile: torch.Tensor):
        """Yield ``(stage, rectangle)`` for every remote corner sub-stage."""
        _ = x_tile
        full = _FullTileCornerExchange.full_vector.view(
            self.partition.n_rows, self.partition.n_cols
        )
        for stage in range(len(self._plan)):
            rows = self._rows[stage]
            cols = self._cols[stage]
            yield stage, full[rows][:, cols].contiguous()


def _install_stand_ins(monkeypatch: pytest.MonkeyPatch, x: torch.Tensor) -> None:
    """Point the operator's exchanges at the full-vector stand-ins."""
    monkeypatch.setattr(dmat, "StagedSlabExchange", _FullTileExchange)
    monkeypatch.setattr(dmat, "StagedCornerExchange", _FullTileCornerExchange)
    _FullTileExchange.full_vector = x
    _FullTileCornerExchange.full_vector = x


# ============================================================================
# 1. Single process over every rank of a grid
# ============================================================================


@pytest.mark.parametrize(
    ("world_size", "grid", "n_alpha"),
    [
        (2, (2, 1), M_STRINGS),
        (2, (1, 2), M_STRINGS),
        (4, (2, 2), M_STRINGS),
        (6, (2, 3), 5),
        (6, (3, 2), 5),
        (8, (8, 1), M_STRINGS),
        (8, (1, 8), M_STRINGS),
    ],
)
@pytest.mark.parametrize("use_local_helpers", [False, True])
def test_tile_operator_matches_full_slice(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    grid: tuple[int, int],
    n_alpha: int,
    use_local_helpers: bool,
) -> None:
    """Every rank's tile reproduces the full-space matvec's block to 1e-12."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup(n_alpha)
    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, world_size, grid=grid)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    _install_stand_ins(monkeypatch, x)
    adets = adets_t.tolist()
    bdets = bdets_t.tolist()

    for rank in range(world_size):
        bounds = tile.for_rank(rank)
        hs = _local_helpers(adets, bdets, bounds) if use_local_helpers else helpers
        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            hs,
            tile,
            rank=rank,
            world_size=world_size,
        )
        got = op(_tile_slice(x, M_a, M_b, bounds))
        want = _tile_slice(expected, M_a, M_b, bounds)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0), (
            rank,
            grid,
            use_local_helpers,
            float((got - want).abs().max()) if got.numel() else 0.0,
        )


@pytest.mark.parametrize("exchange_bytes", [16, 64], ids=["tiny", "small"])
def test_tile_operator_exchange_budget_matches_full(
    monkeypatch: pytest.MonkeyPatch, exchange_bytes: int
) -> None:
    """A row-banded αβ corner exchange reproduces the full-space matvec.

    The budget is far below a corner rectangle at this basis size, so every
    corner stage is split into bands and the operator's α tables are rebased
    per band; the block must still match ``mult_torch`` to 1e-12.
    """
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    world_size = 4
    tile = partition_tiles(M_a, M_b, world_size, grid=(2, 2))
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    _install_stand_ins(monkeypatch, x)
    banded = 0
    for rank in range(world_size):
        bounds = tile.for_rank(rank)
        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            helpers,
            tile,
            rank=rank,
            world_size=world_size,
            exchange_bytes=exchange_bytes,
        )
        plan = op._corner_exchange.pairs()
        n_pairs = (tile.grid.p_alpha - 1) * (tile.grid.p_beta - 1)
        banded += 1 if len(plan) > n_pairs else 0
        got = op(_tile_slice(x, M_a, M_b, bounds))
        want = _tile_slice(expected, M_a, M_b, bounds)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0)
    assert banded > 0, "the exchange budget did not split any corner stage"


def test_resolve_auto_tile_grid_decides_by_memory() -> None:
    """The auto policy tiles only when the 1-D worst case would be large.

    The table bound is conservative, so the 1e8-scale basis stays on the faster
    1-D split while the 1e10-scale basis (tables + block over half the card)
    tiles; without a device budget (CPU runs) the 1-D split is kept, matching
    the pair and spinless policies.
    """
    from torch_sci_solver.torch_impl.distributed_davidson import resolve_auto_tile_grid

    budget = 80 << 30
    assert (
        resolve_auto_tile_grid(
            world_size=16,
            n_rows=10_000,
            n_cols=10_000,
            n_particles_a=27,
            n_particles_b=27,
            n_orb=36,
            budget_bytes=budget,
        )
        is None
    )
    assert resolve_auto_tile_grid(
        world_size=16,
        n_rows=100_000,
        n_cols=100_000,
        n_particles_a=27,
        n_particles_b=27,
        n_orb=36,
        budget_bytes=budget,
    ) == (4, 4)
    assert (
        resolve_auto_tile_grid(
            world_size=16,
            n_rows=100,
            n_cols=100,
            n_particles_a=2,
            n_particles_b=2,
            n_orb=6,
            budget_bytes=None,
        )
        is None
    )
    # A grid with one column slab keeps the full beta tables, so it is never
    # a memory win over the 1-D row split.
    assert (
        resolve_auto_tile_grid(
            world_size=16,
            n_rows=1_000_000,
            n_cols=16,
            n_particles_a=2,
            n_particles_b=2,
            n_orb=32,
            budget_bytes=1 << 30,
        )
        is None
    )


def test_tile_operator_single_rank_matches_mult_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 1x1 grid is exactly ``mult_torch`` on the whole basis."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, 1, grid=(1, 1))
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    _install_stand_ins(monkeypatch, x)
    op = dmat.LocalTPBOperator(
        hii, adets_t, bdets_t, NORB, integrals, helpers, tile, rank=0, world_size=1
    )
    assert torch.allclose(op(x), expected, atol=1e-12, rtol=0.0)


# ============================================================================
# 2. gloo: real exchanges
# ============================================================================


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _tile_worker(
    rank: int,
    world_size: int,
    grid: tuple[int, int],
    n_alpha: int,
    complex_dtype: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's real tile exchange matches the full-space matvec's block."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        hii, adets_t, bdets_t, integrals, helpers, x = _setup(n_alpha, complex_dtype)
        M_a, M_b = len(adets_t), len(bdets_t)
        tile = partition_tiles(M_a, M_b, world_size, grid=grid)
        expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
        bounds = tile.for_rank(rank)
        hs = _local_helpers(adets_t.tolist(), bdets_t.tolist(), bounds)

        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            hs,
            tile,
            rank=rank,
            world_size=world_size,
        )
        got = op(_tile_slice(x, M_a, M_b, bounds))
        want = _tile_slice(expected, M_a, M_b, bounds)
        assert got.shape == want.shape
        diff = float((got - want).abs().max()) if got.numel() else 0.0
        assert diff <= 1e-12, (rank, grid, complex_dtype, diff)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


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


def test_tile_gloo_two_ranks_rows(tmp_path) -> None:
    """gloo, 2 ranks on a 2x1 grid: matches the full matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 2, ((2, 1), M_STRINGS, False), tmp_path, "tile2a")


def test_tile_gloo_two_ranks_cols(tmp_path) -> None:
    """gloo, 2 ranks on a 1x2 grid: the β stream against a remote column slab."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 2, ((1, 2), M_STRINGS, False), tmp_path, "tile2b")


def test_tile_gloo_four_ranks_square(tmp_path) -> None:
    """gloo, 4 ranks on a 2x2 grid: all three streams, 1e-12."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 4, ((2, 2), M_STRINGS, False), tmp_path, "tile4")


def test_tile_gloo_six_ranks_rectangular_uneven(tmp_path) -> None:
    """gloo, 6 ranks on a 2x3 grid of a 5x6 basis (uneven rows), 1e-12."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 6, ((2, 3), 5, False), tmp_path, "tile6")


def test_tile_gloo_six_ranks_three_by_two(tmp_path) -> None:
    """gloo, 6 ranks on a 3x2 grid of a 5x6 basis (uneven rows), 1e-12."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 6, ((3, 2), 5, False), tmp_path, "tile6b")


def test_tile_gloo_eight_ranks_four_by_two(tmp_path) -> None:
    """gloo, 8 ranks on a 4x2 grid of a 5x6 basis: three remote columns each."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 8, ((4, 2), 5, False), tmp_path, "tile8")


def test_tile_gloo_four_ranks_complex(tmp_path) -> None:
    """gloo, 4 ranks on a 2x2 grid with complex integrals, 1e-12."""
    _skip_without_gloo()
    _run_spawned(_tile_worker, 4, ((2, 2), M_STRINGS, True), tmp_path, "tile4c")


# ============================================================================
# 3. p_beta == 1 is bit-identical to the legacy operator
# ============================================================================


def _degenerate_worker(
    rank: int, world_size: int, starts: tuple[int, ...], init_file: str, out_file: str
) -> None:
    """The p_beta=1 tile operator and the row-partition operator agree bitwise."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        hii, adets_t, bdets_t, integrals, helpers, x = _setup()
        M_a, M_b = len(adets_t), len(bdets_t)
        partition = partition_rows_explicit(M_a, M_b, list(starts), world_size)
        tile = TilePartition.from_row_partition(partition)
        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)
        local = list(
            generate_local_torch_helpers(
                adets_t.tolist(), bdets_t.tolist(), NORB, row_start, row_stop, device="cpu"
            )
        )

        legacy = dmat.LocalTPBOperator(
            hii[sl],
            adets_t,
            bdets_t,
            NORB,
            integrals,
            local,
            partition,
            rank=rank,
            world_size=world_size,
        )
        tiled = dmat.LocalTPBOperator(
            hii[sl],
            adets_t,
            bdets_t,
            NORB,
            integrals,
            local,
            tile,
            rank=rank,
            world_size=world_size,
        )
        got_legacy = legacy(x[sl])
        got_tiled = tiled(x[sl])
        assert torch.equal(got_legacy, got_tiled), (
            rank,
            float((got_legacy - got_tiled).abs().max()),
        )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_tile_gloo_degenerate_bit_identical(tmp_path) -> None:
    """gloo, 3 ranks: p_beta=1 is bit-identical to the legacy operator."""
    _skip_without_gloo()
    _run_spawned(_degenerate_worker, 3, ((0, 1, 4, 6),), tmp_path, "tiledeg")


# ============================================================================
# 4. Triton on a tile grid (CUDA only)
# ============================================================================


@pytest.mark.gpu
def test_tile_triton_matches_batched_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Triton tile path matches the batched tile path on every rank."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton operator")
    import torch_sci_solver.torch_impl.mult_triton as mt

    if not mt._HAVE_TRITON:
        pytest.skip("Triton is not importable")

    h1, eri, strings = _synthetic_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cuda")
    adets_t = torch.tensor(strings, dtype=torch.int64, device="cuda")
    bdets_t = torch.tensor(strings, dtype=torch.int64, device="cuda")
    helpers = list(generate_all_torch_helpers(strings, strings, NORB, device="cuda"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(len(strings) * len(strings), generator=gen, dtype=torch.float64, device="cuda")

    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, 4, grid=(2, 2))
    _install_stand_ins(monkeypatch, x)
    for rank in range(4):
        bounds = tile.for_rank(rank)
        x_tile = _tile_slice(x, M_a, M_b, bounds)
        hii_tile = _tile_slice(hii, M_a, M_b, bounds)
        common = dict(partition=tile, rank=rank, world_size=4)
        batched = dmat.LocalTPBOperator(
            hii_tile, adets_t, bdets_t, NORB, integrals, helpers, **common
        )(x_tile)
        triton = dmat.LocalTPBOperator(
            hii_tile,
            adets_t,
            bdets_t,
            NORB,
            integrals,
            helpers,
            matvec_impl="triton",
            **common,
        )(x_tile)
        assert torch.allclose(triton, batched, atol=1e-12, rtol=0.0), (
            rank,
            float((triton - batched).abs().max()) if triton.numel() else 0.0,
        )
