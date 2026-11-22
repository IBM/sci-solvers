"""Tests for the 2-D tile αβ-swap projector.

Covers:

1. Single process: the whole-basis projector (grid 1x1) matches the
   full-space ``(v + τ vᵀ) / 2`` formula on real and complex vectors.
2. gloo multi-rank: every rank's tile of ``P(v)`` equals the full-space
   formula restricted to that tile, on square, rectangular, uneven and
   empty-slab grids, real and complex.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

NORB = 4
# C(4, 2) = 6 strings per spin.
M_STRINGS = 6


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _strings(n: int = M_STRINGS) -> list[int]:
    """The first ``n`` of the ``C(4, 2)`` two-orbital strings, sorted."""
    all_strings = sorted(
        int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(NORB), 2)
    )
    return all_strings[:n]


def _vector(M: int, complex_dtype: bool, seed: int = 11) -> torch.Tensor:
    """A deterministic ``(M * M,)`` test vector."""
    g = torch.Generator().manual_seed(seed)
    if complex_dtype:
        return torch.complex(
            torch.randn(M * M, generator=g, dtype=torch.float64),
            torch.randn(M * M, generator=g, dtype=torch.float64),
        )
    return torch.randn(M * M, generator=g, dtype=torch.float64)


def _reference(
    v: torch.Tensor, adets_t: torch.Tensor, bdets_t: torch.Tensor, M: int
) -> torch.Tensor:
    """The full-space projector ``(v + τ vᵀ) / 2`` on the ``(M, M)`` matrix."""
    sigma = _inter_to_block_signs(adets_t, bdets_t, NORB)
    tau = (sigma * sigma.T).to(v.dtype)
    v2d = v.view(M, M)
    return (v2d + tau * v2d.T) * 0.5


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


def _projector_worker(
    rank: int,
    world_size: int,
    grid: tuple[int, int],
    n_strings: int,
    complex_dtype: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's tile of ``P(v)`` matches the full-space formula."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        strings = _strings(n_strings)
        M = len(strings)
        adets_t = torch.tensor(strings, dtype=torch.int64)
        bdets_t = adets_t.clone()
        tile = partition_tiles(M, M, world_size, grid=grid)
        bounds = tile.for_rank(rank)

        v = _vector(M, complex_dtype)
        want_full = _reference(v, adets_t, bdets_t, M)
        want = want_full[
            bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop
        ].reshape(-1)

        projector = dmat.LocalTileAlphaBetaProjector(
            adets_t,
            bdets_t,
            NORB,
            tile,
            rank=rank,
            world_size=world_size,
        )
        v_tile = (
            v
            .view(M, M)[bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop]
            .reshape(-1)
            .clone()
        )
        got = projector(v_tile)

        assert got.shape == want.shape
        diff = float((got - want).abs().max()) if got.numel() else 0.0
        assert diff <= 1e-12, (rank, grid, n_strings, complex_dtype, diff)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_projector_whole_basis_real() -> None:
    """Single process, 1x1 grid: matches the full-space projector."""
    strings = _strings()
    M = len(strings)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = adets_t.clone()
    tile = partition_tiles(M, M, 1)
    v = _vector(M, False)
    want = _reference(v, adets_t, bdets_t, M).reshape(-1)
    got = dmat.LocalTileAlphaBetaProjector(adets_t, bdets_t, NORB, tile)(v.clone())
    assert float((got - want).abs().max()) <= 1e-12


def test_projector_whole_basis_complex() -> None:
    """Single process, 1x1 grid: the complex path matches the formula."""
    strings = _strings()
    M = len(strings)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = adets_t.clone()
    tile = partition_tiles(M, M, 1)
    v = _vector(M, True)
    want = _reference(v, adets_t, bdets_t, M).reshape(-1)
    got = dmat.LocalTileAlphaBetaProjector(adets_t, bdets_t, NORB, tile)(v.clone())
    assert float((got - want).abs().max()) <= 1e-12


def test_projector_rejects_shape_mismatch() -> None:
    """A non-square basis is refused up front."""
    adets_t = torch.tensor(_strings(), dtype=torch.int64)
    bdets_t = torch.tensor(_strings(5), dtype=torch.int64)
    tile = partition_tiles(6, 5, 1)
    with pytest.raises(ValueError, match="M_a == M_b"):
        dmat.LocalTileAlphaBetaProjector(adets_t, bdets_t, NORB, tile)


def test_projector_gloo_square(tmp_path) -> None:
    """gloo, 4 ranks on a 2x2 grid: one transposed peer exchange."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 4, ((2, 2), M_STRINGS, False), tmp_path, "proj4")


def test_projector_gloo_rectangular(tmp_path) -> None:
    """gloo, 6 ranks on a 2x3 grid: multi-source rectangles."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 6, ((2, 3), M_STRINGS, False), tmp_path, "proj6")


def test_projector_gloo_rectangular_transposed(tmp_path) -> None:
    """gloo, 6 ranks on a 3x2 grid: the other rectangular orientation."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 6, ((3, 2), M_STRINGS, False), tmp_path, "proj6b")


def test_projector_gloo_uneven(tmp_path) -> None:
    """gloo, 6 ranks on a 2x3 grid of a 5-string basis (uneven rows)."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 6, ((2, 3), 5, False), tmp_path, "proj6u")


def test_projector_gloo_empty_slabs(tmp_path) -> None:
    """gloo, 9 ranks on a 3x3 grid of a 2-string basis (empty slabs)."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 9, ((3, 3), 2, False), tmp_path, "proj9e")


def test_projector_gloo_complex(tmp_path) -> None:
    """gloo, 4 ranks on a 2x2 grid with complex vectors."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 4, ((2, 2), M_STRINGS, True), tmp_path, "proj4c")
