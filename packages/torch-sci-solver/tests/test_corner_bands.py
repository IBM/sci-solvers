"""Regression tests for the row-banded αβ corner exchange on rectangular grids.

A staged corner pair index covers two independent messages: the rectangle a
rank receives from its ``+`` partner and the rectangle it sends to its ``-``
partner. On an uneven grid those two rectangles have different shapes, while
the sender and the receiver of one directed message see the same shape (the
receiver's needed slice and the sender's send positions are the same set).
The band plan and its P2P tags therefore have to pair per directed message.

These tests drive the real gloo exchanges on 1x2, 2x2, 2x3 and 3x2 grids with
an ``exchange_bytes`` budget far below one rectangle, so every non-empty
corner is split into several row bands, for real and complex integrals, and
compare every rank's tile matvec with the full-space ``mult_torch`` at
``1e-12``. The 2x3 and 3x2 cases use a 5x6 basis (uneven row slabs) and one
1-row 2x3 case exercises empty row slabs under the same budget. A
single-rank case guards the unchanged exchange-free path.
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
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
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


# ============================================================================
# Real gloo exchanges with a tiny element budget
# ============================================================================


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _corner_worker(
    rank: int,
    world_size: int,
    cases: tuple[tuple[tuple[int, int], int, bool, int], ...],
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's banded tile matvec matches the full-space matvec's block."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        for grid, n_alpha, complex_dtype, exchange_bytes in cases:
            hii, adets_t, bdets_t, integrals, helpers, x = _setup(n_alpha, complex_dtype)
            M_a, M_b = len(adets_t), len(bdets_t)
            tile = partition_tiles(M_a, M_b, world_size, grid=grid)
            expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
            bounds = tile.for_rank(rank)
            op = dmat.LocalTPBOperator(
                _tile_slice(hii, M_a, M_b, bounds),
                adets_t,
                bdets_t,
                NORB,
                integrals,
                _local_helpers(adets_t.tolist(), bdets_t.tolist(), bounds),
                tile,
                rank=rank,
                world_size=world_size,
                exchange_bytes=exchange_bytes,
            )
            n_pairs = (tile.grid.p_alpha - 1) * (tile.grid.p_beta - 1)
            if n_pairs and n_alpha > 1:
                # With a budget below one rectangle every non-empty corner is
                # split, so the plan must be longer than the pair count.
                assert len(op._corner_exchange.pairs()) > n_pairs, (
                    rank,
                    grid,
                    "the exchange budget did not split any corner stage",
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


def test_corner_bands_two_ranks_column_split(tmp_path) -> None:
    """gloo, 2 ranks on a 1x2 grid: real and complex, no corner pairs."""
    _skip_without_gloo()
    cases = (
        ((1, 2), M_STRINGS, False, 16),
        ((1, 2), M_STRINGS, True, 16),
    )
    _run_spawned(_corner_worker, 2, (cases,), tmp_path, "corner1x2")


def test_corner_bands_four_ranks_square(tmp_path) -> None:
    """gloo, 4 ranks on a 2x2 grid: real and complex, every corner banded."""
    _skip_without_gloo()
    cases = (
        ((2, 2), M_STRINGS, False, 16),
        ((2, 2), M_STRINGS, True, 16),
    )
    _run_spawned(_corner_worker, 4, (cases,), tmp_path, "corner2x2")


def test_corner_bands_six_ranks_rectangular(tmp_path) -> None:
    """gloo, 6 ranks on 2x3 and 3x2 grids of a 5x6 basis (uneven rows)."""
    _skip_without_gloo()
    cases = (
        ((2, 3), 5, False, 16),
        ((2, 3), 5, True, 16),
        ((3, 2), 5, False, 16),
        ((3, 2), 5, True, 16),
        # One α row over two row slabs: the second slab is empty.
        ((2, 3), 1, False, 16),
    )
    _run_spawned(_corner_worker, 6, (cases,), tmp_path, "cornerrect")


# ============================================================================
# Single rank: the exchange-free path is unchanged
# ============================================================================


def test_corner_bands_single_rank_matches_mult_torch() -> None:
    """A 1x1 grid with an exchange budget is exactly ``mult_torch``."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, 1, grid=(1, 1))
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    op = dmat.LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        tile,
        rank=0,
        world_size=1,
        exchange_bytes=16,
    )
    assert op._corner_exchange.pairs() == ()
    assert torch.allclose(op(x), expected, atol=1e-12, rtol=0.0)
