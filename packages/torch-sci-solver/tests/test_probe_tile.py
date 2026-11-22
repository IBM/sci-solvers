"""Tests for the tile-restricted warm-start eigvec embedding.

The tile helper must be the rectangle restriction of the row helper: the
cached interleaved-Wick state pasted by string identity into the extended
basis, zero where a string is absent, unnormalized, row-major inside the tile.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import pytest
import torch
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.probe import (
    _embed_eigvec_into_extended_basis,
    _embed_eigvec_into_extended_basis_rows,
    _embed_eigvec_into_extended_basis_tile,
)
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

_N_ORB = 6
_N_ELEC = 3


def _strings(n_orb: int, n_elec: int) -> list[int]:
    """All ``n_elec``-bit strings over ``n_orb`` orbitals, ascending."""
    out = []
    for combo in itertools.combinations(range(n_orb), n_elec):
        mask = 0
        for orb in combo:
            mask |= 1 << orb
        out.append(mask)
    out.sort()
    return out


def _row_rect(
    eigvec_old: torch.Tensor,
    cached_a: list[int],
    cached_b: list[int],
    ext_a: list[int],
    ext_b: list[int],
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Row helper's output on the tile rectangle, flattened row-major."""
    rows = _embed_eigvec_into_extended_basis_rows(
        eigvec_old,
        cached_a,
        cached_b,
        ext_a,
        ext_b,
        row_start,
        row_stop,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    return rows[:, col_start:col_stop].reshape(-1)


def _tile(
    eigvec_old: torch.Tensor,
    cached_a: list[int],
    cached_b: list[int],
    ext_a: list[int],
    ext_b: list[int],
    bound,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the tile helper on one rank bound of a partition."""
    return _embed_eigvec_into_extended_basis_tile(
        eigvec_old,
        cached_a,
        cached_b,
        ext_a,
        ext_b,
        bound.row_start,
        bound.row_stop,
        bound.col_start,
        bound.col_stop,
        device=torch.device("cpu"),
        dtype=dtype,
    )


def test_tile_matches_row_rectangle_with_missing_cached_strings():
    """Tiles equal the row helper's rectangle on 2x2 and 2x3 grids.

    The cached α basis deliberately contains strings absent from the extended
    basis: those cached amplitudes are dropped, and extended rows absent from
    the cached basis are zero.
    """
    all_strings = _strings(_N_ORB, _N_ELEC)
    removed = {2, 9, 14}
    ext_a = [s for i, s in enumerate(all_strings) if i not in removed]
    ext_a = ext_a[::2] + ext_a[1::2]  # non-monotone ordering
    ext_b = list(reversed(all_strings))
    cached_a = all_strings[2:13]  # includes cached strings missing from ext_a
    cached_b = all_strings[3:12]

    assert any(s not in set(ext_a) for s in cached_a)
    assert all(s in set(ext_b) for s in cached_b)

    M_a, M_b = len(ext_a), len(ext_b)
    M_a_old, M_b_old = len(cached_a), len(cached_b)
    eigvec_old = torch.arange(1, M_a_old * M_b_old + 1, dtype=torch.float64)

    for grid, world in (((2, 2), 4), ((2, 3), 6)):
        partition = partition_tiles(M_a, M_b, world, grid=grid)
        for bound in partition.bounds:
            tile = _tile(eigvec_old, cached_a, cached_b, ext_a, ext_b, bound, torch.float64)
            local = bound.local_rows * bound.local_cols
            assert tile.shape == (local,)
            expected = _row_rect(
                eigvec_old,
                cached_a,
                cached_b,
                ext_a,
                ext_b,
                bound.row_start,
                bound.row_stop,
                bound.col_start,
                bound.col_stop,
                torch.float64,
            )
            assert torch.equal(tile, expected)

    # Extended α rows absent from the cached basis are zero in the row helper.
    cached_a_set = set(cached_a)
    full_rows = _embed_eigvec_into_extended_basis_rows(
        eigvec_old,
        cached_a,
        cached_b,
        ext_a,
        ext_b,
        0,
        M_a,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    for i_a, s in enumerate(ext_a):
        if s not in cached_a_set:
            assert torch.count_nonzero(full_rows[i_a]).item() == 0


def test_tiles_assembled_across_ranks_equal_row_and_full_embedding():
    """Every tile pasted back at its bounds rebuilds the row helper's output."""
    all_strings = _strings(_N_ORB, _N_ELEC)
    # Superset case: every cached string appears in the extended basis, so the
    # full (normalized) helper is comparable too.
    ext_a = all_strings[::2] + all_strings[1::2]
    ext_b = list(reversed(all_strings))
    cached_a = sorted(set(all_strings[3:15]))
    cached_b = all_strings[2:13]

    M_a, M_b = len(ext_a), len(ext_b)
    M_a_old, M_b_old = len(cached_a), len(cached_b)
    generator = torch.Generator().manual_seed(20261113)
    eigvec_old = torch.randn(M_a_old * M_b_old, generator=generator, dtype=torch.float64)

    full_rows = _embed_eigvec_into_extended_basis_rows(
        eigvec_old,
        cached_a,
        cached_b,
        ext_a,
        ext_b,
        0,
        M_a,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    norm = torch.linalg.norm(full_rows)
    full_normed = _embed_eigvec_into_extended_basis(
        eigvec_old,
        cached_a,
        cached_b,
        ext_a,
        ext_b,
        device=torch.device("cpu"),
        dtype=torch.float64,
    ).view(M_a, M_b)

    for grid, world in (((2, 2), 4), ((2, 3), 6)):
        partition = partition_tiles(M_a, M_b, world, grid=grid)
        assembled = torch.zeros(M_a, M_b, dtype=torch.float64)
        covered = 0
        for bound in partition.bounds:
            tile = _tile(eigvec_old, cached_a, cached_b, ext_a, ext_b, bound, torch.float64)
            assembled[bound.row_start : bound.row_stop, bound.col_start : bound.col_stop] = (
                tile.view(bound.local_rows, bound.local_cols)
            )
            covered += bound.local_size
        assert covered == M_a * M_b
        assert torch.equal(assembled, full_rows)
        assert torch.allclose(assembled, full_normed * norm, atol=1e-12)


def test_tile_complex_dtypes():
    """The tile paste preserves complex values and requires no conjugation."""
    all_strings = _strings(_N_ORB, _N_ELEC)
    ext_a = all_strings[::2] + all_strings[1::2]
    ext_b = list(reversed(all_strings))
    cached_a = all_strings[4:16]
    cached_b = all_strings[1:10]

    M_a, M_b = len(ext_a), len(ext_b)
    M_a_old, M_b_old = len(cached_a), len(cached_b)
    generator = torch.Generator().manual_seed(7)
    real = torch.randn(M_a_old * M_b_old, generator=generator, dtype=torch.float64)
    imag = torch.randn(M_a_old * M_b_old, generator=generator, dtype=torch.float64)

    for dtype in (torch.complex128, torch.complex64):
        eigvec_old = torch.complex(real, imag).to(dtype)
        partition = partition_tiles(M_a, M_b, 4, grid=(2, 2))
        for bound in partition.bounds:
            tile = _tile(eigvec_old, cached_a, cached_b, ext_a, ext_b, bound, dtype)
            assert tile.dtype == dtype
            assert torch.is_complex(tile)
            expected = _row_rect(
                eigvec_old,
                cached_a,
                cached_b,
                ext_a,
                ext_b,
                bound.row_start,
                bound.row_stop,
                bound.col_start,
                bound.col_stop,
                dtype,
            )
            assert torch.equal(tile, expected)


def test_tile_sign_convention_matches_row_helper():
    """No extra interleaved-to-block sign is applied on the paste.

    The interleaved-to-block sign depends only on the (α, β) string pair, so
    the cached and extended bases share it at every pasted position. The test
    makes that non-trivial: the pasted rectangle contains positions where σ is
    -1 and positions where it is +1, and the tile still matches the row helper
    entry for entry (a stray σ factor would flip the -1 entries).
    """
    all_strings = _strings(_N_ORB, _N_ELEC)
    ext_a = all_strings
    ext_b = list(reversed(all_strings))
    cached_a = ext_a[4:14]
    cached_b = ext_b[2:12]

    M_a, M_b = len(ext_a), len(ext_b)
    M_a_old, M_b_old = len(cached_a), len(cached_b)
    eigvec_old = torch.arange(1, M_a_old * M_b_old + 1, dtype=torch.float64)

    sigma = _inter_to_block_signs(
        torch.tensor(ext_a, dtype=torch.int64),
        torch.tensor(ext_b, dtype=torch.int64),
        _N_ORB,
    )
    pos_a = {s: i for i, s in enumerate(ext_a)}
    pos_b = {s: i for i, s in enumerate(ext_b)}
    pasted = sigma[
        torch.tensor([pos_a[s] for s in cached_a]),
        :,
    ][:, torch.tensor([pos_b[s] for s in cached_b])]
    assert (pasted < 0).any() and (pasted > 0).any()

    partition = partition_tiles(M_a, M_b, 4, grid=(2, 2))
    for bound in partition.bounds:
        tile = _tile(eigvec_old, cached_a, cached_b, ext_a, ext_b, bound, torch.float64)
        expected = _row_rect(
            eigvec_old,
            cached_a,
            cached_b,
            ext_a,
            ext_b,
            bound.row_start,
            bound.row_stop,
            bound.col_start,
            bound.col_stop,
            torch.float64,
        )
        assert torch.equal(tile, expected)


def test_tile_rejects_missing_cached_beta():
    """A cached β-string absent from the extended basis raises, as for rows."""
    all_strings = _strings(_N_ORB, _N_ELEC)
    cached_a = all_strings[:5]
    cached_b = all_strings[:5]
    ext_a = list(all_strings)
    ext_b = all_strings[1:]  # drops cached_b[0]
    eigvec_old = torch.ones(len(cached_a) * len(cached_b), dtype=torch.float64)
    with pytest.raises(ValueError, match="extended basis"):
        _embed_eigvec_into_extended_basis_tile(
            eigvec_old,
            cached_a,
            cached_b,
            ext_a,
            ext_b,
            0,
            len(ext_a),
            0,
            len(ext_b),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
