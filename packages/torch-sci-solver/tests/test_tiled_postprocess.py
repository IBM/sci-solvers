"""Tests for the tiled 1-RDM, 2-RDM, <S^2> and S^2-application post-processing.

The tiled expectations are checked against the serial RDM builders
(:func:`make_one_rdm_torch`, :func:`make_two_rdm_torch` +
:func:`s_squared_from_rdm`) on a small TPB basis, over 1x1, 2x2 and 2x3 gloo
grids (the 2x3 case uses the uneven 5x6 basis), real and complex, plus the
degenerate row-split grids and a row-banded corner exchange. The serial
``SpinSqOperator`` expectation and ``energy_from_rdm`` provide independent
convention oracles; its full-vector ``apply`` also checks the tiled
``S^2 |x>`` block, and the particle-number trace and permutational symmetries
sanity-check the normalisation.
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
from torch_sci_solver.torch_impl import tiled_postprocess as tpost
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.distributed_matvec import LocalTPBOperator
from torch_sci_solver.torch_impl.helpers import (
    generate_all_torch_helpers,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch
from torch_sci_solver.torch_impl.rdm import (
    _inter_to_block_signs,
    energy_from_rdm,
    make_one_rdm_torch,
    make_two_rdm_torch,
    s_squared_from_rdm,
)
from torch_sci_solver.torch_impl.spin_sq import SpinSqOperator

NORB = 4
# C(4, 2) = 6 strings per spin: a 6x6 basis, and a 5x6 rectangular one.
M_STRINGS = 6
N_A = 2
N_B = 2
# The serial oracle and the tiled expectation must agree to fp64 round-off.
ATOL = 1e-10


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


def _setup(n_alpha: int, complex_dtype: bool) -> tuple:
    """Build a normalised full vector and everything the operators need."""
    if complex_dtype:
        h1, eri, strings = _complex_hermitian_h4(seed=0)
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
        integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
        real = torch.randn(
            len(strings[:n_alpha]) * len(strings),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
        )
        imag = torch.randn(
            len(strings[:n_alpha]) * len(strings),
            generator=torch.Generator().manual_seed(8),
            dtype=torch.float64,
        )
        x = torch.complex(real, imag)
    else:
        h1, eri, strings = _synthetic_h4()
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
        integrals = to_torch_integrals(I0, I1, I2, device="cpu")
        x = torch.randn(
            len(strings[:n_alpha]) * len(strings),
            generator=torch.Generator().manual_seed(7),
            dtype=torch.float64,
        )
    # A normalised vector makes Tr γ_σ = n_σ, the particle-number check.
    x = x / x.norm()
    adets = strings[:n_alpha]
    bdets = list(strings)
    adets_t = torch.tensor(adets, dtype=torch.int64)
    bdets_t = torch.tensor(bdets, dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(adets, bdets, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    return hii, adets_t, bdets_t, integrals, helpers, x


def _tile_slice(tensor_flat: torch.Tensor, n_rows: int, n_cols: int, bounds) -> torch.Tensor:
    """The tile's block of a flat ``(n_rows, n_cols)`` vector, in local order."""
    block = tensor_flat.view(n_rows, n_cols)[
        bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop
    ]
    return block.reshape(-1)


def _local_helpers(adets_t: torch.Tensor, bdets_t: torch.Tensor, bounds) -> list:
    """Tile-local helpers for one (row slab, column slab)."""
    return list(
        generate_local_torch_helpers(
            adets_t.tolist(),
            bdets_t.tolist(),
            NORB,
            bounds.row_start,
            bounds.row_stop,
            device="cpu",
            col_start=bounds.col_start,
            col_stop=bounds.col_stop,
        )
    )


def _check_against_serial(
    operator: LocalTPBOperator,
    x_tile: torch.Tensor,
    x: torch.Tensor,
    hii: torch.Tensor,
    adets_t: torch.Tensor,
    bdets_t: torch.Tensor,
    integrals,
    helpers: list,
) -> None:
    """Assert the tiled observables match the serial builders and invariants."""
    n_a, n_b = N_A, N_B
    gamma_a, gamma_b = tpost.tiled_one_rdm(operator, x_tile)
    s_sq = tpost.tiled_spin_squared(operator, x_tile, n_a, n_b)
    fused_a, fused_b, fused_sq = tpost.tiled_rdm1_spin_sq(operator, x_tile, n_a, n_b)

    # The fused sweep and the single-purpose entry points must agree.
    assert torch.allclose(gamma_a, fused_a, atol=1e-12, rtol=0.0)
    assert torch.allclose(gamma_b, fused_b, atol=1e-12, rtol=0.0)
    assert torch.allclose(s_sq, fused_sq, atol=1e-12, rtol=0.0)

    # Caller-provided accumulation buffers receive the reduced observable.
    buffer_a = torch.zeros((NORB, NORB), dtype=x.dtype)
    buffer_b = torch.zeros((NORB, NORB), dtype=x.dtype)
    got_a, got_b = tpost.tiled_one_rdm(operator, x_tile, gamma_a=buffer_a, gamma_b=buffer_b)
    assert got_a is buffer_a and got_b is buffer_b
    assert torch.allclose(got_a, gamma_a, atol=1e-12, rtol=0.0)
    assert torch.allclose(got_b, gamma_b, atol=1e-12, rtol=0.0)

    # Serial oracle: same vector, same basis, spin-block conversion included.
    ref_a, ref_b = make_one_rdm_torch(x, adets_t, bdets_t, NORB, helpers)
    ref_aa, ref_ab, ref_bb = make_two_rdm_torch(x, adets_t, bdets_t, NORB, helpers)
    ref_s2 = s_squared_from_rdm(ref_ab, n_a, n_b)
    assert gamma_a.dtype == ref_a.dtype and gamma_b.dtype == ref_b.dtype
    assert torch.allclose(gamma_a, ref_a, atol=ATOL, rtol=0.0), float((gamma_a - ref_a).abs().max())
    assert torch.allclose(gamma_b, ref_b, atol=ATOL, rtol=0.0), float((gamma_b - ref_b).abs().max())
    assert torch.allclose(s_sq, ref_s2, atol=ATOL, rtol=0.0), float(abs(s_sq - ref_s2))

    # The comparison must not be vacuous: the off-diagonal blocks are populated.
    off_a = gamma_a - torch.diag(gamma_a.diagonal())
    off_b = gamma_b - torch.diag(gamma_b.diagonal())
    assert float(off_a.abs().max()) > 1e-3
    assert float(off_b.abs().max()) > 1e-3

    # Particle-number sanity: the traces are the sector electron counts.
    trace_a = gamma_a.diagonal().sum()
    trace_b = gamma_b.diagonal().sum()
    assert float(abs(trace_a - n_a)) <= ATOL
    assert float(abs(trace_b - n_b)) <= ATOL

    # Tiled spin-resolved 2-RDM: same index order and spin blocks as the
    # serial builders, real and complex.
    Gaa, Gab, Gbb = tpost.tiled_two_rdm(operator, x_tile)
    for got, ref in ((Gaa, ref_aa), (Gab, ref_ab), (Gbb, ref_bb)):
        assert got.dtype == ref.dtype
        assert torch.allclose(got, ref, atol=ATOL, rtol=0.0), float((got - ref).abs().max())

    # Caller-provided accumulation buffers receive the reduced blocks.
    shape2 = (NORB, NORB, NORB, NORB)
    buf_aa = torch.zeros(shape2, dtype=x.dtype)
    buf_ab = torch.zeros(shape2, dtype=x.dtype)
    buf_bb = torch.zeros(shape2, dtype=x.dtype)
    got_aa, got_ab, got_bb = tpost.tiled_two_rdm(
        operator, x_tile, gamma2_a=buf_aa, gamma2_b=buf_bb, gamma2_ab=buf_ab
    )
    assert got_aa is buf_aa and got_ab is buf_ab and got_bb is buf_bb
    assert torch.allclose(got_aa, Gaa, atol=1e-12, rtol=0.0)
    assert torch.allclose(got_ab, Gab, atol=1e-12, rtol=0.0)
    assert torch.allclose(got_bb, Gbb, atol=1e-12, rtol=0.0)

    # Non-vacuity: classes A-D of Γ_αβ write disjoint (diagonal, off-diagonal)
    # slot categories, so every category must be populated.
    off = ~torch.eye(NORB, dtype=torch.bool)
    diag = ~off

    def _category(tensor: torch.Tensor, alpha_diag: bool, beta_diag: bool) -> float:
        """Largest magnitude in one (α-side, β-side) diagonal category."""
        first = diag if alpha_diag else off
        second = diag if beta_diag else off
        mask = first[:, :, None, None] & second[None, None, :, :]
        return float(tensor.abs()[mask].max())

    assert _category(Gab, True, True) > 1e-3, "class A is zero"  # diagonal x diagonal
    assert _category(Gab, True, False) > 1e-3, "class B is zero"  # diagonal x single
    assert _category(Gab, False, True) > 1e-3, "class C is zero"  # single x diagonal
    assert _category(Gab, False, False) > 1e-3, "class D is zero"  # single x single

    # Γ_σσ class 1 writes slots of the form [a, b, c, c] and class 2 writes
    # slots with all four indices distinct; class 0 writes neither, so each
    # pattern isolates one off-diagonal class.
    orbital4 = torch.arange(NORB)
    i_idx, j_idx, k_idx, l_idx = torch.meshgrid(
        orbital4, orbital4, orbital4, orbital4, indexing="ij"
    )
    class1 = (k_idx == l_idx) & (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
    class2 = (
        (i_idx != j_idx)
        & (i_idx != k_idx)
        & (i_idx != l_idx)
        & (j_idx != k_idx)
        & (j_idx != l_idx)
        & (k_idx != l_idx)
    )
    assert float(Gaa.abs()[class1].max()) > 1e-3, "Γaa single-excitation class is zero"
    assert float(Gaa.abs()[class2].max()) > 1e-3, "Γaa double-excitation class is zero"
    assert float(Gbb.abs()[class1].max()) > 1e-3, "Γbb single-excitation class is zero"
    assert float(Gbb.abs()[class2].max()) > 1e-3, "Γbb double-excitation class is zero"

    # Spin-summed 2-RDM Γ2 = Γaa + Γbb + Γab + Γba, with the βα block
    # Γba[p, q, r, s] = Γab[r, s, p, q]. In this chemist index order the
    # particle-number trace contracts the two index pairs as Γ2[p, p, q, q],
    # giving N(N-1).
    spin_summed = Gaa + Gbb + Gab + Gab.permute(2, 3, 0, 1)
    n_total = n_a + n_b
    trace2 = torch.einsum("ppqq->", spin_summed)
    assert float(abs(trace2 - n_total * (n_total - 1))) <= ATOL

    # Permutational symmetries of the convention: swapping both index pairs
    # leaves Γ_σσ invariant and conjugates Γ_αβ (a no-op for real amplitudes).
    assert torch.allclose(Gaa, Gaa.permute(2, 3, 0, 1), atol=ATOL, rtol=0.0)
    assert torch.allclose(Gbb, Gbb.permute(2, 3, 0, 1), atol=ATOL, rtol=0.0)
    assert torch.allclose(Gab, Gab.permute(1, 0, 3, 2).conj(), atol=ATOL, rtol=0.0)

    # Independent <S^2> oracle: the serial scatter operator in spin-block order.
    dtype = x.dtype
    sigma = _inter_to_block_signs(adets_t, bdets_t, NORB).to(dtype=dtype)
    x_block = x * sigma.reshape(-1)
    spin_op = SpinSqOperator(adets_t, bdets_t, NORB, n_a, n_b, dtype=dtype)
    operator_s2 = torch.vdot(x_block, spin_op.apply(x_block))
    assert float(abs(s_sq - operator_s2)) <= ATOL

    # The matched class-D term must be non-zero, otherwise <S^2> would only
    # test class A. Subtract the analytic class-A-only value.
    p_range = torch.arange(NORB)
    real_dtype = x.dtype.to_real() if x.is_complex() else x.dtype
    a_occ = ((adets_t.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(real_dtype)
    b_occ = ((bdets_t.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(real_dtype)
    c2_full = x.abs().pow(2).view(len(adets_t), len(bdets_t))
    class_a = (a_occ.T @ c2_full @ b_occ).diagonal().sum()
    sz = (n_a - n_b) / 2.0
    class_a_only = sz * sz - sz + n_a - class_a
    assert float(abs(s_sq - class_a_only)) > 1e-6, "matched class-D term is zero"

    # energy_from_rdm with the tiled 1-RDM and the tiled 2-RDM must equal the
    # Rayleigh quotient: this catches a wrong parity, index or σ convention.
    h_x = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    rayleigh = torch.vdot(x, h_x)
    energy = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, integrals)
    assert float(abs(energy - rayleigh.real)) <= ATOL

    _check_s2_apply(operator, x_tile, x, h_x, adets_t, bdets_t)


def _serial_s2_vector(
    x: torch.Tensor, adets_t: torch.Tensor, bdets_t: torch.Tensor
) -> torch.Tensor:
    """Full ``S^2 x`` in the interleaved-Wick convention, from the serial operator."""
    sigma = _inter_to_block_signs(adets_t, bdets_t, NORB).to(x.dtype).reshape(-1)
    spin_op = SpinSqOperator(adets_t, bdets_t, NORB, N_A, N_B, dtype=x.dtype)
    return spin_op.apply(x * sigma) * sigma


def _gather_tiles(operator: LocalTPBOperator, *tiles: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Gather each rank's flattened tile into full vectors (identity when alone)."""
    tile = operator._tile
    assert tile is not None  # tile-path operator
    world_size = int(operator._world_size)
    if world_size == 1:
        return tiles
    pieces: list = [None] * world_size
    dist.all_gather_object(pieces, [tile_part.detach().cpu() for tile_part in tiles])
    fulls = [torch.zeros(tile.n, dtype=t.dtype) for t in tiles]
    for rank, rank_pieces in enumerate(pieces):
        bounds = tile.for_rank(rank)
        rows = slice(bounds.row_start, bounds.row_stop)
        cols = slice(bounds.col_start, bounds.col_stop)
        for full, piece in zip(fulls, rank_pieces):
            full.view(tile.n_rows, tile.n_cols)[rows, cols] = piece.view(
                bounds.local_rows, bounds.local_cols
            )
    return tuple(fulls)


def _check_s2_apply(
    operator: LocalTPBOperator,
    x_tile: torch.Tensor,
    x: torch.Tensor,
    h_x: torch.Tensor,
    adets_t: torch.Tensor,
    bdets_t: torch.Tensor,
) -> None:
    """Assert the tiled S² application matches the serial operator.

    The serial ``SpinSqOperator`` is built in spin-block convention, so the
    oracle converts the input and the result with the interleaved-to-block
    signs. The tiled result is compared both block-by-block and after an
    all-gather, its Rayleigh quotient is compared with the existing
    ``tiled_spin_squared`` expectation, and a penalized ``H + λS²`` matvec is
    checked against the serial operator pair.
    """
    M_a, M_b = len(adets_t), len(bdets_t)
    y_ref = _serial_s2_vector(x, adets_t, bdets_t)
    y_local = tpost.tiled_s2_apply(operator, x_tile)
    assert y_local.shape == x_tile.shape and y_local.dtype == x_tile.dtype

    # The owned block is the serial result's slice of the tile.
    piece = y_ref.view(M_a, M_b)[
        operator._row_start : operator._row_stop,
        operator._col_start : operator._col_stop,
    ].reshape(-1)
    assert torch.allclose(y_local, piece, atol=ATOL, rtol=0.0), float((y_local - piece).abs().max())

    # Gathered vector: the serial full-vector apply, and the existing
    # tiled <S^2> expectation as the Rayleigh quotient.
    lam = 0.75
    penalized_local = operator(x_tile) + lam * y_local
    y_full, penalized_full = _gather_tiles(operator, y_local, penalized_local)
    assert torch.allclose(y_full, y_ref, atol=ATOL, rtol=0.0), float((y_full - y_ref).abs().max())
    s_sq = tpost.tiled_spin_squared(operator, x_tile, N_A, N_B)
    assert float(abs(torch.vdot(x, y_full) - s_sq)) <= ATOL

    # Penalized matvec H + λS² against the serial pair.
    reference = h_x + lam * y_ref
    assert torch.allclose(penalized_full, reference, atol=1e-12, rtol=0.0), float(
        (penalized_full - reference).abs().max()
    )

    # Linearity: apply(a·u + b·v) == a·apply(u) + b·apply(v).
    real_dtype = x_tile.dtype.to_real() if x_tile.dtype.is_complex else x_tile.dtype
    generator = torch.Generator().manual_seed(13)
    u = torch.randn(x_tile.shape, generator=generator, dtype=real_dtype).to(x_tile.dtype)
    v = torch.randn(x_tile.shape, generator=generator, dtype=real_dtype).to(x_tile.dtype)
    a, b = 1.5, -0.75
    lhs = tpost.tiled_s2_apply(operator, (a * u + b * v).reshape(-1))
    rhs = a * tpost.tiled_s2_apply(operator, u.reshape(-1)) + b * tpost.tiled_s2_apply(
        operator, v.reshape(-1)
    )
    assert torch.allclose(lhs, rhs, atol=ATOL, rtol=0.0)


def _worker(
    rank: int,
    world_size: int,
    grid: tuple[int, int],
    n_alpha: int,
    complex_dtype: bool,
    exchange_bytes: int | None,
    init_file: str,
    out_file: str,
) -> None:
    """One gloo rank: build the tile operator, post-process, compare to serial."""
    from pathlib import Path

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        hii, adets_t, bdets_t, integrals, helpers, x = _setup(n_alpha, complex_dtype)
        M_a, M_b = len(adets_t), len(bdets_t)
        tile = partition_tiles(M_a, M_b, world_size, grid=grid)
        bounds = tile.for_rank(rank)
        operator = LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            _local_helpers(adets_t, bdets_t, bounds),
            tile,  # type: ignore[arg-type]  # the operator accepts a TilePartition
            rank=rank,
            world_size=world_size,
            exchange_bytes=exchange_bytes,
        )
        _check_against_serial(
            operator,
            _tile_slice(x, M_a, M_b, bounds),
            x,
            hii,
            adets_t,
            bdets_t,
            integrals,
            helpers,
        )
        if exchange_bytes is not None:
            n_pairs = (tile.grid.p_alpha - 1) * (tile.grid.p_beta - 1)
            assert len(operator._corner_exchange.pairs()) > n_pairs, (
                "the exchange budget did not split any corner stage"
            )
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


def test_tiled_postprocess_single_rank() -> None:
    """A 1x1 tile reproduces the serial 1-RDM and <S^2> without a group."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup(M_STRINGS, False)
    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, 1, grid=(1, 1))
    bounds = tile.for_rank(0)
    operator = LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        tile,  # type: ignore[arg-type]  # the operator accepts a TilePartition
        rank=0,
        world_size=1,
    )
    _check_against_serial(
        operator,
        _tile_slice(x, M_a, M_b, bounds),
        x,
        hii,
        adets_t,
        bdets_t,
        integrals,
        helpers,
    )

    # Analytic identity: a single determinant has
    # <S^2> = n_a - N_dbl + S_z^2 - S_z, with no off-diagonal contribution.
    det_index = 1  # ia=0, ib=1: one doubly occupied and one singly occupied orbital
    one_det = torch.zeros_like(x)
    one_det[det_index] = 1.0
    s_sq_det = tpost.tiled_spin_squared(operator, one_det, N_A, N_B)
    ia, ib = divmod(det_index, M_b)
    n_dbl = bin(int(adets_t[ia]) & int(bdets_t[ib])).count("1")
    sz = (N_A - N_B) / 2.0
    expected = (N_A - n_dbl) + sz * sz - sz
    assert abs(float(s_sq_det) - expected) <= ATOL


def test_tiled_postprocess_single_rank_complex() -> None:
    """A 1x1 tile reproduces the complex128 serial 1-RDM, 2-RDM and <S^2>."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup(M_STRINGS, True)
    M_a, M_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(M_a, M_b, 1, grid=(1, 1))
    bounds = tile.for_rank(0)
    operator = LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        tile,  # type: ignore[arg-type]  # the operator accepts a TilePartition
        rank=0,
        world_size=1,
    )
    _check_against_serial(
        operator,
        _tile_slice(x, M_a, M_b, bounds),
        x,
        hii,
        adets_t,
        bdets_t,
        integrals,
        helpers,
    )


@pytest.mark.parametrize(
    ("world_size", "grid", "n_alpha", "complex_dtype"),
    [
        (4, (2, 2), M_STRINGS, False),
        (6, (2, 3), 5, False),
        (4, (2, 2), M_STRINGS, True),
        (6, (2, 3), 5, True),
        (2, (2, 1), M_STRINGS, False),
        (2, (1, 2), M_STRINGS, False),
    ],
    ids=["2x2-real", "2x3-real", "2x2-complex", "2x3-complex", "2x1-real", "1x2-real"],
)
def test_tiled_postprocess_gloo(
    tmp_path,
    world_size: int,
    grid: tuple[int, int],
    n_alpha: int,
    complex_dtype: bool,
) -> None:
    """Tiled 1-RDM and <S^2> match the serial builders over real gloo exchanges."""
    _skip_without_gloo()
    _run_spawned(
        _worker,
        world_size,
        (grid, n_alpha, complex_dtype, None),
        tmp_path,
        f"tpost{world_size}_{grid[0]}x{grid[1]}_{'c' if complex_dtype else 'r'}",
    )


def test_tiled_postprocess_gloo_banded_corners(tmp_path) -> None:
    """A row-banded corner exchange reproduces the serial class-D pairs.

    The budget is far below a corner rectangle at this basis size, so every
    corner sub-stage ships one row and the operator's α tables are the rebased
    band tables; the matched ``Γ_αβ[p, q, q, p]`` sum must still agree.
    """
    _skip_without_gloo()
    _run_spawned(_worker, 4, ((2, 2), M_STRINGS, False, 16), tmp_path, "tpostband")
