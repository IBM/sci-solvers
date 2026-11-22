"""Tests for complex-dtype support in the distributed owner-computes TPB matvec.

Covers:

1. Single process: ``LocalTPBOperator`` (batched) matches ``mult_torch`` on a
   complex Hermitian TPB basis, for the full row block (full and local helpers)
   and for 1/2/3-row sub-blocks (a stand-in exchange over a full complex
   vector, so no process group is needed).
2. gloo, 2 ranks (uneven rows): every rank's complex output block matches the
   corresponding slice of the full-space ``mult_torch``.
3. gloo, 2 ranks: ``LocalAlphaBetaProjector`` matches the full-space
   ``(v + tau vᵀ)/2`` formula with a real ``tau`` and a complex ``v``.
4. β-pass cache parity on complex data: cached and uncached β kernels are
   bit-identical.
5. ``matvec_impl="triton"`` accepts complex integrals through the kernels'
   interleaved-real complex path; the launch itself still requires CUDA.
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
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl.distributed_davidson import partition_rows_explicit
from torch_sci_solver.torch_impl.helpers import (
    generate_all_torch_helpers,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import (
    _mult_beta_only_torch,
    build_beta_pass_cache,
    make_diag_terms_torch,
    mult_torch,
)

NORB = 4
# C(4, 2) = 6 strings per spin -> a 6-row, 36-determinant basis.
M_STRINGS = 6
N = M_STRINGS * M_STRINGS


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _complex_hermitian_h4(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Complex Hermitian 4-orbital integrals (8-fold-symmetric ERIs) + strings.

    Same construction as ``tests/test_complex_matvec.py``'s
    ``_random_complex_hermitian_integrals``: a Hermitian ``hcore`` and a
    Hermitian, ``(pq|rs) == (rs|pq)`` chemist-order ERI built as a sum of
    ``L L†`` outer products.
    """
    norb = NORB
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    strings = [int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(norb), 2)]
    return hcore, eri, sorted(strings)


def _setup() -> tuple:
    """Build the shared complex basis: ``(hii, adets_t, bdets_t, integrals, helpers, x)``."""
    hcore, eri, strings = _complex_hermitian_h4(seed=0)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = torch.tensor(strings, dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(strings, strings, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(N, generator=gen, dtype=torch.float64) + 1j * torch.randn(
        N, generator=gen, dtype=torch.float64
    )
    return hii, adets_t, bdets_t, integrals, helpers, x.to(torch.complex128)


def _local_helpers(strings: list[int], row_start: int, row_stop: int) -> list:
    """Local-row helpers for one α-row block."""
    return list(
        generate_local_torch_helpers(strings, strings, NORB, row_start, row_stop, device="cpu")
    )


class _FullVectorExchange:
    """Stand-in for :class:`StagedRowExchange` that slices a full vector.

    Lets the operator own an arbitrary row sub-block without a process group;
    identical to the one in ``tests/test_distributed_matvec.py``.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, partition, needed_rows, *, rank=None, world_size=None, group=None):
        """Record the partition and the needed rows."""
        _ = group
        self._partition = partition
        self._rank = rank
        self._world_size = world_size
        self._needed = needed_rows

    def rows(self, stage: int) -> torch.Tensor:
        """Sorted global rows in this stage's buffer (stage 0: local range)."""
        if stage == 0:
            row_start, row_stop = self._partition.rows_for_rank(self._rank)
            return torch.arange(row_start, row_stop)
        row_start, row_stop = self._partition.rows_for_rank((self._rank + stage) % self._world_size)
        needed = self._needed
        return needed[(needed >= row_start) & (needed < row_stop)]

    def stages(self, x_local_2d: torch.Tensor):
        """Yield ``(stage, buffer)`` for every stage."""
        for stage in range(self._world_size):
            if stage == 0:
                yield 0, x_local_2d
            else:
                yield (
                    stage,
                    _FullVectorExchange.full_vector.view(-1, self._partition.row_size)[
                        self.rows(stage)
                    ],
                )


# ============================================================================
# 1. Single process on the complex basis
# ============================================================================


@pytest.mark.parametrize("use_local_helpers", [False, True])
def test_operator_complex_single_process(use_local_helpers: bool) -> None:
    """world_size=1: the batched operator reproduces ``mult_torch`` in complex."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    hs = _local_helpers(adets_t.tolist(), 0, M_a) if use_local_helpers else helpers

    op = dmat.LocalTPBOperator(
        hii, adets_t, bdets_t, NORB, integrals, hs, partition, rank=0, world_size=1
    )
    got = op(x)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert not hii.dtype.is_complex
    assert got.dtype == torch.complex128
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_owned_rows", [1, 2, 3])
def test_operator_complex_subblock_matches_full_slice(
    monkeypatch: pytest.MonkeyPatch, n_owned_rows: int
) -> None:
    """A complex row sub-block (1/2/3 of 6 rows) reproduces the full slice."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)

    partition = partition_rows_explicit(M_a, M_b, [0, n_owned_rows, M_a], 2)
    row_start, row_stop = partition.rows_for_rank(0)
    sl = slice(row_start * M_b, row_stop * M_b)

    monkeypatch.setattr(dmat, "StagedRowExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    op = dmat.LocalTPBOperator(
        hii[sl],
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        partition,
        rank=0,
        world_size=2,
    )
    got = op(x[sl])
    assert got.dtype == torch.complex128
    assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0)


# ============================================================================
# 2. gloo, 2 ranks: uneven rows
# ============================================================================


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _operator_worker(
    rank: int, world_size: int, start_rows: tuple[int, ...], init_file: str, out_file: str
) -> None:
    """Every rank's complex operator output equals the full-space matvec slice."""
    _init_gloo(rank, world_size, init_file)
    try:
        hii, adets_t, bdets_t, integrals, helpers, x = _setup()
        strings = adets_t.tolist()
        M_a, M_b = len(adets_t), len(bdets_t)
        partition = partition_rows_explicit(M_a, M_b, list(start_rows), world_size)
        expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)

        for load_local in (False, True):
            hs = _local_helpers(strings, row_start, row_stop) if load_local else helpers
            op = dmat.LocalTPBOperator(
                hii[sl],
                adets_t,
                bdets_t,
                NORB,
                integrals,
                hs,
                partition,
                rank=rank,
                world_size=world_size,
            )
            got = op(x[sl])
            assert got.dtype == torch.complex128, (rank, load_local)
            assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
                rank,
                load_local,
                float((got - expected[sl]).abs().max()),
            )

        if rank == 0:
            from pathlib import Path

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


def test_operator_complex_gloo_two_ranks_uneven(tmp_path) -> None:
    """gloo, 2 ranks, 2/4 complex rows: matches the full matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_operator_worker, 2, ((0, 2, 6),), tmp_path, "cop2")


# ============================================================================
# 3. gloo, 2 ranks: complex αβ-swap projector
# ============================================================================


def _projector_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """The local projector matches ``(v + tau vᵀ)/2`` on each owned block."""
    from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

    _init_gloo(rank, world_size, init_file)
    try:
        _, adets_t, bdets_t, _, _, v = _setup()
        M_a, M_b = len(adets_t), len(bdets_t)
        partition = partition_rows_explicit(M_a, M_b, [0, 2, 6], world_size)

        sigma = _inter_to_block_signs(adets_t, bdets_t, NORB)
        assert not sigma.dtype.is_complex
        tau = sigma * sigma.T
        v2d = v.view(M_a, M_b)
        expected = ((v2d + tau * v2d.T) * 0.5).reshape(-1)

        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)
        # Constructor default (real) dtype with a complex input exercises the
        # buffer reallocation; an explicit complex dtype exercises the initial
        # complex staging buffers.
        for init_dtype in (None, torch.complex128):
            kwargs = {} if init_dtype is None else {"dtype": init_dtype}
            projector = dmat.LocalAlphaBetaProjector(
                adets_t, bdets_t, NORB, partition, rank=rank, world_size=world_size, **kwargs
            )
            got = projector(v[sl])
            assert got.dtype == torch.complex128, (rank, init_dtype)
            assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
                rank,
                init_dtype,
                float((got - expected[sl]).abs().max()),
            )

        if rank == 0:
            from pathlib import Path

            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_local_projector_complex_matches_full_space(tmp_path) -> None:
    """gloo, 2 ranks: local projector equals the full-space complex formula."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 2, (), tmp_path, "cproj")


# ============================================================================
# 4. β-pass cache parity on complex data (CPU, bit-for-bit)
# ============================================================================


@pytest.mark.parametrize("chunk_size", [2_000_000, 5, 1])
def test_beta_pass_cache_complex_bit_identical(chunk_size: int) -> None:
    """``_mult_beta_only_torch(cache=...)`` is bit-identical to ``cache=None``."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    h_beta = next(h for h in helpers if h.task_type == 1)
    cache = build_beta_pass_cache(adets_t, bdets_t, NORB, integrals, helpers)

    uncached = hii * x
    _mult_beta_only_torch(
        uncached, x, adets_t, bdets_t, NORB, integrals, h_beta, M_a, M_b, chunk_size
    )
    cached = hii * x
    _mult_beta_only_torch(
        cached,
        x,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        h_beta,
        M_a,
        M_b,
        chunk_size,
        cache=cache,
    )
    assert uncached.dtype == torch.complex128
    assert torch.equal(uncached, cached)


# ============================================================================
# 5. Triton is real-only
# ============================================================================


def test_operator_triton_complex_requires_cuda() -> None:
    """Complex integrals pass the constructor; a Triton launch still needs CUDA."""
    hii, adets_t, bdets_t, integrals, helpers, _x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    with pytest.raises(RuntimeError, match="CUDA device"):
        dmat.LocalTPBOperator(
            hii,
            adets_t,
            bdets_t,
            NORB,
            integrals,
            helpers,
            partition,
            rank=0,
            world_size=1,
            matvec_impl="triton",
        )
