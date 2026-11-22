"""Tests for the distributed owner-computes TPB operator and αβ-swap projector.

Covers:

1. Single process: ``LocalTPBOperator`` (batched) against ``mult_torch`` for a
   ``world_size=1`` partition, and against a sliced full vector for sub-blocks
   of 1/2/3 rows (a stand-in exchange, so no process group is needed).
2. gloo, 2 and 3 ranks (uneven row counts): every rank's output block matches
   the corresponding slice of the full-space ``mult_torch``.
3. gloo, 2 ranks: ``LocalAlphaBetaProjector`` matches the full-space
   ``(v + τ vᵀ) / 2`` projector.
4. β-pass cache parity: cached and uncached β kernels are bit-identical on CPU.
5. Triton (CUDA only): the operator's ``matvec_impl="triton"`` path matches the
   batched path on one small basis.
6. ``alpha_row_weights``: shape/dtype/formula and a weighted partition check.
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
    partition_rows_explicit,
    partition_rows_weighted,
)
from torch_sci_solver.torch_impl.helpers import (
    generate_all_torch_helpers,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.mult import (
    _mult_beta_only_torch,
    build_beta_pass_cache,
    make_diag_terms_torch,
    mult_torch,
)

NORB = 4
# C(4, 2) = 6 strings per spin -> a 6-row, 36-determinant basis.
M_STRINGS = 6


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _synthetic_h4() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Deterministic 4-orbital / 4-electron integrals and full CI strings."""
    norb = NORB
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
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
    strings = [int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(norb), 2)]
    return h1, eri, sorted(strings)


def _setup() -> tuple:
    """Build the shared small basis: (hii, adets_t, bdets_t, integrals, helpers, x)."""
    h1, eri, strings = _synthetic_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu")
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = torch.tensor(strings, dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(strings, strings, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(len(strings) * len(strings), generator=gen, dtype=torch.float64)
    return hii, adets_t, bdets_t, integrals, helpers, x


def _local_helpers(strings: list[int], row_start: int, row_stop: int) -> list:
    """Local-row helpers for one α-row block."""
    return list(
        generate_local_torch_helpers(strings, strings, NORB, row_start, row_stop, device="cpu")
    )


class _FullVectorExchange:
    """Stand-in for :class:`StagedRowExchange` that slices a full vector.

    Lets the operator own an arbitrary row sub-block without a process group.
    ``stages`` yields stage 0 (the local block) and then, for each remote
    stage, the rows of ``full_vector`` owned by that stage. The global row set
    of a stage is the sorted needed-row intersection with that owner, exactly
    as the real exchange builds its buffers.
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
# 1. Single process
# ============================================================================


def test_operator_single_process_full_helpers() -> None:
    """world_size=1: full helpers reproduce ``mult_torch`` on the whole basis."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)

    op = dmat.LocalTPBOperator(
        hii, adets_t, bdets_t, NORB, integrals, helpers, partition, rank=0, world_size=1
    )
    got = op(x)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0)


def test_operator_single_process_local_helpers() -> None:
    """world_size=1: local helpers (whole block) reproduce ``mult_torch``."""
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    M_a, M_b = len(adets_t), len(bdets_t)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    local = _local_helpers(adets_t.tolist(), 0, M_a)

    op = dmat.LocalTPBOperator(
        hii, adets_t, bdets_t, NORB, integrals, local, partition, rank=0, world_size=1
    )
    got = op(x)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_owned_rows", [1, 2, 3])
@pytest.mark.parametrize("use_local_helpers", [False, True])
def test_operator_subblock_matches_full_slice(
    monkeypatch: pytest.MonkeyPatch, n_owned_rows: int, use_local_helpers: bool
) -> None:
    """A row sub-block (1/2/3 of 6 rows) reproduces the full matvec's slice.

    Stage-0 table slicing is what this exercises; the stand-in exchange slices
    a full vector, so no process group is needed.
    """
    hii, adets_t, bdets_t, integrals, helpers, x = _setup()
    strings = adets_t.tolist()
    M_a, M_b = len(adets_t), len(bdets_t)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)

    partition = partition_rows_explicit(M_a, M_b, [0, n_owned_rows, M_a], 2)
    row_start, row_stop = partition.rows_for_rank(0)
    local = _local_helpers(strings, row_start, row_stop) if use_local_helpers else helpers

    monkeypatch.setattr(dmat, "StagedRowExchange", _FullVectorExchange)
    _FullVectorExchange.full_vector = x
    op = dmat.LocalTPBOperator(
        hii[row_start * M_b : row_stop * M_b],
        adets_t,
        bdets_t,
        NORB,
        integrals,
        local,
        partition,
        rank=0,
        world_size=2,
    )
    got = op(x[row_start * M_b : row_stop * M_b])
    assert torch.allclose(got, expected[row_start * M_b : row_stop * M_b], atol=1e-12, rtol=0.0)


# ============================================================================
# 2. gloo: 2 and 3 ranks, uneven row counts
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
    """Every rank's operator output equals the full-space matvec's slice."""
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
            assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
                rank,
                world_size,
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


def test_operator_gloo_two_ranks_uneven(tmp_path) -> None:
    """gloo, 2 ranks, 2/4 rows: matches the full-vector matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_operator_worker, 2, ((0, 2, 6),), tmp_path, "op2")


def test_operator_gloo_three_ranks_uneven(tmp_path) -> None:
    """gloo, 3 ranks, 1/3/2 rows: matches the full-vector matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_operator_worker, 3, ((0, 1, 4, 6),), tmp_path, "op3")


# ============================================================================
# 3. gloo: local αβ-swap projector
# ============================================================================


def _projector_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """The local projector matches the full-space ``(v + τ vᵀ)/2`` on each block."""
    from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

    _init_gloo(rank, world_size, init_file)
    try:
        _, adets_t, bdets_t, _, _, x = _setup()
        M_a, M_b = len(adets_t), len(bdets_t)
        partition = partition_rows_explicit(M_a, M_b, [0, 2, 6], world_size)

        sigma = _inter_to_block_signs(adets_t, bdets_t, NORB)
        tau = sigma * sigma.T
        v2d = x.view(M_a, M_b)
        expected = ((v2d + tau * v2d.T) * 0.5).reshape(-1)

        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)
        projector = dmat.LocalAlphaBetaProjector(
            adets_t, bdets_t, NORB, partition, rank=rank, world_size=world_size
        )
        got = projector(x[sl])
        assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
            rank,
            float((got - expected[sl]).abs().max()),
        )

        if rank == 0:
            from pathlib import Path

            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_local_projector_matches_full_space(tmp_path) -> None:
    """gloo, 2 ranks: local projector equals the full-space formula."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 2, (), tmp_path, "proj")


# ============================================================================
# 4. β-pass cache parity (CPU, bit-for-bit)
# ============================================================================


@pytest.mark.parametrize("chunk_size", [2_000_000, 5, 1])
def test_beta_pass_cache_bit_identical(chunk_size: int) -> None:
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
    assert torch.equal(uncached, cached)


# ============================================================================
# 5. Triton on the local block (CUDA only)
# ============================================================================


def test_operator_triton_matches_batched_on_cuda() -> None:
    """The operator's Triton path matches the batched path on one small basis."""
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
    x = torch.randn(len(strings) ** 2, generator=gen, dtype=torch.float64, device="cuda")

    M_a, M_b = len(adets_t), len(bdets_t)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    common = dict(partition=partition, rank=0, world_size=1)
    batched = dmat.LocalTPBOperator(hii, adets_t, bdets_t, NORB, integrals, helpers, **common)(x)
    triton = dmat.LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        matvec_impl="triton",
        **common,
    )(x)
    assert torch.allclose(triton, batched, atol=1e-12, rtol=0.0)


# ============================================================================
# 6. Per-row cost weights
# ============================================================================


def _mixed_occupation_strings(n_orb: int = 5) -> list[int]:
    """All strings of 0..n_orb electrons — varied popcounts for the weights."""
    return sorted(
        sum(1 << o for o in occ)
        for n_elec in range(n_orb + 1)
        for occ in itertools.combinations(range(n_orb), n_elec)
    )


def test_alpha_row_weights_formula_and_positive() -> None:
    """Weights are positive fp64 and match the combinatorial cost formula."""
    n_orb = 5
    strings = _mixed_occupation_strings(n_orb)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = torch.tensor(strings[::-1], dtype=torch.int64)

    weights = dmat.alpha_row_weights(adets_t, bdets_t, n_orb)
    assert weights.shape == (len(strings),)
    assert weights.dtype == torch.float64
    assert torch.all(weights > 0.0)
    assert torch.all(torch.isfinite(weights))

    # Independent scalar recomputation of the model for every row.
    M_b = len(strings)
    beta_singles_sum = 0.0
    beta_doubles_sum = 0.0
    for beta in strings[::-1]:
        o_b = bin(beta).count("1")
        v_b = n_orb - o_b
        beta_singles_sum += o_b * v_b
        beta_doubles_sum += (o_b * (o_b - 1) / 2) * (v_b * (v_b - 1) / 2)
    for ia, alpha in enumerate(strings):
        o = bin(alpha).count("1")
        v = n_orb - o
        n_as = o * v
        n_ad = (o * (o - 1) / 2) * (v * (v - 1) / 2)
        expected = (
            M_b * (n_as + n_ad) + n_as * beta_singles_sum + (beta_singles_sum + beta_doubles_sum)
        )
        assert weights[ia].item() == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_alpha_row_weights_weighted_partition_balances() -> None:
    """A weighted partition of the same system is close to cost-balanced."""
    n_orb = 5
    strings = _mixed_occupation_strings(n_orb)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = torch.tensor(strings, dtype=torch.int64)
    weights = dmat.alpha_row_weights(adets_t, bdets_t, n_orb)
    M_b = len(strings)
    world_size = 2

    partition = partition_rows_weighted(len(strings), M_b, weights, world_size)

    costs = torch.zeros(world_size, dtype=torch.float64)
    covered = torch.zeros(len(strings), dtype=torch.int64)
    for rank in range(world_size):
        row_start, row_stop = partition.rows_for_rank(rank)
        costs[rank] = weights[row_start:row_stop].sum()
        covered[row_start:row_stop] += 1
    assert torch.equal(covered, torch.ones_like(covered))
    total = float(weights.sum())
    # Row granularity can leave one heavy row of slack; beyond that the cost
    # of every rank must stay within 1.25x of the balanced share.
    assert float(costs.max()) <= 1.25 * total / world_size + float(weights.max())
