"""CPU (gloo) checks for the local-block distributed Davidson API.

Covers the new dav-core surface of
``torch_sci_solver.torch_impl.distributed_davidson``: explicit/cost-balanced
row partitions, the staged cyclic row exchange, the partition-invariant seed
hash, the replicated global diagonal minima, and
:func:`davidson_distributed_local` against the exact dense spectrum.

No GPU and no solver path: two or three spawned ranks partition a synthetic
symmetric matrix and compare with ``torch.linalg.eigvalsh``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl.distributed_davidson import (
    DistributedContext,
    StagedRowExchange,
    davidson_distributed_local,
    global_smallest_hii,
    hash_uniform,
    partition_rows_explicit,
    partition_rows_weighted,
    partition_vector,
    partition_vector_weighted,
)


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the spawned workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _run_spawned(fn, world_size: int, tmp_path: Path, name: str) -> None:
    """Spawn ``world_size`` ranks and assert the run wrote its OK marker."""
    init_file = tmp_path / f"{name}_init"
    out_file = tmp_path / f"{name}_out"
    mp.start_processes(
        fn,
        args=(world_size, str(init_file), str(out_file)),
        nprocs=world_size,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


# --------------------------------------------------------------------------- #
# (a) RowPartition: coverage, caps, owner_rows -- pure CPU, no process group  #
# --------------------------------------------------------------------------- #
def test_row_partition_explicit_covers_uneven_rows() -> None:
    """Explicit boundaries map rows to owners and back on uneven splits."""
    n_rows, row_size = 7, 3
    partition = partition_rows_explicit(n_rows, row_size, [0, 2, 5, 7], 3)
    assert partition.world_size == 3
    assert partition.n == n_rows * row_size
    assert partition.max_local == 3 * row_size
    assert [partition.rows_for_rank(rank) for rank in range(3)] == [(0, 2), (2, 5), (5, 7)]
    assert [partition.for_rank(rank).local_size for rank in range(3)] == [
        2 * row_size,
        3 * row_size,
        2 * row_size,
    ]

    owners = partition.owner_rows(torch.arange(n_rows))
    assert owners.tolist() == [0, 0, 1, 1, 1, 2, 2]
    for rank in range(3):
        start, stop = partition.rows_for_rank(rank)
        assert partition.global_row(rank, torch.arange(stop - start)).tolist() == list(
            range(start, stop)
        )
    assert partition.owner_rows(torch.empty(0, dtype=torch.int64)).numel() == 0

    # Empty ranks are allowed and never claim a row.
    sparse = partition_rows_explicit(2, 1, [0, 0, 2, 2], 3)
    assert sparse.rows_for_rank(0) == (0, 0)
    assert sparse.owner_rows(torch.tensor([0, 1])).tolist() == [1, 1]
    with pytest.raises(ValueError):
        partition_rows_explicit(5, 1, [0, 3, 4], 2)  # does not cover [0, 5)
    with pytest.raises(ValueError):
        partition_rows_explicit(5, 1, [0, 4, 3, 5], 3)  # not monotone


def test_row_partition_weighted_balances_cost_with_cap() -> None:
    """Weighted cuts follow cost, stay contiguous, deterministic and capped."""
    n_rows, row_size, world_size = 12, 5, 3
    weights = torch.tensor([1.0, 1.0, 1.0, 8.0, 8.0, 1.0, 1.0, 1.0, 8.0, 8.0, 1.0, 1.0])
    partition = partition_rows_weighted(n_rows, row_size, weights, world_size)
    again = partition_rows_weighted(n_rows, row_size, weights, world_size)
    assert partition == again
    assert partition.n == n_rows * row_size

    costs = []
    bounds = [partition.rows_for_rank(rank) for rank in range(world_size)]
    assert bounds[0][0] == 0 and bounds[-1][1] == n_rows
    for rank, (start, stop) in enumerate(bounds):
        assert start == (bounds[rank - 1][1] if rank else 0)
        costs.append(float(weights[start:stop].sum()))
    cap = math.ceil(1.25 * n_rows / world_size)
    assert all(stop - start <= cap for start, stop in bounds)
    # Cost balance beats a count split on this distribution.
    assert max(costs) - min(costs) < 8.0

    # All-zero costs still yield a valid covering.
    zero = partition_rows_weighted(n_rows, row_size, torch.zeros(n_rows), world_size)
    assert zero.rows_for_rank(world_size - 1)[1] == n_rows
    with pytest.raises(ValueError):
        partition_rows_weighted(n_rows, row_size, -torch.ones(n_rows), world_size)
    with pytest.raises(ValueError):
        partition_rows_weighted(n_rows, row_size, weights[:-1], world_size)


def test_vector_partition_weighted_balances_flat_cost() -> None:
    """Weighted flat cuts follow cost, stay contiguous and deterministic."""
    n, world_size = 10, 3
    weights = torch.tensor([10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0])
    partition = partition_vector_weighted(n, world_size, weights)
    assert partition == partition_vector_weighted(n, world_size, weights)
    assert partition.n == n
    assert partition.max_local == max(bound.local_size for bound in partition.bounds)

    costs = []
    previous_stop = 0
    for rank, bound in enumerate(partition.bounds):
        assert bound.start == previous_stop
        assert bound.stop >= bound.start
        previous_stop = bound.stop
        assert partition.for_rank(rank) == bound
        costs.append(float(weights[bound.start : bound.stop].sum()))
    assert previous_stop == n
    assert max(costs) - min(costs) <= 12.0
    assert [bound.local_size for bound in partition.bounds] != [4, 3, 3]

    # Zero cost falls back to the canonical equal split.
    zero = partition_vector_weighted(n, world_size, torch.zeros(n))
    assert [bound.local_size for bound in zero.bounds] == [4, 3, 3]
    with pytest.raises(ValueError):
        partition_vector_weighted(n, world_size, -torch.ones(n))
    with pytest.raises(ValueError):
        partition_vector_weighted(n, world_size, weights[:-1])


def test_local_davidson_rejects_small_bases() -> None:
    """Dense-fitting bases and an undersized block are rejected up front."""
    if dist.is_available() and dist.is_initialized():
        pytest.skip("process group already initialized in this pytest process")
    n = 6
    bounds = partition_vector(n, 1, 0)
    context = DistributedContext(rank=0, world_size=1, backend="gloo", device=torch.device("cpu"))
    hii = torch.arange(n, dtype=torch.float64)
    W = torch.ones(n, dtype=torch.float64)
    matvec = lambda x: x  # noqa: E731 - trivial local operator for validation only
    with pytest.raises(ValueError, match="replicated dense path"):
        davidson_distributed_local(
            hii, W, matvec, bounds, n=n, context=context, max_iters=10, block_size=n, eps=1e-8
        )
    big = 20
    big_bounds = partition_vector(big, 1, 0)
    with pytest.raises(ValueError, match="block_size"):
        davidson_distributed_local(
            torch.arange(big, dtype=torch.float64),
            torch.ones(big, dtype=torch.float64),
            matvec,
            big_bounds,
            n=big,
            context=context,
            max_iters=10,
            block_size=2,
            eps=1e-8,
            n_roots=2,
        )


def test_staged_exchange_world_size_one_without_process_group() -> None:
    """A single-rank exchange yields just the local block, no process group."""
    if dist.is_available() and dist.is_initialized():
        pytest.skip("process group already initialized in this pytest process")
    partition = partition_rows_explicit(3, 2, [0, 3], 1)
    exchange = StagedRowExchange(partition, torch.empty(0, dtype=torch.int64))
    assert exchange.world_size == 1
    assert exchange.n_stages == 1
    x = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    stages = list(exchange.stages(x))
    assert len(stages) == 1
    assert stages[0][0] == 0
    assert torch.equal(stages[0][1], x)
    assert exchange.rows(0).tolist() == [0, 1, 2]


# --------------------------------------------------------------------------- #
# (b) StagedRowExchange under gloo                                             #
# --------------------------------------------------------------------------- #
def _staged_exchange_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Each stage holds exactly the peer rows this rank needs, sorted."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n_rows, row_size = 11, 4
        starts = [rows * n_rows // world_size for rows in range(world_size)] + [n_rows]
        partition = partition_rows_explicit(n_rows, row_size, starts, world_size)
        start, stop = partition.rows_for_rank(rank)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        full = torch.randn(n_rows, row_size, generator=generator, dtype=torch.float64)
        x_local = full[start:stop].clone()

        needed = [
            row
            for row in range(n_rows)
            if not start <= row < stop and ((row * 7 + rank * 11) % 4 != 0)
        ]
        # Always exercise the two rows just across each boundary.
        if start > 0:
            needed.append(start - 1)
        if stop < n_rows:
            needed.append(stop)
        needed = sorted(set(needed))

        exchange = StagedRowExchange(
            partition,
            torch.tensor(needed, dtype=torch.int64),
            rank=rank,
            world_size=world_size,
        )
        assert exchange.n_stages == world_size
        seen: set[int] = set()
        for stage, buffer in exchange.stages(x_local):
            rows = exchange.rows(stage)
            if stage == 0:
                assert torch.equal(buffer, x_local), rank
                assert rows.tolist() == list(range(start, stop)), rank
                continue
            peer = (rank + stage) % world_size
            peer_start, peer_stop = partition.rows_for_rank(peer)
            expected = [row for row in needed if peer_start <= row < peer_stop]
            assert rows.tolist() == expected, (rank, stage, rows.tolist(), expected)
            assert rows.tolist() == sorted(rows.tolist()), (rank, stage)
            assert torch.equal(buffer, full[rows]), (rank, stage)
            seen.update(rows.tolist())
        assert seen == set(needed), (rank, seen, set(needed))
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_staged_row_exchange_two_ranks(tmp_path: Path) -> None:
    """World-size-2 exchange reproduces the peer rows per stage."""
    _skip_without_gloo()
    _run_spawned(_staged_exchange_worker, 2, tmp_path, "staged2")


def test_staged_row_exchange_three_ranks(tmp_path: Path) -> None:
    """World-size-3 exchange follows the cyclic (rank + s) % P pattern."""
    _skip_without_gloo()
    _run_spawned(_staged_exchange_worker, 3, tmp_path, "staged3")


# --------------------------------------------------------------------------- #
# (c) davidson_distributed_local against the exact spectrum                   #
# --------------------------------------------------------------------------- #
def _synthetic_hamiltonian(n: int) -> torch.Tensor:
    """Symmetric diagonal + couplings, several crossing the rank boundary."""
    hii = torch.linspace(0.1, 1.0, n, dtype=torch.float64) ** 2
    hamiltonian = torch.diag(hii)
    off = torch.arange(n - 1)
    hamiltonian[off, off + 1] = 0.25
    hamiltonian[off + 1, off] = 0.25
    for left, right, value in ((1, 15, 0.19), (4, 11, -0.23), (2, 17, 0.17), (7, 12, 0.21)):
        hamiltonian[left, right] = value
        hamiltonian[right, left] = value
    return hamiltonian


def _local_dense_matvec(hamiltonian: torch.Tensor, bounds, world_size: int, dtype):
    """Local block of ``H @ x``; gathers the small input to apply the dense H."""
    local_size = bounds.local_size

    def matvec(x_local: torch.Tensor) -> torch.Tensor:
        parts = [
            torch.empty(local_size, dtype=x_local.dtype, device=x_local.device)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, x_local.contiguous())
        x_full = torch.cat(parts)
        y = hamiltonian[bounds.start : bounds.stop] @ x_full.to(torch.float64)
        return y.to(dtype)

    return matvec


def _local_davidson_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Compare the local-block solver with the dense spectrum on every rank."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n = 20
        hamiltonian = _synthetic_hamiltonian(n)
        exact = torch.linalg.eigvalsh(hamiltonian)
        bounds = partition_vector(n, world_size, rank)
        context = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
        )
        diag = torch.diagonal(hamiltonian).clone()
        hii_local = diag[bounds.start : bounds.stop].contiguous()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(23)
        # A proper global start vector, sliced like the solver's warm start:
        # per-rank independent noise would make a periodic global seed.
        W_full = torch.randn(n, generator=generator, dtype=torch.float64)
        W_local = W_full[bounds.start : bounds.stop].clone()
        matvec = _local_dense_matvec(hamiltonian, bounds, world_size, torch.float64)

        for n_roots, blocks in ((1, (3, 4, 5, 6)), (2, (3, 4, 5, 6))):
            for block_size in blocks:
                energies, vectors, info = davidson_distributed_local(
                    hii_local,
                    W_local,
                    matvec,
                    bounds,
                    n=n,
                    context=context,
                    max_iters=2000,
                    block_size=block_size,
                    eps=1e-10,
                    n_roots=n_roots,
                )
                if n_roots == 1:
                    assert isinstance(energies, float), rank
                    assert vectors.shape == (bounds.local_size,), rank
                    got = torch.tensor([energies], dtype=torch.float64)
                else:
                    assert energies.shape == (n_roots,), rank
                    assert vectors.shape == (n_roots, bounds.local_size), rank
                    got = energies
                assert torch.allclose(got, exact[:n_roots], atol=1e-8), (
                    rank,
                    n_roots,
                    block_size,
                    got.tolist(),
                    exact[:n_roots].tolist(),
                )
                assert info["converged"], (rank, n_roots, block_size)
                assert info["distributed"] is True

                # The returned local blocks assemble into an orthonormal set.
                gram = vectors.reshape(n_roots, -1) @ vectors.reshape(n_roots, -1).T
                dist.all_reduce(gram, op=dist.ReduceOp.SUM)
                assert torch.allclose(gram, torch.eye(n_roots, dtype=gram.dtype), atol=1e-8), (
                    rank,
                    n_roots,
                    block_size,
                    gram,
                )

        # One fp32-subspace run: matvec computes in fp64, returns fp32.
        matvec32 = _local_dense_matvec(hamiltonian, bounds, world_size, torch.float32)
        energy32, vector32, info32 = davidson_distributed_local(
            hii_local.to(torch.float32),
            W_local.to(torch.float32),
            matvec32,
            bounds,
            n=n,
            context=context,
            max_iters=1000,
            block_size=6,
            eps=1e-6,
            n_roots=1,
            seed_salt=5,
        )
        assert vector32.dtype == torch.float32, rank
        assert abs(energy32 - float(exact[0])) < 1e-4, (rank, energy32, float(exact[0]))
        assert info32["converged"], rank
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_local_davidson_matches_exact_spectrum(tmp_path: Path) -> None:
    """n_roots 1/2, block_size 3..6, fp64 1e-8 and one fp32 run at 1e-4."""
    _skip_without_gloo()
    _run_spawned(_local_davidson_worker, 2, tmp_path, "local_davidson")


def test_local_float32_solve_stops_at_its_precision_floor() -> None:
    """The owner-computes loop stops a sub-float64 solve at its floor.

    World size 1 needs no process group, so this runs in-process. The fp32
    floor (~sqrt(nnz)·‖H‖·eps32) makes ``eps=1e-12`` unreachable; without the
    stop the solve burns every restart. fp64 converges at the same tolerance,
    and a reachable fp32 tolerance converges without the flag.
    """
    n = 400
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    h = 0.5 * (a + a.T)
    hii = torch.diagonal(h).clone()
    w = torch.ones(n, dtype=torch.float64) / n**0.5
    bounds = partition_vector(n, 1, 0)
    context = DistributedContext(rank=0, world_size=1, backend="gloo", device=torch.device("cpu"))

    def solve(dtype: torch.dtype, eps: float):
        return davidson_distributed_local(
            hii.to(dtype),
            w.to(dtype),
            lambda v: (h @ v.to(torch.float64)).to(dtype),
            bounds,
            n=n,
            context=context,
            max_iters=80,
            block_size=12,
            eps=eps,
            n_roots=2,
        )

    _e, _v, info32 = solve(torch.float32, 1e-12)
    assert info32["stagnated"] is True and not info32["converged"]
    assert info32["n_outer"] < 80

    _e, _v, info64 = solve(torch.float64, 1e-12)
    assert info64["converged"] and "stagnated" not in info64

    _e, _v, info_reachable = solve(torch.float32, 1e-3)
    assert info_reachable["converged"] and "stagnated" not in info_reachable


# --------------------------------------------------------------------------- #
# (d) hash determinism and global_smallest_hii                               #
# --------------------------------------------------------------------------- #
def _seed_helpers_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Hash values are partition-invariant; global minima match a full sort."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n = 19
        bounds = partition_vector(n, world_size, rank)
        partition = partition_rows_explicit(
            n,
            1,
            [partition_vector(n, world_size, r).start for r in range(world_size)] + [n],
            world_size,
        )
        full_hash = hash_uniform(torch.arange(n))
        local_hash = hash_uniform(torch.arange(bounds.start, bounds.stop))
        assert torch.equal(local_hash, full_hash[bounds.start : bounds.stop]), rank
        # A different partition of the same indices gives bit-identical values.
        scattered = torch.arange(rank, n, world_size)
        assert torch.equal(hash_uniform(scattered), full_hash[scattered]), rank
        assert not torch.equal(hash_uniform(torch.arange(n), salt=9), full_hash), rank

        generator = torch.Generator(device="cpu")
        generator.manual_seed(31)
        hii = torch.randn(n, generator=generator, dtype=torch.float64)
        hii_local = hii[bounds.start : bounds.stop].contiguous()
        pairs = global_smallest_hii(hii_local, partition, k=5)
        expected = sorted(
            ((row, float(value)) for row, value in enumerate(hii.tolist())),
            key=lambda pair: (pair[1], pair[0]),
        )[:5]
        assert pairs == expected, (rank, pairs, expected)

        # Ties break by global row index, including across the rank boundary:
        # rows 9 (rank 0) and 10 (rank 1) share the two smallest values.
        tied = torch.ones(n, dtype=torch.float64)
        tied[9] = tied[10] = 0.4
        tied_pairs = global_smallest_hii(tied[bounds.start : bounds.stop], partition, k=2)
        assert tied_pairs == [(9, 0.4), (10, 0.4)], (rank, tied_pairs)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_hash_uniform_and_global_smallest_hii(tmp_path: Path) -> None:
    """Seeded values are partition-invariant and the global top-k is exact."""
    _skip_without_gloo()
    _run_spawned(_seed_helpers_worker, 2, tmp_path, "seed_helpers")
