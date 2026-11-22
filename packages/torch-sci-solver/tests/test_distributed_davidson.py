"""World-size-2 CPU (gloo) check for the distributed Davidson primitives.

Runs no GPU and touches no solver path. Two spawned ranks partition a fake
determinant vector, apply a fake local Hamiltonian matvec (diagonal plus a
nearest-neighbour coupling that crosses the rank boundary), form the Davidson
projection with all-rank reductions, and compare the resulting Ritz value with
the same algorithm in one process and with the exact dense eigenvalue.

The point is to prove the partition / lookup / reduction arithmetic against
the single-process path; see ``docs/distributed_state_vector.md`` and
``torch_sci_solver.torch_impl.distributed_davidson``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl.distributed_davidson import (
    DistributedContext,
    DistributedVector,
    GlobalSum,
    davidson_distributed,
    local_matvec,
    owner_of_index,
    partition_rows,
    partition_vector,
    projected_matrix,
    rayleigh_quotient,
)


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _synthetic_h4():
    """Deterministic 4-orbital / 4-electron integrals and full CI strings."""
    norb = 4
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
    strings = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), 2)),
        dtype=np.int64,
    )
    return h1, eri, strings


N_DET = 20
COUPLING = 0.25
WORLD_SIZE = 2


# --------------------------------------------------------------------------- #
# Fake Hamiltonian: diagonal + nearest-neighbour coupling (matrix-free)       #
# --------------------------------------------------------------------------- #
def _hii(n: int = N_DET) -> torch.Tensor:
    """Deterministic diagonal with a spread, so the ground state is unique."""
    return torch.linspace(0.1, 1.0, n, dtype=torch.float64) ** 2


def _dense_hamiltonian(n: int = N_DET) -> torch.Tensor:
    """Dense reference ``H = diag(hii) + coupling * nearest-neighbour``."""
    h = torch.diag(_hii(n))
    off = torch.arange(n - 1)
    h[off, off + 1] = COUPLING
    h[off + 1, off] = COUPLING
    return h


def _coupled_local_apply(x_local, x_full, bounds) -> torch.Tensor:
    """Off-diagonal part of ``H @ x`` for this rank's output block.

    Reads ``x_full`` at the two neighbours of every owned index, including the
    neighbours owned by the other rank — this is what exercises the remote
    lookup. The diagonal is handled by ``local_matvec``'s ``hii`` term.
    """
    if bounds.local_size == 0:
        return torch.zeros_like(x_local)
    index = torch.arange(bounds.start, bounds.stop)
    left = torch.zeros_like(x_local)
    right = torch.zeros_like(x_local)
    if bounds.local_size > 1:
        left[1:] = x_full[index[1:] - 1]
        right[:-1] = x_full[index[:-1] + 1]
    if bounds.start > 0:
        left[0] = x_full[bounds.start - 1]
    if bounds.stop < bounds.n:
        right[-1] = x_full[bounds.stop]
    return COUPLING * (left + right)


# --------------------------------------------------------------------------- #
# Child: one gloo rank                                                         #
# --------------------------------------------------------------------------- #
def _distributed_solve(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Run one rank of the two-rank check; rank 0 writes the results as JSON."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.set_num_threads(1)
        bounds = partition_vector(N_DET, world_size, rank)
        hii = _hii()
        hii_local = hii[bounds.start : bounds.stop].clone()

        # 1) One distributed matvec against the dense reference.
        x_global = torch.linspace(-1.0, 1.0, N_DET, dtype=torch.float64)
        x = DistributedVector.from_global(x_global)
        y = local_matvec(x, hii_local, _coupled_local_apply)
        matvec_err = float((y.to_global() - _dense_hamiltonian() @ x_global).abs().max())

        # 2) Rayleigh quotient (two more all-rank reductions).
        rq = rayleigh_quotient(x, y)
        rq_exact = float(x_global @ (_dense_hamiltonian() @ x_global) / (x_global @ x_global))

        # 2b) Complex path: exercises the (real, imag) all-gather and
        #     all-reduce, which is the spinless layout's dtype.
        x_complex = x_global.to(torch.complex128) * (1.0 + 0.5j)
        hii_complex = hii_local.to(torch.complex128)
        y_complex = local_matvec(
            DistributedVector.from_global(x_complex), hii_complex, _coupled_local_apply
        )
        complex_err = float(
            (y_complex.to_global() - _dense_hamiltonian().to(torch.complex128) @ x_complex)
            .abs()
            .max()
        )
        complex_proj = projected_matrix(
            [DistributedVector.from_global(x_complex)],
            [y_complex],
        )
        complex_proj_err = abs(
            complex_proj[0, 0].item() - torch.vdot(x_complex, y_complex.to_global()).item()
        )

        # 3) Davidson-style subspace: two-pass CGS + distributed projection.
        #    Every rank sees the same reduced values, so the control flow is
        #    rank-invariant (a hard requirement for collectives to line up).
        v = DistributedVector.from_global(torch.ones(N_DET, dtype=torch.float64) / N_DET**0.5)
        basis: list[DistributedVector] = []
        products: list[DistributedVector] = []
        energy = float("nan")
        for _ in range(N_DET):
            for _pass in range(2):
                for previous in basis:
                    v = v.add_scaled(previous, -previous.dot(v))
            norm = v.norm()
            if float(norm) < 1e-12:
                break
            v = v.scale(1.0 / norm)
            basis.append(v)
            product = local_matvec(v, hii_local, _coupled_local_apply)
            products.append(product)
            h_proj = projected_matrix(basis, products)
            energy = float(torch.linalg.eigh(h_proj)[0][0])
            v = product.clone()

        if rank == 0:
            Path(out_file).write_text(
                json.dumps({
                    "energy": energy,
                    "matvec_err": matvec_err,
                    "complex_err": complex_err,
                    "complex_proj_err": complex_proj_err,
                    "rayleigh": float(rq),
                    "rayleigh_exact": rq_exact,
                    "n_basis": len(basis),
                })
            )
    finally:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Single-process reference (same algorithm, plain tensors)                    #
# --------------------------------------------------------------------------- #
def _single_process_solve() -> float:
    """Same Davidson-style subspace algorithm on one process; returns the Ritz value."""
    h = _dense_hamiltonian()
    v = torch.ones(N_DET, dtype=torch.float64) / N_DET**0.5
    basis: list[torch.Tensor] = []
    products: list[torch.Tensor] = []
    energy = float("nan")
    for _ in range(N_DET):
        for _pass in range(2):
            for previous in basis:
                v = v - torch.dot(previous, v) * previous
        norm = v.norm()
        if float(norm) < 1e-12:
            break
        v = v / norm
        basis.append(v)
        product = h @ v
        products.append(product)
        energy = float(torch.linalg.eigvalsh(torch.stack(basis) @ torch.stack(products).T)[0])
        v = product.clone()
    return energy


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_partition_math() -> None:
    """Blocks are contiguous, balanced, and owner_of_index inverts them."""
    for n in (0, 1, 3, 5, 7, 16):
        for world_size in (1, 2, 3, 4):
            bounds = [partition_vector(n, world_size, r) for r in range(world_size)]
            assert [b.start for b in bounds] == [0] + [b.stop for b in bounds[:-1]]
            assert bounds[-1].stop == n
            sizes = [b.local_size for b in bounds]
            assert max(sizes) - min(sizes) <= 1
            for r, block in enumerate(bounds):
                for k in range(block.start, block.stop):
                    assert int(owner_of_index(k, n, world_size)) == r
            if n:
                index = torch.arange(n)
                owners = owner_of_index(index, n, world_size)
                assert all(bounds[int(r)].owns(int(k)) for k, r in zip(index, owners))


def test_single_process_fallback() -> None:
    """With no process group every collective is a local no-op."""
    if dist.is_available() and dist.is_initialized():
        pytest.skip("process group already initialized in this pytest process")
    n = 6
    local = torch.arange(n, dtype=torch.float64)
    vector = DistributedVector(local, n)
    assert vector.world_size == 1
    assert torch.equal(vector.to_global(), local)
    assert torch.equal(vector.dot(vector), torch.vdot(local, local))
    assert float(vector.norm()) == pytest.approx(float(local.norm()), rel=1e-15)

    hii = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    product = local_matvec(vector, hii)
    assert torch.equal(product.local, hii * local)

    other = DistributedVector(local + 1.0, n)
    projection = projected_matrix([vector], [other])
    assert projection.shape == (1, 1)
    assert float(projection[0, 0]) == pytest.approx(float(local @ (local + 1.0)), rel=1e-15)

    # GlobalSum is inert without a process group.
    assert isinstance(GlobalSum(), GlobalSum)


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed with the gloo backend is required",
)
def test_two_rank_davidson_matches_single_process(tmp_path: Path) -> None:
    """World-size-2 gloo run reproduces the single-process Ritz value."""
    init_file = tmp_path / "gloo_init"
    out_file = tmp_path / "rank0.json"
    mp.start_processes(
        _distributed_solve,
        args=(WORLD_SIZE, str(init_file), str(out_file)),
        nprocs=WORLD_SIZE,
        start_method="spawn",
        join=True,
    )
    payload = json.loads(out_file.read_text())

    reference = _single_process_solve()
    exact = float(torch.linalg.eigvalsh(_dense_hamiltonian())[0])

    assert payload["n_basis"] == N_DET
    assert payload["matvec_err"] < 1e-13
    assert payload["complex_err"] < 1e-13
    assert payload["complex_proj_err"] < 1e-13
    assert payload["rayleigh"] == pytest.approx(payload["rayleigh_exact"], abs=1e-13)
    assert payload["energy"] == pytest.approx(reference, abs=1e-10)
    assert payload["energy"] == pytest.approx(exact, abs=1e-9)


# --------------------------------------------------------------------------- #
# Regression tests for uneven gathers, owner-computes kernels, and restart
# --------------------------------------------------------------------------- #
def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the extra workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _gather_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank reconstructs the same uneven global vectors."""
    _init_gloo(rank, world_size, init_file)
    try:
        for n in (7, 10, 11, 100):
            expected = torch.arange(n, dtype=torch.float64)
            vector = DistributedVector.from_global(expected)
            assert torch.equal(vector.to_global(), expected), (rank, n)
        for n_rows, row_size in ((5, 3), (7, 2), (9, 1), (5, 1)):
            expected = torch.arange(n_rows * row_size, dtype=torch.float64)
            bounds = partition_rows(n_rows, row_size, world_size, rank)
            vector = DistributedVector.from_global(expected, bounds=bounds)
            assert torch.equal(vector.to_global(), expected), (rank, n_rows, row_size)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def _davidson_blocks_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Small block sizes converge to davidson_torch's eigenvalues."""
    _init_gloo(rank, world_size, init_file)
    try:
        n = 24
        gen = torch.Generator(device="cpu")
        gen.manual_seed(11)
        a = torch.randn(n, n, generator=gen, dtype=torch.float64)
        h = 0.5 * (a + a.T)
        hii = torch.diagonal(h).clone()
        w = torch.ones(n, dtype=torch.float64) / n**0.5
        context = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
        )
        # ``davidson_torch`` itself can miss the ground state at block_size=2,
        # so the tiny-block distributed runs are checked against the exact
        # spectrum (the property that matters), not against the same-block
        # reference solve.
        exact = torch.linalg.eigvalsh(h)[:2]
        for n_roots, blocks in ((1, (2, 3, 4)), (2, (3, 4, 5))):
            for block in blocks:
                dist_e, _dist_v, _ = davidson_distributed(
                    hii,
                    w,
                    lambda v: h @ v,
                    1000,
                    block,
                    1e-8,
                    n_roots=n_roots,
                    context=context,
                )
                got = dist_e if n_roots > 1 else torch.tensor([dist_e], dtype=torch.float64)
                assert torch.allclose(got, exact[:n_roots], atol=1e-6), (
                    rank,
                    n_roots,
                    block,
                    got.tolist(),
                    exact[:n_roots].tolist(),
                )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


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


def test_to_global_compacts_uneven_blocks(tmp_path: Path) -> None:
    """n not divisible by world_size must not shift or drop entries."""
    _skip_without_gloo()
    _run_spawned(_gather_worker, 4, tmp_path, "gather")


def test_small_block_davidson_matches_reference(tmp_path: Path) -> None:
    """block_size 2/3/4 (n_roots 1) and 3/4/5 (n_roots 2) converge."""
    _skip_without_gloo()
    _run_spawned(_davidson_blocks_worker, 2, tmp_path, "blocks")


def test_float32_solve_stops_at_its_precision_floor() -> None:
    """The replicated Krylov loop stops a float32 solve at its floor.

    Calls ``_davidson_distributed_krylov`` directly: ``davidson_distributed``
    delegates to ``davidson_torch`` at ``world_size <= 1``, which would test
    the single-process loop instead of this one. ``eps=1e-10`` is reachable in
    fp64 for this system but far below the fp32 floor
    (~sqrt(nnz)·‖H‖·eps32), so only the float32 solve stops early; a reachable
    float32 tolerance converges without the flag. World size 1 needs no process
    group, so this runs in-process.
    """
    from torch_sci_solver.torch_impl.distributed_davidson import (
        _davidson_distributed_krylov,
    )

    n = 400
    generator = torch.Generator(device="cpu")
    generator.manual_seed(5)
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    h = 0.5 * (a + a.T)
    hii = torch.diagonal(h).clone()
    w = torch.ones(n, dtype=torch.float64) / n**0.5
    context = DistributedContext(rank=0, world_size=1, backend="gloo", device=torch.device("cpu"))

    def solve(dtype: torch.dtype, eps: float) -> dict:
        _e, _v, info = _davidson_distributed_krylov(
            hii.to(dtype),
            w.to(dtype),
            lambda v: (h @ v.to(torch.float64)).to(dtype),
            200,
            12,
            eps,
            n_roots=2,
            de_tol=None,
            project_fn=None,
            context=context,
            verbose=False,
        )
        return info

    info32 = solve(torch.float32, 1e-10)
    assert info32["stagnated"] is True and not info32["converged"]
    assert info32["n_outer"] < 200

    info64 = solve(torch.float64, 1e-10)
    assert info64["converged"] and "stagnated" not in info64

    info_reachable = solve(torch.float32, 1e-3)
    assert info_reachable["converged"] and "stagnated" not in info_reachable
