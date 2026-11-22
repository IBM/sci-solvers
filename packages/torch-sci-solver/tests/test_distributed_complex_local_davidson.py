"""CPU (gloo) checks for complex local-block distributed Davidson support.

Covers the complex part of ``torch_sci_solver.torch_impl.distributed_davidson``:
the Hermitian-correct projections/CGS2/residuals of ``_solve_local_core``, the
``vdot``-based seed and missed-root deflation, and the real-view packing of
complex :class:`StagedRowExchange` payloads.

Two spawned gloo ranks partition a synthetic complex Hermitian matrix and
compare :func:`davidson_distributed_local` with ``torch.linalg.eigvalsh``.
No GPU and no solver path is involved.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl.distributed_davidson import (
    DistributedContext,
    StagedRowExchange,
    davidson_distributed_local,
    partition_rows_explicit,
    partition_vector,
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
# Fixtures: complex Hermitian matrix and its local-block matvec               #
# --------------------------------------------------------------------------- #
def _synthetic_complex_hamiltonian(n: int) -> torch.Tensor:
    """Hermitian complex ``H``: real diagonal, phase-carrying couplings.

    Same diagonal and coupling pattern as the real synthetic matrix in
    ``test_distributed_local_davidson.py``; the nearest-neighbour entries are
    ``+0.25j`` / ``-0.25j`` instead of real, so a projection that forgets the
    conjugate would return wrong energies.
    """
    hii = torch.linspace(0.1, 1.0, n, dtype=torch.float64) ** 2
    hamiltonian = torch.diag(hii).to(torch.complex128)
    off = torch.arange(n - 1)
    hamiltonian[off, off + 1] = 0.25j
    hamiltonian[off + 1, off] = -0.25j
    for left, right, value in (
        (1, 15, 0.19j),
        (4, 11, -0.23),
        (2, 17, 0.17j),
        (7, 12, 0.21),
    ):
        hamiltonian[left, right] = complex(value)
        hamiltonian[right, left] = complex(value).conjugate()
    return hamiltonian


def _local_dense_matvec(hamiltonian: torch.Tensor, bounds, world_size: int):
    """Local block of ``H @ x``; gathers the small input to apply the dense H."""
    local_size = bounds.local_size

    def matvec(x_local: torch.Tensor) -> torch.Tensor:
        parts = [
            torch.empty(local_size, dtype=x_local.dtype, device=x_local.device)
            for _ in range(world_size)
        ]
        dist.all_gather(parts, x_local.contiguous())
        x_full = torch.cat(parts)
        return hamiltonian[bounds.start : bounds.stop] @ x_full

    return matvec


# --------------------------------------------------------------------------- #
# (a) complex davidson_distributed_local against the exact spectrum           #
# --------------------------------------------------------------------------- #
def _complex_local_davidson_worker(
    rank: int, world_size: int, init_file: str, out_file: str
) -> None:
    """Compare complex local energies with the dense spectrum on every rank."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n = 20
        hamiltonian = _synthetic_complex_hamiltonian(n)
        assert torch.allclose(hamiltonian, hamiltonian.mH), rank
        diagonal = torch.diagonal(hamiltonian)
        assert torch.equal(diagonal.imag, torch.zeros(n, dtype=torch.float64)), rank
        exact = torch.linalg.eigvalsh(hamiltonian)
        bounds = partition_vector(n, world_size, rank)
        context = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
        )
        # The diagonal of a Hermitian H is real; keep hii real as the solver's
        # preconditioner contract allows (hii.dtype == W.dtype.to_real()).
        hii_local = diagonal.real[bounds.start : bounds.stop].clone()
        assert hii_local.dtype == torch.float64, rank

        generator = torch.Generator(device="cpu")
        generator.manual_seed(23)
        W_full = torch.complex(
            torch.randn(n, generator=generator, dtype=torch.float64),
            torch.randn(n, generator=generator, dtype=torch.float64),
        )
        W_local = W_full[bounds.start : bounds.stop].clone()
        assert W_local.dtype == torch.complex128, rank
        matvec = _local_dense_matvec(hamiltonian, bounds, world_size)

        for n_roots in (1, 2):
            for block_size in (3, 4, 5, 6):
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
                    assert vectors.dtype == torch.complex128, rank
                    got = torch.tensor([energies], dtype=torch.float64)
                else:
                    assert energies.shape == (n_roots,), rank
                    assert vectors.shape == (n_roots, bounds.local_size), rank
                    assert vectors.dtype == torch.complex128, rank
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

                # The local blocks assemble into a Hermitian-orthonormal set.
                rows = vectors.reshape(n_roots, -1)
                gram = rows @ rows.mH
                dist.all_reduce(gram, op=dist.ReduceOp.SUM)
                assert torch.allclose(
                    gram, torch.eye(n_roots, dtype=torch.complex128), atol=1e-8
                ), (rank, n_roots, block_size, gram)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_complex_local_davidson_matches_exact_spectrum(tmp_path: Path) -> None:
    """n_roots 1/2 and block_size 3..6 match ``eigvalsh`` at 1e-8 on gloo."""
    _skip_without_gloo()
    _run_spawned(_complex_local_davidson_worker, 2, tmp_path, "complex_local_davidson")


# --------------------------------------------------------------------------- #
# (b) complex StagedRowExchange (real-view packing) under gloo                #
# --------------------------------------------------------------------------- #
def _complex_staged_exchange_worker(
    rank: int, world_size: int, init_file: str, out_file: str
) -> None:
    """Each stage holds exactly the peer rows this rank needs, complex-exact."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n_rows, row_size = 11, 4
        starts = [rows * n_rows // world_size for rows in range(world_size)] + [n_rows]
        partition = partition_rows_explicit(n_rows, row_size, starts, world_size)
        start, stop = partition.rows_for_rank(rank)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        full = torch.complex(
            torch.randn(n_rows, row_size, generator=generator, dtype=torch.float64),
            torch.randn(n_rows, row_size, generator=generator, dtype=torch.float64),
        )
        x_local = full[start:stop].clone()
        assert x_local.dtype == torch.complex128, rank

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
        assert exchange.n_stages == world_size, rank
        seen: set[int] = set()
        for stage, buffer in exchange.stages(x_local):
            rows = exchange.rows(stage)
            if stage == 0:
                assert buffer.dtype == torch.complex128, rank
                assert torch.equal(buffer, x_local), rank
                assert rows.tolist() == list(range(start, stop)), rank
                continue
            peer = (rank + stage) % world_size
            peer_start, peer_stop = partition.rows_for_rank(peer)
            expected = [row for row in needed if peer_start <= row < peer_stop]
            assert rows.tolist() == expected, (rank, stage, rows.tolist(), expected)
            assert buffer.shape == (len(expected), row_size), (rank, stage, buffer.shape)
            assert buffer.dtype == torch.complex128, (rank, stage, buffer.dtype)
            # Exact comparison of both components: the peer's row block must
            # survive the real-view packing bit for bit.
            assert torch.equal(buffer.real, full[rows].real), (rank, stage)
            assert torch.equal(buffer.imag, full[rows].imag), (rank, stage)
            assert torch.equal(buffer, full[rows]), (rank, stage)
            seen.update(rows.tolist())
        assert seen == set(needed), (rank, seen, set(needed))
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_complex_staged_row_exchange_two_ranks(tmp_path: Path) -> None:
    """Complex payloads round-trip through the real-view packed P2P exchange."""
    _skip_without_gloo()
    _run_spawned(_complex_staged_exchange_worker, 2, tmp_path, "complex_staged2")


# --------------------------------------------------------------------------- #
# (c) complex local diagonal projector                                        #
# --------------------------------------------------------------------------- #
def _complex_projected_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """A local diagonal +1/0 projector converges to the projected spectrum."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n = 20
        hamiltonian = _synthetic_complex_hamiltonian(n)
        bounds = partition_vector(n, world_size, rank)
        context = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
        )
        hii_local = torch.diagonal(hamiltonian).real[bounds.start : bounds.stop].clone()

        # P = diag(mask): keep the even determinants, project the odd ones out.
        mask = torch.arange(n) % 2 == 0
        mask_complex = mask.to(torch.complex128)
        projected = mask_complex[:, None] * hamiltonian * mask_complex[None, :]
        # The solver works inside range(P) (its seeds and corrections are all
        # projected), so the reference is P H P restricted to range(P); the
        # zero eigenvalues of the projected-out complement are unreachable.
        exact = torch.linalg.eigvalsh(projected[mask][:, mask])
        mask_local = mask_complex[bounds.start : bounds.stop]

        def project_fn(vector: torch.Tensor) -> torch.Tensor:
            return vector * mask_local

        generator = torch.Generator(device="cpu")
        generator.manual_seed(29)
        W_full = torch.complex(
            torch.randn(n, generator=generator, dtype=torch.float64),
            torch.randn(n, generator=generator, dtype=torch.float64),
        )
        W_local = W_full[bounds.start : bounds.stop].clone()
        matvec = _local_dense_matvec(hamiltonian, bounds, world_size)

        for block_size in (4, 5, 6):
            energy, vector, info = davidson_distributed_local(
                hii_local,
                W_local,
                matvec,
                bounds,
                n=n,
                context=context,
                max_iters=2000,
                block_size=block_size,
                eps=1e-10,
                n_roots=1,
                project_fn=project_fn,
            )
            assert isinstance(energy, float), (rank, block_size)
            assert abs(energy - float(exact[0])) < 1e-8, (
                rank,
                block_size,
                energy,
                float(exact[0]),
            )
            assert info["converged"], (rank, block_size)
            assert vector.dtype == torch.complex128, (rank, block_size)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_complex_local_davidson_with_projector(tmp_path: Path) -> None:
    """A complex local ``project_fn`` converges to the projected spectrum."""
    _skip_without_gloo()
    _run_spawned(_complex_projected_worker, 2, tmp_path, "complex_projected")


# --------------------------------------------------------------------------- #
# (d) complex missed-root search                                              #
# --------------------------------------------------------------------------- #
def _complex_missed_root_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """The deflated complex search finds no root below the returned two."""
    _init_gloo(rank, world_size, init_file)
    try:
        torch.set_num_threads(1)
        n = 20
        hamiltonian = _synthetic_complex_hamiltonian(n)
        exact = torch.linalg.eigvalsh(hamiltonian)
        bounds = partition_vector(n, world_size, rank)
        context = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
        )
        hii_local = torch.diagonal(hamiltonian).real[bounds.start : bounds.stop].clone()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(23)
        W_full = torch.complex(
            torch.randn(n, generator=generator, dtype=torch.float64),
            torch.randn(n, generator=generator, dtype=torch.float64),
        )
        W_local = W_full[bounds.start : bounds.stop].clone()
        matvec = _local_dense_matvec(hamiltonian, bounds, world_size)

        energies, _vectors, info = davidson_distributed_local(
            hii_local,
            W_local,
            matvec,
            bounds,
            n=n,
            context=context,
            max_iters=2000,
            block_size=6,
            eps=1e-10,
            n_roots=2,
            missed_root_check=True,
        )
        assert torch.allclose(energies, exact[:2], atol=1e-8), (
            rank,
            energies.tolist(),
            exact[:2].tolist(),
        )
        assert info["converged"], rank
        checks = info["missed_root_checks"]
        assert len(checks) == 1, (rank, checks)
        # The deflated operator's lowest value is the third eigenvalue, above
        # theta_max: a conjugate-projector bug would report a spurious root.
        assert checks[0]["found"] is False, (rank, checks)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_complex_missed_root_check(tmp_path: Path) -> None:
    """The complex root deflation does not invent a missed root."""
    _skip_without_gloo()
    _run_spawned(_complex_missed_root_worker, 2, tmp_path, "complex_missed_root")
