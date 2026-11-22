"""Distributed regressions checked against physical, independent Hamiltonians."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_distributed_solver import _skip_without_gloo, _solver_kwargs
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl.distributed_solver import _rank_device, parse_distributed_config

from sci_solvers.eigen_states import EigenStates


@pytest.fixture(params=["gloo", pytest.param("nccl", marks=pytest.mark.gpu)], ids=["cpu", "gpu"])
def distributed_config(request: pytest.FixtureRequest) -> dict:
    """Run the same physical checks with two CPU ranks or two real CUDA/NCCL ranks."""
    backend = request.param
    if backend == "gloo":
        _skip_without_gloo()
    elif not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and dist.is_available()
        and dist.is_nccl_available()
    ):
        pytest.skip("two CUDA devices and NCCL are required")
    return {"world_size": 2, "backend": backend, "timeout": 300 if backend == "nccl" else 30}


def _hubbard() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Repulsive four-site Hubbard model; PySCF supplies an independent full CI matrix."""
    fci = pytest.importorskip("pyscf.fci.direct_spin1")
    h1 = np.diag([-0.3, 0.1, 0.7, 1.2])
    for i in range(3):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    h1[0, 3] = h1[3, 0] = -0.2
    eri = np.zeros((4,) * 4)
    for i in range(4):
        eri[i, i, i, i] = 2.0
    strings = np.array([3, 5, 6, 9, 10, 12], dtype=np.int64)
    addresses, h = fci.pspace(h1, eri, 4, (2, 2), np=36)
    order = np.argsort(addresses)
    return h1, eri, strings, h[np.ix_(order, order)]


def _assert_eigenpairs(result: EigenStates, h: np.ndarray, n_roots: int) -> None:
    """Check physical eigenpairs without fixing their arbitrary signs or degenerate rotations."""
    assert len(result.states) == n_roots
    energies = np.array([state.energy for state in result.states])
    vectors = np.stack([state.sci_state.amplitudes.ravel() for state in result.states])
    np.testing.assert_allclose(energies, np.linalg.eigvalsh(h)[:n_roots], atol=2e-8, rtol=0)
    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(n_roots), atol=1e-8)
    residuals = np.linalg.norm(vectors @ h.T - energies[:, None] * vectors, axis=1)
    assert np.max(residuals) < 2e-8, residuals
    assert all(state.state == "converged" for state in result.states)


def test_projected_fallback_preserves_physical_singlet(distributed_config: dict) -> None:
    """Two noninteracting electrons have E0=2, not a spurious zero of P H."""
    eps = np.arange(1.0, 5.0)
    strings = np.array([1, 2, 4, 8])
    h = np.diag((eps[:, None] + eps[None, :]).ravel())
    with make_torch_sci_solver(
        **_solver_kwargs(
            n_roots=1,
            apply_ab_projection=True,
            distributed=distributed_config,
        )
    ) as solver:
        result = solver(
            [(strings, strings)],
            np.diag(eps),
            np.zeros((4,) * 4),
            4,
            (1, 1),
            init_vectors=[np.ones(16)],
        )[0]
    _assert_eigenpairs(result, h, 1)
    # In the public alpha/beta spin-block convention, a two-electron singlet has C=C.T.
    amplitudes = result.states[0].sci_state.amplitudes.reshape(4, 4)
    np.testing.assert_allclose(amplitudes, amplitudes.T, atol=1e-8)


@pytest.mark.parametrize("n_alpha_strings", [4, 5])
def test_random_fallback_preserves_row_partition(
    n_alpha_strings: int,
    distributed_config: dict,
) -> None:
    """Even and uneven row partitions solve the same diagonal fermion Hamiltonian."""
    eps = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    strings_a = 1 << np.arange(n_alpha_strings)
    strings_b = 1 << np.arange(3)
    h = np.diag((eps[:n_alpha_strings, None] + eps[None, :3]).ravel())
    with make_torch_sci_solver(
        **_solver_kwargs(
            n_roots=1,
            apply_ab_projection=False,
            distributed=distributed_config,
        )
    ) as solver:
        result = solver(
            [(strings_a, strings_b)],
            np.diag(eps),
            np.zeros((5,) * 4),
            5,
            (1, 1),
            init_vectors=[np.ones(h.shape[0])],
        )[0]
    _assert_eigenpairs(result, h, 1)


def _unequal_memory_worker(rank: int, init_file: str, mode: str, config: dict) -> None:
    """Simulate different free memory, with bounded collective failures instead of hanging CI."""
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if config["backend"] == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.set_debug_level(dist.DebugLevel.DETAIL)
    dist.init_process_group(
        config["backend"],
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=config["timeout"]),
    )
    try:
        h1, eri, strings, h = _hubbard()
        batch = (
            (strings, strings)
            if mode == "tpb"
            else (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
        )
        with make_torch_sci_solver(
            **_solver_kwargs(
                mode=mode,
                device=device,
                davidson_block_size=None,
                apply_ab_projection=False,
                distributed=config,
            )
        ) as solver:
            # These memory readings would independently select blocks 5 and 7.
            block = 5 if rank == 0 else 7
            free_bytes = int((2 * block + 4) * 36 * 8 / solver.budget_fraction) + 1
            with patch.object(solver, "_free_bytes", return_value=free_bytes):
                result = solver([batch], h1, eri, 4, (2, 2))[0]
            if mode == "gdb" and result.states:
                state = result.states[0].sci_state
                pairs = list(zip(state.ci_strs_a, state.ci_strs_b))
                order = [
                    int(np.searchsorted(strings, a)) * len(strings)
                    + int(np.searchsorted(strings, b))
                    for a, b in pairs
                ]
                # Moving beta creation operators past alpha operators contributes one minus per swap.
                signs = np.array([
                    (-1)
                    ** sum(
                        1
                        for p in range(4)
                        for q in range(p)
                        if int(a) & (1 << p) and int(b) & (1 << q)
                    )
                    for a, b in pairs
                ])
                h = signs[:, None] * h[np.ix_(order, order)] * signs[None, :]
            if result.states:
                _assert_eigenpairs(result, h, 2)
            else:
                # Owner-computes results are built on rank 0 only; non-zero
                # ranks participate in the collectives and return an empty
                # EigenStates. Both TPB and GDB take that path here.
                assert rank != 0
            chosen = solver.stats["per_call_log"][0]["subspaces"][0]["davidson_block_size"]
            blocks = [None, None]
            dist.all_gather_object(blocks, chosen)
            assert blocks[0] == blocks[1], blocks
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_auto_block_size_agrees_across_ranks(
    tmp_path: Path,
    mode: str,
    distributed_config: dict,
) -> None:
    """Hardware memory differences must not change collective shapes or the spectrum."""
    _hubbard()  # Check reference dependency before launching children.
    mp.spawn(
        _unequal_memory_worker, args=(str(tmp_path / "init"), mode, distributed_config), nprocs=2
    )


def _memmap_hand_back_worker(rank: int, init_file: str, config: dict) -> None:
    """A forced memmapped hand-back must build a real ``(M_a, M_b)`` SCIState."""
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if config["backend"] == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        config["backend"],
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=config["timeout"]),
    )
    try:
        import torch_sci_solver.torch_impl.distributed_solver as ds

        h1, eri, strings, _ = _hubbard()
        solver = make_torch_sci_solver(
            **_solver_kwargs(
                mode="tpb",
                device=device,
                n_roots=1,
                apply_ab_projection=False,
                distributed=config,
            )
        )
        # Reference amplitudes from the ordinary in-memory gather.
        result = solver([(strings, strings)], h1, eri, 4, (2, 2))[0]
        reference = result.states[0].sci_state.amplitudes.copy() if rank == 0 else None
        # Force the memmap branch (1 byte threshold). qiskit's SCIState rejects
        # a MemmapAmplitudes marker, so the worker must defer construction to
        # the parent, which maps the file and reshapes to (M_a, M_b).
        ds.DISTRIBUTED_HAND_BACK_SMALL_BYTES = 1
        result = solver([(strings, strings)], h1, eri, 4, (2, 2))[0]
        if rank == 0:
            state = result.states[0].sci_state
            assert state.amplitudes.shape == (len(strings), len(strings))
            np.testing.assert_allclose(state.amplitudes, reference, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_memmap_hand_back_builds_sci_state(tmp_path: Path, distributed_config: dict) -> None:
    """The large-result TPB hand-back must materialize a valid qiskit SCIState."""
    _hubbard()  # Check reference dependency before launching children.
    mp.spawn(_memmap_hand_back_worker, args=(str(tmp_path / "init"), distributed_config), nprocs=2)


def _tile_memmap_worker(rank: int, init_file: str, hand_back_dir: str) -> None:
    """Every rank writes its tile rectangle into one shared 2-D memmap."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=120),
    )
    try:
        from torch_sci_solver.torch_impl.distributed_davidson import (
            DistributedContext,
            partition_tiles,
        )
        from torch_sci_solver.torch_impl.distributed_solver import write_tile_memmaps

        m_a, m_b = 6, 6
        tile = partition_tiles(m_a, m_b, 4, grid=(2, 2))
        bound = tile.for_rank(rank)
        values = torch.arange(m_a * m_b, dtype=torch.float64).reshape(m_a, m_b)
        block = values[bound.row_start : bound.row_stop, bound.col_start : bound.col_stop].reshape(
            -1
        )
        context = DistributedContext(
            rank=rank,
            world_size=4,
            backend="gloo",
            device=torch.device("cpu"),
            hand_back_dir=hand_back_dir,
        )
        markers = write_tile_memmaps(
            {"amp": block},
            {"amp": (m_a, m_b)},
            row_start=bound.row_start,
            row_stop=bound.row_stop,
            col_start=bound.col_start,
            col_stop=bound.col_stop,
            context=context,
        )
        if rank == 0:
            assert markers["amp"].shape == (m_a * m_b,)
            assert Path(markers["amp"].path).parent.parent == Path(hand_back_dir)
            written = np.load(markers["amp"].path)
            assert written.shape == (m_a, m_b)
            np.testing.assert_array_equal(written, values.numpy())
    finally:
        dist.destroy_process_group()


def test_create_result_memmap_copes_with_lazy_truncate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hand-back file is creatable where a fresh truncate is not yet visible."""
    from torch_sci_solver.torch_impl.distributed_solver import _create_result_memmap

    real = np.lib.format.open_memmap

    def lazy(path, mode="r+", **kwargs):
        if "w" in mode:
            raise ValueError("mmap length is greater than file size")
        return real(path, mode=mode, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", lazy)
    path = tmp_path / "amp.npy"
    handle = _create_result_memmap(path, np.float64, (7, 5))
    handle[1:3, 2:4] = 3.0
    handle.flush()
    del handle
    array = np.load(path)
    assert array.shape == (7, 5)
    assert float(array[2, 3]) == 3.0
    assert float(array[0, 0]) == 0.0


def test_tile_memmap_writes_each_rectangle(tmp_path: Path) -> None:
    """The tile hand-back fills every rectangle of a shared 2-D file."""
    hand_back = tmp_path / "hand_back"
    hand_back.mkdir()
    mp.spawn(
        _tile_memmap_worker,
        args=(str(tmp_path / "init_tile_memmap"), str(hand_back)),
        nprocs=4,
    )


def test_distributed_preserves_restart_budget(distributed_config: dict) -> None:
    """The existing 20-restart API must still converge on the repulsive Hubbard model."""
    h1, eri, strings, h = _hubbard()
    for distributed in (None, distributed_config):
        with make_torch_sci_solver(
            **_solver_kwargs(
                n_roots=1,
                max_davidson_iters=20,
                apply_ab_projection=False,
                distributed=distributed,
            )
        ) as solver:
            result = solver([(strings, strings)], h1, eri, 4, (2, 2))[0]
        _assert_eigenpairs(result, h, 1)


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
@pytest.mark.parametrize("norb", [1, 2])
def test_dense_path_allows_entire_spectrum(mode: str, norb: int, distributed_config: dict) -> None:
    """N roots in an N-dimensional space need no additional restart direction."""
    eps = np.arange(norb, dtype=float) - 1.0
    eri = np.zeros((norb,) * 4)
    for i in range(norb):
        eri[i, i, i, i] = 0.5
    # On-site repulsion is paid only for a doubly occupied spatial orbital.
    h = np.diag((eps[:, None] + eps[None, :] + 0.5 * np.eye(norb)).ravel())
    strings = 1 << np.arange(norb)
    batch = (
        (strings, strings) if mode == "tpb" else (np.repeat(strings, norb), np.tile(strings, norb))
    )
    with make_torch_sci_solver(
        **_solver_kwargs(
            mode=mode,
            n_roots=norb**2,
            apply_ab_projection=False,
            distributed=distributed_config,
        )
    ) as solver:
        result = solver([batch], np.diag(eps), eri, norb, (1, 1))[0]
    _assert_eigenpairs(result, h, norb**2)


@pytest.mark.parametrize(
    "visible,ids,local_rank,expected",
    [
        (1, [0, 0], 1, 0),
        (1, None, 1, 0),
        (2, [1, 0], 0, 1),
        (2, None, 1, 1),
    ],
)
def test_launcher_respects_visible_device_mapping(
    monkeypatch: pytest.MonkeyPatch,
    visible: int,
    ids: list[int] | None,
    local_rank: int,
    expected: int,
) -> None:
    """CUDA ordinals are relative to the process's visible devices, not its global rank."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible)
    raw = {"world_size": 2, "backend": "nccl"}
    if ids is not None:
        raw["device_ids"] = ids
    cfg = parse_distributed_config(raw, device="cuda", cuda_device_count=visible)
    assert _rank_device(cfg, local_rank) == torch.device("cuda", expected)


def test_persistent_session_uses_updated_integrals(distributed_config: dict) -> None:
    """Adding delta to every orbital shifts four-electron energies by 4 delta."""
    h1, eri, strings, h = _hubbard()
    with make_torch_sci_solver(
        **_solver_kwargs(
            distributed=distributed_config,
        )
    ) as solver:
        first = solver([(strings, strings)], h1, eri, 4, (2, 2))[0]
        h1 += np.eye(4)
        second = solver([(strings, strings)], h1, eri, 4, (2, 2))[0]
    _assert_eigenpairs(first, h, 2)
    _assert_eigenpairs(second, h + 4 * np.eye(36), 2)
