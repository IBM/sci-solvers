"""Tests for the distributed spinless 1-/2-RDM gather.

Covers:

1. Serial fallback (no process group): ``gather_spinless_amplitudes``
   reassembles a full and a scattered ownership index, and
   ``spinless_rdm12_distributed`` matches ``spinless_rdm12_torch`` for real
   and complex amplitudes on both ownership shapes.
2. gloo at 1, 2 and 3 ranks: the gathered vector equals the serial amplitude
   vector on rank 0 (and is empty elsewhere), and both RDMs match the serial
   ``spinless_rdm12_torch`` at 1e-12, for the contiguous ``arange`` block and
   a tile's scattered ``determinant_index``, real and complex.
3. The selected basis is not closed under single excitations, so the
   connected-set subtlety of the 2-RDM Gram term is actually in play.
4. The output-partitioned builder (forced with ``max_replicated_bytes=0``) at
   gloo 2 and 4 ranks: both RDMs match the serial builder, a second run is
   bit-identical, the owned slices and the ``gather``/``gather_to_root``
   handles are consistent, the small-output fallback is bit-identical to the
   serial builder, and the forced path neither gathers amplitudes nor
   allocates a full vector/output tensor.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl.distributed_davidson import partition_vector
from torch_sci_solver.torch_impl.distributed_spinless_rdm import (
    gather_spinless_amplitudes,
    spinless_rdm12_distributed,
)
from torch_sci_solver.torch_impl.rdm_spinless import spinless_rdm12_torch
from torch_sci_solver.torch_impl.spinless_helpers import (
    SpinlessBasis,
    SpinlessHelpers,
    generate_spinless_helpers,
    spinless_basis_from_array,
)
from torch_sci_solver.torch_impl.spinless_tiles import spinless_tile_partition

NMO = 8
N_ELEC = 4
N_SELECTED = 40
SPLIT = 4
CHUNK = 7
TOL = 1e-12
GRIDS = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}
SHAPES = ["contiguous", "tile"]


def _fixture(complex_case: bool) -> tuple[SpinlessBasis, SpinlessHelpers, torch.Tensor]:
    """A selected (non-product) basis, its helpers and a random amplitude vector."""
    combos = list(itertools.combinations(range(NMO), N_ELEC))
    rng = np.random.default_rng(11)
    chosen = sorted(rng.choice(len(combos), size=N_SELECTED, replace=False).tolist())
    strs = np.asarray([sum(1 << bit for bit in combos[i]) for i in chosen], dtype=np.int64)
    basis = spinless_basis_from_array(strs, NMO, N_ELEC)
    helpers = generate_spinless_helpers(basis)
    gen = torch.Generator().manual_seed(7)
    if complex_case:
        x = torch.complex(
            torch.randn(basis.dim, generator=gen, dtype=torch.float64),
            torch.randn(basis.dim, generator=gen, dtype=torch.float64),
        )
    else:
        x = torch.randn(basis.dim, generator=gen, dtype=torch.float64)
    return basis, helpers, x


def _owner_index(
    basis: SpinlessBasis, world_size: int, rank: int, shape: str, grid: tuple[int, int]
) -> torch.Tensor:
    """The global determinant positions of one rank's owned block."""
    if shape == "contiguous":
        bound = partition_vector(basis.dim, world_size, rank)
        return torch.arange(bound.start, bound.stop, dtype=torch.int64)
    _tile, all_bounds = spinless_tile_partition(basis.dets, SPLIT, world_size, grid=grid)
    return all_bounds[rank].determinant_index


# ============================================================================
# 1. Serial fallback
# ============================================================================


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_serial_fallback_matches_serial_rdm(complex_case: bool) -> None:
    """Without a process group, the distributed RDM is the serial result."""
    basis, helpers, x = _fixture(complex_case)
    index = torch.arange(basis.dim, dtype=torch.int64)
    rdm1_ref, rdm2_ref = spinless_rdm12_torch(basis, helpers, x, chunk_size=CHUNK)
    assert torch.equal(gather_spinless_amplitudes(x, index, basis.dim), x)
    rdm1, rdm2 = spinless_rdm12_distributed(basis, helpers, x, index, chunk_size=CHUNK)
    assert rdm1.dtype == x.dtype and rdm2.dtype == x.dtype
    assert torch.allclose(rdm1, rdm1_ref, atol=TOL, rtol=TOL)
    assert torch.allclose(rdm2, rdm2_ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
def test_serial_gather_scatters_a_scattered_index(complex_case: bool) -> None:
    """A scattered (non-contiguous) owned index is scattered into place."""
    basis, _helpers, x = _fixture(complex_case)
    index = torch.arange(0, basis.dim, 2, dtype=torch.int64)
    expected = torch.zeros(basis.dim, dtype=x.dtype)
    expected[index] = x[index]
    assert torch.equal(gather_spinless_amplitudes(x[index], index, basis.dim), expected)


def test_selected_basis_is_not_single_excitation_closed() -> None:
    """The connected set is strictly larger, so the 2-RDM subtlety is in play."""
    basis, helpers, _x = _fixture(False)
    connected, _chunks = helpers.connected_singles()
    assert int(connected.shape[0]) > basis.dim


# ============================================================================
# 2. gloo at 1, 2 and 3 ranks
# ============================================================================


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


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


def _rdm_worker(
    rank: int, world_size: int, complex_case: bool, shape: str, init_file: str, out_file: str
) -> None:
    """Every rank's RDMs equal the serial result; rank 0 gathered the vector."""
    _init_gloo(rank, world_size, init_file)
    try:
        basis, helpers, x = _fixture(complex_case)
        index = _owner_index(basis, world_size, rank, shape, GRIDS[world_size])
        x_local = x[index]
        rdm1_ref, rdm2_ref = spinless_rdm12_torch(basis, helpers, x, chunk_size=CHUNK)

        gathered = gather_spinless_amplitudes(x_local, index, basis.dim)
        if rank == 0:
            assert gathered.shape == (basis.dim,)
            assert torch.equal(gathered, x)
        else:
            assert gathered.numel() == 0
            assert gathered.dtype == x.dtype and gathered.device == x.device

        rdm1, rdm2 = spinless_rdm12_distributed(basis, helpers, x_local, index, chunk_size=CHUNK)
        assert rdm1.shape == rdm1_ref.shape == (NMO, NMO)
        assert rdm2.shape == rdm2_ref.shape == (NMO, NMO, NMO, NMO)
        assert rdm1.dtype == x.dtype and rdm2.dtype == x.dtype
        assert torch.allclose(rdm1, rdm1_ref, atol=TOL, rtol=TOL), (
            rank,
            world_size,
            shape,
            float((rdm1 - rdm1_ref).abs().max()),
        )
        assert torch.allclose(rdm2, rdm2_ref, atol=TOL, rtol=TOL), (
            rank,
            world_size,
            shape,
            float((rdm2 - rdm2_ref).abs().max()),
        )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_rdm_gloo_matches_serial(tmp_path, complex_case: bool, shape: str, world_size: int) -> None:
    """gloo at 1/2/3 ranks: gathered vector and both RDMs match the serial path."""
    _skip_without_gloo()
    _run_spawned(
        _rdm_worker,
        world_size,
        (complex_case, shape),
        tmp_path,
        f"rdm{world_size}{shape}{'c' if complex_case else 'r'}",
    )


# ============================================================================
# 3. Output-partitioned builder: no amplitude gather, bounded output memory
# ============================================================================


def _spy_allocations():
    """Record the ``numel`` of every tensor made by the torch constructors."""
    tracked: list[int] = []
    names = ("empty", "zeros", "full", "arange")
    originals = {name: getattr(torch, name) for name in names}

    def _spy(original):
        """Wrap one constructor so its result's size is recorded."""

        def wrapper(*args, **kwargs):
            tensor = original(*args, **kwargs)
            tracked.append(int(tensor.numel()))
            return tensor

        return wrapper

    for name, original in originals.items():
        setattr(torch, name, _spy(original))

    def restore() -> None:
        """Put the constructors back."""
        for name, original in originals.items():
            setattr(torch, name, original)

    return tracked, restore


def _assert_slice(local: torch.Tensor, partition, full: torch.Tensor) -> None:
    """Assert a partitioned flat slice equals the same range of ``full``."""
    flat = full.reshape(-1)
    assert torch.allclose(local, flat[partition.start : partition.stop], atol=TOL, rtol=TOL), float(
        (local - flat[partition.start : partition.stop]).abs().max()
    )


def _partitioned_worker(
    rank: int,
    world_size: int,
    complex_case: bool,
    shape: str,
    init_file: str,
    out_file: str,
) -> None:
    """Forced partitioned RDM: matches serial, deterministic, never gathers."""
    import torch_sci_solver.torch_impl.distributed_spinless_rdm as dsr

    _init_gloo(rank, world_size, init_file)
    try:
        basis, helpers, x = _fixture(complex_case)
        index = _owner_index(basis, world_size, rank, shape, GRIDS[world_size])
        x_local = x[index]
        rdm1_ref, rdm2_ref = spinless_rdm12_torch(basis, helpers, x, chunk_size=CHUNK)
        nmo = basis.n_spinorb

        rdm1, rdm2 = spinless_rdm12_distributed(
            basis, helpers, x_local, index, chunk_size=CHUNK, max_replicated_bytes=0
        )
        assert torch.allclose(rdm1, rdm1_ref, atol=TOL, rtol=TOL)
        assert torch.allclose(rdm2, rdm2_ref, atol=TOL, rtol=TOL), float(
            (rdm2 - rdm2_ref).abs().max()
        )
        # Pure functions of the basis/partition: a second run is bit-identical.
        again1, again2 = spinless_rdm12_distributed(
            basis, helpers, x_local, index, chunk_size=CHUNK, max_replicated_bytes=0
        )
        assert torch.equal(rdm1, again1) and torch.equal(rdm2, again2)

        part1, part2 = spinless_rdm12_distributed(
            basis,
            helpers,
            x_local,
            index,
            chunk_size=CHUNK,
            max_replicated_bytes=0,
            replicate=False,
        )
        assert part1.partition.local_size == (nmo * nmo + world_size - 1) // world_size
        assert part2.partition.local_size == (nmo**4 + world_size - 1) // world_size
        _assert_slice(part1.data, part1.partition, rdm1_ref)
        _assert_slice(part2.data, part2.partition, rdm2_ref)
        assert torch.allclose(part1.gather(), rdm1_ref, atol=TOL, rtol=TOL)
        assert torch.allclose(part2.gather(), rdm2_ref, atol=TOL, rtol=TOL)
        at_root = part2.gather_to_root(root=0)
        if rank == 0:
            assert at_root is not None
            assert torch.allclose(at_root, rdm2_ref, atol=TOL, rtol=TOL)
        else:
            assert at_root is None

        # The fallback (threshold above the full output) is the old
        # gather-then-serial implementation, bit for bit.
        fallback1, fallback2 = spinless_rdm12_distributed(
            basis, helpers, x_local, index, chunk_size=CHUNK, max_replicated_bytes=1 << 40
        )
        assert torch.equal(fallback1, rdm1_ref)
        assert torch.equal(fallback2, rdm2_ref)

        # No amplitude gather and no full-vector allocation in the forced path.
        def _no_gather(*_args, **_kwargs):
            raise AssertionError("the partitioned builder must not gather amplitudes")

        original_gather_object = dist.gather_object
        original_gather_amplitudes = dsr.gather_spinless_amplitudes
        dist.gather_object = _no_gather
        dsr.gather_spinless_amplitudes = _no_gather
        tracked, restore = _spy_allocations()
        try:
            forced1, forced2 = spinless_rdm12_distributed(
                basis,
                helpers,
                x_local,
                index,
                chunk_size=CHUNK,
                max_replicated_bytes=0,
                replicate=False,
            )
        finally:
            restore()
            dist.gather_object = original_gather_object
            dsr.gather_spinless_amplitudes = original_gather_amplitudes
        assert torch.allclose(forced1.gather(), rdm1_ref, atol=TOL, rtol=TOL)
        assert torch.allclose(forced2.gather(), rdm2_ref, atol=TOL, rtol=TOL)

        connected, _chunks = helpers.connected_singles()
        full_output = nmo**4
        slice_size = (full_output + world_size - 1) // world_size
        b_block = nmo * nmo * int(connected.shape[0])
        assert tracked, "the allocation spy recorded nothing"
        # The peak is the local B chunk (or one output slice), never a dense
        # full-output / full-amplitude workspace. The no-gather monkeypatch
        # above is the direct proof that no vector is collected.
        assert max(tracked) <= max(b_block, slice_size) + nmo * nmo, (max(tracked), b_block)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("world_size", [2, 4])
def test_partitioned_rdm_gloo_matches_serial(
    tmp_path, complex_case: bool, shape: str, world_size: int
) -> None:
    """gloo at 2/4 ranks: the partitioned builder matches serial without a gather."""
    _skip_without_gloo()
    _run_spawned(
        _partitioned_worker,
        world_size,
        (complex_case, shape),
        tmp_path,
        f"rdmpart{world_size}{shape}{'c' if complex_case else 'r'}",
    )


# ============================================================================
# 4. Solver-level wiring: the spinless finish path uses the partitioned RDM
# ============================================================================


SOLVER_NMO = 8
SOLVER_NELEC = 4
SOLVER_STRINGS = np.asarray(
    sorted(
        sum(1 << orbital for orbital in occ)
        for occ in itertools.combinations(range(SOLVER_NMO), SOLVER_NELEC)
    ),
    dtype=np.int64,
)
# Below nmo^4 elements (4096) in complex128, so the full output may not be
# replicated and the finish path must take the partitioned builder.
SOLVER_TINY_BUDGET = 1024
SOLVER_TOL = 1e-10


def _solver_integrals() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Hermitian complex spin-orbital integrals."""
    rng = np.random.default_rng(101)
    hc = rng.standard_normal((SOLVER_NMO, SOLVER_NMO)) + 1j * rng.standard_normal((
        SOLVER_NMO,
        SOLVER_NMO,
    ))
    h1 = hc + hc.conj().T
    eri = np.zeros((SOLVER_NMO,) * 4, dtype=np.complex128)
    for _ in range(SOLVER_NMO + 2):
        g = rng.standard_normal((SOLVER_NMO, SOLVER_NMO)) + 1j * rng.standard_normal((
            SOLVER_NMO,
            SOLVER_NMO,
        ))
        herm = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", herm, herm.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return h1, eri


def _solver_kwargs(**overrides):
    """Deterministic spinless solver settings for the reference and the ranks."""
    kwargs = dict(
        device="cpu",
        dtype=torch.complex128,
        n_roots=1,
        davidson_block_size=8,
        max_davidson_iters=300,
        davidson_tolerance=SOLVER_TOL,
        compute_full_rdm=True,
        verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def _serial_reference() -> tuple[float, np.ndarray, np.ndarray]:
    """Energy and both RDMs from the single-process spinless solve."""
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    h1, eri = _solver_integrals()
    solver = make_torch_sci_solver(**_solver_kwargs())
    try:
        state = solver([SOLVER_STRINGS], h1, eri, SOLVER_NMO, SOLVER_NELEC)[0].states[0]
        return float(state.energy), np.asarray(state.rdm1), np.asarray(state.rdm2)
    finally:
        solver.close()


def _solver_worker(
    rank: int,
    world_size: int,
    partitioned: bool,
    init_file: str,
    out_file: str,
) -> None:
    """One rank of an in-process spinless solve; checks the finish-path contract."""
    import torch_sci_solver.torch_impl.distributed_spinless_rdm as dsr
    from torch_sci_solver.torch_impl.distributed_davidson import DistributedContext
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri = _solver_integrals()
        solver = make_torch_sci_solver(
            **_solver_kwargs(memory_budget_bytes=SOLVER_TINY_BUDGET if partitioned else None)
        )
        solver._distributed_ctx = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
            matvec="local",
        )
        solver._inside_distributed = True

        expected_layout = "partitioned" if partitioned else "replicated"
        original_gather = dsr.gather_spinless_amplitudes

        def _no_gather(*_args, **_kwargs):
            """Fail if the partitioned path gathers the full amplitude vector."""
            raise AssertionError("the partitioned finish path gathered amplitudes")

        if partitioned:
            dsr.gather_spinless_amplitudes = _no_gather
        try:
            result = solver([SOLVER_STRINGS], h1, eri, SOLVER_NMO, SOLVER_NELEC)[0]
        finally:
            dsr.gather_spinless_amplitudes = original_gather

        layout = solver.stats["rdm_accumulation"]
        assert layout == expected_layout, (rank, layout, expected_layout)
        if rank == 0:
            state = result.states[0]
            np.savez(
                f"{out_file}.npz",
                energy=float(state.energy),
                rdm1=np.asarray(state.rdm1),
                rdm2=np.asarray(state.rdm2),
                layout=np.asarray(layout),
            )
            Path(out_file).write_text("ok")
        else:
            # Only rank 0 packs the full tensors into SCIResults.
            assert result.states == [], f"rank {rank} built a result"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("partitioned", [False, True], ids=["replicated", "partitioned"])
def test_solver_finish_path_rdm_matches_serial(tmp_path, partitioned: bool) -> None:
    """gloo, 2 ranks: the spinless solver's full RDMs match the serial solve.

    ``partitioned`` forces the output-partitioned builder with a tiny per-rank
    budget and asserts the finish path never gathers the full amplitude
    vector; the replicated arm leaves the budget unset. In both cases rank 0's
    packed result equals the single-process solve.
    """
    _skip_without_gloo()
    expected_energy, expected_rdm1, expected_rdm2 = _serial_reference()
    name = f"rdm_spinless_solver_{'p' if partitioned else 'r'}"
    _run_spawned(_solver_worker, 2, (partitioned,), tmp_path, name)
    data = np.load(str(tmp_path / f"{name}_out.npz"))
    assert float(data["energy"]) == pytest.approx(expected_energy, abs=SOLVER_TOL)
    np.testing.assert_allclose(data["rdm1"], expected_rdm1, atol=SOLVER_TOL)
    np.testing.assert_allclose(data["rdm2"], expected_rdm2, atol=SOLVER_TOL)
    assert str(data["layout"]) == ("partitioned" if partitioned else "replicated")
