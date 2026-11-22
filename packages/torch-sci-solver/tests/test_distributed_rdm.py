"""Output-partitioned distributed RDM accumulation (TPB, GDB and structural tiles).

The serial/tiled RDM sweeps used to replicate the full ``3 L^4`` accumulator on
every rank and end with one ``all_reduce``. The output-partitioned entry points
split the flattened output across the ranks and process one owner group per
pass, so each rank holds an ``L^4 / world_size`` slice plus one group buffer.

Covers:

1. gloo, 2 and 4 ranks, real and complex: the output-partitioned tiled 1-RDM
   and 2-RDM equal the serial builders at 1e-10 for a TPB tile grid, an
   explicit GDB (pair) tile and a structural pair tile, for two state vectors
   (a "root 0" state and a second root).
2. Allocation guard on every rank: with ``groups=1`` the peak tracked tensor
   allocation is one output slice and no rank ever allocates the full ``3 L^4``
   output; the persistent slice is exactly ``ceil(3 L^4 / world_size)``.
3. Fallback: with a byte threshold above the full output, the distributed
   entry points return the existing replicated accumulator's result
   bit-identically.
4. ``L > 64``: a 70-orbital TPB tile with a storage-free low-rank ERI builds
   the partitioned Γαβ block and matches the serial builder on each rank's
   slice, without gathering the full output.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import pathlib

import numpy as np
import pytest
import torch
import torch.distributed as dist
from test_distributed_pairs import _init_gloo, _run_spawned
from test_mult_pair_blocks import _complex_integrals, _real_integrals
from test_pair_tiled_postprocess import _reference as _pair_reference
from test_pair_tiled_postprocess import _setup as _pair_setup
from test_pair_tiled_postprocess import _tile_operator as _pair_tile_operator
from test_structural_tiled_postprocess import NORB as STRUCT_NORB
from test_structural_tiled_postprocess import _tile_operator as _struct_tile_operator
from test_tiled_postprocess import M_STRINGS as TPB_M_STRINGS
from test_tiled_postprocess import NORB as TPB_NORB
from test_tiled_postprocess import _local_helpers, _tile_slice
from test_tiled_postprocess import _setup as _tpb_setup
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles
from torch_sci_solver.torch_impl.distributed_matvec import LocalTPBOperator
from torch_sci_solver.torch_impl.helpers import generate_local_torch_helpers
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch
from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
    can_partition_pair_tiled_rdm,
    pair_tiled_one_rdm,
    pair_tiled_one_rdm_distributed,
    pair_tiled_two_rdm,
    pair_tiled_two_rdm_distributed,
)
from torch_sci_solver.torch_impl.rdm import (
    make_one_rdm_torch,
    make_two_rdm_ab_torch,
    make_two_rdm_torch,
)
from torch_sci_solver.torch_impl.rdm_pairs import make_one_rdm_pair_torch, make_two_rdm_pair_torch
from torch_sci_solver.torch_impl.tiled_postprocess import (
    can_partition_tiled_rdm,
    tiled_one_rdm_distributed,
    tiled_two_rdm,
    tiled_two_rdm_distributed,
)

ATOL = 1e-10
# [setup, tile grid, alpha-string count]
TPB_CASES = {
    2: ((2, 1), 6),
    4: ((2, 2), 6),
}


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _second_root(x: torch.Tensor) -> torch.Tensor:
    """A deterministic, normalised second state vector of the same sector."""
    shifted = torch.roll(x, 1)
    return shifted / shifted.norm()


def _spy_allocations():
    """Return ``(tracked, restore)`` wrapping the torch tensor constructors.

    ``tracked`` records the ``numel`` of every tensor made by ``torch.zeros``,
    ``torch.empty``, ``torch.full`` and ``torch.arange`` while the spy is
    installed; ``restore`` puts the originals back.
    """
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


def _assert_slice_matches_full(local: torch.Tensor, partition, full: torch.Tensor) -> None:
    """Assert a partitioned flat slice equals the same range of a full tensor.

    Args:
        local: ``(local_size,)`` partitioned flat slice.
        partition: The output partition of ``local``.
        full: Full tensor whose flat index space the slice refers to.
    """
    flat = full.reshape(-1)
    start, stop = partition.start, partition.stop
    torch.testing.assert_close(
        local, flat[start:stop], atol=ATOL, rtol=0.0, msg=f"slice [{start}, {stop})"
    )


# ============================================================================
# 1. TPB tiles
# ============================================================================


def _tpb_worker(
    rank: int,
    world_size: int,
    complex_dtype: bool,
    groups: int,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's partitioned TPB RDMs match the serial builders, two roots."""
    _init_gloo(rank, world_size, init_file)
    try:
        grid, n_alpha = TPB_CASES[world_size]
        hii, adets_t, bdets_t, integrals, helpers, x = _tpb_setup(n_alpha, complex_dtype)
        m_a, m_b = len(adets_t), len(bdets_t)
        tile = partition_tiles(m_a, m_b, world_size, grid=grid)
        bounds = tile.for_rank(rank)
        operator = LocalTPBOperator(
            _tile_slice(hii, m_a, m_b, bounds),
            adets_t,
            bdets_t,
            TPB_NORB,
            integrals,
            _local_helpers(adets_t, bdets_t, bounds),
            tile,  # type: ignore[arg-type]  # the operator accepts a TilePartition
            rank=rank,
            world_size=world_size,
        )
        assert can_partition_tiled_rdm(operator, order=1)
        assert can_partition_tiled_rdm(operator, order=2)
        assert not can_partition_tiled_rdm(operator, order=2, max_local_bytes=1)
        norb = TPB_NORB

        for label, vector in (("root0", x), ("root1", _second_root(x))):
            x_tile = _tile_slice(vector, m_a, m_b, bounds)
            ref_a, ref_b = make_one_rdm_torch(vector, adets_t, bdets_t, norb, helpers)
            ref_aa, ref_ab, ref_bb = make_two_rdm_torch(vector, adets_t, bdets_t, norb, helpers)

            # Replicated result (the partitioned sweep plus the final gather).
            ga, gb = tiled_one_rdm_distributed(
                operator, x_tile, max_replicated_bytes=0, groups=groups
            )
            gamma_aa, gamma_ab, gamma_bb = tiled_two_rdm_distributed(
                operator, x_tile, max_replicated_bytes=0, groups=groups
            )
            for got, want, name in (
                (ga, ref_a, "γα"),
                (gb, ref_b, "γβ"),
                (gamma_aa, ref_aa, "Γαα"),
                (gamma_ab, ref_ab, "Γαβ"),
                (gamma_bb, ref_bb, "Γββ"),
            ):
                torch.testing.assert_close(got, want, atol=ATOL, rtol=0.0, msg=f"{label} {name}")

            # Partitioned handle: the owned flat slice equals the full result.
            part1 = tiled_one_rdm_distributed(operator, x_tile, replicate=False, groups=groups)
            part2 = tiled_two_rdm_distributed(operator, x_tile, replicate=False, groups=groups)
            assert part1.partition.local_size == (2 * norb**2 + world_size - 1) // world_size
            assert part2.partition.local_size == (3 * norb**4 + world_size - 1) // world_size
            _assert_slice_matches_full(part1.data, part1.partition, torch.stack([ref_a, ref_b]))
            _assert_slice_matches_full(
                part2.data, part2.partition, torch.stack([ref_aa, ref_ab, ref_bb])
            )
            # The small-output fallback is the existing replicated accumulator,
            # bit for bit.
            fallback = tiled_two_rdm_distributed(operator, x_tile, max_replicated_bytes=1 << 40)
            plain_aa, plain_ab, plain_bb = tiled_two_rdm(operator, x_tile)
            assert torch.equal(fallback[0], plain_aa)
            assert torch.equal(fallback[1], plain_ab)
            assert torch.equal(fallback[2], plain_bb)
        if rank == 0:
            pathlib.Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("complex_dtype", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("groups", [1, 2])
def test_tpb_tiled_partitioned_matches_serial(
    tmp_path, world_size: int, complex_dtype: bool, groups: int
) -> None:
    """gloo, 2/4 ranks: the partitioned TPB RDMs match the serial builders."""
    _skip_without_gloo()
    _run_spawned(
        _tpb_worker,
        world_size,
        (complex_dtype, groups),
        tmp_path,
        f"rdm_tpb_{world_size}_{'c' if complex_dtype else 'r'}_g{groups}",
    )


# ============================================================================
# 2. GDB (explicit pair) and structural tiles
# ============================================================================


PAIR_CASES = {
    2: ((2, 1), "full_product"),
    4: ((2, 2), "tier"),
}


def _pair_worker(
    rank: int,
    world_size: int,
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's partitioned GDB RDMs match the serial pair builders."""
    _init_gloo(rank, world_size, init_file)
    try:
        grid, basis_name = PAIR_CASES[world_size]
        basis, norb, helpers, x = _pair_setup(basis_name, complex_values, seed=7)
        integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
        hii = make_diag_terms_pair_torch(basis, norb, integrals)
        partition = partition_tiles(
            basis.n_unique_alphas, basis.n_unique_betas, world_size, grid=grid
        )
        operator, bounds = _pair_tile_operator(
            basis, norb, helpers, integrals, hii, partition, rank
        )
        assert can_partition_pair_tiled_rdm(operator, order=1)
        assert can_partition_pair_tiled_rdm(operator, order=2)
        assert not can_partition_pair_tiled_rdm(operator, order=2, max_local_bytes=1)

        for label, vector in (("root0", x), ("root1", _second_root(x))):
            x_local = vector[bounds.pair_index]
            ref = _pair_reference(basis, helpers, norb, vector)
            ga, gb = pair_tiled_one_rdm_distributed(
                operator, x_local, max_replicated_bytes=0, groups=1
            )
            gamma_aa, gamma_ab, gamma_bb = pair_tiled_two_rdm_distributed(
                operator, x_local, max_replicated_bytes=0, groups=1
            )
            for got, want, name in (
                (ga, ref[0], "γα"),
                (gb, ref[1], "γβ"),
                (gamma_aa, ref[2], "Γαα"),
                (gamma_ab, ref[3], "Γαβ"),
                (gamma_bb, ref[4], "Γββ"),
            ):
                torch.testing.assert_close(got, want, atol=ATOL, rtol=0.0, msg=f"{label} {name}")

            part2 = pair_tiled_two_rdm_distributed(operator, x_local, replicate=False, groups=1)
            _assert_slice_matches_full(
                part2.data, part2.partition, torch.stack([ref[2], ref[3], ref[4]])
            )
            # The fallback path is bit-identical to the replicated accumulator.
            fallback = pair_tiled_two_rdm_distributed(
                operator, x_local, max_replicated_bytes=1 << 40
            )
            plain = pair_tiled_two_rdm(operator, x_local)
            for got, want in zip(fallback, plain, strict=True):
                assert torch.equal(got, want)
            fallback1 = pair_tiled_one_rdm_distributed(
                operator, x_local, max_replicated_bytes=1 << 40
            )
            plain1 = pair_tiled_one_rdm(operator, x_local)
            for got, want in zip(fallback1, plain1, strict=True):
                assert torch.equal(got, want)
        if rank == 0:
            pathlib.Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_pair_tiled_partitioned_matches_serial(
    tmp_path, world_size: int, complex_values: bool
) -> None:
    """gloo, 2/4 ranks: the partitioned GDB RDMs match the serial builders."""
    _skip_without_gloo()
    _run_spawned(
        _pair_worker,
        world_size,
        (complex_values,),
        tmp_path,
        f"rdm_pair_{world_size}_{'c' if complex_values else 'r'}",
    )


# ============================================================================
# 3. Allocation guard
# ============================================================================


def _tpb_allocation_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
) -> None:
    """partitioned TPB 2-RDM: peak allocation is the local slice, not 3 L^4."""
    _init_gloo(rank, world_size, init_file)
    try:
        grid, n_alpha = TPB_CASES[world_size]
        hii, adets_t, bdets_t, integrals, helpers, x = _tpb_setup(n_alpha, False)
        m_a, m_b = len(adets_t), len(bdets_t)
        norb = TPB_NORB
        tile = partition_tiles(m_a, m_b, world_size, grid=grid)
        bounds = tile.for_rank(rank)
        operator = LocalTPBOperator(
            _tile_slice(hii, m_a, m_b, bounds),
            adets_t,
            bdets_t,
            norb,
            integrals,
            _local_helpers(adets_t, bdets_t, bounds),
            tile,  # type: ignore[arg-type]
            rank=rank,
            world_size=world_size,
        )
        x_tile = _tile_slice(x, m_a, m_b, bounds)
        full = 3 * norb**4
        slice_size = (full + world_size - 1) // world_size
        tracked, restore = _spy_allocations()
        try:
            part = tiled_two_rdm_distributed(
                operator, x_tile, replicate=False, groups=1, max_replicated_bytes=0
            )
        finally:
            restore()
        assert part.partition.local_size == slice_size
        assert tracked, "the allocation spy recorded nothing"
        assert max(tracked) <= slice_size, (max(tracked), slice_size)
        assert full not in tracked, "the full 3 L^4 output was allocated"
        # The comparison must not be vacuous.
        ref_aa, ref_ab, ref_bb = make_two_rdm_torch(x, adets_t, bdets_t, norb, helpers)
        _assert_slice_matches_full(part.data, part.partition, torch.stack([ref_aa, ref_ab, ref_bb]))
    finally:
        dist.destroy_process_group()
    if rank == 0:
        pathlib.Path(out_file).write_text("ok")


@pytest.mark.parametrize("world_size", [2, 4])
def test_tpb_partitioned_allocation_guard(tmp_path, world_size: int) -> None:
    """gloo, 2/4 ranks: no rank allocates the full 2-RDM output."""
    _skip_without_gloo()
    _run_spawned(_tpb_allocation_worker, world_size, (), tmp_path, f"rdm_tpb_alloc_{world_size}")


def _struct_allocation_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
) -> None:
    """Structural tile: peak allocation is the local slice, not 3 L^4."""
    _init_gloo(rank, world_size, init_file)
    try:
        operator, basis, helpers, x = _struct_tile_operator(rank, world_size, chunk_size=64)
        x_local = x[operator.bounds.pair_index]
        full = 3 * STRUCT_NORB**4
        slice_size = (full + world_size - 1) // world_size
        tracked, restore = _spy_allocations()
        try:
            part = pair_tiled_two_rdm_distributed(
                operator, x_local, replicate=False, groups=1, max_replicated_bytes=0
            )
        finally:
            restore()
        assert part.partition.local_size == slice_size
        assert max(tracked) <= slice_size, (max(tracked), slice_size)
        assert full not in tracked
        ref_aa, ref_ab, ref_bb = make_two_rdm_pair_torch(x, basis, helpers, STRUCT_NORB)
        _assert_slice_matches_full(part.data, part.partition, torch.stack([ref_aa, ref_ab, ref_bb]))
        # The 1-RDM uses the same machinery with the smaller packed output.
        part1 = pair_tiled_one_rdm_distributed(
            operator, x_local, replicate=False, groups=1, max_replicated_bytes=0
        )
        ref_a, ref_b = make_one_rdm_pair_torch(x, basis, helpers, STRUCT_NORB)
        _assert_slice_matches_full(part1.data, part1.partition, torch.stack([ref_a, ref_b]))
        if rank == 0:
            pathlib.Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_structural_partitioned_allocation_guard(tmp_path, world_size: int) -> None:
    """gloo, 2/4 ranks: the structural tile never allocates the full output."""
    _skip_without_gloo()
    _run_spawned(
        _struct_allocation_worker, world_size, (), tmp_path, f"rdm_struct_alloc_{world_size}"
    )


# ============================================================================
# 4. L > 64: a wide TPB tile with a storage-free low-rank ERI
# ============================================================================


WIDE_NORB = 70


def _wide_strings() -> list[int]:
    """Six 2-electron strings straddling the 64-bit limb boundary."""
    high = 65
    base = (1 << 0) | (1 << 64)
    strings = {base, (1 << 1) | (1 << high)}
    for i in (0, 64):
        for j in (1, high):
            strings.add((base ^ (1 << i)) | (1 << j))
    return sorted(strings)


_MASK64 = (1 << 64) - 1


def _limb_array(strings: list[int], n_orb: int) -> torch.Tensor:
    """``(M, n_limbs)`` signed-int64 limb matrix of arbitrary-precision strings."""
    n_limbs = max(1, (n_orb + 63) // 64)
    out = np.zeros((len(strings), n_limbs), dtype=np.int64)
    for k in range(n_limbs):
        words = [(int(v) >> (64 * k)) & _MASK64 for v in strings]
        out[:, k] = [word - (1 << 64) if word >= (1 << 63) else word for word in words]
    return torch.from_numpy(out)


class _LowRankEri:
    """Storage-free low-rank ERI: ``E[p, q, r, s] = Σ_t A[p, r, t] B[q, s, t]``.

    The tiled operator only reads the ERI while building its β-pass cache; the
    post-processing sweep never touches it. A storage-free factorisation keeps
    the ``L = 70`` fixture small while still exercising that construction.
    """

    def __init__(self, n_spinorb: int, rank: int = 4, seed: int = 0) -> None:
        """Draw the two factor tensors."""
        generator = torch.Generator().manual_seed(seed)
        self.left = torch.randn(
            n_spinorb, n_spinorb, rank, generator=generator, dtype=torch.float64
        )
        self.right = torch.randn(
            n_spinorb, n_spinorb, rank, generator=generator, dtype=torch.float64
        )
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        self.shape = (n_spinorb, n_spinorb, n_spinorb, n_spinorb)

    def __getitem__(self, key):
        """Index with four (broadcasting) integer tensors."""
        p, q, r, s = key
        return torch.einsum("...t,...t->...", self.left[p, r], self.right[q, s])


def _wide_integrals(n_orb: int) -> TorchIntegrals:
    """A low-rank integral bundle for the wide fixture."""
    n_spinorb = 2 * n_orb
    generator = torch.Generator().manual_seed(3)
    h1 = torch.randn(n_spinorb, n_spinorb, generator=generator, dtype=torch.float64)
    h1 = 0.5 * (h1 + h1.T)
    return TorchIntegrals(
        I0=0.0,
        I1=h1,
        I2_eri=_LowRankEri(n_spinorb),
        I1_diag=h1.diagonal().contiguous(),
        I2_direct=torch.zeros(n_spinorb, n_spinorb, dtype=torch.float64),
        I2_exchange=torch.zeros(n_spinorb, n_spinorb, dtype=torch.float64),
        n_spinorb=n_spinorb,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )


def _wide_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Rank slice of the partitioned Γαβ at 70 orbitals matches the serial block."""
    _init_gloo(rank, world_size, init_file)
    try:
        n_orb = WIDE_NORB
        integrals = _wide_integrals(n_orb)
        adets = _wide_strings()
        bdets = list(adets)
        adets_t = _limb_array(adets, n_orb)
        bdets_t = _limb_array(bdets, n_orb)
        from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers

        helpers = list(generate_all_torch_helpers(adets, bdets, n_orb))
        hii = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals)
        m_a, m_b = len(adets), len(bdets)
        tile = partition_tiles(m_a, m_b, world_size, grid=(world_size, 1))
        bounds = tile.for_rank(rank)
        local_helpers = list(
            generate_local_torch_helpers(
                adets_t.tolist(),
                bdets_t.tolist(),
                n_orb,
                bounds.row_start,
                bounds.row_stop,
                device="cpu",
                col_start=bounds.col_start,
                col_stop=bounds.col_stop,
            )
        )
        operator = LocalTPBOperator(
            _tile_slice(hii, m_a, m_b, bounds),
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            local_helpers,
            tile,  # type: ignore[arg-type]
            rank=rank,
            world_size=world_size,
        )
        x = torch.randn(m_a * m_b, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        x_tile = _tile_slice(x, m_a, m_b, bounds)
        part = tiled_two_rdm_distributed(
            operator, x_tile, replicate=False, groups=1, max_replicated_bytes=0
        )
        ref_ab = make_two_rdm_ab_torch(x, adets_t, bdets_t, n_orb, helpers)
        start, stop = part.partition.start, part.partition.stop
        block_lo, block_hi = n_orb**4, 2 * n_orb**4
        first, last = max(start, block_lo), min(stop, block_hi)
        if first < last:
            torch.testing.assert_close(
                part.data[first - start : last - start],
                ref_ab.reshape(-1)[first - block_lo : last - block_lo],
                atol=ATOL,
                rtol=0.0,
                msg=f"Γαβ slice [{first}, {last})",
            )
        # The 1-RDM partitioned path is exercised on the same wide tile.
        part1 = tiled_one_rdm_distributed(
            operator, x_tile, replicate=False, groups=1, max_replicated_bytes=0
        )
        ref_a, ref_b = make_one_rdm_torch(x, adets_t, bdets_t, n_orb, helpers)
        _assert_slice_matches_full(part1.data, part1.partition, torch.stack([ref_a, ref_b]))
        if rank == 0:
            pathlib.Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_wide_tpb_partitioned_with_low_rank_eri(tmp_path) -> None:
    """gloo, 2 ranks, L=70: the partitioned Γαβ matches the serial builder."""
    _skip_without_gloo()
    _run_spawned(_wide_worker, 2, (), tmp_path, "rdm_tpb_wide")


# ============================================================================
# 5. Single rank: the partitioned path needs no process group
# ============================================================================


def test_single_rank_partitioned_matches_serial() -> None:
    """No process group: both partitioned builders equal the serial results."""
    hii, adets_t, bdets_t, integrals, helpers, x = _tpb_setup(TPB_M_STRINGS, False)
    m_a, m_b = len(adets_t), len(bdets_t)
    tile = partition_tiles(m_a, m_b, 1, grid=(1, 1))
    bounds = tile.for_rank(0)
    operator = LocalTPBOperator(
        _tile_slice(hii, m_a, m_b, bounds),
        adets_t,
        bdets_t,
        TPB_NORB,
        integrals,
        _local_helpers(adets_t, bdets_t, bounds),
        tile,  # type: ignore[arg-type]
        rank=0,
        world_size=1,
    )
    x_tile = _tile_slice(x, m_a, m_b, bounds)
    ref_a, ref_b = make_one_rdm_torch(x, adets_t, bdets_t, TPB_NORB, helpers)
    ref_aa, ref_ab, ref_bb = make_two_rdm_torch(x, adets_t, bdets_t, TPB_NORB, helpers)
    part1 = tiled_one_rdm_distributed(operator, x_tile, replicate=False)
    part2 = tiled_two_rdm_distributed(operator, x_tile, replicate=False)
    torch.testing.assert_close(part1.gather(), torch.stack([ref_a, ref_b]), atol=ATOL, rtol=0.0)
    torch.testing.assert_close(
        part2.gather(), torch.stack([ref_aa, ref_ab, ref_bb]), atol=ATOL, rtol=0.0
    )
    at_root = part2.gather_to_root(root=0)
    assert at_root is not None
    torch.testing.assert_close(at_root, torch.stack([ref_aa, ref_ab, ref_bb]), atol=ATOL, rtol=0.0)


def test_single_rank_pair_partitioned_matches_serial() -> None:
    """No process group: the partitioned GDB builder equals the serial result."""
    basis, norb, helpers, x = _pair_setup("full_product", False)
    integrals = _real_integrals(norb)
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 1, grid=(1, 1))
    operator, bounds = _pair_tile_operator(basis, norb, helpers, integrals, hii, partition, 0)
    reference = _pair_reference(basis, helpers, norb, x)
    part = pair_tiled_two_rdm_distributed(operator, x[bounds.pair_index], replicate=False)
    torch.testing.assert_close(
        part.gather(),
        torch.stack([reference[2], reference[3], reference[4]]),
        atol=ATOL,
        rtol=0.0,
    )


# ============================================================================
# 6. Solver finish paths: partitioned RDM accumulation
# ============================================================================


SOLVER_NORB = 5
SOLVER_NELEC = (2, 2)
SOLVER_STRINGS = np.array(
    sorted(
        sum(1 << orbital for orbital in occ)
        for occ in itertools.combinations(range(SOLVER_NORB), 2)
    ),
    dtype=np.int64,
)
# Well below 3 * L^4 elements in either dtype, so the full output may not be
# replicated and the finish path must take the partitioned entries.
SOLVER_TINY_BUDGET = 1024
SOLVER_TOL = 1e-10


def _solver_integrals(complex_case: bool) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 5-orbital integrals, real or Hermitian complex."""
    rng = np.random.default_rng(7)
    if not complex_case:
        h1 = rng.standard_normal((SOLVER_NORB, SOLVER_NORB))
        h1 = 0.5 * (h1 + h1.T)
        raw = rng.standard_normal((SOLVER_NORB,) * 4)
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
        return h1, eri / 8.0
    hc = rng.standard_normal((SOLVER_NORB, SOLVER_NORB)) + 1j * rng.standard_normal((
        SOLVER_NORB,
        SOLVER_NORB,
    ))
    h1 = hc + hc.conj().T
    eri = np.zeros((SOLVER_NORB,) * 4, dtype=np.complex128)
    for _ in range(SOLVER_NORB + 2):
        g = rng.standard_normal((SOLVER_NORB, SOLVER_NORB)) + 1j * rng.standard_normal((
            SOLVER_NORB,
            SOLVER_NORB,
        ))
        herm = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", herm, herm.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return h1, eri


def _solver_ci_strings(mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The full-product CI strings for the TPB or GDB (pair) layout."""
    if mode == "tpb":
        return [(SOLVER_STRINGS, SOLVER_STRINGS)]
    pairs_a = np.repeat(SOLVER_STRINGS, len(SOLVER_STRINGS))
    pairs_b = np.tile(SOLVER_STRINGS, len(SOLVER_STRINGS))
    return [(pairs_a, pairs_b)]


def _solver_kwargs(complex_case: bool, mode: str, **overrides):
    """Deterministic solver settings for the serial reference and the ranks."""
    kwargs = dict(
        device="cpu",
        dtype=torch.complex128 if complex_case else torch.float64,
        n_roots=1,
        davidson_block_size=8,
        max_davidson_iters=300,
        davidson_tolerance=SOLVER_TOL,
        davidson_missed_root_check=False,
        compute_full_rdm=True,
        verbose=False,
    )
    if mode == "gdb":
        kwargs.update(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    kwargs.update(overrides)
    return kwargs


def _serial_reference(complex_case: bool, mode: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Energy and both RDMs from the single-process full-RDM solve."""
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    h1, eri = _solver_integrals(complex_case)
    solver = make_torch_sci_solver(**_solver_kwargs(complex_case, mode))
    try:
        state = solver(_solver_ci_strings(mode), h1, eri, SOLVER_NORB, SOLVER_NELEC)[0].states[0]
        return float(state.energy), np.asarray(state.rdm1), np.asarray(state.rdm2)
    finally:
        solver.close()


def test_rdm_replication_cap_uses_explicit_budgets() -> None:
    """Both per-rank budgets feed the replicated-RDM byte bound.

    The smaller of ``memory_budget_bytes`` and a distributed ``budget_bytes``
    (attached as the runtime budget) caps the replicated full output, and the
    module default applies when neither is set.
    """
    from torch_sci_solver.torch_impl.distributed_rdm import DEFAULT_MAX_REPLICATED_BYTES
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    solver = make_torch_sci_solver(device="cpu", memory_budget_bytes=4096)
    try:
        assert solver._rdm_max_replicated_bytes() == 4096
        assert solver._rdm_max_replicated_bytes(1024) == 1024
        assert solver._rdm_max_replicated_bytes(1 << 40) == 4096
        solver._runtime_budget_bytes = 2048
        assert solver._rdm_max_replicated_bytes() == 2048
    finally:
        solver.close()
    solver = make_torch_sci_solver(device="cpu")
    try:
        assert solver._rdm_max_replicated_bytes() == DEFAULT_MAX_REPLICATED_BYTES
        assert solver._rdm_max_replicated_bytes(1 << 20) == 1 << 20
    finally:
        solver.close()


def _solver_finish_worker(
    rank: int,
    world_size: int,
    complex_case: bool,
    mode: str,
    budget: "int | None",
    init_file: str,
    out_file: str,
) -> None:
    """One rank of an in-process solver run; checks its finish-path contract."""
    from torch_sci_solver.torch_impl import pair_tiled_postprocess, tiled_postprocess
    from torch_sci_solver.torch_impl.distributed_davidson import DistributedContext
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri = _solver_integrals(complex_case)
        solver = make_torch_sci_solver(
            **_solver_kwargs(complex_case, mode, memory_budget_bytes=budget)
        )
        solver._distributed_ctx = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
            matvec="local",
            tile="force",
        )
        solver._inside_distributed = True

        expected_layout = "partitioned" if budget is not None else "replicated"
        replicated_calls: list[str] = []

        def _replicated_called(name: str):
            """Fail if the finish path falls back to a replicated builder."""

            def _fail(*_args, **_kwargs):
                replicated_calls.append(name)
                raise AssertionError(f"the replicated builder {name} ran")

            return _fail

        originals = {}
        if expected_layout == "partitioned":
            for module, names in (
                (tiled_postprocess, ("tiled_one_rdm", "tiled_two_rdm")),
                (pair_tiled_postprocess, ("pair_tiled_one_rdm", "pair_tiled_two_rdm")),
            ):
                for name in names:
                    originals[(module, name)] = getattr(module, name)
                    setattr(module, name, _replicated_called(name))

        full_rdm_numel = 3 * SOLVER_NORB**4
        tracked, restore = _spy_allocations()
        try:
            result = solver(_solver_ci_strings(mode), h1, eri, SOLVER_NORB, SOLVER_NELEC)[0]
        finally:
            restore()
            for (module, name), original in originals.items():
                setattr(module, name, original)

        layout = solver.stats["rdm_accumulation"]
        assert layout == expected_layout, (rank, layout, expected_layout)
        assert not replicated_calls, replicated_calls
        if expected_layout == "partitioned" and rank != 0:
            # Only the root gathers the full tensors for the packed result.
            assert full_rdm_numel not in tracked, "a non-root rank allocated the full 3 L^4 RDM"
        if rank == 0:
            state = result.states[0]
            np.savez(
                f"{out_file}.npz",
                energy=float(state.energy),
                rdm1=np.asarray(state.rdm1),
                rdm2=np.asarray(state.rdm2),
                layout=np.asarray(layout),
            )
            pathlib.Path(out_file).write_text("ok")
        else:
            # Only rank 0 packs the full tensors into SCIResults.
            assert result.states == [], f"rank {rank} built a result"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("complex_case", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
@pytest.mark.parametrize("partitioned", [False, True], ids=["replicated", "partitioned"])
def test_solver_finish_path_rdm_matches_serial(
    tmp_path, complex_case: bool, mode: str, partitioned: bool
) -> None:
    """gloo, 2 ranks: the solver's full RDMs match the serial solve.

    ``partitioned`` forces the output-partitioned entries with a tiny
    per-rank budget; the replicated arm leaves the budget unset. In both
    cases rank 0's packed result equals the single-process solve, and on the
    partitioned arm no rank runs a replicated RDM builder and no non-root
    rank allocates the full ``3 L^4`` output.
    """
    _skip_without_gloo()
    expected_energy, expected_rdm1, expected_rdm2 = _serial_reference(complex_case, mode)
    budget = SOLVER_TINY_BUDGET if partitioned else None
    _run_spawned(
        _solver_finish_worker,
        2,
        (complex_case, mode, budget),
        tmp_path,
        f"rdm_solver_{mode}_{'c' if complex_case else 'r'}_{'p' if partitioned else 'r'}",
    )
    out_file = (
        tmp_path
        / f"rdm_solver_{mode}_{'c' if complex_case else 'r'}_{'p' if partitioned else 'r'}_out"
    )
    data = np.load(f"{out_file}.npz")
    assert float(data["energy"]) == pytest.approx(expected_energy, abs=SOLVER_TOL)
    np.testing.assert_allclose(data["rdm1"], expected_rdm1, atol=SOLVER_TOL)
    np.testing.assert_allclose(data["rdm2"], expected_rdm2, atol=SOLVER_TOL)
    observed_layout = str(data["layout"])
    assert observed_layout == ("partitioned" if partitioned else "replicated")
