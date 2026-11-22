"""Tests for the distributed owner-computes spinless ("flat") operator.

Covers:

1. ``world_size=1``: ``LocalSpinlessOperator`` matches ``mult_spinless_torch``
   on random complex vectors, and — with a stand-in gather that returns a
   preset full vector — on arbitrary row sub-blocks (several row ranges).
2. gloo, 2 and 3 ranks, uneven ``N``: every rank's output block matches the
   corresponding slice of the full-space ``mult_spinless_torch`` to 1e-12.
3. ``make_diag_terms_spinless_local`` matches the matching slice of
   ``make_diag_terms_spinless_torch`` (dtype and values), both with and
   without chunking, for complex and real integrals.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyscf.fci import cistring
from torch_sci_solver.torch_impl import distributed_spinless as dsl
from torch_sci_solver.torch_impl.distributed_davidson import RankBounds, partition_vector
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
    generate_spinless_helpers,
    spinless_basis_from_array,
)

NMO = 10
N_ELEC = 5
# C(10, 5) = 252 determinants; dropping one makes both 2- and 3-rank splits
# uneven (126/125 and 84/84/83).
N_DETS = 251

# Row sub-blocks exercised by the single-process tests.
ROW_RANGES = [(0, 60), (60, 200), (200, N_DETS)]


def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int):
    """Complex hcore + a physically valid complex chemist-order eri, flat spin-orbital.

    Same factor construction as ``test_spinless_matvec.py``: ``(pq|rs) = sum_P
    L[P] conj(L[P])`` with a Hermitian factor ``L == L^H``, symmetrized over
    ``(pq) <-> (rs)`` — carries both Hermitian symmetries a spinless CI
    operator needs.
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(nmo + 2):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _real_spinorbital_integrals(nmo: int, seed: int):
    """Real symmetric hcore + an eri with the chemist 8-fold symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((nmo, nmo))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((nmo,) * 4)
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
    return h1, eri


def _setup() -> tuple:
    """Build the shared complex fixture: (hii, basis, helpers, integrals, x)."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(NMO, seed=101)
    strs = np.asarray(cistring.make_strings(range(NMO), N_ELEC))[:N_DETS]
    basis = spinless_basis_from_array(strs, NMO, N_ELEC)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    gen = torch.Generator().manual_seed(7)
    x = torch.complex(
        torch.randn(basis.dim, generator=gen, dtype=torch.float64),
        torch.randn(basis.dim, generator=gen, dtype=torch.float64),
    )
    return hii, basis, helpers, integrals, x


def _random_complex_vector(n: int, seed: int) -> torch.Tensor:
    """Deterministic random complex fp64 length-``n`` vector."""
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(n, generator=gen, dtype=torch.float64),
        torch.randn(n, generator=gen, dtype=torch.float64),
    )


class _FullVectorGather:
    """Stand-in for :class:`FlatGather` that returns a preset full vector.

    Lets the operator own an arbitrary row sub-block without a process group,
    mirroring ``test_distributed_matvec.py``'s ``_FullVectorExchange``.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, bounds, *, group=None):
        """Ignore the partition bookkeeping; the full vector is preset."""
        _ = (bounds, group)

    def start(self, local) -> "_FullVectorGather":
        """Accept the local block; the full vector is already known."""
        _ = local
        return self

    def done(self) -> bool:
        """The preset vector is always available."""
        return True

    def wait(self) -> torch.Tensor:
        """Return the preset full-length vector."""
        return self.full_vector


# ============================================================================
# 1. Single process
# ============================================================================


def test_operator_world_size_one_matches_reference() -> None:
    """world_size=1: the operator reproduces ``mult_spinless_torch`` exactly."""
    hii, basis, helpers, integrals, _ = _setup()
    assert basis.dim == N_DETS
    bounds = partition_vector(basis.dim, 1, 0)
    op = dsl.LocalSpinlessOperator(hii, basis, helpers, integrals, bounds, rank=0, world_size=1)
    for seed in (11, 12, 13):
        v = _random_complex_vector(basis.dim, seed)
        expected = mult_spinless_torch(hii, v, basis, helpers, integrals)
        got = op(v)
        assert got.dtype == expected.dtype
        assert got.shape == (basis.dim,)
        assert torch.allclose(got, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("row_range", ROW_RANGES)
def test_operator_subblock_matches_full_slice(
    monkeypatch: pytest.MonkeyPatch, row_range: tuple[int, int]
) -> None:
    """A row sub-block reproduces the corresponding full-matvec slice.

    The stand-in gather supplies the full vector, so no process group is
    needed; this exercises the ``src_dets`` restriction to the owned rows.
    """
    hii, basis, helpers, integrals, x = _setup()
    expected = mult_spinless_torch(hii, x, basis, helpers, integrals)

    start, stop = row_range
    bounds = RankBounds(n=basis.dim, world_size=1, rank=0, start=start, stop=stop)
    monkeypatch.setattr(dsl, "FlatGather", _FullVectorGather)
    _FullVectorGather.full_vector = x

    op = dsl.LocalSpinlessOperator(
        hii[start:stop], basis, helpers, integrals, bounds, rank=0, world_size=1
    )
    got = op(x[start:stop])
    assert torch.allclose(got, expected[start:stop], atol=1e-12, rtol=0.0)


# ============================================================================
# 2. gloo: 2 and 3 ranks, uneven N
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


def _operator_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank's operator output equals the full-space matvec's slice."""
    _init_gloo(rank, world_size, init_file)
    try:
        hii, basis, helpers, integrals, x = _setup()
        bounds = partition_vector(basis.dim, world_size, rank)
        expected = mult_spinless_torch(hii, x, basis, helpers, integrals)
        op = dsl.LocalSpinlessOperator(
            hii[bounds.start : bounds.stop],
            basis,
            helpers,
            integrals,
            bounds,
            rank=rank,
            world_size=world_size,
        )
        got = op(x[bounds.start : bounds.stop])
        ref = expected[bounds.start : bounds.stop]
        assert got.shape == ref.shape
        assert torch.allclose(got, ref, atol=1e-12, rtol=0.0), (
            rank,
            world_size,
            float((got - ref).abs().max()),
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


def test_operator_gloo_two_ranks_uneven(tmp_path) -> None:
    """gloo, 2 ranks, 126/125 determinants: matches the full matvec's slices."""
    _skip_without_gloo()
    sizes = [partition_vector(N_DETS, 2, rank).local_size for rank in range(2)]
    assert len(set(sizes)) > 1
    _run_spawned(_operator_worker, 2, (), tmp_path, "spinless2")


def test_operator_gloo_three_ranks_uneven(tmp_path) -> None:
    """gloo, 3 ranks, 84/84/83 determinants: matches the full matvec's slices."""
    _skip_without_gloo()
    sizes = [partition_vector(N_DETS, 3, rank).local_size for rank in range(3)]
    assert len(set(sizes)) > 1
    _run_spawned(_operator_worker, 3, (), tmp_path, "spinless3")


# ============================================================================
# 3. Local diagonal
# ============================================================================


@pytest.mark.parametrize("row_range", ROW_RANGES)
@pytest.mark.parametrize("chunk_size", [2_000_000, 7, 1])
def test_diag_local_matches_slice(row_range: tuple[int, int], chunk_size: int) -> None:
    """``make_diag_terms_spinless_local`` equals the matching diagonal slice."""
    _, basis, _, integrals, _ = _setup()
    expected = make_diag_terms_spinless_torch(basis, integrals)

    start, stop = row_range
    got = dsl.make_diag_terms_spinless_local(
        basis.dets[start:stop],
        basis.n_spinorb,
        basis.n_particles,
        integrals,
        chunk_size=chunk_size,
    )
    assert got.shape == (stop - start,)
    assert got.dtype == expected.dtype
    assert torch.allclose(got, expected[start:stop], atol=1e-12, rtol=0.0)


def test_diag_local_real_dtype_matches_slice() -> None:
    """Real integrals return a real diagonal, matching the reference slice."""
    n_orb, n_elec = 7, 3
    hcore, eri = _real_spinorbital_integrals(n_orb, seed=202)
    strs = np.asarray(cistring.make_strings(range(n_orb), n_elec))
    basis = spinless_basis_from_array(strs, n_orb, n_elec)
    integrals = to_torch_integrals_spinless(hcore, eri, dtype=torch.float64)
    expected = make_diag_terms_spinless_torch(basis, integrals)
    assert expected.dtype == torch.float64

    got = dsl.make_diag_terms_spinless_local(
        basis.dets[5:30], basis.n_spinorb, basis.n_particles, integrals, chunk_size=5
    )
    assert got.dtype == torch.float64
    assert torch.allclose(got, expected[5:30], atol=1e-12, rtol=0.0)
