"""Tests for the distributed GDB (pair-mode) owner-computes operator.

Covers:

1. Single process: ``LocalPairOperator`` equals ``mult_pair_torch`` on the
   full-product H4 pair basis (N=36), and ``mult_pair_local_torch`` equals the
   corresponding slice of ``mult_pair_torch`` for several owned row ranges.
   Also checks that every ``pair_range``-restricted stream yields exactly the
   full stream's rows with ``k_bra`` in range.
2. gloo, 2 and 3 ranks with uneven blocks (a 35-pair basis): each rank's
   output block matches the full-space matvec to 1e-12.
3. gloo, 2 ranks: ``LocalPairSwapProjector`` matches the full-space
   ``(v + tau * v[k_swap]) / 2`` formula on each block.
4. Complex Hermitian integrals: the world_size=1 operator matches
   ``mult_pair_torch``, and both match a from-scratch dense CI oracle.
5. ``pair_basis_init_vector_local``: unit global norm, unit entry on the owner
   of ``seed_index``, single process and gloo 2 ranks.
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
from test_complex_matvec import (  # noqa: E402  (local test-module import)
    _bruteforce_dense_hamiltonian,
    _enumerate_dets,
    _random_complex_hermitian_integrals,
)
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.distributed_davidson import (
    DistributedVector,
    partition_vector,
)
from torch_sci_solver.torch_impl.distributed_pairs import (
    LocalPairOperator,
    LocalPairSwapProjector,
    pair_basis_init_vector_local,
)
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult_pairs import (
    make_diag_terms_pair_local,
    make_diag_terms_pair_torch,
    mult_pair_local_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import (
    pair_basis_from_arrays,
    pair_basis_from_full_product,
)

NORB = 4
M_STRINGS = 6
N_PAIRS = M_STRINGS * M_STRINGS  # full Cartesian product of the (2α, 2β) sector

# Row columns of each streamed table kind, in the order tests compare them.
_ROW_KEYS = {
    "asingles": ("k_bra", "k_ket", "cr0", "an0"),
    "adoubles": ("k_bra", "k_ket", "cr0", "cr1", "an0", "an1"),
    "bsingles": ("k_bra", "k_ket", "cr0", "an0"),
    "bdoubles": ("k_bra", "k_ket", "cr0", "cr1", "an0", "an1"),
    "ab_doubles": ("k_bra", "k_ket", "cr_a", "an_a", "cr_b", "an_b"),
}


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _synthetic_h4():
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
    strings = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(NORB), 2)),
        dtype=np.int64,
    )
    return h1, eri, strings


def _pair_setup(n_drop: int = 0) -> tuple:
    """Real H4 fixture: pair basis, helpers, integrals and ``hii``.

    ``n_drop`` removes that many pairs from the end of the full product, which
    makes N=36-``n_drop`` unevenly divisible by 2 and 3 ranks.
    """
    h1, eri, strings = _synthetic_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu")
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    if n_drop:
        pairs_a, pairs_b = pairs_a[:-n_drop], pairs_b[:-n_drop]
    pair_basis = pair_basis_from_arrays(pairs_a, pairs_b, device="cpu")
    helpers = generate_pair_helpers(pair_basis, NORB, device="cpu")
    hii = make_diag_terms_pair_torch(pair_basis, NORB, integrals)
    return pair_basis, helpers, integrals, hii


def _block_pair_setup(n_drop: int = 0) -> tuple:
    """The H4 fixture with the exact Cartesian block decomposition enabled."""
    pair_basis, _, integrals, hii = _pair_setup(n_drop)
    helpers = generate_pair_helpers(pair_basis, NORB, device="cpu", block_matvec="on")
    assert helpers.blocks is not None
    return pair_basis, helpers, integrals, hii


def _random_vector(n: int, seed: int = 7) -> torch.Tensor:
    """Deterministic real random vector."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=gen, dtype=torch.float64)


def _random_complex_vector(n: int, seed: int = 7) -> torch.Tensor:
    """Deterministic complex random vector."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.complex(real, imag)


def _stream_rows(stream, keys: tuple[str, ...]) -> torch.Tensor:
    """Stack one streamed table's rows into an ``(n_rows, len(keys))`` tensor."""
    parts = [torch.stack([t[k] for k in keys], dim=1) for t in stream]
    if not parts:
        return torch.zeros((0, len(keys)), dtype=torch.int64)
    return torch.cat(parts, dim=0)


def _sorted_rows(rows: torch.Tensor) -> list[tuple[int, ...]]:
    """Row tuples sorted for order-independent set comparison."""
    return sorted(tuple(int(v) for v in row) for row in rows)


# ============================================================================
# 1. Single process
# ============================================================================


def test_local_operator_single_process_matches_full() -> None:
    """world_size=1: the operator reproduces ``mult_pair_torch`` exactly."""
    pair_basis, helpers, integrals, hii = _pair_setup()
    n = pair_basis.n_pairs
    assert n == N_PAIRS
    bounds = partition_vector(n, 1, 0)
    op = LocalPairOperator(hii, pair_basis, helpers, NORB, integrals, bounds, rank=0, world_size=1)
    for seed in (7, 8, 9):
        x = _random_vector(n, seed)
        expected = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)
        assert torch.allclose(op(x), expected, atol=1e-12, rtol=0.0)


def test_local_operator_move_cache_matches_and_reuses() -> None:
    """A cached block-path operator matches the uncached matvec and reuses tables."""
    pair_basis, helpers, integrals, hii = _block_pair_setup()
    n = pair_basis.n_pairs
    bounds = partition_vector(n, 1, 0)
    x = _random_vector(n, 9)
    expected = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)

    plain = LocalPairOperator(
        hii, pair_basis, helpers, NORB, integrals, bounds, rank=0, world_size=1
    )
    assert plain._move_cache is None

    cached = LocalPairOperator(
        hii,
        pair_basis,
        helpers,
        NORB,
        integrals,
        bounds,
        rank=0,
        world_size=1,
        move_cache_bytes=1 << 20,
    )
    cache = cached._move_cache
    assert cache is not None and cache.evict
    assert cache.budget_bytes == 1 << 20
    first = cached(x)
    snapshot = (len(cache.entries), cache.used_bytes)
    assert snapshot[0] > 0
    assert 0 < snapshot[1] <= cache.budget_bytes
    second = cached(x)
    assert torch.equal(first, second)
    assert (len(cache.entries), cache.used_bytes) == snapshot
    assert torch.allclose(first, expected, atol=1e-12, rtol=0.0)
    assert torch.allclose(plain(x), expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("budget", [0, 1, 1 << 20])
def test_local_operator_move_cache_budget(budget: int) -> None:
    """Budget 0 disables the cache; a one-byte cap retains no table."""
    pair_basis, helpers, integrals, hii = _block_pair_setup()
    n = pair_basis.n_pairs
    bounds = partition_vector(n, 1, 0)
    x = _random_vector(n, 15)
    expected = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)
    op = LocalPairOperator(
        hii,
        pair_basis,
        helpers,
        NORB,
        integrals,
        bounds,
        rank=0,
        world_size=1,
        move_cache_bytes=budget,
    )
    assert torch.allclose(op(x), expected, atol=1e-12, rtol=0.0)
    if budget == 0:
        assert op._move_cache is None
    else:
        cache = op._move_cache
        assert cache is not None
        assert cache.used_bytes <= budget
        if budget == 1:
            assert cache.used_bytes == 0


def test_local_operator_move_cache_rejects_negative() -> None:
    """A negative byte budget is rejected at construction."""
    pair_basis, helpers, integrals, hii = _block_pair_setup()
    bounds = partition_vector(pair_basis.n_pairs, 1, 0)
    with pytest.raises(ValueError, match="move_cache_bytes"):
        LocalPairOperator(
            hii,
            pair_basis,
            helpers,
            NORB,
            integrals,
            bounds,
            rank=0,
            world_size=1,
            move_cache_bytes=-1,
        )


@pytest.mark.parametrize(
    "row_start,n_local",
    [(0, 6), (5, 12), (12, 24), (0, 36), (18, 18), (36, 0)],
)
def test_local_matvec_matches_full_slice(row_start: int, n_local: int) -> None:
    """A local block reproduces the corresponding slice of the full matvec."""
    pair_basis, helpers, integrals, hii = _pair_setup()
    n = pair_basis.n_pairs
    x = _random_vector(n, 11)
    expected = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)

    row_stop = row_start + n_local
    hii_local = make_diag_terms_pair_local(
        pair_basis.alpha_of_pair[row_start:row_stop],
        pair_basis.beta_of_pair[row_start:row_stop],
        NORB,
        integrals,
    )
    assert torch.allclose(hii_local, hii[row_start:row_stop], atol=1e-12, rtol=0.0)
    got = mult_pair_local_torch(
        hii_local,
        x[row_start:row_stop],
        x,
        pair_basis,
        helpers,
        NORB,
        integrals,
        row_start,
    )
    assert torch.allclose(got, expected[row_start:row_stop], atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("name", list(_ROW_KEYS))
def test_pair_range_streams_are_exact(name: str) -> None:
    """``pair_range`` streams exactly the full stream's rows inside the range."""
    _, helpers, _, _ = _pair_setup()
    keys = _ROW_KEYS[name]
    stream = getattr(helpers, name)
    full = _sorted_rows(_stream_rows(stream(), keys))
    for start, stop in [(0, 0), (0, N_PAIRS), (11, 29), (0, 12), (30, N_PAIRS)]:
        part = _sorted_rows(_stream_rows(stream(pair_range=(start, stop)), keys))
        expected = [row for row in full if start <= row[0] < stop]
        assert part == expected, (name, start, stop)


def test_pair_range_rejects_out_of_bounds() -> None:
    """A range outside the pair set fails loudly."""
    _, helpers, _, _ = _pair_setup()
    with pytest.raises(ValueError, match="out of bounds"):
        for _chunk in helpers.ab_doubles(pair_range=(0, N_PAIRS + 1)):
            pass
    with pytest.raises(ValueError, match="out of bounds"):
        for _chunk in helpers.asingles(pair_range=(10, 5)):
            pass


# ============================================================================
# 2. gloo: operator on 2 and 3 ranks with uneven blocks
# ============================================================================


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _operator_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank's output block equals the full-space matvec's slice."""
    _init_gloo(rank, world_size, init_file)
    try:
        pair_basis, helpers, integrals, _ = _pair_setup(n_drop=1)
        n = pair_basis.n_pairs
        assert n == N_PAIRS - 1  # 35: uneven for both 2 and 3 ranks
        bounds = partition_vector(n, world_size, rank)
        hii_local = make_diag_terms_pair_local(
            pair_basis.alpha_of_pair[bounds.start : bounds.stop],
            pair_basis.beta_of_pair[bounds.start : bounds.stop],
            NORB,
            integrals,
        )
        op = LocalPairOperator(
            hii_local,
            pair_basis,
            helpers,
            NORB,
            integrals,
            bounds,
            rank=rank,
            world_size=world_size,
        )
        x = _random_vector(n)
        hii = make_diag_terms_pair_torch(pair_basis, NORB, integrals)
        expected = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)

        sl = slice(bounds.start, bounds.stop)
        got = op(x[sl])
        assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
            rank,
            world_size,
            float((got - expected[sl]).abs().max()),
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
    """gloo, 2 ranks, 18/17 pairs: matches the full-vector matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_operator_worker, 2, (), tmp_path, "pair_op2")


def test_operator_gloo_three_ranks_uneven(tmp_path) -> None:
    """gloo, 3 ranks, 12/12/11 pairs: matches the full-vector matvec to 1e-12."""
    _skip_without_gloo()
    _run_spawned(_operator_worker, 3, (), tmp_path, "pair_op3")


# ============================================================================
# 3. gloo: α↔β-swap projector
# ============================================================================


def _projector_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """The local projector matches the full-space pair formula on each block."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _pair_ab_swap_permutation

    _init_gloo(rank, world_size, init_file)
    try:
        pair_basis, _, _, _ = _pair_setup()
        n = pair_basis.n_pairs
        full_k_swap, full_tau = _pair_ab_swap_permutation(
            pair_basis, NORB, torch.device("cpu"), torch.float64
        )
        assert full_k_swap is not None and full_tau is not None
        v = _random_vector(n)
        expected = 0.5 * (v + full_tau * v[full_k_swap])

        bounds = partition_vector(n, world_size, rank)
        sl = slice(bounds.start, bounds.stop)
        # The local path holds only the owned slice of the swap map.
        k_swap, tau = _pair_ab_swap_permutation(
            pair_basis,
            NORB,
            torch.device("cpu"),
            torch.float64,
            pair_range=(bounds.start, bounds.stop),
        )
        assert k_swap is not None and tau is not None
        assert torch.equal(k_swap, full_k_swap[sl])
        assert torch.equal(tau, full_tau[sl])
        projector = LocalPairSwapProjector(
            pair_basis, k_swap, tau, bounds, rank=rank, world_size=world_size
        )
        got = projector(expected[sl])
        assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
            rank,
            float((got - expected[sl]).abs().max()),
        )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_local_projector_gloo_two_ranks(tmp_path) -> None:
    """gloo, 2 ranks: the local projector equals the full-space formula."""
    _skip_without_gloo()
    _run_spawned(_projector_worker, 2, (), tmp_path, "pair_proj")


# ============================================================================
# 4. Complex Hermitian integrals
# ============================================================================


def test_complex_local_operator_matches_full_and_oracle() -> None:
    """Complex: world_size=1 operator == ``mult_pair_torch`` == dense oracle."""
    norb, n_a, n_b = 4, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=1)
    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
    pair_basis = pair_basis_from_full_product(adets, bdets, device="cpu")
    helpers = generate_pair_helpers(pair_basis, norb, device="cpu")
    hii = make_diag_terms_pair_torch(pair_basis, norb, integrals)
    assert not hii.dtype.is_complex, "hii must stay real for complex integrals"

    n = pair_basis.n_pairs
    bounds = partition_vector(n, 1, 0)
    op = LocalPairOperator(hii, pair_basis, helpers, norb, integrals, bounds, rank=0, world_size=1)
    x = _random_complex_vector(n)
    full = mult_pair_torch(hii, x, pair_basis, helpers, norb, integrals)

    H_oracle, full_dets = _bruteforce_dense_hamiltonian(hcore, eri, adets, bdets, norb)
    assert len(full_dets) == n
    np.testing.assert_allclose(full.numpy(), H_oracle @ x.numpy(), atol=1e-8)
    assert torch.allclose(op(x), full, atol=1e-12, rtol=0.0)


# ============================================================================
# 5. Cold-start initial vector
# ============================================================================


def test_init_vector_single_process() -> None:
    """world_size=1: unit global norm and the unit seed entry at ``seed_index``."""
    bounds = partition_vector(N_PAIRS, 1, 0)
    for seed in (0, 17, N_PAIRS - 1):
        v = pair_basis_init_vector_local(
            N_PAIRS,
            bounds,
            seed_index=seed,
            device=torch.device("cpu"),
            dtype=torch.float64,
            rank=0,
            world_size=1,
        )
        assert float(torch.linalg.norm(v)) == pytest.approx(1.0, abs=1e-12)
        assert abs(float(v[seed])) == pytest.approx(1.0, abs=1e-6)
        assert int(v.abs().argmax()) == seed


def _init_vector_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every seed lands on its owner; the assembled vector has unit norm."""
    _init_gloo(rank, world_size, init_file)
    try:
        n = N_PAIRS - 1  # 35: uneven ownership
        bounds = partition_vector(n, world_size, rank)
        for seed in (0, 17, n - 1):
            v = pair_basis_init_vector_local(
                n,
                bounds,
                seed_index=seed,
                device=torch.device("cpu"),
                dtype=torch.float64,
                rank=rank,
                world_size=world_size,
            )
            full = DistributedVector(
                v,
                n,
                rank=rank,
                world_size=world_size,
                bounds=bounds,
            ).to_global()
            assert float(torch.linalg.norm(full)) == pytest.approx(1.0, abs=1e-12)
            assert abs(float(full[seed])) == pytest.approx(1.0, abs=1e-6)
            assert int(full.abs().argmax()) == seed
            if bounds.owns(seed):
                local_entry = v[seed - bounds.start]
                assert abs(float(local_entry)) == pytest.approx(1.0, abs=1e-6)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_init_vector_gloo_two_ranks(tmp_path) -> None:
    """gloo, 2 ranks: unit norm and the unit entry on the seed's owner."""
    _skip_without_gloo()
    _run_spawned(_init_vector_worker, 2, (), tmp_path, "pair_init")
