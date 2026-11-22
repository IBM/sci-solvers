"""Multi-GPU dispatch smoke tests for TorchSCISolver.

Skipped when fewer than 2 CUDA devices are visible. On a 2+ GPU host these
verify that the ``devices=[...]`` fan-out preserves per-subspace results
(bit-exact against the single-GPU path), preserves *all* roots when
``n_roots > 1``, correctly accumulates timing stats across the per-GPU worker
processes, propagates worker errors, and rejects malformed device lists at
construction time.

**Why each GPU-touching test runs in its own subprocess.** The multi-GPU path
spawns one worker *process* per GPU. On clusters where GPUs are in
``Exclusive_Process`` compute mode, only one process may hold a context on a
given GPU — so if the pytest process itself ran a single-GPU solve on cuda:0
(initializing a context there), the multi-GPU worker for cuda:0 could never
start. We therefore run every GPU phase inside a fresh spawned process via
``run_in_spawned_process`` (shared with the benchmark); each child exits and
releases its contexts before the next, keeping the pytest process CUDA-free.
That also means the single-GPU reference and the multi-GPU run are separate
top-level children — no nested spawning needed.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import time

import numpy as np
import pytest
import torch

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from torch_sci_solver import make_torch_sci_solver  # noqa: E402
from torch_sci_solver.multi_gpu import run_in_spawned_process  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-GPU tests require >=2 CUDA devices",
)


# --------------------------------------------------------------------------- #
# Fixtures (H4 full-CI — tiny, complete, deterministic)                        #
# --------------------------------------------------------------------------- #
def _h4_integrals(spacing: float = 0.74):
    """RHF integrals + exact FCI energy for a 4-atom H chain."""
    atoms = "; ".join(f"H 0 0 {i * spacing}" for i in range(4))
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return h1, eri, mol.energy_nuc(), norb, mol.nelec, fci.FCI(mf).kernel()[0]


def _full_ci_strings(norb: int, nelec: tuple[int, int]):
    """All CI strings for the full Hilbert space (small systems only)."""

    def strings(n_elec: int) -> np.ndarray:
        out: list[int] = []
        for occ in itertools.combinations(range(norb), n_elec):
            s = 0
            for o in occ:
                s |= 1 << o
            out.append(s)
        return np.array(sorted(out), dtype=np.int64)

    return strings(nelec[0]), strings(nelec[1])


def _repeat_batches(ci_a: np.ndarray, ci_b: np.ndarray, n: int):
    """A list of ``n`` identical ``(ci_a, ci_b)`` subspaces — identical inputs
    make every per-subspace result identical, so any single-vs-multi-GPU
    discrepancy is a dispatch bug, not an input difference."""
    return [(ci_a, ci_b) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Child bodies (run in a spawned process via run_in_spawned_process)           #
# --------------------------------------------------------------------------- #
def _solve_energies(devices, n_subspaces):
    """Solve ``n_subspaces`` identical H4 full-CI subspaces on ``devices``;
    return (energies, e_core, e_fci). Runs in its own process."""
    h1, eri, e_core, norb, nelec, e_fci = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    batches = _repeat_batches(ci_a, ci_b, n_subspaces)
    res = make_torch_sci_solver(devices=devices)(batches, h1, eri, norb, nelec)
    # solver(...) returns list[EigenStates]; ground state is at states[0].
    # Unwrap to bare floats here rather than in the parent: only picklable,
    # CUDA-free values may cross the process boundary.
    return [r.states[0].energy for r in res], e_core, e_fci


def _solve_stats(devices, n_subspaces):
    """Solve on ``devices``; return (n_results, picklable stats subset)."""
    h1, eri, _, norb, nelec, _ = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    solver = make_torch_sci_solver(devices=devices)
    results = solver(_repeat_batches(ci_a, ci_b, n_subspaces), h1, eri, norb, nelec)
    s = solver.stats
    return len(results), {
        "n_subspaces": s["n_subspaces"],
        "spin_squared_len": len(s["spin_squared"]),
        "t_setup": s["t_setup"],
        "t_davidson": s["t_davidson"],
        "t_occ": s["t_occ"],
    }


def _time_solve(devices, n_subspaces):
    """Warm up then time a solve on ``devices``; return the timed wall seconds.
    One solver instance for warm-up + timed run (so a second pool is never
    started while the first still holds the exclusive-mode GPUs)."""
    h1, eri, _, norb, nelec, _ = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    batches = _repeat_batches(ci_a, ci_b, n_subspaces)
    solver = make_torch_sci_solver(devices=devices)
    solver(batches[:1], h1, eri, norb, nelec)  # warm-up (pool start + integrals)
    t0 = time.perf_counter()
    solver(batches, h1, eri, norb, nelec)
    dt = time.perf_counter() - t0
    solver.close()
    return dt


def _solve_all_roots(devices, n_subspaces, n_roots):
    """Solve ``n_subspaces`` identical H4 subspaces with ``n_roots`` roots each on
    ``devices``; return the per-subspace list of ALL root energies.

    Returns every root (not just the ground state) so the caller can verify the
    excited roots actually survive the dispatch path.
    """
    h1, eri, e_core, norb, nelec, _ = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    batches = _repeat_batches(ci_a, ci_b, n_subspaces)
    solver = make_torch_sci_solver(devices=devices, n_roots=n_roots, davidson_block_size=20)
    res = solver(batches, h1, eri, norb, nelec)
    solver.close()
    # Unwrap child-side: only picklable, CUDA-free values cross the boundary.
    return [[s.energy for s in r.states] for r in res], e_core


def _solve_bad_basis(devices):
    """Feed a β-string with the wrong popcount at sub_idx=1; the worker's
    ValueError must propagate back through the pool."""
    h1, eri, _, norb, nelec, _ = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    _, n_b = nelec
    bad_b = np.array([(1 << (n_b + 1)) - 1], dtype=np.int64)  # popcount n_b+1 — invalid
    batches = [(ci_a, ci_b), (ci_a, bad_b), (ci_a, ci_b)]
    make_torch_sci_solver(devices=devices)(batches, h1, eri, norb, nelec)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_multi_gpu_matches_single_gpu_h4_bitexact():
    """4 identical subspaces on 2 GPUs must produce the same per-subspace
    energies as the single-GPU path (and each must recover exact FCI).

    Tolerate float64 round-off in the CUDA reductions (~1e-11) rather than byte
    equality — CUBLAS/CUB reduction order can differ deterministically across
    devices. The single-GPU and multi-GPU solves run as separate spawned
    processes so neither holds cuda:0 while the other's pool starts.
    """
    ref_e, e_core, e_fci = run_in_spawned_process(_solve_energies, ["cuda:0"], 4)
    got_e, _, _ = run_in_spawned_process(_solve_energies, ["cuda:0", "cuda:1"], 4)

    # ``_solve_energies`` unwrapped EigenStates -> ground-state floats child-side,
    # so these are plain energies.
    assert len(ref_e) == len(got_e) == 4
    for i, (a, b) in enumerate(zip(ref_e, got_e)):
        assert a == pytest.approx(b, abs=1e-10), (
            f"subspace {i}: single-GPU {a!r} vs multi-GPU {b!r}"
        )
        # Every subspace must recover FCI on this complete basis.
        assert b + e_core == pytest.approx(e_fci, abs=1e-9)


def test_multi_gpu_preserves_all_roots():
    """n_roots > 1 must survive the multi-GPU dispatch: every subspace comes back
    with all ``n_roots`` roots, matching the single-GPU path root-for-root.

    Workers are separate processes rebuilt from ``_worker_cfg()``, so a knob
    missing from that dict reverts to its default in the child — dropping
    ``n_roots`` would silently return ground states only, while every assertion
    that reads ``states[0]`` still passed. Hence the length check on ``.states``.
    """
    ref, e_core = run_in_spawned_process(_solve_all_roots, ["cuda:0"], 4, 2)
    got, _ = run_in_spawned_process(_solve_all_roots, ["cuda:0", "cuda:1"], 4, 2)

    assert len(ref) == len(got) == 4
    for i, (a, b) in enumerate(zip(ref, got)):
        # The load-bearing assertion: excited roots must not be silently dropped.
        assert len(b) == 2, (
            f"subspace {i}: multi-GPU returned {len(b)} root(s), expected 2 — "
            "excited roots lost in the worker dispatch (is n_roots in _worker_cfg?)"
        )
        assert len(a) == 2, f"subspace {i}: single-GPU returned {len(a)} root(s)"
        for k, (ea, eb) in enumerate(zip(a, b)):
            assert ea == pytest.approx(eb, abs=1e-10), (
                f"subspace {i} root {k}: single-GPU {ea!r} vs multi-GPU {eb!r}"
            )
        # Roots must come back ascending in energy.
        assert b[1] >= b[0] - 1e-8, f"subspace {i}: root 1 ({b[1]}) below root 0 ({b[0]})"


def test_multi_gpu_stats_totals():
    """Per-component stats accumulate correctly across the pool workers: six
    subspaces ⇒ n_subspaces==6, spin_squared has 6 entries, and every timing
    accumulator is strictly positive (no missed increment)."""
    n_results, stats = run_in_spawned_process(_solve_stats, ["cuda:0", "cuda:1"], 6)
    assert n_results == 6
    assert stats["n_subspaces"] == 6
    assert stats["spin_squared_len"] == 6
    assert stats["t_setup"] > 0.0
    assert stats["t_davidson"] > 0.0
    assert stats["t_occ"] > 0.0


def test_multi_gpu_speedup_not_worse():
    """Guard against gross regression only — on tiny H4 each Davidson is sub-ms
    so pool overhead dominates; real speedup is measured by the multi-GPU benchmarks.
    Single and multi are timed in separate processes."""
    t_single = run_in_spawned_process(_time_solve, ["cuda:0"], 8)
    t_multi = run_in_spawned_process(_time_solve, ["cuda:0", "cuda:1"], 8)
    assert t_multi < 3.0 * t_single, (
        f"multi-GPU dispatch unexpectedly slow: t_single={t_single:.3f}s, t_multi={t_multi:.3f}s"
    )


def test_multi_gpu_error_propagates():
    """A malformed CI string in one subspace must abort the batch: the worker's
    ValueError propagates through the pool and is re-raised in the parent."""
    with pytest.raises(ValueError, match="invalid β-strings"):
        run_in_spawned_process(_solve_bad_basis, ["cuda:0", "cuda:1"])


def test_multi_gpu_validation_rejects_bad_devices():
    """The devices= kwarg validation catches the common footguns (no GPU work,
    so this runs in-process)."""
    with pytest.raises(ValueError, match="not allowed"):
        make_torch_sci_solver(devices=[])
    with pytest.raises(ValueError, match="duplicates"):
        make_torch_sci_solver(devices=["cuda:0", "cuda:0"])
    with pytest.raises(ValueError, match="CUDA"):
        make_torch_sci_solver(devices=["cpu", "cuda:0"])
    with pytest.raises(ValueError, match="index"):
        make_torch_sci_solver(devices=["cuda", "cuda:1"])
    if torch.cuda.device_count() < 8:
        with pytest.raises(ValueError, match="out of range"):
            make_torch_sci_solver(devices=["cuda:0", "cuda:7"])


def test_multi_gpu_length_one_is_sequential():
    """devices=['cuda:0'] must match device='cuda:0' (a length-1 list takes the
    single-device path — no pool spawned). Both run in one child; the
    single-device path uses no worker process, so there's no exclusive-mode
    contention between them."""
    ref_e, got_e, e_core, e_fci = run_in_spawned_process(_length_one_body)
    assert ref_e == pytest.approx(got_e, abs=1e-10)
    assert got_e + e_core == pytest.approx(e_fci, abs=1e-9)


def _length_one_body():
    """Child: device='cuda:0' vs devices=['cuda:0'] on one full-CI subspace."""
    h1, eri, e_core, norb, nelec, e_fci = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)
    ref = make_torch_sci_solver(device="cuda:0")([(ci_a, ci_b)], h1, eri, norb, nelec)
    got = make_torch_sci_solver(devices=["cuda:0"])([(ci_a, ci_b)], h1, eri, norb, nelec)
    # solver(...) returns list[EigenStates]; ground state is at states[0].
    # Assert in the parent, not here: run_in_spawned_process only rebuilds
    # ValueError/TypeError/RuntimeError, so a child-side AssertionError arrives as
    # a RuntimeError and loses pytest's assertion rewriting (which prints both
    # energies).
    return ref[0].states[0].energy, got[0].states[0].energy, e_core, e_fci
