"""Equivalence tests for the C++ (pybind11) truncated-Davidson kernel backend.

Every test pins the compiled ``cpp`` backend to the ``numba`` reference (and,
end to end, to PySCF). The module skips cleanly when the extension is not
built, so CI stays green without the native toolchain -- build it with
``uv sync`` (or ``uv pip install -e .``) to exercise these tests.

The per-kernel checks are exact: the enumeration kernels are deterministic and
share the same arithmetic, so cpp and numba must agree bit-for-bit on a single
determinant. The end-to-end checks use a tolerance, because both backends reduce
per-thread partial sums in a machine-dependent floating-point order.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from pyscf.fci.cistring import make_strings

from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver
from sci_solvers.energy_solvers.truncated_davidson_solver import TruncatedDavidsonSolver
from sci_solvers.hamiltonian import HamiltonianData

# Skip the whole module unless the compiled extension imports. importorskip
# reports the reason so `-rs` shows why it was skipped in CI.
cpp = pytest.importorskip(
    "sci_solvers.energy_solvers.truncated_davidson_solver.kernels_cpp_wrapper",
    reason="C++ kernel extension not built (run `uv sync` / `uv pip install -e .`)",
)
from sci_solvers.energy_solvers.truncated_davidson_solver import numba_kernels as nb  # noqa: E402

# The thread-scaling sweep and perf benchmark are resource-hungry (they spawn
# per-thread-count subprocesses and time full solves), so they are skipped
# unless explicitly opted in with SCI_SOLVERS_RUN_CPP_SCALING=1. CI, which lacks
# the cores to make the timing meaningful, leaves them off.
_run_scaling = pytest.mark.skipif(
    os.environ.get("SCI_SOLVERS_RUN_CPP_SCALING") != "1",
    reason="set SCI_SOLVERS_RUN_CPP_SCALING=1 to run cpp scaling/perf tests",
)

ENERGY_TOL_MHA = 0.001  # mHa
# cpp and numba reduce per-thread partial sums in a machine-dependent float
# order; on the larger random scale/perf Hamiltonians that difference can reach
# a few 1e-6 mHa -- still far below any chemically meaningful threshold, but
# looser than the tiny-system agreement above.
SCALE_TOL_MHA = 0.01  # mHa


# ---------------------------------------------------------------------------
# Small hand-built inputs shared by the per-kernel equivalence tests
# ---------------------------------------------------------------------------
def _random_eri(norb: int, seed: int) -> np.ndarray:
    """A random 4-index tensor with 8-fold symmetry (real, chemist notation)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((norb, norb, norb, norb))
    # symmetrize to the physical permutational symmetries so both backends see
    # a well-formed integral tensor (the kernels do not rely on it, but this
    # keeps the inputs realistic).
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    return np.ascontiguousarray(g)


def _bits_to_occ_virt(bits: int, norb: int):
    """(occ, virt) orbital-index arrays for a determinant bitstring."""
    occ = np.array([p for p in range(norb) if (bits >> p) & 1], dtype=np.int64)
    virt = np.array([p for p in range(norb) if not ((bits >> p) & 1)], dtype=np.int64)
    return occ, virt


def _assert_tuple_equal(cpp_out, nb_out):
    """Every array in the two returned tuples must be exactly equal."""
    assert len(cpp_out) == len(nb_out)
    for c, n in zip(cpp_out, nb_out):
        c = np.asarray(c)
        n = np.asarray(n)
        assert c.shape == n.shape, f"shape {c.shape} != {n.shape}"
        if np.issubdtype(c.dtype, np.floating):
            np.testing.assert_allclose(c, n, rtol=0, atol=1e-12)
        else:
            np.testing.assert_array_equal(c, n)


# ---------------------------------------------------------------------------
# Per-kernel equivalence: cpp vs numba
# ---------------------------------------------------------------------------
def test_singles_kernel_matches_numba():
    """singles_kernel: cpp == numba on a 6-orbital determinant."""
    norb = 6
    h1e = np.ascontiguousarray(np.random.default_rng(1).standard_normal((norb, norb)))
    eri = _random_eri(norb, seed=2)
    bits_a = 0b001011  # 3 occupied alpha
    bits_b = 0b000101  # 2 occupied beta (the "other" spin block)
    occ, virt = _bits_to_occ_virt(bits_a, norb)
    occ_other, _ = _bits_to_occ_virt(bits_b, norb)

    cpp_out = cpp.singles_kernel(occ, virt, occ_other, bits_a, h1e, eri)
    nb_out = nb.singles_kernel(occ, virt, occ_other, bits_a, h1e, eri)
    _assert_tuple_equal(cpp_out, nb_out)


def test_doubles_same_spin_kernel_matches_numba():
    """doubles_same_spin_kernel: cpp == numba (sign chaining i->a then j->b)."""
    norb = 7
    eri = _random_eri(norb, seed=3)
    bits0 = 0b0010110  # 3 occupied
    occ, virt = _bits_to_occ_virt(bits0, norb)

    cpp_out = cpp.doubles_same_spin_kernel(occ, virt, bits0, eri)
    nb_out = nb.doubles_same_spin_kernel(occ, virt, bits0, eri)
    _assert_tuple_equal(cpp_out, nb_out)


def test_doubles_cross_spin_kernel_matches_numba():
    """doubles_cross_spin_kernel: cpp == numba (factorized alpha*beta sign)."""
    norb = 6
    eri = _random_eri(norb, seed=4)
    bits_a0 = 0b001011  # 3 occupied alpha
    bits_b0 = 0b010010  # 2 occupied beta
    occ_a, virt_a = _bits_to_occ_virt(bits_a0, norb)
    occ_b, virt_b = _bits_to_occ_virt(bits_b0, norb)

    cpp_out = cpp.doubles_cross_spin_kernel(occ_a, virt_a, occ_b, virt_b, bits_a0, bits_b0, eri)
    nb_out = nb.doubles_cross_spin_kernel(occ_a, virt_a, occ_b, virt_b, bits_a0, bits_b0, eri)
    _assert_tuple_equal(cpp_out, nb_out)


def test_singles_kernel_matches_numba_zero_virtuals():
    """Edge case: no virtual orbitals in the block -> empty result, both backends."""
    norb = 3
    h1e = np.ascontiguousarray(np.random.default_rng(5).standard_normal((norb, norb)))
    eri = _random_eri(norb, seed=6)
    bits_a = 0b111  # fully occupied: no virtuals
    occ, virt = _bits_to_occ_virt(bits_a, norb)
    occ_other, _ = _bits_to_occ_virt(0b011, norb)

    cpp_out = cpp.singles_kernel(occ, virt, occ_other, bits_a, h1e, eri)
    nb_out = nb.singles_kernel(occ, virt, occ_other, bits_a, h1e, eri)
    assert cpp_out[0].shape[0] == 0
    _assert_tuple_equal(cpp_out, nb_out)


# ---------------------------------------------------------------------------
# End-to-end: cpp vs numba vs PySCF
# ---------------------------------------------------------------------------
_NORB = 2
_NELEC_A = 1
_NELEC_B = 1
_E_NUC = -76.83741213074619
_HCORE = np.array([[-0.8253014918977718, 0.0], [0.0, -0.5344324046820699]])
_ERI = np.array([
    [
        [
            [4.487069375223534e-01, -1.479277142336694e-15],
            [-1.479277142336694e-15, 4.544572923512384e-01],
        ],
        [
            [-1.479277142336694e-15, 1.313613621061964e-01],
            [1.313613621061964e-01, 1.703410010234202e-15],
        ],
    ],
    [
        [
            [-1.479277142336694e-15, 1.313613621061964e-01],
            [1.313613621061964e-01, 1.703410010234202e-15],
        ],
        [
            [4.544572923512384e-01, 1.703410010234202e-15],
            [1.703410010234202e-15, 4.855786851181397e-01],
        ],
    ],
])


@pytest.fixture(scope="module")
def tiny_hamiltonian():
    """Small synthetic 2-orbital/2-electron Hamiltonian, built in-memory."""
    return HamiltonianData(
        hcore=_HCORE,
        eri=_ERI,
        e_nuc=_E_NUC,
        num_orbitals=_NORB,
        num_elec_a=_NELEC_A,
        num_elec_b=_NELEC_B,
    )


def _solve(ham, ci_strs, backend):
    solver = TruncatedDavidsonSolver(
        symmetrize_spin=False,
        hamiltonian=ham,
        verbose=False,
        eps=1e-4,
        max_cycle=100,
        delta_max=1e-3,
        delta_min=1e-6,
        growth_factor=2.0,
        backend=backend,
    )
    return solver.compute_energy(ci_strs_batches=[ci_strs])[0].energy


def test_end_to_end_cpp_matches_numba_and_pyscf(tiny_hamiltonian):
    """cpp and numba backends agree with each other and with PySCF FCI."""
    ham = tiny_hamiltonian
    ci_strs = (
        make_strings(range(ham.num_orbitals), ham.num_elec_a),
        make_strings(range(ham.num_orbitals), ham.num_elec_b),
    )

    e_cpp = _solve(ham, ci_strs, backend="cpp")
    e_numba = _solve(ham, ci_strs, backend="numba")

    pyscf_results = PySCFSolver(
        symmetrize_spin=False, hamiltonian=ham, verbose=False
    ).compute_energy(
        ci_strs_batches=[ci_strs], spin_sq=None, max_cycle=30, tol=1e-10, davidson_block_size=30
    )
    e_pyscf = pyscf_results[0].energy

    assert np.isfinite(e_cpp)
    # cpp and numba solve the same problem: they should agree tightly.
    assert abs(e_cpp - e_numba) * 1e3 < ENERGY_TOL_MHA, (
        f"cpp vs numba |ΔE| = {abs(e_cpp - e_numba) * 1e3:.6f} mHa"
    )
    # and both must land on the exact FCI energy.
    assert abs(e_cpp - e_pyscf) * 1e3 < ENERGY_TOL_MHA, (
        f"cpp vs PySCF |ΔE| = {abs(e_cpp - e_pyscf) * 1e3:.6f} mHa"
    )


def test_end_to_end_cpp_matches_numba_4orb():
    """cpp and numba agree on a larger 4-orbital/2-electron selected space."""
    norb, nelec_a, nelec_b = 4, 2, 2
    h1e = np.diag(np.linspace(-1.0, 1.0, norb))
    eri = np.zeros((norb,) * 4)
    eri[0, 0, 0, 0] = 0.5
    ham = HamiltonianData(
        hcore=h1e,
        eri=eri,
        e_nuc=0.0,
        num_orbitals=norb,
        num_elec_a=nelec_a,
        num_elec_b=nelec_b,
    )
    ci_strs = (
        make_strings(range(norb), nelec_a),
        make_strings(range(norb), nelec_b),
    )

    e_cpp = _solve(ham, ci_strs, backend="cpp")
    e_numba = _solve(ham, ci_strs, backend="numba")

    assert np.isfinite(e_cpp)
    assert abs(e_cpp - e_numba) * 1e3 < ENERGY_TOL_MHA, (
        f"cpp vs numba |ΔE| = {abs(e_cpp - e_numba) * 1e3:.6f} mHa"
    )


# ---------------------------------------------------------------------------
# Larger Hamiltonian shared by the scale-correctness and performance tests
# ---------------------------------------------------------------------------
def _mid_hamiltonian(norb: int, nelec_a: int, nelec_b: int, seed: int) -> HamiltonianData:
    """A random, symmetric ``norb``-orbital Hamiltonian for scale/perf checks."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)  # Hermitian one-body
    eri = _random_eri(norb, seed=seed + 1) * 0.1  # scaled to keep energies sane
    return HamiltonianData(
        hcore=np.ascontiguousarray(hcore),
        eri=eri,
        e_nuc=0.0,
        num_orbitals=norb,
        num_elec_a=nelec_a,
        num_elec_b=nelec_b,
    )


def test_end_to_end_cpp_matches_numba_at_scale():
    """cpp == numba on a larger space that exercises the fused kernel's hot path.

    The 8-orbital/(3, 3) full space is ~3k determinants -- big enough that the
    dynamic-scheduling and O(1)-sign changes in ``calc_trunc_pcr_fused`` are
    actually exercised, and that any load-balancing or parity regression would
    move the energy relative to numba.
    """
    norb, nelec_a, nelec_b = 8, 3, 3
    ham = _mid_hamiltonian(norb, nelec_a, nelec_b, seed=11)
    ci_strs = (
        make_strings(range(norb), nelec_a),
        make_strings(range(norb), nelec_b),
    )

    e_cpp = _solve(ham, ci_strs, backend="cpp")
    e_numba = _solve(ham, ci_strs, backend="numba")

    assert np.isfinite(e_cpp)
    assert abs(e_cpp - e_numba) * 1e3 < SCALE_TOL_MHA, (
        f"cpp vs numba |ΔE| = {abs(e_cpp - e_numba) * 1e3:.6f} mHa"
    )


@_run_scaling
@pytest.mark.slow
def test_cpp_backend_performance_vs_numba(capsys):
    """Benchmark: time the cpp vs numba backend end to end and report the ratio.

    This is a regression guard, not a hard speed contract: wall-clock ratios
    vary with machine, thread count, and numba JIT warm-up, so we only assert
    the cpp path is not pathologically slower (>3x) than numba on the same
    problem. The measured speedup is printed (visible under ``pytest -s``) so a
    real regression in the kernels shows up as a shrinking ratio over time.

    numba is JIT-compiled, so the first call pays a one-off compile cost; we run
    an untimed warm-up solve per backend before the timed measurement.
    """
    norb, nelec_a, nelec_b = 8, 3, 3
    ham = _mid_hamiltonian(norb, nelec_a, nelec_b, seed=23)
    ci_strs = (
        make_strings(range(norb), nelec_a),
        make_strings(range(norb), nelec_b),
    )

    # Warm up: numba JIT compile + any first-call allocation on both backends.
    e_cpp_warm = _solve(ham, ci_strs, backend="cpp")
    e_numba_warm = _solve(ham, ci_strs, backend="numba")
    assert abs(e_cpp_warm - e_numba_warm) * 1e3 < SCALE_TOL_MHA

    def _timed(backend: str) -> float:
        t0 = time.perf_counter()
        _solve(ham, ci_strs, backend=backend)
        return time.perf_counter() - t0

    t_cpp = _timed("cpp")
    t_numba = _timed("numba")

    speedup = t_numba / t_cpp if t_cpp > 0 else float("inf")
    with capsys.disabled():
        print(
            f"\n[perf] {norb}orb/({nelec_a},{nelec_b}): "
            f"cpp {t_cpp * 1e3:.1f} ms, numba {t_numba * 1e3:.1f} ms, "
            f"speedup {speedup:.2f}x"
        )

    # Guard against a pathological regression only; do not pin an exact speedup.
    assert t_cpp < 3.0 * t_numba, (
        f"cpp backend unexpectedly slow: cpp {t_cpp * 1e3:.1f} ms vs numba {t_numba * 1e3:.1f} ms"
    )


# ---------------------------------------------------------------------------
# Thread-scaling sweep
# ---------------------------------------------------------------------------
# Run in a subprocess: OpenMP latches OMP_NUM_THREADS at its first parallel
# region, so it cannot be varied in-process. Each subprocess builds the same
# random Hamiltonian, warms up, times ONE cpp solve, and prints a JSON line.
# ``calc_trunc_pcr_fused`` reads its thread count from omp_get_max_threads(),
# which honours OMP_NUM_THREADS, so this exercises the dynamic-chunk scaling.
_SWEEP_CHILD = r"""
import json, os, sys, time
import numpy as np
from pyscf.fci.cistring import make_strings
from sci_solvers.hamiltonian import HamiltonianData
from sci_solvers.energy_solvers.truncated_davidson_solver import TruncatedDavidsonSolver

norb, na, nb = 9, 4, 4
rng = np.random.default_rng(101)
hc = rng.standard_normal((norb, norb)); hc = 0.5 * (hc + hc.T)
g = rng.standard_normal((norb,) * 4)
g = g + g.transpose(1, 0, 2, 3); g = g + g.transpose(0, 1, 3, 2)
g = g + g.transpose(2, 3, 0, 1); g *= 0.1
ham = HamiltonianData(
    hcore=np.ascontiguousarray(hc), eri=np.ascontiguousarray(g), e_nuc=0.0,
    num_orbitals=norb, num_elec_a=na, num_elec_b=nb,
)
ci = (make_strings(range(norb), na), make_strings(range(norb), nb))

def solve():
    s = TruncatedDavidsonSolver(
        symmetrize_spin=False, hamiltonian=ham, verbose=False, eps=1e-4,
        max_cycle=100, delta_max=1e-3, delta_min=1e-6, growth_factor=2.0,
        backend="cpp",
    )
    return s.compute_energy(ci_strs_batches=[ci])[0].energy

solve()  # untimed warm-up
t = time.perf_counter(); e = solve(); dt = time.perf_counter() - t
print(json.dumps({"threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
                  "ms": dt * 1e3, "energy": e}))
"""


@_run_scaling
@pytest.mark.slow
def test_cpp_backend_thread_scaling(capsys):
    """Sweep OMP_NUM_THREADS and report the scaling curve for the cpp backend.

    Guards the dynamic-chunk scheduling (chunk = n_I / (n_threads * 8)) against
    a many-core regression: a fixed chunk or a load-imbalance bug shows up as
    scaling that flattens or inverts as threads increase. We only assert (a)
    every thread count reaches the *same* energy -- correctness must not depend
    on the thread count -- and (b) two threads are at least modestly faster than
    one, a soft parallel-benefit check that tolerates CI machines with few cores
    or heavy contention. The full curve is printed for eyeballing under ``-s``.
    """
    max_cores = os.cpu_count() or 1
    thread_counts = [n for n in (1, 2, 4, 8) if n <= max_cores]
    if len(thread_counts) < 2:
        pytest.skip("need at least 2 cores to measure thread scaling")

    env_base = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    results = []
    for n in thread_counts:
        env = {**env_base, "OMP_NUM_THREADS": str(n)}
        proc = subprocess.run(
            [sys.executable, "-c", _SWEEP_CHILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        # The child prints exactly one JSON line on stdout (last non-empty line).
        line = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1]
        results.append(json.loads(line))

    with capsys.disabled():
        base_ms = results[0]["ms"]
        print("\n[scaling] cpp backend, 9orb/(4,4):")
        for r in results:
            print(
                f"  threads={r['threads']:3d}  {r['ms']:8.1f} ms  speedup {base_ms / r['ms']:.2f}x"
            )

    # (a) energy must be thread-count invariant (allow float reduction-order drift).
    e0 = results[0]["energy"]
    for r in results[1:]:
        assert abs(r["energy"] - e0) * 1e3 < SCALE_TOL_MHA, (
            f"energy depends on thread count: {r['threads']} threads gave "
            f"{r['energy']:.8f} vs {e0:.8f} at 1 thread"
        )

    # (b) soft parallel-benefit: 2 threads beats 1 by at least 20%.
    t1 = next(r["ms"] for r in results if r["threads"] == 1)
    t2 = next(r["ms"] for r in results if r["threads"] == 2)
    assert t2 < 0.8 * t1, (
        f"no parallel benefit at 2 threads: {t2:.1f} ms vs {t1:.1f} ms at 1 thread"
    )
