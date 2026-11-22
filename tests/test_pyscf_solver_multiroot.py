"""Tests for PySCFSolver multi-root support.

Verifies that:
1. ``solve_sci_batch_multiroot`` with ``n_roots=1`` produces the same ground-state
   energy as ``qiskit_addon_sqd.fermion.solve_sci_batch`` (the qiskit path used by
   ``PySCFSolver`` when ``n_roots=1``).
2. ``solve_sci_batch_multiroot`` with ``n_roots > 1`` returns a spectrum that matches
   PySCF's ``kernel_fixed_space`` reference (the same call, just with ``nroots`` passed
   directly without any wrapper).
3. ``PySCFSolver(n_roots=1)`` and ``PySCFSolver(n_roots=3)`` agree on the ground-state
   energy (informational — prints the diff but does not fail).

The test uses a small synthetic random Hamiltonian (norb=6, 3α/3β electrons) with
two batches of ≤50 CI strings each.

All tests are marked ``slow`` (PySCF Davidson is slow to initialise on first import).
Deselect with ``pytest -m "not slow"``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time
import warnings
from itertools import combinations

import numpy as np
import pytest
from pyscf.fci import selected_ci as pyscf_selected_ci
from qiskit_addon_sqd.fermion import solve_sci_batch

from sci_solvers.eigen_states import EigenStates
from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver, solve_sci_batch_multiroot

# All tests here invoke PySCF's Davidson solver, which is slow to initialise
# on first import.  Mark the whole module so callers can deselect with
# ``pytest -m "not slow"``.
pytestmark = pytest.mark.slow

# ── tolerances ────────────────────────────────────────────────────────────────
ATOL = 1e-6  # energy agreement between code paths

# ── shared synthetic problem ───────────────────────────────────────────────────
NORB = 6
NA = 3
NB = 3
NELEC = (NA, NB)
NROOTS = 3
SEED = 42


def _all_strings(norb: int, nelec: int) -> np.ndarray:
    """All CI strings (int64 bitmasks) with ``nelec`` occupied of ``norb`` orbitals."""
    return np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(norb), nelec)),
        dtype=np.int64,
    )


def _random_symmetric_system(seed: int, norb: int, na: int, nb: int):
    """Random symmetric hcore + 8-fold-symmetric eri; returns full CI strings."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    x = rng.standard_normal((norb,) * 4)
    eri = np.zeros_like(x)
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
        eri += x.transpose(perm)
    eri /= 8.0
    return hcore, eri, _all_strings(norb, na), _all_strings(norb, nb)


@pytest.fixture(scope="module")
def problem():
    """Small synthetic Hamiltonian + CI strings, shared across all tests in this module."""
    hcore, eri, sa, sb = _random_symmetric_system(SEED, NORB, NA, NB)
    return hcore, eri, sa, sb


@pytest.fixture(scope="module")
def ci_strings_batches(problem):
    """Two batches of different sizes, both capped at 50 strings per spin sector.

    Using two batches exercises the per-batch loop in both code paths.
    Both are drawn from the sorted CI-string pool so the test stays fast.
    """
    _, _, sa, sb = problem
    # batch 0: first 30 strings per spin sector
    n0 = min(30, len(sa), len(sb))
    batch0 = (sa[:n0], sb[:n0])
    # batch 1: first 50 strings per spin sector
    n1 = min(50, len(sa), len(sb))
    batch1 = (sa[:n1], sb[:n1])
    return [batch0, batch1]


@pytest.fixture(scope="module")
def reference_energies(problem, ci_strings_batches):
    """Ground truth: lowest NROOTS energies per batch from PySCF kernel_fixed_space.

    This calls the PySCF function directly with no wrappers, serving as the
    single source of truth both solver paths are compared against.
    """
    hcore, eri, _, _ = problem
    refs = []
    for ci_strs in ci_strings_batches:
        myci = pyscf_selected_ci.SelectedCI()
        e, _ = pyscf_selected_ci.kernel_fixed_space(
            myci, hcore, eri, NORB, NELEC, ci_strs=ci_strs, nroots=NROOTS
        )
        e = np.sort(np.atleast_1d(np.asarray(e, dtype=float)))[:NROOTS]
        refs.append(e)
    return refs


# ── helpers ────────────────────────────────────────────────────────────────────


def _ground_energies_from_qiskit(hcore, eri, ci_strings_batches):
    """Ground-state energies via the unmodified qiskit solve_sci_batch."""
    results = solve_sci_batch(
        ci_strings=ci_strings_batches,
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=NORB,
        nelec=NELEC,
    )
    return [r.energy for r in results]


def _eigen_states_from_multiroot(hcore, eri, ci_strings_batches, n_roots):
    """EigenStates per batch from solve_sci_batch_multiroot."""
    return solve_sci_batch_multiroot(
        ci_strings=ci_strings_batches,
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=NORB,
        nelec=NELEC,
        n_roots=n_roots,
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_multiroot_n_roots_1_matches_qiskit(problem, ci_strings_batches):
    """solve_sci_batch_multiroot with n_roots=1 gives the same ground-state energy
    as qiskit's solve_sci_batch (the path PySCFSolver uses for n_roots=1).

    Also reports wall-clock time for both paths so the overhead of the new code
    path relative to the qiskit path is visible in the test output (use
    ``pytest -s`` to see the print statements).
    """
    hcore, eri, _, _ = problem

    t0 = time.perf_counter()
    qiskit_energies = _ground_energies_from_qiskit(hcore, eri, ci_strings_batches)
    t1 = time.perf_counter()
    multiroot_states = _eigen_states_from_multiroot(hcore, eri, ci_strings_batches, n_roots=1)
    t2 = time.perf_counter()

    elapsed_qiskit = t1 - t0
    elapsed_multiroot = t2 - t1
    print(f"\n  solve_sci_batch (qiskit, n_roots=1):        {elapsed_qiskit:.3f}s")
    print(f"  solve_sci_batch_multiroot (n_roots=1):      {elapsed_multiroot:.3f}s")
    print(
        f"  ratio (multiroot / qiskit):                 {elapsed_multiroot / elapsed_qiskit:.2f}x"
    )

    assert len(multiroot_states) == len(ci_strings_batches)
    for i, (es, e_qiskit) in enumerate(zip(multiroot_states, qiskit_energies)):
        assert isinstance(es, EigenStates)
        assert len(es.states) == 1, f"batch {i}: expected 1 root, got {len(es.states)}"
        assert abs(es.states[0].energy - e_qiskit) < ATOL, (
            f"batch {i}: multiroot energy {es.states[0].energy} != qiskit {e_qiskit}"
        )


def test_multiroot_nroots_gt1_matches_reference(problem, ci_strings_batches, reference_energies):
    """solve_sci_batch_multiroot with n_roots=NROOTS recovers the reference spectrum."""
    hcore, eri, _, _ = problem

    multiroot_states = _eigen_states_from_multiroot(hcore, eri, ci_strings_batches, n_roots=NROOTS)

    assert len(multiroot_states) == len(ci_strings_batches)
    for i, (es, ref) in enumerate(zip(multiroot_states, reference_energies)):
        assert isinstance(es, EigenStates)
        got = np.array([s.energy for s in es.states])
        assert np.allclose(got, ref, atol=ATOL), (
            f"batch {i}: energies {got} do not match reference {ref}"
        )


def test_pyscf_solver_n_roots_1_matches_n_roots_3_ground_state(problem, ci_strings_batches):
    """PySCFSolver(n_roots=1) and PySCFSolver(n_roots=3) should be close on the ground-state energy.

    The n_roots=1 path goes through qiskit's solve_sci_batch; the n_roots=3 path
    goes through solve_sci_batch_multiroot. Both solve the same subspace with the
    same algorithm, so they should agree well — but they are different code paths
    (different RDM contraction order, different internal normalisation) so a tiny
    numerical discrepancy is possible. This test is informational: it prints the
    difference but does not fail the suite.
    """
    hcore, eri, _, _ = problem

    solver1 = PySCFSolver(symmetrize_spin=False, n_roots=1)
    solver3 = PySCFSolver(symmetrize_spin=False, n_roots=3)

    es1_list = solver1.sci_solver_callable_eigen_states(ci_strings_batches, hcore, eri, NORB, NELEC)
    es3_list = solver3.sci_solver_callable_eigen_states(ci_strings_batches, hcore, eri, NORB, NELEC)

    assert len(es1_list) == len(es3_list) == len(ci_strings_batches)
    for i, (es1, es3) in enumerate(zip(es1_list, es3_list)):
        e1 = es1.states[0].energy
        e3 = es3.states[0].energy
        diff = abs(e1 - e3)
        print(
            f"\n  batch {i}: n_roots=1 ground state = {e1:.10f}, "
            f"n_roots=3 ground state = {e3:.10f}, diff = {diff:.2e}"
        )
        if diff >= ATOL:
            warnings.warn(
                f"batch {i}: ground-state energies differ by {diff:.2e} "
                f"(n_roots=1: {e1}, n_roots=3: {e3}). "
                "This is informational — the two code paths may differ slightly.",
                UserWarning,
                stacklevel=1,
            )


def test_pyscf_solver_n_roots_3_returns_ascending_energies(problem, ci_strings_batches):
    """PySCFSolver(n_roots=3) returns states in ascending energy order per batch."""
    hcore, eri, _, _ = problem

    solver = PySCFSolver(symmetrize_spin=False, n_roots=NROOTS)
    es_list = solver.sci_solver_callable_eigen_states(ci_strings_batches, hcore, eri, NORB, NELEC)

    for i, es in enumerate(es_list):
        energies = [s.energy for s in es.states]
        assert energies == sorted(energies), f"batch {i}: energies not sorted ascending: {energies}"


def test_pyscf_solver_invalid_n_roots():
    """PySCFSolver raises ValueError for n_roots < 1."""
    with pytest.raises(ValueError, match="nroots"):
        PySCFSolver(symmetrize_spin=False, n_roots=0)
