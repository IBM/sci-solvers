"""SQD iteration loop test on benzoic acid (SQD molecules dataset ID 182).

Starting points:
  - Test A: HF prior + 50 shots (a representative starting basis).
    With HF prior pulling recovered configs toward HF, the basis grows
    iteration-over-iteration as 1-RDM populates orbitals.
  - Test B: uniform prior + 100 shots (a typical SQD default).
    Uniform prior gives disconnected first basis; expect Davidson to
    return trivial first iteration, then iterations refine.

For each, we run up to 5 iterations and print the trajectory.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time

import numpy as np
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.shots import load_shots
from torch_sci_solver.torch_impl.sqd_loop import sqd_loop


def _get_fcidump():
    return str(_paths.fcidump_path("benzoic_acid"))


def _get_shots():
    return str(_paths.shots_path("benzoic_acid"))


def _hf_and_singles(k: int) -> list[int]:
    mol = _paths.MOLECULES["benzoic_acid"]
    return _paths.hf_and_singles(mol.norb, mol.nelec_a)[:k]


def _load_integrals():
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    return L, I0, I1, I2


def _print_history(label, result):
    print(f"\n  {label}: history (iter, basis, energy, ΔE, Δocc, matvecs)")
    print("    " + "-" * 78)
    for h in result["history"]:
        print(
            f"    iter={h['iter']:>2}  basis={h['basis_size']:>7}  "
            f"E={h['energy']:.10f}  ΔE={h['delta_E']:.3e}  "
            f"Δocc={h['occ_drift']:.3e}  mv={h['n_matvecs']}"
        )
    print(
        f"  converged={result['converged']}, n_iter={result['n_iter']}, "
        f"final_E={result['final_energy']:.10f}"
    )


def test_sqd_loop_hf_prior_50_shots():
    """Test A: HF prior + 50 shots. First iter == Anchor 3."""
    print("=" * 70)
    print("Test A: HF prior, N_SHOTS=50")
    print("=" * 70)
    L, I0, I1, I2 = _load_integrals()
    n_a = n_b = _paths.MOLECULES["benzoic_acid"].nelec_a

    shots = load_shots(_get_shots())
    items = list(shots.items())[:50]
    bitstrings = [k for k, _ in items]
    probabilities = np.array([v for _, v in items], dtype=np.float64)

    # HF prior
    occ_a0 = np.zeros(L)
    occ_a0[:n_a] = 1.0
    occ_b0 = np.zeros(L)
    occ_b0[:n_b] = 1.0

    t0 = time.perf_counter()
    result = sqd_loop(
        bitstrings=bitstrings,
        probabilities=probabilities,
        n_orb=L,
        n_a=n_a,
        n_b=n_b,
        I0=I0,
        I1=I1,
        I2=I2,
        initial_prior=(occ_a0, occ_b0),
        max_iter=5,
        tol_E=1e-6,
        tol_occ=1e-4,
        recovery_seed=1337,
        device="cpu",
        verbose=True,
    )
    t = time.perf_counter() - t0
    print(f"  wall time: {t:.1f}s")

    _print_history("Test A", result)

    # Iter 0 should match Anchor 3 (test_torch_anchors.py) to 1e-10
    iter0_E = result["history"][0]["energy"]
    expected_anchor3 = -418.3611220506
    diff = abs(iter0_E - expected_anchor3)
    print(f"\n  iter 0 vs Anchor 3: |Δ| = {diff:.3e} (expected ≤ 1e-9)")
    assert diff < 1e-9, f"iter 0 should match Anchor 3, got |Δ|={diff:.3e}"
    print("  ✓ iter 0 reproduces Anchor 3")


def test_sqd_loop_uniform_prior_100_shots():
    """Test B: uniform prior + 100 shots. Disconnected first basis;
    see if subsequent iterations recover."""
    print("\n" + "=" * 70)
    print("Test B: uniform prior, N_SHOTS=100")
    print("=" * 70)
    L, I0, I1, I2 = _load_integrals()
    n_a = n_b = _paths.MOLECULES["benzoic_acid"].nelec_a

    shots = load_shots(_get_shots())
    items = list(shots.items())[:100]
    bitstrings = [k for k, _ in items]
    probabilities = np.array([v for _, v in items], dtype=np.float64)

    t0 = time.perf_counter()
    result = sqd_loop(
        bitstrings=bitstrings,
        probabilities=probabilities,
        n_orb=L,
        n_a=n_a,
        n_b=n_b,
        I0=I0,
        I1=I1,
        I2=I2,
        initial_prior=None,  # uniform
        max_iter=5,
        tol_E=1e-6,
        tol_occ=1e-4,
        recovery_seed=1337,
        device="cpu",
        verbose=True,
    )
    t = time.perf_counter() - t0
    print(f"  wall time: {t:.1f}s")

    _print_history("Test B", result)


if __name__ == "__main__":
    test_sqd_loop_hf_prior_50_shots()
    test_sqd_loop_uniform_prior_100_shots()
    print("\nSQD LOOP TESTS DONE")
