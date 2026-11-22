"""Production-scale SQD loop runs on the bundled benzoic acid (dataset ID 182).

Test cases:
  A. N_SHOTS=200, HF prior, default chunk_size=2_000_000:
     Records convergence trajectory + timings for a 37K-dim first basis.
  B. N_SHOTS=200, HF prior, chunk_size=1_000:
     Same problem but with chunking deliberately forced. Result must
     match Test A bit-for-bit (since chunking is associative for the
     same basis ordering).
  C. (Optional, time-permitting) N_SHOTS=500, HF prior:
     ~350K-dim first basis. May take several minutes per iteration.

These produce Anchor 5 (production-scale SQD fixed point).
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


def _load_integrals():
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    return L, I0, I1, I2


def _load_shots(n_shots):
    items = list(load_shots(_get_shots()).items())[:n_shots]
    bitstrings = [k for k, _ in items]
    probabilities = np.array([v for _, v in items], dtype=np.float64)
    return bitstrings, probabilities


def _hf_prior(L, n_a, n_b):
    occ_a = np.zeros(L)
    occ_a[:n_a] = 1.0
    occ_b = np.zeros(L)
    occ_b[:n_b] = 1.0
    return occ_a, occ_b


def _print_history(label, result, wall_time):
    print(f"\n  {label}: history")
    print(f"  {'iter':>4} {'basis':>9} {'energy':>16} {'ΔE':>11} {'Δocc':>11} {'mv':>5}")
    print("  " + "-" * 70)
    for h in result["history"]:
        print(
            f"  {h['iter']:>4} {h['basis_size']:>9} {h['energy']:>16.10f} "
            f"{h['delta_E']:>11.3e} {h['occ_drift']:>11.3e} "
            f"{h['n_matvecs']:>5}"
        )
    print(
        f"  converged={result['converged']}, n_iter={result['n_iter']}, "
        f"final_E={result['final_energy']:.10f}, wall={wall_time:.1f}s"
    )


def run_test(label, n_shots, chunk_size, max_iter=4):
    """Run one sqd_loop configuration on benzoic acid and print its convergence history."""
    print("\n" + "=" * 70)
    print(f"{label}: N_SHOTS={n_shots}, chunk_size={chunk_size}, max_iter={max_iter}")
    print("=" * 70)
    L, I0, I1, I2 = _load_integrals()
    n_a = n_b = _paths.MOLECULES["benzoic_acid"].nelec_a
    bitstrings, probabilities = _load_shots(n_shots)
    occ_a0, occ_b0 = _hf_prior(L, n_a, n_b)

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
        max_iter=max_iter,
        tol_E=1e-6,
        tol_occ=1e-4,
        recovery_seed=1337,
        device="cpu",
        chunk_size=chunk_size,
        verbose=True,
    )
    t = time.perf_counter() - t0
    _print_history(label, result, t)
    return result, t


def main():
    """Run the production-scale SQD-loop scenarios (A/B/... ) and compare chunking configs."""
    # Test A: production-scale, default chunk
    rA, tA = run_test("Test A", n_shots=200, chunk_size=2_000_000, max_iter=4)

    # Test B: same but tiny chunks — chunking should engage and produce
    # the same result.
    rB, tB = run_test("Test B (small chunk)", n_shots=200, chunk_size=1_000, max_iter=4)

    # Cross-check
    print("\n" + "=" * 70)
    print("CROSS-CHECK: Test A vs Test B")
    print("=" * 70)
    if len(rA["history"]) != len(rB["history"]):
        print(f"  !! iteration count differs: A={len(rA['history'])} vs B={len(rB['history'])}")
    else:
        for hA, hB in zip(rA["history"], rB["history"]):
            assert hA["basis_size"] == hB["basis_size"]
            E_diff = abs(hA["energy"] - hB["energy"])
            occ_diff = abs(hA["occ_drift"] - hB["occ_drift"])
            print(
                f"  iter {hA['iter']}: E diff={E_diff:.3e}, "
                f"Δocc diff={occ_diff:.3e}, "
                f"mv: A={hA['n_matvecs']} B={hB['n_matvecs']}"
            )
            assert E_diff < 1e-9, f"chunking changed E at iter {hA['iter']}"
    print(
        f"\n  Anchor 5 (N_SHOTS=200, HF prior, seed=1337): E_final = {rA['final_energy']:.10f} Ha"
    )
    print(
        f"  Wall: A (default chunk) = {tA:.1f}s; B (chunk=1K) = {tB:.1f}s "
        f"(slower as expected with more chunks)"
    )


if __name__ == "__main__":
    main()
