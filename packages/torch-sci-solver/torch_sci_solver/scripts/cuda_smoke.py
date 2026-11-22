"""CUDA smoke test — verifies torch_sci_solver works on this hardware.

Invoked as ``torch-sci-solver-smoke`` after install, or directly with
``python -m torch_sci_solver.scripts.cuda_smoke``.

What it checks
--------------
For each of Anchors 1, 2, (and 3 if a shots file is available), on the bundled benzoic acid
(SQD molecules dataset ID 182)
(36 electrons in 30 orbitals) unless env vars point elsewhere:
  1. Compute energy via scalar Python reference   → "ground truth on
     YOUR FCIDUMP".
  2. Compute energy via torch CPU                  → must agree with (1)
     to rel error < 1e-10.
  3. Compute energy via torch CUDA (if available)  → must agree with (1)
     to rel error < 1e-10.
  4. Compare to the *recorded* anchor energy (``RECORDED_ANCHOR_ENERGIES``)
     — this is ADVISORY: it matches only on the bundled FCIDUMP and shots.
     The port correctness check is (2) and (3), not (4).

Env vars override the bundled data files:
    export TORCH_SCI_SOLVER_FCIDUMP=/path/to/molecule.fcidump
    export TORCH_SCI_SOLVER_SHOTS=/path/to/shots.npz
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import argparse
import sys
import time

import numpy as np
import torch

from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import (
    diag as diag_scalar,
)
from torch_sci_solver.sbd_ref import (
    setup_integrals_from_pyscf_h1_eri,
)
from torch_sci_solver.torch_impl.sbdiag import diag_torch

# Recorded on the bundled benzoic acid FCIDUMP and shots; the same values as
# tests/test_torch_anchors.py. The scalar-Python reference (computed at runtime) is the
# FCIDUMP-independent ground truth.
RECORDED_ANCHOR_ENERGIES = {
    1: -418.3899102203,  # HF + singles K=30, single diag
    2: -418.4168201135,  # HF + singles K=50, single diag
    3: -418.3611220506,  # recover_configurations N_SHOTS=50, HF prior, single diag
}


def _load_integrals(fcidump_path):
    """Load integrals and the per-spin electron count (closed shell)."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(str(fcidump_path))
    L = raw["NORB"]
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    return L, raw["NELEC"] // 2, I0, I1, I2


def _build_anchor3_basis(L, n_a, n_b, shots_path):
    """Build anchor3 basis."""
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

    from torch_sci_solver.shots import load_shots

    shots = load_shots(str(shots_path))
    items = list(shots.items())[:50]
    bitstrings = [k for k, _ in items]
    probabilities = np.array([v for _, v in items], dtype=np.float64)
    probabilities /= probabilities.sum()
    bm = np.array([[c == "1" for c in bs] for bs in bitstrings], dtype=bool)
    avg_occ_a = np.zeros(L)
    avg_occ_a[:n_a] = 1.0
    avg_occ_b = np.zeros(L)
    avg_occ_b[:n_b] = 1.0
    bm_rec, _ = recover_configurations(
        bm, probabilities, (avg_occ_a, avg_occ_b), n_a, n_b, rand_seed=1337
    )
    ci_a, ci_b = bitstring_matrix_to_ci_strs(bm_rec, open_shell=False)
    return (sorted(int(x) for x in ci_a), sorted(int(x) for x in ci_b))


def run_anchor(name, anchor_num, adets, bdets, L, I0, I1, I2, args, run_scalar_truth=True):
    """Run anchor."""
    print("=" * 70)
    print(f"Anchor {anchor_num}: {name}")
    print(f"  basis dim: {len(adets) * len(bdets)}")
    print("=" * 70)

    # 1. Scalar ground truth (only for the smaller anchors; skipped for big ones)
    E_scalar = None
    t_scalar = None
    if run_scalar_truth:
        print("  Running scalar Python reference (ground truth on YOUR FCIDUMP)...")
        t0 = time.perf_counter()
        E_scalar, _, info_s = diag_scalar(
            adets,
            bdets,
            L,
            I0,
            I1,
            I2,
            max_iteration=50,
            num_block=args.davidson_block_size,
            eps=1e-7,
            verbose=False,
        )
        t_scalar = time.perf_counter() - t0
        print(f"    scalar energy:  {E_scalar:.10f} Ha  ({t_scalar:.1f}s)")

    # 2. Torch CPU
    print("  Running torch on CPU...")
    t0 = time.perf_counter()
    E_cpu, _, info_cpu = diag_torch(
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        max_davidson_iters=50,
        davidson_block_size=args.davidson_block_size,
        eps=1e-7,
        device="cpu",
        chunk_size=args.chunk_size,
        verbose=False,
    )
    t_cpu = time.perf_counter() - t0
    print(f"    torch CPU:      {E_cpu:.10f} Ha  ({t_cpu:.1f}s)")

    # 3. Torch CUDA (if available)
    E_cuda = None
    t_cuda = None
    if args.device == "cuda":
        print("  Running torch on CUDA...")
        t0 = time.perf_counter()
        E_cuda, _, info_cuda = diag_torch(
            adets,
            bdets,
            L,
            I0,
            I1,
            I2,
            max_davidson_iters=50,
            davidson_block_size=args.davidson_block_size,
            eps=1e-7,
            device="cuda",
            chunk_size=args.chunk_size,
            verbose=False,
        )
        torch.cuda.synchronize()
        t_cuda = time.perf_counter() - t0
        print(f"    torch CUDA:     {E_cuda:.10f} Ha  ({t_cuda:.1f}s)")
    print()

    # 4. Verdicts
    verdicts = {
        "name": name,
        "anchor_num": anchor_num,
        "E_scalar": E_scalar,
        "E_cpu": E_cpu,
        "E_cuda": E_cuda,
        "t_scalar": t_scalar,
        "t_cpu": t_cpu,
        "t_cuda": t_cuda,
        "cpu_vs_scalar_ok": None,
        "cuda_vs_scalar_ok": None,
        "matches_recorded": None,
    }

    def _rel(a, b):
        """Rel."""
        return abs(a - b) / max(abs(b), 1.0)

    print("  Comparisons:")
    if E_scalar is not None:
        rel_cpu = _rel(E_cpu, E_scalar)
        ok = rel_cpu < 1e-10
        verdicts["cpu_vs_scalar_ok"] = ok
        flag = "✓" if ok else "✗"
        print(f"    torch CPU vs scalar:  rel error {rel_cpu:.3e}  {flag}")
        if E_cuda is not None:
            rel_cuda = _rel(E_cuda, E_scalar)
            ok = rel_cuda < 1e-10
            verdicts["cuda_vs_scalar_ok"] = ok
            flag = "✓" if ok else "✗"
            print(f"    torch CUDA vs scalar: rel error {rel_cuda:.3e}  {flag}")
    else:
        # No scalar truth available — compare CPU vs CUDA only
        if E_cuda is not None:
            rel_cc = _rel(E_cpu, E_cuda)
            ok = rel_cc < 1e-10
            flag = "✓" if ok else "✗"
            print(f"    torch CPU vs CUDA:    rel error {rel_cc:.3e}  {flag}")

    # Advisory: hardcoded anchor
    if anchor_num in RECORDED_ANCHOR_ENERGIES:
        rec = RECORDED_ANCHOR_ENERGIES[anchor_num]
        rel_rec = _rel(E_cpu, rec)
        matches = rel_rec < 1e-9
        verdicts["matches_recorded"] = matches
        if matches:
            print(
                f"    vs recorded anchor:   rel error {rel_rec:.3e}  ✓ "
                f"(your FCIDUMP matches calibration)"
            )
        else:
            print(f"    vs recorded anchor:   rel error {rel_rec:.3e}  (advisory — see note below)")
    print()
    return verdicts


def main(argv=None):
    """Main."""
    ap = argparse.ArgumentParser(description="torch_sci_solver smoke test")
    ap.add_argument(
        "--device", default=None, help="torch device (default: cuda if available, else cpu)"
    )
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    ap.add_argument("--davidson-block-size", type=int, default=30)
    ap.add_argument(
        "--no-scalar",
        action="store_true",
        help="skip scalar Python reference (faster but loses ground-truth check)",
    )
    args = ap.parse_args(argv)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("torch_sci_solver smoke test")
    print(f"  torch:        {torch.__version__}")
    print(f"  cuda avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device:  {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"  free / total: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    print(f"  using device: {args.device}")
    print()

    fcidump = _paths.fcidump_path()
    if fcidump is None:
        print("ERROR: FCIDUMP not found. Set TORCH_SCI_SOLVER_FCIDUMP.")
        return 2
    print(f"  FCIDUMP: {fcidump}")
    shots = _paths.shots_path()
    if shots is not None:
        print(f"  shots:   {shots}")
    else:
        print("  shots:   (not found; Anchor 3 will be skipped)")
    print()

    L, n, I0, I1, I2 = _load_integrals(fcidump)
    adets_all = _paths.hf_and_singles(L, n)
    print(f"Loaded integrals: L={L}, n={n}+{n}, |HF + singles|={len(adets_all)}")
    print()

    all_verdicts = []

    # Anchor 1: K=30 — small enough to run scalar reference
    v = run_anchor(
        "HF + singles K=30",
        1,
        adets_all[:30],
        list(adets_all[:30]),
        L,
        I0,
        I1,
        I2,
        args,
        run_scalar_truth=not args.no_scalar,
    )
    all_verdicts.append(v)

    # Anchor 2: K=50 — also fine to run scalar (a few minutes)
    v = run_anchor(
        "HF + singles K=50",
        2,
        adets_all[:50],
        list(adets_all[:50]),
        L,
        I0,
        I1,
        I2,
        args,
        run_scalar_truth=not args.no_scalar,
    )
    all_verdicts.append(v)

    # Anchor 3: recover_configurations N_SHOTS=50 (2.4K basis — scalar is fine here too)
    if shots is not None:
        print(f"Building Anchor 3 basis from {shots}...")
        adets3, bdets3 = _build_anchor3_basis(L, n, n, shots)
        print(f"  basis size: {len(adets3) * len(bdets3)}")
        print()
        v = run_anchor(
            "recover_configurations N_SHOTS=50",
            3,
            adets3,
            bdets3,
            L,
            I0,
            I1,
            I2,
            args,
            run_scalar_truth=not args.no_scalar,
        )
        all_verdicts.append(v)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'anchor':<35} {'scalar==CPU':<12} {'scalar==CUDA':<13} {'==recorded':<12}")
    for v in all_verdicts:
        sc = "—" if v["cpu_vs_scalar_ok"] is None else ("✓" if v["cpu_vs_scalar_ok"] else "✗")
        cu = "—" if v["cuda_vs_scalar_ok"] is None else ("✓" if v["cuda_vs_scalar_ok"] else "✗")
        rec = (
            "—"
            if v["matches_recorded"] is None
            else ("✓" if v["matches_recorded"] else "advisory ⚠")
        )
        print(f"  {v['name']:<35} {sc:<12} {cu:<13} {rec:<12}")
    print()

    # Aggregate decision: port is correct iff all torch-vs-scalar checks pass.
    port_ok = all(v["cpu_vs_scalar_ok"] is None or v["cpu_vs_scalar_ok"] for v in all_verdicts)
    if args.device == "cuda":
        port_ok = port_ok and all(
            v["cuda_vs_scalar_ok"] is None or v["cuda_vs_scalar_ok"] for v in all_verdicts
        )
    matches_any_recorded = any(
        v["matches_recorded"] for v in all_verdicts if v["matches_recorded"] is not None
    )
    matches_all_recorded = all(
        v["matches_recorded"] for v in all_verdicts if v["matches_recorded"] is not None
    )

    if port_ok:
        print("Port verdict: ✓ CORRECT — torch matches scalar reference everywhere.")
    else:
        print("Port verdict: ✗ MISMATCH — torch disagrees with scalar reference.")
        return 1

    if matches_all_recorded:
        print("FCIDUMP:      ✓ matches the recorded anchors (bundled data).")
    elif matches_any_recorded:
        print("FCIDUMP:      ⚠ partially matches recorded anchors.")
    else:
        print("FCIDUMP:      ⚠ differs from the recorded anchors.")
        print()
        print("  This is OK. The recorded energies were measured on the bundled")
        print("  benzoic acid data; a different FCIDUMP or shots file gives a")
        print("  different (still valid) ground-state energy. The port verdict")
        print("  above is the correctness check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
