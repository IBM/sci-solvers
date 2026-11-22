"""End-to-end reference run on benzoic acid with a structured CIS-like basis.

This is the *anchor* result for the scalar reference:
the energy our pure-Python SBD port produces on a deterministic
CIS-like basis (HF + single excitations of HF) for the bundled benzoic
acid (SQD molecules dataset ID 182, 36 electrons in 30 orbitals; see
torch_sci_solver/data/README.md).
Future torch implementations are required to reproduce this number
(to ~1e-10 relative).

The α basis is the Hartree-Fock string plus all its single excitations
(``torch_sci_solver._paths.hf_and_singles``), with β = α (closed-shell
singlet). Using this basis guarantees:
  - The HF det is at adets[0] (lowest integer)
  - All three matvec task types fire: αα, ββ, αβ
  - Davidson does real iterative work (not converging in 1 step
    on a disconnected basis)
  - The number is reproducible: anyone running our code on the
    same FCIDUMP gets the same energy.

This is a manual script (no pytest test functions); it is a
*correctness* anchor, not a perf target.

A smaller variant is available via ``run_subset(K)`` which
truncates to the first K α/β-strings.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import sys
import time
from typing import List

import numpy as np
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import (
    diag,
    setup_integrals_from_pyscf_h1_eri,
)


def _get_fcidump():
    return str(_paths.fcidump_path("benzoic_acid"))


def _load_integrals():
    """Parse the benzoic acid FCIDUMP and build I0, I1, I2 in our format."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    nelec_total = raw["NELEC"]
    ms2 = raw["MS2"]
    mol = _paths.MOLECULES["benzoic_acid"]
    assert L == mol.norb and nelec_total == mol.nelec_a + mol.nelec_b and ms2 == 0, (
        f"unexpected header: NORB={L}, NELEC={nelec_total}, MS2={ms2}"
    )
    n_a = (nelec_total + ms2) // 2
    n_b = (nelec_total - ms2) // 2

    h1 = raw["H1"]
    eri_full = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    core = raw["ECORE"]
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri_full, core)
    return L, n_a, n_b, I0, I1, I2


def _do_run(
    adets: List[int],
    bdets: List[int],
    L: int,
    n_a: int,
    I0: float,
    I1,
    I2,
    *,
    eps: float = 1e-7,
    davidson_block_size: int = 30,
    max_davidson_iters: int = 50,
):
    M_a, M_b = len(adets), len(bdets)
    print(f"  basis: {M_a} α × {M_b} β = {M_a * M_b}")
    # Sanity: HF should be first
    hf_alpha = (1 << n_a) - 1
    if adets[0] == hf_alpha:
        print(f"  adets[0] = HF (lowest {n_a} α-orbs occupied) ✓")
    else:
        print(f"  WARNING: adets[0] ({adets[0]}) != HF ({hf_alpha})")

    t0 = time.perf_counter()
    energy, eigvec, info = diag(
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        max_iteration=max_davidson_iters,
        num_block=davidson_block_size,
        eps=eps,
        verbose=True,
    )
    t = time.perf_counter() - t0
    print(f"\nSolve time: {t:.1f}s")
    print("\nRESULT")
    print(f"  Energy   = {energy:.10f}")
    print(f"  matvecs  = {info['n_matvecs']}")
    print(f"  ||R||    = {info['final_residual_norm']:.3e}")
    print(f"  converged = {info['converged']}")
    return energy, info


def run_full():
    """Full HF + singles basis (217 × 217 = 47,089 dets on benzoic acid)."""
    print(f"\n{'=' * 70}")
    print("Benzoic acid reference run: full HF + singles basis")
    print(f"{'=' * 70}")
    L, n_a, n_b, I0, I1, I2 = _load_integrals()
    print(f"  L={L}, n_a={n_a}, n_b={n_b}, core={I0:.10f}")
    adets = _paths.hf_and_singles(L, n_a)
    print(f"  built {len(adets)} α-strings (HF + singles)")
    bdets = list(adets)  # singlet: β = α
    return _do_run(adets, bdets, L, n_a, I0, I1, I2)


def run_subset(k: int):
    """First K HF + singles α-strings (and same K β-strings).
    Use this for quick smoke tests; the full basis is slow in scalar Python.
    """
    print(f"\n{'=' * 70}")
    print(f"Benzoic acid reference run: first {k} HF + singles α-strings")
    print(f"{'=' * 70}")
    L, n_a, n_b, I0, I1, I2 = _load_integrals()
    print(f"  L={L}, n_a={n_a}, n_b={n_b}, core={I0:.10f}")
    adets = _paths.hf_and_singles(L, n_a)[:k]
    bdets = list(adets)
    return _do_run(adets, bdets, L, n_a, I0, I1, I2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Numeric arg → subset of size k; "full" → full run
        if sys.argv[1] == "full":
            run_full()
        else:
            k = int(sys.argv[1])
            run_subset(k)
    else:
        # Default: medium subset
        run_subset(50)
