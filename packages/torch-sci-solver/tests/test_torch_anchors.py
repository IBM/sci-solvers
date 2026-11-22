"""The torch pipeline reproduces all three calibration
anchors to relative ~1e-10.

The torch path goes through:
  Setup       → to_torch_integrals + generate_all_torch_helpers
  Diagonal    → make_diag_terms_torch (zero_excite_batch over all dets)
  Davidson    → davidson_torch (matvec via mult_torch)

Each anchor has a recorded energy (see ``ANCHOR_ENERGIES``). The torch result
should agree to at least 1e-10 relative (i.e. ~10 digits) — the only
sources of difference are float-order-of-summation effects in
``index_add_`` and ``einsum``, which can perturb the final digit.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time

import numpy as np
import pytest
import torch
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.sbdiag import diag_torch

# Recorded reference energies, on the bundled benzoic acid (SQD molecules dataset ID 182,
# 36 electrons in 30 orbitals; see torch_sci_solver/data/README.md).
#
# Provenance: ``diag_torch(adets, bdets, L, I0, I1, I2, max_davidson_iters=50,
# davidson_block_size=30, eps=1e-7, device='cpu')`` -- the same entry point and parameters the
# tests below use -- recorded on CPU, then checked against PySCF's independent selected-CI
# solver (``selected_ci.SCI.kernel_fixed_space``, conv_tol=1e-12) on each same basis: all
# three agree to < 5e-11 Ha. Reproducible to all displayed digits across repeated runs.
#
# Bases: anchors 1-2 use the Hartree-Fock string plus its single excitations, anchor 3 the
# recovered configurations of the first 50 shots.
#
# Role: regression pins for the diag_torch pipeline. A change to any
# matvec / helper / Davidson code path that perturbs these numbers
# beyond rtol=1e-10 is a bug.
ANCHOR_ENERGIES = {
    1: -418.3899102203,  # HF + singles, first 30 strings
    2: -418.4168201135,  # HF + singles, first 50 strings
    3: -418.3611220506,  # recover_configurations, N_SHOTS=50, HF prior, seed=1337
}


# Each anchor runs twice: once on CPU, once on CUDA. The CUDA invocation is
# skipped automatically on nodes without a GPU.
#
# The GPU run is not redundant: the Davidson subspace ``eigh`` runs on whatever
# device it is given, so on CUDA it goes through cuSOLVER syevd rather than CPU
# LAPACK. That is a different backend for the innermost eigensolve, and it is
# shared by the single-GPU and multi-GPU paths alike — so these pinned energies
# are the regression guard for it. A CPU-only run cannot exercise it.
#
# ANCHOR_ENERGIES were recorded on CPU (see provenance above), so the CUDA
# variants are held to a CPU-recorded pin. Measured on A100-40GB: CPU and CUDA
# agree with each other to ~1e-15 relative and both land ~4e-14 from the pin —
# i.e. >3 orders of magnitude inside rtol=1e-10, so one tolerance serves both.
DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]


def _load_integrals():
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(str(_paths.fcidump_path("benzoic_acid")))
    L = raw["NORB"]
    assert L == _paths.MOLECULES["benzoic_acid"].norb
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    return L, I0, I1, I2


def _hf_and_singles(k: int) -> list[int]:
    mol = _paths.MOLECULES["benzoic_acid"]
    return _paths.hf_and_singles(mol.norb, mol.nelec_a)[:k]


def _build_anchor3_basis(L: int, n_a: int, n_b: int):
    """Reproduce Anchor 3's basis exactly (HF prior, seed=1337, first 50 shots)."""
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
    from torch_sci_solver.shots import load_shots

    shots = load_shots(str(_paths.shots_path("benzoic_acid")))
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


def run_anchor(name, adets, bdets, L, I0, I1, I2, expected_energy, rtol=1e-10, device="cpu"):
    """Run diag_torch() on a fixed basis and assert energy matches the anchor value."""
    print("=" * 70)
    print(f"Anchor: {name}  [device={device}]")
    print(f"  basis dim: {len(adets) * len(bdets)}")
    print("=" * 70)
    t0 = time.perf_counter()
    energy, eigvec, info = diag_torch(
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        max_davidson_iters=50,
        davidson_block_size=30,
        eps=1e-7,
        device=device,
        verbose=True,
    )
    t = time.perf_counter() - t0
    print(f"\n  torch solve time: {t:.1f}s")
    print(f"  torch energy:     {energy:.10f}")
    print(f"  scalar energy:    {expected_energy:.10f}")
    diff_abs = abs(energy - expected_energy)
    diff_rel = diff_abs / abs(expected_energy)
    print(f"  |Δ|              {diff_abs:.3e}")
    print(f"  |Δ|/|E|          {diff_rel:.3e}")
    if diff_rel > rtol:
        raise AssertionError(f"{name}: rel error {diff_rel:.3e} > rtol {rtol:.3e}")
    print(f"  ✓ within rtol={rtol:.0e}")
    return energy, info, t


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_anchor1(device):
    """Anchor 1: HF + singles, K=30 basis matches the reference energy."""
    L, I0, I1, I2 = _load_integrals()
    adets = _hf_and_singles(30)
    bdets = list(adets)
    run_anchor(
        "Anchor 1 — HF + singles K=30",
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        expected_energy=ANCHOR_ENERGIES[1],
        device=device,
    )


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_anchor2(device):
    """Anchor 2: HF + singles, K=50 basis matches the reference energy."""
    L, I0, I1, I2 = _load_integrals()
    adets = _hf_and_singles(50)
    bdets = list(adets)
    run_anchor(
        "Anchor 2 — HF + singles K=50",
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        expected_energy=ANCHOR_ENERGIES[2],
        device=device,
    )


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_anchor3(device):
    """Anchor 3: recover_configurations N_SHOTS=50 basis matches the reference energy."""
    L, I0, I1, I2 = _load_integrals()
    mol = _paths.MOLECULES["benzoic_acid"]
    adets, bdets = _build_anchor3_basis(L, mol.nelec_a, mol.nelec_b)
    run_anchor(
        "Anchor 3 — recover_configurations N_SHOTS=50",
        adets,
        bdets,
        L,
        I0,
        I1,
        I2,
        expected_energy=ANCHOR_ENERGIES[3],
        device=device,
    )


if __name__ == "__main__":
    _devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for _dev in _devs:
        test_anchor1(_dev)
        print()
        test_anchor2(_dev)
        print()
        test_anchor3(_dev)
    print("\n" + "=" * 70)
    print(f"ALL ANCHORS REPRODUCED BY TORCH PIPELINE on: {', '.join(_devs)}")
    print("=" * 70)
