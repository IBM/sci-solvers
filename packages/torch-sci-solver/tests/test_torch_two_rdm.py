"""Tests for 2-RDM construction.

Validates Γaa, Γab, Γbb against PySCF on small systems and validates
energy reconstruction (Davidson eigenvalue == I0 + Σ h γ + (1/2) Σ (pq|rs) Γ)
on benzoic acid (SQD molecules dataset ID 182; see test_torch_anchors.py).


Expected: PySCF diffs ≤ 5e-15 on all blocks; Anchor 3 energy
reconstruction agrees with Davidson eigenvalue to ≤ 5e-10 Ha.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import sys
import time

import numpy as np
import torch
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import (
    energy_from_rdm,
    generate_all_torch_helpers,
    make_one_rdm_torch,
    make_two_rdm_torch,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.sbdiag import diag_torch


def _torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Part A — PySCF small-system validation
# ============================================================================


def _civec_pyscf_to_ours(civec_pyscf, adets, bdets, n_orb, n_a, n_b):
    from pyscf.fci import cistring

    strings_a = list(cistring.gen_strings4orblist(range(n_orb), n_a))
    strings_b = list(cistring.gen_strings4orblist(range(n_orb), n_b))
    idx_a = {int(s): i for i, s in enumerate(strings_a)}
    idx_b = {int(s): i for i, s in enumerate(strings_b)}
    Ma, Mb = len(adets), len(bdets)
    out = np.zeros(Ma * Mb)
    for ia in range(Ma):
        for ib in range(Mb):
            out[ia * Mb + ib] = civec_pyscf[idx_a[adets[ia]], idx_b[bdets[ib]]]
    return out


def _enumerate_dets(L, n_elec):
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        m = 0
        for b in combo:
            m |= 1 << b
        out.append(m)
    return sorted(out)


def _run_pyscf_validation(name, mol, atol_rdm=5e-15, atol_e=5e-14):
    from pyscf import ao2mo, fci, scf

    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1_sp = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_sp = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    cisolver = fci.direct_spin1.FCI()
    e_fci, civec = cisolver.kernel(h1_sp, eri_sp, L, (n_a, n_b))
    rdm1_p, rdm2_p = cisolver.make_rdm12s(civec, L, (n_a, n_b))
    Gaa_p, Gab_p, Gbb_p = rdm2_p
    gamma_a_p, gamma_b_p = rdm1_p

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    C = _civec_pyscf_to_ours(civec, adets, bdets, L, n_a, n_b)

    helpers = generate_all_torch_helpers(adets, bdets, L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1_sp, eri_sp, 0.0)
    integrals = to_torch_integrals(I0, I1, I2)
    adets_t = torch.tensor(adets, dtype=torch.int64)
    bdets_t = torch.tensor(bdets, dtype=torch.int64)
    eigvec = torch.tensor(C, dtype=torch.float64)

    gamma_a, gamma_b = make_one_rdm_torch(
        eigvec, adets_t, bdets_t, L, helpers, eigvec_in_interleaved_wick=False
    )
    Gaa, Gab, Gbb = make_two_rdm_torch(
        eigvec, adets_t, bdets_t, L, helpers, eigvec_in_interleaved_wick=False
    )
    E_rec = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, integrals).item()

    da = float(np.abs(gamma_a.numpy() - gamma_a_p).max())
    db = float(np.abs(gamma_b.numpy() - gamma_b_p).max())
    daa = float(np.abs(Gaa.numpy() - Gaa_p).max())
    dab = float(np.abs(Gab.numpy() - Gab_p).max())
    dbb = float(np.abs(Gbb.numpy() - Gbb_p).max())
    de = abs(e_fci - E_rec)
    ok = max(da, db, daa, dab, dbb) <= atol_rdm and de <= atol_e
    print(
        f"  {name:10s} L={L} n=({n_a},{n_b}) | γα {da:.1e} γβ {db:.1e} | "
        f"Γaa {daa:.1e} Γab {dab:.1e} Γbb {dbb:.1e} | E {de:.1e}  "
        f"{'OK' if ok else 'FAIL'}"
    )
    return ok


def test_part_a():
    """Cross-validate the two-RDM energy reconstruction against PySCF on H2/H4/LiH."""
    from pyscf import gto

    print("Part A: PySCF cross-validation")
    cases = [
        ("H2", gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)),
        ("H4", gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", verbose=0)),
        (
            "H4_trip",
            gto.M(
                atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", spin=2, verbose=0
            ),
        ),
        ("LiH", gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)),
    ]
    ok = all(_run_pyscf_validation(name, mol) for name, mol in cases)
    print(f"  Part A: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ============================================================================
# Part B — benzoic acid Anchor 3: Davidson vs energy_from_rdm
# ============================================================================


def _load_integrals():
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    p = _paths.fcidump_path("benzoic_acid")
    if p is None:
        return None
    raw = fcidump.read(str(p))
    L = raw["NORB"]
    assert L == _paths.MOLECULES["benzoic_acid"].norb
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    return L, I0, I1, I2


def _build_anchor3_basis(L: int, n_a: int, n_b: int):
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
    from torch_sci_solver.shots import load_shots

    shots_p = _paths.shots_path("benzoic_acid")
    if shots_p is None:
        return None
    shots = load_shots(str(shots_p))
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


def test_part_b():
    """Benzoic acid Anchor 3: Davidson energy vs energy_from_rdm reconstruction."""
    print("Part B: benzoic acid Anchor 3 — Davidson vs RDM energy reconstruction")

    integrals_data = _load_integrals()
    if integrals_data is None:
        print("  SKIP: FCIDUMP not found.\n")
        return True
    L, I0, I1, I2 = integrals_data

    mol = _paths.MOLECULES["benzoic_acid"]
    basis = _build_anchor3_basis(L, mol.nelec_a, mol.nelec_b)
    if basis is None:
        print("  SKIP: shots fixture not found.\n")
        return True
    adets, bdets = basis
    Ma = len(adets)
    Mb = len(bdets)
    print(f"  basis size: Ma × Mb = {Ma} × {Mb} = {Ma * Mb}")

    device = _torch_device()
    print(f"  device = {device}")

    t0 = time.perf_counter()
    energy_dav, eigvec_dav, info = diag_torch(
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
        verbose=False,
    )
    t_dav = time.perf_counter() - t0
    print(f"  Davidson: E = {energy_dav:.10f} Ha, t = {t_dav:.1f}s")

    # Build helpers + integrals on the same device for RDM construction
    helpers = generate_all_torch_helpers(adets, bdets, L, device=device)
    integrals = to_torch_integrals(I0, I1, I2, device=device)
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    eigvec = eigvec_dav.to(device).to(torch.float64)

    t0 = time.perf_counter()
    gamma_a, gamma_b = make_one_rdm_torch(eigvec, adets_t, bdets_t, L, helpers)
    t_g1 = time.perf_counter() - t0
    n_a_eff = bin(adets[0]).count("1")
    tr_a = gamma_a.diagonal().sum().item()
    tr_b = gamma_b.diagonal().sum().item()
    print(
        f"  1-RDM: t = {t_g1:.2f}s, tr(γα) = {tr_a:.4f}, tr(γβ) = {tr_b:.4f}  (n_α=n_β={n_a_eff})"
    )

    t0 = time.perf_counter()
    Gaa, Gab, Gbb = make_two_rdm_torch(eigvec, adets_t, bdets_t, L, helpers)
    t_g2 = time.perf_counter() - t0
    trace_aa = torch.einsum("pprr->", Gaa).item()
    trace_ab = torch.einsum("pprr->", Gab).item()
    trace_bb = torch.einsum("pprr->", Gbb).item()
    pauli_aa = Gaa.diagonal(dim1=0, dim2=2).diagonal(dim1=0, dim2=1).abs().max().item()
    pauli_bb = Gbb.diagonal(dim1=0, dim2=2).diagonal(dim1=0, dim2=1).abs().max().item()
    print(f"  2-RDM: t = {t_g2:.2f}s")
    print(f"    Σ Γaa[p,p,r,r] = {trace_aa:.4f}  (expect {n_a_eff * (n_a_eff - 1)})")
    print(f"    Σ Γab[p,p,r,r] = {trace_ab:.4f}  (expect {n_a_eff * n_a_eff})")
    print(f"    Σ Γbb[p,p,r,r] = {trace_bb:.4f}  (expect {n_a_eff * (n_a_eff - 1)})")
    print(f"    max |Γaa[p,p,p,p]| = {pauli_aa:.3e}  (Pauli; expect 0)")
    print(f"    max |Γbb[p,p,p,p]| = {pauli_bb:.3e}  (Pauli; expect 0)")

    E_rec = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, integrals).item()
    diff = abs(energy_dav - E_rec)
    print(f"  E_Davidson   = {energy_dav:.10f}")
    print(f"  E_from_RDM   = {E_rec:.10f}")
    print(f"  diff         = {diff:.3e} Ha")
    ok = diff <= 5e-10
    print(f"  Part B: {'PASS' if ok else 'FAIL'}\n")
    return ok


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"torch.__version__ = {torch.__version__}")
    print(f"cuda available    = {torch.cuda.is_available()}")
    print()

    ok_a = test_part_a()
    ok_b = test_part_b()

    final = "PASS" if (ok_a and ok_b) else "FAIL"
    print(f"OVERALL: {final}")
    sys.exit(0 if (ok_a and ok_b) else 1)
