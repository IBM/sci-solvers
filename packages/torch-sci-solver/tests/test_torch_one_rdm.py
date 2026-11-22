"""Tests for full 1-RDM construction (off-diagonals).

Strategy: full FCI on H2, H4, LiH via PySCF, then compute the 1-RDM
two ways and compare:

  1. PySCF's ``fci.direct_spin1.make_rdm1s(civec, norb, nelec)`` →
     reference (rdm1a, rdm1b).
  2. Our ``make_one_rdm_torch`` on the same civec, mapped onto our
     Cartesian-product basis. The two should agree to machine precision.

For each test we also verify:
  - γα.diag().sum() == n_α (electron count conservation).
  - γα is symmetric (for real C).
  - The diagonals match make_one_rdm_diag_torch exactly.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import torch
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import (
    generate_all_torch_helpers,
    make_one_rdm_diag_torch,
    make_one_rdm_torch,
)


def _enumerate_dets(L, n_elec):
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        m = 0
        for b in combo:
            m |= 1 << b
        out.append(m)
    return sorted(out)


def _to_int64(values):
    INT64_OFFSET = 1 << 64
    return [int(v) if int(v) < (1 << 63) else int(v) - INT64_OFFSET for v in values]


def _civec_pyscf_to_ours(civec_pyscf, adets, bdets, n_orb, n_a, n_b):
    """PySCF FCI civec is (n_str_a, n_str_b) over PySCF's det ordering;
    map to our flat (M_a*M_b,) ordering on (adets[ia], bdets[ib])."""
    from pyscf.fci import cistring

    # PySCF's strings are sorted ints (same as our sorting)
    strings_a = list(cistring.gen_strings4orblist(range(n_orb), n_a))
    strings_b = list(cistring.gen_strings4orblist(range(n_orb), n_b))
    # Our adets is also sorted; they should match (closed-shell H4)
    assert sorted(strings_a) == sorted(adets), (
        f"PySCF α-strings {strings_a[:5]} vs ours {adets[:5]}"
    )
    # Build the mapping
    pyscf_to_idx_a = {s: i for i, s in enumerate(strings_a)}
    pyscf_to_idx_b = {s: i for i, s in enumerate(strings_b)}
    M_a, M_b = len(adets), len(bdets)
    C = np.zeros(M_a * M_b, dtype=np.float64)
    for ia in range(M_a):
        ipyscf_a = pyscf_to_idx_a[adets[ia]]
        for ib in range(M_b):
            ipyscf_b = pyscf_to_idx_b[bdets[ib]]
            C[ia * M_b + ib] = civec_pyscf[ipyscf_a, ipyscf_b]
    return C


def _test_full_fci_rdm(mol_atom, n_a_b=None, label=""):
    """Run PySCF FCI on a small system; compare 1-RDM against ours."""
    from pyscf import ao2mo, fci, gto, scf

    mol = gto.M(atom=mol_atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    if n_a_b is None:
        n_a, n_b = mol.nelec
    else:
        n_a, n_b = n_a_b
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)

    cisolver = fci.direct_spin1.FCI()
    e_fci, civec = cisolver.kernel(h1, eri, L, (n_a, n_b))

    # PySCF's 1-RDM (chemist convention)
    rdm1a_ref, rdm1b_ref = cisolver.make_rdm1s(civec, L, (n_a, n_b))

    # Now reconstruct our basis and our C vector
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    C_np = _civec_pyscf_to_ours(civec, adets, bdets, L, n_a, n_b)

    # Set up torch tensors
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(adets, bdets, L, device="cpu"))
    eigvec = torch.tensor(C_np, dtype=torch.float64)

    # 1. Diagonal check
    occ_a, occ_b = make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, L)
    diag_diff_a = np.abs(occ_a.numpy() - np.diag(rdm1a_ref)).max()
    diag_diff_b = np.abs(occ_b.numpy() - np.diag(rdm1b_ref)).max()
    print(f"  {label}: diag γα diff = {diag_diff_a:.3e}, diag γβ diff = {diag_diff_b:.3e}")
    assert diag_diff_a < 1e-12, f"{label}: diag γα mismatch"
    assert diag_diff_b < 1e-12, f"{label}: diag γβ mismatch"

    # 2. Full RDM check
    gamma_a, gamma_b = make_one_rdm_torch(
        eigvec, adets_t, bdets_t, L, helpers, eigvec_in_interleaved_wick=False
    )
    full_diff_a = np.abs(gamma_a.numpy() - rdm1a_ref).max()
    full_diff_b = np.abs(gamma_b.numpy() - rdm1b_ref).max()
    print(f"  {label}: full γα diff = {full_diff_a:.3e}, full γβ diff = {full_diff_b:.3e}")
    print(f"  {label}: γα symmetric err = {np.abs(gamma_a.numpy() - gamma_a.numpy().T).max():.3e}")
    print(f"  {label}: trace(γα) = {gamma_a.trace().item():.6f}, expected {n_a}")
    print(f"  {label}: trace(γβ) = {gamma_b.trace().item():.6f}, expected {n_b}")
    assert full_diff_a < 1e-12, f"{label}: γα off-diag mismatch"
    assert full_diff_b < 1e-12, f"{label}: γβ off-diag mismatch"

    # 3. Electron count
    assert abs(gamma_a.trace().item() - n_a) < 1e-12
    assert abs(gamma_b.trace().item() - n_b) < 1e-12

    # 4. Symmetry
    sym_err_a = np.abs(gamma_a.numpy() - gamma_a.numpy().T).max()
    assert sym_err_a < 1e-13, f"{label}: γα not symmetric ({sym_err_a:.3e})"
    print(f"  {label} ✓ all checks pass\n")


def test_h2():
    """Torch one-RDM matches PySCF FCI's for H2."""
    _test_full_fci_rdm("H 0 0 0; H 0 0 0.74", label="H2")


def test_h4():
    """Torch one-RDM matches PySCF FCI's for H4."""
    _test_full_fci_rdm("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", label="H4")


def test_lih():
    """Torch one-RDM matches PySCF FCI's for LiH."""
    _test_full_fci_rdm("Li 0 0 0; H 0 0 1.6", label="LiH")


if __name__ == "__main__":
    test_h2()
    test_h4()
    test_lih()
    print("ALL FULL-1-RDM TESTS PASSED")
