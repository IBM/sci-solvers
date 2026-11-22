"""Tests for 1-RDM diagonal.

Brute-force reference: for each basis state, weight |C|² × occupancy.
Verify our vectorized version matches exactly.

Also test:
  - HF state on a benzoic acid (dataset ID 182) K=30 basis: occ_α = occ_β = [1.0]*18 + [0.0]*12.
  - Random normalized C on H4: matches scalar enumeration.
  - Sum of occ_α = n_α (electron count conservation).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import torch
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.rdm import make_one_rdm_diag_torch


def _get_fcidump():
    return str(_paths.fcidump_path("benzoic_acid"))


def _hf_and_singles(k: int) -> list[int]:
    mol = _paths.MOLECULES["benzoic_acid"]
    return _paths.hf_and_singles(mol.norb, mol.nelec_a)[:k]


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


def _brute_occ_diag(C, adets, bdets, n_orb):
    """Reference: explicit enumeration."""
    M_a, M_b = len(adets), len(bdets)
    occ_a = np.zeros(n_orb)
    occ_b = np.zeros(n_orb)
    for ia in range(M_a):
        for ib in range(M_b):
            c2 = C[ia * M_b + ib] ** 2
            a = adets[ia]
            b = bdets[ib]
            for p in range(n_orb):
                if (a >> p) & 1:
                    occ_a[p] += c2
                if (b >> p) & 1:
                    occ_b[p] += c2
    return occ_a, occ_b


def test_hf_state_benzoic_acid():
    """HF state: C = [1, 0, 0, ...] → occ_α = first 18 orbitals."""
    mol = _paths.MOLECULES["benzoic_acid"]
    L = mol.norb
    n_a = mol.nelec_a
    adets = _hf_and_singles(30)
    bdets = list(adets)
    N = len(adets) * len(bdets)
    C = torch.zeros(N, dtype=torch.float64)
    C[0] = 1.0  # HF
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    occ_a, occ_b = make_one_rdm_diag_torch(C, adets_t, bdets_t, L)
    expected = np.zeros(L)
    expected[:n_a] = 1.0
    diff_a = np.abs(occ_a.numpy() - expected).max()
    diff_b = np.abs(occ_b.numpy() - expected).max()
    print(f"  HF: occ_α diff={diff_a:.3e}, occ_β diff={diff_b:.3e}")
    print(f"      Σ occ_α = {occ_a.sum().item():.6f} (expected {n_a})")
    assert diff_a < 1e-15, f"HF occ_α off: {diff_a}"
    assert diff_b < 1e-15
    assert abs(occ_a.sum().item() - n_a) < 1e-12
    print("test_hf_state_benzoic_acid passed")


def test_random_c_h4():
    """H4 basis: random normalized C, compare vectorized vs brute-force."""
    L = 4
    n_a = n_b = 2
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    M_a, M_b = len(adets), len(bdets)
    N = M_a * M_b

    rng = np.random.default_rng(seed=5)
    C_np = rng.standard_normal(N)
    C_np /= np.linalg.norm(C_np)
    C_t = torch.tensor(C_np, dtype=torch.float64)

    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    occ_a_t, occ_b_t = make_one_rdm_diag_torch(C_t, adets_t, bdets_t, L)
    occ_a_ref, occ_b_ref = _brute_occ_diag(C_np, adets, bdets, L)

    diff_a = np.abs(occ_a_t.numpy() - occ_a_ref).max()
    diff_b = np.abs(occ_b_t.numpy() - occ_b_ref).max()
    print(f"  H4 random: occ_α diff={diff_a:.3e}, occ_β diff={diff_b:.3e}")
    print(
        f"      Σ occ_α + Σ occ_β = {(occ_a_t.sum() + occ_b_t.sum()).item():.6f} "
        f"(expected {n_a + n_b})"
    )
    assert diff_a < 1e-14
    assert diff_b < 1e-14
    assert abs((occ_a_t.sum() + occ_b_t.sum()).item() - (n_a + n_b)) < 1e-12
    print("test_random_c_h4 passed")


def test_real_eigvec_benzoic_acid_k30():
    """Realistic case: run Anchor 1 (K=30) and check the converged
    eigvec's 1-RDM diag is in [0, 1], sums to n_α/n_β, and is no longer
    purely HF (Davidson corrections must spread some weight)."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump
    from torch_sci_solver.torch_impl.sbdiag import diag_torch

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    n_a = _paths.MOLECULES["benzoic_acid"].nelec_a
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])
    adets = _hf_and_singles(30)
    bdets = list(adets)

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
        device="cpu",
        verbose=False,
    )
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    occ_a, occ_b = make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, L)

    print(
        f"  Anchor 1 occ_α: min={occ_a.min().item():.4f}, "
        f"max={occ_a.max().item():.4f}, sum={occ_a.sum().item():.4f}"
    )
    print(f"  first 5 α: {occ_a[:5].tolist()}")
    print(f"  last  5 α: {occ_a[-5:].tolist()}")
    assert occ_a.min().item() >= -1e-12
    assert occ_a.max().item() <= 1.0 + 1e-12
    assert abs(occ_a.sum().item() - n_a) < 1e-10
    # And not purely HF: some weight should leak into virtual orbitals
    # (orbitals 18..29) because of the CIS-like singles in the basis.
    virtuals = occ_a[n_a:]
    assert virtuals.max().item() > 1e-10, (
        f"virtual occupations all ≈ 0; eigvec was likely pure HF "
        f"(max virt occ = {virtuals.max().item():.3e})"
    )
    print("test_real_eigvec_benzoic_acid_k30 passed")


if __name__ == "__main__":
    test_hf_state_benzoic_acid()
    test_random_c_h4()
    test_real_eigvec_benzoic_acid_k30()
    print("\nALL TESTS PASSED")
