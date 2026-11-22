"""Tests for torch_sci_solver.sbd_ref.

Verifies:
  1. bit accessors (getocc/setocc/bitcount) — trivial sanity.
  2. det_from_alpha_beta — interleaving correct.
  3. orbital_difference + difference — correctly identify differing orbitals.
  4. parity for known fermion-permutation cases.
  5. ZeroExcite / OneExcite / TwoExcite / Hij correctly construct H on a
     small system; FCI ground state matches PySCF FCI to ~1e-12 relative.
  6. Hermiticity of H (real case): hij(A, B) == hij(B, A).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
from torch_sci_solver.sbd_ref import (
    bitcount,
    det_from_alpha_beta,
    difference,
    getocc,
    hij,
    orbital_difference,
    parity,
    setocc,
    setup_integrals_from_pyscf_h1_eri,
)


def assert_close(actual, expected, rtol=1e-10, atol=1e-12, label=""):
    """Raise if not close."""
    a = float(actual)
    e = float(expected)
    if abs(a - e) > atol + rtol * abs(e):
        raise AssertionError(
            f"{label}: actual={a!r}, expected={e!r}, diff={a - e!r}, rtol={rtol}, atol={atol}"
        )


# ============================================================================
# 1. Bit accessors
# ============================================================================


def test_bits():
    """getocc/setocc/bitcount bit-level accessors."""
    # 0b 1011 = 11
    assert getocc(0b1011, 0) is True
    assert getocc(0b1011, 1) is True
    assert getocc(0b1011, 2) is False
    assert getocc(0b1011, 3) is True
    assert getocc(0b1011, 10) is False

    assert setocc(0, 5, True) == 32
    assert setocc(0b11111, 2, False) == 0b11011

    assert bitcount(0b11011, 5) == 4
    assert bitcount(0b11111, 3) == 3  # mask off high bits
    print("test_bits passed")


# ============================================================================
# 2. det_from_alpha_beta
# ============================================================================


def test_det_from_alpha_beta():
    """Interleaving α/β strings into a spin-orbital determinant bitmask."""
    # 4 spatial orbitals. α-bits 0..3, β-bits 0..3.
    # α-string 0b0101 means α-orbitals 0 and 2 occupied → spin-orbitals 0, 4
    # β-string 0b1100 means β-orbitals 2 and 3 occupied → spin-orbitals 5, 7
    # Expected interleaved: bits 0, 4, 5, 7
    L = 4
    a = 0b0101  # α at spatial 0, 2
    b = 0b1100  # β at spatial 2, 3
    det = det_from_alpha_beta(a, b, L)
    expected = (1 << 0) | (1 << 4) | (1 << 5) | (1 << 7)
    assert det == expected, f"det={bin(det)}, expected={bin(expected)}"

    # Edge: all-zero
    assert det_from_alpha_beta(0, 0, L) == 0

    # Edge: full
    assert det_from_alpha_beta(0b1111, 0b1111, L) == (1 << (2 * L)) - 1

    print("test_det_from_alpha_beta passed")


# ============================================================================
# 3. orbital_difference + difference
# ============================================================================


def test_orbital_difference():
    """difference()/orbital_difference() report occupied-orbital deltas between two dets."""
    # Single excitation: orbital 2 → orbital 5
    # a = 0b 00100 = 4    (bit 2 set)
    # b = 0b 10000 = 16   (bit 4 set)
    a = 0b00100
    b = 0b10000
    assert difference(a, b, 5) == 2
    x, y = orbital_difference(a, b, 5)
    assert x == [2]
    assert y == [4]

    # No difference
    assert difference(0b11011, 0b11011, 5) == 0
    x, y = orbital_difference(0b11011, 0b11011, 5)
    assert x == [] and y == []

    # Two-orbital diff
    a = 0b110
    b = 0b011
    assert difference(a, b, 3) == 2
    x, y = orbital_difference(a, b, 3)
    assert x == [2]  # only in a
    assert y == [0]  # only in b

    # Mask should ignore high bits beyond n_bits
    a = 0b11111111  # 8 bits set
    b = 0
    assert difference(a, b, 4) == 4  # only bits 0..3 counted

    print("test_orbital_difference passed")


# ============================================================================
# 4. parity
# ============================================================================


def test_parity_simple():
    """parity() sign convention for excitations with 0/1/2 occupied orbitals between."""
    # Empty between → sign unchanged
    sgn = parity(0b00100100, 2, 5, 1.0)
    # Range [2, 5) of det=...100100: bits 2,3,4 = 1,0,0 → popcount=1 → sgn*=-1
    # bit at start (2) is 1 → additional *-1
    # Net: +1
    assert_close(sgn, 1.0, label="parity full-occ start")

    # One bit between
    sgn = parity(0b00010000, 1, 6, 1.0)
    # det bits 1..5: 0,0,0,1,0 → popcount=1 → sgn=-1
    # bit at 1 is 0 → no extra flip
    # Net: -1
    assert_close(sgn, -1.0, label="parity one between")

    # No bits in range, start unset → +1
    sgn = parity(0b1010, 1, 3, 1.0)
    # det bits 1, 2: 1, 0 → popcount=1 → sgn=-1
    # bit at 1 is 1 → extra -1
    # Net: +1
    assert_close(sgn, 1.0, label="parity start set")

    print("test_parity_simple passed")


# ============================================================================
# 5+6. Full H matrix construction + FCI vs PySCF
# ============================================================================


def _enumerate_dets(L: int, n_elec: int):
    """All bitmasks of length L with exactly n_elec bits set, sorted ascending."""
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        mask = 0
        for bit in combo:
            mask |= 1 << bit
        out.append(mask)
    out.sort()
    return out


def _build_h_matrix_via_hij(adets, bdets, n_spinorb, I0, I1, I2):
    """Build the full H matrix for the Cartesian-product basis by
    pairwise hij calls. O(N²) — tests only.
    """
    n_a = len(adets)
    n_b = len(bdets)
    N = n_a * n_b
    H = np.zeros((N, N), dtype=np.float64)
    L = n_spinorb // 2
    dets = [det_from_alpha_beta(a, b, L) for a in adets for b in bdets]
    for row in range(N):
        for col in range(N):
            H[row, col] = hij(dets[row], dets[col], n_spinorb, I0, I1, I2)
    return H


def test_slater_condon_vs_pyscf_4orb():
    """4-orbital random RHF system: full H matrix built via hij,
    lowest eigenvalue compared to PySCF FCI."""
    from pyscf import ao2mo, fci, gto, scf

    # Random small molecule that yields nontrivial CI
    # Use H4 in a minimal basis — gives 4 spatial orbitals from STO-3G.
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    assert L == 4, f"expected 4 orbitals, got {L}"
    n_a = mol.nelec[0]
    n_b = mol.nelec[1]
    assert n_a == 2 and n_b == 2

    # Get h1 and eri in MO basis
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)  # 4-fold packed
    eri_full = ao2mo.restore(1, eri_mo, L)  # (L,L,L,L)
    core_energy = mol.energy_nuc()

    # PySCF FCI reference
    cisolver = fci.FCI(mf)
    e_pyscf, _ = cisolver.kernel(h1, eri_full, L, (n_a, n_b), ecore=core_energy)
    print(f"  PySCF FCI ground state: {e_pyscf:.12f}")

    # Build our integrals + full H
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri_full, core_energy)
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    H = _build_h_matrix_via_hij(adets, bdets, 2 * L, I0, I1, I2)
    print(
        f"  H shape: {H.shape}, full Cartesian basis size {len(adets)}×{len(bdets)} = {len(adets) * len(bdets)}"
    )

    # Hermitian?
    sym_err = np.abs(H - H.T).max()
    print(f"  H Hermiticity max err: {sym_err:.2e}")
    assert sym_err < 1e-12, f"H not Hermitian: max |H-H^T| = {sym_err}"

    # Lowest eigenvalue must match PySCF FCI
    eigvals = np.linalg.eigvalsh(H)
    e_ours = float(eigvals[0])
    print(f"  ours (eigvalsh of full H built from hij): {e_ours:.12f}")
    print(f"  diff: {e_ours - e_pyscf:.2e}")
    assert_close(e_ours, e_pyscf, rtol=1e-10, label="FCI energy via Hij")

    print("test_slater_condon_vs_pyscf_4orb passed")


def test_h2_minimal():
    """Tiny 2-orbital H2 sanity check — fully analytic structure."""
    from pyscf import ao2mo, fci, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    assert L == 2
    n_a, n_b = mol.nelec

    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff)
    eri_full = ao2mo.restore(1, eri_mo, L)
    core = mol.energy_nuc()

    cisolver = fci.FCI(mf)
    e_pyscf, _ = cisolver.kernel(h1, eri_full, L, (n_a, n_b), ecore=core)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri_full, core)
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    H = _build_h_matrix_via_hij(adets, bdets, 2 * L, I0, I1, I2)

    e_ours = float(np.linalg.eigvalsh(H)[0])
    print(f"  H2 FCI: pyscf={e_pyscf:.12f}, ours={e_ours:.12f}, diff={e_ours - e_pyscf:.2e}")
    assert_close(e_ours, e_pyscf, rtol=1e-10, label="H2 FCI")
    print("test_h2_minimal passed")


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    test_bits()
    test_det_from_alpha_beta()
    test_orbital_difference()
    test_parity_simple()
    test_h2_minimal()
    test_slater_condon_vs_pyscf_4orb()
    print("\nALL A1 TESTS PASSED")
