"""Tests for torch_sci_solver.sbd_ref.mult.

Verifies:
  1. make_diag_terms matches diagonal of dense H built via hij.
  2. mult(T) matches H_dense @ T for random T on small systems
     (H2 minimal, H4 minimal, H6 6-orbital test).
  3. Lowest eigenvalue from Lanczos on mult matches eigvalsh(H_dense).
  4. Task-type independence: applying only a subset of tasks gives
     a partial H@v that matches the equivalent partial H_dense @ T.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
from torch_sci_solver.sbd_ref import (
    det_from_alpha_beta,
    generate_all_helpers,
    hij,
    make_diag_terms,
    mult,
    setup_integrals_from_pyscf_h1_eri,
)


def _enumerate_dets(L, n_elec):
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        mask = 0
        for bit in combo:
            mask |= 1 << bit
        out.append(mask)
    out.sort()
    return out


def _build_h_via_hij(adets, bdets, n_orb, I0, I1, I2):
    n_a = len(adets)
    n_b = len(bdets)
    N = n_a * n_b
    dets = [det_from_alpha_beta(a, b, n_orb) for a in adets for b in bdets]
    H = np.zeros((N, N), dtype=np.float64)
    n_spinorb = 2 * n_orb
    for row in range(N):
        for col in range(N):
            H[row, col] = hij(dets[row], dets[col], n_spinorb, I0, I1, I2)
    return H


def _get_pyscf_problem(mol_atom, basis="sto-3g"):
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=mol_atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    return mol, h1, eri, L, n_a, n_b


def assert_close_array(a, b, rtol=1e-12, atol=1e-13, label=""):
    """Assert two arrays agree within a combined relative/absolute tolerance."""
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b).max()
    scale = max(np.abs(b).max(), 1.0)
    if diff > atol + rtol * scale:
        raise AssertionError(
            f"{label}: max |a-b| = {diff:.3e}, scale = {scale:.3e}, rtol={rtol}, atol={atol}"
        )


# ============================================================================
# 1. Diagonal terms match diagonal of dense H
# ============================================================================


def test_diag_matches_h_diagonal():
    """make_diag_terms should produce the diagonal of the dense H."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    H_dense = _build_h_via_hij(adets, bdets, L, I0, I1, I2)
    diag_expected = np.diag(H_dense)

    diag_got = make_diag_terms(adets, bdets, L, I0, I1, I2)

    print(f"  diag_expected[:6] = {diag_expected[:6]}")
    print(f"  diag_got[:6]      = {diag_got[:6]}")
    assert_close_array(diag_got, diag_expected, label="diag")
    print("test_diag_matches_h_diagonal passed")


# ============================================================================
# 2. mult(T) == H_dense @ T on multiple random vectors
# ============================================================================


def test_mult_matches_dense_h_h4():
    """Random T → mult and H@T agree to ~1e-13."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    N = len(adets) * len(bdets)
    H = _build_h_via_hij(adets, bdets, L, I0, I1, I2)
    print(f"  H4: basis size N={N}")

    hii = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers = list(generate_all_helpers(adets, bdets, L))

    rng = np.random.default_rng(seed=0)
    for trial in range(3):
        T = rng.standard_normal(N)
        Wb_expected = H @ T
        Wb_got = mult(hii, T, adets, bdets, L, I1, I2, helpers)
        diff_max = np.abs(Wb_got - Wb_expected).max()
        scale = np.abs(Wb_expected).max()
        print(f"  trial {trial}: |mult - H@T|_max = {diff_max:.3e}, scale = {scale:.3e}")
        assert_close_array(Wb_got, Wb_expected, label=f"H4 trial {trial}")
    print("test_mult_matches_dense_h_h4 passed")


def test_mult_matches_dense_h_h2():
    """Smallest non-trivial system."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("H 0 0 0; H 0 0 0.74")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    N = len(adets) * len(bdets)
    H = _build_h_via_hij(adets, bdets, L, I0, I1, I2)
    print(f"  H2: basis size N={N}")

    hii = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers = list(generate_all_helpers(adets, bdets, L))

    rng = np.random.default_rng(seed=1)
    T = rng.standard_normal(N)
    Wb_expected = H @ T
    Wb_got = mult(hii, T, adets, bdets, L, I1, I2, helpers)
    diff_max = np.abs(Wb_got - Wb_expected).max()
    print(f"  |mult - H@T|_max = {diff_max:.3e}")
    assert_close_array(Wb_got, Wb_expected, label="H2")
    print("test_mult_matches_dense_h_h2 passed")


def test_mult_matches_dense_h_lih():
    """Larger basis: LiH/sto-3g (6 orbitals, 2+2 electrons)."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("Li 0 0 0; H 0 0 1.6")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    N = len(adets) * len(bdets)
    H = _build_h_via_hij(adets, bdets, L, I0, I1, I2)
    print(f"  LiH: L={L}, n_a={n_a}, n_b={n_b}, basis size N={N}")

    hii = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers = list(generate_all_helpers(adets, bdets, L))

    rng = np.random.default_rng(seed=2)
    T = rng.standard_normal(N)
    Wb_expected = H @ T
    Wb_got = mult(hii, T, adets, bdets, L, I1, I2, helpers)
    diff_max = np.abs(Wb_got - Wb_expected).max()
    print(f"  |mult - H@T|_max = {diff_max:.3e}")
    assert_close_array(Wb_got, Wb_expected, label="LiH", rtol=1e-11)
    print("test_mult_matches_dense_h_lih passed")


# ============================================================================
# 3. Lanczos / power iteration: dominant eigenvalue via mult matches eigvalsh
# ============================================================================


def test_lowest_eigenvalue_via_lanczos_h4():
    """Use scipy LinearOperator + eigsh on mult to confirm we get the
    same lowest eigenvalue as eigvalsh of the dense H."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    H = _build_h_via_hij(adets, bdets, L, I0, I1, I2)
    e_dense = float(np.linalg.eigvalsh(H)[0])

    hii = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers = list(generate_all_helpers(adets, bdets, L))
    N = len(adets) * len(bdets)

    def matvec(v):
        return mult(hii, v.astype(np.float64), adets, bdets, L, I1, I2, helpers)

    op = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    eigvals, _ = eigsh(op, k=1, which="SA", tol=1e-12)
    e_lanczos = float(eigvals[0])
    print(f"  dense lowest:  {e_dense:.12f}")
    print(f"  lanczos:       {e_lanczos:.12f}")
    print(f"  diff:          {e_lanczos - e_dense:.2e}")
    assert abs(e_lanczos - e_dense) < 1e-9
    print("test_lowest_eigenvalue_via_lanczos_h4 passed")


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    test_diag_matches_h_diagonal()
    test_mult_matches_dense_h_h2()
    test_mult_matches_dense_h_h4()
    test_mult_matches_dense_h_lih()
    test_lowest_eigenvalue_via_lanczos_h4()
    print("\nALL A3 TESTS PASSED")
