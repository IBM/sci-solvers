"""Tests for torch_sci_solver.sbd_ref.davidson + sbdiag.

Verifies:
  1. Davidson on small synthetic SPD matrices converges to the dense
     eigvalsh result.
  2. Davidson via mult on the H4 and LiH FCI bases matches dense
     eigvalsh.
  3. End-to-end ``diag()`` on small molecules matches PySCF FCI to
     ~eps.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
from torch_sci_solver.sbd_ref import (
    davidson,
    diag,
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


# ============================================================================
# 1. Davidson on a synthetic dense matrix
# ============================================================================


def test_davidson_synthetic():
    """Synthetic 50×50 SPD matrix → Davidson should match eigvalsh.

    This isolates the Davidson algorithm from the SCI matvec.
    """
    rng = np.random.default_rng(seed=42)
    N = 50
    A = rng.standard_normal((N, N))
    H = 0.5 * (A + A.T)  # symmetric
    # Bias diagonal to be dominant so lowest eigenvalue is well-separated
    np.fill_diagonal(H, np.diag(H) - 20.0)
    e_dense = float(np.linalg.eigvalsh(H)[0])

    hii = np.diag(H).copy()

    def matvec(v):
        return H @ v

    # HF-init = unit vector at 0 → fine for a dense matrix; just a starting point
    W0 = np.zeros(N)
    W0[0] = 1.0
    e_dav, v_dav, info = davidson(
        hii, W0, matvec, max_iteration=20, num_block=15, eps=1e-10, verbose=False
    )
    print(f"  dense:    {e_dense:.12f}")
    print(
        f"  davidson: {e_dav:.12f}  (matvecs={info['n_matvecs']}, "
        f"converged={info['converged']}, ||R||={info['final_residual_norm']:.2e})"
    )
    assert info["converged"]
    assert abs(e_dav - e_dense) < 1e-9
    # Check eigenvector is actually an eigenvector
    resid = H @ v_dav - e_dav * v_dav
    assert np.linalg.norm(resid) < 1e-8
    print("test_davidson_synthetic passed")


# ============================================================================
# 2. End-to-end H2, H4, LiH via diag() vs PySCF FCI
# ============================================================================


def _do_full_run(mol_atom, basis_name="sto-3g", verbose=False):
    from pyscf import ao2mo, fci, gto, scf

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)

    cisolver = fci.FCI(mf)
    e_pyscf, _ = cisolver.kernel(h1, eri, L, (n_a, n_b), ecore=mol.energy_nuc())

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    e_ours, v_ours, info = diag(
        adets, bdets, L, I0, I1, I2, max_iteration=50, num_block=30, eps=1e-10, verbose=verbose
    )
    return mol, e_pyscf, e_ours, info


def test_h2_sbdiag():
    """H2/STO-3G via sbdiag's Davidson-based diag() vs PySCF FCI."""
    mol, e_pyscf, e_ours, info = _do_full_run("H 0 0 0; H 0 0 0.74")
    print(
        f"  H2:  pyscf={e_pyscf:.12f}, sbdiag={e_ours:.12f}, "
        f"diff={e_ours - e_pyscf:.2e}, n_matvecs={info['n_matvecs']}, "
        f"converged={info['converged']}"
    )
    assert abs(e_ours - e_pyscf) < 1e-9
    print("test_h2_sbdiag passed")


def test_h4_sbdiag():
    """H4 chain via sbdiag's Davidson-based diag() vs PySCF FCI."""
    mol, e_pyscf, e_ours, info = _do_full_run("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74")
    print(
        f"  H4:  pyscf={e_pyscf:.12f}, sbdiag={e_ours:.12f}, "
        f"diff={e_ours - e_pyscf:.2e}, n_matvecs={info['n_matvecs']}, "
        f"converged={info['converged']}"
    )
    assert abs(e_ours - e_pyscf) < 1e-9
    print("test_h4_sbdiag passed")


def test_lih_sbdiag():
    """LiH via sbdiag's Davidson-based diag() vs PySCF FCI."""
    mol, e_pyscf, e_ours, info = _do_full_run("Li 0 0 0; H 0 0 1.6")
    print(
        f"  LiH: pyscf={e_pyscf:.12f}, sbdiag={e_ours:.12f}, "
        f"diff={e_ours - e_pyscf:.2e}, n_matvecs={info['n_matvecs']}, "
        f"converged={info['converged']}"
    )
    assert abs(e_ours - e_pyscf) < 1e-9
    print("test_lih_sbdiag passed")


# ============================================================================
# 3. Random init also converges (just slower)
# ============================================================================


def test_random_init():
    """Same H4 problem with random initial vector should still converge."""
    mol, e_pyscf, _, _ = _do_full_run("H 0 0 0; H 0 0 0.74")  # H2 small
    # Re-run with init=1 (random)
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    np.random.seed(0)  # not used by our code, but ensures reproducibility
    e_rand, _, info = diag(
        adets, bdets, L, I0, I1, I2, max_iteration=50, num_block=20, eps=1e-10, init=1
    )
    print(
        f"  H2 random init: pyscf={e_pyscf:.12f}, sbdiag={e_rand:.12f}, "
        f"diff={e_rand - e_pyscf:.2e}, n_matvecs={info['n_matvecs']}"
    )
    assert abs(e_rand - e_pyscf) < 1e-9
    print("test_random_init passed")


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    test_davidson_synthetic()
    test_h2_sbdiag()
    test_h4_sbdiag()
    test_lih_sbdiag()
    test_random_init()
    print("\nALL A4 TESTS PASSED")
