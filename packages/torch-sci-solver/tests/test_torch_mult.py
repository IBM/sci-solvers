"""Tests for torch_sci_solver.torch_impl.mult.

Verifies:
  1. make_diag_terms_torch matches scalar make_diag_terms (machine precision).
  2. mult_torch(T) matches mult(T) of the scalar reference on H2, H4, LiH
     for random T (machine precision).
  3. mult_torch on a realistic 30-orbital basis (benzoic acid, dataset ID 182; first 20 HF + singles
     α-strings) matches the scalar mult. Bitmasks wider than 64 bits are covered by the
     synthetic multi-limb tests in test_torch_determinants.py.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

# scalar reference
import numpy as np
import torch
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import (
    generate_all_helpers,
    make_diag_terms,
    mult,
    setup_integrals_from_pyscf_h1_eri,
)

# torch impl
from torch_sci_solver.torch_impl import (
    generate_all_torch_helpers,
    make_diag_terms_torch,
    mult_torch,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    two_excite_batch,
)
from torch_sci_solver.torch_impl.mult import _alphabeta_pair_matrix


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
    out.sort()
    return out


def _to_int64(values):
    INT64_OFFSET = 1 << 64
    INT64_MAX_UNSIGNED = 1 << 63
    return [int(v) if int(v) < INT64_MAX_UNSIGNED else int(v) - INT64_OFFSET for v in values]


def assert_close_array(a, b, rtol=1e-12, atol=1e-13, label=""):
    """Assert two arrays agree within a combined relative/absolute tolerance."""
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.abs(a - b).max() if a.size else 0.0
    scale = max(np.abs(b).max() if b.size else 1.0, 1.0)
    if diff > atol + rtol * scale:
        raise AssertionError(
            f"{label}: max |a-b| = {diff:.3e}, scale = {scale:.3e}, rtol={rtol}, atol={atol}"
        )


def _get_pyscf_problem(mol_atom, basis="sto-3g"):
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=mol_atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    L = mol.nao
    n_a, n_b = mol.nelec
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), L)
    return mol, h1, eri, L, n_a, n_b


def _matvec_pair_test(mol_atom, label, n_trials=3):
    """Build the same problem in scalar + torch; matvec on random T;
    diff should be machine precision."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem(mol_atom)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    N = len(adets) * len(bdets)

    # Scalar reference
    hii_ref = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers_ref = list(generate_all_helpers(adets, bdets, L))

    # Torch
    integrals_t = to_torch_integrals(I0, I1, I2, device="cpu")
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, L, device="cpu"))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, L, integrals_t)

    diag_diff = np.abs(hii_t.numpy() - hii_ref).max()
    print(f"  {label}: N={N}, max |hii_torch - hii_ref| = {diag_diff:.3e}")
    assert_close_array(hii_t.numpy(), hii_ref, rtol=1e-12, label=f"{label} diag")

    rng = np.random.default_rng(seed=42)
    for trial in range(n_trials):
        T_np = rng.standard_normal(N)
        T_t = torch.tensor(T_np, dtype=torch.float64)

        Wb_ref = mult(hii_ref, T_np, adets, bdets, L, I1, I2, helpers_ref)
        Wb_t = mult_torch(hii_t, T_t, adets_t, bdets_t, L, integrals_t, helpers_t).numpy()

        diff = np.abs(Wb_t - Wb_ref).max()
        scale = max(np.abs(Wb_ref).max(), 1.0)
        print(f"    trial {trial}: max |mult_torch - mult_ref| = {diff:.3e}, scale = {scale:.3e}")
        assert_close_array(Wb_t, Wb_ref, rtol=1e-11, label=f"{label} trial {trial}")
    print(f"  {label} matvec OK")


def test_matvec_int32_tables_bit_identical():
    """int32 table payloads turn the same matvec into the same bits as int64."""
    mol, h1, eri, L, n_a, n_b = _get_pyscf_problem("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74")
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    N = len(adets) * len(bdets)

    integrals_t = to_torch_integrals(I0, I1, I2, device="cpu")
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    hii_t = make_diag_terms_torch(adets_t, bdets_t, L, integrals_t)
    helpers64 = list(generate_all_torch_helpers(adets, bdets, L, device="cpu"))
    helpers32 = list(
        generate_all_torch_helpers(adets, bdets, L, device="cpu", table_dtype=torch.int32)
    )

    rng = np.random.default_rng(seed=7)
    T_t = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
    W64 = mult_torch(hii_t, T_t, adets_t, bdets_t, L, integrals_t, helpers64)
    W32 = mult_torch(hii_t, T_t, adets_t, bdets_t, L, integrals_t, helpers32)
    assert torch.equal(W64, W32)
    assert W64.abs().max().item() > 0


def test_matvec_h2():
    """torch matvec matches the scalar reference matvec for H2."""
    _matvec_pair_test("H 0 0 0; H 0 0 0.74", "H2")


def test_matvec_h4():
    """torch matvec matches the scalar reference matvec for H4."""
    _matvec_pair_test("H 0 0 0; H 0 0 0.74; H 1.5 0 0; H 1.5 0 0.74", "H4")


def test_matvec_lih():
    """torch matvec matches the scalar reference matvec for LiH."""
    _matvec_pair_test("Li 0 0 0; H 0 0 1.6", "LiH")


def test_matvec_benzoic_acid_small():
    """First 20 HF + singles α-strings of benzoic acid (30 orbitals) → 20×20=400 basis."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    assert L == _paths.MOLECULES["benzoic_acid"].norb
    h1 = raw["H1"]
    eri_full = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri_full, raw["ECORE"])

    K = 20
    adets = _hf_and_singles(K)
    bdets = list(adets)
    print(f"  benzoic acid: K={K}, basis dim = {K * K}")

    hii_ref = make_diag_terms(adets, bdets, L, I0, I1, I2)
    helpers_ref = list(generate_all_helpers(adets, bdets, L))

    integrals_t = to_torch_integrals(I0, I1, I2, device="cpu")
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, L, device="cpu"))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, L, integrals_t)

    diag_diff = np.abs(hii_t.numpy() - hii_ref).max()
    print(f"  benzoic acid diag: max diff = {diag_diff:.3e}")
    assert_close_array(hii_t.numpy(), hii_ref, rtol=1e-11, label="benzoic acid diag")

    # HF starting vector
    N = K * K
    T_np = np.zeros(N)
    T_np[0] = 1.0
    T_t = torch.tensor(T_np, dtype=torch.float64)

    Wb_ref = mult(hii_ref, T_np, adets, bdets, L, I1, I2, helpers_ref)
    Wb_t = mult_torch(hii_t, T_t, adets_t, bdets_t, L, integrals_t, helpers_t).numpy()

    diff = np.abs(Wb_t - Wb_ref).max()
    scale = max(np.abs(Wb_ref).max(), 1.0)
    print(f"  benzoic acid mult on HF vec: max diff = {diff:.3e}, scale = {scale:.3e}")
    assert_close_array(Wb_t, Wb_ref, rtol=1e-11, label="benzoic acid HF init")

    # Random vector
    rng = np.random.default_rng(seed=1234)
    T_np = rng.standard_normal(N)
    T_t = torch.tensor(T_np, dtype=torch.float64)
    Wb_ref = mult(hii_ref, T_np, adets, bdets, L, I1, I2, helpers_ref)
    Wb_t = mult_torch(hii_t, T_t, adets_t, bdets_t, L, integrals_t, helpers_t).numpy()
    diff = np.abs(Wb_t - Wb_ref).max()
    scale = max(np.abs(Wb_ref).max(), 1.0)
    print(f"  benzoic acid mult on random vec: max diff = {diff:.3e}, scale = {scale:.3e}")
    assert_close_array(Wb_t, Wb_ref, rtol=1e-11, label="benzoic acid random vec")
    print("test_matvec_benzoic_acid_small passed")


def test_alphabeta_pair_matrix_matches_two_excite_batch():
    """αβ pair matrix elements match ``two_excite_batch`` on the interleaved bra det."""
    rng = np.random.default_rng(seed=7)
    L = 4
    n_exc = 6

    h1 = rng.standard_normal((L, L))
    h1 = 0.5 * (h1 + h1.T)
    # Assign one random value per 8-fold-symmetry orbit so the ERI is exactly
    # symmetric (summing permuted copies would round differently per element).
    raw = rng.standard_normal((L, L, L, L))
    eri = np.empty_like(raw)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    idx = min(
                        (i, j, k, m),
                        (j, i, k, m),
                        (i, j, m, k),
                        (j, i, m, k),
                        (k, m, i, j),
                        (m, k, i, j),
                        (k, m, j, i),
                        (m, k, j, i),
                    )
                    eri[i, j, k, m] = raw[idx]
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu")

    def _random_strings(n):
        occ = [rng.choice(L, size=2, replace=False) for _ in range(n)]
        return torch.tensor([sum(1 << int(i) for i in o) for o in occ], dtype=torch.int64)

    def _random_spin_orbitals(n, spin):
        # α singles live on even spin-orbitals, β singles on odd ones.
        return torch.tensor(2 * rng.integers(0, L, size=n) + spin, dtype=torch.int64)

    cr_a = _random_spin_orbitals(n_exc, 0)
    an_a = _random_spin_orbitals(n_exc, 0)
    cr_b = _random_spin_orbitals(n_exc, 1)
    an_b = _random_spin_orbitals(n_exc, 1)
    bra_a = _random_strings(n_exc)
    bra_b = _random_strings(n_exc)

    # Reference: SBD's two_excite_batch on the interleaved bra determinant
    # (i, j are annihilated and a, b created), flattened over the pair grid.
    det = det_from_alpha_beta_batch(bra_a.repeat_interleave(n_exc), bra_b.repeat(n_exc), L)
    expected = two_excite_batch(
        det,
        None,
        an_a.repeat_interleave(n_exc),
        an_b.repeat(n_exc),
        cr_a.repeat_interleave(n_exc),
        cr_b.repeat(n_exc),
        integrals.I2_eri,
    ).reshape(n_exc, n_exc)

    got = _alphabeta_pair_matrix(cr_a, an_a, cr_b, an_b, bra_a, bra_b, integrals, torch.float64)

    assert expected.abs().max() > 0.0, "reference matrix elements are all zero"
    diff = (got - expected).abs().max().item()
    print(f"  _alphabeta_pair_matrix: max |got - two_excite_batch| = {diff:.3e}")
    assert torch.equal(got, expected), f"max |got - expected| = {diff:.3e}"
    print("test_alphabeta_pair_matrix_matches_two_excite_batch passed")


if __name__ == "__main__":
    test_matvec_h2()
    test_matvec_h4()
    test_matvec_lih()
    test_matvec_benzoic_acid_small()
    test_alphabeta_pair_matrix_matches_two_excite_batch()
    print("\nALL TESTS PASSED")
