"""Tests for complex-dtype support in torch_sci_solver.torch_impl.rdm.

Cross-validates the batched-kernel 1-/2-RDM construction against a brute-force
second-quantized RDM built directly from the dense oracle eigenvector (see
test_complex_matvec.py's _bruteforce_dense_hamiltonian), for a complex
eigenvector. Also checks energy_from_rdm reproduces the Davidson eigenvalue
and that occupation numbers / RDM traces come out real.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
import torch
from test_complex_matvec import (  # noqa: E402  (local test-module import)
    DEVICE_PARAMS,
    _bruteforce_dense_hamiltonian,
    _enumerate_dets,
    _random_complex_hermitian_integrals,
)
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch
from torch_sci_solver.torch_impl.rdm import (
    energy_from_rdm,
    make_one_rdm_diag_torch,
    make_one_rdm_torch,
    make_two_rdm_torch,
    s_squared_from_rdm,
)


def _bruteforce_rdms(psi_full: np.ndarray, full_dets: list, norb: int):
    """1-/2-RDM built directly from ⟨I|a†_p a_q|J⟩ matrix elements on the same
    interleaved-Wick full-determinant basis mult_torch uses — an independent
    check, not a reuse of any rdm.py machinery."""
    n_spinorb = 2 * norb

    def sgn_create_annihilate(d, i, a):
        # sign of a_i acting first, then a†_a, on determinant d (must have i
        # occupied and a unoccupied in d for a nonzero result).
        lo_bits = bin(d & ((1 << i) - 1)).count("1")
        s = -1.0 if (lo_bits % 2) else 1.0
        d2 = d & ~(1 << i)
        lo_bits2 = bin(d2 & ((1 << a) - 1)).count("1")
        s *= -1.0 if (lo_bits2 % 2) else 1.0
        return s

    gamma_so = np.zeros((n_spinorb, n_spinorb), dtype=np.complex128)
    for bra_idx, bra in enumerate(full_dets):
        for ket_idx, ket in enumerate(full_dets):
            c = psi_full[bra_idx].conj() * psi_full[ket_idx]
            if abs(c) < 1e-14:
                continue
            for p in range(n_spinorb):
                for q in range(n_spinorb):
                    if p == q:
                        if (ket >> q) & 1:
                            if bra == ket:
                                gamma_so[p, q] += c
                        continue
                    if not ((ket >> q) & 1) or ((ket >> p) & 1):
                        continue
                    d = (ket & ~(1 << q)) | (1 << p)
                    if d != bra:
                        continue
                    sgn = sgn_create_annihilate(ket, q, p)
                    gamma_so[p, q] += sgn * c

    # spin-block spatial gamma_a/gamma_b (sum out spin, alpha=even, beta=odd)
    gamma_a = gamma_so[0::2, 0::2]
    gamma_b = gamma_so[1::2, 1::2]
    return gamma_a, gamma_b


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_complex_one_rdm_matches_bruteforce(device):
    """1-RDM (diag + full) from the batched kernel matches a from-scratch
    second-quantized RDM built from the same complex eigenvector."""
    norb, n_a, n_b = 4, 2, 1
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=11)
    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals_t = to_torch_integrals(I0, I1, I2, device=device, dtype=torch.complex128)
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, norb, device=device))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, norb, integrals_t)

    def matvec(v):
        return mult_torch(hii_t, v, adets_t, bdets_t, norb, integrals_t, helpers_t)

    H_oracle, full_dets = _bruteforce_dense_hamiltonian(hcore, eri, adets, bdets, norb)
    w, vecs = np.linalg.eigh(H_oracle)
    psi_full = vecs[:, 0]

    dim = len(full_dets)
    W = basis_init_vector_torch(
        len(adets), len(bdets), init=0, dtype=torch.complex128, device=device
    )
    energy, eigvec, info = davidson_torch(
        hii_t, W, matvec, max_davidson_iters=50, davidson_block_size=min(10, dim), eps=1e-10
    )
    np.testing.assert_allclose(energy, w[0], atol=1e-6)

    # align phase/sign between the Davidson eigvec and the oracle eigenvector
    # (both are only defined up to a global phase for a genuinely complex H).
    eigvec_cpu = eigvec.cpu().numpy()
    overlap = np.vdot(psi_full, eigvec_cpu)
    phase = overlap / abs(overlap)
    eigvec_aligned = eigvec_cpu / phase

    occ_a, occ_b = make_one_rdm_diag_torch(
        torch.tensor(eigvec_aligned, dtype=torch.complex128, device=device), adets_t, bdets_t, norb
    )
    assert not occ_a.dtype.is_complex
    assert not occ_b.dtype.is_complex

    gamma_a_ref, gamma_b_ref = _bruteforce_rdms(psi_full, full_dets, norb)
    # occ diag reproduces the brute-force gamma diagonal.
    np.testing.assert_allclose(occ_a.cpu().numpy(), np.diagonal(gamma_a_ref).real, atol=1e-6)
    np.testing.assert_allclose(occ_b.cpu().numpy(), np.diagonal(gamma_b_ref).real, atol=1e-6)

    gamma_a, gamma_b = make_one_rdm_torch(
        torch.tensor(eigvec_aligned, dtype=torch.complex128, device=device),
        adets_t,
        bdets_t,
        norb,
        helpers_t,
        eigvec_in_interleaved_wick=True,
    )
    # gamma must be Hermitian.
    gamma_a_np = gamma_a.cpu().numpy()
    gamma_b_np = gamma_b.cpu().numpy()
    np.testing.assert_allclose(gamma_a_np, gamma_a_np.conj().T, atol=1e-6)
    np.testing.assert_allclose(gamma_b_np, gamma_b_np.conj().T, atol=1e-6)
    np.testing.assert_allclose(gamma_a_np, gamma_a_ref, atol=1e-6)
    np.testing.assert_allclose(gamma_b_np, gamma_b_ref, atol=1e-6)
    # electron-count conservation.
    np.testing.assert_allclose(gamma_a_np.trace().real, n_a, atol=1e-6)
    np.testing.assert_allclose(gamma_b_np.trace().real, n_b, atol=1e-6)


def test_complex_energy_from_rdm_matches_davidson_eigenvalue():
    """energy_from_rdm on the complex 1-/2-RDMs reproduces the Davidson ground
    energy, and returns a real (not complex) scalar."""
    norb, n_a, n_b = 4, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=12)
    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals_t = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
    adets_t = torch.tensor(adets, dtype=torch.int64)
    bdets_t = torch.tensor(bdets, dtype=torch.int64)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, norb, device="cpu"))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, norb, integrals_t)

    def matvec(v):
        return mult_torch(hii_t, v, adets_t, bdets_t, norb, integrals_t, helpers_t)

    dim = len(adets) * len(bdets)
    W = basis_init_vector_torch(len(adets), len(bdets), init=0, dtype=torch.complex128)
    energy, eigvec, info = davidson_torch(
        hii_t, W, matvec, max_davidson_iters=50, davidson_block_size=min(10, dim), eps=1e-10
    )

    Gaa, Gab, Gbb = make_two_rdm_torch(
        eigvec, adets_t, bdets_t, norb, helpers_t, eigvec_in_interleaved_wick=True
    )
    gamma_a, gamma_b = make_one_rdm_torch(
        eigvec, adets_t, bdets_t, norb, helpers_t, eigvec_in_interleaved_wick=True
    )
    e_from_rdm = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, integrals_t)
    assert not e_from_rdm.dtype.is_complex, "energy_from_rdm must return a real scalar"
    np.testing.assert_allclose(float(e_from_rdm.item()), energy, atol=1e-6)


def test_complex_s_squared_from_rdm_is_real():
    """s_squared_from_rdm returns a real (not complex) scalar for complex Gab."""
    norb, n_a, n_b = 4, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=13)
    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals_t = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
    adets_t = torch.tensor(adets, dtype=torch.int64)
    bdets_t = torch.tensor(bdets, dtype=torch.int64)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, norb, device="cpu"))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, norb, integrals_t)

    def matvec(v):
        return mult_torch(hii_t, v, adets_t, bdets_t, norb, integrals_t, helpers_t)

    dim = len(adets) * len(bdets)
    W = basis_init_vector_torch(len(adets), len(bdets), init=0, dtype=torch.complex128)
    _energy, eigvec, _info = davidson_torch(
        hii_t, W, matvec, max_davidson_iters=50, davidson_block_size=min(10, dim), eps=1e-10
    )

    from torch_sci_solver.torch_impl.rdm import make_two_rdm_ab_torch

    Gab = make_two_rdm_ab_torch(
        eigvec, adets_t, bdets_t, norb, helpers_t, eigvec_in_interleaved_wick=True
    )
    s_sq = s_squared_from_rdm(Gab, n_a, n_b)
    assert not s_sq.dtype.is_complex
