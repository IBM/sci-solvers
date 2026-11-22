"""Tests for complex-dtype support in the torch_sci_solver batched matvec + Davidson.

Verifies:
  1. setup_integrals_from_pyscf_h1_eri_vec + to_torch_integrals build a complex128
     TorchIntegrals bundle from complex Hermitian hcore/eri, unchanged real-path
     behavior.
  2. Hermiticity validation rejects non-Hermitian complex hcore/eri.
  3. make_diag_terms_torch returns a real hii for complex integrals.
  4. mult_torch(complex128) matches a brute-force dense second-quantized oracle
     built directly from the same complex Hermitian integrals.
  5. davidson_torch converges to the correct ground-state energy on the complex
     matvec, single-root and multi-root.
  6. torch.vdot vs torch.dot equivalence on real tensors (guards the "safe to
     always use vdot" assumption in davidson.py).
  7. dtype/matvec_impl contract: complex + matvec_impl="triton" is accepted for
     TPB (the Triton kernels carry a complex path); "auto" falls back to
     "batched" under complex dtype on CPU.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    _check_hermitian,
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch

# Each device-parametrized test runs twice: once on CPU, once on CUDA. The
# CUDA invocation is skipped automatically on nodes without a GPU. Mirrors
# packages/torch-sci-solver/tests/test_torch_anchors.py's DEVICE_PARAMS.
DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]


def _enumerate_dets(L: int, n_elec: int) -> list[int]:
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        m = 0
        for b in combo:
            m |= 1 << b
        out.append(m)
    out.sort()
    return out


def _random_complex_hermitian_integrals(norb: int, seed: int = 0):
    """Complex hcore + a physically-valid complex chemist-order eri.

    Same construction as tests/test_complex_hermitian.py's
    _random_complex_hermitian_integrals (kept in sync deliberately — both
    build a genuinely Hermitian complex CI operator).
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _bruteforce_dense_hamiltonian(
    hcore: np.ndarray, eri: np.ndarray, adets: list[int], bdets: list[int], norb: int
) -> np.ndarray:
    """Dense (dim, dim) CI Hamiltonian built directly from second quantization,
    independent of the package's own kernel (a true oracle, not a self-check).

    Determinants are the interleaved-Wick spin-orbital ordering the matvec
    itself uses (bit 2k = alpha spatial k, bit 2k+1 = beta spatial k), and
    ``eri`` is chemist-notation (pq|rs) with the physicist-order antisymmetrized
    two-body operator built from it in the loop below.
    """
    n_spinorb = 2 * norb
    # bits are interleaved per-orbital: build full spin-orbital dets directly.
    full_dets = []
    for a in adets:
        for b in bdets:
            d = 0
            for k in range(norb):
                if (a >> k) & 1:
                    d |= 1 << (2 * k)
                if (b >> k) & 1:
                    d |= 1 << (2 * k + 1)
            full_dets.append(d)
    dim = len(full_dets)

    def occ_list(d: int) -> list[int]:
        return [i for i in range(n_spinorb) if (d >> i) & 1]

    def h1_so(p: int, q: int) -> complex:
        # spin-orbital one-body integral: nonzero only same-spin, RHF-shared.
        if (p % 2) != (q % 2):
            return 0.0
        return hcore[p // 2, q // 2]

    def eri_so(p: int, q: int, r: int, s: int) -> complex:
        # chemist (pq|rs), zero unless spin(p)==spin(q) and spin(r)==spin(s).
        if (p % 2) != (q % 2) or (r % 2) != (s % 2):
            return 0.0
        return eri[p // 2, q // 2, r // 2, s // 2]

    H = np.zeros((dim, dim), dtype=np.complex128)
    for bra_idx, bra in enumerate(full_dets):
        bra_occ = occ_list(bra)
        for ket_idx, ket in enumerate(full_dets):
            ket_occ = set(occ_list(ket))
            # <bra| H |ket> via brute-force excitation counting (small dim only).
            diff_created = sorted(set(bra_occ) - ket_occ)
            diff_annihilated = sorted(ket_occ - set(bra_occ))
            if len(diff_created) != len(diff_annihilated):
                continue
            if len(diff_created) == 0:
                # diagonal
                e = 0.0
                for p in bra_occ:
                    e += h1_so(p, p)
                for i, p in enumerate(bra_occ):
                    for q in bra_occ[i + 1 :]:
                        e += eri_so(p, p, q, q) - eri_so(p, q, q, p)
                H[bra_idx, ket_idx] += e
            elif len(diff_created) == 1:
                (a,) = diff_created
                (i,) = diff_annihilated
                sgn = _perm_sign_single(ket, i, a)
                e = h1_so(a, i)
                for j in ket_occ:
                    if j == i:
                        continue
                    e += eri_so(a, i, j, j) - eri_so(a, j, j, i)
                H[bra_idx, ket_idx] += sgn * e
            elif len(diff_created) == 2:
                a, b = diff_created
                i, j = diff_annihilated
                sgn = _perm_sign_double(ket, i, j, a, b)
                H[bra_idx, ket_idx] += sgn * (eri_so(a, i, b, j) - eri_so(a, j, b, i))
    return H, full_dets


def _perm_sign_single(ket: int, i: int, a: int) -> float:
    """Sign of a†_a a_i acting on |ket> to give (up to sign) |bra>."""
    # count occupied spin-orbitals strictly between i and a in `ket` (annihilate
    # i first, then create a) — standard second-quantization parity count.
    lo, hi = min(i, a), max(i, a)
    count = 0
    for k in range(lo + 1, hi):
        if (ket >> k) & 1:
            count += 1
    return -1.0 if (count % 2) else 1.0


def _perm_sign_double(ket: int, i: int, j: int, a: int, b: int) -> float:
    """Sign of a†_a a†_b a_j a_i acting on |ket>, via two sequential single-sign
    applications on the intermediate determinants (small-dim brute force)."""
    # Remove i, then j from ket (in that order); create b, then a.
    sgn = 1.0
    d = ket
    for idx in (i, j):
        lo_bits = bin(d & ((1 << idx) - 1)).count("1")
        sgn *= -1.0 if (lo_bits % 2) else 1.0
        d &= ~(1 << idx)
    for idx in (b, a):
        lo_bits = bin(d & ((1 << idx) - 1)).count("1")
        sgn *= -1.0 if (lo_bits % 2) else 1.0
        d |= 1 << idx
    return sgn


@pytest.mark.parametrize("norb,n_a,n_b", [(4, 2, 1), (4, 2, 2)])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_complex_matvec_matches_bruteforce_oracle(norb, n_a, n_b, device):
    """mult_torch(complex128) matches a from-scratch second-quantized dense oracle."""
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=1)

    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals_t = to_torch_integrals(I0, I1, I2, device=device, dtype=torch.complex128)
    assert integrals_t.I1.dtype == torch.complex128
    assert integrals_t.I2_eri.dtype == torch.complex128

    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, norb, device=device))

    hii_t = make_diag_terms_torch(adets_t, bdets_t, norb, integrals_t)
    assert not hii_t.dtype.is_complex, "hii must stay real even for complex integrals"

    H_oracle, full_dets = _bruteforce_dense_hamiltonian(hcore, eri, adets, bdets, norb)
    np.testing.assert_allclose(
        H_oracle, H_oracle.conj().T, atol=1e-10, err_msg="oracle H must be Hermitian"
    )
    np.testing.assert_allclose(
        np.diagonal(H_oracle).imag, 0.0, atol=1e-10, err_msg="oracle diagonal must be real"
    )
    np.testing.assert_allclose(hii_t.cpu().numpy(), np.diagonal(H_oracle).real, atol=1e-9)

    rng = np.random.default_rng(seed=7)
    dim = len(full_dets)
    for _ in range(3):
        T_np = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        T_t = torch.tensor(T_np, dtype=torch.complex128, device=device)
        Wb_t = mult_torch(hii_t, T_t, adets_t, bdets_t, norb, integrals_t, helpers_t).cpu().numpy()
        Wb_oracle = H_oracle @ T_np
        np.testing.assert_allclose(Wb_t, Wb_oracle, atol=1e-8)


def test_complex_integrals_reject_non_hermitian_hcore():
    """A non-Hermitian complex hcore raises ValueError, not a silent wrong operator."""
    norb = 3
    rng = np.random.default_rng(2)
    hcore = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    _, eri = _random_complex_hermitian_integrals(norb, seed=2)
    with pytest.raises(ValueError, match="Hermitian"):
        setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)


def test_complex_integrals_reject_non_hermitian_eri():
    """A non-Hermitian complex eri raises ValueError."""
    norb = 3
    hcore, _ = _random_complex_hermitian_integrals(norb, seed=3)
    rng = np.random.default_rng(4)
    eri = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    with pytest.raises(ValueError, match="Hermitian"):
        setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)


def test_check_hermitian_accepts_valid_complex_integrals():
    """The validator does not raise on a genuinely Hermitian complex set."""
    norb = 4
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=5)
    _check_hermitian(hcore, eri)  # must not raise


@pytest.mark.parametrize("n_roots", [1, 2])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_davidson_converges_on_complex_matvec(n_roots, device):
    """davidson_torch on a complex matvec matches the oracle's lowest eigenvalue(s)."""
    norb, n_a, n_b = 4, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=6)
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
    ref = np.sort(np.linalg.eigvalsh(H_oracle))

    dim = len(full_dets)
    W = basis_init_vector_torch(
        len(adets), len(bdets), init=0, dtype=torch.complex128, device=device
    )
    assert W.dtype == torch.complex128

    result = davidson_torch(
        hii_t,
        W,
        matvec,
        max_davidson_iters=50,
        davidson_block_size=min(10, dim),
        eps=1e-9,
        n_roots=n_roots,
    )
    if n_roots == 1:
        energy, _vec, info = result
        np.testing.assert_allclose(energy, ref[0], atol=1e-6)
    else:
        energies, _vecs, info = result
        np.testing.assert_allclose(np.sort(energies.numpy()), ref[:n_roots], atol=1e-6)


def test_vdot_matches_dot_for_real_tensors():
    """torch.vdot is bit-identical to torch.dot on real tensors — the assumption
    the davidson.py torch.dot -> torch.vdot swap relies on to keep the real path
    unaffected."""
    rng = torch.Generator().manual_seed(0)
    a = torch.randn(37, dtype=torch.float64, generator=rng)
    b = torch.randn(37, dtype=torch.float64, generator=rng)
    assert torch.equal(torch.dot(a, b), torch.vdot(a, b))


def test_matvec_impl_triton_accepts_complex_dtype():
    """matvec_impl='triton' + a complex dtype is accepted for TPB.

    The Triton kernels carry an interleaved-real complex path; the GPU tests in
    ``test_complex_triton.py`` compare it with the batched matvec."""
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    solver = TorchSCISolver(device="cpu", dtype=torch.complex128, matvec_impl="triton")
    assert solver.matvec_impl == "triton"


def test_matvec_impl_auto_falls_back_to_batched_for_complex_dtype():
    """matvec_impl='auto' + complex dtype resolves to 'batched', not 'triton'."""
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    solver = TorchSCISolver(device="cpu", dtype=torch.complex128, matvec_impl="auto")
    assert solver.matvec_impl == "batched"


def test_solver_rejects_complex_integrals_with_real_dtype():
    """Complex hcore/eri passed to a real-dtype-constructed solver raises
    TypeError rather than silently dropping the imaginary part."""
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    norb, n_a, n_b = 3, 1, 1
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=8)
    adets = _enumerate_dets(norb, n_a)
    bdets = _enumerate_dets(norb, n_b)

    solver = TorchSCISolver(device="cpu", dtype=torch.float64)
    with pytest.raises(TypeError, match="complex"):
        solver(
            ci_strings=[(np.asarray(adets, dtype=np.int64), np.asarray(bdets, dtype=np.int64))],
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=norb,
            nelec=(n_a, n_b),
        )
