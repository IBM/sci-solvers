"""Tests for the flat spin-orbital ("spinless") torch 1-/2-RDM builder.

Cross-validates against ``sci_solvers.spinless_ci_problem.spinless_rdm12``
(numpy/pyscf, independent of the torch kernels under test) on the ground
state of a small dense-diagonalized system.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
import torch
from pyscf.fci import cistring
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.rdm_spinless import (
    spinless_energy_torch,
    spinless_rdm1_diag_torch,
    spinless_rdm12_torch,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
    generate_spinless_helpers,
    spinless_basis_from_array,
)

from sci_solvers.spinless_ci_problem import (
    build_spinless_complex_ci_problem,
    spinless_energy,
    spinless_rdm12,
)

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]

SYSTEMS = [(3, 1, 10), (4, 2, 11), (5, 2, 12), (6, 3, 13), (6, 2, 14), (7, 2, 15), (5, 4, 16)]


def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int):
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    naux = nmo + 2
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _dense_ground_state(hcore, eri, n, strs):
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    H = problem._dense
    w, v = np.linalg.eigh(0.5 * (H + H.conj().T))
    return w[0], v[:, 0]


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_rdm12_matches_numpy_oracle(nmo: int, n: int, seed: int, device: str) -> None:
    """spinless_rdm12_torch matches sci_solvers.spinless_ci_problem.spinless_rdm12."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    e0, C_ref = _dense_ground_state(hcore, eri, n, strs)

    dm1_ref, dm2_ref = spinless_rdm12(C_ref, nmo, n)
    assert np.max(np.abs(dm1_ref - dm1_ref.conj().T)) < 1e-10, "oracle dm1 should be Hermitian"

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    order = np.argsort(strs.astype(np.int64))
    C_sorted = C_ref[order]
    C_t = torch.tensor(C_sorted, dtype=torch.complex128, device=device)

    dm1_t, dm2_t = spinless_rdm12_torch(basis, helpers, C_t)
    np.testing.assert_allclose(dm1_t.cpu().numpy(), dm1_ref, atol=1e-9)
    np.testing.assert_allclose(dm2_t.cpu().numpy(), dm2_ref, atol=1e-9)

    # trace(dm1) == N (total particle number)
    np.testing.assert_allclose(torch.trace(dm1_t).cpu().numpy(), n, atol=1e-9)


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_rdm12_matches_numpy_oracle_on_truncated_subspace(
    nmo: int, n: int, seed: int, device: str
) -> None:
    """spinless_rdm12_torch matches the oracle on a truncated (selected-CI) subset.

    Regression test: the previous Gram sum restricted its
    intermediate axis to the selected determinants, which is exact on the
    full canonical list (what :func:`test_spinless_rdm12_matches_numpy_oracle`
    builds) but wrong on an arbitrary proper subset, where the selected set
    is not closed under single excitations.
    """
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    full_strs = np.asarray(cistring.make_strings(range(nmo), n))
    rng = np.random.default_rng(seed + 1000)
    keep = max(2, len(full_strs) // 2)
    strs = full_strs[rng.choice(len(full_strs), size=keep, replace=False)]
    e0, C_ref = _dense_ground_state(hcore, eri, n, strs)

    dm1_ref, dm2_ref = spinless_rdm12(C_ref, nmo, n, strs=strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    order = np.argsort(strs.astype(np.int64))
    C_sorted = C_ref[order]
    C_t = torch.tensor(C_sorted, dtype=torch.complex128, device=device)

    dm1_t, dm2_t = spinless_rdm12_torch(basis, helpers, C_t)
    np.testing.assert_allclose(dm1_t.cpu().numpy(), dm1_ref, atol=1e-9)
    np.testing.assert_allclose(dm2_t.cpu().numpy(), dm2_ref, atol=1e-9)

    # partial-trace identity: sum_r dm2[p,q,r,r] = (N-1) dm1[p,q]
    partial_trace = torch.einsum("pqrr->pq", dm2_t)
    np.testing.assert_allclose(
        partial_trace.cpu().numpy(), (n - 1) * dm1_t.cpu().numpy(), atol=1e-9
    )


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_rdm1_diag_matches_full_rdm_diagonal(
    nmo: int, n: int, seed: int, device: str
) -> None:
    """spinless_rdm1_diag_torch (cheap, diagonal-only) matches
    diag(spinless_rdm12_torch(...)[0]) (the full dm1's diagonal).

    Not bit-identical: the full path reaches the diagonal via complex
    ``conj(C) * (occ * C)`` inside an einsum reduction, while the diagonal-only
    path computes real ``occ.T @ |C|**2`` directly via matmul -- an
    algebraically equivalent but differently-ordered set of floating-point
    operations, so the two differ at the ~1e-16 (float64 machine epsilon)
    level, not exactly zero.
    """
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    _e0, C_ref = _dense_ground_state(hcore, eri, n, strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    order = np.argsort(strs.astype(np.int64))
    C_t = torch.tensor(C_ref[order], dtype=torch.complex128, device=device)

    diag_direct = spinless_rdm1_diag_torch(basis, C_t)
    dm1_t, _dm2_t = spinless_rdm12_torch(basis, helpers, C_t)
    diag_from_full = torch.real(torch.diagonal(dm1_t))

    np.testing.assert_allclose(
        diag_direct.cpu().numpy(), diag_from_full.cpu().numpy(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_energy_from_rdm_matches_dense_ground_energy(
    nmo: int, n: int, seed: int, device: str
) -> None:
    """spinless_energy_torch reconstructs the dense oracle's ground energy from the RDMs."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    e0, C_ref = _dense_ground_state(hcore, eri, n, strs)

    e_from_rdm_ref = spinless_energy(C_ref, hcore, eri, n)
    assert abs(e_from_rdm_ref - e0) < 1e-8

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    order = np.argsort(strs.astype(np.int64))
    C_t = torch.tensor(C_ref[order], dtype=torch.complex128, device=device)
    hcore_t = torch.tensor(hcore, dtype=torch.complex128, device=device)
    eri_t = torch.tensor(eri, dtype=torch.complex128, device=device)

    e_from_rdm_t = spinless_energy_torch(basis, helpers, C_t, hcore_t, eri_t)
    assert abs(e_from_rdm_t - e0) < 1e-8


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_rdm12_energy_cross_check_matches_davidson(device: str) -> None:
    """RDM-derived energy on the Davidson eigenvector matches the Davidson eigenvalue.

    Mirrors ``rdm.py``'s own "energy_from_rdm cross-validation" convention
    for the product-mode path.
    """
    from torch_sci_solver.torch_impl.davidson import davidson_torch
    from torch_sci_solver.torch_impl.mult_spinless import (
        make_diag_terms_spinless_torch,
        mult_spinless_torch,
    )
    from torch_sci_solver.torch_impl.spinless_helpers import basis_init_vector_spinless_torch

    nmo, n, seed = 5, 2, 40
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    W = basis_init_vector_spinless_torch(basis.dim, init=0, device=device, dtype=torch.complex128)

    def matvec(v):
        return mult_spinless_torch(hii, v, basis, helpers, integrals)

    energy, eigvec, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=40, davidson_block_size=25, eps=1e-10
    )

    hcore_t = torch.tensor(hcore, dtype=torch.complex128, device=device)
    eri_t = torch.tensor(eri, dtype=torch.complex128, device=device)
    e_from_rdm = spinless_energy_torch(basis, helpers, eigvec, hcore_t, eri_t)
    assert abs(e_from_rdm - float(energy)) < 1e-8
