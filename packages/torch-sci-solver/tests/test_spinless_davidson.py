"""Tests for ``davidson_torch`` on the flat spin-orbital ("spinless") matvec.

``davidson_torch`` itself is unchanged for this path -- it only ever sees a
bare ``hii``/``matvec`` closure of the flat basis dimension, exactly as it
does for GDB pair mode. These tests verify the *combination* (spinless
matvec + unmodified Davidson) reproduces the dense-oracle ground-state
energy, single-root and multi-root.
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
from torch_sci_solver.torch_impl.davidson import davidson_torch
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
    basis_init_vector_spinless_torch,
    generate_spinless_helpers,
    spinless_basis_from_array,
)

from sci_solvers.spinless_ci_problem import build_spinless_complex_ci_problem

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]

# (nmo, n_particles, seed) small systems -- kept in sync deliberately with
# test_spinless_matvec.py's SYSTEMS (test modules here are not a package, so
# cross-file imports don't work; duplication matches this test suite's own
# convention, e.g. test_complex_matvec.py's _random_complex_hermitian_integrals).
SYSTEMS = [(3, 1, 10), (4, 2, 11), (5, 2, 12), (6, 3, 13), (6, 2, 14), (7, 2, 15), (5, 4, 16)]


def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int):
    """Complex hcore + a physically valid complex chemist-order eri, flat spin-orbital."""
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


def _dense_ground_energies(hcore, eri, n, strs, k: int = 1):
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    H = problem._dense
    w = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    return w[:k]


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_davidson_single_root_matches_dense_ground_energy(
    nmo: int, n: int, seed: int, device: str
) -> None:
    """Single-root davidson_torch on the spinless matvec matches the dense ground energy."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    (e_ref,) = _dense_ground_energies(hcore, eri, n, strs, k=1)

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
    assert info["converged"]
    assert abs(energy - e_ref) < 1e-6
    # Eigenvector normalized and a genuine eigenvector of the dense oracle.
    H_dense = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)._dense
    order = np.argsort(strs.astype(np.int64))
    Hv = H_dense[np.ix_(order, order)] @ eigvec.cpu().numpy()
    np.testing.assert_allclose(Hv, energy * eigvec.cpu().numpy(), atol=1e-6)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_davidson_multi_root_matches_dense_low_lying_spectrum(device: str) -> None:
    """Multi-root davidson_torch on the spinless matvec matches the dense low-lying spectrum."""
    nmo, n, seed = 6, 3, 30
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    n_roots = 3
    e_ref = _dense_ground_energies(hcore, eri, n, strs, k=n_roots)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    W = basis_init_vector_spinless_torch(basis.dim, init=0, device=device, dtype=torch.complex128)

    def matvec(v):
        return mult_spinless_torch(hii, v, basis, helpers, integrals)

    energies, eigvecs, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=60, davidson_block_size=25, eps=1e-9, n_roots=n_roots
    )
    energies_sorted = np.sort(energies.cpu().numpy())
    np.testing.assert_allclose(energies_sorted, e_ref, atol=1e-5)
