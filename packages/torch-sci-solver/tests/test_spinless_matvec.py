"""Tests for the flat spin-orbital ("spinless") torch matvec.

Every check cross-validates against
``sci_solvers.spinless_ci_problem.build_spinless_complex_ci_problem`` — a
numpy/pyscf implementation, independent of the torch kernels under test,
that is itself validated to ~1e-14 against a brute-force oracle (see its own
module docstring). That module is treated as ground truth here, mirroring
``test_complex_matvec.py``'s own brute-force-oracle convention.
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
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.spinless_helpers import (
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

# (nmo, n_particles, seed) small systems swept across every test in this file.
SYSTEMS = [(3, 1, 10), (4, 2, 11), (5, 2, 12), (6, 3, 13), (6, 2, 14), (7, 2, 15), (5, 4, 16)]


def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int):
    """Complex hcore + a physically valid complex chemist-order eri, flat spin-orbital.

    Factor construction: ``(pq|rs) = sum_P L[P]
    conj(L[P])`` with a Hermitian factor ``L == L^H``, symmetrized over
    ``(pq)<->(rs)`` — carries both Hermitian symmetries a spinless CI
    operator needs.
    """
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


def _dense_oracle(hcore, eri, n, strs):
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    return problem._dense


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_to_torch_integrals_spinless_matches_flat_arrays(device: str) -> None:
    """No spin-doubling, no quadrant masking: I1/I2_eri are the raw flat arrays."""
    nmo = 4
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=1)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    assert integrals.n_spinorb == nmo
    assert integrals.dtype == torch.complex128
    np.testing.assert_allclose(integrals.I1.cpu().numpy(), hcore)
    np.testing.assert_allclose(integrals.I2_eri.cpu().numpy(), eri)
    p = np.arange(nmo)
    np.testing.assert_allclose(integrals.I2_direct.cpu().numpy(), eri[p[:, None], p[:, None], p, p])
    np.testing.assert_allclose(
        integrals.I2_exchange.cpu().numpy(), eri[p[:, None], p, p, p[:, None]]
    )


def test_to_torch_integrals_spinless_rejects_non_hermitian_hcore() -> None:
    """A non-Hermitian hcore is rejected rather than building a wrong operator."""
    nmo = 3
    hcore = np.zeros((nmo, nmo), dtype=np.complex128)
    hcore[0, 1] = 1.0 + 2.0j  # hcore[1, 0] left at 0 -- not Hermitian
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    with pytest.raises(ValueError, match="Hermitian"):
        to_torch_integrals_spinless(hcore, eri)


def test_to_torch_integrals_spinless_rejects_non_hermitian_eri() -> None:
    """An eri breaking (pq|rs) == conj((qp|sr)) is rejected rather than building a wrong operator."""
    nmo = 3
    hcore, _ = _random_complex_hermitian_spinorbital_integrals(nmo, seed=2)
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    eri[0, 1, 2, 0] = 1.0 + 1.0j  # breaks (pq|rs) == conj((qp|sr))
    with pytest.raises(ValueError, match="eri does not satisfy"):
        to_torch_integrals_spinless(hcore, eri)


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_diag_terms_spinless_matches_dense_diagonal(
    nmo: int, n: int, seed: int, device: str
) -> None:
    """make_diag_terms_spinless_torch matches the dense oracle's diagonal."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    H_dense = _dense_oracle(hcore, eri, n, strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    # basis.dets is sorted; reindex the dense diagonal to match.
    order = np.argsort(strs.astype(np.int64))
    expected = np.real(np.diagonal(H_dense))[order]
    np.testing.assert_allclose(hii.cpu().numpy(), expected, atol=1e-10)


@pytest.mark.parametrize("nmo,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_mult_spinless_torch_matches_dense_oracle(nmo: int, n: int, seed: int, device: str) -> None:
    """H @ v via the batched torch kernel matches H_dense @ v exactly (to ~1e-10)."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    H_dense = _dense_oracle(hcore, eri, n, strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    order = np.argsort(strs.astype(np.int64))
    rng = np.random.default_rng(seed + 1000)
    v = rng.standard_normal(len(strs)) + 1j * rng.standard_normal(len(strs))
    v_sorted = v[order]
    v_t = torch.tensor(v_sorted, dtype=torch.complex128, device=device)

    Hv_torch = mult_spinless_torch(hii, v_t, basis, helpers, integrals).cpu().numpy()
    Hv_dense = (H_dense @ v)[order]
    np.testing.assert_allclose(Hv_torch, Hv_dense, atol=1e-9)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_mult_spinless_torch_matches_dense_oracle_block(device: str) -> None:
    """A (N, k) block input matvecs each column independently, matching the dense oracle."""
    nmo, n, seed = 5, 2, 20
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    H_dense = _dense_oracle(hcore, eri, n, strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    order = np.argsort(strs.astype(np.int64))
    rng = np.random.default_rng(seed + 1000)
    dim = len(strs)
    V = rng.standard_normal((dim, 3)) + 1j * rng.standard_normal((dim, 3))
    V_sorted = V[order]
    V_t = torch.tensor(V_sorted, dtype=torch.complex128, device=device)

    HV_torch = mult_spinless_torch(hii, V_t, basis, helpers, integrals).cpu().numpy()
    HV_dense = (H_dense @ V)[order]
    np.testing.assert_allclose(HV_torch, HV_dense, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_mult_spinless_torch_matches_oracle_at_n_spinorb_64(device: str) -> None:
    """``n_spinorb == 64`` is the single-limb boundary: ``string_mask``'s per-limb
    complement is the full 64-bit ``-1`` mask rather than a ``(1 << 64) - 1``
    Python-int expression (which would be outside int64 range). No other test
    in this file reaches this width, so this pins the boundary explicitly.

    Built via :func:`itertools.combinations` rather than
    ``cistring.make_strings``: at ``nmo == 64``, pyscf already switches to its
    ``OIndexList`` (orbital-index) row format rather than returning plain
    int64 bitmasks (see ``test_spinless_multilimb.py``'s module docstring for
    the same issue one bit further up).
    """
    from itertools import combinations

    from torch_sci_solver.torch_impl.pairs import _to_signed_int64

    nmo, n, seed = 64, 2, 30
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs_py = sorted(sum(1 << b for b in combo) for combo in combinations(range(nmo), n))
    strs = np.asarray([_to_signed_int64(s) for s in strs_py], dtype=np.int64)
    H_dense = _dense_oracle(hcore, eri, n, strs)

    basis = spinless_basis_from_array(strs, nmo, n, device=device)
    assert basis.n_limbs == 1
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    order = np.argsort(strs)
    rng = np.random.default_rng(seed + 1000)
    dim = len(strs)
    T = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    T_sorted = T[order]
    T_t = torch.tensor(T_sorted, dtype=torch.complex128, device=device)

    HT_torch = mult_spinless_torch(hii, T_t, basis, helpers, integrals).cpu().numpy()
    HT_dense = (H_dense @ T)[order]
    np.testing.assert_allclose(HT_torch, HT_dense, atol=1e-9)
