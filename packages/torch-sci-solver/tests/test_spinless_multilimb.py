"""Tests for the flat spin-orbital ("spinless") basis beyond 64 qubits (n_limbs > 1).

``SpinlessBasis.dets`` is a stacked ``(N, n_limbs)`` int64 tensor, so
``n_spinorb`` is not capped at 64. These tests exercise that path directly,
since every other spinless test file stays at ``n_spinorb <= 64`` (where
``n_limbs == 1``) and would not catch a limb-splitting regression.

``pyscf.fci.cistring.make_strings`` returns plain int64 bitmasks only below
its own internal packing limit; past it, rows become ``OIndexList`` arrays
of occupied orbital indices. So canonical determinant lists here are built
directly via :func:`itertools.combinations` rather than ``cistring``, and
cross-checked against it where both agree (``nmo <= 64``).

Cross-validates against ``sci_solvers.spinless_ci_problem`` (numpy/pyscf,
independent of the torch kernels under test and already unbounded in
``n_spinorb`` via Python arbitrary-precision ints) — the same oracle
``test_spinless_matvec.py``/``test_spinless_rdm.py`` use at ``n_spinorb <= 64``.

``SpinlessBasis.dets`` sorts lexicographically by limb (limb 0 — the lowest
64 bits — compares first; see :mod:`torch_sci_solver.torch_impl.
spinless_helpers`'s ``SpinlessBasis`` docstring), not by numeric value once
``n_limbs > 1``. So unlike the ``n_spinorb <= 64`` tests' ``np.argsort(strs.
astype(np.int64))`` reindexing trick, matching a basis row to its source
``strs`` entry here goes through an explicit value lookup
(:func:`_basis_order_for`).
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
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.rdm_spinless import spinless_rdm12_torch
from torch_sci_solver.torch_impl.spinless_helpers import (
    dets_to_python_ints,
    generate_spinless_helpers,
    spinless_basis_from_array,
)

from sci_solvers.spinless_ci_problem import build_spinless_complex_ci_problem, spinless_rdm12

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]

# (n_spinorb, n_particles, seed): widths crossing from 1 to 2 limbs (64 -> 65)
# and landing well inside the 2-limb regime (70, not a multiple of 64 -- the
# top-limb boundary case, see the module docstring). n_particles stays small
# so C(n_spinorb, n_particles) remains dense-diagonalizable -- at these widths
# n_particles=3 already means dim > 50k (tens of GB dense), so only 1 and 2
# are used.
SYSTEMS = [(65, 2, 20), (70, 2, 21), (70, 1, 22)]


def canonical_strs(n_spinorb: int, n_particles: int) -> list[int]:
    """Sorted canonical ``(n_spinorb, n_particles)`` determinant bitmasks.

    Built directly via :func:`itertools.combinations` instead of
    ``pyscf.fci.cistring.make_strings`` (see module docstring).
    """
    return sorted(
        sum(1 << b for b in combo)
        for combo in itertools.combinations(range(n_spinorb), n_particles)
    )


def _random_complex_hermitian_spinorbital_integrals(nmo: int, seed: int):
    """Same construction as ``test_spinless_matvec.py``'s fixture of the same name."""
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


def _basis_order_for(basis, strs: list[int]) -> list[int]:
    """``order[i]`` = ``basis``-row index of ``strs[i]``, for reindexing amplitudes.

    ``basis.dets`` is sorted lexicographically by limb, not numerically, so
    this is a value lookup rather than ``np.argsort`` on the raw ints (see
    module docstring).
    """
    det_ints = dets_to_python_ints(basis.dets).tolist()
    pos = {v: i for i, v in enumerate(det_ints)}
    return [pos[s] for s in strs]


@pytest.mark.parametrize("n_spinorb,n,seed", SYSTEMS)
def test_spinless_basis_crosses_limb_boundary(n_spinorb: int, n: int, seed: int) -> None:
    """spinless_basis_from_array no longer rejects n_spinorb > 64, and round-trips."""
    del seed  # not needed: this test has no randomness, just parametrized with the others
    strs = canonical_strs(n_spinorb, n)
    basis = spinless_basis_from_array(strs, n_spinorb, n, device="cpu")
    assert basis.n_limbs == (n_spinorb + 63) // 64
    assert basis.n_limbs > 1
    recovered = sorted(int(v) for v in dets_to_python_ints(basis.dets))
    assert recovered == strs


@pytest.mark.slow
@pytest.mark.parametrize("n_spinorb,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_mult_spinless_torch_matches_oracle_beyond_64_qubits(
    n_spinorb: int, n: int, seed: int, device: str
) -> None:
    """mult_spinless_torch matches the independent numpy/pyscf oracle at n_spinorb > 64."""
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(n_spinorb, seed)
    strs = canonical_strs(n_spinorb, n)
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    H_dense = problem._dense

    basis = spinless_basis_from_array(strs, n_spinorb, n, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    order = _basis_order_for(basis, strs)  # order[i] = basis row of strs[i]

    rng = np.random.default_rng(seed + 1)
    dim = basis.dim
    T_np = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    T_basis = torch.zeros(dim, dtype=torch.complex128, device=device)
    for oracle_idx, basis_idx in enumerate(order):
        T_basis[basis_idx] = complex(T_np[oracle_idx])

    Wb = mult_spinless_torch(hii, T_basis, basis, helpers, integrals)

    expected = H_dense @ T_np
    got = np.array([complex(Wb[basis_idx]) for basis_idx in order])
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("n_spinorb,n,seed", SYSTEMS)
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_rdm12_matches_oracle_beyond_64_qubits(
    n_spinorb: int, n: int, seed: int, device: str
) -> None:
    """spinless_rdm12_torch matches the independent numpy/pyscf oracle at n_spinorb > 64."""
    basis = spinless_basis_from_array(canonical_strs(n_spinorb, n), n_spinorb, n, device=device)
    strs = canonical_strs(n_spinorb, n)
    helpers = generate_spinless_helpers(basis)
    order = _basis_order_for(basis, strs)

    rng = np.random.default_rng(seed + 2)
    dim = basis.dim
    amp_np = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    amp_np /= np.linalg.norm(amp_np)
    amp_basis = torch.zeros(dim, dtype=torch.complex128, device=device)
    for oracle_idx, basis_idx in enumerate(order):
        amp_basis[basis_idx] = complex(amp_np[oracle_idx])

    dm1_t, dm2_t = spinless_rdm12_torch(basis, helpers, amp_basis)
    dm1_ref, dm2_ref = spinless_rdm12(amp_np, n_spinorb, n, strs=strs)

    np.testing.assert_allclose(dm1_t.cpu().numpy(), dm1_ref, atol=1e-8)
    np.testing.assert_allclose(dm2_t.cpu().numpy(), dm2_ref, atol=1e-8)
    np.testing.assert_allclose(torch.trace(dm1_t).cpu().numpy(), n, atol=1e-8)
