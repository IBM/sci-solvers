"""Regression tests for the spinless Davidson seed on multi-limb bases.

Once ``nmo > 64`` determinants occupy more than one uint64 limb,
``spinless_basis_from_array``'s ``np.unique(..., axis=0)`` sorts rows
lexicographically by the LOW limb first, which is not numeric order — so
basis index 0 is an arbitrary determinant, not the numerically-smallest
(HF) one. The spinless ``__call__`` used to seed Davidson at index 0
regardless, and Davidson happily converged to that determinant's isolated
eigenpair and returned it as the ground state (production symptom: SQD
iteration energies above the HF reference on Ce3Bi4Pd3, ``nmo=100``).

The fix seeds Davidson at ``argmin(hii)`` instead, restoring the invariant
"lowest Ritz value <= min(hii) <= E_HF whenever HF is in the basis"
independently of the limb sort order.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
from torch_sci_solver.torch_impl.mult_spinless import (
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.pairs import basis_init_vector_pair_torch
from torch_sci_solver.torch_impl.spinless_helpers import (
    basis_init_vector_spinless_torch,
    dets_to_python_ints,
    generate_spinless_helpers,
    spinless_basis_from_array,
)

from sci_solvers.energy_solvers.torch_sci_solver import TorchSCISolver

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]


def test_basis_init_vector_seed_index_places_unit_seed_and_keeps_noise() -> None:
    """``seed_index=k`` moves the unit seed to k; the noise pattern is unchanged."""
    n = 7
    w0 = basis_init_vector_spinless_torch(n, init=0, dtype=torch.complex128)
    w3 = basis_init_vector_spinless_torch(n, init=0, seed_index=3, dtype=torch.complex128)
    # Same generator, same n_dets -> identical noise; only the seeded unit moves.
    expected_shift = torch.zeros(n, dtype=torch.complex128)
    expected_shift[3] = 1.0
    expected_shift[0] = -1.0
    np.testing.assert_allclose((w3 - w0).cpu().numpy(), expected_shift.numpy(), atol=1e-12)
    np.testing.assert_allclose(torch.linalg.norm(w3).item(), 1.0, atol=1e-12)
    assert abs(w3[3]) > 0.99


def test_basis_init_vector_default_seed_index_matches_pair_mode() -> None:
    """``seed_index=0`` (the default) reproduces the pair-mode init bit-for-bit."""
    for dtype in (torch.float64, torch.complex128):
        w_spinless = basis_init_vector_spinless_torch(11, init=0, dtype=dtype)
        w_pair = basis_init_vector_pair_torch(11, init=0, dtype=dtype)
        assert torch.equal(w_spinless, w_pair)


def test_basis_init_vector_rejects_out_of_range_seed_index() -> None:
    """A negative seed_index must not silently wrap to the last determinant."""
    with pytest.raises(ValueError, match="seed_index"):
        basis_init_vector_spinless_torch(5, init=0, seed_index=-1)
    with pytest.raises(ValueError, match="seed_index"):
        basis_init_vector_spinless_torch(5, init=0, seed_index=5)


def _det(modes):
    return sum(1 << m for m in modes)


# 68 spin-orbitals: determinants span two uint64 limbs (the regression's
# whole premise). hcore = diag(0, 2, 4, ...) makes the HF determinant
# (lowest-N modes) the lowest-diagonal determinant by a wide margin.
NMO = 68
N_PARTICLES = 4
HF = _det([0, 1, 2, 3])
# Double excitations of HF (rank 2 -> coupled to HF, so the true ground
# state is correlated and lies strictly BELOW min(diag)):
DOUBLES = [_det([0, 1, 4, 64]), _det([0, 2, 5, 65])]
# X lives entirely in the high limb (low limb 0), so the low-limb-first
# lexicographic basis sort puts it at index 0 -- ahead of HF -- even though
# its diagonal is ~40x higher. Any other all-high-limb determinant isolated
# from X (sharing <= 1 mode) has a strictly larger high limb, so X is
# guaranteed index 0.
X = _det([64, 65, 66, 67])


def _build_basis(n_random_isolated: int):
    """HF + doubles + X + mutually isolated random high-energy determinants.

    Two same-N determinants have zero Hamiltonian coupling iff they share at
    most N-3 = 1 mode (Slater-Condon: rank > 2), so every determinant past
    HF and the doubles is chosen to share <= 1 mode with ALL others -- a
    Davidson seeded on any of them converges to that determinant's own
    isolated eigenpair. Random dets keep sum(modes) >= 30 so their
    diagonals stay far above the HF block's.

    The basis must exceed _pick_num_block's subspace-size cap (30):
    davidson_torch dense-diagonalizes the whole space when
    dim <= davidson_block_size, which makes the seed irrelevant -- the
    regression only bites on the Krylov path (production ran dim=101).
    """
    rng = np.random.default_rng(1234)
    dets = [HF] + DOUBLES + [X]
    target = len(dets) + n_random_isolated
    guard = 0
    while len(dets) < target:
        guard += 1
        assert guard < 100_000, "greedy isolated-det search failed"
        cand = tuple(int(m) for m in rng.choice(range(NMO), size=N_PARTICLES, replace=False))
        # Mode 63 sets the sign bit of the (signed int64) low limb, which
        # sorts before every non-negative limb -- including X's 0.
        if 63 in cand:
            continue
        if sum(cand) < 30:
            continue
        if any(len(set(cand) & set(_modes_of(d))) >= 2 for d in dets):
            continue
        dets.append(_det(cand))
    return dets


def _modes_of(det: int):
    return [i for i in range(NMO) if (det >> i) & 1]


STRS = np.array(_build_basis(n_random_isolated=35), dtype=object)


def _small_random_complex_eri(nmo: int, seed: int, scale: float):
    """Random complex Hermitian-valid chemist-order eri, scaled small enough
    that the one-body diagonal gaps dominate (HF stays argmin of ``hii``)."""
    rng = np.random.default_rng(seed)
    naux = 6
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return scale * eri


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_spinless_multilimb_davidson_seed_matches_dense_ground_state(device: str) -> None:
    """The engine's ground root must match the dense oracle on a basis whose
    index 0 is a high-energy isolated determinant.

    Regression: with the old ``init=0`` seed, Davidson converged to X's
    isolated eigenpair (~524 Ha here) and reported it as the ground state,
    while the true ground state of the same basis is the HF-correlated state
    (~12 Ha). Seeding at ``argmin(hii)`` must find the latter.
    """
    hcore = np.diag(2.0 * np.arange(NMO, dtype=np.float64))
    eri = _small_random_complex_eri(NMO, seed=99, scale=0.01)

    basis = spinless_basis_from_array(STRS, NMO, N_PARTICLES, device=device)
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device=device, dtype=torch.complex128)
    hii = make_diag_terms_spinless_torch(basis, integrals)

    order = [int(v) for v in dets_to_python_ints(basis.dets)]
    hf_index = order.index(HF)
    # Scenario preconditions (fail loudly if the limb layout/sort conventions
    # change out from under this test):
    assert basis.n_limbs > 1, "nmo must span multiple limbs for this regression"
    assert basis.dim > 30, (
        "dim must exceed the Davidson subspace cap: at dim <= block size "
        "davidson_torch dense-diagonalizes and the seed is irrelevant"
    )
    assert order[0] == X, "low-limb-first sort must put the isolated det at index 0"
    assert hf_index != 0
    assert int(torch.argmin(hii)) == hf_index, "HF must be the lowest-diagonal determinant"

    # Dense oracle: one streaming matvec of the identity, then eigvalsh.
    eye = torch.eye(basis.dim, dtype=torch.complex128, device=device)
    M = mult_spinless_torch(hii, eye, basis, helpers, integrals).cpu().numpy()
    e_ref = float(np.linalg.eigvalsh(0.5 * (M + M.conj().T))[0])
    min_diag = float(hii.min())
    assert e_ref < min_diag, "HF-coupled doubles must correlate the ground state below min(diag)"

    # The engine, exactly as production calls it (scalar nelec -> spinless path).
    solver = TorchSCISolver(symmetrize_spin=False, device=device, dtype=torch.complex128)
    r0 = solver.sci_solver_callable_eigen_states([STRS], hcore, eri, NMO, N_PARTICLES)[0].states[0]

    assert r0.state == "converged"
    assert r0.energy == pytest.approx(e_ref, abs=1e-6)
    assert r0.energy <= min_diag + 1e-9
