"""End-to-end tests for the spinless (flat spin-orbital) path through
``TorchSCISolver.__call__`` and the ``sci_solvers`` public wrapper.

A scalar ``nelec`` (total particle number, not an ``(n_a, n_b)`` tuple)
dispatches to the spinless path, mirroring
``GeneralAlgebraSolverBase._is_spinless_request``. These tests exercise the
whole stack -- ``make_torch_sci_solver`` / ``TorchSCISolver.__call__`` down
through the flat matvec/Davidson/RDM kernels tested in isolation elsewhere
in this directory -- and the feature-gate rejections (multi-root, spin_sq,
real dtype) documented on ``_solve_spinless``.
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
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

from sci_solvers.energy_solvers.factory import is_complex_capable
from sci_solvers.energy_solvers.torch_sci_solver import TorchSCISolver
from sci_solvers.spinless_ci_problem import build_spinless_complex_ci_problem

DEVICE_PARAMS = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
    ),
]


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


def _dense_ground_energy(hcore, eri, n, strs):
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    H = problem._dense
    return float(np.min(np.linalg.eigvalsh(0.5 * (H + H.conj().T))))


def test_torch_sci_is_complex_capable() -> None:
    """torch_sci is complex-capable."""
    assert is_complex_capable("torch_sci")


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_solver_call_dispatches_on_scalar_nelec(device: str) -> None:
    """A scalar nelec dispatches to the spinless path and matches the dense oracle."""
    nmo, n, seed = 5, 2, 42
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    e_ref = _dense_ground_energy(hcore, eri, n, strs)

    solver = make_torch_sci_solver(device=device, dtype=torch.complex128, compute_full_rdm=True)
    results = solver(
        ci_strings=[strs],
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=nmo,
        nelec=n,  # scalar -- spinless dispatch
    )
    assert len(results) == 1
    es = results[0]
    assert len(es.states) == 1
    r0 = es.states[0]
    assert r0.state == "converged"
    assert abs(r0.energy - e_ref) < 1e-6
    assert len(r0.orbital_occupancies) == 1  # one flat occupancy vector, no alpha/beta split
    assert r0.orbital_occupancies[0].shape == (nmo,)
    np.testing.assert_allclose(np.sum(r0.orbital_occupancies[0]), n, atol=1e-6)
    assert r0.rdm1.shape == (nmo, nmo)
    assert r0.rdm2.shape == (nmo, nmo, nmo, nmo)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_solver_compute_full_rdm_false_matches_true_energy_and_occupancies(device: str) -> None:
    """``compute_full_rdm=False`` (the default) must give the same energy and
    occupancies as ``compute_full_rdm=True`` -- only whether ``rdm1``/``rdm2``
    are populated should differ.

    Regression for an OOM: ``_solve_spinless`` used to call
    ``spinless_rdm12_torch`` (a dense ``(nmo*nmo, nmo*nmo)`` Gram matrix for
    dm2, O(nmo**4) memory) unconditionally to get the occupancy diagonal, even
    when ``compute_full_rdm=False`` threw dm2 away immediately -- this OOM'd
    at nmo=60 on a run that never asked for the full RDM. The occupancy
    diagonal is now built directly via ``spinless_rdm1_diag_torch`` (O(dim *
    nmo)), and the full ``spinless_rdm12_torch`` build only runs when
    ``compute_full_rdm=True``.
    """
    nmo, n, seed = 5, 2, 80
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))

    solver_diag_only = make_torch_sci_solver(
        device=device, dtype=torch.complex128, compute_full_rdm=False
    )
    r_diag_only = solver_diag_only(
        ci_strings=[strs], one_body_tensor=hcore, two_body_tensor=eri, norb=nmo, nelec=n
    )[0].states[0]

    solver_full = make_torch_sci_solver(
        device=device, dtype=torch.complex128, compute_full_rdm=True
    )
    r_full = solver_full(
        ci_strings=[strs], one_body_tensor=hcore, two_body_tensor=eri, norb=nmo, nelec=n
    )[0].states[0]

    assert r_diag_only.state == "converged"
    assert r_full.state == "converged"
    assert r_diag_only.energy == pytest.approx(r_full.energy, abs=1e-10)
    np.testing.assert_allclose(
        r_diag_only.orbital_occupancies[0], r_full.orbital_occupancies[0], atol=1e-10
    )

    assert r_diag_only.rdm1 is None
    assert r_diag_only.rdm2 is None
    assert r_full.rdm1 is not None
    assert r_full.rdm2 is not None
    np.testing.assert_allclose(
        r_full.orbital_occupancies[0], np.real(np.diagonal(r_full.rdm1)), atol=1e-10
    )


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_solver_call_multiple_subspaces(device: str) -> None:
    """ci_strings with several flat-strs entries solves each independently."""
    nmo, n = 5, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=50)
    all_strs = np.asarray(cistring.make_strings(range(nmo), n))
    rng = np.random.default_rng(51)
    sub_a = rng.choice(all_strs, size=len(all_strs) - 1, replace=False)
    sub_b = all_strs  # full canonical

    solver = make_torch_sci_solver(device=device, dtype=torch.complex128)
    results = solver(
        ci_strings=[sub_a, sub_b],
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=nmo,
        nelec=n,
    )
    assert len(results) == 2
    for es in results:
        assert es.states[0].state == "converged"


def test_solver_rejects_real_dtype_with_complex_integrals() -> None:
    """A real-dtype solver (the default) raises on complex integrals rather than truncating."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=60)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    solver = make_torch_sci_solver(device="cpu", dtype=torch.float64)  # real, default
    with pytest.raises(TypeError, match="complex"):
        solver(
            ci_strings=[strs],
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=nmo,
            nelec=n,
        )


def test_solver_rejects_spin_sq_on_spinless_path() -> None:
    """spin_sq has no meaning on the spinless path (no alpha/beta split to target)."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=61)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128)
    with pytest.raises(ValueError, match="spin_sq"):
        solver(
            ci_strings=[strs],
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=nmo,
            nelec=n,
            spin_sq=0.0,
        )


def test_solver_rejects_compute_spin_squared_on_spinless_path() -> None:
    """compute_spin_squared has no meaning on the spinless path either."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=63)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128, compute_spin_squared=True)
    with pytest.raises(ValueError, match="compute_spin_squared"):
        solver(
            ci_strings=[strs],
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=nmo,
            nelec=n,
        )


def test_capabilities_report_spinless_warm_start_and_spin_limitation() -> None:
    """Discovery advertises the probe state and states why S^2 does not apply."""
    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128)
    spinless = solver.capabilities()["modes"]["spinless"]
    assert spinless["warm_start"] is True
    assert spinless["probe_state_type"] == "TorchSpinlessSCIState"
    # The refusals are physics limitations, with a reason a workflow can read.
    assert spinless["spin_squared"] is False
    assert spinless["spin_penalty"] is False
    assert "alpha/beta" in spinless["spin_squared_unsupported_reason"]
    assert "alpha/beta" in spinless["spin_penalty_unsupported_reason"]


def test_solver_warm_start_probe_round_trip() -> None:
    """Cold-with-probe returns a spinless state that warms an extension."""
    nmo, n = 5, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=64)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    solver = make_torch_sci_solver(
        device="cpu", dtype=torch.complex128, davidson_block_size=4, davidson_tolerance=1e-9
    )
    cold = solver(ci_strings=[strs], one_body_tensor=hcore, two_body_tensor=eri, norb=nmo, nelec=n)[
        0
    ].states[0]
    results, states = solver(
        ci_strings=[strs],
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=nmo,
        nelec=n,
        probe_method="warm_start",
    )
    assert results[0].states[0].energy == pytest.approx(cold.energy, abs=1e-10)
    warm = solver(
        ci_strings=[strs],
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=nmo,
        nelec=n,
        probe_method="warm_start",
        probe_state=states[0],
    )[0].states[0]
    assert warm.energy == pytest.approx(cold.energy, abs=1e-10)


def test_solver_accepts_tile_on_spinless_path() -> None:
    """A tile spec is no longer rejected; a basis below one block keeps 1-D."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=62)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    reference = make_torch_sci_solver(device="cpu", dtype=torch.complex128)
    e_ref = float(reference([strs], hcore, eri, nmo, n)[0].states[0].energy)
    solver = make_torch_sci_solver(
        device="cpu",
        dtype=torch.complex128,
        distributed={"world_size": 2, "backend": "gloo", "tile": "auto"},
    )
    try:
        state = solver([strs], hcore, eri, nmo, n)[0].states[0]
    finally:
        solver.close()
    assert float(state.energy) == pytest.approx(e_ref, abs=1e-8)


@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_solver_multiroot_on_spinless_path_matches_dense_low_lying_spectrum(device: str) -> None:
    """Multi-root is supported at the same level as the spatial (alpha/beta) path.

    ``davidson_torch`` is basis-shape-agnostic, so ``n_roots > 1`` at
    construction returns the ``n_roots`` lowest states, ascending in energy,
    exactly as it does on the spatial path.
    """
    nmo, n, seed = 6, 3, 63
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    problem = build_spinless_complex_ci_problem(hcore, eri, n, strs, matrix_free=False)
    H_dense = problem._dense
    n_roots = 3
    e_ref = np.sort(np.linalg.eigvalsh(0.5 * (H_dense + H_dense.conj().T)))[:n_roots]

    solver = make_torch_sci_solver(device=device, dtype=torch.complex128, n_roots=n_roots)
    results = solver(
        ci_strings=[strs],
        one_body_tensor=hcore,
        two_body_tensor=eri,
        norb=nmo,
        nelec=n,
    )
    assert len(results) == 1
    es = results[0]
    assert len(es.states) == n_roots
    energies = np.array([s.energy for s in es.states])
    # states[] is already ascending in energy per the docstring contract.
    np.testing.assert_allclose(np.sort(energies), energies, atol=1e-9)
    np.testing.assert_allclose(energies, e_ref, atol=1e-5)
    for s in es.states:
        assert s.state == "converged"


def test_sci_solvers_wrapper_scalar_nelec_matches_dense_oracle() -> None:
    """The sci_solvers public TorchSCISolver wrapper (not the torch_impl one)."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=70)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    e_ref = _dense_ground_energy(hcore, eri, n, strs)

    solver = TorchSCISolver(
        symmetrize_spin=False, device="cpu", dtype=torch.complex128, compute_full_rdm=True
    )
    results = solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, n)
    assert len(results) == 1
    r0 = results[0].states[0]
    assert abs(r0.energy - e_ref) < 1e-6


def test_sci_solvers_wrapper_rejects_real_dtype() -> None:
    """Same rejection as the torch_impl-level test, through the sci_solvers wrapper."""
    nmo, n = 4, 2
    hcore, eri = _random_complex_hermitian_spinorbital_integrals(nmo, seed=71)
    strs = np.asarray(cistring.make_strings(range(nmo), n))
    solver = TorchSCISolver(symmetrize_spin=False, device="cpu")  # dtype default: real
    with pytest.raises(NotImplementedError, match="complex"):
        solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, n)
