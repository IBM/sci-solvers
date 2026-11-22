"""The spinless path on a genuinely complex spin-orbital ERI, against a brute-force oracle.

The spinless integral helper ``_random_complex_hermitian_spinorbital_integrals`` builds
its ERI as a symmetrized ``L x conj(L)`` sum, which comes out exactly real, so the rest
of the spinless suite only exercises a complex ``hcore``. Here the ERI has a nonzero
imaginary part (asserted), so complex two-body terms reach every layer:

* :func:`~sci_solvers.spinless_ci_problem.build_spinless_complex_ci_problem`, dense and
  matrix-free, entrywise;
* the spinless RDMs, through the RDM-rebuilt energy;
* the solvers end to end: ``torch_sci``'s native spinless engine and the general-algebra
  spinless path (``arpack``, ``davidson``), on the full canonical list and on a selected
  subset.

The oracle is ``_brute_force_dense``: ``H`` applied by explicit ladder operators, with
no spin assumptions.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
import torch
from test_spinless_complex_ci import _all_dets, _brute_force_dense

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.spinless_ci_problem import (
    build_spinless_complex_ci_problem,
    spinless_energy,
    spinless_rdm1,
)

CASES = [(6, 3, 1), (6, 2, 2), (8, 4, 3)]


def _genuinely_complex_spinorbital_integrals(nmo: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian ``hcore`` and a spin-orbital chemist ERI with a nonzero imaginary part.

    Enforces exactly ``(pq|rs) == conj((qp|sr))`` and ``(pq|rs) == (rs|pq)``, the two
    symmetries that make the CI operator Hermitian, and nothing more, so the imaginary
    part survives.
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    eri = rng.standard_normal((nmo,) * 4) + 1j * rng.standard_normal((nmo,) * 4)
    eri = eri + eri.conj().transpose(1, 0, 3, 2)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hc + hc.conj().T, eri


def _selected(dets: list[int]) -> list[int]:
    """A sorted two-thirds subset of ``dets``."""
    rng = np.random.default_rng(9)
    return sorted(rng.choice(dets, len(dets) * 2 // 3, replace=False).tolist())


def test_eri_is_genuinely_complex() -> None:
    """Guard: the fixture's ERI carries an imaginary part and passes the Hermiticity checks."""
    hcore, eri = _genuinely_complex_spinorbital_integrals(6, seed=1)
    assert np.abs(eri.imag).max() > 0.1
    build_spinless_complex_ci_problem(hcore, eri, 3, np.array(_all_dets(6, 3)))


@pytest.mark.parametrize("nmo, n, seed", CASES)
@pytest.mark.parametrize("matrix_free", [False, True])
def test_builder_matches_brute_force(nmo: int, n: int, seed: int, matrix_free: bool) -> None:
    """The spinless operator equals the brute-force one entrywise, dense and matrix-free."""
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed)
    dets = _all_dets(nmo, n)
    problem = build_spinless_complex_ci_problem(
        hcore, eri, n, np.array(dets), matrix_free=matrix_free
    )
    eye = np.eye(len(dets), dtype=np.complex128)
    got = np.column_stack([problem.matvec(eye[:, j]) for j in range(len(dets))])
    np.testing.assert_allclose(got, _brute_force_dense(hcore, eri, nmo, dets), atol=1e-10, rtol=0)


@pytest.mark.parametrize("nmo, n, seed", CASES)
def test_rdm_energy_matches_eigenvalue(nmo: int, n: int, seed: int) -> None:
    """The energy rebuilt from the spinless 1-/2-RDMs equals the brute-force ground eigenvalue."""
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed)
    w, v = np.linalg.eigh(_brute_force_dense(hcore, eri, nmo, _all_dets(nmo, n)))
    assert spinless_energy(v[:, 0], hcore, eri, n) == pytest.approx(w[0], abs=1e-10)


SOLVERS = [
    ("torch_sci", {"dtype": torch.complex128, "device": "cpu", "compute_full_rdm": True}),
    ("arpack", {}),
    ("davidson", {}),
]


@pytest.mark.parametrize("selected", [False, True], ids=["full", "selected"])
@pytest.mark.parametrize("nmo, n, seed", [(6, 3, 1), (8, 4, 3)])
@pytest.mark.parametrize("name, kwargs", SOLVERS, ids=[s[0] for s in SOLVERS])
def test_solver_matches_brute_force(
    name: str, kwargs: dict, nmo: int, n: int, seed: int, selected: bool
) -> None:
    """Two roots end to end: energies and the ground-state 1-RDM match brute force.

    Tolerances allow for the iterative solvers' convergence (Davidson residual), not for
    any operator difference: the operator itself is checked entrywise above.
    """
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed)
    dets = _all_dets(nmo, n)
    if selected:
        dets = _selected(dets)
    w, v = np.linalg.eigh(_brute_force_dense(hcore, eri, nmo, dets))
    strs = np.array(dets, dtype=np.int64)
    rdm1_ref = spinless_rdm1(v[:, 0], nmo, n, strs=strs)[0]

    solver = create_solver(name, symmetrize_spin=False, nroots=2, **kwargs)
    states = solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, n)[0].states
    np.testing.assert_allclose([s.energy for s in states], w[:2], atol=1e-7)
    np.testing.assert_allclose(np.asarray(states[0].rdm1), rdm1_ref, atol=1e-5)
