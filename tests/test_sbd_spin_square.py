"""``<S^2>`` of the state :class:`SBDSolver` returns, on analytically known systems.

``sbd`` reports only an energy; ``<S^2>`` is computed afterwards by PySCF from the
amplitudes the wrapper puts in its ``SCIState``. That makes it the sharpest available
check on those amplitudes: the energy is a variational scalar that stays correct even
when the returned state is wrong (a stub, mis-shaped, or in the wrong sign convention),
whereas ``<S^2>`` is off-diagonal and detects all three.

Each case below diagonalizes a FULL CI space, so the ground state is a spin eigenstate
and ``<S^2>`` has an exact target ``S(S+1)`` — no reference solver needed. Both a
closed-shell singlet (target 0) and open-shell triplets (target 2, with ``na != nb`` and
a non-square amplitude matrix) are covered, since the interleaved-Wick sign bridge is
per-``(alpha, beta)`` pair and a formula that is right only for ``na == nb`` would pass
the singlet alone.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import shutil

import numpy as np
import pytest

# `<S^2>` here must be an INDEPENDENT check of the amplitudes, so it goes through
# PySCF rather than the torch bridge the solver itself applies.
from pyscf.fci import cistring, selected_ci

# ``spin`` is 2 * Ms: the multiplicity target is Ms (Ms + 1) on a full CI ground state.
_CASES = [
    # (id, atom, basis, spin, expected <S^2>)
    ("h4_singlet", "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", "sto-3g", 0, 0.0),
    ("o2_triplet", "O 0 0 0; O 0 0 1.21", "sto-3g", 2, 2.0),
    ("ch2_triplet", "C 0 0 0; H 0 0.98 0.55; H 0 -0.98 0.55", "sto-3g", 2, 2.0),
]


def _full_ci_problem(atom: str, basis: str, spin: int):
    """Build integrals and the complete CI string lists for one small molecule."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    norb = int(mol.nao)
    hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)
    return hcore, eri, norb, nelec, strs_a, strs_b


def _spin_square(state, norb: int, nelec: tuple[int, int]) -> float:
    """``<S^2>`` of an ``SCIState``, via PySCF's spin-block convention."""
    vec = selected_ci._as_SCIvector(
        np.ascontiguousarray(np.asarray(state.amplitudes, dtype=np.float64)),
        (state.ci_strs_a, state.ci_strs_b),
    )
    return float(selected_ci.spin_square(vec, norb, nelec)[0])


@pytest.mark.parametrize(
    ("atom", "basis", "spin", "expected_s2"),
    [pytest.param(*case[1:], id=case[0]) for case in _CASES],
)
def test_tpb_spin_square_matches_multiplicity(
    atom: str, basis: str, spin: int, expected_s2: float
) -> None:
    """A full-CI ground state from ``mode="tpb"`` has ``<S^2> == S(S + 1)``.

    Guards the whole amplitude path at once. Without the interleaved-Wick to spin-block
    sign bridge this fails while the energy stays exact -- the failure mode that made the
    underlying bug invisible.
    """
    pytest.importorskip("sbd", reason="the in-process `sbd` package is not installed")
    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, strs_a, strs_b = _full_ci_problem(atom, basis, spin)

    solver = SBDSolver(symmetrize_spin=False, mode="tpb")
    eigen_states = solver.sci_solver_callable_eigen_states(
        [(strs_a, strs_b)], hcore, eri, norb, nelec
    )
    state = eigen_states[0].states[0].sci_state

    # A uniform stub would still be normalized, so check the shape and the spread of
    # the amplitudes too: together these pin down "a real wavefunction over the
    # subspace we asked for" before <S^2> is trusted to mean anything.
    amplitudes = np.asarray(state.amplitudes)
    assert amplitudes.shape == (len(state.ci_strs_a), len(state.ci_strs_b))
    assert len(np.unique(np.abs(amplitudes.round(12)))) > 1, (
        "amplitudes are all one magnitude -- the solver returned a uniform stub state"
    )
    assert np.linalg.norm(amplitudes) == pytest.approx(1.0, abs=1e-8)

    assert _spin_square(state, norb, nelec) == pytest.approx(expected_s2, abs=1e-6)


def test_tpb_energy_matches_full_ci() -> None:
    """``mode="tpb"`` on a full CI space reproduces PySCF's FCI energy.

    Pairs with the ``<S^2>`` checks: this pins the eigenvalue, those pin the
    eigenvector. A sign-convention or reshape fault moves only the latter, so the two
    together localize a regression to one side or the other.
    """
    pytest.importorskip("sbd", reason="the in-process `sbd` package is not installed")
    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")

    from pyscf.fci import direct_spin1

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    atom, basis, spin = _CASES[0][1], _CASES[0][2], _CASES[0][3]
    hcore, eri, norb, nelec, strs_a, strs_b = _full_ci_problem(atom, basis, spin)

    e_fci, _ = direct_spin1.kernel(hcore, eri, norb, nelec)

    solver = SBDSolver(symmetrize_spin=False, mode="tpb")
    eigen_states = solver.sci_solver_callable_eigen_states(
        [(strs_a, strs_b)], hcore, eri, norb, nelec
    )

    # Electronic energy only, by the SCIResult contract -- and the FCIDUMP the wrapper
    # regenerates carries ECORE=0, so this compares directly against PySCF's FCI.
    assert eigen_states[0].states[0].energy == pytest.approx(e_fci, abs=1e-8)
