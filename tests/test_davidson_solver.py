"""Unit tests for davidson_solver's custom FCI / Selected-CI kernels.

Uses a tiny synthetic 2-orbital / 2-electron (1a,1b) singlet Hamiltonian so
tests run instantly and reference energies can be computed with dense
diagonalization as a cross-check.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest
from pyscf.fci import direct_spin0, direct_spin1
from pyscf.fci.cistring import make_strings

import sci_solvers.energy_solvers.davidson_solver as _shift_dav
from sci_solvers.energy_solvers.davidson_solver import DavidsonSolver
from sci_solvers.hamiltonian import HamiltonianData

fci_kernel = _shift_dav.fci_kernel
sci_kernel_fixed_space = _shift_dav.sci_kernel_fixed_space

ENERGY_TOL_MHA = 0.001  # mHa
SPIN_SQ_TOL = 1e-8

NORB = 2
NELEC_A = 1
NELEC_B = 1
E_NUC = -76.83741213074619

HCORE = np.array([
    [-0.8253014918977718, 0.0],
    [0.0, -0.5344324046820699],
])
ERI = np.array([
    [
        [
            [4.487069375223534e-01, -1.479277142336694e-15],
            [-1.479277142336694e-15, 4.544572923512384e-01],
        ],
        [
            [-1.479277142336694e-15, 1.313613621061964e-01],
            [1.313613621061964e-01, 1.703410010234202e-15],
        ],
    ],
    [
        [
            [-1.479277142336694e-15, 1.313613621061964e-01],
            [1.313613621061964e-01, 1.703410010234202e-15],
        ],
        [
            [4.544572923512384e-01, 1.703410010234202e-15],
            [1.703410010234202e-15, 4.855786851181397e-01],
        ],
    ],
])


@pytest.fixture(scope="module")
def tiny_hamiltonian():
    """Small synthetic 2-orbital/2-electron Hamiltonian, built in-memory."""
    return HamiltonianData(
        hcore=HCORE,
        eri=ERI,
        e_nuc=E_NUC,
        num_orbitals=NORB,
        num_elec_a=NELEC_A,
        num_elec_b=NELEC_B,
    )


@pytest.fixture(scope="module")
def e_ref(tiny_hamiltonian):
    """Ground-truth ground-state energy from pyscf direct_spin1."""
    ham = tiny_hamiltonian
    nelec = (ham.num_elec_a, ham.num_elec_b)
    e, _ = direct_spin1.kernel(
        ham.hcore,
        ham.eri,
        ham.num_orbitals,
        nelec,
        ecore=ham.e_nuc,
        nroots=1,
        tol=1e-10,
        lindep=1e-14,
    )
    return e


@pytest.fixture(scope="module")
def full_ci_strs(tiny_hamiltonian):
    """Full CI string sets (all determinants) for the tiny 2-orbital system."""
    ham = tiny_hamiltonian
    alpha = make_strings(range(ham.num_orbitals), ham.num_elec_a)
    beta = make_strings(range(ham.num_orbitals), ham.num_elec_b)
    return (alpha, beta)


def _assert_close(e, e_ref, label):
    de_mha = abs(e - e_ref) * 1e3
    assert de_mha < ENERGY_TOL_MHA, (
        f"{label}: E={e:.10f}  E_ref={e_ref:.10f}  ΔE={de_mha:.6f} mHa (tol={ENERGY_TOL_MHA} mHa)"
    )


class TestFCIKernel:
    """Tests for the custom fci_kernel (spin1 and spin0 paths)."""

    def test_fci_spin1(self, tiny_hamiltonian, e_ref):
        """spin1 path must match pyscf direct_spin1 reference energy."""
        ham = tiny_hamiltonian
        nelec = (ham.num_elec_a, ham.num_elec_b)
        e, c = fci_kernel(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            spin=None,
            nroots=1,
            ecore=ham.e_nuc,
            shift_mode="ground",
            tol=1e-10,
            lindep=1e-14,
        )
        _assert_close(e, e_ref, "custom_fci_spin1")

    def test_fci_spin0(self, tiny_hamiltonian, e_ref):
        """spin0 path must match both pyscf direct_spin1 and direct_spin0 reference energies."""
        ham = tiny_hamiltonian
        nelec = (ham.num_elec_a, ham.num_elec_b)
        e, c = fci_kernel(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            spin=0,
            nroots=1,
            ecore=ham.e_nuc,
            shift_mode="ground",
            tol=1e-10,
            lindep=1e-14,
        )
        _assert_close(e, e_ref, "custom_fci_spin0")

        # also check against pyscf's own spin0 solver directly
        e_spin0_ref, _ = direct_spin0.kernel(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            ecore=ham.e_nuc,
            nroots=1,
            tol=1e-10,
            lindep=1e-14,
        )
        _assert_close(e, e_spin0_ref, "custom_fci_spin0 vs pyscf_spin0")


class TestSCIKernelFixedSpace:
    """Tests for sci_kernel_fixed_space on the full determinant space."""

    def test_sci_spin1_full_space(self, tiny_hamiltonian, full_ci_strs, e_ref):
        """spin1 path over full CI space must reproduce the reference FCI energy."""
        ham = tiny_hamiltonian
        nelec = (ham.num_elec_a, ham.num_elec_b)
        e, c = sci_kernel_fixed_space(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            full_ci_strs,
            spin=None,
            nroots=1,
            ecore=ham.e_nuc,
            shift_mode="ground",
            tol=1e-10,
            lindep=1e-14,
        )
        _assert_close(e, e_ref, "custom_sci_spin1 (full space)")

    def test_sci_spin0_full_space(self, tiny_hamiltonian, full_ci_strs, e_ref):
        """spin0 path over full CI space must reproduce the reference FCI energy."""
        ham = tiny_hamiltonian
        nelec = (ham.num_elec_a, ham.num_elec_b)
        e, c = sci_kernel_fixed_space(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            full_ci_strs,
            spin=0,
            nroots=1,
            ecore=ham.e_nuc,
            shift_mode="ground",
            tol=1e-10,
            lindep=1e-14,
        )
        _assert_close(e, e_ref, "custom_sci_spin0 (full space)")


def test_sci_spin0_requires_matching_strings(tiny_hamiltonian):
    """spin0 path must reject mismatched alpha/beta string sets."""
    ham = tiny_hamiltonian
    nelec = (ham.num_elec_a, ham.num_elec_b)
    bad_ci_strs = (np.array([1]), np.array([2]))  # deliberately different
    with pytest.raises(ValueError):
        sci_kernel_fixed_space(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            bad_ci_strs,
            spin=0,
            nroots=1,
            ecore=ham.e_nuc,
        )


class TestDavidsonSolver:
    """Tests for the DavidsonSolver class across all shift_mode variants."""

    @classmethod
    @pytest.fixture(scope="class")
    def full_ci_strs(cls, tiny_hamiltonian):
        """Full CI string sets scoped to this test class."""
        ham = tiny_hamiltonian
        alpha = make_strings(range(ham.num_orbitals), ham.num_elec_a)
        beta = make_strings(range(ham.num_orbitals), ham.num_elec_b)
        return (alpha, beta)

    @pytest.mark.parametrize("shift_mode", ["ground", "active", "per_root"])
    def test_shift_modes_match_reference(self, tiny_hamiltonian, full_ci_strs, e_ref, shift_mode):
        """All three shift_mode values must converge to the reference ground-state energy."""
        ham = tiny_hamiltonian
        solver = DavidsonSolver(
            hamiltonian=ham,
            symmetrize_spin=False,
            shift_mode=shift_mode,
        )
        eigen_states_list = solver.sci_solver_callable_eigen_states(
            [full_ci_strs],
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            (ham.num_elec_a, ham.num_elec_b),
        )
        assert len(eigen_states_list) == 1
        e = eigen_states_list[0].states[0].energy
        _assert_close(e, e_ref - ham.e_nuc, f"DavidsonSolver shift_mode={shift_mode}")

    def test_two_batches_independent(self, tiny_hamiltonian, full_ci_strs):
        """Two identical batches should yield the same energy."""
        ham = tiny_hamiltonian
        solver = DavidsonSolver(
            hamiltonian=ham,
            symmetrize_spin=False,
            shift_mode="ground",
        )
        eigen_states_list = solver.sci_solver_callable_eigen_states(
            [full_ci_strs, full_ci_strs],
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            (ham.num_elec_a, ham.num_elec_b),
        )
        assert len(eigen_states_list) == 2
        e0 = eigen_states_list[0].states[0].energy
        e1 = eigen_states_list[1].states[0].energy
        assert abs(e0 - e1) < 1e-12, "Two identical batches must give identical energies"

    def test_spin0_path(self, tiny_hamiltonian, full_ci_strs, e_ref):
        """spin_sq=0 with neleca==nelecb should activate the spin0 code path."""
        ham = tiny_hamiltonian
        solver = DavidsonSolver(
            hamiltonian=ham,
            symmetrize_spin=False,
            shift_mode="ground",
        )
        eigen_states_list = solver.sci_solver_callable_eigen_states(
            [full_ci_strs],
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            (ham.num_elec_a, ham.num_elec_b),
            spin_sq=0,
        )
        e = eigen_states_list[0].states[0].energy
        _assert_close(e, e_ref - ham.e_nuc, "DavidsonSolver spin0 path")

    def test_multiroot_matches_reference(self, tiny_hamiltonian, full_ci_strs):
        """nroots>1 must return that many ascending roots, matching pyscf's multi-root FCI.

        The 2-orbital/2-electron full space has 4 determinants, so nroots=3 is a
        valid request. self.nroots is threaded into sci_kernel_fixed_space (which
        returns parallel [energies],[civecs] lists for nroots>1), one SCIResult
        built per root — restoring the multi-root capability the adapters had.
        """
        ham = tiny_hamiltonian
        nelec = (ham.num_elec_a, ham.num_elec_b)
        nroots = 3
        solver = DavidsonSolver(
            hamiltonian=ham,
            symmetrize_spin=False,
            shift_mode="per_root",
            nroots=nroots,
        )
        assert solver.nroots == nroots
        eigen_states_list = solver.sci_solver_callable_eigen_states(
            [full_ci_strs],
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
        )
        assert len(eigen_states_list) == 1
        states = eigen_states_list[0].states
        assert len(states) == nroots, f"expected {nroots} roots, got {len(states)}"

        energies = [s.energy for s in states]
        assert energies == sorted(energies), f"roots not ascending: {energies}"

        # Cross-check against pyscf direct_spin1 for the lowest nroots (electronic
        # only: ecore=0 matches the solver's kernel call).
        e_ref_multi, _ = direct_spin1.kernel(
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            nelec,
            ecore=0.0,
            nroots=nroots,
            tol=1e-10,
            lindep=1e-14,
        )
        for e_got, e_expected in zip(energies, np.sort(np.atleast_1d(e_ref_multi))):
            _assert_close(e_got, e_expected, "DavidsonSolver multiroot")
