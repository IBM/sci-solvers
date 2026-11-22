"""Unit test comparing TruncatedDavidsonSolver against PySCFSolver on a tiny synthetic 2-orbital/2-electron system."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math

import numpy as np
import pytest
from pyscf.fci import selected_ci
from pyscf.fci.cistring import make_strings
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver
from sci_solvers.energy_solvers.truncated_davidson_solver import TruncatedDavidsonSolver
from sci_solvers.energy_solvers.truncated_davidson_solver.hamiltonian import Hamiltonian
from sci_solvers.energy_solvers.truncated_davidson_solver.solver import make_sci_result
from sci_solvers.hamiltonian import HamiltonianData

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


def test_truncated_davidson_matches_pyscf_full_ci(tiny_hamiltonian):
    """Truncated Davidson energy/spin/civec should match exact PySCF FCI on full CI."""

    ham = tiny_hamiltonian

    alpha_ints = make_strings(range(ham.num_orbitals), ham.num_elec_a)
    beta_ints = make_strings(range(ham.num_orbitals), ham.num_elec_b)
    ci_strs = (alpha_ints, beta_ints)

    td_solver = TruncatedDavidsonSolver(
        symmetrize_spin=True,
        hamiltonian=ham,
        verbose=False,
        eps=1e-4,
        max_cycle=100,
        delta_max=1e-3,
        delta_min=1e-6,
        growth_factor=2.0,
    )
    eigen_results = td_solver.compute_energy(
        ci_strs_batches=[ci_strs],
    )
    sci_state = eigen_results[0].sci_state
    e_td = eigen_results[0].energy
    sci_strs = (sci_state.ci_strs_a, sci_state.ci_strs_b)
    vec_sci = selected_ci._as_SCIvector(sci_state.amplitudes, sci_strs)
    spin_sq, mult = selected_ci.spin_square(
        vec_sci, ham.num_orbitals, (ham.num_elec_a, ham.num_elec_b)
    )

    pyscf_solver = PySCFSolver(
        symmetrize_spin=True,
        hamiltonian=ham,
        verbose=False,
    )
    pyscf_results = pyscf_solver.compute_energy(
        ci_strs_batches=[ci_strs], spin_sq=None, max_cycle=30, tol=1e-10, davidson_block_size=30
    )
    e_pyscf = pyscf_results[0].energy

    # Compare energies
    assert np.isfinite(e_td)
    assert np.isfinite(e_pyscf)

    delta_e_mha = abs(e_td - e_pyscf) * 1e3

    assert delta_e_mha < ENERGY_TOL_MHA, (
        f"|ΔE| = {delta_e_mha:.4f} mHa exceeds tolerance of {ENERGY_TOL_MHA} mHa"
    )

    # Compare spin
    assert spin_sq < SPIN_SQ_TOL, f"<S^2> = {spin_sq:.6f} exceeds tolerance of {SPIN_SQ_TOL}"


# ---------------------------------------------------------------------------
# Hamiltonian helper method tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ham_obj(tiny_hamiltonian):
    """Hamiltonian wrapper around the shared tiny HamiltonianData."""
    return Hamiltonian(tiny_hamiltonian)


def test_flat_index_and_unflatten(tiny_ham_obj):
    """flat_index and unflatten are exact inverses for all (ia, ib) pairs."""
    h = tiny_ham_obj
    for ia in range(h.na):
        for ib in range(h.nb):
            idx = h.flat_index(ia, ib)
            assert idx == ia * h.nb + ib
            assert h.unflatten(idx) == (ia, ib)


def test_to_dense_roundtrip(tiny_ham_obj):
    """A sparse vector with one nonzero entry round-trips through to_dense."""
    from numba.core import types as nbtypes
    from numba.typed import Dict as NumbaDict

    h = tiny_ham_obj
    d = NumbaDict.empty(key_type=nbtypes.int64, value_type=nbtypes.float64)
    ia, ib = 0, 1
    d[h.flat_index(ia, ib)] = 3.14
    dense = h.to_dense(d)
    assert dense.shape == (h.na, h.nb)
    assert dense[ia, ib] == pytest.approx(3.14)
    assert dense.sum() == pytest.approx(3.14)


def test_hf_vector(tiny_ham_obj):
    """HF vector has exactly one nonzero entry equal to 1.0 at the HF determinant."""
    h = tiny_ham_obj
    hf_keys, hf_vals = h.hf_vector()
    assert len(hf_keys) == 1
    assert len(hf_vals) == 1
    idx = int(hf_keys[0])
    val = float(hf_vals[0])
    assert val == pytest.approx(1.0)
    ia, ib = h.unflatten(idx)
    # HF string is the lowest nelec orbitals occupied: bitstring 0b1 for nelec=1
    hf_str = int("1" * h.nelec[0], 2)
    assert h.strs_a[ia] == hf_str
    hf_str_b = int("1" * h.nelec[1], 2)
    assert h.strs_b[ib] == hf_str_b


def test_precomputed_diag_matches_on_the_fly(tiny_hamiltonian):
    """precomputed_diag=True and False should return the same diagonal values."""
    h_pre = Hamiltonian(tiny_hamiltonian, precomputed_diag=True)
    h_fly = Hamiltonian(tiny_hamiltonian, precomputed_diag=False)
    for idx in range(h_pre.na * h_pre.nb):
        assert h_pre.diag(idx) == pytest.approx(h_fly.diag(idx), abs=1e-10), (
            f"diag mismatch at idx={idx}"
        )


# ---------------------------------------------------------------------------
# add_reference_determinant tests
# ---------------------------------------------------------------------------


def _hf_strings(norb, nelec_a, nelec_b):
    """(hf_alpha, hf_beta) reference bitstrings: lowest orbitals occupied."""
    _ = norb
    return int("1" * nelec_a, 2), int("1" * nelec_b, 2)


def _selected_ci_larger():
    """A 4-orbital/2-electron Hamiltonian plus a selected space that omits HF.

    Returns (data, strs_a_no_hf, strs_b_no_hf, hf_a, hf_b). The selected strings
    are the full C(4,2)=6 alpha/beta strings with the HF reference removed, so
    injection restores exactly the missing determinant.
    """
    norb, nelec_a, nelec_b = 4, 2, 2
    h1e = np.diag(np.linspace(-1.0, 1.0, norb))
    eri = np.zeros((norb,) * 4)
    eri[0, 0, 0, 0] = 0.5
    data = HamiltonianData(
        hcore=h1e,
        eri=eri,
        e_nuc=0.0,
        num_orbitals=norb,
        num_elec_a=nelec_a,
        num_elec_b=nelec_b,
    )
    all_a = np.asarray(make_strings(range(norb), nelec_a), dtype=np.int64)
    all_b = np.asarray(make_strings(range(norb), nelec_b), dtype=np.int64)
    hf_a, hf_b = _hf_strings(norb, nelec_a, nelec_b)
    strs_a = np.array([s for s in all_a if s != hf_a], dtype=np.int64)
    strs_b = np.array([s for s in all_b if s != hf_b], dtype=np.int64)
    return data, strs_a, strs_b, hf_a, hf_b


def test_add_reference_determinant_injects_when_missing():
    """By default a selected space missing the HF reference gets it injected, sorted."""
    data, strs_a, strs_b, hf_a, hf_b = _selected_ci_larger()
    assert hf_a not in strs_a and hf_b not in strs_b

    h = Hamiltonian(data, ci_strs=(strs_a, strs_b))

    assert hf_a in h.strs_a, "HF alpha reference not injected"
    assert hf_b in h.strs_b, "HF beta reference not injected"
    assert h.na == len(strs_a) + 1
    assert h.nb == len(strs_b) + 1
    # arrays must remain sorted ascending (selected-CI machinery assumes this)
    assert np.all(np.diff(h.strs_a) > 0)
    assert np.all(np.diff(h.strs_b) > 0)
    # hf_vector now resolves without raising
    hf_keys, hf_vals = h.hf_vector()
    assert len(hf_keys) == 1
    assert float(hf_vals[0]) == pytest.approx(1.0)


def test_add_reference_determinant_false_leaves_space_untouched():
    """With the flag off the space is verbatim and hf_vector raises when HF is absent."""
    data, strs_a, strs_b, hf_a, hf_b = _selected_ci_larger()

    h = Hamiltonian(data, ci_strs=(strs_a, strs_b), add_reference_determinant=False)

    assert h.na == len(strs_a)
    assert h.nb == len(strs_b)
    assert hf_a not in h.strs_a
    with pytest.raises(ValueError, match="Hartree-Fock"):
        h.hf_vector()


def test_add_reference_determinant_no_duplicate_when_present():
    """Injecting into a space that already contains HF does not duplicate it."""
    data, strs_a, strs_b, hf_a, hf_b = _selected_ci_larger()
    full_a = np.sort(np.append(strs_a, hf_a))
    full_b = np.sort(np.append(strs_b, hf_b))

    h = Hamiltonian(data, ci_strs=(full_a, full_b), add_reference_determinant=True)

    assert h.na == len(full_a)
    assert h.nb == len(full_b)
    assert np.sum(h.strs_a == hf_a) == 1
    assert np.sum(h.strs_b == hf_b) == 1


def test_solver_add_reference_determinant_recovers_missing_reference():
    """A solver run on a space missing HF succeeds (default) and fails when disabled."""
    data, strs_a, strs_b, hf_a, hf_b = _selected_ci_larger()
    _ = (data.num_elec_a, data.num_elec_b)

    # Default: injection makes the run succeed, and because the reference was
    # the only determinant missing from the full space, the injected space is
    # the full space -- the energy must match solving the full space directly.
    solver = TruncatedDavidsonSolver(
        symmetrize_spin=False,
        hamiltonian=data,
        eps=1e-6,
        max_cycle=50,
    )
    results = solver.compute_energy(ci_strs_batches=[(strs_a, strs_b)])
    assert np.isfinite(results[0].energy)

    full_a = np.sort(np.append(strs_a, hf_a))
    full_b = np.sort(np.append(strs_b, hf_b))
    results_full = solver.compute_energy(ci_strs_batches=[(full_a, full_b)])
    assert results[0].energy == pytest.approx(results_full[0].energy, abs=1e-8)

    # Disabled: the missing reference propagates as a ValueError.
    solver_off = TruncatedDavidsonSolver(
        symmetrize_spin=False,
        hamiltonian=data,
        eps=1e-6,
        max_cycle=50,
        add_reference_determinant=False,
    )
    with pytest.raises(ValueError, match="Hartree-Fock"):
        solver_off.compute_energy(ci_strs_batches=[(strs_a, strs_b)])


# ---------------------------------------------------------------------------
# add_basis_vector_fused column test
# ---------------------------------------------------------------------------


def _pyscf_matrix_element(h1e, eri, norb, nelec, b_i_dense, b_n_dense):
    """Compute <b_i|H|b_n> using PySCF's FCI machinery (no nuclear repulsion)."""
    from pyscf.fci import direct_spin1

    h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    Hb_n = direct_spin1.contract_2e(h2e, b_n_dense, norb, nelec)
    return float(np.dot(b_i_dense.ravel(), Hb_n.ravel()))


def test_add_basis_vector_fused_col(tiny_ham_obj):
    """col[i] returned by add_basis_vector_fused equals <b_i|H|b_n> for all i.

    Strategy:
      - Build a SubspaceHamiltonianBuilder from the tiny 2-orbital system.
      - Add k random normalised basis vectors b_0 … b_{k-1}, then add b_n.
      - After the last add_basis_vector call, the last column of Hmat is
        exactly the col filled by add_basis_vector_fused.
      - Compare every entry of that column against the PySCF reference
        <b_i|H|b_n> (no nuclear repulsion, since Hamiltonian stores bare
        integrals without E_nuc).
    """
    from sci_solvers.energy_solvers.truncated_davidson_solver import sparse_vector as spv
    from sci_solvers.energy_solvers.truncated_davidson_solver.subspace_hamiltonian import (
        SubspaceHamiltonianBuilder,
    )

    H = tiny_ham_obj
    ham = H.data
    norb = H.norb
    nelec = H.nelec  # (nelec_a, nelec_b)

    rng = np.random.default_rng(42)

    def random_sparse_vector():
        """Normalised random dense CI vector as a compact (keys, vals) sparse vector."""
        v = rng.standard_normal((H.na, H.nb))
        v /= np.linalg.norm(v)
        flat = v.reshape(-1)
        keys = np.arange(len(flat), dtype=np.int64)
        vals = flat.astype(np.float64)
        return spv.from_flat(keys, vals), v

    builder = SubspaceHamiltonianBuilder(H, warmup_jit=True)

    # Add k existing basis vectors first, then add b_n as the last one.
    k = 3  # number of vectors already in the basis before the one being tested
    dense_basis = []
    for _ in range(k):
        sp, dense = random_sparse_vector()
        builder.add_basis_vector(sp)
        dense_basis.append(dense)

    sp_n, dense_n = random_sparse_vector()
    builder.add_basis_vector(sp_n)
    dense_basis.append(dense_n)

    # The last column of Hmat is col filled by add_basis_vector_fused for b_n.
    col_actual = builder.Hmat[:, -1]

    # PySCF reference: <b_i|H|b_n> for i in 0..k (inclusive)
    h1e = np.array(ham.hcore)
    eri_full = ham.eri
    for i, b_i in enumerate(dense_basis):
        expected = _pyscf_matrix_element(h1e, eri_full, norb, nelec, b_i, dense_n)
        assert col_actual[i] == pytest.approx(expected, abs=1e-8), (
            f"col[{i}] = {col_actual[i]:.10f}, <b_{i}|H|b_n> = {expected:.10f}, "
            f"diff = {abs(col_actual[i] - expected):.2e}"
        )


# ---------------------------------------------------------------------------
# Factory end-to-end test
# ---------------------------------------------------------------------------


def test_truncated_davidson_via_factory_end_to_end():
    """Test truncated_davidson solver via factory end-to-end on real SqDRIFT shots."""
    from methane_reference import methane, methane_bit_array, methane_fci_energy

    ham = methane()
    solver = create_solver(
        "truncated_davidson",
        symmetrize_spin=True,
        hamiltonian=ham,
        eps=1e-4,
        max_cycle=3,
        delta_max=1e-3,
        delta_min=1e-6,
        growth_factor=2.0,
        precomputed_diag=False,
        verbose=True,
    )
    bit_array = methane_bit_array()
    hf_a = (1 << ham.num_elec_a) - 1
    hf_b = (1 << ham.num_elec_b) - 1
    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=ham.hcore,
        two_body_tensor=ham.eri,
        bit_array=bit_array,
        samples_per_batch=512,
        norb=ham.num_orbitals,
        nelec=(ham.num_elec_a, ham.num_elec_b),
        max_iterations=5,
        sci_solver=solver.sci_solver_callable,
        max_dim=40,
        num_batches=1,
        symmetrize_spin=True,
        include_configurations=([hf_a], [hf_b]),
        seed=0,
    )
    assert math.isfinite(result.energy)
    # Variational: a truncated subspace (max_dim=40) is never below the exact energy.
    e_fci = methane_fci_energy()
    assert e_fci - 1e-8 <= result.energy < e_fci + 0.2


def test_make_sci_result_from_amps_matches_dense(tiny_hamiltonian):
    """TruncatedDavidsonSolver (sparse path) must match truncated_davidson + dense + make_sci_result.

    Both sides run the same diagonalization. The reference builds the SCIResult
    via _to_dense + make_sci_result (the original dense path). The path under
    test is TruncatedDavidsonSolver.compute_energy, which internally uses
    make_sci_result_from_amps on the nnz-only string subset. They must agree on
    energy, RDM1, RDM2, orbital occupancies, and spin-square.
    """
    from sci_solvers.energy_solvers.truncated_davidson_solver.solver import _to_dense
    from sci_solvers.energy_solvers.truncated_davidson_solver.truncated_davidson import (
        truncated_davidson,
    )

    RDM_TOL = 1e-12

    ham = tiny_hamiltonian
    norb = ham.num_orbitals
    nelec = (ham.num_elec_a, ham.num_elec_b)
    strs_a = make_strings(range(norb), nelec[0])
    strs_b = make_strings(range(norb), nelec[1])

    # The solver exposes the iteration cap as `max_cycle`; the underlying kernel
    # still calls it `max_iter`, so the shared knobs are kept separate from it.
    SHARED_KWARGS = dict(eps=1e-4, delta_max=1e-3, delta_min=1e-6, growth_factor=1.5)
    MAX_ITER = 100

    # --- sparse path (under test) ---
    td_solver = TruncatedDavidsonSolver(
        symmetrize_spin=False,
        hamiltonian=ham,
        verbose=False,
        max_cycle=MAX_ITER,
        **SHARED_KWARGS,
    )
    res_sparse = td_solver.compute_energy(
        ci_strs_batches=[(strs_a, strs_b)],
        compute_rdm1=True,
        compute_rdm2=True,
    )[0]

    # --- dense reference path ---
    E, psi_ritz_list, dim, _ = truncated_davidson(
        ci_strings=(strs_a, strs_b),
        one_body_tensor=ham.hcore,
        two_body_tensor=ham.eri,
        norb=norb,
        nelec=nelec,
        nroots=1,
        n_extra_roots=0,
        precomputed_diag=False,
        verbose=False,
        e_nuc=ham.e_nuc,
        max_iter=MAX_ITER,
        **SHARED_KWARGS,
    )
    energy = float(E[0])
    vec_dense = _to_dense(psi_ritz_list[0], dim)
    res_dense = make_sci_result(
        energy,
        vec_dense,
        strs_a,
        strs_b,
        norb,
        nelec,
        compute_rdm1=True,
        compute_rdm2=True,
    )

    # --- comparisons ---
    assert res_sparse.energy == pytest.approx(res_dense.energy, abs=1e-10)

    assert res_sparse.rdm1 is not None and res_dense.rdm1 is not None
    assert np.max(np.abs(res_sparse.rdm1 - res_dense.rdm1)) < RDM_TOL

    assert res_sparse.rdm2 is not None and res_dense.rdm2 is not None
    assert np.max(np.abs(res_sparse.rdm2 - res_dense.rdm2)) < RDM_TOL

    occ_a_s, occ_b_s = res_sparse.orbital_occupancies
    occ_a_d, occ_b_d = res_dense.orbital_occupancies
    assert np.max(np.abs(occ_a_s - occ_a_d)) < RDM_TOL
    assert np.max(np.abs(occ_b_s - occ_b_d)) < RDM_TOL

    def _spin_sq(res):
        sc = res.sci_state
        vec_sci = selected_ci._as_SCIvector(sc.amplitudes, (sc.ci_strs_a, sc.ci_strs_b))
        ss, _ = selected_ci.spin_square(vec_sci, norb, nelec)
        return ss

    assert abs(_spin_sq(res_sparse) - _spin_sq(res_dense)) < RDM_TOL
