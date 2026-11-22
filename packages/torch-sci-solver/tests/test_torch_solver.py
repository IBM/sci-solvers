"""End-to-end checks that the addon entry points match the qiskit-addon-sqd
contract and reproduce the FCI ground state on a small molecule.

These tests need ``pyscf`` and ``qiskit-addon-sqd`` (the ``[dev]`` extra).
They run on CPU and finish in a couple of seconds.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402
from torch_sci_solver import (  # noqa: E402
    make_torch_sci_solver,
    solve_fermion,
    solve_sci_batch,
)


def _h2_integrals(bond=0.74):
    """RHF integrals for H2 in STO-3G and the exact FCI energy."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {bond}", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    nelec = mol.nelec  # (n_alpha, n_beta)
    e_fci = fci.FCI(mf).kernel()[0]
    return h1, eri, e_core, norb, nelec, e_fci


def _full_ci_strings(norb, nelec):
    """All CI strings for the full Hilbert space (small systems only)."""
    n_a, n_b = nelec

    def strings(n_elec):
        """Strings."""
        out = []
        for occ in itertools.combinations(range(norb), n_elec):
            s = 0
            for o in occ:
                s |= 1 << o
            out.append(s)
        return np.array(sorted(out), dtype=np.int64)

    return strings(n_a), strings(n_b)


def test_solve_fermion_matches_fci_on_h2():
    """Test solve fermion matches fci on h2."""
    h1, eri, e_core, norb, nelec, e_fci = _h2_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)

    energy, sci_state, occs, spin_sq = solve_fermion((ci_a, ci_b), hcore=h1, eri=eri, device="cpu")

    # solve_fermion returns the electronic energy (no nuclear repulsion);
    # add e_core to compare against the total FCI energy.
    assert energy + e_core == pytest.approx(e_fci, abs=1e-9)

    # Return-shape contract: (float, SCIState, (occ_a, occ_b), float).
    occ_a, occ_b = occs
    assert occ_a.shape == (norb,)
    assert occ_b.shape == (norb,)
    assert np.isclose(occ_a.sum(), nelec[0], atol=1e-9)
    assert np.isclose(occ_b.sum(), nelec[1], atol=1e-9)
    assert sci_state.amplitudes.shape == (len(ci_a), len(ci_b))
    assert isinstance(spin_sq, float)


def test_solve_sci_batch_matches_fci_on_h2():
    """Test solve sci batch matches fci on h2."""
    h1, eri, e_core, norb, nelec, e_fci = _h2_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)

    results = solve_sci_batch([(ci_a, ci_b), (ci_a, ci_b)], h1, eri, norb, nelec, device="cpu")

    assert len(results) == 2
    for r in results:
        assert r.energy + e_core == pytest.approx(e_fci, abs=1e-9)
        occ_a, occ_b = r.orbital_occupancies
        assert occ_a.shape == (norb,)


def test_solver_instance_is_reusable_sci_solver():
    """A constructed TorchSCISolver is itself a valid sci_solver callable."""
    h1, eri, e_core, norb, nelec, e_fci = _h2_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)

    solver = make_torch_sci_solver(device="cpu")
    # solver(...) returns list[EigenStates]; ground state is at states[0]
    results = solver([(ci_a, ci_b)], h1, eri, norb, nelec)
    assert results[0].states[0].energy + e_core == pytest.approx(e_fci, abs=1e-9)
    # The instance accumulates stats across calls.
    assert solver.stats["n_subspaces"] == 1


def _h4_integrals(spacing=0.74):
    """RHF integrals for a 4-atom H chain in STO-3G and the exact FCI energy.

    H4 has a more correlated ground state than H2, which makes it a good guard
    against the engine's auto-enabled alpha-beta projection silently shifting
    the energy off the true variational minimum.
    """
    atoms = "; ".join(f"H 0 0 {i * spacing}" for i in range(4))
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    nelec = mol.nelec
    e_fci = fci.FCI(mf).kernel()[0]
    return h1, eri, e_core, norb, nelec, e_fci


def test_full_basis_reproduces_fci_on_h4_by_default():
    """Over a complete CI basis, the default solver must equal exact FCI.

    Regression guard: the public API defaults ``apply_ab_projection=False`` so
    it returns the bare variational minimum, matching qiskit-addon-sqd's
    PySCF-based solver. If the projection ever leaks back into the default, the
    H4 energy drifts ~1e-4 above FCI and this test fails.
    """
    h1, eri, e_core, norb, nelec, e_fci = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)  # complete space = exact FCI

    energy, _, _, _ = solve_fermion((ci_a, ci_b), hcore=h1, eri=eri, device="cpu")
    assert energy + e_core == pytest.approx(e_fci, abs=1e-9)


def test_ab_projection_is_available_as_opt_in():
    """Passing ``apply_ab_projection=True`` reaches the engine projection AND
    reproduces the exact singlet FCI energy.

    Regression guard for the interleaved-Wick / spin-block convention fix:
    over the full CI basis the α↔β-swap projection lives in the same physical
    singlet subspace as the closed-shell FCI ground state, so the projected
    minimum must equal the unprojected FCI to solver precision. Before the
    fix, the projector was written as ``(v + vᵀ)/2`` on the *interleaved-Wick*
    coefficient matrix — a sign-agnostic form only correct in the trivial
    ``τ ≡ 1`` case (``n_α ≤ 1``). For any n_α ≥ 2 the plain swap no longer
    commutes with H, and Davidson converged to a spurious "invariant"
    subspace of H a few mHa above the true singlet minimum (H4: +0.17 mHa,
    naphthalene 10e/10o: +5.5 mHa).
    """
    h1, eri, e_core, norb, nelec, e_fci = _h4_integrals()
    ci_a, ci_b = _full_ci_strings(norb, nelec)

    energy, _, _, _ = solve_fermion(
        (ci_a, ci_b), hcore=h1, eri=eri, device="cpu", apply_ab_projection=True
    )
    assert energy + e_core == pytest.approx(e_fci, abs=1e-9)


def test_compute_full_rdm_spin_summed_rdm2_matches_sci_state_on_open_shell():
    """The adapter's JSON-bound ``rdm2`` must equal ``SCIState.rdm(rank=2,
    spin_summed=True)`` of the same state, element-wise.

    Regression test: ``qiskit_adapter.py`` wrote
    ``Gaa + 2*Gab + Gbb`` as the spin-summed 2-RDM. That equals the correct
    ``Gaa + Gab + Gab.transpose(2,3,0,1) + Gbb`` only when ``Gab`` is
    symmetric under the ``(2,3,0,1)`` transpose, which an energy check alone
    cannot see (an 8-fold-symmetric ERI contraction makes the two forms
    agree) and which a spin-balanced (``n_a == n_b``) state satisfies only
    approximately. An open-shell triplet (``n_a != n_b``) breaks it exactly.
    """
    from pyscf import ao2mo, gto, scf
    from qiskit_addon_sqd.fermion import SCIState

    mol = gto.M(
        atom="C 0 0 0; H 0 0.94 0.6; H 0 -0.94 0.6",
        basis="sto-3g",
        spin=2,
        verbose=0,
    )
    mf = scf.ROHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec
    assert n_a != n_b, "need an open-shell (spin-imbalanced) state for this regression"

    ci_a, ci_b = _full_ci_strings(norb, (n_a, n_b))

    solver = make_torch_sci_solver(device="cpu", compute_full_rdm=True)
    result = solver([(ci_a, ci_b)], h1, eri, norb, (n_a, n_b))[0].states[0]

    sci_state = SCIState(
        amplitudes=result.sci_state.amplitudes,
        ci_strs_a=result.sci_state.ci_strs_a,
        ci_strs_b=result.sci_state.ci_strs_b,
        norb=norb,
        nelec=(n_a, n_b),
    )
    dm2_ref = sci_state.rdm(rank=2, spin_summed=True)

    np.testing.assert_allclose(result.rdm2, dm2_ref, atol=1e-9)

    # partial-trace identity: sum_r dm2[p,q,r,r] = (N-1) dm1[p,q]
    nel = n_a + n_b
    partial_trace = np.einsum("pqrr->pq", result.rdm2)
    np.testing.assert_allclose(partial_trace, (nel - 1) * result.rdm1, atol=1e-9)
