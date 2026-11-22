"""RDMs on the :class:`SBDSolver` results.

``compute_rdm2=True`` asks ``sbd`` itself (``do_rdm = 1``) for the 1- and 2-RDMs, which
``sbd.sbd_solver.assemble_rdms`` turns into PySCF-convention spin-summed tensors. On a
FULL CI space the ground state is unique, so its RDMs are gauge-invariant and can be
compared element-wise against PySCF's FCI ``make_rdm12`` — an independent reference.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import shutil

import numpy as np
import pytest
from pyscf.fci import cistring

_H4 = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"


def _h4_full_ci():
    """Integrals and complete CI string lists for linear H4 / STO-3G."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=_H4, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    norb = int(mol.nao)
    hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)
    return hcore, eri, norb, nelec, strs_a, strs_b


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_rdms_match_pyscf_full_ci(mode: str) -> None:
    """``compute_rdm2=True`` RDMs equal PySCF's FCI RDMs and reproduce the energy."""
    pytest.importorskip("sbd", reason="the in-process `sbd` package is not installed")
    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")

    from pyscf.fci import direct_spin1

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, strs_a, strs_b = _h4_full_ci()
    e_fci, fcivec = direct_spin1.kernel(hcore, eri, norb, nelec)
    ref_dm1, ref_dm2 = direct_spin1.make_rdm12(fcivec, norb, nelec)

    if mode == "gdb":
        # The full product written out as parallel (alpha_i, beta_i) pairs.
        strs_a, strs_b = np.repeat(strs_a, len(strs_b)), np.tile(strs_b, len(strs_a))

    solver = SBDSolver(symmetrize_spin=False, mode=mode, compute_rdm2=True)
    result = solver.sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)[
        0
    ].states[0]

    assert result.rdm1 is not None and result.rdm2 is not None
    np.testing.assert_allclose(result.rdm1, ref_dm1, atol=1e-6)
    np.testing.assert_allclose(result.rdm2, ref_dm2, atol=1e-6)
    e_rdm = np.einsum("pq,pq->", result.rdm1, hcore) + 0.5 * np.einsum(
        "pqrs,pqrs->", result.rdm2, eri
    )
    assert e_rdm == pytest.approx(e_fci, abs=1e-8)
    assert result.energy == pytest.approx(e_fci, abs=1e-8)


def test_rdms_absent_by_default() -> None:
    """Without ``compute_rdm2`` the results carry no RDMs (``do_rdm`` stays 0)."""
    pytest.importorskip("sbd", reason="the in-process `sbd` package is not installed")
    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, strs_a, strs_b = _h4_full_ci()
    result = (
        SBDSolver(symmetrize_spin=False, mode="tpb")
        .sci_solver_callable_eigen_states([(strs_a, strs_b)], hcore, eri, norb, nelec)[0]
        .states[0]
    )
    assert result.rdm1 is None and result.rdm2 is None
