"""Tests for the convergence signal surfaced by the pyscf and dice solvers.

The Dice fixtures use the real output format of Dice with mock values
(energies, residuals, determinant counts, timings); the tests check structure, not values.

The pyscf tests, by contrast, run a live solve on a deliberately small toy system
(N2/STO-3G, 14400 determinants). A production-scale problem — a subspace of ~10M
determinants — does not converge in a test-suite timeframe: a
pipeline run of it was still going after four days of wall clock and had to be
killed. The convergence *mechanism* is system-independent (PySCF sets the same flag
via ``lib.davidson1`` regardless of subspace size), so a toy system that finishes in
seconds exercises exactly the same code path.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest
from qiskit_addon_sqd.fermion import SCIResult

from sci_solvers.eigen_states import EigenStates, SCIResultWithConvergence

# --------------------------------------------------------------------------- #
# The shared type
# --------------------------------------------------------------------------- #


def test_state_defaults_to_none():
    """A result built without an explicit state reads as unknown, not converged."""
    r = SCIResultWithConvergence(energy=-1.0, sci_state=None, orbital_occupancies=None)
    assert r.state is None


def test_plain_sciresult_degrades_to_none():
    """The getattr contract: solvers that know nothing keep returning SCIResult."""
    r = SCIResult(energy=-1.0, sci_state=None, orbital_occupancies=None)
    assert getattr(r, "state", None) is None


def test_is_a_sciresult():
    """Subclassing means existing consumers keep working unchanged."""
    r = SCIResultWithConvergence(energy=-1.0, sci_state=None, orbital_occupancies=None)
    assert isinstance(r, SCIResult)
    assert EigenStates(states=[r]).states[0].energy == -1.0


# --------------------------------------------------------------------------- #
# pyscf convergence capture
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def n2_subspace():
    """N2/STO-3G full CAS (14400 dets) — a toy stand-in for the real target.

    Module-scoped because the SCF is reused across every pyscf test below. See
    the module docstring for why this is a toy system.
    """
    from pyscf import ao2mo, gto, scf
    from pyscf.fci import cistring

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), norb)
    nelec = (mol.nelec[0], mol.nelec[1])
    strs = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    return hcore, eri, norb, nelec, (strs, strs)


def _solve(n2_subspace, **kw):
    from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver

    hcore, eri, norb, nelec, batch = n2_subspace
    solver = PySCFSolver(symmetrize_spin=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        states = solver.sci_solver_callable_eigen_states([batch], hcore, eri, norb, nelec, **kw)
    return states[0].states[0], caught


def test_pyscf_reports_converged(n2_subspace):
    """A generous max_cycle converges, and does so without warning."""
    result, caught = _solve(n2_subspace, max_cycle=200)
    assert result.state == "converged"
    assert not caught


def test_pyscf_reports_not_converged(n2_subspace):
    """max_cycle=1 cannot possibly converge; assert we say so, and warn."""
    result, caught = _solve(n2_subspace, max_cycle=1)
    assert result.state == "not_converged"
    assert any("did not converge" in str(w.message) for w in caught)


def test_pyscf_not_converged_energy_is_worse(n2_subspace):
    """The flag tracks something real: the unconverged energy is materially higher."""
    good, _ = _solve(n2_subspace, max_cycle=200)
    bad, _ = _solve(n2_subspace, max_cycle=1)
    assert bad.energy > good.energy + 0.1


def test_pyscf_preserves_result_payload(n2_subspace):
    """Rebuilding the frozen dataclass must not drop any field."""
    result, _ = _solve(n2_subspace, max_cycle=200)
    assert result.energy == pytest.approx(-131.2265618430, abs=1e-8)
    assert result.sci_state is not None
    assert result.orbital_occupancies is not None


def test_pyscf_conversion_keeps_every_sciresult_field(monkeypatch):
    """The SCIResult -> SCIResultWithConvergence rebuild must carry every field through.

    Driven by ``dataclasses.fields`` so it covers whatever the installed
    qiskit-addon-sqd defines (rdm1/rdm2, and ``carryover`` from 0.14 on) without
    passing ``carryover=`` to an older ``SCIResult`` that lacks the field.
    """
    import sci_solvers.energy_solvers.pyscf_solver as mod

    payload = {f.name: object() for f in dataclasses.fields(SCIResult)}
    monkeypatch.setattr(mod, "solve_sci", lambda *_args, **_kw: SCIResult(**payload))

    strs = np.array([0b0011], dtype=np.int64)
    result = mod.PySCFSolver(symmetrize_spin=False)._solve_one_with_convergence(
        (strs, strs),
        np.zeros((4, 4)),
        np.zeros((4,) * 4),
        norb=4,
        nelec=(2, 2),
        spin_sq=None,
        max_cycle=1,
    )
    assert isinstance(result, SCIResultWithConvergence)
    for name, value in payload.items():
        assert getattr(result, name) is value, name


def test_pyscf_restores_patched_class(n2_subspace):
    """The monkey-patch must not leak past the call."""
    from pyscf.fci import selected_ci

    before = selected_ci.SelectedCI
    _solve(n2_subspace, max_cycle=200)
    assert selected_ci.SelectedCI is before


def test_pyscf_restores_patched_class_on_exception(n2_subspace):
    """The `finally` path: a raising solve must still restore the class."""
    from pyscf.fci import selected_ci

    from sci_solvers.energy_solvers.pyscf_solver import PySCFSolver

    hcore, eri, norb, nelec, batch = n2_subspace
    before = selected_ci.SelectedCI
    solver = PySCFSolver(symmetrize_spin=False)
    with pytest.raises(Exception):
        # A bogus kwarg makes solve_sci raise partway through.
        solver.sci_solver_callable_eigen_states(
            [batch], hcore, eri, norb, nelec, not_a_real_kwarg=object()
        )
    assert selected_ci.SelectedCI is before


def test_pyscf_spin_projected_path_still_reports(n2_subspace):
    """fix_spin_ decorates in place, so the captured instance is still the right one."""
    result, _ = _solve(n2_subspace, max_cycle=200, spin_sq=0.0)
    assert result.state in ("converged", "not_converged")


# --------------------------------------------------------------------------- #
# Dice log parsing
# --------------------------------------------------------------------------- #

# Dice log format (synthetic values). The bare "2048" lines are LOAD-BEARING: real
# logs carry hundreds of bare integer lines (per-batch determinant counts) ahead of the
# iteration table, and an earlier version of the row regex used ``\s`` between columns,
# which — with re.M — matched down across these lines and reported the integer as
# #Davidson instead of 3, flipping every real batch to not_converged. Do not trim them
# from this fixture.
DICE_CONVERGED = """\
Reading integrals
2048
2048
2048
2048
2048
Iter Root       Eps1   #Var. Det.               Energy   #Davidson     Time(s)
    0    0    2.15e+09     4194304      -123.4567871234          3      166.72
 Performing final tight davidson with tol: 1e-05
 Exiting variational iterations

Calculating 1-RDM
"""

# Same run, but Davidson reached Dice's compile-time per-root ceiling.
DICE_AT_CAP = """\
Iter Root       Eps1   #Var. Det.               Energy   #Davidson     Time(s)
    0    0    2.15e+09     4194304      -123.4567871234        800      166.72
 Exiting variational iterations
"""

# Dice died during the variational stage: no exit marker.
DICE_TRUNCATED = """\
Iter Root       Eps1   #Var. Det.               Energy   #Davidson     Time(s)
    0    0    2.15e+09     4194304      -123.4567871234          3      166.72
"""

# The real logs all end like this — an MPI abort in the RDM stage, AFTER the
# energy and the exit marker. Not a diagonalization failure.
DICE_MPI_ABORT_AFTER_ENERGY = """\
Iter Root       Eps1   #Var. Det.               Energy   #Davidson     Time(s)
    0    0    2.15e+09     4194304      -123.4567871234          3      166.72
 Exiting variational iterations

Calculating 2-RDM
Error here
rank 93 in job 1  exiting improperly. There may be more information above.
"""


def _parse_dice(text: str | None, tmp_path):
    """Run DICESolver._parse_convergence against a synthetic log tree.

    Called unbound with a stub self so no Dice binary is needed (the real
    __init__ validates the executable up front).
    """
    from sci_solvers.energy_solvers.dice_solver import DICESolver

    if text is not None:
        cli_dir = tmp_path / "dice_cli_files_testtest"
        cli_dir.mkdir()
        (cli_dir / "dice_solver_logfile.log").write_text(text)

    class _Stub:
        verbose = False

    return DICESolver._parse_convergence(_Stub(), tmp_path, 0, 1)


def test_dice_converged_log(tmp_path):
    """A complete log with a low Davidson count reads as converged."""
    assert _parse_dice(DICE_CONVERGED, tmp_path) == "converged"


def test_dice_ignores_bare_integer_lines(tmp_path):
    """Bare-integer lines above the table must not be read as Davidson counts.

    Regression pin: those lines once inflated the count to the bare integer (> the 800
    cap), flipping every real batch to not_converged. 2048 is a plausible real value,
    so this asserts the verdict rather than the number.
    """
    assert _parse_dice(DICE_CONVERGED, tmp_path) == "converged"


def test_dice_many_bare_integer_lines(tmp_path):
    """Same hazard at the real logs' scale: 256 such lines, still converged."""
    header_at = DICE_CONVERGED.index("Iter Root")
    text = "Reading integrals\n" + "2048\n" * 256 + DICE_CONVERGED[header_at:]
    assert _parse_dice(text, tmp_path) == "converged"


def test_dice_at_iteration_cap_is_not_converged(tmp_path):
    """Hitting the compile-time ceiling means it stopped on count, not residual."""
    assert _parse_dice(DICE_AT_CAP, tmp_path) == "not_converged"


def test_dice_missing_exit_marker_is_not_converged(tmp_path):
    """Dice died mid-variational-stage, so the energy is not trustworthy."""
    assert _parse_dice(DICE_TRUNCATED, tmp_path) == "not_converged"


def test_dice_mpi_abort_after_energy_still_converged(tmp_path):
    """The RDM-stage abort is not a diagonalization failure."""
    assert _parse_dice(DICE_MPI_ABORT_AFTER_ENERGY, tmp_path) == "converged"


def test_dice_no_logfile_is_unknown(tmp_path):
    """Missing log means unknown, never converged."""
    assert _parse_dice(None, tmp_path) is None


def test_dice_header_without_rows_is_unknown(tmp_path):
    """Header but no parseable row: we know nothing, so say nothing."""
    text = "Iter Root       Eps1   #Var. Det.               Energy   #Davidson     Time(s)\n"
    text += " Exiting variational iterations\n"
    assert _parse_dice(text, tmp_path) is None


def test_dice_reads_log_before_cleanup(monkeypatch, tmp_path):
    """clean_temp_dir=True must still report convergence.

    The wrapper's own cleanup would delete the logfile before we could read it,
    so the solver passes clean_temp_dir=False down to solve_hci and removes the
    directory itself afterwards. This pins that ordering: move the cleanup ahead
    of the parse and Dice silently degrades to always-unknown.
    """
    import sci_solvers.energy_solvers.dice_solver as mod

    def fake_solve_hci(*_args, **kwargs):
        # Stand in for Dice: write the log where the real wrapper would, and
        # assert the solver did not ask the wrapper to clean up.
        assert kwargs["clean_temp_dir"] is False
        cli_dir = Path(kwargs["temp_dir"]) / "dice_cli_files_fake"
        cli_dir.mkdir(parents=True)
        (cli_dir / "dice_solver_logfile.log").write_text(DICE_CONVERGED)
        return -123.4567871234, None, (np.zeros(4), np.zeros(4))

    monkeypatch.setattr(mod, "solve_hci", fake_solve_hci)

    solver = mod.DICESolver.__new__(mod.DICESolver)  # skip binary validation
    solver.verbose = False
    solver.temp_dir = str(tmp_path)
    solver.clean_temp_dir = True
    solver.mpirun_options = None
    solver.select_cutoff = 2147483647
    solver.energy_tol = 1e-10
    solver.davidson_tolerance = 1e-5
    solver.davidson_tolerance_loose = None
    solver.max_hci_iters = 1
    solver.fci = False
    solver.nroots = 1

    strs = np.array([0b0011], dtype=np.int64)
    results, _, _ = solver._solve_one_batch(
        (strs, strs), np.zeros((4, 4)), np.zeros((4,) * 4), 4, (1, 1), 0, 1
    )

    # _solve_one_batch now returns one SCIResultWithConvergence per root.
    assert len(results) == 1
    assert results[0].state == "converged"
    # ...and the directory really is gone afterwards.
    assert not list(tmp_path.glob("dice_batch_*"))
