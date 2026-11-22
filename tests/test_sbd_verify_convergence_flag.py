"""``SBDSolver(verify_convergence=...)`` toggles the independent residual check.

Uses a stub of the ``sbd`` backend, so it runs without the compiled package.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import inspect
import types

import numpy as np
import pytest

from sci_solvers.energy_solvers.sbd_solver import SBDSolver


def _make_solver(verify: bool):
    solver = SBDSolver.__new__(SBDSolver)  # skip __init__: no sbd import / MPI
    solver.mode = "tpb"
    solver.sbd_config = None
    solver.compute_rdm2 = False
    solver.temp_dir = None
    solver.clean_temp_dir = True
    solver._device_config = None
    solver._fcidump_path = None
    solver.verify_convergence = verify
    calls = {"verify": 0}
    fake_result = types.SimpleNamespace(energy=-1.0)
    solver._sbd = types.SimpleNamespace(
        sbd_solver=types.SimpleNamespace(solve_sci_batch=lambda *a, **k: [fake_result])
    )

    def fake_with_recomputed(result, *_args, **_kwargs):
        calls["verify"] += 1
        return result

    solver._with_recomputed_convergence = fake_with_recomputed
    return solver, calls


@pytest.fixture(autouse=True)
def _no_base_checks(monkeypatch):
    # The base-class validation is irrelevant here.
    monkeypatch.setattr(
        "sci_solvers.energy_solvers.sbd_solver.EnergyCalculatorBase.sci_solver_callable_eigen_states",
        lambda self, *a, **k: None,
    )


def _call(solver):
    batches = [(np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64))]
    return solver.sci_solver_callable_eigen_states(
        batches, np.eye(2), np.zeros((2, 2, 2, 2)), 2, (1, 1)
    )


def test_verification_is_off_by_default():
    """The costly residual check must be opt-in."""
    assert inspect.signature(SBDSolver.__init__).parameters["verify_convergence"].default is False


def test_verification_skipped_when_disabled():
    """verify_convergence=False: the residual check is not invoked."""
    solver, calls = _make_solver(False)
    _call(solver)
    assert calls["verify"] == 0


def test_verification_runs_when_enabled():
    """verify_convergence=True: the residual check runs once per batch."""
    solver, calls = _make_solver(True)
    _call(solver)
    assert calls["verify"] == 1
