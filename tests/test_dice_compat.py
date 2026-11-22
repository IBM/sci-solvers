"""Tests for ``_dice_compat``, the two additions on top of upstream qiskit-addon-dice-solver.

They run upstream's own ``solve_hci`` with the ``Dice`` subprocess and output reader
stubbed, so they need no ``Dice`` binary. They pin that the patched private functions
are the ones upstream calls, so a moved upstream pin that breaks the shim fails here.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from qiskit_addon_dice_solver import dice_solver as upstream

from sci_solvers.energy_solvers import _dice_compat

CI_STRS = (np.array([0b11]), np.array([0b11]))


@pytest.fixture
def captured_input(monkeypatch, tmp_path):
    """Run solve_hci without Dice; return a getter for the input.dat it wrote."""
    seen: dict[str, str] = {}

    def fake_call_dice(dice_dir, _mpirun_options):
        seen["input"] = (Path(dice_dir) / "input.dat").read_text()

    def fake_read(*_args, **_kwargs):
        return 0.0, None, (np.zeros(2), np.zeros(2))

    # Stub only the subprocess and output parsing; the input writer stays patched-real.
    monkeypatch.setattr(upstream, "_call_dice", fake_call_dice)
    monkeypatch.setattr(upstream, "_read_dice_outputs", fake_read)

    def run(**kwargs):
        _dice_compat.solve_hci(
            np.zeros((2, 2)),
            np.zeros((2, 2, 2, 2)),
            norb=2,
            nelec=(1, 1),
            ci_strs=CI_STRS,
            spin_sq=0.0,
            temp_dir=tmp_path,
            **kwargs,
        )
        return seen["input"]

    return run


def test_upstream_calls_the_patched_functions():
    """Importing the shim swaps in its writer and runner on the upstream module."""
    assert upstream._write_input_files is _dice_compat._write_input_files
    assert upstream._call_dice is _dice_compat._call_dice


def test_default_input_matches_upstream(captured_input):
    """Defaults leave upstream's davidsonTol line and add no loose tolerance."""
    text = captured_input()
    assert "davidsonTol 1e-5\n" in text
    assert "davidsonTolLoose" not in text


def test_davidson_tol_written(captured_input):
    """A non-default tight tolerance replaces upstream's hard-coded value."""
    text = captured_input(davidson_tol=1e-8)
    assert "davidsonTol 1e-08\n" in text
    assert "davidsonTol 1e-5\n" not in text


def test_davidson_tol_loose_written(captured_input):
    """An explicit loose tolerance gets its own line; the tight one appears once."""
    text = captured_input(davidson_tol=1e-3, davidson_tol_loose=1e-6)
    assert "davidsonTolLoose 1e-06\n" in text
    assert text.count("davidsonTol ") == 1
    assert "davidsonTol 0.001\n" in text


def test_tolerances_reset_after_call(captured_input):
    """A call's tolerances do not leak into the next call."""
    captured_input(davidson_tol=1e-8, davidson_tol_loose=1e-9)
    text = captured_input()
    assert "davidsonTol 1e-5\n" in text
    assert "davidsonTolLoose" not in text


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"DICE_BINARY": "/opt/x/Dice", "DICE_BIN_DIR": "/opt/y"}, "/opt/x/Dice"),
        ({"DICE_BIN_DIR": "/opt/y"}, "/opt/y/Dice"),
    ],
)
def test_resolve_dice_binary_env(monkeypatch, env, expected):
    """DICE_BINARY wins over DICE_BIN_DIR, which wins over PATH."""
    monkeypatch.delenv("DICE_BINARY", raising=False)
    monkeypatch.delenv("DICE_BIN_DIR", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _dice_compat.resolve_dice_binary() == expected


def test_resolve_dice_binary_path_then_packaged(monkeypatch, tmp_path):
    """Without env vars, ``Dice`` on PATH is used, else the packaged binary."""
    monkeypatch.delenv("DICE_BINARY", raising=False)
    monkeypatch.delenv("DICE_BIN_DIR", raising=False)
    dice = tmp_path / "Dice"
    dice.write_text("#!/bin/sh\n")
    dice.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _dice_compat.resolve_dice_binary() == str(dice)
    monkeypatch.setenv("PATH", "")
    assert _dice_compat.resolve_dice_binary() == str(Path(upstream.DICE_BIN) / "Dice")


def test_call_dice_runs_resolved_binary(monkeypatch, tmp_path):
    """The patched runner invokes the resolved binary under mpirun."""
    monkeypatch.setenv("DICE_BINARY", "/opt/x/Dice")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        (tmp_path / "spin1RDM.0.0.txt").write_text("")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(_dice_compat.subprocess, "run", fake_run)
    _dice_compat._call_dice(tmp_path, "-np 2")
    assert calls == [["mpirun", "-np", "2", "/opt/x/Dice"]]
