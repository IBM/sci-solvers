"""GDB sanity checks on physical complex Hamiltonians built by PySCF.

The Hamiltonians come from ``src/sanity_checks/gdb_physical.py``, which builds them
from small molecules and cells with PySCF, so these tests read no local data file:

* an NH2 radical, ``(5, 4)``, in a randomly rotated complex orbital basis;
* a periodic H4 chain at a twisted k-point, whose integrals are complex, for closed and
  open shells.

On every GDB-capable complex solver, GDB must reproduce TPB's energy, RDMs and ground
state up to a global phase, within ``gdb_physical.TOLERANCES``; with fulqrum installed,
each ``fulqrum_mode`` must also reproduce ``csr``'s ground states. The YAML case reader
(``gdb_cases``) is checked on synthetic input.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from test_gdb_pair_mode import (
    COMPLEX_CAPABLE_GDB_SOLVERS,
    _skip_if_unavailable,
    _skip_slepc_real_scalar,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sanity_checks"))
from gdb_physical import (  # noqa: E402
    compare_gdb_tpb,
    failures,
    rotated_radical,
    twisted_hydrogen_chain,
)


@pytest.fixture(scope="module")
def radical():
    """The rotated NH2 radical, built once per module (the PySCF FCI is the slow part)."""
    return rotated_radical()


@pytest.mark.parametrize("name", COMPLEX_CAPABLE_GDB_SOLVERS)
def test_rotated_radical_gdb_matches_tpb(name, radical):
    """GDB on the rotated open-shell radical reproduces TPB."""
    _skip_if_unavailable(name)
    _skip_slepc_real_scalar(name, complex_problem=True)
    assert np.abs(radical.eri.imag).max() > 0.1
    assert failures(compare_gdb_tpb(radical, name)[0]) == []


@pytest.mark.parametrize("nelec", [(2, 2), (2, 1), (3, 1)])
@pytest.mark.parametrize("name", COMPLEX_CAPABLE_GDB_SOLVERS)
def test_twisted_chain_gdb_matches_tpb(name, nelec):
    """GDB on a twisted-boundary H4 chain reproduces TPB, closed and open shell."""
    _skip_if_unavailable(name)
    _skip_slepc_real_scalar(name, complex_problem=True)
    ham = twisted_hydrogen_chain(nelec)
    assert np.abs(ham.eri.imag).max() > 0.1
    assert failures(compare_gdb_tpb(ham, name)[0]) == []


@pytest.mark.skipif(importlib.util.find_spec("fulqrum") is None, reason="fulqrum not installed")
@pytest.mark.parametrize("mode", ["csr_fast", "matrix_free"])
@pytest.mark.parametrize("name", ["arpack", "davidson", "lobpcg"])
def test_fulqrum_modes_match_csr(name, mode, radical):
    """Every fulqrum mode passes GDB vs TPB on the radical and reproduces ``csr``'s states."""
    csr_diffs, csr = compare_gdb_tpb(radical, f"{name}@fulqrum")
    assert failures(csr_diffs) == []
    diffs, _ = compare_gdb_tpb(radical, f"{name}@fulqrum:{mode}", reference=csr)
    assert "mode_overlap" in diffs
    assert failures(diffs) == []


def test_suite_yaml_discovery(tmp_path):
    """A suite file unions ``datasets_dir`` with ``datasets`` by name, resolving paths."""
    from gdb_cases import discover

    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "a.yaml").write_text("hamiltonian: {fcidump: ../data/a.fcidump}\n")
    (tmp_path / "cases" / "b.yaml").write_text("hamiltonian: {builtin: real_molecule}\n")
    (tmp_path / "suite.yaml").write_text(
        "defaults: {solvers: [arpack, lobpcg]}\n"
        "datasets_dir: cases\n"
        "datasets:\n"
        "  - {name: b, solvers: [davidson], expected: fail}\n"
        "  - {name: c, case: cases/a.yaml}\n"
    )
    cases = {c.name: c for c in discover([tmp_path / "suite.yaml"])}
    assert sorted(cases) == ["a", "b", "c"]
    assert cases["a"].solvers == ["arpack", "lobpcg"]
    assert cases["a"].spec["hamiltonian"]["fcidump"] == str(tmp_path / "data" / "a.fcidump")
    assert cases["b"].solvers == ["davidson"] and cases["b"].expected_fail
    assert cases["c"].spec["hamiltonian"] == cases["a"].spec["hamiltonian"]
    assert [c.name for c in discover([tmp_path / "cases"], ["primme"])] == ["a", "b"]
    assert all(c.solvers == ["primme"] for c in discover([tmp_path / "cases"], ["primme"]))
