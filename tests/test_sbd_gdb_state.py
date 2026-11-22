"""The ``mode="gdb"`` state :class:`SBDSolver` returns, checked against PySCF's FCI.

``gdb_diag``'s energy is variational and stays right even when the returned state is
wrong, so these tests pin the eigenvector: it must cover the whole diagonalized basis
(not the carryover subset, and not a uniform stub), and on a full CI space it must equal
PySCF's FCI vector up to a global sign once its interleaved-Wick parity is mapped to
the spin-block convention.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from pyscf.fci import cistring

# Before `import sbd`: importing sci_solvers configures singleton MPI, without which sbd's
# MPI_Init hangs on an HPC node when this file runs on its own.
import sci_solvers  # noqa: F401

_H4 = "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"


def _h4_full_ci_pairs():
    """H4 / STO-3G integrals and the full CI space written as parallel pairs."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=_H4, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    norb = int(mol.nao)
    hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = (int(mol.nelec[0]), int(mol.nelec[1]))
    strs_a = np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64)
    pairs_a, pairs_b = np.repeat(strs_a, len(strs_b)), np.tile(strs_b, len(strs_a))
    return hcore, eri, norb, nelec, pairs_a, pairs_b


def _skip_without_sbd() -> None:
    pytest.importorskip("sbd", reason="the in-process `sbd` package is not installed")
    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")


def test_gdb_state_is_the_full_fci_vector() -> None:
    """The GDB state spans every input pair and reproduces PySCF's FCI vector."""
    _skip_without_sbd()
    from pyscf.fci import direct_spin1
    from torch_sci_solver.torch_impl.pairs import PairSCIState
    from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, pairs_a, pairs_b = _h4_full_ci_pairs()
    e_fci, fcivec = direct_spin1.kernel(hcore, eri, norb, nelec)

    result = (
        SBDSolver(symmetrize_spin=False, mode="gdb")
        .sci_solver_callable_eigen_states([(pairs_a, pairs_b)], hcore, eri, norb, nelec)[0]
        .states[0]
    )
    state = result.sci_state

    assert isinstance(state, PairSCIState)
    amplitudes = np.asarray(state.amplitudes)
    assert amplitudes.shape == (len(pairs_a),)
    assert set(zip(state.ci_strs_a.tolist(), state.ci_strs_b.tolist())) == set(
        zip(pairs_a.tolist(), pairs_b.tolist())
    )
    assert np.linalg.norm(amplitudes) == pytest.approx(1.0, abs=1e-8)
    assert result.energy == pytest.approx(e_fci, abs=1e-8)

    sigma = _inter_to_block_signs_pair(
        torch.as_tensor(state.ci_strs_a), torch.as_tensor(state.ci_strs_b), norb
    ).numpy()
    addr_a = cistring.strs2addr(norb, nelec[0], state.ci_strs_a)
    addr_b = cistring.strs2addr(norb, nelec[1], state.ci_strs_b)
    reference = fcivec[addr_a, addr_b]
    assert abs(np.dot(sigma * amplitudes, reference)) == pytest.approx(1.0, abs=1e-8)


def _gdb_energy_in_subprocess(bit_length: int) -> float:
    """H4 full CI through ``SBDSolver(mode="gdb")`` with ``bit_length``, in a fresh process.

    sbd allows one number of words per determinant per process ("gdb_diag was already
    called in this process with a different number of words per determinant"), so a
    ``bit_length`` that changes the word count cannot run after another GDB test.
    """
    code = (
        "import sys; sys.path.insert(0, sys.argv[2]); import test_sbd_gdb_state as t; "
        "from sci_solvers.energy_solvers.sbd_solver import SBDSolver; "
        "h, e, n, ne, a, b = t._h4_full_ci_pairs(); "
        "s = SBDSolver(symmetrize_spin=False, mode='gdb', "
        "sbd_config={'bit_length': int(sys.argv[1])}); "
        "r = s.sci_solver_callable_eigen_states([(a, b)], h, e, n, ne)[0].states[0]; "
        "assert len(r.sci_state.amplitudes) == len(a); print(repr(float(r.energy)))"
    )
    here = str(pathlib.Path(__file__).resolve().parent)
    result = subprocess.run(
        [sys.executable, "-c", code, str(bit_length), here],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return float(result.stdout.strip().splitlines()[-1])


def test_gdb_honors_sbd_config_bit_length() -> None:
    """A non-default ``bit_length`` packs determinants the way the engine reads them."""
    _skip_without_sbd()
    from pyscf.fci import direct_spin1

    hcore, eri, norb, nelec, _, _ = _h4_full_ci_pairs()
    e_fci, _ = direct_spin1.kernel(hcore, eri, norb, nelec)

    # 2 * norb = 8 interleaved bits over 4-bit words: two words per determinant,
    # where the default bit_length (20) would pack each into one. (An odd bit_length
    # over several words is refused: see test_gdb_refuses_odd_multiword_bit_length.)
    assert _gdb_energy_in_subprocess(4) == pytest.approx(e_fci, abs=1e-8)


@pytest.mark.parametrize(
    ("bit_length", "norb", "refused"),
    [(3, 4, True), (7, 6, True), (21, 11, True), (4, 4, False), (20, 11, False), (9, 4, False)],
)
def test_check_gdb_bit_length(bit_length: int, norb: int, refused: bool) -> None:
    """An odd ``bit_length`` is refused only when a determinant spans several words."""
    from sci_solvers.energy_solvers.sbd_solver import _check_gdb_bit_length

    if refused:
        with pytest.raises(ValueError, match="even bit_length"):
            _check_gdb_bit_length(bit_length, norb)
    else:
        _check_gdb_bit_length(bit_length, norb)


def test_gdb_refuses_odd_multiword_bit_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SBDSolver`` raises before ``gdb_diag``, instead of letting it crash."""
    _skip_without_sbd()
    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, pairs_a, pairs_b = _h4_full_ci_pairs()
    solver = SBDSolver(symmetrize_spin=False, mode="gdb", sbd_config={"bit_length": 3})
    backend = solver._sbd.get_backend(solver._device_config.device)

    def _not_called(*_args, **_kwargs):
        raise AssertionError("gdb_diag was called")

    monkeypatch.setattr(backend, "gdb_diag", _not_called)
    with pytest.raises(ValueError, match="even bit_length"):
        solver.sci_solver_callable_eigen_states([(pairs_a, pairs_b)], hcore, eri, norb, nelec)


def test_gdb_odd_single_word_bit_length() -> None:
    """An odd ``bit_length`` that holds the whole determinant in one word still works."""
    _skip_without_sbd()
    from pyscf.fci import direct_spin1

    from sci_solvers.energy_solvers.sbd_solver import SBDSolver

    hcore, eri, norb, nelec, pairs_a, pairs_b = _h4_full_ci_pairs()
    e_fci, _ = direct_spin1.kernel(hcore, eri, norb, nelec)
    # 2 * norb = 8 bits in one 9-bit word.
    solver = SBDSolver(symmetrize_spin=False, mode="gdb", sbd_config={"bit_length": 9})
    result = solver.sci_solver_callable_eigen_states([(pairs_a, pairs_b)], hcore, eri, norb, nelec)[
        0
    ].states[0]
    assert result.energy == pytest.approx(e_fci, abs=1e-8)
    assert len(result.sci_state.amplitudes) == len(pairs_a)
