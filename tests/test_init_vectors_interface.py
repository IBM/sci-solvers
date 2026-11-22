"""The per-call ``init_vectors`` interface across every registered solver.

``torch_sci`` honors a Davidson start vector per batch (its own behaviour is covered
in ``packages/torch-sci-solver/tests/test_init_vectors.py``). Every other solver must
accept ``init_vectors=None`` (or all-``None`` entries) as a cold solve, and raise
``NotImplementedError`` when an actual vector is passed rather than silently ignoring
it -- otherwise a caller timing a warm start would get a cold solve and a misleading
number. ``compute_energy`` must re-index the vectors when it sorts the strings.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("torch_sci_solver")

from pyscf import ao2mo, gto, scf  # noqa: E402

from sci_solvers.energy_calculation_base import reject_init_vectors  # noqa: E402
from sci_solvers.energy_solvers.factory import (  # noqa: E402
    AVAILABLE_SOLVERS,
    create_solver,
    is_general_algebra,
    solver_class,
    supports_init_vectors,
)
from sci_solvers.hamiltonian import HamiltonianData  # noqa: E402

UNSUPPORTED = [n for n in AVAILABLE_SOLVERS if n != "torch_sci"]
# Solvers cheap enough to build and run on CPU without optional extras or binaries.
CPU_SOLVERS = ["pyscf", "davidson", "arpack", "lobpcg", "truncated_davidson"]


def _h4():
    mol = gto.M(atom="; ".join(f"H 0 0 {i * 0.9}" for i in range(4)), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    strs = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), 2)),
        dtype=np.int64,
    )
    return h1, eri, norb, mol.nelec, strs


H1, ERI, NORB, NELEC, STRS = _h4()


def test_only_torch_sci_supports_init_vectors() -> None:
    """The capability flag names exactly the solvers that honor the vector."""
    assert [n for n in AVAILABLE_SOLVERS if supports_init_vectors(n)] == ["torch_sci"]


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_solver_raises_on_a_vector(name: str) -> None:
    """Raised first thing, before the solver touches its own state or any binary."""
    if is_general_algebra(name):
        # Constructible without their optional backends, which load only at solve
        # time; a real instance also exercises the SpinFilterMixin forwarding.
        solver = create_solver(name, symmetrize_spin=False)
    else:
        # The native solvers need binaries / devices to construct; skipping __init__
        # checks the guard comes before anything that would need them.
        cls = solver_class(name)
        solver = cls.__new__(cls)
    vec = np.ones(len(STRS) ** 2)
    with pytest.raises(NotImplementedError, match="init_vectors.*'torch_sci'"):
        solver.sci_solver_callable_eigen_states(
            [(STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[None, vec]
        )


@pytest.mark.parametrize("name", CPU_SOLVERS)
@pytest.mark.parametrize("init_vectors", [None, [None]], ids=["None", "all-None"])
def test_unsupported_solver_accepts_cold_request(name: str, init_vectors) -> None:
    """``None`` is a cold solve everywhere; PySCF used to forward it and crash."""
    solver = create_solver(name, symmetrize_spin=False)
    cold = solver.sci_solver_callable([(STRS, STRS)], H1, ERI, NORB, NELEC)
    res = solver.sci_solver_callable(
        [(STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=init_vectors
    )
    assert res[0].energy == pytest.approx(cold[0].energy, abs=1e-8)


def test_reject_init_vectors_pops_the_kwarg() -> None:
    """The guard removes the kwarg so it never reaches an inner routine like PySCF's."""
    kwargs = {"init_vectors": [None], "tol": 1e-6}
    reject_init_vectors("X", kwargs)
    assert kwargs == {"tol": 1e-6}


def test_torch_sci_wrapper_forwards_init_vectors() -> None:
    """The sci_solvers wrapper passes the vectors through to the engine."""
    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu", davidson_block_size=4)
    cold = solver.sci_solver_callable_eigen_states([(STRS, STRS)], H1, ERI, NORB, NELEC)
    vec = cold[0].states[0].sci_state.amplitudes.ravel()
    warm = solver.sci_solver_callable_eigen_states(
        [(STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[vec]
    )
    perf = solver._solver.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert warm[0].states[0].energy == pytest.approx(cold[0].states[0].energy, abs=1e-9)


def test_compute_energy_reindexes_init_vectors_to_sorted_strings() -> None:
    """compute_energy sorts/dedups each string list; the vector must follow it."""
    ham = HamiltonianData(
        hcore=H1, eri=ERI, e_nuc=0.0, num_orbitals=NORB, num_elec_a=NELEC[0], num_elec_b=NELEC[1]
    )
    solver = create_solver(
        "torch_sci", symmetrize_spin=False, device="cpu", davidson_block_size=4, hamiltonian=ham
    )
    (cold,) = solver.compute_energy(ci_strs_batches=[(STRS, STRS)])
    grid = cold.sci_state.amplitudes
    rng = np.random.default_rng(0)
    pa, pb = rng.permutation(len(STRS)), rng.permutation(len(STRS))
    strs_a = np.concatenate([STRS[pa], STRS[pa[:1]]])  # one duplicate, ignored
    vec = np.concatenate([grid[np.ix_(pa, pb)], np.full((1, len(STRS)), 7.0)]).ravel()
    (warm,) = solver.compute_energy(ci_strs_batches=[(strs_a, STRS[pb])], init_vectors=[vec])
    perf = solver._solver.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["warm"] is True
    assert perf["n_matvecs"] == 1
    assert warm.energy == pytest.approx(cold.energy, abs=1e-9)


def test_compute_energy_rejects_wrong_init_vectors_length() -> None:
    """A short list must raise, not be silently truncated by the re-indexing ``zip``."""
    ham = HamiltonianData(
        hcore=H1, eri=ERI, e_nuc=0.0, num_orbitals=NORB, num_elec_a=NELEC[0], num_elec_b=NELEC[1]
    )
    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu", hamiltonian=ham)
    with pytest.raises(ValueError, match="init_vectors has length 2"):
        solver.compute_energy(ci_strs_batches=[(STRS, STRS)], init_vectors=[None, None])


# --------------------------------------------------------------------------- #
# project_init_vector and last_call_subspace_stats: the caller-facing helpers   #
# --------------------------------------------------------------------------- #
def _reference_project_state(prev, strs_a, strs_b):
    """An independent TPB projection on sorted strings, written without the engine."""
    c = np.asarray(prev.amplitudes)
    ia_p = np.isin(prev.ci_strs_a, strs_a)
    ib_p = np.isin(prev.ci_strs_b, strs_b)
    ia_n = np.searchsorted(strs_a, prev.ci_strs_a[ia_p])
    ib_n = np.searchsorted(strs_b, prev.ci_strs_b[ib_p])
    out = np.zeros((len(strs_a), len(strs_b)), dtype=c.dtype)
    out[np.ix_(ia_n, ib_n)] = c[np.ix_(ia_p, ib_p)]
    n = np.linalg.norm(out)
    return None if n < 1e-8 else (out / n).reshape(-1)


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_project_init_vector_warm_starts_each_mode(mode: str) -> None:
    """The projected previous state is a valid start in every mode, signs included.

    A converged state projected onto the same basis must converge on the first
    residual check: a wrong sign convention (e.g. spin-block fed to GDB, which takes
    interleaved-Wick) would not. A disjoint basis gives None (cold start).
    """
    from sci_solvers.energy_calculation_base import project_init_vector

    solver = create_solver(
        "torch_sci", symmetrize_spin=False, device="cpu", davidson_block_size=4, mode=mode
    )
    if mode == "tpb":
        batch = (STRS, STRS)
    else:
        batch = (np.repeat(STRS, len(STRS)), np.tile(STRS, len(STRS)))
    (prev,) = solver.sci_solver_callable_eigen_states([batch], H1, ERI, NORB, NELEC)
    vec = project_init_vector(prev.states[0].sci_state, batch, mode)
    solver.sci_solver_callable_eigen_states([batch], H1, ERI, NORB, NELEC, init_vectors=[vec])
    (stats,) = solver.last_call_subspace_stats()
    assert stats["warm"] is True
    assert stats["n_matvecs"] == 1

    empty = np.array([], dtype=np.int64)
    disjoint = (empty, empty)
    assert project_init_vector(prev.states[0].sci_state, disjoint, mode) is None


def test_project_init_vector_matches_reference_projection() -> None:
    """On TPB it matches an independent projection exactly, on a changed basis."""
    from sci_solvers.energy_calculation_base import project_init_vector

    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu", davidson_block_size=4)
    old, new = STRS[:5], STRS[1:]  # drops one string, adds one
    (prev,) = solver.sci_solver_callable_eigen_states([(old, old)], H1, ERI, NORB, NELEC)
    state = prev.states[0].sci_state
    np.testing.assert_array_equal(
        project_init_vector(state, (new, new)), _reference_project_state(state, new, new)
    )


def test_last_call_subspace_stats_is_not_stale_after_spinless() -> None:
    """The spinless path keeps no per-batch log: report nothing, not the last spatial call."""
    import torch

    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu", dtype=torch.complex128)
    assert solver.last_call_subspace_stats() == []
    h1c, eric = H1.astype(np.complex128), ERI.astype(np.complex128)
    solver.sci_solver_callable_eigen_states([(STRS, STRS)], h1c, eric, NORB, NELEC)
    assert len(solver.last_call_subspace_stats()) == 1
    nmo, n = 6, 2
    strs = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(nmo), n)),
        dtype=np.int64,
    )
    rng = np.random.default_rng(0)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    solver.sci_solver_callable_eigen_states(
        [strs], hc + hc.conj().T, np.zeros((nmo,) * 4, dtype=np.complex128), nmo, n
    )
    assert solver.last_call_subspace_stats() == []


def test_compute_energy_rejects_vector_sized_for_deduplicated_strings() -> None:
    """A vector sized for the deduplicated grid must not slip through compute_energy.

    compute_energy deduplicates the strings before the solver sees them, so a vector
    the length of the deduplicated grid used to pass the solver's own check and be
    read under the wrong indexing. It is validated against the strings as passed.
    """
    ham = HamiltonianData(
        hcore=H1, eri=ERI, e_nuc=0.0, num_orbitals=NORB, num_elec_a=NELEC[0], num_elec_b=NELEC[1]
    )
    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu", hamiltonian=ham)
    strs_a = np.concatenate([STRS, STRS[:1]])  # one duplicate: 7 as passed, 6 unique
    dedup_sized = np.ones(len(STRS) * len(STRS))
    with pytest.raises(ValueError, match=rf"expected {len(strs_a) * len(STRS)} "):
        solver.compute_energy(ci_strs_batches=[(strs_a, STRS)], init_vectors=[dedup_sized])


@pytest.mark.parametrize("mode", ["tpb", "gdb", "spinless"])
def test_project_init_vector_empty_previous_state_is_cold(mode: str) -> None:
    """An empty previous basis shares nothing: a cold start, not an IndexError."""
    from types import SimpleNamespace

    from sci_solvers.energy_calculation_base import project_init_vector

    empty = np.array([], dtype=np.int64)
    if mode == "spinless":
        prev = SimpleNamespace(amplitudes=np.zeros(0), ci_strs=empty)
        new: Any = np.array([3])
    else:
        shape = (0, 0) if mode == "tpb" else (0,)
        prev = SimpleNamespace(amplitudes=np.zeros(shape), ci_strs_a=empty, ci_strs_b=empty)
        new = (np.array([3]), np.array([3]))
    assert project_init_vector(prev, new, "gdb" if mode == "gdb" else "tpb") is None


def test_project_init_vector_duplicate_previous_strings_take_first() -> None:
    """Duplicates in a (hand-built) previous state take the first occurrence."""
    from types import SimpleNamespace

    from sci_solvers.energy_calculation_base import project_init_vector

    prev = SimpleNamespace(
        amplitudes=np.array([0.6, 9.0, 0.8]),
        ci_strs_a=np.array([3, 3, 5]),
        ci_strs_b=np.array([3, 3, 5]),
    )
    vec = project_init_vector(prev, (np.array([5, 3]), np.array([5, 3])), "gdb")
    np.testing.assert_allclose(vec, [0.8, 0.6])


def test_last_call_subspace_stats_is_not_stale_after_a_raising_call() -> None:
    """A call that raises leaves no log entry: report nothing, not the previous call."""
    solver = create_solver("torch_sci", symmetrize_spin=False, device="cpu")
    solver.sci_solver_callable_eigen_states([(STRS, STRS)], H1, ERI, NORB, NELEC)
    assert len(solver.last_call_subspace_stats()) == 1
    with pytest.raises(ValueError, match="init_vectors"):
        solver.sci_solver_callable_eigen_states(
            [(STRS, STRS)], H1, ERI, NORB, NELEC, init_vectors=[np.ones(3)]
        )
    assert solver.last_call_subspace_stats() == []
