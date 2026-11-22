"""Tests for the general-algebra solvers, the reduction base, and the spin mixin.

Replaces the former adapter / ``MultirootSCISolver`` suite: the math backends are
now standalone solvers on ``GeneralAlgebraSolverBase``, spin projection is the
composable ``SpinFilterMixin``, and there is no adapter registry. Also keeps the
adapter-independent ``EigenProblem`` / ``SymmetrizedEigenProblem`` math tests and
the native ``TruncatedDavidsonSolver`` multiroot coverage. EVSL lives in
``test_evsl.py``.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import math
from itertools import combinations

import numpy as np
import pytest
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from sci_solvers.eigen_problem import EigenProblem, SymmetrizedEigenProblem
from sci_solvers.energy_calculation_base import EnergyCalculatorBase
from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, create_solver
from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase
from sci_solvers.energy_solvers.general_algebra_solvers import (
    ArpackSolver,
    LobpcgSolver,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin, spin_square_of_vector
from sci_solvers.energy_solvers.truncated_davidson_solver import TruncatedDavidsonSolver
from sci_solvers.hamiltonian import HamiltonianData

NROOTS = 3
ATOL_ENERGY = 1e-6
ATOL_RESIDUAL = 1e-4
SPIN_SQ_TARGET = 0.0  # singlet

# General-algebra solvers exercised by the shared sweeps. ``evsl`` is excluded
# (interval-based, costly DOS pre-pass) and covered in test_evsl.py; ``dice`` and
# ``torch_sci`` are native solvers now, not general-algebra ones.
SHARED_SOLVERS = ["arpack", "lobpcg", "primme", "slepc", "pyscf_davidson"]


def _skip_if_unavailable(name: str) -> None:
    """Skip when a solver's optional backend is missing (arpack/lobpcg/pyscf_davidson always run)."""
    if name == "slepc":
        try:
            from slepc4py import SLEPc  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"slepc4py unavailable: {exc}")
    elif name == "primme":
        pytest.importorskip("primme")


def _solver(name: str, **kwargs):
    """Construct a general-algebra solver by name via the factory (with the skip gate)."""
    _skip_if_unavailable(name)
    return create_solver(name, **kwargs)


def _all_strings(norb: int, nelec: int) -> np.ndarray:
    return np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(norb), nelec)),
        dtype=np.int64,
    )


def _random_system(seed: int, norb: int, na: int, nb: int):
    """A random symmetric hcore + 8-fold-symmetric eri with full CI strings."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    x = rng.standard_normal((norb,) * 4)
    eri = np.zeros_like(x)
    for perm in [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ]:
        eri += x.transpose(perm)
    eri /= 8.0
    return hcore, eri, _all_strings(norb, na), _all_strings(norb, nb)


@pytest.fixture(scope="module")
def small_problem():
    """A small closed-shell system, batch inputs, and the dense reference spectrum."""
    norb, na, nb = 6, 3, 3
    hcore, eri, sa, sb = _random_system(11, norb, na, nb)
    problem = EigenProblem(sa, sb, hcore, eri, norb, (na, nb))
    ref = np.linalg.eigvalsh(problem.sparse().toarray())[:NROOTS]
    return problem, hcore, eri, sa, sb, (na, nb), norb, ref


@pytest.fixture(scope="module")
def pyscf_native_energies(small_problem):
    """Lowest NROOTS energies from PySCF's native multi-root kernel_fixed_space."""
    from pyscf.fci import selected_ci

    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    myci = selected_ci.SelectedCI()
    e_native, _ = selected_ci.kernel_fixed_space(
        myci, hcore, eri, norb, nelec, ci_strs=(sa, sb), nroots=NROOTS
    )
    return np.sort(np.atleast_1d(np.asarray(e_native, dtype=float)))[:NROOTS]


@pytest.fixture(scope="module")
def pyscf_native_singlet_energies(small_problem):
    """Lowest NROOTS singlet energies via PySCF's native fix_spin_ projection."""
    from pyscf.fci import addons, selected_ci

    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    myci = addons.fix_spin_(selected_ci.SelectedCI(), ss=SPIN_SQ_TARGET)
    e_native, _ = selected_ci.kernel_fixed_space(
        myci, hcore, eri, norb, nelec, ci_strs=(sa, sb), nroots=NROOTS
    )
    return np.sort(np.atleast_1d(np.asarray(e_native, dtype=float)))[:NROOTS]


@pytest.fixture(scope="module")
def torch_sci_ground_energy(small_problem):
    """Ground-state energy from the native TorchSCISolver (single-root, CPU)."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = create_solver("torch_sci", symmetrize_spin=True, device="cpu")
    results = solver.sci_solver_callable([(sa, sb)], hcore, eri, norb, nelec)
    return results[0].energy


# ---------------------------------------------------------------------------
# General-algebra solvers: agreement, multiroot, spin projection
# ---------------------------------------------------------------------------
def test_general_algebra_solvers_registered():
    """Every general-algebra backend is discoverable via the factory."""
    for name in ("arpack", "lobpcg", "primme", "slepc", "evsl", "pyscf_davidson"):
        assert name in AVAILABLE_SOLVERS


def test_arpack_via_factory_is_general_algebra_solver():
    """create_solver('arpack') builds an ArpackSolver on the general-algebra base + mixin."""
    solver = create_solver("arpack", symmetrize_spin=True)
    assert isinstance(solver, ArpackSolver)
    assert isinstance(solver, GeneralAlgebraSolverBase)
    assert isinstance(solver, SpinFilterMixin)


@pytest.mark.parametrize(
    ("n", "ncv", "raises"),
    [
        (107_374_182, 21, False),  # (ncv - 1) * n = 2**31 - 8: still fits
        (107_374_183, 21, True),  # one more and the last Lanczos offset wraps
        (110_000_000, None, False),  # default ncv = 20: 19 * n fits
        (113_025_456, None, True),  # 19 * n just past 2**31 - 1
        (715_827_882, 2, False),  # small ncv: workd's 3 * n is the binding term, and fits
        (715_827_883, 2, True),
    ],
)
def test_arpack_rejects_int32_overflowing_workspace(monkeypatch, n, ncv, raises):
    """ARPACK's 32-bit offsets: refuse to call eigsh once max(ncv - 1, 3) * n > 2**31 - 1.

    The problem is a stand-in with only a huge ``dim``; ``eigsh`` itself is replaced, so
    nothing of size ``n`` is allocated. A real overflowing call would segfault.
    """
    from sci_solvers.energy_solvers.general_algebra_solvers import arpack as arpack_module

    monkeypatch.setattr(arpack_module, "HAS_ILP64", False)
    called = []

    def fake_eigsh(*_args, **_kw):
        called.append(True)
        return np.zeros(1), np.zeros((1, 1))

    monkeypatch.setattr(arpack_module.spla, "eigsh", fake_eigsh)

    class _HugeProblem:
        dim = n

        def linear_operator(self):
            return None

    solver = ArpackSolver(symmetrize_spin=False, ncv=ncv)
    if raises:
        with pytest.raises(ValueError, match="32-bit index"):
            solver._diagonalize(_HugeProblem(), nroots=1)
        assert not called
    else:
        solver._diagonalize(_HugeProblem(), nroots=1)
        assert called


def test_arpack_int32_guard_skipped_on_ilp64(monkeypatch):
    """An ILP64 SciPy build has 64-bit ARPACK indices, so the guard lets any size through."""
    from sci_solvers.energy_solvers.general_algebra_solvers import arpack as arpack_module

    monkeypatch.setattr(arpack_module, "HAS_ILP64", True)
    arpack_module._check_int32_workspace(10**10, 1, 21)


@pytest.mark.parametrize("name", SHARED_SOLVERS)
def test_solver_recovers_reference(small_problem, name):
    """Every general-algebra solver recovers the lowest NROOTS dense-reference energies."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = _solver(name, symmetrize_spin=True, nroots=NROOTS)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    energies = np.array([s.energy for s in states])
    assert len(states) == NROOTS
    assert np.all(np.diff(energies) >= -ATOL_ENERGY), (name, "not ascending")
    assert np.allclose(energies, ref, atol=ATOL_ENERGY), (name, energies, ref)


@pytest.mark.parametrize("name", SHARED_SOLVERS)
def test_multiroot_matches_pyscf_native(small_problem, pyscf_native_energies, name):
    """Every solver's roots match PySCF's native nroots>1 kernel_fixed_space."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = _solver(name, symmetrize_spin=True, nroots=NROOTS)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    energies = np.array([s.energy for s in states])
    assert np.allclose(energies, pyscf_native_energies, atol=ATOL_ENERGY), name


@pytest.mark.parametrize("name", SHARED_SOLVERS)
def test_nroots_one_matches_torch_sci_solver(small_problem, torch_sci_ground_energy, name):
    """With nroots=1, every solver's ground state matches the native TorchSCISolver."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    # 1e-12, not tighter: PRIMME's a_norm=1.0 makes tol an absolute residual bound, and
    # the exact eigenvector's float64 matvec residual on this operator is ~2.4e-14, so a
    # tol below that is unreachable (PRIMME then exhausts maxiter and raises). 1e-12 is
    # still ~1e6x tighter than the ATOL_ENERGY asserted below.
    solver = _solver(name, symmetrize_spin=True, nroots=1, tol=1e-12)
    results = solver.sci_solver_callable([(sa, sb)], hcore, eri, norb, nelec)
    assert results[0].energy == pytest.approx(torch_sci_ground_energy, abs=ATOL_ENERGY), name


@pytest.mark.parametrize("name", SHARED_SOLVERS)
def test_spin_sq_projection_matches_native(small_problem, pyscf_native_singlet_energies, name):
    """Every solver's spin-projected roots match PySCF's native fix_spin_.

    ``pyscf_davidson`` projects natively (fix_spin_); the matrix-free solvers use
    the ``SpinFilterMixin`` post-hoc ``<S^2>`` filter. Both target the singlet.
    """
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = _solver(name, symmetrize_spin=True, nroots=NROOTS, spin_sq=SPIN_SQ_TARGET)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    energies = np.array([s.energy for s in states])
    assert np.allclose(energies, pyscf_native_singlet_energies, atol=ATOL_ENERGY), name
    for state in states:
        s2 = spin_square_of_vector(state.sci_state.amplitudes, sa, sb, norb, nelec)
        assert abs(s2 - SPIN_SQ_TARGET) < 0.1, (name, s2)


# ---------------------------------------------------------------------------
# EigenStates packing (shared by every general-algebra solver, on the base)
# ---------------------------------------------------------------------------
def test_eigen_states_multiroot(small_problem):
    """sci_solver_callable_eigen_states returns NROOTS ascending roots per batch."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = LobpcgSolver(symmetrize_spin=True, nroots=NROOTS)
    batches = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)

    assert len(batches) == 1
    states = batches[0].states
    assert len(states) == NROOTS
    energies = np.array([s.energy for s in states])
    assert np.all(np.diff(energies) >= -ATOL_ENERGY), "energies not ascending"
    assert np.allclose(energies, ref, atol=ATOL_ENERGY)
    for state in states:
        assert state.orbital_occupancies is not None
        assert state.sci_state.amplitudes.shape == (len(sa), len(sb))
        assert state.rdm1 is not None and state.rdm1.shape == (norb, norb)
        assert state.rdm2 is None  # rdm2 opt-in


def test_eigen_states_compute_rdm2(small_problem):
    """compute_rdm2=True populates rdm2 and recomputes energy from the RDMs."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = LobpcgSolver(symmetrize_spin=True, nroots=NROOTS, compute_rdm2=True)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states

    assert np.allclose([s.energy for s in states], ref, atol=ATOL_ENERGY)
    for state in states:
        assert state.rdm2 is not None and state.rdm2.shape == (norb, norb, norb, norb)
        e_from_rdm = np.einsum("pr,pr->", state.rdm1, hcore) + 0.5 * np.einsum(
            "prqs,prqs->", state.rdm2, eri
        )
        assert state.energy == pytest.approx(e_from_rdm, abs=ATOL_ENERGY)


def test_sci_solver_callable_returns_ground_state(small_problem):
    """The base wrapper unwraps each batch's ground state (root 0)."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = LobpcgSolver(symmetrize_spin=True, nroots=NROOTS)
    results = solver.sci_solver_callable([(sa, sb)], hcore, eri, norb, nelec)
    assert len(results) == 1
    assert results[0].energy == pytest.approx(ref[0], abs=ATOL_ENERGY)


@pytest.mark.parametrize(
    "which, reorder",
    [
        ("strs_a", lambda s: s[::-1]),
        ("strs_b", lambda s: np.roll(s, 1)),
        ("strs_a", lambda s: np.concatenate([s, s[-1:]])),  # duplicate
    ],
    ids=["a-reversed", "b-rotated", "a-duplicate"],
)
def test_tpb_rejects_non_ascending_strings(small_problem, which, reorder):
    """TPB batches must be strictly ascending; anything else raises before solving.

    PySCF's selected-CI RDM and <S^2> code (the real-path packing and the spin
    filter's classification) is silently wrong on unsorted strings while the
    energies stay correct, so the base rejects such batches up front.
    """
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    batch = (reorder(sa), sb) if which == "strs_a" else (sa, reorder(sb))
    solver = create_solver("arpack", symmetrize_spin=False)
    with pytest.raises(ValueError, match=f"batch 1: {which} must be strictly ascending"):
        solver.sci_solver_callable([(sa, sb), batch], hcore, eri, norb, nelec)


def test_compute_energy_sorts_ci_strs_by_default(small_problem):
    """``compute_energy`` sorts/deduplicates CI strings unless ``sort_ci_strs=False``.

    Unsorted input would be rejected by the general-algebra solvers' strict-ascending
    guard, so the default sorts it (``np.unique``) and the solve succeeds; opting out
    restores the rejection.
    """
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    ham = HamiltonianData(
        hcore=hcore,
        eri=eri,
        e_nuc=0.0,
        num_orbitals=norb,
        num_elec_a=nelec[0],
        num_elec_b=nelec[1],
    )
    solver = create_solver("arpack", symmetrize_spin=False, hamiltonian=ham)
    unsorted_batch = (sa[::-1], np.roll(sb, 1))

    (result,) = solver.compute_energy(ci_strs_batches=[unsorted_batch])
    assert result.energy == pytest.approx(ref[0], abs=ATOL_ENERGY)

    with pytest.raises(ValueError, match="must be strictly ascending"):
        solver.compute_energy(ci_strs_batches=[unsorted_batch], sort_ci_strs=False)


def test_general_algebra_via_factory_end_to_end():
    """Factory-built ArpackSolver drives the full SQD recovery loop on real SqDRIFT shots."""
    from methane_reference import methane, methane_bit_array, methane_fci_energy

    ham = methane()
    # nroots=1: the SQD loop only consumes the ground state, and on this subspace each extra
    # root costs ARPACK an order of magnitude more time (multi-root is covered above).
    solver = create_solver("arpack", nroots=1, symmetrize_spin=True, hamiltonian=ham)
    bit_array = methane_bit_array()
    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=ham.hcore,
        two_body_tensor=ham.eri,
        bit_array=bit_array,
        samples_per_batch=100,
        norb=ham.num_orbitals,
        nelec=(ham.num_elec_a, ham.num_elec_b),
        max_iterations=2,
        sci_solver=solver.sci_solver_callable,
        max_dim=100,
        num_batches=1,
        symmetrize_spin=True,
        seed=0,
    )
    assert math.isfinite(result.energy)
    # Variational: a subspace energy is never below the exact one (here ~3e-3 Ha above it).
    e_fci = methane_fci_energy()
    assert e_fci - 1e-8 <= result.energy < e_fci + 1e-2


# ---------------------------------------------------------------------------
# SpinFilterMixin: guard, runtime toggle, escalation
# ---------------------------------------------------------------------------
def test_spin_projection_returns_singlet(small_problem):
    """The projected ground state is a singlet (<S^2> ~ 0)."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = ArpackSolver(symmetrize_spin=True, spin_sq=0.0, nroots=1)
    state = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states[
        0
    ]
    s2 = spin_square_of_vector(np.asarray(state.sci_state.amplitudes), sa, sb, norb, nelec)
    assert abs(s2) < 0.1, f"projected ground state is not a singlet: <S^2>={s2}"


def test_spin_filter_toggle_disables_projection(small_problem):
    """spin_filter=False (at construction or per call) disables the <S^2> projection."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    plain = ArpackSolver(symmetrize_spin=True, nroots=1)
    e_plain = (
        plain
        .sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0]
        .states[0]
        .energy
    )

    off = ArpackSolver(symmetrize_spin=True, spin_sq=0.0, spin_filter=False, nroots=1)
    e_off = (
        off
        .sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0]
        .states[0]
        .energy
    )
    assert e_off == pytest.approx(e_plain, abs=ATOL_ENERGY)

    on = ArpackSolver(symmetrize_spin=True, spin_sq=0.0, nroots=1)
    e_percall = (
        on
        .sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec, spin_filter=False)[0]
        .states[0]
        .energy
    )
    assert e_percall == pytest.approx(e_plain, abs=ATOL_ENERGY)


def test_spin_filter_raises_when_sector_underpopulated(small_problem):
    """The mixin raises a clear error when it cannot fill nroots for the target spin."""
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = ArpackSolver(
        symmetrize_spin=False, nroots=1, spin_sq=999.0, tol_s2=0.1, oversample=1, max_attempts=2
    )
    with pytest.raises(ValueError, match="fewer than"):
        solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)


def test_guard_rejects_downstream_spin_sq():
    """Defining a solver that stacks the mixin over a class also declaring spin_sq raises."""

    class NativeSpin(EnergyCalculatorBase):
        def __init__(self, *, spin_sq=None, **kwargs):
            super().__init__(**kwargs)
            self.spin_sq = spin_sq

        def sci_solver_callable_eigen_states(self, *_a, **_k):  # pragma: no cover
            return []

    with pytest.raises(TypeError, match="spin_sq"):

        class Bad(SpinFilterMixin, NativeSpin):  # noqa: F841 - defining it must raise
            def sci_solver_callable_eigen_states(self, *_a, **_k):  # pragma: no cover
                return []


def test_guard_allows_clean_stack():
    """A clean stack (no shared arg names) defines without the guard firing."""
    assert issubclass(ArpackSolver, SpinFilterMixin)


# ---------------------------------------------------------------------------
# Spin symmetrization: sector policy + solver-level behavior
# ---------------------------------------------------------------------------
def test_sector_policy():
    """The base's parity-sector policy: k==1 -> +1 only; k>1 -> both sectors."""
    solver = ArpackSolver(symmetrize_spin=True, nroots=1)
    assert solver._sectors_for_batch(1) == (+1,)
    assert solver._sectors_for_batch(3) == (+1, -1)


@pytest.mark.parametrize("name", ["arpack", "lobpcg"])
def test_symmetrized_nroots_one_is_ground_state(small_problem, name):
    """symmetrize_spin + nroots=1 solves the +1 sector and gives the ground state."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = _solver(name, symmetrize_spin=True, nroots=1)
    e = solver.sci_solver_callable([(sa, sb)], hcore, eri, norb, nelec)[0].energy
    assert e == pytest.approx(ref[0], abs=ATOL_ENERGY), name


@pytest.mark.parametrize("name", ["arpack", "lobpcg"])
def test_symmetrized_multiroot_merges_sectors(small_problem, name):
    """symmetrize_spin + nroots>1 merges both sectors into the full lowest nroots."""
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = _solver(name, symmetrize_spin=True, nroots=NROOTS)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    e = np.array([s.energy for s in states])
    assert np.allclose(e, ref, atol=ATOL_ENERGY), name


# ---------------------------------------------------------------------------
# EigenProblem / SymmetrizedEigenProblem math (adapter-independent)
# ---------------------------------------------------------------------------
def test_matvec_matches_dense(small_problem):
    """The matrix-free matvec reproduces the dense Hamiltonian."""
    problem = small_problem[0]
    dense = problem.matvec(np.eye(problem.dim))
    assert np.allclose(dense, dense.T, atol=1e-9), "H not symmetric"
    assert np.allclose(dense, problem.sparse().toarray(), atol=1e-9)


@pytest.mark.parametrize("sector", [+1, -1])
def test_symmetrized_view_is_self_adjoint(small_problem, sector):
    """S^T H S is symmetric and its packed matvec is self-adjoint."""
    problem = small_problem[0]
    view = SymmetrizedEigenProblem(problem, sector=sector)
    hs = view.sparse().toarray()
    assert np.allclose(hs, hs.T, atol=1e-9), sector
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal(view.dim), rng.standard_normal(view.dim)
    assert np.isclose(y @ (hs @ x), x @ (hs @ y)), sector


@pytest.mark.parametrize("sector", [+1, -1])
def test_symmetrized_view_matvec_matches_sparse(small_problem, sector):
    """The packed 1-D and block matvec both reproduce S^T H S."""
    problem = small_problem[0]
    view = SymmetrizedEigenProblem(problem, sector=sector)
    hs = view.sparse().toarray()
    rng = np.random.default_rng(1)
    block = rng.standard_normal((view.dim, 3))
    assert np.allclose(view.matvec(block[:, 0]), hs @ block[:, 0], atol=1e-9), sector
    assert np.allclose(view.matvec(block), hs @ block, atol=1e-9), sector


def test_symmetrized_sector_dims_and_spectrum_union(small_problem):
    """The +1 and -1 sector dims sum to D^2 and their spectra union to the full one."""
    problem = small_problem[0]
    d = problem.Na
    vp = SymmetrizedEigenProblem(problem, sector=+1)
    vm = SymmetrizedEigenProblem(problem, sector=-1)
    assert vp.dim == d * (d + 1) // 2
    assert vm.dim == d * (d - 1) // 2
    assert vp.dim + vm.dim == problem.dim
    ep = np.linalg.eigvalsh(vp.sparse().toarray())
    em = np.linalg.eigvalsh(vm.sparse().toarray())
    full = np.linalg.eigvalsh(problem.sparse().toarray())
    assert np.allclose(np.sort(np.concatenate([ep, em])), full, atol=1e-9)


@pytest.mark.parametrize("sector", [+1, -1])
def test_symmetrized_unpack_roundtrip_and_eigenvector(small_problem, sector):
    """Pack/unpack roundtrips; an unpacked eigenvector solves the full H with the right parity."""
    problem = small_problem[0]
    view = SymmetrizedEigenProblem(problem, sector=sector)
    rng = np.random.default_rng(2)
    x = rng.standard_normal(view.dim)
    assert np.allclose(view._compress(view._expand(x)), x, atol=1e-12), sector

    w, v = np.linalg.eigh(view.sparse().toarray())
    full_vec = view.unpack_eigenvectors(v[:, :1])[:, 0]
    assert np.isclose(np.linalg.norm(full_vec), 1.0)
    hfull = problem.sparse().toarray()
    assert np.linalg.norm(hfull @ full_vec - w[0] * full_vec) < 1e-8, sector
    c = full_vec.reshape(problem.Na, problem.Nb)
    assert np.allclose(c, sector * c.T, atol=1e-9), sector


def test_symmetrized_view_rejects_unequal_pools():
    """A non-closed-shell subspace (strs_a != strs_b) is rejected."""
    hcore, eri, sa, sb = _random_system(7, 6, 3, 2)  # na != nb -> different pools
    problem = EigenProblem(sa, sb, hcore, eri, 6, (3, 2))
    with pytest.raises(ValueError, match="identical alpha/beta"):
        SymmetrizedEigenProblem(problem, sector=+1)


# ---------------------------------------------------------------------------
# Native TruncatedDavidsonSolver: multiroot / RDM / symmetrize coverage
# ---------------------------------------------------------------------------
def _numba_available() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except ImportError:
        return False


requires_numba = pytest.mark.skipif(not _numba_available(), reason="numba not installed")


@pytest.fixture()
def no_truncation(monkeypatch):
    """Disable TruncatedDavidsonSolver amplitude truncation for exact comparisons."""
    orig = TruncatedDavidsonSolver.__init__.__kwdefaults__ or {}
    monkeypatch.setattr(
        TruncatedDavidsonSolver.__init__,
        "__kwdefaults__",
        {**orig, "delta_max": 0.0, "delta_min": 0.0},
    )


@requires_numba
def test_truncated_davidson_returns_eigen_states(small_problem, no_truncation):
    """TruncatedDavidsonSolver returns NROOTS ascending roots matching the dense reference."""
    _ = no_truncation
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    solver = TruncatedDavidsonSolver(symmetrize_spin=False, nroots=NROOTS, eps=1e-9, max_cycle=1000)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    assert len(states) == NROOTS
    energies = np.array([s.energy for s in states])
    assert np.all(np.diff(energies) >= -ATOL_ENERGY), "energies not ascending"
    assert np.allclose(energies, ref, atol=ATOL_ENERGY)
    for state in states:
        assert state.sci_state.amplitudes.shape == (len(sa), len(sb))


@requires_numba
def test_truncated_davidson_rdm2_populated(small_problem, no_truncation):
    """compute_rdm2=True populates rdm2 and the energy is recoverable from the RDMs."""
    _ = no_truncation
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = TruncatedDavidsonSolver(symmetrize_spin=False, nroots=1, eps=1e-9, max_cycle=1000)
    state = solver.sci_solver_callable_eigen_states(
        [(sa, sb)], hcore, eri, norb, nelec, compute_rdm1=True, compute_rdm2=True
    )[0].states[0]
    assert state.rdm2 is not None and state.rdm2.shape == (norb, norb, norb, norb)
    e_from_rdm = np.einsum("pr,pr->", state.rdm1, hcore) + 0.5 * np.einsum(
        "prqs,prqs->", state.rdm2, eri
    )
    assert state.energy == pytest.approx(e_from_rdm, abs=ATOL_ENERGY)


@requires_numba
def test_truncated_davidson_multiroot_matches_pyscf_native(
    small_problem, pyscf_native_energies, no_truncation
):
    """TruncatedDavidsonSolver multiroot energies match PySCF's native kernel_fixed_space."""
    _ = no_truncation
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    solver = TruncatedDavidsonSolver(symmetrize_spin=False, nroots=NROOTS, eps=1e-9, max_cycle=1000)
    states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[0].states
    energies = np.array([s.energy for s in states])
    assert np.allclose(energies, pyscf_native_energies, atol=ATOL_ENERGY)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, {"aNorm": 1.0}), ({"a_norm": None}, {}), ({"a_norm": 50.0}, {"aNorm": 50.0})],
)
def test_primme_a_norm_forwarding(small_problem, monkeypatch, kwargs, expected):
    """``a_norm`` (default 1.0) reaches ``primme.eigsh`` as ``aNorm``; ``None`` omits it."""
    import sys
    import types

    problem = small_problem[0]
    seen: dict = {}

    def fake_eigsh(_op, k, **kwargs):
        seen.update(kwargs)
        w, v = np.linalg.eigh(problem.sparse().toarray())
        return w[:k], v[:, :k]

    monkeypatch.setitem(sys.modules, "primme", types.SimpleNamespace(eigsh=fake_eigsh))
    solver = create_solver("primme", symmetrize_spin=True, **kwargs)
    solver._diagonalize(problem, 1)
    assert {k: v for k, v in seen.items() if k == "aNorm"} == expected


# ---------------------------------------------------------------------------
# matvec_impl plumbing (batched / triton / auto)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["arpack", "lobpcg", "primme"])
def test_matvec_impl_defaults_to_batched_and_auto_resolves_on_cpu(name):
    """Default stays ``batched``; ``auto`` on CPU resolves to ``batched``."""
    assert _solver(name, symmetrize_spin=True).matvec_impl == "batched"
    assert _solver(name, symmetrize_spin=True, matvec_impl="auto").matvec_impl == "batched"


@pytest.mark.parametrize("name", ["arpack", "lobpcg", "primme"])
def test_matvec_impl_triton_requires_cuda(name):
    """An explicit ``triton`` on a CPU device is a loud error, not a silent fallback."""
    with pytest.raises(ValueError, match="CUDA"):
        _solver(name, symmetrize_spin=True, device="cpu", matvec_impl="triton")


def test_matvec_impl_rejects_unknown_and_complex_triton():
    """Unknown names and triton+complex dtype fail at construction."""
    import torch

    with pytest.raises(ValueError, match="matvec_impl"):
        create_solver("arpack", symmetrize_spin=True, matvec_impl="bogus")
    with pytest.raises(NotImplementedError, match="complex"):
        create_solver(
            "arpack",
            symmetrize_spin=True,
            device="cuda",
            dtype=torch.complex128,
            matvec_impl="triton",
        )


def test_matvec_impl_forwarded_to_eigen_problem(small_problem, monkeypatch):
    """The solver's resolved ``matvec_impl`` reaches the ``EigenProblem`` it builds."""
    import sci_solvers.energy_solvers.general_algebra_base as base

    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    seen: dict = {}
    real = base.EigenProblem

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(base, "EigenProblem", spy)
    solver = create_solver("arpack", symmetrize_spin=True, matvec_impl="batched")
    solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)
    assert seen["matvec_impl"] == "batched"


@pytest.mark.parametrize("name", ["arpack", "lobpcg", "primme"])
def test_triton_matches_batched_on_cuda(small_problem, name):
    """On CUDA, ``triton`` reproduces ``batched`` energies for the matvec-driven solvers."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    _, hcore, eri, sa, sb, nelec, norb, ref = small_problem
    energies = {}
    for impl in ("batched", "triton"):
        solver = _solver(name, symmetrize_spin=True, nroots=1, device="cuda", matvec_impl=impl)
        states = solver.sci_solver_callable_eigen_states([(sa, sb)], hcore, eri, norb, nelec)[
            0
        ].states
        energies[impl] = states[0].energy
    assert math.isclose(energies["triton"], energies["batched"], abs_tol=ATOL_ENERGY)
    assert math.isclose(energies["triton"], ref[0], abs_tol=ATOL_ENERGY)


def test_triton_block_matvec_matches_batched_on_cuda(small_problem):
    """On CUDA, the triton column-loop ``matmat`` equals the batched ``vmap`` block apply."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("triton")
    _, hcore, eri, sa, sb, nelec, norb, _ = small_problem
    kw = {"device": "cuda"}
    pb = EigenProblem(sa, sb, hcore, eri, norb, nelec, matvec_impl="batched", **kw)
    pt = EigenProblem(sa, sb, hcore, eri, norb, nelec, matvec_impl="triton", **kw)
    x = np.random.default_rng(0).standard_normal((pb.dim, 3))
    assert np.allclose(pt.matvec(x), pb.matvec(x), atol=1e-9)
    assert np.allclose(pt.matvec(x[:, 0]), pb.matvec(x[:, 0]), atol=1e-9)


@pytest.mark.parametrize("block, expected", [(None, None), (10, 10)])
def test_pyscf_davidson_forwards_davidson_block_size(small_problem, monkeypatch, block, expected):
    """``davidson_block_size`` reaches ``kernel_fixed_space`` as ``max_space``; ``None`` omits it."""
    from pyscf.fci import selected_ci

    problem, *_ = small_problem
    seen: dict = {}
    real = selected_ci.kernel_fixed_space

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(selected_ci, "kernel_fixed_space", spy)
    solver = _solver("pyscf_davidson", symmetrize_spin=True, davidson_block_size=block)
    energies = solver._diagonalize(problem, 1).energies
    assert seen.get("max_space") == expected
    np.testing.assert_allclose(
        energies[0], np.linalg.eigvalsh(problem.sparse().toarray())[0], atol=1e-8
    )
