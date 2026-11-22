"""Tests for the EVSL spectrum-slicing backend and the EVSL package internals.

The ``EvslSolver`` replaces the former ``EvslAdapter``; the ``evsl`` package
(``solve_sliced``, DOS, streaming, checkpoint/resume) is unchanged. Real EVSL
solves are gated on a built ``libevsl`` (``@requires_evsl``) and marked
``@pytest.mark.slow``; the pure decision-logic and manifest tests always run.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from itertools import combinations

import numpy as np
import pytest

from sci_solvers.eigen_problem import EigenProblem
from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, create_solver
from sci_solvers.energy_solvers.general_algebra_solvers import EvslSolver
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin

NROOTS = 3
ATOL_ENERGY = 1e-6
ATOL_RESIDUAL = 1e-4

# Test-only fast EVSL knobs (production defaults are overkill for dim-400 toys).
_EVSL_FAST = {"pol_deg_max": 50, "lanczos_dim": 60}


def _all_strings(norb: int, nelec: int) -> np.ndarray:
    return np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(norb), nelec)),
        dtype=np.int64,
    )


def _random_system(seed: int, norb: int, na: int, nb: int):
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


def _evsl_available() -> bool:
    from sci_solvers.energy_solvers.evsl import EvslError, load_evsl

    try:
        load_evsl()
    except EvslError:
        return False
    return True


requires_evsl = pytest.mark.skipif(
    not _evsl_available(),
    reason="libevsl not built/locatable (see scripts/energy_solvers/build_evsl.sh)",
)


# --- registration / construction (always run) -------------------------------
def test_evsl_registered_and_is_general_algebra_solver():
    """'evsl' resolves to an EvslSolver on the general-algebra base + mixin."""
    assert "evsl" in AVAILABLE_SOLVERS
    solver = create_solver("evsl", symmetrize_spin=True)
    assert isinstance(solver, EvslSolver)
    assert isinstance(solver, SpinFilterMixin)


def test_evsl_uses_spin_filter_mixin_for_projection():
    """EVSL has no native spin projection; a spin target is honored via the mixin."""
    solver = EvslSolver(symmetrize_spin=True, spin_sq=0.0)
    assert solver.spin_sq == 0.0
    assert isinstance(solver, SpinFilterMixin)


# --- real solves (gated on libevsl) -----------------------------------------
@pytest.mark.slow
@requires_evsl
def test_evsl_window_solve_matches_dense(small_problem):
    """With an explicit window, EVSL recovers exactly the dense roots inside it."""
    problem, _, _, _, _, _, _, ref = small_problem
    dense = np.linalg.eigvalsh(problem.sparse().toarray())
    lo = float(dense[0]) - 0.05 * (dense[-1] - dense[0])
    hi = 0.5 * (float(ref[NROOTS - 1]) + float(dense[NROOTS]))
    expected = dense[(dense >= lo) & (dense <= hi)]

    solver = EvslSolver(symmetrize_spin=False, window=(lo, hi), tol=1e-9, maxiter=500, **_EVSL_FAST)
    res = solver._diagonalize(problem, expected.size).sorted()
    assert np.allclose(res.energies[: expected.size], expected, atol=ATOL_ENERGY), (
        res.energies,
        expected,
    )
    for j in range(min(NROOTS, res.energies.size)):
        residual = problem.matvec(res.vectors[:, j]) - res.energies[j] * res.vectors[:, j]
        assert np.linalg.norm(residual) < ATOL_RESIDUAL, j


@pytest.mark.slow
@requires_evsl
def test_evsl_auto_window_solve_matches_dense(small_problem):
    """With no explicit window, EVSL's KPM-DOS window auto-detection (``_lower_band``)
    still recovers the correct lowest roots, reusing its DOS moments in
    ``solve_sliced`` instead of running a second, redundant KPM-DOS pass."""
    problem, _, _, _, _, _, _, ref = small_problem
    assert problem.dim >= 200  # exercises real EVSL slicing, not the dense fallback

    solver = EvslSolver(
        symmetrize_spin=False,
        window=None,
        tol=1e-9,
        maxiter=500,
        dos_deg=50,
        dos_nvec=20,
        **_EVSL_FAST,
    )
    res = solver._diagonalize(problem, NROOTS).sorted()
    assert np.allclose(res.energies, ref, atol=ATOL_ENERGY), (res.energies, ref)
    for j in range(res.energies.size):
        residual = problem.matvec(res.vectors[:, j]) - res.energies[j] * res.vectors[:, j]
        assert np.linalg.norm(residual) < ATOL_RESIDUAL, j


@requires_evsl
def test_evsl_falls_back_to_dense_for_small_subspace():
    """Below the slicing threshold (dim < 200) EVSL uses the exact dense path."""
    hcore, eri, sa, sb = _random_system(5, 5, 2, 2)  # dim = C(5,2)^2 = 100 < 200
    problem = EigenProblem(sa, sb, hcore, eri, 5, (2, 2))
    assert problem.dim < 200
    ref = np.linalg.eigvalsh(problem.sparse().toarray())[:NROOTS]
    solver = EvslSolver(symmetrize_spin=False, tol=1e-9, maxiter=500)
    res = solver._diagonalize(problem, NROOTS).sorted()
    assert np.allclose(res.energies, ref, atol=ATOL_ENERGY), (res.energies, ref)


@pytest.mark.slow
@requires_evsl
def test_evsl_dos_integral_counts_states(small_problem):
    """The KPM DOS integrates to ~the total number of eigenvalues."""
    from sci_solvers.energy_solvers.evsl import spectral_density

    problem = small_problem[0]
    energies, dos = spectral_density(problem, n_points=2000, deg=300, n_vec=60)
    total = float(np.trapezoid(dos, energies))
    assert total == pytest.approx(problem.dim, rel=0.1), (total, problem.dim)


@pytest.mark.slow
@requires_evsl
def test_evsl_slice_boundaries_reuses_supplied_dos_moments(small_problem, monkeypatch):
    """Supplying ``dos_moments`` (a) skips the internal ``kpmdos`` call and
    (b) still produces the boundaries/ecount that ``spslicer`` gives for those
    exact moments (a deterministic check — ``kpmdos`` itself draws random probe
    vectors internally, so two independent calls are not expected to agree, but
    ``spslicer`` applied to the *same* moments is a pure function)."""
    from sci_solvers.energy_solvers.evsl import load_evsl
    from sci_solvers.energy_solvers.evsl.solver import (
        _bounds,
        _kpm_moments,
        _register,
        _slice_boundaries_with_count,
    )

    problem = small_problem[0]
    deg, n_vec, n_slices = 50, 20, 2
    dense = np.linalg.eigvalsh(problem.sparse().toarray())
    window = (float(dense[0]) - 0.05 * (dense[-1] - dense[0]), float(dense[NROOTS]))

    lib = load_evsl()
    cb = None
    try:
        cb = _register(problem, lib)
        bounds = _bounds(problem, lib, None, tol=1e-8)
        mu, ecount = _kpm_moments(lib, deg, n_vec, bounds)

        lo, hi = window
        npts = max(2 * n_slices, int(10 * max(ecount, 1.0)))
        win = np.array([lo, hi, bounds[0], bounds[1]], dtype=np.float64)
        expected_sli = lib.spslicer(mu, deg, win, n_slices, npts)
    finally:
        del cb
        lib.finish()

    def _kpmdos_should_not_be_called(*_a, **_kw):
        raise AssertionError("kpmdos should be skipped when dos_moments is supplied")

    monkeypatch.setattr(
        "sci_solvers.energy_solvers.evsl.bindings.EvslLib.kpmdos", _kpmdos_should_not_be_called
    )
    sli_reused, ecount_reused = _slice_boundaries_with_count(
        problem, n_slices, window, deg=deg, bounds=bounds, dos_moments=(mu, ecount, deg)
    )
    assert np.allclose(sli_reused, expected_sli)
    assert ecount_reused == pytest.approx(ecount)


# --- streaming decision logic (pure, no libevsl) ----------------------------
def test_evsl_default_stream_threshold_is_memory_derived(monkeypatch):
    """The auto-stream ceiling is derived from available memory (pure logic).

    The memory reading is pinned: comparing against a second live reading was flaky,
    since free memory moves between the two calls when tests run in parallel.
    """
    from sci_solvers.energy_solvers.evsl import solver as evsl_solver

    for avail, expected in [(40.0, 20.0), (1.5, 1.0), (None, evsl_solver._STREAM_FALLBACK_GB)]:
        monkeypatch.setattr(evsl_solver, "_available_memory_gb", lambda a=avail: a)
        assert evsl_solver._default_stream_threshold_gb() == expected
    monkeypatch.undo()
    assert evsl_solver._default_stream_threshold_gb() >= 1.0  # the live reading, unpinned


def test_evsl_decide_stream_respects_explicit_toggle():
    """An explicit stream=True/False overrides the estimate and never warns."""
    import warnings

    from sci_solvers.energy_solvers.evsl.solver import _decide_stream

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _decide_stream(False, est_gb=1e6, threshold_gb=1.0, stream_path=None) is False
        assert _decide_stream(True, est_gb=0.0, threshold_gb=1e9, stream_path=None) is True


def test_evsl_decide_stream_auto_warns_above_threshold():
    """Auto mode streams and warns once the estimate exceeds the ceiling."""
    from sci_solvers.energy_solvers.evsl.solver import _decide_stream

    with pytest.warns(UserWarning, match="streaming"):
        do_stream = _decide_stream(None, est_gb=80.0, threshold_gb=8.0, stream_path=None)
    assert do_stream is True

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _decide_stream(None, est_gb=1.0, threshold_gb=8.0, stream_path=None) is False


# --- streaming solves (gated on libevsl) ------------------------------------
@pytest.mark.slow
@requires_evsl
def test_evsl_stream_matches_nonstream(small_problem, tmp_path):
    """Streamed and in-RAM solve_sliced return the identical root set."""
    from sci_solvers.energy_solvers.evsl import solve_sliced

    problem = small_problem[0]
    dense = np.linalg.eigvalsh(problem.sparse().toarray())
    lo = float(dense[0]) - 0.05 * (dense[-1] - dense[0])
    hi = 0.5 * (float(dense[NROOTS - 1]) + float(dense[NROOTS]))

    ram = solve_sliced(problem, (lo, hi), n_slices=2, stream=False, **_EVSL_FAST)
    streamed = solve_sliced(
        problem,
        (lo, hi),
        n_slices=2,
        stream=True,
        stream_path=str(tmp_path / "vecs.dat"),
        **_EVSL_FAST,
    )
    assert isinstance(streamed.vectors, np.memmap)
    assert not isinstance(ram.vectors, np.memmap)
    assert streamed.energies.size == ram.energies.size
    assert np.all(np.diff(streamed.energies) >= -ATOL_ENERGY)
    assert np.allclose(streamed.energies, ram.energies, atol=ATOL_ENERGY)
    for j in range(streamed.energies.size):
        col = np.asarray(streamed.vectors[:, j])
        residual = problem.matvec(col) - streamed.energies[j] * col
        assert np.linalg.norm(residual) < ATOL_RESIDUAL, j


# --- checkpoint / resume (manifest logic pure; solves gated) ----------------
def test_evsl_checkpoint_manifest_roundtrip_and_fingerprint_guard(tmp_path):
    """Manifest init/rewrite is atomic; a fingerprint mismatch is rejected (pure logic)."""
    from sci_solvers.energy_solvers.evsl.solver import (
        _init_checkpoint,
        _load_manifest,
        _write_manifest,
    )

    cdir = tmp_path / "ckpt"
    fp = {"dim": 100, "n_slices": 4, "window": [0.0, 1.0], "bounds": [-1.0, 2.0], "spec": {}}
    assert _init_checkpoint(cdir, fp) == set()

    _write_manifest(cdir, fp, done=[0, 2])
    assert _load_manifest(cdir)["done"] == [0, 2]
    assert _init_checkpoint(cdir, fp) == {0, 2}

    other = {**fp, "window": [0.0, 2.0]}
    with pytest.raises(ValueError, match="different problem"):
        _init_checkpoint(cdir, other)


def test_evsl_save_load_slice_roundtrip(tmp_path):
    """A saved slice loads back identically and is marked done (pure logic)."""
    from sci_solvers.eigen_problem import EigenResult
    from sci_solvers.energy_solvers.evsl.solver import (
        _init_checkpoint,
        _load_manifest,
        _load_slice,
        _save_slice,
    )

    cdir = tmp_path / "ckpt"
    fp = {"dim": 3, "n_slices": 2, "window": [0.0, 1.0], "bounds": [-1.0, 2.0], "spec": {}}
    _init_checkpoint(cdir, fp)

    vals = np.array([0.1, 0.2])
    vecs = np.arange(6, dtype=float).reshape(3, 2)
    _save_slice(cdir, 1, EigenResult(vals, vecs))

    assert _load_manifest(cdir)["done"] == [1]
    loaded = _load_slice(cdir, 1)
    assert np.array_equal(loaded.energies, vals)
    assert np.array_equal(loaded.vectors, vecs)
