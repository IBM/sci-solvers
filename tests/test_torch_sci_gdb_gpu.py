"""Real and complex GPU multi-root comparisons through 50,000 determinants.

Run with ``pytest -s tests/test_torch_sci_gdb_gpu.py`` to see timing tables.
Timings are single cold solver calls after device initialization, not speed gates.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
from collections.abc import Callable
from itertools import combinations
from time import perf_counter
from typing import Any

import numpy as np
import pytest
import scipy
import torch
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl import mult, mult_pairs, pair_helpers, qiskit_adapter
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

from sci_solvers.eigen_problem import PairCIProblem
from sci_solvers.energy_solvers.factory import create_solver

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
NORB = 10
NELEC = (5, 4)
NROOTS = 4
PAIR_WORKSPACE_BYTES = 512 * 1024**2


@pytest.fixture(autouse=True)
def pair_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the same bounded enumeration workspace for all drivers and residual checks."""
    # PairCIProblem does not expose this setting as a constructor argument.
    # Its default 8 GiB enumeration budget is too large for small GPUs.
    monkeypatch.setattr(pair_helpers, "DEFAULT_BUDGET_BYTES", PAIR_WORKSPACE_BYTES)


def _synchronize(device: torch.device) -> None:
    """Include outstanding device work in wall-clock measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.mps.synchronize()


def _timed(call: Callable[[], Any], device: torch.device) -> tuple[Any, float]:
    """Time a synchronous call, returning its result and elapsed seconds."""
    _synchronize(device)
    start = perf_counter()
    result = call()
    _synchronize(device)
    return result, perf_counter() - start


@pytest.fixture(scope="module", params=["cuda", "mps"])
def gpu_device(request: pytest.FixtureRequest) -> torch.device:
    """Initialize each available GPU backend; never substitute a CPU run."""
    available = (
        torch.cuda.is_available() if request.param == "cuda" else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"no {request.param} GPU available")
    device = torch.device("cuda:0" if request.param == "cuda" else "mps")
    torch.ones(1, dtype=torch.float32, device=device).square()
    _synchronize(device)
    return device


@pytest.fixture(scope="module", params=["real", "complex"])
def system(
    request: pytest.FixtureRequest, gpu_device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (10,10) hcore, (10,10,10,10) ERI and full (252,)/(210,) spin pools."""
    if request.param == "complex" and gpu_device.type == "mps":
        pytest.skip("complex128 reference comparisons require CUDA; MPS has no float64")
    rng = np.random.default_rng(2026)
    h = rng.standard_normal((NORB, NORB)) * 0.04
    hcore = h + h.T + np.diag(np.linspace(-0.8, 0.8, NORB))
    eri = np.zeros((NORB,) * 4)
    for _ in range(3):
        g = rng.standard_normal((NORB, NORB)) * 0.03
        g += g.T.copy()
        eri += np.einsum("pq,rs->pqrs", g, g)
    if request.param == "complex":
        h_imag = rng.standard_normal((NORB, NORB)) * 0.02
        hcore = hcore + 1j * (h_imag - h_imag.T)
        eri_imag = rng.standard_normal((NORB,) * 4) * 0.001
        eri_imag = 0.5 * (eri_imag - eri_imag.transpose(1, 0, 3, 2))
        eri_imag = 0.5 * (eri_imag + eri_imag.transpose(2, 3, 0, 1))
        eri = eri + 1j * eri_imag
        assert np.max(np.abs(hcore.imag)) > 0.01
        assert np.max(np.abs(eri.imag)) > 0.001
        np.testing.assert_allclose(eri, eri.conj().transpose(1, 0, 3, 2), rtol=0, atol=1e-14)
        np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), rtol=0, atol=1e-14)
    sa, sb = [
        np.array(sorted(sum(1 << i for i in c) for c in combinations(range(NORB), n)))
        for n in NELEC
    ]
    return hcore, eri, sa, sb


def _run_solver(
    name: str,
    mode: str,
    batch: tuple[np.ndarray, np.ndarray],
    hcore: np.ndarray,
    eri: np.ndarray,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve one basis; return (roots,) energies, (roots,N) coefficients and timings.

    Separate eigensolve time from the public call's setup and observable work.
    All drivers use the same device, operator precision and requested tolerance.
    """
    print(f"  Solving {name}/{mode} on {device}", flush=True)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    if np.iscomplexobj(hcore) or np.iscomplexobj(eri):
        dtype = torch.complex128
    tol = 5e-5 if device.type == "mps" else 1e-9
    common = dict(device=device, dtype=dtype, mode=mode, matvec_impl="batched")
    if name == "torch_sci":
        solver = make_torch_sci_solver(
            n_roots=NROOTS,
            apply_ab_projection=False,
            davidson_tolerance=tol,
            davidson_block_size=32,
            max_davidson_iters=100,
            **common,
        )

        def call() -> Any:
            return solver([batch], hcore, eri, NORB, NELEC)[0]

        owner, method = qiskit_adapter, "davidson_torch"
    else:
        solver = create_solver(
            name,
            nroots=NROOTS,
            symmetrize_spin=False,
            tol=tol,
            maxiter=400,
            **({"max_restarts": 100, "ncv": 32} if name == "arpack" else {}),
            **common,
        )

        def call() -> Any:
            return solver.sci_solver_callable_eigen_states([batch], hcore, eri, NORB, NELEC)[0]

        owner, method = solver, "_diagonalize"
    original_solve = getattr(owner, method)
    kernel_owner = mult if mode == "tpb" else mult_pairs
    kernel_name = "mult_torch" if mode == "tpb" else "mult_pair_torch"
    original_matvec = getattr(kernel_owner, kernel_name)
    solve_seconds = 0.0
    matvec_calls = 0

    def measured_solve(*args: Any, **kwargs: Any) -> Any:
        """Time the eigensolver after its Hamiltonian setup has completed."""
        nonlocal solve_seconds
        if name != "torch_sci":
            args[0].hdiag()  # Build the lazy pair context outside the solve timer.
        result, solve_seconds = _timed(lambda: original_solve(*args, **kwargs), device)
        if name == "torch_sci":
            assert result[2]["converged"], result[2]["final_residual_norms"]
        else:
            assert np.all(result.converged), result.converged
        return result

    def checked_matvec(
        hii: torch.Tensor, v: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Check the (N,) diagonal/input/output really execute on the selected GPU.

        On MPS the solver evaluates each root's energy once more as a float64
        Rayleigh quotient on the CPU (MPS has no float64); that call is the only
        one allowed off the device.
        """
        nonlocal matvec_calls
        if device.type == "mps" and v.device.type == "cpu":
            assert hii.device.type == "cpu" and v.dtype in (torch.float64, torch.complex128)
            return original_matvec(hii, v, *args, **kwargs)
        assert hii.device.type == v.device.type == device.type
        assert hii.dtype == (dtype.to_real() if dtype.is_complex else dtype)
        assert v.dtype == dtype
        result = original_matvec(hii, v, *args, **kwargs)
        assert result.device == v.device
        assert result.dtype == dtype
        matvec_calls += 1
        return result

    with monkeypatch.context() as patch:
        patch.setattr(owner, method, measured_solve)
        patch.setattr(kernel_owner, kernel_name, checked_matvec)
        result, total_seconds = _timed(call, device)
    assert matvec_calls > NROOTS
    assert len(result.states) == NROOTS
    energies = np.array([root.energy for root in result.states])
    vectors = np.stack([root.sci_state.amplitudes.ravel() for root in result.states])
    pa, pb = (
        batch
        if mode == "gdb"
        else (np.repeat(batch[0], len(batch[1])), np.tile(batch[1], len(batch[0])))
    )
    assert vectors.shape == (NROOTS, len(pa))
    for root in result.states:
        np.testing.assert_array_equal(root.sci_state.ci_strs_a, batch[0])
        np.testing.assert_array_equal(root.sci_state.ci_strs_b, batch[1])
    if mode == "tpb":
        # Normalize all results to GDB's interleaved-Wick convention.
        sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), NORB).numpy()
        vectors *= sigma
    return (
        energies,
        vectors,
        dict(
            solver=f"{name}/{mode}",
            total_seconds=total_seconds,
            solve_seconds=solve_seconds,
            matvec_calls=matvec_calls,
            tolerance=tol,
            dtype=str(dtype),
        ),
    )


def _check_roots(
    runs: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    problem: PairCIProblem,
    device: torch.device,
) -> None:
    """Check (roots,N) vectors against CPU float64/complex128 residuals and the first run."""
    energy_tol = 5e-5 if device.type == "mps" else 1e-8
    coefficient_tol = 5e-4 if device.type == "mps" else 1e-6
    residual_tol = 1e-4 if device.type == "mps" else 1e-8
    reference_e, reference_v, _ = runs[0]
    # These fixtures must have isolated roots for coefficient-by-coefficient parity.
    assert np.min(np.diff(reference_e)) > 1e-3
    for energies, vectors, timing in runs:
        timing["max_energy_error"] = float(np.max(np.abs(energies - reference_e)))
        np.testing.assert_allclose(energies, reference_e, atol=energy_tol, rtol=0)
        overlaps = np.sum(reference_v.conj() * vectors, axis=1)
        assert np.all(np.abs(overlaps) > 0.5), "roots do not match the reference states"
        phases = overlaps / np.abs(overlaps)
        aligned = vectors * phases.conj()[:, None]
        timing["max_amplitude_error"] = float(np.max(np.abs(aligned - reference_v)))
        np.testing.assert_allclose(aligned, reference_v, atol=coefficient_tol, rtol=0)
        assert np.max(np.linalg.norm(aligned - reference_v, axis=1)) < 10 * coefficient_tol
        np.testing.assert_allclose(
            vectors @ vectors.conj().T, np.eye(NROOTS), atol=coefficient_tol, rtol=0
        )
        print(
            f"  Validating {timing['solver']}: {vectors.shape[1]:,} determinants, "
            "independent CPU residuals (outside solve timing)",
            flush=True,
        )
        residuals = np.linalg.norm(problem.matvec(vectors.T) - vectors.T * energies, axis=0)
        timing["max_residual"] = float(np.max(residuals))
        assert timing["max_residual"] < residual_tol, timing


def _report(
    case: str,
    n_pairs: int,
    device: torch.device,
    runs: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    record_property: Callable[[str, object], None],
) -> None:
    """Print a comparison table and retain machine-readable results in pytest/JUnit."""
    timings = [run[2] for run in runs]
    gdb = next(row for row in timings if row["solver"] == "torch_sci/gdb")
    print(f"\n{case}: {n_pairs:,} determinants, {NROOTS} roots, {device}, {gdb['dtype']}")
    print(f"{'solver':<20} {'total (s)':>12} {'solve (s)':>12} {'solve/GDB':>12} {'matvecs':>9}")
    for row in timings:
        row["solve_over_gdb"] = row["solve_seconds"] / gdb["solve_seconds"]
        print(
            f"{row['solver']:<20} {row['total_seconds']:12.3f} {row['solve_seconds']:12.3f} "
            f"{row['solve_over_gdb']:12.3f} {row['matvec_calls']:9d}"
        )
    record_property(
        "gpu_comparison",
        json.dumps(
            dict(
                case=case,
                n_pairs=n_pairs,
                n_roots=NROOTS,
                device=str(device),
                device_name=torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "Apple MPS",
                torch_version=torch.__version__,
                scipy_version=scipy.__version__,
                torch_threads=torch.get_num_threads(),
                pair_workspace_bytes=PAIR_WORKSPACE_BYTES,
                runs=timings,
            )
        ),
    )


@pytest.mark.parametrize(
    "n_alpha,n_beta", [(50, 40), (100, 100), (200, 100)], ids=["2k", "10k", "20k"]
)
def test_gpu_cartesian_tpb_parity_and_runtime(
    n_alpha: int,
    n_beta: int,
    gpu_device: torch.device,
    system: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare GPU TPB and GDB on exactly the same Cartesian product through 20K pairs."""
    hcore, eri, all_a, all_b = system
    rng = np.random.default_rng(81)
    sa = np.sort(rng.choice(all_a, n_alpha, replace=False))
    sb = np.sort(rng.choice(all_b, n_beta, replace=False))
    pa, pb = np.repeat(sa, len(sb)), np.tile(sb, len(sa))
    runs = [
        _run_solver("torch_sci", mode, batch, hcore, eri, gpu_device, monkeypatch)
        for mode, batch in [("tpb", (sa, sb)), ("gdb", (pa, pb))]
    ]
    problem = PairCIProblem(pa, pb, hcore, eri, NORB, NELEC)
    try:
        _check_roots(runs, problem, gpu_device)
    finally:
        _report("cartesian", len(pa), gpu_device, runs, record_property)


@pytest.mark.parametrize("n_pairs", [2000, 10000, 50000], ids=["2k", "10k", "50k"])
def test_gpu_nonproduct_arpack_lobpcg_parity_and_runtime(
    n_pairs: int,
    gpu_device: torch.device,
    system: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Compare all roots and runtimes on non-product pair sets through 50K determinants."""
    hcore, eri, sa, sb = system
    indices = np.sort(np.random.default_rng(91).choice(len(sa) * len(sb), n_pairs, replace=False))
    pa, pb = sa[indices // len(sb)], sb[indices % len(sb)]
    assert len(np.unique(pa)) * len(np.unique(pb)) > n_pairs
    runs = [
        _run_solver(name, "gdb", (pa, pb), hcore, eri, gpu_device, monkeypatch)
        for name in ["arpack", "lobpcg", "torch_sci"]
    ]
    problem = PairCIProblem(pa, pb, hcore, eri, NORB, NELEC)
    try:
        _check_roots(runs, problem, gpu_device)
    finally:
        _report("nonproduct", n_pairs, gpu_device, runs, record_property)
