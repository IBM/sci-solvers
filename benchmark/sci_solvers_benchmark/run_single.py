"""Time one solver on one fixed subspace: no SQD loop, no configuration recovery.

Reads an FCIDUMP and an input file of bitstrings, optionally subsamples the bitstrings,
splits each into its alpha and beta halves, and calls the solver's
``sci_solver_callable_eigen_states`` once per timed run on the resulting subspace. Every
bitstring is assumed to carry the right electron counts (``prepare_quantum`` and
``prepare_synthetic`` guarantee it).

Usage::

    python -m sci_solvers_benchmark.run_single --fcidump F --shots S --solver_name torch_sci \\
        --solver_init_kwargs '{"device": "cuda"}' --subsample_size 1000 --output out.json

Notes
-----
* Only the solver call is timed; reading the inputs, building the subspace, creating the
  solver, the optional warm-up call and the ⟨S²⟩ evaluation are all outside the timer.
* ``num_runs`` repeats the timed call on the same subspace; mean, sample standard
  deviation and the raw values are reported.
* ``symmetrize_spin=True`` takes the union of the alpha and beta halves as the string set
  of both spins, as ``EnergyCalculatorBase.compute_energy`` does.
* ``inject_hf=True`` reserves one slot of the subsample for the Hartree-Fock bitstring,
  so it is always in the subspace, and draws the rest from the other bitstrings.
* Peak host RAM is the process RSS high-water mark during each timed call. Peak GPU
  memory is the device-wide NVML high-water mark above a baseline (see
  :class:`~sci_solvers_benchmark.memory.GpuRamPoller`).
* ``run_timeout`` caps each solver call (``SIGALRM``, Unix main thread only); a call
  that exceeds it fails the whole cell.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import contextlib
import dataclasses
import json
import signal
import statistics
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from jsonargparse import CLI
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData
from sci_solvers_benchmark.memory import GpuRamPoller, RamPoller
from sci_solvers_benchmark.shots import load_shots
from sci_solvers_benchmark.spin_square import compute_spin_square

# arpack / lobpcg / primme keep a per-subspace cache on the GPU between calls, and
# PyTorch's caching allocator reuses freed memory, so a fresh per-run poller would see ~0
# after the first run. For them one poller spans warm-up and every run with its baseline
# fixed before warm-up, and its peak is reset before each run: each run's reading is then
# its full resident footprint.
_FOOTPRINT_SOLVERS = frozenset({"arpack", "lobpcg", "primme"})

# primme and slepc raise rather than return when they do not converge, so a returned
# result is converged even though it carries no convergence state.
_CONVERGED_IF_RETURNED = frozenset({"primme", "slepc"})


class RunTimeoutError(TimeoutError):
    """A single solver call exceeded ``run_timeout`` seconds."""


@contextlib.contextmanager
def _time_limit(seconds: int | None) -> Iterator[None]:
    """Raise :class:`RunTimeoutError` if the block runs longer than ``seconds``."""
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(_signum: int, _frame: Any) -> None:
        raise RunTimeoutError(f"solver call exceeded run_timeout={seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _cuda_active() -> bool:
    """Whether this process already holds a PyTorch CUDA context.

    Checked with ``is_initialized`` so that nothing here creates a context, which would
    block solvers that need the GPU in exclusive-process mode (e.g. sbd).
    """
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_initialized()
    except ImportError:
        return False


def _sync_cuda() -> None:
    if _cuda_active():
        import torch

        torch.cuda.synchronize()


def hf_bitstring(norb: int, n_alpha: int, n_beta: int) -> str:
    """Hartree-Fock bitstring: beta half then alpha half, occupied orbitals rightmost."""
    beta = "0" * (norb - n_beta) + "1" * n_beta
    alpha = "0" * (norb - n_alpha) + "1" * n_alpha
    return beta + alpha


def sample_bitstrings(counts: dict[str, int], subsample_size: int, seed: int) -> list[str]:
    """Draw ``subsample_size`` distinct bitstrings, weighted by count, without replacement.

    ``0``, or a size at least the number of distinct bitstrings, returns them all in file
    order.
    """
    if subsample_size < 0:
        raise ValueError(f"subsample_size must be non-negative, got {subsample_size}")
    bitstrings = list(counts)
    if subsample_size == 0 or subsample_size >= len(bitstrings):
        return bitstrings
    weights = np.asarray(list(counts.values()), dtype=float)
    idx = np.random.default_rng(seed).choice(
        len(bitstrings), size=subsample_size, replace=False, p=weights / weights.sum()
    )
    return [bitstrings[i] for i in idx]


def _stats(prefix: str, values: list[float], suffix: str = "") -> dict[str, Any]:
    vals = [float(v) for v in values]
    return {
        f"{prefix}_values{suffix}": vals,
        f"{prefix}_mean{suffix}": statistics.fmean(vals),
        f"{prefix}_stdev{suffix}": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def _mem_stats(prefix: str, values: list[float | None]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {}
    return {
        f"{prefix}_values": [round(v, 2) for v in vals],
        f"{prefix}_mean": round(statistics.fmean(vals), 2),
        f"{prefix}_stdev": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 2),
    }


def _convergence(result: Any, solver_name: str) -> str | None:
    state = getattr(result, "state", None)
    if state is None and solver_name in _CONVERGED_IF_RETURNED:
        return "converged"
    return state


def _check_declared_subspace(shots: str, subspace_dim: int) -> None:
    """Fail if a synthetic input would not be solved at the dimension it was built for."""
    if not shots.endswith(".npz"):
        return
    with np.load(shots) as data:
        if "subspace_dim" not in data.files:
            return
        declared = int(data["subspace_dim"])
    if declared != subspace_dim:
        raise ValueError(
            f"{shots} was built for subspace_dim={declared} but the solver would get "
            f"{subspace_dim}; synthetic inputs must be used whole, not subsampled."
        )


def _dice_energy(result: Any, ham: HamiltonianData) -> Any:
    """Replace Dice's eigenvalue with ⟨H⟩ evaluated from its 1- and 2-RDMs.

    The Dice binary always applies an :math:`S^2` level-shift penalty, so the eigenvalue
    it reports is not :math:`\\langle\\psi|H|\\psi\\rangle`. The energy is recomputed as
    :math:`\\sum_{pr} D_{pr} h_{pr} + \\tfrac12 \\sum_{prqs} d_{prqs} (pr|qs)`.
    """
    dm1 = result.sci_state.rdm(rank=1, spin_summed=True)
    dm2 = result.sci_state.rdm(rank=2, spin_summed=True)
    energy = float(
        np.dot(dm1.ravel(), ham.hcore.ravel()) + 0.5 * np.dot(dm2.ravel(), ham.eri.ravel())
    )
    return dataclasses.replace(result, energy=energy, rdm1=dm1, rdm2=dm2)


def run_single(
    fcidump: str,
    shots: str,
    solver_name: str,
    solver_init_kwargs: dict[str, Any] | None = None,
    solver_call_kwargs: dict[str, Any] | None = None,
    subsample_size: int = 0,
    seed: int = 42,
    symmetrize_spin: bool = True,
    inject_hf: bool = True,
    warmup: bool = True,
    num_runs: int = 3,
    nroots: int = 1,
    measure_memory: bool = True,
    run_timeout: int | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    """Time ``num_runs`` solver calls on one subspace and return the measurements.

    Args:
        fcidump: FCIDUMP of the molecule.
        shots: Input bitstrings (see :func:`~sci_solvers_benchmark.shots.load_shots`).
        solver_name: Name passed to ``sci_solvers.energy_solvers.factory.create_solver``.
        solver_init_kwargs: Solver constructor arguments (except ``nroots``).
        solver_call_kwargs: Extra arguments for every solver call (e.g. ``spin_sq``).
        subsample_size: Bitstrings to draw; ``0`` uses them all.
        seed: Subsampling seed.
        symmetrize_spin: Use the union of alpha and beta halves for both spins.
        inject_hf: Always include the Hartree-Fock bitstring.
        warmup: Make one untimed call first.
        num_runs: Timed calls.
        nroots: Lowest eigenpairs to request.
        measure_memory: Record peak host RAM and GPU memory per run.
        run_timeout: Per-call cap in seconds; ``None`` or ``0`` disables it.
        output: If set, write ``{"config": ..., "results": ...}`` there as JSON.

    Returns:
        Subspace dimensions, per-run and aggregated solver time, total and electronic
        energy, ⟨S²⟩, multiplicity, convergence state, memory peaks, and a per-root
        breakdown under ``"roots"``. Top-level energies refer to root 0.

    Raises:
        ValueError: ``nroots`` is also in ``solver_init_kwargs``, ``num_runs < 1``, or a
            synthetic input would be subsampled.
        RunTimeoutError: A call exceeded ``run_timeout``.
        RuntimeError: The solver returned no state.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    init_kwargs = dict(solver_init_kwargs or {})
    if "nroots" in init_kwargs:
        raise ValueError("pass nroots as a top-level argument, not in solver_init_kwargs")
    if nroots != 1:
        init_kwargs["nroots"] = nroots

    ham = HamiltonianData.from_fcidump(fcidump)
    counts = load_shots(shots)
    if inject_hf:
        hf = hf_bitstring(ham.num_orbitals, ham.num_elec_a, ham.num_elec_b)
        rest = {k: v for k, v in counts.items() if k != hf}
        bitstrings = [hf, *sample_bitstrings(rest, max(subsample_size - 1, 0), seed)]
    else:
        bitstrings = sample_bitstrings(counts, subsample_size, seed)

    matrix = np.array([[c == "1" for c in b] for b in bitstrings], dtype=bool)
    strs_a, strs_b = bitstring_matrix_to_ci_strs(matrix, open_shell=not symmetrize_spin)
    subspace_dim = len(strs_a) * len(strs_b)
    _check_declared_subspace(str(shots), subspace_dim)

    solver = create_solver(solver_name, hamiltonian=ham, **init_kwargs)
    call = partial(solver.sci_solver_callable_eigen_states, **(solver_call_kwargs or {}))
    args = (
        [(strs_a, strs_b)],
        ham.hcore,
        ham.eri,
        ham.num_orbitals,
        (ham.num_elec_a, ham.num_elec_b),
    )

    footprint = (
        GpuRamPoller(interval=0.1) if measure_memory and solver_name in _FOOTPRINT_SOLVERS else None
    )
    if footprint is not None:
        footprint.__enter__()
    if warmup:
        call(*args)
        _sync_cuda()

    times: list[float] = []
    runs: list[list[Any]] = []
    peak_ram: list[float | None] = []
    peak_vram: list[float | None] = []
    torch_peak: list[float | None] = []
    for i in range(num_runs):
        _sync_cuda()
        cuda = measure_memory and _cuda_active()
        if cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()
        with contextlib.ExitStack() as stack:
            ram = stack.enter_context(RamPoller(interval=0.5)) if measure_memory else None
            if footprint is not None:
                footprint.reset_peak()
                vram: GpuRamPoller | None = footprint
            elif measure_memory:
                vram = stack.enter_context(GpuRamPoller(interval=0.1))
            else:
                vram = None
            t0 = time.perf_counter()
            with _time_limit(run_timeout):
                (eigen_states,) = call(*args)
            _sync_cuda()
            times.append(time.perf_counter() - t0)
        if measure_memory:
            peak_ram.append(ram.peak_mb if ram else None)
            peak_vram.append(vram.peak_mb if vram else None)
            torch_peak.append(round(torch.cuda.max_memory_allocated() / 1e6, 2) if cuda else None)
        if not eigen_states.states:
            raise RuntimeError(f"{solver_name} returned no state in run {i + 1}/{num_runs}")
        runs.append(list(eigen_states.states))
    if footprint is not None:
        footprint.__exit__(None, None, None)

    if solver_name == "dice":
        runs = [[_dice_energy(r, ham) for r in run] for run in runs]

    roots = []
    for k in range(len(runs[0])):
        energies = [float(run[k].energy) for run in runs]
        spins = [compute_spin_square(run[k].sci_state) for run in runs]
        states = [_convergence(run[k], solver_name) for run in runs]
        roots.append({
            "root_index": k,
            **_stats("electronic_energy", energies),
            **_stats("total_energy", [e + ham.e_nuc for e in energies]),
            **_stats("spin_squared", [s for s, _ in spins]),
            **_stats("multiplicity", [m for _, m in spins]),
            "convergence_values": states,
            "n_converged": sum(s == "converged" for s in states),
        })

    ground = roots[0]
    results: dict[str, Any] = {
        "num_runs": num_runs,
        "n_alpha_dets": len(strs_a),
        "n_beta_dets": len(strs_b),
        "subspace_dim": subspace_dim,
        **_stats("solver_time", times, suffix="_s"),
        **{k: v for k, v in ground.items() if k != "root_index"},
        "roots": roots,
        **_mem_stats("peak_vram_mb", peak_vram),
        **_mem_stats("peak_ram_mb", peak_ram),
        **_mem_stats("torch_alloc_peak_mb", torch_peak),
    }

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "fcidump": fcidump,
            "shots": shots,
            "solver_name": solver_name,
            "solver_init_kwargs": solver_init_kwargs or {},
            "solver_call_kwargs": solver_call_kwargs or {},
            "subsample_size": subsample_size,
            "seed": seed,
            "symmetrize_spin": symmetrize_spin,
            "inject_hf": inject_hf,
            "warmup": warmup,
            "num_runs": num_runs,
            "nroots": nroots,
        }
        out.write_text(json.dumps({"config": config, "results": results}, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(CLI(run_single, as_positional=False), indent=2))
