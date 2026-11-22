"""Shared utilities for SBD/TorchSCI equivalence tests.

The in-process SBD bindings retain native determinant-layout state across solves.
In particular, systems whose determinants require different numbers of SBD words
cannot reliably be run sequentially in the same Python interpreter.

On an NVHPC OpenMPI stack, SBD also requires an explicit one-rank MPI
launcher; singleton MPI initialization is not reliable.

``isolated_sbd_test`` therefore executes each new-SBD test case in its own

    mpirun -np 1 ...

Python process. A successful child exits directly with ``os._exit(0)`` to avoid
the broken native-extension teardown observed on this SBD/OpenMPI build.

Torch-only tests run normally.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
import pathlib
import sys

# The standalone SBD example deliberately initializes SBD before importing
# the rest of the numerical stack. Do the same in an isolated worker.
if "--sbd-worker" in sys.argv:
    worker_idx = sys.argv.index("--sbd-worker")
    input_path = sys.argv[worker_idx + 1]

    with pathlib.Path(input_path).open() as f:
        _worker_payload = json.load(f)

    from sci_solvers._mpi_singleton import configure_singleton_mpi

    configure_singleton_mpi(device=_worker_payload["device"])

    import sbd  # noqa: F401  # initialize native SBD/MPI stack first

import functools
import itertools
import os
import pathlib
import shutil
import subprocess
import time
from typing import NamedTuple

import numpy as np
import pytest

from sci_solvers.eigen_problem import EigenProblem, PairCIProblem
from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

# --------------------------------------------------------------------------- #
# Solver settings
# --------------------------------------------------------------------------- #

TORCH_KWARGS = dict(
    max_davidson_iters=200,
    davidson_block_size=30,
    davidson_tolerance=1e-10,
)

SBD_CONFIG = {
    "eps": 1e-10,
    "max_it": 200,
}

ATOL_SOLVERS = 1e-6
ATOL_TORCH_DENSE = 1e-8

SYMMETRIZE_SPIN = False


# --------------------------------------------------------------------------- #
# Systems / devices
# --------------------------------------------------------------------------- #

_DEVICE_MAP = {
    "cpu": ("cpu", "cpu"),
    "gpu": ("cuda", "gpu"),
}

DEVICES = list(_DEVICE_MAP)

H4_SYSTEM = "h4"
BENZOIC_ACID_SMALL_SYSTEM = "benzoic_acid_k12"
BENZOIC_ACID_DEEP_SYSTEM = "benzoic_acid_k50"


class System(NamedTuple):
    """A Hamiltonian plus the per-spin CI-string sets defining one test subspace."""

    label: str
    hcore: np.ndarray
    eri: np.ndarray
    norb: int
    nelec: tuple[int, int]
    strs_a: np.ndarray
    strs_b: np.ndarray


def _torch_device(device_label: str) -> str:
    import torch

    if device_label == "gpu" and not torch.cuda.is_available():
        pytest.skip("no CUDA device available for the GPU leg")
    return _DEVICE_MAP[device_label][0]


def _sbd_device(device_label: str) -> str:
    sbd = pytest.importorskip(
        "sbd",
        reason="the in-process `sbd` package is not installed",
    )

    if device_label == "gpu":
        backends = list(sbd.available_backends())
        if "gpu" not in backends:
            pytest.skip(f"this sbd build has no GPU backend (available: {backends})")

    return _DEVICE_MAP[device_label][1]


# --------------------------------------------------------------------------- #
# System construction
# --------------------------------------------------------------------------- #


def _all_strings(norb: int, nelec: int) -> np.ndarray:
    """Every ``norb``-bit CI string with exactly ``nelec`` bits set, ascending."""
    return np.array(
        sorted(sum(1 << i for i in c) for c in itertools.combinations(range(norb), nelec)),
        dtype=np.int64,
    )


@functools.lru_cache(maxsize=None)
def _benzoic_acid() -> tuple[HamiltonianData, list[int]]:
    """The bundled benzoic acid (dataset ID 182) Hamiltonian and its HF + singles strings."""
    from torch_sci_solver import _paths

    ham = HamiltonianData.from_fcidump(_paths.fcidump_path("benzoic_acid"))
    return ham, _paths.hf_and_singles(ham.num_orbitals, ham.num_elec_a)


@functools.lru_cache(maxsize=None)
def system(label: str) -> System:
    """Build and cache one named chemistry system."""

    if label == H4_SYSTEM:
        from pyscf import ao2mo, gto, scf

        mol = gto.M(
            atom=("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"),
            basis="sto-3g",
            verbose=0,
        )

        mf = scf.RHF(mol).run(verbose=0)

        norb = int(mol.nao)
        hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.restore(
            1,
            ao2mo.kernel(mol, mf.mo_coeff),
            norb,
        )
        na, nb = (int(n) for n in mol.nelec)

        return System(
            label,
            hcore,
            eri,
            norb,
            (na, nb),
            _all_strings(norb, na),
            _all_strings(norb, nb),
        )

    if label.startswith("benzoic_acid_k"):
        k = int(label.removeprefix("benzoic_acid_k"))

        ham, hf_singles = _benzoic_acid()
        strs = np.array(hf_singles[:k], dtype=np.int64)

        return System(
            label,
            ham.hcore,
            ham.eri,
            ham.num_orbitals,
            (ham.num_elec_a, ham.num_elec_b),
            strs,
            strs.copy(),
        )

    raise ValueError(f"unknown system label: {label!r}")


# --------------------------------------------------------------------------- #
# Subspaces / dense references
# --------------------------------------------------------------------------- #


def _product_as_pairs(
    strs_a: np.ndarray,
    strs_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an alpha x beta product into parallel determinant-pair lists."""
    grid_a, grid_b = np.meshgrid(
        strs_a,
        strs_b,
        indexing="ij",
    )
    return grid_a.ravel(), grid_b.ravel()


def _nonproduct_pairs(
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic genuinely non-product pair subspace."""
    sysm = system(label)

    pairs_a, pairs_b = _product_as_pairs(
        sysm.strs_a,
        sysm.strs_b,
    )

    rng = np.random.default_rng(0)
    keep = np.sort(
        rng.choice(
            len(pairs_a),
            size=2 * len(pairs_a) // 3,
            replace=False,
        )
    )

    return pairs_a[keep], pairs_b[keep]


def _assert_not_a_product(
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
) -> None:
    n_product = len(np.unique(pairs_a)) * len(np.unique(pairs_b))

    assert len(pairs_a) < n_product, (
        f"pair list is a full {n_product}-determinant product, "
        "so it does not exercise the non-product GDB path"
    )


def _occ_patterns(norb: int, n_elec: int, count: int) -> np.ndarray:
    """Up to ``count`` occupation integers with exactly ``n_elec`` bits set.

    Enumerates from the HF determinant (lowest ``n_elec`` orbitals filled) by
    low-order excitations: the HF reference, then single excitations moving one
    electron from an occupied orbital into a virtual one, then doubles, then
    triples, until ``count`` distinct patterns are collected (or the excitation
    manifold is exhausted, whichever comes first — near a closed shell there may
    be very few). This is combinatorial in the excitation level, NOT a scan over
    all integers — the latter is catastrophic near half-filling (for e.g.
    norb=36, n_elec=27 almost no integer has the right popcount, so a linear walk
    effectively never finds ``count``). Only meant to exercise the
    diagonalization path at a chosen size, not to converge a physical energy.
    """
    if n_elec < 0 or n_elec > norb:
        raise ValueError(f"n_elec={n_elec} out of range for norb={norb}")

    hf = (1 << n_elec) - 1  # lowest n_elec orbitals filled
    occ = list(range(n_elec))  # occupied orbital indices in HF
    virt = list(range(n_elec, norb))  # virtual orbital indices in HF
    seen: dict[int, None] = {hf: None}  # dict preserves insertion order, dedups

    def _excite(holes: tuple[int, ...], parts: tuple[int, ...]) -> None:
        det = hf
        for i in holes:
            det &= ~(1 << i)
        for a in parts:
            det |= 1 << a
        seen[det] = None

    # Singles, then doubles, then triples: move k electrons occ -> virt. Stop as
    # soon as we have enough distinct patterns; each level is generated only if
    # the previous ones did not already reach `count`.
    for level in (1, 2, 3):
        if len(seen) >= count:
            break
        for holes in itertools.combinations(occ, level):
            if len(seen) >= count:
                break
            for parts in itertools.combinations(virt, level):
                if len(seen) >= count:
                    break
                _excite(holes, parts)
    return np.array(list(seen)[:count], dtype=np.int64)


def build_gpu_smoke_subspace(
    norb: int, nelec: tuple[int, int], mode: str, n_dets: int, max_dim: int | None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """A deterministic ``(strs_a, strs_b)`` batch matching ``mode``'s contract.

    Returns ``(batches, dim)`` where ``dim`` is the realized subspace dimension
    (what the diagonalizer actually sees). ``max_dim`` caps that dimension so a
    performance run can target a realistic size instead of the tiny default; when
    ``None`` the size is driven by ``n_dets`` (the smoke-test default).

    The two modes interpret ``(strs_a, strs_b)`` differently — the SAME contract
    as ``torch_sci_solver``'s ``mode`` (see ``qiskit.solve_fermion``'s docstring)
    — so the subspace is built to match, or ``mode="gdb"`` either raises (unequal
    lengths) or silently diagonalizes a different space:

    - ``tpb`` (Tensor-Product Basis): ``strs_a`` and ``strs_b`` are independent
      per-spin unique sets; the basis is their Cartesian product, so the dimension
      is ``len(strs_a) * len(strs_b)``. Sizes are chosen near ``sqrt(max_dim)`` per
      spin so the product lands at or just under ``max_dim`` (clamped to what the
      excitation manifold can supply).
    - ``gdb`` (General-Determinant Basis): ``strs_a`` and ``strs_b`` are
      **parallel length-N pair lists** — index ``i`` is determinant
      ``(alpha_i, beta_i)``, NOT an outer product — so the dimension is ``N``.
      We enumerate distinct ``(alpha, beta)`` pairs in row-major order up to the
      target, satisfying SBDSolver's ``len(strs_a) == len(strs_b)`` contract.
    """
    if mode == "tpb":
        # Target a per-spin count so n_a * n_b ~ max_dim (or n_dets each by default).
        per_spin = n_dets if max_dim is None else max(1, int(np.ceil(np.sqrt(max_dim))))
        strs_a = _occ_patterns(norb, nelec[0], per_spin)
        strs_b = _occ_patterns(norb, nelec[1], per_spin)
        # If a spin can't supply `per_spin` patterns (near a closed shell), grow
        # the other spin to recover the product dimension where possible.
        if max_dim is not None and len(strs_a) * len(strs_b) < max_dim:
            if len(strs_a) < per_spin and len(strs_b) == per_spin:
                strs_b = _occ_patterns(
                    norb, nelec[1], (max_dim + len(strs_a) - 1) // max(1, len(strs_a))
                )
            elif len(strs_b) < per_spin and len(strs_a) == per_spin:
                strs_a = _occ_patterns(
                    norb, nelec[0], (max_dim + len(strs_b) - 1) // max(1, len(strs_b))
                )
        dim = len(strs_a) * len(strs_b)
        return [(strs_a, strs_b)], dim

    # gdb: build a genuine parallel-pairs list. Take a per-spin set of side length
    # ceil(sqrt(target)), then enumerate distinct pairs (alpha_i, beta_j) in
    # row-major order up to `target` pairs, so strs_a[k]/strs_b[k] name determinant
    # k. Equal length by construction.
    target = n_dets if max_dim is None else max(1, max_dim)
    side = max(1, int(np.ceil(np.sqrt(target))))
    a_set = _occ_patterns(norb, nelec[0], side)
    b_set = _occ_patterns(norb, nelec[1], side)
    pair_a: list[int] = []
    pair_b: list[int] = []
    for a in a_set:
        for b in b_set:
            if len(pair_a) >= target:
                break
            pair_a.append(int(a))
            pair_b.append(int(b))
        if len(pair_a) >= target:
            break
    return [(np.array(pair_a, dtype=np.int64), np.array(pair_b, dtype=np.int64))], len(pair_a)


@functools.lru_cache(maxsize=None)
def _dense_product_energy(label: str) -> float:
    sysm = system(label)

    problem = EigenProblem(
        sysm.strs_a,
        sysm.strs_b,
        sysm.hcore,
        sysm.eri,
        sysm.norb,
        sysm.nelec,
    )

    return float(np.linalg.eigvalsh(problem.sparse().toarray())[0])


@functools.lru_cache(maxsize=None)
def _dense_nonproduct_pair_energy(label: str) -> float:
    sysm = system(label)

    pairs_a, pairs_b = _nonproduct_pairs(label)

    problem = PairCIProblem(
        pairs_a,
        pairs_b,
        sysm.hcore,
        sysm.eri,
        sysm.norb,
        sysm.nelec,
    )

    return float(np.linalg.eigvalsh(problem.sparse().toarray())[0])


# --------------------------------------------------------------------------- #
# Solver execution
# --------------------------------------------------------------------------- #


def _ground_state_energy(
    solver,
    sysm: System,
    batch: tuple[np.ndarray, np.ndarray],
) -> float:
    (states,) = solver.sci_solver_callable_eigen_states(
        [batch],
        sysm.hcore,
        sysm.eri,
        sysm.norb,
        sysm.nelec,
    )

    energy = float(states.states[0].energy)

    assert energy != -1.0, f"{solver!r} did not converge on {sysm.label}"

    return energy


def _torch_energy(
    sysm: System,
    batch: tuple[np.ndarray, np.ndarray],
    *,
    device_label: str,
    mode: str,
) -> float:
    solver = create_solver(
        "torch_sci",
        symmetrize_spin=SYMMETRIZE_SPIN,
        device=_torch_device(device_label),
        mode=mode,
        **TORCH_KWARGS,
    )

    return _ground_state_energy(
        solver,
        sysm,
        batch,
    )


def _sbd_energy(
    sysm: System,
    batch: tuple[np.ndarray, np.ndarray],
    *,
    device_label: str,
    mode: str,
) -> float:
    ham = HamiltonianData(
        hcore=sysm.hcore,
        eri=sysm.eri,
        e_nuc=0.0,
        num_orbitals=sysm.norb,
        num_elec_a=sysm.nelec[0],
        num_elec_b=sysm.nelec[1],
    )

    solver = create_solver(
        "sbd",
        symmetrize_spin=SYMMETRIZE_SPIN,
        hamiltonian=ham,
        mode=mode,
        device=_sbd_device(device_label),
        sbd_config=SBD_CONFIG,
    )

    (result,) = solver.compute_energy(
        ci_strs_batches=[batch],
    )
    return float(result.energy)


# --------------------------------------------------------------------------- #
# Shared assertions
# --------------------------------------------------------------------------- #


def assert_tpb_agrees(sbd_name: str, label: str, device_label: str) -> None:  # noqa: D103
    sysm = system(label)
    batch = (sysm.strs_a, sysm.strs_b)

    t0 = time.perf_counter()
    if sbd_name == "sbd":
        e_sbd = _isolated_sbd_energy(
            label,
            device_label,
            "tpb",
        )
    else:
        raise ValueError(f"unknown SBD implementation: {sbd_name!r}")
    sbd_seconds = time.perf_counter() - t0

    # These now execute in the ordinary pytest process.
    t0 = time.perf_counter()
    if device_label == "gpu":
        e_torch = _isolated_torch_energy(
            label,
            "gpu",
            "tpb",
        )
    else:
        e_torch = _torch_energy(
            sysm,
            batch,
            device_label="cpu",
            mode="tpb",
        )

    torch_seconds = time.perf_counter() - t0
    e_dense = _dense_product_energy(label)

    print(f"[{label}/tpb/{device_label}] {sbd_name}={sbd_seconds:.3f}s torch={torch_seconds:.3f}s")

    assert e_torch == pytest.approx(e_dense, abs=ATOL_TORCH_DENSE)
    assert e_sbd == pytest.approx(e_dense, abs=ATOL_SOLVERS), (
        e_dense,
        e_sbd,
        sbd_name,
        label,
        device_label,
    )
    assert e_sbd == pytest.approx(e_torch, abs=ATOL_SOLVERS), (
        e_dense,
        e_sbd,
        sbd_name,
        label,
        device_label,
    )


def assert_gdb_full_product_agrees(label: str, device_label: str) -> None:  # noqa: D103
    sysm = system(label)
    batch = _product_as_pairs(sysm.strs_a, sysm.strs_b)

    t0 = time.perf_counter()
    e_sbd = _isolated_sbd_energy(
        label,
        device_label,
        "gdb_full",
    )
    sbd_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_torch = (
        _isolated_torch_energy(label, "gpu", "gdb_full")
        if device_label == "gpu"
        else _torch_energy(sysm, batch, device_label="cpu", mode="gdb")
    )

    torch_seconds = time.perf_counter() - t0
    e_dense = _dense_product_energy(label)

    print(f"[{label}/gdb_full/{device_label}] sbd={sbd_seconds:.3f}s torch={torch_seconds:.3f}s")

    assert e_torch == pytest.approx(e_dense, abs=ATOL_TORCH_DENSE)
    assert e_sbd == pytest.approx(e_dense, abs=ATOL_SOLVERS)
    assert e_sbd == pytest.approx(e_torch, abs=ATOL_SOLVERS)


def assert_gdb_nonproduct_agrees(label: str, device_label: str) -> None:  # noqa: D103
    sysm = system(label)

    pairs_a, pairs_b = _nonproduct_pairs(label)
    _assert_not_a_product(pairs_a, pairs_b)
    batch = (pairs_a, pairs_b)

    t0 = time.perf_counter()
    e_sbd = _isolated_sbd_energy(
        label,
        device_label,
        "gdb_nonproduct",
    )
    sbd_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_torch = (
        _isolated_torch_energy(label, "gpu", "gdb_nonproduct")
        if device_label == "gpu"
        else _torch_energy(sysm, batch, device_label="cpu", mode="gdb")
    )
    torch_seconds = time.perf_counter() - t0
    e_dense = _dense_nonproduct_pair_energy(label)

    print(
        f"[{label}/gdb_nonproduct/{device_label}] sbd={sbd_seconds:.3f}s torch={torch_seconds:.3f}s"
    )

    assert e_torch == pytest.approx(e_dense, abs=ATOL_TORCH_DENSE), (
        e_dense,
        e_torch,
        label,
        device_label,
    )
    assert e_sbd == pytest.approx(e_dense, abs=ATOL_SOLVERS), (
        e_dense,
        e_torch,
        label,
        device_label,
    )
    assert e_sbd == pytest.approx(e_torch, abs=ATOL_SOLVERS), (
        e_dense,
        e_torch,
        label,
        device_label,
    )


def assert_torch_matches_dense(
    label: str,
    device_label: str,
    mode: str,
) -> None:
    """TorchSCISolver == dense reference, independent of SBD."""
    sysm = system(label)

    if mode == "tpb":
        batch = (sysm.strs_a, sysm.strs_b)
        e_dense = _dense_product_energy(label)

    elif mode == "gdb":
        pairs_a, pairs_b = _nonproduct_pairs(label)
        _assert_not_a_product(pairs_a, pairs_b)

        batch = (pairs_a, pairs_b)
        e_dense = _dense_nonproduct_pair_energy(label)

    else:
        raise ValueError(f"unknown mode: {mode!r}")

    t0 = time.perf_counter()
    e_torch = _torch_energy(sysm, batch, device_label=device_label, mode=mode)
    torch_seconds = time.perf_counter() - t0

    print(f"[{label}/{mode}/{device_label}] torch={torch_seconds:.3f}s")

    assert e_torch == pytest.approx(e_dense, abs=ATOL_TORCH_DENSE)


def _sbd_device(device_label: str) -> str:
    sbd = pytest.importorskip(
        "sbd",
        reason="the in-process `sbd` package is not installed",
    )

    if device_label == "gpu":
        backends = list(sbd.available_backends())
        if "gpu" not in backends:
            pytest.skip(f"this sbd build has no GPU backend (available: {backends})")

    return _DEVICE_MAP[device_label][1]


def _launcher_env() -> dict[str, str]:
    """The parent's environment minus the singleton MCA vars ``sci_solvers`` set in it.

    Importing ``sci_solvers`` runs ``configure_singleton_mpi``, which exports
    ``OMPI_MCA_pml=ob1`` / ``OMPI_MCA_btl=self,vader`` / ``ess_singleton_isolated`` into
    this process so a bare ``import sbd`` here does not hang. Those are for a
    *singleton* ``MPI_Init`` only: inherited by the ``mpirun`` launch below they
    override its own transport choice and the worker exits 1 without an error message.
    The worker configures its own MPI (under ``mpirun``, ``configure_singleton_mpi``
    is a no-op), so drop them. A value the user exported is kept.
    """
    from sci_solvers import _mpi_singleton

    env = dict(os.environ)
    for key in _mpi_singleton._OWN_KEYS:
        env.pop(key, None)
    return env


def _isolated_sbd_energy(
    label: str,
    device_label: str,
    subspace: str,
) -> float:
    import json
    import tempfile

    # Skip (not fail) when the native SBD stack the worker needs is absent: the
    # worker imports `sbd` at startup, so probe the same requirements here in the
    # parent before spawning it. `_sbd_device` runs `pytest.importorskip("sbd")`
    # and skips when a GPU leg has no GPU backend.
    _sbd_device(device_label)

    if shutil.which("mpirun") is None:
        pytest.skip("`mpirun` is required for SBD tests")

    with tempfile.TemporaryDirectory(prefix="sbd_test_") as tmp:
        tmp = pathlib.Path(tmp)

        payload_path = tmp / "input.json"
        output_path = tmp / "energy.txt"

        payload_path.write_text(
            json.dumps({
                "label": label,
                "device": device_label,
                "subspace": subspace,
            })
        )

        command = [
            "mpirun",
            "-np",
            "1",
            "--mca",
            "pml",
            "ucx",
            "--mca",
            "btl",
            "^vader,tcp",
            "--mca",
            "coll_cuda_enable",
            "0",
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--sbd-worker",
            str(payload_path),
            str(output_path),
        ]

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=_launcher_env(),
        )

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            pytest.fail(
                f"SBD subprocess exited with code {result.returncode}:\n{result.stderr[-2000:]}",
                pytrace=False,
            )

        if not output_path.exists():
            pytest.fail(
                "SBD subprocess completed without returning an energy",
                pytrace=False,
            )

        return float(output_path.read_text())


def _isolated_torch_energy(
    label: str,
    device_label: str,
    subspace: str,
) -> float:
    import json
    import tempfile

    # The worker runs Torch on `device_label`; skip in the parent when a GPU leg
    # has no CUDA device (matching `_torch_device`) so it doesn't hard-fail there.
    _torch_device(device_label)

    with tempfile.TemporaryDirectory(prefix="torch_test_") as tmp:
        tmp = pathlib.Path(tmp)

        payload_path = tmp / "input.json"
        output_path = tmp / "energy.txt"

        payload_path.write_text(
            json.dumps({
                "label": label,
                "device": device_label,
                "subspace": subspace,
            })
        )

        command = [
            sys.executable,
            str(pathlib.Path(__file__).resolve()),
            "--torch-worker",
            str(payload_path),
            str(output_path),
        ]

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            pytest.fail(
                f"Torch subprocess exited with code {result.returncode}",
                pytrace=False,
            )

        return float(output_path.read_text())


def _run_sbd_worker() -> None:
    import json

    payload_path = pathlib.Path(sys.argv[2])
    output_path = pathlib.Path(sys.argv[3])

    payload = json.loads(payload_path.read_text())

    sysm = system(payload["label"])

    if payload["subspace"] == "tpb":
        batch = (sysm.strs_a, sysm.strs_b)
        mode = "tpb"

    elif payload["subspace"] == "gdb_full":
        batch = _product_as_pairs(sysm.strs_a, sysm.strs_b)
        mode = "gdb"

    elif payload["subspace"] == "gdb_nonproduct":
        batch = _nonproduct_pairs(payload["label"])
        mode = "gdb"

    else:
        raise ValueError(payload["subspace"])

    energy = _sbd_energy(
        sysm,
        batch,
        device_label=payload["device"],
        mode=mode,
    )

    output_path.write_text(repr(energy))

    # Satisfy mpirun, but avoid normal SBD/native-extension teardown.
    from mpi4py import MPI

    if MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.Finalize()

    os._exit(0)


def _run_torch_worker() -> None:
    import json

    payload_path = pathlib.Path(sys.argv[2])
    output_path = pathlib.Path(sys.argv[3])

    payload = json.loads(payload_path.read_text())

    sysm = system(payload["label"])

    if payload["subspace"] == "tpb":
        batch = (sysm.strs_a, sysm.strs_b)
        mode = "tpb"

    elif payload["subspace"] == "gdb_full":
        batch = _product_as_pairs(sysm.strs_a, sysm.strs_b)
        mode = "gdb"

    elif payload["subspace"] == "gdb_nonproduct":
        batch = _nonproduct_pairs(payload["label"])
        mode = "gdb"

    else:
        raise ValueError(payload["subspace"])

    energy = _torch_energy(
        sysm,
        batch,
        device_label=payload["device"],
        mode=mode,
    )

    output_path.write_text(repr(energy))
    os._exit(0)


if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == "--sbd-worker":
        _run_sbd_worker()
    elif sys.argv[1] == "--torch-worker":
        _run_torch_worker()
