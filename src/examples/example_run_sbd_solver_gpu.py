"""Smoke-test the SBDSolver's GPU (Thrust) backend, standalone.

This is a minimal, self-contained check that the in-process ``sbd-eigensolver``
package (backing ``create_solver("sbd", ...)``) is installed with a working GPU
backend and can diagonalize a tiny subspace on the GPU — WITHOUT running the
full SQD outer loop. It is meant to be run directly on a GPU node after:

    source build/energy_solvers/sbd_env.sh   # if you used build_sbd.sh
    uv sync --extra sbd

Run it in either diagonalization mode (add ``--compare-cpu`` to also cross-check
the GPU energy against the CPU backend on the same subspace):

    # TPB (Tensor-Product Basis, alpha (x) beta product subspace) — the default:
    uv run python src/examples/example_run_sbd_solver_gpu.py --fcidump FCIDUMP --mode tpb --compare-cpu

    # GDB (General-Determinant Basis, explicit parallel-pairs determinant list):
    uv run python src/examples/example_run_sbd_solver_gpu.py --fcidump FCIDUMP --mode gdb --compare-cpu

Pass ``--max-dim N`` to size the subspace for a PERFORMANCE run at a realistic
dimension instead of the tiny default — it caps the realized dimension (TPB: the
``n_alpha * n_beta`` product; GDB: the number of parallel pairs) and the run prints
the ``compute_energy`` wall time (and, with ``--compare-cpu``, the GPU/CPU speedup):

    # ~100k-determinant GPU timing, no CPU leg:
    uv run python src/examples/example_run_sbd_solver_gpu.py --fcidump FCIDUMP --mode gdb --max-dim 100000
    # a realistic dimension needs an FCIDUMP with a large enough active space:
    uv run python src/examples/example_run_sbd_solver_gpu.py --mode tpb --max-dim 1000000 \
        --fcidump /shared/path/to/FCIDUMP

Single-process (the commands above): SBDSolver detects the singleton launch and
configures MPI for an in-process run itself (see ``configure_singleton_mpi`` in
``sci_solvers/_mpi_singleton.py``), so a plain ``uv run python ...`` runs the GPU
backend in-process without ``mpirun`` — the supported single-process path. That
config sets ``ess_singleton_isolated=1``, which is what makes the singleton
``import sbd`` return instead of hang: without it Open MPI forks an ``orted``
runtime daemon and then blocks on a pipe read waiting for it, which on an HPC node
never completes. With the flag it runs truly isolated (no daemon), plus the
CUDA-aware UCX transport the GPU backend needs.

If a bare single-process GPU run ever still hangs at ``import sbd`` (e.g. a site
MPI that ignores the isolation flag), run the SAME command under a 1-rank
launcher, which supplies its own runtime daemon so no singleton ``orted`` fork
happens (``configure_singleton_mpi`` detects the launcher and no-ops, leaving the
transport to it):

    mpirun -np 1 uv run python src/examples/example_run_sbd_solver_gpu.py --fcidump FCIDUMP --mode tpb --compare-cpu

Multi-node (one GPU rank per node): launch the SAME script under ``mpirun``/
``srun`` — the launcher's rank env vars are detected, the singleton config is
skipped, and ``sbd`` uses the real cluster fabric. The FCIDUMP path must be
visible to every rank (shared filesystem), so pass ``--fcidump`` explicitly:

    # Open MPI: 4 nodes, 1 rank (1 GPU) each — 4 ranks total.
    mpirun -np 4 --map-by ppr:1:node \
        uv run python src/examples/example_run_sbd_solver_gpu.py \
        --mode gdb --fcidump /shared/path/to/FCIDUMP

    # Slurm equivalent inside an sbatch/salloc allocation (e.g. -N4 --ntasks-per-node=1):
    srun --ntasks-per-node=1 --gpus-per-task=1 \
        uv run python src/examples/example_run_sbd_solver_gpu.py \
        --mode gdb --fcidump /shared/path/to/FCIDUMP

CUDA-aware MPI is REQUIRED for every GPU run. The Thrust backend passes CUDA
*device* pointers into MPI collectives (e.g. the Davidson Allreduce), so MPI's
transport must be CUDA-aware (UCX built with CUDA); a non-CUDA-aware transport
treats the device buffer as host memory and SEGFAULTs in
``mca_coll_cuda_allreduce`` / ``opal_datatype_copy_content_same_ddt``. In the
single-process case ``configure_singleton_mpi`` selects the CUDA-aware UCX
transport for ``device="gpu"``; under ``mpirun``/``srun`` that singleton path is
skipped, so the CUDA-aware transport must come from the site MPI / launcher
instead. If a multi-rank GPU run crashes in that Allreduce, force UCX explicitly
(matching what the singleton path does), e.g.:

    mpirun -np 4 --map-by ppr:1:node \
        --mca pml ucx --mca btl '^vader,tcp' \
        uv run python src/examples/example_run_sbd_solver_gpu.py \
        --mode gdb --fcidump /shared/path/to/FCIDUMP

Note ``--compare-cpu`` is a single-process convenience check; skip it under a
multi-rank ``mpirun`` (only rank 0 returns the energy, so the cross-check is
meaningless on the other ranks). It is fine single-process or at ``mpirun -np 1``.

What it checks, in order:
  1. ``import sbd`` succeeds and reports ``'gpu'`` among ``available_backends()``
     — i.e. the Thrust GPU backend actually compiled. If not, it says so and
     exits non-zero (a CPU-only build reports only ``['cpu']``).
  2. Builds ``create_solver("sbd", device="gpu", mode="tpb")`` on a small
     Hamiltonian and runs one ``compute_energy`` on a tiny CI subspace on the
     GPU, printing the electronic + total energy.
  3. Optionally (``--compare-cpu``) runs the same subspace on ``device="cpu"``
     and asserts the two energies agree to Davidson tolerance, confirming the
     GPU path is not just running but returning the right number.

Exit code is 0 only if the GPU backend was present AND the diagonalization ran
(and, with ``--compare-cpu``, matched CPU), so this doubles as a CI/pytest-free
gate you can wire into a job script.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import argparse
import sys
import time
from pathlib import Path

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData


def main() -> int:
    """Run the GPU smoke test; return a process exit code (0 = ok)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fcidump",
        required=True,
        help="Path to an FCIDUMP.",
    )
    parser.add_argument(
        "--mode",
        choices=("tpb", "gdb"),
        default="tpb",
        help="SBD diagonalization mode (default: tpb).",
    )
    parser.add_argument(
        "--n-dets",
        type=int,
        default=16,
        help="Determinants per spin in the tiny smoke-test subspace (default: 16). "
        "Ignored when --max-dim is given.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Cap the realized subspace DIMENSION for a performance run at a realistic "
        "size. TPB: n_alpha * n_beta product dimension (per-spin sizes ~sqrt(max_dim)). "
        "GDB: number of parallel (alpha, beta) determinant pairs. When set it overrides "
        "--n-dets. The actual dimension may be smaller if the excitation manifold (up to "
        "triples from HF) cannot supply enough distinct patterns.",
    )
    parser.add_argument(
        "--compare-cpu",
        action="store_true",
        help="Also run on CPU and assert the GPU energy matches to tolerance. Skip for "
        "large --max-dim performance runs (the CPU leg dominates the wall time).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Energy match tolerance for --compare-cpu (default: 1e-6 Ha).",
    )
    args = parser.parse_args()

    # 1. Confirm the GPU backend actually compiled into this sbd install. -----
    # This diagnostic `import sbd` triggers MPI_Init, and the transport MCA vars
    # are read ONLY at MPI_Init — so the CUDA-aware (UCX) transport the GPU backend
    # needs must be selected NOW, before the import. Importing `create_solver`
    # above already ran the CPU-default singleton config at `sci_solvers` package
    # import; without this call the diagnostic import below would lock MPI onto the
    # CPU (ob1/self) transport, and the later `create_solver("sbd", device="gpu")`
    # could not switch it (MPI is already initialized), so the GPU Davidson
    # Allreduce would segfault on a device pointer. A no-op under mpirun/srun (the
    # launcher's rank env vars are detected and the launcher's transport is used).
    from sci_solvers._mpi_singleton import configure_singleton_mpi

    configure_singleton_mpi(device="gpu")
    try:
        import sbd

    except ImportError as exc:
        print(f"FAIL: `import sbd` failed — install with `uv sync --extra sbd`: {exc}")
        return 2

    # In essence this example runs unit test code
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
    from sbd_equivalence_utils import build_gpu_smoke_subspace

    backends = list(sbd.available_backends())
    print(f"sbd.available_backends() = {backends}")
    if "gpu" not in backends:
        print(
            "FAIL: no 'gpu' backend in this sbd build (CPU-only). Rebuild with the "
            "NVIDIA HPC SDK on PATH so the Thrust GPU backend compiles:\n"
            "  ./scripts/energy_solvers/build_sbd.sh  # detects nvc++ -> SBD_BUILD_BACKEND=both\n"
            "  source build/energy_solvers/sbd_env.sh && uv sync --extra sbd"
        )
        return 1

    # 2. Build the GPU solver and run one diagonalization on the subspace. -----
    fcidump = args.fcidump
    print(f"Hamiltonian: {fcidump}")
    ham = HamiltonianData.from_fcidump(fcidump)
    norb = ham.num_orbitals
    nelec = (ham.num_elec_a, ham.num_elec_b)
    batches, dim = build_gpu_smoke_subspace(
        norb, nelec, mode=args.mode, n_dets=args.n_dets, max_dim=args.max_dim
    )
    strs_a, strs_b = batches[0]
    if args.mode == "tpb":
        # Independent per-spin sets; basis is their Cartesian product.
        subspace_desc = f"{len(strs_a)} x {len(strs_b)} = {dim} product determinants"
    else:
        # Parallel length-N pair list; index i = determinant (alpha_i, beta_i).
        subspace_desc = f"{dim} parallel (alpha, beta) pairs"
    if args.max_dim is not None and dim < args.max_dim:
        subspace_desc += f" (requested max_dim={args.max_dim}; excitation manifold capped it)"
    print(f"norb={norb} nelec={nelec} | subspace: {subspace_desc} (mode={args.mode})")

    gpu_solver = create_solver(
        "sbd",
        symmetrize_spin=True,
        hamiltonian=ham,
        mode=args.mode,
        device="gpu",  # Thrust GPU backend
    )
    t0 = time.perf_counter()
    (gpu_result,) = gpu_solver.compute_energy(ci_strs_batches=batches)
    gpu_dt = time.perf_counter() - t0
    gpu_e = float(gpu_result.energy)
    print(f"GPU electronic energy: {gpu_e:.8f} Ha")
    print(f"GPU total energy:      {gpu_e + ham.e_nuc:.8f} Ha")
    print(f"GPU compute_energy wall time: {gpu_dt:.3f} s  (dim={dim}, mode={args.mode})")

    # 3. Optional CPU cross-check: same subspace, device='cpu'. ---------------
    if args.compare_cpu:
        cpu_solver = create_solver(
            "sbd",
            symmetrize_spin=True,
            hamiltonian=ham,
            mode=args.mode,
            device="cpu",
        )
        t0 = time.perf_counter()
        (cpu_result,) = cpu_solver.compute_energy(ci_strs_batches=batches)
        cpu_dt = time.perf_counter() - t0
        cpu_e = float(cpu_result.energy)
        diff = abs(gpu_e - cpu_e)
        print(f"CPU electronic energy: {cpu_e:.8f} Ha  |  |GPU-CPU| = {diff:.2e} Ha")
        speedup = f"{cpu_dt / gpu_dt:.1f}x" if gpu_dt > 0 else "n/a"
        print(f"CPU compute_energy wall time: {cpu_dt:.3f} s  |  GPU speedup: {speedup}")
        if diff > args.tol:
            print(f"FAIL: GPU and CPU energies differ by {diff:.2e} > tol {args.tol:.2e}")
            return 1
        print(f"OK: GPU matches CPU within {args.tol:.2e} Ha")

    print("OK: SBD GPU backend ran successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
