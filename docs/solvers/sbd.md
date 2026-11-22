# `sbd` — RIKEN's SBD diagonalizer (in-process package)

`create_solver("sbd", ...)` → `SBDSolver`
(`src/sci_solvers/energy_solvers/sbd_solver.py`). Wraps RIKEN's open-source
**`sbd-eigensolver`** package (<https://github.com/Qiskit/sbd-eigensolver-python>,
`import sbd`, Apache-2.0) — a pybind11 **in-process** API, **no subprocess and no
`mpirun`**. See [arXiv:2601.16637](https://arxiv.org/abs/2601.16637) for the published
benchmarks of its GPU (Thrust) backend: up to ~40× faster than CPU execution on SQD workloads.

`sbd` is the **GPU-capable native solver**: built with the NVIDIA HPC SDK (Thrust),
its Davidson matrix-vector product runs on the GPU. It returns the **ground state
only**.

## Features

| feature | support |
|---|---|
| complex integrals | no |
| multi-root | **no** — `supports_multiroot = False`, one ground state per batch |
| spin projection | none — rejects any non-`None` `spin_sq` |
| GPU | **yes** (`device="gpu"`, Thrust) — also `"gpu-omp"`, `"auto"`, `"cpu"` |
| 1-RDM / 2-RDM | no — returns per-orbital **occupancies** (and amplitudes) |
| basis mode | **TPB** (default) and **GDB** (`mode="tpb"|"gdb"`) |

## Basis modes: TPB vs GDB

`SBDSolver` exposes both of the package's diagonalization modes through the `mode`
constructor argument — the **same `(strs_a, strs_b)` contract as `torch_sci`**:

- **`mode="tpb"`** — *tensor-product basis* (default). `(strs_a, strs_b)` are two
  independent per-spin sets and the subspace is their full **Cartesian product**,
  dimension `len(strs_a) * len(strs_b)`. This delegates directly to the package's
  `sbd.sbd_solver.solve_sci_batch`.
- **`mode="gdb"`** — *general-determinant basis*. `(strs_a, strs_b)` are **parallel
  length-N pair lists**: index `i` is the single determinant `(strs_a[i], strs_b[i])`,
  dimension N. `len(strs_a) == len(strs_b)` is required (it raises otherwise).

The same measurement data therefore defines very different subspaces. As with
`torch_sci`, GDB is deliberately **not** a drop-in for qiskit-addon-sqd's per-spin
recovery loop: feeding a product-style `(strs_a, strs_b)` to `mode="gdb"` silently
diagonalizes a different subspace. See [the torch_sci page](torch_sci.md) and the repo
README's *TPB vs GDB* section for the full contract.

## Installation

`sbd` has **no PyPI wheel** — it compiles at install against local MPI/BLAS (and, when
an NVIDIA HPC SDK is on PATH, the Thrust GPU backend). It is an **optional extra**, so a
plain `uv sync` never builds it and the base install can never break.

```bash
# 1. Prepare the native toolchain and write build/energy_solvers/sbd_env.sh.
#    Auto-detects nvc++ and, if present, enables the GPU (Thrust) backend for one GPU
#    architecture: this node's GPU, or SBD_GPU_ARCH (e.g. cc80, cc90) on a node without one.
bash scripts/energy_solvers/build_sbd.sh

# 2. Source the discovery env vars (MPI_HOME, BLAS_*, OPAL_PREFIX, …) and build.
source build/energy_solvers/sbd_env.sh
uv sync --extra sbd
```

On a GPU node use `uv sync --extra sbd-gpu` instead. It also installs a torch built on a CUDA 12
runtime (PyTorch cu128 index, Linux x86_64), which the GPU backend needs: with a CUDA-13 torch
(today's PyPI default) loaded first, sbd can silently return `Energy = 0`, and `SBDSolver` emits a
`RuntimeWarning` at construction (the run is not stopped). Re-run `uv sync --extra sbd-gpu` (not `uv pip install`) when adding the
extra to an existing environment, so the CUDA 13 packages are removed.

## Usage

```python
from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

ham = HamiltonianData.from_fcidump("FCIDUMP")

solver = create_solver(
    "sbd",
    symmetrize_spin=True,
    hamiltonian=ham,
    mode="tpb",  # or "gdb"
    device="gpu",  # "cpu" (default), "gpu" (Thrust), "gpu-omp", "auto"
    sbd_config=None,  # optional dict of SBD tunables, e.g. {"eps": 1e-8, "max_it": 100}
    fcidump_path=None,  # if set, sbd loads this FCIDUMP directly (ECORE subtracted)
    temp_dir=None,
    clean_temp_dir=True,
)
```

A runnable GPU smoke test (both modes, optional CPU cross-check) lives at
[`src/examples/example_run_sbd_solver_gpu.py`](../../src/examples/example_run_sbd_solver_gpu.py):

```bash
uv run python src/examples/example_run_sbd_solver_gpu.py --mode tpb --compare-cpu
uv run python src/examples/example_run_sbd_solver_gpu.py --mode gdb --max-dim 100000
```

### Single-process vs multi-node MPI

Because `import sbd` runs `MPI_Init`, the solver manages the MPI transport for you
(`configure_singleton_mpi` in `src/sci_solvers/_mpi_singleton.py`, called from
`__init__` before the import):

- **Single-process** (a plain `uv run python …`, the supported default): detected as a
  *singleton* launch and steered onto a safe transport — shared-memory for CPU, the
  CUDA-aware **UCX** transport for `device="gpu"`. Both set
  `OMPI_MCA_ess_singleton_isolated=1`, which stops the singleton `MPI_Init` forking an
  `orted` runtime daemon and then hanging on a pipe read waiting for it — the fix for the
  otherwise-silent import hang on an HPC node. **No `mpirun` needed.**
- **Multi-node** (one GPU rank per node): launch the *same* script under `mpirun`/`srun`.
  The launcher's rank env vars are detected, the singleton config is a **no-op**, and
  `sbd` uses the real cluster fabric. The FCIDUMP must be visible to every rank.

  The GPU backend needs a **CUDA-aware** transport (it passes CUDA device pointers into
  MPI collectives), so if a multi-rank GPU run crashes in the Davidson Allreduce, force
  UCX explicitly: `mpirun … --mca pml ucx --mca btl '^vader,tcp' …`. The example's
  module docstring has the full `mpirun`/`srun` recipes.

## Spin handling

The `sbd` package has **no** spin-projection mechanism — its Davidson returns the
absolute lowest eigenvalue of the spin-free Hamiltonian regardless of total spin. The
wrapper therefore **rejects** any non-`None` `spin_sq` rather than returning a wrong
sector. The generic post-hoc spin filter needs full CI amplitudes and so does not
compose with the occupancy-only `sbd`.

## Output

Each `SCIResult.energy` is the **electronic** energy only (callers add `e_nuc`); the
package subtracts ECORE so the contract holds. The result carries per-orbital
occupancies (density split `[::2]`/`[1::2]`) and ground-state amplitudes — **not** full
1-/2-RDMs. If you need RDMs or spin projection, use [`pyscf`](pyscf.md) or
[`torch_sci`](torch_sci.md).
