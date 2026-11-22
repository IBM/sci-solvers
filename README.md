# sci-solvers

Selected configuration interaction (SCI) eigensolvers behind one interface. Given sampled
electronic configurations (for example from a quantum computer, in
[sample-based quantum diagonalization](https://github.com/Qiskit/qiskit-addon-sqd)), each
solver diagonalizes the molecular Hamiltonian projected onto that subspace. The package
wraps external diagonalizers, provides native CPU/GPU solvers, and ships build recipes for
the external binaries. Every solver plugs into `qiskit-addon-sqd`'s SQD loop as its
`sci_solver`, or can be called standalone on CI-string batches.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/), which is the supported
install. From a clone of this repository:

```console
uv sync
```

Optional backends are extras, e.g. `uv sync --extra fulqrum`, `--extra sbd`, or
`--extra external-solvers` (PRIMME and SLEPc). `torch-sci-solver` (under
`packages/torch-sci-solver`) is a workspace member installed by `uv sync`. Solvers that
run an external binary (Dice, EVSL) need it built first; see
[`scripts/energy_solvers/README.md`](scripts/energy_solvers/README.md).

## Supported solvers:

* pyscf
* torch_sci_solver - "homegrown" torch sci solver, based on torch, supports both GPU and CPU; `mode="tpb"` (Cartesian product, default) or `mode="gdb"` (explicit determinant list) — see [TPB vs GDB](#torch_sci-basis-modes-tpb-vs-gdb)
* SBD - based on https://github.com/r-ccs-cms/sbd
* DICE - based on https://github.com/Qiskit/qiskit-addon-dice-solver python wrapper and C binary from https://github.com/caleb-johnson/Dice.git
* general-algebra family - `arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl`, `pyscf_davidson`: consistent multi-root (ground + excited states) support behind one shared reduction — see [general-algebra solvers](docs/solvers/general_algebra.md)
* fulqrum - not a separate solver: an alternative Hamiltonian builder for the general-algebra solvers (`create_solver("arpack", hamiltonian_backend="fulqrum", ...)`), based on https://github.com/qiskit-community/fulqrum. CPU-only; TPB, GDB and spinless; real and complex integrals; much faster than the default builder. Install with `uv sync --extra fulqrum` (compiles locally; on macOS, prefix it with `CXX="$PWD/scripts/energy_solvers/clang_omp.sh"`, or use `pixi install -e fulqrum`) — see [fulqrum backend](docs/solvers/fulqrum.md)

### general-algebra solvers (multi-root)

Each of `arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl`, `pyscf_davidson` is a
standalone solver name passed straight to `create_solver(name, nroots=k, ...)` — there is no
separate wrapper solver to select an eigensolver through. Raising `nroots` diagonalizes each
sampled CI subspace for the lowest `k` roots. The standard `sci_solver_callable` hands
qiskit-addon-sqd each batch's ground state; call `sci_solver_callable_eigen_states(...)`
instead to get one `EigenStates` per batch — a list of `SCIResult` roots (ascending energy,
`states[0]` is the ground state) with per-root energies, amplitudes, and occupancies.

* `arpack` - `scipy.sparse.linalg.eigsh`, matrix-free over the fast torch matvec;
  fastest dependency-free option
* `davidson` - block Davidson, also matrix-free and dependency-free
* `lobpcg` - `scipy.sparse.linalg.lobpcg` with a diagonal preconditioner, also matrix-free
* `pyscf_davidson` - PySCF fixed-space Davidson; native multi-root, but uses PySCF's own
  (slower) matvec so it lags `arpack`/`lobpcg` at large `norb`
* `slepc` - SLEPc EPS via slepc4py — needs `uv sync --extra slepc`. For a **complex**
  Hermitian operator this also requires a PETSc built with complex scalars: PETSc fixes its
  scalar type at build time, and the default `petsc` wheel is real-scalar, so `slepc` raises
  on a complex problem there. Use `arpack`/`lobpcg`/`davidson`/`primme` for complex work
  unless you have a complex-scalar PETSc build.
* `primme` - PRIMME Davidson/JDQMR — needs `uv sync --extra primme`
* `evsl` - EVSL polynomial/rational-filter Lanczos, for *interval* eigenvalues rather than
  the lowest `nroots` — needs a locally built `libevsl` (see `scripts/energy_solvers/README.md`)

Install all external backends at once with `uv sync --extra external-solvers`.

**Hamiltonian backend.** All of these except `pyscf_davidson` accept
`hamiltonian_backend="fulqrum"`, which builds each batch's projected Hamiltonian with
fulqrum instead of the torch kernel; the eigensolver and everything downstream are
unchanged. It needs `uv sync --extra fulqrum` and runs on CPU only (a GPU `device` warns
and falls back to CPU, or raises with `device_fallback="raise"`). Its tests skip
without the extra (CI included); run them with `uv run --extra fulqrum pytest -rs ...`.
See [docs/solvers/fulqrum.md](docs/solvers/fulqrum.md).

**Spin projection.** Pass `spin_sq=<target>` (e.g. `0.0` for a singlet) to any of these
solvers. `pyscf_davidson` projects natively via PySCF's `fix_spin_`; every other solver is
wrapped automatically in a filter that oversolves and keeps roots whose ⟨S²⟩ matches the
target. Prefer `pyscf_davidson` when the target sector sits deep in the spectrum, where the
filter would need a large oversample to reach it.

See [docs/solvers/general_algebra.md](docs/solvers/general_algebra.md) for the full feature
matrix (complex Hermitian support, RDMs, GDB basis mode) across all seven.


### torch_sci basis modes: TPB vs GDB

`torch_sci` takes `mode="tpb"` (default) or `mode="gdb"`, which decide what subspace a
batch of CI strings defines:

* **TPB** — *tensor product basis*, the historical behaviour. `(strs_a, strs_b)` are two
  independent per-spin sets and the basis is their full **Cartesian product**, of
  dimension `len(strs_a) * len(strs_b)`. Every α string is paired with every β string,
  including combinations never sampled.
* **GDB** — *general determinant basis*. `(strs_a, strs_b)` are **parallel length-N
  lists**: index `i` is the single determinant `(strs_a[i], strs_b[i])`, and the basis
  dimension is N. No expansion.

The same measurement data therefore gives very different subspaces — 1,000 sampled
determinants become ~1e6 under TPB and stay 1,000 under GDB.

See [Example usage](#example-usage) for how to select GDB. Three things to know:

* **Do not feed GDB the output of `bitstring_matrix_to_ci_strs`** (nor
  `compute_energy(bitstring_matrix_batches=...)`, which calls it). That helper dedups each
  spin independently, so GDB would read per-spin-unique arrays as parallel pairs — a
  different, wrong subspace, silently. Extract the α and β halves of each measured
  bitstring yourself and pass those length-N arrays. For the same reason GDB is not a
  drop-in for `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`; use
  `torch_sci_solver.diagonalize_fermionic_hamiltonian_pairs` for a GDB outer loop.
* **GDB returns a flat `PairSCIState`**, whose `amplitudes` is `(N,)`. Anything that
  reshapes amplitudes to 2-D needs adjusting.
* **`gdb_kwargs={"budget_bytes": ...}`** caps the scratch memory used while GDB re-derives
  which determinants interact (TPB gets that from index arithmetic and needs no such
  knob). `None` auto-sizes: a slice of free VRAM on CUDA, a small fixed value on CPU.
  Raise it on a large GPU; lower it if memory is tight. GDB's Triton path tiles the
  block-mode same-spin doubles and the streamed singles and doubles tables; in block
  mode `matvec_impl="triton"` mixes backends per channel (singles and the diagonal
  stay batched).



### torch_sci multi-GPU (single node, batch-parallel)

`torch_sci` can spread an SQD iteration's subspace batches across multiple GPUs on one
node. Pass a list of CUDA devices as `devices=`:

```python
solver = create_solver(
    "torch_sci", symmetrize_spin=True, devices=["cuda:0", "cuda:1"], hamiltonian=ham
)
# or directly: make_torch_sci_solver(devices=["cuda:0", "cuda:1"])
```

Each subspace (batch) is diagonalized on its own GPU via a thread-pool fan-out; the
subspaces are independent, so no cross-GPU communication is needed. This parallelizes the
**batch** axis only — a single large diagonalization still runs on one GPU.

Speedup is bounded by `min(num_batches, len(devices))` and is near-linear only when
`num_batches >= len(devices)` **and** roughly divisible by it. An awkward remainder leaves
GPUs idle in the final wave (e.g. 5 batches on 4 GPUs ≈ 2.5×, not 5×), and `num_batches=1`
sees no speedup at all. `devices=None` (default) keeps single-device behaviour; `device=`
is ignored when `devices=` is set.


## Example usage

See [`src/examples`](src/examples) for runnable scripts, and
[`docs/solvers`](docs/solvers/README.md) for a guide to every solver. A minimal SQD run
with the `torch_sci` solver:

```python
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from torch_sci_solver.shots import load_shots_bit_array

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

# 1. Hamiltonian
ham = HamiltonianData.from_fcidump("path/to/FCIDUMP")

# 2. Solver: the factory picks TorchSCISolver and forwards kwargs to its __init__.
# Default mode="tpb". For GDB pass mode="gdb" (and optionally
# gdb_kwargs={"budget_bytes": ...}), then call solver.compute_energy(
# ci_strs_batches=[(strs_a, strs_b)]) with parallel arrays where strs_a[i], strs_b[i]
# form determinant i, or run torch_sci_solver.diagonalize_fermionic_hamiltonian_pairs
# as the outer loop instead of step 4 (see "TPB vs GDB" above).
solver = create_solver("torch_sci", symmetrize_spin=True, device="cuda", hamiltonian=ham)

# 3. Measured bitstrings -> BitArray (a pickled dict of bitstring -> count, or .npz)
bit_array = load_shots_bit_array("path/to/shots.pkl")

# 4. qiskit-addon-sqd's outer SQD loop, with our solver as sci_solver
result = diagonalize_fermionic_hamiltonian(
    one_body_tensor=ham.hcore,
    two_body_tensor=ham.eri,
    bit_array=bit_array,
    samples_per_batch=512,
    norb=ham.num_orbitals,
    nelec=(ham.num_elec_a, ham.num_elec_b),
    max_iterations=3,
    sci_solver=solver.sci_solver_callable,
    max_dim=2000,
    num_batches=1,
    symmetrize_spin=True,
)

print(f"electronic energy: {result.energy:.8f} Ha")
print(f"total energy:      {result.energy + ham.e_nuc:.8f} Ha")
```

## Development

Install the dev dependencies and the pre-commit hooks once after cloning:

```console
uv sync
uv run pre-commit install
```

Run the tests (the top-level `tests/` and `packages/torch-sci-solver/tests/`):

```console
uv run pytest -v
# skip slow tests:
uv run pytest -v -m "not slow"
```

Tests that need an external binary, a GPU, or an optional extra skip when it is missing.
The FCIDUMPs and samples of SqDRIFT hardware shots of methane and benzoic acid (IDs 001 and
182 of the [SQD molecules dataset](https://doi.org/10.5281/zenodo.23286514)) used by some
tests are bundled (see
[`packages/torch-sci-solver/torch_sci_solver/data/README.md`](packages/torch-sci-solver/torch_sci_solver/data/README.md)).

Lint, format, and type-check:

```console
uv run ruff check --fix
uv run ruff format
uv run mypy
```

`scripts/local_ci.sh` runs the same checks and tests as CI (`.github/workflows/ci.yml`);
`scripts/local_ci.sh --gpu` also runs the GPU, sbd and slow tests that CI skips (it needs an
NVIDIA GPU and the NVIDIA HPC SDK).

Commit messages follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
(`feat:`, `fix:`, `test:`, ...). Contributions are accepted under the
[Developer's Certificate of Origin](https://developercertificate.org/): sign off each commit
with `git commit -s`.

## License

[Apache License 2.0](LICENSE). Every source file carries the header in
[`LICENSE.tmpl`](LICENSE.tmpl).

Parts of `torch-sci-solver` are modified Python ports of RIKEN's
[SBD](https://github.com/r-ccs-cms/sbd) (Apache-2.0); see [`NOTICE`](NOTICE) for the
copyright holders. The bundled test data is from the SQD molecules dataset
(CDLA-Permissive-2.0); see
[`packages/torch-sci-solver/torch_sci_solver/data/README.md`](packages/torch-sci-solver/torch_sci_solver/data/README.md).
