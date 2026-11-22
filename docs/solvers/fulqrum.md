# fulqrum — a Hamiltonian backend for the general-algebra solvers

`create_solver("arpack", hamiltonian_backend="fulqrum", ...)`

[fulqrum](https://github.com/qiskit-community/fulqrum) (IBM, Apache-2.0, `import
fulqrum`) builds the projection of a Hamiltonian onto a set of bitstrings as a CSR
matrix, in C++ with OpenMP. In sci-solvers it isn't a solver of its own: it's an
alternative builder of each batch's projected Hamiltonian for the **general-algebra
solvers**, chosen with `hamiltonian_backend="fulqrum"`. The default,
`hamiltonian_backend="torch"`, is the torch Slater–Condon kernel.

Only the builder changes. The eigensolver, multi-root, `symmetrize_spin`, the `spin_sq`
filter, convergence handling and the RDM/result packing are the same code as with the
default backend, and the results match it to floating-point precision. The source is
in `src/sci_solvers/fulqrum_ci_problem.py`.

## Features

| feature | support |
|---|---|
| solvers | `arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl` — **not** `pyscf_davidson` (it builds its own operator from the raw integrals; rejected loudly) |
| basis mode | **TPB** (`mode="tpb"`, default) and **GDB** (`mode="gdb"`) |
| complex Hermitian integrals | **yes**, TPB and GDB (with the same Hermiticity checks as the torch complex builders) |
| spinless (scalar `nelec`) | **yes** — the operator is built term by term from the spin-orbital integrals |
| multi-root, `spin_sq`, `symmetrize_spin`, 1-/2-RDM | yes — inherited unchanged from the general-algebra pipeline |
| device | **CPU only** — a non-CPU `device` warns and runs on CPU, or raises with `device_fallback="raise"` (see [CPU only](#cpu-only)) |
| operator mode | `fulqrum_mode="csr"` (default), `"csr_fast"` or `"matrix_free"`, in TPB, GDB and spinless (see [Operator modes](#operator-modes)) |

Two contract details:
- Like every general-algebra solver, TPB batches must be **strictly ascending** (sorted,
  unique) CI strings; anything else is rejected.
- GDB pair lists may be unsorted and contain repeats. They are collapsed into the same
  deduplicated, lex-sorted pair basis as the default backend.
- Spinless batches are flat `N`-particle determinant lists, as on the default backend.
  They may be in any order, and duplicates are rejected.

## Performance

Measured end to end through `create_solver("arpack", hamiltonian_backend=...)` on a GPU
node (2× A100, 16 threads), N2 / cc-pVDZ CAS(16o,10e). fulqrum always runs on CPU.

| basis | dimension | default backend, CPU | default backend, GPU | fulqrum |
|---|---|---|---|---|
| TPB | 404,496 | 51 s | 6.4 s | 4.4 s |
| GDB | 40,000 pairs | 160 s | 8.9 s | 0.5 s |

At the largest TPB size, most of fulqrum's time is the matrix build (2.7 s of 4.4 s),
and the CSR matrix takes 0.3 GB. For larger subspaces, memory decides which
[operator mode](#operator-modes) fits. fulqrum's operator setup depends only on the integrals,
so it's cached per Hamiltonian and reused across batches and calls
(`fulqrum_subspace_hamiltonian`).

## Operator modes

`fulqrum_mode` chooses how fulqrum holds and applies each batch's operator. Every mode
gives the same results; they trade memory against time.

| `fulqrum_mode` | what fulqrum builds | peak memory | build | each product |
|---|---|---|---|---|
| `"csr"` (default) | a CSR matrix, counted first and allocated once (`to_csr_linearoperator`) | ~1 matrix | slower | fulqrum's OpenMP SpMV |
| `"csr_fast"` | a row-wise intermediate, then the CSR matrix (`to_csr_linearoperator_fast`; the build used before `fulqrum_mode` existed) | ~2.5 matrices | ~30% faster | the same |
| `"matrix_free"` | nothing: elements are regenerated on every product (`SubspaceHamiltonian.matvec`) | the basis and solver vectors, not the matrix | none | several CSR products |

Measured with the [benchmark](../../benchmark/README.md) on one subspace (molecule 176), a
full lobpcg solve on 16 CPU threads, the same energy in every mode:

| subspace | `csr` | `csr_fast` | `matrix_free` |
|---|---|---|---|
| 3.7M determinants (639M non-zeros) | 101 s, 8.4 GB | 70 s, 19.5 GB | 394 s, 1.4 GB |
| 13.7M determinants | 480 s, 82 GB | 333 s, 163 GB | 2519 s, 3.6 GB |

How to choose:
- `"csr"` when the matrix fits in memory. It's the lowest-memory assembled mode.
- `"csr_fast"` when memory is plentiful and the solve takes few products, so the build
  dominates.
- `"matrix_free"` when one copy of the matrix doesn't fit. It's the only mode whose memory
  doesn't grow with the number of non-zeros: in the table above, 10× the non-zeros cost it
  2.6× the memory, the growth of the basis itself.

Details:
- **Threads:** the CSR builds and products, and the matrix-free products, use OpenMP
  (`OMP_NUM_THREADS`). At one thread, fulqrum's CSR product runs at the same speed as
  SciPy's; at 16 threads it's about 6× faster.
- **Basis order:** fulqrum orders the basis its own way. The operator is presented in the
  caller's order by permuting vectors around each product, so no reordered copy of the
  matrix is made.
- **`"matrix_free"` and assembled solvers:** it can't be combined with a solver set up to
  assemble the matrix (`arpack` with `use_operator=False`, `slepc` with `storage="aij"`):
  the matrix would have to be built densely from products. That combination raises
  `ValueError` at construction.
- **Block solvers:** a block of `k` vectors (lobpcg, or the `symmetrize_spin` sector view)
  costs `k` fulqrum products, so `"matrix_free"`'s extra cost grows with the block size.

## Installation

fulqrum is an **optional extra**. PyPI ships an sdist only, so installing it compiles
the C++/Cython extension locally (about 40 s). A plain `uv sync` never builds it.

**Linux** (tested): the system GCC is enough.

```bash
uv sync --extra fulqrum
```

`FQ_BUILD_PARALLEL=<n>` speeds up the build by compiling with `n` threads.

**macOS** (tested on Apple Silicon, no Homebrew needed; requires Apple's Command Line
Tools for `clang++` and `install_name_tool`, installed with `xcode-select --install` or as
part of Xcode, and already present if you use Homebrew): fulqrum must use torch's bundled
`libomp.dylib`, as two OpenMP runtimes in one process abort (`OMP: Error #15`) once torch
and fulqrum both run parallel code. The compiler wrapper builds fulqrum with Apple clang
against torch's `omp.h` and links it to torch's `libomp`. fulqrum is built without build
isolation (`[tool.uv]` in `pyproject.toml`), so uv installs torch first:

```bash
CXX="$PWD/scripts/energy_solvers/clang_omp.sh" uv sync --extra fulqrum
```

uv caches the built wheel by sdist, not by compiler. If fulqrum was built before without
the wrapper, add `--reinstall-package fulqrum --no-cache` once. Check the link with
`otool -L .venv/lib/python3.*/site-packages/fulqrum/core/spmv.*.so`, which should list
`@rpath/libomp.dylib`.

**pixi** (macOS and Linux): `pixi install -e fulqrum` (or `-e all`) does this out of the
box; see `[tool.pixi]` in `pyproject.toml`.

Check the install:

```bash
uv run --extra fulqrum python -c \
  'import fulqrum; print("Fulqrum import succeeded")'
```

The extra pins `fulqrum>=0.4.5,<0.5`. The `<0.5` guards against API changes in the
unreleased 0.5 on fulqrum's main branch.

## Usage

```python
from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

ham = HamiltonianData.from_fcidump("FCIDUMP")

solver = create_solver(
    "arpack",  # or davidson, lobpcg, primme, slepc, evsl
    hamiltonian_backend="fulqrum",
    mode="tpb",  # or "gdb"
    fulqrum_mode="csr",  # or "csr_fast", "matrix_free"
    symmetrize_spin=True,
    nroots=3,
    hamiltonian=ham,
)
results = solver.compute_energy(ci_strs_batches=[(strs_a, strs_b)])
```

A caller that picks `device="cuda"` on a GPU node makes the fulqrum backend warn and run
on CPU (see [CPU only](#cpu-only)). Pass `device="cpu"` to skip the warning.

## RAMPS subspace expansion

`sci_solvers.fulqrum_ramps` wraps fulqrum's RAMPS. Starting from a core set of
determinants and a target energy, it adds the determinants the Hamiltonian connects to
them whose second-order perturbative energy estimate exceeds `tol`. The result is a
compact determinant set selected by the Hamiltonian.

- `ramps_expand(hcore, eri, strs_a, strs_b, target_energy, *, tol, max_recursion)`:
  spatial determinants as parallel `(alpha, beta)` arrays, for a `mode="gdb"` solver.
  Real or complex Hermitian integrals.
- `ramps_expand_spinless(hcore, eri, strs, target_energy, *, tol, max_recursion)`: flat
  `N`-particle spin-orbital determinants, for the spinless path. Complex Hermitian
  spin-orbital integrals.

```python
from sci_solvers.fulqrum_ramps import ramps_expand

# core: parallel (alpha, beta) determinant arrays; energy: the core's ground energy
strs_a, strs_b = ramps_expand(hcore, eri, core_a, core_b, energy, tol=1e-6, max_recursion=1)
solver = create_solver("arpack", mode="gdb", hamiltonian_backend="fulqrum", symmetrize_spin=False)
energy = solver.sci_solver_callable([(strs_a, strs_b)], hcore, eri, norb, nelec)[0].energy
```

- **Output:** the core first (deduplicated, in input order), then the additions.
  Particle numbers are preserved.
- **Iterate:** repeating "solve, then expand around the result" converges quickly. On
  N2 / 6-31G CAS(8o,10e), whose full CI has 3,136 determinants, three rounds from the HF
  determinant at `tol=1e-6` reach the full-CI energy with 782 determinants.
- **Complex integrals** work unchanged. The score uses only `|H_ij|` and the real
  denominator `E_target − H_jj`, and complex integrals get the same Hermiticity check as
  the CI builders.
- It reuses the cached fulqrum operators, and the selection itself takes milliseconds.
- **Settings:** `tol` is the selection threshold (smaller keeps more). `max_recursion` is
  the number of selection layers; each layer is seeded by the previous one's additions.

## CPU only

fulqrum has no GPU support. When `hamiltonian_backend="fulqrum"` gets a non-CPU `device`
(`"cuda"`, `"cuda:1"`, `"mps"`, …), the solver's `device_fallback` decides, at
construction and in every mode:
- `"warn"` (default): a `UserWarning`, and the solver runs on `device="cpu"`. The
  results are the same as with `device="cpu"`; only time and memory differ. The device
  is replaced before anything reads it, so `matvec_impl="auto"` and GDB's RDM helpers
  use the CPU too.
- `"raise"`: `ValueError`.

Only this case falls back, because it has one equivalent alternative. Other conflicts,
such as `fulqrum_mode="matrix_free"` on a solver configured to assemble the matrix, have
several alternatives with different trade-offs, so they always raise.

Possible extension (not implemented): **GPU support in fulqrum itself**, for example a
CUDA matrix build or matvec. Then a GPU device would be used, not replaced.

## Running the tests

The fulqrum tests are in `tests/test_fulqrum_conventions.py`,
`tests/test_fulqrum_backend.py` and `tests/test_fulqrum_ramps.py`. They're **skipped when fulqrum isn't installed**, which
includes the CI pipeline, so a green CI check doesn't cover them. Run them locally:

```bash
uv sync --extra fulqrum
uv run --extra fulqrum pytest -rs tests/test_fulqrum_conventions.py tests/test_fulqrum_backend.py \
    tests/test_fulqrum_ramps.py
```

On macOS, install with the compiler wrapper first (see [Installation](#installation)); a
fulqrum linked to another libomp makes them abort with `Fatal Python error: Aborted`.

- **Keep `--extra fulqrum` on `uv run`.** A plain `uv run` re-syncs the environment to
  the default dependencies and removes fulqrum, and the tests then skip without failing.
  Calling the venv's interpreter directly (`.venv/bin/python -m pytest ...`) avoids the
  re-sync.
- **Check that nothing was skipped.** `-rs` prints every skip reason. With fulqrum
  installed there should be no `fulqrum extra not installed` (or `could not import
  'fulqrum'`) lines. Without it, only the few tests that need no fulqrum run: the
  rejections and the missing-install error.

## Conventions handled internally

These are pinned by `tests/test_fulqrum_conventions.py`:

- **Ordering:** half-string (TPB) products are indexed beta-major (`ib*Na + ia`), and
  full-string (GDB) subspaces are sorted as `lexsort((alpha, beta))`. Both are reordered
  to sci-solvers' basis order.
- **Signs:** fulqrum's basis differs from the PySCF spin-block one by one global sign.
  GDB is converted to `PairCIProblem`'s interleaved-Wick signs (σ).
- **Complex ERIs:** `integrals_to_fq_fermionic_op` reads a complex chemist ERI
  conjugated, so complex ERIs are passed as `eri.transpose(1, 0, 3, 2)`.
- **Duplicates:** fulqrum keeps duplicate strings as separate rows, so the TPB and
  spinless builders reject them.
- **Spinless:** `integrals_to_fq_fermionic_op` always spin-doubles its input, so it can't
  take spin-orbital integrals. The spinless operator is instead built term by term as a
  `FermionicOperator`, with the raw ERI (no transpose). Its basis signs equal the
  brute-force ladder-operator ones up to one global sign.
