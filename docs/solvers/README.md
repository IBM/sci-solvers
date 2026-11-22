# Supported solvers

This document describes the selected-CI / SQD energy solvers shipped by `sci-solvers`.
It is written as a *methods* description: it states what each solver does, which
Hamiltonians it can diagonalize, and which features (complex integrals, multi-root,
spin projection, RDMs, GPU) it supports — rather than as a step-by-step tutorial.
Concrete snippets are given because they are the most precise way to specify a call,
but the intent is to document the solvers, not to walk a reader through a first run.

- [1. Overview](#1-overview)
- [2. The feature matrix](#2-the-feature-matrix)
- [3. What every solver shares — the contract](#3-what-every-solver-shares--the-contract)
- [4. Two families: native vs general-algebra](#4-two-families-native-vs-general-algebra)
- [5. Feature axes](#5-feature-axes)
  - [5.1 Complex integrals](#51-complex-integrals)
  - [5.2 Multi-root (excited states)](#52-multi-root-excited-states)
  - [5.3 Spin projection](#53-spin-projection)
  - [5.4 TPB vs GDB (torch_sci and sbd)](#54-tpb-vs-gdb-torch_sci-and-sbd)
  - [5.5 GPU and multi-GPU](#55-gpu-and-multi-gpu)
  - [5.6 Reduced density matrices](#56-reduced-density-matrices)
  - [5.7 Interior eigenvalues / spectrum slicing](#57-interior-eigenvalues--spectrum-slicing)
- [6. The solvers, one by one](#6-the-solvers-one-by-one)
- [7. Invocation](#7-invocation)
  - [7.1 Standalone (no outer loop)](#71-standalone-no-outer-loop)

## 1. Overview

Every solver here answers the same question: given a set of sampled electronic
configurations (CI strings), what is the lowest eigenvalue (and optionally the next
few) of the electronic Hamiltonian projected onto the subspace those configurations
span? They differ in *how* they diagonalize that subspace, and therefore in which
problems they can take, how large they scale, and what they hand back.

A solver is constructed by name through one factory, so a config value (a CLI flag, a
YAML key) picks the implementation:

```python
from sci_solvers.energy_solvers.factory import create_solver

solver = create_solver("torch_sci", symmetrize_spin=True, device="cuda", hamiltonian=ham)
```

`create_solver` matches the name case-insensitively and forwards every other kwarg
verbatim to the chosen solver's `__init__`
(`src/sci_solvers/energy_solvers/factory.py`). The full set of registered names is
`AVAILABLE_SOLVERS`; the same registry backs three capability queries that answer
*without constructing* the solver — `is_complex_capable`, `supports_multiroot`,
`is_general_algebra` — so a launcher can filter solvers by feature before it commits
to one.

## 2. The feature matrix

The registered factory names and their confirmed features. "Native spin" means the
solver projects onto a target ⟨S²⟩ inside its own diagonalization; "filter" means the
generic post-hoc spin filter (Section 5.3); "—" means the feature is rejected loudly,
not silently ignored.

| name | class | complex | multi-root | spin projection | GPU | 1-RDM | 2-RDM | family | extra |
|---|---|---|---|---|---|---|---|---|---|
| `pyscf` | `PySCFSolver` | — | yes (`n_roots`) | native (`fix_spin_`) | — | yes | yes | native | — |
| `torch_sci` | `TorchSCISolver` | — | yes (`n_roots`) | native (penalty) | **yes + multi-GPU** | yes | yes | native | — |
| `sbd` | `SBDSolver` | — | **no** | — (S² only) | **yes** (Thrust) | no¹ | no¹ | native | `sbd` extra |
| `dice` | `DICESolver` | — | yes (`nroots`) | S²=0 only | — | no¹ | no¹ | native | external binary |
| `truncated_davidson` | `TruncatedDavidsonSolver` | not yet | yes (`nroots`) | filter | — (reserved) | yes | yes (sparse) | native | — |
| `arpack` | `ArpackSolver` | **yes** | yes (`nroots`) | filter | — | yes | opt | general-algebra | — |
| `davidson` | `MultiRootDavidsonSolver` | **yes** | yes (`nroots`) | filter | — | yes | opt | general-algebra | — |
| `lobpcg` | `LobpcgSolver` | **yes** | yes (`nroots`) | filter | — | yes | opt | general-algebra | — |
| `primme` | `PrimmeSolver` | **yes** | yes (`nroots`) | filter | — | yes | opt | general-algebra | `primme` |
| `slepc` | `SlepcSolver` | **yes**² | yes (`nroots`) | filter | — | yes | opt | general-algebra | `slepc` |
| `evsl` | `EvslSolver` | — | yes (`nroots`) | filter | — | yes | opt | general-algebra | built `libevsl` |
| `pyscf_davidson` | `PyscfDavidsonSolver` | — | yes (`nroots`) | native (`fix_spin_`) | — | yes | opt | general-algebra | — |

¹ `sbd` and `dice` return per-orbital **occupancies** (and, for `sbd`, amplitudes), not
full 1-/2-RDMs. `sbd` runs RIKEN's `sbd-eigensolver` package **in process** (`import
sbd`, no subprocess), while `dice` shells out to an external binary.
² `slepc` is complex-capable *only when PETSc was built with complex scalars*;
otherwise it raises rather than dropping the imaginary part.

**Hamiltonian backend.** The general-algebra solvers except `pyscf_davidson` also take
`hamiltonian_backend="fulqrum"`, which builds the projected Hamiltonian with fulqrum
instead of the torch kernel (TPB, GDB and spinless; real and complex; CPU only; `fulqrum` extra).
It isn't a separate row in this table, because it doesn't change any of these columns.
See [fulqrum.md](fulqrum.md).

## 3. What every solver shares — the contract

All solvers subclass `EnergyCalculatorBase`
(`src/sci_solvers/energy_calculation_base.py`) and implement the same
`sci_solver_callable` contract that qiskit-addon-sqd's outer loop expects. Two class
attributes advertise capabilities without instantiation:

```python
complex_capable: bool = False  # can diagonalize a complex Hermitian operator
supports_multiroot: bool = True  # returns >1 root per batch when asked
```

A subclass overrides these when it differs from the default (`sbd` sets
`supports_multiroot = False`; the complex-capable general-algebra solvers set
`complex_capable = True`).

`supports_init_vectors` (default `False`; queried by the factory's
`supports_init_vectors(name)`) marks the solvers that take a per-call Davidson start
vector, `init_vectors=` (one entry per batch, to warm start from a previous state).
Only `torch_sci` sets it today; see [torch_sci.md](torch_sci.md#warm-start-from-a-previous-state-init_vectors).
Every other solver accepts `init_vectors=None` and raises `NotImplementedError` when an
actual vector is passed, rather than silently solving cold.

Every solver exposes **two** entry points:

- **`sci_solver_callable(...)`** — the hot path. Diagonalizes each batch's subspace and
  hands qiskit-addon-sqd the **ground state only**. This is what you pass as
  `sci_solver=` to `diagonalize_fermionic_hamiltonian`. Raising `nroots` is free for
  this loop — the extra roots are computed but not consumed here.
- **`sci_solver_callable_eigen_states(...)`** — returns one `EigenStates` per batch, a
  list of `SCIResult` roots ascending in energy (`states[0]` is the ground state), with
  per-root energies, amplitudes, occupancies and — where the solver builds them — RDMs.
  This is how you read excited states.

A Hamiltonian can be **bound** in the constructor (`hamiltonian=` — a `HamiltonianData`
or an FCIDUMP path) or passed per call, but not both (the base guards against the
ambiguity). Binding once and reusing the solver across batches is the common shape.

The solvers do the **inner** diagonalization. The **outer** SQD configuration-recovery
loop is qiskit-addon-sqd's `diagonalize_fermionic_hamiltonian`; you feed our
`sci_solver_callable` into it (see [Section 7](#7-invocation)).

## 4. Two families: native vs general-algebra

The solvers split into two families that share the contract but reach the eigenvalue by
different routes.

**Native solvers** (`pyscf`, `torch_sci`, `sbd`, `dice`, `truncated_davidson`) each
wrap their own diagonalizer — PySCF's selected-CI, the torch batched-matvec Davidson,
RIKEN's in-process `sbd-eigensolver` package, the Dice C binary, or the
truncated-Davidson kernel.
They know they are solving a chemistry problem and build the CI subspace themselves.

**General-algebra solvers** (`arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl`,
`pyscf_davidson`) hand the sampled subspace to a general-purpose eigensolver that knows
nothing about chemistry. `GeneralAlgebraSolverBase`
(`src/sci_solvers/energy_solvers/general_algebra_base.py`) owns the shared machinery:
it reduces each batch to an `EigenProblem` (a matrix-free operator with a fast torch
matvec), optionally exploits the closed-shell α↔β swap symmetry to solve on a
half-size packed sector view (`symmetrize_spin`), and packs each eigenpair back into a
`SCIResult` with amplitudes, occupancies and RDMs. A concrete solver writes only
`_diagonalize` — how it turns one `EigenProblem` into its lowest `nroots` eigenpairs.

This is why the general-algebra family is where **complex** and **interior-eigenvalue**
support live: the reduction dispatches complex `hcore`/`eri` to a complex operator built
by the complex-CI path (a `HermitianOperator` for a product basis, a `PairCIProblem` for
a pair basis — see §5.1 and §5.4), and the backend (SciPy, PRIMME, SLEPc, EVSL) decides
what it can solve. `is_general_algebra(name)` answers which family a name belongs to.

## 5. Feature axes

### 5.1 Complex integrals

Real integrals are the default and every solver handles them. **Complex** Hermitian
integrals — from spin–orbit coupling, GHF/GSO references, or complex k-point
Hamiltonians — are supported by the general-algebra solvers whose `_diagonalize`
is conjugate-aware: **`arpack`, `davidson`, `lobpcg`, `primme`, and `slepc`** set
`complex_capable = True`. `evsl` and `pyscf_davidson` do not; of the native solvers,
`torch_sci` is also `complex_capable` but through its **own**, independent mechanism —
see the paragraph below — the rest are not.

The dispatch is automatic. When `hcore`/`eri` are complex, `GeneralAlgebraSolverBase`
builds a complex operator through the complex-CI builders instead of the real one — a
`HermitianOperator` in `mode="tpb"`, a complex `PairCIProblem` in `mode="gdb"` — and a
*scalar* `nelec` selects the spinless spin-orbital path. A solver that is **not**
`complex_capable` rejects complex integrals loudly (it lists the solvers that would work),
rather than silently discarding the imaginary part.

The complex gate is **independent of the basis mode**: it is checked before the
`tpb`/`gdb` branch, so the complex-capable set (`arpack`, `davidson`, `lobpcg`, `primme`,
`slepc`) is the same for a product and a pair basis, and `evsl` / `pyscf_davidson` reject
complex in either mode. Complex GDB additionally relies on the guarded bra-conjugation in
the vendored pair RDM kernels (`rdm_pairs.py`), which is a no-op on real amplitudes.

Two constraints inherited from the complex-CI path:

- The complex path described above is **spatial-orbital**: it wants spatial
  `(norb, norb)` integrals with an `(n_a, n_b)` electron count. Genuine spinless /
  GHF / SOC spin-orbital Hamiltonians go through the dedicated spinless path
  instead (a *scalar* `nelec`), which every general-algebra solver *and*
  `torch_sci` now supports (see the `torch_sci` paragraph below) — `evsl` and
  `pyscf_davidson` do not.
- RDMs on the complex path are built by the complex-correct `make_rdm1_complex` /
  `make_rdm12_complex` (`src/sci_solvers/complex_rdm.py`), which conjugate the bra —
  not the real PySCF builders. On a genuinely **selected** (non-full-CAS)
  `(strs_a, strs_b)` product subspace, these auto-chunk over the `(p, q)`
  operator axis once the *connected set* (the selected determinants plus
  everything reachable by one excitation — needed for the 2-RDM Gram term)
  gets large, since a virtual space that's large relative to the number of
  selected strings can reach a connected set far bigger than the selection
  itself (e.g. 100 selected strings at `norb=40`, 20 electrons per spin can
  reach ~16.5k connected determinants per axis — measured: the 1-RDM alone
  takes ~31 minutes there, CPU-only, with no memory error). Chunking bounds
  peak memory but not wall time: this RDM build is plain CPU numpy with no
  GPU path, so it does not speed up under `device="cuda"` on solvers where
  that applies, and can dominate the total solve time at this kind of scale
  even though the diagonalization itself is fast. Pass `chunk=<int>` to
  `make_rdm1_complex`/`make_rdm12_complex` directly to override the
  automatic block size if the default heuristic is a bad fit.

`slepc` is one asterisk: it is `complex_capable`, but only if the linked PETSc was
built with complex scalars. With a real PETSc it raises and points you at the other
complex-capable names.

`torch_sci` is the other asterisk, and a structural one: it is not a general-algebra
solver (`is_general_algebra("torch_sci")` is `False`, no `_diagonalize`/`HermitianOperator`
entry point), so none of the automatic complex-operator dispatch above applies to it.
Its complex support is its own opt-in, set at construction
(`create_solver("torch_sci", dtype=torch.complex128, ...)`, default `torch.float64`) and
enforced by its own `hcore`/`eri`-dtype check rather than the general-algebra reduction's
— see `docs/solvers/torch_sci.md`'s *Complex integrals* section. It also handles the
spinless (scalar `nelec`) path natively through its own flat-spin-orbital matvec/RDM
kernels (not the general-algebra reduction's `HermitianOperator`/`PairCIProblem` path
either) — multi-root at the same level as the spatial path, no `spin_sq` — see that
doc's *Spinless integrals* section. Because `is_general_algebra("torch_sci")` is
`False`, any check written as "spinless needs a general-algebra solver" is now
wrong; check `supports_spinless` instead (`torch_sci` included — a dedicated
capability query, distinct from `is_complex_capable`: every spinless-capable solver
is complex-capable, but not every complex-capable solver implements the spinless
reduction) if the goal is "can this solver diagonalize a spinless Hamiltonian at all."

### 5.2 Multi-root (excited states)

Every solver except the two SBD solvers returns more than one root when asked. The knob is `nroots`
(canonical) — `pyscf` and `torch_sci` also accept the alias `n_roots` to match their
underlying packages; passing both a non-default `nroots` and `n_roots` is an error.

With `nroots = k`, each batch's `EigenStates` holds `k` `SCIResult`s ascending in
energy: `states[0]` is the ground state, `states[1:]` the excited states. Read them
through `sci_solver_callable_eigen_states`; the plain `sci_solver_callable` still hands
the outer SQD loop only `states[0]`, so raising `nroots` costs the hot loop nothing and
surfaces excited states only to callers that want them.

`sbd` is the exception: SBD's Davidson returns the ground state only, so it sets
`supports_multiroot = False`.

`truncated_davidson` adds `n_extra_roots` — buffer roots solved alongside `nroots` to
stabilize convergence, then discarded.

### 5.3 Spin projection

Targeting a total spin sector (`spin_sq`, e.g. `0.0` for a singlet) is orthogonal to how
a solver diagonalizes, so it is provided three ways:

- **Native, inside the solve.** `pyscf` and `pyscf_davidson` use PySCF's `fix_spin_`;
  `torch_sci` adds an S² penalty to the operator. Prefer these when the target sector
  sits deep in the spectrum, where a filter would need a large oversample to reach it.
- **The post-hoc filter.** `SpinFilterMixin`
  (`src/sci_solvers/energy_solvers/spin_filter_mixin.py`) wraps a solver that does *not*
  project natively: it oversolves, classifies each root's ⟨S²⟩ from its CI amplitudes,
  keeps the roots matching the target, and escalates the oversample until enough are
  found. Every general-algebra solver except `pyscf_davidson` mixes it in
  (`arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl`), and there are
  `*SpinFilterMixin` variants of the native `pyscf`, `dice`, `truncated_davidson`, and
  native-Davidson solvers.
- **Not available.** SBD's diagonalization (`sbd`) has no spin
  machinery — it returns the absolute lowest state regardless of spin, so the wrapper
  rejects any non-`None` `spin_sq`. `dice`'s Riken-branch binary hardcodes S²=0, so it
  accepts only `None` or `0.0` and rejects any other value rather than returning a wrong
  sector.

The filter needs full CI amplitudes to classify ⟨S²⟩, so it does not compose with the
occupancy-only SBD solvers.

### 5.4 TPB vs GDB (torch_sci, sbd, and the general-algebra family)

`torch_sci`, `sbd`, and **every general-algebra solver except `pyscf_davidson`** take a
basis-mode choice as `mode=`:

- **TPB** — *tensor product basis* (default). `(strs_a, strs_b)` are two independent
  per-spin sets and the subspace is their full **Cartesian product**, dimension
  `len(strs_a) * len(strs_b)`. 1,000 sampled determinants become ~10⁶.
- **GDB** — *general determinant basis*. `(strs_a, strs_b)` are **parallel length-N
  lists**: index `i` is the single determinant `(strs_a[i], strs_b[i])`, dimension N.
  1,000 sampled determinants stay 1,000.

The same measurement data therefore defines very different subspaces. GDB has three
sharp edges — do not feed it the output of `bitstring_matrix_to_ci_strs` (it dedups each
spin independently, giving a wrong subspace silently), it returns a flat `(N,)`
amplitude array rather than `(N_a, N_b)`, and (for `torch_sci`) its block-mode Triton
kernel covers the same-spin doubles slices only while the streamed path also tiles the
singles.
See the repo README's *TPB vs GDB* section for the full contract.

`sbd` takes the **same** `mode="tpb"|"gdb"` with the identical `(strs_a, strs_b)`
semantics (see [sbd.md](sbd.md)); `mode="gdb"` there requires `len(strs_a) ==
len(strs_b)` and diagonalizes the parallel-pairs subspace on CPU or GPU.

The **general-algebra family** (`arpack`, `davidson`, `lobpcg`, `primme`, `slepc`,
`evsl`) takes the same `mode="tpb"|"gdb"` on `GeneralAlgebraSolverBase`. Under
`mode="gdb"` each batch builds a `PairCIProblem` over the vendored `torch_sci_solver`
pair engine (full-determinant Slater–Condon, no per-spin factorization) rather than the
product `EigenProblem`; it works for real and complex Hermitian integrals alike and packs
the same energies/occupancies/RDMs. `pyscf_davidson` is the one exception — its PySCF
`kernel_fixed_space` can only build a product basis, so it raises `NotImplementedError`
under `mode="gdb"`.

`slepc` *also* accepts a `storage=` argument, but it is a separate, orthogonal knob:
`"shell"` (matrix-free) vs `"aij"` (assembled CSR) — a PETSc **operator-storage** choice,
not a basis choice. (It was renamed from `mode=` so it no longer collides with the
family-wide basis `mode`.)

### 5.5 GPU and multi-GPU

Two solvers run on GPU:

- **`torch_sci`** defaults to `device="cuda"` and falls back to CPU. It also supports
  **single-node, batch-parallel multi-GPU**: pass `devices=["cuda:0", "cuda:1", ...]`
  and each SQD subspace (batch) is diagonalized on its own GPU via a thread-pool
  fan-out. Speedup is bounded by `min(num_batches, len(devices))` and is near-linear
  only when `num_batches` ≥ and is roughly divisible by `len(devices)`; a single large
  diagonalization still runs on one GPU.
- **`sbd`** runs RIKEN's `sbd-eigensolver` package **in process**; pass `device="gpu"`
  (Thrust) so its Davidson matrix-vector product runs on the GPU — no `mpirun` needed for
  a single-process run (the solver configures the CUDA-aware UCX transport itself; see
  [sbd.md](sbd.md)).

The general-algebra base carries a `device`/`dtype` for its torch matvec, but the
iterative backends themselves (SciPy, PRIMME, SLEPc, EVSL) are CPU solvers. With
`hamiltonian_backend="fulqrum"` the whole operator is CPU-only, and any non-CPU `device`
is rejected (see [fulqrum.md](fulqrum.md#cpu-only)); pass `device="cpu"` explicitly on
GPU nodes.
`truncated_davidson` accepts `device`/`dtype`/`chunk_size` but they are **reserved for
future use and currently unused** — it runs on CPU.

### 5.6 Reduced density matrices

- **Full 1-/2-RDMs:** `pyscf`, `torch_sci`, `truncated_davidson`, and every
  general-algebra solver build them from the CI amplitudes. The general-algebra family
  builds the 1-RDM always and the 2-RDM on request (`compute_rdm2=True`); spin
  projection forces the 2-RDM on because ⟨S²⟩ needs the αβ block.
- **Sparse RDM:** `truncated_davidson` computes RDMs from its sparse solution vector.
- **Complex RDMs:** on the complex path the general-algebra base uses the
  complex-correct builders in `complex_rdm.py` (Section 5.1).
- **Occupancies only:** `sbd` and `dice` return per-orbital occupancies
  (and, for `sbd`, amplitudes), not full RDMs — SBD reports them from its
  density vector, `dice` parses them from the binary.

### 5.7 Interior eigenvalues / spectrum slicing

`evsl` is the interior-eigenvalue solver. With no `window` it runs a KPM density-of-
states pre-pass to estimate a lower band holding at least `nroots` eigenvalues; with
`window=(lo, hi)` it slices that energy interval directly (`n_slices`, polynomial
degrees, Lanczos dimension, and DOS tuning are exposed). It falls back to a dense solve
on tiny subspaces. It is the tool for eigenvalues in the *middle* of the spectrum, where
a lowest-`k` Krylov solver would have to climb from the bottom.

`primme` and `slepc` are preconditioned/`SMALLEST_REAL` Krylov solvers aimed at the
lowest roots; they are not interior-eigenvalue solvers in the EVSL sense.

### 5.8 Davidson iteration knobs

Every Davidson-family solver here bounds its work with two quantities: how large the
subspace grows before a restart, and how much iterating it is allowed to do. The first is
spelled `davidson_block_size` everywhere. The second depends on the **loop structure**,
which genuinely differs between solvers, so it deliberately carries two names:

| solver | loop structure | subspace | iteration budget |
|---|---|---|---|
| `torch_sci` | nested (outer restarts × inner expansion) | `davidson_block_size` | `max_davidson_iters` |
| `sbd` | nested (`--block` × `--iteration`) | `davidson_block_size` | `max_davidson_iters` |
| `pyscf` | flat (restart is an internal flag) | `davidson_block_size` | `max_cycle` |
| `davidson` (native) | flat (same `davidson1` kernel) | `davidson_block_size` | `max_cycle` |
| `davidson` (general-algebra) | flat (same `davidson1` kernel) | `davidson_block_size` | `max_cycle` |
| `truncated_davidson` | flat (restart collapses in place) | `davidson_block_size` | `max_cycle` |
| `dice` | not exposed (compile-time constants) | — | — |

**`max_cycle` is a total, `max_davidson_iters` is a restart count.** On the flat solvers
every iteration — restarts included — spends one cycle from the same budget, so
`max_cycle` is the total matvec count. On the nested solvers the outer loop runs
`max_davidson_iters` times and each pass expands the subspace up to
`davidson_block_size`, so the comparable total is the *product*. To set up a fair
comparison across solvers:

```
max_cycle  ≈  max_davidson_iters × davidson_block_size
```

Two caveats. `truncated_davidson` builds truncated residuals via PCR instead of forming
`H·x`, so an iteration there is not one matvec and the equivalence above compares
iteration counts, not equal work. (Its inner kernels have two interchangeable
implementations selected by `backend="numba"` (default) or `backend="cpp"` — same
algorithm and results, so the iteration budget means the same on either; see
[truncated_davidson.md](truncated_davidson.md).) And on the PySCF-family solvers `davidson_block_size`
is translated before it reaches the kernel: `davidson1` widens its own `max_space` by
`(nroots-1)*4`, so that widening is subtracted first, which is what makes a given block
size mean the same number of vectors on every solver. Passing `max_space` directly is
rejected; pass `davidson_block_size`.

`dice` exposes neither knob — Dice's Davidson iteration cap and block size are compile-time
constants in the binary. Its `max_hci_iters` is the *HCI selection* macro-loop (defaulting
to 1, i.e. a single fixed-space diagonalization), unrelated to Davidson iterations, and
solve quality is controlled instead by `davidson_tolerance` / `davidson_tolerance_loose`.

## 6. The solvers, one by one

Each is a self-contained page.

| solver | page | best for |
|---|---|---|
| **PySCF** | [pyscf.md](pyscf.md) | the reference correctness baseline; native spin, full RDMs |
| **torch_sci** | [torch_sci.md](torch_sci.md) | large active spaces, GPU / multi-GPU, TPB or GDB |
| **SBD** | [sbd.md](sbd.md) | in-process RIKEN `sbd-eigensolver`, GPU (Thrust), TPB or GDB; ground state only |
| **DICE** | [dice.md](dice.md) | the Dice C binary through a Python wrapper |
| **truncated_davidson** | [truncated_davidson.md](truncated_davidson.md) | sparse, single-batch selected-CI with sparse RDMs |
| **general-algebra** | [general_algebra.md](general_algebra.md) | `arpack`/`davidson`/`lobpcg`/`primme`/`slepc`/`evsl`/`pyscf_davidson` — complex integrals, interior eigenvalues, swappable backend |
| **fulqrum backend** | [fulqrum.md](fulqrum.md) | `hamiltonian_backend="fulqrum"` on the general-algebra solvers — a much faster CPU Hamiltonian build, TPB, GDB or spinless |

## 7. Invocation

The inner solver plugs into qiskit-addon-sqd's outer configuration-recovery loop:

```python
from functools import partial
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

ham = HamiltonianData.from_fcidump("FCIDUMP")
# sbd on the GPU; mode="tpb" (default) reads each batch as the α×β product subspace
solver = create_solver("sbd", mode="tpb", symmetrize_spin=True, device="gpu", hamiltonian=ham)

result = diagonalize_fermionic_hamiltonian(
    one_body_tensor=ham.hcore,
    two_body_tensor=ham.eri,
    bit_array=bit_array,
    samples_per_batch=512,
    norb=ham.num_orbitals,
    nelec=(ham.num_elec_a, ham.num_elec_b),
    max_iterations=3,
    sci_solver=partial(solver.sci_solver_callable, spin_sq=None, max_cycle=200),
    max_dim=2000,
    num_batches=1,
    symmetrize_spin=True,
)
print(f"electronic energy: {result.energy:.8f} Ha")
print(f"total energy:      {result.energy + ham.e_nuc:.8f} Ha")
```

To read excited states, call `sci_solver_callable_eigen_states(...)` with `nroots > 1`
instead and inspect `states[1:]` of each returned `EigenStates`. See `src/examples/`
for runnable drivers per solver.

### 7.1 Standalone (no outer loop)

If you already hold the CI-string batches — one `(strs_a, strs_b)` int64-array tuple
per subspace — you can diagonalize them directly with `compute_energy`, no
qiskit-addon-sqd loop required. The Hamiltonian must be bound in the constructor:

```python
from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

ham = HamiltonianData.from_fcidump("FCIDUMP")

# sbd runs its Thrust eigensolver on the GPU; mode="gdb" reads the batch as
# parallel (strs_a[i], strs_b[i]) pairs (dimension N, not N_a·N_b — see §5.4).
solver = create_solver("sbd", mode="gdb", device="gpu", symmetrize_spin=False, hamiltonian=ham)

# one SCIResult per batch (ground state); energy is electronic-only
results = solver.compute_energy(ci_strs_batches=[(strs_a, strs_b)])
e_elec = results[0].energy
print(f"total energy: {e_elec + ham.e_nuc:.8f} Ha")

# occupancies come back on the same SCIResult (sbd returns occupancies only;
# swap in e.g. "arpack"/"torch_sci" for full results[0].rdm1 / .rdm2)
occ_a, occ_b = results[0].orbital_occupancies
```

`compute_energy` also accepts `bitstring_matrix_batches=[bool_matrix, ...]` instead of
`ci_strs_batches`; it converts each with `bitstring_matrix_to_ci_strs`
(`open_shell=not symmetrize_spin`). Do **not** use that path for GDB — see §5.4.
