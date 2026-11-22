# torch-sci-solver

A PyTorch-native **selected configuration interaction (SCI) / SQD eigensolver**.
It reimplements the selected-CI Hamiltonian–vector product of the RIKEN
[SBD](https://github.com/r-ccs-cms/sbd) code and runs the same high-level code
on CPU and GPU.

Jump to: &nbsp;
[Main features](#main-features) &nbsp;·&nbsp;
[Install](#install) &nbsp;·&nbsp;
[Usage](#usage) &nbsp;·&nbsp;
[Tuning and memory](#tuning-and-memory) &nbsp;·&nbsp;
[Precision](#precision) &nbsp;·&nbsp;
[Console script](#console-script) &nbsp;·&nbsp;
[Layout](#layout) &nbsp;·&nbsp;
[Tests](#tests) &nbsp;·&nbsp;
[Short feature examples](#short-feature-examples)

## Main features

Click a feature to jump to a short code example.

- [GPU acceleration (`device="cuda"`)](#feature-gpu)
- [Multi-GPU on one node (`devices=[...]`)](#feature-multi-gpu)
- [Apple MPS (`device="mps"`)](#feature-mps)
- [Distributed state vector and checkpoint/restart (`distributed=...`)](#feature-distributed)
- [TPB — tensor product basis (`mode="tpb"`, Cartesian product)](#feature-tpb)
- [GDB — general determinant basis (`mode="gdb"`, explicit determinant list)](#feature-gdb)
- [Complex integrals (`dtype=torch.complex128`)](#feature-complex-integrals)
- [Spinless Hamiltonian (flat spin-orbital, no α/β split)](#feature-spinless)
- [⟨S²⟩ penalty (`spin_sq=<target>`)](#feature-spin-sq-penalty)
- [⟨S²⟩ measurement (`compute_spin_squared=True`)](#feature-measure-spin-squared)
- [Excited states / multi-root (`n_roots=k`)](#feature-multi-root)
- [Warm start from a previous state (`init_vectors=`)](#feature-warm-start)
- [1-RDM and 2-RDM (`compute_full_rdm=True`)](#feature-rdms)
- [Narrow Davidson subspace (`davidson_dtype=torch.float32` / `torch.complex64`)](#feature-fp32-davidson)

Two matvec backends are provided:

- **`triton`** — fused GPU kernels: output-stationary, register-accumulating
  kernels tile every TPB channel except the diagonal, plus the block-mode GDB
  same-spin doubles; an entry-parallel kernel tiles the streamed GDB and
  spinless singles and doubles tables. In block-mode GDB the singles, αβ doubles
  and diagonal stay batched. Real and complex dtypes are supported. Matches
  SBD's asymptotic scaling and runs several times faster in absolute wall time
  on a single GPU.
- **`batched`** — a portable, fully vectorized backend that runs anywhere
  PyTorch runs. Used as the CPU path and as the correctness reference.

The backend is chosen automatically (`matvec_impl="auto"`): Triton on a CUDA
device, batched on CPU, and batched in GDB mode (pass `matvec_impl="triton"`
there explicitly). On a CUDA device without a working Triton it raises a
clear error rather than silently falling back (pass `matvec_impl="batched"`
to override).

## Install

This package is a workspace member of
[sci-solvers](https://github.com/IBM/sci-solvers) and is installed with it by
`uv sync` at the repository root.

On a Linux CUDA install, `torch` already brings a matching `triton` as a
transitive dependency, so the GPU backend works with no extra step.

Standalone installs use the optional extras:

```bash
pip install -e "packages/torch-sci-solver[pyscf]"  # FCIDUMP / integral handling
pip install -e "packages/torch-sci-solver[sqd]"    # SQD loop integration
```

## Usage

Engine only (no qiskit dependency):

```python
import torch_sci_solver  # light import; torch + numpy only
```

SQD integration (requires the `sqd` extra; mirrors `qiskit_addon_sqd.fermion`):

```python
from torch_sci_solver import make_torch_sci_solver
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

solver = make_torch_sci_solver(device="cuda")  # auto-selects the Triton backend
result = diagonalize_fermionic_hamiltonian(
    h1,
    h2,
    bit_array,
    samples_per_batch=...,
    norb=...,
    nelec=...,
    sci_solver=solver,
)
```

Within sci-solvers, the usual entry point is `create_solver("torch_sci", ...)`; see
[`docs/solvers/torch_sci.md`](../../docs/solvers/torch_sci.md).

Lower-level building blocks, for a single diagonalization without the recovery loop
(`I0, I1, I2` from `torch_sci_solver.sbd_ref.setup_integrals_from_pyscf_h1_eri`):

```python
from torch_sci_solver.torch_impl.sbdiag import diag_torch

energy, eigvec, info = diag_torch(
    adets,
    bdets,
    L,
    I0,
    I1,
    I2,
    max_davidson_iters=50,
    davidson_block_size=30,
    eps=1e-7,
    device="cuda",
)
```

The 1- and 2-RDMs (`make_one_rdm_torch`, `make_two_rdm_torch` in
`torch_sci_solver.torch_impl`) match PySCF's `make_rdm1s` / `make_rdm12s`. The matvec
uses interleaved-Wick parity while PySCF uses spin-block parity; the RDM wrappers
convert by default (`eigvec_in_interleaved_wick=True`). Pass `False` for an
eigenvector already in spin-block order (e.g. PySCF's civec).

## Tuning and memory

- **`chunk_size`** caps the number of lifted matvec entries per inner batched call.
  Larger chunks mean less Python-loop overhead but more peak memory; the result is
  identical to summation-order round-off for any chunk size
  (`tests/test_torch_chunking.py` bounds it at 1e-10 relative). The default
  `2_000_000` suits an 80 GB A100; halve it until an OOM goes away.
- **`davidson_block_size`** sets the Davidson subspace size per restart (default:
  auto-tuned per subspace from the free memory left after setup, never below
  `n_roots`). Its workspace, about `(2·davidson_block_size + 4) × N × 8` bytes for an
  `N`-dimensional basis, is usually the limit on very large bases, so lowering it
  (e.g. to 10) is the first knob to try on OOM.
- `TORCH_SCI_MATVEC_PROFILE=1` prints a per-phase wall-time profile of the last matvec
  of each solver call when `verbose=True`.

## Precision

The default dtype is `float64` and is not affected by `torch.set_default_dtype`. A
relative error between CPU and CUDA above about `1e-10` points to an explicit
reduced-precision choice (e.g. `dtype=torch.float32`). For a complex Hermitian
Hamiltonian pass `dtype=torch.complex128` (or `complex64`). The Triton kernels support
real and complex dtypes on the TPB, GDB and spinless paths: Triton has no complex
scalar, so complex operands are passed as interleaved real pairs. `matvec_impl="auto"`
selects Triton on a CUDA device for TPB and spinless bases and stays batched for GDB
(pass `matvec_impl="triton"` there explicitly); an explicit `"triton"` on a non-CUDA
device raises.

## Console script

`torch-sci-solver-smoke` (`torch_sci_solver.scripts.cuda_smoke`) checks that the
torch CPU and CUDA paths agree with the scalar reference on the bundled benzoic acid FCIDUMP
(or one given by `TORCH_SCI_SOLVER_FCIDUMP`).

## Layout

```
torch_sci_solver/
  torch_impl/     # PyTorch/Triton engine (kernels, dispatch, Davidson, adapter)
  sbd_ref/        # SBD reference port + integral setup
  multi_gpu/      # spawn helper used by the multi-GPU benchmark and tests
  scripts/        # console-script entry point
  data/           # bundled methane and benzoic acid FCIDUMPs and shots (see data/README.md)
  shots.py        # load measured shots into a Qiskit BitArray
  qiskit.py       # qiskit-addon-sqd integration API
tests/
```

## Tests

```bash
uv run pytest -v packages/torch-sci-solver/tests
```

Tests that need molecular data use the bundled methane and benzoic acid FCIDUMPs and shots
(IDs 001 and 182 of the SQD molecules dataset); point `TORCH_SCI_SOLVER_FCIDUMP` /
`TORCH_SCI_SOLVER_SHOTS` at other files to override them.

## Short feature examples

Each snippet below is intentionally minimal — enough to see the knob. All
snippets assume `hcore`, `eri`, `norb` and `nelec = (n_a, n_b)` (except where
the spinless path takes a scalar `nelec`) already in scope.

<a id="feature-gpu"></a>
### GPU acceleration

```python
from torch_sci_solver import make_torch_sci_solver

solver = make_torch_sci_solver(device="cuda")  # or "cpu"; Triton auto-selected on CUDA
```

<a id="feature-multi-gpu"></a>
### Multi-GPU on one node (batch-parallel)

```python
solver = make_torch_sci_solver(devices=["cuda:0", "cuda:1"])
# Each subspace is diagonalized on its own GPU by a persistent one-process-per-GPU
# worker pool (spawned once and reused across solver calls).
# Speedup is bounded by min(num_batches, len(devices)); num_batches=1 sees no speedup.
# `device=` is ignored when `devices=` is set; entries must be indexed CUDA devices.
# TPB/GDB only: a scalar `nelec` (spinless) runs on a single device.
```

<a id="feature-mps"></a>
### Apple MPS

```python
solver = make_torch_sci_solver(device="mps", dtype=torch.float32)
# MPS has no float64/complex128: pass float32 (or complex64 for a complex H).
# `matvec_impl="triton"` and `distributed=` are rejected; `davidson_tolerance`
# below 1e-5 is raised to that floor with a warning.
```

<a id="feature-distributed"></a>
### Distributed state vector / checkpoint-restart

```python
solver = make_torch_sci_solver(
    distributed={
        "world_size": 2,
        "backend": "nccl",   # "nccl" for CUDA ranks, "gloo" for CPU ranks
        "device_ids": [0, 1],
        "persistent": True,  # default: ranks stay up across solver calls
        "matvec": "local",   # owner-computes; "replicated" is the fallback
        "tile": "auto",      # 2-D tile grid: "auto" | "force" | [p_a, p_b]
        # "checkpoint_dir": "/path/to/ckpt",  # periodic Davidson checkpoint/restart
        # "checkpoint_every": 50,             # Davidson iterations between writes
        # "checkpoint_resume": "auto",        # "auto" | "off" | "require"
    },
)
# TPB, GDB and spinless bases, real or complex, single- or multi-root.
# `distributed=True` uses all visible CUDA devices (NCCL) when there are at
# least two, else the launcher's WORLD_SIZE. `devices=` and `distributed=`
# are mutually exclusive. Checkpointing is TPB/GDB only.
# See ../../docs/distributed_state_vector.md.
```

<a id="feature-tpb"></a>
### TPB — tensor product basis

```python
# (strs_a, strs_b) are two INDEPENDENT per-spin sets; the subspace is their full
# Cartesian product, dimension len(strs_a) * len(strs_b). Every α string is paired
# with every β string — 1,000 sampled determinants become ~1e6.
solver = make_torch_sci_solver(device="cuda", mode="tpb")  # default
# `solver(...)` returns list[EigenStates]; each `states[0].sci_state` is an
# SCIState whose amplitudes are shape (len(strs_a), len(strs_b)).
# `matvec_impl="triton"` tiles every TPB channel except the diagonal.
# `matvec_impl="auto"` selects Triton on CUDA.
```

<a id="feature-gdb"></a>
### GDB — general determinant basis

```python
# (strs_a, strs_b) are PARALLEL length-N lists; index i is the single determinant
# (strs_a[i], strs_b[i]) and the dimension is N — no Cartesian expansion.
solver = make_torch_sci_solver(
    device="cuda",
    mode="gdb",
    gdb_kwargs={"budget_bytes": 8 * 1024**3},  # optional; caps scratch memory
)
# Do NOT feed GDB `bitstring_matrix_to_ci_strs` output — it dedups each spin
# independently and would give a wrong subspace silently. Extract the α and β halves
# of each measured bitstring yourself and pass those length-N arrays.
# `solver(...)` returns list[EigenStates]; each `states[0].sci_state` is a flat
# PairSCIState whose amplitudes are shape (N,), not (N_a, N_b).
# `matvec_impl="triton"` tiles the same-spin doubles (block mode) or the
# singles and doubles tables (streamed); in block mode the singles, αβ doubles
# and diagonal stay batched. `"auto"` keeps GDB on the batched pair path.
```

<a id="feature-complex-integrals"></a>
### Complex integrals

```python
import torch

# Opt-in: real-dtype solver + complex hcore/eri raises rather than dropping Im.
solver = make_torch_sci_solver(device="cuda", dtype=torch.complex128)
# Complex Hermitian only; non-Hermitian raises. Triton reads complex operands as
# interleaved real pairs, so `"auto"` still selects it on CUDA.
```

<a id="feature-spinless"></a>
### Spinless Hamiltonian (flat spin-orbital)

```python
import torch

solver = make_torch_sci_solver(device="cuda", dtype=torch.complex128)

# ci_strings: list of FLAT `strs` arrays (NOT (strs_a, strs_b) tuples).
# nelec = scalar total N (NOT an (n_a, n_b) tuple) → spinless path auto-selected.
results = solver(
    ci_strings=[strs],
    one_body_tensor=hcore,
    two_body_tensor=eri,
    norb=nmo,  # spin-orbital count
    nelec=n_total,  # scalar
)
# Real or complex integrals (complex dtype only when they carry imaginary parts).
# `spin_sq` is rejected (no α/β split to target).
```

<a id="feature-spin-sq-penalty"></a>
### ⟨S²⟩ penalty

```python
solver = make_torch_sci_solver(device="cuda")

# Per-call: biases Davidson toward the <S²> = spin_sq sector; the reported energy
# is <ψ|H|ψ> with the penalty stripped before return.
results = solver(
    ci_strings=ci_strings,
    one_body_tensor=hcore,
    two_body_tensor=eri,
    norb=norb,
    nelec=nelec,
    spin_sq=0.0,  # 0.0 = singlet; None = no penalty
    # spin_penalty_lambda=1.0,       # optional strength override
)
# Works in TPB and GDB. Under `distributed=`, GDB applies it on the pair tile
# when a grid is chosen, else on the 1-D owner-computes block (sharing the
# matvec's all-gather); a basis small enough for the replicated fallback keeps
# the full-vector path. The subspace perf record reports `spin_penalty_layout`
# ("tile" | "1d") when the owner-computes path applies it.
# Not available on the spinless path.
```

<a id="feature-measure-spin-squared"></a>
### Measure ⟨S²⟩ of the ground state

```python
solver = make_torch_sci_solver(device="cuda", compute_spin_squared=True)
# Serial TPB/GDB builds the αβ 2-RDM block and reports <S²> via `s_squared_from_rdm`;
# a distributed owner-computes solve reuses the penalty's S² sweep instead.
# One <S²> value per subspace (root 0) at `solver.stats["spin_squared"]`.
# `solver.stats["spin_sq_final"]` is set only by an active `spin_sq` penalty.
# Not available spinless.
```

<a id="feature-multi-root"></a>
### Excited states / multi-root

```python
solver = make_torch_sci_solver(
    device="cuda",
    n_roots=4,
    davidson_block_size=20,  # n_roots must be ≤ davidson_block_size
    # davidson_missed_root_check="auto",  # "auto" | True | False; auto = on for warm starts
)
results = solver(  # returns list[EigenStates], one per batch
    ci_strings=ci_strings,
    one_body_tensor=hcore,
    two_body_tensor=eri,
    norb=norb,
    nelec=nelec,
)
for batch in results:
    for root in batch.states:  # ascending energy; states[0] is the ground state
        print(root.energy)
```

Works in TPB, GDB, and spinless. `davidson_missed_root_check` (`"auto"` by
default: on for warm starts) searches for a missed root and re-solves; the
distributed replicated fallback rejects it. ⟨S²⟩ and the warm-start snapshot
are root-0 only; full RDMs are root-0 only in TPB/GDB (the spinless path
builds them per root). Every root carries its own energy, amplitudes, and
occupancies.

<a id="feature-warm-start"></a>
### Warm start from a previous state

```python
# One entry per subspace. Each entry is a 1-D vector laid out like THIS mode's
# sci_state.amplitudes, indexed by the strings in THIS call:
#   TPB      → flattened (len(strs_a), len(strs_b)), spin-block signs
#   GDB      → one entry per (strs_a[i], strs_b[i]) pair, interleaved-Wick signs
#   spinless → one entry per string
# Strings dropped from the previous basis are absent; new strings start at 0.
# Overlap below 1e-8 of the vector's own norm falls back to a cold start.
results = solver(
    ci_strings=ci_strings,
    one_body_tensor=hcore,
    two_body_tensor=eri,
    norb=norb,
    nelec=nelec,
    init_vectors=[prev_amps_sub0, None, prev_amps_sub2],  # None = cold start for that batch
    tol=3e-3,  # loose stop pairs well with warm
)
# `solver.stats["per_call_log"][-1]["subspaces"][i]["warm"]` records which batches started warm.
# `probe_method="warm_start"` with `probe_state=` caches a converged state and
# warm-extends it on a superset basis; `solver.capabilities()` reports the
# per-mode cached-state type and whether HF augmentation applies.
```

<a id="feature-rdms"></a>
### 1-RDM and 2-RDM

```python
solver = make_torch_sci_solver(device="cuda", compute_full_rdm=True)  # off by default
results = solver(  # list[EigenStates], one per batch
    ci_strings=ci_strings,
    one_body_tensor=hcore,
    two_body_tensor=eri,
    norb=norb,
    nelec=nelec,
)
rdm1, rdm2 = results[0].states[0].rdm1, results[0].states[0].rdm2  # spin-summed; None when off
# TPB/GDB: RDMs are root-0 only; higher roots carry occupancies but no RDMs.
# (The spinless path builds full RDMs for every root.)
```

<a id="feature-fp32-davidson"></a>
### Narrow Davidson subspace (memory saver at 10⁸ determinants)

```python
import torch

solver = make_torch_sci_solver(
    device="cuda",
    davidson_dtype=torch.float32,  # fp32 over fp64; complex64 over complex128 also works
    # davidson_fp32_energy="auto",    # "rq64" (exact, +1 matvec) | "ritz" (fast) | "auto"
    # davidson_fp32_shift="hf",       # Davidson diagonalizes H - shift; "hf" = min(hii)
)
```

Works in TPB, GDB, and spinless, real or complex (`float32` over `float64`,
`complex64` over `complex128`); the matvec stays at the solver `dtype`. Try a
smaller `davidson_block_size` first; use fp32 when the grid still does not fit.
