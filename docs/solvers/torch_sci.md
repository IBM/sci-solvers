# `torch_sci` — the torch batched-matvec solver

`create_solver("torch_sci", ...)` → `TorchSCISolver`
(`src/sci_solvers/energy_solvers/torch_sci_solver.py`). Wraps the `torch_sci_solver`
package's matvec + Davidson + RDM stack.

This is the solver for **large active spaces**: the batched matvec runs on GPU by
default and is the fastest inner diagonalizer here, it scales to multiple GPUs on one
node, and it is the only solver with a basis-mode choice.

## Features

| feature | support |
|---|---|
| complex integrals | yes — opt-in via `dtype=torch.complex128` (see below) |
| spinless (flat spin-orbital, no α/β split) | yes — auto-detected from a *scalar* `nelec` (see below); real or complex dtype, complex only when the integrals carry imaginary parts |
| multi-root | yes — `nroots` / `n_roots`; TPB, GDB, and spinless |
| spin projection | native, via an S² penalty on the operator; a **physics limitation** on the spinless path (no α/β split defines S²), reported by `capabilities()` and `preflight()` |
| GPU | **yes** (`device="cuda"`) + **single-node multi-GPU** (`devices=[...]`) |
| 1-RDM / 2-RDM | yes / yes |
| basis mode | **TPB (default) or GDB**, via `mode=`; spinless is a third, always-scalar-`nelec`-detected path, not a `mode=` value |

## Options

```python
solver = create_solver(
    "torch_sci",
    symmetrize_spin=True,
    hamiltonian=ham,
    device="cuda",  # single device; ignored when devices= is set
    devices=["cuda:0", "cuda:1"],  # opt into batch-parallel multi-GPU
    nroots=1,  # or n_roots; >1 for excited states
    chunk_size=2_000_000,
    davidson_block_size=20,  # Davidson subspace; None auto-tunes
    max_davidson_iters=20,  # outer restart count (work ~ iters * block_size)
    davidson_tolerance=1e-4,
    mode="tpb",  # forwarded to torch_sci_solver; or "gdb"
    # interface knobs (all optional, additive defaults)
    unknown_call_kwargs="ignore",  # "ignore" | "warn" | "error"
    memory_budget_bytes=None,  # per-rank cap: auto-tile, GDB enumeration, Davidson
    table_dtype="auto",  # "auto" | "int32" | "int64" (TPB tables)
    compact_eri="auto",  # "auto" | "on" | "off"
    triton_block=None,  # positive multiple of 16, or None
    diagnostics={},  # dump_dir / matvec_profile / profile_blocks
    deterministic=False,  # True: bitwise-reproducible debug mode (slower on CUDA)
    ab_factorization="off",  # "off" | "cholesky" (factorized TPB αβ pass)
)
```

`mode`, `matvec_impl`, `gdb_kwargs`, and any other `make_torch_sci_solver` kwarg pass
straight through. The `sci_solvers` wrapper additionally accepts the generic aliases
`compute_rdm2` (for `compute_full_rdm`), `maxiter` (for `max_davidson_iters`) and
`tol` (the per-call Davidson residual).

## Public interface: `config`, `capabilities`, `preflight`

The solver object (and the `src/sci_solvers` wrapper, which delegates) exposes three
discovery helpers so workflows do not need to reach into private attributes:

```python
solver.config  # {"requested": {...}, "resolved": {...}}, JSON-friendly
solver.capabilities()  # per-mode support map
solver.preflight(ci_strings, nelec, norb=norb, **call_kwargs)  # plan, no solve
```

- `config["requested"]` is the normalized constructor input, including
  `gdb_kwargs` defaults. `config["resolved"]` lists the effective `mode`,
  `matvec_impl` and `matvec_impl_requested`, dtype names, effective device(s),
  the distributed launch config (or `None`), `gdb_kwargs`,
  `probe_state_types` (the cached-state class each path accepts), and the
  new knobs. `json.dumps(solver.config)` works.
- `capabilities()["modes"]` has one entry per `tpb` / `gdb` mode plus a
  `spinless` entry (selected by a scalar `nelec`, not by `mode=`): layouts,
  dtypes, multi-root, warm start / `init_vectors`, the cached probe-state type
  (`probe_state_type`) and whether HF augmentation applies, full RDMs,
  `<S^2>`, the spin penalty, distributed matvec/tile, the GDB
  block matvec / structural basis / move cache, Triton pass coverage,
  `table_dtype`, the accepted per-call keywords and the configured
  unknown-kwarg policy. Unsupported physics carries a reason:
  `spin_squared_unsupported_reason` / `spin_penalty_unsupported_reason` on the
  spinless entry explain that S² needs the α/β partition the flat basis lacks.
- `preflight(...)` runs every call-time gate — unknown kwargs, `max_cycle`,
  probe/`init_vectors` normalization and warm-start basis union, basis
  dedup/popcount checks, pair-shape and structural gates, `n_roots` vs
  dimension and the distributed matvec/backend gates —
  then returns the plan without running Davidson. The plan reports `basis`,
  `particle_sector`, `matvec_impl`, `tile_spec`/`tile_grid`/`tile_source`, one
  entry per subspace with `dimension`, `path` (`local` / `replicated` /
  `single_process`), `start` (`warm`/`cold`), `warm_source`
  (`probe_state`/`init_vector`), `n_dets_added` on the spinless path,
  `rdm_path` (whether a full RDM was requested) and `rdm_accumulation`
  (`partitioned` / `replicated`, how the distributed finish path builds it),
  plus the cached `probe_state_type`, `build_output_states`
  (whether the call returns a `(results, states)` tuple) and the spinless
  `spin_sq_unsupported_reason` / `compute_spin_squared_unsupported_reason`.
  Errors are the same ones the solve would raise. `norb=` is inferred from
  the highest set bit when omitted; pass it for wide (limb) strings.
  Each subspace entry also carries `capacity`, an estimate of one rank's
  memory: `per_rank_bytes` (Davidson block, matvec work vectors, diagonal, ERI,
  an upper bound on the TPB tables, the explicit pair list, the flat layouts'
  gathered input), `total_bytes`, `device_free_bytes` (read only where that
  creates no device context: CPU, MPS, or a CUDA device this process already
  initialized; `None` for a distributed or multi-device parent),
  `budget_cap_bytes`, `fits`, the `davidson_block` the solver would pick (the
  solver's own sizing rule at each path's length and vector count, so with a
  binding `memory_budget_bytes` it is the block the solve uses),
  `davidson_block_basis` (`explicit`, `budget_cap`, `device_free_estimate`),
  `approximate` (set where the solve sizes from quantities preflight cannot see:
  the cost-weighted TPB row partition, spinless tile stage buffers, a rank's own
  free memory, the after-setup free) and `max_dimension_per_rank` at that block.
  Build-time temporaries, RDM accumulators and the result hand-back are not
  counted.

## Unknown per-call keywords

Per-call keywords this engine does not recognize are governed by
`unknown_call_kwargs`: `"ignore"` (default; the historical behaviour),
`"warn"` (a `UserWarning` naming them) or `"error"` (`TypeError`). The accepted
set is `tol`, `de_tol`, `max_davidson_iters`, `spin_penalty_lambda`,
`init_vectors`, `probe_method`, `probe_kwargs` and `probe_state`; `max_cycle` is
recognized only to raise its targeted rejection. The `qiskit_addon_sqd`
compatibility factory keeps its `apply_ab_projection=False` default but now
warns that `TorchSCISolver`'s native default is `True`; pass it explicitly to
silence the warning.

## Memory, table and diagnostic knobs

- `memory_budget_bytes` is a single per-rank byte cap. It feeds the distributed
  auto-tile policies, the default GDB/spinless enumeration workspace (when
  `gdb_kwargs["budget_bytes"]` is unset) and the automatic Davidson block
  sizing. `None` keeps the current free-memory behaviour.
- `table_dtype="int32"` halves the TPB excitation-table payloads (offsets stay
  int64); the build raises if an index does not fit. GDB and spinless tables
  are always int64, and the Triton kernels require int64 payloads, so
  `"int32"` is rejected with `mode="gdb"` or a resolved Triton matvec.
  `"auto"` keeps int64.
- `compact_eri="on"` / `"off"` forces the 16x-smaller compact spatial ERI
  store on or off; `"auto"` keeps the historical heuristic (compact unless the
  Triton TPB kernels need a flat dense tensor, and only at ≥ 64 spin-orbitals).
- `triton_block` (a positive multiple of 16) overrides the distributed TPB
  tile operator's launch block; `None` keeps its built-in 128. The
  single-process Triton entry/doubles kernels keep their own block.
- `diagnostics={"dump_dir": ..., "matvec_profile": True, "profile_blocks": True}`
  enables the same hooks as `TORCH_SCI_DUMP_DIR`, `TORCH_SCI_MATVEC_PROFILE`
  and `SCIS_PROFILE_BLOCKS`; the environment variables are still honoured.
- `deterministic=True` runs every call under
  `torch.use_deterministic_algorithms(True, warn_only=True)` (the previous state
  is restored afterwards): CUDA `index_add`/`scatter_add` take their
  deterministic implementations, the atomic Triton entry kernels (spinless,
  streamed GDB) and the cuSPARSE products of the factorized αβ pass fall back to
  them, and `CUBLAS_WORKSPACE_CONFIG` defaults to `":4096:8"` when CUDA is not yet
  initialized (ranks and pool workers set it in their own bootstrap). A CPU or
  CUDA run is then bitwise reproducible for a fixed device set, backend, rank
  count and NCCL algorithm choice (pin `NCCL_ALGO` across runs); a different rank
  count still reorders the global reductions. MPS has no deterministic
  accumulating `index_put`, so MPS runs are not bitwise reproducible (the
  constructor warns). The output-stationary Triton kernels (TPB, tile grid, GDB
  block doubles) are deterministic already and keep running. The switch is
  process-wide during a call: do not run solvers with different settings in
  concurrent threads of one process. Slower on CUDA; meant for debugging.

`ab_factorization="cholesky"` evaluates the TPB αβ doubles pass in factorized
form. The spatial ERI supermatrix is decomposed once per Hamiltonian by pivoted
Cholesky, `(pq|rs) = Σ_P B_P[pq] B_P[rs]` to `ab_factorization_tol` (default
`1e-10`; no auxiliary basis, an indefinite synthetic ERI takes a signed
eigendecomposition), and the pass applies `Σ_P A^α_P C (A^β_P)^T` with
`A^σ_P` the σ singles table weighted by `B_P`. It is exact on the product basis
and costs `2·k·N·n_s` multiply-adds against the direct kernel's `N·n_s²` pair
evaluations (`k` factors, `n_s` singles per string), so it pays off for
low-rank molecular integrals and larger orbital spaces: on H₂O/cc-pVDZ
(24 orbitals, `k = 277`) one CPU matvec drops from 1.82 s to 0.88 s at
3.6·10⁵ determinants and breaks even at 9·10⁴. The same-spin channels and
the ERI storage are unchanged. It applies to real single-process TPB solves
(both matvec backends, CPU/CUDA/MPS); `mode="gdb"`, complex dtypes,
`distributed=` and a scalar-`nelec` (spinless) call reject it. The distributed
rejection is a scaling result, not a gap: on the owner-computes layouts a rank's
α singles read ket rows of every other rank, so either the β products
`C (A^β_P)^T` are exchanged once per factor (`k` times the direct path's traffic)
or each rank applies `A^β_P` to every received row, which costs about
`k·N·n_s` per rank regardless of the rank count, against the direct kernel's
`N·n_s²/P`. The factorized pass therefore only wins below `P ≈ n_s / k` ranks
(two or three at 60 orbitals).

`gdb_kwargs["move_cache_bytes"]` is the current spelling for the move-table
cache budget; `gdb_kwargs["tile_move_cache_bytes"]` remains as a deprecated
alias (a `DeprecationWarning` names the replacement). On the GDB tile grid,
`gdb_kwargs["needed_staging"]` (`"off"` / `"auto"` / `"on"`) chooses between
staging each peer's whole tile and staging only the ket-group rectangles the
owned fragments read; `"auto"` keeps the whole tile when the needed plan would
stage most of it.

A tile grid walks the exact Cartesian block decomposition of the pair basis. A
recovered or expanded GDB pair list (for example one carried through a
selected-CI loop) is an irregular subset of the product of its unique α/β
strings, so its exact decomposition splits into one group per distinct partner
set — exact, but with a per-group overhead that dominates the matvec. For such
a list the tiled path solves the *masked product*: it embeds the list in the
full Cartesian product of its strings, keeps the added pairs at zero amplitude,
and masks every Davidson product and search direction, which is exactly
`P H P` on the caller's basis. The returned amplitudes, pair strings, RDMs,
`<S^2>` diagnostic and warm-start snapshot are taken on the caller's list.
The product may be at most 4x the list size; a sparser list keeps its exact
decomposition (or raises when it has none) rather than silently multiplying
its memory. `apply_ab_projection` and a replicated (`matvec="replicated"`)
solve keep the exact decomposition.

## TPB vs GDB

The `mode=` argument decides what subspace a batch of CI strings defines:

- **TPB** (*tensor product basis*, default) — `(strs_a, strs_b)` are independent per-spin
  sets; the basis is their **Cartesian product**, dimension `len(strs_a)·len(strs_b)`.
  1,000 sampled determinants become ~10⁶.
- **GDB** (*general determinant basis*) — `(strs_a, strs_b)` are **parallel length-N
  lists**; index `i` is one determinant, dimension N. 1,000 stay 1,000.

GDB has sharp edges: do **not** feed it `bitstring_matrix_to_ci_strs` output (it dedups
each spin independently → wrong subspace, silently), it returns a flat `(N,)` amplitude
array, and its Triton path tiles the block-mode same-spin doubles and the streamed
singles and doubles tables (in block mode `matvec_impl="triton"` keeps the batched
kernels for the singles and the diagonal).
Extract the α/β halves yourself and pass those length-N arrays; use
`torch_sci_solver.diagonalize_fermionic_hamiltonian_pairs` for a GDB outer loop. See the
repo README's *TPB vs GDB* section for the full contract.

For excited states in GDB, set `nroots=k` and read
`solver.sci_solver_callable_eigen_states(...)[batch].states`: each root contains an
energy, flat `(N,)` amplitudes and orbital occupancies. Full RDMs, spin diagnostics
and warm-start snapshots remain ground-state only, matching TPB. The ground-state
convenience APIs and configuration recovery loop still consume `states[0]`.

The requested root count must fit in the deduplicated CI space, the Davidson subspace,
and any enabled projected sector; otherwise the solve raises `ValueError`. Allow room
for expansion with `davidson_block_size` preferably at least three times `nroots`.
Set `apply_ab_projection=False` to request the lowest states across both α/β-swap
sectors; with projection enabled, only the selected sector is solved when the basis
is closed under the swap.

The GDB/TPB parity tests in `tests/test_torch_sci_gdb_multiroot.py` compare energies,
amplitudes (after convention/sign alignment), occupancies and residuals on Cartesian
products up to 4,900 determinants and six roots. They cover CPU, CUDA and Apple MPS
(using `dtype=torch.float32`, since MPS has no float64 support); unavailable devices
are skipped. Run `uv run pytest tests/test_torch_sci_gdb_multiroot.py -k scaled` to
include the larger cases; the 4,900-determinant cases are marked `slow`.

The dedicated GPU suite, `tests/test_torch_sci_gdb_gpu.py`, runs four-root comparisons
on each available CUDA/MPS backend and skips unavailable backends. Its Cartesian
products contain 2K, 10K and 20K determinants; its non-product sets contain 2K, 10K
and 50K. The latter compare Torch GDB against both ARPACK and LOBPCG. All runs check
energies, amplitudes aligned by a global phase per root, orthogonality and residuals
recomputed on CPU in float64 or complex128. Complex fixtures carry nonzero imaginary
parts in both the one- and two-electron integrals. No dense Hamiltonian is assembled.

Run it with timing tables and machine-readable JUnit properties:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -v -s \
  tests/test_torch_sci_gdb_gpu.py --junitxml=gpu-comparisons.xml -o junit_family=xunit1
```

To run only the six complex CUDA comparisons (skipped when CUDA is unavailable):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run pytest -svx tests/test_torch_sci_gdb_gpu.py -k "cuda and complex" --durations=0
```

Use `-k "cuda and real"` for the original real-valued CUDA cases, or add `and 2k`
to either selection for a smaller run. Each case solves once per compared solver.
CPU residual validation prints progress and is excluded from eigensolve timings.

All cases carry the `gpu` and `slow` markers. CUDA uses float64 and complex128;
MPS uses float32 for real cases and skips the complex128 comparisons.
All solvers within a comparison use the same GPU matvec precision and requested
tolerance. ARPACK/LOBPCG retain their SciPy CPU eigensolvers, including transfers
to/from GPU matvecs. The portable `batched` Torch kernels are used throughout.
For CUDA, select a card with `CUDA_VISIBLE_DEVICES` before running the suite.

Tables report one cold solve per solver after GPU initialization, with synchronization
around timing boundaries. Total time includes setup and result observables;
eigensolve time excludes these (general solvers compute full 1-RDMs, whereas Torch
only computes occupancies here). `solve/GDB` is the eigensolve time divided by Torch
GDB's time; values above one mean the other solver took longer. JUnit properties
also record device, library versions, precision, tolerance, matvec counts and numerical
errors. These are measurements, with no hardware-dependent speed assertions; they
are not repeated, warmed-up performance benchmarks.

## Multi-GPU

`devices=["cuda:0", "cuda:1", ...]` dispatches each SQD subspace (batch) to its own GPU
via a thread-pool fan-out — no cross-GPU communication, since the batches are
independent. Speedup is bounded by `min(num_batches, len(devices))` and near-linear only
when `num_batches` ≥ and is roughly divisible by `len(devices)`; an awkward remainder
leaves GPUs idle in the last wave, and `num_batches=1` sees no speedup. A single large
diagonalization still runs on one GPU. `devices[0]` becomes the primary device and the
plain `device=` argument is then ignored.

## 64-orbital strings using orbital 63

A 64-orbital CI string is one unsigned 64-bit word. Every input form is read as
that unsigned value: uint64 arrays, int64 arrays of signed bit patterns, and
Python ints up to `2**64 - 1` (a negative int is its signed int64 pattern). The
basis is sorted in integer order, so the HF-like string comes first. The returned
`SCIState` / `PairSCIState` strings are int64 as before, or uint64 for both spins
when any string uses orbital 63. qiskit-addon-sqd's spin-symmetrized carryover
(`symmetrize_spin=True`) concatenates its strings with an int64 array and so
turns uint64 into float64; avoid that option for a 64-orbital basis that uses
orbital 63.

## Apple MPS

`device="mps"` runs the single-process solver on the Apple GPU with the batched kernels.
MPS has no float64 or complex128, so pass `dtype=torch.float32` (or `torch.complex64` for
a complex Hermitian Hamiltonian); the constructor rejects a float64/complex128 dtype on
MPS. `davidson_tolerance` is raised to `1e-5` with a warning, because an absolute
float32 residual cannot go much below its own precision floor — roughly
`sqrt(nnz)·‖H‖·eps(float32)`, about `5e-4` at a few hundred Ha and 5·10⁴
determinants. A sub-float64 solve whose residual plateaus there (no 10% improvement
over 15 restarts) stops with `info["stagnated"]` instead of burning the remaining
`max_davidson_iters`; every missed-root search, a single-root Davidson, inherits the
same stop. The reported energy does not wait for the tolerance (it is the float64
Rayleigh quotient below), so on a large space pass a reachable
`davidson_tolerance` (1e-3, as the workflows do) rather than a floor the solve
cannot reach; what the tolerance buys is vector accuracy, roughly `tol/gap`.

The reported energy is a float64 Rayleigh quotient, not the float32 Ritz value:
after the MPS solve, each root's `⟨v|H|v⟩/⟨v|v⟩` is evaluated once with a float64
(complex128 for a complex solve) matvec on CPU copies of the integrals, tables and
basis, so the energy is float64-accurate up to the vector's own convergence
(`‖r‖²/gap`), while the float32 Ritz value is biased by roughly `|E|·eps(float32)`
(1e-4–1e-3 Ha at |E| ≈ 2·10³ Ha). With a `spin_sq` penalty the quotient of the bare
Hamiltonian is the physical energy directly. This is `davidson_fp32_energy`'s
`"rq64"` (the `"auto"` default below `tol = 1e-3`); `"ritz"` keeps the float32 value
and skips the CPU matvec. The vector, occupancies, RDMs and ⟨S²⟩ are unchanged.

The vector itself stays float32 and is converged only to `davidson_tolerance`, so its
amplitudes are resolved to roughly `tol/gap` (about 1e-5 for a unit gap), not to
`eps(float32)`. A selection that thresholds amplitudes below that (an SQD trim cut,
GDB pair screening at 1e-8) sees noise there and can differ by a determinant between
two MPS runs; run such a selection on CPU or CUDA in float64.

The missed-root check orthonormalizes the converged roots with CholeskyQR2 on MPS:
the `k×k` Gram matrix `QᴴQ` is factored in float64 on the CPU and `Q ← Q·R⁻¹`
applied twice on the device. torch's MPS `linalg.qr` aborts the process on tall
blocks (from about 50,000×10), which a 10-root check reaches at about 50k determinants.

Supported on MPS: TPB and GDB (streamed and block matvec, explicit pair lists), the
spinless path with real or complex integrals, strings wider than 64 bits, multi-root
with the missed-root check, full RDMs, `<S²>`, the `spin_sq` penalty, the αβ-swap
projection, `probe_method="warm_start"`, `init_vectors`, `compact_eri` and
`table_dtype="int32"`. Rejected at construction with an explicit error:
`matvec_impl="triton"` (CUDA only) and `distributed=` (ranks run on CUDA under NCCL or on
CPU under gloo, and MPS exposes one device with no collective backend). The options
that only exist on the distributed path (a structural GDB basis, the tile
grid) are therefore unavailable too, and `capabilities()` on an MPS solver reports this.

## Complex integrals

Pass `dtype=torch.complex128` (or `complex64`) to solve a complex Hermitian
Hamiltonian; the default stays `torch.float64` (real, unchanged behavior).
The opt-in is explicit, not auto-detected: complex `hcore`/`eri` handed to a
solver still constructed with the real default raises rather than silently
dropping the imaginary part. Non-Hermitian complex integrals also raise.
`matvec_impl="triton"` supports complex dtypes (the kernels read interleaved
real pairs), so `matvec_impl="auto"` selects it on CUDA for complex problems
too. `compute_full_rdm` / `compute_spin_squared`,
multi-root (`n_roots > 1`) in both TPB and GDB, and the `spin_sq` penalty work
under complex dtype.

## Spinless (flat spin-orbital) integrals

For a genuinely spinless Hamiltonian (GHF / spin-orbit-coupled / generalized-spinor,
e.g. `sci_solvers.spinless_ci_problem`'s consumers) there is no `(n_a, n_b)`
factorization — a single flat block of `nmo` spin-orbitals with a total particle
number `N`. `TorchSCISolver` detects this from `nelec` being a **scalar** total `N`
(not an `(n_a, n_b)` tuple) rather than a `mode=` you set:

```python
solver = create_solver("torch_sci", dtype=torch.complex128, device="cuda")
# ci_strs_batches: list of flat `strs` arrays (NOT (strs_a, strs_b) tuples)
results = solver.sci_solver_callable_eigen_states(ci_strs_batches, hcore, eri, nmo, n_particles)
```

Real or complex: a real spin-orbital Hamiltonian runs with a real `dtype`, and
`dtype=torch.complex128` (or `complex64`) is needed only when the integrals carry
imaginary parts; complex integrals with a real `dtype` are rejected rather than
truncated. Multi-root
(`n_roots > 1` at construction) works at the same level as the spatial path —
`davidson_torch` is basis-shape-agnostic, so passing `n_roots=self.n_roots`
through unchanged is all it takes.

Warm start and probe states work on the flat path. A cold solve with
`probe_method="warm_start"` returns `(results, states)`, where each state is a
`TorchSpinlessSCIState` holding the explicit determinant strings, the flat
eigenvector, the energy, `n_orb`/`n_particles`, the integrals fingerprint and
the solver knobs. Passing it back as `probe_state` on a superset basis embeds
the amplitudes by determinant-string lookup (limb-aware, so bases wider than
64 spin-orbitals work), normalizes globally and falls back to the cold seed if
nothing overlaps. There is **no HF augmentation**: the path has no α/β split,
so there is no HF α/β determinant to anchor; the cached state is the ground
state of the basis the caller passed. `init_vectors` remains supported and is
mutually exclusive with `probe_state`.

`spin_sq` and `compute_spin_squared` are refused as an explicit **physics
limitation**, not a scope cut: `S²` is built from the α/β partition
(`N̂_α − N̂_β`) that a flat spin-orbital basis does not have, so neither the
`⟨S²⟩` diagnostic nor a target spin sector is defined. The same reason is
returned by `capabilities()["modes"]["spinless"]`
(`spin_squared_unsupported_reason`, `spin_penalty_unsupported_reason`) and by
`preflight()` (`spin_sq_unsupported_reason`,
`compute_spin_squared_unsupported_reason`), so a workflow can gate before
calling solve. `compute_full_rdm` works; it is just the full 1- and 2-RDM, not
an `S²` value. Reuses the
same flat-determinant Slater-Condon primitives GDB pair mode already uses
(`one_excite_batch`/`two_excite_batch`, spin-orbital-index-generic) and the
unmodified `davidson_torch` — only the excitation-table generator, matvec
scatter, and RDM builder are new, in
`torch_sci_solver.torch_impl.{spinless_helpers,mult_spinless,rdm_spinless}`.
Validated against `sci_solvers.spinless_ci_problem`'s independent numpy/pyscf
implementation to ~1e-9 across matvec, Davidson, and both RDMs (see
`packages/torch-sci-solver/tests/test_spinless_*.py`).

## Warm start from a previous state (`init_vectors`)

Pass `init_vectors=` per call to start each batch's Davidson from a given vector
instead of the cold HF-like seed. The intended use is the SQD recovery loop: project
the previous iteration's ground state onto this iteration's strings and pass it in.
Together with a loose per-call residual (`tol=3e-3`) this cut the eigensolve to about
¼ of a cold solve at 10⁸ determinants. `project_init_vector` (below) builds the
projected vectors.

```python
results = solver.sci_solver_callable_eigen_states(
    ci_strs_batches,
    hcore,
    eri,
    norb,
    nelec,
    init_vectors=[vec_0, None, vec_2],  # one entry per batch; None = cold start
    tol=3e-3,
)
```

Each entry is a 1-D array laid out like the `sci_state.amplitudes` this solver returns
in that mode, indexed by the strings you pass in this call:

| mode | length | index | signs |
|---|---|---|---|
| TPB | `len(strs_a) * len(strs_b)` | `ia * len(strs_b) + ib` | spin-block |
| GDB | `len(strs_a)` (one per pair) | pair index `i` | interleaved-Wick |
| spinless | `len(strs)` | string index | — |

So a previous result's amplitudes can be passed straight back when the strings are
unchanged. The solver reorders the vector to its internal (sorted, deduplicated)
basis. A duplicated string takes its first occurrence's amplitude, and a string you
did not pass starts at 0.

The vector is then αβ-projected (when `apply_ab_projection` is active) and normalized.
If less than `1e-8` of its norm is left, it has no overlap with this basis, and the
batch falls back to the cold start. `stats["per_call_log"][-1]["subspaces"][i]["warm"]`
on the engine (`solver._solver`) records which batches actually started warm (TPB and
GDB; the spinless path writes no per-call log entry).

It works with multi-root (see below), `spin_sq`, complex dtype,
`seed_strategy="connected"` and multi-GPU. It raises `ValueError` on a length
mismatch, or when combined with `probe_state` (both set the start vector). Unlike
`probe_method="warm_start"`, the new basis need not be a superset of the old one.

### Probe states (`probe_method="warm_start"`)

`probe_method="warm_start"` is the superset-only variant. A cold call returns
`(results, states)`; pass the state(s) back as `probe_state` (one per batch, or a
single state for a one-batch call) on a basis that contains every cached string.
The engine then reuses the converged ground eigenvector directly instead of
projecting caller amplitudes:

| mode | cached state | new basis must contain |
|---|---|---|
| TPB | `TorchSCIState` (`adets`/`bdets`, interleaved-Wick vector) | every cached α- and β-string |
| GDB | `TorchPairSCIState` (`(α, β)` pair lists) | every cached pair |
| spinless | `TorchSpinlessSCIState` (explicit determinant strings, flat vector) | every cached determinant |

The three state types are not interchangeable; passing, say, a `TorchSCIState`
to a scalar-`nelec` call is a `TypeError`. A warm extension returns only
`results` unless `probe_kwargs={"build_output_state": True}`, which returns
`(results, states)` for chaining. The cached state records the Hamiltonian
fingerprint and solver knobs: a changed Hamiltonian, orbital count, particle
number or hard knob raises `ValueError`; a Davidson-only knob mismatch is a
warning (the extension converges at the current knob values). The spinless
path has no HF augmentation — `probe_kwargs={"include_hf_in_core": ...}` warns
and is ignored — because there is no α/β HF determinant to anchor, and its
cached state is simply the ground state of the caller's basis.

GDB pair keys resolve each pair through its `(α_uid, β_uid)` key in the basis'
unique-string axes, with each string normalized to its limb rows. Warm/probe
states and `init_vectors` therefore work on explicit pair bases wider than 64
orbitals as well: the cached `TorchPairSCIState` stores one limb row per pair,
and the embedding/init mapping look rows up exactly as they do for a
single-word basis. A structural (block-described) pair basis accepts
`probe_method="warm_start"` when every cached pair lies in the block spec (the
spec is the caller's whole basis, so it is never extended or HF-augmented), and
`init_vectors` indexed by structural pair: block by block, α-major within a block,
the order of the `sci_state.amplitudes` a structural solve returns.

### Multi-root: one start vector per root, and the missed-root check

With `n_roots > 1` an entry may also be a `(k, len)` array, `1 <= k <= n_roots`: one
start vector per root, lowest first, each laid out as in the table above. A 1-D entry
still seeds root 0 only.

A converged multi-root solve only shows that its pairs are eigenpairs, not that they
are the lowest ones. Davidson does not leave the symmetry sectors its seed spans, so a
warm start can converge without a lower excited root while every residual reads small:

- from the exact ground state alone (a 1-D entry), an N2 SQD batch returned the
  second singlet as root 1, 0.45 Ha above the triplet that is the true root 1;
- from start vectors that are not the lowest roots of this space (a previous batch, or
  the smaller space of an Ext-SQD or PoolGrow final solve), roots in a sector the
  vectors miss are skipped.

So a warm multi-root solve also runs a **missed-root check**. Once converged, it runs a
single-root search on the space orthogonal to the roots found, started from the lowest
determinant not already covered. If the search falls below the highest root returned,
that root is added to the seed and the solve repeated. `davidson_missed_root_check`
sets it: `"auto"` (default) checks exactly the warm multi-root solves, `True` also
checks cold ones (a cold solve can miss a root at a loose tolerance), and `False` never
checks. Each search is recorded in the per-subspace stats as `missed_root_checks`.

The check costs 12–60 matvecs per solve when it finds nothing (more
when it finds a root and re-solves: 91 on the one 10⁸ run where it did), so it pays off
only where the warm start saves more:

| | warm from every root + check | cold + check |
|---|---|---|
| 10⁸ determinants, 3 roots, tol 1e-4 (5 hardware runs, norb 21–35) | 15–43 matvecs | 57–121 matvecs, 1.8–2.9× slower |
| a few hundred to a few thousand determinants (H6, N2 batch) | often more matvecs than cold | — |

Warm-starting root 0 alone saves nothing at 10⁸ (82 against 80 matvecs with the
check); the saving comes from passing every root.

### Correction and restart options

Two opt-in constructor settings change the Davidson step itself; both default to the
existing behaviour.

- `davidson_correction="olsen"`: Olsen's correction (Olsen, Jørgensen & Simons, CPL
  169, 463, 1990), orthogonal to the current Ritz vector. It gave no speed-up on the
  molecules tested (one matvec saved in 3 of 42 warm solves at 10⁸), but it converged a
  stalled multi-root case where the plain correction collapsed onto the subspace.
- `davidson_restart="reuse"` / `"thick"` (single root): at a restart, reuse the stored
  `H x` instead of recomputing it, and with `"thick"` also keep the previous Ritz
  vector. They save matvecs only on a solve that restarts (about 10% at block size 8 to
  12); warm solves usually converge before the first restart.

`compute_energy` re-indexes the vectors when it sorts the strings (`sort_ci_strs=True`).
The other solvers raise `NotImplementedError` if a vector is passed, and accept
`init_vectors=None`. `supports_init_vectors(name)` in the factory says which solvers
honor it.

### Using it from an SQD loop

Let sci-solvers build the vectors. Don't hand-roll the projection: the sign convention
differs by mode, and getting it wrong only slows the solve, it never raises. Passing
GDB a spin-block vector took 11 matvecs instead of 1 on H6.

```python
from sci_solvers.energy_calculation_base import project_init_vector
from sci_solvers.energy_solvers.factory import supports_init_vectors

if settings.warm_start and supports_init_vectors(settings.solver_method):
    init_vectors = [
        project_init_vector(best_result.sci_state, ci_strs, solver_mode, device="cuda")
        for ci_strs in ci_strs_batches
    ]
    kwargs = {"init_vectors": init_vectors}
results = solver.sci_solver_callable_eigen_states(ci_strs_batches, ..., **kwargs)
warm = [s["warm"] for s in solver.last_call_subspace_stats()]  # confirm it was used
```

- **`project_init_vector(prev_state, ci_strs, mode)`** works for TPB, GDB and spinless.
  It copies amplitudes on the shared determinants, puts zeros on new ones, and
  normalizes. It returns `None` (a cold start) when nothing overlaps. It also covers
  spinless determinants wider than 64 bits. Pass the solve device at 10⁸: the key
  lookup takes ~3.4 s per 10⁷ GDB pairs on CPU.
- **`supports_init_vectors(name)`** tells a caller whether a solver honors the vectors.
  Use it instead of checking a sci-solvers version: it asks the solver itself, so it
  cannot drift out of date the way a version check can.
- **`TorchSCISolver.last_call_subspace_stats()`** gives the per-batch `warm`,
  `n_matvecs`, `davidson_dtype` and `davidson_fp32_energy` for the last call, so the
  caller doesn't have to read the engine's private stats.
- **A loose stop and final-only RDMs need nothing new.** The loose stop is `tol=3e-3`
  per call or `davidson_tolerance=3e-3` at construction. Final-only RDMs use
  `compute_full_rdm=False`.
- **fp32 (`davidson_dtype=torch.float32`)** composes with all of the above. At the
  loop's `tol=3e-3` the `"auto"` energy mode picks `"ritz"`, so a warm solve pays no
  extra matvec.

## fp32 Davidson subspace (`davidson_dtype`)

At 10⁸ determinants, Davidson's subspace (`C` and `HC`, each `davidson_block_size × N`)
takes most of the GPU memory. `davidson_dtype=torch.float32` stores the subspace in
fp32, while the matvec still applies H in fp64. Its first step, `hii * v`, promotes the
fp32 vector, so only reading the vector is fp32. This saves memory. It does **not**
speed anything up: a matvec with an fp32 input costs 1.01–1.02× an fp64 one.

```python
solver = create_solver(
    "torch_sci",
    symmetrize_spin=False,
    device="cuda",
    davidson_dtype=torch.float32,  # opt-in; default None keeps fp64
)
```

Measured on A100, 10⁸ determinants:

| block, subspace dtype | peak GPU memory | matvecs at 3e-3, cold / warm | energy |
|---|---|---|---|
| 20, fp64 (default) | 44.3–45.1 GiB | 7 / 5 | reference |
| 20, fp32 | 28.6–29.0 GiB | 7 / 5 | identical |
| 8, fp64 | 26.4–27.2 GiB | 7 / 5 | identical |
| 8, fp32 | 19.6–20.0 GiB | 7 / 5 | identical |

Try a smaller `davidson_block_size` first. Use fp32 when the grid still does not fit
at the block size you need.

- **`davidson_fp32_shift`** (default `"hf"`): Davidson diagonalizes `H − shift` with `shift =
  min(hii)`, so its fp32 Ritz values are about the size of the correlation energy
  (~0.1 Ha) rather than the total energy. The shift does not change the eigenvectors
  or the residual, so `tol` means the same thing. A float sets the shift explicitly.
  `None` turns it off, which is unsafe: in the benchmarks it stalled at a residual of
  ~1e-5 and ended 0.3 mHa off.
- **`davidson_fp32_energy`** (default `"auto"`): how the returned energy is formed.
  - `"rq64"` takes an fp64 Rayleigh quotient of the returned vector. It costs one
    extra matvec per root and is exact: it matches fp64 to ~1e-12 Ha.
  - `"ritz"` skips that matvec and reports the fp32 Ritz value plus the shift. Its
    error is the energy of an fp32-rounded wavefunction, so it does not shrink with
    `tol`, and it grows with the basis. It is 3.5e-7 Ha on H6 (400 determinants),
    1.4e-6 Ha on N2 at 2.2 Å (14,400), and about 1e-7 Ha in the 10⁸ benchmarks. That
    is three orders of magnitude below chemical accuracy.
  - `"auto"` chooses for each call from that call's `tol`. At `tol ≥ 1e-3` it uses
    `"ritz"`: at that residual fp64 itself is already off by more than the Ritz error
    (1.8e-5–1.8e-4 Ha on N2), so the extra matvec buys nothing. That covers the SQD
    warm loop at 3e-3, where a warm solve takes only 1–4 matvecs. Below 1e-3 it uses
    `"rq64"`, so a tight final solve stays exact. The same solver can therefore serve
    both.

  Measured on N2 at 2.2 Å (multireference), energy error against an exact reference:

  | `tol` | fp64 | fp32 `"rq64"` | fp32 `"ritz"` | `"auto"` picks |
  |---|---|---|---|---|
  | 3e-3 | 1.8e-4 | 1.8e-4 | 1.8e-4 | `"ritz"` |
  | 1e-3 | 1.8e-5 | 1.6e-5 | 1.9e-5 | `"ritz"` |
  | 3e-4 | 1.6e-6 | 1.6e-6 | 4.1e-6 | `"rq64"` |
  | 1e-4 | 1.5e-7 | 1.5e-7 | 1.9e-6 | `"rq64"` |
- The returned vector, occupancies, RDMs and `SCIState` are always fp64.
  `stats["per_call_log"]` records `davidson_dtype`, `davidson_fp32_shift` and the
  resolved `davidson_fp32_energy` for each batch.
  It works with TPB and GDB, `init_vectors`, `spin_sq`, multi-root, the αβ projection
  and multi-GPU. With `davidson_block_size=None`, the automatic block sizing accounts
  for the 4-byte elements.
- The narrow subspace is `torch.float32` over `torch.float64`, or `torch.complex64`
  over `torch.complex128`, on every layout (TPB, GDB, spinless). Every other
  combination raises.
- **Stagnation stop:** a float32/complex64 subspace has a residual floor set by its
  own precision (roughly `sqrt(nnz)·‖H‖·eps32`). Once the worst root's residual shows
  no 10% improvement over 15 restarts, the solve stops early and reports `stagnated`
  in its `info` instead of exhausting `max_davidson_iters`. The single-process
  single/multi-root loops and all three distributed Davidson loops apply it; an fp64
  subspace is untouched. A missed-root search that stops this way keeps the
  conservative "undecided → found" mapping, so a slow-but-converging search can cost
  one extra multi-root re-solve.
- **Risk:** parity is measured on HF-dominated molecules. On near-degenerate,
  multireference spectra, check against fp64 before relying on it.

## Notes

- Spin projection is a native penalty (do not stack the generic spin filter on top).
- When a batch fails to converge, its reported energy is overwritten to a sentinel so a
  non-converged root is not mistaken for a good one.
