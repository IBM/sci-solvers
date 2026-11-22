# Distributed-memory state vector for the SBD / `torch_sci` Davidson path

**Status: opt-in feature, wired through the public solver API.**
The default path is unchanged: with no `distributed=` keyword the existing
single-process and batch-parallel `devices=` paths run byte for byte.

The scheme is a distributed selected-basis diagonalization in the spirit of the
SBD-G work (arXiv:2605.01138): the many-body state vector and the Davidson
subspace are partitioned across ranks, the Hamiltonian stays matrix-free, and
the Davidson projection is formed with all-rank reductions. The existing SBD
GPU benchmark line is arXiv:2601.16637.

## 1. API at a glance

`make_torch_sci_solver` (and `TorchSCISolver`) accepts:

```python
solver = make_torch_sci_solver(
    device="cuda",  # ignored per rank when NCCL is chosen
    distributed={
        "world_size": 2,  # ranks (one process per GPU for NCCL)
        "backend": "nccl",  # "nccl" (GPU) or "gloo" (CPU)
        "device_ids": [0, 1],  # NCCL devices, one per rank
        "timeout": 1800,  # seconds
        "budget_bytes": None,  # per-rank memory cap for auto-tiling
        "hand_back": "auto",  # "auto" | "memory" | "memmap"
        "hand_back_bytes": None,  # size threshold for "auto"
        "worker_startup_timeout_s": 600,  # rank init handshake bound
        "checkpoint_dir": None,  # shared-filesystem directory; None disables
        "checkpoint_every": 50,  # Davidson iterations between writes
        "checkpoint_resume": "auto",  # "auto" | "off" | "require"
    },
)
results = solver(ci_strings, h1, eri, norb, nelec, ...)
```

- `distributed=None` / `False` (default): existing behaviour.
- `distributed=True`: all visible CUDA devices (NCCL) when there are at least
  two, else the launcher's `WORLD_SIZE`; with neither, it falls back to the
  single-process path.
- `distributed={"world_size": ..., "backend": ..., "device_ids": ...,
  "master_addr": ..., "master_port": ..., "init_method": ..., "timeout": ...}`:
  explicit launch. `init_method` overrides the default single-node TCP
  rendezvous.
- `tile` (`None` | `"auto"` | `"force"` | `[P_a, P_b]`) selects the 2-D tile
  grid (Section 4.5): `"auto"` applies the memory policy and may keep the 1-D
  split, `"force"` tiles with the auto grid, and `[P_a, P_b]` fixes the grid.
  `exchange_bytes` (positive int) caps one staged tile exchange: an αβ corner
  rectangle larger than the budget is shipped as row bands, so no single P2P
  buffer grows with the halo. The budget is converted to elements with a
  16-byte worst case, and every rank must use the same value.
- `world_size <= 1`, or an initialized group of one rank: fall back to the
  existing single-process solve, byte for byte.
- `budget_bytes` is a per-rank byte cap used by the auto-tiling policies
  (`tile="auto"` for TPB, GDB and spinless). It is combined with the
  constructor's `memory_budget_bytes` by taking the smaller; `None` keeps the
  free-memory estimate. `worker_startup_timeout_s` bounds how long the parent
  waits for every rank's initialization handshake before failing with a named
  error (default 600 s, also settable through `TORCH_SCI_STARTUP_TIMEOUT_S`).
- `hand_back` chooses how a large result crosses the rank boundary:
  `"auto"` (default) gathers in memory below `hand_back_bytes` (default 1 GiB)
  and writes memmapped `.npy` files above it; `"memory"` always gathers;
  `"memmap"` always uses the files. `hand_back_dir` still names the shared
  directory those files live in.
- `checkpoint_dir` (default `None`) enables periodic checkpointing of every CI
  subspace's Davidson state into `checkpoint_dir/subspace_<k>/` (Section 5.6).
  The directory must be on a filesystem every rank can read and write; a
  multi-node run points it at shared storage, exactly like `hand_back_dir`.
  `checkpoint_every` (positive, default 50) sets the number of Davidson
  iterations between writes, and `checkpoint_resume` selects the restart
  policy: `"auto"` (resume a matching checkpoint, cold-start when none exists,
  raise when the state belongs to a different run), `"off"` (ignore any
  existing state and overwrite it) or `"require"` (a missing or incomplete
  checkpoint is an error). An explicit `checkpoint_every` / `checkpoint_resume`
  without `checkpoint_dir` is rejected rather than silently ignored.
- The resolved launch is exposed as `solver.config["resolved"]["distributed"]`;
  `solver.preflight(ci_strings, nelec, norb=...)` reports the per-subspace
  path, tile grid/source and `matvec_impl` a call would use without launching
  ranks, and `solver.capabilities()` reports per-mode support. Validation is
  error-only: a non-positive `world_size`/`timeout`/port, an unsupported
  `init_method` scheme, `device_ids` with gloo, `tile` with
  `matvec="replicated"`, and `exchange_bytes` without a tile spec all raise at
  construction.

The return type is unchanged (`list[EigenStates]`, each with all solved roots),
so callers — including `postprocess_bitstrings`, which forwards
`solver_kwargs` to `make_torch_sci_solver` — do not branch.

**Execution model.** If a process group is already initialized (e.g. a
`torchrun` launch), every rank runs `__call__` in-process. Otherwise the
parent process spawns one worker per rank (`torch.multiprocessing`,
`spawn`), each worker initializes the group and runs the same solver body; only
rank 0's result is returned to the caller, and its cumulative `stats` are copied
back. The parent stays off-GPU (important under `Exclusive_Process`).
Result construction happens on rank 0 only: non-zero ranks participate in the
collectives (occupancies, hand-back barriers) and return an empty
`EigenStates`. In a `torchrun` launch that means rank 0 is the one to read.
Below a size threshold (1 GiB by default) rank 0 gathers the final row blocks
to host; above it every rank writes its rows to memmapped `.npy` files under
`TMPDIR` and the parent process maps them, so no non-zero rank ever holds a
length-`N` array (Section 5.5).

**Supported layouts and modes.** Real and complex α/β TPB (`mode="tpb"`),
GDB (`mode="gdb"`) and flat spinless (a scalar `nelec`) determinant bases,
single- and multi-root. All three layouts default to an owner-computes matvec
(`matvec: "local"`). TPB partitions the α-rows (cost-balanced, Section 3.3):
each rank builds the excitation tables for its own rows once, computes only its
own output block, and exchanges the α-rows it needs through a staged,
double-buffered peer exchange (Section 4.1). GDB and spinless partition the
flat pair/determinant list (Section 4.4): each rank streams only its owned bra
rows, and the operator all-gathers the input once per matvec ("replicated
input, owned output") because the streamed tables cannot be stored per rank.
The αβ-swap projector is applied on the local block in every layout (staged
transpose for TPB, an all-gather for the flat layouts), so the default
symmetric basis no longer falls back. Triton is used automatically on CUDA
ranks for TPB and, with an explicit `matvec_impl="triton"`, for the GDB and
spinless channels (Section 4.2); complex dtypes ride the same kernels as
interleaved real pairs. Under a launcher (torchrun/srun) each rank re-resolves
an `"auto"` request against the device it actually runs on, so an NCCL rank
that was constructed on the host's CPU still selects the Triton kernels; an
explicit `matvec_impl="triton"` on a gloo rank raises instead of being
rewritten to batched. The replicated all-gather matvec (`matvec: "replicated"`)
remains available and is also the fallback for full RDMs on a basis that fits
in one Davidson block or has no tile-capable block decomposition. A
declined auto grid keeps `compute_full_rdm`, `compute_spin_squared` and the
`spin_sq` penalty on the owner-computes path through an implicit `(P, 1)`
partition whenever the tiled path can run; an explicit GDB pair list with no
tile grid applies `spin_sq` and the `<S^2>` diagnostic on the 1-D block instead
(Section 4.5), sharing the matvec's all-gather. The persistent worker pool
is the default (`persistent: true`); `persistent: false` spawns fresh workers
per call.

`davidson_dtype` may name a *narrow* subspace that halves the stored `C`/`HC`
while the matvec stays full precision: `torch.float32` over a real fp64
`dtype`, or `torch.complex64` over `complex128`. The shift/energy knobs
(`davidson_fp32_shift`, `davidson_fp32_energy`) apply to either. Warm starts
(`init_vectors`, `probe_method="warm_start"`) and
`davidson_missed_root_check` work on the owner-computes TPB and GDB paths and
on the spinless flat path; the cached state types are `TorchSCIState`,
`TorchPairSCIState` and `TorchSpinlessSCIState` respectively. The spinless
path embeds the cached eigenvector per rank (contiguous block or scattered
tile) and reduces one global norm before the cold-start fallback, so a warm
extension reproduces a cold solve on the same union basis. A call that lands
on the replicated `davidson_distributed` fallback still rejects the
missed-root check with a clear error rather than dropping it; the narrow
subspace and complex coefficients are supported there too.
`davidson_correction="olsen"` and non-default `davidson_restart` are rejected
under `distributed=` at construction. `devices=` and `distributed=` are
mutually exclusive.

## 2. Why

The current `torch_sci` solver handles two scales differently:

| axis | without `distributed=` | with `distributed=` |
|---|---|---|
| many independent subspaces (SQD batches) | `devices=[cuda:0, cuda:1, ...]` fans **batches** across GPUs in one node (`multi_gpu/`, thread pool) | unchanged |
| one subspace, large determinant count `N` | one process, one device | partition that subspace across ranks (ideally across nodes) |

`N` grows combinatorially with the active space while the integrals grow only
as `L^4`, so a single SQD subspace can outgrow one GPU long before the integrals
do. The concrete target is `N = 10^8`–`10^12` determinants with the *same*
basis contract the solver already uses (TPB `(strs_a, strs_b)` Cartesian
product or GDB pair lists).

No global `Hxy` matrix is ever assembled. The state-vector-sized objects — the
Davidson subspace block `(b, N)` and `hii` — are partitioned here.

## 3. Partition strategy

### 3.1 Global index and contiguous blocks

Determinant indices follow the existing contract:

- TPB: `k = ia * M_b + ib`, `ia` in `[0, M_a)`, `ib` in `[0, M_b)`;
- GDB: `k` is the pair index.

The vector is split into contiguous balanced blocks. For `N` determinants and
`P` ranks,

```
s_r = r * floor(N/P) + min(r, N mod P),     block_r = [s_r, s_{r+1}),
```

the first `N mod P` ranks owning one extra determinant. Every rank computes the
same bounds from `(N, P)` alone, so agreeing on the layout costs no
communication. `partition_vector` and `owner_of_index`
(`torch_impl/distributed_davidson.py`) implement the block and its exact inverse.

Contiguous blocks are the right default because:

- ownership is arithmetic (`k -> rank`), so no distributed hash table exists;
- the basis strings themselves stay replicated (they are `M_a`/`M_b` entries,
  tiny next to `N` determinants), and only the *coefficient* vector moves;
- for GDB the determinant index is the pair index, so a block is a plain slice
  of the pair arrays;
- for TPB a block is a run of rows of the `(M_a, M_b)` coefficient matrix
  (aligning block boundaries to multiples of `M_b` keeps rows whole). β-only
  and αβ-double excitations stay inside a full-row block for their bra rows;
  α-only excitations can land in another block, which is the halo.

### 3.2 TPB alternative: 2-D `P_a x P_b` blocks

A row block (`M_a/P x M_b`) makes β-only terms fully local but α-only terms
remote; a column block is symmetric. SBD's own MPI decomposition exposes the
same axes (its `adet_comm_size`, `bdet_comm_size` and `task_comm_size` options),
i.e. a 2-D string-grid plus a split by task type
(α-only / β-only / αβ-doubles). A 2-D grid `P = P_a * P_b` makes α-only
exchanges row-halo-shaped and β-only exchanges column-halo-shaped. Both
layouts ship: the 1-D block is the default, and
`distributed={"tile": "auto" | [P_a, P_b]}` selects the 2-D tile grid
(Section 4.4).

### 3.3 Load balance

Equal *counts* do not equalize *work*: the number of singles/doubles differs per
string (occupied/virtual counts). The owner-computes path therefore cuts the
α rows on a **cost prefix**: `alpha_row_weights` computes, from the strings
alone, `w(ia) = M_b·(n_αs + n_αd)(ia) + n_αs(ia)·Σ_ib n_βs(ib) +
Σ_ib(n_βs + n_βd)(ib)` with `n_s = o·v` and `n_d = C(o,2)·C(v,2)`;
`partition_rows_weighted` splits the prefix into `P` contiguous segments of
equal cost, capped at `1.25 ×` the mean row count per rank so a rank's memory
stays bounded. The resulting boundaries are replicated, so `owner_of_index` and
the table split stay purely local (`RowPartition.owner_rows` is a
`searchsorted`). The 2-D tile cut weights both axes: `alpha_row_weights` for
the α slabs and its spin-swapped mirror `beta_col_weights` for the β slabs, so
a skew in either string list moves that axis' cut rather than leaving an even
uid split. The spinless tile cuts its two half-string axes on the per-uid
determinant counts, because a selected basis concentrates many determinants on
a few uids. The equal-count `partition_rows` remains the default outside the
owner-computes path.

## 4. Distributed matvec

### 4.1 Current implementation: owner-computes TPB, staged row exchange

The default TPB matvec is `LocalTPBOperator` (`distributed_matvec.py`). Rank
`r` owns a contiguous α-row block `[r0, r1)` (cost-balanced, Section 3.3) and
all β columns, so its output is the contiguous flat range
`[r0*M_b, r1*M_b)`. Per subspace it builds:

- **α tables** for its own bra rows only (`generate_local_torch_helpers`),
  split once by **ket-owner stage**: stage 0 is the local block, stage `s`
  holds the entries whose ket row is owned by rank `(r+s) % P`, with the ket
  index remapped to its position in that stage's receive buffer. Each stage is
  a valid local CSR, so the existing batched kernels apply unchanged through
  their `T_full=` keyword;
- **β tables**: full, because every rank owns all β columns. Their
  `T`-independent setup (integrals × sign, parity masks, occupancy matrix) is
  cached once per subspace (`build_beta_pass_cache`);
- the diagonal `hii` block.

`__call__` starts the exchange, runs the β pass on the local rows while stage
1 is in flight, then for each stage runs α-singles, α-doubles and αβ-doubles
against that stage's buffer. `StagedRowExchange` posts stage `s+1`'s send/recv
before stage `s` is consumed (two alternating buffers), and each rank sends
only the rows the peer's tables actually reference. The old full-length
`x_full` halo scratch is gone: per-rank matvec memory is the two stage buffers
(`≤ 2·N/P`) plus `x`, `y` and the send gather.

The αβ-swap projector is local too (`LocalAlphaBetaProjector`):
`P(v)[i,j] = (v[i,j] + τ[i,j]·v[j,i])/2` needs the columns `[r0, r1)` of every
peer's block, which the same cyclic staged pattern ships as a transposed piece.
So the default `apply_ab_projection=True` symmetric basis runs owner-computes
instead of the replicated fallback.

### 4.2 Kernel selection and matvec limits

- On a CUDA rank the operator uses the tiled Triton kernels (singles, same-spin
  doubles and αβ doubles; the diagonal stays batched) whenever the solver's
  resolved `matvec_impl` is `"triton"` — which
  `"auto"` selects on CUDA when Triton imports. An explicit
  `matvec_impl="batched"` stays batched; gloo/CPU ranks are always batched.
  A launcher-initialized rank re-resolves `"auto"` against the device it
  actually runs on before the solve, so a solver constructed on the host CPU
  still selects Triton on its NCCL ranks; an explicit `matvec_impl="triton"`
  on the CPU-only gloo backend raises instead of being silently rewritten.
  The Triton launchers take explicit offset/to pointers, so they read the
  stage buffer directly. Same-spin `eij` and the β-pass setup are cached once
  per subspace instead of recomputed on every matvec (this also speeds up the
  single-GPU Triton path).
- GDB has a Triton kernel for the block-mode same-spin doubles slices
  (`mult_pair_blocks_triton.py`): an explicit `matvec_impl="triton"` routes
  those slices to it and keeps the batched kernels for the singles, the
  diagonal and the αβ channel, so the two backends mix per channel. `"auto"`
  stays on the fully batched pair path. The kernel reads the ERI through
  ordinary indexing at launch time, so it accepts the compact store. The
  entry-parallel kernel (`mult_entries_triton.py`) has one lane per streamed
  entry and fuses the excitation element with the gather-scale-scatter. The
  streamed (non-block) GDB path routes both its singles and doubles tables to
  it; the spinless distributed operators route the doubles tables (the
  singles stay batched there). It reads a flat dense ERI, so a compact store,
  a non-CUDA tensor, a multi-limb string layout or a block input makes it
  decline and the batched scatter runs instead — the result never depends on
  the flag.
  Complex dtypes are supported on
  the TPB, GDB-block and entry-parallel Triton kernels through the same
  interleaved-real path; `"auto"` on CUDA resolves to `"triton"` for TPB even
  with complex integrals, and in GDB mode an explicit `"triton"` accepts
  complex while `"auto"` stays batched. The TPB αβ kernel forms its ERI element
  creation-index-first, which for a complex Hermitian ERI is the conjugate of
  the batched order (real ERIs are symmetric under the swap), so the complex
  branch negates the imaginary part of the difference; the entry-parallel
  kernel keeps the batched order and needs no correction.
- The 2-D `P_a × P_b` string grid (Section 4.5) shrinks the replicated β
  share and the exchange volumes; Triton runs there through the `T_stride`
  argument.
- `spin_sq`, `compute_full_rdm` and `compute_spin_squared` run on a tile grid
  from the tiled post-processing (Section 4.5): both the TPB and the GDB pair
  layouts compose `H + λ(S^2 − target)` on the owner-computes operator, and a
  declined auto grid keeps all three on the implicit `(P, 1)` partition (the
  TPB penalty also does so when no grid is configured). An explicit GDB pair
  list with no tile grid applies the penalty on its 1-D owner-computes block,
  sharing the matvec's all-gather: the spin-swap rows are re-derived in
  `chunk_size` slices, so only an O(`N/P`) diagonal and sign block are cached
  and no O(K) swap table is built. The `<S^2>` diagnostic uses the same sweep
  there; full RDMs still require the tiled path (a structural GDB basis keeps
  its tile for the diagnostic and penalty too). A solve with no tile configured
  otherwise, or a basis that fits in one Davidson block, keeps the replicated
  fallback (Section 5.5).

### 4.3 Excitation-table locality and determinant lookup

- `generate_all_torch_helpers` now diffs bra rows in chunks of `chunk_rows`
  (default 1024) against all ket strings, so the `(M, M)` Hamming-distance
  matrix is never materialized; the peak host temporary is `O(chunk_rows·M)`.
  The output is bit-identical to the old builder for every chunk size.
- `generate_local_torch_helpers` builds only the owned α rows' tables (a
  `(chunk, M_a)` diff), with local CSR offsets; the β tables stay full. The
  worker also builds `hii` and the σ matrix for its rows only, so setup no
  longer allocates any length-`N` tensor.
- Destination string indices in the tables are **global**, so no lookup is
  needed at matvec time: `k = ia * M_b + ib` (or the pair index), and
  `RowPartition.owner_rows` gives the rank. Bitmask → index lookups (sampled
  configurations, warm-start strings) use `searchsorted` on the globally
  sorted, replicated string arrays. The only requirement is that the global
  basis ordering is fixed *before* partitioning: the
  `seed_strategy="connected"` reorder and GDB dedup must run identically on
  every rank.

### 4.4 GDB and spinless owner-computes; complex coefficients

**GDB (`LocalPairOperator`, `distributed_pairs.py`).** The flat pair list is
split into contiguous rank-ordered blocks (`partition_vector`, or the
cost-weighted split; the input gather reads each rank's block size once).
Rank `r` builds
`make_diag_terms_pair_local` for its own pairs, and `mult_pair_local_torch`
streams the five pair tables restricted to the owned bra range
(`PairHelpers(..., pair_range=...)`): the αβ-double chunk loop is clamped to the
owned pair range, and the same-spin dedup expansion keeps only in-range
partner entries (a per-block binary search over the sorted partner slices).
Each matvec all-gathers the local input into `T_full` — the same length-`N`
vector the replicated fallback holds — and computes only the owned output
block. The gather is a `FlatGather`: the block sizes are exchanged once per
operator, and each matvec posts one asynchronous `all_gather_into_tensor`
while the diagonal and, on the streamed tables, the first excitation chunk
(which never reads the vector) are computed; with even blocks the gathered
buffer is the full vector, with no compaction copy. The block GDB path waits
right after the diagonal: its move tables are cached across matvecs, so its
remaining work reads the kets. The α↔β-swap projector (`LocalPairSwapProjector`) needs the full vector
too, so it all-gathers and returns its slice.

**GDB block fast path.** When the pair basis is a disjoint union of Cartesian
blocks — the benchmark's tier construction, and any full product —
`build_pair_blocks` groups the pairs by identical per-spin partner sets and
verifies the decomposition exactly: candidate groups are fingerprinted, then
every multi-member bucket is compared row by row, and the two `pair_idx` maps
must tile each group in ascending order. The block matvec then enumerates α/β
single moves between group grids and cross-lifts them by block index
arithmetic, instead of streaming pair-keyed tables and testing every row with
a `searchsorted`. `mult_pair_blocks.py` implements the αβ-double cross-lift
and the four same-spin channels, in full and owner-computes form. A full
matvec enumerates forward from the ket groups; the owner-computes form instead
enumerates the inverse moves from each rank's owned bra slab (α rows and β
columns) into the ket group, so the candidate grid — not just the product
kernels — is sized by the owned rows and the enumeration scales as `1/P`. A
same-spin pass pays one candidate grid per owned row or column no matter how
many pairs that row holds, so the owner-computes pair vector is split on a
candidate-cost prefix sum (`pair_work_weights` → `partition_vector_weighted`),
not by pair count. The same per-pair estimate adds the αβ channel's lifted
entries — one per crossed inverse α/β move, counted from each string's single
moves into the ket group's uid sets — converted to candidate units, because
that channel scales with the owned pair count instead. The bounds stay
contiguous, so the gather, memmap hand-back
and warm-start slices are unchanged. The result is exact — the block and
streamed matvecs agree to summation order — and the
streamed tables remain the fallback whenever a structural check or a
size/group gate fails. `gdb_kwargs={"block_matvec": "auto" | "on" | "off"}`
selects the policy (`"auto"` is the solver default; `"on"` skips the size gates
for tests and A/B runs), and `block_table_bytes` caps the decomposition's byte
footprint (which also gates the build). Move enumeration and lifted tiles are
byte-budgeted by `budget_bytes`. `tile_move_cache_bytes` overrides the tiled
operator's move-table cache budget (see below).

**Structural pair input (`gdb_kwargs={"structural_blocks": [...]}`).** An
explicit pair list costs `O(N)` memory per rank (`~80 B/pair` in `PairBasis`),
which caps GDB near 1e9 pairs on a large node. A structural basis names whole
tier blocks instead — a per-spin tier range whose Cartesian product is the
selected space — and `StructuralPairBasis` derives `n_pairs`, the uid axes and
each block's `pair_base` arithmetically, so no length-`N` array or partners CSR
is ever materialised. A pair index is resolved by locating its block and doing
the grid arithmetic. The block matvec, the pair tiles and the pair hand-back
all read the block structure, and the pair 1-/2-RDM builders walk the blocks
too. The solver's structural post-processing (full RDMs, the ⟨S²⟩
diagnostic and the `spin_sq` penalty) runs on the tiled pair path, which
reads every ket from the staged exchange; a declined auto grid uses the
implicit `(P, 1)` uid partition. The streamed table fallback still needs
explicit per-pair arrays and rejects the structural basis. A warm start on a
structural basis only
checks that the cached pairs all lie in the caller's block spec — a block
spec cannot be extended pair by pair — and embeds the cached amplitudes by
looking each owned pair up after resolving it through the block geometry; a
cold-with-probe start skips the HF splice, which needs an explicit pair
list. `init_vectors` are indexed by structural pair, the order of the
amplitudes the structural solve returns, so each rank gathers its owned
entries directly (its contiguous range, or its scattered tile).

**GDB tile (`LocalPairTileOperator`, `distributed_pairs_tile.py`).** The pair
basis's α/β uid axes take the same 2-D grid as TPB: a rank owns the pairs whose
uids fall in its rectangle and, per source stage, receives the ket-group
rectangles its fragments read. Because the slab kernels read dense group
rectangles, the union a rank requests from one peer is that peer's whole tile,
so a stage ships whole peer tiles over P2P (never a full-vector all-gather); the
setup broadcasts each rank's per-stage index lists once as int64 byte images,
keeping the transient setup memory at the tile bytes rather than one Python
object per pair. The stage lists come from the needed-set plan by default: each
rank derives, from the replicated decomposition and its own tile bounds, the
ket-group rectangles its fragments read and stages only those.
`gdb_kwargs={"needed_staging": ...}` selects the policy — `"off"` stages each
peer's whole tile, `"on"` forces the needed-set plan, and `"auto"` (the
default) keeps the whole tile when the needed plan would stage most of it
anyway. Both plans are pure functions of the basis and grid, so every rank
derives the same lists without extra communication, and they produce the same
arithmetic. `pair_tile_partition` cuts the axes by
**pair count** (`pair_axis_weights`), not by uid count: on tier bases the even
uid cut leaves one slab with most of the pairs, measured as a 2× straggler on a
two-rank grid, while the weighted cut evens the pair count to 1.0× in the test
fixture. The slab kernels enumerate the inverse moves from each destination
fragment into the whole source group and then mask that table to the stage's
rectangle. Those tables depend only on the basis, so the operator memoises them
in two byte-bounded `_MoveCache` instances — 60% same-spin, 40% αβ, so the
measured dominant same-spin channel cannot be crowded out — with
`gdb_kwargs["tile_move_cache_bytes"]` overriding the total (default:
`string_cache_bytes`; 0 disables). A fragment is enumerated once while its
table fits and reused across stages and Davidson matvecs; a table larger than
its cache share is recomputed per stage instead, and one oversized table does
not latch an empty cache. The per-channel profile
(`SCIS_PROFILE_BLOCKS=1`) prints both caches' occupancy plus the same-spin/αβ
split. The α↔β projector and the pair hand-back are index-based, so no rank
materialises the full pair vector. `tile="auto"` is a policy here too
(`resolve_auto_pair_tiling`): the 1-D block matvec is the faster layout at
equal size, so a grid is chosen only when its per-matvec full-vector gather
plus the block vectors `N·s + (b + 9)·N/P·s`, together with the replicated
pair basis and helper arrays (`pair_replicated_bytes`: nine length-`N` int64
tables plus the uid/offset arrays), would exceed half the device budget. The
replicated arrays are present under either layout, but they still occupy the
device when the fit is judged. On CPU runs and single ranks the 1-D block
matvec is kept. The policy runs unchanged when a full RDM, `<S^2>` or the
`spin_sq` penalty is requested: a chosen grid keeps that post-processing on
the tiled path, and a declined grid uses a `(P, 1)` uid partition — the pair
analogue of the 1-D row split — instead of the replicated full-vector path.

**Spinless (`LocalSpinlessOperator`, `distributed_spinless.py`).** The
determinant list is split the same way; `make_diag_terms_spinless_local` builds
the owned diagonal block, and the operator calls the existing
`_stream_spinless` enumerator with `src_dets` set to the owned determinant
slice and `target_dets` the full basis, so the Slater–Condon work and the
output scatter scale as `1/P`. There is no projector (no α/β split), but the
finish path builds occupancies as local diagonal sums and, with
`compute_full_rdm`, both RDMs from the owned amplitude blocks through
`spinless_rdm12_distributed` (`distributed_spinless_rdm.py`): the
connected-set axis is reduced in fixed chunks and each rank forms only its
`nmo^4 / P` output slice, so neither the full amplitude vector nor the full
Gram matrix is materialised.

Warm starting (`probe_method="warm_start"` with a `TorchSpinlessSCIState`)
works on both owner-computes layouts. A rank embeds only the determinants it
owns — its contiguous block or its scattered tile — by limb-aware string
lookup against the cached state, and the surviving norm is one all-reduce
(`_global_norm_local`) before the cold-seed fallback decision, so every rank
takes the same branch. No rank materializes a full extended seed, and the
union basis is built deterministically from the caller's strings on every
rank, so the operator ordering matches a cold solve of the same union. The
tile layout owns a scattered, sorted subset of the flat vector, so its
embedding is a gather/scatter over `bounds.determinant_index` rather than a
contiguous paste.

`tile` works on this layout too (`spinless_tiles.py`,
`distributed_spinless_tile.py`): splitting every determinant's bitmask at half
the spin-orbitals gives two half strings, and the sorted unique half strings
turn the explicit basis into a sparse `n_row_uids × n_col_uids` grid that
`partition_tiles` cuts like the TPB `α × β` grid. Both uid axes are weighted by
the per-uid determinant counts rather than the uid counts: a selected basis
concentrates many determinants on a few uids, and the auto-tiling model assumes
a balanced `N/P` per rank. A rank owns the scattered,
sorted determinants inside its rectangle; `LocalSpinlessTileOperator` requests
only the remote ket amplitudes its owned rows read, one ket-owner peer at a
time, and accumulates each stage straight into the owned output, so it never
forms a length-`N` input. The exchanged volume is not reduced (the needed-ket
union measures 0.93–1.00 `N`), but the per-stage receive buffer is `0.50 N` at
two ranks, `0.27 N` on a `2 × 2` grid, `0.13 N` on a `2 × 4` and `0.07 N` on a
`4 × 4`, against `0.91–1.00 N` for a single-buffer compressed gather of the same
set; the price is a ket-side (transpose) enumeration: each matvec enumerates
`local_size + needed_size` source determinants instead of the 1-D operator's
`local_size`, measured at 2.0× on two ranks and 8.7× on a `2 × 4` for
full-Cartesian-product bases. `resolve_auto_spinless_tiling`
keeps the faster 1-D gather unless the 1-D peak `N·s + (b + 2)·N/P·s` exceeds
half the device budget, and a single rank or a CPU run (no budget) stays 1-D.
`spin_sq` and `compute_spin_squared` are refused on the spinless path as a
physics limitation: `S²` is defined from the α/β partition, which a flat
spin-orbital basis does not have. The reason is reported by
`capabilities()["modes"]["spinless"]` and by `preflight()`, not only as a
call-time error. `compute_full_rdm` runs on the owner-computes solve: the RDM
builder consumes the owned amplitude blocks, and only rank 0 receives (and
packs) the full tensors.

**Complex coefficients.** The partition and reductions are index-based, so a
complex Hamiltonian needs no new layout: the batched kernels are dtype-generic,
the TPB stage buffers, `DistributedVector.to_global` and the CPU gather reduce
complex tensors as an explicit `(real, imag)` pair (portable across gloo and
NCCL builds), the αβ projector casts its real τ to the vector dtype, and the
local Davidson uses Hermitian projections/CGS2 (`conj`, `|x|²`, `vdot`). The
replicated `davidson_distributed` fallback carries complex coefficients too:
the Hamiltonian diagonal stays real, the projection is complex Hermitian and
the `C`/`HC` blocks stay partitioned as `DistributedVector`s, so complex calls
that fall back (small basis, full RDMs) keep the same memory profile as real
ones. Only the missed-root search is unavailable on that fallback.
`davidson_dtype=torch.complex64` is the narrow complex subspace.

### 4.5 2-D tile grid (TPB)

`distributed={"tile": "auto" | "force" | [P_a, P_b]}` replaces the 1-D row
block with a `P_a × P_b` grid (`TileGrid` / `TilePartition`,
`partition_tiles`). Rank `i*P_b + j` owns the rectangle `A_i × B_j` of the
`(M_a, M_b)` matrix, stored row-major with local width `local_cols`.

How `"auto"` chooses (`resolve_auto_tile_grid`): the grid itself minimises the
larger slab (`resolve_tile_grid`), but tiling is only selected when it pays.
Tiling buys memory, not speed — it removes the replicated α/β tables at the
cost of the exchanges (about a tenth of the runtime on one node). The policy
therefore models the 1-D footprint as the worst-case table bound
`3·8·(⌈M_a/P⌉·c_a + 2·M_b·c_b)` (`c_s` = candidate singles + doubles of one
spin string; the α tables are row-restricted in the 1-D split) plus the
Davidson block `(b + 9)·N/P` elements, and keeps the 1-D split whenever that
sum is at most half the usable device memory; above it the auto grid is used.
The usable budget is estimated from the card's *free* pool when the decision
runs — the replicated ERIs, basis and helper arrays are already allocated —
with 10% headroom off both the free and the total bytes, so a card holding
more than that in live tensors is not over-estimated. The table bound
overestimates the measured survivors about
twofold, which biases the decision towards tiling. A grid with a single
column slab is never chosen: it keeps the full β tables and the same α rows as
the 1-D split, so it only adds exchange buffers.
`"force"` (and `tile=True`) always tiles with the auto grid; an explicit
`[P_a, P_b]` always tiles with that grid. On CPU runs, where no device budget
is known, `"auto"` keeps the 1-D split (as the pair and spinless policies do).
The budget is the minimum across the ranks, so every rank makes the same
decision on a heterogeneous device mix. A declined auto grid does not send a
`spin_sq` penalty, `compute_full_rdm` or `compute_spin_squared` back to the
replicated solve: those keep the tiled builders on the implicit `(P, 1)`
partition, which is the 1-D row split.

Each rank keeps only its tile's α and β tables
(`generate_local_torch_helpers(..., col_start, col_stop)`), so the replicated
β share — the term that dominates the 1-D layout — shrinks with `P_b`.

`LocalTPBOperator` runs three staged point-to-point streams, all over the world
group with explicit peer lists: the **α stream** ships the source rows the
peer's α tables reference, restricted to the receiver's own column slab; the
**β stream** does the mirror image along columns; the **αβ corner stream** ships
only rectangles with both slabs remote (the `(0, s_b)` and `(s_a, 0)` corners
ride the α and β buffers). Needed-index lists are exchanged once per subspace,
stage 0 is the local tile, and stage `s+1` is posted before stage `s` is
consumed, so the transfer overlaps the kernels. Tables keep global ket labels
until the per-stage split remaps them to buffer positions.

The batched kernels take the stage buffer's row width through `T_stride` (the
β and αβ passes read buffers narrower than the owned output). Triton runs on
the tile grid too: the same two launchers accept `T_stride`, and the kernels
promote the ket payloads to int64 before forming the buffer offset so a large
local tile cannot wrap. The αβ-swap projector has a tile variant
(`LocalTileAlphaBetaProjector`): the needed block `B_j × A_i` meets each peer
tile in a contiguous rectangle whose sender and receiver local slices coincide,
so each rank ships exactly one local tile in total; a square grid with equal
cuts reduces to one pairwise send and a diagonal tile ships nothing. The cold
start seeds the HF entry when the tile owns it and hashes the global index for
the noise; the hand-back gathers per-tile rectangles into `(n_roots, M_a, M_b)`
or writes `handle[r0:r1, c0:c1]` rectangles of a 2-D memmap.

**Tiled post-processing (`tiled_postprocess.py`, `pair_tiled_postprocess.py`).**
`tiled_one_rdm`, `tiled_two_rdm`, `tiled_spin_squared` and the fused
`tiled_rdm1_spin_sq` compute the spin-resolved 1- and 2-RDM and `<S^2>` from a
tile solution over
the operator's existing staged exchanges: the diagonal occupations and the
class-A term are tile-local, the off-diagonals read one slab buffer, and the
`<S^2>` cross term joins only matched class-D pairs on the corner rectangles
(including banded sub-stages). The default builders accumulate replicated
tensors (`(L, L)` for the 1-RDM, `(3, L, L, L, L)` for the 2-RDM) and one
packed SUM all-reduce returns the global result; the full vector is never
materialised. The output-partitioned entry points
(`tiled_one_rdm_distributed`, `tiled_two_rdm_distributed` and their
`pair_tiled_*` mirrors in `distributed_rdm.py`) instead split the flattened
output across the ranks, process one owner group per pass, and keep only each
rank's slice plus one group buffer: peak per-rank output memory is
`O(L^4 / P)` rather than `O(L^4)`. The solver's finish path uses those entries
whenever the layout can partition and the full output exceeds the per-rank
replication bound — `min(memory_budget_bytes, distributed budget_bytes)`, or
the module's 256 MiB default when no cap is set — and then calls
`PartitionedRDM.gather_to_root`, so only rank 0 holds the full tensors for the
packed result. Below the bound (and on the non-tile fallback) the replicated
builders above run unchanged, bit for bit. `stats["rdm_accumulation"]` and
`preflight()["rdm_accumulation"]` report which path a call takes.
`tiled_s2_apply` applies the `S^2` operator on the tile block
through the same staged exchanges and `tiled_s2_diagonal` supplies its
(preconditioner) diagonal, so `spin_sq` composes `H + λ(S^2 − target)` on the
owner-computes path. On a tile grid the solver uses `tiled_spin_squared`, the
`tiled_s2_apply` penalty and, with `compute_full_rdm`, the RDM builders above,
so those knobs no longer force the replicated matvec there. The entry points
require a tile-partition operator and a collective call order; without a tile
grid the 1-D builders are used. For the
pair layout, `pair_tiled_one_rdm`, `pair_tiled_two_rdm` and
`pair_tiled_spin_squared` mirror `rdm_pairs.py` and `spin_sq_pairs.py` on a GDB
pair tile: only the owned bra rows are enumerated and each ket is read from the
stage that owns it, with one packed all-reduce. `pair_tiled_s2_apply` and
`pair_tiled_s2_diagonal` do the same for the pair `spin_sq` penalty, which the
solver composes as `H + λ(S^2 − target)` on the pair tile. An explicit pair
list with no tile grid uses `PairLocalSpinSquared` instead: it applies the same
diagonal and spin-swap sweep to the owned 1-D block, reading every source
amplitude from the matvec's all-gathered vector in `chunk_size` slices, so it
builds no O(K) swap table and allocates no length-`N` temporary — only the
O(`N/P`) diagonal and sign block are cached. It serves both the penalty and the
`<S^2>` diagnostic; its expectation is one `vdot` and one SUM all-reduce per
root. A structural
(block-described) basis is walked by the same builders — the unique-string
axes and the block layout replace the pair list — but only on an explicit
tile grid, because each ket comes from the operator's staged exchange.

### 4.6 Strings wider than 64 bits

A spin string with `n_orb > 64` is carried as a `(M, n_limbs)` int64 limb matrix
(limb `k` holds bits `[64k, 64k+64)`), which is what raises the ceiling from
128 to `4·n_orb` qubits. The builders accept Python ints, 1-D bit patterns or
pre-split matrices (`_as_string_limbs`), and `det_from_alpha_beta_batch`
interleaves limb halves.

The matvec kernels read strings through four private primitives —
`_string_range_popcount`, `_string_bit`, `_string_occ_matrix` and
`_interleaved_range_parity` — that clamp the global bit range or position to
each 64-bit limb and sum. The single-word branch keeps the original
`(1 << hi) - (1 << lo)` expressions, so the ≤64-orbital path is
bit-identical. The Triton kernels take `N_LIMBS: tl.constexpr`, load each limb
per lane, and iterate with `tl.static_range`; the `N_LIMBS == 1` branch
compiles to the original code. In the distributed operator the 2-D string
slices are owned-row slices of the limb matrix, `alpha_row_weights` sums
popcounts over limbs, and `_inter_to_block_signs` (projector τ, spin bridge)
and `make_one_rdm_diag_torch` have limb paths.

The adapter converts `ci_strings` to limb matrices when `norb > 64`
(`_ci_string_limbs`), including dedup/sort, validation, the connected-seed
reorder and the SCIState integer strings. The GDB pair matvec, the pair
diagonal and the explicit-basis block channels take limbs too: the pair
helpers carry `(M, n_limbs)` strings, order them lexicographically with
limb 0 most significant, and mask per limb. A wide explicit pair solve is
supported end to end, including the tiled RDMs and the ⟨S²⟩ diagnostic
(the tile partition, tile operator and block-walked post-processing all have
limb branches). Warm/probe states and `init_vectors` work on a wide pair basis
too: every pair string is normalized to its limb rows and keyed by the basis'
`(α_uid, β_uid)` pair key, so the superset union, the cached-eigenvector
embedding (serial and the owner-computes per-rank variant) and the
`init_vectors` row mapping resolve a multi-limb pair exactly like a
single-word one. The builders
themselves take limb strings on the explicit-pair path and are checked
against an arbitrary-precision Python-int reference at 70 orbitals and
cross-checked with the dense builder output at 130. The structural
(block-described) basis takes limb strings too: its unique axes are
`(M, n_limbs)` matrices, and the block matvec, the tiled operator, the
block-walked RDM/⟨S²⟩ accumulators and the α↔β-swap projector (a lexicographic
row lookup over the sorted unique axes) all have limb branches, checked against
the explicit limb basis at 70 and 130 orbitals. The dense `(2L)^4` ERI is the
first memory wall beyond this range (Section 6).

## 5. Davidson subspace

`davidson_distributed_local` (`distributed_davidson.py`) is the owner-computes
block Davidson. `C` and `HC` are preallocated `(block_size, N_local)` blocks in
the subspace dtype, so per-rank memory is `2·b·N/P`, not `b·N`. It has the same
single-/multi-root return shapes as `davidson_torch`, but the returned vectors
are **local blocks**; the solver's post-processing consumes them directly.
The replicated `davidson_distributed` remains for the real full-vector
fallback: GDB/spinless bases that fit in one Davidson block (its dense escape)
and GDB calls that need a full RDM without a tile grid. The TPB and
explicit-pair GDB penalties and the explicit-pair GDB `<S^2>` diagnostic run on
the owner-computes path. GDB, spinless and complex owner-computes solves use
the same `davidson_distributed_local` core; the flat layouts pass a
`partition_vector` block with `row_size = 1`.

### 5.1 Projection

```
H_proj[i, j] = <C_i | H C_j> = sum_r sum_{k in r} conj(C_i[k]) (H C_j)[k].
```

Each rank forms the local partial `C[:d] @ HC[d]` for a newly appended column
and **one `all_reduce(SUM)`** of the `(d+1,)` vector updates the replicated
`H_proj`. A thick restart rebuilds the full `(d, d)` projection with one fused
all-reduce. All ranks therefore see identical Ritz problems.

### 5.2 Eigendecomposition, Ritz vectors, residuals

`H_proj` is `d x d` (typically 10–30), replicated. `eigh` always runs in fp64
(a narrow subspace's small matrix is cast up), and rank 0 broadcasts the
eigenvalues and coefficients so every rank forms identical Ritz vectors
(cuSOLVER/LAPACK sign choices otherwise differ). Ritz vectors and products are
one GEMM each, `U[:, :k].T @ C[:d]` and `U[:, :k].T @ HC[:d]`, followed by a
global renormalization. Residual norms are reduced for all roots in one
`all_reduce`; the convergence decision is a pure function of those replicated
values. Multi-root is supported: the unconverged root with the largest residual
drives the next correction, and restart keeps the `n_roots + 1` Ritz pairs plus
the new correction (thick restart).

### 5.3 Orthogonalization and control flow

The correction is orthogonalized with two-pass block classical Gram–Schmidt
(CGS2): `coef = C[:d].conj() @ t`, one all-reduce, `t -= C[:d].T @ coef`, twice.
Norms are all-reduced, so the breakdown test is identical on every rank.

**Invariant:** every rank must execute collectives in the same order and the
same number of times. All reductions are `all_reduce` (not `reduce` +
broadcast), so they return identical values on every rank; any branch must be a
pure function of those values (never of rank-local state). Rank-local early
exit is forbidden. `project_fn` is a collective local closure (it performs P2P
inside), and Davidson calls it in lockstep on seeds, products and corrections.

### 5.4 Warm start and fp32

`init_vectors`: each rank maps only its owned rows (`ia[r0:r1]`), applies the
local projector, and normalizes by the **global** norm, so the cold-start
fallback decision is identical everywhere. `probe_method="warm_start"` embeds
the cached interleaved-Wick eigvec with
`_embed_eigvec_into_extended_basis_rows`, and on a tile grid with
`_embed_eigvec_into_extended_basis_tile` for the owned `(rows, cols)`
rectangle: the gather happens on the host and only the owned block moves to
the device, so a warm solve never materializes the cached length-`N` vector on
a GPU. The cached state itself is still replicated on every rank. Cached strings absent from
the extension are zero rows; the global norm decides whether anything survived.
The multi-root seed block mirrors `_make_seed_block`: the caller's starts first,
then one-hot vectors at the globally smallest `hii` (`global_smallest_hii`,
local top-k + one `all_gather_object`), each perturbed by partition-invariant
`hash_uniform` noise and deflated with all-reduced inner products. Multi-root
warm starts run the missed-root search: the roots are orthonormalized through a
replicated Gram matrix + Cholesky, the search deflates with
`v - Q^T(Qv)`, and up to four attempts re-seed as in `_missed_root`.

With a narrow subspace (`davidson_dtype=torch.float32` over fp64, or
`torch.complex64` over complex128), `C`/`HC` halve; the operator still applies
`H` in full precision (`hii_local` promotes the input and the kernels upcast on
read), with the shifted `y - s·v` computed before the cast into the subspace.
The returned vectors are upcast to the full dtype and globally renormalized;
the energy is one extra distributed Rayleigh-quotient matvec per root
(`"rq64"`) or the shifted Ritz value (`"ritz"`), selected by
`davidson_fp32_energy`/`"auto"` exactly as on the single-GPU path. The
replicated fallback implements the same shift/rq64/ritz protocol, using the
full `hii` every rank already holds.

### 5.5 Post-processing and result hand-back

Occupancies are local weighted sums: `make_one_rdm_diag_torch` on the owned
TPB rows, `make_one_rdm_diag_pair_torch` on the owned pairs and
`spinless_rdm1_diag_torch` on the owned determinants, each followed by one
all-reduce of the length-`L` vector(s). The TPB σ-bridge to the spin-block
convention uses the local σ block (GDB amplitudes and spinless occupancies are
already in their returned convention). Rank 0 alone builds the
`SCIState`/`SCIResult` objects; non-zero ranks return an empty `EigenStates`.

The returned amplitudes (`(M_a, M_b)` per root) and the warm-start
`eigvec_inter` snapshot depend on size (`DISTRIBUTED_HAND_BACK_SMALL_BYTES`,
1 GiB):

- **Small**: one `dist.gather` of the σ-signed local blocks to rank 0's host
  buffer, then today's in-memory `SCIState`. Rank 0's result is `torch.save`d
  through the session's temp file as before.
- **Large**: rank 0 creates one `.npy` per root under
  `tempfile.gettempdir()` (honors `TMPDIR`) and broadcasts the directory. Every
  rank writes its rows through `numpy.lib.format.open_memmap`, a barrier
  follows, and rank 0 returns a small marker: `MemmapAmplitudes` in the
  GDB/spinless state objects and `TorchSCIState` for the warm vector, and a
  `DeferredSCIState` for TPB (qiskit's `SCIState.__post_init__` validates the
  amplitude shape, so it cannot hold a marker). `torch.save` pickles only the
  markers; the parent process (session/spawn) or rank 0 (torchrun) opens them
  with `np.load(mmap_mode="r")`, reshapes the TPB amplitudes to `(M_a, M_b)`
  and builds the real `SCIState`, then unlinks the files — the POSIX mapping
  stays valid. Disk space is checked first and the error names `TMPDIR`.
  A rank's temporary directory is node-local, so a **multi-node** run sets
  `distributed={"hand_back_dir": ...}` to a directory every node can see (the
  benchmark uses `<out-dir>/hand_back` on the shared filesystem); otherwise
  the other ranks cannot open rank 0's files. Single-node runs keep the
  faster local `TMPDIR`. A parallel POSIX filesystem is the safe choice for
  that directory. Object-store-backed mounts need the creation fallback in
  `_create_result_memmap`: a large new file's length is not visible to the
  next `fstat` until a byte lands at its end, so the writer extends the file
  itself before mapping it. GDB pair tiles and spinless tiles own a scattered,
  sorted subset of the flat vector, so their large hand-back uses
  `write_pair_tile_memmaps`: one flat file per block, each rank writing its own
  disjoint `pair_index`/`determinant_index` positions.

`spin_sq`, `compute_spin_squared` and `compute_full_rdm` run on the tiled local
path when a tile grid is used (Section 4.5) — including the penalty on the GDB
pair layout — with no full state-vector gather; an explicit GDB pair list with
no tile grid applies the penalty and the `<S^2>` diagnostic on its 1-D block
under the same rule, and the remaining cases keep the replicated fallback. With
`compute_full_rdm` the RDM output itself is accumulated partitioned
(Section 4.5) or replicated, never the state vector.

### 5.6 Checkpoint and restart

A long solve can persist its Davidson state and continue after a job failure.
Checkpointing is opt-in through `distributed={"checkpoint_dir": ...,
"checkpoint_every": ..., "checkpoint_resume": ...}`. When `checkpoint_dir` is
`None`, none of this code runs: the loops take their original path and no file
is touched.

**What is persisted.** Every CI subspace gets its own directory
`checkpoint_dir/subspace_<k>/`:

- `rank_<r>.pt` — one shard per rank, written by that rank, holding only its
  local state: the local Davidson vectors `C` and products `HC` (the
  owner-computes local blocks, or the local `DistributedVector` blocks on the
  replicated path), the small projected matrix `H_proj`, the Ritz
  values/residuals, the next iteration index, the subspace size and block
  size, the matvec/history counters, and the rank's local Ritz vectors so a
  run that has exhausted its budget can still return them. No rank ever
  gathers the full subspace, so the checkpoint's size and I/O stay `1/P` per
  rank.
- `manifest.json` — written by rank 0 only. It records the format version,
  the latest iteration, and the run identity: integrals fingerprint, basis
  identity hash (mode, basis kind and dimensions, particle sector, orbital
  count), world size, backend, matvec layout, dtype and Davidson dtype, block
  size, root count, spin-penalty parameters, the full rank-partition identity
  (row/tile/vector bounds and grid) and the iterate-affecting solver knobs.
  The complete solver-knob snapshot is kept in the manifest's `extra` block
  for diagnosis but is not part of the match.

Both files are written through a same-directory temp file and a rename, so a
reader never sees a half-written file. The write happens at the top of a
Davidson iteration, when `iteration > 0` and
`iteration % checkpoint_every == 0`; that point is reached by every rank with
identical control flow, so all ranks write their own shard in lockstep. Rank
0 writes the manifest after its shard. A crash between the shard writes and
the manifest write leaves an older, still consistent set of shards plus
manifest; the next resume either uses that set or, if the shards disagree
with the manifest, treats the checkpoint as incomplete.

**Resume decision.** `torch.multiprocessing` workers and `torchrun` ranks
must all agree, so rank 0 reads the manifest and broadcasts it, every rank
validates its own shard, and one all-reduce makes the decision collective:

| policy | matching checkpoint | no manifest | manifest for another run | incomplete shard |
|---|---|---|---|---|
| `"auto"` (default) | resume | cold start | `RuntimeError` naming the differing fields | cold start with a warning |
| `"off"` | ignore, cold start | cold start | ignore, cold start | ignore, cold start |
| `"require"` | resume | `RuntimeError` | `RuntimeError` | `RuntimeError` |

The mismatch error names the manifest path and each differing identity field
with its saved and current value, and suggests `checkpoint_resume="off"` (or
clearing the subspace directory) to discard it. A resumed run continues from
the saved iteration, skipping the seed block: the saved `C`/`HC`/`H_proj`
are restored and the loop recomputes the projection and Ritz vectors exactly
as it would have on the uninterrupted run. The final energy therefore matches
the uninterrupted solve to round-off (tests compare at `1e-8`).

**Constraints.** The world size and rank layout are part of the identity: a
checkpoint written by two ranks is not resumed by four, and changing the tile
grid, the matvec layout (`local` vs `replicated`), the row/tile partition, the
basis, the integrals, the dtype/Davidson dtype, the block size, the root
count or the spin penalty is a mismatch. Implementation tolerances and the
iteration budget are deliberately excluded, so an interrupted run may be
resumed with a larger `max_davidson_iters` or a different `tol`. Because the
basis is part of the identity, an SQD outer loop whose CI basis changes
between calls must pass `checkpoint_resume="off"` (or use one directory per
call); otherwise the second call correctly reports the old state as a
different run. The path is
implemented for the owner-computes TPB and GDB Davidson loops (1-D and tiled),
the replicated Krylov fallback (real and complex), and the narrow
`davidson_dtype` subspaces. Two cases are not checkpointed: a basis that fits
in one Davidson block takes the dense escape (there are no iterations to
resume), and the spinless (scalar `nelec`) path rejects `checkpoint_dir`
rather than silently skipping it. The missed-root search after a converged
multi-root solve is not part of the saved state; a restart re-runs it from
the recovered primary solution.

**Observability.** Rank 0 appends one entry per subspace to
`solver.stats["checkpoint_log"]` (`subspace`, `action` — `"resumed"`,
`"cold"`, `"cold_incomplete"` or `"disabled"` — `iteration`, `checkpoint_dir`,
`resume`, `world_size`, and a `reason` for an incomplete checkpoint) and
counts manifest writes in `solver.stats["checkpoint_writes"]`.
`solver.preflight(...)` reports the configured policy and, per subspace,
whether a manifest exists, its iteration, and which fields it could already
compare (`plan["checkpoint"]`); the solve re-checks the full identity.

## 6. Memory accounting per rank

With `P` ranks, fp64 (`itemsize = 8`), Davidson block size `b`, active-space
orbitals `L`, and `N` determinants, the owner-computes TPB path holds:

| item | per rank | notes |
|---|---|---|
| Davidson block `C` | `b * N/P * 8` | 20 GB at `b = 20`, `N = 10^9`, `P = 8` |
| Davidson block `HC` | `b * N/P * 8` | same again — the dominant item |
| checkpoint shard (opt-in) | `~2b * N/P * 8` on disk | `C` + `HC` + the small `H_proj`/Ritz state, §5.6 |
| `hii`, W, eigenvectors, x/y | `~4 * N/P * 8` | local blocks only |
| staged exchange buffers | `≤ 2 * N/P * 8` | two receive buffers, rows a peer references |
| send gather | `~N/P * 8` | worst case when the sent rows are non-contiguous |
| `H_proj`, Gram, Ritz matrices | `O(b^2) * 8` | replicated, kilobytes |
| basis strings `(M_a + M_b)` | `(M_a + M_b) * 8` | replicated, MB for `M ~ 10^5` |
| helper tables | `O(excitations for owned α rows + all β)` | built per rank; the `(M,M)` temporary is gone (§4.3) |
| integrals `(2L)^4` | replicated | 0.8 GB at `L=50`, 12.8 GB at `L=100`, 64.8 GB at `L=150` |
| result hand-back | 0 (small: rank 0 only) | large: rows written to TMPDIR, then mapped by the parent (§5.5) |

For an fp32 or complex64 narrow subspace the `C`/`HC` rows halve; the operator
still runs full precision internally, with one full-precision matvec vector
live.

GDB and spinless use "replicated input, owned output": the Davidson block,
`hii`, the owned seed and the output block are `1/P`, but each rank also holds
one full input vector `T_full` per matvec, the full pair/determinant basis and
the full streamed tables (never stored, so only the current chunk is live). GDB
with the default `apply_ab_projection` additionally holds `k_swap` (int64) and
`τ` (real, cast per call) — as does the replicated path. So the flat layouts
buy the `2b` subspace term, not the state vector itself; this is the documented
trade-off of §4.4.

Two consequences:

1. At large `N` the **Davidson block** (`2b` vectors), not the state vector,
   sets the per-GPU budget. Distributing the vectors is what makes large `N`
   fit; `b` and the fp32 subspace are the remaining levers.
2. At large `L` the **replicated ERI tensor** dominates and this implementation
   does *not* address it (Section 9). SBD subspaces typically have `L <= ~100`,
   where replication is still feasible; beyond that the integrals need their
   own distribution.

## 7. What was needed from `torch_sci_solver`

| need | status |
|---|---|
| basis contract (`adets_t`, `bdets_t`, `k = ia*M_b + ib`; GDB pair index) | used unchanged; basis order is fixed before partitioning |
| diagonal (`make_diag_terms_torch`) | built for the owned α rows only |
| matvec (`build_matvec -> v -> H v`) | batched single-GPU path unchanged (now with per-subspace caches); `LocalTPBOperator` / `LocalAlphaBetaProjector`, `LocalPairOperator` / `LocalPairSwapProjector` and `LocalSpinlessOperator` are the distributed local compute; the replicated fallback wraps the full matvec as all-gather input + local slice |
| helpers (`generate_all_torch_helpers`) | row-chunked builder (no `(M,M)` temporary) plus `generate_local_torch_helpers` for the owned α rows; `PairHelpers(pair_range=...)` and `_stream_spinless(src_dets=...)` restrict the flat layouts |
| integrals (`TorchIntegrals`) | replicated per rank by default. The compact spin-adapted store (`eri_compact`) keeps the spatial `L^4` tensor, 16x smaller, with an exact provenance check; enabled for every matvec except the TPB Triton kernel (which needs a flat dense copy) |
| Davidson (`davidson_torch`) | `davidson_distributed_local` adds the owner-computes contract, the narrow subspace, warm start, missed-root search and complex support; `davidson_distributed` stays for the replicated fallback (real or complex coefficients; no missed-root search) |
| RDMs | output-partitioned tiled builders on the local TPB and GDB pair paths (`distributed_rdm.py`, reached from `tiled_postprocess.py` / `pair_tiled_postprocess.py`), including the implicit `(P, 1)` partition of a declined auto grid; the replicated tiled builders when the full output fits the per-rank replication bound; full-vector kernels on the replicated fallback when no tile is configured or the basis fits one block; local occupancy sums and `spinless_rdm12_distributed` on the spinless path; `rdm_accumulation` in `stats` and `preflight()` reports the choice |
| spin penalty | the TPB tile operator composes `H + λ(S^2 − target)` locally (`tiled_s2_apply`, `tiled_s2_diagonal`), and so does the GDB pair tile (`pair_tiled_s2_apply`, `pair_tiled_s2_diagonal`); a TPB solve without a grid uses the implicit `(P, 1)` partition, and a GDB solve does so when its auto grid is declined; an explicit GDB pair list with no tile grid uses `PairLocalSpinSquared` on the 1-D owner-computes block for the penalty and the `<S^2>` diagnostic |
| stats | rank 0's cumulative stats are copied back to the parent |

## 8. Tests and usage

Scaling regressions are gated by `test_scaling_regression_gpu.py` (`gpu`,
`slow`). On two or more CUDA GPUs it solves a 9·10⁶-determinant synthetic TPB
problem on one GPU (in a spawned process) and on two NCCL ranks (1-D and a
forced tile grid) and requires a ≥ 1.15× per-matvec speedup, a rank-0 peak
below the single-GPU peak, matching converged energies and a Triton matvec. At
this size the replicated tables and exchange buffers dominate rank 0's
allocated peak (measured 0.975×), so the memory check is a strict-shrink
regression gate. The speedup and energy ratios hold on any hardware.

The primary, no-GPU checks are world_size=2 gloo solves on a synthetic
4-orbital / 4-electron full-CI space (N = 36), compared with the
single-process solver:

```bash
cd /path/to/sci-solvers
uv run pytest packages/torch-sci-solver/tests/test_distributed_solver.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_davidson.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_local_path.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_matvec.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_pairs.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_spinless.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_complex_matvec.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_local_davidson.py -v
uv run pytest packages/torch-sci-solver/tests/test_distributed_complex_local_davidson.py -v
```

`test_distributed_solver.py` covers config parsing, the `distributed=` /
`world_size=1` interface, the complex/spinless support dispatch, energy/root
parity (TPB multi-root/single-root and GDB), owner-computes vs replicated
selection (including the local αβ-swap projector), the tiny-block restart
floor, a 127 KiB result hand-back, and the persistent session.
`test_distributed_local_path.py` exercises the public solver end to end:
the local TPB projector, `init_vectors` warm start and its cold fallback, the
fp32 subspace (local and replicated), the missed-root search, plus the
GDB owner-computes path (energy, fp32, complex, missed root, warm init), the
complex TPB path, the fp32/complex64 narrow subspaces and the spinless flat
path.
`test_distributed_matvec.py` checks `LocalTPBOperator` against `mult_torch`
for single-process and 2/3-rank uneven partitions, the local projector against
the full-space formula, β-cache parity, cost weights, and (on CUDA) Triton
parity. `test_distributed_pairs.py` does the same for `LocalPairOperator`
(row-range table parity, 2/3-rank gloo, complex, projector, seeds) and
`test_distributed_spinless.py` for `LocalSpinlessOperator`. The complex
variants (`test_distributed_complex_matvec.py`,
`test_distributed_complex_local_davidson.py`) check the complex kernels,
staged exchange, projector and local Davidson. `test_distributed_local_davidson.py`
covers the partition/exchange primitives and `davidson_distributed_local`
against the exact spectrum. The GPU test is marked `gpu` and `slow` and skips
unless two CUDA devices and NCCL are available:

```bash
uv run pytest packages/torch-sci-solver/tests/test_distributed_solver.py -m gpu -v
```

`test_distributed_davidson.py` covers the low-level primitives (partition
arithmetic, a world_size=4 uneven-`N` gather, the owner-computes TPB matvec
against the full-vector matvec, a standalone Davidson-style subspace, and
tiny-block Davidson against the exact spectrum).

## 9. 8-GPU capacity estimate

Assumptions: one 8-GPU H100 node with NVLink/NCCL; a budget of 0.65 of free
memory per rank; the Davidson working vectors and matvec buffers are
partitioned across the 8 ranks; ERI and the β/α helper tables are replicated per
rank. "Global N" is the total determinant count over all ranks.

The model `(2b + ~9) * N/P * 8 B` (fp64) is a **lower bound on bytes**, i.e. an
upper bound on reachable N: it omits replicated ERI/helper tables and the
per-subspace setup. Measured per-rank usage has added roughly 145 B/N at
representative configurations, so treat the table as a ceiling rather than a
promise. A narrow subspace (`torch.float32`, or `torch.complex64` over
complex128) halves the `2b` term.

| precision | budget/GPU | b=30 | b=10 | b=2 |
|---|---|---|---|---|
| fp64 | 80 GB | <=~0.81B | <=~1.93B | <=~4.29B |
| narrow subspace (fp32 / complex64) | 80 GB | <=~1.43B | <=~2.94B | <=~5.08B |

The narrow row uses `(b + 9) * N/P * 8 B`: only `C`/`HC` halve; the ~9
full-precision working vectors do not.

Replicated ERI `(2L)^4 * 8 B` per rank (fp64, 80 GB, b=30; model ceiling).
The solver builds the bundle with `compact_eri=True` whenever the matvec is not
Triton, which stores only the spatial `L^4` tensor and reconstructs the
spin-orbital entries through a masked indexing wrapper (Section 6); the table
below is the dense form, 16x the compact one. A streamed-GDB Triton run keeps
the compact store: the entry-parallel doubles kernel needs a flat dense ERI
and declines it per table, so that matvec falls back to the batched scatter
(Section 4.2).

| L | ERI per rank | Global N ceiling |
|---|---|---|
| 50 | 0.8 GB | ~1.5B (unchanged) |
| 100 | 12.8 GB | ~1.3B |
| 150 | 64.8 GB | ~0.29B |

Caveats: ERI replication can bind before the Davidson working set; smaller `b`
means more iterations; the replicated fallback's all-gather moves
`(P-1)/P * N * itemsize` per rank per matvec (~3.5 GB per rank at `N=1e9` fp64,
`P=8`, not the ~1 GB of a local block), the flat GDB/spinless owner-computes
paths move one full `N * itemsize` per matvec through an asynchronous
`FlatGather` (their replicated-input trade-off, Section 4.4), while the TPB
owner-computes path
exchanges only the α rows a peer references, double-buffered.

## 10. Feasibility risks and limitations

1. **ERI storage (highest risk).** The dense spin-orbital tensor is
   `(2L)^4 * 8 B` replicated per rank: 0.8 GB at `L=50`, 12.8 GB at `L=100`,
   64.8 GB at `L=150` (fp64; complex128 doubles it). The state-vector
   distribution does not enlarge the reachable `L` on its own. The opt-in
   compact store (`TorchIntegrals.eri_compact`) keeps one spatial `L^4` tensor
   and serves the spin-orbital entries through a masked indexing wrapper — a
   verified 16x reduction (L=60 fp64: 1.66 GB -> 104 MB) with identical
   outputs — and the solver enables it for every non-Triton matvec. Still
   replicated, so the remaining mitigations are density fitting / THC
   (`L^3`) or a channel-aware scheduler that streams ERI blocks through the
   kernel. Orbital-block sharding was evaluated and dropped: its staged union
   measured 96-100% of the block grid on every full-CI basis, so it lowered
   resident storage only, not the per-matvec working set or the peak. For
   physical integrals the `(pq|rs)` supermatrix is a Coulomb
   Gram matrix, hence positive semidefinite, so a pivoted Cholesky
   decomposition factorizes it to any tolerance without an auxiliary basis
   (synthetic test integrals are indefinite and take a signed
   eigendecomposition). The single-process TPB matvec uses it for the αβ
   doubles channel (`ab_factorization="cholesky"`, `ab_factorized.py`'s
   `Σ_P A^α_P C (A^β_P)^T`, exact on the product basis): a compute change on
   the dominant channel, not a storage one, since the same-spin channels still
   read the compact ERI. A factorized same-spin channel (which needs
   intermediate strings outside the selected set) and a distributed factorized
   pass are not implemented.
2. **Flat-layout replicated input.** GDB and spinless now compute only their
   owned output rows, but each matvec all-gathers the full input vector, and
   the pair basis, `k_swap`/`τ` and the integrals stay replicated. Their
   per-rank memory is therefore `O(N)` in the state vector, unlike TPB; the
   `2b` Davidson term is what distribution buys there (Section 4.4). On a tile
   grid the spin_sq penalty, the full RDMs and the ⟨S²⟩ diagnostic run on the
   tiled pair path; a declined auto grid uses the implicit `(P, 1)` uid
   partition, and an explicit GDB pair list with no tile grid applies the
   penalty and the ⟨S²⟩ diagnostic on its 1-D block from the same gather. Only
   full RDMs without a tile grid, or a basis that fits in one Davidson block,
   keep the replicated full vector. The
   structural basis removes the `O(N)` pair list: the block matvec walks it,
   and the block-walked RDM/⟨S²⟩ accumulators run those diagnostics there too
   on any tile grid, including the implicit `(P, 1)` partition of a declined
   auto decision.
3. **Determinant lookup (low risk).** Index ownership is arithmetic; strings
   stay replicated and sorted; bitmask lookups are `searchsorted`. The real
   precondition is a globally fixed basis order before partitioning.
4. **RDM reduction (moderate).** Occupancies are local sums on the distributed
   path; full 1-/2-RDMs run on the tiled local path when a tile grid is used.
   The default builders hold one replicated `3 * L^4` accumulator per rank; the
   output-partitioned entries keep `O(L^4 / P)` instead and gather the full
   tensors to rank 0, and the solver selects them whenever the output exceeds
   the per-rank replication bound. The replicated full-vector path is used only
   without a tile grid (or for a basis that fits one Davidson block).
5. **Result transfer and parent materialization.** Rank 0 returns at most one
   full host vector per root; above 1 GiB the sibling ranks write memmapped
   files under `TMPDIR` and the parent maps them, so neither the worker
   `torch.save` nor the parent load copies a full array. The parent still holds
   the returned amplitudes, as in the single-process path; point `TMPDIR` at a
   large local disk for 10^9-scale results, and set `hand_back_dir` to shared
   storage on a multi-node run.
6. **GPU / NCCL and multi-node.** One process per GPU (no thread-pool fan-out
   inside a rank); NCCL for tensor collectives; multi-node needs
   `torchrun`/MPI rendezvous and matching device affinity. Complex timings must
   be synchronised before any Python branch. All ranks must run the same `eigh`
   implementation so the reduced values are identical.
7. **I/O.** The FCIDUMP is small and can be read per rank; an `integrals_h5`
   with a large ERI should be read in parallel or read once and broadcast.
   Warm-start state is a full CPU tensor (or a memmap-backed tensor above the
   threshold) on every rank, so it does not scale with `P`.
8. **Load balance.** Implemented as a cost-prefix split with a 1.25× row cap
   (Section 3.3); the cap can still leave a little imbalance on pathological
   bases.
9. **Numerical reproducibility.** `all_reduce` sums in a different order than a
   single process, so energies agree to round-off, not bitwise. Tests compare
   at `~1e-8` in fp64.
10. **Helper-table memory.** The row-chunked builder removes the `(M, M)`
    temporary, and the α tables are built for owned rows only. The doubles
    tables (56 B/entry) are still proportional to the local excitation count
    and can dominate at very large `M`; compact dtypes (`int32`/`int8`) are the
    next lever.
11. **Worker startup.** The persistent rank session (`persistent: true`,
   the default) initializes the process group and solver once and serves
   later calls, so an SQD loop does not pay the process/CUDA-context startup
   per iteration; `persistent: false` keeps the per-call spawn path. Rank 0
   builds the results; non-zero ranks return an empty `EigenStates`.
12. **Checkpoint I/O.** The checkpoint state is written at iteration
   boundaries to a shared filesystem, so the write latency is on the critical
   path every `checkpoint_every` iterations (`checkpoint_every=0` is rejected;
   a larger value lowers the I/O cost and raises the redo window). The
   manifest carries the world size and partition identity, so a job must be
   restarted with the same rank count and layout; the spinless path and the
   dense one-block escape are not checkpointed (§5.6).

Out of scope for the current implementation: density fitting / THC (a
two-index factorisation is not a valid DF form), tiled RDMs/`S²` on the
spinless layout, and per-pair arrays for the paths that still need them (see
Sections 4.2, 4.4 and the items above).
