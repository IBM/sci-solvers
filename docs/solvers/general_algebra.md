# The general-algebra family — `arpack`, `davidson`, `lobpcg`, `primme`, `slepc`, `evsl`, `pyscf_davidson`

These seven solvers share one base, `GeneralAlgebraSolverBase`
(`src/sci_solvers/energy_solvers/general_algebra_base.py`). None of them know any
quantum chemistry. The base does the chemistry: for each batch it **reduces** the
sampled α×β CI subspace to a generic operator (`EigenProblem` for real integrals,
`HermitianOperator` for complex Hermitian ones), hands that to the backend's
`_diagonalize`, and **packs** the returned eigenpairs back into `SCIResult`s
(amplitudes, occupancies, RDMs). A concrete solver writes only `_diagonalize` — how it
turns one operator into its lowest `nroots` eigenpairs.

Because the reduction is shared, so are the capabilities: every one of these solvers is
matrix-free over the same operator, honours the same spin-symmetrization, and packs
RDMs the same way. They differ only in the eigensolver behind `_diagonalize`.

## Shared behaviour

- **Complex Hermitian integrals** (SOC / GHF / k-point) build a complex operator
  (`HermitianOperator` in `mode="tpb"`, a complex `PairCIProblem` in `mode="gdb"`); a
  **scalar** `nelec` selects the spinless spin-orbital path. Complex is dispatched only for
  solvers whose `_diagonalize` handles a complex operator (`complex_capable = True`); the
  rest reject complex integrals loudly rather than dropping the imaginary part. **The
  complex gate is checked before the basis-mode branch, so the complex-capable set is the
  same for both `tpb` and `gdb`:** `arpack`, `davidson`, `lobpcg`, `primme`, and `slepc`
  (the last only with a complex-scalar PETSc build); `evsl` and `pyscf_davidson` reject
  complex in either mode.
- **Determinant basis (`mode="tpb"` / `"gdb"`).** By default (`mode="tpb"`) each batch's
  `(strs_a, strs_b)` is a **Cartesian product** basis — every α string paired with every β
  string, flat index `ia*Nb+ib`, dimension `Na·Nb`. With `mode="gdb"` the two arrays are
  read instead as **parallel pair lists**: `(strs_a[i], strs_b[i])` is one determinant, the
  basis is exactly those pairs (deduplicated and lex-sorted), and the dimension is the
  number of unique pairs — a **non-product** ("general determinant basis") subspace. GDB is
  backed by a distinct matrix-free operator (`PairCIProblem`, the vendored
  `torch_sci_solver` pair engine) that applies full-determinant Slater–Condon directly, and
  works for real and complex Hermitian integrals alike. Every general-algebra solver
  supports it **except** `pyscf_davidson`, whose `kernel_fixed_space` can only represent a
  product basis and so raises `NotImplementedError` under `mode="gdb"`. (Feeding a full
  product as a GDB pair list reproduces the TPB result exactly — the correctness anchor for
  the pair path.)
- **Selected (non-full-CAS) determinant lists.** The spinless path accepts an arbitrary
  determinant subset per batch — a strict subset of the canonical `(nmo, N)` list, in any
  order — as selected-CI screening supplies. Both the operator and the RDM
  reconstruction are exact on the subset: the RDM builders restrict each excitation to a
  selected source and re-index the intermediate determinant axis onto the *connected set*
  (the selected determinants plus every destination reachable from them by one
  excitation), so the two-body Gram term still visits intermediates outside the selected
  space without ever materializing the full canonical space. The full canonical list
  routes through the original zero-overhead path unchanged.
- **Spin symmetrization** (`symmetrize_spin=True`) exploits the closed-shell α↔β swap
  symmetry to solve each parity sector on a packed, roughly half-size
  `SymmetrizedEigenProblem` view (requires identical α/β string pools). `nroots==1`
  solves only the even sector; `nroots>1` solves both so a low odd-spin state is not
  missed.
- **RDMs / energy.** With `compute_rdm2=True` the spin-summed 2-RDM is built and the
  energy is recomputed from the RDMs (`⟨dm1,h⟩ + ½⟨dm2,eri⟩`, matching
  `solve_sci`); otherwise the adapter eigenvalue is used and `rdm2` is left `None`.
- **Hamiltonian backend** (`hamiltonian_backend="torch"` / `"fulqrum"`). By default the
  torch Slater–Condon kernel builds each batch's projected Hamiltonian. With
  `hamiltonian_backend="fulqrum"` (needs `uv sync --extra fulqrum`) it is assembled as a
  CSR matrix by [fulqrum](https://github.com/qiskit-community/fulqrum) instead, which is
  much faster to build on CPU. Only the builder changes: the eigensolver, multi-root,
  `symmetrize_spin`, `spin_sq` and the RDM packing are the same, and results match the
  torch backend. It covers `mode="tpb"` and `mode="gdb"` with real or complex Hermitian
  integrals, and the spinless (scalar `nelec`) path. Under `mode="gdb"`
  the operator is presented in `PairCIProblem`'s convention (interleaved-Wick signs), so
  the pair RDM packing is unchanged.
  `pyscf_davidson` rejects it, because it never reads the problem operator. It is
  CPU-only: a non-CPU `device` falls back to CPU with a warning (or raises with
  `device_fallback="raise"`); pass `device="cpu"` to avoid it. Installation, tests and
  conventions: [fulqrum.md](fulqrum.md).
- **Shared knobs:** `nroots`, `tol`, `maxiter`, `compute_rdm2`, `device`, `dtype`,
  `chunk_size`, `mode` (`"tpb"`/`"gdb"`), `hamiltonian_backend` (`"torch"`/`"fulqrum"`),
  `fulqrum_mode` (`"csr"`/`"csr_fast"`/`"matrix_free"`, see [fulqrum.md](fulqrum.md)),
  plus `symmetrize_spin` / `spin_sq`.

## Feature matrix

| solver | complex | GDB | multi-root | native spin | extra install | notable knobs |
|---|:---:|:---:|:---:|:---:|---|---|
| `arpack` | ✅ | ✅ | ✅ | filter | — (SciPy) | `use_operator`, `ncv` |
| `davidson` | ✅ | ✅ | ✅ | filter | — (in-house) | `davidson_block_size`, `max_cycle` |
| `lobpcg` | ✅ | ✅ | ✅ | filter | — (SciPy) | `seed` |
| `primme` | ✅ | ✅ | ✅ | filter | `primme` extra | — |
| `slepc` | ✅* | ✅ | ✅ | filter | `slepc` extra | `storage` (`shell`/`aij`), `eps_type`, `comm` |
| `evsl` | ❌ | ✅ | ✅ | filter | build `libevsl` | `window`, `n_slices`, DOS/Lanczos tuning |
| `pyscf_davidson` | ❌ | ❌ | ✅ | **native** | — (PySCF) | `spin_sq` |

*Except `pyscf_davidson`, spin is a **post-hoc** filter via `SpinFilterMixin`
(oversolve, classify each root by ⟨S²⟩, keep those matching the target). `pyscf_davidson`
projects natively.*

\* `slepc` solves complex Hermitian only when PETSc/SLEPc was built with **complex
scalars**; with a real-scalar PETSc it raises rather than dropping the imaginary part.

## Per-solver notes

### `arpack`
SciPy `eigsh` (`which="SA"`). `use_operator=True` drives the matrix-free
`LinearOperator`; `use_operator=False` assembles the sparse matrix instead. `ncv` sets
the Lanczos basis size.

### `davidson` (`MultiRootDavidsonSolver`)
In-house matrix-free multi-root Davidson (`davidson1`), conjugate-aware and
dtype-generic, so real symmetric and complex Hermitian go through the same iteration.
Distinct from the native single-root real-only `DavidsonSolver`. `davidson_block_size`
bounds the subspace before a restart and `max_cycle` caps total iterations — both the same
knobs (and the same `davidson1` kernel) as `DavidsonSolver`, so a value means the same
thing on either. `max_cycle` is the only accepted spelling here — the shared `maxiter`
names the same quantity, so passing it to this solver raises rather than offering two
names for one knob; omitting both uses the base default. Falls back to dense `eigh` for
small `n`.

### `lobpcg`
SciPy LOBPCG with a complex-safe trial block and diagonal preconditioner. `seed` fixes
the random initial block for reproducibility.

### `primme`
`primme.eigsh`, which handles complex Hermitian natively. Needs the `primme` extra
(`uv sync --extra primme` or `--extra external-solvers`).

### `slepc`
SLEPc/PETSc, matrix-free (`storage="shell"`, default) or assembled CSR (`storage="aij"`).
`storage` is an **operator-storage** choice, *orthogonal* to the family-wide TPB/GDB
`mode` (the CI-basis contract) — a GDB pair basis can be solved with either storage, and
the two knobs never collide because they now have distinct names. `eps_type` selects the
SLEPc eigensolver; `comm` an MPI communicator. Needs the `slepc` extra
(`petsc4py`/`slepc4py`), and complex support depends on the PETSc scalar build (see above).

### `evsl`
Interior-eigenvalue / **spectrum-slicing** solver (real only). With `window=(lo, hi)`
that energy interval is sliced directly; with no window a KPM-DOS pre-pass estimates a
lower band holding at least `nroots` eigenvalues. Tuning: `n_slices`, `dos_deg`,
`dos_nvec`, `lanczos_dim`, `pol_deg_max`, plus streaming/checkpoint options. Needs a
built `libevsl` — `scripts/energy_solvers/build_evsl.sh`, pointed at via `libpath` or the
`SCI_SOLVERS_EVSL_LIB` environment variable. Falls back to dense for `n < 200`.

### `pyscf_davidson`
PySCF `kernel_fixed_space` Davidson reading the raw integrals. The one general-algebra
solver with **native** spin projection: it targets a sector via PySCF `fix_spin_`
(`spin_sq`) instead of the post-hoc filter, so it does **not** mix in `SpinFilterMixin`.
It bypasses the packed sector view and always solves the full space. Real only, and
**product-basis only**: `kernel_fixed_space` builds its `SCIvector` over the Cartesian
product of `strs_a × strs_b`, so `mode="gdb"` raises `NotImplementedError` here — every
other general-algebra solver takes the pair (GDB) path.

## When to use which

- Want a complex Hermitian (SOC/GHF/k-point) solve → `davidson`, `arpack`, `lobpcg`,
  `primme`, or `slepc` (complex-scalar build).
- Interior eigenvalues / a specific energy window → `evsl`.
- A spin sector deep in the spectrum, projected natively (no oversampling) →
  `pyscf_davidson` (or the native [`pyscf`](pyscf.md)).
- No extra dependencies, just works → `arpack` or `davidson`.
- A non-product / explicit `(αᵢ, βᵢ)` pair basis (`mode="gdb"`) → any of these except
  `pyscf_davidson`.
