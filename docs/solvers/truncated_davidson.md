# `truncated_davidson` — the sparse truncated-Davidson solver

`create_solver("truncated_davidson", ...)` → `TruncatedDavidsonSolver`
(`src/sci_solvers/energy_solvers/truncated_davidson_solver/`). A native solver that
hands the raw Hamiltonian data `(ci_strings, h1e, eri, norb, nelec)` straight to the
in-house `truncated_davidson` routine — **no dense matrix or linear operator is ever
formed**.

The distinguishing feature is truncation: each iteration keeps only the largest CI
amplitudes (a per-root growth policy), so the working subspace stays small and the
returned state is **sparse**. That makes it the memory-frugal choice for a single
large batch, and its RDMs are built over the non-zero support only.

## Features

| feature | support |
|---|---|
| complex integrals | no (not yet) |
| multi-root | yes — `nroots` (+ `n_extra_roots` buffer) |
| spin projection | none — `spin_sq` is accepted for API parity but **ignored** |
| GPU | no (`device`/`dtype`/`chunk_size` reserved, currently unused) |
| 1-RDM / 2-RDM | yes / yes — built over the **sparse** non-zero support |
| basis mode | n/a (works on the sampled `(strs_a, strs_b)` grid) |
| batches | **single batch only** — more than one raises `ValueError` |

## Options

```python
solver = create_solver(
    "truncated_davidson",
    symmetrize_spin=True,
    hamiltonian=ham,
    nroots=1,  # lowest nroots states, packed ascending
    n_extra_roots=1,  # GD+k buffer roots for degeneracy resolution
    eps=1e-4,  # per-root energy-change convergence (Hartree)
    tol_residual=None,  # per-root residual threshold; defaults to sqrt(eps)
    max_cycle=50,  # flat iteration cap (an iteration is not one matvec here)
    delta_max=1e-3,  # PCR truncation thresholds
    delta_min=1e-7,
    growth_factor=2.0,  # per-root target subspace growth each iteration
    davidson_block_size=None,  # subspace size before restart; None disables restart
    add_reference_determinant=True,  # inject HF reference if the space lacks it
    backend="numba",  # inner-kernel implementation: "numba" (default) or "cpp"
)
```

## Inner-kernel backend (`numba` vs `cpp`)

The truncated-residual (PCR) inner kernels have two interchangeable implementations,
selected by `backend`:

- **`backend="numba"`** (default) — pure-Python/Numba kernels. Needs **no compiler**; this
  is what a plain `uv sync` gives you.
- **`backend="cpp"`** — an optional pybind11 C++ extension (`.../cpp/kernels.cpp`) for the
  identical algorithm. It is compiled **automatically** during `uv sync` / `uv pip install
  -e .` **when a C++ compiler is on `PATH`**; if none is found the build simply **skips** the
  extension (the install still succeeds on the numba path — it never fails the whole
  install). Requesting `backend="cpp"` when the extension was not built raises a **clear
  `ImportError`** telling you to rebuild — it does **not** silently fall back.

The two backends produce the **same results**; `cpp` is purely a performance path. No dense
matrix or linear operator is formed on either — the truncated PCR construction is unchanged.

## Multi-root

`nroots` lowest states are solved together, with `n_extra_roots` buffer slots
(GD+k) that improve resolving power near degeneracies; the output is always the true
lowest `nroots` by energy. Each root is re-expanded onto only the α/β strings that
actually appear (its non-zero support), so a root's `SCIState` and RDMs are built on
a subset of the batch grid.

## Constraints

- **One batch.** `sci_solver_callable_eigen_states` raises if handed more than one
  `(strs_a, strs_b)` pair — this solver is for a single large diagonalization, not the
  batched SQD inner loop with many subspaces.
- **No complex** integrals yet.
- **No native spin.** `symmetrize_spin` does post-solve CI-vector symmetrization
  (`c = (c + cᵀ)/2` with a singlet-deviation warning), a no-op for open-shell
  (non-square) amplitudes; it is not a spin projector. For a real ⟨S²⟩ target use
  `TruncatedDavidsonSolverSpinFilterMixin` (post-hoc filter) or a natively-projecting
  solver.
- **Restart** (`davidson_block_size`) collapses the subspace onto the current best
  Ritz vectors once it would exceed that size, locking already-converged roots.
  `None` (default) disables it, leaving the subspace unbounded.
- `device`, `dtype`, `chunk_size` are accepted but reserved for future GPU /
  mixed-precision work — they currently have no effect.
