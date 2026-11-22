# `pyscf` — the PySCF selected-CI baseline

`create_solver("pyscf", ...)` → `PySCFSolver`
(`src/sci_solvers/energy_solvers/pyscf_solver.py`).

Wraps PySCF's fixed-space selected-CI kernel. It is the **reference correctness
baseline** for this package: native spin projection, full 1- and 2-RDMs, and the
energy path most other solvers are validated against. It runs on CPU.

## Features

| feature | support |
|---|---|
| complex integrals | no (real only) |
| multi-root | yes — `nroots` / `n_roots` |
| spin projection | native, via PySCF's `fix_spin_` |
| GPU | no |
| 1-RDM / 2-RDM | yes / yes (2-RDM retained unless `with_rdm2=False`) |
| basis mode | n/a (TPB-style outer product) |

## Options

```python
solver = create_solver(
    "pyscf",
    symmetrize_spin=True,
    hamiltonian=ham,
    nroots=1,  # or n_roots=1 (alias); >1 solves excited states
    spin_shift=0.2,  # penalty strength forwarded to fix_spin_ when spin_sq is set
    with_rdm2=True,  # keep the 2-RDM in each SCIResult
)
```

Per-call kwargs on `sci_solver_callable`: `spin_sq` (target ⟨S²⟩, `None` to skip),
`max_cycle`, and any kwarg PySCF's `kernel_fixed_space` accepts.

## How it works

- **Single root** (`n_roots == 1`): one selected-CI solve per batch, wrapped so the
  converged flag is captured and reported on each result.
- **Multi-root** (`n_roots > 1`): the kernel solves for the lowest `n_roots` states;
  each state's energy is recomputed from its RDMs and the states are packed ascending
  in energy into one `EigenStates`.

Spin projection is applied natively inside the solve, so `pyscf` reaches a target spin
sector without oversampling — prefer it (or [`pyscf_davidson`](general_algebra.md#pyscf_davidson))
when the target sector sits deep in the spectrum. There is also a
`PySCFSolverSpinFilterMixin` variant for the generic post-hoc filter, but the native
path is the default and usually the right one.
