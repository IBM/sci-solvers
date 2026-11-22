# `dice` — the Dice heat-bath CI solver

`create_solver("dice", ...)` → `DICESolver`
(`src/sci_solvers/energy_solvers/dice_solver.py`). Wraps
`qiskit-addon-dice-solver`'s `solve_hci`, shelling out to the external **Dice** C++
binary (the RIKEN branch) per batch.

Inside SQD the sampled subspace is already fixed, so the solver runs Dice with
`select_cutoff = INT_MAX` — Dice selects **no** further determinants and simply
diagonalizes the given space. It is a fixed-space diagonalizer here, not a
heat-bath selector.

## Features

| feature | support |
|---|---|
| complex integrals | no |
| multi-root | yes — `nroots` → `solve_hci`'s `n_roots` |
| spin projection | `spin_sq` must be `None` or `0.0` (rejects any other value) |
| GPU | no |
| 1-RDM / 2-RDM | no — returns per-orbital **occupancies** only |
| basis mode | n/a (Dice's own determinant space) |

## Prerequisites

Needs the Dice binary on `PATH` (or passed explicitly) and, for MPI, a working
`mpirun`. The binary is resolved via `qiskit_addon_dice_solver.resolve_dice_binary`.

## Options

```python
solver = create_solver(
    "dice",
    symmetrize_spin=True,
    hamiltonian=ham,
    nroots=1,  # >1 → solve_hci n_roots, lowest nroots states
    mpirun_options="-n 16",  # intra-call MPI parallelism
    temp_dir=None,  # per-batch scratch dir (each batch isolated)
    clean_temp_dir=True,  # cleanup handled by this class, not solve_hci
    select_cutoff=2147483647,  # INT_MAX — never select beyond the given space
    tol=...,  # forwarded to solve_hci as energy_tol
    max_hci_iters=...,  # forwarded to solve_hci as max_iter
    davidson_tolerance=1e-5,  # davidsonTol; davidson_tolerance_loose -> davidsonTolLoose
    fci=False,  # True → ci_strs=None (full FCI over the space)
)
```

## Spin handling

The RIKEN-branch Dice wrapper hardcodes `spin_sq=0.0` internally, so this solver
**only accepts `spin_sq=0.0` or `spin_sq=None`** — any other target raises
`ValueError` rather than silently returning the wrong sector. A
`DICESolverSpinFilterMixin` variant applies the generic post-hoc ⟨S²⟩ filter when a
non-trivial spin target is needed.

## Multi-root

A single `solve_hci` call returns the lowest `nroots` eigenpairs (its `n_roots`
argument), packed ascending in energy into one `EigenStates` per batch. `solve_hci`
returns a bare `(energy, sci_state, occupancies)` triple for one root and parallel
sequences for `n_roots > 1`; the wrapper normalizes both to length-`nroots` lists.

## Output

Dice returns per-orbital **occupancies** (and the CI state), not full 1-/2-RDMs. If
you need RDMs, use [`pyscf`](pyscf.md), [`torch_sci`](torch_sci.md), or a
general-algebra solver with `compute_rdm2=True`.
