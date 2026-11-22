# Physics Module

This module contains physics-related functionality for quantum chemistry calculations.

## Energy Calculation Base Class

The `EnergyCalculatorBase` provides an abstract interface for computing ground state energies

Its method `sci_solver_callable` matches the signature of  `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`
(without additionaly optional args allowed).


## Usage Example

```python
from sci_solvers import EnergyCalculatorBase
import numpy as np

# Create a solver (concrete implementation), e.g. via the factory:
# from sci_solvers.energy_solvers.factory import create_solver
# solver = create_solver("pyscf", verbose=True)
solver = SomeSolver(..., verbose=True)

# Compute energy for a set of determinants
results = solver.sci_solver_callable(
    ci_strs_batches = ...,
    hcore = ...,
    eri = ...,
    norb = ...,
    nelec = ...,
    **kwargs: Any,
)

for subsample_result in results:
    print(f"total_energy {subsample_result.energy:.6f} Hartree.  Subspace dimension: {np.prod(subsample_result.sci_state.amplitudes.shape)}")


```

## Supported solvers

### 1. PySCF Solver (CPU)
- Uses PySCF's `selected_ci.kernel_fixed_space`
- Single-node, multithreaded via BLAS
- Good for small to medium systems (~10M determinants)

### 2. DICE Solver (CPU, MPI)
- Uses IBM Qiskit's DICE eigensolver wrapper
- MPI-based parallelization
- Supports S² penalty for spin enforcement
- Reference solver from SQD paper (arXiv:2405.05068)

### 3. sbd GPU Solver
- Uses RIKEN's sbd chemistry_tpb binary
- GPU-accelerated via Thrust
- Scales to 30M+ determinants
- runs the binary and parses parts of its stdout; reference evaluation scripts would let us remove the hackier parsing

### optional single function callable (matching qiskit-sqd-addon sci_solver signature)

Each concrete `EnergyCalculatorBase` subclass exposes `sci_solver_callable`, matching the
`sci_solver` callable signature expected by `qiskit_addon_sqd.fermion`. Build one via the
factory and grab the bound method:

```python
from sci_solvers.energy_solvers.factory import create_solver

pyscf_sci_solver = create_solver("pyscf", symmetrize_spin=True).sci_solver_callable
dice_sci_solver = create_solver("dice", symmetrize_spin=True).sci_solver_callable
sbd_sci_solver = create_solver("sbd", symmetrize_spin=True).sci_solver_callable
```



## Possible usage

- As part of a full configuration recovery process - e.g. S-CORE
- Bayesian recovery methods
- Hybrid recovery approaches
- Training AI model

The recovery workflows handle the iterative process of selecting determinants, while this module handles the energy calculation for each set of determinants.


### Troubleshooting

To install/build dice and sbd solver please go to `scripts/energy_solvers/README.md`
