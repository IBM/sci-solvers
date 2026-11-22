"""Load Hamiltonian → build TorchSCISolver via the factory → run SQD."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from functools import partial

from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
from torch_sci_solver._paths import (
    fcidump_path,
    shots_path,
)  # not needed when you use your own data
from torch_sci_solver.shots import load_shots_bit_array

from sci_solvers.energy_solvers.factory import create_solver
from sci_solvers.hamiltonian import HamiltonianData

# 1. Hamiltonian

ham = HamiltonianData.from_fcidump(fcidump_path())

# 2. Solver — factory picks TorchSCISolver, forwards kwargs to its __init__
solver = create_solver(
    "torch_sci",
    symmetrize_spin=True,
    device="cuda",  # or "cpu"
    hamiltonian=ham,
)

### instead of using the factory, you can directly instantiate the TorchSCISolver class (or any other solver):
# from sci_solvers.energy_solvers.torch_sci_solver import TorchSCISolver
# solver = TorchSCISolver(
#     symmetrize_spin=True,
#     device="cuda",           # or "cpu"
#     hamiltonian=ham,
# )

# 3. Measurement bitstrings → BitArray (uses the SqDRIFT shots bundled in torch_sci_solver/data/)
bit_array = load_shots_bit_array(str(shots_path()))


### you can use partial to set kwargs for the sci_solver_callable
sci_solver_callable = partial(
    solver.sci_solver_callable,
    spin_sq=None,  # no spin square penalty
    max_davidson_iters=200,  # Davidson restart cap for the inner solver
)

# 4. Qiskit's outer SQD loop, with our solver's callable as sci_solver
result = diagonalize_fermionic_hamiltonian(
    one_body_tensor=ham.hcore,
    two_body_tensor=ham.eri,
    bit_array=bit_array,
    samples_per_batch=512,
    norb=ham.num_orbitals,
    nelec=(ham.num_elec_a, ham.num_elec_b),
    max_iterations=3,
    sci_solver=sci_solver_callable,
    max_dim=2000,
    num_batches=1,
    symmetrize_spin=True,
)

print(f"electronic energy: {result.energy:.8f} Ha")
print(f"total energy:      {result.energy + ham.e_nuc:.8f} Ha")
