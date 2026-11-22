"""End-to-end test: factory-built TorchSCISolver + qiskit-addon-sqd loop on real SqDRIFT shots."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math

from methane_reference import methane, methane_bit_array, methane_fci_energy
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

from sci_solvers.energy_solvers.factory import create_solver


def test_torch_sci_via_factory_end_to_end():
    """Test torch_sci solver via factory end-to-end."""
    ham = methane()
    solver = create_solver(
        "torch_sci",
        symmetrize_spin=True,
        device="cpu",
        hamiltonian=ham,
    )
    bit_array = methane_bit_array()
    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=ham.hcore,
        two_body_tensor=ham.eri,
        bit_array=bit_array,
        samples_per_batch=512,
        norb=ham.num_orbitals,
        nelec=(ham.num_elec_a, ham.num_elec_b),
        max_iterations=3,
        sci_solver=solver.sci_solver_callable,
        max_dim=2000,
        num_batches=1,
        symmetrize_spin=True,
        seed=0,
    )
    assert math.isfinite(result.energy)
    # Variational: never below the exact energy; on methane SQD recovers it to ~1e-6 Ha.
    e_fci = methane_fci_energy()
    assert e_fci - 1e-8 <= result.energy < e_fci + 1e-4
