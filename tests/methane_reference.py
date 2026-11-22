"""Bundled methane test system and its exact full-CI ground energy.

Methane (SQD molecules dataset ID 001, 10 electrons in 10 orbitals) is small enough that
PySCF's full CI gives the exact ground energy in about a second, so end-to-end SQD tests
can check their energy against it.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import functools

from pyscf import fci
from torch_sci_solver._paths import fcidump_path, shots_path
from torch_sci_solver.shots import load_shots_bit_array

from sci_solvers.hamiltonian import HamiltonianData


@functools.lru_cache(maxsize=1)
def methane() -> HamiltonianData:
    """The bundled methane Hamiltonian."""
    return HamiltonianData.from_fcidump(fcidump_path("methane"))


@functools.lru_cache(maxsize=1)
def methane_fci_energy() -> float:
    """Exact (full-CI) electronic ground energy of the bundled methane."""
    ham = methane()
    return float(
        fci.direct_spin1.FCI().kernel(
            ham.hcore, ham.eri, ham.num_orbitals, (ham.num_elec_a, ham.num_elec_b)
        )[0]
    )


def methane_bit_array():
    """The bundled methane SqDRIFT shots as a qiskit ``BitArray``."""
    return load_shots_bit_array(str(shots_path("methane")))
