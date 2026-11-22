__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json

from sci_solvers import energy_calculation_base
from sci_solvers.energy_solvers.factory import create_solver

# maps between a string representation of the requested solver (type and kwargs) + hamiltonian path (fcidump) to a cached solver instance
CACHED_SOLVERS: dict[str, energy_calculation_base.EnergyCalculatorBase] = {}


def get_cached_solver(
    *, solver_type: str, fcidump_path: str, solver_kwargs: dict | None = None, verbose: bool = False
) -> energy_calculation_base.EnergyCalculatorBase:
    """
    Look up a cached solver instance by type, or create and cache it if not already present.
    Example usage:

    ```python
    from sci_solvers.energy_solvers.cached_per_hamiltonian_solver import get_cached_solver
    solver = get_cached_solver(solver_type="dice", fcidump_path="/path/to/molecule.fcidump", solver_kwargs={"mpirun_options" : "-n 16"},  verbose=True)
    solver_call_kwargs = dict(spin_sq=0.0, max_cycle=200}
    ci_strs_batches = ... # list of (strs_a, strs_b) batches
    results = solver.compute_energy(ci_strs_batches, **solver_call_kwargs)
    ```
    """
    if solver_kwargs is None:
        solver_kwargs = {}
    assert isinstance(solver_type, str), f"solver_type must be a string, got {type(solver_type)}"
    assert isinstance(solver_kwargs, dict), (
        f"solver_kwargs must be a dict, got {type(solver_kwargs)}"
    )
    assert isinstance(fcidump_path, str), f"fcidump_path must be a string, got {type(fcidump_path)}"

    hash_value = f"{solver_type}@{json.dumps(solver_kwargs, sort_keys=True)}@{fcidump_path}"
    if hash_value not in CACHED_SOLVERS:
        if verbose:
            print(
                f"cache miss - creating new solver instance for type={solver_type}, fcidump_path={fcidump_path}, solver_kwargs={solver_kwargs}"
            )
        CACHED_SOLVERS[hash_value] = create_solver(
            solver_type, hamiltonian=fcidump_path, **solver_kwargs
        )
    else:
        if verbose:
            print(
                f"cache hit - reusing existing solver instance for type={solver_type}, fcidump_path={fcidump_path}, solver_kwargs={solver_kwargs}"
            )
    return CACHED_SOLVERS[hash_value]
