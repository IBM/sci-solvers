"""Container for the eigenstates of a single diagonalized CI subspace."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import List

from qiskit_addon_sqd.fermion import SCIResult


@dataclass(frozen=True)
class SCIResultWithConvergence(SCIResult):
    """An :class:`SCIResult` carrying the inner (Davidson) eigensolver's state.

    Mirrors ``torch_sci_solver.torch_impl.qiskit_adapter.SCIResultWithConvergence``
    without depending on it (the torch package is standalone), so the dice,
    sbd and pyscf solvers can report convergence too.

    Attributes:
        state: ``"converged"``, ``"not_converged"``, or ``None`` when the solver
            exposes no convergence information. Read it as
            ``getattr(result, "state", None)`` so solvers that return a plain
            :class:`SCIResult` degrade to ``None``.
    """

    state: str | None = None


@dataclass
class EigenStates:
    """Eigenstates of one diagonalization (ground state plus excited states).

    A solver diagonalizes the Hamiltonian on a batch's ``(strs_a, strs_b)``
    subspace and returns the lowest ``nroots`` eigenpairs. This container holds
    those roots for a single batch: ``states[0]`` is the ground state and
    ``states[1:]`` are the excited states, in ascending energy order.

    Single-root solvers return an ``EigenStates`` with exactly one element.
    The qiskit-addon-sqd configuration-recovery loop consumes only the ground
    state; :meth:`~sci_solvers.energy_calculation_base.EnergyCalculatorBase.sci_solver_callable`
    unwraps ``states[0]`` per batch for it, while callers that want excited
    states read them from :meth:`sci_solver_callable_eigen_states`.

    Attributes:
        states: The eigenstates for one batch, one :class:`SCIResult` per root,
            sorted by ascending energy. ``states[0]`` is the ground state.
            ``SCIResult.energy`` values are electronic only (no nuclear
            repulsion), matching qiskit-addon-sqd's convention.
    """

    states: List[SCIResult]
