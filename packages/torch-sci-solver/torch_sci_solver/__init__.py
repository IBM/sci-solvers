"""torch_sci_solver — a PyTorch-native selected-CI / SQD eigensolver.

A GPU-capable selected configuration interaction (SCI) engine with output-
stationary Triton kernels and a portable batched CPU backend. The core engine
depends only on ``torch`` and ``numpy``.

The optional integration with the SQD configuration-recovery loop
(``solve_fermion``, ``solve_sci_batch``, ``make_torch_sci_solver``) mirrors the
API of :mod:`qiskit_addon_sqd.fermion` and requires the ``sqd`` extra. These
symbols are imported lazily, so ``import torch_sci_solver`` stays light and does
not pull in ``qiskit-addon-sqd``:

.. code-block:: python

    from torch_sci_solver import make_torch_sci_solver
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    solver = make_torch_sci_solver(device="cuda")  # auto-selects Triton on GPU
    result = diagonalize_fermionic_hamiltonian(
        h1, h2, bit_array, samples_per_batch=..., norb=..., nelec=...,
        sci_solver=solver,
    )
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

__version__ = "0.8.0"

# Lazily forwarded from the optional qiskit-addon-sqd integration module so the
# base package imports without the ``sqd`` extra installed.
_LAZY_EXPORTS = frozenset({
    "make_torch_sci_solver",
    "solve_fermion",
    "solve_sci_batch",
    "solve_fermion_from_bit_array_pairs",
    "diagonalize_fermionic_hamiltonian_pairs",
    "dump_pair_coupling_magnitudes",
})

__all__ = ["__version__", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str):
    """Return the named attribute."""
    if name in _LAZY_EXPORTS:
        from torch_sci_solver import qiskit as _qiskit

        return getattr(_qiskit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return the attribute name list."""
    return sorted(set(globals()) | _LAZY_EXPORTS)
