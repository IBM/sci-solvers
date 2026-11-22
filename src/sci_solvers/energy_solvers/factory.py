"""Construct an energy calculator instance by name, and query its capabilities.

This is the entry point external tools (e.g. sweep launchers) use to pick a solver
from a string-valued config.
:func:`supports_multiroot` / :func:`is_general_algebra` answer capability questions by
name *without constructing* the solver.
Add a solver by
adding one ``_SOLVER_LOADERS`` entry (and the matching ``SolverType`` literal).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import importlib
from typing import Any, Callable, Literal

from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
)

_GA = "sci_solvers.energy_solvers.general_algebra_solvers"


def _loader(module: str, attr: str) -> "Callable[[], type[EnergyCalculatorBase]]":
    """A lazy ``name -> class`` loader: import the module, return the class.

    Returns the class object, never an instance, so the capability queries below
    can read its flags without constructing anything — safe for solvers whose
    ``__init__`` probes an external binary (dice/sbd) or needs an optional backend
    (primme/slepc import their backend lazily inside ``_diagonalize``).
    """

    def load() -> "type[EnergyCalculatorBase]":
        return getattr(importlib.import_module(module), attr)

    return load


# The single registry: solver name -> lazy class loader. Both ``create_solver``
# (dispatch) and the capability queries read this, so adding a solver is one entry.
_SOLVER_LOADERS: dict[str, Callable[[], type]] = {
    # native solvers (each wraps its own diagonalizer)
    "pyscf": _loader("sci_solvers.energy_solvers.pyscf_solver", "PySCFSolver"),
    "dice": _loader("sci_solvers.energy_solvers.dice_solver", "DICESolver"),
    "sbd": _loader("sci_solvers.energy_solvers.sbd_solver", "SBDSolver"),
    "torch_sci": _loader("sci_solvers.energy_solvers.torch_sci_solver", "TorchSCISolver"),
    "truncated_davidson": _loader(
        "sci_solvers.energy_solvers.truncated_davidson_solver", "TruncatedDavidsonSolver"
    ),
    # general-algebra solvers (diagonalize via the EigenProblem reduction)
    "arpack": _loader(_GA, "ArpackSolver"),
    "davidson": _loader(_GA, "MultiRootDavidsonSolver"),
    "lobpcg": _loader(_GA, "LobpcgSolver"),
    "primme": _loader(_GA, "PrimmeSolver"),
    "slepc": _loader(_GA, "SlepcSolver"),
    "evsl": _loader(_GA, "EvslSolver"),
    "pyscf_davidson": _loader(_GA, "PyscfDavidsonSolver"),
}

# Public: the names ``create_solver`` understands, derived from the registry (sorted
# for deterministic ordering). Useful for sweep launchers enumerating ``--solver``.
AVAILABLE_SOLVERS: tuple[str, ...] = tuple(sorted(_SOLVER_LOADERS))

# Static type alias for callers/type-checkers (lets them flag ``create_solver("pscf",
# ...)`` typos). A Literal cannot be derived from the registry at runtime, so this is
# the one hand-maintained list; a drift test asserts it matches ``AVAILABLE_SOLVERS``.
SolverType = Literal[
    # native solvers (each wraps its own diagonalizer)
    "pyscf",
    "dice",
    "sbd",
    "torch_sci",
    "truncated_davidson",
    # general-algebra solvers (diagonalize via the EigenProblem reduction)
    "arpack",
    "davidson",
    "lobpcg",
    "primme",
    "slepc",
    "evsl",
    "pyscf_davidson",
]


def create_solver(
    solver_type: "SolverType | str",
    **kwargs: Any,
) -> "EnergyCalculatorBase":
    """
    Construct an energy calculator instance by name.

    The unified entry point for sweep launchers and other tools that
    pick a solver from a config value (e.g. CLI arg, YAML key) and
    pass it whatever solver-specific kwargs the user supplied.

    Args:
        solver_type: One of ``"pyscf"``, ``"dice"``, ``"sbd"``,
            ``"torch_sci"``, or ``"multiroot"`` (see
            :data:`AVAILABLE_SOLVERS`). Matched case-insensitively.
        **kwargs: Solver-specific keyword arguments forwarded verbatim
            to the chosen solver's ``__init__``. See each solver class
            for the supported options. Common ones:

            - All solvers: ``verbose``, ``validation_mode``,
              ``symmetrize_spin``.
            - PySCFSolver: ``n_roots`` (>1 solves for excited states, read
              from ``states[1:]`` of each per-batch ``EigenStates``),
              ``spin_shift``, ``with_rdm2``, ``max_cycle``, plus any kwarg
              ``kernel_fixed_space`` accepts.
            - DICESolver: ``mpirun_options``, ``temp_dir``,
              ``clean_temp_dir``, ``mpi_root``, ``auto_setup_mpi``,
              ``select_cutoff``, ``tol`` (HCI energy convergence, Dice's
              ``dE``), ``davidson_tolerance`` (Dice's ``davidsonTol``),
              ``davidson_tolerance_loose`` (Dice's ``davidsonTolLoose``),
              ``max_hci_iters`` (HCI macro-iterations), ``fci``. Dice's
              Davidson iteration cap and block size are compile-time
              constants, so it has no ``max_davidson_iters`` /
              ``davidson_block_size``.
            - SBDSolver (the in-process ``sbd-eigensolver`` package):
              ``mode`` (``"tpb"`` or ``"gdb"``), ``device`` (``"cpu"``,
              ``"gpu"``, ``"gpu-omp"``, ``"auto"``), ``fcidump_path``,
              ``sbd_config`` (dict of SBD tunables, e.g. ``method``, ``eps``,
              ``max_it`` — the iteration cap; ``max_cycle`` is rejected),
              ``temp_dir``, ``clean_temp_dir``, ``compute_rdm2``.
            - TorchSCISolver: ``device``, ``devices`` (list of CUDA
              devices for single-node multi-GPU batch-parallel dispatch,
              e.g. ``["cuda:0", "cuda:1"]``), ``chunk_size``,
              ``davidson_block_size``, ``max_davidson_iters``,
              ``davidson_tolerance``,
              ``apply_ab_projection``, ``n_roots`` (>1 solves for excited
              states, read from ``states[1:]`` of each per-batch
              ``EigenStates``), plus any kwarg
              ``make_torch_sci_solver`` accepts.
            - General-algebra solvers (``arpack``, ``lobpcg``, ``primme``,
              ``slepc``, ``evsl``, ``pyscf_davidson``): ``nroots``, ``tol``,
              ``maxiter``, ``spin_sq``, ``compute_rdm2``, ``device``, ``dtype``,
              ``chunk_size``, ``hamiltonian_backend`` (``"torch"`` or
              ``"fulqrum"``; not ``pyscf_davidson``), plus per-solver knobs (e.g.
              ``window`` for ``evsl``, ``use_operator`` for ``arpack``).

    Returns:
        Initialised energy calculator instance.

    Raises:
        ValueError: If ``solver_type`` isn't one of the known names.
        TypeError: If ``solver_type`` isn't a string.

    Examples:
        >>> solver = create_solver("pyscf", verbose=True)
        >>> result = solver.compute_energy(hamiltonian, configurations)
        >>>
        >>> solver = create_solver("dice", mpirun_options="-n 4")
        >>> result = solver.compute_energy(hamiltonian, configurations, spin_sq=0.0)
        >>>
        >>> solver = create_solver("sbd", mode="gdb", device="gpu")
        >>> result = solver.compute_energy(hamiltonian, configurations)
    """
    if not isinstance(solver_type, str):
        raise TypeError(f"solver_type must be a string, got {type(solver_type)}")
    return solver_class(solver_type)(**kwargs)


def solver_class(solver_type: "SolverType | str") -> "type[EnergyCalculatorBase]":
    """Return the solver *class* for ``solver_type`` (imported, not constructed)."""
    if not isinstance(solver_type, str):
        raise TypeError(f"solver_type must be a string, got {type(solver_type)}")
    key = solver_type.lower()
    try:
        loader = _SOLVER_LOADERS[key]
    except KeyError:
        raise ValueError(
            f"Unknown solver type: {solver_type!r}. Available solvers: "
            f"{', '.join(AVAILABLE_SOLVERS)}"
        ) from None
    return loader()


def is_complex_capable(solver_type: "SolverType | str") -> bool:
    """Whether ``solver_type`` can diagonalize a complex Hermitian operator."""
    return bool(getattr(solver_class(solver_type), "complex_capable", False))


def supports_multiroot(solver_type: "SolverType | str") -> bool:
    """Whether ``solver_type`` returns more than one root per batch when asked."""
    return bool(getattr(solver_class(solver_type), "supports_multiroot", True))


def is_general_algebra(solver_type: "SolverType | str") -> bool:
    """Whether ``solver_type`` diagonalizes via the general-algebra EigenProblem reduction."""
    from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase

    return issubclass(solver_class(solver_type), GeneralAlgebraSolverBase)


def supports_spinless(solver_type: "SolverType | str") -> bool:
    """Whether ``solver_type`` accepts a *scalar* ``nelec`` (the flat spin-orbital,
    no-alpha/beta-split path -- GHF / SOC / generalized-spinor Hamiltonians).

    Not the same set as :func:`is_complex_capable`: every spinless-capable solver is
    also complex-capable (the spinless operator is complex Hermitian by
    construction), but the converse does not hold -- a solver can diagonalize a
    complex-*spatial* ``(n_a, n_b)`` Hamiltonian without implementing the spinless
    (scalar-``nelec``) reduction at all.
    """
    return bool(getattr(solver_class(solver_type), "supports_spinless", False))


def supports_init_vectors(solver_type: "SolverType | str") -> bool:
    """Whether ``solver_type`` honors a per-call ``init_vectors`` (Davidson start vectors).

    Lets a caller (e.g. an SQD loop warm-starting each iteration from the previous
    one) decide before the call whether to build the vectors at all; the other
    solvers raise ``NotImplementedError`` when one is passed.
    """
    return bool(getattr(solver_class(solver_type), "supports_init_vectors", False))
