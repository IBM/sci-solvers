"""Base class for solvers that diagonalize through the generic EigenProblem reduction.

A "general algebra" solver is one that hands the sampled ``alpha x beta`` CI
subspace to a general-purpose eigensolver (SciPy ARPACK/LOBPCG, PRIMME, SLEPc,
EVSL, a matrix-free torch Davidson, ...). Those solvers know nothing about
quantum chemistry, so the problem must first be *reduced* to a generic operator
they understand. That reduction, and the reverse packing of eigenpairs back into
``SCIResult``s, is exactly what this base owns:

* build one :class:`~sci_solvers.eigen_problem.EigenProblem` per batch,
* optionally exploit the closed-shell alpha<->beta swap symmetry to solve on a
  packed :class:`~sci_solvers.eigen_problem.SymmetrizedEigenProblem` sector view,
* pack each eigenpair into a :class:`SCIResult` (amplitudes, occupancies, RDMs).

A concrete subclass writes only :meth:`_diagonalize` — how it turns one
``EigenProblem`` (full space or packed sector view) into its lowest ``nroots``
eigenpairs. See the solvers in
:mod:`sci_solvers.energy_solvers.general_algebra_solvers` (e.g. ``ArpackSolver``)
for minimal examples.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import importlib.util
import os
import warnings
from abc import abstractmethod
from typing import Any, List, Tuple

import numpy as np
import torch
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from sci_solvers.eigen_problem import (
    EigenProblem,
    EigenResult,
    PairCIProblem,
    SymmetrizedEigenProblem,
    resolve_matvec_impl,
)
from sci_solvers.eigen_states import EigenStates, SCIResultWithConvergence
from sci_solvers.energy_calculation_base import EnergyCalculatorBase, reject_init_vectors
from sci_solvers.energy_calculation_base import is_spinless_request as _is_spinless_request_fn
from sci_solvers.hamiltonian import HamiltonianData


def _root_state(converged: "np.ndarray | None", j: int) -> "str | None":
    """Map ``EigenResult.converged[j]`` to :class:`SCIResultWithConvergence`'s ``state``.

    ``converged is None`` (a backend that never sets it, e.g. one that raises
    on failure to converge rather than returning a partial result) maps to
    ``None`` -- the existing "unknown" convention, read via
    ``getattr(result, "state", None)``. Otherwise ``True``/``False`` map to
    the ``"converged"``/``"not_converged"`` strings that convention expects.
    """
    if converged is None:
        return None
    return "converged" if bool(converged[j]) else "not_converged"


def dense_fallback(problem: EigenProblem, nroots: int) -> EigenResult:
    """Exact ``eigh`` on the dense matrix — for tiny/degenerate subspaces.

    Shared by the matrix-free general-algebra solvers, which bail out to this when
    the subspace is too small for an iterative Krylov solve to be worthwhile.
    ``np.linalg.eigh`` handles real symmetric and complex Hermitian matrices alike
    (real eigenvalues in both cases), so no dtype branching is needed — the dtype
    of ``problem.sparse()`` is preserved end to end.
    """
    w, v = np.linalg.eigh(problem.sparse().toarray())
    return EigenResult(w[:nroots], v[:, :nroots])


def _is_complex_problem(problem: Any) -> bool:
    """Whether ``problem`` carries a complex operator dtype.

    Reads the problem's declared ``dtype`` when present (e.g. a
    :class:`~sci_solvers.eigen_problem.HermitianOperator`, numpy dtype), else falls
    back to the diagonal's dtype. Handles the chemistry
    :class:`~sci_solvers.eigen_problem.EigenProblem`'s *torch* dtype
    (``torch.float64``, which exposes an ``is_complex`` bool) too. Real problems
    report a real dtype and are never flagged complex.
    """
    dtype = getattr(problem, "dtype", None)
    if dtype is not None:
        is_complex = getattr(dtype, "is_complex", None)
        if isinstance(is_complex, bool):
            return is_complex  # torch.dtype exposes an ``is_complex`` bool attribute
        return np.iscomplexobj(np.empty(0, dtype=dtype))
    return np.iscomplexobj(problem.hdiag())


def _is_full_canonical_product_strs(
    strs_a: np.ndarray, strs_b: np.ndarray, norb: int, nelec: Tuple[int, int]
) -> bool:
    """Whether ``(strs_a, strs_b)`` is exactly the full canonical ``(n_a, n_b)`` product.

    ``True`` when both string lists, taken in *ascending sorted* order, equal the
    complete ``cistring.make_strings`` list for their respective electron count --
    the same full-canonical detection ``spinless_ci_problem._is_full_canonical``
    does for a single string list, generalized to the two independent alpha/beta
    lists a product-mode CI subspace carries. When ``True`` the selected-CI
    ``row_map`` machinery in :mod:`sci_solvers.complex_rdm` is a no-op identity and
    is skipped, so the RDM path stays byte-identical to the pre-selected-CI
    behavior for the common (unselected) case.
    """
    from pyscf.fci import cistring

    n_a, n_b = int(nelec[0]), int(nelec[1])
    if len(strs_a) != int(cistring.num_strings(norb, n_a)):
        return False
    if len(strs_b) != int(cistring.num_strings(norb, n_b)):
        return False
    canon_a = cistring.make_strings(range(norb), n_a)
    canon_b = cistring.make_strings(range(norb), n_b)
    return np.array_equal(np.sort(strs_a), canon_a) and np.array_equal(np.sort(strs_b), canon_b)


def _selected_row_maps_for_product(
    strs_a: np.ndarray, strs_b: np.ndarray, norb: int, nelec: Tuple[int, int]
) -> Tuple[Any, Any]:
    """Build ``(row_map_a, row_map_b)`` for :func:`sci_solvers.complex_rdm.make_rdm1_complex`
    / ``make_rdm12_complex`` from a product-mode problem's selected ``(strs_a, strs_b)``.

    Returns ``(None, None)`` when the strings are exactly the full canonical product
    (see :func:`_is_full_canonical_product_strs`) -- the zero-overhead full-canonical
    RDM path, unchanged from before selected-CI support existed. Otherwise builds
    two bitmask-keyed :class:`~sci_solvers.complex_rdm.SparseAddressMap` instances
    (one per spin), mirroring
    :func:`sci_solvers.spinless_ci_problem._selected_address_map`'s convention.
    """
    from sci_solvers.complex_rdm import SparseAddressMap

    if _is_full_canonical_product_strs(strs_a, strs_b, norb, nelec):
        return None, None
    row_map_a = SparseAddressMap(
        np.asarray(strs_a, dtype=np.int64), np.arange(len(strs_a), dtype=np.int64)
    )
    row_map_b = SparseAddressMap(
        np.asarray(strs_b, dtype=np.int64), np.arange(len(strs_b), dtype=np.int64)
    )
    return row_map_a, row_map_b


def _numpy_dtype(problem: Any) -> np.dtype:
    """Return the problem's operator dtype as a *numpy* dtype.

    A :class:`~sci_solvers.eigen_problem.HermitianOperator` stores a numpy dtype,
    but :class:`~sci_solvers.eigen_problem.EigenProblem` stores a *torch* dtype
    numpy cannot allocate with. Normalize both: complex -> ``complex128``, else
    ``float64`` (the historical real behavior of the SciPy backends).
    """
    return np.dtype(np.complex128 if _is_complex_problem(problem) else np.float64)


class GeneralAlgebraSolverBase(EnergyCalculatorBase):
    """Diagonalize each batch's CI subspace through the ``EigenProblem`` reduction.

    Owns the sci-problem -> operator reduction, the optional spin-sector
    symmetrization, and the eigenpair -> ``SCIResult`` packing. Subclasses supply
    only :meth:`_diagonalize`.

    Complex Hermitian integrals (spin-orbit coupling / GHF / SOC / k-point) and the
    spinless spin-orbital path are dispatched here: complex ``hcore``/``eri`` build a
    :class:`~sci_solvers.eigen_problem.HermitianOperator` (via the complex-CI
    builders) instead of an :class:`~sci_solvers.eigen_problem.EigenProblem`, and a
    *scalar* ``nelec`` selects the spinless path. Only solvers whose ``_diagonalize``
    handles a complex operator set ``complex_capable`` (inherited from
    :class:`~sci_solvers.energy_calculation_base.EnergyCalculatorBase`) to ``True``;
    the rest reject complex integrals loudly rather than dropping the imaginary part.
    """

    #: Whether this solver's ``_diagonalize`` can consume a non-product
    #: :class:`~sci_solvers.eigen_problem.PairCIProblem` (``mode="gdb"``). Every
    #: matrix-free general-algebra solver drives the operator interface and so can;
    #: :class:`PyscfDavidsonSolver` reads raw integrals through PySCF's product
    #: ``kernel_fixed_space`` and cannot represent a non-product basis, so it sets
    #: this ``False`` and ``mode="gdb"`` is rejected loudly rather than silently
    #: producing a Cartesian-product result.
    supports_gdb: bool = True

    #: Whether this solver diagonalizes the problem object the reduction builds, so
    #: ``hamiltonian_backend="fulqrum"`` can swap in fulqrum's projected Hamiltonian.
    #: :class:`PyscfDavidsonSolver` re-derives its own operator from the raw integrals
    #: through PySCF's ``kernel_fixed_space``, so it sets this ``False`` and the
    #: fulqrum backend is rejected loudly rather than silently ignored.
    supports_fulqrum_backend: bool = True

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        nroots: int = 1,
        tol: float = 1e-8,
        maxiter: int = 400,
        device: "str | torch.device" = "cpu",
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 2_000_000,
        matvec_impl: str = "batched",
        compute_rdm2: bool = False,
        mode: str = "tpb",
        hermiticity_atol: float = 1e-7,
        hermiticity_rtol: float = 1e-5,
        hamiltonian_backend: str = "torch",
        fulqrum_mode: str = "csr",
        device_fallback: str = "warn",
        **kwargs: Any,
    ):
        """Initialize the reduction knobs shared by every general-algebra solver.

        Args:
            symmetrize_spin: Exploit the closed-shell alpha<->beta swap symmetry to
                block-diagonalize ``H`` into parity sectors, solving each on a
                roughly half-size packed
                :class:`~sci_solvers.eigen_problem.SymmetrizedEigenProblem` view.
                Requires identical alpha/beta CI-string pools.
            hamiltonian: Optional bound Hamiltonian (see base class).
            verbose: Print per-batch progress.
            nroots: Number of roots (states) to solve for per batch.
            tol: Convergence tolerance forwarded to :meth:`_diagonalize`.
            maxiter: Iteration ceiling forwarded to :meth:`_diagonalize`.
            device: Torch device for the matvec kernel (``"cpu"`` or ``"cuda"``).
            dtype: Torch float dtype for the matvec kernel.
            chunk_size: Matvec batch size for the torch kernel.
            matvec_impl: ``torch_sci_solver`` matvec backend for the real-valued
                path: ``"batched"`` (default, portable), ``"triton"`` (fused
                kernel; CUDA + real dtype only, errors otherwise) or ``"auto"``
                (``"triton"`` on CUDA when available, else ``"batched"``). The
                complex path always uses ``"batched"`` and rejects an explicit
                ``"triton"``. Block matvecs under Triton apply columns one at a
                time (no ``vmap``).
            compute_rdm2: When ``True`` build the spin-summed 2-RDM per root and
                recompute the energy from the RDMs (``<dm1, h> + 1/2 <dm2, eri>``).
            mode: CI-basis contract for the batches. ``"tpb"`` (default) treats each
                batch as a Cartesian **t**\\ ensor-**p**\\ roduct **b**\\ asis
                (``alpha x beta``, dim ``Na * Nb``); ``"gdb"`` treats each batch as
                an explicit parallel ``(alpha_i, beta_i)`` **g**\\ eneral-\
                **d**\\ eterminant-**b**\\ asis pair list (dim = number of pairs),
                backed by :class:`~sci_solvers.eigen_problem.PairCIProblem`.
            hermiticity_atol: Absolute tolerance for the ERI/hcore Hermiticity
                guards in the complex CI builders (the ``(pq|rs) == conj((qp|sr))``
                checks). Only reaches the complex-Hermitian and spinless paths;
                the real path has no such guard. Two-component (KGHF/X2C1e, AMF-SOC)
                integrals are Hermitian only to SCF/AVAS round-off, so this and
                ``hermiticity_rtol`` are the knobs that admit that round-off without
                masking a genuine non-Hermitian operator. Forwarded verbatim from a
                solver's ``create_solver`` kwargs, so it is reachable from the
                pipeline's ``solver_kwargs`` without any new step setting.
            hermiticity_rtol: Relative tolerance for the same Hermiticity guards.
                ``np.allclose`` compares against ``atol + rtol * |b|``, so a nonzero
                ``rtol`` accepts round-off that scales with the element magnitude.
            hamiltonian_backend: What builds each batch's projected Hamiltonian.
                ``"torch"`` (default) is the torch Slater-Condon kernel
                (:class:`~sci_solvers.eigen_problem.EigenProblem` /
                :func:`~sci_solvers.complex_ci_problem.build_selected_complex_ci_problem`).
                ``"fulqrum"`` assembles it with fulqrum instead
                (:func:`~sci_solvers.fulqrum_ci_problem.build_fulqrum_ci_problem`,
                needs the ``fulqrum`` extra); the eigensolver, spin handling and
                result packing are unchanged. ``"fulqrum"`` supports ``mode="tpb"``
                and ``mode="gdb"`` with real or complex Hermitian integrals, and the
                spinless (scalar ``nelec``) path. It is CPU-only: a non-CPU
                ``device`` is handled as ``device_fallback`` says.
            fulqrum_mode: How the fulqrum backend holds and applies the operator, in
                every basis mode and the spinless path: ``"csr"`` (default; one CSR
                matrix, lowest peak memory of the assembled modes), ``"csr_fast"``
                (about 30% faster build, about 2.5x the peak memory), or
                ``"matrix_free"`` (no matrix; regenerates the elements on every
                product, so memory stays near the basis size but each product costs
                several CSR ones). See :mod:`sci_solvers.fulqrum_ci_problem`. Ignored
                unless ``hamiltonian_backend="fulqrum"``; ``"matrix_free"`` is rejected
                on a solver configured to need an assembled matrix
                (:meth:`_needs_assembled_matrix`).
            device_fallback: What happens when ``hamiltonian_backend`` cannot run on
                ``device`` (today: fulqrum, which is CPU-only, with a GPU device).
                ``"warn"`` (default) warns and runs on ``"cpu"``: the same results,
                with different time and memory. ``"raise"`` raises ``ValueError``.
            **kwargs: Cooperative-inheritance passthrough (see
                :class:`~sci_solvers.energy_calculation_base.EnergyCalculatorBase`).
        """
        super().__init__(
            symmetrize_spin=symmetrize_spin,
            hamiltonian=hamiltonian,
            verbose=verbose,
            nroots=nroots,
            **kwargs,
        )
        if mode not in ("tpb", "gdb"):
            raise ValueError(f"mode must be 'tpb' or 'gdb', got {mode!r}.")
        if device_fallback not in ("warn", "raise"):
            raise ValueError(f"device_fallback must be 'warn' or 'raise', got {device_fallback!r}.")
        if hamiltonian_backend not in ("torch", "fulqrum"):
            raise ValueError(
                f"hamiltonian_backend must be 'torch' or 'fulqrum', got {hamiltonian_backend!r}."
            )
        if hamiltonian_backend == "fulqrum":
            if not self.supports_fulqrum_backend:
                raise ValueError(
                    f"{type(self).__name__} does not support hamiltonian_backend='fulqrum': "
                    "it builds its own operator from the raw integrals, so a fulqrum "
                    "Hamiltonian would be silently ignored. Use a matrix-free "
                    "general-algebra solver (arpack, davidson, lobpcg, primme, slepc, evsl)."
                )
            # fulqrum is CPU-only: its matrix is a CPU CSR array, so a GPU device would
            # be silently ignored (TPB) or only half-honored (GDB's RDM helpers). Fall
            # back to "cpu" before anything reads `device` (matvec_impl, RDM helpers).
            # TODO(fulqrum-gpu): drop this once fulqrum gains GPU support.
            if torch.device(device).type != "cpu":
                message = f"hamiltonian_backend='fulqrum' is CPU-only; got device={device!r}."
                if device_fallback == "raise":
                    raise ValueError(
                        f"{message} Pass device='cpu', or use hamiltonian_backend='torch' to "
                        "run on a GPU."
                    )
                warnings.warn(
                    f"{message} Running on device='cpu' instead; pass "
                    "device_fallback='raise' to make this an error.",
                    stacklevel=2,
                )
                device = "cpu"
            if importlib.util.find_spec("fulqrum") is None:
                raise ImportError(
                    "hamiltonian_backend='fulqrum' needs the optional fulqrum package; "
                    "install it with `uv sync --extra fulqrum`."
                )
        from sci_solvers.fulqrum_ci_problem import FULQRUM_MODES

        if fulqrum_mode not in FULQRUM_MODES:
            raise ValueError(f"fulqrum_mode must be one of {FULQRUM_MODES}, got {fulqrum_mode!r}.")
        self.tol = tol
        self.maxiter = maxiter
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size
        # Resolve now so a bad request fails at construction, not mid-run.
        self.matvec_impl = resolve_matvec_impl(matvec_impl, device, dtype)
        self._matvec_impl_requested = matvec_impl
        if matvec_impl == "triton" and mode == "gdb":
            raise NotImplementedError("matvec_impl='triton' is not available in mode='gdb'.")
        self.compute_rdm2 = compute_rdm2
        self.mode = mode
        self.hermiticity_atol = hermiticity_atol
        self.hermiticity_rtol = hermiticity_rtol
        self.hamiltonian_backend = hamiltonian_backend
        self.fulqrum_mode = fulqrum_mode
        self.device_fallback = device_fallback

    def _needs_assembled_matrix(self) -> bool:
        """Whether this solver, as configured, asks the problem for its matrix.

        ``False`` for the matrix-free solvers; a subclass whose options select an
        assembled matrix (``problem.sparse()``) returns ``True`` for those. A
        matrix-free fulqrum operator would have to be assembled densely for them, so
        that combination is rejected (:meth:`_check_fulqrum_mode`).
        """
        return False

    def _check_fulqrum_mode(self) -> None:
        """Reject ``fulqrum_mode="matrix_free"`` on a solver that needs a matrix.

        Called by a subclass that overrides :meth:`_needs_assembled_matrix`, at the end
        of its ``__init__``, once the options that method reads are set.
        """
        if (
            self.hamiltonian_backend == "fulqrum"
            and self.fulqrum_mode == "matrix_free"
            and self._needs_assembled_matrix()
        ):
            raise ValueError(
                f"fulqrum_mode='matrix_free' cannot be used with {type(self).__name__} "
                "configured to assemble the matrix (it would be built densely); use "
                "fulqrum_mode='csr', or the solver's matrix-free setting."
            )

    @abstractmethod
    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Return the lowest ``nroots`` eigenpairs of one subspace.

        The single hook a concrete general-algebra solver implements. ``problem``
        may be a full :class:`~sci_solvers.eigen_problem.EigenProblem` or a packed
        :class:`~sci_solvers.eigen_problem.SymmetrizedEigenProblem` sector view;
        both expose the same operator interface (``matvec``, ``linear_operator``,
        ``sparse``, ``hdiag``), so the implementation need not distinguish them.
        The caller re-sorts, so returning unsorted eigenpairs is fine.
        """

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: "List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray]",
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: "Tuple[int, int] | int",
        *,
        nroots: int | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """Diagonalize each batch's CI subspace via :meth:`_diagonalize`.

        Three reduction paths are dispatched on the integral dtype and ``nelec``:

        * **real spatial** — real integrals, ``(n_a, n_b)`` tuple: build an
          :class:`~sci_solvers.eigen_problem.EigenProblem` per ``(strs_a, strs_b)``.
        * **complex spatial** — complex Hermitian integrals, ``(n_a, n_b)`` tuple:
          build a :class:`~sci_solvers.eigen_problem.HermitianOperator` via
          :func:`~sci_solvers.complex_ci_problem.build_selected_complex_ci_problem`.
        * **complex spinless** — a *scalar* ``nelec`` (total ``N``): each batch is a
          flat ``strs`` array; see :meth:`_solve_spinless_batches`.

        Args:
            ci_strs_batches: ``(strs_a, strs_b)`` tuples (spatial) or flat ``strs``
                arrays (spinless). In ``mode="tpb"`` each string array must be
                strictly ascending (sorted, unique).
            hcore: One-electron integrals, shape ``(norb, norb)`` (may be complex).
            eri: Two-electron integrals, full 4-index chemists' notation (may be complex).
            norb: Number of spatial orbitals (spin-orbitals ``nmo`` when spinless).
            nelec: ``(n_alpha, n_beta)`` for the spatial path, or a scalar total ``N``
                for the spinless spin-orbital path.
            nroots: Solve-time override of the constructor ``nroots``. Lets a
                decorator (e.g. ``SpinFilterMixin``) oversolve without mutating the
                solver. ``None`` uses ``self.nroots``.
            **kwargs: Accepted for API parity; forwarded to the base guard.
                ``spin_sq`` is rejected here — see Raises.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in input
            order, each holding the lowest ``nroots`` roots (ascending energy).

        Raises:
            TypeError: If ``spin_sq`` is passed. On this family the spin target is a
                *constructor* argument (consumed by
                :class:`~sci_solvers.energy_solvers.spin_filter_mixin.SpinFilterMixin`),
                so a per-call value has nothing to act on.
            ValueError: If, in ``mode="tpb"``, a batch's ``strs_a`` or ``strs_b`` is not
                strictly ascending (sorted, unique).
        """
        reject_init_vectors(type(self).__name__, kwargs)
        # Rejected rather than ignored: nothing downstream reads a per-call spin_sq
        # here (the reduction only sees the operator), so accepting it would return a
        # silently unconstrained solve. Measured on a stretched-H4 subspace, the
        # difference between an honored and an ignored spin_sq is ~10 mHa — a wrong
        # answer that looks entirely successful.
        if "spin_sq" in kwargs:
            raise TypeError(
                f"spin_sq is not accepted per call by {type(self).__name__}; pass it to "
                "the constructor instead. On the general-algebra solvers the spin target "
                "is bound at construction (SpinFilterMixin oversolves and filters roots "
                "by <S^2>), so a per-call value would be silently dropped and the solve "
                "left unconstrained. Note the native solvers (pyscf, torch_sci, dice, "
                "sbd, truncated_davidson) are the opposite: they take spin_sq per call."
            )
        super().sci_solver_callable_eigen_states(ci_strs_batches, hcore, eri, norb, nelec, **kwargs)
        want = self.nroots if nroots is None else nroots
        is_complex = np.iscomplexobj(hcore) or np.iscomplexobj(eri)
        if is_complex and not self.complex_capable:
            # Lazy import: the factory imports the general-algebra solvers, so a
            # module-level import here would be circular.
            from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, is_complex_capable

            capable = ", ".join(repr(n) for n in AVAILABLE_SOLVERS if is_complex_capable(n))
            raise NotImplementedError(
                f"{type(self).__name__} cannot solve complex Hermitian integrals (it "
                "reads the real integrals directly / slices a real spectrum and would "
                f"drop the imaginary part). Use a complex-capable solver: {capable}."
            )

        if self._is_spinless_request(nelec):
            if self._matvec_impl_requested == "triton":
                raise NotImplementedError(
                    "matvec_impl='triton' is not available on the spinless path; "
                    "use 'auto' or 'batched'."
                )
            return self._solve_spinless_batches(ci_strs_batches, hcore, eri, norb, nelec, want)

        if self.mode == "gdb":
            return self._solve_pair_batches(ci_strs_batches, hcore, eri, norb, nelec, want)

        # Checked up front, before any solve: the real-path RDM and <S^2> packing
        # (PySCF selected_ci) is silently wrong on unsorted strings, while the energies
        # and amplitudes still look fine.
        for i, (strs_a, strs_b) in enumerate(ci_strs_batches):
            for name, strs in (("strs_a", strs_a), ("strs_b", strs_b)):
                if not np.all(np.diff(np.asarray(strs)) > 0):
                    raise ValueError(
                        f"batch {i}: {name} must be strictly ascending (sorted, unique) CI "
                        "strings in mode='tpb'. PySCF's selected-CI RDM and <S^2> code, "
                        "which builds the RDMs and the spin filter's classification, gives "
                        "wrong results otherwise. Sort them (e.g. np.unique(strs)), as "
                        "bitstring_matrix_to_ci_strs does."
                    )

        if self.verbose:
            print(
                f"[{type(self).__name__}] nroots={want}, complex={is_complex}, "
                f"len(batches)={len(ci_strs_batches)}, "
                f"per-batch sizes={[(len(b[0]), len(b[1])) for b in ci_strs_batches]}"
            )

        results: List[EigenStates] = []
        for strs_a, strs_b in ci_strs_batches:
            if self.hamiltonian_backend == "fulqrum":
                from sci_solvers.fulqrum_ci_problem import build_fulqrum_ci_problem

                problem = build_fulqrum_ci_problem(
                    strs_a,
                    strs_b,
                    hcore,
                    eri,
                    norb,
                    nelec,
                    atol=self.hermiticity_atol,
                    rtol=self.hermiticity_rtol,
                    fulqrum_mode=self.fulqrum_mode,
                )
            elif is_complex:
                from sci_solvers.complex_ci_problem import build_selected_complex_ci_problem

                if self._matvec_impl_requested == "triton":
                    raise NotImplementedError(
                        "matvec_impl='triton' does not support complex Hamiltonians; "
                        "use 'auto' or 'batched'."
                    )

                problem = build_selected_complex_ci_problem(
                    strs_a,
                    strs_b,
                    hcore,
                    eri,
                    norb,
                    nelec,
                    device=self.device,
                    chunk_size=self.chunk_size,
                    atol=self.hermiticity_atol,
                    rtol=self.hermiticity_rtol,
                )
            else:
                problem = EigenProblem(
                    strs_a,
                    strs_b,
                    hcore,
                    eri,
                    norb,
                    nelec,
                    device=self.device,
                    dtype=self.dtype,
                    chunk_size=self.chunk_size,
                    matvec_impl=self.matvec_impl,
                )
            k = min(want, problem.dim)
            res = self._solve_batch(problem, k)
            results.append(self._get_eigen_states(res, problem, compute_rdm2=self.compute_rdm2))
        return results

    # ---- spin symmetrization (generic; matvec-based backends only) ----------
    def _sectors_for_batch(self, k: int) -> Tuple[int, ...]:
        """Which alpha<->beta parity sectors to solve for ``k`` roots.

        ``k == 1`` -> ``(+1,)`` (the closed-shell ground state is even-spin);
        ``k > 1`` -> ``(+1, -1)`` so a low-lying odd-spin state is not missed.
        """
        return (+1,) if k == 1 else (+1, -1)

    def _solve_batch(self, problem: EigenProblem, k: int) -> EigenResult:
        """Solve one batch, applying the spin-symmetrization policy if enabled.

        Without ``symmetrize_spin`` the full ``D^2`` space is diagonalized. With
        it, each selected parity sector is solved on the packed view and — when
        both sectors are solved — the full-space roots are merged and the lowest
        ``k`` kept. Every general-algebra backend drives the matvec, so all can
        operate on the packed view (no per-backend exceptions).
        """
        if not self.symmetrize_spin:
            return self._diagonalize(problem, k).sorted()

        results = [self._solve_sector(problem, sector, k) for sector in self._sectors_for_batch(k)]
        if len(results) == 1:
            return results[0].sorted()
        energies = np.concatenate([r.energies for r in results])
        vectors = np.column_stack([r.vectors for r in results])
        # Only carry `converged` through if every sector actually reported one --
        # a partial/mixed signal (some sectors known, others not) would be
        # misleading to concatenate positionally.
        converged = (
            np.concatenate([r.converged for r in results])
            if all(r.converged is not None for r in results)
            else None
        )
        merged = EigenResult(energies, vectors, converged).sorted()
        merged_converged = merged.converged[:k] if merged.converged is not None else None
        return EigenResult(merged.energies[:k], merged.vectors[:, :k], merged_converged)

    def _solve_sector(self, problem: EigenProblem, sector: int, k: int) -> EigenResult:
        """Diagonalize one parity ``sector`` on the packed view, returning full vectors."""
        view = SymmetrizedEigenProblem(problem, sector=sector)
        kk = min(k, view.dim)
        packed = self._diagonalize(view, kk).sorted()
        return EigenResult(
            packed.energies, view.unpack_eigenvectors(packed.vectors), packed.converged
        ).sorted()

    # ---- spinless (spinor / GHF / SOC) routing ------------------------------
    @staticmethod
    def _is_spinless_request(nelec: Any) -> bool:
        """Whether ``nelec`` signals the genuinely spinless spin-orbital path.

        Thin delegate to the shared
        :func:`~sci_solvers.energy_calculation_base.is_spinless_request` —
        kept as a method here since existing callers within this class use
        ``self._is_spinless_request(...)``.
        """
        return _is_spinless_request_fn(nelec)

    def _solve_spinless_batches(
        self,
        ci_strs_batches: List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: int,
        want: int,
    ) -> List[EigenStates]:
        """Diagonalize each flat spinless batch via the spinless complex builder.

        Each batch is a single ``strs`` array of ``N``-particle spin-orbital
        determinant bitmasks (no ``(strs_a, strs_b)`` pair). The integrals must be
        complex Hermitian (a real Hamiltonian in this shape is rejected by
        :func:`~sci_solvers.spinless_ci_problem.build_spinless_complex_ci_problem`).
        Spin-sector symmetrization is a spatial alpha/beta concept with no meaning
        for a spinless operator, so each batch is solved directly over the full
        selected subspace (bypassing :meth:`_solve_batch`).
        """
        from sci_solvers.spinless_ci_problem import build_spinless_complex_ci_problem

        if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
            raise ValueError(
                "A scalar nelec selects the spinless (spinor / GHF / SOC) path, "
                "which is complex Hermitian by construction, but the integrals are "
                "real. Pass complex integrals, or use an (n_alpha, n_beta) tuple "
                "for the real spatial-orbital path."
            )

        n = int(nelec)
        if self.verbose:
            print(
                f"[{type(self).__name__}/spinless] nroots={want}, N={n}, nmo={norb}, "
                f"len(batches)={len(ci_strs_batches)}, "
                f"per-batch sizes={[len(b) for b in ci_strs_batches]}"
            )

        extra: dict[str, Any] = {}
        if self.hamiltonian_backend == "fulqrum":
            from sci_solvers.fulqrum_ci_problem import build_fulqrum_spinless_ci_problem

            build: Any = build_fulqrum_spinless_ci_problem
            extra = {"fulqrum_mode": self.fulqrum_mode}
        else:
            build = build_spinless_complex_ci_problem
        results: List[EigenStates] = []
        for strs in ci_strs_batches:
            problem = build(
                hcore,
                eri,
                n,
                strs,
                atol=self.hermiticity_atol,
                rtol=self.hermiticity_rtol,
                **extra,
            )
            k = min(want, problem.dim)
            res = self._diagonalize(problem, k).sorted()
            results.append(
                self._get_spinless_eigen_states(res, problem, compute_rdm2=self.compute_rdm2)
            )
        return results

    @staticmethod
    def _get_spinless_eigen_states(
        res: EigenResult, problem: Any, *, compute_rdm2: bool
    ) -> EigenStates:
        """Build one :class:`SCIResult` per root for a spinless flat CI vector.

        The spinless analog of :meth:`_get_eigen_states`. The eigenvector is a flat
        length-``dim`` 1-D vector over ``problem.strs`` (no ``(Na, Nb)`` reshape), so
        RDMs are built by the spinless ``(N, 0)``-sector helpers in
        :mod:`sci_solvers.spinless_ci_problem`. The state is a
        :class:`~sci_solvers.spinless_ci_problem.SpinlessSCIState` (a plain
        :class:`SCIState` would reject the 1-D amplitudes). The full spin-orbital
        1-RDM lands in the alpha block, so the single occupancy vector is its real
        diagonal. The ``(N, 0)``-sector RDM reuse assumes each batch is the full
        canonical determinant list.
        """
        from sci_solvers.spinless_ci_problem import (
            SpinlessSCIState,
            spinless_rdm1,
            spinless_rdm12,
        )

        nmo = problem.norb
        n = int(problem.nelec[0])
        states: List[SCIResult] = []
        for j in range(res.energies.shape[0]):
            amps = np.ascontiguousarray(res.vectors[:, j], dtype=np.complex128).reshape(-1)
            dm1, (dma, _dmb) = spinless_rdm1(amps, nmo, n, strs=problem.strs)
            occupancies = (np.real(np.diagonal(dma)),)
            if compute_rdm2:
                dm1, dm2 = spinless_rdm12(amps, nmo, n, strs=problem.strs)
                energy = np.real(
                    np.einsum("pr,pr->", dm1, problem.hcore)
                    + 0.5 * np.einsum("prqs,prqs->", dm2, problem.eri)
                )
            else:
                dm2 = None
                energy = np.real(res.energies[j])
            sci_state = SpinlessSCIState(
                amplitudes=amps,
                ci_strs=problem.strs,
                norb=nmo,
                nelec=problem.nelec,
            )
            states.append(
                SCIResultWithConvergence(
                    energy=float(np.real(energy)),
                    sci_state=sci_state,
                    orbital_occupancies=occupancies,
                    rdm1=dm1,
                    rdm2=dm2,
                    state=_root_state(res.converged, j),
                )
            )
        return EigenStates(states=states)

    # ---- non-product (GDB) pair-list routing --------------------------------
    def _solve_pair_batches(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        want: int,
    ) -> List[EigenStates]:
        """Diagonalize each batch as an explicit ``(alpha_i, beta_i)`` pair list.

        The ``mode="gdb"`` counterpart of the product loop in
        :meth:`sci_solver_callable_eigen_states`. Each batch's ``(strs_a, strs_b)``
        is treated as *parallel* pair lists (dim = number of pairs), not a Cartesian
        product, and reduced to a :class:`~sci_solvers.eigen_problem.PairCIProblem`
        backed by the vendored pair engine. Spin-sector symmetrization assumes a
        product basis (identical alpha/beta pools with a factorizable index), which a
        general pair list does not have, so each batch is solved directly over the
        full pair subspace (bypassing :meth:`_solve_batch`), exactly like the
        spinless path. Real and complex Hermitian integrals both flow through here;
        the complex gate in :meth:`sci_solver_callable_eigen_states` already rejects
        complex integrals on solvers that cannot handle them.
        """
        if not self.supports_gdb:
            raise NotImplementedError(
                f"{type(self).__name__} does not support mode='gdb' (non-product "
                "determinant-pair basis): it solves through PySCF's product "
                "kernel_fixed_space and cannot represent a non-product basis. Use a "
                "matrix-free general-algebra solver (arpack, davidson, lobpcg, primme, "
                "slepc, evsl) for gdb mode."
            )
        is_complex = np.iscomplexobj(hcore) or np.iscomplexobj(eri)
        if self.verbose:
            print(
                f"[{type(self).__name__}/gdb] nroots={want}, complex={is_complex}, "
                f"len(batches)={len(ci_strs_batches)}, "
                f"per-batch pairs={[len(b[0]) for b in ci_strs_batches]}"
            )

        results: List[EigenStates] = []
        if self.hamiltonian_backend == "fulqrum":
            from sci_solvers.fulqrum_ci_problem import FulqrumPairCIProblem

            pair_cls: type[PairCIProblem] = FulqrumPairCIProblem
            extra: dict[str, Any] = {
                "atol": self.hermiticity_atol,
                "rtol": self.hermiticity_rtol,
                "fulqrum_mode": self.fulqrum_mode,
            }
        else:
            pair_cls, extra = PairCIProblem, {}
        for strs_a, strs_b in ci_strs_batches:
            problem = pair_cls(
                strs_a,
                strs_b,
                hcore,
                eri,
                norb,
                nelec,
                device=self.device,
                dtype=self.dtype,
                chunk_size=self.chunk_size,
                **extra,
            )
            k = min(want, problem.dim)
            res = self._diagonalize(problem, k).sorted()
            results.append(
                self._get_pair_eigen_states(res, problem, compute_rdm2=self.compute_rdm2)
            )
        return results

    @staticmethod
    def _get_pair_eigen_states(
        res: EigenResult, problem: "PairCIProblem", *, compute_rdm2: bool
    ) -> EigenStates:
        """Build one :class:`SCIResult` per root for a non-product pair CI vector.

        The ``mode="gdb"`` analog of :meth:`_get_eigen_states`. The eigenvector is a
        flat length-``dim`` vector over the (deduped, lex-sorted) determinant pairs
        ``(problem.strs_a[i], problem.strs_b[i])`` — no ``(Na, Nb)`` reshape — so RDMs
        are built by the vendored pair-engine RDM helpers
        (``make_one_rdm_pair_torch`` / ``make_two_rdm_pair_torch``) and spin-summed
        the same way the upstream adapter packs them
        (``rdm1 = ga + gb``, ``rdm2 = Gaa + Gab + Gba + Gbb``). The mixed-spin block
        ``Gba`` is ``Gab`` with its two particles exchanged,
        ``Gba[p,q,r,s] = Gab[r,s,p,q]``. It equals ``Gab`` only when ``Gab`` is symmetric
        under that exchange, which holds for real integrals and for ``n_a == n_b``, but
        not for a complex open shell. The state is a
        :class:`~torch_sci_solver.torch_impl.pairs.PairSCIState` carrying the flat
        amplitudes and the parallel pair-string lists.

        For complex Hermitian integrals the pair eigenvector is complex and the RDMs
        are conjugate-correct (see the guarded bra conjugation in the vendored
        ``rdm_pairs``); occupancies use the real diagonal and the energy is the real
        part of ``<dm1, h> + 1/2 <dm2, eri>``.
        """
        from torch_sci_solver.torch_impl.pairs import PairSCIState
        from torch_sci_solver.torch_impl.rdm_pairs import (
            make_one_rdm_pair_torch,
            make_two_rdm_pair_torch,
        )

        norb = problem.norb
        n_a, n_b = int(problem.nelec[0]), int(problem.nelec[1])
        is_complex = problem.is_complex
        np_dtype = np.complex128 if is_complex else np.float64
        pair_basis = problem.pair_basis
        # `problem._ctx()` builds `pair_helpers` lazily; the solve above already ran
        # matvecs through it, so it is populated. Guard in case a backend solved
        # dense-only without ever calling the matvec.
        problem._ctx()
        pair_helpers = problem.pair_helpers

        states: List[SCIResult] = []
        for j in range(res.energies.shape[0]):
            amps = np.ascontiguousarray(res.vectors[:, j], dtype=np_dtype).reshape(-1)
            eigvec_t = torch.as_tensor(amps, dtype=problem.dtype, device=problem.device)

            ga, gb = make_one_rdm_pair_torch(
                eigvec_t, pair_basis, pair_helpers, norb, chunk_size=problem.chunk_size
            )
            dma = ga.cpu().numpy()
            dmb = gb.cpu().numpy()
            dm1 = dma + dmb
            occupancies = (np.real(np.diagonal(dma)), np.real(np.diagonal(dmb)))
            if compute_rdm2:
                Gaa, Gab, Gbb = make_two_rdm_pair_torch(
                    eigvec_t, pair_basis, pair_helpers, norb, chunk_size=problem.chunk_size
                )
                dm2 = (Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb).cpu().numpy()
                energy = np.real(
                    np.einsum("pr,pr->", dm1, problem.hcore)
                    + 0.5 * np.einsum("prqs,prqs->", dm2, problem.eri)
                )
            else:
                dm2 = None
                energy = np.real(res.energies[j])
            sci_state = PairSCIState(
                amplitudes=amps,
                ci_strs_a=problem.strs_a,
                ci_strs_b=problem.strs_b,
                norb=norb,
                nelec=(n_a, n_b),
            )
            states.append(
                SCIResult(
                    energy=float(np.real(energy)),
                    sci_state=sci_state,
                    orbital_occupancies=occupancies,
                    rdm1=dm1,
                    rdm2=dm2,
                )
            )
        return EigenStates(states=states)

    @staticmethod
    def _get_eigen_states(
        res: EigenResult, problem: EigenProblem, *, compute_rdm2: bool
    ) -> EigenStates:
        """Build one :class:`SCIResult` per solved root as an :class:`EigenStates`.

        Each root becomes a :class:`SCIResult` carrying an energy, its
        :class:`SCIState` (amplitudes reshaped to ``(Na, Nb)``), its
        ``orbital_occupancies`` (the diagonal of the spin-resolved 1-RDMs), and the
        spin-summed ``rdm1``. Roots are ascending in energy, so ``states[0]`` is
        the ground state.

        When ``compute_rdm2`` is ``True`` the spin-summed 2-RDM is also built and
        the energy is recomputed from the RDMs as ``<dm1, h> + 1/2 <dm2, eri>``,
        matching ``qiskit_addon_sqd.fermion.solve_sci``. When ``False`` the
        adapter's eigenvalue is used and ``rdm2`` is left ``None``.

        The real branch builds the RDMs with PySCF's ``selected_ci``, which needs
        ``problem.strs_a``/``strs_b`` strictly ascending (sorted, unique) and is
        silently wrong otherwise; :meth:`sci_solver_callable_eigen_states` rejects
        any other TPB batch up front.
        """
        from pyscf.fci import selected_ci

        is_complex = np.iscomplexobj(res.vectors)
        ci_strs = (problem.strs_a, problem.strs_b)
        if is_complex:
            # Selected-CI row maps (None, None) when strs_a/strs_b are exactly the
            # full canonical product -- the common case, and a zero-overhead no-op
            # there. Built once outside the per-root loop: identical for every root.
            row_map_a, row_map_b = _selected_row_maps_for_product(
                problem.strs_a, problem.strs_b, problem.norb, problem.nelec
            )
        states: List[SCIResult] = []
        for j in range(res.energies.shape[0]):
            if is_complex:
                # Complex Hermitian path: keep complex amplitudes and build RDMs with
                # the conjugate-correct complex builders (the pyscf real make_rdm* would
                # mis-handle a complex vector). Energies are real for a Hermitian H.
                from sci_solvers.complex_rdm import make_rdm1_complex, make_rdm12_complex

                amps = np.ascontiguousarray(
                    res.vectors[:, j].reshape(problem.Na, problem.Nb), dtype=np.complex128
                )
                dm1, (dma, dmb) = make_rdm1_complex(
                    amps,
                    problem.norb,
                    problem.nelec,
                    row_map_a=row_map_a,
                    row_map_b=row_map_b,
                    strs_a=problem.strs_a if row_map_a is not None else None,
                    strs_b=problem.strs_b if row_map_b is not None else None,
                )
                occupancies = (np.real(np.diagonal(dma)), np.real(np.diagonal(dmb)))
                if compute_rdm2:
                    dm1, dm2, _ = make_rdm12_complex(
                        amps,
                        problem.norb,
                        problem.nelec,
                        row_map_a=row_map_a,
                        row_map_b=row_map_b,
                        strs_a=problem.strs_a if row_map_a is not None else None,
                        strs_b=problem.strs_b if row_map_b is not None else None,
                    )
                    energy = np.real(
                        np.einsum("pr,pr->", dm1, problem.hcore)
                        + 0.5 * np.einsum("prqs,prqs->", dm2, problem.eri)
                    )
                else:
                    dm2 = None
                    energy = np.real(res.energies[j])
            else:
                amps = np.ascontiguousarray(
                    res.vectors[:, j].reshape(problem.Na, problem.Nb), dtype=np.float64
                )
                vec = selected_ci._as_SCIvector(amps, ci_strs)
                dma, dmb = selected_ci.make_rdm1s(vec, problem.norb, problem.nelec)
                occupancies = (np.diagonal(dma), np.diagonal(dmb))
                dm1 = selected_ci.make_rdm1(vec, problem.norb, problem.nelec)
                if compute_rdm2:
                    dm2 = selected_ci.make_rdm2(vec, problem.norb, problem.nelec)
                    energy = np.einsum("pr,pr->", dm1, problem.hcore) + 0.5 * np.einsum(
                        "prqs,prqs->", dm2, problem.eri
                    )
                else:
                    dm2 = None
                    energy = res.energies[j]
            sci_state = SCIState(
                amplitudes=amps,
                ci_strs_a=problem.strs_a,
                ci_strs_b=problem.strs_b,
                norb=problem.norb,
                nelec=problem.nelec,
            )
            states.append(
                SCIResultWithConvergence(
                    energy=float(energy),
                    sci_state=sci_state,
                    orbital_occupancies=occupancies,
                    rdm1=dm1,
                    rdm2=dm2,
                    state=_root_state(res.converged, j),
                )
            )
        return EigenStates(states=states)
