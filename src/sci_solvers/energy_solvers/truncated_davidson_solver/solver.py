"""
Truncated-Davidson sci_solver implementation.

Thin wrapper around ``truncated_davidson``, called with the raw Hamiltonian
data ``(ci_strings, one_body_tensor, two_body_tensor, norb, nelec)`` for each
batch. Each requested root's result is wrapped as a separate
:class:`SCIResult`, and all of them are packed into a single
:class:`~sci_solvers.eigen_states.EigenStates`, following the same convention
as the rest of this package.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import time
from typing import Any, List, Tuple, cast

import numpy as np
import torch
from loguru import logger
from pyscf.fci import selected_ci
from pyscf.fci.cistring import make_strings
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from sci_solvers.eigen_states import EigenStates
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    reject_init_vectors,
    reject_spinless_nelec,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin
from sci_solvers.energy_solvers.truncated_davidson_solver.hamiltonian import (
    insert_reference_strings,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.truncated_davidson import (
    truncated_davidson,
)
from sci_solvers.hamiltonian import HamiltonianData

_WARMUP_DONE = False


def _warmup_numba() -> None:
    """Force numba JIT compilation for both addr-table specializations (dense array and NumbaDict) before the real solve."""
    global _WARMUP_DONE
    if _WARMUP_DONE:
        return

    # Dense addr-table path (norb <= 24): tiny 4-orbital system.
    norb_w, nelec_w = 4, (2, 2)
    h1e_w = np.diag([-1.0, -0.5, 0.5, 1.0])
    eri_w = np.zeros((norb_w,) * 4)
    eri_w[0, 0, 0, 0] = 0.5
    strs_w = make_strings(range(norb_w), 2)
    truncated_davidson(
        (strs_w, strs_w),
        h1e_w,
        eri_w,
        norb_w,
        nelec_w,
        nroots=1,
        eps=1.0,
        max_iter=2,
        verbose=False,
    )

    # Dict addr-table path (norb > 24): norb=25 with 2 electrons gives only
    # C(25,2)=300 strings but forces the NumbaDict specialization of
    # _addr_get, compiling add_basis_vector_fused and calc_trunc_pcr_fused
    # with dict-typed addr tables so the first real large-system call
    # doesn't pay JIT cost.
    norb_d, nelec_d = 25, (2, 2)
    h1e_d = np.zeros((norb_d, norb_d))
    np.fill_diagonal(h1e_d, np.linspace(-1.0, 1.0, norb_d))
    eri_d = np.zeros((norb_d,) * 4)
    strs_d = make_strings(range(norb_d), 2)
    truncated_davidson(
        (strs_d, strs_d),
        h1e_d,
        eri_d,
        norb_d,
        nelec_d,
        nroots=1,
        eps=1.0,
        max_iter=2,
        verbose=False,
    )

    _WARMUP_DONE = True


def _to_dense(psi, dim: int) -> np.ndarray:
    """Convert a compact (idx, val) sparse vector to a dense float64 array of length `dim`."""
    vec = np.zeros(dim, dtype=np.float64)
    keys, vals = psi
    vec[keys] = vals
    return vec


def _make_symmetric(c):
    """Symmetrize CI amplitudes in-place: ``c = (c + c.T) / 2``, then renormalize.

    No-op for non-square arrays (open-shell systems where nelec_a != nelec_b).
    Logs a warning if the symmetrized state deviates from a singlet.
    """
    if c.shape[0] != c.shape[1]:
        logger.warning(
            "symmetrize_spin skipped: amplitude array is non-square %s "
            "(open-shell system with nelec_a != nelec_b)",
            c.shape,
        )
        return
    c += c.T
    c *= 0.5
    norm = np.linalg.norm(c)
    if abs(norm - 1) > 1e-6:
        logger.warning("State not singlet %g" % (norm - 1))
    c /= norm


def make_sci_result_from_amps(
    energy: float,
    amps: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    compute_rdm1: bool = False,
    compute_rdm2: bool = False,
    symmetrize_spin: bool = False,
) -> SCIResult:
    """Wrap a pre-shaped ``(na, nb)`` amplitude array as an :class:`SCIResult`.

    Parameters
    ----------
    energy : float
        Electronic energy of this root (no nuclear repulsion).
    amps : (na, nb) float64
        CI amplitude matrix over the provided CI strings.
    strs_a, strs_b : int64 ndarray
        Alpha/beta CI strings for the determinant space (must match amps shape).
        Each must be strictly ascending (sorted, unique): PySCF's ``selected_ci``
        RDM code assumes it and is silently wrong otherwise.
    norb : int
        Number of spatial orbitals.
    nelec : (nelec_a, nelec_b)
    compute_rdm1 : bool
        If True, compute the 1-RDM and orbital occupancies.
    compute_rdm2 : bool
        If True, also compute the 2-RDM (expensive for large spaces).
    """
    amps = np.ascontiguousarray(amps, dtype=np.float64)
    if symmetrize_spin:
        _make_symmetric(amps)
    sci_state = SCIState(
        amplitudes=amps,
        ci_strs_a=strs_a,
        ci_strs_b=strs_b,
        norb=norb,
        nelec=nelec,
    )
    if compute_rdm1 or compute_rdm2:
        vec_sci = selected_ci._as_SCIvector(amps, (strs_a, strs_b))
    if compute_rdm1:
        dma, dmb = selected_ci.make_rdm1s(vec_sci, norb, nelec)
        dm1 = dma + dmb
        orbital_occupancies = (np.diagonal(dma), np.diagonal(dmb))
    else:
        dm1 = None
        orbital_occupancies = None
    dm2 = selected_ci.make_rdm2(vec_sci, norb, nelec) if compute_rdm2 else None
    return SCIResult(
        energy=float(energy),
        sci_state=sci_state,
        orbital_occupancies=orbital_occupancies,
        rdm1=dm1,
        rdm2=dm2,
    )


def make_sci_result(
    energy: float,
    vec: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    compute_rdm1: bool = False,
    compute_rdm2: bool = False,
    symmetrize_spin: bool = False,
) -> SCIResult:
    """Wrap a flat CI vector as an :class:`SCIResult` (reshapes to ``(na, nb)``)."""
    amps = vec.reshape(len(strs_a), len(strs_b))
    return make_sci_result_from_amps(
        energy,
        amps,
        strs_a,
        strs_b,
        norb,
        nelec,
        compute_rdm1=compute_rdm1,
        compute_rdm2=compute_rdm2,
        symmetrize_spin=symmetrize_spin,
    )


class TruncatedDavidsonSolver(EnergyCalculatorBase):
    """
    sci_solver wrapping the truncated Davidson diagonalizer.

    Passes raw Hamiltonian data directly to :func:`truncated_davidson`; no
    dense matrix or linear operator is ever formed. Truncation thresholds and
    convergence parameters are fixed at construction time.

    NOTE: Doesn't support complex numbers yet
    NOTE: ``symmetrize_spin`` performs post-solve CI-vector symmetrization
    (``c = (c + c.T) / 2``) and singlet validation, applied independently to
    each returned root. It is a no-op for open-shell systems (non-square
    amplitude array).
    NOTE: restart is not currently supported by the underlying
    ``truncated_davidson`` solver, single-root or multi-root.
    """

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        nroots: int = 1,
        n_extra_roots: int = 1,
        eps: float = 1e-4,
        tol_residual: float | None = None,
        max_cycle: int = 50,
        delta_max: float = 1e-3,
        delta_min: float = 1e-7,
        growth_factor: float = 2.0,
        precomputed_diag: bool | None = None,
        lindep_cutoff: float = 1e-7,
        plateau_window: int = 3,
        plateau_rel_tol: float = 0.3,
        require_buffer_convergence: bool = True,
        davidson_block_size: int | None = None,
        ignore_res: bool | None = True,
        print_every: int = 5,
        add_reference_determinant: bool = True,
        backend: str = "numba",
        device: "str | torch.device" = "cpu",
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 2_000_000,
        **kwargs: Any,
    ):
        """
        Parameters
        ----------
        symmetrize_spin : bool
            If True, symmetrize the CI vector after solving
            (``c = (c + c.T) / 2``) and warn if the result deviates from a
            singlet, applied independently to each returned root. No-op for
            open-shell systems (non-square amplitude array).
        hamiltonian : HamiltonianData or path, optional
            Pre-built Hamiltonian data or a path to load one from disk.
            If ``None``, must be supplied per call.
        verbose : bool
            Print per-iteration diagnostics from the inner Davidson loop.
        nroots : int
            Number of lowest eigenstates to solve for. ``nroots=1`` recovers
            the original ground-state-only behavior.
        n_extra_roots : int
            Buffer roots solved alongside ``nroots`` (GD+k). Improves subspace
            resolving power near degeneracies; output is always the true lowest
            ``nroots`` by energy.
        eps : float
            Energy-change convergence threshold (Hartree), applied per root.
        tol_residual : float or None
            Per-root residual-norm threshold. Both ``eps`` and ``tol_residual``
            must be satisfied. Defaults to ``sqrt(eps)``.
        max_cycle : int
            Flat cap on Davidson iterations per batch — the loop here is flat
            (a restart collapses the subspace in place rather than starting an
            outer iteration), so this is a total budget, matching ``max_cycle``
            on ``PySCFSolver`` / ``DavidsonSolver`` rather than the nested
            solvers' ``max_davidson_iters``. Note an iteration here is *not* one
            matvec: this algorithm builds truncated residuals via PCR instead of
            forming H·x, so the cross-solver rule of thumb
            ``max_cycle ~= max_davidson_iters * davidson_block_size`` does not
            carry over — equal ``max_cycle`` means equal iteration counts, not
            equal work.
        delta_max : float
            Initial truncation threshold for the PCR step.
        delta_min : float
            Minimum truncation threshold; never lowered below this value.
        growth_factor : float
            Multiplier applied to each root's previous basis-vector size to
            set that root's target truncation size T each iteration.
        precomputed_diag : bool or None
            If True, precompute and cache the Hamiltonian diagonal. ``None``
            auto-selects based on ``norb`` (True if ``norb <= 18``).
        lindep_cutoff : float
            Relative S-eigenvalue cutoff inside the generalized eigensolver.
        plateau_window : int
            Residual-norm history length for stagnation detection.
        plateau_rel_tol : float
            Converge a root if its last ``plateau_window`` residual norms vary
            by less than this fraction (truncation-induced residual floor).
        require_buffer_convergence : bool
            If True, wait for all ``nroots_total`` slots to converge before stopping.
        davidson_block_size : int or None
            Davidson subspace size — basis vectors accumulated before the
            subspace is collapsed (restarted). Same role as
            ``davidson_block_size`` on the other solvers, but note two
            differences here: ``None`` (the default) disables restart entirely
            rather than selecting a default size, and the value maps straight
            through with no ``(nroots-1)*4`` widening correction (that is a
            PySCF-``davidson1`` quirk this kernel does not share). Must be
            ``>= nroots + n_extra_roots``, which the kernel validates.
        ignore_res : bool or None
            If True, skip the residual-norm check and converge on energy change alone.
            Defaults to True when ``nroots=1``, False otherwise.
        print_every : int
            Print a detailed energy summary every this many iterations (verbose only).
        add_reference_determinant : bool
            If True (default), inject the Hartree-Fock reference determinant into a selected-CI
            space that lacks it, so seeding never fails when the space happens to exclude the
            reference. If False, the space is used verbatim and solving raises if the reference
            is needed but absent. No effect on a full FCI space.
        device : str or torch.device
            Reserved for future GPU support; currently unused.
        dtype : torch.dtype
            Reserved for future mixed-precision support; currently unused.
        chunk_size : int
            Reserved for future batched sigma-vector support; currently unused.
        """
        super().__init__(
            hamiltonian=hamiltonian,
            symmetrize_spin=symmetrize_spin,
            verbose=verbose,
            nroots=nroots,
            **kwargs,
        )
        self.n_extra_roots = n_extra_roots
        self.eps = eps
        self.tol_residual = tol_residual
        self.max_cycle = max_cycle
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.growth_factor = growth_factor
        self.precomputed_diag = precomputed_diag
        self.lindep_cutoff = lindep_cutoff
        self.plateau_window = plateau_window
        self.plateau_rel_tol = plateau_rel_tol
        self.require_buffer_convergence = require_buffer_convergence
        self.davidson_block_size = davidson_block_size
        self.ignore_res = ignore_res
        self.print_every = print_every
        self.add_reference_determinant = add_reference_determinant
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

        _warmup_numba()

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        *,
        spin_sq: float | None = None,
        max_cycle: int | None = None,
        compute_rdm1: bool = True,
        compute_rdm2: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> List[EigenStates]:
        """
        Diagonalize the (single) batch via truncated Davidson.

        Calls ``truncated_davidson`` on the first (and only supported) batch
        requesting ``self.nroots`` eigenpairs, then reshapes each returned
        root's vector to ``(na, nb)`` amplitudes and wraps it as an
        :class:`SCIState` / :class:`SCIResult`. All solver hyperparameters
        (``nroots``, ``eps``, ``max_iter``, ``delta_max``, ``delta_min``,
        ``growth_factor``, ``dedup_overlap_threshold``) are read from the
        constructor.

        .. note::
            Only a single batch is supported. Passing more than one batch raises
            a ``ValueError``.

        Args:
            ci_strs_batches: List containing exactly one ``(strs_a, strs_b)``
                int64-array tuple.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals, shape ``(norb,)*4``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``. A scalar ``nelec`` (the spinless
                flat spin-orbital path) is rejected with
                ``NotImplementedError`` — this solver does not implement it
                (it is also real-only, a separate reason the complex-only
                spinless path could never reach it).
            spin_sq: Accepted for API parity with other solvers; ignored.
            max_cycle: Flat cap on Davidson iterations for this call. ``None``
                (default) uses the solver's own ``max_cycle`` set at
                construction — pass it only to override for a single call. This
                was previously accepted and silently ignored.
            **kwargs: Any remaining kwargs are ignored.

        Returns:
            A list with one :class:`~sci_solvers.eigen_states.EigenStates`
            holding ``self.nroots`` :class:`SCIResult` objects (fewer if the
            solver's subspace could not support the full request), ordered
            by ascending energy. Energies are electronic only (no nuclear
            repulsion).
        """
        reject_init_vectors(type(self).__name__, kwargs)
        # A per-call value overrides the constructor's; None defers to it.
        if max_cycle is None:
            max_cycle = self.max_cycle
        super().sci_solver_callable_eigen_states(
            ci_strs_batches,
            hcore,
            eri,
            norb,
            nelec,
            spin_sq=spin_sq,
            max_cycle=max_cycle,
        )
        reject_spinless_nelec(type(self).__name__, nelec)
        # Spatial path from here on (the scalar-nelec spinless case raised above).
        nelec = cast(Tuple[int, int], nelec)
        ci_strs_batches = cast(List[Tuple[np.ndarray, np.ndarray]], ci_strs_batches)

        if self.verbose:
            print(
                f"[TruncatedDavidson] hcore.shape={hcore.shape}, eri.shape={eri.shape}, "
                f"len(batches)={len(ci_strs_batches)}, nroots={self.nroots}, "
                f"per-batch sizes={[(len(b[0]), len(b[1])) for b in ci_strs_batches]}"
            )

        self.last_history: list = []
        results = []
        for strs_a, strs_b in ci_strs_batches:
            t0 = time.perf_counter()
            e_nuc = self.hamiltonian.e_nuc if self.hamiltonian is not None else 0.0
            # Resolve the HF-injected space up front so the strings used to
            # reshape and label the result stay consistent with the space the
            # inner solve actually runs on (injection changes na/nb).
            if self.add_reference_determinant:
                strs_a, strs_b = insert_reference_strings(
                    strs_a, strs_b, nelec, verbose=self.verbose
                )
            E, psi_ritz_list, dim, history = truncated_davidson(
                ci_strings=(strs_a, strs_b),
                one_body_tensor=hcore,
                two_body_tensor=eri,
                norb=norb,
                nelec=nelec,
                nroots=self.nroots,
                n_extra_roots=self.n_extra_roots,
                eps=self.eps,
                tol_residual=self.tol_residual,
                max_iter=max_cycle,
                delta_max=self.delta_max,
                delta_min=self.delta_min,
                growth_factor=self.growth_factor,
                precomputed_diag=self.precomputed_diag,
                lindep_cutoff=self.lindep_cutoff,
                plateau_window=self.plateau_window,
                plateau_rel_tol=self.plateau_rel_tol,
                require_buffer_convergence=self.require_buffer_convergence,
                max_space_size=self.davidson_block_size,
                ignore_res=self.ignore_res,
                print_every=self.print_every,
                verbose=self.verbose,
                e_nuc=e_nuc,
                # Injection (if requested) already applied above so the returned
                # vector matches strs_a/strs_b; don't repeat it inside.
                add_reference_determinant=False,
                backend=self.backend,
            )
            self.last_history.extend(history)
            if self.verbose:
                print(
                    f"[TruncatedDavidson] truncated_davidson runtime: {time.perf_counter() - t0:.3f}s "
                    f"({len(psi_ritz_list)}/{self.nroots} roots returned)"
                )

            order = np.argsort(E[: len(psi_ritz_list)])
            E = E[order]
            psi_ritz_list = [psi_ritz_list[i] for i in order]

            nb_full = len(strs_b)
            sci_results = []
            for root_idx, (energy, psi_ritz) in enumerate(zip(E, psi_ritz_list)):
                t0 = time.perf_counter()
                keys, vals = psi_ritz
                # Map flat indices to (alpha, beta) positions in the full (na, nb) grid
                row_idx = keys // nb_full
                col_idx = keys % nb_full
                # Restrict strs_a/strs_b to strings that actually appear (nnz only)
                uniq_rows, inv_rows = np.unique(row_idx, return_inverse=True)
                uniq_cols, inv_cols = np.unique(col_idx, return_inverse=True)
                amps_sparse = np.zeros((len(uniq_rows), len(uniq_cols)), dtype=np.float64)
                amps_sparse[inv_rows, inv_cols] = vals
                active_strs_a = strs_a[uniq_rows]
                active_strs_b = strs_b[uniq_cols]
                if self.verbose:
                    print(
                        f"[TruncatedDavidson] root {root_idx} sparse subset runtime: "
                        f"{time.perf_counter() - t0:.3f}s "
                        f"(nnz={len(keys)}, na={len(active_strs_a)}, nb={len(active_strs_b)})"
                    )

                t0 = time.perf_counter()
                sci_result = make_sci_result_from_amps(
                    energy,
                    amps_sparse,
                    active_strs_a,
                    active_strs_b,
                    norb,
                    nelec,
                    compute_rdm1=compute_rdm1,
                    compute_rdm2=compute_rdm2,
                    symmetrize_spin=self.symmetrize_spin,
                )
                if self.verbose:
                    print(
                        f"[TruncatedDavidson] root {root_idx} make_sci_result runtime: "
                        f"{time.perf_counter() - t0:.3f}s"
                    )
                sci_results.append(sci_result)

            results.append(EigenStates(states=sci_results))
        return results


class TruncatedDavidsonSolverSpinFilterMixin(SpinFilterMixin, TruncatedDavidsonSolver):
    """:class:`TruncatedDavidsonSolver` with post-hoc ``<S^2>`` projection via the mixin.
    Filters the solver's returned roots by ``<S^2>`` against a ``spin_sq`` target.
    """
