"""
PySCF-based sci_solver implementation.

Thin wrapper around ``pyscf.fci.selected_ci.kernel_fixed_space``, with a
convergence signal surfaced on every result and multi-root support.

Single-root path (``n_roots == 1``)
-------------------------------------
Calls ``qiskit_addon_sqd.fermion.solve_sci`` per batch via
:meth:`PySCFSolver._solve_one_with_convergence`, which monkey-patches
``SelectedCI`` to capture the Davidson ``converged`` flag that
``solve_sci`` would otherwise discard. Returns
:class:`~sci_solvers.eigen_states.SCIResultWithConvergence` so callers
can check ``result.state``.

Multi-root path (``n_roots > 1``)
------------------------------------
``qiskit_addon_sqd`` cannot be used because its ``solve_sci`` is
hard-wired for a single root.  The module-level
:func:`solve_sci_batch_multiroot` helper calls ``kernel_fixed_space``
directly with ``nroots > 1``, iterates over the returned list of CI
vectors to build one :class:`~sci_solvers.eigen_states.SCIResultWithConvergence`
per root, and reads ``myci.converged`` (available because we own the
``myci`` instance) to populate ``state`` on each root.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import warnings
from dataclasses import fields
from typing import Any, List, Tuple, cast

import numpy as np
from pyscf import fci
from pyscf.fci.selected_ci import cre_des_linkstr, des_des_linkstr
from qiskit_addon_sqd.fermion import SCIState, solve_sci

from sci_solvers.eigen_states import EigenStates, SCIResultWithConvergence
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    davidson_block_size_to_max_space,
    reconcile_nroots,
    reject_init_vectors,
    reject_spinless_nelec,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin
from sci_solvers.hamiltonian import HamiltonianData


def _rdm_link_index(ci_strs: Tuple[np.ndarray, np.ndarray], norb: int, nelec: Tuple[int, int]):
    """Link tables for the RDM builders, reused across all roots of a batch.

    These are the NON-tril constructors, which is what ``make_rdm1s`` and
    ``make_rdm2s`` build by default. ``kernel_fixed_space`` uses the tril
    variants for ``contract_2e``; the two are not interchangeable and mixing
    them yields wrong RDMs with no error.
    """
    strsa, strsb = ci_strs
    return (
        cre_des_linkstr(strsa, norb, nelec[0]),
        des_des_linkstr(strsa, norb, nelec[0]),
        cre_des_linkstr(strsb, norb, nelec[1]),
        des_des_linkstr(strsb, norb, nelec[1]),
    )


def solve_sci_batch_multiroot(
    ci_strings: List[Tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    n_roots: int,
    spin_sq: float | None = None,
    spin_shift: float = 0.2,
    max_cycle: int = 200,
    davidson_block_size: int | None = None,
    with_rdm2: bool = True,
    **kwargs: Any,
) -> List[EigenStates]:
    """Diagonalize each CI-string subspace for ``n_roots`` roots via PySCF.

    Mirrors ``solve_sci_batch`` in structure: one ``myci`` setup and one
    ``kernel_fixed_space`` call per batch entry. The only difference is that
    ``nroots`` is forwarded so ``kernel_fixed_space`` returns a list of CI
    vectors, and the post-processing loop iterates over that list to build
    one :class:`~sci_solvers.eigen_states.SCIResultWithConvergence` per root.

    Energy is always recomputed from the 1- and 2-RDMs (matching
    ``qiskit_addon_sqd.fermion.solve_sci``), because the raw Davidson
    eigenvalue is unreliable when ``fix_spin_`` is active — it includes the
    spin-penalty shift and is not the true Hamiltonian energy.

    Args:
        ci_strings: List of ``(strs_a, strs_b)`` int64-array tuples. Each array must
            be strictly ascending (sorted, unique): PySCF's ``selected_ci`` assumes it
            and is silently wrong otherwise.
        one_body_tensor: One-electron integrals, shape ``(norb, norb)``.
        two_body_tensor: Two-electron integrals, chemist notation, ``(norb,)*4``.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        n_roots: Number of lowest roots to solve for. Clamped to the subspace
            dimension when the batch is too small.
        spin_sq: Target ``S^2`` passed to ``fix_spin_``. ``None`` imposes no
            spin, which is the right choice if you want the n lowest states
            regardless of multiplicity.
        spin_shift: Penalty strength for ``fix_spin_``. The PySCF default of
            0.2 Ha is only adequate for a ground-state calculation; increase
            it if wrong-spin states appear in the returned roots.
        max_cycle: Flat Davidson iteration budget — the total number of matvec
            cycles across *all* restarts, since PySCF's Davidson is a single flat
            loop whose restart is an internal flag (triggered when the subspace
            outgrows the block size), not an outer loop. This is why it is kept
            distinct from the nested solvers' ``max_davidson_iters``: as a rough
            equivalence, ``max_cycle ~= max_davidson_iters * davidson_block_size``.
        davidson_block_size: Davidson subspace size — vectors accumulated before
            a restart. Same meaning as on the nested-loop solvers
            (``TorchSCISolver``, ``SBDSolver``), which is *not* how PySCF's own
            ``max_space`` behaves: PySCF widens whatever it is given by
            ``(nroots-1)*4``, so the widening is subtracted before the value is
            handed over, making the effective subspace equal what was requested.
            ``None`` (default) selects ``6*n_roots + 8`` — PySCF's historical
            ``12 + 2*n_roots`` default re-expressed in this post-widening unit, so
            the value reaching PySCF is unchanged. Raises if too small to survive
            the subtraction (``<= (nroots-1)*4``).
        with_rdm2: Whether to retain the 2-RDM in each result. It is always
            built, since the energy expression needs it, but holding it costs
            ``n_batches * n_roots * norb**4`` doubles.
        **kwargs: Forwarded verbatim to ``kernel_fixed_space``.

    Returns:
        One :class:`~sci_solvers.eigen_states.EigenStates` per batch, each
        holding up to ``n_roots``
        :class:`~sci_solvers.eigen_states.SCIResultWithConvergence` instances
        in ascending energy order. Each root's ``state`` is ``"converged"``,
        ``"not_converged"``, or ``None``. Energies are electronic only.
    """
    if n_roots < 1:
        raise ValueError(f"n_roots must be >= 1, got {n_roots}.")
    max_space = davidson_block_size_to_max_space(davidson_block_size, n_roots)

    results: List[EigenStates] = []
    for batch_idx, ci_strs in enumerate(ci_strings):
        # _get_init_guess yields at most na*nb guesses, so asking for more
        # roots than the subspace holds fails inside Davidson. Small batches
        # are routine in early configuration-recovery iterations.
        dim = len(ci_strs[0]) * len(ci_strs[1])
        n_solve = min(n_roots, dim)

        myci = fci.selected_ci.SelectedCI()
        if spin_sq is not None:
            myci = fci.addons.fix_spin_(myci, ss=spin_sq, shift=spin_shift)

        # Discard the raw eigenvalues: when fix_spin_ is active they carry the
        # penalty shift. Energy is recomputed from the RDMs below, matching
        # what solve_sci does for the same reason.
        _, sci_vecs = fci.selected_ci.kernel_fixed_space(
            myci,
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
            ci_strs=ci_strs,
            nroots=n_solve,
            max_cycle=max_cycle,
            max_space=max_space,
            **kwargs,
        )
        # With nroots == 1 PySCF returns a bare (na, nb) array rather than a
        # list; iterating it would walk the alpha rows and silently produce
        # nonsense RDMs and energies.
        if n_solve == 1:
            sci_vecs = [sci_vecs]

        # Read the per-root convergence flags.  kernel_fixed_space does not
        # return them directly, but PySCF writes them onto myci after the
        # Davidson finishes — we own myci here, unlike the n_roots==1 path.
        # converged_flags is a bool array of shape (nroots,); True when the
        # residual of that root fell below tol.
        converged_flags = np.atleast_1d(np.asarray(getattr(myci, "converged", True), dtype=bool))
        if not np.all(converged_flags):
            warnings.warn(
                f"Batch {batch_idx}: Davidson did not converge for roots "
                f"{np.flatnonzero(~converged_flags).tolist()} after {max_cycle} cycles.",
                stacklevel=2,
            )
        if len(sci_vecs) < n_roots:
            warnings.warn(
                f"Batch {batch_idx}: got {len(sci_vecs)} of {n_roots} requested "
                f"roots (subspace dimension {dim}).",
                stacklevel=2,
            )

        # Built once per batch instead of once per RDM call: des_des_linkstr
        # does Python-level sorted(set(...)) over nstrs*nelec int64s, which is
        # not cheap at scale.
        link_index = _rdm_link_index(ci_strs, norb, nelec)

        roots: List[SCIResultWithConvergence] = []
        # TODO: if n_roots == 1 and spin_sq is not None, it is possible to use
        # the energy from kernel_fixed_space directly without computing dm2.
        # The reason we drop the raw eigenvalue is that fix_spin_ biases the
        # Hamiltonian: H' = H + λ(S² − ss), so the eigenvalue is not the true
        # energy. However dm2 is expensive (O(norb^4)); a future optimisation
        # could skip it in the single-root spin-projected case by computing
        # energy from dm1 + a cheaper correction. For n_roots > 1 the full RDM
        # path below is required.
        for root_idx, sci_vec in enumerate(sci_vecs):
            # make_rdm1 just calls make_rdm1s and sums, so build once.
            dm1a, dm1b = myci.make_rdm1s(sci_vec, norb, nelec, link_index)
            occupancies = (np.diagonal(dm1a), np.diagonal(dm1b))
            dm1 = dm1a + dm1b
            # The energy needs dm2 regardless; with_rdm2 only controls whether
            # it is retained in the result.
            dm2 = myci.make_rdm2(sci_vec, norb, nelec, link_index)
            energy = np.einsum("pr,pr->", dm1, one_body_tensor) + 0.5 * np.einsum(
                "prqs,prqs->", dm2, two_body_tensor
            )

            if spin_sq is not None:
                ss, _ = myci.spin_square(sci_vec, norb, nelec)
                if abs(ss - spin_sq) > 0.1:
                    warnings.warn(
                        f"Batch {batch_idx}, root {root_idx}: <S^2> = {ss:.4f} "
                        f"but target is {spin_sq}. The spin penalty "
                        f"(shift={spin_shift}) was too weak to expel this "
                        f"state; the returned spectrum is neither the n lowest "
                        f"roots nor the n lowest roots of the target sector.",
                        stacklevel=2,
                    )

            sci_state = SCIState(
                amplitudes=np.array(sci_vec),
                ci_strs_a=sci_vec._strs[0],
                ci_strs_b=sci_vec._strs[1],
                norb=norb,
                nelec=nelec,
            )
            # Per-root convergence: index into the flags array when available,
            # fall back to None if the array is shorter than expected.
            if root_idx < len(converged_flags):
                root_converged = bool(converged_flags[root_idx])
                root_state: str | None = "converged" if root_converged else "not_converged"
            else:
                root_state = None
            roots.append(
                SCIResultWithConvergence(
                    float(energy),
                    sci_state,
                    orbital_occupancies=occupancies,
                    rdm1=dm1,
                    rdm2=dm2 if with_rdm2 else None,
                    state=root_state,
                )
            )

        # kernel_fixed_space returns roots in ascending energy order already,
        # but sort explicitly to match the EigenStates contract.
        roots.sort(key=lambda r: r.energy)
        results.append(EigenStates(states=roots))

    return results


class PySCFSolver(EnergyCalculatorBase):
    """
    sci_solver wrapping PySCF's selected-CI kernel.

    Suitable for small-to-medium subspaces on CPU. Multithreaded via BLAS;
    supports spin projection via ``spin_sq``.

    Multi-root support: pass ``n_roots > 1`` to solve for excited states in
    addition to the ground state. When ``n_roots == 1`` (default) the solver
    uses :meth:`_solve_one_with_convergence` per batch (capturing PySCF's
    Davidson ``converged`` flag). When ``n_roots > 1`` it calls
    ``kernel_fixed_space`` directly via :func:`solve_sci_batch_multiroot`.
    All results are :class:`~sci_solvers.eigen_states.SCIResultWithConvergence`
    — read the convergence state from ``result.state``, and excited states from
    ``states[1:]`` of each per-batch
    :class:`~sci_solvers.eigen_states.EigenStates`.
    """

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        nroots: int = 1,
        n_roots: int = 1,
        spin_shift: float = 0.2,
        with_rdm2: bool = True,
        **kwargs: Any,
    ):
        """Initialize the PySCFSolver.

        Args:
            symmetrize_spin: Passed to the base class.
            hamiltonian: Optional bound Hamiltonian (see base class).
            verbose: Print solver progress.
            nroots: Number of lowest eigenpairs to solve for per batch. Default
                ``1`` (ground state only, backward-compatible). When ``> 1``, each
                per-batch :class:`~sci_solvers.eigen_states.EigenStates` holds one
                :class:`~sci_solvers.eigen_states.SCIResultWithConvergence` per
                root, ascending in energy — read excited states from ``states[1:]``.
            n_roots: Backward-compatible alias for ``nroots``. Pass one or the
                other, not both (a non-default value on each raises).
            spin_shift: Penalty strength forwarded to ``fix_spin_`` when
                ``spin_sq`` is set at call time. More significant for
                ``n_roots > 1`` (needs to push all wrong-spin states above
                all target-spin roots), but accepted on both paths.
            with_rdm2: Whether to retain the 2-RDM in each result. Only
                relevant for ``n_roots > 1``; the single-root path (via
                ``solve_sci``) always populates ``rdm2``.
        """
        super().__init__(
            hamiltonian=hamiltonian,
            symmetrize_spin=symmetrize_spin,
            verbose=verbose,
            nroots=reconcile_nroots(nroots, n_roots),
            **kwargs,
        )
        self.spin_shift = spin_shift
        self.with_rdm2 = with_rdm2

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        *,
        spin_sq: float | None = None,
        max_cycle: int = 200,
        davidson_block_size: int | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Diagonalize each batch via PySCF's selected_ci kernel.

        When ``n_roots == 1`` calls :meth:`_solve_one_with_convergence` per
        batch, which captures PySCF's Davidson ``converged`` flag.
        When ``n_roots > 1`` calls ``kernel_fixed_space`` directly via
        :func:`solve_sci_batch_multiroot`.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals, shape ``(norb,)*4``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``. A scalar ``nelec`` (the spinless
                flat spin-orbital path) is rejected with ``NotImplementedError``
                — this solver does not implement it; see
                :func:`~sci_solvers.energy_calculation_base.reject_spinless_nelec`.
            spin_sq: Target value for total spin squared (S²).
            max_cycle: Flat Davidson iteration budget — total matvec cycles
                across all restarts (PySCF's Davidson is one flat loop; a
                restart is an internal flag, not an outer loop). Kept distinct
                from the nested solvers' ``max_davidson_iters``, for which
                ``max_cycle ~= max_davidson_iters * davidson_block_size``.
            davidson_block_size: Davidson subspace size — vectors accumulated
                before a restart, carrying the same meaning as on
                ``TorchSCISolver`` / ``SBDSolver``. PySCF widens its own
                ``max_space`` by ``(nroots-1)*4``, so that widening is
                subtracted here and the effective subspace equals what you ask
                for. ``None`` (default) selects ``6*nroots + 8``, i.e. PySCF's
                historical ``12 + 2*nroots`` restated in this unit, leaving the
                value PySCF receives unchanged. Pass this instead of
                ``max_space``, which is rejected.
            **kwargs: Additional kwargs forwarded to the underlying PySCF
                call — e.g. ``tol``, ``lindep``, ``max_memory``.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in
            input order. Each holds one
            :class:`~sci_solvers.eigen_states.SCIResultWithConvergence` per
            solved root, ascending in energy: ``states[0]`` is the ground
            state and ``states[1:]`` are the excited states (present only when
            ``n_roots > 1``). Energies are electronic only.
        """
        reject_init_vectors(type(self).__name__, kwargs)
        # Rejected rather than forwarded: max_space is PySCF's pre-widening
        # spelling of the same knob, so accepting both would let two different
        # subspace sizes be requested at once, and a bare max_space silently means
        # something different here than davidson_block_size does elsewhere.
        if "max_space" in kwargs:
            raise TypeError(
                "max_space is not accepted; pass davidson_block_size instead. It is "
                "the same knob expressed as the subspace actually restarted at "
                "(PySCF widens max_space by (nroots-1)*4 internally), so it carries "
                "the same meaning here as on TorchSCISolver and SBDSolver."
            )
        if self.verbose:
            print(
                f"[PySCF] hcore.shape={hcore.shape}, eri.shape={eri.shape}, "
                f"nroots={self.nroots}, "
                f"len(batches)={len(ci_strs_batches)}, "
                f"per-batch sizes={[(len(b[0]), len(b[1])) for b in ci_strs_batches]}"
            )
        super().sci_solver_callable_eigen_states(
            ci_strs_batches, hcore, eri, norb, nelec, spin_sq=spin_sq, max_cycle=max_cycle
        )
        reject_spinless_nelec(type(self).__name__, nelec)
        # Spatial path from here on (the scalar-nelec spinless case raised above).
        nelec = cast(Tuple[int, int], nelec)
        ci_strs_batches = cast(List[Tuple[np.ndarray, np.ndarray]], ci_strs_batches)

        if self.nroots == 1:
            # Single-root path goes through qiskit-addon-sqd's solve_sci, which
            # splats kwargs into kernel_fixed_space, so translate to PySCF's own
            # max_space here (nroots == 1 makes the widening zero, but route it
            # through the shared helper so the validation still applies).
            max_space = davidson_block_size_to_max_space(davidson_block_size, self.nroots)
            results = [
                self._solve_one_with_convergence(
                    ci_strs,
                    hcore,
                    eri,
                    norb=norb,
                    nelec=nelec,
                    spin_sq=spin_sq,
                    max_cycle=max_cycle,
                    max_space=max_space,
                    **kwargs,
                )
                for ci_strs in ci_strs_batches
            ]
            return [EigenStates(states=[sci_result]) for sci_result in results]

        return solve_sci_batch_multiroot(
            ci_strings=ci_strs_batches,
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=norb,
            nelec=nelec,
            n_roots=self.nroots,
            spin_sq=spin_sq,
            spin_shift=self.spin_shift,
            max_cycle=max_cycle,
            davidson_block_size=davidson_block_size,
            with_rdm2=self.with_rdm2,
            **kwargs,
        )

    def _solve_one_with_convergence(
        self,
        ci_strs: Tuple[np.ndarray, np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        *,
        norb: int,
        nelec: Tuple[int, int],
        spin_sq: float | None,
        max_cycle: int,
        **kwargs: Any,
    ) -> SCIResultWithConvergence:
        """Diagonalize one batch, capturing PySCF's Davidson ``converged`` flag.

        PySCF *does* know whether its Davidson converged: ``lib.davidson1`` sets
        a ``converged`` attribute on the ``SelectedCI`` instance that drives the
        solve. The problem is reaching it. ``solve_sci`` constructs that instance
        internally (``myci = fci.selected_ci.SelectedCI()``) and returns only an
        ``SCIResult``, so the object holding the flag is discarded before we ever
        see it.

        To get a reference, we temporarily replace the ``SelectedCI`` attribute on
        the ``pyscf.fci.selected_ci`` module with a subclass whose ``__init__``
        records every instance it creates. ``solve_sci`` looks that attribute up at
        call time, so during our call it instantiates the subclass and we end up
        holding the very object PySCF writes ``converged`` onto. ``fix_spin_``
        (used when ``spin_sq`` is not None) decorates its argument in place rather
        than returning a new object, so the captured instance is still the right
        one on the spin-projected path.

        The patch is process-global while installed, which is why it is scoped as
        tightly as possible: installed immediately before the call, removed in a
        ``finally`` so an exception from ``solve_sci`` cannot leave the subclass
        permanently bound in PySCF's module namespace. Batches run sequentially
        here, so there is no concurrent-call race to worry about.

        Returns:
            The batch's result as an :class:`SCIResultWithConvergence`. ``state``
            is ``"converged"`` / ``"not_converged"`` from PySCF's flag, or ``None``
            if no flag could be recovered (e.g. a future upstream refactor stops
            routing through ``SelectedCI``).
        """
        from pyscf.fci import selected_ci

        captured: list[Any] = []

        # Save the real class BEFORE overwriting the module attribute; this is
        # what `finally` restores. It is a saved pointer, not a live alias, so
        # after the assignment below the two names refer to different classes.
        original_cls = selected_ci.SelectedCI

        class _CapturingSelectedCI(original_cls):  # type: ignore[valid-type, misc]
            """``SelectedCI`` that appends each new instance to ``captured``."""

            def __init__(self, *args: Any, **kw: Any):
                super().__init__(*args, **kw)
                captured.append(self)

        selected_ci.SelectedCI = _CapturingSelectedCI
        try:
            sci_result = solve_sci(
                ci_strs,
                hcore,
                eri,
                norb=norb,
                nelec=nelec,
                spin_sq=spin_sq,
                max_cycle=max_cycle,
                **kwargs,
            )
        finally:
            # Runs on the exception path too. Without this, one failed batch
            # would leave our subclass installed for the rest of the process,
            # capturing unrelated SelectedCI instances into a list that never
            # drains.
            selected_ci.SelectedCI = original_cls

        # `solve_sci` may build more than one instance, so take the first that
        # actually carries a flag rather than assuming captured[0]. `converged`
        # is per-root, so np.all collapses a multi-root array to a single verdict
        # (this solver is single-root, but the flag's shape is not guaranteed).
        converged: bool | None = None
        for instance in captured:
            flag = getattr(instance, "converged", None)
            if flag is not None:
                converged = bool(np.all(flag))
                break

        if converged is None:
            state = None
        elif converged:
            state = "converged"
        else:
            state = "not_converged"
            warnings.warn(
                f"pyscf Davidson did not converge within max_cycle={max_cycle}. "
                "The energy may be unreliable; try raising max_cycle."
            )

        # SCIResult is frozen, so rebuild rather than mutate. Copy every field
        # through as-is (rdm1/rdm2, and `carryover` on qiskit-addon-sqd >= 0.14),
        # so a field upstream adds is not silently dropped by a hand-kept list.
        return SCIResultWithConvergence(
            **{f.name: getattr(sci_result, f.name) for f in fields(sci_result)},
            state=state,
        )


class PySCFSolverSpinFilterMixin(SpinFilterMixin, PySCFSolver):
    """:class:`PySCFSolver` with post-hoc ``<S^2>`` projection via the mixin.

    An alternative to PySCF's *native* ``spin_sq`` projection (``fix_spin_``):
    rather than penalizing spin during the solve, this filters the returned
    root(s) by ``<S^2>``, using the same composable mechanism as the other
    solvers. PySCF's root count is fixed at construction (``n_roots``), so the
    mixin does not oversample — it filters the roots PySCF returns and raises if
    fewer than ``n_roots`` fall in the target sector. For a sector buried deep in
    the spectrum, prefer PySCF's native path (pass ``spin_sq`` to
    ``sci_solver_callable_eigen_states``) or
    :class:`~sci_solvers.energy_solvers.general_algebra_solvers.PyscfDavidsonSolver`.
    """
