"""
torch_sci_solver-based sci_solver implementation.

Wraps ``torch_sci_solver``'s matvec + Davidson + RDM stack (via
``make_torch_sci_solver``) in this package's :class:`EnergyCalculatorBase`
contract. Runs on GPU by default (``device="cuda"``) and falls back to CPU;
suited to large active spaces where the torch batched matvec is fastest.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
from dataclasses import replace
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from sci_solvers.eigen_states import EigenStates
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    is_spinless_request,
    reconcile_nroots,
)
from sci_solvers.hamiltonian import HamiltonianData


class TorchSCISolver(EnergyCalculatorBase):
    """
    torch_sci based - a torch implementation of SCI, uses GPU by default

    Multi-root support: pass ``n_roots > 1`` to solve for excited states in
    addition to the ground state. Each per-batch
    :class:`~sci_solvers.eigen_states.EigenStates` then holds ``states[0]``
    (ground state) through ``states[n_roots - 1]`` (highest solved root),
    ascending in energy — read excited states from ``states[1:]``. The
    qiskit-addon-sqd recovery loop (driven via
    :meth:`~sci_solvers.energy_calculation_base.EnergyCalculatorBase.sci_solver_callable`)
    still consumes only the ground state, so raising ``n_roots`` is free for the
    hot loop and surfaces excited states only to callers that read them. This
    mirrors the general-algebra solvers' ``nroots`` (named ``n_roots`` here to
    match the underlying ``torch_sci_solver`` package).
    """

    complex_capable = True  # dtype=torch.complex128 opt-in; see __init__'s dtype arg
    supports_spinless = True  # native flat spin-orbital path; scalar nelec, multi-root, no spin_sq
    supports_init_vectors = True  # per-call Davidson start vector, every mode

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        device: torch.device | str = "cuda",
        devices: Optional[List[torch.device | str]] = None,
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
        chunk_size: int = 2_000_000,
        davidson_block_size: Optional[int] = 20,  # None,
        max_davidson_iters: int = 20,
        davidson_tolerance: float = 1e-4,  # 1e-7,
        apply_ab_projection: bool = False,
        nroots: int = 1,
        n_roots: int = 1,
        **solver_kwargs: Any,
    ):
        """Initialize the TorchSCISolver.

        Default parameters are set to match exact energy as sbd tpb app (https://github.com/r-ccs-cms/sbd/tree/main/apps/chemistry_tpb_selected_basis_diagonalization)

        Args:
            device: Single torch device for the diagonalization (default
                ``"cuda"``). Ignored when ``devices`` is given.
            devices: Opt into single-node, multi-GPU **batch-parallel**
                dispatch by passing a list of CUDA devices, e.g.
                ``["cuda:0", "cuda:1"]``. Each SQD subspace (batch) is then
                diagonalized on its own GPU via a thread-pool fan-out. Speedup
                is bounded by ``min(num_batches, len(devices))``: it is near-
                linear only when ``num_batches >= len(devices)`` and roughly
                divisible by it (an awkward remainder leaves GPUs idle in the
                final wave). ``None`` (default) keeps single-device behaviour.
                When set, ``devices[0]`` becomes the primary device and the
                ``device`` argument is not forwarded.
            dtype: Numeric dtype for the diagonalization (default
                ``torch.float64``). Pass ``torch.complex128`` (or
                ``complex64``) to solve a complex Hermitian Hamiltonian —
                opt-in only: passing complex ``hcore``/``eri`` to
                :meth:`sci_solver_callable_eigen_states` while this stays at
                its real default raises rather than silently dropping the
                imaginary part. Complex dtype forces the underlying matvec
                onto the portable ``"batched"`` backend even on
                ``device="cuda"`` (the Triton kernel is real-only); see
                ``torch_sci_solver.torch_impl.qiskit_adapter.TorchSCISolver``'s
                ``matvec_impl`` docs.
            davidson_block_size: Davidson subspace size — vectors accumulated
                before a restart (default ``20``). Total matvec work is roughly
                ``max_davidson_iters * davidson_block_size``, since this solver's
                Davidson is genuinely nested (an outer restart loop over an inner
                subspace-expansion loop). Must be ``>= n_roots``; ``None``
                auto-tunes it.
            max_davidson_iters: Outer Davidson restart count per diagonalization
                (default ``20``). Note this is a *restart* count, not the flat
                per-matvec budget that ``max_cycle`` denotes on the PySCF-family
                solvers — the two are not interchangeable, which is why this
                solver does not accept ``max_cycle``. Override per call via
                :meth:`sci_solver_callable_eigen_states`'s own
                ``max_davidson_iters``.
            davidson_tolerance: Davidson residual (``‖R‖``) convergence
                threshold (default ``1e-4``).
            nroots: Number of lowest eigenpairs (roots) to solve for per batch
                (canonical spelling, matching the general-algebra solvers).
                Default ``1`` (ground state only).
            n_roots: Backward-compatible alias for ``nroots``. Pass one or the
                other, not both (a non-default value on each raises). When
                ``> 1``, Davidson extracts the ``nroots`` lowest Ritz pairs and
                each solved batch's
                :class:`~sci_solvers.eigen_states.EigenStates` holds one
                :class:`SCIResult` per root, ascending in energy — read excited
                states from ``states[1:]``. Must not exceed
                ``davidson_block_size`` (auto-tuned when that is ``None``).
        """
        from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

        # Backwards-compatible alias: older callers spell the full-RDM request
        # ``compute_rdm2``. Translate it to the engine's ``compute_full_rdm``
        # rather than letting the engine ignore it as an unknown constructor
        # keyword.
        if "compute_rdm2" in solver_kwargs:
            if "compute_full_rdm" in solver_kwargs:
                raise ValueError(
                    "pass either compute_rdm2 or compute_full_rdm, not both; "
                    "compute_rdm2 is the deprecated alias of compute_full_rdm."
                )
            solver_kwargs["compute_full_rdm"] = bool(solver_kwargs.pop("compute_rdm2"))

        # Base reconciles the nroots/n_roots spellings, validates, and stores
        # self.nroots (used for the engine below).
        super().__init__(
            hamiltonian=hamiltonian,
            symmetrize_spin=symmetrize_spin,
            verbose=verbose,
            nroots=reconcile_nroots(nroots, n_roots),
        )

        # `devices` (multi-GPU) takes precedence over the single `device`.
        # The engine rebinds its primary device to devices[0], so forwarding
        # a stale `device=` alongside would be misleading — pass one or the
        # other, matching make_torch_sci_solver's own contract.
        device_kwargs: dict[str, Any] = (
            {"devices": devices} if devices is not None else {"device": device}
        )

        self._solver = make_torch_sci_solver(
            **device_kwargs,
            dtype=dtype,
            verbose=verbose,
            chunk_size=chunk_size,
            davidson_block_size=davidson_block_size,
            max_davidson_iters=max_davidson_iters,
            davidson_tolerance=davidson_tolerance,
            apply_ab_projection=apply_ab_projection,
            n_roots=self.nroots,
            **solver_kwargs,
        )
        # The engine owns the basis layout ("tpb" product vs "gdb" parallel pair
        # lists); mirror it so compute_energy does not sort a pair list per array.
        self.mode = getattr(self._solver, "mode", "tpb")

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        *,
        spin_sq: float | None = None,
        max_davidson_iters: int | None = None,
        init_vectors: List[np.ndarray | None] | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Diagonalize each batch via our homegrown torch_sci implementation.

        Positional signature matches ``qiskit_addon_sqd.fermion.solve_sci_batch``
        (``ci_strings, one_body, two_body, norb, nelec`` plus ``spin_sq``); the
        remaining keyword knobs are this solver's own and are **not** PySCF's —
        the torch engine runs its own Davidson, nothing routes to
        ``kernel_fixed_space``.

        Two reduction paths are dispatched on ``nelec``'s type, mirroring
        ``GeneralAlgebraSolverBase``:

        * **spatial** — ``nelec`` an ``(n_a, n_b)`` tuple: ``ci_strs_batches``
          is a list of ``(strs_a, strs_b)`` tuples over separate alpha/beta
          determinant lists.
        * **spinless** (flat spin-orbital, no alpha/beta split) — ``nelec`` a
          *scalar* total particle number ``N``: ``ci_strs_batches`` is a list
          of flat ``strs`` arrays. Complex-only (mirrors
          ``sci_solvers.spinless_ci_problem``). Multi-root (``n_roots > 1``
          at construction) works at the same level as the spatial path
          (``davidson_torch`` is basis-shape-agnostic); ``spin_sq`` is
          rejected (no alpha/beta split to target a spin sector).

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples
                (spatial), or a list of flat ``strs`` int64 arrays (spinless
                — see above).
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals, shape ``(norb,)*4``.
            norb: Number of spatial orbitals (spin-orbitals ``nmo`` on the
                spinless path).
            nelec: ``(n_alpha, n_beta)`` (spatial), or a scalar total ``N``
                (spinless — see above).
            spin_sq: Target value for total spin squared (S²). Spatial path
                only; rejected on the spinless path.
            shift: Strength of the spin-penalty shift (only effective
                when ``spin_sq`` is set).
            max_davidson_iters: Outer Davidson restart count for this call.
                Defaults to the solver's own ``max_davidson_iters`` (set at
                construction) when omitted — pass it only to override for a
                single call. This solver deliberately does **not** accept
                ``max_cycle``: that name denotes a flat total-matvec budget on
                the PySCF-family solvers, whereas this Davidson is nested
                (total work ~ ``max_davidson_iters * davidson_block_size``), so
                the same value would buy very different amounts of work.
            init_vectors: Davidson start vector per batch, to warm start an
                iteration from the previous one's state projected onto the new
                strings (the projection is the caller's job). One entry per
                batch, each ``None`` (cold start) or a 1-D array laid out like
                the ``sci_state.amplitudes`` this solver returns in that mode and
                indexed by the strings passed in this call:

                * TPB: ``(len(strs_a) * len(strs_b),)``, ``ia * len(strs_b) + ib``,
                  spin-block signs;
                * GDB (``mode="gdb"``): ``(len(strs_a),)``, one entry per
                  ``(strs_a[i], strs_b[i])`` pair, interleaved-Wick signs;
                * spinless: ``(len(strs),)``.

                The solver reorders it to its internal basis (strings are sorted
                and deduplicated; a duplicate takes its first occurrence's
                amplitude, a string not passed starts at 0), applies the αβ
                projection, and normalizes it. If less than 1e-8 of its own
                norm is left after that, it falls back to the cold start (so its
                overall scale does not matter). Whether each batch was warm
                started is reported as ``warm`` by
                :meth:`last_call_subspace_stats` (TPB and GDB). Raises ``ValueError``
                on a length mismatch or together with ``probe_state``.
            **kwargs: Additional kwargs forwarded to the underlying
                ``torch_sci_solver`` engine — e.g. ``tol`` (per-call Davidson
                residual threshold, overriding ``davidson_tolerance``),
                ``spin_penalty_lambda``.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in
            input order, each holding one :class:`SCIResult` per solved root,
            ascending in energy: ``states[0]`` is the ground state and
            ``states[1:]`` are the excited states (present only when the solver
            was constructed with ``n_roots > 1``; a default ``n_roots == 1``
            solver returns an ``EigenStates`` of length one). Energies are
            electronic only.
        """
        # complex_capable=True, but only when this solver was itself built
        # with dtype=torch.complex128/64 (see __init__'s dtype arg) — the
        # underlying engine's own dtype is fixed at construction, so a
        # complex Hamiltonian handed to a solver built with the real
        # default cannot be honored. Raise here with the same
        # "list what would work" shape as
        # GeneralAlgebraSolverBase.sci_solver_callable_eigen_states, rather
        # than letting the mismatch surface as an unexplained TypeError deep
        # inside the vendored engine's __call__.
        is_complex = np.iscomplexobj(hcore) or np.iscomplexobj(eri)
        if is_complex and not self._solver.dtype.is_complex:
            from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, is_complex_capable

            capable = ", ".join(repr(n) for n in AVAILABLE_SOLVERS if is_complex_capable(n))
            raise NotImplementedError(
                f"{type(self).__name__} was constructed with dtype="
                f"{self._solver.dtype} (real) but hcore/eri is complex. Pass "
                "dtype=torch.complex128 to TorchSCISolver(...) to solve a "
                f"complex Hermitian Hamiltonian. Complex-capable solvers: {capable}."
            )

        # Generic solver-facing spellings: ``maxiter`` names the same outer
        # Davidson restart count as ``max_davidson_iters`` (``tol`` is already
        # forwarded to the engine unchanged). Reject both at once instead of
        # silently letting one win.
        if "maxiter" in kwargs:
            if max_davidson_iters is not None:
                raise ValueError(
                    "pass either maxiter or max_davidson_iters, not both; "
                    "maxiter is the generic alias of max_davidson_iters."
                )
            max_davidson_iters = int(kwargs.pop("maxiter"))
        # Resolved here, not left to the engine's own default, so the
        # non-convergence warning below can report the count actually used.
        if max_davidson_iters is None:
            max_davidson_iters = self._solver.max_davidson_iters
        if self.verbose:
            if is_spinless_request(nelec):
                sizes_desc = (
                    f"per-batch sizes (spinless, N={nelec})={[len(b) for b in ci_strs_batches]}"
                )
            else:
                sizes_desc = f"per-batch sizes={[(len(b[0]), len(b[1])) for b in ci_strs_batches]}"
            print(
                f"[TorchSCISolver] hcore.shape={hcore.shape}, eri.shape={eri.shape}, "
                f"len(batches)={len(ci_strs_batches)}, {sizes_desc}"
            )
        super().sci_solver_callable_eigen_states(
            ci_strs_batches,
            hcore,
            eri,
            norb,
            nelec,
            spin_sq=spin_sq,
            max_davidson_iters=max_davidson_iters,
            **kwargs,
        )
        # inner solver now returns list[torch EigenStates], one per subspace,
        # each holding all solved roots directly in .states.
        # Cleared first, set only once the engine returns: a call that raises, or a
        # spinless one (which writes no log entry), then reports no stats instead of
        # an earlier call's.
        self._last_call_log_entry = None
        self._last_probe_states = None
        log_len = len(self._solver.stats.get("per_call_log", []))
        ans = self._solver(
            ci_strings=ci_strs_batches,
            one_body_tensor=hcore,
            two_body_tensor=eri,
            norb=norb,
            nelec=nelec,
            spin_sq=spin_sq,
            max_davidson_iters=max_davidson_iters,
            init_vectors=init_vectors,
            **kwargs,
        )
        # The engine returns ``(results, cached_states)`` when the caller asks
        # for a warm-start snapshot (``probe_method="warm_start"``); the state
        # list is a side channel, so keep the result list as ``ans``.
        if isinstance(ans, tuple) and len(ans) == 2:
            ans, self._last_probe_states = ans
        log = self._solver.stats.get("per_call_log", [])
        if len(log) > log_len:
            self._last_call_log_entry = log[-1]
        # Convergence check: inspect the ground-state root (states[0]) of each
        # subspace and overwrite its energy with -1.0 when not converged.
        for i, es in enumerate(ans):
            gs = es.states[0]
            if gs.state == "not_converged":
                logger.warning(
                    f"Did not converge after {max_davidson_iters} Davidson "
                    f"restarts, corrupted energy: {gs.energy}, overwriting to -1."
                )
                new_states = [replace(gs, energy=-1.0)] + list(es.states[1:])
                ans[i] = type(es)(states=new_states)

        # Re-wrap each torch EigenStates into the sci_solvers EigenStates type
        # so the return value matches the declared List[sci_solvers.EigenStates].
        return [EigenStates(states=list(es.states)) for es in ans]

    @property
    def config(self) -> dict[str, Any]:
        """JSON-friendly requested/resolved engine configuration.

        Delegates to the underlying ``torch_sci_solver`` engine so workflows do
        not need to reach into ``self._solver`` for the mode, device, dtype,
        distributed launch config or ``gdb_kwargs``.
        """
        return self._solver.config

    def capabilities(self) -> dict[str, Any]:
        """Per-mode engine support map (layouts, dtypes, RDM/S², distributed, ...)."""
        return self._solver.capabilities()

    def preflight(
        self,
        ci_strings: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        nelec: Tuple[int, int] | int,
        **call_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the engine's call-time gates and return the resolved plan, without solving.

        Args:
            ci_strings: The per-batch CI strings (same shapes as
                :meth:`sci_solver_callable_eigen_states`).
            nelec: ``(n_alpha, n_beta)`` or a scalar total particle number.
            **call_kwargs: The per-call keywords the solve would receive.
                ``norb=`` is strongly recommended (otherwise it is inferred
                from the highest set bit).

        Returns:
            The engine's JSON-friendly plan: effective basis/mode, per-subspace
            dimensions, ``matvec_impl``, tile grid/source, particle sector,
            warm/cold start and RDM path.
        """
        return self._solver.preflight(ci_strings, nelec, **call_kwargs)

    def last_call_subspace_stats(self) -> List[dict[str, Any]]:
        """Per-batch diagnostics of the most recent solve, one dict per batch, in order.

        The public way to confirm what a call actually did, without reaching into the
        engine: ``warm`` (whether that batch started from its ``init_vectors`` entry,
        ``False`` when it fell back to the cold start), ``n_matvecs``,
        ``davidson_dtype``, ``davidson_fp32_shift``, the resolved
        ``davidson_fp32_energy``, and the timings (``t_setup``, ``t_davidson``,
        ``t_occ``) and ``mem_peak_gb``. Empty before the first call, after a spinless
        call (that path keeps no per-batch log) and after a call that raised: never an
        earlier call's stats.
        """
        entry = getattr(self, "_last_call_log_entry", None)
        return [dict(s) for s in entry["subspaces"]] if entry is not None else []
