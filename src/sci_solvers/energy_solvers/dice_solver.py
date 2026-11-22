"""DICE-based sci_solver implementation."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, List, Tuple, cast

import numpy as np

from sci_solvers.eigen_states import EigenStates, SCIResultWithConvergence
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    reject_init_vectors,
    reject_spinless_nelec,
)
from sci_solvers.energy_solvers._dice_compat import resolve_dice_binary, solve_hci
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin
from sci_solvers.hamiltonian import HamiltonianData

# Dice prints a one-row-per-HCI-iteration table. Only the marker below tells us
# the variational stage ran to completion; there is no residual anywhere in the
# log, which is why this solver's signal is a proxy (see _parse_convergence).
_DICE_EXIT_MARKER = "Exiting variational iterations"
_DICE_ITER_HEADER = "Iter Root       Eps1"

# Row layout: Iter, Root, Eps1, #Var.Det., Energy, #Davidson, Time(s).
_DICE_ITER_ROW_NCOLS = 7
_DICE_NDAV_COL = 5

# Dice's Davidson iteration ceiling is a compile-time constant in its C++ source
# (800 per root), not something the wrapper exposes. Hitting it means Davidson
# stopped because it ran out of iterations rather than because it converged.
_DICE_DAVIDSON_CAP = 800


class DICESolver(EnergyCalculatorBase):
    """
    sci_solver wrapping the DICE (Dice is Cheap Eigensolver) C++ binary
    via ``qiskit-addon-dice-solver``'s :func:`solve_hci`.

    Performs a **single Davidson diagonalization** on the projected Hamiltonian of the input bitstrings (no HCI expansion). The defaults are set to enforce this fixed-space mode:

    * ``select_cutoff=2147483647`` (INT_MAX) prevents Dice from selecting any
      additional determinants beyond the supplied CI strings.
    * ``max_hci_iters=1`` means only one HCI macro-iteration is performed — a
      single Davidson solve on the projected subspace.
    * ``tol`` (HCI energy convergence, Dice's ``dE``) has no practical effect
      with ``max_hci_iters=1`` since convergence is never checked across
      iterations. Diagonalization quality is governed by
      ``davidson_tolerance`` (Dice's ``davidsonTol``, default ``1e-5``) and
      ``davidson_tolerance_loose``, both exposed by the wrapper.

    Multi-root: that single Davidson solve returns the lowest ``self.nroots``
    eigenpairs (``solve_hci``'s ``n_roots`` argument), so setting ``nroots > 1``
    yields the ground state plus excited states — the multi-root capability the
    eigensolver adapters previously exposed. ``spin_sq`` projection is still
    restricted to ``0.0``/``None`` (see below).

    Concurrent same-node ``mpirun`` invocations have been observed to crash
    Dice in this environment, so batches are processed sequentially. Use
    ``mpirun_options`` for intra-call MPI parallelism.

    Args:
        verbose: Print per-batch progress / debug output.
        mpirun_options: Options for MPI execution (e.g. ``"-n 16"``).
            Both string and list forms are accepted.
        temp_dir: Directory for temporary DICE files. Each batch gets
            its own subdirectory under this path.
        clean_temp_dir: Whether to clean up temporary files on success. The
            cleanup is performed by this class rather than by ``solve_hci``, so
            that Dice's logfile can be read for a convergence signal first.
        select_cutoff: Coefficient cutoff for HCI determinant selection.
            The default (INT_MAX) suppresses all selection so Dice diagonalizes
            exactly the supplied subspace without adding new configurations.
        tol: HCI energy convergence tolerance, forwarded to ``solve_hci``'s
            ``energy_tol`` (Dice's ``dE``). Note this is *not* a Davidson
            residual tolerance — see ``davidson_tolerance`` for that. Has no
            practical effect at the default ``max_hci_iters=1`` since no
            inter-iteration convergence check is performed.
        davidson_tolerance: Residual tolerance for Dice's *final* (tight)
            Davidson solve, forwarded as ``solve_hci``'s ``davidson_tol``
            (Dice's ``davidsonTol``; default ``1e-5``). At the default
            ``max_hci_iters=1`` this governs the quality of the returned
            energy. Dice caps its Davidson *iterations* at a compile-time
            constant that cannot be configured from here, so unlike the other
            solvers there is no ``max_davidson_iters``/``davidson_block_size``
            knob.
        davidson_tolerance_loose: Residual tolerance for the preliminary
            Davidson that runs on each HCI macro-iteration, forwarded as
            ``solve_hci``'s ``davidson_tol_loose`` (Dice's
            ``davidsonTolLoose``). ``None`` (default) omits the keyword so Dice
            uses its own default. The tight solve is warm-started from this
            one, so setting it tighter than ``davidson_tolerance`` shifts the
            bulk of the work into the loose solve — whose iteration count Dice
            actually reports.
        max_hci_iters: Maximum HCI macro-iterations, forwarded to
            ``solve_hci``'s ``max_iter``. Defaults to ``1`` for a single
            fixed-space Davidson diagonalization. This is an HCI selection loop,
            unrelated to Davidson iterations on the other solvers.
        fci: If ``True``, passes ``ci_strs=None`` to ``solve_hci``, which
            starts the SHCI expansion from the Hartree-Fock determinant
            instead of the supplied bitstrings.
        nroots: Number of lowest roots (states) to solve for per batch,
            forwarded to ``solve_hci``'s ``n_roots``. ``1`` (default) returns
            the ground state only; ``>1`` returns the ground state plus that
            many excited states, ascending in energy.
    """

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        mpirun_options: str | list[str] | None = None,
        temp_dir: str | None = None,
        clean_temp_dir: bool = True,
        select_cutoff: float = 2147483647,
        tol: float = 1e-10,
        davidson_tolerance: float = 1e-5,
        davidson_tolerance_loose: float | None = None,
        max_hci_iters: int = 1,
        fci: bool = False,
        nroots: int = 1,
        **kwargs: Any,
    ):
        """Initialize the DICE HCI solver."""
        super().__init__(
            hamiltonian=hamiltonian,
            symmetrize_spin=symmetrize_spin,
            verbose=verbose,
            nroots=nroots,
            **kwargs,
        )

        # Validate the Dice executable up front. The wrapper package is always
        # installed (it ships the Python API without necessarily the binary), so
        # the relevant failure is a missing/non-executable `Dice`, not an import.
        # resolve_dice_binary() is the same resolution solve_hci uses.
        dice_binary = resolve_dice_binary()
        if not os.access(dice_binary, os.X_OK):
            raise FileNotFoundError(
                f"could not find an executable Dice binary at {dice_binary}. "
                "Set DICE_BINARY to a filepath, put `Dice` on PATH, or follow "
                "scripts/energy_solvers/README.md to build it."
            )
        self._dice_binary = dice_binary

        self.mpirun_options = mpirun_options
        self.temp_dir = temp_dir
        self.clean_temp_dir = clean_temp_dir
        self.select_cutoff = select_cutoff
        self.energy_tol = tol
        self.davidson_tolerance = davidson_tolerance
        self.davidson_tolerance_loose = davidson_tolerance_loose
        self.max_hci_iters = max_hci_iters
        self.fci = fci

    # ------------------------------------------------------------------ #
    # Per-batch worker
    # ------------------------------------------------------------------ #

    def _solve_one_batch(
        self,
        ci_strings: Tuple[np.ndarray, np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        batch_idx: int,
        n_batches: int,
    ) -> Tuple[List[SCIResultWithConvergence], float, float]:
        """
        Run one :func:`solve_hci` call for ``self.nroots`` roots.

        Returns ``(sci_results, convert_time, solve_time)`` where ``sci_results``
        holds one :class:`SCIResultWithConvergence` per root (ordered as Dice
        returns them, i.e. ascending energy).
        """
        strs_a, strs_b = ci_strings

        convert_start = time.time()
        strs_a = np.asarray(strs_a, dtype=np.int64)
        strs_b = np.asarray(strs_b, dtype=np.int64)
        convert_time = time.time() - convert_start

        # Each batch gets its own subdir to avoid collisions on retry / debug.
        base = self.temp_dir or tempfile.gettempdir()
        batch_dir = Path(base) / f"dice_batch_{os.getpid()}_{batch_idx}_{int(time.time() * 1e6)}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(
                f"\n  [DICE Batch {batch_idx + 1}/{n_batches}] "
                f"Processing {len(strs_a)} alpha, {len(strs_b)} beta strings "
                f"(product dim = {len(strs_a) * len(strs_b)})"
            )
            print(
                f"  [DICE Batch {batch_idx + 1}/{n_batches}] Dtype conversion: {convert_time:.4f}s"
            )
            print(f"  [DICE Batch {batch_idx + 1}/{n_batches}] temp_dir={batch_dir}")
            print(
                f"  [DICE Batch {batch_idx + 1}/{n_batches}] mpirun_options={self.mpirun_options}"
            )
            print(
                f"  [DICE Batch {batch_idx + 1}/{n_batches}] "
                f"select_cutoff={self.select_cutoff}, energy_tol={self.energy_tol}, "
                f"davidson_tolerance={self.davidson_tolerance}, "
                f"max_hci_iters={self.max_hci_iters}"
            )

        solve_start = time.time()
        energies, sci_states, occupancies = solve_hci(
            hcore,
            eri,
            norb=norb,
            nelec=nelec,
            n_roots=self.nroots,
            ci_strs=None if self.fci else (strs_a, strs_b),
            select_cutoff=self.select_cutoff,
            energy_tol=self.energy_tol,
            davidson_tol=self.davidson_tolerance,
            davidson_tol_loose=self.davidson_tolerance_loose,
            max_iter=self.max_hci_iters,
            mpirun_options=self.mpirun_options,
            temp_dir=str(batch_dir),
            # Always False, regardless of self.clean_temp_dir: the wrapper would
            # rmtree the dice_cli_files_* directory on its way out, taking the
            # logfile that carries the only convergence evidence with it. We read
            # the log first and then honor self.clean_temp_dir ourselves below.
            clean_temp_dir=False,
        )
        solve_time = time.time() - solve_start

        # solve_hci returns a bare (energy, sci_state, occupancies) triple for a
        # single root and parallel (energies, sci_states, occupancies) sequences
        # for n_roots > 1; normalize both to parallel lists of length nroots so
        # the per-root wrapping below is uniform. (Mirrors the old DiceAdapter's
        # _dice_solve_hci normalization.)
        if self.nroots == 1:
            energies, sci_states, occupancies = [energies], [sci_states], [occupancies]
        else:
            energies = list(energies)
            sci_states = list(sci_states)
            occupancies = list(occupancies)

        if self.verbose:
            print(
                f"  [DICE Batch {batch_idx + 1}/{n_batches}] "
                f"solve_hci (file I/O + subprocess): {solve_time:.4f}s, "
                f"nroots={self.nroots}, ground-state energy={energies[0]:.8f}"
            )

        # Read the log BEFORE any cleanup: it lives in the dice_cli_files_*
        # subdirectory that the block below removes.
        state = self._parse_convergence(batch_dir, batch_idx, n_batches)
        if state == "not_converged":
            warnings.warn(
                f"Dice batch {batch_idx + 1}/{n_batches} shows no evidence of a "
                "completed Davidson (missing the variational-exit marker, or the "
                f"iteration count reached Dice's {_DICE_DAVIDSON_CAP}-per-root "
                "ceiling). The energy may be unreliable. Note Dice prints no "
                "residual, so this signal is a proxy."
            )

        if self.clean_temp_dir:
            # rmtree rather than rmdir: we suppressed the wrapper's own cleanup
            # to keep the logfile readable, so dice_cli_files_* is still here.
            try:
                shutil.rmtree(batch_dir)
            except OSError:
                pass

        # The convergence signal is parsed once per batch (from the shared
        # logfile), so it applies to every root of this batch.
        return (
            [
                SCIResultWithConvergence(
                    energy=root_energy,
                    sci_state=root_state,
                    orbital_occupancies=root_occ,
                    state=state,
                )
                for root_energy, root_state, root_occ in zip(energies, sci_states, occupancies)
            ],
            convert_time,
            solve_time,
        )

    def _parse_convergence(self, batch_dir: Path, batch_idx: int, n_batches: int) -> str | None:
        """Infer whether Dice's diagonalization completed, from its log file.

        This is a **proxy, and a weaker one than the other solvers'.** Dice never
        prints a Davidson residual, so we cannot compare anything against a
        tolerance the way the sbd and pyscf paths do. What the log does offer:

        * ``Exiting variational iterations`` — the variational stage finished.
          Absent means Dice died partway through.
        * the ``#Davidson`` column — if it reached Dice's compile-time ceiling,
          Davidson stopped on iteration count rather than on convergence.

        So ``"converged"`` here means "no evidence of trouble", not "residual
        below tolerance". A run that quietly hit an internal limit while still
        completing the variational stage would read as converged.

        Note the log routinely ends in an MPI abort during the *RDM* stage, well
        after the energy is written; that is not a diagonalization failure and
        must not be reported as one, so only the two signals above are consulted.

        Returns:
            ``"converged"``, ``"not_converged"``, or ``None`` when the log is
            missing or unreadable (unknown, not failure).
        """
        logs = sorted(batch_dir.glob("dice_cli_files_*/dice_solver_logfile.log"))
        if not logs:
            if self.verbose:
                print(
                    f"  [DICE Batch {batch_idx + 1}/{n_batches}] "
                    "no logfile found; convergence unknown"
                )
            return None

        try:
            text = logs[-1].read_text(errors="replace")
        except OSError:
            return None

        if _DICE_EXIT_MARKER not in text:
            return "not_converged"

        # Start at the table header. This matters: Dice logs carry hundreds of
        # bare-integer lines above it (per-batch determinant counts), and reading
        # those as table rows would report a wildly inflated Davidson count.
        header_at = text.find(_DICE_ITER_HEADER)
        if header_at == -1:
            return None

        counts = []
        for line in text[header_at:].splitlines()[1:]:
            cols = line.split()
            if len(cols) != _DICE_ITER_ROW_NCOLS:
                continue
            try:
                counts.append(int(cols[_DICE_NDAV_COL]))
            except ValueError:
                continue  # 7-column line that isn't a data row

        if not counts:
            return None
        if max(counts) >= _DICE_DAVIDSON_CAP:
            return "not_converged"
        return "converged"

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        *,
        spin_sq: float | None = 0.0,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Matches :func:`qiskit_addon_sqd.fermion.solve_sci_batch` signature.

        Run HCI on each batch via DICE.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals, shape ``(norb,)*4``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``. A scalar ``nelec`` (the spinless
                flat spin-orbital path) is rejected with
                ``NotImplementedError`` — this solver does not implement it.
            spin_sq: Target value for total spin squared (S²). MUST be
                ``None`` or ``0.0``. Any other value raises
                :class:`ValueError`. See class docstring for rationale.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in
            input order, each holding the ``self.nroots`` lowest
            :class:`SCIResult`\\ s DICE computes for that batch (ascending
            energy). Energies are electronic only.

        Raises:
            ValueError: If ``spin_sq`` is neither ``None`` nor ``0.0``.
            TypeError: If ``max_cycle`` is passed (see below).
            NotImplementedError: If ``nelec`` is a scalar (the spinless path).
        """
        reject_init_vectors(type(self).__name__, kwargs)
        # Rejected rather than warned-and-ignored: Dice has no flat matvec
        # budget, and its Davidson iteration cap is a compile-time constant we
        # cannot configure. Accepting the kwarg silently let callers believe they
        # had bounded the solve. Use max_hci_iters (HCI macro-iterations) or
        # davidson_tolerance on the constructor instead.
        if "max_cycle" in kwargs:
            raise TypeError(
                "max_cycle is not accepted by DICESolver: Dice exposes no flat "
                "matvec budget, and its Davidson iteration cap is hardcoded at "
                "compile time. Use max_hci_iters (HCI macro-iterations) or "
                "davidson_tolerance on the constructor."
            )

        super().sci_solver_callable_eigen_states(
            ci_strs_batches, hcore, eri, norb, nelec, spin_sq=spin_sq, **kwargs
        )
        reject_spinless_nelec(type(self).__name__, nelec)
        # Spatial path from here on (the scalar-nelec spinless case raised above).
        nelec = cast(Tuple[int, int], nelec)
        ci_strs_batches = cast(List[Tuple[np.ndarray, np.ndarray]], ci_strs_batches)

        # Reject any spin_sq value that would silently give wrong results.
        # The Riken-branch Dice binary does NOT honor the spin keyword in its
        # single-Davidson configuration (verified empirically). The upstream
        # wrapper hardcodes spin_sq=0.0 in solve_fermion with the comment:
        # "Hard-code S^2 = 0 until other values are supported."
        if spin_sq is not None and spin_sq != 0.0:
            raise ValueError(
                f"spin_sq={spin_sq} requested, but the Riken-branch Dice "
                "binary used by qiskit-addon-dice-solver does NOT honor "
                "the spin keyword in its single-Davidson configuration "
                "(verified empirically). The upstream wrapper itself "
                "hardcodes spin_sq=0.0 in solve_fermion with the comment: "
                '"Hard-code S^2 = 0 until other values are supported." '
                "DICESolver only accepts spin_sq=0.0 or spin_sq=None. "
                "If you need a non-zero target spin, either ensure your "
                "input subspace already carries that spin character "
                "(e.g., bitstrings from a circuit prepared in the target "
                "sector), or use PySCFSolver which supports spin projection."
            )

        n_batches = len(ci_strs_batches)
        if n_batches == 0:
            return []

        if self.verbose:
            print(f"\n  [DICE] Dispatching {n_batches} batch(es) sequentially")

        results: List[EigenStates] = []
        total_convert_time = 0.0
        total_solve_time = 0.0
        wall_start = time.time()

        for i, batch in enumerate(ci_strs_batches):
            batch_states, c_t, s_t = self._solve_one_batch(
                batch,
                hcore,
                eri,
                norb,
                nelec,
                i,
                n_batches,
            )
            results.append(EigenStates(states=batch_states))
            total_convert_time += c_t
            total_solve_time += s_t

        wall_time = time.time() - wall_start

        if self.verbose:
            print(f"\n  [DICE Summary] Total batches: {n_batches}")
            print(f"  [DICE Summary] Total conversion time (sum): {total_convert_time:.4f}s")
            print(f"  [DICE Summary] Total solve_hci time (sum): {total_solve_time:.4f}s")
            print(f"  [DICE Summary] Wall-clock time: {wall_time:.4f}s")
            print(f"  [DICE Summary] Average per batch: {total_solve_time / n_batches:.4f}s")

        return results


class DICESolverSpinFilterMixin(SpinFilterMixin, DICESolver):
    """:class:`DICESolver` with composable spin projection.

    Wraps :class:`DICESolver` (the actual single-Davidson Dice solve) in
    :class:`~sci_solvers.energy_solvers.spin_filter_mixin.SpinFilterMixin`, so a
    ``spin_sq`` target filters the returned root(s) by ``<S^2>``.
    """
