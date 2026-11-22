"""A GPU-capable (PyTorch) selected-CI eigensolver for SQD workflows.

This module provides two entry points, both designed as drop-in replacements
for the corresponding functions in :mod:`qiskit_addon_sqd.fermion`:

* :func:`solve_fermion` — the per-batch solver used inside a manual
  self-consistent configuration-recovery loop. Same call signature and return
  shape as ``qiskit_addon_sqd.fermion.solve_fermion``.

* :func:`solve_sci_batch` — the ``sci_solver`` callback consumed by
  ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``. Same call
  signature and return type (``list[SCIResult]``) as
  ``qiskit_addon_sqd.fermion.solve_sci_batch``.

Both delegate to the vectorized PyTorch selected-CI engine in
:mod:`torch_sci_solver`, which runs the matvec, Davidson diagonalization, and
reduced-density-matrix construction on CPU or CUDA. The numerical core has been
validated bit-for-bit against a scalar Python reference and, for the RDMs,
against PySCF's FCI ``make_rdm1s`` / ``make_rdm12s`` to machine precision.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import pathlib
import time as _time
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import cast

import numpy as np
import torch
from qiskit_addon_sqd.fermion import SCIResult, SCIState
from torch import Tensor

from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    one_excite_batch,
    two_excite_batch,
)
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import PairSCIState, pair_basis_from_arrays
from torch_sci_solver.torch_impl.qiskit_adapter import EigenStates as _EigenStates
from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

# Diagnostic prints in the outer loop are on by default and can be
# muted with ``TORCH_SCI_PAIR_QUIET=1``. Every line goes to stdout with
# a flush — a batch scheduler captures each line as soon as it's emitted, which
# matters when a run is killed mid-iteration and we want to know how
# far it got.
_PAIR_QUIET = os.environ.get("TORCH_SCI_PAIR_QUIET", "").strip() in (
    "1",
    "true",
    "True",
    "yes",
)


def _pair_print(msg: str) -> None:
    """Flushing print for outer-loop diagnostics (muted by env var)."""
    if not _PAIR_QUIET:
        print(msg, flush=True)


def _pair_mem_line(prefix: str) -> str:
    """One-line memory footprint summary (host RSS + peak, optional GPU)."""
    parts: list[str] = []
    # Host RSS via psutil if available; else fall back to /proc/self/status.
    try:
        import psutil  # noqa: PLC0415

        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        parts.append(f"host_rss={rss_bytes / 1e9:.2f}GB")
    except Exception:
        try:
            with pathlib.Path("/proc/self/status").open("r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        parts.append(f"host_rss={kb / 1e6:.2f}GB")
                        break
        except Exception:
            pass
    # GPU allocated + peak (per-device) if torch.cuda is available.
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                peak = torch.cuda.max_memory_allocated(i) / 1e9
                parts.append(f"gpu[{i}]_alloc={allocated:.2f}GB peak={peak:.2f}GB")
    except Exception:
        pass
    return f"{prefix}: {', '.join(parts) if parts else '(n/a)'}"


# These engine-construction knobs are forwarded to ``TorchSCISolver``; any
# other keyword arguments passed to the entry points below are forwarded to
# the per-call ``solver(...)`` invocation (e.g. ``max_davidson_iters``, ``tol``,
# ``spin_penalty_lambda``). This mirrors how qiskit-addon-sqd forwards trailing
# kwargs, but the names are this engine's own -- nothing here routes to PySCF's
# ``kernel_fixed_space``, and the pyscf-style ``max_cycle`` alias is rejected.
_SOLVER_INIT_KEYS = frozenset({
    "device",
    "devices",
    "dtype",
    "chunk_size",
    "davidson_block_size",
    "max_davidson_iters",
    "davidson_tolerance",
    "compute_full_rdm",
    "compute_spin_squared",
    "verbose",
    "budget_fraction",
    "gdb_kwargs",
    "seed_strategy",
    "apply_ab_projection",
    "n_roots",
    "mode",
    # These were previously left in the per-call kwargs and silently ignored.
    "matvec_impl",
    "distributed",
    "davidson_dtype",
    "davidson_fp32_shift",
    "davidson_fp32_energy",
    "davidson_correction",
    "davidson_restart",
    "davidson_missed_root_check",
    "davidson_de_tolerance",
    # Interface knobs added with the public config/capabilities contract.
    "unknown_call_kwargs",
    "memory_budget_bytes",
    "table_dtype",
    "compact_eri",
    "triton_block",
    "diagnostics",
    "deterministic",
    "ab_factorization",
    "ab_factorization_tol",
})


def _split_kwargs(kwargs: dict) -> tuple[dict, dict]:
    """Partition kwargs into (solver-constructor, per-call) dictionaries."""
    init_kwargs = {k: v for k, v in kwargs.items() if k in _SOLVER_INIT_KEYS}
    call_kwargs = {k: v for k, v in kwargs.items() if k not in _SOLVER_INIT_KEYS}
    return init_kwargs, call_kwargs


def _popcount(value: int) -> int:
    """Popcount."""
    return int(value).bit_count()


def _as_ci_strs(
    bitstring_matrix: np.ndarray | tuple[Sequence[int], Sequence[int]],
    open_shell: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the ``bitstring_matrix`` argument to ``(ci_strs_a, ci_strs_b)``.

    Accepts either a 2D boolean bitstring matrix (as produced by Qiskit
    primitives) or a length-2 tuple of integer CI-string sequences, exactly
    like ``qiskit_addon_sqd.fermion.solve_fermion``.
    """
    if isinstance(bitstring_matrix, tuple):
        ci_a, ci_b = bitstring_matrix
    else:
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

        ci_a, ci_b = bitstring_matrix_to_ci_strs(
            np.asarray(bitstring_matrix), open_shell=open_shell
        )
    return _exact_ci_strs(ci_a), _exact_ci_strs(ci_b)


def _exact_ci_strs(values) -> np.ndarray:
    """CI strings as an exact integer array: int64, uint64 (bit 63), or object (wider).

    ``np.asarray(..., dtype=np.int64)`` overflows on a 64-orbital string using
    bit 63 and on any string wider than 64 bits; an int64 array of signed bit
    patterns is read as its unsigned words.
    """
    arr = np.asarray(values)
    if arr.dtype.kind in "iu" and arr.dtype.itemsize == 8:
        words = arr.ravel().view(np.uint64)
        if (words >= np.uint64(1 << 63)).any():
            return words.reshape(arr.shape)
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        exact = np.asarray(values, dtype=object)
        if not all(isinstance(v, (int, np.integer)) for v in exact.ravel().tolist()):
            raise TypeError("CI strings must be integers, not floating point.")
        arr = exact
    ints = [int(v) % (1 << 64) if int(v) < 0 else int(v) for v in arr.ravel().tolist()]
    if ints and max(ints) >= 1 << 64:
        return np.array(ints, dtype=object).reshape(arr.shape)
    dtype = np.uint64 if ints and max(ints) >= 1 << 63 else np.int64
    return np.array(ints, dtype=dtype).reshape(arr.shape)


def solve_fermion(
    bitstring_matrix: np.ndarray | tuple[Sequence[int], Sequence[int]],
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    open_shell: bool = False,
    spin_sq: float | None = None,
    **kwargs,
) -> tuple[float, SCIState, tuple[np.ndarray, np.ndarray], float]:
    r"""Approximate the ground state of a molecular Hamiltonian in a subspace.

    This is a drop-in replacement for
    :func:`qiskit_addon_sqd.fermion.solve_fermion` backed by the PyTorch
    selected-CI engine. It projects the Hamiltonian onto the subspace spanned
    by the input configurations and diagonalizes it with a Davidson solver that
    runs on CPU or CUDA.

    Args:
        bitstring_matrix: The configurations defining the subspace onto which
            the Hamiltonian is projected and diagonalized. Either a 2D
            ``numpy.ndarray`` of ``bool`` (one bitstring per row, spin-up in the
            left half and spin-down in the right half of each row) or a length-2
            tuple ``([a_str_0, ...], [b_str_0, ...])`` of integer CI strings.

            **Interpretation depends on ``mode``.** With the default
            ``mode="tpb"`` the tuple's two arrays are treated as
            independent per-spin unique sets and their Cartesian product forms
            the basis. With ``mode="gdb"`` the two arrays are treated
            as **parallel length-N pair lists** (index ``i`` = pair ``i``).
            These are different contracts — passing per-spin-unique arrays to
            ``mode="gdb"`` (as e.g. qiskit_addon_sqd's
            ``bitstring_matrix_to_ci_strs`` would produce them) silently
            re-interprets them as diagonal pairs. For pair mode extract the
            α and β halves per row of your bit_array yourself and pass those
            length-N arrays.
        hcore: The core Hamiltonian (one-electron integrals), shape ``(N, N)``.
        eri: The electron-repulsion (two-electron) integrals in chemist
            convention. Accepts the full ``(N, N, N, N)`` tensor or PySCF's
            4-fold / 8-fold packed forms.
        open_shell: If ``False``, the spin-up and spin-down configurations are
            merged into a single set used for both sectors. Only meaningful when
            ``bitstring_matrix`` is a bitstring matrix.
        spin_sq: Target value of :math:`\langle S^2 \rangle`. ``None`` (default)
            applies no spin penalty and returns the bare variational minimum. A
            float biases the solver toward that spin sector via an augmented
            ``H + λ(S² − spin_sq)`` Davidson solve; the returned energy is the
            physical :math:`\langle \psi | H | \psi \rangle` with the penalty
            contribution removed.
        **kwargs: Engine-construction options (``device``, ``devices``,
            ``dtype``, ``chunk_size``, ``davidson_block_size``, ``max_davidson_iters``,
            ``davidson_tolerance``, ``seed_strategy``, ``budget_fraction``,
            ``apply_ab_projection``, ``compute_full_rdm``,
            ``compute_spin_squared``, ``verbose``) are forwarded to the
            solver constructor. Any remaining kwargs (``max_davidson_iters``, ``tol``,
            ``spin_penalty_lambda``) are forwarded to the per-call
            diagonalization. Pass ``devices=["cuda:0","cuda:1",...]`` to
            fan the subspace batch across multiple GPUs on the same node
            (see :class:`TorchSCISolver`).

    Returns:
        A 4-tuple ``(energy, sci_state, occupancies, spin_sq)``:

        * ``energy`` — minimum energy found in the subspace.
        * ``sci_state`` — the approximate ground state as a
          :class:`qiskit_addon_sqd.fermion.SCIState`.
        * ``occupancies`` — ``(occ_a, occ_b)`` average spin-up / spin-down
          orbital occupancies, in the order expected by
          ``recover_configurations``.
        * ``spin_sq`` — :math:`\langle S^2 \rangle` of the returned state when a
          spin penalty was applied, else ``0.0``.
    """
    ci_a, ci_b = _as_ci_strs(bitstring_matrix, open_shell)
    if ci_a.size == 0 or ci_b.size == 0:
        raise ValueError(
            "Empty CI basis: the bitstring matrix produced no spin-up or spin-down configurations."
        )

    norb = int(np.asarray(hcore).shape[0])
    nelec = (_popcount(int(ci_a.reshape(-1)[0])), _popcount(int(ci_b.reshape(-1)[0])))

    init_kwargs, call_kwargs = _split_kwargs(kwargs)
    solver = make_torch_sci_solver(**init_kwargs)

    # ``solver(...)`` now returns list[EigenStates]; extract the ground-state
    # SCIResult from the single subspace (states[0]).
    eigen_states_list = cast(
        "list[_EigenStates]",
        solver(
            [(ci_a, ci_b)],
            np.asarray(hcore, dtype=np.float64),
            np.asarray(eri, dtype=np.float64),
            norb,
            nelec,
            spin_sq=spin_sq,
            **call_kwargs,
        ),
    )
    result = eigen_states_list[0].states[0]

    # The engine records ⟨S²⟩ of the converged eigenvector only when a spin
    # penalty was active. When no penalty was requested, report 0.0 to match
    # the (energy, state, occ, spin_sq) shape of qiskit-addon-sqd.
    sci_spin_sq = solver.stats.get("spin_sq_final")
    spin_sq_out = 0.0 if sci_spin_sq is None else float(cast(float, sci_spin_sq))

    return result.energy, result.sci_state, result.orbital_occupancies, spin_sq_out


def solve_sci_batch(
    ci_strings: list[tuple[np.ndarray, np.ndarray]],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    spin_sq: float | None = None,
    **kwargs,
) -> list[SCIResult]:
    r"""Diagonalize the Hamiltonian in a batch of subspaces.

    This is a drop-in replacement for
    :func:`qiskit_addon_sqd.fermion.solve_sci_batch`, suitable as the
    ``sci_solver`` argument of
    :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`. Each
    subspace is solved independently with the PyTorch Davidson engine.

    .. note::

       This convenience wrapper constructs a fresh :class:`TorchSCISolver` on
       every call, so it does not cache integrals or accumulate timing
       statistics across the iterations of an SQD run. For production runs ---
       especially on CUDA, where rebuilding the integral tensors every
       iteration is wasteful --- construct a single :class:`TorchSCISolver`
       once and pass *it* as ``sci_solver`` instead. The instance is callable
       with this exact signature and additionally exposes ``solver.stats`` and
       ``solver.format_perf_summary()``.

    Args:
        ci_strings: A list of ``(ci_strs_a, ci_strs_b)`` pairs, one per
            subspace (batch).
        one_body_tensor: One-body integrals, shape ``(norb, norb)``.
        two_body_tensor: Two-body integrals in chemist convention.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)`` electron counts.
        spin_sq: Target :math:`\langle S^2 \rangle` value, or ``None`` for no
            spin penalty. See :func:`solve_fermion`.
        **kwargs: Split between the solver constructor and the per-call
            diagonalization exactly as in :func:`solve_fermion`.

    Returns:
        A list of :class:`qiskit_addon_sqd.fermion.SCIResult`, one per input
        subspace.
    """
    init_kwargs, call_kwargs = _split_kwargs(kwargs)
    solver = make_torch_sci_solver(**init_kwargs)
    # ``solver(...)`` now returns list[EigenStates]; extract the ground-state
    # SCIResult from each subspace to preserve the list[SCIResult] public API.
    eigen_states_list = cast(
        "list[_EigenStates]",
        solver(
            ci_strings,
            one_body_tensor,
            two_body_tensor,
            norb,
            nelec,
            spin_sq=spin_sq,
            **call_kwargs,
        ),
    )
    return [es.states[0] for es in eigen_states_list]


def solve_fermion_from_bit_array_pairs(
    bit_array_or_matrix,
    /,
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    nelec: tuple[int, int],
    norb: int | None = None,
    spin_sq: float | None = None,
    max_pairs: int | None = None,
    seed: int | None = None,
    postselect: bool = True,
    **kwargs,
):
    r"""One-shot pair-mode SQD directly from a raw bit_array.

    Pair-mode counterpart to :func:`solve_fermion`. The natural pair-mode
    input is a raw sampler output — one measurement row per (α, β) pair —
    which qiskit-addon-sqd's ``diagonalize_fermionic_hamiltonian`` normally
    consumes by (a) postselecting rows to the target particle count,
    (b) subsampling to control the subspace size, then (c) deduplicating
    *independently per spin* and forming the Cartesian product for
    Davidson. Step (c) is exactly the semantics pair mode is designed to
    replace: pair mode dedups by ``(α, β)`` **row** and diagonalizes over
    the resulting N-dim subspace, sidestepping the ``M_α × M_β`` blowup
    that motivated batching in the first place.

    Because ``N ≤ n_shots`` and typical noisy-shot samples fit in memory
    comfortably (10k, 100k rows), the whole pair set is diagonalized in
    one Davidson call — no outer SQD loop, no per-batch carryover, no
    ``max_dim`` cap on per-spin sectors. If a future use case produces
    a pair set that's too large for one Davidson, a carryover-based
    outer loop can wrap this function; nothing here precludes it.

    Args:
        bit_array_or_matrix: Either a Qiskit ``BitArray`` (one measurement
            row per shot) or an already-unpacked 2D ``numpy.ndarray`` of
            bool. Row layout: α occupancies in the left half (bits
            ``0..norb-1``), β occupancies in the right half (bits
            ``norb..2*norb-1``) — matches ``qiskit_addon_sqd``'s convention.
        hcore, eri: One- and two-electron integrals (chemist convention),
            same as :func:`solve_fermion`.
        nelec: ``(n_α, n_β)`` — needed for both postselect (dropping rows
            with wrong particle counts) and the pair-mode solver.
        norb: Number of spatial orbitals. Defaults to ``hcore.shape[0]``.
        spin_sq: Target ⟨S²⟩ for the S² penalty; ``None`` (default)
            disables it. Passed through to :func:`solve_fermion`.
        max_pairs: Cap on the pair-set size *after* postselect. If the
            input yields more pairs, we subsample without replacement.
            ``None`` (default) keeps all rows. Set only when you know the
            solver can't hold the full pair set (rarely needed — pair mode
            memory is linear in N).
        seed: RNG seed for the ``max_pairs`` subsample. Ignored otherwise.
        postselect: If ``True`` (default), drop rows whose α-popcount ≠
            ``n_α`` or β-popcount ≠ ``n_β`` (noisy measurements outside
            the physical particle-count sector). Set ``False`` if the
            caller already applied a postselect or wants to include
            imperfect rows.
        **kwargs: Forwarded to :func:`solve_fermion`. Common ones:
            ``device``, ``chunk_size``, ``davidson_block_size``, ``compute_full_rdm``,
            ``compute_spin_squared``, ``max_davidson_iters``, ``tol``.

    Returns:
        The 4-tuple returned by :func:`solve_fermion` — ``(energy,
        sci_state, occupancies, spin_sq)``. The ``sci_state`` is a proper
        :class:`qiskit_addon_sqd.fermion.SCIState` with 2D
        ``(M_α_unique, M_β_unique)`` amplitudes (sparse — zeros at
        ``(α, β)`` positions that are not in the pair basis).

    Raises:
        ValueError: if ``postselect=True`` drops every row (no rows match
            the target particle count).
    """
    if norb is None:
        norb = int(np.asarray(hcore).shape[0])

    # Normalize to a (num_shots, 2*norb) bool matrix.
    rows = _as_bool_bitstring_matrix(bit_array_or_matrix, num_bits=2 * norb)
    n_a, n_b = int(nelec[0]), int(nelec[1])

    # Bit layout — matches ``qiskit_addon_sqd.fermion.bitstring_matrix_to_ci_strs``:
    #   * left half (columns 0..norb-1)      → β (spin-down)
    #   * right half (columns norb..2*norb-1) → α (spin-up)
    # And ``bitstring_matrix_to_integers`` packs each half MSB-first:
    # column 0 has weight ``1 << (norb-1)``, column norb-1 has weight 1.
    # Doing anything else means our α/β integers would not agree with what
    # ``TorchSCISolver`` expects for product-mode strings — which would
    # give wrong physics on asymmetric systems (n_α ≠ n_β).
    beta_bits = rows[:, :norb]
    alpha_bits = rows[:, norb : 2 * norb]
    if postselect:
        keep = (alpha_bits.sum(axis=1) == n_a) & (beta_bits.sum(axis=1) == n_b)
        alpha_bits = alpha_bits[keep]
        beta_bits = beta_bits[keep]
        if alpha_bits.shape[0] == 0:
            raise ValueError(
                f"solve_fermion_from_bit_array_pairs: postselect dropped every "
                f"row (nelec=({n_a}, {n_b}), norb={norb}). Every input row had "
                f"the wrong α- or β-popcount. Check that the row layout follows "
                f"qiskit_addon_sqd's convention (β in the left half, α in the "
                f"right half) and that ``nelec`` matches the sample. If the "
                f"sample is from real quantum hardware and every row has "
                f"noise-corrupted popcount, pre-process with "
                f"``qiskit_addon_sqd.configuration_recovery.recover_configurations`` "
                f"before calling this function."
            )

    # Subsample if a cap is requested.
    if max_pairs is not None and alpha_bits.shape[0] > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(alpha_bits.shape[0], size=int(max_pairs), replace=False)
        alpha_bits = alpha_bits[idx]
        beta_bits = beta_bits[idx]

    # MSB-first packing: column i → bit ``(norb - 1 - i)``. Matches
    # ``bitstring_matrix_to_integers`` exactly so an α or β integer here
    # is bit-identical to what ``bitstring_matrix_to_ci_strs`` would
    # produce on the same rows.
    shift = np.arange(norb - 1, -1, -1, dtype=np.int64)  # [norb-1, ..., 1, 0]
    alpha_int = (alpha_bits.astype(np.int64) << shift[None, :]).sum(axis=1)
    beta_int = (beta_bits.astype(np.int64) << shift[None, :]).sum(axis=1)

    # Call TorchSCISolver directly rather than going through solve_fermion,
    # because solve_fermion infers ``nelec`` from ``popcount(ci_a[0])`` — a
    # heuristic that's fine in product mode (per-spin dedup usually
    # filters to physical popcounts implicitly via recovery) but wrong in
    # pair mode when the caller pre-filters via ``postselect``. We already
    # have the caller-supplied ``nelec``; use it directly and record the
    # penalty ``⟨S²⟩`` in the return tuple the way solve_fermion does.
    init_kwargs, call_kwargs = _split_kwargs(kwargs)
    init_kwargs.setdefault("mode", "gdb")
    solver = make_torch_sci_solver(**init_kwargs)
    # ``solver(...)`` returns list[EigenStates] since the multiroot refactor;
    # this convenience API consumes the ground-state SCIResult at states[0].
    eigen_states_list = cast(
        "list[_EigenStates]",
        solver(
            [(alpha_int, beta_int)],
            np.asarray(hcore, dtype=np.float64),
            np.asarray(eri, dtype=np.float64),
            int(norb),
            (int(n_a), int(n_b)),
            spin_sq=spin_sq,
            **call_kwargs,
        ),
    )
    result = eigen_states_list[0].states[0]
    sci_spin_sq = solver.stats.get("spin_sq_final")
    spin_sq_out = 0.0 if sci_spin_sq is None else float(cast(float, sci_spin_sq))
    return result.energy, result.sci_state, result.orbital_occupancies, spin_sq_out


def dump_pair_coupling_magnitudes(
    sci_state: "PairSCIState",
    hcore: np.ndarray,
    eri: np.ndarray,
    *,
    norb: int,
    nelec: tuple[int, int],
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
    label: str = "",
) -> dict:
    """Evaluate every coupling in a pair-mode ``SCIState``'s implied
    pair-helper tables and print an ``|Δ|``-distribution summary.

    Answers the question "are the pair-mode off-diagonal matrix elements
    genuinely small (making Davidson's tiny variational improvement
    physically explicable), or are they large but the ground state
    isn't shifting because of a kernel issue?" — by dumping the actual
    magnitudes of every entry in every pair-helper table.

    Reconstructs a :class:`PairBasis` from the flat length-N
    ``sci_state.ci_strs_a`` and ``ci_strs_b``, rebuilds pair_helpers on
    ``device``, then evaluates each same-spin table entry with
    ``one_excite_batch`` / ``two_excite_batch`` (the same primitives
    the matvec uses). Same-spin doubles and αβ-doubles use
    ``two_excite_batch``; same-spin singles use ``one_excite_batch``.

    Args:
        sci_state: A ``PairSCIState`` (as returned by pair-mode
            diagonalization). ``sci_state.ci_strs_a`` and ``ci_strs_b``
            are the length-N pair-parallel string arrays.
        hcore, eri: The integrals used in the original diagonalization.
        norb, nelec: System parameters.
        device: Device to rebuild pair_helpers on (usually the same as
            the original run).
        dtype: torch dtype for the rebuild; defaults to float64.
        label: Optional short tag included in each print line — useful
            when calling this on several results in a sweep.

    Returns:
        A dict mapping table name (``"asingles"``, ``"adoubles"``,
        ``"bsingles"``, ``"bdoubles"``, ``"ab"``) to a summary dict with
        keys ``K``, ``min``, ``median``, ``max``, ``mean``, ``sum_sq``.
        Same info as printed.
    """

    if dtype is None:
        dtype = torch.float64
    dev = torch.device(device) if isinstance(device, str) else device

    tag = f" [{label}]" if label else ""
    n_a, n_b = int(nelec[0]), int(nelec[1])

    # Rebuild pair basis + helpers from the SCIState's flat ci_strs.
    alpha_int = np.asarray(sci_state.ci_strs_a, dtype=np.int64)
    beta_int = np.asarray(sci_state.ci_strs_b, dtype=np.int64)
    pair_basis = pair_basis_from_arrays(alpha_int, beta_int, device=dev)
    n_pairs = pair_basis.n_pairs

    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(
        np.asarray(hcore, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        0.0,
    )
    integrals = to_torch_integrals(I0, I1, I2, device=dev, dtype=dtype)
    ph = generate_pair_helpers(pair_basis, norb, device=dev)

    alpha_of_pair = pair_basis.alpha_of_pair.to(dev)
    beta_of_pair = pair_basis.beta_of_pair.to(dev)

    def _bra_det(kb: "Tensor") -> "Tensor":
        """Full-determinant limb tensor for each bra pair index.

        Args:
            kb: ``(K,)`` int64 bra pair indices into the pair basis.

        Returns:
            ``(K, n_limbs)`` int64 tensor encoding each bra determinant over
            ``2 * norb`` spin-orbitals.
        """
        return det_from_alpha_beta_batch(alpha_of_pair[kb], beta_of_pair[kb], norb)

    def _eval_singles(t: dict[str, "Tensor"]) -> "Tensor":
        """Matrix elements of one streamed same-spin single-excitation chunk.

        Args:
            t: One chunk from a :class:`PairHelpers` same-spin singles stream,
                with keys ``k_bra``, ``k_ket``, ``cr0``, ``an0``.

        Returns:
            ``(K,)`` tensor of ``<D_bra|H|D_ket>`` for the chunk's rows.
        """
        bra = _bra_det(t["k_bra"])
        return one_excite_batch(
            bra,
            None,
            t["an0"],
            t["cr0"],
            integrals.I1,
            integrals.I2_eri,
            integrals.n_spinorb,
        )

    def _eval_doubles(t: dict[str, "Tensor"], keys: tuple[str, str, str, str]) -> "Tensor":
        """Matrix elements of one streamed double-excitation chunk.

        Args:
            t: One chunk from a :class:`PairHelpers` doubles stream.
            keys: The chunk's ``(cr0, cr1, an0, an1)`` key names — same-spin
                tables use those literal names, the αβ table uses
                ``("cr_a", "cr_b", "an_a", "an_b")``.

        Returns:
            ``(K,)`` tensor of ``<D_bra|H|D_ket>`` for the chunk's rows.
        """
        cr0, cr1, an0, an1 = (t[k] for k in keys)
        bra = _bra_det(t["k_bra"])
        return two_excite_batch(bra, None, an0, an1, cr0, cr1, integrals.I2_eri)

    def _collect(
        stream: "Callable[[], Iterator[dict[str, Tensor]]]",
        evaluate: "Callable[[dict[str, Tensor]], Tensor]",
    ) -> "Tensor | None":
        """Concatenate a stream's matrix elements, or None if the table is empty.

        This is the one place that deliberately holds O(nnz) values: the
        magnitude histogram is a distribution over all couplings, so there is
        nothing to stream it into. Values only (8 B each), not the index
        tables — the solver path never does this. Diagnostic use only.

        Args:
            stream: A :class:`PairHelpers` table accessor, called to obtain a
                fresh chunk generator.
            evaluate: Maps one chunk to its ``(K,)`` matrix elements.

        Returns:
            All matrix elements of the table concatenated, or ``None`` when the
            table has no rows.
        """
        parts = [evaluate(t) for t in stream()]
        parts = [p for p in parts if p.numel() > 0]
        return torch.cat(parts) if parts else None

    dbl = ("cr0", "cr1", "an0", "an1")
    tables = [
        ("asingles", _collect(ph.asingles, _eval_singles)),
        ("adoubles", _collect(ph.adoubles, lambda t: _eval_doubles(t, dbl))),
        ("bsingles", _collect(ph.bsingles, _eval_singles)),
        ("bdoubles", _collect(ph.bdoubles, lambda t: _eval_doubles(t, dbl))),
        (
            "ab",
            _collect(ph.ab_doubles, lambda t: _eval_doubles(t, ("cr_a", "cr_b", "an_a", "an_b"))),
        ),
    ]

    header = (
        f"[coupling{tag}]  N_pairs={n_pairs} nelec=({n_a},{n_b}) norb={norb}\n"
        f"[coupling{tag}]  table         K       |min|      |med|      |max|     mean|Δ|    Σ|Δ|²"
    )
    print(header, flush=True)
    summary: dict = {}
    for name, vals in tables:
        if vals is None or vals.numel() == 0:
            print(
                f"[coupling{tag}]  {name:8s}      0       -          -          -         -         -",
                flush=True,
            )
            summary[name] = {
                "K": 0,
                "min": None,
                "median": None,
                "max": None,
                "mean": None,
                "sum_sq": 0.0,
            }
            continue
        abs_vals = vals.abs()
        k = int(abs_vals.numel())
        vmin = float(abs_vals.min().item())
        vmed = float(abs_vals.median().item())
        vmax = float(abs_vals.max().item())
        vmean = float(abs_vals.mean().item())
        vsum_sq = float((vals**2).sum().item())
        print(
            f"[coupling{tag}]  {name:8s}  {k:5d}   "
            f"{vmin:.3e}  {vmed:.3e}  {vmax:.3e}  {vmean:.3e}  {vsum_sq:.3e}",
            flush=True,
        )
        summary[name] = {
            "K": k,
            "min": vmin,
            "median": vmed,
            "max": vmax,
            "mean": vmean,
            "sum_sq": vsum_sq,
        }
    return summary


def diagonalize_fermionic_hamiltonian_pairs(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    bit_array,
    *,
    samples_per_batch: int | None = None,
    norb: int,
    nelec: tuple[int, int],
    max_iterations: int = 10,
    num_batches: int = 1,
    max_pairs_per_batch: int | None = None,
    energy_tol: float = 1e-8,
    occupancies_tol: float = 1e-5,
    carryover_threshold: float = 1e-4,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    include_pairs: list[tuple[int, int]] | None = None,
    seed: int | None = None,
    solver=None,
    callback=None,
    use_full_recovered_pool: bool = False,
    add_hf_pair: bool = False,
    **solver_kwargs,
):
    r"""Pair-mode analog of
    :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`.

    Same iterative recovery loop over a noisy ``bit_array`` — postselect
    on iteration 0 (no prior occupancies yet), ``recover_configurations``
    on iteration 1+ using the running RDM occupancies as prior, subsample
    into batches, diagonalize, aggregate best-of-batch, carry over the
    highest-amplitude configurations to the next iteration. The one
    difference: each batch's Davidson runs on the batch's row-deduped
    ``(α, β)`` pair set rather than the Cartesian product of independently
    dedup'd per-spin uniques.

    Use this when the sampler output is a raw noisy ``BitArray`` from a
    quantum device — the recovery step is what makes the whole pipeline
    tolerate the fact that essentially no raw row has the exact target
    particle count. For a pre-recovered ``bit_array`` or a hand-curated
    pair list, prefer :func:`solve_fermion_from_bit_array_pairs` — it's
    a single Davidson call, no outer loop.

    Args:
        one_body_tensor, two_body_tensor: Same integrals as
            :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`.
        bit_array: Qiskit ``BitArray`` of raw noisy measurement shots.
        samples_per_batch: Bitstrings drawn (with replacement) per batch.
            Ignored when ``use_full_recovered_pool=True``.
        norb: Number of spatial orbitals.
        nelec: ``(n_α, n_β)`` — targets for recovery and postselect.
        max_iterations: Outer-loop iteration cap.
        num_batches: Number of independent subsampled batches per
            iteration. The best-energy batch defines the iteration's
            result and carryover.
        max_pairs_per_batch: Optional cap on the pair-set size per batch
            (after row-dedup + carryover + include). ``None`` (default)
            keeps the full deduped set. Set only when a batch produces
            more pairs than Davidson can hold.
        energy_tol, occupancies_tol: Convergence thresholds (same
            semantics as qiskit-addon-sqd).
        carryover_threshold: Amplitudes above this threshold are kept
            as carryover pairs for the next iteration.
        initial_occupancies: Optional ``(α, β)`` orbital-occupancy prior
            for iteration 0 (skips postselect-only fallback).
        include_pairs: Optional list of ``(α_int, β_int)`` tuples always
            included in every batch's pair set.
        seed: RNG seed for subsample + recovery.
        solver: A prebuilt pair-mode ``TorchSCISolver`` (must have
            ``mode="gdb"``). If ``None``, one is constructed
            from ``solver_kwargs``.
        callback: Optional callable invoked with per-iteration
            ``list[SCIResult]`` (mirrors the qiskit-addon-sqd hook).
        use_full_recovered_pool: If True (default False), skip the
            ``subsample`` step entirely and diagonalize the *whole*
            recovered bitstring array as one batch per iteration.
            ``samples_per_batch`` is ignored and ``num_batches`` is
            forced to 1. Use when you want deterministic coverage of
            the sample (no sampling variance) rather than a Monte
            Carlo subsample — pair mode's memory scales with the
            unique-pair count after row-dedup, which is bounded by
            ``bit_array.num_shots`` and typically much smaller.
        add_hf_pair: If True (default False), inject the HF pair
            ``(HF_α, HF_β)`` into every batch's pair set if missing.
            HF has substantial ground-state overlap for closed-shell
            chemistry, and its presence gives Davidson a guaranteed
            non-zero component along a physically meaningful direction
            — otherwise a batch that samples zero configurations near
            HF (common on noisy hardware samples) yields a Davidson
            initial residual that fails to find the ground state.
        **solver_kwargs: Forwarded to ``make_torch_sci_solver`` when
            ``solver`` is ``None``.

    Returns:
        The lowest-energy ``SCIResult`` seen across all iterations —
        matching the return contract of
        :func:`qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`.
    """
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.counts import bit_array_to_arrays
    from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left, subsample

    rng = np.random.default_rng(seed)
    n_a, n_b = int(nelec[0]), int(nelec[1])

    if not use_full_recovered_pool and samples_per_batch is None:
        raise ValueError(
            "samples_per_batch is required when use_full_recovered_pool=False. "
            "Pass an integer or set use_full_recovered_pool=True to diagonalize "
            "the entire recovered bitstring pool per iteration."
        )
    if use_full_recovered_pool and int(num_batches) != 1:
        raise ValueError(
            "use_full_recovered_pool=True forces num_batches=1 (the whole "
            f"pool is a single batch); got num_batches={num_batches}."
        )

    if solver is None:
        solver_kwargs.setdefault("mode", "gdb")
        solver = make_torch_sci_solver(**solver_kwargs)
    elif getattr(solver, "mode", "tpb") != "gdb":
        raise ValueError(
            'diagonalize_fermionic_hamiltonian_pairs requires a GDB-mode solver (mode="gdb").'
        )

    # Pairs are keyed by their unsigned string values throughout the loop (rows,
    # carryover and include pairs alike), so a 64-orbital string using bit 63 has
    # one key; a negative int is taken as its signed int64 bit pattern.
    include_pairs_list: list[tuple[int, int]] = (
        [(int(a) % (1 << 64), int(b) % (1 << 64)) for a, b in include_pairs]
        if include_pairs
        else []
    )
    if add_hf_pair:
        # HF: lowest n_α α-orbitals occupied, lowest n_β β-orbitals occupied.
        # Matches sbdiag / probe_pairs.
        hf_alpha_int = (1 << n_a) - 1
        hf_beta_int = (1 << n_b) - 1
        hf_tuple = (int(hf_alpha_int), int(hf_beta_int))
        if hf_tuple not in include_pairs_list:
            include_pairs_list = [hf_tuple, *include_pairs_list]

    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    if raw_bitstrings.shape[1] != 2 * norb:
        raise ValueError(
            f"bit_array has {raw_bitstrings.shape[1]} bits/shot; expected "
            f"2*norb = {2 * norb}. Check that ``norb`` matches the sample."
        )

    shift_msb = np.arange(norb - 1, -1, -1, dtype=np.uint64)

    def _rows_to_pairs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract (α, β) integer pairs from a bitstring row batch —
        β from the left half, α from the right half, MSB-first packing,
        as unsigned 64-bit words (orbital 63 of a 64-orbital basis is bit 63).
        """
        beta_half = rows[:, :norb]
        alpha_half = rows[:, norb : 2 * norb]
        alpha_int = (alpha_half.astype(np.uint64) << shift_msb[None, :]).sum(
            axis=1, dtype=np.uint64
        )
        beta_int = (beta_half.astype(np.uint64) << shift_msb[None, :]).sum(axis=1, dtype=np.uint64)
        return alpha_int, beta_int

    current_occupancies = initial_occupancies
    best_result = None
    current_result = None
    carryover_pairs: list[tuple[int, int]] = []

    for iter_idx in range(int(max_iterations)):
        iter_t0 = _time.perf_counter()
        # 1. Recovery vs postselect.
        t0 = _time.perf_counter()
        if current_occupancies is None:
            bitstrings, probs = postselect_by_hamming_right_and_left(
                raw_bitstrings,
                raw_probs,
                hamming_right=n_a,
                hamming_left=n_b,
            )
            recovery_kind = "postselect"
            if bitstrings.size == 0:
                raise ValueError(
                    "diagonalize_fermionic_hamiltonian_pairs: the input bit "
                    "array contained no bitstrings with the correct α/β "
                    "particle counts, and no initial_occupancies were given. "
                    "Either pass an initial_occupancies prior so recovery can "
                    "bit-flip noisy shots toward the target sector, or use a "
                    "bit_array that includes at least one physical row."
                )
        else:
            bitstrings, probs = recover_configurations(
                raw_bitstrings,
                raw_probs,
                current_occupancies,
                n_a,
                n_b,
                rand_seed=rng,
            )
            recovery_kind = "recover"
        t_recovery = _time.perf_counter() - t0
        recovery_rows = int(bitstrings.shape[0]) if bitstrings.size else 0
        _pair_print(
            f"[pair_sqd iter={iter_idx}] {recovery_kind}: raw={raw_bitstrings.shape[0]} "
            f"→ pool={recovery_rows}  ({t_recovery:.2f}s)"
        )

        # 2. Subsample into batches — or use the whole recovered pool.
        t0 = _time.perf_counter()
        if use_full_recovered_pool:
            # One "batch" containing every recovered row. Row-dedup in
            # step 3 collapses duplicates naturally.
            subsamples = [bitstrings]
        else:
            subsamples = subsample(
                bitstrings,
                probs,
                samples_per_batch=int(samples_per_batch),
                num_batches=int(num_batches),
                rand_seed=rng,
            )
        t_subsample = _time.perf_counter() - t0
        _pair_print(
            f"[pair_sqd iter={iter_idx}] subsample: n_batches={len(subsamples)} "
            f"rows_per_batch={[int(s.shape[0]) for s in subsamples]}  "
            f"({t_subsample:.2f}s)"
        )

        # 3. Row-decode each batch, merge with carryover + include pairs.
        t0 = _time.perf_counter()
        pair_inputs: list[tuple[np.ndarray, np.ndarray]] = []
        batch_stats: list[dict] = []
        for b_idx, sample_rows in enumerate(subsamples):
            alpha_ints, beta_ints = _rows_to_pairs(sample_rows)
            raw_pair_set = set(zip(alpha_ints.tolist(), beta_ints.tolist()))
            after_raw_dedup = len(raw_pair_set)
            pair_set = set(raw_pair_set)
            for p in carryover_pairs:
                pair_set.add(p)
            after_carryover = len(pair_set)
            for p in include_pairs_list:
                pair_set.add(p)
            after_include = len(pair_set)
            pair_list = sorted(pair_set)  # deterministic lex order
            if max_pairs_per_batch is not None and len(pair_list) > int(max_pairs_per_batch):
                pair_list = pair_list[: int(max_pairs_per_batch)]
            pa = np.asarray([p[0] for p in pair_list], dtype=np.uint64)
            pb = np.asarray([p[1] for p in pair_list], dtype=np.uint64)
            pair_inputs.append((pa, pb))
            batch_stats.append({
                "batch": b_idx,
                "rows": int(sample_rows.shape[0]),
                "after_row_dedup": after_raw_dedup,
                "+carryover": after_carryover,
                "+include": after_include,
                "final_N": pa.shape[0],
            })
        t_prep = _time.perf_counter() - t0
        for bs in batch_stats:
            _pair_print(
                f"[pair_sqd iter={iter_idx}] batch={bs['batch']}: "
                f"rows={bs['rows']} → dedup={bs['after_row_dedup']} "
                f"→ +carryover={bs['+carryover']} → +include={bs['+include']} "
                f"→ N={bs['final_N']}"
            )
        _pair_print(f"[pair_sqd iter={iter_idx}] batch prep: {t_prep:.2f}s")

        # 4. Diagonalize each batch in pair mode.
        # ``solver(...)`` returns list[EigenStates] since the multiroot
        # refactor; configuration recovery consumes each subspace's
        # ground-state SCIResult via ``.states[0]`` here, preserving the
        # ``r.sci_state`` / ``r.energy`` shape the rest of this loop uses.
        t0 = _time.perf_counter()
        eigen_states_list = cast(
            "list[_EigenStates]",
            solver(
                pair_inputs,
                np.asarray(one_body_tensor, dtype=np.float64),
                np.asarray(two_body_tensor, dtype=np.float64),
                int(norb),
                (n_a, n_b),
            ),
        )
        results = [es.states[0] for es in eigen_states_list]
        t_solve = _time.perf_counter() - t0
        if callback is not None:
            callback(results)
        # Per-batch Davidson results (energy + carryover-eligible count).
        # Also surface the Davidson convergence state — hitting the max
        # matvec cap is the load-bearing warning to look for at scale.
        for b_idx, r in enumerate(results):
            amp = r.sci_state.amplitudes
            n_high = int((np.abs(amp) > float(carryover_threshold)).sum())
            conv = getattr(r, "state", "unknown")
            _pair_print(
                f"[pair_sqd iter={iter_idx}] batch={b_idx} result: "
                f"E={r.energy:.6f} Ha, davidson={conv}, "
                f"|amp|>{carryover_threshold}: {n_high}/{amp.shape[0]}"
            )
        _pair_print(f"[pair_sqd iter={iter_idx}] solve: {t_solve:.2f}s")

        # 5. Aggregate best-of-batch and track running best.
        best_in_batch = min(results, key=lambda r: r.energy)
        if best_result is None or best_in_batch.energy < best_result.energy:
            best_result = best_in_batch

        # 6. Convergence check (both energy AND occupancies stable).
        if (
            current_result is not None
            and current_occupancies is not None
            and abs(current_result.energy - best_in_batch.energy) < energy_tol
            and np.linalg.norm(
                np.ravel(current_occupancies) - np.ravel(best_in_batch.orbital_occupancies),
                ord=np.inf,
            )
            < occupancies_tol
        ):
            break
        current_result = best_in_batch
        current_occupancies = current_result.orbital_occupancies

        # 7. Carry over high-amplitude pairs. Pair-mode SCIState is a
        # flat ``PairSCIState`` — ``amplitudes[k]`` is the amplitude
        # on pair ``k``, ``(ci_strs_a[k], ci_strs_b[k])`` is that pair.
        # Just threshold the length-N amplitude vector and gather.
        sci_state = current_result.sci_state
        abs_amps = np.abs(sci_state.amplitudes)
        keep_mask = abs_amps > float(carryover_threshold)
        if keep_mask.any():
            keep_alpha = sci_state.ci_strs_a[keep_mask]
            keep_beta = sci_state.ci_strs_b[keep_mask]
            carryover_pairs = [
                (int(a) % (1 << 64), int(b) % (1 << 64))
                for a, b in zip(keep_alpha.tolist(), keep_beta.tolist())
            ]
        else:
            carryover_pairs = []

        # 8. End-of-iteration summary with wall-clock and memory footprint.
        iter_dt = _time.perf_counter() - iter_t0
        occ_a_sum = float(np.asarray(current_occupancies[0]).sum())
        occ_b_sum = float(np.asarray(current_occupancies[1]).sum())
        _pair_print(
            f"[pair_sqd iter={iter_idx}] carryover kept {len(carryover_pairs)} pairs "
            f"(|amp|>{carryover_threshold})"
        )
        _pair_print(
            f"[pair_sqd iter={iter_idx}] occupancies: "
            f"Σα={occ_a_sum:.4f} (target {n_a}), Σβ={occ_b_sum:.4f} (target {n_b})"
        )
        _pair_print(_pair_mem_line(f"[pair_sqd iter={iter_idx}] mem"))
        _pair_print(
            f"[pair_sqd iter={iter_idx}] wall: {iter_dt:.1f}s  "
            f"(best-so-far E={best_result.energy:.6f} Ha)"
        )

    return best_result


def _as_bool_bitstring_matrix(bit_array_or_matrix, *, num_bits: int) -> np.ndarray:
    """Normalize a Qiskit ``BitArray`` or a 2D bool ndarray to a
    ``(num_shots, num_bits)`` bool matrix.

    For ``BitArray`` inputs we use ``to_bool_array()`` (the inverse of
    ``from_bool_array``). Reading ``ba.array`` directly with
    ``np.unpackbits`` does not round-trip: Qiskit's internal packing
    lays out bits in a byte-order-dependent way that ``unpackbits``
    would misinterpret. ``to_bool_array()`` is the API contract.
    """
    if isinstance(bit_array_or_matrix, np.ndarray):
        arr = np.asarray(bit_array_or_matrix, dtype=bool)
        if arr.ndim != 2 or arr.shape[1] != num_bits:
            raise ValueError(
                f"bitstring matrix must be 2D with shape (num_shots, {num_bits}); got {arr.shape}."
            )
        return arr
    # BitArray path — round-trip via its own decoder.
    arr = bit_array_or_matrix.to_bool_array()
    if arr.ndim != 2 or arr.shape[1] != num_bits:
        raise ValueError(
            f"BitArray decodes to shape {arr.shape}; expected "
            f"(num_shots, {num_bits}) for norb={num_bits // 2}. Check that "
            f"``norb`` matches ``bit_array.num_bits // 2``."
        )
    return arr.astype(bool, copy=False)


def make_torch_sci_solver(**kwargs) -> TorchSCISolver:
    """Construct a :class:`TorchSCISolver`.

    See :class:`TorchSCISolver` for the full list of accepted keyword
    arguments (``device``, ``devices``, ``dtype``, ``chunk_size``,
    ``seed_strategy``, ...). A constructed instance is callable with the
    :func:`solve_sci_batch` signature and is the recommended ``sci_solver`` to
    reuse across the iterations of an SQD run. Pass
    ``devices=["cuda:0","cuda:1",...]`` to spread each subspace batch
    across multiple GPUs on the same node.

    .. note::

       For drop-in compatibility with ``qiskit_addon_sqd.fermion``, this
       factory defaults ``apply_ab_projection=False``, so the solver returns
       the bare variational minimum in the subspace (matching PySCF's
       selected-CI, and reproducing FCI exactly on a complete basis). The
       engine's αβ-swap projection --- useful for suppressing triplet
       contamination from round-off on large noisy bases --- is available as an
       opt-in by passing ``apply_ab_projection=True``, or enforce a spin sector
       explicitly with the ``spin_sq`` argument. Construct
       :class:`TorchSCISolver` directly to get the engine's native default
       (``apply_ab_projection=True``).
    """
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    if "apply_ab_projection" not in kwargs:
        warnings.warn(
            "make_torch_sci_solver() defaults apply_ab_projection=False for "
            "qiskit_addon_sqd drop-in behaviour, while TorchSCISolver's native "
            "default is True. Pass apply_ab_projection explicitly to silence this "
            "warning and make the choice visible.",
            UserWarning,
            stacklevel=2,
        )
        kwargs["apply_ab_projection"] = False
    return TorchSCISolver(**kwargs)
