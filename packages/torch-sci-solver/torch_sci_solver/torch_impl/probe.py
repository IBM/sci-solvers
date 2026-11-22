"""Probe-method capabilities for the torch sci solver.

A "probe" is a per-call hook into the solver's pipeline that exposes
intermediate state for the user to feed back into a subsequent call.
The current and only built-in probe method is **warm_start**: it caches
the converged interleaved-Wick eigenvector along with enough provenance
data to seed Davidson on a basis extension.

Contract — see the factory docstring for the full statement. In one line:

    Given solver knobs (h1, eri, apply_ab_projection, ...) and any
    bases B₀ ⊆ B (set inclusion on α-strings and β-strings independently,
    so B is the Cartesian product of the unions),

        |warm_extend(B₀ → B).energy − cold_solve(B).energy| ≤ davidson_tolerance

This module is plumbing only; the actual Davidson re-seeding hook lives
in the solver factory.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

# ----------------------------------------------------------------------------
# Knob registry — which kwargs affect the converged eigenvalue and must
# therefore match between cold-cache and warm-extend.
# ----------------------------------------------------------------------------

# Set of solver-construction knobs that materially change the operator
# being diagonalized or the convergence target. Any mismatch between a
# cached state's knobs and a warm-extend call's knobs is an error
# (different problem → "exact extension" claim is undefined).
_KNOB_FIELDS: Tuple[str, ...] = (
    "apply_ab_projection",
    "symmetrize_spin",
    "seed_strategy",
    # Davidson knobs — strictly speaking different values still converge
    # to the same answer if both are tight enough, but a mismatch is a
    # signal something unexpected is happening. We compare softly:
    # davidson_tolerance and max_davidson_iters are recorded but a mismatch is just a
    # warning.
)
_KNOB_FIELDS_SOFT: Tuple[str, ...] = (
    "davidson_tolerance",
    "davidson_block_size",
    "max_davidson_iters",
    "chunk_size",
)


# ----------------------------------------------------------------------------
# Fingerprinting
# ----------------------------------------------------------------------------


def _integrals_fingerprint(h1: np.ndarray, eri: np.ndarray, ecore: float = 0.0) -> str:
    """SHA-256 of (h1, eri, ecore) byte representations.

    Used to detect when a warm-extend call is given a different
    Hamiltonian than the cached state. Hash is cheap (~50 µs at
    L=36) and bitwise-strict — even a single float-bit difference fires.

    Complex ``h1``/``eri`` (dtype ``complex64``/``complex128``) are hashed at
    their own complex dtype rather than cast to ``float64`` first — casting
    would silently drop the imaginary part and could collide the
    fingerprints of two different complex Hamiltonians that share a real
    part.
    """
    h1_dtype = np.complex128 if np.iscomplexobj(h1) else np.float64
    eri_dtype = np.complex128 if np.iscomplexobj(eri) else np.float64
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(h1, dtype=h1_dtype).tobytes())
    h.update(np.ascontiguousarray(eri, dtype=eri_dtype).tobytes())
    h.update(np.float64(ecore).tobytes())
    return h.hexdigest()


def _basis_fingerprint(adets: List[int], bdets: List[int]) -> str:
    """Hash of the canonical (sorted+deduped) basis strings.

    For sanity-checking that a state's stored basis is what callers think.
    """
    a = np.asarray(sorted(set(int(x) for x in adets)), dtype=np.int64)
    b = np.asarray(sorted(set(int(x) for x in bdets)), dtype=np.int64)
    h = hashlib.sha256()
    h.update(a.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# State dataclass
# ----------------------------------------------------------------------------


@dataclass
class TorchSCIState:
    """Cached state from a sci_solver call, sufficient to warm-start
    a Davidson run on any superset basis.

    Constructed automatically when ``probe_method='warm_start'`` is set
    on a sci_solver call. Pass back as ``probe_state=...`` on a
    subsequent call to enable warm-start.

    Fields are read-only by convention; user code should not mutate.
    Pickle-safe (the eigvec is kept on CPU; on GPU runs it's transferred
    at solve time).
    """

    # --- The basis the cached eigvec lives on ---
    adets: List[int]  # canonical (sorted+deduped)
    bdets: List[int]
    # --- The converged ground state on this basis ---
    # Stored in interleaved-Wick parity (the native Davidson form) and
    # on CPU for portability. Shape: (M_a * M_b,), normalized.
    eigvec_inter: Tensor  # CPU float64 (complex128 for a complex solve)
    energy: float
    # --- Provenance for validation on warm-extend ---
    n_orb: int
    nelec: Tuple[int, int]
    integrals_fingerprint: str  # hex SHA-256 of (h1, eri, ecore)
    solver_knobs: Dict[str, Any]  # SBD-parity-relevant knob values
    # --- Optional info, for diagnostics ---
    n_matvecs: Optional[int] = None
    final_residual: Optional[float] = None
    # --- Convenience cache fields (filled lazily, not by default) ---
    # Reserved for future tiers (e.g., helpers reuse). Tier 1 keeps
    # these empty.
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_basis(self) -> int:
        """N basis."""
        return len(self.adets) * len(self.bdets)

    @property
    def basis_fingerprint(self) -> str:
        """Basis fingerprint."""
        return _basis_fingerprint(self.adets, self.bdets)

    def __repr__(self) -> str:
        """Return the repr string."""
        return (
            f"TorchSCIState(M_a={len(self.adets)}, M_b={len(self.bdets)}, "
            f"N={self.n_basis:,}, E={self.energy:.10f}, "
            f"int_fp={self.integrals_fingerprint[:8]}..., "
            f"basis_fp={self.basis_fingerprint})"
        )


# ----------------------------------------------------------------------------
# Dedup & superset validation
# ----------------------------------------------------------------------------


def _dedup_and_check_superset(
    cached_adets: List[int],
    cached_bdets: List[int],
    new_adets_in: np.ndarray,
    new_bdets_in: np.ndarray,
) -> Tuple[List[int], List[int], Dict[str, int]]:
    """Compute the union basis in **sorted-int canonical order**.

    This is the same ordering that ``sorted(set(...))`` would give the
    cold-solve path — i.e., the basis layout that triggers the αβ-swap
    projection auto-enable (``torch.equal(adets_t, bdets_t)``) and that
    matches SBD's TPB ordering. Critically, this means a cold solve and
    a warm-extend on the same union basis end up running on identical
    H matrices with identical ordering.

    The previously-tried "preserve cached prefix" ordering was simpler
    for the eigvec embedding (a 2D paste) but caused a subtle bug: an
    αβ-symmetric basis with non-canonical ordering ``adets != bdets``
    (as ordered lists) silently bypassed the αβ-swap projection. The
    warm extension then converged to a different eigenvalue than the
    cold solve. Sorted-int canonical ordering eliminates this whole
    class of issue.

    Args:
        cached_adets, cached_bdets: B₀ canonical strings (sorted+deduped
            already, since that's how cold-solve canonicalizes).
        new_adets_in, new_bdets_in: caller-supplied strings for B (which
            must be a superset of B₀ as sets). Any ordering accepted.

    Returns:
        (union_adets, union_bdets, info_dict) where:
          - union_adets/bdets are ``sorted(set(...))`` of the union.
          - info_dict has counts of input duplicates, core-overlap, and
            newly added strings.

    Raises:
        ValueError: if cached B₀ is not a subset of the new basis.
    """
    cached_a_set = set(int(x) for x in cached_adets)
    cached_b_set = set(int(x) for x in cached_bdets)
    new_a_input = [int(x) for x in np.asarray(new_adets_in).ravel()]
    new_b_input = [int(x) for x in np.asarray(new_bdets_in).ravel()]

    # Count duplicates within the input itself (input lists with repeats).
    α_input_dups = len(new_a_input) - len(set(new_a_input))
    β_input_dups = len(new_b_input) - len(set(new_b_input))

    new_a_set = set(new_a_input)
    new_b_set = set(new_b_input)

    # Strict subset check: every cached string must appear in the input.
    missing_a = cached_a_set - new_a_set
    missing_b = cached_b_set - new_b_set
    if missing_a or missing_b:
        raise ValueError(
            f"Warm-start requires the new basis to be a superset of the "
            f"cached basis. Missing from input: "
            f"{len(missing_a)} α-string(s), {len(missing_b)} β-string(s)."
        )

    # Strings that the caller added (not in cached B₀).
    α_added_set = new_a_set - cached_a_set
    β_added_set = new_b_set - cached_b_set
    # Strings the caller included that are already in B₀ (true "duplicates"
    # with respect to the cached basis — not within the input itself).
    α_overlap = len(new_a_set & cached_a_set)
    β_overlap = len(new_b_set & cached_b_set)

    # Canonical sorted-int order for the union (matches cold-solve's
    # ``sorted(set(int(x) for x in ...))``).
    union_adets = sorted(cached_a_set | α_added_set)
    union_bdets = sorted(cached_b_set | β_added_set)

    info = {
        "α_added": len(α_added_set),
        "β_added": len(β_added_set),
        "α_in_input_overlapping_core": α_overlap,
        "β_in_input_overlapping_core": β_overlap,
        "α_dup_in_input": α_input_dups,
        "β_dup_in_input": β_input_dups,
        "cached_M_a": len(cached_adets),
        "cached_M_b": len(cached_bdets),
        "extended_M_a": len(union_adets),
        "extended_M_b": len(union_bdets),
    }
    return union_adets, union_bdets, info


def _emit_dedup_warning(info: Dict[str, int]) -> None:
    """Emit a warning about duplicates if any were found."""
    α_dup_in = info["α_dup_in_input"]
    β_dup_in = info["β_dup_in_input"]
    α_overlap = info["α_in_input_overlapping_core"]
    β_overlap = info["β_in_input_overlapping_core"]
    α_added = info["α_added"]
    β_added = info["β_added"]

    msgs: List[str] = []
    if α_dup_in > 0 or β_dup_in > 0:
        msgs.append(
            f"input contains duplicates: "
            f"{α_dup_in} α-string repeat(s), {β_dup_in} β-string repeat(s)"
        )
    # Caller-supplied strings that already exist in the cached basis.
    α_input_total = α_overlap + α_added
    β_input_total = β_overlap + β_added
    if α_overlap > 0 or β_overlap > 0:
        msgs.append(
            f"{α_overlap}/{α_input_total} α-string(s) already in core; "
            f"{β_overlap}/{β_input_total} β-string(s) already in core "
            f"(adding {α_added} new α, {β_added} new β)"
        )
    if msgs:
        warnings.warn(
            "warm_start: " + "; ".join(msgs),
            UserWarning,
            stacklevel=2,
        )


# ----------------------------------------------------------------------------
# Eigvec embedding
# ----------------------------------------------------------------------------


def _cached_beta_positions(cached_bdets: List[int], extended_bdets: List[int]) -> np.ndarray:
    """Extended-basis index of every cached β-string.

    Args:
        cached_bdets: β-strings of the cached basis.
        extended_bdets: β-strings of the extended basis.

    Returns:
        ``(M_b_old,)`` int64 positions.

    Raises:
        ValueError: a cached β-string is absent from the extended basis
            (warm-start requires the extended basis to be a superset).
    """
    pos_b_in_new: Dict[int, int] = {int(s): i for i, s in enumerate(extended_bdets)}
    idx_b_new = np.empty(len(cached_bdets), dtype=np.int64)
    for j, s in enumerate(cached_bdets):
        try:
            idx_b_new[j] = pos_b_in_new[int(s)]
        except KeyError:
            raise ValueError(
                f"cached β-string {int(s)} (index {j}) not found in "
                f"extended basis; warm-start requires superset."
            )
    return idx_b_new


def _owned_alpha_positions(
    cached_adets: List[int],
    extended_adets: List[int],
    row_start: int,
    row_stop: int,
) -> np.ndarray:
    """Cached-basis index of every owned extended α row, ``-1`` when absent.

    Args:
        cached_adets: α-strings of the cached basis.
        extended_adets: α-strings of the extended basis.
        row_start: first owned extended α row (inclusive).
        row_stop: one past the last owned extended α row.

    Returns:
        ``(row_stop - row_start,)`` int64 positions; an α-string absent from
        the cached basis stays ``-1``, so that row is left zero.
    """
    pos_a_in_cache = {int(s): i for i, s in enumerate(cached_adets)}
    idx_a_cache = np.full(row_stop - row_start, -1, dtype=np.int64)
    for k, i_a in enumerate(range(row_start, row_stop)):
        idx_a_cache[k] = pos_a_in_cache.get(int(extended_adets[i_a]), -1)
    return idx_a_cache


def _embed_eigvec_into_extended_basis(
    eigvec_inter_old: Tensor,  # (M_a_old * M_b_old,) CPU or device
    cached_adets: List[int],
    cached_bdets: List[int],
    extended_adets: List[int],
    extended_bdets: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Embed the cached eigvec into the extended basis as a Davidson seed.

    The extended basis can have ANY ordering of the strings (not just
    prefix-of-cached); the embedding is a 2D permutation paste:

        W[i_a, i_b] = eigvec_old[ π_a(i_a), π_b(i_b) ]   if both ∈ B₀
                    = 0                                     otherwise

    where π_a maps an index in extended_adets to the corresponding
    index in cached_adets when the string exists there. Same for π_b.

    Ground-state-overlap guarantee: warm-start callers normally augment
    the core basis with HF (via the ``include_hf_in_core`` default of
    the warm_start probe method). Provided the cached state contains
    HF, the embedded eigvec has nonzero amplitude on the HF
    determinant, which has substantial ground-state weight for
    closed-shell chemistry — protecting Davidson from converging to an
    excited state. This function makes no such guarantee on its own;
    it just embeds whatever was cached.

    Returns a (M_a × M_b,) tensor on the target device/dtype, normalized
    to unit norm.
    """
    M_a_old = len(cached_adets)
    M_b_old = len(cached_bdets)
    M_a = len(extended_adets)
    M_b = len(extended_bdets)

    # Position of each cached α-string in the extended basis.
    pos_a_in_new: Dict[int, int] = {int(s): i for i, s in enumerate(extended_adets)}

    # Permutations: idx_a_new[i] = position in extended basis where
    # cached_adets[i] lives. Same for β.
    idx_a_new = np.empty(M_a_old, dtype=np.int64)
    for i, s in enumerate(cached_adets):
        try:
            idx_a_new[i] = pos_a_in_new[int(s)]
        except KeyError:
            raise ValueError(
                f"cached α-string {int(s)} (index {i}) not found in "
                f"extended basis; warm-start requires superset."
            )
    idx_b_new = _cached_beta_positions(cached_bdets, extended_bdets)

    W = torch.zeros(M_a * M_b, dtype=dtype, device=device)
    eig2d_old = eigvec_inter_old.to(device=device, dtype=dtype).view(M_a_old, M_b_old)

    idx_a_t = torch.from_numpy(idx_a_new).to(device)
    idx_b_t = torch.from_numpy(idx_b_new).to(device)
    # Scatter: W_new[idx_a_new[i], idx_b_new[j]] = eig_old[i, j].
    W.view(M_a, M_b)[idx_a_t.unsqueeze(1), idx_b_t.unsqueeze(0)] = eig2d_old

    norm = torch.linalg.norm(W)
    if norm.item() > 0.0:
        W = W / norm
    return W


def _embed_eigvec_into_extended_basis_rows(
    eigvec_inter_old: Tensor,
    cached_adets: List[int],
    cached_bdets: List[int],
    extended_adets: List[int],
    extended_bdets: List[int],
    row_start: int,
    row_stop: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Rows ``[row_start, row_stop)`` of :func:`_embed_eigvec_into_extended_basis`.

    Row-restricted variant for distributed warm starts: each rank pastes only
    its own block of extended α rows, so no rank materializes the full
    ``(M_a, M_b)`` seed. The paste is

        W[i_a, i_b] = eigvec_old[π_a(i_a), π_b(i_b)]   if both strings ∈ B₀
                    = 0                                otherwise,

    where π maps an extended-basis index to its cached-basis index. Rows whose
    α-string is absent from the cached basis are zero.

    Unlike the full-vector function this returns the **unnormalized** rows.
    The caller must apply the *global* norm after stacking every rank's block
    (``‖W‖² = Σ_rank ‖W[r0:r1]‖²``, e.g. one all-reduce); dividing by a
    per-rank block norm would be wrong.

    Args:
        eigvec_inter_old: cached ``(M_a_old * M_b_old,)`` interleaved-Wick
            eigenvector.
        cached_adets: α-strings of the cached basis.
        cached_bdets: β-strings of the cached basis.
        extended_adets: α-strings of the extended basis.
        extended_bdets: β-strings of the extended basis.
        row_start: first owned extended α row (inclusive).
        row_stop: one past the last owned extended α row.
        device: target torch device.
        dtype: target dtype.

    Returns:
        ``(row_stop - row_start, M_b)`` tensor on ``device``/``dtype``,
        unnormalized.

    Raises:
        ValueError: if a cached β-string is absent from the extended basis
            (warm-start requires the extended basis to be a superset).
    """
    M_a_old = len(cached_adets)
    M_b_old = len(cached_bdets)
    M_b = len(extended_bdets)

    # Position of each cached β string in the extended basis (the β axis is
    # full and replicated on every rank).
    idx_b_new = _cached_beta_positions(cached_bdets, extended_bdets)

    # Position of each owned extended α string in the cached basis (-1 when
    # absent: that row stays zero).
    idx_a_cache = _owned_alpha_positions(cached_adets, extended_adets, row_start, row_stop)

    # Gather on the host and move only the owned rows to the device: the
    # cached vector is full-length by contract, and copying it to every rank's
    # GPU would reintroduce exactly the length-N tensor this variant avoids.
    eig2d_old = eigvec_inter_old.detach().to("cpu", dtype=dtype).view(M_a_old, M_b_old)
    W_rows = torch.zeros(row_stop - row_start, M_b, dtype=dtype)

    present = np.nonzero(idx_a_cache >= 0)[0]
    if present.size > 0:
        idx_a_t = torch.from_numpy(idx_a_cache[present])
        idx_b_t = torch.from_numpy(idx_b_new)
        rows_t = torch.from_numpy(present)
        # Gather the cached rows, then scatter their columns into the extended
        # β positions (idx_b_new), mirroring the full function's scatter.
        W_rows[rows_t.unsqueeze(1), idx_b_t.unsqueeze(0)] = eig2d_old[idx_a_t]
    return W_rows.to(device)


def _embed_eigvec_into_extended_basis_tile(
    eigvec_inter: Tensor,
    old_adets: List[int],
    old_bdets: List[int],
    adets: List[int],
    bdets: List[int],
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Tile ``[row_start:row_stop, col_start:col_stop]`` of the warm-start seed.

    Tile-restricted counterpart of :func:`_embed_eigvec_into_extended_basis_rows`
    for the 2-D partitioned distributed path: each rank pastes only its own
    rectangle, so no rank materializes the full ``(M_a, M_b)`` seed nor a
    full-width ``(local_rows, M_b)`` row block. The paste is

        W[i_a, i_b] = eigvec_old[π_a(i_a), π_b(i_b)]   if both strings ∈ B₀
                    = 0                                otherwise,

    where ``i_a`` and ``i_b`` run over the owned row and column ranges and π
    maps an extended-basis index to its cached-basis index. Rows and columns
    whose string is absent from the cached basis are zero.

    Both the cached vector and the Davidson seed are interleaved-Wick, so the
    paste carries no sign factor: the interleaved-to-block sign depends only
    on the (α, β) string pair, and every pasted position pairs the same two
    strings in the cached and extended bases. The convention bridge to
    spin-block happens after the solve, not on the seed.

    Notes:
        The return value is the row-major flattening of the rectangle,
        ``out[(i_a - row_start) * local_cols + (i_b - col_start)] = W[i_a, i_b]``
        with ``local_cols = col_stop - col_start``. It is **unnormalized** —
        the caller divides by the global norm
        ``‖W‖₂ = sqrt(Σ_rank ‖W_tile‖₂²)`` after assembling every rank's block;
        a per-tile norm would be wrong.

    Args:
        eigvec_inter: cached ``(M_a_old * M_b_old,)`` interleaved-Wick
            eigenvector.
        old_adets: α-strings of the cached basis.
        old_bdets: β-strings of the cached basis.
        adets: α-strings of the extended basis.
        bdets: β-strings of the extended basis.
        row_start: first owned extended α row (inclusive).
        row_stop: one past the last owned extended α row.
        col_start: first owned extended β column (inclusive).
        col_stop: one past the last owned extended β column.
        device: target torch device.
        dtype: target dtype.

    Returns:
        ``((row_stop - row_start) * (col_stop - col_start),)`` tensor on
        ``device``/``dtype``, unnormalized.

    Raises:
        ValueError: if a cached β-string is absent from the extended basis
            (warm-start requires the extended basis to be a superset).
    """
    M_a_old = len(old_adets)
    M_b_old = len(old_bdets)
    local_rows = row_stop - row_start
    local_cols = col_stop - col_start

    # Position of each cached β string in the extended basis (checked globally,
    # as in the row helper: a missing cached β would silently lose amplitude).
    idx_b_new = _cached_beta_positions(old_bdets, bdets)

    # Position of each owned extended α string in the cached basis (-1 when
    # absent: that row stays zero).
    idx_a_cache = _owned_alpha_positions(old_adets, adets, row_start, row_stop)

    eig2d_old = eigvec_inter.detach().to("cpu", dtype=dtype).view(M_a_old, M_b_old)
    W_tile = torch.zeros(local_rows, local_cols, dtype=dtype)

    rows = np.nonzero(idx_a_cache >= 0)[0]
    cols = np.nonzero((idx_b_new >= col_start) & (idx_b_new < col_stop))[0]
    if rows.size > 0 and cols.size > 0:
        rows_t = torch.from_numpy(rows)
        idx_a_t = torch.from_numpy(idx_a_cache[rows])
        cols_t = torch.from_numpy(cols)
        idx_b_local = torch.from_numpy(idx_b_new[cols] - col_start)
        # One advanced-index gather of the (present rows × owned columns)
        # sub-block, scattered into the local row-major tile layout.
        W_tile[rows_t.unsqueeze(1), idx_b_local.unsqueeze(0)] = eig2d_old[
            idx_a_t.unsqueeze(1), cols_t.unsqueeze(0)
        ]
    return W_tile.reshape(-1).to(device)


# ----------------------------------------------------------------------------
# Fingerprint validators
# ----------------------------------------------------------------------------


def _validate_fingerprints(
    state: TorchSCIState,
    h1: np.ndarray,
    eri: np.ndarray,
    n_orb: int,
    nelec: Tuple[int, int],
    current_knobs: Dict[str, Any],
    *,
    current_fingerprint: Optional[str] = None,
) -> None:
    """Compare cached fingerprints/knobs against current call. Raise on mismatch.

    If ``current_fingerprint`` is provided (a SHA-256 hex string already
    computed for the current ``(h1, eri)``), skip the in-function hash
    and compare directly. This is a substantial speedup for warm-extend
    on 36-orbital integrals (eri is 13 MB; hashing takes ~30 ms on a
    typical x86 core), since the factory-level integrals cache already
    knows the fingerprint of its stored arrays.

    When ``current_fingerprint`` is None, compute it locally.
    """
    fp_now = (
        current_fingerprint
        if current_fingerprint is not None
        else _integrals_fingerprint(h1, eri, 0.0)
    )
    if fp_now != state.integrals_fingerprint:
        raise ValueError(
            f"warm_start: integrals fingerprint mismatch.\n"
            f"  cached: {state.integrals_fingerprint}\n"
            f"  now:    {fp_now}\n"
            f"The Hamiltonian (h1, eri) differs from the one used to "
            f"build the cached state. Warm-start across different "
            f"Hamiltonians is not supported (the eigenvalue claim is "
            f"undefined). Run a cold solve instead."
        )
    if state.n_orb != n_orb:
        raise ValueError(f"warm_start: norb mismatch (cached {state.n_orb}, now {n_orb})")
    if tuple(state.nelec) != tuple(nelec):
        raise ValueError(f"warm_start: nelec mismatch (cached {state.nelec}, now {nelec})")
    # Hard knob mismatches are errors.
    cached_knobs = state.solver_knobs
    for field_name in _KNOB_FIELDS:
        if field_name in cached_knobs and field_name in current_knobs:
            if cached_knobs[field_name] != current_knobs[field_name]:
                raise ValueError(
                    f"warm_start: solver knob mismatch on '{field_name}': "
                    f"cached {cached_knobs[field_name]!r}, "
                    f"now {current_knobs[field_name]!r}. "
                    f"These knobs change the operator being diagonalized — "
                    f"the warm-extended energy would not match a cold solve."
                )
    # Soft mismatches → warning.
    for field_name in _KNOB_FIELDS_SOFT:
        if field_name in cached_knobs and field_name in current_knobs:
            if cached_knobs[field_name] != current_knobs[field_name]:
                warnings.warn(
                    f"warm_start: solver knob '{field_name}' differs from "
                    f"cached ({cached_knobs[field_name]!r} vs "
                    f"{current_knobs[field_name]!r}). The warm-extended "
                    f"energy will match a cold-solve at the CURRENT knob "
                    f"value, not the cached one.",
                    UserWarning,
                    stacklevel=3,
                )


def _hf_strings(n_orb: int, nelec: Tuple[int, int]) -> Tuple[int, int]:
    """Return the (HF α-string, HF β-string) bitmasks.

    For a system with ``(n_α, n_β)`` electrons in ``L`` spatial orbitals,
    HF occupies the lowest n_α α-orbitals and the lowest n_β β-orbitals:

        HF_α = (1 << n_α) - 1     # bits 0..n_α-1 set
        HF_β = (1 << n_β) - 1     # bits 0..n_β-1 set

    Same convention as SBD's reference and the matvec's
    interleaved-Wick encoding. Used by the warm-start path to guarantee
    a known-good ground-state-overlap component in the cached eigvec.

    Raises:
        ValueError if n_α > n_orb or n_β > n_orb.
    """
    n_a, n_b = nelec
    if n_a > n_orb or n_b > n_orb or n_a < 0 or n_b < 0:
        raise ValueError(
            f"HF construction failed: nelec=({n_a}, {n_b}) inconsistent with n_orb={n_orb}."
        )
    return (1 << n_a) - 1, (1 << n_b) - 1


def _augment_core_with_hf(
    adets_input: List[int],
    bdets_input: List[int],
    n_orb: int,
    nelec: Tuple[int, int],
) -> Tuple[List[int], List[int], Dict[str, bool]]:
    """Return (adets_aug, bdets_aug, info) with HF α/β strings included.

    Adds the HF α-string to ``adets_input`` if missing, and similarly
    for β. Returns the augmented sets in canonical sorted-int order
    (matches the cold-solve canonicalization).

    Default behavior of the warm-start probe method: ensures the cached
    eigvec has nonzero amplitude on the HF determinant, which carries
    substantial ground-state weight for closed-shell chemistry. Without
    this, Davidson seeded on a core that happens to exclude HF can
    converge to an excited state in the extended basis (a real failure
    mode — see warm-start tests `case_anchor3_random_split_50_50` for
    a reproducer of the unaugmented bug).

    Info dict reports whether HF α/β were actually added (vs already
    present), so the caller can emit a user-visible notice when
    augmentation happens.
    """
    hf_a, hf_b = _hf_strings(n_orb, nelec)
    a_set = set(int(x) for x in adets_input)
    b_set = set(int(x) for x in bdets_input)
    α_added = hf_a not in a_set
    β_added = hf_b not in b_set
    a_set.add(hf_a)
    b_set.add(hf_b)
    return (
        sorted(a_set),
        sorted(b_set),
        {
            "hf_alpha_string": hf_a,
            "hf_beta_string": hf_b,
            "hf_alpha_added": α_added,
            "hf_beta_added": β_added,
        },
    )
