"""Warm-start plumbing for pair-mode (no-product-space) sci_solver calls.

Pair-mode analog of :mod:`torch_sci_solver.torch_impl.probe`. Same
contract:

    Given solver knobs and pair bases P₀ ⊆ P (set inclusion on
    ``(α, β)`` tuples), a warm-extend call from a cached P₀-state
    reproduces a cold solve on P to within ``davidson_tolerance``.

The main differences from product-mode are structural:

  * The cached state stores parallel length-N arrays of α- and β-strings
    (``TorchPairSCIState``) rather than sorted-and-deduped per-spin
    lists — the pair basis is 1D and has no product structure.
  * The superset check is a set-containment test on ``(α, β)`` tuples;
    the extended basis's canonical order is lex-sorted (matches
    :func:`torch_sci_solver.torch_impl.pairs.pair_basis_from_arrays`).
  * Embedding the cached eigvec into the extended basis is a
    permuted-scatter into a length-N flat vector (no 2D paste).
  * HF augmentation adds a single ``(HF_α, HF_β)`` pair if missing.

Integrals fingerprinting and solver-knob validation reuse the
product-mode helpers (``_integrals_fingerprint`` / ``_validate_fingerprints``)
since those only touch scalar fields; ``TorchPairSCIState`` exposes the
same attribute names (``integrals_fingerprint``, ``solver_knobs``,
``n_orb``, ``nelec``) so the validator works unchanged.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.probe import _hf_strings

_UINT64_MASK = (1 << 64) - 1


def _signed_limb(value: int) -> int:
    """One unsigned 64-bit word as the signed int64 bit pattern torch stores."""
    word = int(value) & _UINT64_MASK
    return word - (1 << 64) if word >= (1 << 63) else word


def _pair_string_width(values: Any) -> int:
    """Limb width of one spin's pair-string sequence.

    A 2-D input already carries its limbs; a 1-D int64 bit-pattern array is one
    word; a sequence holding Python ints of arbitrary width needs
    ``ceil(bit_length / 64)`` words. Negative Python ints are the signed form of
    a single 64-bit word (the engine's convention), so they stay one wide.
    """
    if isinstance(values, Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        return int(arr.shape[1])
    if arr.ndim != 1:
        raise ValueError(f"pair strings must be 1-D or 2-D, got ndim={arr.ndim}")
    if arr.dtype.kind in "iub":
        return 1
    width = 1
    for value in np.asarray(values, dtype=object).ravel().tolist():
        number = int(value)
        if number >= 0:
            width = max(width, max(1, (number.bit_length() + 63) // 64))
    return width


def _pair_string_rows(values: Any, width: int, *, name: str = "pair strings") -> np.ndarray:
    """Normalize one spin's pair strings to an ``(N, width)`` limb-row matrix.

    Accepts the forms an explicit pair basis carries: Python ints of any width
    (split limb by limb), 1-D signed-int64 bit patterns, or an already-split
    ``(N, k)`` integer matrix (zero-padded on the high limbs to ``width``). The
    signed-int64 bit-pattern convention matches the pair kernels.

    Args:
        values: The α- or β-strings, in any of the forms above.
        width: Limb count of the returned rows.
        name: Label used in error messages.

    Returns:
        ``(N, width)`` int64 limb rows.
    """
    if isinstance(values, Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    if arr.ndim == 2:
        if arr.dtype.kind not in "iu":
            raise TypeError(f"{name} must be integers, got {arr.dtype}")
        if arr.shape[1] > width:
            raise ValueError(f"{name} carry {arr.shape[1]} limb(s), expected at most {width}")
        rows = np.zeros((arr.shape[0], int(width)), dtype=np.int64)
        rows[:, : arr.shape[1]] = arr.astype(np.int64, copy=False)
        return rows
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D or 2-D, got ndim={arr.ndim}")
    if arr.size == 0:
        return np.zeros((0, int(width)), dtype=np.int64)
    if arr.dtype.kind in "iu":
        # One signed-int64 bit pattern per string, occupying the low limb; the
        # high limbs are zero (the string only has bits in its own word).
        rows = np.zeros((arr.shape[0], int(width)), dtype=np.int64)
        rows[:, 0] = arr.astype(np.int64, copy=False)
        return rows
    rows = np.zeros((arr.shape[0], int(width)), dtype=np.int64)
    for index, value in enumerate(np.asarray(values, dtype=object).ravel().tolist()):
        number = int(value)
        if number < 0:
            # A negative Python int is the signed form of one 64-bit word.
            rows[index, 0] = _signed_limb(number)
            continue
        for limb in range(int(width)):
            rows[index, limb] = _signed_limb(number >> (64 * limb))
    return rows


def _rows_to_strings(rows: np.ndarray) -> List[Any]:
    """Inverse of :func:`_pair_string_rows`: flat ints, or limb rows when wide.

    A single-word row returns its Python int, keeping the narrow path's inputs
    (and therefore the canonical basis order) bit-identical.
    """
    if rows.shape[1] == 1:
        return [int(row[0]) for row in rows.tolist()]
    return [[int(limb) for limb in row] for row in rows.tolist()]


def _format_pair_string(row: Any) -> str:
    """Hex form of one limb row (or single-word int), for warm-start errors."""
    if isinstance(row, (int, np.integer)):
        return f"0x{int(row) & _UINT64_MASK:x}"
    value = 0
    for limb in reversed(list(row)):
        value = (value << 64) | (int(limb) & _UINT64_MASK)
    return f"0x{value:x}"


def _pair_uid_keys(
    pair_basis: Any,
    alpha: Any,
    beta: Any,
    *,
    device: torch.device,
) -> Tensor:
    """``alpha_uid * n_beta + beta_uid`` key of every pair in ``alpha``/``beta``.

    Resolves arbitrary pair strings — Python ints, 1-D signed-int64 bit
    patterns, or ``(N, n_limbs)`` limb rows — against the basis' unique-string
    axes, so one key space serves a single-word and a wide (multi-limb) basis.
    For an explicit basis the key is strictly increasing in canonical pair
    order (pairs are lex-sorted by uid), so callers can binary-search it.

    Args:
        pair_basis: Explicit :class:`PairBasis` or a
            :class:`StructuralPairBasis`.
        alpha, beta: The pair strings to key, in any accepted form.
        device: Device the key tensor is built on.

    Returns:
        ``(N,)`` int64 keys on ``device``.

    Raises:
        ValueError: a string is absent from the basis' unique axes.
    """
    from torch_sci_solver.torch_impl.pair_helpers import _dispatch_found_in, _sort_strings
    from torch_sci_solver.torch_impl.pairs import StructuralPairBasis

    unique_alpha = pair_basis.unique_alphas.to(device=device, dtype=torch.int64)
    unique_beta = pair_basis.unique_betas.to(device=device, dtype=torch.int64)
    width = int(unique_alpha.shape[-1]) if unique_alpha.dim() >= 2 else 1

    alpha_rows = torch.from_numpy(_pair_string_rows(alpha, width, name="α-strings")).to(device)
    beta_rows = torch.from_numpy(_pair_string_rows(beta, width, name="β-strings")).to(device)
    if width == 1:
        # The lookup helpers take a flat 1-D value vector for a single word.
        alpha_rows = alpha_rows.squeeze(-1)
        beta_rows = beta_rows.squeeze(-1)

    if isinstance(pair_basis, StructuralPairBasis):
        # Block-major unique arrays: sort a copy and keep the sorted-position
        # -> uid permutation, mirroring the superset check.
        sorted_alpha, alpha_perm = _sort_strings(unique_alpha)
        sorted_beta, beta_perm = _sort_strings(unique_beta)
        found_a, pos_a = _dispatch_found_in(sorted_alpha, alpha_rows)
        found_b, pos_b = _dispatch_found_in(sorted_beta, beta_rows)
        # A not-found query can report an index one past the end; clamp before
        # the permutation gather (the entry is discarded by the missing check).
        uid_a = alpha_perm[pos_a.clamp_max(max(int(sorted_alpha.shape[0]) - 1, 0))]
        uid_b = beta_perm[pos_b.clamp_max(max(int(sorted_beta.shape[0]) - 1, 0))]
    else:
        found_a, uid_a = _dispatch_found_in(unique_alpha, alpha_rows)
        found_b, uid_b = _dispatch_found_in(unique_beta, beta_rows)

    missing = ~(found_a & found_b)
    if bool(missing.any()):
        index = int(torch.nonzero(missing).reshape(-1)[0])
        label = "α" if not bool(found_a[index]) else "β"
        row = alpha_rows[index].tolist() if label == "α" else beta_rows[index].tolist()
        raise ValueError(
            f"pair warm-start input: {label}-string {_format_pair_string(row)} at "
            f"index {index} is not present in the basis' unique strings; warm-start "
            "requires a superset."
        )
    return uid_a * int(unique_beta.shape[0]) + uid_b


@dataclass
class TorchPairSCIState:
    """Cached state from a pair-mode sci_solver call.

    Pair-mode counterpart to :class:`torch_sci_solver.torch_impl.probe.TorchSCIState`.
    Constructed automatically when ``probe_method='warm_start'`` is set
    on a pair-mode call; pass back via ``probe_state=...`` on a
    subsequent call to enable warm-extend.
    """

    # --- The pair basis the cached eigvec lives on ---
    # Length-N, lex-sorted by (α, β). A single-word basis stores one Python int
    # (the signed-int64 bit pattern) per pair; a basis wider than 64 orbitals
    # stores each string as a list of ``n_limbs`` signed-int64 words.
    alpha_of_pair: List[Any]
    beta_of_pair: List[Any]
    # --- The converged ground state on this pair basis ---
    eigvec_inter: Tensor  # CPU float64 (complex128 for a complex solve), interleaved-Wick
    energy: float
    # --- Provenance ---
    n_orb: int
    nelec: Tuple[int, int]
    integrals_fingerprint: str
    solver_knobs: Dict[str, Any]
    # --- Diagnostics ---
    n_matvecs: Optional[int] = None
    final_residual: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_pairs(self) -> int:
        """N."""
        return len(self.alpha_of_pair)

    def __repr__(self) -> str:
        """Return the repr string."""
        return (
            f"TorchPairSCIState(N={self.n_pairs}, E={self.energy:.10f}, "
            f"int_fp={self.integrals_fingerprint[:8]}...)"
        )


def _dedup_and_check_pair_superset(
    cached_alpha: List[Any],
    cached_beta: List[Any],
    new_alpha_in: Any,
    new_beta_in: Any,
) -> Tuple[List[Any], List[Any], Dict[str, int]]:
    """Compute the union pair basis in lex-sorted canonical order.

    Pair-mode analog of
    :func:`torch_sci_solver.torch_impl.probe._dedup_and_check_superset`.
    The canonical order matches
    :func:`torch_sci_solver.torch_impl.pairs.pair_basis_from_arrays`
    (``np.unique`` over the ``(α, β)`` rows sorts lexicographically),
    so a cold pair-solve and a warm-extend on the same union pair basis
    run on identical H matrices with identical ordering.

    Every string is compared as its normalized limb row, so a cached state and
    an input list wider than 64 bits key exactly like a single-word one.

    Args:
        cached_alpha, cached_beta: cached pair basis, parallel sequences of
            length ``N₀`` (Python ints, or limb rows for a wide basis).
        new_alpha_in, new_beta_in: caller-supplied strings, parallel
            length arrays. May contain duplicates and may overlap with
            the cached set. Must contain every cached pair (superset
            requirement).

    Returns:
        ``(union_alpha, union_beta, info)`` with the union sorted
        lex on ``(α, β)`` and an info dict of counts (input duplicates,
        overlap with cached, newly added pairs, N₀ vs N_extended). The
        returned strings are Python ints for a single-word union and limb-row
        lists for a wide one.

    Raises:
        ValueError: if cached pair set is not a subset of the input.
    """
    width = max(
        _pair_string_width(cached_alpha),
        _pair_string_width(cached_beta),
        _pair_string_width(new_alpha_in),
        _pair_string_width(new_beta_in),
    )
    cached_a_rows = _pair_string_rows(cached_alpha, width, name="cached α-strings")
    cached_b_rows = _pair_string_rows(cached_beta, width, name="cached β-strings")
    new_alpha = _pair_string_rows(new_alpha_in, width, name="α-strings")
    new_beta = _pair_string_rows(new_beta_in, width, name="β-strings")
    if new_alpha.shape[0] != new_beta.shape[0]:
        raise ValueError(
            f"pair warm-start input: len(alpha)={new_alpha.shape[0]} but "
            f"len(beta)={new_beta.shape[0]}; parallel length required."
        )
    if width == 1:
        # Single-word fast path: key directly on the signed-int64 patterns,
        # matching the historical set/dedup ordering and memory profile.
        cached_pairs = {
            (int(a), int(b))
            for a, b in zip(cached_a_rows[:, 0].tolist(), cached_b_rows[:, 0].tolist())
        }
        input_pairs_list = [
            (int(a), int(b)) for a, b in zip(new_alpha[:, 0].tolist(), new_beta[:, 0].tolist())
        ]
    else:
        cached_pairs = {
            (tuple(a), tuple(b)) for a, b in zip(cached_a_rows.tolist(), cached_b_rows.tolist())
        }
        input_pairs_list = [
            (tuple(a), tuple(b)) for a, b in zip(new_alpha.tolist(), new_beta.tolist())
        ]
    input_dups = len(input_pairs_list) - len(set(input_pairs_list))
    new_pairs = set(input_pairs_list)
    missing = cached_pairs - new_pairs
    if missing:
        raise ValueError(
            f"Warm-start requires the new pair set to be a superset of the "
            f"cached pair set. Missing from input: {len(missing)} pair(s). "
            f"Cached size: {len(cached_pairs)}, input size: {len(new_pairs)}."
        )
    added = new_pairs - cached_pairs
    overlap = len(new_pairs & cached_pairs)
    union = sorted(cached_pairs | added)  # lex-sorted (α, β), limb 0 first
    if width == 1:
        union_alpha = [pair[0] for pair in union]
        union_beta = [pair[1] for pair in union]
    else:
        union_alpha = _rows_to_strings(np.asarray([pair[0] for pair in union], dtype=np.int64))
        union_beta = _rows_to_strings(np.asarray([pair[1] for pair in union], dtype=np.int64))
    info = {
        "pairs_added": len(added),
        "pairs_in_input_overlapping_core": overlap,
        "pairs_dup_in_input": input_dups,
        "cached_N": len(cached_pairs),
        "extended_N": len(union),
    }
    return union_alpha, union_beta, info


def _emit_pair_dedup_warning(info: Dict[str, int]) -> None:
    """Emit a UserWarning if the input pair list had duplicates or
    non-additive overlap with the cached core."""
    import warnings

    msgs: List[str] = []
    if info["pairs_dup_in_input"] > 0:
        msgs.append(f"input contains {info['pairs_dup_in_input']} duplicate pair(s)")
    if info["pairs_added"] == 0:
        msgs.append("input added no new pairs beyond the cached core")
    if msgs:
        warnings.warn(
            "pair warm-start dedup: " + "; ".join(msgs),
            UserWarning,
            stacklevel=3,
        )


def _embed_eigvec_pair_into_extended(
    eigvec_inter_old: Tensor,
    cached_alpha: List[Any],
    cached_beta: List[Any],
    extended_alpha: List[Any],
    extended_beta: List[Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Embed a cached pair-mode eigvec into an extended pair basis.

    Pair-mode analog of :func:`torch_sci_solver.torch_impl.probe._embed_eigvec_into_extended_basis`.
    The embedding is a 1D permuted scatter: for each cached pair
    ``(α_i, β_i)`` at cached position ``i``, place ``eigvec_old[i]`` at
    the position of the same pair in the extended basis; newly-added
    pairs get zero. Result is normalized.

    Pairs are keyed by the ``(α_uid, β_uid)`` key of their normalized limb
    rows, so a basis wider than 64 orbitals — whose strings may arrive as
    limb rows — keys exactly like a single-word one.

    Args:
        eigvec_inter_old: length-N₀ CPU or device tensor.
        cached_alpha, cached_beta: length-N₀ cached pair basis.
        extended_alpha, extended_beta: length-N_ext extended pair basis
            (must be a superset — every cached pair must appear).
        device, dtype: target tensor placement.

    Returns:
        Length-N_ext tensor on ``device``/``dtype``, unit-normalized.

    Raises:
        ValueError if any cached pair is missing from the extended basis.
    """
    width = max(
        _pair_string_width(cached_alpha),
        _pair_string_width(cached_beta),
        _pair_string_width(extended_alpha),
        _pair_string_width(extended_beta),
    )
    cached_a_rows = _pair_string_rows(cached_alpha, width, name="cached α-strings")
    cached_b_rows = _pair_string_rows(cached_beta, width, name="cached β-strings")
    extended_a_rows = _pair_string_rows(extended_alpha, width, name="extended α-strings")
    extended_b_rows = _pair_string_rows(extended_beta, width, name="extended β-strings")
    n_ext = int(extended_a_rows.shape[0])

    # Uid key of each extended pair: unique rows of the two axes, limb 0 most
    # significant, exactly the key the pair helpers use.
    unique_a, inv_a = np.unique(extended_a_rows, axis=0, return_inverse=True)
    unique_b, inv_b = np.unique(extended_b_rows, axis=0, return_inverse=True)
    ext_keys = inv_a.astype(np.int64) * int(unique_b.shape[0]) + inv_b.astype(np.int64)
    order = np.argsort(ext_keys, kind="stable")
    sorted_keys = ext_keys[order]

    pos_a = {tuple(row): i for i, row in enumerate(unique_a.tolist())}
    pos_b = {tuple(row): i for i, row in enumerate(unique_b.tolist())}
    cached_keys = np.empty(int(cached_a_rows.shape[0]), dtype=np.int64)
    for i, (a_row, b_row) in enumerate(zip(cached_a_rows.tolist(), cached_b_rows.tolist())):
        uid_a = pos_a.get(tuple(a_row))
        uid_b = pos_b.get(tuple(b_row))
        if uid_a is None or uid_b is None:
            raise ValueError(
                f"cached pair (α={_format_pair_string(a_row)}, "
                f"β={_format_pair_string(b_row)}) at index {i} not found in "
                "extended pair basis; warm-start requires a superset."
            )
        cached_keys[i] = uid_a * int(unique_b.shape[0]) + uid_b

    if n_ext:
        found_pos = np.searchsorted(sorted_keys, cached_keys)
        clipped = np.clip(found_pos, 0, n_ext - 1)
        missing = (found_pos >= n_ext) | (sorted_keys[clipped] != cached_keys)
    else:
        missing = np.ones(cached_keys.shape[0], dtype=bool)
    if bool(missing.any()):
        index = int(np.flatnonzero(missing)[0])
        raise ValueError(
            f"cached pair (α={_format_pair_string(cached_a_rows[index].tolist())}, "
            f"β={_format_pair_string(cached_b_rows[index].tolist())}) at index "
            f"{index} not found in extended pair basis; warm-start requires a "
            f"superset."
        )
    idx_new = np.asarray(order, dtype=np.int64)[found_pos]

    W = torch.zeros(n_ext, dtype=dtype, device=device)
    src = eigvec_inter_old.to(device=device, dtype=dtype)
    idx_t = torch.from_numpy(idx_new).to(device)
    W[idx_t] = src
    n = torch.linalg.norm(W)
    if n.item() > 0.0:
        W = W / n
    return W


def _augment_pair_with_hf(
    alpha_input: Any,
    beta_input: Any,
    n_orb: int,
    nelec: Tuple[int, int],
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Add the HF pair ``(HF_α, HF_β)`` to the input if missing.

    Pair-mode counterpart to
    :func:`torch_sci_solver.torch_impl.probe._augment_core_with_hf`.
    Unlike product mode — which adds HF_α to adets and HF_β to bdets
    *independently* — pair mode's HF augmentation is a single-pair
    insert. Returns the augmented lists in lex-sorted canonical order
    matching :func:`pair_basis_from_arrays`. Strings are normalized to limb
    rows first, so a wide (``n_orb > 64``) pair list is augmented like a
    single-word one.
    """
    hf_a, hf_b = _hf_strings(n_orb, nelec)
    width = max(
        _pair_string_width(alpha_input),
        _pair_string_width(beta_input),
        max(1, (int(n_orb) + 63) // 64),
    )
    alpha_rows = _pair_string_rows(alpha_input, width, name="α-strings")
    beta_rows = _pair_string_rows(beta_input, width, name="β-strings")
    pairs = set(zip(map(tuple, alpha_rows.tolist()), map(tuple, beta_rows.tolist())))
    hf_pair = (
        tuple(_pair_string_rows([hf_a], width, name="HF α-string")[0].tolist()),
        tuple(_pair_string_rows([hf_b], width, name="HF β-string")[0].tolist()),
    )
    added = hf_pair not in pairs
    pairs.add(hf_pair)
    sorted_pairs = sorted(pairs)
    aug_alpha = _rows_to_strings(np.asarray([pair[0] for pair in sorted_pairs], dtype=np.int64))
    aug_beta = _rows_to_strings(np.asarray([pair[1] for pair in sorted_pairs], dtype=np.int64))
    return (
        aug_alpha,
        aug_beta,
        {
            "hf_alpha_string": hf_a,
            "hf_beta_string": hf_b,
            "hf_pair_added": added,
        },
    )
