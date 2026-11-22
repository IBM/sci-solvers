"""Warm-start plumbing for the flat spin-orbital ("spinless") sci_solver path.

Spinless path analog of :mod:`torch_sci_solver.torch_impl.probe` and
:mod:`torch_sci_solver.torch_impl.probe_pairs`. The contract is the same:

    Given solver knobs and a determinant set B0 subset of B, a warm-extend
    call from a cached B0-state reproduces a cold solve on B to within
    ``davidson_tolerance``.

The structural differences follow from the basis: a spinless basis is a flat,
sorted list of ``N``-particle determinant bitmasks with no alpha/beta
factorization and no pair structure. A cached state therefore stores the
explicit determinant strings and one flat eigenvector (dim = N), the superset
test is a set-containment check over determinant bitmasks, and the embedding is
a 1-D string lookup.

Determinants can be wider than 64 bits. The cached strings are kept as Python
ints (object arrays at the API boundary), and every lookup packs them into the
same low-limb-first signed-int64 rows :class:`~
torch_sci_solver.torch_impl.spinless_helpers.SpinlessBasis` uses -- so string
equality is limb-aware rather than an int64 truncation.

There is deliberately no Hartree-Fock augmentation here. Product-mode
``warm_start`` adds the HF alpha/beta strings to the cached core to protect the
ground-state overlap; the spinless path has no alpha/beta split and therefore
no HF determinant to anchor. The cached state is the ground state of the basis
the caller passed, and a warm extension is only accepted on a superset of that
basis.
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

from torch_sci_solver.torch_impl.probe import _validate_fingerprints
from torch_sci_solver.torch_impl.spinless_helpers import (
    _n_limbs_for,
    _split_to_limbs,
    dets_to_limb_rows,
)

#: Physics limitation surfaced by ``preflight`` and ``capabilities()``.
SPINLESS_SPIN_SQ_LIMITATION = (
    "spin_sq is not supported on the spinless path: a target S^2 value is a "
    "target in the alpha/beta spin sector, and S^2 is built from the alpha/beta "
    "partition (N_alpha - N_beta). A flat spin-orbital basis has no alpha/beta "
    "split, so neither an S^2 operator nor a target spin sector is defined. Use "
    "a TPB or GDB (alpha, beta) basis for spin projection."
)

#: Physics limitation surfaced by ``preflight`` and ``capabilities()``.
SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION = (
    "compute_spin_squared is not supported on the spinless path: <S^2> is "
    "defined from the alpha/beta spin partition, which a flat spin-orbital "
    "basis does not have. Use a TPB or GDB (alpha, beta) basis to compute it."
)


@dataclass
class TorchSpinlessSCIState:
    """Cached state from a warm-start spinless (flat spin-orbital) solve.

    Constructed automatically when ``probe_method="warm_start"`` is set on a
    call whose ``nelec`` is a scalar (the spinless dispatch). Pass it back as
    ``probe_state=...`` on a later call whose determinant set is a superset of
    :attr:`dets` to seed Davidson from the cached ground state.

    Fields are read-only by convention; user code should not mutate. The
    object is pickle-safe: the eigenvector is kept on CPU, and the determinant
    strings are Python ints, so a basis wider than 64 spin-orbitals survives a
    round-trip through pickle or ``torch.save``.
    """

    # --- The basis the cached eigenvector lives on ---
    # Sorted in the engine's canonical spinless order: lexicographic on the
    # low-limb-first signed-int64 rows (``np.unique(..., axis=0)`` order).
    dets: List[int]
    # --- The converged ground state on this basis ---
    # Length ``len(dets)``, CPU float64 (complex128 for a complex solve). The
    # spinless path has no spin-block/interleaved sign convention, so the
    # values are the amplitudes as the solver uses them. The attribute name
    # matches the product/pair state types so the distributed hand-back
    # materializer recognizes a memmapped large-result vector.
    eigvec_inter: Tensor
    energy: float
    # --- Provenance for validation on warm-extend ---
    n_orb: int  # number of spin-orbitals (nmo), not spatial orbitals
    n_particles: int
    integrals_fingerprint: str  # hex SHA-256 of (h1, eri, ecore)
    solver_knobs: Dict[str, Any]
    # --- Optional diagnostics ---
    n_matvecs: Optional[int] = None
    final_residual: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_basis(self) -> int:
        """N --- the flat basis dimension."""
        return len(self.dets)

    @property
    def nelec(self) -> Tuple[int, int]:
        """``(N, 0)`` embedding used by the shared fingerprint validator."""
        return (int(self.n_particles), 0)

    @property
    def basis_fingerprint(self) -> str:
        """Short hash of the canonical limb rows, for diagnostics."""
        rows = dets_to_limb_rows(self.dets, self.n_orb)
        return hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()[:16]

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the cached state.

        The determinant strings themselves are summarized by
        :attr:`basis_fingerprint` rather than listed: a warm cache can hold
        millions of them, and the description is meant for logs, solver
        configuration dumps and capability reports.
        """
        eigvec = self.eigvec_inter
        return {
            "type": type(self).__name__,
            "n_basis": int(self.n_basis),
            "energy": float(self.energy),
            "n_orb": int(self.n_orb),
            "n_particles": int(self.n_particles),
            "integrals_fingerprint": self.integrals_fingerprint,
            "basis_fingerprint": self.basis_fingerprint,
            "solver_knobs": dict(self.solver_knobs),
            "n_matvecs": self.n_matvecs,
            "final_residual": self.final_residual,
            "eigvec_dtype": str(eigvec.dtype) if isinstance(eigvec, Tensor) else None,
        }

    def __repr__(self) -> str:
        """Return the repr string."""
        return (
            f"TorchSpinlessSCIState(N={self.n_basis:,}, "
            f"n_spinorb={self.n_orb}, N_elec={self.n_particles}, "
            f"E={self.energy:.10f}, "
            f"int_fp={self.integrals_fingerprint[:8]}..., "
            f"basis_fp={self.basis_fingerprint})"
        )


# ----------------------------------------------------------------------------
# Basis canonicalization, dedup and superset validation
# ----------------------------------------------------------------------------


def _as_determinant_ints(values: Any) -> List[int]:
    """Convert an iterable of determinant bitmasks to Python ints without rounding.

    A list holding a value at or above ``2**63`` makes ``np.asarray`` infer
    ``float64`` and silently round the bit pattern, so only integer-typed
    arrays take the fast path; everything else goes through an object array,
    which preserves arbitrary-precision ints.
    """
    try:
        arr = np.asarray(values)
    except (OverflowError, ValueError):
        arr = None
    if arr is not None and arr.dtype.kind in "iu":
        return [int(x) for x in arr.ravel().tolist()]
    return [int(x) for x in np.asarray(values, dtype=object).ravel().tolist()]


def _limb_key(det: int, n_limbs: int) -> Tuple[int, ...]:
    """Sort key matching the engine's low-limb-first signed-int64 row order."""
    return tuple(_split_to_limbs(int(det), n_limbs))


def _canonical_spinless_order(dets: Any, n_orb: int) -> List[int]:
    """Sort determinant strings into the order ``spinless_basis_from_array`` uses.

    ``SpinlessBasis`` is built with ``np.unique(..., axis=0)`` over the
    ``(N, n_limbs)`` signed-int64 limb matrix, which sorts lexicographically by
    limb 0 (the low 64 bits) first --- *not* by numeric determinant value once
    either the sign bit or a second limb is involved. Sorting with the same
    limb key makes a warm-extended union basis bit-identical to the basis a
    cold solve on the same strings builds.
    """
    n_limbs = _n_limbs_for(int(n_orb))
    return sorted({int(det) for det in dets}, key=lambda det: _limb_key(det, n_limbs))


def _dedup_and_check_spinless_superset(
    cached_dets: List[int],
    new_dets_in: Any,
    n_orb: int,
) -> Tuple[List[int], Dict[str, int]]:
    """Compute the union determinant basis and enforce ``cached subset new``.

    The canonical order matches ``spinless_basis_from_array`` (see
    :func:`_canonical_spinless_order`), so a cold spinless solve and a
    warm-extend on the same union basis run identical operators with identical
    ordering.

    Args:
        cached_dets: determinant strings of the cached basis, in the engine's
            canonical order.
        new_dets_in: caller-supplied determinant strings for this call. May
            contain duplicates and may overlap the cached set, but must contain
            every cached determinant.
        n_orb: number of spin-orbitals, used for the limb-aware sort key.

    Returns:
        ``(union_dets, info)`` with the union in canonical order. ``info``
        counts input duplicates, overlap with the cached core, newly added
        determinants, and the two basis sizes.

    Raises:
        ValueError: a cached determinant is absent from the input (a warm
            start requires the new basis to be a superset).
    """
    cached = [int(det) for det in cached_dets]
    new = _as_determinant_ints(new_dets_in)
    cached_set = set(cached)
    new_set = set(new)

    missing = cached_set - new_set
    if missing:
        n_limbs = _n_limbs_for(int(n_orb))
        sample = ", ".join(
            f"{det:#x}" for det in sorted(missing, key=lambda det: _limb_key(det, n_limbs))[:4]
        )
        more = "" if len(missing) <= 4 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Warm-start requires the new spinless determinant basis to be a "
            f"superset of the cached basis. Missing from input: {sample}{more}."
        )

    added = new_set - cached_set
    overlap = len(new_set & cached_set)
    union = _canonical_spinless_order(cached_set | new_set, n_orb)
    info = {
        "dets_added": len(added),
        "dets_in_input_overlapping_core": overlap,
        "dets_dup_in_input": len(new) - len(new_set),
        "cached_N": len(cached_set),
        "extended_N": len(union),
    }
    return union, info


def _emit_spinless_dedup_warning(info: Dict[str, int]) -> None:
    """Warn when the caller's input repeats a determinant or adds nothing new."""
    msgs: List[str] = []
    if info["dets_dup_in_input"] > 0:
        msgs.append(f"input contains {info['dets_dup_in_input']} duplicate determinant(s)")
    if info["dets_added"] == 0:
        msgs.append("input added no new determinants beyond the cached core")
    if msgs:
        warnings.warn(
            "spinless warm_start dedup: " + "; ".join(msgs),
            UserWarning,
            stacklevel=3,
        )


# ----------------------------------------------------------------------------
# Eigenvector embedding
# ----------------------------------------------------------------------------


def _cached_positions(
    cached_dets: List[int],
    target_dets: Any,
    n_orb: int,
) -> np.ndarray:
    """Position of each target determinant in the cached basis, or ``-1``.

    Matching is by limb rows, so it is exact for determinants wider than 64
    bits (two different Python ints never alias through an int64 cast). The
    first occurrence wins if the cached list repeats a determinant.
    """
    n_limbs = _n_limbs_for(int(n_orb))
    positions: Dict[Tuple[int, ...], int] = {}
    for index, det in enumerate(cached_dets):
        positions.setdefault(_limb_key(int(det), n_limbs), index)
    targets = _as_determinant_ints(target_dets)
    out = np.full(len(targets), -1, dtype=np.int64)
    for index, det in enumerate(targets):
        out[index] = positions.get(_limb_key(det, n_limbs), -1)
    return out


def _embed_eigvec_spinless(
    eigvec_old: Tensor,
    cached_dets: List[int],
    extended_dets: List[int],
    n_orb: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    indices: "Optional[Any]" = None,
    normalize: bool = True,
) -> Tensor:
    """Embed a cached spinless eigenvector into an extended determinant basis.

    For every extended-basis determinant that is also in the cached basis, the
    cached amplitude is copied to its new position; determinants added by the
    extension stay zero. The lookup is by determinant string (limb-aware), not
    by index, so any ordering of the extended list is accepted.

    Args:
        eigvec_old: cached ``(N0,)`` eigenvector (CPU or device).
        cached_dets: determinants of the cached basis, parallel to
            ``eigvec_old``.
        extended_dets: determinants of the extended basis. Must contain every
            cached determinant (the union check enforces this before the
            call).
        n_orb: number of spin-orbitals, used for the limb-aware lookup.
        device: target torch device.
        dtype: target dtype.
        indices: optional subset of extended-basis positions to embed (a
            distributed rank's owned rows or scattered tile). ``None`` embeds
            the whole basis.
        normalize: divide by the Euclidean norm when True (the full-vector
            path). A row-restricted embedding must pass ``False`` and apply the
            *global* norm after the ranks' blocks are assembled --- a per-rank
            norm would be wrong.

    Returns:
        A tensor of length ``len(extended_dets)`` (or ``len(indices)``) on
        ``device``/``dtype``.

    Raises:
        ValueError: the cached eigenvector length does not match the cached
            determinant list.
    """
    if indices is None:
        targets = [int(det) for det in extended_dets]
    else:
        index_list = indices.tolist() if isinstance(indices, Tensor) else list(indices)
        targets = [int(extended_dets[int(index)]) for index in index_list]

    positions = _cached_positions(cached_dets, targets, n_orb)
    source = eigvec_old.detach().to("cpu", dtype=dtype).reshape(-1)
    if int(source.numel()) != len(cached_dets):
        raise ValueError(
            f"cached spinless eigenvector has length {int(source.numel())} but the "
            f"cached basis has {len(cached_dets)} determinant(s)."
        )

    out = torch.zeros(len(targets), dtype=dtype)
    present = np.nonzero(positions >= 0)[0]
    if present.size > 0:
        out[torch.from_numpy(present)] = source[torch.from_numpy(positions[present])]
    out = out.to(device)
    if normalize:
        norm = torch.linalg.norm(out)
        if float(norm.item()) > 0.0:
            out = out / norm
    return out


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


def _as_spinless_state_list(
    probe_state_in: Any,
    n_subspaces: int,
) -> List[Optional[TorchSpinlessSCIState]]:
    """Normalize the per-call ``probe_state`` kwarg to one entry per subspace.

    A single :class:`TorchSpinlessSCIState` is the one-subspace shortcut; a
    list/tuple must have one entry (or ``None``) per subspace. A cached state
    from the product or pair path is rejected with a type error naming the
    expected type, rather than being treated as an iterable or silently
    ignored.

    Raises:
        TypeError: ``probe_state`` (or one of its entries) is not a
            :class:`TorchSpinlessSCIState`.
        ValueError: the list length does not match ``len(ci_strings)``.
    """
    if probe_state_in is None:
        return [None] * n_subspaces
    if isinstance(probe_state_in, TorchSpinlessSCIState):
        states: List[Any] = [probe_state_in]
    elif isinstance(probe_state_in, (list, tuple)):
        states = list(probe_state_in)
    else:
        raise TypeError(
            "spinless warm-start requires probe_state to be a "
            f"TorchSpinlessSCIState (got {type(probe_state_in).__name__!r}). "
            "Product-mode TorchSCIState and pair-mode TorchPairSCIState objects "
            "are not interchangeable with the flat spin-orbital path."
        )
    if len(states) != n_subspaces:
        raise ValueError(
            f"probe_state has length {len(states)} but ci_strings has {n_subspaces} subspace(s)."
        )
    for index, state in enumerate(states):
        if state is not None and not isinstance(state, TorchSpinlessSCIState):
            raise TypeError(
                f"spinless warm-start requires probe_state[{index}] to be a "
                f"TorchSpinlessSCIState (got {type(state).__name__!r}). "
                "Product-mode TorchSCIState and pair-mode TorchPairSCIState "
                "objects are not interchangeable with the flat spin-orbital path."
            )
    return states


def _validate_spinless_state(
    state: TorchSpinlessSCIState,
    h1: Optional[np.ndarray],
    eri: Optional[np.ndarray],
    n_orb: int,
    n_particles: int,
    current_knobs: Dict[str, Any],
    *,
    dtype: torch.dtype,
    current_fingerprint: Optional[str] = None,
) -> None:
    """Validate a cached spinless state against the current call.

    Checks the Hamiltonian fingerprint (when the integrals are available),
    ``n_orb``, particle number, the recorded Davidson knobs and the cached
    basis/eigenvector shape. A complex cache handed to a real solver is
    rejected rather than silently dropping its imaginary part during the
    dtype cast.

    Raises:
        TypeError: ``state`` is not a :class:`TorchSpinlessSCIState`, or it is
            complex while the current solver is real.
        ValueError: the fingerprint, orbital count, particle number, recorded
            knob (hard mismatch), basis shape or eigenvector length disagrees.
    """
    if not isinstance(state, TorchSpinlessSCIState):
        raise TypeError(
            "spinless warm-start requires probe_state to be a "
            f"TorchSpinlessSCIState (got {type(state).__name__!r})."
        )
    if h1 is not None and eri is not None:
        _validate_fingerprints(
            state,
            h1,
            eri,
            int(n_orb),
            (int(n_particles), 0),
            current_knobs,
            current_fingerprint=current_fingerprint,
        )
    else:
        if int(state.n_orb) != int(n_orb):
            raise ValueError(f"warm_start: norb mismatch (cached {state.n_orb}, now {n_orb})")
        if int(state.n_particles) != int(n_particles):
            raise ValueError(
                "warm_start: particle-number mismatch "
                f"(cached {state.n_particles}, now {n_particles})"
            )

    # Limb-aware packing raises for a negative or out-of-range determinant.
    dets_to_limb_rows(state.dets, int(n_orb))
    bad_popcount = [det for det in state.dets if int(det).bit_count() != int(n_particles)]
    if bad_popcount:
        raise ValueError(
            "warm_start: cached determinant(s) do not have the current particle "
            f"number {n_particles} (e.g. {int(bad_popcount[0]):#x})."
        )
    eigvec = state.eigvec_inter
    if isinstance(eigvec, Tensor):
        if int(eigvec.numel()) != len(state.dets):
            raise ValueError(
                "warm_start: cached spinless eigenvector length "
                f"{int(eigvec.numel())} does not match the cached basis size "
                f"{len(state.dets)}."
            )
        if eigvec.is_complex() and not dtype.is_complex:
            raise TypeError(
                "warm_start: the cached spinless state is complex but this solver "
                f"was constructed with dtype={dtype} (real). Pass a complex dtype "
                "to solve a complex spinless Hamiltonian — the cached phases are "
                "never silently dropped."
            )
