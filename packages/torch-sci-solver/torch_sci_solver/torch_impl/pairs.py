"""Pair-basis (no-product-space) primitives for the selected-CI engine.

Product mode — the historical default — stores the wavefunction over the
Cartesian product ``{(α_i, β_j) : i ∈ [0, M_α), j ∈ [0, M_β)}`` linearly
indexed as ``k = i * M_β + j`` (dim = M_α * M_β). Pair mode instead stores
the wavefunction over an explicit list of N (α, β) pairs (dim = N).

This module defines the data structures needed by the pair-mode diagonal
and matvec kernels, plus the length-N initial-vector helper. The kernels
themselves live next to their product-mode counterparts (``mult.py``,
``rdm.py``, ``spin_sq.py``) and consume a ``PairBasis`` where they would
otherwise use ``(adets_t, bdets_t)`` and the Cartesian-product index math.

The pair basis exposes three views over the same N pairs:

  * ``alpha_of_pair[k]``, ``beta_of_pair[k]`` — the α- and β-strings
    (as ``int64`` bitmasks) of pair ``k``. Parallel arrays of length N.
  * ``unique_alphas``, ``unique_betas`` — the sorted, deduplicated
    sets of α- and β-strings referenced by any pair. The existing
    per-spin excitation-table builder (``sbd_ref.helpers.generate_excitation``)
    is fed these directly.
  * ``alpha_uid_of_pair[k]``, ``beta_uid_of_pair[k]`` — the position of
    pair ``k``'s α- and β-strings in the unique arrays. This is the
    bridge that lets a per-spin excitation table (indexed by
    uid-in-unique-alphas) be re-projected onto pair space.

Two additional lookup structures are precomputed for the kernels:

  * ``partners_of_alpha`` (CSR) — for each unique α-string, the list of
    pair indices sharing that α. Used by α-only excitation kernels.
  * ``partners_of_beta`` (CSR) — the mirror for β. Used by β-only kernels.

Both are needed at kernel time to answer "given the ket α (or β) of an
excitation, which pair indices in this basis are the matching bra?"
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch
from torch_sci_solver.torch_impl.helpers import _as_string_limbs, _n_limbs


@dataclass(frozen=True)
class PairSCIState:
    """Pair-mode analog of ``qiskit_addon_sqd.fermion.SCIState``.

    ``SCIState`` requires ``amplitudes.shape == (len(ci_strs_a), len(ci_strs_b))``
    — the Cartesian product layout. Pair mode's basis is a length-N list of
    (α, β) pairs; the eigenvector is a length-N 1D vector and the α- and
    β-string arrays are parallel length-N arrays. Storing that in an
    ``SCIState`` fails the 2D shape check, so pair mode returns this
    lookalike dataclass instead. Same public fields (``amplitudes``,
    ``ci_strs_a``, ``ci_strs_b``, ``norb``, ``nelec``); consumers reading
    them by name see equivalent data. Consumers that reshape into 2D
    would already have been wrong on a pair basis — they'll now fail at
    the type check rather than silently mis-index.
    """

    amplitudes: "np.ndarray"
    ci_strs_a: "np.ndarray"
    ci_strs_b: "np.ndarray"
    norb: int
    nelec: "tuple[int, int]"

    def save(self, filename):
        """Save the PairSCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs_a=self.ci_strs_a,
            ci_strs_b=self.ci_strs_b,
            norb=self.norb,
            nelec=self.nelec,
        )


@dataclass(frozen=True)
class PairBasis:
    """An explicit list of (α, β) determinant pairs.

    All tensors live on the same device and are int64. Pair indices run
    over ``[0, N)`` where ``N = alpha_of_pair.shape[0]``.

    For bases with ``n_orb <= 64`` the string fields are 1-D ``(N,)`` tensors.
    For wider bases (``n_orb > 64``) they are ``(N, n_limbs)`` limb matrices
    (limb ``k`` holds bits ``[64k, 64k+64)``). The same applies to the unique
    string arrays: ``unique_alphas`` / ``unique_betas`` are either 1-D ``(M,)``
    or 2-D ``(M, n_limbs)``.

    Fields:
        alpha_of_pair, beta_of_pair: length-N int64 tensors of α- and β-string
            bitmasks (1-D or ``(N, n_limbs)``). ``(alpha_of_pair[k],
            beta_of_pair[k])`` is pair k.
        unique_alphas, unique_betas: sorted unique α- and β-string bitmasks
            (1-D ``(M,)`` or 2-D ``(M, n_limbs)``).
        alpha_uid_of_pair, beta_uid_of_pair: length-N int64. Position of pair
            k's α- and β-strings in ``unique_alphas`` / ``unique_betas``.
        partners_of_alpha_offsets: length ``M_α_unique + 1`` int64 CSR offsets.
        partners_of_alpha_pair_idx: length-N int64. For unique-α index ``a``,
            ``pair_idx[offsets[a] : offsets[a+1]]`` is the list of pair
            indices whose α equals ``unique_alphas[a]``. Sorted by pair index.
        partners_of_alpha_beta_uid: length-N int64. Same slicing — the β-uid
            of each partner pair. Kept alongside so β-lookups on the ket side
            (intersection with ket-α's partners) can be done in uid space.
        partners_of_beta_offsets, partners_of_beta_pair_idx,
        partners_of_beta_alpha_uid: mirror for β.

    Convention:
        In product mode the analogous data is implicit: the pair set is the
        full Cartesian product, ``alpha_of_pair[k] = adets[k // M_β]``,
        ``beta_of_pair[k] = bdets[k % M_β]``, and ``partners_of_alpha[a]``
        is simply ``[a * M_β + j for j in range(M_β)]``. Pair mode makes the
        set explicit and drops the implicit ``M_α × M_β`` grid.
    """

    alpha_of_pair: Tensor
    beta_of_pair: Tensor
    unique_alphas: Tensor
    unique_betas: Tensor
    alpha_uid_of_pair: Tensor
    beta_uid_of_pair: Tensor
    partners_of_alpha_offsets: Tensor
    partners_of_alpha_pair_idx: Tensor
    partners_of_alpha_beta_uid: Tensor
    partners_of_beta_offsets: Tensor
    partners_of_beta_pair_idx: Tensor
    partners_of_beta_alpha_uid: Tensor

    @property
    def n_limbs(self) -> int:
        """Number of 64-bit limbs per string (1 for n_orb <= 64)."""
        return int(self.alpha_of_pair.shape[-1]) if self.alpha_of_pair.dim() >= 2 else 1

    @property
    def n_pairs(self) -> int:
        """N — the flat basis dimension in pair mode."""
        return int(self.alpha_of_pair.shape[0])

    @property
    def n_unique_alphas(self) -> int:
        """Number of distinct α-strings appearing in the pair set."""
        return int(self.unique_alphas.shape[0])

    @property
    def n_unique_betas(self) -> int:
        """Number of distinct β-strings appearing in the pair set."""
        return int(self.unique_betas.shape[0])

    @property
    def device(self) -> torch.device:
        """Torch device shared by all fields."""
        return self.alpha_of_pair.device


@dataclass(frozen=True)
class StructuralPairBlock:
    """One Cartesian block of a :class:`StructuralPairBasis`.

    The block is ``unique_alphas[alpha_start:alpha_stop] ×
    unique_betas[beta_start:beta_stop]``. Its pairs occupy the contiguous
    global pair-index range ``[pair_base, pair_base + n_pairs)`` in row-major
    (α-major) order, so pair ``(i, j)`` of the grid has index
    ``pair_base + (i - alpha_start) * n_beta + (j - beta_start)``.

    Attributes:
        alpha_start: First α uid of the block (inclusive).
        alpha_stop: One past the last α uid of the block.
        beta_start: First β uid of the block (inclusive).
        beta_stop: One past the last β uid of the block.
        pair_base: Global index of the block's first pair.
    """

    alpha_start: int
    alpha_stop: int
    beta_start: int
    beta_stop: int
    pair_base: int

    @property
    def n_alpha(self) -> int:
        """Number of α strings in the block."""
        return int(self.alpha_stop) - int(self.alpha_start)

    @property
    def n_beta(self) -> int:
        """Number of β strings in the block."""
        return int(self.beta_stop) - int(self.beta_start)

    @property
    def n_pairs(self) -> int:
        """Number of pairs in the block."""
        return self.n_alpha * self.n_beta


@dataclass(frozen=True)
class StructuralPairBasis:
    """A pair basis described by Cartesian blocks instead of a pair list.

    The basis is an ordered, pair-disjoint union of
    :class:`StructuralPairBlock` products. Only the unique α/β bitmask arrays
    (``O(M_unique)``) and a small per-string block-membership table are stored
    — no length-``N`` array and no partners CSR. It is therefore the
    representation to use when the flat pair list would not fit. The block
    matvec, the serial RDM builders and the tiled distributed post-processing
    all walk its blocks; only the streamed table fallback still needs explicit
    per-pair arrays and rejects it.

    Blocks may share strings (a string can appear in several blocks), but the
    block ranges are contiguous over the unique arrays and the pair sets are
    disjoint. A pair index is therefore resolved by locating its block and
    doing the grid arithmetic above.

    Attributes:
        blocks: The blocks, in global pair order.
        unique_alphas, unique_betas: int64 bitmask arrays, either 1-D
            ``(M_unique,)`` single words or ``(M_unique, n_limbs)`` limb
            matrices when a block string needs more than 64 bits (limb ``k``
            holds bits ``[64k, 64k + 64)``). Each block's slice is sorted by
            bitmask, but the arrays as a whole are block-major, so they need
            not be globally sorted.
        n_pairs: Total number of pairs (the flat dimension).
        pair_bases: ``(n_blocks,)`` first global pair index of each block,
            strictly increasing.
        alpha_block_ids, beta_block_ids: ``(M_unique, K)`` block ids of each
            string, padded with ``-1``. ``K`` is the largest number of blocks
            containing one string.
    """

    blocks: "tuple[StructuralPairBlock, ...]"
    unique_alphas: Tensor
    unique_betas: Tensor
    n_pairs: int
    pair_bases: Tensor
    alpha_block_ids: Tensor
    beta_block_ids: Tensor

    @property
    def n_unique_alphas(self) -> int:
        """Number of unique α bitmasks stored."""
        return int(self.unique_alphas.shape[0])

    @property
    def n_unique_betas(self) -> int:
        """Number of unique β bitmasks stored."""
        return int(self.unique_betas.shape[0])

    @property
    def n_blocks(self) -> int:
        """Number of Cartesian blocks."""
        return len(self.blocks)

    @property
    def n_limbs(self) -> int:
        """Number of 64-bit limbs per unique string (1 for single-word bases)."""
        return int(self.unique_alphas.shape[-1]) if self.unique_alphas.dim() >= 2 else 1

    @property
    def device(self) -> torch.device:
        """Torch device shared by all fields."""
        return self.unique_alphas.device

    def block_of_pair(self, pair_index: Tensor) -> Tensor:
        """Block id of each global pair index."""
        bases = self.pair_bases.to(pair_index.device)
        block = torch.searchsorted(bases, pair_index.to(torch.int64), right=True) - 1
        return block.clamp(min=0)

    def _block_geometry(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """``(pair_base, n_alpha, n_beta, alpha_start, beta_start)`` per block."""
        device = self.device
        alpha_start = torch.tensor(
            [b.alpha_start for b in self.blocks], dtype=torch.int64, device=device
        )
        beta_start = torch.tensor(
            [b.beta_start for b in self.blocks], dtype=torch.int64, device=device
        )
        n_alpha = torch.tensor([b.n_alpha for b in self.blocks], dtype=torch.int64, device=device)
        n_beta = torch.tensor([b.n_beta for b in self.blocks], dtype=torch.int64, device=device)
        return self.pair_bases.to(device), n_alpha, n_beta, alpha_start, beta_start

    def pair_uids(self, pair_index: Tensor) -> tuple[Tensor, Tensor]:
        """``(α uid, β uid)`` of each global pair index."""
        pair_index = pair_index.to(self.device).to(torch.int64)
        pair_base, n_alpha, n_beta, alpha_start, beta_start = self._block_geometry()
        block = self.block_of_pair(pair_index)
        local = pair_index - pair_base[block]
        row = torch.div(local, n_beta[block], rounding_mode="floor")
        col = local - row * n_beta[block]
        return alpha_start[block] + row, beta_start[block] + col

    def pair_strings(self, pair_index: Tensor) -> tuple[Tensor, Tensor]:
        """``(α bitmask, β bitmask)`` of each global pair index."""
        alpha_uid, beta_uid = self.pair_uids(pair_index)
        return self.unique_alphas[alpha_uid], self.unique_betas[beta_uid]

    def pair_index_of_uids(self, alpha_uid: Tensor, beta_uid: Tensor) -> tuple[Tensor, Tensor]:
        """Locate pairs from their ``(α uid, β uid)`` coordinates.

        Args:
            alpha_uid, beta_uid: parallel ``(n,)`` uid tensors.

        Returns:
            ``(found, pair_index)``. ``found[i]`` is False when the uid pair is
            not in the basis; ``pair_index`` is then meaningless (clamped to a
            valid block) and must be ignored.
        """
        device = self.device
        alpha_uid = alpha_uid.to(device).to(torch.int64)
        beta_uid = beta_uid.to(device).to(torch.int64)
        pair_base, n_alpha, n_beta, alpha_start, beta_start = self._block_geometry()

        def _locate(a_uids: Tensor, b_uids: Tensor) -> "tuple[Tensor, Tensor]":
            """One chunk: match each α membership slot against the β slots."""
            a_ids = self.alpha_block_ids[a_uids]  # (n, Ka)
            b_ids = self.beta_block_ids[b_uids]  # (n, Kb)
            found = torch.zeros(a_uids.shape[0], dtype=torch.bool, device=device)
            block = torch.zeros_like(a_uids)
            for slot in range(a_ids.shape[1]):
                candidate = a_ids[:, slot]
                hit = ((b_ids == candidate.unsqueeze(1)).any(dim=1)) & (candidate >= 0)
                block = torch.where(hit, candidate, block)
                found = found | hit
            index = (
                pair_base[block]
                + (a_uids - alpha_start[block]) * n_beta[block]
                + (b_uids - beta_start[block])
            )
            return found, index

        # Chunk the lookup so the (n, Kb) membership workspace stays bounded
        # for the large owned index sets an owner-computes projector passes in.
        chunk = 1 << 22
        found_parts: list[Tensor] = []
        index_parts: list[Tensor] = []
        for start in range(0, alpha_uid.shape[0], chunk):
            found, index = _locate(
                alpha_uid[start : start + chunk], beta_uid[start : start + chunk]
            )
            found_parts.append(found)
            index_parts.append(index)
        if not found_parts:
            empty = alpha_uid.new_empty(0)
            return empty.to(torch.bool), empty
        return torch.cat(found_parts), torch.cat(index_parts)


def structural_pair_basis_from_blocks(
    blocks: "Sequence[tuple[Iterable[int], Iterable[int]]]",
    *,
    n_orb: int | None = None,
    device: torch.device | str = "cpu",
) -> StructuralPairBasis:
    """Build a :class:`StructuralPairBasis` from Cartesian block string sets.

    Each block is given as two iterables of α- and β-string bitmasks; the
    block is their full Cartesian product. Duplicate strings inside one block
    are collapsed. Blocks must be pair-disjoint, which is checked with the
    sufficient condition that two blocks either have disjoint α string sets
    or disjoint β string sets.

    The unique strings are renumbered block-major so every block covers a
    contiguous uid range: new strings of a block are appended in ascending
    order, and a block whose strings are already present must already form a
    contiguous range (appending new strings is allowed only when that range
    ends at the current end of the axis). A block layout that cannot be
    ordered this way is rejected rather than silently materialised.

    Strings may be Python ints of any width, 1-D integer arrays, or
    ``(M, n_limbs)`` limb matrices. A block whose strings need more than 64
    bits (or that is passed already split) makes the basis store its unique
    arrays as ``(M_unique, n_limbs)`` limb matrices, limb ``k`` holding bits
    ``[64k, 64k + 64)``; every other axis is zero-padded to the same width.
    A basis whose strings all fit in one word keeps the legacy 1-D arrays
    unless ``n_orb`` asks for more limbs.

    Args:
        blocks: One ``(alpha_strings, beta_strings)`` pair per block, in
            global pair order.
        n_orb: optional number of spatial orbitals. When given, the basis is
            widened to ``ceil(n_orb / 64)`` limbs even if every block string
            fits in one word, matching the explicit pair basis' rule. This
            matters above 64 orbitals with a selected space that happens to
            occupy low orbitals only.
        device: Torch device for the returned tensors.

    Returns:
        The structural basis, with the pair index of block ``b``'s grid
        position ``(i, j)`` equal to
        ``pair_base + (i - alpha_start) * n_beta + (j - beta_start)``.

    Raises:
        ValueError: no blocks, an empty block, a pair-overlapping block pair,
            or a block layout that has no contiguous-range renumbering.
        TypeError: a block string array is not integer-typed.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if len(blocks) == 0:
        raise ValueError("structural_pair_basis_from_blocks: need at least one block")

    parsed_alpha: list[np.ndarray] = []
    parsed_beta: list[np.ndarray] = []
    forced_limbs = False
    for index, (alpha, beta) in enumerate(blocks):
        a_np, a_split = _block_string_rows(alpha)
        b_np, b_split = _block_string_rows(beta)
        if a_np.shape[0] == 0 or b_np.shape[0] == 0:
            raise ValueError(f"block {index} is empty on at least one spin")
        forced_limbs = forced_limbs or a_split or b_split
        parsed_alpha.append(a_np)
        parsed_beta.append(b_np)

    n_limbs = max(matrix.shape[1] for matrix in parsed_alpha + parsed_beta)
    if n_orb is not None:
        n_limbs = max(n_limbs, _n_limbs(int(n_orb)))
    alpha_sets = [_pad_limbs(matrix, n_limbs) for matrix in parsed_alpha]
    beta_sets = [_pad_limbs(matrix, n_limbs) for matrix in parsed_beta]
    keep_limbs = forced_limbs or n_limbs > 1

    # Pair-disjointness: two blocks that share an α string and a β string
    # would share the pair. Disjointness on either axis is sufficient.
    a_keys = [set(_row_keys(matrix)) for matrix in alpha_sets]
    b_keys = [set(_row_keys(matrix)) for matrix in beta_sets]
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if a_keys[i].isdisjoint(a_keys[j]) or b_keys[i].isdisjoint(b_keys[j]):
                continue
            raise ValueError(
                f"blocks {i} and {j} share both an α and a β string, so they overlap "
                "as pair sets; structural blocks must be pair-disjoint"
            )

    alpha_order, alpha_ranges = _structural_axis_order(alpha_sets, "α")
    beta_order, beta_ranges = _structural_axis_order(beta_sets, "β")

    pair_base = 0
    out_blocks: list[StructuralPairBlock] = []
    for (a0, a1), (b0, b1) in zip(alpha_ranges, beta_ranges):
        out_blocks.append(
            StructuralPairBlock(
                alpha_start=a0,
                alpha_stop=a1,
                beta_start=b0,
                beta_stop=b1,
                pair_base=pair_base,
            )
        )
        pair_base += (a1 - a0) * (b1 - b0)

    alpha_ids = _structural_membership(alpha_ranges, alpha_order.shape[0])
    beta_ids = _structural_membership(beta_ranges, beta_order.shape[0])

    def _tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(device)

    def _strings(array: np.ndarray) -> np.ndarray:
        return array if keep_limbs else array[:, 0]

    return StructuralPairBasis(
        blocks=tuple(out_blocks),
        unique_alphas=_tensor(_strings(alpha_order)),
        unique_betas=_tensor(_strings(beta_order)),
        n_pairs=int(pair_base),
        pair_bases=_tensor(np.array([b.pair_base for b in out_blocks], dtype=np.int64)),
        alpha_block_ids=_tensor(alpha_ids),
        beta_block_ids=_tensor(beta_ids),
    )


def _block_string_rows(strings) -> tuple[np.ndarray, bool]:
    """Sorted unique ``(M, K)`` signed-int64 limb rows of one block's strings.

    A 1-D input carries one single-word bit pattern per string: an integer
    array keeps its 64-bit pattern (negative entries are the signed form of a
    top-bit-set word), and Python ints of any width are split into as many
    zero-extended limbs as they need. A 2-D input is already a limb matrix and
    keeps its width. Rows are returned in the signed-int64 bit-pattern form
    the pair kernels store and deduplicated with limb 0 compared first.

    Args:
        strings: One block's string iterable.

    Returns:
        ``(rows, was_split)`` where ``was_split`` is True when the input was
        already a 2-D limb matrix.

    Raises:
        TypeError: the array is not integer-typed.
        ValueError: the array is not 1-D or 2-D.
    """
    if isinstance(strings, np.ndarray):
        arr = strings
    else:
        values = list(strings)
        try:
            arr = np.asarray(values)
        except (OverflowError, ValueError):
            arr = np.asarray([int(v) for v in values], dtype=object)
        # Numpy widens Python ints that do not fit a machine word to float64,
        # which would silently round them; reparse the exact originals.
        if (
            arr.dtype.kind == "f"
            and values
            and all(isinstance(v, (int, np.integer)) for v in values)
        ):
            arr = np.asarray([int(v) for v in values], dtype=object)
    was_split = arr.ndim == 2
    if arr.size == 0:
        width = int(arr.shape[1]) if was_split else 1
        return np.zeros((0, width), dtype=np.int64), was_split
    if arr.ndim == 2:
        if arr.dtype.kind not in "iu":
            raise TypeError(f"determinant strings must be integers, got {arr.dtype}")
        words = np.ascontiguousarray(arr.astype(np.uint64, copy=True))
        return _unique_rows(words.view(np.int64)), True
    if arr.ndim != 1:
        raise ValueError(f"determinant strings must be 1-D or 2-D, got ndim={arr.ndim}")
    if arr.dtype == object:
        values = [int(v) for v in arr]
        n_limbs = max(
            1,
            max(
                (1 if v < 0 else max(1, (v.bit_length() + 63) // 64) for v in values),
            ),
        )
        wide = np.zeros((len(values), n_limbs), dtype=np.uint64)
        for k in range(n_limbs):
            shift = 64 * k
            wide[:, k] = [
                (((v if v >= 0 else v & _UINT64_WORD_MASK) >> shift) & _UINT64_WORD_MASK)
                for v in values
            ]
        return _unique_rows(wide.view(np.int64)), n_limbs > 1
    if arr.dtype.kind not in "iu":
        raise TypeError(f"determinant strings must be integers, got {arr.dtype}")
    words = np.ascontiguousarray(arr.astype(np.uint64, copy=True)).reshape(-1, 1)
    return _unique_rows(words.view(np.int64)), False


def _unique_rows(rows: np.ndarray) -> np.ndarray:
    """Unique rows of an ``(M, K)`` int64 matrix, sorted with limb 0 first."""
    view_dtype = np.dtype([(f"limb{k}", np.int64) for k in range(rows.shape[1])])
    unique = np.unique(rows.view(view_dtype).ravel())
    return unique.view(np.int64).reshape(-1, rows.shape[1])


def _pad_limbs(rows: np.ndarray, n_limbs: int) -> np.ndarray:
    """Zero-extend ``(M, K)`` limbs to ``(M, n_limbs)``, ``K <= n_limbs``."""
    if rows.shape[1] == n_limbs:
        return rows
    padded = np.zeros((rows.shape[0], n_limbs), dtype=np.int64)
    padded[:, : rows.shape[1]] = rows
    return padded


def _row_keys(rows: np.ndarray) -> list[tuple[int, ...]]:
    """Hashable per-row keys of an ``(M, K)`` int64 limb matrix."""
    return [tuple(int(v) for v in row) for row in rows.tolist()]


def _structural_axis_order(
    block_sets: "Sequence[np.ndarray]",
    axis: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Renumber one spin's strings so every block is a contiguous uid range.

    Args:
        block_sets: Sorted unique ``(M, n_limbs)`` limb rows of each block on
            this axis (limb 0 compared first).
        axis: ``"α"`` or ``"β"``, for error messages.

    Returns:
        ``(strings_in_uid_order, [(start, stop), ...])``. The string array is
        ``(M_unique, n_limbs)`` int64; the caller drops the limb axis for a
        single-word basis.

    Raises:
        ValueError: a block cannot be placed as a contiguous range.
    """
    order: list[tuple[int, ...]] = []
    position: dict[tuple[int, ...], int] = {}
    ranges: list[tuple[int, int]] = []
    for index, rows in enumerate(block_sets):
        values = _row_keys(rows)
        existing = sorted(position[v] for v in values if v in position)
        new_values = sorted(v for v in values if v not in position)
        if existing:
            if existing[-1] - existing[0] + 1 != len(existing):
                raise ValueError(
                    f"block {index}'s {axis} strings cannot be renumbered into one "
                    "contiguous uid range"
                )
            if new_values and existing[-1] != len(order) - 1:
                raise ValueError(
                    f"block {index} mixes existing and new {axis} strings but its "
                    "existing strings do not end the current uid order"
                )
        for value in new_values:
            position[value] = len(order)
            order.append(value)
        placed = sorted(position[v] for v in values)
        if placed[-1] - placed[0] + 1 != len(placed):
            raise ValueError(
                f"block {index}'s {axis} strings cannot be renumbered into one contiguous uid range"
            )
        ranges.append((placed[0], placed[-1] + 1))
    n_limbs = int(block_sets[0].shape[1])
    strings = np.array(order, dtype=np.int64).reshape(len(order), n_limbs)
    return strings, ranges


def _structural_membership(
    ranges: "Sequence[tuple[int, int]]",
    n_strings: int,
) -> np.ndarray:
    """Padded block-id table: ``(n_strings, max_blocks_per_string)``, ``-1`` padded."""
    counts = np.zeros(n_strings, dtype=np.int64)
    for start, stop in ranges:
        counts[start:stop] += 1
    width = max(1, int(counts.max()) if counts.size else 1)
    table = np.full((n_strings, width), -1, dtype=np.int64)
    fill = np.zeros(n_strings, dtype=np.int64)
    for block, (start, stop) in enumerate(ranges):
        uids = np.arange(start, stop, dtype=np.int64)
        table[uids, fill[start:stop]] = block
        fill[start:stop] += 1
    return table


def _unsigned_words(arr: np.ndarray) -> np.ndarray:
    """Reinterpret an integer array as ``uint64``, preserving its bit pattern."""
    if arr.dtype == np.uint64:
        return arr
    if arr.dtype == np.int64:
        return arr.view(np.uint64)
    return arr.astype(np.uint64)


def _infer_n_orb(*arrays: np.ndarray) -> int:
    """Infer ``n_orb`` as one past the highest set bit across the input strings.

    Accepts the same forms :func:`pair_basis_from_arrays` does: Python-int
    sequences (``object`` arrays), 1-D bit patterns, and ``(M, n_limbs)``
    limb matrices.
    """
    max_bits = 0
    for arr in arrays:
        if arr.size == 0:
            continue
        if arr.ndim == 2:
            words = _unsigned_words(arr)
            for k in range(words.shape[1] - 1, -1, -1):
                top = int(words[:, k].max())
                if top:
                    max_bits = max(max_bits, 64 * k + top.bit_length())
                    break
        elif arr.ndim == 1:
            if arr.dtype == object:
                max_bits = max(max_bits, max((int(v).bit_length() for v in arr), default=0))
            elif arr.dtype.kind in "iu":
                max_bits = max(max_bits, int(_unsigned_words(arr).max()).bit_length())
            else:
                raise TypeError(f"determinant strings must be integers, got {arr.dtype}")
        else:
            raise ValueError(f"determinant strings must be 1-D or 2-D, got ndim={arr.ndim}")
    return max_bits


def _limb_matrix_from_strings(strings, n_orb: int) -> np.ndarray:
    """Signed-int64 ``(M, n_limbs)`` limb matrix from any accepted string form.

    Delegates the parsing and the limb-count guard to
    :func:`~torch_sci_solver.torch_impl.helpers._as_string_limbs` (the same
    normalization the product-mode path uses), zero-extending a 1-D input that
    carries only its low limb, then reinterprets the result with the
    signed-int64 bit-pattern convention the pair kernels expect.
    """
    limbs = _as_string_limbs(strings, n_orb)
    n_limbs = _n_limbs(n_orb)
    if limbs.shape[1] < n_limbs:
        padded = np.zeros((limbs.shape[0], n_limbs), dtype=np.uint64)
        padded[:, : limbs.shape[1]] = limbs
        limbs = padded
    return np.ascontiguousarray(limbs).view(np.int64)


def _lexsort_limbs(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Lexicographic sort of two limb matrices ``(N, n_limbs)``.

    Returns the indices that would sort by ``(primary, secondary)`` with
    limb 0 compared first — the same order ``torch.unique(dim=0)`` and
    :func:`~torch_sci_solver.torch_impl.pair_helpers._found_in_multilimb`
    use, and the plain signed-int64 order when there is one limb.
    """
    # np.lexsort sorts by the last key first, so list the keys in reverse.
    keys = [secondary[:, k] for k in range(secondary.shape[1] - 1, -1, -1)] + [
        primary[:, k] for k in range(primary.shape[1] - 1, -1, -1)
    ]
    return np.lexsort(keys)


def _unique_limbs(limbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows of a limb matrix, returning (unique, inverse_indices).

    ``limbs`` is ``(M, n_limbs)`` int64. Returns unique rows in ascending
    lexicographic order (limb 0 most significant) and the inverse mapping.
    """
    # Use void dtype trick for row-wise unique on structured limb data.
    view_dtype = np.dtype([(f"limb{k}", np.int64) for k in range(limbs.shape[1])])
    view = limbs.view(view_dtype).ravel()
    unique_view, inverse = np.unique(view, return_inverse=True)
    unique = unique_view.view(limbs.dtype).reshape(-1, limbs.shape[1])
    return unique, inverse


def pair_basis_from_arrays(
    strings_a,
    strings_b,
    *,
    n_orb: int | None = None,
    device: torch.device | str = "cpu",
) -> PairBasis:
    """Build a :class:`PairBasis` from two length-N arrays of α- and β-strings.

    Semantics: each index ``i`` defines one pair ``(strings_a[i], strings_b[i])``.
    Duplicate pairs (same α AND same β) are collapsed and the pair set is
    sorted lexicographically by ``(α, β)``. This mirrors product mode's per-
    spin sort — the basis order is a deterministic function of the input,
    not of insertion order.

    Strings wider than 64 bits are carried as ``(M, n_limbs)`` limb matrices
    (limb ``k`` holds bits ``[64k, 64k+64)``). The function infers the limb
    count from the input data (any string >= 2^64) or from ``n_orb`` if given.
    A wide base returns 2-D string fields; a one-word base keeps the legacy
    1-D fields unless the caller explicitly passes a 2-D ``(M, 1)`` matrix.

    Args:
        strings_a, strings_b: parallel iterables of α- and β-string int
            bitmasks. May be Python ints of any width, 1-D arrays, or
            ``(M, n_limbs)`` limb matrices. Must have equal length after
            conversion to arrays.
        n_orb: number of spatial orbitals. If ``None``, inferred from the
            maximum bit position present in the input strings.
        device: torch device for the returned tensors.

    Returns:
        A :class:`PairBasis` on ``device``.

    Raises:
        ValueError: if ``strings_a`` and ``strings_b`` have different lengths,
            if the pair set is empty after dedup, if the two spins mix a 1-D
            bitmask array with a 2-D limb matrix, or if a limb matrix has the
            wrong number of limbs for ``n_orb``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    a_np_raw = np.asarray(strings_a)
    b_np_raw = np.asarray(strings_b)
    if a_np_raw.shape[0] != b_np_raw.shape[0]:
        raise ValueError(
            f"pair_basis_from_arrays: strings_a and strings_b must have equal "
            f"length in pair mode (got {a_np_raw.shape[0]} vs {b_np_raw.shape[0]}); "
            f"each index i defines one (α_i, β_i) pair."
        )
    if a_np_raw.shape[0] == 0:
        raise ValueError("pair_basis_from_arrays: empty pair list.")
    if a_np_raw.ndim != b_np_raw.ndim:
        raise ValueError(
            "pair_basis_from_arrays: α- and β-strings mix a 1-D bitmask array with a "
            "2-D limb matrix; pass both spins in the same representation"
        )

    if n_orb is None:
        n_orb = _infer_n_orb(a_np_raw, b_np_raw)

    # A forced ``(M, 1)`` input keeps its limb axis; a 1-D input below the
    # multi-limb threshold keeps the legacy 1-D representation bit for bit.
    keep_limbs = a_np_raw.ndim == 2 or n_orb > 64

    a_limbs = _limb_matrix_from_strings(a_np_raw, n_orb)
    b_limbs = _limb_matrix_from_strings(b_np_raw, n_orb)

    # Dedup + sort by (α, β) lexicographically, limb 0 compared first.
    order = _lexsort_limbs(a_limbs, b_limbs)
    a_limbs = a_limbs[order]
    b_limbs = b_limbs[order]
    if a_limbs.shape[0] > 1:
        same = np.all(a_limbs[1:] == a_limbs[:-1], axis=1) & np.all(
            b_limbs[1:] == b_limbs[:-1], axis=1
        )
        keep = np.concatenate((np.ones(1, dtype=bool), ~same))
        a_limbs = a_limbs[keep]
        b_limbs = b_limbs[keep]

    n_pairs = a_limbs.shape[0]

    # Uniques + uid maps (per spin, row-wise unique on limb matrices).
    unique_alphas_np, alpha_uid = _unique_limbs(a_limbs)
    unique_betas_np, beta_uid = _unique_limbs(b_limbs)
    alpha_uid = alpha_uid.astype(np.int64, copy=False)
    beta_uid = beta_uid.astype(np.int64, copy=False)

    # Partners CSR: for each unique-α, list the pair indices sharing that α.
    # Since we sorted pairs by (α, β) and α_uid tracks that same order,
    # partners for a given α are already contiguous — no explicit groupby.
    m_alpha = unique_alphas_np.shape[0]
    partners_alpha_offsets_np = np.zeros(m_alpha + 1, dtype=np.int64)
    partners_alpha_pair_idx_np = np.empty(n_pairs, dtype=np.int64)
    partners_alpha_beta_uid_np = np.empty(n_pairs, dtype=np.int64)
    _fill_partners_csr(
        alpha_uid,
        beta_uid,
        partners_alpha_offsets_np,
        partners_alpha_pair_idx_np,
        partners_alpha_beta_uid_np,
    )

    # Partners CSR keyed by β. Requires sorting by β first — build via argsort.
    m_beta = unique_betas_np.shape[0]
    order_by_beta = _lexsort_limbs(b_limbs, a_limbs)
    partners_beta_offsets_np = np.zeros(m_beta + 1, dtype=np.int64)
    partners_beta_pair_idx_np = order_by_beta.astype(np.int64, copy=False)
    partners_beta_alpha_uid_np = alpha_uid[order_by_beta]
    _fill_partners_offsets(beta_uid[order_by_beta], m_beta, partners_beta_offsets_np)

    def to_dev(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device)

    def strings_out(arr):
        return to_dev(arr if keep_limbs else arr[:, 0])

    return PairBasis(
        alpha_of_pair=strings_out(a_limbs),
        beta_of_pair=strings_out(b_limbs),
        unique_alphas=strings_out(unique_alphas_np),
        unique_betas=strings_out(unique_betas_np),
        alpha_uid_of_pair=to_dev(alpha_uid),
        beta_uid_of_pair=to_dev(beta_uid),
        partners_of_alpha_offsets=to_dev(partners_alpha_offsets_np),
        partners_of_alpha_pair_idx=to_dev(partners_alpha_pair_idx_np),
        partners_of_alpha_beta_uid=to_dev(partners_alpha_beta_uid_np),
        partners_of_beta_offsets=to_dev(partners_beta_offsets_np),
        partners_of_beta_pair_idx=to_dev(partners_beta_pair_idx_np),
        partners_of_beta_alpha_uid=to_dev(partners_beta_alpha_uid_np),
    )


def basis_init_vector_pair_torch(
    n_pairs: int,
    init: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> Tensor:
    """Initial wavefunction over an N-pair basis.

    Pair-mode analog of ``sbd_ref.davidson.basis_init_vector`` — the same
    ``init=0`` (unit vector at index 0) / ``init=1`` (random) semantics
    applied to a length-N vector instead of a length-``M_α × M_β`` one. The
    flat length-``n_pairs`` vector is the ``1 × n_pairs`` product, so this
    delegates to :func:`basis_init_vector_torch`.

    Args:
        n_pairs: N — number of pairs in the basis.
        init: 0 = seed at index 0 (matches product-mode HF-seed convention
            once pair 0 is the smallest-int pair after the canonical sort),
            1 = random unit vector.
        device: torch device.
        dtype: float dtype.
        seed: optional seed for ``init=1``.
    """
    return basis_init_vector_torch(1, n_pairs, init, device=device, dtype=dtype, seed=seed)


def pair_basis_from_full_product(
    adets,
    bdets,
    *,
    n_orb: int | None = None,
    device: torch.device | str = "cpu",
) -> PairBasis:
    """Construct the pair basis equivalent to a full ``adets × bdets`` product.

    This is the "same-answer-as-product-mode" reference basis used by the
    full-product correctness test — building it and running pair-mode on it must
    yield the same energy and eigenvector as running product-mode on
    ``(adets, bdets)`` directly.

    Args:
        adets, bdets: sorted iterables of α- and β-string bitmasks. May be
            Python ints of any width, 1-D arrays, or ``(M, n_limbs)`` limb
            matrices.
        n_orb: number of spatial orbitals. If ``None``, inferred from the
            maximum bit position present in the input strings.
        device: torch device.
    """
    a_np_raw = np.asarray(adets)
    b_np_raw = np.asarray(bdets)
    if a_np_raw.ndim != b_np_raw.ndim:
        raise ValueError(
            "pair_basis_from_full_product: α- and β-strings mix a 1-D bitmask array "
            "with a 2-D limb matrix; pass both spins in the same representation"
        )

    if n_orb is None:
        n_orb = _infer_n_orb(a_np_raw, b_np_raw)

    keep_limbs = a_np_raw.ndim == 2 or n_orb > 64
    a_limbs = _limb_matrix_from_strings(a_np_raw, n_orb)
    b_limbs = _limb_matrix_from_strings(b_np_raw, n_orb)

    # Form the Cartesian product, α-major as in product mode.
    m_alpha = a_limbs.shape[0]
    m_beta = b_limbs.shape[0]
    if keep_limbs:
        a_grid = np.repeat(a_limbs, m_beta, axis=0)
        b_grid = np.tile(b_limbs, (m_alpha, 1))
    else:
        a_grid = np.repeat(a_limbs[:, 0], m_beta)
        b_grid = np.tile(b_limbs[:, 0], m_alpha)
    return pair_basis_from_arrays(a_grid, b_grid, n_orb=n_orb, device=device)


def pair_superset_embedding(
    pair_basis: PairBasis, *, n_orb: int | None = None
) -> tuple[PairBasis, Tensor, Tensor]:
    """Embed an explicit pair basis in the Cartesian product of its strings.

    A recovered/expanded pair list is usually an irregular subset of the
    product of its unique α and β strings, so each string has its own partner
    set and the exact Cartesian block decomposition cannot group anything. The
    full product always decomposes. Solving the *masked* product problem (zero
    on the pairs outside the list) has the same eigenpairs on the list, and
    lets the block and tile kernels run on their native structure.

    Args:
        pair_basis: the explicit pair basis.
        n_orb: number of spatial orbitals. Defaults to the basis' limb width
            (``64 * n_limbs``) for a wide basis, so a wide basis whose strings
            only use low orbitals keeps its limb layout; a single-word basis
            infers it from the strings. Pass the solver's ``norb`` to pin it.

    Returns:
        ``(superset, positions, active)``: the full product of the basis's
        unique strings (same lex order as :func:`pair_basis_from_full_product`),
        the superset indices of the basis's pairs in basis order, and the
        superset-length membership mask.
    """
    if n_orb is None and pair_basis.n_limbs > 1:
        n_orb = 64 * pair_basis.n_limbs
    # The product constructor works in numpy; a CUDA/MPS basis must be staged
    # through the host, and the result is moved back to the basis' device.
    superset = pair_basis_from_full_product(
        pair_basis.unique_alphas.detach().to(device="cpu"),
        pair_basis.unique_betas.detach().to(device="cpu"),
        n_orb=n_orb,
        device=pair_basis.device,
    )
    m_beta = int(pair_basis.n_unique_betas)
    positions = pair_basis.alpha_uid_of_pair.to(torch.int64) * m_beta + (
        pair_basis.beta_uid_of_pair.to(torch.int64)
    )
    active = torch.zeros(int(superset.n_pairs), dtype=torch.bool, device=positions.device)
    active[positions] = True
    return superset, positions, active


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


_INT64_MAX_UNSIGNED = 1 << 63
_INT64_OFFSET = 1 << 64
_UINT64_WORD_MASK = (1 << 64) - 1


def _to_signed_int64(v: int) -> int:
    """Convert an unsigned Python int in ``[0, 2^64)`` to signed int64 bit
    pattern. Values in ``[0, 2^63)`` pass through unchanged; values in
    ``[2^63, 2^64)`` wrap to their signed representation. Matches the
    convention used by ``sbdiag._to_int64_python`` for adets/bdets.
    """
    return v if v < _INT64_MAX_UNSIGNED else v - _INT64_OFFSET


def _fill_partners_offsets(sorted_uid: np.ndarray, n_unique: int, offsets_out: np.ndarray) -> None:
    """Fill CSR offsets given a sorted uid array. ``offsets_out`` is
    length ``n_unique + 1``, pre-zeroed."""
    if sorted_uid.shape[0] == 0:
        return
    counts = np.bincount(sorted_uid, minlength=n_unique)
    np.cumsum(counts, out=offsets_out[1:])


def _fill_partners_csr(
    alpha_uid: np.ndarray,
    beta_uid: np.ndarray,
    offsets_out: np.ndarray,
    pair_idx_out: np.ndarray,
    beta_uid_out: np.ndarray,
) -> None:
    """Fill the α-keyed partners CSR from pairs already sorted by α_uid.

    Because ``pair_basis_from_arrays`` sorts pairs lexicographically by
    ``(α, β)``, both ``alpha_uid`` and (within each α-block) ``beta_uid``
    are already in ascending order. The pair-idx list per α is simply the
    contiguous range ``[offsets[a], offsets[a+1])``.
    """
    n_pairs = alpha_uid.shape[0]
    m_alpha = offsets_out.shape[0] - 1
    _fill_partners_offsets(alpha_uid, m_alpha, offsets_out)
    pair_idx_out[:] = np.arange(n_pairs, dtype=np.int64)
    beta_uid_out[:] = beta_uid
