"""Exact Cartesian-block decomposition of an explicit ``(α, β)`` pair basis.

A pair basis is a set of determinants ``(a, b)``; it is a *product basis* exactly
when it equals ``A × B`` for some string sets. Selected-CI bases are often a
disjoint union of such products — the scaling benchmark builds them that way —
but the solver stores only the flat, lex-sorted pair list.

This module recovers an exact decomposition without storing anything of size
``nnz(H)``. Grouping the β strings by their **identical α-partner set** gives
groups ``U_g × B_g`` that are disjoint in β and whose union is the pair set:
for a β string ``b``, its partners ``U(b) = {a : (a, b) ∈ S}`` are read from the
existing β CSR, and two β strings with equal partner sets form one group. The
mirror grouping (α strings by β-partner set) gives disjoint-in-α groups
``A_h × V_h``.

Why this is exact
-----------------
For group ``g`` with shared partner set ``U_g`` and member set ``B_g``, every
``(a, b)`` with ``a ∈ U_g`` and ``b ∈ B_g`` is in ``S``: ``a ∈ U_g`` means
``a`` is a partner of every member β, so ``(a, b)`` is present. Groups are
disjoint because each β uid belongs to exactly one member set. Hence
``S = ⊔_g U_g × B_g`` with no membership test inside a group. When the
partition is degenerate (every group a singleton) the decomposition is exact
but useless, and :func:`build_pair_blocks` returns ``None`` under ``"auto"``.

Detection is fingerprint-bucketed and then **verified exactly**: two int64
rolling sums over the partner uids only suggest equality; every multi-member
bucket is compared slice by slice before it is accepted. A fingerprint
collision splits nothing silently — the build fails and the caller keeps the
streamed path.

Block-major pair order
----------------------
The pair list is lex-sorted by ``(α_uid, β_uid)``, so within a group the
ascending pair indices are exactly the row-major order over
``(alpha_uids sorted, beta_uids sorted)``. ``_Group.pair_idx`` stores them in
that order; the matvec uses ``row * |beta_uids| + col`` as a block position.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import dataclasses
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis

# Two odd constants for the rolling-set fingerprints. Multiplication in int64
# wraps; two independent odd multipliers make a collision require equal
# multiset sums under both, and the exact verification below catches any
# remaining collision.
_FP_C1 = 0x9E3779B97F4A7C15
_FP_C2 = 0xC2B2AE3D27D4EB4F

# Elements compared per verification tile (a bucket's slices are gathered in
# tiles of at most this many int64 values).
_VERIFY_ELEMENTS = 1 << 22


@dataclasses.dataclass(frozen=True)
class _SubBlock:
    """One coarse block's rectangle inside a structural group's grid.

    Attributes:
        block: Index of the coarse block in ``StructuralPairBasis.blocks``.
        row_start, row_stop: Row range in the group's α-uid grid.
        col_start, col_stop: Column range in the group's β-uid grid.
    """

    block: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    @property
    def n_rows(self) -> int:
        """Rows of the group grid covered by the sub-block."""
        return self.row_stop - self.row_start

    @property
    def n_cols(self) -> int:
        """Columns of the group grid covered by the sub-block."""
        return self.col_stop - self.col_start


@dataclasses.dataclass(frozen=True)
class _Group:
    """One Cartesian block ``alpha_uids × beta_uids`` of the pair set.

    Attributes:
        alpha_uids: Sorted α uids in the block (the row axis).
        beta_uids: Sorted β uids in the block (the column axis).
        pair_idx: Global pair indices in row-major (α-major) order, strictly
            increasing; length ``len(alpha_uids) * len(beta_uids)``. ``None``
            for a structural group, whose pair indices are computed from
            ``row_base`` (or ``col_base``/``col_stride``) on demand.
        row_base: Structural α-group layout: global index of grid position
            ``(r, 0)`` minus the β uid of that column, so the index of
            ``(r, c)`` is ``row_base[r] + beta_uids[c]``. ``None`` otherwise.
        col_base: Structural β-group layout: ``col_base[c] + alpha_uids[r] *
            col_stride[c]`` is the index of ``(r, c)``. ``None`` otherwise.
        col_stride: Per-column α stride of the structural β-group layout.
        sub_blocks: Coarse-block rectangles covering the grid, structural only.
    """

    alpha_uids: Tensor
    beta_uids: Tensor
    pair_idx: Optional[Tensor] = None
    row_base: Optional[Tensor] = None
    col_base: Optional[Tensor] = None
    col_stride: Optional[Tensor] = None
    sub_blocks: "tuple[_SubBlock, ...]" = ()

    @property
    def n_rows(self) -> int:
        """Rows of the group grid."""
        return int(self.alpha_uids.numel())

    @property
    def n_cols(self) -> int:
        """Columns of the group grid."""
        return int(self.beta_uids.numel())

    @property
    def n_pairs(self) -> int:
        """Number of pairs in the group."""
        return self.n_rows * self.n_cols


def group_rect_indices(
    group: _Group,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> Tensor:
    """Global pair indices of one rectangle of a group's grid, row-major.

    Works for materialised (``pair_idx``) and structural groups. The result is
    contiguous and has shape ``((row_stop - row_start) * (col_stop - col_start),)``.

    Args:
        group: The group whose grid is indexed.
        row_start, row_stop: Row range of the rectangle.
        col_start, col_stop: Column range of the rectangle.

    Returns:
        The rectangle's global pair indices in row-major order.
    """
    if group.pair_idx is not None:
        grid = group.pair_idx.view(group.n_rows, group.n_cols)
        return grid[row_start:row_stop, col_start:col_stop].reshape(-1).contiguous()
    rows = group.alpha_uids[row_start:row_stop].to(torch.int64)
    cols = group.beta_uids[col_start:col_stop].to(torch.int64)
    if group.row_base is not None:
        return (group.row_base[row_start:row_stop].unsqueeze(1) + cols.unsqueeze(0)).reshape(-1)
    assert group.col_base is not None and group.col_stride is not None
    col_base = group.col_base[col_start:col_stop]
    stride = group.col_stride[col_start:col_stop]
    return (col_base.unsqueeze(0) + rows.unsqueeze(1) * stride.unsqueeze(0)).reshape(-1)


def _group_pair_numel(group: _Group) -> int:
    """Number of pairs of a materialised group (``pair_idx`` must be present)."""
    assert group.pair_idx is not None
    return int(group.pair_idx.numel())


def group_pair_indices(group: _Group) -> Tensor:
    """Global pair indices of every group-grid position, row-major.

    Args:
        group: The group whose grid is indexed.

    Returns:
        The group's global pair indices in row-major grid order.
    """
    return group_rect_indices(group, 0, group.n_rows, 0, group.n_cols)


def group_flat_indices(group: _Group, flat: Tensor) -> Tensor:
    """Global pair indices of flat group-grid positions ``flat``.

    Args:
        group: The group whose grid is indexed.
        flat: Row-major grid positions, ``0 <= flat < n_rows * n_cols``.

    Returns:
        The global pair index of each position.
    """
    n_cols = group.n_cols
    if group.pair_idx is not None:
        return group.pair_idx.to(flat.device)[flat]
    rows = torch.div(flat, n_cols, rounding_mode="floor")
    cols = flat - rows * n_cols
    if group.row_base is not None:
        return group.row_base.to(flat.device)[rows] + group.beta_uids.to(flat.device)[cols]
    assert group.col_base is not None and group.col_stride is not None
    return (
        group.col_base.to(flat.device)[cols]
        + group.alpha_uids.to(flat.device)[rows] * group.col_stride.to(flat.device)[cols]
    )


@dataclasses.dataclass(frozen=True)
class SpinGroups:
    """One grouping of the pair set into products.

    Attributes:
        axis: ``"beta"`` when β strings are grouped by their α-partner set,
            ``"alpha"`` for the mirror grouping.
        group_of_string: ``(M_axis,)`` group id of each grouping-spin uid.
        groups: The Cartesian blocks, in a deterministic bucket order.
        total_pairs: Pairs covered by the groups (equals the basis size).
    """

    axis: str
    group_of_string: Tensor
    groups: tuple[_Group, ...]
    total_pairs: int


@dataclasses.dataclass(frozen=True)
class PairBlocks:
    """Both exact groupings of one pair basis.

    Attributes:
        by_beta: Groups disjoint in β (used by the α channels).
        by_alpha: Groups disjoint in α (used by the β channels).
        n_pairs: Basis size.
        stats: Diagnostic counts and byte estimates.
        structural: True when the groups carry arithmetic pair layouts instead
            of explicit ``pair_idx`` maps.
    """

    by_beta: SpinGroups
    by_alpha: SpinGroups
    n_pairs: int
    stats: dict
    structural: bool = False


def build_pair_blocks(
    pair_basis: "PairBasis | StructuralPairBasis",
    *,
    mode: str = "auto",
    min_pairs: int = 1 << 20,
    max_groups: int = 64,
    min_group_pairs: int = 4096,
    small_group_fraction: float = 0.01,
    byte_cap: Optional[int] = None,
) -> Optional[PairBlocks]:
    """Build the exact Cartesian decomposition of ``pair_basis``.

    Args:
        pair_basis: the lex-sorted pair basis.
        mode: ``"off"`` returns ``None``; ``"auto"`` applies the usefulness
            gates below; ``"on"`` skips the size gates (tests, A/B runs) but
            still requires the exact structural assertions to hold.
        min_pairs: ``"auto"`` refuses bases smaller than this.
        max_groups: ``"auto"`` refuses more groups than this on either axis.
        min_group_pairs: groups smaller than this count toward the small-group
            coverage gate.
        small_group_fraction: ``"auto"`` refuses when groups smaller than
            ``min_group_pairs`` cover more than this fraction of the basis.
        byte_cap: optional cap on the estimated decomposition bytes (the two
            ``pair_idx`` maps plus the uid lists, in bytes).

    Returns:
        A :class:`PairBlocks`, or ``None`` when disabled, gated out, or when
        the exact build fails (a non-tiling group or a non-monotone
        ``pair_idx``; a fingerprint collision is refined into exact subgroups,
        not rejected). Callers must fall back to the streamed path on ``None``;
        this function never returns a partial decomposition.

    Raises:
        ValueError: ``mode`` is not one of ``"off"``, ``"auto"``, ``"on"``.
    """
    mode = str(mode).lower()
    if mode not in ("off", "auto", "on"):
        raise ValueError(f"block_matvec must be 'off', 'auto' or 'on', got {mode!r}")
    if isinstance(pair_basis, StructuralPairBasis):
        if mode == "off":
            return None
        return _structural_pair_blocks(pair_basis)
    if mode == "off":
        return None
    if mode == "auto" and int(pair_basis.n_pairs) < int(min_pairs):
        return None

    by_beta = _spin_groups(
        pair_basis,
        axis="beta",
        mode=mode,
        max_groups=max_groups,
    )
    if by_beta is None:
        return None
    by_alpha = _spin_groups(
        pair_basis,
        axis="alpha",
        mode=mode,
        max_groups=max_groups,
    )
    if by_alpha is None:
        return None

    if mode == "auto":
        for groups in (by_beta, by_alpha):
            small = sum(
                _group_pair_numel(g)
                for g in groups.groups
                if _group_pair_numel(g) < int(min_group_pairs)
            )
            if small > small_group_fraction * int(pair_basis.n_pairs):
                return None

    n = int(pair_basis.n_pairs)
    est_bytes = 2 * n * 8  # the two pair_idx maps, int64
    for groups in (by_beta, by_alpha):
        for g in groups.groups:
            est_bytes += (int(g.alpha_uids.numel()) + int(g.beta_uids.numel())) * 8
    if byte_cap is not None and est_bytes > int(byte_cap):
        return None

    stats = {
        "groups_beta": len(by_beta.groups),
        "groups_alpha": len(by_alpha.groups),
        "largest_group_beta": max(_group_pair_numel(g) for g in by_beta.groups),
        "largest_group_alpha": max(_group_pair_numel(g) for g in by_alpha.groups),
        "block_bytes": est_bytes,
    }
    return PairBlocks(by_beta=by_beta, by_alpha=by_alpha, n_pairs=n, stats=stats)


def _structural_pair_blocks(basis: StructuralPairBasis) -> PairBlocks:
    """Group a structural basis by identical per-spin block-membership sets.

    The block-membership table of the structural basis already encodes each
    string's partner set: two α strings have the same β partners exactly when
    they occur in the same set of coarse blocks, and mirror-wise for β. Those
    membership classes are therefore the ``by_alpha`` / ``by_beta`` groups;
    no pair enumeration is needed.

    A β group ``v`` (membership ``m``) has ``β_uids`` = its membership class
    and ``α_uids`` = the α strings of every block in ``m``; those blocks share
    the group's β strings, so their α ranges are disjoint and concatenate in
    ascending uid order. The mirror construction gives the α groups. Each
    group stores the coarse-block rectangles it is made of and an arithmetic
    pair layout (``row_base`` or ``col_base``/``col_stride``) instead of a
    length-``N`` index map.

    Args:
        basis: The structural pair basis.

    Returns:
        The structural :class:`PairBlocks` decomposition.
    """
    alpha_ids = basis.alpha_block_ids.detach().cpu().numpy()
    beta_ids = basis.beta_block_ids.detach().cpu().numpy()
    _alpha_keys, alpha_bucket = np.unique(alpha_ids, axis=0, return_inverse=True)
    _beta_keys, beta_bucket = np.unique(beta_ids, axis=0, return_inverse=True)
    alpha_bucket = alpha_bucket.reshape(-1)
    beta_bucket = beta_bucket.reshape(-1)

    blocks = basis.blocks
    a_start = np.array([b.alpha_start for b in blocks], dtype=np.int64)
    a_stop = np.array([b.alpha_stop for b in blocks], dtype=np.int64)
    b_start = np.array([b.beta_start for b in blocks], dtype=np.int64)
    b_stop = np.array([b.beta_stop for b in blocks], dtype=np.int64)
    pair_base = np.array([b.pair_base for b in blocks], dtype=np.int64)
    width = b_stop - b_start

    def _tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(basis.device)

    # ---- by_beta: one group per β membership class ---------------------
    beta_groups: list[_Group] = []
    for bucket in range(int(beta_bucket.max()) + 1 if beta_bucket.size else 0):
        beta_uids = np.nonzero(beta_bucket == bucket)[0].astype(np.int64)
        if beta_uids.size == 0:
            continue
        members = np.unique(beta_ids[beta_uids[0]])
        members = members[members >= 0]
        # The blocks of one β class share the class' β strings, so their α
        # ranges are disjoint; sort by start to concatenate in uid order.
        members = members[np.argsort(a_start[members], kind="stable")]
        alpha_uids = np.concatenate([np.arange(a_start[i], a_stop[i]) for i in members])
        row_base = np.empty(alpha_uids.shape[0], dtype=np.int64)
        sub_blocks: list[_SubBlock] = []
        offset = 0
        for i in members:
            n_rows = int(a_stop[i] - a_start[i])
            row_base[offset : offset + n_rows] = (
                pair_base[i]
                + (alpha_uids[offset : offset + n_rows] - a_start[i]) * width[i]
                - b_start[i]
            )
            sub_blocks.append(
                _SubBlock(
                    block=int(i),
                    row_start=offset,
                    row_stop=offset + n_rows,
                    col_start=0,
                    col_stop=int(beta_uids.shape[0]),
                )
            )
            offset += n_rows
        beta_groups.append(
            _Group(
                alpha_uids=_tensor(alpha_uids),
                beta_uids=_tensor(beta_uids),
                row_base=_tensor(row_base),
                sub_blocks=tuple(sub_blocks),
            )
        )

    # ---- by_alpha: one group per α membership class --------------------
    alpha_groups: list[_Group] = []
    for bucket in range(int(alpha_bucket.max()) + 1 if alpha_bucket.size else 0):
        alpha_uids = np.nonzero(alpha_bucket == bucket)[0].astype(np.int64)
        if alpha_uids.size == 0:
            continue
        members = np.unique(alpha_ids[alpha_uids[0]])
        members = members[members >= 0]
        # Mirror argument: these blocks share the class' α strings, so their
        # β ranges are disjoint and concatenate in ascending uid order.
        members = members[np.argsort(b_start[members], kind="stable")]
        beta_uids = np.concatenate([np.arange(b_start[i], b_stop[i]) for i in members])
        col_base = np.empty(beta_uids.shape[0], dtype=np.int64)
        col_stride = np.empty(beta_uids.shape[0], dtype=np.int64)
        sub_blocks = []
        offset = 0
        for i in members:
            n_cols = int(b_stop[i] - b_start[i])
            sel = beta_uids[offset : offset + n_cols]
            col_base[offset : offset + n_cols] = (
                pair_base[i] - a_start[i] * width[i] - b_start[i] + sel
            )
            col_stride[offset : offset + n_cols] = width[i]
            row_lo = int(np.searchsorted(alpha_uids, a_start[i], side="left"))
            row_hi = int(np.searchsorted(alpha_uids, a_stop[i], side="left"))
            sub_blocks.append(
                _SubBlock(
                    block=int(i),
                    row_start=row_lo,
                    row_stop=row_hi,
                    col_start=offset,
                    col_stop=offset + n_cols,
                )
            )
            offset += n_cols
        alpha_groups.append(
            _Group(
                alpha_uids=_tensor(alpha_uids),
                beta_uids=_tensor(beta_uids),
                col_base=_tensor(col_base),
                col_stride=_tensor(col_stride),
                sub_blocks=tuple(sub_blocks),
            )
        )

    n = int(basis.n_pairs)
    stats = {
        "groups_beta": len(beta_groups),
        "groups_alpha": len(alpha_groups),
        "largest_group_beta": max((g.n_pairs for g in beta_groups), default=0),
        "largest_group_alpha": max((g.n_pairs for g in alpha_groups), default=0),
        "block_bytes": int(
            basis.n_unique_alphas
            + basis.n_unique_betas
            + basis.alpha_block_ids.numel()
            + basis.beta_block_ids.numel()
        )
        * 8,
    }
    return PairBlocks(
        by_beta=SpinGroups(
            axis="beta",
            group_of_string=_tensor(beta_bucket),
            groups=tuple(beta_groups),
            total_pairs=n,
        ),
        by_alpha=SpinGroups(
            axis="alpha",
            group_of_string=_tensor(alpha_bucket),
            groups=tuple(alpha_groups),
            total_pairs=n,
        ),
        n_pairs=n,
        stats=stats,
        structural=True,
    )


def _spin_groups(
    pair_basis: PairBasis,
    *,
    axis: str,
    mode: str,
    max_groups: int,
) -> Optional[SpinGroups]:
    """Group one spin's strings by their identical partner set, exactly.

    Args:
        pair_basis: the pair basis.
        axis: ``"beta"`` groups β by α partners; ``"alpha"`` groups α by β.
        mode: ``"auto"`` or ``"on"`` (``"off"`` is handled by the caller).
        max_groups: ``"auto"`` refuses more distinct groups than this.

    Returns:
        The verified grouping, or ``None`` on a failed exactness check or gate.
    """
    device = pair_basis.device
    if axis == "beta":
        offsets = pair_basis.partners_of_beta_offsets
        partner_uid = pair_basis.partners_of_beta_alpha_uid
        axis_uid_of_pair = pair_basis.beta_uid_of_pair
        m_axis = pair_basis.n_unique_betas
    else:
        offsets = pair_basis.partners_of_alpha_offsets
        partner_uid = pair_basis.partners_of_alpha_beta_uid
        axis_uid_of_pair = pair_basis.alpha_uid_of_pair
        m_axis = pair_basis.n_unique_alphas

    bucket = _verified_buckets(offsets, partner_uid, m_axis, device)
    if bucket is None:
        return None
    inverse, counts = bucket
    n_groups = int(counts.shape[0])
    if mode == "auto" and n_groups > int(max_groups):
        return None

    order = torch.argsort(inverse, stable=True)  # axis uids grouped, ascending
    axis_counts = torch.bincount(inverse, minlength=n_groups)
    axis_off = torch.cumsum(axis_counts, 0) - axis_counts

    group_of_pair = inverse[axis_uid_of_pair]
    pair_order = torch.argsort(group_of_pair, stable=True)
    pair_counts = torch.bincount(group_of_pair, minlength=n_groups)
    pair_off = torch.cumsum(pair_counts, 0) - pair_counts

    groups: list[_Group] = []
    for g in range(n_groups):
        size = int(pair_counts[g])
        if size == 0:
            return None  # a bucket with no pairs cannot happen; be strict
        a0 = int(axis_off[g])
        members = order[a0 : a0 + int(axis_counts[g])]  # ascending grouping uids
        p0 = int(pair_off[g])
        pair_idx = pair_order[p0 : p0 + size]

        first = int(members[0])
        s0 = int(offsets[first])
        s1 = int(offsets[first + 1])
        partners = partner_uid[s0:s1]  # ascending by CSR construction
        if axis == "beta":
            alpha_uids, beta_uids = partners, members
        else:
            alpha_uids, beta_uids = members, partners

        expected = int(alpha_uids.numel()) * int(beta_uids.numel())
        if size != expected:
            return None
        if size > 1 and not bool(torch.all(pair_idx[1:] > pair_idx[:-1])):
            return None
        groups.append(_Group(alpha_uids=alpha_uids, beta_uids=beta_uids, pair_idx=pair_idx))

    return SpinGroups(
        axis=axis,
        group_of_string=inverse,
        groups=tuple(groups),
        total_pairs=int(pair_basis.n_pairs),
    )


def _verified_buckets(
    offsets: Tensor,
    partner_uid: Tensor,
    m_axis: int,
    device: torch.device,
) -> Optional[tuple[Tensor, Tensor]]:
    """Bucket strings by partner set and verify every multi-member bucket.

    The fingerprint is ``(length, Σ(uid+1)·C1, Σ(uid+1)·C2)``; equal partner
    sets necessarily collide, but unequal sets can too, so every bucket with
    more than one member is compared exactly (tiled, O(N) total work).

    Args:
        offsets: CSR offsets of the grouping spin's partner lists.
        partner_uid: concatenated partner uids (ascending within each block).
        m_axis: number of strings.
        device: device of the CSR arrays.

    Returns:
        ``(bucket_of_string, bucket_counts)`` with every bucket exact. A
        fingerprint collision splits its bucket into exact subgroups (a member
        whose partner list matches no representative starts a new bucket), so
        unequal partner sets never merge and the decomposition stays exact.
    """
    lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    inv = torch.repeat_interleave(torch.arange(m_axis, dtype=torch.int64, device=device), lengths)
    vals = partner_uid.to(torch.int64) + 1
    fp1 = torch.zeros(m_axis, dtype=torch.int64, device=device)
    fp2 = torch.zeros(m_axis, dtype=torch.int64, device=device)
    fp1.index_add_(0, inv, vals * _FP_C1)
    fp2.index_add_(0, inv, vals * _FP_C2)
    fp = torch.stack((lengths, fp1, fp2), dim=1)
    _unique, inverse = torch.unique(fp, dim=0, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(_unique.shape[0]))

    order = torch.argsort(inverse, stable=True)
    off = torch.cumsum(counts, 0) - counts
    # The two linear sums can collide for unequal partner sets. Verify each
    # multi-member bucket as before; on a collision, split it into exact
    # subgroups instead of rejecting the whole axis. A recovered/expanded
    # pair list has an arbitrary partner set per string, so a single colliding
    # bucket must not make the block (and tile) path unreachable.
    labels = inverse.clone()
    next_label = int(_unique.shape[0])
    for g in range(int(counts.shape[0])):
        n = int(counts[g])
        if n <= 1:
            continue
        c0 = int(off[g])
        members = order[c0 : c0 + n]
        first = int(members[0])
        s0 = int(offsets[first])
        length = int(lengths[first])
        rep = partner_uid[s0 : s0 + length]
        exact = True
        if length:
            rows = max(1, _VERIFY_ELEMENTS // length)
            for start in range(0, n, rows):
                chunk = members[start : start + rows]
                base = offsets[chunk].to(torch.int64)
                idx = base.unsqueeze(1) + torch.arange(length, dtype=torch.int64, device=device)
                block = partner_uid[idx]
                if not bool(torch.equal(block, rep.expand_as(block))):
                    exact = False
                    break
        if exact:
            continue
        pending = members.tolist()
        keep_first = True
        while pending:
            head = pending[0]
            s_head = int(offsets[head])
            length = int(lengths[head])
            head_key = partner_uid[s_head : s_head + length]
            same: list[int] = []
            rest: list[int] = []
            for member in pending[1:]:
                s = int(offsets[member])
                if int(lengths[member]) == length and bool(
                    torch.equal(partner_uid[s : s + length], head_key)
                ):
                    same.append(member)
                else:
                    rest.append(member)
            if keep_first:
                label = g
                keep_first = False
            else:
                label = next_label
                next_label += 1
                labels[head] = label
            for member in same:
                labels[member] = label
            if len(rest) > 64:
                # A pathological collision: exact singleton subgroups, no
                # quadratic splitting.
                for member in rest:
                    labels[member] = next_label
                    next_label += 1
                break
            pending = rest
    labels = torch.unique(labels, return_inverse=True)[1]
    return labels, torch.bincount(labels)
