"""Vectorized block kernels for the GDB pair matvec on block bases.

The generic passes in :mod:`~torch_sci_solver.torch_impl.pair_helpers` enumerate
excitation candidates per unique spin string and test membership with
``searchsorted``; the generic scatter in
:mod:`~torch_sci_solver.torch_impl.mult_pairs` then evaluates one Slater-Condon
element per surviving row. On a *block-decomposed* basis
(:mod:`~torch_sci_solver.torch_impl.pair_blocks`) the pair set is a disjoint
union of Cartesian products ``alpha_uids × beta_uids``, so the same rows follow
from **block index arithmetic** with no per-row pair lookup, and each group's
tiles can be lifted through the factored product-mode kernels of
:mod:`~torch_sci_solver.torch_impl.mult` (one integral/sign factor per
excitation, a cheap parity popcount per spectator) instead of one full
Slater-Condon evaluation per row.

This module supplies four whole-channel kernels:

* :func:`add_ab_blocks` / :func:`add_ab_blocks_local` — the αβ-double channel,
  enumerating α moves between two groups' α-uid sets and β moves between their
  β-uid sets, then crossing them.
* :func:`add_same_spin_blocks` / :func:`add_same_spin_blocks_local` — the four
  same-spin channels. An α move keeps the β string fixed, so it runs inside one
  ``by_beta`` group with the group's α grid as rows and its β grid as the
  spectator axis; a β move mirrors this inside one ``by_alpha`` group. The
  product kernels themselves are reused unchanged, with ``M_a``/``M_b`` set to
  the group's grid dimensions.
* :func:`add_ab_fragment` / :func:`add_same_spin_fragment` — the same channels
  for one rectangle of one group's grid
  (:class:`~torch_sci_solver.torch_impl.pair_tiles.PairFragment`), writing only
  the fragment's own block. The caller scatters each fragment through the
  tile's ``local_of_global`` map, so fragments are emitted in group order and
  never concatenated.
* :func:`add_ab_slab` / :func:`add_same_spin_slab` — the fragment channels
  consuming one staged *source slab*: only a rectangle of one ket group's grid
  is resident, and the ket positions are rebased to that rectangle. Summing a
  channel over all source slabs of a group reproduces the fragment channel
  with the whole group resident.

The diagonal stays on the generic ``hii * T`` multiplication.

Notes
-----
**Row-set argument (αβ).** Let the ordered pair ``(g1, g2)`` be the ket group
and the bra group, with grids ``U1 × B1`` and ``U2 × B2``. A cross-spin double
row is a ket pair ``(a1, b1)`` and a bra pair ``(a2, b2)`` where ``a1 -> a2`` is
a single α excitation and ``b1 -> b2`` a single β excitation. Enumerating every
such α move ``U1 -> U2`` and every β move ``B1 -> B2`` and taking their product
therefore yields every row whose ket lies in ``g1`` and bra in ``g2``. The
``by_beta`` groups partition the basis (disjoint in β), so each pair belongs to
exactly one group; summing over all ordered group pairs reproduces the generic
row set exactly once. Conversely every crossed row is a genuine αβ double
between two basis pairs, so it is a row the generic pass emits too. The sets
are equal, not merely contained.

**Row-set argument (same-spin).** For an α row, ket ``(a1, b)`` and bra
``(a2, b)`` share the β string ``b``, which fixes the ``by_beta`` group. The
Cartesian invariant makes every ``(a, b)`` with ``a ∈ U_g`` a basis pair, so
``a1, a2 ∈ U_g`` and the pair occurs in ``U_g × B_g``. Enumerating the α moves
inside ``U_g`` therefore yields exactly the generic rows of that group, with no
membership test beyond the move lookup in ``U_g``. Groups are disjoint in β, so
the same row cannot arise from two groups. The β channels mirror this with
``by_alpha``.

**Matrix element.** The αβ kernel
:func:`~torch_sci_solver.torch_impl.mult._alphabeta_pair_matrix` evaluates the
parity on the bra strings, and here the bra is the **target** group's string;
the ``cr`` labels are occupied in that target and the ``an`` labels in the ket
source. That matches the generic table, whose ``cr`` is occupied in the bra and
``an`` in the ket. The same-spin kernels derive the same labels from
:func:`_single_moves` / :func:`_double_moves`, and the product kernels compute
the parity from the bra half-bitmasks exactly as in product mode.

**Owner-computes.** ``pair_idx`` of a group is sorted ascending, so a bra row
range ``[row_start, row_stop)`` selects one contiguous block-position window
``[lo, hi)`` via ``searchsorted``. The owned pass enumerates the **inverse**
moves into that window — from its bra strings to the ket group — instead of
enumerating forward from the full ket set and discarding the unowned targets.
For the same-spin channels the owned α rows are bounded by
``[lo // M_b, ceil(hi / M_b))``: the α kernel enumerates the moves from that
slab to the group's α strings and reads ket rows from the replicated full
vector, while the β kernel enumerates from the owned β columns (all columns
when the window wraps rows, because only the boundary rows are partial) and
reads its ket columns from the same slab because a β move leaves α unchanged.
The αβ channel does both from the owned bra α rows and β columns. Every
candidate grid is therefore sized by the owned rows rather than the whole
basis, which is what makes the owner-computes work scale with ``1/P``. The
window's *global* indices are not contiguous (other groups interleave), so the
final scatter maps positions through ``pair_idx[lo:hi] - row_start`` rather
than by arithmetic alone.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult import (
    DEFAULT_CHUNK_SIZE,
    _alphabeta_pair_matrix,
    _apply_alpha_double_chunk_fast,
    _apply_alpha_single_chunk_fast,
    _apply_beta_double_chunk_fast,
    _apply_beta_single_chunk_fast,
)
from torch_sci_solver.torch_impl.pair_blocks import (
    PairBlocks,
    SpinGroups,
    _Group,
    group_flat_indices,
    group_pair_indices,
    group_rect_indices,
)
from torch_sci_solver.torch_impl.pair_helpers import (
    PairHelpers,
    _bit_mask_rows,
    _chunk_pairs,
    _complement_mask,
    _dispatch_bit_positions,
    _dispatch_found_in,
    _sort_strings,
    _triu_indices,
)
from torch_sci_solver.torch_impl.pair_tiles import PairFragment

# Peak bytes per lifted (α move × β move) entry of one tile. A tile holds two
# int64 block-position grids plus the gathered ket amplitudes, the matrix
# elements and their product, and ``_alphabeta_pair_matrix`` keeps roughly a
# dozen (csize_a, csize_b) int64 temporaries alive at its peak. As with the
# calibrations in ``pair_helpers``, this is measured rather than counted and
# leans high (overestimating only makes tiles smaller). Used to floor the
# caller's entry cap by ``pair_helpers.budget_bytes``.
_BYTES_PER_AB_BLOCK_ENTRY = 512

# Same idea for one same-spin tile: the product kernels keep a few
# (excitation × spectator) tensors per entry, so the caller's entry cap is
# floored by the byte budget here too.
_BYTES_PER_SAME_SPIN_BLOCK_ENTRY = 512

# Measured peak bytes per surviving move that the product kernels allocate for
# their integral vectors and per-excitation grids before the tile loop: about
# 56 bytes per orbital for singles (rounded up here) and 448-512 bytes for
# doubles (rounded up with a margin), per 8 bytes of integral element. Used to
# slice each survivor table so one kernel call's setup fits the budget
# regardless of how many candidates the enumeration produced. The kernels'
# tensors follow the integral dtype, so the caps scale with its element size.
_BYTES_PER_SINGLE_SURVIVOR_ORBITAL = 64
_BYTES_PER_DOUBLE_SURVIVOR = 640

# One lifted αβ entry, in units of one same-spin enumeration candidate: the
# same-spin weight counts candidates per pair and the αβ weight counts the
# crossed (inverse α move × inverse β move) entries the kernel lifts and
# scatters. A candidate is one generated bit pattern and one searchsorted
# probe; an entry is a gathered amplitude, `_alphabeta_pair_matrix` and the
# scatter, i.e. about a dozen int64 temporaries. Calibrated by fitting the
# measured per-rank matvec profiles of the 2·10⁷-pair benchmark basis (a
# canonical and a work-weighted partition of the same basis): one entry costs
# 3.3 candidates. Only the order of magnitude matters — a factor of two either
# way still spreads the αβ channel far better than ignoring it.
_AB_ENTRY_COST = 3.3

# Lifted entries per outer-product slab while accumulating the αβ weight, so
# one slab stays small regardless of group shape.
_WEIGHT_SLAB_ENTRIES = 4 << 20

# Opt-in channel timers for scaling diagnostics (``SCIS_PROFILE_BLOCKS=1``). They
# are read per call, so a benchmark can set the variable after importing this
# module. Off by default: no synchronize, no timing, no print.
_PROFILE_ENV = "SCIS_PROFILE_BLOCKS"
_PROFILE_TOTALS: dict[str, float] = {"ab_s": 0.0, "same_spin_s": 0.0}


def profile_enabled() -> bool:
    """Whether the block-channel timers are active."""
    return os.environ.get(_PROFILE_ENV) == "1"


def reset_block_profile() -> None:
    """Zero the per-process channel timers."""
    for key in _PROFILE_TOTALS:
        _PROFILE_TOTALS[key] = 0.0


def block_profile() -> dict[str, float]:
    """Seconds spent in each block channel since the last reset."""
    return dict(_PROFILE_TOTALS) if profile_enabled() else {}


def _sync_profile(tensor: Tensor) -> None:
    """Wait for queued work so a channel's wall time is the measured work."""
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _timed(key: str, fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Run ``fn`` and add its synchronized wall time to the channel timer."""
    if not profile_enabled():
        fn(*args, **kwargs)
        return
    output = args[0]
    assert isinstance(output, Tensor)
    _sync_profile(output)
    start = time.perf_counter()
    try:
        fn(*args, **kwargs)
    finally:
        _sync_profile(output)
        _PROFILE_TOTALS[key] += time.perf_counter() - start


@dataclass
class _MoveCache:
    """Byte-bounded cache of stage-independent move tables.

    The tile operator keys its memos by ``(destination fragment, source group,
    channel, chunk)``: a tile rank applies one fragment against each source
    stage in turn, and the staged slab kernels enumerate the fragment's moves
    against the whole source group before masking them to the stage's
    rectangle. The 1-D block matvec keys the same class by ``(channel, source
    group, destination group, spin side, uid range, string width)`` so the
    streamed/1-D path enumerates each table once across matvecs (and Davidson
    iterations) instead of on every call. Keys never collide across the
    channel, group, spin or single/double order because each is part of them.

    Nothing of size ``nnz(H)`` is stored: entries are the enumerated survivor
    tuples, accounted by ``numel() * element_size()``.

    Attributes:
        budget_bytes: Cap on the resident move-table bytes.
        entries: Keyed move tuples; a ``None`` value records an enumeration
            that produced no survivors.
        used_bytes: Bytes currently held in ``entries``, never above
            ``budget_bytes``.
        full: Latch of the retain-what-fits policy (``evict=False``): set once
            an insert would overflow a cache that already holds entries; a
            first table larger than the whole budget is skipped without
            latching the cache, so a later smaller table can fit.
        evict: Drop least-recently-used entries to make room for a table that
            would overflow the budget. A table larger than the whole budget is
            never admitted. The tiled operator retains its prefix instead
            (``False``); the 1-D block path enables eviction so one shared
            budget can serve both channels.
    """

    budget_bytes: int
    entries: dict[tuple, tuple[Tensor, ...] | None] = field(default_factory=dict)
    used_bytes: int = 0
    full: bool = False
    evict: bool = False


_MEMO_MISS: Any = object()


def _table_bytes(moves: tuple[Tensor, ...] | None) -> int:
    """Resident bytes of one enumerated move table (``None`` is empty)."""
    if moves is None:
        return 0
    return sum(int(t.numel()) * int(t.element_size()) for t in moves)


def _memo_moves(
    cache: _MoveCache | None,
    key: tuple,
    fn: Callable[..., tuple[Tensor, ...] | None],
    *args: Any,
    **kwargs: Any,
) -> tuple[Tensor, ...] | None:
    """Return ``fn(*args, **kwargs)``, memoised under ``key`` when a cache is given.

    The slab and 1-D block kernels route their stage-independent enumerations
    through this so a table is enumerated once instead of once per stage (tile
    path) or once per matvec (1-D path). With ``cache.evict`` a new table makes
    room by dropping least-recently-used entries; a single table that cannot
    fit the whole budget is recomputed rather than admitted, so the accounting
    never exceeds the cap.
    """
    if cache is None:
        return fn(*args, **kwargs)
    cached = cache.entries.get(key, _MEMO_MISS)
    if cached is not _MEMO_MISS:
        if cache.evict:
            # Refresh insertion order: eviction drops the least recently used.
            cache.entries[key] = cache.entries.pop(key)
        return cached
    moves = fn(*args, **kwargs)
    if cache.full:
        return moves
    nbytes = _table_bytes(moves)
    if cache.used_bytes + nbytes <= cache.budget_bytes:
        cache.entries[key] = moves
        cache.used_bytes += nbytes
        return moves
    if not cache.evict:
        # One table larger than the whole budget must not latch an empty
        # cache: smaller tables inserted later can still fit.
        if cache.used_bytes > 0:
            cache.full = True
        return moves
    if nbytes > cache.budget_bytes:
        # A single table larger than the whole budget is never admitted.
        return moves
    while cache.entries and cache.used_bytes + nbytes > cache.budget_bytes:
        oldest = next(iter(cache.entries))
        cache.used_bytes -= _table_bytes(cache.entries.pop(oldest))
    cache.entries[key] = moves
    cache.used_bytes += nbytes
    return moves


def _same_spin_survivor_chunk(
    kind: str,
    n_orb: int,
    budget_bytes: int,
    element_bytes: int = 8,
) -> int:
    """Surviving moves per product-kernel call of one same-spin channel.

    The enumeration itself is bounded by the candidate grid (see
    ``_chunk_pairs``); the kernels' per-survivor setup is a separate cost, so
    each survivor table is sliced to fit this count. Slicing is exact: every
    entry refers to the same bra/ket grids and the source offsets are already
    applied.

    Args:
        kind: ``"singles"`` or ``"doubles"``; selects the measured per-survivor
            setup cost.
        n_orb: spatial orbital count.
        budget_bytes: byte budget for one kernel call's setup.
        element_bytes: size of one integral element in bytes (8 for fp64, 16
            for complex128, 4 for fp32). Real fp64 leaves the caps unchanged;
            complex128 doubles the per-survivor bytes, fp32 halves them.

    Returns:
        Maximum survivors per call, at least one.
    """
    if kind == "doubles":
        per_survivor = _BYTES_PER_DOUBLE_SURVIVOR
    else:
        per_survivor = _BYTES_PER_SINGLE_SURVIVOR_ORBITAL * int(n_orb)
    per_survivor = per_survivor * max(1, int(element_bytes)) // 8
    return max(1, int(budget_bytes) // per_survivor)


def _materialised_pair_idx(group: _Group) -> Tensor:
    """The group's explicit pair-index map (materialised groups only)."""
    assert group.pair_idx is not None
    return group.pair_idx


def _single_group(blocks: PairBlocks, axis: str, group: _Group) -> SpinGroups:
    """A one-block grouping for the fragment kernels.

    Args:
        blocks: The decomposition whose grouping metadata is reused.
        axis: ``"beta"`` or ``"alpha"``.
        group: The single group to carry.

    Returns:
        A :class:`SpinGroups` with ``group`` as its only entry.
    """
    source = blocks.by_beta if axis == "beta" else blocks.by_alpha
    return SpinGroups(
        axis=axis,
        group_of_string=source.group_of_string,
        groups=(group,),
        total_pairs=blocks.n_pairs,
    )


def _structural_group_values(T: Tensor, group: _Group) -> Tensor:
    """Values of every pair of a structural group, in group-grid order."""
    indices = group_rect_indices(group, 0, group.n_rows, 0, group.n_cols).to(T.device)
    return T[indices]


def _structural_fragment_rows(
    group: _Group,
    sub_block,
    axis: str,
    basis,
    first: int,
    stop: int,
) -> tuple[int, int]:
    """Group-row range of a sub-block, restricted to an owned global range.

    Args:
        group: The structural group.
        sub_block: The coarse-block rectangle inside the group.
        axis: ``"beta"`` for the α channel (rows are α uids), ``"alpha"``
            for the β channel.
        basis: The structural pair basis.
        first, stop: Owned global pair range.

    Returns:
        ``(row_start, row_stop)`` in group coordinates; empty when no owned
        pair of the sub-block lies in range.
    """
    block = basis.blocks[sub_block.block]
    if first <= block.pair_base and stop >= block.pair_base + block.n_pairs:
        return sub_block.row_start, sub_block.row_stop
    local_start = max(0, first - block.pair_base)
    local_stop = min(block.n_pairs, stop - block.pair_base)
    if local_stop <= local_start:
        return sub_block.row_start, sub_block.row_start
    width = block.n_beta
    row_lo = local_start // width
    row_hi = (local_stop + width - 1) // width
    if axis == "beta":
        return sub_block.row_start + row_lo, sub_block.row_start + row_hi
    # The β channel's group rows are the α membership class, so a block-local
    # row range maps to a uid interval, then to positions in the sorted class.
    uid_lo = block.alpha_start + row_lo
    uid_hi = block.alpha_start + row_hi
    alpha_uids = group.alpha_uids.to(torch.int64)
    uid_lo_t = torch.tensor(uid_lo, dtype=torch.int64, device=alpha_uids.device)
    uid_hi_t = torch.tensor(uid_hi, dtype=torch.int64, device=alpha_uids.device)
    pos_lo = int(torch.searchsorted(alpha_uids, uid_lo_t).item())
    pos_hi = int(torch.searchsorted(alpha_uids, uid_hi_t).item())
    return max(sub_block.row_start, pos_lo), min(sub_block.row_stop, pos_hi)


def _structural_scatter(
    Wb: Tensor,
    values: Tensor,
    group: _Group,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    first: int,
    stop: int | None,
) -> None:
    """Scatter a group rectangle's values, keeping only the owned indices."""
    indices = group_rect_indices(group, row_start, row_stop, col_start, col_stop).to(Wb.device)
    if stop is None:
        Wb.index_add_(0, indices, values.reshape(-1))
        return
    mask = (indices >= first) & (indices < stop)
    if not bool(mask.any()):
        return
    Wb.index_add_(0, indices[mask] - first, values.reshape(-1)[mask])


def _structural_same_spin_pass(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    row_start: int | None,
    chunk_size: int,
    triton_matvec: bool = False,
) -> None:
    """Same-spin channels over a structural basis, fragment by fragment.

    Each ``by_beta`` group's β strings belong to a set of coarse blocks whose
    α ranges are disjoint; those blocks partition the group's rows and each
    spans every group column, so one fragment kernel call per block covers the
    α channels. The ``by_alpha`` groups mirror this: each coarse block owns a
    contiguous row range and its β range inside the group.

    Args:
        Wb: output vector (full) or owned block (``row_start`` is not ``None``).
        T: input vector, read at the group's global pair indices.
        pair_helpers: helpers carrying ``structural`` and ``blocks``.
        integrals: integral bundle.
        row_start: first owned global pair, or ``None`` for a full matvec.
        chunk_size: per-kernel workspace bound.

    Raises:
        ValueError: the helpers are not structural.
    """
    basis = pair_helpers.structural
    blocks = pair_helpers.blocks
    if basis is None or blocks is None:
        raise ValueError("structural same-spin pass requires a structural basis")
    first = 0 if row_start is None else int(row_start)
    stop = None if row_start is None else first + int(Wb.numel())

    for axis, grouping in (("beta", blocks.by_beta), ("alpha", blocks.by_alpha)):
        for group in grouping.groups:
            single = _single_group(blocks, axis, group)
            t_group = _structural_group_values(T, group) if axis == "beta" else None
            for sub_block in group.sub_blocks:
                row_lo, row_hi = (
                    (sub_block.row_start, sub_block.row_stop)
                    if stop is None
                    else _structural_fragment_rows(group, sub_block, axis, basis, first, stop)
                )
                if row_hi <= row_lo:
                    continue
                if axis == "beta":
                    assert t_group is not None
                    frag = PairFragment(
                        group=0,
                        row_start=row_lo,
                        row_stop=row_hi,
                        col_start=0,
                        col_stop=group.n_cols,
                        pair_index=T.new_empty(0),
                    )
                    out = Wb.new_zeros(frag.n_rows * frag.n_cols)
                    add_same_spin_fragment(
                        out,
                        t_group,
                        pair_helpers,
                        integrals,
                        single,
                        frag,
                        chunk_size=chunk_size,
                        triton_matvec=triton_matvec,
                    )
                else:
                    assert t_group is None
                    slab = group_rect_indices(group, row_lo, row_hi, 0, group.n_cols).to(T.device)
                    t_slab = T[slab].reshape(row_hi - row_lo, group.n_cols)
                    frag = PairFragment(
                        group=0,
                        row_start=row_lo,
                        row_stop=row_hi,
                        col_start=sub_block.col_start,
                        col_stop=sub_block.col_stop,
                        pair_index=T.new_empty(0),
                    )
                    out = Wb.new_zeros(frag.n_rows * frag.n_cols)
                    add_same_spin_fragment(
                        out,
                        t_slab,
                        pair_helpers,
                        integrals,
                        single,
                        frag,
                        chunk_size=chunk_size,
                        triton_matvec=triton_matvec,
                    )
                _structural_scatter(
                    Wb,
                    out,
                    group,
                    row_lo,
                    row_hi,
                    sub_block.col_start,
                    sub_block.col_stop,
                    first,
                    stop,
                )


def _structural_ab_pass(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    row_start: int | None,
    chunk_size: int,
) -> None:
    """The αβ-double channel over a structural basis, fragment by fragment.

    The fragments are the ``by_beta`` groups' coarse blocks: each block's
    α rows span every column of its β membership class, and the block gains
    from every source group independently, exactly the row set
    :func:`add_ab_fragment` enumerates.

    Args:
        Wb: output vector (full) or owned block.
        T: input vector.
        pair_helpers: helpers carrying ``structural`` and ``blocks``.
        integrals: integral bundle.
        row_start: first owned global pair, or ``None`` for a full matvec.
        chunk_size: per-tile workspace bound.

    Raises:
        ValueError: the helpers are not structural.
    """
    basis = pair_helpers.structural
    blocks = pair_helpers.blocks
    if basis is None or blocks is None:
        raise ValueError("structural αβ pass requires a structural basis")
    first = 0 if row_start is None else int(row_start)
    stop = None if row_start is None else first + int(Wb.numel())

    for group_id, group in enumerate(blocks.by_beta.groups):
        for sub_block in group.sub_blocks:
            row_lo, row_hi = (
                (sub_block.row_start, sub_block.row_stop)
                if stop is None
                else _structural_fragment_rows(group, sub_block, "beta", basis, first, stop)
            )
            if row_hi <= row_lo:
                continue
            frag = PairFragment(
                group=group_id,
                row_start=row_lo,
                row_stop=row_hi,
                col_start=0,
                col_stop=group.n_cols,
                pair_index=T.new_empty(0),
            )
            out = Wb.new_zeros(frag.n_rows * frag.n_cols)
            add_ab_fragment(
                out, T, pair_helpers, integrals, blocks.by_beta, frag, chunk_size=chunk_size
            )
            _structural_scatter(Wb, out, group, row_lo, row_hi, 0, group.n_cols, first, stop)


def add_ab_blocks(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cache: _MoveCache | None = None,
) -> None:
    """Add the block-vectorized αβ-double channel of a full pair matvec.

    Args:
        Wb: ``(N,)`` output vector; contributions are added in place.
        T: ``(N,)`` input vector. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        chunk_size: maximum lifted entries ``csize_a * csize_b`` per tile,
            further floored by the helper's byte budget.
        cache: optional byte-bounded move-table cache for the α/β single-move
            enumerations; reused across matvecs. Ignored for a structural
            basis, whose passes are fragment kernels with no memo.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None``.
    """
    if pair_helpers.structural is not None:
        _timed(
            "ab_s",
            _structural_ab_pass,
            Wb,
            T,
            pair_helpers,
            integrals,
            row_start=None,
            chunk_size=chunk_size,
        )
        return
    _timed(
        "ab_s",
        _ab_block_pass,
        Wb,
        T,
        pair_helpers,
        integrals,
        row_start=None,
        chunk_size=chunk_size,
        cache=cache,
    )


def add_ab_blocks_local(
    Wb: Tensor,
    T_full: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    row_start: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cache: _MoveCache | None = None,
) -> None:
    """Add the block-vectorized αβ-double channel of an owner-computes matvec.

    Only bra rows in ``[row_start, row_start + Wb.numel())`` are written; ket
    amplitudes are read from the replicated full vector. The pass enumerates
    the inverse α/β moves from the owned bra slab into each ket group, so the
    enumeration work scales with the owned rows.

    Args:
        Wb: ``(N_local,)`` owned output block; contributions are added in place.
        T_full: ``(N,)`` replicated input vector. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        row_start: global index of the first owned pair.
        chunk_size: maximum lifted entries ``csize_a * csize_b`` per tile,
            further floored by the helper's byte budget.
        cache: optional byte-bounded move-table cache for the inverse α/β
            single-move enumerations; keys carry the owned range so blocks do
            not share tables with a different window. Ignored for a structural
            basis.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None``.
    """
    if pair_helpers.structural is not None:
        _timed(
            "ab_s",
            _structural_ab_pass,
            Wb,
            T_full,
            pair_helpers,
            integrals,
            row_start=int(row_start),
            chunk_size=chunk_size,
        )
        return
    _timed(
        "ab_s",
        _ab_block_pass,
        Wb,
        T_full,
        pair_helpers,
        integrals,
        row_start=int(row_start),
        chunk_size=chunk_size,
        cache=cache,
    )


def add_same_spin_blocks(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
    cache: _MoveCache | None = None,
) -> None:
    """Add the four same-spin channels of a full pair matvec, block by block.

    α moves are enumerated inside each ``by_beta`` group and β moves inside each
    ``by_alpha`` group; the group grids are handed to the factored product-mode
    kernels of :mod:`~torch_sci_solver.torch_impl.mult`.

    Args:
        Wb: ``(N,)`` output vector; contributions are added in place.
        T: ``(N,)`` input vector. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        chunk_size: maximum lifted entries per product-kernel tile.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel (real CUDA tensors only).
        cache: optional byte-bounded move-table cache for the single/double
            move enumerations; reused across matvecs. Ignored for a structural
            basis, whose passes are fragment kernels with no memo.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None``.
    """
    if pair_helpers.structural is not None:
        _timed(
            "same_spin_s",
            _structural_same_spin_pass,
            Wb,
            T,
            pair_helpers,
            integrals,
            row_start=None,
            chunk_size=chunk_size,
            triton_matvec=triton_matvec,
        )
        return
    _timed(
        "same_spin_s",
        _same_spin_block_pass,
        Wb,
        T,
        pair_helpers,
        integrals,
        row_start=None,
        chunk_size=chunk_size,
        triton_matvec=triton_matvec,
        cache=cache,
    )


def add_same_spin_blocks_local(
    Wb: Tensor,
    T_full: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    row_start: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
    cache: _MoveCache | None = None,
) -> None:
    """Add the four same-spin channels of an owner-computes pair matvec.

    Only bra rows in ``[row_start, row_start + Wb.numel())`` are written; ket
    amplitudes are read from the replicated full vector. The owned rows of each
    group are located as one contiguous block-position window, and both the α
    and β move tables are enumerated from that owned slab (inverse moves), so
    the enumeration work scales with the owned rows rather than the group.

    Args:
        Wb: ``(N_local,)`` owned output block; contributions are added in place.
        T_full: ``(N,)`` replicated input vector. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        row_start: global index of the first owned pair.
        chunk_size: maximum lifted entries per product-kernel tile.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel (real CUDA tensors only).
        cache: optional byte-bounded move-table cache for the single/double
            move enumerations; reused across matvecs. Ignored for a structural
            basis.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None``.
    """
    if pair_helpers.structural is not None:
        _timed(
            "same_spin_s",
            _structural_same_spin_pass,
            Wb,
            T_full,
            pair_helpers,
            integrals,
            row_start=int(row_start),
            chunk_size=chunk_size,
            triton_matvec=triton_matvec,
        )
        return
    _timed(
        "same_spin_s",
        _same_spin_block_pass,
        Wb,
        T_full,
        pair_helpers,
        integrals,
        row_start=int(row_start),
        chunk_size=chunk_size,
        triton_matvec=triton_matvec,
        cache=cache,
    )


def _apply_same_spin_double(
    out: Tensor,
    T: Tensor,
    adets: Tensor,
    bdets: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    is_alpha: bool,
    bra: Tensor,
    ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    M_a: int,
    M_b: int,
    chunk_size: int,
    T_full: Tensor | None = None,
    T_stride: int | None = None,
    triton_matvec: bool = False,
    bra_sorted: bool = True,
) -> None:
    """Apply one same-spin doubles survivor slice to an output block.

    Defaults to the batched product kernels of
    :mod:`~torch_sci_solver.torch_impl.mult`; with ``triton_matvec`` it routes
    the slice to the output-stationary kernel of
    :mod:`~torch_sci_solver.torch_impl.mult_pair_blocks_triton`, which raises
    when Triton, a CUDA device or a supported dtype is unavailable.

    Args:
        out: output block, ``(M_a * M_b,)`` row-major, accumulated in place.
        T: input buffer the ket amplitudes are read from.
        adets: α strings of the output rows.
        bdets: β strings of the output columns.
        n_orb: spatial orbital count.
        integrals: integral bundle on the output's device.
        is_alpha: ``True`` for an α-doubles move, ``False`` for β doubles.
        bra, ket, cr0, cr1, an0, an1: the survivor slice's destination and
            source positions and excitation labels.
        M_a, M_b: output block dimensions.
        chunk_size: per-kernel workspace bound (batched path only).
        T_full: global ket vector of an owner-computes α slice; ``None`` when
            ``T`` already holds it.
        T_stride: row width of ``T`` when it differs from ``M_b`` (β slices).
        triton_matvec: route the slice to the Triton kernel.
        bra_sorted: whether ``bra`` is non-decreasing (the inverse enumerators
            group by destination position; the forward full-vector pass does
            not).
    """
    if triton_matvec:
        from torch_sci_solver.torch_impl.mult_pair_blocks_triton import (
            apply_same_spin_doubles,
        )

        apply_same_spin_doubles(
            out,
            T_full if T_full is not None else T,
            adets,
            bdets,
            n_orb,
            integrals,
            bra=bra,
            ket=ket,
            cr0=cr0,
            cr1=cr1,
            an0=an0,
            an1=an1,
            M_a=M_a,
            M_b=M_b,
            is_alpha=is_alpha,
            T_stride=int(M_b if T_stride is None else T_stride),
            bra_sorted=bra_sorted,
        )
        return
    if is_alpha:
        _apply_alpha_double_chunk_fast(
            out,
            T,
            adets,
            bdets,
            n_orb,
            integrals,
            bra_alpha_per_exc=bra,
            ket_alpha_per_exc=ket,
            cr0=cr0,
            cr1=cr1,
            an0=an0,
            an1=an1,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
            T_full=T_full,
        )
    else:
        _apply_beta_double_chunk_fast(
            out,
            T,
            adets,
            bdets,
            n_orb,
            integrals,
            bra_beta_per_exc=bra,
            ket_beta_per_exc=ket,
            cr0=cr0,
            cr1=cr1,
            an0=an0,
            an1=an1,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
            T_stride=T_stride,
        )


def add_ab_fragment(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    groups: SpinGroups,
    frag: PairFragment,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Add the αβ-double channel of one block fragment to a fragment-local block.

    The fragment is the rectangle ``[row_start, row_stop) × [col_start, col_stop)``
    of the bra group's ``|U_g| × |B_g|`` grid. Only that rectangle is produced:
    ``Wb`` is the fragment's own ``frag.n_rows * frag.n_cols`` block in
    row-major order, so position ``local_row * frag.n_cols + local_col`` holds
    the pair at the same position of
    :attr:`~torch_sci_solver.torch_impl.pair_tiles.PairFragment.pair_index`.
    The caller scatters the result into the tile through
    :meth:`~torch_sci_solver.torch_impl.pair_tiles.PairTileBounds.local_of_global`.

    The ket side stays the full source group of every ordered group pair: the
    amplitude is read at ``T[pair_idx1[src_pos]]`` with ``src_pos`` row-major
    over the source group's full grid, exactly as in :func:`add_ab_blocks`.
    Only the bra side is the fragment: the move tables are enumerated from the
    fragment's bra α rows and β columns into each source group, so every
    lifted candidate lies in the fragment slab.

    Args:
        Wb: ``(frag.n_rows * frag.n_cols,)`` fragment output block, row-major
            over the fragment rectangle; contributions are added in place.
        T: ``(N,)`` input vector. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        groups: the ``by_beta`` grouping the fragment belongs to.
        frag: the fragment rectangle to compute.
        chunk_size: maximum lifted entries ``csize_a * csize_b`` per tile,
            further floored by the helper's byte budget.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None`` or ``groups`` is not the
            ``by_beta`` grouping.

    Notes
    -----
    The moves are enumerated in the inverse direction the owner-computes window
    pass uses: from the fragment's bra strings into each source group, which
    sizes the candidate grid by the fragment rather than the group. The
    inverse call swaps the label roles, so ``cr`` (occupied in the target of
    the call) and ``an`` (occupied in its source) are swapped back before
    :func:`~torch_sci_solver.torch_impl.mult._alphabeta_pair_matrix` evaluates
    the element on the bra strings. Because the enumeration starts from the
    fragment slab, every candidate is a fragment member and no pair-position
    window mask is needed. The source group's ``pair_idx1`` stays full, so the
    ket gather is unchanged from the whole-channel pass.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("add_ab_fragment requires pair_helpers.blocks (block_matvec != 'off')")
    if groups.axis != "beta":
        raise ValueError("add_ab_fragment requires the by_beta grouping")

    device = Wb.device
    dtype = integrals.I2_eri.dtype
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)

    # Same tile and enumeration caps as the whole-channel pass.
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_AB_BLOCK_ENTRY))
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 4)
    enum_a = _chunk_pairs(max(1, n_a * (n_orb - n_a)), "singles", enum_budget)
    enum_b = _chunk_pairs(max(1, n_b * (n_orb - n_b)), "singles", enum_budget)

    bra_group = groups.groups[frag.group]
    alpha_uids2 = bra_group.alpha_uids.to(device)
    beta_uids2 = bra_group.beta_uids.to(device)
    dst_alpha = alpha_uids2[frag.row_start : frag.row_stop]
    dst_beta = beta_uids2[frag.col_start : frag.col_stop]
    target_alpha = unique_alphas[dst_alpha]
    target_beta = unique_betas[dst_beta]
    n_dst_alpha = int(dst_alpha.numel())
    n_dst_beta = int(dst_beta.numel())
    n_cols_frag = frag.n_cols

    for source_group in groups.groups:
        pair_idx1 = source_group.pair_idx
        if pair_idx1 is not None:
            pair_idx1 = pair_idx1.to(device)
        n_cols1 = int(source_group.beta_uids.numel())
        alpha_uids1 = source_group.alpha_uids.to(device)
        beta_uids1 = source_group.beta_uids.to(device)
        # One gather per source group when the whole group vector fits a tile;
        # otherwise gather through pair_idx1 per tile. A structural group has
        # no index map, so its indices are computed from the group layout.
        ket_group = T[group_pair_indices(source_group)] if source_group.n_pairs <= chunk else None

        # The β table is enumerated once when the fragment's column slab fits
        # one chunk; a `None` table then skips the α enumeration entirely.
        beta_once: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
        if n_dst_beta <= enum_b:
            beta_once = _single_moves(
                dst_beta,
                beta_uids1,
                unique_betas,
                n_orb=n_orb,
                n_particles=n_b,
                spin_offset=1,
            )
            if beta_once is None:
                continue

        for a_start in range(0, n_dst_alpha, enum_a):
            a_end = min(a_start + enum_a, n_dst_alpha)
            moves_a = _single_moves(
                dst_alpha[a_start:a_end],
                alpha_uids1,
                unique_alphas,
                n_orb=n_orb,
                n_particles=n_a,
                spin_offset=0,
            )
            if moves_a is None:
                continue
            bra_a_chunk, r_src_a, cr_a_inv, an_a_inv = moves_a
            r_dst_a = a_start + bra_a_chunk  # position in the fragment's rows
            cr_a, an_a = an_a_inv, cr_a_inv

            b_starts: Iterable[int] = (
                (0,) if beta_once is not None else range(0, n_dst_beta, enum_b)
            )
            for b_start in b_starts:
                b_moves: tuple[Tensor, Tensor, Tensor, Tensor] | None
                if beta_once is not None:
                    b_moves = beta_once
                else:
                    b_moves = _single_moves(
                        dst_beta[b_start : b_start + enum_b],
                        beta_uids1,
                        unique_betas,
                        n_orb=n_orb,
                        n_particles=n_b,
                        spin_offset=1,
                    )
                if b_moves is None:
                    continue
                bra_b_chunk, c_src_b, cr_b_inv, an_b_inv = b_moves
                c_dst_b = b_start + bra_b_chunk  # position in the fragment's columns
                cr_b, an_b = an_b_inv, cr_b_inv

                m_a = int(r_src_a.numel())
                m_b = int(c_src_b.numel())
                csize_b = min(m_b, chunk)
                csize_a = max(1, chunk // csize_b)

                for a0 in range(0, m_a, csize_a):
                    a1 = min(a0 + csize_a, m_a)
                    src_rows = r_src_a[a0:a1]
                    bra_rows = r_dst_a[a0:a1]
                    src_base = src_rows * n_cols1
                    dst_base = bra_rows * n_cols_frag
                    bra_alpha_strings = target_alpha[bra_rows]
                    for b0 in range(0, m_b, csize_b):
                        b1 = min(b0 + csize_b, m_b)
                        src_pos = src_base.unsqueeze(1) + c_src_b[b0:b1].unsqueeze(0)
                        dst_pos = dst_base.unsqueeze(1) + c_dst_b[b0:b1].unsqueeze(0)

                        if ket_group is not None:
                            t_ket = ket_group[src_pos]
                        elif pair_idx1 is not None:
                            t_ket = T[pair_idx1[src_pos]]
                        else:
                            t_ket = T[group_flat_indices(source_group, src_pos.reshape(-1))]
                            t_ket = t_ket.reshape(src_pos.shape)
                        h = _alphabeta_pair_matrix(
                            cr_a[a0:a1],
                            an_a[a0:a1],
                            cr_b[b0:b1],
                            an_b[b0:b1],
                            bra_alpha_strings,
                            target_beta[c_dst_b[b0:b1]],
                            integrals,
                            dtype,
                        )
                        contrib = h * t_ket
                        Wb.index_add_(0, dst_pos.reshape(-1), contrib.reshape(-1))


def add_ab_slab(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    groups: SpinGroups,
    frag: PairFragment,
    *,
    src_row_offset: int,
    src_row_stop: int,
    src_col_offset: int,
    src_col_stop: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    memo: _MoveCache | None = None,
    memo_key: tuple = (),
) -> None:
    """Add one source slab's αβ-double contribution to a fragment block.

    The destination is the fragment rectangle
    ``[frag.row_start, frag.row_stop) × [frag.col_start, frag.col_stop)`` of
    ``pair_helpers.blocks.by_beta.groups[frag.group]``'s grid, written to
    ``Wb`` in row-major fragment order. The source is the rectangle
    ``[src_row_offset, src_row_stop) × [src_col_offset, src_col_stop)`` of the
    grid of the single ket group carried by ``groups``. ``T`` holds exactly
    that rectangle row-major with width ``src_col_stop - src_col_offset``:
    entry ``(r - src_row_offset) * width + (c - src_col_offset)`` is ket grid
    position ``(r, c)``. A slab belongs to one ket group, so ``groups`` carries
    just that group's block; summing over the source slabs of every group then
    reproduces the whole channel.

    Args:
        Wb: ``(frag.n_rows * frag.n_cols,)`` fragment output block, row-major
            over the destination rectangle; contributions are added in place.
        T: ``((src_row_stop - src_row_offset) * (src_col_stop - src_col_offset),)``
            resident ket amplitudes of the source rectangle, row-major with
            width ``src_col_stop - src_col_offset``. May be complex. Index
            ``(r - src_row_offset) * width + (c - src_col_offset)`` is the
            amplitude of ket grid position ``(r, c)``.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        groups: the ``by_beta`` grouping carrying the single ket source group
            whose staged rectangle ``T`` holds.
        frag: the destination (bra) fragment, built from
            ``pair_helpers.blocks.by_beta``.
        src_row_offset: first source row kept, in ket-grid coordinates.
        src_row_stop: one past the last source row kept.
        src_col_offset: first source column kept, in ket-grid coordinates.
        src_col_stop: one past the last source column kept.
        chunk_size: maximum lifted entries ``csize_a * csize_b`` per tile,
            further floored by the helper's byte budget.
        memo: optional stage-independent move-table cache. When given, the
            enumeration results are looked up and stored under ``memo_key`` so
            a fragment is enumerated once across source stages and matvecs.
        memo_key: Caller-supplied identity of ``(destination fragment, source
            group)``; the channel and chunk positions are appended internally.

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None``, ``groups`` is not the
            ``by_beta`` grouping, or ``groups`` carries more than one block.

    Notes
    -----
    The moves are enumerated in the inverse direction from the fragment's bra
    strings into each ket group, exactly as in :func:`add_ab_fragment`. The
    resulting source positions are then restricted to the staged rectangle:
    α positions to ``[src_row_offset, src_row_stop)`` and β positions to
    ``[src_col_offset, src_col_stop)``, with the survivors rebased by the
    offsets before indexing ``T``. The inverse call swaps the label roles, so
    ``cr``/``an`` are swapped back as in the fragment kernel. The rectangle is
    applied by masking the move tables, never by materializing the ket group's
    full grid; enumeration and tile chunking keep the caps of the whole-channel
    kernel.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("add_ab_slab requires pair_helpers.blocks (block_matvec != 'off')")
    if groups.axis != "beta":
        raise ValueError("add_ab_slab requires the by_beta grouping")
    if len(groups.groups) != 1:
        raise ValueError(
            "add_ab_slab consumes one ket group per call; pass a single-block grouping"
        )

    row_off = int(src_row_offset)
    row_stop = int(src_row_stop)
    col_off = int(src_col_offset)
    col_stop = int(src_col_stop)
    if frag.n_rows == 0 or frag.n_cols == 0:
        return
    if row_stop <= row_off or col_stop <= col_off:
        return

    device = Wb.device
    dtype = integrals.I2_eri.dtype
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)

    # Same tile and enumeration caps as the whole-channel pass.
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_AB_BLOCK_ENTRY))
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 4)
    enum_a = _chunk_pairs(max(1, n_a * (n_orb - n_a)), "singles", enum_budget)
    enum_b = _chunk_pairs(max(1, n_b * (n_orb - n_b)), "singles", enum_budget)

    width = col_stop - col_off

    bra_group = blocks.by_beta.groups[frag.group]
    alpha_uids2 = bra_group.alpha_uids.to(device)
    beta_uids2 = bra_group.beta_uids.to(device)
    dst_alpha = alpha_uids2[frag.row_start : frag.row_stop]
    dst_beta = beta_uids2[frag.col_start : frag.col_stop]
    target_alpha = unique_alphas[dst_alpha]
    target_beta = unique_betas[dst_beta]
    n_dst_alpha = int(dst_alpha.numel())
    n_dst_beta = int(dst_beta.numel())
    n_cols_frag = frag.n_cols

    for source_group in groups.groups:
        alpha_uids1 = source_group.alpha_uids.to(device)
        beta_uids1 = source_group.beta_uids.to(device)

        # The β table is enumerated once when the fragment's column slab fits
        # one chunk; a filtered-empty table then skips the α enumeration.
        beta_once: tuple[Tensor, ...] | None = None
        if n_dst_beta <= enum_b:
            beta_once = _memo_moves(
                memo,
                (*memo_key, "beta", 0, n_dst_beta),
                _single_moves,
                dst_beta,
                beta_uids1,
                unique_betas,
                n_orb=n_orb,
                n_particles=n_b,
                spin_offset=1,
            )
            if beta_once is not None:
                beta_once = _slab_moves(beta_once, col_off, col_stop)
            if beta_once is None:
                continue

        for a_start in range(0, n_dst_alpha, enum_a):
            a_end = min(a_start + enum_a, n_dst_alpha)
            moves_a = _memo_moves(
                memo,
                (*memo_key, "alpha", a_start, a_end),
                _single_moves,
                dst_alpha[a_start:a_end],
                alpha_uids1,
                unique_alphas,
                n_orb=n_orb,
                n_particles=n_a,
                spin_offset=0,
            )
            if moves_a is None:
                continue
            moves_a = _slab_moves(moves_a, row_off, row_stop)
            if moves_a is None:
                continue
            bra_a_chunk, src_rows, cr_a_inv, an_a_inv = moves_a
            r_dst_a = a_start + bra_a_chunk  # position in the fragment's rows
            cr_a, an_a = an_a_inv, cr_a_inv

            b_starts: Iterable[int] = (
                (0,) if beta_once is not None else range(0, n_dst_beta, enum_b)
            )
            b_moves: tuple[Tensor, ...] | None
            for b_start in b_starts:
                if beta_once is not None:
                    b_moves = beta_once
                else:
                    b_moves = _memo_moves(
                        memo,
                        (*memo_key, "beta", b_start, b_start + enum_b),
                        _single_moves,
                        dst_beta[b_start : b_start + enum_b],
                        beta_uids1,
                        unique_betas,
                        n_orb=n_orb,
                        n_particles=n_b,
                        spin_offset=1,
                    )
                    if b_moves is None:
                        continue
                    b_moves = _slab_moves(b_moves, col_off, col_stop)
                    if b_moves is None:
                        continue
                bra_b_chunk, src_cols, cr_b_inv, an_b_inv = b_moves
                c_dst_b = b_start + bra_b_chunk  # position in the fragment's columns
                cr_b, an_b = an_b_inv, cr_b_inv

                m_a = int(src_rows.numel())
                m_b = int(src_cols.numel())
                csize_b = min(m_b, chunk)
                csize_a = max(1, chunk // csize_b)

                for a0 in range(0, m_a, csize_a):
                    a1 = min(a0 + csize_a, m_a)
                    slab_rows = src_rows[a0:a1]
                    bra_rows = r_dst_a[a0:a1]
                    src_base = slab_rows * width
                    dst_base = bra_rows * n_cols_frag
                    bra_alpha_strings = target_alpha[bra_rows]
                    for b0 in range(0, m_b, csize_b):
                        b1 = min(b0 + csize_b, m_b)
                        # Slab-local positions into T's row-major rectangle.
                        src_pos = src_base.unsqueeze(1) + src_cols[b0:b1].unsqueeze(0)
                        dst_pos = dst_base.unsqueeze(1) + c_dst_b[b0:b1].unsqueeze(0)
                        h = _alphabeta_pair_matrix(
                            cr_a[a0:a1],
                            an_a[a0:a1],
                            cr_b[b0:b1],
                            an_b[b0:b1],
                            bra_alpha_strings,
                            target_beta[c_dst_b[b0:b1]],
                            integrals,
                            dtype,
                        )
                        contrib = h * T[src_pos]
                        Wb.index_add_(0, dst_pos.reshape(-1), contrib.reshape(-1))


def add_same_spin_fragment(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    groups: SpinGroups,
    frag: PairFragment,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
) -> None:
    """Add one fragment's contributions of the same-spin channels.

    The channel follows ``groups.axis``: the ``by_beta`` grouping runs the α
    channels (an α move keeps the β string fixed) and the ``by_alpha`` grouping
    runs the β channels. The fragment is the rectangle
    ``[row_start, row_stop) × [col_start, col_stop)`` of the group's grid, and
    ``Wb`` is its own ``frag.n_rows * frag.n_cols`` block in row-major order,
    i.e. position ``local_row * frag.n_cols + local_col``. The caller scatters
    the result into the tile through
    :meth:`~torch_sci_solver.torch_impl.pair_tiles.PairTileBounds.local_of_global`.

    The ``T`` layout is the one each channel reads, supplied by the caller:

    * α (``by_beta``): the group's β-column stage buffer
      ``T[pair_idx.view(n_rows, n_cols)[:, col_start:col_stop]]``, shape
      ``(|U_g|, frag.n_cols)``. An α move leaves β fixed, so ket α rows may be
      any row of the full group while the columns are the owned slab; the
      kernels need no ``T_stride`` (``M_a = frag.n_rows``, ``M_b = frag.n_cols``).
    * β (``by_alpha``): the fragment's α-row slab at the full group width,
      ``T[pair_idx.view(n_rows, n_cols)[row_start:row_stop]]``, shape
      ``(frag.n_rows, n_cols)``. A β move leaves α fixed, so the kernel reads
      ket β columns through ``T_stride = n_cols`` while ``M_b = frag.n_cols``.

    Args:
        Wb: ``(frag.n_rows * frag.n_cols,)`` fragment output block, row-major
            over the fragment rectangle; contributions are added in place.
        T: the channel's input layout described above. May be complex.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        groups: the grouping the fragment belongs to: ``by_beta`` for the α
            channels, ``by_alpha`` for the β channels.
        frag: the fragment rectangle to compute.
        chunk_size: maximum lifted entries per product-kernel tile.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel (real CUDA tensors only).

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None`` or ``groups.axis`` is
            neither ``"alpha"`` nor ``"beta"``.

    Notes
    -----
    Both channels enumerate the inverse moves from the fragment's owned bra
    strings into the full group, so the enumeration work scales with the
    fragment rather than the group, as in the owner-computes window pass. The
    inverse call swaps the label roles, so ``cr``/``an`` are swapped back
    before the product kernels evaluate the elements. The bra index is a
    full-group position offset by the enumeration chunk, which becomes
    output-local by subtracting ``frag.row_start`` (α) or ``frag.col_start``
    (β); the ket index stays a full-group row (α) or column (β). Because the
    enumeration starts from the fragment slab, every candidate is a fragment
    member and no pair-position window mask is needed.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError(
            "add_same_spin_fragment requires pair_helpers.blocks (block_matvec != 'off')"
        )
    axis = groups.axis
    if axis not in ("alpha", "beta"):
        raise ValueError(f"groups.axis must be 'alpha' or 'beta', got {axis!r}")

    device = Wb.device
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_SAME_SPIN_BLOCK_ENTRY))

    group = groups.groups[frag.group]
    alpha_uids = group.alpha_uids.to(device)
    beta_uids = group.beta_uids.to(device)
    n_cols = int(beta_uids.numel())

    # Same enumeration and survivor caps as the whole-channel pass.
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 2)
    n_empty_a = n_orb - n_a
    n_empty_b = n_orb - n_b
    single_a = n_a * n_empty_a
    single_b = n_b * n_empty_b
    double_a = (n_a * (n_a - 1) // 2) * (n_empty_a * (n_empty_a - 1) // 2)
    double_b = (n_b * (n_b - 1) // 2) * (n_empty_b * (n_empty_b - 1) // 2)
    enum_single_a = _chunk_pairs(max(1, single_a), "singles", enum_budget)
    enum_double_a = _chunk_pairs(max(1, double_a), "doubles", enum_budget)
    enum_single_b = _chunk_pairs(max(1, single_b), "singles", enum_budget)
    enum_double_b = _chunk_pairs(max(1, double_b), "doubles", enum_budget)
    element_bytes = int(integrals.I2_eri.element_size())
    survivor_single = _same_spin_survivor_chunk("singles", n_orb, enum_budget, element_bytes)
    survivor_double = _same_spin_survivor_chunk("doubles", n_orb, enum_budget, element_bytes)

    adets = unique_alphas[alpha_uids[frag.row_start : frag.row_stop]]
    bdets = unique_betas[beta_uids[frag.col_start : frag.col_stop]]
    out = Wb.new_zeros(frag.n_rows * frag.n_cols)

    if axis == "beta":
        # ---- α channels: the caller staged the fragment's β columns --------
        if single_a > 0:
            for start in range(frag.row_start, frag.row_stop, enum_single_a):
                end = min(start + enum_single_a, frag.row_stop)
                moves = _single_moves(
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves is None:
                    continue
                bra_chunk, ket, cr_inv, an_inv = moves
                bra = start + bra_chunk - frag.row_start
                cr, an = an_inv, cr_inv
                n_moves = int(bra.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_alpha_single_chunk_fast(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_alpha_per_exc=bra[m0:m1],
                        ket_alpha_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                    )

        if double_a > 0:
            for start in range(frag.row_start, frag.row_stop, enum_double_a):
                end = min(start + enum_double_a, frag.row_stop)
                moves_d = _double_moves(
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves_d is None:
                    continue
                bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                r_dst = start + bra_chunk - frag.row_start
                cr0, cr1 = an0_inv, an1_inv
                an0, an1 = cr0_inv, cr1_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=True,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        triton_matvec=triton_matvec,
                    )
    else:
        # ---- β channels: the caller supplied the full-width row slab --------
        if single_b > 0:
            for start in range(frag.col_start, frag.col_stop, enum_single_b):
                end = min(start + enum_single_b, frag.col_stop)
                moves = _single_moves(
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves is None:
                    continue
                bra_chunk, ket, cr_inv, an_inv = moves
                r_dst = start + bra_chunk - frag.col_start
                cr, an = an_inv, cr_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_beta_single_chunk_fast(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_beta_per_exc=r_dst[m0:m1],
                        ket_beta_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        T_stride=n_cols,
                    )

        if double_b > 0:
            for start in range(frag.col_start, frag.col_stop, enum_double_b):
                end = min(start + enum_double_b, frag.col_stop)
                moves_d = _double_moves(
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves_d is None:
                    continue
                bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                r_dst = start + bra_chunk - frag.col_start
                cr0, cr1 = an0_inv, an1_inv
                an0, an1 = cr0_inv, cr1_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=False,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        T_stride=n_cols,
                        triton_matvec=triton_matvec,
                    )

    Wb.add_(out)


def add_same_spin_slab(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    groups: SpinGroups,
    frag: PairFragment,
    *,
    row_offset: int,
    col_offset: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    memo: _MoveCache | None = None,
    memo_key: tuple = (),
    triton_matvec: bool = False,
) -> None:
    """Add one source slab's same-spin contributions to a fragment block.

    The channel follows ``groups.axis``: the ``by_beta`` grouping runs the α
    channels and the ``by_alpha`` grouping the β channels, as in
    :func:`add_same_spin_fragment`. ``Wb`` is the fragment's own
    ``frag.n_rows * frag.n_cols`` block in row-major order. The ket side is
    restricted to one source slab of the fragment's group:

    * α (``by_beta``): ``T`` is the rectangle
      ``[row_offset, row_offset + n_src_rows) × [frag.col_start, frag.col_stop)``
      of the group's grid, row-major with width ``frag.n_cols``; entry
      ``(r - row_offset) * frag.n_cols + c`` is the group grid position
      ``(r, frag.col_start + c)``. ``row_offset`` is the first source row in
      group coordinates and ``n_src_rows = T.numel() // frag.n_cols``. An α
      move leaves β fixed, so the ket columns are the fragment's own β
      columns. ``col_offset`` is ignored.
    * β (``by_alpha``): ``T`` is the rectangle
      ``[frag.row_start, frag.row_stop) × [col_offset, col_offset + n_src_cols)``
      of the group's grid, row-major with width ``n_src_cols``
      (``T_stride = n_src_cols``); entry ``(r - frag.row_start) * n_src_cols +
      c`` is the group grid position ``(frag.row_start + r, col_offset + c)``.
      ``col_offset`` is the first source column in group coordinates and
      ``n_src_cols = T.numel() // frag.n_rows``. A β move leaves α fixed, so
      the ket rows are the fragment's own α rows. ``row_offset`` is ignored.

    Args:
        Wb: ``(frag.n_rows * frag.n_cols,)`` fragment output block, row-major
            over the fragment rectangle; contributions are added in place.
        T: the channel's staged source-slab layout described above. May be
            complex and must be contiguous.
        pair_helpers: helpers for the same basis; its ``blocks`` field must be
            a non-``None`` :class:`~torch_sci_solver.torch_impl.pair_blocks.PairBlocks`.
        integrals: integral bundle on the output's device.
        groups: the grouping the fragment belongs to: ``by_beta`` for the α
            channels, ``by_alpha`` for the β channels.
        frag: the fragment rectangle to compute.
        row_offset: first source row in group coordinates (α path).
        col_offset: first source column in group coordinates (β path).
        chunk_size: maximum lifted entries per product-kernel tile.
        memo: optional stage-independent move-table cache. When given, the
            enumeration results are looked up and stored under ``memo_key`` so
            a fragment is enumerated once across source stages and matvecs.
        memo_key: Caller-supplied identity of ``(destination fragment, source
            group)``; the channel axis, order and chunk positions are appended
            internally.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel (real CUDA tensors only).

    Raises:
        ValueError: ``pair_helpers.blocks`` is ``None`` or ``groups.axis`` is
            neither ``"alpha"`` nor ``"beta"``.

    Notes
    -----
    Both channels enumerate the inverse moves from the fragment's bra strings
    into the full group, as in the fragment kernel, then keep only the moves
    whose ket position lies in the source slab and rebase it by the slab
    offset. The bra index stays a full-group position made output-local by
    subtracting ``frag.row_start`` (α) or ``frag.col_start`` (β). The slab is
    therefore applied as a mask on the move tables; the group grid is never
    materialized, and the enumeration and survivor caps match the fragment
    kernel.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("add_same_spin_slab requires pair_helpers.blocks (block_matvec != 'off')")
    axis = groups.axis
    if axis not in ("alpha", "beta"):
        raise ValueError(f"groups.axis must be 'alpha' or 'beta', got {axis!r}")

    if frag.n_rows == 0 or frag.n_cols == 0:
        return

    device = Wb.device
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_SAME_SPIN_BLOCK_ENTRY))

    group = groups.groups[frag.group]
    alpha_uids = group.alpha_uids.to(device)
    beta_uids = group.beta_uids.to(device)

    # Same enumeration and survivor caps as the whole-channel pass.
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 2)
    n_empty_a = n_orb - n_a
    n_empty_b = n_orb - n_b
    single_a = n_a * n_empty_a
    single_b = n_b * n_empty_b
    double_a = (n_a * (n_a - 1) // 2) * (n_empty_a * (n_empty_a - 1) // 2)
    double_b = (n_b * (n_b - 1) // 2) * (n_empty_b * (n_empty_b - 1) // 2)
    enum_single_a = _chunk_pairs(max(1, single_a), "singles", enum_budget)
    enum_double_a = _chunk_pairs(max(1, double_a), "doubles", enum_budget)
    enum_single_b = _chunk_pairs(max(1, single_b), "singles", enum_budget)
    enum_double_b = _chunk_pairs(max(1, double_b), "doubles", enum_budget)
    element_bytes = int(integrals.I2_eri.element_size())
    survivor_single = _same_spin_survivor_chunk("singles", n_orb, enum_budget, element_bytes)
    survivor_double = _same_spin_survivor_chunk("doubles", n_orb, enum_budget, element_bytes)

    adets = unique_alphas[alpha_uids[frag.row_start : frag.row_stop]]
    bdets = unique_betas[beta_uids[frag.col_start : frag.col_stop]]
    out = Wb.new_zeros(frag.n_rows * frag.n_cols)

    if axis == "beta":
        # ---- α channels: the source slab is an α-row slab -------------------
        n_src_rows = int(T.numel()) // frag.n_cols
        if n_src_rows == 0:
            return
        src_stop = int(row_offset) + n_src_rows
        if single_a > 0:
            for start in range(frag.row_start, frag.row_stop, enum_single_a):
                end = min(start + enum_single_a, frag.row_stop)
                moves = _memo_moves(
                    memo,
                    (*memo_key, axis, "single", start, end),
                    _single_moves,
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves is None:
                    continue
                moves = _slab_moves(moves, row_offset, src_stop)
                if moves is None:
                    continue
                bra_chunk, ket, cr_inv, an_inv = moves
                bra = start + bra_chunk - frag.row_start
                cr, an = an_inv, cr_inv
                n_moves = int(bra.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_alpha_single_chunk_fast(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_alpha_per_exc=bra[m0:m1],
                        ket_alpha_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                    )

        if double_a > 0:
            for start in range(frag.row_start, frag.row_stop, enum_double_a):
                end = min(start + enum_double_a, frag.row_stop)
                moves_d = _memo_moves(
                    memo,
                    (*memo_key, axis, "double", start, end),
                    _double_moves,
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves_d is None:
                    continue
                moves_d = _slab_moves(moves_d, row_offset, src_stop)
                if moves_d is None:
                    continue
                bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                r_dst = start + bra_chunk - frag.row_start
                cr0, cr1 = an0_inv, an1_inv
                an0, an1 = cr0_inv, cr1_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=True,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        triton_matvec=triton_matvec,
                    )
    else:
        # ---- β channels: the source slab is a β-column slab -----------------
        n_src_cols = int(T.numel()) // frag.n_rows
        if n_src_cols == 0:
            return
        src_stop = int(col_offset) + n_src_cols
        if single_b > 0:
            for start in range(frag.col_start, frag.col_stop, enum_single_b):
                end = min(start + enum_single_b, frag.col_stop)
                moves = _memo_moves(
                    memo,
                    (*memo_key, axis, "single", start, end),
                    _single_moves,
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves is None:
                    continue
                moves = _slab_moves(moves, col_offset, src_stop)
                if moves is None:
                    continue
                bra_chunk, ket, cr_inv, an_inv = moves
                r_dst = start + bra_chunk - frag.col_start
                cr, an = an_inv, cr_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_beta_single_chunk_fast(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_beta_per_exc=r_dst[m0:m1],
                        ket_beta_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        T_stride=n_src_cols,
                    )

        if double_b > 0:
            for start in range(frag.col_start, frag.col_stop, enum_double_b):
                end = min(start + enum_double_b, frag.col_stop)
                moves_d = _memo_moves(
                    memo,
                    (*memo_key, axis, "double", start, end),
                    _double_moves,
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves_d is None:
                    continue
                moves_d = _slab_moves(moves_d, col_offset, src_stop)
                if moves_d is None:
                    continue
                bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                r_dst = start + bra_chunk - frag.col_start
                cr0, cr1 = an0_inv, an1_inv
                an0, an1 = cr0_inv, cr1_inv
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        T,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=False,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=frag.n_rows,
                        M_b=frag.n_cols,
                        chunk_size=chunk,
                        T_stride=n_src_cols,
                        triton_matvec=triton_matvec,
                    )

    Wb.add_(out)


def _slab_moves(
    moves: tuple[Tensor, ...],
    offset: int,
    stop: int,
) -> tuple[Tensor, ...] | None:
    """Restrict a move table to source positions inside ``[offset, stop)``.

    Args:
        moves: the ``(bra, ket, *labels)`` tuple returned by :func:`_single_moves`
            or :func:`_double_moves`; ``ket`` holds positions in the source uid
            set.
        offset: first source position kept, in source-set coordinates.
        stop: one past the last source position kept.

    Returns:
        The same tuple with every array filtered to the surviving moves and
        ``ket`` rebased to ``[0, stop - offset)``, or ``None`` when no move
        survives. The bra positions and the ``cr``/``an`` labels need no
        rebase.
    """
    ket = moves[1]
    keep = (ket >= int(offset)) & (ket < int(stop))
    if not bool(keep.any()):
        return None
    return (moves[0][keep], ket[keep] - int(offset), *(m[keep] for m in moves[2:]))


def _move_degrees(
    source_uids: Tensor,
    target_uids: Tensor,
    unique: Tensor,
    *,
    n_orb: int,
    n_particles: int,
    spin_offset: int,
    chunk: int,
) -> Tensor:
    """Surviving single moves from each source string into the target uid set.

    Args:
        source_uids: ``(n_src,)`` int64 uids into ``unique``.
        target_uids: ``(n_dst,)`` sorted int64 uids into ``unique``.
        unique: sorted unique strings of this spin.
        n_orb: number of spatial orbitals.
        n_particles: electrons of this spin in every string.
        spin_offset: ``0`` for α, ``1`` for β.
        chunk: maximum source strings enumerated per call, which bounds the
            candidate grid.

    Returns:
        ``(n_src,)`` int64 counts; for each source string, how many of its
        single excitations land in the target set. Zero everywhere when
        either spin count makes singles impossible.
    """
    n_src = int(source_uids.numel())
    step = max(1, int(chunk))
    degrees = torch.zeros(n_src, dtype=torch.int64, device=source_uids.device)
    for start in range(0, n_src, step):
        stop = min(start + step, n_src)
        moves = _single_moves(
            source_uids[start:stop],
            target_uids,
            unique,
            n_orb=n_orb,
            n_particles=n_particles,
            spin_offset=spin_offset,
        )
        if moves is None:
            continue
        width = stop - start
        degrees[start:stop] = torch.bincount(moves[0], minlength=width)[:width]
    return degrees


def _ab_entry_weights(pair_helpers: PairHelpers) -> Tensor:
    """Per-pair count of the αβ lifted entries the block kernel emits.

    For a pair ``(a, b)`` in ``by_beta`` group ``g`` the kernel walks every ket
    group ``g1``, enumerates the inverse single moves from ``a`` into ``U_g1``
    and from ``b`` into ``B_g1``, and crosses them, so the pair contributes
    ``deg_α(a → U_g1) · deg_β(b → B_g1)`` entries per ket group. The counts
    depend only on the strings, so each destination group's grid is one sum of
    outer products of per-string degree vectors.

    Args:
        pair_helpers: helpers carrying the exact block decomposition.

    Returns:
        ``(N,)`` float64 entry counts on the helpers' device.

    Raises:
        ValueError: the helpers have no block decomposition.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("_ab_entry_weights requires pair_helpers.blocks")
    if blocks.structural:
        raise ValueError(
            "_ab_entry_weights is unavailable for a structural pair basis: the "
            "per-pair weight vector is length N; partition a structural basis by "
            "pair count instead."
        )
    device = pair_helpers.device
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    alpha_of_pair = pair_helpers.alpha_of_pair
    assert alpha_of_pair is not None
    weights = torch.zeros(int(alpha_of_pair.shape[0]), dtype=torch.float64, device=device)
    groups = blocks.by_beta.groups
    # Same enumeration calibration as the αβ matvec pass: a quarter of the
    # budget per side keeps the temporary candidate grid small on every rank.
    chunk = _chunk_pairs(
        max(1, n_a * (n_orb - n_a)), "singles", int(pair_helpers.budget_bytes) // 4
    )
    for group in groups:
        pair_idx = group.pair_idx
        assert pair_idx is not None
        pair_idx = pair_idx.to(device)
        alpha_uids = group.alpha_uids.to(device)
        beta_uids = group.beta_uids.to(device)
        n_rows = int(alpha_uids.numel())
        n_cols = int(beta_uids.numel())
        if n_rows == 0 or n_cols == 0:
            continue
        deg_a = [
            _move_degrees(
                alpha_uids,
                ket.alpha_uids.to(device),
                unique_alphas,
                n_orb=n_orb,
                n_particles=n_a,
                spin_offset=0,
                chunk=chunk,
            ).to(torch.float64)
            for ket in groups
        ]
        deg_b = [
            _move_degrees(
                beta_uids,
                ket.beta_uids.to(device),
                unique_betas,
                n_orb=n_orb,
                n_particles=n_b,
                spin_offset=1,
                chunk=chunk,
            ).to(torch.float64)
            for ket in groups
        ]
        row_chunk = max(1, _WEIGHT_SLAB_ENTRIES // n_cols)
        for r0 in range(0, n_rows, row_chunk):
            r1 = min(r0 + row_chunk, n_rows)
            entries = torch.zeros(r1 - r0, n_cols, dtype=torch.float64, device=device)
            for da, db in zip(deg_a, deg_b):
                entries += da[r0:r1].unsqueeze(1) * db.unsqueeze(0)
            weights[pair_idx[r0 * n_cols : r1 * n_cols]] = entries.reshape(-1)
    return weights


def pair_work_weights(pair_helpers: PairHelpers) -> Tensor:
    """Per-pair block-matvec cost estimate, for a work-balanced partition.

    The two block channels scale with different per-pair quantities. A
    same-spin pass enumerates every candidate move of each owned source
    string, and a string's candidate count is fixed by its spin, so a
    ``by_beta`` group's α cost is spread over its β columns and a ``by_alpha``
    group's β cost over its α rows: the per-pair weight is
    ``(n_αs + n_αd) / M_b`` from the pair's ``by_beta`` group plus
    ``(n_βs + n_βd) / M_a`` from its ``by_alpha`` group. The αβ pass lifts one
    entry per crossed (inverse α move, inverse β move), so its per-pair weight
    is the entry count from :func:`_ab_entry_weights`, converted to candidate
    units by :data:`_AB_ENTRY_COST`. Only the group geometry and the spin
    counts are read for the same-spin term; the αβ term enumerates single
    moves in bounded chunks at call time and stores nothing.

    Args:
        pair_helpers: helpers carrying the exact block decomposition.

    Returns:
        ``(N,)`` float64 non-negative weights on the helpers' device.

    Raises:
        ValueError: the helpers have no block decomposition.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("pair_work_weights requires pair_helpers.blocks")
    if blocks.structural:
        raise ValueError(
            "pair_work_weights is unavailable for a structural pair basis: the "
            "per-pair weight vector is length N. Partition the structural basis "
            "by pair count instead."
        )
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    n_orb = int(pair_helpers.n_orb)
    v_a = n_orb - n_a
    v_b = n_orb - n_b
    n_as = n_a * v_a
    n_ad = (n_a * (n_a - 1) // 2) * (v_a * (v_a - 1) // 2)
    n_bs = n_b * v_b
    n_bd = (n_b * (n_b - 1) // 2) * (v_b * (v_b - 1) // 2)
    device = pair_helpers.device
    alpha_of_pair = pair_helpers.alpha_of_pair
    assert alpha_of_pair is not None
    weights = torch.zeros(int(alpha_of_pair.shape[0]), dtype=torch.float64, device=device)
    cand_a = float(n_as + n_ad)
    cand_b = float(n_bs + n_bd)
    for group in blocks.by_beta.groups:
        n_cols = int(group.beta_uids.numel())
        if n_cols:
            weights[_materialised_pair_idx(group).to(device)] += cand_a / n_cols
    for group in blocks.by_alpha.groups:
        n_rows = int(group.alpha_uids.numel())
        if n_rows:
            weights[_materialised_pair_idx(group).to(device)] += cand_b / n_rows
    weights += _AB_ENTRY_COST * _ab_entry_weights(pair_helpers)
    return weights


def _single_moves(
    source_uids: Tensor,
    target_uids: Tensor,
    unique: Tensor,
    *,
    n_orb: int,
    n_particles: int,
    spin_offset: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    """Enumerate single excitations between two sorted uid sets of one spin.

    For every source string ``unique[source_uids[r]]``, annihilate each of its
    ``n_particles`` occupied spatial orbitals in turn, create each of its
    ``n_orb - n_particles`` empty ones, and keep the resulting string when it
    equals ``unique[target_uids[d]]``. Because ``target_uids`` is sorted and
    ``unique`` is sorted, the target set is sorted and the membership test is
    one ``searchsorted`` (or a lexicographic binary search for multi-limb strings).

    Args:
        source_uids: ``(n_src,)`` int64 uids into ``unique`` (the ket group).
        target_uids: ``(n_dst,)`` sorted int64 uids into ``unique`` (the bra
            group).
        unique: sorted unique strings of this spin. May be 1-D ``(M,)`` or
            2-D ``(M, n_limbs)`` limb matrices.
        n_orb: number of spatial orbitals.
        n_particles: electrons of this spin in every string.
        spin_offset: ``0`` for α, ``1`` for β; spin-orbital index =
            ``2 * spatial + spin_offset``.

    Returns:
        ``(r_src, r_dst, cr, an)`` — positions in ``source_uids`` and
        ``target_uids`` plus the spin-orbital indices that take the source to
        the target: ``cr`` is occupied in the target, ``an`` in the source.
        ``None`` when the two sets are empty, or no move lands in the target
        set.
    """
    n_src = int(source_uids.numel())
    n_dst = int(target_uids.numel())
    n_empty = int(n_orb) - int(n_particles)
    if n_src == 0 or n_dst == 0 or n_particles <= 0 or n_empty <= 0:
        return None

    device = unique.device
    src = unique[source_uids]
    # A structural group's uid list is ascending but its bitmasks need not be,
    # so sort the target values and map positions back to the caller's order.
    dst, target_order = _sort_strings(unique[target_uids])

    occ = _dispatch_bit_positions(src, n_orb, n_particles)
    empty = _dispatch_bit_positions(src ^ _complement_mask(src, n_orb), n_orb, n_empty)

    n_per = n_particles * n_empty
    an = occ.unsqueeze(-1).expand(-1, -1, n_empty).reshape(n_src, n_per)
    cr = empty.unsqueeze(-2).expand(-1, n_particles, -1).reshape(n_src, n_per)
    if unique.dim() == 1:
        one = torch.tensor(1, dtype=torch.int64, device=device)
        target = src.unsqueeze(-1) ^ (one << an) ^ (one << cr)
        found, dst_pos = _dispatch_found_in(dst, target)
    else:
        n_limbs = int(unique.shape[-1])
        target = (
            src.unsqueeze(-2) ^ _bit_mask_rows(an, n_limbs) ^ _bit_mask_rows(cr, n_limbs)
        ).reshape(-1, n_limbs)
        found, dst_pos = _dispatch_found_in(dst, target)
        found = found.view(n_src, n_per)
        dst_pos = dst_pos.view(n_src, n_per)

    flat = found.reshape(-1).nonzero(as_tuple=True)[0]
    if flat.numel() == 0:
        return None
    r_src = torch.div(flat, n_per, rounding_mode="floor")
    an_flat = an.reshape(-1)[flat]
    cr_flat = cr.reshape(-1)[flat]
    # cr is empty in the source and therefore occupied in the target (bra);
    # an is occupied in the source (ket). That is the generic table's labelling.
    return (
        r_src,
        target_order[dst_pos.reshape(-1)[flat]],
        (cr_flat << 1) + spin_offset,
        (an_flat << 1) + spin_offset,
    )


def _double_moves(
    source_uids: Tensor,
    target_uids: Tensor,
    unique: Tensor,
    *,
    n_orb: int,
    n_particles: int,
    spin_offset: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    """Enumerate double excitations between two sorted uid sets of one spin.

    For every source string ``unique[source_uids[r]]``, annihilate each pair of
    its occupied spatial orbitals and create each pair of its empty ones
    (pairs taken in ascending orbital order), and keep the resulting string
    when it equals ``unique[target_uids[d]]``. The target lookup is one
    ``searchsorted`` (or a lexicographic binary search for multi-limb strings).

    Args:
        source_uids: ``(n_src,)`` int64 uids into ``unique`` (the ket group).
        target_uids: ``(n_dst,)`` sorted int64 uids into ``unique`` (the bra
            group).
        unique: sorted unique strings of this spin. May be 1-D ``(M,)`` or
            2-D ``(M, n_limbs)`` limb matrices.
        n_orb: number of spatial orbitals.
        n_particles: electrons of this spin in every string.
        spin_offset: ``0`` for α, ``1`` for β; spin-orbital index =
            ``2 * spatial + spin_offset``.

    Returns:
        ``(r_src, r_dst, cr0, cr1, an0, an1)`` — positions in ``source_uids``
        and ``target_uids`` plus the spin-orbital indices that take the source
        to the target: ``cr0 < cr1`` are occupied in the target, ``an0 < an1``
        in the source. Each unordered (annihilation pair, creation pair) is
        enumerated once. ``None`` when the two sets are empty, either spin has
        fewer than two electrons or empty orbitals, or no move lands in the
        target set.
    """
    n_src = int(source_uids.numel())
    n_dst = int(target_uids.numel())
    n_empty = int(n_orb) - int(n_particles)
    if n_src == 0 or n_dst == 0 or n_particles < 2 or n_empty < 2:
        return None

    device = unique.device
    src = unique[source_uids]
    # See _single_moves: the caller's target order need not be value-ascending.
    dst, target_order = _sort_strings(unique[target_uids])

    occ = _dispatch_bit_positions(src, n_orb, n_particles)
    empty = _dispatch_bit_positions(src ^ _complement_mask(src, n_orb), n_orb, n_empty)

    an_i, an_j = _triu_indices(n_particles, device=device)
    cr_i, cr_j = _triu_indices(n_empty, device=device)
    n_an = int(an_i.numel())
    n_cr = int(cr_i.numel())
    n_per = n_an * n_cr
    an0 = occ[:, an_i].unsqueeze(-1).expand(-1, -1, n_cr).reshape(n_src, n_per)
    an1 = occ[:, an_j].unsqueeze(-1).expand(-1, -1, n_cr).reshape(n_src, n_per)
    cr0 = empty[:, cr_i].unsqueeze(-2).expand(-1, n_an, -1).reshape(n_src, n_per)
    cr1 = empty[:, cr_j].unsqueeze(-2).expand(-1, n_an, -1).reshape(n_src, n_per)
    if unique.dim() == 1:
        one = torch.tensor(1, dtype=torch.int64, device=device)
        target = src.unsqueeze(-1) ^ (one << an0) ^ (one << an1) ^ (one << cr0) ^ (one << cr1)
        found, dst_pos = _dispatch_found_in(dst, target)
    else:
        n_limbs = int(unique.shape[-1])
        target = (
            src.unsqueeze(-2)
            ^ _bit_mask_rows(an0, n_limbs)
            ^ _bit_mask_rows(an1, n_limbs)
            ^ _bit_mask_rows(cr0, n_limbs)
            ^ _bit_mask_rows(cr1, n_limbs)
        ).reshape(-1, n_limbs)
        found, dst_pos = _dispatch_found_in(dst, target)
        found = found.view(n_src, n_per)
        dst_pos = dst_pos.view(n_src, n_per)

    flat = found.reshape(-1).nonzero(as_tuple=True)[0]
    if flat.numel() == 0:
        return None
    r_src = torch.div(flat, n_per, rounding_mode="floor")
    # cr is empty in the source and therefore occupied in the target (bra);
    # an is occupied in the source (ket). That is the generic table's labelling,
    # and both pairs ascend in orbital order as in the streamed enumerators.
    return (
        r_src,
        target_order[dst_pos.reshape(-1)[flat]],
        (cr0.reshape(-1)[flat] << 1) + spin_offset,
        (cr1.reshape(-1)[flat] << 1) + spin_offset,
        (an0.reshape(-1)[flat] << 1) + spin_offset,
        (an1.reshape(-1)[flat] << 1) + spin_offset,
    )


def _ab_block_pass(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    row_start: int | None,
    chunk_size: int,
    cache: _MoveCache | None = None,
) -> None:
    """Cross-lift the α and β move tables of every ordered group pair into ``Wb``.

    Args:
        Wb: output vector (full) or owned block (``row_start`` is not ``None``).
        T: input vector (full, and also the source of ket amplitudes).
        pair_helpers: helpers carrying the block decomposition and the uid
            tables.
        integrals: integral bundle.
        row_start: first owned global pair, or ``None`` for a full matvec.
        chunk_size: lifted-entry cap per tile before the byte-budget floor.
        cache: optional move-table cache for the single-move enumerations.

    Notes
    -----
    Each side's move table is enumerated in chunks sized by ``_chunk_pairs``
    with a quarter of ``budget_bytes``, so a group pair's candidate grid is
    never materialized whole. A full matvec chunks the *source* side of each
    move; an owner-computes matvec chunks the *destination* side and enumerates
    the inverse moves from the owned bra strings, which sizes the candidate
    grid by the owned rows. The β side is pre-enumerated once when its chunked
    set fits one chunk, which also lets an empty β table skip the α
    enumeration; otherwise the β table is re-derived per α chunk so at most one
    chunk of each side is live at a time.

    The move tables depend only on the basis (and, for the inverse
    enumeration, the owned bra window), so ``cache`` keys them by the channel,
    ordered group pair, spin side, uid range and string width. The full and
    owner-computes modes never share a key: an inverse table's source is the
    destination group's owned slab, not the ket group.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("add_ab_blocks requires pair_helpers.blocks (block_matvec != 'off')")

    device = Wb.device
    dtype = integrals.I2_eri.dtype
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    width_a = int(unique_alphas.shape[-1]) if unique_alphas.dim() > 1 else 1
    width_b = int(unique_betas.shape[-1]) if unique_betas.dim() > 1 else 1
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    local = row_start is not None
    first_row = 0 if row_start is None else int(row_start)
    row_stop = first_row + int(Wb.numel()) if local else 0

    # Tiles are capped by the caller's entry count and by the helper's byte
    # budget with this kernel's (larger) per-entry footprint.
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_AB_BLOCK_ENTRY))

    # Move enumeration is chunked by source strings on the streamed
    # enumerators' int64-temporary calibration, with a quarter of the byte
    # budget per side so the other side and the lifted tiles still fit. This
    # keeps the candidate grid bounded regardless of group shape: a whole-set
    # enumeration would scale as ``n_strings * n_a * n_empty`` per group pair.
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 4)
    enum_a = _chunk_pairs(max(1, n_a * (n_orb - n_a)), "singles", enum_budget)
    enum_b = _chunk_pairs(max(1, n_b * (n_orb - n_b)), "singles", enum_budget)

    for g2_id, g2 in enumerate(blocks.by_beta.groups):
        pair_idx2 = g2.pair_idx
        assert pair_idx2 is not None
        pair_idx2 = pair_idx2.to(device)
        n_cols2 = int(g2.beta_uids.numel())
        alpha_uids2 = g2.alpha_uids.to(device)
        beta_uids2 = g2.beta_uids.to(device)
        target_alpha = unique_alphas[alpha_uids2]
        target_beta = unique_betas[beta_uids2]

        if local:
            bounds = torch.tensor((first_row, row_stop), dtype=torch.int64, device=device)
            lo, hi = (int(v) for v in torch.searchsorted(pair_idx2, bounds).tolist())
            if lo >= hi:
                continue  # this rank owns no pair of the bra group
            owned_local = pair_idx2[lo:hi] - first_row
            row_lo = lo // n_cols2
            row_hi = (hi + n_cols2 - 1) // n_cols2
            # The owned window owns every column of an interior row; only a
            # single-row window can restrict its β column range.
            if row_hi - row_lo == 1:
                col_lo = lo % n_cols2
                col_hi = (hi - 1) % n_cols2 + 1
            else:
                col_lo, col_hi = 0, n_cols2
            dst_alpha = alpha_uids2[row_lo:row_hi]
            dst_beta = beta_uids2[col_lo:col_hi]
        else:
            lo = hi = row_lo = row_hi = 0
            col_lo, col_hi = 0, n_cols2
            dst_alpha = alpha_uids2[:0]
            dst_beta = beta_uids2[:0]
            owned_local = None

        for g1_id, g1 in enumerate(blocks.by_beta.groups):
            pair_idx1 = g1.pair_idx
            assert pair_idx1 is not None
            pair_idx1 = pair_idx1.to(device)
            n_cols1 = int(g1.beta_uids.numel())
            alpha_uids1 = g1.alpha_uids.to(device)
            beta_uids1 = g1.beta_uids.to(device)
            n_alpha_src = int(alpha_uids1.numel())
            n_beta_src = int(beta_uids1.numel())

            # One gather per ket group when the whole group vector fits a tile;
            # otherwise gather through pair_idx1 per tile.
            ket_group = T[pair_idx1] if int(pair_idx1.numel()) <= chunk else None

            # Enumerate the β moves once when the chunked side fits one chunk —
            # the common asymmetric case, e.g. a single β string. A full pass
            # chunks the source group; an owner-computes pass chunks the owned
            # bra slab, so every rank enumerates only the moves into its own
            # rows. A `None` table then skips the α enumeration entirely.
            # Otherwise the β table is re-derived per α chunk, so at most one
            # chunk of each side is live.
            beta_once: tuple[Tensor, ...] | None = None
            if local:
                n_dst_alpha = int(dst_alpha.numel())
                n_dst_beta = int(dst_beta.numel())
                if n_dst_beta <= enum_b:
                    beta_once = _memo_moves(
                        cache,
                        (
                            "ab_local",
                            "beta",
                            g2_id,
                            g1_id,
                            first_row,
                            row_stop,
                            0,
                            n_dst_beta,
                            width_b,
                        ),
                        _single_moves,
                        dst_beta,
                        beta_uids1,
                        unique_betas,
                        n_orb=n_orb,
                        n_particles=n_b,
                        spin_offset=1,
                    )
                    if beta_once is None:
                        continue
                alpha_starts = range(0, n_dst_alpha, enum_a)
            else:
                n_dst_alpha = n_dst_beta = 0
                if n_beta_src <= enum_b:
                    beta_once = _memo_moves(
                        cache,
                        ("ab", "beta", g2_id, g1_id, 0, n_beta_src, width_b),
                        _single_moves,
                        beta_uids1,
                        beta_uids2,
                        unique_betas,
                        n_orb=n_orb,
                        n_particles=n_b,
                        spin_offset=1,
                    )
                    if beta_once is None:
                        continue
                alpha_starts = range(0, n_alpha_src, enum_a)

            for a_start in alpha_starts:
                if local:
                    a_end = min(a_start + enum_a, n_dst_alpha)
                    moves_a = _memo_moves(
                        cache,
                        (
                            "ab_local",
                            "alpha",
                            g2_id,
                            g1_id,
                            first_row,
                            row_stop,
                            a_start,
                            a_end,
                            width_a,
                        ),
                        _single_moves,
                        dst_alpha[a_start:a_end],
                        alpha_uids1,
                        unique_alphas,
                        n_orb=n_orb,
                        n_particles=n_a,
                        spin_offset=0,
                    )
                    if moves_a is None:
                        continue
                    bra_a_chunk, r_src_a, cr_a_inv, an_a_inv = moves_a
                    r_dst_a = row_lo + a_start + bra_a_chunk
                    cr_a, an_a = an_a_inv, cr_a_inv
                else:
                    a_end = min(a_start + enum_a, n_alpha_src)
                    moves_a = _memo_moves(
                        cache,
                        ("ab", "alpha", g2_id, g1_id, a_start, a_end, width_a),
                        _single_moves,
                        alpha_uids1[a_start:a_end],
                        alpha_uids2,
                        unique_alphas,
                        n_orb=n_orb,
                        n_particles=n_a,
                        spin_offset=0,
                    )
                    if moves_a is None:
                        continue
                    r_src_a, r_dst_a, cr_a, an_a = moves_a
                    r_src_a = r_src_a + a_start  # position in the full ket group

                b_starts: Iterable[int] = (
                    (0,)
                    if beta_once is not None
                    else (range(0, n_dst_beta, enum_b) if local else range(0, n_beta_src, enum_b))
                )
                b_moves: tuple[Tensor, ...] | None
                for b_start in b_starts:
                    if beta_once is not None:
                        b_moves = beta_once
                    elif local:
                        b_end = min(b_start + enum_b, n_dst_beta)
                        b_moves = _memo_moves(
                            cache,
                            (
                                "ab_local",
                                "beta",
                                g2_id,
                                g1_id,
                                first_row,
                                row_stop,
                                b_start,
                                b_end,
                                width_b,
                            ),
                            _single_moves,
                            dst_beta[b_start:b_end],
                            beta_uids1,
                            unique_betas,
                            n_orb=n_orb,
                            n_particles=n_b,
                            spin_offset=1,
                        )
                    else:
                        b_end = min(b_start + enum_b, n_beta_src)
                        b_moves = _memo_moves(
                            cache,
                            ("ab", "beta", g2_id, g1_id, b_start, b_end, width_b),
                            _single_moves,
                            beta_uids1[b_start:b_end],
                            beta_uids2,
                            unique_betas,
                            n_orb=n_orb,
                            n_particles=n_b,
                            spin_offset=1,
                        )
                    if b_moves is None:
                        continue
                    if local:
                        bra_b_chunk, c_src_b, cr_b_inv, an_b_inv = b_moves
                        c_dst_b = col_lo + b_start + bra_b_chunk
                        cr_b, an_b = an_b_inv, cr_b_inv
                    else:
                        c_src_b, c_dst_b, cr_b, an_b = b_moves
                        c_src_b = c_src_b + b_start  # position in the full ket group

                    m_a = int(r_src_a.numel())
                    m_b = int(c_src_b.numel())
                    csize_b = min(m_b, chunk)
                    csize_a = max(1, chunk // csize_b)

                    for a0 in range(0, m_a, csize_a):
                        a1 = min(a0 + csize_a, m_a)
                        src_rows = r_src_a[a0:a1]
                        dst_rows = r_dst_a[a0:a1]
                        src_base = src_rows * n_cols1
                        dst_base = dst_rows * n_cols2
                        bra_alpha_strings = target_alpha[dst_rows]
                        for b0 in range(0, m_b, csize_b):
                            b1 = min(b0 + csize_b, m_b)
                            src_pos = src_base.unsqueeze(1) + c_src_b[b0:b1].unsqueeze(0)
                            dst_pos = dst_base.unsqueeze(1) + c_dst_b[b0:b1].unsqueeze(0)

                            if ket_group is not None:
                                t_ket = ket_group[src_pos]
                            else:
                                t_ket = T[pair_idx1[src_pos]]
                            h = _alphabeta_pair_matrix(
                                cr_a[a0:a1],
                                an_a[a0:a1],
                                cr_b[b0:b1],
                                an_b[b0:b1],
                                bra_alpha_strings,
                                target_beta[c_dst_b[b0:b1]],
                                integrals,
                                dtype,
                            )
                            contrib = h * t_ket

                            if owned_local is None:
                                Wb.index_add_(
                                    0, pair_idx2[dst_pos].reshape(-1), contrib.reshape(-1)
                                )
                            else:
                                keep = (dst_pos >= lo) & (dst_pos < hi)
                                out_idx = owned_local[dst_pos[keep] - lo]
                                Wb.index_add_(0, out_idx, contrib[keep])


def _group_window(pair_idx: Tensor, first_row: int, row_stop: int) -> tuple[int, int]:
    """Locate an owned global pair range inside one group's sorted index map.

    Args:
        pair_idx: ``(n_group,)`` strictly increasing global pair indices.
        first_row: first owned global pair (inclusive).
        row_stop: one past the last owned global pair.

    Returns:
        ``(lo, hi)`` — the group positions whose global index lies in
        ``[first_row, row_stop)``. ``lo == hi`` when the group has no owned row.
    """
    bounds = torch.tensor((first_row, row_stop), dtype=torch.int64, device=pair_idx.device)
    lo, hi = (int(v) for v in torch.searchsorted(pair_idx, bounds).tolist())
    return lo, hi


def _same_spin_block_pass(
    Wb: Tensor,
    T: Tensor,
    pair_helpers: PairHelpers,
    integrals: TorchIntegrals,
    *,
    row_start: int | None,
    chunk_size: int,
    triton_matvec: bool = False,
    cache: _MoveCache | None = None,
) -> None:
    """Run the four same-spin channels group by group.

    Args:
        Wb: output vector (full) or owned block (``row_start`` is not ``None``).
        T: input vector (full, and also the source of ket amplitudes).
        pair_helpers: helpers carrying the block decomposition and the uid
            tables.
        integrals: integral bundle.
        row_start: first owned global pair, or ``None`` for a full matvec.
        chunk_size: lifted-entry cap per product-kernel tile, floored by the
            helper's byte budget.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel.
        cache: optional move-table cache for the single/double enumerations.

    Notes
    -----
    Enumeration chunks bound the candidate grid, and each resulting survivor
    table is sliced so one product-kernel call's per-survivor setup also fits
    the budget (see :func:`_same_spin_survivor_chunk`). Sparse survival
    therefore costs calls proportional to the row count, not to the candidate
    count, while a dense-survival group stays bounded.

    Each chunk's move table depends only on the basis and the group's own uid
    list, so ``cache`` keys them by the spin axis, group, single/double order,
    uid range and string width. A full and an owner-computes pass enumerate the
    same arguments for the same range and therefore share the entry.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError(
            "add_same_spin_blocks requires pair_helpers.blocks (block_matvec != 'off')"
        )

    device = Wb.device
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    width_a = int(unique_alphas.shape[-1]) if unique_alphas.dim() > 1 else 1
    width_b = int(unique_betas.shape[-1]) if unique_betas.dim() > 1 else 1
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    local = row_start is not None
    first_row = 0 if row_start is None else int(row_start)
    row_stop = first_row + int(Wb.numel()) if local else 0
    chunk = max(1, int(chunk_size))
    chunk = max(1, min(chunk, int(pair_helpers.budget_bytes) // _BYTES_PER_SAME_SPIN_BLOCK_ENTRY))

    # Enumeration chunks bound the candidate grid (streamed-enumerator
    # calibration at half the budget). The kernels' per-survivor setup is a
    # separate cost, so each survivor table is sliced to its own cap below;
    # sparse survival therefore costs calls proportional to nnz, not to the
    # candidate count.
    enum_budget = max(1, int(pair_helpers.budget_bytes) // 2)
    n_empty_a = n_orb - n_a
    n_empty_b = n_orb - n_b
    single_a = n_a * n_empty_a
    single_b = n_b * n_empty_b
    double_a = (n_a * (n_a - 1) // 2) * (n_empty_a * (n_empty_a - 1) // 2)
    double_b = (n_b * (n_b - 1) // 2) * (n_empty_b * (n_empty_b - 1) // 2)
    enum_single_a = _chunk_pairs(max(1, single_a), "singles", enum_budget)
    enum_double_a = _chunk_pairs(max(1, double_a), "doubles", enum_budget)
    enum_single_b = _chunk_pairs(max(1, single_b), "singles", enum_budget)
    enum_double_b = _chunk_pairs(max(1, double_b), "doubles", enum_budget)
    element_bytes = int(integrals.I2_eri.element_size())
    survivor_single = _same_spin_survivor_chunk("singles", n_orb, enum_budget, element_bytes)
    survivor_double = _same_spin_survivor_chunk("doubles", n_orb, enum_budget, element_bytes)

    # ---- α channels: one by_beta group, β string fixed --------------------
    for g_id, g in enumerate(blocks.by_beta.groups):
        pair_idx = g.pair_idx
        assert pair_idx is not None
        pair_idx = pair_idx.to(device)
        alpha_uids = g.alpha_uids.to(device)
        n_group_rows = int(alpha_uids.numel())
        n_cols = int(g.beta_uids.numel())
        bdets = unique_betas[g.beta_uids.to(device)]
        # Ket rows may be any α string of the group: gather the whole group.
        t_group = T[pair_idx]
        if local:
            lo, hi = _group_window(pair_idx, first_row, row_stop)
            if lo >= hi:
                continue  # this rank owns no pair of the group
            row_lo = lo // n_cols
            n_rows = (hi + n_cols - 1) // n_cols - row_lo
            adets = unique_alphas[alpha_uids[row_lo : row_lo + n_rows]]
            out = Wb.new_zeros(n_rows * n_cols)
        else:
            lo = hi = row_lo = 0
            n_rows = n_group_rows
            adets = unique_alphas[alpha_uids]
            out = Wb.new_zeros(int(pair_idx.numel()))

        if single_a > 0:
            # Full matvecs enumerate away from every source string; an
            # owner-computes matvec enumerates the inverse moves into its owned
            # bra slab, so the candidate grid is sized by the owned rows rather
            # than the whole group. The inverse call swaps the label roles
            # (``cr`` is occupied in the ket there), hence the swap below.
            lo_row = row_lo if local else 0
            hi_row = row_lo + n_rows if local else n_group_rows
            for start in range(lo_row, hi_row, enum_single_a):
                end = min(start + enum_single_a, hi_row)
                moves = _memo_moves(
                    cache,
                    ("same_spin", "alpha", g_id, "single", start, end, width_a),
                    _single_moves,
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves is None:
                    continue
                if local:
                    bra_chunk, ket, cr_inv, an_inv = moves
                    bra = start + bra_chunk - row_lo
                    cr, an = an_inv, cr_inv
                else:
                    ket_chunk, bra, cr, an = moves
                    ket = ket_chunk + start
                n_moves = int(bra.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_alpha_single_chunk_fast(
                        out,
                        t_group,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_alpha_per_exc=bra[m0:m1],
                        ket_alpha_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=n_rows,
                        M_b=n_cols,
                        chunk_size=chunk,
                        T_full=t_group if local else None,
                    )

        if double_a > 0:
            lo_row = row_lo if local else 0
            hi_row = row_lo + n_rows if local else n_group_rows
            for start in range(lo_row, hi_row, enum_double_a):
                end = min(start + enum_double_a, hi_row)
                moves_d = _memo_moves(
                    cache,
                    ("same_spin", "alpha", g_id, "double", start, end, width_a),
                    _double_moves,
                    alpha_uids[start:end],
                    alpha_uids,
                    unique_alphas,
                    n_orb=n_orb,
                    n_particles=n_a,
                    spin_offset=0,
                )
                if moves_d is None:
                    continue
                if local:
                    bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                    r_dst = start + bra_chunk - row_lo
                    cr0, cr1 = an0_inv, an1_inv
                    an0, an1 = cr0_inv, cr1_inv
                else:
                    ket_chunk, r_dst, cr0, cr1, an0, an1 = moves_d
                    ket = ket_chunk + start
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        t_group,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=True,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=n_rows,
                        M_b=n_cols,
                        chunk_size=chunk,
                        T_full=t_group if local else None,
                        triton_matvec=triton_matvec,
                        bra_sorted=local,
                    )

        if local:
            Wb.index_add_(
                0,
                pair_idx[lo:hi] - first_row,
                out[lo - row_lo * n_cols : hi - row_lo * n_cols],
            )
        else:
            Wb.index_add_(0, pair_idx, out)

    # ---- β channels: one by_alpha group, α string fixed -------------------
    for g_id, g in enumerate(blocks.by_alpha.groups):
        pair_idx = g.pair_idx
        assert pair_idx is not None
        pair_idx = pair_idx.to(device)
        alpha_uids = g.alpha_uids.to(device)
        beta_uids = g.beta_uids.to(device)
        n_group_rows = int(alpha_uids.numel())
        n_cols = int(beta_uids.numel())
        bdets = unique_betas[beta_uids]
        t_group = T[pair_idx]
        if local:
            lo, hi = _group_window(pair_idx, first_row, row_stop)
            if lo >= hi:
                continue
            row_lo = lo // n_cols
            row_hi = (hi + n_cols - 1) // n_cols
            n_rows = row_hi - row_lo
            adets = unique_alphas[alpha_uids[row_lo:row_hi]]
            # A β move leaves α unchanged, so the ket rows are the same slab.
            t_slab = t_group.view(n_group_rows, n_cols)[row_lo:row_hi].reshape(-1)
            # A contiguous position window wraps across α rows, so the owned β
            # columns are one contiguous range only when it stays on one row.
            # Otherwise keep every column: at most the two boundary rows are
            # over-computed, and the final scatter masks those positions.
            if n_rows == 1:
                col_lo = lo % n_cols
                col_hi = (hi - 1) % n_cols + 1
            else:
                col_lo, col_hi = 0, n_cols
            out = Wb.new_zeros(n_rows * n_cols)
        else:
            lo = hi = row_lo = col_lo = 0
            col_hi = n_cols
            n_rows = n_group_rows
            adets = unique_alphas[alpha_uids]
            t_slab = t_group
            out = Wb.new_zeros(int(pair_idx.numel()))

        if single_b > 0:
            if local:
                # A single-row window owns only its column range; a window that
                # wraps rows owns every column of both boundary rows.
                lo_col = col_lo if n_rows == 1 else 0
                hi_col = col_hi if n_rows == 1 else n_cols
            else:
                lo_col, hi_col = 0, n_cols
            for start in range(lo_col, hi_col, enum_single_b):
                end = min(start + enum_single_b, hi_col)
                moves = _memo_moves(
                    cache,
                    ("same_spin", "beta", g_id, "single", start, end, width_b),
                    _single_moves,
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves is None:
                    continue
                if local:
                    bra_chunk, ket, cr_inv, an_inv = moves
                    r_dst = start + bra_chunk
                    cr, an = an_inv, cr_inv
                else:
                    ket_chunk, r_dst, cr, an = moves
                    ket = ket_chunk + start
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_single):
                    m1 = min(m0 + survivor_single, n_moves)
                    _apply_beta_single_chunk_fast(
                        out,
                        t_slab,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        bra_beta_per_exc=r_dst[m0:m1],
                        ket_beta_per_exc=ket[m0:m1],
                        cr=cr[m0:m1],
                        an=an[m0:m1],
                        M_a=n_rows,
                        M_b=n_cols,
                        chunk_size=chunk,
                    )

        if double_b > 0:
            if local:
                lo_col = col_lo if n_rows == 1 else 0
                hi_col = col_hi if n_rows == 1 else n_cols
            else:
                lo_col, hi_col = 0, n_cols
            for start in range(lo_col, hi_col, enum_double_b):
                end = min(start + enum_double_b, hi_col)
                moves_d = _memo_moves(
                    cache,
                    ("same_spin", "beta", g_id, "double", start, end, width_b),
                    _double_moves,
                    beta_uids[start:end],
                    beta_uids,
                    unique_betas,
                    n_orb=n_orb,
                    n_particles=n_b,
                    spin_offset=1,
                )
                if moves_d is None:
                    continue
                if local:
                    bra_chunk, ket, cr0_inv, cr1_inv, an0_inv, an1_inv = moves_d
                    r_dst = start + bra_chunk
                    cr0, cr1 = an0_inv, an1_inv
                    an0, an1 = cr0_inv, cr1_inv
                else:
                    ket_chunk, r_dst, cr0, cr1, an0, an1 = moves_d
                    ket = ket_chunk + start
                n_moves = int(r_dst.numel())
                for m0 in range(0, n_moves, survivor_double):
                    m1 = min(m0 + survivor_double, n_moves)
                    _apply_same_spin_double(
                        out,
                        t_slab,
                        adets,
                        bdets,
                        n_orb,
                        integrals,
                        is_alpha=False,
                        bra=r_dst[m0:m1],
                        ket=ket[m0:m1],
                        cr0=cr0[m0:m1],
                        cr1=cr1[m0:m1],
                        an0=an0[m0:m1],
                        an1=an1[m0:m1],
                        M_a=n_rows,
                        M_b=n_cols,
                        chunk_size=chunk,
                        triton_matvec=triton_matvec,
                        bra_sorted=local,
                    )

        if local:
            Wb.index_add_(
                0,
                pair_idx[lo:hi] - first_row,
                out[lo - row_lo * n_cols : hi - row_lo * n_cols],
            )
        else:
            Wb.index_add_(0, pair_idx, out)
