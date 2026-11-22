"""Tiled owner-computes GDB (explicit pair) matvec over ``α``-uid × ``β``-uid slabs.

Distributed-memory companion of
:class:`~torch_sci_solver.torch_impl.distributed_pairs.LocalPairOperator` for a
2-D tile partition. A rank owns the pairs whose α and β strings fall inside its
two uid slabs (:class:`~torch_sci_solver.torch_impl.pair_tiles.PairTileBounds`),
a *scattered*, sorted subset of the flat pair vector. The operator reproduces
the owner-computes matvec of
:func:`~torch_sci_solver.torch_impl.mult_pairs.mult_pair_local_torch` on that
subset while holding at most two remote tiles, instead of the full length-N
vector the 1-D operator all-gathers.

Every block-decomposed channel is evaluated fragment by fragment, exactly as in
:mod:`~torch_sci_solver.torch_impl.mult_pair_blocks`:

* the α same-spin channel feeds :func:`~torch_sci_solver.torch_impl.
  mult_pair_blocks.add_same_spin_slab` with ``groups=blocks.by_beta`` for every
  owned ``by_beta`` fragment and every source row slab;
* the β same-spin channel feeds it with ``groups=blocks.by_alpha`` for every
  owned ``by_alpha`` fragment and every source column slab;
* the αβ channel calls :func:`~torch_sci_solver.torch_impl.mult_pair_blocks.
  add_ab_slab` for every owned ``by_beta`` destination fragment and every
  ``(source group, source row slab, source column slab)`` rectangle.

Source rectangles are staged by :class:`StagedPairExchange`. Stage 0 is the
local tile; stage ``s`` in ``[1, world_size)`` carries the values of the sorted
global pair indices owned by peer ``(rank + s) % world_size`` that this rank
needs, one peer per stage, double buffered and tagged ``tag + s``. The default
*needed-set* plan stages only the ket-group rectangles this rank's fragments
actually read:

* the α channel reads its fragment's β columns at the α rows one or two
  excitations reach inside its own ``by_beta`` group's source rectangle, and
  only from peers that share the tile's β slab;
* the β channel mirrors this inside the ``by_alpha`` groups and only from
  peers that share the α slab;
* the αβ channel reads the cross product of the α positions reachable by one
  α excitation and the β positions reachable by one β excitation, inside the
  source group rectangle and for every source group.

A channel reads a *contiguous* rectangle, because a single hull per reachable
axis suffices, so each of those reads is a rectangle of one ket group's grid.
The plan unions the rectangles over the owned fragments, converts each to its
global pair indices and ships the sorted union. Same-spin rectangles include
two-excitation reachability as well, and αβ includes one-excitation reach, so
the staged set also covers every ket the tiled 1-RDM, 2-RDM and ``S^2`` sweeps
read from the exchange: those tables are single and double excitations, and
the ``S^2`` spin-swap source is one α single crossed with one β single. The
plan is a pure function of the replicated pair decomposition, the partition's
uid slabs and the destination tile's bounds, so every rank derives the same
stage lists; the exchange still broadcasts the sorted index lists once, and a
sender maps a request to positions in its own tile, both as before.

The whole-tile plan is kept as a fallback: ``needed_staging="off"`` forces it
and ``"auto"`` (the default) keeps it when the needed plan would stage at
least :data:`_NEEDED_STAGING_DENSE_FRACTION` of the peer tiles, because a dense
plan pays the sender's gather and the receiver's searchsorted without saving
traffic. The two plans produce the same arithmetic: a rectangle only narrows
the hull the slab kernels mask to, never the set of moves they evaluate.

Required buffer layouts
-----------------------
The slab kernels read the staged rectangle in the layout their docstrings
specify; the operator passes exactly:

* α (``groups=blocks.by_beta``): ``(n_src_rows, frag.n_cols)`` row-major with
  width ``frag.n_cols``, where ``n_src_rows`` is the number of group rows whose
  α uid lies in the source row slab. Entry ``(r - row_offset) * frag.n_cols + c``
  is group grid position ``(r, frag.col_start + c)``; ``col_offset=0``.
* β (``groups=blocks.by_alpha``): ``(frag.n_rows, n_src_cols)`` row-major with
  width ``n_src_cols = src_col_stop - src_col_offset``. Entry
  ``r * n_src_cols + (c - col_offset)`` is group grid position
  ``(frag.row_start + r, c)``. The slab kernel derives and passes
  ``T_stride = n_src_cols`` itself; the operator only has to hand it the
  matching layout.
* αβ: the source rectangle row-major with width ``src_col_stop -
  src_col_offset``; entry ``(r - src_row_offset) * width + (c - src_col_offset)``
  is ket grid position ``(r, c)``.

Notes
-----
With ``y`` the owned output block and ``x`` the owned input block,

.. math::

    y_k = h_{kk} x_k
      + \\sum_{(k \\leftarrow j) \\in \\mathcal{T}} \\langle D_k | H | D_j
        \\rangle x_j, \\qquad k \\in \\text{tile},

over the five excitation families :math:`\\mathcal{T}` of the pair
Hamiltonian. The pair set is a disjoint union of Cartesian blocks
:math:`\\mathcal{S} = \\sqcup_g U_g \\times B_g`
(:mod:`~torch_sci_solver.torch_impl.pair_blocks`), so a tile is the union of
one rectangle per block, and a source tile is the union of one rectangle per
block for each ordered pair of slabs. Summing a slab kernel over every source
slab of a block reproduces the fragment channel, and scattering the fragment
channels through the tile's ``local_of_global`` map reproduces the
owner-computes matvec on the rank's pair set.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE
from torch_sci_solver.torch_impl.distributed_davidson import (
    GlobalSum,
    TilePartition,
    _check_local_block,
    _dist,
    hash_uniform,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    _double_moves,
    _MoveCache,
    _single_moves,
    _timed,
    add_ab_slab,
    add_same_spin_slab,
    block_profile,
    profile_enabled,
    reset_block_profile,
)
from torch_sci_solver.torch_impl.mult_pairs import DEFAULT_CHUNK_SIZE
from torch_sci_solver.torch_impl.pair_blocks import (
    PairBlocks,
    SpinGroups,
    _Group,
    group_rect_indices,
)
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers, _chunk_pairs
from torch_sci_solver.torch_impl.pair_tiles import (
    PairFragment,
    PairTileBounds,
    pair_tile_fragments,
)
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis

# Share of the move-table memory budget reserved for the same-spin channels.
# The same-spin pass dominates the measured tile-matvec time and a tile rank
# re-enumerates its chunks on every source stage, so it keeps the majority of
# the budget; the αβ channel gets the remainder. Splitting the budget keeps an
# αβ table from crowding the same-spin tables out of the shared cache.
_SAME_SPIN_CACHE_FRACTION = 0.6

# Largest share of the peer tiles the needed-set plan may stage before "auto"
# keeps the whole-tile plan. A dense plan saves no traffic but still pays the
# sender-side gather and the receiver-side searchsorted, so the needed plan
# must remove at least a quarter of the values to be selected.
_NEEDED_STAGING_DENSE_FRACTION = 0.75

# Requested staging modes: "off" is the whole-tile plan, "on" the needed-set
# plan, "auto" the needed-set plan subject to the density threshold above.
_NEEDED_STAGING_MODES = ("off", "auto", "on")

_KIND_SINGLES = 1
_KIND_DOUBLES = 2


def _resolve_identity(
    rank: int | None,
    world_size: int | None,
    group: Any | None,
    bounds: PairTileBounds,
) -> tuple[int, int]:
    """Resolve rank/world size, defaulting to the tile bounds' own identity.

    The supplied ``bounds`` fix the partition, so an explicit ``rank`` /
    ``world_size`` may only confirm it. When a process group is initialized it
    must agree with the tile identity. Without one, a multi-rank identity is
    accepted here so a stand-in exchange can drive the operator in a single
    process; the built-in :class:`StagedPairExchange` then refuses to start.
    """
    dist = _dist()
    initialized = dist.is_available() and dist.is_initialized()
    resolved_rank = bounds.rank if rank is None else int(rank)
    resolved_world = bounds.world_size if world_size is None else int(world_size)
    if (resolved_rank, resolved_world) != (bounds.rank, bounds.world_size):
        raise ValueError(
            f"rank/world_size ({resolved_rank}, {resolved_world}) disagree with the "
            f"tile bounds ({bounds.rank}, {bounds.world_size})"
        )
    if initialized:
        actual_rank = dist.get_rank(group)
        actual_world = dist.get_world_size(group)
        if resolved_world != actual_world:
            raise ValueError(
                f"bounds world_size={resolved_world} disagrees with the process group's "
                f"world_size={actual_world}"
            )
        if resolved_rank != actual_rank:
            raise ValueError(
                f"bounds rank={resolved_rank} disagrees with the process group's rank={actual_rank}"
            )
    return resolved_rank, resolved_world


def _slab_positions(
    uids: Tensor,
    slabs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Group-position ranges whose uid falls inside each contiguous uid slab.

    Args:
        uids: Sorted int64 uid array of one group's axis strings.
        slabs: ``(start, stop)`` uid ranges tiling the axis.

    Returns:
        One ``(start, stop)`` position range per slab; a range is empty when the
        group has no string in that slab.
    """
    cpu = uids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    probes = [int(start) for start, _ in slabs] + [int(slabs[-1][1])]
    position = torch.searchsorted(cpu, torch.tensor(probes, dtype=torch.int64)).tolist()
    return tuple((int(position[index]), int(position[index + 1])) for index in range(len(slabs)))


def _rect_pair_index(
    group: _Group,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> Tensor:
    """Global pair indices of a rectangle of a group's grid, row-major.

    The result is contiguous so ``searchsorted`` does not copy a strided
    single-column view internally. Materialised and structural groups both
    use their own layout.
    """
    return group_rect_indices(group, row_start, row_stop, col_start, col_stop)


def _span_in_slab(positions: Tensor, start: int, stop: int) -> Optional[tuple[int, int]]:
    """Hull of one position set inside a half-open slab range.

    Args:
        positions: Ket positions in a source group's uid array.
        start: First position of the slab (inclusive).
        stop: One past the last position of the slab.

    Returns:
        ``(lo, hi)`` with every position in ``[start, stop)`` inside
        ``[lo, hi)``, or ``None`` when none falls in the slab.
    """
    if stop <= start or int(positions.numel()) == 0:
        return None
    inside = positions[(positions >= int(start)) & (positions < int(stop))]
    if int(inside.numel()) == 0:
        return None
    return int(inside.min()), int(inside.max()) + 1


def _merge_span(
    spans: list[Optional[tuple[int, int]]],
    index: int,
    span: Optional[tuple[int, int]],
) -> None:
    """Widen ``spans[index]`` to cover ``span``."""
    if span is None:
        return
    current = spans[index]
    spans[index] = span if current is None else (min(current[0], span[0]), max(current[1], span[1]))


def _reachable_spans(
    source_uids: Tensor,
    target_uids: Tensor,
    unique: Tensor,
    *,
    n_orb: int,
    n_particles: int,
    spin_offset: int,
    orders: Sequence[int],
    single_chunk: int,
    double_chunk: int,
    slabs: Sequence[tuple[int, int]],
) -> tuple[Optional[tuple[int, int]], ...]:
    """Hull, per source slab, of the ket positions one spin reaches.

    Enumerates the inverse moves from every source string into ``target_uids``
    exactly as the slab kernels do, with the same candidate-grid chunk caps,
    and reduces each slab's survivors to one contiguous hull. A hull is all
    the staged layout can express: the slab kernels index a dense rectangle of
    the ket group's grid, so staging the hull changes neither the evaluated
    moves nor their matrix elements.

    Args:
        source_uids: ``(n_src,)`` bra-side uids into ``unique``.
        target_uids: Sorted uid array of the ket group's axis.
        unique: Sorted unique strings of this spin (1-D or limb matrices).
        n_orb: Number of spatial orbitals.
        n_particles: Electrons of this spin in every string.
        spin_offset: ``0`` for α, ``1`` for β.
        orders: Excitation orders to include; same-spin channels use
            ``(1, 2)``, the αβ channel ``(1,)``.
        single_chunk: Source strings per single-excitation enumeration chunk.
        double_chunk: Source strings per double-excitation enumeration chunk.
        slabs: ``(start, stop)`` position ranges of the source slabs, as
            returned by :func:`_slab_positions`.

    Returns:
        One hull per slab (``None`` where the slab holds no reachable ket).
    """
    spans: list[Optional[tuple[int, int]]] = [None] * len(slabs)
    n_src = int(source_uids.numel())
    if n_src == 0 or int(target_uids.numel()) == 0:
        return tuple(spans)
    for order in orders:
        chunk = single_chunk if order == _KIND_SINGLES else double_chunk
        moves_fn = _single_moves if order == _KIND_SINGLES else _double_moves
        step = max(1, int(chunk))
        for start in range(0, n_src, step):
            moves = moves_fn(
                source_uids[start : start + step],
                target_uids,
                unique,
                n_orb=n_orb,
                n_particles=n_particles,
                spin_offset=spin_offset,
            )
            if moves is None:
                continue
            for index, (lo, hi) in enumerate(slabs):
                _merge_span(spans, index, _span_in_slab(moves[1], lo, hi))
    return tuple(spans)


@dataclass(frozen=True)
class _ByBetaRect:
    """One ``by_beta`` group rectangle staged from a source tile.

    The rectangle is ``[row_start, row_stop) × [col_start, col_stop)`` of the
    group's ``|U_g| × |B_g|`` grid, in group coordinates.

    Attributes:
        group: Index of the group in ``blocks.by_beta.groups``.
        row_start: First staged row, in group coordinates.
        row_stop: One past the last staged row.
        col_start: First staged column, in group coordinates.
        col_stop: One past the last staged column.
        alpha_dests: Indices into the owned ``by_beta`` fragments for the α
            same-spin channel; empty when this stage does not carry the
            fragments' β columns.
        ab_dests: Indices into the owned ``by_beta`` fragments for the αβ
            channel; every owned fragment consumes this rectangle.
    """

    group: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    alpha_dests: tuple[int, ...]
    ab_dests: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        """Number of staged rows."""
        return self.row_stop - self.row_start

    @property
    def n_cols(self) -> int:
        """Number of staged columns."""
        return self.col_stop - self.col_start


@dataclass(frozen=True)
class _ByAlphaRect:
    """One ``by_alpha`` group rectangle staged from a source tile.

    The rectangle is ``[row_start, row_stop) × [col_start, col_stop)`` of the
    group's ``|U_h| × |B_h|`` grid, in group coordinates.

    Attributes:
        group: Index of the group in ``blocks.by_alpha.groups``.
        row_start: First staged row, in group coordinates.
        row_stop: One past the last staged row.
        col_start: First staged column, in group coordinates.
        col_stop: One past the last staged column.
        beta_dests: Indices into the owned ``by_alpha`` fragments for the β
            same-spin channel; empty when this stage is not the fragments'
            α-row slab.
    """

    group: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    beta_dests: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        """Number of staged rows."""
        return self.row_stop - self.row_start

    @property
    def n_cols(self) -> int:
        """Number of staged columns."""
        return self.col_stop - self.col_start


@dataclass(frozen=True)
class _TileGeometry:
    """Replicated fragments and per-slab group positions of one destination tile.

    Everything here is derived from the pair decomposition, the partition's uid
    slabs and the destination tile's bounds, so any rank can build the geometry
    of any tile without communication.

    Attributes:
        beta_groups: The ``by_beta`` blocks (α and αβ channels).
        alpha_groups: The ``by_alpha`` blocks (β channel).
        alpha_frags: Owned ``by_beta`` fragments, in ``pair_tile_fragments``
            order; the α and αβ channels index them as destinations.
        beta_frags: Owned ``by_alpha`` fragments; the β channel indexes them.
        alpha_frags_of_group: ``alpha_frags`` indices per ``by_beta`` group.
        beta_frags_of_group: ``beta_frags`` indices per ``by_alpha`` group.
        beta_rows: Per ``by_beta`` group, the group-position range of every
            α-uid slab.
        beta_cols: Per ``by_beta`` group, the group-position range of every
            β-uid slab.
        alpha_rows: Mirror of :attr:`beta_rows` for ``by_alpha`` groups.
        alpha_cols: Mirror of :attr:`beta_cols` for ``by_alpha`` groups.
    """

    beta_groups: tuple[_Group, ...]
    alpha_groups: tuple[_Group, ...]
    alpha_frags: tuple[PairFragment, ...]
    beta_frags: tuple[PairFragment, ...]
    alpha_frags_of_group: tuple[tuple[int, ...], ...]
    beta_frags_of_group: tuple[tuple[int, ...], ...]
    beta_rows: tuple[tuple[tuple[int, int], ...], ...]
    beta_cols: tuple[tuple[tuple[int, int], ...], ...]
    alpha_rows: tuple[tuple[tuple[int, int], ...], ...]
    alpha_cols: tuple[tuple[tuple[int, int], ...], ...]


def _tile_geometry(
    blocks: PairBlocks,
    bounds: PairTileBounds,
    partition: TilePartition,
) -> _TileGeometry:
    """Derive one destination tile's fragments and per-slab group positions.

    Args:
        blocks: The exact block decomposition of the global pair basis.
        bounds: The destination tile's bounds.
        partition: The 2-D partition the bounds were built from.

    Returns:
        The tile geometry shared by the whole-tile and needed-set plans.
    """
    alpha_frags = tuple(pair_tile_fragments(blocks.by_beta, bounds))
    beta_frags = tuple(pair_tile_fragments(blocks.by_alpha, bounds))
    alpha_frags_of_group: list[list[int]] = [[] for _ in blocks.by_beta.groups]
    for index, frag in enumerate(alpha_frags):
        alpha_frags_of_group[frag.group].append(index)
    beta_frags_of_group: list[list[int]] = [[] for _ in blocks.by_alpha.groups]
    for index, frag in enumerate(beta_frags):
        beta_frags_of_group[frag.group].append(index)
    grid = partition.grid
    return _TileGeometry(
        beta_groups=blocks.by_beta.groups,
        alpha_groups=blocks.by_alpha.groups,
        alpha_frags=alpha_frags,
        beta_frags=beta_frags,
        alpha_frags_of_group=tuple(tuple(item) for item in alpha_frags_of_group),
        beta_frags_of_group=tuple(tuple(item) for item in beta_frags_of_group),
        beta_rows=tuple(
            _slab_positions(g.alpha_uids, grid.alpha_bounds) for g in blocks.by_beta.groups
        ),
        beta_cols=tuple(
            _slab_positions(g.beta_uids, grid.beta_bounds) for g in blocks.by_beta.groups
        ),
        alpha_rows=tuple(
            _slab_positions(g.alpha_uids, grid.alpha_bounds) for g in blocks.by_alpha.groups
        ),
        alpha_cols=tuple(
            _slab_positions(g.beta_uids, grid.beta_bounds) for g in blocks.by_alpha.groups
        ),
    )


def _whole_stage_rects(
    geometry: _TileGeometry,
    row_slab: int,
    col_slab: int,
    *,
    same_row_slab: bool,
    same_col_slab: bool,
    ab_dests: tuple[int, ...],
) -> tuple[tuple[_ByBetaRect, ...], tuple[_ByAlphaRect, ...]]:
    """Every group rectangle of one source slab pair, as the whole-tile plan stages it.

    Args:
        geometry: The destination tile's geometry.
        row_slab: Source α-uid slab index.
        col_slab: Source β-uid slab index.
        same_row_slab: Whether the source shares the destination's α slab
            (the β same-spin channel then reads its ``by_alpha`` rectangles).
        same_col_slab: Whether the source shares the destination's β slab
            (the α same-spin channel then reads its ``by_beta`` rectangles).
        ab_dests: Every owned ``by_beta`` fragment; the αβ channel reads every
            source group for each of them.

    Returns:
        ``(beta_rects, alpha_rects)`` for that stage.
    """
    beta_rects: list[_ByBetaRect] = []
    for group_id, group in enumerate(geometry.beta_groups):
        row_start, row_stop = geometry.beta_rows[group_id][row_slab]
        col_start, col_stop = geometry.beta_cols[group_id][col_slab]
        if row_stop <= row_start or col_stop <= col_start:
            continue
        alpha_dests = geometry.alpha_frags_of_group[group_id] if same_col_slab else ()
        beta_rects.append(
            _ByBetaRect(
                group=group_id,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                alpha_dests=alpha_dests,
                ab_dests=ab_dests,
            )
        )
    alpha_rects: list[_ByAlphaRect] = []
    if same_row_slab:
        for group_id, group in enumerate(geometry.alpha_groups):
            row_start, row_stop = geometry.alpha_rows[group_id][row_slab]
            col_start, col_stop = geometry.alpha_cols[group_id][col_slab]
            if row_stop <= row_start or col_stop <= col_start:
                continue
            alpha_rects.append(
                _ByAlphaRect(
                    group=group_id,
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                    beta_dests=geometry.beta_frags_of_group[group_id],
                )
            )
    return tuple(beta_rects), tuple(alpha_rects)


@dataclass(frozen=True)
class _NeededPlan:
    """Needed-set staging plan of one destination tile.

    Attributes:
        beta_rects: Per-stage tuples of ``by_beta`` rectangles; stage 0 is the
            whole local tile, later stages are the fragments' read rectangles.
        alpha_rects: Per-stage tuples of ``by_alpha`` rectangles.
        needed: Per-stage sorted global pair indices the exchange delivers.
        staged_values: Values the needed plan ships over all remote stages.
    """

    beta_rects: tuple[tuple[_ByBetaRect, ...], ...]
    alpha_rects: tuple[tuple[_ByAlphaRect, ...], ...]
    needed: tuple[Tensor, ...]
    staged_values: int


def _needed_stage_plan(
    pair_helpers: PairHelpers,
    bounds: PairTileBounds,
    partition: TilePartition,
    geometry: _TileGeometry,
) -> _NeededPlan:
    """Per-stage ket-group rectangles from each owned fragment's actual reads.

    For every stage (destination rank, source peer) the plan unions the
    rectangles the destination's fragments read and converts them to global
    pair indices. The rectangles are:

    * α same-spin: the hull of the α positions one or two excitations reach
      from each owned ``by_beta`` fragment's rows, inside the source α slab,
      at the fragment's β columns, for peers sharing the β slab;
    * β same-spin: the mirror inside ``by_alpha`` groups, for peers sharing
      the α slab;
    * αβ: for every owned ``by_beta`` fragment and every ``by_beta`` source
      group, the hull of the α positions one α excitation reaches crossed with
      the hull of the β positions one β excitation reaches, both inside the
      source slabs.

    A rectangle's hull is what the slab kernels mask to, so the same moves are
    evaluated on a smaller staged buffer. The two-excitation same-spin hulls
    and the one-excitation αβ hulls also cover the tiled RDM and ``S^2``
    sweeps, whose ket rows are single or double excitations and whose spin
    swap is one α single crossed with one β single.

    Args:
        pair_helpers: Helpers for the global pair basis; ``blocks`` must be
            the exact decomposition.
        bounds: The destination tile's bounds.
        partition: The 2-D partition the bounds were built from; its grid
            supplies every peer's uid slabs.
        geometry: The destination tile's geometry.

    Returns:
        The per-stage rectangles, index lists and value counts. Stage ``s``
        lists what peer ``(bounds.rank + s) % world_size`` ships.

    Raises:
        ValueError: the helpers carry no block decomposition.
    """
    blocks = pair_helpers.blocks
    if blocks is None:
        raise ValueError("the needed-set staging plan requires pair_helpers.blocks")
    grid = partition.grid
    rank, world_size = bounds.rank, bounds.world_size
    n_orb = int(pair_helpers.n_orb)
    n_a = int(pair_helpers.n_a)
    n_b = int(pair_helpers.n_b)
    n_empty_a = n_orb - n_a
    n_empty_b = n_orb - n_b
    device = pair_helpers.device
    unique_alphas = pair_helpers.unique_alphas.to(device)
    unique_betas = pair_helpers.unique_betas.to(device)
    # The chunk caps mirror the slab kernels' enumerations, so the setup pass
    # holds no larger candidate grid than one matvec chunk.
    budget = max(1, int(pair_helpers.budget_bytes))
    ab_budget = max(1, budget // 4)
    ss_budget = max(1, budget // 2)
    single_a = n_a * n_empty_a
    single_b = n_b * n_empty_b
    double_a = (n_a * (n_a - 1) // 2) * (n_empty_a * (n_empty_a - 1) // 2)
    double_b = (n_b * (n_b - 1) // 2) * (n_empty_b * (n_empty_b - 1) // 2)
    ab_single_a = _chunk_pairs(max(1, single_a), "singles", ab_budget)
    ab_single_b = _chunk_pairs(max(1, single_b), "singles", ab_budget)
    ss_single_a = _chunk_pairs(max(1, single_a), "singles", ss_budget)
    ss_single_b = _chunk_pairs(max(1, single_b), "singles", ss_budget)
    ss_double_a = _chunk_pairs(max(1, double_a), "doubles", ss_budget)
    ss_double_b = _chunk_pairs(max(1, double_b), "doubles", ss_budget)

    beta_rects: list[list[_ByBetaRect]] = [[] for _ in range(world_size)]
    alpha_rects: list[list[_ByAlphaRect]] = [[] for _ in range(world_size)]
    beta_groups = geometry.beta_groups
    for dest, frag in enumerate(geometry.alpha_frags):
        bra_group = beta_groups[frag.group]
        bra_alpha = bra_group.alpha_uids[frag.row_start : frag.row_stop].to(device)
        bra_beta = bra_group.beta_uids[frag.col_start : frag.col_stop].to(device)
        # α same-spin: singles and doubles into the fragment's own group.
        own_rows = _reachable_spans(
            bra_alpha,
            bra_group.alpha_uids.to(device),
            unique_alphas,
            n_orb=int(n_orb),
            n_particles=n_a,
            spin_offset=0,
            orders=(_KIND_SINGLES, _KIND_DOUBLES),
            single_chunk=ss_single_a,
            double_chunk=ss_double_a,
            slabs=geometry.beta_rows[frag.group],
        )
        # αβ: one α and one β single into every source group.
        ab_rows = tuple(
            _reachable_spans(
                bra_alpha,
                group.alpha_uids.to(device),
                unique_alphas,
                n_orb=int(n_orb),
                n_particles=n_a,
                spin_offset=0,
                orders=(_KIND_SINGLES,),
                single_chunk=ab_single_a,
                double_chunk=ab_single_a,
                slabs=geometry.beta_rows[group_id],
            )
            for group_id, group in enumerate(beta_groups)
        )
        ab_cols = tuple(
            _reachable_spans(
                bra_beta,
                group.beta_uids.to(device),
                unique_betas,
                n_orb=int(n_orb),
                n_particles=n_b,
                spin_offset=1,
                orders=(_KIND_SINGLES,),
                single_chunk=ab_single_b,
                double_chunk=ab_single_b,
                slabs=geometry.beta_cols[group_id],
            )
            for group_id, group in enumerate(beta_groups)
        )
        for stage in range(1, world_size):
            peer = (rank + stage) % world_size
            row_slab, col_slab = grid.row_slab(peer), grid.col_slab(peer)
            if col_slab == bounds.col_slab:
                span = own_rows[row_slab]
                if span is not None:
                    beta_rects[stage].append(
                        _ByBetaRect(
                            group=frag.group,
                            row_start=span[0],
                            row_stop=span[1],
                            col_start=frag.col_start,
                            col_stop=frag.col_stop,
                            alpha_dests=(dest,),
                            ab_dests=(),
                        )
                    )
            for group_id in range(len(beta_groups)):
                row_span = ab_rows[group_id][row_slab]
                col_span = ab_cols[group_id][col_slab]
                if row_span is None or col_span is None:
                    continue
                beta_rects[stage].append(
                    _ByBetaRect(
                        group=group_id,
                        row_start=row_span[0],
                        row_stop=row_span[1],
                        col_start=col_span[0],
                        col_stop=col_span[1],
                        alpha_dests=(),
                        ab_dests=(dest,),
                    )
                )

    alpha_groups = geometry.alpha_groups
    for dest, frag in enumerate(geometry.beta_frags):
        bra_group = alpha_groups[frag.group]
        bra_beta = bra_group.beta_uids[frag.col_start : frag.col_stop].to(device)
        own_cols = _reachable_spans(
            bra_beta,
            bra_group.beta_uids.to(device),
            unique_betas,
            n_orb=int(n_orb),
            n_particles=n_b,
            spin_offset=1,
            orders=(_KIND_SINGLES, _KIND_DOUBLES),
            single_chunk=ss_single_b,
            double_chunk=ss_double_b,
            slabs=geometry.alpha_cols[frag.group],
        )
        for stage in range(1, world_size):
            peer = (rank + stage) % world_size
            row_slab, col_slab = grid.row_slab(peer), grid.col_slab(peer)
            if row_slab != bounds.row_slab:
                continue
            span = own_cols[col_slab]
            if span is None:
                continue
            alpha_rects[stage].append(
                _ByAlphaRect(
                    group=frag.group,
                    row_start=frag.row_start,
                    row_stop=frag.row_stop,
                    col_start=span[0],
                    col_stop=span[1],
                    beta_dests=(dest,),
                )
            )

    ab_dests = tuple(range(len(geometry.alpha_frags)))
    needed: list[Tensor] = []
    for stage in range(world_size):
        if stage == 0:
            stage_beta, stage_alpha = _whole_stage_rects(
                geometry,
                bounds.row_slab,
                bounds.col_slab,
                same_row_slab=True,
                same_col_slab=True,
                ab_dests=ab_dests,
            )
            beta_rects[stage] = list(stage_beta)
            alpha_rects[stage] = list(stage_alpha)
            needed.append(bounds.pair_index.detach().to(device="cpu", dtype=torch.int64))
            continue
        parts = [
            _rect_pair_index(
                beta_groups[rect.group],
                rect.row_start,
                rect.row_stop,
                rect.col_start,
                rect.col_stop,
            )
            .detach()
            .cpu()
            for rect in beta_rects[stage]
        ] + [
            _rect_pair_index(
                alpha_groups[rect.group],
                rect.row_start,
                rect.row_stop,
                rect.col_start,
                rect.col_stop,
            )
            .detach()
            .cpu()
            for rect in alpha_rects[stage]
        ]
        needed.append(
            torch.unique(torch.cat(parts)) if parts else torch.empty(0, dtype=torch.int64)
        )
    return _NeededPlan(
        beta_rects=tuple(tuple(item) for item in beta_rects),
        alpha_rects=tuple(tuple(item) for item in alpha_rects),
        needed=tuple(needed),
        staged_values=sum(int(item.numel()) for item in needed[1:]),
    )


class StagedPairExchange:
    """Cyclic staged P2P exchange of the scattered pair values a tile needs.

    Stage 0 is the caller's own tile. Stage ``s`` in ``[1, world_size)`` holds
    the values of the global pair indices in ``needed[s]`` — a sorted list owned
    by peer ``(rank + s) % world_size``. The constructor broadcasts every rank's
    per-stage ``needed`` lists once with ``all_gather_object``; each stage then
    sends the values peer ``(rank - s) % world_size`` requested and receives its
    own list, one batched ``isend``/``irecv`` pair per stage. Stage ``s + 1`` is
    posted before stage ``s`` is yielded (double buffering), so communication
    overlaps the caller's kernels, and stage ``s`` uses tag ``tag + s``.

    Complex payloads are packed as their interleaved ``(real, imag)`` view
    (NCCL has no complex scalar type) and received into a real buffer.

    Args:
        local_index: ``(N_local,)`` sorted global pair indices owned by this
            rank, used to map a peer's request to positions in the local tile.
        needed: ``world_size`` sorted global pair index lists; entry ``s`` is
            what stage ``s`` carries (entry 0 is unused for communication).
        rank: This rank.
        world_size: Number of ranks.
        group: Process group used by every collective and P2P op.
        tag: Base P2P tag; stage ``s`` uses ``tag + s``.

    Attributes:
        needed: The per-stage index lists, entry ``s`` being stage ``s``'s order.
        stage_sizes: Per remote stage receive counts, ``stage_sizes[s - 1]``.
        max_stage_size: Largest per-stage receive count, in values; the two
            receive buffers are this long.
        max_send_size: Largest send payload over the stages, in values; a send
            that covers the whole local tile aliases ``x_local`` and still
            counts its length here.

    Raises:
        ValueError: a needed list is not strictly increasing, or its count does
            not match ``world_size``.
        RuntimeError: a multi-rank exchange has no initialized process group, or
            a peer requests a pair index this tile does not own.
    """

    def __init__(
        self,
        local_index: Tensor,
        needed: Sequence[Tensor],
        *,
        rank: int,
        world_size: int,
        group: Optional[Any] = None,
        tag: int = 0,
    ) -> None:
        """Compute the per-stage send positions and receive lengths."""
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0:
            raise ValueError(f"world_size must be > 0, got {self.world_size}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")
        self.group = group
        self._tag = int(tag)
        self.local_index = (
            torch.as_tensor(local_index, dtype=torch.int64).reshape(-1).detach().cpu()
        )
        lists = [
            torch.as_tensor(item, dtype=torch.int64).reshape(-1).detach().cpu() for item in needed
        ]
        if len(lists) != self.world_size:
            raise ValueError(f"needed has {len(lists)} stage lists, expected {self.world_size}")
        for stage, item in enumerate(lists):
            if item.numel() > 1 and not bool(torch.all(item[1:] > item[:-1])):
                raise ValueError(f"needed[{stage}] must be strictly increasing")
        self.needed = lists

        if self.world_size > 1:
            dist = _dist()
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "a multi-rank StagedPairExchange needs an initialized process group "
                    "(torch.distributed.init_process_group)"
                )
            # Only stages >= 1 are exchanged; the stage-0 list is this rank's
            # own tile and is never read from the gather. The index lists ship
            # as their int64 byte images: pickling one bytes object scales with
            # the tile size, while pickling a Python int list would build one
            # object per pair on every rank.
            all_needed: list[Optional[list[bytes]]] = [None] * self.world_size
            dist.all_gather_object(
                all_needed, [item.numpy().tobytes() for item in lists[1:]], group=group
            )
        else:
            all_needed = [[item.numpy().tobytes() for item in lists[1:]]]

        self._send_index: dict[int, Optional[Tensor]] = {}
        for stage in range(1, self.world_size):
            sender = (self.rank - stage) % self.world_size
            sender_lists = all_needed[sender]
            if sender_lists is None:
                raise RuntimeError("failed to gather every rank's needed pair lists")
            payload = sender_lists[stage - 1]
            requested = (
                torch.frombuffer(bytearray(payload), dtype=torch.int64)
                if payload
                else torch.empty(0, dtype=torch.int64)
            )
            self._send_index[stage] = self._positions(requested)
        self._max_recv = max(
            (int(item.numel()) for item in lists[1:]),
            default=0,
        )
        self.max_stage_size = self._max_recv
        self.max_send_size = max(
            (
                int(self.local_index.numel()) if positions is None else int(positions.numel())
                for positions in self._send_index.values()
            ),
            default=0,
        )
        self.stage_sizes = tuple(int(item.numel()) for item in lists[1:])
        self._buffers: Optional[list[Tensor]] = None
        self._buffer_key: Optional[tuple[torch.dtype, torch.device]] = None

    def _positions(self, requested: Tensor) -> Optional[Tensor]:
        """Positions in this tile of the requested global indices.

        ``None`` means the request is exactly this tile's sorted index list, so
        the payload is the local vector itself.

        Raises:
            RuntimeError: the request contains an index this tile does not own.
        """
        if requested.numel() == self.local_index.numel() and torch.equal(
            requested, self.local_index
        ):
            return None
        if requested.numel() == 0:
            return requested
        if self.local_index.numel() == 0:
            raise RuntimeError("a peer requested pair indices from an empty tile")
        position = torch.searchsorted(self.local_index, requested)
        clipped = position.clamp(max=self.local_index.numel() - 1)
        found = (position < self.local_index.numel()) & (self.local_index[clipped] == requested)
        if not bool(found.all()):
            raise RuntimeError("a peer requested pair indices this tile does not own")
        return position

    def _ensure_buffers(self, reference: Tensor) -> None:
        """Allocate the two flat receive buffers for this dtype/device."""
        key = (reference.dtype, reference.device)
        if self._buffers is None or self._buffer_key != key:
            width = 2 if reference.dtype.is_complex else 1
            dtype = reference.dtype.to_real() if reference.dtype.is_complex else reference.dtype
            self._buffers = [
                torch.empty(self._max_recv * width, dtype=dtype, device=reference.device)
                for _ in range(2)
            ]
            self._buffer_key = key

    def _post_stage(
        self, stage: int, x_local: Tensor
    ) -> tuple[list[Any], dict[int, Tensor], Optional[Tensor]]:
        """Issue stage ``stage``'s send and receive; keep payloads alive."""
        dist = _dist()
        ops: list[Any] = []
        sends: dict[int, Tensor] = {}
        send_index = self._send_index[stage]
        payload = x_local if send_index is None else x_local[send_index.to(x_local.device)]
        if payload.numel():
            packed = payload.contiguous()
            if packed.is_complex():
                packed = torch.view_as_real(packed).reshape(-1)
            else:
                packed = packed.reshape(-1)
            sends[stage] = packed
            ops.append(
                dist.P2POp(
                    dist.isend,
                    packed,
                    (self.rank - stage) % self.world_size,
                    group=self.group,
                    tag=self._tag + stage,
                )
            )
        n_recv = int(self.needed[stage].numel())
        recv: Optional[Tensor] = None
        if n_recv:
            width = 2 if x_local.dtype.is_complex else 1
            recv = self._buffers[stage % 2][: n_recv * width]  # type: ignore[index]
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv,
                    (self.rank + stage) % self.world_size,
                    group=self.group,
                    tag=self._tag + stage,
                )
            )
        requests = dist.batch_isend_irecv(ops) if ops else []
        return requests, sends, recv

    @staticmethod
    def _decode(recv: Optional[Tensor], reference: Tensor) -> Tensor:
        """Expose a received flat block as a 1-D value buffer."""
        if recv is None:
            return reference.new_empty(0)
        if reference.dtype.is_complex:
            return torch.view_as_complex(recv.view(-1, 2))
        return recv

    def stages(self, x_local: Tensor) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(stage, buffer)`` for stage ``0 .. world_size - 1``.

        Stage 0 yields ``x_local`` itself; stage ``s > 0`` yields a 1-D buffer
        of the values of ``needed[s]`` in that order.

        Args:
            x_local: This rank's tile, ``(N_local,)`` in sorted pair order.

        Raises:
            ValueError: ``x_local`` does not have one entry per owned pair.
        """
        if x_local.shape != (self.local_index.numel(),):
            raise ValueError(
                f"x_local shape {tuple(x_local.shape)} != ({self.local_index.numel()},)"
            )
        if self.world_size == 1:
            yield 0, x_local
            return
        self._ensure_buffers(x_local)
        pending: dict[int, tuple[list[Any], dict[int, Tensor], Optional[Tensor]]] = {}
        pending[1] = self._post_stage(1, x_local)
        try:
            yield 0, x_local
            for stage in range(1, self.world_size):
                requests, _sends, recv = pending.pop(stage)
                for request in requests:
                    request.wait()
                # Post the next stage before yielding this one so it overlaps
                # the caller's kernels; its parity differs from this stage's.
                if stage + 1 < self.world_size:
                    pending[stage + 1] = self._post_stage(stage + 1, x_local)
                yield stage, self._decode(recv, x_local)
        finally:
            # Early close still owns posted requests; draining them keeps the
            # process group consistent for the caller's next collective.
            for requests, _sends, _recv in pending.values():
                for request in requests:
                    request.wait()


class LocalPairTileOperator:
    """Owner-computes pair-mode matvec over this rank's α-uid × β-uid tile.

    ``__call__`` is collective: every rank calls it once per matvec, in the same
    order, because the source tiles are exchanged with the staged P2P protocol
    of :class:`StagedPairExchange`.

    Args:
        hii_local: ``(N_local,)`` diagonal of the owned pair set, in sorted pair
            order (the caller slices the global diagonal by
            ``bounds.pair_index``).
        pair_basis: the global ``(α, β)`` pair list.
        pair_helpers: helpers for ``pair_basis``; its ``blocks`` decomposition
            must be enabled (``block_matvec != "off"``).
        n_orb: number of spatial orbitals.
        integrals: the matvec :class:`TorchIntegrals` bundle.
        bounds: this rank's tile: its α/β uid slabs and sorted global pair
            indices. It fixes the partition; ``rank`` / ``world_size`` may only
            confirm it.
        partition: the 2-D :class:`TilePartition` the bounds were built from;
            it supplies the other ranks' uid slabs.
        rank, world_size, group: process-group identity and communicator.
        chunk_size: per-kernel workspace bound.
        tag: base P2P tag of the staged exchange.
        move_cache_bytes: total byte budget of the move-table memos; must be
            positive when given. ``None`` falls back to
            ``pair_helpers.string_cache_bytes``, whose ``0`` disables both
            memos. The same-spin channel keeps 60% of the budget and the αβ
            channel the remainder, so the two cannot evict each other.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel; explicit opt-in, real CUDA tensors only.
        needed_staging: staging plan of the remote source stages. ``"off"``
            stages each peer's whole tile, ``"on"`` stages only the ket-group
            rectangles this rank's fragments read, and ``"auto"`` (the default)
            uses the needed plan unless it would stage at least
            ``_NEEDED_STAGING_DENSE_FRACTION`` of the peer tiles. A single-rank
            or empty-tile operator builds no plan and reports ``"tile"``; both
            plans stage nothing for it.

    Attributes:
        n: global pair count N.
        N_local: number of pairs owned by this rank.
        bounds: the :class:`PairTileBounds` this operator serves.
        partition: the :class:`TilePartition` over the uid axes.
        staging_mode: resolved staging plan, ``"tile"`` or ``"needed"``.
        needed_size: values the exchange receives over all remote stages; the
            traffic of the resolved plan.
        stage_sizes: per remote stage receive counts.
        whole_tile_size: values the whole-tile plan would receive over all
            remote stages, for diagnostics.

    Notes
    -----
    Building the needed-set plan runs one inverse-move enumeration per owned
    fragment and source group, with the slab kernels' candidate-grid caps, so
    its setup cost is about one matvec's enumeration pass and is paid once per
    operator, never per matvec. The whole-tile fallback builds no plan.
    """

    def __init__(
        self,
        hii_local: Tensor,
        pair_basis: "PairBasis | StructuralPairBasis",
        pair_helpers: PairHelpers,
        n_orb: int,
        integrals: TorchIntegrals,
        bounds: PairTileBounds,
        partition: TilePartition,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        group: Any | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        tag: int = 0,
        move_cache_bytes: int | None = None,
        triton_matvec: bool = False,
        needed_staging: str = "auto",
    ) -> None:
        """Validate the tile and build the stage plan and the exchange."""
        staging = str(needed_staging).lower()
        if staging not in _NEEDED_STAGING_MODES:
            raise ValueError(
                f"needed_staging must be one of {_NEEDED_STAGING_MODES}, got {needed_staging!r}"
            )
        self.n = pair_basis.n_pairs
        if bounds.n_pairs != self.n:
            raise ValueError(f"bounds.n_pairs={bounds.n_pairs} != pair count {self.n}")
        if hii_local.shape != (bounds.local_size,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({bounds.local_size},)")
        blocks = pair_helpers.blocks
        if blocks is None:
            raise ValueError(
                "LocalPairTileOperator requires pair_helpers.blocks (block_matvec != 'off')"
            )
        if blocks.n_pairs != self.n:
            raise ValueError(f"blocks.n_pairs={blocks.n_pairs} != pair count {self.n}")
        if partition.world_size != bounds.world_size:
            raise ValueError(
                f"partition world_size={partition.world_size} != bounds world_size="
                f"{bounds.world_size}"
            )
        if partition.n_rows != bounds.n_alpha_uids or partition.n_cols != bounds.n_beta_uids:
            raise ValueError(
                f"partition ({partition.n_rows}, {partition.n_cols}) does not match the "
                f"tile uid axes ({bounds.n_alpha_uids}, {bounds.n_beta_uids})"
            )
        if (
            partition.n_rows != pair_basis.n_unique_alphas
            or partition.n_cols != pair_basis.n_unique_betas
        ):
            raise ValueError(
                f"partition ({partition.n_rows}, {partition.n_cols}) does not match the "
                f"basis uid axes ({pair_basis.n_unique_alphas}, {pair_basis.n_unique_betas})"
            )
        bound = partition.for_rank(bounds.rank)
        if (bound.row_slab, bound.col_slab, bound.row_start, bound.row_stop) != (
            bounds.row_slab,
            bounds.col_slab,
            bounds.alpha_start,
            bounds.alpha_stop,
        ) or (bound.col_start, bound.col_stop) != (bounds.beta_start, bounds.beta_stop):
            raise ValueError("bounds do not describe this rank's tile of the partition")

        self._rank, self._world_size = _resolve_identity(rank, world_size, group, bounds)
        self.bounds = bounds
        self.partition = partition
        self.N_local = bounds.local_size
        self._pair_helpers = pair_helpers
        self._blocks = blocks
        self._n_orb = int(n_orb)
        self._integrals = integrals
        self._hii_local = hii_local
        self._chunk_size = int(chunk_size)
        self._triton_matvec = bool(triton_matvec)
        if move_cache_bytes is None:
            cache_bytes = int(pair_helpers.string_cache_bytes)
        else:
            if int(move_cache_bytes) <= 0:
                raise ValueError(f"move_cache_bytes must be > 0 or None, got {move_cache_bytes}")
            cache_bytes = int(move_cache_bytes)
        # Two independent budgets: the same-spin channel is the dominant one, so
        # it keeps the larger share and αβ cannot crowd it out. A disabled
        # fallback budget creates no cache; any positive total still gives
        # same-spin a slot.
        same_spin_bytes = (
            max(1, int(cache_bytes * _SAME_SPIN_CACHE_FRACTION)) if cache_bytes > 0 else 0
        )
        ab_bytes = cache_bytes - same_spin_bytes
        self._ss_move_cache = _MoveCache(same_spin_bytes) if same_spin_bytes > 0 else None
        self._ab_move_cache = _MoveCache(ab_bytes) if ab_bytes > 0 else None

        grid = partition.grid
        geometry = _tile_geometry(blocks, bounds, partition)
        self._beta_groups = geometry.beta_groups
        self._alpha_groups = geometry.alpha_groups
        self._alpha_frags = geometry.alpha_frags
        self._beta_frags = geometry.beta_frags

        # One single-block by_beta grouping per source group, as add_ab_slab
        # consumes exactly one ket group per call.
        self._single_groups = tuple(
            SpinGroups(
                axis="beta",
                group_of_string=blocks.by_beta.group_of_string,
                groups=(group,),
                total_pairs=blocks.by_beta.total_pairs,
            )
            for group in self._beta_groups
        )

        # The whole-tile rectangles are cheap to build and are always needed
        # for stage 0; they are also the fallback and the traffic baseline.
        ab_dests = tuple(range(len(self._alpha_frags)))
        whole_beta: list[tuple[_ByBetaRect, ...]] = []
        whole_alpha: list[tuple[_ByAlphaRect, ...]] = []
        whole_sizes: list[int] = []
        for stage in range(self._world_size):
            if stage == 0:
                row_slab, col_slab = bounds.row_slab, bounds.col_slab
            else:
                peer = (self._rank + stage) % self._world_size
                row_slab, col_slab = grid.row_slab(peer), grid.col_slab(peer)
            beta_rects, alpha_rects = _whole_stage_rects(
                geometry,
                row_slab,
                col_slab,
                same_row_slab=row_slab == bounds.row_slab,
                same_col_slab=col_slab == bounds.col_slab,
                ab_dests=ab_dests,
            )
            whole_beta.append(beta_rects)
            whole_alpha.append(alpha_rects)
            whole_sizes.append(sum(rect.n_rows * rect.n_cols for rect in beta_rects))
        whole_total = sum(whole_sizes[1:])

        plan: Optional[_NeededPlan] = None
        if staging != "off" and self._world_size > 1 and bounds.local_size > 0:
            plan = _needed_stage_plan(pair_helpers, bounds, partition, geometry)
        if plan is not None and (
            staging == "on" or plan.staged_values <= _NEEDED_STAGING_DENSE_FRACTION * whole_total
        ):
            self.staging_mode = "needed"
            self._beta_rects: list[tuple[_ByBetaRect, ...]] = [
                tuple(item) for item in plan.beta_rects
            ]
            self._alpha_rects: list[tuple[_ByAlphaRect, ...]] = [
                tuple(item) for item in plan.alpha_rects
            ]
            self._needed_stages: list[Tensor] = list(plan.needed)
        else:
            self.staging_mode = "tile"
            self._beta_rects = whole_beta
            self._alpha_rects = whole_alpha
            self._needed_stages = []
            for stage in range(self._world_size):
                if stage == 0:
                    needed = bounds.pair_index.detach().to(device="cpu", dtype=torch.int64)
                elif bounds.local_size == 0:
                    # No owned output block reads any peer slab, so there is
                    # nothing to request; peers ask nothing of an empty tile.
                    needed = torch.empty(0, dtype=torch.int64)
                else:
                    parts = [
                        _rect_pair_index(
                            self._beta_groups[rect.group],
                            rect.row_start,
                            rect.row_stop,
                            rect.col_start,
                            rect.col_stop,
                        )
                        .detach()
                        .cpu()
                        for rect in self._beta_rects[stage]
                    ]
                    needed = (
                        torch.sort(torch.cat(parts)).values
                        if parts
                        else torch.empty(0, dtype=torch.int64)
                    )
                self._needed_stages.append(needed)

        self.stage_sizes = tuple(int(item.numel()) for item in self._needed_stages[1:])
        self.needed_size = sum(self.stage_sizes)
        self.whole_tile_size = whole_total

        self._exchange = StagedPairExchange(
            self.bounds.pair_index,
            self._needed_stages,
            rank=self._rank,
            world_size=self._world_size,
            group=group,
            tag=int(tag),
        )

    def _rect_values(
        self,
        stage: int,
        buffer: Tensor,
        x_local: Tensor,
        group: _Group,
        rect: _ByBetaRect | _ByAlphaRect,
    ) -> Tensor:
        """Gather one staged rectangle's ket amplitudes, row-major.

        Stage 0 reads the local vector through ``local_of_global``; a remote
        stage maps the rectangle's global pair indices into the received sorted
        list with ``searchsorted`` and gathers the buffer.
        """
        pair_index = _rect_pair_index(
            group, rect.row_start, rect.row_stop, rect.col_start, rect.col_stop
        )
        if stage == 0:
            local = self.bounds.local_of_global(pair_index.to(self.bounds.pair_index.device))
            return x_local[local.to(x_local.device)]
        order = self._needed_stages[stage].to(buffer.device)
        return buffer[torch.searchsorted(order, pair_index.to(order.device))]

    def _apply_stage(
        self,
        stage: int,
        buffer: Tensor,
        x_local: Tensor,
        y_alpha: list[Tensor],
        y_beta: list[Tensor],
    ) -> None:
        """Run every slab kernel fed by one stage's buffer.

        The buffer is consumed exactly in the layouts documented on the class:
        the α kernel gets ``(n_src_rows, frag.n_cols)`` row-major, the β kernel
        gets ``(frag.n_rows, n_src_cols)`` with ``col_offset`` the first source
        column (the kernel derives ``T_stride = n_src_cols``), and the αβ kernel
        gets the source rectangle row-major with width ``src_col_stop -
        src_col_offset``.
        """
        for beta_rect in self._beta_rects[stage]:
            if not beta_rect.alpha_dests and not beta_rect.ab_dests:
                continue
            values = self._rect_values(
                stage, buffer, x_local, self._beta_groups[beta_rect.group], beta_rect
            )
            values_2d = values.reshape(beta_rect.n_rows, beta_rect.n_cols)
            for dest in beta_rect.alpha_dests:
                frag = self._alpha_frags[dest]
                col_slice = slice(
                    frag.col_start - beta_rect.col_start, frag.col_stop - beta_rect.col_start
                )
                _timed(
                    "same_spin_s",
                    add_same_spin_slab,
                    y_alpha[dest],
                    values_2d[:, col_slice].contiguous(),
                    self._pair_helpers,
                    self._integrals,
                    self._blocks.by_beta,
                    frag,
                    row_offset=beta_rect.row_start,
                    col_offset=0,
                    chunk_size=self._chunk_size,
                    memo=self._ss_move_cache,
                    memo_key=("ss", dest, beta_rect.group),
                    triton_matvec=self._triton_matvec,
                )
            for dest in beta_rect.ab_dests:
                _timed(
                    "ab_s",
                    add_ab_slab,
                    y_alpha[dest],
                    values,
                    self._pair_helpers,
                    self._integrals,
                    self._single_groups[beta_rect.group],
                    self._alpha_frags[dest],
                    src_row_offset=beta_rect.row_start,
                    src_row_stop=beta_rect.row_stop,
                    src_col_offset=beta_rect.col_start,
                    src_col_stop=beta_rect.col_stop,
                    chunk_size=self._chunk_size,
                    memo=self._ab_move_cache,
                    memo_key=("ab", dest, beta_rect.group),
                )

        for alpha_rect in self._alpha_rects[stage]:
            if not alpha_rect.beta_dests:
                continue
            values = self._rect_values(
                stage, buffer, x_local, self._alpha_groups[alpha_rect.group], alpha_rect
            )
            values_2d = values.reshape(alpha_rect.n_rows, alpha_rect.n_cols)
            for dest in alpha_rect.beta_dests:
                frag = self._beta_frags[dest]
                row_slice = slice(
                    frag.row_start - alpha_rect.row_start, frag.row_stop - alpha_rect.row_start
                )
                _timed(
                    "same_spin_s",
                    add_same_spin_slab,
                    y_beta[dest],
                    values_2d[row_slice, :].contiguous(),
                    self._pair_helpers,
                    self._integrals,
                    self._blocks.by_alpha,
                    frag,
                    row_offset=0,
                    col_offset=alpha_rect.col_start,
                    chunk_size=self._chunk_size,
                    memo=self._ss_move_cache,
                    memo_key=("ss", dest, alpha_rect.group),
                    triton_matvec=self._triton_matvec,
                )

    def __call__(self, x_local: Tensor) -> Tensor:
        """Return this rank's owned output block ``H @ x``.

        Args:
            x_local: ``(N_local,)`` owned input block, in sorted pair order.

        Returns:
            ``(N_local,)`` owned output block, in the same order.
        """
        _check_local_block(x_local, self.N_local, "x_local")
        if profile_enabled():
            reset_block_profile()
        y = self._hii_local * x_local
        # The slab kernels accumulate in the operator's precision (the diagonal
        # and integral dtype), so a narrow (fp32/complex64) subspace input must
        # not force these buffers down to its own dtype.
        out_dtype = y.dtype
        y_alpha = [
            torch.zeros(frag.n_rows * frag.n_cols, dtype=out_dtype, device=x_local.device)
            for frag in self._alpha_frags
        ]
        y_beta = [
            torch.zeros(frag.n_rows * frag.n_cols, dtype=out_dtype, device=x_local.device)
            for frag in self._beta_frags
        ]
        for stage, buffer in self._exchange.stages(x_local):
            self._apply_stage(stage, buffer, x_local, y_alpha, y_beta)
        for frag, out in zip(self._alpha_frags, y_alpha):
            y.index_add_(0, self.bounds.local_of_global(frag.pair_index), out)
        for frag, out in zip(self._beta_frags, y_beta):
            y.index_add_(0, self.bounds.local_of_global(frag.pair_index), out)
        if profile_enabled():
            prof = block_profile()
            ss = self._ss_move_cache
            ab = self._ab_move_cache
            ss_entries = 0 if ss is None else len(ss.entries)
            ss_mb = 0.0 if ss is None else ss.used_bytes / (1024**2)
            ss_full = False if ss is None else ss.full
            ab_entries = 0 if ab is None else len(ab.entries)
            ab_mb = 0.0 if ab is None else ab.used_bytes / (1024**2)
            ab_full = False if ab is None else ab.full
            print(
                f"[tile-pair-profile] rank={self._rank}/{self._world_size} "
                f"ab={prof.get('ab_s', 0.0):.3f}s same_spin={prof.get('same_spin_s', 0.0):.3f}s "
                f"memo_ss_entries={ss_entries} memo_ss_mb={ss_mb:.1f} memo_ss_full={ss_full} "
                f"memo_ab_entries={ab_entries} memo_ab_mb={ab_mb:.1f} memo_ab_full={ab_full}",
                flush=True,
            )
        return y


class LocalPairTileSwapProjector:
    """α↔β-swap projector over a rank's scattered pair tile.

    The full-space pair projector is
    ``P(v)[k] = (v[k] + tau[k] * v[k_swap[k]]) / 2``. Unlike the 1-D
    :class:`~torch_sci_solver.torch_impl.distributed_pairs.LocalPairSwapProjector`
    it never gathers the full vector: each owned pair's ``(β, α)`` partner is
    fetched with a dedicated :class:`StagedPairExchange` whose stage lists are
    the partners owned by each peer, and the received buffers are indexed with
    ``searchsorted``.

    ``__call__`` is collective: every rank calls it once per projection, in the
    same order, because of the staged exchange. Davidson invokes it that way.

    Args:
        pair_basis: the global ``(α, β)`` pair list.
        k_swap: ``(N_local,)`` global pair index of the ``(β, α)`` partner of
            each owned pair, in ``bounds.pair_index`` order.
        tau_signs: ``(N_local,)`` real ±1 sign ``sigma[k] * sigma[k_swap[k]]``
            for the owned pairs.
        bounds: this rank's scattered tile. It fixes the partition; ``rank`` /
            ``world_size`` may only confirm it.
        partition: the 2-D :class:`TilePartition` the bounds were built from;
            its grid supplies each peer's uid slabs.
        rank, world_size, group: process-group identity and communicator.
        tag: base P2P tag of the partner exchange; stage ``s`` uses ``tag + s``.

    Raises:
        ValueError: the partition, bounds and basis disagree, or the swap
            partners do not cover every owned pair.
    """

    def __init__(
        self,
        pair_basis: "PairBasis | StructuralPairBasis",
        k_swap: Tensor,
        tau_signs: Tensor,
        bounds: PairTileBounds,
        partition: TilePartition,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        group: Any | None = None,
        tag: int = 4096,
    ) -> None:
        """Validate the tile and build the per-stage partner plan."""
        self.n = pair_basis.n_pairs
        if bounds.n_pairs != self.n:
            raise ValueError(f"bounds.n_pairs={bounds.n_pairs} != pair count {self.n}")
        if k_swap.shape != (bounds.local_size,):
            raise ValueError(
                f"k_swap shape {tuple(k_swap.shape)} != ({bounds.local_size},) owned block"
            )
        if tau_signs.shape != (bounds.local_size,):
            raise ValueError(
                f"tau_signs shape {tuple(tau_signs.shape)} != ({bounds.local_size},) owned block"
            )
        if partition.world_size != bounds.world_size:
            raise ValueError(
                f"partition world_size={partition.world_size} != bounds world_size="
                f"{bounds.world_size}"
            )
        if (
            partition.n_rows != pair_basis.n_unique_alphas
            or partition.n_cols != pair_basis.n_unique_betas
        ):
            raise ValueError(
                f"partition ({partition.n_rows}, {partition.n_cols}) does not match the "
                f"basis uid axes ({pair_basis.n_unique_alphas}, {pair_basis.n_unique_betas})"
            )
        bound = partition.for_rank(bounds.rank)
        if (bound.row_slab, bound.col_slab, bound.row_start, bound.row_stop) != (
            bounds.row_slab,
            bounds.col_slab,
            bounds.alpha_start,
            bounds.alpha_stop,
        ) or (bound.col_start, bound.col_stop) != (bounds.beta_start, bounds.beta_stop):
            raise ValueError("bounds do not describe this rank's tile of the partition")

        self._rank, self._world_size = _resolve_identity(rank, world_size, group, bounds)
        self.bounds = bounds
        self.N_local = bounds.local_size
        self._tau_signs = tau_signs

        local_index = bounds.pair_index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        partners = torch.as_tensor(k_swap, dtype=torch.int64).reshape(-1).detach().cpu()
        swapped_uid = k_swap.to(pair_basis.device)
        if isinstance(pair_basis, StructuralPairBasis):
            swapped_alpha, swapped_beta = pair_basis.pair_uids(swapped_uid)
            swapped_alpha = swapped_alpha.detach().cpu()
            swapped_beta = swapped_beta.detach().cpu()
        else:
            swapped_alpha = pair_basis.alpha_uid_of_pair[swapped_uid].detach().cpu()
            swapped_beta = pair_basis.beta_uid_of_pair[swapped_uid].detach().cpu()

        grid = partition.grid
        needed: list[Tensor] = []
        stage_plan: dict[int, tuple[Tensor, Tensor]] = {}
        for stage in range(self._world_size):
            if stage == 0:
                stage_needed = local_index
            else:
                peer = (self._rank + stage) % self._world_size
                row_start, row_stop = grid.alpha_bounds[grid.row_slab(peer)]
                col_start, col_stop = grid.beta_bounds[grid.col_slab(peer)]
                inside = (
                    (swapped_alpha >= row_start)
                    & (swapped_alpha < row_stop)
                    & (swapped_beta >= col_start)
                    & (swapped_beta < col_stop)
                )
                stage_needed = torch.sort(partners[inside]).values
            needed.append(stage_needed)
            if partners.numel() and stage_needed.numel():
                position = torch.searchsorted(stage_needed, partners)
                clipped = position.clamp(max=stage_needed.numel() - 1)
                in_stage = (position < stage_needed.numel()) & (stage_needed[clipped] == partners)
                if bool(in_stage.any()):
                    stage_plan[stage] = (
                        in_stage.nonzero(as_tuple=False).reshape(-1),
                        position[in_stage],
                    )

        covered = sum(int(plan[0].numel()) for plan in stage_plan.values())
        if covered != bounds.local_size:
            raise ValueError(
                "the α↔β swap partners do not cover every owned pair; the tile "
                "partition and the pair basis disagree"
            )
        self._stage_plan = stage_plan
        self._exchange = StagedPairExchange(
            local_index,
            needed,
            rank=self._rank,
            world_size=self._world_size,
            group=group,
            tag=int(tag),
        )

    def __call__(self, v_local: Tensor) -> Tensor:
        """Apply ``P`` to this rank's owned pair block.

        Args:
            v_local: ``(N_local,)`` owned input block.

        Returns:
            ``(N_local,)`` owned block of ``P(v)``.
        """
        _check_local_block(v_local, self.N_local, "v_local")
        partner_values = torch.zeros_like(v_local)
        for stage, buffer in self._exchange.stages(v_local):
            plan = self._stage_plan.get(stage)
            if plan is None:
                continue
            local_positions, buffer_positions = plan
            partner_values[local_positions.to(v_local.device)] = buffer[
                buffer_positions.to(buffer.device)
            ]
        tau = self._tau_signs.to(device=v_local.device, dtype=v_local.dtype)
        return 0.5 * (v_local + tau * partner_values)


def pair_tile_init_vector_local(
    pair_basis: "PairBasis | StructuralPairBasis",
    bounds: PairTileBounds,
    *,
    seed_index: int = 0,
    device: torch.device | str,
    dtype: torch.dtype,
    rank: int | None = None,
    world_size: int | None = None,
    group: Any | None = None,
) -> Tensor:
    """Cold-start pair-tile vector with partition-invariant noise.

    Scattered-tile analogue of
    :func:`~torch_sci_solver.torch_impl.distributed_pairs.pair_basis_init_vector_local`:
    the owned block is

    .. math::

        w_k = [k = \\mathrm{seed}] + \\epsilon \\cdot u(k), \\qquad
        k \\in \\text{tile},

    where :math:`u(k)` is a hash of the **global** pair index (salt 0), so the
    assembled vector does not depend on the rank count or partition. The seed
    entry is added only on the rank that owns it; the block is then divided by
    the global norm (one all-reduce), so every rank sees the same
    normalization.

    Collective: every rank must call it.

    Args:
        pair_basis: the global ``(α, β)`` pair list.
        bounds: this rank's scattered pair tile.
        seed_index: global pair index that receives the unit entry.
        device: device of the returned block.
        dtype: real or complex dtype of the returned block.
        rank, world_size, group: process-group identity and communicator.

    Returns:
        ``(N_local,)`` owned block of the normalized initial vector.
    """
    n_pairs = pair_basis.n_pairs
    if bounds.n_pairs != n_pairs:
        raise ValueError(f"bounds.n_pairs={bounds.n_pairs} != n_pairs={n_pairs}")
    if not 0 <= int(seed_index) < int(n_pairs):
        raise ValueError(f"seed_index={seed_index} is outside [0, {n_pairs})")
    _resolve_identity(rank, world_size, group, bounds)

    if isinstance(device, str):
        device = torch.device(device)
    global_index = bounds.pair_index.detach().to(device=device, dtype=torch.int64).reshape(-1)
    local = SEED_NOISE_SCALE * hash_uniform(global_index, salt=0).to(dtype)
    probe = torch.tensor([int(seed_index)], dtype=torch.int64, device=global_index.device)
    position = int(torch.searchsorted(global_index, probe).item())
    if position < bounds.local_size and int(global_index[position]) == int(seed_index):
        local[int(bounds.local_of_global(probe.to(bounds.pair_index.device)).item())] += 1.0

    norm_sq = GlobalSum(group).all_reduce(torch.vdot(local, local).real)
    return local / torch.sqrt(norm_sq.clamp_min(0.0))
