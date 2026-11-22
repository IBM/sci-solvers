"""Owner-computes TPB matvec and αβ-swap projector over contiguous α-row blocks.

Distributed-memory companions of :func:`torch_sci_solver.torch_impl.mult.mult_torch`
and of the solver's full-space αβ-swap projector. Rank ``r`` owns the
contiguous α-row block ``[r0, r1)`` and *all* β columns, so its output block is
the flat range ``[r0 * M_b, r1 * M_b)``. Every Hamiltonian contribution to that
block is computed locally; the only communication is the α-row exchange for
kets that live on another rank (the α-only and αβ terms read
``x[ja * M_b + ib]`` with ``ja`` possibly remote).

The α tables are sliced to the owned bra rows and split once by *ket owner
stage*: stage 0 is the local block, stage ``s`` holds exactly the rows owned by
rank ``(rank + s) % P`` that this rank needs, in ascending global-row order.
Each stage's ket indices are remapped to positions inside that stage's buffer,
so the existing batched kernels (with their ``T_full=`` keyword) and the tiled
Triton launchers apply unchanged against the received buffer.

Notes
-----
The TPB basis is the Cartesian product ``k = ia * M_b + ib``. For an owned bra
row ``ia`` the Hamiltonian couples to

.. math::

    (H x)[ia, ib] = h_{ia,ib} x_{ia,ib}
      + \\sum_{ja} A^{\\alpha}_{ia,ja} x_{ja,ib}
      + \\sum_{jb} B^{\\beta}_{ib,jb} x_{ia,jb}
      + \\sum_{ja,jb} C_{ia,ib,ja,jb} x_{ja,jb},

where :math:`A` (α singles + doubles) and :math:`C` (αβ doubles) read remote
α rows while :math:`B` (β singles + doubles) keeps ``ia`` fixed and is fully
local. Splitting :math:`A` and :math:`C` by the owner of ``ja`` therefore turns
the matvec into one local pass plus one pass per remote stage, and the staged
exchange overlaps the transfer of stage ``s + 1`` with the kernels of stage
``s``.

The αβ-swap projector is

.. math::

    P(v)_{ij} = \\frac{1}{2}\\left(v_{ij} + \\tau_{ij} v_{ji}\\right),
    \\qquad \\tau = \\sigma \\odot \\sigma^{\\mathsf{T}},

with :math:`\\sigma` the interleaved-to-block sign factor
(:func:`torch_sci_solver.torch_impl.rdm._inter_to_block_signs`). Rank ``r``
needs the columns ``[r0, r1)`` of every peer's block, which the same cyclic
staged pattern ships as a transposed piece.

Reference: the distributed selected-basis scheme follows arXiv:2605.01138; the
TPB owner-computes structure is the one introduced for the SBD matvec.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Optional

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.distributed_davidson import (
    RankBounds,
    RowPartition,
    StagedCornerExchange,
    StagedRowExchange,
    StagedSlabExchange,
    TileBounds,
    TilePartition,
    _dist,
)
from torch_sci_solver.torch_impl.helpers import TorchHelper
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult import (
    DEFAULT_CHUNK_SIZE,
    _apply_alpha_double_chunk_fast,
    _apply_alpha_single_chunk_fast,
    _apply_alphabeta_chunk_fast,
    _apply_beta_double_chunk_fast,
    _apply_beta_single_chunk_fast,
    _bra_from_offsets,
    _mult_beta_only_torch,
    _popcount64,
    build_beta_pass_cache,
)

# Kernel selectors accepted by the operator. The solver resolves its own
# "auto" before constructing an operator, so only the concrete kernels appear
# here (an explicit "batched" must never reach Triton).
_MATVEC_IMPLS = ("batched", "triton")

# Triton tile size, matching ``mult_triton.mult_triton``'s BLOCK.
_TRITON_BLOCK = 128


def _resolve_rank(
    rank: Optional[int],
    world_size: Optional[int],
    group: Optional[Any],
    partition: RowPartition | TilePartition,
) -> tuple[int, int]:
    """Fill in ``rank`` / ``world_size`` from the process group or partition."""
    if rank is None or world_size is None:
        dist = _dist()
        initialized = dist.is_available() and dist.is_initialized()
        if rank is None:
            rank = dist.get_rank(group) if initialized else 0
        if world_size is None:
            world_size = dist.get_world_size(group) if initialized else partition.world_size
    return int(rank), int(world_size)


def _has(table: Optional[Tensor]) -> bool:
    """Whether a CSR value array exists and is non-empty."""
    return table is not None and table.numel() > 0


def _validate_string_tensors(adets_t: Tensor, bdets_t: Tensor) -> int:
    """Check the two string tensors share one representation and return its width.

    Strings are ``(M,)`` single words or ``(M, n_limbs)`` int64 limb matrices
    (limb ``k`` holds bits ``[64k, 64k + 64)``). Mixing the two forms would let
    a kernel read one tensor as row-major limbs and the other as words, so it
    is rejected here with a clear error rather than producing wrong numbers.

    Returns:
        The limb count: the trailing width of the matrices, or ``1`` for 1-D
        tensors.
    """
    for name, tensor in (("adets_t", adets_t), ("bdets_t", bdets_t)):
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must be int64, got {tensor.dtype}")
        if tensor.ndim not in (1, 2):
            raise ValueError(f"{name} must be 1-D or 2-D, got ndim={tensor.ndim}")
    if adets_t.ndim != bdets_t.ndim:
        raise ValueError(
            "adets_t and bdets_t must use the same string representation: both 1-D "
            f"bitmasks or both (M, n_limbs) matrices; got ndim={adets_t.ndim} and "
            f"ndim={bdets_t.ndim}"
        )
    if adets_t.ndim == 2 and adets_t.shape[1] != bdets_t.shape[1]:
        raise ValueError(
            "adets_t and bdets_t limb matrices must share one width: got "
            f"{int(adets_t.shape[1])} and {int(bdets_t.shape[1])} limbs"
        )
    return int(adets_t.shape[1]) if adets_t.ndim == 2 else 1


def _string_popcount(strings: Tensor) -> Tensor:
    """Popcount of every string, summed over its limbs.

    ``_popcount64`` counts the set bits of one int64 word; a ``(M, n_limbs)``
    limb matrix holds one word per limb, so summing over the trailing axis
    gives the string's total occupation.
    """
    counts = _popcount64(strings)
    return counts.sum(dim=-1) if counts.ndim == 2 else counts


# ============================================================================
# Per-α-row cost weights
# ============================================================================


def alpha_row_weights(adets_t: Tensor, bdets_t: Tensor, n_orb: int) -> Tensor:
    """Per-α-row matvec cost for :func:`partition_rows_weighted`.

    The cost model follows the kernel structure: an owned α row ``ia``
    contributes ``M_b`` outputs per α excitation, each α single additionally
    scatters over every β single, and every owned output also sees the β-pass
    work (one term per β single/double, row-independent). With
    ``o = popcount(det)`` and ``v = n_orb - o``:

    .. math::

        w(ia) = M_b\\,(n_{\\alpha s} + n_{\\alpha d})(ia)
              + n_{\\alpha s}(ia)\\,\\sum_{ib} n_{\\beta s}(ib)
              + \\sum_{ib}\\left(n_{\\beta s} + n_{\\beta d}\\right)(ib),

    where ``n_αs = o v``, ``n_αd = C(o,2) C(v,2)`` (and likewise for β). Only
    the strings are read: no excitation tables are built.

    Args:
        adets_t: ``(M_a,)`` α-string bitmasks, or ``(M_a, n_limbs)`` limb matrices.
        bdets_t: ``(M_b,)`` β-string bitmasks, or ``(M_b, n_limbs)`` limb matrices.
        n_orb: number of spatial orbitals.

    Returns:
        ``(M_a,)`` float64 non-negative weights on ``adets_t.device``.
    """
    o_a = _string_popcount(adets_t).to(torch.float64)
    v_a = float(n_orb) - o_a
    n_as = o_a * v_a
    n_ad = (o_a * (o_a - 1.0) * 0.5) * (v_a * (v_a - 1.0) * 0.5)

    o_b = _string_popcount(bdets_t).to(torch.float64)
    v_b = float(n_orb) - o_b
    n_bs = o_b * v_b
    n_bd = (o_b * (o_b - 1.0) * 0.5) * (v_b * (v_b - 1.0) * 0.5)
    beta_singles_sum = n_bs.sum()
    beta_doubles_sum = n_bd.sum()

    weights = (
        float(int(bdets_t.shape[0])) * (n_as + n_ad)
        + n_as * beta_singles_sum
        + (beta_singles_sum + beta_doubles_sum)
    )
    return weights.clamp(min=0.0)


def beta_col_weights(adets_t: Tensor, bdets_t: Tensor, n_orb: int) -> Tensor:
    """Per-β-column matvec cost for the tile partition's column cut.

    The mirror image of :func:`alpha_row_weights`: an owned β column ``ib``
    contributes ``M_a`` outputs per β excitation, each β single scatters over
    every α single, and every owned output also sees the α-pass work (one term
    per α single/double, column-independent). Swapping the two spin axes in
    the row model gives

    .. math::

        w(ib) = M_a\\,(n_{\\beta s} + n_{\\beta d})(ib)
              + n_{\\beta s}(ib)\\,\\sum_{ia} n_{\\alpha s}(ia)
              + \\sum_{ia}\\left(n_{\\alpha s} + n_{\\alpha d}\\right)(ia),

    with ``n_s = o v`` and ``n_d = C(o,2) C(v,2)``. Only the strings are read:
    no excitation tables are built.

    Args:
        adets_t: ``(M_a,)`` α-string bitmasks, or ``(M_a, n_limbs)`` limb matrices.
        bdets_t: ``(M_b,)`` β-string bitmasks, or ``(M_b, n_limbs)`` limb matrices.
        n_orb: number of spatial orbitals.

    Returns:
        ``(M_b,)`` float64 non-negative weights on ``bdets_t.device``.
    """
    return alpha_row_weights(bdets_t, adets_t, n_orb)


# ============================================================================
# α-table slicing and per-stage split
# ============================================================================


@dataclass(frozen=True)
class _StageTable:
    """One task's CSR table restricted to a ket-owner stage.

    Attributes:
        offsets: ``(local_rows + 1,)`` local CSR offsets.
        bra: ``(n_entries,)`` local bra row of each entry.
        to: ``(n_entries,)`` ket position inside the stage buffer.
        columns: per-entry orbital-index arrays (``cr``/``an`` or the doubles
            quadruple), unchanged from the full table.
        eij: precomputed Triton same-spin factor, when the Triton kernel is used.
        aux: extra per-stage data (the stage's positions in the full table,
            used to slice the β-pass cache).
    """

    offsets: Tensor
    bra: Tensor
    to: Tensor
    columns: tuple[Tensor, ...]
    eij: Optional[Tensor] = None
    aux: Optional[dict] = None


def _band_stage_table(
    table: Optional[_StageTable], row_lo: int, row_hi: int, local_rows: int
) -> Optional[_StageTable]:
    """Restrict a stage table to the buffer rows ``[row_lo, row_hi)``.

    A chunked corner stage ships one row band of its rectangle, so the α table
    must address positions inside that band: entries outside it are dropped and
    the survivors' ket positions are rebased by ``row_lo``. The CSR offsets are
    rebuilt from the surviving entries, which stay sorted by bra.

    Args:
        table: The full stage's table, or ``None``.
        row_lo: First buffer row kept.
        row_hi: One past the last buffer row kept.
        local_rows: Number of output rows the CSR covers.

    Returns:
        The band table, ``table`` itself when the band covers every entry, or
        ``None`` when no entry survives.
    """
    if table is None:
        return None
    if table.to.numel() == 0:
        return table
    if row_lo == 0 and row_hi >= int(table.to.max()) + 1:
        return table
    keep = (table.to >= row_lo) & (table.to < row_hi)
    if not bool(keep.any()):
        return None
    bra = table.bra[keep]
    counts = torch.bincount(bra, minlength=local_rows)
    offsets = torch.zeros(local_rows + 1, dtype=torch.int64, device=bra.device)
    offsets[1:] = torch.cumsum(counts, 0)
    return _StageTable(
        offsets=offsets,
        bra=bra,
        to=table.to[keep] - row_lo,
        columns=tuple(column[keep] for column in table.columns),
        eij=None if table.eij is None else table.eij[keep],
        aux=None,
    )


def _spin_table_arrays(helper: TorchHelper, kind: str, spin: str) -> tuple:
    """Return ``(offsets, to, columns)`` for one α or β table of ``helper``.

    Args:
        helper: the full or local helper carrying the CSR tables.
        kind: ``"singles"`` or ``"doubles"``.
        spin: ``"alpha"`` or ``"beta"``.

    Returns:
        The table's ``(offsets, to, columns)`` triple.
    """
    prefix = "a" if spin == "alpha" else "b"
    if kind == "singles":
        return (
            getattr(helper, f"{prefix}_singles_offsets"),
            getattr(helper, f"{prefix}_singles_to"),
            (getattr(helper, f"{prefix}_singles_cr"), getattr(helper, f"{prefix}_singles_an")),
        )
    return (
        getattr(helper, f"{prefix}_doubles_offsets"),
        getattr(helper, f"{prefix}_doubles_to"),
        (
            getattr(helper, f"{prefix}_doubles_cr0"),
            getattr(helper, f"{prefix}_doubles_cr1"),
            getattr(helper, f"{prefix}_doubles_an0"),
            getattr(helper, f"{prefix}_doubles_an1"),
        ),
    )


def _local_spin_table(
    helper: TorchHelper,
    kind: str,
    start: int,
    stop: int,
    local: int,
    *,
    spin: str,
) -> Optional[tuple]:
    """Restrict one α or β CSR table to the owned bra range ``[start, stop)``.

    ``generate_local_torch_helpers`` already returns local offsets (flagged by
    ``helper.alpha_is_local`` / ``helper.beta_is_local``; ``local`` is kept as a
    fallback for helpers built without the flag); the full builder's CSR is
    sorted by bra, so the owned entries are the contiguous slice
    ``[offset[start], offset[stop])`` and the offsets are rebased by
    ``offset[start]``.

    Args:
        helper: the full or local helper carrying the CSR tables.
        kind: ``"singles"`` or ``"doubles"``.
        start, stop: the owned bra range (α rows or β columns).
        local: the local bra count for this spin.
        spin: ``"alpha"`` or ``"beta"``.

    Returns:
        ``(offsets, to, columns)`` for the owned slice, or ``None`` when the
        helper does not carry the table.
    """
    offsets, to, columns = _spin_table_arrays(helper, kind, spin)
    if offsets is None or to is None:
        return None
    is_local = helper.alpha_is_local if spin == "alpha" else helper.beta_is_local
    bra_count = helper.n_alpha if spin == "alpha" else helper.n_beta
    if is_local or bra_count == local:
        return offsets, to, columns

    offset_start = int(offsets[start].item())
    offset_stop = int(offsets[stop].item())
    local_offsets = (offsets[start : stop + 1] - offsets[start]).contiguous()
    return (
        local_offsets,
        to[offset_start:offset_stop],
        tuple(c[offset_start:offset_stop] for c in columns),
    )


def _alpha_table_arrays(helper: TorchHelper, kind: str) -> tuple:
    """Return ``(offsets, to, columns)`` for one α table of ``helper``."""
    return _spin_table_arrays(helper, kind, "alpha")


def _local_alpha_table(
    helper: TorchHelper,
    kind: str,
    row_start: int,
    row_stop: int,
    local_rows: int,
) -> Optional[tuple]:
    """Restrict one α CSR table to the owned bra rows ``[row_start, row_stop)``."""
    return _local_spin_table(helper, kind, row_start, row_stop, local_rows, spin="alpha")


def _beta_table_arrays(helper: TorchHelper, kind: str) -> tuple:
    """Return ``(offsets, to, columns)`` for one β table of ``helper``."""
    return _spin_table_arrays(helper, kind, "beta")


def _local_beta_table(
    helper: TorchHelper,
    kind: str,
    col_start: int,
    col_stop: int,
    local_cols: int,
) -> Optional[tuple]:
    """Restrict one β CSR table to the owned bra columns ``[col_start, col_stop)``."""
    return _local_spin_table(helper, kind, col_start, col_stop, local_cols, spin="beta")


def _local_beta_helper(
    helper: TorchHelper,
    singles: Optional[tuple],
    doubles: Optional[tuple],
    local_cols: int,
) -> TorchHelper:
    """A copy of ``helper`` whose β tables are the owned-column slices.

    :func:`build_beta_pass_cache` reads the β tables straight off a helper,
    so the tile path hands it a helper carrying the local slices instead of
    rebuilding the tables itself.
    """
    updates: dict[str, Any] = {"n_beta": local_cols, "beta_is_local": True}
    if singles is not None:
        updates["b_singles_offsets"], updates["b_singles_to"] = singles[0], singles[1]
        updates["b_singles_cr"], updates["b_singles_an"] = singles[2]
    if doubles is not None:
        updates["b_doubles_offsets"], updates["b_doubles_to"] = doubles[0], doubles[1]
        updates["b_doubles_cr0"], updates["b_doubles_cr1"] = doubles[2][0], doubles[2][1]
        updates["b_doubles_an0"], updates["b_doubles_an1"] = doubles[2][2], doubles[2][3]
    return replace(helper, **updates)


def _split_by_owner_slab(
    offsets: Tensor,
    to: Tensor,
    columns: tuple[Tensor, ...],
    *,
    owner_slab: Callable[[Tensor], Tensor],
    group_pos: int,
    group_size: int,
    local_rows: int,
    local_offset: int,
    stage_indices: list[Tensor],
    collect_index: bool = False,
) -> list[_StageTable]:
    """Stable split of a local CSR table by the owner slab of its ket index.

    Entry ``k`` goes to stage ``(owner_slab(to[k]) - group_pos) % group_size``.
    Within a stage the original (bra-ascending) order is kept, and the
    per-stage CSR offsets are rebuilt from the per-bra counts, so each stage
    is itself a valid CSR. ``to`` is remapped to the ket's position inside the
    stage buffer: stage 0 indexes the local slab directly, every other stage
    maps through ``searchsorted`` into that stage's sorted global index list.

    Args:
        offsets: ``(local_rows + 1,)`` local CSR offsets of the full table.
        to: ``(n_entries,)`` global ket index of each entry.
        columns: per-entry orbital-index arrays, unchanged from the full table.
        owner_slab: maps global ket indices to their owning slab id.
        group_pos: This rank's slab on the split axis.
        group_size: Number of slabs on the split axis.
        local_rows: Number of CSR rows (owned α or β strings).
        local_offset: First global index of the local slab.
        stage_indices: Per-stage sorted global index lists (index 0 unused).
        collect_index: Whether to store each stage's positions in the full
            table under ``aux["exc_index"]`` (used to slice per-stage caches).
    """
    device = offsets.device
    bra = _bra_from_offsets(offsets, local_rows, device)
    if to.numel() == 0:
        stage = torch.empty(0, dtype=torch.int64, device=device)
    else:
        stage = (owner_slab(to) - group_pos) % group_size

    tables: list[_StageTable] = []
    for s in range(group_size):
        mask = stage == s
        bra_s = bra[mask]
        to_s = to[mask]
        columns_s = tuple(c[mask] for c in columns)

        stage_offsets = torch.zeros(local_rows + 1, dtype=torch.int64, device=device)
        if bra_s.numel() > 0:
            counts = torch.bincount(bra_s, minlength=local_rows)
            stage_offsets[1:] = torch.cumsum(counts, 0)

        if s == 0:
            to_local = to_s - local_offset
        else:
            # ``to`` may be an int32 table payload; searchsorted needs the
            # value dtype to match the int64 stage index list.
            to_local = torch.searchsorted(stage_indices[s].to(to_s.device), to_s.to(torch.int64))
        aux = (
            {"exc_index": torch.nonzero(mask, as_tuple=False).reshape(-1)}
            if collect_index
            else None
        )
        tables.append(
            _StageTable(offsets=stage_offsets, bra=bra_s, to=to_local, columns=columns_s, aux=aux)
        )
    return tables


def _split_alpha_by_stage(
    offsets: Tensor,
    to: Tensor,
    columns: tuple[Tensor, ...],
    *,
    partition: RowPartition,
    rank: int,
    world_size: int,
    local_rows: int,
    row_start: int,
    stage_rows: list[Tensor],
) -> list[_StageTable]:
    """Stable split of a local α CSR table by ket-owner stage."""
    return _split_by_owner_slab(
        offsets,
        to,
        columns,
        owner_slab=partition.owner_rows,
        group_pos=rank,
        group_size=world_size,
        local_rows=local_rows,
        local_offset=row_start,
        stage_indices=stage_rows,
    )


# ============================================================================
# Owner-computes TPB operator
# ============================================================================


def _select_helpers(
    helpers: Iterable[TorchHelper],
) -> tuple[
    Optional[TorchHelper],
    Optional[TorchHelper],
    Optional[TorchHelper],
    Optional[TorchHelper],
]:
    """Pick the α-singles, α-doubles, β-pass and αβ-pass helpers.

    Returns:
        ``(alpha_singles, alpha_doubles, beta, alphabeta)``; any entry is
        ``None`` when the basis has no table for it.
    """
    alpha_singles: Optional[TorchHelper] = None
    alpha_doubles: Optional[TorchHelper] = None
    beta: Optional[TorchHelper] = None
    alphabeta: Optional[TorchHelper] = None
    for h in helpers:
        if h.task_type in (0, 2) and alpha_singles is None and _has(h.a_singles_to):
            alpha_singles = h
        if h.task_type == 2 and _has(h.a_doubles_to):
            alpha_doubles = h
        if h.task_type == 1:
            beta = h
        if h.task_type == 0 and _has(h.a_singles_to) and _has(h.b_singles_to):
            alphabeta = h
    return alpha_singles, alpha_doubles, beta, alphabeta


# Per-excitation keys of ``build_beta_pass_cache``'s dict: a tile slices them by
# its stage's excitation list, while ``α_occ_M`` (a row-axis matrix) is shared.
_BETA_SINGLE_CACHE_KEYS = frozenset({
    "bra_beta_per_exc",
    "β_sgn_all",
    "h1_term_all",
    "sum_β_all",
    "α_int_vec_all",
    "range_mask_all",
})
_BETA_DOUBLE_CACHE_KEYS = frozenset({
    "bra_beta_double_per_exc",
    "eij_β_all",
    "range1_mask_all",
    "range2_mask_all",
})


def _stage_exc_index(table: _StageTable) -> Tensor:
    """The stage's positions in its full table (built with ``collect_index``)."""
    if table.aux is None or "exc_index" not in table.aux:
        raise RuntimeError("stage table was built without its excitation index")
    return table.aux["exc_index"]


def _slice_beta_cache(cache: dict, index: Tensor, keys: frozenset[str]) -> dict:
    """Restrict a β-pass cache to one stage's excitation list.

    The stage tables are masked subsets of the full table in the original
    order, so slicing every per-excitation entry by the same positions keeps
    the cache aligned with the stage's ``bra``/``to``/``cr``/``an`` arrays.
    """
    if not cache:
        return cache
    out = {k: v for k, v in cache.items() if k not in keys}
    for key in keys & cache.keys():
        out[key] = cache[key][index]
    return out


class LocalTPBOperator:
    """Owner-computes TPB matvec over one contiguous α-row block.

    Each rank applies the full Hamiltonian to its own output rows: ``y =
    hii_local * x_local`` plus the β pass on the local block plus, for every
    ket-owner stage, the α-singles / α-doubles / αβ-doubles passes against the
    stage's received rows.

    The batched kernels are dtype-generic: a complex ``x_local`` with a real
    ``hii_local`` (and complex integrals) gives a complex output unchanged.
    ``matvec_impl="triton"`` supports the same complex inputs: its kernels
    take the interleaved ``(re, im)`` real view of every complex operand and
    do the pair arithmetic in registers.

    Args:
        hii_local: ``(N_local,)`` diagonal the operator applies.
        adets_t: ``(M_a,)`` global α-string bitmasks, or ``(M_a, n_limbs)``
            limb matrices (limb ``k`` holds bits ``[64k, 64k + 64)``).
        bdets_t: ``(M_b,)`` global β-string bitmasks, or ``(M_b, n_limbs)``
            limb matrices.
        n_orb: spatial orbital count.
        integrals: the matvec :class:`TorchIntegrals` bundle.
        helpers: either ``generate_all_torch_helpers`` output or the
            ``generate_local_torch_helpers`` output for this rank's rows. The
            α tables are detected per helper by ``n_alpha == local_rows``.
        partition: the global α-row partition (a :class:`RowPartition`) or a
            2-D :class:`TilePartition`; ``rank`` selects this rank's block.
            A tile partition shrinks the β tables and the β pass to the owned
            column slab and exchanges vector slabs with the row and column
            peer groups.
        rank, world_size, group: process-group identity and communicator.
        matvec_impl: ``"batched"`` or ``"triton"`` (already resolved by the
            caller; an explicit ``"batched"`` never invokes Triton).
        chunk_size: per-kernel workspace bound.
    """

    def __init__(
        self,
        hii_local: Tensor,
        adets_t: Tensor,
        bdets_t: Tensor,
        n_orb: int,
        integrals: TorchIntegrals,
        helpers: Iterable[TorchHelper],
        partition: RowPartition,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        matvec_impl: str = "batched",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        exchange_bytes: Optional[int] = None,
    ) -> None:
        """Slice the owned α tables, split them by stage and build the exchange."""
        if matvec_impl not in _MATVEC_IMPLS:
            raise ValueError(
                f"unknown matvec_impl {matvec_impl!r}; expected one of {_MATVEC_IMPLS} "
                "(resolve 'auto' before constructing LocalTPBOperator)"
            )

        self._n_orb = int(n_orb)
        self._integrals = integrals
        self._bdets_t = bdets_t
        self._n_limbs = _validate_string_tensors(adets_t, bdets_t)
        self._chunk_size = int(chunk_size)
        self._exchange_bytes = None if exchange_bytes is None else int(exchange_bytes)
        self._helpers = tuple(helpers)

        self.M_a = int(adets_t.shape[0])
        self.M_b = int(bdets_t.shape[0])
        self._adets_t = adets_t
        # Triton is available to both layouts; validate it up front so a
        # misconfigured backend fails before any table is built.
        self._triton = matvec_impl == "triton"
        self._mt: Any = None
        self._I2_flat: Optional[Tensor] = None
        if self._triton:
            self._init_triton_state(adets_t)

        self._tile: Optional[TilePartition] = (
            partition if isinstance(partition, TilePartition) else None
        )
        if self._tile is not None:
            if self._tile.n_rows != self.M_a or self._tile.n_cols != self.M_b:
                raise ValueError(
                    f"tile partition shape ({self._tile.n_rows}, {self._tile.n_cols}) does not "
                    f"match the basis ({self.M_a}, {self.M_b})"
                )
            self._init_tile(hii_local, rank, world_size, group)
            return
        if partition.n_rows != self.M_a or partition.row_size != self.M_b:
            raise ValueError(
                "partition shape "
                f"({partition.n_rows}, {partition.row_size}) does not match the "
                f"basis ({self.M_a}, {self.M_b})"
            )

        self._rank, self._world_size = _resolve_rank(rank, world_size, group, partition)
        self.bounds: RankBounds | TileBounds = partition.for_rank(self._rank)
        self._row_start, self._row_stop = partition.rows_for_rank(self._rank)
        self._local_rows = self._row_stop - self._row_start
        self.N_local = self._local_rows * self.M_b
        self.n = self.M_a * self.M_b
        self._adets_local = adets_t[self._row_start : self._row_stop]

        if hii_local.shape != (self.N_local,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({self.N_local},)")
        self._hii_local = hii_local

        # Locate the tables to precompute: the α singles (shared by tasks 2/0),
        # the α doubles (task 2), the β pass (task 1) and the αβ pass (task 0).
        (
            alpha_singles_helper,
            alpha_doubles_helper,
            beta_helper,
            alphabeta_helper,
        ) = _select_helpers(self._helpers)

        alpha_singles = (
            _local_alpha_table(
                alpha_singles_helper, "singles", self._row_start, self._row_stop, self._local_rows
            )
            if alpha_singles_helper is not None
            else None
        )
        alpha_doubles = (
            _local_alpha_table(
                alpha_doubles_helper, "doubles", self._row_start, self._row_stop, self._local_rows
            )
            if alpha_doubles_helper is not None
            else None
        )

        # needed_rows: every remote α row read by the owned α-singles,
        # α-doubles and αβ-doubles entries (the αβ pass reuses the singles).
        remote: list[Tensor] = []
        for table in (alpha_singles, alpha_doubles):
            if table is None:
                continue
            to_global = table[1]
            remote_mask = (to_global < self._row_start) | (to_global >= self._row_stop)
            if bool(remote_mask.any()):
                remote.append(to_global[remote_mask])
        if remote:
            needed_rows = torch.unique(torch.cat(remote))
        else:
            needed_rows = torch.empty(0, dtype=torch.int64, device=adets_t.device)

        self.exchange = StagedRowExchange(
            partition,
            needed_rows,
            rank=self._rank,
            world_size=self._world_size,
            group=group,
        )
        stage_rows = [self.exchange.rows(s) for s in range(self._world_size)]

        if alpha_singles is not None:
            self._singles_stages: list[_StageTable] = _split_alpha_by_stage(
                *alpha_singles,
                partition=partition,
                rank=self._rank,
                world_size=self._world_size,
                local_rows=self._local_rows,
                row_start=self._row_start,
                stage_rows=stage_rows,
            )
        else:
            self._singles_stages = []

        if alpha_doubles is not None:
            doubles_stages = _split_alpha_by_stage(
                *alpha_doubles,
                partition=partition,
                rank=self._rank,
                world_size=self._world_size,
                local_rows=self._local_rows,
                row_start=self._row_start,
                stage_rows=stage_rows,
            )
        else:
            doubles_stages = []

        # β pass: full β tables, shared by every rank. The setup is built once
        # against the local α rows (the α-occupancy matrix is row-local).
        self._beta_helper = beta_helper
        self._beta_cache = build_beta_pass_cache(
            self._adets_local, bdets_t, self._n_orb, self._integrals, self._helpers
        )

        # αβ pass: the β-singles table is full and local; the α side is the
        # same per-stage singles table used above.
        self._alphabeta = alphabeta_helper is not None and bool(self._singles_stages)
        self._beta_singles_bra: Optional[Tensor] = None
        if alphabeta_helper is not None:
            beta_offsets = alphabeta_helper.b_singles_offsets
            assert beta_offsets is not None  # selected via _has(b_singles_to)
            self._beta_singles_offsets = beta_offsets
            self._beta_singles_to = alphabeta_helper.b_singles_to
            self._beta_singles_cr = alphabeta_helper.b_singles_cr
            self._beta_singles_an = alphabeta_helper.b_singles_an
            self._beta_singles_bra = _bra_from_offsets(
                self._beta_singles_offsets, self.M_b, integrals.device
            )

        # Triton: precompute the per-stage α-double eij factors.
        if self._triton:
            doubles_stages = self._triton_eij_stages(doubles_stages)
        self._doubles_stages = doubles_stages

    def _init_triton_state(self, adets_t: Tensor) -> None:
        """Validate Triton availability and cache its kernel entry points.

        Args:
            adets_t: The α-string tensor, only used for its device.

        Raises:
            RuntimeError: no CUDA device, or Triton is not importable.
        """
        from torch_sci_solver.torch_impl import mult_triton as mt

        if adets_t.device.type != "cuda":
            raise RuntimeError(
                f"matvec_impl='triton' requires a CUDA device; got device={adets_t.device}. "
                "Use matvec_impl='batched' on CPU."
            )
        if not mt._HAVE_TRITON:
            raise RuntimeError(
                "matvec_impl='triton' requires Triton (pip install triton). "
                "Use matvec_impl='batched' otherwise."
            )
        self._mt = mt
        self._I2_flat = self._integrals.I2_eri.reshape(-1).contiguous()
        self._n_spinorb = int(self._integrals.n_spinorb)

    def _triton_eij_stages(self, tables: list[_StageTable]) -> list[_StageTable]:
        """Attach the Triton same-spin factor to every non-empty doubles stage.

        The factor only depends on the stage's bra rows (local α indices) and
        its orbital columns, so the same precompute serves the 1-D row stages
        and the 2-D tile stages.
        """
        assert self._mt is not None  # only called under matvec_impl="triton"
        return [
            _StageTable(
                table.offsets,
                table.bra,
                table.to,
                table.columns,
                self._mt._samespin_eij(
                    self._adets_local,
                    table.offsets,
                    table.columns[0],
                    table.columns[1],
                    table.columns[2],
                    table.columns[3],
                    self._n_orb,
                    self._integrals,
                    is_alpha=True,
                ),
            )
            if table.to.numel() > 0
            else table
            for table in tables
        ]

    def __call__(self, x_local: Tensor) -> Tensor:
        """Return this rank's owned output block ``H @ x``.

        Args:
            x_local: ``(N_local,)`` owned input block.

        Returns:
            ``(N_local,)`` owned output block.
        """
        if self._tile is not None:
            return self._call_tile(x_local)
        if x_local.shape != (self.N_local,):
            raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({self.N_local},)")

        x_2d = x_local.view(self._local_rows, self.M_b)
        y = self._hii_local * x_local
        if self._local_rows == 0:
            return y

        # First ``next`` posts stage 1's send/recv; the β pass below runs
        # against the local block while that transfer is in flight.
        stages = self.exchange.stages(x_2d)
        _stage, stage_buffer = next(stages)

        if self._beta_helper is not None:
            _mult_beta_only_torch(
                y,
                x_local,
                self._adets_local,
                self._bdets_t,
                self._n_orb,
                self._integrals,
                self._beta_helper,
                self._local_rows,
                self.M_b,
                self._chunk_size,
                cache=self._beta_cache,
            )

        self._apply_stage(y, x_local, 0, stage_buffer)
        for stage, stage_buffer in stages:
            self._apply_stage(y, x_local, stage, stage_buffer)
        return y

    def _apply_stage(self, y: Tensor, x_local: Tensor, stage: int, T_full: Tensor) -> None:
        """Accumulate one ket-owner stage's α contributions into ``y``.

        ``T_full`` is the stage buffer (stage 0: ``x_local`` viewed 2-D); the
        per-stage tables already index it by buffer position.
        """
        singles = self._singles_stages[stage] if self._singles_stages else None
        doubles = self._doubles_stages[stage] if self._doubles_stages else None

        if singles is not None and singles.to.numel() > 0:
            _apply_alpha_single_chunk_fast(
                y,
                x_local,
                self._adets_local,
                self._bdets_t,
                self._n_orb,
                self._integrals,
                bra_alpha_per_exc=singles.bra,
                ket_alpha_per_exc=singles.to,
                cr=singles.columns[0],
                an=singles.columns[1],
                M_a=self._local_rows,
                M_b=self.M_b,
                chunk_size=self._chunk_size,
                T_full=T_full,
            )

        if doubles is not None and doubles.to.numel() > 0:
            if self._triton:
                self._mt._launch_samespin_double(
                    self._adets_local,
                    self._bdets_t,
                    doubles.offsets,
                    doubles.to,
                    doubles.columns[0],
                    doubles.columns[1],
                    doubles.columns[2],
                    doubles.columns[3],
                    doubles.eij,
                    T_full,
                    y,
                    self.M_b,
                    self.N_local,
                    IS_ALPHA=1,
                    RANGE_OFF=0,
                    n_limbs=self._n_limbs,
                    BLOCK=_TRITON_BLOCK,
                )
            else:
                _apply_alpha_double_chunk_fast(
                    y,
                    x_local,
                    self._adets_local,
                    self._bdets_t,
                    self._n_orb,
                    self._integrals,
                    bra_alpha_per_exc=doubles.bra,
                    ket_alpha_per_exc=doubles.to,
                    cr0=doubles.columns[0],
                    cr1=doubles.columns[1],
                    an0=doubles.columns[2],
                    an1=doubles.columns[3],
                    M_a=self._local_rows,
                    M_b=self.M_b,
                    chunk_size=self._chunk_size,
                    T_full=T_full,
                )

        if self._alphabeta and singles is not None and singles.to.numel() > 0:
            if self._triton:
                self._mt._launch_alphabeta(
                    self._adets_local,
                    self._bdets_t,
                    singles.offsets,
                    singles.columns[0],
                    singles.columns[1],
                    singles.to,
                    self._beta_singles_offsets,
                    self._beta_singles_cr,
                    self._beta_singles_an,
                    self._beta_singles_to,
                    self._I2_flat,
                    T_full,
                    y,
                    self.M_b,
                    self._n_spinorb,
                    self.N_local,
                    n_limbs=self._n_limbs,
                    BLOCK=_TRITON_BLOCK,
                )
            else:
                _apply_alphabeta_chunk_fast(
                    y,
                    x_local,
                    self._adets_local,
                    self._bdets_t,
                    self._integrals,
                    bra_alpha_per_a_exc=singles.bra,
                    ket_alpha_per_a_exc=singles.to,
                    cr_α=singles.columns[0],
                    an_α=singles.columns[1],
                    bra_beta_per_b_exc=self._beta_singles_bra,
                    ket_beta_per_b_exc=self._beta_singles_to,
                    cr_β=self._beta_singles_cr,
                    an_β=self._beta_singles_an,
                    M_a=self._local_rows,
                    M_b=self.M_b,
                    chunk_size=self._chunk_size,
                    T_full=T_full,
                )

    # ------------------------------------------------------------------
    # Tile path: the same channels over a 2-D (α-row x β-column) tile
    # ------------------------------------------------------------------

    def _init_tile(
        self,
        hii_local: Tensor,
        rank: Optional[int],
        world_size: Optional[int],
        group: Optional[Any],
    ) -> None:
        """Build the tile tables, the three staged exchanges and the stage splits."""
        tile = self._tile
        assert tile is not None  # set by the constructor's tile branch
        self._rank, self._world_size = _resolve_rank(rank, world_size, group, tile)
        bound = tile.for_rank(self._rank)
        self.bounds = bound
        self._row_start, self._row_stop = bound.row_start, bound.row_stop
        self._col_start, self._col_stop = bound.col_start, bound.col_stop
        self._local_rows = self._row_stop - self._row_start
        self._local_cols = self._col_stop - self._col_start
        self.N_local = self._local_rows * self._local_cols
        self.n = self.M_a * self.M_b
        self._adets_local = self._adets_t[self._row_start : self._row_stop]
        self._bdets_local = self._bdets_t[self._col_start : self._col_stop]

        if hii_local.shape != (self.N_local,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({self.N_local},)")
        self._hii_local = hii_local

        alpha_singles_helper, alpha_doubles_helper, beta_helper, _alphabeta_helper = (
            _select_helpers(self._helpers)
        )
        alpha_singles = (
            _local_alpha_table(
                alpha_singles_helper, "singles", self._row_start, self._row_stop, self._local_rows
            )
            if alpha_singles_helper is not None
            else None
        )
        alpha_doubles = (
            _local_alpha_table(
                alpha_doubles_helper, "doubles", self._row_start, self._row_stop, self._local_rows
            )
            if alpha_doubles_helper is not None
            else None
        )
        beta_singles = (
            _local_beta_table(
                beta_helper, "singles", self._col_start, self._col_stop, self._local_cols
            )
            if beta_helper is not None
            else None
        )
        beta_doubles = (
            _local_beta_table(
                beta_helper, "doubles", self._col_start, self._col_stop, self._local_cols
            )
            if beta_helper is not None
            else None
        )

        needed_alpha = self._needed((alpha_singles, alpha_doubles), self._row_start, self._row_stop)
        needed_beta = self._needed((beta_singles, beta_doubles), self._col_start, self._col_stop)

        self._alpha_exchange = StagedSlabExchange(
            tile,
            "alpha",
            needed_alpha,
            rank=self._rank,
            world_size=self._world_size,
            group=group,
            tag=0,
        )
        self._beta_exchange = StagedSlabExchange(
            tile,
            "beta",
            needed_beta,
            rank=self._rank,
            world_size=self._world_size,
            group=group,
            tag=1024,
        )
        self._corner_exchange = StagedCornerExchange(
            self._alpha_exchange,
            self._beta_exchange,
            tag=2048,
            max_elems=None if self._exchange_bytes is None else self._exchange_bytes // 16,
        )

        grid = tile.grid
        alpha_stage_rows = [
            self._alpha_exchange.stage_indices(stage) for stage in range(grid.p_alpha)
        ]
        beta_stage_cols = [self._beta_exchange.stage_indices(stage) for stage in range(grid.p_beta)]
        self._singles_stages = (
            _split_by_owner_slab(
                *alpha_singles,
                owner_slab=tile.owner_row_slab,
                group_pos=bound.row_slab,
                group_size=grid.p_alpha,
                local_rows=self._local_rows,
                local_offset=self._row_start,
                stage_indices=alpha_stage_rows,
            )
            if alpha_singles is not None
            else []
        )
        self._doubles_stages = (
            _split_by_owner_slab(
                *alpha_doubles,
                owner_slab=tile.owner_row_slab,
                group_pos=bound.row_slab,
                group_size=grid.p_alpha,
                local_rows=self._local_rows,
                local_offset=self._row_start,
                stage_indices=alpha_stage_rows,
            )
            if alpha_doubles is not None
            else []
        )
        if self._triton and self._doubles_stages:
            self._doubles_stages = self._triton_eij_stages(self._doubles_stages)
        self._beta_singles_stages = (
            _split_by_owner_slab(
                *beta_singles,
                owner_slab=tile.owner_col_slab,
                group_pos=bound.col_slab,
                group_size=grid.p_beta,
                local_rows=self._local_cols,
                local_offset=self._col_start,
                stage_indices=beta_stage_cols,
                collect_index=True,
            )
            if beta_singles is not None
            else []
        )
        self._beta_doubles_stages = (
            _split_by_owner_slab(
                *beta_doubles,
                owner_slab=tile.owner_col_slab,
                group_pos=bound.col_slab,
                group_size=grid.p_beta,
                local_rows=self._local_cols,
                local_offset=self._col_start,
                stage_indices=beta_stage_cols,
                collect_index=True,
            )
            if beta_doubles is not None
            else []
        )

        # The corner stages may be row bands of the (s_a, s_b) rectangle; each
        # sub-stage gets its α table restricted to its band and rebased.
        self._corner_alpha_tables: list[Optional[_StageTable]] = []
        if self._singles_stages:
            for s_a, _s_b, row_lo, row_hi in self._corner_exchange.pairs():
                self._corner_alpha_tables.append(
                    _band_stage_table(self._singles_stages[s_a], row_lo, row_hi, self._local_rows)
                )

        # One β cache against the local tables, sliced per stage so every
        # stage's kernels see the cache rows that match its excitation list.
        beta_local_helper = (
            _local_beta_helper(beta_helper, beta_singles, beta_doubles, self._local_cols)
            if beta_helper is not None
            else None
        )
        cache = (
            build_beta_pass_cache(
                self._adets_local,
                self._bdets_local,
                self._n_orb,
                self._integrals,
                [beta_local_helper],
            )
            if beta_local_helper is not None
            else {}
        )
        self._beta_singles_cache = [
            _slice_beta_cache(cache, _stage_exc_index(table), _BETA_SINGLE_CACHE_KEYS)
            for table in self._beta_singles_stages
        ]
        self._beta_doubles_cache = [
            _slice_beta_cache(cache, _stage_exc_index(table), _BETA_DOUBLE_CACHE_KEYS)
            for table in self._beta_doubles_stages
        ]
        self._alphabeta = bool(self._singles_stages) and bool(self._beta_singles_stages)

    @staticmethod
    def _needed(tables: tuple, start: int, stop: int) -> Tensor:
        """Remote ket indices read by ``tables`` (outside ``[start, stop)``)."""
        remote = []
        for table in tables:
            if table is None:
                continue
            to_global = table[1]
            mask = (to_global < start) | (to_global >= stop)
            if bool(mask.any()):
                remote.append(to_global[mask])
        if not remote:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(remote))

    def _call_tile(self, x_local: Tensor) -> Tensor:
        """Owner-computes matvec over one tile of the ``(M_a, M_b)`` matrix."""
        if x_local.shape != (self.N_local,):
            raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({self.N_local},)")
        x_tile = x_local.view(self._local_rows, self._local_cols)
        y = self._hii_local * x_local
        if self._local_rows == 0 or self._local_cols == 0:
            return y

        # Post the α stream first so stage 1 is in flight while the β stream
        # runs; the (s_a, 0) corners ride the α stage buffers and the
        # (0, s_b) corners the β ones, so nothing ships twice.
        alpha_stages = self._alpha_exchange.stages(x_tile)
        stage_a, buffer_a = next(alpha_stages)

        for stage_b, buffer_b in self._beta_exchange.stages(x_tile):
            self._apply_tile_beta(y, stage_b, buffer_b)
            if self._alphabeta and stage_b > 0:
                self._apply_tile_alphabeta(
                    y, self._singles_stages[0], self._beta_singles_stages[stage_b], buffer_b
                )

        self._apply_tile_alpha(y, stage_a, buffer_a)
        if self._alphabeta:
            self._apply_tile_alphabeta(
                y, self._singles_stages[stage_a], self._beta_singles_stages[0], buffer_a
            )
        for stage_a, buffer_a in alpha_stages:
            self._apply_tile_alpha(y, stage_a, buffer_a)
            if self._alphabeta:
                self._apply_tile_alphabeta(
                    y, self._singles_stages[stage_a], self._beta_singles_stages[0], buffer_a
                )

        if self._alphabeta and self._corner_exchange.pairs() and self._corner_alpha_tables:
            pairs = self._corner_exchange.pairs()
            for stage, buffer_c in self._corner_exchange.stages(x_tile):
                stage_b = pairs[stage][1]
                self._apply_tile_alphabeta(
                    y,
                    self._corner_alpha_tables[stage],
                    self._beta_singles_stages[stage_b],
                    buffer_c,
                )
        return y

    def _apply_tile_alpha(self, y: Tensor, stage: int, T: Tensor) -> None:
        """Accumulate one α-row stage's α singles and doubles into ``y``."""
        singles = self._singles_stages[stage] if self._singles_stages else None
        doubles = self._doubles_stages[stage] if self._doubles_stages else None

        if singles is not None and singles.to.numel() > 0:
            _apply_alpha_single_chunk_fast(
                y,
                T,
                self._adets_local,
                self._bdets_local,
                self._n_orb,
                self._integrals,
                bra_alpha_per_exc=singles.bra,
                ket_alpha_per_exc=singles.to,
                cr=singles.columns[0],
                an=singles.columns[1],
                M_a=self._local_rows,
                M_b=self._local_cols,
                chunk_size=self._chunk_size,
                T_full=T,
            )

        if doubles is not None and doubles.to.numel() > 0:
            if self._triton:
                # The α stage buffer is the source rows against the owned
                # column slab, so its width equals the output width.
                self._mt._launch_samespin_double(
                    self._adets_local,
                    self._bdets_local,
                    doubles.offsets,
                    doubles.to,
                    doubles.columns[0],
                    doubles.columns[1],
                    doubles.columns[2],
                    doubles.columns[3],
                    doubles.eij,
                    T,
                    y,
                    self._local_cols,
                    self.N_local,
                    IS_ALPHA=1,
                    RANGE_OFF=0,
                    T_stride=self._local_cols,
                    n_limbs=self._n_limbs,
                    BLOCK=_TRITON_BLOCK,
                )
            else:
                _apply_alpha_double_chunk_fast(
                    y,
                    T,
                    self._adets_local,
                    self._bdets_local,
                    self._n_orb,
                    self._integrals,
                    bra_alpha_per_exc=doubles.bra,
                    ket_alpha_per_exc=doubles.to,
                    cr0=doubles.columns[0],
                    cr1=doubles.columns[1],
                    an0=doubles.columns[2],
                    an1=doubles.columns[3],
                    M_a=self._local_rows,
                    M_b=self._local_cols,
                    chunk_size=self._chunk_size,
                    T_full=T,
                )

    def _apply_tile_beta(self, y: Tensor, stage: int, T: Tensor) -> None:
        """Accumulate one β-column stage's β singles and doubles into ``y``."""
        stride = int(T.shape[1]) if T.numel() else self._local_cols
        singles = self._beta_singles_stages[stage] if self._beta_singles_stages else None
        doubles = self._beta_doubles_stages[stage] if self._beta_doubles_stages else None

        if singles is not None and singles.to.numel() > 0:
            _apply_beta_single_chunk_fast(
                y,
                T,
                self._adets_local,
                self._bdets_local,
                self._n_orb,
                self._integrals,
                bra_beta_per_exc=singles.bra,
                ket_beta_per_exc=singles.to,
                cr=singles.columns[0],
                an=singles.columns[1],
                M_a=self._local_rows,
                M_b=self._local_cols,
                chunk_size=self._chunk_size,
                cache=self._beta_singles_cache[stage],
                T_stride=stride,
            )

        if doubles is not None and doubles.to.numel() > 0:
            _apply_beta_double_chunk_fast(
                y,
                T,
                self._adets_local,
                self._bdets_local,
                self._n_orb,
                self._integrals,
                bra_beta_per_exc=doubles.bra,
                ket_beta_per_exc=doubles.to,
                cr0=doubles.columns[0],
                cr1=doubles.columns[1],
                an0=doubles.columns[2],
                an1=doubles.columns[3],
                M_a=self._local_rows,
                M_b=self._local_cols,
                chunk_size=self._chunk_size,
                cache=self._beta_doubles_cache[stage],
                T_stride=stride,
            )

    def _apply_tile_alphabeta(
        self,
        y: Tensor,
        alpha: Optional[_StageTable],
        beta: _StageTable,
        T: Tensor,
    ) -> None:
        """Accumulate one α/β stage pair's αβ doubles into ``y``.

        ``T`` is the buffer for that pair: the local tile for ``(0, 0)``, the
        β stage buffer for ``(0, s_b)``, the α stage buffer for ``(s_a, 0)``
        and the corner rectangle for ``(s_a, s_b)`` with both remote. ``alpha``
        is the α table rebased to ``T``'s rows — for a chunked corner stage
        that is a row band of the ``(s_a, s_b)`` table, not the full stage.
        """
        if alpha is None or alpha.to.numel() == 0 or beta.to.numel() == 0:
            return
        if self._triton:
            # ``T`` holds the stage pair's ket rectangle: the α side indexes
            # its rows, the β side its columns, so the row width is the
            # buffer's own width, not the owned output width.
            self._mt._launch_alphabeta(
                self._adets_local,
                self._bdets_local,
                alpha.offsets,
                alpha.columns[0],
                alpha.columns[1],
                alpha.to,
                beta.offsets,
                beta.columns[0],
                beta.columns[1],
                beta.to,
                self._I2_flat,
                T,
                y,
                self._local_cols,
                self._n_spinorb,
                self.N_local,
                T_stride=int(T.shape[1]),
                n_limbs=self._n_limbs,
                BLOCK=_TRITON_BLOCK,
            )
        else:
            _apply_alphabeta_chunk_fast(
                y,
                T,
                self._adets_local,
                self._bdets_local,
                self._integrals,
                bra_alpha_per_a_exc=alpha.bra,
                ket_alpha_per_a_exc=alpha.to,
                cr_α=alpha.columns[0],
                an_α=alpha.columns[1],
                bra_beta_per_b_exc=beta.bra,
                ket_beta_per_b_exc=beta.to,
                cr_β=beta.columns[0],
                an_β=beta.columns[1],
                M_a=self._local_rows,
                M_b=self._local_cols,
                chunk_size=self._chunk_size,
                T_full=T,
                T_stride=int(T.shape[1]),
            )


# ============================================================================
# Local-block αβ-swap projector
# ============================================================================


class LocalAlphaBetaProjector:
    """Local-block αβ-swap projector. Requires ``M_a == M_b``.

    The full-space projector is ``P(v)[i, j] = (v[i, j] + τ[i, j] v[j, i]) / 2``
    with ``τ = σ ⊙ σᵀ``. Rank ``r`` owns rows ``[r0, r1)`` of the matrix and
    must receive every peer's columns ``[r0, r1)`` — the peer's rows of the
    transpose — which the cyclic staged pattern ships as
    ``v_peer[:, r0:r1]`` and places transposed into ``vT[:, peer_rows]``.

    ``τ`` is real and is cast to the input dtype, so a complex ``v_local`` is
    supported; the exchange then stages complex payloads as reals.

    Args:
        adets_t: ``(M_a,)`` global α-string bitmasks, or ``(M_a, n_limbs)``
            limb matrices.
        bdets_t: ``(M_b,)`` global β-string bitmasks (``M_b == M_a``), or
            ``(M_b, n_limbs)`` limb matrices.
        n_orb: spatial orbital count.
        partition: the α-row partition; it doubles as the column partition.
        rank, world_size, group: process-group identity and communicator.
        dtype: dtype of the tau factors and the exchange buffers.
    """

    def __init__(
        self,
        adets_t: Tensor,
        bdets_t: Tensor,
        n_orb: int,
        partition: RowPartition,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build the tau factors and preallocate the staged exchange buffers."""
        from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

        _validate_string_tensors(adets_t, bdets_t)
        self.M_a = int(adets_t.shape[0])
        self.M_b = int(bdets_t.shape[0])
        if self.M_a != self.M_b:
            raise ValueError(
                f"the αβ-swap projector requires M_a == M_b; got {self.M_a} != {self.M_b}"
            )
        if partition.n_rows != self.M_a or partition.row_size != self.M_b:
            raise ValueError(
                "partition shape "
                f"({partition.n_rows}, {partition.row_size}) does not match the "
                f"basis ({self.M_a}, {self.M_b})"
            )

        self._rank, self._world_size = _resolve_rank(rank, world_size, group, partition)
        self._group = group
        self._row_start, self._row_stop = partition.rows_for_rank(self._rank)
        self._local_rows = self._row_stop - self._row_start
        self.N_local = self._local_rows * self.M_b

        # sigma[ia, ib] for the owned α rows against all β strings, and the
        # transposed copy restricted to the owned β columns.
        sigma_local = _inter_to_block_signs(
            adets_t[self._row_start : self._row_stop], bdets_t, n_orb
        )
        sigma_t = _inter_to_block_signs(adets_t, bdets_t[self._row_start : self._row_stop], n_orb).T
        self._tau = (sigma_local * sigma_t).to(dtype)

        self._peer_rows = [
            partition.rows_for_rank((self._rank + s) % self._world_size)
            for s in range(self._world_size)
        ]
        self._max_peer_rows = max((int(stop - start) for start, stop in self._peer_rows), default=0)
        # Two receive buffers, alternating by stage parity; each stage fills
        # ``(n_peer_rows, local_rows)`` and is used transposed. A complex dtype
        # is staged through a real ``(..., local_rows, 2)`` buffer
        # (``view_as_real``): NCCL has no complex scalar type.
        self._buffer_dtype = dtype
        self._buffers = self._allocate_buffers(dtype, adets_t.device)

    def _allocate_buffers(self, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        """Allocate the two staged receive buffers for one input dtype.

        Real dtypes receive into a ``(max_peer_rows, local_rows)`` buffer of
        that dtype. Complex dtypes receive into a real
        ``(max_peer_rows, local_rows, 2)`` buffer, viewed back with
        :func:`torch.view_as_complex` after the wait.
        """
        shape: tuple[int, ...]
        if dtype.is_complex:
            shape = (self._max_peer_rows, self._local_rows, 2)
            buffer_dtype = dtype.to_real()
        else:
            shape = (self._max_peer_rows, self._local_rows)
            buffer_dtype = dtype
        buffers = [torch.empty(shape, dtype=buffer_dtype, device=device) for _ in range(2)]
        return buffers[0], buffers[1]

    def __call__(self, v_local: Tensor) -> Tensor:
        """Apply ``P`` to the owned rows of ``v``.

        Args:
            v_local: ``(N_local,)`` owned rows of the ``(M_a, M_b)`` matrix.

        Returns:
            ``(N_local,)`` owned rows of ``P(v)``.
        """
        if v_local.shape != (self.N_local,):
            raise ValueError(f"v_local shape {tuple(v_local.shape)} != ({self.N_local},)")

        dtype = v_local.dtype
        v2d = v_local.view(self._local_rows, self.M_b)
        tau = self._tau.to(dtype)
        if self._buffer_dtype != dtype:
            # ``dtype`` follows the input, as for the full-space projector.
            self._buffer_dtype = dtype
            self._buffers = self._allocate_buffers(dtype, v_local.device)
        v_t = torch.zeros((self._local_rows, self.M_b), dtype=dtype, device=v_local.device)

        # Stage 0: our own block of the transpose.
        v_t[:, self._row_start : self._row_stop] = v2d[:, self._row_start : self._row_stop].T

        if self._world_size > 1 and self._local_rows > 0:
            self._exchange_transpose(v2d, v_t)

        return ((v2d + tau * v_t) * 0.5).reshape(-1)

    def _exchange_transpose(self, v2d: Tensor, v_t: Tensor) -> None:
        """Cyclic staged exchange of the transposed peer blocks into ``v_t``.

        Stage ``s`` receives from ``(rank + s) % P`` its columns ``[r0, r1)``
        (that peer's rows ``[r0, r1)`` of the transpose) and sends to
        ``(rank - s) % P`` the columns that peer needs. Stage ``s + 1`` is
        posted before stage ``s`` is waited on, into the other buffer, so the
        transfer overlaps the consumption of the previous stage.

        Complex payloads are packed as ``view_as_real(...).reshape(-1)`` and
        received into a real buffer (NCCL has no complex scalar type), then
        viewed back as complex; the real path is unchanged.
        """
        dist = _dist()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "LocalAlphaBetaProjector with world_size > 1 requires an initialized "
                "torch.distributed process group"
            )

        world_size = self._world_size
        payloads: dict[int, Tensor] = {}
        pending = None

        def _post(stage: int):
            """Post one stage's send/recv; returns the work list or ``None``."""
            recv_rank = (self._rank + stage) % world_size
            send_rank = (self._rank - stage) % world_size
            recv_rows = self._peer_rows[stage]
            send_rows = self._peer_rows[world_size - stage]
            n_recv = int(recv_rows[1] - recv_rows[0])
            n_send = int(send_rows[1] - send_rows[0])
            if n_recv == 0 and n_send == 0:
                return None

            ops = []
            is_complex = v2d.is_complex()
            if n_recv > 0:
                buf = self._buffers[stage % 2][:n_recv]
                ops.append(dist.P2POp(dist.irecv, buf, recv_rank, group=self._group))
            if n_send > 0:
                payload = v2d[:, send_rows[0] : send_rows[1]].contiguous()
                payloads[stage] = payload
                # Pack complex payloads as reals; a real path sends the same
                # flat view it always has.
                send = (
                    torch.view_as_real(payload).reshape(-1) if is_complex else payload.reshape(-1)
                )
                ops.append(dist.P2POp(dist.isend, send, send_rank, group=self._group))
            return dist.batch_isend_irecv(ops)

        if world_size > 1:
            pending = _post(1)
        for stage in range(1, world_size):
            if stage + 1 < world_size:
                next_pending = _post(stage + 1)
            else:
                next_pending = None

            if pending is not None:
                for work in pending:
                    work.wait()

            recv_rows = self._peer_rows[stage]
            n_recv = int(recv_rows[1] - recv_rows[0])
            if n_recv > 0:
                buf = self._buffers[stage % 2][:n_recv]
                if v2d.is_complex():
                    buf = torch.view_as_complex(buf)
                v_t[:, recv_rows[0] : recv_rows[1]] = buf.T

            pending = next_pending
        payloads.clear()


@dataclass(frozen=True)
class _TilePeerPlan:
    """One peer's rectangle of the tile transpose exchange.

    The shipped slice and its placement in ``v_t`` share the same local
    coordinates: the sender contributes ``v[SR, SC]`` and the receiver places
    its reply transposed into ``v_t[SR, SC]``, where ``SR = A_i ∩ B_p`` and
    ``SC = B_j ∩ A_p`` are the intersections of the two tiles' slabs.

    Attributes:
        peer: Peer rank.
        row_start: First local row of the slice (inclusive).
        row_stop: One past the last local row.
        col_start: First local column of the slice (inclusive).
        col_stop: One past the last local column.
        recv_offset: Offset of this peer's reply in the flat receive buffer.
    """

    peer: int
    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    recv_offset: int

    @property
    def n_rows(self) -> int:
        """Rows of the shipped slice."""
        return self.row_stop - self.row_start

    @property
    def n_cols(self) -> int:
        """Columns of the shipped slice."""
        return self.col_stop - self.col_start

    @property
    def n_elems(self) -> int:
        """Elements of the shipped slice (and of the reply)."""
        return self.n_rows * self.n_cols


class LocalTileAlphaBetaProjector:
    """Tile-block αβ-swap projector over a 2-D rank grid. Requires ``M_a == M_b``.

    The full-space projector is ``P(v)[ia, ib] = (v[ia, ib] + τ[ia, ib] v[ib, ia]) / 2``
    with ``τ = σ ⊙ σᵀ``. A rank owning the tile ``A_i × B_j`` (α rows × β
    columns) needs ``v[ib, ia]`` for ``ib ∈ B_j`` and ``ia ∈ A_i`` — the global
    rectangle ``B_j × A_i``. Tiles partition the matrix, so that rectangle
    meets the peer tile ``A_p × B_q`` in ``(B_j ∩ A_p) × (A_i ∩ B_q)``, a
    contiguous slice of both tiles whose sender and receiver local coordinates
    coincide. Every rank therefore ships and receives exactly one local tile
    in total (the intersection sizes factorise to ``|A_i|·|B_j|``), whatever
    the grid shape. On a square grid with equal α/β cuts the only non-empty
    intersection is the transposed peer ``(j, i)``, so the exchange degrades
    to a single pairwise transposed send.

    The rectangles' union is exactly the needed block, so every ``(ia, ib)``
    receives one contribution and the rest of ``v_t`` stays zero.

    ``τ`` is real and is cast to the input dtype, so a complex ``v_local`` is
    supported; the exchange then stages complex payloads as reals.

    Args:
        adets_t: ``(M_a,)`` global α-string bitmasks, or ``(M_a, n_limbs)``
            limb matrices.
        bdets_t: ``(M_b,)`` global β-string bitmasks (``M_b == M_a``), or
            ``(M_b, n_limbs)`` limb matrices.
        n_orb: spatial orbital count.
        partition: the 2-D tile partition; the α and β string sets must be
            the same, since the swap maps one onto the other.
        rank, world_size, group: process-group identity and communicator.
        dtype: dtype of the tau factors and the exchange buffers.
    """

    def __init__(
        self,
        adets_t: Tensor,
        bdets_t: Tensor,
        n_orb: int,
        partition: TilePartition,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        group: Optional[Any] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Build the tile tau factors and the static per-peer exchange plan."""
        from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

        _validate_string_tensors(adets_t, bdets_t)
        self.M_a = int(adets_t.shape[0])
        self.M_b = int(bdets_t.shape[0])
        if self.M_a != self.M_b:
            raise ValueError(
                f"the αβ-swap projector requires M_a == M_b; got {self.M_a} != {self.M_b}"
            )
        if partition.n_rows != self.M_a or partition.n_cols != self.M_b:
            raise ValueError(
                "partition shape "
                f"({partition.n_rows}, {partition.n_cols}) does not match the "
                f"basis ({self.M_a}, {self.M_b})"
            )

        self._rank, self._world_size = _resolve_rank(rank, world_size, group, partition)
        self._group = group
        self._partition = partition
        self._bounds = partition.for_rank(self._rank)
        bound = self._bounds
        self._row_start, self._row_stop = bound.row_start, bound.row_stop
        self._col_start, self._col_stop = bound.col_start, bound.col_stop
        self._local_rows = bound.local_rows
        self._local_cols = bound.local_cols
        self.N_local = bound.local_size

        # tau[ia, ib] = σ[ia, ib] · σ[ib, ia] over the owned tile. The second
        # factor reads the transposed sign matrix on the same string set.
        sigma_local = _inter_to_block_signs(
            adets_t[self._row_start : self._row_stop],
            bdets_t[self._col_start : self._col_stop],
            n_orb,
        )
        sigma_t = _inter_to_block_signs(
            adets_t[self._col_start : self._col_stop],
            bdets_t[self._row_start : self._row_stop],
            n_orb,
        ).T
        self._tau = (sigma_local * sigma_t).to(dtype)

        # The local (diagonal) part of the transpose: the tile intersected
        # with its own transposed slabs. On an equal-cut square grid this is
        # the whole tile.
        self._diag_rows = (
            max(self._row_start, self._col_start),
            min(self._row_stop, self._col_stop),
        )
        self._diag_cols = (
            max(self._col_start, self._row_start),
            min(self._col_stop, self._row_stop),
        )
        if self._diag_rows[1] < self._diag_rows[0]:
            self._diag_rows = (self._diag_rows[0], self._diag_rows[0])
        if self._diag_cols[1] < self._diag_cols[0]:
            self._diag_cols = (self._diag_cols[0], self._diag_cols[0])

        self._peers = self._build_exchange_plan()
        self._recv_total = max((plan.recv_offset + plan.n_elems for plan in self._peers), default=0)
        self._buffer_dtype = dtype
        self._recv_flat = self._allocate_recv(dtype, adets_t.device)

    def _build_exchange_plan(self) -> tuple[_TilePeerPlan, ...]:
        """Intersect the tile with every peer tile in the needed rectangle.

        The needed rectangle of rank ``(i, j)`` is ``B_j × A_i``. Its
        intersection with the peer tile ``(p, q)`` is
        ``(B_j ∩ A_p) × (A_i ∩ B_q)``: peer ``p``'s rows stop where
        ``B_j`` stops and start where it starts, and vice versa for the
        columns. Empty intersections are skipped on both sides, so the
        send/receive lists stay symmetric.
        """
        grid = self._partition.grid
        plans: list[_TilePeerPlan] = []
        offset = 0
        for peer in range(self._world_size):
            if peer == self._rank:
                continue
            peer_rows = grid.rows(grid.row_slab(peer))
            peer_cols = grid.cols(grid.col_slab(peer))
            row_start = max(self._row_start, peer_cols[0])
            row_stop = min(self._row_stop, peer_cols[1])
            col_start = max(self._col_start, peer_rows[0])
            col_stop = min(self._col_stop, peer_rows[1])
            if row_stop <= row_start or col_stop <= col_start:
                continue
            plan = _TilePeerPlan(
                peer=peer,
                row_start=row_start,
                row_stop=row_stop,
                col_start=col_start,
                col_stop=col_stop,
                recv_offset=offset,
            )
            plans.append(plan)
            offset += plan.n_elems
        return tuple(plans)

    def _allocate_recv(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Allocate the flat receive buffer for one input dtype.

        Real dtypes receive into a ``(recv_total,)`` buffer; complex dtypes
        into a real ``(2 * recv_total,)`` buffer, viewed back with
        :func:`torch.view_as_complex` per reply.
        """
        if dtype.is_complex:
            return torch.empty(2 * self._recv_total, dtype=dtype.to_real(), device=device)
        return torch.empty(self._recv_total, dtype=dtype, device=device)

    def __call__(self, v_local: Tensor) -> Tensor:
        """Apply ``P`` to the owned tile of ``v``.

        Args:
            v_local: ``(N_local,)`` owned tile of the ``(M_a, M_b)`` matrix.

        Returns:
            ``(N_local,)`` owned tile of ``P(v)``.
        """
        if v_local.shape != (self.N_local,):
            raise ValueError(f"v_local shape {tuple(v_local.shape)} != ({self.N_local},)")

        dtype = v_local.dtype
        if dtype != self._buffer_dtype:
            # ``dtype`` follows the input, as for the full-space projector.
            self._buffer_dtype = dtype
            self._recv_flat = self._allocate_recv(dtype, v_local.device)
        v2d = v_local.view(self._local_rows, self._local_cols)
        tau = self._tau.to(dtype)
        v_t = torch.zeros_like(v2d)

        # The local block of the transpose.
        diag_rows = slice(
            self._diag_rows[0] - self._row_start, self._diag_rows[1] - self._row_start
        )
        diag_cols = slice(
            self._diag_cols[0] - self._col_start, self._diag_cols[1] - self._col_start
        )
        v_t[diag_rows, diag_cols] = v2d[diag_rows, diag_cols].T

        if self._world_size > 1 and self.N_local > 0:
            self._exchange_transpose(v2d, v_t)

        return ((v2d + tau * v_t) * 0.5).reshape(-1)

    def _exchange_transpose(self, v2d: Tensor, v_t: Tensor) -> None:
        """One all-peers exchange of the transposed rectangles into ``v_t``.

        Every peer plan posts its receive before its send with the default
        tag, and each rank pair exchanges exactly one message each way, so the
        batches pair deterministically. The total payload each rank sends and
        receives is one local tile.
        """
        dist = _dist()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "LocalTileAlphaBetaProjector with world_size > 1 requires an initialized "
                "torch.distributed process group"
            )
        if not self._peers:
            # The needed block lies entirely in the local tile (e.g. a
            # diagonal tile of a square grid).
            return

        is_complex = v2d.is_complex()
        payloads: list[Tensor] = []
        ops = []
        for plan in self._peers:
            rows = slice(plan.row_start - self._row_start, plan.row_stop - self._row_start)
            cols = slice(plan.col_start - self._col_start, plan.col_stop - self._col_start)
            n_real = 2 * plan.n_elems if is_complex else plan.n_elems
            recv = self._recv_flat[plan.recv_offset : plan.recv_offset + n_real]
            ops.append(dist.P2POp(dist.irecv, recv, plan.peer, group=self._group))
            payload = v2d[rows, cols].contiguous()
            payloads.append(payload)
            send = torch.view_as_real(payload).reshape(-1) if is_complex else payload.reshape(-1)
            ops.append(dist.P2POp(dist.isend, send, plan.peer, group=self._group))
        works = dist.batch_isend_irecv(ops)
        for work in works:
            work.wait()

        for plan in self._peers:
            n_real = 2 * plan.n_elems if is_complex else plan.n_elems
            reply = self._recv_flat[plan.recv_offset : plan.recv_offset + n_real]
            if is_complex:
                reply = torch.view_as_complex(reply.view(-1, 2))
            block = reply.view(plan.n_cols, plan.n_rows)
            rows = slice(plan.row_start - self._row_start, plan.row_stop - self._row_start)
            cols = slice(plan.col_start - self._col_start, plan.col_stop - self._col_start)
            v_t[rows, cols] = block.T
        payloads.clear()
