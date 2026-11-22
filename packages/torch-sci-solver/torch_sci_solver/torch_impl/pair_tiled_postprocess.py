"""Tiled 1-RDM, 2-RDM and <S^2> for a distributed GDB (explicit pair) state.

Each rank of the tiled GDB solver owns a scattered set of pair amplitudes — the
pairs whose ``α`` and ``β`` strings fall inside its two uid slabs
(:class:`~torch_sci_solver.torch_impl.distributed_pairs_tile.LocalPairTileOperator`).
This module computes the spin-resolved 1-RDM, the spin-resolved 2-RDM and the
``<S^2>`` expectation from those amplitudes without ever materialising the full
length-N pair vector.

Ownership and staging
---------------------
Every observable is accumulated *bra-major*: a rank processes a pair-table row
only when the row's bra pair ``k_bra`` lies in its own tile, so each row has
exactly one owner and the per-rank partials sum to the full expectation. The
bra amplitude is read from the owned block; the ket amplitude may live on any
rank and is read from the matching stage of the operator's staged exchange
(:class:`~torch_sci_solver.torch_impl.distributed_pairs_tile.StagedPairExchange`):
stage 0 is the owned tile, stage ``s > 0`` carries the values of
``operator._exchange.needed[s]`` — a sorted subset of one peer's tile (the
whole tile when the operator staged the whole-tile fallback). A ``searchsorted``
against each stage's sorted index list maps a ket pair to its buffer position,
so a row contributes in exactly the one stage that owns its ket. Every ket
these sweeps read is inside the needed set the operator's plan stages: the
tables are single and double excitations, and the ``S^2`` spin-swap source is
one α single crossed with one β single.

For an explicit pair list the tables are re-derived with the streaming
enumerators of :class:`~torch_sci_solver.torch_impl.pair_helpers.PairHelpers`,
restricted to the owned tile's contiguous pair runs, so no table of size
``nnz(H)`` is stored. A structural basis stores no per-pair arrays, so its
tables are block-walked instead: every owned pair's excitation candidates are
re-derived from the O(M_unique) unique-string arrays and each partner pair is
resolved with :meth:`~torch_sci_solver.torch_impl.pairs.StructuralPairBasis.
pair_index_of_uids`. Both representations emit the same ``k_bra``/``k_ket``
columns, so the accumulators below are shared.

The exchange delivers one stage at a time, so a sweep re-enumerates the owned
rows once per stage: the trade is O(chunk) memory for a factor ``world_size``
in enumeration work.

The 1-D owner-computes layout has no tile and no staged exchange.
:class:`PairLocalSpinSquared` applies the same ``S^2`` diagonal and spin-swap
sweep to the owned contiguous pair block, reading every source amplitude from
the matvec's all-gathered full vector instead. It serves both the ``spin_sq``
penalty and the ``<S^2>`` diagnostic.

Determinant strings are either 1-D single words or ``(M, n_limbs)`` limb
matrices for ``n_orb > 64`` (limb 0 holds the lowest bits). Every string
operation below dispatches on that shape, so the single-word path keeps its
legacy expressions bit for bit.

Conventions
-----------
The amplitudes are the interleaved-Wick eigenvector the matvec consumes, while
the serial builders in :mod:`torch_sci_solver.torch_impl.rdm_pairs` work in
spin-block order. Every buffer is therefore scaled by
``σ = _inter_to_block_signs_pair(α, β, n_orb)`` of its global pair indices
before it is contracted. Bra factors are conjugated (a no-op for real
amplitudes), single-pair weights use ``|C|^2``, and the parity signs are the
spin-block ``parity_batch`` parities of the serial builders.

References
----------
The owner-computes GDB decomposition and its staged exchange are described in
``docs/distributed_state_vector.md``; the index conventions follow the serial
builders :func:`torch_sci_solver.torch_impl.rdm_pairs.make_one_rdm_pair_torch`
and :func:`torch_sci_solver.torch_impl.rdm_pairs.make_two_rdm_pair_torch`.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import _occupancy_bits, parity_batch
from torch_sci_solver.torch_impl.distributed_davidson import GlobalSum
from torch_sci_solver.torch_impl.distributed_pairs_tile import LocalPairTileOperator
from torch_sci_solver.torch_impl.distributed_rdm import (
    DEFAULT_MAX_REPLICATED_BYTES,
    PartitionedRDM,
    _can_partition_rdm,
    _reduce_partitioned,
    should_replicate,
)
from torch_sci_solver.torch_impl.pair_helpers import (
    PairHelpers,
    _bit_mask_below_positions,
    _bit_mask_rows,
    _complement_mask,
    _dispatch_bit_positions,
    _dispatch_found_in,
    _sort_strings,
    _triu_indices,
)
from torch_sci_solver.torch_impl.pairs import StructuralPairBasis
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair
from torch_sci_solver.torch_impl.spin_sq import _popcount_tensor

if TYPE_CHECKING:
    from torch_sci_solver.torch_impl.distributed_pairs import LocalPairOperator

# Pair entries per sign chunk. Bounds the transient ``(chunk, n_orb)`` int64
# occupancy array ``_inter_to_block_signs_pair`` builds for a large stage list.
_SIGN_CHUNK = 1 << 16


# ============================================================================
# Shared sweep geometry
# ============================================================================


@dataclass(frozen=True)
class _Rows:
    """Rows of one streamed table chunk whose ket pair rides the current stage.

    Attributes:
        bra: ``(n,)`` positions of the bra pairs in the owned block.
        ket: ``(n,)`` positions of the ket pairs in the stage buffer.
        global_bra: ``(n,)`` global pair indices of the bra pairs.
        columns: ``(n,)`` orbital-index columns, in the table's own order.
    """

    bra: Tensor
    ket: Tensor
    global_bra: Tensor
    columns: tuple[Tensor, ...]


@dataclass(frozen=True)
class _StructuralPairStrings:
    """Block-arithmetic α/β string lookup for a structural pair basis.

    Indexing resolves the requested global pair indices through the basis'
    block geometry (O(M_unique) tables) instead of an ``(N,)`` per-pair array,
    so every accumulator that reads ``strings[index]`` runs unchanged on a
    structural basis.

    Attributes:
        basis: The structural basis.
        alpha: Read the α strings when True, the β strings otherwise.
    """

    basis: StructuralPairBasis
    alpha: bool

    @property
    def device(self) -> torch.device:
        """Device of the basis' unique-string tables."""
        return self.basis.device

    def __getitem__(self, index: Tensor) -> Tensor:
        """Return the requested spin's bitmask at each global pair index."""
        alpha, beta = self.basis.pair_strings(index)
        return alpha if self.alpha else beta


def _sweep_strings(
    helpers: PairHelpers, local_index: Tensor, device: torch.device
) -> tuple[
    "StructuralPairBasis | None",
    "Tensor | _StructuralPairStrings",
    "Tensor | _StructuralPairStrings",
    Tensor,
    Tensor,
]:
    """Resolve the global and owned α/β strings for either basis representation."""
    structural = getattr(helpers, "structural", None)
    if structural is not None:
        alpha_of_pair: Tensor | _StructuralPairStrings = _StructuralPairStrings(structural, True)
        beta_of_pair: Tensor | _StructuralPairStrings = _StructuralPairStrings(structural, False)
        alpha_local, beta_local = structural.pair_strings(local_index)
    else:
        alpha_of_pair = helpers.alpha_of_pair.to(device=device).to(torch.int64)
        beta_of_pair = helpers.beta_of_pair.to(device=device).to(torch.int64)
        alpha_local = alpha_of_pair[local_index]
        beta_local = beta_of_pair[local_index]
    return structural, alpha_of_pair, beta_of_pair, alpha_local, beta_local


@dataclass(frozen=True)
class _Sweep:
    """Per-call geometry shared by every staged pair-table sweep.

    Attributes:
        operator: The tile operator whose staged exchange is swept.
        helpers: Pair helpers of the global basis (replicated arrays only).
        n_orb: Number of spatial orbitals.
        dtype: Amplitude dtype (real or complex).
        device: Device of the amplitude block and the stage buffers.
        chunk_size: Workspace bound of one accumulation slice.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        local_index: ``(N_local,)`` sorted global pair indices of the tile.
        alpha_of_pair: α string of a global pair index; a ``(N,)`` tensor for
            an explicit basis or a block-arithmetic lookup for a structural one.
        beta_of_pair: β string of a global pair index, in the same form.
        local_blocked: ``(N_local,)`` σ-scaled owned amplitudes.
        runs: Contiguous global pair runs the tile owns.
        structural: The structural basis when the sweep block-walks its rows,
            ``None`` on the explicit-basis streamed-table path.
    """

    operator: LocalPairTileOperator
    helpers: PairHelpers
    n_orb: int
    dtype: torch.dtype
    device: torch.device
    chunk_size: int
    x_local: Tensor
    local_index: Tensor
    alpha_of_pair: "Tensor | _StructuralPairStrings"
    beta_of_pair: "Tensor | _StructuralPairStrings"
    local_blocked: Tensor
    runs: tuple[tuple[int, int], ...]
    structural: "StructuralPairBasis | None" = None

    @classmethod
    def build(cls, operator: LocalPairTileOperator, x_local: Tensor) -> "_Sweep":
        """Validate a tile block and derive the shared sweep geometry.

        Args:
            operator: A :class:`LocalPairTileOperator` with its staged exchange.
            x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.

        Returns:
            The sweep geometry of this call.

        Raises:
            ValueError: ``x_local`` does not have one entry per owned pair.
            TypeError: ``x_local`` is not a floating or complex tensor.
        """
        if x_local.shape != (operator.N_local,):
            raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({operator.N_local},)")
        if not x_local.is_floating_point() and not x_local.is_complex():
            raise TypeError(f"x_local must be floating or complex, got {x_local.dtype}")
        device = x_local.device
        dtype = x_local.dtype
        helpers = operator._pair_helpers
        local_index = operator.bounds.pair_index.detach().to(device=device, dtype=torch.int64)
        structural, alpha_of_pair, beta_of_pair, alpha_local, beta_local = _sweep_strings(
            helpers, local_index, device
        )
        local_blocked = x_local * _block_signs(
            alpha_local,
            beta_local,
            int(operator._n_orb),
            dtype=dtype,
            device=device,
        )
        return cls(
            operator=operator,
            helpers=helpers,
            n_orb=int(operator._n_orb),
            dtype=dtype,
            device=device,
            chunk_size=max(1, int(operator._chunk_size)),
            x_local=x_local,
            local_index=local_index,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            local_blocked=local_blocked,
            runs=_owned_runs(operator.bounds.pair_index),
            structural=structural,
        )

    def block_signs(self, index: Tensor) -> Tensor:
        """Interleaved-Wick → spin-block signs of a global pair-index list.

        Args:
            index: ``(n,)`` global pair indices.

        Returns:
            ``(n,)`` ±1 signs in the amplitude dtype.
        """
        return _block_signs(
            self.alpha_of_pair[index],
            self.beta_of_pair[index],
            self.n_orb,
            dtype=self.dtype,
            device=self.device,
        )

    def stages(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield ``(σ-scaled buffer, sorted global index)`` per exchange stage.

        Stage 0 is the owned block under :attr:`local_index`; stage ``s > 0``
        is the peer tile the exchange delivers in
        ``operator._exchange.needed[s]`` order. The returned values are a new
        σ-scaled tensor, so the exchange's double-buffered payload is never
        mutated.
        """
        needed = self.operator._exchange.needed
        for stage, buffer in self.operator._exchange.stages(self.x_local):
            if stage == 0:
                yield self.local_blocked, self.local_index
            else:
                index = needed[stage].detach().to(device=self.device, dtype=torch.int64)
                yield buffer * self.block_signs(index), index


@dataclass(frozen=True)
class _FlatSweep:
    """Sweep geometry for the 1-D owner-computes pair block.

    Mirrors :class:`_Sweep` for a
    :class:`~torch_sci_solver.torch_impl.distributed_pairs.LocalPairOperator`:
    the owned pairs are one contiguous global range, and source amplitudes are
    read from the operator's all-gathered full vector instead of a staged
    exchange.

    Attributes:
        helpers: Pair helpers of the global basis.
        n_orb: Number of spatial orbitals.
        dtype: Amplitude dtype used for the swap signs.
        device: Device of the owned block.
        chunk_size: Workspace bound of one swap slice.
        local_index: ``(N_local,)`` contiguous global pair indices.
        alpha_of_pair: α string lookup for a global pair index.
        beta_of_pair: β string lookup for a global pair index.
        structural: The structural basis, or ``None`` for an explicit list.
    """

    helpers: PairHelpers
    n_orb: int
    dtype: torch.dtype
    device: torch.device
    chunk_size: int
    local_index: Tensor
    alpha_of_pair: "Tensor | _StructuralPairStrings"
    beta_of_pair: "Tensor | _StructuralPairStrings"
    structural: "StructuralPairBasis | None" = None

    @classmethod
    def build(cls, operator: "LocalPairOperator", *, dtype: torch.dtype) -> "_FlatSweep":
        """Derive the flat sweep geometry of one owned pair block."""
        device = operator.hii_local.device
        helpers = operator.pair_helpers
        local_index = torch.arange(
            operator.bounds.start, operator.bounds.stop, dtype=torch.int64, device=device
        )
        structural, alpha_of_pair, beta_of_pair, _, _ = _sweep_strings(helpers, local_index, device)
        return cls(
            helpers=helpers,
            n_orb=operator.n_orb,
            dtype=dtype,
            device=device,
            chunk_size=max(1, operator.chunk_size),
            local_index=local_index,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            structural=structural,
        )

    def block_signs(self, index: Tensor) -> Tensor:
        """Interleaved-Wick → spin-block signs of a global pair-index list."""
        return _block_signs(
            self.alpha_of_pair[index],
            self.beta_of_pair[index],
            self.n_orb,
            dtype=self.dtype,
            device=self.device,
        )


_SweepLike = _Sweep | _FlatSweep


def _block_signs(
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    n_orb: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Spin-block signs of a list of pairs, in bounded chunks.

    Calls :func:`~torch_sci_solver.torch_impl.rdm_pairs._inter_to_block_signs_pair`
    on at most :data:`_SIGN_CHUNK` pairs at a time so the ``(chunk, n_orb)``
    int64 occupancy workspace stays small on a large peer tile.

    Args:
        alpha_of_pair: ``(n,)`` or ``(n, n_limbs)`` α string of each pair.
        beta_of_pair: ``(n,)`` or ``(n, n_limbs)`` β string of each pair.
        n_orb: Number of spatial orbitals.
        dtype: Amplitude dtype of the returned signs.
        device: Device of the returned signs.

    Returns:
        ``(n,)`` ±1 signs in ``dtype``.
    """
    n = int(alpha_of_pair.shape[0])
    signs = torch.empty(n, dtype=dtype, device=device)
    for start in range(0, n, _SIGN_CHUNK):
        stop = min(start + _SIGN_CHUNK, n)
        chunk = _inter_to_block_signs_pair(
            alpha_of_pair[start:stop], beta_of_pair[start:stop], n_orb
        )
        signs[start:stop] = chunk.to(dtype=dtype)
    return signs


def _owned_runs(pair_index: Tensor) -> tuple[tuple[int, int], ...]:
    """Contiguous global pair runs ``[start, stop)`` covered by a tile.

    A tile is scattered over the flat pair vector — one rectangle per block —
    so its sorted index list is a union of ascending runs. Restricting a
    streamed table to each run enumerates exactly the owned bra rows.

    Args:
        pair_index: ``(N_local,)`` sorted global pair indices of the tile.

    Returns:
        The tile's contiguous runs, in ascending order.
    """
    values = pair_index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if values.numel() == 0:
        return ()
    cuts = (
        torch.nonzero(values[1:] != values[:-1] + 1, as_tuple=False).reshape(-1) + 1
        if values.numel() > 1
        else torch.zeros(0, dtype=torch.int64)
    )
    starts = [0] + cuts.tolist()
    stops = cuts.tolist() + [values.numel()]
    return tuple(
        (int(values[start]), int(values[stop - 1]) + 1)
        for start, stop in zip(starts, stops, strict=True)
    )


def _owned_rows(
    helpers: PairHelpers,
    kind: str,
    runs: tuple[tuple[int, int], ...],
) -> Iterator[dict[str, Tensor]]:
    """Stream one pair table, restricted to the owned bra-pair runs.

    Args:
        helpers: Pair helpers of the global basis.
        kind: Table name — ``"asingles"``, ``"adoubles"``, ``"bsingles"``,
            ``"bdoubles"`` or ``"ab_doubles"``.
        runs: Contiguous global pair runs the tile owns.

    Yields:
        The table chunks, each holding only rows whose bra pair is owned.
    """
    stream = getattr(helpers, kind)
    for start, stop in runs:
        yield from stream(pair_range=(start, stop))


def _table_chunks(sweep: "_Sweep", kind: str) -> Iterator[dict[str, Tensor]]:
    """Stream one excitation table restricted to the sweep's owned bra pairs.

    Args:
        sweep: The sweep geometry of this call.
        kind: Table name — ``"asingles"``, ``"adoubles"``, ``"bsingles"``,
            ``"bdoubles"`` or ``"ab_doubles"``.

    Yields:
        Table chunks with ``k_bra``, ``k_ket`` and the table's columns; every
        ``k_bra`` is owned by this rank and each row appears exactly once.
    """
    if sweep.structural is None:
        yield from _owned_rows(sweep.helpers, kind, sweep.runs)
    else:
        yield from _structural_table_chunks(sweep, sweep.structural, kind)


def _structural_target_uids(
    sorted_unique: Tensor,
    sorted_to_uid: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """``(found, uid)`` of each candidate spin string, in the basis' uid space.

    Args:
        sorted_unique: ``(M_unique,)`` ascending unique strings of one spin, or
            ``(M_unique, n_limbs)`` lexicographically ascending limb matrices.
        sorted_to_uid: ``(M_unique,)`` position in the basis' unique array of
            each sorted entry (the order returned by the lexicographic sort).
        targets: ``(n,)`` candidate bitmasks, or ``(n, n_limbs)`` limb matrices.

    Returns:
        ``(found, uid)``. ``found[i]`` is False when the candidate is not a
        string of the basis; ``uid[i]`` is then meaningless.
    """
    found, position = _dispatch_found_in(sorted_unique, targets)
    return found, sorted_to_uid[position.clamp(max=sorted_to_uid.numel() - 1)]


def _structural_table_chunks(
    sweep: "_Sweep",
    basis: StructuralPairBasis,
    kind: str,
) -> Iterator[dict[str, Tensor]]:
    """Block-walk one excitation table of a structural basis.

    The owned pairs are visited in chunks; each pair's occupied and empty
    orbitals are expanded into candidates and the partner pairs are resolved
    with :meth:`StructuralPairBasis.pair_index_of_uids`. The workspace is one
    chunk of at most ``sweep.chunk_size`` candidate rows (or one pair's full
    candidate set when that is larger), never a length-N table.

    Args:
        sweep: The sweep geometry of this call.
        basis: The structural basis of the sweep.
        kind: Table name, as in :func:`_table_chunks`.

    Yields:
        Table chunks with ``k_bra``, ``k_ket`` and the table's columns, in the
        orbital conventions of the explicit enumerators.
    """
    if kind == "ab_doubles":
        yield from _structural_ab_chunks(
            sweep, basis, int(sweep.helpers.n_a), int(sweep.helpers.n_b)
        )
    else:
        is_alpha = kind in ("asingles", "adoubles")
        order = 1 if kind in ("asingles", "bsingles") else 2
        yield from _structural_same_spin_chunks(sweep, basis, is_alpha=is_alpha, order=order)


def _structural_same_spin_chunks(
    sweep: "_Sweep",
    basis: StructuralPairBasis,
    *,
    is_alpha: bool,
    order: int,
) -> Iterator[dict[str, Tensor]]:
    """Stream same-spin single/double rows of the owned structural pairs.

    Mirrors :func:`torch_sci_solver.torch_impl.pair_helpers._stream_same_spin`:
    for every owned bra pair, annihilate ``order`` occupied orbitals of its
    excited-spin string, create ``order`` empty ones, and keep the candidates
    whose resulting ``(excited string, spectator string)`` pair is in the
    basis.

    Args:
        sweep: The sweep geometry of this call.
        basis: The structural basis of the sweep.
        is_alpha: True for α rows, False for β rows.
        order: 1 for singles, 2 for doubles.

    Yields:
        Chunks with ``k_bra``, ``k_ket``, ``cr0``/``an0`` and, for doubles,
        ``cr1``/``an1``. ``cr`` is occupied in the bra, ``an`` in the ket.
    """
    device = sweep.device
    n_orb = sweep.n_orb
    n_part = int(sweep.helpers.n_a) if is_alpha else int(sweep.helpers.n_b)
    n_empty = n_orb - n_part
    if n_part < order or n_empty < order:
        return
    if order == 1:
        n_per = n_part * n_empty
    else:
        n_per = (n_part * (n_part - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    if n_per == 0:
        return
    unique = (basis.unique_alphas if is_alpha else basis.unique_betas).to(device)
    if unique.numel() == 0:
        return
    sorted_unique, sorted_to_uid = _sort_strings(unique)
    spin_offset = 0 if is_alpha else 1
    one = torch.tensor(1, dtype=torch.int64, device=device)
    string_mask = _complement_mask(unique, n_orb)
    if order == 2:
        an_i, an_j = _triu_indices(n_part, device=device)
        cr_i, cr_j = _triu_indices(n_empty, device=device)
        n_op = int(an_i.shape[0])
        n_ep = int(cr_i.shape[0])
    step = max(1, int(sweep.chunk_size) // n_per)
    local_index = sweep.local_index
    for start in range(0, int(local_index.numel()), step):
        stop = min(start + step, int(local_index.numel()))
        pair_index = local_index[start:stop]
        alpha_uid, beta_uid = basis.pair_uids(pair_index)
        src = unique[alpha_uid if is_alpha else beta_uid]
        occ = _dispatch_bit_positions(src, n_orb, n_part)
        empty = _dispatch_bit_positions(src ^ string_mask, n_orb, n_empty)
        if order == 1:
            an0 = occ.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
            cr0 = empty.unsqueeze(-2).expand(-1, n_part, -1).reshape(-1)
        else:
            an0 = occ[:, an_i].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            an1 = occ[:, an_j].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            cr0 = empty[:, cr_i].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
            cr1 = empty[:, cr_j].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
        source = torch.arange(int(pair_index.numel()), dtype=torch.int64, device=device)
        source = source.unsqueeze(-1).expand(-1, n_per).reshape(-1)
        target = src[source]
        if src.dim() == 1:
            target = target ^ (one << an0) ^ (one << cr0)
            if order == 2:
                target = target ^ (one << an1) ^ (one << cr1)
        else:
            n_limbs = int(src.shape[-1])
            target = target ^ _bit_mask_rows(an0, n_limbs) ^ _bit_mask_rows(cr0, n_limbs)
            if order == 2:
                target = target ^ _bit_mask_rows(an1, n_limbs) ^ _bit_mask_rows(cr1, n_limbs)
        found_string, target_uid = _structural_target_uids(sorted_unique, sorted_to_uid, target)
        spectator = (beta_uid if is_alpha else alpha_uid)[source]
        if is_alpha:
            found_pair, k_ket = basis.pair_index_of_uids(target_uid, spectator)
        else:
            found_pair, k_ket = basis.pair_index_of_uids(spectator, target_uid)
        keep = found_string & found_pair
        if not bool(keep.any()):
            continue
        out = {
            "k_bra": pair_index[source][keep],
            "k_ket": k_ket[keep].to(device=device),
            "cr0": 2 * an0[keep] + spin_offset,
            "an0": 2 * cr0[keep] + spin_offset,
        }
        if order == 2:
            out["cr1"] = 2 * an1[keep] + spin_offset
            out["an1"] = 2 * cr1[keep] + spin_offset
        yield out


def _structural_ab_chunks(
    sweep: "_Sweep",
    basis: StructuralPairBasis,
    n_a: int,
    n_b: int,
) -> Iterator[dict[str, Tensor]]:
    """Stream αβ double-excitation rows of the owned structural pairs.

    Mirrors :func:`torch_sci_solver.torch_impl.pair_helpers._stream_ab_doubles`:
    one α-single candidate times one β-single candidate per owned bra pair,
    kept when both excited strings exist and the resulting pair is in the
    basis. The cross product is expanded in chunk-size slices whenever many
    per-pair candidates survive.

    Args:
        sweep: The sweep geometry of this call.
        basis: The structural basis of the sweep.
        n_a, n_b: α and β electron counts.

    Yields:
        Chunks with ``k_bra``, ``k_ket`` and the four spin-orbital labels
        ``cr_a``, ``an_a``, ``cr_b``, ``an_b``.
    """
    device = sweep.device
    n_orb = sweep.n_orb
    n_a_empty = n_orb - n_a
    n_b_empty = n_orb - n_b
    n_per_a = n_a * n_a_empty
    n_per_b = n_b * n_b_empty
    unique_a = basis.unique_alphas.to(device)
    unique_b = basis.unique_betas.to(device)
    if unique_a.numel() == 0 or unique_b.numel() == 0:
        return
    if n_per_a == 0 or n_per_b == 0:
        return
    sorted_a, perm_a = _sort_strings(unique_a)
    sorted_b, perm_b = _sort_strings(unique_b)
    one = torch.tensor(1, dtype=torch.int64, device=device)
    mask_a = _complement_mask(unique_a, n_orb)
    mask_b = _complement_mask(unique_b, n_orb)
    pair_step = max(1, int(sweep.chunk_size) // max(n_per_a + n_per_b, 1))
    cross_cap = max(1, int(sweep.chunk_size))
    local_index = sweep.local_index
    for start in range(0, int(local_index.numel()), pair_step):
        stop = min(start + pair_step, int(local_index.numel()))
        pair_index = local_index[start:stop]
        alpha_uid, beta_uid = basis.pair_uids(pair_index)
        n_here = int(pair_index.numel())

        a_src = unique_a[alpha_uid]
        a_occ = _dispatch_bit_positions(a_src, n_orb, n_a)
        a_empty = _dispatch_bit_positions(a_src ^ mask_a, n_orb, n_a_empty)
        a_source = torch.arange(n_here, dtype=torch.int64, device=device)
        a_source = a_source.unsqueeze(-1).expand(-1, n_per_a).reshape(-1)
        a_an = a_occ.unsqueeze(-1).expand(-1, -1, n_a_empty).reshape(-1)
        a_cr = a_empty.unsqueeze(-2).expand(-1, n_a, -1).reshape(-1)
        a_target = a_src[a_source]
        if a_src.dim() == 1:
            a_target = a_target ^ (one << a_an) ^ (one << a_cr)
        else:
            a_limbs = int(a_src.shape[-1])
            a_target = a_target ^ _bit_mask_rows(a_an, a_limbs) ^ _bit_mask_rows(a_cr, a_limbs)
        a_found, a_uid = _structural_target_uids(sorted_a, perm_a, a_target)

        b_src = unique_b[beta_uid]
        b_occ = _dispatch_bit_positions(b_src, n_orb, n_b)
        b_empty = _dispatch_bit_positions(b_src ^ mask_b, n_orb, n_b_empty)
        b_source = torch.arange(n_here, dtype=torch.int64, device=device)
        b_source = b_source.unsqueeze(-1).expand(-1, n_per_b).reshape(-1)
        b_an = b_occ.unsqueeze(-1).expand(-1, -1, n_b_empty).reshape(-1)
        b_cr = b_empty.unsqueeze(-2).expand(-1, n_b, -1).reshape(-1)
        b_target = b_src[b_source]
        if b_src.dim() == 1:
            b_target = b_target ^ (one << b_an) ^ (one << b_cr)
        else:
            b_limbs = int(b_src.shape[-1])
            b_target = b_target ^ _bit_mask_rows(b_an, b_limbs) ^ _bit_mask_rows(b_cr, b_limbs)
        b_found, b_uid = _structural_target_uids(sorted_b, perm_b, b_target)

        a_row = a_source[a_found]
        b_row = b_source[b_found]
        a_uid = a_uid[a_found]
        b_uid = b_uid[b_found]
        a_an, a_cr = a_an[a_found], a_cr[a_found]
        b_an, b_cr = b_an[b_found], b_cr[b_found]
        a_cnt = torch.bincount(a_row, minlength=n_here)
        b_cnt = torch.bincount(b_row, minlength=n_here)
        n_cross = a_cnt * b_cnt
        total = int(n_cross.sum())
        if total == 0:
            continue
        a_off = torch.cumsum(a_cnt, 0) - a_cnt
        b_off = torch.cumsum(b_cnt, 0) - b_cnt
        cross_cum = torch.cumsum(n_cross, 0)
        for lo in range(0, total, cross_cap):
            hi = min(lo + cross_cap, total)
            flat = torch.arange(lo, hi, dtype=torch.int64, device=device)
            row = torch.searchsorted(cross_cum, flat, right=True)
            pos = flat - (cross_cum[row] - n_cross[row])
            b_here = b_cnt[row]
            a_local = torch.div(pos, b_here, rounding_mode="floor")
            b_local = pos - a_local * b_here
            a_at = a_off[row] + a_local
            b_at = b_off[row] + b_local
            found, k_ket = basis.pair_index_of_uids(a_uid[a_at], b_uid[b_at])
            if not bool(found.any()):
                continue
            yield {
                "k_bra": pair_index[row[found]],
                "k_ket": k_ket[found].to(device=device),
                "cr_a": 2 * a_an[a_at[found]],
                "an_a": 2 * a_cr[a_at[found]],
                "cr_b": 2 * b_an[b_at[found]] + 1,
                "an_b": 2 * b_cr[b_at[found]] + 1,
            }


def _resolve_rows(
    chunk: dict[str, Tensor],
    local_index: Tensor,
    stage_index: Tensor,
    columns: tuple[str, ...],
) -> "_Rows | None":
    """Select the rows of one chunk whose ket pair rides the current stage.

    Args:
        chunk: One streamed table chunk (``k_bra``, ``k_ket`` and columns).
        local_index: ``(N_local,)`` sorted global indices of the owned tile.
        stage_index: Sorted global indices the stage buffer carries.
        columns: Column names to carry over for the selected rows.

    Returns:
        The selected :class:`_Rows`, or ``None`` when no row's ket lives in
        this stage.

    Raises:
        RuntimeError: a streamed row's bra pair is outside the owned tile.
    """
    if local_index.numel() == 0 or stage_index.numel() == 0:
        return None
    device = local_index.device
    ket = chunk["k_ket"].to(device=device)
    position = torch.searchsorted(stage_index, ket)
    clipped = position.clamp(max=stage_index.numel() - 1)
    keep = (position < stage_index.numel()) & (stage_index[clipped] == ket)
    if not bool(keep.any()):
        return None
    global_bra = chunk["k_bra"].to(device=device)[keep]
    bra = torch.searchsorted(local_index, global_bra).clamp(max=local_index.numel() - 1)
    if not torch.equal(local_index[bra], global_bra):
        raise RuntimeError(
            "a streamed pair-table row's bra pair is outside the owned tile; the "
            "pair ranges and the tile index disagree"
        )
    return _Rows(
        bra=bra,
        ket=clipped[keep],
        global_bra=global_bra,
        columns=tuple(chunk[name].to(device=device)[keep] for name in columns),
    )


def _select_rows(rows: _Rows, keep: Tensor) -> "_Rows | None":
    """Restrict selected rows to a boolean mask over their columns.

    Args:
        rows: The resolved rows of one chunk.
        keep: ``(n,)`` boolean mask over the rows.

    Returns:
        The masked rows, or ``None`` when the mask is empty.
    """
    if not bool(keep.any()):
        return None
    return _Rows(
        bra=rows.bra[keep],
        ket=rows.ket[keep],
        global_bra=rows.global_bra[keep],
        columns=tuple(column[keep] for column in rows.columns),
    )


def _row_slices(rows: _Rows, step: int) -> Iterator[_Rows]:
    """Split selected rows into accumulation slices of at most ``step``.

    Args:
        rows: The resolved rows of one chunk.
        step: Maximum rows per accumulation slice.

    Yields:
        The row slices, in order.
    """
    n = int(rows.bra.numel())
    size = max(1, int(step))
    for start in range(0, n, size):
        stop = min(start + size, n)
        yield _Rows(
            bra=rows.bra[start:stop],
            ket=rows.ket[start:stop],
            global_bra=rows.global_bra[start:stop],
            columns=tuple(column[start:stop] for column in rows.columns),
        )


def _occupancy(
    strings: "Tensor | _StructuralPairStrings",
    index: Tensor,
    n_orb: int,
    dtype: torch.dtype,
) -> Tensor:
    """``(n, n_orb)`` 0/1 occupancy of each pair's spin string.

    Args:
        strings: Global spin strings: ``(N,)`` single words, ``(N, n_limbs)``
            limb matrices, or a block-arithmetic lookup for a structural basis.
        index: ``(n,)`` global pair indices whose strings are taken.
        n_orb: Number of spatial orbitals.
        dtype: dtype of the returned occupancy.

    Returns:
        ``(n, n_orb)`` occupancy in ``dtype``.
    """
    return _occupancy_bits(strings[index], n_orb).to(dtype)


def _sign(
    strings: "Tensor | _StructuralPairStrings",
    global_bra: Tensor,
    start: Tensor,
    stop: Tensor,
    dtype: torch.dtype,
    sign: Tensor | None = None,
) -> Tensor:
    """Spin-block parity of the bra string over the spatial range ``[start, stop)``.

    Args:
        strings: Global spin strings: an ``(N,)`` tensor or a block-arithmetic
            lookup for a structural basis.
        global_bra: ``(n,)`` global pair indices of the bra pairs.
        start: ``(n,)`` first spatial index of the range.
        stop: ``(n,)`` one past the last spatial index.
        dtype: dtype of the returned signs.
        sign: Optional ``(n,)`` incoming sign to multiply into, for a second
            range on the same string.

    Returns:
        ``(n,)`` ±1 signs in ``dtype``.
    """
    if sign is None:
        sign = torch.ones(global_bra.shape, dtype=dtype, device=strings.device)
    return parity_batch(strings[global_bra], None, start, stop, sign)


def _exchange_group(operator: LocalPairTileOperator) -> object | None:
    """Process group of the operator's staged exchange.

    ``None`` means ``WORLD`` to :class:`~torch_sci_solver.torch_impl.
    distributed_davidson.GlobalSum`, which is also what an exchange built
    without an explicit group uses.
    """
    return getattr(operator._exchange, "group", None)


# ============================================================================
# 1-RDM
# ============================================================================


def pair_tiled_one_rdm(
    operator: LocalPairTileOperator,
    x_local: Tensor,
    *,
    gamma_a: Tensor | None = None,
    gamma_b: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(γα, γβ)`` of a tiled pair vector, replicated on every rank.

    Matches :func:`torch_sci_solver.torch_impl.rdm_pairs.make_one_rdm_pair_torch`
    entry for entry: interleaved-Wick signs from ``_inter_to_block_signs_pair``,
    conjugated bra, ``|C|^2`` diagonal weights.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a
            :class:`~torch_sci_solver.torch_impl.distributed_davidson.TilePartition`.
            Its staged exchange is reused and must belong to the same process
            group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block, in
            ``operator.bounds.pair_index`` order.
        gamma_a: Optional ``(L, L)`` accumulation buffer receiving ``γα``.
        gamma_b: Optional ``(L, L)`` accumulation buffer receiving ``γβ``.

    Returns:
        ``(γα, γβ)``, each ``(L, L)`` in the amplitude dtype. Entry ``[p, q]``
        is ``⟨Ψ| a†_{p,σ} a_{q,σ} |Ψ⟩`` in the spin-block convention of the
        serial builder.

    Raises:
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not a floating or complex tensor.

    Notes
    -----
    The diagonal is the local ``Σ_k |C_k|^2 occ_σ(k, p)``. Every off-diagonal
    entry is a single-excitation row ``k_bra → k_ket``:

    .. math::

        γ_σ[cr, an] += sgn_σ(k_bra)\\, C^*[k_bra]\\, C[k_ket],

    with ``sgn_σ`` the spin-block parity of the bra pair's σ string over the
    spatial range between the creation and annihilation orbitals. The bra is
    owned by the processing rank and the ket is read from the stage that owns
    it, so each row contributes exactly once and one packed
    ``all_reduce(SUM)`` restores the full matrix.
    """
    sweep = _Sweep.build(operator, x_local)
    n_orb = sweep.n_orb
    packed = torch.zeros((2, n_orb, n_orb), dtype=sweep.dtype, device=sweep.device)
    _accumulate_one_rdm(sweep, packed)

    GlobalSum(_exchange_group(operator)).all_reduce(packed)
    reduced_a, reduced_b = packed[0], packed[1]
    if gamma_a is not None:
        gamma_a.add_(reduced_a)
        reduced_a = gamma_a
    if gamma_b is not None:
        gamma_b.add_(reduced_b)
        reduced_b = gamma_b
    return reduced_a, reduced_b


def _accumulate_one_rdm(sweep: _Sweep, packed: Any) -> None:
    """Accumulate one rank's partial ``(γα, γβ)`` into ``packed``.

    Args:
        sweep: The sweep geometry of this call.
        packed: ``(2, L, L)`` accumulation target; the partitioned path passes
            a dense packed target with the same ``[0]``/``[1]`` block access.
    """
    _add_one_rdm_diagonal(sweep, packed)

    for blocked_stage, stage_index in sweep.stages():
        for kind, is_alpha in (("asingles", True), ("bsingles", False)):
            gamma = packed[0 if is_alpha else 1]
            strings = sweep.alpha_of_pair if is_alpha else sweep.beta_of_pair
            for chunk in _table_chunks(sweep, kind):
                rows = _resolve_rows(chunk, sweep.local_index, stage_index, ("cr0", "an0"))
                if rows is None:
                    continue
                for part in _row_slices(rows, sweep.chunk_size):
                    _add_one_rdm_singles(gamma, part, strings, blocked_stage, sweep, is_alpha)


def _add_one_rdm_diagonal(sweep: _Sweep, packed: Tensor) -> None:
    """Add the ``|C|^2``-weighted diagonal occupations of the owned pairs.

    Args:
        sweep: The sweep geometry of this call.
        packed: ``(2, L, L)`` accumulation tensor.
    """
    n_orb = sweep.n_orb
    dtype = sweep.dtype
    # |C|^2 is real for a complex vector via conj*C, as in the serial builder.
    weights = (sweep.x_local.conj() * sweep.x_local).to(dtype)
    alpha_bits = _occupancy(sweep.alpha_of_pair, sweep.local_index, n_orb, dtype)
    beta_bits = _occupancy(sweep.beta_of_pair, sweep.local_index, n_orb, dtype)
    occ_a = (weights.unsqueeze(-1) * alpha_bits).sum(dim=0)
    occ_b = (weights.unsqueeze(-1) * beta_bits).sum(dim=0)
    diagonal = torch.arange(n_orb, dtype=torch.int64, device=sweep.device) * (n_orb + 1)
    packed[0].view(-1).index_add_(0, diagonal, occ_a)
    packed[1].view(-1).index_add_(0, diagonal, occ_b)


def _add_one_rdm_singles(
    gamma: Tensor,
    rows: _Rows,
    strings: "Tensor | _StructuralPairStrings",
    blocked_stage: Tensor,
    sweep: _Sweep,
    is_alpha: bool,
) -> None:
    """Accumulate one α- or β-single slice into ``γ_σ[cr, an]``.

    Mirrors ``_acc_alpha_singles_1rdm_pair`` / ``_acc_beta_singles_1rdm_pair``
    in :mod:`torch_sci_solver.torch_impl.rdm_pairs`.

    Args:
        gamma: ``(L, L)`` accumulation tensor of the spin.
        rows: The slice's resolved rows.
        strings: Global strings of the excited spin, in either sweep form.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
        sweep: The sweep geometry of this call.
        is_alpha: ``True`` for an α row, ``False`` for a β row.
    """
    cr, an = rows.columns
    if is_alpha:
        cr_sp = cr // 2
        an_sp = an // 2
    else:
        cr_sp = (cr - 1) // 2
        an_sp = (an - 1) // 2
    sign = _sign(
        strings,
        rows.global_bra,
        torch.minimum(cr_sp, an_sp),
        torch.maximum(cr_sp, an_sp),
        sweep.dtype,
    )
    # conj() on the bra amplitude: a no-op for real C, conjugate-correct for
    # a complex C.
    contribution = sign * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    gamma.view(-1).index_add_(0, cr_sp * sweep.n_orb + an_sp, contribution)


# ============================================================================
# <S^2>
# ============================================================================


def pair_tiled_spin_squared(
    operator: LocalPairTileOperator,
    x_local: Tensor,
    n_a: int,
    n_b: int,
) -> Tensor:
    """Return ``⟨Ψ|S^2|Ψ⟩`` of a tiled pair vector, replicated on every rank.

    Matches the replicated GDB diagnostic, which builds the αβ block with
    :func:`~torch_sci_solver.torch_impl.rdm_pairs.make_two_rdm_ab_pair_torch`
    and feeds it to
    :func:`~torch_sci_solver.torch_impl.rdm.s_squared_from_rdm`.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        n_a: α electron count of the sector.
        n_b: β electron count of the sector.

    Returns:
        0-dim real tensor with the ``S(S+1)`` eigenvalue convention of
        :func:`~torch_sci_solver.torch_impl.rdm.s_squared_from_rdm`.

    Raises:
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not a floating or complex tensor.

    Notes
    -----
    ``S² = S_z² − S_z + N_α − Σ_{p,q} Γ_αβ[p, q, q, p]``. Only ``Γ_αβ`` class A
    and the *matched* class-D rows survive that index pattern; classes B and C
    force ``p = q`` with ``cr = an`` and vanish:

    .. math::

        \\sum_{p,q} Γ_{αβ}[p,q,q,p] =
            \\sum_{k,p} |C_k|^2 occ_α(k,p)\\,occ_β(k,p)
            + \\sum_{matched} sgn_α\\,sgn_β\\,C^*[k_bra]\\,C[k_ket],

    where a class-D row ``(cr_α, an_α, cr_β, an_β)`` is matched when
    ``(cr_β, an_β) = (an_α, cr_α)`` in spatial orbitals. The class-A scalar is
    local to the owned pairs; every matched row is a single ``ab_doubles`` row,
    read from the stage that owns its ket. One scalar ``all_reduce(SUM)``
    combines the partials.
    """
    sweep = _Sweep.build(operator, x_local)
    n_orb = sweep.n_orb
    dtype = sweep.dtype

    # Class A: the doubly-occupied diagonal of the αβ block, owned pairs only.
    weights = (sweep.x_local.conj() * sweep.x_local).to(dtype)
    alpha_bits = _occupancy(sweep.alpha_of_pair, sweep.local_index, n_orb, dtype)
    beta_bits = _occupancy(sweep.beta_of_pair, sweep.local_index, n_orb, dtype)
    doubly = (alpha_bits * beta_bits).sum(dim=1)
    cross = (weights * doubly).sum()

    for blocked_stage, stage_index in sweep.stages():
        for chunk in _table_chunks(sweep, "ab_doubles"):
            rows = _resolve_rows(
                chunk,
                sweep.local_index,
                stage_index,
                ("cr_a", "an_a", "cr_b", "an_b"),
            )
            if rows is None:
                continue
            cr_a, an_a, cr_b, an_b = rows.columns
            cr_a_sp = cr_a // 2
            an_a_sp = an_a // 2
            cr_b_sp = (cr_b - 1) // 2
            an_b_sp = (an_b - 1) // 2
            matched = _select_rows(rows, (an_a_sp == cr_b_sp) & (an_b_sp == cr_a_sp))
            if matched is None:
                continue
            for part in _row_slices(matched, sweep.chunk_size):
                p_cr_a, p_an_a, p_cr_b, p_an_b = part.columns
                sign_a = _sign(
                    sweep.alpha_of_pair,
                    part.global_bra,
                    torch.minimum(p_cr_a // 2, p_an_a // 2),
                    torch.maximum(p_cr_a // 2, p_an_a // 2),
                    dtype,
                )
                sign_b = _sign(
                    sweep.beta_of_pair,
                    part.global_bra,
                    torch.minimum((p_cr_b - 1) // 2, (p_an_b - 1) // 2),
                    torch.maximum((p_cr_b - 1) // 2, (p_an_b - 1) // 2),
                    dtype,
                )
                bra = sweep.local_blocked[part.bra].conj()
                contribution = sign_a * sign_b * bra * blocked_stage[part.ket]
                cross = cross + contribution.sum()

    GlobalSum(_exchange_group(operator)).all_reduce(cross)
    s_z = (float(n_a) - float(n_b)) / 2.0
    s_sq = cross.new_tensor(s_z * s_z - s_z + float(n_a)) - cross
    return s_sq.real if s_sq.is_complex() else s_sq


def pair_tiled_s2_apply(operator: LocalPairTileOperator, x_local: Tensor) -> Tensor:
    """Return the owned pair block of ``S^2 |x>`` without a full-vector fallback.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.
            Its staged exchange is reused and must belong to the same process
            group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block, in
            ``operator.bounds.pair_index`` order.

    Returns:
        ``(N_local,)`` owned block of ``S^2 |x>`` in the interleaved-Wick
        convention and the tile's local pair order, with ``x_local``'s dtype.
        Only this rank's output block is returned; no collective is needed,
        because every bra of the off-diagonal term is owned.

    Raises:
        NotImplementedError: ``operator`` is not a tile operator.
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not a floating or complex tensor.

    Notes
    -----
    In a fixed ``(n_α, n_β)`` sector the serial pair operator
    (:class:`~torch_sci_solver.torch_impl.spin_sq_pairs.PairSpinSqOperator`)
    is

    .. math::

        S^2 = S_z^2 - S_z + N_\\alpha
            + \\sum_{p,q} a^\\dagger_{p\\alpha} a^\\dagger_{q\\beta}
              a_{q\\alpha} a_{p\\beta}.

    The ``p = q`` part is diagonal, ``-N_{dbl}(k)``, so pair ``k`` carries
    ``n_\\alpha - N_{dbl}(k) + S_z^2 - S_z``. The ``p \\neq q`` part moves an
    α electron ``q → p`` and a β electron ``p → q``: the owned target pair
    ``(α_k, β_k)`` receives

    .. math::

        \\alpha_k' = (\\alpha_k \\setminus \\{p\\}) \\cup \\{q\\}, \\qquad
        \\beta_k' = (\\beta_k \\setminus \\{q\\}) \\cup \\{p\\},

    for every ``p`` in ``α_k`` but not ``β_k`` and every ``q`` in ``β_k`` but
    not ``α_k``, with the Jordan-Wigner sign of the four-operator product. The
    source pair ``k'`` is resolved in the replicated pair table and its
    σ-scaled amplitude is read from the exchange stage that owns it, one stage
    per re-enumeration of the owned rows. Every row's bra is owned, so the
    sweep produces the whole owned output block and no collective follows; the
    σ conversion back to interleaved-Wick is one elementwise multiply.
    """
    if not isinstance(operator, LocalPairTileOperator):
        raise NotImplementedError(
            "tiled post-processing requires a LocalPairTileOperator; the 1-D "
            "owner-computes path has a different exchange layout"
        )
    sweep = _Sweep.build(operator, x_local)
    if sweep.local_index.numel() == 0:
        return torch.zeros_like(x_local)

    n_orb = sweep.n_orb
    n_a = int(sweep.helpers.n_a)
    n_b = int(sweep.helpers.n_b)
    sz = 0.5 * (n_a - n_b)
    n_dbl = _popcount_tensor(
        sweep.alpha_of_pair[sweep.local_index] & sweep.beta_of_pair[sweep.local_index]
    )
    if n_dbl.dim() >= 2:
        n_dbl = n_dbl.sum(dim=-1)
    n_dbl = n_dbl.to(sweep.dtype)
    out_block = ((float(n_a) - n_dbl) + (sz * sz - sz)) * sweep.local_blocked

    lookup = _PairLookup.build(sweep)
    # A target pair's α-only set is bounded by ``n_α`` and by ``L - n_β``
    # (the β-empty orbitals); symmetrically for β. Bound one re-enumeration
    # slice to ``chunk_size`` swap candidates.
    max_candidates = max(1, min(n_a, n_orb - n_b) * min(n_b, n_orb - n_a))
    step = max(1, sweep.chunk_size // max_candidates)
    for blocked_stage, stage_index in sweep.stages():
        if stage_index.numel() == 0:
            continue
        for start in range(0, sweep.local_index.numel(), step):
            rows = _swap_rows(sweep, lookup, start, min(start + step, sweep.local_index.numel()))
            if rows is None:
                continue
            target, source, sign = rows
            position = torch.searchsorted(stage_index, source)
            clipped = position.clamp(max=stage_index.numel() - 1)
            keep = (position < stage_index.numel()) & (stage_index[clipped] == source)
            if not bool(keep.any()):
                continue
            out_block.index_add_(0, target[keep], sign[keep] * blocked_stage[position[keep]])
    return out_block * sweep.block_signs(sweep.local_index)


def pair_tiled_s2_diagonal(operator: LocalPairTileOperator) -> Tensor:
    """Return the owned pair block's ``S^2`` diagonal for the penalty.

    In a fixed ``(n_α, n_β)`` sector a pair determinant ``k`` carries
    ``n_\alpha - N_{dbl}(k) + S_z^2 - S_z`` (see :func:`pair_tiled_s2_apply`).
    The diagonal carries no excitation signs, so the interleaved-Wick and
    spin-block diagonals coincide.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.

    Returns:
        ``(N_local,)`` owned diagonal in ``operator.bounds.pair_index`` order,
        in the dtype of ``operator``'s diagonal block.

    Raises:
        NotImplementedError: ``operator`` is not a tile operator.
    """
    if not isinstance(operator, LocalPairTileOperator):
        raise NotImplementedError(
            "tiled post-processing requires a LocalPairTileOperator; the 1-D "
            "owner-computes path has a different exchange layout"
        )
    dtype = operator._hii_local.dtype
    device = operator._hii_local.device
    local_index = operator.bounds.pair_index.detach().to(device=device, dtype=torch.int64)
    if local_index.numel() == 0:
        return torch.empty(0, dtype=dtype, device=device)
    helpers = operator._pair_helpers
    structural = getattr(helpers, "structural", None)
    if structural is not None:
        alpha, beta = structural.pair_strings(local_index)
    else:
        alpha = helpers.alpha_of_pair.to(device=device).to(torch.int64)[local_index]
        beta = helpers.beta_of_pair.to(device=device).to(torch.int64)[local_index]
    return _s2_diagonal_block(alpha, beta, int(helpers.n_a), int(helpers.n_b), dtype=dtype)


def _s2_diagonal_block(
    alpha: Tensor, beta: Tensor, n_a: int, n_b: int, *, dtype: torch.dtype
) -> Tensor:
    """``n_α - N_dbl(k) + S_z^2 - S_z`` for a block of pair strings."""
    n_dbl = _popcount_tensor(alpha & beta)
    if n_dbl.dim() >= 2:
        n_dbl = n_dbl.sum(dim=-1)
    n_dbl = n_dbl.to(dtype)
    sz = 0.5 * (float(n_a) - float(n_b))
    return (float(n_a) - n_dbl) + (sz * sz - sz)


@dataclass(frozen=True)
class _PairLookup:
    """Replicated ``(α, β) → global pair index`` maps of the swap sweep.

    Attributes:
        unique_alphas: ``(M_α,)`` ascending unique α strings on the sweep
            device, as single words or lexicographically ascending
            ``(M_α, n_limbs)`` limb matrices. For a structural basis these are
            the basis' unique strings *sorted*, and :attr:`alpha_perm` maps a
            sorted position back to the basis' uid; for an explicit basis the
            uid order is already sorted, so the sorted position is the uid.
        unique_betas: ``(M_β,)`` ascending unique β strings, same convention.
        pair_keys: ``(N,)`` sorted ``α_uid * M_β + β_uid`` pair keys, or
            ``None`` for a structural basis.
        m_beta: Number of unique β strings.
        structural: The structural basis when pair membership is resolved from
            the block geometry, ``None`` on the explicit pair-key path.
        alpha_perm: ``(M_α,)`` sorted position → basis uid, structural only.
        beta_perm: ``(M_β,)`` sorted position → basis uid, structural only.
    """

    unique_alphas: Tensor
    unique_betas: Tensor
    pair_keys: "Tensor | None"
    m_beta: int
    structural: "StructuralPairBasis | None" = None
    alpha_perm: "Tensor | None" = None
    beta_perm: "Tensor | None" = None

    @classmethod
    def build(cls, sweep: _SweepLike) -> "_PairLookup":
        """Move the helpers' replicated lookup arrays onto the sweep device.

        Args:
            sweep: The sweep geometry of this call.

        Returns:
            The lookup tables of the global basis. A structural basis carries
            sorted unique strings plus the sorted-position → uid maps, because
            its unique arrays are block-major rather than globally sorted.
        """
        helpers = sweep.helpers
        if sweep.structural is not None:
            unique_alphas, alpha_perm = _sort_strings(
                helpers.unique_alphas.to(device=sweep.device, dtype=torch.int64)
            )
            unique_betas, beta_perm = _sort_strings(
                helpers.unique_betas.to(device=sweep.device, dtype=torch.int64)
            )
            return cls(
                unique_alphas=unique_alphas,
                unique_betas=unique_betas,
                pair_keys=None,
                m_beta=int(helpers.m_beta),
                structural=sweep.structural,
                alpha_perm=alpha_perm,
                beta_perm=beta_perm,
            )
        unique_alphas, _ = _sort_strings(
            helpers.unique_alphas.to(device=sweep.device, dtype=torch.int64)
        )
        unique_betas, _ = _sort_strings(
            helpers.unique_betas.to(device=sweep.device, dtype=torch.int64)
        )
        return cls(
            unique_alphas=unique_alphas,
            unique_betas=unique_betas,
            pair_keys=helpers.pair_keys.to(device=sweep.device, dtype=torch.int64),
            m_beta=int(helpers.m_beta),
        )


def _swap_rows(
    sweep: _SweepLike,
    lookup: _PairLookup,
    start: int,
    stop: int,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Spin-swap rows of the owned target pairs in ``[start, stop)``.

    For each target pair the candidates are the p-major cross product of its
    α-only and β-only orbitals (``p ∈ α \\ β``, ``q ∈ β \\ α``), each mapped
    to its source pair and serial spin-swap sign.

    Args:
        sweep: The sweep geometry of this call.
        lookup: Replicated pair-lookup tables of the global basis.
        start: First owned pair position of the slice.
        stop: One past the last owned pair position of the slice.

    Returns:
        ``(target, source, sign)``, each ``(n,)``: local positions of the
        target pairs, global indices of the source pairs and the ±1 signs in
        the amplitude dtype; ``None`` when no target has a candidate.
    """
    device = sweep.device
    index = sweep.local_index[start:stop]
    alpha = sweep.alpha_of_pair[index]
    beta = sweep.beta_of_pair[index]
    alpha_only = alpha & ~beta
    beta_only = beta & ~alpha
    n_p = _popcount_tensor(alpha_only)
    n_q = _popcount_tensor(beta_only)
    if n_p.dim() >= 2:
        n_p = n_p.sum(dim=-1)
        n_q = n_q.sum(dim=-1)
    block_sizes = n_p * n_q
    total = int(block_sizes.sum())
    if total == 0:
        return None

    p_cols = torch.nonzero(_occupancy_bits(alpha_only, sweep.n_orb).bool(), as_tuple=True)[1]
    q_cols = torch.nonzero(_occupancy_bits(beta_only, sweep.n_orb).bool(), as_tuple=True)[1]
    row = torch.repeat_interleave(torch.arange(index.numel(), device=device), block_sizes)
    block_starts = torch.cumsum(block_sizes, 0) - block_sizes
    local = torch.arange(total, device=device) - torch.repeat_interleave(block_starts, block_sizes)
    n_q_of_row = n_q[row]
    p_start = torch.cumsum(n_p, 0) - n_p
    q_start = torch.cumsum(n_q, 0) - n_q
    p = p_cols[p_start[row] + local // n_q_of_row]
    q = q_cols[q_start[row] + local % n_q_of_row]

    one = torch.ones((), dtype=torch.int64, device=device)
    if alpha.dim() == 1:
        p_bit = one << p
        q_bit = one << q
    else:
        n_limbs = int(alpha.shape[-1])
        p_bit = _bit_mask_rows(p, n_limbs)
        q_bit = _bit_mask_rows(q, n_limbs)
    source_alpha = (alpha[row] ^ p_bit) | q_bit
    source_beta = (beta[row] ^ q_bit) | p_bit
    sign = _swap_sign(source_alpha, source_beta, p, q).to(sweep.dtype)

    alpha_ok, alpha_uid = _dispatch_found_in(lookup.unique_alphas, source_alpha)
    beta_ok, beta_uid = _dispatch_found_in(lookup.unique_betas, source_beta)
    if lookup.structural is not None:
        # The structural unique arrays are block-major, so map the sorted
        # position back to the basis uid before resolving the pair.
        assert lookup.alpha_perm is not None and lookup.beta_perm is not None
        alpha_uid = lookup.alpha_perm[alpha_uid.clamp(max=lookup.alpha_perm.numel() - 1)]
        beta_uid = lookup.beta_perm[beta_uid.clamp(max=lookup.beta_perm.numel() - 1)]
        found, position = lookup.structural.pair_index_of_uids(alpha_uid, beta_uid)
        keep = alpha_ok & beta_ok & found
        return row[keep] + start, position[keep].to(device=device), sign[keep]
    key = alpha_uid * lookup.m_beta + beta_uid
    assert lookup.pair_keys is not None
    position = torch.searchsorted(lookup.pair_keys, key)
    pair_ok = (position < lookup.pair_keys.numel()) & (
        lookup.pair_keys[position.clamp(max=lookup.pair_keys.numel() - 1)] == key
    )
    keep = alpha_ok & beta_ok & pair_ok
    return row[keep] + start, position[keep], sign[keep]


def _swap_sign(source_alpha: Tensor, source_beta: Tensor, p: Tensor, q: Tensor) -> Tensor:
    """Jordan-Wigner sign of one serial spin-swap matrix element.

    Vectorized form of ``_spin_swap_sign`` in
    :mod:`torch_sci_solver.torch_impl.spin_sq`: the sign of
    ``⟨J| a†_{pα} a†_{qβ} a_{qα} a_{pβ} |I⟩`` for the source pair ``I`` with
    ``q ∈ α_I \\setminus β_I`` and ``p ∈ β_I \\setminus α_I``, including the
    ``-1`` from anticommuting the α and β operators.

    Args:
        source_alpha: ``(n,)`` or ``(n, n_limbs)`` α strings of the source pairs.
        source_beta: ``(n,)`` or ``(n, n_limbs)`` β strings of the source pairs.
        p: ``(n,)`` β-only orbital moved to the α string.
        q: ``(n,)`` α-only orbital moved to the β string.

    Returns:
        ``(n,)`` int64 ±1 signs.
    """
    one = torch.ones((), dtype=torch.int64, device=source_alpha.device)
    if source_alpha.dim() == 1:
        p_bit = one << p
        q_bit = one << q
    else:
        n_limbs = int(source_alpha.shape[-1])
        p_bit = _bit_mask_rows(p, n_limbs)
        q_bit = _bit_mask_rows(q, n_limbs)

    def below(mask: Tensor, k: Tensor) -> Tensor:
        """Bits of ``mask`` strictly below orbital ``k``."""
        if mask.dim() == 1:
            return mask & ((one << k) - 1)
        return mask & _bit_mask_below_positions(k, int(mask.shape[-1]))

    def parity(mask: Tensor, k: Tensor) -> Tensor:
        """±1 from the parity of the bits of ``mask`` below orbital ``k``."""
        count = _popcount_tensor(below(mask, k))
        if count.dim() >= 2:
            count = count.sum(dim=-1)
        return 1 - 2 * (count & 1)

    sign = parity(source_beta, p)
    sign = sign * parity(source_alpha, q)
    sign = sign * parity(source_beta & ~p_bit, q)
    sign = sign * parity(source_alpha & ~q_bit, p)
    return -sign


class PairLocalSpinSquared:
    """Owner-computes ``S^2`` on a 1-D pair block without a tile grid.

    The 1-D :class:`~torch_sci_solver.torch_impl.distributed_pairs.
    LocalPairOperator` all-gathers the full input once per matvec. This class
    reuses that vector: it applies the ``S^2`` diagonal to the owned block and
    scatters the spin-swap sources straight from ``T_full``, so no O(K) swap
    table and no staged exchange is needed. The candidate enumeration is the
    same vectorized sweep as the tiled path, chunked to ``chunk_size``.

    Args:
        operator: The 1-D owner-computes pair operator.
        sign_dtype: dtype of the swap signs (the resolved Davidson subspace
            dtype); defaults to the operator's diagonal dtype.

    Attributes:
        n: Global pair count.
    """

    def __init__(
        self,
        operator: "LocalPairOperator",
        *,
        sign_dtype: torch.dtype | None = None,
    ) -> None:
        """Build the cached sweep geometry, lookup, diagonal and signs."""
        dtype = operator.hii_local.dtype if sign_dtype is None else sign_dtype
        self._operator = operator
        self._sweep = _FlatSweep.build(operator, dtype=dtype)
        self._lookup = _PairLookup.build(self._sweep)
        helpers = operator.pair_helpers
        local_index = self._sweep.local_index
        alpha = self._sweep.alpha_of_pair[local_index]
        beta = self._sweep.beta_of_pair[local_index]
        self._diag = _s2_diagonal_block(
            alpha,
            beta,
            int(helpers.n_a),
            int(helpers.n_b),
            dtype=operator.hii_local.dtype,
        )
        self._signs = self._sweep.block_signs(local_index)
        n_orb = operator.n_orb
        max_candidates = max(
            1,
            min(int(helpers.n_a), n_orb - int(helpers.n_b))
            * min(int(helpers.n_b), n_orb - int(helpers.n_a)),
        )
        self._step = max(1, self._sweep.chunk_size // max_candidates)
        self._group = operator.group

    @property
    def n(self) -> int:
        """Global pair count."""
        return self._operator.n

    def diagonal(self) -> Tensor:
        """Return the owned ``S^2`` diagonal in the diagonal dtype."""
        return self._diag

    def apply(self, x_local: Tensor, T_full: Tensor) -> Tensor:
        """Return this rank's owned block of ``S^2 |x>``.

        Args:
            x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
            T_full: ``(N,)`` replicated input vector from the operator's
                all-gather.

        Returns:
            ``(N_local,)`` owned block in ``x_local``'s dtype and convention.
        """
        device = x_local.device
        signs = self._signs.to(device=device, dtype=x_local.dtype)
        blocked = x_local * signs
        out = blocked * self._diag.to(device=device, dtype=x_local.dtype)
        n_local = self._sweep.local_index.numel()
        for start in range(0, n_local, self._step):
            rows = _swap_rows(self._sweep, self._lookup, start, min(start + self._step, n_local))
            if rows is None:
                continue
            target, source, sign = rows
            source = source.to(device=T_full.device)
            source_signs = self._sweep.block_signs(source).to(device=device, dtype=x_local.dtype)
            contribution = sign.to(device=device, dtype=x_local.dtype) * (
                T_full[source] * source_signs
            )
            out.index_add_(0, target.to(device=device), contribution)
        return out * signs

    def expectation(self, x_local: Tensor, *, T_full: Tensor | None = None) -> Tensor:
        """Return the globally reduced ``<x|S^2|x>`` scalar.

        Args:
            x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
            T_full: optional pre-gathered full vector; gathered when omitted.
        """
        if T_full is None:
            T_full = self._operator.gather_full(x_local)
        value = torch.vdot(x_local, self.apply(x_local, T_full)).real
        GlobalSum(self._group).all_reduce(value)
        return value


# ============================================================================
# 2-RDM
# ============================================================================


def pair_tiled_two_rdm(
    operator: LocalPairTileOperator,
    x_local: Tensor,
    *,
    gamma2_a: Tensor | None = None,
    gamma2_b: Tensor | None = None,
    gamma2_ab: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(Γαα, Γαβ, Γββ)`` of a tiled pair vector, replicated per rank.

    Matches :func:`torch_sci_solver.torch_impl.rdm_pairs.make_two_rdm_pair_torch`
    entry for entry: classes 0/1/2 of the same-spin blocks and classes A/B/C/D
    of the αβ block, the ``[p, q, r, s]`` chemist storage order, the
    ``(cr, an)`` slot conventions and the canonical double-excitation parity.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.
            Its staged exchange is reused and must belong to the same process
            group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        gamma2_a: Optional ``(L, L, L, L)`` buffer receiving ``Γαα`` (added).
        gamma2_b: Optional ``(L, L, L, L)`` buffer receiving ``Γββ`` (added).
        gamma2_ab: Optional ``(L, L, L, L)`` buffer receiving ``Γαβ`` (added).

    Returns:
        ``(Γαα, Γαβ, Γββ)``, each ``(L, L, L, L)`` in the amplitude dtype, in
        the conventions of the serial builder. Entry ``[p, q, r, s]`` of
        ``Γαβ`` is ``⟨Ψ| a†_{pα} a†_{rβ} a_{sβ} a_{qα} |Ψ⟩``.

    Raises:
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not a floating or complex tensor.

    Notes
    -----
    The serial decomposition maps onto the tiled sweep as follows:

    - class 0 (``Γσσ``) and class A (``Γαβ``) are local ``|C|^2`` weighted
      occupations of the owned pairs;
    - the same-spin class 1 (single excitation × spectator orbital) and class 2
      (double excitation) are

      .. math::

          w_k = sgn_k\\, C^*[k_bra]\\, C[k_ket],

      scattered to the four Slater-Condon slots of the serial builders;
    - ``Γαβ`` class B (``Γ[p,p,cr_β,an_β]``) and class C
      (``Γ[cr_α,an_α,r,r]``) weight a β- or α-single row by the bra pair's
      other-spin occupancy;
    - ``Γαβ`` class D adds ``sgn_α sgn_β C^*[k_bra] C[k_ket]`` at
      ``[cr_α, an_α, cr_β, an_β]``.

    Every row's bra is owned and its ket is read from the stage holding it, so
    each row is accumulated once; the three blocks share one packed
    ``all_reduce(SUM)``.
    """
    sweep = _Sweep.build(operator, x_local)
    n_orb = sweep.n_orb
    accumulated = torch.zeros(
        (3, n_orb, n_orb, n_orb, n_orb), dtype=sweep.dtype, device=sweep.device
    )
    _accumulate_two_rdm_pair(sweep, accumulated)
    gamma_aa, gamma_ab, gamma_bb = accumulated[0], accumulated[1], accumulated[2]

    GlobalSum(_exchange_group(operator)).all_reduce(accumulated)
    if gamma2_a is not None:
        gamma2_a.add_(gamma_aa)
        gamma_aa = gamma2_a
    if gamma2_b is not None:
        gamma2_b.add_(gamma_bb)
        gamma_bb = gamma2_b
    if gamma2_ab is not None:
        gamma2_ab.add_(gamma_ab)
        gamma_ab = gamma2_ab
    return gamma_aa, gamma_ab, gamma_bb


def _accumulate_two_rdm_pair(sweep: _Sweep, accumulated: Any) -> None:
    """Accumulate one rank's partial ``(Γαα, Γαβ, Γββ)`` into ``accumulated``.

    Args:
        sweep: The sweep geometry of this call.
        accumulated: ``(3, L, L, L, L)`` accumulation target; the partitioned
            path passes a dense packed target with the same block access.
    """
    n_orb = sweep.n_orb
    gamma_aa, gamma_ab, gamma_bb = accumulated[0], accumulated[1], accumulated[2]

    alpha_occ = _occupancy(sweep.alpha_of_pair, sweep.local_index, n_orb, sweep.dtype)
    beta_occ = _occupancy(sweep.beta_of_pair, sweep.local_index, n_orb, sweep.dtype)
    _add_same_spin_class_0(gamma_aa, sweep, sweep.alpha_of_pair)
    _add_same_spin_class_0(gamma_bb, sweep, sweep.beta_of_pair)
    _add_ab_class_a(gamma_ab, sweep)

    expanded_step = max(1, sweep.chunk_size // max(n_orb, 1))
    for blocked_stage, stage_index in sweep.stages():
        for chunk in _table_chunks(sweep, "asingles"):
            rows = _resolve_rows(chunk, sweep.local_index, stage_index, ("cr0", "an0"))
            if rows is None:
                continue
            for part in _row_slices(rows, expanded_step):
                _add_same_spin_class_1(
                    gamma_aa, part, sweep, blocked_stage, alpha_occ, is_alpha=True
                )
                _add_ab_class_c(gamma_ab, part, sweep, blocked_stage, beta_occ)
        for chunk in _table_chunks(sweep, "adoubles"):
            rows = _resolve_rows(
                chunk, sweep.local_index, stage_index, ("cr0", "cr1", "an0", "an1")
            )
            if rows is None:
                continue
            for part in _row_slices(rows, sweep.chunk_size):
                _add_same_spin_class_2(gamma_aa, part, sweep, blocked_stage, is_alpha=True)
        for chunk in _table_chunks(sweep, "bsingles"):
            rows = _resolve_rows(chunk, sweep.local_index, stage_index, ("cr0", "an0"))
            if rows is None:
                continue
            for part in _row_slices(rows, expanded_step):
                _add_same_spin_class_1(
                    gamma_bb, part, sweep, blocked_stage, beta_occ, is_alpha=False
                )
                _add_ab_class_b(gamma_ab, part, sweep, blocked_stage, alpha_occ)
        for chunk in _table_chunks(sweep, "bdoubles"):
            rows = _resolve_rows(
                chunk, sweep.local_index, stage_index, ("cr0", "cr1", "an0", "an1")
            )
            if rows is None:
                continue
            for part in _row_slices(rows, sweep.chunk_size):
                _add_same_spin_class_2(gamma_bb, part, sweep, blocked_stage, is_alpha=False)
        for chunk in _table_chunks(sweep, "ab_doubles"):
            rows = _resolve_rows(
                chunk, sweep.local_index, stage_index, ("cr_a", "an_a", "cr_b", "an_b")
            )
            if rows is None:
                continue
            for part in _row_slices(rows, sweep.chunk_size):
                _add_ab_class_d(gamma_ab, part, sweep, blocked_stage)


# ============================================================================
# Output-partitioned entry points
# ============================================================================


def can_partition_pair_tiled_rdm(
    operator: LocalPairTileOperator,
    *,
    order: int = 2,
    dtype: torch.dtype | None = None,
    max_local_bytes: int | None = None,
) -> bool:
    """Whether a tiled pair RDM can be accumulated output-partitioned.

    The GDB/structural tiled pair sweep processes every bra pair exactly once
    and can therefore fill an output slice exactly like the TPB path; the
    predicate only rejects a malformed operator or a per-rank slice above the
    optional byte bound.

    Args:
        operator: The operator the RDM would be built from.
        order: ``1`` for the 1-RDM (two ``L^2`` blocks), ``2`` for the 2-RDM
            (three ``L^4`` blocks).
        dtype: Output dtype; defaults to the operator's diagonal dtype.
        max_local_bytes: Optional largest acceptable per-rank output slice.

    Returns:
        True when the layout can build the requested RDM partitioned.

    Raises:
        ValueError: ``order`` is neither 1 nor 2.
    """
    return _can_partition_rdm(
        int(operator._n_orb),
        int(operator._exchange.world_size),
        isinstance(operator, LocalPairTileOperator),
        order=order,
        dtype=dtype,
        max_local_bytes=max_local_bytes,
        default_dtype=operator._hii_local.dtype,
    )


def pair_tiled_one_rdm_distributed(
    operator: LocalPairTileOperator,
    x_local: Tensor,
    *,
    replicate: bool = True,
    groups: int = 1,
    max_replicated_bytes: int = DEFAULT_MAX_REPLICATED_BYTES,
    gamma_a: Tensor | None = None,
    gamma_b: Tensor | None = None,
) -> tuple[Tensor, Tensor] | PartitionedRDM:
    """Return ``(γα, γβ)``, replicated or output-partitioned.

    Identical values to :func:`pair_tiled_one_rdm`. The partitioned path splits
    the flattened ``(2, L, L)`` output across the ranks and runs
    ``ceil(world_size / groups)`` sweep passes, each ending in one
    ``all_reduce(SUM)`` of a dense group buffer, so the peak per-rank output
    memory is the local slice plus one group buffer.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        replicate: Return ``(γα, γβ)`` on every rank; ``False`` leaves the
            result partitioned behind a :class:`PartitionedRDM`.
        groups: Owner slices accumulated per pass (see
            :func:`~torch_sci_solver.torch_impl.tiled_postprocess.
            tiled_one_rdm_distributed`).
        max_replicated_bytes: Full-output byte bound below which the existing
            replicated accumulator is used verbatim (bit-identical fallback).
        gamma_a: Optional ``(L, L)`` accumulation buffer (replicated path only).
        gamma_b: Optional ``(L, L)`` accumulation buffer for ``γβ``.

    Returns:
        ``(γα, γβ)`` when ``replicate`` is True, otherwise the packed
        ``(2, L, L)`` :class:`PartitionedRDM`.

    Raises:
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not floating or complex.
    """
    sweep = _Sweep.build(operator, x_local)
    n_orb = sweep.n_orb
    n = 2 * n_orb * n_orb
    group = _exchange_group(operator)
    world_size = int(operator._exchange.world_size)
    if replicate and (
        not can_partition_pair_tiled_rdm(operator, order=1)
        or should_replicate(
            n, sweep.dtype, world_size=world_size, max_replicated_bytes=max_replicated_bytes
        )
    ):
        return pair_tiled_one_rdm(operator, x_local, gamma_a=gamma_a, gamma_b=gamma_b)

    partitioned = _reduce_partitioned(
        2,
        n_orb * n_orb,
        (2, n_orb, n_orb),
        lambda target: _accumulate_one_rdm(sweep, target),
        n=n,
        world_size=world_size,
        rank=int(operator._exchange.rank),
        dtype=sweep.dtype,
        device=sweep.device,
        group=group,
        groups=groups,
    )
    if not replicate:
        return partitioned
    full = partitioned.gather(group=group)
    reduced_a, reduced_b = full[0], full[1]
    if gamma_a is not None:
        gamma_a.add_(reduced_a)
        reduced_a = gamma_a
    if gamma_b is not None:
        gamma_b.add_(reduced_b)
        reduced_b = gamma_b
    return reduced_a, reduced_b


def pair_tiled_two_rdm_distributed(
    operator: LocalPairTileOperator,
    x_local: Tensor,
    *,
    replicate: bool = True,
    groups: int = 1,
    max_replicated_bytes: int = DEFAULT_MAX_REPLICATED_BYTES,
    gamma2_a: Tensor | None = None,
    gamma2_b: Tensor | None = None,
    gamma2_ab: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | PartitionedRDM:
    """Return ``(Γαα, Γαβ, Γββ)``, replicated or output-partitioned.

    Identical values to :func:`pair_tiled_two_rdm`. The partitioned path splits
    the flattened ``(3, L, L, L, L)`` output across the ranks and runs
    ``ceil(world_size / groups)`` sweep passes, each ending in one
    ``all_reduce(SUM)`` of a dense group buffer, so the peak per-rank output
    memory is ``3 L^4 / world_size`` plus one group buffer.

    Args:
        operator: A :class:`LocalPairTileOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        replicate: Return the three blocks on every rank; ``False`` leaves the
            result packed and partitioned behind a :class:`PartitionedRDM` of
            shape ``(3, L, L, L, L)``.
        groups: Owner slices accumulated per pass.
        max_replicated_bytes: Full-output byte bound below which the existing
            replicated accumulator is used verbatim (bit-identical fallback).
        gamma2_a: Optional ``(L, L, L, L)`` buffer receiving ``Γαα``.
        gamma2_b: Optional ``(L, L, L, L)`` buffer receiving ``Γββ``.
        gamma2_ab: Optional ``(L, L, L, L)`` buffer receiving ``Γαβ``.

    Returns:
        ``(Γαα, Γαβ, Γββ)`` when ``replicate`` is True, otherwise the packed
        :class:`PartitionedRDM`.

    Raises:
        ValueError: ``x_local`` does not have one entry per owned pair.
        TypeError: ``x_local`` is not floating or complex.
    """
    sweep = _Sweep.build(operator, x_local)
    n_orb = sweep.n_orb
    n = 3 * n_orb**4
    group = _exchange_group(operator)
    world_size = int(operator._exchange.world_size)
    if replicate and (
        not can_partition_pair_tiled_rdm(operator, order=2)
        or should_replicate(
            n, sweep.dtype, world_size=world_size, max_replicated_bytes=max_replicated_bytes
        )
    ):
        return pair_tiled_two_rdm(
            operator, x_local, gamma2_a=gamma2_a, gamma2_b=gamma2_b, gamma2_ab=gamma2_ab
        )

    partitioned = _reduce_partitioned(
        3,
        n_orb**4,
        (3, n_orb, n_orb, n_orb, n_orb),
        lambda target: _accumulate_two_rdm_pair(sweep, target),
        n=n,
        world_size=world_size,
        rank=int(operator._exchange.rank),
        dtype=sweep.dtype,
        device=sweep.device,
        group=group,
        groups=groups,
    )
    if not replicate:
        return partitioned
    full = partitioned.gather(group=group)
    gamma_aa, gamma_ab, gamma_bb = full[0], full[1], full[2]
    if gamma2_a is not None:
        gamma2_a.add_(gamma_aa)
        gamma_aa = gamma2_a
    if gamma2_b is not None:
        gamma2_b.add_(gamma_bb)
        gamma_bb = gamma2_b
    if gamma2_ab is not None:
        gamma2_ab.add_(gamma_ab)
        gamma_ab = gamma2_ab
    return gamma_aa, gamma_ab, gamma_bb


def _add_same_spin_class_0(
    target: Tensor, sweep: _Sweep, strings: "Tensor | _StructuralPairStrings"
) -> None:
    """Scatter the same-spin ``Γσσ`` class-0 ``|C|^2`` diagonal term.

    Mirrors ``_aa_class_0_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`, over the owned pairs.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        sweep: The sweep geometry of this call.
        strings: Global α or β strings, in either sweep form.
    """
    n_orb = sweep.n_orb
    occ = _occupancy(strings, sweep.local_index, n_orb, sweep.dtype)
    weights = (sweep.x_local.conj() * sweep.x_local).to(sweep.dtype)
    diagonal = (occ.T * weights) @ occ
    off = ~torch.eye(n_orb, dtype=torch.bool, device=sweep.device)
    p_idx, r_idx = torch.where(off)
    flat = target.view(-1)
    flat.index_add_(
        0,
        p_idx * n_orb**3 + p_idx * n_orb**2 + r_idx * n_orb + r_idx,
        diagonal[p_idx, r_idx],
    )
    flat.index_add_(
        0,
        p_idx * n_orb**3 + r_idx * n_orb**2 + r_idx * n_orb + p_idx,
        -diagonal[p_idx, r_idx],
    )


def _add_ab_class_a(target: Tensor, sweep: _Sweep) -> None:
    """Scatter the ``Γαβ`` class-A diagonal term over the owned pairs.

    Mirrors ``_ab_class_a_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        sweep: The sweep geometry of this call.
    """
    n_orb = sweep.n_orb
    weights = (sweep.x_local.conj() * sweep.x_local).to(sweep.dtype).unsqueeze(-1)
    alpha_occ = _occupancy(sweep.alpha_of_pair, sweep.local_index, n_orb, sweep.dtype)
    beta_occ = _occupancy(sweep.beta_of_pair, sweep.local_index, n_orb, sweep.dtype)
    diagonal = (alpha_occ * weights).T @ beta_occ
    p_range = torch.arange(n_orb, dtype=torch.int64, device=sweep.device)
    p_idx, r_idx = torch.meshgrid(p_range, p_range, indexing="ij")
    flat_index = (
        p_idx.reshape(-1) * n_orb**3
        + p_idx.reshape(-1) * n_orb**2
        + r_idx.reshape(-1) * n_orb
        + r_idx.reshape(-1)
    )
    target.view(-1).index_add_(0, flat_index, diagonal.reshape(-1))


def _add_same_spin_class_1(
    target: Tensor,
    rows: _Rows,
    sweep: _Sweep,
    blocked_stage: Tensor,
    occupancy: Tensor,
    *,
    is_alpha: bool,
) -> None:
    """Accumulate one same-spin single-excitation × spectator slice.

    Mirrors ``_aa_class_1_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs` (with ``cr_is_beta`` for the
    ββ block).

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        rows: The slice's resolved rows.
        sweep: The sweep geometry of this call.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
        occupancy: ``(N_local, L)`` occupancy of the owned excited-spin strings.
        is_alpha: ``True`` for the αα block, ``False`` for ββ.
    """
    n_orb = sweep.n_orb
    cr, an = rows.columns
    if is_alpha:
        cr_sp = cr // 2
        an_sp = an // 2
        strings = sweep.alpha_of_pair
    else:
        cr_sp = (cr - 1) // 2
        an_sp = (an - 1) // 2
        strings = sweep.beta_of_pair
    sign = _sign(
        strings,
        rows.global_bra,
        torch.minimum(cr_sp, an_sp),
        torch.maximum(cr_sp, an_sp),
        sweep.dtype,
    )
    weight = sign * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    spectator = occupancy[rows.bra]
    cr_mask = torch.zeros_like(spectator)
    cr_mask.scatter_(1, cr_sp.unsqueeze(1), 1.0)
    contribution = weight.unsqueeze(1) * (spectator * (1.0 - cr_mask))
    cr_idx = cr_sp.unsqueeze(1).expand(-1, n_orb)
    an_idx = an_sp.unsqueeze(1).expand(-1, n_orb)
    x_idx = torch.arange(n_orb, dtype=torch.int64, device=target.device).unsqueeze(0)
    x_idx = x_idx.expand_as(cr_idx)
    flat = target.view(-1)
    flat.index_add_(
        0,
        (cr_idx * n_orb**3 + an_idx * n_orb**2 + x_idx * n_orb + x_idx).reshape(-1),
        contribution.reshape(-1),
    )
    flat.index_add_(
        0,
        (cr_idx * n_orb**3 + x_idx * n_orb**2 + x_idx * n_orb + an_idx).reshape(-1),
        -contribution.reshape(-1),
    )
    flat.index_add_(
        0,
        (x_idx * n_orb**3 + an_idx * n_orb**2 + cr_idx * n_orb + x_idx).reshape(-1),
        -contribution.reshape(-1),
    )
    flat.index_add_(
        0,
        (x_idx * n_orb**3 + x_idx * n_orb**2 + cr_idx * n_orb + an_idx).reshape(-1),
        contribution.reshape(-1),
    )


def _add_same_spin_class_2(
    target: Tensor,
    rows: _Rows,
    sweep: _Sweep,
    blocked_stage: Tensor,
    *,
    is_alpha: bool,
) -> None:
    """Accumulate one same-spin double-excitation slice.

    Mirrors ``_aa_class_2_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`: two range parities on the bra
    string and the ``A > J`` / ``B < I`` disjointness flip.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        rows: The slice's resolved rows.
        sweep: The sweep geometry of this call.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
        is_alpha: ``True`` for the αα block, ``False`` for ββ.
    """
    n_orb = sweep.n_orb
    cr0, cr1, an0, an1 = rows.columns
    if is_alpha:
        cr0_sp = cr0 // 2
        cr1_sp = cr1 // 2
        an0_sp = an0 // 2
        an1_sp = an1 // 2
        strings = sweep.alpha_of_pair
    else:
        cr0_sp = (cr0 - 1) // 2
        cr1_sp = (cr1 - 1) // 2
        an0_sp = (an0 - 1) // 2
        an1_sp = (an1 - 1) // 2
        strings = sweep.beta_of_pair
    first = torch.minimum(cr0_sp, cr1_sp)
    second = torch.maximum(cr0_sp, cr1_sp)
    low = torch.minimum(an0_sp, an1_sp)
    high = torch.maximum(an0_sp, an1_sp)
    sign = _sign(
        strings,
        rows.global_bra,
        torch.minimum(first, low),
        torch.maximum(first, low),
        sweep.dtype,
    )
    sign = _sign(
        strings,
        rows.global_bra,
        torch.minimum(second, high),
        torch.maximum(second, high),
        sweep.dtype,
        sign,
    )
    sign = torch.where((low > second) | (high < first), -sign, sign)
    contribution = sign * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    flat = target.view(-1)
    flat.index_add_(
        0, cr0_sp * n_orb**3 + an0_sp * n_orb**2 + cr1_sp * n_orb + an1_sp, contribution
    )
    flat.index_add_(
        0, cr0_sp * n_orb**3 + an1_sp * n_orb**2 + cr1_sp * n_orb + an0_sp, -contribution
    )
    flat.index_add_(
        0, cr1_sp * n_orb**3 + an0_sp * n_orb**2 + cr0_sp * n_orb + an1_sp, -contribution
    )
    flat.index_add_(
        0, cr1_sp * n_orb**3 + an1_sp * n_orb**2 + cr0_sp * n_orb + an0_sp, contribution
    )


def _add_ab_class_b(
    target: Tensor,
    rows: _Rows,
    sweep: _Sweep,
    blocked_stage: Tensor,
    alpha_occupancy: Tensor,
) -> None:
    """Accumulate one ``Γαβ`` class-B (α-diag × β-single) slice.

    Mirrors ``_ab_class_b_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        rows: The slice's resolved β-single rows.
        sweep: The sweep geometry of this call.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
        alpha_occupancy: ``(N_local, L)`` α occupancy of the owned pairs.
    """
    n_orb = sweep.n_orb
    cr, an = rows.columns
    cr_sp = (cr - 1) // 2
    an_sp = (an - 1) // 2
    sign = _sign(
        sweep.beta_of_pair,
        rows.global_bra,
        torch.minimum(cr_sp, an_sp),
        torch.maximum(cr_sp, an_sp),
        sweep.dtype,
    )
    weight = sign * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    contribution = weight.unsqueeze(1) * alpha_occupancy[rows.bra]
    p_idx = torch.arange(n_orb, dtype=torch.int64, device=target.device).unsqueeze(0)
    p_idx = p_idx.expand_as(contribution)
    cr_idx = cr_sp.unsqueeze(1).expand_as(contribution)
    an_idx = an_sp.unsqueeze(1).expand_as(contribution)
    flat = target.view(-1)
    flat.index_add_(
        0,
        (p_idx * n_orb**3 + p_idx * n_orb**2 + cr_idx * n_orb + an_idx).reshape(-1),
        contribution.reshape(-1),
    )


def _add_ab_class_c(
    target: Tensor,
    rows: _Rows,
    sweep: _Sweep,
    blocked_stage: Tensor,
    beta_occupancy: Tensor,
) -> None:
    """Accumulate one ``Γαβ`` class-C (α-single × β-diag) slice.

    Mirrors ``_ab_class_c_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        rows: The slice's resolved α-single rows.
        sweep: The sweep geometry of this call.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
        beta_occupancy: ``(N_local, L)`` β occupancy of the owned pairs.
    """
    n_orb = sweep.n_orb
    cr, an = rows.columns
    cr_sp = cr // 2
    an_sp = an // 2
    sign = _sign(
        sweep.alpha_of_pair,
        rows.global_bra,
        torch.minimum(cr_sp, an_sp),
        torch.maximum(cr_sp, an_sp),
        sweep.dtype,
    )
    weight = sign * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    contribution = weight.unsqueeze(1) * beta_occupancy[rows.bra]
    r_idx = torch.arange(n_orb, dtype=torch.int64, device=target.device).unsqueeze(0)
    r_idx = r_idx.expand_as(contribution)
    cr_idx = cr_sp.unsqueeze(1).expand_as(contribution)
    an_idx = an_sp.unsqueeze(1).expand_as(contribution)
    flat = target.view(-1)
    flat.index_add_(
        0,
        (cr_idx * n_orb**3 + an_idx * n_orb**2 + r_idx * n_orb + r_idx).reshape(-1),
        contribution.reshape(-1),
    )


def _add_ab_class_d(
    target: Tensor,
    rows: _Rows,
    sweep: _Sweep,
    blocked_stage: Tensor,
) -> None:
    """Accumulate one ``Γαβ`` class-D (α-single × β-single) slice.

    Mirrors ``_ab_class_d_pair`` in
    :mod:`torch_sci_solver.torch_impl.rdm_pairs`.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        rows: The slice's resolved ``ab_doubles`` rows.
        sweep: The sweep geometry of this call.
        blocked_stage: σ-scaled stage buffer holding the ket amplitudes.
    """
    n_orb = sweep.n_orb
    cr_a, an_a, cr_b, an_b = rows.columns
    cr_a_sp = cr_a // 2
    an_a_sp = an_a // 2
    cr_b_sp = (cr_b - 1) // 2
    an_b_sp = (an_b - 1) // 2
    sign_a = _sign(
        sweep.alpha_of_pair,
        rows.global_bra,
        torch.minimum(cr_a_sp, an_a_sp),
        torch.maximum(cr_a_sp, an_a_sp),
        sweep.dtype,
    )
    sign_b = _sign(
        sweep.beta_of_pair,
        rows.global_bra,
        torch.minimum(cr_b_sp, an_b_sp),
        torch.maximum(cr_b_sp, an_b_sp),
        sweep.dtype,
    )
    contribution = sign_a * sign_b * sweep.local_blocked[rows.bra].conj() * blocked_stage[rows.ket]
    flat = target.view(-1)
    flat.index_add_(
        0, cr_a_sp * n_orb**3 + an_a_sp * n_orb**2 + cr_b_sp * n_orb + an_b_sp, contribution
    )
