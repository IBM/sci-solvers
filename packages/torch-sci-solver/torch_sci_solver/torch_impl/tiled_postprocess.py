"""Tiled 1-RDM, 2-RDM and <S^2> expectations for a distributed TPB state vector.

Each rank owns one rectangular tile ``A_i x B_j`` of the ``(M_a, M_b)``
coefficient matrix and holds only its own amplitudes. This module computes the
spin-resolved 1-RDM, the spin-resolved 2-RDM and the ``<S^2>`` expectation from
those tiles without ever materialising the full vector: the bra factor of every
contraction is an owned row or column, and the ket factor is either in the
owned tile or in one of the stage buffers the tile matvec's exchanges already
ship.

Term classes (matching the serial builders in :mod:`torch_sci_solver.torch_impl.rdm`):

- ``γ_α`` / ``γ_β`` diagonal and the ``Γ_αβ`` class-A scalar are local.
- ``γ_α`` off-diagonal reads ``C[ja, ib]`` for every β column, so it consumes
  the α stage buffers (each rank in a row group sums its own column slab).
- ``γ_β`` off-diagonal mirrors that against the β stage buffers.
- The same-spin ``Γ_σσ`` class 0 and ``Γ_αβ`` class A are local; the same-spin
  classes 1-2 and ``Γ_αβ`` classes B/C read one slab buffer (the α buffer sums
  the β spectator slab and vice versa).
- ``Γ_αβ`` class D pairs each α-single with each β-single; its entries are
  joined stage by stage — the ``(0, 0)`` and ``(s_a, 0)`` pairs on the α
  buffers, the ``(0, s_b)`` pairs on the β buffers and the ``(s_a, s_b)`` pairs
  on the corner rectangles (including their row-banded sub-stages) — so the
  global ``n_α × n_β`` product is never formed, only one stage's join.
- The ``<S^2>`` cross term is ``Γ_αβ[p, q, q, p]``, i.e. the matched
  α-single / β-single pairs of class D with ``(cr_β, an_β) = (an_α, cr_α)``.
  Those pairs are split over the same stage pairs as the matvec: ``(s_a, 0)``
  rides the α buffers, ``(0, s_b)`` the β buffers and ``(s_a, s_b)`` the
  corner rectangles.

The exchanges are the ones :class:`LocalTPBOperator` already built for the
matvec (the matvec's needed sets contain every ket the singles and doubles
tables read), so no exchange is rebuilt and no kernel is duplicated. Every rank
accumulates its tile's partial contributions and a single ``all_reduce(SUM)``
per accumulated observable combines them: the
partition is exact because each α-single entry is processed only by the ranks
owning its bra row (one per column slab) and each β-single entry only by the
ranks owning its bra column (one per row slab), while each class-D pair is
processed once by the rank owning both bra indices.

Conventions
-----------
- ``C`` is the interleaved-Wick eigenvector the matvec consumes. The serial
  builders work in spin-block order, so every amplitude is multiplied by
  ``σ = _inter_to_block_signs(adets, bdets, n_orb)`` first. ``σ`` is a function
  of the global ``(ia, ib)`` and is recomputed from the buffer's global index
  lists, so received blocks are consistent with the owned tile.
- Bra factors are conjugated (Hermitian inner product); a no-op for real
  amplitudes. Occupation weights use ``|C|^2``.
- The αβ-swap projector is not used: it maps ``(ia, ib)`` to ``(ib, ia)``,
  which is a different index shift from the RDM's single-excitation kets.

References
----------
The owner-computes TPB decomposition and its staged exchanges are described in
``docs/distributed_state_vector.md``; the RDM indexing follows the serial
chemist-convention builders in ``rdm.py``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import parity_batch
from torch_sci_solver.torch_impl.distributed_davidson import GlobalSum
from torch_sci_solver.torch_impl.distributed_matvec import LocalTPBOperator, _StageTable
from torch_sci_solver.torch_impl.distributed_rdm import (
    DEFAULT_MAX_REPLICATED_BYTES,
    PartitionedRDM,
    _can_partition_rdm,
    _reduce_partitioned,
    should_replicate,
)
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs, _occupancy_bits


@dataclass(frozen=True)
class _SinglesMeta:
    """Contraction metadata of one stage's singles table.

    Attributes:
        bra: ``(n,)`` owned bra index (row for α, column for β).
        to: ``(n,)`` ket position inside the stage buffer.
        cr_sp: ``(n,)`` spatial creation orbital.
        an_sp: ``(n,)`` spatial annihilation orbital.
        sign: ``(n,)`` spin-block parity of the bra string.
        match_key: ``(n,)`` ``<S^2>`` join key. For α this is
            ``an_sp * L + cr_sp`` (the pattern read at ``Γ[p, q, q, p]``); for
            β it is ``cr_sp * L + an_sp``.
    """

    bra: Tensor
    to: Tensor
    cr_sp: Tensor
    an_sp: Tensor
    sign: Tensor
    match_key: Tensor


def tiled_one_rdm(
    operator: LocalTPBOperator,
    x_local: Tensor,
    *,
    gamma_a: Optional[Tensor] = None,
    gamma_b: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Return ``(γ_α, γ_β)`` of a tiled vector, replicated on every rank.

    Args:
        operator: A :class:`LocalTPBOperator` built with a
            :class:`~torch_sci_solver.torch_impl.distributed_davidson.TilePartition`
            (the tile path). Its staged exchanges and singles stage tables are
            reused; they must belong to the same process group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block, in the
            tile's local row-major order.
        gamma_a: Optional ``(L, L)`` accumulation buffer. The reduced ``γ_α``
            is added into it.
        gamma_b: Optional ``(L, L)`` accumulation buffer for ``γ_β``.

    Returns:
        ``(γ_α, γ_β)``, each ``(L, L)`` in the amplitude dtype
        (complex for a complex vector). Entry ``[p, q]`` is
        ``⟨Ψ| a†_{p,σ} a_{q,σ} |Ψ⟩`` in the spin-block convention of
        :func:`torch_sci_solver.torch_impl.rdm.make_one_rdm_torch`.

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
        ValueError: ``x_local`` has the wrong shape.

    Notes
    -----
    The diagonal is ``Σ_{ib} |C[ia, ib]|^2`` over every determinant occupying
    orbital ``p``; the off-diagonal is

    .. math::

        γ_{σ}[p, q] = \\sum_{bra} sgn(bra)\\,
            \\sum_{other} C^*[bra, other]\\, C[ket, other],

    where ``ket`` is the single-excitation partner of ``bra`` and the inner sum
    runs over the other spin's strings. The bra is owned, so the inner sum is
    split across the ranks of the row (α) or column (β) group and restored by
    one ``all_reduce(SUM)``.
    """
    packed, _cross = _accumulate_tiles(operator, x_local, want_rdm=True, want_spin_sq=False)
    assert packed is not None  # requested above
    GlobalSum(operator._alpha_exchange.group).all_reduce(packed)
    reduced_a, reduced_b = packed[0], packed[1]
    if gamma_a is not None:
        gamma_a.add_(reduced_a)
        reduced_a = gamma_a
    if gamma_b is not None:
        gamma_b.add_(reduced_b)
        reduced_b = gamma_b
    return reduced_a, reduced_b


def tiled_spin_squared(
    operator: LocalTPBOperator,
    x_local: Tensor,
    n_a: int,
    n_b: int,
    *,
    cross_term: Optional[Tensor] = None,
) -> Tensor:
    """Return ``⟨Ψ|S^2|Ψ⟩`` of a tiled vector, replicated on every rank.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        n_a: α electron count of the sector.
        n_b: β electron count of the sector.
        cross_term: Optional 0-dim buffer receiving the reduced
            ``Σ_{p,q} Γ_αβ[p, q, q, p]`` cross term (added).

    Returns:
        0-dim real tensor with the ``S(S+1)`` eigenvalue convention of
        :func:`torch_sci_solver.torch_impl.rdm.s_squared_from_rdm`.

    Notes
    -----
    ``S^2 = S_z^2 - S_z + N_α - Σ_{p,q} Γ_αβ[p, q, q, p]``. Only class A
    (local ``|C|^2`` with both orbitals doubly occupied) and the *matched*
    class-D pairs contribute; classes B and C repeat an index pair with
    ``cr != an`` and vanish in the ``p, q, q, p`` pattern:

    .. math::

        \\sum_{p,q} Γ_{αβ}[p,q,q,p] =
            \\sum_{ia,ib} occ_α(ia, p)\\,occ_β(ib, p)\\,|C[ia,ib]|^2
            + \\sum_{matched} sgn_α\\,sgn_β\\,C^*[ia,ib]\\,C[ja,jb],

    with ``(cr_β, an_β) = (an_α, cr_α)``. The matched pairs are joined on a
    per-stage key, so the cost is the number of matches, not the full
    ``n_{αs} × n_{βs}`` product.
    """
    _packed, cross = _accumulate_tiles(operator, x_local, want_rdm=False, want_spin_sq=True)
    assert cross is not None  # requested above
    GlobalSum(operator._alpha_exchange.group).all_reduce(cross)
    if cross_term is not None:
        cross_term.add_(cross)
    sz = (n_a - n_b) / 2.0
    s_sq = cross.new_tensor(sz * sz - sz + n_a) - cross
    return s_sq.real if s_sq.is_complex() else s_sq


def tiled_rdm1_spin_sq(
    operator: LocalTPBOperator,
    x_local: Tensor,
    n_a: int,
    n_b: int,
    *,
    gamma_a: Optional[Tensor] = None,
    gamma_b: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused ``tiled_one_rdm`` + ``tiled_spin_squared`` in one staged sweep.

    The α, β and corner sweeps are performed once, so the exchange traffic is
    one matvec-equivalent instead of one pass per observable. The ``(2, L, L)``
    RDM and the ``<S^2>`` scalar share one packed ``all_reduce(SUM)``.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        n_a: α electron count.
        n_b: β electron count.
        gamma_a: Optional ``(L, L)`` buffer receiving ``γ_α``.
        gamma_b: Optional ``(L, L)`` buffer receiving ``γ_β``.

    Returns:
        ``(γ_α, γ_β, ⟨S^2⟩)`` with the conventions of the two single-purpose
        entry points.
    """
    packed, cross = _accumulate_tiles(operator, x_local, want_rdm=True, want_spin_sq=True)
    assert packed is not None and cross is not None  # requested above
    L = int(operator._n_orb)
    combined = torch.cat([packed.reshape(-1), cross.reshape(1)])
    GlobalSum(operator._alpha_exchange.group).all_reduce(combined)
    packed = combined[: 2 * L * L].view(2, L, L)
    cross = combined[2 * L * L]
    reduced_a, reduced_b = packed[0], packed[1]
    if gamma_a is not None:
        gamma_a.add_(reduced_a)
        reduced_a = gamma_a
    if gamma_b is not None:
        gamma_b.add_(reduced_b)
        reduced_b = gamma_b
    sz = (n_a - n_b) / 2.0
    s_sq = cross.new_tensor(sz * sz - sz + n_a) - cross
    if s_sq.is_complex():
        s_sq = s_sq.real
    return reduced_a, reduced_b, s_sq


def tiled_s2_apply(operator: LocalTPBOperator, x_local: Tensor) -> Tensor:
    """Return the owned tile of ``S^2 |x>`` without a full-vector fallback.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition. Its
            staged exchanges and singles stage tables are reused; they must
            belong to the same process group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block, in the
            tile's local row-major order.

    Returns:
        ``(N_local,)`` owned block of ``S^2 |x>`` in the interleaved-Wick
        convention and the tile's local row-major order, with ``x_local``'s
        dtype. Only this rank's output block is returned; no collective is
        needed, because every bra of the off-diagonal term is owned.

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not floating or complex.

    Notes
    -----
    In a fixed ``(n_α, n_β)`` sector

    .. math::

        S^2 = S_z^2 - S_z + N_\\alpha
            + \\sum_{p,q} a^\\dagger_{p\\alpha} a^\\dagger_{q\\beta}
              a_{q\\alpha} a_{p\\beta}.

    The ``p = q`` part of the double sum is diagonal, ``-N_{dbl}(I)``, so the
    diagonal action is ``n_\\alpha(I) - N_{dbl}(I) + S_z(I)^2 - S_z(I)``. The
    ``p \\neq q`` part swaps an occupied α orbital ``q`` with an empty α
    orbital ``p`` — an α single excitation ``(cr_α, an_α) = (p, q)`` — while
    moving the β electron the other way, ``(cr_β, an_β) = (q, p)``. Between
    interleaved-Wick determinants its matrix element is

    .. math::

        \\langle J|S^2|I\\rangle = -\\sigma_J\\,\\sigma_I\\,s_\\alpha\\,s_\\beta,

    with ``s_σ`` the single-excitation parity the stage tables already carry
    and ``σ`` the interleaved-to-block sign. The matched ``(s_α, s_β)`` stage
    pairs are the ones :func:`tiled_spin_squared` reads: ``(s_a, 0)`` rides
    the α buffers, ``(0, s_b)`` the β buffers and ``(s_a, s_b)`` the corner
    rectangles. Each contribution is scattered into the owned bra tile, and
    the σ conversion is one elementwise multiply on the output tile.
    """
    _validate_tile_input(operator, x_local)

    dtype = x_local.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    L = int(operator._n_orb)
    local_rows, local_cols = int(operator._local_rows), int(operator._local_cols)
    x_tile = x_local.view(local_rows, local_cols)
    if local_rows == 0 or local_cols == 0:
        # An empty slab owns no bra and ships no payload; peers' needed lists
        # cannot reference it, so skipping the exchange cannot strand them.
        return torch.zeros_like(x_local)

    # Interleaved-to-block conversion of the input and the diagonal action.
    sigma_local = _inter_to_block_signs(operator._adets_local, operator._bdets_local, L).to(dtype)
    blocked_local = x_tile * sigma_local
    alpha_bits = _occupancy_bits(operator._adets_local, L).to(real_dtype)
    beta_bits = _occupancy_bits(operator._bdets_local, L).to(real_dtype)
    n_alpha = alpha_bits.sum(dim=1, keepdim=True)  # (local_rows, 1)
    n_beta = beta_bits.sum(dim=1, keepdim=True).T  # (1, local_cols)
    n_dbl = alpha_bits @ beta_bits.T
    sz = 0.5 * (n_alpha - n_beta)
    y_block = (n_alpha - n_dbl + sz * sz - sz).to(dtype) * blocked_local

    matched = operator._alphabeta and bool(operator._beta_singles_stages)
    alpha_singles: list[_StageTable] = operator._singles_stages
    beta_singles: list[_StageTable] = operator._beta_singles_stages
    chunk = int(operator._chunk_size)
    meta_cache: dict[int, _SinglesMeta] = {}

    def meta(table: Optional[_StageTable], *, alpha: bool) -> Optional[_SinglesMeta]:
        """Per-table contraction metadata, computed once per sweep."""
        if table is None or table.to.numel() == 0:
            return None
        entry = meta_cache.get(id(table))
        if entry is None:
            entry = _singles_meta(operator, table, alpha=alpha, real_dtype=real_dtype)
            meta_cache[id(table)] = entry
        return entry

    # --- α stream: the (0, 0) and (s_a, 0) spin-swap pairs ----------------
    for stage, buffer in operator._alpha_exchange.stages(x_tile):
        if not matched:
            continue
        alpha_meta = meta(alpha_singles[stage], alpha=True)
        beta_meta = meta(beta_singles[0], alpha=False)
        if alpha_meta is None or beta_meta is None:
            continue
        blocked = blocked_local if stage == 0 else _scale_alpha_stage(operator, stage, buffer)
        _s2_scatter(y_block, alpha_meta, beta_meta, blocked, chunk)

    # --- β stream: the (0, s_b) spin-swap pairs; (0, 0) ran above ---------
    for stage, buffer in operator._beta_exchange.stages(x_tile):
        if not matched or stage == 0:
            continue
        alpha_meta = meta(alpha_singles[0], alpha=True)
        beta_meta = meta(beta_singles[stage], alpha=False)
        if alpha_meta is None or beta_meta is None:
            continue
        blocked = _scale_beta_stage(operator, stage, buffer)
        _s2_scatter(y_block, alpha_meta, beta_meta, blocked, chunk)

    # --- Corner stream: the (s_a, s_b) pairs with both kets remote --------
    corner = operator._corner_exchange
    pairs = corner.pairs()
    if pairs:
        alpha_tables = operator._corner_alpha_tables
        for stage, buffer in corner.stages(x_tile):
            if not matched or not alpha_tables:
                continue
            s_a, s_b, row_lo, row_hi = pairs[stage]
            alpha_meta = meta(alpha_tables[stage], alpha=True)
            beta_meta = meta(beta_singles[s_b], alpha=False)
            if alpha_meta is None or beta_meta is None:
                continue
            rows = operator._alpha_exchange.stage_indices(s_a)[row_lo:row_hi]
            cols = operator._beta_exchange.stage_indices(s_b)
            blocked = _scale_corner(operator, rows, cols, buffer)
            _s2_scatter(y_block, alpha_meta, beta_meta, blocked, chunk)

    return (y_block * sigma_local).reshape(-1)


def tiled_s2_diagonal(operator: LocalTPBOperator) -> Tensor:
    """Return the owned tile's ``S^2`` diagonal for the penalty preconditioner.

    In a fixed ``(n_α, n_β)`` sector the diagonal action of ``S^2`` on a
    determinant is ``n_α(I) - N_dbl(I) + S_z(I)^2 - S_z(I)`` (see
    :func:`tiled_s2_apply`); it carries no excitation signs, so the
    interleaved-Wick and spin-block diagonals coincide.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition.

    Returns:
        ``(N_local,)`` owned diagonal in the tile's local row-major order, in
        the real dtype of ``operator``'s diagonal block (``operator._hii_local``).

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
    """
    if operator._tile is None:
        raise NotImplementedError(
            "tiled post-processing requires a TilePartition operator; the 1-D "
            "owner-computes path has a different exchange layout"
        )
    dtype = operator._hii_local.dtype
    if int(operator.N_local) == 0:
        return torch.empty(0, dtype=dtype, device=operator._hii_local.device)
    L = int(operator._n_orb)
    alpha_bits = _occupancy_bits(operator._adets_local, L).to(torch.float64)
    beta_bits = _occupancy_bits(operator._bdets_local, L).to(torch.float64)
    n_alpha = alpha_bits.sum(dim=1, keepdim=True)  # (local_rows, 1)
    n_beta = beta_bits.sum(dim=1, keepdim=True).T  # (1, local_cols)
    n_dbl = alpha_bits @ beta_bits.T
    sz = 0.5 * (n_alpha - n_beta)
    diagonal = (n_alpha - n_dbl + sz * sz - sz).reshape(-1)
    return diagonal.to(dtype)


# ============================================================================
# Spin-resolved 2-RDM
# ============================================================================


def tiled_two_rdm(
    operator: LocalTPBOperator,
    x_local: Tensor,
    *,
    gamma2_a: Optional[Tensor] = None,
    gamma2_b: Optional[Tensor] = None,
    gamma2_ab: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(Γ_αα, Γ_αβ, Γ_ββ)`` of a tiled vector, replicated on every rank.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition. Its
            staged exchanges and singles/doubles stage tables are reused; they
            must belong to the same process group the caller uses.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block, in the
            tile's local row-major order.
        gamma2_a: Optional ``(L, L, L, L)`` accumulation buffer receiving
            ``Γ_αα`` (added).
        gamma2_b: Optional ``(L, L, L, L)`` accumulation buffer for ``Γ_ββ``.
        gamma2_ab: Optional ``(L, L, L, L)`` accumulation buffer for ``Γ_αβ``.

    Returns:
        ``(Γ_αα, Γ_αβ, Γ_ββ)``, each ``(L, L, L, L)`` in the amplitude dtype
        (complex for a complex vector), in the conventions of
        :func:`torch_sci_solver.torch_impl.rdm.make_two_rdm_torch`: entry
        ``[p, q, r, s]`` of ``Γ_αβ`` is ``⟨Ψ| a†_{pα} a†_{rβ} a_{sβ} a_{qα} |Ψ⟩``
        and the same-spin blocks are the analogous α-only / β-only
        expectations. ``energy_from_rdm(γ_α, γ_β, Γ_αα, Γ_αβ, Γ_ββ, integrals)``
        therefore equals the Rayleigh quotient.

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
        ValueError: ``x_local`` has the wrong shape.

    Notes
    -----
    The serial builders decompose ``Γ`` into diagonal classes, single-excitation
    classes and ``Γ_αβ`` class D:

    - class 0 (``Γ_σσ``) and class A (``Γ_αβ``) are ``|C|^2`` weighted
      occupations of the owned tile;
    - the same-spin class 1 (single excitation × spectator orbital) and class 2
      (double excitation) are

      .. math::

          w_k = sgn_k \\sum_{other} C^*[bra_k, other]\\, C[ket_k, other],

      summed over the other spin's owned slab, and scattered to the four
      Slater-Condon slots;
    - ``Γ_αβ`` class B/C are the same weights further weighted by the owned
      occupancy of the α (B) or β (C) spectator slab;
    - ``Γ_αβ`` class D joins, per stage pair, every α entry with every β entry
      of the paired stage:

      .. math::

          Γ_{αβ}[cr_α, an_α, cr_β, an_β] +=
              sgn_α\\,sgn_β\\,C^*[ia, ib]\\,C[ja, jb].

    The bra factor ``C[ia, ib]`` is always an owned tile entry; the ket factor
    ``C[ja, jb]`` comes from the local tile, one slab buffer or the corner
    rectangle of that stage pair. Each rank accumulates its partial blocks into
    replicated ``(L, L, L, L)`` tensors and one packed ``all_reduce(SUM)``
    restores the full result; only the tile, the stage buffers and the
    accumulated blocks are ever resident.
    """
    accumulated = _accumulate_two_rdm(operator, x_local)
    GlobalSum(operator._alpha_exchange.group).all_reduce(accumulated)
    gamma_aa, gamma_ab, gamma_bb = accumulated[0], accumulated[1], accumulated[2]
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


# ============================================================================
# Output-partitioned entry points
# ============================================================================


def can_partition_tiled_rdm(
    operator: LocalTPBOperator,
    *,
    order: int = 2,
    dtype: Optional[torch.dtype] = None,
    max_local_bytes: Optional[int] = None,
) -> bool:
    """Whether a tiled full RDM can be accumulated output-partitioned.

    Preflight predicate for a caller choosing between the replicated and the
    partitioned entry points. The partitioned path needs a tile-path operator
    (the 1-D owner-computes layout has different stage tables) and a positive
    orbital count; an optional byte bound rejects layouts whose per-rank output
    slice is too large.

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
        int(operator._world_size),
        operator._tile is not None,
        order=order,
        dtype=dtype,
        max_local_bytes=max_local_bytes,
        default_dtype=operator._hii_local.dtype,
    )


def tiled_one_rdm_distributed(
    operator: LocalTPBOperator,
    x_local: Tensor,
    *,
    replicate: bool = True,
    groups: int = 1,
    max_replicated_bytes: int = DEFAULT_MAX_REPLICATED_BYTES,
    gamma_a: Optional[Tensor] = None,
    gamma_b: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor] | PartitionedRDM:
    """Return ``(γ_α, γ_β)``, replicated or output-partitioned.

    Identical values to :func:`tiled_one_rdm`. The partitioned path splits the
    flattened ``(2, L, L)`` output across the ranks and runs
    ``ceil(world_size / groups)`` sweep passes, each ending in one
    ``all_reduce(SUM)`` of a dense group buffer. Peak per-rank output memory is
    the local ``2 L^2 / world_size`` slice (plus one group buffer), instead of
    the replicated ``2 L^2``.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        replicate: Return the reduced ``(γ_α, γ_β)`` on every rank. ``False``
            leaves the result partitioned behind a :class:`PartitionedRDM`.
        groups: Owner slices accumulated per pass. ``1`` minimizes memory at
            ``world_size`` passes; ``world_size`` is one pass with a fully
            replicated buffer.
        max_replicated_bytes: Full-output byte bound below which the existing
            replicated accumulator is used verbatim (bit-identical fallback).
        gamma_a: Optional ``(L, L)`` accumulation buffer (replicated path only).
        gamma_b: Optional ``(L, L)`` accumulation buffer for ``γ_β``.

    Returns:
        ``(γ_α, γ_β)`` when ``replicate`` is True, otherwise the packed
        ``(2, L, L)`` :class:`PartitionedRDM`.

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not floating or complex.
    """
    _validate_tile_input(operator, x_local)
    L = int(operator._n_orb)
    n = 2 * L * L
    world_size = int(operator._world_size)
    dtype = x_local.dtype
    if replicate and (
        not can_partition_tiled_rdm(operator, order=1)
        or should_replicate(
            n, dtype, world_size=world_size, max_replicated_bytes=max_replicated_bytes
        )
    ):
        return tiled_one_rdm(operator, x_local, gamma_a=gamma_a, gamma_b=gamma_b)

    group = operator._alpha_exchange.group
    partitioned = _reduce_partitioned(
        2,
        L * L,
        (2, L, L),
        lambda target: _accumulate_tiles(
            operator, x_local, want_rdm=True, want_spin_sq=False, packed=target
        ),
        n=n,
        world_size=world_size,
        rank=int(operator._rank),
        dtype=dtype,
        device=x_local.device,
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


def tiled_two_rdm_distributed(
    operator: LocalTPBOperator,
    x_local: Tensor,
    *,
    replicate: bool = True,
    groups: int = 1,
    max_replicated_bytes: int = DEFAULT_MAX_REPLICATED_BYTES,
    gamma2_a: Optional[Tensor] = None,
    gamma2_b: Optional[Tensor] = None,
    gamma2_ab: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor] | PartitionedRDM:
    """Return ``(Γ_αα, Γ_αβ, Γ_ββ)``, replicated or output-partitioned.

    Identical values to :func:`tiled_two_rdm`. The partitioned path splits the
    flattened ``(3, L, L, L, L)`` output across the ranks and runs
    ``ceil(world_size / groups)`` sweep passes, each ending in one
    ``all_reduce(SUM)`` of a dense group buffer. Peak per-rank output memory is
    ``3 L^4 / world_size`` plus one group buffer, instead of ``3 L^4``.

    Args:
        operator: A :class:`LocalTPBOperator` built with a tile partition.
        x_local: ``(N_local,)`` owned interleaved-Wick amplitude block.
        replicate: Return the three blocks on every rank. ``False`` leaves the
            result packed and partitioned behind a :class:`PartitionedRDM` of
            shape ``(3, L, L, L, L)``.
        groups: Owner slices accumulated per pass, as in
            :func:`tiled_one_rdm_distributed`.
        max_replicated_bytes: Full-output byte bound below which the existing
            replicated accumulator is used verbatim (bit-identical fallback).
        gamma2_a: Optional ``(L, L, L, L)`` buffer receiving ``Γ_αα``.
        gamma2_b: Optional ``(L, L, L, L)`` buffer receiving ``Γ_ββ``.
        gamma2_ab: Optional ``(L, L, L, L)`` buffer receiving ``Γ_αβ``.

    Returns:
        ``(Γ_αα, Γ_αβ, Γ_ββ)`` when ``replicate`` is True, otherwise the packed
        :class:`PartitionedRDM`.

    Raises:
        NotImplementedError: ``operator`` was built without a tile partition.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not floating or complex.
    """
    _validate_tile_input(operator, x_local)
    L = int(operator._n_orb)
    n = 3 * L**4
    world_size = int(operator._world_size)
    dtype = x_local.dtype
    if replicate and (
        not can_partition_tiled_rdm(operator, order=2)
        or should_replicate(
            n, dtype, world_size=world_size, max_replicated_bytes=max_replicated_bytes
        )
    ):
        return tiled_two_rdm(
            operator,
            x_local,
            gamma2_a=gamma2_a,
            gamma2_b=gamma2_b,
            gamma2_ab=gamma2_ab,
        )

    group = operator._alpha_exchange.group
    partitioned = _reduce_partitioned(
        3,
        L**4,
        (3, L, L, L, L),
        lambda target: _accumulate_two_rdm(operator, x_local, accumulated=target),
        n=n,
        world_size=world_size,
        rank=int(operator._rank),
        dtype=dtype,
        device=x_local.device,
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


# ============================================================================
# Shared staged sweep
# ============================================================================


def _validate_tile_input(operator: LocalTPBOperator, x_local: Tensor) -> None:
    """Reject a non-tile operator or a badly shaped amplitude block.

    Args:
        operator: The tile operator to check.
        x_local: The owned amplitude block.

    Raises:
        NotImplementedError: ``operator`` is not on the tile path.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not a floating or complex tensor.
    """
    if operator._tile is None:
        raise NotImplementedError(
            "tiled post-processing requires a TilePartition operator; the 1-D "
            "owner-computes path has a different exchange layout"
        )
    if x_local.shape != (operator.N_local,):
        raise ValueError(f"x_local shape {tuple(x_local.shape)} != ({operator.N_local},)")
    if not x_local.is_floating_point() and not x_local.is_complex():
        raise TypeError(f"x_local must be floating or complex, got {x_local.dtype}")


def _accumulate_tiles(
    operator: LocalTPBOperator,
    x_local: Tensor,
    *,
    want_rdm: bool,
    want_spin_sq: bool,
    packed: Any = None,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Accumulate this rank's RDM and <S^2> partials with one staged sweep.

    Args:
        operator: The tile operator supplying the exchanges and stage tables.
        x_local: ``(N_local,)`` owned amplitude block.
        want_rdm: Accumulate the ``(2, L, L)`` packed ``(γ_α, γ_β)``.
        want_spin_sq: Accumulate the ``Γ_αβ[p, q, q, p]`` cross term.
        packed: Optional caller-provided ``(2, L, L)`` accumulation target. The
            partitioned path passes a
            :class:`~torch_sci_solver.torch_impl.distributed_rdm.PackedTarget`
            here so the same sweep writes through its output filter.

    Returns:
        ``(packed, cross)``; either is ``None`` when not requested.

    Raises:
        NotImplementedError: ``operator`` is not on the tile path.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not a floating or complex tensor.
    """
    _validate_tile_input(operator, x_local)

    dtype = x_local.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    device = x_local.device
    L = int(operator._n_orb)
    local_rows, local_cols = int(operator._local_rows), int(operator._local_cols)
    x_tile = x_local.view(local_rows, local_cols)

    if packed is None and want_rdm:
        packed = torch.zeros((2, L, L), dtype=dtype, device=device)
    cross = torch.zeros((), dtype=dtype, device=device) if want_spin_sq else None

    # Local terms: the diagonal occupations and the class-A doubly-occupied
    # weighted sum. |C|^2 is real even for a complex vector.
    amplitude_sq = x_tile.abs().pow(2)
    alpha_bits = _occupancy_bits(operator._adets_local, L).to(real_dtype)
    beta_bits = _occupancy_bits(operator._bdets_local, L).to(real_dtype)
    weight_a = amplitude_sq.sum(dim=1)
    weight_b = amplitude_sq.sum(dim=0)
    if packed is not None:
        occ_a = alpha_bits.T @ weight_a
        occ_b = beta_bits.T @ weight_b
        diagonal = torch.arange(L, device=device, dtype=torch.int64) * (L + 1)
        packed[0].view(-1).index_add_(0, diagonal, occ_a.to(dtype))
        packed[1].view(-1).index_add_(0, diagonal, occ_b.to(dtype))
    if cross is not None:
        diag_ab = alpha_bits.T @ amplitude_sq @ beta_bits
        cross = cross + diag_ab.diagonal().sum()

    if local_rows == 0 or local_cols == 0:
        # An empty slab owns no bra and ships no payload; peers' needed lists
        # cannot reference it, so skipping the exchange cannot strand them.
        return packed, cross

    sigma_local = _inter_to_block_signs(operator._adets_local, operator._bdets_local, L).to(dtype)
    blocked_local = x_tile * sigma_local
    meta_cache: dict[int, _SinglesMeta] = {}

    def meta(table: Optional[_StageTable], *, alpha: bool) -> Optional[_SinglesMeta]:
        """Per-table contraction metadata, computed once per sweep."""
        if table is None or table.to.numel() == 0:
            return None
        key = id(table)
        entry = meta_cache.get(key)
        if entry is None:
            entry = _singles_meta(operator, table, alpha=alpha, real_dtype=real_dtype)
            meta_cache[key] = entry
        return entry

    # --- α stream: γ_α off-diagonal and the (s_a, 0) class-D pairs ----------
    alpha_singles: list[_StageTable] = operator._singles_stages
    beta_singles: list[_StageTable] = operator._beta_singles_stages
    matched = want_spin_sq and operator._alphabeta and bool(beta_singles)
    for stage, buffer in operator._alpha_exchange.stages(x_tile):
        if packed is None and not matched:
            continue
        blocked = blocked_local if stage == 0 else _scale_alpha_stage(operator, stage, buffer)
        table = alpha_singles[stage] if alpha_singles else None
        alpha_meta = meta(table, alpha=True)
        if alpha_meta is not None and packed is not None:
            _add_alpha_rdm(packed[0], alpha_meta, blocked_local, blocked, L, operator._chunk_size)
        if matched and alpha_meta is not None:
            beta_meta = meta(beta_singles[0], alpha=False)
            if beta_meta is not None:
                assert cross is not None  # matched implies want_spin_sq
                cross = cross + _matched_sum(
                    alpha_meta, beta_meta, blocked_local, blocked, operator._chunk_size
                )

    # --- β stream: γ_β off-diagonal and the (0, s_b) class-D pairs ----------
    for stage, buffer in operator._beta_exchange.stages(x_tile):
        if packed is None and not matched:
            continue
        blocked = blocked_local if stage == 0 else _scale_beta_stage(operator, stage, buffer)
        table = beta_singles[stage] if beta_singles else None
        beta_meta = meta(table, alpha=False)
        if beta_meta is not None and packed is not None:
            _add_beta_rdm(packed[1], beta_meta, blocked_local, blocked, L, operator._chunk_size)
        if matched and stage > 0 and alpha_singles:
            alpha_meta = meta(alpha_singles[0], alpha=True)
            if alpha_meta is not None and beta_meta is not None:
                assert cross is not None  # matched implies want_spin_sq
                cross = cross + _matched_sum(
                    alpha_meta, beta_meta, blocked_local, blocked, operator._chunk_size
                )

    # --- Corner stream: the (s_a, s_b) class-D pairs with both kets remote --
    corner = operator._corner_exchange
    pairs = corner.pairs()
    if pairs and local_rows and local_cols:
        alpha_tables = operator._corner_alpha_tables
        for stage, buffer in corner.stages(x_tile):
            if not matched or not alpha_tables:
                continue
            s_a, s_b, row_lo, row_hi = pairs[stage]
            alpha_meta = meta(alpha_tables[stage], alpha=True)
            beta_meta = meta(beta_singles[s_b], alpha=False)
            if alpha_meta is None or beta_meta is None:
                continue
            rows = operator._alpha_exchange.stage_indices(s_a)[row_lo:row_hi]
            cols = operator._beta_exchange.stage_indices(s_b)
            blocked = _scale_corner(operator, rows, cols, buffer)
            assert cross is not None  # matched implies want_spin_sq
            cross = cross + _matched_sum(
                alpha_meta, beta_meta, blocked_local, blocked, operator._chunk_size
            )
    return packed, cross


def _singles_meta(
    operator: LocalTPBOperator,
    table: _StageTable,
    *,
    alpha: bool,
    real_dtype: torch.dtype,
) -> _SinglesMeta:
    """Build the contraction metadata of one stage's singles table.

    Args:
        operator: The tile operator owning the local string blocks.
        table: The stage table (``bra`` local, ``to`` buffer-local).
        alpha: ``True`` for an α table, ``False`` for a β table.
        real_dtype: Real dtype of the amplitude, used for the ±1 signs.

    Returns:
        The table's :class:`_SinglesMeta`.
    """
    device = table.bra.device
    L = int(operator._n_orb)
    bra = table.bra.to(torch.int64)
    to = table.to.to(torch.int64)
    cr = table.columns[0].to(torch.int64)
    an = table.columns[1].to(torch.int64)
    if alpha:
        cr_sp = cr // 2
        an_sp = an // 2
        bra_strings = operator._adets_local[bra]
        match_key = an_sp * L + cr_sp
    else:
        cr_sp = (cr - 1) // 2
        an_sp = (an - 1) // 2
        bra_strings = operator._bdets_local[bra]
        match_key = cr_sp * L + an_sp
    start = torch.minimum(cr_sp, an_sp)
    end = torch.maximum(cr_sp, an_sp)
    sign = torch.ones(cr_sp.shape, dtype=real_dtype, device=device)
    sign = parity_batch(bra_strings, None, start, end, sign)
    return _SinglesMeta(bra=bra, to=to, cr_sp=cr_sp, an_sp=an_sp, sign=sign, match_key=match_key)


def _scale_alpha_stage(operator: LocalTPBOperator, stage: int, buffer: Tensor) -> Tensor:
    """Spin-block scale an α stage buffer (remote rows, owned columns).

    Args:
        operator: The tile operator.
        stage: Ket-owner stage (non-zero).
        buffer: ``(rows, local_cols)`` raw received block.

    Returns:
        The block multiplied by ``σ`` of its global ``(ia, ib)`` indices.
    """
    rows = operator._alpha_exchange.stage_indices(stage)
    sigma = _inter_to_block_signs(
        operator._adets_t[rows.to(operator._adets_t.device)],
        operator._bdets_local,
        operator._n_orb,
    )
    return buffer * sigma.to(buffer.dtype)


def _scale_beta_stage(operator: LocalTPBOperator, stage: int, buffer: Tensor) -> Tensor:
    """Spin-block scale a β stage buffer (owned rows, remote columns).

    Args:
        operator: The tile operator.
        stage: Ket-owner stage (non-zero).
        buffer: ``(local_rows, cols)`` raw received block.

    Returns:
        The block multiplied by ``σ`` of its global ``(ia, ib)`` indices.
    """
    cols = operator._beta_exchange.stage_indices(stage)
    sigma = _inter_to_block_signs(
        operator._adets_local,
        operator._bdets_t[cols.to(operator._bdets_t.device)],
        operator._n_orb,
    )
    return buffer * sigma.to(buffer.dtype)


def _scale_corner(operator: LocalTPBOperator, rows: Tensor, cols: Tensor, buffer: Tensor) -> Tensor:
    """Spin-block scale a corner buffer (both index sets remote).

    Args:
        operator: The tile operator.
        rows: Global α-row indices of the corner rectangle.
        cols: Global β-column indices of the corner rectangle.
        buffer: ``(rows, cols)`` raw received rectangle.

    Returns:
        The rectangle multiplied by ``σ`` of its global ``(ia, ib)`` indices.
    """
    sigma = _inter_to_block_signs(
        operator._adets_t[rows.to(operator._adets_t.device)],
        operator._bdets_t[cols.to(operator._bdets_t.device)],
        operator._n_orb,
    )
    return buffer * sigma.to(buffer.dtype)


def _add_alpha_rdm(
    gamma_a: Tensor,
    entry: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one α stage's off-diagonal ``γ_α`` contributions.

    Args:
        gamma_a: ``(L, L)`` accumulation tensor.
        entry: The stage's α metadata.
        blocked_local: σ-scaled owned tile ``(local_rows, local_cols)``.
        blocked_buffer: σ-scaled stage buffer ``(rows, local_cols)``.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.
    """
    n_entries = int(entry.bra.numel())
    width = max(int(blocked_local.shape[1]), 1)
    step = max(1, int(chunk_entries) // width)
    flat = gamma_a.view(-1)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        product = (blocked_local[bra].conj() * blocked_buffer[ket]).sum(dim=1)
        index = entry.cr_sp[lo:hi] * L + entry.an_sp[lo:hi]
        flat.index_add_(0, index, entry.sign[lo:hi] * product)


def _add_beta_rdm(
    gamma_b: Tensor,
    entry: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one β stage's off-diagonal ``γ_β`` contributions.

    Args:
        gamma_b: ``(L, L)`` accumulation tensor.
        entry: The stage's β metadata.
        blocked_local: σ-scaled owned tile ``(local_rows, local_cols)``.
        blocked_buffer: σ-scaled stage buffer ``(local_rows, cols)``.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.
    """
    n_entries = int(entry.bra.numel())
    height = max(int(blocked_local.shape[0]), 1)
    step = max(1, int(chunk_entries) // height)
    flat = gamma_b.view(-1)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        product = (blocked_local[:, bra].conj() * blocked_buffer[:, ket]).sum(dim=0)
        index = entry.cr_sp[lo:hi] * L + entry.an_sp[lo:hi]
        flat.index_add_(0, index, entry.sign[lo:hi] * product)


def _matched_sum(
    alpha: _SinglesMeta,
    beta: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    chunk_pairs: int,
) -> Tensor:
    """Sum one stage pair's matched class-D contributions to ``Γ_αβ[p,q,q,p]``.

    Args:
        alpha: α metadata of the stage pair.
        beta: β metadata of the stage pair.
        blocked_local: σ-scaled owned tile; the bra factor ``C[ia, ib]``.
        blocked_buffer: σ-scaled stage buffer; the ket factor ``C[ja, jb]``.
        chunk_pairs: Workspace bound, in matched pairs.

    Returns:
        0-dim tensor holding this stage pair's cross-term contribution.
    """
    alpha_index, beta_index = _matched_pairs(alpha.match_key, beta.match_key, blocked_local.device)
    if alpha_index.numel() == 0:
        return blocked_local.new_zeros(())
    total = blocked_local.new_zeros(())
    step = max(1, int(chunk_pairs))
    for lo in range(0, int(alpha_index.numel()), step):
        hi = min(lo + step, int(alpha_index.numel()))
        a = alpha_index[lo:hi]
        b = beta_index[lo:hi]
        bra_value = blocked_local[alpha.bra[a], beta.bra[b]]
        ket_value = blocked_buffer[alpha.to[a], beta.to[b]]
        total = total + (alpha.sign[a] * beta.sign[b] * bra_value.conj() * ket_value).sum()
    return total


def _s2_scatter(
    target: Tensor,
    alpha: _SinglesMeta,
    beta: _SinglesMeta,
    blocked_buffer: Tensor,
    chunk_pairs: int,
) -> None:
    """Scatter one stage pair's spin-swap contributions into the owned tile.

    Args:
        target: ``(local_rows, local_cols)`` block-convention output tile.
        alpha: α metadata of the stage pair.
        beta: β metadata of the paired stage.
        blocked_buffer: σ-scaled stage buffer holding the ket amplitudes.
        chunk_pairs: Workspace bound, in matched pairs.

    Notes
    -----
    Each matched pair contributes ``-sgn_α sgn_β C[ja, jb]`` to
    ``target[ia, ib]``, the matrix element of the spin-swap part of ``S^2``.
    The bra ``(ia, ib)`` is owned, so the scatter is local and no collective
    is needed; the sort/search of :func:`_matched_pairs` keeps the work at the
    number of matches, not the full ``n_{αs} × n_{βs}`` product.
    """
    alpha_index, beta_index = _matched_pairs(alpha.match_key, beta.match_key, blocked_buffer.device)
    if alpha_index.numel() == 0:
        return
    local_cols = int(target.shape[1])
    flat = target.view(-1)
    step = max(1, int(chunk_pairs))
    for lo in range(0, int(alpha_index.numel()), step):
        hi = min(lo + step, int(alpha_index.numel()))
        a = alpha_index[lo:hi]
        b = beta_index[lo:hi]
        ket_value = blocked_buffer[alpha.to[a], beta.to[b]]
        index = alpha.bra[a] * local_cols + beta.bra[b]
        flat.index_add_(0, index, -alpha.sign[a] * beta.sign[b] * ket_value)


def _matched_pairs(key_a: Tensor, key_b: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """Join two per-entry keys, returning the matched index arrays.

    Each α entry contributes one pair per β entry with the same key, so the
    result is the exact sparse set of matched class-D pairs; entries with no
    partner are dropped. The sort/search runs on the host because the keys are
    small integer payloads, and only the matched indices are moved back.

    Args:
        key_a: ``(n_a,)`` α join keys.
        key_b: ``(n_b,)`` β join keys.
        device: Device of the returned index tensors.

    Returns:
        ``(alpha_index, beta_index)``, both ``(n_pairs,)`` int64 tensors.
    """
    if key_a.numel() == 0 or key_b.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    keys_a = key_a.detach().cpu().numpy()
    keys_b = key_b.detach().cpu().numpy()
    order = np.argsort(keys_b, kind="stable")
    sorted_b = keys_b[order]
    lo = np.searchsorted(sorted_b, keys_a, side="left")
    hi = np.searchsorted(sorted_b, keys_a, side="right")
    counts = hi - lo
    keep = counts > 0
    if not bool(keep.any()):
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty
    lengths = counts[keep]
    alpha_index = np.repeat(np.flatnonzero(keep), lengths)
    starts = np.repeat(lo[keep], lengths)
    group_offset = np.repeat(np.cumsum(lengths) - lengths, lengths)
    within = np.arange(alpha_index.size, dtype=np.int64) - group_offset
    beta_index = order[starts + within]
    return (
        torch.from_numpy(alpha_index).to(device),
        torch.from_numpy(beta_index).to(device),
    )


# ============================================================================
# Tiled 2-RDM accumulation
# ============================================================================


@dataclass(frozen=True)
class _DoublesMeta:
    """Contraction metadata of one stage's doubles table.

    Attributes:
        bra: ``(n,)`` owned bra index (row for α, column for β).
        to: ``(n,)`` ket position inside the stage buffer.
        cr0: ``(n,)`` first spatial creation orbital.
        an0: ``(n,)`` first spatial annihilation orbital.
        cr1: ``(n,)`` second spatial creation orbital.
        an1: ``(n,)`` second spatial annihilation orbital.
        sign: ``(n,)`` canonical two-electron spin-block parity of the bra
            string, as in the serial ``_aa_class_2`` / ``_bb_class_2``.
    """

    bra: Tensor
    to: Tensor
    cr0: Tensor
    an0: Tensor
    cr1: Tensor
    an1: Tensor
    sign: Tensor


def _accumulate_two_rdm(
    operator: LocalTPBOperator, x_local: Tensor, *, accumulated: Any = None
) -> Tensor:
    """Accumulate this rank's ``(Γ_αα, Γ_αβ, Γ_ββ)`` partials.

    The sweep visits the same three streams as the tile matvec — α stages,
    β stages, corner stages — and reuses the exchanges, the stage tables and
    the σ scaling of the 1-RDM sweep. Every term is accumulated into one
    ``(3, L, L, L, L)`` tensor so the caller can reduce all three blocks with a
    single collective.

    Args:
        operator: The tile operator supplying the exchanges and stage tables.
        x_local: ``(N_local,)`` owned amplitude block.
        accumulated: Optional caller-provided ``(3, L, L, L, L)`` accumulation
            target. The partitioned path passes a
            :class:`~torch_sci_solver.torch_impl.distributed_rdm.PackedTarget`
            so the same sweep writes through its output filter.

    Returns:
        ``(3, L, L, L, L)`` tensor of this rank's unsynchronised blocks
        (``Γ_αα``, ``Γ_αβ``, ``Γ_ββ``).

    Raises:
        NotImplementedError: ``operator`` is not on the tile path.
        ValueError: ``x_local`` has the wrong shape.
        TypeError: ``x_local`` is not a floating or complex tensor.
    """
    _validate_tile_input(operator, x_local)
    dtype = x_local.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    device = x_local.device
    L = int(operator._n_orb)
    local_rows, local_cols = int(operator._local_rows), int(operator._local_cols)
    chunk = int(operator._chunk_size)
    x_tile = x_local.view(local_rows, local_cols)

    if accumulated is None:
        accumulated = torch.zeros((3, L, L, L, L), dtype=dtype, device=device)
    gamma_aa, gamma_ab, gamma_bb = accumulated[0], accumulated[1], accumulated[2]

    # Local classes: the same-spin class 0 and the αβ class A. |C|^2 is real
    # even for a complex vector; the occupancy sums stay real and are cast once
    # for the (possibly complex) scatters.
    amplitude_sq = x_tile.abs().pow(2)
    alpha_bits = _occupancy_bits(operator._adets_local, L).to(real_dtype)
    beta_bits = _occupancy_bits(operator._bdets_local, L).to(real_dtype)
    weight_a = amplitude_sq.sum(dim=1)
    weight_b = amplitude_sq.sum(dim=0)
    _add_class_0(gamma_aa, ((alpha_bits.T * weight_a) @ alpha_bits).to(dtype), L)
    _add_class_0(gamma_bb, ((beta_bits.T * weight_b) @ beta_bits).to(dtype), L)
    _add_ab_class_a(gamma_ab, (alpha_bits.T @ amplitude_sq @ beta_bits).to(dtype), L)

    if local_rows == 0 or local_cols == 0:
        # An empty slab owns no bra and ships no payload; peers' needed lists
        # cannot reference it, so skipping the exchanges cannot strand them.
        return accumulated

    sigma_local = _inter_to_block_signs(operator._adets_local, operator._bdets_local, L).to(dtype)
    blocked_local = x_tile * sigma_local
    singles_cache: dict[int, _SinglesMeta] = {}
    doubles_cache: dict[int, _DoublesMeta] = {}

    def singles_meta(table: Optional[_StageTable], *, alpha: bool) -> Optional[_SinglesMeta]:
        """Per-table singles metadata, computed once per sweep."""
        if table is None or table.to.numel() == 0:
            return None
        key = id(table)
        entry = singles_cache.get(key)
        if entry is None:
            entry = _singles_meta(operator, table, alpha=alpha, real_dtype=real_dtype)
            singles_cache[key] = entry
        return entry

    def doubles_meta(table: Optional[_StageTable], *, alpha: bool) -> Optional[_DoublesMeta]:
        """Per-table doubles metadata, computed once per sweep."""
        if table is None or table.to.numel() == 0:
            return None
        key = id(table)
        entry = doubles_cache.get(key)
        if entry is None:
            entry = _doubles_meta(operator, table, alpha=alpha, real_dtype=real_dtype)
            doubles_cache[key] = entry
        return entry

    alpha_singles: list[_StageTable] = operator._singles_stages
    alpha_doubles: list[_StageTable] = operator._doubles_stages
    beta_singles: list[_StageTable] = operator._beta_singles_stages
    beta_doubles: list[_StageTable] = operator._beta_doubles_stages
    # The stage-0 tables pair with every remote stage of the other spin.
    alpha_singles_zero = singles_meta(alpha_singles[0], alpha=True) if alpha_singles else None
    beta_singles_zero = singles_meta(beta_singles[0], alpha=False) if beta_singles else None

    # --- α stream: Γaa classes 1-2, Γab class C, and the (0, 0)/(s_a, 0)
    # class-D pairs whose α ket rides the α buffer and β ket is local.
    for stage, buffer in operator._alpha_exchange.stages(x_tile):
        blocked = blocked_local if stage == 0 else _scale_alpha_stage(operator, stage, buffer)
        entry = singles_meta(alpha_singles[stage], alpha=True) if alpha_singles else None
        if entry is not None:
            _add_same_spin_class_1(
                gamma_aa,
                entry,
                blocked_local,
                blocked,
                alpha_bits,
                alpha=True,
                L=L,
                chunk_entries=chunk,
            )
            _add_ab_class_c(gamma_ab, entry, blocked_local, blocked, beta_bits, L, chunk)
            if beta_singles_zero is not None:
                _add_ab_class_d(
                    gamma_ab, entry, beta_singles_zero, blocked_local, blocked, L, chunk
                )
        if alpha_doubles:
            doubles_entry = doubles_meta(alpha_doubles[stage], alpha=True)
            if doubles_entry is not None:
                _add_same_spin_class_2(
                    gamma_aa,
                    doubles_entry,
                    blocked_local,
                    blocked,
                    alpha=True,
                    L=L,
                    chunk_entries=chunk,
                )

    # --- β stream: Γbb classes 1-2, Γab class B, and the (0, s_b) class-D
    # pairs whose β ket rides the β buffer; (0, 0) was handled above.
    for stage, buffer in operator._beta_exchange.stages(x_tile):
        blocked = blocked_local if stage == 0 else _scale_beta_stage(operator, stage, buffer)
        entry = singles_meta(beta_singles[stage], alpha=False) if beta_singles else None
        if entry is not None:
            _add_same_spin_class_1(
                gamma_bb,
                entry,
                blocked_local,
                blocked,
                beta_bits,
                alpha=False,
                L=L,
                chunk_entries=chunk,
            )
            _add_ab_class_b(gamma_ab, entry, blocked_local, blocked, alpha_bits, L, chunk)
            if stage > 0 and alpha_singles_zero is not None:
                _add_ab_class_d(
                    gamma_ab, alpha_singles_zero, entry, blocked_local, blocked, L, chunk
                )
        if beta_doubles:
            doubles_entry = doubles_meta(beta_doubles[stage], alpha=False)
            if doubles_entry is not None:
                _add_same_spin_class_2(
                    gamma_bb,
                    doubles_entry,
                    blocked_local,
                    blocked,
                    alpha=False,
                    L=L,
                    chunk_entries=chunk,
                )

    # --- Corner stream: the (s_a, s_b) class-D pairs with both kets remote.
    # A sub-stage may be a row band of its rectangle, so the α table is the
    # banded, rebased one the matvec uses.
    pairs = operator._corner_exchange.pairs()
    alpha_tables = operator._corner_alpha_tables
    if pairs and alpha_tables and beta_singles:
        for stage, buffer in operator._corner_exchange.stages(x_tile):
            s_a, s_b, row_lo, row_hi = pairs[stage]
            entry_alpha = singles_meta(alpha_tables[stage], alpha=True)
            entry_beta = singles_meta(beta_singles[s_b], alpha=False)
            if entry_alpha is None or entry_beta is None:
                continue
            rows = operator._alpha_exchange.stage_indices(s_a)[row_lo:row_hi]
            cols = operator._beta_exchange.stage_indices(s_b)
            blocked = _scale_corner(operator, rows, cols, buffer)
            _add_ab_class_d(gamma_ab, entry_alpha, entry_beta, blocked_local, blocked, L, chunk)
    return accumulated


def _doubles_meta(
    operator: LocalTPBOperator,
    table: _StageTable,
    *,
    alpha: bool,
    real_dtype: torch.dtype,
) -> _DoublesMeta:
    """Build the contraction metadata of one stage's doubles table.

    The sign is the serial ``_aa_class_2`` / ``_bb_class_2`` canonical
    two-electron Slater-Condon parity: two range parities on the bra string
    followed by the ``A > J`` / ``B < I`` disjointness flip.

    Args:
        operator: The tile operator owning the local string blocks.
        table: The doubles stage table (``bra`` local, ``to`` buffer-local).
        alpha: ``True`` for an α table, ``False`` for a β table.
        real_dtype: Real dtype of the amplitude, used for the ±1 signs.

    Returns:
        The table's :class:`_DoublesMeta`.
    """
    device = table.bra.device
    bra = table.bra.to(torch.int64)
    to = table.to.to(torch.int64)
    if alpha:
        columns = tuple(column.to(torch.int64) // 2 for column in table.columns)
        bra_strings = operator._adets_local[bra]
    else:
        columns = tuple((column.to(torch.int64) - 1) // 2 for column in table.columns)
        bra_strings = operator._bdets_local[bra]
    cr0, cr1, an0, an1 = columns
    first = torch.minimum(cr0, cr1)
    second = torch.maximum(cr0, cr1)
    low = torch.minimum(an0, an1)
    high = torch.maximum(an0, an1)
    sign = torch.ones(cr0.shape, dtype=real_dtype, device=device)
    sign = parity_batch(
        bra_strings, None, torch.minimum(first, low), torch.maximum(first, low), sign
    )
    sign = parity_batch(
        bra_strings, None, torch.minimum(second, high), torch.maximum(second, high), sign
    )
    skip = (low > second) | (high < first)
    sign = torch.where(skip, -sign, sign)
    return _DoublesMeta(bra=bra, to=to, cr0=cr0, an0=an0, cr1=cr1, an1=an1, sign=sign)


def _add_class_0(target: Tensor, diag: Tensor, L: int) -> None:
    """Scatter the same-spin ``Γ_σσ`` class-0 diagonal term.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        diag: ``(L, L)`` ``Σ_bra w_bra occ(ia, p) occ(ia, r)`` partial.
        L: Spatial orbital count.

    Notes
    -----
    ``Γ[p, p, r, r] += diag[p, r]`` and ``Γ[p, q, q, p] -= diag[p, q]`` for
    ``p != r``; the ``(p, p, p, p)`` slot vanishes by Pauli.
    """
    off = ~torch.eye(L, dtype=torch.bool, device=diag.device)
    p_idx, r_idx = torch.where(off)
    flat = target.view(-1)
    flat.index_add_(0, p_idx * L**3 + p_idx * L**2 + r_idx * L + r_idx, diag[p_idx, r_idx])
    flat.index_add_(0, p_idx * L**3 + r_idx * L**2 + r_idx * L + p_idx, -diag[p_idx, r_idx])


def _add_ab_class_a(target: Tensor, diag: Tensor, L: int) -> None:
    """Scatter the ``Γ_αβ`` class-A diagonal term ``Γ[p, p, r, r] = diag[p, r]``.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        diag: ``(L, L)`` ``Σ_bra |C|^2 occ_α(ia, p) occ_β(ib, r)`` partial.
        L: Spatial orbital count.
    """
    orbital = torch.arange(L, dtype=torch.int64, device=diag.device)
    p_idx, r_idx = torch.meshgrid(orbital, orbital, indexing="ij")
    flat_index = (
        p_idx.reshape(-1) * L**3
        + p_idx.reshape(-1) * L**2
        + r_idx.reshape(-1) * L
        + r_idx.reshape(-1)
    )
    target.view(-1).index_add_(0, flat_index, diag.reshape(-1))


def _add_same_spin_class_1(
    target: Tensor,
    entry: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    bits: Tensor,
    *,
    alpha: bool,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one stage's same-spin single-excitation × spectator terms.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        entry: The stage's singles metadata.
        blocked_local: σ-scaled owned tile ``(local_rows, local_cols)``; the
            bra factor and, for stage 0, the ket factor.
        blocked_buffer: σ-scaled stage buffer (α: ``(rows, local_cols)``,
            β: ``(local_rows, cols)``); the ket factor.
        bits: ``(local_strings, L)`` occupancy of the owned bra strings, α or β.
        alpha: ``True`` to sum over the owned β slab, ``False`` over the α slab.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.

    Notes
    -----
    Per entry ``w = sgn Σ_other C^*[bra] C[ket]`` is scattered to
    ``(cr, an, x, x) += w``, ``(cr, x, x, an) -= w``,
    ``(x, an, cr, x) -= w`` and ``(x, x, cr, an) += w`` for every spectator
    orbital ``x`` occupied in the bra string other than ``cr``.
    """
    n_entries = int(entry.bra.numel())
    if n_entries == 0:
        return
    width = int(blocked_local.shape[1] if alpha else blocked_local.shape[0])
    step = max(1, int(chunk_entries) // max(width, 1))
    flat = target.view(-1)
    orbital = torch.arange(L, dtype=torch.int64, device=target.device)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        if alpha:
            weight = (blocked_local[bra].conj() * blocked_buffer[ket]).sum(dim=1)
        else:
            weight = (blocked_local[:, bra].conj() * blocked_buffer[:, ket]).sum(dim=0)
        weight = entry.sign[lo:hi] * weight
        cr = entry.cr_sp[lo:hi]
        an = entry.an_sp[lo:hi]
        bra_occ = bits[bra]
        cr_mask = torch.zeros_like(bra_occ)
        cr_mask.scatter_(1, cr.unsqueeze(1), 1.0)
        contrib = weight.unsqueeze(1) * (bra_occ * (1.0 - cr_mask))
        cr_idx = cr.unsqueeze(1).expand(-1, L)
        an_idx = an.unsqueeze(1).expand(-1, L)
        x_idx = orbital.unsqueeze(0).expand(hi - lo, L)
        flat.index_add_(
            0,
            (cr_idx * L**3 + an_idx * L**2 + x_idx * L + x_idx).reshape(-1),
            contrib.reshape(-1),
        )
        flat.index_add_(
            0,
            (cr_idx * L**3 + x_idx * L**2 + x_idx * L + an_idx).reshape(-1),
            -contrib.reshape(-1),
        )
        flat.index_add_(
            0,
            (x_idx * L**3 + an_idx * L**2 + cr_idx * L + x_idx).reshape(-1),
            -contrib.reshape(-1),
        )
        flat.index_add_(
            0,
            (x_idx * L**3 + x_idx * L**2 + cr_idx * L + an_idx).reshape(-1),
            contrib.reshape(-1),
        )


def _add_same_spin_class_2(
    target: Tensor,
    entry: _DoublesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    *,
    alpha: bool,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one stage's same-spin double-excitation terms.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        entry: The stage's doubles metadata.
        blocked_local: σ-scaled owned tile; the bra factor.
        blocked_buffer: σ-scaled stage buffer; the ket factor.
        alpha: ``True`` to sum over the owned β slab, ``False`` over the α slab.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.

    Notes
    -----
    Per entry ``w = sgn Σ_other C^*[bra] C[ket]`` is scattered to
    ``(cr0, an0, cr1, an1) += w``, ``(cr0, an1, cr1, an0) -= w``,
    ``(cr1, an0, cr0, an1) -= w`` and ``(cr1, an1, cr0, an0) += w``.
    """
    n_entries = int(entry.bra.numel())
    if n_entries == 0:
        return
    width = int(blocked_local.shape[1] if alpha else blocked_local.shape[0])
    step = max(1, int(chunk_entries) // max(width, 1))
    flat = target.view(-1)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        if alpha:
            weight = (blocked_local[bra].conj() * blocked_buffer[ket]).sum(dim=1)
        else:
            weight = (blocked_local[:, bra].conj() * blocked_buffer[:, ket]).sum(dim=0)
        weight = entry.sign[lo:hi] * weight
        cr0 = entry.cr0[lo:hi]
        an0 = entry.an0[lo:hi]
        cr1 = entry.cr1[lo:hi]
        an1 = entry.an1[lo:hi]
        flat.index_add_(0, cr0 * L**3 + an0 * L**2 + cr1 * L + an1, weight)
        flat.index_add_(0, cr0 * L**3 + an1 * L**2 + cr1 * L + an0, -weight)
        flat.index_add_(0, cr1 * L**3 + an0 * L**2 + cr0 * L + an1, -weight)
        flat.index_add_(0, cr1 * L**3 + an1 * L**2 + cr0 * L + an0, weight)


def _add_ab_class_b(
    target: Tensor,
    entry: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    alpha_bits: Tensor,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one β stage's ``Γ_αβ`` class-B (α-diag × β-single) terms.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        entry: The stage's β singles metadata.
        blocked_local: σ-scaled owned tile; the bra factor.
        blocked_buffer: σ-scaled β stage buffer; the ket factor.
        alpha_bits: ``(local_rows, L)`` occupancy of the owned α strings.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.

    Notes
    -----
    ``Γ[p, p, cr_β, an_β] += sgn_β Σ_ia occ_α(ia, p) C^*[ia, ib] C[ia, jb]``;
    each rank sums its own α row slab and the all-reduce restores the full sum.
    """
    n_entries = int(entry.bra.numel())
    if n_entries == 0:
        return
    height = int(blocked_local.shape[0])
    step = max(1, int(chunk_entries) // max(height, 1))
    flat = target.view(-1)
    occupancy = alpha_bits.to(target.dtype)
    orbital = torch.arange(L, dtype=torch.int64, device=target.device)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        product = blocked_local[:, bra].conj() * blocked_buffer[:, ket]
        value = entry.sign[lo:hi].unsqueeze(0) * (occupancy.T @ product)
        p_idx = orbital.unsqueeze(1).expand(-1, hi - lo).reshape(-1)
        cr_idx = entry.cr_sp[lo:hi].unsqueeze(0).expand(L, -1).reshape(-1)
        an_idx = entry.an_sp[lo:hi].unsqueeze(0).expand(L, -1).reshape(-1)
        flat.index_add_(0, p_idx * L**3 + p_idx * L**2 + cr_idx * L + an_idx, value.reshape(-1))


def _add_ab_class_c(
    target: Tensor,
    entry: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    beta_bits: Tensor,
    L: int,
    chunk_entries: int,
) -> None:
    """Accumulate one α stage's ``Γ_αβ`` class-C (α-single × β-diag) terms.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        entry: The stage's α singles metadata.
        blocked_local: σ-scaled owned tile; the bra factor.
        blocked_buffer: σ-scaled α stage buffer; the ket factor.
        beta_bits: ``(local_cols, L)`` occupancy of the owned β strings.
        L: Spatial orbital count.
        chunk_entries: Workspace bound, in lifted entries.

    Notes
    -----
    ``Γ[cr_α, an_α, r, r] += sgn_α Σ_ib occ_β(ib, r) C^*[ia, ib] C[ja, ib]``;
    each rank sums its own β column slab and the all-reduce restores the sum.
    """
    n_entries = int(entry.bra.numel())
    if n_entries == 0:
        return
    width = int(blocked_local.shape[1])
    step = max(1, int(chunk_entries) // max(width, 1))
    flat = target.view(-1)
    occupancy = beta_bits.to(target.dtype)
    orbital = torch.arange(L, dtype=torch.int64, device=target.device)
    for lo in range(0, n_entries, step):
        hi = min(lo + step, n_entries)
        bra = entry.bra[lo:hi]
        ket = entry.to[lo:hi]
        product = blocked_local[bra].conj() * blocked_buffer[ket]
        value = entry.sign[lo:hi].unsqueeze(1) * (product @ occupancy)
        r_idx = orbital.unsqueeze(0).expand(hi - lo, -1).reshape(-1)
        cr_idx = entry.cr_sp[lo:hi].unsqueeze(1).expand(-1, L).reshape(-1)
        an_idx = entry.an_sp[lo:hi].unsqueeze(1).expand(-1, L).reshape(-1)
        flat.index_add_(0, cr_idx * L**3 + an_idx * L**2 + r_idx * L + r_idx, value.reshape(-1))


def _add_ab_class_d(
    target: Tensor,
    alpha: _SinglesMeta,
    beta: _SinglesMeta,
    blocked_local: Tensor,
    blocked_buffer: Tensor,
    L: int,
    chunk_pairs: int,
) -> None:
    """Accumulate one stage pair's ``Γ_αβ`` class-D (α-single × β-single) terms.

    Args:
        target: ``(L, L, L, L)`` accumulation tensor.
        alpha: α metadata of the stage pair.
        beta: β metadata of the paired stage.
        blocked_local: σ-scaled owned tile; the bra factor ``C[ia, ib]``.
        blocked_buffer: σ-scaled buffer of the stage pair; the ket factor
            ``C[ja, jb]``.
        L: Spatial orbital count.
        chunk_pairs: Workspace bound, in joined pairs.

    Notes
    -----
    The join is per stage pair: every α entry of the stage is combined with
    every β entry of the paired stage, so the ``(0, 0)``, ``(s_a, 0)``,
    ``(0, s_b)`` and ``(s_a, s_b)`` pairs are each processed by one buffer and
    the global ``n_α × n_β`` product is never formed. Each pair contributes
    ``sgn_α sgn_β C^*[ia, ib] C[ja, jb]`` to
    ``Γ[cr_α, an_α, cr_β, an_β]``.
    """
    n_alpha = int(alpha.bra.numel())
    n_beta = int(beta.bra.numel())
    if n_alpha == 0 or n_beta == 0:
        return
    step = max(1, int(chunk_pairs) // max(n_beta, 1))
    flat = target.view(-1)
    for lo in range(0, n_alpha, step):
        hi = min(lo + step, n_alpha)
        bra_value = blocked_local[alpha.bra[lo:hi].unsqueeze(1), beta.bra.unsqueeze(0)]
        ket_value = blocked_buffer[alpha.to[lo:hi].unsqueeze(1), beta.to.unsqueeze(0)]
        sign = alpha.sign[lo:hi].unsqueeze(1) * beta.sign.unsqueeze(0)
        contrib = sign * bra_value.conj() * ket_value
        index = (
            alpha.cr_sp[lo:hi].unsqueeze(1) * L**3
            + alpha.an_sp[lo:hi].unsqueeze(1) * L**2
            + beta.cr_sp.unsqueeze(0) * L
            + beta.an_sp.unsqueeze(0)
        )
        flat.index_add_(0, index.reshape(-1), contrib.reshape(-1))
