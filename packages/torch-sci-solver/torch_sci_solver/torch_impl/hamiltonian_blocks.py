"""Disconnected blocks of a selected-CI Hamiltonian.

``H`` couples two determinants only when they differ by at most a double
excitation, so a basis whose CI strings form clusters further apart is block
diagonal, and Davidson never leaves the block its start vector lies in.
Seeding at the lowest diagonal entry does not settle which block holds the
ground state: a block whose lowest diagonal entry is higher can still hold a
lower eigenvalue when its couplings are stronger. This module finds the blocks
(the connected components of the excitation graph) and the lowest diagonal
entry of each, so that the solver can run one solve per block.

Notes
-----
A product (TPB) basis ``A × B`` has the blocks ``C_α × C_β``, where ``C_α`` and
``C_β`` range over the components of the per-spin string graphs, whose edges
join strings a single or a double excitation apart (Hamming distance 2 or 4).
Every coupling keeps α inside its string component and β inside its own (an
αβ double is a single in each spin), and two determinants of one
``C_α × C_β`` are joined by moving α first, then β. When the α and β strings
are the same set, the spin flip ``(α, β) → (β, α)`` commutes with ``H`` and
maps ``C_p × C_q`` onto ``C_q × C_p``, so the two have one spectrum and are
merged into one block (the αβ-swap projector couples them anyway). A pair
(GDB) basis is not a product, so its blocks come from union-find over the pair
excitation tables. Both are blocks of the sparsity pattern: matrix elements
that vanish by symmetry can split a block further, which no pattern-level test
sees.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.helpers import _as_string_limbs, _pairwise_diff

# Elements of one pairwise Hamming-distance chunk in :func:`string_forest` (each
# temporary of the distance pass is this many 8-byte words).
_DIFF_CHUNK_ELEMENTS = 1 << 22


def _compress(parent: Tensor) -> Tensor:
    """Point every node of a union-find forest straight at its root."""
    while True:
        grand = parent[parent]
        if torch.equal(grand, parent):
            return parent
        parent = grand


def union_edges(parent: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Merge the components that the edges ``(u[i], v[i])`` join.

    Args:
        parent: Compressed union-find forest: every node points at its root,
            the smallest node of its component.
        u, v: Edge end points.

    Returns:
        The merged forest, compressed the same way.
    """
    u = u.to(device=parent.device, dtype=torch.int64)
    v = v.to(device=parent.device, dtype=torch.int64)
    while u.numel():
        root_u, root_v = parent[u], parent[v]
        cross = root_u != root_v
        if not bool(cross.any()):
            break
        u, v, root_u, root_v = u[cross], v[cross], root_u[cross], root_v[cross]
        # Hook every larger root under the smallest root an edge offers it.
        parent = parent.scatter_reduce(
            0, torch.maximum(root_u, root_v), torch.minimum(root_u, root_v), reduce="amin"
        )
        parent = _compress(parent)
    return parent


def edge_forest(n: int, edges: Iterable[tuple[Tensor, Tensor]], device: Any = "cpu") -> Tensor:
    """Union-find forest of ``n`` nodes joined by streamed edge chunks.

    Args:
        n: Number of nodes.
        edges: ``(u, v)`` end-point chunks, e.g. ``(k_bra, k_ket)`` of an
            excitation table.
        device: Device the forest lives on.

    Returns:
        ``(n,)`` int64 forest; ``forest[i]`` is the smallest node of ``i``'s
        component. The stream is left unread once every node is in one
        component, which no further edge can change.
    """
    parent = torch.arange(n, dtype=torch.int64, device=device)
    for u, v in edges:
        parent = union_edges(parent, u, v)
        if not bool(parent.any()):
            break
    return parent


def string_forest(strings: Any, n_orb: int, *, rows: Optional[tuple[int, int]] = None) -> Tensor:
    """Union-find forest of CI strings at most a double excitation apart.

    Args:
        strings: The strings, in any form :func:`_as_string_limbs` accepts.
            Equal strings at two positions (a block-major structural array)
            are joined too.
        n_orb: Number of spatial orbitals (the valid bit count).
        rows: Only the edges leaving strings ``[start, stop)`` are merged, so
            that ranks can share the ``O(M^2)`` distance work; ``None`` merges
            all of them.

    Returns:
        ``(M,)`` int64 CPU forest (see :func:`edge_forest`).
    """
    limbs = _as_string_limbs(strings, n_orb)
    m = int(limbs.shape[0])
    start, stop = (0, m) if rows is None else rows
    chunk = max(1, _DIFF_CHUNK_ELEMENTS // max(1, m * int(limbs.shape[1])))
    parent = torch.arange(m, dtype=torch.int64)
    for c0 in range(start, stop, chunk):
        diff = _pairwise_diff(limbs[c0 : min(c0 + chunk, stop)], n_orb, kets_limbs=limbs)
        # Equal-popcount strings differ in an even number of bits: 0, 2 (a
        # single) or 4 (a double).
        ia, ja = np.nonzero(diff <= 4)
        parent = union_edges(parent, torch.from_numpy(ia + c0), torch.from_numpy(ja))
        if not bool(parent.any()):
            break  # one component: the remaining rows cannot split it
    return parent


def merge_rank_forests(parent: Tensor, context: Any) -> Tensor:
    """Join forests built from disjoint shares of the edges into the global one.

    Args:
        parent: This rank's forest, built from its share of the edges.
        context: Distributed context whose ranks hold the shares (``None`` or
            a single rank returns ``parent``).

    Returns:
        The global forest, identical on every rank, on the context's device.

    Notes
    -----
    Every label stays a node of its global component and only ever decreases.
    A pass gives each node the smallest label of its component on every rank
    (an all-reduce MIN) and compresses. When a pass changes nothing, every
    per-rank component carries one label, so every global component does.
    """
    if context is None or context.world_size <= 1:
        return parent
    reduce_min = all_reduce_fn(context, "min")
    parent = parent.to(context.device)
    labels = _compress(reduce_min(parent.clone()))
    while True:
        lowest = torch.full_like(labels, labels.numel()).scatter_reduce(
            0, parent, labels, reduce="amin"
        )
        merged = _compress(reduce_min(lowest[parent].contiguous()))
        if torch.equal(merged, labels):
            return labels
        labels = merged


def product_block_labels(
    forest_a: Tensor,
    forest_b: Tensor,
    *,
    symmetric: bool,
    index: Optional[Tensor] = None,
    device: Any = "cpu",
) -> tuple[Tensor, int, Tensor]:
    """Block of each determinant of a product basis ``A × B``.

    Args:
        forest_a, forest_b: :func:`string_forest` of the α and β strings.
        symmetric: The α and β strings are the same set; ``C_p × C_q`` and
            ``C_q × C_p`` are then one block (see the module notes).
        index: Flat determinant indices ``i * M_b + j`` to label; ``None``
            labels the whole basis in row-major order.
        device: Device of the returned labels.

    Returns:
        ``(labels, n_blocks, trivial)``: int64 block ids on ``device``, the
        size of the id space, and a CPU bool per block id that is True when
        every determinant in the block is uncoupled.
    """
    roots_a, comp_a = torch.unique(forest_a, return_inverse=True)
    roots_b, comp_b = torch.unique(forest_b, return_inverse=True)
    n_a, n_b = int(roots_a.numel()), int(roots_b.numel())
    single_a = torch.bincount(comp_a, minlength=n_a) == 1
    single_b = torch.bincount(comp_b, minlength=n_b) == 1
    comp_a, comp_b = comp_a.to(device), comp_b.to(device)
    if index is None:
        p, q = comp_a[:, None], comp_b[None, :]
    else:
        index = index.to(device=device, dtype=torch.int64)
        m_b = int(forest_b.numel())
        p, q = comp_a[torch.div(index, m_b, rounding_mode="floor")], comp_b[index % m_b]
    if symmetric:
        p, q = torch.minimum(p, q), torch.maximum(p, q)
    labels = (p * n_b + q).reshape(-1)
    trivial = (single_a[:, None] & single_b[None, :]).reshape(-1)
    return labels, n_a * n_b, trivial


def graph_block_labels(
    forest: Tensor,
    *,
    swap: Optional[Tensor] = None,
    index: Optional[Tensor] = None,
) -> tuple[Tensor, int, Tensor]:
    """Block of each node of an explicit basis (pair list or spinless determinants).

    Args:
        forest: :func:`edge_forest` of the whole basis.
        swap: Optional ``(n,)`` αβ-swap permutation; a component and its
            mirror then form one block, as the αβ-swap projector couples them.
        index: Node indices to label; ``None`` labels every node.

    Returns:
        ``(labels, n_blocks, trivial)`` as in :func:`product_block_labels`,
        with the labels on the forest's device.
    """
    isolated = torch.bincount(forest, minlength=forest.numel())[forest] == 1
    roots = forest if swap is None else torch.minimum(forest, forest[swap.to(forest.device)])
    uniq, labels = torch.unique(roots, return_inverse=True)
    n_blocks = int(uniq.numel())
    trivial = torch.ones(n_blocks, dtype=torch.int64, device=forest.device).scatter_reduce(
        0, labels, isolated.to(torch.int64), reduce="amin"
    )
    if index is not None:
        labels = labels[index.to(device=labels.device, dtype=torch.int64)]
    return labels, n_blocks, trivial.bool().cpu()


@dataclass(frozen=True)
class DiagonalBlocks:
    """The blocks of ``H`` over one rank's entries, with their lowest diagonal entries.

    Attributes:
        labels: ``(N_local,)`` block id of each owned entry, on the device of
            the diagonal.
        minima: ``(n_blocks,)`` CPU float64 lowest diagonal entry of each block
            (``inf`` for an id no entry carries).
        seeds: ``(n_blocks,)`` CPU int64 global index of that entry.
        trivial: ``(n_blocks,)`` CPU bool; True when every entry of the block is
            uncoupled, so its eigenvalues are its diagonal entries.
        sizes: ``(n_blocks,)`` CPU int64 number of entries of each block, over
            all ranks.
        reduce_sum: In-place SUM all-reduce over the ranks.
        string_forests: For a product basis, the α and β :func:`string_forest`,
            so that a block ``C_p × C_q`` can be rebuilt from its strings.
    """

    labels: Tensor
    minima: Tensor
    seeds: Tensor
    trivial: Tensor
    sizes: Tensor
    reduce_sum: Callable[[Tensor], Tensor]
    string_forests: Optional[tuple[Tensor, Tensor]] = None

    @property
    def n_nonempty(self) -> int:
        """Number of block ids that some entry carries."""
        return int(torch.isfinite(self.minima).sum())

    def weights(self, vector: Tensor) -> Tensor:
        """``(n_blocks,)`` CPU squared norm of ``vector`` (this rank's entries) per block."""
        weight = torch.zeros(self.minima.numel(), dtype=torch.float64, device=vector.device)
        weight.scatter_add_(0, self.labels, vector.detach().abs().pow(2).to(torch.float64))
        return self.reduce_sum(weight).cpu()


def all_reduce_fn(context: Any, op: str) -> Callable[[Tensor], Tensor]:
    """In-place ``"min"`` or ``"sum"`` all-reduce over ``context``'s group (none on one rank)."""
    if context is None or context.world_size <= 1:
        return lambda tensor: tensor
    import torch.distributed as dist

    reduce_op = dist.ReduceOp.MIN if op == "min" else dist.ReduceOp.SUM

    def reduce(tensor: Tensor) -> Tensor:
        """All-reduce ``tensor`` in place and return it."""
        dist.all_reduce(tensor, op=reduce_op, group=context.group)
        return tensor

    return reduce


def diagonal_blocks(
    diagonal: Tensor,
    labels: Tensor,
    n_blocks: int,
    trivial: Tensor,
    *,
    index: Optional[Tensor] = None,
    context: Any = None,
) -> Optional[DiagonalBlocks]:
    """Find each block's lowest diagonal entry, globally.

    Args:
        diagonal: ``(N_local,)`` diagonal of the operator Davidson sees (spin
            penalty included); real, or complex with a zero imaginary part.
        labels: ``(N_local,)`` block id of each entry, on ``diagonal``'s device.
        n_blocks: Size of the block-id space.
        trivial: ``(n_blocks,)`` bool, see :class:`DiagonalBlocks`.
        index: Global index of each entry; ``None`` when the entries are the
            whole basis in order.
        context: Distributed context of a rank-local diagonal, or ``None``.

    Returns:
        The blocks, or ``None`` when every entry lies in one block. Every rank
        runs the same all-reduces, so all ranks agree on the result.
    """
    values = (diagonal.real if diagonal.is_complex() else diagonal).to(torch.float64)
    device = values.device
    labels = labels.to(device)
    reduce_min = all_reduce_fn(context, "min")
    minima = torch.full((n_blocks,), math.inf, dtype=torch.float64, device=device)
    minima = reduce_min(minima.scatter_reduce(0, labels, values, reduce="amin"))
    position = (
        torch.arange(values.numel(), device=device)
        if index is None
        else index.to(device=device, dtype=torch.int64)
    )
    hit = values == minima[labels]
    seeds = torch.full((n_blocks,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=device)
    seeds = reduce_min(seeds.scatter_reduce(0, labels[hit], position[hit], reduce="amin"))
    minima_cpu = minima.cpu()
    if int(torch.isfinite(minima_cpu).sum()) <= 1:
        return None
    reduce_sum = all_reduce_fn(context, "sum")
    sizes = torch.zeros(n_blocks, dtype=torch.int64, device=device)
    sizes = reduce_sum(sizes.scatter_add_(0, labels, torch.ones_like(labels)))
    return DiagonalBlocks(
        labels=labels,
        minima=minima_cpu,
        seeds=seeds.cpu(),
        trivial=trivial.cpu(),
        sizes=sizes.cpu(),
        reduce_sum=reduce_sum,
    )
