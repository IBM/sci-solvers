"""Complex-correct selected-CI reduced density matrices.

The RDM builders in ``pyscf.fci.selected_ci`` (and the torch ``rdm.py``) assume a
*real* CI vector: they contract amplitudes without conjugating the bra, so for a
**complex** CI vector they silently drop the imaginary cross-terms and return a
wrong (non-Hermitian) density matrix. This module provides self-contained,
complex-correct 1- and 2-RDM builders for an arbitrary selected determinant list,
so :class:`~sci_solvers.energy_solvers.multiroot_sci_solver.MultirootSCISolver`
can report correct energies, occupancies, and ``<S^2>`` on complex Hermitian
integrals.

The one rule that makes them complex-correct is: **conjugate the bra amplitude**,

    gamma_pq = sum_{IJ} <I| E_pq |J> conj(C_I) C_J,

realized by contracting the excitation result against ``C.conj()``.

Efficiency
----------
The single-excitation tensor ``B[p, q] = E_pq |C>`` (spin-resolved into ``Ba`` and
``Bb``) is built **once** with a single vectorized scatter-add per spin over the
flattened PySCF ``cistring`` link table -- no Python loop over determinants or
orbitals. Both RDMs are then pure ``einsum`` contractions of that tensor:

* ``dm1 = <C| E_pq |C>`` -- one contraction of ``B`` against ``conj(C)``.
* ``dm2`` uses the adjoint identity ``E_pq^dagger = E_qp`` so that
  ``<C| E_pq E_rs |C> = <E_qp C | E_rs C> = sum_ij conj(B[q,p,ij]) B[r,s,ij]`` --
  a single ``einsum`` over the *already-built* ``B``. This keeps the peak memory at
  ``O(norb^2 * Na * Nb)`` (the ``B`` tensor); it never materializes an
  ``O(norb^4)`` intermediate, and the whole build is BLAS-bound.

These operate on **spin-block** amplitudes ``C`` of shape ``(Na, Nb)`` -- the
eigenvectors the solver already returns (the interleaved-Wick ``sigma`` similarity
is folded into the matvec in ``build_selected_complex_ci_problem``), so there is no
interleaved-Wick sign to re-apply here.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Tuple

import numpy as np


class SparseAddressMap:
    """Canonical determinant address -> row/index, without a dense ``num_strings``-length array.

    Drop-in replacement for the dense ``remap``/``canon_to_conn`` arrays used to filter
    the selected-CI link tables: those were allocated as ``np.full(num_strings(norb, nele), -1)``
    and indexed directly by canonical address, which is both larger than ``2**31`` (the ceiling
    ``pyscf.fci.cistring.strs2addr`` enforces) and infeasible to allocate for a large active
    space (e.g. ``norb=56, nele=36`` -> ``C(56, 36) ~ 7.9e14`` addresses). Since only a handful
    of addresses (the selected determinants and their single-excitation neighbors) are ever
    looked up, a sorted-keys + ``searchsorted`` map costs ``O(len(addrs))`` instead.
    """

    def __init__(self, addrs: np.ndarray, rows: np.ndarray):
        """Build the map from parallel ``addrs`` (keys) and ``rows`` (values) arrays."""
        order = np.argsort(addrs, kind="stable")
        self._addrs = np.asarray(addrs, dtype=np.int64)[order]
        self._rows = np.asarray(rows, dtype=np.int64)[order]

    def __getitem__(self, queries: np.ndarray) -> np.ndarray:
        """Vectorized lookup; ``-1`` for an address not in the map."""
        queries = np.asarray(queries, dtype=np.int64)
        pos = np.searchsorted(self._addrs, queries)
        pos_c = np.clip(pos, 0, len(self._addrs) - 1)
        found = (pos < len(self._addrs)) & (self._addrs[pos_c] == queries)
        out = np.full(queries.shape, -1, dtype=np.int64)
        out[found] = self._rows[pos_c[found]]
        return out

    def present_addrs(self) -> np.ndarray:
        """All mapped addresses, ascending -- the sparse analogue of ``np.nonzero(dense >= 0)[0]``."""
        return self._addrs


def _link_arrays(norb: int, nele: int):
    """Flatten a PySCF ``cistring`` link table into scatter-ready index arrays.

    ``cistring.gen_linkstr_index`` returns an ``(nstr, nlink, 4)`` table whose rows
    ``(p, q, dst, sign)`` mean ``a^+_p a_q |str_src> = sign |str_dst>`` (``src`` is
    the row index). Returning the flattened ``(p, q, dst, src, sign)`` arrays lets
    :func:`_apply_excitations` apply *every* single excitation of one spin in a
    single vectorized ``np.add.at`` instead of a Python double loop.
    """
    from pyscf.fci import cistring

    link = cistring.gen_linkstr_index(range(norb), nele).astype(np.int64)
    nstr, nlink = link.shape[0], link.shape[1]
    p = link[:, :, 0].ravel()
    q = link[:, :, 1].ravel()
    dst = link[:, :, 2].ravel()
    sign = link[:, :, 3].ravel().astype(np.float64)
    src = np.repeat(np.arange(nstr, dtype=np.int64), nlink)
    return p, q, dst, src, sign


def _link_arrays_selected(norb: int, nele: int, strs: np.ndarray):
    """Selected-source ``_link_arrays`` -- never enumerates the full canonical space.

    Builds the flattened ``(p, q, dst, src, sign)`` link arrays for a **selected**
    determinant list ``strs`` (an arbitrary subset of the canonical ``(norb, nele)``
    strings). Unlike :func:`_link_arrays`, whose ``src``/``dst`` are **canonical
    ``cistring`` addresses** (meaningful only because the full-canonical table's row
    index *is* its address), here ``src``/``dst`` are the determinant **bitmasks**
    themselves -- the address space :class:`SparseAddressMap` keys the selected
    ``row_map``/connected-set index on for this path.

    This never calls ``pyscf.fci.cistring.gen_linkstr_index``/``strs2addr``: both
    ultimately rely on pyscf's C ``binomial()`` helper, an unmemoized Pascal's-triangle
    recursion that is exponential in ``norb`` once it exceeds a small in-code cache
    (``norb > 20``) and stores addresses in 32-bit ints, so at e.g. ``norb=56`` a
    *single* source string takes on the order of a minute (and can silently overflow)
    regardless of how small ``len(strs)`` is. Building the link table directly from
    bitmasks -- annihilate each occupied orbital, create each virtual orbital, sign
    via the parity of the bits strictly between them (the same convention as
    ``cistring``'s ``FCIcre_des_sign``) -- is a handful of vectorized numpy ops over
    ``len(strs) * nele * nvir`` entries, with no combinatorial addressing at all.

    ``strs`` need not be sorted (unlike the ``cistring``-backed version this replaces).
    """
    strs = np.ascontiguousarray(np.asarray(strs, dtype=np.int64))
    nsel = strs.shape[0]
    orbs = np.arange(norb, dtype=np.int64)

    occ_mask = ((strs[:, None] >> orbs[None, :]) & 1).astype(bool)  # (nsel, norb)
    order = np.argsort(~occ_mask, axis=1, kind="stable")  # occupied orbitals first
    occ_idx = np.sort(order[:, :nele], axis=1)  # (nsel, nele) ascending occ orbitals
    vir_idx = np.sort(order[:, nele:], axis=1)  # (nsel, nvir) ascending vir orbitals

    # Diagonal (p == q) occupied-occupied entries: the string is unchanged, sign +1.
    p_diag = occ_idx
    q_diag = occ_idx
    dst_diag = np.broadcast_to(strs[:, None], (nsel, nele))
    sign_diag = np.ones((nsel, nele), dtype=np.float64)
    src_diag = np.broadcast_to(strs[:, None], (nsel, nele))

    # Off-diagonal (p != q): annihilate occupied i, create virtual a.
    src0 = strs[:, None, None]
    occ_b = occ_idx[:, :, None]  # (nsel, nele, 1)
    vir_b = vir_idx[:, None, :]  # (nsel, 1, nvir)
    dst = (src0 ^ (np.int64(1) << occ_b)) | (np.int64(1) << vir_b)  # (nsel, nele, nvir)
    lo = np.minimum(occ_b, vir_b)
    hi = np.maximum(occ_b, vir_b)
    mask = (np.int64(1) << hi) - (np.int64(1) << (lo + 1))
    # np.bitwise_count returns an unsigned dtype narrow enough that 1 - 2*(pc % 2)
    # underflows in unsigned arithmetic; cast to a signed type before the sign formula.
    popcount = np.bitwise_count((src0 & mask).astype(np.uint64)).astype(np.int64)
    sign = (1 - 2 * (popcount % 2)).astype(np.float64)
    p_off = np.broadcast_to(vir_b, dst.shape)
    q_off = np.broadcast_to(occ_b, dst.shape)
    src_off = np.broadcast_to(src0, dst.shape)

    p = np.concatenate([p_diag.ravel(), p_off.ravel()])
    q = np.concatenate([q_diag.ravel(), q_off.ravel()])
    dst_all = np.concatenate([dst_diag.ravel(), dst.ravel()])
    sign_all = np.concatenate([sign_diag.ravel(), sign.ravel()])
    src_all = np.concatenate([src_diag.ravel(), src_off.ravel()])
    return p, q, dst_all, src_all, sign_all


def _restrict_to_connected_set(link, row_map: SparseAddressMap):
    """Restrict a link table to selected *sources* and re-index onto the connected set.

    ``link`` is the ``(p, q, dst, src, sign)`` tuple from :func:`_link_arrays` (whose
    ``src``/``dst`` are canonical ``cistring`` addresses) or :func:`_link_arrays_selected`
    (whose ``src``/``dst`` are determinant bitmasks instead). ``row_map`` maps each
    address in that same space to its row in the selected amplitude vector, or ``-1``
    when that determinant is not selected.

    For a CI vector supported only on the selected set ``S``, the reduced density
    matrices are

        gamma_pq          = sum_K conj(C_K) B[p, q, K]         (1-RDM)
        <E_pq E_rs>       = sum_K conj(B[q, p, K]) B[r, s, K]  (2-RDM Gram term)

    with ``B[p, q, K] = sum_{J in S} sign * C_J`` over links ``E_pq|J> = sign|K>``.
    The **source** ``J`` must lie in ``S`` (else ``C_J = 0``), but the intermediate
    ``K`` ranges over the full *connected set* -- every determinant reachable from
    ``S`` by one excitation, not just ``S`` itself. Restricting ``K`` to ``S`` (as a
    selected -> selected filter would) is correct for the 1-RDM (where ``conj(C_K)``
    already vanishes off ``S``) but **wrong for the 2-RDM Gram term**, which visits
    determinants outside ``S`` as the E_rs intermediate.

    So this keeps only links with a **selected source**, forms the connected set
    ``S union {reachable dst}``, and assigns it a dense contiguous index (via a
    :class:`SparseAddressMap`, since the canonical addresses themselves can be far
    larger than the connected set and too large to index a dense array by) that both
    :func:`_connected_link_and_embed` (single-axis embedding) and
    :func:`_connected_link_and_embed_product` (joint two-axis embedding) build on.

    Returns ``(p, q, dst_c, src_c, sign, canon_to_conn, n_conn)`` where ``dst_c``/
    ``src_c`` are the link's ``dst``/``src`` re-indexed onto the connected set and
    ``canon_to_conn`` is the :class:`SparseAddressMap` doing that re-indexing (so a
    caller can also map its own addresses, e.g. the selected set itself, onto the
    same connected index).
    """
    p, q, dst, src, sign = link
    keep = row_map[src] >= 0
    p, q, dst, src, sign = p[keep], q[keep], dst[keep], src[keep], sign[keep]

    # Connected set: selected determinants (so C is representable even for the
    # operators that never move them) plus every reachable destination.
    sel_addr = row_map.present_addrs()
    connected = np.union1d(sel_addr, dst)
    canon_to_conn = SparseAddressMap(connected, np.arange(connected.shape[0], dtype=np.int64))
    n_conn = int(connected.shape[0])

    src_c = canon_to_conn[src]
    dst_c = canon_to_conn[dst]
    return p, q, dst_c, src_c, sign, canon_to_conn, n_conn


def _connected_link_and_embed(link, row_map: SparseAddressMap, C: np.ndarray, axis: int):
    """Single-axis specialization of :func:`_restrict_to_connected_set`.

    Embeds ``C`` (at its *original* ``(Na, Nb)`` shape) onto the connected axis
    selected by ``axis`` (0 = alpha rows, 1 = beta columns); the other axis of
    ``C`` is carried through unchanged. Correct when only one axis is selected
    (the spinless ``(N, 0)`` sector, where the untouched axis is trivial) --
    see :func:`_connected_link_and_embed_product` for the case where *both*
    axes are independently selected.

    Returns ``(p, q, dst_c, src_c, sign, C_conn, n_conn)`` where ``dst_c``/``src_c``
    index the connected axis and ``C_conn`` is ``C`` re-laid onto it.
    """
    p, q, dst_c, src_c, sign, canon_to_conn, n_conn = _restrict_to_connected_set(link, row_map)
    sel_addr = row_map.present_addrs()

    # Embed C (selected rows) onto the connected axis: C_conn[conn_idx of S] = C[S].
    sel_conn = canon_to_conn[sel_addr]  # connected index of each selected determinant
    if axis == 0:
        C_conn = np.zeros((n_conn, C.shape[1]), dtype=C.dtype)
        C_conn[sel_conn, :] = C[row_map[sel_addr], :]
    else:
        C_conn = np.zeros((C.shape[0], n_conn), dtype=C.dtype)
        C_conn[:, sel_conn] = C[:, row_map[sel_addr]]
    return p, q, dst_c, src_c, sign, C_conn, n_conn


def _connected_link_and_embed_product(
    link_a, link_b, row_map_a: SparseAddressMap, row_map_b: SparseAddressMap, C: np.ndarray
):
    """Joint two-axis embedding for a genuinely selected *product* CI subspace.

    Unlike :func:`_connected_link_and_embed`, which embeds one axis and leaves
    the other at its original (full or trivial) width, this embeds **both**
    the alpha and beta determinant axes onto their own connected sets
    simultaneously -- required when a CI vector is supported on an explicit
    product ``span{|a>|b>}`` over two independently-truncated string sets
    (the TPB selected-CI subspace ``build_selected_complex_ci_problem``
    builds), not just a single-axis-selected subspace.

    ``C`` is the dense ``(Na_sel, Nb_sel)`` amplitude block on the *selected*
    product basis -- exactly the shape the SCI solver's eigenvector already
    has, indexed in the same order as ``row_map_a``/``row_map_b``'s selected
    rows/columns. No outer-product scatter is needed to build it; only its
    *placement* into the bigger joint connected grid.

    Returns ``(link_a_conn, link_b_conn, C_conn, n_conn_a, n_conn_b)`` where
    ``link_a_conn``/``link_b_conn`` are ``(p, q, dst, src, sign)`` tuples with
    ``dst``/``src`` re-indexed onto the alpha/beta connected sets
    respectively, and ``C_conn`` is the ``(n_conn_a, n_conn_b)`` joint
    embedding of ``C`` (zero outside the selected rows/columns).
    """
    p_a, q_a, dst_a, src_a, sign_a, canon_to_conn_a, n_conn_a = _restrict_to_connected_set(
        link_a, row_map_a
    )
    p_b, q_b, dst_b, src_b, sign_b, canon_to_conn_b, n_conn_b = _restrict_to_connected_set(
        link_b, row_map_b
    )

    sel_addr_a = row_map_a.present_addrs()
    sel_addr_b = row_map_b.present_addrs()
    sel_conn_a = canon_to_conn_a[sel_addr_a]  # connected index of each selected alpha det
    sel_conn_b = canon_to_conn_b[sel_addr_b]  # connected index of each selected beta det

    # Place the existing dense selected block into the joint connected grid --
    # a single dense assignment, not a scatter-add: C's rows/columns are
    # already ordered to match row_map_a/row_map_b's selected rows/columns.
    C_conn = np.zeros((n_conn_a, n_conn_b), dtype=C.dtype)
    C_conn[np.ix_(sel_conn_a, sel_conn_b)] = C[np.ix_(row_map_a[sel_addr_a], row_map_b[sel_addr_b])]

    return (
        (p_a, q_a, dst_a, src_a, sign_a),
        (p_b, q_b, dst_b, src_b, sign_b),
        C_conn,
        n_conn_a,
        n_conn_b,
    )


# ============================================================================
# Chunked product-mode 1-/2-RDM -- memory-safe at a large connected set
# ============================================================================
#
# The unchunked product-mode path above materializes Ba/Bb as dense
# (norb, norb, n_conn_a, n_conn_b) arrays -- fine when n_conn_a * n_conn_b is
# small, but a selected subspace with a large virtual space can have a
# connected set (selected dets union every single-excitation-reachable
# destination) far bigger than the selected set itself: e.g. norb=40,
# n_elec=20 per spin, 100 selected determinants can reach ~16.5k connected
# determinants per axis, making the (norb, norb, n_conn_a, n_conn_b) tensor
# infeasible (norb**2 * 16.5k**2 * 16 bytes ~ 6.4 TiB).
#
# The (norb, norb) leading axes are independent of the n_conn_a * n_conn_b
# blowup, so -- mirroring spinless_ci_problem._spinless_rdm12_chunked's
# already-proven "chunk over the flattened (p, q) operator axis" pattern for
# its (Bb == 0) alpha-only case -- this section reconstructs the same
# quantities in blocks, at O(chunk * n_conn_a * n_conn_b) peak Ba/Bb
# residency instead of O(norb**2 * n_conn_a * n_conn_b).


def _sort_link_by_pq(link, norb: int):
    """Sort a ``(p, q, dst, src, sign)`` link tuple into CSR form by ``flat_pq = p*norb+q``.

    Returns ``(dst, src, sign, starts)`` where ``starts`` is the
    ``(norb*norb + 1,)`` offset array (``starts[a] : starts[a+1]`` is the
    contiguous slice of links for flattened operator index ``a``), letting a
    caller slice out a block of ``(p, q)`` rows in ``O(links_in_block)`` with
    no per-block scan of the full table -- the same role
    ``spinless_ci_problem._alpha_link_sorted``'s CSR layout plays there.
    """
    p, q, dst, src, sign = link
    flat = (p * norb + q).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    starts = np.searchsorted(flat_s, np.arange(norb * norb + 1, dtype=np.int64))
    return dst[order], src[order], sign[order], starts


def _build_excitation_block_product(
    C_conn: np.ndarray, link_sorted, a0: int, a1: int, *, transpose: bool
) -> np.ndarray:
    """Build the ``(a1 - a0, n_conn_a, n_conn_b)`` slice of ``Ba`` or ``Bb``.

    ``link_sorted`` is one spin's CSR-sorted link table from
    :func:`_sort_link_by_pq` (alpha: rows index ``C_conn``; beta: columns).
    ``transpose=False`` builds an alpha block (scatters into the row axis of
    ``C_conn``, keeping the column axis intact); ``transpose=True`` builds a
    beta block (scatters into the column axis, keeping the row axis intact --
    mirrors :func:`_apply_excitations_b`'s transpose-scatter-transpose shape).
    Only the requested block's links are scattered, so peak residency is
    ``(block, n_conn_a, n_conn_b)`` rather than the full
    ``(norb*norb, n_conn_a, n_conn_b)``.
    """
    dst, src, sign, starts = link_sorted
    lo, hi = int(starts[a0]), int(starts[a1])
    n_conn_a, n_conn_b = C_conn.shape
    if transpose:
        out = np.zeros((a1 - a0, n_conn_b, n_conn_a), dtype=C_conn.dtype)
        if hi > lo:
            rows_local = (
                np.repeat(np.arange(a0, a1, dtype=np.int64), np.diff(starts[a0 : a1 + 1])) - a0
            )
            np.add.at(out, (rows_local, dst[lo:hi]), sign[lo:hi, None] * C_conn[:, src[lo:hi]].T)
        return out.transpose(0, 2, 1)
    out = np.zeros((a1 - a0, n_conn_a, n_conn_b), dtype=C_conn.dtype)
    if hi > lo:
        rows_local = np.repeat(np.arange(a0, a1, dtype=np.int64), np.diff(starts[a0 : a1 + 1])) - a0
        np.add.at(out, (rows_local, dst[lo:hi]), sign[lo:hi, None] * C_conn[src[lo:hi], :])
    return out


# Dimension threshold (n_conn_a * n_conn_b) above which chunk=None auto-chunks
# the product-mode RDM build. Mirrors spinless_ci_problem._RDM_CHUNK_DIM_THRESHOLD's
# role for the alpha-only path; kept as a separate constant since the two
# paths' "dim" means different things (a single determinant count there, the
# joint connected-grid size here).
_PRODUCT_RDM_CHUNK_DIM_THRESHOLD = 50_000


def _resolve_rdm_chunk_product(chunk, n_conn_a: int, n_conn_b: int, norb: int):
    """Resolve the ``chunk`` policy for the product-mode path.

    Same contract as ``spinless_ci_problem._resolve_rdm_chunk``: ``None``
    returned means "no chunking, use the direct unchunked path" (exact for
    small problems); a positive int is the flattened ``(p, q)`` block size.
    """
    if chunk is not None:
        return int(chunk) if int(chunk) > 0 else None
    dim = n_conn_a * n_conn_b
    if dim >= _PRODUCT_RDM_CHUNK_DIM_THRESHOLD:
        # Aim for a ~0.5 GB per-tensor block: 0.5e9 / (16 * dim) rows, clamped
        # to [1, norb**2]. Two block tensors (Ba_block, Bb_block) are live at
        # once, so actual peak is ~2x this target -- same headroom choice
        # _resolve_rdm_chunk makes for its single alpha-only block.
        rows = max(1, int(0.5e9 / (16.0 * max(1, dim))))
        return min(norb * norb, rows)
    return None


def _build_product_connected_bundle(
    C: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    row_map_a: SparseAddressMap,
    row_map_b: SparseAddressMap,
    strs_a,
    strs_b,
):
    """Build the ``(link_a_conn, link_b_conn, C_conn, n_conn_a, n_conn_b)`` bundle
    shared by both the unchunked and chunked product-mode paths.

    Separated out so the chunk-size decision (which needs ``n_conn_a`` /
    ``n_conn_b``, only known after this step) and the actual ``Ba``/``Bb``
    materialization (the expensive, size-dependent part) are two distinct
    stages -- this step itself is cheap (link-table + connected-set
    bookkeeping only, no ``(norb, norb, ...)``-shaped allocation).
    """
    neleca, nelecb = int(nelec[0]), int(nelec[1])
    link_a = (
        _link_arrays_selected(norb, neleca, strs_a)
        if strs_a is not None
        else _link_arrays(norb, neleca)
    )
    link_b = (
        _link_arrays_selected(norb, nelecb, strs_b)
        if strs_b is not None
        else _link_arrays(norb, nelecb)
    )
    return _connected_link_and_embed_product(link_a, link_b, row_map_a, row_map_b, C)


def _make_rdm12_complex_product_chunked(
    norb: int,
    link_a_conn,
    link_b_conn,
    C_conn: np.ndarray,
    chunk: int,
    *,
    need_2rdm: bool = True,
    need_gab: bool = True,
) -> Tuple[np.ndarray, np.ndarray, "np.ndarray | None", "np.ndarray | None"]:
    """Chunked product-mode 1-/2-RDM (+ alpha-beta block), without materializing
    the full ``(norb, norb, n_conn_a, n_conn_b)`` ``Ba``/``Bb`` tensors.

    Takes the connected-set bundle :func:`_build_product_connected_bundle`
    already built (so the chunk-size decision in
    :func:`_resolve_rdm_chunk_product`, which needs ``C_conn.shape``, happens
    exactly once regardless of which path runs). Builds ``Ba``/``Bb`` in
    blocks of ``chunk`` flattened ``(p, q)`` operator rows (see the
    module-level note above this section), reconstructing the same ``dma``,
    ``dmb``, ``dm2``, ``Gab`` :func:`make_rdm1_complex` /
    :func:`make_rdm12_complex` return. The 2-RDM Gram term
    ``T[p,q,r,s] = sum_ij conj(B[q,p,ij]) B[r,s,ij]`` (``B = Ba + Bb``) is
    assembled as a block-by-block Gram matrix exactly as
    :func:`~sci_solvers.spinless_ci_problem._spinless_rdm12_chunked` does for
    its alpha-only case; the alpha-beta-only block ``Gab = sum_ij
    conj(Ba[q,p,ij]) Bb[r,s,ij]`` is new here (the spinless path never needs
    it, since its ``Bb`` is identically zero) but built the same way, from
    ``Ba``/``Bb`` block pairs kept separate before the spin sum.

    Returns ``(dma, dmb, dm2, Gab)`` -- callers combine ``dma + dmb`` for the
    spin-summed 1-RDM (see :func:`make_rdm1_complex`); ``dm2``/``Gab`` are
    ``None`` when the caller doesn't need them (skips the ``O(norb**4)`` Gram
    matrix build entirely).
    """
    link_a_sorted = _sort_link_by_pq(link_a_conn, norb)
    link_b_sorted = _sort_link_by_pq(link_b_conn, norb)

    P = norb * norb
    n_conn_a, n_conn_b = C_conn.shape
    Cc = C_conn.conj()

    dma_flat = np.empty(P, dtype=np.complex128)
    dmb_flat = np.empty(P, dtype=np.complex128)
    G = np.empty((P, P), dtype=np.complex128) if need_2rdm else None
    Gab_flat = np.empty((P, P), dtype=np.complex128) if need_gab else None

    def _block(i0, i1):
        Ba_i = _build_excitation_block_product(C_conn, link_a_sorted, i0, i1, transpose=False)
        Bb_i = _build_excitation_block_product(C_conn, link_b_sorted, i0, i1, transpose=True)
        return Ba_i, Bb_i

    Cc_flat = Cc.reshape(-1)
    n_ij = n_conn_a * n_conn_b

    if not need_2rdm and not need_gab:
        # dm1-only fast path: dma/dmb are Hermitian (dma[q,p] == conj(dma[p,q])),
        # so only the upper triangle (p <= q) needs a block build at all --
        # the lower triangle is filled from its conjugate transpose. Halves
        # the number of (block, n_conn_a, n_conn_b) scatter-builds, which
        # dominate wall time at a large connected set (each is a multi-GB
        # allocation + scatter, not cheap to skip).
        for p in range(norb):
            for q in range(p, norb):
                a_pq = p * norb + q
                Ba_pq, Bb_pq = _block(a_pq, a_pq + 1)
                v_a = (Ba_pq.reshape(1, n_ij) @ Cc_flat)[0]
                v_b = (Bb_pq.reshape(1, n_ij) @ Cc_flat)[0]
                dma_flat[a_pq] = v_a
                dmb_flat[a_pq] = v_b
                if q != p:
                    dma_flat[q * norb + p] = np.conj(v_a)
                    dmb_flat[q * norb + p] = np.conj(v_b)
        return dma_flat.reshape(norb, norb), dmb_flat.reshape(norb, norb), None, None

    blocks: dict = {}
    for i0 in range(0, P, chunk):
        i1 = min(i0 + chunk, P)
        Ba_i, Bb_i = blocks.pop(i0, None) or _block(i0, i1)
        # dma/dmb[p, q] = <C| E_pq |C> = sum_ij conj(C_ij) B[pq, ij], contracted
        # as an explicit reshape + matmul rather than np.einsum: even with
        # einsum(..., optimize=True), this "aij,ij->a" shape (a tiny leading
        # dim, a large contracted (i, j) pair) does not reliably get lowered to
        # a BLAS gemv -- measured ~5x slower than the equivalent matmul at the
        # tensor sizes a large connected set reaches. Same reasoning applies to
        # the Gram-matrix contractions below.
        dma_flat[i0:i1] = Ba_i.reshape(i1 - i0, n_ij) @ Cc_flat
        dmb_flat[i0:i1] = Bb_i.reshape(i1 - i0, n_ij) @ Cc_flat
        if not need_2rdm and not need_gab:
            continue
        B_i = Ba_i + Bb_i
        B_i_flat = B_i.reshape(i1 - i0, n_ij)
        Ba_i_flat_conj = Ba_i.reshape(i1 - i0, n_ij).conj() if need_gab else None
        for j0 in range(0, P, chunk):
            j1 = min(j0 + chunk, P)
            if j0 == i0:
                Ba_j, Bb_j = Ba_i, Bb_i
                B_j_flat = B_i_flat
            else:
                Ba_j, Bb_j = _block(j0, j1)
                blocks[j0] = (Ba_j, Bb_j)
                B_j_flat = (Ba_j + Bb_j).reshape(j1 - j0, n_ij)
            if G is not None:
                G[i0:i1, j0:j1] = B_i_flat.conj() @ B_j_flat.T
            if Gab_flat is not None:
                Bb_j_flat = Bb_j.reshape(j1 - j0, n_ij)
                Gab_flat[i0:i1, j0:j1] = Ba_i_flat_conj @ Bb_j_flat.T

    dma = dma_flat.reshape(norb, norb)
    dmb = dmb_flat.reshape(norb, norb)
    dm2 = None
    if G is not None:
        dm1 = dma + dmb
        # T[p,q,r,s] = sum_ij conj(B[q,p,ij]) B[r,s,ij] = G[(q,p), (r,s)]; G is
        # indexed by flattened (row, col), so reshape to (row, col, row, col)
        # then swap the first two axes to put (p, q) in front.
        T = np.transpose(G.reshape(norb, norb, norb, norb), (1, 0, 2, 3))
        dm2 = T
        for qq in range(norb):
            dm2[:, qq, qq, :] -= dm1
    Gab = None
    if Gab_flat is not None:
        Gab = np.transpose(Gab_flat.reshape(norb, norb, norb, norb), (1, 0, 2, 3))
    return dma, dmb, dm2, Gab


def _apply_excitations_a(C: np.ndarray, link, norb: int, *, row_map=None):
    """``Ba[p, q] = E^alpha_pq |C>`` as an ``(norb, norb, Ndet, Nb)`` tensor.

    One vectorized scatter-add over the flattened alpha link table: for every link
    ``(p, q, dst, src, sign)`` accumulate ``sign * C[src, :]`` into ``Ba[p, q, dst, :]``.

    ``row_map=None`` (default) is the full-canonical path (``Ndet == Na``),
    byte-identical to before. When ``row_map`` is given, ``C`` is a **selected**
    subspace column and the alpha determinant axis is re-indexed onto the *connected
    set* via :func:`_connected_link_and_embed`; the returned ``C_conn`` (the embedded
    amplitudes on that axis) is returned alongside ``t`` so the downstream
    contractions bind to the same axis. Returns ``t`` when ``row_map is None``, else
    ``(t, C_conn)``.
    """
    Na, Nb = C.shape
    if row_map is not None:
        p, q, dst, src, sign, C_conn, n_conn = _connected_link_and_embed(link, row_map, C, axis=0)
        t = np.zeros((norb, norb, n_conn, Nb), dtype=C.dtype)
        np.add.at(t, (p, q, dst), sign[:, None] * C_conn[src, :])
        return t, C_conn
    p, q, dst, src, sign = link
    t = np.zeros((norb, norb, Na, Nb), dtype=C.dtype)
    np.add.at(t, (p, q, dst), sign[:, None] * C[src, :])
    return t


def _apply_excitations_b(C: np.ndarray, link, norb: int, *, row_map=None):
    """``Bb[p, q] = E^beta_pq |C>`` as an ``(norb, norb, Na, Ndet)`` tensor.

    The beta strings index the *columns* of ``C``; scatter into a
    ``(norb, norb, Ndet, Na)`` buffer (beta axis third) and swap the trailing axes
    back, so the single vectorized ``np.add.at`` mirrors the alpha path.

    ``row_map`` (default ``None``) has the same selected-subspace meaning as in
    :func:`_apply_excitations_a` — here it re-indexes the *beta* determinant axis
    onto the connected set and returns the embedded ``(Na, n_conn)`` ``C_conn``
    alongside the tensor. Returns ``tT`` transposed when ``row_map is None``, else
    ``(t, C_conn)``.
    """
    Na, Nb = C.shape
    if row_map is not None:
        p, q, dst, src, sign, C_conn, n_conn = _connected_link_and_embed(link, row_map, C, axis=1)
        tT = np.zeros((norb, norb, n_conn, Na), dtype=C.dtype)
        np.add.at(tT, (p, q, dst), sign[:, None] * C_conn[:, src].T)
        return tT.transpose(0, 1, 3, 2), C_conn
    p, q, dst, src, sign = link
    tT = np.zeros((norb, norb, Nb, Na), dtype=C.dtype)
    np.add.at(tT, (p, q, dst), sign[:, None] * C[:, src].T)
    return tT.transpose(0, 1, 3, 2)


def _single_excitation_tensors(
    C: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    row_map_a=None,
    row_map_b=None,
    strs_a=None,
    strs_b=None,
):
    """Build the spin-resolved single-excitation tensors ``(Ba, Bb, C_eff)``.

    ``Ba[p, q] = E^alpha_pq |C>`` and ``Bb[p, q] = E^beta_pq |C>``. Shared by all
    builders here so ``C`` is excited only once. Returns ``C_eff``, the amplitude
    vector aligned to the (shared) determinant axis of ``Ba``/``Bb`` -- equal to
    ``C`` on the full-canonical path and the connected-set embedding on the selected
    path -- so the downstream contractions bind the bra ``conj(C_eff)`` to the same
    axis ``B`` lives on.

    ``row_map_a`` / ``row_map_b`` (default ``None``) re-index the alpha / beta
    determinant axis onto the *connected set* of a selected CI subspace -- see
    :func:`_apply_excitations_a` (single-axis) / :func:`_connected_link_and_embed_product`
    (both axes). Both ``None`` is the full-canonical behavior, byte-identical to
    before (``C_eff is C``).

    ``strs_a`` / ``strs_b`` (default ``None``), the sorted selected alpha/beta
    determinant bitmasks parallel to ``row_map_a``/``row_map_b``, route the
    alpha/beta link table through the selected-source :func:`_link_arrays_selected`
    -- which never enumerates the full canonical ``C(norb, neleca)``/``C(norb,
    nelecb)`` link table. On a large active space (e.g. 28 spin-orbitals, 12
    electrons) that full table is hundreds of GB and aborts the process, so
    ``strs_a``/``strs_b`` are required for the selected path to be memory-safe
    there; when one is ``None`` that axis's selected path falls back to
    filtering the full :func:`_link_arrays` table (the small-system behavior).

    Supported selected configurations:

    * both ``row_map_a``/``row_map_b`` ``None`` -- full canonical (any ``nelec``).
    * ``row_map_a`` only, with ``nelecb == 0`` -- the spinless ``(N, 0)`` sector,
      where the beta link table is empty so ``Bb == 0`` and the shared axis is the
      alpha connected set.
    * both ``row_map_a`` and ``row_map_b`` given -- a genuine product-mode
      selected CI subspace (``span{|a>|b>}`` over two independently-truncated
      string sets, e.g. the TPB subspace ``build_selected_complex_ci_problem``
      builds). Both axes are embedded onto their own connected set via
      :func:`_connected_link_and_embed_product`; ``Ba``/``Bb`` and ``C_eff``
      then all live on the joint ``(n_conn_a, n_conn_b)`` grid.
    """
    C = np.ascontiguousarray(C)
    neleca, nelecb = int(nelec[0]), int(nelec[1])

    if row_map_a is None and row_map_b is None:
        Ba = _apply_excitations_a(C, _link_arrays(norb, neleca), norb)
        Bb = _apply_excitations_b(C, _link_arrays(norb, nelecb), norb)
        return Ba, Bb, C

    if row_map_a is not None and row_map_b is None and nelecb == 0:
        # Spinless (N, 0) sector: only the alpha axis is nontrivial. B lives on the
        # alpha connected set; Bb is identically zero, shaped to match for the sum.
        # Build the alpha link table from the selected sources only when strs_a is
        # given -- never enumerate the full canonical link table for a large CAS.
        link_a = (
            _link_arrays_selected(norb, neleca, strs_a)
            if strs_a is not None
            else _link_arrays(norb, neleca)
        )
        Ba, C_conn = _apply_excitations_a(C, link_a, norb, row_map=row_map_a)
        Bb = np.zeros_like(Ba)
        return Ba, Bb, C_conn

    if row_map_a is not None and row_map_b is not None:
        # Product-mode selected subspace: both axes independently truncated.
        link_a = (
            _link_arrays_selected(norb, neleca, strs_a)
            if strs_a is not None
            else _link_arrays(norb, neleca)
        )
        link_b = (
            _link_arrays_selected(norb, nelecb, strs_b)
            if strs_b is not None
            else _link_arrays(norb, nelecb)
        )
        link_a_conn, link_b_conn, C_conn, n_conn_a, n_conn_b = _connected_link_and_embed_product(
            link_a, link_b, row_map_a, row_map_b, C
        )
        p_a, q_a, dst_a, src_a, sign_a = link_a_conn
        p_b, q_b, dst_b, src_b, sign_b = link_b_conn
        Ba = np.zeros((norb, norb, n_conn_a, n_conn_b), dtype=C.dtype)
        np.add.at(Ba, (p_a, q_a, dst_a), sign_a[:, None] * C_conn[src_a, :])
        # Beta strings index C_conn's columns; scatter into a (norb, norb,
        # n_conn_b, n_conn_a) buffer (beta axis third) and swap the trailing
        # axes back, mirroring _apply_excitations_b's single-axis approach.
        BbT = np.zeros((norb, norb, n_conn_b, n_conn_a), dtype=C.dtype)
        np.add.at(BbT, (p_b, q_b, dst_b), sign_b[:, None] * C_conn[:, src_b].T)
        Bb = BbT.transpose(0, 1, 3, 2)
        return Ba, Bb, C_conn

    raise NotImplementedError(
        "Selected-CI RDMs require either a single row_map (alpha-only, "
        "nelecb == 0, the spinless sector), both row_map_a and row_map_b "
        "(product-mode selected subspace), or neither (full canonical); got "
        f"row_map_a={'set' if row_map_a is not None else None}, "
        f"row_map_b={'set' if row_map_b is not None else None}, nelecb={nelecb}."
    )


def make_rdm1_complex(
    C: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    row_map_a=None,
    row_map_b=None,
    strs_a=None,
    strs_b=None,
    chunk=None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Complex-correct spin-summed and spin-resolved 1-RDMs.

    Args:
        C: Complex CI amplitudes, shape ``(Na, Nb)`` (spin-block convention).
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        row_map_a: Optional address -> selected-alpha-row map (``-1`` for unselected),
            restricting the alpha excitations to a selected subspace (see
            :func:`_single_excitation_tensors`). The address space is canonical
            ``cistring`` addresses when ``strs_a is None``, or determinant bitmasks
            when ``strs_a`` is given (see :func:`_link_arrays_selected`). ``None``
            (default) is the full-canonical path.
        row_map_b: Optional address -> selected-beta-column map, same address-space
            convention as ``row_map_a`` (keyed by ``strs_b``). ``None`` (default)
            is the full-canonical path. Passing both ``row_map_a`` and ``row_map_b``
            selects the product-mode path (see :func:`_single_excitation_tensors`);
            passing only ``row_map_a`` with ``nelec[1] == 0`` selects the spinless
            ``(N, 0)`` sector.
        strs_a: Optional sorted selected alpha determinant bitmasks parallel to
            ``row_map_a``; routes the alpha link table through the selected-source
            builder so the full canonical link table is never enumerated (required
            for a large selected CAS -- see :func:`_single_excitation_tensors`).
        strs_b: Optional sorted selected beta determinant bitmasks parallel to
            ``row_map_b``; same purpose as ``strs_a`` for the beta link table
            (product-mode path only -- the spinless path never has a nontrivial
            beta axis).
        chunk: Product-mode-only memory knob (ignored on the full-canonical and
            spinless paths, which are always cheap enough not to need it).
            ``None`` (default) auto-chunks only once the joint connected-set
            size ``n_conn_a * n_conn_b`` reaches
            :data:`_PRODUCT_RDM_CHUNK_DIM_THRESHOLD` -- below that, the direct
            :func:`_single_excitation_tensors` path runs unchanged. A positive
            int forces that block size (flattened ``(p, q)`` operator rows);
            a non-positive int forces the unchunked path regardless of size.
            See :func:`_make_rdm12_complex_product_chunked` for why this
            exists: the unchunked path's ``Ba``/``Bb`` are
            ``(norb, norb, n_conn_a, n_conn_b)``, which can be infeasible even
            for a small selected set when the connected set it reaches is
            large (e.g. a large virtual space).

    Returns:
        ``(dm1, (dma, dmb))`` where ``dm1 = dma + dmb`` is the spin-summed 1-RDM
        (``dm1[p, q] = <a^+_p a_q>``, Hermitian) and ``dma``/``dmb`` are the
        spin-resolved alpha/beta 1-RDMs. The real diagonals of ``dma``/``dmb`` are
        the spin-resolved orbital occupancies.
    """
    C = np.ascontiguousarray(C)
    if row_map_a is not None and row_map_b is not None:
        link_a_conn, link_b_conn, C_conn, n_conn_a, n_conn_b = _build_product_connected_bundle(
            C, norb, nelec, row_map_a, row_map_b, strs_a, strs_b
        )
        resolved = _resolve_rdm_chunk_product(chunk, n_conn_a, n_conn_b, norb)
        if resolved is not None:
            dma, dmb, _dm2, _gab = _make_rdm12_complex_product_chunked(
                norb, link_a_conn, link_b_conn, C_conn, resolved, need_2rdm=False, need_gab=False
            )
            return dma + dmb, (dma, dmb)
    Ba, Bb, C_eff = _single_excitation_tensors(
        C, norb, nelec, row_map_a=row_map_a, row_map_b=row_map_b, strs_a=strs_a, strs_b=strs_b
    )
    # Conjugate the bra: gamma_pq = <C| E_pq |C> = sum_ij conj(C_ij) (E_pq C)_ij.
    # optimize=True: numpy's default einsum path lacks BLAS dispatch for this
    # contraction shape and is far slower at the tensor sizes a selected CAS
    # with a large connected set reaches -- see the chunked driver's identical
    # note.
    dma = np.einsum("ij,pqij->pq", C_eff.conj(), Ba, optimize=True)
    dmb = np.einsum("ij,pqij->pq", C_eff.conj(), Bb, optimize=True)
    return dma + dmb, (dma, dmb)


def make_rdm12_complex(
    C: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    row_map_a=None,
    row_map_b=None,
    strs_a=None,
    strs_b=None,
    chunk=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complex-correct spin-summed 1- and 2-RDM plus the alpha-beta 2-RDM block.

    Args:
        C: Complex CI amplitudes, shape ``(Na, Nb)`` (spin-block convention).
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        row_map_a: Optional address -> selected-alpha-row map (``-1`` for unselected),
            in the same address space as :func:`make_rdm1_complex`'s ``row_map_a``;
            ``None`` (default) is the full-canonical path.
        row_map_b: Optional address -> selected-beta-column map; ``None`` (default)
            is the full-canonical path. See :func:`make_rdm1_complex` for the
            spinless-vs-product-mode dispatch rule.
        strs_a: Optional sorted selected alpha determinant bitmasks parallel to
            ``row_map_a``; routes the alpha link table through the selected-source
            builder so the full canonical link table is never enumerated (required
            for a large selected CAS -- see :func:`_single_excitation_tensors`).
        strs_b: Optional sorted selected beta determinant bitmasks parallel to
            ``row_map_b`` (product-mode path only); see :func:`make_rdm1_complex`.
        chunk: Product-mode-only memory knob; see :func:`make_rdm1_complex`'s
            ``chunk`` for the full contract. Here it additionally bounds the
            2-RDM/``Gab`` Gram-matrix build, not just the 1-RDM.

    Returns:
        ``(dm1, dm2, dm2_ab)``:

        * ``dm1`` -- spin-summed 1-RDM, ``dm1[p, q] = <a^+_p a_q>``.
        * ``dm2`` -- spin-summed 2-RDM in the ``pyscf.fci.selected_ci.make_rdm2``
          convention ``dm2[p, q, r, s] = <a^+_p a^+_r a_s a_q>``, so the electronic
          energy is ``<dm1, hcore> + 0.5 * einsum("prqs,prqs->", dm2, eri)``
          (matching the real path's energy line).
        * ``dm2_ab`` -- the alpha-beta 2-RDM block in PySCF chemist convention
          ``dm2_ab[p, q, r, s] = <a^+_pa a^+_rb a_sb a_qa>``, consumed by
          :func:`s_squared_complex`.
    """
    C = np.ascontiguousarray(C)
    if row_map_a is not None and row_map_b is not None:
        link_a_conn, link_b_conn, C_conn, n_conn_a, n_conn_b = _build_product_connected_bundle(
            C, norb, nelec, row_map_a, row_map_b, strs_a, strs_b
        )
        resolved = _resolve_rdm_chunk_product(chunk, n_conn_a, n_conn_b, norb)
        if resolved is not None:
            dma, dmb, dm2, Gab = _make_rdm12_complex_product_chunked(
                norb, link_a_conn, link_b_conn, C_conn, resolved
            )
            assert dm2 is not None and Gab is not None  # need_2rdm / need_gab default True
            return dma + dmb, dm2, Gab
    Ba, Bb, C_eff = _single_excitation_tensors(
        C, norb, nelec, row_map_a=row_map_a, row_map_b=row_map_b, strs_a=strs_a, strs_b=strs_b
    )
    B = Ba + Bb  # E_pq|C>, spin-summed
    dm1 = np.einsum("ij,pqij->pq", C_eff.conj(), B, optimize=True)

    # <C| E_pq E_rs |C> via the adjoint identity E_pq^dagger = E_qp:
    #   <C| E_pq E_rs |C> = <E_qp C | E_rs C> = sum_ij conj(B[q,p,ij]) B[r,s,ij].
    # No O(norb^4) intermediate -- a single contraction of the prebuilt B tensor.
    # optimize=True: see the chunked driver's note -- essential, not cosmetic,
    # at connected-set scale.
    T = np.einsum("qpij,rsij->pqrs", B.conj(), B, optimize=True)
    # dm2[p,q,r,s] = <a^+_p a^+_r a_s a_q> = <E_pq E_rs> - delta_{qr} <E_ps>.
    dm2 = T
    for q in range(norb):
        dm2[:, q, q, :] -= dm1

    # Alpha-beta block for <S^2>: Gab[p,q,r,s] = <a^+_pa a^+_rb a_sb a_qa>
    #   = <(E^a_pq)(E^b_rs)> = <E^a_qp C | E^b_rs C> (different spins: no delta
    #   correction). Reuse the spin-resolved single-excitation tensors.
    Gab = np.einsum("qpij,rsij->pqrs", Ba.conj(), Bb, optimize=True)
    return dm1, dm2, Gab


def energy_from_rdm(
    dm1: np.ndarray, dm2: np.ndarray, hcore: np.ndarray, eri: np.ndarray
) -> complex:
    """Electronic energy ``<dm1, hcore> + 0.5 * einsum("prqs,prqs->", dm2, eri)``.

    Uses the same contraction the real path uses in
    :meth:`~sci_solvers.energy_solvers.multiroot_sci_solver.MultirootSCISolver._get_eigen_states`.
    For a Hermitian problem the imaginary part is ~machine-zero; the caller takes
    the real part.
    """
    e1 = np.einsum("pr,pr->", dm1, hcore)
    e2 = 0.5 * np.einsum("prqs,prqs->", dm2, eri)
    return e1 + e2


def s_squared_complex(C: np.ndarray, norb: int, nelec: Tuple[int, int]) -> float:
    """``<S^2>`` of a complex CI state, from its complex alpha-beta 2-RDM.

    Uses ``<S^2> = (S_z^2 - S_z) + N_alpha - sum_{p,q} Gab[p, q, q, p]`` with
    ``S_z = (N_alpha - N_beta) / 2`` -- the same formula as the torch
    ``s_squared_from_rdm`` (``rdm.py``), fed the complex-correct alpha-beta 2-RDM.
    Only the alpha-beta block is needed, so it is built directly (skipping the
    spin-summed 2-RDM). The result is real for a physical state; the
    (machine-zero) imaginary part is dropped.

    Args:
        C: Complex CI amplitudes, shape ``(Na, Nb)``.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.

    Returns:
        ``<S^2>`` as a Python ``float`` (the ``S(S+1)`` eigenvalue convention: 0
        for a singlet, 2 for a triplet, 6 for a quintet).
    """
    C = np.ascontiguousarray(C)
    neleca, nelecb = int(nelec[0]), int(nelec[1])
    Ba, Bb, _ = _single_excitation_tensors(C, norb, nelec)
    Gab = np.einsum("qpij,rsij->pqrs", Ba.conj(), Bb, optimize=True)
    s_z = (neleca - nelecb) / 2.0
    cross = np.einsum("pqqp->", Gab)
    val = (s_z * s_z - s_z) + neleca - cross
    return float(np.real(val))
