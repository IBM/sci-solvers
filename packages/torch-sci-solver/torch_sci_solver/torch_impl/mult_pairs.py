"""Pair-mode (no-product-space) matvec kernels.

Product-mode kernels in :mod:`torch_sci_solver.torch_impl.mult` broadcast
each α-excitation over all ``M_β`` β-strings (and vice versa) because they
assume the basis is the Cartesian product. Pair-mode has no such implicit
coupling — the excitation tables already encode the shared spectators (see
:mod:`torch_sci_solver.torch_impl.pair_helpers`), so the matvec becomes a
straightforward per-entry scatter:

    for each row (k_bra, k_ket, cr, an) in the pair-keyed table:
        matelem = ⟨D_{k_bra} | H_{single} | D_{k_ket}⟩   # Slater-Condon
        Wb[k_bra] += matelem * T[k_ket]

Those rows are *streamed*, not stored: ``pair_helpers`` re-derives each table
in bounded chunks on every matvec, because the full row set is O(nnz(H)) and
would not fit at the basis sizes this path exists to reach (see the
:mod:`~torch_sci_solver.torch_impl.pair_helpers` module docstring). Peak
memory is therefore flat in basis dimension.

There is no ``(M_a, M_b)`` reshape, no broadcast over an unused spectator, and
no per-spin factorization of α- and β-parities — the full-det Slater-Condon
primitives (:mod:`torch_sci_solver.torch_impl.determinants`) handle the
signs correctly for any pair of dets.

Correctness anchor: on a full Cartesian product basis, the pair-mode matvec
must produce the exact same ``H @ v`` as the product-mode kernel, up to
floating-point summation order. The unit tests cross-check this on H2/LiH.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    one_excite_batch,
    two_excite_batch,
    zero_excite_batch,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_entries_triton import (
    apply_double_entries,
    apply_single_entries,
)
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    _MoveCache,
    add_ab_blocks,
    add_ab_blocks_local,
    add_same_spin_blocks,
    add_same_spin_blocks_local,
)
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis

if TYPE_CHECKING:
    from torch_sci_solver.torch_impl.distributed_davidson import FlatGather

DEFAULT_CHUNK_SIZE = 2_000_000

# Slater-Condon workspace per lifted entry. Used to cap the scatter chunk so
# peak memory honours the helper's byte budget instead of the entry-count
# default, which would reserve gigabytes.
#
# These are CALIBRATION NUMBERS, measured as peak-RSS delta of a single
# kernel call at norb=36 (2L=72), not derived from the code. ``mult.py``'s
# docstring quotes 1.2 KB for ``one_excite_batch`` from an older kernel; the
# current one costs ~3.2 KB. Re-measure if the kernels or norb change a lot:
# call the kernel on n entries in a fresh process and divide the RSS delta
# by n. Overestimating is safe (smaller chunks); underestimating overshoots
# the budget.
_BYTES_PER_SINGLE_ENTRY = 3200
_BYTES_PER_DOUBLE_ENTRY = 64


def _entry_chunk(chunk_size: int, budget_bytes: int, bytes_per_entry: int) -> int:
    """Entries per Slater-Condon batch: the caller's cap, floored by the budget."""
    return max(1, min(int(chunk_size), int(budget_bytes) // bytes_per_entry))


def make_diag_terms_pair_torch(
    pair_basis: "PairBasis | StructuralPairBasis",
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Diagonal of ``H`` in pair mode: ``hii[k] = ⟨D_k | H | D_k⟩``.

    Pair-mode analog of :func:`torch_sci_solver.torch_impl.mult.make_diag_terms_torch`.
    The α- and β-strings of pair ``k`` are read directly from the pair
    basis's parallel arrays; the diagonal Slater-Condon primitive is
    unchanged.

    Always returns a REAL tensor, even for complex integrals: the diagonal
    expectation value of a Hermitian operator is mathematically real, and
    ``davidson_torch``'s multi-root path sorts ``hii`` with ``torch.argsort``,
    which torch does not support for a complex dtype on CPU — mirrors the
    identical treatment in ``make_diag_terms_torch`` and
    ``make_diag_terms_spinless_torch``.

    Args:
        pair_basis: the explicit ``(α, β)`` list.
        n_orb: number of spatial orbitals.
        integrals: TorchIntegrals bundle.
        chunk_size: max pairs per Slater-Condon batch (memory bound).

    Returns:
        Length-N tensor of diagonal energies on the integrals' device.

    Raises:
        ValueError: ``pair_basis`` is a structural basis, which stores no
            per-pair arrays. Build the owned diagonal slices with
            :func:`make_diag_terms_pair_local` from
            :meth:`StructuralPairBasis.pair_strings` instead.
    """
    if isinstance(pair_basis, StructuralPairBasis):
        raise ValueError(
            "make_diag_terms_pair_torch needs the explicit per-pair arrays; a "
            "structural pair basis supports only the block matvec path. Use "
            "make_diag_terms_pair_local on the pair strings of the indices you need."
        )
    return _pair_diag_terms(
        pair_basis.alpha_of_pair,
        pair_basis.beta_of_pair,
        n_orb,
        integrals,
        chunk_size=chunk_size,
    )


def make_diag_terms_pair_local(
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Diagonal of ``H`` for a contiguous slice of the pair arrays.

    Owner-computes companion of :func:`make_diag_terms_pair_torch`: a rank
    builds only the diagonal of its owned pair block, so the full length-N
    ``hii`` is never materialized. Formula, dtype and device handling are
    identical to the global builder — only the input arrays differ (they are
    the ``[row_start:row_stop]`` slice of the basis's parallel arrays).

    Args:
        alpha_of_pair: ``(n_local,)`` α-string of each owned pair.
        beta_of_pair: ``(n_local,)`` β-string of each owned pair.
        n_orb: number of spatial orbitals.
        integrals: TorchIntegrals bundle.
        chunk_size: max pairs per Slater-Condon batch (memory bound).

    Returns:
        ``(n_local,)`` real diagonal energies on the integrals' device.
    """
    return _pair_diag_terms(alpha_of_pair, beta_of_pair, n_orb, integrals, chunk_size=chunk_size)


def _pair_diag_terms(
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int,
) -> Tensor:
    """Diagonal expectation values for parallel α/β pair arrays (real dtype)."""
    if alpha_of_pair.shape != beta_of_pair.shape:
        raise ValueError(
            "alpha_of_pair and beta_of_pair must be parallel arrays; got "
            f"{tuple(alpha_of_pair.shape)} vs {tuple(beta_of_pair.shape)}"
        )
    device = integrals.device
    dtype = integrals.dtype
    hii_dtype = dtype.to_real() if dtype.is_complex else dtype
    n_pairs = int(alpha_of_pair.shape[0])
    alpha_of_pair = alpha_of_pair.to(device)
    beta_of_pair = beta_of_pair.to(device)

    hii = torch.empty(n_pairs, dtype=hii_dtype, device=device)
    chunk = max(1, int(chunk_size))
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        alpha_chunk = alpha_of_pair[start:end]
        beta_chunk = beta_of_pair[start:end]
        det = det_from_alpha_beta_batch(alpha_chunk, beta_chunk, n_orb)
        hii_chunk = zero_excite_batch(
            det,
            None,
            integrals.n_spinorb,
            integrals.I0,
            integrals.I1_diag,
            integrals.I2_direct,
            integrals.I2_exchange,
        )
        hii[start:end] = hii_chunk.real if dtype.is_complex else hii_chunk
    return hii


def mult_pair_torch(
    hii: Tensor,
    T: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
    cache: _MoveCache | None = None,
) -> Tensor:
    """``Wb = H @ T`` in pair mode.

    Streams the five pair-keyed excitation tables and scatters each entry's
    contribution into ``Wb``. The diagonal pass is a single length-N
    elementwise multiply.

    Args:
        hii: (N,) diagonal of H (from :func:`make_diag_terms_pair_torch`).
        T: (N,) input vector.
        pair_basis: the (α, β) pair list.
        pair_helpers: excitation-table streamer; its rows are re-derived on
            each call rather than held in memory.
        n_orb: number of spatial orbitals.
        integrals: integrals bundle on the pair basis's device.
        chunk_size: max entries per Slater-Condon batch, applied within each
            streamed chunk.
        triton_matvec: route the passes to Triton: in block mode the
            same-spin doubles slices to the output-stationary kernel, in the
            streamed (non-block) mode all singles and doubles tables to the
            entry-parallel kernels. Unsupported devices/layouts fall back to
            the batched kernels.
        cache: optional byte-bounded move-table cache shared by the block
            channels; a Davidson loop passes the same instance on every matvec
            so the stage-independent tables are enumerated once. Only the block
            kernels consult it: the streamed fallback and a structural basis
            ignore it.

    Returns:
        (N,) output vector on the same device as ``hii``.
    """
    n_pairs = pair_basis.n_pairs
    if T.shape != (n_pairs,):
        raise ValueError(f"pair-mode T shape {T.shape} != ({n_pairs},)")

    Wb = hii * T  # diagonal
    device = Wb.device
    # The block kernels index pairs through ``pair_helpers.blocks``; only the
    # streamed fallback needs the explicit pair arrays (and a structural basis
    # does not have them).
    common: dict[str, Any] = {}
    if pair_helpers.blocks is None:
        if isinstance(pair_basis, StructuralPairBasis):
            raise ValueError(
                "a structural pair basis supports only the block matvec path; "
                "enable the block decomposition (block_matvec != 'off')"
            )
        alpha_of_pair = pair_basis.alpha_of_pair.to(device)
        beta_of_pair = pair_basis.beta_of_pair.to(device)
        common = dict(
            Wb=Wb,
            T=T,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            n_orb=n_orb,
            integrals=integrals,
        )
    budget = pair_helpers.budget_bytes
    single_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_SINGLE_ENTRY)
    double_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_DOUBLE_ENTRY)

    # Same-spin channels: block index arithmetic when the basis has an exact
    # Cartesian decomposition, otherwise stream the generic tables.
    if pair_helpers.blocks is not None:
        add_same_spin_blocks(
            Wb,
            T,
            pair_helpers,
            integrals,
            chunk_size=chunk_size,
            triton_matvec=triton_matvec,
            cache=cache,
        )
    else:
        # Same-spin singles: α then β.
        for stream in (pair_helpers.asingles, pair_helpers.bsingles):
            for t in stream():
                _scatter_single_pair(
                    k_bra=t["k_bra"],
                    k_ket=t["k_ket"],
                    cr=t["cr0"],
                    an=t["an0"],
                    chunk_size=single_chunk,
                    triton_matvec=triton_matvec,
                    **common,
                )

        # Same-spin doubles: α then β.
        for stream in (pair_helpers.adoubles, pair_helpers.bdoubles):
            for t in stream():
                _scatter_double_pair(
                    k_bra=t["k_bra"],
                    k_ket=t["k_ket"],
                    cr0=t["cr0"],
                    cr1=t["cr1"],
                    an0=t["an0"],
                    an1=t["an1"],
                    chunk_size=double_chunk,
                    triton_matvec=triton_matvec,
                    **common,
                )

    # αβ-doubles: block index arithmetic when the basis has an exact Cartesian
    # decomposition, otherwise stream the generic table. two_excite_batch takes
    # any four spin-orbital indices; for a cross-spin double the α and β
    # orbitals differ on both creation and annihilation.
    if pair_helpers.blocks is not None:
        add_ab_blocks(Wb, T, pair_helpers, integrals, chunk_size=chunk_size, cache=cache)
    else:
        for t in pair_helpers.ab_doubles():
            _scatter_double_pair(
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr0=t["cr_a"],
                cr1=t["cr_b"],
                an0=t["an_a"],
                an1=t["an_b"],
                chunk_size=double_chunk,
                triton_matvec=triton_matvec,
                **common,
            )
    return Wb


def prefetch_while_pending(stream: Iterable[Any], pending: "FlatGather") -> Iterator[Any]:
    """``stream`` with at most one item drawn while ``pending`` is still in flight.

    The excitation-table streams do not read the input vector, so the first
    chunk is enumerated while the all-gather runs; at most one chunk is held
    ahead, which bounds the extra memory to one table chunk.
    """
    iterator = iter(stream)
    ahead: list[Any] = []
    if not pending.done():
        ahead.extend(itertools.islice(iterator, 1))
    return itertools.chain(ahead, iterator)


def mult_pair_local_torch(
    hii_local: Tensor,
    T_local: Tensor,
    T_full: Tensor | None,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    integrals: TorchIntegrals,
    row_start: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    triton_matvec: bool = False,
    cache: _MoveCache | None = None,
    T_pending: "FlatGather | None" = None,
) -> Tensor:
    """Owner-computes pair matvec: ``y = H_local x`` from the replicated ``T_full``.

    Rank-local companion of :func:`mult_pair_torch` for the distributed GDB
    path. The five pair-keyed tables are streamed with
    ``pair_range=(row_start, row_start + N_local)``, so only entries whose bra
    pair belongs to this rank are enumerated; each surviving entry's matrix
    element is evaluated from its **global** bra pair and scattered into the
    local output at ``k_bra - row_start``, reading the ket amplitude from the
    full vector. ``T_full`` is therefore the only length-N tensor, while the
    Slater-Condon work and the output block are both ~``1/P``.

    Notes
    -----
    The pair-space Hamiltonian is

    .. math::

        (H x)_k = h_{kk} x_k
          + \\sum_{(k \\leftarrow j) \\in \\mathcal{T}}
            \\langle D_k | H | D_j \\rangle x_j,

    over the five streamed tables :math:`\\mathcal{T}` (α/β singles, α/α and
    β/β doubles, αβ doubles). For a rank owning the pair block
    :math:`[s_r, s_{r+1})` only the rows with :math:`k \\in [s_r, s_{r+1})`
    contribute to its output, and every needed :math:`x_j` is available in
    the replicated full vector — no halo exchange is required, unlike TPB.

    Args:
        hii_local: ``(N_local,)`` diagonal of the owned pair block.
        T_local: ``(N_local,)`` owned input block (used for the diagonal).
        T_full: ``(N,)`` replicated input vector, same values as ``T_local``
            on ``[row_start, row_start + N_local)``.
        pair_basis: the global ``(α, β)`` pair list.
        pair_helpers: excitation-table streamer for ``pair_basis``.
        n_orb: number of spatial orbitals.
        integrals: integrals bundle on the pair basis's device.
        row_start: global index of the first owned pair.
        chunk_size: max entries per Slater-Condon batch.
        triton_matvec: route the passes to Triton: in block mode the
            same-spin doubles slices to the output-stationary kernel, in the
            streamed (non-block) mode all singles and doubles tables to the
            entry-parallel kernels. Unsupported devices/layouts fall back to
            the batched kernels.
        cache: optional byte-bounded move-table cache shared by the block
            channels; an owner-computes operator passes the same instance on
            every matvec so its inverse move tables are enumerated once. Only
            the block kernels consult it: the streamed fallback and a
            structural basis ignore it.
        T_pending: a started :class:`~torch_sci_solver.torch_impl.distributed_davidson.
            FlatGather` of ``T_local`` in place of ``T_full`` (pass ``T_full=None``).
            The diagonal, and on the streamed tables the first excitation chunk,
            are computed while the gather is in flight.

    Returns:
        ``(N_local,)`` owned output block.
    """
    if (T_full is None) == (T_pending is None):
        raise ValueError("pass exactly one of T_full and T_pending")
    n_local = int(hii_local.shape[0])
    n_pairs = pair_basis.n_pairs
    row_start = int(row_start)
    row_stop = row_start + n_local
    if T_local.shape != (n_local,):
        raise ValueError(f"pair-mode T_local shape {T_local.shape} != ({n_local},)")
    if T_full is not None and T_full.shape != (n_pairs,):
        raise ValueError(f"pair-mode T_full shape {T_full.shape} != ({n_pairs},)")
    if row_start < 0 or row_stop > n_pairs:
        raise ValueError(f"owned pair range [{row_start}, {row_stop}) is outside [0, {n_pairs})")

    if pair_helpers.blocks is None and isinstance(pair_basis, StructuralPairBasis):
        raise ValueError(
            "a structural pair basis supports only the block matvec path; "
            "enable the block decomposition (block_matvec != 'off')"
        )
    pair_range = (row_start, row_stop)
    y = hii_local * T_local  # diagonal
    device = y.device
    first_stream: Iterable[Any] = ()
    if T_pending is not None:
        # Enumerate ahead while the gather runs: the first streamed table, or
        # nothing on the block path (its move tables are cached across calls).
        if pair_helpers.blocks is None:
            first_stream = prefetch_while_pending(
                pair_helpers.asingles(pair_range=pair_range), T_pending
            )
        T_full = T_pending.wait()
        if T_full.shape != (n_pairs,):
            raise ValueError(f"pair-mode T_full shape {T_full.shape} != ({n_pairs},)")
    common: dict[str, Any] = {}
    if pair_helpers.blocks is None:
        alpha_of_pair = pair_basis.alpha_of_pair.to(device)
        beta_of_pair = pair_basis.beta_of_pair.to(device)
        common = dict(
            Wb=y,
            T=T_full,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            n_orb=n_orb,
            integrals=integrals,
            bra_range=pair_range,
        )
    budget = pair_helpers.budget_bytes
    single_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_SINGLE_ENTRY)
    double_chunk = _entry_chunk(chunk_size, budget, _BYTES_PER_DOUBLE_ENTRY)

    # Same-spin channels: block index arithmetic when the basis has an exact
    # Cartesian decomposition, otherwise stream the generic tables.
    if pair_helpers.blocks is not None:
        add_same_spin_blocks_local(
            y,
            T_full,
            pair_helpers,
            integrals,
            row_start,
            chunk_size=chunk_size,
            triton_matvec=triton_matvec,
            cache=cache,
        )
    else:
        # Same-spin singles: α then β (α already started when prefetched).
        alpha_singles = (
            first_stream if T_pending is not None else pair_helpers.asingles(pair_range=pair_range)
        )
        for tables in (alpha_singles, pair_helpers.bsingles(pair_range=pair_range)):
            for t in tables:
                _scatter_single_pair(
                    k_bra=t["k_bra"],
                    k_ket=t["k_ket"],
                    cr=t["cr0"],
                    an=t["an0"],
                    chunk_size=single_chunk,
                    triton_matvec=triton_matvec,
                    **common,
                )

        # Same-spin doubles: α then β.
        for stream in (pair_helpers.adoubles, pair_helpers.bdoubles):
            for t in stream(pair_range=pair_range):
                _scatter_double_pair(
                    k_bra=t["k_bra"],
                    k_ket=t["k_ket"],
                    cr0=t["cr0"],
                    cr1=t["cr1"],
                    an0=t["an0"],
                    an1=t["an1"],
                    chunk_size=double_chunk,
                    triton_matvec=triton_matvec,
                    **common,
                )

    # αβ-doubles: same spin-agnostic two_excite_batch as the global pass, or
    # the block kernel when the basis decomposes exactly.
    if pair_helpers.blocks is not None:
        add_ab_blocks_local(
            y, T_full, pair_helpers, integrals, row_start, chunk_size=chunk_size, cache=cache
        )
    else:
        for t in pair_helpers.ab_doubles(pair_range=pair_range):
            _scatter_double_pair(
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr0=t["cr_a"],
                cr1=t["cr_b"],
                an0=t["an_a"],
                an1=t["an_b"],
                chunk_size=double_chunk,
                triton_matvec=triton_matvec,
                **common,
            )
    return y


# ----------------------------------------------------------------------
# Per-table chunked scatter helpers
# ----------------------------------------------------------------------


def _scatter_single_pair(
    *,
    Wb: Tensor,
    T: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    chunk_size: int,
    bra_range: tuple[int, int] | None = None,
    triton_matvec: bool = False,
) -> None:
    """Scatter one_excite_batch contributions for a same-spin single table.

    For each entry ``i``, the bra det is ``(α_of_pair[k_bra[i]],
    β_of_pair[k_bra[i]])`` and the excitation orbitals are
    ``(cr[i], an[i])``. Calls ``one_excite_batch`` with the bra det (its
    definition — the electron is annihilated at ``an`` and created at
    ``cr`` — matches the SBD convention used throughout this package)
    and scatters ``matelem * T[k_ket]`` into ``Wb[k_bra]``.

    ``bra_range=(start, stop)`` selects the owner-computes mode: only entries
    whose ``k_bra`` lies in ``[start, stop)`` contribute (the caller's stream
    normally already guarantees this; the filter keeps a non-restricted stream
    correct too), and ``Wb`` is the rank-local output block indexed by
    ``k_bra - start``.

    With ``triton_matvec`` the filtered table goes to the entry-parallel
    Triton kernel (:func:`~torch_sci_solver.torch_impl.mult_entries_triton.
    apply_single_entries`) with the same bra/ket roles and the owner-computes
    ``offset``; an unsupported device or layout falls back to the chunked
    batched scatter below.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    offset = 0
    if bra_range is not None:
        offset = int(bra_range[0])
        keep = (k_bra >= offset) & (k_bra < int(bra_range[1]))
        if not bool(keep.all()):
            k_bra, k_ket = k_bra[keep], k_ket[keep]
            cr, an = cr[keep], an[keep]
            k_total = int(k_bra.shape[0])
            if k_total == 0:
                return
    if triton_matvec and apply_single_entries(
        Wb,
        T,
        integrals,
        k_bra=k_bra,
        k_ket=k_ket,
        cr=cr,
        an=an,
        mode="pair",
        alpha_of_pair=alpha_of_pair,
        beta_of_pair=beta_of_pair,
        offset=offset,
    ):
        return
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        bra_alpha = alpha_of_pair[kb]
        bra_beta = beta_of_pair[kb]
        bra = det_from_alpha_beta_batch(bra_alpha, bra_beta, n_orb)
        matelem = one_excite_batch(
            bra,
            None,
            an[start:end],
            cr[start:end],
            integrals.I1,
            integrals.I2_eri,
            integrals.n_spinorb,
        )
        Wb.index_add_(0, kb if offset == 0 else kb - offset, matelem * T[kk])


def _scatter_double_pair(
    *,
    Wb: Tensor,
    T: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    chunk_size: int,
    bra_range: tuple[int, int] | None = None,
    triton_matvec: bool = False,
) -> None:
    """Scatter two_excite_batch contributions for a double-excitation table.

    Handles same-spin doubles (α-α or β-β) and cross-spin αβ-doubles
    uniformly: the primitive is spin-agnostic once you have four
    spin-orbital indices.

    ``two_excite_batch(det, i, j, a, b, I2)`` uses the SBD convention where
    ``i, j`` are the *annihilation* orbitals and ``a, b`` are the *creation*
    orbitals. Our helpers store ``cr0, cr1`` (creations) and ``an0, an1``
    (annihilations) — we thread them as ``i=an0, j=an1, a=cr0, b=cr1``.

    ``bra_range=(start, stop)`` selects the owner-computes mode; see
    :func:`_scatter_single_pair`.

    With ``triton_matvec`` the filtered table goes to the entry-parallel
    Triton kernel (:func:`~torch_sci_solver.torch_impl.mult_entries_triton.
    apply_double_entries`) with the same bra/ket roles and the owner-computes
    ``offset``; an unsupported device or layout falls back to the chunked
    batched scatter below.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    offset = 0
    if bra_range is not None:
        offset = int(bra_range[0])
        keep = (k_bra >= offset) & (k_bra < int(bra_range[1]))
        if not bool(keep.all()):
            k_bra, k_ket = k_bra[keep], k_ket[keep]
            cr0, cr1 = cr0[keep], cr1[keep]
            an0, an1 = an0[keep], an1[keep]
            k_total = int(k_bra.shape[0])
            if k_total == 0:
                return
    if triton_matvec and apply_double_entries(
        Wb,
        T,
        integrals,
        k_bra=k_bra,
        k_ket=k_ket,
        cr0=cr0,
        cr1=cr1,
        an0=an0,
        an1=an1,
        mode="pair",
        alpha_of_pair=alpha_of_pair,
        beta_of_pair=beta_of_pair,
        offset=offset,
    ):
        return
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        bra_alpha = alpha_of_pair[kb]
        bra_beta = beta_of_pair[kb]
        bra = det_from_alpha_beta_batch(bra_alpha, bra_beta, n_orb)
        matelem = two_excite_batch(
            bra,
            None,
            an0[start:end],
            an1[start:end],
            cr0[start:end],
            cr1[start:end],
            integrals.I2_eri,
        )
        Wb.index_add_(0, kb if offset == 0 else kb - offset, matelem * T[kk])
