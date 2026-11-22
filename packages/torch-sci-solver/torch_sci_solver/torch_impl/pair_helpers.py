"""Streaming pair-keyed excitation tables for the pair-mode (no-product-space) matvec.

Product-mode helpers (:mod:`torch_sci_solver.torch_impl.helpers`) enumerate
α-excitations independently of β and vice versa: the matvec then broadcasts
each α-excitation over *all* β-strings, implicitly assuming the basis is a
Cartesian product.

Pair mode has no such implicit coupling. For each α-excitation ``a → a'`` an
entry contributes only when both ``(a, β)`` and ``(a', β)`` exist as pairs in
the basis. Which candidate excitations survive that filter can only be found
by enumerating them and testing membership.

Streaming, not stored
---------------------
There are ``nnz(H)`` surviving entries, and ``nnz(H) / dim`` grows with basis
size — order 1000 at a few 10k determinants on a 36-orbital problem. So
materializing the surviving entries costs O(nnz) memory: ~1.5 GB at 26k
determinants, tens to hundreds of GB at the 1e6–1e7 determinants this solver
targets. That is more memory than storing ``H`` itself, for a table that only
holds *indices* — the matrix elements still get recomputed every matvec.

So we do not store them. :class:`PairHelpers` holds only the O(N) pair-basis
arrays and *re-derives* each table in chunks on every pass, yielding one chunk
at a time:

    for chunk in helpers.asingles():
        ...  # chunk["k_bra"], chunk["k_ket"], chunk["cr0"], chunk["an0"]

Peak memory is set by ``budget_bytes`` (defaults to ``DEFAULT_BUDGET_BYTES``,
see :func:`resolve_budget_bytes`) and is independent of
both ``nnz(H)`` and the basis dimension. The cost is re-enumerating on every
pass instead of once — the deliberate trade for a solver that must fit in
cluster memory before it is fast.

Table kinds — five, one per matvec code path:

Same-spin (:meth:`~PairHelpers.asingles`, :meth:`~PairHelpers.adoubles`,
:meth:`~PairHelpers.bsingles`, :meth:`~PairHelpers.bdoubles`):

  * ``k_bra``   pair index of the bra ``(bra_string, spectator)``
  * ``k_ket``   pair index of the ket ``(ket_string, spectator)``
                (shared spectator with k_bra by construction)
  * ``cr0, an0``            spin-orbital indices of a single excitation
  * ``cr0, cr1, an0, an1``  the four indices of a double

Cross-spin (:meth:`~PairHelpers.ab_doubles`):

  * ``k_bra``, ``k_ket``  bra and ket pair indices
  * ``cr_a, an_a, cr_b, an_b``  the four spin-orbital indices

Excitation direction: in every table the ``cr``/``an`` indices describe the
excitation that takes the **ket to the bra** — ``D_bra = a+_cr ... a_an ... D_ket``
— so ``cr`` is occupied in the bra and ``an`` in the ket. That is the order the
matrix-element and RDM consumers (``mult_pairs``, ``rdm_pairs``) read them in:
``<D_bra|H|D_ket>`` evaluates ``one_excite_batch(bra, an, cr)`` /
``two_excite_batch(bra, an0, an1, cr0, cr1)``, and the 1-RDM accumulates
``conj(C_bra) C_ket`` into ``gamma[cr, an]`` (``= <a+_cr a_an>``). With complex
integrals the opposite labelling would apply ``conj(H)`` instead of ``H``.

Implementation
--------------
One shared primitive: pair membership. Pairs are lex-sorted by
``(α_uid, β_uid)``, so ``pair_keys[k] = α_uid[k] * m_beta + β_uid[k]`` is
sorted and ``torch.searchsorted`` answers "is this (α, β) in the basis, and
at which index?" exactly, in log time, on CPU and CUDA alike — no hash table.

The cross-spin pass enumerates a slice of *pairs*, looks each excited string up
in the per-spin unique array, then looks the resulting pair up in ``pair_keys``,
and yields the survivors.

The same-spin passes do not enumerate per pair, because a same-spin excitation
leaves the spectator untouched: every pair sharing a spin string would re-derive
an identical candidate set. They instead enumerate once per *unique string* and
expand the survivors over the pairs containing them, via the partners CSR — see
:func:`_stream_same_spin_dedup`. On a GDB basis, where each string recurs across
many pairs, that removes the dominant cost of these passes.
:func:`_stream_same_spin` is retained as the per-pair reference the dedup path is
tested against.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import _popcount64
from torch_sci_solver.torch_impl.pair_blocks import PairBlocks, build_pair_blocks
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis

# Peak bytes per enumeration chunk. Chunk sizes are derived from this and the
# per-pair candidate count, so peak memory is flat in basis size.
#
# This is NOT a total-memory figure. It bounds the *intermediate workspace of
# one chunk* while the excitation rows are being re-derived — nothing else. The
# Hamiltonian, the basis, the Davidson vectors and the spin-orbital ERI tensor
# all live outside it.
#
# Same default on every device, CPU included. A "biggest chunk that fits free
# VRAM" auto-sizer was tried on CUDA and measured *slower* at every dimension
# tested (up to 4.7x, at 10k pairs on an A100) than a flat 8192 MB budget —
# oversized chunks apparently thrash whatever cache/occupancy sweet spot the
# kernels want, so more free memory is not more throughput. Pass an explicit
# budget_bytes to override if a given shape benefits from something else.
DEFAULT_BUDGET_BYTES = 8192 * 1024**2


def resolve_budget_bytes(budget_bytes: "int | None") -> int:
    """Per-chunk workspace budget: explicit if given, else :data:`DEFAULT_BUDGET_BYTES`.

    Args:
        budget_bytes: An explicit budget, or None to use the default.

    Returns:
        The byte budget for one enumeration chunk.
    """
    if budget_bytes is not None:
        return max(1, int(budget_bytes))
    return DEFAULT_BUDGET_BYTES


# Int64 temporaries live at once inside one chunk of each enumerator, used
# only to turn ``budget_bytes`` into a chunk size. Calibrated against measured
# peak RSS of a single enumeration pass rather than counted from the source,
# since torch fuses and caches allocations. Overestimating is cheap (smaller
# chunks, marginally slower); underestimating overshoots the budget, so these
# lean high.
_TEMPS_PER_CANDIDATE = {"singles": 16, "doubles": 20, "ab": 24}

# Bytes per surviving αβ cross-product row. The αβ pass filters each spin before
# crossing them (see :func:`_stream_ab_doubles`), so the cross product holds only
# survivors — but on a densely connected basis every candidate survives, and the
# product is then as large as the naive one. This bounds that stage: ~16 int64
# temporaries live per row (pair index, within-pair offsets, both survivor
# indices, both uids, four orbital indices, the query key, the lookup result and
# its mask).
#
# Measured, not counted: at a 512 MB budget a densely connected 10,000-pair
# basis peaked at 559 MB with 16, and at ~300 MB with 32, so torch keeps roughly
# twice as much alive as the visible locals suggest. Overestimating only costs
# smaller slices.
_BYTES_PER_CROSS_ROW = 32 * 8


def _chunk_pairs(n_per_pair: int, kind: str, budget_bytes: int) -> int:
    """Pairs per chunk that keep one enumeration chunk within ``budget_bytes``.

    A chunk materializes roughly ``chunk × n_per_pair`` int64 candidates
    across ``_TEMPS_PER_CANDIDATE[kind]`` temporaries. Always at least 1, so
    a single pair with a huge candidate count still makes progress (it just
    exceeds the budget — unavoidable without splitting the candidate axis).
    """
    per_pair_bytes = max(1, n_per_pair * 8 * _TEMPS_PER_CANDIDATE[kind])
    return max(1, int(budget_bytes) // per_pair_bytes)


@dataclass
class _AbCache:
    """Byte-bounded cache of per-unique-string α/β single survivors.

    The αβ-double pass enumerates a pair chunk's single excitations once per
    unique string (see :func:`_stream_ab_doubles`). Those survivors depend only
    on the string, and the chunk layout repeats on every Davidson matvec, so
    the per-chunk results are keyed by ``(chunk_start, chunk_stop, is_alpha)``
    and reused. Once ``budget_bytes`` is exhausted later chunks are recomputed;
    nothing of size ``nnz(H)`` is stored.

    Attributes:
        budget_bytes: Cap on the resident survivor bytes.
        entries: ``(start, stop, is_alpha) -> (cnt, uid, an, cr)``.
        used_bytes: Bytes currently held in ``entries``.
        full: Set once the budget is exhausted (or one chunk alone exceeds it).
    """

    budget_bytes: int
    entries: dict[tuple[int, int, bool], tuple[Tensor, Tensor, Tensor, Tensor]] = field(
        default_factory=dict
    )
    used_bytes: int = 0
    full: bool = False


@dataclass(frozen=True)
class PairHelpers:
    """Streaming pair-keyed excitation tables over an explicit ``(α, β)`` basis.

    Holds only O(N) arrays. The five table-kind methods are generators that
    re-derive their entries in chunks on each call; nothing of size
    ``nnz(H)`` is ever allocated. See the module docstring for why.

    Attributes:
        n_orb: Number of spatial orbitals.
        device: Device the streamed chunks are produced on.
        budget_bytes: Peak-memory target for one enumeration chunk.
        last_nnz: Rows yielded by the most recent full pass of each table
            kind, keyed by method name. Populated as a side effect of
            streaming (a pass must complete before its entry is final), so
            it is zero until the first matvec. Diagnostics only.
    """

    n_orb: int
    device: torch.device
    n_a: int
    n_b: int
    m_beta: int
    unique_alphas: Tensor
    unique_betas: Tensor
    m_alpha: int
    budget_bytes: int = DEFAULT_BUDGET_BYTES
    string_cache_bytes: int = 0
    alpha_of_pair: "Tensor | None" = None
    beta_of_pair: "Tensor | None" = None
    alpha_uid_of_pair: "Tensor | None" = None
    beta_uid_of_pair: "Tensor | None" = None
    pair_keys: "Tensor | None" = None
    # Partners CSR per spin, with the spectator uids pre-offset into a globally
    # ascending key space (see generate_pair_helpers). Used by the deduplicated
    # same-spin enumerator to expand one excitation over every pair that shares
    # its string. ``None`` for a structural basis, which has no pair list.
    partners_a_offsets: "Tensor | None" = None
    partners_a_pair_idx: "Tensor | None" = None
    partners_a_spectator_key: "Tensor | None" = None
    partners_b_offsets: "Tensor | None" = None
    partners_b_pair_idx: "Tensor | None" = None
    partners_b_spectator_key: "Tensor | None" = None
    blocks: "PairBlocks | None" = field(default=None, repr=False)
    structural: "StructuralPairBasis | None" = field(default=None, repr=False)
    last_nnz: dict[str, int] = field(default_factory=dict)
    _ab_cache: "_AbCache | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Attach the byte-bounded per-string survivor cache."""
        cache = _AbCache(self.string_cache_bytes) if self.string_cache_bytes > 0 else None
        object.__setattr__(self, "_ab_cache", cache)

    def _require_explicit(self, name: str) -> None:
        """Reject a streamed-table request on a structural basis.

        Args:
            name: Table kind, for the error message.

        Raises:
            ValueError: always, when the helpers carry a structural basis.
        """
        if self.structural is not None:
            raise ValueError(
                f"PairHelpers.{name} is unavailable for a structural pair basis: "
                "the structural basis supports only the block matvec path "
                "(pair_helpers.blocks); the streamed excitation tables need the "
                "explicit per-pair arrays it does not store."
            )

    # ---- same-spin -----------------------------------------------------
    def asingles(self, pair_range: tuple[int, int] | None = None) -> Iterator[dict[str, Tensor]]:
        """Stream α single excitations, chunk by chunk.

        Args:
            pair_range: Optional ``(start, stop)`` pair-index range. When given,
                only rows whose bra pair ``k_bra`` lies in ``[start, stop)`` are
                yielded; ``None`` streams the full table.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("asingles")
        return self._same_spin("asingles", is_alpha=True, order=1, pair_range=pair_range)

    def adoubles(self, pair_range: tuple[int, int] | None = None) -> Iterator[dict[str, Tensor]]:
        """Stream α double excitations, chunk by chunk.

        Args:
            pair_range: Optional ``(start, stop)`` pair-index range; see
                :meth:`asingles`.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("adoubles")
        return self._same_spin("adoubles", is_alpha=True, order=2, pair_range=pair_range)

    def bsingles(self, pair_range: tuple[int, int] | None = None) -> Iterator[dict[str, Tensor]]:
        """Stream β single excitations, chunk by chunk.

        Args:
            pair_range: Optional ``(start, stop)`` pair-index range; see
                :meth:`asingles`.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("bsingles")
        return self._same_spin("bsingles", is_alpha=False, order=1, pair_range=pair_range)

    def bdoubles(self, pair_range: tuple[int, int] | None = None) -> Iterator[dict[str, Tensor]]:
        """Stream β double excitations, chunk by chunk.

        Args:
            pair_range: Optional ``(start, stop)`` pair-index range; see
                :meth:`asingles`.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("bdoubles")
        return self._same_spin("bdoubles", is_alpha=False, order=2, pair_range=pair_range)

    def _same_spin(
        self,
        name: str,
        *,
        is_alpha: bool,
        order: int,
        pair_range: tuple[int, int] | None = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Dispatch one same-spin table kind to the deduplicated enumerator.

        Enumerates once per unique string rather than once per pair; see
        :func:`_stream_same_spin_dedup` for why that is the same row set.
        ``pair_range`` restricts the partner expansion to the owned pairs.
        """
        assert self.partners_a_offsets is not None
        assert self.partners_a_pair_idx is not None
        assert self.partners_a_spectator_key is not None
        assert self.partners_b_offsets is not None
        assert self.partners_b_pair_idx is not None
        assert self.partners_b_spectator_key is not None
        gen = _stream_same_spin_dedup(
            unique_source=self.unique_alphas if is_alpha else self.unique_betas,
            partners_offsets=self.partners_a_offsets if is_alpha else self.partners_b_offsets,
            partners_pair_idx=self.partners_a_pair_idx if is_alpha else self.partners_b_pair_idx,
            partners_spectator_key=(
                self.partners_a_spectator_key if is_alpha else self.partners_b_spectator_key
            ),
            # α blocks are keyed by β-uid and vice versa.
            spectator_stride=self.m_beta if is_alpha else self.m_alpha,
            n_orb=self.n_orb,
            n_particles=self.n_a if is_alpha else self.n_b,
            is_alpha=is_alpha,
            order=order,
            device=self.device,
            budget_bytes=self.budget_bytes,
            pair_range=pair_range,
        )
        return self._counted(name, gen)

    # ---- cross-spin ----------------------------------------------------
    def ab_doubles(self, pair_range: tuple[int, int] | None = None) -> Iterator[dict[str, Tensor]]:
        """Stream αβ double excitations (one α single × one β single), chunk by chunk.

        Args:
            pair_range: Optional ``(start, stop)`` pair-index range. When given,
                the pair chunk loop is clamped to it, so only rows whose bra
                pair ``k_bra`` lies in ``[start, stop)`` are enumerated; ``None``
                streams the full table.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("ab_doubles")
        assert self.alpha_uid_of_pair is not None
        assert self.beta_uid_of_pair is not None
        assert self.pair_keys is not None
        gen = _stream_ab_doubles(
            alpha_uid_of_pair=self.alpha_uid_of_pair,
            beta_uid_of_pair=self.beta_uid_of_pair,
            unique_alphas=self.unique_alphas,
            unique_betas=self.unique_betas,
            pair_keys=self.pair_keys,
            m_beta=self.m_beta,
            n_orb=self.n_orb,
            n_a=self.n_a,
            n_b=self.n_b,
            device=self.device,
            budget_bytes=self.budget_bytes,
            pair_range=pair_range,
            cache=self._ab_cache,
        )
        return self._counted("ab_doubles", gen)

    def _counted(self, name: str, gen: Iterator[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
        """Pass chunks through, recording the pass's total row count in ``last_nnz``."""
        total = 0
        for chunk in gen:
            total += int(chunk["k_bra"].shape[0])
            yield chunk
        self.last_nnz[name] = total

    def nnz(self) -> dict[str, int]:
        """Row count of every table kind, by streaming each one.

        A full re-enumeration of all five tables — the same work as one
        matvec's index derivation. For diagnostics and memory accounting;
        do not call it on a hot path.

        Raises:
            ValueError: the helpers carry a structural pair basis.
        """
        self._require_explicit("nnz")
        for stream in (self.asingles, self.adoubles, self.bsingles, self.bdoubles, self.ab_doubles):
            for _ in stream():
                pass
        return dict(self.last_nnz)


def generate_pair_helpers(
    pair_basis: "PairBasis | StructuralPairBasis",
    n_orb: int,
    *,
    device: "torch.device | str | None" = None,
    budget_bytes: "int | None" = None,
    block_matvec: str = "off",
    block_table_bytes: "int | None" = None,
) -> PairHelpers:
    """Build the excitation-table accessor for a pair basis.

    Cheap for an explicit :class:`PairBasis`: copies the O(N) pair arrays to
    ``device`` and builds the sorted pair-key array. No excitation enumeration
    happens here — that is deferred to the table generators on
    :class:`PairHelpers`, which run in chunks.

    For a :class:`StructuralPairBasis` only the O(M_unique) unique arrays and
    the block decomposition are built; the streamed table generators are then
    unavailable and raise, because the structural basis stores no pair list.

    Args:
        pair_basis: the explicit pair list or a structural basis.
        n_orb: number of spatial orbitals.
        device: device for the streamed chunks. Defaults to the pair basis's.
        budget_bytes: peak workspace for one enumeration chunk. ``None``
            (default) uses :data:`DEFAULT_BUDGET_BYTES`. Pass a number to pin
            it — lower on a memory-tight node, higher for fewer and larger
            chunks.
        block_matvec: ``"off"`` (default) never builds the Cartesian block
            decomposition; ``"auto"`` builds it when the usefulness gates pass;
            ``"on"`` forces the exact build (tests, A/B runs). The solver
            passes its ``gdb_kwargs["block_matvec"]``. A structural basis has
            no streamed fallback, so ``"off"`` is rejected for it.
        block_table_bytes: optional cap on the decomposition's byte footprint
            (the two pair-index maps plus the uid lists). The build is gated
            out when the estimate exceeds it; ``None`` leaves it uncapped. The
            solver passes its ``gdb_kwargs["block_table_bytes"]``. Ignored for
            a structural basis, whose footprint is inherent.

    Returns:
        A :class:`PairHelpers` bound to this basis.

    Raises:
        ValueError: a structural basis was combined with
            ``block_matvec="off"``.
    """
    if device is None:
        device = pair_basis.device
    if isinstance(device, str):
        device = torch.device(device)
    budget_bytes = resolve_budget_bytes(budget_bytes)

    if isinstance(pair_basis, StructuralPairBasis):
        if str(block_matvec).lower() == "off":
            raise ValueError(
                "a structural pair basis supports only the block matvec path; "
                "block_matvec='off' would select the streamed tables, which need "
                "the explicit per-pair arrays a structural basis does not store"
            )
        return _generate_structural_helpers(pair_basis, n_orb, device, budget_bytes)

    unique_alphas = pair_basis.unique_alphas.to(device)
    unique_betas = pair_basis.unique_betas.to(device)
    alpha_of_pair = pair_basis.alpha_of_pair.to(device)
    beta_of_pair = pair_basis.beta_of_pair.to(device)
    alpha_uid = pair_basis.alpha_uid_of_pair.to(device).to(torch.int64)
    beta_uid = pair_basis.beta_uid_of_pair.to(device).to(torch.int64)
    m_beta = int(unique_betas.shape[0])

    # pair_keys[k] = α_uid[k] * m_beta + β_uid[k]. Pairs are lex-sorted by
    # (α_uid, β_uid), so this is monotonically non-decreasing and
    # torch.searchsorted lookups are exact.
    pair_keys = alpha_uid * m_beta + beta_uid

    # Partners CSR, moved onto ``device`` for the deduplicated same-spin
    # enumerator. Its spectator uids are pre-offset by
    # ``block_id * spectator_stride`` so the payload becomes globally ascending:
    # it already ascends inside each block, and blocks sit in ascending block
    # order, so one batched ``searchsorted`` can locate a spectator inside any
    # row's block. Both strides are bounded by the unique-string counts, so the
    # product stays far below int64 range (see _stream_same_spin_dedup).
    m_alpha = int(unique_alphas.shape[0])
    p_a_off = pair_basis.partners_of_alpha_offsets.to(device).to(torch.int64)
    p_a_idx = pair_basis.partners_of_alpha_pair_idx.to(device).to(torch.int64)
    p_a_spec = pair_basis.partners_of_alpha_beta_uid.to(device).to(torch.int64)
    p_b_off = pair_basis.partners_of_beta_offsets.to(device).to(torch.int64)
    p_b_idx = pair_basis.partners_of_beta_pair_idx.to(device).to(torch.int64)
    p_b_spec = pair_basis.partners_of_beta_alpha_uid.to(device).to(torch.int64)
    a_blk = torch.repeat_interleave(
        torch.arange(m_alpha, dtype=torch.int64, device=device), p_a_off[1:] - p_a_off[:-1]
    )
    b_blk = torch.repeat_interleave(
        torch.arange(m_beta, dtype=torch.int64, device=device), p_b_off[1:] - p_b_off[:-1]
    )
    p_a_key = p_a_spec + a_blk * m_beta
    p_b_key = p_b_spec + b_blk * m_alpha

    # Electron count per spin, read off the first pair. Every pair carries the
    # same count (enforced upstream by _validate_ci_basis), so any pair works.
    n_a = int(_popcount64(alpha_of_pair[:1]).sum().item())
    n_b = int(_popcount64(beta_of_pair[:1]).sum().item())

    return PairHelpers(
        n_orb=n_orb,
        device=device,
        n_a=n_a,
        n_b=n_b,
        m_beta=m_beta,
        alpha_of_pair=alpha_of_pair,
        beta_of_pair=beta_of_pair,
        alpha_uid_of_pair=alpha_uid,
        beta_uid_of_pair=beta_uid,
        unique_alphas=unique_alphas,
        unique_betas=unique_betas,
        pair_keys=pair_keys,
        partners_a_offsets=p_a_off,
        partners_a_pair_idx=p_a_idx,
        partners_a_spectator_key=p_a_key,
        partners_b_offsets=p_b_off,
        partners_b_pair_idx=p_b_idx,
        partners_b_spectator_key=p_b_key,
        m_alpha=m_alpha,
        budget_bytes=int(budget_bytes),
        string_cache_bytes=int(budget_bytes) // 4,
        blocks=build_pair_blocks(pair_basis, mode=block_matvec, byte_cap=block_table_bytes),
    )


def _generate_structural_helpers(
    pair_basis: StructuralPairBasis,
    n_orb: int,
    device: torch.device,
    budget_bytes: int,
) -> PairHelpers:
    """Build helpers for a structural basis, without any length-N array.

    Args:
        pair_basis: the structural basis.
        n_orb: number of spatial orbitals.
        device: device for the unique arrays and the matvec workspace.
        budget_bytes: peak workspace for one enumeration chunk.

    Returns:
        Helpers whose ``structural`` field is set and whose per-pair fields
        are all ``None``.

    Raises:
        ValueError: the basis carries no blocks.
    """
    unique_alphas = pair_basis.unique_alphas.to(device)
    unique_betas = pair_basis.unique_betas.to(device)
    blocks = build_pair_blocks(pair_basis, mode="auto")
    if blocks is None:
        raise ValueError("structural_pair_basis_from_blocks produced no blocks")
    n_a = int(_popcount64(unique_alphas[:1]).sum().item())
    n_b = int(_popcount64(unique_betas[:1]).sum().item())
    return PairHelpers(
        n_orb=n_orb,
        device=device,
        n_a=n_a,
        n_b=n_b,
        m_beta=int(unique_betas.shape[0]),
        unique_alphas=unique_alphas,
        unique_betas=unique_betas,
        m_alpha=int(unique_alphas.shape[0]),
        budget_bytes=int(budget_bytes),
        string_cache_bytes=int(budget_bytes) // 4,
        blocks=blocks,
        structural=pair_basis,
    )


# ----------------------------------------------------------------------
# Chunked enumerators
# ----------------------------------------------------------------------


def _bit_positions_of_pattern(strings: Tensor, n_orb: int, n_set: int) -> Tensor:
    """Sorted set-bit positions for each of ``strings``.

    Precondition: every string has exactly ``n_set`` bits set among positions
    ``[0, n_orb)``.

    Returns a ``(N, n_set)`` int64 tensor whose row ``k`` holds the positions
    of the set bits of ``strings[k]`` in ascending order. Uses
    ``torch.nonzero`` on the (N, n_orb) occupancy matrix; its row-major output
    puts each row's positions contiguously.
    """
    n = int(strings.shape[0])
    if n == 0 or n_set == 0:
        return torch.zeros((n, n_set), dtype=torch.int64, device=strings.device)
    p_range = torch.arange(n_orb, dtype=torch.int64, device=strings.device)
    occ = ((strings.unsqueeze(-1) >> p_range.unsqueeze(0)) & 1) == 1  # (N, norb)
    nz = torch.nonzero(occ, as_tuple=False)  # (N * n_set, 2), row-major
    return nz[:, 1].view(n, n_set)


def _bit_positions_of_pattern_multilimb(strings: Tensor, n_orb: int, n_set: int) -> Tensor:
    """N-limb analogue of :func:`_bit_positions_of_pattern`.

    ``strings`` is a ``(N, n_limbs)`` stacked determinant tensor (limb ``k``
    holds bits ``[64k, 64k+64)``) rather than the single-limb ``(N,)`` tensor
    :func:`_bit_positions_of_pattern` takes, so ``n_orb`` may exceed 64.

    Precondition: every row has exactly ``n_set`` bits set among positions
    ``[0, n_orb)``.

    Returns a ``(N, n_set)`` int64 tensor whose row ``k`` holds the positions
    of the set bits of ``strings[k]`` in ascending order.
    """
    n = int(strings.shape[0])
    if n == 0 or n_set == 0:
        return torch.zeros((n, n_set), dtype=torch.int64, device=strings.device)
    p_range = torch.arange(n_orb, dtype=torch.int64, device=strings.device)
    limb_idx = p_range >> 6
    bit_pos = p_range & 63
    # (N, n_orb, n_limbs) -> gather the limb holding each position -> (N, n_orb)
    limbs = strings.unsqueeze(1).expand(n, n_orb, strings.shape[-1])
    limb = torch.gather(limbs, -1, limb_idx.expand(n, n_orb).unsqueeze(-1)).squeeze(-1)
    occ = ((limb >> bit_pos.unsqueeze(0)) & 1) == 1  # (N, n_orb)
    nz = torch.nonzero(occ, as_tuple=False)  # (N * n_set, 2), row-major
    return nz[:, 1].view(n, n_set)


def _bit_mask_rows(positions: Tensor, n_limbs: int) -> Tensor:
    """``(N,)`` bit positions to a ``(N, n_limbs)`` int64 mask with that bit set.

    The limb index is ``position // 64``, so the result XORs straight into a
    ``(N, n_limbs)`` limb matrix.
    """
    limb = positions >> 6
    bit = positions & 63
    mask = torch.zeros((*positions.shape, n_limbs), dtype=torch.int64, device=positions.device)
    mask.scatter_(-1, limb.unsqueeze(-1), (torch.ones_like(bit) << bit).unsqueeze(-1))
    return mask


def _bit_mask_below_positions(positions: Tensor, n_limbs: int) -> Tensor:
    """``(N,)`` bit positions to a ``(N, n_limbs)`` int64 mask of lower bits.

    Row ``n`` has every bit in ``[0, positions[n])`` set. Limb ``k`` covers
    positions ``[64k, 64k + 64)``, so a limb entirely below ``positions[n]``
    is all ones and a limb at or above it is zero. Used to count the bits of a
    limb matrix below an orbital without materialising a ``(N, n_orb)``
    occupancy matrix.
    """
    device = positions.device
    shift = positions.unsqueeze(-1) - 64 * torch.arange(
        n_limbs, dtype=torch.int64, device=device
    ).unsqueeze(0)
    low = (torch.ones_like(shift) << shift.clamp(0, 63)) - 1
    return torch.where(shift >= 64, -torch.ones_like(shift), low)


def _complement_mask(strings: Tensor, n_orb: int) -> Tensor:
    """Mask of the ``n_orb`` valid bits, shaped to XOR against ``strings``.

    A 1-D (single-word) ``strings`` gets the legacy scalar mask ``(1 << n_orb) - 1``
    so that expression stays bit-identical for ``n_orb < 64``; a single word
    holds no bit at position ``>= 64``, so the mask is all ones for
    ``n_orb >= 64``. A limb matrix gets one mask word per limb. Limb ``k``
    covers bit positions ``[64k, 64k + 64)``, so its mask word holds the part
    of ``[0, n_orb)`` inside that window — a limb entirely above ``n_orb`` (a
    caller-padded high word) gets zero.
    """
    if strings.dim() == 1:
        # A single word holds at most 64 bits, so for n_orb >= 64 the mask is
        # the all-ones int64 pattern rather than an unrepresentable 1 << n_orb.
        return torch.as_tensor(
            (1 << n_orb) - 1 if n_orb < 64 else -1, dtype=torch.int64, device=strings.device
        )
    n_limbs = int(strings.shape[-1])
    mask = torch.full((n_limbs,), -1, dtype=torch.int64, device=strings.device)
    for k in range(n_limbs):
        bits = min(int(n_orb), 64 * (k + 1)) - 64 * k
        if bits <= 0:
            mask[k] = 0
        elif bits < 64:
            mask[k] = (1 << bits) - 1
    return mask


def _sort_strings(strings: Tensor) -> tuple[Tensor, Tensor]:
    """Sort a batch of strings lexicographically, limb 0 compared first.

    Returns ``(sorted, order)`` like ``torch.sort``. Single-word strings use
    ``torch.sort`` unchanged; limb matrices are stably sorted column by column
    from the last limb to limb 0, which yields the lexicographic row order
    :func:`_found_in_multilimb` expects.
    """
    if strings.dim() == 1:
        return torch.sort(strings)
    order = torch.arange(strings.shape[0], dtype=torch.int64, device=strings.device)
    for k in range(strings.shape[1] - 1, -1, -1):
        order = order[torch.argsort(strings[order, k], stable=True)]
    return strings[order], order


def _lookup_pair_keys_torch(pair_keys: Tensor, query_keys: Tensor) -> tuple[Tensor, Tensor]:
    """Batched ``(α_uid, β_uid) → pair_idx`` lookup via ``torch.searchsorted``.

    ``pair_keys`` is a sorted int64 tensor of length ``N`` with values
    ``α_uid[k] * m_beta + β_uid[k]``. Returns ``(mask, pair_idx)`` where
    ``mask[i]`` is True iff ``query_keys[i]`` is a real pair, in which case
    ``pair_idx[i]`` is that pair's index. Values of ``pair_idx`` where
    ``~mask`` are meaningless and must be ignored.
    """
    if query_keys.numel() == 0 or pair_keys.numel() == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=query_keys.device)
        return torch.zeros(0, dtype=torch.bool, device=query_keys.device), empty
    pos = torch.searchsorted(pair_keys, query_keys)
    n = pair_keys.shape[0]
    found = (pair_keys[torch.clamp(pos, max=n - 1)] == query_keys) & (pos < n)
    return found, pos


def _found_in(unique: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """``(mask, uid)`` for each of ``values`` looked up in sorted ``unique``."""
    m = int(unique.shape[0])
    uid = torch.searchsorted(unique, values)
    found = (unique[torch.clamp(uid, max=max(m - 1, 0))] == values) & (uid < m)
    return found, uid


def _found_in_multilimb(unique: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """N-limb analogue of :func:`_found_in`.

    ``unique`` is a ``(M, n_limbs)`` tensor sorted lexicographically ascending
    (e.g. via ``torch.unique(..., dim=0)``) and ``values`` is a ``(N, n_limbs)``
    query tensor. Returns ``(mask, uid)`` where ``mask[i]`` is True iff
    ``values[i]`` is a row of ``unique``, in which case ``uid[i]`` is that
    row's index.

    ``torch.searchsorted`` has no multi-column form, so membership is found by
    a vectorized binary search comparing rows lexicographically (limb 0 most
    significant, matching ``torch.unique(dim=0)``'s own sort order). Dispatches
    to :func:`_found_in` when ``n_limbs == 1``: the general bisection is an
    order of magnitude slower than ``torch.searchsorted`` at that width, and
    every pre-existing (single-limb) caller must stay on the fast path.
    """
    n_limbs = unique.shape[-1]
    if n_limbs == 1:
        return _found_in(unique.squeeze(-1), values.squeeze(-1))

    m = int(unique.shape[0])
    n = int(values.shape[0])
    device = unique.device
    if n == 0 or m == 0:
        return (
            torch.zeros(n, dtype=torch.bool, device=device),
            torch.zeros(n, dtype=torch.int64, device=device),
        )

    # Vectorized lexicographic upper_bound: find, for each query row, the
    # smallest index idx such that unique[idx] > query (lexicographically).
    # Invariant: unique[0:lo] <= query, unique[hi:m] > query.
    lo = torch.zeros(n, dtype=torch.int64, device=device)
    hi = torch.full((n,), m, dtype=torch.int64, device=device)
    iters = max(1, math.ceil(math.log2(max(m, 2))) + 1)
    for _ in range(iters):
        mid = (lo + hi) // 2
        cand = unique[torch.clamp(mid, max=m - 1)]
        lt = torch.zeros(n, dtype=torch.bool, device=device)
        eq = torch.ones(n, dtype=torch.bool, device=device)
        for k in range(n_limbs):
            lt = lt | (eq & (cand[:, k] < values[:, k]))
            eq = eq & (cand[:, k] == values[:, k])
        le = lt | eq  # unique[mid] <= query
        lo = torch.where(le, mid + 1, lo)
        hi = torch.where(le, hi, mid)
    # lo - 1 is the last index with unique[idx] <= query; it's an exact
    # match iff that index exists and the row actually equals query.
    uid = lo - 1
    uid_c = torch.clamp(uid, min=0, max=m - 1)
    found = (uid >= 0) & (unique[uid_c] == values).all(dim=-1)
    return found, uid_c


def _dispatch_found_in(unique: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Dispatch to single-limb or multi-limb lookup based on ``unique`` shape."""
    if unique.dim() >= 2:
        return _found_in_multilimb(unique, values)
    return _found_in(unique, values)


def _dispatch_bit_positions(strings: Tensor, n_orb: int, n_set: int) -> Tensor:
    """Dispatch to single-limb or multi-limb bit positions based on ``strings`` shape."""
    if strings.dim() >= 2:
        return _bit_positions_of_pattern_multilimb(strings, n_orb, n_set)
    return _bit_positions_of_pattern(strings, n_orb, n_set)


def _triu_indices(n: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
    """Sorted upper-triangular index pairs ``(i, j)`` with ``0 <= i < j < n``."""
    if n < 2:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty
    tri = torch.triu_indices(n, n, offset=1, device=device)
    return tri[0].to(torch.int64), tri[1].to(torch.int64)


def _stream_same_spin(
    *,
    source_of_pair: Tensor,
    unique_target: Tensor,
    pair_keys: Tensor,
    m_beta: int,
    n_orb: int,
    n_particles: int,
    is_alpha: bool,
    order: int,
    device: torch.device,
    budget_bytes: int,
) -> Iterator[dict[str, Tensor]]:
    """Yield chunks of same-spin single (``order=1``) or double (``order=2``) excitations.

    For each pair ``k`` in the chunk, enumerate every way to annihilate
    ``order`` occupied orbitals of its source string and create ``order``
    empty ones, keep the candidates whose excited string is in
    ``unique_target``, then keep those whose resulting pair (excited string
    with ``k``'s spectator unchanged) is in the basis.

    Orbital indices are returned as spin-orbitals: ``2·spatial`` for α,
    ``2·spatial + 1`` for β. Supports multi-limb strings (``n_orb > 64``).
    """
    n_pairs = int(source_of_pair.shape[0])
    n_empty = n_orb - n_particles
    if n_pairs == 0 or n_particles < order or n_empty < order:
        return
    if order == 1:
        n_per_pair = n_particles * n_empty
    else:
        n_per_pair = (n_particles * (n_particles - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    if n_per_pair == 0:
        return

    chunk = _chunk_pairs(n_per_pair, "singles" if order == 1 else "doubles", budget_bytes)
    one = torch.tensor(1, dtype=torch.int64, device=device)
    string_mask = _complement_mask(source_of_pair, n_orb)
    n_limbs = 1 if source_of_pair.dim() == 1 else int(source_of_pair.shape[-1])
    spin_offset = 0 if is_alpha else 1
    if order == 2:
        an_i, an_j = _triu_indices(n_particles, device=device)
        cr_i, cr_j = _triu_indices(n_empty, device=device)

    for cs in range(0, n_pairs, chunk):
        ce = min(cs + chunk, n_pairs)
        src_chunk = source_of_pair[cs:ce]
        # Occupied / empty orbital positions, computed for this chunk only —
        # holding them for all N would itself be O(N · n_orb).
        occ_pos = _dispatch_bit_positions(src_chunk, n_orb, n_particles)
        empty_pos = _dispatch_bit_positions(src_chunk ^ string_mask, n_orb, n_empty)

        if order == 1:
            # (c, n_particles, n_empty) grid: annihilate occ i, create empty a.
            an0 = occ_pos.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
            cr0 = empty_pos.unsqueeze(-2).expand(-1, n_particles, -1).reshape(-1)
            # Unused for singles; a length-0 tensor rather than None so the
            # second orbital pair keeps one type through the loop below.
            an1 = cr1 = torch.zeros(0, dtype=torch.int64, device=device)
        else:
            # (c, n_occ_pairs, n_empty_pairs) grid over unordered index pairs.
            n_op = an_i.shape[0]
            n_ep = cr_i.shape[0]
            an0 = occ_pos[:, an_i].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            an1 = occ_pos[:, an_j].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            cr0 = empty_pos[:, cr_i].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
            cr1 = empty_pos[:, cr_j].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)

        pair_idx = (
            torch
            .arange(cs, ce, dtype=torch.int64, device=device)
            .unsqueeze(-1)
            .expand(-1, n_per_pair)
            .reshape(-1)
        )
        if source_of_pair.dim() == 1:
            target = source_of_pair[pair_idx] ^ (one << an0) ^ (one << cr0)
            if order == 2:
                target = target ^ (one << an1) ^ (one << cr1)
        else:
            target = (
                source_of_pair[pair_idx]
                ^ _bit_mask_rows(an0, n_limbs)
                ^ _bit_mask_rows(cr0, n_limbs)
            )
            if order == 2:
                target = target ^ _bit_mask_rows(an1, n_limbs) ^ _bit_mask_rows(cr1, n_limbs)

        tgt_found, tgt_uid = _dispatch_found_in(unique_target, target)
        if not tgt_found.any():
            continue
        pair_idx, tgt_uid = pair_idx[tgt_found], tgt_uid[tgt_found]
        an0, cr0 = an0[tgt_found], cr0[tgt_found]
        if order == 2:
            an1, cr1 = an1[tgt_found], cr1[tgt_found]

        # The spectator string is unchanged, so its uid is already encoded in
        # the bra's pair key: key = α_uid * m_beta + β_uid. Recover it with a
        # divmod instead of another lookup.
        bra_key = pair_keys[pair_idx]
        if is_alpha:
            query = tgt_uid * m_beta + (bra_key % m_beta)
        else:
            query = (bra_key // m_beta) * m_beta + tgt_uid
        mask, k_ket = _lookup_pair_keys_torch(pair_keys, query)
        if not mask.any():
            continue

        # The ket is the bra with an -> cr; the emitted labels describe the
        # ket -> bra excitation (see the module docstring), so they swap here.
        out = {
            "k_bra": pair_idx[mask],
            "k_ket": k_ket[mask],
            "cr0": 2 * an0[mask] + spin_offset,
            "an0": 2 * cr0[mask] + spin_offset,
        }
        if order == 2:
            out["cr1"] = 2 * an1[mask] + spin_offset
            out["an1"] = 2 * cr1[mask] + spin_offset
        yield out


# Bytes per expanded same-spin row in the dedup path's expansion stage. Each row
# carries k_bra, k_ket, up to four orbital indices and the intersection
# bookkeeping (flat index, excitation id, within-block offset, both block bases
# and lengths, the search key and its result). Sized like
# _BYTES_PER_CROSS_ROW: measured rather than counted, leaning high, since torch
# keeps roughly twice as much alive as the visible locals suggest.
_BYTES_PER_EXPAND_ROW = 24 * 8


def _block_pair_bounds(
    partners_offsets: Tensor,
    partners_pair_idx: Tensor,
    block_ids: Tensor,
    pair_start: int,
    pair_stop: int,
) -> tuple[Tensor, Tensor]:
    """In-range sub-slice of each sorted partner block of a CSR.

    ``partners_pair_idx`` lists, for every unique string, the pair indices of
    the pairs carrying that string in ascending order. That holds for both
    spins: the α blocks are contiguous runs of the pair index, and each β block
    is ordered by α uid, which is pair-index order at fixed β (pairs are
    lex-sorted by ``(α_uid, β_uid)``). ``block_ids`` selects the blocks to
    query. Torch has no ``searchsorted`` over blocks with different bases, so
    this runs a batched binary search per block; memory is O(len(block_ids)).

    Args:
        partners_offsets: CSR offsets, length ``m_unique + 1``.
        partners_pair_idx: CSR pair-index payload, sorted within each block.
            It holds every pair index exactly once, so its length is N.
        block_ids: Unique-string indices whose blocks are queried.
        pair_start: First pair index kept (inclusive).
        pair_stop: One past the last pair index kept (exclusive).

    Returns:
        ``(start, length)`` per block: the global index of the first in-range
        entry and the number of in-range entries (``0`` when none).
    """
    n_blocks = int(block_ids.shape[0])
    device = partners_offsets.device
    if n_blocks == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        return empty, empty

    starts = partners_offsets[block_ids]
    lengths = partners_offsets[block_ids + 1] - starts
    n_entries = int(partners_pair_idx.shape[0])
    max_len = int(lengths.max())

    def _lower_bound(target: int) -> Tensor:
        """First position inside each block whose pair index is ``>= target``."""
        lo = torch.zeros(n_blocks, dtype=torch.int64, device=device)
        hi = lengths.clone()
        for _ in range(max_len.bit_length()):
            mid = (lo + hi) // 2
            probe = torch.minimum(mid, (lengths - 1).clamp_min(0))
            vals = partners_pair_idx[torch.clamp(starts + probe, max=n_entries - 1)]
            advance = (mid < lengths) & (vals < target)
            lo = torch.where(advance, mid + 1, lo)
            hi = torch.where(advance, hi, mid)
        return lo

    lo = _lower_bound(pair_start)
    hi = _lower_bound(pair_stop)
    return starts + lo, hi - lo


def _stream_same_spin_dedup(
    *,
    unique_source: Tensor,
    partners_offsets: Tensor,
    partners_pair_idx: Tensor,
    partners_spectator_key: Tensor,
    spectator_stride: int,
    n_orb: int,
    n_particles: int,
    is_alpha: bool,
    order: int,
    device: torch.device,
    budget_bytes: int,
    pair_range: tuple[int, int] | None = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield same-spin excitation chunks, enumerating once per *unique string*.

    Why this exists
    ---------------
    :func:`_stream_same_spin` enumerates candidates once per *pair*. But a
    same-spin excitation of pair ``k`` touches only ``k``'s own spin string —
    the spectator is untouched and cannot change which candidates exist, nor
    which of them land in the basis. So every pair sharing an α-string
    re-derives an identical candidate set and repeats identical lookups.

    In a GDB basis each unique string appears in many pairs. The truncated
    Cartesian product the scaling benchmark uses has
    ``n_pairs / m_unique = sqrt(n_pairs)`` pairs per string — ~220x at 50k
    determinants, ~2200x at 5M — so that repetition, not the filtering,
    dominates the same-spin passes. At norb=36 with 27 electrons per spin the
    doubles table costs 12,636 candidates per pair against 243 for singles,
    which is ~96% of all same-spin candidate work.

    This enumerator does the candidate work once per unique string and then
    *expands* survivors over the pairs containing them, taking the cost from
    ``O(n_pairs · n_per_string)`` to ``O(m_unique · n_per_string + nnz)``.

    How the expansion works
    -----------------------
    A surviving excitation ``u -> v`` contributes one row for every spectator
    string shared by ``u`` and ``v``: pair ``(u, s)`` couples to ``(v, s)``
    exactly when both are in the basis. The partners CSR lists, per unique
    string, the pairs carrying it with their spectator uids ascending, so the
    shared spectators are the **sorted intersection** of ``u``'s and ``v``'s
    blocks.

    That intersection replaces the keyed ``searchsorted`` into ``pair_keys``:
    searching ``u``'s spectators inside ``v``'s block yields membership and the
    ket pair's position together. The search is batched across all rows by
    pre-offsetting each entry with ``block_id * spectator_stride``, which makes
    the CSR payload globally ascending (it ascends within each block already,
    and blocks are laid out in ascending block order) so one
    :func:`torch.searchsorted` serves every row at once.

    Row order differs from :func:`_stream_same_spin`, which walks pairs in
    ascending ``k_bra``. Here rows group by source string, so ``k_bra`` is not
    monotonic across the pass. Every consumer scatter-adds into an output
    vector (:mod:`~torch_sci_solver.torch_impl.mult_pairs`,
    :mod:`~torch_sci_solver.torch_impl.rdm_pairs`), so the row *set* is what
    matters, not its order.

    Restricting to a pair range
    ---------------------------
    With ``pair_range=(start, stop)`` only bra partners whose pair index lies
    in ``[start, stop)`` are expanded. Each bra block of the partners CSR is
    sorted, so its in-range entries are one contiguous sub-slice, located with
    :func:`_block_pair_bounds` (a batched binary search, since the block bases
    differ). The candidate work per unique string is unchanged, but the
    expansion — the O(nnz) part of the pass — shrinks to the owned rows, and
    every yielded ``k_bra`` is in range. ``None`` reproduces the full pass.

    Args:
        unique_source: sorted unique strings of the excited spin. The excited
            string must land back in this same array to survive.
        partners_offsets: CSR offsets into the partner arrays, length
            ``m_unique + 1``.
        partners_pair_idx: pair index of each partner entry.
        partners_spectator_key: spectator uid of each partner entry, already
            offset by ``block_id * spectator_stride`` so the array is globally
            ascending. See :func:`generate_pair_helpers`.
        spectator_stride: the stride used for that offsetting — an exclusive
            upper bound on the raw spectator uids.
        n_orb: number of spatial orbitals.
        n_particles: electrons in the excited spin.
        is_alpha: True for α (spin-orbital ``2·spatial``), False for β
            (``2·spatial + 1``).
        order: 1 for singles, 2 for doubles.
        device: device for the yielded tensors.
        budget_bytes: peak workspace for one chunk.
        pair_range: optional ``(start, stop)`` bra-pair range to restrict the
            expansion to; ``None`` yields every row.

    Yields:
        The same dicts :func:`_stream_same_spin` yields, with the same keys.
    """
    m_unique = int(unique_source.shape[0])
    n_empty = n_orb - n_particles
    if m_unique == 0 or n_particles < order or n_empty < order:
        return
    if order == 1:
        n_per_str = n_particles * n_empty
    else:
        n_per_str = (n_particles * (n_particles - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    if n_per_str == 0:
        return

    # Two stages are live at once (the per-string candidate grid, then the
    # expansion slice), so each gets half the budget — the split rationale is
    # the same as in _stream_ab_doubles.
    stage_budget = max(1, int(budget_bytes) // 2)
    chunk = _chunk_pairs(n_per_str, "singles" if order == 1 else "doubles", stage_budget)
    expand_cap = max(1, stage_budget // _BYTES_PER_EXPAND_ROW)
    one = torch.tensor(1, dtype=torch.int64, device=device)
    string_mask = _complement_mask(unique_source, n_orb)
    n_limbs = 1 if unique_source.dim() == 1 else int(unique_source.shape[-1])
    spin_offset = 0 if is_alpha else 1
    if order == 2:
        an_i, an_j = _triu_indices(n_particles, device=device)
        cr_i, cr_j = _triu_indices(n_empty, device=device)

    blk_start = partners_offsets[:-1]
    blk_len = partners_offsets[1:] - blk_start
    stride_t = torch.tensor(int(spectator_stride), dtype=torch.int64, device=device)

    if pair_range is not None:
        pair_start, pair_stop = int(pair_range[0]), int(pair_range[1])
        n_pairs = int(partners_pair_idx.shape[0])
        if not 0 <= pair_start <= pair_stop <= n_pairs:
            raise ValueError(f"pair_range {pair_range} is out of bounds for {n_pairs} pairs")

    for cs in range(0, m_unique, chunk):
        ce = min(cs + chunk, m_unique)
        src = unique_source[cs:ce]
        occ_pos = _dispatch_bit_positions(src, n_orb, n_particles)
        empty_pos = _dispatch_bit_positions(src ^ string_mask, n_orb, n_empty)

        if order == 1:
            an0 = occ_pos.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
            cr0 = empty_pos.unsqueeze(-2).expand(-1, n_particles, -1).reshape(-1)
            an1 = cr1 = torch.zeros(0, dtype=torch.int64, device=device)
        else:
            n_op = an_i.shape[0]
            n_ep = cr_i.shape[0]
            an0 = occ_pos[:, an_i].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            an1 = occ_pos[:, an_j].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            cr0 = empty_pos[:, cr_i].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
            cr1 = empty_pos[:, cr_j].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)

        src_uid = (
            torch
            .arange(cs, ce, dtype=torch.int64, device=device)
            .unsqueeze(-1)
            .expand(-1, n_per_str)
            .reshape(-1)
        )
        if unique_source.dim() == 1:
            target = unique_source[src_uid] ^ (one << an0) ^ (one << cr0)
            if order == 2:
                target = target ^ (one << an1) ^ (one << cr1)
        else:
            target = (
                unique_source[src_uid] ^ _bit_mask_rows(an0, n_limbs) ^ _bit_mask_rows(cr0, n_limbs)
            )
            if order == 2:
                target = target ^ _bit_mask_rows(an1, n_limbs) ^ _bit_mask_rows(cr1, n_limbs)

        tgt_found, tgt_uid = _dispatch_found_in(unique_source, target)
        if not tgt_found.any():
            continue
        src_uid, tgt_uid = src_uid[tgt_found], tgt_uid[tgt_found]
        an0, cr0 = an0[tgt_found], cr0[tgt_found]
        if order == 2:
            an1, cr1 = an1[tgt_found], cr1[tgt_found]

        # Expansion. Walk each surviving excitation's *bra* partner block; a
        # batched search of those spectators inside the *ket* block gives both
        # membership and the ket pair's position.
        if pair_range is None:
            bra_start = blk_start[src_uid]
            bra_len = blk_len[src_uid]
        else:
            all_start, all_len = _block_pair_bounds(
                partners_offsets,
                partners_pair_idx,
                torch.arange(cs, ce, dtype=torch.int64, device=device),
                pair_start,
                pair_stop,
            )
            sel = src_uid - cs
            bra_start = all_start[sel]
            bra_len = all_len[sel]
        exc_cum = torch.cumsum(bra_len, 0)
        total = int(exc_cum[-1].item()) if exc_cum.numel() else 0
        if total == 0:
            continue

        for lo in range(0, total, expand_cap):
            hi = min(lo + expand_cap, total)
            flat = torch.arange(lo, hi, dtype=torch.int64, device=device)
            # Excitation each flat slot belongs to, and its offset inside that
            # excitation's bra block. Empty blocks are stepped over naturally:
            # their exc_cum entry repeats, so searchsorted skips past them.
            e = torch.searchsorted(exc_cum, flat, right=True)
            pos = flat - (exc_cum[e] - bra_len[e])

            bra_at = bra_start[e] + pos
            # Raw spectator uid of the bra entry, recovered from its global key.
            spec = partners_spectator_key[bra_at] - src_uid[e] * stride_t

            ket_uid = tgt_uid[e]
            # Same global-key space as the stored array, so one batched search
            # over the whole CSR lands inside the ket block by construction.
            want = ket_uid * stride_t + spec
            found_at = torch.searchsorted(partners_spectator_key, want)
            # searchsorted returns an insertion point, so an absent spectator
            # can still land inside the ket block. Compare the key to confirm a
            # real hit; clamp first so the gather is always in bounds.
            n_part = partners_spectator_key.shape[0]
            found_at = torch.clamp(found_at, max=n_part - 1)
            ok = partners_spectator_key[found_at] == want
            if not ok.any():
                continue
            e_ok = e[ok]
            out = {
                "k_bra": partners_pair_idx[bra_at[ok]],
                "k_ket": partners_pair_idx[found_at[ok]],
                # ket -> bra labels: the swap of the bra -> ket enumeration.
                "cr0": 2 * an0[e_ok] + spin_offset,
                "an0": 2 * cr0[e_ok] + spin_offset,
            }
            if order == 2:
                out["cr1"] = 2 * an1[e_ok] + spin_offset
                out["an1"] = 2 * cr1[e_ok] + spin_offset
            yield out


def _stream_ab_doubles(
    *,
    alpha_uid_of_pair: Tensor,
    beta_uid_of_pair: Tensor,
    unique_alphas: Tensor,
    unique_betas: Tensor,
    pair_keys: Tensor,
    m_beta: int,
    n_orb: int,
    n_a: int,
    n_b: int,
    device: torch.device,
    budget_bytes: int,
    pair_range: tuple[int, int] | None = None,
    cache: "_AbCache | None" = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield chunks of αβ double excitations: one α single × one β single.

    Both excited strings must exist in their spin's unique array, and the
    resulting ``(α_ket, β_ket)`` must be a pair in the basis.

    Enumerate per unique string, not per pair
    -----------------------------------------
    The α-side survivor set depends only on the pair's α string, and the β-side
    only on its β string. A pair basis repeats strings heavily (the block
    product the scaling benchmark uses has tens to thousands of pairs per
    unique string), so enumerating per pair re-derives the same candidate grid
    and repeats the same unique-array lookups once per partner. This pass is
    the dominant one (~70% of candidate work), so it enumerates each spin's
    survivors once per *unique string* in the pair chunk and expands them over
    the pairs by uid. The cross product and the ``pair_keys`` membership filter
    stay per pair — that is the true, irreducible row set.

    ``cache`` memoises a chunk's per-string survivors across Davidson matvecs
    (the chunk layout is fixed for a given pair range and budget), bounded by
    :class:`_AbCache`. A miss recomputes; a full cache stops inserting.

    Filter each spin *before* crossing them
    ---------------------------------------
    This is the dominant pass — at norb=36 with 27 electrons per spin there are
    ``n_per_a * n_per_b`` = 243 * 243 = 59,049 αβ candidates per determinant,
    against 243 same-spin singles and 12,636 same-spin doubles, so ~70% of all
    candidate work happens here.

    The obvious implementation builds the full ``n_per_a x n_per_b`` product and
    then tests membership on it. That is wasteful, because ``a_target`` depends
    only on *which α move* was made — the β choice cannot change it. So the same
    243 α lookups get repeated 243 times, and likewise for β: 59,049 lookups per
    determinant where 486 suffice.

    So: enumerate and filter each spin on its own ``(chunk, n_per_x)`` grid, then
    cross only the survivors. On a sparse basis almost nothing survives (with a
    few thousand α strings in the basis out of C(36,27), an α move lands in-basis
    with probability ~1e-4), so the product is essentially never materialized —
    which cuts the peak memory this pass needs as well as its work.

    The cross product is still bounded, because a *densely* connected basis (a
    fully expanded Cartesian product, say) has every candidate surviving and the
    survivor cross product is as large as the naive one. Its flat index space is
    therefore walked in slices of at most ``cross_cap`` rows.

    With ``pair_range=(start, stop)`` the pair chunk loop is clamped to that
    range, so only entries whose bra pair ``k_bra`` lies in ``[start, stop)``
    are enumerated. The per-pair filtering is unchanged, so this yields
    exactly the full pass's rows with ``k_bra`` in range, in pair order.

    Args:
        alpha_uid_of_pair: ``(N,)`` α-uid of each pair (index into
            ``unique_alphas``).
        beta_uid_of_pair: ``(N,)`` β-uid of each pair.
        unique_alphas: sorted unique α strings.
        unique_betas: sorted unique β strings.
        pair_keys: sorted ``α_uid * m_beta + β_uid`` lookup keys.
        m_beta: number of unique β strings.
        n_orb: number of spatial orbitals.
        n_a: α electron count.
        n_b: β electron count.
        device: device for the yielded tensors.
        budget_bytes: peak workspace for one chunk.
        pair_range: optional ``(start, stop)`` bra-pair range to enumerate;
            ``None`` enumerates every pair.
        cache: optional :class:`_AbCache` memoising per-string survivors across
            matvecs; ``None`` recomputes every pass.

    Yields:
        Dicts with ``k_bra``, ``k_ket`` and the four orbital indices.
    """
    n_pairs = int(alpha_uid_of_pair.shape[0])
    if pair_range is None:
        pair_first, pair_last = 0, n_pairs
    else:
        pair_first, pair_last = int(pair_range[0]), int(pair_range[1])
        if not 0 <= pair_first <= pair_last <= n_pairs:
            raise ValueError(f"pair_range {pair_range} is out of bounds for {n_pairs} pairs")
    n_a_empty, n_b_empty = n_orb - n_a, n_orb - n_b
    n_per_a, n_per_b = n_a * n_a_empty, n_b * n_b_empty
    if n_pairs == 0 or n_per_a * n_per_b == 0:
        return

    # Two stages are live at once — the per-spin grids stay resident while the
    # survivor cross product is walked — so each gets half the budget rather
    # than all of it, keeping their sum within it.
    #
    # The outer chunk now only has to hold the two per-spin grids, which cost
    # ``n_per_a + n_per_b`` per pair instead of their product, so for a given
    # budget the chunk is far larger and there are far fewer of them.
    stage_budget = max(1, int(budget_bytes) // 2)
    chunk = _chunk_pairs(n_per_a + n_per_b, "ab", stage_budget)
    cross_cap = max(1, stage_budget // _BYTES_PER_CROSS_ROW)
    one = torch.tensor(1, dtype=torch.int64, device=device)

    def _spin_survivors(
        strings: Tensor, n_elec: int, n_empty: int, n_per: int, unique: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-spin single excitations that land in ``unique``, grouped by input string.

        Returns ``(count_per_string, uid, an, cr)`` where the three flat arrays
        are the survivors in input-major order, so string ``r``'s survivors
        occupy ``[offsets[r], offsets[r] + count[r])`` for an exclusive-cumsum
        offset. Supports multi-limb strings (``n_orb > 64``).
        """
        occ = _dispatch_bit_positions(strings, n_orb, n_elec)
        empty = _dispatch_bit_positions(strings ^ _complement_mask(strings, n_orb), n_orb, n_empty)
        rows = strings.shape[0]
        an = occ.unsqueeze(-1).expand(-1, -1, n_empty).reshape(rows, n_per)
        cr = empty.unsqueeze(-2).expand(-1, n_elec, -1).reshape(rows, n_per)
        if strings.dim() == 1:
            target = strings.unsqueeze(-1) ^ (one << an) ^ (one << cr)
            found, uid = _dispatch_found_in(unique, target)
        else:
            n_limbs = int(strings.shape[-1])
            target = (
                strings.unsqueeze(-2) ^ _bit_mask_rows(an, n_limbs) ^ _bit_mask_rows(cr, n_limbs)
            ).reshape(-1, n_limbs)
            found, uid = _dispatch_found_in(unique, target)
            found = found.view(rows, n_per)
            uid = uid.view(rows, n_per)
        # nonzero() on the row-major flattening returns ascending indices, so the
        # survivors come out grouped by string and ordered within each string.
        sel = found.reshape(-1).nonzero(as_tuple=True)[0]
        return (
            found.sum(1),
            uid.reshape(-1)[sel],
            an.reshape(-1)[sel],
            cr.reshape(-1)[sel],
        )

    def _chunk_survivors(
        uids: Tensor,
        n_elec: int,
        n_empty: int,
        n_per: int,
        unique: Tensor,
        cache_key: tuple[int, int, bool],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-pair single survivors, enumerated once per unique string.

        ``uids`` indexes ``unique``. The chunk's distinct uids are enumerated
        once (``n_unique · n_per`` candidate grids and lookups instead of
        ``n_pairs · n_per``) and expanded back over the pairs by uid. ``cache``
        memoises the per-unique-string result across matvecs, within its byte
        budget.
        """
        u_uids, inv = torch.unique(uids, return_inverse=True)
        cached = None if cache is None else cache.entries.get(cache_key)
        if cached is not None:
            u_cnt, u_uid_s, u_an_s, u_cr_s = cached
        else:
            u_cnt, u_uid_s, u_an_s, u_cr_s = _spin_survivors(
                unique[u_uids], n_elec, n_empty, n_per, unique
            )
            if cache is not None and not cache.full:
                nbytes = sum(int(t.numel()) * 8 for t in (u_cnt, u_uid_s, u_an_s, u_cr_s))
                if nbytes <= cache.budget_bytes - cache.used_bytes:
                    cache.entries[cache_key] = (u_cnt, u_uid_s, u_an_s, u_cr_s)
                    cache.used_bytes += nbytes
                elif cache.used_bytes == 0:
                    # One chunk alone exceeds the budget: caching cannot pay.
                    cache.full = True
        cnt = u_cnt[inv]
        total = int(cnt.sum().item())
        if total == 0:
            return cnt, u_uid_s[:0], u_an_s[:0], u_cr_s[:0]
        u_off = torch.cumsum(u_cnt, 0) - u_cnt
        pair_off = u_off[inv]
        flat = torch.arange(total, dtype=torch.int64, device=device)
        # Inclusive cumsum + right=True steps over pairs with zero survivors,
        # the same convention as the cross-product row mapping below.
        pair_cum = torch.cumsum(cnt, 0)
        row = torch.searchsorted(pair_cum, flat, right=True)
        src = pair_off[row] + (flat - (pair_cum[row] - cnt[row]))
        return cnt, u_uid_s[src], u_an_s[src], u_cr_s[src]

    for cs in range(pair_first, pair_last, chunk):
        ce = min(cs + chunk, pair_last)
        a_cnt, a_uid_s, a_an_s, a_cr_s = _chunk_survivors(
            alpha_uid_of_pair[cs:ce],
            n_a,
            n_a_empty,
            n_per_a,
            unique_alphas,
            (cs, ce, True),
        )
        b_cnt, b_uid_s, b_an_s, b_cr_s = _chunk_survivors(
            beta_uid_of_pair[cs:ce],
            n_b,
            n_b_empty,
            n_per_b,
            unique_betas,
            (cs, ce, False),
        )

        # Cross only the survivors: pair r contributes a_cnt[r] * b_cnt[r] rows.
        n_cross = a_cnt * b_cnt
        total = int(n_cross.sum().item())
        if total == 0:
            continue
        cross_cum = torch.cumsum(n_cross, 0)
        a_off = torch.cumsum(a_cnt, 0) - a_cnt
        b_off = torch.cumsum(b_cnt, 0) - b_cnt

        for lo in range(0, total, cross_cap):
            hi = min(lo + cross_cap, total)
            flat = torch.arange(lo, hi, dtype=torch.int64, device=device)
            # Which pair each flat cross index belongs to, and its offset within
            # that pair's block. Pairs with no survivors are skipped naturally:
            # their cross_cum entry repeats, so searchsorted steps past them.
            row = torch.searchsorted(cross_cum, flat, right=True)
            pos = flat - (cross_cum[row] - n_cross[row])
            b_here = b_cnt[row]
            a_local = pos // b_here
            b_local = pos - a_local * b_here
            a_at = a_off[row] + a_local
            b_at = b_off[row] + b_local

            mask, k_ket = _lookup_pair_keys_torch(pair_keys, a_uid_s[a_at] * m_beta + b_uid_s[b_at])
            if not mask.any():
                continue
            a_at, b_at = a_at[mask], b_at[mask]
            yield {
                "k_bra": row[mask] + cs,
                "k_ket": k_ket[mask],
                # ket -> bra labels: the swap of the bra -> ket enumeration.
                "cr_a": 2 * a_an_s[a_at],
                "an_a": 2 * a_cr_s[a_at],
                "cr_b": 2 * b_an_s[b_at] + 1,
                "an_b": 2 * b_cr_s[b_at] + 1,
            }
