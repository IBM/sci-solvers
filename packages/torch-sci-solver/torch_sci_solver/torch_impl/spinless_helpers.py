"""Flat spin-orbital ("spinless") basis + streaming excitation tables.

A spinless Hamiltonian (GHF / spin-orbit-coupled / generalized-spinor) has no
alpha/beta factorization: the wavefunction lives over an explicit list of
``N`` flat ``n_spinorb``-bit determinants (dim = N), not a Cartesian product
or an ``(alpha, beta)`` pair list. This module is the flat-determinant
analogue of :mod:`torch_sci_solver.torch_impl.pairs` (the basis container +
initial-vector helper) and
:mod:`torch_sci_solver.torch_impl.pair_helpers` (the streaming excitation-
table generator) — genuinely simpler than both, since there is no per-spin
partner/uid bookkeeping to maintain: each basis row already *is* the unique
element, so membership testing is a single ``torch.searchsorted`` against the
sorted determinant list, with no "expand over partners sharing a spectator"
step (:func:`~torch_sci_solver.torch_impl.pair_helpers._stream_same_spin_dedup`'s
whole reason to exist).

Excitation enumeration mirrors :func:`~torch_sci_solver.torch_impl.pair_helpers.
_stream_same_spin` (the non-dedup reference there): for each determinant,
enumerate every way to annihilate ``order`` occupied spin-orbitals and create
``order`` empty ones (``order=1`` singles, ``order=2`` doubles), then keep the
candidates whose excited determinant is also in the (sorted) basis.

Determinants are stored as a stacked ``(N, n_limbs)`` int64 tensor (see
:mod:`torch_sci_solver.torch_impl.determinants`'s module docstring for the
limb convention), so ``n_spinorb`` is not capped at 64.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE
from torch_sci_solver.torch_impl.helpers import _n_limbs
from torch_sci_solver.torch_impl.pair_helpers import (
    DEFAULT_BUDGET_BYTES,
    _bit_positions_of_pattern_multilimb,
    _found_in_multilimb,
    _triu_indices,
    resolve_budget_bytes,
)
from torch_sci_solver.torch_impl.pairs import _to_signed_int64, basis_init_vector_pair_torch

# int64 temporaries alive per candidate during one enumeration chunk. Fewer
# than pair mode's (_TEMPS_PER_CANDIDATE): no uid/partner bookkeeping, no
# per-pair spectator key — just the candidate's excitation indices, its
# target determinant, and the searchsorted result.
_TEMPS_PER_CANDIDATE = {"singles": 8, "doubles": 12}


def _chunk_dets(n_per_det: int, kind: str, budget_bytes: int) -> int:
    """Determinants per chunk that keep one enumeration chunk within ``budget_bytes``."""
    per_det_bytes = max(1, n_per_det * 8 * _TEMPS_PER_CANDIDATE[kind])
    return max(1, int(budget_bytes) // per_det_bytes)


@dataclass(frozen=True)
class SpinlessSCIState:
    """Spinless analog of ``qiskit_addon_sqd.fermion.SCIState``.

    ``SCIState`` requires ``amplitudes.shape == (len(ci_strs_a), len(ci_strs_b))``
    -- a Cartesian alpha/beta product. A spinless basis is a single length-N
    list of N-particle spin-orbital determinants; the eigenvector is a flat
    length-N 1-D vector and there is one string array, not two. Mirrors
    ``torch_sci_solver.torch_impl.pairs.PairSCIState`` (the pair-mode
    lookalike) and ``sci_solvers.spinless_ci_problem.SpinlessSCIState`` (the
    numpy/general-algebra equivalent) -- kept here rather than imported from
    ``sci_solvers`` since ``torch_sci_solver`` is the standalone lower-level
    package (``sci_solvers`` depends on it, not the reverse).
    """

    amplitudes: "np.ndarray"  # flat length-N 1-D vector
    ci_strs: "np.ndarray"  # parallel length-N spin-orbital bitmask list
    norb: int  # number of spin-orbitals (nmo)
    nelec: "tuple[int, int]"  # (N, 0) embedding for the (N, 0)-sector RDM reuse

    def save(self, filename):
        """Save the SpinlessSCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs=self.ci_strs,
            norb=self.norb,
            nelec=self.nelec,
        )


def _n_limbs_for(n_spinorb: int) -> int:
    """Number of int64 limbs needed to hold an ``n_spinorb``-bit determinant."""
    return _n_limbs(n_spinorb)


def _string_mask_limbs(n_spinorb: int, n_limbs: int, device: torch.device) -> Tensor:
    """Per-limb complement mask covering exactly bits ``[0, n_spinorb)``.

    Limb ``k`` gets its low ``min(64, n_spinorb - 64k)`` bits set and the rest
    zero, so XORing a det against this (broadcast over the batch) complements
    only the ``n_spinorb`` orbital bits — any unused high bits in the last
    limb (when ``n_spinorb`` isn't a multiple of 64) stay zero rather than
    being wrongly flagged "empty".
    """
    masks = []
    for k in range(n_limbs):
        bits_in_limb = max(0, min(64, n_spinorb - 64 * k))
        masks.append(-1 if bits_in_limb == 64 else (1 << bits_in_limb) - 1)
    return torch.tensor(masks, dtype=torch.int64, device=device)


def _flip_bits(dets: Tensor, *positions: Tensor) -> Tensor:
    """XOR-flip the bit at each position in ``positions`` of every det.

    ``dets`` is ``(batch, n_limbs)``; each entry of ``positions`` is a
    ``(batch,)`` int64 tensor of bit indices in ``[0, 64 * n_limbs)``. The
    positions within one det are assumed distinct (true for the excitation
    indices this is used for — annihilation/creation operators never target
    the same spin-orbital twice), so composing their single-bit masks with
    ``|`` before one final XOR against ``dets`` is equivalent to XORing them
    in one at a time.
    """
    batch, n_limbs = dets.shape
    one = torch.tensor(1, dtype=torch.int64, device=dets.device)
    mask = torch.zeros_like(dets)
    for pos in positions:
        limb_idx = (pos >> 6).unsqueeze(-1)
        bit = (one << (pos & 63)).unsqueeze(-1)
        contrib = torch.zeros(batch, n_limbs, dtype=torch.int64, device=dets.device)
        contrib.scatter_(-1, limb_idx, bit)
        mask = mask | contrib
    return dets ^ mask


@dataclass(frozen=True)
class SpinlessBasis:
    """An explicit, sorted, deduplicated list of flat spin-orbital determinants.

    Attributes:
        dets: sorted, deduplicated ``(N, n_limbs)`` int64 determinant
            bitmasks. Sorted lexicographically by limb index (limb 0 — the
            lowest 64 bits, see :mod:`torch_sci_solver.torch_impl.
            determinants` for the per-limb bit layout — compares first),
            which is :func:`torch.unique`'s own ``dim=0`` row order and is
            *not* the dets' numeric order once ``n_limbs > 1``. Lookups only
            need self-consistency with that order (every one goes through
            :func:`~torch_sci_solver.torch_impl.pair_helpers.
            _found_in_multilimb`, which compares limbs in the same order) —
            but the sort order is *not* semantically neutral: the Davidson
            seed used to assume index 0 was the HF determinant, which this
            order only guarantees while ``n_limbs == 1``. The seed is now
            chosen by diagonal (:func:`basis_init_vector_spinless_torch`'s
            ``seed_index``, set to the argmin of ``hii`` by the adapter),
            not by basis order.
        n_spinorb: number of spin-orbitals (``nmo``); each determinant is a
            bitmask over ``[0, n_spinorb)``.
        n_particles: total particle number ``N`` (every determinant has this
            popcount; not validated here, callers should have already
            filtered to a uniform particle number).
    """

    dets: Tensor
    n_spinorb: int
    n_particles: int

    @property
    def dim(self) -> int:
        """N — the flat basis dimension."""
        return int(self.dets.shape[0])

    @property
    def n_limbs(self) -> int:
        """Number of int64 limbs each determinant occupies."""
        return _n_limbs_for(self.n_spinorb)

    @property
    def device(self) -> torch.device:
        """Torch device of ``dets``."""
        return self.dets.device


def _split_to_limbs(value: int, n_limbs: int) -> list[int]:
    """Split an unsigned Python int into ``n_limbs`` signed-int64 limbs, low limb first."""
    LO_MASK = (1 << 64) - 1
    return [_to_signed_int64((value >> (64 * k)) & LO_MASK) for k in range(n_limbs)]


def dets_to_limb_rows(dets: Any, n_spinorb: int) -> "np.ndarray":
    """Pack Python-int determinant strings into an ``(N, n_limbs)`` int64 matrix.

    The matrix uses the same low-limb-first signed-int64 layout as
    :func:`spinless_basis_from_array`, so two determinants compare equal here
    exactly when they compare equal in a :class:`SpinlessBasis` -- including
    determinants wider than 64 bits, where a single int64 cannot hold the
    string. The row lexicographic order of this matrix is the basis order.

    Args:
        dets: iterable of determinant bitmasks (Python ints, or an integer /
            object array).
        n_spinorb: number of spin-orbitals (``nmo``).

    Returns:
        ``(len(dets), n_limbs)`` int64 matrix, low limb first.

    Raises:
        ValueError: a determinant is negative or sets a bit at or above
            ``n_spinorb``.
    """
    n_limbs = _n_limbs_for(n_spinorb)
    values = [int(det) for det in dets]
    out = np.empty((len(values), n_limbs), dtype=np.int64)
    for i, value in enumerate(values):
        if value < 0 or value.bit_length() > int(n_spinorb):
            raise ValueError(
                f"determinant {value:#x} does not fit in n_spinorb={int(n_spinorb)} spin-orbitals."
            )
        out[i] = _split_to_limbs(value, n_limbs)
    return out


def dets_to_python_ints(dets: Tensor) -> "np.ndarray":
    """Reassemble a ``(N, n_limbs)`` determinant tensor into unsigned Python ints.

    Inverse of the per-row packing :func:`_split_to_limbs` does, generalized
    to a whole basis. Returns an ``(N,)`` object-dtype array (Python
    arbitrary-precision ints), since once ``n_limbs > 1`` no fixed-width
    numpy integer type can hold the full bitmask. Needed at API boundaries
    that still expect a flat per-determinant bitmask (e.g.
    :class:`SpinlessSCIState.ci_strs`), where the stacked limb layout is an
    internal-only representation.
    """
    INT64_OFFSET = 1 << 64
    rows = dets.cpu().tolist()
    out = []
    for row in rows:
        v = 0
        for k, limb in enumerate(row):
            v |= ((limb + INT64_OFFSET) & (INT64_OFFSET - 1)) << (64 * k)
        out.append(v)
    return np.array(out, dtype=object)


def spinless_basis_from_array(
    strs,
    n_spinorb: int,
    n_particles: int,
    *,
    device: "torch.device | str" = "cpu",
) -> SpinlessBasis:
    """Build a :class:`SpinlessBasis` from an iterable of determinant bitmasks.

    Dedups and sorts ``strs``, mirroring
    :func:`~torch_sci_solver.torch_impl.pairs.pair_basis_from_arrays`'s
    dedup/sort convention (see :class:`SpinlessBasis` for the resulting sort
    order once ``n_spinorb > 64``).

    Args:
        strs: iterable of ``N``-particle spin-orbital determinant bitmasks.
        n_spinorb: number of spin-orbitals (``nmo``).
        n_particles: total particle number ``N``.
        device: torch device for the returned basis.

    Raises:
        ValueError: ``strs`` is empty.
    """
    if isinstance(device, str):
        device = torch.device(device)

    n_limbs = _n_limbs_for(n_spinorb)
    rows = [_split_to_limbs(int(x), n_limbs) for x in strs]
    arr = np.array(rows, dtype=np.int64).reshape(-1, n_limbs)
    if arr.shape[0] == 0:
        raise ValueError("spinless_basis_from_array: empty determinant list.")
    dets_np = np.unique(arr, axis=0)
    dets = torch.from_numpy(np.ascontiguousarray(dets_np)).to(device)
    return SpinlessBasis(dets=dets, n_spinorb=n_spinorb, n_particles=n_particles)


def basis_init_vector_spinless_torch(
    n_dets: int,
    init: int = 0,
    device: "torch.device | str" = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    seed_index: int = 0,
) -> Tensor:
    """Initial wavefunction over an N-determinant spinless basis.

    Same seeded-or-random recipe as
    :func:`~torch_sci_solver.torch_impl.pairs.basis_init_vector_pair_torch`,
    but the seeded index is caller-chosen rather than fixed at 0. Pair mode's
    index-0 convention ("pair 0 is the smallest-int pair after the canonical
    sort") does not carry over: a multi-limb spinless basis sorts
    lexicographically by the *low* limb first, so index 0 is not the
    numerically smallest determinant and has no claim to being HF. Callers
    that know the physical seed determinant pass its index via
    ``seed_index`` (the adapter passes the argmin of the diagonal).

    Args:
        n_dets: N — number of determinants in the basis.
        init: 0 = seed at ``seed_index``, 1 = random unit vector.
        device: torch device.
        dtype: float dtype.
        seed: optional seed for ``init=1``.
        seed_index: basis index receiving the unit seed when ``init=0``
            (default 0, preserving the single-limb behavior).

    Raises:
        ValueError: ``init`` is neither 0 nor 1, or ``seed_index`` is out of
            range.
    """
    if init == 1:
        return basis_init_vector_pair_torch(n_dets, init=1, device=device, dtype=dtype, seed=seed)
    if init != 0:
        raise ValueError(f"init must be 0 (seeded) or 1 (random), got {init}")
    if not 0 <= seed_index < n_dets:
        raise ValueError(f"seed_index must be in [0, {n_dets}), got {seed_index}")
    if isinstance(device, str):
        device = torch.device(device)
    w = torch.zeros(n_dets, dtype=dtype, device=device)
    w[seed_index] = 1.0
    # Add random noise, same fix as basis_init_vector_torch in product mode
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    noise = torch.randn(n_dets, generator=gen, dtype=dtype, device=device)
    noise = noise / torch.linalg.norm(noise).clamp_min(torch.finfo(dtype).tiny)
    w = w + SEED_NOISE_SCALE * noise
    w = w / torch.linalg.norm(w)
    return w


def _stream_spinless(
    *,
    dets: Tensor,
    n_spinorb: int,
    n_particles: int,
    order: int,
    device: torch.device,
    budget_bytes: int,
    src_dets: "Tensor | None" = None,
    target_dets: "Tensor | None" = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield chunks of flat single (``order=1``) or double (``order=2``) excitations.

    For each determinant in the chunk, enumerate every way to annihilate
    ``order`` occupied spin-orbitals and create ``order`` empty ones over the
    whole ``[0, n_spinorb)`` space (no alpha/beta split), keep the candidates
    whose excited determinant is also in ``target_dets``, via
    :func:`torch_sci_solver.torch_impl.pair_helpers._found_in_multilimb`. Mirrors
    :func:`torch_sci_solver.torch_impl.pair_helpers._stream_same_spin` with
    the per-spin split removed (one enumeration over the full orbital space
    instead of one per spin sector).

    ``src_dets`` (the determinants enumerated **from**, ``k_bra``'s index
    space) and ``target_dets`` (the determinants excitations are matched
    **against**, ``k_ket``'s index space) default to ``dets`` each --
    the ordinary selected-CI matvec table, where both the pre- and
    post-excitation determinant must be selected. :meth:`SpinlessHelpers.
    connected_singles` passes them independently (``src_dets`` = the
    connected set, ``target_dets`` = ``dets``) to build the wider table the
    2-RDM Gram term needs -- see :mod:`torch_sci_solver.torch_impl.
    rdm_spinless`.
    """
    if src_dets is None:
        src_dets = dets
    if target_dets is None:
        target_dets = dets
    n_dets = int(src_dets.shape[0])
    n_empty = n_spinorb - n_particles
    if n_dets == 0 or n_particles < order or n_empty < order:
        return
    if order == 1:
        n_per_det = n_particles * n_empty
    else:
        n_per_det = (n_particles * (n_particles - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    if n_per_det == 0:
        return

    chunk = _chunk_dets(n_per_det, "singles" if order == 1 else "doubles", budget_bytes)
    n_limbs = src_dets.shape[-1]
    string_mask = _string_mask_limbs(n_spinorb, n_limbs, device)
    if order == 2:
        an_i, an_j = _triu_indices(n_particles, device=device)
        cr_i, cr_j = _triu_indices(n_empty, device=device)

    for cs in range(0, n_dets, chunk):
        ce = min(cs + chunk, n_dets)
        src_chunk = src_dets[cs:ce]
        occ_pos = _bit_positions_of_pattern_multilimb(src_chunk, n_spinorb, n_particles)
        empty_pos = _bit_positions_of_pattern_multilimb(src_chunk ^ string_mask, n_spinorb, n_empty)

        if order == 1:
            an0 = occ_pos.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
            cr0 = empty_pos.unsqueeze(-2).expand(-1, n_particles, -1).reshape(-1)
        else:
            n_op = an_i.shape[0]
            n_ep = cr_i.shape[0]
            an0 = occ_pos[:, an_i].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            an1 = occ_pos[:, an_j].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
            cr0 = empty_pos[:, cr_i].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
            cr1 = empty_pos[:, cr_j].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)

        k_bra = (
            torch
            .arange(cs, ce, dtype=torch.int64, device=device)
            .unsqueeze(-1)
            .expand(-1, n_per_det)
            .reshape(-1)
        )
        if order == 1:
            target = _flip_bits(src_dets[k_bra], an0, cr0)
        else:
            target = _flip_bits(src_dets[k_bra], an0, cr0, an1, cr1)

        found, k_ket = _found_in_multilimb(target_dets, target)
        if not found.any():
            continue

        out = {
            "k_bra": k_bra[found],
            "k_ket": k_ket[found],
            "cr0": cr0[found],
            "an0": an0[found],
        }
        if order == 2:
            out["cr1"] = cr1[found]
            out["an1"] = an1[found]
        yield out


def _all_single_targets_spinless(
    *,
    dets: Tensor,
    n_spinorb: int,
    n_particles: int,
    device: torch.device,
    budget_bytes: int,
) -> Iterator[dict[str, Tensor]]:
    """Yield ``{"k_bra", "cr0", "an0", "target"}`` for every single excitation of ``dets``.

    Same enumeration as :func:`_stream_spinless` with ``order=1``, but
    without the membership filter on the excited string: every candidate is
    yielded, with the raw excited-determinant
    bitmask (``target``) in place of its looked-up index. ``k_bra`` is the
    enumeration source row index into ``dets`` (always valid, as in
    :func:`_stream_spinless`). Used to discover the *connected set* (see
    :func:`_connected_set_spinless`) and by :meth:`SpinlessHelpers.
    connected_singles`, which looks ``target`` up in the connected set
    instead of ``dets``.
    """
    n_dets = int(dets.shape[0])
    n_empty = n_spinorb - n_particles
    if n_dets == 0 or n_particles < 1 or n_empty < 1:
        return
    n_per_det = n_particles * n_empty
    if n_per_det == 0:
        return

    chunk = _chunk_dets(n_per_det, "singles", budget_bytes)
    n_limbs = dets.shape[-1]
    string_mask = _string_mask_limbs(n_spinorb, n_limbs, device)

    for cs in range(0, n_dets, chunk):
        ce = min(cs + chunk, n_dets)
        src_chunk = dets[cs:ce]
        occ_pos = _bit_positions_of_pattern_multilimb(src_chunk, n_spinorb, n_particles)
        empty_pos = _bit_positions_of_pattern_multilimb(src_chunk ^ string_mask, n_spinorb, n_empty)

        an0 = occ_pos.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
        cr0 = empty_pos.unsqueeze(-2).expand(-1, n_particles, -1).reshape(-1)
        k_bra = (
            torch
            .arange(cs, ce, dtype=torch.int64, device=device)
            .unsqueeze(-1)
            .expand(-1, n_per_det)
            .reshape(-1)
        )
        target = _flip_bits(dets[k_bra], an0, cr0)
        yield {"k_bra": k_bra, "cr0": cr0, "an0": an0, "target": target}


def _connected_set_spinless(basis: SpinlessBasis, *, budget_bytes: int) -> Tensor:
    """Connected set ``S ∪ {single-excitation reachable targets}``, sorted (see :class:`SpinlessBasis`).

    Required for a correct selected-CI 2-RDM Gram term: see
    :mod:`torch_sci_solver.torch_impl.rdm_spinless` and
    ``sci_solvers.complex_rdm._restrict_to_connected_set``, whose docstring
    derives why the Gram term's intermediate axis must span this set rather
    than the selected set ``S`` (= ``basis.dets``) alone.
    """
    targets = [basis.dets]
    for t in _all_single_targets_spinless(
        dets=basis.dets,
        n_spinorb=basis.n_spinorb,
        n_particles=basis.n_particles,
        device=basis.device,
        budget_bytes=budget_bytes,
    ):
        targets.append(t["target"])
    return torch.unique(torch.cat(targets, dim=0), dim=0)


@dataclass(frozen=True)
class SpinlessHelpers:
    """Streaming excitation-table accessor over a :class:`SpinlessBasis`.

    Holds only the ``O(N)`` determinant list; the two table-kind methods are
    generators that re-derive their entries in chunks on each call, exactly
    as :class:`~torch_sci_solver.torch_impl.pair_helpers.PairHelpers` does —
    nothing of size ``nnz(H)`` is ever materialized.
    """

    basis: SpinlessBasis
    budget_bytes: int = DEFAULT_BUDGET_BYTES
    last_nnz: dict[str, int] = field(default_factory=dict)

    def singles(self) -> Iterator[dict[str, Tensor]]:
        """Stream flat single excitations, chunk by chunk."""
        return self._counted("singles", self._stream(order=1))

    def doubles(self) -> Iterator[dict[str, Tensor]]:
        """Stream flat double excitations, chunk by chunk."""
        return self._counted("doubles", self._stream(order=2))

    def connected_singles(self) -> tuple[Tensor, Iterator[dict[str, Tensor]]]:
        """Connected-set singles table for the 2-RDM Gram term.

        Returns ``(connected, chunks)``: ``connected`` is the sorted
        determinant list ``S ∪ {reachable targets}`` (see
        :func:`_connected_set_spinless`); ``chunks`` streams
        ``{"k_bra", "k_ket", "cr0", "an0"}`` -- unlike :meth:`singles`,
        enumerated **from** ``connected`` (so ``k_bra`` indexes ``connected``,
        not ``basis.dets``) and filtered against the *selected* set, so
        ``k_ket`` indexes ``basis.dets`` as before. This is the opposite
        direction from :meth:`singles` (which enumerates from ``basis.dets``
        and filters against ``basis.dets``): :func:`~torch_sci_solver.
        torch_impl.rdm_spinless.build_b_tensor_spinless` scatters
        ``amplitudes[k_ket]`` (always a valid selected-set read) onto ``B``'s
        ``k_bra``-indexed axis, which must therefore range over the full
        connected set for the Gram term to see every intermediate the
        selected set's single excitations reach -- restricting it to
        ``basis.dets`` (:meth:`singles`'s table) is exactly selected-CI's
        2-RDM bug.
        """
        connected = _connected_set_spinless(self.basis, budget_bytes=self.budget_bytes)
        chunks = _stream_spinless(
            dets=self.basis.dets,
            n_spinorb=self.basis.n_spinorb,
            n_particles=self.basis.n_particles,
            order=1,
            device=self.basis.device,
            budget_bytes=self.budget_bytes,
            src_dets=connected,
            target_dets=self.basis.dets,
        )
        return connected, self._counted("connected_singles", chunks)

    def _stream(self, *, order: int) -> Iterator[dict[str, Tensor]]:
        return _stream_spinless(
            dets=self.basis.dets,
            n_spinorb=self.basis.n_spinorb,
            n_particles=self.basis.n_particles,
            order=order,
            device=self.basis.device,
            budget_bytes=self.budget_bytes,
        )

    def _counted(self, name: str, gen: Iterator[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
        total = 0
        for chunk in gen:
            total += int(chunk["k_bra"].shape[0])
            yield chunk
        self.last_nnz[name] = total

    def nnz(self) -> dict[str, int]:
        """Row count of both table kinds, by streaming each one (diagnostics only)."""
        for stream in (self.singles, self.doubles):
            for _ in stream():
                pass
        return dict(self.last_nnz)


def generate_spinless_helpers(
    basis: SpinlessBasis,
    *,
    budget_bytes: "int | None" = None,
) -> SpinlessHelpers:
    """Build the streaming excitation-table accessor for a :class:`SpinlessBasis`.

    Trivial (no precomputation beyond what ``basis`` already holds) — the
    flat analogue of
    :func:`~torch_sci_solver.torch_impl.pair_helpers.generate_pair_helpers`.
    """
    return SpinlessHelpers(basis=basis, budget_bytes=resolve_budget_bytes(budget_bytes))
