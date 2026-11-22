"""Vectorized H @ T matvec for the SBD selected-basis algorithm — chunked.

Each of the three SBD tasks (α-only, β-only, αβ-doubles) is processed
in chunks of at most ``chunk_size`` excitation entries. Within each
chunk, the work is a single fully-batched Slater-Condon call + one
``index_add_`` scatter into the output vector — same as before.
Chunking keeps the per-call workspace memory bounded:

  - ``one_excite_batch``:  ~ 1.2 KB / entry  (2L=72)
                            dominated by two (batch, 2L) ERI gathers
  - ``two_excite_batch``:  ~ 160 B / entry
                            just parity scratch + 2 ERI gathers

Default chunk_size = 2,000,000. This gives ~2.4 GB workspace per
``one_excite_batch`` call at 2L=72 — fits comfortably on any modern
GPU. For tasks where the batch is already smaller than chunk_size
(typical for K ≤ 200), this degenerates to the un-chunked code path
with no overhead.

Layout / scheduling details
---------------------------
The wavefunction lives over a Cartesian-product basis
``{(adets[ia], bdets[ib]) : ia in [0, M_a), ib in [0, M_b)}``,
linearly indexed as ``k = ia * M_b + ib``.

For each task type:
  - task 2 (α-only):     chunk axis: α-excitation entries.
                          per-entry batch factor: M_b
  - task 1 (β-only):     chunk axis: β-excitation entries.
                          per-entry batch factor: M_a
  - task 0 (αβ-doubles): chunk axis: α-single entries.
                          per-entry batch factor: n_β_singles

So the effective batch per chunk for task 0 is ``chunk × n_β_singles``;
caller should size ``chunk_size`` to bound *total* lifted entries.

The full bra determinant ``DetI = det_from_alpha_beta(adets[ia], bdets[ib])``
is rebuilt for every excitation entry. Cheap and avoids gather-pattern
complexity of caching. Per-chunk this is one ``det_from_alpha_beta_batch``
call on the chunked indices.

All scatter-adds use ``index_add_`` which handles repeated indices
correctly (atomic on CUDA, deterministic on CPU).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

Derived from SBD (https://github.com/r-ccs-cms/sbd), licensed under the Apache
License 2.0: Copyright 2025 RIKEN and its contributors, and the other SBD
copyright holders listed in NOTICE. This file is a modified Python version of
the original C++ code.

SPDX-License-Identifier: Apache-2.0
"""


import os as _os
import time as _time
from typing import Iterable

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import (
    _bit64_mask_below,
    _bra_from_offsets,
    _popcount64,
    det_from_alpha_beta_batch,
    parity_batch,
    two_excite_batch,
    zero_excite_batch,
)
from torch_sci_solver.torch_impl.helpers import TorchHelper
from torch_sci_solver.torch_impl.integrals import TorchIntegrals

# Default chunk size — covers a ~2-3 GB workspace on a single matvec.
DEFAULT_CHUNK_SIZE = 2_000_000


# ============================================================================
# Optional per-phase profiling inside matvec
# ============================================================================
#
# When ``TORCH_SCI_MATVEC_PROFILE=1``, the matvec kernels record per-phase
# time into the module-level ``MATVEC_PROFILE`` dict. Each phase is wrapped
# in a sync-then-time block so GPU work is properly accounted for. Cost
# when the env var is OFF: a single ``os.environ.get`` lookup per matvec
# (microseconds, negligible vs. the multi-second matvec cost).
#
# The reset() helper is called at the START of every matvec when profiling
# is enabled, so what shows up in MATVEC_PROFILE is the last-completed
# matvec's breakdown. The qiskit_adapter dumps this after each Davidson
# call when verbose=True.
#
# Phases tracked:
#   - 'expand_reshape':  the index materialization via expand/reshape
#                        in _mult_alphabeta_torch (suspected bottleneck)
#   - 'double_excite_chunk':  the actual Slater-Condon-driven gather/scatter
#   - 'alpha_only':      task_type=2 total (all chunks)
#   - 'beta_only':       task_type=1 total
#   - 'alphabeta':       task_type=0 total (sum of above sub-phases)
#   - 'diagonal':        the hii * T pass
#
# Times are wall-clock in seconds, accumulated over chunks within one
# matvec. Counts are number of chunks processed.
_MATVEC_PROFILE_ENABLED = _os.environ.get("TORCH_SCI_MATVEC_PROFILE", "").strip() in (
    "1",
    "true",
    "True",
    "yes",
)
MATVEC_PROFILE: dict = {}


def _matvec_profile_reset():
    """Matvec profile reset."""
    if not _MATVEC_PROFILE_ENABLED:
        return
    MATVEC_PROFILE.clear()


def _matvec_profile_add(phase: str, dt: float, *, count: int = 1):
    """Matvec profile add."""
    if not _MATVEC_PROFILE_ENABLED:
        return
    if phase not in MATVEC_PROFILE:
        MATVEC_PROFILE[phase] = {"t": 0.0, "n": 0}
    MATVEC_PROFILE[phase]["t"] += dt
    MATVEC_PROFILE[phase]["n"] += count


def _sync(device):
    """Synchronize if CUDA. Cheap no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _matvec_profile_format() -> str:
    """Matvec profile format."""
    if not MATVEC_PROFILE:
        return "(profiling not enabled or no data)"
    rows = []
    total = sum(d["t"] for d in MATVEC_PROFILE.values())
    rows.append(f"  {'phase':<25s}  {'time (s)':>10s}  {'%':>5s}  {'count':>7s}")
    for phase, d in sorted(MATVEC_PROFILE.items(), key=lambda kv: kv[1]["t"], reverse=True):
        pct = 100 * d["t"] / total if total > 0 else 0
        rows.append(f"  {phase:<25s}  {d['t']:>10.3f}  {pct:>4.1f}%  {d['n']:>7d}")
    return "\n".join(rows)


# ============================================================================
# Diagonal-of-H builder
# ============================================================================


def make_diag_terms_torch(
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Tensor:
    """Compute hii[k] = ⟨DetI|H|DetI⟩ for k = ia*M_b + ib, all (ia, ib).

    Chunked over the (ia, ib) flat index so a million-dim basis doesn't
    materialize a 1M × 2L bool occupancy tensor in one shot.

    SBD-scalar equivalent: ``make_diag_terms``.

    Notes:
        ``adets_t``/``bdets_t`` may be single-word ``(M,)`` tensors or
        ``(M, n_limbs)`` limb matrices; the underlying determinant and
        diagonal kernels already work limb-wise, so one path serves both.

        ``⟨D|H|D⟩`` is mathematically real for Hermitian integrals even when
        ``integrals.dtype`` is complex (a diagonal expectation value of a
        Hermitian operator). ``hii`` is returned as the matching real dtype
        (e.g. ``complex128`` → ``float64``) rather than complex — Davidson
        uses it as a real preconditioner denominator (``E[0] - hii``), which
        would otherwise need a complex subtraction/comparison that isn't
        well-defined for ordering. Any roundoff-induced imaginary part from
        the complex einsum in ``zero_excite_batch`` is dropped via ``.real``.
    """
    device = integrals.device
    M_a = adets_t.shape[0]
    M_b = bdets_t.shape[0]
    N = M_a * M_b

    hii_dtype = integrals.dtype.to_real() if integrals.dtype.is_complex else integrals.dtype
    hii = torch.empty(N, dtype=hii_dtype, device=device)
    chunk = max(1, int(chunk_size))

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        k_flat = torch.arange(start, end, device=device, dtype=torch.int64)
        ia_flat = k_flat // M_b
        ib_flat = k_flat % M_b
        alpha_bm = adets_t[ia_flat]
        beta_bm = bdets_t[ib_flat]
        det = det_from_alpha_beta_batch(alpha_bm, beta_bm, n_orb)
        hii_chunk = zero_excite_batch(
            det,
            None,
            integrals.n_spinorb,
            integrals.I0,
            integrals.I1_diag,
            integrals.I2_direct,
            integrals.I2_exchange,
        )
        hii[start:end] = hii_chunk.real if integrals.dtype.is_complex else hii_chunk
    return hii


# ============================================================================
# Matvec
# ============================================================================


def mult_torch(
    hii: Tensor,
    T: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cache: dict | None = None,
) -> Tensor:
    """Wb = H @ T, chunked along the per-task batch axis.

    Args:
        hii, T:        (M_a*M_b,) tensors on integrals device.
        adets_t, bdets_t: ``(M_a,)``/``(M_b,)`` single-word int64 strings, or
            ``(M_a, n_limbs)``/``(M_b, n_limbs)`` limb matrices for bases with
            more than 64 spatial orbitals.
        n_orb:         spatial orbital count L.
        integrals:     TorchIntegrals bundle.
        helpers:       iterable of TorchHelpers.
        chunk_size:    max lifted entries per inner batched call.
        cache:         precomputed β-pass setup from
            :func:`build_beta_pass_cache`; ``None`` (default) recomputes it
            inside the β pass, exactly as before.

    Returns:
        Wb = H @ T, shape (M_a*M_b,)

    TODO: closed-shell speedup for the ``adets == bdets`` case.
        When the wavefunction is spin-symmetric (``C = +/-C.T`` in the
        ``(M_a, M_b)`` layout — the closed-shell sector consumed by
        ``sci_solvers.eigen_problem.SymmetrizedEigenProblem``), ``H`` can be
        contracted on the symmetric sector directly instead of over the full
        product space, in the spirit of ``pyscf.fci.direct_spin0`` /
        ``libfci.FCIcontract_2e_spin0`` (~2x for closed shells). Two levers,
        in increasing payoff:
          * The α-only (task 2) and β-only (task 1) passes are related by the
            α<->β swap, so one is redundant under ``C = +/-C.T``. Cheap to skip,
            but small: per profiling these two together are only ~10% of the
            matvec (see below).
          * The αβ-doubles pass (task 0) dominates — ~89% of the matvec on the
            D=252 closed-shell case below. The real ~2x requires a spin0-aware
            ``_apply_alphabeta_chunk_fast`` that exploits ``C = +/-C.T`` to
            compute one triangle and symmetrize, rather than iterating the full
            α-single x β-single outer product.
        A ``mult_torch_spin0`` here (+ a ``"batched_spin0"`` entry in
        ``build_matvec``) would let ``SymmetrizedEigenProblem`` dispatch to it
        without expanding to the full space first; its packed-basis interface
        stays unchanged, only the inner apply does.

        Per-task breakdown reproduced with (CPU, float64)::

            # TORCH_SCI_MATVEC_PROFILE=1 python this_snippet.py
            import time
            from itertools import combinations
            import numpy as np
            from sci_solvers.eigen_problem import EigenProblem
            import torch_sci_solver.torch_impl.mult as M

            def all_strings(norb, ne):
                return np.array(
                    sorted(sum(1 << i for i in c) for c in combinations(range(norb), ne)),
                    dtype=np.int64,
                )

            rng = np.random.default_rng(3)
            norb, ne = 10, 5                       # closed shell -> D = C(10,5) = 252
            hcore = rng.standard_normal((norb, norb)); hcore = 0.5 * (hcore + hcore.T)
            x = rng.standard_normal((norb,) * 4); eri = np.zeros_like(x)
            for perm in [(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),
                         (2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)]:
                eri += x.transpose(perm)
            eri /= 8.0
            sa = all_strings(norb, ne)
            p = EigenProblem(sa, sa, hcore, eri, norb, (ne, ne))   # dim = 252**2 = 63504
            v = np.random.default_rng(0).standard_normal(p.dim)
            p.matvec(v)                            # warm up
            M.MATVEC_PROFILE.clear()
            t = time.perf_counter()
            for _ in range(5):
                p.matvec(v)
            print("mean matvec ms:", (time.perf_counter() - t) / 5 * 1e3)
            print(M.MATVEC_PROFILE)                # per-phase seconds

        Observed (Apple M-series CPU, one representative run): mean matvec
        ~600 ms; alphabeta ~530 ms (~89%), alpha_only ~29 ms, beta_only ~29 ms,
        diagonal negligible. Absolute times are machine-dependent; the ~89%
        alphabeta share is the point.
    """
    device = integrals.device
    M_a = adets_t.shape[0]
    M_b = bdets_t.shape[0]
    N = M_a * M_b
    if T.shape != (N,):
        raise ValueError(f"T shape {T.shape} != ({N},)")

    _matvec_profile_reset()

    t0 = _time.perf_counter()
    Wb = hii * T  # diagonal pass
    _sync(device)
    _matvec_profile_add("diagonal", _time.perf_counter() - t0)

    for h in helpers:
        t0 = _time.perf_counter()
        if h.task_type == 2:
            _mult_alpha_only_torch(
                Wb, T, adets_t, bdets_t, n_orb, integrals, h, M_a, M_b, chunk_size
            )
            _sync(device)
            _matvec_profile_add("alpha_only", _time.perf_counter() - t0)
        elif h.task_type == 1:
            _mult_beta_only_torch(
                Wb, T, adets_t, bdets_t, n_orb, integrals, h, M_a, M_b, chunk_size, cache=cache
            )
            _sync(device)
            _matvec_profile_add("beta_only", _time.perf_counter() - t0)
        elif h.task_type == 0:
            _mult_alphabeta_torch(Wb, T, adets_t, bdets_t, integrals, h, M_a, M_b, chunk_size)
            _sync(device)
            _matvec_profile_add("alphabeta", _time.perf_counter() - t0)
        else:
            raise ValueError(f"unknown task_type {h.task_type}")
    return Wb


# ============================================================================
# Determinant-string primitives (single word or multi-limb)
# ============================================================================
#
# A determinant string is a packed bitmask over ``n_orb`` spatial orbitals. It
# is carried either as a single int64 word (``n_orb <= 64``) or as a
# ``(..., n_limbs)`` int64 limb matrix, limb ``k`` holding bits
# ``[64k, 64k + 64)``. Every primitive below dispatches on ``strings.dim()``:
# a 1-D tensor is the single-word form, anything wider is a limb matrix whose
# trailing axis is the limb index. The single-word branch keeps the exact
# expressions the kernels used before limbs existed, so one-word results stay
# bit-identical.


def _string_range_popcount(strings: Tensor, lo, hi) -> Tensor:
    """Popcount of the bits in the per-entry global range ``[lo, hi)``.

    Args:
        strings: ``(N,)`` single-word strings, or a limb matrix whose trailing
            axis is the limb index (leading batch axes are allowed).
        lo, hi: per-entry bit positions as int64 tensors (or Python ints),
            ``0 <= lo <= hi <= 64 * n_limbs``. Broadcast against ``strings``.

    Returns:
        int64 popcounts of the common broadcast shape.

    Notes:
        The single-word branch uses the module's original mask expression
        ``(1 << hi) - (1 << lo)``, so one-word results are bit-identical to
        the pre-limb kernels. The limb branch clamps ``[lo, hi)`` to each
        64-bit limb and sums the per-limb popcounts; this is the same
        construction as ``determinants._popcount_in_range``.
    """
    if not torch.is_tensor(lo):
        lo = torch.as_tensor(lo, dtype=torch.int64, device=strings.device)
    if not torch.is_tensor(hi):
        hi = torch.as_tensor(hi, dtype=torch.int64, device=strings.device)
    # Helper tables may carry int32 orbital indices; the per-limb masks are
    # 64-bit words, so widen the bounds before masking.
    lo = lo.to(torch.int64)
    hi = hi.to(torch.int64)
    if strings.dim() == 1:
        one64 = torch.tensor(1, dtype=torch.int64, device=strings.device)
        return _popcount64(strings & ((one64 << hi) - (one64 << lo)))

    total = None
    for k in range(strings.shape[-1]):
        offset = 64 * k
        limb_lo = torch.clamp(lo - offset, 0, 64)
        limb_hi = torch.clamp(hi - offset, 0, 64)
        mask = _bit64_mask_below(limb_hi) ^ _bit64_mask_below(limb_lo)
        count = _popcount64(strings[..., k] & mask)
        total = count if total is None else total + count
    return total


def _string_bit(strings: Tensor, pos) -> Tensor:
    """Occupancy (0 or 1) at the per-entry bit positions ``pos``.

    Args:
        strings: ``(N,)`` single-word strings, or a limb matrix whose trailing
            axis is the limb index (leading batch axes are allowed).
        pos: per-entry bit positions as an int64 tensor (or Python int).
            Broadcast against ``strings``.

    Returns:
        int64 tensor of 0/1 occupancies of the broadcast shape.

    Notes:
        The single-word branch is ``(strings >> pos) & 1``. The limb branch
        gathers the limb holding each bit, then shifts within that word.
    """
    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos, dtype=torch.int64, device=strings.device)
    pos = pos.to(torch.int64)
    if strings.dim() == 1:
        return (strings >> pos) & 1

    batch = torch.broadcast_shapes(strings.shape[:-1], pos.shape)
    words = strings.expand(*batch, strings.shape[-1])
    pos = pos.expand(batch)
    limb = torch.gather(words, -1, (pos >> 6).unsqueeze(-1)).squeeze(-1)
    return (limb >> (pos & 63)) & 1


def _string_occ_matrix(strings: Tensor, n_orb: int) -> Tensor:
    """``(N, n_orb)`` occupancy matrix of a batch of determinant strings.

    Args:
        strings: ``(N,)`` single-word strings, or ``(N, n_limbs)`` limb
            matrices.
        n_orb: spatial orbital count (the number of valid bits).

    Returns:
        int64 tensor with ``occ[b, j] = bit j of strings[b]``.

    Notes:
        The single-word branch is the kernels' original
        ``(strings.unsqueeze(-1) >> arange(n_orb)) & 1``. The limb branch
        gathers, for every orbital, the limb that holds its bit.
    """
    j = torch.arange(n_orb, dtype=torch.int64, device=strings.device)
    if strings.dim() == 1:
        return (strings.unsqueeze(-1) >> j) & 1

    limb_idx = (j >> 6).unsqueeze(0).expand(strings.shape[0], n_orb)
    words = torch.gather(strings, 1, limb_idx)
    return (words >> (j & 63)) & 1


def _interleaved_range_parity(alpha_strings: Tensor, beta_strings: Tensor, lo, hi) -> Tensor:
    """Parity count of an interleaved spin-orbital range ``[lo, hi)``.

    The bra determinant interleaves α and β half-strings (α on the even spin
    orbitals, β on the odd ones). For a spin-orbital range ``[lo, hi)`` the
    count of occupied bits is

    .. math::

        \\alpha_{[(lo+1)//2,\\,(hi+1)//2)} + \\beta_{[lo//2,\\,hi//2)}
            + b_{\\mathrm{start}},

    where ``b_start`` is the occupation at ``lo // 2`` of the α string for
    even ``lo`` and of the β string for odd ``lo`` — the bit-at-start term
    SBD's parity adds on top of the range popcount.

    Args:
        alpha_strings, beta_strings: α/β half-strings, single-word or limb,
            broadcast against each other and against ``lo``/``hi``.
        lo, hi: per-entry spin-orbital range bounds.

    Returns:
        int64 count whose parity is the fermion sign factor.
    """
    count = _string_range_popcount(alpha_strings, (lo + 1) // 2, (hi + 1) // 2)
    count = count + _string_range_popcount(beta_strings, lo // 2, hi // 2)
    bit = torch.where(
        (lo & 1) == 0,
        _string_bit(alpha_strings, lo // 2),
        _string_bit(beta_strings, lo // 2),
    )
    return count + bit


def _alphabeta_pair_matrix(
    cr_a: Tensor,
    an_a: Tensor,
    cr_b: Tensor,
    an_b: Tensor,
    bra_alpha_strings: Tensor,
    bra_beta_strings: Tensor,
    integrals: TorchIntegrals,
    dtype: torch.dtype,
) -> Tensor:
    """Matrix elements of cross-spin (αβ) double excitations, one per pair.

    Args:
        cr_a: α creation spin-orbital indices, shape ``(n_alpha,)``.
        an_a: α annihilation spin-orbital indices, shape ``(n_alpha,)``.
        cr_b: β creation spin-orbital indices, shape ``(n_beta,)``.
        an_b: β annihilation spin-orbital indices, shape ``(n_beta,)``.
        bra_alpha_strings: α half-bitmasks of the bra determinants, shape
            ``(n_alpha,)`` (single word) or ``(n_alpha, n_limbs)``.
        bra_beta_strings: β half-bitmasks of the bra determinants, shape
            ``(n_beta,)`` (single word) or ``(n_beta, n_limbs)``.
        integrals: integral bundle; only ``I2_eri`` is read.
        dtype: dtype of the parity sign and of the returned matrix elements
            (normally ``integrals.I2_eri.dtype``, real or complex).

    Returns:
        ``(n_alpha, n_beta)`` tensor of matrix elements.

    Notes
    -----
    With ``I = min(cr_a, cr_b)``, ``J = max(cr_a, cr_b)``,
    ``A = min(an_a, an_b)`` and ``B = max(an_a, an_b)`` (spin-orbital
    indices), each element is

    .. math::

        H_{\\alpha\\beta} = \\left(I2[I, A, J, B] - I2[J, A, I, B]\\right)
            \\, s_{\\mathrm{skip}} \\, (-1)^{p_1 + p_2},

    where the creation index leads each chemist pair so complex ERIs are not
    conjugated, and ``s_skip = -1`` when ``A > J`` or ``B < I`` (the
    excitation intervals cross) and ``+1`` otherwise. A parity range
    ``[lo, hi)`` — from ``(I, A)`` for ``p_1`` and ``(J, B)`` for ``p_2`` —
    is evaluated on the bra strings, with α occupying the even spin orbitals
    and β the odd ones:

    .. math::

        p = \\alpha_{[(lo+1)//2,\\,(hi+1)//2)} + \\beta_{[lo//2,\\,hi//2)}
            + b_{\\mathrm{start}},

    where ``b_start`` is the occupation bit at ``lo // 2`` of the α string
    for even ``lo`` and of the β string for odd ``lo``. This is the element
    ``two_excite_batch`` produces from the interleaved bra determinant,
    without materializing that determinant.
    """
    # Broadcast the α tables down columns and the β tables across rows, so
    # every per-pair quantity below has shape (n_alpha, n_beta).
    cr_a = cr_a.unsqueeze(1)
    an_a = an_a.unsqueeze(1)
    cr_b = cr_b.unsqueeze(0)
    an_b = an_b.unsqueeze(0)

    # I, J, A, B per-pair: (n_alpha, n_beta)
    I_pair = torch.minimum(cr_a, cr_b)
    J_pair = torch.maximum(cr_a, cr_b)
    A_pair = torch.minimum(an_a, an_b)
    B_pair = torch.maximum(an_a, an_b)

    # Integral lookups
    # Creation index first in each pair: (IA|JB) - (JA|IB). The swapped order
    # (AI|BJ) is its complex conjugate -- see two_excite_batch.
    val_IAJB = integrals.I2_eri[I_pair, A_pair, J_pair, B_pair]
    val_JAIB = integrals.I2_eri[J_pair, A_pair, I_pair, B_pair]
    e_pair = val_IAJB - val_JAIB  # (n_alpha, n_beta)
    del val_IAJB, val_JAIB

    # Skip sign: -1 if A > J or B < I else +1
    skip = (A_pair > J_pair) | (B_pair < I_pair)
    e_pair = torch.where(skip, -e_pair, e_pair)
    del skip

    # Parity ranges (spin-orbital, per-pair)
    r1_lo = torch.minimum(I_pair, A_pair)
    r1_hi = torch.maximum(I_pair, A_pair)
    r2_lo = torch.minimum(J_pair, B_pair)
    r2_hi = torch.maximum(J_pair, B_pair)
    del I_pair, J_pair, A_pair, B_pair

    # Broadcast the bra α/β strings to (n_alpha, n_beta) batch axes. A
    # single-word string gets a singleton limb axis after its pair axis, so
    # every input is an unambiguous ``(..., n_limbs)`` limb matrix: (n_alpha, 1,
    # n_limbs) for α and (1, n_beta, n_limbs) for β.
    α_str = bra_alpha_strings.unsqueeze(1)
    β_str = bra_beta_strings.unsqueeze(0)
    if α_str.dim() == 2:
        α_str = α_str.unsqueeze(-1)
    if β_str.dim() == 2:
        β_str = β_str.unsqueeze(-1)

    # Total parity factor = (-1)^(α_popcount + β_popcount + bit_at_start) per
    # range; see _interleaved_range_parity for the range decomposition.
    r1_total = _interleaved_range_parity(α_str, β_str, r1_lo, r1_hi)
    del r1_lo, r1_hi
    r2_total = _interleaved_range_parity(α_str, β_str, r2_lo, r2_hi)
    del r2_lo, r2_hi

    # Combined parity: (-1)^(r1_total + r2_total)
    parity_sgn = 1.0 - 2.0 * ((r1_total + r2_total) & 1).to(dtype)  # (n_alpha, n_beta)
    del r1_total, r2_total

    return e_pair * parity_sgn


def _apply_alphabeta_chunk_fast(
    Wb,
    T,
    adets_t,
    bdets_t,
    integrals,
    bra_alpha_per_a_exc,
    ket_alpha_per_a_exc,
    cr_α,
    an_α,
    bra_beta_per_b_exc,
    ket_beta_per_b_exc,
    cr_β,
    an_β,
    M_a,
    M_b,
    chunk_size=2_000_000,
    T_full=None,
    T_stride=None,
):
    """Lean αβ (cross-spin) double-excitation kernel — per-pair, no cubes.

    For each (k_α, k_β) the matrix element has the form::

        H[k_α, k_β; α, β] = (I2[I,A,J,B] - I2[J,A,I,B]) × skip_sgn(k_α, k_β)
                            × parity_range1(α, β) × parity_range2(α, β)

    where ``I=min(cr0,cr1), J=max(cr0,cr1), A=min(an0,an1), B=max(an0,an1)``:
    I/J come from the *creation* indices and A/B from the *annihilation*
    indices. This is ``two_excite_batch(det, an0, an1, cr0, cr1, I2)`` (SBD's
    ``i, j`` are annihilated, ``a, b`` created), with the creation index first
    in each chemist pair so complex ERIs are not conjugated.

    Optimization vs the slow generic path
    (``two_excite_batch``):
      - Skips ``det_from_alpha_beta_batch`` (which loops ``L`` times to
        interleave α and β bits into the full det). For L=36 that's
        ~72 bit-ops per batch entry, a noticeable fraction of total
        per-pair work.
      - Skips two ``parity_batch`` calls (each does popcount-in-range
        plus a ``getocc_batch`` for the bit-at-start). Replaces them
        with cheap mask + popcount on the raw α and β strings.

    Memory footprint matches the slow chunked path: tensors are at most
    (chunk_α × n_b_exc) entries — never the full (chunk × n_b_exc × M)
    cube. This makes the kernel safe at production scale (n_b_exc and
    M_a, M_b in the tens of thousands).

    Correctness: verified bit-for-bit against the slow path on H4, LiH,
    H6 (zero floating-point difference).
    """
    _ = M_a  # retained for call-site signature compatibility with the full-vector kernel
    n_a_exc = bra_alpha_per_a_exc.shape[0]
    n_b_exc = bra_beta_per_b_exc.shape[0]
    if n_a_exc == 0 or n_b_exc == 0:
        return

    device = adets_t.device
    dtype = integrals.I2_eri.dtype

    # ``T_full`` is the global ket vector when the output block is a local row
    # slice (owner-computes matvec); ``T`` is then the local input block only.
    # ``T_stride`` is the T buffer's row width when it differs from the output
    # width ``M_b`` (a tile's stage buffer holds fewer columns than it owns).
    T_2d = (T_full if T_full is not None else T).view(-1, M_b if T_stride is None else T_stride)

    # Pre-gather the per-α-exc bra/ket α-strings (one value per excitation).
    bra_α_string_all = adets_t[bra_alpha_per_a_exc]  # (n_a_exc,)
    # β-strings indexed by per-β-exc bra/ket β-string indices.
    bra_β_string_all = bdets_t[bra_beta_per_b_exc]  # (n_b_exc,)

    # Pre-broadcast the β-axis ket/bra scatter indices to (1, n_b_exc).
    ket_β_idx_b = ket_beta_per_b_exc.unsqueeze(0)  # (1, n_b_exc)
    bra_β_idx_b = bra_beta_per_b_exc.unsqueeze(0)  # (1, n_b_exc)

    chunk_α = max(1, chunk_size // max(n_b_exc, 1))

    for start in range(0, n_a_exc, chunk_α):
        end = min(start + chunk_α, n_a_exc)

        # Per-chunk α-axis scatter indices, shape (csize, 1) for broadcast.
        bra_α_idx_c = bra_alpha_per_a_exc[start:end].unsqueeze(1)
        ket_α_idx_c = ket_alpha_per_a_exc[start:end].unsqueeze(1)

        # Per-pair matrix elements, shape (csize, n_b_exc).
        H_pair = _alphabeta_pair_matrix(
            cr_α[start:end],
            an_α[start:end],
            cr_β,
            an_β,
            bra_α_string_all[start:end],
            bra_β_string_all,
            integrals,
            dtype,
        )

        # Gather T at ket positions and scatter to bra positions.
        # T_pair[k_α_local, k_β] = T_2d[ket_α[k_α_local], ket_β[k_β]]
        T_pair = T_2d[ket_α_idx_c, ket_β_idx_b]  # (csize, n_b_exc)
        contrib = H_pair * T_pair
        del H_pair, T_pair

        # Scatter: Wb[bra_α * M_b + bra_β] += contrib
        bra_global = bra_α_idx_c * M_b + bra_β_idx_b  # (csize, n_b_exc)
        if _MATVEC_PROFILE_ENABLED:
            _sync(device)
            _t_scatter = _time.perf_counter()
        Wb.index_add_(0, bra_global.reshape(-1), contrib.reshape(-1))
        if _MATVEC_PROFILE_ENABLED:
            _sync(device)
            _matvec_profile_add("αβ_scatter", _time.perf_counter() - _t_scatter)
        del contrib, bra_global


def _apply_alpha_double_chunk_fast(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    bra_alpha_per_exc,
    ket_alpha_per_exc,
    cr0,
    cr1,
    an0,
    an1,
    M_a,
    M_b,
    chunk_size=2_000_000,
    T_full=None,
):
    """Fast α-double excitation kernel.

    Exploits factorization of the matrix element for pure α-only double
    excitations. In our interleaved-Wick spin-orbital encoding (α at
    even positions, β at odd positions), the matrix element factors as::

        ⟨α'β | a†_{2p}a†_{2q} a_{2r}a_{2s} | αβ⟩
            = integral(p,q,r,s)
              × α_sgn(α, α', p,q,r,s)
              × β_sgn(β, p,q,r,s)

    The integral and α_sgn are **independent of the β-string**; only β_sgn
    varies with β. So we compute integral × α_sgn ONCE per α-excitation
    (O(n_exc) work), then compute β_sgn per β-string (O(n_exc × M_b)
    cheap popcounts — NO integral lookup), and combine.

    Speedup vs the chunked path: avoids n_exc × M_b full Slater-Condon
    evaluations. Each (α-exc, β-string) pair now costs ~4 popcounts +
    2 bit-tests instead of one full integral lookup + parity scan.

    To bound memory, we chunk along the α-excitation axis with
    ``chunk_α = chunk_size // M_b``. Each chunk materializes tensors of
    size ``chunk_α × M_b ≈ chunk_size`` entries — same memory envelope
    as the chunked baseline.

    The β-parity scan ranges are derived from the α-excitation: for the
    two parity calls in ``two_excite_batch``, the ranges are
    ``[min(I,A), max(I,A))`` and ``[min(J,B), max(J,B))`` in
    *spin-orbital* indexing (so even endpoints). The β-bits sit at odd
    positions inside those ranges, contributing β-popcount =
    popcount(β-string, low_orbital, high_orbital) over the β-orbital
    range [start/2, end/2).

    Args:
        bra_alpha_per_exc:  (n_exc,) α-bra index for each excitation
        ket_alpha_per_exc:  (n_exc,) α-ket index for each excitation
        cr0, cr1, an0, an1: (n_exc,) spin-orbital indices (even = α range).
        chunk_size:         max entries per intermediate tensor; chunk
                             count = ceil(n_exc * M_b / chunk_size).
    """
    n_exc_total = bra_alpha_per_exc.shape[0]
    if n_exc_total == 0:
        return

    dtype = integrals.I2_eri.dtype

    # ----- α-side factor: integral × α-sign computed ONCE for all n_exc -----
    # This is O(n_exc) memory, much smaller than n_exc × M_b. No chunking
    # needed for this step.
    bra_alpha_bm = adets_t[bra_alpha_per_exc]  # (n_exc,) or (n_exc, n_limbs)
    zero_beta = torch.zeros_like(bra_alpha_bm)
    bra_a = det_from_alpha_beta_batch(bra_alpha_bm, zero_beta, n_orb)
    # (an, cr) order: see two_excite_batch.
    eij_α_all = two_excite_batch(bra_a, None, an0, an1, cr0, cr1, integrals.I2_eri)  # (n_exc,)

    # Precompute the β-orbital parity ranges for all excitations (still
    # O(n_exc) memory). ``_string_range_popcount`` turns them into per-limb
    # masks at popcount time, so the same bounds serve one-word and multi-limb
    # β strings.
    I_α = torch.minimum(an0, an1) // 2
    J_α = torch.maximum(an0, an1) // 2
    A_α = torch.minimum(cr0, cr1) // 2
    B_α = torch.maximum(cr0, cr1) // 2
    range1_lo_β_all = torch.minimum(I_α, A_α)
    range1_hi_β_all = torch.maximum(I_α, A_α)
    range2_lo_β_all = torch.minimum(J_α, B_α)
    range2_hi_β_all = torch.maximum(J_α, B_α)

    # Materialize (chunk_α × M_b) tensors only; chunk along excitation axis.
    Wb_2d = Wb.view(M_a, M_b)
    T_2d = (T_full if T_full is not None else T).view(-1, M_b)
    chunk_exc = max(1, chunk_size // max(M_b, 1))

    for start in range(0, n_exc_total, chunk_exc):
        end = min(start + chunk_exc, n_exc_total)

        # Slice all per-excitation arrays.
        eij_α = eij_α_all[start:end]  # (csize,)
        r1lo = range1_lo_β_all[start:end].unsqueeze(1)  # (csize, 1)
        r1hi = range1_hi_β_all[start:end].unsqueeze(1)
        r2lo = range2_lo_β_all[start:end].unsqueeze(1)
        r2hi = range2_hi_β_all[start:end].unsqueeze(1)

        range1_β_pop = _string_range_popcount(bdets_t, r1lo, r1hi)  # (csize, M_b)
        range2_β_pop = _string_range_popcount(bdets_t, r2lo, r2hi)
        β_sgn = 1.0 - 2.0 * ((range1_β_pop + range2_β_pop) & 1).to(dtype)  # (csize, M_b)
        # Free the popcount tensors early (next iteration won't reuse them).
        del range1_β_pop, range2_β_pop

        T_rows = T_2d[ket_alpha_per_exc[start:end]]  # (csize, M_b)
        contrib = (eij_α.unsqueeze(1) * β_sgn) * T_rows  # (csize, M_b)
        del β_sgn, T_rows

        Wb_2d.index_add_(0, bra_alpha_per_exc[start:end], contrib)
        del contrib


def _beta_double_setup(
    bdets_t,
    n_orb,
    integrals,
    bra_beta_per_exc,
    cr0,
    cr1,
    an0,
    an1,
):
    """T-independent per-excitation setup of the fast β-double kernel.

    Returns the integral x β-sign factor and the α-orbital parity ranges.
    Identical expressions to the ones the kernel used to inline, so a cached
    and an uncached matvec are bit-identical.

    Notes:
        For a β-double, ``I_β = min(an0, an1) // 2`` etc. are spatial
        orbitals; the α-parity range is the spin-orbital range shifted by one
        (odd endpoints), so the α-orbital range is ``[min(I,A) + 1, max(I,A) + 1)``.

        The ``range1_mask_all`` / ``range2_mask_all`` entries hold the
        ``(n_exc, 2)`` ``[lo, hi)`` bounds of those ranges under their
        historical names: the kernels build the single-word mask or the
        per-limb masks from them, so one cache serves either string width.
    """
    # ----- β-side factor: integral × β-sign computed ONCE per excitation -----
    # O(n_exc) memory — fine without chunking.
    bra_beta_bm = bdets_t[bra_beta_per_exc]
    zero_alpha = torch.zeros_like(bra_beta_bm)
    bra_b = det_from_alpha_beta_batch(zero_alpha, bra_beta_bm, n_orb)

    # (an, cr) order: see two_excite_batch.
    eij_β_all = two_excite_batch(bra_b, None, an0, an1, cr0, cr1, integrals.I2_eri)  # (n_exc,)
    # Precompute α-orbital range bounds (still O(n_exc) memory).
    # For spin-orbital range [2I+1, 2A+1), α-orbital range = [min(I,A)+1, max(I,A)+1).
    I_β = torch.minimum(an0, an1) // 2
    J_β = torch.maximum(an0, an1) // 2
    A_β = torch.minimum(cr0, cr1) // 2
    B_β = torch.maximum(cr0, cr1) // 2
    range1_lo_α_all = torch.minimum(I_β, A_β) + 1
    range1_hi_α_all = torch.maximum(I_β, A_β) + 1
    range2_lo_α_all = torch.minimum(J_β, B_β) + 1
    range2_hi_α_all = torch.maximum(J_β, B_β) + 1

    return {
        "eij_β_all": eij_β_all,
        "range1_mask_all": torch.stack((range1_lo_α_all, range1_hi_α_all), dim=-1),
        "range2_mask_all": torch.stack((range2_lo_α_all, range2_hi_α_all), dim=-1),
    }


def _apply_beta_double_chunk_fast(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    bra_beta_per_exc,
    ket_beta_per_exc,
    cr0,
    cr1,
    an0,
    an1,
    M_a,
    M_b,
    chunk_size=2_000_000,
    cache=None,
    T_stride=None,
):
    """Fast β-double excitation kernel.

    Symmetric to :func:`_apply_alpha_double_chunk_fast`. β-double
    excitations have spin-orbital indices at odd positions (= 2k+1).
    The integral × β-sign is computed once per β-excitation, and the
    α-parity sign is computed per α-string via popcount over the
    corresponding α-orbital ranges.

    Chunks along the β-excitation axis so each intermediate is bounded
    to ``chunk_size`` entries.

    Args:
        bra_beta_per_exc:   (n_exc,) β-bra index for each excitation
        ket_beta_per_exc:   (n_exc,) β-ket index for each excitation
        cr0, cr1, an0, an1: (n_exc,) spin-orbital indices (odd = β range).
        chunk_size:         max entries per intermediate.
        cache:              precomputed output of :func:`build_beta_pass_cache`;
                            ``None`` (default) recomputes the setup here.
    """
    n_exc_total = bra_beta_per_exc.shape[0]
    if n_exc_total == 0:
        return

    dtype = integrals.I2_eri.dtype

    if cache is None:
        setup = _beta_double_setup(bdets_t, n_orb, integrals, bra_beta_per_exc, cr0, cr1, an0, an1)
        eij_β_all = setup["eij_β_all"]
        range1_all = setup["range1_mask_all"]
        range2_all = setup["range2_mask_all"]
    else:
        eij_β_all = cache["eij_β_all"]
        range1_all = cache["range1_mask_all"]
        range2_all = cache["range2_mask_all"]

    # Chunk along β-excitation axis so each intermediate is (chunk × M_a).
    # ``Wb`` is the output block (width ``M_b``); ``T`` is the source buffer,
    # whose row width can differ (``T_stride``) when a tile stages a column
    # subset.
    Wb_2d = Wb.view(M_a, M_b)
    T_2d = T.view(M_a, M_b if T_stride is None else T_stride)
    chunk_exc = max(1, chunk_size // max(M_a, 1))

    for start in range(0, n_exc_total, chunk_exc):
        end = min(start + chunk_exc, n_exc_total)

        eij_β = eij_β_all[start:end]  # (csize,)
        # Each range entry is a (csize, 2) [lo, hi) pair.
        r1lo = range1_all[start:end, 0].unsqueeze(1)  # (csize, 1)
        r1hi = range1_all[start:end, 1].unsqueeze(1)
        r2lo = range2_all[start:end, 0].unsqueeze(1)
        r2hi = range2_all[start:end, 1].unsqueeze(1)

        range1_α_pop = _string_range_popcount(adets_t, r1lo, r1hi)  # (csize, M_a)
        range2_α_pop = _string_range_popcount(adets_t, r2lo, r2hi)
        α_sgn = 1.0 - 2.0 * ((range1_α_pop + range2_α_pop) & 1).to(dtype)  # (csize, M_a)
        del range1_α_pop, range2_α_pop

        T_cols = T_2d[:, ket_beta_per_exc[start:end]]  # (M_a, csize)
        # contrib[ia, k] = eij_β[k] × α_sgn[k, ia] × T_cols[ia, k]
        contrib = (eij_β.unsqueeze(1) * α_sgn).T * T_cols  # (M_a, csize)
        del α_sgn, T_cols

        Wb_2d.index_add_(1, bra_beta_per_exc[start:end], contrib)
        del contrib


def _apply_alpha_single_chunk_fast(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    bra_alpha_per_exc,
    ket_alpha_per_exc,
    cr,
    an,
    M_a,
    M_b,
    chunk_size=2_000_000,
    T_full=None,
):
    """Fast α-single excitation kernel.

    Mathematical structure:
        H_α-single[α',β; α,β] = α_sgn(α, p, r) × β_sgn(β, p, r) ×
                                  (h1[p, r] + sum_α(α, p, r) + sum_β(β, p, r))

    where cr=2p, an=2r are α-orbital spin-orbital indices (even).

    Parity factoring (same as α-doubles): parity over [min(cr,an), max(cr,an))
    of the full det splits into α-bit popcount (depends only on α-string) and
    β-bit popcount (depends only on β-string).

    Integral factoring: the Σ_j term has
        Σ_j (I2[cr,j,j,an] - I2[cr,j,an,j]) × occ_full(j)
    which splits across spin sector. Position j is α iff j is even (j=2j_α);
    position j is β iff j is odd (j=2j_β+1). So:
        sum_α(α, p, r) = Σ_{j_α ∈ α-occ} (I2[2p, 2j_α, 2j_α, 2r] - I2[2p, 2j_α, 2r, 2j_α])
        sum_β(β, p, r) = Σ_{j_β ∈ β-occ} (I2[2p, 2j_β+1, 2j_β+1, 2r] - I2[2p, 2j_β+1, 2r, 2j_β+1])

    The β-sum becomes a small matmul:
        integral_vec[k, j_β] = I2[cr_k, 2j_β+1, 2j_β+1, an_k] - I2[cr_k, 2j_β+1, an_k, 2j_β+1]
        β_occ[k_β, j_β] = bit j_β of β-string-k_β
        sum_β[k, k_β] = integral_vec[k, :] @ β_occ[k_β, :].T = (n_exc, L) @ (L, M_b) matmul
    This is O(n_exc × L × M_b) — much cheaper than the current O(n_exc × M_b × L²+) gather.

    Chunks along the excitation axis to bound memory.
    """
    n_exc_total = bra_alpha_per_exc.shape[0]
    if n_exc_total == 0:
        return

    device = adets_t.device
    dtype = integrals.I2_eri.dtype

    # --- α-side sign and integral factors: O(n_exc) work, done once ---
    bra_alpha_bm = adets_t[bra_alpha_per_exc]  # (n_exc,)
    zero_beta = torch.zeros_like(bra_alpha_bm)
    bra_a = det_from_alpha_beta_batch(bra_alpha_bm, zero_beta, n_orb)

    # α-sign: parity_batch on (α-only det), range [min(cr,an), max(cr,an))
    start = torch.minimum(cr, an)
    end = torch.maximum(cr, an)
    sgn_init = torch.ones(n_exc_total, dtype=dtype, device=device)
    α_sgn_all = parity_batch(bra_a, None, start, end, sgn_init)  # (n_exc,)

    # α-side integral pieces: h1 + sum_α (depends on α-string)
    # h1: shape (n_exc,)
    h1_term_all = integrals.I1[an, cr]  # NOTE: SBD uses I1[a, i] not I1[i, a]
    # — see one_excite_batch line 386.
    # We must follow the SAME convention: one_excite_batch does I1[a, i].
    # Here cr corresponds to `a` (creation) and an to `i` (annihilation). So
    # the term is I1[cr, an]? Let me read one_excite_batch carefully again:
    #   "i" param is annihilation, "a" param is creation
    #   "e1 = I1[a, i]" → I1[creation, annihilation]
    # Our parameters: cr is creation, an is annihilation. So h1 = I1[cr, an].
    h1_term_all = integrals.I1[cr, an]  # (n_exc,)

    # sum_α: per-α-exc contribution. Need:
    #   For each excitation k, for each α-orbital j_α, compute
    #   (I2_eri[cr_k, 2j_α, 2j_α, an_k] - I2_eri[cr_k, 2j_α, an_k, 2j_α])
    #   × α_occ_k(j_α)
    # where α_occ_k is the α-occupancy of the bra α-string for this excitation.
    L = n_orb
    j_α = torch.arange(L, dtype=torch.int64, device=device)  # (L,)
    j_α_2 = 2 * j_α  # (L,)

    # I2 lookups per (k, j_α): shape (n_exc, L)
    # Match one_excite_batch convention:
    #   I2_aijj = I2_eri[a, i, j, j] = I2_eri[creation, annihilation, j, j]
    #   I2_ajji = I2_eri[a, j, j, i] = I2_eri[creation, j, j, annihilation]
    # Here cr=creation (a), an=annihilation (i).
    cr_b = cr.unsqueeze(-1)  # (n_exc, 1)
    an_b = an.unsqueeze(-1)  # (n_exc, 1)
    j_b = j_α_2.unsqueeze(0)  # (1, L)
    I2_α_aijj = integrals.I2_eri[cr_b, an_b, j_b, j_b]  # (n_exc, L)
    I2_α_ajji = integrals.I2_eri[cr_b, j_b, j_b, an_b]  # (n_exc, L)
    α_int_vec = I2_α_aijj - I2_α_ajji  # (n_exc, L)

    # α-occupancy of the bra α-string at each α-orbital j: shape (n_exc, L)
    α_occ_per_exc = _string_occ_matrix(bra_alpha_bm, L).to(dtype)
    # sum_α: (n_exc,)
    sum_α_all = (α_int_vec * α_occ_per_exc).sum(dim=-1)  # (n_exc,)
    del I2_α_aijj, I2_α_ajji, α_int_vec, α_occ_per_exc

    # --- β-side sign factor: precompute β-range bounds per excitation ---
    # For α-single with cr, an even: range [min(cr, an), max(cr, an)) is even-even.
    # β-bits inside this range are at odd positions; corresponding β-orbital
    # range is [start//2, end//2). Same as α-doubles β-parity logic.
    range_lo_β_all = start // 2  # (n_exc,)
    range_hi_β_all = end // 2

    # --- β-side integral piece: integral_β_vec per excitation ---
    # For each k and each β-orbital j_β, compute
    #   I2_eri[cr_k, an_k, 2j_β+1, 2j_β+1] - I2_eri[cr_k, 2j_β+1, 2j_β+1, an_k]
    # → integral_β_vec of shape (n_exc, L). Same convention as α side.
    j_β_2p1 = 2 * j_α + 1  # (L,)
    j_β_b = j_β_2p1.unsqueeze(0)  # (1, L)
    I2_β_aijj = integrals.I2_eri[cr_b, an_b, j_β_b, j_β_b]  # (n_exc, L)
    I2_β_ajji = integrals.I2_eri[cr_b, j_β_b, j_β_b, an_b]  # (n_exc, L)
    β_int_vec_all = I2_β_aijj - I2_β_ajji  # (n_exc, L)
    del I2_β_aijj, I2_β_ajji

    # --- β-occupancy matrix: (M_b, L) — used by all chunks ---
    β_occ_M = _string_occ_matrix(bdets_t, L).to(dtype)  # (M_b, L)

    # --- chunked scatter ---
    Wb_2d = Wb.view(M_a, M_b)
    T_2d = (T_full if T_full is not None else T).view(-1, M_b)
    chunk_exc = max(1, chunk_size // max(M_b, 1))

    for start_k in range(0, n_exc_total, chunk_exc):
        end_k = min(start_k + chunk_exc, n_exc_total)

        # Per-chunk slices.
        α_sgn = α_sgn_all[start_k:end_k]  # (csize,)
        h1_term = h1_term_all[start_k:end_k]  # (csize,)
        sum_α = sum_α_all[start_k:end_k]  # (csize,)
        β_int_vec = β_int_vec_all[start_k:end_k]  # (csize, L)
        range_lo_β = range_lo_β_all[start_k:end_k].unsqueeze(1)  # (csize, 1)
        range_hi_β = range_hi_β_all[start_k:end_k].unsqueeze(1)

        # β-sign per (k, β-string): popcount-based sign, shape (csize, M_b)
        β_pop = _string_range_popcount(bdets_t, range_lo_β, range_hi_β)  # (csize, M_b)
        β_sgn = 1.0 - 2.0 * (β_pop & 1).to(dtype)  # (csize, M_b)
        del β_pop

        # sum_β: (csize, M_b) — matmul of (csize, L) @ (L, M_b)
        sum_β = β_int_vec @ β_occ_M.T  # (csize, M_b)

        # Combine: matrix_element[k, k_β] = α_sgn[k] × β_sgn[k, k_β] × (h1[k] + sum_α[k] + sum_β[k, k_β])
        e_total = (h1_term + sum_α).unsqueeze(1) + sum_β  # (csize, M_b)
        del sum_β
        scaled = α_sgn.unsqueeze(1) * β_sgn * e_total  # (csize, M_b)
        del β_sgn, e_total

        # Apply to T and scatter
        T_rows = T_2d[ket_alpha_per_exc[start_k:end_k]]  # (csize, M_b)
        contrib = scaled * T_rows  # (csize, M_b)
        del scaled, T_rows
        Wb_2d.index_add_(0, bra_alpha_per_exc[start_k:end_k], contrib)
        del contrib


def _beta_single_setup(
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    bra_beta_per_exc,
    cr,
    an,
):
    """T-independent per-excitation setup of the fast β-single kernel.

    Returns the β-sign, the ``h1 + sum_β`` terms, the α-side integral vector,
    the α-parity range mask and the α-occupancy matrix. Identical expressions
    to the ones the kernel used to inline, so a cached and an uncached matvec
    are bit-identical.

    Notes:
        For a β-single the parity range ``[min(cr,an), max(cr,an))`` is
        odd-odd, so the α-orbital range is ``[(lo+1)//2, (hi+1)//2)``. The
        ``sum_α`` term of the matrix element is the matmul
        ``α_int_vec @ α_occ_M.T`` over the ``L`` spatial orbitals.

        The ``range_mask_all`` entry holds the ``(n_exc, 2)`` ``[lo, hi)``
        pair under its historical name; the kernel builds the single-word
        mask or the per-limb masks from it, so one cache serves either string
        width.
    """
    device = bdets_t.device
    dtype = integrals.I2_eri.dtype

    # --- β-side sign and integral factors ---
    bra_beta_bm = bdets_t[bra_beta_per_exc]
    zero_alpha = torch.zeros_like(bra_beta_bm)
    bra_b = det_from_alpha_beta_batch(zero_alpha, bra_beta_bm, n_orb)

    start = torch.minimum(cr, an)
    end = torch.maximum(cr, an)
    sgn_init = torch.ones(bra_beta_per_exc.shape[0], dtype=dtype, device=device)
    β_sgn_all = parity_batch(bra_b, None, start, end, sgn_init)

    h1_term_all = integrals.I1[cr, an]  # (n_exc,)

    L = n_orb
    j_orb = torch.arange(L, dtype=torch.int64, device=device)
    j_β_2p1 = 2 * j_orb + 1
    j_α_2 = 2 * j_orb

    # β-side sum_β: (n_exc,) — depends on β-string only
    # Match one_excite_batch convention: I2_eri[creation, annihilation, j, j] - I2_eri[creation, j, j, annihilation]
    cr_b = cr.unsqueeze(-1)
    an_b = an.unsqueeze(-1)
    j_β_b = j_β_2p1.unsqueeze(0)
    I2_β_aijj = integrals.I2_eri[cr_b, an_b, j_β_b, j_β_b]
    I2_β_ajji = integrals.I2_eri[cr_b, j_β_b, j_β_b, an_b]
    β_int_vec = I2_β_aijj - I2_β_ajji  # (n_exc, L)
    β_occ_per_exc = _string_occ_matrix(bra_beta_bm, L).to(dtype)
    sum_β_all = (β_int_vec * β_occ_per_exc).sum(dim=-1)  # (n_exc,)

    # α-side integral vector for (n_exc × M_a) sum_α computation
    j_α_b = j_α_2.unsqueeze(0)
    I2_α_aijj = integrals.I2_eri[cr_b, an_b, j_α_b, j_α_b]
    I2_α_ajji = integrals.I2_eri[cr_b, j_α_b, j_α_b, an_b]
    α_int_vec_all = I2_α_aijj - I2_α_ajji  # (n_exc, L)

    # α-sign range bounds — for β-single, α-orbitals are at positions
    # (start+1, end+1) divided by 2. start and end are odd.
    range_lo_α_all = (start + 1) // 2  # equivalent to start//2 + 1 when start is odd
    range_hi_α_all = (end + 1) // 2

    # α-occupancy matrix (M_a, L)
    α_occ_M = _string_occ_matrix(adets_t, L).to(dtype)

    return {
        "β_sgn_all": β_sgn_all,
        "h1_term_all": h1_term_all,
        "sum_β_all": sum_β_all,
        "α_int_vec_all": α_int_vec_all,
        "range_mask_all": torch.stack((range_lo_α_all, range_hi_α_all), dim=-1),
        "α_occ_M": α_occ_M,
    }


def _apply_beta_single_chunk_fast(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    bra_beta_per_exc,
    ket_beta_per_exc,
    cr,
    an,
    M_a,
    M_b,
    chunk_size=2_000_000,
    cache=None,
    T_stride=None,
):
    """Fast β-single excitation kernel.

    Symmetric to :func:`_apply_alpha_single_chunk_fast`. β-single
    excitations have cr, an at odd spin-orbital positions = 2k+1.

    Parity range [min(cr, an), max(cr, an)) is odd-odd. α-bits in this
    range are at even positions; α-orbitals span [min(cr,an)//2+1,
    max(cr,an)//2+1) (the +1 because the spin-orbital range starts at
    an odd position, so the first even position inside it is start+1
    → α-orbital (start+1)//2).

    Args:
        cache: precomputed output of :func:`build_beta_pass_cache`; ``None``
            (default) recomputes the setup here.
    """
    n_exc_total = bra_beta_per_exc.shape[0]
    if n_exc_total == 0:
        return

    dtype = integrals.I2_eri.dtype

    if cache is None:
        setup = _beta_single_setup(adets_t, bdets_t, n_orb, integrals, bra_beta_per_exc, cr, an)
        β_sgn_all = setup["β_sgn_all"]
        h1_term_all = setup["h1_term_all"]
        sum_β_all = setup["sum_β_all"]
        α_int_vec_all = setup["α_int_vec_all"]
        range_all = setup["range_mask_all"]
        α_occ_M = setup["α_occ_M"]
    else:
        β_sgn_all = cache["β_sgn_all"]
        h1_term_all = cache["h1_term_all"]
        sum_β_all = cache["sum_β_all"]
        α_int_vec_all = cache["α_int_vec_all"]
        range_all = cache["range_mask_all"]
        α_occ_M = cache["α_occ_M"]

    # ``Wb`` is the output block (width ``M_b``); ``T`` may hold a column
    # subset of width ``T_stride`` (tile stage buffer).
    Wb_2d = Wb.view(M_a, M_b)
    T_2d = T.view(M_a, M_b if T_stride is None else T_stride)
    chunk_exc = max(1, chunk_size // max(M_a, 1))

    for start_k in range(0, n_exc_total, chunk_exc):
        end_k = min(start_k + chunk_exc, n_exc_total)

        β_sgn = β_sgn_all[start_k:end_k]
        h1_term = h1_term_all[start_k:end_k]
        sum_β = sum_β_all[start_k:end_k]
        α_int_vec = α_int_vec_all[start_k:end_k]  # (csize, L)
        # Each range entry is a (csize, 2) [lo, hi) pair.
        range_lo_α = range_all[start_k:end_k, 0].unsqueeze(1)  # (csize, 1)
        range_hi_α = range_all[start_k:end_k, 1].unsqueeze(1)

        α_pop = _string_range_popcount(adets_t, range_lo_α, range_hi_α)  # (csize, M_a)
        α_sgn = 1.0 - 2.0 * (α_pop & 1).to(dtype)
        del α_pop

        sum_α = α_int_vec @ α_occ_M.T  # (csize, M_a)

        e_total = (h1_term + sum_β).unsqueeze(1) + sum_α  # (csize, M_a)
        del sum_α
        # For β-singles, the scatter is along the β-index dimension of Wb_2d.
        # scaled[ia, k] = β_sgn[k] × α_sgn[k, ia] × e_total[k, ia]
        scaled = β_sgn.unsqueeze(1) * α_sgn * e_total  # (csize, M_a)
        del α_sgn, e_total

        T_cols = T_2d[:, ket_beta_per_exc[start_k:end_k]]  # (M_a, csize)
        contrib = (scaled * T_cols.T).T  # (M_a, csize)
        del scaled, T_cols
        Wb_2d.index_add_(1, bra_beta_per_exc[start_k:end_k], contrib)
        del contrib


def build_beta_pass_cache(
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    integrals: TorchIntegrals,
    helpers: Iterable[TorchHelper],
) -> dict:
    """Precompute the ``T``-independent setup of the full β pass.

    The β pass of :func:`mult_torch` (and of the distributed owner-computes
    operator) recomputes, for every excitation, the integral times sign
    factor, the parity range masks and the occupancy matrix on every matvec.
    None of those depend on the input vector, so they are built once here and
    handed to :func:`_mult_beta_only_torch` / the two fast β kernels through
    their ``cache=`` keyword.

    Args:
        adets_t: ``(M_a,)`` α-string bitmasks. The α-occupancy matrix is built
            from it: pass the local α slice when the output block is a row
            slice (owner-computes matvec).
        bdets_t: ``(M_b,)`` β-string bitmasks (always full).
        n_orb: spatial orbital count.
        integrals: the matvec's :class:`TorchIntegrals` bundle.
        helpers: iterable containing the ``task_type == 1`` β helper.

    Returns:
        Dict with the keys consumed by the kernels: ``bra_beta_per_exc``,
        ``β_sgn_all``, ``h1_term_all``, ``sum_β_all``, ``α_int_vec_all``,
        ``range_mask_all`` (β singles) and ``bra_beta_double_per_exc``,
        ``eij_β_all``, ``range1_mask_all``, ``range2_mask_all`` (β doubles),
        plus the shared ``α_occ_M``. Empty when no β helper is present.

    Notes:
        The cached tensors are exactly the ones the kernels computed inline,
        so ``cache=None`` and a cached call are bit-identical. Every term is
        an ``O(n_exc)`` rather than ``O(n_exc × M)`` quantity except the
        ``(M_a, L)`` α-occupancy matrix, so the cache is small next to the
        tables themselves.

        The three ``range*_mask_all`` entries hold ``(n_exc, 2)`` ``[lo, hi)``
        bit-range pairs under their historical names; the kernels turn each
        pair into the single-word mask or the per-limb masks, so the cache
        works for one-word and multi-limb strings alike. The per-excitation
        entries stay sliceable along their first axis.
    """
    helpers = tuple(helpers)
    out: dict = {}
    h = next((h for h in helpers if h.task_type == 1), None)
    if h is None:
        return out

    device = integrals.device
    M_b = int(bdets_t.shape[0])

    if h.b_singles_to is not None and h.b_singles_to.shape[0] > 0:
        bra_beta_per_exc = _bra_from_offsets(h.b_singles_offsets, M_b, device)
        out["bra_beta_per_exc"] = bra_beta_per_exc
        out.update(
            _beta_single_setup(
                adets_t, bdets_t, n_orb, integrals, bra_beta_per_exc, h.b_singles_cr, h.b_singles_an
            )
        )

    if h.b_doubles_to is not None and h.b_doubles_to.shape[0] > 0:
        bra_beta_per_exc = _bra_from_offsets(h.b_doubles_offsets, M_b, device)
        out["bra_beta_double_per_exc"] = bra_beta_per_exc
        out.update(
            _beta_double_setup(
                bdets_t,
                n_orb,
                integrals,
                bra_beta_per_exc,
                h.b_doubles_cr0,
                h.b_doubles_cr1,
                h.b_doubles_an0,
                h.b_doubles_an1,
            )
        )

    return out


# ============================================================================
# Per-task implementations (chunked)
# ============================================================================


def _mult_alpha_only_torch(Wb, T, adets_t, bdets_t, n_orb, integrals, h, M_a, M_b, chunk_size):
    """Task type 2: α-singles + α-doubles, β-string fixed.

    Chunks along α-exc entries. Each chunk lifts to size ``chunk × M_b``.
    """
    device = integrals.device

    # ---- α singles (fast factored path) ----
    if h.a_singles_to is not None and h.a_singles_to.shape[0] > 0:
        bra_alpha_per_exc = _bra_from_offsets(h.a_singles_offsets, M_a, device)

        t0 = _time.perf_counter()
        _apply_alpha_single_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_alpha_per_exc=bra_alpha_per_exc,
            ket_alpha_per_exc=h.a_singles_to,
            cr=h.a_singles_cr,
            an=h.a_singles_an,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
        )
        _sync(device)
        _matvec_profile_add("α-singles_apply", _time.perf_counter() - t0)

    # ---- α doubles (fast factored path: integral × α-sign computed once,
    # β-sign computed per β-string via cheap popcounts) ----
    if h.a_doubles_to is not None and h.a_doubles_to.shape[0] > 0:
        bra_alpha_per_exc = _bra_from_offsets(h.a_doubles_offsets, M_a, device)

        t0 = _time.perf_counter()
        _apply_alpha_double_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_alpha_per_exc=bra_alpha_per_exc,
            ket_alpha_per_exc=h.a_doubles_to,
            cr0=h.a_doubles_cr0,
            cr1=h.a_doubles_cr1,
            an0=h.a_doubles_an0,
            an1=h.a_doubles_an1,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
        )
        _sync(device)
        _matvec_profile_add("α-doubles_apply", _time.perf_counter() - t0)


def _mult_beta_only_torch(
    Wb,
    T,
    adets_t,
    bdets_t,
    n_orb,
    integrals,
    h,
    M_a,
    M_b,
    chunk_size,
    *,
    cache=None,
    T_stride=None,
):
    """Task type 1: β-singles + β-doubles, α-string fixed.

    ``cache`` is the output of :func:`build_beta_pass_cache` built from this
    helper's β tables; ``None`` recomputes the per-excitation setup, exactly
    as before.
    """
    device = integrals.device

    # ---- β singles (fast factored path) ----
    if h.b_singles_to is not None and h.b_singles_to.shape[0] > 0:
        if cache is None:
            bra_beta_per_exc = _bra_from_offsets(h.b_singles_offsets, M_b, device)
        else:
            bra_beta_per_exc = cache["bra_beta_per_exc"]

        t0 = _time.perf_counter()
        _apply_beta_single_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_beta_per_exc=bra_beta_per_exc,
            ket_beta_per_exc=h.b_singles_to,
            cr=h.b_singles_cr,
            an=h.b_singles_an,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
            cache=cache,
            T_stride=T_stride,
        )
        _sync(device)
        _matvec_profile_add("β-singles_apply", _time.perf_counter() - t0)

    # ---- β doubles (fast factored path) ----
    if h.b_doubles_to is not None and h.b_doubles_to.shape[0] > 0:
        if cache is None:
            bra_beta_per_exc = _bra_from_offsets(h.b_doubles_offsets, M_b, device)
        else:
            bra_beta_per_exc = cache["bra_beta_double_per_exc"]

        t0 = _time.perf_counter()
        _apply_beta_double_chunk_fast(
            Wb,
            T,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            bra_beta_per_exc=bra_beta_per_exc,
            ket_beta_per_exc=h.b_doubles_to,
            cr0=h.b_doubles_cr0,
            cr1=h.b_doubles_cr1,
            an0=h.b_doubles_an0,
            an1=h.b_doubles_an1,
            M_a=M_a,
            M_b=M_b,
            chunk_size=chunk_size,
            cache=cache,
            T_stride=T_stride,
        )
        _sync(device)
        _matvec_profile_add("β-doubles_apply", _time.perf_counter() - t0)


def _mult_alphabeta_torch(
    Wb, T, adets_t, bdets_t, integrals, h, M_a, M_b, chunk_size, *, T_stride=None
):
    """Task type 0: αβ-doubles via outer product of α-singles × β-singles.

    Uses the lean factored kernel that skips det_from_alpha_beta_batch
    interleaving and parity_batch overhead, computing parity directly
    from α/β strings via mask + popcount. Same memory profile as the
    slow path (per-pair, no cubes).
    """
    device = integrals.device
    if h.a_singles_to is None or h.b_singles_to is None:
        return
    if h.a_singles_to.shape[0] == 0 or h.b_singles_to.shape[0] == 0:
        return

    bra_alpha_per_exc = _bra_from_offsets(h.a_singles_offsets, M_a, device)
    bra_beta_per_exc = _bra_from_offsets(h.b_singles_offsets, M_b, device)

    t0 = _time.perf_counter()
    _apply_alphabeta_chunk_fast(
        Wb,
        T,
        adets_t,
        bdets_t,
        integrals,
        bra_alpha_per_a_exc=bra_alpha_per_exc,
        ket_alpha_per_a_exc=h.a_singles_to,
        cr_α=h.a_singles_cr,
        an_α=h.a_singles_an,
        bra_beta_per_b_exc=bra_beta_per_exc,
        ket_beta_per_b_exc=h.b_singles_to,
        cr_β=h.b_singles_cr,
        an_β=h.b_singles_an,
        M_a=M_a,
        M_b=M_b,
        chunk_size=chunk_size,
        T_stride=T_stride,
    )
    _sync(device)
    _matvec_profile_add("αβ_double_excite_chunk", _time.perf_counter() - t0)
