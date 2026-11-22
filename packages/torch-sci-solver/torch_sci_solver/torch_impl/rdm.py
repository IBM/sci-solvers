"""Reduced density matrix construction for SQD wavefunctions.

Three levels of detail, in increasing cost order:

  - ``make_one_rdm_diag_torch(eigvec, ...)`` — (γα_diag, γβ_diag), both
    (L,) tensors. This is what ``recover_configurations`` consumes as
    the SQD prior. No helper tables needed; O(M_a × L) work. **Diagonal
    occupations are convention-invariant** — squared coefficients make
    σ_I drop out.

  - ``make_one_rdm_torch(eigvec, ..., helpers)`` — full (γα, γβ), each
    (L, L). Matches PySCF's ``make_rdm1s``. Off-diagonals require
    spin-block parity AND a spin-block eigenvector (see "Convention"
    below). When eigvec comes from this package's matvec/Davidson,
    pass ``eigvec_in_interleaved_wick=True`` (default) and the wrapper
    will sign-convert to spin-block before constructing γ.

  - ``make_two_rdm_torch(eigvec, ..., helpers)`` — full (Γaa, Γab, Γbb),
    each (L, L, L, L) — PySCF chemist convention. Same eigenvector
    semantics as the 1-RDM wrapper.

  - ``energy_from_rdm(γ, Γ, integrals)`` cross-validation:
    E = I0 + Σ h(γα+γβ) + (1/2)Σ(pq|rs)(Γaa + 2Γab + Γbb) must equal
    the Davidson eigenvalue.

Convention
----------
- Spatial orbital indices ``p, q, r, s`` ∈ [0, L).
- Spin-orbital indices ``P, Q, R, S`` ∈ [0, 2L), with P=2p ↔ α-spin and
  P=2p+1 ↔ β-spin (same convention as the matvec).
- γα[p, q] = ⟨Ψ| a†_{p,α} a_{q,α} |Ψ⟩.
- 2-RDM in **chemist** notation (matches PySCF):
    Γ[p, q, r, s] = ⟨Ψ| a†_p a†_r a_s a_q |Ψ⟩.

Two Slater-determinant orderings are in play in this package:
- **Interleaved-Wick** (matvec convention):
    |I⟩_inter = a†_{P_1} a†_{P_2} ... |0⟩, P in interleaved
    spin-orbital order (P=2k or 2k+1, sorted as integers).
- **Spin-block** (PySCF convention):
    |I⟩_block = (Π α-creators in order)(Π β-creators in order) |0⟩.

The two differ by a per-state sign σ_I:
    σ_I = (-1)^{Σ_{α-orb j in I} (# β-orbs in I with bit index < j)}
and ``C_inter[I] = σ_I · C_block[I]``.

The matvec uses interleaved-Wick parity, so its eigenvector ``C`` lives
in the interleaved basis. The RDM construction below uses spin-block
parity (matches PySCF). To bridge the two, the public wrappers apply
``_inter_to_block_signs`` to the eigenvector by default. Set
``eigvec_in_interleaved_wick=False`` when feeding e.g. PySCF's civec
directly (which is already in spin-block).

The wavefunction is laid out as ``C[ia * M_b + ib] = ⟨adets[ia], bdets[ib] | Ψ⟩``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Iterable, Tuple

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import (
    _bit64_mask_below,
    _bra_from_offsets,
    _occupancy_bits,
    _popcount64,
    _string_limbs,
    parity_batch,
)
from torch_sci_solver.torch_impl.helpers import TorchHelper
from torch_sci_solver.torch_impl.integrals import TorchIntegrals, _dense_eri_from_integrals

# ============================================================================
# Convention bridge: interleaved-Wick eigenvector → spin-block eigenvector
# ============================================================================


def _inter_to_block_signs(adets_t: Tensor, bdets_t: Tensor, n_orb: int) -> Tensor:
    """Per-(ia, ib) sign factor σ such that ``C_block = σ * C_inter``.

    σ[ia, ib] = (-1)^{Σ_{α-orb j in adets[ia]} popcount(bdets[ib] & ((1<<j) - 1))}

    The strings may be ``(M,)`` single words or ``(M, n_limbs)`` int64 limb
    matrices (limb ``k`` holds bits ``[64k, 64k + 64)``). For a limb matrix the
    β popcount below orbital ``j`` is summed over limbs, each contributing the
    overlap of its 64-bit window with ``[0, j)``.

    Returns a (M_a, M_b) float tensor of ±1, matching adets_t.device.
    """
    device = adets_t.device
    L = n_orb
    if adets_t.ndim == 1 and bdets_t.ndim == 1:
        p_range = torch.arange(L, device=device, dtype=torch.int64)
        α_occ = ((adets_t.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(torch.int64)  # (M_a, L)
        # β_below[ib, j] = popcount(bdets[ib] & ((1 << j) - 1)) = # β-bits below spatial j.
        one = torch.tensor(1, device=device, dtype=torch.int64)
        masks_below = (one << p_range) - 1  # (L,)
        masked = bdets_t.unsqueeze(1) & masks_below.unsqueeze(0)  # (M_b, L)
        # popcount on int64; PyTorch ≥2.1 has torch.bitwise_popcount; fall back to bit twiddle.
        if hasattr(torch, "bitwise_popcount"):
            β_below = torch.bitwise_popcount(masked).to(torch.int64)  # (M_b, L)
        else:
            x = masked
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            β_below = ((x * 0x0101010101010101) >> 56).to(torch.int64)
    else:
        a_limbs = _string_limbs(adets_t)
        b_limbs = _string_limbs(bdets_t)
        p_range = torch.arange(L, device=device, dtype=torch.int64)
        α_occ = _occupancy_bits(a_limbs, L)
        β_below = torch.zeros((b_limbs.shape[0], L), dtype=torch.int64, device=device)
        for k in range(b_limbs.shape[1]):
            below = _bit64_mask_below(p_range - 64 * k).unsqueeze(0)  # (1, L)
            β_below += _popcount64(b_limbs[:, k].unsqueeze(1) & below)

    # σ_exp[ia, ib] = Σ_j α_occ[ia, j] × β_below[ib, j]
    # CUDA has no int64 matmul, so contract in float64 and round back. Values
    # are bounded by L², well inside float64's exact integer range for any
    # practical L. MPS has no float64 at all, so fall back to float32 there —
    # still exact for the same bounded values (well within float32's 2^24
    # integer range).
    contract_dtype = torch.float32 if device.type == "mps" else torch.float64
    σ_exp_f = α_occ.to(contract_dtype) @ β_below.to(contract_dtype).T  # (M_a, M_b)
    σ_exp = σ_exp_f.to(torch.int64)
    σ = 1.0 - 2.0 * (σ_exp & 1).to(contract_dtype)  # ±1
    return σ


# ============================================================================
# 1-RDM diagonal (existing fast path — kept for SQD-loop usage)
# ============================================================================


def make_one_rdm_diag_torch(
    eigvec: Tensor, adets_t: Tensor, bdets_t: Tensor, n_orb: int
) -> Tuple[Tensor, Tensor]:
    """Compute the diagonals of γα and γβ from the wavefunction.

    Args:
        eigvec:  (M_a * M_b,) torch tensor of CI coefficients.
        adets_t: (M_a,) α half-bitmasks, or (M_a, n_limbs) limb matrices.
        bdets_t: (M_b,) β half-bitmasks, or (M_b, n_limbs) limb matrices.
        n_orb:   number of spatial orbitals L.

    Returns:
        (occ_α, occ_β): each a (L,) torch tensor.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    M_a = adets_t.shape[0]
    M_b = bdets_t.shape[0]
    if eigvec.shape != (M_a * M_b,):
        raise ValueError(f"eigvec shape {eigvec.shape} != ({M_a * M_b},)")

    # |C|^2, not C^2 — occupation numbers are real even for complex C
    # (abs().pow(2) is a no-op vs. plain pow(2) for real C, so the real path
    # is unaffected).
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    C2 = eigvec.abs().pow(2)
    C2_2d = C2.view(M_a, M_b)
    weights_per_alpha = C2_2d.sum(dim=1)
    weights_per_beta = C2_2d.sum(dim=0)

    # Occupation-bit indicators are always real (0/1), independent of
    # eigvec's dtype — cast to the real counterpart of dtype rather than
    # dtype itself, so occ_a/occ_b (a weighted sum of real occupancies)
    # come out real, not complex-with-zero-imaginary-part.
    if adets_t.ndim == 1 and bdets_t.ndim == 1:
        p_range = torch.arange(n_orb, device=device, dtype=torch.int64)
        alpha_bits = ((adets_t.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(real_dtype)
        beta_bits = ((bdets_t.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(real_dtype)
    else:
        alpha_bits = _occupancy_bits(adets_t, n_orb).to(real_dtype)
        beta_bits = _occupancy_bits(bdets_t, n_orb).to(real_dtype)

    occ_a = alpha_bits.T @ weights_per_alpha
    occ_b = beta_bits.T @ weights_per_beta
    return occ_a, occ_b


# ============================================================================
# Helpers
# ============================================================================


def _find_helper(helpers, task_types_priority):
    """Pick first helper whose task_type is in the priority list and which
    has the requested arrays populated."""
    for tt in task_types_priority:
        for h in helpers:
            if h.task_type == tt:
                return h
    return None


# ============================================================================
# Full 1-RDM (diagonal + off-diagonal)
# ============================================================================


def make_one_rdm_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = 2_000_000,
    eigvec_in_interleaved_wick: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Full (γα, γβ) 1-RDMs in PySCF ``make_rdm1s`` format.

    Args:
        eigvec:    (M_a * M_b,) CI coefficients.
        adets_t:   (M_a,) int64 α half-bitmasks, or (M_a, n_limbs) limb
                   matrices (limb ``k`` holds bits ``[64k, 64k + 64)``).
        bdets_t:   (M_b,) int64 β half-bitmasks, or (M_b, n_limbs) limb
                   matrices.
        n_orb:     L = number of spatial orbitals.
        helpers:   iterable of TorchHelpers — must include either task_type 2
                   (α-singles + α-doubles) or task_type 0 (α-singles + β-singles)
                   for α off-diagonals, and similarly task_type 1 or 0 for β.
        chunk_size: max lifted entries per inner batched call.
        eigvec_in_interleaved_wick:
            If True (default), assume ``eigvec`` lives in the matvec's
            interleaved-Wick basis and convert to spin-block before
            constructing γ. Pass False when ``eigvec`` is already in
            spin-block convention (e.g., PySCF's civec, or a vector
            already converted).

    Returns:
        (γα, γβ), each (L, L) torch tensor on the same device/dtype as eigvec.
        Both are symmetric (for real C); diagonals satisfy
        ``γα.diag().sum() == n_α`` (electron-count conservation).
    """
    dtype = eigvec.dtype
    L = n_orb
    helpers = list(helpers)

    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs(adets_t, bdets_t, L).to(dtype=dtype)
        eigvec = eigvec * sigma.reshape(-1)

    # 1. Diagonal — copy from make_one_rdm_diag_torch. occ_a/occ_b are real
    # (occupation numbers), but gamma_a/gamma_b must be eigvec's own dtype
    # so the complex off-diagonal accumulation below (index_add_ with a
    # complex contrib) can write into them; a no-op cast for real eigvec.
    occ_a, occ_b = make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, n_orb)
    gamma_a = torch.diag(occ_a).to(dtype=dtype)
    gamma_b = torch.diag(occ_b).to(dtype=dtype)

    # 2. Off-diagonals from α-singles. Either task 2 or task 0 helper.
    h_a = _find_helper(helpers, [2, 0])
    if h_a is not None and h_a.a_singles_to is not None and h_a.a_singles_to.shape[0] > 0:
        _accumulate_singles_into_one_rdm(
            gamma_a, eigvec, adets_t, n_orb, h_a, chunk_size, beta=False
        )

    # 3. Off-diagonals from β-singles. Either task 1 or task 0.
    h_b = _find_helper(helpers, [1, 0])
    if h_b is not None and h_b.b_singles_to is not None and h_b.b_singles_to.shape[0] > 0:
        _accumulate_singles_into_one_rdm(
            gamma_b, eigvec, bdets_t, n_orb, h_b, chunk_size, beta=True
        )

    return gamma_a, gamma_b


def _accumulate_singles_into_one_rdm(gamma, eigvec, dets_t, n_orb, h, chunk_size, *, beta):
    """Add one spin block's singles contributions to ``gamma[p, q]``.

    ``dets_t`` holds the determinants of the spin being accumulated. ``beta``
    selects the β helper fields, the odd spin-orbital numbering and the
    transposed ``eigvec`` layout (β sums over the α index, α over β). p, q are
    spatial orbital indices.

    Uses **spin-block parity** (PySCF / standard quantum chemistry
    convention): the parity counts only the accumulated spin's bits in
    [min(p, q), max(p, q)) of the bra determinant. The other spin acts as a
    spectator that doesn't contribute to the anticommutation sign. This is
    essential for compatibility with PySCF's ``make_rdm1s`` output and for the
    energy formula E = Tr(γ h) + (1/2) Tr(Γ V) to match the Davidson
    eigenvalue.

    Note: this differs from the parity convention used inside the matvec
    (which is interleaved-Wick on the full det). Both produce the same
    Davidson eigenvalue, but the off-diagonals of γ differ between
    conventions. We use the spin-block convention here.
    """
    device = gamma.device
    dtype = gamma.dtype
    L = n_orb
    n_dets = dets_t.shape[0]

    if beta:
        offsets = h.b_singles_offsets
        to = h.b_singles_to
        cr_so = h.b_singles_cr
        an_so = h.b_singles_an
        # β spin-orbitals are odd (2k + 1), α are even (2k).
        spin_shift = 1
        # Sum over the α index ia: columns of eigvec are rows of its transpose.
        vec = eigvec.view(-1, n_dets).T
    else:
        offsets = h.a_singles_offsets
        to = h.a_singles_to
        cr_so = h.a_singles_cr
        an_so = h.a_singles_an
        spin_shift = 0
        # Sum over the β index ib: rows of eigvec.
        vec = eigvec.view(n_dets, -1)

    bra_per_exc = _bra_from_offsets(offsets, n_dets, device)
    n_exc = bra_per_exc.shape[0]

    chunk_eff = max(1, chunk_size // max(vec.shape[1], 1))

    gamma_flat = gamma.view(-1)

    for start in range(0, n_exc, chunk_eff):
        end = min(start + chunk_eff, n_exc)
        csize = end - start

        bra_chunk = bra_per_exc[start:end]
        to_chunk = to[start:end]
        cr_sp_chunk = (cr_so[start:end] - spin_shift) // 2
        an_sp_chunk = (an_so[start:end] - spin_shift) // 2

        # Parity uses the accumulated spin's determinants only (not the full
        # det) with SPATIAL indices: the other spin acts as spectator.
        bra_dets = dets_t[bra_chunk]
        start_sp = torch.minimum(cr_sp_chunk, an_sp_chunk)
        end_sp = torch.maximum(cr_sp_chunk, an_sp_chunk)
        sgn_per_exc = torch.ones(csize, dtype=dtype, device=device)
        sgn_per_exc = parity_batch(bra_dets, None, start_sp, end_sp, sgn_per_exc)

        # Per-excitation sum over the other spin's index:
        # Σ C*[bra, other] C[to, other] — the bra factor is conjugated
        # (no-op for real C).
        bra_rows = vec[bra_chunk]  # (csize, other-spin extent)
        to_rows = vec[to_chunk]
        sums = (bra_rows.conj() * to_rows).sum(dim=1)  # (csize,)

        contrib = sgn_per_exc * sums

        flat_idx = cr_sp_chunk * L + an_sp_chunk
        gamma_flat.index_add_(0, flat_idx, contrib)


# ============================================================================
# 2-RDM αβ block — PySCF chemist convention
# ============================================================================
#
# Γ_ab[p, q, r, s] = ⟨a†_pα a†_rβ a_sβ a_qα⟩
#
# In spin-block Wick (α and β as independent commuting sectors), the matrix
# element factorizes per basis-state pair:
#
#   ⟨I_α I_β| a†_pα a†_rβ a_sβ a_qα |J_α J_β⟩
#       = ⟨I_α| a†_p a_q |J_α⟩  ×  ⟨I_β| a†_r a_s |J_β⟩
#
# Four contribution classes, by (I_α=J_α vs not) × (I_β=J_β vs not):
#   A. (=, =): both diagonal → contributes only to Γ_ab[p, p, r, r]
#              for p occupied in α-side and r occupied in β-side.
#   B. (=, ≠): α-diagonal, β-single → contributes to Γ_ab[p, p, cr_β, an_β]
#              for each β-single helper entry, summed over occupied α-orbs p.
#   C. (≠, =): α-single, β-diagonal → contributes to Γ_ab[cr_α, an_α, r, r].
#   D. (≠, ≠): α-single × β-single (αβ-double) → Cartesian product over
#              the two helpers; contributes to Γ_ab[cr_α, an_α, cr_β, an_β].
#
# Storage convention: we accumulate at (cr_·, an_·) slots rather than (an_·, cr_·).
# By the exact (real-C) symmetry Γ_ab[p, q, r, s] = Γ_ab[q, p, s, r], these
# yield the same final tensor, and this matches the 1-RDM convention used
# in _accumulate_singles_into_one_rdm.
# ============================================================================


def make_two_rdm_ab_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = 2_000_000,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """αβ block of the 2-RDM in PySCF chemist convention.

    See ``make_one_rdm_torch`` for the meaning of
    ``eigvec_in_interleaved_wick``.

    Returns:
        Γ_ab of shape (L, L, L, L), matching PySCF ``make_rdm12s(...)[1][1]``.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = n_orb
    M_a = adets_t.shape[0]
    M_b = bdets_t.shape[0]
    helpers = list(helpers)

    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs(adets_t, bdets_t, L).to(dtype=dtype)
        eigvec = eigvec * sigma.reshape(-1)

    Gab = torch.zeros((L, L, L, L), dtype=dtype, device=device)

    h_a = _find_helper(helpers, [2, 0])
    h_b = _find_helper(helpers, [1, 0])

    # Class A — diagonal × diagonal
    _ab_class_a(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b)

    # Class B — α-diag × β-single
    if h_b is not None and h_b.b_singles_to is not None and h_b.b_singles_to.shape[0] > 0:
        _ab_class_b(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_b, chunk_size)

    # Class C — α-single × β-diag
    if h_a is not None and h_a.a_singles_to is not None and h_a.a_singles_to.shape[0] > 0:
        _ab_class_c(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_a, chunk_size)

    # Class D — α-single × β-single (αβ-double excitation)
    if (
        h_a is not None
        and h_a.a_singles_to is not None
        and h_a.a_singles_to.shape[0] > 0
        and h_b is not None
        and h_b.b_singles_to is not None
        and h_b.b_singles_to.shape[0] > 0
    ):
        _ab_class_d(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_a, h_b, chunk_size)

    return Gab


def _ab_class_a(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b):
    """Diagonal-diagonal: Γ_ab[p, p, r, r] += Σ |C|² occ_α(p) occ_β(r)."""
    device = Gab.device
    dtype = Gab.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    L = n_orb
    # Occupancy indicators and |C|^2 are both real; keep this matmul entirely
    # in the real dtype and cast the (real-valued) result once for the
    # complex-typed Gab scatter below.
    α_occ = _occupancy_bits(adets_t, L).to(real_dtype)  # (M_a, L)
    β_occ = _occupancy_bits(bdets_t, L).to(real_dtype)  # (M_b, L)
    # |C|^2, not C^2 — this class is fully diagonal (bra == ket on both
    # sides), so no conjugate/index-order distinction arises, only the
    # abs() (no-op for real C).
    C2 = eigvec.abs().pow(2).view(M_a, M_b)
    diag_ab = (α_occ.T @ C2 @ β_occ).to(dtype)  # (L, L), [p, r]
    # Scatter to Γ_ab[p, p, r, r]
    p_idx = torch.arange(L, device=device, dtype=torch.int64)
    r_idx = torch.arange(L, device=device, dtype=torch.int64)
    P, R = torch.meshgrid(p_idx, r_idx, indexing="ij")
    flat_idx = P.reshape(-1) * L**3 + P.reshape(-1) * L**2 + R.reshape(-1) * L + R.reshape(-1)
    Gab.view(-1).index_add_(0, flat_idx, diag_ab.reshape(-1))


def _ab_class_b(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_b, chunk_size):
    """α-diag × β-single: Γ_ab[p, p, cr_β, an_β] += sgn_β × Σ_{ia} occ_α(ia, p)
                                                     × C[ia, ib_β] × C[ia, jb_β].

    Note: cr_β, an_β are the SBD-helper "cr in bra, an in ket" indices, and we
    store at slot (cr_β, an_β) — see header note on transpose-pair convention.
    """
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    α_occ = _occupancy_bits(adets_t, L).to(dtype)  # (M_a, L)

    bra_β_per_exc = _bra_from_offsets(h_b.b_singles_offsets, M_b, device)
    n_β_exc = bra_β_per_exc.shape[0]

    cr_β_sp_all = (h_b.b_singles_cr - 1) // 2
    an_β_sp_all = (h_b.b_singles_an - 1) // 2

    eigvec_2d = eigvec.view(M_a, M_b)
    chunk_β = max(1, chunk_size // max(L, 1))

    Gab_flat = Gab.view(-1)
    for start in range(0, n_β_exc, chunk_β):
        end = min(start + chunk_β, n_β_exc)
        csize = end - start
        bra_β_chunk = bra_β_per_exc[start:end]
        jb_chunk = h_b.b_singles_to[start:end]
        cr_β_sp_chunk = cr_β_sp_all[start:end]
        an_β_sp_chunk = an_β_sp_all[start:end]

        # β parity (spin-block: bdets only, spatial indices)
        bra_β = bdets_t[bra_β_chunk]
        start_sp = torch.minimum(cr_β_sp_chunk, an_β_sp_chunk)
        end_sp = torch.maximum(cr_β_sp_chunk, an_β_sp_chunk)
        sgn_β = torch.ones(csize, dtype=dtype, device=device)
        sgn_β = parity_batch(bra_β, None, start_sp, end_sp, sgn_β)

        # ia_prod[ia, k] = C*[ia, ib_β] × C[ia, jb_β] — bra (ib_β = cr_β side)
        # conjugated; no-op for real C.
        bra_cols = eigvec_2d[:, bra_β_chunk]  # (M_a, csize)
        jb_cols = eigvec_2d[:, jb_chunk]  # (M_a, csize)
        ia_prod = bra_cols.conj() * jb_cols

        # v[p, k] = Σ_ia occ_α(ia, p) × ia_prod[ia, k]
        v = α_occ.T @ ia_prod  # (L, csize)

        # Add sgn_β[k] × v[p, k] to Γ_ab[p, p, cr_β[k], an_β[k]]
        p_idx = torch.arange(L, device=device, dtype=torch.int64)
        p_flat = p_idx.unsqueeze(1).expand(-1, csize).reshape(-1)
        cr_β_flat = cr_β_sp_chunk.unsqueeze(0).expand(L, -1).reshape(-1)
        an_β_flat = an_β_sp_chunk.unsqueeze(0).expand(L, -1).reshape(-1)
        sgn_flat = sgn_β.unsqueeze(0).expand(L, -1).reshape(-1)
        v_flat = v.reshape(-1)

        flat_idx = p_flat * L**3 + p_flat * L**2 + cr_β_flat * L + an_β_flat
        Gab_flat.index_add_(0, flat_idx, sgn_flat * v_flat)


def _ab_class_c(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_a, chunk_size):
    """α-single × β-diag: Γ_ab[cr_α, an_α, r, r] += sgn_α × Σ_{ib} occ_β(ib, r)
    × C[ia_α, ib] × C[ja_α, ib].
    """
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    β_occ = _occupancy_bits(bdets_t, L).to(dtype)  # (M_b, L)

    bra_α_per_exc = _bra_from_offsets(h_a.a_singles_offsets, M_a, device)
    n_α_exc = bra_α_per_exc.shape[0]

    cr_α_sp_all = h_a.a_singles_cr // 2
    an_α_sp_all = h_a.a_singles_an // 2

    eigvec_2d = eigvec.view(M_a, M_b)
    chunk_α = max(1, chunk_size // max(L, 1))

    Gab_flat = Gab.view(-1)
    for start in range(0, n_α_exc, chunk_α):
        end = min(start + chunk_α, n_α_exc)
        csize = end - start
        bra_α_chunk = bra_α_per_exc[start:end]
        ja_chunk = h_a.a_singles_to[start:end]
        cr_α_sp_chunk = cr_α_sp_all[start:end]
        an_α_sp_chunk = an_α_sp_all[start:end]

        bra_α = adets_t[bra_α_chunk]
        start_sp = torch.minimum(cr_α_sp_chunk, an_α_sp_chunk)
        end_sp = torch.maximum(cr_α_sp_chunk, an_α_sp_chunk)
        sgn_α = torch.ones(csize, dtype=dtype, device=device)
        sgn_α = parity_batch(bra_α, None, start_sp, end_sp, sgn_α)

        bra_rows = eigvec_2d[bra_α_chunk]  # (csize, M_b)
        ja_rows = eigvec_2d[ja_chunk]
        # bra (cr_α side) conjugated; no-op for real C.
        ib_prod = bra_rows.conj() * ja_rows  # (csize, M_b)

        # v[k, r] = Σ_ib occ_β(ib, r) × ib_prod[k, ib]
        v = ib_prod @ β_occ  # (csize, L)

        r_idx = torch.arange(L, device=device, dtype=torch.int64)
        r_flat = r_idx.unsqueeze(0).expand(csize, -1).reshape(-1)
        cr_α_flat = cr_α_sp_chunk.unsqueeze(1).expand(-1, L).reshape(-1)
        an_α_flat = an_α_sp_chunk.unsqueeze(1).expand(-1, L).reshape(-1)
        sgn_flat = sgn_α.unsqueeze(1).expand(-1, L).reshape(-1)
        v_flat = v.reshape(-1)

        flat_idx = cr_α_flat * L**3 + an_α_flat * L**2 + r_flat * L + r_flat
        Gab_flat.index_add_(0, flat_idx, sgn_flat * v_flat)


def _ab_class_d(Gab, eigvec, adets_t, bdets_t, n_orb, M_a, M_b, h_a, h_b, chunk_size):
    """α-single × β-single: Γ_ab[cr_α, an_α, cr_β, an_β] +=
    Σ_{α-entry, β-entry} sgn_α × sgn_β × C[ia_α, ib_β] × C[ja_α, jb_β].
    """
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb

    bra_α_per_exc = _bra_from_offsets(h_a.a_singles_offsets, M_a, device)
    bra_β_per_exc = _bra_from_offsets(h_b.b_singles_offsets, M_b, device)
    n_α_exc = bra_α_per_exc.shape[0]
    n_β_exc = bra_β_per_exc.shape[0]

    cr_α_sp_all = h_a.a_singles_cr // 2
    an_α_sp_all = h_a.a_singles_an // 2
    cr_β_sp_all = (h_b.b_singles_cr - 1) // 2
    an_β_sp_all = (h_b.b_singles_an - 1) // 2

    # Precompute α parities (cheap; n_α_exc entries)
    sgn_α_all = torch.ones(n_α_exc, dtype=dtype, device=device)
    bra_α = adets_t[bra_α_per_exc]
    start_α = torch.minimum(cr_α_sp_all, an_α_sp_all)
    end_α = torch.maximum(cr_α_sp_all, an_α_sp_all)
    sgn_α_all = parity_batch(bra_α, None, start_α, end_α, sgn_α_all)

    # Precompute β parities
    sgn_β_all = torch.ones(n_β_exc, dtype=dtype, device=device)
    bra_β = bdets_t[bra_β_per_exc]
    start_β = torch.minimum(cr_β_sp_all, an_β_sp_all)
    end_β = torch.maximum(cr_β_sp_all, an_β_sp_all)
    sgn_β_all = parity_batch(bra_β, None, start_β, end_β, sgn_β_all)

    eigvec_2d = eigvec.view(M_a, M_b)
    Gab_flat = Gab.view(-1)

    # Chunk over α-entries to bound (chunk_α × n_β_exc) memory
    chunk_α = max(1, chunk_size // max(n_β_exc, 1))
    for start in range(0, n_α_exc, chunk_α):
        end = min(start + chunk_α, n_α_exc)
        csize = end - start

        ia_chunk = bra_α_per_exc[start:end]
        ja_chunk = h_a.a_singles_to[start:end]
        cr_α_sp_chunk = cr_α_sp_all[start:end]
        an_α_sp_chunk = an_α_sp_all[start:end]
        sgn_α_chunk = sgn_α_all[start:end]

        # Gather C[ia_α, ib_β] and C[ja_α, jb_β] over all (k_α in chunk, k_β)
        # ia_rows: (csize, M_b); index by bra_β_per_exc (1D) gives (csize, n_β_exc)
        ia_rows = eigvec_2d[ia_chunk]  # (csize, M_b)
        ja_rows = eigvec_2d[ja_chunk]  # (csize, M_b)
        C_ia_ib = ia_rows[:, bra_β_per_exc]  # (csize, n_β_exc)
        C_ja_jb = ja_rows[:, h_b.b_singles_to]  # (csize, n_β_exc)

        # C_ia_ib is the bra amplitude (both ia_α, ib_β are bra-side indices)
        # and is conjugated; no-op for real C.
        contribs = (
            sgn_α_chunk.unsqueeze(1) * sgn_β_all.unsqueeze(0) * C_ia_ib.conj() * C_ja_jb
        )  # (csize, n_β_exc)

        # Scatter to Γ_ab[cr_α[k_α], an_α[k_α], cr_β[k_β], an_β[k_β]]
        cr_α_flat = cr_α_sp_chunk.unsqueeze(1).expand(-1, n_β_exc).reshape(-1)
        an_α_flat = an_α_sp_chunk.unsqueeze(1).expand(-1, n_β_exc).reshape(-1)
        cr_β_flat = cr_β_sp_all.unsqueeze(0).expand(csize, -1).reshape(-1)
        an_β_flat = an_β_sp_all.unsqueeze(0).expand(csize, -1).reshape(-1)

        flat_idx = cr_α_flat * L**3 + an_α_flat * L**2 + cr_β_flat * L + an_β_flat
        Gab_flat.index_add_(0, flat_idx, contribs.reshape(-1))


# ============================================================================
# 2-RDM αα and ββ blocks — PySCF chemist convention
# ============================================================================
#
# Γ_aa[p, q, r, s] = ⟨a†_pα a†_rα a_sα a_qα⟩
#
# Three contribution classes (β is spectator with |C|² weight):
#   Class 0: I_α = J_α  (operator must net-preserve the α-occupation)
#       Two sub-cases:
#         (p, p, r, r) with p ≠ r: +occ(p) occ(r)
#         (p, q, q, p) with p ≠ q: -occ(p) occ(q)
#       (Note: (p, p, p, p) ≡ 0 by Pauli.)
#   Class 1: single α-excitation × spectator α-orbital
#       4 sub-cases per (entry, spectator x) pair, signs ±sgn_helper.
#   Class 2: double α-excitation
#       4 sub-cases per entry: ±sgn_canonical (2-electron Slater-Condon).
# ============================================================================


def _make_two_rdm_same_spin_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int,
    eigvec_in_interleaved_wick: bool,
    is_alpha: bool,
) -> Tensor:
    """``αα`` or ``ββ`` block of the 2-RDM (the shared wrapper body).

    Args:
        eigvec: ``(M_a * M_b,)`` interleaved-Wick eigenvector.
        adets_t, bdets_t: α and β string tensors.
        n_orb: number of spatial orbitals.
        helpers: the excitation helpers.
        chunk_size: enumeration chunk size.
        eigvec_in_interleaved_wick: convert ``eigvec`` to spin-block first.
        is_alpha: build the αα block instead of the ββ one.

    Returns:
        The ``(L, L, L, L)`` same-spin block.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = n_orb
    M_a = adets_t.shape[0]
    M_b = bdets_t.shape[0]
    helpers = list(helpers)
    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs(adets_t, bdets_t, L).to(dtype=dtype)
        eigvec = eigvec * sigma.reshape(-1)

    G = torch.zeros((L, L, L, L), dtype=dtype, device=device)
    h = _find_helper(helpers, [2, 0] if is_alpha else [1, 0])
    spin_dets = adets_t if is_alpha else bdets_t
    class_0 = _aa_class_0 if is_alpha else _bb_class_0
    class_1 = _aa_class_1 if is_alpha else _bb_class_1
    class_2 = _aa_class_2 if is_alpha else _bb_class_2
    singles = "a_singles_to" if is_alpha else "b_singles_to"
    doubles = "a_doubles_to" if is_alpha else "b_doubles_to"

    class_0(G, eigvec, spin_dets, n_orb, M_a, M_b)
    if h is not None and getattr(h, singles) is not None and getattr(h, singles).shape[0] > 0:
        class_1(G, eigvec, spin_dets, n_orb, M_a, M_b, h, chunk_size)
    if h is not None and getattr(h, doubles) is not None and getattr(h, doubles).shape[0] > 0:
        class_2(G, eigvec, spin_dets, n_orb, M_a, M_b, h, chunk_size)
    return G


def make_two_rdm_aa_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = 2_000_000,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """αα block of the 2-RDM in PySCF chemist convention.

    Matches ``PySCF.fci.direct_spin1.FCI().make_rdm12s(...)[1][0]``.
    See ``make_one_rdm_torch`` re ``eigvec_in_interleaved_wick``.
    """
    return _make_two_rdm_same_spin_torch(
        eigvec,
        adets_t,
        bdets_t,
        n_orb,
        helpers,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        is_alpha=True,
    )


def make_two_rdm_bb_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = 2_000_000,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """ββ block of the 2-RDM — symmetric to αα with α↔β role swap."""
    return _make_two_rdm_same_spin_torch(
        eigvec,
        adets_t,
        bdets_t,
        n_orb,
        helpers,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        is_alpha=False,
    )


# ---------- αα blocks ----------


def _aa_class_0(Gaa, eigvec, adets_t, n_orb, M_a, M_b):
    """Diagonal: (p,p,r,r) += occ_α(p)occ_α(r) for p≠r;
                  (p,q,q,p) -= occ_α(p)occ_α(q) for p≠q.
    Both weighted by Σ_ib |C[ia,ib]|² per α-string."""
    L = n_orb
    device = Gaa.device
    dtype = Gaa.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    α_occ = _occupancy_bits(adets_t, L).to(real_dtype)  # (M_a, L)
    # |C|^2, not C^2 — fully diagonal class, no-op for real C (see _ab_class_a).
    C2 = eigvec.abs().pow(2).view(M_a, M_b)
    weights_per_α = C2.sum(dim=1)  # (M_a,)
    # diag_aa[p, r] = Σ_ia w_ia × occ_α(ia, p) × occ_α(ia, r); cast to the
    # (possibly complex) Gaa dtype once, for the index_add_ scatter below.
    diag_aa = ((α_occ.T * weights_per_α) @ α_occ).to(dtype)  # (L, L)

    eye = torch.eye(L, dtype=torch.bool, device=device)
    off = ~eye
    p_idx, r_idx = torch.where(off)  # both (L*(L-1),)

    # (p, p, r, r) += diag_aa[p, r]
    flat_a = p_idx * L**3 + p_idx * L**2 + r_idx * L + r_idx
    Gaa.view(-1).index_add_(0, flat_a, diag_aa[p_idx, r_idx])
    # (p, q, q, p) -= diag_aa[p, q]
    p_idx2, q_idx = torch.where(off)
    flat_b = p_idx2 * L**3 + q_idx * L**2 + q_idx * L + p_idx2
    Gaa.view(-1).index_add_(0, flat_b, -diag_aa[p_idx2, q_idx])


def _aa_class_1(Gaa, eigvec, adets_t, n_orb, M_a, M_b, h_a, chunk_size):
    """Single α-excitation × spectator α-orb x ∈ adets[ia] ∩ adets[ja].

    Four sub-cases per (entry, x):
        slot (cr, an, x, x)  → +sgn_helper
        slot (cr, x,  x, an) → -sgn_helper
        slot (x, an, cr, x)  → -sgn_helper
        slot (x, x,  cr, an) → +sgn_helper
    All weighted by Σ_ib C[ia, ib] × C[ja, ib] (β-spectator)."""
    L = n_orb
    device = Gaa.device
    dtype = Gaa.dtype
    α_occ = _occupancy_bits(adets_t, L).to(dtype)  # (M_a, L)

    bra_α_per_exc = _bra_from_offsets(h_a.a_singles_offsets, M_a, device)
    n_α_exc = bra_α_per_exc.shape[0]

    cr_α_sp_all = h_a.a_singles_cr // 2
    an_α_sp_all = h_a.a_singles_an // 2

    sgn_α_all = torch.ones(n_α_exc, dtype=dtype, device=device)
    bra_α = adets_t[bra_α_per_exc]
    start_α = torch.minimum(cr_α_sp_all, an_α_sp_all)
    end_α = torch.maximum(cr_α_sp_all, an_α_sp_all)
    sgn_α_all = parity_batch(bra_α, None, start_α, end_α, sgn_α_all)

    eigvec_2d = eigvec.view(M_a, M_b)
    Gaa_flat = Gaa.view(-1)

    chunk_α = max(1, chunk_size // max(L, 1))
    for start in range(0, n_α_exc, chunk_α):
        end = min(start + chunk_α, n_α_exc)
        csize = end - start

        ia_chunk = bra_α_per_exc[start:end]
        ja_chunk = h_a.a_singles_to[start:end]
        cr_α_sp_chunk = cr_α_sp_all[start:end]
        an_α_sp_chunk = an_α_sp_all[start:end]
        sgn_α_chunk = sgn_α_all[start:end]

        # ib_sums per entry (bra conjugated; no-op for real C)
        bra_rows = eigvec_2d[ia_chunk]
        ja_rows = eigvec_2d[ja_chunk]
        ib_sums = (bra_rows.conj() * ja_rows).sum(dim=1)  # (csize,)
        weight = sgn_α_chunk * ib_sums  # (csize,)

        # Spectator mask = (occupied in adets[ia]) AND (x ≠ cr).
        # By single-excitation structure, adets[ia] ∩ adets[ja] = adets[ia] - {cr}.
        ia_occ = α_occ[ia_chunk]  # (csize, L)
        cr_mask = torch.zeros((csize, L), dtype=dtype, device=device)
        cr_mask.scatter_(1, cr_α_sp_chunk.unsqueeze(1), 1.0)
        spec_mask = ia_occ * (1.0 - cr_mask)  # (csize, L)
        contrib = weight.unsqueeze(1) * spec_mask  # (csize, L)

        cr_idx = cr_α_sp_chunk.unsqueeze(1).expand(-1, L)
        an_idx = an_α_sp_chunk.unsqueeze(1).expand(-1, L)
        x_idx = torch.arange(L, device=device, dtype=torch.int64).unsqueeze(0).expand(csize, -1)

        # Slot 1: (cr, an, x, x) +=
        flat_1 = (cr_idx * L**3 + an_idx * L**2 + x_idx * L + x_idx).reshape(-1)
        Gaa_flat.index_add_(0, flat_1, contrib.reshape(-1))
        # Slot 2: (cr, x, x, an) -=
        flat_2 = (cr_idx * L**3 + x_idx * L**2 + x_idx * L + an_idx).reshape(-1)
        Gaa_flat.index_add_(0, flat_2, -contrib.reshape(-1))
        # Slot 3: (x, an, cr, x) -=
        flat_3 = (x_idx * L**3 + an_idx * L**2 + cr_idx * L + x_idx).reshape(-1)
        Gaa_flat.index_add_(0, flat_3, -contrib.reshape(-1))
        # Slot 4: (x, x, cr, an) +=
        flat_4 = (x_idx * L**3 + x_idx * L**2 + cr_idx * L + an_idx).reshape(-1)
        Gaa_flat.index_add_(0, flat_4, contrib.reshape(-1))


def _aa_class_2(Gaa, eigvec, adets_t, n_orb, M_a, M_b, h_a, chunk_size):
    """Double α-excitation. For each entry (ia, ja, cr0, cr1, an0, an1):

        Γ_aa[cr0, an0, cr1, an1] += sgn × Σ_ib C[ia, ib] C[ja, ib]
        Γ_aa[cr0, an1, cr1, an0] -= "
        Γ_aa[cr1, an0, cr0, an1] -= "
        Γ_aa[cr1, an1, cr0, an0] += "

    sgn = spin-block 2-electron Slater-Condon parity on adets[ia] with spatial idx.
    (Same algebraic form as ``two_excite_batch`` but adets-only.)"""
    L = n_orb
    device = Gaa.device
    dtype = Gaa.dtype

    bra_α_per_exc = _bra_from_offsets(h_a.a_doubles_offsets, M_a, device)
    n_α_exc = bra_α_per_exc.shape[0]
    if n_α_exc == 0:
        return

    cr0_sp = h_a.a_doubles_cr0 // 2
    cr1_sp = h_a.a_doubles_cr1 // 2
    an0_sp = h_a.a_doubles_an0 // 2
    an1_sp = h_a.a_doubles_an1 // 2

    # Spin-block canonical sign (analog of two_excite_batch on adets/spatial)
    bra_all = adets_t[bra_α_per_exc]
    I_sp = torch.minimum(cr0_sp, cr1_sp)
    J_sp = torch.maximum(cr0_sp, cr1_sp)
    A_sp = torch.minimum(an0_sp, an1_sp)
    B_sp = torch.maximum(an0_sp, an1_sp)
    sgn_canonical = torch.ones(n_α_exc, dtype=dtype, device=device)
    sgn_canonical = parity_batch(
        bra_all, None, torch.minimum(I_sp, A_sp), torch.maximum(I_sp, A_sp), sgn_canonical
    )
    sgn_canonical = parity_batch(
        bra_all, None, torch.minimum(J_sp, B_sp), torch.maximum(J_sp, B_sp), sgn_canonical
    )
    skip = (A_sp > J_sp) | (B_sp < I_sp)
    sgn_canonical = torch.where(skip, -sgn_canonical, sgn_canonical)

    eigvec_2d = eigvec.view(M_a, M_b)
    Gaa_flat = Gaa.view(-1)

    chunk_α = max(1, chunk_size // max(M_b, 1))
    for start in range(0, n_α_exc, chunk_α):
        end = min(start + chunk_α, n_α_exc)

        ia_chunk = bra_α_per_exc[start:end]
        ja_chunk = h_a.a_doubles_to[start:end]
        cr0_chunk = cr0_sp[start:end]
        cr1_chunk = cr1_sp[start:end]
        an0_chunk = an0_sp[start:end]
        an1_chunk = an1_sp[start:end]
        sgn_chunk = sgn_canonical[start:end]

        bra_rows = eigvec_2d[ia_chunk]
        ja_rows = eigvec_2d[ja_chunk]
        ib_sums = (bra_rows.conj() * ja_rows).sum(dim=1)  # bra conjugated
        contrib_pos = sgn_chunk * ib_sums

        # 4 slots
        f1 = cr0_chunk * L**3 + an0_chunk * L**2 + cr1_chunk * L + an1_chunk
        f2 = cr0_chunk * L**3 + an1_chunk * L**2 + cr1_chunk * L + an0_chunk
        f3 = cr1_chunk * L**3 + an0_chunk * L**2 + cr0_chunk * L + an1_chunk
        f4 = cr1_chunk * L**3 + an1_chunk * L**2 + cr0_chunk * L + an0_chunk
        Gaa_flat.index_add_(0, f1, contrib_pos)
        Gaa_flat.index_add_(0, f2, -contrib_pos)
        Gaa_flat.index_add_(0, f3, -contrib_pos)
        Gaa_flat.index_add_(0, f4, contrib_pos)


# ---------- ββ blocks (mirror of αα with α↔β roles) ----------


def _bb_class_0(Gbb, eigvec, bdets_t, n_orb, M_a, M_b):
    """Bb class 0."""
    L = n_orb
    device = Gbb.device
    dtype = Gbb.dtype
    real_dtype = dtype.to_real() if dtype.is_complex else dtype
    β_occ = _occupancy_bits(bdets_t, L).to(real_dtype)
    # |C|^2, not C^2 — fully diagonal class, no-op for real C (see _ab_class_a).
    C2 = eigvec.abs().pow(2).view(M_a, M_b)
    weights_per_β = C2.sum(dim=0)  # (M_b,)
    diag_bb = ((β_occ.T * weights_per_β) @ β_occ).to(dtype)
    off = ~torch.eye(L, dtype=torch.bool, device=device)
    p_idx, r_idx = torch.where(off)
    flat_a = p_idx * L**3 + p_idx * L**2 + r_idx * L + r_idx
    Gbb.view(-1).index_add_(0, flat_a, diag_bb[p_idx, r_idx])
    p2, q2 = torch.where(off)
    flat_b = p2 * L**3 + q2 * L**2 + q2 * L + p2
    Gbb.view(-1).index_add_(0, flat_b, -diag_bb[p2, q2])


def _bb_class_1(Gbb, eigvec, bdets_t, n_orb, M_a, M_b, h_b, chunk_size):
    """Bb class 1."""
    L = n_orb
    device = Gbb.device
    dtype = Gbb.dtype
    β_occ = _occupancy_bits(bdets_t, L).to(dtype)

    bra_β_per_exc = _bra_from_offsets(h_b.b_singles_offsets, M_b, device)
    n_β_exc = bra_β_per_exc.shape[0]

    cr_β_sp_all = (h_b.b_singles_cr - 1) // 2
    an_β_sp_all = (h_b.b_singles_an - 1) // 2

    sgn_β_all = torch.ones(n_β_exc, dtype=dtype, device=device)
    bra_β = bdets_t[bra_β_per_exc]
    sgn_β_all = parity_batch(
        bra_β,
        None,
        torch.minimum(cr_β_sp_all, an_β_sp_all),
        torch.maximum(cr_β_sp_all, an_β_sp_all),
        sgn_β_all,
    )

    eigvec_2d = eigvec.view(M_a, M_b)
    Gbb_flat = Gbb.view(-1)
    chunk_β = max(1, chunk_size // max(L, 1))
    for start in range(0, n_β_exc, chunk_β):
        end = min(start + chunk_β, n_β_exc)
        csize = end - start
        ib_chunk = bra_β_per_exc[start:end]
        jb_chunk = h_b.b_singles_to[start:end]
        cr_β_sp_chunk = cr_β_sp_all[start:end]
        an_β_sp_chunk = an_β_sp_all[start:end]
        sgn_β_chunk = sgn_β_all[start:end]

        # ia_sums = Σ_ia C*[ia, ib] × C[ia, jb] — bra conjugated, no-op for real C
        bra_cols = eigvec_2d[:, ib_chunk]
        jb_cols = eigvec_2d[:, jb_chunk]
        ia_sums = (bra_cols.conj() * jb_cols).sum(dim=0)
        weight = sgn_β_chunk * ia_sums

        ib_occ = β_occ[ib_chunk]
        cr_mask = torch.zeros((csize, L), dtype=dtype, device=device)
        cr_mask.scatter_(1, cr_β_sp_chunk.unsqueeze(1), 1.0)
        spec_mask = ib_occ * (1.0 - cr_mask)
        contrib = weight.unsqueeze(1) * spec_mask

        cr_idx = cr_β_sp_chunk.unsqueeze(1).expand(-1, L)
        an_idx = an_β_sp_chunk.unsqueeze(1).expand(-1, L)
        x_idx = torch.arange(L, device=device, dtype=torch.int64).unsqueeze(0).expand(csize, -1)
        Gbb_flat.index_add_(
            0, (cr_idx * L**3 + an_idx * L**2 + x_idx * L + x_idx).reshape(-1), contrib.reshape(-1)
        )
        Gbb_flat.index_add_(
            0, (cr_idx * L**3 + x_idx * L**2 + x_idx * L + an_idx).reshape(-1), -contrib.reshape(-1)
        )
        Gbb_flat.index_add_(
            0, (x_idx * L**3 + an_idx * L**2 + cr_idx * L + x_idx).reshape(-1), -contrib.reshape(-1)
        )
        Gbb_flat.index_add_(
            0, (x_idx * L**3 + x_idx * L**2 + cr_idx * L + an_idx).reshape(-1), contrib.reshape(-1)
        )


def _bb_class_2(Gbb, eigvec, bdets_t, n_orb, M_a, M_b, h_b, chunk_size):
    """Bb class 2."""
    L = n_orb
    device = Gbb.device
    dtype = Gbb.dtype

    bra_β_per_exc = _bra_from_offsets(h_b.b_doubles_offsets, M_b, device)
    n_β_exc = bra_β_per_exc.shape[0]
    if n_β_exc == 0:
        return

    cr0_sp = (h_b.b_doubles_cr0 - 1) // 2
    cr1_sp = (h_b.b_doubles_cr1 - 1) // 2
    an0_sp = (h_b.b_doubles_an0 - 1) // 2
    an1_sp = (h_b.b_doubles_an1 - 1) // 2

    bra_all = bdets_t[bra_β_per_exc]
    I_sp = torch.minimum(cr0_sp, cr1_sp)
    J_sp = torch.maximum(cr0_sp, cr1_sp)
    A_sp = torch.minimum(an0_sp, an1_sp)
    B_sp = torch.maximum(an0_sp, an1_sp)
    sgn_canonical = torch.ones(n_β_exc, dtype=dtype, device=device)
    sgn_canonical = parity_batch(
        bra_all, None, torch.minimum(I_sp, A_sp), torch.maximum(I_sp, A_sp), sgn_canonical
    )
    sgn_canonical = parity_batch(
        bra_all, None, torch.minimum(J_sp, B_sp), torch.maximum(J_sp, B_sp), sgn_canonical
    )
    skip = (A_sp > J_sp) | (B_sp < I_sp)
    sgn_canonical = torch.where(skip, -sgn_canonical, sgn_canonical)

    eigvec_2d = eigvec.view(M_a, M_b)
    Gbb_flat = Gbb.view(-1)
    chunk_β = max(1, chunk_size // max(M_a, 1))
    for start in range(0, n_β_exc, chunk_β):
        end = min(start + chunk_β, n_β_exc)
        ib_chunk = bra_β_per_exc[start:end]
        jb_chunk = h_b.b_doubles_to[start:end]
        cr0_chunk = cr0_sp[start:end]
        cr1_chunk = cr1_sp[start:end]
        an0_chunk = an0_sp[start:end]
        an1_chunk = an1_sp[start:end]
        sgn_chunk = sgn_canonical[start:end]

        bra_cols = eigvec_2d[:, ib_chunk]
        jb_cols = eigvec_2d[:, jb_chunk]
        ia_sums = (bra_cols.conj() * jb_cols).sum(dim=0)  # bra conjugated
        contrib_pos = sgn_chunk * ia_sums
        f1 = cr0_chunk * L**3 + an0_chunk * L**2 + cr1_chunk * L + an1_chunk
        f2 = cr0_chunk * L**3 + an1_chunk * L**2 + cr1_chunk * L + an0_chunk
        f3 = cr1_chunk * L**3 + an0_chunk * L**2 + cr0_chunk * L + an1_chunk
        f4 = cr1_chunk * L**3 + an1_chunk * L**2 + cr0_chunk * L + an0_chunk
        Gbb_flat.index_add_(0, f1, contrib_pos)
        Gbb_flat.index_add_(0, f2, -contrib_pos)
        Gbb_flat.index_add_(0, f3, -contrib_pos)
        Gbb_flat.index_add_(0, f4, contrib_pos)


# ============================================================================
# Full 2-RDM API + energy reconstruction
# ============================================================================


def make_two_rdm_torch(
    eigvec: Tensor,
    adets_t: Tensor,
    bdets_t: Tensor,
    n_orb: int,
    helpers: Iterable[TorchHelper],
    *,
    chunk_size: int = 2_000_000,
    eigvec_in_interleaved_wick: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Full (Γaa, Γab, Γbb), PySCF chemist convention.

    Each tensor is (L, L, L, L) and matches PySCF ``make_rdm12s(...)[1]``.

    See ``make_one_rdm_torch`` for the meaning of
    ``eigvec_in_interleaved_wick``.

    Energy reconstruction:
        E = I0 + Σ h[p,q] (γα + γβ)[p,q]
              + (1/2) Σ (pq|rs) (Γaa + 2 Γab + Γbb)[p,q,r,s]
    must equal the Davidson eigenvalue.
    """
    helpers = list(helpers)
    # Convert once here to avoid 3× redundant sign flips inside the block fns.
    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs(adets_t, bdets_t, n_orb).to(dtype=eigvec.dtype)
        eigvec = eigvec * sigma.reshape(-1)
    Gaa = make_two_rdm_aa_torch(
        eigvec,
        adets_t,
        bdets_t,
        n_orb,
        helpers,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    Gab = make_two_rdm_ab_torch(
        eigvec,
        adets_t,
        bdets_t,
        n_orb,
        helpers,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    Gbb = make_two_rdm_bb_torch(
        eigvec,
        adets_t,
        bdets_t,
        n_orb,
        helpers,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    return Gaa, Gab, Gbb


def energy_from_rdm(
    gamma_a: Tensor,
    gamma_b: Tensor,
    Gaa: Tensor,
    Gab: Tensor,
    Gbb: Tensor,
    integrals: TorchIntegrals,
) -> Tensor:
    """E = I0 + Σ h(γα+γβ) + (1/2) Σ (pq|rs)(Γaa + 2Γab + Γbb).

    The integrals are stored in spin-orbital form (2L × 2L and (2L)^4);
    we extract the spatial (αα-block) to act on spatial-indexed γ and Γ.
    For non-magnetic systems all four (αα|αα), (αα|ββ), (ββ|αα), (ββ|ββ)
    ERI blocks equal the spatial chemist tensor (pq|rs), so the spatial
    formula above subsumes all four cross-spin sums into one. ``I2_eri``
    may be a ``CompactEri``; the compact form is materialised with
    ``CompactEri.to_dense()`` before the slice."""
    h1 = integrals.I1[0::2, 0::2]  # (L, L)
    eri = _dense_eri_from_integrals(integrals)[0::2, 0::2, 0::2, 0::2]  # (L, L, L, L)
    e1 = (h1 * (gamma_a + gamma_b)).sum()
    e2 = 0.5 * (eri * (Gaa + 2.0 * Gab + Gbb)).sum()
    energy = integrals.I0 + e1 + e2
    # Mathematically real for Hermitian integrals + correctly-built RDMs;
    # .real drops any roundoff-induced imaginary residue (no-op for a
    # real-dtype energy already).
    return energy.real if energy.is_complex() else energy


def s_squared_from_rdm(Gab: Tensor, n_a: int, n_b: int) -> Tensor:
    """⟨Ψ|S²|Ψ⟩ from the spin-resolved 1-RDMs and the αβ 2-RDM.

    For a spin-free Hamiltonian, S² = S_+ S_- + S_z² - S_z. The αβ block
    of the 2-RDM carries the only off-diagonal information needed:

        ⟨S²⟩ = (S_z² − S_z) + N_α − Σ_{p,q} Γ_αβ[p, q, q, p]

    where S_z = (N_α − N_β)/2. Γ_αβ follows PySCF chemist convention
    (Γ_αβ[p, q, r, s] = ⟨a†_pα a†_rβ a_sβ a_qα⟩), the same convention
    produced by ``make_two_rdm_ab_torch``.

    For a pure singlet (S=0) the result is 0; pure triplet (S=1) is 2;
    pure quintet (S=2) is 6 — the standard PySCF ``spin_square0``
    convention, i.e. S² as the eigenvalue S(S+1).

    Args:
        Gab:      (L, L, L, L) — αβ 2-RDM in PySCF chemist convention.
        n_a, n_b: integers — α and β electron counts. (Passed
                  explicitly rather than read from γ traces so the
                  caller controls the sector.)

    Returns scalar tensor on the same device/dtype as Gab.
    """
    n_alpha_t = torch.tensor(float(n_a), dtype=Gab.dtype, device=Gab.device)
    s_z = (n_a - n_b) / 2.0
    sz_sq_minus_sz = torch.tensor(s_z * s_z - s_z, dtype=Gab.dtype, device=Gab.device)
    cross_term = torch.einsum("pqqp->", Gab)
    s_sq = sz_sq_minus_sz + n_alpha_t - cross_term
    # Mathematically real; .real is a no-op for a real-dtype Gab.
    return s_sq.real if s_sq.is_complex() else s_sq
