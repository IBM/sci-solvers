"""Pair-mode (no-product-space) reduced density matrix builders.

Product-mode versions in :mod:`torch_sci_solver.torch_impl.rdm` operate on
the ``(M_α, M_β)``-reshaped eigenvector; pair mode instead scatter-adds over
the length-N pair basis using the pair-keyed excitation tables built by
:mod:`torch_sci_solver.torch_impl.pair_helpers`.

The pair-mode kernels are structurally simpler than the product-mode ones:
each excitation-table row is *already* a single-scalar contribution (the
"sum over the spectator basis" that product-mode does with
``eigvec.view(M_a, M_b).sum(dim=1)`` is replaced by the pair helper's
one-row-per-shared-spectator enumeration). The RDM builders here look like:

    for each row (k_bra, k_ket, cr, an) in the α-singles pair table:
        sgn = spin_block_parity(α_of_pair[k_bra], cr_sp, an_sp)
        γα[cr_sp, an_sp] += sgn × eigvec[k_bra] × eigvec[k_ket]

with chunking to bound per-call intermediate memory (all classes cost
O(K × L) or O(K) per chunk, where K is the pair-table entry count).

A :class:`StructuralPairBasis` has no per-pair arrays, so it is handled by a
separate block-walking path (the "Structural (block-described) basis" section):
the diagonal terms iterate over the Cartesian blocks and the excitation rows
are re-derived from the O(M_unique) unique-string arrays plus the exact
``by_alpha`` / ``by_beta`` block grouping. That path allocates O(chunk) rows
per step and never a length-N temporary; its results match the explicit pair
list's, because each block group is a full product whose spectator axis
carries every partner of its strings.

Convention
----------
The RDM code uses spin-block parity (α- and β-only parities over spatial
indices, matching PySCF's ``make_rdm1s`` / ``make_rdm12s``). The matvec
uses interleaved-Wick parity on the full det. To bridge, the public
wrappers apply :func:`_inter_to_block_signs_pair` to the eigenvector by
default. Set ``eigvec_in_interleaved_wick=False`` when passing a vector
that is already in spin-block form.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Iterator, Tuple

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import _occupancy_bits, parity_batch
from torch_sci_solver.torch_impl.pair_blocks import PairBlocks, build_pair_blocks
from torch_sci_solver.torch_impl.pair_helpers import (
    PairHelpers,
    _bit_mask_rows,
    _complement_mask,
    _dispatch_bit_positions,
    _dispatch_found_in,
    _sort_strings,
    _triu_indices,
)
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis

DEFAULT_CHUNK_SIZE = 2_000_000

# Pairs per interleaved-Wick → spin-block sign chunk. Bounds the transient
# ``(rows, n_orb)`` occupancy workspace the structural builders use when they
# scale a row slice's amplitudes.
_SIGN_CHUNK = 1 << 16


# ======================================================================
# Convention bridge: interleaved-Wick eigenvector → spin-block eigenvector
# ======================================================================


def _inter_to_block_signs_pair(alpha_of_pair: Tensor, beta_of_pair: Tensor, n_orb: int) -> Tensor:
    """Per-pair sign σ_k such that ``C_block[k] = σ_k × C_inter[k]``.

    Pair-mode analog of :func:`torch_sci_solver.torch_impl.rdm._inter_to_block_signs`.
    The product-mode version returns an ``(M_α, M_β)`` matrix and relies on
    the α × β outer-product structure of the basis; pair mode evaluates
    the same formula per-pair and returns a length-N vector.

        σ_k = (-1)^{Σ_{α-orb j ∈ α_of_pair[k]} popcount(β_of_pair[k] & ((1<<j) - 1))}

    The strings may be 1-D single words or ``(N, n_limbs)`` limb matrices;
    on the limb path the β-popcount below ``j`` is summed over limbs by an
    exclusive prefix sum of the β-occupancy bits.
    """
    device = alpha_of_pair.device
    L = n_orb
    if alpha_of_pair.dim() == 1:
        p_range = torch.arange(L, device=device, dtype=torch.int64)
        # (N, L) α-occupancy at each spatial index.
        alpha_occ = ((alpha_of_pair.unsqueeze(1) >> p_range.unsqueeze(0)) & 1).to(torch.int64)
        one = torch.tensor(1, device=device, dtype=torch.int64)
        masks_below = (one << p_range) - 1  # (L,) — mask of bits strictly below j
        masked = beta_of_pair.unsqueeze(1) & masks_below.unsqueeze(0)  # (N, L)
        if hasattr(torch, "bitwise_popcount"):
            beta_below = torch.bitwise_popcount(masked).to(torch.int64)
        else:
            x = masked
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            beta_below = ((x * 0x0101010101010101) >> 56).to(torch.int64)
    else:
        alpha_occ = _occupancy_bits(alpha_of_pair, L)
        beta_occ = _occupancy_bits(beta_of_pair, L)
        # Bits of β strictly below each spatial index: exclusive prefix sum.
        beta_below = torch.cumsum(beta_occ, dim=1) - beta_occ
    # σ_exp[k] = Σ_j α_occ[k, j] × β_below[k, j]. Values are bounded by
    # L², well inside int64 (and float64's exact integer range) for any
    # practical L, so the summation stays exact.
    sigma_exp = (alpha_occ * beta_below).sum(dim=1)
    # MPS has no float64 support; both dtypes represent these signs exactly.
    sign_dtype = torch.float32 if device.type == "mps" else torch.float64
    return 1.0 - 2.0 * (sigma_exp & 1).to(sign_dtype)


# ======================================================================
# Full 1-RDM: γα, γβ in PySCF ``make_rdm1s`` format
# ======================================================================


def make_one_rdm_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eigvec_in_interleaved_wick: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Full ``(γα, γβ)`` 1-RDMs in PySCF format, pair-mode analog of
    :func:`torch_sci_solver.torch_impl.rdm.make_one_rdm_torch`.

    Returns each block as a length-``(L, L)`` symmetric tensor with
    diagonal traces equal to the electron counts.

    A :class:`StructuralPairBasis` is walked block by block: the build
    allocates O(chunk_size + M_unique) workspace and never a length-N array
    (cf. :func:`_make_one_rdm_structural`).
    """
    if isinstance(pair_basis, StructuralPairBasis):
        return _make_one_rdm_structural(
            eigvec,
            pair_basis,
            pair_helpers,
            n_orb,
            chunk_size=chunk_size,
            eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        )
    device = eigvec.device
    dtype = eigvec.dtype
    L = n_orb
    alpha_of_pair = pair_basis.alpha_of_pair.to(device)
    beta_of_pair = pair_basis.beta_of_pair.to(device)

    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs_pair(alpha_of_pair, beta_of_pair, L).to(dtype=dtype)
        eigvec = eigvec * sigma

    # Diagonals — pair-mode weighted occupancy.
    occ_a, occ_b = make_one_rdm_diag_pair_torch(eigvec, alpha_of_pair, beta_of_pair, L)
    gamma_a = torch.diag(occ_a)
    gamma_b = torch.diag(occ_b)

    for t in pair_helpers.asingles():
        _acc_alpha_singles_1rdm_pair(
            gamma_a=gamma_a,
            eigvec=eigvec,
            alpha_of_pair=alpha_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            n_orb=L,
            chunk_size=chunk_size,
        )
    for t in pair_helpers.bsingles():
        _acc_beta_singles_1rdm_pair(
            gamma_b=gamma_b,
            eigvec=eigvec,
            beta_of_pair=beta_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            n_orb=L,
            chunk_size=chunk_size,
        )
    return gamma_a, gamma_b


def make_one_rdm_diag_pair_torch(
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    n_orb: int,
) -> Tuple[Tensor, Tensor]:
    """Diagonal 1-RDM in pair mode: per-orbital α- and β-spin occupancies.

    The diagonal is convention-invariant (σ² = 1), so the caller may pass
    the eigenvector in either interleaved-Wick or spin-block form. The
    strings may be 1-D single words or ``(N, n_limbs)`` limb matrices.
    """
    dtype = eigvec.dtype
    # |C_k|^2 (real for complex C via conj*C, plain square for real C — byte-identical).
    weights = (eigvec.conj() * eigvec).to(dtype)
    alpha_bits = _occupancy_bits(alpha_of_pair, n_orb).to(dtype)
    beta_bits = _occupancy_bits(beta_of_pair, n_orb).to(dtype)
    occ_a = (weights.unsqueeze(-1) * alpha_bits).sum(dim=0)
    occ_b = (weights.unsqueeze(-1) * beta_bits).sum(dim=0)
    return occ_a, occ_b


def _acc_alpha_singles_1rdm_pair(
    *,
    gamma_a: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    chunk_size: int,
) -> None:
    """γα[cr_sp, an_sp] += sgn_α × C[k_bra] × C[k_ket] per α-single entry.

    Uses spin-block parity on the α-string of the bra pair with spatial
    indices — the "α acts alone, β spectator" convention that matches
    PySCF ``make_rdm1s``. Each pair-table row is one scalar contribution
    (the shared-β enumeration was done at helper-build time).
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = gamma_a.device
    dtype = gamma_a.dtype
    L = n_orb
    gamma_flat = gamma_a.view(-1)
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        cr_sp = cr[start:end] // 2  # α spin-orbital = 2*spatial
        an_sp = an[start:end] // 2
        bra_alpha = alpha_of_pair[kb]
        start_sp = torch.minimum(cr_sp, an_sp)
        end_sp = torch.maximum(cr_sp, an_sp)
        sgn = torch.ones(kb.shape[0], dtype=dtype, device=device)
        sgn = parity_batch(bra_alpha, None, start_sp, end_sp, sgn)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        contrib = sgn * eigvec[kb].conj() * eigvec[kk]
        flat_idx = cr_sp * L + an_sp
        gamma_flat.index_add_(0, flat_idx, contrib)


def _acc_beta_singles_1rdm_pair(
    *,
    gamma_b: Tensor,
    eigvec: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    chunk_size: int,
) -> None:
    """γβ[cr_sp, an_sp] += sgn_β × C[k_bra] × C[k_ket] per β-single entry."""
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = gamma_b.device
    dtype = gamma_b.dtype
    L = n_orb
    gamma_flat = gamma_b.view(-1)
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        cr_sp = (cr[start:end] - 1) // 2  # β spin-orbital = 2*spatial + 1
        an_sp = (an[start:end] - 1) // 2
        bra_beta = beta_of_pair[kb]
        start_sp = torch.minimum(cr_sp, an_sp)
        end_sp = torch.maximum(cr_sp, an_sp)
        sgn = torch.ones(kb.shape[0], dtype=dtype, device=device)
        sgn = parity_batch(bra_beta, None, start_sp, end_sp, sgn)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        contrib = sgn * eigvec[kb].conj() * eigvec[kk]
        flat_idx = cr_sp * L + an_sp
        gamma_flat.index_add_(0, flat_idx, contrib)


# ======================================================================
# 2-RDM αβ block — PySCF chemist convention
# ======================================================================


def make_two_rdm_ab_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """αβ block of the 2-RDM, pair-mode analog of
    :func:`torch_sci_solver.torch_impl.rdm.make_two_rdm_ab_torch`.

    Same four-class decomposition (diag×diag, α-diag × β-single,
    α-single × β-diag, α-single × β-single) as product mode, but each
    class replaces its "sum over the spectator basis" with a per-entry
    scalar contribution from the pair-keyed excitation tables.

    A :class:`StructuralPairBasis` is walked block by block with no
    length-N temporary; see :func:`_make_two_rdm_ab_structural`.
    """
    if isinstance(pair_basis, StructuralPairBasis):
        return _make_two_rdm_ab_structural(
            eigvec,
            pair_basis,
            pair_helpers,
            n_orb,
            chunk_size=chunk_size,
            eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        )
    device = eigvec.device
    dtype = eigvec.dtype
    L = n_orb
    alpha_of_pair = pair_basis.alpha_of_pair.to(device)
    beta_of_pair = pair_basis.beta_of_pair.to(device)
    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs_pair(alpha_of_pair, beta_of_pair, L).to(dtype=dtype)
        eigvec = eigvec * sigma

    Gab = torch.zeros((L, L, L, L), dtype=dtype, device=device)

    # Class A: (α diag, β diag) — Γ_ab[p, p, r, r] += Σ_k |C_k|² occ_α(k,p) occ_β(k,r)
    _ab_class_a_pair(Gab, eigvec, alpha_of_pair, beta_of_pair, L)

    # Class B: (α diag, β single) — Γ_ab[p, p, cr_β, an_β] += sgn_β occ_α(k_bra,p) C[k_bra] C[k_ket]
    for t in pair_helpers.bsingles():
        _ab_class_b_pair(
            Gab=Gab,
            eigvec=eigvec,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            n_orb=L,
            chunk_size=chunk_size,
        )

    # Class C: (α single, β diag) — Γ_ab[cr_α, an_α, r, r] += sgn_α occ_β(k_bra,r) C[k_bra] C[k_ket]
    for t in pair_helpers.asingles():
        _ab_class_c_pair(
            Gab=Gab,
            eigvec=eigvec,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            n_orb=L,
            chunk_size=chunk_size,
        )

    # Class D: (α single, β single) — Γ_ab[cr_α, an_α, cr_β, an_β] += sgn_α sgn_β C[k_bra] C[k_ket]
    for t in pair_helpers.ab_doubles():
        _ab_class_d_pair(
            Gab=Gab,
            eigvec=eigvec,
            alpha_of_pair=alpha_of_pair,
            beta_of_pair=beta_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr_alpha=t["cr_a"],
            an_alpha=t["an_a"],
            cr_beta=t["cr_b"],
            an_beta=t["an_b"],
            n_orb=L,
            chunk_size=chunk_size,
        )
    return Gab


def _ab_class_a_pair(
    Gab: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    n_orb: int,
) -> None:
    """Γ_ab[p, p, r, r] += Σ_k |C_k|² × α_occ(k, p) × β_occ(k, r)."""
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    alpha_occ = _occupancy_bits(alpha_of_pair, L).to(dtype)  # (N, L)
    beta_occ = _occupancy_bits(beta_of_pair, L).to(dtype)  # (N, L)
    # |C_k|^2 (real for complex C via conj*C, plain square for real C — byte-identical).
    weights = (eigvec.conj() * eigvec).to(dtype).unsqueeze(-1)  # (N, 1)
    # diag_ab[p, r] = Σ_k w_k × α_occ[k, p] × β_occ[k, r]
    diag_ab = (alpha_occ * weights).T @ beta_occ  # (L, L)
    p_idx = torch.arange(L, device=device, dtype=torch.int64)
    P, R = torch.meshgrid(p_idx, p_idx, indexing="ij")
    flat_idx = P.reshape(-1) * L**3 + P.reshape(-1) * L**2 + R.reshape(-1) * L + R.reshape(-1)
    Gab.view(-1).index_add_(0, flat_idx, diag_ab.reshape(-1))


def _ab_class_b_pair(
    *,
    Gab: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    chunk_size: int,
) -> None:
    """Γ_ab[p, p, cr_β_sp, an_β_sp] += sgn_β × α_occ(bra_α, p) × C[k_bra] × C[k_ket].

    ``alpha_of_pair[k_bra] == alpha_of_pair[k_ket]`` for β-single pair entries
    (same α, β differs by a single excitation), so the α-occupancy vector is
    a per-entry (not per-pair) quantity.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    p_range = torch.arange(L, device=device, dtype=torch.int64)
    Gab_flat = Gab.view(-1)
    chunk = max(1, int(chunk_size // max(L, 1)))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        csize = end - start
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        cr_sp = (cr[start:end] - 1) // 2
        an_sp = (an[start:end] - 1) // 2
        # β-only parity on β_of_pair[k_bra], spatial indices.
        bra_beta = beta_of_pair[kb]
        start_sp = torch.minimum(cr_sp, an_sp)
        end_sp = torch.maximum(cr_sp, an_sp)
        sgn = torch.ones(csize, dtype=dtype, device=device)
        sgn = parity_batch(bra_beta, None, start_sp, end_sp, sgn)
        # α-occupancy of the bra pair's α-string (α doesn't change).
        bra_alpha_bm = alpha_of_pair[kb]  # (csize,) or (csize, n_limbs)
        alpha_occ_chunk = _occupancy_bits(bra_alpha_bm, L).to(dtype)  # (csize, L)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        weight = sgn * eigvec[kb].conj() * eigvec[kk]  # (csize,)
        contrib = weight.unsqueeze(1) * alpha_occ_chunk  # (csize, L)

        p_flat = p_range.unsqueeze(0).expand(csize, -1).reshape(-1)
        cr_flat = cr_sp.unsqueeze(1).expand(-1, L).reshape(-1)
        an_flat = an_sp.unsqueeze(1).expand(-1, L).reshape(-1)
        flat_idx = p_flat * L**3 + p_flat * L**2 + cr_flat * L + an_flat
        Gab_flat.index_add_(0, flat_idx, contrib.reshape(-1))


def _ab_class_c_pair(
    *,
    Gab: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    chunk_size: int,
) -> None:
    """Γ_ab[cr_α_sp, an_α_sp, r, r] += sgn_α × β_occ(bra_β, r) × C[k_bra] × C[k_ket]."""
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    p_range = torch.arange(L, device=device, dtype=torch.int64)
    Gab_flat = Gab.view(-1)
    chunk = max(1, int(chunk_size // max(L, 1)))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        csize = end - start
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        cr_sp = cr[start:end] // 2
        an_sp = an[start:end] // 2
        bra_alpha = alpha_of_pair[kb]
        start_sp = torch.minimum(cr_sp, an_sp)
        end_sp = torch.maximum(cr_sp, an_sp)
        sgn = torch.ones(csize, dtype=dtype, device=device)
        sgn = parity_batch(bra_alpha, None, start_sp, end_sp, sgn)
        bra_beta_bm = beta_of_pair[kb]
        beta_occ_chunk = _occupancy_bits(bra_beta_bm, L).to(dtype)  # (csize, L)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        weight = sgn * eigvec[kb].conj() * eigvec[kk]
        contrib = weight.unsqueeze(1) * beta_occ_chunk  # (csize, L)

        r_flat = p_range.unsqueeze(0).expand(csize, -1).reshape(-1)
        cr_flat = cr_sp.unsqueeze(1).expand(-1, L).reshape(-1)
        an_flat = an_sp.unsqueeze(1).expand(-1, L).reshape(-1)
        flat_idx = cr_flat * L**3 + an_flat * L**2 + r_flat * L + r_flat
        Gab_flat.index_add_(0, flat_idx, contrib.reshape(-1))


def _ab_class_d_pair(
    *,
    Gab: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr_alpha: Tensor,
    an_alpha: Tensor,
    cr_beta: Tensor,
    an_beta: Tensor,
    n_orb: int,
    chunk_size: int,
) -> None:
    """Γ_ab[cr_α_sp, an_α_sp, cr_β_sp, an_β_sp] += sgn_α × sgn_β × C[k_bra] × C[k_ket]."""
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = Gab.device
    dtype = Gab.dtype
    L = n_orb
    Gab_flat = Gab.view(-1)
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        csize = end - start
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        cr_a_sp = cr_alpha[start:end] // 2
        an_a_sp = an_alpha[start:end] // 2
        cr_b_sp = (cr_beta[start:end] - 1) // 2
        an_b_sp = (an_beta[start:end] - 1) // 2
        bra_alpha = alpha_of_pair[kb]
        bra_beta = beta_of_pair[kb]
        sgn_a = torch.ones(csize, dtype=dtype, device=device)
        sgn_a = parity_batch(
            bra_alpha,
            None,
            torch.minimum(cr_a_sp, an_a_sp),
            torch.maximum(cr_a_sp, an_a_sp),
            sgn_a,
        )
        sgn_b = torch.ones(csize, dtype=dtype, device=device)
        sgn_b = parity_batch(
            bra_beta,
            None,
            torch.minimum(cr_b_sp, an_b_sp),
            torch.maximum(cr_b_sp, an_b_sp),
            sgn_b,
        )
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        contrib = sgn_a * sgn_b * eigvec[kb].conj() * eigvec[kk]
        flat_idx = cr_a_sp * L**3 + an_a_sp * L**2 + cr_b_sp * L + an_b_sp
        Gab_flat.index_add_(0, flat_idx, contrib)


# ======================================================================
# 2-RDM αα and ββ blocks — PySCF chemist convention
# ======================================================================


def _make_two_rdm_same_spin_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int,
    eigvec_in_interleaved_wick: bool,
    is_alpha: bool,
) -> Tensor:
    """``αα`` or ``ββ`` block on an explicit pair basis (shared wrapper body).

    Args:
        eigvec: ``(N,)`` interleaved-Wick eigenvector.
        pair_basis: the explicit pair basis.
        pair_helpers: excitation tables for the basis.
        n_orb: number of spatial orbitals.
        chunk_size: enumeration chunk size.
        eigvec_in_interleaved_wick: convert ``eigvec`` to spin-block first.
        is_alpha: build the αα block instead of the ββ one.

    Returns:
        The ``(L, L, L, L)`` same-spin block.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = n_orb
    alpha_of_pair = pair_basis.alpha_of_pair.to(device)
    beta_of_pair = pair_basis.beta_of_pair.to(device)
    if eigvec_in_interleaved_wick:
        sigma = _inter_to_block_signs_pair(alpha_of_pair, beta_of_pair, L).to(dtype=dtype)
        eigvec = eigvec * sigma
    spin_of_pair = alpha_of_pair if is_alpha else beta_of_pair
    extra = {} if is_alpha else {"cr_is_beta": True}
    G = torch.zeros((L, L, L, L), dtype=dtype, device=device)
    _aa_class_0_pair(G, eigvec, spin_of_pair, L)
    singles = pair_helpers.asingles() if is_alpha else pair_helpers.bsingles()
    for t in singles:
        _aa_class_1_pair(
            Gaa=G,
            eigvec=eigvec,
            alpha_of_pair=spin_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr=t["cr0"],
            an=t["an0"],
            n_orb=L,
            chunk_size=chunk_size,
            **extra,
        )
    doubles = pair_helpers.adoubles() if is_alpha else pair_helpers.bdoubles()
    for t in doubles:
        _aa_class_2_pair(
            Gaa=G,
            eigvec=eigvec,
            alpha_of_pair=spin_of_pair,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr0=t["cr0"],
            cr1=t["cr1"],
            an0=t["an0"],
            an1=t["an1"],
            n_orb=L,
            chunk_size=chunk_size,
            **extra,
        )
    return G


def make_two_rdm_aa_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """αα block, pair-mode analog of
    :func:`torch_sci_solver.torch_impl.rdm.make_two_rdm_aa_torch`.

    A :class:`StructuralPairBasis` is walked block by block with no
    length-N temporary; see :func:`_make_two_rdm_same_spin_structural`.
    """
    if isinstance(pair_basis, StructuralPairBasis):
        return _make_two_rdm_same_spin_structural(
            eigvec,
            pair_basis,
            pair_helpers,
            n_orb,
            chunk_size=chunk_size,
            eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
            is_alpha=True,
        )
    return _make_two_rdm_same_spin_pair_torch(
        eigvec,
        pair_basis,
        pair_helpers,
        n_orb,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        is_alpha=True,
    )


def make_two_rdm_bb_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eigvec_in_interleaved_wick: bool = True,
) -> Tensor:
    """ββ block — mirror of αα with α↔β roles swapped."""
    if isinstance(pair_basis, StructuralPairBasis):
        return _make_two_rdm_same_spin_structural(
            eigvec,
            pair_basis,
            pair_helpers,
            n_orb,
            chunk_size=chunk_size,
            eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
            is_alpha=False,
        )
    return _make_two_rdm_same_spin_pair_torch(
        eigvec,
        pair_basis,
        pair_helpers,
        n_orb,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
        is_alpha=False,
    )


def _aa_class_0_pair(Gaa: Tensor, eigvec: Tensor, spin_of_pair: Tensor, n_orb: int) -> None:
    """Diagonal same-spin: for each pair k, |C_k|² occ(k, p) occ(k, r) scatters
    into (p, p, r, r) with p ≠ r and into (p, q, q, p) with -sign, p ≠ q.

    Works for either αα or ββ blocks by passing the corresponding
    ``spin_of_pair`` array (``alpha_of_pair`` or ``beta_of_pair``).
    """
    device = Gaa.device
    dtype = Gaa.dtype
    L = n_orb
    occ = _occupancy_bits(spin_of_pair, L).to(dtype)  # (N, L)
    # |C_k|^2 (real for complex C via conj*C, plain square for real C — byte-identical).
    weights = (eigvec.conj() * eigvec).to(dtype)  # (N,)
    diag = (occ.T * weights) @ occ  # (L, L) — Σ_k w_k occ(k,p) occ(k,r)
    off = ~torch.eye(L, dtype=torch.bool, device=device)
    p_idx, r_idx = torch.where(off)
    flat_a = p_idx * L**3 + p_idx * L**2 + r_idx * L + r_idx
    Gaa.view(-1).index_add_(0, flat_a, diag[p_idx, r_idx])
    flat_b = p_idx * L**3 + r_idx * L**2 + r_idx * L + p_idx
    Gaa.view(-1).index_add_(0, flat_b, -diag[p_idx, r_idx])


def _aa_class_1_pair(
    *,
    Gaa: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr: Tensor,
    an: Tensor,
    n_orb: int,
    chunk_size: int,
    cr_is_beta: bool = False,
) -> None:
    """Single α-excitation × spectator α-orbital x ∈ occ(α_bra) \\ {cr}.

    Same four-slot scatter as product mode; the ``spec_mask`` (occupied
    in α-bra AND not the creation orbital) is built per-entry from the
    pair's α-string. ``cr_is_beta=True`` converts β spin-orbital indices
    ``2k+1`` to spatial via ``(cr - 1) // 2`` — reuses this helper for
    the ββ class 1 with β replacing α.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = Gaa.device
    dtype = Gaa.dtype
    L = n_orb
    p_range = torch.arange(L, device=device, dtype=torch.int64)
    Gaa_flat = Gaa.view(-1)
    chunk = max(1, int(chunk_size // max(L, 1)))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        csize = end - start
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        if cr_is_beta:
            cr_sp = (cr[start:end] - 1) // 2
            an_sp = (an[start:end] - 1) // 2
        else:
            cr_sp = cr[start:end] // 2
            an_sp = an[start:end] // 2
        bra = alpha_of_pair[kb]
        start_sp = torch.minimum(cr_sp, an_sp)
        end_sp = torch.maximum(cr_sp, an_sp)
        sgn = torch.ones(csize, dtype=dtype, device=device)
        sgn = parity_batch(bra, None, start_sp, end_sp, sgn)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        weight = sgn * eigvec[kb].conj() * eigvec[kk]  # (csize,)

        bra_occ = _occupancy_bits(bra, L).to(dtype)  # (csize, L)
        cr_mask = torch.zeros((csize, L), dtype=dtype, device=device)
        cr_mask.scatter_(1, cr_sp.unsqueeze(1), 1.0)
        spec_mask = bra_occ * (1.0 - cr_mask)
        contrib = weight.unsqueeze(1) * spec_mask  # (csize, L)

        cr_idx = cr_sp.unsqueeze(1).expand(-1, L)
        an_idx = an_sp.unsqueeze(1).expand(-1, L)
        x_idx = p_range.unsqueeze(0).expand(csize, -1)
        flat_1 = (cr_idx * L**3 + an_idx * L**2 + x_idx * L + x_idx).reshape(-1)
        flat_2 = (cr_idx * L**3 + x_idx * L**2 + x_idx * L + an_idx).reshape(-1)
        flat_3 = (x_idx * L**3 + an_idx * L**2 + cr_idx * L + x_idx).reshape(-1)
        flat_4 = (x_idx * L**3 + x_idx * L**2 + cr_idx * L + an_idx).reshape(-1)
        c_flat = contrib.reshape(-1)
        Gaa_flat.index_add_(0, flat_1, c_flat)
        Gaa_flat.index_add_(0, flat_2, -c_flat)
        Gaa_flat.index_add_(0, flat_3, -c_flat)
        Gaa_flat.index_add_(0, flat_4, c_flat)


def _aa_class_2_pair(
    *,
    Gaa: Tensor,
    eigvec: Tensor,
    alpha_of_pair: Tensor,
    k_bra: Tensor,
    k_ket: Tensor,
    cr0: Tensor,
    cr1: Tensor,
    an0: Tensor,
    an1: Tensor,
    n_orb: int,
    chunk_size: int,
    cr_is_beta: bool = False,
) -> None:
    """Double α-excitation. Four scatter slots with alternating signs,
    same as product-mode ``_aa_class_2``. Reused for ββ with
    ``cr_is_beta=True``.
    """
    k_total = int(k_bra.shape[0])
    if k_total == 0:
        return
    device = Gaa.device
    dtype = Gaa.dtype
    L = n_orb
    Gaa_flat = Gaa.view(-1)
    chunk = max(1, int(chunk_size))
    for start in range(0, k_total, chunk):
        end = min(start + chunk, k_total)
        csize = end - start
        kb = k_bra[start:end]
        kk = k_ket[start:end]
        if cr_is_beta:
            cr0_sp = (cr0[start:end] - 1) // 2
            cr1_sp = (cr1[start:end] - 1) // 2
            an0_sp = (an0[start:end] - 1) // 2
            an1_sp = (an1[start:end] - 1) // 2
        else:
            cr0_sp = cr0[start:end] // 2
            cr1_sp = cr1[start:end] // 2
            an0_sp = an0[start:end] // 2
            an1_sp = an1[start:end] // 2
        bra = alpha_of_pair[kb]
        I_sp = torch.minimum(cr0_sp, cr1_sp)
        J_sp = torch.maximum(cr0_sp, cr1_sp)
        A_sp = torch.minimum(an0_sp, an1_sp)
        B_sp = torch.maximum(an0_sp, an1_sp)
        sgn = torch.ones(csize, dtype=dtype, device=device)
        sgn = parity_batch(bra, None, torch.minimum(I_sp, A_sp), torch.maximum(I_sp, A_sp), sgn)
        sgn = parity_batch(bra, None, torch.minimum(J_sp, B_sp), torch.maximum(J_sp, B_sp), sgn)
        skip = (A_sp > J_sp) | (B_sp < I_sp)
        sgn = torch.where(skip, -sgn, sgn)
        # conj() on the bra amplitude: no-op for real C, conjugate-correct for complex.
        contrib_pos = sgn * eigvec[kb].conj() * eigvec[kk]
        f1 = cr0_sp * L**3 + an0_sp * L**2 + cr1_sp * L + an1_sp
        f2 = cr0_sp * L**3 + an1_sp * L**2 + cr1_sp * L + an0_sp
        f3 = cr1_sp * L**3 + an0_sp * L**2 + cr0_sp * L + an1_sp
        f4 = cr1_sp * L**3 + an1_sp * L**2 + cr0_sp * L + an0_sp
        Gaa_flat.index_add_(0, f1, contrib_pos)
        Gaa_flat.index_add_(0, f2, -contrib_pos)
        Gaa_flat.index_add_(0, f3, -contrib_pos)
        Gaa_flat.index_add_(0, f4, contrib_pos)


# ======================================================================
# Full 2-RDM wrapper
# ======================================================================


def make_two_rdm_pair_torch(
    eigvec: Tensor,
    pair_basis: "PairBasis | StructuralPairBasis",
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eigvec_in_interleaved_wick: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Full ``(Γaa, Γab, Γbb)`` in PySCF chemist convention.

    Convention detail — apply σ once at the wrapper level and pass
    ``eigvec_in_interleaved_wick=False`` to the three block builders,
    matching :func:`torch_sci_solver.torch_impl.rdm.make_two_rdm_torch`
    (avoids three redundant sign flips).

    For a :class:`StructuralPairBasis` σ cannot be materialised as a
    length-N vector, so each block builder applies it per row instead; the
    three builds together allocate O(chunk_size + M_unique) workspace and
    no length-N array.
    """
    if isinstance(pair_basis, StructuralPairBasis):
        return (
            _make_two_rdm_same_spin_structural(
                eigvec,
                pair_basis,
                pair_helpers,
                n_orb,
                chunk_size=chunk_size,
                eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
                is_alpha=True,
            ),
            _make_two_rdm_ab_structural(
                eigvec,
                pair_basis,
                pair_helpers,
                n_orb,
                chunk_size=chunk_size,
                eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
            ),
            _make_two_rdm_same_spin_structural(
                eigvec,
                pair_basis,
                pair_helpers,
                n_orb,
                chunk_size=chunk_size,
                eigvec_in_interleaved_wick=eigvec_in_interleaved_wick,
                is_alpha=False,
            ),
        )
    if eigvec_in_interleaved_wick:
        alpha_of_pair = pair_basis.alpha_of_pair.to(eigvec.device)
        beta_of_pair = pair_basis.beta_of_pair.to(eigvec.device)
        sigma = _inter_to_block_signs_pair(alpha_of_pair, beta_of_pair, n_orb).to(
            dtype=eigvec.dtype
        )
        eigvec = eigvec * sigma
    Gaa = make_two_rdm_aa_pair_torch(
        eigvec,
        pair_basis,
        pair_helpers,
        n_orb,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    Gab = make_two_rdm_ab_pair_torch(
        eigvec,
        pair_basis,
        pair_helpers,
        n_orb,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    Gbb = make_two_rdm_bb_pair_torch(
        eigvec,
        pair_basis,
        pair_helpers,
        n_orb,
        chunk_size=chunk_size,
        eigvec_in_interleaved_wick=False,
    )
    return Gaa, Gab, Gbb


# ======================================================================
# Structural (block-described) basis
# ======================================================================
#
# ``StructuralPairBasis`` stores Cartesian blocks instead of a length-N pair
# list, so the builders below walk the blocks: every loop allocates O(chunk)
# rows and every table is re-derived from the O(M_unique) unique-string
# arrays and the block-membership tables. No temporary of length N is made.
#
# Same-spin rows never leave one group of the exact block decomposition: a
# β string's partner set is exactly the α set of the group that holds it, so
# an α excitation ``a -> a'`` has a row for every spectator of the group iff
# ``a'`` is in that group's α set too. Rows are therefore enumerated once per
# unique string and expanded over the group's spectator axis. Cross-spin rows
# need the ``(a', b')`` membership lookup, which
# :meth:`StructuralPairBasis.pair_index_of_uids` resolves from the
# block-membership table without a pair list.


def _structural_grouping(
    pair_basis: StructuralPairBasis,
    pair_helpers: PairHelpers,
) -> PairBlocks:
    """Exact block decomposition of a structural basis.

    Reuses the decomposition the helpers already carry for the block matvec
    when present; otherwise builds it (O(M_unique) memory, no pair list).

    Args:
        pair_basis: The structural basis.
        pair_helpers: Helpers bound to the same basis.

    Returns:
        The structural :class:`PairBlocks` decomposition.
    """
    blocks = getattr(pair_helpers, "blocks", None)
    if blocks is not None and getattr(blocks, "structural", False):
        return blocks
    built = build_pair_blocks(pair_basis, mode="on")
    assert built is not None and built.structural
    return built


def _structural_pair_chunks(
    pair_basis: StructuralPairBasis,
    chunk_size: int,
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Yield ``(pair_index, alpha_uid, beta_uid)`` chunks over a structural basis.

    Walks each Cartesian block in global pair order, so every pair appears
    exactly once and each chunk holds at most ``chunk_size`` pairs. The
    workspace is one chunk of block arithmetic — never a length-N array.

    Args:
        pair_basis: The structural basis.
        chunk_size: Maximum pairs per yielded chunk.
        device: Device of the yielded index tensors.

    Yields:
        ``(pair_index, alpha_uid, beta_uid)``, each ``(n,)`` int64 with
        ``n <= chunk_size``.
    """
    step = max(1, int(chunk_size))
    for block in pair_basis.blocks:
        n_beta = int(block.n_beta)
        for start in range(0, int(block.n_pairs), step):
            stop = min(start + step, int(block.n_pairs))
            local = torch.arange(start, stop, dtype=torch.int64, device=device)
            yield (
                int(block.pair_base) + local,
                int(block.alpha_start) + torch.div(local, n_beta, rounding_mode="floor"),
                int(block.beta_start) + local.remainder(n_beta),
            )


def _structural_occupancy(strings: Tensor, n_orb: int, dtype: torch.dtype) -> Tensor:
    """``(n, n_orb)`` 0/1 occupancy of each spin string, in ``dtype``.

    ``strings`` may be a 1-D single-word array or an ``(n, n_limbs)`` limb
    matrix; :func:`_occupancy_bits` sums the limbs.
    """
    return _occupancy_bits(strings, n_orb).to(dtype)


def _repeat_over_columns(strings: Tensor, rows: int, cols: int) -> Tensor:
    """Repeat a per-row string slice over each row's ``cols`` entries.

    ``strings`` is ``(rows,)`` or ``(rows, n_limbs)``; the result is
    ``(rows * cols,)`` or ``(rows * cols, n_limbs)`` in row-major order.
    """
    if strings.dim() == 1:
        return strings.unsqueeze(1).expand(rows, cols).reshape(-1)
    width = int(strings.shape[-1])
    return strings.unsqueeze(1).expand(rows, cols, width).reshape(-1, width)


def _repeat_over_rows(strings: Tensor, rows: int, cols: int) -> Tensor:
    """Repeat a per-column string slice over each column's ``rows`` entries.

    ``strings`` is ``(cols,)`` or ``(cols, n_limbs)``; the result is
    ``(rows * cols,)`` or ``(rows * cols, n_limbs)`` in row-major order.
    """
    if strings.dim() == 1:
        return strings.unsqueeze(0).expand(rows, cols).reshape(-1)
    width = int(strings.shape[-1])
    return strings.unsqueeze(0).expand(rows, cols, width).reshape(-1, width)


def _structural_scaled_amplitudes(
    eigvec: Tensor,
    index: Tensor,
    alpha: Tensor,
    beta: Tensor,
    n_orb: int,
    *,
    sign_scale: bool,
) -> Tensor:
    """``σ(alpha, beta) × C[index]`` in the spin-block convention.

    ``sign_scale=False`` returns ``C[index]`` unchanged, for a vector the
    caller already converted. The signs are evaluated in bounded chunks so
    the transient ``(rows, n_orb)`` occupancy workspace stays O(_SIGN_CHUNK)
    even for a large ``chunk_size``.
    """
    values = eigvec[index]
    if not sign_scale:
        return values
    signs = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    for start in range(0, int(values.numel()), _SIGN_CHUNK):
        stop = min(start + _SIGN_CHUNK, int(values.numel()))
        signs[start:stop] = _inter_to_block_signs_pair(
            alpha[start:stop], beta[start:stop], n_orb
        ).to(dtype=values.dtype)
    return signs * values


def _structural_same_spin_rows(
    pair_basis: StructuralPairBasis,
    blocks: PairBlocks,
    eigvec: Tensor,
    n_orb: int,
    n_particles: int,
    *,
    is_alpha: bool,
    order: int,
    chunk_size: int,
    sign_scale: bool,
) -> Iterator[dict[str, Tensor]]:
    """Stream same-spin excitation rows of a structural basis.

    Mirrors :func:`torch_sci_solver.torch_impl.pair_helpers._stream_same_spin_dedup`:
    the occupied and empty positions of each *unique* source string are
    enumerated once, a target string survives when it lies in the source
    pair's own block group, and the row is then expanded over that group's
    spectator axis. The row set equals the explicit pair list's; only the
    order differs, and every consumer scatter-adds.

    Args:
        pair_basis: The structural basis.
        blocks: Its exact decomposition (``by_beta`` for α, ``by_alpha`` for β).
        eigvec: ``(N,)`` amplitudes in structural pair order.
        n_orb: Number of spatial orbitals.
        n_particles: Electrons of the excited spin.
        is_alpha: True for α rows, False for β rows.
        order: 1 for singles, 2 for doubles.
        chunk_size: Soft cap on the rows of one yielded dict.
        sign_scale: Apply the interleaved-Wick → spin-block sign per row.

    Yields:
        Dicts with ``bra_alpha``, ``bra_beta`` (the bra pair's strings),
        ``c_bra``, ``c_ket`` (σ-scaled amplitudes), ``cr0``, ``an0`` and,
        for doubles, ``cr1``, ``an1``. Orbital labels follow the explicit
        tables: ``cr`` is occupied in the bra, ``an`` in the ket.
    """
    device = eigvec.device
    groups = blocks.by_beta if is_alpha else blocks.by_alpha
    unique = (pair_basis.unique_alphas if is_alpha else pair_basis.unique_betas).to(device)
    other_unique = (pair_basis.unique_betas if is_alpha else pair_basis.unique_alphas).to(device)
    m_unique = int(unique.shape[0])
    if m_unique == 0:
        return
    sorted_unique, sorted_to_uid = _sort_strings(unique)
    n_empty = int(n_orb) - int(n_particles)
    if int(n_particles) < order or n_empty < order:
        return
    if order == 1:
        n_per = int(n_particles) * n_empty
    else:
        n_per = (int(n_particles) * (int(n_particles) - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    if n_per == 0:
        return
    spin_offset = 0 if is_alpha else 1
    one = torch.tensor(1, dtype=torch.int64, device=device)
    string_mask = _complement_mask(unique, n_orb)
    if order == 2:
        an_i, an_j = _triu_indices(int(n_particles), device=device)
        cr_i, cr_j = _triu_indices(n_empty, device=device)
    source_chunk = max(1, int(chunk_size) // n_per)
    row_step = max(1, int(chunk_size))

    for group in groups.groups:
        group_strings = (group.alpha_uids if is_alpha else group.beta_uids).to(device)
        spectator_uids = (group.beta_uids if is_alpha else group.alpha_uids).to(device)
        n_rows = int(group_strings.numel())
        n_cols = int(spectator_uids.numel())
        if n_rows == 0 or n_cols == 0:
            continue
        row_base = None if group.row_base is None else group.row_base.to(device)
        col_base = None if group.col_base is None else group.col_base.to(device)
        col_stride = None if group.col_stride is None else group.col_stride.to(device)

        for src_start in range(0, n_rows, source_chunk):
            src_stop = min(src_start + source_chunk, n_rows)
            n_src = src_stop - src_start
            src = unique[group_strings[src_start:src_stop]]
            occ_pos = _dispatch_bit_positions(src, n_orb, int(n_particles))
            empty_pos = _dispatch_bit_positions(src ^ string_mask, n_orb, n_empty)

            if order == 1:
                an0 = occ_pos.unsqueeze(-1).expand(-1, -1, n_empty).reshape(-1)
                cr0 = empty_pos.unsqueeze(-2).expand(-1, int(n_particles), -1).reshape(-1)
            else:
                n_op = int(an_i.shape[0])
                n_ep = int(cr_i.shape[0])
                an0 = occ_pos[:, an_i].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
                an1 = occ_pos[:, an_j].unsqueeze(-1).expand(-1, -1, n_ep).reshape(-1)
                cr0 = empty_pos[:, cr_i].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)
                cr1 = empty_pos[:, cr_j].unsqueeze(-2).expand(-1, n_op, -1).reshape(-1)

            src_local = (
                torch
                .arange(n_src, dtype=torch.int64, device=device)
                .unsqueeze(-1)
                .expand(-1, n_per)
                .reshape(-1)
            )
            target = src[src_local]
            if src.dim() == 1:
                target = target ^ (one << an0) ^ (one << cr0)
                if order == 2:
                    target = target ^ (one << an1) ^ (one << cr1)
            else:
                n_limbs = int(src.shape[-1])
                target = target ^ _bit_mask_rows(an0, n_limbs) ^ _bit_mask_rows(cr0, n_limbs)
                if order == 2:
                    target = target ^ _bit_mask_rows(an1, n_limbs) ^ _bit_mask_rows(cr1, n_limbs)

            found, position = _dispatch_found_in(sorted_unique, target)
            tgt_uid = sorted_to_uid[position.clamp(max=m_unique - 1)]
            in_group_pos = torch.searchsorted(group_strings, tgt_uid)
            in_group = (in_group_pos < n_rows) & (
                group_strings[in_group_pos.clamp(max=n_rows - 1)] == tgt_uid
            )
            keep = found & in_group
            if not bool(keep.any()):
                continue
            src_pos = src_start + src_local[keep]
            tgt_pos = in_group_pos[keep]
            an0 = an0[keep]
            cr0 = cr0[keep]
            if order == 2:
                an1 = an1[keep]
                cr1 = cr1[keep]

            for e0 in range(0, int(src_pos.numel()), row_step):
                e1 = min(e0 + row_step, int(src_pos.numel()))
                rows = e1 - e0
                width = max(1, row_step // rows)
                for c0 in range(0, n_cols, width):
                    c1 = min(c0 + width, n_cols)
                    cols = c1 - c0
                    src_sel = src_pos[e0:e1]
                    tgt_sel = tgt_pos[e0:e1]
                    spect = spectator_uids[c0:c1]
                    if is_alpha:
                        assert row_base is not None
                        k_bra = row_base[src_sel].unsqueeze(1) + spect.unsqueeze(0)
                        k_ket = row_base[tgt_sel].unsqueeze(1) + spect.unsqueeze(0)
                        ba = _repeat_over_columns(unique[group_strings[src_sel]], rows, cols)
                        ka = _repeat_over_columns(unique[group_strings[tgt_sel]], rows, cols)
                        bb = kbet = _repeat_over_rows(other_unique[spect], rows, cols)
                    else:
                        assert col_base is not None and col_stride is not None
                        k_bra = col_base[src_sel].unsqueeze(1) + spect.unsqueeze(0) * col_stride[
                            src_sel
                        ].unsqueeze(1)
                        k_ket = col_base[tgt_sel].unsqueeze(1) + spect.unsqueeze(0) * col_stride[
                            tgt_sel
                        ].unsqueeze(1)
                        bb = _repeat_over_columns(unique[group_strings[src_sel]], rows, cols)
                        kbet = _repeat_over_columns(unique[group_strings[tgt_sel]], rows, cols)
                        ba = ka = _repeat_over_rows(other_unique[spect], rows, cols)

                    kb = k_bra.reshape(-1)
                    kk = k_ket.reshape(-1)
                    out = {
                        "bra_alpha": ba,
                        "bra_beta": bb,
                        "c_bra": _structural_scaled_amplitudes(
                            eigvec, kb, ba, bb, n_orb, sign_scale=sign_scale
                        ),
                        "c_ket": _structural_scaled_amplitudes(
                            eigvec, kk, ka, kbet, n_orb, sign_scale=sign_scale
                        ),
                        "cr0": (2 * an0[e0:e1] + spin_offset)
                        .unsqueeze(1)
                        .expand(rows, cols)
                        .reshape(-1),
                        "an0": (2 * cr0[e0:e1] + spin_offset)
                        .unsqueeze(1)
                        .expand(rows, cols)
                        .reshape(-1),
                    }
                    if order == 2:
                        out["cr1"] = (
                            (2 * an1[e0:e1] + spin_offset)
                            .unsqueeze(1)
                            .expand(rows, cols)
                            .reshape(-1)
                        )
                        out["an1"] = (
                            (2 * cr1[e0:e1] + spin_offset)
                            .unsqueeze(1)
                            .expand(rows, cols)
                            .reshape(-1)
                        )
                    yield out


def _structural_ab_rows(
    pair_basis: StructuralPairBasis,
    eigvec: Tensor,
    n_orb: int,
    n_a: int,
    n_b: int,
    *,
    chunk_size: int,
    sign_scale: bool,
) -> Iterator[dict[str, Tensor]]:
    """Stream αβ double-excitation rows of a structural basis.

    Same per-unique-string enumeration as
    :func:`torch_sci_solver.torch_impl.pair_helpers._stream_ab_doubles`; the
    target-pair membership test uses
    :meth:`StructuralPairBasis.pair_index_of_uids`, which resolves a pair
    from the block-membership table without a length-N pair list.

    Args:
        pair_basis: The structural basis.
        eigvec: ``(N,)`` amplitudes in structural pair order.
        n_orb: Number of spatial orbitals.
        n_a, n_b: α and β electron counts.
        chunk_size: Soft cap on the rows of one yielded dict.
        sign_scale: Apply the interleaved-Wick → spin-block sign per row.

    Yields:
        Dicts with ``bra_alpha``, ``bra_beta``, ``c_bra``, ``c_ket`` and the
        four spin-orbital labels ``cr_a``, ``an_a``, ``cr_b``, ``an_b``, in
        the explicit ``ab_doubles`` convention.
    """
    device = eigvec.device
    L = int(n_orb)
    unique_a = pair_basis.unique_alphas.to(device)
    unique_b = pair_basis.unique_betas.to(device)
    sorted_a, perm_a = _sort_strings(unique_a)
    sorted_b, perm_b = _sort_strings(unique_b)
    m_a = int(unique_a.shape[0])
    m_b = int(unique_b.shape[0])
    n_a_empty = L - int(n_a)
    n_b_empty = L - int(n_b)
    n_per_a = int(n_a) * n_a_empty
    n_per_b = int(n_b) * n_b_empty
    if m_a == 0 or m_b == 0 or n_per_a == 0 or n_per_b == 0:
        return
    one = torch.tensor(1, dtype=torch.int64, device=device)
    mask_a = _complement_mask(unique_a, L)
    mask_b = _complement_mask(unique_b, L)
    pair_chunk = max(1, int(chunk_size) // max(n_per_a + n_per_b, 1))
    cross_cap = max(1, int(chunk_size))

    def _string_survivors(
        uids: Tensor,
        n_elec: int,
        n_empty: int,
        n_per: int,
        unique: Tensor,
        mask: Tensor,
        sorted_unique: Tensor,
        perm: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-unique-string single excitations that land in ``unique``."""
        strings = unique[uids]
        rows = int(strings.shape[0])
        occ = _dispatch_bit_positions(strings, L, n_elec)
        empty = _dispatch_bit_positions(strings ^ mask, L, n_empty)
        an = occ.unsqueeze(-1).expand(-1, -1, n_empty).reshape(rows, n_per)
        cr = empty.unsqueeze(-2).expand(-1, n_elec, -1).reshape(rows, n_per)
        if strings.dim() == 1:
            target = (strings.unsqueeze(-1) ^ (one << an) ^ (one << cr)).reshape(-1)
        else:
            n_limbs = int(strings.shape[-1])
            target = (
                strings.unsqueeze(-2) ^ _bit_mask_rows(an, n_limbs) ^ _bit_mask_rows(cr, n_limbs)
            ).reshape(-1, n_limbs)
        found, position = _dispatch_found_in(sorted_unique, target)
        found = found.reshape(rows, n_per)
        uid = perm[position.clamp(max=int(sorted_unique.shape[0]) - 1)]
        sel = found.reshape(-1).nonzero(as_tuple=True)[0]
        return found.sum(1), uid.reshape(-1)[sel], an.reshape(-1)[sel], cr.reshape(-1)[sel]

    def _expand(
        counts: Tensor,
        uids: Tensor,
        an: Tensor,
        cr: Tensor,
        inverse: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Expand per-unique-string survivors back over the chunk's pairs."""
        cnt = counts[inverse]
        total = int(cnt.sum().item())
        if total == 0:
            return cnt, uids[:0], an[:0], cr[:0]
        u_off = torch.cumsum(counts, 0) - counts
        pair_off = u_off[inverse]
        flat = torch.arange(total, dtype=torch.int64, device=device)
        pair_cum = torch.cumsum(cnt, 0)
        row = torch.searchsorted(pair_cum, flat, right=True)
        src = pair_off[row] + (flat - (pair_cum[row] - cnt[row]))
        return cnt, uids[src], an[src], cr[src]

    for pair_index, a_uid, b_uid in _structural_pair_chunks(pair_basis, pair_chunk, device):
        u_a, inv_a = torch.unique(a_uid, return_inverse=True)
        u_b, inv_b = torch.unique(b_uid, return_inverse=True)
        a_surv = _string_survivors(
            u_a, int(n_a), n_a_empty, n_per_a, unique_a, mask_a, sorted_a, perm_a
        )
        b_surv = _string_survivors(
            u_b, int(n_b), n_b_empty, n_per_b, unique_b, mask_b, sorted_b, perm_b
        )
        a_cnt, a_uid_s, a_an_s, a_cr_s = _expand(*a_surv, inv_a)
        b_cnt, b_uid_s, b_an_s, b_cr_s = _expand(*b_surv, inv_b)

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
            row = torch.searchsorted(cross_cum, flat, right=True)
            pos = flat - (cross_cum[row] - n_cross[row])
            b_here = b_cnt[row]
            a_local = torch.div(pos, b_here, rounding_mode="floor")
            b_local = pos - a_local * b_here
            a_at = a_off[row] + a_local
            b_at = b_off[row] + b_local

            found, k_ket = pair_basis.pair_index_of_uids(a_uid_s[a_at], b_uid_s[b_at])
            found = found.to(device)
            k_ket = k_ket.to(device)
            if not bool(found.any()):
                continue
            a_at = a_at[found]
            b_at = b_at[found]
            row = row[found]
            k_ket = k_ket[found]
            kb = pair_index[row]
            ba = unique_a[a_uid[row]]
            bb = unique_b[b_uid[row]]
            ka = unique_a[a_uid_s[a_at]]
            kbet = unique_b[b_uid_s[b_at]]
            yield {
                "bra_alpha": ba,
                "bra_beta": bb,
                "c_bra": _structural_scaled_amplitudes(
                    eigvec, kb, ba, bb, L, sign_scale=sign_scale
                ),
                "c_ket": _structural_scaled_amplitudes(
                    eigvec, k_ket, ka, kbet, L, sign_scale=sign_scale
                ),
                "cr_a": 2 * a_an_s[a_at],
                "an_a": 2 * a_cr_s[a_at],
                "cr_b": 2 * b_an_s[b_at] + 1,
                "an_b": 2 * b_cr_s[b_at] + 1,
            }


def _structural_same_spin_singles(
    gamma: Tensor,
    rows: dict[str, Tensor],
    n_orb: int,
    *,
    is_alpha: bool,
) -> None:
    """``γσ[cr, an] += sgn_σ × C*[bra] C[ket]`` per same-spin single row."""
    cr = rows["cr0"]
    an = rows["an0"]
    if is_alpha:
        cr_sp = cr // 2
        an_sp = an // 2
        bra = rows["bra_alpha"]
    else:
        cr_sp = (cr - 1) // 2
        an_sp = (an - 1) // 2
        bra = rows["bra_beta"]
    sgn = torch.ones(cr.shape, dtype=gamma.dtype, device=gamma.device)
    sgn = parity_batch(bra, None, torch.minimum(cr_sp, an_sp), torch.maximum(cr_sp, an_sp), sgn)
    contrib = sgn * rows["c_bra"].conj() * rows["c_ket"]
    gamma.view(-1).index_add_(0, cr_sp * n_orb + an_sp, contrib)


def _structural_same_spin_class_1(
    Gaa: Tensor,
    rows: dict[str, Tensor],
    n_orb: int,
    *,
    is_alpha: bool,
) -> None:
    """Single same-spin excitation × spectator orbital, four-slot scatter."""
    cr = rows["cr0"]
    an = rows["an0"]
    if is_alpha:
        cr_sp = cr // 2
        an_sp = an // 2
        bra = rows["bra_alpha"]
    else:
        cr_sp = (cr - 1) // 2
        an_sp = (an - 1) // 2
        bra = rows["bra_beta"]
    sgn = torch.ones(cr.shape, dtype=Gaa.dtype, device=Gaa.device)
    sgn = parity_batch(bra, None, torch.minimum(cr_sp, an_sp), torch.maximum(cr_sp, an_sp), sgn)
    weight = sgn * rows["c_bra"].conj() * rows["c_ket"]

    bra_occ = _structural_occupancy(bra, n_orb, Gaa.dtype)
    cr_mask = torch.zeros_like(bra_occ)
    cr_mask.scatter_(1, cr_sp.unsqueeze(1), 1.0)
    contribution = weight.unsqueeze(1) * (bra_occ * (1.0 - cr_mask))

    cr_idx = cr_sp.unsqueeze(1).expand(-1, n_orb)
    an_idx = an_sp.unsqueeze(1).expand(-1, n_orb)
    x_idx = torch.arange(n_orb, dtype=torch.int64, device=Gaa.device).unsqueeze(0)
    x_idx = x_idx.expand_as(cr_idx)
    flat = Gaa.view(-1)
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


def _structural_same_spin_class_2(
    Gaa: Tensor,
    rows: dict[str, Tensor],
    n_orb: int,
    *,
    is_alpha: bool,
) -> None:
    """Double same-spin excitation, four scatter slots with alternating signs."""
    cr0 = rows["cr0"]
    cr1 = rows["cr1"]
    an0 = rows["an0"]
    an1 = rows["an1"]
    if is_alpha:
        cr0_sp = cr0 // 2
        cr1_sp = cr1 // 2
        an0_sp = an0 // 2
        an1_sp = an1 // 2
        bra = rows["bra_alpha"]
    else:
        cr0_sp = (cr0 - 1) // 2
        cr1_sp = (cr1 - 1) // 2
        an0_sp = (an0 - 1) // 2
        an1_sp = (an1 - 1) // 2
        bra = rows["bra_beta"]
    first = torch.minimum(cr0_sp, cr1_sp)
    second = torch.maximum(cr0_sp, cr1_sp)
    low = torch.minimum(an0_sp, an1_sp)
    high = torch.maximum(an0_sp, an1_sp)
    sgn = torch.ones(cr0.shape, dtype=Gaa.dtype, device=Gaa.device)
    sgn = parity_batch(bra, None, torch.minimum(first, low), torch.maximum(first, low), sgn)
    sgn = parity_batch(bra, None, torch.minimum(second, high), torch.maximum(second, high), sgn)
    sgn = torch.where((low > second) | (high < first), -sgn, sgn)
    contribution = sgn * rows["c_bra"].conj() * rows["c_ket"]
    flat = Gaa.view(-1)
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


def _structural_ab_class_b(Gab: Tensor, rows: dict[str, Tensor], n_orb: int) -> None:
    """``Γαβ[p, p, cr_β, an_β] += sgn_β × occ_α(bra, p) × C*[bra] C[ket]``."""
    cr = rows["cr0"]
    an = rows["an0"]
    cr_sp = (cr - 1) // 2
    an_sp = (an - 1) // 2
    sgn = torch.ones(cr.shape, dtype=Gab.dtype, device=Gab.device)
    sgn = parity_batch(
        rows["bra_beta"], None, torch.minimum(cr_sp, an_sp), torch.maximum(cr_sp, an_sp), sgn
    )
    weight = sgn * rows["c_bra"].conj() * rows["c_ket"]
    contribution = weight.unsqueeze(1) * _structural_occupancy(rows["bra_alpha"], n_orb, Gab.dtype)
    p_idx = torch.arange(n_orb, dtype=torch.int64, device=Gab.device).unsqueeze(0)
    p_idx = p_idx.expand_as(contribution)
    cr_idx = cr_sp.unsqueeze(1).expand_as(contribution)
    an_idx = an_sp.unsqueeze(1).expand_as(contribution)
    Gab.view(-1).index_add_(
        0,
        (p_idx * n_orb**3 + p_idx * n_orb**2 + cr_idx * n_orb + an_idx).reshape(-1),
        contribution.reshape(-1),
    )


def _structural_ab_class_c(Gab: Tensor, rows: dict[str, Tensor], n_orb: int) -> None:
    """``Γαβ[cr_α, an_α, r, r] += sgn_α × occ_β(bra, r) × C*[bra] C[ket]``."""
    cr = rows["cr0"]
    an = rows["an0"]
    cr_sp = cr // 2
    an_sp = an // 2
    sgn = torch.ones(cr.shape, dtype=Gab.dtype, device=Gab.device)
    sgn = parity_batch(
        rows["bra_alpha"], None, torch.minimum(cr_sp, an_sp), torch.maximum(cr_sp, an_sp), sgn
    )
    weight = sgn * rows["c_bra"].conj() * rows["c_ket"]
    contribution = weight.unsqueeze(1) * _structural_occupancy(rows["bra_beta"], n_orb, Gab.dtype)
    r_idx = torch.arange(n_orb, dtype=torch.int64, device=Gab.device).unsqueeze(0)
    r_idx = r_idx.expand_as(contribution)
    cr_idx = cr_sp.unsqueeze(1).expand_as(contribution)
    an_idx = an_sp.unsqueeze(1).expand_as(contribution)
    Gab.view(-1).index_add_(
        0,
        (cr_idx * n_orb**3 + an_idx * n_orb**2 + r_idx * n_orb + r_idx).reshape(-1),
        contribution.reshape(-1),
    )


def _structural_ab_class_d(Gab: Tensor, rows: dict[str, Tensor], n_orb: int) -> None:
    """``Γαβ[cr_α, an_α, cr_β, an_β] += sgn_α sgn_β × C*[bra] C[ket]``."""
    cr_a = rows["cr_a"]
    an_a = rows["an_a"]
    cr_b = rows["cr_b"]
    an_b = rows["an_b"]
    cr_a_sp = cr_a // 2
    an_a_sp = an_a // 2
    cr_b_sp = (cr_b - 1) // 2
    an_b_sp = (an_b - 1) // 2
    sgn_a = torch.ones(cr_a.shape, dtype=Gab.dtype, device=Gab.device)
    sgn_a = parity_batch(
        rows["bra_alpha"],
        None,
        torch.minimum(cr_a_sp, an_a_sp),
        torch.maximum(cr_a_sp, an_a_sp),
        sgn_a,
    )
    sgn_b = torch.ones(cr_a.shape, dtype=Gab.dtype, device=Gab.device)
    sgn_b = parity_batch(
        rows["bra_beta"],
        None,
        torch.minimum(cr_b_sp, an_b_sp),
        torch.maximum(cr_b_sp, an_b_sp),
        sgn_b,
    )
    contribution = sgn_a * sgn_b * rows["c_bra"].conj() * rows["c_ket"]
    Gab.view(-1).index_add_(
        0,
        cr_a_sp * n_orb**3 + an_a_sp * n_orb**2 + cr_b_sp * n_orb + an_b_sp,
        contribution,
    )


def _structural_same_spin_class_0(
    Gaa: Tensor,
    eigvec: Tensor,
    pair_basis: StructuralPairBasis,
    n_orb: int,
    chunk_size: int,
    *,
    is_alpha: bool,
) -> None:
    """Same-spin ``Γσσ`` class-0 ``|C|²`` diagonal term over all pairs."""
    device = Gaa.device
    dtype = Gaa.dtype
    unique = (pair_basis.unique_alphas if is_alpha else pair_basis.unique_betas).to(device)
    diagonal = torch.zeros((n_orb, n_orb), dtype=dtype, device=device)
    for k, a_uid, b_uid in _structural_pair_chunks(pair_basis, chunk_size, device):
        occ = _structural_occupancy(unique[a_uid if is_alpha else b_uid], n_orb, dtype)
        weights = (eigvec[k].conj() * eigvec[k]).to(dtype)
        diagonal = diagonal + (occ.T * weights) @ occ
    off = ~torch.eye(n_orb, dtype=torch.bool, device=device)
    p_idx, r_idx = torch.where(off)
    flat = Gaa.view(-1)
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


def _structural_ab_class_a(
    Gab: Tensor,
    eigvec: Tensor,
    pair_basis: StructuralPairBasis,
    n_orb: int,
    chunk_size: int,
) -> None:
    """``Γαβ`` class-A ``|C|²`` diagonal term over all pairs."""
    device = Gab.device
    dtype = Gab.dtype
    unique_a = pair_basis.unique_alphas.to(device)
    unique_b = pair_basis.unique_betas.to(device)
    diagonal = torch.zeros((n_orb, n_orb), dtype=dtype, device=device)
    for k, a_uid, b_uid in _structural_pair_chunks(pair_basis, chunk_size, device):
        weights = (eigvec[k].conj() * eigvec[k]).to(dtype)
        alpha_occ = _structural_occupancy(unique_a[a_uid], n_orb, dtype)
        beta_occ = _structural_occupancy(unique_b[b_uid], n_orb, dtype)
        diagonal = diagonal + (alpha_occ * weights.unsqueeze(-1)).T @ beta_occ
    p_range = torch.arange(n_orb, dtype=torch.int64, device=device)
    p_idx, r_idx = torch.meshgrid(p_range, p_range, indexing="ij")
    flat_index = (
        p_idx.reshape(-1) * n_orb**3
        + p_idx.reshape(-1) * n_orb**2
        + r_idx.reshape(-1) * n_orb
        + r_idx.reshape(-1)
    )
    Gab.view(-1).index_add_(0, flat_index, diagonal.reshape(-1))


def _make_one_rdm_structural(
    eigvec: Tensor,
    pair_basis: StructuralPairBasis,
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int,
    eigvec_in_interleaved_wick: bool,
) -> Tuple[Tensor, Tensor]:
    """Structural-basis ``(γα, γβ)``: block-walked diagonal plus singles rows.

    Allocates O(chunk_size + M_unique) workspace and the ``(L, L)`` outputs;
    never an array of length N.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = int(n_orb)
    blocks = _structural_grouping(pair_basis, pair_helpers)
    unique_a = pair_basis.unique_alphas.to(device)
    unique_b = pair_basis.unique_betas.to(device)
    occ_a = torch.zeros(L, dtype=dtype, device=device)
    occ_b = torch.zeros(L, dtype=dtype, device=device)
    for k, a_uid, b_uid in _structural_pair_chunks(pair_basis, chunk_size, device):
        weights = (eigvec[k].conj() * eigvec[k]).to(dtype)
        occ_a = occ_a + (
            weights.unsqueeze(-1) * _structural_occupancy(unique_a[a_uid], L, dtype)
        ).sum(dim=0)
        occ_b = occ_b + (
            weights.unsqueeze(-1) * _structural_occupancy(unique_b[b_uid], L, dtype)
        ).sum(dim=0)
    gamma_a = torch.diag(occ_a)
    gamma_b = torch.diag(occ_b)
    sign_scale = bool(eigvec_in_interleaved_wick)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        int(pair_helpers.n_a),
        is_alpha=True,
        order=1,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_same_spin_singles(gamma_a, rows, L, is_alpha=True)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        int(pair_helpers.n_b),
        is_alpha=False,
        order=1,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_same_spin_singles(gamma_b, rows, L, is_alpha=False)
    return gamma_a, gamma_b


def _make_two_rdm_same_spin_structural(
    eigvec: Tensor,
    pair_basis: StructuralPairBasis,
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int,
    eigvec_in_interleaved_wick: bool,
    is_alpha: bool,
) -> Tensor:
    """Structural-basis αα or ββ 2-RDM block (classes 0, 1 and 2).

    Allocates O(chunk_size + M_unique) workspace and the ``(L, L, L, L)``
    output; never an array of length N.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = int(n_orb)
    blocks = _structural_grouping(pair_basis, pair_helpers)
    G = torch.zeros((L, L, L, L), dtype=dtype, device=device)
    _structural_same_spin_class_0(G, eigvec, pair_basis, L, chunk_size, is_alpha=is_alpha)
    n_particles = int(pair_helpers.n_a if is_alpha else pair_helpers.n_b)
    sign_scale = bool(eigvec_in_interleaved_wick)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        n_particles,
        is_alpha=is_alpha,
        order=1,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_same_spin_class_1(G, rows, L, is_alpha=is_alpha)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        n_particles,
        is_alpha=is_alpha,
        order=2,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_same_spin_class_2(G, rows, L, is_alpha=is_alpha)
    return G


def _make_two_rdm_ab_structural(
    eigvec: Tensor,
    pair_basis: StructuralPairBasis,
    pair_helpers: PairHelpers,
    n_orb: int,
    *,
    chunk_size: int,
    eigvec_in_interleaved_wick: bool,
) -> Tensor:
    """Structural-basis αβ 2-RDM block (classes A, B, C and D).

    Allocates O(chunk_size + M_unique) workspace and the ``(L, L, L, L)``
    output; never an array of length N.
    """
    device = eigvec.device
    dtype = eigvec.dtype
    L = int(n_orb)
    blocks = _structural_grouping(pair_basis, pair_helpers)
    Gab = torch.zeros((L, L, L, L), dtype=dtype, device=device)
    _structural_ab_class_a(Gab, eigvec, pair_basis, L, chunk_size)
    sign_scale = bool(eigvec_in_interleaved_wick)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        int(pair_helpers.n_b),
        is_alpha=False,
        order=1,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_ab_class_b(Gab, rows, L)
    for rows in _structural_same_spin_rows(
        pair_basis,
        blocks,
        eigvec,
        L,
        int(pair_helpers.n_a),
        is_alpha=True,
        order=1,
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_ab_class_c(Gab, rows, L)
    for rows in _structural_ab_rows(
        pair_basis,
        eigvec,
        L,
        int(pair_helpers.n_a),
        int(pair_helpers.n_b),
        chunk_size=chunk_size,
        sign_scale=sign_scale,
    ):
        _structural_ab_class_d(Gab, rows, L)
    return Gab
