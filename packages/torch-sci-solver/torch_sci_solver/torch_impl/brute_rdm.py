"""Brute-force 2-RDM (and 1-RDM) for testing.

Builds γ and Γ by directly applying operators to the bitstring basis,
using **spin-block Wick** (α-orbital index 2k, β-orbital index 2k+1;
α and β operators effectively commute as a convention because their
parity is computed on disjoint sub-bitstrings).

PySCF convention:
    γα[p, q] = ⟨a†_pα a_qα⟩
    Γ_aa[p, q, r, s] = ⟨a†_pα a†_rα a_sα a_qα⟩
    Γ_ab[p, q, r, s] = ⟨a†_pα a†_rβ a_sβ a_qα⟩
    Γ_bb[p, q, r, s] = ⟨a†_pβ a†_rβ a_sβ a_qβ⟩

Slow (O((M_a M_b)² L⁴)); intended only for small-system validation
of the vectorized implementations.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import List, Tuple

import numpy as np


def _occupied_positions_sorted(det: int, n_bits: int) -> List[int]:
    """Occupied positions sorted."""
    return [i for i in range(n_bits) if (det >> i) & 1]


def _apply_a_alpha(det_alpha: int, q: int, n_orb: int) -> Tuple[int, int]:
    """Apply α-annihilation a_qα to |det_α⟩. Returns (sign, new_det_α) or (0, det).

    Uses α-Wick (parity counts α-bits below q in det_α only)."""
    if not ((det_alpha >> q) & 1):
        return 0, det_alpha
    pos = _occupied_positions_sorted(det_alpha, n_orb)
    idx = pos.index(q)
    sign = 1 if idx % 2 == 0 else -1
    return sign, det_alpha & ~(1 << q)


def _apply_adag_alpha(det_alpha: int, p: int, n_orb: int) -> Tuple[int, int]:
    """Apply α-creation a†_pα to |det_α⟩. Returns (sign, new_det_α) or (0, det)."""
    if (det_alpha >> p) & 1:
        return 0, det_alpha
    pos = _occupied_positions_sorted(det_alpha, n_orb)
    n_below = sum(1 for o in pos if o < p)
    sign = 1 if n_below % 2 == 0 else -1
    return sign, det_alpha | (1 << p)


def _matrix_element_alpha_one_body(
    det_a_bra: int, det_a_ket: int, p: int, q: int, n_orb: int
) -> int:
    """⟨det_a_bra| a†_pα a_qα |det_a_ket⟩ = sign (or 0)."""
    sgn_a, intermediate = _apply_a_alpha(det_a_ket, q, n_orb)
    if sgn_a == 0:
        return 0
    sgn_b, result = _apply_adag_alpha(intermediate, p, n_orb)
    if sgn_b == 0:
        return 0
    if result != det_a_bra:
        return 0
    return sgn_a * sgn_b


def _matrix_element_alpha_two_body(
    det_a_bra: int, det_a_ket: int, p: int, q: int, r: int, s: int, n_orb: int
) -> int:
    """⟨det_a_bra| a†_p a†_r a_s a_q |det_a_ket⟩ (all α-spin). Returns sign or 0."""
    sgn1, st1 = _apply_a_alpha(det_a_ket, q, n_orb)
    if sgn1 == 0:
        return 0
    sgn2, st2 = _apply_a_alpha(st1, s, n_orb)
    if sgn2 == 0:
        return 0
    sgn3, st3 = _apply_adag_alpha(st2, r, n_orb)
    if sgn3 == 0:
        return 0
    sgn4, st4 = _apply_adag_alpha(st3, p, n_orb)
    if sgn4 == 0:
        return 0
    if st4 != det_a_bra:
        return 0
    return sgn1 * sgn2 * sgn3 * sgn4


def brute_one_rdm(C, adets, bdets, n_orb):
    """Build (γα, γβ), each (L, L), in PySCF spin-block convention."""
    L = n_orb
    Ma, Mb = len(adets), len(bdets)
    ga = np.zeros((L, L))
    gb = np.zeros((L, L))

    # γα[p, q] = sum_{ia, ja, ib} C[ia,ib] C[ja,ib] ⟨adets[ia]| a†_p a_q |adets[ja]⟩
    for p in range(L):
        for q in range(L):
            val = 0.0
            for ia in range(Ma):
                for ja in range(Ma):
                    me = _matrix_element_alpha_one_body(adets[ia], adets[ja], p, q, L)
                    if me == 0:
                        continue
                    for ib in range(Mb):
                        val += C[ia * Mb + ib] * me * C[ja * Mb + ib]
            ga[p, q] = val
    # γβ analogously, with bdets and α/β indices summed in the other order
    for p in range(L):
        for q in range(L):
            val = 0.0
            for ib in range(Mb):
                for jb in range(Mb):
                    me = _matrix_element_alpha_one_body(bdets[ib], bdets[jb], p, q, L)
                    if me == 0:
                        continue
                    for ia in range(Ma):
                        val += C[ia * Mb + ib] * me * C[ia * Mb + jb]
            gb[p, q] = val
    return ga, gb


def brute_two_rdm_ab(C, adets, bdets, n_orb):
    """Γ_ab[p, q, r, s] = ⟨a†_pα a†_rβ a_sβ a_qα⟩

    In spin-block Wick (α and β commute): factorizes per state pair as
    ⟨α-side⟩ × ⟨β-side⟩.
    """
    L = n_orb
    Ma, Mb = len(adets), len(bdets)
    Gab = np.zeros((L, L, L, L))

    # For each (p, q, r, s): sum over (ia, ja, ib, jb)
    for p in range(L):
        for q in range(L):
            for ia in range(Ma):
                for ja in range(Ma):
                    me_a = _matrix_element_alpha_one_body(adets[ia], adets[ja], p, q, L)
                    if me_a == 0:
                        continue
                    for r in range(L):
                        for s in range(L):
                            for ib in range(Mb):
                                for jb in range(Mb):
                                    me_b = _matrix_element_alpha_one_body(
                                        bdets[ib], bdets[jb], r, s, L
                                    )
                                    if me_b == 0:
                                        continue
                                    Gab[p, q, r, s] += (
                                        C[ia * Mb + ib] * C[ja * Mb + jb] * me_a * me_b
                                    )
    return Gab


def brute_two_rdm_aa(C, adets, bdets, n_orb):
    """Γ_aa[p, q, r, s] = ⟨a†_pα a†_rα a_sα a_qα⟩ (β spectator)."""
    L = n_orb
    Ma, Mb = len(adets), len(bdets)
    Gaa = np.zeros((L, L, L, L))
    for p in range(L):
        for q in range(L):
            for r in range(L):
                for s in range(L):
                    val = 0.0
                    for ia in range(Ma):
                        for ja in range(Ma):
                            me = _matrix_element_alpha_two_body(adets[ia], adets[ja], p, q, r, s, L)
                            if me == 0:
                                continue
                            for ib in range(Mb):
                                val += C[ia * Mb + ib] * me * C[ja * Mb + ib]
                    Gaa[p, q, r, s] = val
    return Gaa


def brute_two_rdm_bb(C, adets, bdets, n_orb):
    """Γ_bb analog of Γ_aa with β bitmasks."""
    L = n_orb
    Ma, Mb = len(adets), len(bdets)
    Gbb = np.zeros((L, L, L, L))
    for p in range(L):
        for q in range(L):
            for r in range(L):
                for s in range(L):
                    val = 0.0
                    for ib in range(Mb):
                        for jb in range(Mb):
                            me = _matrix_element_alpha_two_body(bdets[ib], bdets[jb], p, q, r, s, L)
                            if me == 0:
                                continue
                            for ia in range(Ma):
                                val += C[ia * Mb + ib] * me * C[ia * Mb + jb]
                    Gbb[p, q, r, s] = val
    return Gbb
