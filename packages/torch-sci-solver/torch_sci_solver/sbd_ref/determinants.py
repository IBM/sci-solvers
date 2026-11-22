"""Scalar pure-Python reference for SBD's determinant + Slater-Condon kernel.

Faithful port of sbd/include/sbd/chemistry/basic/determinants.h, with the
following simplifications since we don't need MPI:

- Determinants are represented as a single Python int (a packed bitmask).
  SBD packs into ``std::vector<size_t>`` with a configurable ``bit_length``
  to keep the packing trick portable across compilers; Python ints are
  arbitrary-precision so we just use a single int for the entire bitmask.
  This means our bit indices match SBD's: bit ``i`` in the int is bit ``i``
  in SBD's bitstring (lowest bit = orbital 0).

- Spin-orbital convention matches SBD exactly:
    bit 2*k     ↔ spatial orbital k, α-spin
    bit 2*k + 1 ↔ spatial orbital k, β-spin

- Half-bitstrings (α-only or β-only) are also packed Python ints over L
  bits. ``det_from_alpha_beta`` interleaves them into the full 2L-bit
  spin-orbital bitmask in the same layout SBD uses.

This module performs ZERO numerical work — it's used to validate the
torch implementation later. Every function maps 1:1 to an SBD function;
SBD locations are noted in docstrings for cross-reference.
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


from typing import List, Tuple

# ============================================================================
# Basic bit accessors  (SBD: determinants.h lines 140-205)
# ============================================================================


def getocc(det: int, x: int) -> bool:
    """Return True iff bit `x` of `det` is set.

    SBD: ``getocc(det, bit_length, x)`` in determinants.h:169-174.
    We drop ``bit_length`` since Python ints aren't split into words.
    """
    return bool((det >> x) & 1)


def setocc(det: int, x: int, val: bool) -> int:
    """Return a new int with bit `x` of `det` set to `val`.

    SBD: ``setocc(det, bit_length, x, y)`` in determinants.h:149-166.
    SBD mutates the vector in place; Python ints are immutable so we
    return a new int.
    """
    if val:
        return det | (1 << x)
    return det & ~(1 << x)


def bitcount(det: int, n_bits: int) -> int:
    """Popcount of the low `n_bits` bits of `det`.

    SBD: ``bitcount(det, bit_length, L)`` in determinants.h:176-190.
    SBD masks off bits past position L; we do the same.
    """
    mask = (1 << n_bits) - 1
    return (det & mask).bit_count()


# ============================================================================
# Det-level operations  (SBD: determinants.h lines 12-90, 365-432)
# ============================================================================


def det_from_alpha_beta(a: int, b: int, L: int) -> int:
    """Interleave α and β half-bitstrings into a full 2L-bit spin-orbital
    bitmask.

    SBD: ``DetFromAlphaBeta(A, B, bit_length, L)`` in determinants.h:12-53.
    Convention: bit ``2k`` of result ← bit ``k`` of `a` (α-spin orbital k),
                bit ``2k+1`` of result ← bit ``k`` of `b` (β-spin orbital k).

    Args:
        a: α-string, bits 0..L-1 mark occupied α-spin orbitals
        b: β-string, bits 0..L-1 mark occupied β-spin orbitals
        L: number of spatial orbitals

    Returns:
        Full det bitmask with bits 0..2L-1 used.
    """
    out = 0
    for k in range(L):
        if (a >> k) & 1:
            out |= 1 << (2 * k)
        if (b >> k) & 1:
            out |= 1 << (2 * k + 1)
    return out


def difference(a: int, b: int, n_bits: int) -> int:
    """Number of bit positions where `a` and `b` differ (popcount of XOR).

    SBD: ``difference(a, b, bit_length, L)`` in determinants.h:365-386.
    For two determinants of the same particle number, this is even —
    2 for a single excitation, 4 for a double, etc.
    """
    mask = (1 << n_bits) - 1
    return ((a ^ b) & mask).bit_count()


def orbital_difference(a: int, b: int, n_bits: int) -> Tuple[List[int], List[int]]:
    """Return (x, y) where:
      x = bit positions set in `a` but not `b`
      y = bit positions set in `b` but not `a`

    SBD: ``OrbitalDifference(a, b, bit_length, L, x, y)`` in
    determinants.h:388-432. Used to identify which orbitals differ
    between two determinants.
    """
    mask = (1 << n_bits) - 1
    diff_a = (a & ~b) & mask  # in a only
    diff_b = (b & ~a) & mask  # in b only
    x = [i for i in range(n_bits) if (diff_a >> i) & 1]
    y = [i for i in range(n_bits) if (diff_b >> i) & 1]
    return x, y


# ============================================================================
# Parity (fermion anticommutation sign)  (SBD: determinants.h lines 230-281)
# ============================================================================


def parity(det: int, start: int, end: int, sgn: float) -> float:
    """Update `sgn` for moving a fermion from orbital `start` to orbital
    `end` (or vice versa) given that `det` is the bitmask of the *bra*
    determinant.

    SBD: ``parity(det, bit_length, start, end, sgn)`` in
    determinants.h:230-281. Returns the new sign rather than mutating
    an out-parameter.

    Algorithm:
        1. Count bits of `det` in the range [start, end).
        2. Multiply sgn by (-1)^count.
        3. If bit `start` of `det` is set, multiply sgn by an extra -1.

    Steps 2+3 together give the parity of bits strictly between `start`
    and `end` (i.e. in the open interval (start, end)), which is the
    correct fermion sign for moving an electron between those two
    positions through the occupied "Fermi sea".

    Requires start <= end. Caller must order arguments.
    """
    if start > end:
        raise ValueError("parity: start > end")

    # Bits at positions in [start, end). Build mask, popcount.
    mask = ((1 << end) - 1) ^ ((1 << start) - 1)
    nonzero = (det & mask).bit_count()

    sgn *= -1.0 if (nonzero & 1) else 1.0
    if (det >> start) & 1:
        sgn *= -1.0
    return sgn


# ============================================================================
# Slater-Condon evaluators  (SBD: determinants.h lines 434-562)
# ============================================================================
# These work with spin-orbital indices, where i, a, ... are in [0, 2L).
# The integrals (one_int, two_int) are objects with .value(...) methods
# that return numbers; see integrals.py.


def zero_excite(det: int, n_spinorb: int, I0, I1, I2) -> float:
    """⟨D|H|D⟩, the diagonal Hamiltonian matrix element for det `D`.

    SBD: ``ZeroExcite(det, bit_length, L, I0, I1, I2)`` in
    determinants.h:434-457.

    Note: SBD takes the spatial orbital count L and loops over 2L
    spin-orbitals. We take the spin-orbital count n_spinorb = 2L
    directly to avoid the off-by-2 trap.

    Returns:
        I0 + Σ_{i∈D} h_ii + Σ_{i<j ∈D} [(ii|jj) − δ_{spin(i)=spin(j)} (ij|ji)]
    """
    energy = float(I0)
    for i in range(n_spinorb):
        if not getocc(det, i):
            continue
        energy += I1.value(i, i)
        for j in range(i + 1, n_spinorb):
            if not getocc(det, j):
                continue
            energy += I2.direct_value(i, j)
            if (i % 2) == (j % 2):
                energy -= I2.exchange_value(i, j)
    return energy


def one_excite(det: int, i: int, a: int, I1, I2, n_spinorb: int) -> float:
    """Slater-Condon rule for two determinants differing in *one*
    spin-orbital.

    SBD: ``OneExcite(det, bit_length, i, a, I1, I2)`` in
    determinants.h:459-481.

    Important convention from SBD: ``det`` here is the *bra* (DetI in
    mult.h). The pair (i, a) is "the two orbitals that differ" — SBD
    names them i and a but the convention is *the caller* knows which
    one is bra-only vs ket-only. From mult.h, when called via the helper:
        - first arg (named "i") receives bra-only orbital (where electron is in bra)
        - second arg (named "a") receives ket-only orbital (where electron is in ket)

    Functionally, the formula is symmetric in the sense that swapping
    i and a only flips the parity sign (via parity()), and the integral
    indices are also flipped consistently. So as long as we use SBD's
    exact code, the answer is right.

    Returns:
        sgn × [ I1(a, i) + Σ_{j∈det} (I2(a,i,j,j) − I2(a,j,j,i)) ]
        with sgn = parity(det, min(i,a), max(i,a))
    """
    sgn = parity(det, min(i, a), max(i, a), 1.0)
    energy = I1.value(a, i)
    for j in range(n_spinorb):
        if getocc(det, j):
            energy += I2.value(a, i, j, j) - I2.value(a, j, j, i)
    return sgn * energy


def two_excite(det: int, i: int, j: int, a: int, b: int, I2) -> float:
    """Slater-Condon rule for two determinants differing in *two*
    spin-orbitals.

    SBD: ``TwoExcite(det, bit_length, i, j, a, b, I1, I2)`` in
    determinants.h:483-501.

    Sorts (i,j) → (I,J) and (a,b) → (A,B), then:
        sgn = parity(det, min(I,A), max(I,A)) * parity(det, min(J,B), max(J,B))
        if A > J or B < I: sgn *= -1
        return sgn * (I2(A,I,B,J) - I2(A,J,B,I))

    Doubles depend only on two-electron integrals.
    """
    i_lo = min(i, j)
    j_hi = max(i, j)
    a_lo = min(a, b)
    b_hi = max(a, b)
    sgn = parity(det, min(i_lo, a_lo), max(i_lo, a_lo), 1.0)
    sgn = parity(det, min(j_hi, b_hi), max(j_hi, b_hi), sgn)
    if a_lo > j_hi or b_hi < i_lo:
        sgn *= -1.0
    return sgn * (I2.value(a_lo, i_lo, b_hi, j_hi) - I2.value(a_lo, j_hi, b_hi, i_lo))


def hij(det_a: int, det_b: int, n_spinorb: int, I0, I1, I2) -> float:
    """Generic ⟨D_a|H|D_b⟩ for arbitrary spin-orbital bitmasks.

    SBD: ``Hij(DetA, DetB, bit_length, L, I0, I1, I2, orbDiff)`` in
    determinants.h:503-562.

    Dispatches on the number of differing spin-orbitals:
        0 → zero_excite(D_b, ...)
        1 → one_excite(D_b, d[0], c[0], I1, I2)
        2 → two_excite(D_b, d[0], d[1], c[0], c[1], I2)
        ≥3 → 0

    Where ``c = orbital_difference(D_a, D_b).x`` and
          ``d = orbital_difference(D_a, D_b).y``.

    Note SBD picks the *ket* (``DetB``) as the det passed to OneExcite /
    TwoExcite for the parity computation. We preserve that.
    """
    c, d = orbital_difference(det_a, det_b, n_spinorb)
    nc = len(c)
    if nc == 0:
        return zero_excite(det_b, n_spinorb, I0, I1, I2)
    if nc == 1:
        return one_excite(det_b, d[0], c[0], I1, I2, n_spinorb)
    if nc == 2:
        return two_excite(det_b, d[0], d[1], c[0], c[1], I2)
    return 0.0
