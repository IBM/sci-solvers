"""Integral storage matching SBD's oneInt + twoInt access pattern.

Faithful port of sbd/include/sbd/chemistry/basic/integrals.h plus the
RHF construction code from makeintegrals.h.

We provide:
  - OneInt:   spin-orbital-indexed one-electron Hamiltonian h[p,q]
  - TwoInt:   chemist-notation two-electron integrals (pq|rs)
              spin-orbital indexed with spin selection rule built into .value()

For the *reference* implementation here we use dense numpy storage rather
than SBD's 8-fold-symmetric packed format. The packed format optimizes for
memory, but the API is what matters: value(p,q,r,s) must return the same
number as SBD would. We verify that by reading FCIDUMP records and
populating all 8 symmetric permutations.

Spin-orbital indexing: index ``i`` where ``i % 2 == 0`` is α-spin of
spatial orbital ``i // 2``; ``i % 2 == 1`` is β-spin.

For L=36 spatial orbitals (2L=72 spin-orbitals):
    OneInt:  72 × 72        ≈ 41 KB  (float64)
    TwoInt:  72⁴            ≈ 215 MB (float64)
    Direct:  72 × 72        ≈ 41 KB
    Exch:    72 × 72        ≈ 41 KB
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


from typing import Tuple

import numpy as np
import numpy.typing as npt


class OneInt:
    """One-electron integrals indexed by spin-orbital.

    SBD: ``oneInt<ElemT>`` in integrals.h:10-22.
    SBD layout: store[i*norbs + j] where norbs = 2L. We use a 2L×2L array.
    """

    __slots__ = ("n_spinorb", "_h")

    def __init__(self, n_spinorb: int, dtype: npt.DTypeLike = np.float64):
        """Initialize the instance."""
        self.n_spinorb = n_spinorb
        self._h: np.ndarray = np.zeros((n_spinorb, n_spinorb), dtype=dtype)

    def value(self, i: int, j: int) -> float:
        """Value."""
        return float(self._h[i, j])

    def __setitem__(self, key, val):
        """Set the item for the key."""
        self._h[key] = val


class TwoInt:
    """Two-electron integrals (pq|rs) in chemist notation, indexed by
    spin-orbital.

    SBD: ``twoInt<ElemT>`` in integrals.h:64-108 (RHF / closed-shell branch).

    The chemist's integral (pq|rs) factorizes as
        ∫∫ ψ_p*(1) ψ_q(1) (1/r12) ψ_r*(2) ψ_s(2)
    so for orbital integrals: only nonzero when spin(p)=spin(q) and
    spin(r)=spin(s). SBD enforces this in ``Value()`` by returning 0
    immediately if the spin-pair condition fails. We do the same.

    Storage: dense ``(2L)^4`` array. Inefficient (factor 8 redundancy
    vs. SBD's packed form) but trivial to index and dump to GPU later.
    """

    __slots__ = ("n_spinorb", "_eri", "_direct", "_exchange")

    def __init__(self, n_spinorb: int, dtype: npt.DTypeLike = np.float64):
        """Initialize the instance."""
        self.n_spinorb = n_spinorb
        self._eri: np.ndarray = np.zeros((n_spinorb,) * 4, dtype=dtype)
        self._direct: np.ndarray = np.zeros((n_spinorb, n_spinorb), dtype=dtype)
        self._exchange: np.ndarray = np.zeros((n_spinorb, n_spinorb), dtype=dtype)

    def value(self, p: int, q: int, r: int, s: int) -> float:
        """(pq|rs) chemist notation.

        SBD: ``twoInt::Value(i, j, k, l)`` integrals.h:91-99.
        Returns 0 if spin(p)!=spin(q) or spin(r)!=spin(s).
        """
        if (p % 2) != (q % 2):
            return 0.0
        if (r % 2) != (s % 2):
            return 0.0
        return float(self._eri[p, q, r, s])

    def direct_value(self, i: int, j: int) -> float:
        """(ii|jj). SBD: ``DirectValue(i,j)`` integrals.h:100-102."""
        return float(self._direct[i, j])

    def exchange_value(self, i: int, j: int) -> float:
        """(ij|ji). SBD: ``ExchangeValue(i,j)`` integrals.h:103-105."""
        return float(self._exchange[i, j])


# ============================================================================
# Building integrals from PySCF mole / FCIDUMP file
# ============================================================================


def setup_integrals_from_pyscf_h1_eri(
    h1_spatial: np.ndarray, eri_spatial: np.ndarray, core_energy: float
) -> Tuple[float, OneInt, TwoInt]:
    """Build (I0, I1, I2) from PySCF spatial-orbital integrals.

    Args:
        h1_spatial:  (L, L) one-electron h matrix in spatial-orbital basis.
        eri_spatial: 4-index tensor of (ij|kl) chemist integrals in
                     spatial-orbital basis. Accepted shapes:
                       - (L, L, L, L) full
                       - (L*(L+1)/2, L*(L+1)/2) 4-fold packed (PySCF's
                         ``ao2mo.restore(4, ...)`` output, or what
                         PySCF FCI usually consumes)
                       - flat 8-fold packed (length L*(L+1)/2 * (L*(L+1)/2 + 1) / 2)
                     For the reference we accept all and convert to full
                     (L,L,L,L) before populating spin-orbital tensor.
        core_energy: scalar nuclear+core contribution (the ``I0`` in SBD).

    Returns:
        (I0, OneInt, TwoInt) — same triple SBD's SetupIntegrals returns.
    """
    L = h1_spatial.shape[0]
    n_spinorb = 2 * L

    # Bring eri to full (L,L,L,L)
    eri_full = _to_full_eri(eri_spatial, L)

    # ---- OneInt ----
    I1 = OneInt(n_spinorb)
    for i in range(L):
        for j in range(L):
            val = float(h1_spatial[i, j])
            # RHF: same h(i,j) for both α and β spin
            I1._h[2 * i, 2 * j] = val
            I1._h[2 * i + 1, 2 * j + 1] = val

    # ---- TwoInt ----
    I2 = TwoInt(n_spinorb)
    # Populate (pq|rs) for spin pairs that are both-α or both-β
    # using the spatial integrals. PySCF eri_full[i,j,k,l] is (ij|kl) chemist.
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for m in range(L):
                    val = float(eri_full[i, j, k, m])
                    # (αα|αα), (αα|ββ), (ββ|αα), (ββ|ββ) — all the
                    # same-spin-within-bra and same-spin-within-ket combinations
                    I2._eri[2 * i, 2 * j, 2 * k, 2 * m] = val  # αα|αα
                    I2._eri[2 * i, 2 * j, 2 * k + 1, 2 * m + 1] = val  # αα|ββ
                    I2._eri[2 * i + 1, 2 * j + 1, 2 * k, 2 * m] = val  # ββ|αα
                    I2._eri[2 * i + 1, 2 * j + 1, 2 * k + 1, 2 * m + 1] = val  # ββ|ββ

    # Direct + Exchange caches (SBD: integrals.h direct/exchange tables built
    # in makeintegrals.h:160-172). Direct(i,j) = (ii|jj), Exchange(i,j) = (ij|ji).
    for p in range(n_spinorb):
        for q in range(n_spinorb):
            I2._direct[p, q] = I2.value(p, p, q, q)
            I2._exchange[p, q] = I2.value(p, q, q, p)

    return float(core_energy), I1, I2


def _to_full_eri(eri: np.ndarray, L: int) -> np.ndarray:
    """Coerce a PySCF-style ERI representation to full (L,L,L,L).

    Cases (PySCF ao2mo.restore conventions):
      - already (L,L,L,L)              → return as-is
      - 4-fold packed (N*(N+1)/2,)^2  → use ao2mo.restore(1, ...)
      - 8-fold packed flat            → use ao2mo.restore(1, ...)

    Complex ``eri`` (a complex Hermitian two-body tensor) is only accepted
    already in the full ``(L, L, L, L)`` form — PySCF's 4-/8-fold packed
    layouts assume the real 8-fold permutational symmetry a general complex
    Hermitian tensor does not have, so ``ao2mo.restore`` is not meaningful
    for complex input.
    """
    if eri.ndim == 4 and eri.shape == (L, L, L, L):
        return eri.astype(eri.dtype, copy=False)
    if np.iscomplexobj(eri):
        raise ValueError(
            f"complex eri must already be full (L, L, L, L) = ({L}, {L}, {L}, {L}); "
            f"got shape {eri.shape}. PySCF's packed 4-/8-fold formats assume real "
            "8-fold permutational symmetry, which a general complex Hermitian "
            "two-body tensor does not have."
        )
    from pyscf import ao2mo

    return ao2mo.restore(1, eri.astype(np.float64), L)
