"""⟨S²⟩ of an ``SCIState`` with an O(N)-memory matrix-vector product.

``pyscf.fci.selected_ci.spin_square`` allocates a second ``(N_a, N_b)`` array internally,
which runs out of memory on the largest benchmark subspaces; this applies S² directly.

Notes
-----
For a fixed :math:`(n_\\alpha, n_\\beta)` sector,

.. math:: S^2 = S_z^2 - S_z + n_\\alpha - \\sum_{p,q} a^\\dagger_{p\\alpha} a_{q\\alpha}
          a^\\dagger_{q\\beta} a_{p\\beta},

the identity :math:`S^2 = S_+ S_- + S_z^2 - S_z` with :math:`S_+ S_-` written in second
quantization. For a determinant :math:`|I\\rangle` with alpha string :math:`I_a` and beta
string :math:`I_b`, the diagonal is :math:`n_\\alpha - \\mathrm{popcount}(I_a \\wedge I_b)
+ S_z^2 - S_z`. The off-diagonal spin-swap term (:math:`p \\ne q`) moves an alpha electron
:math:`q \\to p` and a beta electron :math:`p \\to q`; its Jordan-Wigner sign is the
product of the alpha and beta single-excitation signs times :math:`-1`.

Then :math:`\\langle S^2\\rangle = c^\\top (S^2 c)` and the multiplicity is
:math:`2S + 1 = \\sqrt{1 + 4\\langle S^2\\rangle}`.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import math
from typing import Any

import numpy as np


def _popcount(x: np.ndarray) -> np.ndarray:
    """Bit count of each int64, by the SWAR reduction."""
    x = x.astype(np.int64, copy=False).view(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def _below(det: int, orb: int) -> int:
    """Number of occupied orbitals strictly below ``orb``."""
    return bin(det & ((1 << orb) - 1)).count("1")


def _swap_tables(dets: np.ndarray, norb: int, side: str) -> dict:
    """Spin-swap connections of one spin's strings, keyed by the orbital pair ``(p, q)``.

    Alpha: ``q`` occupied and ``p`` empty, target ``I ^ (1<<q) ^ (1<<p)``. Beta: ``p``
    occupied and ``q`` empty. Each value is ``(source index, target index, sign)``
    arrays; only targets present in ``dets`` are kept.
    """
    index = {int(d): i for i, d in enumerate(dets.astype(np.int64).tolist())}
    entries: dict[tuple[int, int], tuple[list, list, list]] = {}
    for i, det in enumerate(dets.astype(np.int64).tolist()):
        occ = [k for k in range(norb) if (det >> k) & 1]
        emp = [k for k in range(norb) if not (det >> k) & 1]
        q_orbs, p_orbs = (occ, emp) if side == "alpha" else (emp, occ)
        for q in q_orbs:
            for p in p_orbs:
                j = index.get(det ^ (1 << q) ^ (1 << p))
                if j is None:
                    continue
                first, second = (q, p) if side == "alpha" else (p, q)
                sign = (-1) ** _below(det, first) * (-1) ** _below(det ^ (1 << first), second)
                src, dst, sgn = entries.setdefault((p, q), ([], [], []))
                src.append(i)
                dst.append(j)
                sgn.append(sign)
    return {
        key: (np.array(s, dtype=np.int64), np.array(d, dtype=np.int64), np.array(g, float))
        for key, (s, d, g) in entries.items()
    }


def _s2_matvec(
    c: np.ndarray, strs_a: np.ndarray, strs_b: np.ndarray, n_a: int, n_b: int, norb: int
) -> np.ndarray:
    """Return :math:`S^2 c` for amplitudes ``c`` of shape ``(N_a, N_b)``."""
    sz = 0.5 * (n_a - n_b)
    doubly = _popcount(strs_a[:, None].astype(np.int64) & strs_b[None, :].astype(np.int64))
    out = ((n_a - doubly) + sz * sz - sz) * c
    a_tab = _swap_tables(strs_a, norb, "alpha")
    b_tab = _swap_tables(strs_b, norb, "beta")
    for pq in a_tab.keys() & b_tab.keys():
        a_src, a_dst, a_sign = a_tab[pq]
        b_src, b_dst, b_sign = b_tab[pq]
        for ia, ja, sa in zip(a_src.tolist(), a_dst.tolist(), a_sign.tolist()):
            np.add.at(out, (ja, b_dst), -sa * b_sign * c[ia, b_src])
    return out


def compute_spin_square(sci_state: Any) -> tuple[float, float]:
    """Return ``(⟨S²⟩, 2S+1)`` of a ``qiskit_addon_sqd.fermion.SCIState``."""
    c = np.asarray(sci_state.amplitudes)
    n_a, n_b = sci_state.nelec
    s2c = _s2_matvec(c, sci_state.ci_strs_a, sci_state.ci_strs_b, n_a, n_b, sci_state.norb)
    ss = float(np.real(np.vdot(c.ravel(), s2c.ravel())))
    return ss, math.sqrt(max(0.0, 1.0 + 4.0 * ss))
