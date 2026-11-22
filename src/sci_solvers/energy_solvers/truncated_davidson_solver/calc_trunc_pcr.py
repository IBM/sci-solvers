"""
calc_trunc_pcr.py
==================

Implements CALC_TRUNC_PCR (Fig. 4 of Cotton, J. Chem. Phys. 157, 224105
(2022)) -- it computes the diagonally preconditioned residual

    r = [D_hat - E]^-1 [H_hat - E] psi

without ever forming the full sigma-vector, by generating only the
determinants connected to psi by a single- or double-excitation and
discarding (or never even fully evaluating) anything whose amplitude
falls below a threshold delta.

Loop structure (matching Fig. 4 exactly)
------------------------------------------
    r <- psi                          (the D_hat = H_hat "diagonal" part)
    for c_I, |d_I> in psi:
        # 1-electron moves -- kept only if |c_I'| > delta.
        for i occupied, a unoccupied in |d_I>:
            c_I' <- <d_I'|H|d_I> * c_I
            c_I' <- c_I' / (E_{d_I'} - E)
            if |c_I'| > delta:
                r <- r + c_I'|d_I'>

        # 2-electron moves -- the key optimization. For fixed i,j, the
        # matrix element <d_I'|H|d_I> for a two-electron move to a,b is
        # proportional to a single two-electron integral u_abij =
        # (ai|bj) sorted by descending magnitude, once a term's raw amplitude < delta (before preconditioning), we short-circuit.
        for i<j occupied in |d_I>:
            for (a,b) in sorted_by_|u_abij|_descending:    # precomputed
                if a or b not virtual in |d_I>: continue   # skip
                if |u_abij * c_I| < delta: break           # short-circuit!
                c_I' <- <d_I'|H|d_I> * c_I
                c_I' <- c_I' / (E_{d_I'} - E)
                if |c_I'| > delta: r <- r + c_I'|d_I'>
    return r

This is implemented separately for the three double-excitation
categories (alpha-alpha, beta-beta, cross alpha-beta).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Dict, Tuple

import numpy as np

from sci_solvers.energy_solvers.truncated_davidson_solver.subspace_hamiltonian import (
    BACKEND_CPP,
    _load_backend,
)


def _flatten_tables(tables: "SortedIntegralTables") -> None:
    """
    One-time (per Hamiltonian) conversion of the dict-of-tuple-keyed
    {u_abij} tables into dense 3D arrays, stored as tables._flat, so the
    fused numba kernel can index same_a[i, j, :] directly instead of doing
    a tuple-key dict lookup.

    Both same_spin and cross_spin have a fixed number of (a,b)
    entries per (i,j) -- norb*(norb-1) and norb*norb respectively --
    so this is a plain dense array.

    Layout of the three parallel arrays at [i, j]:
        a: orbital index that the electron from orbital i moves to
        b: orbital index that the electron from orbital j moves to
        v: integral value eri[a, i, b, j]
    All three arrays are sorted by descending |v|, enabling the short-circuit
    in CALC_TRUNC_PCR.
    """
    norb = tables.norb
    m_same = norb * (norb - 1)
    same_a = np.zeros((norb, norb, m_same), dtype=np.int64)
    same_b = np.zeros((norb, norb, m_same), dtype=np.int64)
    same_v = np.zeros((norb, norb, m_same), dtype=np.float64)
    for (i, j), (a_arr, b_arr, v_arr) in tables.same_spin.items():
        same_a[i, j] = a_arr
        same_b[i, j] = b_arr
        same_v[i, j] = v_arr

    m_cross = norb * norb
    cross_a = np.zeros((norb, norb, m_cross), dtype=np.int64)
    cross_b = np.zeros((norb, norb, m_cross), dtype=np.int64)
    cross_v = np.zeros((norb, norb, m_cross), dtype=np.float64)
    for (i, j), (a_arr, b_arr, v_arr) in tables.cross_spin.items():
        cross_a[i, j] = a_arr
        cross_b[i, j] = b_arr
        cross_v[i, j] = v_arr

    tables._flat = (same_a, same_b, same_v, cross_a, cross_b, cross_v)


class SortedIntegralTables:
    """
    Precomputes, once per Hamiltonian (independent of any particular
    determinant), the sorted-by-magnitude {u_abij} tables that make the
    two-electron-move short-circuit in CALC_TRUNC_PCR possible

        same_spin[(i, j)]  for i < j -> (a_arr, b_arr, val_arr)
        cross_spin[(i, j)] for all i, j -> (a_arr, b_arr, val_arr)

    where val_arr = eri[a, i, b, j] and all three arrays are sorted by
    descending |val_arr|. `a_arr`/`b_arr` range over *all* orbital
    indices (a != b) regardless of occupation.
    """

    def __init__(self, eri: np.ndarray, norb: int):
        """
        Parameters
        ----------
        eri : np.ndarray, shape (norb, norb, norb, norb)
            Two-electron repulsion integrals in the (ai|bj) convention.
        norb : int
            Number of spatial orbitals.
        """
        self.norb = norb
        self.same_spin: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.cross_spin: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._flat: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

        # Same-spin excitations move two same-spin electrons, so the two
        # destination orbitals a != b (Pauli exclusion within one spin).
        same_spin_ab = [(a, b) for a in range(norb) for b in range(norb) if a != b]
        for i in range(norb):
            for j in range(i + 1, norb):
                vals = np.array([eri[a, i, b, j] for a, b in same_spin_ab])
                order = np.argsort(-np.abs(vals))
                a_arr = np.array([same_spin_ab[k][0] for k in order], dtype=np.int64)
                b_arr = np.array([same_spin_ab[k][1] for k in order], dtype=np.int64)
                self.same_spin[(i, j)] = (a_arr, b_arr, vals[order])

        # Cross-spin excitations move one alpha and one beta electron --
        # these occupy independent spin-orbital spaces, so a == b (same
        # spatial orbital, opposite spin) is a perfectly valid and often
        # important configuration and must NOT be excluded here.
        cross_spin_ab = [(a, b) for a in range(norb) for b in range(norb)]
        for i in range(norb):
            for j in range(norb):
                vals = np.array([eri[a, i, b, j] for a, b in cross_spin_ab])
                order = np.argsort(-np.abs(vals))
                a_arr = np.array([cross_spin_ab[k][0] for k in order], dtype=np.int64)
                b_arr = np.array([cross_spin_ab[k][1] for k in order], dtype=np.int64)
                self.cross_spin[(i, j)] = (a_arr, b_arr, vals[order])

        _flatten_tables(self)


def calc_trunc_pcr(builder, tables, psi, E, delta):
    """Compute the truncated PCR r = [D_hat - E]^-1 [H_hat - E] psi.

    Parameters
    ----------
    builder : SubspaceHamiltonianBuilder
    tables : SortedIntegralTables
    psi : (keys: int64[:], vals: float64[:])
        Flat-map sparse vector (growable or fixed).
    E : float
    delta : float

    Returns
    -------
    r : (keys: int64[:], vals: float64[:])
        Flat-map sparse residual vector.
    """
    psi_keys, psi_vals = psi
    same_a, same_b, same_v, cross_a, cross_b, cross_v = tables._flat

    use_dense_hdiag, hdiag, hdiag1, jdiag, kdiag = builder.H.diag_inputs()

    _nk = _load_backend(builder.backend)
    kwargs = dict(
        psi_keys=psi_keys,
        psi_vals=psi_vals,
        norb=builder.norb,
        nb=np.int64(builder.H.nb),
        nelec_a=builder.nelec_a,
        nelec_b=builder.nelec_b,
        bits_a=builder._bits_a,
        bits_b=builder._bits_b,
        addr_table_a=builder._addr_table_a,
        addr_table_b=builder._addr_table_b,
        h1e=builder.h1e,
        eri=builder.eri,
        use_dense_hdiag=use_dense_hdiag,
        hdiag=hdiag,
        hdiag1=hdiag1,
        jdiag=jdiag,
        kdiag=kdiag,
        same_a=same_a,
        same_b=same_b,
        same_v=same_v,
        cross_a=cross_a,
        cross_b=cross_b,
        cross_v=cross_v,
        E=np.float64(E),
        delta=np.float64(delta),
    )
    if builder.backend == BACKEND_CPP:
        kwargs["use_dense_addr"] = builder._use_direct_table
        kwargs["registry"] = builder.registry
    return _nk.calc_trunc_pcr_fused(**kwargs)
