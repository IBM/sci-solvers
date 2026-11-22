"""Hamiltonian-vector multiply (H@v) for the SBD TPB algorithm.

Faithful port of sbd/include/sbd/chemistry/tpb/mult.h::mult (SBD_TRADMODE
branch, lines 113-883) and qcham.h::makeQChamDiagTerms (lines 290-340),
simplified for single-task / no-MPI use.

Algorithm
---------
The wavefunction lives over the Cartesian product basis
``{(adets[ia], bdets[ib]) : ia in [0, M_a), ib in [0, M_b)}``.
Linear index: ``k = ia * M_b + ib``.

We compute Wb = H @ T in four passes:

  1. Diagonal:    Wb[k] += hii[k] * T[k]
                  where hii[k] = ⟨DetI(ia,ib)|H|DetI(ia,ib)⟩

  2. taskType=2 (α-only excitations, β fixed):
       for each bra (ia, ib):
         DetI = det_from_alpha_beta(adets[ia], bdets[ib])
         for each (ja, cr_so, an_so) in α-singles from ia:
             Wb[ia*M_b + ib] += one_excite(DetI, cr, an, ...) * T[ja*M_b + ib]
         for each (ja, cr0, cr1, an0, an1) in α-doubles from ia:
             Wb[ia*M_b + ib] += two_excite(DetI, cr0, cr1, an0, an1, ...) * T[ja*M_b + ib]

  3. taskType=1 (β-only excitations, α fixed): symmetric.

  4. taskType=0 (αβ-doubles, outer product):
       for each bra (ia, ib):
         DetI = det_from_alpha_beta(adets[ia], bdets[ib])
         for each (ja, cr_a, an_a) in α-singles from ia:
           for each (jb, cr_b, an_b) in β-singles from ib:
             Wb[ia*M_b + ib] += two_excite(DetI, cr_a, cr_b, an_a, an_b, ...)
                                  * T[ja*M_b + jb]

Note that DetI is the *bra* determinant, used inside OneExcite/TwoExcite
to determine which orbitals are occupied (for the Coulomb sum in
OneExcite, and for the parity in both).

This is the SCALAR PYTHON REFERENCE — slow by design (Python loops
over every excitation). Used to validate the eventual vectorized torch
port. Not intended for production.
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


from typing import List

import numpy as np

from torch_sci_solver.sbd_ref.determinants import (
    det_from_alpha_beta,
    one_excite,
    two_excite,
    zero_excite,
)
from torch_sci_solver.sbd_ref.helpers import TaskHelper

# ============================================================================
# Diagonal terms  (SBD: qcham.h::makeQChamDiagTerms lines 290-340)
# ============================================================================


def make_diag_terms(
    adets: List[int], bdets: List[int], n_orb: int, I0: float, I1, I2
) -> np.ndarray:
    """Compute diagonal of H over the (adets × bdets) Cartesian basis.

    Returns:
        ``hii``: 1-D numpy array of length M_a * M_b. Indexing:
        ``hii[ia * M_b + ib] = zero_excite(det_from_alpha_beta(adets[ia], bdets[ib]), ...)``
    """
    n_spinorb = 2 * n_orb
    M_a = len(adets)
    M_b = len(bdets)
    hii = np.zeros(M_a * M_b, dtype=np.float64)
    for ia in range(M_a):
        for ib in range(M_b):
            det = det_from_alpha_beta(adets[ia], bdets[ib], n_orb)
            hii[ia * M_b + ib] = zero_excite(det, n_spinorb, I0, I1, I2)
    return hii


# ============================================================================
# Matvec  (SBD: mult.h SBD_TRADMODE branch lines 113-883)
# ============================================================================


def mult(
    hii: np.ndarray,
    T: np.ndarray,
    adets: List[int],
    bdets: List[int],
    n_orb: int,
    I1,
    I2,
    helpers: List[TaskHelper],
) -> np.ndarray:
    """Compute Wb = H @ T using precomputed helpers.

    Args:
        hii:     diagonal of H (length M_a*M_b), from make_diag_terms.
        T:       input vector (length M_a*M_b).
        adets:   sorted α-bitmasks.
        bdets:   sorted β-bitmasks.
        n_orb:   number of spatial orbitals L.
        I1, I2:  integral storage (OneInt, TwoInt).
        helpers: list of TaskHelpers, one per task type. The matvec uses
                 whichever task_types are present; for a complete H@v
                 you need at least one each of taskType 0, 1, 2.

    Returns:
        Wb = H @ T, shape (M_a*M_b,)
    """
    n_spinorb = 2 * n_orb
    M_a = len(adets)
    M_b = len(bdets)
    N = M_a * M_b
    if T.shape != (N,):
        raise ValueError(f"T shape {T.shape} != ({N},)")
    if hii.shape != (N,):
        raise ValueError(f"hii shape {hii.shape} != ({N},)")

    Wb = np.zeros(N, dtype=np.float64)

    # ---- 1. Diagonal pass  (SBD: mult.h lines 169-174) ----
    Wb += hii * T

    # ---- 2-4. Task passes ----
    for h in helpers:
        if h.task_type == 2:
            _mult_alpha_only(Wb, T, adets, bdets, n_orb, n_spinorb, I1, I2, h, M_b)
        elif h.task_type == 1:
            _mult_beta_only(Wb, T, adets, bdets, n_orb, n_spinorb, I1, I2, h, M_b)
        elif h.task_type == 0:
            _mult_alphabeta(Wb, T, adets, bdets, n_orb, I2, h, M_b)
        else:
            raise ValueError(f"unknown task_type {h.task_type}")

    return Wb


def _mult_alpha_only(Wb, T, adets, bdets, n_orb, n_spinorb, I1, I2, h, M_b):
    """taskType=2 — α-singles + α-doubles, β fixed.

    SBD: mult.h lines 729-764.
    """
    M_a = len(adets)
    M_b_ = len(bdets)
    for ia in range(M_a):
        s_to = h.singles_from_alpha[ia]
        s_cran = h.singles_alpha_cran[ia]
        d_to = h.doubles_from_alpha[ia]
        d_cran = h.doubles_alpha_cran[ia]
        for ib in range(M_b_):
            DetI = det_from_alpha_beta(adets[ia], bdets[ib], n_orb)
            bra_idx = ia * M_b + ib
            # α-singles
            for k in range(len(s_to)):
                ja = s_to[k]
                cr_so, an_so = s_cran[k]
                eij = one_excite(DetI, cr_so, an_so, I1, I2, n_spinorb)
                Wb[bra_idx] += eij * T[ja * M_b + ib]
            # α-doubles
            for k in range(len(d_to)):
                ja = d_to[k]
                cr0, cr1, an0, an1 = d_cran[k]
                eij = two_excite(DetI, cr0, cr1, an0, an1, I2)
                Wb[bra_idx] += eij * T[ja * M_b + ib]


def _mult_beta_only(Wb, T, adets, bdets, n_orb, n_spinorb, I1, I2, h, M_b):
    """taskType=1 — β-singles + β-doubles, α fixed.

    SBD: mult.h lines 765-800.
    """
    M_a = len(adets)
    M_b_ = len(bdets)
    for ia in range(M_a):
        for ib in range(M_b_):
            DetI = det_from_alpha_beta(adets[ia], bdets[ib], n_orb)
            bra_idx = ia * M_b + ib
            # β-singles
            for k in range(len(h.singles_from_beta[ib])):
                jb = h.singles_from_beta[ib][k]
                cr_so, an_so = h.singles_beta_cran[ib][k]
                eij = one_excite(DetI, cr_so, an_so, I1, I2, n_spinorb)
                Wb[bra_idx] += eij * T[ia * M_b + jb]
            # β-doubles
            for k in range(len(h.doubles_from_beta[ib])):
                jb = h.doubles_from_beta[ib][k]
                cr0, cr1, an0, an1 = h.doubles_beta_cran[ib][k]
                eij = two_excite(DetI, cr0, cr1, an0, an1, I2)
                Wb[bra_idx] += eij * T[ia * M_b + jb]


def _mult_alphabeta(Wb, T, adets, bdets, n_orb, I2, h, M_b):
    """taskType=0 — αβ-doubles via outer product of α-single × β-single.

    SBD: mult.h lines 801-827.
    """
    M_a = len(adets)
    M_b_ = len(bdets)
    for ia in range(M_a):
        a_to = h.singles_from_alpha[ia]
        a_cran = h.singles_alpha_cran[ia]
        for ib in range(M_b_):
            DetI = det_from_alpha_beta(adets[ia], bdets[ib], n_orb)
            bra_idx = ia * M_b + ib
            b_to = h.singles_from_beta[ib]
            b_cran = h.singles_beta_cran[ib]
            for j in range(len(a_to)):
                ja = a_to[j]
                cr_a, an_a = a_cran[j]
                for k in range(len(b_to)):
                    jb = b_to[k]
                    cr_b, an_b = b_cran[k]
                    # SBD's call: TwoExcite(DetI, bit_length,
                    #               α_cr, β_cr,   α_an, β_an, I2)
                    # i.e., (i, j, a, b) = (cr_a, cr_b, an_a, an_b)
                    eij = two_excite(DetI, cr_a, cr_b, an_a, an_b, I2)
                    Wb[bra_idx] += eij * T[ja * M_b + jb]
