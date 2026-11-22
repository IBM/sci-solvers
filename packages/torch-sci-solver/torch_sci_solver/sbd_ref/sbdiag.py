"""Top-level driver — analog of SBD's ``sbd::tpb::diag``.

Faithful port of sbdiag.h lines 126-407 (the bare-H energy calculation
path), stripped of MPI and RDM / carryover output. Returns the
ground-state energy and eigenvector. Higher-level features
(spin projection, RDM construction, configuration recovery) are
deliberately out of scope here.
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

import numpy as np

from torch_sci_solver.sbd_ref.davidson import basis_init_vector, davidson
from torch_sci_solver.sbd_ref.helpers import generate_all_helpers
from torch_sci_solver.sbd_ref.mult import make_diag_terms, mult


def diag(
    adets: List[int],
    bdets: List[int],
    n_orb: int,
    I0: float,
    I1,
    I2,
    *,
    max_iteration: int = 50,
    num_block: int = 30,
    eps: float = 1e-8,
    init: int = 0,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, dict]:
    """Ground-state energy + eigenvector via Davidson on H, no spin
    projection.

    Args:
        adets, bdets:  sorted α and β bitmask lists, each over n_orb bits.
                       Index 0 of each is the HF reference contribution
                       (lexicographically lowest, which for typical
                       orbital ordering = lowest spatial orbitals occupied).
        n_orb:         number of spatial orbitals.
        I0, I1, I2:    integral storage from
                       ``setup_integrals_from_pyscf_h1_eri``.
        max_iteration: outer Davidson restart count (SBD --iteration).
        num_block:     subspace expansion per restart (SBD --block).
        eps:           residual-norm convergence threshold (SBD --tolerance).
        init:          0 = HF init (default), 1 = random.
        verbose:       per-iteration prints.

    Returns:
        (energy, eigvec, info)
        ``info`` is the Davidson info dict from davidson().

    SBD entry: ``sbd::tpb::diag`` in sbdiag.h lines 126-407, method=0
    (matrix-free) branch, no RDM.
    """
    M_a = len(adets)
    M_b = len(bdets)
    if verbose:
        print(f"[sbdiag] basis: M_a={M_a}, M_b={M_b}, dim={M_a * M_b}")

    # --- Setup helpers (SBD: lines 238-250) ---
    helpers = list(generate_all_helpers(adets, bdets, n_orb))
    if verbose:
        # Report how many excitations each task type generates
        n_a_singles = sum(len(x) for x in helpers[0].singles_from_alpha)
        n_a_doubles = sum(len(x) for x in helpers[0].doubles_from_alpha)
        n_b_singles = sum(len(x) for x in helpers[1].singles_from_beta)
        n_b_doubles = sum(len(x) for x in helpers[1].doubles_from_beta)
        print(f"[sbdiag] α-singles: {n_a_singles}, α-doubles: {n_a_doubles}")
        print(f"[sbdiag] β-singles: {n_b_singles}, β-doubles: {n_b_doubles}")

    # --- Initial wavefunction (SBD: line 265) ---
    W = basis_init_vector(M_a, M_b, init=init)

    # --- Diagonal terms (SBD: line 311-315) ---
    hii = make_diag_terms(adets, bdets, n_orb, I0, I1, I2)
    if verbose:
        print(f"[sbdiag] diagonal: min={hii.min():.10f}, max={hii.max():.10f}")
        print(f"[sbdiag]           HF diag (hii[0]) = {hii[0]:.10f}")

    # --- Davidson (SBD: lines 332-338) ---
    def matvec(v):
        """Matvec."""
        return mult(hii, v, adets, bdets, n_orb, I1, I2, helpers)

    energy, eigvec, info = davidson(hii, W, matvec, max_iteration, num_block, eps, verbose=verbose)

    if verbose:
        print(
            f"[sbdiag] Davidson done: converged={info['converged']}, "
            f"n_matvecs={info['n_matvecs']}, "
            f"final ||R||={info['final_residual_norm']:.3e}"
        )
        print(f"[sbdiag] Energy = {energy:.12f}")

    return energy, eigvec, info
