"""Davidson eigensolver matching SBD's chemistry/tpb/davidson.h.

Faithful port of sbd::Davidson (the SBD_TRADMODE branch around lines
412-637 of davidson.h), stripped of MPI parallelization.

The algorithm
-------------
A subspace Davidson with diagonal preconditioner. Each outer "iteration"
expands a subspace of size up to ``num_block`` via repeated matvec +
Gram-Schmidt + preconditioned correction. The Ritz vector after each
expansion is the running estimate of the ground state; convergence is
declared when the residual norm falls below ``eps``.

When the subspace fills (ib == num_block - 1) without convergence,
the outer loop restarts: C[0] is reset to the current best Ritz vector
W and the inner loop repeats.

Default SBD values:
    max_iteration = 1   (one outer restart)
    num_block     = 10  (subspace expansion per restart)
    eps           = 1e-4

For production reproductions we'll use larger values
(``--iteration 50 --block 30 --tolerance 1e-8`` per SBD's run.sh).
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


from typing import Callable, Optional, Tuple

import numpy as np

# ============================================================================
# BasisInitVector  (SBD: davidson.h lines 104-142)
# ============================================================================


def basis_init_vector(
    n_alpha: int, n_beta: int, init: int = 0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Initial wavefunction over the (n_alpha × n_beta) Cartesian basis.

    SBD: ``BasisInitVector(W, adet, bdet, ..., init)`` lines 104-142.

    Args:
        n_alpha, n_beta: basis sizes
        init: 0 = HF (unit vector at index 0), 1 = random uniform
        rng:  optional numpy Generator for ``init=1``
    """
    N = n_alpha * n_beta
    W = np.zeros(N, dtype=np.float64)
    if init == 0:
        W[0] = 1.0
    elif init == 1:
        if rng is None:
            rng = np.random.default_rng()
        W[:] = rng.standard_normal(N)
        n = np.linalg.norm(W)
        if n > 0:
            W /= n
    else:
        raise ValueError(f"init must be 0 (HF) or 1 (random), got {init}")
    return W


# ============================================================================
# Davidson  (SBD: davidson.h lines 412-637, SBD_TRADMODE branch)
# ============================================================================


def davidson(
    hii: np.ndarray,
    W: np.ndarray,
    matvec: Callable[[np.ndarray], np.ndarray],
    max_iteration: int,
    num_block: int,
    eps: float,
    *,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, dict]:
    """Run Davidson to find the lowest eigenpair of H, with H accessed
    only via ``matvec``.

    Args:
        hii:           1-D array, the diagonal of H. Used by the
                       preconditioner.
        W:             initial guess (length N). Modified in place by
                       the algorithm; the returned eigvec is also W.
        matvec:        callable v → H @ v
        max_iteration: number of outer restart iterations
        num_block:     maximum subspace expansion per restart
        eps:           residual-norm threshold for convergence

    Returns:
        (energy, eigvec, info_dict). info_dict has keys:
            'converged' (bool), 'n_outer' (int), 'n_matvecs' (int),
            'final_residual_norm' (float), 'history' (list of
            (it, ib, energy, residual_norm) tuples).

    SBD lines this function maps to (davidson.h):
        437-454: setup (allocate C, HC, H_proj, etc.)
        468-631: outer iteration loop
        475-619: inner subspace expansion (ib loop)
    """
    N = W.shape[0]
    if hii.shape != (N,):
        raise ValueError(f"hii shape {hii.shape} != ({N},)")

    eps_reg = 1.0e-12

    # Subspace basis (rows are vectors)  -- SBD: C[num_block][N], HC[num_block][N]
    C = np.zeros((num_block, N), dtype=np.float64)
    HC = np.zeros((num_block, N), dtype=np.float64)
    # Hermitian projected matrix  -- SBD: H[num_block * num_block]
    H_proj = np.zeros((num_block, num_block), dtype=np.float64)

    history = []
    n_matvecs = 0
    converged = False
    last_E = 0.0
    last_resnorm = float("inf")

    for it in range(max_iteration):
        # SBD line 471-473: C[0] = W
        C[0] = W

        for ib in range(num_block):
            # SBD line 478: HC[ib] = mult(C[ib])
            HC[ib] = matvec(C[ib])
            n_matvecs += 1

            # SBD lines 504-507: project new column of H_proj
            for jb in range(ib + 1):
                # <C[jb] | HC[ib]>
                H_proj[jb, ib] = np.dot(C[jb], HC[ib])
                H_proj[ib, jb] = H_proj[jb, ib]  # real symmetric

            # SBD line 514: diagonalize (ib+1) × (ib+1) submatrix
            U_sub = H_proj[: ib + 1, : ib + 1].copy()
            E, U = np.linalg.eigh(U_sub)
            # E is ascending; U[:, 0] gives Ritz vector for lowest eigenvalue

            # SBD lines 516-520: W = Σ_kb U[kb, 0] * C[kb]
            #  (in-place reconstruction)
            W_new = (U[:, 0:1] * C[: ib + 1]).sum(axis=0)
            # SBD lines 521-525: R = -Σ_kb U[kb, 0] * HC[kb]
            R = -(U[:, 0:1] * HC[: ib + 1]).sum(axis=0)
            # SBD lines 538-541: R += E[0] * W
            R += E[0] * W_new

            # SBD lines 558-562: normalize W and R, store norms
            norm_W = np.linalg.norm(W_new)
            if norm_W > 0:
                W_new /= norm_W
            norm_R = np.linalg.norm(R)
            if norm_R > 0:
                R /= norm_R

            # Update outer W to the current best Ritz vector
            W[:] = W_new
            last_E = float(E[0])
            last_resnorm = float(norm_R)

            history.append((it, ib, last_E, last_resnorm))
            if verbose:
                print(f"  Davidson iter {it}.{ib}: E={last_E:.12f}, ||R||={last_resnorm:.3e}")

            # SBD lines 588-591: convergence check
            if norm_R < eps:
                converged = True
                break

            # SBD lines 593-616: preconditioned correction + Gram-Schmidt
            if ib < num_block - 1:
                # SBD lines 595-602: C[ib+1] = R / (E[0] - dii) with reg.
                denom = E[0] - hii
                # Where |denom| <= eps_reg, perturb to (denom - eps_reg)
                small = np.abs(denom) <= eps_reg
                denom_safe = denom - eps_reg * small
                C[ib + 1] = R / denom_safe

                # SBD lines 605-613: Gram-Schmidt against C[0..ib]
                for kb in range(ib + 1):
                    olap = np.dot(C[kb], C[ib + 1])
                    C[ib + 1] -= olap * C[kb]

                # SBD lines 615-616: normalize C[ib+1]
                norm_C = np.linalg.norm(C[ib + 1])
                if norm_C > 0:
                    C[ib + 1] /= norm_C

        if converged:
            break

    info = {
        "converged": converged,
        "n_outer": it + 1,
        "n_matvecs": n_matvecs,
        "final_residual_norm": last_resnorm,
        "history": history,
    }
    return last_E, W, info
