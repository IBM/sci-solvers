"""SQD iteration loop on top of `diag_torch`.

Pipeline (one iteration):

    1. recover_configurations(shots, probabilities, prior_occupancies,
                                 n_a, n_b, seed)
                                 → bm_recovered                # ndarray (n_shots, 2L)
    2. bitstring_matrix_to_ci_strs(bm_recovered)
                                 → (ci_strs_α, ci_strs_β)
       dedupe + sort             → (adets, bdets)              # closed-shell:
                                                                   adets == bdets
    3. diag_torch(adets, bdets, ...)
                                 → energy, eigvec, info
    4. make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, L)
                                 → (occ_α_new, occ_β_new)
    5. Convergence: |E_n - E_{n-1}| < tol_E and ||occ_n - occ_{n-1}|| < tol_occ

Iteration 0 uses a user-supplied prior, defaulting to **uniform**
``[n_a/L]`` (the uniform-prior SQD workflow convention). If you want HF
instead (and the connectivity it provides for
the first-iteration basis), pass
``initial_prior=([1.0]*n_a + [0.0]*(L-n_a),) * 2``.

Convergence behavior with uniform first prior: as established in the
Anchor 3 calibration work, uniform priors on small subsample shots
produce disconnected first-iteration bases. The iteration loop is
robust to this — even a disconnected basis returns a valid (if
trivial) eigvec, the 1-RDM diagonal is then the per-string occupancy
distribution, and subsequent iterations refine. Whether convergence
to the true ground state happens within max_iter depends on the shot
fixture and basis sizes.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import List, Optional, Tuple

import numpy as np
import torch

from torch_sci_solver.sbd_ref.integrals import OneInt, TwoInt
from torch_sci_solver.torch_impl.rdm import make_one_rdm_diag_torch
from torch_sci_solver.torch_impl.sbdiag import _to_int64_python, diag_torch


def _bm_from_bitstrings(bitstrings, dtype=bool):
    """Bool matrix of shape (n_shots, 2L) from list of '0'/'1' strings."""
    return np.array([[c == "1" for c in bs] for bs in bitstrings], dtype=dtype)


def sqd_loop(
    bitstrings: List[str],
    probabilities: np.ndarray,
    n_orb: int,
    n_a: int,
    n_b: int,
    I0: float,
    I1: OneInt,
    I2: TwoInt,
    *,
    initial_prior: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    max_iter: int = 10,
    tol_E: float = 1e-6,
    tol_occ: float = 1e-4,
    recovery_seed: int = 1337,
    diag_max_davidson_iters: int = 50,
    diag_davidson_block_size: int = 30,
    diag_davidson_tolerance: float = 1e-7,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 2_000_000,
    verbose: bool = True,
) -> dict:
    """Run iterative SQD on the user's shots.

    Args:
        bitstrings:     list of '0'/'1' strings, one per shot. Length n_shots.
        probabilities:  (n_shots,) array of shot weights. Will be normalized.
        n_orb, n_a, n_b: spatial-orbital count and electron counts.
        I0, I1, I2:     integrals (from setup_integrals_from_pyscf_h1_eri).
        initial_prior:  optional (occ_α, occ_β), each (L,) array. Default
                        is uniform ``[n_a/L, ...]``, the uniform-prior SQD
                        workflow convention.
        max_iter:       max SQD iterations.
        tol_E:          stop when |ΔE| < tol_E across one iteration.
        tol_occ:        stop when ||occ_new - occ_old|| < tol_occ.
        recovery_seed:  seed passed to recover_configurations.
        diag_max_davidson_iters, diag_davidson_block_size, diag_davidson_tolerance:
                        Davidson parameters for the inner CI solve.
        device, dtype, chunk_size:
                        torch backend for the diag.
        verbose:        per-iteration prints.

    Returns:
        A dict with these keys:
            'converged' (bool): whether tol_E + tol_occ were satisfied.
            'n_iter' (int):     number of iterations completed.
            'final_energy' (float)
            'final_eigvec' (torch.Tensor)
            'final_basis': (adets, bdets) from the last iteration.
            'final_occupancies': (occ_α, occ_β) numpy arrays.
            'history': list of dicts, one per iteration, with keys:
                'iter', 'basis_size', 'M_a', 'M_b', 'energy', 'delta_E',
                'occ_drift', 'n_matvecs', 'final_residual'.
    """
    if isinstance(device, str):
        device = torch.device(device)
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

    n_shots = len(bitstrings)
    probabilities = np.asarray(probabilities, dtype=np.float64).copy()
    probabilities /= probabilities.sum()
    bm = _bm_from_bitstrings(bitstrings)

    # Initial prior
    if initial_prior is None:
        occ_a = np.full(n_orb, n_a / n_orb, dtype=np.float64)
        occ_b = np.full(n_orb, n_b / n_orb, dtype=np.float64)
    else:
        occ_a = np.asarray(initial_prior[0], dtype=np.float64).copy()
        occ_b = np.asarray(initial_prior[1], dtype=np.float64).copy()
        if occ_a.shape != (n_orb,) or occ_b.shape != (n_orb,):
            raise ValueError(f"initial_prior shapes must be ({n_orb},)")

    if verbose:
        print(f"[sqd_loop] starting: {n_shots} shots, max_iter={max_iter}, device={device}")
        print(f"[sqd_loop] initial prior: α uniform = {occ_a[0]:.4f}, β uniform = {occ_b[0]:.4f}")

    history = []
    last_E = float("inf")
    last_occ_a = occ_a.copy()
    last_occ_b = occ_b.copy()
    converged = False
    n_done = 0
    eigvec = None
    energy = float("nan")
    adets: List[int] = []
    bdets: List[int] = []

    for it in range(max_iter):
        # 1. Recovery
        bm_rec, _ = recover_configurations(
            bm,
            probabilities,
            (occ_a, occ_b),
            num_elec_a=n_a,
            num_elec_b=n_b,
            rand_seed=recovery_seed,
        )
        ci_a, ci_b = bitstring_matrix_to_ci_strs(bm_rec, open_shell=False)
        adets = sorted(int(x) for x in ci_a)
        bdets = sorted(int(x) for x in ci_b)
        basis_size = len(adets) * len(bdets)

        # 2. Diag (only if basis changed and is non-empty)
        if len(adets) == 0 or len(bdets) == 0:
            if verbose:
                print(f"[sqd_loop] iter {it}: empty basis, aborting")
            break
        energy, eigvec, info = diag_torch(
            adets,
            bdets,
            n_orb,
            I0,
            I1,
            I2,
            max_davidson_iters=diag_max_davidson_iters,
            davidson_block_size=diag_davidson_block_size,
            eps=diag_davidson_tolerance,
            init=0,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
            verbose=False,
        )

        # 3. 1-RDM diag for the next iteration's prior
        adets_t = torch.tensor(_to_int64_python(adets), dtype=torch.int64, device=device)
        bdets_t = torch.tensor(_to_int64_python(bdets), dtype=torch.int64, device=device)
        occ_a_t, occ_b_t = make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, n_orb)
        occ_a_new = occ_a_t.cpu().numpy()
        occ_b_new = occ_b_t.cpu().numpy()

        # 4. Convergence metrics
        delta_E = float("inf") if it == 0 else abs(energy - last_E)
        occ_drift = float(
            np.linalg.norm(np.concatenate([occ_a_new - last_occ_a, occ_b_new - last_occ_b]))
        )

        record = dict(
            iter=it,
            basis_size=basis_size,
            M_a=len(adets),
            M_b=len(bdets),
            energy=float(energy),
            delta_E=delta_E,
            occ_drift=occ_drift,
            n_matvecs=info["n_matvecs"],
            final_residual=info["final_residual_norm"],
        )
        history.append(record)

        if verbose:
            print(
                f"[sqd_loop] iter {it}: |α|×|β| = {len(adets)}×{len(bdets)} "
                f"= {basis_size}, E = {energy:.10f}, "
                f"|ΔE| = {delta_E:.3e}, |Δocc| = {occ_drift:.3e}, "
                f"matvecs = {info['n_matvecs']}"
            )

        # 5. Convergence check (only after iter 0)
        if it > 0 and delta_E < tol_E and occ_drift < tol_occ:
            converged = True
            n_done = it + 1
            if verbose:
                print(f"[sqd_loop] converged after {n_done} iterations")
            break

        # Roll over for next iteration
        last_E = energy
        last_occ_a = occ_a_new.copy()
        last_occ_b = occ_b_new.copy()
        occ_a = occ_a_new
        occ_b = occ_b_new
        n_done = it + 1

    return dict(
        converged=converged,
        n_iter=n_done,
        final_energy=float(energy),
        final_eigvec=eigvec,
        final_basis=(adets, bdets),
        final_occupancies=(occ_a, occ_b),
        history=history,
    )
