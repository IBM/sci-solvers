"""
Multi-root truncated Davidson algorithm (Cotton, J. Chem. Phys. 157, 224105 (2022), Fig. 3),
extended to the lowest `nroots` eigenpairs, with subspace restart.

Buffer roots (GD+k-style) are solved alongside the requested `nroots` and held to the same
convergence criteria, preventing the last tracked root from being under-resolved at the
tracked/buffer boundary. Output is the true lowest-`nroots` eigenpairs from the full converged
set (see `_select_bottom_k`). Every candidate residual is accepted into the basis
unconditionally; `lindep_cutoff` is only used inside `_solve_generalized_eigenproblem`.

Restart: whenever appending this iteration's accepted correction vectors would push the
subspace size above `max_space_size`, the subspace is collapsed. The new basis is built from
the current best Ritz vectors for *all* tracked+buffer slots (converged or not -- this is the
"locking" of already-converged roots across restarts) plus this iteration's freshly computed
correction vectors, so no PCR work already done this round is discarded.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import math
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

import sci_solvers.energy_solvers.truncated_davidson_solver.sparse_vector as spv
from sci_solvers.energy_solvers.truncated_davidson_solver.calc_trunc_pcr import (
    SortedIntegralTables,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.calc_trunc_pcr import (
    calc_trunc_pcr as calc_trunc_pcr_fig4,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.hamiltonian import Hamiltonian
from sci_solvers.energy_solvers.truncated_davidson_solver.numba_diag import (
    _count_singles_doubles,
    excite_single_double_nb,
    excite_single_nb,
    hii_batch,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.subspace_hamiltonian import (
    SubspaceHamiltonianBuilder,
)


def _solve_generalized_eigenproblem(
    Hmat: np.ndarray, Smat: np.ndarray, nroots: int = 1, cutoff: float = 1e-7
):
    """Solve H c = E S c for the lowest `nroots` pairs via Loewdin orthogonalization.

    Returns (E, C, effective_rank): E shape (k,) ascending, C shape (n, k),
    k = min(nroots, rank(S) under cutoff).
    """
    n = Smat.shape[0]
    if n == 1:
        E = np.array([Hmat[0, 0] / Smat[0, 0]])
        C = np.array([[1.0 / np.sqrt(Smat[0, 0])]])
        return E, C, 1

    w, U = eigh(Smat)
    w_max = w[-1]
    keep = w > cutoff * w_max
    Uk = U[:, keep]
    wk = w[keep]
    effective_rank = int(keep.sum())

    X = Uk / np.sqrt(wk)
    Hred = X.T @ Hmat @ X
    evals, evecs = eigh(Hred)

    k = min(nroots, evals.size)
    E = evals[:k]
    C = X @ evecs[:, :k]
    return E, C, effective_rank


def _build_ritz_vectors(C, basis):
    """Construct Ritz vectors from subspace coefficients C (n_basis, k) and basis."""
    k = C.shape[1]
    psi = []
    for r in range(k):
        acc_keys, acc_vals = spv.init_flat_growable()
        for ci, (bi_idx, bi_val) in zip(C[:, r], basis):
            if ci != 0.0:
                acc_keys, acc_vals = spv.axpy_grow(acc_keys, acc_vals, bi_idx, bi_val, float(ci))
        psi.append((acc_keys, acc_vals))
    return psi


def _sort_roots_by_similarity(E_new, psi_new, psi_prev):
    """Reorder (E_new, psi_new) to best match previous-iteration roots via Hungarian matching
    on squared overlaps. No-op on the first iteration or if root count changed.
    """
    if psi_prev is None or len(psi_new) == 1:
        return E_new, psi_new

    n_new = len(psi_new)
    n_prev = len(psi_prev)
    if n_new != n_prev:
        return E_new, psi_new

    # cost[a, b] = -|<psi_new[a] | psi_prev[b]>|^2
    # (linear_sum_assignment minimizes cost, so negate the overlap)
    cost = np.zeros((n_new, n_prev))
    for a in range(n_new):
        for b in range(n_prev):
            ak, av = psi_new[a]
            bk, bv = psi_prev[b]
            cost[a, b] = -(spv.dot(ak, av, bk, bv) ** 2)

    row_idx, col_idx = linear_sum_assignment(cost)
    # col_idx[a] tells us which "previous root slot" new root `a` maps to.
    # We want to reorder new roots into previous-root-slot order.
    order = np.argsort(col_idx)
    perm = row_idx[order]

    return E_new[perm], [psi_new[i] for i in perm]


def _excite_single_double(ref, norb):
    nelec = bin(ref).count("1")
    nvir = norb - nelec
    buf = np.empty(_count_singles_doubles(nelec, nvir), dtype=np.int64)
    n = excite_single_double_nb(np.int64(ref), norb, buf)
    return set(buf[:n].tolist())


def _excite_single(ref, norb):
    nelec = bin(ref).count("1")
    nvir = norb - nelec
    buf = np.empty(1 + nelec * nvir, dtype=np.int64)
    n = excite_single_nb(np.int64(ref), norb, buf)
    return set(buf[:n].tolist())


def _sd_excited_pairs(str_a0, str_b0, norb):
    """All (str_a, str_b) pairs reachable from (str_a0, str_b0) by a single or
    double excitation of the combined determinant (pure-alpha, pure-beta, or
    one-one mixed)."""
    a_sd = _excite_single_double(str_a0, norb)
    b_sd = _excite_single_double(str_b0, norb)
    a_s = _excite_single(str_a0, norb)
    b_s = _excite_single(str_b0, norb)

    pairs: set[tuple[int, int]] = set()
    pairs.update((sa, str_b0) for sa in a_sd)
    pairs.update((str_a0, sb) for sb in b_sd)
    pairs.update((sa, sb) for sa in a_s for sb in b_s)
    return pairs


def _lowest_diagonal_seed_vectors_sd_otf(H, k):
    """Seed vectors for the `k` lowest-H_ii determinants in the SD neighborhood
    of the HF reference, without scanning all na*nb determinants.

    Uses hii_batch for a single vectorized diagonal evaluation over the
    candidate set, then selects the k lowest by energy.
    """
    norb = H.norb
    neleca, nelecb = H.nelec

    hf_a = int((1 << neleca) - 1)
    hf_b = int((1 << nelecb) - 1)

    cand_pairs = list(_sd_excited_pairs(hf_a, hf_b, norb))

    if H.selected:
        valid_a = set(H.strs_a.tolist())
        valid_b = set(H.strs_b.tolist())
        cand_pairs = [(sa, sb) for sa, sb in cand_pairs if sa in valid_a and sb in valid_b]

    if not cand_pairs:
        return []

    k = min(k, len(cand_pairs))

    sa_arr = np.array([p[0] for p in cand_pairs], dtype=np.int64)
    sb_arr = np.array([p[1] for p in cand_pairs], dtype=np.int64)
    diag = np.empty(len(cand_pairs), dtype=np.float64)
    hdiag1, jdiag, kdiag = H._diag_ints
    hii_batch(sa_arr, sb_arr, hdiag1, jdiag, kdiag, neleca, nelecb, diag)

    idx = np.argpartition(diag, k - 1)[:k]
    idx = idx[np.argsort(diag[idx])]

    seeds = []
    for ii in idx:
        sa, sb = cand_pairs[ii]
        addr_a = int(np.searchsorted(H.strs_a, sa))
        addr_b = int(np.searchsorted(H.strs_b, sb))
        flat_idx = addr_a * H.nb + addr_b
        seeds.append(spv.from_flat(*spv.from_single(flat_idx, 1.0)))
    return seeds


def _lowest_diagonal_seed_vectors(H, k):
    """Return `k` unit vectors for the determinants with the smallest H_II, ascending.

    Pass k = nroots + n_extra_roots so every tracked and buffer root gets a distinct
    starting direction, reducing near-parallel first residuals.
    """
    dim = H.na * H.nb
    if H._hdiag is not None:
        diag = np.asarray(H._hdiag).reshape(-1)
        k = min(k, dim)
        idx = np.argpartition(diag, k - 1)[:k]
        idx = idx[np.argsort(diag[idx])]
    else:
        return _lowest_diagonal_seed_vectors_sd_otf(H, k)

    seeds = []
    for i in idx:
        seeds.append(spv.from_flat(*spv.from_single(int(i), 1.0)))
    return seeds


def _select_bottom_k(E_all, psi_all, k):
    """Return the `k` lowest-energy (E, psi) pairs from the full tracked+buffer set.

    Sorts by energy rather than trusting slot order, which similarity-tracking can
    leave slightly out of ascending-energy order near degeneracies.
    """
    order = np.argsort(E_all)
    k = min(k, order.size)
    idx = order[:k]
    return E_all[idx], [spv.from_flat(*psi_all[i]) for i in idx]


def _restart_subspace(H, psi_all, k_full, accepted, backend="numba"):
    """Collapse the subspace: rebuild a fresh SubspaceHamiltonianBuilder from the
    current best Ritz vectors (racked+buffer).

    Parameters
    ----------
    H : Hamiltonian
    psi_all : list of SparseVector
        Current-iteration Ritz vectors, at least k_full long.
    k_full : int
        Number of tracked+buffer slots to retain (nroots_total, or fewer if the
        subspace rank hasn't supported that many yet).
    accepted : list of (slot_idx, SparseVector)
        This round's freshly computed, truncated correction vectors.
    backend : str
        ``"numba"`` or ``"cpp"``.

    Returns
    -------
    SubspaceHamiltonianBuilder
        A new builder containing only the retained + accepted vectors.
    """
    new_builder = SubspaceHamiltonianBuilder(H, backend=backend)

    for r in range(k_full):
        vk, vv = psi_all[r]
        spv.normalize(vk, vv)
        new_builder.add_basis_vector(spv.from_flat(vk, vv))

    for _slot_idx, vec in accepted:
        new_builder.add_basis_vector(vec)

    return new_builder


def truncated_davidson(
    ci_strings: tuple[np.ndarray, np.ndarray] | None,
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    nroots: int = 1,
    n_extra_roots: int = 1,
    eps: float = 1e-4,
    tol_residual: float | None = None,
    max_iter: int = 50,
    delta_max: float = 1e-3,
    delta_min: float = 1e-7,
    growth_factor: float = 2.0,
    precomputed_diag: bool | None = None,
    lindep_cutoff: float = 1e-7,
    plateau_window: int = 3,
    plateau_rel_tol: float = 0.3,
    require_buffer_convergence: bool = True,
    max_space_size: int | None = None,
    ignore_res: bool | None = True,
    verbose: bool = False,
    print_every: int = 5,
    e_nuc: float = 0.0,
    add_reference_determinant: bool = True,
    backend: str = "numba",
) -> tuple[np.ndarray, list, int, list]:
    """
    Multi-root truncated Davidson solver, with subspace restart.

    Subspace matrices H and S are built incrementally via SubspaceHamiltonianBuilder;
    CALC_TRUNC_PCR generates residuals per active root using a sorted integral table.
    All candidate residuals are added to the basis unconditionally, unless doing so
    would exceed `max_space_size`, in which case a restart is applied.

    Parameters
    ----------
    ci_strings : (strs_a, strs_b)
        Alpha/beta CI strings (int64 occupation bitmasks). Full FCI strings give an
        exact solve; a selected-CI subset restricts the space.
    one_body_tensor : (norb, norb) float64
        One-electron integrals.
    two_body_tensor : array_like
        Two-electron integrals, chemist notation.
    norb : int
    nelec : (nelec_a, nelec_b)
    nroots : int
        Number of eigenpairs to return.
    n_extra_roots : int
        Buffer roots solved and convergence-tracked alongside `nroots` (GD+k). They
        improve subspace resolving power near degeneracies; output is always the true
        lowest `nroots` by energy (see `_select_bottom_k`). Set 0 to disable.
    eps : float
        Per-root energy-change convergence threshold.
    tol_residual : float or None
        Per-root residual-norm threshold. Both `eps` and `tol_residual` must be
        satisfied. Defaults to sqrt(eps).
    max_iter : int
    delta_max, delta_min : float
        Initial / minimum PCR truncation threshold.
    growth_factor : float
        Per-slot growth factor for the truncation target size T each iteration.
    precomputed_diag : bool
        Read Hamiltonian diagonal from cache.
    lindep_cutoff : float
        Relative S-eigenvalue cutoff inside `_solve_generalized_eigenproblem`.
    plateau_window : int
        Residual-norm history length for stagnation detection.
    plateau_rel_tol : float
        Accept a root as converged if its last `plateau_window` residual norms vary
        by less than this fraction of their magnitude and `eps` is already satisfied
        (handles the truncation-induced residual floor).
    require_buffer_convergence : bool
        If True, wait for all `nroots_total` slots to converge before stopping.
        If False, stop once the first `nroots` slots converge (cheaper, but the last
        requested root may be less accurate when near-degenerate with its buffer
        neighbor).
    max_space_size : int or None
        Collapse (restart) the subspace whenever n_basis is above this cap. Default is no restart.
    ignore_res : bool or None
        If True, skip the residual-norm check and converge on energy change alone. Defaults to
        True when nroots=1, False otherwise.
    add_reference_determinant : bool
        If True (default), inject the Hartree-Fock reference determinant into the selected
        `ci_strings` space when it is absent, so seeding never fails on a space that happens to
        exclude the reference. If False, use `ci_strings` verbatim and raise if the reference is
        needed but missing. No effect on the full FCI space (`ci_strings=None`).
    verbose : bool

    Returns
    -------
    (E_ritz, psi_ritz, dim, history)
        E_ritz : (k,) ascending, k = min(nroots, achieved subspace rank).
        psi_ritz : list of k SparseVector Ritz vectors.
        dim : na * nb (dense CI vector length).
        history : list of per-iteration diagnostic dicts.
    """
    if tol_residual is None:
        tol_residual = math.sqrt(eps)
    if ignore_res is None:
        ignore_res = nroots == 1
    if precomputed_diag is None:
        precomputed_diag = norb <= 18

    H = Hamiltonian.from_arrays(
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        ci_strs=ci_strings,
        precomputed_diag=precomputed_diag,
        verbose=verbose,
        e_nuc=e_nuc,
        add_reference_determinant=add_reference_determinant,
    )
    builder = SubspaceHamiltonianBuilder(H, backend=backend)
    t0 = time.perf_counter()
    tables = SortedIntegralTables(builder.eri, builder.norb)
    t_tables = time.perf_counter() - t0
    if verbose:
        print(f"  SortedIntegralTables: {t_tables * 1e3:.1f}ms")

    if nroots == 1:
        n_extra_roots = 0
    nroots_total = nroots + n_extra_roots

    space_cap = math.inf if max_space_size is None else max_space_size
    if space_cap < nroots_total:
        raise ValueError(
            f"max_space_size ({space_cap}) must be >= nroots_total "
            f"({nroots_total}) or restart can never retain all tracked/buffer roots."
        )

    t0 = time.perf_counter()
    if nroots == 1:
        init_vectors = [H.hf_vector()]
        if verbose:
            print(
                f"  seeding initial basis with HF determinant (nroots=1, "
                f"n_extra_roots={n_extra_roots})"
            )
    else:
        init_vectors = _lowest_diagonal_seed_vectors(H, nroots_total)
        if verbose:
            print(
                f"  seeding initial basis with {len(init_vectors)} "
                f"lowest-diagonal determinant(s) (nroots={nroots}, "
                f"n_extra_roots={n_extra_roots})"
            )
    for v in init_vectors:
        builder.add_basis_vector(v)
    t_init = time.perf_counter() - t0
    if verbose:
        print(f"  init subspace: {t_init * 1e3:.1f}ms")

    E_save = np.full(nroots_total, np.nan)
    converged = np.zeros(nroots_total, dtype=bool)
    last_contrib_size = [1] * nroots_total
    residual_history: dict[int, list[float]] = {r: [] for r in range(nroots_total)}

    psi_ritz_prev = None
    history = []
    n = 1
    t_iter_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        Hmat, Smat = builder.Hmat, builder.Smat
        t_hmat = time.perf_counter() - t0

        t0 = time.perf_counter()
        E_all, C, effective_rank = _solve_generalized_eigenproblem(
            Hmat, Smat, nroots=nroots_total, cutoff=lindep_cutoff
        )
        t_eig = time.perf_counter() - t0

        k_avail = E_all.size
        k_tracked = min(nroots, k_avail)
        k_full = min(nroots_total, k_avail)
        if k_avail < nroots and verbose:
            print(
                f"  WARNING: subspace rank only supports {k_avail}/{nroots} "
                f"requested roots this iteration (effective_rank={effective_rank})"
            )

        t0 = time.perf_counter()
        psi_all = _build_ritz_vectors(C, builder.basis)
        t_ritz = time.perf_counter() - t0

        t0 = time.perf_counter()
        E_all, psi_all = _sort_roots_by_similarity(E_all, psi_all, psi_ritz_prev)
        t_sort = time.perf_counter() - t0

        E_tracked = E_all[:k_tracked]

        history.append({
            "iteration": n,
            "n_basis": len(builder.basis),
            "basis_sizes": [len(b[0]) for b in builder.basis],
            "E_ritz": E_tracked.copy(),
            "effective_rank": effective_rank,
            "restarted": False,
        })
        if verbose:
            print(
                f"iter {n:3d}   n_basis={len(builder.basis):3d}   "
                f"|b_n|={len(builder.basis[-1][0]):7d}   "
                f"rank={effective_rank:3d}   "
                f"E_ritz={np.array2string(E_tracked, precision=10)}"
            )
            if n % print_every == 0:
                e_elec = float(E_tracked[0])
                e_nuc = H.data.e_nuc
                t_iter = time.perf_counter() - t_iter_start
                sep = "─" * 48
                print(
                    f"  {sep}\n"
                    f"  iter {n}\n"
                    f"  solve time:        {t_iter:.1f} s\n"
                    f"  electronic energy: {e_elec:.10f} Ha\n"
                    f"  total energy:      {e_elec + e_nuc:.10f} Ha\n"
                    f"  {sep}"
                )

        if n > 1:
            de = np.full(k_full, np.inf)
            valid = ~np.isnan(E_save[:k_full])
            de[valid] = np.abs(E_save[:k_full][valid] - E_all[:k_full][valid])
            energy_ok = de < eps
        else:
            energy_ok = np.zeros(k_full, dtype=bool)

        E_save = np.full(nroots_total, np.nan)
        E_save[:k_full] = E_all[:k_full]

        candidates = []
        residual_norms = {}

        for r in range(k_avail):
            is_tracked = r < nroots
            if converged[r]:
                continue

            pk, pv = psi_all[r]
            spv.normalize(pk, pv)

            T = math.ceil(growth_factor * last_contrib_size[r])
            delta = min(delta_max, max(delta_min, 0.1 / T))

            t0 = time.perf_counter()
            res_keys, res_vals = calc_trunc_pcr_fig4(builder, tables, (pk, pv), E_all[r], delta)
            t_pcr = time.perf_counter() - t0
            if not ignore_res:
                residual_norms[r] = spv.dot(res_keys, res_vals, res_keys, res_vals) ** 0.5

            t0 = time.perf_counter()
            res_keys, res_vals = spv.compact_truncate(res_keys, res_vals, T)
            t_trunc = time.perf_counter() - t0

            spv.compact_normalize(res_vals)
            res_compact = (res_keys, res_vals)
            candidates.append((r, res_compact))

            if verbose:
                tag = "buffer" if not is_tracked else "root"
                norm_str = f"|r|_pretrunc={residual_norms[r]:.3e}  " if not ignore_res else ""
                print(
                    f"         {tag} {r}: delta={delta:.2e}  T={T}  "
                    f"{norm_str}"
                    f"t_pcr={t_pcr * 1e3:.1f}ms  t_trunc={t_trunc * 1e3:.1f}ms"
                )

        for r in range(k_full):
            if converged[r]:
                continue

            if ignore_res:
                if energy_ok[r]:
                    converged[r] = True
                continue

            res_norm = residual_norms.get(r)
            if res_norm is None:
                continue

            hist = residual_history[r]
            hist.append(res_norm)
            if len(hist) > plateau_window:
                hist.pop(0)
            plateaued = len(hist) == plateau_window and (
                max(hist) - min(hist) < plateau_rel_tol * max(hist[-1], 1e-300)
            )

            if energy_ok[r] and (res_norm < tol_residual or plateaued):
                converged[r] = True
                if verbose and plateaued and res_norm >= tol_residual:
                    print(
                        f"         slot {r} converged via residual-plateau "
                        f"detection (residual stuck at {res_norm:.3e}, "
                        f"above tol_residual={tol_residual:.3e} but not "
                        f"decreasing -- likely a truncation-induced floor)"
                    )

        if require_buffer_convergence:
            all_done = (
                k_avail >= nroots_total and np.all(converged[:k_full]) and k_full == nroots_total
            )
        else:
            all_done = k_avail >= nroots and np.all(converged[:nroots])

        if all_done:
            E_out, psi_out = _select_bottom_k(E_all, psi_all, nroots)
            return E_out, psi_out, H.na * H.nb, history

        if n >= max_iter:
            E_out, psi_out = _select_bottom_k(E_all, psi_all, nroots)
            return E_out, psi_out, H.na * H.nb, history

        accepted = list(candidates)

        if len(accepted) == 0:
            E_out, psi_out = _select_bottom_k(E_all, psi_all, nroots)
            return E_out, psi_out, H.na * H.nb, history

        n_basis_after_growth = len(builder.basis) + len(accepted)
        do_restart = n_basis_after_growth > space_cap

        t0 = time.perf_counter()
        if do_restart:
            n_basis_before = len(builder.basis)
            builder = _restart_subspace(H, psi_all, k_full, accepted, backend=backend)
            for slot_idx, vec in accepted:
                last_contrib_size[slot_idx] = len(vec[0])
            history[-1]["restarted"] = True
            if verbose:
                print(
                    f"         ** restart **  n_basis {n_basis_before} -> "
                    f"{len(builder.basis)}  (cap={space_cap})"
                )
        else:
            for slot_idx, vec in accepted:
                builder.add_basis_vector(vec)
                last_contrib_size[slot_idx] = len(vec[0])
        t_add_basis = time.perf_counter() - t0

        if verbose:
            print(
                f"         t_add_basis={t_add_basis * 1e3:.1f}ms  "
                f"t_Hmat={t_hmat * 1e3:.1f}ms   t_eig={t_eig * 1e3:.1f}ms   "
                f"t_ritz={t_ritz * 1e3:.1f}ms   t_sort={t_sort * 1e3:.1f}ms"
            )

        if nroots_total > 1:
            psi_ritz_prev = psi_all
        n += 1
