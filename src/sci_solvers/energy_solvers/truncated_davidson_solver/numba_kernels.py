"""
numba_kernels.py
================

Parallelization strategy (add_basis_vector_fused / calc_trunc_pcr_fused)
-------------------------------------------------------------------------
The two big driver loops (`for J_idx, cJ in new_vector.items()` and
`for I_idx, cI in psi.items()`) cannot simply be turned into
`prange` loops as written, for two reasons:

  1.`NumbaDict.items()` is not directly `prange`-iterable -- prange
     needs an integer index range.
  2. Each iteration scatters accumulations into a shared mutable
     structure (`col`, a float64 array, or `r`, a NumbaDict) at
     data-dependent indices. Writing `col[i] += ...` from multiple
     threads is a data race whenever two threads land on the same
     index, and NumbaDict is not thread-safe for concurrent writes/
     resizes at all (even at disjoint keys).

So both functions now: snapshot the driving dict into flat
`keys`/`values` arrays, split the index range into `n_threads`
contiguous chunks, give every thread its own private accumulator
(a private row of `col`, or a private `NumbaDict`) to write into
lock-free, and only after the parallel region do a reduction that
sums the per-thread partial results into the final output.

"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
from numba import get_num_threads, njit, prange
from numba.core import types
from numba.extending import overload
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

from sci_solvers.energy_solvers.truncated_davidson_solver.hash_table import (
    EMPTY,
    ht_add,
    ht_add_grow,
    ht_new,
    ht_new_growable,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.numba_diag import (
    hii_from_bits,
)


def _addr_get(table, key):  # noqa: ARG001
    """Return address for `key` from either an int64 array or a NumbaDict[int64,int64].

    Array tables store -1 at positions not in the determinant space; dict tables return -1 for missing keys.
    """
    raise NotImplementedError  # Python body never called; overloads below handle it


@overload(_addr_get, inline="always")
def _addr_get_array(table, key):  # noqa: ARG001
    if isinstance(table, types.Array):

        def impl(table, key):
            return table[key]

        return impl


@overload(_addr_get, inline="always")
def _addr_get_dict(table, key):  # noqa: ARG001
    if isinstance(table, types.DictType):

        def impl(table, key):
            return table.get(key, np.int64(-1))

        return impl


@njit(inline="always")
def _ediag(
    use_dense_hdiag,
    new_flat,
    bits_a_val,
    bits_b_val,
    hdiag,
    hdiag1,
    jdiag,
    kdiag,
    nelec_a,
    nelec_b,
):
    """
    Diagonal element for the target determinant (bits_a_val, bits_b_val),
    whose flat index is new_flat.

    use_dense_hdiag=True  -> O(1) lookup into the real precomputed
        hdiag array; no on-the-fly compute.
    use_dense_hdiag=False -> O(nelec_a^2 + nelec_b^2 + nelec_a*nelec_b)
        computed on-the-fly via hii_from_bits.
    """
    if use_dense_hdiag:
        return hdiag[new_flat]
    return hii_from_bits(bits_a_val, bits_b_val, hdiag1, jdiag, kdiag, nelec_a, nelec_b)


@njit(cache=True, inline="always")
def _build_diag_ST(occ_a, occ_b, norb, jdiag, kdiag, Sa_out, Sb_out, Ta_out, Tb_out):
    """Precompute per-I rank-update arrays Sa/Sb/Ta/Tb for O(1) diagonal updates."""
    for p in range(norb):
        sa = 0.0
        ta = 0.0
        for k in occ_a:
            sa += jdiag[p, k] - kdiag[p, k]
            ta += jdiag[p, k]
        Sa_out[p] = sa
        Ta_out[p] = ta

        sb = 0.0
        tb = 0.0
        for k in occ_b:
            sb += jdiag[p, k] - kdiag[p, k]
            tb += jdiag[p, k]
        Sb_out[p] = sb
        Tb_out[p] = tb


@njit(inline="always")
def _delta_single(i, a, hdiag1, jdiag, kdiag, Sx, Ty):
    """O(1) diagonal change for a single excitation i->a."""
    return (hdiag1[a] - hdiag1[i]) + (Sx[a] - Sx[i] - (jdiag[a, i] - kdiag[a, i])) + (Ty[a] - Ty[i])


@njit(inline="always")
def _delta_double_same_spin(i, j, a, b, hdiag1, jdiag, kdiag, Sx, Ty):
    """O(1) diagonal change for a same-spin double excitation i,j->a,b."""
    return (
        (hdiag1[a] + hdiag1[b] - hdiag1[i] - hdiag1[j])
        + (Sx[a] + Sx[b] - Sx[i] - Sx[j])
        - (jdiag[a, i] - kdiag[a, i])
        - (jdiag[a, j] - kdiag[a, j])
        - (jdiag[b, i] - kdiag[b, i])
        - (jdiag[b, j] - kdiag[b, j])
        + (jdiag[a, b] - kdiag[a, b])
        + (jdiag[i, j] - kdiag[i, j])
        + (Ty[a] + Ty[b] - Ty[i] - Ty[j])
    )


@njit(inline="always")
def _delta_double_cross(i, a, j, b, hdiag1, jdiag, kdiag, Sa, Sb, Ta, Tb):
    """O(1) diagonal change for a cross-spin double excitation."""
    d_aa = (hdiag1[a] - hdiag1[i]) + (Sa[a] - Sa[i] - (jdiag[a, i] - kdiag[a, i]))
    d_bb = (hdiag1[b] - hdiag1[j]) + (Sb[b] - Sb[j] - (jdiag[b, j] - kdiag[b, j]))
    d_cross = (
        (Ta[b] - Ta[j]) + (Tb[a] - Tb[i]) - jdiag[i, b] - jdiag[a, j] + jdiag[a, b] + jdiag[i, j]
    )
    return d_aa + d_bb + d_cross


@njit(cache=True, inline="always")
def _initial_cap(n_I, n_threads, min_cap=64):
    """Estimate per-thread flat-map seed capacity."""
    expected = 2 * n_I // n_threads if n_threads > 0 else 2 * n_I
    cap = expected + 1
    if cap < min_cap:
        cap = min_cap
    p = 1
    while p < cap:
        p <<= 1
    return p


@njit(cache=True, inline="always")
def _accum_single(bits, elem, addr_table, stride, fixed_offset, registry, cJ, col_local, t):
    """Accumulate contributions from same-spin excitations into col_local[t].

    Covers alpha singles/doubles (stride=nb, fixed_offset=ib) and beta
    singles/doubles (stride=1, fixed_offset=ia*nb).
    """
    for k in range(bits.shape[0]):
        a = _addr_get(addr_table, bits[k])
        if a < 0:
            continue
        I_idx = a * stride + fixed_offset
        if I_idx in registry:
            e = elem[k]
            for i_basis, cI in registry[I_idx].items():
                col_local[t, i_basis] += cI * cJ * e


@njit(cache=True, inline="always")
def _accum_cross(bits_a, bits_b, elem, addr_table_a, addr_table_b, nb, registry, cJ, col_local, t):
    """Accumulate contributions from alpha-beta double excitations into col_local[t]."""
    for k in range(bits_a.shape[0]):
        na = _addr_get(addr_table_a, bits_a[k])
        nbb = _addr_get(addr_table_b, bits_b[k])
        if na < 0 or nbb < 0:
            continue
        I_idx = na * nb + nbb
        if I_idx in registry:
            e = elem[k]
            for i_basis, cI in registry[I_idx].items():
                col_local[t, i_basis] += cI * cJ * e


@njit(parallel=True)
def add_basis_vector_fused(
    new_idx,  # int64[:] -- compact sorted index array
    new_val,  # float64[:] -- compact value array
    n_new,  # int64: index of this new basis vector
    registry,  # NumbaDict[int64, DictType(int64,float64)]
    norb,
    nb,
    nelec_a,
    nelec_b,
    bits_a,
    bits_b,
    addr_table_a,
    addr_table_b,
    h1e,
    eri,
    use_dense_hdiag,  # bool: True -> O(1) dense hdiag lookup, False -> inline hii_from_bits
    hdiag,  # (na*nb,) float64 when use_dense_hdiag else empty placeholder
    hdiag1,  # (norb,) float64 -- O(norb), not O(na*nb)
    jdiag,  # (norb, norb) float64 -- (pp|qq)
    kdiag,  # (norb, norb) float64 -- (pq|qp)
    col,  # preallocated, zeroed float64[n_new+1]
):
    """
    Add b_n into the registry and populate `col`.

    Two-pass design (serial register, parallel accumulate):

    Pass 1 (serial): insert every determinant of `new_vector` into
    `registry` so that within-vector self-connections are visible before
    any connections are evaluated.  Registry mutation is not thread-safe,
    so this pass stays serial.

    Pass 2 (parallel): snapshot `new_vector` into flat arrays, split into
    `n_threads` chunks, and let each thread accumulate into a private row
    of `col_local`.  The registry is read-only in this pass.  After the
    parallel region, the per-thread rows are summed into `col`.

    Parameters
    ----------
    new_idx : int64[:]
        Compact sorted determinant index array for basis vector b_n.
    new_val : float64[:]
        Compact coefficient array (parallel to new_idx).
    n_new : int64
        Column index of b_n in the subspace Hamiltonian.
    registry : NumbaDict[int64, NumbaDict[int64, float64]]
        Maps flat determinant index -> {basis_idx: coefficient} for all
        basis vectors registered so far.
    norb : int
        Number of spatial orbitals.
    nb : int
        Number of beta strings.
    nelec_a, nelec_b : int
        Number of alpha/beta electrons.
    bits_a, bits_b : int64[:]
        Bitmask lookup; bits_a[ia] is the bitmask for alpha string ia.
    addr_table_a, addr_table_b : int64[:] or NumbaDict[int64, int64]
        Address tables mapping bitmask -> string index (-1 if not in space).
    h1e : float64[:, :]
        One-electron integrals.
    eri : float64[:, :, :, :]
        Two-electron integrals (physicist notation).
    use_dense_hdiag : bool
        True  -> diagonal looked up from `hdiag` in O(1).
        False -> diagonal computed on-the-fly via hii_from_bits.
    hdiag : float64[:]
        Precomputed diagonal, shape (na*nb,).  Ignored when use_dense_hdiag
        is False (pass an empty placeholder).
    hdiag1 : float64[:]
        One-body diagonal integrals, shape (norb,).
    jdiag : float64[:, :]
        Coulomb diagonal (pp|qq), shape (norb, norb).
    kdiag : float64[:, :]
        Exchange diagonal (pq|qp), shape (norb, norb).
    col : float64[:]
        Pre-allocated, zeroed output of length n_new+1.  Accumulated
        in-place; caller owns allocation and zeroing.
    """
    # pass 1: register every determinant of new_vector so self-connections are captured.
    # Serial: registry mutation is not thread-safe.
    n_J = new_idx.shape[0]
    for _p in range(n_J):
        J_idx = new_idx[_p]
        cJ = new_val[_p]
        if J_idx in registry:
            registry[J_idx][n_new] = cJ
        else:
            d = NumbaDict.empty(key_type=types.int64, value_type=types.float64)
            d[n_new] = cJ
            registry[J_idx] = d

    # pass 2: generate connections; new_idx/new_val are already flat arrays -- no snapshot needed.
    # Chunk across threads; each thread accumulates into its own private row of col_local.
    J_idx_arr = new_idx
    cJ_arr = new_val

    # Splitting work accross threads
    n_threads = get_num_threads()
    ncol = col.shape[0]
    col_local = np.zeros((n_threads, ncol), dtype=np.float64)
    chunk = (n_J + n_threads - 1) // n_threads if n_threads > 0 else n_J

    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n_J:
            end = n_J

        occ_a_buf = np.empty(nelec_a, dtype=np.int64)
        virt_a_buf = np.empty(norb - nelec_a, dtype=np.int64)
        occ_b_buf = np.empty(nelec_b, dtype=np.int64)
        virt_b_buf = np.empty(norb - nelec_b, dtype=np.int64)

        for idx in range(start, end):  # loop over slice
            J_idx = J_idx_arr[idx]
            cJ = cJ_arr[idx]

            ia = J_idx // nb
            ib = J_idx % nb
            bits_a0 = bits_a[ia]
            bits_b0 = bits_b[ib]
            _decode_bits(bits_a0, norb, occ_a_buf, virt_a_buf)
            _decode_bits(bits_b0, norb, occ_b_buf, virt_b_buf)

            # Every basis vector that has non-zero coefficient at J_idx
            bucket = registry[J_idx]
            # Get diagonal term for J_idx
            e_diag = _ediag(
                use_dense_hdiag,
                J_idx,
                bits_a0,
                bits_b0,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                nelec_a,
                nelec_b,
            )

            # Add cicJ*H_JJ for every basis vector i
            for i_basis, ci in bucket.items():
                col_local[t, i_basis] += ci * cJ * e_diag

            # Single excitation (alpha)
            bits, elem, _, _ = singles_kernel(occ_a_buf, virt_a_buf, occ_b_buf, bits_a0, h1e, eri)
            _accum_single(bits, elem, addr_table_a, nb, ib, registry, cJ, col_local, t)

            # Single excitation (beta)
            bits, elem, _, _ = singles_kernel(occ_b_buf, virt_b_buf, occ_a_buf, bits_b0, h1e, eri)
            _accum_single(
                bits,
                elem,
                addr_table_b,
                np.int64(1),
                ia * nb,
                registry,
                cJ,
                col_local,
                t,
            )

            # double alpha-alpha
            bits, elem = doubles_same_spin_kernel(occ_a_buf, virt_a_buf, bits_a0, eri)
            _accum_single(bits, elem, addr_table_a, nb, ib, registry, cJ, col_local, t)

            # double beta-beta
            bits, elem = doubles_same_spin_kernel(occ_b_buf, virt_b_buf, bits_b0, eri)
            _accum_single(
                bits,
                elem,
                addr_table_b,
                np.int64(1),
                ia * nb,
                registry,
                cJ,
                col_local,
                t,
            )

            # cross alpha-beta double
            bits_a_new, bits_b_new, elem = doubles_cross_spin_kernel(
                occ_a_buf, virt_a_buf, occ_b_buf, virt_b_buf, bits_a0, bits_b0, eri
            )
            _accum_cross(
                bits_a_new,
                bits_b_new,
                elem,
                addr_table_a,
                addr_table_b,
                nb,
                registry,
                cJ,
                col_local,
                t,
            )

    # Serial reduction of per-thread partial columns
    for t in range(n_threads):
        for i in range(ncol):
            col[i] += col_local[t, i]


@njit(cache=True, inline="always")
def _decode_bits(bits0, norb, occ_out, virt_out):
    """Decode a bitmask into two sorted arrays: occupied and virtual
    (unoccupied) orbital indices (preallocated, correct size)."""
    oc = 0
    vc = 0
    for p in range(norb):
        if (bits0 >> np.int64(p)) & np.int64(1):
            occ_out[oc] = p
            oc += 1
        else:
            virt_out[vc] = p
            vc += 1


@njit(cache=True, inline="always")
def _accum_singles(
    bits,
    elem,
    i_arr,
    a_arr,
    addr_table,
    nb,
    fixed_row,
    fixed_col,
    cI,
    keys_t,
    vals_t,
    cnt_t,
    use_dense_hdiag,
    hdiag,
    hdiag1,
    jdiag,
    kdiag,
    E_I,
    Sx,
    Ty,
    E,
    level_shift,
    trunc_eps,
    delta,
    is_alpha,
):
    """Accumulate single-excitation contributions into the thread-private flat map.

    Returns (keys_t, vals_t) — caller must reassign (ht_add_grow may reallocate).
    """
    for k in range(bits.shape[0]):
        new_addr = _addr_get(addr_table, bits[k])
        if new_addr < 0:
            continue
        if is_alpha:
            new_flat = new_addr * nb + fixed_col
        else:
            new_flat = fixed_row * nb + new_addr
        cIp = elem[k] * cI
        if use_dense_hdiag:
            e_diag = hdiag[new_flat]
        else:
            e_diag = E_I + _delta_single(i_arr[k], a_arr[k], hdiag1, jdiag, kdiag, Sx, Ty)
        denom = e_diag - (E - level_shift)
        if abs(denom) < trunc_eps:
            denom = trunc_eps if denom >= 0.0 else -trunc_eps
        cIp /= denom
        if abs(cIp) > delta:
            keys_t, vals_t = ht_add_grow(keys_t, vals_t, cnt_t, new_flat, cIp)
    return keys_t, vals_t


@njit(cache=True, inline="always")
def _accum_same_spin_doubles(
    occ_buf,
    nelec_exc,
    bits0,
    fixed_addr,
    is_alpha,
    ia,
    same_a,
    same_b,
    same_v,
    addr_table,
    use_dense_hdiag,
    hdiag,
    hdiag1,
    jdiag,
    kdiag,
    E_I,
    Sx,
    Ty,
    nb,
    cI,
    E,
    delta,
    out_idx,
    out_val,
    keys_t,
    vals_t,
    cnt_t,
):
    """Accumulate same-spin double-excitation contributions into the thread-private flat map.

    Returns (keys_t, vals_t) — caller must reassign.
    """
    for oi in range(nelec_exc):
        i = occ_buf[oi]
        for oj in range(oi + 1, nelec_exc):
            j = occ_buf[oj]
            n_out = pcr_doubles_same_spin_kernel(
                bits0,
                fixed_addr,
                nb,
                is_alpha,
                ia,
                same_a[i, j],
                same_b[i, j],
                same_v[i, j],
                addr_table,
                use_dense_hdiag,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                E_I,
                Sx,
                Ty,
                cI,
                E,
                delta,
                i,
                j,
                out_idx,
                out_val,
            )
            for k in range(n_out):
                keys_t, vals_t = ht_add_grow(keys_t, vals_t, cnt_t, out_idx[k], out_val[k])
    return keys_t, vals_t


@njit(parallel=True)
def calc_trunc_pcr_fused(
    psi_keys,  # int64[:] -- flat-map keys array
    psi_vals,  # float64[:] -- flat-map values array
    norb,
    nb,
    nelec_a,
    nelec_b,
    bits_a,
    bits_b,
    addr_table_a,
    addr_table_b,
    h1e,
    eri,
    use_dense_hdiag,  # bool: True -> O(1) dense hdiag lookup, False -> inline hii_from_bits
    hdiag,  # (na*nb,) float64 when use_dense_hdiag else empty placeholder
    hdiag1,  # (norb,) float64 -- O(norb), not O(na*nb)
    jdiag,  # (norb, norb) float64 -- (pp|qq)
    kdiag,  # (norb, norb) float64 -- (pq|qp)
    same_a,
    same_b,
    same_v,
    cross_a,
    cross_b,
    cross_v,
    E,
    delta,
    level_shift=1e-3,
    trunc_eps=1e-8,
):
    """
    Compute the truncated PCR correction vector r for a single Davidson eigenpair (E, psi).

    Per-thread accumulator is a GROWABLE flat map (keys_t, vals_t, cnt_t); the final
    merge uses the exact live counts (cnt_t[0]) to size a FIXED output map and uses
    ht_add. Returns (keys, vals) hash table pair.
    """
    # Snapshot the EMPTY-holed flat map into dense index/value arrays for prange.
    n_I = np.int64(0)
    for _i in range(psi_keys.shape[0]):
        if psi_keys[_i] != EMPTY:
            n_I += np.int64(1)
    I_idx_arr = np.empty(n_I, dtype=np.int64)
    cI_arr = np.empty(n_I, dtype=np.float64)
    _p = np.int64(0)
    for _i in range(psi_keys.shape[0]):
        if psi_keys[_i] != EMPTY:
            I_idx_arr[_p] = psi_keys[_i]
            cI_arr[_p] = psi_vals[_i]
            _p += np.int64(1)

    n_threads = get_num_threads()
    r_locals_keys = NumbaList()
    r_locals_vals = NumbaList()
    r_locals_cnt = NumbaList()
    n_cap = _initial_cap(n_I, n_threads)
    for _t in range(n_threads):
        k0, v0, c0 = ht_new_growable(n_cap)
        r_locals_keys.append(k0)
        r_locals_vals.append(v0)
        r_locals_cnt.append(c0)

    max_ab = norb * norb
    chunk = (n_I + n_threads - 1) // n_threads if n_threads > 0 else n_I

    for t in prange(n_threads):
        start = t * chunk
        end = start + chunk
        if end > n_I:
            end = n_I

        keys_t = r_locals_keys[np.int64(t)]
        vals_t = r_locals_vals[np.int64(t)]
        cnt_t = r_locals_cnt[np.int64(t)]

        occ_a_buf = np.empty(nelec_a, dtype=np.int64)
        virt_a_buf = np.empty(norb - nelec_a, dtype=np.int64)
        occ_b_buf = np.empty(nelec_b, dtype=np.int64)
        virt_b_buf = np.empty(norb - nelec_b, dtype=np.int64)
        Sa_buf = np.empty(norb, dtype=np.float64)
        Sb_buf = np.empty(norb, dtype=np.float64)
        Ta_buf = np.empty(norb, dtype=np.float64)
        Tb_buf = np.empty(norb, dtype=np.float64)
        out_idx = np.empty(max_ab, dtype=np.int64)
        out_val = np.empty(max_ab, dtype=np.float64)

        for idx in range(start, end):
            I_idx = I_idx_arr[idx]
            cI = cI_arr[idx]

            # seed r <- psi (Fig. 4 line 2)
            keys_t, vals_t = ht_add_grow(keys_t, vals_t, cnt_t, I_idx, cI)

            ia = I_idx // nb
            ib = I_idx % nb
            bits_a0 = bits_a[ia]
            bits_b0 = bits_b[ib]

            _decode_bits(bits_a0, norb, occ_a_buf, virt_a_buf)
            _decode_bits(bits_b0, norb, occ_b_buf, virt_b_buf)

            if use_dense_hdiag:
                E_I = hdiag[I_idx]
            else:
                E_I = hii_from_bits(bits_a0, bits_b0, hdiag1, jdiag, kdiag, nelec_a, nelec_b)
            _build_diag_ST(occ_a_buf, occ_b_buf, norb, jdiag, kdiag, Sa_buf, Sb_buf, Ta_buf, Tb_buf)

            # ---- singles (alpha) ----
            bits, elem, i_arr, a_arr = singles_kernel(
                occ_a_buf, virt_a_buf, occ_b_buf, bits_a0, h1e, eri
            )
            keys_t, vals_t = _accum_singles(
                bits,
                elem,
                i_arr,
                a_arr,
                addr_table_a,
                nb,
                ia,
                ib,
                cI,
                keys_t,
                vals_t,
                cnt_t,
                use_dense_hdiag,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                E_I,
                Sa_buf,
                Tb_buf,
                E,
                level_shift,
                trunc_eps,
                delta,
                True,
            )

            # ---- singles (beta) ----
            bits, elem, i_arr, a_arr = singles_kernel(
                occ_b_buf, virt_b_buf, occ_a_buf, bits_b0, h1e, eri
            )
            keys_t, vals_t = _accum_singles(
                bits,
                elem,
                i_arr,
                a_arr,
                addr_table_b,
                nb,
                ia,
                ib,
                cI,
                keys_t,
                vals_t,
                cnt_t,
                use_dense_hdiag,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                E_I,
                Sb_buf,
                Ta_buf,
                E,
                level_shift,
                trunc_eps,
                delta,
                False,
            )

            # ---- same-spin alpha-alpha ----
            keys_t, vals_t = _accum_same_spin_doubles(
                occ_a_buf,
                nelec_a,
                bits_a0,
                ib,
                np.int64(1),
                ia,
                same_a,
                same_b,
                same_v,
                addr_table_a,
                use_dense_hdiag,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                E_I,
                Sa_buf,
                Tb_buf,
                nb,
                cI,
                E,
                delta,
                out_idx,
                out_val,
                keys_t,
                vals_t,
                cnt_t,
            )

            # ---- same-spin beta-beta ----
            keys_t, vals_t = _accum_same_spin_doubles(
                occ_b_buf,
                nelec_b,
                bits_b0,
                ia,
                np.int64(0),
                ia,
                same_a,
                same_b,
                same_v,
                addr_table_b,
                use_dense_hdiag,
                hdiag,
                hdiag1,
                jdiag,
                kdiag,
                E_I,
                Sb_buf,
                Ta_buf,
                nb,
                cI,
                E,
                delta,
                out_idx,
                out_val,
                keys_t,
                vals_t,
                cnt_t,
            )

            # ---- cross-spin ----
            for oi in range(nelec_a):
                i = occ_a_buf[oi]
                for oj in range(nelec_b):
                    j = occ_b_buf[oj]
                    n_out = pcr_doubles_cross_spin_kernel(
                        bits_a0,
                        bits_b0,
                        nb,
                        cross_a[i, j],
                        cross_b[i, j],
                        cross_v[i, j],
                        addr_table_a,
                        addr_table_b,
                        use_dense_hdiag,
                        hdiag,
                        hdiag1,
                        jdiag,
                        kdiag,
                        nelec_a,
                        nelec_b,
                        cI,
                        E,
                        delta,
                        i,
                        j,
                        out_idx,
                        out_val,
                    )
                    for k in range(n_out):
                        keys_t, vals_t = ht_add_grow(keys_t, vals_t, cnt_t, out_idx[k], out_val[k])

        r_locals_keys[t] = keys_t
        r_locals_vals[t] = vals_t

    # ---- serial merge: size output exactly from per-thread counts, then use ht_add ----
    total = np.int64(0)
    for t in range(n_threads):
        total += r_locals_cnt[t][0]

    out_keys, out_vals = ht_new(total)
    for t in range(n_threads):
        kt = r_locals_keys[t]
        vt = r_locals_vals[t]
        for i in range(kt.shape[0]):
            k = kt[i]
            if k != EMPTY:
                ht_add(out_keys, out_vals, k, vt[i])

    return out_keys, out_vals


@njit(cache=True)
def singles_kernel(
    occ: np.ndarray,
    virt: np.ndarray,
    occ_other: np.ndarray,
    bits0: int,
    h1e: np.ndarray,
    eri: np.ndarray,
):
    """
    All single excitations i -> a within one spin block, holding the other spin fixed.
    Returns (new_bits, elem, i_arr, a_arr) arrays of length len(occ)*len(virt).
    i_arr/a_arr track source/destination orbitals for O(1) diagonal updates.
    """
    n_occ = occ.shape[0]
    n_virt = virt.shape[0]
    n = n_occ * n_virt
    new_bits = np.empty(n, dtype=np.int64)
    elem = np.empty(n, dtype=np.float64)
    i_arr = np.empty(n, dtype=np.int64)
    a_arr = np.empty(n, dtype=np.int64)

    idx = 0
    for oi in range(n_occ):
        i = occ[oi]
        for vi in range(n_virt):
            a = virt[vi]
            lo = i if i < a else a
            hi = a if i < a else i
            p = 0
            for x in occ:
                if lo < x < hi:
                    p += 1
            sign = 1.0 if (p % 2 == 0) else -1.0

            e = h1e[a, i]
            for k in occ:
                e += eri[a, i, k, k] - eri[a, k, k, i]
            for k in occ_other:
                e += eri[a, i, k, k]

            new_bits[idx] = (bits0 & ~(np.int64(1) << i)) | (np.int64(1) << a)
            elem[idx] = e * sign
            i_arr[idx] = i
            a_arr[idx] = a
            idx += 1

    return new_bits, elem, i_arr, a_arr


@njit(cache=True)
def doubles_same_spin_kernel(occ: np.ndarray, virt: np.ndarray, bits0: int, eri: np.ndarray):
    """
    All same-spin double excitations i<j -> a<b within one spin block.
    Returns (new_bits, elem) arrays of length C(n_occ,2)*C(n_virt,2).

    Sign is computed by sequential single-excitation application
    (i->a first, then j->b on the resulting intermediate determinant)
    -- exact, not approximate, since it mirrors direct operator
    application; see subspace_hamiltonian.py's module docstring.
    """
    n_occ = occ.shape[0]
    n_virt = virt.shape[0]
    n_occ_pairs = n_occ * (n_occ - 1) // 2
    n_virt_pairs = n_virt * (n_virt - 1) // 2
    n = n_occ_pairs * n_virt_pairs
    new_bits = np.empty(n, dtype=np.int64)
    elem = np.empty(n, dtype=np.float64)

    idx = 0
    for oi in range(n_occ):
        i = occ[oi]
        for oj in range(oi + 1, n_occ):
            j = occ[oj]
            for vi in range(n_virt):
                a = virt[vi]
                # sign1: excite i -> a from the original occ list
                lo1 = i if i < a else a
                hi1 = a if i < a else i
                p1 = 0
                for x in occ:
                    if lo1 < x < hi1:
                        p1 += 1
                s1 = 1.0 if (p1 % 2 == 0) else -1.0

                for vj in range(vi + 1, n_virt):
                    b = virt[vj]
                    # sign2: excite j -> b from the intermediate
                    # (occ - {i} + {a}); equivalent to counting within
                    # occ (excluding i) and adding 1 if a itself lies
                    # strictly between j and b.
                    lo2 = j if j < b else b
                    hi2 = b if j < b else j
                    p2 = 0
                    for x in occ:
                        if x == i:
                            continue
                        if lo2 < x < hi2:
                            p2 += 1
                    if lo2 < a < hi2:
                        p2 += 1
                    s2 = 1.0 if (p2 % 2 == 0) else -1.0

                    sign = s1 * s2
                    e = sign * (eri[a, i, b, j] - eri[a, j, b, i])

                    new_bits[idx] = (
                        (bits0 & ~(np.int64(1) << i) & ~(np.int64(1) << j))
                        | (np.int64(1) << a)
                        | (np.int64(1) << b)
                    )
                    elem[idx] = e
                    idx += 1

    return new_bits, elem


@njit(cache=True)
def doubles_cross_spin_kernel(
    occ_a: np.ndarray,
    virt_a: np.ndarray,
    occ_b: np.ndarray,
    virt_b: np.ndarray,
    bits_a0: int,
    bits_b0: int,
    eri: np.ndarray,
):
    """
    All cross-spin (one alpha + one beta) double excitations
    i_alpha -> a_alpha, j_beta -> b_beta. Returns (new_bits_a,
    new_bits_b, elem) arrays of length
    len(occ_a)*len(virt_a)*len(occ_b)*len(virt_b).

    Alpha and beta excitations are independent fermion sectors under
    the alpha-block-then-beta-block ordering convention, so the total
    sign simply factorizes as sign_alpha * sign_beta -- no chaining/
    intermediate-state coupling is needed here (unlike the same-spin
    case above).
    """
    n_oa = occ_a.shape[0]
    n_va = virt_a.shape[0]
    n_ob = occ_b.shape[0]
    n_vb = virt_b.shape[0]
    n = n_oa * n_va * n_ob * n_vb
    new_bits_a = np.empty(n, dtype=np.int64)
    new_bits_b = np.empty(n, dtype=np.int64)
    elem = np.empty(n, dtype=np.float64)

    idx = 0
    for oi in range(n_oa):
        i = occ_a[oi]
        for vi in range(n_va):
            a = virt_a[vi]
            lo1 = i if i < a else a
            hi1 = a if i < a else i
            p1 = 0
            for x in occ_a:
                if lo1 < x < hi1:
                    p1 += 1
            s1 = 1.0 if (p1 % 2 == 0) else -1.0
            nba = (bits_a0 & ~(np.int64(1) << i)) | (np.int64(1) << a)

            for oj in range(n_ob):
                j = occ_b[oj]
                for vj in range(n_vb):
                    b = virt_b[vj]
                    lo2 = j if j < b else b
                    hi2 = b if j < b else j
                    p2 = 0
                    for x in occ_b:
                        if lo2 < x < hi2:
                            p2 += 1
                    s2 = 1.0 if (p2 % 2 == 0) else -1.0

                    sign = s1 * s2
                    e = sign * eri[a, i, b, j]
                    nbb = (bits_b0 & ~(np.int64(1) << j)) | (np.int64(1) << b)

                    new_bits_a[idx] = nba
                    new_bits_b[idx] = nbb
                    elem[idx] = e
                    idx += 1

    return new_bits_a, new_bits_b, elem


@njit(cache=True)
def _popcount(x: np.int64) -> np.int64:
    """Count set bits in x (Kernighan's method)."""
    c = np.int64(0)
    while x:
        x &= x - np.int64(1)
        c += np.int64(1)
    return c


@njit(cache=True)
def pcr_doubles_same_spin_kernel(
    bits0: np.int64,
    ib_fixed: np.int64,
    nb: np.int64,
    use_alpha: np.int64,
    ia_fixed: np.int64,
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    val_arr: np.ndarray,
    addr_table: np.ndarray,
    use_dense_hdiag: np.bool_,
    hdiag: np.ndarray,
    hdiag1: np.ndarray,
    jdiag: np.ndarray,
    kdiag: np.ndarray,
    E_I: np.float64,
    Sx: np.ndarray,
    Ty: np.ndarray,
    cI: np.float64,
    E: np.float64,
    delta: np.float64,
    i: np.int64,
    j: np.int64,
    out_idx: np.ndarray,
    out_val: np.ndarray,
) -> np.int64:
    """
    Thresholded same-spin double-excitation contribution for one (i,j) pair.

    Iterates the pre-sorted (a,b) table; occupied destinations are skipped (don't
    short-circuit); raw magnitude against delta triggers early exit. Diagonal uses
    E_I + _delta_double_same_spin (O(1)) when use_dense_hdiag is False, else hdiag[new_flat].
    Writes accepted entries into out_idx/out_val; returns number written.
    """
    n_out = np.int64(0)
    floor = np.float64(1e-10)

    n_ab = a_arr.shape[0]
    for k in range(n_ab):
        a = a_arr[k]
        b = b_arr[k]
        if (bits0 >> a) & np.int64(1):
            continue
        if (bits0 >> b) & np.int64(1):
            continue

        raw_c = val_arr[k] * cI
        if abs(raw_c) < delta:
            break

        # sign for i->a
        lo1 = i if i < a else a
        hi1 = a if i < a else i
        mask1 = ((np.int64(1) << hi1) - np.int64(1)) & ~((np.int64(1) << (lo1 + 1)) - np.int64(1))
        s1 = np.int64(1) if (_popcount(bits0 & mask1) % 2 == 0) else np.int64(-1)

        # intermediate bits after i->a
        bits_mid = (bits0 & ~(np.int64(1) << i)) | (np.int64(1) << a)

        # sign for j->b on the intermediate
        lo2 = j if j < b else b
        hi2 = b if j < b else j
        mask2 = ((np.int64(1) << hi2) - np.int64(1)) & ~((np.int64(1) << (lo2 + 1)) - np.int64(1))
        s2 = np.int64(1) if (_popcount(bits_mid & mask2) % 2 == 0) else np.int64(-1)

        cIp = np.float64(s1 * s2) * val_arr[k] * cI

        new_bits = (
            (bits0 & ~(np.int64(1) << i) & ~(np.int64(1) << j))
            | (np.int64(1) << a)
            | (np.int64(1) << b)
        )
        new_addr = _addr_get(addr_table, new_bits)
        if new_addr < np.int64(0):
            continue

        if use_alpha:
            new_flat = new_addr * nb + ib_fixed
        else:
            new_flat = ia_fixed * nb + new_addr

        if use_dense_hdiag:
            e_diag = hdiag[new_flat]
        else:
            e_diag = E_I + _delta_double_same_spin(i, j, a, b, hdiag1, jdiag, kdiag, Sx, Ty)

        denom = e_diag - E
        if abs(denom) < floor:
            denom = floor if denom >= np.float64(0) else -floor
        cIp /= denom

        if abs(cIp) > delta:
            out_idx[n_out] = new_flat
            out_val[n_out] = cIp
            n_out += np.int64(1)

    return n_out


@njit(cache=True)
def pcr_doubles_cross_spin_kernel(
    bits_a0: np.int64,
    bits_b0: np.int64,
    nb: np.int64,
    a_arr: np.ndarray,  # sorted-by-|val| alpha destination
    b_arr: np.ndarray,  # beta destination
    val_arr: np.ndarray,
    addr_table_a: np.ndarray,
    addr_table_b: np.ndarray,
    use_dense_hdiag: np.bool_,
    hdiag: np.ndarray,
    hdiag1: np.ndarray,
    jdiag: np.ndarray,
    kdiag: np.ndarray,
    nelec_a: np.int64,
    nelec_b: np.int64,
    cI: np.float64,
    E: np.float64,
    delta: np.float64,
    i: np.int64,  # source alpha orbital
    j: np.int64,  # source beta orbital
    out_idx: np.ndarray,
    out_val: np.ndarray,
) -> np.int64:
    """
    Thresholded cross-spin double-excitation contribution for one (i,j)
    pair. Alpha and beta signs are independent. Same occupancy-then-
    magnitude check order as pcr_doubles_same_spin_kernel above.
    """
    n_out = np.int64(0)
    floor = np.float64(1e-10)
    n_ab = a_arr.shape[0]

    for k in range(n_ab):
        a = a_arr[k]
        b = b_arr[k]
        # a must be alpha-virtual, b must be beta-virtual -- skip,
        # don't short-circuit.
        if (bits_a0 >> a) & np.int64(1):
            continue
        if (bits_b0 >> b) & np.int64(1):
            continue

        raw_c = val_arr[k] * cI
        if abs(raw_c) < delta:
            break

        # alpha sign: i->a
        lo1 = i if i < a else a
        hi1 = a if i < a else i
        mask1 = ((np.int64(1) << hi1) - np.int64(1)) & ~((np.int64(1) << (lo1 + 1)) - np.int64(1))
        s1 = np.int64(1) if (_popcount(bits_a0 & mask1) % 2 == 0) else np.int64(-1)

        # beta sign: j->b
        lo2 = j if j < b else b
        hi2 = b if j < b else j
        mask2 = ((np.int64(1) << hi2) - np.int64(1)) & ~((np.int64(1) << (lo2 + 1)) - np.int64(1))
        s2 = np.int64(1) if (_popcount(bits_b0 & mask2) % 2 == 0) else np.int64(-1)

        cIp = np.float64(s1 * s2) * val_arr[k] * cI

        new_bits_a = (bits_a0 & ~(np.int64(1) << i)) | (np.int64(1) << a)
        new_bits_b = (bits_b0 & ~(np.int64(1) << j)) | (np.int64(1) << b)
        new_ia = _addr_get(addr_table_a, new_bits_a)
        new_ib = _addr_get(addr_table_b, new_bits_b)
        if new_ia < np.int64(0) or new_ib < np.int64(0):
            continue

        new_flat = new_ia * nb + new_ib
        e_diag = _ediag(
            use_dense_hdiag,
            new_flat,
            new_bits_a,
            new_bits_b,
            hdiag,
            hdiag1,
            jdiag,
            kdiag,
            nelec_a,
            nelec_b,
        )
        denom = e_diag - E
        if abs(denom) < floor:
            denom = floor if denom >= np.float64(0) else -floor
        cIp /= denom

        if abs(cIp) > delta:
            out_idx[n_out] = new_flat
            out_val[n_out] = cIp
            n_out += np.int64(1)

    return n_out


@njit(cache=True)
def build_addr_dict(bits: np.ndarray) -> NumbaDict:
    """Build a bitmask->address NumbaDict from a sorted int64 bitstring array, in compiled code."""
    d = NumbaDict.empty(key_type=types.int64, value_type=types.int64)
    for i in range(bits.shape[0]):
        d[bits[i]] = np.int64(i)
    return d


def warmup(norb_hint: int = 4):
    """
    Trigger numba's JIT compilation once, eagerly, with tiny dummy
    inputs, so the (one-time, ~1s) compilation cost happens at a
    predictable point (e.g. Hamiltonian/builder construction) rather
    than silently inside the first real Davidson iteration.
    """
    # Dummy data is written for norb=4; enforce a minimum so that orbital
    # indices in occ/virt/a_arr/b_arr never exceed norb_hint.
    norb_hint = max(norb_hint, 4)
    occ = np.array([0, 1], dtype=np.int64)
    virt = np.array([2, 3], dtype=np.int64)
    h1e = np.zeros((norb_hint, norb_hint))
    eri = np.zeros((norb_hint, norb_hint, norb_hint, norb_hint))
    singles_kernel(occ, virt, occ, 0, h1e, eri)
    doubles_same_spin_kernel(occ, virt, 0, eri)
    doubles_cross_spin_kernel(occ, virt, occ, virt, 0, 0, eri)
    # PCR kernels -- exercise both the dense-hdiag and inline-hdiag paths
    addr_table = np.arange(1 << norb_hint, dtype=np.int64)
    hdiag1 = np.zeros(norb_hint, dtype=np.float64)
    jdiag = np.zeros((norb_hint, norb_hint), dtype=np.float64)
    kdiag = np.zeros((norb_hint, norb_hint), dtype=np.float64)
    dense_hdiag = np.zeros(1 << (2 * norb_hint), dtype=np.float64)
    empty_dense = np.zeros(0, dtype=np.float64)
    empty1 = np.zeros(0, dtype=np.float64)
    empty2 = np.zeros((0, 0), dtype=np.float64)
    a_arr = np.array([2, 3], dtype=np.int64)
    b_arr = np.array([3, 2], dtype=np.int64)
    val_arr = np.array([0.1, 0.05], dtype=np.float64)
    out_idx = np.empty(4, dtype=np.int64)
    out_val = np.empty(4, dtype=np.float64)
    for use_dense, hdiag_arg, h1_arg, j_arg, k_arg in (
        (np.bool_(True), dense_hdiag, empty1, empty2, empty2),
        (np.bool_(False), empty_dense, hdiag1, jdiag, kdiag),
    ):
        _Sx = np.zeros(norb_hint, dtype=np.float64)
        _Ty = np.zeros(norb_hint, dtype=np.float64)
        pcr_doubles_same_spin_kernel(
            np.int64(3),  # bits0
            np.int64(0),  # ib_fixed
            np.int64(norb_hint),  # nb
            np.int64(1),  # use_alpha
            np.int64(0),  # ia_fixed
            a_arr,  # a_arr
            b_arr,  # b_arr
            val_arr,  # val_arr
            addr_table,  # addr_table
            use_dense,  # use_dense_hdiag
            hdiag_arg,  # hdiag
            h1_arg,  # hdiag1
            j_arg,  # jdiag
            k_arg,  # kdiag
            np.float64(0.0),  # E_I
            _Sx,  # Sx
            _Ty,  # Ty
            np.float64(1.0),  # cI
            np.float64(-1.0),  # E
            np.float64(1e-12),  # delta
            np.int64(0),  # i
            np.int64(1),  # j
            out_idx,  # out_idx
            out_val,  # out_val
        )
        pcr_doubles_cross_spin_kernel(
            np.int64(3),  # bits_a0
            np.int64(3),  # bits_b0
            np.int64(norb_hint),  # nb
            a_arr,  # a_arr
            b_arr,  # b_arr
            val_arr,  # val_arr
            addr_table,  # addr_table_a
            addr_table,  # addr_table_b
            use_dense,  # use_dense_hdiag
            hdiag_arg,  # hdiag
            h1_arg,  # hdiag1
            j_arg,  # jdiag
            k_arg,  # kdiag
            np.int64(2),  # nelec_a
            np.int64(2),  # nelec_b
            np.float64(1.0),  # cI
            np.float64(-1.0),  # E
            np.float64(1e-12),  # delta
            np.int64(0),  # i
            np.int64(0),  # j
            out_idx,  # out_idx
            out_val,  # out_val
        )

    # excitation generators (used by _lowest_diagonal_seed_vectors_sd_otf)
    from sci_solvers.energy_solvers.truncated_davidson_solver.numba_diag import (
        _count_singles_doubles,
        excite_single_double_nb,
        excite_single_nb,
    )

    ref = np.int64(3)  # |0011> -- 2 electrons in 4 orbitals
    sd_buf = np.empty(_count_singles_doubles(2, norb_hint - 2), dtype=np.int64)
    excite_single_double_nb(ref, norb_hint, sd_buf)
    s_buf = np.empty(1 + 2 * (norb_hint - 2), dtype=np.int64)
    excite_single_nb(ref, norb_hint, s_buf)
