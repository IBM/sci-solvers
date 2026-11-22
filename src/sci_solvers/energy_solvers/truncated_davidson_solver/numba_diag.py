__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
from numba import int64, njit, prange, types
from numba.typed import Dict
from pyscf.fci import cistring


# ----------------------------------------------------------------------
# Precompute the O(norb^2)/O(norb^4) "diagonal" integral arrays.
# ----------------------------------------------------------------------
def precompute_diag_ints(h1e, eri_full):
    """Extract the three O(norb²) diagonal integral arrays needed for H_ii.

    Returns:
        hdiag1 : (norb,)     h_pp = diagonal of h1e
        jdiag  : (norb,norb) J_pq = (pp|qq) Coulomb integrals
        kdiag  : (norb,norb) K_pq = (pq|qp) exchange integrals
    """
    jdiag = np.ascontiguousarray(np.einsum("iijj->ij", eri_full))
    kdiag = np.ascontiguousarray(np.einsum("ijji->ij", eri_full))
    hdiag1 = np.ascontiguousarray(np.diagonal(h1e).copy())
    return hdiag1, jdiag, kdiag


@njit(cache=True, parallel=True, fastmath=True)
def hii_batch(str_a, str_b, hdiag1, jdiag, kdiag, neleca, nelecb, out):
    """Compute H_ii in parallel for a batch of determinants given as bitmask arrays.

    Thin parallel wrapper around `hii_from_bits`; see that docstring for the
    formula and argument details.

    Args:
        str_a, str_b : (n,) int64 alpha/beta occupation bitmasks
        out          : (n,) float64 output array, written in place
    """
    n = str_a.shape[0]
    for k in prange(n):
        out[k] = hii_from_bits(str_a[k], str_b[k], hdiag1, jdiag, kdiag, neleca, nelecb)


@njit(cache=True, inline="always", fastmath=True)
def hii_from_bits(bits_a, bits_b, hdiag1, jdiag, kdiag, neleca, nelecb):
    """Compute H_ii from a single alpha/beta bitmask pair, inlined at the call site.

    H_ii = Σ_p h_pp  +  Σ_{i<j,αα} (J_ij − K_ij)
                      +  Σ_{i<j,ββ} (J_ij − K_ij)
                      +  Σ_{i∈α,j∈β} J_ij

    Args:
        bits_a, bits_b : int64 alpha/beta occupation bitmasks
        hdiag1         : (norb,)     diagonal one-electron integrals h_pp
        jdiag          : (norb,norb) Coulomb integrals J_pq = (pp|qq)
        kdiag          : (norb,norb) exchange integrals K_pq = (pq|qp)
        neleca/b       : number of alpha/beta electrons

    Returns:
        float64 H_ii for the given determinant
    """
    occ_a = np.empty(neleca, dtype=np.int64)
    occ_b = np.empty(nelecb, dtype=np.int64)

    # Extract occupied orbital indices from the bit-packed determinant strings.
    # Each iteration isolates the lowest set bit via s & (-s), then computes its position as floor(log2(low)).
    # XOR clears that bit before the next pass.
    s = bits_a
    for i in range(neleca):
        low = s & (-s)
        occ_a[i] = int64(np.log2(low))
        s ^= low
    s = bits_b
    for i in range(nelecb):
        low = s & (-s)
        occ_b[i] = int64(np.log2(low))
        s ^= low

    e = 0.0
    for i in range(neleca):
        e += hdiag1[occ_a[i]]
    for i in range(nelecb):
        e += hdiag1[occ_b[i]]

    # same-spin: strict i<j, no 0.5 prefactor, no self-pairs
    for i in range(neleca):
        pi = occ_a[i]
        for j in range(i + 1, neleca):
            pj = occ_a[j]
            e += jdiag[pi, pj] - kdiag[pi, pj]
    for i in range(nelecb):
        pi = occ_b[i]
        for j in range(i + 1, nelecb):
            pj = occ_b[j]
            e += jdiag[pi, pj] - kdiag[pi, pj]

    # opposite-spin: already minimal, no triangular structure available
    for i in range(neleca):
        pi = occ_a[i]
        for j in range(nelecb):
            pj = occ_b[j]
            e += jdiag[pi, pj]

    return e


# ----------------------------------------------------------------------
# Numba-typed dict cache helpers: keyed by flat CI index -> H_ii value.
# ----------------------------------------------------------------------
@njit(cache=True)
def _find_missing(idx, cache):
    """Boolean mask of which idx entries are NOT yet in the cache."""
    mask = np.empty(idx.shape[0], dtype=np.bool_)
    for i in range(idx.shape[0]):
        mask[i] = idx[i] not in cache
    return mask


@njit(cache=True)
def _insert_into_cache(idx, vals, cache):
    """Write idx[i] -> vals[i] into cache for all i."""
    for i in range(idx.shape[0]):
        cache[idx[i]] = vals[i]


@njit(cache=True)
def _gather_from_cache(idx, cache, out):
    """Read out[i] = cache[idx[i]] for all i; all keys must already be present."""
    for i in range(idx.shape[0]):
        out[i] = cache[idx[i]]


# ----------------------------------------------------------------------
# Main entry point: cache-aware, only computes H_ii for indices not
# already known. Pass the SAME cache dict in across iterations.
# ----------------------------------------------------------------------
def hii_on_the_fly_scalar(idx, nb, nelec, cache, strings_a, strings_b, diag_ints):
    """Compute or retrieve a single H_ii value without any array allocation.

    idx : int flat CI-vector index (row-major: i = ia*nb + ib)
    nb  : number of beta strings (precomputed by the caller)
    """
    if idx in cache:
        return cache[idx]
    neleca, nelecb = nelec
    ia, ib = idx // nb, idx % nb
    hdiag1, jdiag, kdiag = diag_ints
    val = hii_from_bits(strings_a[ia], strings_b[ib], hdiag1, jdiag, kdiag, neleca, nelecb)
    cache[idx] = val
    return val


def hii_on_the_fly(
    h1e,
    eri_full,
    norb,
    nelec,
    idx,
    cache,
    strings_a=None,
    strings_b=None,
    diag_ints=None,
    nb=None,
):
    """
    idx    : 1-D array of flat CI-vector indices
    cache  : numba.typed.Dict (int64 -> float64).
    strings_a, strings_b, diag_ints : optional precomputed tables.
    nb     : number of beta strings; derived from strings_b or nelec if not given.
    """
    neleca, nelecb = nelec
    if nb is None:
        nb = len(strings_b) if strings_b is not None else cistring.num_strings(norb, nelecb)

    missing_mask = _find_missing(idx, cache)
    n_missing = missing_mask.sum()

    if n_missing > 0:
        missing_idx = idx[missing_mask]
        ia, ib = missing_idx // nb, missing_idx % nb

        if strings_a is None or strings_b is None:
            strings_a = np.asarray(
                cistring.gen_strings4orblist(range(norb), neleca), dtype=np.int64
            )
            strings_b = np.asarray(
                cistring.gen_strings4orblist(range(norb), nelecb), dtype=np.int64
            )

        if diag_ints is None:
            diag_ints = precompute_diag_ints(h1e, eri_full)
        hdiag1, jdiag, kdiag = diag_ints

        str_a = strings_a[ia]
        str_b = strings_b[ib]

        new_vals = np.empty(missing_idx.shape[0], dtype=np.float64)
        hii_batch(str_a, str_b, hdiag1, jdiag, kdiag, neleca, nelecb, new_vals)

        _insert_into_cache(missing_idx, new_vals, cache)

    out = np.empty(idx.shape[0], dtype=np.float64)
    _gather_from_cache(idx, cache, out)
    return out, n_missing


class HDiagCache:
    """Convenience wrapper: builds string tables + J/K integrals once,
    holds a persistent numba-dict cache of already-computed H_ii values,
    reused across repeated calls (e.g. Davidson iterations)."""

    def __init__(self, h1e, eri_full, norb, nelec, diag_ints=None, strings_a=None, strings_b=None):
        """
        strings_a/strings_b : optional int64 arrays of alpha/beta
            determinant strings, in the SAME order as the `ia`/`ib`
            addresses used elsewhere (e.g. Hamiltonian.strs_a/strs_b).
            Pass these explicitly for a selected-CI space; omitting
            them regenerates the full FCI string space (original,
            full-CI-only behavior), which would silently be the WRONG
            table for a selected/truncated determinant space.
        """
        self.norb = norb
        self.nelec = nelec
        neleca, nelecb = nelec
        if strings_a is None:
            strings_a = cistring.gen_strings4orblist(range(norb), neleca)
        if strings_b is None:
            strings_b = cistring.gen_strings4orblist(range(norb), nelecb)
        self.strings_a = np.asarray(strings_a, dtype=np.int64)
        self.strings_b = np.asarray(strings_b, dtype=np.int64)
        self.nb = len(self.strings_b)
        self.diag_ints = diag_ints if diag_ints is not None else precompute_diag_ints(h1e, eri_full)
        self.cache = Dict.empty(key_type=types.int64, value_type=types.float64)

    def __call__(self, idx):
        """Return H_ii for the given flat CI index or indices.

        Args:
            idx : int or 1-D array-like of int64 flat CI indices
                  (row-major: i = ia*nb + ib)

        Returns:
            scalar float when ``idx`` is an int; otherwise
            ``(out, n_newly_computed)`` where ``out`` is a ``(len(idx),)``
            float64 array.
        """
        if isinstance(idx, (int, np.integer)):
            return hii_on_the_fly_scalar(
                idx,
                self.nb,
                self.nelec,
                self.cache,
                self.strings_a,
                self.strings_b,
                self.diag_ints,
            )
        out, n_missing = hii_on_the_fly(
            None,
            None,
            self.norb,
            self.nelec,
            idx,
            self.cache,
            strings_a=self.strings_a,
            strings_b=self.strings_b,
            diag_ints=self.diag_ints,
            nb=self.nb,
        )
        return out, n_missing

    def __len__(self):
        """Number of H_ii values currently stored in the cache."""
        return len(self.cache)


# ----------------------------------------------------------------------
# Excitation generators: enumerate single/double excited strings from a
# reference bitmask into pre-allocated output arrays.
# ----------------------------------------------------------------------
@njit(cache=True)
def _count_singles_doubles(nelec, nvir):
    """Upper bound on the number of single+double excitations from one string."""
    n_singles = nelec * nvir
    n_doubles = (nelec * (nelec - 1) // 2) * (nvir * (nvir - 1) // 2)
    return 1 + n_singles + n_doubles  # +1 for ref itself


@njit(cache=True)
def excite_single_double_nb(ref, norb, out):
    """Write all SD-excited strings of `ref` into `out`; return count written.

    `out` must have length >= _count_singles_doubles(nelec, nvir).
    Includes `ref` itself as the first entry.
    """
    nelec = int64(0)
    nvir = int64(0)
    for p in range(norb):
        if (ref >> p) & 1:
            nelec += 1
        else:
            nvir += 1

    occ = np.empty(nelec, dtype=np.int64)
    vir = np.empty(nvir, dtype=np.int64)
    oi = int64(0)
    vi = int64(0)
    for p in range(norb):
        if (ref >> p) & 1:
            occ[oi] = p
            oi += 1
        else:
            vir[vi] = p
            vi += 1

    n = int64(0)
    out[n] = ref
    n += 1

    for ix in range(nelec):
        i = occ[ix]
        for ax in range(nvir):
            a = vir[ax]
            out[n] = (ref & ~(int64(1) << i)) | (int64(1) << a)
            n += 1

    for x in range(nelec):
        for y in range(x + 1, nelec):
            i = occ[x]
            j = occ[y]
            for u in range(nvir):
                for w in range(u + 1, nvir):
                    a = vir[u]
                    b = vir[w]
                    out[n] = (
                        (ref & ~(int64(1) << i) & ~(int64(1) << j))
                        | (int64(1) << a)
                        | (int64(1) << b)
                    )
                    n += 1

    return n


@njit(cache=True)
def excite_single_nb(ref, norb, out):
    """Write all singly-excited strings of `ref` into `out`; return count written.

    Includes `ref` itself. `out` must have length >= 1 + nelec*nvir.
    """
    nelec = int64(0)
    nvir = int64(0)
    for p in range(norb):
        if (ref >> p) & 1:
            nelec += 1
        else:
            nvir += 1

    occ = np.empty(nelec, dtype=np.int64)
    vir = np.empty(nvir, dtype=np.int64)
    oi = int64(0)
    vi = int64(0)
    for p in range(norb):
        if (ref >> p) & 1:
            occ[oi] = p
            oi += 1
        else:
            vir[vi] = p
            vi += 1

    n = int64(0)
    out[n] = ref
    n += 1

    for ix in range(nelec):
        i = occ[ix]
        for ax in range(nvir):
            a = vir[ax]
            out[n] = (ref & ~(int64(1) << i)) | (int64(1) << a)
            n += 1

    return n
