"""
Unified sparse CI vector module: fixed-capacity flat, growable flat, and compact sorted representations.

FLAT MAP (open-addressing, EMPTY-sentinel keys/vals arrays):
  - Growable (init_flat_growable / ht_add_grow / ht_set_grow) -- resizes on demand.
  count/multiply/dot/normalize share one implementation for both flavors.

COMPACT (sorted, hole-free idx/val arrays):
  Produced by from_flat() or compact_truncate(). Uses two-pointer merge-join
  instead of open-addressing, so compact_dot() is O(n) not O(capacity).
  compact_normalize() normalizes in-place.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
from numba import njit

from sci_solvers.energy_solvers.truncated_davidson_solver.hash_table import (
    EMPTY,
    _slot,
    ht_add_grow,
    ht_new_growable,
    ht_set_grow,
)

# =====================================================================
# FLAT representation -- shared read path (fixed OR growable)
# =====================================================================


def init_flat_growable(capacity: int = 16):
    """Return an empty growable flat vector (keys, vals) with initial capacity `capacity`.

    Grows automatically on insert. Default of 16 suits small seed/unit vectors.
    """
    keys, vals, _cnt = ht_new_growable(capacity)
    return keys, vals


def from_single(idx: int, value: float = 1.0, capacity: int = 4):
    """Return a single-entry growable flat vector with v[idx] = value."""
    keys, vals = init_flat_growable(capacity)
    return set_value(keys, vals, np.int64(idx), value)


@njit(cache=True)
def count(keys: np.ndarray) -> np.int64:
    """Return the number of live (non-EMPTY) entries. O(capacity).

    Use when you need the exact live count (e.g. sizing output allocations).
    Don't call just to pick the smaller side of a dot() -- see dot().
    """
    n = np.int64(0)
    for i in range(keys.shape[0]):
        if keys[i] != EMPTY:
            n += np.int64(1)
    return n


@njit(cache=True, inline="always")
def _cnt_box(keys: np.ndarray) -> np.ndarray:
    """Return a transient length-1 cnt array initialised to count(keys).

    Avoids threading a persistent cnt through every caller at the cost of
    one extra O(capacity) scan per growable-insert call.
    """
    c = np.empty(1, dtype=np.int64)
    c[0] = count(keys)
    return c


@njit(cache=True)
def multiply(s: float, keys: np.ndarray, vals: np.ndarray) -> None:
    """Scale v in-place: vals[i] *= s for every live entry."""
    for i in range(keys.shape[0]):
        if keys[i] != EMPTY:
            vals[i] = s * vals[i]


@njit(cache=True)
def dot(keys1: np.ndarray, vals1: np.ndarray, keys2: np.ndarray, vals2: np.ndarray) -> float:
    """Dot product of two flat vectors (fixed or growable).

    Iterates the smaller-capacity side and probes the larger; uses capacity as
    a proxy for live count (load <= 0.5 makes this equivalent for the swap decision
    without paying two extra full-array scans to compute exact counts).
    Not valid on compact vectors -- use compact_dot() for those.
    """
    if keys1.shape[0] > keys2.shape[0]:
        keys1, vals1, keys2, vals2 = keys2, vals2, keys1, vals1

    total = 0.0
    for i in range(keys1.shape[0]):
        k = keys1[i]
        if k != EMPTY:
            j = _slot(keys2, k)
            if keys2[j] == k:
                total += vals1[i] * vals2[j]
    return total


@njit(cache=True)
def normalize(keys: np.ndarray, vals: np.ndarray, tol: float = 1e-14) -> None:
    """Normalize v in-place: v /= ||v||. No-op if ||v|| < tol."""
    norm_v = dot(keys, vals, keys, vals) ** 0.5
    if norm_v < tol:
        return
    multiply(1.0 / norm_v, keys, vals)


@njit(cache=True)
def axpy_grow(
    base_keys: np.ndarray, base_vals: np.ndarray, keys: np.ndarray, vals: np.ndarray, s: float = 1.0
):
    """Growable axpy: base += s * vec, resizing base as needed.

    Returns (keys, vals) -- caller MUST reassign:
        base_keys, base_vals = axpy_grow(base_keys, base_vals, keys, vals, s)
    """
    cnt = _cnt_box(base_keys)
    for i in range(keys.shape[0]):
        k = keys[i]
        if k != EMPTY:
            base_keys, base_vals = ht_add_grow(base_keys, base_vals, cnt, k, s * vals[i])
    return base_keys, base_vals


@njit(cache=True)
def set_value(keys: np.ndarray, vals: np.ndarray, key: np.int64, value: float):
    """Set v[key] = value (overwrite). Growable only; grows if needed.

    Returns (keys, vals) -- caller MUST reassign.
    """
    cnt = _cnt_box(keys)
    return ht_set_grow(keys, vals, cnt, key, value)


# =====================================================================
# COMPACT representation (sorted, hole-free idx/val arrays)
# =====================================================================
#
# CONTRACT: `idx` must be sorted ascending on entry to every function below.
# from_flat() establishes this invariant.


@njit(cache=True, fastmath=True)
def compact_dot(idx1: np.ndarray, val1: np.ndarray, idx2: np.ndarray, val2: np.ndarray) -> float:
    """Dot product of two compact sorted vectors via two-pointer merge join: O(n+m).

    fastmath=True enables SIMD reduction; data-dependent pointer advances in the
    while loop prevent vectorization there, but the accumulator benefits.
    """
    i = 0
    j = 0
    n1 = idx1.shape[0]
    n2 = idx2.shape[0]
    total = 0.0
    while i != n1 and j != n2:
        ki = idx1[i]
        kj = idx2[j]
        if ki < kj:
            i += 1
        elif ki > kj:
            j += 1
        else:
            total += val1[i] * val2[j]
            i += 1
            j += 1
    return total


@njit(cache=True)
def from_flat(keys: np.ndarray, vals: np.ndarray):
    """Convert a flat (fixed or growable) vector to a compact, sorted (idx, val) pair.

    Use at the boundary where a flat vector becomes a Davidson basis vector.
    """
    n = 0
    for i in range(keys.shape[0]):
        if keys[i] != EMPTY:
            n += 1

    out_idx = np.empty(n, dtype=np.int64)
    out_val = np.empty(n, dtype=np.float64)
    p = 0
    for i in range(keys.shape[0]):
        k = keys[i]
        if k != EMPTY:
            out_idx[p] = k
            out_val[p] = vals[i]
            p += 1

    order = np.argsort(out_idx)
    return out_idx[order], out_val[order]


@njit(cache=True)
def compact_truncate(keys: np.ndarray, vals: np.ndarray, target_size: int):
    """Return new dense flat arrays (sorted by index) keeping the top-`target_size` entries by magnitude."""
    n = count(keys)
    target_size = max(int(target_size), 1)

    if n <= target_size:
        out_keys = np.empty(n, dtype=keys.dtype)
        out_vals = np.empty(n, dtype=vals.dtype)
        p = 0
        for i in range(keys.shape[0]):
            k = keys[i]
            if k != EMPTY:
                out_keys[p] = k
                out_vals[p] = vals[i]
                p += 1
        order = np.argsort(out_keys)
        return out_keys[order], out_vals[order]

    absvals = np.empty(n, dtype=np.float64)
    p = 0
    for i in range(keys.shape[0]):
        if keys[i] != EMPTY:
            absvals[p] = abs(vals[i])
            p += 1
    threshold = np.partition(absvals, n - target_size)[n - target_size]

    out_keys = np.empty(target_size, dtype=keys.dtype)
    out_vals = np.empty(target_size, dtype=vals.dtype)
    kept = 0
    for i in range(keys.shape[0]):
        k = keys[i]
        if k != EMPTY and abs(vals[i]) >= threshold:
            out_keys[kept] = k
            out_vals[kept] = vals[i]
            kept += 1
            if kept == target_size:
                break

    order = np.argsort(out_keys[:kept])
    return out_keys[:kept][order], out_vals[:kept][order]


@njit(cache=True, fastmath=True)
def compact_norm(val: np.ndarray) -> float:
    """Return ||v|| = sqrt(sum(v_i**2)) with one pass, zero allocation."""
    total = 0.0
    for i in range(val.shape[0]):
        v = val[i]
        total += v * v
    return total**0.5


@njit(cache=True)
def compact_multiply(s: float, val: np.ndarray) -> None:
    """Scale v in-place. No EMPTY check needed -- compact vectors have no holes."""
    for i in range(val.shape[0]):
        val[i] = s * val[i]


@njit(cache=True)
def compact_normalize(val: np.ndarray, tol: float = 1e-14) -> None:
    """Normalize v in-place: v /= ||v||. No-op if ||v|| < tol."""
    norm_v = compact_norm(val)
    if norm_v < tol:
        return
    compact_multiply(1.0 / norm_v, val)
