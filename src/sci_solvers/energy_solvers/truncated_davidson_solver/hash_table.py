__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
from numba import njit

EMPTY = np.int64(-1)


@njit(cache=True, inline="always", fastmath=True)
def ht_new(capacity):
    """Allocate a flat hash table sized for at least `capacity` entries (load <= 0.5).
    Returns (keys, vals):
    """
    target = capacity * 2
    cap = np.int64(1) << np.int64(np.ceil(np.log2(target)))
    keys = np.full(cap, EMPTY, dtype=np.int64)
    vals = np.zeros(cap, dtype=np.float64)
    return keys, vals


@njit(cache=True, inline="always")
def _slot(keys, key):
    """Return the slot index for `key` via open-addressing linear probing.
    Uses a Fibonacci/golden-ratio mix to spread keys before masking. Stops at
    the first empty slot or the slot that already holds `key`.
    """
    cap = keys.shape[0]
    mask = cap - 1
    h = (key * np.int64(0x9E3779B97F4A7C15)) & mask
    while keys[h] != EMPTY and keys[h] != key:
        h = (h + 1) & mask
    return h


@njit(cache=True, inline="always")
def ht_add(keys, vals, key, delta):
    """Accumulate `delta` into the entry for `key`, inserting it if absent."""
    i = _slot(keys, key)
    if keys[i] == EMPTY:
        keys[i] = key
    vals[i] += delta


@njit(cache=True, inline="always")
def ht_get(keys, vals, key, default):
    """Return the value stored for `key`, or `default` if the key is absent."""
    i = _slot(keys, key)
    return vals[i] if keys[i] == key else default


@njit(cache=True, inline="always")
def ht_contains(keys, key):
    """Return True if `key` is present in the hash table."""
    i = _slot(keys, key)
    return keys[i] == key


# ---------------------------------------------------------------------------
# Growable variant
# ---------------------------------------------------------------------------
# ht_new/ht_add above require the caller to pre-know an upper bound on the
# number of unique keys that will ever be inserted. This variant allows to
# increase the size of the hash table if needed.


@njit(cache=True, inline="always")
def ht_new_growable(capacity):
    """Like ht_new, but also returns a length-1 int64 `cnt` box (starts at 0)."""
    keys, vals = ht_new(capacity)
    cnt = np.zeros(1, dtype=np.int64)
    return keys, vals, cnt


@njit(cache=True)
def ht_grow(keys, vals):
    """Double capacity and rehash every live entry. Returns new (keys, vals)."""
    old_cap = keys.shape[0]
    new_cap = old_cap * 2
    new_keys = np.full(new_cap, EMPTY, dtype=np.int64)
    new_vals = np.zeros(new_cap, dtype=np.float64)
    for i in range(old_cap):
        k = keys[i]
        if k != EMPTY:
            j = _slot(new_keys, k)
            new_keys[j] = k
            new_vals[j] = vals[i]
    return new_keys, new_vals


@njit(cache=True)
def ht_add_grow(keys, vals, cnt, key, delta):
    """Insert-or-accumulate, growing (doubling + rehashing) first if needed
    to keep load factor <= 0.5. Returns (keys, vals); `cnt` is updated
    in place. Caller must reassign keys/vals after every call.
    """
    i = _slot(keys, key)
    if keys[i] == EMPTY:
        if (cnt[0] + 1) * 2 > keys.shape[0]:
            keys, vals = ht_grow(keys, vals)
            i = _slot(keys, key)  # re-probe: old slot index is invalid post-rehash
        keys[i] = key
        cnt[0] += 1
    vals[i] += delta
    return keys, vals


@njit(cache=True)
def ht_set_grow(keys, vals, cnt, key, value):
    """Insert-or-OVERWRITE. Grows first if needed. Returns (keys,
    vals); `cnt` is updated in place. Caller must reassign keys/vals.
    """
    i = _slot(keys, key)
    if keys[i] == EMPTY:
        if (cnt[0] + 1) * 2 > keys.shape[0]:
            keys, vals = ht_grow(keys, vals)
            i = _slot(keys, key)
        keys[i] = key
        cnt[0] += 1
    vals[i] = value
    return keys, vals
