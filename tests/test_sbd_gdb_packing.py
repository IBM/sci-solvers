"""Tests for the vectorized GDB determinant packing in the SBD solver.

``mode="gdb"`` interleaves each parallel ``(alpha_i, beta_i)`` CI-string pair into
one ``2 * norb``-bit determinant, laid out as the backend's word-vector form
(``list[size_t]``, word ``w`` holding interleaved bits ``[w * bit_length,
(w + 1) * bit_length)`` little-endian — what ``sbd.from_string`` produces). The
pack/unpack were vectorized in numpy so the cost scales with the determinant count
N (the 10M+ GDB regime), *independently of the orbital count* — so the matrices
below deliberately span both small active spaces and ``norb`` wide enough that
``2 * norb > 64`` (e.g. 36, 40), the case a single-``uint64`` interleave could not
handle.

These target the pure-numpy vectorized helpers directly, so they need no ``sbd``
install: they assert the pack/unpack are exact inverses, and that the packed word
layout matches the backend's documented ``from_string`` convention bit-for-bit.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest

from sci_solvers.energy_solvers.sbd_solver import (
    _pack_gdb_dets_vectorized,
    _unpack_gdb_dets_vectorized,
)

# norb chosen to span both 2*norb <= 64 and 2*norb > 64 (the regime a single-uint64
# interleave could not represent); bit_length spans the sub-word, exact-word, and
# wider-than-a-word cases relative to those norb.
NORBS = (1, 4, 20, 32, 36, 40, 63)
BIT_LENGTHS = (1, 3, 8, 20, 32, 64)


def _random_pairs(norb: int, n_dets: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n_dets`` random ``(alpha, beta)`` occupation integers in ``[0, 2**norb)``."""
    rng = np.random.default_rng(seed)
    hi = 1 << norb
    strs_a = rng.integers(0, hi, size=n_dets, dtype=np.uint64).astype(np.int64)
    strs_b = rng.integers(0, hi, size=n_dets, dtype=np.uint64).astype(np.int64)
    return strs_a, strs_b


@pytest.mark.parametrize("norb", NORBS)
@pytest.mark.parametrize("bit_length", BIT_LENGTHS)
def test_pack_unpack_roundtrip(norb: int, bit_length: int) -> None:
    """unpack(pack(a, b)) recovers the original per-spin CI strings exactly.

    This is the property the GDB path relies on: the interleave into determinant
    words and the de-interleave back into ``(strs_a, strs_b)`` are exact inverses,
    for every ``norb`` (including ``2 * norb > 64``) and ``bit_length``.
    """
    strs_a, strs_b = _random_pairs(norb, n_dets=64, seed=norb * 100 + bit_length)
    dets = _pack_gdb_dets_vectorized(strs_a, strs_b, norb, bit_length)
    got_a, got_b = _unpack_gdb_dets_vectorized(dets, norb, bit_length)
    np.testing.assert_array_equal(got_a, strs_a)
    np.testing.assert_array_equal(got_b, strs_b)


@pytest.mark.parametrize("norb", NORBS)
@pytest.mark.parametrize("bit_length", BIT_LENGTHS)
def test_pack_word_layout_matches_from_string_convention(norb: int, bit_length: int) -> None:
    """Packed words follow the backend's ``from_string`` layout bit-for-bit.

    ``from_string`` sets, for each set bit ``i`` of the value, word ``i //
    bit_length`` bit ``i % bit_length``; here bit ``2k`` is alpha orbital ``k`` and
    bit ``2k + 1`` beta orbital ``k``. We recompute that layout independently (from
    the input integers, no packing) and require the vectorized packer to match, so
    the words handed to ``sort_bitarray`` / ``gdb_diag`` are exactly what the C++
    expects.
    """
    strs_a, strs_b = _random_pairs(norb, n_dets=32, seed=7 + norb + bit_length)
    total_bits = 2 * norb
    num_words = (total_bits + bit_length - 1) // bit_length

    dets = _pack_gdb_dets_vectorized(strs_a, strs_b, norb, bit_length)
    assert len(dets) == len(strs_a)

    for a, b, det in zip(strs_a, strs_b, dets):
        assert len(det) == num_words
        expected = [0] * num_words
        for k in range(norb):
            if (int(a) >> k) & 1:  # alpha orbital k -> interleaved bit 2k
                i = 2 * k
                expected[i // bit_length] |= 1 << (i % bit_length)
            if (int(b) >> k) & 1:  # beta orbital k -> interleaved bit 2k+1
                i = 2 * k + 1
                expected[i // bit_length] |= 1 << (i % bit_length)
        assert [int(w) for w in det] == expected


def test_wide_active_space_uses_multiple_words() -> None:
    """A ``2 * norb > 64`` case genuinely spills into multiple words.

    Guards the core of the fix: at ``norb = 36`` the 72-bit determinant cannot live
    in one ``uint64``, so the packer must emit more than one word and still
    round-trip — the situation the earlier ``2 * norb <= 64`` gate skipped.
    """
    norb, bit_length = 36, 20
    assert 2 * norb > 64  # the regime under test
    strs_a, strs_b = _random_pairs(norb, n_dets=16, seed=2024)
    dets = _pack_gdb_dets_vectorized(strs_a, strs_b, norb, bit_length)
    assert all(len(det) == (2 * norb + bit_length - 1) // bit_length for det in dets)
    assert (2 * norb + bit_length - 1) // bit_length > 1  # more than one word
    got_a, got_b = _unpack_gdb_dets_vectorized(dets, norb, bit_length)
    np.testing.assert_array_equal(got_a, strs_a)
    np.testing.assert_array_equal(got_b, strs_b)


def test_all_bits_set_and_empty_determinant() -> None:
    """Boundary occupations: fully filled and empty per-spin strings round-trip.

    Exercises the high-order interleaved bits (``2 * norb - 1``) — the ones most
    likely to be dropped by an off-by-one in the word slicing.
    """
    norb, bit_length = 40, 20
    full = (1 << norb) - 1
    strs_a = np.array([full, 0, full], dtype=np.int64)
    strs_b = np.array([0, full, full], dtype=np.int64)
    dets = _pack_gdb_dets_vectorized(strs_a, strs_b, norb, bit_length)
    got_a, got_b = _unpack_gdb_dets_vectorized(dets, norb, bit_length)
    np.testing.assert_array_equal(got_a, strs_a)
    np.testing.assert_array_equal(got_b, strs_b)
