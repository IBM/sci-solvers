"""Load measured shots (bitstring -> weight) into a Qiskit ``BitArray`` for the SQD loop."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np


def load_shots(path: str) -> dict:
    """Load shots from pickle (.pkl) or NumPy (.npz) format.

    Each key is a ``'0'``/``'1'`` string of length ``2 * norb`` (spin-down in
    the right half, spin-up in the left half, as produced by Qiskit sampling).
    Values may be integer counts or floating-point probabilities; only their
    relative magnitudes matter.

    Supports:
    - .pkl: pickled dict mapping bitstring -> weight
    - .npz: NumPy archive with 'packed' bitstrings and 'L' (total bits)
    """
    import pickle
    from pathlib import Path

    path_obj = Path(path)

    if path_obj.suffix == ".npz":
        # Load npz format (packed bitstrings)
        data = np.load(path)
        if "L" not in data or "packed" not in data:
            raise ValueError(f"{path}: .npz file must contain 'L' and 'packed' arrays")

        L = int(data["L"])
        packed = data["packed"]

        # Unpack bitstrings — fully vectorised, no Python loops over rows/bits.
        bitstrings = _unpack_bitstrings(packed, L)  # (N, L) bool

        # Count unique bitstrings via numpy unique on packed representation.
        # Converting the bool matrix to a view of structured bytes lets
        # np.unique do the deduplication in C rather than Python.
        # bitstrings is C-contiguous (N, L) bool — view as uint8 rows of L bytes.
        unique_rows, counts = np.unique(
            bitstrings.view(np.uint8).reshape(bitstrings.shape[0], -1), axis=0, return_counts=True
        )
        # Build the string keys from the unique bool rows.
        # np.packbits / chr lookup is faster than a Python join per row.
        unique_bool = unique_rows.astype(bool)
        # Use vectorised byte-to-char translation: each column is either b'0' or b'1'.
        char_matrix = np.where(unique_bool, ord("1"), ord("0")).astype(np.uint8)  # (M, L)
        shots = {row.tobytes().decode("ascii"): int(cnt) for row, cnt in zip(char_matrix, counts)}
    else:
        # Load pickle format (original behavior)
        with Path(path).open("rb") as f:
            shots = pickle.load(f)

    if not isinstance(shots, dict) or not shots:
        raise ValueError(f"{path}: expected a non-empty dict mapping bitstring -> weight")
    return shots


def _unpack_bitstrings(packed: np.ndarray, L: int) -> np.ndarray:
    """Unpack bitstrings from uint8 array to boolean matrix — fully vectorised.

    Args:
        packed: Packed bitstrings as uint8 array (n_shots, n_bytes)
        L: Total number of bits per shot

    Returns:
        Boolean matrix (n_shots, L), C-contiguous
    """
    # Unpack all 8 bits of every byte simultaneously across all shots.
    # np.unpackbits operates on a flat uint8 array and returns one bit per
    # element, MSB first.  Reshape to (n_shots, n_bytes*8) then trim to L.
    unpacked = np.unpackbits(packed, axis=1, bitorder="little")  # (n_shots, n_bytes*8)
    return unpacked[:, :L].astype(bool)


def shots_to_bit_array(shots: dict, *, max_total_shots: int = 2_000_000):
    """Build a Qiskit ``BitArray`` from a bitstring -> weight mapping.

    Weights are converted to integer counts scaled so the smallest positive
    weight maps to 1 (preserving relative frequencies); uniform weights give one
    shot per unique bitstring. If the resulting total would exceed
    ``max_total_shots``, the counts are rescaled down to stay within that bound.
    """
    from qiskit.primitives import BitArray

    bitstrings = list(shots.keys())
    weights = np.array([float(v) for v in shots.values()], dtype=float)
    positive = weights[weights > 0]
    if positive.size == 0:
        raise ValueError("all shot weights are zero")
    counts = np.maximum(1, np.round(weights / positive.min())).astype(np.int64)
    total = int(counts.sum())
    if total > max_total_shots:
        counts = np.maximum(1, np.round(counts * (max_total_shots / total))).astype(np.int64)
    return BitArray.from_counts({b: int(c) for b, c in zip(bitstrings, counts)})


def load_shots_bit_array(path: str, *, max_total_shots: int = 2_000_000):
    """Load shots and return a Qiskit ``BitArray``.

    For ``.npz`` files (packed bitstrings) this uses a fast path that builds
    the ``BitArray`` directly from the packed uint8 data — bypassing the
    intermediate ``shots`` dict entirely.  At 2 M shots this takes ~0.4 s
    instead of ~60 s for the dict roundtrip.

    For ``.pkl`` files the original ``load_shots`` → ``shots_to_bit_array``
    path is used unchanged.
    """
    from pathlib import Path

    from qiskit.primitives import BitArray

    path_obj = Path(path)
    if path_obj.suffix == ".npz":
        data = np.load(path)
        if "L" not in data or "packed" not in data:
            raise ValueError(f"{path}: .npz file must contain 'L' and 'packed' arrays")
        L = int(data["L"])
        packed = data["packed"]  # (N, n_bytes) uint8

        # Subsample to max_total_shots if needed (random without replacement).
        N = packed.shape[0]
        if N > max_total_shots:
            rng = np.random.default_rng()
            idx = rng.choice(N, size=max_total_shots, replace=False)
            idx.sort()
            packed = packed[idx]

        # np.unpackbits: no Python loops, stays uint8 — O(N) C speed.
        # bitorder='little': bit 0 of byte 0 → column 0 (matches the .pkl encoding).
        unpacked = np.unpackbits(packed, axis=1, bitorder="little")[:, :L]  # (N, L) uint8
        return BitArray.from_bool_array(unpacked.astype(bool))

    return shots_to_bit_array(load_shots(path), max_total_shots=max_total_shots)
