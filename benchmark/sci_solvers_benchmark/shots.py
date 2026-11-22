"""Read a benchmark input file into a ``{bitstring: count}`` dict.

Two formats, both written by this package:

- ``.json`` -- ``{"counts": {bitstring: count}}`` (quantum-sampled inputs, see
  ``prepare_quantum``).
- ``.npz`` -- ``bitstrings`` ``(N, n_qubits)`` 0/1 array plus ``probs`` ``(N,)`` (synthetic
  inputs, see ``prepare_synthetic``). Counts are recovered as ``round(p / min(probs))``,
  which keeps the probability ratios exactly.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
from pathlib import Path

import numpy as np

SUPPORTED_SUFFIXES = (".json", ".npz")


def load_shots(path: str | Path) -> dict[str, int]:
    """Return ``{bitstring: count}`` for a benchmark input file.

    Raises:
        ValueError: Unsupported suffix, or the file holds no bitstrings.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(p.read_text())
        counts = data.get("counts", data)
    elif suffix == ".npz":
        with np.load(p) as data:
            bitstrings, probs = data["bitstrings"], data["probs"]
        min_prob = float(probs.min())
        counts = {
            "".join(str(int(b)) for b in row): max(1, round(float(q) / min_prob))
            for row, q in zip(bitstrings, probs)
        }
    else:
        raise ValueError(f"unsupported input format {suffix!r} ({p})")
    if not isinstance(counts, dict) or not counts:
        raise ValueError(f"{p}: expected a non-empty {{bitstring: count}} mapping")
    return {str(k): int(v) for k, v in counts.items()}
