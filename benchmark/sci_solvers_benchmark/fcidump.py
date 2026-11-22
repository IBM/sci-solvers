"""Read the ``&FCI`` header of an FCIDUMP file without loading its integrals."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import re
from pathlib import Path


def read_fcidump_header(path: str | Path) -> dict[str, int]:
    """Return ``norb``, ``nelec``, ``ms2``, ``n_alpha`` and ``n_beta`` from an FCIDUMP.

    Only the namelist before ``&END`` is read, so this is cheap even for large files.

    Args:
        path: FCIDUMP file.

    Returns:
        The header fields, with ``n_alpha = (nelec + ms2) / 2`` and
        ``n_beta = nelec - n_alpha``.

    Raises:
        ValueError: ``NORB``, ``NELEC`` or ``MS2`` is missing from the header.
    """
    header = []
    with Path(path).open() as f:
        for line in f:
            header.append(line)
            if "&END" in line.upper() or line.strip() == "/":
                break
    text = " ".join(header)
    out: dict[str, int] = {}
    for tag in ("NORB", "NELEC", "MS2"):
        m = re.search(rf"\b{tag}\s*=\s*(-?\d+)", text, re.IGNORECASE)
        if m is None:
            raise ValueError(f"{path}: no {tag} in the FCIDUMP header")
        out[tag.lower()] = int(m.group(1))
    out["n_alpha"] = (out["nelec"] + out["ms2"]) // 2
    out["n_beta"] = out["nelec"] - out["n_alpha"]
    return out
