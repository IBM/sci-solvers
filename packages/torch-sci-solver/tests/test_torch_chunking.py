"""Chunking regression: forcing tiny chunk_size produces the same Wb as
default (large) chunk_size, to machine precision.

The chunk loop is just an outer Python loop over slices of the
excitation tables; it should not introduce any numerical drift because
``index_add_`` is associative across writes within one matvec, and we
always start from the same hii*T pre-pass.

We test:
  - benzoic acid (dataset ID 182) K=50 (Anchor 2 basis) — largest case we run in scalar Python.
  - Forced chunk_sizes: 100, 1000, 100_000 (all smaller than typical
    batch sizes so the loop runs multiple iterations).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import torch
from torch_sci_solver import _paths
from torch_sci_solver.sbd_ref import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import (
    generate_all_torch_helpers,
    make_diag_terms_torch,
    mult_torch,
    to_torch_integrals,
)


def _get_fcidump():
    return str(_paths.fcidump_path("benzoic_acid"))


def _hf_and_singles(k: int) -> list[int]:
    mol = _paths.MOLECULES["benzoic_acid"]
    return _paths.hf_and_singles(mol.norb, mol.nelec_a)[:k]


def _to_int64(values):
    INT64_OFFSET = 1 << 64
    return [int(v) if int(v) < (1 << 63) else int(v) - INT64_OFFSET for v in values]


def test_chunking_matches_unchunked():
    """Chunked matvec must produce the same result as the unchunked path."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump

    raw = fcidump.read(_get_fcidump())
    L = raw["NORB"]
    h1 = raw["H1"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), L)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, raw["ECORE"])

    adets = _hf_and_singles(50)
    bdets = list(adets)
    N = len(adets) * len(bdets)
    print(f"benzoic acid K=50: N = {N}")

    integrals = to_torch_integrals(I0, I1, I2, device="cpu")
    adets_t = torch.tensor(_to_int64(adets), dtype=torch.int64)
    bdets_t = torch.tensor(_to_int64(bdets), dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(adets, bdets, L, device="cpu"))

    # Reference: huge chunk_size (no chunking in practice for K=50)
    hii_ref = make_diag_terms_torch(adets_t, bdets_t, L, integrals, chunk_size=100_000_000)
    rng = np.random.default_rng(seed=7)
    T_np = rng.standard_normal(N)
    T_t = torch.tensor(T_np, dtype=torch.float64)
    Wb_ref = mult_torch(
        hii_ref, T_t, adets_t, bdets_t, L, integrals, helpers, chunk_size=100_000_000
    ).numpy()

    for cs in [100, 1000, 10_000, 100_000]:
        hii_c = make_diag_terms_torch(adets_t, bdets_t, L, integrals, chunk_size=cs)
        Wb_c = mult_torch(
            hii_c, T_t, adets_t, bdets_t, L, integrals, helpers, chunk_size=cs
        ).numpy()
        diag_diff = np.abs(hii_c.numpy() - hii_ref.numpy()).max()
        mv_diff = np.abs(Wb_c - Wb_ref).max()
        scale = max(np.abs(Wb_ref).max(), 1.0)
        print(
            f"  chunk_size={cs:>8d}: diag max diff = {diag_diff:.3e}, "
            f"matvec max diff = {mv_diff:.3e}, scale = {scale:.3e}"
        )
        assert diag_diff < 1e-12, f"diag failed at chunk_size={cs}"
        assert mv_diff < 1e-10 * scale, f"matvec failed at chunk_size={cs}"
    print("test_chunking_matches_unchunked passed")


if __name__ == "__main__":
    test_chunking_matches_unchunked()
    print("\nCHUNKING REGRESSION OK")
