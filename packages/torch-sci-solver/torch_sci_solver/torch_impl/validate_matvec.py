"""Correctness gate for matvec backends."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from itertools import combinations

import numpy as np
import torch

from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
from torch_sci_solver.torch_impl.sbdiag import _to_int64_python


def _setup(adets, bdets, n_orb, I0, I1, I2, device, dtype, chunk_size):
    """Setup."""
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    adets_t = torch.tensor(_to_int64_python(adets), dtype=torch.int64, device=device)
    bdets_t = torch.tensor(_to_int64_python(bdets), dtype=torch.int64, device=device)
    helpers = list(generate_all_torch_helpers(adets, bdets, n_orb, device=device))
    hii = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals, chunk_size=chunk_size)
    return hii, adets_t, bdets_t, integrals, helpers


def compare_matvec_impls(
    adets,
    bdets,
    n_orb,
    I0,
    I1,
    I2,
    *,
    impls=("batched", "triton"),
    n_vectors=4,
    seed=0,
    device="cuda",
    dtype=torch.float64,
    chunk_size=2_000_000,
) -> dict:
    """Apply each backend's matvec to shared random vectors; report max|delta| vs impls[0]."""
    if isinstance(device, str):
        device = torch.device(device)
    hii, adets_t, bdets_t, integrals, helpers = _setup(
        adets, bdets, n_orb, I0, I1, I2, device, dtype, chunk_size
    )
    n = len(adets) * len(bdets)
    gen = torch.Generator(device=device).manual_seed(seed)
    vs = [torch.randn(n, generator=gen, device=device, dtype=dtype) for _ in range(n_vectors)]
    ref = impls[0]
    ref_mv = build_matvec(
        ref, hii, adets_t, bdets_t, n_orb, integrals, helpers, chunk_size=chunk_size
    )
    ref_out = [ref_mv(v) for v in vs]
    diffs = {ref: 0.0}
    for impl in impls[1:]:
        mv = build_matvec(
            impl, hii, adets_t, bdets_t, n_orb, integrals, helpers, chunk_size=chunk_size
        )
        diffs[impl] = max(float((mv(v) - r).abs().max().item()) for v, r in zip(vs, ref_out))
    return diffs


def _h4_system():
    """H4 system."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.kernel(mol, c), mol.nao).astype(np.float64)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, float(mol.energy_nuc()))
    n_orb = int(mol.nao)
    na = mol.nelectron // 2
    adets = sorted(sum(1 << o for o in c) for c in combinations(range(n_orb), na))
    return adets, list(adets), n_orb, I0, I1, I2
