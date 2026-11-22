"""Multi-limb determinant strings in the Triton TPB kernels (CUDA only).

The Triton kernels take ``(M,)`` single-word strings or ``(M, n_limbs)`` int64
limb matrices (limb ``k`` holds bits ``[64k, 64k + 64)``). These tests
represent one small basis twice — as single words and as two-limb matrices
with a zero high limb — and check that

1. the tiled Triton matvec (direct and through ``LocalTPBOperator``'s row and
   tile paths) reproduces the batched matvec at ``1e-12`` for both
   representations, and
2. the two-limb Triton output matches the scalar reference
   (``sbd_ref.mult``) to ``1e-12``.

The whole module needs CUDA and Triton and is skipped otherwise.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref.helpers import generate_all_helpers as scalar_helpers
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.sbd_ref.mult import make_diag_terms as scalar_make_diag_terms
from torch_sci_solver.sbd_ref.mult import mult as scalar_mult
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl.distributed_davidson import (
    partition_rows_explicit,
    partition_tiles,
)
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch
from torch_sci_solver.torch_impl.mult_triton import mult_triton

pytestmark = pytest.mark.gpu

NORB = 6
# C(6, 2) = 15 strings per spin -> a 15 x 15 TPB basis.
STRINGS = [int(sum(1 << o for o in occ)) for occ in itertools.combinations(range(NORB), 2)]


def _skip_without_triton() -> None:
    """Skip unless this process has CUDA and an importable Triton build."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernels")
    import torch_sci_solver.torch_impl.mult_triton as mt

    if not mt._HAVE_TRITON:
        pytest.skip("Triton is not importable")


def _limb_matrix(values: list[int], n_limbs: int) -> torch.Tensor:
    """Stack Python-int strings as an ``(M, n_limbs)`` signed-int64 matrix."""
    mask = (1 << 64) - 1
    rows = []
    for value in values:
        row = [(value >> (64 * k)) & mask for k in range(n_limbs)]
        rows.append([limb - (1 << 64) if limb >= 1 << 63 else limb for limb in row])
    return torch.tensor(rows, dtype=torch.int64)


def _synthetic_h6() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 6-orbital / 4-electron spatial integrals."""
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
    eri = np.zeros_like(raw)
    for perm in [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ]:
        eri += raw.transpose(perm)
    eri /= 8.0
    return h1, eri


def _setup(n_limbs: int) -> tuple:
    """Shared CUDA tensors and the reference batched matvec for one encoding."""
    _skip_without_triton()
    device = torch.device("cuda")
    h1, eri = _synthetic_h6()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=device)

    adets = list(STRINGS)
    bdets = list(STRINGS)
    helpers = list(generate_all_torch_helpers(adets, bdets, NORB, device=device))
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    if n_limbs > 1:
        adets_t = _limb_matrix(adets, n_limbs).to(device)
        bdets_t = _limb_matrix(bdets, n_limbs).to(device)
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)

    M_a, M_b = len(adets), len(bdets)
    gen = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(M_a * M_b, generator=gen, dtype=torch.float64, device=device)

    batched = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    scalar = scalar_mult(
        scalar_make_diag_terms(adets, bdets, NORB, I0, I1, I2),
        x.cpu().numpy(),
        adets,
        bdets,
        NORB,
        I1,
        I2,
        list(scalar_helpers(adets, bdets, NORB)),
    )
    return device, integrals, helpers, adets_t, bdets_t, hii, x, batched, scalar


def test_triton_single_word_matches_batched() -> None:
    """The ``N_LIMBS=1`` kernel keeps the single-word result bit-for-bit."""
    _skip_without_triton()
    device, integrals, helpers, adets_t, bdets_t, hii, x, batched, _ = _setup(1)
    got = mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, batched, atol=1e-12, rtol=0.0), (
        float((got - batched).abs().max()) if got.numel() else 0.0
    )


def test_triton_two_limbs_matches_batched_and_scalar() -> None:
    """A two-limb encoding gives the one-word batched and scalar results."""
    _skip_without_triton()
    device, integrals, helpers, adets_t, bdets_t, hii, x, batched, scalar = _setup(2)
    got = mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, batched, atol=1e-12, rtol=0.0), (
        float((got - batched).abs().max()) if got.numel() else 0.0
    )
    assert np.allclose(got.cpu().numpy(), scalar, atol=1e-12, rtol=0.0), float(
        np.abs(got.cpu().numpy() - scalar).max()
    )


@pytest.mark.parametrize("encoding", [1, 2])
def test_triton_operator_row_and_tile_paths(encoding: int) -> None:
    """Both distributed Triton launch sites match the batched matvec."""
    _skip_without_triton()
    device, integrals, helpers, adets_t, bdets_t, hii, x, batched, _ = _setup(encoding)
    M_a, M_b = len(adets_t), len(bdets_t)

    rows = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    row_op = dmat.LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        rows,
        rank=0,
        world_size=1,
        matvec_impl="triton",
    )
    got_rows = row_op(x)
    assert torch.allclose(got_rows, batched, atol=1e-12, rtol=0.0), (
        encoding,
        float((got_rows - batched).abs().max()),
    )

    tile = partition_tiles(M_a, M_b, 1, grid=(1, 1))
    tile_op = dmat.LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        tile,
        rank=0,
        world_size=1,
        matvec_impl="triton",
    )
    got_tile = tile_op(x)
    assert torch.allclose(got_tile, batched, atol=1e-12, rtol=0.0), (
        encoding,
        float((got_tile - batched).abs().max()),
    )
