"""Tests for the compact spatial-only ERI wrapper.

Covers:

1. Elementwise parity of :class:`CompactEri` against the dense spin-orbital
   tensor for scalar keys, 1-D and 2-D broadcasting index tensors, and mixed
   int/tensor tuples (real and complex).
2. Batched ``mult_torch`` parity with compact vs dense ``I2_eri`` at
   ``atol=1e-12`` (real and complex).
3. GDB pair-mode ``mult_pair_torch`` parity on a full product pair basis.
4. The 16x memory reduction (``storage_bytes`` against the dense footprint).
5. Provenance rejection: a dense ERI without the four-block spin structure
   falls back to the dense conversion and still reproduces the dense matvec.
6. ``to_dense()`` parity with the dense spin-orbital tensor, and
   ``energy_from_rdm`` parity on compact vs dense bundles (real and complex).
7. The slice / ``None`` / Ellipsis error message pointing at ``to_dense()``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import Tensor
from torch_sci_solver.sbd_ref.integrals import TwoInt
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    CompactEri,
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch, mult_pair_torch
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
from torch_sci_solver.torch_impl.rdm import energy_from_rdm

NORB = 4
N_ELEC = 2


def _strings(norb: int, nelec: int) -> list[int]:
    """All ``nelec``-electron determinant bitmasks over ``norb`` orbitals."""
    return [sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), nelec)]


def _real_spatial(seed: int = 0) -> np.ndarray:
    """Deterministic real spatial ERI with exact 8-fold chemist symmetry."""
    rng = np.random.default_rng(seed)
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
    return eri / 8.0


def _complex_spatial(seed: int = 0) -> np.ndarray:
    """Deterministic complex spatial ERI with both Hermitian chemist symmetries."""
    rng = np.random.default_rng(seed)
    eri = np.zeros((NORB,) * 4, dtype=np.complex128)
    for _ in range(NORB + 2):
        g = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
        basis = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", basis, basis.conj())
    return 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))


def _dense_from_spatial(spatial: Tensor) -> Tensor:
    """Build the dense spin-orbital tensor from the four-block structure."""
    n_orb = spatial.shape[0]
    dense = torch.zeros((2 * n_orb,) * 4, dtype=spatial.dtype, device=spatial.device)
    for sp in (0, 1):
        for sq in (0, 1):
            dense[sp::2, sp::2, sq::2, sq::2] = spatial
    return dense


def _bundles(complex_dtype: bool, *, compact_eri: bool):
    """Build the raw ``(I0, I1, I2)`` triple and the requested torch bundle."""
    if complex_dtype:
        spatial = _complex_spatial()
        dtype = torch.complex128
        hcore = spatial[:, :, 0, 0].copy()
        hcore = 0.5 * (hcore + hcore.conj().T)
    else:
        spatial = _real_spatial()
        dtype = torch.float64
        hcore = spatial[:, :, 0, 0].copy()
        hcore = 0.5 * (hcore + hcore.T)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, spatial, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=dtype, compact_eri=compact_eri)
    return I0, I1, I2, integrals


def _matvec_inputs(integrals, complex_dtype: bool):
    """Build ``(hii, x, adets_t, bdets_t, helpers)`` for the 4-orbital basis."""
    strings = _strings(NORB, N_ELEC)
    adets_t = torch.tensor(strings, dtype=torch.int64)
    bdets_t = torch.tensor(strings, dtype=torch.int64)
    helpers = list(generate_all_torch_helpers(strings, strings, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    gen = torch.Generator().manual_seed(11)
    n = len(strings) * len(strings)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if complex_dtype:
        imag = torch.randn(n, generator=gen, dtype=torch.float64)
        x = torch.complex(real, imag)
    else:
        x = real
    return hii, x, adets_t, bdets_t, helpers


# ============================================================================
# 1. Elementwise parity
# ============================================================================


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_scalar_keys_match_dense(complex_dtype: bool) -> None:
    """Every scalar (p, q, r, s) key returns the dense tensor's entry."""
    spatial_np = _complex_spatial() if complex_dtype else _real_spatial()
    spatial = torch.tensor(spatial_np)
    compact = CompactEri(spatial)
    dense = _dense_from_spatial(spatial)
    n_spinorb = 2 * NORB

    for key in itertools.product(range(n_spinorb), repeat=4):
        assert torch.equal(compact[key], dense[key]), key
        # The same key with tensor scalars must agree too.
        tensor_key = tuple(torch.tensor(k) for k in key)
        assert torch.equal(compact[tensor_key], dense[tensor_key]), key


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_one_dim_index_tensors_match_dense(complex_dtype: bool) -> None:
    """1-D index tensors (all-vector and mixed int/tensor) match the dense tensor."""
    spatial_np = _complex_spatial() if complex_dtype else _real_spatial()
    spatial = torch.tensor(spatial_np)
    compact = CompactEri(spatial)
    dense = _dense_from_spatial(spatial)
    n_spinorb = 2 * NORB
    gen = torch.Generator().manual_seed(5)

    for _ in range(4):
        p = torch.randint(n_spinorb, (7,), generator=gen)
        q = torch.randint(n_spinorb, (7,), generator=gen)
        r = torch.randint(n_spinorb, (7,), generator=gen)
        s = torch.randint(n_spinorb, (7,), generator=gen)
        cases = [
            (p, q, r, s),
            (int(p[0]), q, r, s),
            (p, int(q[0]), int(r[0]), s),
            (p, q, int(r[0]), int(s[0])),
            (torch.tensor(int(p[0])), q, torch.tensor(int(r[0])), s),
        ]
        for key in cases:
            got = compact[key]
            want = dense[key]
            assert got.shape == want.shape, key
            assert torch.equal(got, want), key


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_broadcast_index_tensors_match_dense(complex_dtype: bool) -> None:
    """2-D broadcast index tensors (all-vector and mixed int/tensor) match the dense tensor."""
    spatial_np = _complex_spatial() if complex_dtype else _real_spatial()
    spatial = torch.tensor(spatial_np)
    compact = CompactEri(spatial)
    dense = _dense_from_spatial(spatial)
    n_spinorb = 2 * NORB
    gen = torch.Generator().manual_seed(9)
    p = torch.randint(n_spinorb, (3, 1), generator=gen)
    q = torch.randint(n_spinorb, (1, 5), generator=gen)
    r = torch.randint(n_spinorb, (3, 1), generator=gen)
    s = torch.randint(n_spinorb, (1, 5), generator=gen)

    cases = [
        (p, q, r, s),
        (int(p[0, 0]), q, r, s),
        (p, int(q[0, 0]), r, int(s[0, 0])),
        (p, q, int(r[0, 0]), int(s[0, 0])),
        (torch.tensor(int(p[0, 0])), q, torch.tensor(int(r[0, 0])), s),
    ]
    for key in cases:
        got = compact[key]
        want = dense[key]
        assert got.shape == want.shape, key
        assert torch.equal(got, want), key


def test_unsupported_keys_and_dense_methods_raise() -> None:
    """Slices, None, Ellipsis, bad key shapes and dense conversions fail loudly."""
    compact = CompactEri(torch.zeros(NORB, NORB, NORB, NORB))
    bad_keys = [
        0,
        (0, 0, 0),
        (0, 0, 0, 0, 0),
        (slice(None), 0, 0, 0),
        (0, None, 0, 0),
        (0, 0, Ellipsis, 0),
        (torch.tensor(1.0), 0, 0, 0),
        (torch.tensor(True), 0, 0, 0),
        (True, 0, 0, 0),
        ([0, 1], 0, 0, 0),
    ]
    for key in bad_keys:
        with pytest.raises((TypeError, NotImplementedError)):
            compact[key]
    with pytest.raises(NotImplementedError):
        compact.reshape(-1)
    with pytest.raises(NotImplementedError):
        compact.contiguous()
    with pytest.raises(NotImplementedError):
        compact.flatten()


def test_slice_error_points_at_to_dense() -> None:
    """Slice / None / Ellipsis keys raise a NotImplementedError naming ``to_dense``."""
    compact = CompactEri(torch.zeros(NORB, NORB, NORB, NORB))
    for key in [
        (slice(None), 0, 0, 0),
        (0, slice(None), 0, 0),
        (0, 0, None, 0),
        (0, 0, 0, Ellipsis),
        (slice(None), slice(None), slice(None), slice(None)),
    ]:
        with pytest.raises(NotImplementedError, match="to_dense"):
            compact[key]


# ============================================================================
# 2. Batched matvec parity
# ============================================================================


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_matvec_matches_dense(complex_dtype: bool) -> None:
    """``mult_torch`` with compact vs dense ``I2_eri`` agrees to 1e-12."""
    _, _, I2, dense = _bundles(complex_dtype, compact_eri=False)
    _, _, _, compact = _bundles(complex_dtype, compact_eri=True)

    assert dense.eri_compact is False
    assert compact.eri_compact is True
    assert isinstance(dense.I2_eri, Tensor)
    assert isinstance(compact.I2_eri, CompactEri)
    assert torch.equal(dense.I2_eri, torch.tensor(I2._eri, dtype=dense.dtype))
    assert compact.I2_eri.shape == dense.I2_eri.shape
    assert compact.I2_eri.dtype == dense.I2_eri.dtype

    hii_dense, x, adets_t, bdets_t, helpers = _matvec_inputs(dense, complex_dtype)
    hii_compact, _, _, _, _ = _matvec_inputs(compact, complex_dtype)
    assert torch.equal(hii_dense, hii_compact)

    want = mult_torch(hii_dense, x, adets_t, bdets_t, NORB, dense, helpers)
    got = mult_torch(hii_compact, x, adets_t, bdets_t, NORB, compact, helpers)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0)


# ============================================================================
# 3. GDB pair-mode matvec parity
# ============================================================================


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_pair_matvec_matches_dense(complex_dtype: bool) -> None:
    """GDB pair-mode ``mult_pair_torch`` agrees with compact and dense ``I2_eri``."""
    _, _, _, dense = _bundles(complex_dtype, compact_eri=False)
    _, _, _, compact = _bundles(complex_dtype, compact_eri=True)

    strings = _strings(NORB, N_ELEC)
    alpha = np.repeat(strings, len(strings))
    beta = np.tile(strings, len(strings))
    basis = pair_basis_from_arrays(alpha, beta)
    helpers = generate_pair_helpers(basis, NORB, device="cpu", block_matvec="on")
    assert helpers.blocks is not None

    gen = torch.Generator().manual_seed(13)
    real = torch.randn(basis.n_pairs, generator=gen, dtype=torch.float64)
    if complex_dtype:
        imag = torch.randn(basis.n_pairs, generator=gen, dtype=torch.float64)
        x = torch.complex(real, imag)
    else:
        x = real

    hii_dense = make_diag_terms_pair_torch(basis, NORB, dense)
    hii_compact = make_diag_terms_pair_torch(basis, NORB, compact)
    assert torch.equal(hii_dense, hii_compact)

    want = mult_pair_torch(hii_dense, x, basis, helpers, NORB, dense)
    got = mult_pair_torch(hii_compact, x, basis, helpers, NORB, compact)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0)


# ============================================================================
# 4. Memory footprint
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_storage_is_spatial_only(dtype: torch.dtype) -> None:
    """The wrapper stores exactly ``L^4 * itemsize`` bytes — 16x below dense."""
    spatial = torch.zeros(NORB, NORB, NORB, NORB, dtype=dtype)
    compact = CompactEri(spatial)
    itemsize = torch.empty((), dtype=dtype).element_size()
    dense_bytes = (2 * NORB) ** 4 * itemsize

    assert compact.storage_bytes == NORB**4 * itemsize
    assert compact.storage_bytes * 16 == dense_bytes
    assert compact.numel() == (2 * NORB) ** 4
    assert compact.element_size() == itemsize
    assert compact.shape == torch.Size((2 * NORB,) * 4)
    assert compact.dtype == dtype
    assert compact.device == torch.device("cpu")


# ============================================================================
# 5. Provenance rejection
# ============================================================================


def _perturbed_two_int(I2: TwoInt, mutate: Callable[[np.ndarray], None]) -> TwoInt:
    """Copy ``I2`` and apply ``mutate`` to the dense host ERI, rebuilding the caches."""
    eri = I2._eri.copy()
    mutate(eri)
    p = np.arange(I2.n_spinorb).reshape(-1, 1)
    q = np.arange(I2.n_spinorb).reshape(1, -1)
    bad = TwoInt(I2.n_spinorb, dtype=eri.dtype)
    bad._eri = eri
    bad._direct = eri[p, p, q, q]
    bad._exchange = eri[p, q, q, p]
    return bad


def _mixed_block_nonzero(eri: np.ndarray) -> None:
    """Make a spin-forbidden quad nonzero."""
    eri[0, 1, 0, 0] = 1.0


def _nonzero_block_mismatch(eri: np.ndarray) -> None:
    """Break the equality between a nonzero quad and the spatial block."""
    eri[0::2, 0::2, 1::2, 1::2] += 0.5


@pytest.mark.parametrize("mutate", [_mixed_block_nonzero, _nonzero_block_mismatch])
def test_non_rhf_structure_falls_back_to_dense(mutate: Callable[[np.ndarray], None]) -> None:
    """A dense ERI without the four-block structure falls back and matches dense."""
    I0, I1, I2, valid = _bundles(False, compact_eri=True)
    bad = _perturbed_two_int(I2, mutate)

    dense = to_torch_integrals(I0, I1, bad, device="cpu", compact_eri=False)
    fallback = to_torch_integrals(I0, I1, bad, device="cpu", compact_eri=True)

    assert fallback.eri_compact is False
    assert isinstance(fallback.I2_eri, Tensor)
    assert torch.equal(fallback.I2_eri, dense.I2_eri)

    hii_dense, x, adets_t, bdets_t, helpers = _matvec_inputs(dense, False)
    hii_fallback, _, _, _, _ = _matvec_inputs(fallback, False)
    want = mult_torch(hii_dense, x, adets_t, bdets_t, NORB, dense, helpers)
    got = mult_torch(hii_fallback, x, adets_t, bdets_t, NORB, fallback, helpers)
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0)

    if mutate is _nonzero_block_mismatch:
        # The perturbation is visible to the matvec, so a wrongly-used compact
        # representation (which would drop it) could not sneak through.
        valid_hii, valid_x, valid_adets, valid_bdets, valid_helpers = _matvec_inputs(valid, False)
        valid_out = mult_torch(
            valid_hii, valid_x, valid_adets, valid_bdets, NORB, valid, valid_helpers
        )
        assert not torch.allclose(got, valid_out, atol=1e-8, rtol=0.0)


# ============================================================================
# 6. Dense materialisation and energy_from_rdm
# ============================================================================


def _random_rdms(
    n_orb: int, dtype: torch.dtype, seed: int = 7
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Random ``(γα, γβ, Γaa, Γab, Γbb)`` with the shapes ``energy_from_rdm`` takes."""
    gen = torch.Generator().manual_seed(seed)

    def draw(shape: tuple[int, ...]) -> Tensor:
        real = torch.randn(shape, generator=gen, dtype=torch.float64)
        if dtype.is_complex:
            imag = torch.randn(shape, generator=gen, dtype=torch.float64)
            return torch.complex(real, imag).to(dtype)
        return real

    gamma_a = draw((n_orb, n_orb))
    gamma_b = draw((n_orb, n_orb))
    Gaa = draw((n_orb,) * 4)
    Gab = draw((n_orb,) * 4)
    Gbb = draw((n_orb,) * 4)
    return gamma_a, gamma_b, Gaa, Gab, Gbb


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_to_dense_matches_dense(complex_dtype: bool) -> None:
    """``to_dense()`` reproduces the dense spin-orbital tensor exactly."""
    spatial_np = _complex_spatial() if complex_dtype else _real_spatial()
    spatial = torch.tensor(spatial_np)
    compact = CompactEri(spatial)
    dense = _dense_from_spatial(spatial)

    got = compact.to_dense()
    assert got.shape == dense.shape
    assert got.dtype == spatial.dtype
    assert got.device == spatial.device
    assert torch.equal(got, dense)


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_energy_from_rdm_matches_dense(complex_dtype: bool) -> None:
    """``energy_from_rdm`` returns the same energy for compact and dense bundles."""
    _, _, _, dense = _bundles(complex_dtype, compact_eri=False)
    _, _, _, compact = _bundles(complex_dtype, compact_eri=True)
    assert isinstance(compact.I2_eri, CompactEri)

    gamma_a, gamma_b, Gaa, Gab, Gbb = _random_rdms(NORB, dense.dtype)
    want = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, dense)
    got = energy_from_rdm(gamma_a, gamma_b, Gaa, Gab, Gbb, compact)
    assert not got.dtype.is_complex
    assert got.dtype == want.dtype
    assert torch.equal(got, want)
