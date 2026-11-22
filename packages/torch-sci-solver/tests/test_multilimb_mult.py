"""Multi-limb (strings wider than 64 bits) support in the full-space TPB matvec.

Verifies:

1. Single-word strings (``n_orb <= 64``) are bit-identical to the same basis
   carried as a forced one-limb ``(M, 1)`` matrix, for real and complex
   integral dtypes; the real case also reproduces the scalar reference
   ``sbd_ref.mult`` on a dense random system.
2. Two-limb (``n_orb=70``) and three-limb (``n_orb=130``) selected bases
   reproduce ``sbd_ref.mult.make_diag_terms`` and one matvec of
   ``sbd_ref.mult.mult`` to ``1e-11`` for every task type: α-only (different
   α/β sets), β-only, αβ (both singles tables) and all tasks together, for
   real and complex integral dtypes.

The limb systems are far too large for the dense ``(2L)^4`` ERI tensor the
standard builders use (3 GB at ``n_orb=70``, terabytes at 130), so the tests
define a low-rank chemist-order ERI

.. math::

    (pq|rs) = \\tfrac{1}{2}\\left(A_{pqrs} + A_{rspq}\\right), \\qquad
    A_{pqrs} = \\sum_c g_c[p, q]\\, \\overline{g_c[r, s]},

with Hermitian ``g_c``. This is exactly Hermitian and particle-exchange
symmetric, and is evaluated on demand on both the scalar and torch sides.

For genuinely complex integrals the scalar reference's SBD ordering yields the
conjugate of the operator the fast kernels build (its ``(pq|rs)`` pairs lead
with the annihilation index, so the Hermitian conjugate is taken). The
matvec is linear in the integrals with real coefficients, so the complex
oracle is evaluated by splitting the complex set into its real and imaginary
parts, running the real scalar reference on each, and recombining:
``H(conj I) = H(I.real) - i H(I.imag)``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from typing import Optional

import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref import generate_all_helpers, make_diag_terms, mult

# torch impl
from torch_sci_solver.torch_impl import (
    TorchIntegrals,
    generate_all_torch_helpers,
    make_diag_terms_torch,
    mult_torch,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.helpers import _n_limbs
from torch_sci_solver.torch_impl.integrals import setup_integrals_from_pyscf_h1_eri_vec
from torch_sci_solver.torch_impl.mult import build_beta_pass_cache

_MASK64 = (1 << 64) - 1


# ============================================================================
# String helpers
# ============================================================================


def _split_limbs(value: int, n_limbs: int) -> list[int]:
    """Split a Python-int string into ``n_limbs`` 64-bit limbs."""
    return [(value >> (64 * k)) & _MASK64 for k in range(n_limbs)]


def _limb_array(strings: list[int], n_orb: int) -> np.ndarray:
    """``(M, n_limbs)`` uint64 limb matrix of Python-int strings."""
    n_limbs = _n_limbs(n_orb)
    return np.array([_split_limbs(v, n_limbs) for v in strings], dtype=np.uint64)


def _limb_tensor(strings: list[int], n_orb: int) -> torch.Tensor:
    """``(M, n_limbs)`` int64 limb tensor (signed bit pattern) of strings."""
    arr = np.ascontiguousarray(_limb_array(strings, n_orb).view(np.int64))
    return torch.from_numpy(arr.copy())


def _single_word_tensor(strings: list[int]) -> torch.Tensor:
    """``(M,)`` int64 tensor in the signed bit-pattern convention."""
    return torch.tensor([v if v < (1 << 63) else v - (1 << 64) for v in strings], dtype=torch.int64)


def _enumerate_strings(n_orb: int, n_elec: int) -> list[int]:
    """All fixed-particle-number strings over ``n_orb`` orbitals, ascending."""
    return sorted(
        sum(1 << b for b in combo) for combo in itertools.combinations(range(n_orb), n_elec)
    )


def _neighbor_strings(base: int, n_orb: int, count: int, seed: int) -> list[int]:
    """Single/double-excitation neighbours of ``base``, sorted, particle number fixed."""
    rng = np.random.default_rng(seed)
    occupied = [k for k in range(n_orb) if (base >> k) & 1]
    empty = [k for k in range(n_orb) if not (base >> k) & 1]
    reachable = (
        1
        + len(occupied) * len(empty)
        + (len(occupied) * (len(occupied) - 1) // 2) * (len(empty) * (len(empty) - 1) // 2)
    )
    count = min(count, reachable)
    strings = {base}
    while len(strings) < count:
        if len(occupied) >= 2 and len(empty) >= 2 and rng.random() < 0.5:
            a, b = rng.choice(occupied, size=2, replace=False)
            c, d = rng.choice(empty, size=2, replace=False)
            value = base ^ (1 << int(a)) ^ (1 << int(b)) ^ (1 << int(c)) ^ (1 << int(d))
        else:
            value = base ^ (1 << int(rng.choice(occupied))) ^ (1 << int(rng.choice(empty)))
        strings.add(value)
    return sorted(strings)


def _selected_strings(n_orb: int, n_alpha: int, n_beta: int) -> tuple[list[int], list[int]]:
    """Deterministic α/β sets whose top orbitals cross the 64-bit edges."""
    base_a = sum(1 << o for o in {0, 3, n_orb - 1, max(0, n_orb - 5)})
    base_b = sum(1 << o for o in {1, 4, n_orb - 2, max(0, n_orb - 6)})
    return (
        _neighbor_strings(base_a, n_orb, n_alpha, seed=3),
        _neighbor_strings(base_b, n_orb, n_beta, seed=5),
    )


# ============================================================================
# Low-rank chemist-order ERI, evaluated on demand
# ============================================================================


class _LowRankERI:
    """Spatial chemist ERI ``(pq|rs)`` from a low-rank Hermitian factorization.

    ``A[p,q,r,s] = sum_c g_c[p,q] conj(g_c[r,s])`` with Hermitian ``g_c``;
    the returned tensor is ``0.5 * (A[p,q,r,s] + A[r,s,p,q])``, exactly
    Hermitian and particle-exchange symmetric.
    """

    def __init__(self, factors: list[np.ndarray], *, real: bool) -> None:
        """Store the factor matrices; ``real`` skips the exchange term."""
        self._factors = factors
        self._real = real

    def spatial(self, p: int, q: int, r: int, s: int) -> complex:
        """Evaluate ``(pq|rs)`` for one spatial index quadruple."""
        total = 0.0
        for g in self._factors:
            total = total + g[p, q] * np.conj(g[r, s])
        if self._real:
            return total
        exchange = 0.0
        for g in self._factors:
            exchange = exchange + g[r, s] * np.conj(g[p, q])
        return 0.5 * (total + exchange)


class _TorchERI:
    """On-demand ``(2L)^4`` spin-orbital ERI supporting torch advanced indexing.

    The fast kernels index the ERI tensor directly, but a dense ``(2L)^4``
    tensor does not fit for ``n_orb >= 70``. This object evaluates the same
    low-rank expression as :class:`_LowRankERI` for the requested indices and
    zeroes the spin-forbidden entries, matching
    ``setup_integrals_from_pyscf_h1_eri``'s spin-orbital layout.
    """

    def __init__(
        self,
        eri: _LowRankERI,
        n_spinorb: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Materialize the factor matrices as ``dtype`` torch tensors."""
        self.dtype = dtype
        self.device = device
        self.shape = (n_spinorb,) * 4
        self._real = eri._real
        self._factors = [torch.tensor(g, dtype=dtype, device=device) for g in eri._factors]

    def __getitem__(self, index) -> torch.Tensor:
        """Gather the requested ``(p, q, r, s)`` entries with broadcasting."""
        p, q, r, s = (
            t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.int64, device=self.device)
            for t in index
        )
        shape = torch.broadcast_shapes(p.shape, q.shape, r.shape, s.shape)
        p, q, r, s = (t.expand(shape) for t in (p, q, r, s))
        pp, qq, rr, ss = p // 2, q // 2, r // 2, s // 2

        total = None
        for g in self._factors:
            term = g[pp, qq] * torch.conj(g[rr, ss])
            total = term if total is None else total + term
        if self._real:
            out = total
        else:
            exchange = None
            for g in self._factors:
                term = g[rr, ss] * torch.conj(g[pp, qq])
                exchange = term if exchange is None else exchange + term
            out = 0.5 * (total + exchange)

        same_spin = ((p & 1) == (q & 1)) & ((r & 1) == (s & 1))
        return torch.where(same_spin, out, torch.zeros_like(out))


def _random_factors(n_orb: int, n_factors: int, seed: int, *, complex_: bool) -> list[np.ndarray]:
    """Hermitian ``n_factors`` matrices; real symmetric when ``complex_`` is false."""
    rng = np.random.default_rng(seed)
    factors = []
    for _ in range(n_factors):
        g = rng.standard_normal((n_orb, n_orb))
        if complex_:
            g = g + 1j * rng.standard_normal((n_orb, n_orb))
            factors.append(g + g.conj().T)
        else:
            factors.append(0.5 * (g + g.T))
    return factors


class _ScalarOneInt:
    """Same-spin one-electron block matching ``setup_integrals_from_pyscf_h1_eri``."""

    def __init__(self, hcore: np.ndarray, part: Optional[str] = None) -> None:
        """Store the spatial matrix and the real/imag part selector."""
        self._h = hcore
        self._part = part

    def value(self, i: int, j: int) -> float:
        """``I1[i, j]``, zero across spins."""
        if (i % 2) != (j % 2):
            return 0.0
        value = self._h[i // 2, j // 2]
        if self._part == "real":
            return value.real
        if self._part == "imag":
            return value.imag
        return value


class _ScalarTwoInt:
    """On-demand two-electron integrals with SBD's spin-selection rule."""

    def __init__(self, eri: _LowRankERI, part: Optional[str] = None) -> None:
        """Store the low-rank ERI evaluator and an optional real/imag selector."""
        self._eri = eri
        self._part = part
        self._cache: dict[tuple[int, int, int, int], complex] = {}

    def value(self, p: int, q: int, r: int, s: int) -> complex:
        """``(pq|rs)``, zero when either chemist pair has mixed spin."""
        if (p % 2) != (q % 2) or (r % 2) != (s % 2):
            return 0.0
        key = (p, q, r, s)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = self._eri.spatial(p // 2, q // 2, r // 2, s // 2)
        if self._part == "real":
            value = value.real
        elif self._part == "imag":
            value = value.imag
        self._cache[key] = value
        return value

    def direct_value(self, i: int, j: int) -> complex:
        """``(ii|jj)``."""
        return self.value(i, i, j, j)

    def exchange_value(self, i: int, j: int) -> complex:
        """``(ij|ji)``."""
        return self.value(i, j, j, i)


def _spatial_hcore(n_orb: int, seed: int, *, complex_: bool) -> np.ndarray:
    """Random Hermitian ``(n_orb, n_orb)`` one-electron matrix."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((n_orb, n_orb))
    if complex_:
        hcore = hcore + 1j * rng.standard_normal((n_orb, n_orb))
    return 0.5 * (hcore + hcore.conj().T)


def _torch_integrals(
    n_orb: int, hcore: np.ndarray, eri: _LowRankERI, dtype: torch.dtype
) -> TorchIntegrals:
    """Build a :class:`TorchIntegrals` bundle around the on-demand ERI."""
    n_spinorb = 2 * n_orb
    device = torch.device("cpu")

    I1 = np.zeros((n_spinorb, n_spinorb), dtype=np.complex128 if dtype.is_complex else np.float64)
    I1[0::2, 0::2] = hcore
    I1[1::2, 1::2] = hcore
    I1_t = torch.tensor(I1, dtype=dtype, device=device)

    eri_t = _TorchERI(eri, n_spinorb, dtype, device)
    idx = torch.arange(n_spinorb, dtype=torch.int64)
    p = idx.unsqueeze(1).expand(n_spinorb, n_spinorb)
    q = idx.unsqueeze(0).expand(n_spinorb, n_spinorb)
    I2_direct = eri_t[p, p, q, q]
    I2_exchange = eri_t[p, q, q, p]

    return TorchIntegrals(
        I0=0.37,
        I1=I1_t,
        I2_eri=eri_t,  # type: ignore[arg-type]
        I1_diag=torch.diagonal(I1_t).contiguous(),
        I2_direct=I2_direct,
        I2_exchange=I2_exchange,
        n_spinorb=n_spinorb,
        device=device,
        dtype=dtype,
    )


def _scalar_integrals(hcore: np.ndarray, eri: _LowRankERI, part: Optional[str] = None):
    """Scalar-reference ``(I0, I1, I2)`` around the on-demand ERI.

    ``I0`` carries no imaginary part, so the imaginary split gets ``I0 = 0``.
    """
    I0 = 0.0 if part == "imag" else 0.37
    return I0, _ScalarOneInt(hcore, part), _ScalarTwoInt(eri, part)


# ============================================================================
# Oracle helpers
# ============================================================================


def _helpers_for(helpers: list, which: str) -> list:
    """Select the task helper(s) that ``which`` names."""
    if which == "all":
        return list(helpers)
    task_type = {"alpha": 2, "beta": 1, "alphabeta": 0}[which]
    return [h for h in helpers if h.task_type == task_type]


def _scalar_complex_matvec(
    hii_re: np.ndarray,
    hii_im: np.ndarray,
    T: np.ndarray,
    adets: list[int],
    bdets: list[int],
    n_orb: int,
    I1_re,
    I1_im,
    I2_re,
    I2_im,
    helpers: list,
) -> np.ndarray:
    """Scalar matvec oracle for complex integrals via the real/imag split.

    ``H(I_real + i I_imag)`` is linear in the integrals with real
    coefficients, so the real reference evaluates each part separately:
    ``H(conj I) = H(I_real) - i H(I_imag)``, which is the operator the fast
    kernels build (their chemist pairs lead with the creation index).
    """
    X = T.real if np.iscomplexobj(T) else T
    Y = T.imag if np.iscomplexobj(T) else None

    AX = mult(hii_re, X, adets, bdets, n_orb, I1_re, I2_re, helpers)
    BX = mult(hii_im, X, adets, bdets, n_orb, I1_im, I2_im, helpers)
    if Y is None:
        return AX - 1j * BX

    AY = mult(hii_re, Y, adets, bdets, n_orb, I1_re, I2_re, helpers)
    BY = mult(hii_im, Y, adets, bdets, n_orb, I1_im, I2_im, helpers)
    return (AX + BY) + 1j * (AY - BX)


def _assert_close(got: np.ndarray, expected: np.ndarray, label: str) -> None:
    """Assert ``got == expected`` to 1e-11 absolute (no relative slack)."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    diff = np.abs(got - expected).max() if got.size else 0.0
    scale = np.abs(expected).max() if expected.size else 0.0
    assert diff <= 1e-11, f"{label}: max |got - expected| = {diff:.3e} (scale {scale:.3e})"


# ============================================================================
# 1. Single-word regression
# ============================================================================


def _dense_system(n_orb: int, seed: int, *, complex_: bool):
    """Dense standard-builder integrals (``n_orb`` small enough for ``(2L)^4``)."""
    hcore = _spatial_hcore(n_orb, seed + 1, complex_=complex_)
    factors = _random_factors(n_orb, 2, seed + 2, complex_=complex_)
    eri = np.zeros((n_orb,) * 4, dtype=np.complex128 if complex_ else np.float64)
    for g in factors:
        eri += np.einsum("pq,rs->pqrs", g, np.conj(g))
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def test_single_word_bit_identical_to_forced_one_limb() -> None:
    """Single-word strings and a forced ``(M, 1)`` limb matrix agree bit for bit."""
    n_orb = 5
    adets = _enumerate_strings(n_orb, 2)
    bdets = _enumerate_strings(n_orb, 3)

    for complex_ in (False, True):
        hcore, eri = _dense_system(n_orb, seed=11, complex_=complex_)
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.123)
        dtype = torch.complex128 if complex_ else torch.float64
        integrals_t = to_torch_integrals(I0, I1, I2, dtype=dtype)
        helpers_t = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))

        adets_word = _single_word_tensor(adets)
        bdets_word = _single_word_tensor(bdets)
        adets_limb = _limb_tensor(adets, n_orb)
        bdets_limb = _limb_tensor(bdets, n_orb)
        assert adets_limb.shape == (len(adets), 1)

        hii_word = make_diag_terms_torch(adets_word, bdets_word, n_orb, integrals_t)
        hii_limb = make_diag_terms_torch(adets_limb, bdets_limb, n_orb, integrals_t)
        assert torch.equal(hii_word, hii_limb), f"complex={complex_}: hii differs"

        n_dim = len(adets) * len(bdets)
        rng = np.random.default_rng(seed=42)
        for _ in range(3):
            T_np = rng.standard_normal(n_dim)
            if complex_:
                T_np = T_np + 1j * rng.standard_normal(n_dim)
            T_t = torch.tensor(T_np, dtype=dtype)
            W_word = mult_torch(
                hii_word, T_t, adets_word, bdets_word, n_orb, integrals_t, helpers_t
            )
            W_limb = mult_torch(
                hii_limb, T_t, adets_limb, bdets_limb, n_orb, integrals_t, helpers_t
            )
            assert torch.equal(W_word, W_limb), f"complex={complex_}: matvec differs"

        if not complex_:
            hii_ref = make_diag_terms(adets, bdets, n_orb, I0, I1, I2)
            helpers_ref = list(generate_all_helpers(adets, bdets, n_orb))
            _assert_close(hii_word.numpy(), hii_ref, "single-word hii")

            T_np = rng.standard_normal(n_dim)
            W_ref = mult(hii_ref, T_np, adets, bdets, n_orb, I1, I2, helpers_ref)
            W_word = mult_torch(
                hii_word,
                torch.tensor(T_np, dtype=torch.float64),
                adets_word,
                bdets_word,
                n_orb,
                integrals_t,
                helpers_t,
            )
            _assert_close(W_word.numpy(), W_ref, "single-word matvec")


# ============================================================================
# 2. Limb oracle (n_orb = 70, 130)
# ============================================================================


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_integrals", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("which", ["alpha", "beta", "alphabeta", "all"])
def test_limb_matvec_matches_scalar_oracle(n_orb: int, complex_integrals: bool, which: str) -> None:
    """Limb hii and one matvec per task match the scalar reference to 1e-11."""
    adets, bdets = _selected_strings(n_orb, n_alpha=20, n_beta=17)
    assert max(adets) >> 64 and max(bdets) >> 64

    hcore = _spatial_hcore(n_orb, seed=7, complex_=complex_integrals)
    eri = _LowRankERI(
        _random_factors(n_orb, 2, seed=9, complex_=complex_integrals), real=not complex_integrals
    )
    dtype = torch.complex128 if complex_integrals else torch.float64

    # Torch side: same basis carried as two- or three-limb matrices.
    adets_t = _limb_tensor(adets, n_orb)
    bdets_t = _limb_tensor(bdets, n_orb)
    assert adets_t.shape[1] == _n_limbs(n_orb)
    integrals_t = _torch_integrals(n_orb, hcore, eri, dtype)
    helpers_t = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    hii_t = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals_t)

    # Scalar side: Python-int strings.
    helpers_ref = list(generate_all_helpers(adets, bdets, n_orb))
    helpers_ref = _helpers_for(helpers_ref, which)
    helpers_t_sel = _helpers_for(helpers_t, which)

    if complex_integrals:
        I0, I1_re, I2_re = _scalar_integrals(hcore, eri, part="real")
        I0_im, I1_im, I2_im = _scalar_integrals(hcore, eri, part="imag")
        hii_re = make_diag_terms(adets, bdets, n_orb, I0, I1_re, I2_re)
        hii_im = make_diag_terms(adets, bdets, n_orb, I0_im, I1_im, I2_im)
        _assert_close(hii_t.numpy(), hii_re, f"n_orb={n_orb} {which} complex hii")
    else:
        I0, I1, I2 = _scalar_integrals(hcore, eri)
        hii_ref = make_diag_terms(adets, bdets, n_orb, I0, I1, I2)
        _assert_close(hii_t.numpy(), hii_ref, f"n_orb={n_orb} {which} hii")

    n_dim = len(adets) * len(bdets)
    rng = np.random.default_rng(seed=1234)
    T_np = rng.standard_normal(n_dim)
    if complex_integrals:
        T_np = T_np + 1j * rng.standard_normal(n_dim)
    T_t = torch.tensor(T_np, dtype=dtype)

    cache = None
    if which == "beta":
        # Exercise both the cached and the inline β-pass setup.
        cache = build_beta_pass_cache(adets_t, bdets_t, n_orb, integrals_t, helpers_t_sel)
        W_cached = mult_torch(
            hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, helpers_t_sel, cache=cache
        )
        W_inline = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, helpers_t_sel)
        assert torch.equal(W_cached, W_inline), "cached and inline β pass differ"

    W_t = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, helpers_t_sel, cache=cache)

    if complex_integrals:
        W_ref = _scalar_complex_matvec(
            hii_re,
            hii_im,
            T_np,
            adets,
            bdets,
            n_orb,
            I1_re,
            I1_im,
            I2_re,
            I2_im,
            helpers_ref,
        )
    else:
        W_ref = mult(hii_ref, T_np, adets, bdets, n_orb, I1, I2, helpers_ref)

    assert W_t.dtype == dtype
    _assert_close(W_t.numpy(), W_ref, f"n_orb={n_orb} {which} matvec")


def test_limb_int32_tables_bit_identical() -> None:
    """int32 helper payloads give the same bits as int64 on a two-limb basis."""
    n_orb = 70
    adets, bdets = _selected_strings(n_orb, n_alpha=12, n_beta=10)
    hcore = _spatial_hcore(n_orb, seed=7, complex_=False)
    eri = _LowRankERI(_random_factors(n_orb, 2, seed=9, complex_=False), real=True)
    integrals_t = _torch_integrals(n_orb, hcore, eri, torch.float64)
    adets_t = _limb_tensor(adets, n_orb)
    bdets_t = _limb_tensor(bdets, n_orb)
    hii_t = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals_t)

    helpers64 = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    helpers32 = list(
        generate_all_torch_helpers(adets, bdets, n_orb, device="cpu", table_dtype=torch.int32)
    )
    T_t = torch.tensor(
        np.random.default_rng(seed=5).standard_normal(len(adets) * len(bdets)),
        dtype=torch.float64,
    )

    W64 = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, helpers64)
    W32 = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, helpers32)
    assert torch.equal(W64, W32)

    # The β-pass cache keeps the same guarantee with its narrowed tables.
    beta32 = _helpers_for(helpers32, "beta")
    cache32 = build_beta_pass_cache(adets_t, bdets_t, n_orb, integrals_t, beta32)
    W32_cached = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, beta32, cache=cache32)
    beta64 = _helpers_for(helpers64, "beta")
    W64_beta = mult_torch(hii_t, T_t, adets_t, bdets_t, n_orb, integrals_t, beta64)
    assert torch.equal(W32_cached, W64_beta)
