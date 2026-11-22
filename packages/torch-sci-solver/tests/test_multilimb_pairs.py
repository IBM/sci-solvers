"""Multi-limb (strings wider than 64 bits) support in the GDB pair-mode path.

Verifies that the pair-mode (GDB) matvec and diagonal kernels accept determinant
strings carried as ``(M, n_limbs)`` limb matrices (limb ``k`` holds bits
``[64k, 64k+64)``) end-to-end, mirroring what the TPB path already does.

1. Single-word strings (``n_orb <= 64``) carried as a forced one-limb
   ``(M, 1)`` matrix are bit-identical to the native single-word path, for real
   and complex integral dtypes; the real case also reproduces the scalar
   reference ``sbd_ref.mult`` on a dense random system.

2. Two-limb (``n_orb=70``) and three-limb (``n_orb=130``) pair bases reproduce
   the scalar reference ``sbd_ref.make_diag_terms`` and one matvec of
   ``sbd_ref.mult`` to 1e-11 for every channel: α-only, β-only, αβ-doubles, and
   all tasks together, for real and complex integral dtypes.

3. A CPU test that mixes ``(M,)`` and ``(M, n_limbs)`` inputs and verifies the
   same clear guard error as the TPB path when mismatched limb counts are given.
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
from torch_sci_solver.torch_impl import TorchIntegrals, to_torch_integrals
from torch_sci_solver.torch_impl.helpers import _n_limbs
from torch_sci_solver.torch_impl.integrals import setup_integrals_from_pyscf_h1_eri_vec
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch, mult_pair_torch
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import (
    PairBasis,
    pair_basis_from_arrays,
    pair_basis_from_full_product,
)

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


def _rows_as_ints(rows: torch.Tensor) -> list[int]:
    """Python-int strings from a 1-D bitmask or ``(M, n_limbs)`` limb tensor."""
    arr = rows.numpy()
    if arr.ndim == 1:
        return [int(v) & _MASK64 for v in arr]
    values = []
    for row in arr:
        value = 0
        for k, limb in enumerate(row):
            value |= (int(limb) & _MASK64) << (64 * k)
        values.append(value)
    return values


def _pair_permutation(pair_basis: PairBasis, adets: list[int], bdets: list[int]) -> np.ndarray:
    """Map each pair-basis index to its ``ia * M_b + ib`` product-mode index."""
    alpha_index = {int(v): i for i, v in enumerate(adets)}
    beta_index = {int(v): i for i, v in enumerate(bdets)}
    m_beta = len(bdets)
    a_values = _rows_as_ints(pair_basis.alpha_of_pair)
    b_values = _rows_as_ints(pair_basis.beta_of_pair)
    return np.array(
        [alpha_index[a] * m_beta + beta_index[b] for a, b in zip(a_values, b_values)],
        dtype=np.int64,
    )


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

    def element_size(self) -> int:
        """Byte width of one ERI entry, as ``Tensor.element_size`` reports it."""
        return torch.empty((), dtype=self.dtype).element_size()


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


class _ChannelHelpers:
    """PairHelpers proxy exposing only the streamed channels under test.

    ``mult_pair_torch`` consumes every stream the helpers expose; restricting
    the proxy to one channel is how a single-channel matvec is compared with
    the matching subset of SBD task helpers.
    """

    _METHODS = {
        "alpha": ("asingles", "adoubles"),
        "beta": ("bsingles", "bdoubles"),
        "alphabeta": ("ab_doubles",),
    }

    def __init__(self, helpers, which: str) -> None:
        """Keep the channels named by ``which`` or all five for ``"all"``."""
        object.__setattr__(self, "_helpers", helpers)
        channels = self._METHODS.values() if which == "all" else (self._METHODS[which],)
        keep = {name for names in channels for name in names}
        object.__setattr__(self, "_keep", keep)

    def __getattr__(self, name: str):
        """Delegate every non-stream attribute to the real helpers."""
        return getattr(self._helpers, name)

    def _stream(self, name: str, pair_range=None):
        if name not in self._keep:
            return iter(())
        return getattr(self._helpers, name)(pair_range)

    def asingles(self, pair_range=None):
        """α singles when the α channel is enabled, else an empty stream."""
        return self._stream("asingles", pair_range)

    def adoubles(self, pair_range=None):
        """α doubles when the α channel is enabled, else an empty stream."""
        return self._stream("adoubles", pair_range)

    def bsingles(self, pair_range=None):
        """β singles when the β channel is enabled, else an empty stream."""
        return self._stream("bsingles", pair_range)

    def bdoubles(self, pair_range=None):
        """β doubles when the β channel is enabled, else an empty stream."""
        return self._stream("bdoubles", pair_range)

    def ab_doubles(self, pair_range=None):
        """αβ doubles when that channel is enabled, else an empty stream."""
        return self._stream("ab_doubles", pair_range)


def _helpers_for(helpers: list, which: str) -> list:
    """Select the task helper(s) that ``which`` names."""
    if which == "all":
        return list(helpers)
    task_type = {"alpha": 2, "beta": 1, "alphabeta": 0}[which]
    return [h for h in helpers if h.task_type == task_type]


def _scalar_complex_matvec(
    T: np.ndarray,
    adets: list[int],
    bdets: list[int],
    n_orb: int,
    I0_re,
    I1_re,
    I2_re,
    I0_im,
    I1_im,
    I2_im,
    helpers: list,
) -> np.ndarray:
    """Scalar matvec oracle for complex integrals via the real/imag split.

    ``H(I_real + i I_imag)`` is linear in the integrals with real
    coefficients, so the real reference evaluates each part separately:
    ``H(conj I) = H(I_real) - i H(I_imag)``, which is the operator the fast
    kernels build (their chemist pairs lead with the creation index).
    """
    X = T.real
    Y = T.imag
    hii_re = make_diag_terms(adets, bdets, n_orb, I0_re, I1_re, I2_re)
    hii_im = make_diag_terms(adets, bdets, n_orb, I0_im, I1_im, I2_im)

    AX = mult(hii_re, X, adets, bdets, n_orb, I1_re, I2_re, helpers)
    BX = mult(hii_im, X, adets, bdets, n_orb, I1_im, I2_im, helpers)
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
# 1. Single-word regression (bit-identical with forced one-limb)
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


def test_pair_single_word_bit_identical_to_forced_one_limb() -> None:
    """Single-word strings and a forced ``(M, 1)`` limb matrix agree bit for bit in pair mode."""
    n_orb = 5
    adets = _enumerate_strings(n_orb, 2)
    bdets = _enumerate_strings(n_orb, 3)

    for complex_ in (False, True):
        hcore, eri = _dense_system(n_orb, seed=11, complex_=complex_)
        I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.123)
        dtype = torch.complex128 if complex_ else torch.float64
        integrals_t = to_torch_integrals(I0, I1, I2, dtype=dtype)

        # Python-int input keeps the legacy 1-D pair-basis representation.
        pair_basis_word = pair_basis_from_full_product(adets, bdets, n_orb=n_orb, device="cpu")
        assert pair_basis_word.alpha_of_pair.dim() == 1

        # A forced one-limb ``(M, 1)`` input keeps the limb axis.
        adets_limb = _limb_tensor(adets, n_orb)
        bdets_limb = _limb_tensor(bdets, n_orb)
        assert adets_limb.shape == (len(adets), 1)
        pair_basis_limb = pair_basis_from_full_product(
            adets_limb.numpy().view(np.int64),
            bdets_limb.numpy().view(np.int64),
            n_orb=n_orb,
            device="cpu",
        )
        assert pair_basis_limb.alpha_of_pair.shape == (pair_basis_word.n_pairs, 1)

        # Diagonal terms
        hii_word = make_diag_terms_pair_torch(pair_basis_word, n_orb, integrals_t)
        hii_limb = make_diag_terms_pair_torch(pair_basis_limb, n_orb, integrals_t)
        assert torch.equal(hii_word, hii_limb), f"complex={complex_}: hii differs"

        # Matvec
        helpers_word = generate_pair_helpers(pair_basis_word, n_orb, device="cpu")
        helpers_limb = generate_pair_helpers(pair_basis_limb, n_orb, device="cpu")
        n_dim = pair_basis_word.n_pairs
        rng = np.random.default_rng(seed=42)
        permutation = _pair_permutation(pair_basis_word, adets, bdets)
        assert np.array_equal(permutation, np.arange(n_dim))
        for _ in range(3):
            T_np = rng.standard_normal(n_dim)
            if complex_:
                T_np = T_np + 1j * rng.standard_normal(n_dim)
            T_t = torch.tensor(T_np, dtype=dtype)
            W_word = mult_pair_torch(
                hii_word, T_t, pair_basis_word, helpers_word, n_orb, integrals_t
            )
            W_limb = mult_pair_torch(
                hii_limb, T_t, pair_basis_limb, helpers_limb, n_orb, integrals_t
            )
            assert torch.equal(W_word, W_limb), f"complex={complex_}: matvec differs"

        if not complex_:
            # Compare to scalar reference
            hii_ref = make_diag_terms(adets, bdets, n_orb, I0, I1, I2)
            helpers_ref = list(generate_all_helpers(adets, bdets, n_orb))
            _assert_close(hii_word.numpy(), hii_ref, "single-word pair hii")

            T_np = rng.standard_normal(n_dim)
            W_ref = mult(hii_ref, T_np, adets, bdets, n_orb, I1, I2, helpers_ref)
            W_word = mult_pair_torch(
                hii_word,
                torch.tensor(T_np, dtype=torch.float64),
                pair_basis_word,
                helpers_word,
                n_orb,
                integrals_t,
            )
            _assert_close(W_word.numpy(), W_ref, "single-word pair matvec")


# ============================================================================
# 2. Limb oracle (n_orb = 70, 130)
# ============================================================================


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_integrals", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("which", ["alpha", "beta", "alphabeta", "all"])
def test_pair_limb_matvec_matches_scalar_oracle(
    n_orb: int, complex_integrals: bool, which: str
) -> None:
    """Limb hii and one matvec per task match the scalar reference to 1e-11 in pair mode."""
    adets, bdets = _selected_strings(n_orb, n_alpha=20, n_beta=17)
    assert max(adets) >> 64 and max(bdets) >> 64

    hcore = _spatial_hcore(n_orb, seed=7, complex_=complex_integrals)
    eri = _LowRankERI(
        _random_factors(n_orb, 2, seed=9, complex_=complex_integrals), real=not complex_integrals
    )
    dtype = torch.complex128 if complex_integrals else torch.float64

    # Torch side: Python-int lists become a two- or three-limb pair basis.
    pair_basis = pair_basis_from_full_product(adets, bdets, n_orb=n_orb, device="cpu")
    assert pair_basis.n_limbs == _n_limbs(n_orb)
    assert pair_basis.alpha_of_pair.dim() == 2
    assert pair_basis.n_pairs == len(adets) * len(bdets)
    permutation = _pair_permutation(pair_basis, adets, bdets)
    assert np.array_equal(np.sort(permutation), np.arange(pair_basis.n_pairs))

    integrals_t = _torch_integrals(n_orb, hcore, eri, dtype)
    helpers_t = generate_pair_helpers(pair_basis, n_orb, device="cpu")
    hii_t = make_diag_terms_pair_torch(pair_basis, n_orb, integrals_t)

    # Scalar side: Python-int strings on the full Cartesian product.
    helpers_ref = _helpers_for(list(generate_all_helpers(adets, bdets, n_orb)), which)
    if complex_integrals:
        I0_re, I1_re, I2_re = _scalar_integrals(hcore, eri, part="real")
        I0_im, I1_im, I2_im = _scalar_integrals(hcore, eri, part="imag")
        hii_ref = make_diag_terms(adets, bdets, n_orb, I0_re, I1_re, I2_re)
    else:
        I0, I1, I2 = _scalar_integrals(hcore, eri)
        hii_ref = make_diag_terms(adets, bdets, n_orb, I0, I1, I2)
    _assert_close(hii_t.numpy(), hii_ref[permutation], f"n_orb={n_orb} {which} pair hii")

    n_dim = pair_basis.n_pairs
    rng = np.random.default_rng(seed=1234)
    T_ref = rng.standard_normal(n_dim)
    if complex_integrals:
        T_ref = T_ref + 1j * rng.standard_normal(n_dim)
    T_t = torch.tensor(T_ref[permutation], dtype=dtype)

    W_t = mult_pair_torch(
        hii_t, T_t, pair_basis, _ChannelHelpers(helpers_t, which), n_orb, integrals_t
    )

    if complex_integrals:
        W_ref = _scalar_complex_matvec(
            T_ref,
            adets,
            bdets,
            n_orb,
            I0_re,
            I1_re,
            I2_re,
            I0_im,
            I1_im,
            I2_im,
            helpers_ref,
        )
    else:
        W_ref = mult(hii_ref, T_ref, adets, bdets, n_orb, I1, I2, helpers_ref)

    assert W_t.dtype == dtype
    _assert_close(W_t.numpy(), W_ref[permutation], f"n_orb={n_orb} {which} pair matvec")


def test_pair_limb_nonproduct_subset_matches_product() -> None:
    """A genuinely non-product limb pair basis agrees with the full-product matvec."""
    n_orb = 70
    adets, bdets = _selected_strings(n_orb, n_alpha=12, n_beta=10)
    hcore = _spatial_hcore(n_orb, seed=7, complex_=False)
    eri = _LowRankERI(_random_factors(n_orb, 2, seed=9, complex_=False), real=True)
    integrals_t = _torch_integrals(n_orb, hcore, eri, torch.float64)

    full_basis = pair_basis_from_full_product(adets, bdets, n_orb=n_orb, device="cpu")
    full_perm = _pair_permutation(full_basis, adets, bdets)
    n_full = full_basis.n_pairs
    prod_to_full = np.empty(n_full, dtype=np.int64)
    prod_to_full[full_perm] = np.arange(n_full)

    # Keep roughly half the pairs, offset per α so the subset is non-product.
    keep = []
    for i in range(len(adets)):
        for j in range(len(bdets)):
            if (i + j) % 2 == 0 or j == (3 * i) % len(bdets):
                keep.append((adets[i], bdets[j]))
    sub_a = [pair[0] for pair in keep]
    sub_b = [pair[1] for pair in keep]
    sub_basis = pair_basis_from_arrays(sub_a, sub_b, n_orb=n_orb, device="cpu")
    assert sub_basis.n_limbs == 2
    # Full pair-basis index of each subset pair.
    sub_idx = prod_to_full[_pair_permutation(sub_basis, adets, bdets)]

    # Nonzero amplitudes only on the subset: the full matvec restricted to the
    # subset rows is then exactly the subset Hamiltonian's matvec.
    T_full = np.zeros(n_full)
    T_full[sub_idx] = np.random.default_rng(seed=17).standard_normal(sub_idx.size)

    hii_full = make_diag_terms_pair_torch(full_basis, n_orb, integrals_t)
    helpers_full = generate_pair_helpers(full_basis, n_orb, device="cpu")
    W_full = mult_pair_torch(
        hii_full,
        torch.tensor(T_full, dtype=torch.float64),
        full_basis,
        helpers_full,
        n_orb,
        integrals_t,
    )

    hii_sub = make_diag_terms_pair_torch(sub_basis, n_orb, integrals_t)
    helpers_sub = generate_pair_helpers(sub_basis, n_orb, device="cpu")
    W_sub = mult_pair_torch(
        hii_sub,
        torch.tensor(T_full[sub_idx], dtype=torch.float64),
        sub_basis,
        helpers_sub,
        n_orb,
        integrals_t,
    )

    _assert_close(hii_sub.numpy(), hii_full.numpy()[sub_idx], "non-product subset hii")
    _assert_close(W_sub.numpy(), W_full.numpy()[sub_idx], "non-product subset matvec")


def test_pair_limb_block_matvec_matches_streamed() -> None:
    """The block-decomposed limb matvec matches the streamed one on a Cartesian basis."""
    n_orb = 70
    adets, bdets = _selected_strings(n_orb, n_alpha=8, n_beta=7)
    pa = [a for a in adets for _ in bdets]
    pb = [b for _ in adets for b in bdets]
    basis = pair_basis_from_arrays(pa, pb, n_orb=n_orb, device="cpu")
    assert basis.n_limbs == 2

    hcore = _spatial_hcore(n_orb, seed=7, complex_=False)
    eri = _LowRankERI(_random_factors(n_orb, 2, seed=9, complex_=False), real=True)
    integrals_t = _torch_integrals(n_orb, hcore, eri, torch.float64)
    hii = make_diag_terms_pair_torch(basis, n_orb, integrals_t)
    T = torch.tensor(
        np.random.default_rng(seed=1).standard_normal(basis.n_pairs), dtype=torch.float64
    )

    streamed = generate_pair_helpers(basis, n_orb, device="cpu")
    blocked = generate_pair_helpers(basis, n_orb, device="cpu", block_matvec="on")
    assert blocked.blocks is not None
    W_streamed = mult_pair_torch(hii, T, basis, streamed, n_orb, integrals_t)
    W_blocked = mult_pair_torch(hii, T, basis, blocked, n_orb, integrals_t)
    _assert_close(W_blocked.numpy(), W_streamed.numpy(), "block vs streamed limb matvec")


# ============================================================================
# 3. Mixed input guard (same clear error as TPB path)
# ============================================================================


def test_pair_mixed_limb_guard() -> None:
    """Mixing (M,) and (M, n_limbs) inputs raises the same clear guard error as TPB."""
    n_orb = 70
    adets, bdets = _selected_strings(n_orb, n_alpha=10, n_beta=8)
    pa = [pair[0] for pair in zip(adets, bdets)]
    pb = [pair[1] for pair in zip(adets, bdets)]

    # Python-int strings are converted to limbs for a wide basis.
    pair_basis = pair_basis_from_arrays(pa, pb, n_orb=n_orb, device="cpu")
    assert pair_basis.alpha_of_pair.dim() == 2
    assert pair_basis.n_limbs == _n_limbs(n_orb)

    # One spin as a 1-D string array, the other as a 2-D limb matrix.
    pa_mixed = pa[:5]
    pb_mixed = _limb_tensor(pb[:5], n_orb)
    with pytest.raises(ValueError, match="mix a 1-D bitmask array with a 2-D limb matrix"):
        pair_basis_from_arrays(pa_mixed, pb_mixed, n_orb=n_orb, device="cpu")

    # A limb matrix with the wrong width is rejected with the TPB guard message.
    wrong = np.zeros((5, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="limb matrix has 3 limbs, expected 2 for n_orb=70"):
        pair_basis_from_arrays(wrong, wrong, n_orb=n_orb, device="cpu")


def test_pair_basis_from_arrays_infers_n_orb() -> None:
    """pair_basis_from_arrays infers n_orb from max bit position when not given."""
    pa = [1 << 66 | 1 << 2, 1 << 66 | 1 << 5]  # bits 2, 5, 66
    pb = [1 << 67 | 1 << 3, 1 << 67 | 1 << 4]  # bits 3, 4, 67

    # Without n_orb, should infer from highest bit (67 -> n_orb=68)
    pair_basis = pair_basis_from_arrays(pa, pb, device="cpu")
    # n_orb inferred as 68 -> 2 limbs
    assert pair_basis.n_limbs == 2

    # With explicit n_orb=70 -> 2 limbs
    pair_basis2 = pair_basis_from_arrays(pa, pb, n_orb=70, device="cpu")
    assert pair_basis2.n_limbs == 2

    # With n_orb=130 -> 3 limbs
    pair_basis3 = pair_basis_from_arrays(pa, pb, n_orb=130, device="cpu")
    assert pair_basis3.n_limbs == 3


def test_pair_basis_limb_properties() -> None:
    """PairBasis.n_limbs property reflects the limb count correctly."""
    n_orb = 5  # 1 limb
    adets = _enumerate_strings(n_orb, 2)
    bdets = _enumerate_strings(n_orb, 3)
    pair_basis = pair_basis_from_arrays(adets, bdets, n_orb=n_orb, device="cpu")
    assert pair_basis.n_limbs == 1
    assert pair_basis.alpha_of_pair.dim() == 1

    n_orb = 70  # 2 limbs
    pa = [1 << 66 | 3, 1 << 66 | 5]
    pb = [1 << 67 | 3, 1 << 67 | 4]
    pair_basis2 = pair_basis_from_arrays(pa, pb, n_orb=n_orb, device="cpu")
    assert pair_basis2.n_limbs == 2
    assert pair_basis2.alpha_of_pair.dim() == 2
    assert pair_basis2.alpha_of_pair.shape[1] == 2
