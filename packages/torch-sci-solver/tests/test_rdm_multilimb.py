"""Multi-limb (strings wider than 64 bits) support in the product-mode RDM builders.

The product-mode builders in ``torch_impl/rdm.py`` used to do single-word bit
operations (``strings.unsqueeze(1) >> torch.arange(n_orb)``) and rejected
``L > 64``. They now accept the same ``(M, n_limbs)`` limb matrices as the
determinant primitives (limb ``k`` holds bits ``[64k, 64k + 64)``).

1. An 8-orbital basis whose strings are padded to ``(M, 2)`` limbs gives
   bit-identical 1-RDMs, αα/αβ/ββ 2-RDMs and the full 2-RDM to the same basis
   in 1-D form, for real and complex amplitudes.

2. On a small basis the builders match the explicit serial builders in
   ``torch_impl/brute_rdm.py`` (real amplitudes) and a from-scratch
   second-quantized reference that conjugates the bra coefficient (complex
   amplitudes).

3. At 70 and 130 orbitals a selected product basis reproduces the same
   arbitrary-precision Python-int reference, including the spin-summed
   ``(L, L, L, L)`` 2-RDM and ``<S^2>`` from the αβ block.

The dense ``(L, L, L, L)`` RDM is the cost driver at wide ``L`` (no ERI is
built here), so the 130-orbital block test runs one block at a time and is
marked slow.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import gc

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.brute_rdm import (
    brute_one_rdm,
    brute_two_rdm_aa,
    brute_two_rdm_ab,
    brute_two_rdm_bb,
)
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.rdm import (
    make_one_rdm_diag_torch,
    make_one_rdm_torch,
    make_two_rdm_aa_torch,
    make_two_rdm_ab_torch,
    make_two_rdm_bb_torch,
    make_two_rdm_torch,
    s_squared_from_rdm,
)

_MASK64 = (1 << 64) - 1


# ============================================================================
# Basis construction
# ============================================================================


def _small_strings() -> list[int]:
    """Six 2-electron strings over 8 orbitals: HF, four singles and a double."""
    base = 0b0011  # orbitals 0, 1
    strings = {base, (1 << 2) | (1 << 3)}
    for i in (0, 1):
        for j in (2, 3):
            strings.add((base ^ (1 << i)) | (1 << j))
    return sorted(strings)


def _small_basis() -> tuple[list[int], list[int]]:
    """An open-shell selected basis: six α strings (2e) and three β ones (1e)."""
    return _small_strings(), [1 << 1, 1 << 2, 1 << 4]


def _wide_strings(n_orb: int) -> list[int]:
    """Six 2-electron strings that straddle the 64-bit limb boundary.

    For ``n_orb=70`` the occupied orbitals are 0/1 (limb 0) and 64/65 (limb
    1); for ``n_orb=130`` the high orbitals are 64 and 128, so all three
    limbs carry bits.
    """
    high = 128 if n_orb > 128 else 65
    base = (1 << 0) | (1 << 64)
    strings = {base, (1 << 1) | (1 << high)}
    for i in (0, 64):
        for j in (1, high):
            strings.add((base ^ (1 << i)) | (1 << j))
    return sorted(strings)


def _n_limbs(n_orb: int) -> int:
    """Number of int64 limbs a per-spin string needs for ``n_orb`` orbitals."""
    return max(1, (int(n_orb) + 63) // 64)


def _signed_int64(word: int) -> int:
    """One unsigned 64-bit limb as the signed int64 bit pattern torch stores."""
    return word - (1 << 64) if word >= (1 << 63) else word


def _limb_array(strings: list[int], n_orb: int) -> np.ndarray:
    """``(M, n_limbs)`` signed-int64 limb matrix of arbitrary-precision strings."""
    n_limbs = _n_limbs(n_orb)
    out = np.zeros((len(strings), n_limbs), dtype=np.int64)
    for k in range(n_limbs):
        out[:, k] = [_signed_int64((int(v) >> (64 * k)) & _MASK64) for v in strings]
    return out


def _two_limb_words(strings: list[int]) -> torch.Tensor:
    """Pad 8-orbital strings to ``(M, 2)`` int64 limb matrices.

    Limb 0 holds the real bits and limb 1 is zero padding, which exercises the
    limb dispatch on a basis that also fits in one word.
    """
    arr = _limb_array(strings, 64)
    padded = np.zeros((len(strings), 2), dtype=np.int64)
    padded[:, 0] = arr[:, 0]
    return torch.from_numpy(padded)


def _one_word_tensor(strings: list[int]) -> torch.Tensor:
    """``(M,)`` int64 tensor of strings that fit in one word."""
    return torch.tensor([_signed_int64(v) for v in strings], dtype=torch.int64)


# ============================================================================
# Interleaved-Wick → spin-block bridge (arbitrary-precision Python ints)
# ============================================================================


def _popcount(x: int) -> int:
    """Population count of an arbitrary-precision Python int."""
    return bin(x).count("1")


def _bits(mask: int) -> list[int]:
    """Ascending bit positions of a Python-int mask."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _single_bit(x: int) -> int:
    """Position of the only set bit of ``x``."""
    return x.bit_length() - 1


def _sigma(alpha: int, beta: int, n_orb: int) -> float:
    """Interleaved-Wick → spin-block sign for one determinant (Python ints)."""
    exponent = 0
    for j in range(n_orb):
        if (alpha >> j) & 1:
            exponent += _popcount(beta & ((1 << j) - 1))
    return -1.0 if exponent & 1 else 1.0


def _spin_block_coeffs(
    eigvec: torch.Tensor, adets: list[int], bdets: list[int], n_orb: int
) -> np.ndarray:
    """Interleaved-Wick amplitudes converted to the spin-block convention."""
    flat_alphas = [a for a in adets for _ in bdets]
    flat_betas = [b for _ in adets for b in bdets]
    signs = np.array([_sigma(a, b, n_orb) for a, b in zip(flat_alphas, flat_betas)])
    return signs * eigvec.detach().numpy()


# ============================================================================
# Brute-force reference: direct second-quantized matrix elements
# ============================================================================


def _apply_ops(
    ops: tuple[tuple[int, bool, int], ...], alpha: int, beta: int
) -> tuple[int, int, int] | None:
    """Apply ``ops`` right-to-left to a spin-block determinant.

    ``ops`` is a list of ``(spin, create, orbital)`` with spin 0 = α, 1 = β.
    Returns ``(alpha, beta, sign)``, or ``None`` when an operator annihilates
    an empty orbital or creates on an occupied one. Signs are counted over the
    acting spin's own string, i.e. in the spin-block ordering.
    """
    sign = 1
    a, b = alpha, beta
    for spin, create, orbital in reversed(ops):
        word = a if spin == 0 else b
        bit = 1 << orbital
        if create == bool(word & bit):
            return None
        if _popcount(word & (bit - 1)) & 1:
            sign = -sign
        word ^= bit
        if spin == 0:
            a = word
        else:
            b = word
    return a, b, sign


class _SerialRDMs:
    """Explicit serial 1-RDM/2-RDM of a product-basis state.

    Evaluated one determinant pair at a time from arbitrary-precision Python
    ints. The bra coefficient is conjugated, so the same reference serves real
    and complex amplitudes.
    """

    def __init__(self, alphas: list[int], betas: list[int], coeffs: np.ndarray, n_orb: int) -> None:
        """Evaluate every matrix element of the documented PySCF convention."""
        self.n_orb = n_orb
        self.n_alpha = _popcount(alphas[0])
        self.n_beta = _popcount(betas[0])
        self.real = not np.iscomplexobj(coeffs)
        dtype = np.float64 if self.real else np.complex128
        L = n_orb
        self.ga = np.zeros((L, L), dtype=dtype)
        self.gb = np.zeros((L, L), dtype=dtype)
        # Sparse 2-RDM blocks: the state spans a few dozen determinants, so
        # only O(N²) entries are ever nonzero.
        self.gaa: dict[tuple[int, int, int, int], complex] = {}
        self.gab: dict[tuple[int, int, int, int], complex] = {}
        self.gbb: dict[tuple[int, int, int, int], complex] = {}

        def add(table, key, value):
            table[key] = table.get(key, 0.0) + value

        n = len(alphas)
        for j in range(n):
            aj, bj, cj = alphas[j], betas[j], coeffs[j]
            for k in range(n):
                ak, bk = alphas[k], betas[k]
                weight = np.conj(coeffs[k]) * cj
                if weight == 0.0:
                    continue
                # γ_α[p, q] = ⟨a†_p a_q⟩: q is occupied in the ket and p is
                # empty after removing it (p may equal q — the diagonal).
                if bk == bj:
                    for q in _bits(aj):
                        rest = aj ^ (1 << q)
                        if (rest & ~ak) or _popcount(ak & ~rest) != 1:
                            continue
                        p = _single_bit(ak & ~rest)
                        res = _apply_ops(((0, True, p), (0, False, q)), aj, bj)
                        if res is not None and res[:2] == (ak, bk):
                            self.ga[p, q] += weight * res[2]
                if ak == aj:
                    for q in _bits(bj):
                        rest = bj ^ (1 << q)
                        if (rest & ~bk) or _popcount(bk & ~rest) != 1:
                            continue
                        p = _single_bit(bk & ~rest)
                        res = _apply_ops(((1, True, p), (1, False, q)), aj, bj)
                        if res is not None and res[:2] == (ak, bk):
                            self.gb[p, q] += weight * res[2]
                # Γ_αα[p, q, r, s] = ⟨a†_p a†_r a_s a_q⟩: q, s are occupied in
                # the ket; after removing them the two created orbitals are
                # fixed by the bra, and p/r are their two orderings. Slots may
                # repeat (p = q recreates the annihilated orbital).
                if bk == bj:
                    for q in _bits(aj):
                        for s in _bits(aj):
                            if s == q:
                                continue
                            rest = aj ^ (1 << q) ^ (1 << s)
                            if (rest & ~ak) or _popcount(ak & ~rest) != 2:
                                continue
                            created = _bits(ak & ~rest)
                            for p in created:
                                for r in created:
                                    if r == p:
                                        continue
                                    res = _apply_ops(
                                        ((0, True, p), (0, True, r), (0, False, s), (0, False, q)),
                                        aj,
                                        bj,
                                    )
                                    if res is not None and res[:2] == (ak, bk):
                                        add(self.gaa, (p, q, r, s), weight * res[2])
                if ak == aj:
                    for q in _bits(bj):
                        for s in _bits(bj):
                            if s == q:
                                continue
                            rest = bj ^ (1 << q) ^ (1 << s)
                            if (rest & ~bk) or _popcount(bk & ~rest) != 2:
                                continue
                            created = _bits(bk & ~rest)
                            for p in created:
                                for r in created:
                                    if r == p:
                                        continue
                                    res = _apply_ops(
                                        ((1, True, p), (1, True, r), (1, False, s), (1, False, q)),
                                        aj,
                                        bj,
                                    )
                                    if res is not None and res[:2] == (ak, bk):
                                        add(self.gbb, (p, q, r, s), weight * res[2])
                # Γ_αβ[p, q, r, s] = ⟨a†_pα a†_rβ a_sβ a_qα⟩: one α and one β
                # move, each with the same one-orbital choice as the 1-RDM.
                for q in _bits(aj):
                    rest_a = aj ^ (1 << q)
                    if (rest_a & ~ak) or _popcount(ak & ~rest_a) != 1:
                        continue
                    p = _single_bit(ak & ~rest_a)
                    for s in _bits(bj):
                        rest_b = bj ^ (1 << s)
                        if (rest_b & ~bk) or _popcount(bk & ~rest_b) != 1:
                            continue
                        r = _single_bit(bk & ~rest_b)
                        res = _apply_ops(
                            ((0, True, p), (1, True, r), (1, False, s), (0, False, q)), aj, bj
                        )
                        if res is not None and res[:2] == (ak, bk):
                            add(self.gab, (p, q, r, s), weight * res[2])

    def s_squared(self) -> float:
        """⟨S²⟩ = (S_z² − S_z) + N_α − Σ_{p,q} Γ_αβ[p, q, q, p]."""
        n_a, n_b = self.n_alpha, self.n_beta
        s_z = 0.5 * (n_a - n_b)
        cross = sum(v for (p, q, r, s), v in self.gab.items() if q == r and s == p)
        return float((s_z * s_z - s_z + n_a - cross).real)

    def dense(self, table: dict[tuple[int, int, int, int], complex]) -> np.ndarray:
        """A dense ``(L, L, L, L)`` array from one sparse block."""
        L = self.n_orb
        dtype = np.float64 if self.real else np.complex128
        arr = np.zeros((L, L, L, L), dtype=dtype)
        for key, value in table.items():
            arr[key] = value
        return arr

    def spin_summed_two_rdm(self) -> dict[tuple[int, int, int, int], complex]:
        """Sparse PySCF spin-summed 2-RDM ``Gaa + Gab + Gab^{(r,s,p,q)} + Gbb``."""
        out: dict[tuple[int, int, int, int], complex] = {}

        def add(key, value):
            out[key] = out.get(key, 0.0) + value

        for key, value in self.gaa.items():
            add(key, value)
        for key, value in self.gbb.items():
            add(key, value)
        for (p, q, r, s), value in self.gab.items():
            add((p, q, r, s), value)
            add((r, s, p, q), value)
        return out


def _compare_sparse(
    got: torch.Tensor,
    table: dict[tuple[int, int, int, int], complex],
    n_orb: int,
    atol: float = 1e-11,
) -> None:
    """Check a dense 2-RDM lookup against a sparse Python-int reference."""
    nonzero = got.view(-1).nonzero().squeeze(1)
    seen = set()
    for flat in nonzero.tolist():
        key = (
            flat // (n_orb**3),
            (flat // (n_orb**2)) % n_orb,
            (flat // n_orb) % n_orb,
            flat % n_orb,
        )
        seen.add(key)
        got_value = complex(got[key].item())
        want_value = complex(table.get(key, 0.0))
        assert abs(got_value.real - want_value.real) < atol
        assert abs(got_value.imag - want_value.imag) < atol
    # Every reference entry large enough to matter is in the dense block.
    for key, value in table.items():
        if abs(value) > 1e-10:
            assert key in seen, f"missing reference entry {key}"
            assert abs(complex(got[key].item()) - complex(value)) < atol


# ============================================================================
# 1. Small basis: padded 2-limb strings are bit-identical to the 1-D form
# ============================================================================


@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_two_limb_small_basis_bit_identical(complex_coeffs: bool) -> None:
    """8-orbital basis carried as 2 limbs matches the 1-D basis bit for bit."""
    n_orb = 8
    adets, bdets = _small_basis()
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)

    m_a, m_b = len(adets), len(bdets)
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(m_a * m_b, generator=gen, dtype=torch.float64)
    if complex_coeffs:
        eigvec = real + 1j * torch.randn(m_a * m_b, generator=gen, dtype=torch.float64)
    else:
        eigvec = real

    adets_1d, bdets_1d = _one_word_tensor(adets), _one_word_tensor(bdets)
    adets_limb, bdets_limb = _two_limb_words(adets), _two_limb_words(bdets)
    assert adets_limb.shape == (m_a, 2)

    diag_1d = make_one_rdm_diag_torch(eigvec, adets_1d, bdets_1d, n_orb)
    diag_limb = make_one_rdm_diag_torch(eigvec, adets_limb, bdets_limb, n_orb)
    for got, want in zip(diag_limb, diag_1d):
        assert torch.equal(got, want)

    ga_1d, gb_1d = make_one_rdm_torch(eigvec, adets_1d, bdets_1d, n_orb, helpers)
    ga_limb, gb_limb = make_one_rdm_torch(eigvec, adets_limb, bdets_limb, n_orb, helpers)
    assert torch.equal(ga_limb, ga_1d)
    assert torch.equal(gb_limb, gb_1d)

    for builder in (make_two_rdm_aa_torch, make_two_rdm_ab_torch, make_two_rdm_bb_torch):
        got = builder(eigvec, adets_limb, bdets_limb, n_orb, helpers)
        want = builder(eigvec, adets_1d, bdets_1d, n_orb, helpers)
        assert torch.equal(got, want)

    rdm_1d = make_two_rdm_torch(eigvec, adets_1d, bdets_1d, n_orb, helpers)
    rdm_limb = make_two_rdm_torch(eigvec, adets_limb, bdets_limb, n_orb, helpers)
    for got, want in zip(rdm_limb, rdm_1d):
        assert torch.equal(got, want)

    # The interleaved-Wick → spin-block bridge must also agree between forms.
    ga_1d_b, _ = make_one_rdm_torch(
        eigvec, adets_1d, bdets_1d, n_orb, helpers, eigvec_in_interleaved_wick=False
    )
    ga_limb_b, _ = make_one_rdm_torch(
        eigvec, adets_limb, bdets_limb, n_orb, helpers, eigvec_in_interleaved_wick=False
    )
    assert torch.equal(ga_limb_b, ga_1d_b)


# ============================================================================
# 2. Small basis: explicit serial builders, real and complex
# ============================================================================


def _small_case(
    complex_coeffs: bool,
) -> tuple[int, list[int], list[int], torch.Tensor, "_SerialRDMs"]:
    """Small selected basis, random amplitudes, and the spin-block reference."""
    n_orb = 8
    adets, bdets = _small_basis()
    gen = torch.Generator().manual_seed(13)
    real = torch.randn(len(adets) * len(bdets), generator=gen, dtype=torch.float64)
    if complex_coeffs:
        eigvec = real + 1j * torch.randn(
            len(adets) * len(bdets), generator=gen, dtype=torch.float64
        )
    else:
        eigvec = real
    eigvec = eigvec / torch.linalg.norm(eigvec)
    coeffs = _spin_block_coeffs(eigvec, adets, bdets, n_orb)
    flat_alphas = [a for a in adets for _ in bdets]
    flat_betas = [b for _ in adets for b in bdets]
    reference = _SerialRDMs(flat_alphas, flat_betas, coeffs, n_orb)
    return n_orb, adets, bdets, eigvec, reference


@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_small_basis_matches_serial_reference(complex_coeffs: bool) -> None:
    """Builders match the explicit second-quantized reference in both bases."""
    n_orb, adets, bdets, eigvec, reference = _small_case(complex_coeffs)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = _one_word_tensor(adets)
    bdets_t = _one_word_tensor(bdets)

    # Default: eigvec is interleaved-Wick, builders apply σ internally.
    ga, gb = make_one_rdm_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    np.testing.assert_allclose(ga.numpy(), reference.ga, atol=1e-12)
    np.testing.assert_allclose(gb.numpy(), reference.gb, atol=1e-12)

    eigvec_block = torch.tensor(_spin_block_coeffs(eigvec, adets, bdets, n_orb), dtype=eigvec.dtype)
    ga_b, gb_b = make_one_rdm_torch(
        eigvec_block, adets_t, bdets_t, n_orb, helpers, eigvec_in_interleaved_wick=False
    )
    np.testing.assert_allclose(ga_b.numpy(), reference.ga, atol=1e-12)
    np.testing.assert_allclose(gb_b.numpy(), reference.gb, atol=1e-12)

    builders = (
        (make_two_rdm_aa_torch, "gaa"),
        (make_two_rdm_ab_torch, "gab"),
        (make_two_rdm_bb_torch, "gbb"),
    )
    for builder, name in builders:
        table = getattr(reference, name)
        got = builder(eigvec, adets_t, bdets_t, n_orb, helpers)
        np.testing.assert_allclose(got.numpy(), reference.dense(table), atol=1e-12)
        got_b = builder(
            eigvec_block, adets_t, bdets_t, n_orb, helpers, eigvec_in_interleaved_wick=False
        )
        np.testing.assert_allclose(got_b.numpy(), reference.dense(table), atol=1e-12)
        del got, got_b


def test_small_basis_matches_brute_rdm_real() -> None:
    """Real amplitudes match the explicit serial builders in ``brute_rdm``."""
    n_orb, adets, bdets, eigvec, _reference = _small_case(complex_coeffs=False)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = _one_word_tensor(adets)
    bdets_t = _one_word_tensor(bdets)
    coeffs = _spin_block_coeffs(eigvec, adets, bdets, n_orb)

    ga, gb = make_one_rdm_torch(
        torch.tensor(coeffs), adets_t, bdets_t, n_orb, helpers, eigvec_in_interleaved_wick=False
    )
    ga_brute, gb_brute = brute_one_rdm(coeffs, adets, bdets, n_orb)
    np.testing.assert_allclose(ga.numpy(), ga_brute, atol=1e-12)
    np.testing.assert_allclose(gb.numpy(), gb_brute, atol=1e-12)

    for builder, brute in (
        (make_two_rdm_aa_torch, brute_two_rdm_aa),
        (make_two_rdm_ab_torch, brute_two_rdm_ab),
        (make_two_rdm_bb_torch, brute_two_rdm_bb),
    ):
        got = builder(
            torch.tensor(coeffs), adets_t, bdets_t, n_orb, helpers, eigvec_in_interleaved_wick=False
        )
        want = brute(coeffs, adets, bdets, n_orb)
        np.testing.assert_allclose(got.numpy(), want, atol=1e-12)
        del got, want


# ============================================================================
# 3. Spin-summed 2-RDM and <S²> from the RDM
# ============================================================================


@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_spin_summed_two_rdm_and_s_squared(complex_coeffs: bool) -> None:
    """The spin-summed 2-RDM has shape (L,L,L,L) and <S²> matches the reference."""
    n_orb, adets, bdets, eigvec, reference = _small_case(complex_coeffs)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = _one_word_tensor(adets)
    bdets_t = _one_word_tensor(bdets)

    Gaa, Gab, Gbb = make_two_rdm_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    spin_summed = Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb
    assert spin_summed.shape == (n_orb, n_orb, n_orb, n_orb)

    want = reference.spin_summed_two_rdm()
    for key, value in want.items():
        assert abs(complex(spin_summed[key].item()) - value) < 1e-12, key
    # The trace of the spin-summed 2-RDM is N(N-1).
    n_elec = reference.n_alpha + reference.n_beta
    trace = torch.einsum("pprr->", spin_summed).real.item()
    assert abs(trace - n_elec * (n_elec - 1)) < 1e-10

    s2 = s_squared_from_rdm(Gab, reference.n_alpha, reference.n_beta)
    assert not s2.dtype.is_complex
    assert abs(float(s2.item()) - reference.s_squared()) < 1e-12


# ============================================================================
# 4. Wide bases: 70/130 orbitals against the Python-int reference
# ============================================================================


def _wide_case(
    n_orb: int, *, complex_coeffs: bool = False
) -> tuple[list[int], list[int], torch.Tensor, "_SerialRDMs"]:
    """Selected product basis at ``n_orb``, amplitudes and reference RDMs."""
    strings = _wide_strings(n_orb)
    adets, bdets = strings, strings
    m_a, m_b = len(adets), len(bdets)
    gen = torch.Generator().manual_seed(11)
    real = torch.randn(m_a * m_b, generator=gen, dtype=torch.float64)
    if complex_coeffs:
        eigvec = real + 1j * torch.randn(m_a * m_b, generator=gen, dtype=torch.float64)
    else:
        eigvec = real
    # ⟨S²⟩'s sector constants are quadratic in the amplitudes, so the state
    # must be normalized for the RDM-derived value to match the reference.
    eigvec = eigvec / torch.linalg.norm(eigvec)
    coeffs = _spin_block_coeffs(eigvec, adets, bdets, n_orb)
    flat_alphas = [a for a in adets for _ in bdets]
    flat_betas = [b for _ in adets for b in bdets]
    reference = _SerialRDMs(flat_alphas, flat_betas, coeffs, n_orb)
    return adets, bdets, eigvec, reference


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_wide_one_rdm_matches_reference(n_orb: int, complex_coeffs: bool) -> None:
    """1-RDM at 70/130 orbitals matches the arbitrary-precision reference."""
    adets, bdets, eigvec, reference = _wide_case(n_orb, complex_coeffs=complex_coeffs)
    assert _limb_array(adets, n_orb).shape[1] == _n_limbs(n_orb)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = torch.from_numpy(_limb_array(adets, n_orb))
    bdets_t = torch.from_numpy(_limb_array(bdets, n_orb))

    ga, gb = make_one_rdm_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    np.testing.assert_allclose(ga.numpy(), reference.ga, atol=1e-11)
    np.testing.assert_allclose(gb.numpy(), reference.gb, atol=1e-11)


@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_wide_two_rdm_and_spin_summed_at_70(complex_coeffs: bool) -> None:
    """All 2-RDM blocks and the spin-summed (L,L,L,L) tensor at 70 orbitals."""
    n_orb = 70
    adets, bdets, eigvec, reference = _wide_case(n_orb, complex_coeffs=complex_coeffs)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = torch.from_numpy(_limb_array(adets, n_orb))
    bdets_t = torch.from_numpy(_limb_array(bdets, n_orb))

    Gaa = make_two_rdm_aa_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    _compare_sparse(Gaa, reference.gaa, n_orb)
    del Gaa
    Gbb = make_two_rdm_bb_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    _compare_sparse(Gbb, reference.gbb, n_orb)
    del Gbb
    Gab = make_two_rdm_ab_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    _compare_sparse(Gab, reference.gab, n_orb)

    s2 = s_squared_from_rdm(Gab, reference.n_alpha, reference.n_beta)
    assert abs(float(s2.item()) - reference.s_squared()) < 1e-11
    del Gab

    # Full builder and spin-summed contraction at the same width.
    Gaa, Gab, Gbb = make_two_rdm_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    spin_summed = Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb
    assert spin_summed.shape == (n_orb, n_orb, n_orb, n_orb)
    _compare_sparse(spin_summed, reference.spin_summed_two_rdm(), n_orb)
    del Gaa, Gab, Gbb, spin_summed
    gc.collect()


@pytest.mark.slow
@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_wide_two_rdm_matches_reference_at_130(complex_coeffs: bool) -> None:
    """All 2-RDM blocks at 130 orbitals match the reference to 1e-11.

    Each block is a dense ``(L, L, L, L)`` tensor (several GB at this width),
    so this runs one block at a time and compares against the sparse reference
    rather than a dense one.
    """
    n_orb = 130
    adets, bdets, eigvec, reference = _wide_case(n_orb, complex_coeffs=complex_coeffs)
    helpers = generate_all_torch_helpers(adets, bdets, n_orb)
    adets_t = torch.from_numpy(_limb_array(adets, n_orb))
    bdets_t = torch.from_numpy(_limb_array(bdets, n_orb))

    for builder, table in (
        (make_two_rdm_aa_torch, reference.gaa),
        (make_two_rdm_ab_torch, reference.gab),
        (make_two_rdm_bb_torch, reference.gbb),
    ):
        got = builder(eigvec, adets_t, bdets_t, n_orb, helpers)
        _compare_sparse(got, table, n_orb)
        del got
        gc.collect()

    Gab = make_two_rdm_ab_torch(eigvec, adets_t, bdets_t, n_orb, helpers)
    s2 = s_squared_from_rdm(Gab, reference.n_alpha, reference.n_beta)
    assert abs(float(s2.item()) - reference.s_squared()) < 1e-11
    del Gab
    gc.collect()
