"""Multi-limb (strings wider than 64 bits) support in the pair-mode RDM builders.

The explicit-:class:`PairBasis` post-processing builders in
``torch_impl/rdm_pairs.py`` used to do 1-D bit operations
(``strings.unsqueeze(1) >> torch.arange(n_orb)``, ``1 << pos``). They now
accept ``(M, n_limbs)`` limb matrices like the pair matvec does.

1. An 8-orbital basis whose strings are carried as ``(M, 2)`` limb matrices
   (limb ``k`` holds bits ``[64k, 64k + 64)``, so a padded high limb) gives
   bit-identical 1-RDMs, αα/αβ/ββ 2-RDMs and ⟨S²⟩ to the same basis in 1-D
   form, for real and complex amplitudes.

2. At 70 and 130 orbitals a selected pair basis of a few dozen pairs
   reproduces a brute-force reference — arbitrary-precision Python ints and
   the direct second-quantized matrix elements — to 1e-11.

3. ⟨S²⟩ from :class:`PairSpinSqOperator` matches the reference cross term.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import replace

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import PairBasis, pair_basis_from_arrays
from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm
from torch_sci_solver.torch_impl.rdm_pairs import (
    make_one_rdm_diag_pair_torch,
    make_one_rdm_pair_torch,
    make_two_rdm_aa_pair_torch,
    make_two_rdm_ab_pair_torch,
    make_two_rdm_bb_pair_torch,
    make_two_rdm_pair_torch,
)
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

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


def _wide_strings(n_orb: int) -> list[int]:
    """Six 2-electron strings that straddle the 64-bit limb boundary.

    For ``n_orb=70`` the occupied orbitals are 0/1 (limb 0) and 64/65
    (limb 1); for ``n_orb=130`` the high orbitals are 64 and 128, so all
    three limbs carry bits.
    """
    high = 128 if n_orb > 128 else 65
    base = (1 << 0) | (1 << 64)
    strings = {base, (1 << 1) | (1 << high)}
    for i in (0, 64):
        for j in (1, high):
            strings.add((base ^ (1 << i)) | (1 << j))
    return sorted(strings)


def _product_basis(strings: list[int], n_orb: int) -> PairBasis:
    """The full Cartesian product of one spin's string set with itself."""
    alphas = np.repeat(np.array(strings, dtype=object), len(strings))
    betas = np.tile(np.array(strings, dtype=object), len(strings))
    return pair_basis_from_arrays(alphas, betas, n_orb=n_orb)


def _two_limb(basis: PairBasis) -> PairBasis:
    """The same basis with every string field padded to ``(M, 2)`` limbs.

    Limb ``k`` holds bits ``[64k, 64k + 64)``; the 8-orbital strings live
    entirely in limb 0 and limb 1 is zero padding.
    """

    def pad(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([t, torch.zeros_like(t)], dim=-1)

    return replace(
        basis,
        alpha_of_pair=pad(basis.alpha_of_pair),
        beta_of_pair=pad(basis.beta_of_pair),
        unique_alphas=pad(basis.unique_alphas),
        unique_betas=pad(basis.unique_betas),
    )


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


def _sigma(alpha: int, beta: int, n_orb: int) -> float:
    """Interleaved-Wick → spin-block sign for one determinant (Python ints)."""
    exponent = 0
    for j in range(n_orb):
        if (alpha >> j) & 1:
            exponent += _popcount(beta & ((1 << j) - 1))
    return -1.0 if exponent & 1 else 1.0


# ============================================================================
# Brute-force reference: direct second-quantized matrix elements
# ============================================================================


def _popcount(x: int) -> int:
    """Population count of an arbitrary-precision Python int."""
    return bin(x).count("1")


def _single_bit(x: int) -> int:
    """Position of the only set bit of ``x``."""
    return x.bit_length() - 1


def _apply_ops(ops, alpha: int, beta: int):
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


class _ReferenceRDMs:
    """Brute-force 1-RDM/2-RDM of a pair-basis state, from Python-int strings."""

    def __init__(self, alphas: list[int], betas: list[int], coeffs, n_orb: int):
        """Evaluate every matrix element of the documented PySCF convention."""
        self.n_orb = n_orb
        self.n_alpha = _popcount(alphas[0])
        self.n_beta = _popcount(betas[0])
        L = n_orb
        real = not np.iscomplexobj(coeffs)
        dtype = np.float64 if real else np.complex128
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

    def dense(self, table) -> np.ndarray:
        """A dense ``(L, L, L, L)`` complex array from one sparse block."""
        L = self.n_orb
        arr = np.zeros((L, L, L, L), dtype=np.complex128)
        for key, value in table.items():
            arr[key] = value
        return arr


def _bits(mask: int) -> list[int]:
    """Ascending bit positions of a Python-int mask."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _spin_block_coeffs(eigvec: torch.Tensor, basis: PairBasis, n_orb: int) -> np.ndarray:
    """Interleaved-Wick amplitudes converted to the spin-block convention."""
    alphas = _rows_as_ints(basis.alpha_of_pair)
    betas = _rows_as_ints(basis.beta_of_pair)
    signs = np.array([_sigma(a, b, n_orb) for a, b in zip(alphas, betas)])
    return signs * eigvec.numpy()


# ============================================================================
# 1. Small basis: 2-limb strings are bit-identical to the 1-D form
# ============================================================================


@pytest.mark.parametrize("complex_coeffs", [False, True], ids=["real", "complex"])
def test_two_limb_small_basis_bit_identical(complex_coeffs: bool) -> None:
    """8-orbital basis carried as 2 limbs matches the 1-D basis bit for bit."""
    n_orb = 8
    basis_1d = _product_basis(_small_strings(), n_orb)
    basis_limb = _two_limb(basis_1d)
    assert basis_limb.n_limbs == 2

    helpers_1d = generate_pair_helpers(basis_1d, n_orb)
    helpers_limb = generate_pair_helpers(basis_limb, n_orb)
    # The excitation tables themselves must be the same row set.
    assert helpers_1d.nnz() == helpers_limb.nnz()

    n_pairs = basis_1d.n_pairs
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(n_pairs, generator=gen, dtype=torch.float64)
    imag = torch.randn(n_pairs, generator=gen, dtype=torch.float64)
    dtype = torch.complex128 if complex_coeffs else torch.float64
    eigvec = real + 1j * imag if complex_coeffs else real

    diag_1d = make_one_rdm_diag_pair_torch(
        eigvec, basis_1d.alpha_of_pair, basis_1d.beta_of_pair, n_orb
    )
    diag_limb = make_one_rdm_diag_pair_torch(
        eigvec, basis_limb.alpha_of_pair, basis_limb.beta_of_pair, n_orb
    )
    for got, want in zip(diag_limb, diag_1d):
        assert torch.equal(got, want)

    ga_1d, gb_1d = make_one_rdm_pair_torch(eigvec, basis_1d, helpers_1d, n_orb)
    ga_limb, gb_limb = make_one_rdm_pair_torch(eigvec, basis_limb, helpers_limb, n_orb)
    assert torch.equal(ga_limb, ga_1d)
    assert torch.equal(gb_limb, gb_1d)

    rdm_1d = make_two_rdm_pair_torch(eigvec, basis_1d, helpers_1d, n_orb)
    rdm_limb = make_two_rdm_pair_torch(eigvec, basis_limb, helpers_limb, n_orb)
    for got, want in zip(rdm_limb, rdm_1d):
        assert got.dtype == dtype
        assert torch.equal(got, want)

    s2_1d = s_squared_from_rdm(rdm_1d[1], 2, 2)
    s2_limb = s_squared_from_rdm(rdm_limb[1], 2, 2)
    assert torch.equal(s2_limb, s2_1d)

    op_1d = PairSpinSqOperator(basis_1d, n_orb, 2, 2)
    op_limb = PairSpinSqOperator(basis_limb, n_orb, 2, 2)
    assert torch.equal(op_limb.diagonal(), op_1d.diagonal())
    probe = torch.randn(n_pairs, generator=gen, dtype=torch.float64)
    assert torch.equal(op_limb.apply(probe), op_1d.apply(probe))


# ============================================================================
# 2. Wide bases: limb builders match the Python-int second-quantized reference
# ============================================================================


def _wide_case(n_orb: int, *, complex_coeffs: bool = False):
    """Selected pair basis at ``n_orb``, amplitudes, and the reference RDMs."""
    basis = _product_basis(_wide_strings(n_orb), n_orb)
    alphas = _rows_as_ints(basis.alpha_of_pair)
    betas = _rows_as_ints(basis.beta_of_pair)
    gen = torch.Generator().manual_seed(11)
    real = torch.randn(basis.n_pairs, generator=gen, dtype=torch.float64)
    if complex_coeffs:
        eigvec = real + 1j * torch.randn(basis.n_pairs, generator=gen, dtype=torch.float64)
    else:
        eigvec = real
    # ⟨S²⟩'s sector constants are quadratic in the amplitudes, so the state
    # must be normalized for the RDM-derived value to match the operator.
    eigvec = eigvec / torch.linalg.norm(eigvec)
    coeffs = _spin_block_coeffs(eigvec, basis, n_orb)
    reference = _ReferenceRDMs(alphas, betas, coeffs, n_orb)
    return basis, eigvec, coeffs, reference


@pytest.mark.parametrize("n_orb", [70, 130])
def test_wide_one_rdm_and_spin_sq_match_reference(n_orb: int) -> None:
    """1-RDM and ⟨S²⟩ at 70/130 orbitals match the Python-int reference."""
    basis, eigvec, coeffs, reference = _wide_case(n_orb)
    assert basis.n_limbs == (2 if n_orb == 70 else 3)
    helpers = generate_pair_helpers(basis, n_orb)

    ga, gb = make_one_rdm_pair_torch(eigvec, basis, helpers, n_orb)
    np.testing.assert_allclose(ga.numpy(), reference.ga, atol=1e-11)
    np.testing.assert_allclose(gb.numpy(), reference.gb, atol=1e-11)

    op = PairSpinSqOperator(basis, n_orb, 2, 2)
    block_vec = torch.from_numpy(coeffs)
    s2 = torch.dot(op.apply(block_vec), block_vec).real
    assert abs(float(s2) - reference.s_squared()) < 1e-11


def test_wide_two_rdm_matches_reference_at_70() -> None:
    """All three 2-RDM blocks at 70 orbitals match the reference to 1e-11."""
    n_orb = 70
    basis, eigvec, _, reference = _wide_case(n_orb)
    helpers = generate_pair_helpers(basis, n_orb)

    builders = (
        (make_two_rdm_aa_pair_torch, reference.gaa),
        (make_two_rdm_ab_pair_torch, reference.gab),
        (make_two_rdm_bb_pair_torch, reference.gbb),
    )
    for builder, table in builders:
        got = builder(eigvec, basis, helpers, n_orb)
        np.testing.assert_allclose(got.numpy(), reference.dense(table), atol=1e-11)


@pytest.mark.slow
def test_wide_two_rdm_matches_reference_at_130() -> None:
    """All three 2-RDM blocks at 130 orbitals match the reference to 1e-11.

    Each block is a dense ``(L, L, L, L)`` tensor, so this runs one block at
    a time and compares against the sparse reference rather than a dense one.
    """
    n_orb = 130
    basis, eigvec, _, reference = _wide_case(n_orb)
    helpers = generate_pair_helpers(basis, n_orb)

    builders = (
        (make_two_rdm_aa_pair_torch, reference.gaa),
        (make_two_rdm_ab_pair_torch, reference.gab),
        (make_two_rdm_bb_pair_torch, reference.gbb),
    )
    for builder, table in builders:
        got = builder(eigvec, basis, helpers, n_orb)
        nonzero = got.view(-1).nonzero().squeeze(1)
        seen = set()
        for flat in nonzero.tolist():
            p, q, r, s = (
                flat // (n_orb**3),
                (flat // (n_orb**2)) % n_orb,
                (flat // n_orb) % n_orb,
                flat % n_orb,
            )
            seen.add((p, q, r, s))
            assert abs(float(got[p, q, r, s]) - table.get((p, q, r, s), 0.0)) < 1e-11
        # Every reference entry large enough to matter is in the dense block.
        for key, value in table.items():
            if abs(value) > 1e-10:
                assert key in seen, f"missing reference entry {key}"
                assert abs(float(got[key]) - value) < 1e-11
        del got
