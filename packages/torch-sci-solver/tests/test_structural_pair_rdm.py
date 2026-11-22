"""Post-processing parity for a structural (block-described) pair basis.

The same tier basis is built twice — once as a :class:`StructuralPairBasis`
(Cartesian blocks, no length-N array) and once as the explicit lex-sorted pair
list — and the 1-RDM, the αα/αβ/ββ 2-RDM and ⟨S²⟩ must agree to 1e-10 after
permuting the amplitudes between the two pair orders. A second test records
every ``torch`` allocation made by the structural builders and asserts none is
of length N.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest
import torch
from test_pair_blocks import _strings  # noqa: E402
from torch import Tensor
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import (
    StructuralPairBasis,
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)
from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm
from torch_sci_solver.torch_impl.rdm_pairs import (
    make_one_rdm_pair_torch,
    make_two_rdm_aa_pair_torch,
    make_two_rdm_ab_pair_torch,
    make_two_rdm_bb_pair_torch,
    make_two_rdm_pair_torch,
)

NORB = 8
NELEC = 4
MAX_LEVEL = 2
ATOL = 1e-10


def _tier_spec() -> list[tuple[np.ndarray, np.ndarray]]:
    """The benchmark-shaped tier blocks, as structural block specifications."""
    tiers = [_strings(NORB, NELEC, level) for level in range(MAX_LEVEL + 1)]
    return [
        (tiers[level - b], tiers[b]) for level in range(MAX_LEVEL + 1) for b in range(level + 1)
    ]


def _grid_spec() -> list[tuple[np.ndarray, np.ndarray]]:
    """A four-block product grid over disjoint halves of every string."""
    strings = np.sort(np.concatenate([_strings(NORB, NELEC, level) for level in range(NELEC + 1)]))
    half = len(strings) // 2
    halves = (strings[:half], strings[half:])
    return [(alpha, beta) for alpha in halves for beta in halves]


def _structural(spec) -> StructuralPairBasis:
    """The block specification as a structural basis."""
    return structural_pair_basis_from_blocks(spec)


def _explicit(spec) -> object:
    """The same tier basis as an explicit, lex-sorted pair list."""
    alphas, betas = [], []
    for alpha, beta in spec:
        alphas.append(np.repeat(alpha, len(beta)))
        betas.append(np.tile(beta, len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas))


def _structural_to_explicit_perm(basis: StructuralPairBasis, explicit) -> Tensor:
    """Structural pair index -> explicit pair index, by (α, β) bitmask."""
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    lookup = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(
            zip(explicit.alpha_of_pair.tolist(), explicit.beta_of_pair.tolist())
        )
    }
    return torch.tensor([lookup[(int(a), int(b))] for a, b in zip(alpha, beta)])


def _random_vector(n: int, complex_values: bool) -> Tensor:
    """Deterministic unit-scale amplitude vector."""
    generator = torch.Generator().manual_seed(17)
    real = torch.randn(n, generator=generator, dtype=torch.float64)
    if not complex_values:
        return real
    imag = torch.randn(n, generator=generator, dtype=torch.float64)
    return torch.complex(real, imag)


def _assert_close(got: Tensor, expected: Tensor, label: str) -> None:
    """Assert one observable matches the explicit-basis reference to ``ATOL``."""
    assert got.dtype == expected.dtype, (label, got.dtype, expected.dtype)
    assert got.shape == expected.shape, (label, got.shape, expected.shape)
    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0.0, msg=label)


@pytest.mark.parametrize("complex_values", [False, True])
def test_structural_rdms_match_explicit(complex_values: bool) -> None:
    """1-RDM, 2-RDM and ⟨S²⟩ equal the explicit-pair-list results to 1e-10."""
    spec = _tier_spec()
    structural = _structural(spec)
    explicit = _explicit(spec)
    assert structural.n_pairs == explicit.n_pairs
    # ``perm`` maps a structural pair index to its explicit pair index; feed
    # the explicit builders the inverse image of the same amplitudes.
    perm = _structural_to_explicit_perm(structural, explicit)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(structural.n_pairs)

    helpers_s = generate_pair_helpers(structural, NORB, block_matvec="on")
    helpers_e = generate_pair_helpers(explicit, NORB, block_matvec="on")
    assert helpers_s.structural is structural

    x_s = _random_vector(structural.n_pairs, complex_values)
    x_e = x_s[inverse]

    gamma_a_s, gamma_b_s = make_one_rdm_pair_torch(x_s, structural, helpers_s, NORB)
    gamma_a_e, gamma_b_e = make_one_rdm_pair_torch(x_e, explicit, helpers_e, NORB)
    _assert_close(gamma_a_s, gamma_a_e, "γα")
    _assert_close(gamma_b_s, gamma_b_e, "γβ")

    Gaa_s, Gab_s, Gbb_s = make_two_rdm_pair_torch(x_s, structural, helpers_s, NORB)
    Gaa_e, Gab_e, Gbb_e = make_two_rdm_pair_torch(x_e, explicit, helpers_e, NORB)
    _assert_close(Gaa_s, Gaa_e, "Γαα")
    _assert_close(Gab_s, Gab_e, "Γαβ")
    _assert_close(Gbb_s, Gbb_e, "Γββ")

    # The agreement must not be vacuous: the excitations in this basis are
    # cross-block, so a block-local-only implementation would leave the
    # off-diagonal 1-RDM at zero.
    off_diagonal = float((gamma_a_s - torch.diag(gamma_a_s.diagonal())).abs().max())
    assert off_diagonal > 1e-3, "the α 1-RDM singles contribution is zero"

    _assert_close(
        make_two_rdm_aa_pair_torch(x_s, structural, helpers_s, NORB),
        make_two_rdm_aa_pair_torch(x_e, explicit, helpers_e, NORB),
        "Γαα single block",
    )
    _assert_close(
        make_two_rdm_ab_pair_torch(x_s, structural, helpers_s, NORB),
        make_two_rdm_ab_pair_torch(x_e, explicit, helpers_e, NORB),
        "Γαβ single block",
    )
    _assert_close(
        make_two_rdm_bb_pair_torch(x_s, structural, helpers_s, NORB),
        make_two_rdm_bb_pair_torch(x_e, explicit, helpers_e, NORB),
        "Γββ single block",
    )

    s2_s = s_squared_from_rdm(Gab_s, NELEC, NELEC)
    s2_e = s_squared_from_rdm(Gab_e, NELEC, NELEC)
    _assert_close(s2_s, s2_e, "⟨S²⟩")


@pytest.mark.parametrize("complex_values", [False, True])
def test_structural_path_allocates_no_length_n_array(
    complex_values: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every tracked torch allocation is bounded by the chunk and the output size."""
    spec = _grid_spec()
    structural = _structural(spec)
    helpers = generate_pair_helpers(structural, NORB, block_matvec="on")
    x = _random_vector(structural.n_pairs, complex_values)
    chunk_size = 32
    membership_width = max(
        int(structural.alpha_block_ids.shape[1]), int(structural.beta_block_ids.shape[1])
    )
    bound = max(
        chunk_size,
        NORB**4,
        structural.n_unique_alphas * membership_width,
        structural.n_unique_betas * membership_width,
    )
    assert structural.n_pairs > bound

    allocations: list[int] = []
    for name in ("empty", "zeros", "full", "arange"):
        original = getattr(torch, name)

        def spy(*args, _original=original, **kwargs):
            tensor = _original(*args, **kwargs)
            allocations.append(int(tensor.numel()))
            return tensor

        monkeypatch.setattr(torch, name, spy)

    make_one_rdm_pair_torch(x, structural, helpers, NORB, chunk_size=chunk_size)
    make_two_rdm_pair_torch(x, structural, helpers, NORB, chunk_size=chunk_size)
    make_two_rdm_ab_pair_torch(x, structural, helpers, NORB, chunk_size=chunk_size)

    assert allocations, "the allocation spy recorded nothing"
    assert max(allocations) <= bound, (max(allocations), bound)
