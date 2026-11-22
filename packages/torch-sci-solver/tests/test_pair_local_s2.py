"""Flat 1-D owner-computes ``S^2`` for the distributed GDB spin penalty.

``PairLocalSpinSquared`` applies the pair-mode ``S^2`` operator to a rank's
contiguous pair block, reading every spin-swap source from the matvec's
all-gathered full vector instead of a staged tile exchange. It must reproduce
the serial :class:`PairSpinSqOperator` (in the interleaved-Wick convention the
amplitudes use) on one rank and on uneven multi-rank gloo splits, for real and
complex amplitudes, on single-word and multi-limb strings, and for explicit and
structural pair bases. The sweep is chunked: it must never allocate an O(K)
swap table, a length-``N`` temporary, or a ``(step, n_orb)`` occupancy array
wider than the configured ``chunk_size`` admits.

Covers:

1. Single process, real and complex: the diagonal, the applied block and the
   reduced expectation equal the serial operator for several ``chunk_size``
   values, and ``gather_full`` + ``matvec_from_full`` reproduces ``__call__``.
2. gloo, 2 and 3 ranks on a 35-pair basis (uneven blocks): each rank's applied
   block matches the serial slice, the gathered vector equals the full input,
   the expectation matches, and an allocation spy sees no length-``N`` tensor.
3. The penalized composition ``H + lam (S^2 - target)`` uses exactly one
   all-gather and matches the serial matvec plus the serial ``S^2`` block.
4. A structural (block-described) basis gives the same flat ``S^2`` as the
   explicit pair list of the same space, under the string permutation.
5. Multi-limb strings at 70 orbitals, real and complex, match the serial
   operator.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pathlib

import pytest
import torch
import torch.distributed as dist
from test_distributed_pairs import (
    NORB,
    _init_gloo,
    _pair_setup,
    _random_complex_vector,
    _random_vector,
    _run_spawned,
)
from test_pair_tiled_postprocess_multilimb import _product_basis, _wide_perm
from test_rdm_pairs_multilimb import _wide_strings
from test_structural_tiled_postprocess import _explicit, _structural
from torch_sci_solver.torch_impl.distributed_davidson import partition_vector
from torch_sci_solver.torch_impl.distributed_pairs import LocalPairOperator
from torch_sci_solver.torch_impl.mult_pairs import (
    make_diag_terms_pair_local,
    make_diag_terms_pair_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiled_postprocess import PairLocalSpinSquared
from torch_sci_solver.torch_impl.qiskit_adapter import _PairLocalPenalizedOperator
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

# The flat sweep and the serial operator perform the same single scattered
# sweep, so they agree to roundoff.
ATOL = 1e-12


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _serial_s2(basis, helpers, n_orb: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Serial ``S^2 x`` in the amplitude convention, and the serial diagonal."""
    operator = PairSpinSqOperator(basis, n_orb, int(helpers.n_a), int(helpers.n_b))
    sigma = _inter_to_block_signs_pair(helpers.alpha_of_pair, helpers.beta_of_pair, n_orb).to(
        x.dtype
    )
    return sigma * operator.apply(x * sigma), operator.diagonal()


def _operator(
    basis,
    helpers,
    integrals,
    rank: int,
    world_size: int,
    *,
    n_orb: int = NORB,
    chunk_size: int = 64,
) -> LocalPairOperator:
    """One rank's 1-D owner-computes pair operator."""
    bounds = partition_vector(basis.n_pairs, world_size, rank)
    if integrals is None:
        hii_local = torch.zeros(bounds.local_size, dtype=torch.float64)
    else:
        alpha = basis.alpha_of_pair[bounds.start : bounds.stop]
        beta = basis.beta_of_pair[bounds.start : bounds.stop]
        hii_local = make_diag_terms_pair_local(alpha, beta, n_orb, integrals)
    return LocalPairOperator(
        hii_local,
        basis,
        helpers,
        n_orb,
        integrals,
        bounds,
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
    )


# ============================================================================
# 1. Single process
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("chunk_size", [3, 64])
def test_local_s2_single_process_matches_serial(complex_values: bool, chunk_size: int) -> None:
    """world_size=1: diagonal, apply and expectation equal the serial operator."""
    pair_basis, helpers, integrals, hii = _pair_setup()
    n = pair_basis.n_pairs
    x = _random_complex_vector(n) if complex_values else _random_vector(n)
    expected, expected_diag = _serial_s2(pair_basis, helpers, NORB, x)

    operator = _operator(pair_basis, helpers, integrals, 0, 1, chunk_size=chunk_size)
    s2 = PairLocalSpinSquared(operator, sign_dtype=x.dtype)
    T_full = operator.gather_full(x)
    assert torch.equal(T_full, x)

    assert torch.allclose(operator.matvec_from_full(x, T_full), operator(x), atol=ATOL, rtol=0.0)
    assert torch.allclose(s2.diagonal(), expected_diag, atol=0.0, rtol=0.0)
    got = s2.apply(x, T_full)
    assert got.shape == x.shape
    assert torch.allclose(got, expected, atol=ATOL, rtol=0.0)
    assert float(s2.expectation(x)) == pytest.approx(float(torch.vdot(x, expected).real), abs=ATOL)

    # The chunked sweep bound: one slice enumerates at most ``chunk_size``
    # candidates per re-enumeration, and each pair contributes at most
    # ``max_candidates`` spin swaps.
    max_candidates = max(
        1,
        min(int(helpers.n_a), NORB - int(helpers.n_b))
        * min(int(helpers.n_b), NORB - int(helpers.n_a)),
    )
    assert s2._step == max(1, chunk_size // max_candidates)
    assert s2._step * max_candidates <= max(chunk_size, max_candidates)


def test_local_s2_rejects_a_foreign_gather() -> None:
    """A full vector of the wrong length is refused."""
    pair_basis, helpers, integrals, _ = _pair_setup()
    operator = _operator(pair_basis, helpers, integrals, 0, 1)
    x = _random_vector(pair_basis.n_pairs)
    with pytest.raises(ValueError, match="T_full shape"):
        operator.matvec_from_full(x, torch.zeros(pair_basis.n_pairs + 1))
    with pytest.raises(ValueError, match="x_local shape"):
        operator.gather_full(torch.zeros(pair_basis.n_pairs + 1))


# ============================================================================
# 2. gloo: 2 and 3 ranks with uneven blocks
# ============================================================================


def _flat_s2_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Uneven-split parity, the shared-gather composition and an allocation guard."""
    _init_gloo(rank, world_size, init_file)
    try:
        pair_basis, helpers, integrals, _ = _pair_setup(n_drop=1)
        n = pair_basis.n_pairs
        assert n == 35  # uneven for both 2 and 3 ranks
        chunk_size = 7
        operator = _operator(
            pair_basis, helpers, integrals, rank, world_size, chunk_size=chunk_size
        )
        bounds = operator.bounds
        sl = slice(bounds.start, bounds.stop)
        x = _random_vector(n)
        expected, expected_diag = _serial_s2(pair_basis, helpers, NORB, x)
        s2 = PairLocalSpinSquared(operator)

        T_full = operator.gather_full(x[sl])
        assert torch.allclose(T_full, x, atol=0.0, rtol=0.0), (rank, world_size)
        got = s2.apply(x[sl], T_full)
        assert torch.allclose(got, expected[sl], atol=ATOL, rtol=0.0), (
            rank,
            world_size,
            float((got - expected[sl]).abs().max()),
        )
        assert torch.allclose(s2.diagonal(), expected_diag[sl], atol=0.0, rtol=0.0)
        expected_value = float(torch.vdot(x, expected).real)
        assert float(s2.expectation(x[sl])) == pytest.approx(expected_value, abs=ATOL)

        # The gathered vector serves ``H`` too: the composed operator posts a
        # single all-gather per matvec.
        calls = [0]
        original_start = operator._gather.start

        def _counting_start(local, _original=original_start, _calls=calls):
            _calls[0] += 1
            return _original(local)

        operator._gather.start = _counting_start
        try:
            lam, target = 0.3, 2.0
            composed = _PairLocalPenalizedOperator(operator, s2, lam, target)(x[sl])
        finally:
            del operator._gather.start
        assert calls == [1], (rank, calls)

        hii = make_diag_terms_pair_torch(pair_basis, NORB, integrals)
        full_hx = mult_pair_torch(hii, x, pair_basis, helpers, NORB, integrals)
        expected_composed = full_hx[sl] + lam * (expected[sl] - target * x[sl])
        assert torch.allclose(composed, expected_composed, atol=ATOL, rtol=0.0)

        # Allocation guard: the sweep may allocate its local block, one
        # slice's candidates and the O(n_orb) occupancy rows, never a
        # length-N or O(K) temporary.
        tracked: list[int] = []
        originals = {name: getattr(torch, name) for name in ("empty", "zeros", "full", "arange")}

        def _spy(original):
            def wrapper(*args, **kwargs):
                tensor = original(*args, **kwargs)
                tracked.append(int(tensor.numel()))
                return tensor

            return wrapper

        try:
            for name, original in originals.items():
                setattr(torch, name, _spy(original))
            s2.apply(x[sl], T_full)
        finally:
            for name, original in originals.items():
                setattr(torch, name, original)
        n_local = int(bounds.local_size)
        assert tracked, "the allocation spy recorded nothing"
        assert n not in tracked, (rank, "a length-N tensor was allocated")
        assert max(tracked) <= 8 * n_local + 64, (rank, max(tracked))

        # Complex coefficients ride the same sweep.
        xc = _random_complex_vector(n)
        expected_c, _ = _serial_s2(pair_basis, helpers, NORB, xc)
        got_c = s2.apply(xc[sl], operator.gather_full(xc[sl]))
        assert torch.allclose(got_c, expected_c[sl], atol=ATOL, rtol=0.0)
        assert float(s2.expectation(xc[sl])) == pytest.approx(
            float(torch.vdot(xc, expected_c).real), abs=ATOL
        )
        if rank == 0:
            pathlib.Path(out_file).open("w").write("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_local_s2_gloo_uneven_matches_serial(tmp_path, world_size: int) -> None:
    """gloo, 2/3 ranks, 35 pairs: parity, one gather, bounded allocations."""
    _skip_without_gloo()
    _run_spawned(_flat_s2_worker, world_size, (), tmp_path, f"pair_local_s2_{world_size}")


# ============================================================================
# 3. Structural basis
# ============================================================================


def test_local_s2_structural_matches_explicit() -> None:
    """The block-walked flat sweep matches the explicit pair list of the space."""
    structural = _structural()
    explicit = _explicit()
    assert structural.n_pairs == explicit.n_pairs
    helpers_s = generate_pair_helpers(structural, 8, device="cpu", block_matvec="on")
    helpers_e = generate_pair_helpers(explicit, 8, device="cpu")
    perm = _wide_perm(structural, explicit)  # structural index -> explicit index
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(structural.n_pairs)

    x_s = _random_vector(structural.n_pairs)
    x_e = x_s[inverse]  # the same state in the explicit pair order
    expected, expected_diag = _serial_s2(explicit, helpers_e, 8, x_e)

    operator_s = _operator(structural, helpers_s, None, 0, 1, n_orb=8)
    operator_e = _operator(explicit, helpers_e, None, 0, 1, n_orb=8)
    s2_s = PairLocalSpinSquared(operator_s)
    s2_e = PairLocalSpinSquared(operator_e)

    assert torch.allclose(s2_s.diagonal()[inverse], expected_diag, atol=0.0, rtol=0.0)
    assert torch.allclose(s2_e.diagonal(), expected_diag, atol=0.0, rtol=0.0)
    got_s = s2_s.apply(x_s, x_s)
    assert torch.allclose(got_s[inverse], expected, atol=ATOL, rtol=0.0)
    assert torch.allclose(s2_e.apply(x_e, x_e), expected, atol=ATOL, rtol=0.0)
    assert float(s2_s.expectation(x_s)) == pytest.approx(
        float(torch.vdot(x_e, expected).real), abs=ATOL
    )


# ============================================================================
# 4. Multi-limb strings
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_local_s2_multilimb_matches_serial(complex_values: bool) -> None:
    """70 orbitals (two limbs): the flat sweep matches the serial operator."""
    n_orb = 70
    basis = _product_basis(_wide_strings(n_orb), n_orb)
    helpers = generate_pair_helpers(basis, n_orb, device="cpu")
    n = basis.n_pairs
    x = _random_complex_vector(n) if complex_values else _random_vector(n)
    expected, expected_diag = _serial_s2(basis, helpers, n_orb, x)

    operator = _operator(basis, helpers, None, 0, 1, n_orb=n_orb, chunk_size=5)
    s2 = PairLocalSpinSquared(operator, sign_dtype=x.dtype)
    assert torch.allclose(s2.diagonal(), expected_diag, atol=0.0, rtol=0.0)
    got = s2.apply(x, x)
    assert torch.allclose(got, expected, atol=ATOL, rtol=0.0)
    assert float(s2.expectation(x)) == pytest.approx(float(torch.vdot(x, expected).real), abs=ATOL)
