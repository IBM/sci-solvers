"""Tests for torch_sci_solver.torch_impl.integrals — vec integrals builder.

Companion to ``test_sbd_ref.py``, which exercises the scalar reference
:func:`setup_integrals_from_pyscf_h1_eri`. This file asserts that the
numpy-vectorized :func:`setup_integrals_from_pyscf_h1_eri_vec` — routed
through ``qiskit_adapter`` on the production path — produces bit-identical
OneInt/TwoInt objects.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.integrals import setup_integrals_from_pyscf_h1_eri_vec


def _random_symmetric_h1(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((L, L))
    return 0.5 * (h1 + h1.T)


def _random_8fold_symmetric_eri(L: int, seed: int) -> np.ndarray:
    """8-fold-symmetric random ERI, matching real chemistry integrals.

    An SCF-derived (ij|kl) tensor has the eight-way permutation symmetry
    (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (kl|ji) = (lk|ij) = (lk|ji).
    Building the test tensor with this symmetry guarantees the vec builder
    stress-tests the same input shape production sees.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((L, L, L, L))
    eri = np.zeros_like(x)
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
        eri += x.transpose(perm)
    return eri / 8.0


def test_vec_matches_scalar_L6():
    """Small L=6 sanity: bit-identical OneInt/TwoInt fields."""
    L = 6
    h1 = _random_symmetric_h1(L, seed=0)
    eri = _random_8fold_symmetric_eri(L, seed=1)

    _, I1s, I2s = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    _, I1v, I2v = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)

    assert np.array_equal(I1s._h, I1v._h), "OneInt._h differs"
    assert np.array_equal(I2s._eri, I2v._eri), "TwoInt._eri differs"
    assert np.array_equal(I2s._direct, I2v._direct), "TwoInt._direct differs"
    assert np.array_equal(I2s._exchange, I2v._exchange), "TwoInt._exchange differs"


def test_vec_matches_scalar_L10():
    """L=10 spans the doubly-nested-loop range at scale the scalar handles fast.

    Nontrivial α/β interleave (2L = 20 spin-orbitals) and enough data volume
    (10⁴ = 10 000 ERI entries per spin-block quadrant) that a bit-level bug
    in the strided assignment would show up on at least one element.
    """
    L = 10
    h1 = _random_symmetric_h1(L, seed=2)
    eri = _random_8fold_symmetric_eri(L, seed=3)

    _, I1s, I2s = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.5)
    _, I1v, I2v = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.5)

    assert np.array_equal(I1s._h, I1v._h)
    assert np.array_equal(I2s._eri, I2v._eri)
    assert np.array_equal(I2s._direct, I2v._direct)
    assert np.array_equal(I2s._exchange, I2v._exchange)


def test_vec_accepts_packed_eri():
    """4-fold-packed ERI input (PySCF's ``ao2mo.restore(4, ...)`` shape).

    ``qiskit_adapter`` sees full ``(L, L, L, L)`` in production, but the
    ``_to_full_eri`` helper both callers share supports packed inputs.
    Round-tripping through it must give the same result on both paths.
    """
    from pyscf import ao2mo

    L = 8
    h1 = _random_symmetric_h1(L, seed=4)
    eri_full = _random_8fold_symmetric_eri(L, seed=5)
    eri_packed = ao2mo.restore(4, eri_full, L)  # 4-fold-packed

    _, I1s, I2s = setup_integrals_from_pyscf_h1_eri(h1, eri_packed, 0.0)
    _, I1v, I2v = setup_integrals_from_pyscf_h1_eri_vec(h1, eri_packed, 0.0)

    assert np.array_equal(I1s._h, I1v._h)
    assert np.array_equal(I2s._eri, I2v._eri)
    assert np.array_equal(I2s._direct, I2v._direct)
    assert np.array_equal(I2s._exchange, I2v._exchange)
