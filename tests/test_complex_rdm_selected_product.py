"""Tests for the product-mode selected-CI complex RDM path (``complex_rdm.py``).

Before this module, ``make_rdm1_complex``/``make_rdm12_complex`` supported only:
  * the full-canonical path (no row_map), and
  * the spinless ``(N, 0)`` sector (``row_map_a`` alone, ``nelec[1] == 0``).

A genuine **product-mode** selected CI subspace -- two independently-truncated
``(strs_a, strs_b)`` string sets, exactly what
``build_selected_complex_ci_problem`` builds and the SQD pipeline always
passes -- raised ``NotImplementedError`` ("mixed alpha/beta product subspace
... not currently used by any caller"). It *is* used, by every real-world
complex ``arpack``/``lobpcg``/``davidson``/``primme`` solve on a genuinely
selected (not full-CAS) subspace, and previously crashed with an
``IndexError`` inside ``_apply_excitations_a`` instead (the RDM extraction
call site in ``general_algebra_base.py`` never passed ``row_map_a``/
``row_map_b`` at all, so it always took the full-canonical path regardless of
the problem's actual selected basis).

These tests exercise the new ``_connected_link_and_embed_product`` joint
two-axis embedding directly (via ``make_rdm1_complex``/``make_rdm12_complex``)
and the full ``create_solver("arpack", ...)`` -> ``sci_solver_callable``
integration path that previously crashed.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest

from sci_solvers.complex_ci_problem import build_selected_complex_ci_problem
from sci_solvers.complex_rdm import (
    SparseAddressMap,
    energy_from_rdm,
    make_rdm1_complex,
    make_rdm12_complex,
)
from sci_solvers.energy_solvers.factory import create_solver


def _random_complex_hermitian_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex ``hcore`` and a physically valid complex chemist-order ``eri``.

    Same construction as ``tests/test_complex_hermitian.py``'s helper of the
    same name (kept in sync deliberately: both describe one physical family
    of complex Hermitian CI Hamiltonians).
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _selected_strings(norb: int, n_elec: int, picks: tuple[int, ...]) -> np.ndarray:
    """A strict subset of the canonical ``(norb, n_elec)`` CI strings, sorted ascending."""
    from pyscf.fci import cistring

    canon = np.asarray(cistring.make_strings(range(norb), n_elec), dtype=np.int64)
    return np.sort(canon[list(picks)])


def _embed_into_full_canonical(
    C_sel: np.ndarray, strs_a: np.ndarray, strs_b: np.ndarray, norb: int, nelec: tuple[int, int]
) -> np.ndarray:
    """Zero-pad ``C_sel`` (shape ``(len(strs_a), len(strs_b))``) onto the full
    canonical ``(Na, Nb)`` grid, at the rows/columns ``strs_a``/``strs_b`` name.

    The reference oracle for the product-mode selected path: since RDMs are
    linear/bilinear functionals of the amplitude vector, the full-canonical
    formula applied to this zero-padded embedding must reduce *exactly* to
    the selected-CI formula applied to ``C_sel`` directly -- this is the
    strongest available cross-check because it reuses the already-proven
    full-canonical code path as ground truth, rather than a second
    hand-written implementation that could share the same bug.
    """
    from pyscf.fci import cistring

    n_a, n_b = int(nelec[0]), int(nelec[1])
    canon_a = np.asarray(cistring.make_strings(range(norb), n_a), dtype=np.int64)
    canon_b = np.asarray(cistring.make_strings(range(norb), n_b), dtype=np.int64)
    idx_a = np.searchsorted(canon_a, strs_a)
    idx_b = np.searchsorted(canon_b, strs_b)
    C_full = np.zeros((len(canon_a), len(canon_b)), dtype=C_sel.dtype)
    C_full[np.ix_(idx_a, idx_b)] = C_sel
    return C_full


@pytest.mark.parametrize("norb,n_a,n_b", [(5, 2, 2), (4, 2, 1)])
def test_product_mode_rdm_matches_zero_padded_full_canonical(norb, n_a, n_b):
    """The product-mode selected-CI 1-/2-RDM matches the zero-padded full-canonical oracle."""
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=1)
    sel_a = _selected_strings(norb, n_a, (0, 2, 4, 6) if norb == 5 else (0, 2))
    sel_b = _selected_strings(norb, n_b, (1, 3, 5, 7) if norb == 5 else (0,))

    prob = build_selected_complex_ci_problem(
        sel_a, sel_b, hcore, eri, norb, (n_a, n_b), device="cpu"
    )
    w, v = np.linalg.eigh(prob.sparse().toarray())
    C_sel = v[:, 0].reshape(len(sel_a), len(sel_b))

    row_map_a = SparseAddressMap(sel_a, np.arange(len(sel_a)))
    row_map_b = SparseAddressMap(sel_b, np.arange(len(sel_b)))
    dm1_sel, (dma_sel, dmb_sel) = make_rdm1_complex(
        C_sel,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
    )
    dm1_sel2, dm2_sel, gab_sel = make_rdm12_complex(
        C_sel,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
    )

    C_full = _embed_into_full_canonical(C_sel, sel_a, sel_b, norb, (n_a, n_b))
    dm1_full, (dma_full, dmb_full) = make_rdm1_complex(C_full, norb, (n_a, n_b))
    dm1_full2, dm2_full, gab_full = make_rdm12_complex(C_full, norb, (n_a, n_b))

    np.testing.assert_allclose(dm1_sel, dm1_full, atol=1e-9)
    np.testing.assert_allclose(dma_sel, dma_full, atol=1e-9)
    np.testing.assert_allclose(dmb_sel, dmb_full, atol=1e-9)
    np.testing.assert_allclose(dm1_sel2, dm1_full2, atol=1e-9)
    np.testing.assert_allclose(dm2_sel, dm2_full, atol=1e-9)
    np.testing.assert_allclose(gab_sel, gab_full, atol=1e-9)

    # Hermiticity and electron-count conservation.
    np.testing.assert_allclose(dm1_sel, dm1_sel.conj().T, atol=1e-9)
    assert np.isclose(dma_sel.trace().real, n_a, atol=1e-8)
    assert np.isclose(dmb_sel.trace().real, n_b, atol=1e-8)

    # Energy reconstructed from the selected-CI RDMs equals the exact selected
    # ground-state eigenvalue.
    e = energy_from_rdm(dm1_sel2, dm2_sel, hcore, eri)
    assert np.isclose(e.real if np.iscomplexobj(e) else e, w[0], atol=1e-8)


def test_product_mode_rdm_full_canonical_bit_identical():
    """Passing the full canonical (strs_a, strs_b) is byte-identical to omitting row maps.

    Guards the ``row_map_a=row_map_b=None`` regression: the selected-CI row
    maps are a no-op on the full canonical product, so the RDMs must match to
    the bit -- mirrors
    ``test_spinless_complex_ci.py::test_spinless_selected_subspace_full_canonical_bit_identical``.
    """
    from pyscf.fci import cistring

    norb, n_a, n_b = 4, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=2)
    strs_a = np.asarray(cistring.make_strings(range(norb), n_a), dtype=np.int64)
    strs_b = np.asarray(cistring.make_strings(range(norb), n_b), dtype=np.int64)

    prob = build_selected_complex_ci_problem(
        strs_a, strs_b, hcore, eri, norb, (n_a, n_b), device="cpu"
    )
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0].reshape(len(strs_a), len(strs_b))

    row_map_a = SparseAddressMap(strs_a, np.arange(len(strs_a)))
    row_map_b = SparseAddressMap(strs_b, np.arange(len(strs_b)))

    dm1_none, dm2_none, gab_none = make_rdm12_complex(C, norb, (n_a, n_b))
    dm1_full, dm2_full, gab_full = make_rdm12_complex(
        C, norb, (n_a, n_b), row_map_a=row_map_a, row_map_b=row_map_b, strs_a=strs_a, strs_b=strs_b
    )
    np.testing.assert_allclose(dm1_none, dm1_full, atol=1e-12)
    np.testing.assert_allclose(dm2_none, dm2_full, atol=1e-12)
    np.testing.assert_allclose(gab_none, gab_full, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 2, 5, 1_000_000])
def test_product_mode_rdm_chunked_matches_unchunked(chunk):
    """The chunked product-mode RDM path reproduces the unchunked RDMs exactly.

    Chunking blocks Ba/Bb over the flattened (p, q) operator-index axis to cap
    peak Ba/Bb residency (see ``_make_rdm12_complex_product_chunked``); it
    must not change the result for any block size, from ``chunk=1`` (one
    (p, q) row at a time) to a block larger than norb**2 (a single block,
    i.e. equivalent to unchunked). Mirrors
    ``test_spinless_complex_ci.py::test_spinless_rdm12_chunked_matches_unchunked``.
    """
    norb, n_a, n_b = 5, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=21)
    sel_a = _selected_strings(norb, n_a, (0, 2, 4, 6))
    sel_b = _selected_strings(norb, n_b, (1, 3, 5, 7))

    prob = build_selected_complex_ci_problem(
        sel_a, sel_b, hcore, eri, norb, (n_a, n_b), device="cpu"
    )
    _, v = np.linalg.eigh(prob.sparse().toarray())
    C = v[:, 0].reshape(len(sel_a), len(sel_b))

    row_map_a = SparseAddressMap(sel_a, np.arange(len(sel_a)))
    row_map_b = SparseAddressMap(sel_b, np.arange(len(sel_b)))

    dm1_ref, dm2_ref, gab_ref = make_rdm12_complex(
        C,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
        chunk=-1,
    )
    dm1, dm2, gab = make_rdm12_complex(
        C,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
        chunk=chunk,
    )
    np.testing.assert_allclose(dm1, dm1_ref, atol=1e-9)
    np.testing.assert_allclose(dm2, dm2_ref, atol=1e-9)
    np.testing.assert_allclose(gab, gab_ref, atol=1e-9)

    dm1a_ref, (dma_ref, dmb_ref) = make_rdm1_complex(
        C,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
        chunk=-1,
    )
    dm1a, (dma, dmb) = make_rdm1_complex(
        C,
        norb,
        (n_a, n_b),
        row_map_a=row_map_a,
        row_map_b=row_map_b,
        strs_a=sel_a,
        strs_b=sel_b,
        chunk=chunk,
    )
    np.testing.assert_allclose(dm1a, dm1a_ref, atol=1e-9)
    np.testing.assert_allclose(dma, dma_ref, atol=1e-9)
    np.testing.assert_allclose(dmb, dmb_ref, atol=1e-9)


def test_arpack_solves_selected_product_subspace_end_to_end():
    """create_solver('arpack', ...).sci_solver_callable on a genuinely selected
    (strs_a, strs_b) no longer raises -- the exact shape that previously
    crashed with ``IndexError`` inside ``_apply_excitations_a`` (the RDM
    extraction call site dropped the selected-subspace row maps entirely).
    """
    norb, n_a, n_b = 5, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=3)
    sel_a = _selected_strings(norb, n_a, (0, 2, 4, 6))
    sel_b = _selected_strings(norb, n_b, (1, 3, 5, 7))

    prob = build_selected_complex_ci_problem(
        sel_a, sel_b, hcore, eri, norb, (n_a, n_b), device="cpu"
    )
    w = np.linalg.eigvalsh(prob.sparse().toarray())

    solver = create_solver("arpack", symmetrize_spin=False)
    results = solver.sci_solver_callable([(sel_a, sel_b)], hcore, eri, norb, (n_a, n_b))

    assert np.isclose(results[0].energy, w[0], atol=1e-6)
    np.testing.assert_allclose(results[0].rdm1, results[0].rdm1.conj().T, atol=1e-8)
    assert np.isclose(results[0].rdm1.trace().real, n_a + n_b, atol=1e-6)


@pytest.mark.parametrize("name", ["arpack", "lobpcg", "davidson"])
def test_complex_capable_solvers_agree_on_selected_product_subspace(name):
    """Every complex_capable general-algebra solver agrees on the same selected subspace."""
    norb, n_a, n_b = 5, 2, 2
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=4)
    sel_a = _selected_strings(norb, n_a, (0, 2, 4, 6))
    sel_b = _selected_strings(norb, n_b, (1, 3, 5, 7))

    prob = build_selected_complex_ci_problem(
        sel_a, sel_b, hcore, eri, norb, (n_a, n_b), device="cpu"
    )
    w = np.linalg.eigvalsh(prob.sparse().toarray())

    solver = create_solver(name, symmetrize_spin=False)
    results = solver.sci_solver_callable([(sel_a, sel_b)], hcore, eri, norb, (n_a, n_b))
    assert np.isclose(results[0].energy, w[0], atol=1e-6), name
