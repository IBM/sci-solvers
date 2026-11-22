"""Pin the fulqrum conventions that ``hamiltonian_backend="fulqrum"`` relies on.

fulqrum builds the projection of a Hamiltonian onto a set of bitstrings. Each test
fixes one convention the builders in ``sci_solvers.fulqrum_ci_problem`` depend on, checked operator-level
(entrywise, not just spectrally) against the sci-solvers builders on synthetic
integrals:

* **Half-string (TPB) layout:** ``Subspace([alpha_strs, beta_strs])`` indexes the
  product as ``ib * n_a + ia`` (beta-major), so a vector maps to
  ``vec.reshape(n_b, n_a).T``; duplicate strings are *not* removed.
* **Signs:** fulqrum's basis differs from the PySCF spin-block one by one global
  sign, so no per-determinant fix is needed.
* **Complex ERIs:** ``integrals_to_fq_fermionic_op`` reads a complex chemist ERI
  conjugated; passing ``eri.transpose(1, 0, 3, 2)`` (``== conj(eri)``) is exact.
* **Full-string (GDB) layout:** ``Subspace([full_strs])`` sorts by the full
  ``beta + alpha`` string, i.e. ``np.lexsort((alpha, beta))``; converting to the
  interleaved-Wick ``PairCIProblem`` signs takes ``sigma``.
* **Spinless:** a hand-built spin-orbital ``FermionicOperator`` takes the raw ERI
  (the conjugation is specific to ``integrals_to_fq_fermionic_op``).

Skipped unless the optional ``fulqrum`` extra is installed
(``uv sync --extra fulqrum``).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools
from typing import Any

import numpy as np
import pytest
import torch
from test_complex_tpb_double_excitations import _full_strs, _genuinely_complex_integrals
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

from sci_solvers.complex_ci_problem import build_complex_ci_problem
from sci_solvers.eigen_problem import EigenProblem, PairCIProblem
from sci_solvers.spinless_ci_problem import build_spinless_complex_ci_problem

fq = pytest.importorskip("fulqrum")
from fulqrum.convert.integrals import integrals_to_fq_fermionic_op  # noqa: E402

NORB = 4
NELEC = (2, 2)


def _real_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Real symmetric ``hcore`` and a real 8-fold-symmetric chemist ERI."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    eri = np.zeros((norb,) * 4)
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb))
        eri += np.einsum("pq,rs->pqrs", g + g.T, g + g.T)
    return h + h.T, eri


def _bits(x: int, width: int) -> str:
    """fulqrum bitstring of ``x``: rightmost character is mode 0."""
    return format(int(x), f"0{width}b")


def _hsub(hcore: np.ndarray, eri: np.ndarray) -> Any:
    """fulqrum ``SubspaceHamiltonian`` for spatial integrals (no constant)."""
    fop = integrals_to_fq_fermionic_op(np.ascontiguousarray(hcore), np.ascontiguousarray(eri))
    return fq.SubspaceHamiltonian(fop.extended_jw_transformation())


def _dense(hsub: Any, subspace: Any) -> np.ndarray:
    """Dense projected matrix of ``hsub`` on ``subspace``, in fulqrum's basis order."""
    hsub.update_subspace(subspace)
    op = hsub.to_csr_linearoperator()
    return op @ np.eye(op.shape[0], dtype=op.dtype)


def _assert_equal_up_to_global_sign(a: np.ndarray, b: np.ndarray) -> None:
    """``a == +b`` or ``a == -b`` entrywise (one sign for the whole matrix)."""
    sign = 1.0 if np.abs(a - b).max() <= np.abs(a + b).max() else -1.0
    np.testing.assert_allclose(a, sign * b, atol=1e-10, rtol=0)


def _tpb_to_fq(n_a: int, n_b: int) -> np.ndarray:
    """``perm[k]`` = sci-solvers ``ia * Nb + ib`` row of fulqrum row ``k = ib * Na + ia``."""
    return np.array([ia * n_b + ib for ib in range(n_b) for ia in range(n_a)])


def _selected_strs() -> tuple[np.ndarray, np.ndarray]:
    """A selected (non-full) alpha/beta string pair, ascending."""
    strs_a, strs_b = _full_strs(NORB, NELEC)
    rng = np.random.default_rng(1)
    return (
        np.sort(rng.choice(strs_a, len(strs_a) - 2, replace=False)),
        np.sort(rng.choice(strs_b, len(strs_b) - 1, replace=False)),
    )


def test_half_string_index_is_beta_major() -> None:
    """The product of ``(alpha, beta)`` half strings is indexed ``ib * n_a + ia``."""
    strs_a, strs_b = _selected_strs()
    sub = fq.Subspace([[_bits(x, NORB) for x in strs_a], [_bits(x, NORB) for x in strs_b]])
    assert len(sub) == len(strs_a) * len(strs_b)
    for ia, a in enumerate(strs_a):
        for ib, b in enumerate(strs_b):
            assert sub.get_bitstring_index(_bits(b, NORB) + _bits(a, NORB)) == ib * len(strs_a) + ia


def test_duplicate_strings_are_not_removed() -> None:
    """fulqrum keeps duplicate half strings, so the adapter must deduplicate first."""
    strs_a, strs_b = _full_strs(NORB, NELEC)
    alpha = [_bits(strs_a[0], NORB), _bits(strs_a[0], NORB), _bits(strs_a[1], NORB)]
    assert len(fq.Subspace([alpha, [_bits(strs_b[0], NORB)]])) == 3


def test_real_tpb_matches_eigen_problem() -> None:
    """Real TPB: fulqrum's matrix, permuted to ``ia * Nb + ib``, is ``EigenProblem``'s up to one sign."""
    hcore, eri = _real_integrals(NORB)
    strs_a, strs_b = _selected_strs()
    sub = fq.Subspace([[_bits(x, NORB) for x in strs_a], [_bits(x, NORB) for x in strs_b]])
    got = _dense(_hsub(hcore, eri), sub)
    assert got.dtype == np.float64
    ref = EigenProblem(strs_a, strs_b, hcore, eri, NORB, NELEC).sparse().toarray()
    perm = _tpb_to_fq(len(strs_a), len(strs_b))
    _assert_equal_up_to_global_sign(got, ref[np.ix_(perm, perm)])


@pytest.mark.parametrize("nelec", [(2, 1), (2, 2)])
def test_complex_eri_needs_transpose(nelec: tuple[int, int]) -> None:
    """Complex TPB: the raw ERI gives a conjugated operator; ``eri.transpose(1, 0, 3, 2)`` is exact."""
    hcore, eri = _genuinely_complex_integrals(NORB, seed=5)
    strs_a, strs_b = _full_strs(NORB, nelec)
    sub = fq.Subspace([[_bits(x, NORB) for x in strs_a], [_bits(x, NORB) for x in strs_b]])
    ref = build_complex_ci_problem(hcore, eri, NORB, nelec).sparse().toarray()
    perm = _tpb_to_fq(len(strs_a), len(strs_b))
    ref = ref[np.ix_(perm, perm)]

    raw = _dense(_hsub(hcore, eri), sub)
    assert raw.dtype == np.complex128
    assert min(np.abs(raw - ref).max(), np.abs(raw + ref).max()) > 1e-3
    _assert_equal_up_to_global_sign(_dense(_hsub(hcore, eri.transpose(1, 0, 3, 2)), sub), ref)


@pytest.mark.parametrize("dtype", ["real", "complex"])
def test_gdb_full_strings_match_full_space_block(dtype: str) -> None:
    """GDB: full strings sort as ``lexsort((alpha, beta))``; the matrix is the full-space block."""
    if dtype == "real":
        hcore, eri = _real_integrals(NORB, seed=2)
        fq_eri = eri
    else:
        hcore, eri = _genuinely_complex_integrals(NORB, seed=2)
        fq_eri = eri.transpose(1, 0, 3, 2)
    strs_a, strs_b = _full_strs(NORB, NELEC)
    rng = np.random.default_rng(3)
    all_pairs = [(ia, ib) for ia in range(len(strs_a)) for ib in range(len(strs_b))]
    picked = sorted(all_pairs[k] for k in rng.choice(len(all_pairs), 20, replace=False))
    pa = strs_a[[ia for ia, _ in picked]]
    pb = strs_b[[ib for _, ib in picked]]
    rows = np.array([ia * len(strs_b) + ib for ia, ib in picked])  # full-space rows

    sub = fq.Subspace([[_bits(b, NORB) + _bits(a, NORB) for a, b in zip(pa, pb)]])
    got = _dense(_hsub(hcore, fq_eri), sub)
    order = np.lexsort((pa, pb))  # fulqrum row k is input pair order[k]
    fq_strs = [sub.get_n_th_bitstring(k) for k in range(len(pa))]
    assert fq_strs == [_bits(pb[i], NORB) + _bits(pa[i], NORB) for i in order]

    if dtype == "real":
        full = EigenProblem(strs_a, strs_b, hcore, eri, NORB, NELEC).sparse().toarray()
    else:
        full = build_complex_ci_problem(hcore, eri, NORB, NELEC).sparse().toarray()
    block = full[np.ix_(rows, rows)]
    _assert_equal_up_to_global_sign(got, block[np.ix_(order, order)])


def test_gdb_sigma_maps_to_pair_problem() -> None:
    """Real GDB: ``sigma`` converts fulqrum's spin-block matrix to ``PairCIProblem``'s signs."""
    hcore, eri = _real_integrals(NORB, seed=4)
    strs_a, strs_b = _full_strs(NORB, NELEC)
    rng = np.random.default_rng(5)
    pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
    pairs = np.unique(pairs[rng.choice(len(pairs), 20, replace=False)], axis=0)  # problem order
    pa, pb = pairs[:, 0], pairs[:, 1]

    sub = fq.Subspace([[_bits(b, NORB) + _bits(a, NORB) for a, b in zip(pa, pb)]])
    got = _dense(_hsub(hcore, eri), sub)
    inv = np.argsort(np.lexsort((pa, pb)))  # problem row i is fulqrum row inv[i]
    got = got[np.ix_(inv, inv)]
    ref = PairCIProblem(pa, pb, hcore, eri, NORB, NELEC).sparse().toarray()
    sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), NORB).numpy()
    _assert_equal_up_to_global_sign(got, sigma[:, None] * ref * sigma[None, :])


@pytest.mark.parametrize("selected", [False, True])
def test_spinless_hand_built_operator_takes_raw_eri(selected: bool) -> None:
    """Spinless: a spin-orbital ``FermionicOperator`` built from the raw ERI matches sci-solvers."""
    nmo, n = 5, 2
    hcore, eri = _genuinely_complex_integrals(nmo, seed=6)
    dets = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(nmo), n)),
        dtype=np.int64,
    )
    if selected:
        dets = np.sort(np.random.default_rng(7).choice(dets, len(dets) - 3, replace=False))

    terms = [
        ("+-", [p, q], complex(hcore[p, q])) for p, q in itertools.product(range(nmo), repeat=2)
    ]
    terms += [
        ("++--", [p, r, s, q], complex(0.5 * eri[p, q, r, s]))
        for p, q, r, s in itertools.product(range(nmo), repeat=4)
    ]
    fop = fq.FermionicOperator(nmo, terms)
    fop.combine_repeat_terms()
    hsub = fq.SubspaceHamiltonian(fop.extended_jw_transformation())
    got = _dense(hsub, fq.Subspace([[_bits(d, nmo) for d in dets]]))

    ref = build_spinless_complex_ci_problem(hcore, eri, n, dets).sparse().toarray()
    _assert_equal_up_to_global_sign(got, ref)
