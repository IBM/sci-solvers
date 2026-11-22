"""The complex selected-CI TPB operator on a genuinely complex ERI.

Each double excitation must read the ERI with the creation index first in each
chemist pair, ``(cr an|cr an)``; the swapped order is its complex conjugate. The two
are equal for real 8-fold-symmetric integrals and both give a Hermitian operator, so
only an ERI with a nonzero imaginary part (asserted by the fixture guard) can tell
them apart. ``tests/test_complex_hermitian.py::_random_complex_hermitian_integrals``
produces an exactly real ERI and so does not cover this.

References, all independent of the TPB kernel:

* a brute-force Jordan-Wigner Fock-space build of
  ``H = sum h_pq a+_p a_q + 1/2 sum (pq|rs) a+_p a+_r a_s a_q``,
* :func:`~sci_solvers.complex_ci_problem.build_complex_ci_problem` (PySCF
  ``direct_nosym``, full space),
* :class:`~sci_solvers.eigen_problem.PairCIProblem` (the GDB pair kernel).

Every operator check is entrywise, not just spectral: a Hermitian ``H`` and its
conjugate ``conj(H) = H.T`` share one spectrum, so eigenvalues alone cannot tell a
correct operator from a conjugated one.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools

import numpy as np
import pytest
import torch
from pyscf.fci import cistring
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

from sci_solvers.complex_ci_problem import (
    build_complex_ci_problem,
    build_selected_complex_ci_problem,
)
from sci_solvers.eigen_problem import EigenProblem, PairCIProblem
from sci_solvers.energy_solvers.factory import create_solver

NORB = 4
NELECS = [(2, 1), (2, 2)]


def _genuinely_complex_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian ``hcore`` and a chemist ERI with a nonzero imaginary part.

    Enforces exactly the two symmetries the CI operator needs, ``(pq|rs) ==
    conj((qp|sr))`` and ``(pq|rs) == (rs|pq)``, and nothing more, so the ERI keeps
    its imaginary part (unlike a density-fit ``L x conj(L)`` form symmetrized over
    ``(pq)<->(rs)``, which collapses to a real tensor).
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    eri = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    eri = eri + eri.conj().transpose(1, 0, 3, 2)
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hcore, eri


def _brute_force_tpb_matrix(
    hcore: np.ndarray, eri: np.ndarray, norb: int, strs_a: np.ndarray, strs_b: np.ndarray
) -> np.ndarray:
    """Dense ``H`` on the ``strs_a x strs_b`` product, built in Fock space via Jordan-Wigner.

    Modes ``0..norb-1`` are alpha and ``norb..2*norb-1`` beta; the returned matrix is in
    the sci-solvers ``ia * Nb + ib`` order. The Jordan-Wigner basis differs from the
    PySCF spin-block one by at most a per-determinant sign, so compare spectra.
    """
    n = 2 * norb
    dim = 2**n
    ann = []
    for j in range(n):
        a = np.zeros((dim, dim))
        for x in range(dim):
            if x >> j & 1:
                a[x ^ (1 << j), x] = (-1) ** bin(x & ((1 << j) - 1)).count("1")
        ann.append(a)
    cre = [a.T for a in ann]
    h = np.zeros((dim, dim), dtype=complex)
    for s in (0, norb):
        for p, q in itertools.product(range(norb), repeat=2):
            h += hcore[p, q] * cre[p + s] @ ann[q + s]
    for s, t in itertools.product((0, norb), repeat=2):
        for p, q, r, u in itertools.product(range(norb), repeat=4):
            h += 0.5 * eri[p, q, r, u] * cre[p + s] @ cre[r + t] @ ann[u + t] @ ann[q + s]
    dets = [int(b) << norb | int(a) for a in strs_a for b in strs_b]
    return h[np.ix_(dets, dets)]


def _full_strs(norb: int, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Every alpha and every beta string, ascending (PySCF canonical order)."""
    return (
        np.asarray(cistring.make_strings(range(norb), nelec[0]), dtype=np.int64),
        np.asarray(cistring.make_strings(range(norb), nelec[1]), dtype=np.int64),
    )


def test_eri_is_genuinely_complex() -> None:
    """Guard: the fixture really carries an imaginary ERI part and passes the builders' checks."""
    hcore, eri = _genuinely_complex_integrals(NORB)
    assert np.abs(eri.imag).max() > 0.1
    build_complex_ci_problem(hcore, eri, NORB, NELECS[0])  # raises if not Hermitian


@pytest.mark.parametrize("nelec", NELECS)
def test_selected_tpb_matches_full_space_builder(nelec: tuple[int, int]) -> None:
    """On the full product, the selected TPB operator equals PySCF's full-space one entrywise."""
    hcore, eri = _genuinely_complex_integrals(NORB, seed=1)
    strs_a, strs_b = _full_strs(NORB, nelec)
    selected = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, NORB, nelec)
    full = build_complex_ci_problem(hcore, eri, NORB, nelec)
    np.testing.assert_allclose(
        selected.sparse().toarray(), full.sparse().toarray(), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("nelec", NELECS)
def test_selected_tpb_spectrum_matches_brute_force(nelec: tuple[int, int]) -> None:
    """The selected TPB spectrum equals the brute-force Fock-space spectrum."""
    hcore, eri = _genuinely_complex_integrals(NORB, seed=2)
    strs_a, strs_b = _full_strs(NORB, nelec)
    selected = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, NORB, nelec)
    ref = _brute_force_tpb_matrix(hcore, eri, NORB, strs_a, strs_b)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(selected.sparse().toarray()), np.linalg.eigvalsh(ref), atol=1e-10
    )


def _subset(strs_a: np.ndarray, strs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A selected (non-full) product subset and its rows in the full ``ia * Nb + ib`` basis."""
    rng = np.random.default_rng(0)
    ia = np.sort(rng.choice(len(strs_a), len(strs_a) - 2, replace=False))
    ib = np.sort(rng.choice(len(strs_b), len(strs_b) - 1, replace=False))
    rows = (ia[:, None] * len(strs_b) + ib[None, :]).reshape(-1)
    return strs_a[ia], strs_b[ib], rows


@pytest.mark.parametrize("nelec", NELECS)
def test_selected_tpb_subset_matches_full_space_block(nelec: tuple[int, int]) -> None:
    """On a selected (non-full) subset, the TPB matrix is the matching block of the full one.

    A matrix element depends only on its two determinants, so the selected operator must
    equal the full-space PySCF operator restricted to the selected rows and columns.
    """
    hcore, eri = _genuinely_complex_integrals(NORB, seed=3)
    strs_a, strs_b = _full_strs(NORB, nelec)
    sub_a, sub_b, rows = _subset(strs_a, strs_b)
    tpb = build_selected_complex_ci_problem(sub_a, sub_b, hcore, eri, NORB, nelec)
    full = build_complex_ci_problem(hcore, eri, NORB, nelec).sparse().toarray()
    np.testing.assert_allclose(tpb.sparse().toarray(), full[np.ix_(rows, rows)], atol=1e-10, rtol=0)


def _real_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Real symmetric ``hcore`` and a real 8-fold-symmetric chemist ERI."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    eri = np.zeros((norb,) * 4)
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb))
        eri += np.einsum("pq,rs->pqrs", g + g.T, g + g.T)
    return h + h.T, eri


def _integrals(kind: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Genuinely complex, or real, integrals for ``NORB`` orbitals."""
    if kind == "complex":
        return _genuinely_complex_integrals(NORB, seed=seed)
    return _real_integrals(NORB, seed=seed)


def _tpb_operator(kind, sub_a, sub_b, hcore, eri, nelec) -> np.ndarray:
    """Dense selected TPB operator for real or complex integrals."""
    if kind == "complex":
        return (
            build_selected_complex_ci_problem(sub_a, sub_b, hcore, eri, NORB, nelec)
            .sparse()
            .toarray()
        )
    return EigenProblem(sub_a, sub_b, hcore, eri, NORB, nelec).sparse().toarray()


@pytest.mark.parametrize("kind", ["complex", "real"])
@pytest.mark.parametrize("nelec", NELECS)
def test_pair_kernel_is_sigma_h_sigma_of_selected_tpb(nelec: tuple[int, int], kind: str) -> None:
    """The GDB pair operator is the TPB operator in interleaved-Wick signs, ``sigma . H . sigma``.

    On the same determinants (pairs in TPB ``ia * Nb + ib`` order), with
    ``sigma`` = ``_inter_to_block_signs_pair``. Entrywise, so a conjugated operator
    (``sigma . conj(H) . sigma``, same spectrum) fails it on complex integrals.
    """
    hcore, eri = _integrals(kind, seed=3)
    strs_a, strs_b = _full_strs(NORB, nelec)
    sub_a, sub_b, _ = _subset(strs_a, strs_b)
    tpb = _tpb_operator(kind, sub_a, sub_b, hcore, eri, nelec)
    pairs = np.array([(a, b) for a in sub_a for b in sub_b], dtype=np.int64)
    gdb = PairCIProblem(pairs[:, 0], pairs[:, 1], hcore, eri, NORB, nelec)
    sigma = _inter_to_block_signs_pair(
        torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1]), NORB
    ).numpy()
    np.testing.assert_allclose(
        gdb.sparse().toarray(), sigma[:, None] * tpb * sigma[None, :], atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("kind", ["complex", "real"])
@pytest.mark.parametrize("nelec", NELECS)
def test_gdb_states_match_tpb(nelec: tuple[int, int], kind: str) -> None:
    """GDB and TPB give the same physical states on the same determinants.

    Two roots through ``arpack``: energies, ``rdm1``, and the amplitudes themselves,
    which agree after the interleaved-Wick sign ``sigma`` up to one global phase
    per root -- the convention-level check energies and RDMs alone cannot make.
    ``rdm2`` is compared for ``n_a == n_b`` only (see
    ``test_gdb_rdm2_matches_tpb_closed_shell``).
    """
    hcore, eri = _integrals(kind, seed=5)
    strs_a, strs_b = _full_strs(NORB, nelec)
    pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
    out = {}
    for mode, batch in (("tpb", (strs_a, strs_b)), ("gdb", (pairs[:, 0], pairs[:, 1]))):
        solver = create_solver("arpack", symmetrize_spin=False, mode=mode, nroots=2)
        out[mode] = solver.sci_solver_callable_eigen_states([batch], hcore, eri, NORB, nelec)[
            0
        ].states
    sigma = _inter_to_block_signs_pair(
        torch.as_tensor(pairs[:, 0]), torch.as_tensor(pairs[:, 1]), NORB
    ).numpy()
    for t, g in zip(out["tpb"], out["gdb"]):
        assert g.energy == pytest.approx(t.energy, abs=1e-8)
        np.testing.assert_allclose(g.rdm1, t.rdm1, atol=1e-6)
    # Ground state (non-degenerate): amplitudes, in the GDB basis order.
    t0, g0 = out["tpb"][0].sci_state, out["gdb"][0].sci_state
    tpb_amp = {
        (int(a), int(b)): np.asarray(t0.amplitudes)[ia, ib]
        for ia, a in enumerate(t0.ci_strs_a)
        for ib, b in enumerate(t0.ci_strs_b)
    }
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
    ref = np.array([
        sigma[key[(int(a), int(b))]] * tpb_amp[(int(a), int(b))]
        for a, b in zip(g0.ci_strs_a, g0.ci_strs_b)
    ])
    got = np.asarray(g0.amplitudes).reshape(-1)
    phase = np.vdot(ref, got) / abs(np.vdot(ref, got))
    np.testing.assert_allclose(got, phase * ref, atol=1e-6)


@pytest.mark.parametrize("kind", ["complex", "real"])
def test_gdb_rdm2_matches_tpb_closed_shell(kind: str) -> None:
    """GDB and TPB give the same spin-summed 2-RDM (``n_a == n_b``)."""
    nelec = (2, 2)
    hcore, eri = _integrals(kind, seed=6)
    strs_a, strs_b = _full_strs(NORB, nelec)
    pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
    out = {}
    for mode, batch in (("tpb", (strs_a, strs_b)), ("gdb", (pairs[:, 0], pairs[:, 1]))):
        solver = create_solver("arpack", symmetrize_spin=False, mode=mode, compute_rdm2=True)
        out[mode] = solver.sci_solver_callable_eigen_states([batch], hcore, eri, NORB, nelec)[
            0
        ].states[0]
    assert out["gdb"].energy == pytest.approx(out["tpb"].energy, abs=1e-8)
    np.testing.assert_allclose(out["gdb"].rdm2, out["tpb"].rdm2, atol=1e-6)


def test_qiskit_pair_coupling_diagnostic_matches_pair_operator(capsys) -> None:
    """``dump_pair_coupling_magnitudes`` evaluates the same off-diagonal elements as the matvec.

    The diagnostic reads the pair-helper tables with the same ``cr``/``an`` call order as
    ``mult_pairs`` (real integrals only), so the multiset of its matrix elements must
    equal the off-diagonal entries of the ``PairCIProblem`` operator, each counted once
    per ``(bra, ket)`` table row.
    """
    from torch_sci_solver import dump_pair_coupling_magnitudes
    from torch_sci_solver.torch_impl.pairs import PairSCIState

    nelec = (2, 2)
    hcore, eri = _real_integrals(NORB, seed=7)
    strs_a, strs_b = _full_strs(NORB, nelec)
    pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
    problem = PairCIProblem(pairs[:, 0], pairs[:, 1], hcore, eri, NORB, nelec)
    state = PairSCIState(
        amplitudes=np.zeros(problem.dim),
        ci_strs_a=problem.strs_a,
        ci_strs_b=problem.strs_b,
        norb=NORB,
        nelec=nelec,
    )
    summary = dump_pair_coupling_magnitudes(state, hcore, eri, norb=NORB, nelec=nelec)
    capsys.readouterr()
    off = problem.sparse().toarray()
    np.fill_diagonal(off, 0.0)
    total_sq = sum(v["sum_sq"] for v in summary.values())
    assert total_sq == pytest.approx(float((off**2).sum()), rel=1e-10)


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_arpack_energy_matches_brute_force(mode: str) -> None:
    """End to end: ``arpack`` TPB and GDB both reach the exact ground energy."""
    nelec = (2, 1)
    hcore, eri = _genuinely_complex_integrals(NORB, seed=4)
    strs_a, strs_b = _full_strs(NORB, nelec)
    e_exact = np.linalg.eigvalsh(_brute_force_tpb_matrix(hcore, eri, NORB, strs_a, strs_b))[0]
    if mode == "tpb":
        batch = (strs_a, strs_b)
    else:
        pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
        batch = (pairs[:, 0], pairs[:, 1])
    solver = create_solver("arpack", symmetrize_spin=False, mode=mode)
    energy = solver.sci_solver_callable([batch], hcore, eri, NORB, nelec)[0].energy
    assert energy == pytest.approx(e_exact, abs=1e-8)
