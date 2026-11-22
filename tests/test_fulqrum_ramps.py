"""``ramps_expand`` and ``ramps_expand_spinless``: fulqrum's RAMPS on sci-solvers determinants.

Synthetic integrals only. RAMPS adds Hamiltonian-connected determinants with a large
perturbative energy estimate. These tests pin the wrappers' contract (core kept and
first, particle numbers kept, string conversion round-trips, deduplication, inputs
validated) and their usefulness (the energy over the expanded set improves, and
iterating "solve, then expand" reaches the full-CI energy). They cover real and genuinely
complex spatial integrals and the spinless path. Skipped unless the optional ``fulqrum``
extra is installed.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import pytest
from pyscf import fci
from test_complex_tpb_double_excitations import _genuinely_complex_integrals
from test_fulqrum_backend import _real_integrals
from test_spinless_complex_ci import _all_dets, _brute_force_dense
from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

pytest.importorskip("fulqrum")

from sci_solvers.complex_ci_problem import build_complex_ci_problem  # noqa: E402
from sci_solvers.energy_solvers.factory import create_solver  # noqa: E402
from sci_solvers.fulqrum_ramps import ramps_expand, ramps_expand_spinless  # noqa: E402

NORB = 6
NELEC = (3, 3)
HF = ((1 << NELEC[0]) - 1, (1 << NELEC[1]) - 1)


def _gdb_energy(hcore, eri, strs_a, strs_b, nelec=NELEC) -> float:
    """Ground energy over the explicit determinant pairs (``arpack``, ``mode="gdb"``)."""
    solver = create_solver(
        "arpack", symmetrize_spin=False, mode="gdb", hamiltonian_backend="fulqrum"
    )
    return solver.sci_solver_callable([(strs_a, strs_b)], hcore, eri, NORB, nelec)[0].energy


def _spinless_energy(hcore, eri, strs, nmo, n) -> float:
    """Ground energy over the spinless determinants (``arpack``, sorted input)."""
    solver = create_solver("arpack", symmetrize_spin=False, hamiltonian_backend="fulqrum")
    return solver.sci_solver_callable([np.sort(strs)], hcore, eri, nmo, n)[0].energy


def _integrals(kind: str, seed: int):
    """Real, or genuinely complex Hermitian, spatial integrals for ``NORB`` orbitals."""
    return (
        _real_integrals(NORB, seed) if kind == "real" else _genuinely_complex_integrals(NORB, seed)
    )


def _full_ci(kind: str, hcore, eri, nelec) -> float:
    """Full-CI ground energy, real or complex."""
    if kind == "real":
        return float(fci.direct_spin1.kernel(hcore, eri, NORB, nelec)[0])
    return float(
        np.linalg.eigvalsh(build_complex_ci_problem(hcore, eri, NORB, nelec).sparse().toarray())[0]
    )


@pytest.fixture(scope="module")
def system() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Real integrals, their full-CI energy, and the HF determinant's energy."""
    hcore, eri = _real_integrals(NORB, seed=11)
    e_fci = _full_ci("real", hcore, eri, NELEC)
    e_hf = float(_gdb_energy(hcore, eri, np.array([HF[0]]), np.array([HF[1]])))
    return hcore, eri, e_fci, e_hf


def test_core_is_kept_first_and_particle_numbers_preserved(system) -> None:
    """The output starts with the core, in input order, and every determinant has (3, 3)."""
    hcore, eri, _, e_hf = system
    core_a = np.array([HF[0], 0b001011, 0b010101], dtype=np.int64)
    core_b = np.array([HF[1], HF[1], 0b100011], dtype=np.int64)
    out_a, out_b = ramps_expand(hcore, eri, core_a, core_b, e_hf, tol=1e-6)
    assert out_a.dtype == np.int64 and out_b.dtype == np.int64
    assert len(out_a) == len(out_b) > len(core_a)
    np.testing.assert_array_equal(out_a[:3], core_a)
    np.testing.assert_array_equal(out_b[:3], core_b)
    assert {bin(int(x)).count("1") for x in out_a} == {NELEC[0]}
    assert {bin(int(x)).count("1") for x in out_b} == {NELEC[1]}
    assert len(set(zip(out_a.tolist(), out_b.tolist()))) == len(out_a)  # no duplicates


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_open_shell_keeps_alpha_and_beta_apart(kind: str) -> None:
    """Open shell (3, 2): additions keep 3 alpha and 2 beta electrons, never swapped.

    A closed-shell system cannot detect a swapped alpha/beta split in the bitstring
    conversion, because both halves have the same electron count.
    """
    nelec = (3, 2)
    hcore, eri = _integrals(kind, seed=12)
    hf_a, hf_b = np.array([(1 << 3) - 1]), np.array([(1 << 2) - 1])
    e_hf = _gdb_energy(hcore, eri, hf_a, hf_b, nelec)
    out_a, out_b = ramps_expand(hcore, eri, hf_a, hf_b, e_hf, tol=1e-6)
    assert len(out_a) > 1
    assert {bin(int(x)).count("1") for x in out_a} == {3}
    assert {bin(int(x)).count("1") for x in out_b} == {2}
    assert _gdb_energy(hcore, eri, out_a, out_b, nelec) < e_hf


def test_duplicate_core_pairs_are_collapsed(system) -> None:
    """Repeated core pairs appear once; the result equals the deduplicated call's."""
    hcore, eri, _, e_hf = system
    once = ramps_expand(hcore, eri, np.array([HF[0]]), np.array([HF[1]]), e_hf, tol=1e-6)
    twice = ramps_expand(
        hcore, eri, np.array([HF[0], HF[0]]), np.array([HF[1], HF[1]]), e_hf, tol=1e-6
    )
    np.testing.assert_array_equal(once[0], twice[0])
    np.testing.assert_array_equal(once[1], twice[1])


@pytest.mark.parametrize("spinless", [False, True])
def test_two_determinant_core(system, spinless: bool) -> None:
    """A core of exactly two determinants (which fulqrum's Subspace.copy() cannot copy).

    The result is the union of the two single-determinant expansions: with one layer,
    each addition depends only on the determinant it is connected to.
    """
    if spinless:
        nmo, n = 8, 4
        hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
        core = np.array([0b00001111, 0b00010111])
        energy = _spinless_energy(hcore, eri, core, nmo, n)
        out = ramps_expand_spinless(hcore, eri, core, energy, tol=1e-6)
        union = set().union(
            *(
                ramps_expand_spinless(hcore, eri, core[i : i + 1], energy, tol=1e-6).tolist()
                for i in range(2)
            )
        )
        got = out.tolist()
    else:
        hcore, eri, _, e_hf = system
        core_a = np.array([HF[0], 0b001011])
        core_b = np.array([HF[1], HF[1]])
        out_a, out_b = ramps_expand(hcore, eri, core_a, core_b, e_hf, tol=1e-6)
        union = set().union(
            *(
                set(
                    zip(
                        *ramps_expand(
                            hcore, eri, core_a[i : i + 1], core_b[i : i + 1], e_hf, tol=1e-6
                        )
                    )
                )
                for i in range(2)
            )
        )
        got = list(zip(out_a.tolist(), out_b.tolist()))
        core = list(zip(core_a.tolist(), core_b.tolist()))
    assert list(got[:2]) == list(core if not spinless else core.tolist())
    assert len(set(got)) == len(got)
    assert set(got) == union


def test_smaller_tol_keeps_more_and_lowers_the_energy(system) -> None:
    """A tighter threshold gives a superset with a lower (or equal) variational energy."""
    hcore, eri, _, e_hf = system
    hf_a, hf_b = np.array([HF[0]]), np.array([HF[1]])
    loose = ramps_expand(hcore, eri, hf_a, hf_b, e_hf, tol=1e-3)
    tight = ramps_expand(hcore, eri, hf_a, hf_b, e_hf, tol=1e-6)
    assert set(zip(*loose)) <= set(zip(*tight))
    assert len(tight[0]) > len(loose[0]) > 1
    assert _gdb_energy(hcore, eri, *tight) <= _gdb_energy(hcore, eri, *loose) + 1e-12
    assert _gdb_energy(hcore, eri, *loose) < e_hf


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_iterated_expansion_reaches_full_ci(kind: str) -> None:
    """Solve, expand around the result, repeat: the energy reaches full CI, monotonically."""
    hcore, eri = _integrals(kind, seed=11)
    e_fci = _full_ci(kind, hcore, eri, NELEC)
    strs_a, strs_b = np.array([HF[0]]), np.array([HF[1]])
    energy = _gdb_energy(hcore, eri, strs_a, strs_b)
    energies = [energy]
    for _ in range(6):
        strs_a, strs_b = ramps_expand(hcore, eri, strs_a, strs_b, energy, tol=1e-8)
        energy = _gdb_energy(hcore, eri, strs_a, strs_b)
        energies.append(energy)
    assert all(b <= a + 1e-10 for a, b in zip(energies, energies[1:]))
    assert energies[-1] == pytest.approx(e_fci, abs=1e-8)


def test_spinless_iterated_expansion_reaches_full_ci() -> None:
    """Spinless: solve-then-expand keeps N particles, the core first, and reaches full CI."""
    nmo, n = 8, 4
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
    e_fci = float(np.linalg.eigvalsh(_brute_force_dense(hcore, eri, nmo, _all_dets(nmo, n)))[0])
    core = np.array([(1 << n) - 1], dtype=np.int64)
    strs, energy = core, _spinless_energy(hcore, eri, core, nmo, n)
    energies = [energy]
    for _ in range(6):
        strs = ramps_expand_spinless(hcore, eri, strs, energy, tol=1e-8)
        assert strs.dtype == np.int64 and strs[0] == core[0]
        assert {bin(int(x)).count("1") for x in strs} == {n}
        assert len(set(strs.tolist())) == len(strs)
        energy = _spinless_energy(hcore, eri, strs, nmo, n)
        energies.append(energy)
    assert all(b <= a + 1e-10 for a, b in zip(energies, energies[1:]))
    assert energies[-1] == pytest.approx(e_fci, abs=1e-8)


def test_spinless_core_is_kept_first_in_input_order() -> None:
    """A multi-determinant core comes back first, in its input (not sorted) order."""
    nmo, n = 8, 4
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
    core = np.array([0b11100001, 0b00001111, 0b00110011, 0b00001111], dtype=np.int64)
    unique_core = np.array([0b11100001, 0b00001111, 0b00110011], dtype=np.int64)
    energy = _spinless_energy(hcore, eri, unique_core, nmo, n)
    out = ramps_expand_spinless(hcore, eri, core, energy, tol=1e-6)
    np.testing.assert_array_equal(out[:3], unique_core)
    assert len(out) > 3 and len(set(out.tolist())) == len(out)


def test_rejects_bad_input(system) -> None:
    """Mismatched, mixed-count or out-of-range strings, bad ``tol``/``max_recursion``, non-Hermitian integrals."""
    hcore, eri, _, e_hf = system
    a, b = np.array([HF[0]]), np.array([HF[1]])
    with pytest.raises(ValueError, match="parallel"):
        ramps_expand(hcore, eri, a, np.array([HF[1], HF[1]]), e_hf)
    with pytest.raises(ValueError, match="parallel"):
        ramps_expand(hcore, eri, np.array([], dtype=np.int64), np.array([], dtype=np.int64), e_hf)
    with pytest.raises(ValueError, match="strs_a must have the same particle number"):
        ramps_expand(hcore, eri, np.array([HF[0], 0b1111]), np.array([HF[1], HF[1]]), e_hf)
    with pytest.raises(ValueError, match="strs_b must have the same particle number"):
        ramps_expand(hcore, eri, np.array([HF[0], HF[0]]), np.array([HF[1], 0b11]), e_hf)
    wide = HF[0] | 1 << NORB  # one bit past the alpha half
    with pytest.raises(ValueError, match=r"strs_a must be in \[0, 2\*\*6\)"):
        ramps_expand(hcore, eri, np.array([wide]), b, e_hf)
    with pytest.raises(ValueError, match=r"strs_b must be in \[0, 2\*\*6\)"):
        ramps_expand(hcore, eri, a, np.array([-1]), e_hf)
    with pytest.raises(ValueError, match="tol must be positive"):
        ramps_expand(hcore, eri, a, b, e_hf, tol=0.0)
    with pytest.raises(ValueError, match="max_recursion"):
        ramps_expand(hcore, eri, a, b, e_hf, max_recursion=0)
    hc, ec = _genuinely_complex_integrals(NORB)
    ec = ec.copy()
    ec[0, 1, 2, 3] += 0.5j
    with pytest.raises(ValueError, match="Hermitian|does not satisfy"):
        ramps_expand(hc, ec, a, b, e_hf)


def test_spinless_rejects_bad_input() -> None:
    """Spinless guards: real integrals; empty, mixed-particle-number or too-wide lists."""
    hcore, eri = _genuinely_complex_spinorbital_integrals(6, seed=1)
    with pytest.raises(ValueError, match="real integrals"):
        ramps_expand_spinless(hcore.real, eri.real, np.array([7]), -1.0)
    with pytest.raises(ValueError, match="at least one"):
        ramps_expand_spinless(hcore, eri, np.array([], dtype=np.int64), -1.0)
    with pytest.raises(ValueError, match="same particle number"):
        ramps_expand_spinless(hcore, eri, np.array([7, 3]), -1.0)
    with pytest.raises(ValueError, match=r"strs must be in \[0, 2\*\*6\)"):
        ramps_expand_spinless(hcore, eri, np.array([0b1000011]), -1.0)
