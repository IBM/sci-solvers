"""Tests for torch_sci_solver.sbd_ref.helpers.

Verifies:
  1. generate_excitation produces the same (bra, ket) excitation pairs
     as brute-force enumeration by ``difference``.
  2. CrAn orbital indices recover the right orbital differences.
  3. Task types 0, 1, 2 correctly include/exclude singles vs doubles
     and α vs β.
  4. Symmetry: SinglesFromAlpha[ia] contains ja iff SinglesFromAlpha[ja]
     contains ia (i.e., the excitation relation is symmetric on dets
     differing by 2 bits).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

from torch_sci_solver.sbd_ref import (
    difference,
    generate_excitation,
    orbital_difference,
)


def _enumerate_dets(L, n_elec):
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        mask = 0
        for bit in combo:
            mask |= 1 << bit
        out.append(mask)
    out.sort()
    return out


# ============================================================================
# 1+2+4. α-excitations: 6-orb 3-electron basis  (taskType=2 path)
# ============================================================================


def test_alpha_excitations_6orb_3e():
    """α-string excitation tables for a 6-orbital, 3-electron basis."""
    L = 6
    n_a = 3
    adets = _enumerate_dets(L, n_a)
    bdets = adets  # symmetric for the helper test; β table not used in taskType=2
    print(f"  basis: {len(adets)} α-strings, {len(bdets)} β-strings")

    h = generate_excitation(adets, bdets, L, task_type=2)

    # Brute-force ground truth: for each (ia, ja), label by diff
    n_singles_bf = 0
    n_doubles_bf = 0
    for ia in range(len(adets)):
        for ja in range(len(adets)):
            d = difference(adets[ia], adets[ja], L)
            if d == 2:
                n_singles_bf += 1
            elif d == 4:
                n_doubles_bf += 1

    n_singles_h = sum(len(x) for x in h.singles_from_alpha)
    n_doubles_h = sum(len(x) for x in h.doubles_from_alpha)
    print(f"  brute-force: {n_singles_bf} singles, {n_doubles_bf} doubles")
    print(f"  helper:      {n_singles_h} singles, {n_doubles_h} doubles")
    assert n_singles_h == n_singles_bf, f"singles mismatch: {n_singles_h} vs {n_singles_bf}"
    assert n_doubles_h == n_doubles_bf, f"doubles mismatch: {n_doubles_h} vs {n_doubles_bf}"

    # For each (ia, ja) recovered by helper, verify the CrAn matches what
    # orbital_difference reports — and the orbital indices are spin-orbital
    # (× 2 for α, no +1).
    for ia in range(len(adets)):
        for k, ja in enumerate(h.singles_from_alpha[ia]):
            cr_so, an_so = h.singles_alpha_cran[ia][k]
            assert cr_so % 2 == 0, f"α single cr_so={cr_so} should be even"
            assert an_so % 2 == 0, f"α single an_so={an_so} should be even"
            # Half indices = spatial orbital indices
            cr_spatial = cr_so // 2
            an_spatial = an_so // 2
            # Verify by direct comparison to orbital_difference
            x_ref, y_ref = orbital_difference(adets[ia], adets[ja], L)
            assert x_ref == [cr_spatial], (
                f"cr mismatch at (ia={ia}, ja={ja}): helper {cr_spatial} vs ref {x_ref}"
            )
            assert y_ref == [an_spatial], (
                f"an mismatch at (ia={ia}, ja={ja}): helper {an_spatial} vs ref {y_ref}"
            )

        for k, ja in enumerate(h.doubles_from_alpha[ia]):
            cr0_so, cr1_so, an0_so, an1_so = h.doubles_alpha_cran[ia][k]
            assert all(x % 2 == 0 for x in (cr0_so, cr1_so, an0_so, an1_so))
            x_ref, y_ref = orbital_difference(adets[ia], adets[ja], L)
            cr_spatial = [cr0_so // 2, cr1_so // 2]
            an_spatial = [an0_so // 2, an1_so // 2]
            assert sorted(x_ref) == sorted(cr_spatial), (
                f"cr mismatch at (ia={ia}, ja={ja}, double): {cr_spatial} vs {x_ref}"
            )
            assert sorted(y_ref) == sorted(an_spatial), (
                f"an mismatch at (ia={ia}, ja={ja}, double): {an_spatial} vs {y_ref}"
            )

    # 4. Excitation relation symmetry: ia→ja is single iff ja→ia is single
    pair_set_single = {(ia, ja) for ia in range(len(adets)) for ja in h.singles_from_alpha[ia]}
    for ia, ja in pair_set_single:
        assert (ja, ia) in pair_set_single, (
            f"asymmetric single excitation: ({ia},{ja}) in but ({ja},{ia}) not"
        )

    print("test_alpha_excitations_6orb_3e passed")


def test_beta_excitations_6orb_3e():
    """Same as alpha test but task_type=1 (β-only). β indices = 2k+1."""
    L = 6
    n_b = 3
    adets = _enumerate_dets(L, n_b)
    bdets = adets
    h = generate_excitation(adets, bdets, L, task_type=1)

    n_singles_h = sum(len(x) for x in h.singles_from_beta)
    n_doubles_h = sum(len(x) for x in h.doubles_from_beta)
    n_singles_bf = 0
    n_doubles_bf = 0
    for ib in range(len(bdets)):
        for jb in range(len(bdets)):
            d = difference(bdets[ib], bdets[jb], L)
            if d == 2:
                n_singles_bf += 1
            elif d == 4:
                n_doubles_bf += 1
    assert n_singles_h == n_singles_bf
    assert n_doubles_h == n_doubles_bf

    # CrAn for β must be ODD (2k+1)
    for ib in range(len(bdets)):
        for k, jb in enumerate(h.singles_from_beta[ib]):
            cr_so, an_so = h.singles_beta_cran[ib][k]
            assert cr_so % 2 == 1, f"β single cr_so={cr_so} should be odd"
            assert an_so % 2 == 1, f"β single an_so={an_so} should be odd"
        for k, jb in enumerate(h.doubles_from_beta[ib]):
            cr0, cr1, an0, an1 = h.doubles_beta_cran[ib][k]
            assert all(x % 2 == 1 for x in (cr0, cr1, an0, an1))

    # α tables MUST be empty for task_type=1
    assert h.singles_from_alpha == []
    assert h.doubles_from_alpha == []
    print("test_beta_excitations_6orb_3e passed")


def test_alphabeta_task_zero():
    """task_type=0 builds SinglesFromAlpha AND SinglesFromBeta (no doubles)."""
    L = 4
    n_a = n_b = 2
    adets = _enumerate_dets(L, n_a)
    bdets = _enumerate_dets(L, n_b)
    h = generate_excitation(adets, bdets, L, task_type=0)

    # Singles populated, doubles empty
    assert len(h.singles_from_alpha) == len(adets)
    assert len(h.singles_from_beta) == len(bdets)
    assert h.doubles_from_alpha == []
    assert h.doubles_from_beta == []

    # Verify counts vs brute force
    n_a_singles = sum(
        1
        for ia in range(len(adets))
        for ja in range(len(adets))
        if difference(adets[ia], adets[ja], L) == 2
    )
    n_b_singles = sum(
        1
        for ib in range(len(bdets))
        for jb in range(len(bdets))
        if difference(bdets[ib], bdets[jb], L) == 2
    )
    assert sum(len(x) for x in h.singles_from_alpha) == n_a_singles
    assert sum(len(x) for x in h.singles_from_beta) == n_b_singles

    print("test_alphabeta_task_zero passed")


# ============================================================================
# 3. task type filtering: empty tables where they shouldn't be populated
# ============================================================================


def test_task_type_filtering():
    """generate_excitation() filters correctly by task_type (0, 1, 2)."""
    L = 4
    n = 2
    adets = bdets = _enumerate_dets(L, n)
    h2 = generate_excitation(adets, bdets, L, task_type=2)
    h1 = generate_excitation(adets, bdets, L, task_type=1)
    h0 = generate_excitation(adets, bdets, L, task_type=0)

    # task 2: only α singles + α doubles
    assert h2.singles_from_alpha and h2.doubles_from_alpha
    assert not h2.singles_from_beta and not h2.doubles_from_beta

    # task 1: only β singles + β doubles
    assert h1.singles_from_beta and h1.doubles_from_beta
    assert not h1.singles_from_alpha and not h1.doubles_from_alpha

    # task 0: α singles AND β singles; no doubles either side
    assert h0.singles_from_alpha and h0.singles_from_beta
    assert not h0.doubles_from_alpha and not h0.doubles_from_beta

    print("test_task_type_filtering passed")


# ============================================================================
# Stress: counts match an analytical formula for a clean FCI basis
# ============================================================================


def test_counts_analytical_4orb_2e():
    """For C(L, n) basis: each det has exactly n*(L-n) single-excited
    neighbors and C(n,2)*C(L-n,2) double-excited neighbors.
    Verify total counts match.
    """
    L = 4
    n = 2
    n_dets = 6  # C(4, 2)
    singles_per_det = n * (L - n)  # 2*2 = 4
    doubles_per_det = (n * (n - 1) // 2) * ((L - n) * (L - n - 1) // 2)  # 1 * 1 = 1
    expected_singles = n_dets * singles_per_det
    expected_doubles = n_dets * doubles_per_det

    adets = _enumerate_dets(L, n)
    bdets = adets
    h = generate_excitation(adets, bdets, L, task_type=2)
    got_singles = sum(len(x) for x in h.singles_from_alpha)
    got_doubles = sum(len(x) for x in h.doubles_from_alpha)
    print(
        f"  4-orb, 2-e: singles {got_singles} (expected {expected_singles}), "
        f"doubles {got_doubles} (expected {expected_doubles})"
    )
    assert got_singles == expected_singles
    assert got_doubles == expected_doubles
    print("test_counts_analytical_4orb_2e passed")


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    test_task_type_filtering()
    test_counts_analytical_4orb_2e()
    test_alpha_excitations_6orb_3e()
    test_beta_excitations_6orb_3e()
    test_alphabeta_task_zero()
    print("\nALL A2 TESTS PASSED")
