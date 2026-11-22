"""Multi-limb determinant strings in the TPB helper builders.

The builders carry strings wider than 64 bits as ``(M, n_limbs)`` int64 limb
matrices (limb ``k`` holds bits ``[64k, 64k + 64)``). These tests check:

1. The single-word path is unchanged: Python ints, 1-D bit patterns and
   ``(M, 1)`` limb matrices give bit-identical tables.
2. Two- and three-limb tables match the scalar reference
   (:func:`torch_sci_solver.sbd_ref.generate_all_helpers`) row for row, for
   strings with bits above the 64-bit boundary.
3. ``det_from_alpha_beta_batch`` accepts multi-limb α/β halves and reproduces
   the Python-int interleaving, while its single-word behaviour is unchanged.
4. Tile-local α rows / β columns built from limb strings equal the full
   builder's slices.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref import generate_all_helpers as scalar_generate_all_helpers
from torch_sci_solver.sbd_ref.helpers import generate_excitation
from torch_sci_solver.torch_impl.determinants import det_from_alpha_beta_batch
from torch_sci_solver.torch_impl.helpers import (
    _as_string_limbs,
    _n_limbs,
    generate_all_torch_helpers,
    generate_local_torch_helpers,
    torch_helper_from_scalar,
)


def _split_limbs(value: int, n_limbs: int) -> list[int]:
    """Split a Python-int string into ``n_limbs`` 64-bit limbs."""
    return [(value >> (64 * k)) & ((1 << 64) - 1) for k in range(n_limbs)]


def _neighbor_strings(base: int, n_orb: int, count: int, seed: int) -> list[int]:
    """Single/double-excitation neighbours of ``base``, sorted (particle number fixed)."""
    rng = np.random.default_rng(seed)
    occupied = [k for k in range(n_orb) if (base >> k) & 1]
    empty = [k for k in range(n_orb) if not (base >> k) & 1]
    # Only fixed-particle-number flips are reachable: 1 + o*v + C(o,2)C(v,2).
    reachable = (
        1
        + len(occupied) * len(empty)
        + (len(occupied) * (len(occupied) - 1) // 2) * (len(empty) * (len(empty) - 1) // 2)
    )
    count = min(count, reachable)
    strings = {base}
    while len(strings) < count:
        if len(occupied) >= 2 and len(empty) >= 2 and rng.random() < 0.5:
            a, b = rng.choice(occupied, size=2, replace=False)
            c, d = rng.choice(empty, size=2, replace=False)
            value = base ^ (1 << int(a)) ^ (1 << int(b)) ^ (1 << int(c)) ^ (1 << int(d))
        else:
            value = base ^ (1 << int(rng.choice(occupied))) ^ (1 << int(rng.choice(empty)))
        strings.add(value)
    return sorted(strings)


def _selected_strings(n_orb: int) -> tuple[list[int], list[int]]:
    """Deterministic α/β string sets whose top orbitals cross the 64-bit edge."""
    base_a = sum(1 << o for o in {0, 3, n_orb - 1, max(0, n_orb - 5)})
    base_b = sum(1 << o for o in {1, 4, n_orb - 2, max(0, n_orb - 6)})
    adets = _neighbor_strings(base_a, n_orb, 22, seed=3)
    bdets = _neighbor_strings(base_b, n_orb, 17, seed=5)
    return adets, bdets


def _rows(offsets: torch.Tensor, payloads: list[torch.Tensor]) -> list[list[tuple[int, ...]]]:
    """Flatten a CSR table into one list of index tuples per bra row."""
    offsets = offsets.tolist()
    columns = [p.tolist() for p in payloads]
    rows = []
    for row in range(len(offsets) - 1):
        start, stop = offsets[row], offsets[row + 1]
        rows.append([tuple(int(c[i]) for c in columns) for i in range(start, stop)])
    return rows


def _assert_scalar_parity(adets: list[int], bdets: list[int], n_orb: int) -> None:
    """Every torch table equals the scalar reference, row for row."""
    helpers = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    scalar = scalar_generate_all_helpers(adets, bdets, n_orb)
    task2, task1, task0 = helpers
    ref2, ref1, ref0 = scalar

    # α singles and doubles (task 2).
    assert _rows(
        task2.a_singles_offsets, [task2.a_singles_to, task2.a_singles_cr, task2.a_singles_an]
    ) == [
        [(ja, cr, an) for ja, (cr, an) in zip(row_ja, row_cran)]
        for row_ja, row_cran in zip(ref2.singles_from_alpha, ref2.singles_alpha_cran)
    ]
    assert _rows(
        task2.a_doubles_offsets,
        [
            task2.a_doubles_to,
            task2.a_doubles_cr0,
            task2.a_doubles_cr1,
            task2.a_doubles_an0,
            task2.a_doubles_an1,
        ],
    ) == [
        [(ja, *cran) for ja, cran in zip(row_ja, row_cran)]
        for row_ja, row_cran in zip(ref2.doubles_from_alpha, ref2.doubles_alpha_cran)
    ]

    # β singles and doubles (task 1).
    assert _rows(
        task1.b_singles_offsets, [task1.b_singles_to, task1.b_singles_cr, task1.b_singles_an]
    ) == [
        [(jb, cr, an) for jb, (cr, an) in zip(row_jb, row_cran)]
        for row_jb, row_cran in zip(ref1.singles_from_beta, ref1.singles_beta_cran)
    ]
    assert _rows(
        task1.b_doubles_offsets,
        [
            task1.b_doubles_to,
            task1.b_doubles_cr0,
            task1.b_doubles_cr1,
            task1.b_doubles_an0,
            task1.b_doubles_an1,
        ],
    ) == [
        [(jb, *cran) for jb, cran in zip(row_jb, row_cran)]
        for row_jb, row_cran in zip(ref1.doubles_from_beta, ref1.doubles_beta_cran)
    ]

    # The αβ task shares the same singles arrays.
    assert torch.equal(task0.a_singles_to, task2.a_singles_to)
    assert torch.equal(task0.b_singles_to, task1.b_singles_to)
    assert torch.equal(task0.a_singles_offsets, task2.a_singles_offsets)
    _ = ref0


def _assert_tables_equal(left: list, right: list) -> None:
    """Assert two helper triples carry identical tables and locality flags."""
    assert len(left) == len(right) == 3
    for h_left, h_right in zip(left, right):
        assert h_left.task_type == h_right.task_type
        assert h_left.n_alpha == h_right.n_alpha
        assert h_left.n_beta == h_right.n_beta
        assert h_left.alpha_is_local == h_right.alpha_is_local
        assert h_left.beta_is_local == h_right.beta_is_local
        for field in (
            "a_singles_offsets",
            "a_singles_to",
            "a_singles_cr",
            "a_singles_an",
            "a_doubles_offsets",
            "a_doubles_to",
            "a_doubles_cr0",
            "a_doubles_cr1",
            "a_doubles_an0",
            "a_doubles_an1",
            "b_singles_offsets",
            "b_singles_to",
            "b_singles_cr",
            "b_singles_an",
            "b_doubles_offsets",
            "b_doubles_to",
            "b_doubles_cr0",
            "b_doubles_cr1",
            "b_doubles_an0",
            "b_doubles_an1",
        ):
            value_left = getattr(h_left, field)
            value_right = getattr(h_right, field)
            assert (value_left is None) == (value_right is None), field
            if value_left is not None:
                assert torch.equal(value_left, value_right), field


def test_single_word_inputs_are_unchanged() -> None:
    """Python ints, 1-D bit patterns and (M, 1) matrices build the same tables."""
    adets, bdets = _selected_strings(4)
    as_list = list(generate_all_torch_helpers(adets, bdets, 4, device="cpu"))
    as_array = list(
        generate_all_torch_helpers(
            np.asarray(adets, dtype=np.int64), np.asarray(bdets, dtype=np.int64), 4, device="cpu"
        )
    )
    limb_matrix = np.asarray([_split_limbs(v, 1) for v in adets], dtype=np.int64)
    bdets_matrix = np.asarray([_split_limbs(v, 1) for v in bdets], dtype=np.int64)
    as_limbs = list(generate_all_torch_helpers(limb_matrix, bdets_matrix, 4, device="cpu"))
    _assert_tables_equal(as_list, as_array)
    _assert_tables_equal(as_list, as_limbs)


@pytest.mark.parametrize("n_orb", [70, 130])
def test_multilimb_tables_match_scalar_reference(n_orb: int) -> None:
    """Two- and three-limb strings reproduce the scalar reference exactly."""
    adets, bdets = _selected_strings(n_orb)
    assert max(adets) >> 64 > 0 and max(bdets) >> 64 > 0
    _assert_scalar_parity(adets, bdets, n_orb)


@pytest.mark.parametrize("n_orb", [70, 130])
def test_python_ints_and_limb_matrix_agree(n_orb: int) -> None:
    """Big Python ints and a pre-split limb matrix build identical tables."""
    adets, bdets = _selected_strings(n_orb)
    n_limbs = _n_limbs(n_orb)
    as_ints = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    a_matrix = np.asarray([_split_limbs(v, n_limbs) for v in adets], dtype=np.uint64)
    b_matrix = np.asarray([_split_limbs(v, n_limbs) for v in bdets], dtype=np.uint64)
    as_limbs = list(generate_all_torch_helpers(a_matrix, b_matrix, n_orb, device="cpu"))
    _assert_tables_equal(as_ints, as_limbs)


def test_top_limb_orbital_indices() -> None:
    """An excitation on an orbital above bit 63 maps to the right spin-orbital."""
    n_orb = 70
    low = (1 << 2) | (1 << 5)
    high = (1 << 2) | (1 << 66)  # single excitation 5 → 66
    adets = sorted([low, high])
    helpers = list(generate_all_torch_helpers(adets, [low], n_orb, device="cpu"))
    task2 = helpers[0]
    rows = _rows(
        task2.a_singles_offsets, [task2.a_singles_to, task2.a_singles_cr, task2.a_singles_an]
    )
    # Row 1 (bra = high) has the ket low reachable: create 66, annihilate 5.
    assert rows[1] == [(0, 2 * 66, 2 * 5)]
    # Row 0 (bra = low) reaches high by the reverse excitation.
    assert rows[0] == [(1, 2 * 5, 2 * 66)]


def test_det_from_alpha_beta_multilimb() -> None:
    """Multi-limb α/β halves interleave to the Python-int determinant."""
    n_orb = 70
    n_limbs = _n_limbs(n_orb)
    rng = np.random.default_rng(11)
    alphas = []
    betas = []
    for _ in range(6):
        alpha = 0
        beta = 0
        for _ in range(3):
            alpha |= 1 << int(rng.integers(0, n_orb))
            beta |= 1 << int(rng.integers(0, n_orb))
        alphas.append(alpha)
        betas.append(beta)
    a_matrix = torch.tensor([_split_limbs(v, n_limbs) for v in alphas], dtype=torch.int64)
    b_matrix = torch.tensor([_split_limbs(v, n_limbs) for v in betas], dtype=torch.int64)
    dets = det_from_alpha_beta_batch(a_matrix, b_matrix, n_orb)
    assert dets.shape == (6, (2 * n_orb + 63) // 64)

    for row, (alpha, beta) in enumerate(zip(alphas, betas)):
        expected = 0
        for k in range(n_orb):
            expected |= ((alpha >> k) & 1) << (2 * k)
            expected |= ((beta >> k) & 1) << (2 * k + 1)
        got = 0
        for k, limb in enumerate(dets[row].tolist()):
            got |= (limb & ((1 << 64) - 1)) << (64 * k)
        assert got == expected


def test_det_from_alpha_beta_single_word_unchanged() -> None:
    """A single-word α/β half keeps the original interleaving."""
    alpha = torch.tensor([0b1011, 0b0110], dtype=torch.int64)
    beta = torch.tensor([0b0011, 0b1100], dtype=torch.int64)
    dets = det_from_alpha_beta_batch(alpha, beta, 4)
    assert dets.shape == (2, 1)
    expected = []
    for a, b in zip(alpha.tolist(), beta.tolist()):
        value = 0
        for k in range(4):
            value |= ((a >> k) & 1) << (2 * k)
            value |= ((b >> k) & 1) << (2 * k + 1)
        expected.append(value)
    assert dets[:, 0].tolist() == expected


def _slice_full(full_offsets: torch.Tensor, payloads: list[torch.Tensor], lo: int, hi: int) -> list:
    """The CSR slice of rows ``[lo, hi)`` with rebased offsets."""
    start = int(full_offsets[lo].item())
    stop = int(full_offsets[hi].item())
    return [full_offsets[lo : hi + 1] - full_offsets[lo], *[p[start:stop] for p in payloads]]


@pytest.mark.parametrize("n_orb", [70, 130])
def test_local_tables_multilimb_slice_parity(n_orb: int) -> None:
    """Local α rows / β columns of limb strings equal the full builder's slices."""
    adets, bdets = _selected_strings(n_orb)
    task2, task1, task0 = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    row_start, row_stop = 4, 15
    col_start, col_stop = 3, 12
    local2, local1, local0 = list(
        generate_local_torch_helpers(
            adets,
            bdets,
            n_orb,
            row_start,
            row_stop,
            device="cpu",
            col_start=col_start,
            col_stop=col_stop,
        )
    )

    # α tables are row slices of task 2; β tables are column slices.
    expected = _slice_full(
        task2.a_singles_offsets,
        [task2.a_singles_to, task2.a_singles_cr, task2.a_singles_an],
        row_start,
        row_stop,
    )
    got = [local2.a_singles_offsets, local2.a_singles_to, local2.a_singles_cr, local2.a_singles_an]
    for left, right in zip(got, expected):
        assert torch.equal(left, right)

    expected = _slice_full(
        task2.a_doubles_offsets,
        [
            task2.a_doubles_to,
            task2.a_doubles_cr0,
            task2.a_doubles_cr1,
            task2.a_doubles_an0,
            task2.a_doubles_an1,
        ],
        row_start,
        row_stop,
    )
    got = [
        local2.a_doubles_offsets,
        local2.a_doubles_to,
        local2.a_doubles_cr0,
        local2.a_doubles_cr1,
        local2.a_doubles_an0,
        local2.a_doubles_an1,
    ]
    for left, right in zip(got, expected):
        assert torch.equal(left, right)

    expected = _slice_full(
        task1.b_singles_offsets,
        [task1.b_singles_to, task1.b_singles_cr, task1.b_singles_an],
        col_start,
        col_stop,
    )
    got = [local0.b_singles_offsets, local0.b_singles_to, local0.b_singles_cr, local0.b_singles_an]
    for left, right in zip(got, expected):
        assert torch.equal(left, right)

    expected = _slice_full(
        task1.b_doubles_offsets,
        [
            task1.b_doubles_to,
            task1.b_doubles_cr0,
            task1.b_doubles_cr1,
            task1.b_doubles_an0,
            task1.b_doubles_an1,
        ],
        col_start,
        col_stop,
    )
    got = [
        local1.b_doubles_offsets,
        local1.b_doubles_to,
        local1.b_doubles_cr0,
        local1.b_doubles_cr1,
        local1.b_doubles_an0,
        local1.b_doubles_an1,
    ]
    for left, right in zip(got, expected):
        assert torch.equal(left, right)

    # The same local tables from a pre-split (M, n_limbs) input.
    n_limbs = _n_limbs(n_orb)
    a_matrix = np.asarray([_split_limbs(v, n_limbs) for v in adets], dtype=np.uint64)
    b_matrix = np.asarray([_split_limbs(v, n_limbs) for v in bdets], dtype=np.uint64)
    from_matrix = list(
        generate_local_torch_helpers(
            a_matrix,
            b_matrix,
            n_orb,
            row_start,
            row_stop,
            device="cpu",
            col_start=col_start,
            col_stop=col_stop,
        )
    )
    _assert_tables_equal([local2, local1, local0], from_matrix)


def _bit_positions(value: int) -> list[int]:
    """Ascending positions of the set bits of a non-negative Python int."""
    positions = []
    while value:
        low = value & -value
        positions.append(low.bit_length() - 1)
        value ^= low
    return positions


def _python_int_rows(
    strings: list[int], n_orb: int, spin_shift: int
) -> tuple[list[list[tuple[int, ...]]], list[list[tuple[int, ...]]]]:
    """Independent Python-int singles/doubles tables for one spin sector.

    Scans every ordered ``(bra, ket)`` pair with arbitrary-precision int
    arithmetic: the orbitals in ``bra & ~ket`` are the ``cr`` columns and
    those in ``ket & ~bra`` are the ``an`` columns, both ascending. This
    deliberately does not reuse ``sbd_ref`` so the tensor builders are
    checked against two independent references.
    """
    mask = (1 << n_orb) - 1
    singles: list[list[tuple[int, ...]]] = []
    doubles: list[list[tuple[int, ...]]] = []
    for bra in strings:
        row_s: list[tuple[int, ...]] = []
        row_d: list[tuple[int, ...]] = []
        for ja, ket in enumerate(strings):
            cr = _bit_positions((bra & ~ket) & mask)
            an = _bit_positions((ket & ~bra) & mask)
            if len(cr) == 1 and len(an) == 1:
                row_s.append((ja, 2 * cr[0] + spin_shift, 2 * an[0] + spin_shift))
            elif len(cr) == 2 and len(an) == 2:
                row_d.append((
                    ja,
                    2 * cr[0] + spin_shift,
                    2 * cr[1] + spin_shift,
                    2 * an[0] + spin_shift,
                    2 * an[1] + spin_shift,
                ))
        singles.append(row_s)
        doubles.append(row_d)
    return singles, doubles


@pytest.mark.parametrize("n_orb", [70, 130])
def test_tables_match_python_int_reference(n_orb: int) -> None:
    """α/β CSR tables equal an arbitrary-precision Python-int scan row for row."""
    adets, bdets = _selected_strings(n_orb)
    assert max(adets) >> 64 > 0 and max(bdets) >> 64 > 0
    task2, task1, _task0 = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))

    a_singles, a_doubles = _python_int_rows(adets, n_orb, spin_shift=0)
    b_singles, b_doubles = _python_int_rows(bdets, n_orb, spin_shift=1)

    assert (
        _rows(
            task2.a_singles_offsets,
            [task2.a_singles_to, task2.a_singles_cr, task2.a_singles_an],
        )
        == a_singles
    )
    assert (
        _rows(
            task2.a_doubles_offsets,
            [
                task2.a_doubles_to,
                task2.a_doubles_cr0,
                task2.a_doubles_cr1,
                task2.a_doubles_an0,
                task2.a_doubles_an1,
            ],
        )
        == a_doubles
    )
    assert (
        _rows(
            task1.b_singles_offsets,
            [task1.b_singles_to, task1.b_singles_cr, task1.b_singles_an],
        )
        == b_singles
    )
    assert (
        _rows(
            task1.b_doubles_offsets,
            [
                task1.b_doubles_to,
                task1.b_doubles_cr0,
                task1.b_doubles_cr1,
                task1.b_doubles_an0,
                task1.b_doubles_an1,
            ],
        )
        == b_doubles
    )


@pytest.mark.parametrize("n_orb", [70, 130])
def test_scalar_fallback_matches_vectorized_builder(n_orb: int) -> None:
    """``torch_helper_from_scalar`` carries wide Python-int tables unchanged."""
    adets, bdets = _selected_strings(n_orb)
    fallback = [
        torch_helper_from_scalar(generate_excitation(adets, bdets, n_orb, task_type))
        for task_type in (2, 1, 0)
    ]
    vectorized = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    _assert_tables_equal(fallback, vectorized)


def test_as_string_limbs_rejects_bad_shapes() -> None:
    """A limb matrix with the wrong width and a non-integer input are rejected."""
    with pytest.raises(ValueError):
        _as_string_limbs(np.zeros((3, 1), dtype=np.int64), 70)
    with pytest.raises(TypeError):
        _as_string_limbs(np.zeros((3, 2), dtype=np.float64), 70)
