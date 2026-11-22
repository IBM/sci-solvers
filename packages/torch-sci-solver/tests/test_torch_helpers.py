"""Tests for torch_sci_solver.torch_impl.helpers.

Round-trip test: flat CSR helpers ↔ scalar list helpers contain
exactly the same information.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref import generate_excitation
from torch_sci_solver.torch_impl import (
    generate_all_torch_helpers,
    torch_helper_from_scalar,
)
from torch_sci_solver.torch_impl.helpers import (
    _narrow_table_indices,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.probe import (
    _embed_eigvec_into_extended_basis,
    _embed_eigvec_into_extended_basis_rows,
)


def _enumerate_dets(L, n_elec):
    out = []
    for combo in itertools.combinations(range(L), n_elec):
        m = 0
        for b in combo:
            m |= 1 << b
        out.append(m)
    out.sort()
    return out


def _compare_singles(scalar_to, scalar_cran, offsets, flat_to, flat_cr, flat_an, label):
    """Verify scalar list-of-lists == flat CSR."""
    M = len(scalar_to)
    assert offsets.shape == (M + 1,), f"{label}: offsets shape {offsets.shape}"
    pos = 0
    for ia in range(M):
        n = len(scalar_to[ia])
        assert offsets[ia].item() == pos, f"{label}: offset[{ia}]"
        # Check ja indices
        flat_slice = flat_to[pos : pos + n].tolist()
        assert flat_slice == list(scalar_to[ia]), f"{label}: to[{ia}]"
        # Check (cr, an) tuples
        for k in range(n):
            cr, an = scalar_cran[ia][k]
            assert flat_cr[pos + k].item() == cr, f"{label}: cr[{ia},{k}]"
            assert flat_an[pos + k].item() == an, f"{label}: an[{ia},{k}]"
        pos += n
    assert offsets[M].item() == pos


def _compare_doubles(
    scalar_to, scalar_cran, offsets, flat_to, flat_cr0, flat_cr1, flat_an0, flat_an1, label
):
    M = len(scalar_to)
    pos = 0
    for ia in range(M):
        n = len(scalar_to[ia])
        assert offsets[ia].item() == pos
        flat_slice = flat_to[pos : pos + n].tolist()
        assert flat_slice == list(scalar_to[ia])
        for k in range(n):
            cr0, cr1, an0, an1 = scalar_cran[ia][k]
            assert flat_cr0[pos + k].item() == cr0, f"{label}: cr0[{ia},{k}]"
            assert flat_cr1[pos + k].item() == cr1
            assert flat_an0[pos + k].item() == an0
            assert flat_an1[pos + k].item() == an1
        pos += n


def test_helper_roundtrip_6orb_3e():
    """Torch excitation-helper tables round-trip against the scalar tables (6 orb, 3e)."""
    L = 6
    adets = _enumerate_dets(L, 3)
    bdets = adets

    # Task type 2: α singles + α doubles
    scalar2 = generate_excitation(adets, bdets, L, task_type=2)
    torch2 = torch_helper_from_scalar(scalar2)
    _compare_singles(
        scalar2.singles_from_alpha,
        scalar2.singles_alpha_cran,
        torch2.a_singles_offsets,
        torch2.a_singles_to,
        torch2.a_singles_cr,
        torch2.a_singles_an,
        "task2 α singles",
    )
    _compare_doubles(
        scalar2.doubles_from_alpha,
        scalar2.doubles_alpha_cran,
        torch2.a_doubles_offsets,
        torch2.a_doubles_to,
        torch2.a_doubles_cr0,
        torch2.a_doubles_cr1,
        torch2.a_doubles_an0,
        torch2.a_doubles_an1,
        "task2 α doubles",
    )
    print(
        f"  task2: α-singles={torch2.a_singles_to.shape[0]}, "
        f"α-doubles={torch2.a_doubles_to.shape[0]}"
    )
    # β tables empty
    assert torch2.b_singles_to is None
    assert torch2.b_doubles_to is None

    # Task type 1: β singles + β doubles
    scalar1 = generate_excitation(adets, bdets, L, task_type=1)
    torch1 = torch_helper_from_scalar(scalar1)
    _compare_singles(
        scalar1.singles_from_beta,
        scalar1.singles_beta_cran,
        torch1.b_singles_offsets,
        torch1.b_singles_to,
        torch1.b_singles_cr,
        torch1.b_singles_an,
        "task1 β singles",
    )
    _compare_doubles(
        scalar1.doubles_from_beta,
        scalar1.doubles_beta_cran,
        torch1.b_doubles_offsets,
        torch1.b_doubles_to,
        torch1.b_doubles_cr0,
        torch1.b_doubles_cr1,
        torch1.b_doubles_an0,
        torch1.b_doubles_an1,
        "task1 β doubles",
    )
    print(
        f"  task1: β-singles={torch1.b_singles_to.shape[0]}, "
        f"β-doubles={torch1.b_doubles_to.shape[0]}"
    )
    assert torch1.a_singles_to is None
    assert torch1.a_doubles_to is None

    # Task type 0: α singles + β singles (no doubles)
    scalar0 = generate_excitation(adets, bdets, L, task_type=0)
    torch0 = torch_helper_from_scalar(scalar0)
    _compare_singles(
        scalar0.singles_from_alpha,
        scalar0.singles_alpha_cran,
        torch0.a_singles_offsets,
        torch0.a_singles_to,
        torch0.a_singles_cr,
        torch0.a_singles_an,
        "task0 α singles",
    )
    _compare_singles(
        scalar0.singles_from_beta,
        scalar0.singles_beta_cran,
        torch0.b_singles_offsets,
        torch0.b_singles_to,
        torch0.b_singles_cr,
        torch0.b_singles_an,
        "task0 β singles",
    )
    print(
        f"  task0: α-singles={torch0.a_singles_to.shape[0]}, "
        f"β-singles={torch0.b_singles_to.shape[0]}"
    )
    assert torch0.a_doubles_to is None
    assert torch0.b_doubles_to is None

    print("test_helper_roundtrip_6orb_3e passed")


def test_generate_all_torch_helpers():
    """generate_all_torch_helpers returns consistent task-type-0/1/2 helper tables."""
    L = 6
    adets = _enumerate_dets(L, 3)
    bdets = adets
    h2, h1, h0 = generate_all_torch_helpers(adets, bdets, L)
    assert h2.task_type == 2
    assert h1.task_type == 1
    assert h0.task_type == 0
    # Spot-check the singles count matches a brute-force count
    from torch_sci_solver.sbd_ref import difference

    n_a_singles_bf = sum(
        1
        for ia in range(len(adets))
        for ja in range(len(adets))
        if difference(adets[ia], adets[ja], L) == 2
    )
    assert h2.a_singles_to.shape[0] == n_a_singles_bf, (
        f"task2 α singles: got {h2.a_singles_to.shape[0]}, expected {n_a_singles_bf}"
    )
    print("test_generate_all_torch_helpers passed")


def test_vec_helpers_match_scalar_L10_5e():
    """generate_all_torch_helpers matches the scalar sbd_ref reference.

    Full C(10, 5) = 252-string basis on both spins (dim = 63,504). Large
    enough to exercise nontrivial α-doubles/β-doubles tables (~10k entries
    per spin) — a size where the scalar Python-loop reference is still fast
    enough to serve as the ground truth in CI. Production runs the
    numpy-vectorized path; the scalar reference in
    :mod:`torch_sci_solver.sbd_ref.helpers` exists as documentation of the
    algorithm and as the equivalence check this test performs.
    """
    import torch as _torch

    L = 10
    adets = _enumerate_dets(L, 5)
    bdets = _enumerate_dets(L, 5)

    # Production output — the (only) vectorized numpy path.
    pub2, pub1, pub0 = generate_all_torch_helpers(adets, bdets, L, "cpu")

    # Scalar reference: build the three TaskHelpers by hand from sbd_ref and
    # convert them to TorchHelpers for a tensor-level equality check.
    scalar2 = torch_helper_from_scalar(generate_excitation(adets, bdets, L, 2))
    scalar1 = torch_helper_from_scalar(generate_excitation(adets, bdets, L, 1))
    scalar0 = torch_helper_from_scalar(generate_excitation(adets, bdets, L, 0))

    def _assert_all_fields_equal(a, b, label):
        for f in (
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
            ra = getattr(a, f)
            rb = getattr(b, f)
            if ra is None and rb is None:
                continue
            assert ra is not None and rb is not None, f"{label}: {f} None mismatch"
            assert _torch.equal(ra, rb), f"{label}: {f} differs"

    _assert_all_fields_equal(pub2, scalar2, "task2 (α)")
    _assert_all_fields_equal(pub1, scalar1, "task1 (β)")
    _assert_all_fields_equal(pub0, scalar0, "task0 (αβ)")

    # Regression guard: non-trivial doubles were actually exercised (a
    # silently-empty doubles table would still pass the field-by-field
    # equality since both sides would be empty).
    assert pub2.a_doubles_to.shape[0] > 100
    assert pub1.b_doubles_to.shape[0] > 100

    print(
        f"test_vec_helpers_match_scalar_L10_5e passed "
        f"(α-singles={pub2.a_singles_to.shape[0]}, "
        f"α-doubles={pub2.a_doubles_to.shape[0]})"
    )


# ---------------------------------------------------------------------------
# Row-chunked builder, owner-computes local block, row-restricted embed
# ---------------------------------------------------------------------------

_HELPER_FIELDS = (
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
)

_BETA_FIELDS = (
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
)


def _assert_helpers_bitidentical(got, expected, label):
    """Every tensor of every helper matches exactly (dtype + values)."""
    for h_got, h_exp in zip(got, expected, strict=True):
        assert h_got.task_type == h_exp.task_type, f"{label}: task_type"
        assert h_got.n_alpha == h_exp.n_alpha, f"{label}: n_alpha"
        assert h_got.n_beta == h_exp.n_beta, f"{label}: n_beta"
        for field in _HELPER_FIELDS:
            r_got = getattr(h_got, field)
            r_exp = getattr(h_exp, field)
            if r_exp is None:
                assert r_got is None, f"{label}: {field} should be None"
                continue
            assert r_got is not None, f"{label}: {field} is None"
            assert r_got.dtype == r_exp.dtype, f"{label}: {field} dtype"
            assert torch.equal(r_got, r_exp), f"{label}: {field} differs"


def _assert_csr_slice_singles(local, full, r0, r1, label):
    """Local singles CSR equals full CSR rows [r0, r1) with shifted offsets."""
    off_l, to_l, cr_l, an_l = local
    off_f, to_f, cr_f, an_f = full
    base = int(off_f[r0])
    stop = int(off_f[r1])
    assert off_l.shape == (r1 - r0 + 1,), f"{label}: offsets shape"
    assert torch.equal(off_l, off_f[r0 : r1 + 1] - base), f"{label}: offsets"
    assert off_l[-1].item() == stop - base, f"{label}: trailing offset"
    assert torch.equal(to_l, to_f[base:stop]), f"{label}: to"
    assert torch.equal(cr_l, cr_f[base:stop]), f"{label}: cr"
    assert torch.equal(an_l, an_f[base:stop]), f"{label}: an"


def _assert_csr_slice_doubles(local, full, r0, r1, label):
    """Local doubles CSR equals full CSR rows [r0, r1) with shifted offsets."""
    off_l, to_l, cr0_l, cr1_l, an0_l, an1_l = local
    off_f, to_f, cr0_f, cr1_f, an0_f, an1_f = full
    base = int(off_f[r0])
    stop = int(off_f[r1])
    assert off_l.shape == (r1 - r0 + 1,), f"{label}: offsets shape"
    assert torch.equal(off_l, off_f[r0 : r1 + 1] - base), f"{label}: offsets"
    assert off_l[-1].item() == stop - base, f"{label}: trailing offset"
    assert torch.equal(to_l, to_f[base:stop]), f"{label}: to"
    assert torch.equal(cr0_l, cr0_f[base:stop]), f"{label}: cr0"
    assert torch.equal(cr1_l, cr1_f[base:stop]), f"{label}: cr1"
    assert torch.equal(an0_l, an0_f[base:stop]), f"{label}: an0"
    assert torch.equal(an1_l, an1_f[base:stop]), f"{label}: an1"


def test_generate_all_torch_helpers_chunked_bitidentical():
    """Row chunking is bit-identical to the single-chunk (full-matrix) output."""
    for L, n_elec in ((6, 3), (8, 4), (10, 5)):
        adets = _enumerate_dets(L, n_elec)
        bdets = _enumerate_dets(L, n_elec)
        # chunk_rows >= M is the old full-matrix path: exactly one chunk.
        reference = generate_all_torch_helpers(adets, bdets, L, "cpu", chunk_rows=len(adets) + 1)
        for chunk_rows in (1, 2, 7, 1024):
            got = generate_all_torch_helpers(adets, bdets, L, "cpu", chunk_rows=chunk_rows)
            _assert_helpers_bitidentical(got, reference, f"L={L} chunk_rows={chunk_rows}")
    print("test_generate_all_torch_helpers_chunked_bitidentical passed")


def test_generate_local_torch_helpers_slices_full_tables():
    """Local α tables are the owned-row CSR slice of the full tables; β is full."""
    L = 8
    adets = _enumerate_dets(L, 3)  # C(8, 3) = 56 strings
    bdets = _enumerate_dets(L, 3)
    M_a = len(adets)
    M_b = len(bdets)
    full2, full1, full0 = generate_all_torch_helpers(adets, bdets, L)

    for chunk_rows in (1, 1024):
        for r0, r1 in ((0, 1), (0, 17), (17, 43), (43, M_a), (0, M_a)):
            loc2, loc1, loc0 = generate_local_torch_helpers(
                adets, bdets, L, r0, r1, chunk_rows=chunk_rows
            )
            n_local = r1 - r0
            label = f"[{r0}, {r1}) chunk_rows={chunk_rows}"

            assert loc2.n_alpha == n_local, f"{label}: task2 n_alpha"
            assert loc0.n_alpha == n_local, f"{label}: task0 n_alpha"
            assert loc1.n_alpha == M_a, f"{label}: task1 n_alpha"
            for h in (loc2, loc1, loc0):
                assert h.n_beta == M_b, f"{label}: n_beta"

            # The full CSR is sorted by bra, so the owned rows are the contiguous
            # slice [offset[r0], offset[r1]).
            _assert_csr_slice_singles(
                (loc2.a_singles_offsets, loc2.a_singles_to, loc2.a_singles_cr, loc2.a_singles_an),
                (
                    full2.a_singles_offsets,
                    full2.a_singles_to,
                    full2.a_singles_cr,
                    full2.a_singles_an,
                ),
                r0,
                r1,
                f"task2 α singles {label}",
            )
            _assert_csr_slice_doubles(
                (
                    loc2.a_doubles_offsets,
                    loc2.a_doubles_to,
                    loc2.a_doubles_cr0,
                    loc2.a_doubles_cr1,
                    loc2.a_doubles_an0,
                    loc2.a_doubles_an1,
                ),
                (
                    full2.a_doubles_offsets,
                    full2.a_doubles_to,
                    full2.a_doubles_cr0,
                    full2.a_doubles_cr1,
                    full2.a_doubles_an0,
                    full2.a_doubles_an1,
                ),
                r0,
                r1,
                f"task2 α doubles {label}",
            )
            _assert_csr_slice_singles(
                (loc0.a_singles_offsets, loc0.a_singles_to, loc0.a_singles_cr, loc0.a_singles_an),
                (
                    full0.a_singles_offsets,
                    full0.a_singles_to,
                    full0.a_singles_cr,
                    full0.a_singles_an,
                ),
                r0,
                r1,
                f"task0 α singles {label}",
            )

            # β tables are the ordinary full ones on both β-bearing tasks.
            for field in _BETA_FIELDS:
                assert torch.equal(getattr(loc1, field), getattr(full1, field)), (
                    f"task1 {field} {label}"
                )
                full0_field = getattr(full0, field)
                loc0_field = getattr(loc0, field)
                if full0_field is None:  # task 0 tabulates no β doubles
                    assert loc0_field is None, f"task0 {field} {label} should be None"
                else:
                    assert torch.equal(loc0_field, full0_field), f"task0 {field} {label}"

    # Regression guard: the tables sliced above were non-empty.
    assert full2.a_singles_to.numel() > 0
    assert full2.a_doubles_to.numel() > 0
    print("test_generate_local_torch_helpers_slices_full_tables passed")


def test_embed_eigvec_rows_matches_full_embedding():
    """Stacked row blocks equal the full embedding times its global norm."""
    L = 6
    all_strings = _enumerate_dets(L, 3)  # C(6, 3) = 20 strings
    order = [7, 3, 19, 0, 11, 5, 14, 2, 9, 18, 1, 16, 8, 4, 17, 12, 6, 15, 10, 13]
    extended_adets = [all_strings[i] for i in order]
    extended_bdets = [all_strings[i] for i in reversed(order)]
    cached_adets = [all_strings[i] for i in (7, 19, 0, 5, 14, 9, 1, 8, 17, 6, 10)]
    cached_bdets = [all_strings[i] for i in (3, 11, 2, 18, 16, 4, 12, 15, 13)]

    M_a_old, M_b_old = len(cached_adets), len(cached_bdets)
    M_a, M_b = len(extended_adets), len(extended_bdets)
    generator = torch.Generator().manual_seed(1234)
    eigvec_old = torch.randn(M_a_old * M_b_old, generator=generator, dtype=torch.float64)

    full = _embed_eigvec_into_extended_basis(
        eigvec_old,
        cached_adets,
        cached_bdets,
        extended_adets,
        extended_bdets,
        torch.device("cpu"),
        torch.float64,
    )
    full_2d = full.view(M_a, M_b)

    ranges = ((0, 9), (9, M_a))
    blocks = [
        _embed_eigvec_into_extended_basis_rows(
            eigvec_old,
            cached_adets,
            cached_bdets,
            extended_adets,
            extended_bdets,
            r0,
            r1,
            torch.device("cpu"),
            torch.float64,
        )
        for r0, r1 in ranges
    ]
    for (r0, r1), block in zip(ranges, blocks, strict=True):
        assert block.shape == (r1 - r0, M_b)
        assert block.dtype == torch.float64

    stacked = torch.cat(blocks, dim=0)
    norm = torch.linalg.norm(stacked)
    assert norm.item() > 0.0
    # The full function applies the global normalization; the row variant does not.
    assert torch.allclose(full_2d * norm, stacked, atol=1e-12)
    assert torch.allclose(stacked / norm, full_2d, atol=1e-12)

    # Extended α rows absent from the cached basis are zero.
    cached_a_set = set(cached_adets)
    for i_a, s in enumerate(extended_adets):
        if s not in cached_a_set:
            assert torch.count_nonzero(stacked[i_a]).item() == 0
    print("test_embed_eigvec_rows_matches_full_embedding passed")


def test_embed_eigvec_rows_rejects_missing_cached_beta():
    """A cached β-string missing from the extended basis raises (superset contract)."""
    L = 6
    all_strings = _enumerate_dets(L, 3)
    cached_adets = all_strings[:5]
    cached_bdets = all_strings[:5]
    extended_adets = list(all_strings)
    extended_bdets = all_strings[1:]  # drops cached_bdets[0]
    eigvec_old = torch.ones(len(cached_adets) * len(cached_bdets), dtype=torch.float64)
    with pytest.raises(ValueError, match="extended basis"):
        _embed_eigvec_into_extended_basis_rows(
            eigvec_old,
            cached_adets,
            cached_bdets,
            extended_adets,
            extended_bdets,
            0,
            len(extended_adets),
            torch.device("cpu"),
            torch.float64,
        )
    print("test_embed_eigvec_rows_rejects_missing_cached_beta passed")


def test_generate_local_torch_helpers_beta_column_range():
    """A β column range produces the same CSR slice as the full β tables."""
    L = 8
    adets = _enumerate_dets(L, 3)
    bdets = _enumerate_dets(L, 3)
    M_b = len(bdets)
    full2, full1, full0 = generate_all_torch_helpers(adets, bdets, L)

    for r0, r1 in ((0, M_b), (0, 13), (13, 40), (40, M_b)):
        for c0, c1 in ((0, M_b), (0, 7), (7, 31), (31, M_b), (5, M_b)):
            loc2, loc1, loc0 = generate_local_torch_helpers(
                adets, bdets, L, r0, r1, col_start=c0, col_stop=c1
            )
            label = f"rows [{r0}, {r1}) cols [{c0}, {c1})"
            assert loc2.alpha_is_local and loc0.alpha_is_local, label
            for h in (loc2, loc1, loc0):
                assert h.beta_is_local, f"{label}: beta_is_local"
                assert h.n_beta == c1 - c0, f"{label}: n_beta"
            _assert_csr_slice_singles(
                (
                    loc1.b_singles_offsets,
                    loc1.b_singles_to,
                    loc1.b_singles_cr,
                    loc1.b_singles_an,
                ),
                (
                    full1.b_singles_offsets,
                    full1.b_singles_to,
                    full1.b_singles_cr,
                    full1.b_singles_an,
                ),
                c0,
                c1,
                f"task1 β singles {label}",
            )
            _assert_csr_slice_doubles(
                (
                    loc1.b_doubles_offsets,
                    loc1.b_doubles_to,
                    loc1.b_doubles_cr0,
                    loc1.b_doubles_cr1,
                    loc1.b_doubles_an0,
                    loc1.b_doubles_an1,
                ),
                (
                    full1.b_doubles_offsets,
                    full1.b_doubles_to,
                    full1.b_doubles_cr0,
                    full1.b_doubles_cr1,
                    full1.b_doubles_an0,
                    full1.b_doubles_an1,
                ),
                c0,
                c1,
                f"task1 β doubles {label}",
            )
            # task 0 shares the same β-singles tensors (no duplicate storage)
            assert loc0.b_singles_to.data_ptr() == loc1.b_singles_to.data_ptr(), label
            assert torch.equal(full0.b_singles_to, full1.b_singles_to), label

    # No column range keeps the full β tables and says so in the flags.
    loc2, loc1, loc0 = generate_local_torch_helpers(adets, bdets, L, 0, 4)
    assert not loc1.beta_is_local and not loc0.beta_is_local
    assert loc1.n_beta == M_b
    assert loc2.alpha_is_local


def test_table_dtype_int32_matches_int64():
    """int32 payloads carry identical values; offsets stay int64."""
    L = 8
    adets = _enumerate_dets(L, 3)
    reference = generate_all_torch_helpers(adets, adets, L)
    narrowed = generate_all_torch_helpers(adets, adets, L, table_dtype=torch.int32)
    for h_ref, h_got in zip(reference, narrowed, strict=True):
        for field in _HELPER_FIELDS:
            expected = getattr(h_ref, field)
            got = getattr(h_got, field)
            if expected is None:
                assert got is None, field
                continue
            if field.endswith("_offsets"):
                assert got.dtype == torch.int64, f"{field}: offsets stay int64"
            else:
                assert got.dtype == torch.int32, f"{field}: payload is int32"
            assert torch.equal(expected.to(got.dtype), got), field

    # The bounds check trips before an unsafe cast.
    with pytest.raises(OverflowError):
        _narrow_table_indices(np.array([2**31], dtype=np.int64), torch.int32, "to")
    with pytest.raises(ValueError):
        _narrow_table_indices(np.array([0], dtype=np.int64), torch.int16, "to")


if __name__ == "__main__":
    test_helper_roundtrip_6orb_3e()
    test_generate_all_torch_helpers()
    test_vec_helpers_match_scalar_L10_5e()
    print("\nALL TESTS PASSED")
