"""Tests for the vectorized block kernels of the GDB pair matvec.

Covers:

1. Full-vector parity: ``mult_pair_torch`` with ``block_matvec="on"`` equals
   the generic streamed matvec on a full product, a tier-block basis and an
   irregular subset, for real and complex integrals and several vectors.
2. Column (one-hot) parity: every ket column's contribution matches.
3. Owner-computes parity: ``mult_pair_local_torch`` on arbitrary row ranges
   (including a single row and an empty block) equals the full matvec's slice,
   single-process and on two uneven gloo ranks.
4. Label/convention guard: the αβ kernel's per-entry matrix elements for a tiny
   two-by-two group equal ``two_excite_batch`` on the interleaved bra
   determinant, and the ``an``/``cr`` labels really take the ket string to the
   bra string. The same-spin channels are checked in isolation against the
   generic streams, and every channel against a dense Slater-Condon reference.
5. Memory bounds: αβ and same-spin enumeration chunks stay within their
   computed source-count and tile-entry caps on dense-survival asymmetric
   bases, and an empty side skips the other side's enumeration.
6. Degenerate and asymmetric fixtures: zero-electron spins, a fully occupied
   spin, and a many-string × one-string basis all match the generic matvec.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_complex_matvec import _random_complex_hermitian_integrals  # noqa: E402
from test_pair_blocks import _full_product, _tier_blocks  # noqa: E402
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import mult_pair_blocks
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    one_excite_batch,
    two_excite_batch,
)
from torch_sci_solver.torch_impl.distributed_davidson import partition_vector
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import (
    _alphabeta_pair_matrix,
    _apply_alpha_double_chunk_fast,
    _apply_alpha_single_chunk_fast,
    _apply_beta_double_chunk_fast,
    _apply_beta_single_chunk_fast,
)
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    _double_moves,
    _MoveCache,
    _single_moves,
    add_ab_blocks,
    add_ab_blocks_local,
    add_same_spin_blocks,
    add_same_spin_blocks_local,
)
from torch_sci_solver.torch_impl.mult_pairs import (
    _scatter_double_pair,
    _scatter_single_pair,
    make_diag_terms_pair_local,
    make_diag_terms_pair_torch,
    mult_pair_local_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import _chunk_pairs, generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import (
    PairBasis,
    pair_basis_from_arrays,
    pair_superset_embedding,
)

# The eight chemist-index permutations that make a real ERI 8-fold symmetric.
_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _real_integrals(norb: int, seed: int = 0, device: str = "cpu"):
    """Deterministic real integrals with exact 8-fold chemist symmetry.

    Each ERI element copies one random value assigned to its symmetry orbit,
    so the tensor is symmetric by construction (summing permuted copies would
    round differently per element).
    """
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    eri = np.empty_like(raw)
    for idx in itertools.product(range(norb), repeat=4):
        eri[idx] = raw[min(tuple(idx[p] for p in perm) for perm in _PERMUTATIONS)]
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device=device)


def _complex_integrals(norb: int, device: str = "cpu"):
    """Deterministic complex Hermitian integrals on ``norb`` spatial orbitals."""
    hcore, eri = _random_complex_hermitian_integrals(norb, seed=1)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device=device, dtype=torch.complex128)


def _irregular_subset() -> PairBasis:
    """A random 26-pair subset of the full 4-orbital product (singleton blocks)."""
    base = _full_product(4, 2)
    keep = np.sort(np.random.default_rng(3).choice(base.n_pairs, 26, replace=False))
    return pair_basis_from_arrays(base.alpha_of_pair.numpy()[keep], base.beta_of_pair.numpy()[keep])


def _uneven_basis() -> PairBasis:
    """The full 4-orbital product minus one pair: 35 pairs, uneven for 2 ranks."""
    base = _full_product(4, 2)
    return pair_basis_from_arrays(base.alpha_of_pair.numpy()[:-1], base.beta_of_pair.numpy()[:-1])


def _holed_product(norb: int, nocc: int, hole: float, seed: int = 0) -> PairBasis:
    """A product basis with a fraction of its pairs removed (recovered-style).

    Each string then has its own partner set, and the linear fingerprints used
    to bucket strings collide on this size, which used to reject the whole axis.
    """
    base = _full_product(norb, nocc)
    n_keep = int(round((1.0 - hole) * base.n_pairs))
    keep = np.sort(np.random.default_rng(seed).permutation(base.n_pairs)[:n_keep])
    return pair_basis_from_arrays(
        base.alpha_of_pair.numpy()[keep], base.beta_of_pair.numpy()[keep], n_orb=norb
    )


def _make_basis(name: str) -> tuple[PairBasis, int]:
    """Build one named fixture basis and its spatial-orbital count."""
    if name == "full_product":
        return _full_product(4, 2), 4
    if name == "tier":
        return _tier_blocks(6, 3, max_level=2), 6
    if name == "irregular":
        return _irregular_subset(), 4
    raise ValueError(name)


def _random_vector(n: int, seed: int, *, complex_values: bool) -> torch.Tensor:
    """Deterministic random real or complex vector."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if not complex_values:
        return real
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return torch.complex(real, imag)


def _helpers_for(basis: PairBasis, norb: int, mode: str):
    """Pair helpers for ``basis`` with the requested ``block_matvec`` mode."""
    return generate_pair_helpers(basis, norb, device="cpu", block_matvec=mode)


def _assert_close(got: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Assert exact-tolerance agreement (``atol=1e-12``, ``rtol=0``)."""
    if got.numel() == 0 and expected.numel() == 0:
        return
    diff = float((got - expected).abs().max())
    assert torch.allclose(got, expected, atol=1e-12, rtol=0.0), (label, diff)


# ============================================================================
# 1. Full-vector parity with the generic streamed matvec
# ============================================================================


@pytest.mark.parametrize("basis_name", ["full_product", "tier", "irregular"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_block_matvec_matches_generic(basis_name: str, complex_values: bool) -> None:
    """The block-kernel matvec equals the generic streamed matvec exactly."""
    basis, norb = _make_basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    assert on.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)

    for seed in (0, 1, 2):
        v = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
        expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
        got = mult_pair_torch(hii, v, basis, on, norb, integrals)
        _assert_close(got, expected, f"{basis_name} seed={seed}")


def test_block_matvec_refines_fingerprint_collisions() -> None:
    """A recovered-style holed product still block-decomposes exactly.

    Removing ten percent of a 70-string product's pairs gives each string its
    own partner set; the linear fingerprints collide, which used to reject the
    whole axis (and with it every tiled GDB path). The refinement must split
    the colliding buckets exactly and keep the block matvec equal to the
    streamed one.
    """
    basis = _holed_product(8, 4, 0.10)
    norb = 8
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    assert on.blocks is not None
    assert len(on.blocks.by_beta.groups) > 1
    hii = make_diag_terms_pair_torch(basis, norb, integrals)

    for seed in (0, 1, 2):
        v = _random_vector(basis.n_pairs, seed, complex_values=False)
        expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
        got = mult_pair_torch(hii, v, basis, on, norb, integrals)
        _assert_close(got, expected, f"holed seed={seed}")


def test_masked_superset_matvec_matches_the_streamed_basis() -> None:
    """The masked product operator equals the streamed matvec on a recovered list.

    A 10%-holed product has no useful exact block decomposition of its own
    (every string gets its own partner set). The block kernel runs on the
    product superset with the inactive rows zeroed; on the active pairs the
    result must equal the streamed matvec of the original basis exactly.
    """
    basis = _holed_product(8, 4, 0.10)
    norb = 8
    superset, positions, active = pair_superset_embedding(basis)
    assert int(superset.n_pairs) > basis.n_pairs
    assert int(active.sum()) == basis.n_pairs
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    super_helpers = _helpers_for(superset, norb, "on")
    assert super_helpers.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    hii_super = make_diag_terms_pair_torch(superset, norb, integrals)

    for seed in (0, 1):
        v = _random_vector(basis.n_pairs, seed, complex_values=False)
        x = torch.zeros(superset.n_pairs, dtype=v.dtype)
        x[positions] = v
        masked = mult_pair_torch(hii_super, x, superset, super_helpers, norb, integrals) * active
        expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
        _assert_close(masked[positions], expected, f"superset seed={seed}")


def test_superset_embedding_keeps_the_device_and_a_wide_limb_width() -> None:
    """The product is built off-device, and a wide basis keeps its limb layout.

    ``pair_basis_from_full_product`` works in numpy, so a CUDA/MPS basis must be
    staged through the host; a basis built with ``n_orb > 64`` keeps its two-limb
    strings even when every string only uses low orbitals.
    """
    basis = _holed_product(8, 4, 0.10)
    superset, positions, _ = pair_superset_embedding(basis, n_orb=8)
    assert torch.equal(superset.alpha_of_pair[positions], basis.alpha_of_pair)

    wide = pair_basis_from_arrays(
        basis.alpha_of_pair.detach().cpu().numpy(),
        basis.beta_of_pair.detach().cpu().numpy(),
        n_orb=70,
    )
    assert wide.n_limbs == 2
    wide_superset, wide_positions, _ = pair_superset_embedding(wide)
    assert wide_superset.n_limbs == 2
    assert torch.equal(wide_superset.alpha_of_pair[wide_positions], wide.alpha_of_pair)

    if torch.backends.mps.is_available():
        # The same host-staging path a CUDA basis takes.
        mps = pair_basis_from_arrays(
            basis.alpha_of_pair.numpy(),
            basis.beta_of_pair.numpy(),
            n_orb=8,
            device="mps",
        )
        moved, moved_positions, _ = pair_superset_embedding(mps)
        assert moved.device.type == "mps"
        assert torch.equal(moved.alpha_of_pair[moved_positions].cpu(), basis.alpha_of_pair)


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.gpu
def test_block_matvec_matches_generic_on_cuda(complex_values: bool) -> None:
    """The block and streamed matvecs agree on CUDA (GPU-marked).

    Basis, integrals and helpers are built directly on the device, and the
    result is compared with the CPU streamed matvec, so a device-placement bug
    in either path shows up as a numerical difference or an error.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    dev = torch.device("cuda")
    basis_cpu, norb = _make_basis("tier")
    basis = pair_basis_from_arrays(
        basis_cpu.alpha_of_pair.numpy(), basis_cpu.beta_of_pair.numpy(), device=dev
    )
    integrals = (
        _complex_integrals(norb, device=str(dev))
        if complex_values
        else _real_integrals(norb, device=str(dev))
    )
    integrals_cpu = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = generate_pair_helpers(basis, norb, device=dev, block_matvec="off")
    on = generate_pair_helpers(basis, norb, device=dev, block_matvec="on")
    assert on.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    v = _random_vector(basis.n_pairs, 3, complex_values=complex_values).to(dev)
    expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
    got = mult_pair_torch(hii, v, basis, on, norb, integrals)

    ref_helpers = _helpers_for(basis_cpu, norb, "off")
    ref_hii = make_diag_terms_pair_torch(basis_cpu, norb, integrals_cpu)
    ref = mult_pair_torch(ref_hii, v.cpu(), basis_cpu, ref_helpers, norb, integrals_cpu)
    _assert_close(expected.cpu(), ref, "cuda streamed vs cpu")
    _assert_close(got.cpu(), ref, "cuda block vs cpu")

    row_start, row_stop = 9, 31
    hii_local = make_diag_terms_pair_local(
        basis.alpha_of_pair[row_start:row_stop],
        basis.beta_of_pair[row_start:row_stop],
        norb,
        integrals,
    )
    got_local = mult_pair_local_torch(
        hii_local,
        v[row_start:row_stop],
        v,
        basis,
        on,
        norb,
        integrals,
        row_start,
    )
    _assert_close(got_local.cpu(), ref[row_start:row_stop], "cuda local")


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7])
def test_small_tiles_match_generic(chunk_size: int, complex_values: bool) -> None:
    """Tiny lifted tiles and a tiny byte budget still match the generic matvec.

    Also forces the per-tile ket gather (the group does not fit one tile) and
    exercises the multi-tile loops, which the default ``chunk_size`` skips.
    """
    basis, norb = _make_basis("tier")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    # A 1 KiB budget floors the tile size to min(chunk_size, 2).
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1024, block_matvec="on")
    assert on.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(basis.n_pairs, 3, complex_values=complex_values)
    expected = mult_pair_torch(hii, x, basis, off, norb, integrals)

    got = mult_pair_torch(hii, x, basis, on, norb, integrals, chunk_size=chunk_size)
    _assert_close(got, expected, f"full chunk={chunk_size}")

    row_start, row_stop = 9, 31
    hii_local = make_diag_terms_pair_local(
        basis.alpha_of_pair[row_start:row_stop],
        basis.beta_of_pair[row_start:row_stop],
        norb,
        integrals,
    )
    got = mult_pair_local_torch(
        hii_local,
        x[row_start:row_stop],
        x,
        basis,
        on,
        norb,
        integrals,
        row_start,
        chunk_size=chunk_size,
    )
    _assert_close(got, expected[row_start:row_stop], f"local chunk={chunk_size}")


def test_float32_vector_promotes_like_generic() -> None:
    """An fp32 input promotes to the fp64 output exactly as the generic path does."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    gen = torch.Generator().manual_seed(4)
    v = torch.randn(basis.n_pairs, generator=gen, dtype=torch.float32)
    expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
    got = mult_pair_torch(hii, v, basis, on, norb, integrals)
    assert got.dtype == expected.dtype == torch.float64
    _assert_close(got, expected, "fp32 input")


@pytest.mark.parametrize("complex_values", [False, True])
def test_one_hot_columns_match_generic(complex_values: bool) -> None:
    """Every ket column's block αβ contribution matches the generic one."""
    basis, norb = _make_basis("tier")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    dtype = torch.complex128 if complex_values else torch.float64

    for j in range(basis.n_pairs):
        v = torch.zeros(basis.n_pairs, dtype=dtype)
        v[j] = 1.0
        expected = mult_pair_torch(hii, v, basis, off, norb, integrals)
        got = mult_pair_torch(hii, v, basis, on, norb, integrals)
        _assert_close(got, expected, f"one-hot ket={j}")


# ============================================================================
# 2. Owner-computes parity
# ============================================================================


@pytest.mark.parametrize("basis_name", ["full_product", "tier", "irregular"])
def test_local_block_matches_full_slice(basis_name: str) -> None:
    """A block-on local matvec equals the generic full matvec's owned slice."""
    basis, norb = _make_basis(basis_name)
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    assert on.blocks is not None
    n = basis.n_pairs
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(n, 11, complex_values=False)
    expected = mult_pair_torch(hii, x, basis, off, norb, integrals)

    ranges = [(0, 6), (5, 12), (12, 24), (0, n), (n // 2, 1), (n, 0)]
    for row_start, n_local in ranges:
        if row_start > n:
            continue
        n_local = min(n_local, n - row_start)
        row_stop = row_start + n_local
        hii_local = make_diag_terms_pair_local(
            basis.alpha_of_pair[row_start:row_stop],
            basis.beta_of_pair[row_start:row_stop],
            norb,
            integrals,
        )
        got = mult_pair_local_torch(
            hii_local,
            x[row_start:row_stop],
            x,
            basis,
            on,
            norb,
            integrals,
            row_start,
        )
        _assert_close(got, expected[row_start:row_stop], f"{basis_name} [{row_start},{row_stop})")


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _gloo_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank's block-on local output equals the generic full slice."""
    _init_gloo(rank, world_size, init_file)
    try:
        norb = 4
        basis = _uneven_basis()
        n = basis.n_pairs
        assert n == 35  # uneven for 2 ranks
        integrals = _real_integrals(norb)
        off = _helpers_for(basis, norb, "off")
        on = _helpers_for(basis, norb, "on")
        assert on.blocks is not None

        bounds = partition_vector(n, world_size, rank)
        hii = make_diag_terms_pair_torch(basis, norb, integrals)
        hii_local = make_diag_terms_pair_local(
            basis.alpha_of_pair[bounds.start : bounds.stop],
            basis.beta_of_pair[bounds.start : bounds.stop],
            norb,
            integrals,
        )
        x = _random_vector(n, 7, complex_values=False)
        expected = mult_pair_torch(hii, x, basis, off, norb, integrals)
        got = mult_pair_local_torch(
            hii_local,
            x[bounds.start : bounds.stop],
            x,
            basis,
            on,
            norb,
            integrals,
            bounds.start,
        )
        sl = slice(bounds.start, bounds.stop)
        assert torch.allclose(got, expected[sl], atol=1e-12, rtol=0.0), (
            rank,
            world_size,
            float((got - expected[sl]).abs().max()),
        )
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_local_block_gloo_two_ranks_uneven(tmp_path) -> None:
    """gloo, 2 ranks, 18/17 pairs: each block matches the full slice to 1e-12."""
    _skip_without_gloo()
    init_file = tmp_path / "pair_blocks_init"
    out_file = tmp_path / "pair_blocks_out"
    mp.start_processes(
        _gloo_worker,
        args=(2, str(init_file), str(out_file)),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


# ============================================================================
# 3. Label / convention guard against a dense determinant evaluation
# ============================================================================


@pytest.mark.parametrize("complex_values", [False, True])
def test_move_labels_match_dense_two_excite(complex_values: bool) -> None:
    """Per-entry H equals ``two_excite_batch`` on the bra det, and labels are right."""
    norb, n_a, n_b = 4, 2, 2
    # A 2x2 product whose strings differ by exactly one bit per spin:
    # α: 0011 <-> 0101, β: 0011 <-> 0110.
    a1, a2 = 0b0011, 0b0101
    b1, b2 = 0b0011, 0b0110
    basis = pair_basis_from_arrays(np.repeat([a1, a2], 2), np.tile([b1, b2], 2))
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    assert len(helpers.blocks.by_beta.groups) == 1
    group = helpers.blocks.by_beta.groups[0]
    assert int(group.pair_idx.numel()) == 4

    moves_a = _single_moves(
        group.alpha_uids,
        group.alpha_uids,
        helpers.unique_alphas,
        n_orb=norb,
        n_particles=n_a,
        spin_offset=0,
    )
    moves_b = _single_moves(
        group.beta_uids,
        group.beta_uids,
        helpers.unique_betas,
        n_orb=norb,
        n_particles=n_b,
        spin_offset=1,
    )
    assert moves_a is not None and moves_b is not None
    r_src_a, r_dst_a, cr_a, an_a = moves_a
    r_src_b, r_dst_b, cr_b, an_b = moves_b
    source_alpha = helpers.unique_alphas[group.alpha_uids][r_src_a]
    target_alpha = helpers.unique_alphas[group.alpha_uids][r_dst_a]
    source_beta = helpers.unique_betas[group.beta_uids][r_src_b]
    target_beta = helpers.unique_betas[group.beta_uids][r_dst_b]

    # an is occupied in the ket source, cr in the bra target.
    assert torch.equal(
        source_alpha ^ (torch.tensor(1) << (an_a // 2)) ^ (torch.tensor(1) << (cr_a // 2)),
        target_alpha,
    )
    assert torch.equal(
        source_beta ^ (torch.tensor(1) << (an_b // 2)) ^ (torch.tensor(1) << (cr_b // 2)),
        target_beta,
    )

    got = _alphabeta_pair_matrix(
        cr_a,
        an_a,
        cr_b,
        an_b,
        target_alpha,
        target_beta,
        integrals,
        integrals.I2_eri.dtype,
    )
    n_a_moves, n_b_moves = target_alpha.numel(), target_beta.numel()
    bra_det = det_from_alpha_beta_batch(
        target_alpha.repeat_interleave(n_b_moves), target_beta.repeat(n_a_moves), norb
    )
    expected = two_excite_batch(
        bra_det,
        None,
        an_a.repeat_interleave(n_b_moves),
        an_b.repeat(n_a_moves),
        cr_a.repeat_interleave(n_b_moves),
        cr_b.repeat(n_a_moves),
        integrals.I2_eri,
    ).reshape(n_a_moves, n_b_moves)

    assert float(expected.abs().max()) > 0.0, "hand-picked entries are all zero"
    assert torch.equal(got, expected), float((got - expected).abs().max())


# ============================================================================
# 4. Same-spin block channels: isolated parity and enumeration labels
# ============================================================================


def _generic_same_spin(
    x: torch.Tensor,
    basis: PairBasis,
    helpers,
    norb: int,
    integrals,
) -> torch.Tensor:
    """Same-spin contribution of the generic streamed scatter into a zero vector."""
    dtype = torch.complex128 if x.is_complex() else torch.float64
    Wb = torch.zeros(basis.n_pairs, dtype=dtype)
    common = dict(
        alpha_of_pair=basis.alpha_of_pair,
        beta_of_pair=basis.beta_of_pair,
        n_orb=norb,
        integrals=integrals,
    )
    for stream in (helpers.asingles, helpers.bsingles):
        for t in stream():
            _scatter_single_pair(
                Wb=Wb,
                T=x,
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr=t["cr0"],
                an=t["an0"],
                chunk_size=10_000,
                **common,
            )
    for stream in (helpers.adoubles, helpers.bdoubles):
        for t in stream():
            _scatter_double_pair(
                Wb=Wb,
                T=x,
                k_bra=t["k_bra"],
                k_ket=t["k_ket"],
                cr0=t["cr0"],
                cr1=t["cr1"],
                an0=t["an0"],
                an1=t["an1"],
                chunk_size=10_000,
                **common,
            )
    return Wb


@pytest.mark.parametrize("basis_name", ["full_product", "tier", "irregular"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_same_spin_block_matches_generic(basis_name: str, complex_values: bool) -> None:
    """The block same-spin channels equal the four generic streams exactly."""
    basis, norb = _make_basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    assert on.blocks is not None

    for seed in (0, 1, 2):
        x = _random_vector(basis.n_pairs, seed, complex_values=complex_values)
        expected = _generic_same_spin(x, basis, off, norb, integrals)
        got = torch.zeros_like(expected)
        add_same_spin_blocks(got, x, on, integrals)
        _assert_close(got, expected, f"{basis_name} seed={seed}")


@pytest.mark.parametrize("basis_name", ["full_product", "tier", "irregular"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_same_spin_local_matches_generic_slice(basis_name: str, complex_values: bool) -> None:
    """An owner-computes same-spin block matches the generic owned slice.

    The ranges include windows that stay on one α row, windows that wrap across
    several rows, a single row and an empty block.
    """
    basis, norb = _make_basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    n = basis.n_pairs
    x = _random_vector(n, 11, complex_values=complex_values)
    expected = _generic_same_spin(x, basis, off, norb, integrals)

    ranges = [(0, 6), (5, 12), (12, 24), (0, n), (n // 2, 1), (max(0, n - 5), 5), (n, 0)]
    for row_start, n_local in ranges:
        if row_start > n:
            continue
        n_local = min(n_local, n - row_start)
        row_stop = row_start + n_local
        got = torch.zeros_like(expected[row_start:row_stop])
        add_same_spin_blocks_local(got, x, on, integrals, row_start)
        _assert_close(got, expected[row_start:row_stop], f"{basis_name} [{row_start},{row_stop})")


def test_same_spin_float32_vector_promotes_like_generic() -> None:
    """An fp32 input promotes to the fp64 output exactly as the generic path does."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(basis.n_pairs, generator=gen, dtype=torch.float32)
    expected = _generic_same_spin(x, basis, off, norb, integrals)
    got = torch.zeros_like(expected)
    add_same_spin_blocks(got, x, on, integrals)
    assert got.dtype == expected.dtype == torch.float64
    _assert_close(got, expected, "fp32 input")


@pytest.mark.parametrize("spin", [0, 1])
def test_double_moves_enumeration_matches_bruteforce(spin: int) -> None:
    """``_double_moves`` labels take the source to the target and cover every double."""
    basis, norb = _make_basis("tier")
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    groups = helpers.blocks.by_beta if spin == 0 else helpers.blocks.by_alpha
    unique = helpers.unique_alphas if spin == 0 else helpers.unique_betas
    n_particles = helpers.n_a if spin == 0 else helpers.n_b
    seen: set[tuple[int, int]] = set()

    for g in groups.groups:
        uids = g.alpha_uids if spin == 0 else g.beta_uids
        moves = _double_moves(
            uids, uids, unique, n_orb=norb, n_particles=n_particles, spin_offset=spin
        )
        strings = unique[uids]
        brute = {
            (i, j)
            for i, u in enumerate(strings.tolist())
            for j, v in enumerate(strings.tolist())
            if (u ^ v).bit_count() == 4
        }
        if not brute:
            assert moves is None
            continue
        assert moves is not None
        r_src, r_dst, cr0, cr1, an0, an1 = moves
        got = set(zip(r_src.tolist(), r_dst.tolist()))
        assert got == brute

        source = strings[r_src]
        target = strings[r_dst]
        one = torch.tensor(1, dtype=torch.int64)
        moved = source ^ (one << (an0 // 2)) ^ (one << (an1 // 2))
        moved = moved ^ (one << (cr0 // 2)) ^ (one << (cr1 // 2))
        assert torch.equal(moved, target)
        assert bool(torch.all(an0 < an1)) and bool(torch.all(cr0 < cr1))
        assert bool(torch.all(((source >> (an0 // 2)) & 1) == 1))
        assert bool(torch.all(((source >> (an1 // 2)) & 1) == 1))
        assert bool(torch.all(((source >> (cr0 // 2)) & 1) == 0))
        assert bool(torch.all(((source >> (cr1 // 2)) & 1) == 0))
        seen |= got

    assert seen, "the tier basis has no double moves"


@pytest.mark.parametrize("complex_values", [False, True])
def test_alpha_double_block_matches_dense_two_excite(complex_values: bool) -> None:
    """The α-double block kernel reproduces dense Slater-Condon elements.

    The basis is all six two-electron α strings against one β string, so every
    α double is one row of a single 6×1 group and ``two_excite_batch`` on the
    interleaved bra determinant is the reference.
    """
    norb, n_a = 4, 2
    base = _full_product(norb, n_a)
    alphas = base.unique_alphas.numpy()
    beta0 = base.unique_betas.numpy()[:1]
    basis = pair_basis_from_arrays(alphas, np.tile(beta0, alphas.shape[0]))
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    groups = helpers.blocks.by_beta.groups
    assert len(groups) == 1
    group = groups[0]
    assert int(group.alpha_uids.numel()) == 6
    assert int(group.beta_uids.numel()) == 1

    moves = _double_moves(
        group.alpha_uids,
        group.alpha_uids,
        helpers.unique_alphas,
        n_orb=norb,
        n_particles=n_a,
        spin_offset=0,
    )
    assert moves is not None
    r_src, r_dst, cr0, cr1, an0, an1 = moves
    adets = helpers.unique_alphas[group.alpha_uids]
    bdets = helpers.unique_betas[group.beta_uids]
    t_ket = _random_vector(
        int(adets.numel()) * int(bdets.numel()), 5, complex_values=complex_values
    )
    out = torch.zeros_like(t_ket)
    _apply_alpha_double_chunk_fast(
        out,
        t_ket,
        adets,
        bdets,
        norb,
        integrals,
        bra_alpha_per_exc=r_dst,
        ket_alpha_per_exc=r_src,
        cr0=cr0,
        cr1=cr1,
        an0=an0,
        an1=an1,
        M_a=int(adets.numel()),
        M_b=int(bdets.numel()),
        chunk_size=2_000_000,
    )

    n_moves = int(r_src.numel())
    bra_det = det_from_alpha_beta_batch(adets[r_dst], bdets[0].repeat(n_moves), norb)
    elements = two_excite_batch(bra_det, None, an0, an1, cr0, cr1, integrals.I2_eri)
    expected = torch.zeros_like(out)
    expected.index_add_(0, r_dst, elements * t_ket[r_src])

    assert float(expected.abs().max()) > 0.0, "hand-picked entries are all zero"
    assert torch.equal(out, expected), float((out - expected).abs().max())


# ============================================================================
# 5. αβ enumeration memory bound
# ============================================================================


def _wide_asymmetric_basis(n_alpha: int, n_betas: int) -> tuple[PairBasis, int]:
    """Random 10-electron α strings against one or two β strings on 20 orbitals.

    The β strings have equal electron count and the second is a single
    ``0 -> 10`` excitation of the first, so the group admits β moves when
    ``n_betas == 2`` and none when ``n_betas == 1``.
    """
    norb, n_elec = 20, 10
    rng = np.random.default_rng(0)
    seen: set[int] = set()
    while len(seen) < n_alpha:
        draw = rng.integers(0, 1 << norb, size=4 * (n_alpha - len(seen)) + 64, dtype=np.int64)
        seen.update(int(x) for x in draw[np.bitwise_count(draw) == n_elec])
    alphas = np.array(sorted(seen)[:n_alpha], dtype=np.int64)
    hf = (1 << n_elec) - 1
    betas = np.array([hf, (hf ^ 1) | (1 << n_elec)][:n_betas], dtype=np.int64)
    basis = pair_basis_from_arrays(np.repeat(alphas, n_betas), np.tile(betas, n_alpha))
    return basis, norb


def _record_single_moves(monkeypatch: pytest.MonkeyPatch, calls: list[int]) -> None:
    """Append each ``_single_moves`` call's source-set size to ``calls``."""
    original = mult_pair_blocks._single_moves

    def _recording_single_moves(
        source_uids: torch.Tensor,
        target_uids: torch.Tensor,
        unique: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...] | None:
        calls.append(int(source_uids.numel()))
        return original(source_uids, target_uids, unique, **kwargs)

    monkeypatch.setattr(mult_pair_blocks, "_single_moves", _recording_single_moves)


def test_ab_enumeration_is_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The αβ move tables are chunked by source strings within the byte budget.

    A 19,999-string α side with a live β move exercises the α chunk loop; every
    enumerator call must be bounded by the computed chunk size, never by the
    whole group, and the channel output must still match the generic stream.
    """
    n_alpha, n_betas = 19_999, 2
    basis, norb = _wide_asymmetric_basis(n_alpha, n_betas)
    budget = 1 << 20
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=budget, block_matvec="on")
    assert on.blocks is not None
    assert len(on.blocks.by_beta.groups) == 1
    assert int(on.blocks.by_beta.groups[0].alpha_uids.numel()) == n_alpha

    calls: list[int] = []
    _record_single_moves(monkeypatch, calls)
    x = _random_vector(basis.n_pairs, 2, complex_values=False)
    got = torch.zeros(basis.n_pairs, dtype=torch.float64)
    add_ab_blocks(got, x, on, integrals)

    n_a = int(on.n_a)
    enum_chunk = _chunk_pairs(n_a * (norb - n_a), "singles", budget // 4)
    assert enum_chunk < n_alpha
    assert calls
    assert max(calls) <= enum_chunk
    assert n_alpha not in calls
    assert enum_chunk in calls  # the α source set really is split

    expected = torch.zeros(basis.n_pairs, dtype=torch.float64)
    common = dict(
        alpha_of_pair=basis.alpha_of_pair,
        beta_of_pair=basis.beta_of_pair,
        n_orb=norb,
        integrals=integrals,
    )
    for t in off.ab_doubles():
        _scatter_double_pair(
            Wb=expected,
            T=x,
            k_bra=t["k_bra"],
            k_ket=t["k_ket"],
            cr0=t["cr_a"],
            cr1=t["cr_b"],
            an0=t["an_a"],
            an1=t["an_b"],
            chunk_size=2_000_000,
            **common,
        )
    assert float(expected.abs().max()) > 0.0, "the basis admits no αβ rows"
    _assert_close(got, expected, "chunked αβ")


def test_ab_enumeration_skips_side_without_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single β string has no β move, so the α table is never enumerated.

    Both the channel and the full matvec must stay exact while every
    enumeration call remains chunk-bounded.
    """
    n_alpha = 19_999
    basis, norb = _wide_asymmetric_basis(n_alpha, 1)
    budget = 1 << 20
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=budget, block_matvec="on")
    assert on.blocks is not None
    x = _random_vector(basis.n_pairs, 3, complex_values=False)

    calls: list[int] = []
    _record_single_moves(monkeypatch, calls)
    got = torch.zeros(basis.n_pairs, dtype=torch.float64)
    add_ab_blocks(got, x, on, integrals)
    assert calls and max(calls) == 1  # only the one-string β source set
    assert not bool(got.any()), "a one-β-string basis cannot admit an αβ row"

    calls.clear()
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    expected = mult_pair_torch(hii, x, basis, off, norb, integrals)
    got_full = mult_pair_torch(hii, x, basis, on, norb, integrals)
    _assert_close(got_full, expected, "asymmetric full matvec")

    n_a = int(on.n_a)
    same_spin_chunk = _chunk_pairs(n_a * (norb - n_a), "singles", budget // 2)
    ab_chunk = _chunk_pairs(n_a * (norb - n_a), "singles", budget // 4)
    assert max(calls) <= max(same_spin_chunk, ab_chunk)
    assert n_alpha not in calls


def test_ab_local_enumeration_uses_owned_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An owner-computes αβ pass enumerates from its owned bra rows only.

    The wide fixture has one α group of 19,999 strings. A one-row owned block
    must enumerate one α and one β string per call — not the whole group — and
    still match the full channel on that row.
    """
    n_alpha, n_betas = 19_999, 2
    basis, norb = _wide_asymmetric_basis(n_alpha, n_betas)
    integrals = _real_integrals(norb)
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1 << 20, block_matvec="on")
    assert on.blocks is not None

    calls: list[int] = []
    _record_single_moves(monkeypatch, calls)
    x = _random_vector(basis.n_pairs, 5, complex_values=False)
    got = torch.zeros(1, dtype=torch.float64)
    add_ab_blocks_local(got, x, on, integrals, row_start=0)

    assert calls and max(calls) <= 1
    assert n_alpha not in calls

    expected = torch.zeros(basis.n_pairs, dtype=torch.float64)
    add_ab_blocks(expected, x, on, integrals)
    _assert_close(got, expected[:1], "owned αβ row")


def test_block_kernels_require_pair_blocks() -> None:
    """The block entry points reject helpers built with ``block_matvec="off"``."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "off")
    x = _random_vector(basis.n_pairs, 0, complex_values=False)
    Wb = torch.zeros(basis.n_pairs, dtype=torch.float64)
    with pytest.raises(ValueError, match="blocks"):
        add_ab_blocks(Wb, x, helpers, integrals)
    with pytest.raises(ValueError, match="blocks"):
        add_same_spin_blocks(Wb, x, helpers, integrals)


# ============================================================================
# 6. Same-spin memory bounds, multi-chunk offsets, dense guards, degeneracies
# ============================================================================


def _dense_same_spin_basis() -> tuple[PairBasis, int]:
    """Dense-survival asymmetric fixture on 12 orbitals with 6+6 electrons.

    Every ``C(12, 6) = 924`` α string paired with the HF β string, plus the HF
    α string paired with every β string. Both the α group (924 x 1) and the β
    group (1 x 924) are closed under same-spin moves, so every candidate
    survives and the kernels' setup is maximal.
    """
    norb, nelec = 12, 6
    strings = np.array(
        [sum(1 << int(x) for x in combo) for combo in itertools.combinations(range(norb), nelec)],
        dtype=np.int64,
    )
    hf = (1 << nelec) - 1
    alphas = np.concatenate([strings, np.full(len(strings), hf, dtype=np.int64)])
    betas = np.concatenate([np.full(len(strings), hf, dtype=np.int64), strings])
    return pair_basis_from_arrays(alphas, betas), norb


def _record_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    single_calls: list[int],
    double_calls: list[int],
) -> None:
    """Record each move-enumeration call's source-set size."""
    original_single = mult_pair_blocks._single_moves
    original_double = mult_pair_blocks._double_moves

    def _recording_single(source_uids: torch.Tensor, *args: object, **kwargs: object):
        single_calls.append(int(source_uids.numel()))
        return original_single(source_uids, *args, **kwargs)

    def _recording_double(source_uids: torch.Tensor, *args: object, **kwargs: object):
        double_calls.append(int(source_uids.numel()))
        return original_double(source_uids, *args, **kwargs)

    monkeypatch.setattr(mult_pair_blocks, "_single_moves", _recording_single)
    monkeypatch.setattr(mult_pair_blocks, "_double_moves", _recording_double)


def _record_kernel_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple[int, int]]]:
    """Record each product-kernel call's ``(chunk_size, survivors)``."""
    recorded: dict[str, list[tuple[int, int]]] = {}
    for name in (
        "_apply_alpha_single_chunk_fast",
        "_apply_alpha_double_chunk_fast",
        "_apply_beta_single_chunk_fast",
        "_apply_beta_double_chunk_fast",
    ):
        original = getattr(mult_pair_blocks, name)
        recorded[name] = []

        def _wrapped(*args: object, _original=original, _name=name, **kwargs: object):  # noqa: ANN202
            bra = kwargs.get("bra_alpha_per_exc", kwargs.get("bra_beta_per_exc"))
            recorded[_name].append((int(kwargs["chunk_size"]), int(bra.numel())))
            return _original(*args, **kwargs)

        monkeypatch.setattr(mult_pair_blocks, name, _wrapped)
    return recorded


def test_same_spin_enumeration_is_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-spin enumeration and survivor slices honour the budget on a dense basis.

    The 924-string α (and β) side survives completely: the enumeration stays
    candidate-bounded and every survivor table is sliced to the per-call setup
    cap, while the tile floor bounds the lifted tiles.
    """
    basis, norb = _dense_same_spin_basis()
    budget = 1 << 20
    chunk_size = 2_000_000
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=budget, block_matvec="on")
    assert on.blocks is not None
    x = _random_vector(basis.n_pairs, 4, complex_values=False)

    single_calls: list[int] = []
    double_calls: list[int] = []
    _record_enumeration(monkeypatch, single_calls, double_calls)
    kernel_calls = _record_kernel_calls(monkeypatch)

    got = torch.zeros(basis.n_pairs, dtype=torch.float64)
    add_same_spin_blocks(got, x, on, integrals, chunk_size=chunk_size)

    n_a, n_empty = int(on.n_a), norb - int(on.n_a)
    enum_budget = budget // 2
    enum_single = _chunk_pairs(n_a * n_empty, "singles", enum_budget)
    enum_double = _chunk_pairs(
        (n_a * (n_a - 1) // 2) * (n_empty * (n_empty - 1) // 2), "doubles", enum_budget
    )
    survivor_single = mult_pair_blocks._same_spin_survivor_chunk(
        "singles", norb, enum_budget, integrals.I2_eri.element_size()
    )
    survivor_double = mult_pair_blocks._same_spin_survivor_chunk(
        "doubles", norb, enum_budget, integrals.I2_eri.element_size()
    )
    n_strings = int(on.m_alpha)
    assert enum_single < n_strings and enum_double < n_strings
    assert survivor_single < enum_single * n_a * n_empty  # dense survival is sliced
    assert survivor_double < enum_double * (n_a * (n_a - 1) // 2) * (n_empty * (n_empty - 1) // 2)
    assert single_calls and max(single_calls) <= enum_single
    assert double_calls and max(double_calls) <= enum_double
    assert n_strings not in single_calls + double_calls
    assert max(single_calls) == enum_single
    assert max(double_calls) == enum_double

    tile_cap = max(1, min(chunk_size, budget // 512))
    assert all(calls for calls in kernel_calls.values()), "a channel did not run"
    for name, calls in kernel_calls.items():
        cap = survivor_double if "double" in name else survivor_single
        assert {chunk for chunk, _ in calls} == {tile_cap}, name
        assert max(survivors for _, survivors in calls) <= cap, name
        assert cap in {survivors for _, survivors in calls}, name
    assert tile_cap < chunk_size

    expected = _generic_same_spin(x, basis, off, norb, integrals)
    _assert_close(got, expected, "dense same-spin memory bound")


def test_same_spin_local_enumeration_uses_owned_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An owner-computes same-spin pass enumerates from its owned slab only.

    With a one-row owned block every α and β move enumeration sees one string,
    not the 19,999-string group, while the row still matches the full channel.
    """
    n_alpha, n_betas = 19_999, 2
    basis, norb = _wide_asymmetric_basis(n_alpha, n_betas)
    integrals = _real_integrals(norb)
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1 << 20, block_matvec="on")
    assert on.blocks is not None

    single_calls: list[int] = []
    double_calls: list[int] = []
    _record_enumeration(monkeypatch, single_calls, double_calls)
    x = _random_vector(basis.n_pairs, 6, complex_values=False)
    got = torch.zeros(1, dtype=torch.float64)
    add_same_spin_blocks_local(got, x, on, integrals, 0)

    assert single_calls and max(single_calls) <= 1
    assert double_calls and max(double_calls) <= 1
    assert n_alpha not in single_calls + double_calls

    expected = torch.zeros(basis.n_pairs, dtype=torch.float64)
    add_same_spin_blocks(expected, x, on, integrals)
    _assert_close(got, expected[:1], "owned same-spin row")


def test_pair_work_weights_follow_group_geometry() -> None:
    """Per-pair weights split each group's candidate cost over its grid."""
    n_alpha, n_betas = 9_999, 2
    basis, norb = _wide_asymmetric_basis(n_alpha, n_betas)
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1 << 20, block_matvec="on")
    assert on.blocks is not None
    weights = mult_pair_blocks.pair_work_weights(on)
    assert weights.shape == (basis.n_pairs,)

    n_a, n_b = int(on.n_a), int(on.n_b)
    v_a, v_b = norb - n_a, norb - n_b
    cand_a = n_a * v_a + (n_a * (n_a - 1) // 2) * (v_a * (v_a - 1) // 2)
    cand_b = n_b * v_b + (n_b * (n_b - 1) // 2) * (v_b * (v_b - 1) // 2)
    expected = cand_a / n_betas + cand_b / n_alpha
    same_spin = weights - mult_pair_blocks._AB_ENTRY_COST * mult_pair_blocks._ab_entry_weights(on)
    assert torch.allclose(same_spin, torch.full_like(weights, expected))

    off = _helpers_for(basis, norb, "off")
    with pytest.raises(ValueError, match="blocks"):
        mult_pair_blocks.pair_work_weights(off)
    with pytest.raises(ValueError, match="blocks"):
        mult_pair_blocks._ab_entry_weights(off)


def test_ab_entry_weights_count_crossed_single_moves() -> None:
    """Each pair's αβ weight is its crossed single-move count.

    The full ``C(4, 2)`` α × β space is closed under single moves and every
    string has ``2 · 2 = 4`` singles, so every pair contributes ``4 · 4 = 16``
    lifted entries to the one by_beta group and the total weight is the same
    for every pair.
    """
    norb, nelec = 4, 2
    strings = np.array(
        [sum(1 << int(x) for x in combo) for combo in itertools.combinations(range(norb), nelec)],
        dtype=np.int64,
    )
    basis = pair_basis_from_arrays(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1 << 20, block_matvec="on")
    assert on.blocks is not None
    entries = mult_pair_blocks._ab_entry_weights(on)
    assert torch.equal(entries, torch.full_like(entries, 16.0))

    weights = mult_pair_blocks.pair_work_weights(on)
    v = norb - nelec
    cand = nelec * v + (nelec * (nelec - 1) // 2) * (v * (v - 1) // 2)
    expected = 2 * cand / len(strings) + mult_pair_blocks._AB_ENTRY_COST * 16.0
    assert torch.allclose(weights, torch.full_like(weights, expected))


def _dense_single_beta_column(norb: int = 16, nelec: int = 8) -> tuple[PairBasis, int]:
    """Every ``C(norb, nelec)`` α string against the HF β string.

    The whole spin-string set is closed under same-spin moves, so every
    candidate survives; with a single β column the α kernel setup is maximal.
    """
    strings = np.array(
        [sum(1 << int(x) for x in combo) for combo in itertools.combinations(range(norb), nelec)],
        dtype=np.int64,
    )
    beta = np.full(len(strings), (1 << nelec) - 1, dtype=np.int64)
    return pair_basis_from_arrays(strings, beta), norb


def test_same_spin_dense_fixture_peak_rss() -> None:
    """The 12,870-string dense fixture peaks within twice its byte budget.

    Runs in a fresh process so ``ru_maxrss`` reflects only this fixture. The
    candidate-bounded enumeration plus survivor slicing should stay near the
    128 MiB budget; the pre-slicing code measured 448 MiB (3.5x) here, so this
    threshold fails on that implementation.
    """
    budget = 128 << 20
    tests_dir = Path(__file__).resolve().parent
    code = f"""
import resource, sys
import torch
sys.path.insert(0, {str(tests_dir)!r})
from test_mult_pair_blocks import _dense_single_beta_column, _real_integrals
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
import torch_sci_solver.torch_impl.mult_pair_blocks as mpb

basis, norb = _dense_single_beta_column()
ints = _real_integrals(norb)
on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes={budget}, block_matvec="on")
x = torch.randn(basis.n_pairs, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mpb.add_same_spin_blocks(
    torch.zeros(basis.n_pairs, dtype=torch.float64), x, on, ints, chunk_size=2_000_000
)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1024 ** 2 if sys.platform == "darwin" else 1024
print((after - before) / scale)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    peak_mib = float(result.stdout.strip().splitlines()[-1])
    assert peak_mib < 2 * budget / 1024**2, peak_mib


@pytest.mark.parametrize("complex_values", [False, True])
def test_same_spin_multi_chunk_enumeration(
    monkeypatch: pytest.MonkeyPatch, complex_values: bool
) -> None:
    """Chunk size one still reproduces every channel's source offsets.

    With one source string per enumeration call, ``ket = r_src + start`` is the
    only thing that can carry the chunk offset, so parity here guards all four
    channels' offset handling.
    """
    basis, norb = _make_basis("tier")
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = generate_pair_helpers(basis, norb, device="cpu", budget_bytes=1024, block_matvec="on")
    assert on.blocks is not None
    x = _random_vector(basis.n_pairs, 6, complex_values=complex_values)

    single_calls: list[int] = []
    double_calls: list[int] = []
    _record_enumeration(monkeypatch, single_calls, double_calls)
    kernel_calls = _record_kernel_calls(monkeypatch)

    got = torch.zeros_like(_generic_same_spin(x, basis, off, norb, integrals))
    add_same_spin_blocks(got, x, on, integrals)

    assert single_calls and max(single_calls) == 1
    assert double_calls and max(double_calls) == 1
    assert all(len(calls) >= 2 for calls in kernel_calls.values()), (
        "each channel must be called across several chunks"
    )

    expected = _generic_same_spin(x, basis, off, norb, integrals)
    _assert_close(got, expected, "multi-chunk enumeration")


def _one_beta_column() -> tuple[PairBasis, int]:
    """All six two-electron α strings against the HF β string: one 6x1 group."""
    norb, nelec = 4, 2
    base = _full_product(norb, nelec)
    alphas = base.unique_alphas.numpy()
    beta = np.full(len(alphas), (1 << nelec) - 1, dtype=np.int64)
    return pair_basis_from_arrays(alphas, beta), norb


def _one_alpha_row() -> tuple[PairBasis, int]:
    """The HF α string against all six two-electron β strings: one 1x6 group."""
    norb, nelec = 4, 2
    base = _full_product(norb, nelec)
    betas = base.unique_betas.numpy()
    alpha = np.full(len(betas), (1 << nelec) - 1, dtype=np.int64)
    return pair_basis_from_arrays(alpha, betas), norb


@pytest.mark.parametrize("complex_values", [False, True])
def test_alpha_single_block_matches_dense_one_excite(complex_values: bool) -> None:
    """The α-single block kernel reproduces dense one-excitation elements."""
    basis, norb = _one_beta_column()
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    group = helpers.blocks.by_beta.groups[0]
    adets = helpers.unique_alphas[group.alpha_uids]
    bdets = helpers.unique_betas[group.beta_uids]
    moves = _single_moves(
        group.alpha_uids,
        group.alpha_uids,
        helpers.unique_alphas,
        n_orb=norb,
        n_particles=int(helpers.n_a),
        spin_offset=0,
    )
    assert moves is not None
    r_src, r_dst, cr, an = moves
    t = _random_vector(int(adets.numel()) * int(bdets.numel()), 7, complex_values=complex_values)
    out = torch.zeros_like(t)
    _apply_alpha_single_chunk_fast(
        out,
        t,
        adets,
        bdets,
        norb,
        integrals,
        bra_alpha_per_exc=r_dst,
        ket_alpha_per_exc=r_src,
        cr=cr,
        an=an,
        M_a=int(adets.numel()),
        M_b=int(bdets.numel()),
        chunk_size=2_000_000,
    )

    n_moves = int(r_src.numel())
    bra_det = det_from_alpha_beta_batch(adets[r_dst], bdets[0].repeat(n_moves), norb)
    elements = one_excite_batch(
        bra_det, None, an, cr, integrals.I1, integrals.I2_eri, integrals.n_spinorb
    )
    expected = torch.zeros_like(out)
    expected.index_add_(0, r_dst, elements * t[r_src])
    assert float(expected.abs().max()) > 0.0, "hand-picked entries are all zero"
    _assert_close(out, expected, "α single dense")


@pytest.mark.parametrize("complex_values", [False, True])
def test_beta_single_block_matches_dense_one_excite(complex_values: bool) -> None:
    """The β-single block kernel reproduces dense one-excitation elements."""
    basis, norb = _one_alpha_row()
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    group = helpers.blocks.by_alpha.groups[0]
    adets = helpers.unique_alphas[group.alpha_uids]
    bdets = helpers.unique_betas[group.beta_uids]
    moves = _single_moves(
        group.beta_uids,
        group.beta_uids,
        helpers.unique_betas,
        n_orb=norb,
        n_particles=int(helpers.n_b),
        spin_offset=1,
    )
    assert moves is not None
    r_src, r_dst, cr, an = moves
    t = _random_vector(int(adets.numel()) * int(bdets.numel()), 7, complex_values=complex_values)
    out = torch.zeros_like(t)
    _apply_beta_single_chunk_fast(
        out,
        t,
        adets,
        bdets,
        norb,
        integrals,
        bra_beta_per_exc=r_dst,
        ket_beta_per_exc=r_src,
        cr=cr,
        an=an,
        M_a=int(adets.numel()),
        M_b=int(bdets.numel()),
        chunk_size=2_000_000,
    )

    n_moves = int(r_src.numel())
    bra_det = det_from_alpha_beta_batch(adets[0].repeat(n_moves), bdets[r_dst], norb)
    elements = one_excite_batch(
        bra_det, None, an, cr, integrals.I1, integrals.I2_eri, integrals.n_spinorb
    )
    t2 = t.view(int(adets.numel()), int(bdets.numel()))
    expected = torch.zeros_like(t2)
    expected.index_add_(1, r_dst, elements.unsqueeze(0) * t2[:, r_src])
    assert float(expected.abs().max()) > 0.0, "hand-picked entries are all zero"
    _assert_close(out.view_as(expected), expected, "β single dense")


@pytest.mark.parametrize("complex_values", [False, True])
def test_beta_double_block_matches_dense_two_excite(complex_values: bool) -> None:
    """The β-double block kernel reproduces dense two-excitation elements."""
    basis, norb = _one_alpha_row()
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    group = helpers.blocks.by_alpha.groups[0]
    adets = helpers.unique_alphas[group.alpha_uids]
    bdets = helpers.unique_betas[group.beta_uids]
    moves = _double_moves(
        group.beta_uids,
        group.beta_uids,
        helpers.unique_betas,
        n_orb=norb,
        n_particles=int(helpers.n_b),
        spin_offset=1,
    )
    assert moves is not None
    r_src, r_dst, cr0, cr1, an0, an1 = moves
    t = _random_vector(int(adets.numel()) * int(bdets.numel()), 8, complex_values=complex_values)
    out = torch.zeros_like(t)
    _apply_beta_double_chunk_fast(
        out,
        t,
        adets,
        bdets,
        norb,
        integrals,
        bra_beta_per_exc=r_dst,
        ket_beta_per_exc=r_src,
        cr0=cr0,
        cr1=cr1,
        an0=an0,
        an1=an1,
        M_a=int(adets.numel()),
        M_b=int(bdets.numel()),
        chunk_size=2_000_000,
    )

    n_moves = int(r_src.numel())
    bra_det = det_from_alpha_beta_batch(adets[0].repeat(n_moves), bdets[r_dst], norb)
    elements = two_excite_batch(bra_det, None, an0, an1, cr0, cr1, integrals.I2_eri)
    t2 = t.view(int(adets.numel()), int(bdets.numel()))
    expected = torch.zeros_like(t2)
    expected.index_add_(1, r_dst, elements.unsqueeze(0) * t2[:, r_src])
    assert float(expected.abs().max()) > 0.0, "hand-picked entries are all zero"
    assert torch.equal(out.view_as(expected), expected), float(
        (out.view_as(expected) - expected).abs().max()
    )


def _degenerate_or_asymmetric_basis(kind: str) -> tuple[PairBasis, int]:
    """Named degenerate / asymmetric same-spin fixtures."""
    if kind in ("no_alpha", "no_beta"):
        norb, nelec = 4, 2
        base = _full_product(norb, nelec)
        strings = (base.unique_betas if kind == "no_alpha" else base.unique_alphas).numpy()
        empty = np.zeros(len(strings), dtype=np.int64)
        if kind == "no_alpha":
            return pair_basis_from_arrays(empty, strings), norb
        return pair_basis_from_arrays(strings, empty), norb
    if kind == "fully_occupied":
        norb = 3
        full = (1 << norb) - 1
        return pair_basis_from_arrays([full], [full]), norb
    if kind == "asymmetric":
        return _wide_asymmetric_basis(300, 1)
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["no_alpha", "no_beta", "fully_occupied", "asymmetric"])
def test_same_spin_degenerate_and_asymmetric(kind: str) -> None:
    """Degenerate spins and a many-string × one-string basis match the generic path."""
    basis, norb = _degenerate_or_asymmetric_basis(kind)
    integrals = _real_integrals(norb)
    off = _helpers_for(basis, norb, "off")
    on = _helpers_for(basis, norb, "on")
    if on.blocks is None:
        pytest.skip("fixture did not decompose into blocks")
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(basis.n_pairs, 1, complex_values=False)
    expected = mult_pair_torch(hii, x, basis, off, norb, integrals)
    got = mult_pair_torch(hii, x, basis, on, norb, integrals)
    _assert_close(got, expected, kind)


# ============================================================================
# 7. Byte-bounded move-table cache
# ============================================================================


def _table_bytes(cache: _MoveCache) -> list[int]:
    """Resident bytes of every cached move table, in insertion order."""
    return [
        0 if moves is None else sum(int(t.numel()) * int(t.element_size()) for t in moves)
        for moves in cache.entries.values()
    ]


def _count_enumerations(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count every ``_single_moves`` / ``_double_moves`` call in the module."""
    counts = {"calls": 0}
    for name in ("_single_moves", "_double_moves"):
        real = getattr(mult_pair_blocks, name)

        def wrapper(*args: object, __real=real, **kwargs: object) -> object:
            counts["calls"] += 1
            return __real(*args, **kwargs)

        monkeypatch.setattr(mult_pair_blocks, name, wrapper)
    return counts


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("basis_name", ["full_product", "tier"])
def test_move_cache_matches_uncached(basis_name: str, complex_values: bool) -> None:
    """Cached full and owner-computes matvecs equal the uncached ones."""
    basis, norb = _make_basis(basis_name)
    integrals = _complex_integrals(norb) if complex_values else _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(basis.n_pairs, 5, complex_values=complex_values)

    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)
    cache = _MoveCache(1 << 30, evict=True)
    got = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=cache)
    _assert_close(got, expected, f"full cache {basis_name}")
    assert cache.entries
    assert 0 < cache.used_bytes <= cache.budget_bytes

    row_start, n_local = 5, 12
    row_stop = row_start + n_local
    hii_local = make_diag_terms_pair_local(
        basis.alpha_of_pair[row_start:row_stop],
        basis.beta_of_pair[row_start:row_stop],
        norb,
        integrals,
    )
    expected_local = mult_pair_local_torch(
        hii_local,
        x[row_start:row_stop],
        x,
        basis,
        helpers,
        norb,
        integrals,
        row_start,
    )
    # The same cache serves the owner-computes pass: same-spin keys are shared
    # (identical enumeration arguments) while the inverse αβ tables get their
    # own keys, so a collision would break this parity check.
    got_local = mult_pair_local_torch(
        hii_local,
        x[row_start:row_stop],
        x,
        basis,
        helpers,
        norb,
        integrals,
        row_start,
        cache=cache,
    )
    _assert_close(got_local, expected_local, f"local cache {basis_name}")
    assert 0 < cache.used_bytes <= cache.budget_bytes


@pytest.mark.parametrize("channel", ["same_spin", "ab"])
@pytest.mark.parametrize("local", [False, True])
def test_block_entry_points_move_cache(channel: str, local: bool) -> None:
    """Each full/local block entry point accepts and populates a cache."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    x = _random_vector(basis.n_pairs, 9, complex_values=False)
    row_start, n_local = 11, 20
    cache = _MoveCache(1 << 30, evict=True)

    def run(with_cache: bool) -> torch.Tensor:
        selected = cache if with_cache else None
        if channel == "same_spin":
            if local:
                out = torch.zeros(n_local, dtype=x.dtype)
                add_same_spin_blocks_local(out, x, helpers, integrals, row_start, cache=selected)
            else:
                out = torch.zeros(basis.n_pairs, dtype=x.dtype)
                add_same_spin_blocks(out, x, helpers, integrals, cache=selected)
        elif local:
            out = torch.zeros(n_local, dtype=x.dtype)
            add_ab_blocks_local(out, x, helpers, integrals, row_start, cache=selected)
        else:
            out = torch.zeros(basis.n_pairs, dtype=x.dtype)
            add_ab_blocks(out, x, helpers, integrals, cache=selected)
        return out

    plain = run(False)
    cached = run(True)
    _assert_close(cached, plain, f"{channel} local={local}")
    assert cache.entries
    assert 0 < cache.used_bytes <= cache.budget_bytes


def test_move_cache_stays_within_budget() -> None:
    """Resident bytes stay capped, with eviction and a single oversized table."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(basis.n_pairs, 13, complex_values=False)
    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)

    # Measure the tables under an ample budget, then cap below their total.
    warm = _MoveCache(1 << 30, evict=True)
    mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=warm)
    total = warm.used_bytes
    assert total > 0
    sizes = _table_bytes(warm)
    largest = max(sizes)
    assert 0 < largest < total  # several tables, so eviction can occur

    budget = largest + (total - largest) // 2
    cache = _MoveCache(budget, evict=True)
    got = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=cache)
    _assert_close(got, expected, "evicting cache")
    assert 0 < cache.used_bytes <= budget
    assert cache.used_bytes < total  # not every table was retained

    # A zero-byte cache stores nothing with a footprint and still computes.
    zero = _MoveCache(0, evict=True)
    got = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=zero)
    _assert_close(got, expected, "zero-byte cache")
    assert zero.used_bytes == 0

    # A one-byte cap admits no non-empty table (a single oversized entry).
    tiny = _MoveCache(1, evict=True)
    got = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=tiny)
    _assert_close(got, expected, "one-byte cache")
    assert tiny.used_bytes == 0
    assert all(size == 0 for size in _table_bytes(tiny))


def test_move_cache_reuses_across_matvecs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With room for every table, a second matvec re-enumerates nothing."""
    basis, norb = _make_basis("tier")
    integrals = _real_integrals(norb)
    helpers = _helpers_for(basis, norb, "on")
    assert helpers.blocks is not None
    hii = make_diag_terms_pair_torch(basis, norb, integrals)
    x = _random_vector(basis.n_pairs, 17, complex_values=False)
    counts = _count_enumerations(monkeypatch)

    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)
    enumerations = counts["calls"]
    assert enumerations > 0

    cache = _MoveCache(1 << 30, evict=True)
    counts["calls"] = 0
    first = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=cache)
    assert counts["calls"] == enumerations  # the first pass fills the cache
    _assert_close(first, expected, "first cached matvec")
    snapshot = (len(cache.entries), cache.used_bytes)

    counts["calls"] = 0
    second = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=cache)
    assert counts["calls"] == 0  # every table was a hit
    assert torch.equal(first, second)
    assert (len(cache.entries), cache.used_bytes) == snapshot


def test_move_cache_disabled_for_structural_basis() -> None:
    """A structural basis enumerates fragments and never populates the cache."""
    from test_pair_structure import _structural  # local import avoids a cycle

    basis = _structural()
    norb = 5
    integrals = _real_integrals(norb)
    helpers = generate_pair_helpers(basis, norb, block_matvec="on")
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    hii = make_diag_terms_pair_local(alpha, beta, norb, integrals)
    x = torch.randn(basis.n_pairs, dtype=torch.float64)
    expected = mult_pair_torch(hii, x, basis, helpers, norb, integrals)

    cache = _MoveCache(1 << 30, evict=True)
    got = mult_pair_torch(hii, x, basis, helpers, norb, integrals, cache=cache)
    _assert_close(got, expected, "structural")
    assert cache.used_bytes == 0
    assert not cache.entries
