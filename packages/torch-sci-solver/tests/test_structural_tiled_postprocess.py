"""Block-walked tiled post-processing for a structural pair basis.

A structural (block-described) pair basis stores no per-pair arrays, so the
tiled RDM / ⟨S²⟩ accumulators re-derive every row from the O(M_unique)
unique-string tables and resolve partner pairs with
:meth:`StructuralPairBasis.pair_index_of_uids` instead of streaming the
replicated excitation tables.

Covers:

1. gloo, 2 and 4 ranks: the tiled 1-RDM, the αα/αβ/ββ 2-RDM and ⟨S²⟩ of a
   structural tile operator equal the serial structural builders (the
   explicit-pair-list reference) to 1e-10.
2. Allocation guard on every rank: the peak tracked tensor allocation stays
   within O(chunk + M_unique + local output) and nothing of length N is
   allocated. On the 8-orbital tier basis with ``chunk_size=64`` and 2 ranks
   the measured peak is 12288 elements — the ``(3, L, L, L, L)`` 2-RDM
   accumulator — while N = 430.
3. gloo, 2 ranks: a structural GDB solve with ``compute_full_rdm=True`` and
   ``compute_spin_squared=True`` returns the same spin-summed 1-/2-RDM,
   ⟨S²⟩ and energy as the explicit-basis solve of the same determinant space.
4. gloo, 2 ranks: the ``spin_sq`` penalty runs on the structural tile and
   reproduces the explicit-basis penalized energy and final ⟨S²⟩.
5. Multi-limb structural blocks (70 and 130 spatial orbitals): the serial and
   tiled accumulators match the arbitrary-precision Python-int reference and
   the explicit limb builders, real and complex, with a slow 130-orbital dense
   2-RDM comparison and a 2-rank gloo tiled sweep.
6. Output-partitioned accumulation (forced with ``max_replicated_bytes=0``) at
   gloo 2 and 4 ranks, real and complex: the replicated results equal the
   serial builders and the packed local slice equals the same flat range of
   the serial 2-RDM, so no rank materialises the full ``3 L^4`` output.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pathlib

import numpy as np
import pytest
import torch
from test_distributed_pairs import _init_gloo, _run_spawned
from test_pair_blocks import _strings
from test_rdm_pairs_multilimb import (
    _ReferenceRDMs,
    _rows_as_ints,
    _sigma,
    _wide_strings,
)
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.distributed_pairs_tile import LocalPairTileOperator
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_local
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
    pair_tiled_one_rdm,
    pair_tiled_s2_apply,
    pair_tiled_s2_diagonal,
    pair_tiled_spin_squared,
    pair_tiled_two_rdm,
)
from torch_sci_solver.torch_impl.pair_tiles import pair_tile_bounds, pair_tile_partition
from torch_sci_solver.torch_impl.pairs import (
    StructuralPairBasis,
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)
from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm
from torch_sci_solver.torch_impl.rdm_pairs import (
    _inter_to_block_signs_pair,
    make_one_rdm_pair_torch,
    make_two_rdm_aa_pair_torch,
    make_two_rdm_ab_pair_torch,
    make_two_rdm_bb_pair_torch,
    make_two_rdm_pair_torch,
)
from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

NORB = 8
NELEC = 4
MAX_LEVEL = 2
ATOL = 1e-8
# Small enough that every accumulator slice is a handful of rows.
CHUNK_SIZE = 64


def _tier_spec() -> list[tuple[np.ndarray, np.ndarray]]:
    """The benchmark-shaped tier blocks, as structural block specifications."""
    tiers = [_strings(NORB, NELEC, level) for level in range(MAX_LEVEL + 1)]
    return [
        (tiers[level - b], tiers[b]) for level in range(MAX_LEVEL + 1) for b in range(level + 1)
    ]


def _structural() -> StructuralPairBasis:
    """The 8-orbital tier basis, described structurally."""
    return structural_pair_basis_from_blocks(_tier_spec())


def _explicit() -> object:
    """The same tier basis as an explicit, lex-sorted pair list."""
    alphas, betas = [], []
    for alpha, beta in _tier_spec():
        alphas.append(np.repeat(alpha, len(beta)))
        betas.append(np.tile(beta, len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas))


def _wide_spec(n_orb: int) -> list[tuple[list[int], list[int]]]:
    """Three disjoint pair blocks whose strings straddle the limb boundary."""
    strings = _wide_strings(n_orb)
    return [(strings[:3], strings[:3]), (strings[:3], strings[3:]), (strings[3:], strings[:3])]


def _wide_structural(n_orb: int) -> StructuralPairBasis:
    """The limb structural basis of :func:`_wide_spec`."""
    return structural_pair_basis_from_blocks(_wide_spec(n_orb))


def _wide_explicit(n_orb: int):
    """The same limb block spec as an explicit lex-sorted pair list."""
    alphas, betas = [], []
    for alpha, beta in _wide_spec(n_orb):
        alphas.append(np.repeat(np.array(alpha, dtype=object), len(beta)))
        betas.append(np.tile(np.array(beta, dtype=object), len(alpha)))
    return pair_basis_from_arrays(np.concatenate(alphas), np.concatenate(betas), n_orb=n_orb)


def _wide_perm(basis: StructuralPairBasis, explicit) -> torch.Tensor:
    """Structural pair index -> explicit pair index, by arbitrary-precision string."""
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    lookup = {
        (a, b): index
        for index, (a, b) in enumerate(
            zip(
                _rows_as_ints(explicit.alpha_of_pair),
                _rows_as_ints(explicit.beta_of_pair),
            )
        )
    }
    return torch.tensor([lookup[(a, b)] for a, b in zip(_rows_as_ints(alpha), _rows_as_ints(beta))])


def _wide_state(
    n_orb: int, complex_values: bool
) -> tuple[StructuralPairBasis, object, torch.Tensor, _ReferenceRDMs]:
    """A normalized limb structural state and its arbitrary-precision reference."""
    basis = _wide_structural(n_orb)
    helpers = generate_pair_helpers(basis, n_orb, block_matvec="on")
    generator = torch.Generator().manual_seed(17)
    real = torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
    x = real
    if complex_values:
        x = real + 1j * torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
    x = x / torch.linalg.norm(x)
    alpha, beta = basis.pair_strings(torch.arange(basis.n_pairs))
    alphas = _rows_as_ints(alpha)
    betas = _rows_as_ints(beta)
    coeffs = np.array([_sigma(a, b, n_orb) for a, b in zip(alphas, betas)]) * x.numpy()
    reference = _ReferenceRDMs(alphas, betas, coeffs, n_orb)
    return basis, helpers, x, reference


def _assert_dense_matches_sparse(
    got: torch.Tensor, table: dict, n_orb: int, atol: float = 1e-10
) -> None:
    """Compare a dense 2-RDM block with the reference's nonzero entries."""
    for key, value in table.items():
        if abs(value) > atol:
            assert abs(complex(got[key]) - complex(value)) < atol, key
    for flat in got.view(-1).nonzero().squeeze(1).tolist():
        key = (
            flat // n_orb**3,
            (flat // n_orb**2) % n_orb,
            (flat // n_orb) % n_orb,
            flat % n_orb,
        )
        assert abs(complex(got[key]) - complex(table.get(key, 0.0))) < atol, key


def _h1_eri() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 8-orbital integrals with exact 8-fold chemist symmetry."""
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
    eri = np.zeros_like(raw)
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
        eri += raw.transpose(perm)
    eri /= 8.0
    return h1, eri


def _torch_integrals():
    """The same integrals as the package's matvec bundle."""
    h1, eri = _h1_eri()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    return to_torch_integrals(I0, I1, I2)


def _tile_operator(
    rank: int,
    world_size: int,
    *,
    chunk_size: int,
) -> tuple[LocalPairTileOperator, StructuralPairBasis, object, torch.Tensor]:
    """One rank's structural tile operator and the global test vector."""
    basis = _structural()
    helpers = generate_pair_helpers(basis, NORB, block_matvec="on")
    integrals = _torch_integrals()
    partition = pair_tile_partition(helpers, world_size, grid=(world_size, 1))
    bounds = pair_tile_bounds(partition, basis, rank)
    alpha, beta = basis.pair_strings(bounds.pair_index)
    hii = make_diag_terms_pair_local(alpha, beta, NORB, integrals, chunk_size=chunk_size)
    operator = LocalPairTileOperator(
        hii,
        basis,
        helpers,
        NORB,
        integrals,
        bounds,
        partition,
        rank=rank,
        world_size=world_size,
        chunk_size=chunk_size,
    )
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
    return operator, basis, helpers, x


def _serial_reference(
    basis: StructuralPairBasis, helpers, x: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Spin-resolved serial RDMs and ⟨S²⟩ of a full structural vector."""
    gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, NORB)
    gamma2_aa, gamma2_ab, gamma2_bb = make_two_rdm_pair_torch(x, basis, helpers, NORB)
    spin_sq = s_squared_from_rdm(gamma2_ab, NELEC, NELEC)
    return gamma_a, gamma_b, gamma2_aa, gamma2_ab, gamma2_bb, spin_sq


def _assert_close(got: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Assert one observable of the structural tile equals the serial builder."""
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=0.0, msg=label)


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_wide_structural_serial_builders_match_reference(n_orb: int, complex_values: bool) -> None:
    """Block-walked limb RDMs at 70/130 match the arbitrary-precision reference."""
    basis, helpers, x, reference = _wide_state(n_orb, complex_values)
    assert basis.n_limbs == (2 if n_orb == 70 else 3)

    gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, n_orb)
    np.testing.assert_allclose(gamma_a.numpy(), reference.ga, atol=1e-10)
    np.testing.assert_allclose(gamma_b.numpy(), reference.gb, atol=1e-10)

    if n_orb == 130:
        # The dense (L, L, L, L) blocks are not materialised at this size; the
        # 1-RDM above already exercises the limb occupancy and sign paths.
        return
    blocks = make_two_rdm_pair_torch(x, basis, helpers, n_orb)
    for got, table in zip(blocks, (reference.gaa, reference.gab, reference.gbb), strict=True):
        _assert_dense_matches_sparse(got, table, n_orb)


@pytest.mark.slow
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_wide_structural_two_rdm_matches_reference_at_130(complex_values: bool) -> None:
    """At 130 orbitals each structural 2-RDM block matches the Python-int reference.

    The blocks are dense ``(130, 130, 130, 130)`` tensors, so they are built
    and compared one at a time against the reference's sparse tables.
    """
    n_orb = 130
    basis, helpers, x, reference = _wide_state(n_orb, complex_values)
    builders = (
        (make_two_rdm_aa_pair_torch, reference.gaa),
        (make_two_rdm_ab_pair_torch, reference.gab),
        (make_two_rdm_bb_pair_torch, reference.gbb),
    )
    for builder, table in builders:
        got = builder(x, basis, helpers, n_orb)
        _assert_dense_matches_sparse(got, table, n_orb)
        del got


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_wide_structural_tiled_observables_match_serial(n_orb: int, complex_values: bool) -> None:
    """A limb structural tile reproduces the serial builders and reference S²."""
    basis, helpers, x, reference = _wide_state(n_orb, complex_values)
    partition = pair_tile_partition(helpers, 1)
    bounds = pair_tile_bounds(partition, basis, 0)
    operator = LocalPairTileOperator(
        torch.zeros(bounds.local_size, dtype=torch.float64),
        basis,
        helpers,
        n_orb,
        None,
        bounds,
        partition,
        rank=0,
        world_size=1,
    )
    x_local = x[bounds.pair_index]

    gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, n_orb)
    got_a, got_b = pair_tiled_one_rdm(operator, x_local)
    _assert_close(got_a, gamma_a, "γα")
    _assert_close(got_b, gamma_b, "γβ")

    spin_sq = pair_tiled_spin_squared(operator, x_local, helpers.n_a, helpers.n_b)
    assert abs(float(spin_sq) - reference.s_squared()) < 1e-10

    if n_orb == 70:
        reference_blocks = make_two_rdm_pair_torch(x, basis, helpers, n_orb)
        got_blocks = pair_tiled_two_rdm(operator, x_local)
        for got, want, label in zip(
            got_blocks,
            reference_blocks,
            ("Γαα", "Γαβ", "Γββ"),
            strict=True,
        ):
            _assert_close(got, want, label)

    # The spin-swap operator is shared by both representations; compare the
    # tiled structural result with the explicit-limb operator through the
    # (α, β) permutation.
    explicit = _wide_explicit(n_orb)
    perm = _wide_perm(basis, explicit)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(basis.n_pairs)
    serial = PairSpinSqOperator(explicit, n_orb, helpers.n_a, helpers.n_b)
    sigma = _inter_to_block_signs_pair(explicit.alpha_of_pair, explicit.beta_of_pair, n_orb).to(
        x.dtype
    )
    serial_apply = sigma * serial.apply(x[inverse] * sigma)
    torch.testing.assert_close(
        pair_tiled_s2_apply(operator, x_local), serial_apply[perm], atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        pair_tiled_s2_diagonal(operator),
        serial.diagonal()[perm],
        atol=1e-12,
        rtol=0.0,
    )


def _postprocess_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
) -> None:
    """Every rank's tiled observables match the serial structural builders."""
    import torch.distributed as dist

    _init_gloo(rank, world_size, init_file)
    try:
        operator, basis, helpers, x = _tile_operator(rank, world_size, chunk_size=CHUNK_SIZE)
        reference = _serial_reference(basis, helpers, x)
        x_local = x[operator.bounds.pair_index]

        # Allocation guard: record the numel of every torch constructor call
        # made by the three accumulators. The bound is the largest legitimate
        # workspace: the 2-RDM output, the per-chunk occupancy/candidate rows,
        # the membership tables and this rank's local output block.
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
            gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
            gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
            spin_sq = pair_tiled_spin_squared(operator, x_local, NELEC, NELEC)
        finally:
            for name, original in originals.items():
                setattr(torch, name, original)

        _assert_close(gamma_a, reference[0], "γα")
        _assert_close(gamma_b, reference[1], "γβ")
        _assert_close(gamma2_aa, reference[2], "Γαα")
        _assert_close(gamma2_ab, reference[3], "Γαβ")
        _assert_close(gamma2_bb, reference[4], "Γββ")
        _assert_close(spin_sq, reference[5], "⟨S²⟩")
        # The agreement must not be vacuous: the tier basis has cross-block
        # singles, so a block-local-only sweep would leave γα diagonal.
        off_diagonal = gamma_a - torch.diag(torch.diagonal(gamma_a))
        assert float(off_diagonal.abs().max()) > 1e-3, "γα singles contribution is zero"

        membership_width = max(
            int(basis.alpha_block_ids.shape[1]), int(basis.beta_block_ids.shape[1])
        )
        local_size = int(operator.bounds.pair_index.numel())
        bound = max(
            CHUNK_SIZE,
            3 * NORB**4,
            2 * NORB**2,
            2 * NORB * CHUNK_SIZE,
            local_size * NORB,
            2 * local_size,
            basis.n_unique_alphas * membership_width,
            basis.n_unique_betas * membership_width,
        )
        assert tracked, "the allocation spy recorded nothing"
        assert max(tracked) <= bound, (max(tracked), bound)
        # No length-N array: each rank owns a strict subset and never builds
        # the global pair axis.
        assert local_size < basis.n_pairs
        assert basis.n_pairs not in tracked, "a length-N tensor was allocated"
    finally:
        dist.destroy_process_group()
    pathlib.Path(out_file).open("w").write("ok")


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_gloo_structural_tiled_postprocess_matches_serial(tmp_path, world_size: int) -> None:
    """gloo, 2-4 ranks: all five spin-resolved tensors and ⟨S²⟩ match at 1e-10."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _postprocess_worker,
        world_size,
        (),
        tmp_path,
        f"struct_tiled_post_{world_size}",
    )


WIDE_NORB = 70


def _wide_postprocess_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
) -> None:
    """gloo rank: limb structural tiled observables match the serial builders."""
    import torch.distributed as dist

    _init_gloo(rank, world_size, init_file)
    try:
        basis = _wide_structural(WIDE_NORB)
        helpers = generate_pair_helpers(basis, WIDE_NORB, block_matvec="on")
        partition = pair_tile_partition(helpers, world_size, grid=(world_size, 1))
        bounds = pair_tile_bounds(partition, basis, rank)
        operator = LocalPairTileOperator(
            torch.zeros(bounds.local_size, dtype=torch.float64),
            basis,
            helpers,
            WIDE_NORB,
            None,
            bounds,
            partition,
            rank=rank,
            world_size=world_size,
            chunk_size=CHUNK_SIZE,
        )
        generator = torch.Generator().manual_seed(29)
        x = torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
        x = x / torch.linalg.norm(x)
        gamma_a, gamma_b = make_one_rdm_pair_torch(x, basis, helpers, WIDE_NORB)
        gamma2_aa, gamma2_ab, gamma2_bb = make_two_rdm_pair_torch(x, basis, helpers, WIDE_NORB)
        reference = (
            gamma_a,
            gamma_b,
            gamma2_aa,
            gamma2_ab,
            gamma2_bb,
            s_squared_from_rdm(gamma2_ab, helpers.n_a, helpers.n_b),
        )
        x_local = x[bounds.pair_index]

        gamma_a, gamma_b = pair_tiled_one_rdm(operator, x_local)
        gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm(operator, x_local)
        spin_sq = pair_tiled_spin_squared(operator, x_local, helpers.n_a, helpers.n_b)

        _assert_close(gamma_a, reference[0], "γα")
        _assert_close(gamma_b, reference[1], "γβ")
        _assert_close(gamma2_aa, reference[2], "Γαα")
        _assert_close(gamma2_ab, reference[3], "Γαβ")
        _assert_close(gamma2_bb, reference[4], "Γββ")
        _assert_close(spin_sq, reference[5], "⟨S²⟩")
        assert float((gamma_a - torch.diag(torch.diagonal(gamma_a))).abs().max()) > 1e-3
    finally:
        dist.destroy_process_group()
    pathlib.Path(out_file).open("w").write("ok")


@pytest.mark.slow
def test_gloo_wide_structural_tiled_postprocess_matches_serial(tmp_path) -> None:
    """gloo, 2 ranks: the limb structural tiled post-processing matches serial."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _wide_postprocess_worker,
        2,
        (),
        tmp_path,
        "struct_wide_tiled_post",
    )


def _explicit_solver_reference(spin_sq: float | None) -> dict[str, object]:
    """Single-process explicit-basis solve of the same determinant space."""
    h1, eri = _h1_eri()
    explicit = _explicit()
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=4,
        max_davidson_iters=300,
        davidson_tolerance=1e-9,
        davidson_missed_root_check=False,
        verbose=False,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
        compute_full_rdm=True,
        compute_spin_squared=True,
    )
    result = solver(
        [(explicit.alpha_of_pair.numpy(), explicit.beta_of_pair.numpy())],
        h1,
        eri,
        NORB,
        (NELEC, NELEC),
        spin_sq=spin_sq,
    )
    state = result[0].states[0]
    reference: dict[str, object] = {
        "energy": float(state.energy),
        "rdm1": np.asarray(state.rdm1).copy(),
        "rdm2": np.asarray(state.rdm2).copy(),
        "spin_squared": float(list(solver.stats["spin_squared"])[-1]),
    }
    if spin_sq is not None:
        reference["spin_sq_final"] = float(solver.stats["spin_sq_final"])
    return reference


def _solver_worker(
    rank: int,
    world_size: int,
    expected: dict[str, object],
    spin_sq: float | None,
    init_file: str,
    out_file: str,
) -> None:
    """One rank of the structural solve; rank 0 compares with the reference."""
    import torch.distributed as dist

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri = _h1_eri()
        solver = make_torch_sci_solver(
            device="cpu",
            matvec_impl="batched",
            davidson_block_size=4,
            max_davidson_iters=300,
            davidson_tolerance=1e-9,
            davidson_missed_root_check=False,
            verbose=False,
            mode="gdb",
            gdb_kwargs={
                "block_matvec": "on",
                "structural_blocks": [_tier_spec()],
            },
            compute_full_rdm=True,
            compute_spin_squared=True,
            distributed={
                "world_size": world_size,
                "backend": "gloo",
                "timeout": 120,
                "persistent": False,
                "matvec": "local",
                "tile": "force",
            },
        )
        placeholder = [(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))]
        result = solver(placeholder, h1, eri, NORB, (NELEC, NELEC), spin_sq=spin_sq)
        if rank == 0:
            state = result[0].states[0]
            assert abs(float(state.energy) - float(expected["energy"])) < ATOL
            np.testing.assert_allclose(state.rdm1, expected["rdm1"], atol=ATOL)
            np.testing.assert_allclose(state.rdm2, expected["rdm2"], atol=ATOL)
            spin_squared = float(list(solver.stats["spin_squared"])[-1])
            assert abs(spin_squared - float(expected["spin_squared"])) < ATOL
            if spin_sq is not None:
                final = float(solver.stats["spin_sq_final"])
                assert abs(final - float(expected["spin_sq_final"])) < ATOL
    finally:
        dist.destroy_process_group()
    pathlib.Path(out_file).open("w").write("ok")


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [2, 4])
def test_gloo_structural_solver_full_rdm_and_spin_squared(tmp_path, world_size: int) -> None:
    """gloo, 2/4 ranks: full RDMs, ⟨S²⟩ and the energy match the explicit solve."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    expected = _explicit_solver_reference(None)
    _run_spawned(
        _solver_worker,
        world_size,
        (expected, None),
        tmp_path,
        f"struct_solver_full_rdm_{world_size}",
    )


@pytest.mark.slow
def test_gloo_structural_solver_spin_sq_penalty(tmp_path) -> None:
    """gloo, 2 ranks: the structural tiled penalty matches the explicit basis."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    expected = _explicit_solver_reference(1.0)
    _run_spawned(
        _solver_worker,
        2,
        (expected, 1.0),
        tmp_path,
        "struct_solver_penalty",
    )


# ============================================================================
# 6. Output-partitioned accumulation on the structural tile
# ============================================================================


def _partitioned_worker(
    rank: int,
    world_size: int,
    complex_values: bool,
    init_file: str,
    out_file: str,
) -> None:
    """Forced partitioned structural RDMs match the serial builders on each rank."""
    import torch.distributed as dist
    from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
        pair_tiled_one_rdm_distributed,
        pair_tiled_two_rdm_distributed,
    )

    _init_gloo(rank, world_size, init_file)
    try:
        operator, basis, helpers, x = _tile_operator(rank, world_size, chunk_size=CHUNK_SIZE)
        if complex_values:
            generator = torch.Generator().manual_seed(31)
            real = torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
            x = torch.complex(
                real, torch.randn(basis.n_pairs, generator=generator, dtype=torch.float64)
            )
        reference = _serial_reference(basis, helpers, x)
        x_local = x[operator.bounds.pair_index]

        gamma_a, gamma_b = pair_tiled_one_rdm_distributed(operator, x_local, max_replicated_bytes=0)
        gamma2_aa, gamma2_ab, gamma2_bb = pair_tiled_two_rdm_distributed(
            operator, x_local, max_replicated_bytes=0
        )
        _assert_close(gamma_a, reference[0], "γα")
        _assert_close(gamma_b, reference[1], "γβ")
        _assert_close(gamma2_aa, reference[2], "Γαα")
        _assert_close(gamma2_ab, reference[3], "Γαβ")
        _assert_close(gamma2_bb, reference[4], "Γββ")

        part = pair_tiled_two_rdm_distributed(
            operator, x_local, replicate=False, max_replicated_bytes=0
        )
        packed = torch.stack([reference[2], reference[3], reference[4]]).reshape(-1)
        start, stop = part.partition.start, part.partition.stop
        torch.testing.assert_close(part.data, packed[start:stop], atol=1e-10, rtol=0.0)
        if rank == 0:
            pathlib.Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("complex_values", [False, True], ids=["real", "complex"])
def test_gloo_structural_partitioned_matches_serial(
    tmp_path, world_size: int, complex_values: bool
) -> None:
    """gloo, 2/4 ranks: output-partitioned structural RDMs match the serial builders."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    _run_spawned(
        _partitioned_worker,
        world_size,
        (complex_values,),
        tmp_path,
        f"struct_part_{world_size}_{'c' if complex_values else 'r'}",
    )
