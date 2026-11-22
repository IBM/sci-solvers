"""Multi-limb strings in the distributed TPB path and the qiskit adapter.

Covers the pieces a basis wider than 64 spatial orbitals exercises:

1. ``rdm._inter_to_block_signs`` on ``(M, n_limbs)`` half-strings against a
   scalar Python-int reference (and the single-word path unchanged).
2. The string-representation guard: mixing ``(M,)`` and ``(M, n_limbs)``
   tensors is rejected with a clear error.
3. ``_reorder_for_connected_seed`` on limb matrices picks the same seed as the
   equivalent single-word reorder.
4. The owner-computes tile operator at ``n_orb=70`` with two-limb strings
   reproduces the corresponding slice of the full-space limb matvec
   (``mult_torch``) at ``1e-12``, and the scalar ``sbd_ref`` matvec likewise.
5. One gloo 2-rank run at ``n_orb=70``.
6. An adapter-level solve at ``n_orb=70``: the TPB energy matches a dense
   scalar-reference diagonalization of the same selected basis, with the
   αβ-swap projector enabled; the unprojected run matches the full spectrum's
   ground state.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import gc

import numpy as np
import pytest

# The 130-orbital cases use this module's low-rank on-demand ERI, where the
# dense (2L)^4 ERI the section-2 fixture uses no longer fits comfortably.
import test_multilimb_mult as limb_ref
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.sbd_ref.helpers import generate_all_helpers as scalar_helpers
from torch_sci_solver.sbd_ref.mult import make_diag_terms as scalar_make_diag_terms
from torch_sci_solver.sbd_ref.mult import mult as scalar_mult
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl.distributed_davidson import (
    partition_rows_explicit,
    partition_tiles,
)
from torch_sci_solver.torch_impl.helpers import (
    generate_all_torch_helpers,
    generate_local_torch_helpers,
)
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch
from torch_sci_solver.torch_impl.qiskit_adapter import (
    _ci_string_limbs,
    _reorder_for_connected_seed,
    _validate_ci_basis,
)
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

NORB = 70


# ============================================================================
# Shared fixtures: strings, limb packing, integrals, scalar oracle
# ============================================================================


def _neighbor_strings(base: int, n_orb: int, count: int, seed: int) -> list[int]:
    """Single/double-excitation neighbours of ``base``, sorted (particle number fixed)."""
    rng = np.random.default_rng(seed)
    occupied = [k for k in range(n_orb) if (base >> k) & 1]
    empty = [k for k in range(n_orb) if not (base >> k) & 1]
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


def _limb_rows(values: list[int], n_limbs: int = 2) -> np.ndarray:
    """Stack Python-int strings as an ``(M, n_limbs)`` signed-int64 matrix."""
    mask = (1 << 64) - 1
    rows = []
    for value in values:
        row = [(value >> (64 * k)) & mask for k in range(n_limbs)]
        rows.append([limb - (1 << 64) if limb >= 1 << 63 else limb for limb in row])
    return np.asarray(rows, dtype=np.int64)


def _limb_tensor(values: list[int], n_limbs: int = 2) -> torch.Tensor:
    """``_limb_rows`` as a torch int64 tensor."""
    return torch.tensor(_limb_rows(values, n_limbs), dtype=torch.int64)


class _SpatialOneInt:
    """Spin-orbital ``OneInt`` view over an RHF spatial ``h1`` (for sbd_ref)."""

    def __init__(self, h1: np.ndarray) -> None:
        """Store the spatial matrix; α and β share it."""
        self._h = np.asarray(h1, dtype=np.float64)
        self.n_spinorb = 2 * self._h.shape[0]

    def value(self, i: int, j: int) -> float:
        """``h(i, j)`` for same-spin indices, zero across spins."""
        if (i % 2) != (j % 2):
            return 0.0
        return float(self._h[i // 2, j // 2])


class _SpatialTwoInt:
    """Spin-orbital ``TwoInt`` view over a spatial chemist ERI (for sbd_ref)."""

    def __init__(self, eri: np.ndarray) -> None:
        """Store the spatial (L, L, L, L) ERI."""
        self._eri = np.asarray(eri, dtype=np.float64)
        self.n_spinorb = 2 * self._eri.shape[0]

    def value(self, p: int, q: int, r: int, s: int) -> float:
        """``(pq|rs)`` with the same spin-selection rule as ``TwoInt.value``."""
        if (p % 2) != (q % 2) or (r % 2) != (s % 2):
            return 0.0
        return float(self._eri[p // 2, q // 2, r // 2, s // 2])

    def direct_value(self, i: int, j: int) -> float:
        """``(ii|jj)``."""
        return self.value(i, i, j, j)

    def exchange_value(self, i: int, j: int) -> float:
        """``(ij|ji)``."""
        return self.value(i, j, j, i)


def _norb70_integrals() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 70-orbital spatial integrals with an 8-fold-symmetric ERI.

    A generic (rather than low-rank) ERI keeps the selected-basis spectrum
    free of the exact degeneracies a rank-1 tensor produces, which matters
    because the reference and the solver must agree on *which* state is the
    ground state. Only the values matter here: the same arrays feed the torch
    bundle and the scalar oracle.
    """
    rng = np.random.default_rng(1)
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


def _torch_system(adets: list[int], bdets: list[int], h1: np.ndarray, eri: np.ndarray) -> tuple:
    """Torch integrals, limb strings, helpers, ``hii`` and a random probe vector."""
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu")
    del I1, I2
    adets_t = _limb_tensor(adets)
    bdets_t = _limb_tensor(bdets)
    helpers = list(generate_all_torch_helpers(adets, bdets, NORB, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    x = torch.randn(
        len(adets) * len(bdets),
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float64,
    )
    return integrals, adets_t, bdets_t, helpers, hii, x


def _scalar_matvec(
    adets: list[int], bdets: list[int], x: np.ndarray, h1: np.ndarray, eri: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar-reference ``(hii, H x)`` for the same basis and integrals."""
    I1 = _SpatialOneInt(h1)
    I2 = _SpatialTwoInt(eri)
    helpers = list(scalar_helpers(adets, bdets, NORB))
    hii = scalar_make_diag_terms(adets, bdets, NORB, 0.0, I1, I2)
    return hii, scalar_mult(hii, x, adets, bdets, NORB, I1, I2, helpers)


def _scalar_sigma(adets: list[int], bdets: list[int], n_orb: int) -> np.ndarray:
    """Python-int reference for :func:`_inter_to_block_signs`."""
    sigma = np.ones((len(adets), len(bdets)), dtype=np.float64)
    for ia, alpha in enumerate(adets):
        occupied = [p for p in range(n_orb) if (alpha >> p) & 1]
        for ib, beta in enumerate(bdets):
            exponent = sum(bin(beta & ((1 << p) - 1)).count("1") for p in occupied)
            sigma[ia, ib] = 1.0 - 2.0 * (exponent & 1)
    return sigma


# ============================================================================
# 1. Convention bridge on limb matrices
# ============================================================================


def test_inter_to_block_signs_limb_matches_scalar() -> None:
    """Two-limb half-strings give the scalar ±1 matrix exactly."""
    base_a = (1 << 0) | (1 << 3) | (1 << 65) | (1 << 69)
    base_b = (1 << 1) | (1 << 4) | (1 << 64) | (1 << 68)
    adets = _neighbor_strings(base_a, NORB, 6, seed=3)
    bdets = _neighbor_strings(base_b, NORB, 5, seed=5)
    assert max(adets) >> 64 > 0 and max(bdets) >> 64 > 0

    got = _inter_to_block_signs(_limb_tensor(adets), _limb_tensor(bdets), NORB)
    expected = _scalar_sigma(adets, bdets, NORB)
    assert np.array_equal(got.numpy(), expected)

    # A (M, 1) limb matrix of a single-word basis is the same ±1 matrix.
    small = _neighbor_strings((1 << 0) | (1 << 3), 60, 5, seed=7)
    one_d = _inter_to_block_signs(
        torch.tensor(small, dtype=torch.int64), torch.tensor(small, dtype=torch.int64), 60
    )
    one_d_matrix = _inter_to_block_signs(
        torch.tensor(small, dtype=torch.int64).unsqueeze(-1),
        torch.tensor(small, dtype=torch.int64).unsqueeze(-1),
        60,
    )
    assert torch.equal(one_d, one_d_matrix)
    assert np.array_equal(one_d.numpy(), _scalar_sigma(small, small, 60))


def test_string_representation_guard() -> None:
    """Mixing a 1-D and a 2-D string tensor raises a clear error."""
    one_d = torch.zeros(3, dtype=torch.int64)
    two_d = torch.zeros((3, 2), dtype=torch.int64)
    assert dmat._validate_string_tensors(one_d, one_d) == 1
    assert dmat._validate_string_tensors(two_d, two_d) == 2
    with pytest.raises(ValueError, match="same string representation"):
        dmat._validate_string_tensors(one_d, two_d)
    with pytest.raises(TypeError, match="int64"):
        dmat._validate_string_tensors(one_d.to(torch.int32), one_d.to(torch.int32))


def test_connected_seed_reorder_limb_matches_single_word() -> None:
    """The limb reorder picks the same seed as the Python-int reference."""
    strings = _neighbor_strings((1 << 0) | (1 << 3) | (1 << 65), NORB, 7, seed=11)
    # Python-int reference: count single/double-excitation neighbours per row.
    counts = [
        sum(1 for other in strings if (value ^ other).bit_count() in (2, 4)) for value in strings
    ]
    best = int(np.argmax(counts))
    if counts[best] == 0 or best == 0:
        expected_perm = list(range(len(strings)))
    else:
        expected_perm = [best, *range(best), *range(best + 1, len(strings))]

    reordered, perm = _reorder_for_connected_seed(_limb_rows(strings))
    assert perm.tolist() == expected_perm
    assert [int(row[0]) | (int(row[1]) << 64) for row in reordered.tolist()] == [
        strings[i] for i in expected_perm
    ]


def test_alpha_row_weights_limb_matches_scalar() -> None:
    """The cost weights sum the occupation over limbs."""
    adets = _neighbor_strings((1 << 0) | (1 << 3) | (1 << 65) | (1 << 69), NORB, 4, seed=3)
    bdets = _neighbor_strings((1 << 1) | (1 << 4) | (1 << 68) | (1 << 64), NORB, 5, seed=5)
    weights = dmat.alpha_row_weights(_limb_tensor(adets), _limb_tensor(bdets), NORB)
    assert weights.shape == (len(adets),)
    assert weights.dtype == torch.float64

    beta_singles = sum(bin(b).count("1") * (NORB - bin(b).count("1")) for b in bdets)
    beta_doubles = sum(
        (bin(b).count("1") * (bin(b).count("1") - 1) / 2)
        * ((NORB - bin(b).count("1")) * (NORB - bin(b).count("1") - 1) / 2)
        for b in bdets
    )
    for ia, alpha in enumerate(adets):
        occupied = bin(alpha).count("1")
        empty = NORB - occupied
        singles = occupied * empty
        doubles = (occupied * (occupied - 1) / 2) * (empty * (empty - 1) / 2)
        expected = (
            len(bdets) * (singles + doubles) + singles * beta_singles + beta_singles + beta_doubles
        )
        assert weights[ia].item() == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_scalar_oracle_matches_package_integrals() -> None:
    """The lightweight spatial-integral view matches the package's bundle."""
    n_orb = 4
    rng = np.random.default_rng(2)
    h1 = rng.standard_normal((n_orb, n_orb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((n_orb,) * 4)
    eri = (
        sum(
            raw.transpose(p)
            for p in [
                (0, 1, 2, 3),
                (1, 0, 2, 3),
                (0, 1, 3, 2),
                (1, 0, 3, 2),
                (2, 3, 0, 1),
                (3, 2, 0, 1),
                (2, 3, 1, 0),
                (3, 2, 1, 0),
            ]
        )
        / 8.0
    )
    I1 = _SpatialOneInt(h1)
    I2 = _SpatialTwoInt(eri)
    _, pkg_one, pkg_two = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
    for i in range(2 * n_orb):
        for j in range(2 * n_orb):
            assert I1.value(i, j) == pkg_one.value(i, j)
            assert I2.direct_value(i, j) == pkg_two.direct_value(i, j)
            assert I2.exchange_value(i, j) == pkg_two.exchange_value(i, j)
            for k in range(0, 2 * n_orb, 3):
                assert I2.value(i, j, k, j) == pkg_two.value(i, j, k, j)


# ============================================================================
# 2. Single-process tile operator at n_orb=70
# ============================================================================


class _FullTileExchange:
    """Stand-in for ``StagedSlabExchange`` that slices a full vector."""

    full_vector: torch.Tensor | None = None

    def __init__(self, partition, axis, needed, *, rank=None, world_size=None, group=None, tag=0):
        """Record the partition, axis and needed indices."""
        _ = group, tag
        self.partition = partition
        self.axis = axis
        self.rank = rank
        self.world_size = world_size
        self._needed = torch.as_tensor(needed, dtype=torch.int64).reshape(-1).cpu()
        grid = partition.grid
        self._bound = partition.for_rank(rank)
        if axis == "alpha":
            self._group_pos, self._group_size = self._bound.row_slab, grid.p_alpha
            self._start, self._stop = self._bound.row_start, self._bound.row_stop
        else:
            self._group_pos, self._group_size = self._bound.col_slab, grid.p_beta
            self._start, self._stop = self._bound.col_start, self._bound.col_stop
        self.all_needed = [self._needed] * world_size

    def _peer_range(self, stage: int) -> tuple[int, int]:
        """Global index range of the stage's peer slab."""
        pos = (self._group_pos + stage) % self._group_size
        grid = self.partition.grid
        return grid.rows(pos) if self.axis == "alpha" else grid.cols(pos)

    def stage_indices(self, stage: int) -> torch.Tensor:
        """Sorted global indices the stage's buffer holds."""
        if stage == 0:
            return torch.arange(self._start, self._stop)
        lo, hi = self._peer_range(stage)
        needed = self._needed
        return needed[(needed >= lo) & (needed < hi)]

    def send_positions(self, peer_rank: int) -> torch.Tensor:
        """Positions this tile would ship (unused: the stand-in has no sends)."""
        _ = peer_rank
        return torch.empty(0, dtype=torch.int64)

    def stages(self, x_tile: torch.Tensor):
        """Yield ``(stage, block)`` for every stage."""
        full = _FullTileExchange.full_vector.view(self._bound.n_rows, self._bound.n_cols)
        rows = slice(self._bound.row_start, self._bound.row_stop)
        cols = slice(self._bound.col_start, self._bound.col_stop)
        for stage in range(self._group_size):
            if stage == 0:
                yield 0, x_tile
            elif self.axis == "alpha":
                yield stage, full[self.stage_indices(stage)][:, cols].contiguous()
            else:
                yield stage, full[rows][:, self.stage_indices(stage)].contiguous()


class _FullTileCornerExchange:
    """Stand-in for ``StagedCornerExchange`` that slices a full vector.

    Mirrors the real exchange's sub-stage plan, including the row bands a
    ``max_elems`` budget produces.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, alpha, beta, *, tag=0, max_elems=None):
        """Record the α/β stage lists and the banded sub-stage plan."""
        _ = tag
        self.partition = alpha.partition
        self.rank = alpha.rank
        self._alpha = alpha
        self._beta = beta
        grid = alpha.partition.grid
        pairs = tuple(
            (stage_a, stage_b)
            for stage_a in range(1, grid.p_alpha)
            for stage_b in range(1, grid.p_beta)
        )
        rows = {k: alpha.stage_indices(sa) for k, (sa, _) in enumerate(pairs)}
        cols = {k: beta.stage_indices(sb) for k, (_, sb) in enumerate(pairs)}
        self._plan: list[tuple[int, int, int, int]] = []
        self._rows: dict[int, torch.Tensor] = {}
        self._cols: dict[int, torch.Tensor] = {}
        for stage, (s_a, s_b) in enumerate(pairs):
            n_rows = int(rows[stage].numel())
            n_cols = int(cols[stage].numel())
            if max_elems is None or n_cols == 0 or n_rows * n_cols <= int(max_elems):
                bands = ((0, n_rows),)
            else:
                rows_per = max(1, int(max_elems) // n_cols)
                bands = tuple((lo, min(lo + rows_per, n_rows)) for lo in range(0, n_rows, rows_per))
            for row_lo, row_hi in bands:
                sub = len(self._plan)
                self._plan.append((s_a, s_b, row_lo, row_hi))
                self._rows[sub] = rows[stage][row_lo:row_hi]
                self._cols[sub] = cols[stage]

    def pairs(self) -> tuple[tuple[int, int, int, int], ...]:
        """``(s_a, s_b, row_lo, row_hi)`` of every sub-stage, in stage order."""
        return tuple(self._plan)

    def stages(self, x_tile: torch.Tensor):
        """Yield ``(stage, rectangle)`` for every remote corner sub-stage."""
        _ = x_tile
        full = _FullTileCornerExchange.full_vector.view(
            self.partition.n_rows, self.partition.n_cols
        )
        for stage in range(len(self._plan)):
            yield stage, full[self._rows[stage]][:, self._cols[stage]].contiguous()


def _install_stand_ins(monkeypatch: pytest.MonkeyPatch, x: torch.Tensor) -> None:
    """Point the operator's exchanges at the full-vector stand-ins."""
    monkeypatch.setattr(dmat, "StagedSlabExchange", _FullTileExchange)
    monkeypatch.setattr(dmat, "StagedCornerExchange", _FullTileCornerExchange)
    _FullTileExchange.full_vector = x
    _FullTileCornerExchange.full_vector = x


def _tile_slice(tensor_flat: torch.Tensor, n_rows: int, n_cols: int, bounds) -> torch.Tensor:
    """The tile's block of a flat ``(n_rows, n_cols)`` vector, in local order."""
    return tensor_flat.view(n_rows, n_cols)[
        bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop
    ].reshape(-1)


def test_tile_operator_norb70_limb_matches_full_matvec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rank's 2x2 tile reproduces the limb ``mult_torch`` block to 1e-12."""
    h1, eri = _norb70_integrals()
    adets = _neighbor_strings((1 << 0) | (1 << 3) | (1 << 65) | (1 << 69), NORB, 4, seed=3)
    bdets = _neighbor_strings((1 << 1) | (1 << 4) | (1 << 68) | (1 << 64), NORB, 4, seed=5)
    assert max(adets) >> 64 > 0 and max(bdets) >> 64 > 0
    integrals, adets_t, bdets_t, helpers, hii, x = _torch_system(adets, bdets, h1, eri)
    M_a, M_b = len(adets), len(bdets)
    expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    _, scalar = _scalar_matvec(adets, bdets, x.numpy(), h1, eri)

    tile = partition_tiles(M_a, M_b, 4, grid=(2, 2))
    _install_stand_ins(monkeypatch, x)
    for rank in range(4):
        bounds = tile.for_rank(rank)
        local = list(
            generate_local_torch_helpers(
                adets,
                bdets,
                NORB,
                bounds.row_start,
                bounds.row_stop,
                device="cpu",
                col_start=bounds.col_start,
                col_stop=bounds.col_stop,
            )
        )
        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            local,
            tile,
            rank=rank,
            world_size=4,
        )
        got = op(_tile_slice(x, M_a, M_b, bounds))
        want = _tile_slice(expected, M_a, M_b, bounds)
        assert torch.allclose(got, want, atol=1e-12, rtol=0.0), (
            rank,
            float((got - want).abs().max()) if got.numel() else 0.0,
        )
        want_scalar = _tile_slice(torch.as_tensor(scalar), M_a, M_b, bounds)
        assert torch.allclose(got, want_scalar, atol=1e-12, rtol=0.0), rank


# ============================================================================
# 2b. Low-rank limb systems at 70 and 130 orbitals, real and complex
# ============================================================================


def _limb_low_rank_system(n_orb: int, complex_integrals: bool) -> tuple:
    """Selected limb basis, on-demand integrals, full matvec and its slice oracle."""
    adets, bdets = limb_ref._selected_strings(n_orb, n_alpha=6, n_beta=5)
    assert max(adets) >> 64 > 0 and max(bdets) >> 64 > 0
    dtype = torch.complex128 if complex_integrals else torch.float64
    hcore = limb_ref._spatial_hcore(n_orb, seed=7, complex_=complex_integrals)
    eri = limb_ref._LowRankERI(
        limb_ref._random_factors(n_orb, 2, seed=9, complex_=complex_integrals),
        real=not complex_integrals,
    )
    integrals = limb_ref._torch_integrals(n_orb, hcore, eri, dtype)
    adets_t = limb_ref._limb_tensor(adets, n_orb)
    bdets_t = limb_ref._limb_tensor(bdets, n_orb)
    helpers = list(generate_all_torch_helpers(adets, bdets, n_orb, device="cpu"))
    hii = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals)

    rng = np.random.default_rng(3)
    n = len(adets) * len(bdets)
    x_np = rng.standard_normal(n)
    if complex_integrals:
        x_np = x_np + 1j * rng.standard_normal(n)
    x = torch.tensor(x_np, dtype=dtype)
    expected = mult_torch(hii, x, adets_t, bdets_t, n_orb, integrals, helpers)
    return adets, bdets, adets_t, bdets_t, integrals, hii, x, expected


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_integrals", [False, True], ids=["real", "complex"])
def test_tile_operator_limb_matches_full_matvec(
    n_orb: int, complex_integrals: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every rank's 2x2 local limb block reproduces the full matvec to 1e-12."""
    adets, bdets, adets_t, bdets_t, integrals, hii, x, expected = _limb_low_rank_system(
        n_orb, complex_integrals
    )
    M_a, M_b = len(adets), len(bdets)
    tile = partition_tiles(M_a, M_b, 4, grid=(2, 2))
    _install_stand_ins(monkeypatch, x)
    for rank in range(4):
        bounds = tile.for_rank(rank)
        local = list(
            generate_local_torch_helpers(
                adets,
                bdets,
                n_orb,
                bounds.row_start,
                bounds.row_stop,
                device="cpu",
                col_start=bounds.col_start,
                col_stop=bounds.col_stop,
            )
        )
        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            local,
            tile,
            rank=rank,
            world_size=4,
        )
        got = op(_tile_slice(x, M_a, M_b, bounds))
        want = _tile_slice(expected, M_a, M_b, bounds)
        diff = float((got - want).abs().max()) if got.numel() else 0.0
        assert diff <= 1e-12, (n_orb, complex_integrals, rank, diff)


# ============================================================================
# 3. gloo, 2 ranks, n_orb=70
# ============================================================================


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    """Initialize the gloo group used by the workers."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _norb70_worker(rank: int, world_size: int, init_file: str, out_file: str) -> None:
    """Every rank's limb operator block equals the full-space matvec's slice."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri = _norb70_integrals()
        adets = _neighbor_strings((1 << 0) | (1 << 3) | (1 << 65) | (1 << 69), NORB, 4, seed=3)
        bdets = _neighbor_strings((1 << 1) | (1 << 4) | (1 << 68) | (1 << 64), NORB, 5, seed=5)
        integrals, adets_t, bdets_t, helpers, hii, x = _torch_system(adets, bdets, h1, eri)
        M_a, M_b = len(adets), len(bdets)
        expected = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)

        partition = partition_rows_explicit(M_a, M_b, [0, 2, M_a], world_size)
        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)
        local = list(
            generate_local_torch_helpers(adets, bdets, NORB, row_start, row_stop, device="cpu")
        )
        op = dmat.LocalTPBOperator(
            hii[sl],
            adets_t,
            bdets_t,
            NORB,
            integrals,
            local,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = op(x[sl])
        diff = float((got - expected[sl]).abs().max()) if got.numel() else 0.0
        assert diff <= 1e-12, (rank, diff)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


def test_operator_gloo_two_ranks_norb70(tmp_path) -> None:
    """gloo, 2 ranks at n_orb=70 with two-limb strings, 1e-12."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    init_file = tmp_path / "norb70_init"
    out_file = tmp_path / "norb70_out"
    mp.start_processes(
        _norb70_worker,
        args=(2, str(init_file), str(out_file)),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


def _limb_gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
    n_orb: int,
    complex_integrals: bool,
) -> None:
    """Every rank's owner-computes limb block equals the full-space matvec's slice."""
    from pathlib import Path

    _init_gloo(rank, world_size, init_file)
    try:
        adets, bdets, adets_t, bdets_t, integrals, hii, x, expected = _limb_low_rank_system(
            n_orb, complex_integrals
        )
        M_a, M_b = len(adets), len(bdets)
        cut = (M_a + 1) // 2
        partition = partition_rows_explicit(M_a, M_b, [0, cut, M_a], world_size)
        row_start, row_stop = partition.rows_for_rank(rank)
        sl = slice(row_start * M_b, row_stop * M_b)
        local = list(
            generate_local_torch_helpers(adets, bdets, n_orb, row_start, row_stop, device="cpu")
        )
        op = dmat.LocalTPBOperator(
            hii[sl],
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            local,
            partition,
            rank=rank,
            world_size=world_size,
        )
        got = op(x[sl])
        diff = float((got - expected[sl]).abs().max()) if got.numel() else 0.0
        assert diff <= 1e-12, (rank, diff)
        if rank == 0:
            Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("n_orb", [70, 130])
@pytest.mark.parametrize("complex_integrals", [False, True], ids=["real", "complex"])
def test_operator_gloo_two_ranks_limb(tmp_path, n_orb: int, complex_integrals: bool) -> None:
    """gloo, 2 ranks, row-restricted limb tables match the full matvec to 1e-12."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    init_file = tmp_path / f"limb_init_{n_orb}_{int(complex_integrals)}"
    out_file = tmp_path / f"limb_out_{n_orb}_{int(complex_integrals)}"
    mp.start_processes(
        _limb_gloo_worker,
        args=(2, str(init_file), str(out_file), n_orb, complex_integrals),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


# ============================================================================
# 4. Adapter-level solve at n_orb=70
# ============================================================================


def _dense_hamiltonian(
    adets: list[int], bdets: list[int], h1: np.ndarray, eri: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense selected-basis Hamiltonian and its scalar ``hii`` and ``sigma``."""
    I1 = _SpatialOneInt(h1)
    I2 = _SpatialTwoInt(eri)
    helpers = list(scalar_helpers(adets, bdets, NORB))
    hii = scalar_make_diag_terms(adets, bdets, NORB, 0.0, I1, I2)
    n = len(adets) * len(bdets)
    dense = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        unit = np.zeros(n, dtype=np.float64)
        unit[k] = 1.0
        dense[:, k] = scalar_mult(hii, unit, adets, bdets, NORB, I1, I2, helpers)
    return dense, hii, _scalar_sigma(adets, bdets, NORB)


def _swap_projector(sigma: np.ndarray) -> np.ndarray:
    """Matrix form of ``P(v)[i, j] = (v[i, j] + τ[i, j] v[j, i]) / 2``."""
    m = sigma.shape[0]
    tau = sigma * sigma.T
    swap = np.zeros((m * m, m * m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            swap[i * m + j, j * m + i] = tau[i, j]
    return 0.5 * (np.eye(m * m) + swap)


def _spin_swap_sign_py(I_a: int, I_b: int, p: int, q: int) -> int:
    """Python-int ``⟨J|a†_pα a†_qβ a_qα a_pβ|I⟩`` sign, for any string width.

    Independent oracle for :func:`spin_sq._spin_swap_sign`, whose int64 mask
    is only correct up to 64 bits.
    """
    sign = -1
    sign *= -1 if (I_b & ((1 << p) - 1)).bit_count() & 1 else 1
    I_b = I_b & ~(1 << p)
    sign *= -1 if (I_a & ((1 << q) - 1)).bit_count() & 1 else 1
    I_a = I_a & ~(1 << q)
    sign *= -1 if (I_b & ((1 << q) - 1)).bit_count() & 1 else 1
    sign *= -1 if (I_a & ((1 << p) - 1)).bit_count() & 1 else 1
    return sign


def _dense_s2_matrix(adets: list[int], bdets: list[int], n_a: int, n_b: int) -> np.ndarray:
    """Dense S² matrix in the spin-block convention, from Python-int strings.

    Same construction as :func:`spin_sq.s_squared_matrix_torch`, but with
    Python-int popcounts so it stays valid for strings wider than 64 bits.
    """
    m_a, m_b = len(adets), len(bdets)
    sz = 0.5 * (n_a - n_b)
    sz_const = sz * sz - sz
    a_index = {value: i for i, value in enumerate(adets)}
    b_index = {value: i for i, value in enumerate(bdets)}
    matrix = np.zeros((m_a * m_b, m_a * m_b), dtype=np.float64)
    for ia, I_a in enumerate(adets):
        for ib, I_b in enumerate(bdets):
            row = ia * m_b + ib
            matrix[row, row] = n_a - (I_a & I_b).bit_count() + sz_const
            q_orb = I_a & ~I_b
            while q_orb:
                q = (q_orb & -q_orb).bit_length() - 1
                q_orb &= q_orb - 1
                p_orb = I_b & ~I_a
                while p_orb:
                    p = (p_orb & -p_orb).bit_length() - 1
                    p_orb &= p_orb - 1
                    J_a = (I_a ^ (1 << q)) | (1 << p)
                    J_b = (I_b ^ (1 << p)) | (1 << q)
                    if J_a in a_index and J_b in b_index:
                        col = a_index[J_a] * m_b + b_index[J_b]
                        matrix[col, row] += _spin_swap_sign_py(I_a, I_b, p, q)
    return 0.5 * (matrix + matrix.T)


def test_serial_spin_sq_operator_norb70_matches_dense() -> None:
    """The serial S² operator is limb-aware at n_orb=70 (dense Python-int oracle)."""
    from torch_sci_solver.torch_impl.spin_sq import SpinSqOperator

    base = (1 << 0) | (1 << 3) | (1 << 65) | (1 << 69)
    strings = _neighbor_strings(base, NORB, 6, seed=3)
    assert max(strings) >> 64 > 0
    n_a = n_b = 4
    dense = _dense_s2_matrix(strings, strings, n_a, n_b)

    op = SpinSqOperator(_limb_tensor(strings), _limb_tensor(strings), NORB, n_a, n_b)
    eye = torch.eye(len(strings) ** 2, dtype=torch.float64)
    applied = torch.stack([op.apply(eye[k]) for k in range(eye.shape[0])], dim=1)
    assert torch.allclose(applied, torch.as_tensor(dense), atol=1e-12, rtol=0.0)
    assert torch.allclose(
        op.diagonal(), torch.as_tensor(np.diag(dense).copy()), atol=1e-12, rtol=0.0
    )

    # A mixed one-word / limb-matrix call is rejected like the TPB matvec.
    with pytest.raises(ValueError, match="same string representation"):
        SpinSqOperator(
            _limb_tensor(strings),
            torch.tensor([0b11, 0b101], dtype=torch.int64),
            NORB,
            2,
            2,
        )


def test_adapter_norb70_serial_penalty_matches_tiled_and_dense() -> None:
    """Serial and ``(P, 1)`` tiled spin_sq penalties agree with a dense oracle."""
    pytest.importorskip("qiskit_addon_sqd")
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    h1, eri = _norb70_integrals()
    base = (1 << 0) | (1 << 3) | (1 << 65) | (1 << 69)
    strings = _neighbor_strings(base, NORB, 6, seed=3)
    n_a = n_b = 4
    target = 0.0
    lam = 0.3

    dense, _, _ = _dense_hamiltonian(strings, strings, h1, eri)
    # The scalar reference Hamiltonian is in the interleaved-Wick convention;
    # the block-convention S² matrix must be similarity-transformed by the
    # interleaved-to-block sign matrix before the two are added.
    sigma = _scalar_sigma(strings, strings, NORB).reshape(-1)
    dense_s2 = sigma[:, None] * _dense_s2_matrix(strings, strings, n_a, n_b) * sigma[None, :]
    _, eigvecs = np.linalg.eigh(dense + lam * (dense_s2 - target * np.eye(dense.shape[0])))
    psi = eigvecs[:, 0]
    oracle_energy = float(psi @ dense @ psi)
    oracle_s2 = float(psi @ dense_s2 @ psi)

    solver_kwargs = dict(
        device="cpu",
        davidson_block_size=8,
        max_davidson_iters=300,
        davidson_tolerance=1e-9,
        davidson_missed_root_check=False,
        apply_ab_projection=False,
        verbose=False,
    )

    def _solve(distributed=None) -> tuple[float, float, float, dict]:
        """Run the penalty solve; return augmented E, physical E, ⟨S²⟩ and the log."""
        kwargs = dict(solver_kwargs)
        if distributed is not None:
            kwargs["distributed"] = distributed
        solver = make_torch_sci_solver(**kwargs)
        try:
            result = solver(
                [(strings, strings)],
                h1,
                eri,
                NORB,
                (n_a, n_b),
                spin_sq=target,
                spin_penalty_lambda=lam,
            )[0]
            energy = float(result.states[0].energy)
            s2_value = float(solver.stats["spin_sq_final"])
            subspace = dict(solver.stats["per_call_log"][-1]["subspaces"][0])
        finally:
            solver.close()
        augmented = energy + lam * (s2_value - target)
        return augmented, energy, s2_value, subspace

    serial_augmented, serial_energy, serial_s2, _ = _solve()
    tiled_augmented, tiled_energy, tiled_s2, tiled_subspace = _solve({
        "world_size": 2,
        "backend": "gloo",
        "timeout": 180,
        "persistent": False,
    })
    # The penalty runs on the owner-computes path with the default (P, 1) grid.
    assert tiled_subspace["distributed_matvec"] == "local"
    assert tiled_subspace["tile_grid"] == [2, 1]

    # The tiled (P, 1) path is the limb-aware reference; the serial path must
    # now match it and the dense oracle.
    assert abs(serial_energy - tiled_energy) < 1e-8
    assert abs(serial_s2 - tiled_s2) < 1e-8
    assert abs(serial_augmented - tiled_augmented) < 1e-8
    assert abs(serial_energy - oracle_energy) < 1e-8, (serial_energy, oracle_energy)
    assert abs(serial_s2 - oracle_s2) < 1e-8, (serial_s2, oracle_s2)
    assert abs(tiled_energy - oracle_energy) < 1e-8
    assert abs(tiled_s2 - oracle_s2) < 1e-8


def test_adapter_norb70_matches_scalar_reference() -> None:
    """The TPB solver energy equals a dense scalar-reference diagonalization."""
    pytest.importorskip("qiskit_addon_sqd")
    from torch_sci_solver import solve_sci_batch

    h1, eri = _norb70_integrals()
    base = (1 << 0) | (1 << 3) | (1 << 65) | (1 << 69)
    strings = _neighbor_strings(base, NORB, 6, seed=3)
    assert all(bin(s).count("1") == 4 for s in strings)

    dense, _, sigma = _dense_hamiltonian(strings, strings, h1, eri)
    projected = _swap_projector(sigma)
    assert np.allclose(projected @ projected, projected, atol=1e-12)
    # The spin-independent selected-basis Hamiltonian commutes with the swap.
    assert np.allclose(dense @ projected, projected @ dense, atol=1e-10)

    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    symmetric = eigenvectors[:, eigenvalues > 0.5]
    reference = float(np.linalg.eigvalsh(symmetric.T @ dense @ symmetric)[0])
    unprojected_reference = float(np.linalg.eigvalsh(dense)[0])

    # A deterministic non-basis-vector start: exercises the limb ``_first_index``
    # mapping as well as the connected-seed reorder, both applied to the
    # caller-order vector.
    start = np.random.default_rng(5).standard_normal(len(strings) ** 2)
    result = solve_sci_batch(
        [(strings, strings)],
        h1,
        eri,
        NORB,
        (4, 4),
        device="cpu",
        apply_ab_projection=True,
        seed_strategy="connected",
        init_vectors=[start],
    )[0]
    assert np.isclose(result.energy, reference, atol=1e-8), (
        result.energy,
        reference,
    )
    # The converged state lies in the projector's +1 eigenspace.
    amplitudes_inter = result.sci_state.amplitudes.reshape(-1) * sigma.reshape(-1)
    assert np.allclose(projected @ amplitudes_inter, amplitudes_inter, atol=1e-6)
    # The state carries the full Python-int strings for a >64-orbital basis
    # (the connected seed reorders the basis, so compare as a set).
    assert result.sci_state.amplitudes.shape == (len(strings), len(strings))
    assert sorted(int(v) for v in result.sci_state.ci_strs_a) == strings
    assert max(int(v) for v in result.sci_state.ci_strs_a) >> 64 > 0

    del result
    gc.collect()

    result = solve_sci_batch(
        [(strings, strings)],
        h1,
        eri,
        NORB,
        (4, 4),
        device="cpu",
        apply_ab_projection=False,
        seed_strategy="connected",
        init_vectors=[start],
    )[0]
    assert np.isclose(result.energy, unprojected_reference, atol=1e-8), (
        result.energy,
        unprojected_reference,
    )


def test_ci_string_limbs_accepts_both_forms() -> None:
    """Python ints and an explicit (M, n_limbs) matrix normalize identically."""
    strings = _neighbor_strings((1 << 0) | (1 << 65), NORB, 4, seed=9)
    from_ints = _ci_string_limbs(strings, NORB)
    from_matrix = _ci_string_limbs(_limb_rows(strings), NORB)
    assert from_ints is not None and from_matrix is not None
    assert np.array_equal(from_ints, from_matrix)
    # A single-word basis stays on the legacy 1-D path.
    assert _ci_string_limbs([0b1011, 0b0110], 4) is None
    # A string wider than the limb budget is rejected, not truncated.
    with pytest.raises(ValueError, match="does not fit"):
        _ci_string_limbs([1 << 128], NORB)


def test_validate_ci_basis_limb_rejects_bad_strings() -> None:
    """Bit-range and particle-count checks work on limb matrices."""
    good = _neighbor_strings((1 << 0) | (1 << 65), NORB, 3, seed=9)
    # All strings have popcount 2; declaring 3 α electrons must fail.
    with pytest.raises(ValueError, match="popcount"):
        _validate_ci_basis(_limb_rows(good), _limb_rows(good), NORB, (3, 2))
    # A bit above orbital 69 lands in the top limb and must be rejected.
    too_high = _limb_rows([(1 << 0) | (1 << 69) | (1 << 70)])
    with pytest.raises(ValueError, match="above orbital 69"):
        _validate_ci_basis(too_high, _limb_rows(good[:1]), NORB, (2, 2))
    # Mixing the two representations is a clear error.
    with pytest.raises(ValueError, match="same representation"):
        _validate_ci_basis(
            _limb_rows(good), np.asarray([0b11, 0b101], dtype=np.int64), NORB, (2, 2)
        )


def test_gdb_pair_mode_accepts_wide_strings() -> None:
    """A wide pair basis carries limb strings through the pair plumbing.

    The explicit GDB path no longer rejects strings above 64 bits: the pair
    basis stores ``(N, n_limbs)`` limb matrices and the αβ-swap lookup keys on
    the uid axes, so a swap-closed product pair set resolves on the limb path.
    """
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
    from torch_sci_solver.torch_impl.qiskit_adapter import _pair_ab_swap_permutation

    strings = [1 << 0, 1 << 64]
    basis = pair_basis_from_arrays(
        [a for a in strings for _ in strings],
        [b for _ in strings for b in strings],
        n_orb=NORB,
    )
    assert basis.n_pairs == 4
    assert basis.n_limbs == 2
    k_swap, tau = _pair_ab_swap_permutation(basis, NORB, torch.device("cpu"), torch.float64)
    assert k_swap is not None  # the product pair set is closed under the swap
    assert torch.equal(k_swap[k_swap], torch.arange(4))
    # τ[k] = σ[k]·σ[k_swap[k]] is a ±1 sign, symmetric under the swap.
    assert torch.all(tau.abs() == 1)
    assert torch.equal(tau, tau[k_swap])

    # A diagonal pair set maps to itself; an asymmetric set has no partner.
    diagonal = pair_basis_from_arrays(strings, strings, n_orb=NORB)
    k_swap_d, _tau_d = _pair_ab_swap_permutation(diagonal, NORB, torch.device("cpu"), torch.float64)
    assert k_swap_d is not None and torch.equal(k_swap_d, torch.arange(2))
    asymmetric = pair_basis_from_arrays(
        [strings[0], strings[0]], [strings[0], strings[1]], n_orb=NORB
    )
    k_swap_a, tau_a = _pair_ab_swap_permutation(
        asymmetric, NORB, torch.device("cpu"), torch.float64
    )
    assert k_swap_a is None and tau_a is None


def test_smallest_int_pair_seed_is_integer_order_on_limbs() -> None:
    """The ``smallest_int`` seed is the integer-smallest pair, not limb row 0.

    The limb basis sorts limb 0 (the low word) first, so ``(1<<64)|(1<<65)``
    (low word 0) lands at index 0 although it is the largest integer here.
    """
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
    from torch_sci_solver.torch_impl.qiskit_adapter import _smallest_int_pair_index

    strings = [0b11, 0b101, (1 << 0) | (1 << 64), (1 << 64) | (1 << 65)]
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]
    basis = pair_basis_from_arrays(pairs_a, pairs_b, n_orb=70)
    assert basis.n_limbs == 2
    alpha = [int(r[0]) | (int(r[1]) << 64) for r in basis.alpha_of_pair.tolist()]
    beta = [int(r[0]) | (int(r[1]) << 64) for r in basis.beta_of_pair.tolist()]
    assert (alpha[0], beta[0]) != (0b11, 0b11)  # the limb order is not integer order
    index = _smallest_int_pair_index(basis)
    assert (alpha[index], beta[index]) == min(zip(alpha, beta)) == (0b11, 0b11)
    single = pair_basis_from_arrays([3, 5], [3, 3])
    assert _smallest_int_pair_index(single) == 0


def test_gdb_wide_cold_seed_reaches_the_coupled_ground_state() -> None:
    """At a loose tolerance a wide GDB cold start still finds the ground state.

    Orbitals 64/65 carry only a one-body energy, so the pair made of them is an
    exact eigenvector of its own sector. Seeded there (limb row 0), Davidson
    converged at once to that sector's energy; the integer-order seed starts in
    the coupled low-orbital sector.
    """
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    n_orb = 66
    rng = np.random.default_rng(4)
    h1_low = rng.standard_normal((4, 4))
    h1 = np.zeros((n_orb, n_orb))
    h1[:4, :4] = 0.5 * (h1_low + h1_low.T)
    h1[64, 64] = h1[65, 65] = 3.0
    raw = rng.standard_normal((4,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = np.zeros((n_orb,) * 4)
    eri[:4, :4, :4, :4] = sum(raw.transpose(p) for p in perms) / 8.0
    low = [0b11, 0b101, 0b110, 0b1001, 0b1010, 0b1100]
    strings = low + [(1 << 0) | (1 << 64), (1 << 64) | (1 << 65)]
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]
    energies = []
    for tol in (1e-5, 1e-10):
        solver = make_torch_sci_solver(
            mode="gdb",
            device="cpu",
            davidson_tolerance=tol,
            apply_ab_projection=False,
            max_davidson_iters=100,
        )
        state = solver([(pairs_a, pairs_b)], h1, eri, n_orb, (2, 2))[0].states[0]
        energies.append(float(state.energy))
    assert energies[0] == pytest.approx(energies[1], abs=1e-6)
    assert energies[1] < 12.0 - 1.0


@pytest.mark.slow
def test_gdb_wide_solve_matches_product_mode() -> None:
    """A cold explicit GDB solve above 64 orbitals matches the TPB product solve.

    The dense spin-orbital ERI is ``(2 * 65)^4`` fp64, so this runs only under
    ``-m slow``; the solve itself is four pairs. ``make_torch_sci_solver``
    keeps the compact spatial ERI store for the pair path.
    """
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    n_orb = 65
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((n_orb, n_orb))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.standard_normal((n_orb, n_orb, n_orb, n_orb))
    eri = 0.5 * (g + g.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    del g

    strings = [1 << 0, 1 << 64]
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]
    gdb = make_torch_sci_solver(mode="gdb", device="cpu", n_roots=1)
    e_gdb = float(gdb([(pairs_a, pairs_b)], h1, eri, n_orb, (1, 1))[0].states[0].energy)
    tpb = make_torch_sci_solver(mode="tpb", device="cpu", n_roots=1)
    e_tpb = float(tpb([(strings, strings)], h1, eri, n_orb, (1, 1))[0].states[0].energy)
    assert e_gdb == pytest.approx(e_tpb, abs=1e-8)

    diag = make_torch_sci_solver(
        mode="gdb",
        device="cpu",
        n_roots=1,
        compute_full_rdm=True,
        compute_spin_squared=True,
    )
    result = diag([(pairs_a, pairs_b)], h1, eri, n_orb, (1, 1))[0]
    rdm1 = np.asarray(result.states[0].rdm1)
    rdm2 = np.asarray(result.states[0].rdm2)
    assert np.trace(rdm1) == pytest.approx(2.0, abs=1e-10)
    assert rdm2.shape == (n_orb,) * 4
    assert float(diag.stats["spin_squared"][0]) == pytest.approx(0.0, abs=1e-8)


_WIDE_GDB_NORB = 70
# Seven one-electron strings straddling the 64-bit boundary: the 7x7 product
# pair set is large enough (> one Davidson block) to take the owner-computes
# GDB path, which the tiled diagnostics require.
_WIDE_GDB_STRINGS = [1 << 0, 1 << 1, 1 << 2, 1 << 64, 1 << 65, 1 << 66, 1 << 69]


def _wide_gdb_system() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Deterministic 65-orbital integrals and the wide product pair set."""
    n_orb = _WIDE_GDB_NORB
    rng = np.random.default_rng(1)
    h1 = rng.standard_normal((n_orb, n_orb))
    h1 = 0.5 * (h1 + h1.T)
    g = rng.standard_normal((n_orb, n_orb, n_orb, n_orb))
    eri = 0.5 * (g + g.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    pairs_a = [a for a in _WIDE_GDB_STRINGS for _ in _WIDE_GDB_STRINGS]
    pairs_b = [b for _ in _WIDE_GDB_STRINGS for b in _WIDE_GDB_STRINGS]
    return h1, eri, pairs_a, pairs_b


def _wide_gdb_tiled_worker(
    rank: int,
    world_size: int,
    expected: dict,
    init_file: str,
    out_file: str,
) -> None:
    """One rank of the tiled wide GDB diagnostics solve; rank 0 compares."""
    from pathlib import Path

    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri, pairs_a, pairs_b = _wide_gdb_system()
        solver = make_torch_sci_solver(
            mode="gdb",
            device="cpu",
            n_roots=1,
            compute_full_rdm=True,
            compute_spin_squared=True,
            davidson_missed_root_check=False,
            gdb_kwargs={"block_matvec": "on"},
            distributed={
                "world_size": world_size,
                "backend": "gloo",
                "timeout": 120,
                "persistent": False,
                "matvec": "local",
                "tile": "force",
            },
        )
        result = solver([(pairs_a, pairs_b)], h1, eri, _WIDE_GDB_NORB, (1, 1))
        if rank == 0:
            state = result[0].states[0]
            assert float(state.energy) == pytest.approx(expected["energy"], abs=1e-8)
            np.testing.assert_allclose(state.rdm1, expected["rdm1"], atol=1e-8)
            np.testing.assert_allclose(state.rdm2, expected["rdm2"], atol=1e-8)
            spin_squared = float(list(solver.stats["spin_squared"])[-1])
            assert spin_squared == pytest.approx(expected["spin_squared"], abs=1e-8)
            assert solver.stats["per_call_log"][-1]["subspaces"][0]["distributed_matvec"] == "local"
        Path(out_file).write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_gloo_wide_gdb_tiled_diagnostics_match_the_serial_solve(tmp_path) -> None:
    """gloo, 2 ranks at n_orb=65: tiled RDMs and <S^2> match the serial solve.

    The pair basis carries ``(M, 2)`` limb strings, so this exercises the limb
    branches of the tile partition, the tiled operator and the block-walked
    post-processing end to end.
    """
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    h1, eri, pairs_a, pairs_b = _wide_gdb_system()
    serial = make_torch_sci_solver(
        mode="gdb",
        device="cpu",
        n_roots=1,
        compute_full_rdm=True,
        compute_spin_squared=True,
        davidson_missed_root_check=False,
        gdb_kwargs={"block_matvec": "on"},
    )
    reference = serial([(pairs_a, pairs_b)], h1, eri, _WIDE_GDB_NORB, (1, 1))[0].states[0]
    expected = {
        "energy": float(reference.energy),
        "rdm1": np.asarray(reference.rdm1).copy(),
        "rdm2": np.asarray(reference.rdm2).copy(),
        "spin_squared": float(list(serial.stats["spin_squared"])[-1]),
    }
    init_file = tmp_path / "wide_gdb_init"
    out_file = tmp_path / "wide_gdb_out"
    mp.start_processes(
        _wide_gdb_tiled_worker,
        args=(2, expected, str(init_file), str(out_file)),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"


def test_triton_string_width_resolution() -> None:
    """The Triton launchers infer a 2-D width and reject malformed pairs."""
    from torch_sci_solver.torch_impl.mult_triton import _string_n_limbs

    one_d = torch.zeros(3, dtype=torch.int64)
    two_d = torch.zeros((3, 2), dtype=torch.int64)
    assert _string_n_limbs(one_d, one_d) == 1
    assert _string_n_limbs(two_d, two_d) == 2
    assert _string_n_limbs(two_d, two_d, n_limbs=2) == 2
    with pytest.raises(ValueError, match="n_limbs"):
        _string_n_limbs(two_d, two_d, n_limbs=3)
    with pytest.raises(ValueError, match="share one representation"):
        _string_n_limbs(one_d, two_d)


# ============================================================================
# 5. Wide explicit-pair warm / probe / init_vectors at 70 orbitals
# ============================================================================


def _wide_pair_basis(strings: list[int]):
    """The full ``len(strings) x len(strings)`` product pair basis at 70 orbitals."""
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays

    alpha = [a for a in strings for _ in strings]
    beta = [b for _ in strings for b in strings]
    return pair_basis_from_arrays(
        _limb_rows(alpha), _limb_rows(beta), n_orb=_WIDE_GDB_NORB, device="cpu"
    )


def test_wide_pair_keys_embed_and_init_mapping_are_limb_aware() -> None:
    """The warm/init key plumbing resolves limb rows like single words.

    Exercises ``_pair_uid_keys``, ``_dedup_and_check_pair_superset``,
    ``_embed_eigvec_pair_into_extended`` and the ``_first_index`` key matrix on
    a 70-orbital (two-limb) explicit pair basis.
    """
    from torch_sci_solver.torch_impl.probe_pairs import (
        _dedup_and_check_pair_superset,
        _embed_eigvec_pair_into_extended,
        _pair_uid_keys,
    )
    from torch_sci_solver.torch_impl.qiskit_adapter import _first_index, _pair_key_tensor

    basis = _wide_pair_basis(_WIDE_GDB_STRINGS)
    core_strings = _WIDE_GDB_STRINGS[:5]
    core = _wide_pair_basis(core_strings)
    n_beta = basis.n_unique_betas
    ext_keys = basis.alpha_uid_of_pair * n_beta + basis.beta_uid_of_pair

    # Every string form resolves to the basis' own (α_uid, β_uid) key.
    keys = _pair_uid_keys(
        basis,
        basis.alpha_of_pair.cpu().numpy(),
        basis.beta_of_pair.cpu().numpy(),
        device=torch.device("cpu"),
    )
    assert torch.equal(keys, ext_keys)
    whole_alpha = [
        int(row[0]) | (int(row[1]) << 64) for row in basis.alpha_of_pair.cpu().numpy().tolist()
    ]
    whole_beta = [
        int(row[0]) | (int(row[1]) << 64) for row in basis.beta_of_pair.cpu().numpy().tolist()
    ]
    assert torch.equal(
        _pair_uid_keys(basis, whole_alpha, whole_beta, device=torch.device("cpu")), ext_keys
    )

    # Union/dedup: a core subset extends to the full basis, in canonical order.
    union_alpha, union_beta, info = _dedup_and_check_pair_superset(
        core.alpha_of_pair.cpu().numpy().tolist(),
        core.beta_of_pair.cpu().numpy().tolist(),
        basis.alpha_of_pair.cpu().numpy(),
        basis.beta_of_pair.cpu().numpy(),
    )
    assert np.asarray(union_alpha).shape == (basis.n_pairs, 2)
    assert info["pairs_added"] == basis.n_pairs - core.n_pairs
    assert info["extended_N"] == basis.n_pairs
    with pytest.raises(ValueError, match="superset"):
        _dedup_and_check_pair_superset(
            basis.alpha_of_pair.cpu().numpy().tolist(),
            basis.beta_of_pair.cpu().numpy().tolist(),
            core.alpha_of_pair.cpu().numpy(),
            core.beta_of_pair.cpu().numpy(),
        )

    # Embedding places each cached amplitude at its extended pair position.
    eigvec = torch.arange(1, basis.n_pairs + 1, dtype=torch.float64)
    eigvec = eigvec / eigvec.norm()
    core_keys = _pair_uid_keys(
        basis,
        core.alpha_of_pair.cpu().numpy(),
        core.beta_of_pair.cpu().numpy(),
        device=torch.device("cpu"),
    )
    positions = torch.searchsorted(ext_keys, core_keys)
    assert torch.equal(ext_keys[positions], core_keys)
    cached = eigvec[positions]
    embedded = _embed_eigvec_pair_into_extended(
        cached,
        core.alpha_of_pair.cpu().numpy().tolist(),
        core.beta_of_pair.cpu().numpy().tolist(),
        basis.alpha_of_pair.cpu().numpy().tolist(),
        basis.beta_of_pair.cpu().numpy().tolist(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    expected = torch.zeros(basis.n_pairs, dtype=torch.float64)
    expected[positions] = cached
    expected = expected / expected.norm()
    torch.testing.assert_close(embedded, expected, atol=1e-14, rtol=0.0)

    # The init-vector key matrix maps a reversed canonical caller order back
    # onto the internal basis.
    caller_rows = _pair_key_tensor(
        basis.alpha_of_pair.cpu().numpy()[::-1],
        basis.beta_of_pair.cpu().numpy()[::-1],
        width=basis.n_limbs,
        device=torch.device("cpu"),
    )
    internal_rows = _pair_key_tensor(
        basis.alpha_of_pair,
        basis.beta_of_pair,
        width=basis.n_limbs,
        device=torch.device("cpu"),
    )
    k = _first_index(caller_rows, internal_rows, device=torch.device("cpu"))
    assert bool((k >= 0).all())
    assert torch.equal(caller_rows[k], internal_rows)
    # A caller vector laid out in the reversed order maps to the internal one.
    internal_vec = torch.arange(basis.n_pairs, dtype=torch.float64)
    caller_vec = internal_vec.flip(0)
    assert torch.equal(caller_vec[k], internal_vec)


def _wide_pair_complex_integrals() -> tuple[np.ndarray, np.ndarray]:
    """A 70-orbital complex Hermitian ``(h1, eri)`` set with both ERI symmetries."""
    rng = np.random.default_rng(3)
    n_orb = _WIDE_GDB_NORB
    hc = rng.standard_normal((n_orb, n_orb)) + 1j * rng.standard_normal((n_orb, n_orb))
    h1 = hc + hc.conj().T
    raw = rng.standard_normal((n_orb,) * 4) + 1j * rng.standard_normal((n_orb,) * 4)
    eri = raw + np.transpose(raw, (1, 0, 3, 2)).conj()
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return h1, eri


@pytest.mark.slow
@pytest.mark.parametrize("complex_integrals", [False, True], ids=["real", "complex"])
def test_wide_pair_warm_start_and_init_vectors(complex_integrals: bool) -> None:
    """70-orbital pair warm-extend and ``init_vectors`` match the cold energies.

    A cold-with-probe solve caches a 5x5 product pair state; warm-extending it
    to the 7x7 product basis must reproduce a cold solve on that basis. A
    non-superset extension raises, and an ``init_vectors`` entry laid out in a
    reversed caller order (limb rows) seeds the exact cold eigenvector.
    """
    pytest.importorskip("qiskit_addon_sqd")
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    h1, eri, pairs_a, pairs_b = _wide_gdb_system()
    h1 = np.asarray(h1)
    eri = np.asarray(eri)
    if complex_integrals:
        h1, eri = _wide_pair_complex_integrals()
    a_rows = _limb_rows(pairs_a)
    b_rows = _limb_rows(pairs_b)
    core_strings = set(_WIDE_GDB_STRINGS[:5])
    core_mask = np.array([
        a in core_strings and b in core_strings for a, b in zip(pairs_a, pairs_b)
    ])
    core_a = a_rows[core_mask]
    core_b = b_rows[core_mask]
    assert core_a.shape[0] == 25

    solver = make_torch_sci_solver(
        mode="gdb",
        device="cpu",
        dtype=torch.complex128 if complex_integrals else torch.float64,
        n_roots=1,
        davidson_block_size=8,
        max_davidson_iters=200,
        davidson_tolerance=1e-9,
        davidson_missed_root_check=False,
        apply_ab_projection=False,
        verbose=False,
    )
    try:
        # (1) Cold solve on the 5x5 core with a cached probe state.
        core_results, core_states = solver(
            [(core_a, core_b)],
            h1,
            eri,
            _WIDE_GDB_NORB,
            (1, 1),
            probe_method="warm_start",
        )
        core_state = core_states[0]
        assert core_state is not None
        assert np.asarray(core_state.alpha_of_pair).shape == (25, 2)
        e_core = float(core_results[0].states[0].energy)

        # (2) Warm-extend to the full 7x7 basis.
        warm_result = solver(
            [(a_rows, b_rows)],
            h1,
            eri,
            _WIDE_GDB_NORB,
            (1, 1),
            probe_method="warm_start",
            probe_state=[core_state],
        )[0]
        e_warm = float(warm_result.states[0].energy)
        warm_subspace = solver.stats["per_call_log"][-1]["subspaces"][0]
        assert warm_subspace["warm"] is True

        # (3) Cold solve on the same extended basis, for reference. The solver
        # reuses the integrals bundle, so this costs one Davidson solve.
        cold_result = solver([(a_rows, b_rows)], h1, eri, _WIDE_GDB_NORB, (1, 1))[0]
        cold_sci_result = cold_result.states[0]
        cold_state = cold_sci_result.sci_state
        e_cold = float(cold_sci_result.energy)
        assert e_warm == pytest.approx(e_cold, abs=1e-8)
        assert e_core >= e_cold - 1e-9  # variational bound

        # (4) A non-superset extension must raise before solving.
        dropped = np.ones(a_rows.shape[0], dtype=bool)
        dropped[int(np.flatnonzero(core_mask)[0])] = False
        with pytest.raises(ValueError, match="superset"):
            solver(
                [(a_rows[dropped], b_rows[dropped])],
                h1,
                eri,
                _WIDE_GDB_NORB,
                (1, 1),
                probe_method="warm_start",
                probe_state=[core_state],
            )

        # (5) init_vectors: the exact cold eigenvector, laid out in a reversed
        # canonical caller order, maps back onto the internal basis and
        # converges at once.
        amplitudes = np.asarray(cold_state.amplitudes)
        canon_a = np.asarray(cold_state.ci_strs_a)
        canon_b = np.asarray(cold_state.ci_strs_b)
        init_result = solver(
            [(np.ascontiguousarray(canon_a[::-1]), np.ascontiguousarray(canon_b[::-1]))],
            h1,
            eri,
            _WIDE_GDB_NORB,
            (1, 1),
            init_vectors=[np.ascontiguousarray(amplitudes[::-1])],
        )[0]
        e_init = float(init_result.states[0].energy)
        init_subspace = solver.stats["per_call_log"][-1]["subspaces"][0]
        assert init_subspace["warm"] is True
        assert init_subspace["n_matvecs"] is not None and init_subspace["n_matvecs"] <= 3
        assert e_init == pytest.approx(e_cold, abs=1e-8)
    finally:
        solver.close()


def _wide_pair_warm_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_file: str,
) -> None:
    """One rank of a wide GDB tile warm-extend; rank 0 compares to a cold solve."""
    from pathlib import Path

    from torch_sci_solver.torch_impl.distributed_davidson import DistributedContext
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    _init_gloo(rank, world_size, init_file)
    try:
        h1, eri, pairs_a, pairs_b = _wide_gdb_system()
        a_rows = _limb_rows(pairs_a)
        b_rows = _limb_rows(pairs_b)
        core_strings = set(_WIDE_GDB_STRINGS[:5])
        core_mask = np.array([
            a in core_strings and b in core_strings for a, b in zip(pairs_a, pairs_b)
        ])
        solver = make_torch_sci_solver(
            mode="gdb",
            device="cpu",
            n_roots=1,
            davidson_block_size=8,
            max_davidson_iters=200,
            davidson_tolerance=1e-9,
            davidson_missed_root_check=False,
            apply_ab_projection=False,
            gdb_kwargs={"block_matvec": "on"},
            verbose=False,
        )
        solver._distributed_ctx = DistributedContext(
            rank=rank,
            world_size=world_size,
            backend="gloo",
            device=torch.device("cpu"),
            matvec="local",
            tile="force",
        )
        solver._inside_distributed = True
        core_state = solver(
            [(a_rows[core_mask], b_rows[core_mask])],
            h1,
            eri,
            _WIDE_GDB_NORB,
            (1, 1),
            probe_method="warm_start",
        )[1][0]
        # Only rank 0 builds the cached state; in a launched run the parent
        # passes it back to every worker on the next call.
        broadcast = [core_state]
        dist.broadcast_object_list(broadcast, src=0)
        core_state = broadcast[0]
        assert core_state is not None
        warm_result = solver(
            [(a_rows, b_rows)],
            h1,
            eri,
            _WIDE_GDB_NORB,
            (1, 1),
            probe_method="warm_start",
            probe_state=[core_state],
        )[0]
        warm_perf = solver.stats["per_call_log"][-1]["subspaces"][0]
        cold_result = solver([(a_rows, b_rows)], h1, eri, _WIDE_GDB_NORB, (1, 1))[0]
        assert warm_perf["warm"] is True
        assert warm_perf["distributed_matvec"] == "local"
        if rank == 0:
            e_warm = float(warm_result.states[0].energy)
            e_cold = float(cold_result.states[0].energy)
            assert e_warm == pytest.approx(e_cold, abs=1e-8)
            Path(out_file).write_text("ok")
        else:
            assert warm_result.states == []
        solver.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
def test_gloo_wide_gdb_tiled_warm_state_matches_cold(tmp_path) -> None:
    """gloo, 2 ranks at n_orb=70: a tiled pair warm-extend matches the cold energy.

    The cached state stores one limb row per pair, so this exercises the
    owner-computes per-rank embed through the ``(α_uid, β_uid)`` pair key.
    """
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    init_file = tmp_path / "wide_warm_init"
    out_file = tmp_path / "wide_warm_out"
    mp.start_processes(
        _wide_pair_warm_worker,
        args=(2, str(init_file), str(out_file)),
        nprocs=2,
        start_method="spawn",
        join=True,
    )
    assert out_file.read_text() == "ok"
