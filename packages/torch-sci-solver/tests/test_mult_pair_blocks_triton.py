"""GPU parity tests for the Triton same-spin doubles block kernel.

Every test compares the explicit Triton request against the default batched
product kernels on the same device and dtype, so the check is a dispatch and
kernel-arithmetic parity, not a floating-point reference comparison:

1. 1-D full block layout (``add_same_spin_blocks``) on materialised and
   structural bases: the forward enumeration path whose move table is grouped
   by source, so the Triton launcher has to stable-sort it.
2. 1-D owner-computes layout (``add_same_spin_blocks_local``): the inverse
   enumeration path on several owned row ranges.
3. Fragment layout (``add_same_spin_fragment``): α and β channels on a
   rectangle of a group's grid.
4. Tile layout (``LocalPairTileOperator``): a 2x2 grid driven by a stand-in
   exchange, and an uneven 4x3 grid with empty tiles.
5. Complex dtypes: the capability check admits them, and the 1-D,
   owner-computes, slab/fragment and tile entry points reproduce the batched
   same-spin doubles path on complex integrals.

The module is skipped unless Triton is importable and a CUDA device is
present; fp64 parity uses ``atol=1e-12``, fp32 and complex64 ``atol=1e-5``.
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

pytest.importorskip("triton")

import torch_sci_solver.torch_impl.distributed_pairs_tile as dpt  # noqa: E402
from test_pair_blocks import _full_product, _strings, _tier_blocks  # noqa: E402
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri  # noqa: E402
from torch_sci_solver.torch_impl.distributed_davidson import partition_tiles  # noqa: E402
from torch_sci_solver.torch_impl.integrals import (  # noqa: E402
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult_pair_blocks import (  # noqa: E402
    add_same_spin_blocks,
    add_same_spin_blocks_local,
    add_same_spin_fragment,
)
from torch_sci_solver.torch_impl.mult_pair_blocks_triton import (  # noqa: E402
    same_spin_doubles_available,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers  # noqa: E402
from torch_sci_solver.torch_impl.pair_tiles import (  # noqa: E402
    PairFragment,
    pair_tile_bounds,
)
from torch_sci_solver.torch_impl.pairs import (  # noqa: E402
    pair_basis_from_arrays,
    structural_pair_basis_from_blocks,
)

pytestmark = pytest.mark.gpu

_ATOL = {
    torch.float64: 1e-12,
    torch.float32: 1e-5,
    torch.complex128: 1e-12,
    torch.complex64: 1e-5,
}
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


def _cuda() -> torch.device:
    """Return the CUDA device, skipping when none is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _integrals(norb: int, device: torch.device, dtype: torch.dtype, seed: int = 0):
    """Deterministic real integrals with exact 8-fold chemist symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    eri = np.empty_like(raw)
    for idx in itertools.product(range(norb), repeat=4):
        eri[idx] = raw[min(tuple(idx[p] for p in perm) for perm in _PERMUTATIONS)]
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)


def _complex_integrals(norb: int, device: torch.device, dtype: torch.dtype):
    """Deterministic complex Hermitian integrals on one device.

    The ERI is projected onto the two symmetries a complex chemist-order ERI
    carries, ``(pq|rs) = (rs|pq)`` and ``(pq|rs) = conj((qp|sr))``. A
    density-fitting sum is real after that projection, so it would not
    exercise the complex arithmetic in the same-spin block.
    """
    rng = np.random.default_rng(1)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    g = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    eri = 0.5 * (g + g.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)


def _basis(name: str, device: torch.device):
    """One named block-decomposed basis on ``device`` and its orbital count."""
    if name == "full_product":
        cpu, norb = _full_product(4, 2), 4
    elif name == "tier":
        cpu, norb = _tier_blocks(6, 3, max_level=2), 6
    elif name == "structural":
        norb, nelec, max_level = 6, 3, 2
        tiers = [_strings(norb, nelec, level) for level in range(max_level + 1)]
        spec = [
            (tiers[level - b], tiers[b]) for level in range(max_level + 1) for b in range(level + 1)
        ]
        return structural_pair_basis_from_blocks(spec, device=device), norb
    else:
        raise ValueError(name)
    return (
        pair_basis_from_arrays(cpu.alpha_of_pair.numpy(), cpu.beta_of_pair.numpy(), device=device),
        norb,
    )


def _helpers(basis, norb: int, device: torch.device):
    """Block-decomposed helpers on ``device``."""
    helpers = generate_pair_helpers(basis, norb, device=device, block_matvec="on")
    assert helpers.blocks is not None
    return helpers


def _randn(n: int, dtype: torch.dtype, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Deterministic random vector on ``device``."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=gen, dtype=dtype).to(device)


def _randc(n: int, dtype: torch.dtype, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Deterministic random complex vector on ``device``."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return (real + 1j * imag).to(dtype).to(device)


def _integrals_for(norb: int, device: torch.device, dtype: torch.dtype):
    """Real or complex integrals matching ``dtype``."""
    if dtype.is_complex:
        return _complex_integrals(norb, device, dtype)
    return _integrals(norb, device, dtype)


def _rand_vector(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Deterministic vector of ``dtype`` on ``device``."""
    if dtype.is_complex:
        return _randc(n, dtype, device)
    return _randn(n, dtype, device)


def _assert_parity(
    got: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, label: str
) -> None:
    """Assert Triton/batched agreement at the dtype's parity tolerance."""
    if got.numel() == 0 and expected.numel() == 0:
        return
    diff = float((got - expected).abs().max())
    assert torch.allclose(got, expected, atol=_ATOL[dtype], rtol=0.0), (label, diff)


class _FullVectorExchange:
    """Stand-in exchange that slices a full device vector by stage lists."""

    full: torch.Tensor | None = None

    def __init__(self, local_index, needed, *, rank, world_size, group=None, tag=0):
        """Record the plan; the transport is the class-level vector."""
        _ = (local_index, group, tag)
        self.needed = tuple(needed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def stages(self, x_local):
        """Yield stage 0 and the peer slices of the full vector."""
        yield 0, x_local
        full = _FullVectorExchange.full
        assert full is not None
        for stage in range(1, self.world_size):
            yield stage, full[self.needed[stage].to(full.device)]


def _tile_output(
    basis,
    helpers,
    integrals,
    partition,
    x: torch.Tensor,
    rank: int,
    *,
    triton_matvec: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> torch.Tensor:
    """Run one rank's tile operator with the stand-in exchange."""
    tile = pair_tile_bounds(partition, basis, rank)
    monkeypatch.setattr(dpt, "StagedPairExchange", _FullVectorExchange)
    _FullVectorExchange.full = x
    operator = dpt.LocalPairTileOperator(
        torch.zeros(tile.local_size, dtype=x.dtype, device=x.device),
        basis,
        helpers,
        helpers.n_orb,
        integrals,
        tile,
        partition,
        rank=rank,
        world_size=partition.world_size,
        triton_matvec=triton_matvec,
    )
    return operator(x[tile.pair_index])


_DTYPES = [torch.float64, torch.float32, torch.complex128, torch.complex64]
_IDS = ["fp64", "fp32", "c128", "c64"]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
@pytest.mark.parametrize("basis_name", ["full_product", "tier", "structural"])
def test_same_spin_blocks_triton_matches_batched(basis_name, dtype) -> None:
    """The 1-D full-vector channel matches the batched product kernels."""
    device = _cuda()
    basis, norb = _basis(basis_name, device)
    integrals = _integrals_for(norb, device, dtype)
    helpers = _helpers(basis, norb, device)
    x = _rand_vector(basis.n_pairs, dtype, device)

    expected = torch.zeros_like(x)
    got = torch.zeros_like(x)
    add_same_spin_blocks(expected, x, helpers, integrals)
    add_same_spin_blocks(got, x, helpers, integrals, triton_matvec=True)
    _assert_parity(got, expected, dtype, f"1-D {basis_name}")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
@pytest.mark.parametrize("basis_name", ["full_product", "tier", "structural"])
def test_same_spin_blocks_local_triton_matches_batched(basis_name, dtype) -> None:
    """The owner-computes channel matches the batched kernels on every range."""
    device = _cuda()
    basis, norb = _basis(basis_name, device)
    integrals = _integrals_for(norb, device, dtype)
    helpers = _helpers(basis, norb, device)
    n = basis.n_pairs
    x = _rand_vector(n, dtype, device)
    ranges = [(0, min(6, n)), (n // 3, min(7, n - n // 3)), (0, n), (n, 0)]
    for row_start, n_local in ranges:
        expected = torch.zeros(n_local, dtype=dtype, device=device)
        got = torch.zeros(n_local, dtype=dtype, device=device)
        add_same_spin_blocks_local(expected, x, helpers, integrals, row_start)
        add_same_spin_blocks_local(got, x, helpers, integrals, row_start, triton_matvec=True)
        _assert_parity(got, expected, dtype, f"local {basis_name} [{row_start}+{n_local})")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_same_spin_fragment_triton_matches_batched(dtype) -> None:
    """The α and β fragment channels match the batched product kernels."""
    device = _cuda()
    basis, norb = _basis("tier", device)
    integrals = _integrals_for(norb, device, dtype)
    helpers = _helpers(basis, norb, device)
    blocks = helpers.blocks
    assert blocks is not None
    x = _rand_vector(basis.n_pairs, dtype, device)

    # The largest block of each grouping is the one with same-spin moves on
    # both axes; its full rectangle sends every move through the kernel.
    alpha_index, alpha_group = max(
        enumerate(blocks.by_beta.groups), key=lambda item: item[1].n_pairs
    )
    alpha_grid = x[alpha_group.pair_idx].view(alpha_group.n_rows, alpha_group.n_cols)
    alpha_frag = PairFragment(
        group=alpha_index,
        row_start=0,
        row_stop=alpha_group.n_rows,
        col_start=0,
        col_stop=alpha_group.n_cols,
        pair_index=torch.empty(0, dtype=torch.int64, device=device),
    )
    t_alpha = alpha_grid[:, alpha_frag.col_start : alpha_frag.col_stop].contiguous().reshape(-1)
    expected = torch.zeros(alpha_frag.n_rows * alpha_frag.n_cols, dtype=dtype, device=device)
    got = torch.zeros_like(expected)
    add_same_spin_fragment(expected, t_alpha, helpers, integrals, blocks.by_beta, alpha_frag)
    add_same_spin_fragment(
        got, t_alpha, helpers, integrals, blocks.by_beta, alpha_frag, triton_matvec=True
    )
    _assert_parity(got, expected, dtype, "α fragment")

    beta_index, beta_group = max(
        enumerate(blocks.by_alpha.groups), key=lambda item: item[1].n_pairs
    )
    beta_grid = x[beta_group.pair_idx].view(beta_group.n_rows, beta_group.n_cols)
    beta_frag = PairFragment(
        group=beta_index,
        row_start=0,
        row_stop=beta_group.n_rows,
        col_start=0,
        col_stop=beta_group.n_cols,
        pair_index=torch.empty(0, dtype=torch.int64, device=device),
    )
    t_beta = beta_grid[beta_frag.row_start : beta_frag.row_stop].contiguous().reshape(-1)
    expected = torch.zeros(beta_frag.n_rows * beta_frag.n_cols, dtype=dtype, device=device)
    got = torch.zeros_like(expected)
    add_same_spin_fragment(expected, t_beta, helpers, integrals, blocks.by_alpha, beta_frag)
    add_same_spin_fragment(
        got, t_beta, helpers, integrals, blocks.by_alpha, beta_frag, triton_matvec=True
    )
    _assert_parity(got, expected, dtype, "β fragment")


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_tile_operator_triton_matches_batched(dtype, monkeypatch) -> None:
    """Every tile of a 2x2 grid matches the batched owner-computes operator."""
    device = _cuda()
    basis, norb = _basis("tier", device)
    integrals = _integrals_for(norb, device, dtype)
    helpers = _helpers(basis, norb, device)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 4, grid=(2, 2))
    x = _rand_vector(basis.n_pairs, dtype, device)
    for rank in range(4):
        got = _tile_output(
            basis,
            helpers,
            integrals,
            partition,
            x,
            rank,
            triton_matvec=True,
            monkeypatch=monkeypatch,
        )
        expected = _tile_output(
            basis,
            helpers,
            integrals,
            partition,
            x,
            rank,
            triton_matvec=False,
            monkeypatch=monkeypatch,
        )
        _assert_parity(got, expected, dtype, f"tile rank={rank}")


def test_pair_tile_operator_triton_uneven_empty_grid(monkeypatch) -> None:
    """A 4x3 grid over a 2x5 basis: uneven slabs and empty tiles still match."""
    device = _cuda()
    alphas = np.array([0b0011, 0b1100], dtype=np.int64)
    betas = np.array(
        [sum(1 << int(x) for x in combo) for combo in itertools.combinations(range(4), 2)][:5],
        dtype=np.int64,
    )
    basis = pair_basis_from_arrays(
        np.repeat(alphas, betas.size), np.tile(betas, alphas.size), device=device
    )
    integrals = _integrals(4, device, torch.float64)
    helpers = _helpers(basis, 4, device)
    partition = partition_tiles(basis.n_unique_alphas, basis.n_unique_betas, 12, grid=(4, 3))
    x = _randn(basis.n_pairs, torch.float64, device)
    empty_tiles = 0
    for rank in range(12):
        tile = pair_tile_bounds(partition, basis, rank)
        empty_tiles += int(tile.is_empty)
        got = _tile_output(
            basis,
            helpers,
            integrals,
            partition,
            x,
            rank,
            triton_matvec=True,
            monkeypatch=monkeypatch,
        )
        expected = _tile_output(
            basis,
            helpers,
            integrals,
            partition,
            x,
            rank,
            triton_matvec=False,
            monkeypatch=monkeypatch,
        )
        _assert_parity(got, expected, torch.float64, f"uneven rank={rank}")
    assert empty_tiles > 0


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64], ids=["c128", "c64"])
def test_complex_request_is_available(dtype) -> None:
    """The capability check admits a complex triple sharing one dtype."""
    device = _cuda()
    integrals = _complex_integrals(4, device, dtype)
    x = _randc(16, dtype, device)
    assert same_spin_doubles_available(torch.zeros_like(x), x, integrals)
    # A real operand in the triple cannot share the kernel's single flag.
    real_out = torch.zeros(16, dtype=torch.float64, device=device)
    assert not same_spin_doubles_available(real_out, x, integrals)
