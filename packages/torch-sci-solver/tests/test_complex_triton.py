"""Complex-dtype support in the tiled Triton TPB matvec.

Triton has no complex scalar type. The kernels take the interleaved
``(re, im)`` real view of every complex operand and an ``IS_COMPLEX``
constexpr selects the pair arithmetic. This module covers

1. the CPU-checkable launch-time pieces: :func:`_samespin_eij` keeps the
   complex value and the batched kernels' excitation order, the launch
   wrappers pass real views with ``IS_COMPLEX`` set, mixed real/complex
   launches are rejected, and the solver gate admits complex Triton for the
   TPB path but still rejects it in GDB mode;
2. the CUDA kernels themselves (``@pytest.mark.gpu``): ``mult_triton`` and
   both ``LocalTPBOperator`` Triton launch sites reproduce the batched
   complex matvec to ``1e-12`` (complex128) and ``1e-5`` (complex64),
   including a 2x2 tile grid whose corner stages exercise a ``T_stride``
   narrower than the output width.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl import distributed_matvec as dmat
from torch_sci_solver.torch_impl import mult_triton as mt
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    two_excite_batch,
)
from torch_sci_solver.torch_impl.distributed_davidson import (
    partition_rows_explicit,
    partition_tiles,
)
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch

NORB = 4
STRINGS = sorted(
    int(sum(1 << orb for orb in occ)) for occ in itertools.combinations(range(NORB), 2)
)


# ============================================================================
# Shared setup
# ============================================================================


def _complex_h4() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic complex Hermitian 4-orbital integrals.

    ``hcore`` is complex Hermitian, and the ERI is projected onto the two
    symmetries a complex chemist-order ERI carries, ``(pq|rs) = (rs|pq)`` and
    ``(pq|rs) = conj((qp|sr))``. (A density-fitting sum ``L L^H`` alone gives
    the conjugate of the first symmetry, so its symmetrized form is real and
    cannot exercise the complex arithmetic in either same-spin block.)
    """
    rng = np.random.default_rng(0)
    hc = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
    hcore = hc + hc.conj().T
    g = rng.standard_normal((NORB,) * 4) + 1j * rng.standard_normal((NORB,) * 4)
    eri = 0.5 * (g + g.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hcore, eri


def _build(dtype: torch.dtype, device: str):
    """Complex integrals, the full helper set and the real diagonal."""
    hcore, eri = _complex_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    adets_t = torch.tensor(STRINGS, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(STRINGS, dtype=torch.int64, device=device)
    helpers = list(generate_all_torch_helpers(STRINGS, STRINGS, NORB, device=device))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    return integrals, helpers, adets_t, bdets_t, hii


def _random_complex_vector(n: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Seeded complex random vector in ``dtype``."""
    gen = torch.Generator(device=device).manual_seed(7)
    real = torch.randn(n, generator=gen, dtype=torch.float64, device=device)
    imag = torch.randn(n, generator=gen, dtype=torch.float64, device=device)
    return (real + 1j * imag).to(dtype)


def _skip_without_triton() -> None:
    """Skip unless this process has CUDA and an importable Triton build."""
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernels")
    if not mt._HAVE_TRITON:
        pytest.skip("Triton is not importable")


class _CaptureKernel:
    """Stand-in for a ``triton.jit`` kernel that records each launch."""

    def __init__(self) -> None:
        """Create an empty call log."""
        self.calls: list[tuple] = []

    def __getitem__(self, grid):
        """Return the launcher for one grid, recording args and kwargs."""

        def _launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return _launch


def _install_capture(monkeypatch: pytest.MonkeyPatch, name: str) -> _CaptureKernel:
    """Replace ``mt.<name>`` with a capture stub for one test."""
    capture = _CaptureKernel()
    if not hasattr(mt, "triton"):
        # The module-level import is guarded; the launch wrappers only need
        # ``triton.cdiv`` when Triton is not installed.
        monkeypatch.setattr(
            mt, "triton", SimpleNamespace(cdiv=lambda n, b: -(-n // b)), raising=False
        )
    monkeypatch.setattr(mt, name, capture, raising=False)
    return capture


# ============================================================================
# 1. CPU: host-side element factor and launch-time argument mapping
# ============================================================================


@pytest.mark.parametrize("dtype", [torch.complex128, torch.complex64])
def test_samespin_eij_keeps_complex_in_batched_order(dtype: torch.dtype) -> None:
    """``_samespin_eij`` reproduces the batched kernel's complex element.

    The batched same-spin kernels pass the annihilation indices as
    ``two_excite_batch``'s ``i, j`` and the creation indices as ``a, b``; the
    swapped order would conjugate a complex Hermitian ERI's elements.
    """
    integrals, helpers, adets_t, _bdets_t, _hii = _build(dtype, "cpu")
    helper = helpers[0]
    assert helper.task_type == 2 and helper.a_doubles_to.numel() > 0

    got = mt._samespin_eij(
        adets_t,
        helper.a_doubles_offsets,
        helper.a_doubles_cr0,
        helper.a_doubles_cr1,
        helper.a_doubles_an0,
        helper.a_doubles_an1,
        NORB,
        integrals,
        is_alpha=True,
    )
    assert got.dtype == dtype and got.is_complex()

    bra = mt._bra_from_offsets(helper.a_doubles_offsets, len(adets_t), torch.device("cpu"))
    bra_bm = adets_t[bra]
    bra_det = det_from_alpha_beta_batch(bra_bm, torch.zeros_like(bra_bm), NORB)
    want = two_excite_batch(
        bra_det,
        None,
        helper.a_doubles_an0,
        helper.a_doubles_an1,
        helper.a_doubles_cr0,
        helper.a_doubles_cr1,
        integrals.I2_eri,
    )
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    assert not torch.allclose(got, want.conj(), atol=1e-8, rtol=0.0), (
        "the swapped creation/annihilation order would conjugate a complex element"
    )


def test_alphabeta_kernel_index_order_is_the_conjugate_of_the_batched_order() -> None:
    """The αβ kernel's integral order is conjugated relative to the batched path.

    ``_alphabeta_kernel`` forms ``I2[A, I, B, J] - I2[A, J, B, I]`` (creation
    index first) while the batched path forms ``I2[I, A, J, B] -
    I2[J, A, I, B]``. For a complex Hermitian ERI the two differ by
    conjugation, which the kernel's complex branch compensates; a real ERI is
    symmetric under the swap, so only a complex one exposes the difference.
    """
    hcore, eri = _complex_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device="cpu", dtype=torch.complex128)
    I2_t = integrals.I2_eri
    n = int(integrals.n_spinorb)
    gen = torch.Generator().manual_seed(5)
    cra, crb, ana, anb = (torch.randint(0, n, (128,), generator=gen) for _ in range(4))
    i_lo = torch.minimum(cra, crb)
    j_hi = torch.maximum(cra, crb)
    a_lo = torch.minimum(ana, anb)
    b_hi = torch.maximum(ana, anb)
    batched = I2_t[i_lo, a_lo, j_hi, b_hi] - I2_t[j_hi, a_lo, i_lo, b_hi]
    kernel = I2_t[a_lo, i_lo, b_hi, j_hi] - I2_t[a_lo, j_hi, b_hi, i_lo]
    torch.testing.assert_close(kernel.conj(), batched, atol=1e-12, rtol=0.0)
    assert not torch.allclose(kernel, batched, atol=1e-3, rtol=0.0), (
        "the test integrals must expose the complex asymmetry this order implies"
    )


def test_launch_alphabeta_complex_uses_real_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The αβ launch hands Triton interleaved real views with ``IS_COMPLEX``."""
    capture = _install_capture(monkeypatch, "_alphabeta_kernel")
    ints = torch.zeros(4, dtype=torch.int64)
    I2 = torch.randn(5, dtype=torch.complex128)
    T = torch.randn(6, dtype=torch.complex128)
    Wb = torch.randn(6, dtype=torch.complex128)
    mt._launch_alphabeta(
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        I2,
        T,
        Wb,
        3,
        2,
        6,
    )
    assert len(capture.calls) == 1
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 1
    torch.testing.assert_close(args[10], torch.view_as_real(I2), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[11], torch.view_as_real(T), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[12], torch.view_as_real(Wb), atol=0.0, rtol=0.0)


def test_launch_alphabeta_real_passes_tensors_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real launch is untouched: same tensors, ``IS_COMPLEX`` off."""
    capture = _install_capture(monkeypatch, "_alphabeta_kernel")
    ints = torch.zeros(4, dtype=torch.int64)
    I2 = torch.randn(5, dtype=torch.float64)
    T = torch.randn(6, dtype=torch.float64)
    Wb = torch.randn(6, dtype=torch.float64)
    mt._launch_alphabeta(
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        I2,
        T,
        Wb,
        3,
        2,
        6,
    )
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 0
    assert args[10] is I2 and args[11] is T and args[12] is Wb


def test_launch_samespin_complex_uses_real_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same-spin launch maps ``eij``/``T``/``Wb`` to interleaved reals."""
    capture = _install_capture(monkeypatch, "_samespin_double_kernel")
    ints = torch.zeros(4, dtype=torch.int64)
    eij = torch.randn(4, dtype=torch.complex64)
    T = torch.randn(6, dtype=torch.complex64)
    Wb = torch.randn(6, dtype=torch.complex64)
    mt._launch_samespin_double(
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        eij,
        T,
        Wb,
        3,
        6,
        IS_ALPHA=1,
        RANGE_OFF=0,
    )
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 1
    torch.testing.assert_close(args[8], torch.view_as_real(eij), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[9], torch.view_as_real(T), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[10], torch.view_as_real(Wb), atol=0.0, rtol=0.0)


def test_launch_rejects_mixed_real_and_complex() -> None:
    """One ``IS_COMPLEX`` flag governs every operand, so mixing is rejected."""
    ints = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="mix complex and real"):
        mt._launch_samespin_double(
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
            ints,
            ints,
            ints,
            ints,
            ints,
            ints,
            torch.randn(4, dtype=torch.float64),
            torch.randn(6, dtype=torch.complex128),
            torch.randn(6, dtype=torch.complex128),
            3,
            6,
            IS_ALPHA=1,
            RANGE_OFF=0,
        )


def test_solver_gate_admits_complex_triton_for_tpb() -> None:
    """The TPB solver keeps an explicit complex Triton request."""
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    solver = TorchSCISolver(device="cpu", dtype=torch.complex128, matvec_impl="triton")
    assert solver.matvec_impl == "triton"


def test_solver_accepts_complex_triton_in_gdb_mode() -> None:
    """The GDB pair-block Triton kernel handles complex dtypes via interleaved real views."""
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    solver = TorchSCISolver(device="cpu", mode="gdb", dtype=torch.complex128, matvec_impl="triton")
    assert solver.matvec_impl == "triton"


# ============================================================================
# 2. CUDA: the kernels themselves
# ============================================================================


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.complex128, 1e-12), (torch.complex64, 1e-5)],
    ids=["complex128", "complex64"],
)
def test_mult_triton_complex_matches_batched(dtype: torch.dtype, tol: float) -> None:
    """The tiled matvec reproduces the batched complex result."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, hii = _build(dtype, "cuda")
    n = int(adets_t.numel()) * int(bdets_t.numel())
    x = _random_complex_vector(n, dtype, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    got = mt.mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.complex128, 1e-12), (torch.complex64, 1e-5)],
    ids=["complex128", "complex64"],
)
def test_operator_row_path_complex_matches_batched(dtype: torch.dtype, tol: float) -> None:
    """The distributed row operator's Triton launch sites handle complex input."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, hii = _build(dtype, "cuda")
    M_a, M_b = int(adets_t.numel()), int(bdets_t.numel())
    x = _random_complex_vector(M_a * M_b, dtype, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    partition = partition_rows_explicit(M_a, M_b, [0, M_a], 1)
    op = dmat.LocalTPBOperator(
        hii,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        helpers,
        partition,
        rank=0,
        world_size=1,
        matvec_impl="triton",
    )
    got = op(x)
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


def _tile_slice(tensor_flat: torch.Tensor, n_rows: int, n_cols: int, bounds) -> torch.Tensor:
    """The tile's block of a flat ``(n_rows, n_cols)`` tensor, in local order."""
    return (
        tensor_flat
        .view(n_rows, n_cols)[
            bounds.row_start : bounds.row_stop, bounds.col_start : bounds.col_stop
        ]
        .reshape(-1)
        .contiguous()
    )


class _FullTileExchange:
    """Stand-in for :class:`StagedSlabExchange` that slices a full vector.

    Mirrors the real exchange's stage index lists so the operator builds the
    same tables; only the transfer is replaced by a slice of ``full_vector``,
    which lets one process drive every rank of a tile grid.
    """

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
    """Stand-in for :class:`StagedCornerExchange` that slices a full vector.

    Mirrors the real exchange's sub-stage plan, so every corner rectangle
    (whose ``T_stride`` is narrower than the owned output width) is exercised
    without a process group.
    """

    full_vector: torch.Tensor | None = None

    def __init__(self, alpha, beta, *, tag=0, max_elems=None):
        """Record the α/β stage lists and the sub-stage plan."""
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
        for stage, (_s_a, _s_b) in enumerate(pairs):
            n_rows = int(rows[stage].numel())
            n_cols = int(cols[stage].numel())
            if max_elems is None or n_cols == 0 or n_rows * n_cols <= int(max_elems):
                bands = ((0, n_rows),)
            else:
                rows_per = max(1, int(max_elems) // n_cols)
                bands = tuple((lo, min(lo + rows_per, n_rows)) for lo in range(0, n_rows, rows_per))
            for row_lo, row_hi in bands:
                sub = len(self._plan)
                self._plan.append((_s_a, _s_b, row_lo, row_hi))
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
            rows = self._rows[stage]
            cols = self._cols[stage]
            yield stage, full[rows][:, cols].contiguous()


def _install_stand_ins(monkeypatch: pytest.MonkeyPatch, x: torch.Tensor) -> None:
    """Point the operator's exchanges at the full-vector stand-ins."""
    monkeypatch.setattr(dmat, "StagedSlabExchange", _FullTileExchange)
    monkeypatch.setattr(dmat, "StagedCornerExchange", _FullTileCornerExchange)
    _FullTileExchange.full_vector = x
    _FullTileCornerExchange.full_vector = x


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.complex128, 1e-12), (torch.complex64, 1e-5)],
    ids=["complex128", "complex64"],
)
def test_operator_tile_grid_complex_matches_batched(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tol: float
) -> None:
    """Every rank of an uneven 2x4 tile grid matches the batched complex matvec.

    Six rows/columns over a ``(2, 4)`` grid give column slabs of 2, 2, 1 and 1,
    so a corner rectangle is narrower than the owned column slab and the
    kernels' ``T_stride`` differs from the output width.
    """
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, hii = _build(dtype, "cuda")
    M_a, M_b = int(adets_t.numel()), int(bdets_t.numel())
    x = _random_complex_vector(M_a * M_b, dtype, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)

    world_size = 8
    tile = partition_tiles(M_a, M_b, world_size, grid=(2, 4))
    _install_stand_ins(monkeypatch, x)
    narrow_stride = False
    for rank in range(world_size):
        bounds = tile.for_rank(rank)
        op = dmat.LocalTPBOperator(
            _tile_slice(hii, M_a, M_b, bounds),
            adets_t,
            bdets_t,
            NORB,
            integrals,
            helpers,
            tile,
            rank=rank,
            world_size=world_size,
            matvec_impl="triton",
        )
        narrow_stride |= any(
            int(op._beta_exchange.stage_indices(stage_b).numel()) != op._local_cols
            for _stage_a, stage_b, _row_lo, _row_hi in op._corner_exchange.pairs()
        )
        got = op(_tile_slice(x, M_a, M_b, bounds))
        expected = _tile_slice(want, M_a, M_b, bounds)
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, atol=tol, rtol=0.0), (
            rank,
            float((got - expected).abs().max()) if got.numel() else 0.0,
        )
    assert narrow_stride, "no corner stage exercised T_stride != local column slab"
