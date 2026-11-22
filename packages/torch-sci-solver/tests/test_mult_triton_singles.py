"""Output-stationary Triton same-spin singles kernels.

The TPB Triton matvec tiles the α- and β-singles passes with
``mult_triton._singles_kernel``. This module covers

1. the CPU-checkable element: ``_singles_source_factor`` reproduces the
   source-spin ``one_excite_batch`` sum, and source factor plus in-kernel
   spectator direct contraction times the full-determinant parity reproduces
   the ``one_excite_batch`` element of the bra determinant (single-word and
   limb strings, real and complex);
2. the CPU-checkable dispatch: the layout predicate accepts the canonical
   helper tables and rejects layouts the kernel cannot address, the launch
   wrapper passes complex operands as interleaved real views, and
   ``TRITON_PASS_BACKENDS`` marks both singles passes tiled;
3. the CUDA kernels (``@pytest.mark.gpu``): ``_launch_singles`` reproduces the
   batched singles kernels at ``1e-12`` (``float64``/``complex128``) and
   ``1e-5`` (``float32``/``complex64``), including padded and narrow 2-D tile
   stage buffers (``T_stride != M_b``) and two-limb strings, and
   ``mult_triton`` tiles both channels, falling back to the batched fast path
   when the predicate rejects the layout.
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
from torch_sci_solver.torch_impl import mult_triton as mt
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    one_excite_batch,
    parity_batch,
)
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import (
    _apply_alpha_single_chunk_fast,
    _apply_beta_single_chunk_fast,
    _bra_from_offsets,
    _interleaved_range_parity,
    _string_occ_matrix,
    make_diag_terms_torch,
    mult_torch,
)

NORB = 4
STRINGS = sorted(
    int(sum(1 << orb for orb in occ)) for occ in itertools.combinations(range(NORB), 2)
)

_DTYPES = [
    (torch.float64, 1e-12),
    (torch.float32, 1e-5),
    (torch.complex128, 1e-12),
    (torch.complex64, 1e-5),
]


# ============================================================================
# Shared setup
# ============================================================================


def _real_h4() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic real symmetric 4-orbital integrals."""
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


def _complex_h4() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic complex Hermitian 4-orbital integrals (as the solver builds)."""
    rng = np.random.default_rng(0)
    hc = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
    hcore = hc + hc.conj().T
    g = rng.standard_normal((NORB,) * 4) + 1j * rng.standard_normal((NORB,) * 4)
    eri = 0.5 * (g + g.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hcore, eri


def _build(dtype: torch.dtype, device: str):
    """Integrals, the three helpers, the strings and the real diagonal."""
    if dtype.is_complex:
        hcore, eri = _complex_h4()
    else:
        hcore, eri = _real_h4()
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    adets_t = torch.tensor(STRINGS, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(STRINGS, dtype=torch.int64, device=device)
    helpers = list(generate_all_torch_helpers(STRINGS, STRINGS, NORB, device=device))
    hii = make_diag_terms_torch(adets_t, bdets_t, NORB, integrals)
    return integrals, helpers, adets_t, bdets_t, hii


def _helper(helpers, task_type: int):
    """The helper of one task type."""
    return next(h for h in helpers if h.task_type == task_type)


def _tables(h, is_alpha: bool):
    """The singles CSR tables of one channel."""
    if is_alpha:
        return h.a_singles_offsets, h.a_singles_to, h.a_singles_cr, h.a_singles_an
    return h.b_singles_offsets, h.b_singles_to, h.b_singles_cr, h.b_singles_an


def _random_vector(n: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Seeded random vector in ``dtype`` (complex draws real and imaginary parts)."""
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if dtype.is_complex:
        imag = torch.randn(n, generator=gen, dtype=torch.float64)
        return (real + 1j * imag).to(dtype).to(device)
    return real.to(dtype).to(device)


def _limb_matrix(values: list[int], n_limbs: int) -> torch.Tensor:
    """Stack Python-int strings as an ``(M, n_limbs)`` signed-int64 matrix."""
    mask = (1 << 64) - 1
    rows = []
    for value in values:
        row = [(value >> (64 * k)) & mask for k in range(n_limbs)]
        rows.append([limb - (1 << 64) if limb >= 1 << 63 else limb for limb in row])
    return torch.tensor(rows, dtype=torch.int64)


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
# 1. CPU: the tiled element is the batched one_excite element
# ============================================================================


@pytest.mark.parametrize(("dtype", "tol"), _DTYPES, ids=[str(d[0]) for d in _DTYPES])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
def test_source_factor_is_the_source_spin_one_excite_sum(
    dtype: torch.dtype, tol: float, is_alpha: bool
) -> None:
    """``_singles_source_factor`` is ``one_excite_batch`` on the source-spin det."""
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cpu")
    h = _helper(helpers, 2 if is_alpha else 1)
    off, _to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t

    got = mt._singles_source_factor(strings, off, cr, an, NORB, integrals, is_alpha=is_alpha)

    bra = _bra_from_offsets(off, len(strings), torch.device("cpu"))
    bra_bm = strings[bra]
    zero = torch.zeros_like(bra_bm)
    if is_alpha:
        source_det = det_from_alpha_beta_batch(bra_bm, zero, NORB)
    else:
        source_det = det_from_alpha_beta_batch(zero, bra_bm, NORB)
    want = one_excite_batch(source_det, None, an, cr, integrals.I1, integrals.I2_eri, 2 * NORB)
    sgn = parity_batch(
        source_det,
        None,
        torch.minimum(cr, an),
        torch.maximum(cr, an),
        torch.ones(cr.shape[0], dtype=integrals.I2_eri.dtype),
    )
    torch.testing.assert_close(sgn * got, want, atol=tol, rtol=0.0)


@pytest.mark.parametrize(("dtype", "tol"), _DTYPES, ids=[str(d[0]) for d in _DTYPES])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
@pytest.mark.parametrize("n_limbs", [1, 2], ids=["word", "two_limbs"])
def test_element_is_the_full_one_excite_element(
    dtype: torch.dtype, tol: float, is_alpha: bool, n_limbs: int
) -> None:
    """Source factor + spectator direct sum times parity is the full element.

    This is the element the kernel evaluates: the full-determinant parity
    (:func:`_interleaved_range_parity` is the host form of the kernel's
    ``_range_parity``), ``e0`` and the spectator-spin ``I2`` direct
    contraction. Its opposite-spin exchange term is exactly zero in the dense
    TPB ERI, which is why the kernel omits it.
    """
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cpu")
    if n_limbs > 1:
        adets_t = _limb_matrix(STRINGS, n_limbs)
        bdets_t = _limb_matrix(STRINGS, n_limbs)
    h = _helper(helpers, 2 if is_alpha else 1)
    off, _to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t

    e0 = mt._singles_source_factor(strings, off, cr, an, NORB, integrals, is_alpha=is_alpha)
    bra = _bra_from_offsets(off, len(strings), torch.device("cpu"))

    j = torch.arange(NORB, dtype=torch.int64)
    spec_pos = 2 * j + (1 if is_alpha else 0)
    cr_b, an_b = cr.unsqueeze(-1), an.unsqueeze(-1)
    spec_pos_b = spec_pos.unsqueeze(0)
    direct = integrals.I2_eri[cr_b, an_b, spec_pos_b, spec_pos_b]  # (n_exc, L)
    exchange = integrals.I2_eri[cr_b, spec_pos_b, spec_pos_b, an_b]
    torch.testing.assert_close(exchange, torch.zeros_like(exchange), atol=0.0, rtol=0.0)
    spectator = bdets_t if is_alpha else adets_t
    contraction = direct @ _string_occ_matrix(spectator, NORB).to(direct.dtype).T

    lo = torch.minimum(cr, an)
    hi = torch.maximum(cr, an)
    for k in range(cr.shape[0]):
        key = int(bra[k])
        other_rows = range(len(bdets_t)) if is_alpha else range(len(adets_t))
        for other in other_rows:
            ia, ib = (key, other) if is_alpha else (other, key)
            det = det_from_alpha_beta_batch(adets_t[ia : ia + 1], bdets_t[ib : ib + 1], NORB)
            want = one_excite_batch(
                det, None, an[k : k + 1], cr[k : k + 1], integrals.I1, integrals.I2_eri, 2 * NORB
            )
            parity = _interleaved_range_parity(
                adets_t[ia : ia + 1], bdets_t[ib : ib + 1], lo[k : k + 1], hi[k : k + 1]
            )
            sgn = 1.0 - 2.0 * (parity & 1).to(want.dtype)
            got = (sgn * (e0[k] + contraction[k, other])).reshape(1)
            torch.testing.assert_close(got, want, atol=tol, rtol=0.0)


def _emulate_singles_kernel(integrals, helpers, adets_t, bdets_t, x, is_alpha: bool):
    """Torch emulation of ``_singles_kernel``'s per-lane work.

    Same tables, element, parity and ``T`` indexing as the kernel, but executed
    with torch ops, so the kernel's layout and accumulation logic can be
    checked without a GPU.
    """
    M_a, M_b = len(adets_t), len(bdets_t)
    h = _helper(helpers, 2 if is_alpha else 1)
    off, to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t
    e0 = mt._singles_source_factor(strings, off, cr, an, NORB, integrals, is_alpha=is_alpha)
    bra = _bra_from_offsets(off, len(strings), x.device)

    j = torch.arange(NORB, dtype=torch.int64, device=x.device)
    spec_pos = 2 * j + (1 if is_alpha else 0)
    spectator = bdets_t if is_alpha else adets_t
    spectator_occ = _string_occ_matrix(spectator, NORB).to(x.dtype)  # (M_spec, L)
    direct = integrals.I2_eri[
        cr.unsqueeze(-1), an.unsqueeze(-1), spec_pos.unsqueeze(0), spec_pos.unsqueeze(0)
    ]  # (n_exc, L)
    contraction = direct @ spectator_occ.T  # (n_exc, M_spec)
    lo, hi = torch.minimum(cr, an), torch.maximum(cr, an)

    T = x.view(M_a, M_b)
    Wb = torch.zeros(M_a, M_b, dtype=x.dtype, device=x.device)
    for k in range(len(to)):
        key = int(bra[k])
        if is_alpha:
            parity = _interleaved_range_parity(
                adets_t[key : key + 1], bdets_t, lo[k : k + 1], hi[k : k + 1]
            )
            elem = (1.0 - 2.0 * (parity & 1).to(x.dtype)) * (e0[k] + contraction[k])
            Wb[key] += elem * T[to[k]]
        else:
            parity = _interleaved_range_parity(
                adets_t, bdets_t[key : key + 1], lo[k : k + 1], hi[k : k + 1]
            )
            elem = (1.0 - 2.0 * (parity & 1).to(x.dtype)) * (e0[k] + contraction[k])
            Wb[:, key] += elem * T[:, to[k]]
    return Wb.reshape(-1)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["real", "complex"])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
@pytest.mark.parametrize("n_limbs", [1, 2], ids=["word", "two_limbs"])
def test_emulated_kernel_matches_the_batched_singles_contribution(
    dtype: torch.dtype, is_alpha: bool, n_limbs: int
) -> None:
    """The kernel's algorithm (emulated in torch) reproduces the batched pass."""
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cpu")
    if n_limbs > 1:
        adets_t = _limb_matrix(STRINGS, n_limbs)
        bdets_t = _limb_matrix(STRINGS, n_limbs)
    n = len(STRINGS) ** 2
    x = _random_vector(n, dtype, "cpu")
    want = _singles_contribution(integrals, helpers, adets_t, bdets_t, x, is_alpha)
    got = _emulate_singles_kernel(integrals, helpers, adets_t, bdets_t, x, is_alpha)
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0), float((got - want).abs().max())


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["real", "complex"])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
def test_kernel_flat_offsets_select_the_direct_integrals(
    dtype: torch.dtype, is_alpha: bool
) -> None:
    """The kernel's ``off0 + j * step`` arithmetic lands on ``I2[cr, an, p, p]``.

    The kernel never indexes the ERI four-dimensionally; it walks a flat stride
    from ``I2[cr, an, p_0, p_0]``. This checks that arithmetic directly,
    including the complex interleaved element offsets.
    """
    integrals, helpers, _adets_t, _bdets_t, _hii = _build(dtype, "cpu")
    off, _to, cr, an = _tables(_helper(helpers, 2 if is_alpha else 1), is_alpha)
    n_spinorb = 2 * NORB
    j = torch.arange(NORB, dtype=torch.int64)
    pos = 2 * j + (1 if is_alpha else 0)

    off0 = (cr * n_spinorb + an) * n_spinorb * n_spinorb
    step = 2 * (n_spinorb + 1)
    if is_alpha:
        off0 = off0 + n_spinorb + 1
    flat_offsets = off0.unsqueeze(-1) + j.unsqueeze(0) * step

    flat = integrals.I2_eri.reshape(-1)
    if dtype.is_complex:
        real = torch.view_as_real(flat).reshape(-1)
        got = real[flat_offsets * 2] + 1j * real[flat_offsets * 2 + 1]
    else:
        got = flat[flat_offsets]
    want = integrals.I2_eri[cr.unsqueeze(-1), an.unsqueeze(-1), pos.unsqueeze(0), pos.unsqueeze(0)]
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


# ============================================================================
# 2. CPU: dispatch, launches and pass map
# ============================================================================


def test_layout_predicate_accepts_the_canonical_helper_tables() -> None:
    """The full-basis helper tables and buffers fit the kernel."""
    integrals, helpers, adets_t, bdets_t, hii = _build(torch.float64, "cpu")
    T = torch.zeros(len(STRINGS) ** 2, dtype=torch.float64)
    for task_type, is_alpha in ((2, True), (1, False)):
        off, to, cr, an = _tables(_helper(helpers, task_type), is_alpha)
        rows = len(adets_t) if is_alpha else len(bdets_t)
        assert mt._singles_layout_supported(
            adets_t,
            bdets_t,
            off,
            to,
            cr,
            an,
            T,
            hii,
            M=rows,
            n_orb=NORB,
            factor_dtype=integrals.I2_eri.dtype,
        )


def test_layout_predicate_accepts_two_limb_strings() -> None:
    """An ``(M, 2)`` limb encoding wide enough for ``n_orb`` fits the kernel."""
    integrals, helpers, _adets_t, bdets_t, hii = _build(torch.float64, "cpu")
    adets_t = _limb_matrix(STRINGS, 2)
    bdets_t = _limb_matrix(STRINGS, 2)
    off, to, cr, an = _tables(_helper(helpers, 2), True)
    T = torch.zeros(len(STRINGS) ** 2, dtype=torch.float64)
    assert mt._singles_layout_supported(
        adets_t,
        bdets_t,
        off,
        to,
        cr,
        an,
        T,
        hii,
        M=len(adets_t),
        n_orb=128,
        factor_dtype=integrals.I2_eri.dtype,
    )


def test_layout_predicate_rejects_unserviceable_layouts() -> None:
    """Layouts the kernel cannot address are rejected, never raised."""
    integrals, helpers, adets_t, bdets_t, hii = _build(torch.float64, "cpu")
    off, to, cr, an = _tables(_helper(helpers, 2), True)
    T = torch.zeros(len(STRINGS) ** 2, dtype=torch.float64)

    def _supported(
        *,
        adets_t=adets_t,
        bdets_t=bdets_t,
        off=off,
        to=to,
        cr=cr,
        an=an,
        T=T,
        Wb=hii,
        M=len(adets_t),
        n_orb=NORB,
        factor_dtype=integrals.I2_eri.dtype,
    ):
        return mt._singles_layout_supported(
            adets_t, bdets_t, off, to, cr, an, T, Wb, M=M, n_orb=n_orb, factor_dtype=factor_dtype
        )

    assert _supported()
    assert not _supported(off=off.int())
    assert not _supported(to=to.repeat(2)[::2])  # non-contiguous view
    assert not _supported(off=off[:-1])
    assert not _supported(adets_t=adets_t.int())
    assert not _supported(bdets_t=bdets_t.repeat(2)[::2])
    assert not _supported(adets_t=torch.zeros(len(STRINGS), dtype=torch.int64), n_orb=65)
    assert not _supported(T=T.to(torch.complex128))
    assert not _supported(Wb=hii.to(torch.complex128))
    assert not _supported(T=T.repeat(2)[::2])


def test_launch_singles_complex_uses_real_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The singles launch maps every complex operand to its interleaved view."""
    capture = _install_capture(monkeypatch, "_singles_kernel")
    ints = torch.zeros(4, dtype=torch.int64)
    e0 = torch.randn(4, dtype=torch.complex128)
    I2 = torch.randn(5, dtype=torch.complex128)
    T = torch.randn(6, dtype=torch.complex128)
    Wb = torch.randn(6, dtype=torch.complex128)
    mt._launch_singles(
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        ints,
        ints,
        ints,
        ints,
        e0,
        I2,
        T,
        Wb,
        3,
        6,
        8,
        IS_ALPHA=1,
    )
    assert len(capture.calls) == 1
    grid, args, kwargs = capture.calls[0]
    assert grid == (1,)
    assert kwargs["IS_COMPLEX"] == 1
    assert kwargs["IS_ALPHA"] == 1
    assert kwargs["N_LIMBS"] == 1
    torch.testing.assert_close(args[6], torch.view_as_real(e0), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[7], torch.view_as_real(I2), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[8], torch.view_as_real(T), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[9], torch.view_as_real(Wb), atol=0.0, rtol=0.0)
    assert args[10] == 3 and args[11] == 6 and args[12] == 3 and args[13] == 8


def test_launch_singles_real_passes_tensors_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real launch is untouched: same tensors, ``IS_COMPLEX`` off."""
    capture = _install_capture(monkeypatch, "_singles_kernel")
    ints = torch.zeros(4, dtype=torch.int64)
    e0 = torch.randn(4, dtype=torch.float64)
    I2 = torch.randn(5, dtype=torch.float64)
    T = torch.randn(6, dtype=torch.float64)
    Wb = torch.randn(6, dtype=torch.float64)
    mt._launch_singles(
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(3, dtype=torch.int64),
        ints,
        ints,
        ints,
        ints,
        e0,
        I2,
        T,
        Wb,
        3,
        6,
        8,
        IS_ALPHA=0,
        T_stride=5,
    )
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 0
    assert args[6] is e0 and args[7] is I2 and args[8] is T and args[9] is Wb
    assert args[12] == 5


def test_launch_singles_rejects_mixed_real_and_complex() -> None:
    """One ``IS_COMPLEX`` flag governs every operand, so mixing is rejected."""
    ints = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError, match="mix complex and real"):
        mt._launch_singles(
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int64),
            ints,
            ints,
            ints,
            ints,
            torch.randn(4, dtype=torch.float64),
            torch.randn(5, dtype=torch.complex128),
            torch.randn(6, dtype=torch.complex128),
            torch.randn(6, dtype=torch.complex128),
            3,
            6,
            8,
            IS_ALPHA=1,
        )


def test_apply_singles_falls_back_when_the_layout_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected layout runs the batched fast kernel instead of the launch."""
    integrals, helpers, adets_t, bdets_t, _hii = _build(torch.float64, "cpu")
    M_a, M_b = len(adets_t), len(bdets_t)
    n = M_a * M_b
    x = _random_vector(n, torch.float64, "cpu")
    want = _singles_contribution(integrals, helpers, adets_t, bdets_t, x, True)

    monkeypatch.setattr(mt, "_singles_layout_supported", lambda *a, **k: False)

    def _no_launch(*_args, **_kwargs):  # pragma: no cover - the fallback must skip it
        raise AssertionError("the tiled kernel was launched for a rejected layout")

    monkeypatch.setattr(mt, "_launch_singles", _no_launch)
    got = torch.zeros_like(want)
    mt._apply_singles_triton(
        got,
        x,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        _helper(helpers, 2),
        M_a,
        M_b,
        n,
        is_alpha=True,
        BLOCK=128,
        chunk_size=2_000_000,
        I2_flat=integrals.I2_eri.reshape(-1).contiguous(),
    )
    torch.testing.assert_close(got, want, atol=1e-12, rtol=0.0)


def test_triton_pass_backends_marks_both_singles_tiled() -> None:
    """The benchmark metadata reports tiled singles for the TPB path."""
    assert mt.TRITON_PASS_BACKENDS["alpha_singles"] == "tiled"
    assert mt.TRITON_PASS_BACKENDS["beta_singles"] == "tiled"


# ============================================================================
# 3. CUDA: the kernels
# ============================================================================


def _singles_contribution(integrals, helpers, adets_t, bdets_t, x, is_alpha: bool) -> torch.Tensor:
    """Batched singles contribution of one channel to a zero output."""
    M_a, M_b = len(adets_t), len(bdets_t)
    Wb = torch.zeros(M_a * M_b, dtype=x.dtype, device=x.device)
    h = _helper(helpers, 2 if is_alpha else 1)
    off, to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t
    bra = _bra_from_offsets(off, len(strings), x.device)
    if is_alpha:
        _apply_alpha_single_chunk_fast(
            Wb,
            x,
            adets_t,
            bdets_t,
            NORB,
            integrals,
            bra_alpha_per_exc=bra,
            ket_alpha_per_exc=to,
            cr=cr,
            an=an,
            M_a=M_a,
            M_b=M_b,
            T_full=x.view(M_a, M_b),
        )
    else:
        _apply_beta_single_chunk_fast(
            Wb,
            x,
            adets_t,
            bdets_t,
            NORB,
            integrals,
            bra_beta_per_exc=bra,
            ket_beta_per_exc=to,
            cr=cr,
            an=an,
            M_a=M_a,
            M_b=M_b,
        )
    return Wb


@pytest.mark.gpu
@pytest.mark.parametrize(("dtype", "tol"), _DTYPES, ids=[str(d[0]) for d in _DTYPES])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
def test_launch_singles_matches_batched(dtype: torch.dtype, tol: float, is_alpha: bool) -> None:
    """The tiled singles kernel reproduces the batched contribution."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cuda")
    M_a, M_b = len(adets_t), len(bdets_t)
    x = _random_vector(M_a * M_b, dtype, "cuda")
    want = _singles_contribution(integrals, helpers, adets_t, bdets_t, x, is_alpha)

    h = _helper(helpers, 2 if is_alpha else 1)
    off, to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t
    e0 = mt._singles_source_factor(strings, off, cr, an, NORB, integrals, is_alpha=is_alpha)
    got = torch.zeros_like(want)
    mt._launch_singles(
        adets_t,
        bdets_t,
        off,
        to,
        cr,
        an,
        e0,
        integrals.I2_eri.reshape(-1).contiguous(),
        x,
        got,
        M_b,
        M_a * M_b,
        int(integrals.n_spinorb),
        IS_ALPHA=1 if is_alpha else 0,
    )
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


@pytest.mark.gpu
@pytest.mark.parametrize(("dtype", "tol"), _DTYPES, ids=[str(d[0]) for d in _DTYPES])
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
@pytest.mark.parametrize("n_limbs", [1, 2], ids=["word", "two_limbs"])
def test_launch_singles_padded_stage_buffer_matches_batched(
    dtype: torch.dtype, tol: float, is_alpha: bool, n_limbs: int
) -> None:
    """A 2-D stage buffer with ``T_stride > M_b`` serves both channels."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cuda")
    M_a, M_b = len(adets_t), len(bdets_t)
    x = _random_vector(M_a * M_b, dtype, "cuda")
    want = _singles_contribution(integrals, helpers, adets_t, bdets_t, x, is_alpha)

    if n_limbs > 1:
        adets_t = _limb_matrix(STRINGS, n_limbs).to("cuda")
        bdets_t = _limb_matrix(STRINGS, n_limbs).to("cuda")
    stride = M_b + 3
    stage = torch.zeros(M_a, stride, dtype=dtype, device="cuda")
    stage[:, :M_b] = x.view(M_a, M_b)

    h = _helper(helpers, 2 if is_alpha else 1)
    off, to, cr, an = _tables(h, is_alpha)
    strings = adets_t if is_alpha else bdets_t
    e0 = mt._singles_source_factor(strings, off, cr, an, NORB, integrals, is_alpha=is_alpha)
    got = torch.zeros_like(want)
    mt._launch_singles(
        adets_t,
        bdets_t,
        off,
        to,
        cr,
        an,
        e0,
        integrals.I2_eri.reshape(-1).contiguous(),
        stage,
        got,
        M_b,
        M_a * M_b,
        int(integrals.n_spinorb),
        IS_ALPHA=1 if is_alpha else 0,
        T_stride=stride,
        n_limbs=n_limbs,
    )
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


@pytest.mark.gpu
@pytest.mark.parametrize(("dtype", "tol"), _DTYPES, ids=[str(d[0]) for d in _DTYPES])
def test_launch_singles_narrow_beta_stage_buffer_matches_batched(
    dtype: torch.dtype, tol: float
) -> None:
    """A β stage buffer narrower than the output width (a tile column slab).

    The β-singles table is filtered to the stage's ket columns and remapped to
    buffer positions, exactly as the distributed tile path stages it.
    """
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, _hii = _build(dtype, "cuda")
    M_a, M_b = len(adets_t), len(bdets_t)
    x = _random_vector(M_a * M_b, dtype, "cuda")
    h = _helper(helpers, 1)
    off, to, cr, an = _tables(h, False)

    cols = torch.arange(0, M_b, 2, device="cuda")  # narrower than M_b
    keep = torch.isin(to, cols)
    to_kept = to[keep]
    positions = torch.searchsorted(cols, to_kept)
    bra_kept = _bra_from_offsets(off, M_b, x.device)[keep]
    counts = torch.bincount(bra_kept, minlength=M_b)
    off_kept = torch.zeros(M_b + 1, dtype=torch.int64, device="cuda")
    off_kept[1:] = torch.cumsum(counts, dim=0)
    cr_kept, an_kept = cr[keep], an[keep]

    stride = int(cols.numel())
    stage = x.view(M_a, M_b)[:, cols].contiguous()
    want = torch.zeros(M_a * M_b, dtype=dtype, device="cuda")
    _apply_beta_single_chunk_fast(
        want,
        stage,
        adets_t,
        bdets_t,
        NORB,
        integrals,
        bra_beta_per_exc=bra_kept,
        ket_beta_per_exc=positions,
        cr=cr_kept,
        an=an_kept,
        M_a=M_a,
        M_b=M_b,
        T_stride=stride,
    )

    e0 = mt._singles_source_factor(
        bdets_t, off_kept, cr_kept, an_kept, NORB, integrals, is_alpha=False
    )
    got = torch.zeros_like(want)
    mt._launch_singles(
        adets_t,
        bdets_t,
        off_kept,
        positions,
        cr_kept,
        an_kept,
        e0,
        integrals.I2_eri.reshape(-1).contiguous(),
        stage,
        got,
        M_b,
        M_a * M_b,
        int(integrals.n_spinorb),
        IS_ALPHA=0,
        T_stride=stride,
    )
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


def _count_singles_launches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the ``IS_ALPHA`` of every tiled singles launch."""
    real = mt._launch_singles
    calls: list[int] = []

    def _launch(*args, **kwargs):
        calls.append(int(kwargs["IS_ALPHA"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(mt, "_launch_singles", _launch)
    return calls


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float64, 1e-12), (torch.complex128, 1e-12)],
    ids=["float64", "complex128"],
)
def test_mult_triton_tiles_both_singles_channels(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, tol: float
) -> None:
    """``mult_triton`` runs the tiled singles kernels on both channels."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, hii = _build(dtype, "cuda")
    n = len(STRINGS) ** 2
    x = _random_vector(n, dtype, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    cache = mt.build_samespin_cache(adets_t, bdets_t, NORB, integrals, helpers)
    assert "alpha_singles_e0" in cache and "beta_singles_e0" in cache

    calls = _count_singles_launches(monkeypatch)
    got = mt.mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers, cache=cache)
    assert sorted(calls) == [0, 1]
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float64, 1e-12), (torch.complex128, 1e-12), (torch.complex64, 1e-5)],
    ids=["float64", "complex128", "complex64"],
)
def test_mult_triton_two_limb_strings_matches_batched(dtype: torch.dtype, tol: float) -> None:
    """Two-limb strings keep the tiled singles (and doubles) on the batched result."""
    _skip_without_triton()
    integrals, helpers, _adets_t, _bdets_t, hii = _build(dtype, "cuda")
    adets_t = _limb_matrix(STRINGS, 2).to("cuda")
    bdets_t = _limb_matrix(STRINGS, 2).to("cuda")
    n = len(STRINGS) ** 2
    x = _random_vector(n, dtype, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    got = mt.mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert torch.allclose(got, want, atol=tol, rtol=0.0), float((got - want).abs().max())


@pytest.mark.gpu
def test_mult_triton_singles_fallback_matches_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected singles layout falls back to the batched fast path."""
    _skip_without_triton()
    integrals, helpers, adets_t, bdets_t, hii = _build(torch.float64, "cuda")
    n = len(STRINGS) ** 2
    x = _random_vector(n, torch.float64, "cuda")
    want = mult_torch(hii, x, adets_t, bdets_t, NORB, integrals, helpers)

    monkeypatch.setattr(mt, "_singles_layout_supported", lambda *a, **k: False)
    calls = _count_singles_launches(monkeypatch)
    got = mt.mult_triton(hii, x, adets_t, bdets_t, NORB, integrals, helpers)
    assert calls == []
    assert torch.allclose(got, want, atol=1e-12, rtol=0.0), float((got - want).abs().max())
