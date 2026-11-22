"""CPU launch-plumbing tests for the GDB same-spin doubles Triton kernel.

The kernel takes complex operands as their interleaved ``(re, im)`` real views
(``view_as_real``) with an ``IS_COMPLEX`` constexpr, exactly as the TPB
kernels of :mod:`~torch_sci_solver.torch_impl.mult_triton` do. These tests
cover the host-side pieces only --- the argument mapping, the unchanged real
launch, the mixed-dtype rejection and the complex-safe host element factor ---
so they need neither Triton nor a CUDA device; the kernels themselves are
covered by the GPU-marked tests in ``test_mult_pair_blocks_triton.py``.
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
from torch_sci_solver.torch_impl import mult_pair_blocks_triton as mpt
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    two_excite_batch,
)
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult_triton import _bra_from_offsets

NORB = 4
STRINGS = sorted(
    int(sum(1 << orb for orb in occ)) for occ in itertools.combinations(range(NORB), 2)
)
_COMPLEX_DTYPES = [torch.complex128, torch.complex64]
_COMPLEX_IDS = ["c128", "c64"]


class _CaptureKernel:
    """Stand-in for the ``triton.jit`` kernel that records each launch."""

    def __init__(self) -> None:
        """Create an empty call log."""
        self.calls: list[tuple] = []

    def __getitem__(self, grid):
        """Return the launcher for one grid, recording args and kwargs."""

        def _launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return _launch


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> _CaptureKernel:
    """Replace the kernel (and ``triton.cdiv``) with capture stubs."""
    capture = _CaptureKernel()
    monkeypatch.setattr(mpt, "triton", SimpleNamespace(cdiv=lambda n, b: -(-n // b)), raising=False)
    monkeypatch.setattr(mpt, "_same_spin_doubles_kernel", capture, raising=False)
    return capture


def _complex(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Seeded complex vector of ``dtype`` with nonzero real and imaginary parts."""
    gen = torch.Generator().manual_seed(3)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    imag = torch.randn(n, generator=gen, dtype=torch.float64)
    return (real + 1j * imag).to(dtype)


def _complex_integrals(dtype: torch.dtype):
    """Complex Hermitian integrals with a genuinely complex same-spin factor.

    The ERI is projected onto the two symmetries a complex chemist-order ERI
    carries, ``(pq|rs) = (rs|pq)`` and ``(pq|rs) = conj((qp|sr))``. A
    density-fitting sum is real after that projection, so it would not
    exercise the complex arithmetic in either same-spin block.
    """
    rng = np.random.default_rng(1)
    hc = rng.standard_normal((NORB, NORB)) + 1j * rng.standard_normal((NORB, NORB))
    hcore = hc + hc.conj().T
    g = rng.standard_normal((NORB,) * 4) + 1j * rng.standard_normal((NORB,) * 4)
    eri = 0.5 * (g + g.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device="cpu", dtype=dtype)


def _launch(
    eij: torch.Tensor, T: torch.Tensor, out: torch.Tensor, *, is_alpha: bool = True
) -> None:
    """Call the launch wrapper with throwaway integer tables."""
    ints = torch.zeros(4, dtype=torch.int64)
    mpt._launch_same_spin_doubles(
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        ints,
        eij,
        T,
        out,
        3,
        6,
        3,
        is_alpha=is_alpha,
        n_limbs=1,
        BLOCK=16,
    )


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES, ids=_COMPLEX_IDS)
@pytest.mark.parametrize("is_alpha", [True, False], ids=["alpha", "beta"])
def test_launch_same_spin_complex_uses_real_pairs(
    dtype: torch.dtype, is_alpha: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complex launch hands Triton interleaved real views with ``IS_COMPLEX``."""
    capture = _install_capture(monkeypatch)
    eij = _complex(4, dtype)
    T = _complex(6, dtype)
    out = _complex(6, dtype)
    _launch(eij, T, out, is_alpha=is_alpha)
    assert len(capture.calls) == 1
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 1
    assert kwargs["IS_FP64"] == 1
    assert kwargs["IS_ALPHA"] == (1 if is_alpha else 0)
    assert kwargs["RANGE_OFF"] == (0 if is_alpha else 1)
    torch.testing.assert_close(args[8], torch.view_as_real(eij), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[9], torch.view_as_real(T), atol=0.0, rtol=0.0)
    torch.testing.assert_close(args[10], torch.view_as_real(out), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dtype, is_fp64", [(torch.float64, 1), (torch.float32, 0)], ids=["fp64", "fp32"]
)
def test_launch_same_spin_real_passes_tensors_unchanged(
    dtype: torch.dtype, is_fp64: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real launch is untouched: same tensors, ``IS_COMPLEX`` off."""
    capture = _install_capture(monkeypatch)
    eij = torch.randn(4, dtype=dtype)
    T = torch.randn(6, dtype=dtype)
    out = torch.randn(6, dtype=dtype)
    _launch(eij, T, out)
    _grid, args, kwargs = capture.calls[0]
    assert kwargs["IS_COMPLEX"] == 0
    assert kwargs["IS_FP64"] == is_fp64
    assert args[8] is eij and args[9] is T and args[10] is out


def test_launch_rejects_mixed_real_and_complex() -> None:
    """One ``IS_COMPLEX`` flag governs every operand, so mixing is rejected."""
    with pytest.raises(ValueError, match="mix complex and real"):
        _launch(
            torch.randn(4, dtype=torch.float64),
            _complex(6, torch.complex128),
            _complex(6, torch.complex128),
        )


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES, ids=_COMPLEX_IDS)
def test_move_element_factor_keeps_complex_in_batched_order(dtype: torch.dtype) -> None:
    """``_move_element_factor`` reproduces the batched kernel's complex element.

    The batched same-spin kernels pass the annihilation indices as
    ``two_excite_batch``'s ``i, j`` and the creation indices as ``a, b``; the
    swapped order would conjugate a complex Hermitian ERI's elements.
    """
    integrals = _complex_integrals(dtype)
    adets = torch.tensor(STRINGS, dtype=torch.int64)
    helpers = generate_all_torch_helpers(STRINGS, STRINGS, NORB, device="cpu")
    helper = next(h for h in helpers if h.task_type == 2 and h.a_doubles_to.numel() > 0)

    bra = _bra_from_offsets(helper.a_doubles_offsets, len(adets), torch.device("cpu"))
    got = mpt._move_element_factor(
        bra,
        helper.a_doubles_an0,
        helper.a_doubles_an1,
        helper.a_doubles_cr0,
        helper.a_doubles_cr1,
        adets,
        NORB,
        integrals,
        is_alpha=True,
    )
    assert got.dtype == dtype and got.is_complex()

    bra_bm = adets[bra]
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
