"""CPU tests for the selectable matvec backend (Triton path validated on A100)."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pytest
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl.matvec_dispatch import MATVEC_IMPLS
from torch_sci_solver.torch_impl.validate_matvec import _h4_system, compare_matvec_impls


def test_default_impl_is_batched():
    # Default is "auto"; on CPU it must resolve to the portable batched path
    # (never hard-error for lack of CUDA/Triton).
    """Test default impl is batched."""
    assert make_torch_sci_solver(device="cpu").matvec_impl == "batched"


def test_auto_resolves_to_batched_on_cpu():
    """Test auto resolves to batched on cpu."""
    assert make_torch_sci_solver(device="cpu", matvec_impl="auto").matvec_impl == "batched"


def test_auto_resolves_to_triton_on_cuda(monkeypatch):
    """Test auto resolves to triton on cuda."""
    import torch_sci_solver.torch_impl.mult_triton as mt

    monkeypatch.setattr(mt, "_HAVE_TRITON", True)
    assert make_torch_sci_solver(device="cuda", matvec_impl="auto").matvec_impl == "triton"


def test_auto_raises_on_cuda_without_triton(monkeypatch):
    """Test auto raises on cuda without triton."""
    import torch_sci_solver.torch_impl.mult_triton as mt

    monkeypatch.setattr(mt, "_HAVE_TRITON", False)
    with pytest.raises(RuntimeError):
        make_torch_sci_solver(device="cuda", matvec_impl="auto")


def test_explicit_batched_accepted():
    """Test explicit batched accepted."""
    assert make_torch_sci_solver(device="cpu", matvec_impl="batched").matvec_impl == "batched"


def test_invalid_impl_rejected():
    """Test invalid impl rejected."""
    with pytest.raises(ValueError):
        make_torch_sci_solver(device="cpu", matvec_impl="not-a-kernel")


def test_known_impls():
    """Test known impls."""
    assert "batched" in MATVEC_IMPLS and "triton" in MATVEC_IMPLS


def test_validation_gate_reference_is_self_consistent():
    """Test validation gate reference is self consistent."""
    adets, bdets, n_orb, I0, I1, I2 = _h4_system()  # noqa: N806
    diffs = compare_matvec_impls(adets, bdets, n_orb, I0, I1, I2, impls=("batched",), device="cpu")
    assert diffs == {"batched": 0.0}


def test_triton_requires_cuda():
    """Test triton requires cuda."""
    import torch
    from torch_sci_solver.torch_impl.mult_triton import mult_triton

    with pytest.raises(RuntimeError):
        mult_triton(
            torch.zeros(4, dtype=torch.float64),
            torch.zeros(4, dtype=torch.float64),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            2,
            None,
            [],
        )
