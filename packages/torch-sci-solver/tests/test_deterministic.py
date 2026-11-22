"""Tests for ``deterministic=True``.

1. Scope: every call runs under torch's deterministic algorithms, and the
   process-wide switch is restored afterwards (and left alone when the caller
   enabled it); ``CUBLAS_WORKSPACE_CONFIG`` gets its reproducible default.
2. The atomic Triton entry kernels decline in deterministic mode before
   touching their tables, so the batched path runs.
3. The switch is restored when a call raises, and MPS warns that it has no
   deterministic accumulating ``index_put``.
4. CUDA (``@pytest.mark.gpu``): two runs of the spinless and streamed-GDB Triton
   paths and of the factorized αβ pass agree bitwise.
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
from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver, make_torch_sci_solver

_PERMS = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
_PERMS += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]


def _system(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """8-fold symmetric random integrals."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    return h1, sum(raw.transpose(p) for p in _PERMS) / 8.0


def _strings(norb: int, nocc: int) -> np.ndarray:
    """All ``nocc``-electron strings of ``norb`` orbitals."""
    combos = itertools.combinations(range(norb), nocc)
    return np.array(sorted(sum(1 << o for o in c) for c in combos), dtype=np.int64)


@pytest.fixture(autouse=True)
def _restore_cublas_workspace(monkeypatch):
    """Every ``deterministic=True`` construction may set ``CUBLAS_WORKSPACE_CONFIG``;
    record its prior state so no test leaks it into the session."""
    import os

    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", previous or "")
    if previous is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    yield


def test_calls_run_under_deterministic_algorithms(monkeypatch) -> None:
    """The flag is on inside the solve and restored afterwards."""
    seen: list[bool] = []
    original = TorchSCISolver._solve_one_subspace

    def recording(self, *args, **kwargs):
        seen.append(torch.are_deterministic_algorithms_enabled())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TorchSCISolver, "_solve_one_subspace", recording)
    h1, eri = _system(5)
    strings = _strings(5, 2)
    assert not torch.are_deterministic_algorithms_enabled()
    for flag in (True, False):
        solver = make_torch_sci_solver(device="cpu", apply_ab_projection=False, deterministic=flag)
        solver([(strings, strings)], h1, eri, 5, (2, 2))
        assert not torch.are_deterministic_algorithms_enabled()
    assert seen == [True, False]


def test_caller_enabled_mode_is_left_alone() -> None:
    """A caller that enabled deterministic algorithms keeps them after the call."""
    h1, eri = _system(5)
    strings = _strings(5, 2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        solver = make_torch_sci_solver(device="cpu", apply_ab_projection=False, deterministic=True)
        solver([(strings, strings)], h1, eri, 5, (2, 2))
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(False)


def test_flag_is_restored_when_a_call_raises(monkeypatch) -> None:
    """An exception inside the solve leaves the process-wide switch off again."""

    def failing(_self, *_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(TorchSCISolver, "_solve_one_subspace", failing)
    h1, eri = _system(5)
    strings = _strings(5, 2)
    solver = make_torch_sci_solver(device="cpu", apply_ab_projection=False, deterministic=True)
    with pytest.raises(RuntimeError, match="boom"):
        solver([(strings, strings)], h1, eri, 5, (2, 2))
    assert not torch.are_deterministic_algorithms_enabled()


def test_mps_deterministic_mode_warns() -> None:
    """MPS has no deterministic accumulating index_put, which the constructor says."""
    with pytest.warns(RuntimeWarning, match="not bitwise reproducible"):
        make_torch_sci_solver(
            device="mps", dtype=torch.float32, davidson_tolerance=1e-5, deterministic=True
        )


def test_cublas_workspace_gets_its_reproducible_default(monkeypatch) -> None:
    """``deterministic=True`` sets ``CUBLAS_WORKSPACE_CONFIG`` unless the caller did."""
    if torch.cuda.is_initialized():
        pytest.skip("CUDA is already initialized in this process")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    make_torch_sci_solver(device="cpu", deterministic=True)
    import os

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    make_torch_sci_solver(device="cpu", deterministic=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_entry_kernels_decline_before_reading_tables(monkeypatch) -> None:
    """Under deterministic algorithms the atomic kernels return False untouched."""
    import torch_sci_solver.torch_impl.mult_entries_triton as entries

    monkeypatch.setattr(entries, "_HAVE_TRITON", True)

    class _Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"the kernel read its inputs ({name})")

    poison = _Untouchable()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        assert not entries.apply_single_entries(
            poison,
            poison,
            poison,
            k_bra=poison,
            k_ket=poison,
            cr=poison,
            an=poison,
            mode="spinless",
        )
        assert not entries.apply_double_entries(
            poison,
            poison,
            poison,
            k_bra=poison,
            k_ket=poison,
            cr0=poison,
            cr1=poison,
            an0=poison,
            an1=poison,
            mode="spinless",
        )
    finally:
        torch.use_deterministic_algorithms(False)


def _cuda_run(kind: str):
    """One deterministic CUDA solve of the requested path."""
    h1, eri = _system(7, seed=4)
    strings = _strings(7, 3)
    common = dict(device="cuda", apply_ab_projection=False, deterministic=True)
    if kind == "spinless":
        solver = make_torch_sci_solver(matvec_impl="triton", **common)
        state = solver([_strings(10, 4)], *_system(10, seed=5), 10, 4)[0].states[0]
    elif kind == "gdb_streamed":
        solver = make_torch_sci_solver(
            mode="gdb", matvec_impl="triton", gdb_kwargs={"block_matvec": "off"}, **common
        )
        pairs = (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
        state = solver([pairs], h1, eri, 7, (3, 3))[0].states[0]
    else:
        solver = make_torch_sci_solver(matvec_impl="batched", ab_factorization="cholesky", **common)
        state = solver([(strings, strings)], h1, eri, 7, (3, 3))[0].states[0]
    return float(state.energy), np.asarray(state.sci_state.amplitudes).copy()


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["spinless", "gdb_streamed", "ab_factorized"])
def test_cuda_runs_are_bitwise_reproducible(kind: str) -> None:
    """Deterministic CUDA runs of the formerly atomic paths agree bitwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if kind != "ab_factorized":
        pytest.importorskip("triton")
    first, second = _cuda_run(kind), _cuda_run(kind)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
