"""The CUDA-runtime guard that stops :class:`SBDSolver` returning a silent ``Energy = 0``.

sbd's Thrust backend links ``libcudart.so.12``; with a CUDA-13 torch imported first the process
holds ``libcudart.so.13`` and no ``.so.12``, and the diagonalization returns zeros (observed on
an A100). ``_cuda_runtime_conflict`` decides that from the process's memory map. These tests
feed it map listings shaped like the ones seen in the working and failing runs; no GPU or ``sbd``
build is needed.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from pathlib import Path

import pytest

from sci_solvers.energy_solvers.sbd_solver import _cuda_runtime_conflict

_VENV = "/workspace/.venv/lib/python3.12/site-packages/nvidia"
_NV12 = "/workspace/build/energy_solvers/nvhpc_workdir/nvhpc/Linux_x86_64/24.5/cuda/12.4/targets/x86_64-linux/lib"


def _line(path: str) -> str:
    """One /proc/self/maps line mapping ``path``."""
    return f"7f0000000000-7f0000100000 r-xp 00000000 00:2a 123 {path}"


def _maps(tmp_path: Path, *paths: str) -> str:
    """Write a fake maps file mapping ``paths`` and return its location."""
    f = tmp_path / "maps"
    f.write_text("\n".join(_line(p) for p in paths) + "\n")
    return str(f)


def test_cuda13_torch_alone_is_a_conflict(tmp_path: Path) -> None:
    """torch's ``libcudart.so.13`` with no ``.so.12`` mapped is the failing state."""
    maps = _maps(tmp_path, f"{_VENV}/cu13/lib/libcudart.so.13", "/usr/lib64/libc.so.6")
    assert _cuda_runtime_conflict(maps) == "libcudart.so.13"


def test_cuda12_torch_is_fine(tmp_path: Path) -> None:
    """A CUDA-12 torch (cu126/cu128) shares ``libcudart.so.12`` with sbd: no conflict."""
    maps = _maps(tmp_path, f"{_VENV}/cuda_runtime/lib/libcudart.so.12")
    assert _cuda_runtime_conflict(maps) is None


def test_sbd_runtime_loaded_first_is_fine(tmp_path: Path) -> None:
    """Both runtimes mapped, NVHPC's ``.so.12`` among them (sbd loaded first / preloaded): fine."""
    maps = _maps(tmp_path, f"{_NV12}/libcudart.so.12.4.127", f"{_VENV}/cu13/lib/libcudart.so.13")
    assert _cuda_runtime_conflict(maps) is None


def test_no_cuda_runtime_is_fine(tmp_path: Path) -> None:
    """Nothing CUDA mapped yet (sbd imported before torch): nothing to complain about."""
    assert _cuda_runtime_conflict(_maps(tmp_path, "/usr/lib64/libc.so.6")) is None


def test_unreadable_maps_is_not_a_conflict(tmp_path: Path) -> None:
    """Without ``/proc`` (non-Linux) the guard stays silent rather than blocking construction."""
    assert _cuda_runtime_conflict(str(tmp_path / "does-not-exist")) is None


def test_gpu_device_with_a_conflict_warns_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU solver built in a conflicting process warns, naming the cause and fixes, but runs.

    ``sbd`` is stubbed so no ``MPI_Init`` runs and construction can complete.
    """
    import sys
    import types

    import sci_solvers.energy_solvers.sbd_solver as mod

    stub = types.ModuleType("sbd")
    device_config = types.ModuleType("sbd.device_config")
    device_config.DeviceConfig = lambda device: types.SimpleNamespace(device=device)
    monkeypatch.setitem(sys.modules, "sbd", stub)
    monkeypatch.setitem(sys.modules, "sbd.device_config", device_config)
    monkeypatch.setattr(mod, "_cuda_runtime_conflict", lambda *a, **k: "libcudart.so.13")
    with pytest.warns(RuntimeWarning, match=r"CUDA 12 runtime.*libcudart\.so\.13.*LD_PRELOAD"):
        mod.SBDSolver(symmetrize_spin=False, device="gpu")


@pytest.mark.parametrize("device", ["cpu", "CPU"])
def test_cpu_device_never_trips_the_guard(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check only applies to the GPU backends; a CPU solver never calls it.

    ``sbd`` is stubbed so no ``MPI_Init`` runs: the constructor only needs the module and its
    ``DeviceConfig``, and this keeps the test independent of an ``sbd`` build.
    """
    import sys
    import types

    import sci_solvers.energy_solvers.sbd_solver as mod

    stub = types.ModuleType("sbd")
    device_config = types.ModuleType("sbd.device_config")
    device_config.DeviceConfig = lambda device: types.SimpleNamespace(device=device)
    monkeypatch.setitem(sys.modules, "sbd", stub)
    monkeypatch.setitem(sys.modules, "sbd.device_config", device_config)
    called: list[str] = []
    monkeypatch.setattr(mod, "_cuda_runtime_conflict", lambda *a, **k: called.append("x") or "boom")
    mod.SBDSolver(symmetrize_spin=False, device=device)
    assert not called
