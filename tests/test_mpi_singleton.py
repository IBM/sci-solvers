"""Tests for :mod:`sci_solvers._mpi_singleton` — the singleton MPI transport guard.

This module only manipulates ``os.environ`` (the MCA vars Open MPI reads at
``MPI_Init``); it never imports ``sbd`` and needs no MPI runtime, so the whole
policy is unit-testable in-process. The behaviour under test is subtle enough to
warrant pinning:

- a bare (non-launcher) ``import sbd`` must be steered onto a safe transport so it
  returns instead of hanging (the CPU default is shared-memory only), while a
  GPU-device call must instead select the CUDA-aware UCX transport;
- the ``ess_singleton_isolated`` flag is common to both bundles: it stops the
  singleton ``MPI_Init`` forking an ``orted`` daemon and then blocking on a pipe
  read waiting for it (the observed hang), a runtime concern independent of the
  transport — so a GPU call keeps it and only upgrades the transport to UCX;
- a value the USER exported is never touched, so an explicit ``OMPI_MCA_*`` wins;
- under a real launcher (``mpirun``/``srun``) the whole thing is a no-op so the
  cluster fabric is left alone.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os

import pytest

import sci_solvers._mpi_singleton as m

# Every env var the module reads or writes; the fixture clears and restores these
# so a test starts from a clean slate and leaves the process environment untouched.
_MANAGED_ENV = (
    "OMPI_MCA_ess_singleton_isolated",
    "OMPI_MCA_btl",
    "OMPI_MCA_pml",
    *m._LAUNCHER_ENV_VARS,
)

# The three transport keys a call may set, in a stable order for assertions.
_TRANSPORT_KEYS = ("OMPI_MCA_ess_singleton_isolated", "OMPI_MCA_btl", "OMPI_MCA_pml")


@pytest.fixture(autouse=True)
def _isolate_env_and_own_keys():
    """Snapshot/restore ``os.environ`` and the module's ``_OWN_KEYS`` per test.

    ``configure_singleton_mpi`` mutates both process-global state (the MCA env
    vars) and module-global state (``_OWN_KEYS``, which records what the module
    itself set). Without restoring both, tests would leak into each other and into
    the real process environment.
    """
    saved_env = {k: os.environ.get(k) for k in _MANAGED_ENV}
    saved_own = set(m._OWN_KEYS)
    for k in _MANAGED_ENV:
        os.environ.pop(k, None)
    m._OWN_KEYS.clear()
    try:
        yield
    finally:
        m._OWN_KEYS.clear()
        m._OWN_KEYS.update(saved_own)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _transport() -> dict[str, str | None]:
    """Current value of the three transport MCA keys (``None`` when unset)."""
    return {k: os.environ.get(k) for k in _TRANSPORT_KEYS}


_CPU_TRANSPORT = {
    "OMPI_MCA_ess_singleton_isolated": "1",
    "OMPI_MCA_btl": "self,vader",
    "OMPI_MCA_pml": "ob1",
}
_GPU_TRANSPORT = {
    # The isolation flag is common to both bundles — it skips the singleton
    # `orted` daemon fork that otherwise hangs the import on a pipe read, a
    # runtime concern independent of the transport. The GPU path differs only in
    # the CUDA-aware transport.
    "OMPI_MCA_ess_singleton_isolated": "1",
    "OMPI_MCA_btl": "^vader,tcp",
    "OMPI_MCA_pml": "ucx",
}


# --------------------------------------------------------------------------- #
# _is_gpu_device
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "device, expected",
    [
        ("gpu", True),
        ("GPU", True),  # case-insensitive
        ("gpu-omp", True),  # any gpu* variant (Thrust or OpenMP offload)
        ("gpu_omp", True),
        ("Gpu-Anything", True),
        ("cpu", False),
        ("auto", False),  # 'auto' is NOT assumed to be a GPU device
        ("", False),
        (None, False),
    ],
)
def test_is_gpu_device(device, expected):
    """Only a device string starting with ``gpu`` (any case) counts as GPU."""
    assert m._is_gpu_device(device) is expected


# --------------------------------------------------------------------------- #
# configure_singleton_mpi — CPU default (login-node safety)
# --------------------------------------------------------------------------- #


def test_cpu_default_sets_isolated_shared_memory_transport():
    """A singleton with no device (the package-import call) gets the CPU bundle.

    This is what stops a bare ``import sbd`` hanging on a login node: the
    shared-memory-only ``self,vader``/``ob1`` transport plus the
    ``ess_singleton_isolated`` flag that skips the fabric probe.
    """
    m.configure_singleton_mpi(None)
    assert _transport() == _CPU_TRANSPORT


def test_explicit_cpu_device_matches_default():
    """``device="cpu"`` selects the same CPU transport as no device."""
    m.configure_singleton_mpi("cpu")
    assert _transport() == _CPU_TRANSPORT


# --------------------------------------------------------------------------- #
# configure_singleton_mpi — GPU (CUDA-aware transport, no isolation)
# --------------------------------------------------------------------------- #


def test_gpu_selects_ucx_with_isolation():
    """A direct GPU singleton gets UCX AND the isolation flag.

    ``ess_singleton_isolated`` skips the singleton ``orted`` daemon fork that
    otherwise hangs the import on a pipe read; it is a runtime/daemon concern,
    independent of UCX's device-pointer handling, so the GPU bundle keeps it. The
    GPU path differs from CPU only in the CUDA-aware transport.
    """
    m.configure_singleton_mpi("gpu")
    assert _transport() == _GPU_TRANSPORT


def test_cpu_default_then_gpu_keeps_isolation_and_upgrades_transport():
    """The real scenario: ``import sci_solvers`` (CPU) then ``SBDSolver(device='gpu')``.

    The CPU-default call plants the isolation flag and the ob1/vader transport; the
    later GPU call must overwrite the transport with UCX (or the GPU Allreduce
    segfaults on the device buffer) while KEEPING the isolation flag (common to
    both bundles — it prevents the singleton daemon-fork hang). Both must be in
    place before the (later) ``import sbd``.
    """
    m.configure_singleton_mpi(None)
    assert _transport() == _CPU_TRANSPORT  # CPU default first

    m.configure_singleton_mpi("gpu")
    assert _transport() == _GPU_TRANSPORT  # isolation kept, transport = ucx


# --------------------------------------------------------------------------- #
# configure_singleton_mpi — never clobber a user's explicit export
# --------------------------------------------------------------------------- #


def test_user_exported_transport_is_preserved():
    """An ``OMPI_MCA_*`` the user exported themselves is left untouched."""
    os.environ["OMPI_MCA_pml"] = "user-choice"
    m.configure_singleton_mpi(None)
    assert os.environ["OMPI_MCA_pml"] == "user-choice"  # not overwritten with ob1
    # The keys the module DID own are still set.
    assert os.environ["OMPI_MCA_ess_singleton_isolated"] == "1"
    assert os.environ["OMPI_MCA_btl"] == "self,vader"


def test_user_exported_isolation_is_preserved_by_gpu_call():
    """A user-exported ``ess_singleton_isolated`` is left as the user set it.

    The GPU bundle also sets this key to ``"1"``, but the module must not treat a
    user export as one of its own: it stays out of ``_OWN_KEYS`` and keeps the
    user's value verbatim (here a non-default ``"0"``), while the transport is
    still upgraded to UCX.
    """
    os.environ["OMPI_MCA_ess_singleton_isolated"] = "0"  # user's own export
    m.configure_singleton_mpi("gpu")
    assert os.environ["OMPI_MCA_ess_singleton_isolated"] == "0"  # not overwritten
    assert "OMPI_MCA_ess_singleton_isolated" not in m._OWN_KEYS
    assert os.environ["OMPI_MCA_pml"] == "ucx"
    assert os.environ["OMPI_MCA_btl"] == "^vader,tcp"


# --------------------------------------------------------------------------- #
# configure_singleton_mpi — no-op under a real launcher
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("launcher_var", m._LAUNCHER_ENV_VARS)
def test_noop_under_launcher(launcher_var):
    """Any launcher env var present -> the function touches nothing.

    Under ``mpirun``/``srun`` the real cluster fabric is exactly what the job
    needs, so the guard must not set (or clear) any MCA var — for every launcher
    variable the module recognizes.
    """
    os.environ[launcher_var] = "1"
    m.configure_singleton_mpi("gpu")
    assert _transport() == {k: None for k in _TRANSPORT_KEYS}


def test_launcher_does_not_disturb_preexisting_mca_vars():
    """Under a launcher, even MCA vars already in the environment are left as-is."""
    os.environ["SLURM_PROCID"] = "0"
    os.environ["OMPI_MCA_pml"] = "ucx"
    m.configure_singleton_mpi(None)
    assert os.environ["OMPI_MCA_pml"] == "ucx"
    assert os.environ.get("OMPI_MCA_btl") is None  # nothing added


# --------------------------------------------------------------------------- #
# configure_singleton_mpi — idempotence
# --------------------------------------------------------------------------- #


def test_repeated_cpu_calls_are_idempotent():
    """Calling the CPU path twice yields the same environment."""
    m.configure_singleton_mpi(None)
    first = _transport()
    m.configure_singleton_mpi(None)
    assert _transport() == first == _CPU_TRANSPORT


def test_repeated_gpu_calls_are_idempotent():
    """Calling the GPU path twice yields the same environment."""
    m.configure_singleton_mpi("gpu")
    first = _transport()
    m.configure_singleton_mpi("gpu")
    assert _transport() == first == _GPU_TRANSPORT


def test_gpu_then_cpu_swaps_transport_back():
    """A CPU call after a GPU call swaps the transport back to shared-memory.

    Order is not expected in practice (the package-import CPU call precedes any
    solver), but the ``_OWN_KEYS`` bookkeeping should still be self-consistent:
    the isolation flag (common to both bundles) stays set, and the module-owned
    transport keys are re-set across calls without leaking.
    """
    m.configure_singleton_mpi("gpu")
    assert _transport() == _GPU_TRANSPORT
    m.configure_singleton_mpi(None)
    assert _transport() == _CPU_TRANSPORT
