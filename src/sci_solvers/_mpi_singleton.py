"""Make a singleton (non-``mpirun``) ``import sbd`` safe on an HPC login node.

The in-process ``sbd-eigensolver`` package runs ``MPI_Init`` at ``import sbd``.
When NOT launched under a parallel launcher (``mpirun``/``mpiexec``/``srun``),
that is a *singleton* init which, on an HPC login node, otherwise HANGS standing
up its runtime and probing the cluster fabric (InfiniBand/PMIx). :func:
`configure_singleton_mpi` detects that case and steers a singleton onto
in-process/shared-memory-only transports so the import returns instead.

It is called from two places, for defence in depth:

- :mod:`sci_solvers` package import, so that ``import sci_solvers`` (or importing
  anything from it) makes a subsequent bare ``import sbd`` safe; and
- :class:`~sci_solvers.energy_solvers.sbd_solver.SBDSolver`'s ``__init__``, just
  before its lazy ``import sbd`` — the supported entry point, safe on its own.

Both share this one implementation so there is a single source of truth. It is
idempotent, cheap, and a no-op under a launcher, so calling it more than once
(or when sbd is never imported) costs nothing and is always safe.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os

# Environment variables a parallel launcher (mpirun/mpiexec, or SLURM's srun)
# sets in every rank's environment. If ANY of these is present we were launched
# under a real multi-rank job and must NOT touch the MPI transport — the fabric
# (InfiniBand/PMIx) is exactly what the job needs. If NONE is present, an
# ``import sbd`` will do a *singleton* MPI_Init that hangs on the fabric probe.
_LAUNCHER_ENV_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI mpirun/mpiexec
    "OMPI_COMM_WORLD_RANK",
    "PMIX_RANK",  # PMIx (Open MPI 4+, some srun configs)
    "PMI_RANK",  # MPICH / Intel MPI / Hydra
    "PMI_SIZE",
    "SLURM_PROCID",  # srun
    "SLURM_NTASKS",
)

# `ess_singleton_isolated=1` is the crux of making a NON-launcher `import sbd`
# return instead of hang, and it belongs on BOTH the CPU and GPU singleton paths.
# Without it, a singleton `MPI_Init` forks an `orted` runtime daemon to act as its
# own one-process resource manager and then BLOCKS on a `read()` of the pipe it
# expects that daemon to write the session rendezvous into. When the daemon fails
# to start / hand back (relocated OpenMPI, PATH quirks, an HPC node), the parent
# waits on that pipe forever — the observed hang is a bare `pipe_read` in the main
# thread, NOT a fabric probe or a UCX/PMIx rendezvous. `mpirun -np 1` never hits
# this because mpirun IS the daemon, so no `orted` fork happens (which is why the
# whole module no-ops under a launcher). Setting the flag makes Open MPI run truly
# isolated: no daemon fork, no pipe, no hang. It governs RUNTIME/daemon startup
# only; it does not change a collective's CUDA-awareness (that comes from the
# transport — see the GPU bundle), so it is safe on the GPU path too.
#
# CPU singleton transport: shared-memory only (no fabric), matched point-to-point
# layer, plus the isolation flag above. NOT CUDA-aware, so it must NOT be used for
# the GPU backend — the GPU bundle overrides the transport with UCX (see below).
_SINGLETON_MPI_MCA_CPU = {
    "OMPI_MCA_ess_singleton_isolated": "1",
    "OMPI_MCA_btl": "self,vader",
    "OMPI_MCA_pml": "ob1",
}

# GPU singleton transport: the sbd Thrust backend hands MPI collectives (e.g.
# Allreduce in Davidson) raw CUDA *device* pointers, so the transport must be
# CUDA-aware or MPI treats the device buffer as host memory and SEGFAULTs in
# opal_datatype_copy_content_same_ddt / mca_coll_cuda_allreduce. UCX is the
# CUDA-aware path; we also exclude the non-CUDA-aware vader/tcp BTLs. It KEEPS
# `ess_singleton_isolated=1` for the same reason the CPU bundle does — to skip the
# singleton `orted` daemon fork that otherwise hangs the import on a pipe read
# (the flag is a runtime/daemon concern, orthogonal to UCX's device-pointer
# handling; the earlier belief that it forced a host-memcpy `self` Allreduce and
# segfaulted the GPU pointer was wrong — that segfault was the CPU *transport*,
# which UCX here replaces).
_SINGLETON_MPI_MCA_GPU = {
    "OMPI_MCA_ess_singleton_isolated": "1",
    "OMPI_MCA_pml": "ucx",
    "OMPI_MCA_btl": "^vader,tcp",
}


# Keys this module set itself (not the user). Lets a later GPU-device call
# override the CPU transport WE defaulted, while still never clobbering a value
# the user explicitly exported. Transport MCA vars are read only at MPI_Init, so
# an override before `import sbd` is effective; after it, it is a harmless no-op.
_OWN_KEYS: set[str] = set()


def _is_gpu_device(device: str | None) -> bool:
    """True for any sbd device string that runs on the GPU (Thrust or OpenMP)."""
    return device is not None and device.lower().startswith("gpu")


def configure_singleton_mpi(device: str | None = None) -> None:
    """Make a singleton (non-``mpirun``) ``import sbd`` return instead of hang.

    A no-op when launched under a parallel launcher (``mpirun``/``srun`` —
    detected via :data:`_LAUNCHER_ENV_VARS`), so genuine multi-node jobs keep the
    cluster fabric untouched. For a singleton it sets the isolation +
    transport MCA vars for the target ``device``:

    - ``device`` CPU/unknown (the default, and the package-import call): the
      shared-memory-only :data:`_SINGLETON_MPI_MCA_CPU` transport.
    - ``device`` GPU (``"gpu"``, ``"gpu-omp"``, …): the CUDA-aware
      :data:`_SINGLETON_MPI_MCA_GPU` (UCX) transport, because the Thrust backend
      passes device pointers into MPI collectives and the CPU transport is not
      CUDA-aware (it SEGFAULTs on the device buffer). A GPU singleton runs on a
      GPU node, where UCX is the right transport anyway.

    BOTH bundles set ``ess_singleton_isolated=1`` — the flag that stops a
    singleton ``MPI_Init`` forking an ``orted`` daemon and then hanging on a pipe
    read waiting for it (see the bundle comments); it is a runtime/daemon concern,
    independent of the transport, so it applies to CPU and GPU alike. The two
    bundles differ only in the point-to-point transport (CPU shared-memory vs.
    CUDA-aware UCX).

    A value the USER explicitly exported is always left untouched, so an explicit
    ``OMPI_MCA_*`` always wins. A CPU default that THIS module set earlier (e.g.
    the package-import call) IS overridden when a later GPU-device call needs the
    CUDA-aware transport — otherwise ``import sci_solvers`` (CPU default) would
    pin ``pml=ob1``/``btl=self,vader`` and the GPU backend would segfault on the
    device pointer. (The isolation flag is common to both bundles, so it is simply
    kept across a CPU→GPU upgrade; the general removal step below still clears any
    module-owned key a selected bundle happens to lack.) It never sets
    ``OPAL_PREFIX`` (a build/relocation concern handled by ``sbd_env.sh``); it only
    steers the transport. The transport MCA vars are read only at ``MPI_Init``
    (i.e. ``import sbd``), so the GPU-aware call must run before that import; after
    it, an override is a harmless no-op. Idempotent.
    """
    launched_under_mpi = any(os.environ.get(v) for v in _LAUNCHER_ENV_VARS)
    if launched_under_mpi:
        return
    settings = _SINGLETON_MPI_MCA_GPU if _is_gpu_device(device) else _SINGLETON_MPI_MCA_CPU
    for key, value in settings.items():
        # Set the key if it is unset, OR if the only reason it is set is that WE
        # set it earlier (a CPU default now being upgraded for the GPU backend).
        # Never overwrite a value the user exported themselves.
        if key not in os.environ or key in _OWN_KEYS:
            os.environ[key] = value
            _OWN_KEYS.add(key)
    # Remove any MCA key WE planted earlier that the selected bundle does not
    # include — e.g. a CPU-default `ess_singleton_isolated=1` that must be gone
    # before a GPU singleton's `import sbd`, since it would fault the Allreduce on
    # the device pointer. Only keys this module set are cleared; a user's export
    # (not tracked in `_OWN_KEYS`) is always preserved.
    for key in list(_OWN_KEYS):
        if key not in settings and os.environ.get(key) is not None:
            del os.environ[key]
            _OWN_KEYS.discard(key)
