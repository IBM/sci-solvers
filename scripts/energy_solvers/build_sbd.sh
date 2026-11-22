#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail

# Prepare the native prerequisites for the in-process `sbd-eigensolver` package
# (import name `sbd`) that backs the `"sbd"` solver (TPB + GDB modes). This script
# does NOT clone or build anything itself — the package is installed by uv from its
# PyPI sdist and compiles its own pybind11 extension.
#
# The upstream sdist has no wheel: at install time it compiles its pybind11
# extension against a local MPI and BLAS (and, for GPU, the NVIDIA HPC SDK's
# nvc++), discovered through environment variables. This script locates those
# toolchains, AUTO-DETECTS whether a GPU build is possible (an NVIDIA GPU plus
# nvc++) and prepares a GPU-capable build when it is — otherwise a CPU build —
# persists the discovery variables to a sourceable env file, and prints the
# exact `uv sync --extra sbd` invocation. Running a plain `uv sync` (no extras)
# never touches sbd, so the base install can never be broken by a missing
# compiler, MPI, or GPU toolchain.

# Determine the directory this script lives in (resolve symlinks).
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="${ENERGY_SOLVERS_BUILD_ROOT:-$REPO_ROOT/build/energy_solvers}"
mkdir -p "$BUILD_ROOT"
ENV_FILE="$BUILD_ROOT/sbd_env.sh"

# --------------------------------------------------------------------------- #
# 0. Pick up a local NVHPC installed by `install_nvhpc.sh`. That script installs
#    the HPC SDK to a deterministic path under the SAME BUILD_ROOT but does not
#    persist any env vars, so a fresh shell running this script wouldn't find it.
#    We probe that known layout (compilers dir with bin/nvc++, and the bundled
#    OpenMPI 4) so `install_nvhpc.sh` -> `build_sbd.sh` works with no manual
#    setup. Caller-provided NVHPC_HOME / MPI_HOME / on-PATH tools still win.
# --------------------------------------------------------------------------- #
NVHPC_LOCAL_ROOT="$BUILD_ROOT/nvhpc_workdir/nvhpc/Linux_x86_64/24.5"
NVHPC_LOCAL_COMPILERS="$NVHPC_LOCAL_ROOT/compilers"
# The working OpenMPI 4 (NOT the top-level comm_libs/mpi indirection, which is
# broken on single-CUDA HPC SDK 24.5 installs — see install_nvhpc.sh).
NVHPC_LOCAL_MPI="$NVHPC_LOCAL_ROOT/comm_libs/12.4/openmpi4/latest"
if [ -z "${NVHPC_HOME:-}" ] && [ -x "$NVHPC_LOCAL_COMPILERS/bin/nvc++" ]; then
    NVHPC_HOME="$NVHPC_LOCAL_COMPILERS"
    echo "Found NVHPC from install_nvhpc.sh at: $NVHPC_HOME"
fi

# --------------------------------------------------------------------------- #
# 1. MPI. The package requires an MPI toolchain (mpicxx wrapper + libs) and
#    mpi4py at runtime. Honor a caller-provided MPI_HOME; otherwise prefer the
#    OpenMPI bundled with a local NVHPC (so MPI and nvc++ come from one
#    toolchain), else infer it from `mpicxx` on PATH (prefix is two levels up
#    from bin/mpicxx).
# --------------------------------------------------------------------------- #
if [ -z "${MPI_HOME:-}" ]; then
    if [ -x "$NVHPC_LOCAL_MPI/bin/mpicxx" ]; then
        MPI_HOME="$NVHPC_LOCAL_MPI"
        echo "Using NVHPC-bundled OpenMPI from install_nvhpc.sh."
    elif command -v mpicxx >/dev/null 2>&1; then
        MPICXX_PATH="$(command -v mpicxx)"
        MPI_HOME="$(cd "$(dirname "$MPICXX_PATH")/.." && pwd)"
    else
        echo "Error: no MPI found. Set MPI_HOME to an MPI install prefix, or put" >&2
        echo "       mpicxx on PATH (e.g. 'conda install -c conda-forge openmpi'," >&2
        echo "       or load your cluster's MPI module), then re-run." >&2
        exit 1
    fi
fi
echo "MPI_HOME=$MPI_HOME"

# The OpenMPI that NVIDIA bundles in the HPC SDK is built with a hardcoded
# install prefix from NVIDIA's OWN build farm (e.g. /proj/nv/libraries/...).
# When it runs from our RELOCATED unpack, a normal `mpirun` launch can still
# find its siblings relative to argv[0], but a SINGLETON MPI_Init (which is
# exactly what `import sbd` does from a plain `python`, not under mpirun) cannot
# — so OpenMPI falls back to the baked-in path, fails to find orted / its help
# files / MCA components, and aborts in ess/singleton with ORTE_ERROR_LOG. The
# cure is OPAL_PREFIX pointing at the OpenMPI install root, which we emit into
# the env file below. Detect whether MPI_HOME is such a bundled OpenMPI (its
# path lives under an NVHPC `comm_libs/.../openmpi*` tree) so we set OPAL_PREFIX
# ONLY then — a system/conda MPI resolves its own prefix and must not be
# overridden.
NVHPC_OPENMPI=0
case "$MPI_HOME" in
    *comm_libs/*openmpi*) NVHPC_OPENMPI=1 ;;
esac
if [ "$NVHPC_OPENMPI" = "1" ]; then
    # OPAL_PREFIX is the OpenMPI install root; MPI_HOME already points at it.
    # Resolve the `latest` symlink so the value is stable across SDK updates.
    OPAL_PREFIX="$(cd "$MPI_HOME" && pwd -P)"
    echo "OPAL_PREFIX=$OPAL_PREFIX (relocated NVHPC OpenMPI — needed for singleton MPI_Init)"
fi

# --------------------------------------------------------------------------- #
# 2. BLAS. The extension links a BLAS/LAPACK. Honor caller-provided BLAS_LIBS
#    / BLAS_LIB_PATH; otherwise fall back to conda-forge openblas when a conda
#    prefix is active.
#
#    IMPORTANT: the package's setup.py comma-splits BLAS_LIBS and prepends `-l`
#    to EACH entry itself, so this must hold BARE library names (e.g.
#    "openblas" or "openblas,lapack"), NOT `-l`-prefixed flags. Passing
#    "-lopenblas" makes setup.py emit `-l-lopenblas`, which the linker rejects
#    with `cannot find -l-lopenblas`.
# --------------------------------------------------------------------------- #
if [ -z "${BLAS_LIBS:-}" ]; then
    BLAS_LIBS="openblas"
fi
if [ -z "${BLAS_LIB_PATH:-}" ] && [ -n "${CONDA_PREFIX:-}" ]; then
    BLAS_LIB_PATH="$CONDA_PREFIX/lib"
fi
echo "BLAS_LIBS=$BLAS_LIBS"
echo "BLAS_LIB_PATH=${BLAS_LIB_PATH:-<unset — using system default library path>}"

# --------------------------------------------------------------------------- #
# 3. GPU (NVHPC/Thrust) profile — auto-detected by default. The package's
#    setup.py selects the backend from `SBD_BUILD_BACKEND` (auto|cpu|gpu|
#    all|gpu_thrust|gpu_omp_offload); its own `auto` only probes for `nvc++`.
#
#    Whether to build the GPU backends is decided by the GPU TOOLCHAIN (the
#    `nvc++` compiler): if `nvc++` is found, a GPU-capable build is prepared.
#    Which GPU to build for needs a GPU (see SBD_GPU_ARCH below), so on a node
#    without one the target is given explicitly:
#      * NVHPC_HOME    -> the SDK `compilers` dir that holds `bin/nvc++`
#      * SBD_BUILD_BACKEND=all -> compile the CPU, Thrust GPU and OMP-offload GPU
#                         backends so the venv works with or without a GPU at
#                         runtime (sbd >= 1.7 loads one backend per process, so
#                         they coexist; `both` was removed upstream).
#      * SBD_GPU_ARCH  -> the target compute capability: ONE architecture per
#                         build. With NVHPC 26.9 (CCCL 3.4.2) a binary built for
#                         several (`-gpu=cc80,cc90`) runs correctly only on the
#                         FIRST listed architecture: on the others Thrust kernels
#                         launch with the first one's tuning (NVIDIA/cccl#11572,
#                         fixed in CCCL by NVIDIA/cccl#11585), so e.g. a
#                         `cc80,cc90` `thrust::fill` leaves half a vector unset
#                         on an H100 with no error, while `-gpu=cc90,cc80` aborts
#                         on an A100. The DEFAULT is therefore the compute
#                         capability of the GPU `nvidia-smi` reports (building
#                         ON a GPU node). Without a working `nvidia-smi` (a node
#                         with no GPU, or a job with none allocated) there is no
#                         safe default and the script stops: set SBD_GPU_ARCH
#                         (e.g. cc80 for A100, cc90 for H100), one build per GPU
#                         model. A caller-provided SBD_GPU_ARCH wins outright; a
#                         value naming several architectures is passed through
#                         with a warning.
#                         We ALSO append the bundled CUDA toolkit version as a
#                         `cuda<ver>` token (e.g. cc90,cuda12.4). nvc++
#                         passes SBD_GPU_ARCH straight into `-gpu=`, and without a
#                         cuda<ver> token it probes the GPU *driver* to choose a
#                         toolkit — on a driver-less login node it then falls
#                         back to its built-in default (CUDA 11.8), which the
#                         SDK's single-CUDA (12.4) install does NOT ship, so the
#                         compile dies with "CUDA version 11.8 is not available".
#                         Pinning `cuda<ver>` to the version the SDK actually
#                         bundles makes the GPU compile work with no driver.
#    Honor a caller-provided value for any of these (their choice wins). If
#    `nvc++` is absent, fall back to a CPU-only build and say so — never claim
#    GPU while emitting a CPU build.
#    Force a mode explicitly with SBD_BUILD_BACKEND=cpu|gpu|all, or skip
#    detection entirely with SBD_SKIP_GPU_DETECT=1.
# --------------------------------------------------------------------------- #
# Locate the nvc++ compiler (honor NVHPC_HOME/bin/nvc++, else PATH), setting
# NVCXX and (deriving) NVHPC_HOME. A visible GPU is NOT required to build.
# Returns 0 and sets NVCXX when found, 1 otherwise.
_find_nvcxx() {
    NVCXX=""
    if [ -n "${NVHPC_HOME:-}" ] && [ -x "$NVHPC_HOME/bin/nvc++" ]; then
        NVCXX="$NVHPC_HOME/bin/nvc++"
    elif command -v nvc++ >/dev/null 2>&1; then
        NVCXX="$(command -v nvc++)"
        # Derive NVHPC_HOME (the compilers dir) as the parent of bin/nvc++.
        if [ -z "${NVHPC_HOME:-}" ]; then
            NVHPC_HOME="$(cd "$(dirname "$NVCXX")/.." && pwd)"
        fi
    fi
    [ -n "$NVCXX" ]
}

# Set SBD_GPU_ARCH for a GPU build (skipped if the caller already pinned it) and
# append the bundled CUDA-toolkit token. Requires NVHPC_HOME/NVCXX from
# _find_nvcxx. Called for BOTH the auto-detected and the caller-pinned-GPU path,
# so a GPU build always gets a target architecture however it was requested.
_setup_gpu_arch() {
    # Compute capability: nvc++ passes SBD_GPU_ARCH verbatim into -gpu= and
    # compiles GPU code for the EXACT arch(es) named. Build ONE architecture (see
    # section 3 above for why several in one binary is wrong). ARCH_FROM_CALLER
    # records whether the caller pinned it, so the CUDA token is appended only
    # when they did not.

    # CUDA toolkit version. A `cuda<ver>` token is appended to -gpu= so nvc++ uses
    # the toolkit the SDK actually bundles instead of probing the GPU driver
    # (absent on a login node) and defaulting to CUDA 11.8, which a single-CUDA
    # install does not ship. Detect the bundled version from the SDK `cuda/<ver>`
    # dirs (NVHPC_HOME is the `compilers` dir, so the SDK root is its parent).
    NVHPC_SDK_ROOT="$(cd "$NVHPC_HOME/.." && pwd)"
    CUDA_VER=""
    if [ -d "$NVHPC_SDK_ROOT/cuda" ]; then
        # Highest NN.N version dir under cuda/ (ignore the 'lib64' etc.).
        CUDA_VER="$(ls -1 "$NVHPC_SDK_ROOT/cuda" 2>/dev/null \
                    | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
    fi
    CUDA_TOK=""
    [ -n "$CUDA_VER" ] && CUDA_TOK=",cuda${CUDA_VER}"

    ARCH_FROM_CALLER=0
    if [ -n "${SBD_GPU_ARCH:-}" ]; then
        ARCH_FROM_CALLER=1
        n_cc="$(printf '%s\n' "$SBD_GPU_ARCH" | tr ',' '\n' | grep -c '^cc' || true)"
        if [ "$n_cc" -gt 1 ]; then
            echo "WARNING: SBD_GPU_ARCH=$SBD_GPU_ARCH names several architectures. A binary" >&2
            echo "  built for several (with NVHPC 26.9 at least) runs correctly only on the" >&2
            echo "  first one listed (NVIDIA/cccl#11572); build once per GPU model instead." >&2
        fi
    else
        # The GPU this node has. `|| true`: nvidia-smi can exist and still fail (no
        # driver, or no GPU allocated to the job), which must not end the script
        # under `set -euo pipefail`.
        CAP=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                   | head -n1 | tr -d ' .' || true)"
            printf '%s' "$CAP" | grep -qE '^[0-9]+$' || CAP=""
        fi
        if [ -z "$CAP" ]; then
            echo "Error: no GPU visible here (no working nvidia-smi), and a GPU build" >&2
            echo "       needs one architecture: set SBD_GPU_ARCH to the target GPU" >&2
            echo "       (cc80 for A100, cc90 for H100, ...) and re-run, once per GPU" >&2
            echo "       model. Or set SBD_BUILD_BACKEND=cpu for a CPU-only build." >&2
            exit 1
        fi
        SBD_GPU_ARCH="cc${CAP}"
        echo "Building for this node's GPU: $SBD_GPU_ARCH"
    fi

    # Append the CUDA toolkit token (skip if a caller-pinned value already has
    # one — their choice wins).
    if [ -n "$CUDA_TOK" ] \
       && { [ "$ARCH_FROM_CALLER" = "0" ] || ! printf '%s' "$SBD_GPU_ARCH" | grep -qi 'cuda'; }; then
        SBD_GPU_ARCH="${SBD_GPU_ARCH}${CUDA_TOK}"
    fi

    echo "NVHPC_HOME=$NVHPC_HOME"
    echo "nvc++=$NVCXX"
    echo "SBD_GPU_ARCH=$SBD_GPU_ARCH"
    if [ "$ARCH_FROM_CALLER" = "0" ]; then
        echo "  (this node's GPU only; for another GPU model build again with"
        echo "   SBD_GPU_ARCH=... e.g. cc80 for A100, cc90 for H100)"
    fi
}

GPU_ENABLED=0
if [ -n "${SBD_BUILD_BACKEND:-}" ]; then
    # Caller pinned the profile — respect it, only help fill in the GPU knobs.
    case "$(printf '%s' "$SBD_BUILD_BACKEND" | tr '[:upper:]' '[:lower:]')" in
        gpu | gpu_thrust | all | gpu_omp_offload) GPU_ENABLED=1 ;;
        *) GPU_ENABLED=0 ;;
    esac
    echo "SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND (caller-provided)"
    # A caller-pinned GPU profile still needs nvc++ and a target architecture, or
    # the GPU compile targets nvc++'s own default arch, which may not be the GPU's.
    if [ "$GPU_ENABLED" = "1" ]; then
        if _find_nvcxx; then
            _setup_gpu_arch
        else
            echo "WARNING: SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND requests a GPU build but"
            echo "  nvc++ (NVIDIA HPC SDK) was not found. Load your cluster's NVHPC module"
            echo "  or set NVHPC_HOME to the SDK 'compilers' dir, then re-run — otherwise"
            echo "  'uv sync --extra sbd' will fail to compile the GPU backend."
        fi
    fi
elif [ "${SBD_SKIP_GPU_DETECT:-0}" = "1" ]; then
    echo "SBD_SKIP_GPU_DETECT=1 — skipping GPU detection, preparing a CPU build."
    SBD_BUILD_BACKEND="cpu"
else
    # Auto-detect on the GPU TOOLCHAIN.
    if _find_nvcxx; then
        GPU_ENABLED=1
        SBD_BUILD_BACKEND="all"  # CPU + Thrust GPU + OMP-offload GPU in one venv
        echo "Found nvc++ — preparing a GPU-capable build (CPU + Thrust + OMP-offload GPU)."
        _setup_gpu_arch
        echo "SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND"
    else
        SBD_BUILD_BACKEND="cpu"
        echo "nvc++ (NVIDIA HPC SDK) not found — CPU build only."
        echo "  For a GPU build, load your cluster's NVHPC module or set NVHPC_HOME"
        echo "  to the SDK 'compilers' dir (containing bin/nvc++), then re-run."
        echo "SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND"
    fi
fi

# --------------------------------------------------------------------------- #
# Persist the discovery variables so `uv sync --extra sbd` can be run with them
# exported. Re-run this script (or `source` the file) in any fresh shell.
# --------------------------------------------------------------------------- #
{
    echo "# Generated by scripts/energy_solvers/build_sbd.sh — source before"
    echo "# 'uv sync --extra sbd' so the sbd-eigensolver build finds MPI/BLAS"
    echo "# (and, when a GPU was detected, compiles the Thrust GPU backend)."
    echo "export MPI_HOME=$(printf %q "$MPI_HOME")"
    # OPAL_PREFIX lets the relocated NVHPC OpenMPI find orted / help / MCA at
    # runtime (see the note in section 1). Required for a singleton MPI_Init
    # such as `import sbd`; harmless for mpirun launches. Emitted only for a
    # bundled NVHPC OpenMPI, never for a system/conda MPI.
    if [ "$NVHPC_OPENMPI" = "1" ] && [ -n "${OPAL_PREFIX:-}" ]; then
        echo "export OPAL_PREFIX=$(printf %q "$OPAL_PREFIX")"
    fi
    # NOTE: the singleton-MPI transport settings (OMPI_MCA_ess_singleton_isolated
    # / btl / pml) that stop a non-mpirun `import sbd` hanging on an HPC login
    # node are NO LONGER written here. SBDSolver sets them at runtime, in
    # __init__ before `import sbd`, and ONLY when it detects it is not under a
    # launcher (no mpirun/srun env vars) — so multi-node jobs keep the cluster
    # fabric and nobody has to remember to override a sourced default. See
    # `_configure_singleton_mpi` in src/sci_solvers/energy_solvers/sbd_solver.py.
    # OPAL_PREFIX stays here: it is a build/relocation concern (finding orted /
    # help / MCA components for the relocated NVHPC OpenMPI), needed by every
    # launch mode, and set once at build time rather than per solver.
    echo "export BLAS_LIBS=$(printf %q "$BLAS_LIBS")"
    if [ -n "${BLAS_LIB_PATH:-}" ]; then
        echo "export BLAS_LIB_PATH=$(printf %q "$BLAS_LIB_PATH")"
    fi
    echo "export SBD_BUILD_BACKEND=$(printf %q "$SBD_BUILD_BACKEND")"
    if [ "$GPU_ENABLED" = "1" ]; then
        if [ -n "${NVHPC_HOME:-}" ]; then
            echo "export NVHPC_HOME=$(printf %q "$NVHPC_HOME")"
        fi
        if [ -n "${SBD_GPU_ARCH:-}" ]; then
            echo "export SBD_GPU_ARCH=$(printf %q "$SBD_GPU_ARCH")"
        fi
    fi
    # Put the MPI (and the HPC SDK, when there is one) found above on PATH so both the build
    # (mpicxx/nvc++ for `uv sync --extra sbd`) and later runtime (mpirun for `uv run`)
    # resolve them, and their libs + the CUDA runtime on LD_LIBRARY_PATH so the compiled
    # extension and mpi4py load at runtime. Written for every MPI_HOME / NVHPC_HOME, not only
    # an SDK from install_nvhpc.sh: one given by the caller is not on the loader path
    # either, unless a cluster `module load` put it there, and repeating it is harmless.
    sdk_bin="" sdk_lib=""
    if [ -n "${NVHPC_HOME:-}" ]; then
        sdk_root="$(cd "$NVHPC_HOME/.." && pwd)"
        sdk_bin="$(printf %q "$NVHPC_HOME/bin"):"
        sdk_lib="$(printf %q "$NVHPC_HOME/lib"):"
        [ -d "$sdk_root/cuda/lib64" ] && sdk_lib+="$(printf %q "$sdk_root/cuda/lib64"):"
    fi
    echo "export PATH=$sdk_bin$(printf %q "$MPI_HOME/bin"):\$PATH"
    echo "export LD_LIBRARY_PATH=$sdk_lib$(printf %q "$MPI_HOME/lib"):\${LD_LIBRARY_PATH:-}"
    if [ "$GPU_ENABLED" = "1" ]; then
        # NVHPC 24.5 ships CUDA runtime 12.4, but the `sbd-gpu` extra's torch is built for a
        # newer CUDA 12 and needs symbols 12.4 lacks: with NVHPC's libcudart first on the path,
        # `import torch` fails (`undefined symbol: cudaGetDriverEntryPointByVersion`) in every
        # process that sources this file, e.g. each `mpirun` SBD worker. Put the venv's own
        # CUDA runtime first; it is backward compatible with what sbd was built against.
        # Resolved when this file is sourced, so it applies once `uv sync` has installed torch.
        echo "for _cudart in $(printf %q "${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}")/lib/python*/site-packages/nvidia/cuda_runtime/lib; do"
        echo "    [ -d \"\$_cudart\" ] && export LD_LIBRARY_PATH=\"\$_cudart:\$LD_LIBRARY_PATH\""
        echo "done"
        echo "unset _cudart"
    fi
} > "$ENV_FILE"

if [ "$GPU_ENABLED" = "1" ]; then
    BUILD_KIND="a GPU-capable build (SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND)"
else
    BUILD_KIND="a CPU build (SBD_BUILD_BACKEND=$SBD_BUILD_BACKEND)"
fi

cat <<EOF

Wrote SBD build environment for $BUILD_KIND to:
  $ENV_FILE

To install the sbd-eigensolver package into the sci-solvers venv, run:

  source "$ENV_FILE"
  uv sync --extra sbd

A plain 'uv sync' (no extras) never builds sbd, so the base install stays safe.
After installing, confirm the compiled backends with:

  uv run python src/examples/example_run_sbd_solver_gpu.py --mode tpb --compare-cpu

(expect ['cpu'] for a CPU build, or ['cpu', 'gpu', 'gpu-omp'] when the GPU
backends compiled). To override detection, set SBD_BUILD_BACKEND=cpu|gpu|all (or
SBD_SKIP_GPU_DETECT=1 to force CPU) before re-running this script.

A raw 'import sbd' does a singleton MPI_Init that HANGS on an HPC login node
probing the cluster fabric. sci_solvers steers a singleton to run in-process, so
importing it FIRST makes a subsequent bare import safe (and any create_solver
("sbd", ...) path, incl. the example above, is safe on its own):

  uv run python -c "import sci_solvers; import sbd; print(sbd.available_backends())"

Only a bare 'import sbd' with sci_solvers never imported still hangs outside a
launcher; there, set the singleton MCA vars yourself:

  OMPI_MCA_ess_singleton_isolated=1 OMPI_MCA_btl=self,vader OMPI_MCA_pml=ob1 \
    uv run python -c "import sbd; print(sbd.available_backends())"

Note: all backends can live in one venv — sbd >= 1.7 imports only the backend a
process asks for. 'sbd.has_backend_conflict()' reports True if something loaded
the CPU and OMP-offload backends together, which demotes offload to the host.
EOF
