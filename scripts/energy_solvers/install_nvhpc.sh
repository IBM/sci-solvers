#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail
set -x

# nvhpc contains multiple tools that are needed for building the solvers
# MPI is used for its mpirun command
# sbd Thrust path is written for "nvc++" from the NVIDIA HPC SDK, not plain `nvcc`.

# Determine the directory this script lives in (resolve symlinks).
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# Build SCRIPTS live in scripts/energy_solvers/; build ARTIFACTS (the *_workdir
# dirs) go under build/energy_solvers/ at the repo root. Override the artifact
# location with ENERGY_SOLVERS_BUILD_ROOT if you want them elsewhere.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="${ENERGY_SOLVERS_BUILD_ROOT:-$REPO_ROOT/build/energy_solvers}"
NVHPC_BUILD_BASE_DIR="$BUILD_ROOT/nvhpc_workdir"

mkdir -p "$NVHPC_BUILD_BASE_DIR"
cd "$NVHPC_BUILD_BASE_DIR"

wget --progress=dot:giga https://developer.download.nvidia.com/hpc-sdk/24.5/nvhpc_2024_245_Linux_x86_64_cuda_12.4.tar.gz
tar xpzf nvhpc_2024_245_Linux_x86_64_cuda_12.4.tar.gz

export NVHPC_SILENT=true
export NVHPC_INSTALL_DIR="$NVHPC_BUILD_BASE_DIR/nvhpc"
export NVHPC_INSTALL_TYPE=single      # "single" for local install, no network share

nvhpc_2024_245_Linux_x86_64_cuda_12.4/install

# Clean up the tarball and installer scratch dir to save ~10 GB.
rm -rf nvhpc_2024_245_Linux_x86_64_cuda_12.4*.tar.gz \
       nvhpc_2024_245_Linux_x86_64_cuda_12.4


# The compilers now live at:
# $SBD_GPU_BUILD_HOME/nvhpc/Linux_x86_64/24.5/compilers/bin/

# The MPI implementation you want is at:
# $SBD_GPU_BUILD_HOME/nvhpc/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/latest/ # **NOT** the top-level `comm_libs/mpi/` (which is an indirection dir that, on HPC SDK 24.5, is broken on single-CUDA installs).

#Should print an nvc++ banner. We'll test the MPI wrapper after Step 3, when it's properly on PATH.
"$NVHPC_BUILD_BASE_DIR/nvhpc/Linux_x86_64/24.5/compilers/bin/nvc++" --version || { echo "Error: nvc++ sanity test failed!"; exit 1; }
