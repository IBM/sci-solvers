#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail
set -x

# Make `conda activate` work in this script
source "$(conda info --base)/etc/profile.d/conda.sh"

### the formal guide for qiskit-addon-dice-solver has steps that assume either sudo rights or a docker,
### so we use conda to bypass this issue. The conda environment is not needed once the package is built,
### and uv can be used to install qiskit-addon-dice-solver from the local directory
###


# Determine the directory this script lives in (resolve symlinks).
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
echo "Current script directory: $SCRIPT_DIR"

# Build SCRIPTS live in scripts/energy_solvers/; build ARTIFACTS (the *_workdir
# dirs) go under build/energy_solvers/ at the repo root. Override the artifact
# location with ENERGY_SOLVERS_BUILD_ROOT if you want them elsewhere.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="${ENERGY_SOLVERS_BUILD_ROOT:-$REPO_ROOT/build/energy_solvers}"
DICE_BUILD_BASE_DIR="$BUILD_ROOT/dice_workdir"
echo "DICE_BUILD_BASE_DIR is set to '$DICE_BUILD_BASE_DIR'"

# NVHPC installation is assumed to sit under the same build root.
NVHPC_BUILD_BASE_DIR="$BUILD_ROOT/nvhpc_workdir"
if [ ! -d "$NVHPC_BUILD_BASE_DIR" ]; then
    echo "Error: expected NVHPC directory at '$NVHPC_BUILD_BASE_DIR' but it does not exist." >&2
    exit 1
fi
echo "NVHPC_BUILD_BASE_DIR is set to '$NVHPC_BUILD_BASE_DIR'"


# Add MPI to PATH from NVHPC installation.
# Use the versioned openmpi4 path (comm_libs/12.4/openmpi4/latest), NOT the
# un-versioned comm_libs/openmpi4 indirection: install_nvhpc.sh warns the
# latter is broken on single-CUDA (NVHPC_INSTALL_TYPE=single) installs, and
# the SBD build + runtime hints already standardize on the versioned path.
export PATH="$NVHPC_BUILD_BASE_DIR/nvhpc/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/latest/bin:$PATH"
echo "Added MPI to PATH: $NVHPC_BUILD_BASE_DIR/nvhpc/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/latest/bin"

mkdir -p $DICE_BUILD_BASE_DIR
cd $DICE_BUILD_BASE_DIR

ENV_DIR="$DICE_BUILD_BASE_DIR/env_for_dice"

conda create -p "$ENV_DIR" -c conda-forge git hdf5 -y

conda activate "$ENV_DIR"

# Verify conda environment is active
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: Failed to activate conda environment at $ENV_DIR"
    exit 1
fi
echo "✓ Conda environment activated: $CONDA_PREFIX"



rm -rf qiskit-addon-dice-solver

# The Python wrapper (qiskit-addon-dice-solver) is NOT built here: it is installed by
# uv as a git dependency (see pyproject.toml [tool.uv.sources]) and finds the binary
# via PATH/LD_LIBRARY_PATH at runtime (sci_solvers.energy_solvers._dice_compat). This
# script builds only the C++ Dice binary, from its own upstream (caleb-johnson/Dice,
# cloned by the inner build script), into a `qiskit_addon_dice_solver/bin/` workdir.
DICE_WORKDIR="$DICE_BUILD_BASE_DIR/qiskit-addon-dice-solver"
mkdir -p "$DICE_WORKDIR/qiskit_addon_dice_solver/bin"
cd "$DICE_WORKDIR"

# now, copy the build script that sits next to this script, into the workdir
cp $SCRIPT_DIR/DO_NOT_RUN_DIRECTLY_build_dice_without_admin_rights.sh "$DICE_WORKDIR/"

source ./DO_NOT_RUN_DIRECTLY_build_dice_without_admin_rights.sh

conda deactivate

set +x

echo ""
echo "============================================"
echo "✓ Build complete!"
echo "============================================"


cat <<EOF

if you get missing binaries/library errors you may want to use:
export NVHPC_BUILD_BASE_DIR=$NVHPC_BUILD_BASE_DIR
export DICE_BUILD_BASE_DIR=$DICE_BUILD_BASE_DIR
export PATH=\$NVHPC_BUILD_BASE_DIR/nvhpc/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/latest/bin:\$DICE_BUILD_BASE_DIR/env_for_dice/bin:\$PATH
export LD_LIBRARY_PATH=\$DICE_BUILD_BASE_DIR/env_for_dice/lib:\$LD_LIBRARY_PATH

============================================
Next steps to use this build with sci-solvers
============================================

The Python wrapper (qiskit-addon-dice-solver) is installed by 'uv sync'
directly from git (see pyproject.toml [tool.uv.sources]); it does not
need this build to be present. This script only produces the C++
'Dice' binary (+ its boost/hdf5 libraries) at
  $DICE_BUILD_BASE_DIR/qiskit-addon-dice-solver/qiskit_addon_dice_solver/bin/
Expose that directory on PATH and LD_LIBRARY_PATH so the wrapper finds it:
   export PATH="$DICE_BUILD_BASE_DIR/qiskit-addon-dice-solver/qiskit_addon_dice_solver/bin:\$PATH"
   export LD_LIBRARY_PATH="$DICE_BUILD_BASE_DIR/qiskit-addon-dice-solver/qiskit_addon_dice_solver/bin:\${LD_LIBRARY_PATH:-}"
(Or set DICE_BINARY to the 'Dice' filepath directly.) The whole
dice_workdir/ is gitignored, so nothing here shows up in 'git status'.

EOF
