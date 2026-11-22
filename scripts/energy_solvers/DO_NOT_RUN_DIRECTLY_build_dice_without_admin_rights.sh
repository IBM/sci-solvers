#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail

#########################################################
## note - this script is called by "build_dice.sh"
## ideally, you do not need to run this script yourself
#########################################################

# This script assumes you've already:
# 1. Activated the for_dice conda environment
# 2. Installed HDF5 via: conda install -c conda-forge hdf5

# Check if CONDA_PREFIX is set (means conda env is active)
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: No conda environment is active!"
    echo "Please run: conda activate for_dice (or your name for it)"
    exit 1
fi

echo "Using conda environment: $CONDA_PREFIX"
echo "Checking for HDF5..."
if [ ! -f "$CONDA_PREFIX/include/hdf5.h" ]; then
    echo "ERROR: HDF5 not found in conda environment!"
    echo "Please install it: conda install -c conda-forge hdf5"
    exit 1
fi
echo "✓ HDF5 found at: $CONDA_PREFIX"

# Force OpenMPI wrappers to use system g++ instead of nvc++
export OMPI_CXX=g++
export OMPI_CC=gcc
export OMPI_FC=gfortran

echo "Configured MPI wrappers to use:"
echo "  CXX: $OMPI_CXX"
echo "  CC: $OMPI_CC"

# Start with a fresh external dir
rm -rf external/ && mkdir external

# Get Boost
echo "Downloading Boost..."
wget --progress=dot:giga https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz \
    && tar -xzf boost_1_85_0.tar.gz -C $(pwd)/external \
    && rm boost_1_85_0.tar.gz

# Clone Dice default branch
echo "Cloning Dice..."
git clone https://github.com/caleb-johnson/Dice.git $(pwd)/external/Dice

# Fix the Dice Makefile to look for Boost headers in the right place
# The headers are in $(BOOST)/boost/ not $(BOOST)/include/boost/
sed -i 's|INCLUDE_BOOST = -I$(BOOST)/include|INCLUDE_BOOST = -I$(BOOST)|g' $(pwd)/external/Dice/Makefile

# Set up environment variables
export BOOST_ROOT=$(pwd)/external/boost_1_85_0
export CURC_HDF5_ROOT=$CONDA_PREFIX  # Use conda's HDF5
DICE_ROOT=$(pwd)/external/Dice
DICE_BIN_PATH=$(pwd)/qiskit_addon_dice_solver/bin

echo "BOOST_ROOT: $BOOST_ROOT"
echo "CURC_HDF5_ROOT: $CURC_HDF5_ROOT"
echo "DICE_ROOT: $DICE_ROOT"

# Build boost with MPI support
echo "Building Boost (this takes ~10-15 minutes)..."
cd $BOOST_ROOT
# Build the libraries Dice needs, but also install ALL headers (including format)
./bootstrap.sh --with-libraries=serialization,mpi,system,filesystem,program_options
# Tell boost to use the MPI compiler wrappers
echo "using mpi ;" >> project-config.jam
# Build and install headers
./b2 -j16 headers
./b2 -j16

# Build DICE
echo "Building DICE (this takes ~5 minutes)..."
cd $DICE_ROOT
make -j16 Dice

# Put the runtime libraries in the package
echo "Copying binaries to package..."
mkdir -p $DICE_BIN_PATH
cp $DICE_ROOT/bin/Dice $DICE_BIN_PATH
cp $BOOST_ROOT/stage/lib/*.so* $DICE_BIN_PATH

# Also copy HDF5 libraries from conda
cp $CONDA_PREFIX/lib/libhdf5*.so* $DICE_BIN_PATH/ 2>/dev/null || true

echo ""
echo "✓ Build complete!"
#echo "Now run: pip install -e ."
