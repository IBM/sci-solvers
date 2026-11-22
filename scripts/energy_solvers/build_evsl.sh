#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
set -euo pipefail
set -x

# Build EVSL (EigenValues Slicing Library, https://github.com/eigs/EVSL) as a
# SHARED library so the ctypes EVSL wrapper
# (sci_solvers.energy_solvers.evsl) can dlopen it.
#
# EVSL is a matrix-free polynomial/rational-filter Lanczos eigensolver. We use
# only its standard-problem, matrix-free path (SetAMatvec + kpmdos + spslicer +
# ChebLanTr), which is exactly what a determinant-space CI Hamiltonian needs:
# no assembled matrix, no B factorization, no MPI.
#
# Key facts baked into this script (verified against the upstream headers at
# commit b9d8150, the source lives one level down under EVSL_1.1.1/):
#   * Build system is autotools: ./configure (NOT CMake). The default build is
#     a static libevsl.a; --enable-shared is required for the .so/.dylib.
#   * The init/teardown symbols are EVSLStart / EVSLFinish (no underscore) --
#     the ctypes wrapper binds those names.
#   * -fPIC is injected automatically by configure on Linux when --enable-shared
#     is set; on macOS the shared build uses -dynamiclib and produces .dylib
#     (do NOT hand-add -fPIC there).
#   * BLAS/LAPACK: EVSL bundles its own C reference BLAS/LAPACK and can build
#     with no external library at all. We prefer the same BLAS/LAPACK as the
#     rest of the stack to avoid symbol clashes with torch's bundled MKL.
#     Selection precedence (highest first), applied on BOTH platforms:
#       1. Generic override -- set BLAS_LIB_DIR (and optionally BLAS_LIB_NAME,
#          LAPACK_LIB_DIR, LAPACK_LIB_NAME) to point at any BLAS/LAPACK, e.g. a
#          site openblas under lib64 or a split libblas/liblapack. Defaults:
#          BLAS_LIB_NAME=openblas, LAPACK_LIB_DIR=$BLAS_LIB_DIR (one library
#          exporting both), LAPACK_LIB_NAME=$BLAS_LIB_NAME. On macOS this
#          override replaces Accelerate.
#       2. OPENBLAS_ROOT shorthand -- normalizes to BLAS_LIB_DIR=$OPENBLAS_ROOT/lib
#          (falls back to lib64), name openblas, LAPACK same.
#       3. Platform default:
#            - macOS -> Apple Accelerate (--with-framework-accelerate)
#            - Linux -> conda-forge openblas (LAPACK folded in), from an active
#                       CONDA_PREFIX or a freshly-created env, shared so the
#                       `-Wl,-z,defs` shared link has no unresolved symbols.

# Determine the directory this script lives in (resolve symlinks).
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || echo "${BASH_SOURCE[0]}")")" && pwd)"
echo "Current script directory: $SCRIPT_DIR"

# Build SCRIPTS live in scripts/energy_solvers/; build ARTIFACTS (the *_workdir
# dirs) go under build/energy_solvers/ at the repo root. Override the artifact
# location with ENERGY_SOLVERS_BUILD_ROOT if you want them elsewhere.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="${ENERGY_SOLVERS_BUILD_ROOT:-$REPO_ROOT/build/energy_solvers}"
EVSL_BUILD_BASE_DIR="$BUILD_ROOT/evsl_workdir"
echo "EVSL_BUILD_BASE_DIR is set to '$EVSL_BUILD_BASE_DIR'"

EVSL_REPO_URL="${EVSL_REPO_URL:-https://github.com/eigs/EVSL.git}"
# The maintained source sits under a versioned subdirectory in the repo.
EVSL_SRC_SUBDIR="EVSL_1.1.1"

OS="$(uname -s)"
case "$OS" in
    Darwin) LIB_SUFFIX="dylib" ;;
    *)      LIB_SUFFIX="so" ;;
esac
echo "Detected OS '$OS'; expecting libevsl.$LIB_SUFFIX"

mkdir -p "$EVSL_BUILD_BASE_DIR"
cd "$EVSL_BUILD_BASE_DIR"

# ---------------------------------------------------------------------------
# 1. Fetch source
# ---------------------------------------------------------------------------
if [ ! -d "EVSL" ]; then
    git clone --depth 1 "$EVSL_REPO_URL" EVSL
fi
EVSL_SRC="$EVSL_BUILD_BASE_DIR/EVSL/$EVSL_SRC_SUBDIR"
if [ ! -d "$EVSL_SRC" ]; then
    echo "ERROR: expected EVSL source at '$EVSL_SRC' but it does not exist." >&2
    echo "The repo layout may have changed; inspect $EVSL_BUILD_BASE_DIR/EVSL." >&2
    exit 1
fi
cd "$EVSL_SRC"

# ---------------------------------------------------------------------------
# 2. Configure (platform-specific BLAS/LAPACK) + build the shared library
# ---------------------------------------------------------------------------
CONFIGURE_FLAGS=(--enable-shared)

# ---------------------------------------------------------------------------
# Resolve BLAS/LAPACK. Fills BLAS_LIB_DIR/BLAS_LIB_NAME/LAPACK_LIB_DIR/
# LAPACK_LIB_NAME (linked as -l<name> from -L<dir>) OR sets USE_ACCELERATE=1.
# Precedence: generic override (BLAS_LIB_DIR) -> OPENBLAS_ROOT shorthand ->
# platform default (macOS Accelerate; Linux conda/openblas autodetect).
# See the header comment for the full contract.
# ---------------------------------------------------------------------------
USE_ACCELERATE=0

# OPENBLAS_ROOT is a shorthand for the generic vars: probe lib then lib64 for a
# libopenblas.* and feed the winner in as BLAS_LIB_DIR.
if [ -z "${BLAS_LIB_DIR:-}" ] && [ -z "${LAPACK_LIB_DIR:-}" ] && [ -n "${OPENBLAS_ROOT:-}" ]; then
    for _cand in "$OPENBLAS_ROOT/lib" "$OPENBLAS_ROOT/lib64"; do
        if ls "$_cand"/libopenblas.* >/dev/null 2>&1; then
            BLAS_LIB_DIR="$_cand"
            break
        fi
    done
    # If neither had a match, fall through to lib so validation reports it.
    BLAS_LIB_DIR="${BLAS_LIB_DIR:-$OPENBLAS_ROOT/lib}"
fi

if [ -n "${BLAS_LIB_DIR:-}" ] || [ -n "${LAPACK_LIB_DIR:-}" ]; then
    # Generic override (either platform). A single lib dir given via LAPACK_LIB_DIR
    # alone also lands here and back-fills BLAS from it below.
    BLAS_LIB_DIR="${BLAS_LIB_DIR:-$LAPACK_LIB_DIR}"
elif [ "$OS" = "Darwin" ]; then
    # Apple Accelerate provides BLAS + LAPACK; no conda env needed.
    USE_ACCELERATE=1
else
    # Linux default: use openblas (LAPACK folded in) as a *shared* library so
    # EVSL's `-Wl,-z,defs` shared link resolves cleanly. Resolution order:
    #   active CONDA_PREFIX (with openblas)  ->  a fresh conda env (if `conda`
    #   is available)  ->  bail out with a clear message.
    if [ -n "${CONDA_PREFIX:-}" ] && [ -e "$CONDA_PREFIX/lib/libopenblas.so" ]; then
        BLAS_LIB_DIR="$CONDA_PREFIX/lib"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        ENV_DIR="$EVSL_BUILD_BASE_DIR/env"
        if [ ! -d "$ENV_DIR" ]; then
            conda create -p "$ENV_DIR" -c conda-forge openblas make cmake git -y
        fi
        conda activate "$ENV_DIR"
        BLAS_LIB_DIR="$CONDA_PREFIX/lib"
    else
        echo "ERROR: no BLAS/LAPACK found. Either:" >&2
        echo "  * set BLAS_LIB_DIR=/path/to/lib (dir with lib<BLAS_LIB_NAME>.so; default name openblas)," >&2
        echo "  * set OPENBLAS_ROOT=/path/to/openblas (the dir containing lib/libopenblas.so), or" >&2
        echo "  * install conda so this script can create an env with conda-forge openblas." >&2
        echo "As a last resort, drop these flags to fall back to EVSL's bundled reference" >&2
        echo "BLAS/LAPACK (slow but self-contained)." >&2
        exit 1
    fi
fi

if [ "$USE_ACCELERATE" -eq 1 ]; then
    CONFIGURE_FLAGS+=(--with-framework-accelerate)
    echo "Using BLAS/LAPACK: Apple Accelerate framework"
else
    # Apply defaults: openblas exports both BLAS and LAPACK, so LAPACK mirrors
    # BLAS unless the user split them across separate libraries/dirs.
    BLAS_LIB_NAME="${BLAS_LIB_NAME:-openblas}"
    LAPACK_LIB_DIR="${LAPACK_LIB_DIR:-$BLAS_LIB_DIR}"
    LAPACK_LIB_NAME="${LAPACK_LIB_NAME:-$BLAS_LIB_NAME}"

    # Validate that each requested library actually exists (any of .so, a
    # versioned .so.N, .dylib, or a static .a).
    for _spec in "BLAS:$BLAS_LIB_DIR:$BLAS_LIB_NAME" "LAPACK:$LAPACK_LIB_DIR:$LAPACK_LIB_NAME"; do
        _kind="${_spec%%:*}"; _rest="${_spec#*:}"
        _dir="${_rest%%:*}"; _name="${_rest#*:}"
        _found=0
        for _cand in "$_dir/lib$_name.so" "$_dir/lib$_name.dylib" "$_dir/lib$_name.a"; do
            [ -e "$_cand" ] && { _found=1; break; }
        done
        # Also accept a versioned shared object (libfoo.so.0, libfoo.3.dylib).
        # Test each glob on its own so an unmatched pattern doesn't mask a match.
        if [ "$_found" -eq 0 ]; then
            for _glob in "$_dir/lib$_name".so.* "$_dir/lib$_name".*.dylib; do
                [ -e "$_glob" ] && { _found=1; break; }
            done
        fi
        if [ "$_found" -eq 0 ]; then
            echo "ERROR: expected lib$_name.{so,dylib,a} ($_kind) under '$_dir' but did not find it." >&2
            ls -la "$_dir" >&2 || true
            exit 1
        fi
    done
    echo "Using BLAS:   -l$BLAS_LIB_NAME from $BLAS_LIB_DIR"
    echo "Using LAPACK: -l$LAPACK_LIB_NAME from $LAPACK_LIB_DIR"

    # Pass BOTH the lib name/dir flags AND an explicit LDFLAGS: EVSL 1.1.1's
    # LIB/makefile drops the `--with-*-lib-dir` value when it emits SOLIBS, so
    # relying on it alone gives `-l<name>` with no search path -- exactly the
    # "cannot find -l<name>" failure on HPC nodes with a hand-rolled toolchain.
    # LDFLAGS puts `-L<dir>` on the shared-link line AND bakes the rpath so
    # runtime dlopen works.
    LDFLAGS_VAL="-L$BLAS_LIB_DIR -Wl,-rpath,$BLAS_LIB_DIR"
    if [ "$LAPACK_LIB_DIR" != "$BLAS_LIB_DIR" ]; then
        LDFLAGS_VAL="$LDFLAGS_VAL -L$LAPACK_LIB_DIR -Wl,-rpath,$LAPACK_LIB_DIR"
    fi
    CONFIGURE_FLAGS+=(
        --with-blas-lib-name="$BLAS_LIB_NAME"     --with-blas-lib-dir="$BLAS_LIB_DIR"
        --with-lapack-lib-name="$LAPACK_LIB_NAME" --with-lapack-lib-dir="$LAPACK_LIB_DIR"
        LDFLAGS="$LDFLAGS_VAL"
    )
    # Put the dir(s) on the loader path so the step-4 smoke driver resolves them.
    if [ "$OS" = "Darwin" ]; then
        export DYLD_LIBRARY_PATH="$BLAS_LIB_DIR:$LAPACK_LIB_DIR:${DYLD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$BLAS_LIB_DIR:$LAPACK_LIB_DIR:${LD_LIBRARY_PATH:-}"
    fi
fi

echo "Configure flags: ${CONFIGURE_FLAGS[*]}"
# Wipe artifacts from any previous (possibly-misconfigured) build so the new
# configure/flags take effect. `make clean` is safe on a fresh tree.
make clean >/dev/null 2>&1 || true
./configure "${CONFIGURE_FLAGS[@]}"
make

# ---------------------------------------------------------------------------
# 3. Locate the produced shared library + header
# ---------------------------------------------------------------------------
# The shared recipe produces a versioned file (libevsl-<ver>.<suffix>) plus an
# unversioned symlink (libevsl.<suffix>). Find whichever exists.
EVSL_LIB="$(find "$EVSL_SRC" -name "libevsl*.$LIB_SUFFIX" | head -n1 || true)"
EVSL_INC="$(find "$EVSL_SRC" -name "evsl.h" -path "*/include/*" | head -n1 || true)"
if [ -z "$EVSL_INC" ]; then
    EVSL_INC="$EVSL_SRC/INC/evsl.h"
fi

if [ -z "$EVSL_LIB" ] || [ ! -e "$EVSL_LIB" ]; then
    echo "ERROR: no libevsl.$LIB_SUFFIX found under $EVSL_SRC after make." >&2
    echo "       Check that --enable-shared took effect (default build is static .a)." >&2
    find "$EVSL_SRC" -name "libevsl*" >&2 || true
    exit 1
fi
EVSL_LIB_DIR="$(cd "$(dirname "$EVSL_LIB")" && pwd)"
echo "Built shared library: $EVSL_LIB"
echo "Header:               $EVSL_INC"

# ---------------------------------------------------------------------------
# 4. Smoke test: build + run the matrix-free Laplacian driver.
#    LapPLanN_MatFree exercises exactly the API the adapter uses
#    (EVSLStart -> SetAMatvec -> kpmdos -> spslicer -> find_pol -> Cheb Lanczos
#    -> free_pol -> EVSLFinish) via a user-supplied MVFunc, so a clean run here
#    is a strong signal the ctypes path will work.
# ---------------------------------------------------------------------------
SMOKE_DIR="$EVSL_SRC/TESTS/PLanN"
if [ -d "$SMOKE_DIR" ]; then
    ( cd "$SMOKE_DIR" && make ) || echo "WARNING: smoke driver build failed (library still usable)."
    SMOKE_EXE="$SMOKE_DIR/LapPLanN_MatFree.ex"
    if [ -x "$SMOKE_EXE" ]; then
        # The .ex links -levsl; ensure the loader can find the shared lib.
        if [ "$OS" = "Darwin" ]; then
            DYLD_LIBRARY_PATH="$EVSL_LIB_DIR:${DYLD_LIBRARY_PATH:-}" "$SMOKE_EXE" || \
                echo "WARNING: smoke run returned nonzero; inspect output above."
        else
            LD_LIBRARY_PATH="$EVSL_LIB_DIR:${LD_LIBRARY_PATH:-}" "$SMOKE_EXE" || \
                echo "WARNING: smoke run returned nonzero; inspect output above."
        fi
    fi
fi

set +x

cat <<EOF

============================================
✓ EVSL build complete!
============================================

Shared library: $EVSL_LIB
Library dir:    $EVSL_LIB_DIR
Header:         $EVSL_INC

To use the EVSL adapter, point it at the shared library. The adapter searches
(in order): the SCI_SOLVERS_EVSL_LIB env var, the system loader path, then this
build tree. The simplest setup:

    export SCI_SOLVERS_EVSL_LIB="$EVSL_LIB"

On Linux you may also need the library dir on the loader path at runtime:

    export LD_LIBRARY_PATH="$EVSL_LIB_DIR:\${LD_LIBRARY_PATH:-}"

To build against a custom BLAS/LAPACK (either platform), point the build at it:

    # a site openblas (one library exporting BLAS + LAPACK)
    BLAS_LIB_DIR=/opt/openblas/lib ./build_evsl.sh
    # or the OPENBLAS_ROOT shorthand (probes lib/ then lib64/)
    OPENBLAS_ROOT=/opt/openblas ./build_evsl.sh
    # split BLAS and LAPACK libraries / dirs
    BLAS_LIB_DIR=/a BLAS_LIB_NAME=blas \\
    LAPACK_LIB_DIR=/b LAPACK_LIB_NAME=lapack ./build_evsl.sh

The whole evsl_workdir/ lives under build/ which is gitignored, so nothing here
shows up in 'git status'.
EOF
