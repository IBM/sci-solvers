#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
# Run the checks of .github/workflows/ci.yml locally, in the same order and with the same
# commands, so a branch can be checked before it is pushed. Keep the two files in sync.
#
# Usage: scripts/local_ci.sh [--gpu] [--workers N]
#
#   --gpu      Also cover what CI cannot: install CUDA torch and a GPU build of sbd
#              (`--extra sbd-gpu` instead of `--extra cpu`) and run the whole suite,
#              slow tests included. Needs an NVIDIA GPU, the NVIDIA HPC SDK (nvc++) and an
#              MPI, found by scripts/energy_solvers/build_sbd.sh. An SDK installed with
#              scripts/energy_solvers/install_nvhpc.sh is found automatically; otherwise
#              set NVHPC_HOME (or put nvc++ on PATH) and MPI_HOME (or put mpicxx on PATH).
#   --workers  pytest-xdist workers (default 4).
#
# Like CI, the tests run in their own environment (build/local_ci_venv, or
# build/local_ci_gpu_venv with --gpu), so the default .venv is left alone. CI's
# `pytest -n auto` starts one worker per CPU; on a shared machine that can be hundreds, so
# here the worker count is fixed and each worker is limited to one BLAS/OpenMP/numba thread.
set -uo pipefail

gpu=0
workers=4
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) gpu=1; shift ;;
        --workers) workers="$2"; shift 2 ;;
        -h|--help) sed -n '6,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# CI pins the interpreter too (uv would otherwise pick the newest release).
export UV_PYTHON="${UV_PYTHON:-3.13}"

if [ "$gpu" = 1 ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
        echo "--gpu needs a working NVIDIA GPU (nvidia-smi failed)" >&2
        exit 2
    fi
    export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/build/local_ci_gpu_venv"
    # Locates the HPC SDK and MPI, and writes the env file the sbd build and runtime need.
    if ! bash scripts/energy_solvers/build_sbd.sh; then
        echo "scripts/energy_solvers/build_sbd.sh failed; see its output above" >&2
        exit 2
    fi
    sbd_env="${ENERGY_SOLVERS_BUILD_ROOT:-$REPO_ROOT/build/energy_solvers}/sbd_env.sh"
    # shellcheck disable=SC1090
    source "$sbd_env"
    extras=(--extra sbd-gpu --extra benchmark)
    tests=(pytest -n "$workers")
else
    export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/build/local_ci_venv"
    extras=(--extra cpu --extra benchmark)
    tests=(pytest -m "not slow" -n "$workers" -v --durations=15)
fi

failed=()
step() {  # step <name> <command...>
    local name="$1"
    shift
    echo "==> $name"
    if "$@"; then echo "    ok"; else echo "    FAILED"; failed+=("$name"); fi
}

# lint job
step "License headers" bash -c "python3 scripts/check_license_headers.py \$(git ls-files '*.py' '*.sh')"
step "Ruff" uvx ruff@0.17.3 check
step "Ruff format" uvx ruff@0.17.3 format --check

# test job
# Every `uv run` gets the same extras as `uv sync`: without them it re-syncs the venv to the
# default install and swaps the torch the extras chose (e.g. CUDA 12 torch for a CUDA 13 one).
step "Install" uv sync "${extras[@]}"
step "mypy" uv run "${extras[@]}" mypy
if [ "$gpu" = 1 ]; then
    # Again, now that torch is installed: the env file puts the venv's CUDA runtime ahead of
    # the HPC SDK's older one only if it exists when sourced, which a fresh venv's does not.
    # shellcheck disable=SC1090
    source "$sbd_env"
    # Fail before the long run if torch cannot see the GPU, rather than skip every GPU test.
    step "CUDA visible to torch" uv run "${extras[@]}" python -c \
        "import torch, sys; print('torch', torch.__version__); sys.exit(not torch.cuda.is_available())"
fi
step "Tests" uv run "${extras[@]}" "${tests[@]}"

if [ ${#failed[@]} -eq 0 ]; then
    echo "All checks passed."
else
    echo "Failed: ${failed[*]}"
    exit 1
fi
