#!/bin/bash
# (C) Copyright IBM Corp. 2026
#
# SPDX-License-Identifier: Apache-2.0
#
# Apple clang++ with OpenMP, used as CXX to build fulqrum on macOS against torch's OpenMP
# runtime, so the process only ever loads torch/lib/libomp.dylib (two OpenMP runtimes in one
# process abort with "OMP: Error #15"). Usage:
#   CXX="$PWD/scripts/energy_solvers/clang_omp.sh" uv sync --extra fulqrum
# pixi sets CXX itself (see [tool.pixi] in pyproject.toml).
#
# fulqrum is built without build isolation ([tool.uv] no-build-isolation-package), so uv
# installs torch first and runs this inside the target venv ($VIRTUAL_ENV). There:
# - fulqrum's setup.py passes a bare -fopenmp, which Apple clang rejects; it is forwarded to
#   the frontend instead, with torch's omp.h.
# - linking needs a libomp whose install name is @rpath/libomp.dylib; the linker records that
#   name in fulqrum's extensions. torch's copy is named /opt/llvm-openmp/lib/libomp.dylib (the
#   path it was built at, which doesn't exist here), so we link against a renamed copy of it.
# - the rpath @loader_path/../../torch/lib then resolves @rpath/libomp.dylib from
#   fulqrum/<pkg>/*.so to torch's libomp in the same site-packages. It goes first: dyld tries
#   rpaths in order, and a conda python's link flags add <env>/lib, holding conda's libomp.
# Anything else (no bare -fopenmp, or no torch in the env) passes through to Apple clang.

torch_dir="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)' \
    2>/dev/null)"
args=()
omp=0
link=1
prev=
for a in "$@"; do
    if [[ "$a" == -fopenmp && "$prev" != -X* && -n "$torch_dir" ]]; then
        args+=(-Xclang -fopenmp -I"$torch_dir/include")
        omp=1
    else
        args+=("$a")
    fi
    [[ "$a" == -c ]] && link=0
    prev="$a"
done
if ((omp && link)); then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    cp "$torch_dir/lib/libomp.dylib" "$tmp/libomp.dylib"
    install_name_tool -id @rpath/libomp.dylib "$tmp/libomp.dylib" 2>/dev/null
    args=(-Wl,-rpath,@loader_path/../../torch/lib "$tmp/libomp.dylib" "${args[@]}")
fi
/usr/bin/clang++ "${args[@]}"
