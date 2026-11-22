"""Build definition for the C++ / pybind11 kernel extension.

Package metadata (name, version, dependencies, package discovery) lives in
``pyproject.toml``; this module contributes *only* the compiled extension. The
repository-root ``setup.py`` imports :func:`ext_modules` from here so a standard
``uv sync`` (or ``uv pip install -e .``) builds it through the normal build
backend -- there is no hand-run ``build_ext`` step.

The extension is built under its fully-qualified package name so an editable
(``src``-on-path) install resolves it directly::

    sci_solvers.energy_solvers.truncated_davidson_solver.cpp.kernels_cpp

Developer note: an editable install does NOT recompile the extension when
``kernels.cpp`` changes. After editing the C++ source, rebuild explicitly with
``uv pip install -e . --reinstall-package sci_solvers``.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os
import shutil
import subprocess
import sys
import sysconfig

from pybind11.setup_helpers import Pybind11Extension

_KERNEL = "sci_solvers.energy_solvers.truncated_davidson_solver.cpp.kernels_cpp"
_SOURCE = "src/sci_solvers/energy_solvers/truncated_davidson_solver/cpp/kernels.cpp"

# Fallback C/C++ compilers to try, in order, when the one recorded in the
# interpreter's sysconfig is not on PATH.
_CC_FALLBACKS = ("cc", "gcc", "clang")
_CXX_FALLBACKS = ("c++", "g++", "clang++")


def _first_on_path(candidates: tuple[str, ...]) -> str | None:
    """First candidate resolvable on PATH (its first whitespace-split token)."""
    for cand in candidates:
        if cand and shutil.which(cand.split()[0]):
            return cand
    return None


def _resolve_compilers() -> None:
    """Point the build at an available compiler when sysconfig's is missing.

    setuptools compiles the extension with the ``CC``/``CXX`` recorded in the
    interpreter's sysconfig (falling back to environment variables of the same
    name, which take precedence). ``uv``-managed python-build-standalone
    interpreters record ``clang``/``clang++`` even on hosts that only have
    ``gcc``/``g++`` installed, so the build fails with "No such file or
    directory: 'clang++'". When the recorded (or already-overridden) compiler
    is not on PATH, fall back to the first available common alternative by
    exporting ``CC``/``CXX`` -- setuptools honours those over sysconfig.

    Respects an explicit ``CC``/``CXX`` the caller set if it resolves; only
    replaces a value whose executable cannot be found.
    """
    for var, fallbacks in (("CC", _CC_FALLBACKS), ("CXX", _CXX_FALLBACKS)):
        recorded = os.environ.get(var) or sysconfig.get_config_var(var) or ""
        if recorded and shutil.which(recorded.split()[0]):
            continue  # the recorded compiler is available -- leave it alone
        chosen = _first_on_path(fallbacks)
        if chosen is not None:
            os.environ[var] = chosen
            print(
                f"WARNING: recorded {var}={recorded!r} not found on PATH -- "
                f"falling back to {var}={chosen!r} for the C++ kernel build.",
                file=sys.stderr,
            )


def _omp_flags() -> tuple[list[str], list[str]]:
    """Return (compile_flags, link_flags) enabling OpenMP for the host platform.

    macOS ships a clang that rejects ``-fopenmp``; we use libomp from Homebrew
    instead. Linux GCC/clang accept ``-fopenmp`` directly.
    """
    if sys.platform != "darwin":
        return ["-fopenmp"], ["-fopenmp"]

    # On macOS, Apple clang rejects -fopenmp. We need the include path from
    # Homebrew libomp to parse #pragma omp directives, but we do NOT link
    # -lomp explicitly: the OpenMP runtime (libomp.dylib) is already loaded
    # by torch/numba at process startup, so linking our own copy would create
    # a duplicate-runtime abort. We use -undefined dynamic_lookup to defer
    # symbol resolution to whatever omp runtime is already in the process.
    try:
        prefix = (
            subprocess
            .check_output(["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001 -- brew missing/failed: build without OpenMP
        print(
            "WARNING: brew libomp not found -- building WITHOUT OpenMP. "
            "Install with: brew install libomp",
            file=sys.stderr,
        )
        return [], []

    compile_flags = [f"-I{prefix}/include", "-Xpreprocessor", "-fopenmp"]
    # No -lomp: reuse the runtime already loaded by torch/numba.
    link_flags = ["-undefined", "dynamic_lookup"]
    return compile_flags, link_flags


def ext_modules() -> list[Pybind11Extension]:
    """Return the list of pybind11 extensions to build."""
    _resolve_compilers()
    omp_compile, omp_link = _omp_flags()
    return [
        Pybind11Extension(
            _KERNEL,
            [_SOURCE],
            cxx_std=17,
            extra_compile_args=["-O3"] + omp_compile,
            extra_link_args=omp_link,
        ),
    ]
