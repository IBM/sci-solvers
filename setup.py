"""Root build entry point: delegates the C++ extension definition to
``src/sci_solvers/energy_solvers/truncated_davidson_solver/cpp/setup.py``.

All package metadata lives in ``pyproject.toml``; this file exists only because
setuptools' ``build_ext`` command must be hooked from the project root. It
carries no build logic of its own -- see the ``cpp`` module for the extension
definition, OpenMP handling, and compiler fallback.

The C++ ``cpp`` backend is optional: the pure-Python/numba backend is the
default and needs no compiler. So if no C++ compiler is available, or the
extension fails to compile, we print a warning and finish the build *without*
the extension rather than failing the whole install. Requesting ``backend="cpp"``
at runtime then raises a clear error (see ``kernels_cpp_wrapper.py``).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import importlib.util
import sys
from pathlib import Path

from pybind11.setup_helpers import build_ext as _pybind_build_ext
from setuptools import setup

_CPP_SETUP = (
    Path(__file__).parent / "src/sci_solvers/energy_solvers/truncated_davidson_solver/cpp/setup.py"
)
_spec = importlib.util.spec_from_file_location("_td_cpp_build", _CPP_SETUP)
_cpp_build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cpp_build)


class _OptionalBuildExt(_pybind_build_ext):
    """Build the C++ kernel extension, but never fail the install if it can't.

    The ``cpp`` backend is opt-in and the numba backend covers the same API, so
    a missing compiler or a compile error degrades to "extension not built"
    (with a warning) instead of aborting the whole build. Extensions that fail
    to compile are dropped from ``self.extensions`` so the later inplace-copy
    step does not chase a ``.so`` that was never produced.
    """

    def build_extension(self, ext):  # noqa: D102 -- inherits base contract
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001 -- any build failure is non-fatal
            print(
                f"WARNING: could not build the optional C++ kernel extension "
                f"'{ext.name}': {exc}. Continuing without it -- the default "
                f"numba backend is unaffected; only backend='cpp' is disabled.",
                file=sys.stderr,
            )
            # Drop it so copy_extensions_to_source() / run() won't look for the
            # missing artifact and re-raise.
            self.extensions = [e for e in self.extensions if e is not ext]


try:
    _ext_modules = _cpp_build.ext_modules()
except Exception as exc:  # noqa: BLE001 -- e.g. no compiler resolvable at all
    print(
        f"WARNING: skipping the optional C++ kernel extension: {exc}. "
        f"The default numba backend is unaffected; only backend='cpp' is disabled.",
        file=sys.stderr,
    )
    _ext_modules = []

setup(ext_modules=_ext_modules, cmdclass={"build_ext": _OptionalBuildExt})
