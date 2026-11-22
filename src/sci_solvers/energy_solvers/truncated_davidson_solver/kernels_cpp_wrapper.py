"""
kernels_cpp_wrapper.py
======================

Thin Python adapter that presents the same API surface as numba_kernels.py
but delegates to the compiled C++ extension (kernels_cpp.so).

Both the C++ extension and the numba path take the sorted integral tables as
dense 3-D numpy arrays same_a[i, j, :] etc.; the C++ side indexes them by
pointer arithmetic (fixed row width per table), so no per-cell adapter or copy
is needed.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os

# Prevent abort when torch/numba have already loaded a copy of libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from sci_solvers.energy_solvers.truncated_davidson_solver.cpp import (  # type: ignore[attr-defined]
        kernels_cpp as _cpp,
    )
except ImportError as exc:
    # The C++ extension is OPTIONAL: the build degrades to "not compiled" when
    # no C++ compiler is available or compilation fails (see the root setup.py),
    # so reaching here means backend="cpp" was requested but the extension was
    # never built. Surface a clear, actionable message -- the default numba
    # backend needs none of this.
    raise ImportError(
        "The optional C++ kernel backend (backend='cpp') is not available: the "
        "compiled extension 'kernels_cpp' was not built for this install. This "
        "happens when no C++ compiler was found at build time (or it failed to "
        "compile) -- the install still succeeds because the default numba "
        "backend needs no compiler.\n"
        "To enable the C++ backend, ensure a C++ compiler (g++/clang++) is on "
        "PATH and rebuild:\n"
        "    uv pip install -e . --reinstall-package sci_solvers\n"
        "  (set CXX=... to pick a specific compiler). Or simply use the default "
        "backend='numba', which is equivalent in results."
    ) from exc

EMPTY = _cpp.EMPTY


# ---------------------------------------------------------------------------
# Public API (mirrors numba_kernels.py)
# ---------------------------------------------------------------------------


def make_registry():
    """Return an opaque C++ registry object (replaces NumbaDict registry)."""
    return _cpp.make_registry()


def build_addr_maps(registry, bits_a, bits_b):
    """Build unordered_map address tables in the registry for norb > 24 lookups."""
    _cpp.build_addr_maps(registry, bits_a, bits_b)


def add_basis_vector_fused(
    new_idx,
    new_val,
    n_new,
    registry,
    norb,
    nb,
    nelec_a,
    nelec_b,
    bits_a,
    bits_b,
    addr_table_a,
    addr_table_b,
    use_dense_addr,
    h1e,
    eri,
    use_dense_hdiag,
    hdiag,
    hdiag1,
    jdiag,
    kdiag,
    col,
):
    """Fuse a new basis vector into the registry and fill ``col`` (mirrors numba_kernels)."""
    _cpp.add_basis_vector_fused(
        new_idx,
        new_val,
        int(n_new),
        registry,
        int(norb),
        int(nb),
        int(nelec_a),
        int(nelec_b),
        bits_a,
        bits_b,
        addr_table_a,
        addr_table_b,
        bool(use_dense_addr),
        h1e,
        eri,
        bool(use_dense_hdiag),
        hdiag,
        hdiag1,
        jdiag,
        kdiag,
        col,
    )


def calc_trunc_pcr_fused(
    psi_keys,
    psi_vals,
    norb,
    nb,
    nelec_a,
    nelec_b,
    bits_a,
    bits_b,
    addr_table_a,
    addr_table_b,
    use_dense_addr,
    registry,
    h1e,
    eri,
    use_dense_hdiag,
    hdiag,
    hdiag1,
    jdiag,
    kdiag,
    same_a,
    same_b,
    same_v,
    cross_a,
    cross_b,
    cross_v,
    E,
    delta,
    level_shift=1e-3,
    trunc_eps=1e-8,
):
    """Compute the truncated preconditioned correction residual (mirrors numba_kernels)."""
    result = _cpp.calc_trunc_pcr_fused(
        psi_keys,
        psi_vals,
        int(norb),
        int(nb),
        int(nelec_a),
        int(nelec_b),
        bits_a,
        bits_b,
        addr_table_a,
        addr_table_b,
        bool(use_dense_addr),
        registry,
        h1e,
        eri,
        bool(use_dense_hdiag),
        hdiag,
        hdiag1,
        jdiag,
        kdiag,
        same_a,  # dense [norb, norb, norb*(norb-1)] int64
        same_b,
        same_v,  # dense [norb, norb, norb*(norb-1)] float64
        cross_a,  # dense [norb, norb, norb*norb] int64
        cross_b,
        cross_v,  # dense [norb, norb, norb*norb] float64
        float(E),
        float(delta),
        float(level_shift),
        float(trunc_eps),
    )
    return result


def singles_kernel(occ, virt, occ_other, bits0, h1e, eri):
    """Enumerate all single excitations within one spin block (mirrors numba_kernels)."""
    return _cpp.singles_kernel(occ, virt, occ_other, int(bits0), h1e, eri)


def doubles_same_spin_kernel(occ, virt, bits0, eri):
    """Enumerate all same-spin double excitations within one spin block (mirrors numba_kernels)."""
    return _cpp.doubles_same_spin_kernel(occ, virt, int(bits0), eri)


def doubles_cross_spin_kernel(occ_a, virt_a, occ_b, virt_b, bits_a0, bits_b0, eri):
    """Enumerate all cross-spin (alpha+beta) double excitations (mirrors numba_kernels)."""
    return _cpp.doubles_cross_spin_kernel(
        occ_a, virt_a, occ_b, virt_b, int(bits_a0), int(bits_b0), eri
    )


def warmup(norb_hint: int = 4):
    """No-op: C++ extension has no JIT compilation overhead."""
    pass
