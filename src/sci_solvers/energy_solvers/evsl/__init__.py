"""Matrix-free EVSL (spectrum-slicing Lanczos) bindings and interval solver.

EVSL (https://github.com/eigs/EVSL) is a matrix-free polynomial-filter Lanczos
eigensolver built for *interior* eigenvalues: instead of the lowest ``nroots``,
it finds every eigenvalue in a requested energy window ``[lo, hi]``. That is the
natural fit for a determinant-space CI Hamiltonian, where we often want the
level structure in a band rather than just the extremal states.

The pipeline is DOS-first, matching EVSL's own design:

* :func:`spectral_density` runs the kernel-polynomial DOS (``kpmdos``). This is
  the cheap go/no-go: it tells you how dense the spectrum is in the target
  window *before* any eigenpair work. If the levels there are too dense for any
  polynomial filter to separate, the DOS itself (or a thermal average from it)
  is the deliverable.
* :func:`slice_boundaries` turns that DOS into ``n_slices`` sub-intervals of
  roughly equal eigenvalue count (``spslicer``).
* :func:`solve_slice` diagonalizes one slice with thick-restart Chebyshev
  Lanczos (``ChebLanTr``), returning the eigenpairs in that interval.
* :func:`solve_sliced` loops the slices, concatenates, deduplicates the
  boundary-straddling roots, and returns a single :class:`EigenResult`.

Everything is matrix-free: EVSL only ever sees ``problem.matvec`` through a C
callback, so the interleaved-Wick/spin-block parity handled inside
:class:`~sci_solvers.eigen_problem.EigenProblem` never crosses the boundary.

Only the standard (not generalized) real-symmetric path is bound. EVSL as
released is single-node (OpenMP only); nothing here assumes MPI.

Loading the shared library
--------------------------
EVSL is an optional native dependency. Build it with
``scripts/energy_solvers/build_evsl.sh`` and point the loader at the result via
the ``SCI_SOLVERS_EVSL_LIB`` environment variable (or place ``libevsl`` on the
system loader path). :func:`load_evsl` resolves the library lazily, so importing
this package never fails just because EVSL is absent.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from sci_solvers.energy_solvers.evsl.bindings import (
    MATVEC,
    EvslError,
    EvslLib,
    load_evsl,
)
from sci_solvers.energy_solvers.evsl.solver import (
    SliceSpec,
    slice_boundaries,
    solve_slice,
    solve_sliced,
    spectral_density,
)

__all__ = [
    "MATVEC",
    "EvslError",
    "EvslLib",
    "load_evsl",
    "SliceSpec",
    "spectral_density",
    "slice_boundaries",
    "solve_slice",
    "solve_sliced",
]
