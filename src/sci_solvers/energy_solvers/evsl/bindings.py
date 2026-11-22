"""Thin ``ctypes`` wrapper around the EVSL shared library.

This module binds only the standard-problem, matrix-free surface of EVSL:
global init/teardown, the matrix-free matvec registration, the DOS
(``kpmdos`` + ``spslicer``), and the thick-restart Chebyshev-Lanczos interval
solve (``set_pol_def`` / ``find_pol`` / ``free_pol`` / ``ChebLanTr``), plus
``LanTrbounds`` for spectral-bound estimation and ``StatsPrint`` for timing.

The function *names and signatures below are verified against the upstream
headers* (``INC/evsl.h`` and ``INC/struct.h`` at commit ``b9d8150``):

* init/teardown are ``EVSLStart`` / ``EVSLFinish`` (no underscore — the plan's
  ``EVSL_Start`` does not exist in the source).
* the matvec callback typedef is ``void (*MVFunc)(double *x, double *y,
  void *data)`` — argument order ``(x, y, data)``.
* ``ChebLanTr`` allocates its output arrays (eigenvalues, vectors, residuals)
  with EVSL's own allocator; the caller copies them into numpy and frees them
  through EVSL, never through Python.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import ctypes
import os
import platform
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_double,
    c_int,
    c_void_p,
)
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Matvec callback typedef: void (*MVFunc)(double *x, double *y, void *data)
# ---------------------------------------------------------------------------
MATVEC = CFUNCTYPE(None, POINTER(c_double), POINTER(c_double), c_void_p)


class polparams(Structure):  # noqa: N801 - mirrors the C struct name exactly
    """ctypes mirror of EVSL's ``polparams`` (``INC/struct.h``).

    ``set_pol_def`` fills the input fields with defaults and ``find_pol`` fills
    the output fields; the caller only allocates the struct (it is a plain
    value type, not opaque) and passes it by reference. The field order and
    types match the header exactly — an incorrect layout silently corrupts the
    filter parameters, so this must not drift from the source.
    """

    _fields_ = [
        # ---- input to find_pol ----
        ("max_deg", c_int),
        ("min_deg", c_int),
        ("damping", c_int),
        ("thresh_ext", c_double),
        ("thresh_int", c_double),
        ("intvtol", c_double),
        # ---- output from find_pol ----
        ("type", c_int),
        ("mu", POINTER(c_double)),
        ("cc", c_double),
        ("dd", c_double),
        ("gam", c_double),
        ("bar", c_double),
        # ---- both input and output ----
        ("deg", c_int),
    ]


class EvslError(RuntimeError):
    """Raised when an EVSL routine returns a nonzero error code or is missing."""


def _candidate_lib_names() -> tuple[str, ...]:
    """Return platform-appropriate ``libevsl`` filenames for the loader."""
    if platform.system() == "Darwin":
        return ("libevsl.dylib", "libevsl-1.1.1.dylib")
    return ("libevsl.so", "libevsl-1.1.1.so")


def _build_tree_candidates() -> list[Path]:
    """Return likely ``libevsl`` paths inside the repo's build tree.

    ``build_evsl.sh`` installs the shared library under
    ``build/energy_solvers/evsl_workdir/EVSL/EVSL_1.1.1/EVSL/lib``. We walk up
    from this file to the repo root and probe that location so a freshly built
    EVSL is found without any environment setup.
    """
    here = Path(__file__).resolve()
    # src/sci_solvers/energy_solvers/evsl/bindings.py -> repo root is 5 up.
    for parent in here.parents:
        build = parent / "build" / "energy_solvers" / "evsl_workdir"
        if build.is_dir():
            return [p for name in _candidate_lib_names() for p in build.rglob(name)]
    return []


def _find_library() -> str:
    """Locate the EVSL shared library.

    Search order: the ``SCI_SOLVERS_EVSL_LIB`` env var (an explicit path), then
    the bare library names on the system loader path, then the repo build tree
    populated by ``build_evsl.sh``.

    Raises:
        EvslError: If no ``libevsl`` can be located.
    """
    explicit = os.environ.get("SCI_SOLVERS_EVSL_LIB")
    if explicit:
        return explicit

    names = _candidate_lib_names()
    tried: list[str] = list(names)
    for name in names:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            continue

    for path in _build_tree_candidates():
        return str(path)

    raise EvslError(
        "Could not locate the EVSL shared library. Build it with "
        "scripts/energy_solvers/build_evsl.sh and set SCI_SOLVERS_EVSL_LIB to "
        f"the resulting libevsl path (tried loader names: {tried})."
    )


class EvslLib:
    """One-instance-per-process ``ctypes`` handle to EVSL.

    EVSL keeps global state (the registered matvec, the current problem type),
    so a process must use a single :class:`EvslLib` and bracket work between
    :meth:`start` and :meth:`finish`. The instance owns the live callback
    object: the callback *must* be stored on ``self`` for as long as EVSL might
    call it, or Python garbage-collects it and the next matvec segfaults.

    Args:
        libpath: Explicit path to ``libevsl``. If omitted, :func:`_find_library`
            resolves it from the env var / loader path / repo build tree.
    """

    def __init__(self, libpath: str | None = None):
        """Load the library and declare argument/return types for bound symbols."""
        self.libpath = libpath or _find_library()
        self._lib = ctypes.CDLL(self.libpath)
        self._cb: "ctypes._FuncPointer | None" = None  # keep callback alive
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        """Set ``argtypes``/``restype`` for every symbol we call.

        ctypes defaults every argument and return to ``int``; declaring the real
        signatures is what makes pointer/double arguments marshal correctly.
        """
        lib = self._lib
        dp = POINTER(c_double)
        dpp = POINTER(POINTER(c_double))

        lib.EVSLStart.argtypes = []
        lib.EVSLStart.restype = c_int
        lib.EVSLFinish.argtypes = []
        lib.EVSLFinish.restype = c_int

        lib.SetAMatvec.argtypes = [c_int, MATVEC, c_void_p]
        lib.SetAMatvec.restype = c_int
        lib.SetStdEig.argtypes = []
        lib.SetStdEig.restype = c_int

        # int LanTrbounds(int lanm, int maxit, double tol, double *vinit,
        #                 int bndtype, double *lammin, double *lammax, FILE*)
        lib.LanTrbounds.argtypes = [c_int, c_int, c_double, dp, c_int, dp, dp, c_void_p]
        lib.LanTrbounds.restype = c_int

        # int kpmdos(int Mdeg, int damping, int nvec, double *ab,
        #            double *mu, double *ecnt)
        lib.kpmdos.argtypes = [c_int, c_int, c_int, dp, dp, dp]
        lib.kpmdos.restype = c_int

        # int spslicer(double *sli, double *mu, int Mdeg, double *intv,
        #              int n_int, int npts)
        lib.spslicer.argtypes = [dp, dp, c_int, dp, c_int, c_int]
        lib.spslicer.restype = c_int

        lib.StatsPrint.argtypes = [c_void_p]
        lib.StatsPrint.restype = None

        lib.set_pol_def.argtypes = [POINTER(polparams)]
        lib.set_pol_def.restype = None
        lib.find_pol.argtypes = [dp, POINTER(polparams)]
        lib.find_pol.restype = c_int
        lib.free_pol.argtypes = [POINTER(polparams)]
        lib.free_pol.restype = None

        # int ChebLanTr(int lanm, int nev, double *intv, int maxit, double tol,
        #               double *vinit, polparams *pol, int *nev2,
        #               double **vals, double **W, double **resW, FILE*)
        lib.ChebLanTr.argtypes = [
            c_int,
            c_int,
            dp,
            c_int,
            c_double,
            dp,
            POINTER(polparams),
            POINTER(c_int),
            dpp,
            dpp,
            dpp,
            c_void_p,
        ]
        lib.ChebLanTr.restype = c_int

        # EVSL's own allocator frees the ChebLanTr output arrays. `evsl_Free`
        # is the public free; fall back to libc free if the symbol name differs.
        self._evsl_free = getattr(lib, "evsl_Free", None)
        if self._evsl_free is not None:
            self._evsl_free.argtypes = [c_void_p]
            self._evsl_free.restype = None

    # ---- global lifecycle ------------------------------------------------
    def start(self) -> None:
        """Initialize EVSL global state (``EVSLStart``)."""
        self._check(self._lib.EVSLStart(), "EVSLStart")

    def finish(self) -> None:
        """Tear down EVSL global state (``EVSLFinish``)."""
        self._check(self._lib.EVSLFinish(), "EVSLFinish")

    def set_std_eig(self) -> None:
        """Select the standard (non-generalized) eigenproblem (``SetStdEig``)."""
        self._check(self._lib.SetStdEig(), "SetStdEig")

    # ---- matrix-free operator registration -------------------------------
    def set_matvec(self, n: int, callback: "ctypes._FuncPointer") -> None:
        """Register a matrix-free operator (``SetAMatvec``).

        The callback object is stored on ``self`` so Python does not collect it
        while EVSL still holds the pointer.
        """
        self._cb = callback  # anchor: a local would be GC'd and segfault
        self._check(self._lib.SetAMatvec(n, callback, None), "SetAMatvec")

    # ---- thin passthroughs (numpy in, error-checked) ---------------------
    def lantr_bounds(self, n: int, lanm: int, maxit: int, tol: float) -> tuple[float, float]:
        """Estimate ``[lambda_min, lambda_max]`` via ``LanTrbounds``.

        A random ``vinit`` seeds the Lanczos run. ``bndtype=2`` requests the
        wider (safer) of EVSL's two bound estimators so the DOS interval fully
        brackets the spectrum.
        """
        rng = np.random.default_rng(0)
        vinit = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)
        lammin = c_double(0.0)
        lammax = c_double(0.0)
        self._check(
            self._lib.LanTrbounds(
                lanm, maxit, tol, _dp(vinit), 2, byref(lammin), byref(lammax), None
            ),
            "LanTrbounds",
        )
        return lammin.value, lammax.value

    def kpmdos(
        self, mdeg: int, damping: int, nvec: int, xintv: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Kernel-polynomial DOS (``kpmdos``).

        Returns the ``Mdeg+1`` Chebyshev moments ``mu`` and the estimated
        eigenvalue count ``ecount`` in ``[xintv[0], xintv[1]]``.
        """
        mu = np.zeros(mdeg + 1, dtype=np.float64)
        ecount = c_double(0.0)
        self._check(
            self._lib.kpmdos(mdeg, damping, nvec, _dp(xintv), _dp(mu), byref(ecount)),
            "kpmdos",
        )
        return mu, ecount.value

    def spslicer(
        self, mu: np.ndarray, mdeg: int, xintv: np.ndarray, n_int: int, npts: int
    ) -> np.ndarray:
        """Cut the DOS into ``n_int`` equal-count slices (``spslicer``).

        Returns ``n_int + 1`` slice boundaries.
        """
        sli = np.zeros(n_int + 1, dtype=np.float64)
        self._check(
            self._lib.spslicer(_dp(sli), _dp(mu), mdeg, _dp(xintv), n_int, npts),
            "spslicer",
        )
        return sli

    def stats_print(self) -> None:
        """Print EVSL's internal timing breakdown to stdout (``StatsPrint``).

        Unlike the interval/DOS routines (which accept a ``NULL`` ``FILE*`` and
        silently skip logging), ``StatsPrint`` unconditionally dereferences its
        stream, so ``NULL`` segfaults. We hand it a real ``FILE*`` for the C
        stdout obtained from libc.
        """
        stream = _c_stdout()
        if stream is None:  # no usable stdout handle; skip rather than crash
            return
        self._lib.StatsPrint(stream)

    # ---- polynomial filter -----------------------------------------------
    def make_pol(
        self,
        xintv: np.ndarray,
        *,
        damping: int = 1,
        thresh_int: float = 0.8,
        thresh_ext: float = 0.2,
        max_deg: int = 300,
    ) -> polparams:
        """Build a filter polynomial for ``xintv`` (``set_pol_def`` + ``find_pol``).

        The returned struct must be released with :meth:`free_pol`. Defaults
        follow the upstream restarted-Lanczos driver (``LapPLanR``).
        """
        pol = polparams()
        self._lib.set_pol_def(byref(pol))
        pol.damping = damping
        pol.thresh_int = thresh_int
        pol.thresh_ext = thresh_ext
        pol.max_deg = max_deg
        self._check(self._lib.find_pol(_dp(xintv), byref(pol)), "find_pol")
        return pol

    def free_pol(self, pol: polparams) -> None:
        """Release a filter polynomial built by :meth:`make_pol` (``free_pol``)."""
        self._lib.free_pol(byref(pol))

    # ---- interval solve --------------------------------------------------
    def cheb_lan_tr(
        self,
        lanm: int,
        nev: int,
        xintv: np.ndarray,
        maxit: int,
        tol: float,
        vinit: np.ndarray,
        pol: polparams,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Thick-restart Chebyshev-Lanczos interval solve (``ChebLanTr``).

        Finds every eigenpair in ``[xintv[0], xintv[1]]``. EVSL allocates the
        output eigenvalue, eigenvector, and residual arrays; this method copies
        each into an owned numpy array and frees the EVSL buffers, so the
        returned arrays are safe to keep after :meth:`finish`.

        Returns:
            ``(vals, vecs, res)`` with ``vals`` shape ``(nev2,)``, ``vecs``
            shape ``(n, nev2)`` (column ``j`` is eigenvector ``j``), and ``res``
            the per-root residual norms.
        """
        nev2 = c_int(0)
        vals_p = POINTER(c_double)()
        vecs_p = POINTER(c_double)()
        res_p = POINTER(c_double)()
        self._check(
            self._lib.ChebLanTr(
                lanm,
                nev,
                _dp(xintv),
                maxit,
                tol,
                _dp(vinit),
                byref(pol),
                byref(nev2),
                byref(vals_p),
                byref(vecs_p),
                byref(res_p),
                None,
            ),
            "ChebLanTr",
        )
        k = nev2.value
        if k <= 0:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((n, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        # Copy out of EVSL-owned memory before freeing it.
        vals = np.ctypeslib.as_array(vals_p, shape=(k,)).copy()
        # W is column-major: k eigenvectors of length n laid out contiguously.
        vecs = np.ctypeslib.as_array(vecs_p, shape=(k, n)).copy().T
        res = np.ctypeslib.as_array(res_p, shape=(k,)).copy()
        self._free(vals_p)
        self._free(vecs_p)
        self._free(res_p)
        return vals, np.ascontiguousarray(vecs), res

    # ---- helpers ---------------------------------------------------------
    def _free(self, ptr: "ctypes._Pointer") -> None:
        """Free an EVSL-allocated buffer through EVSL's allocator when available."""
        if self._evsl_free is not None:
            self._evsl_free(ctypes.cast(ptr, c_void_p))

    @staticmethod
    def _check(code: int, name: str) -> None:
        """Raise :class:`EvslError` on a nonzero EVSL return code."""
        if code:
            raise EvslError(f"{name} returned nonzero error code {code}.")


def _dp(arr: np.ndarray) -> "ctypes._Pointer":
    """Return a ``double *`` for a contiguous float64 numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(POINTER(c_double))


def _c_stdout() -> "c_void_p | None":
    """Return the C-runtime ``FILE*`` for stdout, or ``None`` if unavailable.

    ``StatsPrint`` needs a non-NULL stream. libc exposes stdout differently per
    platform: on glibc it is the ``stdout`` symbol; on macOS it is ``__stdoutp``.
    We probe both and fall back to ``fdopen(1, "w")`` if neither resolves.
    """
    libc = ctypes.CDLL(None)
    for name in ("__stdoutp", "stdout"):
        try:
            return c_void_p.in_dll(libc, name)  # type: ignore[arg-type]
        except (ValueError, AttributeError):
            continue
    try:
        libc.fdopen.restype = c_void_p
        libc.fdopen.argtypes = [c_int, ctypes.c_char_p]
        handle = libc.fdopen(1, b"w")
        return c_void_p(handle) if handle else None
    except (AttributeError, OSError):
        return None


def load_evsl(libpath: str | None = None) -> EvslLib:
    """Load EVSL and return a ready :class:`EvslLib` (does not call ``start``)."""
    return EvslLib(libpath)
