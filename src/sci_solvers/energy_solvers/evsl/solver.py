"""DOS, slicing, and interval-solve driver on top of the EVSL bindings.

The four public entry points mirror the EVSL workflow and can be used
independently of the adapter layer:

* :func:`spectral_density` — kernel-polynomial DOS.
* :func:`slice_boundaries` — DOS -> equal-count slice boundaries.
* :func:`solve_slice` — one interval solved with Chebyshev-Lanczos.
* :func:`solve_sliced` — multi-slice driver returning a single
  :class:`~sci_solvers.eigen_problem.EigenResult`.

All of them consume the same matrix-free ``problem`` object the rest of the
package uses (anything exposing ``dim`` and ``matvec``); nothing here assembles
a matrix.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import ctypes
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sci_solvers.eigen_problem import EigenResult
from sci_solvers.energy_solvers.evsl.bindings import MATVEC, EvslLib, load_evsl

# Fraction of *available* system memory the eigenvectors may occupy before
# solve_sliced auto-switches to the on-disk (memmap) streaming path. Deriving the
# ceiling from live availability (rather than a fixed number of GB) keeps the
# switch streamlined across machines: on a beefy node with lots of free RAM the
# in-RAM path is used for larger problems, while on a laptop it engages sooner.
_STREAM_MEM_FRACTION = 0.5

# Fallback ceiling (GB) used only when available memory cannot be probed (no
# psutil, unknown platform). ~10^6 dim x ~10^4 roots is 80 GB, far above this.
_STREAM_FALLBACK_GB = 8.0


def _available_memory_gb() -> float | None:
    """Best-effort available system memory in GB, or ``None`` if unknowable.

    Tries ``psutil`` first, then a couple of stdlib probes (``os.sysconf`` on
    POSIX). Returns ``None`` when nothing works, so callers can fall back to a
    fixed ceiling rather than guess.
    """
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.virtual_memory().available) / 1e9
    except Exception:
        pass
    try:
        import os

        # Linux/macOS expose page size and available pages via sysconf.
        page = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        if page > 0 and avail > 0:
            return float(page) * float(avail) / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    return None


def _default_stream_threshold_gb() -> float:
    """Auto-trigger ceiling (GB): a fraction of available RAM, else the fallback.

    Used as the default ``stream_threshold_gb`` so a large solve on a big node
    streams only when its eigenvectors would actually threaten that node's free
    memory, without the caller hand-tuning a number.
    """
    avail = _available_memory_gb()
    if avail is None:
        return _STREAM_FALLBACK_GB
    return max(avail * _STREAM_MEM_FRACTION, 1.0)


def _make_callback(problem) -> "ctypes._FuncPointer":
    """Wrap ``problem.matvec`` as an EVSL ``MVFunc`` callback.

    The returned callback reads ``x`` and writes ``y`` in place through numpy
    views over the EVSL-provided buffers, so no ownership crosses the boundary.
    The caller must anchor the returned object (see :meth:`EvslLib.set_matvec`)
    or Python will collect it and the next matvec segfaults.
    """
    n = int(problem.dim)

    def cb(x_ptr, y_ptr, _data):  # matches MVFunc: (double*, double*, void*)
        x = np.ctypeslib.as_array(x_ptr, shape=(n,))
        y = np.ctypeslib.as_array(y_ptr, shape=(n,))
        y[:] = problem.matvec(np.ascontiguousarray(x))

    return MATVEC(cb)


def _register(problem, lib: EvslLib) -> "ctypes._FuncPointer":
    """Start EVSL, register the matrix-free operator, select the standard problem.

    Returns the anchored callback so the caller can keep it alive for the
    duration of the EVSL session.
    """
    lib.start()
    cb = _make_callback(problem)
    lib.set_matvec(int(problem.dim), cb)
    lib.set_std_eig()
    return cb


def _bounds(
    problem, lib: EvslLib, bounds: "tuple[float, float] | None", tol: float
) -> tuple[float, float]:
    """Return spectral bounds, estimating them with ``LanTrbounds`` if not given."""
    if bounds is not None:
        return float(bounds[0]), float(bounds[1])
    n = int(problem.dim)
    lanm = min(50, max(10, n - 1))
    return lib.lantr_bounds(n, lanm=lanm, maxit=200, tol=tol)


def _kpm_moments(
    lib: EvslLib, deg: int, n_vec: int, bounds: "tuple[float, float]"
) -> tuple[np.ndarray, float]:
    """Raw KPM Chebyshev moments + eigenvalue-count estimate over ``bounds``.

    Thin wrapper around ``kpmdos`` shared by :func:`spectral_density` and
    callers (like :class:`~sci_solvers.energy_solvers.general_algebra_solvers.evsl.EvslSolver`)
    that need the moments themselves to avoid recomputing an equivalent KPM-DOS
    pass a second time (each pass costs ``deg * n_vec`` matvecs).
    """
    lmin, lmax = float(bounds[0]), float(bounds[1])
    xintv = np.array([lmin, lmax, lmin, lmax], dtype=np.float64)
    return lib.kpmdos(deg, 1, n_vec, xintv)


def spectral_density(
    problem,
    n_points: int = 1000,
    deg: int = 300,
    n_vec: int = 100,
    bounds: "tuple[float, float] | None" = None,
    *,
    lib: EvslLib | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the spectral density (DOS) of ``problem`` via ``kpmdos``.

    This is the cheap decision point: it reveals how dense the spectrum is in
    the target window before any eigenpair work. If the levels there are denser
    than any filter can separate, the DOS itself (or a thermal average built
    from it) is the deliverable rather than eigenpairs.

    Args:
        problem: Matrix-free operator (needs ``dim`` and ``matvec``).
        n_points: Number of energy grid points at which the DOS is evaluated.
        deg: Chebyshev expansion degree for the DOS (higher = finer resolution).
        n_vec: Number of random probe vectors averaged in the KPM estimator.
        bounds: ``(lambda_min, lambda_max)``; estimated with ``LanTrbounds`` if
            omitted.
        lib: An existing :class:`EvslLib` to reuse; a fresh one is loaded and
            torn down here if omitted.

    Returns:
        ``(energies, dos)`` — the energy grid and the density of states on it,
        normalized so that ``trapezoid(dos, energies) == problem.dim`` (a proper
        DOS whose integral is the state count). Callers can integrate it directly
        for a cumulative eigenvalue count, or slice/threshold it as a density.
    """
    own = lib is None
    lib = lib or load_evsl()
    cb = None
    try:
        cb = _register(problem, lib)
        lmin, lmax = _bounds(problem, lib, bounds, tol=1e-8)
        mu, _ecount = _kpm_moments(lib, deg, n_vec, (lmin, lmax))
        energies, dos = _dos_from_moments(mu, lmin, lmax, n_points, int(problem.dim))
        return energies, dos
    finally:
        del cb  # drop the anchor before teardown
        if own:
            lib.finish()


def _dos_from_moments(
    mu: np.ndarray, lmin: float, lmax: float, n_points: int, dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a DOS curve from KPM Chebyshev moments on ``n_points`` grid.

    Maps ``[lmin, lmax]`` to Chebyshev's ``[-1, 1]``, evaluates the moment
    expansion ``sum_k mu_k T_k(x)`` with the standard ``1/(pi sqrt(1-x^2))``
    weight, clips tiny negative values from truncation, then rescales so the
    curve integrates to ``dim``. EVSL's ``kpmdos`` moments carry an internal
    normalization constant that is convention-dependent; fixing the integral to
    the known state count (every eigenvalue counted once) gives a DOS callers can
    integrate for a cumulative count without knowing that constant. The *shape*
    is what KPM estimates — the total is exact by construction — so this is a
    normalization, not a correction to the density.
    """
    c = 0.5 * (lmax + lmin)
    h = 0.5 * (lmax - lmin)
    # Interior grid (avoid the endpoints where the weight diverges).
    x = np.cos(np.pi * (np.arange(n_points) + 0.5) / n_points)
    energies = c + h * x
    k = np.arange(len(mu))
    # T_k(x) = cos(k * arccos(x))
    tk = np.cos(np.outer(np.arccos(np.clip(x, -1.0, 1.0)), k))
    dos = (tk @ mu) / (np.pi * np.sqrt(np.clip(1.0 - x * x, 1e-12, None)))
    dos = np.clip(dos, 0.0, None)
    order = np.argsort(energies)
    energies, dos = energies[order], dos[order]
    area = float(np.trapezoid(dos, energies))
    if area > 0:
        dos = dos * (dim / area)
    return energies, dos


def slice_boundaries(
    problem,
    n_slices: int,
    window: "tuple[float, float]",
    deg: int = 300,
    n_vec: int = 100,
    bounds: "tuple[float, float] | None" = None,
    *,
    lib: EvslLib | None = None,
) -> np.ndarray:
    """Return ``n_slices + 1`` boundaries splitting ``window`` by equal DOS count.

    Runs ``kpmdos`` over the full spectrum then ``spslicer`` restricted to
    ``window`` so each slice holds a roughly equal number of eigenvalues.

    Args:
        problem: Matrix-free operator.
        n_slices: Number of sub-intervals to cut ``window`` into.
        window: ``(lo, hi)`` energy range to slice.
        deg: DOS Chebyshev degree.
        n_vec: DOS probe-vector count.
        bounds: Full spectral bounds; estimated if omitted.
        lib: Reusable :class:`EvslLib`.

    Returns:
        Ascending array of ``n_slices + 1`` slice boundaries.
    """
    sli, _ecount = _slice_boundaries_with_count(
        problem, n_slices, window, deg=deg, n_vec=n_vec, bounds=bounds, lib=lib
    )
    return sli


def _slice_boundaries_with_count(
    problem,
    n_slices: int,
    window: "tuple[float, float]",
    *,
    deg: int = 300,
    n_vec: int = 100,
    bounds: "tuple[float, float] | None" = None,
    lib: EvslLib | None = None,
    dos_moments: "tuple[np.ndarray, float, int] | None" = None,
) -> tuple[np.ndarray, float]:
    """Like :func:`slice_boundaries`, but also return the DOS eigenvalue count.

    ``spslicer`` needs the ``kpmdos`` moments anyway, and that call yields an
    ``ecount`` estimate of how many eigenvalues fall in the window. The streaming
    driver uses that count to size the on-disk eigenvector buffer and to decide
    (by estimated footprint) whether to stream at all, so we surface it here
    instead of recomputing the DOS.

    Args:
        dos_moments: Optional ``(mu, ecount, deg_used)`` from a KPM-DOS pass a
            caller already ran over the same operator/bounds (e.g.
            :class:`~sci_solvers.energy_solvers.general_algebra_solvers.evsl.EvslSolver`'s
            own window-detection pass). When given, the ``kpmdos`` call here is
            skipped entirely (it would otherwise cost another ``deg * n_vec``
            matvecs recomputing an equivalent estimate) and ``spslicer`` runs
            against the supplied moments, using ``deg_used`` (not ``deg``) since
            that is the degree they were actually computed at. ``None`` (the
            default) reproduces the original behavior exactly.

    Returns:
        ``(boundaries, ecount)`` — ``n_slices + 1`` ascending boundaries and the
        estimated number of eigenvalues in ``window``.
    """
    own = lib is None
    lib = lib or load_evsl()
    cb = None
    try:
        cb = _register(problem, lib)
        lmin, lmax = _bounds(problem, lib, bounds, tol=1e-8)
        lo, hi = float(window[0]), float(window[1])
        if dos_moments is not None:
            mu, ecount, deg = dos_moments
        else:
            # kpmdos uses the FULL bounds; spslicer's window is [lo, hi].
            full = np.array([lmin, lmax, lmin, lmax], dtype=np.float64)
            mu, ecount = lib.kpmdos(deg, 1, n_vec, full)
        npts = max(2 * n_slices, int(10 * max(ecount, 1.0)))
        win = np.array([lo, hi, lmin, lmax], dtype=np.float64)
        return lib.spslicer(mu, deg, win, n_slices, npts), float(ecount)
    finally:
        del cb
        if own:
            lib.finish()


@dataclass
class SliceSpec:
    """Tuning parameters for a single-slice Chebyshev-Lanczos solve.

    Attributes:
        lo: Slice lower bound.
        hi: Slice upper bound.
        max_iter: Maximum Lanczos iterations (``ChebLanTr`` ``maxit``).
        lanczos_dim: Thick-restart Krylov subspace size (``ChebLanTr``'s
            ``lanm``). Rule of thumb: ~2x the expected roots in the slice; too
            small and the restart thrashes.
        tol: Residual tolerance for convergence.
        pol_deg_max: Upper bound on the filter polynomial degree (``find_pol``
            picks the actual degree from the slice width; narrow slices need
            high degree — watch the matvec count in ``StatsPrint``).
        nev: Expected root count passed to ``ChebLanTr`` as ``nev``. ``None``
            (default) falls back to ``lanczos_dim`` (the original behavior).
            Setting this independently of ``lanczos_dim`` matters when the
            true root count is much smaller than the Krylov size: EVSL's
            early-exit convergence check is gated by ``nev``, so leaving it as
            large as ``lanczos_dim`` forces every restart to run a full
            ``lanczos_dim``-step sweep before the shortcut can even fire.
    """

    lo: float
    hi: float
    max_iter: int = 3000
    lanczos_dim: int = 200
    tol: float = 1e-8
    pol_deg_max: int = 300
    nev: int | None = None


def solve_slice(
    problem,
    spec: SliceSpec,
    bounds: "tuple[float, float]",
    *,
    lib: EvslLib | None = None,
) -> EigenResult:
    """Diagonalize the eigenpairs of ``problem`` inside ``[spec.lo, spec.hi]``.

    Uses thick-restart Chebyshev Lanczos (``ChebLanTr``) — bounded memory,
    unlike the non-restart ``ChebLanNr``. Returns the roots in the slice sorted
    ascending; the eigenvectors are copied out of EVSL-owned memory.

    Args:
        problem: Matrix-free operator.
        spec: Slice bounds and Lanczos/filter tuning.
        bounds: Full ``(lambda_min, lambda_max)`` spectral bounds.
        lib: Reusable :class:`EvslLib`.

    Returns:
        The eigenpairs found in the slice, as an :class:`EigenResult`.
    """
    own = lib is None
    lib = lib or load_evsl()
    cb = None
    try:
        cb = _register(problem, lib)
        n = int(problem.dim)
        lmin, lmax = float(bounds[0]), float(bounds[1])
        xintv = np.array([spec.lo, spec.hi, lmin, lmax], dtype=np.float64)
        pol = lib.make_pol(xintv, max_deg=spec.pol_deg_max)
        try:
            rng = np.random.default_rng(0)
            vinit = np.ascontiguousarray(rng.standard_normal(n), dtype=np.float64)
            # `nev` is an upper estimate of roots in the slice; ChebLanTr
            # returns the true count (nev2). Cap the Krylov/nev by n.
            nev = min(spec.nev if spec.nev is not None else spec.lanczos_dim, n)
            lanm = min(spec.lanczos_dim, n)
            vals, vecs, _res = lib.cheb_lan_tr(
                lanm, nev, xintv, spec.max_iter, spec.tol, vinit, pol, n
            )
        finally:
            lib.free_pol(pol)
        return EigenResult(vals, vecs).sorted()
    finally:
        del cb
        if own:
            lib.finish()


def solve_sliced(
    problem,
    window: "tuple[float, float]",
    n_slices: int,
    *,
    bounds: "tuple[float, float] | None" = None,
    dedup_tol: float = 1e-6,
    verbose: bool = False,
    libpath: str | None = None,
    stream: bool | None = None,
    stream_threshold_gb: float | None = None,
    stream_path: "str | Path | None" = None,
    checkpoint_dir: "str | Path | None" = None,
    dos_moments: "tuple[np.ndarray, float, int] | None" = None,
    **spec_kw,
) -> EigenResult:
    """Solve every eigenpair in ``window`` by looping ``n_slices`` sub-intervals.

    The window is split into equal-count slices (via :func:`slice_boundaries`),
    each is solved with :func:`solve_slice`, and the results are concatenated,
    sorted, and deduplicated: an eigenvalue near a boundary can appear in both
    adjacent slices, so roots within ``dedup_tol`` whose eigenvectors overlap
    are merged.

    Memory: the eigenvectors dominate — a wide interior band at large ``dim`` can
    hold thousands of roots, and ``dim * n_roots * 8`` bytes easily exceeds RAM
    (10^6 dim x 10^4 roots is ~80 GB). When that estimated footprint exceeds
    ``stream_threshold_gb`` the driver switches to a **streaming** path: each
    slice's vectors are written to an on-disk float64 memmap as they are produced,
    so peak RAM is one slice's worth of vectors rather than the whole set. The
    returned :class:`EigenResult` then carries a :class:`numpy.memmap` in
    ``vectors`` — still a ``(dim, n_roots)`` array-like, so ``.sorted()`` /
    slicing / RDM building keep working, but backed by the file.

    Args:
        problem: Matrix-free operator.
        window: ``(lo, hi)`` energy range.
        n_slices: Number of slices.
        bounds: Full spectral bounds; estimated once if omitted.
        dedup_tol: Energy tolerance for merging duplicate boundary roots.
        verbose: If true, print EVSL timing (``StatsPrint``) after the loop.
        libpath: Explicit ``libevsl`` path; auto-resolved if omitted.
        stream: Force the on-disk path (``True``) or the in-RAM path (``False``).
            ``None`` (default) decides automatically from the estimated
            eigenvector footprint vs ``stream_threshold_gb``, warning when it
            engages streaming so a surprise switch is never silent.
        stream_threshold_gb: Estimated-footprint ceiling (GB) for the automatic
            switch. ``None`` (default) derives it from available system memory
            (half of it, min 1 GB) so a big node streams later than a laptop.
            Ignored when ``stream`` is not ``None``.
        stream_path: File for the eigenvector memmap when streaming. A temporary
            file is created (and left on disk) if omitted; the path is reported in
            the streaming warning so callers can find/clean it up.
        checkpoint_dir: Directory for per-slice checkpoints. When set, each slice's
            eigenpairs are saved as soon as it converges and a manifest records the
            completed slices; rerunning with the same directory loads finished
            slices instead of re-solving them, so a crashed long run resumes where
            it stopped. The directory fingerprints the run (window, ``n_slices``,
            bounds, ``dim``, spec) and a rerun with an incompatible problem raises
            rather than mixing results. ``None`` disables checkpointing.
        dos_moments: Optional ``(mu, ecount, deg_used)`` from a KPM-DOS pass a
            caller already ran over this operator/bounds — forwarded to
            :func:`_slice_boundaries_with_count` to skip its own ``kpmdos`` call
            (another ``deg * n_vec`` matvecs otherwise spent recomputing an
            equivalent estimate). ``None`` (default) reproduces the original
            behavior: a fresh ``kpmdos`` pass at ``deg=pol_deg_max``.
        **spec_kw: Extra :class:`SliceSpec` fields (``max_iter``, ``lanczos_dim``,
            ``tol``, ``pol_deg_max``, ``nev``).

    Returns:
        All eigenpairs in ``window``, deduplicated and sorted ascending. In the
        streaming case ``vectors`` is a :class:`numpy.memmap`.
    """
    lib = load_evsl(libpath)
    cb = None
    n = int(problem.dim)
    try:
        cb = _register(problem, lib)
        if bounds is None:
            bounds = _bounds(problem, lib, None, tol=1e-8)
        sli, ecount = _slice_boundaries_with_count(
            problem,
            n_slices,
            window,
            bounds=bounds,
            lib=lib,
            deg=int(spec_kw.get("pol_deg_max", 300)),
            dos_moments=dos_moments,
        )

        # Decide streaming from the estimated eigenvector footprint. ecount is the
        # DOS estimate of roots in the window; guard it to at least n_slices.
        est_roots = max(float(ecount), float(n_slices), 1.0)
        est_gb = n * est_roots * 8 / 1e9
        threshold_gb = (
            stream_threshold_gb
            if stream_threshold_gb is not None
            else _default_stream_threshold_gb()
        )
        do_stream = _decide_stream(stream, est_gb, threshold_gb, stream_path)

        # Set up checkpoint/resume: validate any existing manifest against this
        # run's fingerprint and learn which slices are already done.
        cdir: Path | None = None
        done: frozenset[int] = frozenset()
        if checkpoint_dir is not None:
            cdir = Path(checkpoint_dir)
            fingerprint = _checkpoint_fingerprint(problem, window, n_slices, bounds, spec_kw)
            done = frozenset(_init_checkpoint(cdir, fingerprint))
            if done and verbose:
                print(f"[evsl] resuming: {len(done)}/{n_slices} slices cached in {cdir}")

        if do_stream:
            result = _run_slices_streaming(
                problem,
                lib,
                sli,
                n_slices,
                bounds,
                spec_kw,
                est_roots=est_roots,
                dedup_tol=dedup_tol,
                verbose=verbose,
                stream_path=stream_path,
                cdir=cdir,
                done=done,
            )
        else:
            result = _run_slices_in_ram(
                problem,
                lib,
                sli,
                n_slices,
                bounds,
                spec_kw,
                dedup_tol=dedup_tol,
                verbose=verbose,
                cdir=cdir,
                done=done,
            )
        if verbose:
            lib.stats_print()
        return result
    finally:
        del cb
        lib.finish()


def _decide_stream(
    stream: bool | None,
    est_gb: float,
    threshold_gb: float,
    stream_path: "str | Path | None",
) -> bool:
    """Resolve the streaming toggle, warning when it auto-engages.

    ``stream`` overrides the estimate when set; otherwise the on-disk path is
    chosen when the estimated eigenvector footprint exceeds ``threshold_gb``, and
    a :class:`UserWarning` explains what happened and how to control it (a silent
    switch to disk-backed output would be surprising).
    """
    if stream is not None:
        return bool(stream)
    if est_gb <= threshold_gb:
        return False
    where = str(stream_path) if stream_path is not None else "a temporary file"
    warnings.warn(
        f"EVSL: estimated eigenvector footprint ~{est_gb:.1f} GB exceeds the "
        f"streaming threshold ~{threshold_gb:.1f} GB (default: half of available "
        f"system memory); streaming eigenvectors to disk ({where}) to bound peak "
        f"memory. Pass stream=False to force the in-RAM path, stream=True to "
        f"always stream, or raise stream_threshold_gb.",
        UserWarning,
        stacklevel=3,
    )
    return True


# ---- checkpoint / resume -------------------------------------------------
# Long interval solves (hundreds of slices, minutes each) crash — nodes die, jobs
# hit walltime. With ``checkpoint_dir`` set, each slice's eigenpairs are written to
# disk the moment it converges and a manifest records which slices are done; a
# rerun with the same directory loads finished slices instead of re-solving them.
# Crash-safety comes from writing each slice to a temp file and atomically renaming
# it into place, so a crash mid-slice leaves no partial slice file (it is simply
# redone on resume). The manifest also fingerprints the run (window, n_slices,
# bounds, dim, spec) so a directory built for a different problem is rejected rather
# than silently mixing incompatible results.

_MANIFEST_NAME = "manifest.json"


def _checkpoint_fingerprint(problem, window, n_slices, bounds, spec_kw) -> dict:
    """Identity of a sliced run — a resume must match it exactly.

    Rounded floats keep the comparison stable across float formatting round-trips
    through JSON while still catching a genuinely different window/bounds.
    """
    return {
        "dim": int(problem.dim),
        "n_slices": int(n_slices),
        "window": [round(float(window[0]), 12), round(float(window[1]), 12)],
        "bounds": [round(float(bounds[0]), 12), round(float(bounds[1]), 12)],
        "spec": {k: spec_kw[k] for k in sorted(spec_kw)},
    }


def _load_manifest(cdir: Path) -> dict | None:
    """Read ``manifest.json`` from a checkpoint dir, or ``None`` if absent/corrupt."""
    path = cdir / _MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _init_checkpoint(cdir: Path, fingerprint: dict) -> set[int]:
    """Prepare ``cdir`` for a run and return the set of already-completed slices.

    Creates the directory and manifest on first use. On resume, validates the
    stored fingerprint against ``fingerprint`` and raises on mismatch (never mixes
    results from two different problems); returns the completed-slice indices from
    a compatible manifest.
    """
    cdir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(cdir)
    if manifest is None:
        _write_manifest(cdir, fingerprint, done=[])
        return set()
    stored = manifest.get("fingerprint")
    if stored != fingerprint:
        raise ValueError(
            f"EVSL checkpoint at {cdir} was built for a different problem "
            f"(stored fingerprint {stored} != current {fingerprint}). Point "
            f"checkpoint_dir at a fresh directory or delete this one to restart."
        )
    return {int(j) for j in manifest.get("done", [])}


def _write_manifest(cdir: Path, fingerprint: dict, done) -> None:
    """Atomically (temp file + rename) rewrite the manifest with the done set."""
    payload = {"fingerprint": fingerprint, "done": sorted(int(j) for j in done)}
    tmp = cdir / (_MANIFEST_NAME + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(cdir / _MANIFEST_NAME)


def _slice_paths(cdir: Path, j: int) -> "tuple[Path, Path]":
    """``(energies, vectors)`` .npy paths for slice ``j`` in the checkpoint dir."""
    return cdir / f"slice_{j:05d}_vals.npy", cdir / f"slice_{j:05d}_vecs.npy"


def _save_slice(cdir: Path, j: int, res: EigenResult) -> None:
    """Persist slice ``j``'s eigenpairs, then mark it done in the manifest.

    Each array is written to a ``.tmp`` file and atomically renamed, so a crash
    leaves either the complete slice or nothing — never a truncated file that a
    resume would trust. The manifest is updated only after both arrays are in
    place, so ``done`` never names a slice whose files are incomplete.
    """
    vals_path, vecs_path = _slice_paths(cdir, j)
    for path, arr in ((vals_path, res.energies), (vecs_path, res.vectors)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Write via an open handle so np.save doesn't re-append ".npy" to the
        # ".tmp" name (it does when the path doesn't already end in .npy).
        with tmp.open("wb") as fh:
            np.save(fh, np.asarray(arr))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    manifest = _load_manifest(cdir) or {"done": []}
    done = set(int(x) for x in manifest.get("done", []))
    done.add(int(j))
    _write_manifest(cdir, manifest.get("fingerprint", {}), done)


def _load_slice(cdir: Path, j: int) -> EigenResult:
    """Load a completed slice's eigenpairs saved by :func:`_save_slice`."""
    vals_path, vecs_path = _slice_paths(cdir, j)
    return EigenResult(np.load(vals_path), np.load(vecs_path))


def _run_slices_in_ram(
    problem,
    lib,
    sli,
    n_slices,
    bounds,
    spec_kw,
    *,
    dedup_tol,
    verbose,
    cdir=None,
    done=frozenset(),
) -> EigenResult:
    """Original in-RAM driver: accumulate all slices, concatenate, dedup, sort.

    When ``cdir`` is set, already-completed slices (``done``) are loaded from the
    checkpoint instead of re-solved, and each freshly solved slice is persisted.
    """
    n = int(problem.dim)
    all_vals: list[np.ndarray] = []
    all_vecs: list[np.ndarray] = []
    for j in range(n_slices):
        if cdir is not None and j in done:
            res = _load_slice(cdir, j)
            if verbose:
                print(f"[evsl] slice {j + 1}/{n_slices}: {res.energies.size} roots (cached)")
        else:
            spec = SliceSpec(lo=float(sli[j]), hi=float(sli[j + 1]), **spec_kw)
            res = solve_slice(problem, spec, bounds, lib=lib)
            if verbose:
                print(
                    f"[evsl] slice {j + 1}/{n_slices} "
                    f"[{spec.lo:.6f}, {spec.hi:.6f}]: {res.energies.size} roots"
                )
            if cdir is not None:
                _save_slice(cdir, j, res)
        all_vals.append(res.energies)
        all_vecs.append(res.vectors)

    if not all_vals or sum(v.size for v in all_vals) == 0:
        return EigenResult(np.empty(0), np.empty((n, 0)))

    vals = np.concatenate(all_vals)
    vecs = (
        np.column_stack([v for v in all_vecs if v.size])
        if any(v.size for v in all_vecs)
        else np.empty((n, 0))
    )
    return _dedup(EigenResult(vals, vecs).sorted(), dedup_tol)


def _run_slices_streaming(
    problem,
    lib,
    sli,
    n_slices,
    bounds,
    spec_kw,
    *,
    est_roots,
    dedup_tol,
    verbose,
    stream_path,
    cdir=None,
    done=frozenset(),
) -> EigenResult:
    """On-disk driver: write each slice's vectors to a memmap, then compact once.

    Peak RAM is one slice's eigenvectors plus O(total_roots) floats (the running
    energy list) — never the full ``(dim, total_roots)`` block. The raw memmap is
    over-allocated from the DOS estimate and grown on overflow; a single
    column-at-a-time compaction pass then applies the boundary dedup and writes
    the survivors in ascending-energy order into a right-sized memmap.

    When ``cdir`` is set, completed slices (``done``) are loaded from the
    checkpoint and streamed into the memmap instead of re-solved; each freshly
    solved slice is persisted so a crash loses at most the in-progress slice.
    """
    n = int(problem.dim)
    raw_path = _resolve_stream_path(stream_path, tag="raw")
    # Over-allocate columns: DOS undercounts near boundaries + duplicates appear
    # in overlapping slices, so pad the estimate generously.
    capacity = max(int(math.ceil(est_roots * 1.3)) + n_slices, 1)
    raw = np.memmap(raw_path, dtype=np.float64, mode="w+", shape=(n, capacity))
    energies = np.empty(capacity, dtype=np.float64)

    filled = 0
    for j in range(n_slices):
        if cdir is not None and j in done:
            res = _load_slice(cdir, j)
            tag = "cached"
        else:
            spec = SliceSpec(lo=float(sli[j]), hi=float(sli[j + 1]), **spec_kw)
            res = solve_slice(problem, spec, bounds, lib=lib)
            if cdir is not None:
                _save_slice(cdir, j, res)
            tag = "streamed"
        k = int(res.energies.size)
        if verbose:
            print(f"[evsl] slice {j + 1}/{n_slices}: {k} roots ({tag})")
        if k == 0:
            continue
        if filled + k > capacity:  # rare: DOS underestimated the count -> grow
            raw, energies, capacity = _grow_stream(
                raw, raw_path, energies, capacity, need=filled + k, n=n
            )
        raw[:, filled : filled + k] = res.vectors
        energies[filled : filled + k] = res.energies
        filled += k
    raw.flush()

    if filled == 0:
        del raw
        _unlink_quiet(raw_path)
        return EigenResult(np.empty(0), np.empty((n, 0)))

    # Compact: dedup boundary duplicates and write survivors in energy order into
    # a right-sized memmap, one column at a time (never all columns in RAM).
    result = _compact_streamed(
        raw,
        energies[:filled],
        filled,
        n,
        dedup_tol,
        out_path=_resolve_stream_path(stream_path, tag="vecs"),
    )
    del raw
    _unlink_quiet(raw_path)
    return result


def _resolve_stream_path(stream_path: "str | Path | None", *, tag: str) -> Path:
    """Pick the memmap file path for a streaming phase.

    ``tag`` distinguishes the raw over-allocated write buffer (``"raw"``) from the
    compacted output (``"vecs"``). When the caller gave an explicit ``stream_path``
    it is used for the compacted output and suffixed for the raw buffer; otherwise
    a leftover-on-disk temp file is created for each (mirrors the tempfile pattern
    used by the dice/sbd solvers).
    """
    if stream_path is not None:
        base = Path(stream_path)
        return base if tag == "vecs" else base.with_suffix(base.suffix + ".raw")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=f".evsl_{tag}.dat")
    handle.close()
    return Path(handle.name)


def _unlink_quiet(path: Path) -> None:
    """Delete ``path`` if present, ignoring races/permission errors."""
    try:
        path.unlink()
    except OSError:
        pass


def _grow_stream(
    raw: np.memmap,
    raw_path: Path,
    energies: np.ndarray,
    capacity: int,
    *,
    need: int,
    n: int,
) -> "tuple[np.memmap, np.ndarray, int]":
    """Enlarge the raw column buffer in place on disk when the DOS undercounted.

    Rare: the memmap is over-allocated from the DOS estimate, but a slice can
    still overflow it. We flush, reopen the same file at a larger column count
    (memmap ``r+`` with a bigger shape extends the backing file), and grow the
    parallel in-RAM energy array to match. Returns the new ``(raw, energies,
    capacity)``.
    """
    new_capacity = max(need, int(math.ceil(capacity * 1.5)))
    raw.flush()
    del raw
    grown = np.memmap(raw_path, dtype=np.float64, mode="r+", shape=(n, new_capacity))
    new_energies = np.empty(new_capacity, dtype=np.float64)
    new_energies[:capacity] = energies[:capacity]
    return grown, new_energies, new_capacity


def _compact_streamed(
    raw: np.memmap,
    energies: np.ndarray,
    filled: int,
    n: int,
    dedup_tol: float,
    *,
    out_path: Path,
) -> EigenResult:
    """Dedup + energy-sort the raw memmap into a right-sized output memmap.

    Sorts the ``filled`` energies (cheap: O(filled) floats in RAM), walks them in
    ascending order, drops boundary-straddling duplicates (same energy within
    ``dedup_tol`` *and* (anti)parallel eigenvector — the overlap check reads two
    columns, memmap-safe), and copies each survivor's column from ``raw`` to the
    output memmap one at a time. Peak extra RAM is two columns, never the whole
    block; the output is a real ``(n, k)`` memmap so the ``EigenResult`` contract
    holds.
    """
    order = np.argsort(energies, kind="stable")
    e_sorted = energies[order]

    # First determine which sorted positions survive dedup (two-column overlap
    # test against the previously kept column, exactly like the in-RAM _dedup).
    keep: list[int] = []
    if filled:
        keep.append(0)
        for i in range(1, filled):
            col_i = order[i]
            prev = order[keep[-1]]
            close = abs(e_sorted[i] - e_sorted[keep[-1]]) <= dedup_tol
            if close:
                overlap = abs(float(raw[:, col_i] @ raw[:, prev])) > 0.9
            else:
                overlap = False
            if not (close and overlap):
                keep.append(i)

    k = len(keep)
    out = np.memmap(out_path, dtype=np.float64, mode="w+", shape=(n, k))
    out_e = np.empty(k, dtype=np.float64)
    for dst, pos in enumerate(keep):
        src = int(order[pos])
        out[:, dst] = raw[:, src]
        out_e[dst] = e_sorted[pos]
    out.flush()
    return EigenResult(out_e, out)


def _dedup(result: EigenResult, tol: float) -> EigenResult:
    """Drop boundary-straddling duplicate roots from a sorted :class:`EigenResult`.

    Two roots are duplicates when their energies agree within ``tol`` *and*
    their eigenvectors are (anti)parallel (``|<u, v>|`` near 1) — the overlap
    check guards against merging genuine near-degeneracies with distinct
    eigenvectors.
    """
    e, v = result.energies, result.vectors
    if e.size <= 1:
        return result
    keep = [0]
    for i in range(1, e.size):
        j = keep[-1]
        close = abs(e[i] - e[j]) <= tol
        overlap = abs(float(v[:, i] @ v[:, j])) > 0.9 if close else False
        if not (close and overlap):
            keep.append(i)
    idx = np.array(keep, dtype=int)
    return EigenResult(e[idx], v[:, idx])
