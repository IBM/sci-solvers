"""EVSL spectrum-slicing selected-CI solver (needs a built ``libevsl``)."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np

from sci_solvers.eigen_problem import EigenProblem, EigenResult
from sci_solvers.energy_solvers.general_algebra_base import GeneralAlgebraSolverBase, dense_fallback
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin


class EvslSolver(SpinFilterMixin, GeneralAlgebraSolverBase):
    """EVSL spectrum-slicing solver (needs a built ``libevsl``).

    With no ``window`` a KPM-DOS pre-pass estimates a lower band holding at least
    ``nroots`` eigenvalues; with ``window=(lo, hi)`` that interval is sliced
    directly. See ``scripts/energy_solvers/build_evsl.sh``.
    """

    _accepts_solve_nroots = True  # its _diagonalize honors a solve-time nroots (mixin oversampling)

    def __init__(
        self,
        *,
        window: "tuple[float, float] | None" = None,
        n_slices: int | None = None,
        dos_deg: int = 300,
        dos_nvec: int = 60,
        lanczos_dim: int = 200,
        pol_deg_max: int = 300,
        libpath: str | None = None,
        stream: bool | None = None,
        stream_threshold_gb: float | None = None,
        stream_path: "str | None" = None,
        checkpoint_dir: "str | None" = None,
        **kwargs: Any,
    ):
        """Store slicing/DOS tuning, streaming policy, and the library path."""
        super().__init__(**kwargs)
        self.window = window
        self.n_slices = n_slices
        self.dos_deg = dos_deg
        self.dos_nvec = dos_nvec
        self.lanczos_dim = lanczos_dim
        self.pol_deg_max = pol_deg_max
        self.libpath = libpath
        self.stream = stream
        self.stream_threshold_gb = stream_threshold_gb
        self.stream_path = stream_path
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def _nev_target(problem: EigenProblem, nroots: int) -> int:
        """Expected root count for a window sized to hold ``nroots`` eigenvalues.

        Shared by :meth:`_lower_band` (sizing the auto-detected window) and
        :meth:`_diagonalize` (sizing ``ChebLanTr``'s ``nev``, decoupled from
        ``lanczos_dim`` so the same target is used consistently for both).
        """
        return min(problem.dim, max(nroots + 4, int(2 * nroots)))

    def _diagonalize(self, problem: EigenProblem, nroots: int) -> EigenResult:
        """Lowest ``nroots`` eigenpairs via spectrum slicing, dense fallback for small ``n``."""
        from sci_solvers.energy_solvers.evsl import load_evsl, solve_sliced
        from sci_solvers.energy_solvers.evsl.solver import _bounds, _register

        n = int(problem.dim)
        if n < 200 or nroots >= n:
            return dense_fallback(problem, nroots)

        lib = load_evsl(self.libpath)
        cb = None
        dos_moments = None
        try:
            cb = _register(problem, lib)
            lmin, lmax = _bounds(problem, lib, None, tol=self.tol)
            if self.window is not None:
                lo, hi = float(self.window[0]), float(self.window[1])
            else:
                lo, hi, dos_moments = self._lower_band(problem, lib, nroots, (lmin, lmax))
        finally:
            del cb
            lib.finish()

        n_slices = self.n_slices or max(1, min(8, nroots // 8 + 1))
        result = solve_sliced(
            problem,
            (lo, hi),
            n_slices=n_slices,
            bounds=(lmin, lmax),
            max_iter=self.maxiter,
            lanczos_dim=self.lanczos_dim,
            nev=self._nev_target(problem, nroots),
            tol=self.tol,
            pol_deg_max=self.pol_deg_max,
            libpath=self.libpath,
            stream=self.stream,
            stream_threshold_gb=self.stream_threshold_gb,
            stream_path=self.stream_path,
            checkpoint_dir=self.checkpoint_dir,
            dos_moments=dos_moments,
        )
        order = np.argsort(result.energies)
        k = min(nroots, result.energies.size)
        if k == 0:  # slicing found nothing (window too tight) -> dense fallback
            return dense_fallback(problem, nroots)
        pick = order[:k]
        energies = np.asarray(result.energies[pick])
        vecs = np.asarray(result.vectors[:, pick])
        return EigenResult(energies, vecs)

    def _lower_band(
        self, problem: EigenProblem, lib: Any, nroots: int, bounds: "tuple[float, float]"
    ) -> "tuple[float, float, tuple[np.ndarray, float, int]]":
        """Estimate a lower window ``[lmin, hi]`` holding >= ``nroots`` eigenvalues via KPM DOS.

        Also returns the raw ``(mu, ecount, deg)`` moments from this pass so the
        caller can hand them to ``solve_sliced(..., dos_moments=...)`` and skip
        its own, otherwise-redundant KPM-DOS pass.
        """
        from sci_solvers.energy_solvers.evsl.solver import _dos_from_moments, _kpm_moments

        lmin, lmax = bounds
        mu, ecount = _kpm_moments(lib, self.dos_deg, self.dos_nvec, bounds)
        energies, dos = _dos_from_moments(mu, lmin, lmax, 2000, int(problem.dim))
        widths = np.gradient(energies)
        cum = np.cumsum(dos * widths)
        target = self._nev_target(problem, nroots)
        idx = int(np.searchsorted(cum, target))
        idx = min(idx, energies.size - 1)
        hi = float(energies[idx])
        span = lmax - lmin
        return lmin - 1e-6 * span, hi + 1e-3 * span, (mu, ecount, self.dos_deg)
