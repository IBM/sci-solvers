"""Composable spin-projection mixin for any ``EnergyCalculatorBase`` solver.

Spin projection (targeting a total ``S^2``) is orthogonal to *how* a solver
diagonalizes, so it lives here as a mixin rather than being baked into the
reduction. Mixed in to the LEFT of a real solver base, it oversolves and keeps
only the roots whose ``<S^2>`` matches the target, escalating the oversample
until enough are found:

    class ArpackSolver(SpinFilterMixin, GeneralAlgebraSolverBase): ...

It works on any solver whose returned ``SCIResult``s carry full CI amplitudes
(so ``<S^2>`` can be classified). If the wrapped solver honors a solve-time
``nroots`` it is oversolved to gather enough target-spin roots; a single-root
native solver simply has its returned root(s) filtered.

Because the mixin *wraps* ``sci_solver_callable_eigen_states`` via the MRO, it
must sit *above* the class that implements that method.

Solvers that project spin natively (torch_sci's penalty, PySCF's ``fix_spin_``)
should NOT mix this in; the definition-time guard below rejects stacking it on a
solver that already owns ``spin_sq``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import inspect
from typing import Any, List, Tuple, cast

import numpy as np

from sci_solvers.eigen_states import EigenStates
from sci_solvers.energy_calculation_base import is_spinless_request


def spin_square_of_vector(
    vector: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    norb: int,
    nelec: "tuple[int, int]",
) -> float:
    """Return ``<S^2>`` of a full-space CI vector.

    ``vector`` is a ``(Na*Nb,)`` or ``(Na, Nb)`` amplitude array in the
    ``ia * Nb + ib`` product-space layout (the convention
    :class:`~sci_solvers.eigen_problem.EigenProblem` uses). Real amplitudes go
    through ``pyscf`` ``spin_square``; complex Hermitian amplitudes (SOC / GHF /
    complex-spatial path) go through the conjugate-correct
    :func:`~sci_solvers.complex_rdm.s_squared_complex`. Defined here so the
    ``<S^2>`` classification lives with the spin-projection mixin that uses it.
    For real amplitudes ``strs_a``/``strs_b`` must be strictly ascending (sorted,
    unique): PySCF's ``selected_ci`` assumes it and is silently wrong otherwise.
    """
    na, nb = len(strs_a), len(strs_b)
    amps2d = np.asarray(vector).reshape(na, nb)
    if np.iscomplexobj(amps2d):
        from sci_solvers.complex_rdm import s_squared_complex

        return s_squared_complex(np.ascontiguousarray(amps2d, dtype=np.complex128), norb, nelec)

    from pyscf.fci import selected_ci

    amps = np.ascontiguousarray(amps2d.astype(np.float64))
    s2, _ = selected_ci.spin_square(selected_ci._as_SCIvector(amps, (strs_a, strs_b)), norb, nelec)
    return float(s2)


def _own_kw_params(cls: type) -> set[str]:
    """Names of ``cls``'s OWN ``__init__`` keyword params (excludes self/*args/**kwargs)."""
    init = cls.__dict__.get("__init__")  # own __init__ only, not inherited
    if init is None:
        return set()
    try:
        params = inspect.signature(init).parameters.values()
    except (TypeError, ValueError):
        return set()
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {p.name for p in params if p.kind in kinds and p.name != "self"}


class SpinFilterMixin:
    """Add post-hoc ``spin_sq`` projection to any solver it is mixed into.

    Constructor args (all consumed here, forwarded remainder flows down the MRO):

    * ``spin_sq``: target total spin squared (e.g. ``0.0`` singlet). ``None``
      disables projection entirely (the mixin becomes a passthrough).
    * ``tol_s2``: a root is kept if ``abs(<S^2> - spin_sq) < tol_s2``.
    * ``oversample``: initial multiple of ``nroots`` to solve for.
    * ``max_attempts``: escalating solve attempts before raising when the target
      sector is underpopulated.
    * ``spin_filter``: runtime on/off switch (default ``True``). Also honored as a
      per-call keyword on :meth:`sci_solver_callable_eigen_states`.
    """

    # Whether the wrapped solver accepts a solve-time ``nroots`` so this mixin can
    # oversolve. Default False (just filter the roots the solver returns); a
    # mixed-in solver that supports it overrides this to True. It must be set on a
    # class *before* the mixin in the MRO (i.e. the concrete solver), not on a
    # base after it, or this default would shadow the override.
    _accepts_solve_nroots: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject stacking on a solver that also declares one of this mixin's args.

        Runs when a subclass is *defined*. If any class further down the MRO
        redeclares one of ``SpinFilterMixin``'s own ``__init__`` parameters, that
        value would be consumed here and silently starved downstream — so raise.
        """
        super().__init_subclass__(**kwargs)
        mine = _own_kw_params(SpinFilterMixin)
        mro = cls.__mro__
        for downstream in mro[mro.index(SpinFilterMixin) + 1 :]:
            clash = mine & _own_kw_params(downstream)
            if clash:
                raise TypeError(
                    f"{cls.__name__}: SpinFilterMixin consumes {sorted(mine)}; "
                    f"{downstream.__name__} further down the MRO also declares "
                    f"{sorted(clash)}, which would be silently starved. Rename those "
                    f"parameters or do not mix SpinFilterMixin into that solver "
                    f"(a solver that projects spin natively should not use it)."
                )

    def __init__(
        self,
        *,
        spin_sq: float | None = None,
        tol_s2: float = 0.1,
        oversample: int = 2,
        max_attempts: int = 4,
        spin_filter: bool = True,
        **kwargs: Any,
    ):
        """Consume the projection knobs and forward the rest up the MRO."""
        super().__init__(**kwargs)  # sets self.nroots etc. on the real base below
        self.spin_sq = spin_sq
        self.tol_s2 = tol_s2
        self.oversample = oversample
        self.max_attempts = max_attempts
        self.spin_filter = spin_filter
        # Under projection the raw eigenvalue is unreliable, so energies are
        # recomputed from the RDMs — a projection concern, not the base's.
        if spin_sq is not None and spin_filter:
            self.compute_rdm2 = True

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: "List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray]",
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: "Tuple[int, int] | int",
        *,
        nroots: int | None = None,
        spin_filter: bool | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """Oversolve (when possible), then keep the roots whose ``<S^2>`` matches ``spin_sq``.

        A pure passthrough when ``spin_sq is None`` or ``not active`` (the
        common case) -- ``nelec`` and ``ci_strs_batches`` flow through
        unexamined to the wrapped solver, so this works on the spinless path
        (a scalar ``nelec``) exactly as far as the wrapped solver itself
        does. The classification step below (``_keep_target_spin`` /
        ``_spin_square``) is spatial-only (``<S^2>`` has no meaning without
        an alpha/beta split, and it reads ``SCIState.ci_strs_a``/``ci_strs_b``,
        which a spinless ``SpinlessSCIState`` does not carry), so an active
        filter on a scalar ``nelec`` is rejected outright rather than left to
        fail with an unrelated-looking ``AttributeError`` deep in
        ``_spin_square``.
        """
        want = getattr(self, "nroots", 1) if nroots is None else nroots
        active = self.spin_filter if spin_filter is None else spin_filter
        # Oversample only if the wrapped solver takes a solve-time nroots; some
        # solvers forward **kwargs into an inner routine that would reject it.
        can_oversample = self._accepts_solve_nroots

        def _solve(k: int) -> List[EigenStates]:
            extra = {"nroots": k} if can_oversample else {}
            # A mixin: the next class in the MRO is the wrapped solver.
            return super(SpinFilterMixin, self).sci_solver_callable_eigen_states(  # type: ignore[misc]
                ci_strs_batches, hcore, eri, norb, nelec, **extra, **kwargs
            )

        if self.spin_sq is None or not active:
            return _solve(want)

        if is_spinless_request(nelec):
            raise NotImplementedError(
                f"{type(self).__name__}: spin_sq={self.spin_sq!r} was set (an active "
                "spin-projection filter), but nelec is a scalar (the spinless flat "
                "spin-orbital path, no alpha/beta split) -- <S^2> classification has "
                "no meaning there. Omit spin_sq on the spinless path (mirrors the "
                "native torch_sci spinless path, which rejects spin_sq outright for "
                "the same reason)."
            )

        spatial_nelec = cast(Tuple[int, int], nelec)  # the scalar case raised above
        k = self.oversample * want if can_oversample else want
        attempts = self.max_attempts if can_oversample else 1
        last: List[EigenStates] = []
        for _ in range(attempts):
            last = [self._keep_target_spin(es, want, norb, spatial_nelec) for es in _solve(k)]
            if all(len(es.states) >= want for es in last):
                return last
            k *= 2

        short = [i for i, es in enumerate(last) if len(es.states) < want]
        raise ValueError(
            f"SpinFilterMixin: batches {short} yielded fewer than {want} root(s) with "
            f"<S^2> within {self.tol_s2} of the target {self.spin_sq}"
            + (f" after {self.max_attempts} attempt(s)" if can_oversample else "")
            + ". Increase oversample/max_attempts/tol_s2 (or the solver's nroots), or "
            "use a solver with native spin projection for a deeply buried sector."
        )

    def _keep_target_spin(
        self, eigen_states: EigenStates, want: int, norb: int, nelec: Tuple[int, int]
    ) -> EigenStates:
        """Keep the lowest ``want`` roots whose ``<S^2>`` matches ``spin_sq``."""
        target = self.spin_sq
        assert target is not None  # only called with an active filter
        kept = [
            s
            for s in eigen_states.states
            if abs(self._spin_square(s, norb, nelec) - target) < self.tol_s2
        ]
        return EigenStates(states=kept[:want])

    @staticmethod
    def _spin_square(sci_result: Any, norb: int, nelec: Tuple[int, int]) -> float:
        """``<S^2>`` of a returned root, from its ``SCIState`` amplitudes.

        Amplitudes are passed through with their native dtype (not cast to
        ``float64``) so a complex Hermitian state keeps its imaginary part;
        :func:`spin_square_of_vector` dispatches real vs complex.
        """
        st = sci_result.sci_state
        amps = np.asarray(st.amplitudes).ravel()
        return spin_square_of_vector(amps, st.ci_strs_a, st.ci_strs_b, norb, nelec)
