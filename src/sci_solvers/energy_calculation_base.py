"""
Base class for sci_solver implementations.

Defines the qiskit-addon-sqd compatible ``sci_solver_callable``
interface that all solvers in this package implement. Each concrete
subclass wraps a different diagonalizer (RIKEN's sbd binary,
qiskit-addon-dice-solver, PySCF's selected_ci) and exposes a uniform
per-batch dispatch interface.

The single contract is::

    sci_solver_callable(ci_strs_batches, hcore, eri, norb, nelec, **kwargs)
        -> List[SCIResult]

where ``ci_strs_batches`` is a list of ``(strs_a, strs_b)`` int64-array
tuples — one batch per element — and the returned ``SCIResult.energy``
values are ELECTRONIC (no nuclear repulsion). Adding ``e_nuc`` is the
caller's responsibility, matching qiskit-addon-sqd's convention.

Subclasses MAY accept additional optional keyword arguments beyond the
protocol minimum (e.g. ``max_cycle``, ``symmetrize_spin``, ``spin_sq``);
those are the per-solver knobs documented on each concrete solver class.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
from qiskit_addon_sqd.fermion import SCIResult, bitstring_matrix_to_ci_strs

from sci_solvers.eigen_states import EigenStates
from sci_solvers.hamiltonian import HamiltonianData


def reconcile_nroots(nroots: int, n_roots: int, *, default: int = 1) -> int:
    """Reconcile the ``nroots`` / ``n_roots`` spellings of the root count.

    ``nroots`` is the canonical spelling; ``n_roots`` is kept as a backward-compatible alias.
    Whichever one differs from ``default`` is used; if BOTH are set to a
    non-default value they conflict and a ``ValueError`` is raised.
    """
    if nroots != default and n_roots != default:
        raise ValueError(
            f"Pass either `nroots` or `n_roots`, not both (got nroots={nroots}, n_roots={n_roots})."
        )
    return nroots if nroots != default else n_roots


def is_spinless_request(nelec: Any) -> bool:
    """Whether ``nelec`` signals the genuinely spinless spin-orbital path.

    The spatial path is keyed on the ``(n_alpha, n_beta)`` factorization, so a
    *scalar* total particle number ``N`` (an ``int``/integer, not a 2-tuple) is
    the unambiguous signal for the spinless spin-orbital request: a single flat
    block of ``nmo`` spin-orbitals, batches given as flat ``strs`` lists rather
    than ``(strs_a, strs_b)`` pairs. The single canonical check every dispatch
    site in this package uses — originally
    ``GeneralAlgebraSolverBase._is_spinless_request`` (kept as a thin delegate
    there for that class's own callers), promoted here so
    ``reject_spinless_nelec``, ``SpinFilterMixin``, and the ambiguity guard
    below share one definition instead of re-writing the same ``isinstance``
    check.
    """
    return isinstance(nelec, (int, np.integer))


def reject_spinless_nelec(solver_name: str, nelec: "Tuple[int, int] | int") -> None:
    """Raise ``NotImplementedError`` if ``nelec`` signals the spinless path.

    A scalar ``nelec`` (total particle number ``N``, not an ``(n_a, n_b)``
    tuple) is the spinless spin-orbital request's signal — see
    ``GeneralAlgebraSolverBase._is_spinless_request`` /
    ``TorchSCISolver``'s docstring. Solvers that only implement the spatial
    ``(strs_a, strs_b)`` / ``(n_a, n_b)`` reduction should call this first
    thing in their own ``sci_solver_callable_eigen_states`` (after the
    ``super()`` ambiguity-guard call), so a spinless caller gets a clear,
    named error instead of a bare tuple-unpack/index ``TypeError`` several
    calls deep in the implementation.

    Args:
        solver_name: the concrete solver's class name, for the error message
            (pass ``type(self).__name__``).
        nelec: the ``nelec`` argument as received.

    Raises:
        NotImplementedError: ``nelec`` is a scalar (int or numpy integer).
    """
    if is_spinless_request(nelec):
        from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, supports_spinless

        capable = ", ".join(repr(n) for n in AVAILABLE_SOLVERS if supports_spinless(n))
        raise NotImplementedError(
            f"{solver_name} does not implement the spinless (flat spin-orbital, no "
            f"alpha/beta split) path -- got a scalar nelec={nelec!r} (total particle "
            "number), which signals that path rather than an (n_a, n_b) tuple. Use "
            f"one of the solvers that do: {capable}."
        )


def project_init_vector(
    prev_state: Any,
    ci_strs: "Tuple[np.ndarray, np.ndarray] | np.ndarray",
    mode: str = "tpb",
    *,
    min_norm: float = 1e-8,
    device: Any = "cpu",
) -> "np.ndarray | None":
    """Project a previous solve's state onto a new basis, as an ``init_vectors`` entry.

    Warm starting an SQD iteration from the previous one means mapping the previous
    ground state onto this iteration's strings, which drop some and add others. This
    builds the entry for one batch: amplitudes copied on the determinants both bases
    share, zero on the new ones, normalized. It is laid out exactly as
    ``init_vectors`` expects in each mode, so a caller never handles orderings or
    sign conventions. That holds because every mode's ``init_vectors`` contract uses
    the convention that mode's own ``sci_state.amplitudes`` come back in (TPB
    spin-block, GDB interleaved-Wick), so amplitudes are copied, never re-signed.

    Args:
        prev_state: the previous result's ``sci_state``, from the same solver mode:
            a ``SCIState`` (TPB, ``(M_a, M_b)`` amplitudes), a ``PairSCIState`` (GDB,
            1-D amplitudes over parallel ``ci_strs_a``/``ci_strs_b``) or a
            ``SpinlessSCIState`` (1-D amplitudes over ``ci_strs``).
        ci_strs: the new batch's strings, as passed to the solver: ``(strs_a,
            strs_b)`` in TPB and GDB, a flat ``strs`` array on the spinless path. Any
            order; duplicates take the first occurrence, as the solver does.
        mode: ``"tpb"`` or ``"gdb"`` (the solver's ``mode``). Ignored for a spinless
            state, which is recognized by its ``ci_strs`` field. Taken explicitly
            rather than inferred from the amplitudes' shape, which is ambiguous for
            a single-determinant subspace.
        min_norm: overlap below which the bases share nothing worth starting from.
        device: torch device for the key lookup, a sort over every determinant. CPU
            takes ~3.4 s per 10⁷ GDB pairs; pass the solve device (e.g. ``"cuda"``)
            at 10⁸. The result is always a CPU numpy array.

    Returns:
        The 1-D entry, or ``None`` (cold start) when the overlap norm is below
        ``min_norm``, e.g. for disjoint bases.

    Raises:
        ValueError: ``mode`` is not ``"tpb"``/``"gdb"``, or ``ci_strs`` does not have
            the shape that mode takes.
    """
    import torch
    from torch_sci_solver.torch_impl.qiskit_adapter import _first_index, _signed_int64_keys
    from torch_sci_solver.torch_impl.spinless_helpers import _n_limbs_for, _split_to_limbs

    amps = np.asarray(prev_state.amplitudes)
    if amps.size == 0:
        return None  # an empty previous basis shares nothing: cold start
    dev = torch.device(device)

    def _index(prev_keys: np.ndarray, new_keys: np.ndarray) -> np.ndarray:
        # Position of each new key among the previous ones (first occurrence), or -1.
        # The engine's own vectorized lookup: a per-key dict is ~85 s at 10⁸ GDB pairs.
        return _first_index(prev_keys, new_keys, device=dev).cpu().numpy()

    if not hasattr(prev_state, "ci_strs_a") and hasattr(prev_state, "ci_strs"):
        # Spinless determinants may exceed 64 bits (nmo > 64): key them by limbs.
        prev_strs = np.asarray(prev_state.ci_strs, dtype=object).ravel()
        new_strs = np.asarray(ci_strs, dtype=object).ravel()
        n_limbs = _n_limbs_for(max(int(x).bit_length() for x in [*prev_strs, *new_strs, 1]))

        def _rows(strs: np.ndarray) -> np.ndarray:
            return np.array([_split_to_limbs(int(x), n_limbs) for x in strs], dtype=np.int64)

        idx = _index(_rows(prev_strs), _rows(new_strs))
        out = np.where(idx >= 0, amps.ravel()[np.maximum(idx, 0)], 0)
    elif mode == "gdb":
        strs_a, strs_b = (_signed_int64_keys(s) for s in ci_strs)
        if len(strs_a) != len(strs_b):
            raise ValueError("GDB ci_strs must be equal-length (strs_a, strs_b) pair lists.")
        prev = np.stack(
            [_signed_int64_keys(prev_state.ci_strs_a), _signed_int64_keys(prev_state.ci_strs_b)],
            axis=1,
        )
        idx = _index(prev, np.stack([strs_a, strs_b], axis=1))
        out = np.where(idx >= 0, amps.ravel()[np.maximum(idx, 0)], 0)
    elif mode == "tpb":
        strs_a, strs_b = (_signed_int64_keys(s) for s in ci_strs)
        grid = amps.reshape(len(prev_state.ci_strs_a), len(prev_state.ci_strs_b))
        ia = _index(_signed_int64_keys(prev_state.ci_strs_a), strs_a)
        ib = _index(_signed_int64_keys(prev_state.ci_strs_b), strs_b)
        out = grid[np.ix_(np.maximum(ia, 0), np.maximum(ib, 0))]
        out = np.where((ia >= 0)[:, None] & (ib >= 0)[None, :], out, 0).ravel()
    else:
        raise ValueError(f'mode must be "tpb" or "gdb", got {mode!r}')

    out = out.astype(amps.dtype, copy=False)
    norm = float(np.linalg.norm(out))
    return None if norm < min_norm else out / norm


def reject_init_vectors(solver_name: str, kwargs: dict[str, Any]) -> None:
    """Pop ``init_vectors`` from ``kwargs``; raise if it carries any start vector.

    ``init_vectors`` is the per-call Davidson start vector, one entry per batch (see
    ``TorchSCISolver.sci_solver_callable_eigen_states``). Solvers that cannot honor it
    call this first thing in their own ``sci_solver_callable_eigen_states``, so that
    the kwarg is removed before ``kwargs`` reaches an inner routine that would reject
    it with an unrelated ``TypeError``, and a caller who did pass a vector gets a clear
    error instead of a silently cold solve with misleading timings. ``None`` and an
    all-``None`` list mean "cold start", which every solver already does, so they pass.

    Args:
        solver_name: the concrete solver's class name, for the error message
            (pass ``type(self).__name__``).
        kwargs: the solver's ``**kwargs`` dict, mutated in place.

    Raises:
        NotImplementedError: some ``init_vectors`` entry is not ``None``.
    """
    init_vectors = kwargs.pop("init_vectors", None)
    if init_vectors is None or all(v is None for v in init_vectors):
        return
    from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, supports_init_vectors

    capable = ", ".join(repr(n) for n in AVAILABLE_SOLVERS if supports_init_vectors(n))
    raise NotImplementedError(
        f"{solver_name} does not accept a Davidson start vector (init_vectors). Pass "
        f"init_vectors=None for a cold solve, or use one of the solvers that do: {capable}."
    )


def davidson_block_size_to_max_space(
    davidson_block_size: int | None, nroots: int, *, default_max_space: int | None = None
) -> int:
    """Convert ``davidson_block_size`` to the ``max_space`` according to PySCF's convention, also used by a few other solvers that are based on its davidson kernel.

    ``davidson_block_size`` is defined as the subspace the solver *actually*
    restarts at — the same meaning it has on the nested-loop solvers
    (:class:`~sci_solvers.energy_solvers.torch_sci_solver.TorchSCISolver`,
    :class:`~sci_solvers.energy_solvers.sbd_solver.SBDSolver`). PySCF's
    ``davidson1`` does not treat ``max_space`` that way: it widens whatever it is
    handed by ``(nroots-1)*4`` (``pyscf/lib/linalg_helper.py``), so the value a
    caller passes is not the subspace it restarts at. That widening is subtracted
    here, making the effective subspace equal the requested block size on every
    solver.

    ``None`` selects the caller's historical default, given as
    ``default_max_space`` in PySCF's own pre-widening units so the value handed to
    ``davidson1`` is bit-identical to what that solver passed before
    ``davidson_block_size`` existed — behaviour, memory and convergence unchanged.
    These defaults differ per solver and are deliberately preserved:
    ``PySCFSolver`` used ``12 + 2*nroots`` (i.e. ``6*nroots + 8`` post-widening),
    while ``DavidsonSolver`` inherited ``sci_kernel_fixed_space``'s flat ``12``
    (``4*nroots + 8`` post-widening). Omitting it falls back to PySCF's default.

    Raises:
        ValueError: If ``davidson_block_size`` does not survive the subtraction
            (``<= (nroots-1)*4``). It cannot then be expressed as a positive
            ``max_space``, which is the domain PySCF accepts.
    """
    widening = (nroots - 1) * 4
    if davidson_block_size is None:
        return 12 + 2 * nroots if default_max_space is None else default_max_space
    max_space = davidson_block_size - widening
    if max_space < 1:
        default_block = (
            6 * nroots + 8 if default_max_space is None else default_max_space + widening
        )
        raise ValueError(
            f"davidson_block_size={davidson_block_size} is too small for nroots={nroots}: "
            f"PySCF's Davidson widens max_space by (nroots-1)*4={widening}, leaving "
            f"max_space={max_space}, but it must be >= 1. Pass "
            f"davidson_block_size > {widening} (the default is {default_block})."
        )
    return max_space


class EnergyCalculatorBase(ABC):
    """
    Abstract base class for ``sci_solver_callable`` wrappers.

    Concrete subclasses implement :meth:`sci_solver_callable` to
    diagonalize a Hamiltonian on the subspace spanned by each input
    batch's ``(strs_a, strs_b)`` outer product.
    """

    ## Capability flags
    # whether the solver can diagonalize a complex Hermitian operator, may be overrided on subclasses
    complex_capable: bool = False
    # whether the solver returns more than one root per batch when asked , may be overrided on subclasses
    supports_multiroot: bool = True
    # whether sci_solver_callable_eigen_states accepts a *scalar* nelec (total N, the
    # flat spin-orbital / GHF / SOC path with no alpha/beta split) in addition to the
    # (n_a, n_b) tuple -- may be overridden on subclasses. Always implies
    # complex_capable=True (the spinless operator is complex Hermitian by
    # construction), but the converse does not hold: a solver can be complex_capable
    # on the spatial (n_a, n_b) path without implementing the spinless reduction at
    # all (see e.g. the "cannot serve this layout" note on TorchSCISolver's own
    # dtype-vs-spinless distinction before its native spinless support landed).
    supports_spinless: bool = False
    # whether sci_solver_callable_eigen_states honors a per-call ``init_vectors`` (one
    # Davidson start vector per batch) -- may be overridden on subclasses. Solvers that
    # don't call ``reject_init_vectors``, so an ``init_vectors`` passed to them raises
    # rather than silently giving a cold solve.
    supports_init_vectors: bool = False

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        nroots: int = 1,
        **kwargs: Any,
    ):
        """
        Args:
            hamiltonian: Optional HamiltonianData instance.
                If provided in this constructor you may not override it in sci_solver_callable to avoid ambiguity.
                you can provide either:
                - a HamiltonianData instance directly, or
                - a path to an FCIDUMP file (as a string or Path) from which the HamiltonianData instance will be created
                If None (the default), then no Hamiltonian is stored/"bound" to this class instance.
            symmetrize_spin:
            verbose: Print solver-specific progress / debug output.
            nroots: Number of lowest roots (states) to solve for per batch. Stored
                as ``self.nroots`` for the subclasses to use and implement it.
            **kwargs: Forwarded up the  ``super().__init__``.
        """
        super().__init__(**kwargs)
        self.verbose = verbose
        if isinstance(hamiltonian, (str, os.PathLike)):
            hamiltonian = HamiltonianData.from_fcidump(os.fspath(hamiltonian))
        self.hamiltonian = hamiltonian
        self.symmetrize_spin = symmetrize_spin
        self.nroots = nroots
        if self.nroots < 1:
            raise ValueError(f"nroots must be >= 1, got {self.nroots}")

    def compute_energy(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | None = None,
        bitstring_matrix_batches: List[np.ndarray] | None = None,
        sort_ci_strs: bool = True,
        **solver_call_kwargs: Any,
    ) -> List[SCIResult]:
        """
        Compute energies for each batch of CI strs.

        Note - to get from bitstrings into ci_strs you can use:
        ```
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
        ci_strs = bitstring_matrix_to_ci_strs(bitstrings_to_use, open_shell=False) #or open_shell==True, depending on your case
        ```

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples.  You must provide exactly one of `ci_strs_batches` or `bitstring_matrix`
            bitstring_matrix: List of matrices, expected to be np.ndarray of type bool. You must provide exactly one of `ci_strs_batches` or `bitstring_matrix`
            sort_ci_strs: Sort and deduplicate each batch's alpha/beta CI strings with
                ``np.unique`` before solving. Defaults to ``True`` so solvers that require
                strictly ascending strings (the general-algebra family) work out of the
                box; the bitstring path is already sorted, so this is a no-op there.
                Pair-mode (``mode="gdb"``) solvers are exempt: their
                ``(strs_a, strs_b)`` are parallel ``(alpha_i, beta_i)`` lists, so
                sorting each one independently would break the pairing (they
                dedup/sort the pair list themselves). Pass ``False`` to forward the
                strings exactly as given.
            **solver_call_kwargs: Additional kwargs forwarded to ``sci_solver_callable``.
        """
        if self.hamiltonian is None:
            raise Exception(
                "Hamiltonian data must be provided in constructor to compute_energy. Please provide a HamiltonianData instance in the constructor"
            )

        assert (ci_strs_batches is not None) ^ (bitstring_matrix_batches is not None), (
            "You must provide exactly one of `ci_strs_batches` or `bitstring_matrix`"
        )

        if ci_strs_batches is not None:
            assert isinstance(ci_strs_batches, list)
            if len(ci_strs_batches) == 0:
                return []
            assert isinstance(ci_strs_batches[0], tuple)
            assert isinstance(ci_strs_batches[0][0], np.ndarray)
            assert isinstance(ci_strs_batches[0][1], np.ndarray)
            use_ci_strs_batches = ci_strs_batches
        else:
            assert isinstance(bitstring_matrix_batches, list)
            if len(bitstring_matrix_batches) == 0:
                return []
            assert isinstance(bitstring_matrix_batches[0], np.ndarray)
            assert bitstring_matrix_batches[0].dtype == bool
            use_ci_strs_batches = [
                bitstring_matrix_to_ci_strs(x, open_shell=not self.symmetrize_spin)
                for x in bitstring_matrix_batches
            ]

        # np.unique sorts and deduplicates; solvers in the general-algebra family require
        # strictly ascending strings (e.g. PySCF's selected-CI RDM code is silently wrong
        # otherwise), and the bitstring path already produces them. Pair-mode ("gdb")
        # batches are parallel (alpha_i, beta_i) lists -- strs_a[i] and strs_b[i] form one
        # determinant -- so deduplicating the two arrays independently would break the
        # pairing and change their lengths; those backends dedup/sort the pairs themselves.
        if sort_ci_strs and getattr(self, "mode", "tpb") != "gdb":
            # init_vectors are indexed by the strings as given, so re-index each one
            # to the sorted strings (first occurrence of a duplicate, as np.unique keeps).
            init_vectors = solver_call_kwargs.get("init_vectors")
            if init_vectors is not None:
                if len(init_vectors) != len(use_ci_strs_batches):
                    raise ValueError(
                        f"init_vectors has length {len(init_vectors)} but there are "
                        f"{len(use_ci_strs_batches)} batch(es); pass one entry (or None) "
                        "per batch."
                    )
            sorted_batches = []
            sorted_vectors: list[np.ndarray | None] = []
            for k, (strs_a, strs_b) in enumerate(use_ci_strs_batches):
                uniq_a, ia = np.unique(strs_a, return_index=True)
                uniq_b, ib = np.unique(strs_b, return_index=True)
                sorted_batches.append((uniq_a, uniq_b))
                vec = None if init_vectors is None else init_vectors[k]
                if vec is not None:
                    # Validated here against the strings as passed: past this point
                    # the solver sees the deduplicated grid, so a wrong length would
                    # be reported against the wrong size, or a vector sized for the
                    # deduplicated grid accepted under the wrong indexing.
                    expected = len(strs_a) * len(strs_b)
                    if np.size(vec) != expected:
                        raise ValueError(
                            f"init_vectors[{k}] has {np.size(vec)} entries, expected "
                            f"{expected} (len(strs_a) * len(strs_b), the strings as passed)."
                        )
                    vec = np.asarray(vec).reshape(len(strs_a), len(strs_b))
                    vec = vec[np.ix_(ia, ib)].ravel()
                sorted_vectors.append(vec)
            if init_vectors is not None:
                solver_call_kwargs["init_vectors"] = sorted_vectors
            use_ci_strs_batches = sorted_batches

        ans = self.sci_solver_callable(
            use_ci_strs_batches,
            hcore=self.hamiltonian.hcore,
            eri=self.hamiltonian.eri,
            norb=self.hamiltonian.num_orbitals,
            nelec=(self.hamiltonian.num_elec_a, self.hamiltonian.num_elec_b),
            **solver_call_kwargs,
        )
        return ans

    @abstractmethod
    def sci_solver_callable_eigen_states(  # type: ignore[return]  # body is only the guard
        self,
        ci_strs_batches: "List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray]",
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: "Tuple[int, int] | int",
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Diagonalize the Hamiltonian on each batch's CI-string subspace.

        This is the multi-root primitive every concrete solver implements. It
        returns the lowest eigenpairs of each batch (one or more roots).
        :meth:`sci_solver_callable` wraps this and hands qiskit-addon-sqd the
        ground state per batch, so single-root callers never touch this method
        directly.

        Concrete implementations should call ``super().sci_solver_callable_eigen_states(...)``
        first to run the bound-Hamiltonian ambiguity guard (below).

        Note - to get from bitstrings into ci_strs you can use:
        ```
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
        ci_strs = bitstring_matrix_to_ci_strs(bitstrings_to_use, open_shell=False) #or open_shell==True, depending on your case
        ```

        Two reduction paths, dispatched on ``nelec``'s type (mirrors
        ``GeneralAlgebraSolverBase``/``TorchSCISolver``, the two families that
        implement both):

        * **spatial** — ``nelec`` an ``(n_a, n_b)`` tuple: ``ci_strs_batches``
          is a list of ``(strs_a, strs_b)`` tuples, each defining a subspace
          via the outer product of unique alpha and beta CI strings.
        * **spinless** (flat spin-orbital, no alpha/beta split — GHF / SOC /
          generalized-spinor Hamiltonians) — ``nelec`` a *scalar* total
          particle number ``N``: ``ci_strs_batches`` is a list of flat
          ``strs`` arrays, one ``N``-particle spin-orbital determinant
          bitmask list per subspace. **Not every solver implements this
          path** — a solver that doesn't should raise ``NotImplementedError``
          promptly (see e.g. ``PySCFSolver``, ``DICESolver``, ``SBDSolver``,
          ``DavidsonSolver``, ``TruncatedDavidsonSolver``) rather than let a bare tuple-unpack or
          index crash with an unrelated-looking ``TypeError``/``ValueError``
          deep inside the implementation.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` tuples (spatial), or
                a list of flat ``strs`` int64 arrays (spinless — see above).
                Each spatial tuple defines a subspace via the outer product
                of unique alpha and beta CI strings (int64 arrays where bit
                ``i`` of each integer indicates occupancy of orbital ``i``).
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals. Either full 4-D
                ``(norb,)*4`` or a packed form (4-fold / 8-fold) — what's
                accepted depends on the concrete solver.
            norb: Number of spatial orbitals (spin-orbitals ``nmo`` on the
                spinless path).
            nelec: ``(n_alpha, n_beta)`` (spatial), or a scalar total ``N``
                (spinless — see above; not every solver implements this).
            **kwargs: Solver-specific optional arguments. See each
                concrete solver for the accepted set.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per input batch,
            in input order. Each holds one :class:`SCIResult` per solved root,
            ascending in energy (``states[0]`` is the ground state). Single-root
            solvers return an ``EigenStates`` of length one. ``energy`` fields
            are electronic only; callers add ``e_nuc`` themselves to obtain
            chemical totals.

        Raises:
            NotImplementedError: ``nelec`` is a scalar (the spinless path) and
                this solver does not implement it.
        """

        if self.hamiltonian is not None:
            # A scalar nelec (the spinless path) has no (n_a, n_b) to compare against
            # a bound spatial Hamiltonian -- constructing one alongside a scalar-nelec
            # call is nonsensical (the spinless reduction never binds a Hamiltonian, so a
            # caller on this path should not pass "hamiltonian" in its solver kwargs), so flag it as the same ambiguity rather than
            # let `nelec != (tuple)` spuriously fire on every call.
            nelec_mismatch = (
                True
                if is_spinless_request(nelec)
                else nelec != (self.hamiltonian.num_elec_a, self.hamiltonian.num_elec_b)
            )
            if (
                (id(hcore) != id(self.hamiltonian.hcore))
                or (id(eri) != id(self.hamiltonian.eri))
                or (norb != self.hamiltonian.num_orbitals)
                or nelec_mismatch
            ):
                raise Exception(
                    "Ambiguous Hamiltonian: provided in constructor but also passed to sci_solver_callable. Please provide the Hamiltonian in only one place to avoid ambiguity."
                )

    def sci_solver_callable(
        self,
        ci_strs_batches: "List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray]",
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: "Tuple[int, int] | int",
        **kwargs: Any,
    ) -> List[SCIResult]:
        """
        Diagonalize each batch's CI-string subspace, returning ground states.

        The qiskit-addon-sqd entry point: matches the ``sci_solver`` signature
        expected by ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``.
        Thin wrapper over :meth:`sci_solver_callable_eigen_states` that keeps
        each batch's ground state (root 0) and discards the excited states,
        since the SQD recovery loop consumes only the ground state. To access
        excited states, call :meth:`sci_solver_callable_eigen_states` directly.

        Note - to get from bitstrings into ci_strs you can use:
        ```
        from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
        ci_strs = bitstring_matrix_to_ci_strs(bitstrings_to_use, open_shell=False) #or open_shell==True, depending on your case
        ```

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` tuples (spatial), or
                a list of flat ``strs`` arrays (spinless — a scalar ``nelec``;
                see :meth:`sci_solver_callable_eigen_states`). Each spatial
                tuple defines a subspace via the outer product of unique
                alpha and beta CI strings (int64 arrays where bit ``i`` of
                each integer indicates occupancy of orbital ``i``).
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals. Either full 4-D
                ``(norb,)*4`` or a packed form (4-fold / 8-fold) — what's
                accepted depends on the concrete solver.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)`` (spatial), or a scalar total ``N``
                (spinless — not every solver implements this; see
                :meth:`sci_solver_callable_eigen_states`).
            **kwargs: Solver-specific optional arguments. See each
                concrete solver for the accepted set.

        Returns:
            One :class:`SCIResult` (the ground state) per input batch, in input
            order. ``energy`` fields are electronic only; callers add ``e_nuc``
            themselves to obtain chemical totals.
        """
        eigen_states_list = self.sci_solver_callable_eigen_states(
            ci_strs_batches=ci_strs_batches,
            hcore=hcore,
            eri=eri,
            norb=norb,
            nelec=nelec,
            **kwargs,
        )
        return [eigen_states.states[0] for eigen_states in eigen_states_list]

    def __repr__(self) -> str:
        """String representation of the EnergyCalculatorBase instance."""
        return f"{self.__class__.__name__}(verbose={self.verbose})"
