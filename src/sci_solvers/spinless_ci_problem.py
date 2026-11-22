"""Spinless (spinor_mo / GHF / SOC) complex Hermitian selected-CI problem.

The complex Hermitian CI path in :mod:`sci_solvers.complex_ci_problem` is
**spatial-orbital + ``(n_alpha, n_beta)``** only: it takes ``(norb, norb)`` /
``(norb,)^4`` spatial integrals and spin-doubles them internally to an interleaved
``2i``/``2i+1`` spin-orbital layout that populates only the four *same-spin* ERI
blocks -- it zeros every spin-flip block. That is exactly wrong for a **genuinely
spinless** spin-orbital Hamiltonian: a single flat block of ``nmo`` complex
spin-orbitals whose off-diagonal spin blocks (spin-flip) are non-zero and which
carries only a *total* particle number ``N`` (no ``(n_a, n_b)`` factorization).
This is the operator a GHF / spin-orbit-coupled / generalized-spinor calculation
produces (e.g. ``pbc_hamiltonian_generator``'s ``spinor_mo`` integrals).

This module builds a complex-Hermitian **selected** CI operator directly over an
arbitrary list of ``N``-particle spin-orbital determinants, with **no** spin
assumption. It reuses the package's spin-agnostic scalar Slater-Condon core in
``torch_sci_solver.sbd_ref.determinants`` (``one_excite`` / ``two_excite`` /
``parity`` / ``orbital_difference``), which indexes arbitrary spin-orbital
bitmasks and carries ``complex128`` through unchanged. Only the diagonal
:func:`_zero_excite_spinless` is reimplemented locally to drop the same-spin
``(i % 2) == (j % 2)`` exchange guard that ``sbd_ref``'s ``zero_excite`` applies
(correct for the spin-doubled spatial layout, wrong for a spinless operator, where
the exchange term applies to *every* occupied pair -- the integrals themselves
carry the spin structure).

The result is wrapped as a matrix-free
:class:`~sci_solvers.eigen_problem.HermitianOperator`, so the complex-capable
adapters diagonalize it through the same interface as the spatial complex path.

Scope / limitations
--------------------
* Input is a single ``(nmo, nmo)`` complex Hermitian ``hcore``, a
  ``(nmo,)^4`` complex chemist-order ``eri`` (with both Hermitian symmetries),
  a *scalar* total particle number ``n_particles``, and an explicit list of
  spin-orbital determinant bitmasks ``strs``. There is **no** ``(n_a, n_b)`` split.
* A real Hamiltonian is rejected (use the real ``EigenProblem`` path); an
  ``(n_a, n_b)`` tuple is rejected (use
  :func:`~sci_solvers.complex_ci_problem.build_selected_complex_ci_problem`), so
  the three paths stay disentangled.
* The reduced density matrices (see :func:`spinless_rdm12`) reuse the complex-
  correct builders in :mod:`sci_solvers.complex_rdm` via the ``(N, 0)`` sector
  embedding. They are **selected-CI-aware**: pass the selected determinant list as
  ``strs`` and the RDMs are correct for an arbitrary determinant **subset** in any
  order. Each excitation is restricted to a *selected source* (the ket amplitude is
  zero otherwise), and the intermediate determinant axis of ``B = E_pq|C>`` is
  re-indexed onto the **connected set** -- the selected determinants plus every
  destination reachable from them by one excitation (via a ``row_map`` built as a
  :class:`~sci_solvers.complex_rdm.SparseAddressMap` keyed directly on determinant
  bitmasks -- see :func:`~sci_solvers.complex_rdm._link_arrays_selected`). The
  connected set (not just the selected set) is
  required because the two-body Gram term ``sum_K conj(B[q,p,K]) B[r,s,K]`` sums over
  intermediates ``K`` outside the selected space; its size is bounded by
  ``len(strs) * n_links``, never the astronomically larger full canonical space.
  Omitting ``strs`` (or passing the full canonical
  ``cistring.make_strings(range(nmo), N)`` list) takes the zero-overhead
  full-canonical path, byte-identical to before.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from sci_solvers.complex_rdm import SparseAddressMap
from sci_solvers.eigen_problem import HermitianOperator


# ---------------------------------------------------------------------------
# Complex spin-orbital integral adapter
# ---------------------------------------------------------------------------
class _ComplexSpinOrbitalIntegrals:
    """``.value`` adapter over raw complex spin-orbital integral arrays.

    Exposes the exact access pattern the ``sbd_ref`` Slater-Condon evaluators
    expect (``I1.value(i, j)``, ``I2.value(p, q, r, s)``, ``I2.direct_value``,
    ``I2.exchange_value``) but over the **complex** ``(nmo, nmo)`` ``hcore`` and
    ``(nmo,)^4`` chemist-order ``eri`` -- with **no** ``float()`` cast (which would
    drop the imaginary part) and **no** ``(i % 2) == (j % 2)`` spin-selection
    (which would zero every spin-flip block). The ``sbd_ref`` ``OneInt``/``TwoInt``
    classes do both, so they cannot be reused for a spinless complex operator.
    """

    __slots__ = ("n_spinorb", "_h", "_eri")

    def __init__(self, hcore: np.ndarray, eri: np.ndarray):
        """Store the complex integral arrays; ``n_spinorb`` is ``nmo``."""
        self.n_spinorb = int(hcore.shape[0])
        self._h = hcore
        self._eri = eri

    def value(self, i: int, j: int):
        """``h[i, j]`` (one-electron; used via the ``I1`` role)."""
        return self._h[i, j]

    # The two-electron role (``I2``) uses the (p, q, r, s) / direct / exchange API.
    def value4(self, p: int, q: int, r: int, s: int):
        """``(pq|rs)`` chemist notation, complex, no spin-selection rule."""
        return self._eri[p, q, r, s]

    def direct_value(self, i: int, j: int):
        """``(ii|jj)``."""
        return self._eri[i, i, j, j]

    def exchange_value(self, i: int, j: int):
        """``(ij|ji)``."""
        return self._eri[i, j, j, i]


class _TwoIntView:
    """Two-electron facade whose ``.value`` is the 4-index ERI (chemist order)."""

    __slots__ = ("_ints",)

    def __init__(self, ints: _ComplexSpinOrbitalIntegrals):
        """Wrap the shared integral store."""
        self._ints = ints

    def value(self, p: int, q: int, r: int, s: int):
        """``(pq|rs)``."""
        return self._ints.value4(p, q, r, s)

    def direct_value(self, i: int, j: int):
        """``(ii|jj)``."""
        return self._ints.direct_value(i, j)

    def exchange_value(self, i: int, j: int):
        """``(ij|ji)``."""
        return self._ints.exchange_value(i, j)


# ---------------------------------------------------------------------------
# Spinless Slater-Condon (diagonal drops the same-spin exchange guard)
# ---------------------------------------------------------------------------
def _zero_excite_spinless(det: int, n_spinorb: int, I0, I1, I2):
    """``<D|H|D>`` for a spinless spin-orbital determinant.

    Same as ``sbd_ref.determinants.zero_excite`` but **without** the
    ``(i % 2) == (j % 2)`` exchange guard: for a genuinely spinless operator the
    exchange term ``(ij|ji)`` applies to *every* occupied pair (the spin structure
    lives in the integrals, not in an orbital-index parity rule). Carries complex
    values through unchanged.
    """
    from torch_sci_solver.sbd_ref.determinants import getocc

    energy = I0
    for i in range(n_spinorb):
        if not getocc(det, i):
            continue
        energy = energy + I1.value(i, i)
        for j in range(i + 1, n_spinorb):
            if not getocc(det, j):
                continue
            energy = energy + I2.direct_value(i, j) - I2.exchange_value(i, j)
    return energy


def _hij_spinless(det_a: int, det_b: int, n_spinorb: int, I0, I1, I2):
    """Generic ``<D_a|H|D_b>`` for arbitrary spinless spin-orbital bitmasks.

    Mirrors ``sbd_ref.determinants.hij`` (dispatch on the number of differing
    spin-orbitals) but calls :func:`_zero_excite_spinless` on the diagonal. The
    one- and two-excitation branches reuse ``sbd_ref``'s ``one_excite`` /
    ``two_excite`` unchanged -- they are spin-agnostic and complex-safe.
    """
    from torch_sci_solver.sbd_ref.determinants import (
        one_excite,
        orbital_difference,
        two_excite,
    )

    c, d = orbital_difference(det_a, det_b, n_spinorb)
    nc = len(c)
    if nc == 0:
        return _zero_excite_spinless(det_b, n_spinorb, I0, I1, I2)
    if nc == 1:
        return one_excite(det_b, d[0], c[0], I1, I2, n_spinorb)
    if nc == 2:
        return two_excite(det_b, d[0], d[1], c[0], c[1], I2)
    return 0.0


# ---------------------------------------------------------------------------
# Spinless CI state (PairSCIState-style lookalike)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpinlessSCIState:
    """Spinless analog of ``qiskit_addon_sqd.fermion.SCIState``.

    ``SCIState`` requires ``amplitudes.shape == (len(ci_strs_a), len(ci_strs_b))``
    -- a Cartesian alpha/beta product. A spinless basis is a single length-``N``
    list of ``N``-particle spin-orbital determinants; the eigenvector is a flat
    length-``N`` 1-D vector and there is one string array, not two. Storing that in
    an ``SCIState`` fails its 2-D shape check, so -- exactly as ``PairSCIState``
    does for pair mode -- this lookalike is returned instead. Consumers reading the
    fields by name see equivalent data; consumers that reshape to 2-D fail loudly
    at the type check rather than silently mis-indexing.
    """

    amplitudes: np.ndarray  # flat length-N 1-D vector
    ci_strs: np.ndarray  # parallel length-N spin-orbital bitmask list
    norb: int  # number of spin-orbitals (nmo)
    nelec: Tuple[int, int]  # (N, 0) embedding for the (N,0)-sector RDM reuse

    def save(self, filename):
        """Save the SpinlessSCIState object to an .npz file."""
        np.savez(
            filename,
            amplitudes=self.amplitudes,
            ci_strs=self.ci_strs,
            norb=self.norb,
            nelec=self.nelec,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
# Auto matrix-free threshold: the "Large" regime where the dense (dim, dim)
# operator (16 * dim**2 bytes) stops fitting in RAM. dim >= 50_000 -> ~40 GB;
# nmo >= 48 spin-orbitals is the smallest system that realistically reaches it.
_MATRIXFREE_DIM_THRESHOLD = 50_000
_MATRIXFREE_NMO_THRESHOLD = 48


def _cistring_row_to_int(row) -> int:
    """Convert one row of ``cistring.make_strings`` output to a bitmask int.

    Below pyscf's internal int64-packing limit, each row is already a plain
    bitmask scalar. Past it, ``cistring.make_strings`` switches to returning
    an ``OIndexList`` row -- an array of *occupied orbital indices* rather
    than a bitmask -- so ``int()`` on it raises. Detect the shape and handle
    both: a 0-d (scalar) row converts directly, an array row is the set of
    occupied positions to OR together.
    """
    import numpy as np

    arr = np.asarray(row)
    if arr.ndim == 0:
        return int(arr)
    bitmask = 0
    for idx in arr.tolist():
        bitmask |= 1 << int(idx)
    return bitmask


def _is_full_canonical(det_list, nmo: int, n: int) -> bool:
    """Whether ``det_list`` is exactly the full canonical ``(nmo, N)`` string set.

    The matrix-free engine (:func:`_build_spinless_matrixfree`) contracts on the
    full ``(N, 0)`` CI grid via ``pyscf.fci.direct_nosym``, so it is valid only for
    the complete canonical determinant list -- the same full-canonical contract the
    RDM reuse (see the module docstring) already requires. A genuinely selected
    strict subset has no matrix-free representation and must use the dense path.

    The check is cheap: match the length against ``cistring.num_strings`` and then
    compare against the canonical list element-by-element (the canonical order is
    what the ``(N, 0)`` embedding assumes anyway, so any full-length reordering
    would also break the RDM path -- we require the canonical order outright).
    """
    from pyscf.fci import cistring

    if len(det_list) != int(cistring.num_strings(nmo, n)):
        return False
    canonical = [_cistring_row_to_int(s) for s in cistring.make_strings(range(nmo), n)]
    return det_list == canonical


def build_spinless_complex_ci_problem(
    hcore: np.ndarray,
    eri: np.ndarray,
    n_particles: int,
    strs: np.ndarray,
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
    matrix_free: Union[bool, None] = None,
) -> HermitianOperator:
    """Build a complex Hermitian *spinless* selected-CI problem.

    The spin-orbital counterpart of
    :func:`~sci_solvers.complex_ci_problem.build_selected_complex_ci_problem` for a
    **single flat block of spin-orbitals** with no alpha/beta factorization. The
    operator is Hermitian for complex Hermitian integrals (real spectrum). Two
    engines assemble it, dispatched transparently:

    * **Dense** (:func:`_build_spinless_dense`): an ``O(dim**2)`` Slater-Condon
      double loop (:func:`_hij_spinless`) materializing the full ``(dim, dim)``
      matrix. Correct for **any** selected determinant list -- including a genuinely
      selected strict subset -- and the trusted small-size reference.
    * **Matrix-free** (:func:`_build_spinless_matrixfree`): reuses
      ``pyscf.fci.direct_nosym.contract_2e`` on the ``(N, 0)`` sector, which applies
      ``H`` to a CI vector on the **un-doubled** ``nmo``-orbital integrals with no
      ``O(dim**2)`` storage. It is element-for-element identical to the dense
      operator (verified to ~1e-14), so no ``sigma`` similarity is needed. It is
      valid **only for the full canonical determinant list** (see
      :func:`_is_full_canonical`), the regime the auto-routed solver / SQD supplies.

    Dispatch: ``matrix_free=None`` (default) picks matrix-free automatically when the
    list is full canonical **and** the problem is Large
    (``dim >= 50_000`` or ``nmo >= 48``), else dense. ``matrix_free=True`` forces the
    matrix-free engine (raising if the list is not full canonical);
    ``matrix_free=False`` forces the dense engine.

    Args:
        hcore: Complex Hermitian ``(nmo, nmo)`` one-electron spin-orbital integrals.
        eri: Full dense complex chemist-order ``(nmo, nmo, nmo, nmo)`` ERI.
        n_particles: Total particle number ``N`` (a scalar -- **not** an
            ``(n_a, n_b)`` tuple).
        strs: Selected ``N``-particle spin-orbital determinant bitmasks (int array).
        check_hermitian: If ``True`` (default) verify ``hcore`` is Hermitian and the
            ERI carries both Hermitian symmetries before building.
        atol: Absolute tolerance for the Hermiticity checks. The default
            ``1e-7`` covers the near-zero-element noise floor of a
            two-component SCF export (elements that should vanish but carry
            ~1e-8 absolute round-off), which no ``rtol`` can absorb.
        rtol: Relative tolerance for the Hermiticity checks. Integrals from a
            two-component SCF (SOC / GHF) are Hermitian only up to a round-off that
            scales with their magnitude, so a residual of ``rtol * max|element|``
            (often ~1e-7 for O(1e-1) integrals) is round-off, not a genuine
            violation. ``np.allclose`` compares against ``atol + rtol * |b|``, so a
            nonzero ``rtol`` accepts that scaled round-off while still catching a
            residual comparable to the operator itself.
        matrix_free: ``None`` (auto, default), ``True`` (force matrix-free), or
            ``False`` (force dense). See dispatch above.

    Returns:
        A matrix-free :class:`~sci_solvers.eigen_problem.HermitianOperator` of
        dimension ``len(strs)`` whose ``matvec`` applies the complex Hermitian
        spinless selected-CI Hamiltonian and whose ``hdiag`` is its (real)
        diagonal. Carries ``strs``, ``norb=nmo``, ``nelec=(N, 0)``, ``hcore``,
        ``eri`` as attributes for the downstream RDM/energy consumers. The dense
        engine additionally attaches the assembled matrix as ``problem._dense``;
        the matrix-free engine leaves ``problem._dense`` as ``None``.

    Raises:
        ValueError: If shapes are inconsistent, the integrals are real, an
            ``(n_a, n_b)`` tuple is passed for ``n_particles``, (when
            ``check_hermitian``) the integrals are not Hermitian, or
            ``matrix_free=True`` is requested for a non-canonical determinant list.
    """
    if isinstance(n_particles, (tuple, list, np.ndarray)):
        raise ValueError(
            "build_spinless_complex_ci_problem takes a scalar total particle "
            f"number, got n_particles={n_particles!r}. For an (n_alpha, n_beta) "
            "spatial-orbital problem use build_selected_complex_ci_problem."
        )

    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    nmo = int(hcore.shape[0])
    n = int(n_particles)

    if hcore.shape != (nmo, nmo):
        raise ValueError(f"hcore shape {hcore.shape} != ({nmo}, {nmo}).")
    if eri.shape != (nmo, nmo, nmo, nmo):
        raise ValueError(
            f"eri must be the full dense (nmo,)*4 tensor, got {eri.shape} for nmo={nmo}."
        )
    if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        raise ValueError(
            "build_spinless_complex_ci_problem is for complex Hermitian integrals; "
            "got real integrals. Use the real EigenProblem / chemistry solver path."
        )

    if check_hermitian:
        if not np.allclose(hcore, hcore.conj().T, atol=atol, rtol=rtol):
            raise ValueError("hcore is not Hermitian (hcore != hcore^H) within tolerance.")
        # The CI operator 0.5 sum (pq|rs) a_p^ a_r^ a_s a_q is Hermitian iff the ERI
        # has both (pq|rs) == conj((qp|sr)) and (pq|rs) == (rs|pq).
        if not np.allclose(eri, np.conj(np.transpose(eri, (1, 0, 3, 2))), atol=atol, rtol=rtol):
            raise ValueError(
                "eri does not satisfy (pq|rs) == conj((qp|sr)) within tolerance; "
                "the resulting CI operator would not be Hermitian."
            )
        if not np.allclose(eri, np.transpose(eri, (2, 3, 0, 1)), atol=atol, rtol=rtol):
            raise ValueError(
                "eri does not satisfy (pq|rs) == (rs|pq) within tolerance; "
                "the resulting CI operator would not be Hermitian."
            )

    strs = np.ascontiguousarray(strs).astype(object)  # arbitrary-precision Python ints
    det_list = [int(s) for s in strs.tolist()]
    dim = len(det_list)

    full_canonical = _is_full_canonical(det_list, nmo, n)
    if matrix_free is True:
        if not full_canonical:
            raise ValueError(
                "matrix_free=True requires the full canonical cistring list for "
                f"(nmo={nmo}, N={n}); got a selected/reordered list of {dim} strings. "
                "Use the dense engine (matrix_free=False / None) for a selected subset."
            )
        use_matrix_free = True
    elif matrix_free is False:
        use_matrix_free = False
    else:  # auto
        is_large = dim >= _MATRIXFREE_DIM_THRESHOLD or nmo >= _MATRIXFREE_NMO_THRESHOLD
        use_matrix_free = full_canonical and is_large

    if use_matrix_free:
        return _build_spinless_matrixfree(hcore, eri, nmo, n, det_list)
    return _build_spinless_dense(hcore, eri, nmo, n, det_list)


def _build_spinless_dense(hcore, eri, nmo: int, n: int, det_list) -> HermitianOperator:
    """Dense ``O(dim**2)`` spinless operator via the Slater-Condon double loop.

    Assembles the full ``(dim, dim)`` complex Hermitian matrix from
    :func:`_hij_spinless` -- correct for an arbitrary selected determinant list, and
    the trusted small-size reference for the matrix-free engine. See
    :func:`build_spinless_complex_ci_problem` for the public contract.
    """
    dim = len(det_list)
    I1 = _ComplexSpinOrbitalIntegrals(hcore, eri)
    I2 = _TwoIntView(I1)
    I0 = 0.0

    # Assemble the dense Hermitian operator once. dim = len(strs) is the selected-CI
    # dimension (the sampled subspace), not the full CAS -- the same regime the
    # spatial selected path assembles for the numpy adapters.
    H = np.zeros((dim, dim), dtype=np.complex128)
    for a in range(dim):
        da = det_list[a]
        H[a, a] = _hij_spinless(da, da, nmo, I0, I1, I2)
        for b in range(a + 1, dim):
            db = det_list[b]
            hab = _hij_spinless(da, db, nmo, I0, I1, I2)
            H[a, b] = hab
            # Hermitian: the (b, a) element is the conjugate. hij is evaluated on
            # the ket for parity, so compute the mirror directly to avoid relying
            # on a convention and symmetrize explicitly below.
            H[b, a] = np.conj(hab)
    # Symmetrize to kill any residual asymmetry from finite-precision integrals.
    H = 0.5 * (H + H.conj().T)
    hdiag = np.real(np.diagonal(H)).copy()

    def matvec(x: np.ndarray) -> np.ndarray:
        """Apply H to a ``(dim,)`` vector or ``(dim, k)`` block."""
        return H @ np.asarray(x, dtype=np.complex128)

    problem = HermitianOperator.from_matvec(matvec, hdiag, dim, dtype=np.complex128)
    _attach_spinless_attributes(problem, det_list, nmo, n, hcore, eri)
    # Also expose the assembled matrix for the dense adapter / tests.
    problem._dense = H
    return problem


def _build_spinless_matrixfree(hcore, eri, nmo: int, n: int, det_list) -> HermitianOperator:
    """Matrix-free spinless operator via ``pyscf.fci.direct_nosym`` on ``(N, 0)``.

    The spinless operator over ``nmo`` spin-orbitals with a total of ``N`` particles
    is a single spin species on the ``(N, 0)`` sector, so ``direct_nosym.contract_2e``
    applies ``H`` to a CI vector matrix-free on the **un-doubled** ``nmo``-orbital
    integrals -- peak RAM ``O(dim * k + nmo**4)`` instead of the dense ``16 * dim**2``.
    Element-for-element identical to :func:`_build_spinless_dense` (verified to
    ~1e-14), so no ``sigma`` similarity is needed.

    Requires ``det_list`` to be the full canonical ``(nmo, N)`` string list (the
    caller guarantees this via :func:`_is_full_canonical`): ``contract_2e`` operates
    on the complete ``(na, 1)`` CI grid, and the amplitude order it consumes/returns
    is exactly ``cistring.make_strings(range(nmo), N)`` -- the same order the
    ``(N, 0)`` RDM reuse assumes.
    """
    from pyscf.fci import cistring, direct_nosym

    dim = len(det_list)
    na = int(cistring.num_strings(nmo, n))
    assert na == dim, f"matrix-free engine expects full canonical dim {na}, got {dim}"
    nelec = (n, 0)

    # absorb_h1e folds the one-body term into a two-body kernel h2e once (0.5 for
    # the CI operator's 1/2 convention, matching the dense builder and the oracle).
    h2e = direct_nosym.absorb_h1e(hcore, eri, nmo, nelec, 0.5)

    def _apply_column(col_flat: np.ndarray) -> np.ndarray:
        """H @ (single (dim,) column) via one contract_2e on the (na, 1) grid."""
        ci = np.ascontiguousarray(col_flat, dtype=np.complex128).reshape(na, 1)
        return direct_nosym.contract_2e(h2e, ci, nmo, nelec).reshape(-1)

    def matvec(x: np.ndarray) -> np.ndarray:
        """Apply H to a ``(dim,)`` vector or ``(dim, k)`` block, column by column.

        contract_2e is not vmap-friendly, so a block is applied one column at a
        time -- exactly as the spatial complex path
        (``build_selected_complex_ci_problem``) does for its torch kernel.
        """
        xa = np.asarray(x, dtype=np.complex128)
        if xa.ndim == 1:
            return _apply_column(xa)
        cols = [_apply_column(xa[:, j]) for j in range(xa.shape[1])]
        return np.column_stack(cols)

    # Diagonal for preconditioning: the real diagonal of H, matrix-free via the
    # local Slater-Condon diagonal (O(dim * nmo**2), no dim**2 storage). Only its
    # real part is used by HermitianOperator, and it is only a preconditioner.
    I1 = _ComplexSpinOrbitalIntegrals(hcore, eri)
    I2 = _TwoIntView(I1)
    hdiag = np.empty(dim, dtype=np.float64)
    for a, da in enumerate(det_list):
        hdiag[a] = np.real(_zero_excite_spinless(da, nmo, 0.0, I1, I2))

    problem = HermitianOperator.from_matvec(matvec, hdiag, dim, dtype=np.complex128)
    _attach_spinless_attributes(problem, det_list, nmo, n, hcore, eri)
    # Matrix-free: no dense matrix is materialized.
    problem._dense = None
    return problem


def _attach_spinless_attributes(problem, det_list, nmo: int, n: int, hcore, eri) -> None:
    """Attach the physical attributes the downstream RDM/energy/state consumers read.

    Mirrors ``build_selected_complex_ci_problem``: ``hcore``/``eri`` stay the ORIGINAL
    complex arrays (not float64-cast) so the complex-correct RDM/energy rebuild sees
    the true integrals.
    """
    problem.strs = np.asarray(det_list, dtype=object)
    problem.norb = nmo
    problem.nelec = (n, 0)
    problem.hcore = hcore
    problem.eri = eri


# ---------------------------------------------------------------------------
# RDMs / energy / spin via the (N, 0) sector embedding of complex_rdm
# ---------------------------------------------------------------------------
# Auto-chunk threshold: reuse the same "Large" cutoff as the operator engine. The
# single-excitation tensor B = E_pq|C> is (nmo, nmo, Na) complex128 == 16*nmo**2*Na
# bytes; at nmo=120, Na=50k that is ~11.5 GB, so above this dim we build B in
# blocks of (p, q) rows instead of all at once (the (N,0) sector has no beta
# contribution -- Bb == 0, Gab == 0 -- so only the alpha channel is built).
_RDM_CHUNK_DIM_THRESHOLD = 50_000


def _selected_address_map(strs) -> SparseAddressMap:
    """Map selected determinant bitmasks to their row in the amplitude vector.

    ``strs`` may be any subset of ``cistring.make_strings(range(nmo), N)``, in any
    order. Returns the ``row_map`` the ``complex_rdm`` link filter consumes to
    restrict every excitation to selected -> selected transitions: a
    :class:`~sci_solvers.complex_rdm.SparseAddressMap` where ``row_map[bitmask]`` is
    the row of that determinant in the selected amplitude vector, or ``-1`` when the
    determinant is not selected.

    Keyed directly on the determinant **bitmask** (not a pyscf canonical ``cistring``
    address): the link table this ``row_map`` filters
    (:func:`~sci_solvers.complex_rdm._link_arrays_selected`) is itself built from
    bitmasks, bypassing pyscf's ``cistring`` addressing entirely (its C ``binomial()``
    helper is an unmemoized, exponential-in-``norb`` recursion once ``norb > 20`` --
    unusable at e.g. ``nmo=56``, regardless of how few determinants are selected).
    """
    det_ints = np.asarray([int(s) for s in strs], dtype=np.int64)
    rows = np.arange(det_ints.shape[0], dtype=np.int64)
    return SparseAddressMap(det_ints, rows)


def _is_full_canonical_strs(strs, nmo: int, n: int) -> bool:
    """Whether ``strs`` is exactly the full canonical ``(nmo, N)`` string list.

    When it is (the default full-CAS regime), the selected-CI ``row_map`` is a
    no-op identity and is skipped so the RDM path stays byte-identical to before.
    """
    return _is_full_canonical([int(s) for s in strs], nmo, n)


def _row_map_for(strs, nmo: int, n: int):
    """Return the selected-CI ``row_map`` for ``strs``, or ``None`` for full canonical.

    ``None`` is returned when ``strs`` is ``None`` or exactly the full canonical
    ``(nmo, N)`` list (in canonical order), so the RDM builders take their
    zero-overhead full-canonical path with no behavior change. Otherwise a
    :func:`_selected_address_map` restricting excitations to the selected subset is
    built and returned.
    """
    if strs is None:
        return None
    if _is_full_canonical_strs(strs, nmo, n):
        return None
    return _selected_address_map(strs)


def _sorted_strs_for(strs, row_map):
    """Sorted-ascending ``int64`` bitmasks for the selected-source link builder.

    Returns ``None`` on the full-canonical path (``row_map is None``), else the
    selected determinant bitmasks sorted ascending. Not required for correctness --
    :func:`~sci_solvers.complex_rdm._link_arrays_selected` makes no ordering
    assumption on ``strs`` -- but harmless and kept for a deterministic link-table
    row order. The link builder addresses sources by bitmask directly and the
    ``row_map`` filter keys on those same bitmasks, so re-sorting the strs here does
    not disturb the amplitude-to-row alignment ``row_map`` encodes.
    """
    if row_map is None:
        return None
    return np.sort(np.asarray([int(s) for s in strs], dtype=np.int64))


def _alpha_link_sorted(nmo: int, n: int, *, row_map=None, strs=None):
    """Flatten the alpha ``cistring`` link table, sorted by the ``(p, q)`` operator.

    Returns ``(flat_pq, dst, src, sign, starts, n_det)`` where the link entries are
    sorted by ``flat_pq = p * nmo + q`` and ``starts`` is the ``(nmo*nmo + 1,)``
    CSR-style offset array (``starts[a] : starts[a+1]`` is the contiguous slice of
    links for operator index ``a``). This lets :func:`_build_Ba_block` slice out a
    block of ``(p, q)`` rows in ``O(links_in_block)`` with no per-block scan of the
    full table. ``n_det`` is the size of the determinant axis ``dst`` indexes into.

    When ``row_map`` is given, the link table is restricted to a selected
    (non-full-canonical) determinant list: only links with a **selected source** are
    kept (the ket amplitude ``C_J`` is zero otherwise), and both ``src``/``dst`` are
    re-indexed onto the *connected set* -- the selected determinants plus every
    destination reachable from them by one excitation. The intermediate ``dst`` must
    span this connected set (not just the selected set) because the 2-RDM Gram term
    ``sum_K conj(B[q,p,K]) B[r,s,K]`` sums over intermediates outside the selected
    space; ``n_det`` is then ``len(connected)``. ``row_map=None`` (default) keeps the
    full-canonical link table unchanged (``n_det == num_strings(nmo, n)``).

    ``strs`` (default ``None``), the sorted selected determinant bitmasks parallel to
    ``row_map``, builds the selected path's link table via the selected-source
    :func:`~sci_solvers.complex_rdm._link_arrays_selected` so the full canonical link
    table is never enumerated (required for a large CAS). It is ignored on the
    full-canonical path (``row_map is None``).
    """
    from pyscf.fci import cistring

    from sci_solvers.complex_rdm import _link_arrays, _link_arrays_selected

    if row_map is not None and strs is not None:
        p, q, dst, src, sign = _link_arrays_selected(nmo, n, strs)
    else:
        p, q, dst, src, sign = _link_arrays(nmo, n)
    if row_map is not None:
        keep = row_map[src] >= 0
        p, q, dst, src, sign = p[keep], q[keep], dst[keep], src[keep], sign[keep]
        sel_addr = row_map.present_addrs()
        connected = np.union1d(sel_addr, dst)
        canon_to_conn = SparseAddressMap(connected, np.arange(connected.shape[0], dtype=np.int64))
        src = canon_to_conn[src]
        dst = canon_to_conn[dst]
        n_det = int(connected.shape[0])
    else:
        n_det = int(cistring.num_strings(nmo, n))
    flat = (p * nmo + q).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    starts = np.searchsorted(flat_s, np.arange(nmo * nmo + 1, dtype=np.int64))
    return flat_s, dst[order], src[order], sign[order], starts, n_det


def _embed_connected(
    Cvec: np.ndarray, nmo: int, n: int, row_map: SparseAddressMap, *, strs=None
) -> np.ndarray:
    """Lay the selected amplitude vector onto the connected-set determinant axis.

    Mirrors the ``src``/``dst`` re-indexing of :func:`_alpha_link_sorted`: builds the
    same connected set (selected determinants union their single-excitation
    reachable destinations) and returns ``C_conn`` of length ``len(connected)``,
    zero except at the connected index of each selected determinant. The chunked
    ``B`` block and the ``dm1`` contraction ``B @ conj(C_conn)`` then bind to this
    shared connected axis.

    ``strs`` (the sorted selected bitmasks parallel to ``row_map``) builds the link
    table from the selected sources only (see :func:`_alpha_link_sorted`); the
    connected set it produces is identical to the full-canonical filter's.
    """
    from sci_solvers.complex_rdm import _link_arrays, _link_arrays_selected

    if strs is not None:
        _p, _q, dst, src, _sign = _link_arrays_selected(nmo, n, strs)
    else:
        _p, _q, dst, src, _sign = _link_arrays(nmo, n)
    keep = row_map[src] >= 0
    dst = dst[keep]
    sel_addr = row_map.present_addrs()
    connected = np.union1d(sel_addr, dst)
    canon_to_conn = SparseAddressMap(connected, np.arange(connected.shape[0], dtype=np.int64))
    C_conn = np.zeros(connected.shape[0], dtype=Cvec.dtype)
    C_conn[canon_to_conn[sel_addr]] = Cvec[row_map[sel_addr]]
    return C_conn


def _build_Ba_block(Cvec: np.ndarray, link_sorted, nmo: int, a0: int, a1: int) -> np.ndarray:
    """Build the ``(a1 - a0, n_det)`` slice ``B[a0:a1] = E_pq|C>`` of the alpha tensor.

    ``a`` is the flattened operator index ``p * nmo + q``; ``Cvec`` is the amplitude
    vector aligned to the determinant axis ``link_sorted`` indexes (the selected
    connected-set embedding on the selected path, the full canonical vector
    otherwise). Only the links for the requested block are scattered, so peak
    residency is ``(block, n_det)`` complex128 rather than the full
    ``(nmo*nmo, n_det)``.
    """
    _ = nmo
    _flat_s, dst, src, sign, starts, n_det = link_sorted
    out = np.zeros((a1 - a0, n_det), dtype=Cvec.dtype)
    lo, hi = int(starts[a0]), int(starts[a1])
    if hi > lo:
        rows_local = np.repeat(np.arange(a0, a1, dtype=np.int64), np.diff(starts[a0 : a1 + 1])) - a0
        np.add.at(out, (rows_local, dst[lo:hi]), sign[lo:hi] * Cvec[src[lo:hi]])
    return out


def _spinless_rdm12_chunked(
    C_flat: np.ndarray, nmo: int, n: int, chunk: int, *, row_map=None, strs=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Chunked ``(N, 0)`` spinless 1- and 2-RDM without materializing full ``B``.

    Reconstructs the same quantities as :func:`sci_solvers.complex_rdm.make_rdm12_complex`
    for the ``(N, 0)`` sector (where ``Bb == 0``), but builds the alpha
    single-excitation tensor ``B = E_pq|C>`` in blocks of ``chunk`` flattened
    ``(p, q)`` operator rows. The 2-RDM Gram term
    ``T[p,q,r,s] = sum_ij conj(B[q,p,ij]) B[r,s,ij]`` is assembled as the
    block-by-block Gram matrix ``G[a, b] = sum_ij conj(B_a) B_b`` (``a, b`` flattened
    operator indices), then reindexed ``T = G`` reshaped and ``(p, q)`` transposed.
    Peak ``B``-residency is ``2 * chunk * Na`` complex128; the unavoidable ``(nmo^4)``
    result ``T`` is the only large allocation left.
    """
    Cvec = np.ascontiguousarray(C_flat).reshape(-1).astype(np.complex128)
    if row_map is not None:
        # Re-lay C onto the connected-set axis the link table now indexes, so the
        # B-block (indexed by connected dst) and the dm1 contraction share the axis.
        Cvec = _embed_connected(Cvec, nmo, n, row_map, strs=strs)
    Cc = Cvec.conj()
    P = nmo * nmo
    link_sorted = _alpha_link_sorted(nmo, n, row_map=row_map, strs=strs)

    dm1_flat = np.empty(P, dtype=np.complex128)
    G = np.empty((P, P), dtype=np.complex128)

    for i0 in range(0, P, chunk):
        i1 = min(i0 + chunk, P)
        Bi = _build_Ba_block(Cvec, link_sorted, nmo, i0, i1)  # (bi, Na)
        # dm1[p, q] = <C| E_pq |C> = sum_i conj(C_i) B[pq, i].
        dm1_flat[i0:i1] = Bi @ Cc
        for j0 in range(0, P, chunk):
            j1 = min(j0 + chunk, P)
            Bj = Bi if j0 == i0 else _build_Ba_block(Cvec, link_sorted, nmo, j0, j1)
            G[i0:i1, j0:j1] = Bi.conj() @ Bj.T

    dm1 = dm1_flat.reshape(nmo, nmo)
    # T[p,q,r,s] = sum_ij conj(B[q,p,ij]) B[r,s,ij] = G[(q,p), (r,s)]; G is indexed
    # by flattened (row, col), so reshape to (row, col, row, col) then swap the first
    # two axes to put (p, q) in front.
    T = np.transpose(G.reshape(nmo, nmo, nmo, nmo), (1, 0, 2, 3))
    # dm2[p,q,r,s] = <E_pq E_rs> - delta_{qr} <E_ps>  (selected_ci.make_rdm2 order).
    dm2 = T
    for qq in range(nmo):
        dm2[:, qq, qq, :] -= dm1
    return dm1, dm2


def _resolve_rdm_chunk(chunk: Union[int, None], dim: int, nmo: int) -> Union[int, None]:
    """Resolve the ``chunk`` policy: ``None`` -> auto (chunk large problems only).

    Returns ``None`` to signal "no chunking, use the direct complex_rdm reuse" (the
    exact small-size path), or a positive block size in flattened ``(p, q)`` units.
    """
    if chunk is not None:
        return int(chunk) if int(chunk) > 0 else None
    if dim >= _RDM_CHUNK_DIM_THRESHOLD:
        # Aim for a ~0.5 GB B-block: 0.5e9 / (16 * dim) rows, clamped to [1, nmo**2].
        rows = max(1, int(0.5e9 / (16.0 * max(1, dim))))
        return min(nmo * nmo, rows)
    return None


def spinless_rdm12(
    amplitudes: np.ndarray,
    nmo: int,
    n_particles: int,
    *,
    strs=None,
    chunk: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Complex-correct spin-orbital 1- and 2-RDM for a spinless CI vector.

    Reuses :func:`sci_solvers.complex_rdm.make_rdm12_complex` via the ``(N, 0)``
    sector embedding: the flat amplitudes are reshaped to ``(Nsel, 1)`` and passed
    with ``nelec=(N, 0)`` (the beta link table is empty, so the beta contribution
    vanishes exactly). Validated to reproduce a brute-force spin-orbital oracle to
    ~1e-14.

    For large ``Nsel`` the single-excitation tensor ``B = E_pq|C>``
    (``(nmo, nmo, Nsel)`` complex128) is the dominant allocation, so the RDMs are
    then reconstructed **in blocks** of ``(p, q)`` operator rows via
    :func:`_spinless_rdm12_chunked` -- identical results, peak ``B``-residency
    ``2 * chunk * Nsel`` instead of ``nmo**2 * Nsel``. The chunked path is a
    self-contained alpha-only reconstruction (``Bb == 0`` on the ``(N, 0)`` sector),
    so ``complex_rdm`` is left untouched.

    Selected-CI support: pass ``strs`` (the selected determinant list parallel to
    ``amplitudes``) to reconstruct the RDMs over an arbitrary **subset** of the
    canonical ``(nmo, N)`` determinants. A ``row_map`` derived from ``strs`` keeps
    only links with a selected source and re-indexes the intermediate determinant
    axis onto the *connected set* (the selected determinants plus their
    single-excitation reachable destinations), threaded into both the direct and
    chunked paths. The connected set is required for the two-body Gram term, which
    visits intermediates outside the selected space; it is bounded by
    ``len(strs) * n_links`` rather than the full canonical dimension. When ``strs``
    is ``None`` or the full canonical list, the zero-overhead full-canonical path
    runs unchanged (byte-identical).

    Args:
        amplitudes: Flat complex CI amplitudes, length ``Nsel``.
        nmo: Number of spin-orbitals.
        n_particles: Total particle number ``N``.
        strs: Optional selected determinant bitmasks parallel to ``amplitudes``.
            ``None`` (default) assumes the full canonical
            ``cistring.make_strings(range(nmo), N)`` ordering.
        chunk: Block size (in flattened ``(p, q)`` operator rows) for the chunked
            reconstruction. ``None`` (default) auto-selects: the direct
            ``complex_rdm`` reuse for small problems, a block build for
            ``Nsel >= 50_000``. A positive value forces that block size; a
            non-positive value forces the direct (unchunked) path.

    Returns:
        ``(dm1, dm2)`` spin-orbital reduced density matrices (``dm1`` Hermitian,
        ``dm2`` in the ``pyscf.fci.selected_ci.make_rdm2`` convention).
    """
    C = np.ascontiguousarray(amplitudes).reshape(-1, 1).astype(np.complex128)
    dim = C.shape[0]
    row_map = _row_map_for(strs, int(nmo), int(n_particles))
    strs_sorted = _sorted_strs_for(strs, row_map)
    resolved = _resolve_rdm_chunk(chunk, dim, int(nmo))
    if resolved is not None:
        return _spinless_rdm12_chunked(
            C, int(nmo), int(n_particles), resolved, row_map=row_map, strs=strs_sorted
        )

    from sci_solvers.complex_rdm import make_rdm12_complex

    dm1, dm2, _ = make_rdm12_complex(
        C, int(nmo), (int(n_particles), 0), row_map_a=row_map, strs_a=strs_sorted
    )
    return dm1, dm2


def spinless_rdm1(
    amplitudes: np.ndarray,
    nmo: int,
    n_particles: int,
    *,
    strs=None,
    chunk: Union[int, None] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Complex-correct spinless 1-RDM via the ``(N, 0)`` embedding.

    See :func:`spinless_rdm12` for the embedding, ordering, and selected-CI
    (``strs``) contract. Returns ``(dm1, (dma, dmb))``; for the spinless case
    ``dmb`` is zero and ``dma`` carries the full spin-orbital 1-RDM, whose real
    diagonal is the orbital occupancy.

    For large ``Nsel`` the 1-RDM is built in blocks of ``(p, q)`` operator rows
    (``dma[p, q] = sum_i conj(C_i) (E_pq C)_i``) so the full ``B`` tensor is never
    materialized; ``chunk`` follows the same policy as :func:`spinless_rdm12`.
    """
    C = np.ascontiguousarray(amplitudes).reshape(-1, 1).astype(np.complex128)
    dim = C.shape[0]
    row_map = _row_map_for(strs, int(nmo), int(n_particles))
    strs_sorted = _sorted_strs_for(strs, row_map)
    resolved = _resolve_rdm_chunk(chunk, dim, int(nmo))
    if resolved is not None:
        nmo_i, n_i = int(nmo), int(n_particles)
        Cvec = C[:, 0]
        if row_map is not None:
            # Re-lay C onto the connected-set axis the (selected) link table indexes.
            Cvec = _embed_connected(Cvec, nmo_i, n_i, row_map, strs=strs_sorted)
        Cc = Cvec.conj()
        P = nmo_i * nmo_i
        link_sorted = _alpha_link_sorted(nmo_i, n_i, row_map=row_map, strs=strs_sorted)
        dma_flat = np.empty(P, dtype=np.complex128)
        for i0 in range(0, P, resolved):
            i1 = min(i0 + resolved, P)
            Bi = _build_Ba_block(Cvec, link_sorted, nmo_i, i0, i1)
            dma_flat[i0:i1] = Bi @ Cc
        dma = dma_flat.reshape(nmo_i, nmo_i)
        dmb = np.zeros_like(dma)
        return dma + dmb, (dma, dmb)

    from sci_solvers.complex_rdm import make_rdm1_complex

    return make_rdm1_complex(
        C, int(nmo), (int(n_particles), 0), row_map_a=row_map, strs_a=strs_sorted
    )


def spinless_energy(
    amplitudes: np.ndarray, hcore: np.ndarray, eri: np.ndarray, n_particles: int
) -> float:
    """Electronic energy of a spinless CI vector, rebuilt from its RDMs.

    Uses ``<dm1, hcore> + 0.5 * einsum("prqs,prqs->", dm2, eri)`` (the same
    contraction the solver uses); the imaginary part (~machine-zero for a Hermitian
    operator) is dropped.
    """
    nmo = int(hcore.shape[0])
    dm1, dm2 = spinless_rdm12(amplitudes, nmo, n_particles)
    e = np.einsum("pr,pr->", dm1, hcore) + 0.5 * np.einsum("prqs,prqs->", dm2, eri)
    return float(np.real(e))
