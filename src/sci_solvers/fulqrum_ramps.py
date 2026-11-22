"""RAMPS subspace expansion with fulqrum, for spatial and spinless determinants.

fulqrum's RAMPS (``fulqrum.ramps.ramps_open``) grows a determinant subspace around a
core: starting from every core determinant it enumerates the determinants the
Hamiltonian connects to it, and adds each one whose second-order perturbative energy
estimate ``prefactor * |H_ij|^2 / (E_target - H_jj)`` exceeds ``tol``. With
``max_recursion > 1`` the added determinants seed the next layer. The result is a
compact, Hamiltonian-selected determinant set (a general-determinant basis), typically
much smaller than a Cartesian-product or all-excitations expansion of similar accuracy.

The score uses only ``|H_ij|`` and the real ``E_target - H_jj``, so RAMPS applies
unchanged to complex Hermitian operators. Two wrappers follow the sci-solvers string
conventions:

* :func:`ramps_expand`: spatial determinants as parallel ``(alpha, beta)`` arrays, for
  ``mode="gdb"`` solvers; real or complex Hermitian integrals.
* :func:`ramps_expand_spinless`: flat ``N``-particle spin-orbital determinants, for the
  spinless (scalar ``nelec``) path; complex Hermitian spin-orbital integrals.

Both reuse the cached fulqrum operators of :mod:`sci_solvers.fulqrum_ci_problem`. Needs
the optional ``fulqrum`` extra.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from typing import Any

import numpy as np

from sci_solvers.complex_ci_problem import _check_hermitian_integrals
from sci_solvers.fulqrum_ci_problem import (
    fulqrum_spinless_subspace_hamiltonian,
    fulqrum_subspace_hamiltonian,
)


def _validate(
    hcore: np.ndarray,
    eri: np.ndarray,
    tol: float,
    max_recursion: int,
    check_hermitian: bool,
    atol: float,
    rtol: float,
) -> int:
    """Check the shared inputs; return the number of (spin-)orbitals."""
    n = int(hcore.shape[0])
    if hcore.shape != (n, n) or eri.shape != (n,) * 4:
        raise ValueError(f"hcore must be (n, n) and eri (n,)*4; got {hcore.shape} and {eri.shape}.")
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}.")
    if max_recursion < 1:
        raise ValueError(f"max_recursion must be >= 1, got {max_recursion!r}.")
    if check_hermitian and (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        _check_hermitian_integrals(hcore, eri, atol=atol, rtol=rtol)
    return n


def _check_strings(strs: list[int], width: int, name: str) -> None:
    """Every string fits in ``width`` bits and all have the same number of set bits."""
    if any(not 0 <= d < 1 << width for d in strs):
        raise ValueError(f"every string in {name} must be in [0, 2**{width}).")
    if len({d.bit_count() for d in strs}) != 1:
        raise ValueError(f"every string in {name} must have the same particle number.")


def _ramps(
    hsub: Any, core_bits: list[str], target_energy: float, tol: float, max_recursion: int
) -> list[str]:
    """Run ``ramps_open`` on ``core_bits``; return every bitstring of the grown subspace."""
    import fulqrum as fq
    from fulqrum.ramps import ramps_open

    # fulqrum 0.4.5's Subspace.copy() (called by ramps_open) segfaults on exactly two
    # bitstrings. A two-row core is padded with the vacuum, which the
    # particle-conserving Hamiltonian couples to nothing, and the vacuum is dropped.
    vacuum = "0" * len(core_bits[0])
    padded = len(core_bits) == 2 and vacuum not in core_bits
    grown = ramps_open(
        hsub,
        fq.Subspace([core_bits + [vacuum] if padded else core_bits]),
        float(target_energy),
        int(max_recursion),
        float(tol),
    )
    bits = [grown.get_n_th_bitstring(k) for k in range(len(grown))]
    return [b for b in bits if b != vacuum] if padded else bits


def ramps_expand(
    hcore: np.ndarray,
    eri: np.ndarray,
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    target_energy: float,
    *,
    tol: float = 1e-6,
    max_recursion: int = 1,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow a spatial determinant subspace around a core with fulqrum's RAMPS.

    Args:
        hcore: One-electron integrals, ``(norb, norb)``, real or complex Hermitian.
        eri: Chemist-notation two-electron integrals, ``(norb,)*4``.
        strs_a: Alpha CI strings (bitmask integers) of the core determinants, parallel
            to ``strs_b``: core determinant ``i`` is ``(strs_a[i], strs_b[i])``.
            Repeated pairs are collapsed.
        strs_b: Beta CI strings of the core determinants, parallel to ``strs_a``.
        target_energy: Electronic energy (no nuclear repulsion) of the state the
            expansion targets, typically the core's ground-state energy. It sets the
            perturbative denominators.
        tol: Selection threshold on the perturbative energy estimate; smaller keeps
            more determinants.
        max_recursion: Number of selection layers, each seeded by the previous layer's
            additions.
        check_hermitian: For complex integrals, verify the Hermiticity symmetries the
            CI builders check.
        atol: Absolute tolerance for those checks.
        rtol: Relative tolerance for those checks.

    Returns:
        ``(strs_a, strs_b)``: the expanded determinant set as parallel int64 arrays,
        every core determinant included (first, in their input order, deduplicated),
        followed by the additions in the order RAMPS selected them. Particle numbers are
        preserved, since the Hamiltonian conserves them.

    Raises:
        ValueError: If the integral shapes are inconsistent, the string arrays are not
            parallel or empty, ``tol`` is not positive, ``max_recursion < 1``, or
            (``check_hermitian``) complex integrals are not Hermitian.

    Notes:
        For a core determinant (or a determinant added in an earlier layer) ``i`` with
        prefactor ``p_i``, each connected determinant ``j`` outside the subspace scores

        .. math::

            \\epsilon_j = p_i \\, \\frac{|H_{ij}|^2}{E_{\\text{target}} - H_{jj}},

        with ``p_i = 1 / E_target`` for the core, and ``j`` is added when
        ``|epsilon_j| > tol``. An added ``j`` seeds the next layer with
        ``p_j = epsilon_j / (E_target - H_jj)``. ``H_jj`` and ``E_target`` are real for
        a Hermitian ``H``, so the score is the same for real and complex integrals.
    """
    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    norb = _validate(hcore, eri, tol, max_recursion, check_hermitian, atol, rtol)
    strs_a = np.asarray(strs_a, dtype=np.int64).reshape(-1)
    strs_b = np.asarray(strs_b, dtype=np.int64).reshape(-1)
    if strs_a.shape != strs_b.shape or strs_a.size == 0:
        raise ValueError(
            "strs_a and strs_b must be non-empty parallel arrays of equal length; got "
            f"{strs_a.shape} and {strs_b.shape}."
        )

    _check_strings(strs_a.tolist(), norb, "strs_a")
    _check_strings(strs_b.tolist(), norb, "strs_b")

    # Core pairs, deduplicated, in input order.
    core: dict[tuple[int, int], None] = {}
    for a, b in zip(strs_a.tolist(), strs_b.tolist()):
        core.setdefault((a, b), None)

    # fulqrum full string: beta half then alpha half, rightmost character is mode 0.
    core_bits = [format(b, f"0{norb}b") + format(a, f"0{norb}b") for a, b in core]
    grown = _ramps(
        fulqrum_subspace_hamiltonian(hcore, eri), core_bits, target_energy, tol, max_recursion
    )

    # Core first (input order), then the additions in the order RAMPS inserted them.
    out = dict(core)
    mask = (1 << norb) - 1
    for bits in grown:
        full = int(bits, 2)
        out.setdefault((full & mask, full >> norb), None)
    pairs = np.array(list(out), dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0].copy(), pairs[:, 1].copy()


def ramps_expand_spinless(
    hcore: np.ndarray,
    eri: np.ndarray,
    strs: np.ndarray,
    target_energy: float,
    *,
    tol: float = 1e-6,
    max_recursion: int = 1,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> np.ndarray:
    """Grow a spinless (flat spin-orbital) determinant subspace with fulqrum's RAMPS.

    The spinless counterpart of :func:`ramps_expand`, on the operator
    :func:`~sci_solvers.fulqrum_ci_problem.fulqrum_spinless_subspace_hamiltonian` builds.

    Args:
        hcore: Complex Hermitian spin-orbital one-electron integrals, ``(nmo, nmo)``.
        eri: Complex chemist-notation spin-orbital ERI, ``(nmo,)*4``.
        strs: ``N``-particle determinant bitmasks over ``nmo`` spin-orbitals (the core).
            Repeated determinants are collapsed.
        target_energy: Electronic energy of the targeted state, typically the core's
            ground energy.
        tol: Selection threshold on the perturbative energy estimate.
        max_recursion: Number of selection layers.
        check_hermitian: Verify the Hermiticity symmetries before building.
        atol: Absolute tolerance for those checks.
        rtol: Relative tolerance for those checks.

    Returns:
        The expanded determinants as an int64 array: the core first (deduplicated, in
        input order), then the additions. The particle number is preserved.

    Raises:
        ValueError: If the integrals are real or inconsistently shaped, ``strs`` is empty
            or mixes particle numbers, ``tol`` is not positive, ``max_recursion < 1``, or
            (``check_hermitian``) the integrals are not Hermitian.
    """
    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        raise ValueError(
            "the spinless path is for complex Hermitian integrals; got real integrals."
        )
    nmo = _validate(hcore, eri, tol, max_recursion, check_hermitian, atol, rtol)
    core = list(dict.fromkeys(int(x) for x in np.asarray(strs).reshape(-1).tolist()))
    if not core:
        raise ValueError("strs must contain at least one determinant.")
    _check_strings(core, nmo, "strs")

    grown = _ramps(
        fulqrum_spinless_subspace_hamiltonian(hcore, eri),
        [format(d, f"0{nmo}b") for d in core],
        target_energy,
        tol,
        max_recursion,
    )
    out = dict.fromkeys(core)
    for bits in grown:
        out.setdefault(int(bits, 2), None)
    return np.array(list(out), dtype=np.int64)
