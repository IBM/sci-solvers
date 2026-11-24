"""Adapter exposing this package's torch matvec + Davidson + RDM machinery
as a ``sci_solver`` for ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``.

Why use this instead of the in-house ``sqd_loop``?

The qiskit_addon_sqd loop provides several production features that the
package's standalone ``sqd_loop`` doesn't (and shouldn't reinvent):

- ``max_dim``        — basis-size cap per spin sector. The right primary
                       knob for fitting large bases on bounded device RAM.
                       (Far more principled than tuning ``--num-block``.)
- ``num_batches``    — ensemble averaging across subsampled batches; tends
                       to improve SQD convergence.
- ``include_configurations``
                     — pin anchor configs (HF, important determinants).
- ``carryover_threshold``
                     — keep high-weight CI strings across iterations.
- ``callback``       — per-iteration introspection hook.
- Standard ``BitArray`` input — what Qiskit primitives produce.

The torch path (matvec, Davidson, 2-RDM, energy reconstruction) stays
identical; we just plug into qiskit's outer loop instead of running our
own.

Typical use::

    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian
    from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

    solver = make_torch_sci_solver(device='cuda', chunk_size=2_000_000)

    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=h1_spatial,
        two_body_tensor=eri_spatial,
        bit_array=bit_array,
        samples_per_batch=samples_per_batch,
        norb=L, nelec=(n_a, n_b),
        max_iterations=10,
        sci_solver=solver,
        max_dim=10000,           # per-spin-sector basis cap
        num_batches=1,           # raise for ensemble averaging
    )
    print(result.energy)         # converged SQD energy
    occ_a, occ_b = result.orbital_occupancies
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import atexit
import contextlib
import dataclasses
import inspect
import multiprocessing
import os
import pathlib
import queue
import threading
import time as _time
import warnings
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Tuple, cast

import numpy as np
import torch
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch
from torch_sci_solver.torch_impl.determinants import _signed_int64
from torch_sci_solver.torch_impl.helpers import _n_limbs, generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import (
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.matvec_dispatch import MATVEC_IMPLS, build_matvec
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
from torch_sci_solver.torch_impl.probe import (
    TorchSCIState,
    _dedup_and_check_superset,
    _embed_eigvec_into_extended_basis,
    _emit_dedup_warning,
    _integrals_fingerprint,
    _validate_fingerprints,
)
from torch_sci_solver.torch_impl.rdm import (
    make_one_rdm_diag_torch,
    make_one_rdm_torch,
    make_two_rdm_torch,
)
from torch_sci_solver.torch_impl.sbdiag import _to_int64_python

if TYPE_CHECKING:
    from torch_sci_solver.torch_impl.integrals import TorchIntegrals
    from torch_sci_solver.torch_impl.probe_spinless import TorchSpinlessSCIState


def _as_integrals_array(x: np.ndarray) -> np.ndarray:
    """Coerce an integrals array (``hcore``/``eri``) to a numeric numpy array,
    preserving a complex dtype rather than forcing ``float64``.

    Complex input (``complex64``/``complex128``) is cast to ``complex128``
    (the only complex precision the engine supports); any other dtype
    (real float, or e.g. a plain Python list/int array) is cast to
    ``float64`` exactly as before. This keeps the real path byte-identical
    while letting complex Hermitian integrals flow through without their
    imaginary part being silently dropped.
    """
    arr = np.asarray(x)
    return (
        arr.astype(np.complex128, copy=False)
        if np.iscomplexobj(arr)
        else arr.astype(np.float64, copy=False)
    )


def _spinless_output_state(
    sci_result: Any,
    *,
    n_orb: int,
    n_particles: int,
    integrals_fingerprint: str,
    solver_knobs: dict,
    n_matvecs: "Optional[int]" = None,
    final_residual: "Optional[float]" = None,
    eigvec_override: Any = None,
) -> "TorchSpinlessSCIState":
    """Build a :class:`TorchSpinlessSCIState` from one converged ground root.

    The spinless path already returns its eigenvector in the internal
    canonical order (``sci_result.sci_state.ci_strs`` parallel to
    ``sci_result.sci_state.amplitudes``), so the cached state is a direct
    snapshot: no sign bridge or basis re-indexing is needed.

    ``amplitudes`` may be a :class:`MemmapAmplitudes` marker when a large
    distributed solve hands its result back through files. The large
    owner-computes path writes the warm vector under its own file name and
    passes that marker as ``eigvec_override``, so materializing the result
    amplitudes cannot unlink the snapshot's file first. Without an override the
    result marker is kept as-is and mapped by the same materializer.
    """
    from torch_sci_solver.torch_impl.probe_spinless import TorchSpinlessSCIState

    sci_state = sci_result.sci_state
    amplitudes = eigvec_override if eigvec_override is not None else sci_state.amplitudes
    if isinstance(amplitudes, torch.Tensor):
        eigvec = amplitudes.detach().to("cpu").clone()
    elif isinstance(amplitudes, np.ndarray):
        eigvec = torch.from_numpy(np.array(amplitudes, copy=True))
    else:
        eigvec = amplitudes
    return TorchSpinlessSCIState(
        dets=[int(det) for det in np.asarray(sci_state.ci_strs, dtype=object).ravel()],
        eigvec_inter=eigvec,
        energy=float(sci_result.energy),
        n_orb=int(n_orb),
        n_particles=int(n_particles),
        integrals_fingerprint=str(integrals_fingerprint),
        solver_knobs=dict(solver_knobs),
        n_matvecs=n_matvecs,
        final_residual=final_residual,
    )


def _validate_ci_basis(
    adets: np.ndarray,
    bdets: np.ndarray,
    norb: int,
    nelec: "Tuple[int, int]",
    *,
    sub_idx: "Optional[int]" = None,
) -> None:
    """Sanity-check that ``(adets, bdets)`` is a well-formed CI basis.

    A valid CI basis for an (n_α, n_β) electron system in ``norb``
    spatial orbitals satisfies, for every string ``s`` in adets or bdets:

      1. ``s.bit_length() <= norb`` — no bits set above orbital ``norb - 1``.
      2. ``popcount(s) == n_α`` (for adets) or ``popcount(s) == n_β`` (for bdets).

    Without (2), the downstream excitation-table builder (``generate_excitation``
    in sbd_ref/helpers.py) can hit ``IndexError: list index out of range``
    because ``orbital_difference(a, b, norb)`` returns lists of length
    ``popcount(a XOR b) / 2`` PER SIDE only when the popcounts of ``a`` and
    ``b`` match; otherwise the per-side lengths can be 0/1/3/4 (etc.) and
    code that assumes ``cr[1]`` exists for a "double excitation" crashes.

    Without (1), bits set above ``norb`` either get masked off silently
    (producing wrong physics with no error) or, in some downstream paths,
    cause similar inscrutable IndexErrors.

    This validator turns either failure into a clear ``ValueError`` that
    names the offending strings.

    Args:
        adets, bdets: int64 arrays of CI bitmasks, or ``(M, n_limbs)`` int64
            limb matrices (limb ``k`` holds bits ``[64k, 64k + 64)``). Need not
            be sorted/deduped (the validator is order-independent).
        norb: number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        sub_idx: optional subspace index to include in the error message
            (helps when the solver is called with a list of subspaces).

    Raises:
        ValueError: with a descriptive message that identifies which sector
            (α or β) is bad and lists a representative subset of the
            offending strings.
    """
    n_a, n_b = nelec
    a_arr = np.asarray(adets)
    b_arr = np.asarray(bdets)

    # Compose a context prefix that names the subspace if known. Keeps the
    # error messages locatable when a multi-subspace ci_strings input has
    # exactly one bad subspace.
    ctx = f"subspace {sub_idx}: " if sub_idx is not None else ""

    if a_arr.ndim == 2 or b_arr.ndim == 2:
        _validate_ci_basis_limbs(a_arr, b_arr, norb, nelec, ctx)
        return

    a_arr = _signed_int64_keys(adets)
    b_arr = _signed_int64_keys(bdets)

    # (1) bit-position check: every set bit must be in [0, norb). Checked on the
    # unsigned view: a string with bit 63 set is a negative int64 pattern.
    if a_arr.size > 0 and norb < 64:
        bad_mask = (a_arr.view(np.uint64) >> np.uint64(norb)) > 0
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            sample = a_arr.view(np.uint64)[bad_mask][:5].tolist()
            raise ValueError(
                f"{ctx}invalid α-strings: {n_bad}/{a_arr.size} have bits set "
                f"above orbital {norb - 1} (this CI basis uses norb={norb} "
                f"so valid bit positions are 0..{norb - 1}). Sample bad "
                f"α-strings (first {len(sample)}): {sample} "
                f"(max bit positions: "
                f"{[int(x).bit_length() - 1 for x in sample]}). "
                f"Check that your bitstring encoding matches the (norb, nelec) "
                f"you're passing to the solver."
            )
    if b_arr.size > 0 and norb < 64:
        bad_mask = (b_arr.view(np.uint64) >> np.uint64(norb)) > 0
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            sample = b_arr.view(np.uint64)[bad_mask][:5].tolist()
            raise ValueError(
                f"{ctx}invalid β-strings: {n_bad}/{b_arr.size} have bits set "
                f"above orbital {norb - 1} (this CI basis uses norb={norb}). "
                f"Sample bad β-strings (first {len(sample)}): {sample}. "
                f"Check that your bitstring encoding matches (norb, nelec)."
            )

    # (2) popcount check: every string must have the right particle count.
    if a_arr.size > 0:
        pop_a = _popcount64(a_arr)
        bad_mask = pop_a != n_a
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            bad_strings = a_arr[bad_mask]
            bad_popcounts = pop_a[bad_mask]
            # Show up to 5 examples with their actual popcount.
            sample = [(int(s), int(p)) for s, p in zip(bad_strings[:5], bad_popcounts[:5])]
            unique_bad = sorted(set(int(p) for p in bad_popcounts.tolist()))
            raise ValueError(
                f"{ctx}invalid α-strings: {n_bad}/{a_arr.size} have popcount "
                f"!= n_alpha (={n_a}). Found α-strings with popcounts "
                f"{unique_bad} mixed in with the expected popcount-{n_a} "
                f"strings — a CI basis must keep all α-strings in the same "
                f"particle-number sector. Sample bad α-strings "
                f"(string, popcount): {sample}. "
                f"This usually means the bitstring encoding is wrong (off-by-"
                f"one in nelec, or sampling without enforcing the particle "
                f"count), or that adets and bdets got swapped."
            )
    if b_arr.size > 0:
        pop_b = _popcount64(b_arr)
        bad_mask = pop_b != n_b
        if bad_mask.any():
            n_bad = int(bad_mask.sum())
            bad_strings = b_arr[bad_mask]
            bad_popcounts = pop_b[bad_mask]
            sample = [(int(s), int(p)) for s, p in zip(bad_strings[:5], bad_popcounts[:5])]
            unique_bad = sorted(set(int(p) for p in bad_popcounts.tolist()))
            raise ValueError(
                f"{ctx}invalid β-strings: {n_bad}/{b_arr.size} have popcount "
                f"!= n_beta (={n_b}). Found β-strings with popcounts "
                f"{unique_bad} mixed in with the expected popcount-{n_b} "
                f"strings — a CI basis must keep all β-strings in the same "
                f"particle-number sector. Sample bad β-strings "
                f"(string, popcount): {sample}. "
                f"This usually means the bitstring encoding is wrong (off-by-"
                f"one in nelec, or sampling without enforcing the particle "
                f"count)."
            )


def _popcount64(arr: np.ndarray) -> np.ndarray:
    """Numpy-vectorised popcount for int64 arrays.

    Standard bit-twiddling SWAR (SIMD-Within-A-Register). Works for non-
    negative int64 values, which is the FCI-string encoding produced by
    qiskit-addon-sqd's ``bitstring_matrix_to_ci_strs``. Ported verbatim
    from SBD's wrapper so behaviour matches bit-for-bit.
    """
    x = arr.astype(np.int64, copy=True)
    m1 = np.int64(0x5555555555555555)
    m2 = np.int64(0x3333333333333333)
    m4 = np.int64(0x0F0F0F0F0F0F0F0F)
    h01 = np.int64(0x0101010101010101)
    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    return ((x * h01) >> 56) & np.int64(0x7F)


def _limb_count(norb: int) -> int:
    """Number of int64 limbs a per-spin string needs for ``norb`` orbitals."""
    return _n_limbs(norb)


def _signed_int64_limb(value: int) -> int:
    """One unsigned 64-bit limb as the signed int64 bit pattern torch stores."""
    return _signed_int64(value)


def _ints_to_limb_matrix(values, n_limbs: int) -> np.ndarray:
    """Pack Python-int strings into an ``(M, n_limbs)`` signed-int64 limb matrix.

    Limb ``k`` holds bits ``[64k, 64k + 64)``; a limb whose high bit is set is
    stored as its negative two's-complement value, the convention every torch
    bitwise kernel in this package uses.
    """
    values = [int(v) for v in values]
    out = np.zeros((len(values), n_limbs), dtype=np.int64)
    mask = (1 << 64) - 1
    for k in range(n_limbs):
        out[:, k] = [_signed_int64_limb((v >> (64 * k)) & mask) for v in values]
    return out


def _limb_rows_as_ints(limbs) -> list[int]:
    """Reassemble ``(M, n_limbs)`` signed-int64 limbs into Python ints."""
    out = []
    for row in np.asarray(limbs, dtype=np.int64).tolist():
        value = 0
        for k, limb in enumerate(row):
            value |= (int(limb) & ((1 << 64) - 1)) << (64 * k)
        out.append(value)
    return out


def _ci_string_limbs(strings, norb: int) -> "np.ndarray | None":
    """Normalize one spin's CI strings to an ``(M, n_limbs)`` int64 limb matrix.

    Accepts the forms a caller may pass: a sequence of Python ints of any
    width, a 1-D integer array (its low 64 bits, zero-extended), or an
    already-split ``(M, n_limbs)`` integer matrix. Returns ``None`` when
    ``norb <= 64``: a single word is what the legacy 1-D tensor path already
    carries, and returning it unchanged keeps that path bit-identical.

    Args:
        strings: the per-spin CI strings, in any of the forms above.
        norb: number of spatial orbitals.

    Returns:
        ``(M, n_limbs)`` signed-int64 limb matrix, or ``None`` for a
        single-word basis.

    Raises:
        TypeError: non-integer input.
        ValueError: an unexpected shape, a negative Python int, a string that
            does not fit in ``ceil(norb / 64)`` limbs, or a 2-D matrix whose
            trailing width does not match that limb count.
    """
    n_limbs = _limb_count(norb)
    arr = np.asarray(strings)
    if arr.dtype.kind == "f" and not isinstance(strings, np.ndarray):
        # A list holding a value >= 2**63 infers float64; keep the exact ints
        # (a list of genuine floats still fails the check below).
        exact = np.asarray(strings, dtype=object)
        if all(isinstance(v, (int, np.integer)) for v in exact.ravel().tolist()):
            arr = exact
    if arr.size == 0:
        return None if n_limbs == 1 else np.zeros((0, n_limbs), dtype=np.int64)
    if arr.ndim == 2:
        if arr.dtype.kind not in "iu":
            raise TypeError(f"CI strings must be integers, got {arr.dtype}")
        if arr.shape[1] != n_limbs:
            raise ValueError(
                f"CI-string limb matrix has {arr.shape[1]} limb(s), expected {n_limbs} "
                f"for norb={norb}"
            )
        return None if n_limbs == 1 else np.ascontiguousarray(arr, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"CI strings must be 1-D or 2-D, got ndim={arr.ndim}")
    if arr.dtype.kind == "f":
        raise TypeError("CI strings must be integers, not floating point.")
    if n_limbs == 1:
        return None
    if arr.dtype.kind == "O":
        values = [int(v) for v in arr]
        if any(v < 0 for v in values):
            raise ValueError("CI strings must be non-negative")
        limit = 1 << (64 * n_limbs)
        wide = [v for v in values if v >= limit]
        if wide:
            raise ValueError(
                f"CI string {wide[0]:#x} does not fit in {n_limbs} limb(s) for norb={norb}; "
                f"valid bit positions are 0..{norb - 1}"
            )
        return _ints_to_limb_matrix(values, n_limbs)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"CI strings must be integers, got {arr.dtype}")
    out = np.zeros((arr.shape[0], n_limbs), dtype=np.int64)
    out[:, 0] = arr.view(np.int64) if arr.dtype.itemsize == 8 else arr.astype(np.int64)
    return out


def _validate_ci_basis_limbs(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    norb: int,
    nelec: "Tuple[int, int]",
    ctx: str,
) -> None:
    """Limb-matrix counterpart of :func:`_validate_ci_basis`'s two checks.

    Bits above ``norb`` are located limb by limb (a limb's valid window is
    ``[64k, min(64k + 64, norb))``) and popcounts are summed over limbs.
    """
    n_a, n_b = nelec
    for label, arr, expected in (("α", a_arr, n_a), ("β", b_arr, n_b)):
        if arr.ndim != 2:
            raise ValueError(
                f"{ctx}{label}-strings mix a 1-D bitmask array with a 2-D limb matrix; "
                "pass both spins in the same representation"
            )
        arr = np.asarray(arr, dtype=np.int64)
        n_rows, n_limbs = arr.shape
        if n_rows == 0:
            continue

        bad = np.zeros(n_rows, dtype=bool)
        for k in range(n_limbs):
            bits = int(norb) - 64 * k
            if bits <= 0:
                bad |= arr[:, k] != 0
            elif bits < 64:
                bad |= (arr[:, k] >> np.int64(bits)) != 0
        if bad.any():
            sample = [v for v, is_bad in zip(_limb_rows_as_ints(arr), bad) if is_bad][:5]
            raise ValueError(
                f"{ctx}invalid {label}-strings: {int(bad.sum())}/{n_rows} have bits set "
                f"above orbital {norb - 1} (this CI basis uses norb={norb} so valid bit "
                f"positions are 0..{norb - 1}). Sample bad {label}-strings (first "
                f"{len(sample)}): {sample}. Check that your bitstring encoding matches "
                f"the (norb, nelec) you're passing to the solver."
            )

        pop = np.zeros(n_rows, dtype=np.int64)
        for k in range(n_limbs):
            pop += _popcount64(arr[:, k])
        bad = pop != expected
        if bad.any():
            bad_values = _limb_rows_as_ints(arr)
            sample = [
                (value, int(count))
                for value, count, is_bad in zip(bad_values, pop.tolist(), bad)
                if is_bad
            ][:5]
            unique_bad = sorted(set(int(p) for p in pop[bad].tolist()))
            raise ValueError(
                f"{ctx}invalid {label}-strings: {int(bad.sum())}/{n_rows} have popcount "
                f"!= n_{'alpha' if label == 'α' else 'beta'} (={expected}). Found "
                f"{label}-strings with popcounts {unique_bad} mixed in with the expected "
                f"popcount-{expected} strings — a CI basis must keep all {label}-strings "
                f"in the same particle-number sector. Sample bad {label}-strings "
                f"(string, popcount): {sample}. This usually means the bitstring encoding "
                "is wrong (off-by-one orbital index, or sampling without enforcing the "
                "particle count)."
            )


def _rank_consistent_memory_budget(dist_ctx) -> "Optional[int]":
    """Usable device-memory estimate, agreed across all ranks.

    The per-rank estimate comes from the card's remaining free pool (the
    replicated ERIs and helpers are already allocated when the auto policies
    run), so it models what a new layout can still claim rather than the card's
    nominal size. Auto tiling must make the same decision on every rank: one
    rank running the tile collectives while another keeps the 1-D gather or
    row exchange deadlocks the solve. The per-rank budget can differ on a
    heterogeneous device mix, so the group reduces to the minimum — the same
    conservative choice ``_pick_num_block`` makes — with ``-1`` standing for
    "unknown" (CPU), which forces the no-budget decision everywhere.

    Args:
        dist_ctx: Distributed context of this solve (rank device and group).

    Returns:
        The smallest usable-memory estimate across the ranks, or ``None``
        when any rank cannot estimate one.
    """
    import torch.distributed as dist

    from torch_sci_solver.torch_impl.distributed_davidson import _device_memory_budget

    local = _device_memory_budget(dist_ctx.device)
    known = torch.tensor(
        [-1 if local is None else int(local)],
        dtype=torch.int64,
        device=dist_ctx.device if dist_ctx.device.type == "cuda" else "cpu",
    )
    if dist.is_available() and dist.is_initialized() and dist_ctx.world_size > 1:
        dist.all_reduce(known, op=dist.ReduceOp.MIN, group=dist_ctx.group)
    value = int(known.item())
    return None if value < 0 else value


def _reorder_for_connected_seed(strs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder a CI-string list so the most-connected determinant is first.

    Counts, for each string, how many others sit at Hamming distance 2
    (single excitation) or 4 (double excitation). Picks the string with
    the most such neighbors as the new index 0; the rest follow in their
    original order with the chosen string removed from its old slot.

    Why: SBD-style Davidson seeds its initial vector on basis element 0
    (``W = [1, 0, ..., 0]``). If element 0 is isolated in the basis's
    coupling graph — no other determinant within a single or double
    excitation — the first matvec gives ``H @ W = H_00 * W``, residual
    is zero, and Davidson exits with the seed's diagonal element as the
    "ground state". Picking a heavily-connected seed avoids this.

    Side effect (observed empirically on closed-shell
    problems): the most-connected determinant is the chemistry-dominant
    configuration, which is singlet by construction. Davidson seeded
    there explores a Krylov subspace concentrated on singlet character
    and tends to converge to the lowest singlet eigenstate rather than
    the absolute lowest (which can be a quintet/singlet mixture in the
    truncated basis). This is a convergence pathology that matches SBD's
    behavior on a 36-orbital benchmark.

    Args:
        strs: 1-D array of int64-encoded CI strings, or an ``(M, n_limbs)``
            limb matrix (the Hamming distance then sums over limbs).

    Returns:
        ``(reordered_strs, permutation)`` — ``reordered_strs[i] = strs[permutation[i]]``.
        Caller is responsible for applying the same ``permutation`` to any
        amplitude or determinant array that must stay aligned with ``strs``.
        If no string in the pool has any close neighbor (subspace is
        completely fragmented) OR the most-connected determinant is already
        at index 0, returns ``(strs, identity_permutation)``.
    """
    n = len(strs)
    identity = np.arange(n, dtype=np.int64)
    if n <= 1:
        return strs, identity

    strs_int = np.asarray(strs, dtype=np.int64)
    n_close = np.zeros(n, dtype=np.int64)
    for i in range(n):
        xor = strs_int ^ strs_int[i]
        d = _popcount64(xor)
        if d.ndim == 2:
            d = d.sum(axis=-1)
        n_close[i] = int(np.sum((d == 2) | (d == 4)))

    best_idx = int(np.argmax(n_close))
    if n_close[best_idx] == 0 or best_idx == 0:
        return strs, identity

    perm = np.concatenate([
        np.array([best_idx], dtype=np.int64),
        np.arange(0, best_idx, dtype=np.int64),
        np.arange(best_idx + 1, n, dtype=np.int64),
    ])
    return strs[perm], perm


# Below this fraction of its own norm left (after the mapping and the αβ
# projection), a caller's start vector has no overlap with the basis and the solve
# falls back to the cold start. Relative, so a vector's overall scale is irrelevant.
_INIT_VECTOR_MIN_NORM = 1e-8

# davidson_fp32_energy="auto": at a residual tol this loose the energy is already
# uncertain by more than the fp32 Ritz error (measured on H6 and N2 2.2 Å), so the
# fp64 Rayleigh-quotient matvec is skipped; tighter, it is kept.
_FP32_RITZ_MIN_TOL = 1e-3


def _validate_init_vector(
    vec: Any, expected_len: int, *, dtype: torch.dtype, sub_idx: int, n_roots: int = 1
) -> np.ndarray:
    """Check one ``init_vectors`` entry and return it as a numpy array.

    ``expected_len`` is the caller-side length (``len(strings_a) * len(strings_b)``
    in product mode, the pair or determinant count otherwise). A complex entry
    for a real ``dtype`` is rejected rather than having its imaginary part
    dropped. With ``n_roots > 1`` an entry may also be ``(k, expected_len)``,
    ``1 <= k <= n_roots``: one start vector per root, lowest first.
    """
    arr = np.asarray(vec)
    if arr.ndim == 2 and n_roots > 1:
        if not 1 <= arr.shape[0] <= n_roots or arr.shape[1] != expected_len:
            raise ValueError(
                f"init_vectors[{sub_idx}] has shape {arr.shape}, expected "
                f"(k, {expected_len}) with 1 <= k <= n_roots ({n_roots}): one start "
                "vector per root over the ci_strings passed for this subspace."
            )
    elif arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(
            f"init_vectors[{sub_idx}] has shape {arr.shape}, expected ({expected_len},) — "
            "one amplitude per determinant of the ci_strings passed for this subspace, "
            "in the layout this mode returns in sci_state.amplitudes."
        )
    if np.iscomplexobj(arr) and not dtype.is_complex:
        raise TypeError(
            f"init_vectors[{sub_idx}] is complex but this solver was constructed with "
            f"dtype={dtype} (real)."
        )
    return arr


def _signed_int64_keys(strs: Any) -> np.ndarray:
    """CI strings as the signed-int64 bit patterns the engine stores them as.

    Same mapping as ``sbdiag._to_int64_python`` (a 64-bit string with the top bit set
    becomes negative), but vectorized: int64/uint64 input is reinterpreted in place,
    with no per-element Python loop, which matters at the 10⁸ GDB pairs this keys.
    A string wider than 64 bits raises instead of wrapping to a wrong key (TPB and
    GDB support ``norb <= 64``; the spinless path keys multi-limb rows instead).
    """
    if torch.is_tensor(strs):
        strs = strs.detach().cpu().numpy()
    if isinstance(strs, np.ndarray) and strs.dtype.kind in "iu":
        if strs.dtype.itemsize == 8:
            return strs.view(np.int64).ravel()
        return strs.astype(np.int64).ravel()
    if isinstance(strs, np.ndarray) and strs.dtype.kind == "f":
        raise TypeError("CI strings must be integers, not floating point.")
    # Anything else (a list, an object array) element by element: np.asarray on a
    # list holding a value >= 2**63 infers float64 and rounds the string's bits.
    # A negative value down to -2**63 is already a signed int64 bit pattern (the
    # form int64 arrays and the engine's cached pair states carry).
    out = []
    for value in np.asarray(strs, dtype=object).ravel().tolist():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"CI strings must be integers, got {type(value).__name__}")
        value = int(value)
        if not -(1 << 63) <= value < 1 << 64:
            raise ValueError(
                f"CI string {value:#x} does not fit in 64 bits; TPB and GDB support at "
                "most 64 spatial orbitals."
            )
        out.append(value)
    return np.asarray(_to_int64_python(out), dtype=np.int64)


def _pair_arrays_from_strings(
    strings_a: Any, strings_b: Any, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a caller's pair strings to matching int64 index arrays.

    A basis wider than 64 orbitals carries each spin's strings as an
    ``(M, n_limbs)`` limb matrix, the same representation the TPB path uses; a
    single-word basis keeps the legacy 1-D signed int64 arrays.

    Args:
        strings_a: caller's α strings (1-D words or ``(M, n_limbs)`` limbs).
        strings_b: caller's β strings in the same representation.
        norb: number of spatial orbitals.

    Returns:
        ``(a_arr, b_arr)`` index arrays of equal length.

    Raises:
        ValueError: the two spins use different representations, or their
            lengths differ.
    """
    a_wide = _ci_string_limbs(strings_a, norb)
    b_wide = _ci_string_limbs(strings_b, norb)
    if a_wide is None and b_wide is None:
        a_arr = _signed_int64_keys(strings_a)
        b_arr = _signed_int64_keys(strings_b)
    elif a_wide is not None and b_wide is not None:
        a_arr, b_arr = a_wide, b_wide
    else:
        raise ValueError(
            "pair mode requires both spins in the same string representation: "
            "both 1-D single-word arrays or both (M, n_limbs) limb matrices"
        )
    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(
            f"pair mode requires len(strings_a) == len(strings_b) "
            f"(got {a_arr.shape[0]} and {b_arr.shape[0]}) — each index "
            f"defines one (α_i, β_i) pair."
        )
    return a_arr, b_arr


def _ci_string_words(strs: Any) -> "list[int]":
    """One-word CI strings as unsigned Python ints, the canonical one-word form.

    Accepts uint64 arrays, int64 arrays holding signed bit patterns, and lists or
    object arrays of Python ints up to ``2**64 - 1``. Every form maps to the same
    unsigned values, so a string with orbital 63 occupied sorts after the
    lower-orbital strings, as an integer does.
    """
    return _signed_int64_keys(strs).view(np.uint64).tolist()


def _word_bit_patterns(words: "list[int]") -> np.ndarray:
    """Unsigned one-word strings as the signed int64 bit patterns the engine stores."""
    return np.asarray(_to_int64_python(words), dtype=np.int64)


def _ci_strs_arrays(a_words: "list[int]", b_words: "list[int]") -> "tuple[np.ndarray, np.ndarray]":
    """Both spins' one-word strings for an ``SCIState``, in one dtype.

    int64, or uint64 for both when either spin uses bit 63: a mixed pair would
    promote to float64 (and lose bits) as soon as a caller concatenates them.
    """
    top = any(words and max(words) >= 1 << 63 for words in (a_words, b_words))
    dtype = np.uint64 if top else np.int64
    return np.array(a_words, dtype=dtype), np.array(b_words, dtype=dtype)


def _pair_strs_arrays(alpha: np.ndarray, beta: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Pair strings for a ``PairSCIState``: the unsigned view when bit 63 is used."""
    if alpha.ndim == 1 and ((alpha < 0).any() or (beta < 0).any()):
        return alpha.view(np.uint64), beta.view(np.uint64)
    return alpha, beta


def _first_index(
    caller_keys: "np.ndarray | torch.Tensor",
    internal_keys: "np.ndarray | torch.Tensor",
    *,
    device: torch.device,
) -> torch.Tensor:
    """Position of each internal basis key among the caller's keys, or -1.

    Keys are signed-int64 rows: shape ``(n,)`` for CI strings, ``(n, k)`` for GDB
    ``(α, β)`` pairs or multi-limb spinless determinants. A key the caller passed
    more than once maps to its first occurrence; a key the caller did not pass
    (the HF determinant the engine adds under ``probe_method``) maps to -1.
    Looking the internal keys up covers every reordering the engine applies —
    sort, dedup, the ``seed_strategy="connected"`` reorder — without tracking a
    permutation.

    Runs on ``device`` with integer sorts only. Multi-column keys are folded one
    column at a time: the running row id and the next column's rank (both below
    ``n = n_caller + n_internal``) are packed as ``id * n + rank`` and re-ranked, so
    the id stays below ``n`` and the packing never overflows int64 for any number of
    columns. Row-wise ``np.unique(axis=0)`` is ~10x slower at the 10⁸ pairs a GDB
    basis reaches.
    """
    c = torch.as_tensor(caller_keys, dtype=torch.int64, device=device)
    i = torch.as_tensor(internal_keys, dtype=torch.int64, device=device)
    if c.dim() == 2:
        n_c = c.shape[0]
        both = torch.cat([c, i])
        row_id = torch.unique(both[:, 0], return_inverse=True)[1]
        for j in range(1, both.shape[1]):
            col = torch.unique(both[:, j], return_inverse=True)[1]
            row_id = torch.unique(row_id * len(both) + col, return_inverse=True)[1]
        c, i = row_id[:n_c], row_id[n_c:]
    if c.shape[0] == 0:
        # Nothing to match against (clamp_max(-1) below would index an empty tensor).
        return torch.full((i.shape[0],), -1, dtype=torch.int64, device=device)
    sorted_c, order = torch.sort(c, stable=True)  # stable: duplicates keep caller order
    pos = torch.searchsorted(sorted_c, i).clamp_max(len(sorted_c) - 1)
    return torch.where(sorted_c[pos] == i, order[pos], -1)


def _pair_key_tensor(alpha: Any, beta: Any, *, width: int, device: torch.device) -> torch.Tensor:
    """``(N, 2 * width)`` signed-int64 key rows of an α/β pair list.

    A single-word pair list keeps the legacy ``(N, 2)`` key matrix of signed
    bit patterns; a limb-matrix input (``n_orb > 64``) concatenates each
    spin's limbs, so ``_first_index`` resolves a wide pair basis through the
    same row lookup. Tensor inputs are padded and concatenated on ``device``
    with no host round-trip.

    Args:
        alpha, beta: Parallel α-/β-string sequences (Python ints, 1-D bit
            patterns, ``(N, n_limbs)`` limb matrices, or tensors).
        width: Limb count of the internal pair basis.
        device: Device of the returned key matrix.

    Returns:
        ``(N, 2 * width)`` int64 key rows on ``device``.
    """
    from torch_sci_solver.torch_impl.probe_pairs import (
        _pair_string_rows,
        _pair_string_width,
    )

    resolved = max(int(width), _pair_string_width(alpha), _pair_string_width(beta))
    if resolved != int(width):
        return _pair_key_tensor(alpha, beta, width=resolved, device=device)

    def rows(values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            out = values.to(device=device, dtype=torch.int64)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            if out.shape[1] < width:
                out = torch.cat([out, out.new_zeros((out.shape[0], width - out.shape[1]))], dim=1)
            return out
        return torch.from_numpy(_pair_string_rows(values, width)).to(device)

    return torch.cat([rows(alpha), rows(beta)], dim=1)


def _finish_init_vector(
    W: torch.Tensor,
    project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ref_norm: float,
    norm_fn: Optional[Callable[[torch.Tensor], float]] = None,
) -> Optional[torch.Tensor]:
    """Project and normalize a mapped start vector, or return ``None`` to fall back.

    The norm is checked after ``project_fn`` because Davidson applies it to the
    start vector first: a vector with no component in the projected subspace
    would otherwise become 0 and stop Davidson at E = 0 as "converged". It is
    compared to ``ref_norm``, the norm of the vector the caller passed, so a
    correctly shaped vector with a small overall scale still warm-starts.

    Args:
        W: the mapped start vector (a full vector or one rank's row block).
        project_fn: optional projector applied before the norm check.
        ref_norm: norm of the caller's vector, the scale the surviving norm is
            measured against.
        norm_fn: optional norm implementation; the distributed local path
            passes the global norm so every rank takes the same branch.

    Returns:
        The normalized vector, or ``None`` when too little survives.
    """
    if project_fn is not None:
        W = project_fn(W)
    norm = float(torch.linalg.norm(W).item()) if norm_fn is None else float(norm_fn(W))
    if norm < _INIT_VECTOR_MIN_NORM * ref_norm or norm == 0.0:
        return None
    return W / norm


def _gather_or_zero(vec: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """``vec[k]`` where ``k >= 0``, 0 where ``k == -1`` (a string the caller did not pass)."""
    zero = torch.zeros((), dtype=vec.dtype, device=vec.device)
    return torch.where(k >= 0, vec[k.clamp_min(0)], zero)


def _resolve_start_vector(
    W: torch.Tensor,
    project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    cold_fn: Callable[[], torch.Tensor],
    caller_vec: np.ndarray,
) -> Tuple[torch.Tensor, bool]:
    """``(start_vector, warm)``: the mapped vector, or the cold start if nothing is left.

    The one place a caller's mapped ``init_vectors`` entry is finalized, so the
    fallback rule is the same in every mode (see :func:`_finish_init_vector`).
    ``caller_vec`` is the entry as the caller passed it, whose norm the surviving
    norm is measured against.
    """
    W_warm = _finish_init_vector(W, project_fn, float(np.linalg.norm(caller_vec)))
    if W_warm is None:
        return cold_fn(), False
    return W_warm, True


def _resolve_start_vectors(
    caller_vec: np.ndarray,
    map_fn: Callable[[np.ndarray], torch.Tensor],
    project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    cold_fn: Callable[[], torch.Tensor],
    norm_fn: Optional[Callable[[torch.Tensor], float]] = None,
) -> Tuple[torch.Tensor, bool]:
    """:func:`_resolve_start_vector` for a 1-D entry or a ``(k, len)`` one per root.

    ``map_fn`` maps one caller-side row onto the internal basis. Each row is mapped
    and finished separately (projected, normalized, dropped if nothing is left);
    the surviving rows are returned ``(k', N)``, or as one ``(N,)`` vector when a
    single row survives, which is exactly the 1-D path. No row left: cold start.
    ``norm_fn`` is forwarded to :func:`_finish_init_vector`.
    """
    if caller_vec.ndim == 1:
        W = _finish_init_vector(
            map_fn(caller_vec), project_fn, float(np.linalg.norm(caller_vec)), norm_fn
        )
        return (cold_fn(), False) if W is None else (W, True)
    rows = []
    for row in caller_vec:
        w = _finish_init_vector(map_fn(row), project_fn, float(np.linalg.norm(row)), norm_fn)
        if w is not None:
            rows.append(w)
    if not rows:
        return cold_fn(), False
    return (rows[0] if len(rows) == 1 else torch.stack(rows)), True


def _global_norm_local(W: torch.Tensor, context: Any) -> torch.Tensor:
    """Global Euclidean norm of a row-block vector, identical on every rank."""
    import torch.distributed as dist

    partial = torch.linalg.norm(W.reshape(-1)).reshape(1).to(torch.float64)
    if context.world_size > 1:
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=context.group)
    return torch.sqrt(partial[0].clamp_min(0.0))


def _finish_init_vector_local(
    W: torch.Tensor,
    project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ref_norm: float,
    context: Any,
) -> Optional[torch.Tensor]:
    """Distributed :func:`_finish_init_vector`: project, global-normalize, or ``None``.

    The caller's start vector is mapped to this rank's rows only, so the
    surviving norm must be reduced across ranks before the cold-start fallback
    decision — every rank then takes the same branch.
    """
    return _finish_init_vector(
        W,
        project_fn,
        ref_norm,
        norm_fn=lambda vec: float(_global_norm_local(vec, context).item()),
    )


def _resolve_start_vectors_local(
    caller_vec: np.ndarray,
    map_fn: Callable[[np.ndarray], torch.Tensor],
    project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    cold_fn: Callable[[], torch.Tensor],
    context: Any,
) -> Tuple[torch.Tensor, bool]:
    """Distributed :func:`_resolve_start_vectors` over per-rank row blocks.

    Each row is mapped to this rank's rows, projected with the local (collective)
    projector, and normalized by its **global** norm; the cold-start decision is
    therefore identical on every rank.
    """
    return _resolve_start_vectors(
        caller_vec,
        map_fn,
        project_fn,
        cold_fn,
        norm_fn=lambda vec: float(_global_norm_local(vec, context).item()),
    )


def _gather_local_row_blocks(
    blocks: torch.Tensor, partition: Any, context: Any
) -> Optional[torch.Tensor]:
    """Gather ``(n_roots, N_local)`` row blocks to rank 0 as one CPU tensor.

    The small-result hand-back: rank 0 returns
    ``(n_roots, partition.n)`` on CPU and every other rank returns ``None``.
    Only used below the size threshold; above it the caller writes memmapped
    files instead, so no non-zero rank ever holds the full vector.

    ``partition`` is either a :class:`RowPartition`-like object (an explicit
    ``bounds`` list plus ``max_local``) or a flat :class:`RankBounds` from
    ``partition_vector``; the latter is expanded with the canonical flat
    layout, which is exactly what ``DistributedVector.to_global`` reassembles.

    Complex blocks are gathered as an explicit ``(real, imag)`` pair: NCCL has
    no complex scalar type, so a complex collective is not portable.
    """
    import torch.distributed as dist

    from torch_sci_solver.torch_impl.distributed_davidson import partition_vector

    is_complex = blocks.is_complex()
    payload = torch.view_as_real(blocks.contiguous()) if is_complex else blocks
    explicit = getattr(partition, "bounds", None)
    if explicit is None:
        rank_bounds = [
            partition_vector(int(partition.n), context.world_size, rank)
            for rank in range(context.world_size)
        ]
        max_local = (int(partition.n) + context.world_size - 1) // context.world_size
    else:
        rank_bounds = list(explicit)
        max_local = int(partition.max_local)
    padded = torch.zeros(
        payload.shape[0], max_local, *payload.shape[2:], dtype=payload.dtype, device=payload.device
    )
    padded[:, : payload.shape[1]] = payload
    if context.rank == 0:
        gathered = [torch.empty_like(padded) for _ in range(context.world_size)]
    else:
        gathered = None
    dist.gather(padded, gathered, dst=0, group=context.group)
    if context.rank != 0:
        return None
    full = torch.empty(
        payload.shape[0], int(partition.n), *payload.shape[2:], dtype=payload.dtype, device="cpu"
    )
    for rank, chunk in enumerate(gathered):
        bounds = rank_bounds[rank]
        full[:, bounds.start : bounds.stop] = chunk[:, : bounds.local_size].cpu()
    return torch.view_as_complex(full.contiguous()) if is_complex else full


def _gather_local_tile_blocks(
    blocks: torch.Tensor, partition: Any, context: Any
) -> Optional[torch.Tensor]:
    """Gather ``(n_roots, N_local)`` tile blocks to rank 0 as one CPU tensor.

    The tile counterpart of :func:`_gather_local_row_blocks`: every rank's
    block is scattered back into its rectangle of the ``(M_a, M_b)`` matrix
    on rank 0, so the caller can reshape the result exactly as it does for
    the row-block gather. Uses an object gather because the per-rank blocks
    differ in shape; it is only used below the small-result threshold, where
    the full vector fits on the host anyway. Complex blocks travel as their
    natural complex tensors (object collectives pickle them).
    """
    import torch.distributed as dist

    local = blocks.detach().cpu()
    n_roots = int(local.shape[0])
    m_a = int(partition.n_rows)
    m_b = int(partition.n_cols)
    payloads: list[Any] = [None] * int(context.world_size)
    dist.gather_object(
        (local,),
        payloads if context.rank == 0 else None,
        dst=0,
        group=context.group,
    )
    if context.rank != 0:
        return None
    full = torch.zeros(n_roots, m_a, m_b, dtype=local.dtype, device="cpu")
    for rank, (chunk,) in enumerate(payloads):
        bound = partition.for_rank(rank)
        view = chunk.to(local.dtype).reshape(n_roots, bound.local_rows, bound.local_cols)
        full[:, bound.row_start : bound.row_stop, bound.col_start : bound.col_stop] = view
    return full.reshape(n_roots, m_a * m_b)


def _gather_local_pair_tile_blocks(
    blocks: torch.Tensor, bounds: Any, context: Any
) -> Optional[torch.Tensor]:
    """Gather ``(n_roots, N_local)`` scattered pair-tile blocks to rank 0.

    The GDB counterpart of :func:`_gather_local_tile_blocks`: a rank owns a
    sorted, scattered subset ``bounds.pair_index`` of the flat pair vector, so
    each rank ships its global pair indices alongside its values with one
    ``all_gather_object`` and rank 0 scatters them into ``(n_roots, n_pairs)``
    on CPU (every other rank returns ``None``). Object collectives pickle the
    natural complex tensors, so no real/imag split is needed. Only used below
    the small-result threshold; above it the caller writes memmapped files.
    """
    return _gather_local_scattered_blocks(blocks, bounds.pair_index, int(bounds.n_pairs), context)


def _gather_local_scattered_blocks(
    blocks: torch.Tensor, index: torch.Tensor, n: int, context: Any
) -> Optional[torch.Tensor]:
    """Gather ``(n_roots, N_local)`` scattered blocks to rank 0 as one CPU tensor.

    Spinless-tile counterpart of :func:`_gather_local_pair_tile_blocks`: a rank
    owns a sorted, scattered subset ``index`` of the flat spinless vector, so
    each rank ships its global determinant positions alongside its values with
    one ``all_gather_object`` and rank 0 scatters them into ``(n_roots, n)`` on
    CPU (every other rank returns ``None``). Object collectives pickle the
    natural complex tensors, so no real/imag split is needed. Only used below
    the small-result threshold; above it the caller writes memmapped files.
    """
    import torch.distributed as dist

    local = blocks.detach().cpu()
    n_roots = int(local.shape[0])
    global_index = index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    payloads: list[Any] = [None] * int(context.world_size)
    dist.all_gather_object(payloads, (global_index, local), group=context.group)
    if context.rank != 0:
        return None
    full = torch.zeros(n_roots, int(n), dtype=local.dtype, device="cpu")
    for indices, values in payloads:
        full[:, indices] = values.to(local.dtype)
    return full


class _ScatteredIndexLayout:
    """Index layout of one rank's sorted, scattered local block.

    ``global_smallest_hii`` asks a layout for the global position of local
    entries, and only ever about the owning rank, so a scattered determinant
    list (a spinless tile's ``determinant_index``) answers every query. The
    contiguous owner-computes block uses the same view with an ``arange``.
    """

    def __init__(self, index: torch.Tensor, n: int, world_size: int, rank: int) -> None:
        """Store the owned global indices and the global shape."""
        self._index = index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        self._n = int(n)
        self._world_size = int(world_size)
        self._rank = int(rank)

    @property
    def n(self) -> int:
        """Global determinant count."""
        return self._n

    @property
    def world_size(self) -> int:
        """Number of ranks."""
        return self._world_size

    def _check_rank(self, rank: int) -> None:
        """Reject ranks other than the block owner."""
        if rank != self._rank:
            raise ValueError(f"this layout holds rank {self._rank}'s block, not rank {rank}")

    def local_size(self, rank: int) -> int:
        """Number of entries this rank owns."""
        self._check_rank(rank)
        return int(self._index.numel())

    def local_to_global(self, rank: int, index: torch.Tensor) -> torch.Tensor:
        """Map local positions to global determinant positions."""
        self._check_rank(rank)
        idx = torch.as_tensor(index, dtype=torch.int64)
        return self._index[idx.cpu()].to(idx.device)


@dataclass(frozen=True)
class SCIResultWithConvergence(SCIResult):
    """SCIResultWithConvergence."""

    state: str | None = None


_PAIR_SWAP_LOOKUP_CHUNK = 1 << 22  # pairs per searchsorted probe (~64 MB query)

# Largest allowed growth of the caller's pair list when a tiled GDB solve
# embeds an irregular list in the Cartesian product of its unique strings. See
# ``pair_superset_embedding``: the product is the only layout the block and
# tile kernels can amortize without an exact decomposition, and its cost is
# linear in this ratio. Four still solves a list holding a quarter of its
# product while a sparser recovered list gets a clear error instead of a
# silent 10x memory bill.
_PAIR_SUPERSET_MAX_RATIO = 4.0

# A group covering less than this fraction of the pair basis marks a block
# decomposition as unstructured: a product-like basis has one group covering
# everything, and a coarse-block basis has a few large groups, while a
# recovered pair list splits into dozens of tiny partner-set groups whose
# per-group overhead dominates the block/tile matvec.
_PAIR_BLOCKS_STRUCTURED_FRACTION = 0.2


def _pair_ab_swap_permutation(
    pair_basis,
    n_orb: int,
    device: torch.device,
    dtype: torch.dtype,
    pair_range: "tuple[int, int] | None" = None,
    pair_indices: "torch.Tensor | None" = None,
) -> "Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]":
    """Compute the α↔β-swap permutation and τ signs for pair-mode ab-projection.

    Returns ``(k_swap, tau)`` on ``device`` if the pair set is closed under
    the (α, β) → (β, α) map — for every pair k there exists a pair k' with
    α_of_pair[k'] = β_of_pair[k] and β_of_pair[k'] = α_of_pair[k]. When
    closure fails, returns ``(None, None)`` so the caller can skip the
    projection silently. ``tau[k] = σ[k] × σ[k_swap[k]]`` is the per-pair
    ±1 sign needed for the interleaved-Wick projector
    ``P(v)[k] = (v[k] + τ[k] · v[k_swap[k]]) / 2``.

    ``PairBasis`` is lex-sorted by ``(α, β)``, so the swapped partner is found
    with ``np.searchsorted`` over a structured view, chunked to bound memory.
    A per-pair Python dict costs ~200 bytes per entry — tens of GB at 2·10⁸
    pairs per rank, which OOM-killed the GDB local path on a 64 GiB node.

    Args:
        pair_basis: the lex-sorted pair basis.
        n_orb: number of spatial orbitals, for the interleaved-Wick signs.
        device: device of the returned tensors.
        dtype: real or complex dtype of ``tau``.
        pair_range: optional ``(start, stop)`` restricting the returned block.
            The closure check still covers the whole basis, so every rank
            reaches the same include/skip decision.
        pair_indices: optional global pair indices whose swap partners and
            signs are returned, in the given order — a scattered tile's owned
            pairs. Mutually exclusive with ``pair_range``; the closure check
            still covers the whole basis, so every rank reaches the same
            decision.

    Returns:
        ``(k_swap, tau)`` indexed by global pair index (or the ``pair_range``
        slice, or in ``pair_indices`` order), or ``(None, None)`` when the
        basis is not swap-closed.
    """
    from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

    if pair_range is not None and pair_indices is not None:
        raise ValueError("pass at most one of pair_range and pair_indices")
    n = int(pair_basis.n_pairs)
    if pair_indices is None:
        start, stop = pair_range if pair_range is not None else (0, n)
        k_swap_np = np.empty(stop - start, dtype=np.int64)
        selected_np = None
    else:
        start = stop = 0
        selected_np = torch.as_tensor(pair_indices, dtype=torch.int64).reshape(-1).cpu().numpy()
        if selected_np.size and (int(selected_np.min()) < 0 or int(selected_np.max()) >= n):
            raise ValueError(f"pair_indices outside [0, {n})")
        k_swap_np = None
    chunk = _PAIR_SWAP_LOOKUP_CHUNK  # pairs per probe; the query array is ~64 MB at this size
    if pair_basis.alpha_of_pair.dim() >= 2:
        # Limb path: the lexicographic pair order is also the (α_uid, β_uid)
        # uid order, so the swapped partner is the uid key (β_uid, α_uid)
        # found by one searchsorted over the scalar key α_uid·M_β + β_uid.
        m_b = int(pair_basis.n_unique_betas)
        uid_a = pair_basis.alpha_uid_of_pair.to(torch.int64).cpu().numpy()
        uid_b = pair_basis.beta_uid_of_pair.to(torch.int64).cpu().numpy()
        key_np = uid_a * m_b + uid_b
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            query = uid_b[lo:hi] * m_b + uid_a[lo:hi]
            idx = np.searchsorted(key_np, query)
            found = idx < n
            exact = np.zeros(hi - lo, dtype=bool)
            exact[found] = key_np[idx[found]] == query[found]
            if not exact.all():
                return None, None
            if selected_np is None:
                a, b = max(lo, start), min(hi, stop)
                if a < b:
                    k_swap_np[a - start : b - start] = idx[a - lo : b - lo]
        if selected_np is not None:
            query = uid_b[selected_np] * m_b + uid_a[selected_np]
            idx = np.searchsorted(key_np, query)
            found = idx < n
            exact = np.zeros(selected_np.size, dtype=bool)
            exact[found] = key_np[idx[found]] == query[found]
            if not exact.all():
                return None, None
            k_swap_np = idx
    else:
        # The cpu-tensor copies die with each assignment, so only ``keys``
        # (16 bytes per pair) stays resident while the lookup runs.
        keys = np.empty(n, dtype=[("a", np.int64), ("b", np.int64)])
        keys["a"] = pair_basis.alpha_of_pair.cpu().numpy()
        keys["b"] = pair_basis.beta_of_pair.cpu().numpy()
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            query = np.empty(hi - lo, dtype=keys.dtype)
            query["a"] = keys["b"][lo:hi]
            query["b"] = keys["a"][lo:hi]
            idx = np.searchsorted(keys, query)
            found = idx < n
            exact = np.zeros(hi - lo, dtype=bool)
            exact[found] = (keys["a"][idx[found]] == query["a"][found]) & (
                keys["b"][idx[found]] == query["b"][found]
            )
            if not exact.all():
                return None, None
            if selected_np is None:
                a, b = max(lo, start), min(hi, stop)
                if a < b:
                    k_swap_np[a - start : b - start] = idx[a - lo : b - lo]
        if selected_np is not None:
            query = np.empty(selected_np.size, dtype=keys.dtype)
            query["a"] = keys["b"][selected_np]
            query["b"] = keys["a"][selected_np]
            idx = np.searchsorted(keys, query)
            found = idx < n
            exact = np.zeros(selected_np.size, dtype=bool)
            exact[found] = (keys["a"][idx[found]] == query["a"][found]) & (
                keys["b"][idx[found]] == query["b"][found]
            )
            if not exact.all():
                return None, None
            k_swap_np = idx
    k_swap = torch.from_numpy(k_swap_np).to(device)
    sigma = _inter_to_block_signs_pair(
        pair_basis.alpha_of_pair.to(device),
        pair_basis.beta_of_pair.to(device),
        n_orb,
    ).to(dtype=dtype)
    if selected_np is None:
        tau = (sigma[start:stop] * sigma[k_swap]).contiguous()
    else:
        selected = torch.from_numpy(selected_np).to(device)
        tau = (sigma[selected] * sigma[k_swap]).contiguous()
    return k_swap, tau


def _integer_argmin(columns: "list[torch.Tensor]") -> int:
    """Row index of the lexicographic minimum of unsigned 64-bit key columns.

    ``columns`` are equal-length int64 tensors, most significant first; each is
    compared as unsigned words (``w ^ INT64_MIN`` maps unsigned order onto signed
    order). The first column is scanned in full, later ones only on the rows that
    tie, so the transient memory is one key column plus the tie set.
    """
    flip = torch.iinfo(torch.int64).min
    keys = columns[0] ^ flip
    candidates = torch.nonzero(keys == keys.min()).flatten()
    del keys
    for column in columns[1:]:
        if candidates.numel() == 1:
            break
        keys = column[candidates] ^ flip
        candidates = candidates[keys == keys.min()]
    return int(candidates[0].item())


def _limb_columns(strings: torch.Tensor) -> "list[torch.Tensor]":
    """The limb columns of ``(M,)`` or ``(M, n_limbs)`` strings, most significant first."""
    if strings.dim() == 1:
        return [strings]
    return [strings[:, k] for k in range(strings.shape[1] - 1, -1, -1)]


def _smallest_int_pair_index(pair_basis: Any) -> int:
    """Index of the pair with the smallest ``(α, β)`` integer value.

    A single-word pair basis is sorted by its signed int64 ``(α, β)`` bit
    patterns, so this is index 0 unless a string uses bit 63 (orbital 63 of a
    64-orbital basis), which sorts first as a negative pattern. A limb basis is
    sorted by its ``(N, n_limbs)`` rows with limb 0 (bits ``[0, 64)``) compared
    first, which is not integer order either: index 0 is then the pair whose
    *low* word is smallest, e.g. one with every electron above orbital 63. A
    seed there can sit in a sector the Hamiltonian barely couples to the
    HF-like determinants.

    Notes:
        Integer order compares the most significant limb first, each limb as an
        unsigned 64-bit word (:func:`_integer_argmin`), over
        ``α[:, L-1], …, α[:, 0], β[:, L-1], …, β[:, 0]``. A structural basis
        stores no flat pair arrays; every block is a full Cartesian product
        ``U_b × V_b``, so its smallest pair is ``(min U_b, min V_b)``, and the
        seed is the smallest of those per-block minima, located with
        ``pair_index_of_uids``.
    """
    from torch_sci_solver.torch_impl.pairs import StructuralPairBasis

    if isinstance(pair_basis, StructuralPairBasis):
        best = None
        for block in pair_basis.blocks:
            if block.alpha_stop <= block.alpha_start or block.beta_stop <= block.beta_start:
                continue
            alphas = pair_basis.unique_alphas[block.alpha_start : block.alpha_stop]
            betas = pair_basis.unique_betas[block.beta_start : block.beta_stop]
            ia = block.alpha_start + _integer_argmin(_limb_columns(alphas))
            ib = block.beta_start + _integer_argmin(_limb_columns(betas))
            key = tuple(
                int(v) % (1 << 64)
                for v in torch
                .cat([
                    torch.stack(_limb_columns(pair_basis.unique_alphas[ia : ia + 1])),
                    torch.stack(_limb_columns(pair_basis.unique_betas[ib : ib + 1])),
                ])
                .flatten()
                .tolist()
            )
            if best is None or key < best[0]:
                best = (key, ia, ib)
        if best is None:
            return 0
        device = pair_basis.unique_alphas.device
        found, index = pair_basis.pair_index_of_uids(
            torch.tensor([best[1]], dtype=torch.int64, device=device),
            torch.tensor([best[2]], dtype=torch.int64, device=device),
        )
        return int(index[0].item()) if bool(found[0].item()) else 0
    alpha = getattr(pair_basis, "alpha_of_pair", None)
    if alpha is None or int(alpha.shape[0]) == 0:
        return 0
    beta = pair_basis.beta_of_pair
    if alpha.dim() == 1 and int(alpha.min()) >= 0 and int(beta.min()) >= 0:
        return 0
    return _integer_argmin(_limb_columns(alpha) + _limb_columns(beta))


def _pair_strings_at(pair_basis, indices: "torch.Tensor"):
    """Alpha/beta bitmasks of the given global pair indices.

    Works for an explicit :class:`PairBasis` (a gather from the flat arrays)
    and for a structural basis (block-grid arithmetic).
    """
    from torch_sci_solver.torch_impl.pairs import StructuralPairBasis

    if isinstance(pair_basis, StructuralPairBasis):
        return pair_basis.pair_strings(indices)
    return pair_basis.alpha_of_pair[indices], pair_basis.beta_of_pair[indices]


def _pair_blocks_are_structured(blocks) -> bool:
    """Whether a block decomposition groups the pairs into large rectangles.

    A product-like basis has one group per spin, covering the whole basis; a
    coarse-block basis has a handful of large groups. A recovered pair list
    gives each string its own partner set, so the exact decomposition splits
    into dozens of tiny groups whose per-group overhead dominates the block
    and tile matvecs: still exact, but not a scalable operator layout. A
    missing decomposition (``None``) counts as unstructured.

    Args:
        blocks: A :class:`PairBlocks` decomposition or ``None``.

    Returns:
        True when the largest group covers at least
        ``_PAIR_BLOCKS_STRUCTURED_FRACTION`` of the basis.
    """
    if blocks is None:
        return False
    n = int(blocks.n_pairs)
    if n <= 0:
        return True
    stats = blocks.stats
    largest = max(int(stats["largest_group_beta"]), int(stats["largest_group_alpha"]))
    return largest >= _PAIR_BLOCKS_STRUCTURED_FRACTION * n


def _check_structural_pair_superset(
    pair_basis,
    cached_alpha: "list[int]",
    cached_beta: "list[int]",
    device: torch.device,
) -> None:
    """Raise unless every cached warm-start pair is in a structural basis.

    A structural basis is the caller's block spec, so it cannot be extended
    pair by pair the way an explicit pair list can. Warming from a cached
    state therefore only needs the superset direction: every cached pair must
    be present, or the embedding would silently drop the amplitude of the
    missing pairs and the Ritz value would not bracket the cached energy.

    The basis' unique α/β arrays are block-major rather than globally sorted,
    so membership goes through a sorted copy plus the sorted-position → uid
    permutation, mirroring the post-processing lookups. Pair membership is
    then resolved from the block geometry by
    :meth:`StructuralPairBasis.pair_index_of_uids`.

    Args:
        pair_basis: The structural basis of the current solve.
        cached_alpha, cached_beta: Cached pair basis, parallel lists.
        device: Device holding both the basis and the cached lists.

    Raises:
        ValueError: at least one cached pair is missing from the basis.
    """
    from torch_sci_solver.torch_impl.pair_helpers import _dispatch_found_in, _sort_strings

    # The structural unique arrays are block-major; sort rows lexicographically
    # (limb 0 first) and keep the sorted-position -> uid permutation.
    unique_alphas, alpha_perm = _sort_strings(
        pair_basis.unique_alphas.to(device=device, dtype=torch.int64)
    )
    unique_betas, beta_perm = _sort_strings(
        pair_basis.unique_betas.to(device=device, dtype=torch.int64)
    )
    alpha = torch.as_tensor(cached_alpha, dtype=torch.int64, device=device)
    beta = torch.as_tensor(cached_beta, dtype=torch.int64, device=device)
    a_found, a_pos = _dispatch_found_in(unique_alphas, alpha)
    b_found, b_pos = _dispatch_found_in(unique_betas, beta)
    found, _ = pair_basis.pair_index_of_uids(alpha_perm[a_pos], beta_perm[b_pos])
    present = a_found & b_found & found
    n_missing = int(present.numel()) - int(present.sum().item())
    if n_missing:
        raise ValueError(
            "Warm-start requires the new pair set to be a superset of the "
            f"cached pair set. Missing from the structural basis: {n_missing} "
            f"pair(s). Cached size: {present.numel()}, structural basis size: "
            f"{int(pair_basis.n_pairs)}."
        )


def _pair_ab_swap(
    pair_basis,
    n_orb: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    pair_range: "Optional[tuple[int, int]]" = None,
    pair_indices: "Optional[torch.Tensor]" = None,
):
    """Dispatch the α↔β-swap lookup on the pair-basis representation."""
    from torch_sci_solver.torch_impl.pairs import StructuralPairBasis

    if isinstance(pair_basis, StructuralPairBasis):
        indices = pair_indices
        if indices is None and pair_range is not None:
            indices = torch.arange(pair_range[0], pair_range[1], dtype=torch.int64, device=device)
        return _structural_pair_ab_swap_permutation(
            pair_basis, n_orb, device, dtype, pair_indices=indices
        )
    return _pair_ab_swap_permutation(
        pair_basis,
        n_orb,
        device,
        dtype,
        pair_range=pair_range,
        pair_indices=pair_indices,
    )


def _structural_pair_ab_swap_permutation(
    pair_basis,
    n_orb: int,
    device: torch.device,
    dtype: torch.dtype,
    pair_indices: "Optional[torch.Tensor]" = None,
):
    """α↔β-swap partners and τ signs for a structural pair basis.

    The same closure test as :func:`_pair_ab_swap_permutation`, but the swapped
    partner is located with the structural block layout instead of a sorted
    pair-key array: each string keeps its block membership, so the swapped uid
    pair is looked up in ``pair_index_of_uids``. The unique string axes are
    sorted lexicographically (limb 0 first) and searched row-wise, so single-word
    and ``(M, n_limbs)`` limb bases take the same lookup.

    Args:
        pair_basis: the structural basis.
        n_orb: number of spatial orbitals, for the interleaved-Wick signs.
        device: device of the returned tensors.
        dtype: real or complex dtype of ``tau``.
        pair_indices: optional global pair indices whose partners and signs are
            returned. ``None`` returns the full basis, which is only safe when
            the caller has the full vector anyway.

    Returns:
        ``(k_swap, tau)``, or ``(None, None)`` when the basis is not closed
        under ``(α, β) → (β, α)``.
    """
    from torch_sci_solver.torch_impl.pair_helpers import _dispatch_found_in, _sort_strings
    from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

    # The include/skip decision must be identical on every rank, so it uses a
    # block-level criterion over the whole basis, not just the owned pairs:
    # every block's swapped rectangle ``V_i × U_i`` must itself be a block.
    # This is the structural shape an α↔β-symmetric basis has; anything else
    # is conservatively treated as not closed (projection skipped).
    block_ranges = [
        (b.alpha_start, b.alpha_stop, b.beta_start, b.beta_stop) for b in pair_basis.blocks
    ]
    range_set = set(block_ranges)
    for alpha_start, alpha_stop, beta_start, beta_stop in block_ranges:
        swapped = (beta_start, beta_stop, alpha_start, alpha_stop)
        if swapped not in range_set:
            return None, None
        if not torch.equal(
            pair_basis.unique_alphas[swapped[0] : swapped[1]],
            pair_basis.unique_betas[beta_start:beta_stop],
        ) or not torch.equal(
            pair_basis.unique_betas[swapped[2] : swapped[3]],
            pair_basis.unique_alphas[alpha_start:alpha_stop],
        ):
            return None, None

    n = int(pair_basis.n_pairs)
    if pair_indices is None:
        indices = torch.arange(n, dtype=torch.int64, device=device)
    else:
        indices = torch.as_tensor(pair_indices, dtype=torch.int64, device=device).reshape(-1)
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= n):
            raise ValueError(f"pair_indices outside [0, {n})")

    alpha_uid, beta_uid = pair_basis.pair_uids(indices)
    alpha_str = pair_basis.unique_alphas[alpha_uid]
    beta_str = pair_basis.unique_betas[beta_uid]

    alpha_values, alpha_order = _sort_strings(pair_basis.unique_alphas)
    beta_values, beta_order = _sort_strings(pair_basis.unique_betas)

    def _lookup(values: torch.Tensor, order: torch.Tensor, query: torch.Tensor):
        found, pos = _dispatch_found_in(values, query)
        return found, order[pos.clamp(max=int(values.shape[0]) - 1)]

    found_beta, swapped_beta_uid = _lookup(beta_values, beta_order, alpha_str)
    found_alpha, swapped_alpha_uid = _lookup(alpha_values, alpha_order, beta_str)
    found, k_swap = pair_basis.pair_index_of_uids(swapped_alpha_uid, swapped_beta_uid)
    if not bool((found & found_alpha & found_beta).all()):
        raise ValueError(
            "structural α↔β swap: the block layout is closed under the swap but "
            "an owned pair's swapped partner could not be located"
        )

    sigma = _inter_to_block_signs_pair(alpha_str.to(device), beta_str.to(device), n_orb)
    swap_sigma = _inter_to_block_signs_pair(
        pair_basis.unique_alphas[swapped_alpha_uid].to(device),
        pair_basis.unique_betas[swapped_beta_uid].to(device),
        n_orb,
    )
    return k_swap, (sigma * swap_sigma).to(dtype=dtype).contiguous()


@dataclass
class EigenStates:
    """Container for all solved roots of one CI subspace.

    Mirrors ``sci_solvers.eigen_states.EigenStates`` from the parent package
    without depending on it (the torch package is standalone). Each element
    of ``states`` is one :class:`SCIResultWithConvergence`, sorted ascending
    in energy. ``states[0]`` is the ground state; ``states[1:]`` are excited
    states. Single-root solves produce an ``EigenStates`` of length one.

    This is the return type of :meth:`TorchSCISolver.__call__` so that all
    roots are directly accessible from the call return value, matching the
    pattern used by ``sci_solvers.MultirootSCISolver``.
    """

    states: List[SCIResultWithConvergence]


# ===========================================================================
# Process-per-GPU worker pool
# ===========================================================================
#
# The multi-device fan-out uses one OS *process* per GPU rather than one
# thread, because each Davidson solve is only ~15-30% GPU work — the rest is
# Python (connectivity enumeration in ``setup`` + per-iteration dispatch) that
# holds the GIL. With threads the workers serialize on the GIL and both GPUs
# sit idle; measured 0.95x (a slowdown). With processes each worker has its own
# GIL and the subspaces genuinely overlap; measured ~1.95x on 2 GPUs.
#
# Design constraints (see also the module-level multi-GPU notes):
#   * ``spawn`` start method — fork + an initialized CUDA context deadlocks.
#   * The parent stays OFF-GPU (GPUs may be in Exclusive_Process compute mode,
#     one context per device): it ships the CPU-side ``(I0, I1, I2)`` integrals
#     bundle and each worker builds its own device ``TorchIntegrals``.
#   * The pool is PERSISTENT across ``__call__``s — worker startup is ~10s
#     (CUDA context + Triton JIT), and qiskit calls the same solver once per
#     SQD iteration, so a per-call pool would pay that cost every iteration.
#
# All symbols here are module-level (no closures) so they survive ``spawn``
# re-import into the child.

# Shared spawn context — a single module-level instance so every pool uses the
# CUDA-safe start method regardless of the process-global default.
_MP_CTX = multiprocessing.get_context("spawn")


@dataclass
class _Task:
    """One subspace of work shipped to a worker. Every field is picklable.

    ``integrals_bundle`` is the CPU ``(I0, I1, I2)`` numpy bundle; it is only
    attached when the worker might not yet have this fingerprint cached (the
    first N tasks of a batch with a new ``fp``), else ``None`` to avoid
    re-pickling ~13 MB every subspace.
    """

    sub_idx: int
    strings_a: Any
    strings_b: Any
    fp: Optional[str]
    integrals_bundle: Optional[tuple]
    one_body_tensor: Any
    two_body_tensor: Any
    norb: int
    nelec: tuple
    n_a: int
    n_b: int
    spin_sq: Optional[float]
    spin_sq_target: Optional[float]
    probe_method: Optional[str]
    probe_kwargs_in: dict
    build_output_states: bool
    solver_knob_snapshot: dict
    call_max_iter: int
    call_eps: float
    call_de_tol: Optional[float]
    call_spin_lambda: float
    probe_state_this: Any
    call_idx: int
    init_vector: Any = None


@dataclass
class _Result:
    """A worker's response for one subspace. Picklable (CPU-only payloads)."""

    sub_idx: int
    ok: bool
    sci_result: Any = None
    output_state: Any = None
    perf: Any = None
    stats_delta: Optional[dict] = None
    exc: Any = None
    # Which device ran this task and which integrals fingerprint it now has
    # cached — lets the parent confirm every worker has the current bundle
    # before it stops shipping it (workers pull from a shared queue, so the
    # parent can't know a priori which worker got which task).
    worker_dev: Optional[str] = None
    cached_fp: Optional[str] = None


class _Shutdown:
    """Sentinel enqueued once per worker to end its loop."""


# Stats keys the worker's throwaway ``stats`` dict accumulates and the parent
# sums back. Kept in one place so the worker delta and the parent merge agree.
_ACCUM_STATS_KEYS = (
    "t_seed",
    "t_setup",
    "t_spin_sq",
    "t_davidson",
    "t_occ",
    "t_rdm",
    "t_integrals_build",
)


def _pickle_safe_exc(e: BaseException) -> BaseException:
    """Return an exception safe to pickle back to the parent, preserving type
    and message for the common builtin str-only exceptions (so callers /
    ``pytest.raises(match=...)`` still match). Anything exotic is downgraded to
    a RuntimeError carrying the original repr."""
    for cls in (ValueError, TypeError, KeyError, RuntimeError, MemoryError):
        if isinstance(e, cls):
            try:
                return cls(*e.args)
            except Exception:
                break
    return RuntimeError(repr(e))


def _build_standin_solver(cfg: dict, dev_str: str) -> "TorchSCISolver":
    """Build a single-device ``TorchSCISolver`` inside a worker from the shipped
    scalar config. This stand-in supplies every ``self.*`` attribute that
    ``_solve_one_subspace`` reads (fresh locks, fresh zeroed stats), so the
    method runs unchanged and its stats mutations land in a throwaway dict we
    diff into a per-subspace delta."""
    return TorchSCISolver(device=dev_str, devices=None, **cfg)


def _run_task_in_worker(
    solver: "TorchSCISolver", dev: torch.device, integ_cache: dict, task: _Task
) -> _Result:
    """Execute one subspace on this worker's device and return a ``_Result``.

    Builds (and per-``fp`` caches on this device) the ``TorchIntegrals`` from the
    shipped CPU bundle, resets the stand-in's stats so the delta is clean, runs
    the unchanged ``_solve_one_subspace``, and captures the stats it accumulated.
    """
    # --- device integrals, cached per fingerprint (LRU-1, mirrors the parent) ---
    integ = integ_cache.get(task.fp) if task.fp is not None else None
    t_integ = 0.0
    if integ is None:
        if task.integrals_bundle is None:
            # Protocol invariant: on a cache miss the parent must have shipped
            # the bundle (it attaches it to the first N tasks of a new-fp batch).
            raise RuntimeError(
                f"worker {dev} has no cached integrals for fp={task.fp!r} and the "
                "task carried no bundle — integrals-shipping protocol violated"
            )
        I0, I1, I2 = task.integrals_bundle
        _t0 = _time.perf_counter()
        integ = to_torch_integrals(
            I0,
            I1,
            I2,
            device=dev,
            dtype=solver.dtype,
            # The compact spin-adapted store is 16x smaller than the dense
            # spin-orbital tensor; the Triton kernels index a flat dense copy,
            # and below 32 spatial orbitals the dense tensor is small enough
            # that the wrapper's indexing arithmetic is not worth it.
            compact_eri=_resolve_compact_eri(
                getattr(solver, "compact_eri", "auto"),
                matvec_impl=getattr(solver, "matvec_impl", "batched"),
                mode=getattr(solver, "mode", "tpb"),
                n_spinorb=int(I1.n_spinorb),
            ),
        )
        t_integ = _time.perf_counter() - _t0
        if task.fp is not None:
            integ_cache.clear()  # LRU-1
            integ_cache[task.fp] = integ

    # --- fresh stats so we can diff exactly this subspace's contribution ---
    solver.stats = {k: 0.0 for k in _ACCUM_STATS_KEYS}
    solver.stats["n_subspaces"] = 0
    solver.stats["spin_squared"] = []
    solver.stats["spin_sq_final"] = None
    solver.stats["n_calls"] = task.call_idx  # for TORCH_SCI_DUMP_DIR filenames

    with _deterministic_algorithms(bool(getattr(solver, "deterministic", False))):
        sci_result, output_state, perf = solver._solve_one_subspace(
            sub_idx=task.sub_idx,
            strings_a=task.strings_a,
            strings_b=task.strings_b,
            probe_state_this=task.probe_state_this,
            one_body_tensor=task.one_body_tensor,
            two_body_tensor=task.two_body_tensor,
            norb=task.norb,
            nelec=task.nelec,
            n_a=task.n_a,
            n_b=task.n_b,
            spin_sq=task.spin_sq,
            spin_sq_target=task.spin_sq_target,
            integrals=integ,
            get_integrals_fingerprint=(lambda _fp=task.fp: _fp),
            probe_method=task.probe_method,
            probe_kwargs_in=task.probe_kwargs_in,
            build_output_states=task.build_output_states,
            solver_knob_snapshot=task.solver_knob_snapshot,
            call_max_iter=task.call_max_iter,
            call_eps=task.call_eps,
            call_de_tol=task.call_de_tol,
            call_spin_lambda=task.call_spin_lambda,
            device=dev,
            init_vector=task.init_vector,
        )

    # --- assemble the stats delta the parent will merge ---
    delta = {k: float(solver.stats.get(k, 0.0)) for k in _ACCUM_STATS_KEYS}
    delta["t_integrals_build"] += t_integ
    delta["n_subspaces"] = 1
    delta["spin_squared"] = list(solver.stats.get("spin_squared", []))
    delta["spin_sq_final"] = solver.stats.get("spin_sq_final")
    return _Result(
        sub_idx=task.sub_idx,
        ok=True,
        sci_result=sci_result,
        output_state=output_state,
        perf=perf,
        stats_delta=delta,
        worker_dev=str(dev),
        cached_fp=task.fp,
    )


def _worker_main(dev_str: str, cfg: dict, task_q, result_q, ready_q) -> None:
    """Worker process entry point: pin the device, build the stand-in solver,
    signal readiness, then drain tasks until a ``_Shutdown`` sentinel."""
    try:
        if cfg.get("deterministic"):
            # cuBLAS reads its workspace setting when this process creates its
            # first handle, so it must be in the environment beforehand.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.set_device(dev_str)
        dev = torch.device(dev_str)
        solver = _build_standin_solver(cfg, dev_str)
    except BaseException as e:  # startup failure — tell the parent, then exit.
        ready_q.put(("error", dev_str, repr(e)))
        return
    integ_cache: dict = {}
    ready_q.put(("ready", dev_str))

    while True:
        task = task_q.get()
        if isinstance(task, _Shutdown):
            break
        try:
            result_q.put(_run_task_in_worker(solver, dev, integ_cache, task))
        except BaseException as e:  # never raise across the boundary
            result_q.put(
                _Result(
                    sub_idx=task.sub_idx,
                    ok=False,
                    exc=_pickle_safe_exc(e),
                )
            )


def _pool_finalize(procs, task_q) -> None:
    """Module-level finalizer (weakref.finalize / atexit target). Must NOT hold a
    reference to the pool or solver, or the finalizer never fires."""
    try:
        for _ in procs:
            with contextlib.suppress(Exception):
                task_q.put(_Shutdown())
    except Exception:
        pass
    for p in procs:
        with contextlib.suppress(Exception):
            p.join(timeout=5.0)
        if p.is_alive():
            with contextlib.suppress(Exception):
                p.terminate()


class _GpuWorkerPool:
    """Persistent pool of one spawned worker process per CUDA device.

    A single shared task queue is drained by all workers, so any free worker
    grabs the next subspace (the same load-balancing the thread-pool semaphore
    gave, inverted). Results return out of order and the caller slots them by
    ``sub_idx``. The pool lives across ``__call__``s so the ~10s per-worker
    CUDA/Triton startup is paid once, not per SQD iteration.
    """

    # Longest we wait on a single result before checking worker liveness. Big
    # enough not to trip on a genuinely long Davidson; small enough to surface a
    # dead worker (CUDA OOM / segfault) in bounded time.
    _HEARTBEAT_S = 120.0
    # Cold worker spawn (torch/CUDA/Triton re-import + context init per process)
    # can exceed 120s on batch-scheduled nodes with slow shared-filesystem imports.
    # Override via TORCH_SCI_STARTUP_TIMEOUT_S when needed.
    _STARTUP_TIMEOUT_S = float(os.environ.get("TORCH_SCI_STARTUP_TIMEOUT_S", "600"))

    def __init__(self, devices: tuple, cfg: dict):
        """Store devices + the picklable scalar config; nothing is spawned yet."""
        self._devices = [str(d) for d in devices]
        self._cfg = cfg
        self._task_q = _MP_CTX.Queue()
        self._result_q = _MP_CTX.Queue()
        self._ready_q = _MP_CTX.Queue()
        self._procs: list = []
        self._started = False
        self._closed = False
        self._finalizer = None
        # (device_str) -> fp currently cached on that worker. Lets us ship the
        # CPU integrals bundle until every worker has the current fingerprint,
        # regardless of how the shared queue happened to distribute tasks.
        self._worker_fp: dict = {}

    def start(self) -> None:
        """Spawn one worker per device and block until all report ready."""
        if self._started:
            return
        for dev_str in self._devices:
            p = _MP_CTX.Process(
                target=_worker_main,
                args=(dev_str, self._cfg, self._task_q, self._result_q, self._ready_q),
                daemon=True,
                name=f"tsci-gpu-{dev_str.replace(':', '')}",
            )
            p.start()
            self._procs.append(p)

        # Wait for a READY (or error) token from every worker.
        pending = set(self._devices)
        while pending:
            try:
                msg = self._ready_q.get(timeout=self._STARTUP_TIMEOUT_S)
            except queue.Empty:
                dead = [(p.name, p.exitcode) for p in self._procs if not p.is_alive()]
                self.close()
                raise RuntimeError(
                    f"GPU worker pool startup timed out after {self._STARTUP_TIMEOUT_S}s; "
                    f"still waiting on {sorted(pending)}; dead workers: {dead}"
                )
            kind = msg[0]
            if kind == "ready":
                pending.discard(msg[1])
            else:  # ("error", dev, repr)
                self.close()
                raise RuntimeError(f"GPU worker for {msg[1]} failed to start: {msg[2]}")

        # Register teardown only now that workers exist — a single-device solver
        # never reaches here, so it never spawns or registers anything.
        self._finalizer = weakref.finalize(self, _pool_finalize, self._procs, self._task_q)
        atexit.register(self.close)
        self._started = True

    def submit_batch(self, tasks: list, fp: Optional[str], cpu_bundle: Optional[tuple]) -> dict:
        """Enqueue all tasks, collect one result per task, re-raise the first
        worker exception. Returns ``{sub_idx: _Result}`` for the successes.

        Integrals shipping: a worker builds and caches its device integrals from
        ``cpu_bundle`` on first sight of ``fp``. Because the shared queue may
        hand any task to any worker, we attach ``cpu_bundle`` to every task
        until we have confirmed (via returned results) that ALL workers hold
        ``fp`` — then stop, so subsequent calls re-pickle nothing. This is
        robust to uneven task distribution and to batches smaller than the
        worker count (e.g. a 1-subspace warm-up that only seeds one worker).

        The pool stays alive on error (workers loop back to the queue), so the
        next call reuses it.
        """
        if not self._started:
            raise RuntimeError("pool used before start()")

        all_confirmed = fp is not None and all(self._worker_fp.get(d) == fp for d in self._devices)
        for t in tasks:
            # Attach the bundle unless every worker is already known to hold fp.
            t.integrals_bundle = None if all_confirmed else cpu_bundle
            self._task_q.put(t)

        results: dict = {}
        first_exc: Optional[BaseException] = None
        pending = len(tasks)
        while pending:
            try:
                r = self._result_q.get(timeout=self._HEARTBEAT_S)
            except queue.Empty:
                dead = [(p.name, p.exitcode) for p in self._procs if not p.is_alive()]
                if dead:
                    self.close()
                    raise RuntimeError(
                        f"GPU worker(s) died: {dead}; {pending} subspace(s) still "
                        "outstanding — likely CUDA OOM or a segfault in the solve"
                    )
                continue  # workers alive, just a long Davidson — keep waiting
            pending -= 1
            if r.ok:
                results[r.sub_idx] = r
                if r.worker_dev is not None and r.cached_fp is not None:
                    self._worker_fp[r.worker_dev] = r.cached_fp
            elif first_exc is None:
                first_exc = r.exc if isinstance(r.exc, BaseException) else RuntimeError(str(r.exc))
        if first_exc is not None:
            raise first_exc
        return results

    def close(self) -> None:
        """Idempotent shutdown: stop workers, join, terminate stragglers."""
        if self._closed:
            return
        self._closed = True
        if self._finalizer is not None:
            self._finalizer.detach()  # we're doing the cleanup explicitly now
        _pool_finalize(self._procs, self._task_q)
        with contextlib.suppress(Exception):
            atexit.unregister(self.close)


#: Per-call keywords the engine actually consumes. Anything else is governed by
#: :attr:`TorchSCISolver.unknown_call_kwargs`. ``max_cycle`` is listed because it
#: is recognized only to be rejected with a targeted message.
CALL_KWARG_NAMES: frozenset[str] = frozenset({
    "max_davidson_iters",
    "tol",
    "de_tol",
    "spin_penalty_lambda",
    "init_vectors",
    "probe_method",
    "probe_kwargs",
    "probe_state",
    "max_cycle",
})

#: Values accepted by ``unknown_call_kwargs``.
UNKNOWN_CALL_KWARGS_POLICIES = ("ignore", "warn", "error")

#: Values accepted by ``table_dtype``, mapped to the torch dtype (``None`` is
#: the current int64 reference build).
TABLE_DTYPES: dict[str, "Optional[torch.dtype]"] = {
    "auto": torch.int64,
    "int32": torch.int32,
    "int64": torch.int64,
}

#: Values accepted by ``ab_factorization``.
AB_FACTORIZATION_MODES = ("off", "cholesky")

#: Values accepted by ``compact_eri``.
COMPACT_ERI_MODES = ("auto", "on", "off")

#: Recognized keys of ``TorchSCISolver(diagnostics=...)``.
DIAGNOSTIC_KEYS = frozenset({"dump_dir", "matvec_profile", "profile_blocks"})


def _dtype_name(dtype: "torch.dtype") -> str:
    """Return a stable JSON-friendly name for a torch dtype (``"torch.float64"``)."""
    return str(dtype)


def _jsonable(value: Any) -> Any:
    """Convert a config value to JSON-friendly primitives.

    Torch dtypes/devices and tuples are converted to strings/lists so
    ``json.dumps(solver.config)`` works; numpy scalars become Python scalars.
    """
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def _resolve_compact_eri(choice: str, *, matvec_impl: str, mode: str, n_spinorb: int) -> bool:
    """Resolve the ``compact_eri`` policy for one integrals build.

    ``"auto"`` keeps the historical heuristic: the 16x-smaller compact
    spin-adapted store is used whenever the matvec does not need a flat dense
    tensor (the Triton TPB kernels index one) and the spin-orbital ERI is at
    least 64x64x64x64. ``"on"`` forces it and ``"off"`` forces the dense tensor.
    """
    if choice == "on":
        return True
    if choice == "off":
        return False
    return (matvec_impl != "triton" or mode == "gdb") and int(n_spinorb) >= 64


def _use_memmap_hand_back(n_bytes: int, mode: "Optional[str]", threshold: "Optional[int]") -> bool:
    """Whether a distributed result of ``n_bytes`` is handed back through memmaps.

    ``mode`` is the resolved policy: ``"memory"`` always gathers in memory,
    ``"memmap"`` always writes files, and ``"auto"`` compares the result size
    against ``threshold`` (falling back to
    :data:`DISTRIBUTED_HAND_BACK_SMALL_BYTES`).
    """
    mode = mode or "auto"
    if mode == "memory":
        return False
    if mode == "memmap":
        return True
    if threshold is None:
        from torch_sci_solver.torch_impl.distributed_solver import (
            DISTRIBUTED_HAND_BACK_SMALL_BYTES,
        )

        threshold = DISTRIBUTED_HAND_BACK_SMALL_BYTES
    return int(n_bytes) > int(threshold)


#: Recognized keys of ``TorchSCISolver(gdb_kwargs=...)``, with their defaults.
#: These apply only to the GDB (explicit-pair) basis; they are meaningless for
#: the TPB Cartesian-product basis, which is why they live in their own bundle
#: instead of the main signature. See :class:`TorchSCISolver` for each key.
GDB_KWARG_DEFAULTS: dict[str, Any] = {
    "budget_bytes": None,
    "block_matvec": "auto",
    "block_table_bytes": None,
    "structural_blocks": None,
    "move_cache_bytes": None,
    "needed_staging": "auto",
    "tile_move_cache_bytes": None,
}


def _validate_gdb_kwargs(gdb_kwargs: "Optional[dict[str, Any]]", *, mode: str) -> dict[str, Any]:
    """Fill defaults into ``gdb_kwargs`` and reject anything unrecognized.

    A plain dict of options loses the typo protection that named parameters get
    for free, so the keys are checked explicitly here. Supplying GDB options
    while running the TPB basis is also an error rather than a silent no-op —
    otherwise setting a budget and observing no change looks like a bug in the
    solver rather than a misuse of it.

    Args:
        gdb_kwargs: Caller-supplied options, or None for all defaults.
        mode: The solver's basis mode, ``"tpb"`` or ``"gdb"``. Any non-empty
            ``gdb_kwargs`` is a mistake under ``"tpb"``.

    Returns:
        A new dict with every key in :data:`GDB_KWARG_DEFAULTS` populated.

    Raises:
        ValueError: On an unknown key, or on GDB options given under TPB.
    """
    supplied = dict(gdb_kwargs or {})
    unknown = set(supplied) - set(GDB_KWARG_DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown gdb_kwargs key(s): {sorted(unknown)}. "
            f"Recognized: {sorted(GDB_KWARG_DEFAULTS)}."
        )
    if supplied and mode != "gdb":
        raise ValueError(
            f'gdb_kwargs={supplied} was given but mode="{mode}", where these '
            f'options have no effect. Pass mode="gdb" to use the general '
            f"determinant basis, or drop gdb_kwargs."
        )
    out = {**GDB_KWARG_DEFAULTS, **supplied}
    mode_flag = out["block_matvec"]
    if isinstance(mode_flag, bool):
        out["block_matvec"] = "on" if mode_flag else "off"
    elif str(mode_flag).lower() not in ("auto", "on", "off"):
        raise ValueError(
            f"gdb_kwargs['block_matvec'] must be 'auto', 'on' or 'off', got {mode_flag!r}"
        )
    else:
        out["block_matvec"] = str(mode_flag).lower()
    table_bytes = out["block_table_bytes"]
    if table_bytes is not None and int(table_bytes) <= 0:
        raise ValueError(
            f"gdb_kwargs['block_table_bytes'] must be positive or None, got {table_bytes!r}"
        )
    move_cache = out["tile_move_cache_bytes"]
    if move_cache is not None:
        warnings.warn(
            "gdb_kwargs['tile_move_cache_bytes'] is deprecated; pass "
            "gdb_kwargs['move_cache_bytes'] instead (the tile-only spelling is "
            "kept as an alias for one release).",
            DeprecationWarning,
            stacklevel=3,
        )
        move_cache = int(move_cache)
        if move_cache <= 0:
            raise ValueError(
                f"gdb_kwargs['tile_move_cache_bytes'] must be positive or None, got {move_cache!r}"
            )
        out["tile_move_cache_bytes"] = move_cache
    block_cache = out["move_cache_bytes"]
    if block_cache is not None:
        block_cache = int(block_cache)
        if block_cache <= 0:
            raise ValueError(
                f"gdb_kwargs['move_cache_bytes'] must be positive or None, got {block_cache!r}"
            )
        if move_cache is not None and move_cache != block_cache:
            raise ValueError(
                "gdb_kwargs['move_cache_bytes'] and gdb_kwargs['tile_move_cache_bytes'] "
                f"disagree ({block_cache!r} vs {move_cache!r}); the latter is the "
                "deprecated tile-only alias, so pass only move_cache_bytes."
            )
        out["move_cache_bytes"] = block_cache
    else:
        # The tile-only key is the deprecated spelling of the same budget.
        out["move_cache_bytes"] = move_cache
    needed_staging = str(out["needed_staging"]).lower()
    if needed_staging not in ("off", "auto", "on"):
        raise ValueError(
            "gdb_kwargs['needed_staging'] must be 'off', 'auto' or 'on', "
            f"got {out['needed_staging']!r}"
        )
    out["needed_staging"] = needed_staging
    if out["structural_blocks"] is not None:
        if mode != "gdb":
            raise ValueError(
                "gdb_kwargs['structural_blocks'] requires mode='gdb'; a structural "
                "basis describes a general determinant basis"
            )
        if out["block_matvec"] == "off":
            raise ValueError(
                "gdb_kwargs['structural_blocks'] cannot be combined with "
                "block_matvec='off': a structural basis supports only the block "
                "matvec path (it stores no explicit pair list for the streamed tables)"
            )
        if not isinstance(out["structural_blocks"], (list, tuple)):
            raise ValueError(
                "gdb_kwargs['structural_blocks'] must be a sequence of "
                "(alpha_strings, beta_strings) blocks"
            )
    return out


# MPS has no float64: on the float32 path, Davidson's ‖R‖ < eps criterion
# (an absolute residual, not scaled to the energy) cannot go below roughly
# |E| * finfo(float32).eps no matter how many iterations run — that floor is
# set by float32's own precision, not by the algorithm. For typical quantum
# chemistry energy scales (|E| ~ 1-1000 Ha) that floor works out to roughly
# 1e-7 to 1e-4, so this is used as a single practical threshold rather than
# requiring callers to scale it to their own |E|.
MPS_DAVIDSON_TOLERANCE_FLOOR = 1e-5

# The dtypes MPS can hold: no float64 and no complex128.
MPS_DTYPE_NAMES = ("torch.float32", "torch.complex64")


@contextlib.contextmanager
def _deterministic_algorithms(enabled: bool):
    """Run a block under ``torch.use_deterministic_algorithms(True, warn_only=True)``.

    A no-op when ``enabled`` is false or the caller already enabled deterministic
    algorithms; otherwise the previous (off) state and its ``warn_only`` flag are
    restored on exit, so the process-wide switch is set only for the duration of
    a solve. In this mode torch's CUDA ``index_add``/``scatter_add`` use their
    deterministic implementations, and the solver's atomic Triton entry kernels
    and the cuSPARSE products of the factorized αβ pass decline in favour of
    them; ``warn_only`` turns any remaining nondeterministic op into a warning
    rather than an error (a caller that already enabled the strict mode keeps
    it). The switch is process-wide, so concurrent threads see it too.
    """
    if not enabled or torch.are_deterministic_algorithms_enabled():
        yield
        return
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False, warn_only=previous_warn_only)


def _tensors_to(obj: Any, device: "torch.device | str") -> Any:
    """A copy of a dataclass (basis or helper table) with its tensors on ``device``."""
    changes: dict[str, Any] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if torch.is_tensor(value):
            changes[field.name] = value.to(device)
        elif field.name == "device":
            changes[field.name] = torch.device(device)
    return dataclasses.replace(obj, **changes)


def _rayleigh_quotient_wide(matvec: Callable, vec: torch.Tensor, dtype: torch.dtype) -> float:
    """``Re⟨v|H|v⟩ / ⟨v|v⟩`` of ``vec`` with a CPU ``matvec`` in the wide ``dtype``."""
    v = vec.detach().cpu().to(dtype)
    hv = matvec(v)
    return float((torch.vdot(v, hv).real / torch.vdot(v, v).real).item())


def _clamp_davidson_tolerance_for_device(davidson_tolerance: float, device: torch.device) -> float:
    """Raise ``davidson_tolerance`` to :data:`MPS_DAVIDSON_TOLERANCE_FLOOR` on
    MPS, warning that it was overridden. A no-op (and no warning) on any
    other device, or when already at/above the floor.

    Below the floor Davidson would silently burn the full
    ``max_davidson_iters`` budget every call without ever converging (see
    the module comment above), so this overrides rather than only warns —
    the goal is a solver instance whose runs are always meaningful.
    """
    if device.type != "mps" or davidson_tolerance >= MPS_DAVIDSON_TOLERANCE_FLOOR:
        return davidson_tolerance
    warnings.warn(
        f"davidson_tolerance={davidson_tolerance:.1e} on device='mps' is below "
        f"{MPS_DAVIDSON_TOLERANCE_FLOOR:.0e}. MPS has no float64 support, and for "
        f"typical quantum chemistry energy scales (|E| ~ 1-1000 Ha) a residual "
        f"this tight would likely never be reached — Davidson would burn the "
        f"full max_davidson_iters budget on every call instead of converging. "
        f"Overriding davidson_tolerance to {MPS_DAVIDSON_TOLERANCE_FLOOR:.0e} for "
        f"this call so runs stay meaningful; pass a value >= "
        f"{MPS_DAVIDSON_TOLERANCE_FLOOR:.0e} explicitly to silence this.",
        RuntimeWarning,
        stacklevel=3,
    )
    return MPS_DAVIDSON_TOLERANCE_FLOOR


class _TiledPenalizedOperator:
    """Compose ``H + lam (S^2 - target)`` on the owner-computes tile block.

    The bare operator applies ``H`` to the rank's tile; the ``S^2`` term reuses
    the same staged exchanges through
    :func:`torch_sci_solver.torch_impl.tiled_postprocess.tiled_s2_apply`, so
    the penalty adds no full-vector fallback. Only ``n`` and the call are
    exposed, which is what the local Davidson driver reads.
    """

    def __init__(self, operator: Any, lam: float, target: float) -> None:
        from torch_sci_solver.torch_impl.tiled_postprocess import tiled_s2_apply

        self._operator = operator
        self._apply_s2 = tiled_s2_apply
        self._lam = float(lam)
        self._target = float(target)
        self.n = operator.n

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Return ``(H + lam (S^2 - target)) v`` on this rank's tile."""
        y = self._operator(v) + self._lam * self._apply_s2(self._operator, v)
        if self._target != 0.0:
            y = y - (self._lam * self._target) * v
        return y


class _PairTiledPenalizedOperator:
    """Compose ``H + lam (S^2 - target)`` on the owner-computes pair tile.

    The bare operator applies ``H`` to the rank's owned pairs; the ``S^2`` term
    reuses the same staged exchange through
    :func:`torch_sci_solver.torch_impl.pair_tiled_postprocess.pair_tiled_s2_apply`,
    so the penalty adds no full-vector fallback. Only ``n`` and the call are
    exposed, which is what the local Davidson driver reads.
    """

    def __init__(self, operator: Any, lam: float, target: float) -> None:
        from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
            pair_tiled_s2_apply,
        )

        self._operator = operator
        self._apply_s2 = pair_tiled_s2_apply
        self._lam = float(lam)
        self._target = float(target)
        self.n = operator.n

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Return ``(H + lam (S^2 - target)) v`` on this rank's pair block."""
        y = self._operator(v) + self._lam * self._apply_s2(self._operator, v)
        if self._target != 0.0:
            y = y - (self._lam * self._target) * v
        return y


class _PairLocalPenalizedOperator:
    """Compose ``H + lam (S^2 - target)`` on the 1-D GDB owner-computes block.

    One all-gather serves both terms: the bare operator applies ``H`` from the
    gathered vector and
    :class:`~torch_sci_solver.torch_impl.pair_tiled_postprocess.PairLocalSpinSquared`
    applies ``S^2`` from the same vector, so the penalty adds no second
    collective and no O(K) swap table. Only ``n`` and the call are exposed,
    which is what the local Davidson driver reads.
    """

    def __init__(self, operator: Any, s2: Any, lam: float, target: float) -> None:
        self._operator = operator
        self._s2 = s2
        self._lam = float(lam)
        self._target = float(target)
        self.n = operator.n

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        """Return ``(H + lam (S^2 - target)) v`` on this rank's pair block."""
        T_full = self._operator.gather_full(v)
        y = self._operator.matvec_from_full(v, T_full)
        y = y + self._lam * self._s2.apply(v, T_full)
        if self._target != 0.0:
            y = y - (self._lam * self._target) * v
        return y


class TorchSCISolver:
    """SCI solver — class form of the former closure-style factory.

    Use via the ``make_torch_sci_solver(...)`` factory which returns a
    ``TorchSCISolver`` instance. The instance is callable and matches the
    ``sci_solver`` contract expected by
    ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``:

        solver(ci_strings, h1, eri, norb, nelec, *, spin_sq=None, **kw)

    Public surface (unchanged from the closure version):
      - ``solver(...)``                  — callable solver
      - ``solver.stats``                 — dict of cumulative perf timings
      - ``solver.format_perf_summary()`` — multi-line perf table

    Internal structure (introduced by the v0.11.0 refactor):
      - ``__init__``        stores factory kwargs as attributes and
                            initializes the integrals cache, stats dict,
                            and recovery-gap tracker.
      - ``__call__``        per-call orchestration: probe-arg
                            normalization, integrals build/cache,
                            per-subspace loop, per-call aggregate.
      - utility methods     ``_free_bytes``, ``_fmt``,
                            ``_pick_num_block``, ``_get_integrals_fingerprint``
                            (extracted from former closures).
      - ``format_perf_summary``  — exposes the cumulative perf log.

    The per-subspace loop body stays inline in ``__call__`` for now. A
    follow-on can split it into named methods if needed; the conversion
    from closure-style to class-style already gives the main debuggability
    win (pdb can step into ``self.X`` instead of free vars).

    Behavioral parity: byte-identical to the v0.11.0 closure factory.
    """

    def __init__(
        self,
        *,
        device: "str | torch.device" = "cpu",
        devices: "Optional[list[str | torch.device]]" = None,
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 2_000_000,
        davidson_block_size: Optional[int] = None,
        max_davidson_iters: int = 20,
        davidson_tolerance: float = 1e-7,
        davidson_de_tolerance: Optional[float] = None,
        compute_full_rdm: bool = False,
        compute_spin_squared: bool = False,
        verbose: bool = False,
        budget_fraction: float = 0.65,
        gdb_kwargs: "Optional[dict[str, Any]]" = None,
        seed_strategy: str = "smallest_int",
        apply_ab_projection: bool = True,
        matvec_impl: str = "auto",
        mode: Literal["tpb", "gdb"] = "tpb",
        n_roots: int = 1,
        davidson_dtype: Optional[torch.dtype] = None,
        davidson_fp32_shift: "Literal['hf'] | float | None" = "hf",
        davidson_fp32_energy: Literal["auto", "rq64", "ritz"] = "auto",
        davidson_correction: Literal["davidson", "olsen"] = "davidson",
        davidson_restart: Literal["recompute", "reuse", "thick"] = "recompute",
        davidson_missed_root_check: "Literal['auto'] | bool" = "auto",
        distributed: "bool | dict[str, Any] | None" = None,
        unknown_call_kwargs: Literal["ignore", "warn", "error"] = "ignore",
        memory_budget_bytes: Optional[int] = None,
        table_dtype: Literal["auto", "int32", "int64"] = "auto",
        compact_eri: Literal["auto", "on", "off"] = "auto",
        triton_block: Optional[int] = None,
        diagnostics: "Optional[dict[str, Any]]" = None,
        deterministic: bool = False,
        ab_factorization: Literal["off", "cholesky"] = "off",
        ab_factorization_tol: float = 1e-10,
    ) -> None:
        """Build a ``sci_solver`` callable matching the qiskit_addon_sqd contract.

        Args:
            device:           torch device for the diag (typically 'cuda').
            dtype:            float dtype (float64 recommended for stability).
            chunk_size:       matvec batch size; lower on matvec OOM.
            davidson_block_size:        Davidson subspace size. If None, auto-tune per
                              subspace from the *free* memory remaining after
                              helpers+integrals are on device — far more
                              accurate than scaling from total device RAM,
                              especially at 100M+ basis sizes where helpers
                              are non-trivial.
            max_davidson_iters: Outer Davidson restart count per diagonalization
                (default ``20``). Total matvec work scales as
                ``max_davidson_iters * davidson_block_size``. Used when a per-call
                ``max_davidson_iters`` override is not supplied.
            davidson_tolerance:         Davidson ‖R‖ convergence threshold.
                              An *absolute* residual, not scaled to the energy.
                              On ``device="mps"`` (float32, no float64 support)
                              a value below ``MPS_DAVIDSON_TOLERANCE_FLOOR``
                              (``1e-5``) is unlikely to ever be reached for
                              typical quantum chemistry energy scales, so it
                              is overridden to that floor (per call, and at
                              construction) with a ``RuntimeWarning`` — not
                              just warned about — so a run on this device
                              stays meaningful without the caller having to
                              scale the tolerance to their own energy.
            davidson_de_tolerance:      Opt-in eigenvalue-stability check.
                              ``None`` (default) disables it: convergence is
                              residual-only (``davidson_tolerance``). Pass a
                              value to additionally require that a root's
                              Ritz value has also stopped changing by more
                              than ``sqrt(davidson_de_tolerance)`` before it
                              is declared converged (pyscf-style
                              two-condition stopping criterion; recommended
                              value: ``davidson_de_tolerance=davidson_tolerance``;
                              see :func:`~torch_sci_solver.torch_impl.davidson.davidson_torch`).
            compute_full_rdm: also build and return spin-summed 1- and 2-RDMs.
            compute_spin_squared:
                also build the αβ 2-RDM block and compute ⟨S²⟩ via the
                `s_squared_from_rdm` formula. Validated against PySCF's
                ``spin_square0`` to machine precision on singlet / triplet /
                quintet small systems. Cost: one Γαβ build per subspace
                (a few seconds on A100 at 50M-config basis). Printed in the
                ``[sci_solver]`` verbose line and exposed on
                ``solver.stats['spin_squared']`` (list, one entry per
                subspace across the whole run).
            verbose:          per-subspace memory + Davidson prints.
            budget_fraction:  fraction of free memory the Davidson subspace is
                              allowed to consume (default 0.80; raise if you
                              want to push closer to the limit, lower if
                              stability matters more than throughput).
            gdb_kwargs:       options that only apply to the GDB (pair) basis,
                              i.e. ``mode="gdb"``. Bundled here rather
                              than flattened into this signature so the
                              basis-specific knobs stay in one place. Unknown
                              keys raise, so a typo is not silently ignored.
                              Passing this under ``mode="tpb"`` also
                              raises. Recognized keys (see
                              :data:`GDB_KWARG_DEFAULTS`):

                              ``budget_bytes`` — peak workspace for one
                              excitation-enumeration chunk. GDB re-derives the
                              excitation rows on every matvec instead of storing
                              them (a stored table is O(nnz(H)); see
                              :mod:`~torch_sci_solver.torch_impl.pair_helpers`),
                              so this sets how much it materializes at a time,
                              and therefore how many chunks — and kernel
                              launches — each matvec costs. ``None`` (default)
                              uses a flat 8192 MB on every device — measured
                              fastest on an A100 across dimensions; a "biggest
                              chunk that fits free VRAM" auto-sizer was tried
                              and was up to 4.7x slower. Lower it if VRAM is
                              tight. Unrelated to ``budget_fraction``, which
                              sizes the persistent Davidson subspace.

                              ``block_matvec`` — ``"auto"`` (default),
                              ``"on"`` or ``"off"``: use the exact
                              Cartesian-block matvec when the pair basis
                              decomposes into disjoint ``α_uids × β_uids``
                              groups. ``"auto"`` applies the size and group
                              gates; ``"on"`` skips them (tests, A/B runs);
                              ``"off"`` always streams the pair-keyed
                              tables. A failed structural check falls back
                              to the streamed tables.

                              ``block_table_bytes`` — optional cap on the
                              block decomposition's byte footprint (the two
                              pair-index maps plus uid lists); the build is
                              gated out when the estimate exceeds it.
                              ``None`` leaves it uncapped.
            seed_strategy:    which determinant to place at index 0 of the
                              basis — i.e., which determinant Davidson seeds
                              its initial vector ``W = [1, 0, ..., 0]`` on.
                              One of:
                                * ``'smallest_int'`` (default) — sort the
                                  basis by integer value and seed from the
                                  lowest-diagonal determinant (``argmin``
                                  of diag H). For closed-shell
                                  chemistry this lands on the HF-like
                                  determinant, which is in the singlet-
                                  ground-state Krylov basin. Empirically
                                  gives clean singlets up through N_BITS=7000
                                  on a 36-orbital benchmark when combined with
                                  αβ-swap projection (auto-enabled for
                                  symmetric bases) and the post-recovery
                                  dedup (always on).
                                * ``'connected'`` — pick the determinant
                                  with the most single/double-excitation
                                  neighbors in the basis. Matches SBD/TPB's
                                  behavior. Mostly an SBD-comparison opt-in
                                  now that we have a better default; on
                                  this benchmark it returns higher-energy
                                  singlets at small basis and occasionally
                                  steers Davidson onto a quintet at
                                  intermediate basis. O(N²) bit-popcount,
                                  sub-second at N=14k; cost goes to
                                  ``stats['t_seed']``.
            apply_ab_projection:
                              When True (default), apply the αβ-swap projection
                              to Davidson on αβ-symmetric bases (auto-detected).
                              Set False to skip it even on symmetric bases.

                              The projection restricts Davidson to the physical
                              α↔β-swap eigenspace of H that contains the closed-
                              shell singlet subspace. Because the matvec stores
                              the wavefunction in the interleaved-Wick basis
                              (see ``rdm.py`` "Convention"), the projector
                              carries a per-(iₐ, i_b) sign — it is
                              ``P(v)[i,j] = (v[i,j] + τ[i,j] · v[j,i]) / 2``
                              with ``τ = σ · σᵀ`` and σ the interleaved-Wick →
                              spin-block sign matrix. This makes Davidson track
                              the true singlet ground state at all basis sizes:
                              at small scale it reproduces PySCF's exact FCI on
                              the closed-shell CI space; at large scale it
                              suppresses spurious triplet contamination from
                              round-off in the chunked matvec, which without
                              the projection can steer Davidson onto a wrong
                              eigenstate. Set ``apply_ab_projection=False``
                              only for SBD/TPB parity comparison (SBD has no
                              equivalent projection) or when you want the bare
                              variational minimum on the *unrestricted* CI
                              basis (which for closed-shell singlets may not
                              be the singlet minimum on truncated subspaces —
                              enforce the singlet with ``spin_sq=0.0`` if you
                              need that instead).

        Returns:
            A callable matching ``qiskit_addon_sqd.fermion.solve_sci_batch``::

                sci_solver(ci_strings, one_body, two_body, norb, nelec, *,
                              spin_sq=None, **kwargs) -> list[SCIResult]

            suitable as the ``sci_solver=...`` argument of
            ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``.

            Notes on the kwargs surface:
              - ``spin_sq=None`` (default) — no penalty. Bit-identical to the
                v0.10.18 code path: no S² operator built, no augmented matvec,
                no extra memory. Use this for the unmodified variational
                answer.
              - ``spin_sq=<float>`` — augmented Davidson solves
                ``H + λ · (S² − spin_sq)`` with λ = 1.0 by default. ``spin_sq``
                is the target value of ⟨S²⟩ (e.g., 0.0 for a singlet, 2.0
                for a triplet, 6.0 for a quintet). The penalty strength λ
                can be overridden by passing ``spin_penalty_lambda=<float>``
                as a kwarg to this call; otherwise λ defaults to 1.0, which
                puts the lowest triplet 2 Ha above the lowest singlet —
                sufficient when the chemistry singlet–triplet gap is sub-eV.
                S² is built via the scatter-based ``SpinSqOperator`` with
                O(M_a · n_a · (L − n_a)) precompute and ~O(L² + N_off) per
                matvec, where N_off is the count of nonzero off-diagonal
                elements of S² in the basis — typically a small multiple
                of N. No N² memory cost.

                The reported energy is the physical ⟨ψ|H|ψ⟩ of the converged
                eigenstate — the penalty contribution ``λ(⟨S²⟩ − target)`` is
                subtracted before returning, so the penalty is invisible to
                the caller. The eigenvector's ⟨S²⟩ is exposed as
                ``solver.stats['spin_sq_final']`` for verification.
              - ``max_davidson_iters`` and ``tol`` are forwarded to the underlying
                Davidson as outer restart count and ‖R‖ threshold respectively.
                (``max_cycle`` is rejected: it was a misleading pyscf-style
                alias for ``max_davidson_iters``, and nothing here calls pyscf.)
              - Other pyscf-only kwargs (``nroots``, ``max_space``, …) are
                silently ignored.

            matvec_impl:      Davidson matvec backend. ``"auto"`` (default)
                              uses ``"batched"`` on CPU and the tiled Triton
                              kernels on a CUDA device. On CUDA without a working
                              Triton it raises (rather than silently running the
                              slower batched path); install Triton or pass
                              ``matvec_impl="batched"`` explicitly to override.
                              Pass ``"triton"`` to require the Triton kernels or
                              ``"batched"`` to force the portable path. The
                              resolved value is stored on ``self.matvec_impl`` and
                              recorded in benchmark metadata. The TPB Triton
                              kernels support real and complex dtypes (complex
                              operands are passed as interleaved real pairs); the
                              GDB pair-block Triton kernel is real-only, so a
                              complex dtype there still requires
                              ``"batched"``.
            mode:             Basis-space mode, ``"tpb"`` (default) or ``"gdb"``.

                              ``"tpb"`` — *tensor product basis*, the historical
                              behaviour. Stores the wavefunction over the
                              Cartesian product ``{(α_i, β_j) : i, j}`` of dim
                              ``M_α · M_β``: the two spin sectors are
                              deduplicated independently and every α-string
                              couples with every β-string.

                              ``"gdb"`` — *general determinant basis*. The caller
                              supplies an explicit list of ``(α_i, β_i)`` pairs
                              and the wavefunction has dim ``N`` = number of
                              distinct pairs. In GDB mode input arrays
                              ``(strings_a, strings_b)`` are treated as parallel
                              length-N lists (index ``i`` defines pair ``i``,
                              duplicates collapsed by ``(α, β)``) rather than as
                              two independent sets crossed via product. Use GDB
                              when the sampler already produced meaningful
                              (α, β) pairings you want to preserve — e.g., raw
                              measurement bitstrings — instead of blowing them up
                              through the product.

                              **GDB is not a drop-in for qiskit_addon_sqd's outer loop.**
                              ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``
                              dedups its bitstring matrix independently per spin
                              (via ``bitstring_matrix_to_ci_strs``) and passes
                              the resulting per-spin-unique arrays to the
                              ``sci_solver`` callable as if forming their
                              Cartesian product. Feeding that flow to a
                              ``mode="gdb"`` solver would silently
                              misinterpret the per-spin-unique arrays as
                              parallel pairs — an entirely different (and
                              usually wrong) subspace. The correct pair-mode
                              SQD workflow is: raw bit_array → extract α/β
                              halves per row → call
                              ``solve_fermion((alpha, beta), mode="gdb")``
                              directly, bypassing
                              ``diagonalize_fermionic_hamiltonian``. If you need
                              an SQD outer loop, wrap your own.
            n_roots:          number of lowest eigenpairs to find per subspace
                              (default 1 — backward-compatible, returns the
                              ground state only). When > 1, Davidson extracts
                              the ``n_roots`` lowest Ritz pairs; each root
                              produces its own occupancies, energy, and
                              ``SCIResult``. All roots are accessible via the
                              return value of ``__call__``: each element of the
                              returned ``list[EigenStates]`` holds ``states[0]``
                              (ground state) through ``states[n_roots-1]``
                              (highest solved root). ``n_roots`` must not exceed
                              ``davidson_block_size`` (auto-tuned or user-supplied).
                              It must also fit in the CI space (or projected
                              sector).
                              Full RDMs, spin diagnostics and warm-start
                              snapshots are computed for root 0 only.
            davidson_dtype:   storage dtype of Davidson's subspace (``C``, ``HC``,
                              the Ritz problem). ``None`` (default) keeps it in
                              ``dtype``. The narrow subspace — ``torch.float32``
                              over a real fp64 ``dtype``, or ``torch.complex64``
                              over complex128 — halves the subspace memory
                              (``C``/``HC`` dominate the peak at 10⁸
                              determinants) while the matvec still applies H in
                              full precision: its first op, ``hii * T``, promotes
                              the narrow input, so only the vector read is
                              narrow. Buys capacity, not speed. Any other
                              combination raises.
            davidson_fp32_shift: only with an fp32 ``davidson_dtype``: Davidson sees
                              ``H − shift·I`` so its fp32 Ritz values are
                              correlation-sized rather than total energies (fp32
                              resolves ~1e-7 relative). ``"hf"`` (default) uses
                              ``min(hii)``, the HF-like diagonal; a float sets it.
                              ``None`` disables it — unsafe: measured to stall at
                              ~1e-5 residual, 0.3 mHa off.
            davidson_fp32_energy: with an fp32 ``davidson_dtype``, or on
                              ``device="mps"`` (where the whole solve is float32):
                              how the returned energy is formed. ``"rq64"``: an fp64
                              Rayleigh quotient of the returned vector, one extra
                              matvec per root, exact. ``"ritz"``: Davidson's fp32
                              Ritz value plus the shift, no extra matvec; its error
                              is that of an fp32-rounded wavefunction, does not
                              shrink with ``tol`` and grows with the basis (3.5e-7
                              Ha at 400 determinants, 1.4e-6 Ha at 14,400 on N2
                              2.2 Å). ``"auto"`` (default) picks ``"ritz"`` when the
                              call's ``tol >= 1e-3`` and ``"rq64"`` otherwise: at
                              that residual the energy is already uncertain by more
                              than the Ritz error (1.8e-5–1.8e-4 Ha on N2 in fp64),
                              so the matvec is saved where it is free, e.g. the SQD
                              warm loop at 3e-3, and kept where it matters. On MPS
                              the fp64 quotient uses CPU copies of the integrals,
                              tables and basis (MPS has no float64), one CPU matvec
                              per root, and also replaces the penalty-stripped
                              energy under ``spin_sq``; ``"ritz"`` keeps the float32
                              Ritz value, biased by about ``|E| eps(float32)``.
            davidson_correction: ``"davidson"`` (default): the diagonal-preconditioned
                              correction ``(theta - D)^-1 r``. ``"olsen"``: Olsen's
                              correction, orthogonal to the Ritz vector (see
                              ``davidson_torch``). Every mode and root count.
            davidson_restart: what single-root Davidson keeps across a restart.
                              ``"recompute"`` (default): the Ritz vector, whose
                              ``H x`` costs one matvec per restart. ``"reuse"``: also
                              its ``H x``, formed from stored products. ``"thick"``:
                              ``"reuse"`` plus the previous Ritz vector (GD+1). Both
                              confirm convergence after a restart with one fresh
                              matvec. Multi-root already restarts from stored products.
            davidson_missed_root_check: ``n_roots > 1`` only: after convergence, search
                              for a root the solve missed and re-solve with it (see
                              ``davidson_torch``). ``"auto"`` (default): on exactly when
                              the solve starts from ``init_vectors``, where it is needed
                              for correctness: a warm multi-root start, even from the
                              exact ground state, can converge without a lower excited
                              root (N2 SQD batch: root 1 off by 0.45 Ha). ``True`` also
                              checks cold solves, which can miss a root at a loose
                              tolerance; ``False`` never checks.
            distributed:      Opt-in distributed-memory solve of each subspace.
                              ``None``/``False`` (default) keeps the existing
                              single-process / batch-parallel ``devices=`` path
                              byte for byte. ``True`` uses all visible CUDA
                              devices (NCCL) when there are at least two, else
                              the launcher's ``WORLD_SIZE``. A dict selects the
                              launch explicitly: ``world_size``, ``backend``
                              (``"gloo"`` or ``"nccl"``), ``device_ids``,
                              ``master_addr``, ``master_port``, ``init_method``
                              and ``timeout`` (seconds). The state vector and
                              the Davidson subspace are partitioned across
                              ranks; ``C``/``HC`` projections and residual norms
                              are all-reduced. Supported: TPB, GDB and
                              spinless bases, real and complex, single- or
                              multi-root. ``matvec`` selects the owner-computes
                              local operator (``"local"``, the default) or the
                              replicated fallback; ``tile`` selects a 2-D grid
                              (``"auto"``, ``"force"`` or ``(p_alpha, p_beta)``),
                              ``exchange_bytes`` bands large αβ corners, and
                              ``hand_back_dir`` names a shared directory for the
                              memmapped hand-back on multi-node runs. Mutually
                              exclusive with ``devices=``.
                              See ``docs/distributed_state_vector.md``.
            unknown_call_kwargs: policy for per-call keywords this engine does
                              not recognize: ``"ignore"`` (default, the
                              historical behaviour), ``"warn"`` (emit a
                              ``UserWarning`` naming them) or ``"error"``
                              (raise ``TypeError``). The recognized set is
                              :data:`CALL_KWARG_NAMES`; note that ``max_cycle``
                              is recognized only to raise its own targeted
                              error, and pyscf-only names such as ``nroots`` are
                              unknown here.
            memory_budget_bytes: optional per-rank byte cap. It bounds the
                              auto-tile policy, the default GDB/spinless
                              enumeration workspace (when
                              ``gdb_kwargs["budget_bytes"]`` is unset) and the
                              automatic Davidson block sizing. ``None``
                              (default) keeps the current free-memory
                              behaviour. A ``distributed={"budget_bytes": ...}``
                              cap is combined with it by taking the smaller.
            table_dtype: index-payload dtype of the TPB excitation tables,
                              ``"auto"`` (default: int64, the reference build),
                              ``"int32"`` (halves the table memory; rejects
                              indices above :data:`INT32_MAX`) or ``"int64"``.
                              The GDB and spinless tables are always int64, and
                              the Triton kernels require int64 payloads.
            compact_eri: ``"auto"`` (default), ``"on"`` or ``"off"`` — whether
                              the spatial ERI is stored in the 16x-smaller
                              compact spin-adapted form. ``"auto"`` keeps the
                              historical heuristic (compact when the Triton
                              kernels do not need the dense tensor and the
                              spin-orbital ERI is at least 64^4). The spinless
                              path has no compact store.
            triton_block: optional Triton launch block (a positive multiple of
                              16), or ``None`` (default) for the kernels' own
                              choice. It drives the distributed TPB tile
                              operator, whose block constant is read per launch;
                              the single-process Triton entry/doubles kernels
                              keep their built-in block.
            diagnostics: optional dict of diagnostic switches:
                              ``dump_dir`` (per-call basis dump directory, same
                              as ``TORCH_SCI_DUMP_DIR``), ``matvec_profile``
                              (enable the ``TORCH_SCI_MATVEC_PROFILE`` phase
                              timers) and ``profile_blocks`` (enable the
                              ``SCIS_PROFILE_BLOCKS`` channel timers). Explicit
                              environment variables are still honoured.
            deterministic: ``True`` runs every call under torch's deterministic
                              algorithms: CUDA ``index_add``/``scatter_add`` take
                              their deterministic implementations, the atomic
                              Triton entry kernels (spinless, streamed GDB) and
                              the cuSPARSE products of the factorized αβ pass
                              fall back to them, and ``CUBLAS_WORKSPACE_CONFIG``
                              defaults to ``":4096:8"`` when CUDA is not yet
                              initialized. A CPU or CUDA run is then bitwise
                              reproducible for a fixed device set, backend, rank
                              count and NCCL algorithm choice (pin ``NCCL_ALGO``
                              across runs); a different rank count still reorders
                              the global reductions, and MPS has no deterministic
                              accumulating ``index_put`` (warned). The switch is
                              process-wide for the duration of a call, so do not
                              run solvers with different settings concurrently in
                              threads of one process. Slower on CUDA; a debugging
                              aid.
            ab_factorization: ``"off"`` (default) or ``"cholesky"``. With
                              ``"cholesky"`` the TPB αβ doubles pass runs in
                              factorized form, ``Σ_P A^α_P C (A^β_P)^T``, from a
                              pivoted Cholesky decomposition of the spatial ERI
                              (computed once per Hamiltonian; no auxiliary basis).
                              Exact to ``ab_factorization_tol``; faster when the
                              ERI is low rank against the singles per string.
                              Real TPB solves only, not with ``distributed=``.
            ab_factorization_tol: largest residual diagonal of the pivoted
                              Cholesky (an indefinite ERI supermatrix, e.g.
                              synthetic integrals, takes a signed
                              eigendecomposition with this cutoff).
        """

        _factory_kwargs = {key: value for key, value in locals().items() if key != "self"}
        # Kept separately: ``_validate_gdb_kwargs`` fills defaults, so afterwards
        # the dict is non-empty even when the caller supplied nothing.
        self._gdb_kwargs_supplied = bool(gdb_kwargs)

        # --- constructor validation (error-only; every accepted value keeps
        # its historical behaviour) ---
        if not (dtype.is_floating_point or dtype.is_complex):
            raise ValueError(f"dtype must be a floating-point or complex torch dtype, got {dtype}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
        if davidson_block_size is not None and int(davidson_block_size) <= 0:
            raise ValueError(
                f"davidson_block_size must be positive or None, got {davidson_block_size!r}"
            )
        if int(max_davidson_iters) <= 0:
            raise ValueError(f"max_davidson_iters must be positive, got {max_davidson_iters!r}")
        if float(davidson_tolerance) <= 0.0:
            raise ValueError(f"davidson_tolerance must be positive, got {davidson_tolerance!r}")
        if davidson_de_tolerance is not None and float(davidson_de_tolerance) <= 0.0:
            raise ValueError(
                f"davidson_de_tolerance must be positive or None, got {davidson_de_tolerance!r}"
            )
        if not 0.0 < float(budget_fraction) <= 1.0:
            raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction!r}")
        if seed_strategy not in ("smallest_int", "connected"):
            raise ValueError(
                f"seed_strategy must be 'smallest_int' or 'connected', got {seed_strategy!r}"
            )
        if unknown_call_kwargs not in UNKNOWN_CALL_KWARGS_POLICIES:
            raise ValueError(
                f"unknown_call_kwargs must be one of {UNKNOWN_CALL_KWARGS_POLICIES}, "
                f"got {unknown_call_kwargs!r}"
            )
        if memory_budget_bytes is not None and int(memory_budget_bytes) <= 0:
            raise ValueError(
                f"memory_budget_bytes must be positive or None, got {memory_budget_bytes!r}"
            )
        if table_dtype not in TABLE_DTYPES:
            raise ValueError(
                f"table_dtype must be one of {tuple(TABLE_DTYPES)}, got {table_dtype!r}"
            )
        if compact_eri not in COMPACT_ERI_MODES:
            raise ValueError(f"compact_eri must be one of {COMPACT_ERI_MODES}, got {compact_eri!r}")
        if triton_block is not None:
            triton_block = int(triton_block)
            if triton_block <= 0 or triton_block % 16:
                raise ValueError(
                    f"triton_block must be a positive multiple of 16, or None, got {triton_block!r}"
                )
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise TypeError(f"diagnostics must be a dict or None, got {type(diagnostics).__name__}")
        diagnostics = dict(diagnostics or {})
        unknown_diagnostics = set(diagnostics) - set(DIAGNOSTIC_KEYS)
        if unknown_diagnostics:
            raise ValueError(
                f"unknown diagnostics key(s) {sorted(unknown_diagnostics)}; "
                f"recognized: {sorted(DIAGNOSTIC_KEYS)}"
            )
        if diagnostics.get("dump_dir") is not None:
            diagnostics["dump_dir"] = str(diagnostics["dump_dir"])
        for flag in ("matvec_profile", "profile_blocks"):
            if flag in diagnostics:
                diagnostics[flag] = bool(diagnostics[flag])
        if not isinstance(deterministic, bool):
            raise TypeError(f"deterministic must be a bool, got {type(deterministic).__name__}")
        if deterministic:
            # cuBLAS GEMMs are reproducible only with a fixed workspace, which it
            # reads when a process creates its first handle; ranks and pool
            # workers set it in their own bootstrap as well.
            if not torch.cuda.is_initialized():
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            elif "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                warnings.warn(
                    "deterministic=True but CUDA is already initialized in this process "
                    "without CUBLAS_WORKSPACE_CONFIG; set it (e.g. ':4096:8') before the "
                    "first CUDA call for reproducible cuBLAS products.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Store factory kwargs as instance attributes (formerly closure vars).
        self.device = torch.device(device) if isinstance(device, str) else device
        # Multi-device users pass ``devices=[cuda:0, cuda:1, ...]`` and typically
        # leave ``device=`` at its default ("cpu"). For ``matvec_impl='auto'`` to
        # correctly detect that this is a CUDA setup, rebind ``self.device`` to
        # the head of the devices list NOW, before the matvec_impl block below
        # — otherwise auto-resolution sees "cpu" and falls back to the ~5x
        # slower ``'batched'`` backend even though every worker will actually
        # run on CUDA. The full devices-list validation still happens further
        # below; this is just an early primary-device hint for matvec_impl
        # resolution.
        if devices is not None and len(devices) > 0:
            _head = devices[0]
            _head_dev = torch.device(_head) if isinstance(_head, str) else _head
            if _head_dev.type == "cuda":
                self.device = _head_dev
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.davidson_block_size = davidson_block_size
        self.max_davidson_iters = max_davidson_iters
        self.davidson_tolerance = _clamp_davidson_tolerance_for_device(
            davidson_tolerance, self.device
        )
        self.davidson_de_tolerance = davidson_de_tolerance
        self.compute_full_rdm = compute_full_rdm
        self.compute_spin_squared = compute_spin_squared
        self.verbose = verbose
        self.budget_fraction = budget_fraction
        if mode not in ("tpb", "gdb"):
            raise ValueError(f'mode must be "tpb" or "gdb", got {mode!r}')
        self.mode = mode
        self.gdb_kwargs = _validate_gdb_kwargs(gdb_kwargs, mode=mode)
        self.seed_strategy = seed_strategy
        self.apply_ab_projection = apply_ab_projection
        self.unknown_call_kwargs = unknown_call_kwargs
        self.memory_budget_bytes = None if memory_budget_bytes is None else int(memory_budget_bytes)
        self.table_dtype = table_dtype
        self._table_dtype_torch = TABLE_DTYPES[table_dtype]
        self.compact_eri = compact_eri
        self.triton_block = triton_block
        self.diagnostics = diagnostics
        self.deterministic = deterministic
        if ab_factorization not in AB_FACTORIZATION_MODES:
            raise ValueError(
                f"ab_factorization must be one of {AB_FACTORIZATION_MODES}, "
                f"got {ab_factorization!r}"
            )
        if not float(ab_factorization_tol) > 0.0:
            raise ValueError(f"ab_factorization_tol must be > 0, got {ab_factorization_tol!r}")
        if ab_factorization != "off":
            if mode != "tpb":
                raise ValueError(
                    "ab_factorization factorizes the TPB αβ doubles pass; mode='gdb' "
                    "keeps its pair kernels. Pass ab_factorization='off'."
                )
            if dtype.is_complex:
                raise ValueError(
                    "ab_factorization supports real integrals only; pass "
                    "ab_factorization='off' for a complex dtype."
                )
        self.ab_factorization = ab_factorization
        self.ab_factorization_tol = float(ab_factorization_tol)
        self._ab_factor_cache: "Optional[tuple[Any, tuple[torch.Tensor, torch.Tensor]]]" = None
        self._cpu_wide_integrals_cache: "Optional[tuple[Any, Any, bool, Any]]" = None
        # Distributed execution options, applied by ``distributed_solver`` for
        # the duration of one distributed call (the frozen context cannot carry
        # them). Defaults keep the non-distributed and size-based behaviour.
        self._runtime_hand_back = "auto"
        self._runtime_hand_back_bytes = None
        self._runtime_budget_bytes = None
        if diagnostics.get("matvec_profile"):
            # ``mult`` snapshots the env var at import time; setting the module
            # flag is the only way to enable the per-phase timers afterwards.
            import torch_sci_solver.torch_impl.mult as _mult_module

            _mult_module._MATVEC_PROFILE_ENABLED = True
        if diagnostics.get("profile_blocks"):
            # Read per call by ``mult_pair_blocks.profile_enabled``.
            os.environ["SCIS_PROFILE_BLOCKS"] = "1"
        if n_roots < 1:
            raise ValueError(f"n_roots must be >= 1, got {n_roots}")
        if davidson_block_size is not None and int(n_roots) > int(davidson_block_size):
            raise ValueError(
                f"n_roots ({n_roots}) cannot exceed davidson_block_size "
                f"({davidson_block_size}); increase davidson_block_size or reduce n_roots."
            )
        self.n_roots = n_roots
        if davidson_dtype is not None and davidson_dtype != dtype:
            # The narrow subspace keeps the full-precision matvec and halves only
            # the stored C/HC: fp32 over fp64, complex64 over complex128. A real
            # fp32 subspace cannot hold the eigenvector of a complex Hermitian H.
            narrow = {torch.float64: torch.float32, torch.complex128: torch.complex64}
            if narrow.get(dtype) != davidson_dtype:
                raise ValueError(
                    f"davidson_dtype={davidson_dtype} with dtype={dtype} is not supported; "
                    "the only mixed combinations are davidson_dtype=torch.float32 with "
                    "dtype=torch.float64, and davidson_dtype=torch.complex64 with "
                    "dtype=torch.complex128 (narrow subspace, full-precision matvec)."
                )
        if davidson_fp32_shift is not None and davidson_fp32_shift != "hf":
            davidson_fp32_shift = float(davidson_fp32_shift)
        if davidson_fp32_energy not in ("auto", "rq64", "ritz"):
            raise ValueError(
                'davidson_fp32_energy must be "auto", "rq64" or "ritz", '
                f"got {davidson_fp32_energy!r}"
            )
        self.davidson_dtype = davidson_dtype
        self.davidson_fp32_shift = davidson_fp32_shift
        self.davidson_fp32_energy = davidson_fp32_energy
        if davidson_correction not in ("davidson", "olsen"):
            raise ValueError(
                f'davidson_correction must be "davidson" or "olsen", got {davidson_correction!r}'
            )
        if davidson_restart not in ("recompute", "reuse", "thick"):
            raise ValueError(
                'davidson_restart must be "recompute", "reuse" or "thick", '
                f"got {davidson_restart!r}"
            )
        self.davidson_correction = davidson_correction
        self.davidson_restart = davidson_restart
        if davidson_missed_root_check not in ("auto", True, False):
            raise ValueError(
                'davidson_missed_root_check must be "auto", True or False, '
                f"got {davidson_missed_root_check!r}"
            )
        self.davidson_missed_root_check = davidson_missed_root_check
        valid_impls = ("auto",) + MATVEC_IMPLS
        if matvec_impl not in valid_impls:
            raise ValueError(f"matvec_impl must be one of {valid_impls}, got {matvec_impl!r}")
        matvec_impl_requested = matvec_impl
        if self.mode == "gdb":
            # The GDB Triton kernels cover the doubles channels only: an
            # explicit ``"triton"`` routes the block mode's same-spin doubles
            # slices and the streamed mode's doubles tables to their kernels,
            # keeping the batched kernels for the singles; ``"auto"`` stays on
            # the fully batched pair path.
            if matvec_impl in ("auto", "batched"):
                matvec_impl = "batched"
        elif matvec_impl == "auto":
            # CPU defaults to the portable batched path. A CUDA device defaults
            # to the fast tiled Triton kernels; if Triton is unavailable we raise
            # rather than silently falling back to batched, so a misconfigured
            # GPU box surfaces loudly instead of running ~6x slower unnoticed.
            # The resolved value is stored and recorded in benchmark metadata.
            # Complex dtypes resolve the same way: the TPB Triton kernels read
            # them as interleaved real pairs, so there is no complex fallback.
            if self.device.type == "cuda":
                try:
                    from torch_sci_solver.torch_impl.mult_triton import _HAVE_TRITON
                except Exception:
                    _HAVE_TRITON = False
                if not _HAVE_TRITON:
                    raise RuntimeError(
                        "matvec_impl='auto' selects the Triton backend on a CUDA "
                        "device, but Triton is not available (failed to import "
                        "triton). Install a working Triton (e.g. `pip install "
                        "triton`) to use the fast kernels, or pass "
                        "matvec_impl='batched' explicitly to use the portable "
                        "backend and avoid this error."
                    )
                matvec_impl = "triton"
            else:
                matvec_impl = "batched"
        self.matvec_impl = matvec_impl
        # The caller's request before resolution: an explicit "triton" that a
        # path cannot honour is an error, while an "auto" that resolved to
        # "triton" may fall back to the batched kernels where they are the
        # only implementation.
        self._matvec_impl_requested = matvec_impl_requested
        if self._table_dtype_torch == torch.int32 and self.matvec_impl == "triton":
            raise ValueError(
                "table_dtype='int32' narrows the TPB excitation tables, but the Triton "
                "kernels read int64 payloads. Pass matvec_impl='batched' or "
                "table_dtype='int64'."
            )
        if self._table_dtype_torch == torch.int32 and self.mode == "gdb":
            raise ValueError(
                "table_dtype='int32' applies to the TPB excitation tables only; the GDB "
                "pair tables are always int64. Drop table_dtype or use mode='tpb'."
            )

        # Multi-GPU: `devices=[...]` opts into thread-pool fan-out across the
        # listed CUDA devices. `None` keeps the historical single-device
        # behaviour with `self.devices == (self.device,)`. `self.device`
        # remains the "primary" — used for verbose formatting, the
        # single-slot caches, and any single-device code paths untouched by
        # this feature.
        if devices is None:
            self.devices: tuple[torch.device, ...] = (self.device,)
        else:
            devs = tuple(torch.device(d) if isinstance(d, str) else d for d in devices)
            if len(devs) == 0:
                raise ValueError("devices=[] is not allowed; pass devices=None for single-device.")
            if len(set(devs)) != len(devs):
                raise ValueError(f"devices contains duplicates: {devs}")
            if len(devs) >= 2:
                for d in devs:
                    if d.type != "cuda":
                        raise ValueError(
                            f"multi-device requires all devices be CUDA (got {d}); "
                            "cross-device threading is only useful when kernels can "
                            "run in parallel on separate GPUs."
                        )
                    if d.index is None:
                        raise ValueError(
                            f"CUDA device must specify an index (got {d}); "
                            "use e.g. 'cuda:0', not 'cuda'."
                        )
                if torch.cuda.is_available():
                    max_idx = torch.cuda.device_count() - 1
                    for d in devs:
                        if d.index > max_idx:
                            raise ValueError(
                                f"device {d} out of range (torch.cuda.device_count()={max_idx + 1})"
                            )
                # Rebind primary to the head of the user-supplied list, so
                # single-slot code paths (verbose, spin_sq_final tracker,
                # etc.) speak about a device the user actually named.
                self.device = devs[0]
            else:
                # Length-1 list is equivalent to device=devs[0], no threading.
                self.device = devs[0]
            self.devices = devs
        if self.device.type == "mps":
            # Checked after ``devices=`` resolution, which can rebind the primary
            # device to MPS.
            if deterministic:
                warnings.warn(
                    "deterministic=True on device='mps': torch has no deterministic MPS "
                    "implementation of accumulating index_put, so MPS runs are not "
                    "bitwise reproducible; use device='cpu' or 'cuda' for that.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if dtype in (torch.float64, torch.complex128):
                raise ValueError(
                    f"dtype={dtype} is not available on device='mps': the MPS framework "
                    "has no float64 or complex128. Pass dtype=torch.float32 (or "
                    "torch.complex64 for a complex Hermitian Hamiltonian) on MPS."
                )
            if matvec_impl_requested == "triton":
                raise ValueError(
                    "matvec_impl='triton' needs a CUDA device; device='mps' runs the "
                    "batched kernels. Pass matvec_impl='auto' or 'batched'."
                )
            self.davidson_tolerance = _clamp_davidson_tolerance_for_device(
                self.davidson_tolerance, self.device
            )

        # Thread-safety knobs (only contended when len(self.devices) >= 2).
        # Both locks are cheap: they guard tiny critical sections
        # (a float add or a single print block per subspace).
        self._stats_lock = threading.Lock()
        self._print_lock = threading.Lock()

        # Cached across sci_solver calls — qiskit's loop passes the same
        # (one_body, two_body) every iteration, so rebuilding the (I0, I1, I2)
        # bundle is pure waste. Cache by array identity.
        #
        # Value tuple layout:
        #   (I0, I1, I2, integrals_by_dev, h1_view, eri_view, fp_or_None)
        # where ``integrals_by_dev`` is a ``dict[torch.device, TorchIntegrals]``.
        # Single-device runs keep exactly one entry; multi-device runs populate
        # one entry per device up-front so the pool workers never race on
        # integrals construction.
        self._integrals_cache: dict = {}

        # Cumulative timing across all sci_solver calls in the lifetime of this
        # solver. Exposed as ``solver.stats`` after construction so the CLI can
        # print a summary after diagonalize_fermionic_hamiltonian returns.
        self.stats = {
            "n_calls": 0,
            "n_subspaces": 0,
            "t_recovery_gap": 0.0,  # CPU; qiskit_addon_sqd's recovery + bookkeeping
            "t_integrals_build": 0.0,  # one-time, after first call this is ~0
            "t_seed": 0.0,  # connected-seed reorder (0 if seed_strategy='smallest_int')
            "t_setup": 0.0,  # helpers + hii + W per subspace
            "t_davidson": 0.0,
            "t_occ": 0.0,
            "t_rdm": 0.0,  # 0 unless compute_full_rdm=True
            "t_spin_sq": 0.0,  # 0 unless compute_spin_squared=True
            "t_total_in_solver": 0.0,
            "spin_squared": [],  # one entry per subspace (compute_spin_squared=True)
            "spin_sq_final": None,  # ⟨S²⟩ of the last subspace's converged eigvec
            # (only populated when spin_sq penalty is active —
            #  computed for free during the penalty-strip step)
            # "partitioned" | "replicated" | "occupancies_only" for the last
            # subspace that built full RDMs (see _rdm_accumulation_path).
            "rdm_accumulation": None,
            "per_call_log": [],  # list of dicts, one per sci_solver call.
            # Each dict contains the per-subspace breakdown
            # so we can see whether t_setup or t_davidson
            # is growing with iteration, GPU memory growth,
            # basis size growth, etc. Inspect with:
            #     for r in solver.stats['per_call_log']:
            #         print(r)
        }
        self._last_exit = None  # scalar attribute (was a [None] cell in the closure form)

        # Opt-in distributed solve. Parse/validate here so a bad config fails at
        # construction, not on the first call. ``_distributed_config is None``
        # (disabled, or world_size <= 1) keeps every existing path untouched.
        from torch_sci_solver.torch_impl.distributed_solver import (
            parse_distributed_config,
        )

        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.distributed = distributed
        self._distributed_config = parse_distributed_config(
            distributed, device=self.device, cuda_device_count=cuda_device_count
        )
        self._distributed_ctx = None
        self._inside_distributed = False
        self._distributed_session = None
        self._distributed_session_finalizer = None
        if self._distributed_config is not None:
            if len(self.devices) > 1:
                raise ValueError(
                    "devices= (batch-parallel multi-GPU) and distributed= are mutually exclusive"
                )
            if self.ab_factorization != "off":
                raise ValueError(
                    "ab_factorization runs in the single-process TPB matvec; the "
                    "distributed owner-computes operators keep the direct αβ kernels. "
                    "Pass ab_factorization='off' with distributed=."
                )
            if self.device.type == "mps":
                # Ranks run on CUDA under NCCL or on CPU under gloo; an MPS
                # request would otherwise be replaced by CPU ranks unannounced.
                raise ValueError(
                    "distributed= runs one rank per CUDA device (NCCL) or CPU ranks (gloo); "
                    "MPS exposes a single device and no collective backend. Drop "
                    "distributed= to solve on device='mps', or pass device='cpu' for gloo ranks."
                )
            # ``davidson_dtype`` (narrow subspace) is supported on the
            # owner-computes TPB path and on the replicated fallback (every rank
            # holds the full ``hii``, so the shift is local). A multi-root
            # missed-root search still needs the owner-computes path and is
            # rejected when the call actually takes the replicated fallback
            # (see ``_run_davidson_distributed``).
            # Complex TPB uses the same local path (the batched kernels are
            # dtype-generic); a small complex basis falls back to the
            # replicated ``davidson_distributed`` path, which carries complex
            # coefficients (only the missed-root search is unavailable there).
            if davidson_correction != "davidson":
                raise NotImplementedError(
                    "distributed=True supports the default Davidson correction only"
                )
            if davidson_restart != "recompute":
                raise NotImplementedError("distributed=True supports the default restart mode only")
        self._factory_kwargs = {
            **_factory_kwargs,
            "triton_block": triton_block,
            "diagnostics": dict(diagnostics),
            "distributed": None,
        }

        # Public, JSON-friendly view of what the caller asked for and what the
        # engine resolved. Workflows should read this instead of private
        # attributes (``_solver.mode``, ``engine.gdb_kwargs``, ...).
        requested = dict(self._factory_kwargs)
        requested["devices"] = devices
        requested["gdb_kwargs"] = self.gdb_kwargs
        requested["distributed"] = distributed
        requested["dtype"] = dtype
        requested["davidson_dtype"] = davidson_dtype
        self._requested_config = _jsonable(requested)
        knobs = self._solver_knob_values()
        self._resolved_config = _jsonable({
            "mode": knobs["mode"],
            "matvec_impl": knobs["matvec_impl"],
            "matvec_impl_requested": self._matvec_impl_requested,
            "dtype": knobs["dtype"],
            "davidson_dtype": knobs["davidson_dtype"]
            if knobs["davidson_dtype"] is not None
            else knobs["dtype"],
            "device": knobs["device"],
            "devices": knobs["devices"],
            "distributed": knobs["distributed"],
            "gdb_kwargs": knobs["gdb_kwargs"],
            "unknown_call_kwargs": knobs["unknown_call_kwargs"],
            # Cached-state type per path (a scalar nelec selects the spinless
            # one at call time); JSON-friendly so workflows can discover the
            # probe contract without constructing a solve.
            "probe_state_types": {
                "tpb": "TorchSCIState",
                "gdb": "TorchPairSCIState",
                "spinless": "TorchSpinlessSCIState",
            },
            "memory_budget_bytes": knobs["memory_budget_bytes"],
            "table_dtype": knobs["table_dtype"],
            "compact_eri": knobs["compact_eri"],
            "triton_block": knobs["triton_block"],
            "diagnostics": knobs["diagnostics"],
            "deterministic": knobs["deterministic"],
            "ab_factorization": knobs["ab_factorization"],
            "ab_factorization_tol": knobs["ab_factorization_tol"],
        })

        # Multi-GPU process pool — created lazily on the first multi-device
        # __call__ and reused across SQD iterations. None for single-device.
        # The pool tracks per-worker integrals-fingerprint caching itself, so
        # the ~13 MB CPU bundle is re-pickled only until all workers hold it.
        self._worker_pool: "Optional[_GpuWorkerPool]" = None

    # Constructor params deliberately NOT shipped to workers: the device is set
    # per-worker, `devices` must be None in a worker (it takes the sequential
    # path), and `distributed` is an execution choice, not a solve knob.
    _WORKER_CFG_EXEMPT = frozenset({"device", "devices", "distributed"})

    def _worker_cfg(self) -> dict:
        """Picklable scalar config to rebuild an equivalent single-device solver
        inside each worker process. Excludes device (set per-worker), devices,
        and anything holding a lock or device tensor.

        Asserts completeness against ``__init__``'s signature. Workers are separate
        processes rebuilt *only* from this dict, so a knob omitted here silently
        reverts to its default in the child — the solve then runs with different
        parameters than the user asked for and still returns plausible numbers. That
        failure mode is invisible to any test that only checks the ground state, so
        we turn it into a loud error at pool-construction time instead.
        """
        cfg = self._worker_cfg_raw()
        expected = {
            name
            for name in inspect.signature(type(self).__init__).parameters
            if name not in ("self",) and name not in self._WORKER_CFG_EXEMPT
        }
        missing = expected - set(cfg)
        if missing:
            raise RuntimeError(
                f"_worker_cfg() is missing solver knob(s) {sorted(missing)}: they "
                "would silently revert to defaults in each GPU worker process. Add "
                "them to _worker_cfg_raw(), or to _WORKER_CFG_EXEMPT if a worker "
                "must not inherit them."
            )
        return cfg

    def _worker_cfg_raw(self) -> dict:
        """The actual knob->value mapping shipped to workers (see _worker_cfg)."""
        knobs = self._solver_knob_values()
        return {
            "dtype": knobs["dtype"],
            "chunk_size": knobs["chunk_size"],
            "davidson_block_size": knobs["davidson_block_size"],
            "max_davidson_iters": knobs["max_davidson_iters"],
            "davidson_tolerance": knobs["davidson_tolerance"],
            "davidson_de_tolerance": knobs["davidson_de_tolerance"],
            "compute_full_rdm": knobs["compute_full_rdm"],
            "compute_spin_squared": knobs["compute_spin_squared"],
            "verbose": knobs["verbose"],
            "budget_fraction": knobs["budget_fraction"],
            "gdb_kwargs": dict(knobs["gdb_kwargs"]) if knobs["mode"] == "gdb" else {},
            "seed_strategy": knobs["seed_strategy"],
            "apply_ab_projection": knobs["apply_ab_projection"],
            "mode": knobs["mode"],
            "matvec_impl": knobs["matvec_impl"],  # already resolved from "auto"
            # Must be forwarded: _solve_one_subspace reads self.n_roots, so a
            # worker rebuilt without it would silently solve for the ground
            # state only and drop every excited root under multi-GPU.
            "n_roots": knobs["n_roots"],
            "davidson_dtype": knobs["davidson_dtype"],
            "davidson_fp32_shift": knobs["davidson_fp32_shift"],
            "davidson_fp32_energy": knobs["davidson_fp32_energy"],
            "davidson_correction": knobs["davidson_correction"],
            "davidson_restart": knobs["davidson_restart"],
            "davidson_missed_root_check": knobs["davidson_missed_root_check"],
            "unknown_call_kwargs": knobs["unknown_call_kwargs"],
            "memory_budget_bytes": knobs["memory_budget_bytes"],
            "table_dtype": knobs["table_dtype"],
            "compact_eri": knobs["compact_eri"],
            "triton_block": knobs["triton_block"],
            "diagnostics": dict(knobs["diagnostics"]),
            "deterministic": knobs["deterministic"],
            "ab_factorization": knobs["ab_factorization"],
            "ab_factorization_tol": knobs["ab_factorization_tol"],
        }

    def _solver_knob_values(self) -> dict[str, Any]:
        """Every constructor knob as a name -> value mapping.

        The single source for the config, worker, checkpoint and probe-state
        views: each selects and transforms the subset it needs, so a knob added
        to ``__init__`` cannot be silently omitted from one view. ``distributed``
        is stored as the parsed ``_distributed_config`` and is the one name that
        differs from its parameter.
        """
        values = {
            name: getattr(self, name)
            for name in inspect.signature(type(self).__init__).parameters
            if name not in ("self", "distributed")
        }
        values["distributed"] = self._distributed_config
        return values

    @property
    def config(self) -> dict[str, Any]:
        """JSON-friendly view of the requested and resolved solver setup.

        ``requested`` is the normalized constructor input (every keyword, with
        ``gdb_kwargs`` defaults filled); ``resolved`` is what the engine will
        actually use: the mode, resolved and requested ``matvec_impl``, dtype
        names, effective device(s), the distributed launch config (or ``None``),
        the per-path ``probe_state_types`` and the new interface knobs.
        ``json.dumps(solver.config)`` works.

        Returns:
            ``{"requested": {...}, "resolved": {...}}``.
        """
        return {
            "requested": _jsonable(self._requested_config),
            "resolved": _jsonable(self._resolved_config),
        }

    def _effective_budget_cap(self, runtime_cap: "Optional[int]" = None) -> "Optional[int]":
        """Effective explicit per-rank byte cap, or ``None`` when unbounded.

        An explicit ``memory_budget_bytes`` and/or a distributed
        ``budget_bytes`` cap wins (the smaller of the two); the rank-consistent
        device estimate applies only when this returns ``None``.

        Args:
            runtime_cap: the call's distributed ``budget_bytes``; defaults to
                the runtime attribute a spawned rank carries.

        Returns:
            The cap in bytes, or ``None``.
        """
        explicit = self.memory_budget_bytes
        distributed_cap = runtime_cap if runtime_cap is not None else self._runtime_budget_bytes
        if distributed_cap is not None:
            explicit = (
                int(distributed_cap) if explicit is None else min(explicit, int(distributed_cap))
            )
        return None if explicit is None else int(explicit)

    def _memory_budget(self, dist_ctx: Any = None) -> "Optional[int]":
        """Per-rank byte budget for auto-tiling and enumeration sizing.

        An explicit ``memory_budget_bytes`` and/or
        ``distributed={"budget_bytes": ...}`` cap wins (the smaller of the two)
        and is identical on every rank by construction. Otherwise the existing
        rank-consistent free-memory estimate is used, reduced across ranks so
        the auto policies cannot disagree.
        """
        explicit = self._effective_budget_cap()
        if explicit is not None:
            return int(explicit)
        if dist_ctx is not None:
            return _rank_consistent_memory_budget(dist_ctx)
        return None

    def _rdm_max_replicated_bytes(self, runtime_budget: "Optional[int]" = None) -> int:
        """Largest full RDM output the replicated accumulator may hold per rank.

        An explicit ``memory_budget_bytes`` and/or ``distributed={"budget_bytes":
        ...}`` cap bounds the replicated ``(3, L, L, L, L)`` accumulator: a full
        tensor larger than the per-rank budget is accumulated partitioned
        instead. Without an explicit cap the module default applies (the
        single-rank, small-basis fallback), and the cap never exceeds it.

        Args:
            runtime_budget: The call's ``distributed["budget_bytes"]``; defaults
                to the solver's attached runtime budget.

        Returns:
            The byte bound passed to the RDM builders' ``should_replicate``.
        """
        from torch_sci_solver.torch_impl.distributed_rdm import (
            DEFAULT_MAX_REPLICATED_BYTES,
        )

        explicit = self._effective_budget_cap(runtime_budget)
        if explicit is None:
            return DEFAULT_MAX_REPLICATED_BYTES
        return max(0, min(int(explicit), DEFAULT_MAX_REPLICATED_BYTES))

    def _preflight_capacity(
        self,
        entry: dict,
        cfg: Any,
        norb: int,
        *,
        kind: str,
        nelec: Any,
        matvec_impl: str,
    ) -> dict[str, Any]:
        """Estimated per-rank memory of one planned subspace, in bytes.

        Gives a workflow the numbers to size a run before launching it: what the
        Davidson block, the matvec work vectors, the ERI, the excitation tables
        and (on the 1-D flat layouts) the gathered input cost on one rank, whether
        that fits the free budget, the block size the solver would pick, and the
        largest per-rank dimension that would still fit at that block.

        Notes
        -----
        With ``N_r`` the per-rank dimension (``⌈N / P⌉`` on the owner-computes
        layouts, ``N`` otherwise), ``s`` the coefficient itemsize and ``s_d`` the
        Davidson itemsize (complex counted as two reals), the terms are
        ``2 b N_r s_d`` (``C``, ``HC``), ``v N_r s`` work vectors, ``N_r s_r`` for
        the real diagonal, the ERI (``L⁴ s``
        compact or ``(2L)⁴ s`` dense as the rank-resolved backend takes it), ``N s``
        for the gathered input of the 1-D GDB/spinless layouts, ``16 N`` for an
        explicit pair list, and for TPB an upper bound on the tables,
        ``M (3 n_s + 5 n_d) w`` per spin with ``n_s ≤ n_e (L - n_e)`` and
        ``n_d ≤ C(n_e, 2) C(L - n_e, 2)`` each capped at ``M - 1``.

        The block is the solver's own rule (:meth:`_block_for_budget`) at the length
        and vector count each solve path sizes with: TPB ``N`` and 4 vectors (+3 with
        the factorized αβ pass) single-process, the rank's rows and 9 on the
        owner-computes operator; GDB the global pair count and 4 on every path;
        spinless ``N`` and 4 single-process, ``N_r`` and ``P + 4`` on the 1-D
        gather. A distributed run then takes ``min(max(b, n_roots + 3), N)``. The
        budget is the device memory left after setup (device free minus the fixed
        terms) capped by ``memory_budget_bytes`` (or the distributed
        ``budget_bytes``); ``davidson_block_basis`` names what set it and
        ``approximate`` flags estimates built on quantities preflight cannot see
        (the cost-weighted TPB row partition, the spinless tile stage buffers, a
        rank's own free memory, the after-setup free). Device memory is read only
        where that creates no device context: CPU, an available MPS device, or a
        CUDA device this process already initialized, on a single-device solver.
        Build-time temporaries, RDM accumulators and the result hand-back are not
        counted.
        """
        from math import comb

        n = int(entry["dimension"])
        world = int(cfg["world_size"]) if cfg is not None else 1
        owner_computes = cfg is not None and entry.get("path") == "local"
        tiled = entry.get("tile_grid") is not None
        n_rank = -(-n // world) if owner_computes else n
        elem = torch.empty((), dtype=self.dtype).element_size()
        sub_elem, work_elem = self._davidson_element_bytes()
        diag_elem = torch.empty((), dtype=self.dtype.to_real()).element_size()
        gathered = 0
        approximate = False
        if kind == "tpb" and owner_computes:
            work_vectors = 9
            pick_n, pick_vectors, cap_to_n = n_rank, 9, False
            approximate = world > 1  # the solve cuts rows by cost, not by count
        elif kind == "tpb":
            work_vectors = 4 + (3 if self.ab_factorization != "off" else 0)
            pick_n, pick_vectors, cap_to_n = n, work_vectors, True
        elif kind == "gdb":
            work_vectors = 4
            gathered = n * elem if owner_computes and not tiled else 0
            pick_n, pick_vectors, cap_to_n = n, 4, True
        elif owner_computes and not tiled:
            gathered = n * elem
            work_vectors = world + 4
            pick_n, pick_vectors, cap_to_n = n_rank, work_vectors, False
        elif owner_computes:
            # The tile operator counts its stage buffers, known only once built;
            # one block column each for the receive buffer and the send payload.
            work_vectors = 6
            pick_n, pick_vectors, cap_to_n = n_rank, work_vectors, False
            approximate = True
        else:
            work_vectors = 4
            pick_n, pick_vectors, cap_to_n = n, 4, True
        n_orb = int(norb)
        if kind == "spinless":
            eri = n_orb**4 * elem
        else:
            compact = _resolve_compact_eri(
                self.compact_eri, matvec_impl=matvec_impl, mode=self.mode, n_spinorb=2 * n_orb
            )
            eri = (n_orb**4 if compact else (2 * n_orb) ** 4) * elem
        tables = None
        if kind == "tpb":
            index_bytes = 4 if self._table_dtype_torch == torch.int32 else 8

            def spin_tables(m: int, n_e: int, rows: int) -> int:
                n_s = min(n_e * (n_orb - n_e), max(m - 1, 0))
                n_d = min(comb(n_e, 2) * comb(n_orb - n_e, 2), max(m - 1, 0))
                return rows * (3 * n_s + 5 * n_d) * index_bytes + 2 * (m + 1) * 8

            m_a, m_b = int(entry["n_alpha"]), int(entry["n_beta"])
            alpha_rows = -(-m_a // world) if owner_computes else m_a
            tables = spin_tables(m_a, int(nelec[0]), alpha_rows) + spin_tables(
                m_b, int(nelec[1]), m_b
            )
        pair_list = 16 * n if entry.get("layout") == "pairs" else 0
        cap = self._effective_budget_cap(None if cfg is None else cfg.get("budget_bytes"))
        device_free = None
        dev = self.device
        readable = (
            dev.type == "cpu"
            or (dev.type == "mps" and torch.backends.mps.is_available())
            or (dev.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_initialized())
        )
        if cfg is None and len(self.devices) == 1 and readable:
            device_free = self._free_bytes(dev)
        fixed = eri + (tables or 0) + pair_list + gathered + diag_elem * n_rank
        budget, basis = None, None
        if device_free is not None:
            budget, basis = max(device_free - fixed, 0), "device_free_estimate"
        if cap is not None and (budget is None or cap <= budget):
            budget, basis = cap, "budget_cap"
        block = None
        if self.davidson_block_size is not None:
            block, basis = int(self.davidson_block_size), "explicit"
        elif budget is not None:
            block = self._block_for_budget(
                budget, pick_n, n_roots=self.n_roots, matvec_vectors=pick_vectors, cap_to_n=cap_to_n
            )
        if block is not None and cfg is not None:
            block = min(max(block, self.n_roots + 3), n)
        if basis == "device_free_estimate" or (cfg is not None and basis == "budget_cap"):
            approximate = True
        work = work_vectors * n_rank * work_elem
        davidson = None if block is None else 2 * block * n_rank * sub_elem
        total = None if davidson is None else fixed + work + davidson
        fits = None
        if total is not None and (device_free is not None or cap is not None):
            fits = (device_free is None or total <= device_free) and (
                cap is None or davidson + work <= cap
            )
        max_dimension = None
        if block is not None and (device_free is not None or cap is not None):
            per_element = 2 * block * sub_elem + work_vectors * work_elem
            limits = []
            if device_free is not None:
                room = max(0, device_free - eri - (tables or 0)) * self.budget_fraction
                per_rank_fixed = diag_elem + (world * elem if gathered else 0)
                limits.append(int(room) // (per_element + per_rank_fixed))
            if cap is not None:
                limits.append(int(cap * self.budget_fraction) // per_element)
            max_dimension = int(min(limits))
        return {
            "per_rank_bytes": {
                "davidson": davidson,
                "matvec_work": work,
                "diagonal": diag_elem * n_rank,
                "eri": eri,
                "tables_upper_bound": tables,
                "pair_list": pair_list,
                "gathered_input": gathered,
            },
            "total_bytes": total,
            "device_free_bytes": device_free,
            "budget_cap_bytes": cap,
            "fits": fits,
            "davidson_block": block,
            "davidson_block_basis": basis,
            "approximate": approximate,
            "dimension_per_rank": n_rank,
            "max_dimension_per_rank": max_dimension,
        }

    def _rdm_accumulation_path(
        self,
        *,
        n_full: int,
        can_partition: bool,
        world_size: int,
        dtype: torch.dtype,
        runtime_budget: "Optional[int]" = None,
    ) -> str:
        """Choose how this subspace's full 1-/2-RDM is accumulated.

        ``"partitioned"`` keeps only each rank's output slice and gathers the
        full tensors to rank 0; ``"replicated"`` is the existing replicated
        builder/full-vector fallback, used whenever the layout cannot partition
        or the full output fits the per-rank replication bound;
        ``"occupancies_only"`` means no full RDM was requested.

        Args:
            n_full: Elements of the full output (``3 L^4`` for the spinful
                paths, ``nmo^4`` for the spinless one).
            can_partition: Whether the layout has an output-partitioned
                builder for this solve.
            world_size: Ranks in the reduction group.
            dtype: Output dtype used for the byte estimate.
            runtime_budget: Optional distributed ``budget_bytes`` override.
        """
        from torch_sci_solver.torch_impl.distributed_rdm import should_replicate

        if not self.compute_full_rdm:
            return "occupancies_only"
        if not can_partition:
            return "replicated"
        if should_replicate(
            int(n_full),
            dtype,
            world_size=int(world_size),
            max_replicated_bytes=self._rdm_max_replicated_bytes(runtime_budget),
        ):
            return "replicated"
        return "partitioned"

    def _rdm_plan_entry(self, *, n_full: int, can_partition: bool, cfg: Any) -> str:
        """The planned RDM accumulation path for one subspace.

        Args:
            n_full: elements in the full output the accumulation would build.
            can_partition: the subspace runs on a layout the partitioned
                accumulators can use.
            cfg: the distributed launch config, or ``None``.

        Returns:
            The accumulation path recorded in the plan.
        """
        return self._rdm_accumulation_path(
            n_full=int(n_full),
            can_partition=can_partition,
            world_size=int(cfg["world_size"]) if cfg is not None else 1,
            dtype=self.dtype,
            runtime_budget=None if cfg is None else cfg.get("budget_bytes"),
        )

    def _filter_unknown_call_kwargs(self, kwargs: dict) -> dict:
        """Apply :attr:`unknown_call_kwargs` to one call's keyword arguments.

        Returns the kwargs with unknown names removed for ``"ignore"`` and
        ``"warn"`` (after warning), and raises ``TypeError`` for ``"error"``.
        ``max_cycle`` is in :data:`CALL_KWARG_NAMES` so its own targeted
        rejection still runs.
        """
        unknown = sorted(set(kwargs) - CALL_KWARG_NAMES)
        if not unknown:
            return kwargs
        policy = self.unknown_call_kwargs
        if policy == "error":
            raise TypeError(
                f"unknown per-call keyword argument(s) {unknown}; recognized: "
                f"{sorted(CALL_KWARG_NAMES)}. Pass unknown_call_kwargs='ignore' "
                "to drop them silently or 'warn' to warn."
            )
        if policy == "warn":
            warnings.warn(
                f"ignoring unknown per-call keyword argument(s) {unknown}; "
                f"recognized: {sorted(CALL_KWARG_NAMES)}.",
                UserWarning,
                stacklevel=3,
            )
        return {key: value for key, value in kwargs.items() if key in CALL_KWARG_NAMES}

    def capabilities(self) -> dict[str, Any]:
        """Per-mode support map for discovery and workflow gating.

        The result is JSON-friendly and answers what a mode/basis supports
        without constructing a solve: layouts, dtypes, spinless and multi-root
        support, warm start / ``init_vectors`` (including the cached state type
        and whether HF augmentation applies), full RDMs, ``<S^2>``, the spin
        penalty, distributed matvec/tile layouts, the GDB block
        matvec / structural basis / move cache, Triton pass coverage, the
        accepted per-call keywords and the unknown-keyword policy. Unsupported
        physics is reported with a reason, not just a ``False``: the spinless
        entry carries ``spin_squared_unsupported_reason`` and
        ``spin_penalty_unsupported_reason`` because ``S^2`` needs an alpha/beta
        partition that a flat spin-orbital basis does not have.

        Returns:
            A dict with one entry per mode under ``"modes"`` plus a
            ``"spinless"`` entry (a call-time layout selected from a scalar
            ``nelec``, not a ``mode=`` value), ``"accepted_call_kwargs"`` and
            the configured ``"unknown_call_kwargs"`` policy.
        """
        from torch_sci_solver.torch_impl.mult_entries_triton import (
            TRITON_ENTRY_PASS_BACKENDS,
        )
        from torch_sci_solver.torch_impl.mult_triton import TRITON_PASS_BACKENDS
        from torch_sci_solver.torch_impl.probe_spinless import (
            SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION,
            SPINLESS_SPIN_SQ_LIMITATION,
        )

        triton_passes = dict(TRITON_PASS_BACKENDS)
        gdb_triton_passes = {
            key.removeprefix("gdb_streamed_"): value
            for key, value in TRITON_ENTRY_PASS_BACKENDS.items()
            if key.startswith("gdb_streamed_")
        }
        spinless_triton_passes = {
            key.removeprefix("spinless_"): value
            for key, value in TRITON_ENTRY_PASS_BACKENDS.items()
            if key.startswith("spinless_")
        }
        dtypes = [
            "torch.float64",
            "torch.float32",
            "torch.complex128",
            "torch.complex64",
        ]
        distributed_common = {
            "distributed_matvec": ["local", "replicated"],
            "distributed_tile": True,
        }
        modes = {
            "tpb": {
                "layouts": ["product"],
                "dtypes": dtypes,
                "spinless": True,
                "multi_root": True,
                "warm_start": True,
                "probe_state_type": "TorchSCIState",
                "warm_start_hf_augmentation": True,
                "init_vectors": True,
                "full_rdm": True,
                "spin_squared": True,
                "spin_penalty": True,
                **distributed_common,
                "gdb_block_matvec": False,
                "gdb_structural": False,
                "gdb_move_cache": False,
                "triton_passes": triton_passes,
                "table_dtype": ["torch.int64", "torch.int32"],
            },
            "gdb": {
                "layouts": ["pairs", "structural"],
                "dtypes": dtypes,
                "spinless": True,
                # A scalar nelec selects the spinless path, where the GDB
                # options have no effect, so gdb_kwargs + scalar nelec raises.
                "spinless_with_gdb_kwargs": False,
                "multi_root": True,
                "warm_start": True,
                "probe_state_type": "TorchPairSCIState",
                "warm_start_hf_augmentation": True,
                "init_vectors": True,
                "full_rdm": True,
                "spin_squared": True,
                "spin_penalty": True,
                **distributed_common,
                "gdb_block_matvec": True,
                "gdb_structural": True,
                "gdb_move_cache": True,
                "triton_passes": gdb_triton_passes,
                "table_dtype": ["torch.int64"],
            },
            # ``mode=`` never takes "spinless"; a scalar ``nelec`` selects it.
            # Kept in the same map so a workflow can gate on it without
            # knowing the dispatch rule.
            "spinless": {
                "layouts": ["flat_determinants", "tiled_half_strings"],
                # Real spin-orbital integrals take a real dtype; a complex
                # dtype is needed only when they carry imaginary parts.
                "dtypes": dtypes,
                "spinless": True,
                "multi_root": True,
                "warm_start": True,
                "probe_state_type": "TorchSpinlessSCIState",
                # No alpha/beta split means no HF alpha/beta strings to add:
                # the cached state is the ground state of the caller's basis.
                "warm_start_hf_augmentation": False,
                "init_vectors": True,
                "full_rdm": True,
                # S^2 is built from the alpha/beta partition; a flat
                # spin-orbital basis has none, so both the diagnostic and the
                # penalty are physics limitations, not scope cuts.
                "spin_squared": False,
                "spin_squared_unsupported_reason": SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION,
                "spin_penalty": False,
                "spin_penalty_unsupported_reason": SPINLESS_SPIN_SQ_LIMITATION,
                "distributed_matvec": ["local"],
                "distributed_tile": True,
                "gdb_block_matvec": False,
                "gdb_structural": False,
                "gdb_move_cache": False,
                "triton_passes": spinless_triton_passes,
                "table_dtype": ["torch.int64"],
            },
        }
        matvec_impls = ["auto", *MATVEC_IMPLS]
        if self.device.type == "mps":
            # MPS has no float64/complex128, no Triton and no distributed ranks
            # (the constructor rejects each), and the structural GDB basis runs
            # only on the distributed owner-computes path.
            matvec_impls = [impl for impl in matvec_impls if impl != "triton"]
            triton_passes = {}
            for entry in modes.values():
                entry["dtypes"] = [d for d in entry["dtypes"] if d in MPS_DTYPE_NAMES]
                # The structural basis and the spinless half-string tiles exist
                # only on the distributed path.
                entry["layouts"] = [
                    layout
                    for layout in entry["layouts"]
                    if layout not in ("structural", "tiled_half_strings")
                ]
                entry["distributed_matvec"] = []
                entry["distributed_tile"] = False
                entry["gdb_structural"] = False
                entry["triton_passes"] = {}
        return {
            "modes": modes,
            "accepted_call_kwargs": sorted(CALL_KWARG_NAMES - {"max_cycle"}),
            "rejected_call_kwargs": ["max_cycle"],
            "unknown_call_kwargs": self.unknown_call_kwargs,
            "unknown_call_kwarg_policies": list(UNKNOWN_CALL_KWARGS_POLICIES),
            "matvec_impls": matvec_impls,
            "triton_pass_backends": triton_passes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "table_dtype": self.table_dtype,
            "compact_eri": self.compact_eri,
            "triton_block": self.triton_block,
            "triton_block_applies_to": ["distributed_tpb_tile"],
            "deterministic": self.deterministic,
            "ab_factorization": self.ab_factorization,
            "ab_factorization_modes": {
                "tpb": list(AB_FACTORIZATION_MODES),
                "gdb": ["off"],
                "spinless": ["off"],
            },
        }

    @staticmethod
    def _infer_norb(ci_strings: Any, nelec: Any) -> int:
        """Infer ``norb`` from the highest set bit when the caller omitted it.

        Only valid for single-word CI strings; a limb matrix is ambiguous in
        its padding, so the caller must pass ``norb=`` explicitly.
        """
        arrays = (
            list(ci_strings)
            if isinstance(nelec, (int, np.integer))
            else [side for pair in ci_strings for side in pair]
        )
        highest = 0
        for values in arrays:
            arr = np.asarray(values)
            if arr.ndim == 2:
                raise ValueError(
                    "preflight cannot infer norb from multi-limb CI strings; pass norb=."
                )
            if arr.size and arr.dtype.kind in "iu" and arr.dtype.itemsize == 8:
                # Unsigned view: a signed int64 pattern with bit 63 set is negative.
                highest = max(highest, int(arr.ravel().view(np.uint64).max()).bit_length())
            elif arr.size:
                # Python ints of any width; a negative one is a signed int64 pattern.
                words = (int(v) % (1 << 64) if int(v) < 0 else int(v) for v in arr.ravel())
                highest = max(highest, max(words).bit_length())
        if highest <= 0:
            raise ValueError("preflight could not infer norb from empty CI strings; pass norb=.")
        return highest

    def _preflight_distributed(self) -> tuple[Any, str]:
        """Resolved ``(distributed config, matvec_impl)`` without launching ranks."""
        cfg = self._distributed_config
        if cfg is None:
            return None, self.matvec_impl
        from torch_sci_solver.torch_impl.distributed_solver import (
            _rank_device,
            _worker_matvec_impl,
        )

        dev = _rank_device(cfg, 0)
        matvec = _worker_matvec_impl(
            self._matvec_impl_requested,
            mode=self.mode,
            dev=dev,
            backend=cfg["backend"],
            table_dtype=self.table_dtype,
        )
        return cfg, matvec

    def _distributed_checkpoint_settings(self) -> "Optional[dict]":
        """Resolved checkpoint options for this rank, or ``None`` when disabled.

        A spawned worker rebuilds its solver with ``distributed=None``, so the
        settings come from the runtime attributes attached by
        ``distributed_solver._apply_runtime_config``; the parent (and an
        in-process torchrun rank) reads the parsed config directly.
        """
        runtime = getattr(self, "_runtime_checkpoint", None)
        if runtime is not None:
            return runtime
        cfg = self._distributed_config
        if cfg is None or cfg.get("checkpoint_dir") is None:
            return None
        return {
            "dir": cfg["checkpoint_dir"],
            "every": int(cfg["checkpoint_every"]),
            "resume": str(cfg["checkpoint_resume"]),
        }

    def _checkpoint_identity_knobs(self) -> dict[str, Any]:
        """Solver knobs that change the Davidson iterate a checkpoint holds."""
        knobs = self._solver_knob_values()
        return {
            "apply_ab_projection": bool(knobs["apply_ab_projection"]),
            "davidson_correction": str(knobs["davidson_correction"]),
            "davidson_restart": str(knobs["davidson_restart"]),
            # A narrow subspace saves the vectors of ``H - shift``; resuming
            # with another shift would misreport the final energy.
            "davidson_fp32_shift": knobs["davidson_fp32_shift"],
            "davidson_fp32_energy": knobs["davidson_fp32_energy"],
        }

    def _make_davidson_checkpoint(
        self,
        *,
        sub_idx: int,
        context: Any,
        basis_kind: str,
        basis_arrays: tuple,
        n_alpha: int,
        n_beta: int,
        particle_sector: tuple[int, int],
        dimension: int,
        norb: int,
        partition: Any,
        layout_kind: str,
        chosen_nb: int,
        spin_sq_target: "Optional[float]",
        spin_penalty_lambda: float,
        get_integrals_fingerprint: Callable[[], str],
        solver_knob_snapshot: dict,
    ) -> "Optional[Any]":
        """Build this subspace's Davidson checkpoint hook (or ``None``).

        The metadata is the manifest identity: a resumed run must match the
        integrals, basis, particle sector, world size, rank layout, dtype,
        block size, root count, spin penalty and iterate-affecting solver
        knobs. Tolerances and iteration budgets are deliberately excluded so a
        resumed job may run with a larger budget; they are still recorded in
        the manifest's ``extra`` block.

        Args:
            sub_idx: CI subspace index (the ``subspace_<k>`` directory).
            context: This rank's :class:`DistributedContext`.
            basis_kind: ``"product"``, ``"pairs"`` or ``"structural"``.
            basis_arrays: Replicated integer basis arrays to hash.
            n_alpha, n_beta: Basis dimensions along the two spin axes (or the
                unique α/β counts for an explicit pair basis).
            particle_sector: Physical ``(n_alpha, n_beta)`` particle counts.
            dimension: Global CI dimension of this subspace.
            norb: Number of spatial (or spin) orbitals.
            partition: Rank partition of the solve, or ``None`` for the
                canonical contiguous split.
            layout_kind: ``"local"`` (owner-computes) or ``"replicated"``.
            chosen_nb: Davidson block size.
            spin_sq_target, spin_penalty_lambda: Spin penalty parameters; a
                change alters the operator and invalidates the checkpoint.
            get_integrals_fingerprint: Lazily computes the Hamiltonian hash.
            solver_knob_snapshot: Full solver-knob record for the manifest.

        Returns:
            The hook, or ``None`` when checkpointing is disabled or the call is
            not distributed.
        """
        settings = self._distributed_checkpoint_settings()
        if settings is None or context is None:
            return None
        from torch_sci_solver.torch_impl.distributed_solver import (
            checkpoint_basis_hash,
            checkpoint_partition_identity,
            make_checkpoint_hook,
        )

        metadata = {
            "mode": self.mode,
            "basis_kind": basis_kind,
            "basis_hash": checkpoint_basis_hash(*basis_arrays),
            "n_alpha": int(n_alpha),
            "n_beta": int(n_beta),
            "dimension": int(dimension),
            "norb": int(norb),
            "particle_sector": [int(particle_sector[0]), int(particle_sector[1])],
            "world_size": int(context.world_size),
            "backend": str(context.backend),
            "matvec": str(context.matvec),
            "tile_spec": context.tile,
            "layout_kind": layout_kind,
            "partition": checkpoint_partition_identity(partition, context.world_size),
            "dtype": str(self.dtype),
            "davidson_dtype": str(self.davidson_dtype or self.dtype),
            "block_size": int(chosen_nb),
            "n_roots": int(self.n_roots),
            "spin_sq_target": spin_sq_target,
            "spin_penalty_lambda": float(spin_penalty_lambda),
            "integrals_fingerprint": get_integrals_fingerprint(),
            "solver_knobs": self._checkpoint_identity_knobs(),
        }
        return make_checkpoint_hook(
            settings,
            subspace=sub_idx,
            metadata=metadata,
            context=context,
            stats=self.stats,
            extra={"solver_knob_snapshot": dict(solver_knob_snapshot)},
        )

    def _preflight_checkpoint(
        self,
        cfg: Any,
        subspaces: list[dict[str, Any]],
        *,
        norb: int,
        nelec: tuple[int, int],
        spin_penalty_lambda: float = 1.0,
        one_body_tensor: Any = None,
        two_body_tensor: Any = None,
    ) -> dict[str, Any]:
        """Report the checkpoint each subspace would resume from.

        Only fields the preflight plan can know exactly are compared (the
        basis/layout hash and the resolved block size need the real solve).
        The solve re-checks the full manifest identity before loading any
        state.
        """
        settings = self._distributed_checkpoint_settings()
        if settings is None or cfg is None:
            return {
                "enabled": False,
                "dir": None,
                "every": None,
                "resume": None,
                "subspaces": [],
            }
        from torch_sci_solver.torch_impl.distributed_solver import (
            CHECKPOINT_FORMAT_VERSION,
            inspect_checkpoint,
        )

        fingerprint = None
        if one_body_tensor is not None and two_body_tensor is not None:
            fingerprint = _integrals_fingerprint(
                _as_integrals_array(one_body_tensor),
                _as_integrals_array(two_body_tensor),
                0.0,
            )
        n_a, n_b = nelec
        entries = []
        for subspace in subspaces:
            metadata = {
                "mode": self.mode,
                "world_size": int(cfg["world_size"]),
                "backend": str(cfg["backend"]),
                "dtype": str(self.dtype),
                "davidson_dtype": str(self.davidson_dtype or self.dtype),
                "n_roots": int(self.n_roots),
                "basis_kind": subspace.get("layout"),
                "dimension": int(subspace["dimension"]),
                "n_alpha": int(subspace["n_alpha"]),
                "n_beta": int(subspace["n_beta"]),
                "norb": int(norb),
                "particle_sector": [int(n_a), int(n_b)],
                "spin_sq_target": subspace.get("spin_sq"),
                "spin_penalty_lambda": float(spin_penalty_lambda),
                "solver_knobs": self._checkpoint_identity_knobs(),
            }
            if fingerprint is not None:
                metadata["integrals_fingerprint"] = fingerprint
            state = inspect_checkpoint(settings["dir"], subspace["sub_idx"], metadata)
            would_resume = None
            if state["manifest"] and settings["resume"] != "off":
                would_resume = bool(state["matches"])
            elif settings["resume"] == "off":
                would_resume = False
            entries.append({
                "sub_idx": int(subspace["sub_idx"]),
                "would_resume": would_resume,
                **state,
            })
        return {
            "enabled": True,
            "dir": settings["dir"],
            "every": int(settings["every"]),
            "resume": settings["resume"],
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "subspaces": entries,
        }

    def _pop_davidson_overrides(self, kwargs: dict, *, validate: bool) -> dict[str, Any]:
        """Pop the per-call Davidson overrides and the penalty strength.

        Shared by :meth:`_preflight_parse_call` and :meth:`_call_body`.

        Args:
            kwargs: the mutable per-call keyword dict; recognized keys are popped.
            validate: reject non-positive ``max_davidson_iters`` / ``tol`` /
                ``de_tol``. Preflight passes ``True``; the solve keeps its
                historical unchecked parsing.

        Returns:
            The parsed ``call_max_iter``, ``call_eps``, ``call_de_tol`` and
            ``call_spin_lambda`` values.
        """
        if "max_cycle" in kwargs:
            raise TypeError(
                "max_cycle is no longer accepted; it was a misleading pyscf-style "
                "alias (nothing here calls pyscf) for the Davidson outer restart "
                "count. Pass max_davidson_iters instead — note it is a restart count, "
                "not a flat matvec budget: total work ~ max_davidson_iters * davidson_block_size."
            )
        call_max_iter = int(kwargs.pop("max_davidson_iters", self.max_davidson_iters))
        if validate and call_max_iter <= 0:
            raise ValueError(f"max_davidson_iters must be positive, got {call_max_iter}")
        call_eps = _clamp_davidson_tolerance_for_device(
            float(kwargs.pop("tol", self.davidson_tolerance)), self.device
        )
        if validate and call_eps <= 0.0:
            raise ValueError(f"tol must be positive, got {call_eps!r}")
        de_raw = kwargs.pop("de_tol", self.davidson_de_tolerance)
        call_de_tol = None if de_raw is None else float(de_raw)
        if validate and call_de_tol is not None and call_de_tol <= 0.0:
            raise ValueError(f"de_tol must be positive or None, got {call_de_tol!r}")
        call_spin_lambda = float(kwargs.pop("spin_penalty_lambda", 1.0))
        return {
            "call_max_iter": call_max_iter,
            "call_eps": call_eps,
            "call_de_tol": call_de_tol,
            "call_spin_lambda": call_spin_lambda,
        }

    def _normalize_start_kwargs(self, kwargs: dict, ci_strings: Any) -> dict[str, Any]:
        """Pop and normalize the probe/``init_vectors`` start-vector keywords.

        Shared by :meth:`_preflight_parse_call` and :meth:`_call_body`; the two
        call sites previously carried byte-identical copies.

        Args:
            kwargs: the mutable per-call keyword dict; recognized keys are popped.
            ci_strings: the per-subspace string arrays, used for length checks.

        Returns:
            The validated ``probe_method``, ``probe_kwargs`` and ``probe_state``
            values plus the per-subspace ``probe_list`` and ``init_list``.
        """
        probe_method = kwargs.pop("probe_method", None)
        probe_kwargs = kwargs.pop("probe_kwargs", None) or {}
        probe_state = kwargs.pop("probe_state", None)
        init_vectors = kwargs.pop("init_vectors", None)
        if init_vectors is not None and probe_state is not None:
            raise ValueError(
                "init_vectors and probe_state are mutually exclusive: both set the "
                "Davidson start vector. Pass one or the other."
            )
        init_list = self._init_vector_list(init_vectors, len(ci_strings))
        if probe_method is not None and probe_method != "warm_start":
            raise ValueError(
                f"probe_method must be None or 'warm_start', got {probe_method!r}. "
                "(Other probe methods are not yet implemented.)"
            )
        if probe_state is not None:
            if isinstance(probe_state, TorchSCIState):
                probe_list: List[Optional[TorchSCIState]] = [probe_state]
            else:
                probe_list = list(probe_state)
            if len(probe_list) != len(ci_strings):
                raise ValueError(
                    f"probe_state has length {len(probe_list)} but ci_strings has "
                    f"{len(ci_strings)} subspace(s)."
                )
        else:
            probe_list = [None] * len(ci_strings)
        return {
            "probe_method": probe_method,
            "probe_kwargs": probe_kwargs,
            "probe_state": probe_state,
            "probe_list": probe_list,
            "init_list": init_list,
        }

    def _preflight_parse_call(self, ci_strings: Any, kwargs: dict) -> dict[str, Any]:
        """Run the shared per-call keyword gates and return normalized values."""
        davidson = self._pop_davidson_overrides(kwargs, validate=True)
        start = self._normalize_start_kwargs(kwargs, ci_strings)
        return {**davidson, **start}

    def _structural_pair_spec_and_gates(
        self,
        sub_idx: int,
        *,
        spin_sq: Any,
        tile_spec: Any,
        matvec: "Optional[str]",
    ) -> Any:
        """Return this subspace's structural block spec after the GDB gates.

        Args:
            sub_idx: subspace index.
            spin_sq: the call's spin target; the penalty needs an explicit grid.
            tile_spec: the resolved distributed tile spec, or ``None``.
            matvec: the resolved distributed matvec mode, or ``None`` when the
                call is not distributed.

        Returns:
            The structural spec for this subspace, or ``None`` for an explicit
            pair basis.

        Raises:
            ValueError: the spec list is too short, or a requested feature
                cannot run on a structural basis.
        """
        structural_specs = self.gdb_kwargs.get("structural_blocks")
        structural_spec = None
        if structural_specs is not None:
            if sub_idx >= len(structural_specs):
                raise ValueError(
                    "gdb_kwargs['structural_blocks'] must hold one entry per CI "
                    f"subspace; there is no entry for subspace {sub_idx}"
                )
            structural_spec = structural_specs[sub_idx]
        if structural_spec is None:
            return None
        if self.compute_full_rdm or self.compute_spin_squared or spin_sq is not None:
            if tile_spec is None or tile_spec == "auto":
                raise ValueError(
                    "full RDMs, the <S^2> diagnostic and the spin_sq penalty "
                    "on a structural pair basis need an explicit distributed "
                    "tile grid: the block-walked accumulator reads each ket "
                    "from the operator's staged exchange. Pass "
                    "distributed={'tile': 'force'} or a (p_alpha, p_beta) grid."
                )
        if self.seed_strategy != "smallest_int":
            raise ValueError(
                f"seed_strategy={self.seed_strategy!r} needs the streamed "
                "excitation tables; a structural pair basis supports only "
                "seed_strategy='smallest_int'."
            )
        if matvec != "local":
            raise ValueError(
                "a structural pair basis supports only the distributed "
                "owner-computes GDB path (distributed=..., matvec='local'). "
                "Use the explicit GDB basis for single-process solves."
            )
        return structural_spec

    def _pair_tile_grid(
        self,
        *,
        cfg: Any,
        tile_spec: Any,
        n: int,
        block_cap: int,
        budget: int,
        pair_helpers: Any,
        postprocess: bool,
        postprocess_fallback: bool,
    ) -> tuple[Any, Any, Any]:
        """Resolve the pair-basis tile spec, grid and source.

        Args:
            cfg: the distributed launch config.
            tile_spec: the call's tile request.
            n: pair-basis dimension.
            block_cap: Davidson block cap.
            budget: per-rank byte budget.
            pair_helpers: the pair helpers the operator will use.
            postprocess: a full RDM / ``<S^2>`` / penalty request.
            postprocess_fallback: allow the pairs-only ``(world_size, 1)``
                fallback when the auto grid is declined.

        Returns:
            ``(tile_spec, grid, source)``.
        """
        from torch_sci_solver.torch_impl.distributed_davidson import resolve_tile_grid
        from torch_sci_solver.torch_impl.pair_tiles import (
            pair_replicated_bytes,
            resolve_auto_pair_tiling,
        )

        grid = None
        source = tile_spec
        if tile_spec == "auto":
            if pair_helpers.blocks is not None and resolve_auto_pair_tiling(
                world_size=cfg["world_size"],
                n_pairs=n,
                block_size=block_cap,
                itemsize=int(self.dtype.itemsize),
                budget_bytes=budget,
                replicated_bytes=pair_replicated_bytes(pair_helpers),
            ):
                grid = resolve_tile_grid(
                    cfg["world_size"], pair_helpers.m_alpha, pair_helpers.m_beta
                )
            else:
                tile_spec = None
                source = None
                if (
                    postprocess_fallback
                    and cfg["matvec"] == "local"
                    and pair_helpers.blocks is not None
                    and postprocess
                ):
                    tile_spec = (cfg["world_size"], 1)
                    grid = tile_spec
                    source = "postprocess"
        elif tile_spec is not None:
            grid = (
                tuple(int(entry) for entry in tile_spec)
                if tile_spec != "force"
                else resolve_tile_grid(cfg["world_size"], pair_helpers.m_alpha, pair_helpers.m_beta)
            )
            source = "force" if tile_spec == "force" else "explicit"
        return tile_spec, grid, source

    def _preflight_tile_plan(
        self,
        *,
        cfg: Any,
        n_rows: int,
        n_cols: int,
        n_a: int,
        n_b: int,
        norb: int,
        spin_sq: Any,
        kind: str,
        n_total: "Optional[int]" = None,
        pair_helpers: Any = None,
        spinless: Any = None,
    ) -> dict[str, Any]:
        """Resolve the matvec path and tile grid/source for one subspace.

        Mirrors the branch conditions in ``_solve_one_subspace`` /
        ``_solve_one_subspace_pair`` / ``_solve_spinless`` without constructing
        an operator or touching a process group.
        """
        block_cap = (
            int(self.davidson_block_size)
            if self.davidson_block_size is not None
            else max(30, self.n_roots + 3)
        )
        if n_total is not None:
            n = int(n_total)
        elif kind == "spinless":
            n = int(spinless.dim)
        else:
            n = n_rows * n_cols
        if cfg is None:
            return {
                "path": "single_process",
                "tile_spec": None,
                "tile_grid": None,
                "tile_source": None,
            }
        tile_spec = cfg.get("tile")
        postprocess = spin_sq is not None or self.compute_full_rdm or self.compute_spin_squared
        budget = self._effective_budget_cap(None if cfg is None else cfg.get("budget_bytes"))
        if budget is None:
            from torch_sci_solver.torch_impl.distributed_davidson import (
                _device_memory_budget,
            )
            from torch_sci_solver.torch_impl.distributed_solver import _rank_device

            budget = _device_memory_budget(_rank_device(cfg, 0))
        if kind == "spinless":
            from torch_sci_solver.torch_impl.spinless_tiles import (
                resolve_auto_spinless_tiling,
                spinless_tile_partition,
            )

            auto_grid = None
            if tile_spec == "auto":
                if not resolve_auto_spinless_tiling(
                    world_size=cfg["world_size"],
                    n_dets=n,
                    block_size=block_cap,
                    itemsize=int(self.dtype.itemsize),
                    budget_bytes=budget,
                ):
                    tile_spec = None
            if tile_spec is not None:
                local = n > block_cap
                if not local:
                    # The solve keeps the replicated path for a basis that fits
                    # in one Davidson block, ignoring the tile spec.
                    return {
                        "path": "replicated",
                        "tile_spec": tile_spec,
                        "tile_grid": None,
                        "tile_source": "ignored",
                    }
                grid = (
                    None
                    if tile_spec in ("auto", "force")
                    else tuple(int(entry) for entry in tile_spec)
                )
                split = int(norb) // 2
                partition, _bounds = spinless_tile_partition(
                    spinless.dets, split, cfg["world_size"], grid=grid
                )
                auto_grid = (partition.grid.p_alpha, partition.grid.p_beta)
                return {
                    "path": "local",
                    "tile_spec": tile_spec,
                    "tile_grid": auto_grid,
                    "tile_source": "explicit" if grid is not None else tile_spec,
                }
            return {
                "path": "local" if n > block_cap else "replicated",
                "tile_spec": None,
                "tile_grid": None,
                "tile_source": None,
            }
        if kind == "structural":
            tile_spec, grid, source = self._pair_tile_grid(
                cfg=cfg,
                tile_spec=tile_spec,
                n=n,
                block_cap=block_cap,
                budget=budget,
                pair_helpers=pair_helpers,
                postprocess=postprocess,
                postprocess_fallback=False,
            )
            local = cfg["matvec"] == "local"
            return {
                "path": "local" if local else "replicated",
                "tile_spec": tile_spec,
                "tile_grid": grid,
                "tile_source": "ignored" if not local and tile_spec is not None else source,
            }
        if kind == "pairs":
            tile_spec, grid, source = self._pair_tile_grid(
                cfg=cfg,
                tile_spec=tile_spec,
                n=n,
                block_cap=block_cap,
                budget=budget,
                pair_helpers=pair_helpers,
                postprocess=postprocess,
                postprocess_fallback=True,
            )
            tiled = tile_spec is not None
            # The spin_sq penalty and the <S^2> diagnostic also run on the 1-D
            # owner-computes block; full RDMs still need a tile grid.
            local = (
                cfg["matvec"] == "local" and (not self.compute_full_rdm or tiled) and n > block_cap
            )
            return {
                "path": "local" if local else "replicated",
                "tile_spec": tile_spec,
                "tile_grid": grid,
                "tile_source": source if local else ("ignored" if tile_spec is not None else None),
            }
        # TPB
        from torch_sci_solver.torch_impl.distributed_davidson import (
            resolve_auto_tile_grid,
            resolve_tile_grid,
        )

        grid = None
        if tile_spec == "auto":
            grid = resolve_auto_tile_grid(
                world_size=cfg["world_size"],
                n_rows=n_rows,
                n_cols=n_cols,
                n_particles_a=int(n_a),
                n_particles_b=int(n_b),
                n_orb=int(norb),
                budget_bytes=budget,
                itemsize=int(self.dtype.itemsize),
                block_size=block_cap,
            )
        elif tile_spec == "force":
            grid = resolve_tile_grid(cfg["world_size"], n_rows, n_cols)
        elif tile_spec not in (None, "auto"):
            grid = tuple(int(entry) for entry in tile_spec)
        auto_declined = tile_spec == "auto" and grid is None
        tile_postprocess_1d = postprocess and (
            auto_declined or (spin_sq is not None and tile_spec is None)
        )
        source = None
        if tile_postprocess_1d and grid is None:
            grid = (cfg["world_size"], 1)
            source = "postprocess"
        elif tile_spec in ("auto", "force"):
            source = tile_spec
        elif tile_spec is not None:
            source = "explicit"
        tile_path = tile_postprocess_1d or (
            tile_spec is not None and (tile_spec != "auto" or grid is not None)
        )
        local = (
            cfg["matvec"] == "local"
            and (spin_sq is None or tile_path)
            and (not self.compute_full_rdm or tile_path)
            and (not self.compute_spin_squared or tile_path)
            and n > block_cap
        )
        return {
            "path": "local" if local else "replicated",
            "tile_spec": tile_spec,
            "tile_grid": grid,
            "tile_source": source if local else ("ignored" if tile_spec is not None else None),
        }

    def _preflight_tpb(
        self,
        ci_strings: Any,
        norb: int,
        nelec: tuple,
        spin_sq: Any,
        parsed: dict,
        cfg: Any,
        *,
        one_body_tensor: Any = None,
        two_body_tensor: Any = None,
    ) -> list[dict[str, Any]]:
        """Plan the TPB (Cartesian-product) subspaces."""
        from torch_sci_solver.torch_impl.probe import _augment_core_with_hf

        n_a, n_b = nelec
        subspaces: list[dict[str, Any]] = []
        for sub_idx, (strings_a, strings_b) in enumerate(ci_strings):
            init_vector = parsed["init_list"][sub_idx]
            probe_state = parsed["probe_list"][sub_idx]
            if init_vector is not None:
                _validate_init_vector(
                    init_vector,
                    len(strings_a) * len(strings_b),
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )
            adets_limbs = _ci_string_limbs(strings_a, norb)
            bdets_limbs = _ci_string_limbs(strings_b, norb)
            if adets_limbs is not None:
                n_limbs = _limb_count(norb)
                adets = sorted(set(_limb_rows_as_ints(adets_limbs)))
                bdets = sorted(set(_limb_rows_as_ints(bdets_limbs)))
            else:
                adets = sorted(set(_ci_string_words(strings_a)))
                bdets = sorted(set(_ci_string_words(strings_b)))
            if adets_limbs is not None:
                _validate_ci_basis(
                    _ints_to_limb_matrix(adets, n_limbs),
                    _ints_to_limb_matrix(bdets, n_limbs),
                    norb,
                    nelec,
                    sub_idx=sub_idx,
                )
            else:
                _validate_ci_basis(
                    _word_bit_patterns(adets),
                    _word_bit_patterns(bdets),
                    norb,
                    nelec,
                    sub_idx=sub_idx,
                )
            cold_with_probe = parsed["probe_method"] == "warm_start" and probe_state is None
            if cold_with_probe and bool(parsed["probe_kwargs"].get("include_hf_in_core", True)):
                adets, bdets, _hf_info = _augment_core_with_hf(adets, bdets, norb, nelec)
            warm_start_active = parsed["probe_method"] == "warm_start" and probe_state is not None
            if warm_start_active:
                if not isinstance(probe_state, TorchSCIState):
                    raise TypeError(
                        "product-mode warm-start requires probe_state to be a "
                        f"TorchSCIState (got {type(probe_state).__name__!r})."
                    )
                if one_body_tensor is not None and two_body_tensor is not None:
                    from torch_sci_solver.torch_impl.probe import _validate_fingerprints

                    _validate_fingerprints(
                        probe_state,
                        _as_integrals_array(one_body_tensor),
                        _as_integrals_array(two_body_tensor),
                        norb,
                        nelec,
                        {},
                    )
                adets_new, bdets_new, _dedup_info = _dedup_and_check_superset(
                    probe_state.adets,
                    probe_state.bdets,
                    (
                        np.array(_ci_string_words(strings_a), dtype=object)
                        if adets_limbs is None
                        else _limb_rows_as_ints(adets_limbs)
                    ),
                    (
                        np.array(_ci_string_words(strings_b), dtype=object)
                        if bdets_limbs is None
                        else _limb_rows_as_ints(bdets_limbs)
                    ),
                )
                adets = [int(x) for x in adets_new]
                bdets = [int(x) for x in bdets_new]
            m_a = len(adets)
            m_b = len(bdets)
            n = m_a * m_b
            if self.n_roots > n:
                raise ValueError(f"n_roots ({self.n_roots}) cannot exceed the CI dimension ({n}).")
            plan = self._preflight_tile_plan(
                cfg=cfg,
                n_rows=m_a,
                n_cols=m_b,
                n_a=n_a,
                n_b=n_b,
                norb=norb,
                spin_sq=spin_sq,
                kind="tpb",
            )
            projected = bool(self.apply_ab_projection and m_a == m_b and list(adets) == list(bdets))
            warm = warm_start_active or init_vector is not None
            subspaces.append({
                "sub_idx": sub_idx,
                "layout": "product",
                "n_alpha": m_a,
                "n_beta": m_b,
                "dimension": n,
                "spin_sq": None if spin_sq is None else float(spin_sq),
                "ab_projection": projected,
                "start": "warm" if warm else "cold",
                "rdm_path": "full_1_and_2" if self.compute_full_rdm else "occupancies_only",
                "rdm_accumulation": self._rdm_plan_entry(
                    n_full=3 * int(norb) ** 4,
                    can_partition=plan["path"] == "local" and plan["tile_grid"] is not None,
                    cfg=cfg,
                ),
                "spin_squared": bool(self.compute_spin_squared),
                **plan,
            })
        return subspaces

    def _preflight_pairs(
        self,
        ci_strings: Any,
        norb: int,
        nelec: tuple,
        spin_sq: Any,
        parsed: dict,
        cfg: Any,
        *,
        one_body_tensor: Any = None,
        two_body_tensor: Any = None,
    ) -> list[dict[str, Any]]:
        """Plan the GDB (explicit pair list or structural block) subspaces."""
        from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
        from torch_sci_solver.torch_impl.pairs import (
            pair_basis_from_arrays,
            structural_pair_basis_from_blocks,
        )
        from torch_sci_solver.torch_impl.probe_pairs import (
            TorchPairSCIState,
            _augment_pair_with_hf,
            _dedup_and_check_pair_superset,
        )

        n_a, n_b = nelec
        subspaces: list[dict[str, Any]] = []
        for sub_idx, (strings_a, strings_b) in enumerate(ci_strings):
            init_vector = parsed["init_list"][sub_idx]
            probe_state = parsed["probe_list"][sub_idx]
            structural_spec = self._structural_pair_spec_and_gates(
                sub_idx,
                spin_sq=spin_sq,
                tile_spec=None if cfg is None else cfg.get("tile"),
                matvec=None if cfg is None else cfg["matvec"],
            )
            if structural_spec is not None:
                pair_basis = structural_pair_basis_from_blocks(structural_spec, device="cpu")
                n = pair_basis.n_pairs
                if self.n_roots > n:
                    raise ValueError(
                        f"n_roots ({self.n_roots}) cannot exceed the CI dimension ({n})."
                    )
                if init_vector is not None:
                    # Indexed by structural pair, the order of the returned amplitudes.
                    _validate_init_vector(
                        init_vector, n, dtype=self.dtype, sub_idx=sub_idx, n_roots=self.n_roots
                    )
                warm_start_active = (
                    parsed["probe_method"] == "warm_start" and probe_state is not None
                )
                if warm_start_active:
                    if not isinstance(probe_state, TorchPairSCIState):
                        raise TypeError(
                            "pair-mode warm-start requires probe_state to be a "
                            f"TorchPairSCIState (got {type(probe_state).__name__!r})."
                        )
                    if one_body_tensor is not None and two_body_tensor is not None:
                        from torch_sci_solver.torch_impl.probe import _validate_fingerprints

                        _validate_fingerprints(
                            probe_state,
                            _as_integrals_array(one_body_tensor),
                            _as_integrals_array(two_body_tensor),
                            norb,
                            nelec,
                            {},
                        )
                    _check_structural_pair_superset(
                        pair_basis,
                        probe_state.alpha_of_pair,
                        probe_state.beta_of_pair,
                        torch.device("cpu"),
                    )
                pair_helpers = generate_pair_helpers(
                    pair_basis,
                    norb,
                    device="cpu",
                    budget_bytes=(
                        self.gdb_kwargs["budget_bytes"]
                        if self.gdb_kwargs["budget_bytes"] is not None
                        else self.memory_budget_bytes
                    ),
                    block_matvec=self.gdb_kwargs["block_matvec"],
                    block_table_bytes=self.gdb_kwargs["block_table_bytes"],
                )
                plan = self._preflight_tile_plan(
                    cfg=cfg,
                    n_rows=0,
                    n_cols=0,
                    n_a=n_a,
                    n_b=n_b,
                    norb=norb,
                    spin_sq=spin_sq,
                    kind="structural",
                    n_total=n,
                    pair_helpers=pair_helpers,
                )
                subspaces.append({
                    "sub_idx": sub_idx,
                    "layout": "structural",
                    "n_alpha": int(pair_basis.n_unique_alphas),
                    "n_beta": int(pair_basis.n_unique_betas),
                    "dimension": n,
                    "spin_sq": None if spin_sq is None else float(spin_sq),
                    "ab_projection": bool(self.apply_ab_projection),
                    "start": "cold",
                    "rdm_path": ("full_1_and_2" if self.compute_full_rdm else "occupancies_only"),
                    "rdm_accumulation": self._rdm_plan_entry(
                        n_full=3 * int(norb) ** 4,
                        can_partition=plan["path"] == "local" and plan["tile_grid"] is not None,
                        cfg=cfg,
                    ),
                    "spin_squared": bool(self.compute_spin_squared),
                    **plan,
                })
                continue
            a_arr, b_arr = _pair_arrays_from_strings(strings_a, strings_b, norb)
            if init_vector is not None:
                _validate_init_vector(
                    init_vector,
                    a_arr.shape[0],
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )
            warm_start_active = parsed["probe_method"] == "warm_start" and probe_state is not None
            cold_with_probe = parsed["probe_method"] == "warm_start" and probe_state is None
            if cold_with_probe and bool(parsed["probe_kwargs"].get("include_hf_in_core", True)):
                aug_alpha, aug_beta, _hf_info = _augment_pair_with_hf(
                    a_arr.tolist(), b_arr.tolist(), norb, nelec
                )
                a_arr = np.asarray(aug_alpha, dtype=np.int64)
                b_arr = np.asarray(aug_beta, dtype=np.int64)
            if warm_start_active:
                if not isinstance(probe_state, TorchPairSCIState):
                    raise TypeError(
                        "pair-mode warm-start requires probe_state to be a "
                        f"TorchPairSCIState (got {type(probe_state).__name__!r}). A "
                        "cached product-mode TorchSCIState cannot warm-extend a pair "
                        "solve since it stores per-spin lists rather than a pair list."
                    )
                if one_body_tensor is not None and two_body_tensor is not None:
                    from torch_sci_solver.torch_impl.probe import _validate_fingerprints

                    _validate_fingerprints(
                        probe_state,
                        _as_integrals_array(one_body_tensor),
                        _as_integrals_array(two_body_tensor),
                        norb,
                        nelec,
                        {},
                    )
                union_alpha, union_beta, _pair_dedup = _dedup_and_check_pair_superset(
                    probe_state.alpha_of_pair,
                    probe_state.beta_of_pair,
                    a_arr,
                    b_arr,
                )
                a_arr = np.asarray(union_alpha, dtype=np.int64)
                b_arr = np.asarray(union_beta, dtype=np.int64)
            pair_basis = pair_basis_from_arrays(a_arr, b_arr, n_orb=norb, device="cpu")
            n = pair_basis.n_pairs
            _validate_ci_basis(
                pair_basis.unique_alphas.cpu().numpy(),
                pair_basis.unique_betas.cpu().numpy(),
                norb,
                nelec,
                sub_idx=sub_idx,
            )
            if self.n_roots > n:
                raise ValueError(f"n_roots ({self.n_roots}) cannot exceed the CI dimension ({n}).")
            pair_helpers = generate_pair_helpers(
                pair_basis,
                norb,
                device="cpu",
                budget_bytes=(
                    self.gdb_kwargs["budget_bytes"]
                    if self.gdb_kwargs["budget_bytes"] is not None
                    else self.memory_budget_bytes
                ),
                block_matvec=self.gdb_kwargs["block_matvec"],
                block_table_bytes=self.gdb_kwargs["block_table_bytes"],
            )
            plan = self._preflight_tile_plan(
                cfg=cfg,
                n_rows=0,
                n_cols=0,
                n_a=n_a,
                n_b=n_b,
                norb=norb,
                spin_sq=spin_sq,
                kind="pairs",
                n_total=n,
                pair_helpers=pair_helpers,
            )
            if plan["path"] == "local" and plan["tile_spec"] is not None:
                if pair_helpers.blocks is None:
                    raise ValueError(
                        "distributed GDB tile partitioning needs the pair block "
                        "decomposition; pass gdb_kwargs={'block_matvec': 'on'}"
                    )
            if (
                self.gdb_kwargs.get("tile_move_cache_bytes") is not None
                and cfg is not None
                and not (plan["path"] == "local" and plan["tile_spec"] is not None)
            ):
                raise ValueError(
                    "gdb_kwargs['tile_move_cache_bytes'] is only honored by the tiled "
                    "owner-computes GDB path (distributed matvec='local', tile "
                    "configured, and basis larger than one Davidson block). Use "
                    "gdb_kwargs['move_cache_bytes'] for the 1-D block path."
                )
            if self.gdb_kwargs.get("move_cache_bytes") is not None and pair_helpers.blocks is None:
                raise ValueError(
                    "gdb_kwargs['move_cache_bytes'] caches the block matvec's move "
                    "tables, but this basis has no block decomposition (block_matvec "
                    "is off or was gated out by its size/byte caps). Pass "
                    "gdb_kwargs={'block_matvec': 'on'} or drop the cache budget."
                )
            pair_projected = False
            if self.apply_ab_projection:
                k_swap, _tau = _pair_ab_swap_permutation(
                    pair_basis, norb, torch.device("cpu"), self.dtype
                )
                pair_projected = k_swap is not None
            subspaces.append({
                "sub_idx": sub_idx,
                "layout": "pairs",
                "n_alpha": int(pair_basis.n_unique_alphas),
                "n_beta": int(pair_basis.n_unique_betas),
                "dimension": n,
                "spin_sq": None if spin_sq is None else float(spin_sq),
                "ab_projection": pair_projected,
                "start": "warm" if (warm_start_active or init_vector is not None) else "cold",
                "rdm_path": "full_1_and_2" if self.compute_full_rdm else "occupancies_only",
                "rdm_accumulation": self._rdm_plan_entry(
                    n_full=3 * int(norb) ** 4,
                    can_partition=plan["path"] == "local" and plan["tile_grid"] is not None,
                    cfg=cfg,
                ),
                "spin_squared": bool(self.compute_spin_squared),
                "gdb_block_matvec": pair_helpers.blocks is not None,
                **plan,
            })
        return subspaces

    def _parse_spinless_call_kwargs(
        self, spin_sq: Any, kwargs: dict, *, validate_davidson: bool
    ) -> dict[str, Any]:
        """Pop and validate the per-call keywords of the flat spinless path.

        Shared by :meth:`_preflight_spinless` and :meth:`_solve_spinless` so the
        gate stack cannot drift between planning and solving. The one deliberate
        difference is the Davidson override check: preflight rejects non-positive
        values (``validate_davidson=True``) while the solve keeps its historical
        unchecked parsing.

        Args:
            spin_sq: the call's ``spin_sq`` request, rejected on this path.
            kwargs: the mutable per-call keyword dict; recognized keys are popped.
            validate_davidson: reject non-positive ``max_davidson_iters`` / ``tol``
                / ``de_tol``.

        Returns:
            The parsed ``call_max_iter``, ``call_eps``, ``call_de_tol``,
            ``probe_method``, ``probe_kwargs``, ``probe_state``, ``init_vectors``
            and ``build_output_states`` values.
        """
        from torch_sci_solver.torch_impl.probe_spinless import (
            SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION,
            SPINLESS_SPIN_SQ_LIMITATION,
        )

        if spin_sq is not None:
            raise ValueError(SPINLESS_SPIN_SQ_LIMITATION)
        if self.compute_spin_squared:
            raise ValueError(SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION)
        if self.ab_factorization != "off":
            raise ValueError(
                "ab_factorization factorizes the TPB αβ doubles pass; a scalar-nelec "
                "(spinless) solve has no α/β channels. Pass ab_factorization='off'."
            )
        if "max_cycle" in kwargs:
            raise TypeError(
                "max_cycle is no longer accepted; it was a misleading pyscf-style "
                "alias (nothing here calls pyscf) for the Davidson outer restart "
                "count. Pass max_davidson_iters instead — note it is a restart count, "
                "not a flat matvec budget: total work ~ max_davidson_iters * davidson_block_size."
            )

        probe_method = kwargs.pop("probe_method", None)
        probe_kwargs = kwargs.pop("probe_kwargs", None) or {}
        probe_state_in = kwargs.pop("probe_state", None)
        init_vectors_in = kwargs.pop("init_vectors", None)
        if init_vectors_in is not None and probe_state_in is not None:
            raise ValueError(
                "init_vectors and probe_state are mutually exclusive: both set the "
                "Davidson start vector. Pass one or the other."
            )
        if probe_state_in is not None and probe_method != "warm_start":
            raise ValueError(
                "probe_state requires probe_method='warm_start' on the spinless path; "
                "without it the cached state would be silently ignored."
            )
        if probe_method is not None and probe_method != "warm_start":
            raise ValueError(
                f"probe_method must be None or 'warm_start', got {probe_method!r}. "
                "(Other probe methods are not yet implemented.)"
            )
        if "spin_penalty_lambda" in kwargs:
            raise NotImplementedError(
                "spin_penalty_lambda is not supported on the spinless path: the "
                "penalty is applied to an S^2 operator, and no S^2 is defined "
                "without an alpha/beta split. Drop spin_penalty_lambda."
            )
        unsupported = set(kwargs) - {"max_davidson_iters", "tol", "de_tol"}
        if unsupported:
            raise NotImplementedError(
                f"the spinless path does not support {sorted(unsupported)}. "
                "Recognized per-call keywords here: max_davidson_iters, tol, de_tol, "
                "init_vectors, probe_method, probe_kwargs and probe_state."
            )
        if probe_method == "warm_start" and "include_hf_in_core" in probe_kwargs:
            warnings.warn(
                "spinless warm_start has no Hartree-Fock alpha/beta anchor to add; "
                "probe_kwargs['include_hf_in_core'] is ignored on this path.",
                UserWarning,
                stacklevel=2,
            )

        call_max_iter = int(kwargs.pop("max_davidson_iters", self.max_davidson_iters))
        if validate_davidson and call_max_iter <= 0:
            raise ValueError(f"max_davidson_iters must be positive, got {call_max_iter}")
        call_eps = _clamp_davidson_tolerance_for_device(
            float(kwargs.pop("tol", self.davidson_tolerance)), self.device
        )
        if validate_davidson and call_eps <= 0.0:
            raise ValueError(f"tol must be positive, got {call_eps!r}")
        de_raw = kwargs.pop("de_tol", self.davidson_de_tolerance)
        call_de_tol = None if de_raw is None else float(de_raw)
        if validate_davidson and call_de_tol is not None and call_de_tol <= 0.0:
            raise ValueError(f"de_tol must be positive or None, got {call_de_tol!r}")

        build_output_states = probe_method == "warm_start" and (
            probe_state_in is None or bool(probe_kwargs.get("build_output_state", False))
        )
        return {
            "call_max_iter": call_max_iter,
            "call_eps": call_eps,
            "call_de_tol": call_de_tol,
            "probe_method": probe_method,
            "probe_kwargs": probe_kwargs,
            "probe_state": probe_state_in,
            "init_vectors": init_vectors_in,
            "build_output_states": build_output_states,
        }

    def _preflight_spinless(
        self,
        ci_strings: Any,
        n_particles: int,
        norb: int,
        spin_sq: Any,
        kwargs: dict,
        *,
        one_body_tensor: Any = None,
        two_body_tensor: Any = None,
    ) -> dict[str, Any]:
        """Plan the flat spin-orbital path selected by a scalar ``nelec``.

        Mirrors the call-time gates of :meth:`_solve_spinless`, including the
        warm-start basis union and fingerprint validation, without running
        Davidson. The ``spin_sq`` / ``compute_spin_squared`` refusals are
        physics limitations (no alpha/beta split defines ``S^2``); their
        reasons are reported in the plan regardless of whether they were
        requested, so a workflow can gate on them before calling solve.
        """
        from torch_sci_solver.torch_impl.probe_spinless import (
            SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION,
            SPINLESS_SPIN_SQ_LIMITATION,
            _as_spinless_state_list,
            _dedup_and_check_spinless_superset,
            _validate_spinless_state,
        )
        from torch_sci_solver.torch_impl.spinless_helpers import spinless_basis_from_array

        parsed = self._parse_spinless_call_kwargs(spin_sq, kwargs, validate_davidson=True)
        call_max_iter = parsed["call_max_iter"]
        call_eps = parsed["call_eps"]
        call_de_tol = parsed["call_de_tol"]
        init_vectors_in = parsed["init_vectors"]
        probe_state_in = parsed["probe_state"]
        build_output_states = parsed["build_output_states"]
        if self._distributed_checkpoint_settings() is not None:
            raise NotImplementedError(
                "distributed checkpointing is not implemented for the spinless path; "
                "drop distributed={'checkpoint_dir': ...} for a scalar-nelec solve or "
                "use a TPB/GDB (alpha, beta) basis."
            )
        if len(self.devices) > 1:
            raise ValueError(
                "the spinless path runs on one device per process; devices= "
                "(batch-parallel multi-GPU) is not supported with a scalar nelec. "
                "Use a single device= or call the solver once per device."
            )
        if self.mode == "gdb" and self._gdb_kwargs_supplied:
            raise ValueError(
                "a scalar nelec selects the flat spinless path, where GDB gdb_kwargs "
                'have no effect. Drop gdb_kwargs (and mode="gdb") for a spinless solve.'
            )

        init_list = self._init_vector_list(init_vectors_in, len(ci_strings))
        probe_list = _as_spinless_state_list(probe_state_in, len(ci_strings))
        h1_arr = _as_integrals_array(one_body_tensor) if one_body_tensor is not None else None
        eri_arr = _as_integrals_array(two_body_tensor) if two_body_tensor is not None else None
        current_fingerprint = None
        if h1_arr is not None and eri_arr is not None and any(probe_list):
            current_fingerprint = _integrals_fingerprint(h1_arr, eri_arr, 0.0)
        current_knobs = self._spinless_knob_snapshot()

        cfg, matvec = self._preflight_distributed()
        subspaces: list[dict[str, Any]] = []
        for sub_idx, strs in enumerate(ci_strings):
            init_vector = init_list[sub_idx]
            probe_state = probe_list[sub_idx]
            if init_vector is not None:
                _validate_init_vector(
                    init_vector,
                    len(strs),
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )
            solve_strs = strs
            dedup_info = None
            if probe_state is not None:
                _validate_spinless_state(
                    probe_state,
                    h1_arr,
                    eri_arr,
                    norb,
                    n_particles,
                    current_knobs,
                    dtype=self.dtype,
                    current_fingerprint=current_fingerprint,
                )
                solve_strs, dedup_info = _dedup_and_check_spinless_superset(
                    probe_state.dets, strs, norb
                )
            basis = spinless_basis_from_array(solve_strs, norb, n_particles, device="cpu")
            n = int(basis.dim)
            if self.n_roots > n:
                raise ValueError(f"n_roots ({self.n_roots}) cannot exceed the CI dimension ({n}).")
            plan = self._preflight_tile_plan(
                cfg=cfg,
                n_rows=0,
                n_cols=0,
                n_a=n_particles,
                n_b=0,
                norb=norb,
                spin_sq=spin_sq,
                kind="spinless",
                spinless=basis,
            )
            warm_source = (
                "probe_state"
                if probe_state is not None
                else ("init_vector" if init_vector is not None else None)
            )
            subspaces.append({
                "sub_idx": sub_idx,
                "layout": "flat_determinants" if plan["tile_grid"] is None else "tiles",
                "n_alpha": None,
                "n_beta": None,
                "dimension": n,
                "spin_sq": None,
                "ab_projection": False,
                "start": "warm" if warm_source is not None else "cold",
                "warm_source": warm_source,
                "n_dets_added": None if dedup_info is None else dedup_info["dets_added"],
                "rdm_path": "full_1_and_2" if self.compute_full_rdm else "occupancies_only",
                "rdm_accumulation": self._rdm_plan_entry(
                    n_full=int(norb) ** 4,
                    can_partition=plan["path"] == "local",
                    cfg=cfg,
                ),
                "spin_squared": False,
                **plan,
            })
        return {
            "mode": self.mode,
            "basis": "spinless",
            "particle_sector": {"n_particles": int(n_particles), "n_alpha": None, "n_beta": None},
            "n_roots": self.n_roots,
            "matvec_impl": matvec,
            "matvec_impl_requested": self._matvec_impl_requested,
            "distributed": _jsonable(cfg),
            "tile_spec": cfg.get("tile") if cfg is not None else None,
            "tile_grid": subspaces[0]["tile_grid"] if subspaces else None,
            "tile_source": subspaces[0]["tile_source"] if subspaces else None,
            "spin_sq": None,
            "spin_sq_unsupported_reason": SPINLESS_SPIN_SQ_LIMITATION,
            "rdm_path": "full_1_and_2" if self.compute_full_rdm else "occupancies_only",
            "rdm_accumulation": subspaces[0]["rdm_accumulation"] if subspaces else None,
            "compute_full_rdm": bool(self.compute_full_rdm),
            "compute_spin_squared": False,
            "compute_spin_squared_unsupported_reason": (SPINLESS_COMPUTE_SPIN_SQUARED_LIMITATION),
            "probe_state_type": "TorchSpinlessSCIState",
            "build_output_states": build_output_states,
            "warm_start": any(entry["start"] == "warm" for entry in subspaces),
            "call_max_iter": call_max_iter,
            "call_tol": call_eps,
            "call_de_tol": call_de_tol,
            "subspaces": [
                {
                    **entry,
                    "capacity": self._preflight_capacity(
                        entry,
                        cfg,
                        norb,
                        kind="spinless",
                        nelec=n_particles,
                        matvec_impl=matvec,
                    ),
                }
                for entry in subspaces
            ],
        }

    def preflight(
        self,
        ci_strings: Any,
        nelec: Any,
        *,
        norb: "Optional[int]" = None,
        one_body_tensor: Any = None,
        two_body_tensor: Any = None,
        **call_kwargs: Any,
    ) -> dict[str, Any]:
        """Run every call-time gate without solving and return the resolved plan.

        The same validation the next solve would perform runs first: unknown
        per-call keywords (according to :attr:`unknown_call_kwargs`), the
        ``max_cycle`` rejection, Davidson overrides, probe/``init_vectors``
        normalization, basis deduplication/popcount checks, pair-shape and
        structural-basis gates, ``n_roots`` vs basis dimension, the
        distributed matvec/backend gates and
        the mode-specific feature gates. No Davidson iteration, process group,
        operator or device allocation is started.

        Args:
            ci_strings: The same per-subspace input the solve would receive.
            nelec: ``(n_alpha, n_beta)`` or a scalar total particle number
                (the spinless path).
            norb: Number of spatial orbitals (spin-orbitals on the spinless
                path). When omitted it is inferred from the highest set bit;
                pass it explicitly for wide/limb strings.
            one_body_tensor: Optional integrals, used only to validate a warm
                start's cached integral fingerprint.
            two_body_tensor: Optional two-electron integrals for the same
                fingerprint check.
            **call_kwargs: The per-call keywords the solve would receive,
                including ``spin_sq``, ``tol``, ``init_vectors``, ...

        Returns:
            A JSON-friendly plan: the effective mode/basis, particle sector,
            per-subspace dimensions and layouts, resolved ``matvec_impl``, tile
            grid and its source, warm/cold start, and the RDM path. The
            ``rdm_path`` key reports whether a full RDM was requested; the
            ``rdm_accumulation`` key (also per subspace) reports whether it is
            accumulated ``"partitioned"`` or ``"replicated"``. The
            ``subspaces`` entries also report each subspace's replicated vs
            owner-computes path. Without an explicit ``memory_budget_bytes`` /
            distributed ``budget_bytes`` cap, an ``"auto"`` tile decision uses
            the current free-memory estimate rather than the post-setup budget
            the real call sees; pass a cap for a deterministic plan.

        Raises:
            The same ``ValueError`` / ``TypeError`` /
            ``NotImplementedError`` the solve would raise for these inputs.
        """
        # ``spin_sq`` is an explicit argument of ``__call__``, so it arrives in
        # ``**call_kwargs`` here; pop it before the unknown-name filter.
        spin_sq = call_kwargs.pop("spin_sq", None)
        call_kwargs = self._filter_unknown_call_kwargs(dict(call_kwargs))
        if not self.dtype.is_complex and (
            (one_body_tensor is not None and np.iscomplexobj(one_body_tensor))
            or (two_body_tensor is not None and np.iscomplexobj(two_body_tensor))
        ):
            raise TypeError(
                f"one_body_tensor/two_body_tensor is complex but this solver was "
                f"constructed with dtype={self.dtype} (real). Pass "
                "dtype=torch.complex128 (or complex64) to TorchSCISolver(...) to "
                "solve a complex Hermitian Hamiltonian — the imaginary part is "
                "never silently dropped."
            )
        if norb is None:
            norb = self._infer_norb(ci_strings, nelec)
        norb = int(norb)
        if isinstance(nelec, (int, np.integer)):
            return self._preflight_spinless(
                ci_strings,
                int(nelec),
                norb,
                spin_sq,
                call_kwargs,
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
            )
        n_a, n_b = nelec
        spin_sq_target = None
        if spin_sq is not None:
            try:
                spin_sq_target = float(spin_sq)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"spin_sq must be None or convertible to float, got {spin_sq!r}"
                ) from exc
        parsed = self._preflight_parse_call(ci_strings, call_kwargs)
        if call_kwargs:
            raise TypeError(f"internal preflight error: unparsed call kwargs {sorted(call_kwargs)}")
        cfg, matvec = self._preflight_distributed()
        if cfg is not None:
            # Mirror the replicated-fallback gate in ``run_distributed_call``:
            # a call that lands on ``davidson_distributed`` rejects the
            # missed-root search before launching any rank. The fallback now
            # carries complex coefficients too, so it applies to every dtype.
            tile_postprocess = cfg.get("tile") not in (None, "auto")
            local_penalty = spin_sq_target is not None and cfg["matvec"] == "local"
            replicated = (
                cfg["matvec"] == "replicated"
                or (spin_sq_target is not None and not local_penalty)
                or (self.compute_full_rdm and not tile_postprocess)
                or (self.compute_spin_squared and not tile_postprocess)
            )
            if replicated:
                warm_state_requested = any(state is not None for state in parsed["probe_list"])
                missed_root = self.n_roots > 1 and (
                    self.davidson_missed_root_check is True
                    or (self.davidson_missed_root_check == "auto" and warm_state_requested)
                )
                if missed_root:
                    raise NotImplementedError(
                        "davidson_missed_root_check on a multi-root distributed solve "
                        "requires the owner-computes path; this call uses the "
                        "replicated matvec"
                    )
        if self.mode == "gdb":
            basis = "structural" if self.gdb_kwargs.get("structural_blocks") else "pairs"
            subspaces = self._preflight_pairs(
                ci_strings,
                norb,
                (n_a, n_b),
                spin_sq_target,
                parsed,
                cfg,
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
            )
        else:
            basis = "product"
            subspaces = self._preflight_tpb(
                ci_strings,
                norb,
                (n_a, n_b),
                spin_sq_target,
                parsed,
                cfg,
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
            )
        probe_state_requested = any(state is not None for state in parsed["probe_list"])
        build_output_states = parsed["probe_method"] == "warm_start" and (
            not probe_state_requested
            or bool(parsed["probe_kwargs"].get("build_output_state", False))
        )
        return {
            "mode": self.mode,
            "basis": basis,
            "particle_sector": {"n_alpha": int(n_a), "n_beta": int(n_b)},
            "n_roots": self.n_roots,
            "matvec_impl": matvec,
            "matvec_impl_requested": self._matvec_impl_requested,
            "distributed": _jsonable(cfg),
            "tile_spec": subspaces[0]["tile_spec"] if subspaces else None,
            "tile_grid": subspaces[0]["tile_grid"] if subspaces else None,
            "tile_source": subspaces[0]["tile_source"] if subspaces else None,
            "spin_sq": spin_sq_target,
            # S^2 support is mode-dependent: the product and pair paths build
            # it from the alpha/beta partition, while the spinless plan (above)
            # reports a physics-limitation reason instead.
            "spin_sq_unsupported_reason": None,
            "compute_spin_squared_unsupported_reason": None,
            "probe_state_type": ("TorchPairSCIState" if self.mode == "gdb" else "TorchSCIState"),
            "build_output_states": build_output_states,
            "rdm_path": "full_1_and_2" if self.compute_full_rdm else "occupancies_only",
            "rdm_accumulation": subspaces[0]["rdm_accumulation"] if subspaces else None,
            "compute_full_rdm": bool(self.compute_full_rdm),
            "compute_spin_squared": bool(self.compute_spin_squared),
            "warm_start": any(entry["start"] == "warm" for entry in subspaces),
            "call_max_iter": parsed["call_max_iter"],
            "call_tol": parsed["call_eps"],
            "call_de_tol": parsed["call_de_tol"],
            "checkpoint": self._preflight_checkpoint(
                cfg,
                subspaces,
                norb=norb,
                nelec=(n_a, n_b),
                spin_penalty_lambda=parsed["call_spin_lambda"],
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
            ),
            "subspaces": [
                {
                    **entry,
                    "capacity": self._preflight_capacity(
                        entry,
                        cfg,
                        norb,
                        kind="gdb" if self.mode == "gdb" else "tpb",
                        nelec=(n_a, n_b),
                        matvec_impl=matvec,
                    ),
                }
                for entry in subspaces
            ],
        }

    def close(self) -> None:
        """Tear down the multi-GPU worker pool or distributed session, if any.

        Idempotent and safe to call on a single-device solver (no-op). The
        distributed session is also closed by a ``weakref.finalize`` guard.
        """
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool = None
        if self._distributed_session is not None:
            self._distributed_session.close()
            self._distributed_session = None
            if self._distributed_session_finalizer is not None:
                self._distributed_session_finalizer.detach()
                self._distributed_session_finalizer = None

    def __enter__(self) -> "TorchSCISolver":
        """Support ``with make_torch_sci_solver(...) as solver:`` for explicit
        deterministic pool teardown."""
        return self

    def __exit__(self, *exc) -> None:
        """Tear down the worker pool on context exit."""
        self.close()

    def _refines_fp32_energy(self, call_eps: float) -> bool:
        """Whether this call reports float64 CPU Rayleigh quotients for an MPS solve.

        MPS holds every vector in float32, so a Ritz value carries float32 rounding
        of the order ``|E| eps(float32)`` per accumulation; with ``rq64`` (and with
        ``auto`` below ``tol = 1e-3``, the same rule as the narrow subspace) the
        reported energy is instead ``⟨v|H|v⟩ / ⟨v|v⟩`` of the converged vector,
        evaluated once per root by a float64 matvec on the CPU. ``ritz`` keeps the
        float32 value.
        """
        if self.device.type != "mps":
            return False
        mode = self.davidson_fp32_energy
        if mode == "auto":
            mode = "ritz" if call_eps >= _FP32_RITZ_MIN_TOL else "rq64"
        return mode == "rq64"

    def _wide_dtype(self) -> torch.dtype:
        """float64, or complex128 for a complex solve."""
        return torch.complex128 if self.dtype.is_complex else torch.float64

    def _cpu_wide_integrals(
        self, one_body_tensor: Any, two_body_tensor: Any, *, spinless: bool
    ) -> "TorchIntegrals":
        """The Hamiltonian's integrals on the CPU in the wide dtype, kept per Hamiltonian.

        Matched by identity first and by content second, like the device integrals
        cache, so an SQD loop's per-iteration copies build them once.
        """
        cached = self._cpu_wide_integrals_cache
        if cached is not None and cached[2] == spinless:
            same = (cached[0] is one_body_tensor and cached[1] is two_body_tensor) or (
                np.shape(cached[0]) == np.shape(one_body_tensor)
                and np.shape(cached[1]) == np.shape(two_body_tensor)
                and np.array_equal(cached[0], one_body_tensor)
                and np.array_equal(cached[1], two_body_tensor)
            )
            if same:
                self._cpu_wide_integrals_cache = (
                    one_body_tensor,
                    two_body_tensor,
                    spinless,
                    cached[3],
                )
                return cached[3]
        wide = self._wide_dtype()
        if spinless:
            from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless

            integrals = to_torch_integrals_spinless(
                one_body_tensor, two_body_tensor, core_energy=0.0, device="cpu", dtype=wide
            )
        else:
            I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(
                _as_integrals_array(one_body_tensor), _as_integrals_array(two_body_tensor), 0.0
            )
            integrals = to_torch_integrals(
                I0,
                I1,
                I2,
                device="cpu",
                dtype=wide,
                compact_eri=_resolve_compact_eri(
                    self.compact_eri,
                    matvec_impl="batched",
                    mode=self.mode,
                    n_spinorb=int(I1.n_spinorb),
                ),
            )
        self._cpu_wide_integrals_cache = (one_body_tensor, two_body_tensor, spinless, integrals)
        return integrals

    def _ab_factors(
        self, two_body_tensor: Any, norb: int
    ) -> "Optional[tuple[torch.Tensor, torch.Tensor]]":
        """Signed ERI factors for the factorized αβ pass, or ``None`` when it is off.

        Factorized once per Hamiltonian. The last ERI and its factors are kept and
        matched by identity first and by content second, exactly like the integrals
        cache the rest of the matvec reads, so the two always describe the same
        Hamiltonian: a fresh copy of the same integrals (an SQD loop passes them
        every iteration) hits both, and, as for the integrals cache, an array
        changed in place between calls is not detected. A PySCF-packed ERI is
        unpacked to ``(L, L, L, L)`` first.
        """
        if self.ab_factorization == "off":
            return None
        cached = self._ab_factor_cache
        if cached is not None:
            if cached[0] is two_body_tensor:
                return cached[1]
            if np.shape(cached[0]) == np.shape(two_body_tensor) and np.array_equal(
                cached[0], two_body_tensor
            ):
                self._ab_factor_cache = (two_body_tensor, cached[1])
                return cached[1]
        from torch_sci_solver.sbd_ref.integrals import _to_full_eri
        from torch_sci_solver.torch_impl.ab_factorized import factorize_eri

        full = _to_full_eri(_as_integrals_array(two_body_tensor), int(norb))
        factors = factorize_eri(torch.as_tensor(full), tol=self.ab_factorization_tol)
        self._ab_factor_cache = (two_body_tensor, factors)
        return factors

    def _free_bytes(self, device: "Optional[torch.device]" = None) -> int:
        """Free bytes on ``device`` (defaults to ``self.device``).

        For multi-GPU dispatch we want the free memory of the specific device
        the current subspace is going to run on, not the primary one. CUDA reads
        the device's free memory, CPU the available host RAM, and MPS the smaller
        of host RAM and the Metal working-set headroom.
        """
        dev = device if device is not None else self.device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            free, _total = torch.cuda.mem_get_info(dev)
            return int(free)
        try:
            import psutil

            host = int(psutil.virtual_memory().available)
        except ImportError:
            host = 32 * 1024**3
        if dev.type == "mps":
            # Unified memory: Metal caps the GPU working set below physical RAM,
            # so the free budget is the smaller of that headroom and host RAM.
            metal = torch.mps.recommended_max_memory() - torch.mps.driver_allocated_memory()
            return max(0, min(host, int(metal)))
        return host

    @staticmethod
    def _fmt(n: int) -> str:
        """Fmt."""
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} PB"

    def _pick_num_block(
        self,
        M_a: int,
        M_b: int,
        device: "Optional[torch.device]" = None,
        n_roots: int = 1,
        *,
        local_n: "Optional[int]" = None,
        matvec_vectors: int = 4,
        cap_to_n: bool = True,
    ) -> int:
        """Choose self.davidson_block_size to fit the Davidson subspace in the currently-free
        memory. Davidson allocates ``C, HC`` each ``(self.davidson_block_size, N)`` plus a
        handful of length-N working vectors. With the v0.10.7+ matmul fix,
        no transient (self.davidson_block_size, N) broadcast is materialized.

        Note: ``self._free_bytes()`` is read AFTER helpers + hii + W are on-device,
        so this is the actual remaining budget for Davidson's own arrays.

        We size as ``(2·b·s_d + v·s) × N ≤ self.budget_fraction × free`` with ``s_d``/``s``
        the Davidson and matvec element sizes (:meth:`_davidson_element_bytes`),
        which has been calibrated against actual fp64 GPU runs (matches observed
        27 GB at davidson_block_size=30, N=53.6M).

        Under multi-GPU dispatch, ``device`` must be the specific GPU this
        subspace will run on so the free-memory reading is meaningful.

        ``n_roots`` is the number of eigenpairs requested. The returned value
        is always >= n_roots so that ``davidson_torch`` never raises
        "n_roots > davidson_block_size".

        Distributed owner-computes path: ``local_n`` overrides the ``M_a × M_b``
        product with the per-rank block length (each rank stores ``C``/``HC`` as
        ``(d, N_local)``), ``matvec_vectors`` is the number of length-``N_local``
        buffers the local operator and the staged exchange hold live (9 here),
        and ``cap_to_n=False`` leaves the block uncapped by the local
        length, since a rank may own fewer rows than the global block size.
        """
        if self.davidson_block_size is not None:
            return int(self.davidson_block_size)
        N = M_a * M_b if local_n is None else int(local_n)
        free = self._free_bytes(device=device)
        # An explicit per-rank cap also bounds the automatic Davidson sizing;
        # without one the historical free-memory estimate is used unchanged.
        cap = self._effective_budget_cap()
        if cap is not None:
            free = min(free, cap)
        chosen = self._block_for_budget(
            free, N, n_roots=n_roots, matvec_vectors=matvec_vectors, cap_to_n=cap_to_n
        )
        if self._distributed_ctx is not None:
            import torch.distributed as dist

            # Restart dimensions must agree even when ranks have different free memory.
            common = torch.tensor(chosen, device=device, dtype=torch.int64)
            dist.all_reduce(common, op=dist.ReduceOp.MIN, group=self._distributed_ctx.group)
            chosen = int(common.item())
        return chosen

    def _davidson_element_bytes(self) -> tuple[int, int]:
        """Bytes per element of the Davidson block and of the matvec work vectors.

        The subspace (``C``, ``HC``) is stored in Davidson's dtype, so a narrow
        subspace (fp32 over fp64, complex64 over complex128) halves it; the work
        vectors stay in the matvec dtype, since the matvec output is full precision
        even then (its first op promotes the narrow input). A complex element is
        twice its real counterpart: complex128 is 16 bytes, complex64 is 8.
        """
        sub = torch.empty((), dtype=self.davidson_dtype or self.dtype).element_size()
        work = torch.empty((), dtype=self.dtype).element_size()
        return sub, work

    def _block_for_budget(
        self, free: int, n: int, *, n_roots: int, matvec_vectors: int, cap_to_n: bool
    ) -> int:
        """The Davidson block the picker chooses for ``free`` bytes over ``n`` elements.

        Shared by :meth:`_pick_num_block` and the preflight capacity estimate, so the
        predicted block is the one a solve uses for the same budget.
        """
        budget = int(free * self.budget_fraction)
        sub_elem, work_elem = self._davidson_element_bytes()
        per_block = 2 * n * sub_elem
        overhead = matvec_vectors * n * work_elem
        if budget <= overhead:
            heuristic = 2
        else:
            heuristic = int(min(30, max(2, (budget - overhead) // per_block)))
        # Bound the subspace size to the actual CI space dimension N, but
        # never below n_roots (which would immediately raise in davidson_torch).
        return max(n_roots, min(heuristic, n) if cap_to_n else heuristic)

    def _cold_pair_init_vector(
        self, n_pairs: int, best_pair_idx: int, dev: torch.device
    ) -> torch.Tensor:
        """GDB cold start: the HF-like seed, moved to ``best_pair_idx`` if nonzero."""
        from torch_sci_solver.torch_impl.pairs import basis_init_vector_pair_torch

        W = basis_init_vector_pair_torch(n_pairs, init=0, device=dev, dtype=self.dtype)
        if best_pair_idx != 0:
            W[0] = 0.0
            W[best_pair_idx] = 1.0
        return W

    def _narrow_shift(
        self,
        hmin: torch.Tensor,
        reduce_min: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> float:
        """Resolve the shift subtracted inside a narrow-subspace Davidson solve.

        Args:
            hmin: The diagonal the shift is taken from (possibly rank-local).
            reduce_min: Optional in-place MIN all-reduce for a rank-local
                diagonal.

        Returns:
            ``"hf"`` maps to the diagonal minimum; any other setting is the
            configured float, or ``0.0`` when unset.
        """
        if self.davidson_fp32_shift == "hf":
            shift = hmin.min().reshape(1).to(torch.float64)
            if reduce_min is not None:
                reduce_min(shift)
            return float(shift.item())
        return float(self.davidson_fp32_shift or 0.0)

    def _narrow_matvec(
        self,
        op: Callable[[torch.Tensor], torch.Tensor],
        e_shift: float,
        dav_dtype: torch.dtype,
        *,
        cast_input: bool = False,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the shifted full-precision matvec cast into the narrow subspace.

        Args:
            op: Full-precision matvec; it must allocate a fresh output tensor.
            e_shift: Shift subtracted in full precision.
            dav_dtype: Narrow subspace dtype.
            cast_input: Cast ``v`` to the full dtype before the subtraction; the
                owner-computes operator does not promote its input the way the
                full-vector kernels do.

        Returns:
            The ``matvec_dav`` closure handed to the Davidson driver.
        """

        def matvec_dav(v: torch.Tensor) -> torch.Tensor:
            # In place on the matvec's fresh output: ``y - e_shift * v`` would
            # hold two more full-precision N-vectors, the memory the narrow
            # subspace exists to save.
            y = op(v)
            if e_shift:
                y.sub_(v.to(self.dtype) if cast_input else v, alpha=e_shift)
            return y.to(dav_dtype)

        return matvec_dav

    def _finish_narrow_result(
        self,
        energies: Any,
        vecs: torch.Tensor,
        info: dict,
        call_eps: float,
        e_shift: float,
        *,
        rq_fn: Callable[[torch.Tensor], float],
        normalize_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, bool, str]:
        """Finish the shared narrow-subspace energy/vector tail.

        Args:
            energies: Ritz values returned by the Davidson driver.
            vecs: Ritz vectors (``(N,)`` for one root, ``(n_roots, N)`` else).
            info: Davidson info dict; the rq64 pass bumps ``n_matvecs``.
            call_eps: Call tolerance, for ``davidson_fp32_energy="auto"``.
            e_shift: Shift added back to the Ritz energies in ``"ritz"`` mode.
            rq_fn: One root's full-precision Rayleigh quotient; the distributed
                local caller reduces its numerator and denominator.
            normalize_fn: Renormalizes the full-dtype vectors; the distributed
                local caller uses the global norm.

        Returns:
            ``(energies, vecs, single, energy_mode)``, energies and vectors in
            the solver's full dtype.
        """
        single = self.n_roots == 1
        vecs = (vecs.unsqueeze(0) if single else vecs).to(self.dtype)
        vecs = normalize_fn(vecs)
        energy_mode = self.davidson_fp32_energy
        if energy_mode == "auto":
            energy_mode = "ritz" if call_eps >= _FP32_RITZ_MIN_TOL else "rq64"
        if energy_mode == "rq64":
            # vdot + .real: Hermitian Rayleigh quotient, real for a real/Hermitian H.
            energies_t = torch.tensor([rq_fn(v) for v in vecs], dtype=torch.float64)
            info["n_matvecs"] = info.get("n_matvecs", 0) + len(vecs)
        else:
            energies_t = torch.atleast_1d(torch.as_tensor(energies, dtype=torch.float64)) + e_shift
        return energies_t, vecs, single, energy_mode

    def _run_davidson(
        self,
        hii: torch.Tensor,
        W: torch.Tensor,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        call_max_iter: int,
        chosen_nb: int,
        call_eps: float,
        *,
        call_de_tol: Optional[float],
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        warm: bool = False,
    ) -> Tuple[Any, dict]:
        """:meth:`_run_davidson_once`, repeated once from argmin(hii) if it ends above min(hii).

        Notes
        -----
        Single-root, unprojected solves only. The lowest eigenvalue of a
        Hermitian ``H`` satisfies ``λ_min <= min_i H_ii``, so a Ritz value above
        ``min(hii)`` (by more than ``max(call_eps, 1e-9 |min(hii)|)``) is not the
        ground state: Davidson does not leave the block of ``H`` its start vector
        lies in, and
        that block holds no lowest-diagonal determinant (CI strings that no
        single or double excitation connects; a ``connected`` or caller-supplied
        start). The solve is then repeated once from the lowest-diagonal
        determinant, with a ``RuntimeWarning``. ``hii`` is the diagonal of the
        operator Davidson sees (spin penalty included). A projected solve is not
        checked: its lowest eigenvalue on the projector's range is not bounded by
        ``min(hii)``. A multi-root seed block already holds the lowest-diagonal
        unit vectors, and its missed-root check covers the rest.
        """
        result, perf = self._run_davidson_once(
            hii,
            W,
            matvec,
            call_max_iter,
            chosen_nb,
            call_eps,
            call_de_tol=call_de_tol,
            project_fn=project_fn,
            warm=warm,
        )
        if project_fn is not None or self.n_roots > 1:
            return result, perf
        energy = float(result[0])
        floor = float(hii.min())
        if energy <= floor + max(call_eps, 1e-9 * max(1.0, abs(floor))):
            return result, perf
        seed_index = int(torch.argmin(hii))
        warnings.warn(
            f"Davidson ended at E = {energy:.8f}, above min(diag H) = {floor:.8f}, "
            "which no lowest eigenvalue can be: the start vector lies in a block of H "
            "that no single or double excitation connects to the lowest-diagonal "
            "determinant (disconnected CI strings, or a 'connected' or caller-supplied "
            f"start). Repeating the solve once from that determinant (index {seed_index}); "
            "removing disconnected strings from the basis avoids the second solve.",
            RuntimeWarning,
            stacklevel=2,
        )
        start = basis_init_vector_torch(
            1, W.shape[-1], init=0, device=W.device, dtype=W.dtype, seed_index=seed_index
        )
        return self._run_davidson_once(
            hii,
            start,
            matvec,
            call_max_iter,
            chosen_nb,
            call_eps,
            call_de_tol=call_de_tol,
            project_fn=project_fn,
        )

    def _run_davidson_once(
        self,
        hii: torch.Tensor,
        W: torch.Tensor,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        call_max_iter: int,
        chosen_nb: int,
        call_eps: float,
        *,
        call_de_tol: Optional[float],
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        warm: bool = False,
    ) -> Tuple[Any, dict]:
        """``davidson_torch``, with the subspace in ``self.davidson_dtype`` when set.

        Returns exactly what ``davidson_torch`` returns (single- or multi-root shape),
        always with the energies and vectors in ``self.dtype``, plus a perf dict
        ``{"davidson_dtype", "davidson_fp32_shift", "davidson_fp32_energy"}`` (the last
        is the resolved ``"rq64"``/``"ritz"``, or ``None`` without an fp32 subspace).
        With the default ``davidson_dtype`` it is a plain ``davidson_torch`` call.

        Notes
        -----
        With an fp32 subspace Davidson diagonalizes ``H' = H − s·I``, ``s`` the shift:
        the fp64 matvec computes ``H v`` (its ``hii * v`` promotes an fp32 ``v``), the
        shift is subtracted in fp64, and only the result is cast to fp32. The shift
        leaves the eigenvectors and the residual ``‖H'v − λ'v‖ = ‖Hv − λv‖`` unchanged,
        so ``tol`` keeps its meaning, but brings the Ritz values from total energies
        (~10–100 Ha, ~1e-5 Ha resolution in fp32) down to correlation size. The
        returned vectors are cast back to fp64 and renormalized; the energy is either
        ``⟨v|H|v⟩`` in fp64 (``"rq64"``, one matvec per root) or ``λ' + s``
        (``"ritz"``). Either way it is the energy of the operator Davidson saw, so the
        caller's ``spin_sq`` penalty strip still applies.
        """
        dav_dtype = self.davidson_dtype or self.dtype
        check = self._missed_root_check(warm)
        # N <= block: Davidson builds and diagonalizes H densely in W's dtype; keep
        # fp64 there (tiny bases, memory is irrelevant).
        if dav_dtype == self.dtype or W.shape[-1] <= chosen_nb:
            result = davidson_torch(
                hii,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                n_roots=self.n_roots,
                de_tol=call_de_tol,
                project_fn=project_fn,
                verbose=False,
                correction=self.davidson_correction,
                restart=self.davidson_restart,
                **check,
            )
            return result, {
                "davidson_dtype": str(self.dtype),
                "davidson_fp32_shift": 0.0,
                "davidson_fp32_energy": None,
            }

        e_shift = self._narrow_shift(hii)
        matvec_dav = self._narrow_matvec(matvec, e_shift, dav_dtype)

        energies, vecs, info = davidson_torch(
            (hii - e_shift).to(dav_dtype.to_real()),
            W.to(dav_dtype),
            matvec_dav,
            call_max_iter,
            chosen_nb,
            call_eps,
            n_roots=self.n_roots,
            de_tol=call_de_tol,
            project_fn=project_fn,
            verbose=False,
            correction=self.davidson_correction,
            restart=self.davidson_restart,
            **check,
        )
        energies_t, vecs, single, energy_mode = self._finish_narrow_result(
            energies,
            vecs,
            info,
            call_eps,
            e_shift,
            rq_fn=lambda v: float(torch.vdot(v, matvec(v)).real.item()),
            normalize_fn=lambda v: v / torch.linalg.norm(v, dim=1, keepdim=True),
        )
        perf = {
            "davidson_dtype": str(dav_dtype),
            "davidson_fp32_shift": e_shift,
            "davidson_fp32_energy": energy_mode,
        }
        if single:
            return (float(energies_t[0]), vecs[0], info), perf
        return (energies_t, vecs, info), perf

    def _run_davidson_distributed(
        self,
        hii: torch.Tensor,
        W: torch.Tensor,
        matvec: Callable[[torch.Tensor], torch.Tensor],
        call_max_iter: int,
        chosen_nb: int,
        call_eps: float,
        *,
        call_de_tol: Optional[float],
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        context,
        warm: bool = False,
        checkpoint: "Optional[Any]" = None,
    ) -> Tuple[Any, dict]:
        """Run one subspace's Davidson through the replicated fallback.

        Returns the same shape as :meth:`_run_davidson` (single-root
        ``(float, (N,), info)`` or multi-root ``((n_roots,), (n_roots, N),
        info)``), with the eigenvectors all-gathered to full vectors on every
        rank so the caller's existing post-processing is unchanged.

        A narrow subspace (``self.davidson_dtype``) diagonalizes ``H - s``
        with the same shift/rq64/ritz protocol as :meth:`_run_davidson`; every
        rank holds the full ``hii``/``W``, so no reduction is needed. Complex
        coefficients (``dtype`` complex128 or a complex64 narrow subspace) take
        the same path: the diagonal stays real, every inner product is
        Hermitian and the replicated ``C``/``HC`` blocks are partitioned as
        ``DistributedVector``s. The missed-root search is still rejected here:
        ``davidson_distributed`` has no deflation hook for it.
        """
        from torch_sci_solver.torch_impl.distributed_davidson import davidson_distributed

        if self._missed_root_check(warm).get("missed_root_check"):
            raise NotImplementedError(
                "davidson_missed_root_check on a multi-root distributed solve requires "
                "the owner-computes path; this call uses the replicated matvec"
            )
        dav_dtype = self.davidson_dtype or self.dtype
        if dav_dtype == self.dtype:
            result = davidson_distributed(
                hii,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                n_roots=self.n_roots,
                de_tol=call_de_tol,
                project_fn=project_fn,
                context=context,
                verbose=self.verbose,
                checkpoint=checkpoint,
            )
            return result, {
                "davidson_dtype": str(self.dtype),
                "davidson_fp32_shift": 0.0,
                "davidson_fp32_energy": None,
                "distributed_matvec": "replicated",
                "warm": warm,
            }

        e_shift = self._narrow_shift(hii)
        matvec_dav = self._narrow_matvec(matvec, e_shift, dav_dtype)

        energies, vecs, info = davidson_distributed(
            (hii - e_shift).to(dav_dtype.to_real()),
            W.to(dav_dtype),
            matvec_dav,
            call_max_iter,
            chosen_nb,
            call_eps,
            n_roots=self.n_roots,
            de_tol=call_de_tol,
            project_fn=project_fn,
            context=context,
            verbose=self.verbose,
            checkpoint=checkpoint,
        )
        # The full vectors are all-gathered and already normalized; renormalize
        # defensively, exactly like the serial narrow-subspace path.
        energies_t, vecs, single, energy_mode = self._finish_narrow_result(
            energies,
            vecs,
            info,
            call_eps,
            e_shift,
            rq_fn=lambda v: float(torch.vdot(v, matvec(v)).real.item()),
            normalize_fn=lambda v: v / torch.linalg.norm(v, dim=1, keepdim=True),
        )
        perf = {
            "davidson_dtype": str(dav_dtype),
            "davidson_fp32_shift": e_shift,
            "davidson_fp32_energy": energy_mode,
            "distributed_matvec": "replicated",
            "warm": warm,
        }
        if single:
            return (float(energies_t[0]), vecs[0], info), perf
        return (energies_t, vecs, info), perf

    def _run_davidson_distributed_local(
        self,
        hii_local: torch.Tensor,
        W_local: torch.Tensor,
        operator,
        bounds,
        *,
        context,
        call_max_iter: int,
        chosen_nb: int,
        call_eps: float,
        call_de_tol: Optional[float],
        project_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        warm: bool,
        checkpoint: "Optional[Any]" = None,
    ) -> Tuple[Any, dict]:
        """One subspace's Davidson on the owner-computes local operator.

        The fp64 path is a direct :func:`davidson_distributed_local` call. An
        fp32 subspace diagonalizes ``H - s`` locally (``s`` is the
        ``davidson_fp32_shift``, ``"hf"`` meaning the global ``min(hii)``), and
        the energy is either one distributed fp64 Rayleigh-quotient matvec per
        root (``"rq64"``) or the shifted Ritz value (``"ritz"``), mirroring
        :meth:`_run_davidson`. Returned vectors are always fp64 local blocks,
        globally renormalized.

        Notes
        -----
        The operator applies ``H`` in full precision: ``hii_local`` promotes the
        narrow (fp32 or complex64) subspace vector, and the kernels upcast on
        read. Only the result is cast back down into the subspace, so the shift
        keeps full precision.
        """
        import torch.distributed as dist

        from torch_sci_solver.torch_impl.distributed_davidson import (
            davidson_distributed_local,
        )

        check = self._missed_root_check(warm)
        dav_dtype = self.davidson_dtype or self.dtype
        n_global = int(operator.n)
        if dav_dtype == self.dtype:
            result = davidson_distributed_local(
                hii_local,
                W_local,
                operator,
                bounds,
                n=n_global,
                context=context,
                max_iters=call_max_iter,
                block_size=chosen_nb,
                eps=call_eps,
                n_roots=self.n_roots,
                de_tol=call_de_tol,
                project_fn=project_fn,
                missed_root_check=bool(check.get("missed_root_check", False)),
                verbose=self.verbose,
                checkpoint=checkpoint,
            )
            return result, {
                "davidson_dtype": str(self.dtype),
                "davidson_fp32_shift": 0.0,
                "davidson_fp32_energy": None,
                "distributed_matvec": "local",
                "warm": warm,
            }

        def reduce_min(value: torch.Tensor) -> None:
            if context.world_size > 1:
                dist.all_reduce(value, op=dist.ReduceOp.MIN, group=context.group)

        e_shift = self._narrow_shift(hii_local, reduce_min)
        matvec_dav = self._narrow_matvec(operator, e_shift, dav_dtype, cast_input=True)

        energies, vecs, info = davidson_distributed_local(
            (hii_local - e_shift).to(dav_dtype.to_real()),
            W_local.to(dav_dtype),
            matvec_dav,
            bounds,
            n=n_global,
            context=context,
            max_iters=call_max_iter,
            block_size=chosen_nb,
            eps=call_eps,
            n_roots=self.n_roots,
            de_tol=call_de_tol,
            project_fn=project_fn,
            missed_root_check=bool(check.get("missed_root_check", False)),
            verbose=self.verbose,
            checkpoint=checkpoint,
        )

        def global_normalize(v: torch.Tensor) -> torch.Tensor:
            # Ritz vectors are normalized against the local block inside
            # Davidson; renormalize with the global norm before post-processing.
            norm_sq = v.abs().pow(2).sum(dim=1)
            if context.world_size > 1:
                dist.all_reduce(norm_sq, op=dist.ReduceOp.SUM, group=context.group)
            return v / torch.sqrt(norm_sq.clamp_min(0.0)).unsqueeze(1)

        def global_rq(v: torch.Tensor) -> float:
            product = operator(v)
            numerator = torch.vdot(v, product).real.reshape(1).to(torch.float64)
            denominator = torch.vdot(v, v).real.reshape(1).to(torch.float64)
            if context.world_size > 1:
                dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=context.group)
                dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=context.group)
            return float(numerator.item() / denominator.item())

        energies_t, vecs, single, energy_mode = self._finish_narrow_result(
            energies,
            vecs,
            info,
            call_eps,
            e_shift,
            rq_fn=global_rq,
            normalize_fn=global_normalize,
        )
        perf = {
            "davidson_dtype": str(dav_dtype),
            "davidson_fp32_shift": e_shift,
            "davidson_fp32_energy": energy_mode,
            "distributed_matvec": "local",
            "warm": warm,
        }
        if single:
            return (float(energies_t[0]), vecs[0], info), perf
        return (energies_t, vecs, info), perf

    @staticmethod
    def _penalty_stripped_energies(
        energies_raw: torch.Tensor,
        n_found: int,
        penalty: "Optional[tuple[float, float]]",
        penalty_s2: "Optional[list[float]]",
    ) -> list[float]:
        """Physical energies after stripping the ``spin_sq`` penalty.

        Args:
            energies_raw: raw (penalty-augmented) Ritz values.
            n_found: number of converged roots.
            penalty: ``(lambda, target)`` when the penalty ran, else ``None``.
            penalty_s2: per-root ``<S^2>`` expectations when the penalty ran.

        Returns:
            One float per found root, in root order.
        """
        energies_phys = [float(e) for e in energies_raw[:n_found]]
        if penalty is not None and penalty_s2 is not None:
            lam, target = penalty
            energies_phys = [
                energies_phys[root] - lam * (penalty_s2[root] - target) for root in range(n_found)
            ]
        return energies_phys

    def _append_davidson_verbose(
        self,
        verbose_lines: list[str],
        *,
        t_dav: float,
        info: dict,
        t_occ: float,
        line_energies: Any,
        energy: float,
        rdm_note: "Optional[str]" = None,
        s2_value: "Optional[float]" = None,
        spin_sq_final: "Optional[float]" = None,
        spin_sq_target: Any = None,
        spin_lambda: float = 1.0,
    ) -> None:
        """Append the per-subspace Davidson/occupancy/energy diagnostics.

        Args:
            verbose_lines: list the formatted lines are appended to.
            t_dav: Davidson wall time.
            info: Davidson info dict (``n_matvecs``).
            t_occ: occupancy-build wall time.
            line_energies: values printed in the multi-root list.
            energy: reported ground-state energy.
            rdm_note: optional full-RDM note appended to the first line.
            s2_value: optional ``<S^2>`` diagnostic.
            spin_sq_final: final penalty expectation, when the penalty ran.
            spin_sq_target: requested spin target, for the strip line.
            spin_lambda: penalty strength, for the strip line.
        """
        line = (
            f"               Davidson {t_dav:.2f}s "
            f"({info.get('n_matvecs', '?')} matvecs)  | occ {t_occ:.2f}s"
        )
        if rdm_note is not None:
            line += rdm_note
        if s2_value is not None:
            line += f"  | ⟨S²⟩ = {s2_value:.6f}"
        if self.n_roots > 1:
            e_str = ", ".join(f"{float(e):.8f}" for e in line_energies)
            line += f"  | n_roots={self.n_roots} E=[{e_str}]"
        verbose_lines.append(line)
        if spin_sq_final is not None:
            verbose_lines.append(
                f"               ⟨S²⟩ (penalty-strip) = "
                f"{spin_sq_final:.6e}  "
                f"| stripped "
                f"{spin_lambda * (spin_sq_final - float(spin_sq_target)):+.6e} Ha"
            )
        verbose_lines.append(f"               E = {energy:.10f} Ha")

    def _subspace_perf(
        self,
        *,
        sub_idx: int,
        m_a: int,
        m_b: int,
        n: int,
        chosen_nb: int,
        t_setup: float,
        t_dav: float,
        t_occ: float,
        info: dict,
        warm: bool,
        perf_dav: dict,
        energy: float,
        dev: torch.device,
        pair_mode: bool = False,
        rdm_accumulation: "Optional[str]" = None,
    ) -> dict[str, Any]:
        """The per-subspace performance breakdown shared by every solve path.

        Args:
            sub_idx: subspace index.
            m_a, m_b: unique α and β string counts.
            n: basis dimension.
            chosen_nb: Davidson block size.
            t_setup, t_dav, t_occ: setup, Davidson and occupancy wall times.
            info: Davidson info dict.
            warm: whether the solve warm-started.
            perf_dav: the Davidson driver's own perf entries.
            energy: reported ground-state energy.
            dev: device the solve ran on (CUDA peak memory only).
            pair_mode: include the GDB pair-mode marker.
            rdm_accumulation: RDM accumulation path, when the path reports it.

        Returns:
            The ``stats["per_call_log"]`` entry for this subspace.
        """
        mem_peak_gb = torch.cuda.max_memory_allocated(dev) / 1e9 if dev.type == "cuda" else 0.0
        perf = {
            "sub_idx": sub_idx,
            "M_a": m_a,
            "M_b": m_b,
            "N": n,
            "davidson_block_size": chosen_nb,
            "t_setup": t_setup,
            "t_davidson": t_dav,
            "t_occ": t_occ,
            "n_matvecs": info.get("n_matvecs", None),
            "warm": warm,
            # Per missed-root search ({found, energy, n_matvecs}); None when not checked.
            "missed_root_checks": info.get("missed_root_checks"),
            **perf_dav,
            "mem_peak_gb": mem_peak_gb,
            "E": float(energy),
        }
        if pair_mode:
            perf["pair_mode"] = True
        if rdm_accumulation is not None:
            perf["rdm_accumulation"] = rdm_accumulation
        return perf

    def _finish_distributed_local_subspace(
        self,
        *,
        sub_idx: int,
        all_eigvecs: torch.Tensor,
        energies_raw: torch.Tensor,
        info: dict,
        perf_dav: dict,
        adets: List[int],
        bdets: List[int],
        adets_t: torch.Tensor,
        bdets_t: torch.Tensor,
        norb: int,
        n_a: int,
        n_b: int,
        partition,
        operator,
        penalty,
        context,
        dev: torch.device,
        t_setup: float,
        t_dav: float,
        chosen_nb: int,
        warm: bool,
        build_output_states: bool,
        get_integrals_fingerprint: Callable[[], str],
        solver_knob_snapshot: dict,
        verbose_lines: List[str],
    ):
        """Post-process the owner-computes Davidson result without a full-vector copy.

        Occupancies are local weighted sums reduced across ranks. The returned
        amplitudes are handed back by size: below
        ``DISTRIBUTED_HAND_BACK_SMALL_BYTES`` rank 0 gathers the ``σ``-signed row
        blocks to host and builds the ordinary ``SCIState``/``SCIResult``
        objects; above it every rank writes its rows into memmapped ``.npy``
        files under ``TMPDIR`` and rank 0 returns small markers that the parent
        materializes. In both cases no non-zero rank builds a result and rank 0
        holds at most one full host vector per root.

        When the partition is a tile grid and ``compute_spin_squared`` is on,
        the ``<S^2>`` diagnostic is accumulated from the operator's staged
        exchanges (:func:`tiled_spin_squared`) instead of a full-vector
        two-RDM build.

        Returns:
            ``(EigenStates, Optional[TorchSCIState], perf)`` — the empty
            ``EigenStates`` on non-zero ranks (only rank 0's is read back).
        """
        import torch.distributed as dist

        from torch_sci_solver.torch_impl.distributed_solver import (
            DeferredSCIState,
            MemmapAmplitudes,
            total_hand_back_bytes,
            write_row_memmaps,
        )
        from torch_sci_solver.torch_impl.rdm import (
            _inter_to_block_signs,
            make_one_rdm_diag_torch,
        )

        M_a = len(adets)
        M_b = len(bdets)
        N = M_a * M_b
        n_found = all_eigvecs.shape[0]
        work_dtype = all_eigvecs.dtype
        itemsize = 16 if work_dtype.is_complex else 8
        from torch_sci_solver.torch_impl.distributed_davidson import TilePartition

        tiled = isinstance(partition, TilePartition)
        bound = partition.for_rank(context.rank)
        if tiled:
            row_start, row_stop = bound.row_start, bound.row_stop
            col_start, col_stop = bound.col_start, bound.col_stop
        else:
            row_start, row_stop = partition.rows_for_rank(context.rank)
            col_start, col_stop = 0, M_b
        adets_local = adets_t[row_start:row_stop]
        bdets_local = bdets_t[col_start:col_stop]
        sigma_local = _inter_to_block_signs(adets_local, bdets_local, norb).to(dtype=work_dtype)
        t_occ_total = 0.0
        occupancies: List[tuple[np.ndarray, np.ndarray]] = []
        for root_idx in range(n_found):
            t0_occ = _time.perf_counter()
            occ_a_t, occ_b_t = make_one_rdm_diag_torch(
                all_eigvecs[root_idx], adets_local, bdets_local, norb
            )
            if context.world_size > 1:
                dist.all_reduce(occ_a_t, op=dist.ReduceOp.SUM, group=context.group)
                dist.all_reduce(occ_b_t, op=dist.ReduceOp.SUM, group=context.group)
            if context.rank == 0:
                occupancies.append((occ_a_t.cpu().numpy(), occ_b_t.cpu().numpy()))
            t_occ_total += _time.perf_counter() - t0_occ
        penalty_s2: Optional[list[float]] = None
        if penalty is not None:
            # Each root's augmented Ritz value carries lam * (<S^2> - target);
            # strip it with the same tiled expectation the diagnostic uses.
            from torch_sci_solver.torch_impl.tiled_postprocess import (
                tiled_spin_squared as _tiled_spin_squared,
            )

            t0_pen = _time.perf_counter()
            penalty_s2 = [
                float(_tiled_spin_squared(operator, all_eigvecs[r], n_a, n_b).item())
                for r in range(n_found)
            ]
            with self._stats_lock:
                self.stats["t_spin_sq"] += _time.perf_counter() - t0_pen
                self.stats["spin_sq_final"] = penalty_s2[0] if penalty_s2 else None
        t_s2 = 0.0
        s2_value = None
        if self.compute_spin_squared and tiled:
            # The owner-computes path never holds the full vector; the tiled
            # diagnostic reuses the operator's staged exchanges and reduces
            # the partial cross term with one all-reduce.
            from torch_sci_solver.torch_impl.tiled_postprocess import (
                tiled_spin_squared,
            )

            if penalty_s2 is not None:
                # The penalty strip already ran the same expectation.
                s2_value = penalty_s2[0]
            else:
                t0_s2 = _time.perf_counter()
                s2_value = float(tiled_spin_squared(operator, all_eigvecs[0], n_a, n_b).item())
                t_s2 = _time.perf_counter() - t0_s2
        rdm1_t = rdm2_t = None
        rdm_accumulation = "occupancies_only"
        if self.compute_full_rdm and tiled:
            # The tiled builders accumulate the spin-resolved 1- and 2-RDMs
            # from the same staged exchanges. The output-partitioned entries
            # keep only each rank's slice and gather the full tensors to rank
            # 0; the replicated builders remain the fallback when the layout
            # cannot partition or the full output fits the replication bound.
            from torch_sci_solver.torch_impl.tiled_postprocess import (
                can_partition_tiled_rdm,
                tiled_one_rdm_distributed,
                tiled_two_rdm_distributed,
            )

            t0_rdm = _time.perf_counter()
            rdm_accumulation = self._rdm_accumulation_path(
                n_full=3 * int(norb) ** 4,
                can_partition=can_partition_tiled_rdm(operator, order=2, dtype=all_eigvecs.dtype),
                world_size=context.world_size,
                dtype=all_eigvecs.dtype,
            )
            if rdm_accumulation == "partitioned":
                one = tiled_one_rdm_distributed(operator, all_eigvecs[0], replicate=False)
                two = tiled_two_rdm_distributed(operator, all_eigvecs[0], replicate=False)
                full1 = one.gather_to_root(group=context.group)
                full2 = two.gather_to_root(group=context.group)
                if context.rank == 0:
                    assert full1 is not None and full2 is not None
                    rdm1_t = full1[0] + full1[1]
                    rdm2_t = full2[0] + full2[1] + full2[1].permute(2, 3, 0, 1) + full2[2]
                del full1, full2
            else:
                from torch_sci_solver.torch_impl.tiled_postprocess import (
                    tiled_one_rdm,
                    tiled_two_rdm,
                )

                ga, gb = tiled_one_rdm(operator, all_eigvecs[0])
                Gaa, Gab, Gbb = tiled_two_rdm(operator, all_eigvecs[0])
                rdm1_t = ga + gb
                rdm2_t = Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb
                del ga, gb, Gaa, Gab, Gbb
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            with self._stats_lock:
                self.stats["t_rdm"] += _time.perf_counter() - t0_rdm
        with self._stats_lock:
            self.stats["t_occ"] += t_occ_total
            self.stats["spin_squared"].append(s2_value)
            self.stats["t_spin_sq"] += t_s2
            self.stats["rdm_accumulation"] = rdm_accumulation

        signed = (all_eigvecs * sigma_local.reshape(-1)).to(work_dtype)
        hand_back_bytes = total_hand_back_bytes(
            n_found, N, warm_state=build_output_states, itemsize=itemsize
        )
        amplitudes: List[Any] = []
        eigvec_inter: Any = None
        if not _use_memmap_hand_back(
            hand_back_bytes, self._runtime_hand_back, self._runtime_hand_back_bytes
        ):
            stack = [signed]
            if build_output_states:
                stack.append(all_eigvecs[:1].to(work_dtype))
            gather_fn = _gather_local_tile_blocks if tiled else _gather_local_row_blocks
            gathered = gather_fn(torch.cat(stack, dim=0), partition, context)
            if context.rank == 0:
                amplitudes = [gathered[k].reshape(M_a, M_b).numpy() for k in range(n_found)]
                if build_output_states:
                    eigvec_inter = gathered[n_found].clone()
        else:
            blocks = {f"amp_{k}": signed[k] for k in range(n_found)}
            shapes: dict[str, tuple[int, ...]] = {f"amp_{k}": (N,) for k in range(n_found)}
            if build_output_states:
                blocks["eigvec_inter"] = all_eigvecs[0].to(work_dtype)
                shapes["eigvec_inter"] = (N,)
            if tiled:
                from torch_sci_solver.torch_impl.distributed_solver import (
                    write_tile_memmaps,
                )

                markers = write_tile_memmaps(
                    blocks,
                    {name: (M_a, M_b) for name in blocks},
                    row_start=row_start,
                    row_stop=row_stop,
                    col_start=col_start,
                    col_stop=col_stop,
                    context=context,
                )
            else:
                markers = write_row_memmaps(
                    blocks,
                    shapes,
                    row_start=row_start * M_b,
                    row_stop=row_stop * M_b,
                    context=context,
                )
            if context.rank == 0:
                amplitudes = [markers[f"amp_{k}"] for k in range(n_found)]
                if build_output_states:
                    eigvec_inter = markers["eigvec_inter"]

        energies_phys = self._penalty_stripped_energies(energies_raw, n_found, penalty, penalty_s2)
        all_sci_results: List[SCIResultWithConvergence] = []
        output_state = None
        energy = energies_phys[0]
        if context.rank == 0:
            from torch_sci_solver.torch_impl.probe import TorchSCIState

            # A (M, n_limbs) string tensor means the basis is wider than 64
            # bits; keep the exact Python ints in an object array rather than
            # overflowing int64.
            if adets_t.ndim == 2:
                ci_strs_a = np.array(adets, dtype=object)
                ci_strs_b = np.array(bdets, dtype=object)
            else:
                ci_strs_a, ci_strs_b = _ci_strs_arrays(adets, bdets)
            rdm1_np = None if rdm1_t is None else rdm1_t.cpu().numpy()
            rdm2_np = None if rdm2_t is None else rdm2_t.cpu().numpy()
            for root_idx in range(n_found):
                amplitude = amplitudes[root_idx]
                if isinstance(amplitude, MemmapAmplitudes):
                    # qiskit's SCIState validates the amplitude shape, so a
                    # memmap marker cannot live in it: defer construction to
                    # the parent, which maps the file and reshapes it.
                    sci_state: Any = DeferredSCIState(
                        amplitudes=amplitude,
                        ci_strs_a=ci_strs_a,
                        ci_strs_b=ci_strs_b,
                        norb=norb,
                        nelec=(n_a, n_b),
                        shape=(M_a, M_b),
                    )
                else:
                    sci_state = SCIState(
                        amplitudes=amplitude,
                        ci_strs_a=ci_strs_a,
                        ci_strs_b=ci_strs_b,
                        norb=norb,
                        nelec=(n_a, n_b),
                    )
                all_sci_results.append(
                    SCIResultWithConvergence(
                        energy=energies_phys[root_idx],
                        sci_state=sci_state,
                        orbital_occupancies=occupancies[root_idx],
                        rdm1=rdm1_np if root_idx == 0 else None,
                        rdm2=rdm2_np if root_idx == 0 else None,
                        state="converged" if info["converged"] else "not_converged",
                    )
                )
            if build_output_states:
                output_state = TorchSCIState(
                    adets=list(adets),
                    bdets=list(bdets),
                    eigvec_inter=eigvec_inter,
                    energy=energy,
                    n_orb=norb,
                    nelec=(n_a, n_b),
                    integrals_fingerprint=get_integrals_fingerprint(),
                    solver_knobs=dict(solver_knob_snapshot),
                    n_matvecs=info.get("n_matvecs", None),
                    final_residual=info.get("final_residual", None),
                )

        if self.verbose:
            self._append_davidson_verbose(
                verbose_lines,
                t_dav=t_dav,
                info=info,
                t_occ=t_occ_total,
                line_energies=energies_raw[:n_found],
                energy=energy,
            )
        if verbose_lines:
            with self._print_lock:
                for _ln in verbose_lines:
                    print(_ln)

        perf = self._subspace_perf(
            sub_idx=sub_idx,
            m_a=M_a,
            m_b=M_b,
            n=N,
            chosen_nb=chosen_nb,
            t_setup=t_setup,
            t_dav=t_dav,
            t_occ=t_occ_total,
            info=info,
            warm=warm,
            perf_dav=perf_dav,
            energy=energy,
            dev=dev,
            rdm_accumulation=rdm_accumulation,
        )
        if tiled:
            perf["tile_grid"] = [int(partition.grid.p_alpha), int(partition.grid.p_beta)]
        return EigenStates(states=all_sci_results), output_state, perf

    def _finish_pair_local_subspace(
        self,
        *,
        sub_idx: int,
        dav_result,
        perf_dav: dict,
        pair_basis,
        bounds,
        operator,
        penalty,
        context,
        norb: int,
        nelec: tuple[int, int],
        dev: torch.device,
        t_setup: float,
        t_dav: float,
        chosen_nb: int,
        warm: bool,
        build_output_states: bool,
        get_integrals_fingerprint: Callable[[], str],
        solver_knob_snapshot: dict,
        verbose_lines: List[str],
        block_stats: "Optional[dict]" = None,
        partition=None,
        s2_operator: "Optional[Any]" = None,
        penalty_layout: "Optional[str]" = None,
        carry_basis=None,
        carry_positions: "Optional[torch.Tensor]" = None,
    ):
        """Post-process an owner-computes GDB Davidson result on rank 0.

        Pair-mode mirror of :meth:`_finish_distributed_local_subspace`.
        Occupancies are this rank's sum over its owned pairs, all-reduced across
        ranks; the returned amplitudes (and the warm-start snapshot) are handed
        back by size — an in-memory gather to rank 0 below
        ``DISTRIBUTED_HAND_BACK_SMALL_BYTES``, memmapped ``.npy`` files above
        it. A scattered GDB tile hands its owned pair indices back instead of a
        contiguous row range. Only rank 0 builds the
        ``PairSCIState``/``SCIResult`` objects; the other ranks return an empty
        ``EigenStates``. Full RDMs are accumulated on the tiled path from the
        operator's staged exchanges. The ``<S^2>`` diagnostic runs there too on
        a tile, and on the flat block it reads the penalty's all-gathered vector
        instead (one gather per root, no O(K) swap table).

        When the solve ran on a Cartesian superset (``carry_basis`` and
        ``carry_positions``), the hand-back is written in the caller's pair
        layout: each rank keeps only its owned superset pairs that belong to
        ``carry_basis`` and writes them at their original positions, so the
        returned state, its pair strings and the warm-start snapshot describe
        the caller's list, never the padded product.

        Args:
            penalty: ``(λ, target)`` when the solve ran with the ``spin_sq``
                penalty, else ``None``.
            s2_operator: The 1-D
                :class:`~torch_sci_solver.torch_impl.pair_tiled_postprocess.
                PairLocalSpinSquared` when the flat sweep runs (the penalty, the
                ``<S^2>`` diagnostic, or both).
            penalty_layout: ``"tile"`` when the penalty was applied on a pair
                tile (its expectation comes from the staged sweep), ``"1d"``
                when it was applied on the flat owner-computes block, ``None``
                without a penalty.
            carry_basis: The caller's pair basis when ``pair_basis`` is a
                superset; the returned state and strings are built from it.
            carry_positions: ``(carry_basis.n_pairs,)`` superset indices of the
                caller's pairs, increasing. ``None`` keeps the solved basis.

        Returns:
            ``(EigenStates, Optional[TorchPairSCIState], perf)`` — the empty
            ``EigenStates`` on non-zero ranks (only rank 0's is read back).
        """
        import torch.distributed as dist

        from torch_sci_solver.torch_impl.distributed_davidson import TilePartition
        from torch_sci_solver.torch_impl.distributed_solver import (
            total_hand_back_bytes,
            write_pair_tile_memmaps,
            write_row_memmaps,
        )
        from torch_sci_solver.torch_impl.pairs import PairSCIState
        from torch_sci_solver.torch_impl.probe_pairs import TorchPairSCIState
        from torch_sci_solver.torch_impl.rdm_pairs import make_one_rdm_diag_pair_torch

        energies_raw, all_eigvecs, info = dav_result
        if not isinstance(energies_raw, torch.Tensor):
            energies_raw = torch.tensor([energies_raw], dtype=torch.float64)
            all_eigvecs = all_eigvecs.unsqueeze(0)
        n_found = all_eigvecs.shape[0]
        n_pairs = int(pair_basis.n_pairs)
        n_a, n_b = nelec
        work_dtype = all_eigvecs.dtype
        itemsize = 16 if work_dtype.is_complex else 8
        tiled = isinstance(partition, TilePartition)
        if tiled:
            pair_index = bounds.pair_index
            alpha_local, beta_local = _pair_strings_at(pair_basis, pair_index)
            row_start = row_stop = 0
        else:
            row_start, row_stop = bounds.start, bounds.stop
            alpha_local, beta_local = _pair_strings_at(
                pair_basis, torch.arange(row_start, row_stop)
            )
        carry_n = int(carry_basis.n_pairs) if carry_basis is not None else n_pairs
        carry_pos = None
        carry_local_index = None
        carry_local_mask = None
        if carry_positions is not None:
            # The caller's pairs sit at increasing superset indices, so each
            # owned superset index is looked up by searchsorted; the matches
            # keep the same (increasing) order the writers require.
            carry_pos = carry_positions.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
            owned = (
                bounds.pair_index.detach().to(device="cpu", dtype=torch.int64)
                if tiled
                else torch.arange(row_start, row_stop, dtype=torch.int64)
            )
            position = torch.searchsorted(carry_pos, owned)
            carry_local_mask = (position < carry_pos.numel()) & (
                carry_pos[position.clamp_max(carry_pos.numel() - 1)] == owned
            )
            carry_local_index = position[carry_local_mask]

        t_occ_total = 0.0
        occupancies: List[tuple[np.ndarray, np.ndarray]] = []
        for root_idx in range(n_found):
            t0_occ = _time.perf_counter()
            occ_a_t, occ_b_t = make_one_rdm_diag_pair_torch(
                all_eigvecs[root_idx], alpha_local, beta_local, norb
            )
            if context.world_size > 1:
                dist.all_reduce(occ_a_t, op=dist.ReduceOp.SUM, group=context.group)
                dist.all_reduce(occ_b_t, op=dist.ReduceOp.SUM, group=context.group)
            if context.rank == 0:
                occupancies.append((occ_a_t.cpu().numpy(), occ_b_t.cpu().numpy()))
            t_occ_total += _time.perf_counter() - t0_occ
        penalty_s2: Optional[list[float]] = None
        if penalty is not None:
            # Each root's augmented Ritz value carries lam * (<S^2> - target).
            t0_pen = _time.perf_counter()
            if penalty_layout == "tile":
                from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
                    pair_tiled_spin_squared as _pair_tiled_spin_squared,
                )

                penalty_s2 = [
                    float(_pair_tiled_spin_squared(operator, all_eigvecs[r], n_a, n_b).item())
                    for r in range(n_found)
                ]
            else:
                # The flat sweep reads its sources from the operator's
                # all-gather and reduces the expectation across ranks, so every
                # rank calls it once per root.
                assert s2_operator is not None
                penalty_s2 = [
                    float(s2_operator.expectation(all_eigvecs[r])) for r in range(n_found)
                ]
            with self._stats_lock:
                self.stats["t_spin_sq"] += _time.perf_counter() - t0_pen
                self.stats["spin_sq_final"] = penalty_s2[0] if penalty_s2 else None
        s2_value = None
        rdm1_t = rdm2_t = None
        rdm_accumulation = "occupancies_only"
        if tiled and (self.compute_spin_squared or self.compute_full_rdm):
            # The pair-tile builders accumulate from the operator's staged
            # exchanges. Full RDMs use the output-partitioned entries when the
            # layout can partition and the output does not fit the replication
            # bound, so only rank 0 gathers the full tensors; the replicated
            # builders remain the fallback.
            from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
                can_partition_pair_tiled_rdm,
                pair_tiled_one_rdm_distributed,
                pair_tiled_spin_squared,
                pair_tiled_two_rdm_distributed,
            )

            t0_post = _time.perf_counter()
            if self.compute_spin_squared:
                if penalty_s2 is not None:
                    s2_value = penalty_s2[0]
                else:
                    s2_value = float(
                        pair_tiled_spin_squared(operator, all_eigvecs[0], n_a, n_b).item()
                    )
            if self.compute_full_rdm:
                rdm_accumulation = self._rdm_accumulation_path(
                    n_full=3 * int(norb) ** 4,
                    can_partition=can_partition_pair_tiled_rdm(
                        operator, order=2, dtype=all_eigvecs.dtype
                    ),
                    world_size=context.world_size,
                    dtype=all_eigvecs.dtype,
                )
                if rdm_accumulation == "partitioned":
                    one = pair_tiled_one_rdm_distributed(operator, all_eigvecs[0], replicate=False)
                    two = pair_tiled_two_rdm_distributed(operator, all_eigvecs[0], replicate=False)
                    full1 = one.gather_to_root(group=context.group)
                    full2 = two.gather_to_root(group=context.group)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    if context.rank == 0:
                        assert full1 is not None and full2 is not None
                        rdm1_t = full1[0] + full1[1]
                        rdm2_t = full2[0] + full2[1] + full2[1].permute(2, 3, 0, 1) + full2[2]
                    del full1, full2
                else:
                    from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
                        pair_tiled_one_rdm,
                        pair_tiled_two_rdm,
                    )

                    ga, gb = pair_tiled_one_rdm(operator, all_eigvecs[0])
                    Gaa, Gab, Gbb = pair_tiled_two_rdm(operator, all_eigvecs[0])
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    rdm1_t = ga + gb
                    rdm2_t = Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb
                    del ga, gb, Gaa, Gab, Gbb
            with self._stats_lock:
                self.stats["t_rdm"] += _time.perf_counter() - t0_post
        elif self.compute_spin_squared:
            # Flat owner-computes block: the penalty's sweep reads its sources
            # from the matvec's all-gather, so the diagnostic costs one gather
            # and one vdot/all-reduce per root and builds no O(K) table. Full
            # RDMs keep requiring a tile grid.
            if penalty_s2 is not None:
                s2_value = penalty_s2[0]
            else:
                assert s2_operator is not None
                t0_s2 = _time.perf_counter()
                s2_value = float(s2_operator.expectation(all_eigvecs[0]))
                with self._stats_lock:
                    self.stats["t_spin_sq"] += _time.perf_counter() - t0_s2
        with self._stats_lock:
            self.stats["t_occ"] += t_occ_total
            self.stats["spin_squared"].append(s2_value)
            self.stats["rdm_accumulation"] = rdm_accumulation

        # The in-memory hand-back gathers the solved (possibly superset) block
        # even though the returned state keeps the caller's length, so size the
        # decision on the solved basis; the memmap shapes below stay at the
        # caller's length.
        hand_back_bytes = total_hand_back_bytes(
            n_found, n_pairs, warm_state=build_output_states, itemsize=itemsize
        )
        amplitudes: List[Any] = []
        eigvec_full: Any = None
        if not _use_memmap_hand_back(
            hand_back_bytes, self._runtime_hand_back, self._runtime_hand_back_bytes
        ):
            stack = [all_eigvecs]
            if build_output_states:
                stack.append(all_eigvecs[:1])
            if tiled:
                gathered = _gather_local_pair_tile_blocks(torch.cat(stack, dim=0), bounds, context)
            else:
                gathered = _gather_local_row_blocks(
                    torch.cat(stack, dim=0),
                    partition if partition is not None else bounds,
                    context,
                )
            if context.rank == 0:
                amplitudes = [gathered[k].numpy() for k in range(n_found)]
                if build_output_states:
                    eigvec_full = gathered[n_found].clone()
                if carry_pos is not None:
                    amplitudes = [np.asarray(amp)[carry_pos.numpy()] for amp in amplitudes]
                    if eigvec_full is not None:
                        eigvec_full = eigvec_full[carry_pos]
        else:
            blocks = {f"amp_{k}": all_eigvecs[k] for k in range(n_found)}
            shapes: dict[str, tuple[int, ...]] = {f"amp_{k}": (carry_n,) for k in range(n_found)}
            if build_output_states:
                blocks["eigvec_inter"] = all_eigvecs[0]
                shapes["eigvec_inter"] = (carry_n,)
            if carry_local_mask is not None:
                # Write the caller's pairs at their original positions; the
                # padding pairs never reach the file.
                keep = carry_local_mask.to(device=all_eigvecs.device)
                blocks = {name: block[keep] for name, block in blocks.items()}
                markers = write_pair_tile_memmaps(
                    blocks,
                    shapes,
                    pair_index=carry_local_index,
                    context=context,
                )
            elif tiled:
                markers = write_pair_tile_memmaps(
                    blocks,
                    shapes,
                    pair_index=bounds.pair_index,
                    context=context,
                )
            else:
                markers = write_row_memmaps(
                    blocks,
                    shapes,
                    row_start=row_start,
                    row_stop=row_stop,
                    context=context,
                )
            if context.rank == 0:
                amplitudes = [markers[f"amp_{k}"] for k in range(n_found)]
                if build_output_states:
                    eigvec_full = markers["eigvec_inter"]

        energies_phys = self._penalty_stripped_energies(energies_raw, n_found, penalty, penalty_s2)
        all_sci_results: List[SCIResultWithConvergence] = []
        output_state = None
        energy = energies_phys[0]
        if context.rank == 0:
            # The returned state carries the explicit pair strings, so rank 0
            # expands a structural basis here once (and only here). A carried
            # superset solve returns the caller's pairs, not the padded product.
            ci_strs_a_t, ci_strs_b_t = _pair_strings_at(
                carry_basis if carry_basis is not None else pair_basis,
                torch.arange(carry_n),
            )
            ci_strs_a, ci_strs_b = _pair_strs_arrays(
                ci_strs_a_t.cpu().numpy(), ci_strs_b_t.cpu().numpy()
            )
            rdm1_np = None if rdm1_t is None else rdm1_t.cpu().numpy()
            rdm2_np = None if rdm2_t is None else rdm2_t.cpu().numpy()
            all_sci_results = [
                SCIResultWithConvergence(
                    energy=energies_phys[root_idx],
                    sci_state=PairSCIState(
                        amplitudes=amplitudes[root_idx],
                        ci_strs_a=ci_strs_a,
                        ci_strs_b=ci_strs_b,
                        norb=norb,
                        nelec=(n_a, n_b),
                    ),
                    orbital_occupancies=occupancies[root_idx],
                    rdm1=rdm1_np if root_idx == 0 else None,
                    rdm2=rdm2_np if root_idx == 0 else None,
                    state="converged" if info["converged"] else "not_converged",
                )
                for root_idx in range(n_found)
            ]
            if build_output_states:
                output_state = TorchPairSCIState(
                    alpha_of_pair=ci_strs_a_t.cpu().tolist(),
                    beta_of_pair=ci_strs_b_t.cpu().tolist(),
                    eigvec_inter=eigvec_full,
                    energy=energy,
                    n_orb=norb,
                    nelec=(n_a, n_b),
                    integrals_fingerprint=get_integrals_fingerprint(),
                    solver_knobs=dict(solver_knob_snapshot),
                    n_matvecs=info.get("n_matvecs", None),
                    final_residual=info.get("final_residual", None),
                )

        if self.verbose:
            self._append_davidson_verbose(
                verbose_lines,
                t_dav=t_dav,
                info=info,
                t_occ=t_occ_total,
                line_energies=energies_raw[:n_found],
                energy=energy,
            )
        if verbose_lines:
            with self._print_lock:
                for _ln in verbose_lines:
                    print(_ln)

        perf = self._subspace_perf(
            sub_idx=sub_idx,
            m_a=pair_basis.n_unique_alphas,
            m_b=pair_basis.n_unique_betas,
            n=n_pairs,
            chosen_nb=chosen_nb,
            t_setup=t_setup,
            t_dav=t_dav,
            t_occ=t_occ_total,
            info=info,
            warm=warm,
            perf_dav=perf_dav,
            energy=energy,
            dev=dev,
            pair_mode=True,
            rdm_accumulation=rdm_accumulation,
        )
        if carry_basis is not None:
            perf["pair_basis_N"] = carry_n
            perf["pair_superset"] = True
            perf["pair_superset_ratio"] = n_pairs / max(1, carry_n)
        if tiled:
            perf["tile_grid"] = [int(partition.grid.p_alpha), int(partition.grid.p_beta)]
        if penalty_layout is not None:
            perf["spin_penalty_layout"] = penalty_layout
        if block_stats is not None:
            perf["gdb_block_matvec"] = True
            perf["gdb_block_groups_beta"] = block_stats["groups_beta"]
            perf["gdb_block_groups_alpha"] = block_stats["groups_alpha"]
            perf["gdb_block_bytes"] = block_stats["block_bytes"]
        else:
            perf["gdb_block_matvec"] = False
        return EigenStates(states=all_sci_results), output_state, perf

    def _missed_root_check(self, warm: bool) -> dict:
        """The ``missed_root_check`` kwarg for one multi-root solve (empty for one root)."""
        if self.n_roots == 1:
            return {}
        mode = self.davidson_missed_root_check
        return {"missed_root_check": bool(warm) if mode == "auto" else bool(mode)}

    @staticmethod
    def _init_vector_list(init_vectors: Optional[List[Any]], n_sub: int) -> List[Any]:
        """Normalize the per-call ``init_vectors`` kwarg to one entry per subspace."""
        if init_vectors is None:
            return [None] * n_sub
        out = list(init_vectors)
        if len(out) != n_sub:
            raise ValueError(
                f"init_vectors has length {len(out)} but ci_strings has {n_sub} "
                "subspace(s); pass one entry (or None) per subspace."
            )
        return out

    @staticmethod
    def _record_gpu_peak_memory(perf: dict, dev: torch.device) -> None:
        """Add this subspace's CUDA peak memory (bytes / 1e9) to its perf dict.

        ``_run_one`` resets the allocator peak counters just before the solve,
        so the values cover exactly one subspace. The keys are omitted on CPU
        rather than reported as zero. In a distributed run only rank 0's stats
        reach the parent, so these peaks describe rank 0's device, not the
        maximum across ranks.
        """
        if dev.type != "cuda":
            return
        perf["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(dev) / 1024**3
        perf["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved(dev) / 1024**3

    def __call__(
        self,
        ci_strings,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        spin_sq: Optional[float] = None,
        **kwargs,
    ):
        """Diagonalize H in subspaces. Drop-in for
        ``qiskit_addon_sqd.fermion.solve_sci_batch``.

        Args:
            ci_strings: list of (strings_a, strings_b) pairs.
            one_body_tensor, two_body_tensor, norb, nelec: standard SCI inputs.
            spin_sq: target S² value. ``None`` (default) → no penalty.
                A float → enable the S² penalty biasing Davidson toward
                the ``⟨S²⟩ = spin_sq`` sector. The reported energy is the
                physical ⟨ψ|H|ψ⟩, with the penalty contribution stripped
                before return. See the factory docstring for details.
            **kwargs: per-call overrides. Recognized:
                  - ``max_davidson_iters`` → Davidson outer restart count
                    (overrides this solver's ``self.max_davidson_iters`` for this call)
                  - ``tol``       → Davidson ‖R‖ convergence threshold
                    (overrides this solver's ``self.davidson_tolerance`` for this call)
                  - ``de_tol``    → opt-in Davidson eigenvalue-stability check
                    (overrides this solver's ``self.davidson_de_tolerance``
                    for this call); ``None`` disables it, a value's
                    effective threshold is its sqrt (recommended: ``tol``'s
                    value)
                  - ``spin_penalty_lambda`` → penalty strength λ (default 1.0)
                  - ``init_vectors`` → per-subspace Davidson start vector: a list
                    with one entry per ``ci_strings`` subspace, each ``None``
                    (cold start) or a 1-D array laid out like the
                    ``sci_state.amplitudes`` this mode returns, indexed by the
                    strings passed in this call — product mode: flattened
                    ``(len(strings_a), len(strings_b))``, spin-block signs; GDB:
                    one entry per ``(strings_a[i], strings_b[i])`` pair,
                    interleaved-Wick signs; spinless: one entry per string.
                    Strings dropped from the previous basis are simply absent;
                    strings the caller did not pass start at 0. The vector is
                    reordered to the engine's internal order, projected and
                    normalized; one with less than 1e-8 of its own norm left
                    after projection falls back to the cold start (its overall
                    scale does not matter). ``stats["per_call_log"]`` records
                    ``warm`` per subspace. Mutually exclusive with
                    ``probe_state``.
                  - ``probe_method='warm_start'`` / ``probe_state`` /
                    ``probe_kwargs`` → cache and reuse the converged
                    ground-state eigenvector on a superset basis. The cached
                    state type depends on the path: ``TorchSCIState`` (TPB),
                    ``TorchPairSCIState`` (GDB) or ``TorchSpinlessSCIState``
                    (a scalar ``nelec``); the types are not interchangeable.
                    A cold call with ``probe_method`` returns
                    ``(results, states)``; a warm extension returns
                    ``(results, states)`` only when
                    ``probe_kwargs={'build_output_state': True}``. The
                    spinless path has no alpha/beta split and therefore no HF
                    augmentation (``include_hf_in_core`` is ignored with a
                    warning); its ``spin_sq``/``compute_spin_squared``
                    refusals are physics limitations reported by
                    ``preflight`` and ``capabilities()``.
                Other kwargs are silently ignored — including pyscf-only knobs
                (``max_cycle``, ``nroots``, ``max_space``, …) that have no
                equivalent in this matvec/Davidson stack. Note ``max_cycle`` was
                previously accepted here as an alias for ``max_davidson_iters``; it is
                not any more, and passing it now has NO effect.
        """
        with _deterministic_algorithms(self.deterministic):
            return self._call_body(
                ci_strings,
                one_body_tensor,
                two_body_tensor,
                norb,
                nelec,
                spin_sq=spin_sq,
                **kwargs,
            )

    def _call_body(
        self,
        ci_strings,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        spin_sq: Optional[float] = None,
        **kwargs,
    ):
        """The body of :meth:`__call__`, run inside the solver's determinism scope."""
        # Unknown-kwarg policy first, so an "error" policy raises here in the
        # parent rather than inside a spawned rank or a distributed worker.
        kwargs = self._filter_unknown_call_kwargs(kwargs)

        # Complex/real dtype consistency: opt-in only (see TorchSCISolver's
        # ``dtype`` docstring). A caller passing complex hcore/eri to a
        # real-dtype solver would otherwise have the imaginary part silently
        # dropped several calls downstream (e.g. inside
        # setup_integrals_from_pyscf_h1_eri_vec) — raise here instead, right
        # where the mismatch is easiest to diagnose.
        if not self.dtype.is_complex and (
            np.iscomplexobj(one_body_tensor) or np.iscomplexobj(two_body_tensor)
        ):
            raise TypeError(
                f"one_body_tensor/two_body_tensor is complex but this solver was "
                f"constructed with dtype={self.dtype} (real). Pass "
                "dtype=torch.complex128 (or complex64) to TorchSCISolver(...) to "
                "solve a complex Hermitian Hamiltonian — the imaginary part is "
                "never silently dropped."
            )

        # Opt-in distributed dispatch. ``_distributed_config is None`` for the
        # default solver (and for world_size <= 1), so this block is skipped
        # entirely and the call below is the existing single-process path.
        # ``_inside_distributed`` guards the recursive call a spawned/in-process
        # rank makes: that inner call must run the normal solver body with the
        # distributed Davidson, not try to launch again.
        if self._distributed_config is not None and not self._inside_distributed:
            from torch_sci_solver.torch_impl.distributed_solver import (
                NOT_DISTRIBUTED,
                run_distributed_call,
            )

            distributed_result = run_distributed_call(
                self,
                ci_strings,
                one_body_tensor,
                two_body_tensor,
                norb,
                nelec,
                spin_sq,
                kwargs,
            )
            if distributed_result is not NOT_DISTRIBUTED:
                return distributed_result

        # Spinless (flat spin-orbital) dispatch: a *scalar* nelec (total N, not
        # an (n_a, n_b) tuple) signals the genuinely spinless path — no α/β
        # factorization, one flat determinant per ci_strings entry — mirroring
        # GeneralAlgebraSolverBase._is_spinless_request. Detected per call (not
        # a constructor mode= the caller must remember to pass), and dispatched
        # before the unconditional ``n_a, n_b = nelec`` unpack below, which
        # would otherwise raise on a scalar. See _solve_spinless's docstring
        # for the feature scope: multi-root, init_vectors and warm-start probe
        # states are supported; spin_sq / compute_spin_squared are rejected as
        # a physics limitation (no alpha/beta split defines S^2).
        if isinstance(nelec, (int, np.integer)):
            # The spinless path runs on one process/device and reads no GDB
            # options; reject configurations that would otherwise be silently
            # ignored here instead of appearing to take effect.
            if len(self.devices) > 1:
                raise ValueError(
                    "the spinless path runs on one device per process; devices= "
                    "(batch-parallel multi-GPU) is not supported with a scalar nelec. "
                    "Use a single device= or call the solver once per device."
                )
            if self.mode == "gdb" and self._gdb_kwargs_supplied:
                raise ValueError(
                    "a scalar nelec selects the flat spinless path, where GDB gdb_kwargs "
                    'have no effect. Drop gdb_kwargs (and mode="gdb") for a spinless solve.'
                )
            return self._solve_spinless(
                ci_strings,
                one_body_tensor,
                two_body_tensor,
                norb,
                int(nelec),
                spin_sq=spin_sq,
                **kwargs,
            )

        # spin_sq=None: no penalty, code path bit-identical to v0.10.18.
        # spin_sq=<float>: augmented Davidson on H' = H + λ·(S² − spin_sq).
        # We validate the type here; the actual S² construction happens
        # later, gated behind ``if spin_sq is not None``.
        if spin_sq is not None:
            try:
                spin_sq_target = float(spin_sq)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"spin_sq must be None or convertible to float, got {spin_sq!r}"
                ) from exc

        # --- probe-method dispatch (warm-start etc.) ---
        # Pop probe kwargs early so they don't interact with the pyscf
        # forward-compat layer below. When ``probe_method`` is None, the
        # rest of this function behaves identically to v0.11.0 — same
        # return type (list[SCIResult]), no state objects built, zero
        # overhead. When set, we track the (probe_method, probe_kwargs,
        # probe_state) tuple through the per-subspace loop and decide at
        # the end whether to package + return states.
        start = self._normalize_start_kwargs(kwargs, ci_strings)
        probe_method = start["probe_method"]
        probe_kwargs_in = start["probe_kwargs"]
        probe_state_in = start["probe_state"]
        init_vector_list = start["init_list"]
        probe_state_list = start["probe_list"]

        # Whether to build & return state objects from this call. Cold
        # (no probe_state) → always build. Warm (probe_state given) →
        # only build if explicitly requested via probe_kwargs.
        if probe_method == "warm_start":
            if probe_state_in is None:
                # Cold solve with probe enabled → caller wants states back.
                build_output_states = True
            else:
                build_output_states = bool(probe_kwargs_in.get("build_output_state", False))
        else:
            build_output_states = False

        # Snapshot of solver-construction knobs that materially affect
        # the converged eigenvalue. Recorded into TorchSCIState; checked
        # on warm-extend. See probe._KNOB_FIELDS / _KNOB_FIELDS_SOFT.
        knobs = self._solver_knob_values()
        _solver_knob_snapshot = {
            "apply_ab_projection": knobs["apply_ab_projection"],
            "symmetrize_spin": False,  # adapter doesn't expose this yet
            "seed_strategy": knobs["seed_strategy"],
            "davidson_tolerance": knobs["davidson_tolerance"],
            "davidson_de_tolerance": knobs["davidson_de_tolerance"],
            "davidson_block_size": knobs["davidson_block_size"],
            "max_davidson_iters": knobs["max_davidson_iters"],
            "chunk_size": knobs["chunk_size"],
        }

        # Per-call overrides (with sensible falls-through to the values bound at
        # solver-construction time). ``max_davidson_iters`` is the outer Davidson
        # restart count, NOT a flat matvec budget: it goes straight to
        # ``davidson_torch``'s ``max_davidson_iters``, and total work scales as
        # ``max_davidson_iters * davidson_block_size``. (Formerly accepted under the pyscf-style
        # name ``max_cycle``, which was misleading — nothing here calls pyscf.)
        # Rejected rather than ignored: ``max_cycle`` used to drive this knob, so
        # silently dropping it would quietly cut the Davidson budget (a caller
        # passing max_cycle=200 would fall back to the default) and surface only
        # as an unexplained convergence regression.
        davidson = self._pop_davidson_overrides(kwargs, validate=False)
        call_max_iter = davidson["call_max_iter"]
        call_eps = davidson["call_eps"]
        call_de_tol = davidson["call_de_tol"]
        call_spin_lambda = davidson["call_spin_lambda"]

        t_call_start = _time.perf_counter()
        # Recovery gap = time since last sci_solver call (= qiskit_addon_sqd's
        # recovery + outer-loop bookkeeping). For the first call this is 0.
        if self._last_exit is not None:
            gap = t_call_start - self._last_exit
            self.stats["t_recovery_gap"] += gap
            if self.verbose:
                print(f"  [recovery] qiskit recovery+bookkeeping: {gap:.2f}s")
        self.stats["n_calls"] += 1

        # --- integrals (cache by array identity; qiskit reuses the arrays) ---
        # qiskit's outer loop typically reuses the same (one_body, two_body)
        # arrays across iterations, so id()-based caching works. But some
        # outer loops deep-copy or recreate the arrays each iteration —
        # id() changes, cache misses, and (worse) we'd accumulate stale
        # entries indefinitely without LRU-1 eviction below.
        #
        # Fallback: when id() misses, check byte-content equality against
        # the cached entry. The integrals don't change across iterations of
        # an SQD loop (same molecular Hamiltonian throughout), so a content
        # match means we can reuse. Cheap: O(L²+L⁴) byte compare vs the
        # O(L⁴) torch tensor build it avoids.
        #
        # Cache value layout (7-tuple):
        #   (I0, I1, I2, integrals_by_dev, h1_view, eri_view, fp_or_None)
        # where ``integrals_by_dev`` is a ``dict[torch.device, TorchIntegrals]``
        # so multi-GPU dispatch can share the (h1, eri) pair across devices
        # without rebuilding the CPU-side scalar helpers, and
        # ``fp_or_None`` is a SHA-256 hex string when the warm-start probe
        # path has previously computed it on this entry, else None.
        # Lazily computed: cold-only callers never pay for the hash.
        t0 = _time.perf_counter()
        cache_key = (id(one_body_tensor), id(two_body_tensor))
        _hit = None
        if cache_key in self._integrals_cache:
            _hit = self._integrals_cache[cache_key]
        else:
            # Content-equality fallback.
            for k, entry in self._integrals_cache.items():
                _h1_view = entry[4]
                _eri_view = entry[5]
                if (
                    _h1_view.shape == one_body_tensor.shape
                    and _eri_view.shape == two_body_tensor.shape
                    and np.array_equal(_h1_view, one_body_tensor)
                    and np.array_equal(_eri_view, two_body_tensor)
                ):
                    _hit = entry
                    break
            if _hit is not None:
                I0, I1, I2, integrals_by_dev, _h1_view, _eri_view, _fp = _hit
            else:
                I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(
                    _as_integrals_array(one_body_tensor),
                    _as_integrals_array(two_body_tensor),
                    0.0,
                )
                # Build the on-device bundle in the PARENT only for single-device
                # runs. For multi-GPU the parent must stay off-GPU (workers build
                # their own device integrals from the CPU (I0,I1,I2) bundle) —
                # allocating here would init a parent CUDA context on devices[0]
                # and, under Exclusive_Process compute mode, block that GPU's
                # worker from ever starting.
                if len(self.devices) == 1:
                    integrals_by_dev = {
                        self.device: to_torch_integrals(
                            I0,
                            I1,
                            I2,
                            device=self.device,
                            dtype=self.dtype,
                            compact_eri=_resolve_compact_eri(
                                self.compact_eri,
                                matvec_impl=self.matvec_impl,
                                mode=self.mode,
                                n_spinorb=int(I1.n_spinorb),
                            ),
                        )
                    }
                else:
                    integrals_by_dev = {}
                self.stats["t_integrals_build"] += _time.perf_counter() - t0

            # LRU-1: evict all prior entries before inserting the new one.
            # The integrals don't change within an SQD loop, so we never
            # need more than the most-recent entry. Without this, every
            # iteration with a new id() (deep-copied arrays) leaks a full
            # integrals tensor copy to GPU — ~13 MB per iter at 36 orbitals,
            # ~180 MB after 14 iters, plus memory fragmentation that
            # compounds with each Davidson allocation.
            if self._integrals_cache:
                self._integrals_cache.clear()
            # Keep numpy views so future iterations can content-check.
            # Fingerprint slot starts as None; populated on first probe call.
            self._integrals_cache[cache_key] = (
                I0,
                I1,
                I2,
                integrals_by_dev,
                _as_integrals_array(one_body_tensor),
                _as_integrals_array(two_body_tensor),
                None,
            )
            _hit = self._integrals_cache[cache_key]
        # Unpack the hit (for both branches).
        I0, I1, I2, integrals_by_dev, _h1_view, _eri_view, _cached_fp = _hit

        # Ensure the (single) dispatch device has an on-device integrals bundle.
        # SINGLE-DEVICE ONLY: the multi-GPU path uses one process per GPU and the
        # parent must stay OFF-GPU (Exclusive_Process compute mode allows one
        # context per device) — each worker builds its own ``TorchIntegrals``
        # from the shipped CPU ``(I0, I1, I2)`` bundle, so the parent must not
        # allocate any device tensor here.
        if len(self.devices) == 1:
            for _dev in self.devices:
                if _dev not in integrals_by_dev:
                    _tbuild = _time.perf_counter()
                    integrals_by_dev[_dev] = to_torch_integrals(
                        I0,
                        I1,
                        I2,
                        device=_dev,
                        dtype=self.dtype,
                        compact_eri=_resolve_compact_eri(
                            self.compact_eri,
                            matvec_impl=self.matvec_impl,
                            mode=self.mode,
                            n_spinorb=int(I1.n_spinorb),
                        ),
                    )
                    self.stats["t_integrals_build"] += _time.perf_counter() - _tbuild

        # Helper closure to get the (possibly cached) fingerprint of the
        # current integrals arrays. Computed lazily — cold-only callers
        # (no probe_method) never pay for the SHA-256. Guarded by
        # _stats_lock so parallel subspaces don't double-compute the hash
        # on first warm-extend of a new (h1, eri) pair.
        def _get_integrals_fingerprint() -> str:
            """Get integrals fingerprint."""
            nonlocal _hit
            with self._stats_lock:
                _, _, _, _, _h1_v, _eri_v, fp = _hit
                if fp is None:
                    fp = _integrals_fingerprint(_h1_v, _eri_v, 0.0)
                    # Mutate the cache in place: rewrite the tuple with fp filled.
                    self._integrals_cache[cache_key] = (
                        _hit[0],
                        _hit[1],
                        _hit[2],
                        _hit[3],
                        _h1_v,
                        _eri_v,
                        fp,
                    )
                    _hit = self._integrals_cache[cache_key]
                return fp

        n_a, n_b = nelec
        n_sub = len(ci_strings)
        # Pre-sized so parallel workers can write to their sub_idx slot
        # without a lock. Sequential path replaces None entries in-order.
        results: List[Optional[EigenStates]] = [None] * n_sub
        # Probe-method outputs: parallel list of TorchSCIState | None, one
        # per subspace. Populated only when build_output_states is True
        # (i.e., warm_start with no probe_state, or warm_start with
        # build_output_state=True). Returned as the second element of
        # the result tuple in that case.
        output_states: List[Optional[TorchSCIState]] = [None] * n_sub

        # Per-call performance breakdown — one entry per subspace, aggregated
        # below into a single dict appended to self.stats['per_call_log']. Lets
        # us see *which* of t_setup/t_davidson/t_occ grows with iteration,
        # plus basis-size and memory growth. Sample inspection::
        #
        #     for r in solver.stats['per_call_log']:
        #         print(f"call {r['call_idx']}: N={r['N']} "
        #               f"setup={r['t_setup']:.1f}s davidson={r['t_davidson']:.1f}s "
        #               f"matvecs={r['n_matvecs']} mem_peak={r['mem_peak_gb']:.1f}GB")
        per_call_subspaces: List[Optional[dict]] = [None] * n_sub

        # Per-subspace worker used by both the sequential and multi-GPU
        # branches. Pinned to ``dev`` via ``torch.cuda.device(dev)`` so every
        # allocation and CUDA sync inside ``_solve_one_subspace`` lands on
        # the right GPU, and the post-subspace ``empty_cache()`` flushes
        # the correct allocator.
        def _run_one(sub_idx: int, strings_a, strings_b, dev: torch.device):
            """Run one subspace on the specified device."""
            with self._stats_lock:
                self.stats["n_subspaces"] += 1
            probe_state_this: Optional[TorchSCIState] = probe_state_list[sub_idx]
            _ctx = torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()
            with _ctx:
                if dev.type == "cuda":
                    # Each record covers one subspace: drop the previous
                    # subspace's peaks before solving. In distributed mode every
                    # rank resets, but only rank 0's perf reaches the parent.
                    torch.cuda.reset_peak_memory_stats(dev)
                try:
                    eigen_states, output_state, perf = self._solve_one_subspace(
                        sub_idx=sub_idx,
                        strings_a=strings_a,
                        strings_b=strings_b,
                        probe_state_this=probe_state_this,
                        one_body_tensor=one_body_tensor,
                        two_body_tensor=two_body_tensor,
                        norb=norb,
                        nelec=nelec,
                        n_a=n_a,
                        n_b=n_b,
                        spin_sq=spin_sq,
                        spin_sq_target=(float(spin_sq_target) if spin_sq is not None else None),
                        integrals=integrals_by_dev[dev],
                        get_integrals_fingerprint=_get_integrals_fingerprint,
                        probe_method=probe_method,
                        probe_kwargs_in=probe_kwargs_in,
                        build_output_states=build_output_states,
                        solver_knob_snapshot=_solver_knob_snapshot,
                        call_max_iter=call_max_iter,
                        call_eps=call_eps,
                        call_de_tol=call_de_tol,
                        call_spin_lambda=call_spin_lambda,
                        device=dev,
                        init_vector=init_vector_list[sub_idx],
                    )
                    self._record_gpu_peak_memory(perf, dev)
                finally:
                    # Flush the caching allocator so the next subspace on
                    # this GPU sees the freed bytes. Correctly scoped to
                    # ``dev`` because we're inside torch.cuda.device(dev).
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
            return sub_idx, eigen_states, output_state, perf

        if len(self.devices) == 1:
            # Historical sequential path. Bit-identical to pre-multi-GPU
            # behaviour (single-slot integrals, single device, no locks
            # contended).
            _only_dev = self.devices[0]
            for sub_idx, (strings_a, strings_b) in enumerate(ci_strings):
                _, eigen_states, output_state, perf = _run_one(
                    sub_idx, strings_a, strings_b, _only_dev
                )
                results[sub_idx] = eigen_states
                output_states[sub_idx] = output_state
                per_call_subspaces[sub_idx] = perf
        else:
            # Multi-GPU: one PROCESS per device (own GIL each), so the
            # host-heavy per-subspace Python (setup enumeration + dispatch)
            # genuinely overlaps instead of serializing on the GIL as it did
            # with threads. The pool is persistent across __call__s (worker
            # startup is ~10s) and lives on the solver instance.
            if self._worker_pool is None:
                self._worker_pool = _GpuWorkerPool(self.devices, self._worker_cfg())
                self._worker_pool.start()

            # Workers build their own device integrals from the CPU bundle,
            # cached per fingerprint. The pool decides when to actually attach
            # the ~13 MB bundle (it ships until every worker has confirmed this
            # fp, then stops) — see _GpuWorkerPool.submit_batch. We leave
            # integrals_bundle=None here; submit_batch fills it as needed.
            fp = _get_integrals_fingerprint()
            cpu_bundle = (I0, I1, I2)

            tasks = []
            for i, (strings_a, strings_b) in enumerate(ci_strings):
                tasks.append(
                    _Task(
                        sub_idx=i,
                        strings_a=strings_a,
                        strings_b=strings_b,
                        fp=fp,
                        integrals_bundle=None,
                        one_body_tensor=one_body_tensor,
                        two_body_tensor=two_body_tensor,
                        norb=norb,
                        nelec=nelec,
                        n_a=n_a,
                        n_b=n_b,
                        spin_sq=spin_sq,
                        spin_sq_target=(float(spin_sq_target) if spin_sq is not None else None),
                        probe_method=probe_method,
                        probe_kwargs_in=probe_kwargs_in,
                        build_output_states=build_output_states,
                        solver_knob_snapshot=_solver_knob_snapshot,
                        call_max_iter=call_max_iter,
                        call_eps=call_eps,
                        call_de_tol=call_de_tol,
                        call_spin_lambda=call_spin_lambda,
                        probe_state_this=probe_state_list[i],
                        call_idx=self.stats["n_calls"],
                        init_vector=init_vector_list[i],
                    )
                )

            results_by_idx = self._worker_pool.submit_batch(
                tasks, fp, cpu_bundle
            )  # raises on error

            for sub_idx, r in results_by_idx.items():
                results[sub_idx] = r.sci_result
                output_states[sub_idx] = r.output_state
                per_call_subspaces[sub_idx] = r.perf
                d = r.stats_delta
                for _k in _ACCUM_STATS_KEYS:
                    self.stats[_k] += d.get(_k, 0.0)
                self.stats["n_subspaces"] += d.get("n_subspaces", 0)
                self.stats["spin_squared"].extend(d.get("spin_squared", []))
                if d.get("spin_sq_final") is not None:
                    self.stats["spin_sq_final"] = d["spin_sq_final"]

        # --- aggregate per-call summary ---
        # One entry per sci_solver call, holding both per-subspace details
        # and call-level totals. This is the structure to scan when
        # diagnosing performance: t_setup growing with call_idx points at
        # helpers/hii build cost; t_davidson growing points at matvec or
        # Krylov subspace cost; t_recovery_gap growing points at the
        # qiskit-side outer loop, not us.
        self.stats["per_call_log"].append({
            "call_idx": self.stats["n_calls"],
            "n_subspaces": len(per_call_subspaces),
            "subspaces": per_call_subspaces,
            "t_call_total": _time.perf_counter() - t_call_start,
            "t_recovery_gap": (
                t_call_start - self._last_exit if self._last_exit is not None else 0.0
            ),
            "t_setup_call": sum(s["t_setup"] for s in per_call_subspaces),
            "t_davidson_call": sum(s["t_davidson"] for s in per_call_subspaces),
            "t_occ_call": sum(s["t_occ"] for s in per_call_subspaces),
            "mem_peak_call_gb": max((s["mem_peak_gb"] for s in per_call_subspaces), default=0.0),
        })

        if self.verbose:
            # Print the FULL cumulative perf table after every call so the
            # user sees the growth trend in real time, and the last printout
            # in the log is automatically the final summary — no need to
            # access ``solver.stats`` or call any helper at the end of the
            # run.
            print()
            print(f"  [perf] cumulative through call {self.stats['n_calls']}:")
            for line in self.format_perf_summary().splitlines():
                print(f"  [perf] {line}")
            print()

        self._last_exit = _time.perf_counter()
        self.stats["t_total_in_solver"] += self._last_exit - t_call_start

        # ``results`` / ``per_call_subspaces`` are pre-sized with None slots
        # so parallel workers can write to their sub_idx directly; every
        # slot is guaranteed to be populated by the time we get here
        # (the multi-GPU branch collects all futures and the sequential
        # branch fills them in-order). Narrow the type for callers.
        results_out = cast(List[EigenStates], results)

        # When probe_method was set AND we built output states, return a
        # (results, states) tuple per the probe API contract:
        #   - cold solve with probe_method='warm_start' → (results, states)
        #     so the caller has the cache to warm-extend later.
        #   - warm-extend (probe_state given) → just `results` by default;
        #     opt-in to (results, states) via probe_kwargs['build_output_state']=True
        #     when the caller wants to chain extensions.
        # The return type is list[EigenStates] — each element holds all solved
        # roots for one subspace (states[0] = ground state, states[1:] = excited).
        # Callers that need only the ground-state SCIResult (e.g. the qiskit
        # outer loop) should call via EnergyCalculatorBase.sci_solver_callable
        # or extract states[0] themselves (see qiskit.py shims).
        if probe_method is not None and build_output_states:
            return results_out, output_states
        return results_out

    def _solve_one_subspace(
        self,
        *,
        sub_idx: int,
        strings_a,
        strings_b,
        probe_state_this,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        n_a: int,
        n_b: int,
        spin_sq,
        spin_sq_target,
        integrals,
        get_integrals_fingerprint,
        probe_method,
        probe_kwargs_in: dict,
        build_output_states: bool,
        solver_knob_snapshot: dict,
        call_max_iter: int,
        call_eps: float,
        call_de_tol: Optional[float],
        call_spin_lambda: float,
        device: "Optional[torch.device]" = None,
        init_vector: Any = None,
    ):
        """Solve a single subspace. Returns (SCIResult, Optional[State], perf_dict).

        Extracted in the v0.11.0 phase-2 refactor from the inline body of
        ``__call__``'s per-subspace loop. Identical behavior to the inline
        form; the method just makes the 16 sections (dedup → HF aug →
        warm-start order → connected-seed reorder → helpers/hii/W → spin_sq
        penalty setup → davidson_block_size → Davidson → strip-penalty → occupancies
        → σ bridge → state snapshot → ⟨S²⟩ diag → full RDMs → package
        SCIResult → build TorchSCIState → perf dict) easier to navigate.

        Side effects:
          - Mutates ``self.stats`` (per-component time accumulators,
            ``spin_squared`` list, etc). All mutations are wrapped in
            ``self._stats_lock`` so the multi-GPU pool cannot race.
          - Mutates the (caller-supplied) ``get_integrals_fingerprint``
            closure's underlying cache when first warm-extend on a new
            integrals pair.
          - Buffers all subspace stdout output into a local list and
            flushes it under ``self._print_lock`` at the end so multi-GPU
            runs still produce readable, per-subspace-grouped output.

        Args:
            device: the CUDA/CPU device this subspace runs on. Defaults to
                ``self.device`` (the primary). Under multi-GPU dispatch,
                ``__call__`` passes the worker's assigned GPU here so every
                per-subspace tensor allocation and CUDA sync lands on the
                right device.

        Returns:
            eigen_states : EigenStates for this subspace, holding all solved
                roots (``states[0]`` = ground state, ``states[1:]`` = excited).
            output_state : Optional[TorchSCIState] if ``build_output_states``
                else ``None``.
            perf : dict with per-subspace timings/sizes; appended to
                ``per_call_subspaces`` by the caller.
        """
        # Local rebinding for the imports/types used inside the body —
        # these were factory-locals in v0.10 and __call__-locals in v0.11.0
        # phase-1; we re-import here at method scope so the body code is
        # unchanged.

        dev = device if device is not None else self.device
        # Accumulator for verbose / user-notice output; flushed under
        # self._print_lock at the end of the method so multi-GPU runs
        # don't interleave lines from different subspaces.
        verbose_lines: list[str] = []

        output_state = None  # set below if build_output_states=True

        # Pair-mode branch — bypass the product-mode dedup/HF/warm-start/
        # reorder path entirely and dispatch to _solve_one_subspace_pair
        # below, which routes through the pair-basis kernels
        # (mult_pairs / pair_helpers / rdm_pairs). Features not yet ported
        # to pair mode raise a clear error at the top of that method so
        # users see which phase is still missing rather than an internal
        # shape mismatch. Return value shape matches the product-mode
        # branch (EigenStates, Optional[TorchSCIState], perf-dict), with
        # one SCIResult per root in either mode.
        if self.mode == "gdb":
            return self._solve_one_subspace_pair(
                sub_idx=sub_idx,
                strings_a=strings_a,
                strings_b=strings_b,
                probe_state_this=probe_state_this,
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                norb=norb,
                nelec=nelec,
                n_a=n_a,
                n_b=n_b,
                spin_sq=spin_sq,
                spin_sq_target=spin_sq_target,
                integrals=integrals,
                get_integrals_fingerprint=get_integrals_fingerprint,
                probe_method=probe_method,
                probe_kwargs_in=probe_kwargs_in,
                build_output_states=build_output_states,
                solver_knob_snapshot=solver_knob_snapshot,
                call_max_iter=call_max_iter,
                call_eps=call_eps,
                call_de_tol=call_de_tol,
                call_spin_lambda=call_spin_lambda,
                device=device,
                init_vector=init_vector,
            )

        # Validated against the caller's strings, before the dedup below changes
        # the basis size.
        if init_vector is not None:
            init_vector = _validate_init_vector(
                init_vector,
                len(strings_a) * len(strings_b),
                dtype=self.dtype,
                sub_idx=sub_idx,
                n_roots=self.n_roots,
            )

        # Dedup matches SBD's LoadAlphaDets → sort_bitarray → std::unique
        # behaviour. qiskit_addon_sqd's bitstring_matrix_to_ci_strs
        # deduplicates the input bitstrings BEFORE recovery, but
        # recover_configurations can probabilistically bit-flip two
        # different noisy bitstrings to the same recovered CI string.
        # Those post-recovery duplicates only appear here. Without this
        # dedup, ``adets[i] == adets[j]`` creates duplicate basis rows in
        # H_B (rank-deficient), Davidson's Gram-Schmidt orthogonalizes
        # against near-redundant vectors, and the resulting Ritz vectors
        # are not clean H_B eigenstates — observed empirically as
        # spurious ⟨S²⟩ on closed-shell systems at large basis.
        #
        # Strings wider than one int64 word (norb > 64) are normalized to
        # (M, n_limbs) limb matrices up front; everything below that is
        # representation-agnostic then works on the canonical Python-int
        # lists and materializes the tensors once, at ``adets_t``.
        adets_limbs_in = _ci_string_limbs(strings_a, norb)
        bdets_limbs_in = _ci_string_limbs(strings_b, norb)
        limb_strings = adets_limbs_in is not None
        if limb_strings:
            n_limbs = _limb_count(norb)
            # Caller-order Python ints, kept for the first-occurrence lookup
            # and the warm-start set checks below.
            adets_caller = _limb_rows_as_ints(adets_limbs_in)
            bdets_caller = _limb_rows_as_ints(bdets_limbs_in)
            adets = sorted(set(adets_caller))
            bdets = sorted(set(bdets_caller))
        else:
            adets = sorted(set(_ci_string_words(strings_a)))
            bdets = sorted(set(_ci_string_words(strings_b)))

        # --- input validation ---
        # Reject malformed CI bases up front with a clear ``ValueError``,
        # rather than letting the downstream excitation-table builder
        # crash with ``IndexError: list index out of range`` (a real bug
        # encountered in production: an RL reward function fed raw
        # LLM-sampled bitstrings without filtering for particle count,
        # producing α/β strings with popcounts ranging over 14..30
        # instead of the uniform n_α / n_β a CI basis requires).
        # The validator names the offending strings and explains the
        # likely cause. Runs in O(M_a + M_b) — cheap.
        if limb_strings:
            _validate_ci_basis(
                _ints_to_limb_matrix(adets, n_limbs),
                _ints_to_limb_matrix(bdets, n_limbs),
                norb,
                nelec,
                sub_idx=sub_idx,
            )
        else:
            _validate_ci_basis(
                _word_bit_patterns(adets),
                _word_bit_patterns(bdets),
                norb,
                nelec,
                sub_idx=sub_idx,
            )

        # --- warm-start HF augmentation (cold solve only) ---
        # On cold solves where probe_method='warm_start' is set, we
        # auto-include the HF α-string and HF β-string in the basis
        # (if not already present). This protects against a real
        # Davidson failure mode on the warm-extend path: when the
        # user's core basis happens to lack HF and the extended
        # basis carries most of the GS weight on HF-like dets,
        # warm-start's embedded eigvec is near-orthogonal to the
        # true GS in the extended basis and Davidson can converge
        # to an excited state. Ensuring HF is in the cached state
        # (hence in any subsequent extension via the superset check)
        # eliminates this failure mode.
        #
        # Controlled by probe_kwargs['include_hf_in_core'] which
        # defaults to True. Set False for non-chemistry Hamiltonians
        # or other unusual cases.
        #
        # Only triggers on cold solves (probe_state is None). On
        # warm-extend, the cached state already contains HF (it was
        # auto-added at cache time), so the extension basis will
        # contain it too via the superset check.
        cold_with_probe = probe_method == "warm_start" and probe_state_this is None
        if cold_with_probe and bool(probe_kwargs_in.get("include_hf_in_core", True)):
            from torch_sci_solver.torch_impl.probe import _augment_core_with_hf

            adets, bdets, hf_info = _augment_core_with_hf(adets, bdets, norb, nelec)
            # User-visible notice — printed unconditionally (not
            # gated on self.verbose), since HF augmentation modifies the
            # basis the user passed in.
            if hf_info["hf_alpha_added"] or hf_info["hf_beta_added"]:
                parts = []
                if hf_info["hf_alpha_added"]:
                    parts.append(f"HF α-string 0x{hf_info['hf_alpha_string']:x}")
                if hf_info["hf_beta_added"]:
                    parts.append(f"HF β-string 0x{hf_info['hf_beta_string']:x}")
                # Unconditional user-visible notice (basis was modified).
                # Print immediately under the lock — verbose_lines flushes
                # only when self.verbose is True, but this notice must
                # surface either way.
                with self._print_lock:
                    print(
                        f"[probe:warm_start] added {' and '.join(parts)} "
                        f"to core (was not present in input); "
                        f"disable via probe_kwargs={{'include_hf_in_core': False}}"
                    )

        # --- warm-start basis ordering: cached prefix + new appended ---
        # When a probe_state is provided we override the canonical
        # ordering to: cached B₀ exactly, then newly-added strings
        # in sorted order. This preserves the B₀ → B prefix
        # invariant required by the eigvec embedding.
        warm_start_active = probe_method == "warm_start" and probe_state_this is not None
        if warm_start_active:
            if not isinstance(probe_state_this, TorchSCIState):
                raise TypeError(
                    "product-mode warm-start requires probe_state to be a "
                    f"TorchSCIState (got {type(probe_state_this).__name__!r})."
                )
            # Validate integrals + knob fingerprints first. Pass the
            # factory-cached fingerprint to skip the ~30ms SHA-256
            # recompute on 36-orbital integrals.
            _validate_fingerprints(
                probe_state_this,
                _as_integrals_array(one_body_tensor),
                _as_integrals_array(two_body_tensor),
                norb,
                nelec,
                solver_knob_snapshot,
                current_fingerprint=get_integrals_fingerprint(),
            )
            # Dedup-aware union basis: preserves B₀ order, appends
            # new (not-in-B₀) strings in sorted order. Emits a
            # UserWarning if duplicates or core-overlap detected. The limb
            # path feeds the Python-int view, since a 2-D matrix would be
            # read as one key per limb rather than one per string.
            adets, bdets, dedup_info = _dedup_and_check_superset(
                probe_state_this.adets,
                probe_state_this.bdets,
                (
                    np.asarray(adets_caller)
                    if limb_strings
                    else np.array(_ci_string_words(strings_a), dtype=object)
                ),
                (
                    np.asarray(bdets_caller)
                    if limb_strings
                    else np.array(_ci_string_words(strings_b), dtype=object)
                ),
            )
            _emit_dedup_warning(dedup_info)
            if self.verbose:
                verbose_lines.append(
                    f"  [warm_start] core M_a={dedup_info['cached_M_a']}, "
                    f"M_b={dedup_info['cached_M_b']} → extended "
                    f"M_a={dedup_info['extended_M_a']}, "
                    f"M_b={dedup_info['extended_M_b']} "
                    f"(+{dedup_info['α_added']}α, "
                    f"+{dedup_info['β_added']}β)"
                )

        # --- DEBUG dump for SBD parity check ---
        # Controlled by env var TORCH_SCI_DUMP_DIR. When set, dumps the
        # post-dedup (adets, bdets) arrays for this subspace to
        # ``{TORCH_SCI_DUMP_DIR}/torch_sci_call{N}_sub{sub_idx}_*.npy``
        # where N is the solver-call counter (0-indexed, increments
        # every time qiskit calls our sci_solver). Each call writes its
        # own files; nothing overwrites. Pair with
        # ``scripts/diff_basis.py`` to compare against SBD's
        # ``AlphaDets.txt`` output.
        #
        # No-op when TORCH_SCI_DUMP_DIR is unset (production runs).
        import os as _os

        _dump_dir = self.diagnostics.get("dump_dir") or _os.environ.get("TORCH_SCI_DUMP_DIR")
        if _dump_dir:
            _dump_path = pathlib.Path(_dump_dir)
            _dump_path.mkdir(exist_ok=True, parents=True)
            # 1-indexed; we're inside the n_calls-th call. Read under the lock
            # since __call__ bumps n_calls before dispatch — but concurrent
            # subspaces of the same call read the same, stable value.
            with self._stats_lock:
                _call_n = self.stats["n_calls"]
            if limb_strings:
                adets_dump = _ints_to_limb_matrix(adets, n_limbs)
                bdets_dump = _ints_to_limb_matrix(bdets, n_limbs)
            else:
                adets_dump = _word_bit_patterns(adets)
                bdets_dump = _word_bit_patterns(bdets)
            np.save(
                _dump_path / f"torch_sci_call{_call_n}_sub{sub_idx}_adets.npy",
                adets_dump,
            )
            np.save(
                _dump_path / f"torch_sci_call{_call_n}_sub{sub_idx}_bdets.npy",
                bdets_dump,
            )
            if self.verbose:
                verbose_lines.append(
                    f"  [debug] dumped call{_call_n}/sub{sub_idx} basis "
                    f"({len(adets)}α × {len(bdets)}β) to {_dump_dir}"
                )

        # --- connected-seed reorder (matches SBD's _reorder_for_connected_seed) ---
        # See helper docstring for the rationale. We detect αβ-symmetric
        # basis (same_pool) and apply ONE permutation to both lists so
        # the symmetry is preserved exactly. For asymmetric bases each
        # spin sector picks its own connected seed independently.
        #
        # When warm_start is active we SKIP the reorder: the cached
        # state's basis ordering is the canonical one (whatever was
        # picked the first time around), and any reorder here would
        # destroy the B₀-prefix invariant the eigvec embedding relies
        # on. The cached state's solver_knobs already encode the
        # self.seed_strategy used at cache time; _validate_fingerprints
        # enforces consistency between cache and current call.
        best_seed_a = None
        if self.seed_strategy == "connected" and not warm_start_active:
            t0 = _time.perf_counter()
            if limb_strings:
                # Reorder the limb matrices row-wise; the Hamming distance the
                # helper computes sums over limbs, so ``best_seed_a`` below is
                # the full Python-int string.
                adets_arr = _ints_to_limb_matrix(adets, n_limbs)
                bdets_arr = _ints_to_limb_matrix(bdets, n_limbs)
                same_pool = (M_a := len(adets_arr)) == (M_b := len(bdets_arr)) and np.array_equal(
                    adets_arr, bdets_arr
                )
                adets_arr, _perm_a = _reorder_for_connected_seed(adets_arr)
                if same_pool:
                    bdets_arr = adets_arr
                else:
                    bdets_arr, _perm_b = _reorder_for_connected_seed(bdets_arr)
                adets = _limb_rows_as_ints(adets_arr)
                bdets = _limb_rows_as_ints(bdets_arr)
            else:
                adets_arr = _word_bit_patterns(adets)
                bdets_arr = _word_bit_patterns(bdets)
                same_pool = (M_a := len(adets_arr)) == (M_b := len(bdets_arr)) and np.array_equal(
                    adets_arr, bdets_arr
                )
                adets_arr, _perm_a = _reorder_for_connected_seed(adets_arr)
                if same_pool:
                    bdets_arr = adets_arr
                else:
                    bdets_arr, _perm_b = _reorder_for_connected_seed(bdets_arr)
                adets = adets_arr.view(np.uint64).tolist()
                bdets = bdets_arr.view(np.uint64).tolist()
            best_seed_a = int(adets[0])
            with self._stats_lock:
                self.stats["t_seed"] += _time.perf_counter() - t0
        elif self.seed_strategy != "smallest_int":
            raise ValueError(
                f"self.seed_strategy must be 'connected' or 'smallest_int', "
                f"got {self.seed_strategy!r}"
            )

        M_a = len(adets)
        M_b = len(bdets)
        N = M_a * M_b

        # --- per-subspace setup (helpers, hii, W) ---
        t0 = _time.perf_counter()
        if limb_strings:
            adets_t = torch.as_tensor(
                np.ascontiguousarray(_ints_to_limb_matrix(adets, n_limbs)),
                dtype=torch.int64,
                device=dev,
            )
            bdets_t = torch.as_tensor(
                np.ascontiguousarray(_ints_to_limb_matrix(bdets, n_limbs)),
                dtype=torch.int64,
                device=dev,
            )
        else:
            adets_t = torch.tensor(_to_int64_python(adets), dtype=torch.int64, device=dev)
            bdets_t = torch.tensor(_to_int64_python(bdets), dtype=torch.int64, device=dev)

        # Caller-order keys for the init-vector lookups: limb rows for wide
        # strings (``_first_index`` folds multi-column keys), the signed-int64
        # bit patterns otherwise.
        if limb_strings:
            caller_a_keys = adets_limbs_in
            caller_b_keys = bdets_limbs_in
        else:
            caller_a_keys = _signed_int64_keys(strings_a)
            caller_b_keys = _signed_int64_keys(strings_b)

        dist_ctx = self._distributed_ctx
        # Resolve the tile spec before choosing the owner-computes path: on a
        # tile grid the <S^2> diagnostic comes from the operator's staged
        # exchanges, so it does not need the replicated fallback.
        tile_spec = getattr(dist_ctx, "tile", None) if dist_ctx is not None else None
        tile_grid = None
        if tile_spec == "auto":
            from torch_sci_solver.torch_impl.distributed_davidson import (
                resolve_auto_tile_grid,
            )

            tile_grid = resolve_auto_tile_grid(
                world_size=dist_ctx.world_size,
                n_rows=M_a,
                n_cols=M_b,
                n_particles_a=int(n_a),
                n_particles_b=int(n_b),
                n_orb=int(norb),
                budget_bytes=self._memory_budget(dist_ctx),
                itemsize=int(self.dtype.itemsize),
                block_size=(
                    int(self.davidson_block_size)
                    if self.davidson_block_size is not None
                    else max(30, self.n_roots + 3)
                ),
            )
        # A penalty or post-processing run always uses the tile machinery: with
        # no explicit grid a (P, 1) partition reproduces the 1-D row split, so
        # the tiled RDM/S^2 builders can keep the work on the owner-computes
        # path. A declined auto grid takes that same partition rather than the
        # replicated full-vector fallback, which the policy never considered.
        auto_declined = tile_spec == "auto" and tile_grid is None
        postprocess_requested = (
            spin_sq is not None or self.compute_full_rdm or self.compute_spin_squared
        )
        tile_postprocess_1d = postprocess_requested and (
            auto_declined or (spin_sq is not None and tile_spec is None)
        )
        if tile_postprocess_1d and tile_grid is None and dist_ctx is not None:
            tile_grid = (dist_ctx.world_size, 1)
        tile_path = tile_postprocess_1d or (
            tile_spec is not None and (tile_spec != "auto" or tile_grid is not None)
        )
        # Owner-computes distributed TPB path. Chosen only
        # when the whole CI space does not fit in one Davidson block, so the
        # replicated dense escape still handles tiny bases, and when the
        # post-processing does not need a full vector: the spin_sq penalty,
        # full RDMs and the <S^2> diagnostic run on the local path when a tile
        # grid supplies the staged exchanges they need, and keep the
        # replicated fallback otherwise.
        _auto_block_cap = max(30, self.n_roots + 3)
        local_tpb = (
            dist_ctx is not None
            and dist_ctx.matvec == "local"
            and (spin_sq is None or tile_path)
            and (not self.compute_full_rdm or tile_path)
            and (not self.compute_spin_squared or tile_path)
            and N
            > (
                int(self.davidson_block_size)
                if self.davidson_block_size is not None
                else _auto_block_cap
            )
        )
        local_partition = None
        local_bounds = None
        local_operator = None
        local_project_fn = None
        warm = False
        if local_tpb:
            from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE
            from torch_sci_solver.torch_impl.distributed_davidson import (
                _TileIndexLayout,
                global_smallest_hii,
                hash_uniform,
                partition_rows_weighted,
            )
            from torch_sci_solver.torch_impl.distributed_matvec import (
                LocalAlphaBetaProjector,
                LocalTileAlphaBetaProjector,
                LocalTPBOperator,
                alpha_row_weights,
                beta_col_weights,
            )
            from torch_sci_solver.torch_impl.helpers import generate_local_torch_helpers
            from torch_sci_solver.torch_impl.probe import (
                _embed_eigvec_into_extended_basis_rows,
                _embed_eigvec_into_extended_basis_tile,
            )
            from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

            if tile_spec is None:
                grid = (dist_ctx.world_size, 1) if tile_postprocess_1d else None
            elif tile_spec == "auto":
                grid = tile_grid
            elif tile_spec == "force":
                grid = None
            else:
                grid = tuple(int(p) for p in tile_spec)
            tiled = grid is not None or tile_spec == "force"
            if tiled:
                from torch_sci_solver.torch_impl.distributed_davidson import (
                    partition_tiles,
                )

                local_partition = partition_tiles(
                    M_a,
                    M_b,
                    dist_ctx.world_size,
                    grid=grid,
                    row_weights=alpha_row_weights(adets_t, bdets_t, norb),
                    col_weights=beta_col_weights(adets_t, bdets_t, norb),
                )
                local_bounds = local_partition.for_rank(dist_ctx.rank)
                row_start, row_stop = local_bounds.row_start, local_bounds.row_stop
                col_start, col_stop = local_bounds.col_start, local_bounds.col_stop
            else:
                local_partition = partition_rows_weighted(
                    M_a, M_b, alpha_row_weights(adets_t, bdets_t, norb), dist_ctx.world_size
                )
                local_bounds = local_partition.for_rank(dist_ctx.rank)
                row_start, row_stop = local_partition.rows_for_rank(dist_ctx.rank)
                col_start, col_stop = 0, M_b
            adets_local = adets_t[row_start:row_stop]
            bdets_local = bdets_t[col_start:col_stop]
            helpers = list(
                generate_local_torch_helpers(
                    adets,
                    bdets,
                    norb,
                    row_start,
                    row_stop,
                    device=dev,
                    col_start=col_start if tiled else None,
                    col_stop=col_stop if tiled else None,
                    table_dtype=self._table_dtype_torch,
                )
            )
            hii_local = make_diag_terms_torch(
                adets_local, bdets_local, norb, integrals, chunk_size=self.chunk_size
            )
            local_operator = LocalTPBOperator(
                hii_local,
                adets_t,
                bdets_t,
                norb,
                integrals,
                helpers,
                local_partition,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                group=dist_ctx.group,
                matvec_impl=self.matvec_impl,
                chunk_size=self.chunk_size,
                exchange_bytes=getattr(dist_ctx, "exchange_bytes", None),
            )
            if self.apply_ab_projection and M_a == M_b and torch.equal(adets_t, bdets_t):
                projector_cls = LocalTileAlphaBetaProjector if tiled else LocalAlphaBetaProjector
                local_project_fn = projector_cls(
                    adets_t,
                    bdets_t,
                    norb,
                    local_partition,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    group=dist_ctx.group,
                    dtype=self.dtype,
                ).__call__

            if tiled:
                local_global_index = local_bounds.local_to_global(
                    torch.arange(local_bounds.local_size, dtype=torch.int64)
                )
            else:
                local_global_index = torch.arange(
                    local_bounds.start, local_bounds.stop, dtype=torch.int64
                )

            # Cold-start seed: the global argmin of the diagonal, as in the
            # replicated branch below; ``connected`` and multi-root solves keep
            # basis index 0.
            if self.seed_strategy == "connected" or self.n_roots > 1:
                seed_index = 0
            else:
                seed_layout = _TileIndexLayout(local_bounds) if tiled else local_partition
                seed_pairs = global_smallest_hii(hii_local, seed_layout, k=1, group=dist_ctx.group)
                seed_index = seed_pairs[0][0]

            def _cold_local() -> torch.Tensor:
                """Cold start on this rank's block: seed unit entry plus hashed noise."""
                noise = hash_uniform(local_global_index.to(dev), salt=0).to(dtype=self.dtype)
                w_cold = SEED_NOISE_SCALE * noise
                if w_cold.shape[0] > 0:
                    if tiled:
                        hf_pos = int(
                            torch.searchsorted(
                                local_global_index, torch.tensor(seed_index, dtype=torch.int64)
                            ).item()
                        )
                        owns_hf = (
                            hf_pos < local_global_index.numel()
                            and int(local_global_index[hf_pos]) == seed_index
                        )
                    else:
                        hf_pos = seed_index - local_bounds.start
                        owns_hf = 0 <= hf_pos < w_cold.shape[0]
                    if owns_hf:
                        w_cold[hf_pos] = 1.0
                return w_cold / _global_norm_local(w_cold, dist_ctx)

            if warm_start_active:
                # Restricted embedding of the cached interleaved-Wick eigvec:
                # each rank pastes only its own extended rows (1-D) or its own
                # rectangle (tile) and the surviving norm is decided globally.
                if tiled:
                    W = _embed_eigvec_into_extended_basis_tile(
                        probe_state_this.eigvec_inter,
                        probe_state_this.adets,
                        probe_state_this.bdets,
                        adets,
                        bdets,
                        row_start,
                        row_stop,
                        col_start,
                        col_stop,
                        device=dev,
                        dtype=self.dtype,
                    )
                else:
                    # The 1-D layout owns a row-major ``(rows, M_b)`` slice, whose
                    # flat form is exactly the contiguous ``N_local`` block the
                    # owner-computes Davidson takes.
                    W = _embed_eigvec_into_extended_basis_rows(
                        probe_state_this.eigvec_inter,
                        probe_state_this.adets,
                        probe_state_this.bdets,
                        adets,
                        bdets,
                        row_start,
                        row_stop,
                        device=dev,
                        dtype=self.dtype,
                    ).reshape(-1)
                _warm_norm = _global_norm_local(W, dist_ctx)
                if float(_warm_norm.item()) == 0.0:
                    W = _cold_local()
                else:
                    W = W / _warm_norm
                    warm = True
            elif init_vector is not None:
                ia = _first_index(caller_a_keys, adets_t, device=dev)
                ia = ia[row_start:row_stop]
                ib = _first_index(caller_b_keys, bdets_t, device=dev)
                ib = ib[col_start:col_stop]
                _sigma_init_local = _inter_to_block_signs(adets_local, bdets_local, norb).to(
                    dtype=self.dtype
                )

                def _map_init_local(row: np.ndarray) -> torch.Tensor:
                    vec2d = torch.as_tensor(row, dtype=self.dtype, device=dev).reshape(
                        len(strings_a), len(strings_b)
                    )
                    w = vec2d[ia.clamp_min(0)[:, None], ib.clamp_min(0)[None, :]]
                    w[ia < 0] = 0
                    w[:, ib < 0] = 0
                    del vec2d
                    return (w * _sigma_init_local).reshape(-1)

                W, warm = _resolve_start_vectors_local(
                    init_vector, _map_init_local, local_project_fn, _cold_local, dist_ctx
                )
            else:
                W = _cold_local()
        else:
            helpers = list(
                generate_all_torch_helpers(
                    adets, bdets, norb, device=dev, table_dtype=self._table_dtype_torch
                )
            )
            hii = make_diag_terms_torch(
                adets_t, bdets_t, norb, integrals, chunk_size=self.chunk_size
            )
            # Cold-start seed: the lowest-diagonal determinant, not basis index 0
            # (the smallest-α × smallest-β pair). Strings that form separate
            # clusters make the product space block diagonal, Davidson does not
            # leave the block its seed lies in, and index 0 need not share a block
            # with the ground state; seeding at argmin(hii) keeps the Ritz value
            # <= min(hii). ``connected`` keeps its most-connected determinant, and
            # a multi-root seed block already adds the lowest-diagonal unit vectors.
            seed_index = (
                int(torch.argmin(hii))
                if self.seed_strategy != "connected" and self.n_roots == 1
                else 0
            )
            if warm_start_active:
                # Embed the cached interleaved-Wick eigvec into the
                # extended basis. The cached state's basis is the prefix
                # of (adets, bdets) by construction, so this is a 2D
                # paste into the top-left (M_a_old × M_b_old) corner.
                # _embed_... handles self.dtype/device conversion and
                # normalization.
                W = _embed_eigvec_into_extended_basis(
                    probe_state_this.eigvec_inter,
                    probe_state_this.adets,
                    probe_state_this.bdets,
                    adets,
                    bdets,
                    device=dev,
                    dtype=self.dtype,
                )
            elif init_vector is not None:
                # Caller's spin-block amplitudes over (strings_a, strings_b), re-indexed
                # to (adets, bdets) and converted to interleaved-Wick (C_inter = σ·C_block,
                # σ² = 1). Projected and normalized right before Davidson, which falls
                # back to the cold start if nothing is left.
                from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

                ia = _first_index(caller_a_keys, adets_t, device=dev)
                ib = _first_index(caller_b_keys, bdets_t, device=dev)
                _sigma_init = _inter_to_block_signs(adets_t, bdets_t, norb).to(dtype=self.dtype)

                def _map_init(row: np.ndarray) -> torch.Tensor:
                    vec2d = torch.as_tensor(row, dtype=self.dtype, device=dev).reshape(
                        len(strings_a), len(strings_b)
                    )
                    # One advanced index straight to (M_a, M_b): a two-step vec2d[ia][:, ib]
                    # would also hold an (M_a, len(strings_b)) intermediate. Missing strings
                    # (index -1) gather an arbitrary row/column, zeroed below.
                    w = vec2d[ia.clamp_min(0)[:, None], ib.clamp_min(0)[None, :]]
                    w[ia < 0] = 0
                    w[:, ib < 0] = 0
                    del vec2d
                    return (w * _sigma_init).reshape(-1)

                # The 1-D entry's mapped vector; per-root rows are mapped at resolve time.
                W = _map_init(init_vector if init_vector.ndim == 1 else init_vector[0])
            else:
                W = basis_init_vector_torch(
                    M_a, M_b, init=0, device=dev, dtype=self.dtype, seed_index=seed_index
                )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_setup = _time.perf_counter() - t0
        with self._stats_lock:
            self.stats["t_setup"] += t_setup

        # --- spin_sq penalty setup, EARLY (before _pick_num_block) ---
        # We construct the SpinSqOperator and its scratch buffers BEFORE
        # picking the Davidson self.davidson_block_size, so that _pick_num_block sees
        # the post-penalty free memory and chooses a self.davidson_block_size that
        # leaves room for Davidson's subspace.
        #
        # Memory budget for spin_sq=<float> path, all of size N × 8 B:
        #   - σ-conversion vector (interleaved-Wick ↔ spin-block)
        #   - S² operator's diagonal vector
        #   - one scratch buffer for σ * v
        #   - one scratch buffer for the S² apply result
        # = 4 × N × 8 B persistent for the duration of the subspace solve.
        # On top, the matvec call itself materializes Hv (1 × N × 8 B),
        # but Davidson immediately consumes that — no further transients
        # because the augmented matvec composition is in-place on Hv.
        _S2_op = None
        _sigma = None
        _S2v_buf = None
        _v_block_buf = None
        t_penalty_setup = 0.0
        tiled_penalty = spin_sq is not None and local_tpb
        if spin_sq is not None and not tiled_penalty:
            from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs
            from torch_sci_solver.torch_impl.spin_sq import SpinSqOperator

            _t0 = _time.perf_counter()
            _S2_op = SpinSqOperator(
                adets_t,
                bdets_t,
                norb,
                n_a,
                n_b,
                device=dev,
                dtype=self.dtype,
            )
            _sigma = (
                _inter_to_block_signs(adets_t, bdets_t, norb)
                .to(dtype=self.dtype)
                .reshape(-1)
                .contiguous()
            )
            # Pre-allocate scratch buffers — one σ-converted input, one
            # S² apply output. Reused across every Davidson matvec call
            # for this subspace; freed when the subspace iteration ends.
            _S2v_buf = torch.empty(N, dtype=self.dtype, device=dev)
            _v_block_buf = torch.empty(N, dtype=self.dtype, device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_penalty_setup = _time.perf_counter() - _t0
            with self._stats_lock:
                self.stats["t_spin_sq"] += t_penalty_setup

        if local_tpb:
            # Per rank the Davidson block is (b, N_local); include the local
            # operator's buffers (2 stage buffers + y + x + send gathers) in
            # the overhead instead of the replicated path's 4 full vectors.
            chosen_nb = self._pick_num_block(
                1,
                max(1, local_bounds.local_size),
                device=dev,
                n_roots=self.n_roots,
                local_n=max(1, local_bounds.local_size),
                matvec_vectors=9,
                cap_to_n=False,
            )
        else:
            # The factorized αβ pass holds three more N-element work arrays per
            # matvec (see FactorizedAlphaBeta).
            chosen_nb = self._pick_num_block(
                M_a,
                M_b,
                device=dev,
                n_roots=self.n_roots,
                matvec_vectors=4 + (3 if self.ab_factorization != "off" else 0),
            )
        if self._distributed_ctx is not None:
            # The distributed thick restart keeps every root plus the new
            # correction; leave room for it even when the auto-tuner or the
            # user picked a tiny block.
            chosen_nb = min(max(chosen_nb, self.n_roots + 3), N)

        if self.verbose:
            free_after_setup = self._free_bytes(device=dev)
            sub_b, work_b = self._davidson_element_bytes()
            dav_estimate = (2 * chosen_nb * sub_b + 4 * work_b) * (
                local_bounds.local_size if local_tpb else N
            )
            verbose_lines.append(
                f"  [sci_solver] subspace {sub_idx}: "
                f"M_a×M_b = {M_a}×{M_b} = {N:,}  "
                f"| dev={dev}  "
                f"| davidson_block_size={chosen_nb}  "
                f"| free={self._fmt(free_after_setup)}  "
                f"| Davidson est ≈ {self._fmt(dav_estimate)}"
            )
            if local_tpb:
                verbose_lines.append(
                    f"               distributed matvec: owner-computes rows "
                    f"[{local_partition.rows_for_rank(dist_ctx.rank)[0]}, "
                    f"{local_partition.rows_for_rank(dist_ctx.rank)[1]}) "
                    f"N_local={local_bounds.local_size:,} "
                    f"(cost-balanced)"
                )
            verbose_lines.append(f"               helpers/hii/W setup: {t_setup:.2f}s")
            if self.seed_strategy == "connected" and best_seed_a is not None:
                verbose_lines.append(
                    f"               seed: '{self.seed_strategy}'  "
                    f"| adets[0] = 0x{best_seed_a:x} "
                    f"(popcount {bin(best_seed_a).count('1')})"
                )
            if _S2_op is not None:
                verbose_lines.append(
                    f"               spin_sq penalty: ENABLED "
                    f"(target={float(spin_sq_target):g}, "
                    f"λ={call_spin_lambda:g}, "
                    f"S² build {t_penalty_setup:.2f}s, "
                    f"n_α_triples={int(_S2_op.alpha_i.shape[0])}, "
                    f"n_β_triples={int(_S2_op.beta_i.shape[0])})"
                )

        # --- Davidson ---
        if local_tpb:
            matvec = None
        else:
            matvec = build_matvec(
                self.matvec_impl,
                hii,
                adets_t,
                bdets_t,
                norb,
                integrals,
                helpers,
                chunk_size=self.chunk_size,
                ab_factors=self._ab_factors(two_body_tensor, norb),
            )

        # Augment matvec with spin_sq penalty when active. This block
        # is skipped entirely when spin_sq is None — same code path as
        # v0.10.18, zero memory cost.
        hii_for_davidson = hii_local if local_tpb else hii  # default: bare H diagonal
        _local_op_for_dav = local_operator
        if tiled_penalty:
            from torch_sci_solver.torch_impl.tiled_postprocess import (
                tiled_s2_diagonal,
            )

            hii_for_davidson = hii_local + call_spin_lambda * (
                tiled_s2_diagonal(local_operator) - float(spin_sq_target)
            )
            _local_op_for_dav = _TiledPenalizedOperator(
                local_operator, call_spin_lambda, float(spin_sq_target)
            )
            verbose_lines.append(
                f"               spin_sq penalty: ENABLED on the tile operator "
                f"(target={float(spin_sq_target):g}, λ={call_spin_lambda:g})"
            )
        if spin_sq is not None and not tiled_penalty:
            _lam = call_spin_lambda
            _target = float(spin_sq_target)
            _bare_matvec = matvec

            def matvec(v):  # redef intentional
                # H matvec (allocates Hv, one (N,) tensor).
                """Matvec."""
                Hv = _bare_matvec(v)
                # σ-convert v → spin-block, write into scratch.
                torch.mul(v, _sigma, out=_v_block_buf)
                # S² apply, writes into scratch (no new alloc).
                _S2_op.apply(_v_block_buf, out=_S2v_buf)
                # σ-convert back to interleaved-Wick, in-place.
                _S2v_buf.mul_(_sigma)
                # In-place compose Hv + λ(S2v - target·v):
                #     Hv += λ · S2v
                Hv.add_(_S2v_buf, alpha=_lam)
                if _target != 0.0:
                    #     Hv += -(λ · target) · v
                    Hv.add_(v, alpha=-_lam * _target)
                return Hv

            # Augment the diagonal preconditioner: the operator
            # Davidson sees is H' = H + λ(S² − target). The S²
            # diagonal is σ-invariant (σ² = 1) so no σ-bridge is
            # needed for hii. Passing the bare H diagonal stalls
            # Davidson at large λ.
            hii_for_davidson = hii + _lam * (_S2_op.diagonal().real - _target)

        # When the basis is αβ-symmetric (the SBD-style ``bdet = adet``
        # construction used by qiskit_addon_sqd's ``symmetrize_spin=True``
        # path), build a projection onto the αβ-swap-symmetric subspace
        # and feed it to Davidson. Without this, accumulated round-off
        # in the chunked matvec lets the antisymmetric (triplet) sector
        # leak in, and at large basis sizes the triplet eigenvalue dips
        # below the singlet's, causing Davidson to converge to the
        # wrong state.
        #
        # Convention detail — the matvec stores the wavefunction in the
        # interleaved-Wick basis (see rdm.py "Convention"), which differs
        # from the spin-block basis by a per-(iₐ, i_b) sign
        #   σ[i, j] = (-1)^{Σ_{p in occ_a(i)} popcount(occ_b(j) & (1<<p) - 1)}.
        # In the spin-block basis σ_ab acts as v[i, j] → (-1)^{n_α n_β} v[j, i],
        # so in the interleaved basis it acts as
        #   σ_ab v[i, j] = (-1)^{n_α n_β} · τ[i, j] · v[j, i],   τ = σ · σᵀ.
        # For closed-shell singlets (n_α = n_β = N) the singlet subspace is
        # the (-1)^N-eigenspace of σ_ab, and after canceling the two
        # N-dependent signs the projector reduces to
        #   P(v)[i, j] = (v[i, j] + τ[i, j] · v[j, i]) / 2   in the interleaved basis.
        # The previous sign-agnostic (v + vᵀ)/2 form is only correct in the
        # τ ≡ 1 case (in practice: n_α ≤ 1). For any N ≥ 2 the plain swap
        # v[i,j] ↔ v[j,i] no longer commutes with H — the α-β Wick sign
        # is non-uniform across the CI basis — so applying (v + vᵀ)/2
        # forces Davidson onto a *wrong* invariant subspace of H and
        # returns an energy several mHa above the exact singlet FCI
        # (verified against PySCF on H4/H6/H8/naphthalene 10e10o).
        #
        # Can be turned off at factory time via ``apply_ab_projection=False``
        # for SBD parity comparison (SBD has no equivalent projection).
        project_fn = local_project_fn if local_tpb else None
        if (
            not local_tpb
            and self.apply_ab_projection
            and M_a == M_b
            and torch.equal(adets_t, bdets_t)
        ):
            _M = M_a  # captured by closure

            # Reuse the σ matrix built by the spin_sq penalty path when
            # available; otherwise compute it now (an (M_a, M_b) ±1 tensor
            # — cost is dwarfed by a single matvec).
            if _sigma is not None:
                _sigma_2d = _sigma.view(_M, _M)
            else:
                from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

                _sigma_2d = _inter_to_block_signs(adets_t, bdets_t, norb).to(dtype=self.dtype)
            _tau = (_sigma_2d * _sigma_2d.T).contiguous()  # (M, M), ±1

            def project_fn(v: torch.Tensor) -> torch.Tensor:
                # P(v)[i, j] = (v[i, j] + τ[i, j] · v[j, i]) / 2
                """Project fn."""
                v2d = v.view(_M, _M)
                return ((v2d + _tau * v2d.T) * 0.5).reshape(-1)

            if self.verbose:
                verbose_lines.append(
                    f"               αβ-swap projection: ENABLED (M_a == M_b == {_M})"
                )
        elif local_tpb and self.verbose:
            verbose_lines.append(
                "               αβ-swap projection: "
                + ("ENABLED (local-block)" if project_fn is not None else "DISABLED")
            )
        elif self.verbose:
            if not self.apply_ab_projection:
                verbose_lines.append(
                    "               αβ-swap projection: DISABLED "
                    "(forced off via apply_ab_projection=False)"
                )
            else:
                verbose_lines.append(
                    f"               αβ-swap projection: DISABLED "
                    f"(basis not αβ-symmetric: M_a={M_a}, M_b={M_b}, "
                    f"equal={M_a == M_b and torch.equal(adets_t, bdets_t)})"
                )

        if not local_tpb:
            warm = False
            if init_vector is not None:
                _cold_tpb = lambda: basis_init_vector_torch(  # noqa: E731
                    M_a, M_b, init=0, device=dev, dtype=self.dtype, seed_index=seed_index
                )
                if init_vector.ndim == 1:
                    W, warm = _resolve_start_vector(W, project_fn, _cold_tpb, init_vector)
                else:
                    W, warm = _resolve_start_vectors(init_vector, _map_init, project_fn, _cold_tpb)
            if not warm and self.verbose:
                verbose_lines.append(
                    f"               init_vectors: no overlap with this basis "
                    f"(< {_INIT_VECTOR_MIN_NORM:g} of its norm left); cold start"
                )

        if self._distributed_ctx is not None and not local_tpb and self.verbose:
            verbose_lines.append(
                "               distributed matvec: replicated "
                "(owner-computes needs TPB with N > davidson_block_size; full "
                "RDMs and the <S^2> diagnostic and penalty need a tile grid)"
            )

        checkpoint = self._make_davidson_checkpoint(
            sub_idx=sub_idx,
            context=self._distributed_ctx,
            basis_kind="product",
            basis_arrays=(adets_t, bdets_t),
            n_alpha=M_a,
            n_beta=M_b,
            particle_sector=(n_a, n_b),
            dimension=N,
            norb=norb,
            partition=local_partition if local_tpb else None,
            layout_kind="local" if local_tpb else "replicated",
            chosen_nb=chosen_nb,
            spin_sq_target=spin_sq_target,
            spin_penalty_lambda=call_spin_lambda,
            get_integrals_fingerprint=get_integrals_fingerprint,
            solver_knob_snapshot=solver_knob_snapshot,
        )
        t0 = _time.perf_counter()
        if local_tpb:
            _dav_result, perf_dav = self._run_davidson_distributed_local(
                hii_for_davidson,
                W,
                _local_op_for_dav,
                local_bounds,
                context=self._distributed_ctx,
                call_max_iter=call_max_iter,
                chosen_nb=chosen_nb,
                call_eps=call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                warm=warm,
                checkpoint=checkpoint,
            )
        elif self._distributed_ctx is not None:
            # Replicated fallback (small basis / post-processing without a tile
            # grid): the ``C``/``HC`` blocks stay partitioned as
            # ``DistributedVector``s while the operator applies on a gathered
            # full vector. The narrow subspace and complex coefficients are
            # both supported; only the missed-root search is not (it needs the
            # owner-computes path and is rejected in ``_run_davidson_distributed``).
            _dav_result, perf_dav = self._run_davidson_distributed(
                hii_for_davidson,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                context=self._distributed_ctx,
                warm=warm,
                checkpoint=checkpoint,
            )
        else:
            _dav_result, perf_dav = self._run_davidson(
                hii_for_davidson,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                warm=warm,
            )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_dav = _time.perf_counter() - t0
        with self._stats_lock:
            self.stats["t_davidson"] += t_dav

        # Unpack Davidson result — single-root returns (float, Tensor(N,), dict);
        # multi-root returns (Tensor(n_roots,), Tensor(n_roots,N), dict).
        if self.n_roots == 1:
            energy_raw, eigvecs_raw, info = _dav_result
            # Wrap into 1-element tensors so the per-root loop below is uniform.
            energies_raw = torch.tensor([energy_raw], dtype=torch.float64)
            all_eigvecs = eigvecs_raw.unsqueeze(0)  # (1, N)
        else:
            energies_raw, all_eigvecs, info = _dav_result  # (n_roots,) CPU, (n_roots,N) dev

        n_found = all_eigvecs.shape[0]

        if local_tpb:
            # Owner-computes post-processing: local occupancies + size-dependent
            # hand-back, no full-vector gather on non-zero ranks.
            return self._finish_distributed_local_subspace(
                sub_idx=sub_idx,
                all_eigvecs=all_eigvecs,
                energies_raw=energies_raw,
                info=info,
                perf_dav=perf_dav,
                adets=adets,
                bdets=bdets,
                adets_t=adets_t,
                bdets_t=bdets_t,
                norb=norb,
                n_a=n_a,
                n_b=n_b,
                partition=local_partition,
                operator=local_operator,
                penalty=((call_spin_lambda, float(spin_sq_target)) if tiled_penalty else None),
                context=self._distributed_ctx,
                dev=dev,
                t_setup=t_setup,
                t_dav=t_dav,
                chosen_nb=chosen_nb,
                warm=warm,
                build_output_states=build_output_states,
                get_integrals_fingerprint=get_integrals_fingerprint,
                solver_knob_snapshot=solver_knob_snapshot,
                verbose_lines=verbose_lines,
            )

        # ---------------------------------------------------------------
        # Per-root post-processing: penalty strip, occupancies, σ-bridge,
        # SCIResult construction. Produces one SCIResult per root.
        # ---------------------------------------------------------------
        all_sci_results: List[SCIResultWithConvergence] = []

        from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

        sigma = _inter_to_block_signs(adets_t, bdets_t, norb).to(dtype=self.dtype)
        sigma_flat = sigma.reshape(-1)

        t_occ_total = 0.0

        # MPS solves in float32: report float64 CPU Rayleigh quotients instead of
        # the float32 Ritz values (see _refines_fp32_energy).
        energy_matvec64 = None
        if self._refines_fp32_energy(call_eps):
            cpu = torch.device("cpu")
            integrals64 = self._cpu_wide_integrals(one_body_tensor, two_body_tensor, spinless=False)
            adets64, bdets64 = adets_t.cpu(), bdets_t.cpu()
            energy_matvec64 = build_matvec(
                "batched",
                make_diag_terms_torch(adets64, bdets64, norb, integrals64),
                adets64,
                bdets64,
                norb,
                integrals64,
                [_tensors_to(h, cpu) for h in helpers],
                chunk_size=self.chunk_size,
            )
            perf_dav["davidson_fp32_energy"] = "rq64"

        for root_idx in range(n_found):
            eigvec = all_eigvecs[root_idx]  # (N,) on device
            energy = float(energies_raw[root_idx].item())

            # --- strip the spin-sq penalty contribution from the energy ---
            # Davidson minimized H' = H + λ(S² − target), so ``energy`` is
            # the augmented eigenvalue E_aug = E_phys + λ(⟨S²⟩ − target).
            # The user view is that the penalty is a tool to bias Davidson
            # toward a spin sector — the reported energy should be E_phys,
            # the H expectation value of the converged eigenstate. Compute
            # ⟨S²⟩ once on the final vector (one S² matvec via the same
            # operator built for the penalty, negligible vs Davidson),
            # subtract the penalty bias.
            #
            # Mathematical identity used: for the Ritz value of a normalised
            # Davidson eigenvector ψ in the augmented operator H',
            #   E_aug = ⟨ψ|H'|ψ⟩ = ⟨ψ|H|ψ⟩ + λ(⟨ψ|S²|ψ⟩ − target)
            # so ⟨ψ|H|ψ⟩ = E_aug − λ(⟨ψ|S²|ψ⟩ − target).
            if _S2_op is not None:
                # Reuse the pre-allocated scratch buffers.
                torch.mul(eigvec, _sigma, out=_v_block_buf)
                _S2_op.apply(_v_block_buf, out=_S2v_buf)
                # vdot + .real: ⟨x|S²|x⟩ is Hermitian-conjugate-linear and real
                # (== dot for the real path); plain dot/float() is wrong for
                # complex eigvecs.
                _s2_value = float(torch.vdot(_v_block_buf, _S2v_buf).real.item())
                _norm_sq = float(torch.vdot(eigvec, eigvec).real.item())
                if _norm_sq > 0.0:
                    _s2_value /= _norm_sq
                energy = energy - call_spin_lambda * (_s2_value - float(spin_sq_target))
                if root_idx == 0:
                    # Only record ground state ⟨S²⟩ in stats (matches prior behaviour)
                    with self._stats_lock:
                        self.stats["spin_sq_final"] = _s2_value
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            if energy_matvec64 is not None:
                # ⟨ψ|H|ψ⟩ of the bare Hamiltonian is the physical energy itself,
                # so it also replaces the penalty-stripped value.
                energy = _rayleigh_quotient_wide(energy_matvec64, eigvec, self._wide_dtype())

            # --- occupancies for SCIResult ---
            t0_occ = _time.perf_counter()
            occ_a_t, occ_b_t = make_one_rdm_diag_torch(eigvec, adets_t, bdets_t, norb)
            occupancies = (occ_a_t.cpu().numpy(), occ_b_t.cpu().numpy())
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_occ_total += _time.perf_counter() - t0_occ

            # --- convention bridge: convert eigvec to spin-block ---
            eigvec_block = (eigvec * sigma_flat).detach().cpu().numpy()

            # qiskit's SCIState stores one bitmask per string; a basis wider
            # than 64 bits keeps the exact Python ints in an object array rather
            # than overflowing int64, and a 64-orbital basis using bit 63 gets
            # uint64 for both spins.
            if limb_strings:
                ci_strs_a = np.array(adets, dtype=object)
                ci_strs_b = np.array(bdets, dtype=object)
            else:
                ci_strs_a, ci_strs_b = _ci_strs_arrays(adets, bdets)
            sci_state = SCIState(
                amplitudes=eigvec_block.reshape(M_a, M_b),
                ci_strs_a=ci_strs_a,
                ci_strs_b=ci_strs_b,
                norb=norb,
                nelec=(n_a, n_b),
            )

            # --- full RDMs (optional, ground state only to bound cost) ---
            rdm1 = None
            rdm2 = None
            t_rdm = 0.0
            if self.compute_full_rdm and root_idx == 0:
                t0 = _time.perf_counter()
                ga, gb = make_one_rdm_torch(
                    eigvec, adets_t, bdets_t, norb, helpers, chunk_size=self.chunk_size
                )
                Gaa, Gab, Gbb = make_two_rdm_torch(
                    eigvec, adets_t, bdets_t, norb, helpers, chunk_size=self.chunk_size
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                rdm1 = (ga + gb).cpu().numpy()
                rdm2 = (Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb).cpu().numpy()
                del ga, gb, Gaa, Gab, Gbb
                t_rdm = _time.perf_counter() - t0
                with self._stats_lock:
                    self.stats["t_rdm"] += t_rdm

            all_sci_results.append(
                SCIResultWithConvergence(
                    energy=float(energy),
                    sci_state=sci_state,
                    orbital_occupancies=occupancies,
                    rdm1=rdm1,
                    rdm2=rdm2,
                    state="converged" if info["converged"] else "not_converged",
                )
            )

        with self._stats_lock:
            self.stats["t_occ"] += t_occ_total

        # Ground-state result (root 0) drives diagnostics and warm-start snapshot.
        gs_result = all_sci_results[0]
        energy = float(gs_result.energy)
        eigvec = all_eigvecs[0]  # ground-state interleaved-Wick eigvec (device)

        # --- ⟨S²⟩ diagnostic (optional, ground state only) ---
        s2_value = None
        if self.compute_spin_squared:
            from torch_sci_solver.torch_impl.rdm import (
                make_two_rdm_ab_torch,
                s_squared_from_rdm,
            )

            t0 = _time.perf_counter()
            Gab_s = make_two_rdm_ab_torch(
                eigvec, adets_t, bdets_t, norb, helpers, chunk_size=self.chunk_size
            )
            s2_t = s_squared_from_rdm(Gab_s, n_a, n_b)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            s2_value = float(s2_t.item())
            _t_s2_diag = _time.perf_counter() - t0
            with self._stats_lock:
                self.stats["spin_squared"].append(s2_value)
                self.stats["t_spin_sq"] += _t_s2_diag
            del Gab_s
        else:
            with self._stats_lock:
                self.stats["spin_squared"].append(None)

        # --- snapshot interleaved-Wick eigvec for warm-start cache ---
        # Only the ground-state eigvec is cached (warm-start is ground-state only).
        eigvec_inter_cpu_for_state: Optional[torch.Tensor] = None
        if build_output_states:
            # Preserve the coefficients' dtype: a complex Hermitian solve needs
            # the cached phases, so widening the real cache to fp64 must not
            # become a real-only cast for complex128/complex64.
            state_dtype = torch.complex128 if eigvec.is_complex() else torch.float64
            # Move before widening: MPS cannot hold the fp64/complex128 copy.
            eigvec_inter_cpu_for_state = eigvec.detach().cpu().to(dtype=state_dtype).clone()

        if self.verbose:
            _spin_sq_final = None
            if _S2_op is not None:
                with self._stats_lock:
                    _spin_sq_final = self.stats["spin_sq_final"]
            self._append_davidson_verbose(
                verbose_lines,
                t_dav=t_dav,
                info=info,
                t_occ=t_occ_total,
                line_energies=energies_raw[:n_found],
                energy=energy,
                rdm_note="  | full RDM (root 0 only)" if self.compute_full_rdm else None,
                s2_value=s2_value,
                spin_sq_final=_spin_sq_final,
                spin_sq_target=spin_sq_target,
                spin_lambda=call_spin_lambda,
            )

            from torch_sci_solver.torch_impl.mult import (
                MATVEC_PROFILE,
                _matvec_profile_format,
            )

            if MATVEC_PROFILE:
                verbose_lines.append("               [matvec profile, last matvec only]")
                for pline in _matvec_profile_format().splitlines():
                    verbose_lines.append(f"               {pline}")

        # --- build TorchSCIState (probe-method output) ---
        # Ground-state only (warm-start is single-root).
        if build_output_states:
            fp_int = get_integrals_fingerprint()
            output_state = TorchSCIState(
                adets=list(adets),
                bdets=list(bdets),
                eigvec_inter=eigvec_inter_cpu_for_state,
                energy=float(energy),
                n_orb=norb,
                nelec=(n_a, n_b),
                integrals_fingerprint=fp_int,
                solver_knobs=dict(solver_knob_snapshot),
                n_matvecs=info.get("n_matvecs", None),
                final_residual=info.get("final_residual", None),
            )

        perf = self._subspace_perf(
            sub_idx=sub_idx,
            m_a=M_a,
            m_b=M_b,
            n=N,
            chosen_nb=chosen_nb,
            t_setup=t_setup,
            t_dav=t_dav,
            t_occ=t_occ_total,
            info=info,
            warm=warm,
            perf_dav=perf_dav,
            energy=energy,
            dev=dev,
        )

        if verbose_lines:
            with self._print_lock:
                for _ln in verbose_lines:
                    print(_ln)

        return EigenStates(states=all_sci_results), output_state, perf

    def _solve_one_subspace_pair(
        self,
        *,
        sub_idx: int,
        strings_a,
        strings_b,
        probe_state_this,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        n_a: int,
        n_b: int,
        spin_sq,
        spin_sq_target,
        integrals,
        get_integrals_fingerprint,
        probe_method,
        probe_kwargs_in: dict,
        build_output_states: bool,
        solver_knob_snapshot: dict,
        call_max_iter: int,
        call_eps: float,
        call_de_tol: Optional[float],
        call_spin_lambda: float,
        device: "Optional[torch.device]" = None,
        init_vector: Any = None,
    ):
        """Solve a single subspace in GDB mode (``mode="gdb"``).

        Parallel to :meth:`_solve_one_subspace` but routes through the
        pair-basis kernels. The explicit pair basis supports the previously
        gated features (a structural basis is more restricted; see the gates
        in the body):

          * ``compute_full_rdm``, ``compute_spin_squared``: full RDM
            builders in :mod:`torch_sci_solver.torch_impl.rdm_pairs`.
          * ``spin_sq=<float>`` penalty: :class:`PairSpinSqOperator`.
          * ``apply_ab_projection=True`` is honored when the pair set
            is closed under ``(α, β) → (β, α)``; silently skipped
            otherwise rather than corrupting Davidson by projecting
            outside the basis.
          * ``seed_strategy="connected"`` seeds Davidson at the pair
            with the highest coupling degree (Davidson is basis-
            ordering-invariant, so we seed via ``W[best_idx] = 1``
            rather than permuting the pair basis).
          * ``probe_method="warm_start"`` uses
            :mod:`torch_sci_solver.torch_impl.probe_pairs` for the
            pair-aware superset check, HF augmentation, and eigvec
            embedding. Cached state type is ``TorchPairSCIState``.
          * On a structural basis a warm start only checks that the cached
            pairs all lie in the caller's block spec (a block spec cannot
            be extended pair by pair), and a cold-with-probe start skips
            the HF splice; ``init_vectors`` are indexed by structural pair, the
            order of the amplitudes the structural solve returns.
        """
        # Local imports of the pair-mode modules — kept out of the product-
        # mode hot path so that installations without the pair extension
        # (unlikely — same package, no optional dep) don't pay a cold-import
        # tax on every product-mode subspace.
        from torch_sci_solver.torch_impl.mult_pairs import (
            make_diag_terms_pair_torch,
            mult_pair_torch,
        )
        from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
        from torch_sci_solver.torch_impl.pairs import (
            PairSCIState,
            pair_basis_from_arrays,
        )
        from torch_sci_solver.torch_impl.probe_pairs import (
            TorchPairSCIState,
            _augment_pair_with_hf,
            _dedup_and_check_pair_superset,
            _embed_eigvec_pair_into_extended,
            _emit_pair_dedup_warning,
        )
        from torch_sci_solver.torch_impl.rdm_pairs import (
            make_one_rdm_diag_pair_torch,
            make_one_rdm_pair_torch,
            make_two_rdm_ab_pair_torch,
            make_two_rdm_pair_torch,
        )

        dev = device if device is not None else self.device
        verbose_lines: list[str] = []

        # --- pair-mode feature gates ---------------------------------------
        if self.seed_strategy not in ("smallest_int", "connected"):
            raise ValueError(
                f"seed_strategy must be 'smallest_int' or 'connected', got {self.seed_strategy!r}"
            )
        # apply_ab_projection: honored when the pair set is closed under
        # (alpha, beta) -> (beta, alpha); skipped otherwise (see docstring).

        # --- basis construction --------------------------------------------
        # A structural basis is supplied per subspace as a sequence of
        # ``(alpha_strings, beta_strings)`` Cartesian blocks; the explicit pair
        # list is never build in that case.
        _dist_ctx = self._distributed_ctx
        structural_spec = self._structural_pair_spec_and_gates(
            sub_idx,
            spin_sq=spin_sq,
            tile_spec=getattr(_dist_ctx, "tile", None),
            matvec=None if _dist_ctx is None else _dist_ctx.matvec,
        )

        if structural_spec is not None:
            from torch_sci_solver.torch_impl.pairs import structural_pair_basis_from_blocks

            pair_basis = structural_pair_basis_from_blocks(
                structural_spec, n_orb=int(norb), device=dev
            )
            n_pairs = pair_basis.n_pairs
            if init_vector is not None:
                # A structural init vector is indexed by structural pair (block
                # by block, α-major within a block): the order of the amplitudes
                # a structural solve returns.
                init_vector = _validate_init_vector(
                    init_vector,
                    n_pairs,
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )
            # A warm start does not reshape a structural basis: the caller's
            # block spec is the whole basis, so the cached pairs must already
            # be in it. A cold-with-probe start seeds from the smallest-pair
            # entry like a plain cold start; splicing the HF pair in is an
            # explicit-list operation and is skipped here.
            cold_with_probe = probe_method == "warm_start" and probe_state_this is None
            warm_start_active = probe_method == "warm_start" and probe_state_this is not None
            if cold_with_probe and bool(probe_kwargs_in.get("include_hf_in_core", True)):
                if self.verbose:
                    verbose_lines.append(
                        "  [warm_start:pair] structural basis: HF augmentation "
                        "skipped (the block spec is caller-supplied)"
                    )
            if warm_start_active:
                if not isinstance(probe_state_this, TorchPairSCIState):
                    raise TypeError(
                        "pair-mode warm-start requires probe_state to be a "
                        "TorchPairSCIState (got "
                        f"{type(probe_state_this).__name__!r})."
                    )
                _validate_fingerprints(
                    probe_state_this,
                    _as_integrals_array(one_body_tensor),
                    _as_integrals_array(two_body_tensor),
                    norb,
                    nelec,
                    solver_knob_snapshot,
                    current_fingerprint=get_integrals_fingerprint(),
                )
                _check_structural_pair_superset(
                    pair_basis,
                    probe_state_this.alpha_of_pair,
                    probe_state_this.beta_of_pair,
                    dev,
                )
        else:
            # In pair mode strings_a[i] and strings_b[i] together define pair i.
            # A basis wider than 64 orbitals carries each spin's strings as an
            # ``(M, n_limbs)`` limb matrix, the same representation the TPB path
            # uses; a single-word basis keeps the legacy 1-D arrays.
            a_arr, b_arr = _pair_arrays_from_strings(strings_a, strings_b, norb)
            # The pair-basis constructor rejects unequal-length inputs at the
            # boundary — surface it as a clear ValueError, not a silent shape
            # mismatch downstream.
            if a_arr.shape[0] != b_arr.shape[0]:
                raise ValueError(
                    f"pair mode requires len(strings_a) == len(strings_b) "
                    f"(got {a_arr.shape[0]} and {b_arr.shape[0]}) — each index "
                    f"defines one (α_i, β_i) pair."
                )
            if init_vector is not None:
                init_vector = _validate_init_vector(
                    init_vector,
                    a_arr.shape[0],
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )

            # --- warm-start / HF augmentation (pair mode) ------------------
            # On cold solves with probe_method='warm_start', optionally splice
            # the HF pair into the input so the cached eigvec's ground-state
            # overlap is protected against user-supplied cores that happen
            # to exclude HF (a Davidson failure mode observed in product mode).
            # On warm-extend, dedup+superset check against the cached pair
            # basis and canonicalize to lex-sorted order; the cached eigvec
            # will then embed by pair-tuple lookup below.
            cold_with_probe = probe_method == "warm_start" and probe_state_this is None
            warm_start_active = probe_method == "warm_start" and probe_state_this is not None
            if cold_with_probe and bool(probe_kwargs_in.get("include_hf_in_core", True)):
                aug_alpha, aug_beta, hf_info = _augment_pair_with_hf(
                    a_arr.tolist(), b_arr.tolist(), norb, nelec
                )
                if hf_info["hf_pair_added"]:
                    with self._print_lock:
                        print(
                            f"[probe:warm_start:pair] added HF pair "
                            f"(α=0x{hf_info['hf_alpha_string']:x}, "
                            f"β=0x{hf_info['hf_beta_string']:x}) "
                            f"to core (was not present in input); "
                            f"disable via probe_kwargs={{'include_hf_in_core': False}}"
                        )
                a_arr = np.asarray(aug_alpha, dtype=np.int64)
                b_arr = np.asarray(aug_beta, dtype=np.int64)

            if warm_start_active:
                # Cached state must be a TorchPairSCIState — reject a product-
                # mode state passed to a pair-mode call so the caller sees the
                # mismatch cleanly rather than an attribute error downstream.
                if not isinstance(probe_state_this, TorchPairSCIState):
                    raise TypeError(
                        "pair-mode warm-start requires probe_state to be a "
                        "TorchPairSCIState (got "
                        f"{type(probe_state_this).__name__!r}). A cached "
                        "product-mode TorchSCIState cannot warm-extend a pair "
                        "solve since it stores per-spin lists rather than a "
                        "pair list."
                    )
                _validate_fingerprints(
                    probe_state_this,
                    _as_integrals_array(one_body_tensor),
                    _as_integrals_array(two_body_tensor),
                    norb,
                    nelec,
                    solver_knob_snapshot,
                    current_fingerprint=get_integrals_fingerprint(),
                )
                union_alpha, union_beta, dedup_info = _dedup_and_check_pair_superset(
                    probe_state_this.alpha_of_pair,
                    probe_state_this.beta_of_pair,
                    a_arr,
                    b_arr,
                )
                _emit_pair_dedup_warning(dedup_info)
                a_arr = np.asarray(union_alpha, dtype=np.int64)
                b_arr = np.asarray(union_beta, dtype=np.int64)
                if self.verbose:
                    verbose_lines.append(
                        f"  [warm_start:pair] core N={dedup_info['cached_N']} "
                        f"→ extended N={dedup_info['extended_N']} "
                        f"(+{dedup_info['pairs_added']} pair(s))"
                    )

            pair_basis = pair_basis_from_arrays(a_arr, b_arr, n_orb=norb, device=dev)
            n_pairs = pair_basis.n_pairs
            # The flat pair lists are replicated per rank; nothing below reads
            # them, so release the 16 bytes/pair before the solve starts.
            del a_arr, b_arr

        # --- input validation (per-string popcount + bit_length) ----------
        # Reuse the product-mode validator on the *unique* per-spin arrays.
        # It catches malformed CI strings (wrong particle count, bits above
        # norb) before the excitation enumerator would crash with an
        # inscrutable IndexError.
        _validate_ci_basis(
            pair_basis.unique_alphas.cpu().numpy(),
            pair_basis.unique_betas.cpu().numpy(),
            norb,
            nelec,
            sub_idx=sub_idx,
        )

        # --- setup (helpers, hii, W) ---------------------------------------
        t0 = _time.perf_counter()
        pair_helpers = generate_pair_helpers(
            pair_basis,
            norb,
            device=dev,
            budget_bytes=(
                self.gdb_kwargs["budget_bytes"]
                if self.gdb_kwargs["budget_bytes"] is not None
                else self.memory_budget_bytes
            ),
            block_matvec=self.gdb_kwargs["block_matvec"],
            block_table_bytes=self.gdb_kwargs["block_table_bytes"],
        )
        # Owner-computes GDB: the flat pair list is partitioned
        # contiguously and each rank streams only its owned bra rows. Full RDMs
        # run on the tiled path from the staged exchanges (explicit or
        # structural basis); without a tile grid they keep the replicated
        # full-vector fallback. The spin_sq penalty and the <S^2> diagnostic run
        # on the tile when a grid reached the local path and on the 1-D block
        # otherwise (explicit lists only; a structural basis keeps its tile).
        dist_ctx = self._distributed_ctx
        _auto_block_cap = max(30, self.n_roots + 3)
        gdb_tile_spec = getattr(dist_ctx, "tile", None) if dist_ctx is not None else None
        _block_cap = (
            int(self.davidson_block_size)
            if self.davidson_block_size is not None
            else _auto_block_cap
        )
        # --- irregular pair lists: the Cartesian superset -------------------
        # A recovered or expanded pair list is an irregular subset of the
        # product of its unique strings: each string then has its own partner
        # set, so the exact block decomposition splits into dozens of tiny
        # groups (or is gated out), and the block/tile kernels cannot amortize
        # their per-group work. A tile grid needs that decomposition, so an
        # irregular list can otherwise not tile at all. Solve the *masked
        # product* instead: the extra pairs keep zero amplitude, the product
        # gives every kernel its native one-rectangle structure, and ``P H P``
        # on the product reproduces the operator on the caller's list exactly.
        # Only the owner-computes path masks, so the embedding is adopted below
        # only once that path is chosen; the candidate is built here so the
        # "auto" policy judges a tiled solve's real size.
        superset_eligible = (
            dist_ctx is not None
            and dist_ctx.matvec == "local"
            and gdb_tile_spec is not None
            and structural_spec is None
            and self.gdb_kwargs["block_matvec"] != "off"
            and not self.apply_ab_projection
            and not _pair_blocks_are_structured(pair_helpers.blocks)
        )
        superset_ratio = None
        if superset_eligible:
            # The product of the unique strings is known from the counts, so the
            # policy below can judge the size it would actually solve without
            # building the product; the embedding is built only once the tiled
            # path is chosen, so a declined "auto" pays nothing.
            superset_ratio = (
                int(pair_basis.n_unique_alphas) * int(pair_basis.n_unique_betas)
            ) / max(1, int(n_pairs))
            if superset_ratio > _PAIR_SUPERSET_MAX_RATIO:
                superset_eligible = False
        gdb_auto_declined = False
        if gdb_tile_spec == "auto":
            # "auto" is a policy, not a promise to tile: the 1-D block matvec
            # is faster at equal size, so a grid is used only when the
            # structures it removes would not fit the device budget.
            from torch_sci_solver.torch_impl.pair_tiles import (
                pair_replicated_bytes,
                resolve_auto_pair_tiling,
            )

            # The tiled operator walks the block decomposition, so a basis
            # without one (block_matvec off, or gated out by its size/byte
            # caps) cannot tile: auto must not choose a path that raises. A
            # superset candidate supplies the decomposition the exact build
            # cannot, so the policy judges the size it would actually solve.
            replicated_bytes = pair_replicated_bytes(pair_helpers)
            if superset_eligible:
                assert superset_ratio is not None
                replicated_bytes = int(replicated_bytes * superset_ratio)
            if (pair_helpers.blocks is None and not superset_eligible) or (
                not resolve_auto_pair_tiling(
                    world_size=dist_ctx.world_size,
                    n_pairs=(
                        int(pair_basis.n_unique_alphas) * int(pair_basis.n_unique_betas)
                        if superset_eligible
                        else n_pairs
                    ),
                    block_size=_block_cap,
                    itemsize=int(self.dtype.itemsize),
                    budget_bytes=self._memory_budget(dist_ctx),
                    replicated_bytes=replicated_bytes,
                )
            ):
                gdb_tile_spec = None
                gdb_auto_declined = True
        if (
            gdb_auto_declined
            and dist_ctx is not None
            and dist_ctx.matvec == "local"
            and (pair_helpers.blocks is not None or superset_eligible)
            and (structural_spec is not None or n_pairs > _block_cap)
            and (spin_sq is not None or self.compute_full_rdm or self.compute_spin_squared)
        ):
            # A (P, 1) uid grid is the pair-layout analogue of the TPB 1-D
            # split: it keeps the tiled post-processing on the owner-computes
            # path rather than falling back to the replicated full vector.
            gdb_tile_spec = (dist_ctx.world_size, 1)
        gdb_tiled = gdb_tile_spec is not None
        local_gdb = (
            dist_ctx is not None
            and dist_ctx.matvec == "local"
            # A structural basis has no explicit pair list for the 1-D sweep, so
            # its penalty runs on a tile or on the replicated operator.
            and (spin_sq is None or structural_spec is None or gdb_tiled)
            and (not self.compute_full_rdm or gdb_tiled)
            and (
                structural_spec is not None
                or n_pairs
                > (
                    int(self.davidson_block_size)
                    if self.davidson_block_size is not None
                    else _auto_block_cap
                )
            )
        )
        # The penalty runs on the owner-computes path in one of two layouts:
        # on the pair tile when a grid reached that operator, otherwise on the
        # 1-D streamed block (explicit pair bases only; a structural basis
        # keeps requiring a tile grid). A grid that did not reach the local
        # path keeps the replicated penalty operator.
        gdb_penalty_layout = None
        if spin_sq is not None and local_gdb:
            if gdb_tiled:
                gdb_penalty_layout = "tile"
            elif structural_spec is None:
                gdb_penalty_layout = "1d"
        gdb_tiled_penalty = gdb_penalty_layout == "tile"
        # Adopt the masked-product layout on the tiled owner-computes path. The
        # original basis is kept for the returned states, which are sliced back
        # to the caller's pairs.
        orig_pair_basis = pair_basis
        orig_n_pairs = int(n_pairs)
        superset_positions = None
        superset_active = None
        if local_gdb and gdb_tiled and superset_eligible:
            from torch_sci_solver.torch_impl.pairs import pair_superset_embedding

            superset, candidate_positions, candidate_active = pair_superset_embedding(
                pair_basis, n_orb=norb
            )
            rebuilt = generate_pair_helpers(
                superset,
                norb,
                device=dev,
                budget_bytes=(
                    self.gdb_kwargs["budget_bytes"]
                    if self.gdb_kwargs["budget_bytes"] is not None
                    else self.memory_budget_bytes
                ),
                block_matvec="on",
                block_table_bytes=self.gdb_kwargs["block_table_bytes"],
            )
            if rebuilt.blocks is None:
                if pair_helpers.blocks is None:
                    raise ValueError(
                        "the Cartesian superset of this pair list produced no block "
                        "decomposition; raise gdb_kwargs['block_table_bytes'] (its "
                        "byte cap gated the build out)"
                    )
                # An exact (if unstructured) decomposition exists, so a tight
                # block-table cap keeps the slower but correct layout.
            else:
                pair_basis = superset
                n_pairs = int(superset.n_pairs)
                pair_helpers = rebuilt
                superset_positions = candidate_positions
                superset_active = candidate_active
        if (
            self.gdb_kwargs.get("tile_move_cache_bytes") is not None
            and dist_ctx is not None
            and not (local_gdb and gdb_tile_spec is not None)
        ):
            raise ValueError(
                "gdb_kwargs['tile_move_cache_bytes'] is only honored by the tiled "
                "owner-computes GDB path (distributed matvec='local', tile "
                "configured, and basis larger than one Davidson block). Use "
                "gdb_kwargs['move_cache_bytes'] for the 1-D block path."
            )
        if self.gdb_kwargs.get("move_cache_bytes") is not None and pair_helpers.blocks is None:
            raise ValueError(
                "gdb_kwargs['move_cache_bytes'] caches the block matvec's move "
                "tables, but this basis has no block decomposition (block_matvec "
                "is off or was gated out by its size/byte caps). Pass "
                "gdb_kwargs={'block_matvec': 'on'} or drop the cache budget."
            )
        local_bounds = None
        local_partition = None
        local_operator = None
        local_project_fn = None
        hii_local = None
        superset_local_active = None
        row_start = row_stop = 0
        tiled = False
        if local_gdb:
            from torch_sci_solver.torch_impl.distributed_davidson import (
                partition_vector,
                partition_vector_weighted,
            )
            from torch_sci_solver.torch_impl.distributed_pairs import (
                LocalPairOperator,
                LocalPairSwapProjector,
            )
            from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_local

            assert dist_ctx is not None  # local_gdb implies a distributed context
            tile_spec = gdb_tile_spec
            tiled = tile_spec is not None
            assert superset_active is None or tiled  # the superset needs the tile mask
            if tiled:
                # The tile operator reproduces the block-decomposed matvec
                # fragment by fragment, so the decomposition must exist.
                if pair_helpers.blocks is None:
                    raise ValueError(
                        "distributed GDB tile partitioning needs the pair block "
                        "decomposition; pass gdb_kwargs={'block_matvec': 'on'}"
                    )
                from torch_sci_solver.torch_impl.distributed_pairs_tile import (
                    LocalPairTileOperator,
                    LocalPairTileSwapProjector,
                )
                from torch_sci_solver.torch_impl.pair_tiles import (
                    pair_tile_bounds,
                    pair_tile_partition,
                )

                grid = None if tile_spec in ("auto", "force") else tuple(int(p) for p in tile_spec)
                # Pair-count weights keep the per-tile work comparable; the even
                # uid cut leaves one slab with most of the pairs on tier bases.
                local_partition = pair_tile_partition(pair_helpers, dist_ctx.world_size, grid=grid)
                local_bounds = pair_tile_bounds(local_partition, pair_basis, dist_ctx.rank)
                owned = local_bounds.pair_index
                if superset_active is not None:
                    superset_local_active = superset_active[owned].to(device=dev)
                owned_alpha, owned_beta = _pair_strings_at(pair_basis, owned)
                hii_local = make_diag_terms_pair_local(
                    owned_alpha,
                    owned_beta,
                    norb,
                    integrals,
                    chunk_size=self.chunk_size,
                )
                local_operator = LocalPairTileOperator(
                    hii_local,
                    pair_basis,
                    pair_helpers,
                    norb,
                    integrals,
                    local_bounds,
                    local_partition,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    group=dist_ctx.group,
                    chunk_size=self.chunk_size,
                    move_cache_bytes=self.gdb_kwargs["move_cache_bytes"],
                    needed_staging=self.gdb_kwargs["needed_staging"],
                    triton_matvec=self.matvec_impl == "triton",
                )
                if self.apply_ab_projection:
                    k_swap, tau_signs = _pair_ab_swap(
                        pair_basis, norb, dev, self.dtype, pair_indices=owned
                    )
                    if k_swap is not None:
                        local_project_fn = LocalPairTileSwapProjector(
                            pair_basis,
                            k_swap,
                            tau_signs,
                            local_bounds,
                            local_partition,
                            rank=dist_ctx.rank,
                            world_size=dist_ctx.world_size,
                            group=dist_ctx.group,
                            tag=4096,
                        ).__call__
            elif pair_helpers.blocks is not None and structural_spec is None:
                # The same-spin enumeration is sized by each rank's owned
                # rows/columns, not by its pair count, so an equal-count split
                # can leave ranks several times apart. Split on the replicated
                # candidate-cost weights instead; the bounds remain contiguous
                # and every consumer below uses this same partition.
                from torch_sci_solver.torch_impl.mult_pair_blocks import pair_work_weights

                local_partition = partition_vector_weighted(
                    n_pairs,
                    dist_ctx.world_size,
                    pair_work_weights(pair_helpers),
                )
                local_bounds = local_partition.for_rank(dist_ctx.rank)
                row_start, row_stop = local_bounds.start, local_bounds.stop
            else:
                # A structural basis has no per-pair weight vector, so it takes
                # the equal-count partition (the input gather accepts any
                # rank-ordered contiguous tiling).
                local_bounds = partition_vector(n_pairs, dist_ctx.world_size, dist_ctx.rank)
                row_start, row_stop = local_bounds.start, local_bounds.stop
            if not tiled:
                local_alpha, local_beta = _pair_strings_at(
                    pair_basis, torch.arange(row_start, row_stop)
                )
                hii_local = make_diag_terms_pair_local(
                    local_alpha,
                    local_beta,
                    norb,
                    integrals,
                    chunk_size=self.chunk_size,
                )
                local_operator = LocalPairOperator(
                    hii_local,
                    pair_basis,
                    pair_helpers,
                    norb,
                    integrals,
                    local_bounds,
                    rank=dist_ctx.rank,
                    world_size=dist_ctx.world_size,
                    group=dist_ctx.group,
                    chunk_size=self.chunk_size,
                    triton_matvec=self.matvec_impl == "triton",
                    move_cache_bytes=self.gdb_kwargs["move_cache_bytes"],
                )
                if self.apply_ab_projection:
                    k_swap, tau_signs = _pair_ab_swap(
                        pair_basis, norb, dev, self.dtype, pair_range=(row_start, row_stop)
                    )
                    if k_swap is not None:
                        local_project_fn = LocalPairSwapProjector(
                            pair_basis,
                            k_swap,
                            tau_signs,
                            local_bounds,
                            rank=dist_ctx.rank,
                            world_size=dist_ctx.world_size,
                            group=dist_ctx.group,
                        ).__call__
            hii = None  # the full diagonal is never materialized on this path
        else:
            hii = make_diag_terms_pair_torch(
                pair_basis, norb, integrals, chunk_size=self.chunk_size
            )
        # Seed selection. ``smallest_int`` seeds a single-root solve at the
        # lowest-diagonal pair (the global argmin over the ranks' diagonal
        # blocks on the owner-computes path), as the TPB cold start does: pairs
        # drawn from string clusters more than a double excitation apart split H
        # into blocks, Davidson does not leave its seed's block, and the
        # smallest-int pair need not share a block with the ground state. A
        # multi-root seed block already adds the lowest-diagonal pairs, so it
        # keeps the smallest-int pair (``_smallest_int_pair_index``). The
        # ``connected`` strategy seeds at the pair with the highest degree
        # in the pair-helper coupling tables (α-single/double, β-single/
        # double, αβ-double). Davidson is basis-ordering-invariant, so we
        # do this via ``W[best_idx] = 1.0`` instead of permuting the pair
        # basis and rebuilding CSRs.
        if superset_positions is not None:
            best_pair_idx = int(superset_positions[0].item())
        elif self.n_roots > 1:
            best_pair_idx = _smallest_int_pair_index(pair_basis)
        elif hii is not None:
            best_pair_idx = int(torch.argmin(hii))
        elif hii_local is not None:
            from torch_sci_solver.torch_impl.distributed_davidson import (
                _index_layout,
                global_smallest_hii,
            )

            best_pair_idx = global_smallest_hii(
                hii_local,
                _index_layout(n_pairs, local_bounds, dist_ctx),
                k=1,
                group=dist_ctx.group,
            )[0][0]
        else:
            best_pair_idx = _smallest_int_pair_index(pair_basis)
        t_seed = 0.0
        if self.seed_strategy == "connected":
            _t0 = _time.perf_counter()
            deg = torch.zeros(n_pairs, dtype=torch.int64, device=dev)
            for stream in (
                pair_helpers.asingles,
                pair_helpers.adoubles,
                pair_helpers.bsingles,
                pair_helpers.bdoubles,
                pair_helpers.ab_doubles,
            ):
                for tbl in stream():
                    kb = tbl["k_bra"]
                    deg.index_add_(0, kb, torch.ones_like(kb))
            if superset_active is not None:
                # Inactive product pairs have full partner sets, so their degree
                # would win the seed; keep the start inside the caller's list.
                deg[~superset_active] = -1
            best_pair_idx = int(torch.argmax(deg).item())
            t_seed = _time.perf_counter() - _t0
            with self._stats_lock:
                self.stats["t_seed"] += t_seed
        if local_gdb:
            assert local_bounds is not None and dist_ctx is not None

            if tiled:
                from torch_sci_solver.torch_impl.distributed_pairs_tile import (
                    pair_tile_init_vector_local,
                )

                def _cold_local() -> torch.Tensor:
                    """Owned-tile cold start: global seed entry plus hashed noise."""
                    return pair_tile_init_vector_local(
                        pair_basis,
                        local_bounds,
                        seed_index=best_pair_idx,
                        device=dev,
                        dtype=self.dtype,
                        rank=dist_ctx.rank,
                        world_size=dist_ctx.world_size,
                        group=dist_ctx.group,
                    )

                owned_alpha, owned_beta = _pair_strings_at(pair_basis, local_bounds.pair_index)
            else:
                from torch_sci_solver.torch_impl.distributed_pairs import (
                    pair_basis_init_vector_local,
                )

                def _cold_local() -> torch.Tensor:
                    """Owned-block cold start: global seed entry plus hashed noise."""
                    return pair_basis_init_vector_local(
                        n_pairs,
                        local_bounds,
                        seed_index=best_pair_idx,
                        device=dev,
                        dtype=self.dtype,
                        rank=dist_ctx.rank,
                        world_size=dist_ctx.world_size,
                        group=dist_ctx.group,
                    )

                owned_alpha, owned_beta = _pair_strings_at(
                    pair_basis, torch.arange(row_start, row_stop)
                )

            warm = False
            if warm_start_active:
                # Row-restricted warm embed: resolve the owned pairs and the
                # cached pairs to the basis' (α_uid, β_uid) key and gather the
                # cached amplitude of every owned pair. The key is limb-aware,
                # so a wide pair basis keys exactly like a single-word one; the
                # earlier dedup+superset check guarantees every cached pair is
                # present (newly added pairs simply keep their zero here).
                from torch_sci_solver.torch_impl.probe_pairs import _pair_uid_keys

                local_keys = _pair_uid_keys(pair_basis, owned_alpha, owned_beta, device=dev)
                cached_keys = _pair_uid_keys(
                    pair_basis,
                    probe_state_this.alpha_of_pair,
                    probe_state_this.beta_of_pair,
                    device=dev,
                )
                n_cached = int(cached_keys.numel())
                if n_cached:
                    order = torch.argsort(cached_keys)
                    sorted_keys = cached_keys[order]
                    pos = torch.searchsorted(sorted_keys, local_keys)
                    clipped = pos.clamp_max(n_cached - 1)
                    src = torch.where(
                        sorted_keys[clipped] == local_keys,
                        order[clipped],
                        torch.full_like(pos, -1),
                    )
                else:
                    src = torch.full((local_keys.numel(),), -1, dtype=torch.int64, device=dev)
                W = torch.zeros(local_bounds.local_size, dtype=self.dtype, device=dev)
                have = src >= 0
                if bool(have.any()):
                    old = probe_state_this.eigvec_inter
                    W[have] = old[src[have].cpu()].to(device=dev, dtype=self.dtype)
                _warm_norm = _global_norm_local(W, dist_ctx)
                if float(_warm_norm.item()) == 0.0:
                    W = _cold_local()
                else:
                    W = W / _warm_norm
                    warm = True
            elif init_vector is not None:
                if structural_spec is not None:
                    # Indexed by structural pair: the owned entries are the
                    # owned pair indices themselves.
                    k = (
                        local_bounds.pair_index.to(device=dev, dtype=torch.int64)
                        if tiled
                        else torch.arange(row_start, row_stop, dtype=torch.int64, device=dev)
                    )
                else:
                    k = _first_index(
                        _pair_key_tensor(
                            strings_a, strings_b, width=pair_basis.n_limbs, device=dev
                        ),
                        _pair_key_tensor(
                            pair_basis.alpha_of_pair,
                            pair_basis.beta_of_pair,
                            width=pair_basis.n_limbs,
                            device=dev,
                        ),
                        device=dev,
                    )
                    k = k[local_bounds.pair_index] if tiled else k[row_start:row_stop]

                def _map_init_local(row: np.ndarray) -> torch.Tensor:
                    return _gather_or_zero(torch.as_tensor(row, dtype=self.dtype, device=dev), k)

                W, warm = _resolve_start_vectors_local(
                    init_vector, _map_init_local, local_project_fn, _cold_local, dist_ctx
                )
            else:
                W = _cold_local()
        elif warm_start_active:
            # Embed the cached eigvec into the extended pair basis. The cached
            # pairs are keyed by their (α_uid, β_uid) pair key inside the union
            # (a limb-aware key, so wide bases embed like single-word ones);
            # newly-added pairs get zero. Normalization is done inside the
            # embed helper.
            W = _embed_eigvec_pair_into_extended(
                probe_state_this.eigvec_inter,
                probe_state_this.alpha_of_pair,
                probe_state_this.beta_of_pair,
                pair_basis.alpha_of_pair.cpu().tolist(),
                pair_basis.beta_of_pair.cpu().tolist(),
                device=dev,
                dtype=self.dtype,
            )
        elif init_vector is not None:
            # Caller's pair amplitudes (already interleaved-Wick, the convention
            # this mode returns), re-indexed to the deduped lex-sorted pair basis.
            # Projected and normalized right before Davidson.
            k = _first_index(
                _pair_key_tensor(strings_a, strings_b, width=pair_basis.n_limbs, device=dev),
                _pair_key_tensor(
                    pair_basis.alpha_of_pair,
                    pair_basis.beta_of_pair,
                    width=pair_basis.n_limbs,
                    device=dev,
                ),
                device=dev,
            )

            def _map_init(row: np.ndarray) -> torch.Tensor:
                return _gather_or_zero(torch.as_tensor(row, dtype=self.dtype, device=dev), k)

            # The 1-D entry's mapped vector; per-root rows are mapped at resolve time.
            W = _map_init(init_vector if init_vector.ndim == 1 else init_vector[0])
        else:
            W = self._cold_pair_init_vector(n_pairs, best_pair_idx, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_setup = _time.perf_counter() - t0
        with self._stats_lock:
            self.stats["t_setup"] += t_setup

        # Pick Davidson davidson_block_size using the flat N. _pick_num_block only
        # reads M_a and M_b through their product, so passing (n_pairs, 1)
        # gives the right budget. The verbose print still names them
        # M_a × M_b for consistency with product mode; the "= N" is the
        # thing the user cares about.
        chosen_nb = self._pick_num_block(n_pairs, 1, device=dev, n_roots=self.n_roots)
        if self._distributed_ctx is not None:
            chosen_nb = min(max(chosen_nb, self.n_roots + 3), n_pairs)
        if self.verbose:
            free_after_setup = self._free_bytes(device=dev)
            sub_b, work_b = self._davidson_element_bytes()
            dav_estimate = (2 * chosen_nb * sub_b + 4 * work_b) * n_pairs
            verbose_lines.append(
                f"  [sci_solver:pair] subspace {sub_idx}: "
                f"N={n_pairs} pairs "
                f"(unique α={pair_basis.n_unique_alphas}, "
                f"β={pair_basis.n_unique_betas})  "
                f"| dev={dev}  "
                f"| davidson_block_size={chosen_nb}  "
                f"| free={self._fmt(free_after_setup)}  "
                f"| Davidson est ≈ {self._fmt(dav_estimate)}"
            )
            # Coupling counts need a full enumeration pass (nothing is stored),
            # so they are computed only when the caller asked for verbose output.
            # A structural basis has no streamed tables to enumerate.
            if structural_spec is None:
                _nnz = pair_helpers.nnz()
                verbose_lines.append(
                    f"               αs={_nnz['asingles']}, "
                    f"αd={_nnz['adoubles']}, "
                    f"βs={_nnz['bsingles']}, "
                    f"βd={_nnz['bdoubles']}, "
                    f"αβ={_nnz['ab_doubles']}"
                )
            verbose_lines.append(f"               setup: {t_setup:.2f}s")

        # --- spin_sq penalty setup (pair mode) -----------------------------
        # Same policy as product mode: build the S² operator + σ bridge +
        # scratch buffers *before* picking davidson_block_size so the Davidson
        # subspace sizing sees the post-penalty free-memory budget.
        _S2_op = None
        _sigma_pair = None
        _S2v_buf = None
        _v_block_buf = None
        t_penalty_setup = 0.0
        if spin_sq is not None and gdb_penalty_layout is None:
            from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair
            from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

            _t0 = _time.perf_counter()
            _S2_op = PairSpinSqOperator(pair_basis, norb, n_a, n_b, device=dev, dtype=self.dtype)
            _sigma_pair = (
                _inter_to_block_signs_pair(
                    pair_basis.alpha_of_pair.to(dev),
                    pair_basis.beta_of_pair.to(dev),
                    norb,
                )
                .to(dtype=self.dtype)
                .contiguous()
            )
            _S2v_buf = torch.empty(n_pairs, dtype=self.dtype, device=dev)
            _v_block_buf = torch.empty(n_pairs, dtype=self.dtype, device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_penalty_setup = _time.perf_counter() - _t0
            with self._stats_lock:
                self.stats["t_spin_sq"] += t_penalty_setup

        # --- Davidson ------------------------------------------------------
        # The local path never builds the full diagonal or the full matvec.
        if local_gdb:
            hii_for_davidson = hii_local
        else:
            assert hii is not None  # the local path is the only one without it
            hii_for_davidson = hii

        # One move-table cache per subspace so the Davidson iterations reuse
        # the stage-independent enumeration instead of rebuilding it per call.
        _pair_move_cache = None
        if self.gdb_kwargs.get("move_cache_bytes") is not None:
            from torch_sci_solver.torch_impl.mult_pair_blocks import _MoveCache

            _pair_move_cache = _MoveCache(int(self.gdb_kwargs["move_cache_bytes"]), evict=True)

        def matvec(v):
            """Matvec."""
            assert hii is not None  # never called on the local path
            return mult_pair_torch(
                hii,
                v,
                pair_basis,
                pair_helpers,
                norb,
                integrals,
                chunk_size=self.chunk_size,
                triton_matvec=self.matvec_impl == "triton",
                cache=_pair_move_cache,
            )

        if spin_sq is not None and gdb_penalty_layout is None:
            _lam = call_spin_lambda
            _target = float(spin_sq_target)
            _bare_matvec = matvec

            def matvec(v):  # redef intentional — augmented matvec
                """Matvec."""
                Hv = _bare_matvec(v)
                torch.mul(v, _sigma_pair, out=_v_block_buf)
                _S2_op.apply(_v_block_buf, out=_S2v_buf)
                _S2v_buf.mul_(_sigma_pair)  # σ² = 1 → back to interleaved-Wick
                Hv.add_(_S2v_buf, alpha=_lam)
                if _target != 0.0:
                    Hv.add_(v, alpha=-_lam * _target)
                return Hv

            hii_for_davidson = hii + _lam * (_S2_op.diagonal().real - _target)
            if self.verbose:
                verbose_lines.append(
                    f"               spin_sq penalty: ENABLED "
                    f"(target={_target:g}, λ={_lam:g}, "
                    f"S² build {t_penalty_setup:.2f}s, "
                    f"|off-diag|={int(_S2_op.k_src.shape[0])})"
                )

        # --- αβ-swap projection (pair mode) --------------------------------
        # In product mode this projects onto the closed-shell singlet
        # subspace via ``P(v)[i,j] = (v[i,j] + τ[i,j] · v[j,i]) / 2`` on
        # the (M_a, M_b) grid; requires the basis to be αβ-symmetric
        # (adets == bdets). In pair mode we need the pair set closed
        # under (α, β) ↔ (β, α) swap — for each pair k = (α, β), the
        # swapped pair (β, α) must also be in the basis. When this
        # closure holds we can enumerate a ``k_swap`` permutation and
        # apply the same projector formula element-wise. When it
        # doesn't, we skip (silently) rather than corrupt Davidson.
        project_fn = local_project_fn
        if project_fn is None and self.apply_ab_projection:
            k_swap, tau_signs = _pair_ab_swap_permutation(pair_basis, norb, dev, self.dtype)
            if k_swap is not None:

                def project_fn(v: torch.Tensor) -> torch.Tensor:
                    """Project fn."""
                    return 0.5 * (v + tau_signs * v[k_swap])

                if self.verbose:
                    verbose_lines.append(
                        "               αβ-swap projection: ENABLED (pair set closed under α↔β)"
                    )
            elif self.verbose:
                verbose_lines.append(
                    "               αβ-swap projection: DISABLED "
                    "(pair set not closed under α↔β swap)"
                )
        elif project_fn is not None and self.verbose:
            verbose_lines.append("               αβ-swap projection: ENABLED (local-block)")
        elif self.verbose:
            verbose_lines.append(
                "               αβ-swap projection: DISABLED (apply_ab_projection=False)"
            )
        if superset_local_active is not None:
            assert project_fn is None  # superset adoption excludes the AB projector
            # Masked-product layout: every Davidson vector must stay supported
            # on the caller's pairs, so the driver's projection hook carries
            # the membership mask. It is applied to each matvec product and to
            # every new search direction, which turns the product operator into
            # ``P H P`` and keeps the inactive (zero-amplitude) pairs out of the
            # subspace, where they would otherwise act as spurious zero modes.
            _superset_mask = superset_local_active

            def project_fn(v: torch.Tensor) -> torch.Tensor:
                """Zero this rank's pair-superset entries outside the caller's list."""
                return torch.where(_superset_mask, v, torch.zeros_like(v))

            if self.verbose:
                verbose_lines.append(
                    f"               pair superset: ACTIVE ({orig_n_pairs:,} → "
                    f"{n_pairs:,} pairs, masked)"
                )

        if not local_gdb:
            # The cached-state embed above already set the start vector; a
            # caller-supplied init_vector may still fail to overlap and fall
            # back to the cold seed below.
            warm = warm_start_active
            if init_vector is not None:
                _cold_gdb = lambda: self._cold_pair_init_vector(  # noqa: E731
                    n_pairs, best_pair_idx, dev
                )
                if init_vector.ndim == 1:
                    W, warm = _resolve_start_vector(W, project_fn, _cold_gdb, init_vector)
                else:
                    W, warm = _resolve_start_vectors(init_vector, _map_init, project_fn, _cold_gdb)
                if not warm and self.verbose:
                    verbose_lines.append(
                        f"               init_vectors: no overlap with this basis "
                        f"(< {_INIT_VECTOR_MIN_NORM:g} of its norm left); cold start"
                    )

        _gdb_hii_for_dav = hii_local
        _gdb_op_for_dav = local_operator
        _gdb_s2_op = None
        if gdb_tiled_penalty:
            assert hii_local is not None and local_operator is not None
            from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
                pair_tiled_s2_diagonal,
            )

            _gdb_hii_for_dav = hii_local + call_spin_lambda * (
                pair_tiled_s2_diagonal(local_operator) - float(spin_sq_target)
            )
            _gdb_op_for_dav = _PairTiledPenalizedOperator(
                local_operator, call_spin_lambda, float(spin_sq_target)
            )
            verbose_lines.append(
                f"               spin_sq penalty: ENABLED on the pair tile "
                f"(target={float(spin_sq_target):g}, λ={call_spin_lambda:g})"
            )
        elif gdb_penalty_layout == "1d" or (
            local_gdb and not gdb_tiled and self.compute_spin_squared
        ):
            # The same flat sweep serves the penalty and the <S^2> diagnostic;
            # only the penalty augments the diagonal and the operator.
            assert hii_local is not None and local_operator is not None
            from torch_sci_solver.torch_impl.pair_tiled_postprocess import (
                PairLocalSpinSquared,
            )

            _t0_pen = _time.perf_counter()
            _gdb_s2_op = PairLocalSpinSquared(
                local_operator,
                sign_dtype=(self.davidson_dtype or self.dtype),
            )
            with self._stats_lock:
                self.stats["t_spin_sq"] += _time.perf_counter() - _t0_pen
            if gdb_penalty_layout == "1d":
                _gdb_hii_for_dav = hii_local + call_spin_lambda * (
                    _gdb_s2_op.diagonal() - float(spin_sq_target)
                )
                _gdb_op_for_dav = _PairLocalPenalizedOperator(
                    local_operator, _gdb_s2_op, call_spin_lambda, float(spin_sq_target)
                )
                verbose_lines.append(
                    f"               spin_sq penalty: ENABLED on the 1-D pair block "
                    f"(target={float(spin_sq_target):g}, λ={call_spin_lambda:g})"
                )
        if superset_local_active is not None and _gdb_hii_for_dav.numel() > 0:
            # Inactive superset pairs must not be chosen as one-hot starts (the
            # drivers seed at the smallest diagonal entries) or resonate with
            # the preconditioner. They carry no amplitude, so a large finite
            # diagonal leaves the masked operator unchanged; infinity would
            # turn ``hii * 0`` into NaN.
            _inactive_fill = 4.0 * float(_gdb_hii_for_dav.abs().max().item()) + 1.0
            _gdb_hii_for_dav = _gdb_hii_for_dav.clone()
            _gdb_hii_for_dav[~superset_local_active] = _inactive_fill

        if structural_spec is not None:
            basis_arrays = (
                pair_basis.unique_alphas,
                pair_basis.unique_betas,
                pair_basis.pair_bases,
                pair_basis.alpha_block_ids,
                pair_basis.beta_block_ids,
            )
            basis_kind = "structural"
        else:
            basis_arrays = (pair_basis.alpha_of_pair, pair_basis.beta_of_pair)
            basis_kind = "pairs"
        checkpoint_dimension = n_pairs
        if superset_active is not None:
            # The solve runs on the padded product, but a resume must belong to
            # the caller's list: two recovered sets can share one product (and
            # rank partition), and their masked operators differ, so the saved
            # subspace block cannot be reused across them.
            basis_arrays = (orig_pair_basis.alpha_of_pair, orig_pair_basis.beta_of_pair)
            checkpoint_dimension = orig_n_pairs
        checkpoint = self._make_davidson_checkpoint(
            sub_idx=sub_idx,
            context=dist_ctx,
            basis_kind=basis_kind,
            basis_arrays=basis_arrays,
            n_alpha=int(pair_basis.unique_alphas.shape[0]),
            n_beta=int(pair_basis.unique_betas.shape[0]),
            particle_sector=(n_a, n_b),
            dimension=checkpoint_dimension,
            norb=norb,
            partition=local_partition if local_gdb else None,
            layout_kind="local" if local_gdb else "replicated",
            chosen_nb=chosen_nb,
            spin_sq_target=spin_sq_target,
            spin_penalty_lambda=call_spin_lambda,
            get_integrals_fingerprint=get_integrals_fingerprint,
            solver_knob_snapshot=solver_knob_snapshot,
        )
        t0 = _time.perf_counter()
        if local_gdb:
            assert hii_local is not None and local_bounds is not None and dist_ctx is not None
            _dav_result, perf_dav = self._run_davidson_distributed_local(
                _gdb_hii_for_dav,
                W,
                _gdb_op_for_dav,
                local_bounds,
                context=dist_ctx,
                call_max_iter=call_max_iter,
                chosen_nb=chosen_nb,
                call_eps=call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                warm=warm,
                checkpoint=checkpoint,
            )
        elif self._distributed_ctx is not None:
            # Replicated fallback (small basis / post-processing without a tile
            # grid); see the TPB branch above for the complex/narrow-subspace
            # support and the missed-root limitation.
            _dav_result, perf_dav = self._run_davidson_distributed(
                hii_for_davidson,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                context=self._distributed_ctx,
                warm=warm,
                checkpoint=checkpoint,
            )
        else:
            _dav_result, perf_dav = self._run_davidson(
                hii_for_davidson,
                W,
                matvec,
                call_max_iter,
                chosen_nb,
                call_eps,
                call_de_tol=call_de_tol,
                project_fn=project_fn,
                warm=warm,
            )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_dav = _time.perf_counter() - t0
        with self._stats_lock:
            self.stats["t_davidson"] += t_dav

        if local_gdb:
            assert local_bounds is not None and dist_ctx is not None
            return self._finish_pair_local_subspace(
                sub_idx=sub_idx,
                dav_result=_dav_result,
                perf_dav=perf_dav,
                pair_basis=pair_basis,
                bounds=local_bounds,
                context=dist_ctx,
                norb=norb,
                nelec=(n_a, n_b),
                dev=dev,
                t_setup=t_setup,
                t_dav=t_dav,
                chosen_nb=chosen_nb,
                warm=warm,
                build_output_states=build_output_states,
                get_integrals_fingerprint=get_integrals_fingerprint,
                solver_knob_snapshot=solver_knob_snapshot,
                verbose_lines=verbose_lines,
                block_stats=(
                    pair_helpers.blocks.stats if pair_helpers.blocks is not None else None
                ),
                partition=local_partition,
                operator=local_operator,
                penalty=(
                    (call_spin_lambda, float(spin_sq_target))
                    if gdb_penalty_layout is not None
                    else None
                ),
                s2_operator=_gdb_s2_op,
                penalty_layout=gdb_penalty_layout,
                carry_basis=(orig_pair_basis if superset_active is not None else None),
                carry_positions=superset_positions,
            )

        # Match the TPB result contract: one result per root, with expensive
        # RDMs, diagnostics and the warm-start snapshot kept on root 0.
        energies_raw, all_eigvecs, info = _dav_result
        if not isinstance(energies_raw, torch.Tensor):
            energies_raw = torch.tensor([energies_raw], dtype=torch.float64)
            all_eigvecs = all_eigvecs.unsqueeze(0)  # (1, n_pairs)

        all_sci_results: List[SCIResultWithConvergence] = []
        t_occ = 0.0
        t_rdm = 0.0
        # MPS solves in float32: report float64 CPU Rayleigh quotients instead of
        # the float32 Ritz values (see _refines_fp32_energy).
        energy_matvec64 = None
        if self._refines_fp32_energy(call_eps) and not local_gdb:
            cpu = torch.device("cpu")
            integrals64 = self._cpu_wide_integrals(one_body_tensor, two_body_tensor, spinless=False)
            basis64 = _tensors_to(pair_basis, cpu)
            helpers64 = generate_pair_helpers(
                basis64,
                norb,
                device=cpu,
                budget_bytes=(
                    self.gdb_kwargs["budget_bytes"]
                    if self.gdb_kwargs["budget_bytes"] is not None
                    else self.memory_budget_bytes
                ),
                block_matvec=self.gdb_kwargs["block_matvec"],
                block_table_bytes=self.gdb_kwargs["block_table_bytes"],
            )
            hii64 = make_diag_terms_pair_torch(
                basis64, norb, integrals64, chunk_size=self.chunk_size
            )

            def energy_matvec64(v: torch.Tensor) -> torch.Tensor:
                """The float64 pair matvec on the CPU copy of the basis."""
                return mult_pair_torch(
                    hii64, v, basis64, helpers64, norb, integrals64, chunk_size=self.chunk_size
                )

            perf_dav["davidson_fp32_energy"] = "rq64"
        for root_idx in range(all_eigvecs.shape[0]):
            energy = float(energies_raw[root_idx].item())
            eigvec = all_eigvecs[root_idx]  # (n_pairs,)

            # --- strip spin_sq penalty from the reported energy ---------------
            # Davidson minimized H' = H + λ(S² − target); report the physical
            # ⟨ψ|H|ψ⟩ = E_aug − λ(⟨S²⟩ − target). Same one-S²-matvec trick as
            # product mode; buffers are reused.
            if _S2_op is not None:
                torch.mul(eigvec, _sigma_pair, out=_v_block_buf)
                _S2_op.apply(_v_block_buf, out=_S2v_buf)
                # vdot + .real: ⟨x|S²|x⟩ is Hermitian-conjugate-linear and real
                # (== dot for the real path); plain dot/float() is wrong for
                # complex eigvecs.
                _s2_value = float(torch.vdot(_v_block_buf, _S2v_buf).real.item())
                _norm_sq = float(torch.vdot(eigvec, eigvec).real.item())
                if _norm_sq > 0.0:
                    _s2_value /= _norm_sq
                energy = float(energy) - call_spin_lambda * (_s2_value - float(spin_sq_target))
                if root_idx == 0:
                    with self._stats_lock:
                        self.stats["spin_sq_final"] = _s2_value
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
            if energy_matvec64 is not None:
                # ⟨ψ|H|ψ⟩ of the bare Hamiltonian is the physical energy itself,
                # so it also replaces the penalty-stripped value.
                energy = _rayleigh_quotient_wide(energy_matvec64, eigvec, self._wide_dtype())

            # --- occupancies (diagonal 1-RDM) ---------------------------------
            t0 = _time.perf_counter()
            occ_a_t, occ_b_t = make_one_rdm_diag_pair_torch(
                eigvec,
                pair_basis.alpha_of_pair,
                pair_basis.beta_of_pair,
                norb,
            )
            occupancies = (occ_a_t.cpu().numpy(), occ_b_t.cpu().numpy())
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_occ += _time.perf_counter() - t0

            # --- full RDMs (optional) ------------------------------------------
            # Same policy as product mode: gated on ``compute_full_rdm``. In
            # pair mode we build γα, γβ, Γaa, Γab, Γbb via the pair-keyed
            # RDM builders, then combine into (rdm1, rdm2) with the same
            # PySCF spin-summed convention the product branch returns.
            rdm1 = None
            rdm2 = None
            if self.compute_full_rdm and root_idx == 0:
                t0 = _time.perf_counter()
                ga, gb = make_one_rdm_pair_torch(
                    eigvec, pair_basis, pair_helpers, norb, chunk_size=self.chunk_size
                )
                Gaa, Gab, Gbb = make_two_rdm_pair_torch(
                    eigvec, pair_basis, pair_helpers, norb, chunk_size=self.chunk_size
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                rdm1 = (ga + gb).cpu().numpy()
                rdm2 = (Gaa + Gab + Gab.permute(2, 3, 0, 1) + Gbb).cpu().numpy()
                del ga, gb, Gaa, Gab, Gbb
                t_rdm = _time.perf_counter() - t0
                with self._stats_lock:
                    self.stats["t_rdm"] += t_rdm

            # --- PairSCIState (flat length-N) ---------------------------------
            # Pair mode's SCIResult carries a length-N flat amplitudes vector
            # with parallel length-N ci_strs_a / ci_strs_b arrays — the pair-
            # native form. A previous version of this code allocated an
            # ``(M_α_unique, M_β_unique)`` sparse 2D matrix to be
            # bit-shape-compatible with qiskit_addon_sqd.fermion.SCIState (so
            # ``diagonalize_fermionic_hamiltonian``'s carryover step could
            # unpack ``.shape[1]``). At large scale
            # (``M_α_unique ≈ M_β_unique ≈ 200k``) that allocation is 320 GB
            # per iteration — an out-of-memory disaster. Pair mode is not a
            # drop-in for ``diagonalize_fermionic_hamiltonian``'s outer loop
            # (see the ``mode`` docstring); use
            # :func:`diagonalize_fermionic_hamiltonian_pairs` instead, which
            # consumes this flat form directly.
            eigvec_np = eigvec.detach().cpu().numpy()
            pair_strs_a, pair_strs_b = _pair_strs_arrays(
                pair_basis.alpha_of_pair.cpu().numpy(), pair_basis.beta_of_pair.cpu().numpy()
            )
            sci_state = PairSCIState(
                amplitudes=eigvec_np,
                ci_strs_a=pair_strs_a,
                ci_strs_b=pair_strs_b,
                norb=norb,
                nelec=(n_a, n_b),
            )

            sci_result = SCIResultWithConvergence(
                energy=float(energy),
                sci_state=sci_state,
                orbital_occupancies=occupancies,
                rdm1=rdm1,
                rdm2=rdm2,
                state="converged" if info["converged"] else "not_converged",
            )

            all_sci_results.append(sci_result)

        with self._stats_lock:
            self.stats["t_occ"] += t_occ
        energy = all_sci_results[0].energy
        eigvec = all_eigvecs[0]

        # --- ⟨S²⟩ diagnostic (optional, via Γab) --------------------------
        # Same policy as product mode: build Γαβ only (skip Γaa/Γbb) and
        # feed into ``s_squared_from_rdm``. The interleaved-Wick → spin-
        # block conversion happens inside ``make_two_rdm_ab_pair_torch``
        # by default.
        s2_value = None
        if self.compute_spin_squared:
            from torch_sci_solver.torch_impl.rdm import s_squared_from_rdm

            t0 = _time.perf_counter()
            Gab_s = make_two_rdm_ab_pair_torch(
                eigvec, pair_basis, pair_helpers, norb, chunk_size=self.chunk_size
            )
            s2_t = s_squared_from_rdm(Gab_s, n_a, n_b)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            s2_value = float(s2_t.item())
            _t_s2_diag = _time.perf_counter() - t0
            with self._stats_lock:
                self.stats["spin_squared"].append(s2_value)
                self.stats["t_spin_sq"] += _t_s2_diag
            del Gab_s
        else:
            with self._stats_lock:
                self.stats["spin_squared"].append(None)

        if self.verbose:
            _spin_sq_final = None
            if _S2_op is not None:
                with self._stats_lock:
                    _spin_sq_final = self.stats["spin_sq_final"]
            self._append_davidson_verbose(
                verbose_lines,
                t_dav=t_dav,
                info=info,
                t_occ=t_occ,
                line_energies=[res.energy for res in all_sci_results],
                energy=energy,
                rdm_note=f"  | full RDM {t_rdm:.2f}s" if self.compute_full_rdm else None,
                s2_value=s2_value,
                spin_sq_final=_spin_sq_final,
                spin_sq_target=spin_sq_target,
                spin_lambda=call_spin_lambda,
            )

        # --- build TorchPairSCIState (probe-method output) ---------------
        # Populated only on cold-solve-with-probe or on warm-extend with
        # explicit build_output_state=True. The eigvec is snapshotted on
        # CPU in interleaved-Wick (matching the cache format the next
        # call's warm-embedded expects).
        output_state = None
        if build_output_states:
            fp_int = get_integrals_fingerprint()
            state_dtype = torch.complex128 if eigvec.is_complex() else torch.float64
            output_state = TorchPairSCIState(
                alpha_of_pair=pair_basis.alpha_of_pair.cpu().tolist(),
                beta_of_pair=pair_basis.beta_of_pair.cpu().tolist(),
                eigvec_inter=eigvec.detach().cpu().to(dtype=state_dtype).clone(),
                energy=float(energy),
                n_orb=norb,
                nelec=(n_a, n_b),
                integrals_fingerprint=fp_int,
                solver_knobs=dict(solver_knob_snapshot),
                n_matvecs=info.get("n_matvecs", None),
                final_residual=info.get("final_residual", None),
            )

        perf = self._subspace_perf(
            sub_idx=sub_idx,
            m_a=pair_basis.n_unique_alphas,
            m_b=pair_basis.n_unique_betas,
            n=n_pairs,
            chosen_nb=chosen_nb,
            t_setup=t_setup,
            t_dav=t_dav,
            t_occ=t_occ,
            info=info,
            warm=warm,
            perf_dav=perf_dav,
            energy=energy,
            dev=dev,
            pair_mode=True,
        )
        if pair_helpers.blocks is not None:
            _blocks = pair_helpers.blocks
            perf["gdb_block_matvec"] = True
            perf["gdb_block_groups_beta"] = _blocks.stats["groups_beta"]
            perf["gdb_block_groups_alpha"] = _blocks.stats["groups_alpha"]
            perf["gdb_block_bytes"] = _blocks.stats["block_bytes"]
        else:
            perf["gdb_block_matvec"] = False

        if verbose_lines:
            with self._print_lock:
                for _ln in verbose_lines:
                    print(_ln)

        return EigenStates(states=all_sci_results), output_state, perf

    def _spinless_knob_snapshot(self) -> dict[str, Any]:
        """Solver knobs recorded in a :class:`TorchSpinlessSCIState`.

        The spinless operator is fixed by the Hamiltonian and the particle
        number alone --- there is no alpha/beta projection or seed-strategy
        reorder to change it --- so only the Davidson/precision knobs are
        recorded. The shared validator treats the tolerance/block/iteration
        entries as soft mismatches (a warning); ``dtype`` and ``n_roots`` are
        informational and carried for diagnostics.
        """
        knobs = self._solver_knob_values()
        return {
            "davidson_tolerance": knobs["davidson_tolerance"],
            "davidson_de_tolerance": knobs["davidson_de_tolerance"],
            "davidson_block_size": knobs["davidson_block_size"],
            "max_davidson_iters": knobs["max_davidson_iters"],
            "chunk_size": knobs["chunk_size"],
            "dtype": str(knobs["dtype"]),
            "davidson_dtype": str(knobs["davidson_dtype"]) if knobs["davidson_dtype"] else None,
            "matvec_impl": knobs["matvec_impl"],
            "n_roots": knobs["n_roots"],
        }

    def _solve_spinless(
        self,
        ci_strings,
        one_body_tensor,
        two_body_tensor,
        norb,
        n_particles: int,
        *,
        spin_sq: Optional[float] = None,
        **kwargs: Any,
    ) -> "List[EigenStates] | tuple[List[EigenStates], List[Optional[Any]]]":
        """Diagonalize H over a flat spinless (spin-orbital) basis, per subspace.

        Dispatched from ``__call__`` when ``nelec`` is a scalar total particle
        number rather than an ``(n_a, n_b)`` tuple — the flat spin-orbital
        (GHF / SOC / generalized-spinor) path with no alpha/beta
        factorization, mirroring
        ``GeneralAlgebraSolverBase._solve_spinless_batches``.

        Multi-root (``n_roots > 1``, set at construction) is supported at the
        same level as the product-mode (alpha/beta) path: ``davidson_torch``
        is fully generic in the basis dimension and already returns multiple
        roots given ``n_roots=self.n_roots`` regardless of what the
        wavefunction's index space means (product, GDB pair, or flat
        spinless) — see the ``EigenStates``/``SCIResultWithConvergence``
        unpacking below, which mirrors ``_solve_one_subspace``'s own
        ``n_roots == 1`` vs ``n_roots > 1`` result-shape branch exactly.
        ``probe_method="warm_start"`` is supported with
        :class:`~torch_sci_solver.torch_impl.probe_spinless.TorchSpinlessSCIState`.
        ``spin_sq`` and ``compute_spin_squared`` remain genuine physics
        limitations (no alpha/beta split defines ``S^2``), surfaced with a
        stated reason by ``preflight`` and ``capabilities()``.

        Args:
            ci_strings: list of flat ``strs`` arrays — one ``N``-particle
                spin-orbital determinant bitmask list per subspace (NOT
                ``(strings_a, strings_b)`` tuples).
            one_body_tensor: ``(nmo, nmo)`` complex Hermitian ``hcore``.
            two_body_tensor: ``(nmo, nmo, nmo, nmo)`` complex chemist-order
                ``eri``.
            norb: ``nmo``, the number of spin-orbitals.
            n_particles: total particle number ``N``.
            spin_sq: must be ``None`` — rejected otherwise (see Raises).
            **kwargs: ``max_davidson_iters`` / ``tol`` / ``de_tol`` /
                ``init_vectors`` per-call overrides, same names and contract as
                the product-mode ``__call__`` (an ``init_vectors`` entry holds
                one amplitude per string of that subspace), plus the
                ``probe_method`` / ``probe_kwargs`` / ``probe_state`` warm-start
                trio. A cold solve with ``probe_method="warm_start"`` returns
                ``(results, states)``; a warm extension returns ``(results,
                states)`` only when ``probe_kwargs={'build_output_state': True}``.
                Other kwargs are rejected with a targeted message.

        Returns:
            One ``EigenStates`` per entry of ``ci_strings``, in order. Each
            holds ``self.n_roots`` ``SCIResultWithConvergence`` objects,
            ascending in energy (``states[0]`` is the ground state). When a
            warm-start snapshot is requested, a parallel list of
            ``TorchSpinlessSCIState | None`` (one per subspace) is returned as
            the second element of a tuple.

        Raises:
            NotImplementedError: an unsupported kwarg is passed.
            TypeError: a ``probe_state`` from another mode is supplied.
            ValueError: ``spin_sq`` is not ``None``, or a warm extension's
                determinant basis/fingerprint/particle sector disagrees with
                the cached state.
        """
        from torch_sci_solver.torch_impl.probe_spinless import (
            _as_spinless_state_list,
            _dedup_and_check_spinless_superset,
            _embed_eigvec_spinless,
            _emit_spinless_dedup_warning,
            _validate_spinless_state,
        )

        parsed = self._parse_spinless_call_kwargs(spin_sq, kwargs, validate_davidson=False)
        probe_method = parsed["probe_method"]
        probe_state_in = parsed["probe_state"]
        init_vectors_in = parsed["init_vectors"]
        build_output_states = parsed["build_output_states"]
        probe_state_list = _as_spinless_state_list(probe_state_in, len(ci_strings))
        if (
            self._distributed_ctx is not None
            and self._distributed_checkpoint_settings() is not None
        ):
            raise NotImplementedError(
                "distributed checkpointing is not implemented for the spinless path; "
                "drop distributed={'checkpoint_dir': ...} for a scalar-nelec solve or "
                "use a TPB/GDB (alpha, beta) basis."
            )

        from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
        from torch_sci_solver.torch_impl.mult_spinless import (
            make_diag_terms_spinless_torch,
            mult_spinless_torch,
        )
        from torch_sci_solver.torch_impl.rdm_spinless import (
            spinless_rdm1_diag_torch,
            spinless_rdm12_torch,
        )
        from torch_sci_solver.torch_impl.spinless_helpers import (
            SpinlessSCIState,
            _split_to_limbs,
            basis_init_vector_spinless_torch,
            dets_to_python_ints,
            generate_spinless_helpers,
            spinless_basis_from_array,
        )

        dev = self.device
        call_max_iter = parsed["call_max_iter"]
        call_eps = parsed["call_eps"]
        call_de_tol = parsed["call_de_tol"]

        # hcore/eri are the same flat spinless integrals for every subspace in
        # this call — build the TorchIntegrals bundle once, outside the loop,
        # mirroring the product/pair-mode integrals cache's intent (avoid
        # rebuilding per subspace) without the full fingerprint-cache
        # machinery.
        integrals = to_torch_integrals_spinless(
            one_body_tensor, two_body_tensor, core_energy=0.0, device=dev, dtype=self.dtype
        )

        init_vector_list = self._init_vector_list(init_vectors_in, len(ci_strings))
        # Warm-start provenance is per call: hash the Hamiltonian once (only
        # when a probe state is requested) and record the same knob snapshot in
        # every cached state.
        h1_arr = _as_integrals_array(one_body_tensor) if probe_method is not None else None
        eri_arr = _as_integrals_array(two_body_tensor) if probe_method is not None else None
        integrals_fingerprint = (
            _integrals_fingerprint(h1_arr, eri_arr, 0.0)
            if h1_arr is not None and eri_arr is not None
            else None
        )
        solver_knobs = self._spinless_knob_snapshot()
        output_states: List[Optional[Any]] = [None] * len(ci_strings)

        results: List[EigenStates] = []
        for sub_idx, strs in enumerate(ci_strings):
            probe_state = probe_state_list[sub_idx]
            solve_strs = strs
            if probe_state is not None:
                _validate_spinless_state(
                    probe_state,
                    h1_arr,
                    eri_arr,
                    norb,
                    n_particles,
                    solver_knobs,
                    dtype=self.dtype,
                    current_fingerprint=integrals_fingerprint,
                )
                solve_strs, dedup_info = _dedup_and_check_spinless_superset(
                    probe_state.dets, strs, norb
                )
                _emit_spinless_dedup_warning(dedup_info)
                if self.verbose:
                    with self._print_lock:
                        print(
                            f"  [sci_solver:spinless:warm] core N={dedup_info['cached_N']} "
                            f"→ extended N={dedup_info['extended_N']} "
                            f"(+{dedup_info['dets_added']} determinant(s))"
                        )
            basis = spinless_basis_from_array(solve_strs, norb, n_particles, device=dev)
            helpers = generate_spinless_helpers(basis, budget_bytes=self.memory_budget_bytes)
            _ctx = self._distributed_ctx
            _auto_cap = max(30, self.n_roots + 3)
            _block_cap = (
                int(self.davidson_block_size) if self.davidson_block_size is not None else _auto_cap
            )
            # A spinless basis has no basis-defined α/β axes, but splitting each
            # determinant's bitmask at half the spin-orbitals still gives the
            # row_uid × col_uid grid the tile operator owns.
            _tile_spec = getattr(_ctx, "tile", None) if _ctx is not None else None
            if _tile_spec == "auto":
                assert _ctx is not None  # "auto" implies a distributed context
                # "auto" is a policy, not a promise to tile: the 1-D operator
                # enumerates only the owned rows and is faster at equal size, so
                # a grid is used only when the gathered input would not fit.
                from torch_sci_solver.torch_impl.spinless_tiles import (
                    resolve_auto_spinless_tiling,
                )

                if not resolve_auto_spinless_tiling(
                    world_size=_ctx.world_size,
                    n_dets=int(basis.dim),
                    block_size=_block_cap,
                    itemsize=int(self.dtype.itemsize),
                    budget_bytes=self._memory_budget(_ctx),
                ):
                    _tile_spec = None
            # Owner-computes spinless path: flat, or tiled when a tile spec is
            # configured. A basis that fits in one Davidson block keeps the
            # replicated path, so the tile spec is ignored there (same as
            # TPB/GDB). A full-RDM request runs on the owner-computes solve and
            # gathers the eigenvector once for the RDM build.
            _local_spinless = _ctx is not None and basis.dim > _block_cap
            if _local_spinless:
                eigen_states, local_info = self._solve_spinless_local_subspace(
                    sub_idx=sub_idx,
                    strings=solve_strs,
                    basis=basis,
                    helpers=helpers,
                    integrals=integrals,
                    norb=norb,
                    n_particles=n_particles,
                    init_vector=init_vector_list[sub_idx],
                    probe_state=probe_state,
                    call_max_iter=call_max_iter,
                    call_eps=call_eps,
                    call_de_tol=call_de_tol,
                    dev=dev,
                    tile_spec=_tile_spec,
                    build_output_states=build_output_states,
                )
                results.append(eigen_states)
                if build_output_states and eigen_states.states:
                    output_states[sub_idx] = _spinless_output_state(
                        eigen_states.states[0],
                        n_orb=norb,
                        n_particles=n_particles,
                        integrals_fingerprint=integrals_fingerprint or "",
                        solver_knobs=solver_knobs,
                        n_matvecs=local_info.get("n_matvecs"),
                        final_residual=local_info.get(
                            "final_residual", local_info.get("final_residual_norm")
                        ),
                        eigvec_override=local_info.get("warm_eigvec"),
                    )
                continue
            hii = make_diag_terms_spinless_torch(basis, integrals)
            # Seed Davidson at the lowest-diagonal determinant, NOT basis
            # index 0. The old init=0 assumed "index 0 = numerically
            # smallest determinant = HF", which np.unique's low-limb-first
            # lexicographic row sort only guarantees while n_limbs == 1:
            # past 64 spin-orbitals index 0 is an arbitrary determinant, and
            # Davidson happily "converged" to its isolated eigenpair and
            # returned it as the ground state. Seeding at argmin(hii) makes
            # the lowest Ritz value <= min(hii) <= E_HF (whenever HF is in
            # the basis) independent of the limb sort order, which
            # _found_in_multilimb's binary search depends on -- so the sort
            # itself must not change. The same seed is the fallback when an
            # init_vectors entry has no overlap with this basis.
            seed_index = int(torch.argmin(hii))

            def cold() -> "torch.Tensor":
                return basis_init_vector_spinless_torch(
                    basis.dim, init=0, seed_index=seed_index, device=dev, dtype=self.dtype
                )

            warm = False
            if probe_state is not None:
                # String-lookup embedding onto the extended (union) basis:
                # determinants added by the extension stay zero, and the cached
                # amplitudes are normalized globally. A cache with no surviving
                # component (for example an all-zero stored eigenvector) falls
                # back to the same cold seed instead of handing Davidson a zero
                # vector.
                W = _embed_eigvec_spinless(
                    probe_state.eigvec_inter,
                    probe_state.dets,
                    solve_strs,
                    norb,
                    device=dev,
                    dtype=self.dtype,
                )
                if float(torch.linalg.norm(W).item()) == 0.0:
                    W = cold()
                    if self.verbose:
                        with self._print_lock:
                            print(
                                f"  [sci_solver:spinless] probe_state[{sub_idx}]: no "
                                "overlap with this basis; cold start"
                            )
                else:
                    warm = True
            elif init_vector_list[sub_idx] is None:
                W = cold()
            else:
                vec = _validate_init_vector(
                    init_vector_list[sub_idx],
                    len(strs),
                    dtype=self.dtype,
                    sub_idx=sub_idx,
                    n_roots=self.n_roots,
                )
                n_limbs = int(basis.dets.shape[1])
                caller_rows = np.array(
                    [_split_to_limbs(int(x), n_limbs) for x in strs], dtype=np.int64
                ).reshape(-1, n_limbs)
                k = _first_index(caller_rows, basis.dets, device=dev)

                def _map_init(row: np.ndarray, k: torch.Tensor = k) -> torch.Tensor:
                    return _gather_or_zero(torch.as_tensor(row, dtype=self.dtype, device=dev), k)

                # No projector on the spinless path (no alpha/beta split to project).
                W, warm = _resolve_start_vectors(vec, _map_init, None, cold)
                if not warm and self.verbose:
                    with self._print_lock:
                        print(
                            f"  [sci_solver:spinless] init_vectors[{sub_idx}]: no overlap "
                            "with this basis; cold start"
                        )

            def matvec(v: "torch.Tensor") -> "torch.Tensor":
                """Matvec."""
                return mult_spinless_torch(
                    hii,
                    v,
                    basis,
                    helpers,
                    integrals,
                    chunk_size=self.chunk_size,
                    triton_matvec=self.matvec_impl == "triton",
                )

            _dav_result, _perf_dav = self._run_davidson(
                hii,
                W,
                matvec,
                call_max_iter,
                self._pick_num_block(basis.dim, 1, device=dev, n_roots=self.n_roots),
                call_eps,
                call_de_tol=call_de_tol,
                project_fn=None,
                warm=warm,
            )

            # Unpack Davidson result -- single-root returns (float, Tensor(N,),
            # dict); multi-root returns (Tensor(n_roots,), Tensor(n_roots,N),
            # dict). Mirrors _solve_one_subspace's own product-mode branch
            # exactly (davidson_torch is basis-shape-agnostic either way).
            if self.n_roots == 1:
                energy_raw, eigvecs_raw, info = _dav_result
                energies_raw = torch.tensor([energy_raw], dtype=torch.float64)
                all_eigvecs = eigvecs_raw.unsqueeze(0)  # (1, N)
            else:
                energies_raw, all_eigvecs, info = _dav_result

            n_found = all_eigvecs.shape[0]
            sci_results: List[SCIResultWithConvergence] = []
            # MPS solves in float32: report float64 CPU Rayleigh quotients instead
            # of the float32 Ritz values (see _refines_fp32_energy).
            energy_matvec64 = None
            if self._refines_fp32_energy(call_eps):
                cpu = torch.device("cpu")
                integrals64 = self._cpu_wide_integrals(
                    one_body_tensor, two_body_tensor, spinless=True
                )
                basis64 = _tensors_to(basis, cpu)
                helpers64 = generate_spinless_helpers(
                    basis64, budget_bytes=self.memory_budget_bytes
                )
                hii64 = make_diag_terms_spinless_torch(basis64, integrals64)

                def energy_matvec64(v: torch.Tensor) -> torch.Tensor:
                    """The float64 spinless matvec on the CPU copy of the basis."""
                    return mult_spinless_torch(
                        hii64, v, basis64, helpers64, integrals64, chunk_size=self.chunk_size
                    )

                _perf_dav["davidson_fp32_energy"] = "rq64"
            for root_idx in range(n_found):
                eigvec = all_eigvecs[root_idx]
                energy = float(energies_raw[root_idx].item())
                if energy_matvec64 is not None:
                    energy = _rayleigh_quotient_wide(energy_matvec64, eigvec, self._wide_dtype())

                # Occupancies are needed unconditionally, but the full
                # spinless_rdm12_torch build is not: it forms a dense
                # (nmo*nmo, nmo*nmo) Gram matrix for dm2 (O(nmo**4) memory)
                # that goes unused whenever compute_full_rdm is False (the
                # default), which OOM'd on a nmo=60 run that never asked for
                # the full RDM. spinless_rdm1_diag_torch computes just the
                # occupancy diagonal, matching the product/pair-mode
                # branches' own diagonal-only-unless-requested policy.
                occupancies = (spinless_rdm1_diag_torch(basis, eigvec).cpu().numpy(),)

                rdm1_np = None
                rdm2_np = None
                if self.compute_full_rdm:
                    dm1_t, dm2_t = spinless_rdm12_torch(basis, helpers, eigvec)
                    rdm1_np = dm1_t.cpu().numpy()
                    rdm2_np = dm2_t.cpu().numpy()

                sci_state = SpinlessSCIState(
                    amplitudes=eigvec.detach().cpu().numpy(),
                    ci_strs=dets_to_python_ints(basis.dets),
                    norb=norb,
                    nelec=(n_particles, 0),
                )
                sci_results.append(
                    SCIResultWithConvergence(
                        energy=energy,
                        sci_state=sci_state,
                        orbital_occupancies=occupancies,
                        rdm1=rdm1_np,
                        rdm2=rdm2_np,
                        state="converged" if info["converged"] else "not_converged",
                    )
                )
                if self.verbose:
                    with self._print_lock:
                        print(
                            f"  [sci_solver:spinless] N={basis.dim} | dev={dev} | "
                            f"root={root_idx} E = {energy:.10f} Ha | "
                            f"{info.get('n_matvecs', '?')} matvecs"
                        )
            results.append(EigenStates(states=sci_results))
            if build_output_states and sci_results:
                output_states[sub_idx] = _spinless_output_state(
                    sci_results[0],
                    n_orb=norb,
                    n_particles=n_particles,
                    integrals_fingerprint=integrals_fingerprint or "",
                    solver_knobs=solver_knobs,
                    n_matvecs=info.get("n_matvecs"),
                    final_residual=info.get("final_residual", info.get("final_residual_norm")),
                )
        if probe_method is not None and build_output_states:
            return results, output_states
        return results

    def _solve_spinless_local_subspace(
        self,
        *,
        sub_idx: int,
        strings,
        basis,
        helpers,
        integrals,
        norb: int,
        n_particles: int,
        init_vector,
        probe_state,
        call_max_iter: int,
        call_eps: float,
        call_de_tol,
        dev: torch.device,
        tile_spec=None,
        build_output_states: bool = False,
    ) -> "tuple[EigenStates, dict]":
        """One spinless subspace on the distributed owner-computes path.

        The flat determinant list is partitioned contiguously, or — when
        ``tile_spec`` is set — into one ``row_uid × col_uid`` rectangle per rank
        with the bitmask split at half the spin-orbitals. Each rank builds its
        diagonal block and owns its amplitude block; the operator gathers only
        the input it reads (a full vector on the flat layout, one staged ket
        stage on a tile) while enumerating only its owned rows. Occupancies are
        local sums reduced across ranks; rank 0 alone builds the result
        (non-zero ranks return an empty ``EigenStates``). A small result is
        gathered to rank 0, a large one is written to memmapped ``.npy`` files
        under ``TMPDIR`` — the same size-dependent hand-back as the TPB path,
        with the tile writing its scattered determinant positions.

        A warm-start ``probe_state`` is embedded per rank: each rank pastes
        only the amplitudes whose determinant it owns (a contiguous block or a
        scattered tile), and the surviving norm is reduced globally before the
        cold-start fallback decision, exactly as the TPB owner-computes path
        does. A per-rank block norm would be wrong.

        Args:
            probe_state: a validated ``TorchSpinlessSCIState``, or ``None``.
                Mutually exclusive with ``init_vector`` (enforced by the
                caller).
            tile_spec: ``None`` for the flat owner-computes layout, ``"force"``
                (or ``"auto"`` already resolved to tile) for the default tile
                grid, or an explicit ``(p_rows, p_cols)``.

        Returns:
            ``(EigenStates, info)``. ``info`` is the Davidson diagnostics dict
            (``n_matvecs``, residual, convergence) for the cached-state
            snapshot; it is informational on every rank.
        """
        import torch.distributed as dist

        from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE
        from torch_sci_solver.torch_impl.distributed_davidson import (
            global_smallest_hii,
            hash_uniform,
            partition_vector,
        )
        from torch_sci_solver.torch_impl.distributed_solver import (
            total_hand_back_bytes,
            write_pair_tile_memmaps,
            write_row_memmaps,
        )
        from torch_sci_solver.torch_impl.distributed_spinless import (
            LocalSpinlessOperator,
            make_diag_terms_spinless_local,
        )
        from torch_sci_solver.torch_impl.probe_spinless import _embed_eigvec_spinless
        from torch_sci_solver.torch_impl.rdm_spinless import spinless_rdm1_diag_torch
        from torch_sci_solver.torch_impl.spinless_helpers import (
            SpinlessBasis,
            SpinlessSCIState,
            _split_to_limbs,
            dets_to_python_ints,
        )

        ctx = self._distributed_ctx
        assert ctx is not None  # dispatched only when a distributed context is set
        n_dets = int(basis.dim)
        tile = None
        if tile_spec is None:
            bounds = partition_vector(n_dets, ctx.world_size, ctx.rank)
            owned_index = torch.arange(bounds.start, bounds.stop, device=dev, dtype=torch.int64)
        else:
            from torch_sci_solver.torch_impl.spinless_tiles import spinless_tile_partition

            # Splitting the bitmask at half the spin-orbitals keeps both uid
            # axes populated for the usual selected basis; a degenerate axis
            # (empty slabs) is representable and simply carries no rows.
            split = int(norb) // 2
            grid = (
                None if tile_spec in ("auto", "force") else tuple(int(entry) for entry in tile_spec)
            )
            tile, all_bounds = spinless_tile_partition(basis.dets, split, ctx.world_size, grid=grid)
            bounds = all_bounds[ctx.rank]
            owned_index = bounds.determinant_index.to(dev)
        row_start, row_stop = (bounds.start, bounds.stop) if tile is None else (0, 0)
        local_dets = basis.dets[owned_index]
        hii_local = make_diag_terms_spinless_local(
            local_dets, norb, n_particles, integrals, chunk_size=self.chunk_size
        )
        if tile is None:
            operator = LocalSpinlessOperator(
                hii_local,
                basis,
                helpers,
                integrals,
                bounds,
                rank=ctx.rank,
                world_size=ctx.world_size,
                group=ctx.group,
                chunk_size=self.chunk_size,
                triton_matvec=self.matvec_impl == "triton",
            )
        else:
            from torch_sci_solver.torch_impl.distributed_spinless_tile import (
                LocalSpinlessTileOperator,
            )

            operator = LocalSpinlessTileOperator(
                hii_local,
                basis,
                helpers,
                integrals,
                bounds,
                tile,
                rank=ctx.rank,
                world_size=ctx.world_size,
                group=ctx.group,
                chunk_size=self.chunk_size,
                triton_matvec=self.matvec_impl == "triton",
            )
        local_basis = SpinlessBasis(dets=local_dets, n_spinorb=norb, n_particles=n_particles)

        # Both layouts answer the seed helper through the same scattered view:
        # the flat block is its contiguous range, a tile its sorted subset.
        index_layout = _ScatteredIndexLayout(owned_index, n_dets, ctx.world_size, ctx.rank)

        def _cold() -> torch.Tensor:
            """Global-argmin(hii) seed plus partition-invariant hashed noise."""
            seed_index = global_smallest_hii(hii_local, index_layout, k=1, group=ctx.group)[0][0]
            w = torch.zeros(bounds.local_size, dtype=self.dtype, device=dev)
            probe = torch.tensor([seed_index], dtype=torch.int64, device=dev)
            position = int(torch.searchsorted(owned_index, probe)[0])
            if position < bounds.local_size and int(owned_index[position]) == int(seed_index):
                w[position] = 1.0
            noise = hash_uniform(owned_index, salt=0).to(dtype=self.dtype)
            noise_norm = _global_norm_local(noise, ctx)
            if float(noise_norm.item()) > 0.0:
                w = w + SEED_NOISE_SCALE * (noise / noise_norm)
            return w / _global_norm_local(w, ctx)

        warm = False
        if probe_state is not None:
            # Owned-row embedding of the cached spinless eigenvector: this rank
            # pastes only the determinants it owns (``owned_index``), and the
            # surviving norm is reduced globally below before the fallback
            # decision, so every rank takes the same branch.
            W = _embed_eigvec_spinless(
                probe_state.eigvec_inter,
                probe_state.dets,
                strings,
                norb,
                device=dev,
                dtype=self.dtype,
                indices=owned_index.detach().cpu(),
                normalize=False,
            )
            warm_norm = _global_norm_local(W, ctx)
            if float(warm_norm.item()) == 0.0:
                W = _cold()
            else:
                W = W / warm_norm
                warm = True
        elif init_vector is None:
            W = _cold()
        else:
            vec = _validate_init_vector(
                init_vector,
                len(strings),
                dtype=self.dtype,
                sub_idx=sub_idx,
                n_roots=self.n_roots,
            )
            n_limbs = int(basis.dets.shape[1])
            caller_rows = np.array(
                [_split_to_limbs(int(x), n_limbs) for x in strings],
                dtype=np.int64,
            ).reshape(-1, n_limbs)
            # Position of each internal determinant among the caller's strings.
            k = _first_index(caller_rows, basis.dets, device=dev)[owned_index]

            def _map_init(row: np.ndarray) -> torch.Tensor:
                return _gather_or_zero(torch.as_tensor(row, dtype=self.dtype, device=dev), k)

            W, warm = _resolve_start_vectors_local(vec, _map_init, None, _cold, ctx)

        if tile is None:
            # The replicated input gather holds one full N vector per rank:
            # count its P block-columns in the overhead.
            matvec_vectors = ctx.world_size + 4
        else:
            # One block column is N_local values; the tile operator stages at
            # most its receive buffer plus one send payload, so count those
            # columns instead of the gather's P.
            staged = int(operator.max_stage_size) + int(operator.max_send_size)
            matvec_vectors = 4 + -(-staged // max(1, bounds.local_size))
        chosen_nb = self._pick_num_block(
            1,
            max(1, bounds.local_size),
            device=dev,
            n_roots=self.n_roots,
            local_n=max(1, bounds.local_size),
            matvec_vectors=matvec_vectors,
            cap_to_n=False,
        )
        chosen_nb = min(max(chosen_nb, self.n_roots + 3), n_dets)
        _dav_result, _perf_dav = self._run_davidson_distributed_local(
            hii_local,
            W,
            operator,
            bounds,
            context=ctx,
            call_max_iter=call_max_iter,
            chosen_nb=chosen_nb,
            call_eps=call_eps,
            call_de_tol=call_de_tol,
            project_fn=None,
            warm=warm,
        )
        if self.n_roots == 1:
            energy_raw, eigvecs_raw, info = _dav_result
            energies_raw = torch.tensor([energy_raw], dtype=torch.float64)
            all_eigvecs = eigvecs_raw.unsqueeze(0)
        else:
            energies_raw, all_eigvecs, info = _dav_result
        n_found = all_eigvecs.shape[0]

        # Full RDMs: the output-partitioned builder accumulates the Gram rows
        # directly from the owned amplitude blocks and gathers the full tensors
        # to rank 0 only. The gather-then-serial build stays the fallback when
        # the layout cannot partition, and a small full output keeps the
        # replicated return tuple.
        rdm1_np = None
        rdm2_np = None
        rdm_accumulation = "occupancies_only"
        if self.compute_full_rdm:
            from torch_sci_solver.torch_impl.distributed_spinless_rdm import (
                PartitionedRDM,
                can_partition_spinless_rdm,
                spinless_rdm12_distributed,
            )

            nmo = int(basis.n_spinorb)
            can_partition = can_partition_spinless_rdm(
                basis, world_size=ctx.world_size, dtype=all_eigvecs.dtype
            )
            rdm_accumulation = self._rdm_accumulation_path(
                n_full=nmo**4,
                can_partition=can_partition,
                world_size=ctx.world_size,
                dtype=all_eigvecs.dtype,
            )
            if not can_partition:
                from torch_sci_solver.torch_impl.distributed_spinless_rdm import (
                    gather_spinless_amplitudes,
                )
                from torch_sci_solver.torch_impl.rdm_spinless import spinless_rdm12_torch

                full = gather_spinless_amplitudes(
                    all_eigvecs[0], owned_index, n_dets, group=ctx.group
                )
                if ctx.rank == 0:
                    dm1_t, dm2_t = spinless_rdm12_torch(
                        basis, helpers, full, chunk_size=self.chunk_size
                    )
                    rdm1_np = dm1_t.cpu().numpy()
                    rdm2_np = dm2_t.cpu().numpy()
            elif rdm_accumulation == "partitioned":
                part1, part2 = spinless_rdm12_distributed(
                    basis,
                    helpers,
                    all_eigvecs[0],
                    owned_index,
                    group=ctx.group,
                    chunk_size=self.chunk_size,
                    replicate=False,
                )
                assert isinstance(part1, PartitionedRDM) and isinstance(part2, PartitionedRDM)
                full1 = part1.gather_to_root(group=ctx.group)
                full2 = part2.gather_to_root(group=ctx.group)
                if ctx.rank == 0:
                    assert full1 is not None and full2 is not None
                    rdm1_np = full1.cpu().numpy()
                    rdm2_np = full2.cpu().numpy()
                del full1, full2
            else:
                dm1_t, dm2_t = spinless_rdm12_distributed(
                    basis,
                    helpers,
                    all_eigvecs[0],
                    owned_index,
                    group=ctx.group,
                    chunk_size=self.chunk_size,
                    replicate=True,
                    max_replicated_bytes=self._rdm_max_replicated_bytes(),
                )
                if ctx.rank == 0:
                    rdm1_np = dm1_t.cpu().numpy()
                    rdm2_np = dm2_t.cpu().numpy()
        with self._stats_lock:
            self.stats["rdm_accumulation"] = rdm_accumulation
        info["rdm_accumulation"] = rdm_accumulation

        occupancies: List[np.ndarray] = []
        for root_idx in range(n_found):
            occ = spinless_rdm1_diag_torch(local_basis, all_eigvecs[root_idx])
            if ctx.world_size > 1:
                dist.all_reduce(occ, op=dist.ReduceOp.SUM, group=ctx.group)
            if ctx.rank == 0:
                occupancies.append(occ.cpu().numpy())

        work_dtype = all_eigvecs.dtype
        itemsize = 16 if work_dtype.is_complex else 8
        hand_back_bytes = total_hand_back_bytes(
            n_found, n_dets, warm_state=build_output_states, itemsize=itemsize
        )
        amplitudes: List[Any] = []
        if not _use_memmap_hand_back(
            hand_back_bytes, self._runtime_hand_back, self._runtime_hand_back_bytes
        ):
            if tile is None:
                gathered = _gather_local_row_blocks(all_eigvecs, bounds, ctx)
            else:
                gathered = _gather_local_scattered_blocks(all_eigvecs, owned_index, n_dets, ctx)
            if ctx.rank == 0:
                amplitudes = [gathered[k].numpy() for k in range(n_found)]
        else:
            blocks = {f"amp_{k}": all_eigvecs[k] for k in range(n_found)}
            shapes: dict[str, tuple[int, ...]] = {f"amp_{k}": (n_dets,) for k in range(n_found)}
            if build_output_states:
                # The warm snapshot gets its own file so materializing the
                # result amplitudes (which unlinks their file) cannot remove
                # the snapshot's file before it is mapped.
                blocks["warm_eigvec"] = all_eigvecs[0]
                shapes["warm_eigvec"] = (n_dets,)
            if tile is None:
                markers = write_row_memmaps(
                    blocks,
                    shapes,
                    row_start=row_start,
                    row_stop=row_stop,
                    context=ctx,
                )
            else:
                # Scattered write at this rank's own determinant positions; the
                # helper is the generic 1-D index-keyed variant used by GDB.
                markers = write_pair_tile_memmaps(
                    blocks,
                    shapes,
                    pair_index=owned_index,
                    context=ctx,
                )
            if ctx.rank == 0:
                amplitudes = [markers[f"amp_{k}"] for k in range(n_found)]
                if build_output_states:
                    info["warm_eigvec"] = markers["warm_eigvec"]

        sci_results: List[SCIResultWithConvergence] = []
        if ctx.rank == 0:
            ci_strs = dets_to_python_ints(basis.dets)
            for root_idx in range(n_found):
                sci_results.append(
                    SCIResultWithConvergence(
                        energy=float(energies_raw[root_idx].item()),
                        sci_state=SpinlessSCIState(
                            amplitudes=amplitudes[root_idx],
                            ci_strs=ci_strs,
                            norb=norb,
                            nelec=(n_particles, 0),
                        ),
                        orbital_occupancies=(occupancies[root_idx],),
                        rdm1=rdm1_np if root_idx == 0 else None,
                        rdm2=rdm2_np if root_idx == 0 else None,
                        state="converged" if info["converged"] else "not_converged",
                    )
                )
                if self.verbose:
                    with self._print_lock:
                        print(
                            f"  [sci_solver:spinless:dist] N={n_dets} | dev={dev} | "
                            f"root={root_idx} E = {float(energies_raw[root_idx]):.10f} Ha | "
                            f"{info.get('n_matvecs', '?')} matvecs"
                        )
        return EigenStates(states=sci_results), info

    def format_perf_summary(self) -> str:
        """Return a multi-line string summarizing per-call performance.

        Useful at the end of an SQD run to see whether per-iteration cost
        is growing (and which component is the culprit). Call as::

            print(solver.self.format_perf_summary())

        The columns to watch when diagnosing slowdown:
          - ``N_max`` — largest subspace this call saw. If this grows
            faster than overall time, scaling is sublinear (good).
          - ``setup``, ``davidson``, ``occ`` — per-component time. Track
            which one grows fastest across calls.
          - ``gap`` — time spent OUTSIDE our solver (in qiskit's outer
            loop). Growth here is not our problem.
          - ``mem_peak`` — peak GPU memory. Sudden jumps indicate
            allocation pattern changes; steady growth indicates a leak.
        """
        log = self.stats["per_call_log"]
        if not log:
            return "(no calls logged yet)"
        lines = [
            f"{'call':>4s}  {'N_max':>12s}  "
            f"{'setup':>7s}  {'davidson':>9s}  {'occ':>5s}  "
            f"{'gap':>6s}  {'total':>7s}  {'mem_peak':>9s}  "
            f"{'matvec_sum':>10s}"
        ]
        for r in log:
            n_max = max(s["N"] for s in r["subspaces"])
            mv_sum = sum(s["n_matvecs"] or 0 for s in r["subspaces"])
            lines.append(
                f"{r['call_idx']:>4d}  {n_max:>12,d}  "
                f"{r['t_setup_call']:>7.1f}  {r['t_davidson_call']:>9.1f}  "
                f"{r['t_occ_call']:>5.1f}  "
                f"{r['t_recovery_gap']:>6.1f}  {r['t_call_total']:>7.1f}  "
                f"{r['mem_peak_call_gb']:>7.2f}GB  {mv_sum:>10d}"
            )
        return "\n".join(lines)


def make_torch_sci_solver(**kwargs) -> "TorchSCISolver":
    """Construct a :class:`TorchSCISolver`.

    Thin wrapper preserved for backwards-compatibility with code that imports
    ``make_torch_sci_solver``. New code can construct ``TorchSCISolver``
    directly — both forms give identical behavior.
    """
    return TorchSCISolver(**kwargs)
