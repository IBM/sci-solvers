"""
hamiltonian.py
==============

High-level Hamiltonian access used by the truncated Davidson driver.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time
from typing import Tuple

import numpy as np
from numba.core import types as nbtypes
from numba.typed import Dict as NumbaDict
from pyscf.fci import cistring, direct_spin1, selected_ci

import sci_solvers.energy_solvers.truncated_davidson_solver.sparse_vector as spv
from sci_solvers.energy_solvers.truncated_davidson_solver.numba_diag import (
    HDiagCache,
    precompute_diag_ints,
)
from sci_solvers.energy_solvers.truncated_davidson_solver.sparse_vector import from_single
from sci_solvers.hamiltonian import HamiltonianData


def hf_reference_string(nelec: int) -> int:
    """Hartree-Fock reference bitstring: the lowest `nelec` orbitals occupied."""
    return int("1" * nelec, 2) if nelec > 0 else 0


def insert_reference_string(strs: np.ndarray, nelec: int) -> Tuple[np.ndarray, bool]:
    """Return `(strs_with_hf, inserted)` inserting the HF reference if absent.

    `strs` is assumed sorted ascending (as the selected-CI machinery requires);
    the returned array stays sorted. If the reference is already present, the
    input array is returned unchanged with ``inserted=False``.
    """
    strs = np.asarray(strs, dtype=np.int64)
    hf = hf_reference_string(nelec)
    pos = int(np.searchsorted(strs, hf))
    if pos < strs.shape[0] and strs[pos] == hf:
        return strs, False
    return np.insert(strs, pos, hf), True


def insert_reference_strings(
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    nelec: Tuple[int, int],
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure both alpha/beta string arrays contain the HF reference determinant.

    Convenience wrapper around `insert_reference_string` for the (alpha, beta)
    pair, keeping both arrays sorted. Emits a note per spin channel when it had
    to inject and `verbose` is True.
    """
    strs_a, inserted_a = insert_reference_string(strs_a, nelec[0])
    strs_b, inserted_b = insert_reference_string(strs_b, nelec[1])
    if verbose:
        if inserted_a:
            print("  injecting missing HF alpha reference determinant into ci_strs")
        if inserted_b:
            print("  injecting missing HF beta reference determinant into ci_strs")
    return strs_a, strs_b


class Hamiltonian:
    """
    Wraps a HamiltonianData instance and exposes:
      - dense <-> sparse CI vector conversion
      - diagonal elements of H (for the diagonal preconditioner D-hat)
      - the Hartree-Fock reference determinant as a sparse vector
      - an exact FCI solve, for benchmarking small test systems
    """

    @classmethod
    def from_arrays(
        cls,
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        ci_strs: Tuple[np.ndarray, np.ndarray] | None = None,
        precomputed_diag: bool = False,
        verbose: bool = False,
        e_nuc: float = 0.0,
        add_reference_determinant: bool = True,
    ) -> "Hamiltonian":
        """Construct from raw integral arrays (matches the stub's input signature).

        ci_strs : optional (strs_a, strs_b) int64 arrays of alpha/beta
            determinant bitstrings, restricting the CI space to a
            selected-CI subset instead of the full FCI space. Every
            string must have popcount matching the corresponding entry
            of `nelec`. The Hartree-Fock reference string is inserted
            automatically if absent (see `add_reference_determinant`).
        e_nuc : float, optional
            Nuclear repulsion energy (ECORE from FCIDUMP). Used only for
            reporting total energy; does not affect the eigensolver.
        add_reference_determinant : bool, optional
            If True (default), inject the Hartree-Fock reference string
            into a selected `ci_strs` space when missing. If False,
            leave `ci_strs` untouched (`hf_vector` then raises if the
            reference is absent). Ignored for the full FCI space.
        """
        data = HamiltonianData(
            hcore=one_body_tensor,
            eri=two_body_tensor,
            e_nuc=float(e_nuc),
            num_orbitals=norb,
            num_elec_a=nelec[0],
            num_elec_b=nelec[1],
        )
        return cls(
            data,
            precomputed_diag=precomputed_diag,
            ci_strs=ci_strs,
            verbose=verbose,
            add_reference_determinant=add_reference_determinant,
        )

    def __init__(
        self,
        data: HamiltonianData,
        precomputed_diag: bool = False,
        ci_strs: Tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
        add_reference_determinant: bool = True,
    ):
        """
        Parameters
        ----------
        data:
            Hamiltonian integrals and system metadata.
        precomputed_diag:
            If True, materialise the full O(na*nb) diagonal up front (fast
            lookup, high memory). If False, compute H_ii on the fly from
            compact O(norb^2) arrays (low memory, small per-element cost).
        ci_strs:
            Optional (strs_a, strs_b) int64 bitstring arrays restricting the
            CI space to a selected-CI subset. The HF reference is inserted
            automatically when missing (see `add_reference_determinant`).
            Pass None for the full FCI space. Each array must be strictly
            ascending (sorted, unique): the HF lookup (``np.searchsorted``) and
            PySCF's ``selected_ci`` assume it and are silently wrong otherwise.
        verbose:
            Print timing information when True.
        add_reference_determinant:
            If True (default), inject the Hartree-Fock reference string into a
            selected `ci_strs` space when absent, keeping the string arrays
            sorted. If False, use `ci_strs` verbatim (`hf_vector` then raises
            when the reference is missing). No effect on the full FCI space.
        """
        self.verbose = verbose
        self.data = data
        self.norb = data.num_orbitals
        self.nelec = (data.num_elec_a, data.num_elec_b)
        self.nelec_a = data.num_elec_a
        self.nelec_b = data.num_elec_b
        self.h1e = data.hcore
        self.eri = data.eri

        # ---- determinant space: full FCI (ci_strs=None) or selected CI ----
        # ci_strs=None: generate canonical sorted string arrays via make_strings.
        # ci_strs=(strs_a, strs_b): use caller-supplied strings as-is (assumed sorted+unique).
        self.selected = ci_strs is not None
        _t0 = time.perf_counter()
        if ci_strs is not None:
            strs_a, strs_b = ci_strs
            self.strs_a = np.asarray(strs_a, dtype=np.int64)
            self.strs_b = np.asarray(strs_b, dtype=np.int64)
            if add_reference_determinant:
                self.strs_a, self.strs_b = insert_reference_strings(
                    self.strs_a, self.strs_b, self.nelec, verbose=self.verbose
                )
        else:
            self.strs_a = np.asarray(
                cistring.make_strings(range(self.norb), data.num_elec_a), dtype=np.int64
            )
            self.strs_b = np.asarray(
                cistring.make_strings(range(self.norb), data.num_elec_b), dtype=np.int64
            )
        self.na = self.strs_a.shape[0]
        self.nb = self.strs_b.shape[0]
        if self.verbose:
            print(
                f"  ci_strings build:  {(time.perf_counter() - _t0) * 1e3:.1f}ms  (na={self.na}, nb={self.nb})"
            )

        if precomputed_diag:
            # Dense O(na*nb) diagonal, O(1) indexed directly by the
            # kernels. For a selected ci space, delegate to PySCF's own
            # selected_ci.make_hdiag.
            _t0 = time.perf_counter()
            if self.selected:
                # TODO: adapt this for spin0
                self._hdiag = selected_ci.make_hdiag(
                    data.hcore,
                    data.eri,
                    (self.strs_a, self.strs_b),
                    self.norb,
                    self.nelec,
                    compress=True,
                )
            else:
                # TODO: adapt this for spin0
                self._hdiag = direct_spin1.make_hdiag(data.hcore, data.eri, self.norb, self.nelec)
            if self.verbose:
                print(f"make_hdiag time: {time.perf_counter() - _t0:.3f} s")
            self._hdiag_cache = None
            self._diag_ints = None
        else:
            # On-the-fly path: H_ii for any target is computed inline
            # from its occupation bits via hii_from_bits, using only the
            # small O(norb)/O(norb^2) hdiag1/jdiag/kdiag arrays -- the
            # full O(na*nb) diagonal is never materialized. Computed
            # once here (O(norb^4) einsum) and reused by both the
            # kernels (via diag_inputs()) and HDiagCache (single-index
            # `diag(idx)` convenience queries). strings_a/strings_b are
            # passed explicitly so HDiagCache indexes into the SAME
            # (possibly selected) string arrays as everything else,
            # rather than silently regenerating the full FCI space.
            self._hdiag = None
            self._diag_ints = precompute_diag_ints(data.hcore, data.eri)
            self._hdiag_cache = HDiagCache(
                data.hcore,
                data.eri,
                self.norb,
                self.nelec,
                diag_ints=self._diag_ints,
                strings_a=self.strs_a,
                strings_b=self.strs_b,
            )

    def diag_inputs(self):
        """
        Returns (use_dense_hdiag, hdiag, hdiag1, jdiag, kdiag) for the
        numba kernels (add_basis_vector_fused / calc_trunc_pcr_fused /
        pcr_doubles_*_kernel):

          precomputed_diag=True  -> use_dense_hdiag=True; hdiag is the
              real O(na*nb) array, indexed directly (O(1)); hdiag1/
              jdiag/kdiag are empty placeholders, never touched.
          precomputed_diag=False -> use_dense_hdiag=False; hdiag is an
              empty placeholder, never touched; hdiag1/jdiag/kdiag
              drive the inline O(nelec^2) hii_from_bits computation.
        """
        if self._hdiag is not None:
            empty1 = np.zeros(0, dtype=np.float64)
            empty2 = np.zeros((0, 0), dtype=np.float64)
            return True, self._hdiag, empty1, empty2, empty2
        else:
            assert self._diag_ints is not None  # set whenever _hdiag is None
            hdiag1, jdiag, kdiag = self._diag_ints
            empty_dense = np.zeros(0, dtype=np.float64)
            return False, empty_dense, hdiag1, jdiag, kdiag

    def flat_index(self, ia: int, ib: int) -> int:
        """Row-major flat index into the (na, nb) CI coefficient matrix."""
        return ia * self.nb + ib

    def unflatten(self, idx: int):
        """Inverse of flat_index: returns (ia, ib) from a flat determinant index."""
        return divmod(idx, self.nb)

    def to_dense(self, v: NumbaDict) -> np.ndarray:
        """Convert a sparse CI vector to a dense (na, nb) array."""
        dense = np.zeros((self.na, self.nb))
        for idx, val in v.items():
            ia, ib = self.unflatten(idx)
            dense[ia, ib] = val
        return dense

    def from_dense(self, dense: np.ndarray, atol: float = 0.0) -> NumbaDict:
        """Convert a dense (na, nb) CI array to a sparse vector, dropping entries with |c| <= atol."""
        flat = dense.reshape(-1)
        nz = np.nonzero(np.abs(flat) > atol)[0]
        d = NumbaDict.empty(key_type=nbtypes.int64, value_type=nbtypes.float64)
        for i in nz:
            d[int(i)] = float(flat[i])
        return d

    def hf_vector(self):
        """Return the Hartree-Fock reference as a compact (idx, val) sparse vector.

        Uses the lowest `nelec_a` alpha and `nelec_b` beta orbitals. Looked up by
        string value (not assumed to be address 0 -- not guaranteed for selected-CI
        subsets). Raises if the reference isn't in the space.
        """
        hf_a = int("1" * self.nelec[0], 2)
        hf_b = int("1" * self.nelec[1], 2)
        ia = int(np.searchsorted(self.strs_a, hf_a))
        ib = int(np.searchsorted(self.strs_b, hf_b))
        if ia >= self.na or self.strs_a[ia] != hf_a:
            raise ValueError(
                "Hartree-Fock alpha string not present in ci_strs -- "
                "selected-CI spaces must include the reference determinant."
            )
        if ib >= self.nb or self.strs_b[ib] != hf_b:
            raise ValueError(
                "Hartree-Fock beta string not present in ci_strs -- "
                "selected-CI spaces must include the reference determinant."
            )
        return spv.from_flat(*from_single(self.flat_index(ia, ib), 1.0))

    def diag(self, idx: int) -> float:
        """D-hat diagonal element for determinant `idx`."""
        if self._hdiag_cache is not None:
            return self._hdiag_cache(idx)
        return float(self._hdiag[idx])
