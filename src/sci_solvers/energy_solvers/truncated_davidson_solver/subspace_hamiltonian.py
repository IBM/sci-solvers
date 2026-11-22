"""
subspace_hamiltonian.py
========================

Implements the construction of the subspace energy matrix H without ever forming a full sigma-vector.

    H_ij = b_i . H_hat b_j = sum_{I in b_i, J in b_j} c_iI c_jJ <d_I|H|d_J>

The approach mirrors the Dice SHCI implementation: for every determinant J appearing in a basis vector,
generate only the (few) determinants I connected to it by the 2-body Hamiltonian (itself, single, and
double excitations -- Slater-Condon's rules say everything else is exactly zero), and immediately
accumulate each contribution into whichever H_ij element(s) it belongs to, using a registry that records
which basis vector(s) each determinant occurs in and with what coefficient(s).

H is built incrementally: adding a new basis vector b_n only requires generating connections from the
determinantsnof b_n and looking them up against the registry of all previously seen determinants.

Determinant addressing follows `hamiltonian.Hamiltonian`'s convention:
a determinant is identified by a flat integer index
    idx = ia * nb + ib
where ia, ib are PySCF `cistring` alpha/beta string addresses.

Sign convention
----------------
Determinants: |occ_a> ⊗ |occ_b> = (prod alpha a^+_p, incr. order) (prod beta a^+_q, incr. order) |vac>
i.e. all alpha creators precede all beta creators.

Consequences:
  * Same-spin single excitation: sign = (-1)^p, p = # occupied orbitals
    of that spin strictly between the annihilated/created indices (see `excite`).
  * Mixed alpha-beta double excitation: sign = product of the two
    single-excitation signs (blocks don't interleave).
  * Same-spin double excitation: apply `excite` twice in sequence on
    that spin's list -- exact, since it's just sequential operator application.

Matches PySCF's cistring/direct_spin1 convention (verified vs. PySCF's
dense sigma vector; see test_subspace_hamiltonian.py).

Performance
-----------
The actual Slater-Condon connection generation (looping over single and
double excitations and evaluating matrix elements) is numba-JIT
compiled -- see `numba_kernels.py`. Determinant addressing (occupation
bitmask -> PySCF string address) uses an O(1) dict lookup precomputed
once at construction time, rather than repeated calls into
`cistring.str2addr`.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time

import numpy as np
from numba.core import types
from numba.typed import Dict as NumbaDict
from pyscf import ao2mo

import sci_solvers.energy_solvers.truncated_davidson_solver.numba_kernels as _nk_numba
import sci_solvers.energy_solvers.truncated_davidson_solver.sparse_vector as spv
from sci_solvers.energy_solvers.truncated_davidson_solver.hamiltonian import Hamiltonian

_compact_dot = spv.compact_dot

_INNER_DICT_TYPE = types.DictType(types.int64, types.float64)

BACKEND_NUMBA = "numba"
BACKEND_CPP = "cpp"
_DEFAULT_BACKEND = BACKEND_NUMBA


def _load_backend(backend: str):
    if backend == BACKEND_CPP:
        from sci_solvers.energy_solvers.truncated_davidson_solver import (  # noqa: PLC0415
            kernels_cpp_wrapper as _mod,
        )

        return _mod
    return _nk_numba


class SubspaceHamiltonianBuilder:
    """
    Incremental, determinant-driven builder of the (small) subspace
    energy matrix H, avoiding full sigma-vector formation.

    Usage:
        builder = SubspaceHamiltonianBuilder(hamiltonian)
        builder.add_basis_vector(b1)   # -> Hmat, Smat now 1x1
        builder.add_basis_vector(b2)   # -> Hmat, Smat now 2x2
        ...
        builder.Hmat, builder.Smat     # current subspace matrices
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian,
        warmup_jit: bool = True,
        backend: str = _DEFAULT_BACKEND,
    ):
        """
        Parameters
        ----------
        hamiltonian : Hamiltonian
            Provides the integrals / matrix-element evaluation used to
            compute entries of the subspace Hmat as basis vectors are added.
        warmup_jit : bool, optional
            If True (default), trigger any JIT-compilation (numba backend only).
        backend : str, optional
            ``"numba"`` (default) or ``"cpp"`` to select the kernel backend.
        """
        self._nk = _load_backend(backend)
        self.backend = backend
        self.H = hamiltonian
        self.norb = hamiltonian.norb
        self.nelec_a = hamiltonian.nelec_a
        self.nelec_b = hamiltonian.nelec_b
        self.h1e = np.ascontiguousarray(hamiltonian.h1e, dtype=np.float64)

        # Full (norb,norb,norb,norb) chemist-notation ERI tensor
        self.eri = np.ascontiguousarray(
            ao2mo.restore(1, hamiltonian.eri, self.norb), dtype=np.float64
        )

        # Determinant space: whatever Hamiltonian was constructed with --
        # the full FCI string space, or a selected-CI subset (`ci_strs`).
        self._bits_a = hamiltonian.strs_a
        self._bits_b = hamiltonian.strs_b

        # Precompute, once, an O(1) bitmask -> address lookup for each
        # spin. When 2^norb is small enough, use a direct NumPy array
        # indexed by the raw bitmask -- this lets the *entire* batch of
        # addresses coming back from a numba kernel be resolved with a
        # single vectorized fancy-index operation instead of a Python
        # loop of dict lookups. Falls back to a dict for larger norb,
        # where a 2^norb-sized array would be impractically large.
        # For norb <= 24: dense 2^norb table for O(1) lookups (both backends).
        # For norb > 24, numba falls back to a typed dict; C++ passes the sorted CI
        # string arrays directly and does O(log n) bisect lookups instead.
        _DIRECT_TABLE_MAX_BITS = 24

        _t0 = time.perf_counter()
        if self.norb <= _DIRECT_TABLE_MAX_BITS:
            self._use_direct_table = True
            self._addr_table_a = np.full(1 << self.norb, -1, dtype=np.int64)
            self._addr_table_a[self._bits_a] = np.arange(self._bits_a.shape[0])
            self._addr_table_b = np.full(1 << self.norb, -1, dtype=np.int64)
            self._addr_table_b[self._bits_b] = np.arange(self._bits_b.shape[0])
        elif backend == BACKEND_CPP:
            # C++ kernels use an unordered_map built into the registry capsule.
            # addr_table_{a,b} are unused for lookups but kept as placeholders.
            self._use_direct_table = False
            self._addr_table_a = self._bits_a
            self._addr_table_b = self._bits_b
        else:
            self._use_direct_table = False
            # 2^norb too large for a dense array; build a numba typed dict
            # via a compiled function — avoids a 94M-iteration Python loop.
            self._addr_table_a = _nk_numba.build_addr_dict(self._bits_a)
            self._addr_table_b = _nk_numba.build_addr_dict(self._bits_b)
        if hamiltonian.verbose:
            print(f"  addr_table build: {(time.perf_counter() - _t0) * 1e3:.1f}ms")

        # Precompute virtual-orbital complements per address on demand
        self._all_orbs = np.arange(self.norb, dtype=np.int64)

        if warmup_jit:
            self._nk.warmup(norb_hint=min(4, self.norb))

        self.basis: list[tuple[np.ndarray, np.ndarray]] = []  # compact sorted (idx, val) pairs
        if backend == BACKEND_CPP:
            from sci_solvers.energy_solvers.truncated_davidson_solver import (  # noqa: PLC0415
                kernels_cpp_wrapper as _cw,
            )

            self.registry = _cw.make_registry()
            if not self._use_direct_table:
                _cw.build_addr_maps(self.registry, self._bits_a, self._bits_b)
        else:
            self.registry = NumbaDict.empty(key_type=types.int64, value_type=_INNER_DICT_TYPE)
        self.Hmat = np.zeros((0, 0))
        self.Smat = np.zeros((0, 0))

    def add_basis_vector(self, new_vector):
        """Append new_vector to the basis and incrementally update Hmat and Smat.

        `new_vector` is a compact, sorted (idx, val) pair produced by
        spv.truncate_compact() or spv.from_flat().
        """
        n_new = len(self.basis)
        new_idx, new_val = new_vector
        self.basis.append((new_idx, new_val))

        use_dense_hdiag, hdiag, hdiag1, jdiag, kdiag = self.H.diag_inputs()

        col = np.zeros(n_new + 1, dtype=np.float64)
        extra = {"use_dense_addr": self._use_direct_table} if self.backend == BACKEND_CPP else {}
        self._nk.add_basis_vector_fused(
            new_idx,
            new_val,
            np.int64(n_new),
            self.registry,
            self.norb,
            np.int64(self.H.nb),
            self.nelec_a,
            self.nelec_b,
            self._bits_a,
            self._bits_b,
            self._addr_table_a,
            self._addr_table_b,
            **extra,
            h1e=self.h1e,
            eri=self.eri,
            use_dense_hdiag=use_dense_hdiag,
            hdiag=hdiag,
            hdiag1=hdiag1,
            jdiag=jdiag,
            kdiag=kdiag,
            col=col,
        )

        Hnew = np.zeros((n_new + 1, n_new + 1))
        Hnew[:n_new, :n_new] = self.Hmat
        Hnew[:, n_new] = col
        Hnew[n_new, :] = col
        self.Hmat = Hnew

        Snew = np.zeros((n_new + 1, n_new + 1))
        Snew[:n_new, :n_new] = self.Smat
        for i in range(n_new + 1):
            ki, vi = self.basis[i]
            s = _compact_dot(ki, vi, new_idx, new_val)  # merge join (both sorted)
            Snew[i, n_new] = s
            Snew[n_new, i] = s
        self.Smat = Snew

        return self.Hmat, self.Smat
