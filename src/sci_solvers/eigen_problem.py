"""Projected-Hamiltonian problem and eigenpair container for a CI subspace.

These two objects are backend-agnostic and useful beyond the energy solvers:

* :class:`EigenProblem` owns the Hamiltonian representation for one sampled
  ``alpha x beta`` CI subspace. It exposes a matrix-free ``matvec`` (via
  ``torch_sci_solver``'s batched kernel), a lazily assembled sparse matrix, the
  diagonal (for preconditioning), and a scipy ``LinearOperator`` view. Anything
  that needs to diagonalize a projected CI Hamiltonian can consume it without
  pulling in the SQD pipeline or a specific eigensolver.
* :class:`EigenResult` is the plain ``(energies, vectors)`` container adapters
  return, with an energy-ascending :meth:`~EigenResult.sorted` helper.
* :class:`SymmetrizedEigenProblem` is a same-interface *view* over an
  ``EigenProblem`` for closed-shell subspaces: it block-diagonalizes ``H`` by the
  alpha<->beta swap symmetry and exposes one ``+/-1`` parity sector on a packed,
  roughly half-size basis, so an adapter solves the smaller block unchanged.

The matrix-free ``matvec`` reuses ``torch_sci_solver``'s batched Hamiltonian
kernel (already a hard dependency). PySCF's ``contract_2e`` builds the same
operator but is orders of magnitude slower at large active spaces (seconds per
matvec at ``norb`` ~ 36 vs sub-millisecond for the torch kernel), which would
dominate every ARPACK/LOBPCG/PRIMME iteration.

The torch kernel stores amplitudes in *interleaved-Wick* parity, which differs
from PySCF's spin-block convention by a per-``(ia, ib)`` sign ``sigma``. We wrap
that sign into ``matvec`` (``H_block = sigma . H_inter . sigma``) so the operator,
its eigenvectors, and the amplitudes handed back are all in the spin-block
convention the rest of the ecosystem (PySCF, qiskit-addon-sqd) expects.

Conventions
-----------
* ``eri`` is the full 4-index array in chemists' notation ``(pq|rs)``, matching
  :class:`~sci_solvers.hamiltonian.HamiltonianData`.
* ``nelec = (n_alpha, n_beta)``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import torch


def _is_complex_dtype(dtype) -> bool:
    """Whether ``dtype`` is complex, for either a numpy or a torch dtype.

    ``EigenProblem.dtype`` is a ``torch.dtype`` (float64) while a complex
    :class:`HermitianOperator` carries a numpy complex dtype; this normalizes both
    so a single ``is_complex`` check works regardless of the operator's origin.
    ``None`` (no dtype attribute) is treated as real.
    """
    if dtype is None:
        return False
    if isinstance(dtype, torch.dtype):
        return dtype.is_complex
    return np.issubdtype(np.dtype(dtype), np.complexfloating)


def resolve_matvec_impl(matvec_impl: str, device, dtype) -> str:
    """Validate ``matvec_impl`` and resolve ``"auto"`` to a concrete backend.

    ``"batched"`` is the portable Torch kernel; ``"triton"`` is the fused
    output-stationary kernel (real dtype, CUDA, Triton installed). ``"auto"``
    picks ``"triton"`` on CUDA when Triton is importable and a real dtype is
    used, else ``"batched"`` -- mirroring ``TorchSCISolver``'s resolution, except
    that a CUDA device without Triton falls back silently instead of raising.
    An explicit ``"triton"`` that cannot be honoured is a hard error rather than
    a silent downgrade, so a mis-set benchmark config fails loudly.
    """
    from torch_sci_solver.torch_impl.matvec_dispatch import MATVEC_IMPLS

    if matvec_impl not in ("auto",) + tuple(MATVEC_IMPLS):
        raise ValueError(
            f"matvec_impl must be one of {('auto',) + tuple(MATVEC_IMPLS)}, got {matvec_impl!r}"
        )
    dev = torch.device(device) if isinstance(device, str) else device
    if matvec_impl == "auto":
        if dtype.is_complex or dev.type != "cuda":
            return "batched"
        try:
            from torch_sci_solver.torch_impl.mult_triton import _HAVE_TRITON
        except Exception:
            _HAVE_TRITON = False
        return "triton" if _HAVE_TRITON else "batched"
    if matvec_impl == "triton":
        if dtype.is_complex:
            raise NotImplementedError(
                f"matvec_impl='triton' does not support complex dtype ({dtype}); "
                "the Triton kernel is real-only. Use 'auto' or 'batched'."
            )
        if dev.type != "cuda":
            raise ValueError(f"matvec_impl='triton' requires a CUDA device; got device={dev}.")
    return matvec_impl


# ============================================================================
# Problem object: owns the Hamiltonian representation for one CI subspace
# ============================================================================
class EigenProblem:
    """Projected Hamiltonian on ``span{|a>|b>}``, flat index ``ia * Nb + ib``.

    The subspace is the Cartesian product of the alpha and beta CI strings.
    ``matvec`` and ``hdiag`` are backed by ``torch_sci_solver``'s batched kernel
    (fast at large ``norb``); ``sigma`` bridges its interleaved-Wick parity to
    the spin-block convention so the operator and eigenvectors match PySCF. All
    representations are cached on first use.
    """

    def __init__(
        self,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        *,
        device: "str | torch.device" = "cpu",
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 2_000_000,
        matvec_impl: str = "batched",
    ):
        """Build a projected-Hamiltonian problem for one subspace.

        Args:
            strs_a: Alpha CI strings (int64 bitmask array).
            strs_b: Beta CI strings (int64 bitmask array).
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron integrals, full 4-index chemists' notation.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            device: Torch device for the matvec kernel (``"cpu"`` or ``"cuda"``).
            dtype: Torch float dtype (float64 recommended for stability).
            chunk_size: Matvec batch size for the torch kernel.
            matvec_impl: ``torch_sci_solver`` matvec backend — ``"batched"``
                (default, portable Torch kernel), ``"triton"`` (fused Triton
                kernel; real dtype, CUDA only) or ``"auto"`` (``"triton"`` on
                CUDA when available, else ``"batched"``). Validated and resolved
                here by :func:`resolve_matvec_impl`.
        """
        self.strs_a = np.ascontiguousarray(strs_a, dtype=np.int64)
        self.strs_b = np.ascontiguousarray(strs_b, dtype=np.int64)
        self.hcore = np.asarray(hcore, dtype=np.float64)
        self.eri = np.asarray(eri, dtype=np.float64)
        self.norb = int(norb)
        self.nelec = (int(nelec[0]), int(nelec[1]))
        self.Na, self.Nb = len(self.strs_a), len(self.strs_b)
        self.dim = self.Na * self.Nb
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.matvec_impl = resolve_matvec_impl(str(matvec_impl), self.device, dtype)
        self._H: sps.csr_matrix | None = None
        self._ctx_cache: tuple[Callable, Callable, torch.Tensor, torch.Tensor] | None = None

    # ---- matrix-free matvec (torch_sci_solver batched kernel) ------------
    def _ctx(self) -> tuple[Callable, Callable, torch.Tensor, torch.Tensor]:
        """Cache the torch matvec/matmat closures, diagonal, and spin-block signs.

        Returns ``(matvec_inter, matmat_inter, hii, sigma)`` where
        ``matvec_inter`` applies ``H`` to a single ``(dim,)`` vector in
        interleaved-Wick parity, ``matmat_inter`` is its ``torch.vmap`` over the
        column axis of a ``(dim, k)`` block (one batched apply instead of ``k``
        sequential ones), ``hii`` is the Hamiltonian diagonal, and ``sigma`` is
        the per-determinant ``+/-1`` interleaved-Wick to spin-block sign vector
        (flattened, length ``dim``).
        """
        if self._ctx_cache is None:
            from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
            from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
            from torch_sci_solver.torch_impl.integrals import to_torch_integrals
            from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
            from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
            from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs
            from torch_sci_solver.torch_impl.sbdiag import _to_int64_python

            adets = _to_int64_python(self.strs_a.tolist())
            bdets = _to_int64_python(self.strs_b.tolist())
            adets_t = torch.tensor(adets, dtype=torch.int64, device=self.device)
            bdets_t = torch.tensor(bdets, dtype=torch.int64, device=self.device)
            i0, i1, i2 = setup_integrals_from_pyscf_h1_eri(self.hcore, self.eri, 0.0)
            integrals = to_torch_integrals(i0, i1, i2, device=self.device, dtype=self.dtype)
            helpers = list(generate_all_torch_helpers(adets, bdets, self.norb, device=self.device))
            hii = make_diag_terms_torch(
                adets_t, bdets_t, self.norb, integrals, chunk_size=self.chunk_size
            )
            matvec_inter = build_matvec(
                self.matvec_impl,
                hii,
                adets_t,
                bdets_t,
                self.norb,
                integrals,
                helpers,
                chunk_size=self.chunk_size,
            )
            if self.matvec_impl == "triton":
                # Triton kernels launch on raw flat pointers for one vector and
                # are opaque to ``torch.vmap``, so map the columns explicitly.
                def matmat_inter(block):
                    """Apply the single-vector Triton matvec column by column."""
                    return torch.stack(
                        [matvec_inter(block[:, j].contiguous()) for j in range(block.shape[1])],
                        dim=1,
                    )

            else:
                # Batch the single-vector kernel over the column axis; ~k× faster
                # than looping for block solvers (LOBPCG/PRIMME).
                matmat_inter = torch.vmap(matvec_inter, in_dims=1, out_dims=1)
            sigma = (
                _inter_to_block_signs(adets_t, bdets_t, self.norb).to(dtype=self.dtype).reshape(-1)
            )
            self._ctx_cache = (matvec_inter, matmat_inter, hii, sigma)
        return self._ctx_cache

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``H`` (spin-block convention) to ``x`` of shape ``(dim,)`` or ``(dim, k)``.

        A 2-D ``x`` is applied as one batched (``vmap``) call rather than column
        by column; the ``sigma`` similarity is applied on both sides to return
        the operator in spin-block convention.

        This is the *numpy* entry point: it accepts and returns ``np.ndarray``,
        moving data to/from ``self.device`` each call. SciPy solvers
        (ARPACK/LOBPCG/PRIMME) need it. Torch-native drivers (the ``torch_sci``
        adapter) should call :meth:`matvec_torch` instead to stay on-device — on
        CUDA the numpy round-trip is a hard host sync on the hot path.
        """
        y = self.matvec_torch(torch.as_tensor(x, dtype=self.dtype, device=self.device))
        return y.cpu().numpy()

    def matvec_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``H`` (spin-block convention) fully on-device: Tensor -> Tensor.

        Same operator as :meth:`matvec` but with no numpy conversion — the input
        stays on ``self.device`` and the result is returned as a torch Tensor on
        the same device. This is what a torch driver (e.g. the ``torch_sci``
        Davidson adapter) should call so the whole iteration avoids the per-apply
        device<->host sync that :meth:`matvec` incurs. Accepts ``(dim,)`` (single
        vector, applied via the scalar kernel) or ``(dim, k)`` (block, applied via
        one batched ``vmap`` call); the ``sigma`` similarity is folded in on both
        sides to return the operator in the spin-block convention.
        """
        matvec_inter, matmat_inter, _, sigma = self._ctx()
        x = x.to(dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            return matvec_inter(x * sigma) * sigma
        block = x * sigma[:, None]
        return matmat_inter(block) * sigma[:, None]

    def linear_operator(self) -> spla.LinearOperator:
        """Return a symmetric scipy ``LinearOperator`` view of ``H``.

        ``matmat`` maps :meth:`matvec` over a block's columns in one batched
        ``torch.vmap`` call, so block solvers (LOBPCG/PRIMME) get a single
        batched apply per macro-iteration instead of ``k`` sequential ones.
        """
        return spla.LinearOperator(
            (self.dim, self.dim),
            matvec=self.matvec,
            matmat=self.matvec,
            rmatvec=self.matvec,  # H is symmetric
            dtype=np.float64,
        )

    # ---- diagonal --------------------------------------------------------
    def hdiag(self) -> np.ndarray:
        """Return the Hamiltonian diagonal (used for preconditioning).

        The diagonal is invariant under the ``sigma`` similarity (``sigma**2 = 1``),
        so the torch kernel's diagonal is already the spin-block diagonal.
        """
        return self.hdiag_torch().cpu().numpy().ravel()

    def hdiag_torch(self) -> torch.Tensor:
        """Return the Hamiltonian diagonal as a ``(dim,)`` Tensor on ``self.device``.

        The torch-native counterpart of :meth:`hdiag`, so a torch driver can seed
        its preconditioner without a device->host copy. As with :meth:`hdiag`, the
        diagonal is invariant under the ``sigma`` similarity (``sigma**2 = 1``), so
        the kernel diagonal is already the spin-block diagonal.
        """
        _, _, hii, _ = self._ctx()
        return hii.reshape(-1)

    # ---- sparse matrix ---------------------------------------------------
    def sparse(self) -> sps.csr_matrix:
        """Materialize ``P H P`` as a scipy CSR matrix (cached).

        Assembled by applying :meth:`matvec` to the identity, so it builds a
        dense ``(dim, dim)`` intermediate before compressing to CSR — ``O(dim^2)``
        memory (``~800 MB`` at ``dim = 1e4``). Only use for adapters that need an
        explicit matrix and for modest ``dim``; prefer the matrix-free
        :meth:`matvec` / :meth:`linear_operator` path for large subspaces (the
        default ARPACK/SLEPc/PRIMME configurations already do).
        """
        if self._H is None:
            cols = self.matvec(np.eye(self.dim, dtype=np.float64))
            self._H = sps.csr_matrix(cols)
        return self._H


@dataclass
class EigenResult:
    """Eigenpairs returned by an adapter.

    Attributes:
        energies: Shape ``(nroots,)``, ideally ascending.
        vectors: Shape ``(dim, nroots)``, columns aligned with ``energies``.
        converged: Optional per-root convergence flags, shape ``(nroots,)``,
            aligned with ``energies``/``vectors``. ``None`` (the default) means
            "unknown" -- e.g. a backend that raises on failure to converge
            rather than returning a partial result (arpack, primme, slepc)
            has no need to set this; a backend that can silently return a
            non-converged result (lobpcg, pyscf_davidson, evsl) should set it
            from whatever convergence signal it actually has, so
            :class:`~sci_solvers.energy_solvers.general_algebra_base.GeneralAlgebraSolverBase`
            can build a
            :class:`~sci_solvers.eigen_states.SCIResultWithConvergence`
            instead of a plain ``SCIResult``.
    """

    energies: np.ndarray
    vectors: np.ndarray
    converged: np.ndarray | None = None

    def sorted(self) -> "EigenResult":
        """Return a copy with roots sorted by ascending energy.

        Sorting is on the *real* part of the energies. For a real symmetric
        operator the energies are already real and this is a plain ascending
        sort (behavior unchanged). For a complex Hermitian operator the
        eigenvalues are real in exact arithmetic but a backend may return them
        with a negligible imaginary part; sorting on ``.real`` keeps the ordering
        well-defined without discarding the (tiny) imaginary residual.
        ``converged`` (if set) is reordered along with the other two.
        """
        order = np.argsort(np.real(self.energies))
        converged = self.converged[order] if self.converged is not None else None
        return EigenResult(self.energies[order], self.vectors[:, order], converged)


# ============================================================================
# Spin-symmetrized view: block-diagonalize H by the alpha<->beta swap symmetry
# ============================================================================
class SymmetrizedEigenProblem:
    """Packed-sector view of an :class:`EigenProblem` for a closed-shell subspace.

    When the alpha and beta CI-string pools are identical (``strs_a == strs_b``,
    ``Na == Nb == D`` — the closed-shell ``open_shell=False`` case), the CI
    coefficient matrix ``C`` (shape ``(D, D)``) commutes with the alpha<->beta
    swap ``P: C -> C.T``. Since ``[H, P] = 0``, ``H`` is block-diagonal in the
    ``+/-1`` eigenspaces of ``P``:

    * ``sector=+1`` (``C = +C.T``): even total spin ``S = 0, 2, 4, ...``, dimension
      ``D(D+1)/2``.
    * ``sector=-1`` (``C = -C.T``): odd total spin ``S = 1, 3, 5, ...``, dimension
      ``D(D-1)/2``.

    This view exposes the same operator interface an adapter consumes
    (:attr:`dim`, :meth:`matvec`, :meth:`linear_operator`, :meth:`hdiag`,
    :meth:`sparse`) but on the *packed* sector, so an adapter diagonalizes the
    smaller block without knowing symmetrization happened. The physical
    attributes (``strs_a``, ``strs_b``, ``norb``, ``nelec``, ``hcore``, ``eri``,
    ``Na``, ``Nb``) are delegated to the wrapped full-space problem so downstream
    RDM building is unchanged; :meth:`unpack_eigenvectors` maps packed
    eigenvectors back to the full ``(D*D,)`` product-space layout first.

    The packed basis is *orthonormal*: off-diagonal packed components carry a
    ``1/sqrt(2)`` weight so the embedding ``S`` (packed -> full) is an isometry
    and ``S^T H S`` stays symmetric. Equivalently, for a symmetric full operator
    ``H``, :meth:`matvec` computes ``S^T H S`` in the packed basis.
    """

    def __init__(self, base: EigenProblem, sector: int = +1):
        """Wrap ``base`` as a single-parity-sector view.

        Args:
            base: The full-space problem. Its alpha/beta pools must be identical
                (closed shell), else a :class:`ValueError` is raised.
            sector: ``+1`` for the ``C = +C.T`` (even-spin) block or ``-1`` for
                the ``C = -C.T`` (odd-spin) block.

        Raises:
            ValueError: If ``base``'s alpha/beta CI-string pools differ, or
                ``sector`` is not ``+1``/``-1``.
        """
        if sector not in (+1, -1):
            raise ValueError(f"sector must be +1 or -1, got {sector!r}.")
        if base.Na != base.Nb or not np.array_equal(base.strs_a, base.strs_b):
            raise ValueError(
                "symmetrize_spin requires identical alpha/beta CI-string pools "
                "(closed-shell, open_shell=False); got "
                f"Na={base.Na}, Nb={base.Nb}."
            )
        self.base = base
        self.sector = int(sector)
        self.D = base.Na
        # Upper triangle including (k=0) or excluding (k=1) the diagonal.
        self._iu, self._ju = np.triu_indices(self.D, k=(1 if sector == -1 else 0))
        diag = self._iu == self._ju
        # Orthonormal packing: off-diagonals weighted 1/sqrt(2) so S is an isometry.
        self._w = np.where(diag, 1.0, 1.0 / np.sqrt(2.0))
        self.dim = int(self._iu.size)  # D(D+1)/2 (sector +1) or D(D-1)/2 (sector -1)
        # Numpy result dtype for the packed intermediates. Detected complex-aware
        # from the base operator's dtype: an EigenProblem stores a torch.dtype
        # (float64) and a complex HermitianOperator stores a numpy complex dtype,
        # so ``_is_complex_dtype`` normalizes both to a numpy dtype before the
        # check (the diagonal is real even for a complex Hermitian operator, so it
        # cannot be used as the signal). The real packing matrix S keeps S^T H S
        # Hermitian, so a complex base yields a complex-Hermitian packed block; a
        # real base stays float64 (byte-for-byte unchanged).
        self._dtype = (
            np.complex128 if _is_complex_dtype(getattr(base, "dtype", None)) else np.float64
        )
        self._H: sps.csr_matrix | None = None
        # Lazily materialized device tensors for the torch expand/compress path
        # (built on first matvec_torch call, then reused): (iu, ju, w).
        self._torch_pack: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def dtype(self) -> np.dtype:
        """Numpy operator dtype of the packed block (complex iff the base is).

        Exposed so the adapters' complex-detection (``_is_complex_problem``,
        which reads ``problem.dtype``) engages the complex-Hermitian code paths
        for a packed complex sector — the diagonal alone is real and cannot serve
        as the signal.
        """
        return np.dtype(self._dtype)

    # ---- physical attributes delegated to the full-space problem ---------
    @property
    def strs_a(self) -> np.ndarray:
        """Alpha CI strings of the wrapped full-space problem."""
        return self.base.strs_a

    @property
    def strs_b(self) -> np.ndarray:
        """Beta CI strings of the wrapped full-space problem."""
        return self.base.strs_b

    @property
    def norb(self) -> int:
        """Number of spatial orbitals."""
        return self.base.norb

    @property
    def nelec(self) -> tuple[int, int]:
        """``(n_alpha, n_beta)``."""
        return self.base.nelec

    @property
    def hcore(self) -> np.ndarray:
        """One-electron integrals of the wrapped problem."""
        return self.base.hcore

    @property
    def eri(self) -> np.ndarray:
        """Two-electron integrals of the wrapped problem."""
        return self.base.eri

    @property
    def Na(self) -> int:
        """Number of alpha strings (== ``D``)."""
        return self.base.Na

    @property
    def Nb(self) -> int:
        """Number of beta strings (== ``D``)."""
        return self.base.Nb

    # ---- packed <-> full embedding ---------------------------------------
    def _expand(self, x: np.ndarray) -> np.ndarray:
        """Embed a packed vector ``(dim,)`` into the full ``(D*D,)`` layout."""
        C = np.zeros((self.D, self.D), dtype=self._dtype)
        v = x * self._w
        C[self._iu, self._ju] = v
        C[self._ju, self._iu] = self.sector * v
        return C.ravel()

    def _compress(self, y: np.ndarray) -> np.ndarray:
        """Adjoint of :meth:`_expand`: full ``(D*D,)`` -> packed ``(dim,)``."""
        Y = y.reshape(self.D, self.D)
        return 0.5 * (Y[self._iu, self._ju] + self.sector * Y[self._ju, self._iu]) / self._w

    # ---- operator interface (packed sector) ------------------------------
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``S^T H S`` to a packed ``(dim,)`` vector or ``(dim, k)`` block.

        A 2-D ``x`` expands all columns into one full ``(D*D, k)`` block and
        applies the base matvec once (it batches columns internally), then
        compresses each column back — preserving the batched-kernel speedup for
        block solvers (LOBPCG/PRIMME).

        .. note::

            The benefit here is reduced **memory / Krylov dimension** (packed
            vectors of length ``D(D+1)/2`` instead of ``D**2``, and ~4x smaller
            dense operators via :meth:`sparse`), **not** per-apply runtime: the
            operator is applied by expanding to the full ``(D*D,)`` space, calling
            ``base.matvec``, and compressing back, so each apply still does the
            full-space FLOPs. A genuine per-apply speedup needs a spin0-aware
            contraction in ``torch_sci_solver`` — see the ``TODO`` on
            ``torch_sci_solver.torch_impl.mult.mult_torch``.
        """
        if x.ndim == 1:
            return self._compress(self.base.matvec(self._expand(x)))
        full = np.column_stack([self._expand(x[:, j]) for j in range(x.shape[1])])
        y = self.base.matvec(full)
        return np.column_stack([self._compress(y[:, j]) for j in range(y.shape[1])])

    # ---- torch-native packed operator (no numpy round-trip) --------------
    def _torch_pack_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cache ``(iu, ju, w)`` as tensors on the base problem's device."""
        if self._torch_pack is None:
            dev = self.base.device
            dt = self.base.dtype
            iu = torch.as_tensor(self._iu, dtype=torch.int64, device=dev)
            ju = torch.as_tensor(self._ju, dtype=torch.int64, device=dev)
            w = torch.as_tensor(self._w, dtype=dt, device=dev)
            self._torch_pack = (iu, ju, w)
        return self._torch_pack

    def _expand_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Torch analog of :meth:`_expand`: packed ``(dim,)`` -> full ``(D*D,)``."""
        iu, ju, w = self._torch_pack_tensors()
        C = torch.zeros((self.D, self.D), dtype=self.base.dtype, device=self.base.device)
        v = x * w
        C[iu, ju] = v
        C[ju, iu] = self.sector * v
        return C.reshape(-1)

    def _compress_torch(self, y: torch.Tensor) -> torch.Tensor:
        """Torch analog of :meth:`_compress`: full ``(D*D,)`` -> packed ``(dim,)``."""
        iu, ju, w = self._torch_pack_tensors()
        Y = y.reshape(self.D, self.D)
        return 0.5 * (Y[iu, ju] + self.sector * Y[ju, iu]) / w

    def matvec_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``S^T H S`` fully on-device: packed Tensor -> packed Tensor.

        The torch-native counterpart of :meth:`matvec`: it expands each packed
        column to the full space, calls :meth:`EigenProblem.matvec_torch` (which
        never leaves the device), and compresses back — all in torch, so a torch
        driver on a packed sector view keeps the same on-device iteration the full
        problem gets. Accepts ``(dim,)`` or ``(dim, k)``. Same per-apply FLOPs as
        :meth:`matvec` (see that method's note); the win is avoiding the host
        sync, not fewer FLOPs.
        """
        x = x.to(dtype=self.base.dtype, device=self.base.device)
        if x.ndim == 1:
            return self._compress_torch(self.base.matvec_torch(self._expand_torch(x)))
        full = torch.stack([self._expand_torch(x[:, j]) for j in range(x.shape[1])], dim=1)
        y = self.base.matvec_torch(full)
        return torch.stack([self._compress_torch(y[:, j]) for j in range(y.shape[1])], dim=1)

    def hdiag_torch(self) -> torch.Tensor:
        """Packed approximate diagonal as a device Tensor (see :meth:`hdiag`)."""
        iu, ju, _ = self._torch_pack_tensors()
        Hd = self.base.hdiag_torch().reshape(self.D, self.D)
        return Hd[iu, ju]

    def linear_operator(self) -> spla.LinearOperator:
        """Return a symmetric scipy ``LinearOperator`` view of the packed block."""
        return spla.LinearOperator(
            (self.dim, self.dim),
            matvec=self.matvec,
            matmat=self.matvec,
            rmatvec=self.matvec,  # S^T H S is Hermitian (S real) -> A^H == A
            dtype=self._dtype,
        )

    def hdiag(self) -> np.ndarray:
        """Return an approximate packed diagonal (Jacobi preconditioner).

        Used only to build a diagonal preconditioner (LOBPCG ``M`` / PRIMME
        ``OPinv``) and to seed LOBPCG's initial guess, so an approximation is
        both sufficient and standard. The *exact* packed diagonal of
        ``S^T H S`` for an off-diagonal packed entry ``(i < j)`` is
        ``H[ij, ij] + sector * H[ij, ji]``, but the coupling term ``H[ij, ji]``
        is an off-diagonal element of the full ``H`` that ``base.hdiag()`` does
        not expose. We therefore drop it and return the full-space diagonal
        restricted to the packed indices, ``H[ij, ij]`` (with diagonal packed
        entries ``(i, i)`` exact at ``H[ii, ii]``) — the dominant, preconditioner-
        relevant part. This is intentionally *not* ``diag(sparse())``.
        """
        Hd = self.base.hdiag().reshape(self.D, self.D)
        return Hd[self._iu, self._ju]

    def sparse(self) -> sps.csr_matrix:
        """Materialize the packed block ``S^T H S`` as a scipy CSR matrix (cached).

        Assembled by applying :meth:`matvec` to the identity (dense
        ``(dim, dim)`` intermediate) — same cost note as
        :meth:`EigenProblem.sparse`, but on the smaller packed dimension.
        """
        if self._H is None:
            cols = self.matvec(np.eye(self.dim, dtype=self._dtype))
            self._H = sps.csr_matrix(cols)
        return self._H

    # ---- back to full space ----------------------------------------------
    def unpack_eigenvectors(self, vectors: np.ndarray) -> np.ndarray:
        """Map packed eigenvectors ``(dim, k)`` to full product space ``(D*D, k)``.

        Each column is embedded via :meth:`_expand`; because ``S`` is an isometry
        the full vectors inherit the packed columns' normalization. ``k == 0``
        (e.g. an eigensolver that converged zero roots) returns an empty
        ``(D*D, 0)`` array rather than crashing ``np.column_stack`` on an
        empty list.
        """
        vectors = np.asarray(vectors, dtype=self._dtype)
        if vectors.ndim == 1:
            return self._expand(vectors)
        if vectors.shape[1] == 0:
            return np.empty((self.D * self.D, 0), dtype=self._dtype)
        return np.column_stack([self._expand(vectors[:, j]) for j in range(vectors.shape[1])])


# ============================================================================
# General complex-Hermitian problem: explicit matrix or matvec, NOT chemistry
# ============================================================================
class HermitianOperator:
    """General Hermitian operator built from an explicit matrix or a matvec.

    This is a *separate* entry point from :class:`EigenProblem`. Where
    ``EigenProblem`` represents a real symmetric CI Hamiltonian assembled from
    chemistry integrals (and is hardwired to ``float64``), ``HermitianOperator``
    wraps an arbitrary Hermitian operator ``A`` (complex or real) supplied
    directly, so complex-valued Hermitian matrices with real eigenvalues can be
    diagonalized without touching the chemistry path.

    It exposes the same duck-typed operator interface the adapters consume
    (:attr:`dim`, :meth:`matvec`, :meth:`hdiag`, :meth:`sparse`,
    :meth:`linear_operator`), so the complex-capable adapters (dense fallback,
    SciPy ARPACK/LOBPCG, in-house Davidson) work on it unchanged. The operator
    dtype is preserved end to end: a real input yields a ``float64`` problem that
    behaves exactly like the real path, while a complex input yields a
    ``complex128`` problem whose :meth:`linear_operator` uses a genuine conjugate
    transpose for ``rmatvec`` (``A^H``, not ``A^T``).

    Construct with either a matrix (``HermitianOperator(A)``) or a matrix-free
    operator (``HermitianOperator.from_matvec(matvec, hdiag, dim, dtype=...)``).
    """

    def __init__(
        self,
        matrix,
        *,
        check_hermitian: bool = True,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ):
        """Build a problem from an explicit dense or sparse Hermitian matrix.

        Args:
            matrix: A square ``(n, n)`` dense array or scipy sparse matrix. Real
                input is kept real (``float64``); complex input is kept complex
                (``complex128``).
            check_hermitian: If ``True`` (default) verify ``A == A^H`` and raise
                :class:`ValueError` otherwise. Set ``False`` to skip the check
                for large operators you already trust.
            atol: Absolute tolerance for the Hermiticity check.
            rtol: Relative tolerance for the Hermiticity check.

        Raises:
            ValueError: If ``matrix`` is not square, or (when ``check_hermitian``)
                not Hermitian within tolerance.
        """
        if sps.issparse(matrix):
            A = matrix.tocsr()
            if A.shape[0] != A.shape[1]:
                raise ValueError(f"matrix must be square, got shape {A.shape}.")
            self._dense = None
            self._sparse_in = A
            self.dtype = np.result_type(A.dtype, np.float64)
            self.dim = int(A.shape[0])
            if check_hermitian:
                diff = A - A.getH()
                if diff.nnz and np.max(np.abs(diff.data)) > atol + rtol * _abs_max_sparse(A):
                    raise ValueError(
                        "matrix is not Hermitian (A != A^H) within tolerance; "
                        "pass check_hermitian=False to bypass."
                    )
        else:
            A = np.asarray(matrix)
            if A.ndim != 2 or A.shape[0] != A.shape[1]:
                raise ValueError(f"matrix must be a square 2-D array, got shape {A.shape}.")
            self.dtype = np.result_type(A.dtype, np.float64)
            A = A.astype(self.dtype, copy=False)
            self._dense = A
            self._sparse_in = None
            self.dim = int(A.shape[0])
            if check_hermitian and not np.allclose(A, A.conj().T, atol=atol, rtol=rtol):
                raise ValueError(
                    "matrix is not Hermitian (A != A^H) within tolerance; "
                    "pass check_hermitian=False to bypass."
                )
        self._matvec_fn: Callable | None = None
        self._hdiag: np.ndarray | None = None
        self._H: sps.csr_matrix | None = None

    @classmethod
    def from_matvec(
        cls,
        matvec: Callable[[np.ndarray], np.ndarray],
        hdiag: np.ndarray,
        dim: int,
        *,
        dtype=np.complex128,
    ) -> "HermitianOperator":
        """Build a matrix-free problem from a Hermitian ``matvec`` and diagonal.

        The caller guarantees ``matvec`` applies a Hermitian operator (no check
        is possible without materializing the matrix). ``hdiag`` is the operator
        diagonal, used only for preconditioning/guesses (its real part is used).

        Args:
            matvec: Applies ``A`` to a ``(dim,)`` vector or ``(dim, k)`` block.
            hdiag: Operator diagonal, shape ``(dim,)``.
            dim: Operator dimension ``n``.
            dtype: Result dtype (``complex128`` by default; pass ``float64`` for a
                real operator).

        Returns:
            A matrix-free :class:`HermitianOperator`.
        """
        obj = cls.__new__(cls)
        obj.dim = int(dim)
        obj.dtype = np.dtype(dtype)
        obj._dense = None
        obj._sparse_in = None
        obj._matvec_fn = matvec
        obj._hdiag = np.asarray(hdiag).reshape(-1)
        obj._H = None
        return obj

    # ---- operator interface ----------------------------------------------
    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``A`` to ``x`` of shape ``(dim,)`` or ``(dim, k)``."""
        if self._matvec_fn is not None:
            return self._matvec_fn(x)
        if self._dense is not None:
            return self._dense @ x
        return self._sparse_in @ x

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``A^H`` (the adjoint) to ``x``.

        For a Hermitian operator ``A^H == A``, so this equals :meth:`matvec`; it
        exists so :meth:`linear_operator` can advertise a correct adjoint (a
        conjugate transpose, not a plain transpose) to SciPy.
        """
        return self.matvec(x)

    def hdiag(self) -> np.ndarray:
        """Return the (real) operator diagonal, used for preconditioning."""
        if self._hdiag is not None:
            return np.real(self._hdiag)
        if self._dense is not None:
            return np.real(np.diagonal(self._dense))
        return np.real(self._sparse_in.diagonal())

    def linear_operator(self) -> spla.LinearOperator:
        """Return a scipy ``LinearOperator`` view of ``A`` with the true adjoint.

        ``rmatvec`` is the conjugate transpose ``A^H`` (equal to ``matvec`` for a
        Hermitian operator), and the operator dtype matches the input, so SciPy's
        complex-Hermitian code paths (``eigsh``/``lobpcg``) engage for complex
        input while real input behaves exactly as before.
        """
        return spla.LinearOperator(
            (self.dim, self.dim),
            matvec=self.matvec,
            matmat=self.matvec,
            rmatvec=self.rmatvec,  # A^H (== A for Hermitian A)
            dtype=self.dtype,
        )

    def sparse(self) -> sps.csr_matrix:
        """Materialize ``A`` as a scipy CSR matrix (cached).

        Returns the stored matrix directly when built from one; for a matrix-free
        problem it assembles ``A`` by applying :meth:`matvec` to the identity
        (dense ``(dim, dim)`` intermediate — same cost note as
        :meth:`EigenProblem.sparse`).
        """
        if self._H is None:
            if self._sparse_in is not None:
                self._H = self._sparse_in.astype(self.dtype)
            elif self._dense is not None:
                self._H = sps.csr_matrix(self._dense)
            else:
                cols = self.matvec(np.eye(self.dim, dtype=self.dtype))
                self._H = sps.csr_matrix(cols)
        return self._H


def _abs_max_sparse(A) -> float:
    """Largest absolute entry of a sparse matrix (0.0 if empty)."""
    return float(np.max(np.abs(A.data))) if A.nnz else 0.0


# ============================================================================
# Non-product (GDB) problem: an explicit (alpha_i, beta_i) determinant pair list
# ============================================================================
class PairCIProblem:
    """Projected Hamiltonian on an explicit ``(alpha_i, beta_i)`` determinant-pair list.

    Where :class:`EigenProblem` spans the *Cartesian product* of the alpha and
    beta CI strings (flat index ``ia * Nb + ib``), this problem spans an explicit
    parallel list of ``(alpha_i, beta_i)`` pairs — the "GDB" / general-determinant
    basis contract used by ``SBDSolver`` and ``torch_sci_solver``. The dimension is
    the number of *pairs*, not the product ``Na * Nb``.

    It is backed directly by ``torch_sci_solver``'s vendored **pair engine**
    (``pairs``/``mult_pairs``/``rdm_pairs``/``pair_helpers``), which applies the
    full-determinant Slater--Condon rules to a pair basis. That kernel carries **no
    per-spin parity factorization and no ``sigma`` fold** — the amplitudes are
    in interleaved-Wick signs, the convention the pair RDM builders expect, so the
    operator is ``sigma . H . sigma`` of the spin-block one (``sigma`` from
    ``_inter_to_block_signs_pair``) and, unlike :class:`EigenProblem`, there is no
    interleaved-Wick similarity to wrap around
    ``matvec``. Real integrals go through ``to_torch_integrals``; complex Hermitian
    integrals go through :func:`~sci_solvers.complex_ci_problem._complex_torch_integrals`,
    yielding a genuinely Hermitian complex operator.

    Exposes the same duck-typed operator interface the adapters consume
    (:attr:`dim`, :meth:`matvec`, :meth:`matvec_torch`, :meth:`hdiag`,
    :meth:`hdiag_torch`, :meth:`linear_operator`, :meth:`sparse`) plus the fields
    result-packing needs (:attr:`pair_basis`, :attr:`pair_helpers`, the deduped +
    lex-sorted :attr:`strs_a`/:attr:`strs_b`, :attr:`norb`, :attr:`nelec`,
    :attr:`hcore`, :attr:`eri`). It deliberately has **no** ``Na``/``Nb`` attributes,
    so nothing can accidentally reshape it back into a product.

    Note ``pair_basis_from_arrays`` dedups and lex-sorts the pair list, so the pair
    order generally differs from the input order — always read :attr:`strs_a` /
    :attr:`strs_b` back from the basis when aligning amplitudes.
    """

    def __init__(
        self,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        *,
        device: "str | torch.device" = "cpu",
        dtype: torch.dtype = torch.float64,
        chunk_size: int = 2_000_000,
    ):
        """Build a projected-Hamiltonian problem for one determinant-pair list.

        Args:
            strs_a: Alpha CI strings (int64 bitmask array), parallel to ``strs_b``:
                index ``i`` defines the pair ``(strs_a[i], strs_b[i])``.
            strs_b: Beta CI strings (int64 bitmask array), parallel to ``strs_a``.
            hcore: One-electron integrals, shape ``(norb, norb)`` (may be complex).
            eri: Two-electron integrals, full 4-index chemists' notation (may be
                complex Hermitian).
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            device: Torch device for the matvec kernel (``"cpu"`` or ``"cuda"``).
            dtype: Torch float/complex dtype. Overridden to ``torch.complex128``
                automatically when the integrals are complex.
            chunk_size: Matvec/RDM batch size for the pair kernels.
        """
        strs_a = np.ascontiguousarray(strs_a, dtype=np.int64)
        strs_b = np.ascontiguousarray(strs_b, dtype=np.int64)
        if strs_a.shape != strs_b.shape:
            raise ValueError(
                "strs_a and strs_b must be parallel pair lists of equal length, got "
                f"{strs_a.shape} and {strs_b.shape}."
            )
        self.norb = int(norb)
        self.nelec = (int(nelec[0]), int(nelec[1]))
        self.device = torch.device(device) if isinstance(device, str) else device
        self.chunk_size = chunk_size

        is_complex = np.iscomplexobj(hcore) or np.iscomplexobj(eri)
        self.is_complex = bool(is_complex)
        if is_complex:
            self.hcore = np.asarray(hcore, dtype=np.complex128)
            self.eri = np.asarray(eri, dtype=np.complex128)
            self.dtype = torch.complex128
            self._np_dtype: type[np.complex128] | type[np.float64] = np.complex128
        else:
            self.hcore = np.asarray(hcore, dtype=np.float64)
            self.eri = np.asarray(eri, dtype=np.float64)
            self.dtype = dtype
            self._np_dtype = np.float64

        # Build the (deduped, lex-sorted) pair basis eagerly so `dim`, `strs_a`,
        # and `strs_b` reflect the *basis* order the amplitudes come back in.
        from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays

        self.pair_basis = pair_basis_from_arrays(
            strs_a.tolist(), strs_b.tolist(), device=self.device
        )
        self.dim = int(self.pair_basis.n_pairs)
        # Read the pair strings back from the basis (dedup + lex sort reorders them).
        self.strs_a = self.pair_basis.alpha_of_pair.cpu().numpy().astype(np.int64)
        self.strs_b = self.pair_basis.beta_of_pair.cpu().numpy().astype(np.int64)

        self.pair_helpers = None
        self._ctx_cache: tuple[Callable, torch.Tensor] | None = None
        self._H: sps.csr_matrix | None = None

    # ---- matrix-free matvec (torch_sci_solver pair kernel) ---------------
    def _ctx(self) -> tuple[Callable, torch.Tensor]:
        """Cache the pair matvec closure and the Hamiltonian diagonal.

        Returns ``(matvec_pair, hii)`` where ``matvec_pair`` applies ``H`` to a
        single ``(dim,)`` Tensor and ``hii`` is the ``(dim,)`` diagonal. Unlike
        :meth:`EigenProblem._ctx` there is **no** ``sigma`` sign vector: the pair
        kernel applies the full-determinant Slater--Condon rules directly, so the
        operator is already in the amplitude convention the RDM builders expect.
        """
        if self._ctx_cache is None:
            from torch_sci_solver.torch_impl.mult_pairs import (
                make_diag_terms_pair_torch,
                mult_pair_torch,
            )
            from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers

            if self.is_complex:
                from sci_solvers.complex_ci_problem import _complex_torch_integrals

                integrals = _complex_torch_integrals(
                    self.hcore, self.eri, self.norb, self.device, self.dtype
                )
            else:
                from torch_sci_solver.sbd_ref.integrals import (
                    setup_integrals_from_pyscf_h1_eri,
                )
                from torch_sci_solver.torch_impl.integrals import to_torch_integrals

                i0, i1, i2 = setup_integrals_from_pyscf_h1_eri(self.hcore, self.eri, 0.0)
                integrals = to_torch_integrals(i0, i1, i2, device=self.device, dtype=self.dtype)

            self.pair_helpers = generate_pair_helpers(
                self.pair_basis, self.norb, device=self.device
            )
            hii = make_diag_terms_pair_torch(
                self.pair_basis, self.norb, integrals, chunk_size=self.chunk_size
            )

            def matvec_pair(v: torch.Tensor) -> torch.Tensor:
                return mult_pair_torch(
                    hii,
                    v,
                    self.pair_basis,
                    self.pair_helpers,
                    self.norb,
                    integrals,
                    chunk_size=self.chunk_size,
                )

            self._ctx_cache = (matvec_pair, hii)
        return self._ctx_cache

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply ``H`` to ``x`` of shape ``(dim,)`` or ``(dim, k)``.

        The *numpy* entry point SciPy solvers (ARPACK/LOBPCG/PRIMME) need: it
        accepts and returns ``np.ndarray``, moving data to/from ``self.device``
        each call. Torch-native drivers should call :meth:`matvec_torch` instead.
        """
        y = self.matvec_torch(torch.as_tensor(x, dtype=self.dtype, device=self.device))
        return y.cpu().numpy()

    def matvec_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``H`` fully on-device: Tensor -> Tensor.

        Accepts ``(dim,)`` (single vector) or ``(dim, k)`` (block). The pair
        kernel's scatter uses complex ``index_add_``, which ``torch.vmap`` does not
        compose with, so a block is applied **column by column** rather than via a
        batched ``vmap`` (the same caveat the complex product path documents).
        """
        matvec_pair, _ = self._ctx()
        x = x.to(dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            return matvec_pair(x)
        cols = [matvec_pair(x[:, j]) for j in range(x.shape[1])]
        return torch.stack(cols, dim=1)

    def linear_operator(self) -> spla.LinearOperator:
        """Return a scipy ``LinearOperator`` view of ``H`` with the true adjoint.

        ``rmatvec`` is :meth:`matvec` — ``H`` is Hermitian (``H^H == H``) — and the
        operator dtype matches the integrals, so SciPy's complex-Hermitian code
        paths engage for complex integrals while real integrals behave as before.
        """
        return spla.LinearOperator(
            (self.dim, self.dim),
            matvec=self.matvec,
            matmat=self.matvec,
            rmatvec=self.matvec,  # H is Hermitian (H^H == H)
            dtype=self._np_dtype,
        )

    # ---- diagonal --------------------------------------------------------
    def hdiag(self) -> np.ndarray:
        """Return the (real) Hamiltonian diagonal (used for preconditioning)."""
        return np.real(self.hdiag_torch().cpu().numpy().ravel())

    def hdiag_torch(self) -> torch.Tensor:
        """Return the Hamiltonian diagonal as a ``(dim,)`` Tensor on ``self.device``."""
        _, hii = self._ctx()
        return hii.reshape(-1)

    # ---- sparse matrix ---------------------------------------------------
    def sparse(self) -> sps.csr_matrix:
        """Materialize ``H`` as a scipy CSR matrix (cached).

        Assembled by applying :meth:`matvec` to the identity (dense ``(dim, dim)``
        intermediate — same cost note as :meth:`EigenProblem.sparse`). Only for
        adapters that need an explicit matrix and for modest ``dim``.
        """
        if self._H is None:
            cols = self.matvec(np.eye(self.dim, dtype=self._np_dtype))
            self._H = sps.csr_matrix(cols)
        return self._H
