"""Selected-CI problems whose projected Hamiltonian is built by fulqrum.

`fulqrum <https://github.com/qiskit-community/fulqrum>`_ (IBM, optional ``fulqrum``
extra) builds the projection of a Hamiltonian onto a set of bitstrings as a CSR
matrix, much faster than the torch Slater-Condon kernel behind
:class:`~sci_solvers.eigen_problem.EigenProblem`. This module wraps that matrix as a
:class:`~sci_solvers.eigen_problem.HermitianOperator` carrying the same chemistry
attributes the general-algebra reduction reads (``strs_a``/``strs_b``, ``Na``/``Nb``,
``norb``, ``nelec``, ``hcore``, ``eri``), so every general-algebra eigensolver, the
spin symmetrization and the result packing consume it unchanged. It backs
``hamiltonian_backend="fulqrum"`` on
:class:`~sci_solvers.energy_solvers.general_algebra_base.GeneralAlgebraSolverBase`.

fulqrum's conventions, adapted here (pinned by ``tests/test_fulqrum_conventions.py``):

* it sorts each half-string list and indexes the product **beta-major**
  (``ib * Na + ia``); the matrix is reordered to the input strings' ``ia * Nb + ib``;
* its basis differs from the PySCF spin-block one by one global sign, which leaves
  the matrix unchanged;
* ``integrals_to_fq_fermionic_op`` reads a complex chemist ERI conjugated, so a
  complex ERI is passed as ``eri.transpose(1, 0, 3, 2)``;
* full-string (GDB) subspaces are sorted by the full ``beta + alpha`` string, i.e.
  ``np.lexsort((alpha, beta))``, in spin-block signs.

The operator comes in three ``fulqrum_mode`` values, shared by every builder:

* ``"csr"`` (default): fulqrum's direct CSR build (``to_csr_linearoperator``), which
  counts the non-zeros first and allocates once, so its peak is about one matrix;
* ``"csr_fast"``: fulqrum's two-stage build (``to_csr_linearoperator_fast``): about
  30% faster, but it holds an intermediate row-wise copy, so its peak is about 2.5
  matrices;
* ``"matrix_free"``: fulqrum's matrix-free SpMV on the ``SubspaceHamiltonian``, which
  regenerates the elements on every product: memory of the subspace and term tables
  only, several times slower per product.

Both CSR modes apply the matrix with fulqrum's OpenMP SpMV. In every mode the matrix
stays in fulqrum's own basis order and the input-order operator is applied by
permuting vectors, ``(H x)[i] = (H_fq x_fq)[perm[i]]`` with ``x_fq[perm] = x``, so no
reordered copy of the matrix is made.

Three builders: :func:`build_fulqrum_ci_problem` for the product basis (TPB),
:class:`FulqrumPairCIProblem` for an explicit determinant-pair list (GDB), and
:func:`build_fulqrum_spinless_ci_problem` for the spinless (flat spin-orbital) path,
whose operator is built term by term from spin-orbital integrals (the spatial
``integrals_to_fq_fermionic_op`` always spin-doubles, so it cannot take them).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import hashlib
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sps

from sci_solvers.complex_ci_problem import _check_hermitian_integrals
from sci_solvers.eigen_problem import HermitianOperator, PairCIProblem

# The most recently built SubspaceHamiltonian, keyed by an integral digest. Its setup
# (operator split, grouping) depends only on the integrals, so it is reused across
# batches and calls; holding one entry bounds memory to the Hamiltonian in use.
_HSUB_CACHE: dict[str, Any] = {}


def _integrals_digest(hcore: np.ndarray, eri: np.ndarray) -> str:
    """Content digest of the integrals (callers pass fresh arrays, so not ``id``)."""
    h = hashlib.blake2b(digest_size=16)
    for a in (hcore, eri):
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fulqrum_subspace_hamiltonian(hcore: np.ndarray, eri: np.ndarray) -> Any:
    """Return a cached fulqrum ``SubspaceHamiltonian`` for spatial integrals.

    Electronic only: no constant term, matching the sci-solvers convention that
    callers add ``e_nuc`` themselves. Real integrals give a ``float64`` operator,
    complex ones a ``complex128`` operator.

    Args:
        hcore: One-electron integrals, ``(norb, norb)``.
        eri: Chemist-notation two-electron integrals, ``(norb,)*4``.

    Returns:
        A ``fulqrum.SubspaceHamiltonian``; set its subspace with ``update_subspace``.
    """
    import fulqrum as fq
    from fulqrum.convert.integrals import integrals_to_fq_fermionic_op

    is_complex = np.iscomplexobj(hcore) or np.iscomplexobj(eri)
    dtype = np.complex128 if is_complex else np.float64
    h1 = np.ascontiguousarray(hcore, dtype=dtype)
    h2 = np.ascontiguousarray(eri, dtype=dtype)
    key = _integrals_digest(h1, h2)
    hsub = _HSUB_CACHE.get(key)
    if hsub is None:
        # For a Hermitian ERI transpose(1, 0, 3, 2) == conj(eri), which undoes the
        # conjugated read; for a real 8-fold-symmetric ERI it would be a no-op.
        h2_fq = np.ascontiguousarray(h2.transpose(1, 0, 3, 2)) if is_complex else h2
        fop = integrals_to_fq_fermionic_op(h1, h2_fq)
        hsub = fq.SubspaceHamiltonian(fop.extended_jw_transformation())
        _HSUB_CACHE.clear()
        _HSUB_CACHE[key] = hsub
    return hsub


FULQRUM_MODES = ("csr", "csr_fast", "matrix_free")


def _fulqrum_operator(
    hsub: Any, mode: str, perm: np.ndarray, signs: np.ndarray | None = None
) -> HermitianOperator:
    """Present fulqrum's operator on its current subspace in the caller's basis order.

    Args:
        hsub: A ``fulqrum.SubspaceHamiltonian`` whose subspace is set.
        mode: One of :data:`FULQRUM_MODES` (see the module docstring).
        perm: ``perm[i]`` is fulqrum's index of the caller's basis state ``i``.
        signs: Optional ``(dim,)`` array of ``+-1``: present ``S H S`` with
            ``S = diag(signs)`` (the GDB interleaved-Wick convention).

    Returns:
        A matrix-free :class:`~sci_solvers.eigen_problem.HermitianOperator` with
        ``(A x)[i] = s_i (H_fq y)[perm[i]]``, ``y[perm] = s * x``. Nothing of size
        ``nnz`` is copied. In the CSR modes its :meth:`sparse` returns the reordered
        matrix, built on request only (it costs a full copy).

    Raises:
        ValueError: If ``mode`` is not one of :data:`FULQRUM_MODES`.
    """
    if mode not in FULQRUM_MODES:
        raise ValueError(f"fulqrum_mode must be one of {FULQRUM_MODES}, got {mode!r}.")
    dtype = np.dtype(hsub.dtype)
    dim = int(perm.shape[0])
    if mode == "matrix_free":
        fq_op: Any = hsub
        # The module caches one SubspaceHamiltonian and reuses it across batches, so
        # a later build may replace its subspace before this operator's products run.
        subspace = hsub.spmv.subspace

        def ensure() -> None:
            if hsub.spmv.subspace is not subspace:
                hsub.update_subspace(subspace)

        diag_fq = np.asarray(hsub.diagonal_vector())
    else:
        fq_op = hsub.to_csr_linearoperator() if mode == "csr" else hsub.to_csr_linearoperator_fast()
        # fulqrum's CSRLinearOperator stores ``self._matvec = self.matvec``, a reference
        # cycle that keeps its matrix alive until the cyclic garbage collector runs. Only
        # ``fq_op.matvec`` (the class method) is called below, so the attribute can go.
        vars(fq_op).pop("_matvec", None)

        def ensure() -> None:
            return None

        diag_fq = fq_op.matrix.diagonal()

    def apply(v: np.ndarray) -> np.ndarray:
        ensure()
        x_fq = np.empty(dim, dtype=dtype)
        x_fq[perm] = v if signs is None else signs * v
        y = np.asarray(fq_op.matvec(x_fq))[perm]
        return y if signs is None else signs * y

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 1:
            return apply(x)
        return np.column_stack([apply(x[:, j]) for j in range(x.shape[1])])

    problem = HermitianOperator.from_matvec(matvec, np.real(diag_fq)[perm], dim, dtype=dtype)
    # HermitianOperator is chemistry-agnostic; attach the mode as the builders attach theirs.
    vars(problem)["fulqrum_mode"] = mode
    if mode != "matrix_free":
        matrix = fq_op.matrix
        # Cached here, not on ``problem``: a closure over ``problem`` stored on ``problem``
        # is a reference cycle, which keeps ``fq_op``'s matrix alive until the cyclic
        # garbage collector happens to run.
        cache: list[sps.csr_matrix] = []

        def sparse() -> sps.csr_matrix:
            if not cache:
                h = sps.csr_matrix(matrix)[perm][:, perm]
                if signs is not None:
                    s_diag = sps.diags(signs)
                    h = s_diag @ h @ s_diag
                cache.append(sps.csr_matrix(h, dtype=dtype))
            return cache[0]

        problem.sparse = sparse  # type: ignore[method-assign]
    return problem


def build_fulqrum_ci_problem(
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
    fulqrum_mode: str = "csr",
) -> HermitianOperator:
    """Build the product-basis (TPB) selected-CI problem with fulqrum.

    The fulqrum counterpart of :class:`~sci_solvers.eigen_problem.EigenProblem`
    (real integrals) and
    :func:`~sci_solvers.complex_ci_problem.build_selected_complex_ci_problem`
    (complex Hermitian integrals): the basis is ``span{|a>|b>}`` over the explicit
    alpha/beta strings, in the input strings' ``ia * Nb + ib`` order and the PySCF
    spin-block sign convention.

    Args:
        strs_a: Alpha CI strings (bitmask integers), unique.
        strs_b: Beta CI strings (bitmask integers), unique.
        hcore: One-electron integrals, ``(norb, norb)``, real or complex Hermitian.
        eri: Chemist-notation two-electron integrals, ``(norb,)*4``.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)`` (documentary; the strings define the basis).
        check_hermitian: For complex integrals, verify the Hermiticity symmetries
            ``build_selected_complex_ci_problem`` checks before building.
        atol: Absolute tolerance for those checks.
        rtol: Relative tolerance for those checks.
        fulqrum_mode: How fulqrum holds and applies the operator; one of
            :data:`FULQRUM_MODES` (see the module docstring).

    Returns:
        A :class:`~sci_solvers.eigen_problem.HermitianOperator` (``float64`` or
        ``complex128``) applying fulqrum's operator in the input strings' order, with
        ``strs_a``, ``strs_b``, ``Na``, ``Nb``, ``norb``, ``nelec``, ``hcore``,
        ``eri`` and ``fulqrum_mode`` attached.

    Raises:
        ValueError: If the integral shapes are inconsistent, a string list has
            duplicates, or (complex integrals, ``check_hermitian``) the integrals
            are not Hermitian.
    """
    import fulqrum as fq

    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    norb = int(norb)
    if hcore.shape != (norb, norb):
        raise ValueError(f"hcore shape {hcore.shape} != ({norb}, {norb}).")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"eri must be the full dense (norb,)*4 tensor, got {eri.shape} for norb={norb}."
        )
    if check_hermitian and (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        _check_hermitian_integrals(hcore, eri, atol=atol, rtol=rtol)

    strs_a = np.asarray(strs_a)
    strs_b = np.asarray(strs_b)
    for name, strs in (("strs_a", strs_a), ("strs_b", strs_b)):
        if len(np.unique(strs)) != len(strs):
            raise ValueError(f"{name} contains duplicate CI strings; pass unique strings.")

    hsub = fulqrum_subspace_hamiltonian(hcore, eri)
    hsub.update_subspace(
        fq.Subspace([
            [format(int(x), f"0{norb}b") for x in strs_a],
            [format(int(x), f"0{norb}b") for x in strs_b],
        ])
    )
    # fulqrum sorts each list (ascending integers) and indexes the product as
    # rank_b * Na + rank_a; row i = ia * Nb + ib of the input order sits there.
    rank_a = np.argsort(np.argsort(strs_a))
    rank_b = np.argsort(np.argsort(strs_b))
    perm = (rank_b[None, :] * len(strs_a) + rank_a[:, None]).reshape(-1)
    problem = _fulqrum_operator(hsub, fulqrum_mode, perm)

    # The chemistry attributes the reduction reads, as build_selected_complex_ci_problem
    # attaches them (HermitianOperator itself is chemistry-agnostic).
    attrs = {
        "strs_a": strs_a,
        "strs_b": strs_b,
        "Na": int(len(strs_a)),
        "Nb": int(len(strs_b)),
        "norb": norb,
        "nelec": (int(nelec[0]), int(nelec[1])),
        "hcore": hcore,
        "eri": eri,
    }
    for name, value in attrs.items():
        setattr(problem, name, value)
    return problem


class FulqrumPairCIProblem(PairCIProblem):
    """A :class:`~sci_solvers.eigen_problem.PairCIProblem` whose operator fulqrum builds.

    Same pair basis (deduplicated, lex-sorted ``(alpha, beta)`` pairs), attributes and
    RDM helpers as the parent, so the general-algebra GDB path and its result packing
    consume it unchanged; only the matrix comes from fulqrum. It is presented in
    the parent's operator convention, ``sigma . H . sigma`` (interleaved-Wick signs,
    ``sigma`` from ``_inter_to_block_signs_pair``), for real and complex integrals
    alike.

    The parent's torch pair kernel is never run for the matvec; it only builds the
    ``pair_helpers`` the RDM builders need, lazily, when packing results.
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
        check_hermitian: bool = True,
        atol: float = 1e-7,
        rtol: float = 1e-5,
        fulqrum_mode: str = "csr",
        **kwargs: Any,
    ):
        """Build the pair basis (parent) and the fulqrum operator over it.

        Args:
            strs_a: Alpha CI strings, parallel to ``strs_b`` (pair ``i`` is
                ``(strs_a[i], strs_b[i])``); duplicates are collapsed by the parent.
            strs_b: Beta CI strings, parallel to ``strs_a``.
            hcore: One-electron integrals, ``(norb, norb)``, real or complex Hermitian.
            eri: Chemist-notation two-electron integrals, ``(norb,)*4``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            check_hermitian: For complex integrals, run the same Hermiticity checks
                as the torch complex builders before building.
            atol: Absolute tolerance for those checks.
            rtol: Relative tolerance for those checks.
            fulqrum_mode: How fulqrum holds and applies the operator; one of
                :data:`FULQRUM_MODES` (see the module docstring).
            **kwargs: Forwarded to :class:`PairCIProblem` (``device``, ``dtype``,
                ``chunk_size``), which the RDM builders use.
        """
        super().__init__(strs_a, strs_b, hcore, eri, norb, nelec, **kwargs)
        if check_hermitian and self.is_complex:
            _check_hermitian_integrals(self.hcore, self.eri, atol=atol, rtol=rtol)

        import fulqrum as fq
        import torch
        from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair

        width = self.norb
        full = [
            format(int(b), f"0{width}b") + format(int(a), f"0{width}b")
            for a, b in zip(self.strs_a, self.strs_b)
        ]
        hsub = fulqrum_subspace_hamiltonian(self.hcore, self.eri)
        hsub.update_subspace(fq.Subspace([full]))
        # fulqrum row r holds basis pair order[r]; inv maps basis row i to fulqrum row.
        inv = np.argsort(np.lexsort((self.strs_a, self.strs_b)))
        sigma = _inter_to_block_signs_pair(
            torch.as_tensor(self.strs_a), torch.as_tensor(self.strs_b), width
        ).numpy()
        self._fq = _fulqrum_operator(hsub, fulqrum_mode, inv, sigma.astype(np.float64))
        self.fulqrum_mode = fulqrum_mode

    def _ctx(self) -> Any:
        """Build only the ``pair_helpers`` the RDM builders need, not the torch kernel.

        The result packing calls ``problem._ctx()`` to populate ``pair_helpers``; the
        parent would also assemble the torch pair matvec and diagonal, which this
        problem never uses.
        """
        if self.pair_helpers is None:
            from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers

            self.pair_helpers = generate_pair_helpers(
                self.pair_basis, self.norb, device=self.device
            )
        return None

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply the fulqrum operator to ``x`` of shape ``(dim,)`` or ``(dim, k)``."""
        return self._fq.matvec(np.asarray(x, dtype=self._np_dtype))

    def matvec_torch(self, x: Any) -> Any:
        """Torch entry point, via the numpy CSR matvec (fulqrum's matrix is on CPU)."""
        import torch

        y = self.matvec(x.detach().cpu().numpy())
        return torch.as_tensor(y, dtype=self.dtype, device=self.device)

    def hdiag(self) -> np.ndarray:
        """Return the (real) operator diagonal."""
        return self._fq.hdiag()

    def hdiag_torch(self) -> Any:
        """Return the operator diagonal as a Tensor on ``self.device``."""
        import torch

        return torch.as_tensor(self._fq.hdiag(), dtype=self.dtype, device=self.device)

    def sparse(self) -> sps.csr_matrix:
        """Return the operator as a CSR matrix in the pair-basis order.

        A full reordered copy, built on request; with ``fulqrum_mode="matrix_free"``
        it is assembled densely (see :meth:`HermitianOperator.sparse`).
        """
        return self._fq.sparse()


def fulqrum_spinless_subspace_hamiltonian(hcore: np.ndarray, eri: np.ndarray) -> Any:
    """Return a cached fulqrum ``SubspaceHamiltonian`` for spin-orbital integrals.

    ``H = sum_pq h_pq a+_p a_q + 1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q`` over the
    ``nmo`` spin-orbitals, built term by term as a ``FermionicOperator``. Unlike
    :func:`fulqrum_subspace_hamiltonian`, the ERI is passed as given: the conjugated
    read is specific to ``integrals_to_fq_fermionic_op``. Shares the one-entry cache
    with the spatial builder (the key includes the integrals' shapes and contents).

    Args:
        hcore: Complex Hermitian one-electron integrals, ``(nmo, nmo)``.
        eri: Complex chemist-notation spin-orbital ERI, ``(nmo,)*4``.

    Returns:
        A complex ``fulqrum.SubspaceHamiltonian``; set its subspace with
        ``update_subspace``.
    """
    import itertools

    import fulqrum as fq

    h1 = np.ascontiguousarray(hcore, dtype=np.complex128)
    h2 = np.ascontiguousarray(eri, dtype=np.complex128)
    key = "spinless:" + _integrals_digest(h1, h2)
    hsub = _HSUB_CACHE.get(key)
    if hsub is None:
        nmo = h1.shape[0]
        terms: list[tuple[str, list[int], complex]] = [
            ("+-", [p, q], complex(h1[p, q]))
            for p, q in itertools.product(range(nmo), repeat=2)
            if abs(h1[p, q]) > 1e-14
        ]
        terms += [
            ("++--", [p, r, s, q], complex(0.5 * h2[p, q, r, s]))
            for p, q, r, s in itertools.product(range(nmo), repeat=4)
            if abs(h2[p, q, r, s]) > 1e-14
        ]
        fop = fq.FermionicOperator(nmo, terms)
        fop.combine_repeat_terms()
        hsub = fq.SubspaceHamiltonian(fop.extended_jw_transformation())
        _HSUB_CACHE.clear()
        _HSUB_CACHE[key] = hsub
    return hsub


def build_fulqrum_spinless_ci_problem(
    hcore: np.ndarray,
    eri: np.ndarray,
    n_particles: int,
    strs: np.ndarray,
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
    fulqrum_mode: str = "csr",
) -> HermitianOperator:
    """Build the spinless selected-CI problem with fulqrum.

    The fulqrum counterpart of
    :func:`~sci_solvers.spinless_ci_problem.build_spinless_complex_ci_problem`: the
    basis is the given ``N``-particle spin-orbital determinants, in their input order.
    fulqrum sorts the subspace internally (ascending integers); the matrix is mapped
    back to the input order, and its basis signs equal the brute-force
    ladder-operator ones up to one global sign, which leaves the matrix unchanged.

    Args:
        hcore: Complex Hermitian one-electron integrals, ``(nmo, nmo)``.
        eri: Complex chemist-notation spin-orbital ERI, ``(nmo,)*4``.
        n_particles: Total particle number ``N`` (scalar).
        strs: Unique ``N``-particle determinant bitmasks over ``nmo`` spin-orbitals.
        check_hermitian: Verify the Hermiticity symmetries before building.
        atol: Absolute tolerance for those checks.
        rtol: Relative tolerance for those checks.
        fulqrum_mode: How fulqrum holds and applies the operator; one of
            :data:`FULQRUM_MODES` (see the module docstring).

    Returns:
        A complex :class:`~sci_solvers.eigen_problem.HermitianOperator` applying
        fulqrum's operator in the input order, with ``strs``, ``norb=nmo``,
        ``nelec=(N, 0)``, ``hcore``, ``eri`` and ``fulqrum_mode`` attached, as the
        spinless result packing reads them.

    Raises:
        ValueError: If the integrals are real or inconsistently shaped, ``strs`` has
            duplicates or a wrong particle number, or (``check_hermitian``) the
            integrals are not Hermitian.
    """
    import fulqrum as fq

    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    nmo = int(hcore.shape[0])
    n = int(n_particles)
    if hcore.shape != (nmo, nmo) or eri.shape != (nmo,) * 4:
        raise ValueError(
            f"hcore must be (nmo, nmo) and eri (nmo,)*4; got {hcore.shape} and {eri.shape}."
        )
    if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        raise ValueError(
            "the spinless path is for complex Hermitian integrals; got real integrals."
        )
    if check_hermitian:
        _check_hermitian_integrals(hcore, eri, atol=atol, rtol=rtol)

    det_list = [int(x) for x in np.asarray(strs).tolist()]
    if len(set(det_list)) != len(det_list):
        raise ValueError("strs contains duplicate determinants; pass unique determinants.")
    if any(bin(d).count("1") != n for d in det_list):
        raise ValueError(f"every determinant in strs must have exactly {n} particles.")

    hsub = fulqrum_spinless_subspace_hamiltonian(hcore, eri)
    hsub.update_subspace(fq.Subspace([[format(d, f"0{nmo}b") for d in det_list]]))
    # fulqrum row k holds the k-th smallest determinant; input row i sits at its rank.
    rank = np.argsort(np.argsort(np.array(det_list, dtype=object)))
    problem = _fulqrum_operator(hsub, fulqrum_mode, rank)

    attrs = {
        "strs": np.asarray(det_list, dtype=object),
        "norb": nmo,
        "nelec": (n, 0),
        "hcore": hcore,
        "eri": eri,
    }
    for name, value in attrs.items():
        setattr(problem, name, value)
    return problem
