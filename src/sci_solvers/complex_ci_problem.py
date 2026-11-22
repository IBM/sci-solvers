"""Complex Hermitian CI problem over the full CAS FCI space.

This is the complex-integral counterpart to the real chemistry
:class:`~sci_solvers.eigen_problem.EigenProblem`. Where ``EigenProblem`` assembles
a *real* selected-CI Hamiltonian via ``torch_sci_solver``'s FCI kernel, that
kernel's Slater-Condon evaluation assumes real integrals and would silently
produce a complex-*symmetric* (not Hermitian) operator for complex integrals.
So for complex Hermitian integrals -- e.g. a single non-Gamma k-point or a
spin-orbit-coupled Hamiltonian of a periodic system -- we build the matvec from
PySCF's :mod:`pyscf.fci.direct_nosym`, which is written for non-symmetric /
complex integrals: it splits the contraction into real/imaginary parts, keeps
the full complex product, and never routes the ERI through ``ao2mo.restore``
(which rejects complex input).

The result is wrapped as a matrix-free
:class:`~sci_solvers.eigen_problem.HermitianOperator`, so the complex-capable
adapters (``davidson``/``arpack``/``lobpcg`` and the dense fallback) diagonalize
it through the same interface they use for the real path.

This module offers two builders:

* :func:`build_complex_ci_problem` -- the **full CAS FCI space** via
  ``pyscf.fci.direct_nosym`` (above). Correct for any full active space, but the
  dense complex contraction is slow at large ``norb`` and cannot represent an
  *arbitrary selected* determinant list.
* :func:`build_selected_complex_ci_problem` -- an **arbitrary selected CI space**
  (a chosen ``(strs_a, strs_b)`` determinant product basis, exactly what the SQD
  pipeline samples) via ``torch_sci_solver``'s batched Slater-Condon kernel. This
  is the fast path for the quantum-hardware selected-CI runs, and the reason the
  complex Hermitian support was added.

The torch kernel was written for real integrals, but its element evaluators index
the ERI by *sorted* spin-orbital labels, so feeding it a complex Hermitian ERI
yields a genuinely Hermitian operator (real spectrum) with no algorithm change --
only the integral / amplitude dtype has to become complex. The complex path
therefore reuses the *unmodified* real kernel functions through a complex
``TorchIntegrals`` bundle; the real ``float64`` path in
:class:`~sci_solvers.eigen_problem.EigenProblem` is untouched.

Scope / limitations
--------------------
* :func:`build_complex_ci_problem` covers the full CAS FCI space for
  ``(neleca, nelecb)`` in ``norb`` orbitals -- dimension
  ``num_strings(norb, neleca) * num_strings(norb, nelecb)``.
* :func:`build_selected_complex_ci_problem` covers an arbitrary product basis
  ``span{|a>|b>}`` for explicit ``(strs_a, strs_b)`` -- dimension
  ``len(strs_a) * len(strs_b)`` -- the selected-CI analogue of
  :class:`~sci_solvers.eigen_problem.EigenProblem` but complex-Hermitian-capable.
* A real Hamiltonian is rejected by both (use the real ``EigenProblem`` path) so
  the two paths stay disentangled and the real path's behavior is never altered.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import Tuple

import numpy as np

from sci_solvers.eigen_problem import HermitianOperator
from sci_solvers.hamiltonian import HamiltonianData


def _check_hermitian_integrals(
    hcore: np.ndarray, eri: np.ndarray, *, atol: float, rtol: float
) -> None:
    """Raise ``ValueError`` unless ``hcore``/``eri`` give a Hermitian CI operator.

    For the two-electron CI operator ``0.5 * sum (pq|rs) a_p^ a_r^ a_s a_q`` to be
    Hermitian, the chemist-notation ERI must satisfy BOTH:

    * ``(pq|rs) == conj((qp|sr))`` -- conjugate / bra-ket exchange;
    * ``(pq|rs) == (rs|pq)`` -- particle-1 <-> particle-2 exchange.

    A physical complex ERI (from ``<pr|g|qs>`` of a Hermitian two-body operator) has
    both; either one alone leaves the CI operator non-Hermitian, so both are checked
    and a violation fails loudly rather than returning a silently-wrong operator.
    """
    if not np.allclose(hcore, hcore.conj().T, atol=atol, rtol=rtol):
        raise ValueError("hcore is not Hermitian (hcore != hcore^H) within tolerance.")
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


def build_complex_ci_problem(
    hcore: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> HermitianOperator:
    """Build a full-CAS complex Hermitian CI problem as a :class:`HermitianOperator`.

    Args:
        hcore: One-electron integrals, complex ``(norb, norb)``, Hermitian
            (``hcore == hcore.conj().T``).
        eri: Two-electron integrals, full dense chemist-order complex
            ``(norb, norb, norb, norb)`` tensor.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        check_hermitian: If ``True`` (default) verify ``hcore`` is Hermitian and
            the ERI carries the expected Hermitian symmetry ``(pq|rs) ==
            conj((qp|sr))`` before building; raise :class:`ValueError` otherwise.
        atol: Absolute tolerance for the Hermiticity checks. The default
            ``1e-7`` covers the near-zero-element noise floor of a
            two-component SCF export (elements that should vanish but carry
            ~1e-8 absolute round-off), which no ``rtol`` can absorb.
        rtol: Relative tolerance for the Hermiticity checks. Integrals from a
            two-component SCF (SOC / GHF) are Hermitian only up to a round-off that
            scales with their magnitude; ``np.allclose`` compares against
            ``atol + rtol * |b|``, so a nonzero ``rtol`` accepts that scaled
            round-off while still catching a residual comparable to the operator.

    Returns:
        A matrix-free :class:`~sci_solvers.eigen_problem.HermitianOperator` whose
        ``matvec`` applies the complex Hermitian CI Hamiltonian and whose
        ``hdiag`` is the (real) FCI diagonal.

    Raises:
        ValueError: If shapes are inconsistent, the operator is not complex, or
            (when ``check_hermitian``) the integrals are not Hermitian.
    """
    from pyscf.fci import cistring, direct_nosym

    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    norb = int(norb)
    neleca, nelecb = int(nelec[0]), int(nelec[1])

    if hcore.shape != (norb, norb):
        raise ValueError(f"hcore shape {hcore.shape} != ({norb}, {norb}).")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"eri must be the full dense (norb,)*4 tensor, got {eri.shape} for norb={norb}."
        )
    if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        raise ValueError(
            "build_complex_ci_problem is for complex Hermitian integrals; got real "
            "integrals. Use the real EigenProblem / chemistry solver path instead."
        )

    # Work in a common complex dtype so direct_nosym's real/imag split engages.
    dtype = np.result_type(hcore.dtype, eri.dtype, np.complex128)
    hcore = hcore.astype(dtype, copy=False)
    eri = eri.astype(dtype, copy=False)

    if check_hermitian:
        _check_hermitian_integrals(hcore, eri, atol=atol, rtol=rtol)

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    dim = int(na * nb)

    # Fold h1 into the 2e tensor once (direct_nosym.contract_1e is float64-only,
    # so the 1-body term must go through absorb_h1e/contract_2e for complex).
    h2e = direct_nosym.absorb_h1e(hcore, eri, norb, (neleca, nelecb), 0.5)

    def matvec(x: np.ndarray) -> np.ndarray:
        """Apply H to a (dim,) vector or (dim, k) block of CI amplitudes."""
        x = np.asarray(x, dtype=dtype)
        if x.ndim == 1:
            civec = x.reshape(na, nb)
            hc = direct_nosym.contract_2e(h2e, civec, norb, (neleca, nelecb))
            return np.asarray(hc, dtype=dtype).reshape(-1)
        cols = []
        for j in range(x.shape[1]):
            civec = x[:, j].reshape(na, nb)
            hc = direct_nosym.contract_2e(h2e, civec, norb, (neleca, nelecb))
            cols.append(np.asarray(hc, dtype=dtype).reshape(-1))
        return np.column_stack(cols)

    hdiag = np.asarray(direct_nosym.make_hdiag(hcore, eri, norb, (neleca, nelecb))).reshape(-1)

    return HermitianOperator.from_matvec(matvec, hdiag, dim, dtype=dtype)


def build_complex_ci_problem_from_hamiltonian(
    hamiltonian: HamiltonianData,
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> HermitianOperator:
    """Build a full-CAS complex Hermitian CI problem from a :class:`HamiltonianData`.

    Convenience wrapper over :func:`build_complex_ci_problem` that reads the
    integrals and electron counts off ``hamiltonian``. Raises the same errors;
    in particular a real ``hamiltonian`` is rejected (use the real path).
    """
    return build_complex_ci_problem(
        hamiltonian.hcore,
        hamiltonian.eri,
        hamiltonian.num_orbitals,
        (hamiltonian.num_elec_a, hamiltonian.num_elec_b),
        check_hermitian=check_hermitian,
        atol=atol,
        rtol=rtol,
    )


def _complex_torch_integrals(hcore: np.ndarray, eri: np.ndarray, norb: int, device, dtype):
    """Build a complex spin-orbital :class:`TorchIntegrals` from spatial integrals.

    Mirrors ``torch_sci_solver.sbd_ref.setup_integrals_from_pyscf_h1_eri`` but is
    complex-aware and fully vectorized: the reference builder casts every element
    through ``float(...)`` and stores ``np.float64`` arrays, which would discard
    the imaginary part. Here the ``(2L, 2L)`` one-electron and ``(2L,)^4``
    two-electron spin-orbital tensors are assembled directly as complex arrays
    with the same interleaved-spin layout (``2i``/``2i+1`` for alpha/beta) and the
    same spin-selection zeros (only same-spin-within-bra / same-spin-within-ket
    blocks are populated), then handed to the batched matvec.

    Args:
        hcore: Complex Hermitian ``(norb, norb)`` one-electron integrals.
        eri: Complex chemist-order ``(norb, norb, norb, norb)`` two-electron
            integrals with the two Hermitian symmetries the CI operator needs.
        norb: Number of spatial orbitals ``L``.
        device: Target torch device.
        dtype: Complex torch dtype (``torch.complex128``).

    Returns:
        A ``torch_sci_solver.torch_impl.integrals.TorchIntegrals`` carrying complex
        tensors, ready for ``build_matvec`` / ``make_diag_terms_torch``.
    """
    import torch
    from torch_sci_solver.torch_impl.integrals import TorchIntegrals

    n = 2 * norb
    hc = np.asarray(hcore)
    er = np.asarray(eri)

    # One-electron: same h(i,j) for the alpha (2i) and beta (2i+1) spin channels.
    I1 = np.zeros((n, n), dtype=np.complex128)
    I1[0::2, 0::2] = hc
    I1[1::2, 1::2] = hc

    # Two-electron: populate the four same-spin-within-bra / same-spin-within-ket
    # blocks (aa|aa), (aa|bb), (bb|aa), (bb|bb) with the spatial (ij|kl); the
    # remaining spin combinations stay zero (spin selection).
    I2 = np.zeros((n, n, n, n), dtype=np.complex128)
    I2[0::2, 0::2, 0::2, 0::2] = er  # aa|aa
    I2[0::2, 0::2, 1::2, 1::2] = er  # aa|bb
    I2[1::2, 1::2, 0::2, 0::2] = er  # bb|aa
    I2[1::2, 1::2, 1::2, 1::2] = er  # bb|bb

    I1_t = torch.tensor(I1, dtype=dtype, device=device)
    I2_t = torch.tensor(I2, dtype=dtype, device=device)
    I1_diag = torch.diagonal(I1_t).contiguous()
    # Direct(p,q) = (pp|qq), Exchange(p,q) = (pq|qp) — vectorized diagonals of I2.
    I2_direct = torch.einsum("ppqq->pq", I2_t).contiguous()
    I2_exchange = torch.einsum("pqqp->pq", I2_t).contiguous()

    return TorchIntegrals(
        I0=0.0,
        I1=I1_t,
        I2_eri=I2_t,
        I1_diag=I1_diag,
        I2_direct=I2_direct,
        I2_exchange=I2_exchange,
        n_spinorb=n,
        device=device,
        dtype=dtype,
    )


def build_selected_complex_ci_problem(
    strs_a: np.ndarray,
    strs_b: np.ndarray,
    hcore: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: Tuple[int, int],
    *,
    device: str = "cpu",
    chunk_size: int = 2_000_000,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> HermitianOperator:
    """Build a complex Hermitian *selected* CI problem via the torch kernel.

    This is the complex-Hermitian counterpart of
    :class:`~sci_solvers.eigen_problem.EigenProblem` for an **arbitrary selected
    determinant list** -- the product basis ``span{|a>|b>}`` over the explicit
    alpha/beta string arrays, exactly the space the SQD pipeline samples. It
    reuses ``torch_sci_solver``'s batched Slater-Condon kernel (fast at large
    ``norb``) with a *complex* integral bundle; the kernel's element evaluators
    index the ERI by sorted spin-orbital labels, so complex Hermitian integrals
    give a genuinely Hermitian operator (real spectrum) with no algorithm change.

    The kernel stores amplitudes in interleaved-Wick parity, which differs from
    the spin-block convention by a per-determinant ``+/-1`` sign ``sigma``; that
    similarity (``H_block = sigma . H_inter . sigma``) is folded into the returned
    ``matvec`` so the operator, its eigenvectors, and the diagonal are all in the
    spin-block convention the rest of the ecosystem expects -- matching
    :class:`~sci_solvers.eigen_problem.EigenProblem`.

    Args:
        strs_a: Alpha CI strings (int64 bitmask array).
        strs_b: Beta CI strings (int64 bitmask array).
        hcore: Complex Hermitian ``(norb, norb)`` one-electron integrals.
        eri: Full dense complex chemist-order ``(norb, norb, norb, norb)`` ERI.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)`` (documentary; the basis is defined by the
            explicit string arrays).
        device: Torch device for the matvec kernel (``"cpu"`` or ``"cuda"``).
        chunk_size: Matvec batch size for the torch kernel.
        check_hermitian: If ``True`` (default) verify ``hcore`` is Hermitian and
            the ERI carries both Hermitian symmetries before building.
        atol: Absolute tolerance for the Hermiticity checks. The default
            ``1e-7`` covers the near-zero-element noise floor of a
            two-component SCF export (elements that should vanish but carry
            ~1e-8 absolute round-off), which no ``rtol`` can absorb.
        rtol: Relative tolerance for the Hermiticity checks. Integrals from a
            two-component SCF (SOC / GHF) are Hermitian only up to a round-off that
            scales with their magnitude; ``np.allclose`` compares against
            ``atol + rtol * |b|``, so a nonzero ``rtol`` accepts that scaled
            round-off while still catching a residual comparable to the operator.

    Returns:
        A matrix-free :class:`~sci_solvers.eigen_problem.HermitianOperator` of
        dimension ``len(strs_a) * len(strs_b)`` whose ``matvec`` applies the
        complex Hermitian selected-CI Hamiltonian (spin-block convention) and
        whose ``hdiag`` is the (real) selected-CI diagonal.

    Raises:
        ValueError: If shapes are inconsistent, the integrals are real, or (when
            ``check_hermitian``) the integrals are not Hermitian.
    """
    import torch
    from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
    from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
    from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
    from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs
    from torch_sci_solver.torch_impl.sbdiag import _to_int64_python

    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    norb = int(norb)

    if hcore.shape != (norb, norb):
        raise ValueError(f"hcore shape {hcore.shape} != ({norb}, {norb}).")
    if eri.shape != (norb, norb, norb, norb):
        raise ValueError(
            f"eri must be the full dense (norb,)*4 tensor, got {eri.shape} for norb={norb}."
        )
    if not (np.iscomplexobj(hcore) or np.iscomplexobj(eri)):
        raise ValueError(
            "build_selected_complex_ci_problem is for complex Hermitian integrals; "
            "got real integrals. Use the real EigenProblem / chemistry solver path."
        )

    if check_hermitian:
        _check_hermitian_integrals(hcore, eri, atol=atol, rtol=rtol)

    dev = torch.device(device) if isinstance(device, str) else device
    dtype = torch.complex128

    strs_a = np.ascontiguousarray(strs_a, dtype=np.int64)
    strs_b = np.ascontiguousarray(strs_b, dtype=np.int64)
    dim = int(len(strs_a) * len(strs_b))

    adets = _to_int64_python(strs_a.tolist())
    bdets = _to_int64_python(strs_b.tolist())
    adets_t = torch.tensor(adets, dtype=torch.int64, device=dev)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=dev)

    integrals = _complex_torch_integrals(hcore, eri, norb, dev, dtype)
    helpers = list(generate_all_torch_helpers(adets, bdets, norb, device=dev))
    hii = make_diag_terms_torch(adets_t, bdets_t, norb, integrals, chunk_size=chunk_size)
    matvec_inter = build_matvec(
        "batched", hii, adets_t, bdets_t, norb, integrals, helpers, chunk_size=chunk_size
    )
    sigma = _inter_to_block_signs(adets_t, bdets_t, norb).to(dtype=dtype).reshape(-1)

    def matvec(x: np.ndarray) -> np.ndarray:
        """Apply H (spin-block convention) to a (dim,) vector or (dim, k) block.

        Columns are applied one at a time: ``torch.vmap`` (used on the real path
        for block solves) does not compose with the complex kernel's in-place
        ``index_add_`` accumulation, and the numpy adapters that consume this
        problem already iterate columns for a block.
        """
        xt = torch.as_tensor(np.asarray(x, dtype=np.complex128), dtype=dtype, device=dev)
        if xt.ndim == 1:
            y = matvec_inter(xt * sigma) * sigma
            return y.cpu().numpy()
        cols = [(matvec_inter(xt[:, j] * sigma) * sigma).cpu().numpy() for j in range(xt.shape[1])]
        return np.column_stack(cols)

    # The diagonal is invariant under the sigma similarity (sigma**2 == 1); its
    # imaginary part is ~0 for a Hermitian operator, and HermitianOperator takes
    # the real part for preconditioning.
    hdiag = hii.cpu().numpy().reshape(-1)

    problem = HermitianOperator.from_matvec(matvec, hdiag, dim, dtype=np.complex128)

    # Attach the physical attributes that the numpy consumers of a real
    # EigenProblem read, so this complex problem is a drop-in there:
    #   * SymmetrizedEigenProblem.__init__ reads Na/Nb/strs_a/strs_b,
    #   * MultirootSCISolver._get_eigen_states reads strs_a/strs_b/Na/Nb/norb/
    #     nelec/hcore/eri.
    # hcore/eri stay the ORIGINAL complex arrays (not float64-cast) so the
    # complex-correct RDM/energy rebuild sees the true integrals.
    problem.strs_a = strs_a
    problem.strs_b = strs_b
    problem.Na = int(len(strs_a))
    problem.Nb = int(len(strs_b))
    problem.norb = norb
    problem.nelec = (int(nelec[0]), int(nelec[1]))
    problem.hcore = hcore
    problem.eri = eri
    return problem
