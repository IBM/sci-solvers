"""Factorized αβ channel of the product-basis (TPB) matvec.

The cross-spin two-body term of a spin-free Hamiltonian is

.. math::

    H_{\\alpha\\beta} = \\sum_{pq,rs} (pq|rs)\\, E^{\\alpha}_{pq} E^{\\beta}_{rs},
    \\qquad E^{\\sigma}_{pq} = a^\\dagger_{p\\sigma} a_{q\\sigma}.

On a product basis :math:`|I_\\alpha I_\\beta\\rangle` its matrix elements factorize
exactly, because the α and β operators act on separate strings:
:math:`\\langle I_\\alpha I_\\beta | E^\\alpha_{pq} E^\\beta_{rs} | J_\\alpha J_\\beta\\rangle
= \\langle I_\\alpha|E_{pq}|J_\\alpha\\rangle \\langle I_\\beta|E_{rs}|J_\\beta\\rangle`
(in the spin-block sign convention). Writing the ERI supermatrix as a sum of
rank-one terms, :math:`(pq|rs) = \\sum_P s_P B_P[pq] B_P[rs]` with
:math:`s_P = \\pm 1`, the αβ doubles block of the matvec becomes

.. math::

    \\sigma = \\sum_P s_P\\, A^\\alpha_P\\, C\\, (A^\\beta_P)^T, \\qquad
    A^\\sigma_P[I, J] = \\sum_{pq} B_P[pq] \\langle I|E^\\sigma_{pq}|J\\rangle,

restricted here to the off-diagonal single excitations (the diagonal and the
mean-field single terms stay in the diagonal and singles passes, unchanged).
:math:`A^\\sigma_P` has the sparsity of the σ singles table, so each factor
costs two sparse-times-dense products of :math:`N n_s` multiply-adds against
the :math:`N n_s^2` pair evaluations of the direct αβ kernel; it pays off
when the factor count is small against :math:`n_s` (low-rank molecular
integrals, larger orbital spaces).

The factors come from a pivoted Cholesky decomposition of the ERI supermatrix
(Beebe and Linderberg, Int. J. Quantum Chem. 12, 683 (1977); Koch, Sánchez de
Merás and Pedersen, J. Chem. Phys. 118, 9481 (2003)), exact to the chosen
tolerance and needing no auxiliary basis; an indefinite supermatrix (synthetic
test integrals) takes a signed eigendecomposition instead.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import warnings

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.helpers import TorchHelper
from torch_sci_solver.torch_impl.mult import _bra_from_offsets, _string_range_popcount
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

__all__ = ["FactorizedAlphaBeta", "factorize_eri"]


def factorize_eri(eri: Tensor, *, tol: float = 1e-10) -> tuple[Tensor, Tensor]:
    """Signed rank-one factors of a real spatial ERI supermatrix.

    Args:
        eri: ``(L, L, L, L)`` real chemist-notation ERI ``(pq|rs)``.
        tol: largest residual diagonal magnitude (pivoted Cholesky) or smallest
            kept eigenvalue magnitude (eigendecomposition fallback).

    Returns:
        ``(factors, signs)``: ``(k, L * L)`` float64 factors and ``(k,)`` ±1
        signs with ``(pq|rs) ≈ Σ_P signs[P] factors[P, pq] factors[P, rs]``.

    Raises:
        ValueError: a complex or non-square ERI, or a supermatrix that is not
            symmetric under ``(pq) ↔ (rs)`` to ``1e-10`` of its largest element.

    Notes
    -----
    The supermatrix :math:`V_{(pq),(rs)} = (pq|rs)` is first symmetrized,
    :math:`V \\leftarrow (V + V^T)/2` (integral programs leave a ~1e-12
    asymmetry, which the direct kernels see as the choice between
    :math:`(pq|rs)` and :math:`(rs|pq)`). Pivoted Cholesky with signed diagonal
    pivots keeps the residual diagonal :math:`d = \\mathrm{diag}(V - \\sum_P s_P
    L_P L_P^T)`, picks :math:`p = \\arg\\max |d|` and appends
    :math:`L_k = (V_{:,p} - \\sum_{j<k} s_j L_j L_{j,p}) / \\sqrt{|d_p|}` with
    :math:`s_k = \\mathrm{sign}(d_p)`, stopping when :math:`\\max |d| < \\mathrm{tol}`.
    For a positive semidefinite :math:`V` (physical integrals, a Coulomb Gram
    matrix) every sign is +1 and each element's error is bounded by ``tol``; the
    signed pivots also absorb the round-off-level negative directions of real
    integrals. A residual above ``10 tol`` after stopping means an indefinite
    matrix the diagonal cannot steer (synthetic integrals); the eigenvalues
    :math:`V = U \\Lambda U^T` are used instead, keeping
    :math:`|\\lambda| > \\mathrm{tol}` with factors
    :math:`\\sqrt{|\\lambda_P|} U_{:,P}` and signs :math:`\\mathrm{sign}(\\lambda_P)`.
    """
    if eri.is_complex():
        raise ValueError("the factorized αβ channel supports real integrals only")
    if eri.dim() != 4 or len(set(eri.shape)) != 1:
        raise ValueError(f"eri must be (L, L, L, L), got {tuple(eri.shape)}")
    n_orb = int(eri.shape[0])
    n = n_orb * n_orb
    v = eri.detach().to(device="cpu", dtype=torch.float64).reshape(n, n).clone()
    scale = float(v.abs().max()) if n else 0.0
    block = max(1, (1 << 24) // max(n, 1))  # rows per block: ~128 MB of float64 temporaries
    asym = 0.0
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        upper = v[lo:hi, lo:]
        lower = v[lo:, lo:hi].T
        asym = max(asym, float((upper - lower).abs().max()))
        mean = 0.5 * (upper + lower)
        v[lo:hi, lo:] = mean
        v[lo:, lo:hi] = mean.T
    if asym > 1e-10 * max(scale, 1.0):
        raise ValueError(
            f"the ERI supermatrix is not symmetric under (pq) <-> (rs) (max asymmetry {asym:.2e})"
        )
    capacity = min(n, max(4 * n_orb, 16))
    factors = torch.empty((capacity, n), dtype=torch.float64)
    signs = torch.empty(capacity, dtype=torch.float64)
    diag = v.diagonal().clone()
    k = 0
    while k < n:
        pivot = int(torch.argmax(diag.abs()))
        d_p = float(diag[pivot])
        if abs(d_p) < tol:
            break
        if k == capacity:
            capacity = min(n, 2 * capacity)
            factors = torch.cat([factors, torch.empty((capacity - k, n), dtype=torch.float64)])
            signs = torch.cat([signs, torch.empty(capacity - k, dtype=torch.float64)])
        column = v[:, pivot].clone()
        if k:
            column -= factors[:k].T @ (signs[:k] * factors[:k, pivot])
        sign = 1.0 if d_p > 0 else -1.0
        row = column / (abs(d_p) ** 0.5)
        factors[k] = row
        signs[k] = sign
        diag -= sign * row * row
        k += 1
    factors, signs = factors[:k].clone(), signs[:k].clone()
    residual = 0.0
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        rebuilt = (factors[:, lo:hi].T * signs) @ factors
        residual = max(residual, float((v[lo:hi] - rebuilt).abs().max()))
    if residual <= 10.0 * tol:
        return factors, signs
    values, vectors = torch.linalg.eigh(v)
    keep = values.abs() > tol
    factors = (vectors[:, keep] * values[keep].abs().sqrt()).T.contiguous()
    return factors, torch.sign(values[keep])


def _excitation_signs(strings: Tensor, ket: Tensor, cr: Tensor, an: Tensor) -> Tensor:
    """Spin-block sign ``<I|a†_p a_q|J>`` of each single excitation of a table.

    The sign is :math:`(-1)^{n}`, :math:`n` the occupied orbitals of the ket
    string strictly between ``p`` and ``q`` (spatial indices).
    """
    lo = torch.minimum(cr, an) + 1
    hi = torch.maximum(cr, an)
    count = _string_range_popcount(strings[ket], lo, torch.maximum(hi, lo))
    return 1 - 2 * (count & 1)


class FactorizedAlphaBeta:
    """The αβ doubles pass of a TPB matvec from signed ERI factors.

    Built once per subspace from the αβ helper (``task_type == 0``), whose α and
    β singles tables give the sparsity of :math:`A^\\alpha_P` and
    :math:`A^\\beta_P`; :meth:`__call__` adds the αβ doubles contribution to an
    output vector and replaces that helper in the matvec.

    Args:
        adets_t, bdets_t: ``(M,)`` single-word or ``(M, n_limbs)`` limb strings.
        n_orb: number of spatial orbitals ``L``.
        helper: the αβ (``task_type == 0``) helper of the subspace.
        factors: ``(k, L * L)`` factors from :func:`factorize_eri`.
        signs: ``(k,)`` ±1 factor signs.
        dtype: the matvec dtype (real).
        chunk_size: bound on the elements of one gathered block in the MPS
            ``index_add`` fallback (the sparse CSR path needs no chunking).

    Notes
    -----
    The engine's coefficients use the interleaved-Wick sign convention,
    :math:`C^{\\mathrm{inter}} = \\sigma \\odot C^{\\mathrm{block}}`; the factorized
    form holds in the spin-block convention, so the pass applies
    :math:`\\sigma \\odot \\left(\\sum_P s_P A^\\alpha_P (\\sigma \\odot C) (A^\\beta_P)^T\\right)`.
    A call holds three ``N``-element work arrays at its peak (the σ-signed
    input, one factor's α product and its transpose or the β product), and the
    pass keeps the σ bridge as int8 (``N`` bytes). On CPU and CUDA the two
    products per factor are sparse CSR times dense; MPS has no sparse CSR
    kernels and takes a chunked ``index_add`` fallback.
    """

    def __init__(
        self,
        adets_t: Tensor,
        bdets_t: Tensor,
        n_orb: int,
        helper: TorchHelper,
        factors: Tensor,
        signs: Tensor,
        *,
        dtype: torch.dtype,
        chunk_size: int = 2_000_000,
    ) -> None:
        """Gather the per-excitation factor indices, signs and the σ bridge."""
        if helper.task_type != 0:
            raise ValueError("FactorizedAlphaBeta needs the αβ (task_type == 0) helper")
        if dtype.is_complex:
            raise ValueError("the factorized αβ channel supports real dtypes only")
        device = adets_t.device
        self.m_a = int(adets_t.shape[0])
        self.m_b = int(bdets_t.shape[0])
        self.n_orb = int(n_orb)
        self.dtype = dtype
        self._chunk_size = max(1, int(chunk_size))
        self._sparse = device.type in ("cpu", "cuda")
        self.empty = (
            helper.a_singles_to is None
            or helper.b_singles_to is None
            or helper.a_singles_to.shape[0] == 0
            or helper.b_singles_to.shape[0] == 0
        )
        if self.empty:
            return
        tables = {}
        for spin, strings, offsets, ket, cr, an, m in (
            (
                "a",
                adets_t,
                helper.a_singles_offsets,
                helper.a_singles_to,
                helper.a_singles_cr,
                helper.a_singles_an,
                self.m_a,
            ),
            (
                "b",
                bdets_t,
                helper.b_singles_offsets,
                helper.b_singles_to,
                helper.b_singles_cr,
                helper.b_singles_an,
                self.m_b,
            ),
        ):
            ket = ket.to(device=device, dtype=torch.int64)
            p = cr.to(device=device, dtype=torch.int64) // 2  # α even, β odd spin orbitals
            q = an.to(device=device, dtype=torch.int64) // 2
            tables[spin] = dict(
                offsets=offsets.to(device=device, dtype=torch.int64),
                bra=_bra_from_offsets(offsets.to(device), m, device),
                ket=ket,
                pq=p * self.n_orb + q,
                sign=_excitation_signs(strings, ket, p, q).to(dtype),
                m=m,
            )
        self._tables = tables
        self._factors = factors.to(device=device, dtype=dtype)
        self._signs = [float(s) for s in signs.tolist()]
        sigma = _inter_to_block_signs(adets_t, bdets_t, self.n_orb)
        self._sigma = sigma.to(torch.int8)
        self._sigma_t = sigma.T.contiguous().to(torch.int8)
        del sigma

    def _apply(self, spin: str, values: Tensor, dense: Tensor) -> Tensor:
        """``A @ dense`` for the spin's single-excitation matrix with ``values``."""
        t = self._tables[spin]
        # cuSPARSE does not promise a fixed summation order; under deterministic
        # algorithms the index_add path (deterministic in that mode) runs instead.
        if self._sparse and not (dense.is_cuda and torch.are_deterministic_algorithms_enabled()):
            with warnings.catch_warnings():
                # torch flags sparse CSR as beta on first use; nothing to act on.
                warnings.filterwarnings("ignore", message=".*[Ss]parse CSR.*")
                a = torch.sparse_csr_tensor(t["offsets"], t["ket"], values, (t["m"], t["m"]))
                return a @ dense
        out = torch.zeros((t["m"], dense.shape[1]), dtype=dense.dtype, device=dense.device)
        step = max(1, self._chunk_size // max(int(dense.shape[1]), 1))
        for lo in range(0, int(values.shape[0]), step):
            hi = lo + step
            out.index_add_(0, t["bra"][lo:hi], values[lo:hi].unsqueeze(1) * dense[t["ket"][lo:hi]])
        return out

    def __call__(self, out: Tensor, x: Tensor) -> None:
        """Add the αβ doubles contribution of ``H x`` to ``out`` (interleaved, flat)."""
        if self.empty:
            return
        # The matvec dtype: a narrow (float32) Davidson input meets float64 factors.
        c = x.view(self.m_a, self.m_b).to(self.dtype) * self._sigma
        out_2d = out.view(self.m_a, self.m_b)
        a, b = self._tables["a"], self._tables["b"]
        for factor, sign in zip(self._factors, self._signs):
            x_alpha = self._apply("a", a["sign"] * factor[a["pq"]], c)
            x_alpha_t = x_alpha.T.contiguous()
            del x_alpha
            y_t = self._apply("b", b["sign"] * factor[b["pq"]], x_alpha_t)
            del x_alpha_t
            y_t.mul_(self._sigma_t)
            out_2d.add_(y_t.T, alpha=sign)
            del y_t
