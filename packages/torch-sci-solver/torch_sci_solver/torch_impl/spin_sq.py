"""S² operator for spin-penalty Davidson (v0.11.0).

Architecture
------------
This module is loaded LAZILY — its imports happen only when the caller
requests a non-None ``spin_sq``. The default ``spin_sq=None`` code path
in ``make_torch_sci_solver`` never touches this file. Per the v0.10.18 →
v0.11.0 migration contract, the spin_sq=None path must be bit-identical
in both runtime and peak memory to v0.10.18 — that contract is enforced
by gating every reference to this module behind ``if spin_sq is not None``.

Operator form
-------------
For our spin-free chemistry Hamiltonian, with fixed (n_α, n_β) sector::

    S² = S_+ S_- + S_z² - S_z
       = N_α + Σ_{p,q} a†_pα a†_qβ a_qα a_pβ  +  (constant in sector)

The (p, q) double-creation operator with this index pattern acts as a
"spin-swap": α moves q → p, β moves p → q. For p = q it reduces to
``-n_pα n_pβ`` (which combined with the N_α term gives the standard
"singly-occupied count" diagonal of S² on a Slater determinant). For
p ≠ q it's a genuine off-diagonal coupling.

Two implementations
-------------------
- ``s_squared_matrix_torch``: builds the (N, N) S² matrix explicitly via
  the determinant comparison rules. O(N²) memory; FOR VALIDATION ONLY
  on small bases (N ≤ a few thousand). Used in the H4 / H2 / O2 tests
  to confirm that ⟨c|S²|c⟩ from this matrix matches the 2-RDM-derived
  value in ``rdm.s_squared_from_rdm``.

- ``s_squared_matvec`` (planned for v0.11.0-rc2): scatter-based fast
  matvec using the orbital-pair iteration described above. O(N · L²)
  work, O(N) memory, suitable for production-scale bases. Will reuse the
  same Slater-Condon sign machinery as ``mult.py`` but with the trivial
  unit "integrals" tensor since S² has structure-1 matrix elements.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.torch_impl.helpers import _as_string_limbs


# Internal: popcount of a masked bitmask, of any width (matches the one in
# qiskit_adapter for the single-word case).
def _popcount(x: int) -> int:
    """Popcount of a non-negative Python-int bitmask (any width)."""
    return x.bit_count()


def _limb_rows_as_ints(limbs: np.ndarray) -> list[int]:
    """Reassemble ``(M, n_limbs)`` uint64 limb rows into Python-int strings."""
    out: list[int] = []
    for row in limbs:
        value = 0
        for k, limb in enumerate(row):
            value |= int(limb) << (64 * k)
        out.append(value)
    return out


def _spin_swap_sign(I_a: int, I_b: int, p: int, q: int) -> int:
    """Sign of ⟨J | a†_pα a†_qβ a_qα a_pβ | I⟩.

    Computes the Jordan-Wigner phase from anticommuting the four
    operators past the determinant ``|I⟩ = a†_α_1 ... a†_α_Nα a†_β_1 ... a†_β_Nβ |0⟩``
    where the orbital indices are in ascending order within each spin
    block.

    Convention: α and β occupations are stored in separate bitmasks
    (I_a, I_b), bit ``k`` = orbital ``k`` occupied. Within each spin
    block the orbitals are ordered low → high; in the full det they
    are α-block first, then β-block.

    Returns +1 or -1; the caller has already verified that the
    operation is non-zero (α-q occupied AND α-p empty AND β-p occupied
    AND β-q empty in I).
    """
    # The four operators in the order applied: a_pβ, a_qα, a†_qβ, a†_pα.
    # Each β-spin operator must anticommute past all α creators on its
    # left in our (α-block, β-block) ordering. a_pβ crosses N_α of them;
    # a†_qβ crosses (N_α - 1) of them (one α electron already removed by
    # a_qα). Combined: (-1)^(N_α) · (-1)^(N_α - 1) = -1, irrespective of
    # N_α. The remaining four "bits below" sign contributions are spin-
    # local and computed below.

    # a_pβ removes β-p. Sign = (-1)^(# β-electrons below p in I_b).
    sign_a_p_beta = (-1) ** _popcount(I_b & ((1 << p) - 1))
    # State now has β-p removed.
    I_b_after_step1 = I_b & ~(1 << p)

    # a_qα removes α-q. Sign = (-1)^(# α-electrons below q in I_a).
    sign_a_q_alpha = (-1) ** _popcount(I_a & ((1 << q) - 1))
    # State now has α-q removed.
    I_a_after_step2 = I_a & ~(1 << q)

    # a†_qβ adds β-q. Sign = (-1)^(# β-electrons below q in current state).
    sign_a_dag_q_beta = (-1) ** _popcount(I_b_after_step1 & ((1 << q) - 1))

    # a†_pα adds α-p. Sign = (-1)^(# α-electrons below p in current state).
    sign_a_dag_p_alpha = (-1) ** _popcount(I_a_after_step2 & ((1 << p) - 1))

    return -1 * (sign_a_p_beta * sign_a_q_alpha * sign_a_dag_q_beta * sign_a_dag_p_alpha)


def s_squared_matrix_torch(
    adets: Tensor,
    bdets: Tensor,
    n_a: int,
    n_b: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Build the explicit (N, N) S² matrix on the determinant basis.

    SLOW — for validation on small systems only. Memory is O(N²·dtype)
    so for a 400M-determinant basis this would need 1.3 EB. Use
    ``s_squared_matvec`` for production runs.

    Convention: rows/columns indexed by ``ia * M_b + ib`` (alpha-outer,
    beta-inner) matching the flat amplitude layout used by Davidson.

    Args:
        adets, bdets: int64 1-D tensors of α / β determinant bitmasks.
        n_a, n_b:     α / β electron counts (used for S_z constants).
        device, dtype: torch placement.

    Returns:
        S2: ``(N, N)`` symmetric torch tensor with N = len(adets) * len(bdets).
    """
    if device is None:
        device = adets.device
    M_a = int(adets.shape[0])
    M_b = int(bdets.shape[0])
    N = M_a * M_b

    # Constant part of the diagonal: (S_z² - S_z) — same for all dets in
    # a fixed sector.
    sz = 0.5 * (n_a - n_b)
    sz_sq_minus_sz = sz * sz - sz

    # Lookup tables: bitmask → index. Used to find J after spin-swap.
    a_idx = {int(adets[i].item()): i for i in range(M_a)}
    b_idx = {int(bdets[i].item()): i for i in range(M_b)}

    S2 = torch.zeros((N, N), dtype=dtype, device=device)

    for ia in range(M_a):
        Ia = int(adets[ia].item())
        for ib in range(M_b):
            Ib = int(bdets[ib].item())
            row = ia * M_b + ib

            # Diagonal: (N_α - N_dbl(I)) + (S_z² - S_z) where N_dbl is
            # the number of orbitals doubly-occupied in I (both α and β set).
            n_dbl = _popcount(Ia & Ib)
            S2[row, row] = (n_a - n_dbl) + sz_sq_minus_sz

            # Off-diagonal: spin-swap on every (p, q) pair with p ≠ q.
            # Effect: J_a = Ia ⊕ {p} ⊕ {q} (after checking α-q in Ia,
            # α-p not in Ia), J_b = Ib ⊕ {p} ⊕ {q} (β-p in Ib, β-q not).
            # Enumerate via the "spin-up-only" (α=1, β=0) and "spin-down-
            # only" (α=0, β=1) orbital sets — these are the only orbitals
            # where the spin-swap can fire.
            alpha_only = Ia & ~Ib  # bits set: α=1, β=0
            beta_only = Ib & ~Ia  # bits set: α=0, β=1

            q_orb = alpha_only
            while q_orb:
                q = (q_orb & -q_orb).bit_length() - 1  # lowest set bit
                q_orb &= q_orb - 1
                p_orb = beta_only
                while p_orb:
                    p = (p_orb & -p_orb).bit_length() - 1
                    p_orb &= p_orb - 1
                    # Build J
                    Ja = (Ia ^ (1 << q)) | (1 << p)
                    Jb = (Ib ^ (1 << p)) | (1 << q)
                    if Ja in a_idx and Jb in b_idx:
                        ja = a_idx[Ja]
                        jb = b_idx[Jb]
                        col = ja * M_b + jb
                        sign = _spin_swap_sign(Ia, Ib, p, q)
                        # NB: the column = ket, the row = bra. Our basis
                        # state at ``col`` is |I⟩, and we computed
                        # ⟨J | S² | I⟩ where J has index ``row``... wait,
                        # let me re-check: row is built from Ia, Ib so
                        # row IS the bra here. Need to swap.
                        # Correction: the bra is J (after spin-swap) and
                        # the ket is I. So we want S2[row_J, col_I].
                        # In our loop, row = ia * M_b + ib = bra-position
                        # for I; the col corresponding to J should be at
                        # ja * M_b + jb. So actually we want S2[col, row] += sign.
                        # But since the operator we're computing IS the bra
                        # transition ⟨J|S²|I⟩, and we have row = I-position,
                        # col = J-position, the matrix element S²[J, I] =
                        # ⟨J|S²|I⟩ goes at S2[col, row].
                        S2[col, row] += sign

    # Ensure symmetric (matches H's Hermiticity).
    S2 = 0.5 * (S2 + S2.T)
    return S2


# ============================================================================
# Fast scatter-based S² matvec (v0.11.0+)
# ============================================================================


class SpinSqOperator:
    """Precomputed S² operator with fast (no N² matrix) matvec.

    Construction is one-time per basis (uses CPU bit-twiddling to enumerate
    spin-swap connections); the resulting GPU tables drive a per-(p, q)
    gather/scatter matvec. Memory is O(M_a · n_a · (L - n_a) + M_b · n_b ·
    (L - n_b)) for the operator tables plus O(N) for the diagonal vector
    — drastically less than the O(N²) reference matrix.

    The α-side and β-side signs in the spin-swap operator
    ``a†_pα a†_qβ a_qα a_pβ`` factorize independently::

        ⟨J | a†_pα a†_qβ a_qα a_pβ | I⟩
          = (-1)  ·  sign_α(I_a, p, q)  ·  sign_β(I_b, p, q)

    (the overall -1 comes from anticommuting the two β operators past N_α
    and (N_α - 1) α-creators respectively). This factorization is what
    makes the precompute-then-scatter approach efficient: each spin's
    table holds (state index, swapped state index, pq-index, ±1 sign)
    tuples, and at matvec time we iterate over the L² (p, q) pairs and
    gather/scatter the α × β outer product into the output.

    The operator is built in *spin-block* convention (no σ flips). The
    caller is responsible for converting between interleaved-Wick and
    spin-block as needed at the matvec boundary (the adapter does this
    via ``_inter_to_block_signs``).

    Diagonal: ``⟨I|S²|I⟩ = (n_α − N_dbl(I)) + (S_z² − S_z)`` where
    ``N_dbl(I) = popcount(I_a & I_b)`` and ``S_z = (n_a - n_b)/2``.
    """

    def __init__(
        self,
        adets: Tensor,
        bdets: Tensor,
        n_orb: int,
        n_a: int,
        n_b: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize the instance."""
        if device is None:
            device = adets.device
        self.device = device
        self.dtype = dtype
        self.L = n_orb
        self.n_a = n_a
        self.n_b = n_b
        self.M_a = int(adets.shape[0])
        self.M_b = int(bdets.shape[0])
        self.N = self.M_a * self.M_b

        adets_cpu = adets.cpu().numpy()
        bdets_cpu = bdets.cpu().numpy()
        if adets_cpu.ndim != bdets_cpu.ndim:
            raise ValueError(
                "adets and bdets must use the same string representation: both 1-D "
                f"bitmasks or both (M, n_limbs) matrices; got ndim={adets_cpu.ndim} and "
                f"ndim={bdets_cpu.ndim}"
            )
        # Normalize to (M, n_limbs) uint64 limb matrices: a single-word input is
        # one limb, so the same bit-twiddling serves n_orb <= 64 and wider bases.
        adets_limbs = _as_string_limbs(adets_cpu, n_orb)
        bdets_limbs = _as_string_limbs(bdets_cpu, n_orb)

        # ---- Diagonal: (n_α − N_dbl) + (S_z² − S_z) ----
        sz = 0.5 * (n_a - n_b)
        sz_const = sz * sz - sz

        # N_dbl[ia, ib] = Σ_limbs popcount(adets[ia, limb] & bdets[ib, limb])
        ab_and = adets.to(device).unsqueeze(1) & bdets.to(device).unsqueeze(0)
        if ab_and.dim() == 3:
            N_dbl = _popcount_tensor(ab_and).sum(dim=-1).to(dtype)
        else:
            N_dbl = _popcount_tensor(ab_and).to(dtype)
        self.diag = ((n_a - N_dbl) + sz_const).reshape(-1).contiguous()

        # ---- α-side and β-side spin-swap tables ----
        # Build on CPU (Python bit ops); transfer to device once.
        self._build_swap_table(adets_limbs, kind="alpha")
        self._build_swap_table(bdets_limbs, kind="beta")

    def _build_swap_table(self, dets_cpu: np.ndarray, kind: str) -> None:
        """Enumerate all valid (i, j, p, q, sign) spin-swap connections.

        ``dets_cpu`` is an ``(M, n_limbs)`` uint64 limb matrix; each row is
        reassembled into a Python int, so the enumeration is width-agnostic
        (one limb for ``n_orb <= 64``, several for wider bases).

        For α: i = source α-string index, j = target α-string index after
        swap, valid iff α-q occupied AND α-p empty AND p ≠ q. For β:
        analogous, but the validity test is β-p occupied AND β-q empty.

        Note the asymmetry — for the same operator
        ``a†_pα a†_qβ a_qα a_pβ`` acting on |I⟩, the α-side needs
        (α-q=1, α-p=0) and the β-side needs (β-p=1, β-q=0). When iterating
        (p, q) pairs at matvec time we use the SAME pq_idx for both
        spins, so the αβ-join is by (p, q) index and the validity
        directions are baked into each side's table.
        """
        L = self.L
        M = int(dets_cpu.shape[0])
        strings = _limb_rows_as_ints(dets_cpu)
        # Basis lookup; the first occurrence wins (the basis is unique).
        index: dict[int, int] = {}
        for i, value in enumerate(strings):
            index.setdefault(value, i)
        # kind == "alpha": α-q occupied (first index in pq is q for α-side)
        # kind == "beta":  β-p occupied (first index in pq is p for β-side)
        # Both tables conceptually iterate over (q, p) pairs:
        #   α-table validity at (q, p): α-q occupied AND α-p empty
        #   β-table validity at (q, p): β-p occupied AND β-q empty   (swapped roles)
        # We adopt the convention that pq_idx = q * L + p uniformly so the
        # α-side and β-side tables join on the same pq_idx at matvec time.

        ia_list = []
        ja_list = []
        pq_list = []
        sign_list = []
        for i in range(M):
            I_bm = strings[i]
            # Enumerate (q, p) pairs based on kind.
            if kind == "alpha":
                occ_orbs = [k for k in range(L) if (I_bm >> k) & 1]  # α-q
                unocc_orbs = [k for k in range(L) if not ((I_bm >> k) & 1)]  # α-p
                q_set, p_set = occ_orbs, unocc_orbs
            else:  # beta: (β-p occupied, β-q empty), so we swap variable roles
                occ_orbs = [k for k in range(L) if (I_bm >> k) & 1]  # β-p
                unocc_orbs = [k for k in range(L) if not ((I_bm >> k) & 1)]  # β-q
                # Iterate "p" over occupied and "q" over empty for β-side
                p_set, q_set = occ_orbs, unocc_orbs

            for q in q_set:
                for p in p_set:
                    if p == q:
                        continue
                    J_bm = I_bm ^ (1 << p) ^ (1 << q)
                    # Lookup J in the basis
                    j = index.get(J_bm)
                    if j is None:
                        continue
                    # Sign for this spin-side: see _spin_swap_sign breakdown.
                    if kind == "alpha":
                        # sign_α = (-1)^(α-below-q) × (-1)^(α-below-p in I-{q})
                        s1 = (-1) ** _popcount(I_bm & ((1 << q) - 1))
                        I_mod = I_bm & ~(1 << q)
                        s2 = (-1) ** _popcount(I_mod & ((1 << p) - 1))
                    else:  # beta
                        # sign_β = (-1)^(β-below-p) × (-1)^(β-below-q in I-{p})
                        s1 = (-1) ** _popcount(I_bm & ((1 << p) - 1))
                        I_mod = I_bm & ~(1 << p)
                        s2 = (-1) ** _popcount(I_mod & ((1 << q) - 1))
                    ia_list.append(i)
                    ja_list.append(j)
                    pq_list.append(q * L + p)
                    sign_list.append(s1 * s2)

        n_pq = L * L
        if len(ia_list) == 0:
            i_t = torch.empty(0, dtype=torch.int64, device=self.device)
            j_t = torch.empty(0, dtype=torch.int64, device=self.device)
            s_t = torch.empty(0, dtype=self.dtype, device=self.device)
            offsets = torch.zeros(n_pq + 1, dtype=torch.int64, device=self.device)
        else:
            pq_np = np.asarray(pq_list, dtype=np.int64)
            sort_idx = np.argsort(pq_np, kind="stable")
            i_arr = np.asarray(ia_list, dtype=np.int64)[sort_idx]
            j_arr = np.asarray(ja_list, dtype=np.int64)[sort_idx]
            pq_arr = pq_np[sort_idx]
            s_arr = np.asarray(sign_list, dtype=np.int8)[sort_idx]
            i_t = torch.from_numpy(i_arr).to(self.device)
            j_t = torch.from_numpy(j_arr).to(self.device)
            s_t = torch.from_numpy(s_arr).to(device=self.device, dtype=self.dtype)
            # offsets[pq] = first index in tables with this pq
            offsets_np = np.searchsorted(pq_arr, np.arange(n_pq + 1, dtype=np.int64))
            offsets = torch.from_numpy(offsets_np).to(self.device)

        if kind == "alpha":
            self.alpha_i = i_t
            self.alpha_j = j_t
            self.alpha_sign = s_t
            self.alpha_offsets = offsets
        else:
            self.beta_i = i_t
            self.beta_j = j_t
            self.beta_sign = s_t
            self.beta_offsets = offsets

    def apply(self, c: Tensor, out: Tensor | None = None) -> Tensor:
        """Compute (S² @ c). c shape (N,), c.device == self.device.

        c is interpreted in spin-block convention (the convention the
        operator was built in). Caller responsible for σ-conversion
        when integrating with interleaved-Wick matvecs.

        Args:
            c:   (N,) input vector.
            out: optional pre-allocated (N,) output buffer to write into.
                 If provided, no new (N,)-sized tensor is allocated by this
                 call. The caller is responsible for ensuring ``out`` is
                 not aliased with ``c``.

        Returns:
            (N,) S² @ c result (either ``out`` if supplied, else newly
            allocated).
        """
        if out is None:
            output = c * self.diag
        else:
            torch.mul(c, self.diag, out=out)
            output = out
        c2d = c.view(self.M_a, self.M_b)
        # Move offsets to CPU once to avoid per-iter sync.
        a_off = self.alpha_offsets.cpu().tolist()
        b_off = self.beta_offsets.cpu().tolist()
        n_pq = self.L * self.L
        for pq in range(n_pq):
            a_lo, a_hi = a_off[pq], a_off[pq + 1]
            b_lo, b_hi = b_off[pq], b_off[pq + 1]
            if a_lo == a_hi or b_lo == b_hi:
                continue
            a_i = self.alpha_i[a_lo:a_hi]  # (n_α,)
            a_j = self.alpha_j[a_lo:a_hi]
            a_s = self.alpha_sign[a_lo:a_hi]
            b_i = self.beta_i[b_lo:b_hi]
            b_j = self.beta_j[b_lo:b_hi]
            b_s = self.beta_sign[b_lo:b_hi]
            # c_sub[α, β] = c2d[a_i[α], b_i[β]]
            c_sub = c2d[a_i[:, None], b_i[None, :]]  # (n_α, n_β)
            # outer product of signs, then -1 prefactor
            contrib = (-1.0) * (a_s[:, None] * b_s[None, :]) * c_sub
            # scatter to output[a_j, b_j]
            flat_dst = (a_j[:, None] * self.M_b + b_j[None, :]).reshape(-1)
            output.scatter_add_(0, flat_dst, contrib.reshape(-1))
        return output

    def diagonal(self) -> Tensor:
        """Return the diagonal of S² in the basis (length N), useful for
        preconditioner augmentation. Note ``self.diag`` is in spin-block
        convention but σ² = 1 so it's also the interleaved-Wick diagonal.
        """
        return self.diag


def _popcount_tensor(x: Tensor) -> Tensor:
    """Vectorised int64 popcount, GPU-friendly. Used for the diagonal."""
    if hasattr(torch, "bitwise_popcount"):
        return torch.bitwise_popcount(x).to(torch.int64)
    from torch_sci_solver.torch_impl.determinants import _popcount64

    return _popcount64(x)
