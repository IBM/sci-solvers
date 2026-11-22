"""Pair-mode (no-product-space) S² operator for the spin-penalty Davidson path.

A spin-swap ``a†_pα a†_qβ a_qα a_pβ`` contribution fires only when *both*
the source and target ``(α, β)`` pairs exist in the caller-supplied pair
list. For a given source pair ``(I_a, I_b)`` the only orbitals that can
participate are ``q ∈ alpha_only = I_a & ~I_b`` (α-occupied, β-empty) and
``p ∈ beta_only = I_b & ~I_a`` (β-occupied, α-empty) — see
``s_squared_matrix_torch`` in ``spin_sq.py`` for the scalar reference this
mirrors. That bounds the candidate count per pair by
``popcount(alpha_only) × popcount(beta_only)``, which is generally far
smaller than the unique-string cross product product-mode's swap tables
enumerate, and scales with the pair count N rather than with the size of
the unique-α/β pools.

Everything below — candidate generation, existence lookup, and sign — is
built from batched tensor ops (bit shifts, ``nonzero``, ``repeat_interleave``,
``searchsorted``) with no per-pair or per-candidate Python loop, so the same
code is GPU-efficient and gets CPU vectorization for free.

Existence lookup uses two `searchsorted` calls against ``unique_alphas`` /
``unique_betas`` to resolve target strings to uids, then one more against
``alpha_uid_of_pair * n_unique_betas + beta_uid_of_pair`` — which is already
sorted, because ``PairBasis`` sorts pairs lexicographically by ``(α, β)``
and uid assignment is rank-preserving — to resolve (uid, uid) to a pair
index. No Python dict.

Cost:
  * Diagonal: ``N`` popcounts + a length-N add.
  * Build:    O(N · n_a · n_b) candidate generation + O(log N) per-candidate
              searchsorted lookups, all vectorized.
  * Apply:    O(K) where K is the number of surviving triples.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import torch
from torch import Tensor

from torch_sci_solver.torch_impl.determinants import _occupancy_bits
from torch_sci_solver.torch_impl.pair_helpers import (
    _bit_mask_below_positions,
    _bit_mask_rows,
    _dispatch_found_in,
)
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis
from torch_sci_solver.torch_impl.spin_sq import _popcount_tensor


class PairSpinSqOperator:
    """S² operator restricted to a pair basis.

    Same public interface as :class:`SpinSqOperator` — ``apply(c, out=...)``
    and ``diagonal()`` — so it plugs into the same Davidson penalty
    machinery in :meth:`TorchSCISolver._solve_one_subspace_pair`.

    Args:
        pair_basis: the explicit ``(α, β)`` list.
        n_orb: number of spatial orbitals.
        n_a, n_b: α and β electron counts (used for the ``S_z² − S_z``
            diagonal constant).
        device, dtype: torch placement for the operator tables.
    """

    def __init__(
        self,
        pair_basis: PairBasis,
        n_orb: int,
        n_a: int,
        n_b: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        """Initialize the pair-mode S² operator.

        Raises:
            ValueError: ``pair_basis`` is a structural basis, which stores no
                per-pair arrays; the S² penalty needs them and is therefore
                available only on the explicit-basis path.
        """
        if isinstance(pair_basis, StructuralPairBasis):
            raise ValueError(
                "PairSpinSqOperator needs the explicit per-pair arrays; a structural "
                "pair basis supports only the block matvec path, so the spin_sq "
                "penalty is not available with it."
            )
        if device is None:
            device = pair_basis.device
        self.device = device
        self.dtype = dtype
        self.L = n_orb
        self.n_a = n_a
        self.n_b = n_b
        self.N = pair_basis.n_pairs

        alpha_of_pair = pair_basis.alpha_of_pair.to(device)
        beta_of_pair = pair_basis.beta_of_pair.to(device)

        # --- Diagonal: (n_α − N_dbl(pair)) + (S_z² − S_z) -------------------
        # N_dbl[k] = popcount(alpha_of_pair[k] & beta_of_pair[k]) — the number
        # of orbitals doubly occupied in pair k. Length-N, no outer product.
        sz = 0.5 * (n_a - n_b)
        sz_const = sz * sz - sz
        n_dbl = _popcount_tensor(alpha_of_pair & beta_of_pair)
        if n_dbl.dim() == 2:
            n_dbl = n_dbl.sum(dim=-1)  # sum the per-limb counts
        n_dbl = n_dbl.to(dtype)
        self.diag = ((n_a - n_dbl) + sz_const).contiguous()

        # --- Off-diagonal swap table (pair-restricted, fully vectorized) ----
        self.k_src, self.k_tgt, self.sign = _build_swap_triples(
            alpha_of_pair,
            beta_of_pair,
            pair_basis.unique_alphas.to(device),
            pair_basis.unique_betas.to(device),
            pair_basis.alpha_uid_of_pair.to(device),
            pair_basis.beta_uid_of_pair.to(device),
            n_orb,
            dtype,
        )

    def apply(self, c: Tensor, out: Tensor | None = None) -> Tensor:
        """Compute ``S² @ c`` in pair mode.

        c is interpreted in spin-block convention (same as
        :meth:`SpinSqOperator.apply` in the product-mode module).

        Args:
            c: (N,) input vector.
            out: optional pre-allocated (N,) output buffer.

        Returns:
            (N,) result tensor (``out`` if provided, else newly allocated).
        """
        if out is None:
            output = c * self.diag
        else:
            torch.mul(c, self.diag, out=out)
            output = out
        if self.k_src.shape[0] > 0:
            output.scatter_add_(0, self.k_tgt, self.sign * c[self.k_src])
        return output

    def diagonal(self) -> Tensor:
        """Return the length-N diagonal of S² on the pair basis."""
        return self.diag


def _build_swap_triples(
    alpha_of_pair: Tensor,
    beta_of_pair: Tensor,
    unique_alphas: Tensor,
    unique_betas: Tensor,
    alpha_uid_of_pair: Tensor,
    beta_uid_of_pair: Tensor,
    n_orb: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Enumerate every spin-swap ``(k_src, k_tgt, sign)`` triple, vectorized.

    Per source pair k, candidates are every ``(q, p)`` with q a set bit of
    ``alpha_only = alpha_of_pair[k] & ~beta_of_pair[k]`` and p a set bit of
    ``beta_only = beta_of_pair[k] & ~alpha_of_pair[k]``. Steps:

      1. Extract each pair's candidate bit positions via a dense (N, L)
         boolean broadcast + ``nonzero`` (row-sorted COO).
      2. Expand to the per-pair Cartesian product of (q, p) via the
         standard "ragged join by repeat/cumsum/gather" trick — no Python
         loop, and no candidate ever depends on pool size, only on N and
         each pair's own ``alpha_only``/``beta_only`` popcount.
      3. Compute target strings ``J_a``, ``J_b`` and the Slater-Condon sign
         (mirrors ``_spin_swap_sign`` in spin_sq.py) elementwise.
      4. Resolve existence via ``searchsorted`` (uid, then pair index) and
         mask out candidates whose target isn't in the basis.
    """
    device = alpha_of_pair.device
    N = alpha_of_pair.shape[0]
    n_unique_betas = unique_betas.shape[0]
    empty_i = torch.empty(0, dtype=torch.int64, device=device)
    empty_f = torch.empty(0, dtype=dtype, device=device)

    alpha_only = alpha_of_pair & ~beta_of_pair
    beta_only = beta_of_pair & ~alpha_of_pair
    n_q = _popcount_tensor(alpha_only)  # (N,) candidates per pair, α side
    n_p = _popcount_tensor(beta_only)  # (N,) candidates per pair, β side
    if n_q.dim() == 2:
        n_q = n_q.sum(dim=-1)  # sum the per-limb counts
        n_p = n_p.sum(dim=-1)
    block_sizes = n_q * n_p
    total = int(block_sizes.sum().item())
    if total == 0:
        return empty_i, empty_i, empty_f

    q_rows, q_cols = torch.nonzero(_occupancy_bits(alpha_only, n_orb).bool(), as_tuple=True)
    p_rows, p_cols = torch.nonzero(_occupancy_bits(beta_only, n_orb).bool(), as_tuple=True)
    del q_rows, p_rows  # implied by n_q/n_p row-block structure below

    # Ragged per-pair Cartesian join: for pair k, combos are laid out
    # q-major (n_q[k] blocks of n_p[k] each). `local_idx` is the position
    # within pair k's block; floor-div/mod against n_p recovers the (q, p)
    # sub-indices, which are then used to gather the actual orbital values.
    combo_row = torch.repeat_interleave(torch.arange(N, device=device), block_sizes)
    block_starts = torch.cumsum(block_sizes, 0) - block_sizes
    local_idx = torch.arange(total, device=device) - torch.repeat_interleave(
        block_starts, block_sizes
    )
    n_p_of_combo = n_p[combo_row]
    local_q_idx = local_idx // n_p_of_combo
    local_p_idx = local_idx % n_p_of_combo

    q_row_starts = torch.cumsum(n_q, 0) - n_q
    p_row_starts = torch.cumsum(n_p, 0) - n_p
    q_val = q_cols[q_row_starts[combo_row] + local_q_idx]
    p_val = p_cols[p_row_starts[combo_row] + local_p_idx]

    I_a = alpha_of_pair[combo_row]
    I_b = beta_of_pair[combo_row]
    one = torch.ones((), dtype=torch.int64, device=device)
    if alpha_of_pair.dim() == 1:
        q_bit = one << q_val
        p_bit = one << p_val
    else:
        n_limbs = int(alpha_of_pair.shape[-1])
        q_bit = _bit_mask_rows(q_val, n_limbs)
        p_bit = _bit_mask_rows(p_val, n_limbs)
    J_a = (I_a ^ q_bit) | p_bit
    J_b = (I_b ^ p_bit) | q_bit

    # Sign: _spin_swap_sign(I_a, I_b, p, q), vectorized (see spin_sq.py).
    def below_count(mask: Tensor, k: Tensor) -> Tensor:
        if mask.dim() == 1:
            return _popcount_tensor(mask & ((one << k) - 1))
        n_limbs = int(mask.shape[-1])
        return _popcount_tensor(mask & _bit_mask_below_positions(k, n_limbs)).sum(dim=-1)

    sign_a_p_beta = 1 - 2 * (below_count(I_b, p_val) & 1)
    I_b_after1 = I_b & ~p_bit
    sign_a_q_alpha = 1 - 2 * (below_count(I_a, q_val) & 1)
    I_a_after2 = I_a & ~q_bit
    sign_a_dag_q_beta = 1 - 2 * (below_count(I_b_after1, q_val) & 1)
    sign_a_dag_p_alpha = 1 - 2 * (below_count(I_a_after2, p_val) & 1)
    sign = -1 * sign_a_p_beta * sign_a_q_alpha * sign_a_dag_q_beta * sign_a_dag_p_alpha

    # Existence: J_a -> uid, J_b -> uid, then (uid, uid) -> pair index.
    # alpha_uid_of_pair * n_unique_betas + beta_uid_of_pair is already
    # sorted (PairBasis sorts pairs lexicographically by (α, β), and uid
    # assignment is rank-preserving), so a third searchsorted resolves the
    # pair index directly — no Python dict.
    valid_a, a_uid = _dispatch_found_in(unique_alphas, J_a)
    valid_b, b_uid = _dispatch_found_in(unique_betas, J_b)

    pair_key = alpha_uid_of_pair * n_unique_betas + beta_uid_of_pair
    cand_key = a_uid * n_unique_betas + b_uid
    k_tgt = torch.searchsorted(pair_key, cand_key).clamp(max=N - 1)
    valid_pair = pair_key[k_tgt] == cand_key

    valid = valid_a & valid_b & valid_pair
    return combo_row[valid], k_tgt[valid], sign[valid].to(dtype)
