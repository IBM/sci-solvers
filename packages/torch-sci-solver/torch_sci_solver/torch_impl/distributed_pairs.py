"""Owner-computes GDB (pair-mode) matvec and α↔β-swap projector.

Distributed-memory companion of :func:`torch_sci_solver.torch_impl.mult_pairs.mult_pair_torch`.
The pair basis lists N ``(α, β)`` determinant pairs explicitly, and the
matvec is a scatter over the ``nnz(H)`` excitation rows streamed by
:class:`~torch_sci_solver.torch_impl.pair_helpers.PairHelpers`. Rank ``r``
owns the flat pair range ``[s_r, s_{r+1})`` and computes only the rows whose
bra pair lies in that range.

Replicated input, owned output
------------------------------
The pair tables are streamed precisely because their row count is
``nnz(H)``, which can exceed the basis dimension by orders of magnitude at
the sizes this path targets (see the :mod:`~torch_sci_solver.torch_impl.pair_helpers`
module docstring). Materializing an ``nnz/P`` row table per rank would not
fit either, and exchanging a compact index set would in the worst case ship
nearly the whole vector. So the GDB operator uses the same design as the
spinless operator:

1. one collective all-gather of the local input block into a full ``T_full``
   (:class:`~torch_sci_solver.torch_impl.distributed_davidson.FlatGather`, posted
   asynchronously; the diagonal and the first streamed table chunk are computed
   while it runs),
   i.e. the same length-N vector the replicated fallback already holds, and
   the Davidson blocks — the dominant memory item — stay distributed;
2. each rank streams the five tables restricted to its owned bra pair range
   (``pair_range``) and evaluates/scatters only those matrix elements.

The Slater-Condon work and the output block therefore scale as ``1/P``.
Unlike TPB, no ket halo exchange is needed: every ket amplitude is in the
replicated ``T_full``.

Notes
-----
The pair-space Hamiltonian is

.. math::

    (H x)_k = h_{kk} x_k
      + \\sum_{(k \\leftarrow j) \\in \\mathcal{T}}
        \\langle D_k | H | D_j \\rangle x_j,

over the five streamed tables :math:`\\mathcal{T}` (α/β singles, α/α and
β/β doubles, αβ doubles). Rank ``r`` computes

.. math::

    y_k = (H x)_k, \\qquad k \\in [s_r, s_{r+1}),

so only the rows with :math:`k` in the owned block are enumerated. The
α↔β-swap projector is the element-wise pair formula

.. math::

    P(v)_k = \\frac{1}{2}\\left(v_k + \\tau_k v_{k^{\\mathrm{swap}}}\\right),

which needs the full vector but touches each owned entry once.

Reference: the distributed selected-basis scheme follows arXiv:2605.01138.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time
from typing import Any

import torch
from torch import Tensor

from torch_sci_solver.torch_impl.davidson import SEED_NOISE_SCALE
from torch_sci_solver.torch_impl.distributed_davidson import (
    FlatGather,
    GlobalSum,
    RankBounds,
    _check_local_block,
    _dist,
    hash_uniform,
)
from torch_sci_solver.torch_impl.integrals import TorchIntegrals
from torch_sci_solver.torch_impl.mult_pair_blocks import (
    _MoveCache,
    block_profile,
    profile_enabled,
    reset_block_profile,
)
from torch_sci_solver.torch_impl.mult_pairs import DEFAULT_CHUNK_SIZE, mult_pair_local_torch
from torch_sci_solver.torch_impl.pair_helpers import PairHelpers
from torch_sci_solver.torch_impl.pairs import PairBasis, StructuralPairBasis


def _resolve_identity(
    rank: int | None,
    world_size: int | None,
    group: Any | None,
    bounds: RankBounds,
) -> tuple[int, int]:
    """Resolve rank/world size, defaulting to the bounds' own identity.

    The supplied ``bounds`` fix the partition, so an explicit ``rank`` /
    ``world_size`` may only confirm it. When a process group is initialized it
    must agree with the bounds' world size, since the collectives below run
    over that group.
    """
    dist = _dist()
    initialized = dist.is_available() and dist.is_initialized()
    actual = dist.get_world_size(group) if initialized else 1
    resolved_rank = bounds.rank if rank is None else int(rank)
    resolved_world = bounds.world_size if world_size is None else int(world_size)
    if resolved_rank != bounds.rank or resolved_world != bounds.world_size:
        raise ValueError(
            f"rank/world_size ({resolved_rank}, {resolved_world}) disagree with the "
            f"bounds ({bounds.rank}, {bounds.world_size})"
        )
    if initialized and resolved_world != actual:
        raise ValueError(
            f"bounds world_size={resolved_world} disagrees with the process group's "
            f"world_size={actual}"
        )
    if not initialized and resolved_world > 1:
        raise ValueError(
            "a multi-rank pair operator needs an initialized process group "
            "(torch.distributed.init_process_group)"
        )
    return resolved_rank, resolved_world


class LocalPairOperator:
    """Owner-computes pair-mode matvec over this rank's pair block.

    ``__call__`` is collective: every rank calls it once per matvec, in the
    same order, because the input block is all-gathered into ``T_full``.

    Args:
        hii_local: ``(N_local,)`` diagonal of the owned pair block (from
            :func:`~torch_sci_solver.torch_impl.mult_pairs.make_diag_terms_pair_local`).
        pair_basis: the global ``(α, β)`` pair list.
        pair_helpers: excitation-table streamer for ``pair_basis``.
        n_orb: number of spatial orbitals.
        integrals: the matvec :class:`TorchIntegrals` bundle.
        bounds: this rank's contiguous flat pair block. It fixes the partition;
            ``rank`` / ``world_size`` may only confirm it.
        rank, world_size, group: process-group identity and communicator.
        chunk_size: per-kernel workspace bound.
        triton_matvec: route the same-spin doubles slices to the Triton
            output-stationary kernel; explicit opt-in, real CUDA tensors only.
        move_cache_bytes: byte budget of this operator's move-table cache.
            ``None`` (the default) and ``0`` disable it, so a matvec's peak
            memory is unchanged; a positive value enables it and bounds the
            resident cache bytes. The block kernels key the cache by channel,
            group, spin and uid range, so a Davidson loop reuses every table it
            has room for. A structural basis never caches (its passes are
            fragment kernels with no memo).

    Attributes:
        n: global pair count N.
        N_local: number of pairs owned by this rank.
        bounds: the :class:`RankBounds` block this operator serves.
    """

    def __init__(
        self,
        hii_local: Tensor,
        pair_basis: "PairBasis | StructuralPairBasis",
        pair_helpers: PairHelpers,
        n_orb: int,
        integrals: TorchIntegrals,
        bounds: RankBounds,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        group: Any | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        triton_matvec: bool = False,
        move_cache_bytes: int | None = None,
    ) -> None:
        """Validate the owned block and store the matvec inputs."""
        self.n = pair_basis.n_pairs
        if bounds.n != self.n:
            raise ValueError(f"bounds.n={bounds.n} != pair count {self.n}")
        if hii_local.shape != (bounds.local_size,):
            raise ValueError(f"hii_local shape {tuple(hii_local.shape)} != ({bounds.local_size},)")
        self._rank, self._world_size = _resolve_identity(rank, world_size, group, bounds)
        self.bounds = bounds
        self.N_local = bounds.local_size
        self._pair_basis = pair_basis
        self._pair_helpers = pair_helpers
        self._n_orb = int(n_orb)
        self._integrals = integrals
        self._hii_local = hii_local
        self._chunk_size = int(chunk_size)
        self._triton_matvec = bool(triton_matvec)
        self._group = group
        if move_cache_bytes is None:
            cache_bytes = 0
        else:
            cache_bytes = int(move_cache_bytes)
            if cache_bytes < 0:
                raise ValueError(f"move_cache_bytes must be >= 0 or None, got {move_cache_bytes}")
        # Unlike the tiled operator this does not inherit
        # ``pair_helpers.string_cache_bytes``: the 1-D path's default must stay
        # memory-neutral until the caller opts in. A structural basis has no
        # block enumeration to memoise.
        self._move_cache = (
            _MoveCache(cache_bytes, evict=True)
            if cache_bytes > 0 and pair_helpers.structural is None
            else None
        )
        self._gather = FlatGather(bounds, group=group)

    @property
    def pair_basis(self) -> "PairBasis | StructuralPairBasis":
        """The global ``(α, β)`` pair list."""
        return self._pair_basis

    @property
    def pair_helpers(self) -> PairHelpers:
        """Excitation-table streamer for :attr:`pair_basis`."""
        return self._pair_helpers

    @property
    def n_orb(self) -> int:
        """Number of spatial orbitals."""
        return self._n_orb

    @property
    def chunk_size(self) -> int:
        """Per-kernel workspace bound."""
        return self._chunk_size

    @property
    def group(self) -> Any | None:
        """The process group this operator's collectives use, if any."""
        return self._group

    @property
    def hii_local(self) -> Tensor:
        """``(N_local,)`` diagonal of the owned pair block."""
        return self._hii_local

    def _matvec(
        self,
        x_local: Tensor,
        *,
        T_full: Tensor | None = None,
        T_pending: Any | None = None,
    ) -> Tensor:
        """Apply the owner-computes matvec with a supplied or pending gather."""
        return mult_pair_local_torch(
            self._hii_local,
            x_local,
            T_full,
            self._pair_basis,
            self._pair_helpers,
            self._n_orb,
            self._integrals,
            self.bounds.start,
            chunk_size=self._chunk_size,
            triton_matvec=self._triton_matvec,
            cache=self._move_cache,
            T_pending=T_pending,
        )

    def gather_full(self, x_local: Tensor) -> Tensor:
        """All-gather the owned block into the replicated length-N vector.

        This is collective: every rank must call it in the same order.
        """
        _check_local_block(x_local, self.N_local, "x_local")
        return self._gather.start(x_local).wait()

    def matvec_from_full(self, x_local: Tensor, T_full: Tensor) -> Tensor:
        """Return the owned ``H @ x`` block from an already-gathered vector.

        Args:
            x_local: ``(N_local,)`` owned input block.
            T_full: ``(N,)`` replicated input vector from :meth:`gather_full`.
        """
        _check_local_block(x_local, self.N_local, "x_local")
        if T_full.shape != (self.n,):
            raise ValueError(f"T_full shape {tuple(T_full.shape)} != ({self.n},)")
        return self._matvec(x_local, T_full=T_full)

    def __call__(self, x_local: Tensor) -> Tensor:
        """Return this rank's owned output block ``H @ x``.

        Args:
            x_local: ``(N_local,)`` owned input block.

        Returns:
            ``(N_local,)`` owned output block.
        """
        _check_local_block(x_local, self.N_local, "x_local")
        if not profile_enabled():
            # The gather runs while the diagonal and the first table chunk are
            # computed; the matvec waits on it before the first ket read.
            return self._matvec(x_local, T_pending=self._gather.start(x_local))
        # Diagnostic path (SCIS_PROFILE_BLOCKS=1): the gather and the matvec are
        # timed separately on every rank, and each rank prints its own line so
        # the per-rank log shows the straggler as well as the channel split.
        reset_block_profile()
        _sync = (
            (lambda: torch.cuda.synchronize(x_local.device)) if x_local.is_cuda else (lambda: None)
        )
        _sync()
        t0 = time.perf_counter()
        T_full = self._gather.start(x_local).wait()
        _sync()
        t1 = time.perf_counter()
        y = self._matvec(x_local, T_full=T_full)
        _sync()
        t2 = time.perf_counter()
        prof = block_profile()
        print(
            f"[profile] rank={self._rank}/{self._world_size} n_local={self.N_local} "
            f"gather={t1 - t0:.3f}s matvec={t2 - t1:.3f}s "
            f"ab={prof.get('ab_s', 0.0):.3f}s same_spin={prof.get('same_spin_s', 0.0):.3f}s",
            flush=True,
        )
        return y


class LocalPairSwapProjector:
    """Local-block α↔β-swap projector over this rank's pair block.

    The full-space pair projector is
    ``P(v)[k] = (v[k] + tau[k] * v[k_swap[k]]) / 2``. It needs the full vector
    (the swapped pair of an owned one may live anywhere), so each rank
    all-gathers ``v_local`` and returns its owned slice.

    ``__call__`` is collective: every rank calls it once per projection, in
    the same order, because of the all-gather. Davidson invokes it that way.

    Args:
        pair_basis: the global ``(α, β)`` pair list.
        k_swap: ``(N_local,)`` global pair index of the ``(β, α)`` partner of
            each owned pair only — the full-length map is never held.
        tau_signs: ``(N_local,)`` real ±1 sign ``sigma[k] * sigma[k_swap[k]]``
            for the owned pairs.
        bounds: this rank's contiguous flat pair block. It fixes the partition;
            ``rank`` / ``world_size`` may only confirm it.
        rank, world_size, group: process-group identity and communicator.
    """

    def __init__(
        self,
        pair_basis: "PairBasis | StructuralPairBasis",
        k_swap: Tensor,
        tau_signs: Tensor,
        bounds: RankBounds,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        group: Any | None = None,
    ) -> None:
        """Validate the owned block and store the permutation arrays."""
        self.n = pair_basis.n_pairs
        if bounds.n != self.n:
            raise ValueError(f"bounds.n={bounds.n} != pair count {self.n}")
        if k_swap.shape != (bounds.local_size,):
            raise ValueError(
                f"k_swap shape {tuple(k_swap.shape)} != ({bounds.local_size},) owned block"
            )
        if tau_signs.shape != (bounds.local_size,):
            raise ValueError(
                f"tau_signs shape {tuple(tau_signs.shape)} != ({bounds.local_size},) owned block"
            )
        _resolve_identity(rank, world_size, group, bounds)  # validates the identity
        self.bounds = bounds
        self.N_local = bounds.local_size
        self._k_swap = k_swap
        self._tau_signs = tau_signs
        self._gather = FlatGather(bounds, group=group)

    def __call__(self, v_local: Tensor) -> Tensor:
        """Apply ``P`` to the owned pair block.

        Args:
            v_local: ``(N_local,)`` owned input block.

        Returns:
            ``(N_local,)`` owned block of ``P(v)``.
        """
        _check_local_block(v_local, self.N_local, "v_local")
        v = self._gather.start(v_local).wait()
        k_swap = self._k_swap.to(v.device)
        tau = self._tau_signs.to(device=v.device, dtype=v.dtype)
        owned = v[self.bounds.start : self.bounds.stop]
        return 0.5 * (owned + tau * v[k_swap])


def pair_basis_init_vector_local(
    n_pairs: int,
    bounds: RankBounds,
    *,
    seed_index: int = 0,
    device: torch.device | str,
    dtype: torch.dtype,
    rank: int | None = None,
    world_size: int | None = None,
    group: Any | None = None,
) -> Tensor:
    """Cold-start pair vector, owned block only, with partition-invariant noise.

    Mirrors :func:`~torch_sci_solver.torch_impl.pairs.basis_init_vector_pair_torch`
    (``init=0``) without materializing a length-N noise vector: rank ``r``
    builds

    .. math::

        w_k = [k = \\mathrm{seed}] + \\epsilon \\cdot u(k), \\qquad
        k \\in [s_r, s_{r+1}),

    where :math:`u(k) \\in [-1, 1)` is a counter-based hash of the **global**
    index (:func:`~torch_sci_solver.torch_impl.distributed_davidson.hash_uniform`,
    salt 0) and :math:`\\epsilon` is ``SEED_NOISE_SCALE``. The owned block is
    then divided by the global norm :math:`\\sqrt{\\sum_r \\|w_r\\|^2}` (one
    all-reduce), so every rank sees the same normalization and the assembled
    vector has unit norm. The hash makes the result independent of the rank
    count and partition; the noise differs from the
    single-process ``randn`` stream at the 1e-7 level by design.

    Collective: every rank must call it.

    Args:
        n_pairs: global pair count N.
        bounds: this rank's contiguous flat pair block.
        seed_index: global pair index that receives the unit entry.
        device: device of the returned block.
        dtype: real or complex dtype of the returned block.
        rank, world_size, group: process-group identity and communicator.

    Returns:
        ``(N_local,)`` owned block of the normalized initial vector.
    """
    if bounds.n != int(n_pairs):
        raise ValueError(f"bounds.n={bounds.n} != n_pairs={n_pairs}")
    if not 0 <= int(seed_index) < int(n_pairs):
        raise ValueError(f"seed_index={seed_index} is outside [0, {n_pairs})")
    _resolve_identity(rank, world_size, group, bounds)

    if isinstance(device, str):
        device = torch.device(device)
    index = torch.arange(bounds.start, bounds.stop, dtype=torch.int64, device=device)
    local = SEED_NOISE_SCALE * hash_uniform(index, salt=0).to(dtype)
    if bounds.owns(int(seed_index)):
        local[int(seed_index) - bounds.start] += 1.0

    norm_sq = GlobalSum(group).all_reduce(torch.vdot(local, local).real)
    return local / torch.sqrt(norm_sq.clamp_min(0.0))
