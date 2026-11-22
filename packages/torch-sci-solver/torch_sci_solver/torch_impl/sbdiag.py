"""Top-level driver — torch analog of ``torch_sci_solver.sbd_ref.sbdiag.diag``.

Builds the helpers and integrals on CPU once (cheap, ragged enumeration
isn't worth GPU-vectorizing for the basis sizes SBD targets), then runs
Davidson + matvec entirely on the target device.

The interface mirrors the scalar ``diag()`` so swapping the import line
gets you the GPU version.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from typing import List, Literal, Tuple

import torch
from torch import Tensor

from torch_sci_solver.sbd_ref.integrals import OneInt, TwoInt
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch
from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.matvec_dispatch import MATVEC_IMPLS, build_matvec
from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
from torch_sci_solver.torch_impl.mult_pairs import (
    make_diag_terms_pair_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import basis_init_vector_pair_torch, pair_basis_from_arrays


def _to_int64_python(values):
    """Convert iterable of Python ints in [0, 2^64) to signed int64
    bit pattern (subtract 2^64 if the high bit is set).
    """
    INT64_OFFSET = 1 << 64
    INT64_MAX_UNSIGNED = 1 << 63
    return [int(v) if int(v) < INT64_MAX_UNSIGNED else int(v) - INT64_OFFSET for v in values]


def _resolve_matvec_impl(matvec_impl: str, device: torch.device, mode: str) -> str:
    """Validate ``matvec_impl`` and resolve ``"auto"`` for one basis mode.

    The pair (GDB) driver stays on the portable batched path for ``"auto"``,
    mirroring the solver's mode policy; the product (TPB) driver picks the fused
    Triton kernel on a CUDA device when Triton imports, else batched. A concrete
    name is returned unchanged so ``build_matvec`` never sees ``"auto"``.
    """
    valid_impls = ("auto",) + tuple(MATVEC_IMPLS)
    if matvec_impl not in valid_impls:
        raise ValueError(f"matvec_impl must be one of {valid_impls}, got {matvec_impl!r}")
    if matvec_impl != "auto":
        return matvec_impl
    if mode == "gdb" or device.type != "cuda":
        return "batched"
    try:
        from torch_sci_solver.torch_impl.mult_triton import _HAVE_TRITON
    except Exception:  # pragma: no cover - defensive import guard
        _HAVE_TRITON = False
    return "triton" if _HAVE_TRITON else "batched"


def diag_torch(
    adets: List[int],
    bdets: List[int],
    n_orb: int,
    I0: float,
    I1: OneInt,
    I2: TwoInt,
    *,
    max_davidson_iters: int = 50,
    davidson_block_size: int = 30,
    eps: float = 1e-8,
    de_tol: float | None = None,
    init: int = 0,
    n_roots: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 2_000_000,
    matvec_impl: str = "batched",
    verbose: bool = False,
    mode: Literal["tpb", "gdb"] = "tpb",
    pair_budget_bytes: int | None = None,
    block_matvec: str = "off",
) -> Tuple[float | Tensor, Tensor, dict]:
    """Lowest eigenpairs — torch port of ``diag()``.

    Args:
        adets, bdets:  In TPB mode (``mode="tpb"``, default), sorted α and β
                       bitmask lists (Python ints), each over ``n_orb`` bits,
                       joined into an ``M_α · M_β`` Cartesian product basis.
                       ``adets[0]`` = HF reference.
                       In GDB mode (``mode="gdb"``), interpreted as parallel
                       length-N lists: index ``i`` defines pair
                       ``(adets[i], bdets[i])`` and the basis dimension is
                       ``N`` after dedup by ``(α, β)``. See
                       :mod:`torch_sci_solver.torch_impl.pairs`.
        n_orb:         number of spatial orbitals.
        I0, I1, I2:    integrals from ``setup_integrals_from_pyscf_h1_eri``.
        max_davidson_iters: outer Davidson restart count.
        davidson_block_size:     subspace expansion per restart.
        eps:           residual-norm convergence threshold.
        de_tol:        opt-in eigenvalue-stability check — see
                       :func:`~torch_sci_solver.torch_impl.davidson.davidson_torch`.
                       ``None`` (default) disables it (residual-only). Pass
                       a value to enable it; the effective threshold is
                       ``sqrt(de_tol)`` (recommended: ``de_tol=eps``).
        init:          0 = HF, 1 = random.
        n_roots:       number of lowest eigenpairs to return (default 1).
                       When > 1, returns
                       ``(Tensor(n_roots,), Tensor(n_roots, N), info)``.
        device:        torch device for tensors.
        dtype:         float dtype.
        matvec_impl:   ``"batched"`` (default, portable), ``"triton"`` (fused
                       GPU kernels; every TPB channel except the diagonal is
                       tiled, as are the GDB doubles channels/tables — the
                       block-mode GDB singles and the diagonal stay batched)
                       or ``"auto"``. ``"auto"`` resolves here — to
                       ``"batched"`` on CPU and always in GDB mode, and to
                       ``"triton"`` on a CUDA device when Triton imports — so
                       the product kernel never receives ``"auto"``.
        verbose:       per-iteration prints.
        mode:          See :class:`~torch_sci_solver.torch_impl.qiskit_adapter.TorchSCISolver`.
                       ``"gdb"`` routes to :func:`_diag_torch_pair`, which
                       diagonalizes over the explicit pair basis.
        pair_budget_bytes: GDB only. Peak workspace per excitation-
                       enumeration chunk; ``None`` uses the default. See
                       :func:`~torch_sci_solver.torch_impl.pair_helpers.resolve_budget_bytes`.
        block_matvec:  GDB only. ``"off"`` (default) keeps the streamed pair
                       tables; ``"auto"`` builds the Cartesian block
                       decomposition when its usefulness gates pass; ``"on"``
                       forces it. See
                       :func:`~torch_sci_solver.torch_impl.pair_helpers.generate_pair_helpers`.

    Returns:
        For ``n_roots == 1``: ``(energy, eigvec, info)`` — energy is a
        Python float, eigvec is a ``(N,)`` torch tensor on ``device``.
        For ``n_roots > 1``: ``(energies, eigvecs, info)`` — energies is
        a ``(n_roots,)`` float64 Tensor on CPU, eigvecs is a
        ``(n_roots, N)`` Tensor on ``device``.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if mode not in ("tpb", "gdb"):
        raise ValueError(f'mode must be "tpb" or "gdb", got {mode!r}')
    matvec_impl = _resolve_matvec_impl(matvec_impl, device, mode)
    if mode == "gdb":
        return _diag_torch_pair(
            adets=adets,
            bdets=bdets,
            n_orb=n_orb,
            I0=I0,
            I1=I1,
            I2=I2,
            max_davidson_iters=max_davidson_iters,
            davidson_block_size=davidson_block_size,
            eps=eps,
            de_tol=de_tol,
            init=init,
            n_roots=n_roots,
            device=device,
            dtype=dtype,
            chunk_size=chunk_size,
            matvec_impl=matvec_impl,
            verbose=verbose,
            pair_budget_bytes=pair_budget_bytes,
            block_matvec=block_matvec,
        )

    M_a, M_b = len(adets), len(bdets)
    if verbose:
        print(f"[sbdiag-torch] basis: M_a={M_a}, M_b={M_b}, dim={M_a * M_b}, device={device}")

    # Integrals to device (one-time setup, cost is dwarfed by Davidson)
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)

    # Bitmask tensors (Python ints → signed int64)
    adets_t = torch.tensor(_to_int64_python(adets), dtype=torch.int64, device=device)
    bdets_t = torch.tensor(_to_int64_python(bdets), dtype=torch.int64, device=device)

    # Helpers — built on CPU then shipped (ragged enumeration not vectorizable nicely)
    helpers = list(generate_all_torch_helpers(adets, bdets, n_orb, device=device))
    if verbose:
        # Report how many excitations each task type generates
        n_a_s = helpers[0].a_singles_to.shape[0] if helpers[0].a_singles_to is not None else 0
        n_a_d = helpers[0].a_doubles_to.shape[0] if helpers[0].a_doubles_to is not None else 0
        n_b_s = helpers[1].b_singles_to.shape[0] if helpers[1].b_singles_to is not None else 0
        n_b_d = helpers[1].b_doubles_to.shape[0] if helpers[1].b_doubles_to is not None else 0
        print(f"[sbdiag-torch] α-singles: {n_a_s}, α-doubles: {n_a_d}")
        print(f"[sbdiag-torch] β-singles: {n_b_s}, β-doubles: {n_b_d}")

    # Initial vector
    W = basis_init_vector_torch(M_a, M_b, init=init, device=device, dtype=dtype)

    # Diagonal of H
    hii = make_diag_terms_torch(adets_t, bdets_t, n_orb, integrals, chunk_size=chunk_size)
    if verbose:
        print(f"[sbdiag-torch] diagonal: min={hii.min().item():.10f}, max={hii.max().item():.10f}")
        print(f"[sbdiag-torch]           HF diag (hii[0]) = {hii[0].item():.10f}")

    # Davidson driver
    matvec = build_matvec(
        matvec_impl, hii, adets_t, bdets_t, n_orb, integrals, helpers, chunk_size=chunk_size
    )

    energy, eigvec, info = davidson_torch(
        hii,
        W,
        matvec,
        max_davidson_iters,
        davidson_block_size,
        eps,
        n_roots=n_roots,
        de_tol=de_tol,
        verbose=verbose,
    )

    if verbose:
        print(
            f"[sbdiag-torch] Davidson done: converged={info['converged']}, "
            f"n_matvecs={info['n_matvecs']}, "
            f"final ||R||={info['final_residual_norm']:.3e}"
        )
        if isinstance(energy, Tensor):
            print(f"[sbdiag-torch] Energies = {energy.tolist()}")
        else:
            print(f"[sbdiag-torch] Energy = {energy:.12f}")

    return energy, eigvec, info


def _diag_torch_pair(
    *,
    adets: List[int],
    bdets: List[int],
    n_orb: int,
    I0: float,
    I1: OneInt,
    I2: TwoInt,
    max_davidson_iters: int,
    davidson_block_size: int,
    eps: float,
    init: int,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
    verbose: bool,
    n_roots: int = 1,
    de_tol: float | None = None,
    pair_budget_bytes: int | None = None,
    matvec_impl: str = "batched",
    block_matvec: str = "off",
) -> Tuple[float | Tensor, Tensor, dict]:
    """Pair-mode driver — parallel to ``diag_torch``'s product path.

    ``adets`` and ``bdets`` are interpreted as parallel length-N lists: each
    index ``i`` defines pair ``(adets[i], bdets[i])``. Duplicates are collapsed
    and the basis is sorted lexicographically by ``(α, β)``, so the returned
    eigenvector is indexed by that sorted pair order, not by input order.

    Uses the same multi-root Davidson as the product path.

    Args:
        adets: α-string bitmasks, one per pair (parallel to ``bdets``).
        bdets: β-string bitmasks, one per pair (parallel to ``adets``).
        n_orb: Number of spatial orbitals.
        I0: Scalar (core) energy offset.
        I1: One-electron integrals in spin-orbital form.
        I2: Two-electron integrals in spin-orbital form.
        max_davidson_iters: Davidson outer restart cap.
        davidson_block_size: Davidson subspace size per restart.
        eps: Davidson residual-norm convergence threshold.
        n_roots: Number of lowest eigenpairs; must fit in the deduplicated
            pair basis and the Davidson subspace.
        de_tol: opt-in eigenvalue-stability check — see
            :func:`~torch_sci_solver.torch_impl.davidson.davidson_torch`.
            ``None`` (default) disables it (residual-only). Pass a value to
            enable it; the effective threshold is ``sqrt(de_tol)``
            (recommended: ``de_tol=eps``).
        init: Initial-vector strategy passed to
            :func:`~torch_sci_solver.torch_impl.pairs.basis_init_vector_pair_torch`.
        device: Torch device the basis, integrals and Davidson run on.
        dtype: Torch float dtype (float64 recommended).
        chunk_size: Slater-Condon batch cap per matvec call. Peak memory is
            additionally floored by the pair helpers' own byte budget.
        verbose: Print basis, coupling counts and convergence diagnostics.
            Coupling counts cost one extra enumeration pass, so they are only
            computed when this is set.
        pair_budget_bytes: Peak workspace per enumeration chunk; ``None``
            uses the default (see ``pair_helpers.DEFAULT_BUDGET_BYTES``).
        matvec_impl: Resolved matvec backend. ``"triton"`` routes the doubles
            channels (block mode: same-spin doubles slices; streamed mode: all
            doubles tables) to their Triton kernels and keeps the singles and
            diagonal batched. The streamed entry kernel falls back to its
            batched scatter off CUDA or on an unsupported layout; the block
            same-spin kernel requires Triton on a CUDA device. ``"batched"``
            is the portable path.
        block_matvec: ``"off"`` (default) streams the pair tables; ``"auto"``
            builds the Cartesian block decomposition when its usefulness gates
            pass; ``"on"`` forces it.

    Returns:
        ``(energy, eigvec, info)`` — a float and ``(n_pairs,)`` tensor for
        one root, or ``(n_roots,)`` energies and ``(n_roots, n_pairs)``
        vectors for multiple roots. Vectors follow the sorted pair order;
        ``info`` contains Davidson's convergence diagnostics.
    """
    adets_signed = _to_int64_python(adets)
    bdets_signed = _to_int64_python(bdets)
    pair_basis = pair_basis_from_arrays(adets_signed, bdets_signed, device=device)
    n_pairs = pair_basis.n_pairs
    if verbose:
        print(
            f"[sbdiag-torch] pair mode: n_pairs={n_pairs}, "
            f"unique α={pair_basis.n_unique_alphas}, "
            f"unique β={pair_basis.n_unique_betas}, "
            f"device={device}"
        )

    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    pair_helpers = generate_pair_helpers(
        pair_basis,
        n_orb,
        device=device,
        budget_bytes=pair_budget_bytes,
        block_matvec=block_matvec,
    )
    if verbose:
        # Excitation tables are streamed, not stored, so counting them costs a
        # full enumeration pass — done only under verbose.
        nnz = pair_helpers.nnz()
        print(
            f"[sbdiag-torch] pair couplings: αs={nnz['asingles']}, "
            f"αd={nnz['adoubles']}, "
            f"βs={nnz['bsingles']}, "
            f"βd={nnz['bdoubles']}, "
            f"αβ={nnz['ab_doubles']}"
        )

    hii = make_diag_terms_pair_torch(pair_basis, n_orb, integrals, chunk_size=chunk_size)
    W = basis_init_vector_pair_torch(n_pairs, init=init, device=device, dtype=dtype)

    def matvec(v):
        """Matvec."""
        return mult_pair_torch(
            hii,
            v,
            pair_basis,
            pair_helpers,
            n_orb,
            integrals,
            chunk_size=chunk_size,
            triton_matvec=matvec_impl == "triton",
        )

    energy, eigvec, info = davidson_torch(
        hii,
        W,
        matvec,
        max_davidson_iters,
        davidson_block_size,
        eps,
        n_roots=n_roots,
        de_tol=de_tol,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[sbdiag-torch] pair Davidson: converged={info['converged']}, "
            f"n_matvecs={info['n_matvecs']}, "
            f"final ||R||={info['final_residual_norm']:.3e}"
        )
        if isinstance(energy, Tensor):
            print(f"[sbdiag-torch] pair Energies = {energy.tolist()}")
        else:
            print(f"[sbdiag-torch] pair Energy = {energy:.12f}")
    return energy, eigvec, info
