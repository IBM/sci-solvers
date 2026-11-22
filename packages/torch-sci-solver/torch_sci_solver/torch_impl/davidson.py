"""Torch Davidson — direct port of ``torch_sci_solver.sbd_ref.davidson``.

Almost a literal numpy → torch translation. Same algorithm, same
restart structure, same SBD-traceable line numbers. The differences:

  - All vector storage is torch tensors on the integrals device.
  - ``np.linalg.eigh`` → ``torch.linalg.eigh``, run **on-device**. The
    subspace matrix is tiny (≤ davidson_block_size × davidson_block_size, e.g. 30×30) so
    cuSOLVER syevd is fast, and keeping it on the device avoids a
    host<->device round-trip (``.cpu()`` + ``.to(device)``) every inner
    iteration. That round-trip was a hard CUDA synchronization point that
    serialized the multi-GPU worker threads (see ``qiskit_adapter.py``'s
    ThreadPoolExecutor fan-out) — with each thread blocked on a sync while
    holding the GIL, the GPUs sat idle >90% of the time. Eigenvector-sign
    consistency across eigh backends is **not required** here: every
    consumer of the returned eigenvector is sign-invariant (energy is the
    Ritz value ``E[0]``; ⟨S²⟩ and 1-/2-RDM occupancies are quadratic in ψ),
    and eigh returns eigenvalues ascending on all backends.
  - Host synchronizations in the inner loop are minimized to **two** per
    iteration: the convergence check (``‖R‖ < eps``, optionally combined
    with ``|ΔE| < de_tol`` into one 0-dim bool read — see below), and the
    DGKS reorthogonalization test below (both need a Python bool for
    control flow, both are single 0-dim scalar reads — cheap next to the
    O(ib·N) Gram-Schmidt pass they gate). Norm-normalization guards use
    ``norm.clamp_min(tiny)`` instead of an ``if norm.item() > 0`` sync, and
    the latest ``E[0]`` / ``norm_R`` are kept as 0-dim device tensors and
    materialized to Python floats once at return (or per-iteration when
    ``verbose=True``, a debug path that reintroduces the syncs).
  - Convergence is residual-only (``‖R‖ < eps``) by default, bit-identical
    to ``sbd_ref.davidson``. An **opt-in** ``de_tol`` parameter (``None`` by
    default) additionally requires ``|ΔE| < sqrt(de_tol)`` before declaring
    convergence, matching pyscf's ``linalg_helper.py`` two-condition
    stopping criterion (``abs(de) < tol and dx_norm < toloose``) — this
    guards against a residual that dips below ``eps`` on a wrong/
    excited-state Ritz vector while the eigenvalue is still drifting, but
    only when a caller explicitly asks for it. See ``de_tol`` in
    ``davidson_torch``'s docstring.

History tuples are ``(it, ib, float_E, float_resnorm)`` — same shape as
scalar so downstream consumers don't change.

Multi-root (n_roots > 1)
------------------------
When ``n_roots > 1`` the same single-root loop runs, but at each inner
step all ``n_roots`` Ritz pairs are extracted from the current projected
subspace.  The correction vector is generated from the *unconverged root
with the largest residual norm*, using that root's Ritz value in the
preconditioner.  This ensures the subspace is steered toward whichever
excited state is farthest from convergence, not always root 0.

Restarts retain both Ritz vectors and their Hamiltonian products, using
``H(CU) = (HC)U`` to avoid applying H again to every retained root. After
a restart, convergence is confirmed with fresh products; a failed check
refreshes the retained products and continues. Subspace projections and
two-pass classical Gram-Schmidt use matrix-vector operations instead of
per-vector Python loops.

Return type when ``n_roots > 1``:
    ``(Tensor(n_roots,) on CPU, Tensor(n_roots, N) on device, info_dict)``

When ``n_roots == 1`` (default) the function is bit-identical to the
pre-n_roots code and returns ``(float, Tensor(N,), info_dict)``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

Derived from SBD (https://github.com/r-ccs-cms/sbd), licensed under the Apache
License 2.0: Copyright 2025 RIKEN and its contributors, and the other SBD
copyright holders listed in NOTICE. This file is a modified Python version of
the original C++ code.

SPDX-License-Identifier: Apache-2.0
"""


import math
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor

# DGKS criterion (Daniel-Gragg-Kaufman-Stewart): threshold to trigger a 2nd GS pass
DGKS_ETA = 0.5**2

# Total norm of the perturbation added to a one-hot seed vector, applied to a
# unit-normalized random direction so it does not scale with N. Needs ``hii``
# spread > ``eps / SEED_NOISE_SCALE`` to defeat the false convergence it guards
# against; raise it if a near-degenerate block ever false-converges.
SEED_NOISE_SCALE = 1e-7


def basis_init_vector_torch(
    n_alpha: int,
    n_beta: int,
    init: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
) -> Tensor:
    """Initial wavefunction over the (n_alpha × n_beta) Cartesian basis.

    Direct torch analog of ``sbd_ref.davidson.basis_init_vector``.

    Args:
        n_alpha, n_beta: basis sizes.
        init: 0 = HF (unit vector at index 0, plus a tiny fixed-seed
              perturbation — see below), 1 = random uniform.
        device: torch device.
        dtype: float dtype.
        seed: optional seed for init=1 (uses torch.Generator on the
              given device).
    """
    if isinstance(device, str):
        device = torch.device(device)
    N = n_alpha * n_beta
    W = torch.zeros(N, dtype=dtype, device=device)
    if init == 0:
        W[0] = 1.0
        # A pure one-hot HF guess can already be an exact eigenvector of the
        # local Hamiltonian block. The residual is then exactly 0 at the
        # Davidson step, so the loop thinks wrongly convergence is reached.
        # But the eigenvalue isnt the lowest.  Add tiny amount of noise to mitigate.
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        noise = torch.randn(N, generator=gen, dtype=dtype, device=device)
        noise = noise / torch.linalg.norm(noise).clamp_min(torch.finfo(dtype).tiny)
        W = W + SEED_NOISE_SCALE * noise
        W = W / torch.linalg.norm(W)
    elif init == 1:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        W = torch.randn(N, generator=gen, dtype=dtype, device=device)
        n = torch.linalg.norm(W)
        if n.item() > 0:
            W = W / n
    else:
        raise ValueError(f"init must be 0 (HF) or 1 (random), got {init}")
    return W


def davidson_torch(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    max_davidson_iters: int,
    davidson_block_size: int,
    eps: float,
    *,
    n_roots: int = 1,
    de_tol: Optional[float] = None,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    verbose: bool = False,
    correction: str = "davidson",
    restart: str = "recompute",
    missed_root_check: bool = False,
) -> Union[Tuple[float, Tensor, dict], Tuple[Tensor, Tensor, dict]]:
    """Davidson eigensolver on torch tensors.

    Direct port of ``sbd_ref.davidson.davidson`` for ``n_roots=1``.
    Extended to extract multiple lowest eigenpairs for ``n_roots > 1``.

    The multi-root path uses the same single-root Krylov expansion but
    steers the correction vector toward the unconverged root with the
    largest residual norm at each inner step (rather than always using
    root 0).  This does NOT grow the subspace beyond ``(davidson_block_size, N)``
    — memory cost is unchanged compared to the single-root solver.

    Args:
        hii:           (N,) torch tensor — diagonal of H.
        W:             (N,) torch tensor — initial guess. NOT modified
                       in place; the returned tensor is the eigenvector.
        matvec:        callable v: Tensor → H @ v: Tensor.
        max_davidson_iters: outer restart count. A sub-float64 subspace
                       (float32/complex64) has a residual floor set by its own
                       precision; once that residual has stopped improving (no
                       10% over 15 restarts) the solve stops early and sets
                       ``info["stagnated"]`` instead of exhausting the budget.
        davidson_block_size:     subspace size per restart. Must be >= n_roots. May
                       safely exceed N (= ``hii.shape[0]``).
        eps:           ‖R‖ threshold for convergence (all roots). Unchanged in
                       meaning/value from before ``de_tol`` existed.
        n_roots:       number of lowest eigenpairs to return.
                       Default 1 — backward-compatible single-root mode.
        de_tol:        Opt-in eigenvalue-stability check. ``None`` (default)
                       disables it entirely: convergence is residual-only
                       (``‖R‖ < eps``), bit-identical to ``sbd_ref.davidson``.
                       Pass a value to additionally require that the Ritz
                       value has stopped changing by more than
                       ``sqrt(de_tol)`` (note the sqrt: the *effective*
                       threshold is ``sqrt(de_tol)``, not ``de_tol`` itself)
                       before declaring convergence — mirrors pyscf's
                       ``linalg_helper.py`` two-condition stopping criterion
                       (``abs(de) < tol and dx_norm < toloose``), which guards
                       against a residual that dips below ``eps`` on a
                       transient/wrong Ritz vector while the eigenvalue is
                       still drifting. Recommended value when enabling this:
                       ``de_tol=eps``, giving an effective threshold of
                       ``sqrt(eps)`` — deliberately *looser* than ``eps`` so
                       it only catches gross energy drift and never tightens
                       ordinary convergence. On the very first inner iteration
                       there is no previous energy to compare against, so the
                       de-check (when enabled) is forced false (never
                       converges on iteration 0).
        project_fn:    optional callable v: Tensor → P(v): Tensor that
                       projects onto a symmetry-preserved subspace.
                       When provided, it is applied to:
                         - the initial W,
                         - each ``HC[ib] = matvec(C[ib])`` after matvec,
                         - each ``C[ib+1]`` after Gram-Schmidt.
                       Davidson then renormalizes where appropriate.
                       Used to enforce αβ-swap symmetry against round-off
                       drift on the closed-shell singlet ground state —
                       see ``make_torch_sci_solver`` for context.
        correction:    ``"davidson"`` (default, the existing behaviour):
                       ``delta = (theta - D)^-1 r``. ``"olsen"``: Olsen's
                       correction, ``delta = (theta - D)^-1 (r - e x)`` with
                       ``e`` chosen so that ``delta`` is orthogonal to the
                       Ritz vector ``x`` (see Notes). Both roots paths.
        restart:       what the single-root path keeps across an outer
                       restart. ``"recompute"`` (default, the existing
                       behaviour): the Ritz vector, whose ``H x`` is then
                       applied again (one matvec per restart). ``"reuse"``:
                       the Ritz vector and its ``H x``, formed from the stored
                       products as ``(HC)^T u``, so no matvec is spent.
                       ``"thick"``: ``"reuse"`` plus the previous inner step's
                       Ritz vector, orthogonalized (GD+1, Stathopoulos & Saad
                       1998), whose product is formed the same way. Both
                       non-default modes confirm convergence after a restart
                       with one fresh matvec, as the multi-root path does,
                       since stored products accumulate round-off. Ignored
                       when ``n_roots > 1`` (multi-root already restarts from
                       stored products).
        W:             may also be ``(k, N)`` when ``n_roots > 1``: ``k <=
                       n_roots`` start vectors, one per root (e.g. a previous
                       solve's roots), filling the seed block before the
                       lowest-diagonal unit vectors. ``(N,)`` keeps the
                       existing behaviour.
        missed_root_check: ``n_roots > 1`` only. After convergence, search for an
                       eigenvalue below the highest root returned that the solve
                       missed, and re-solve with it in the seed until none is
                       found (see Notes). A converged multi-root solve only shows
                       that its pairs are eigenpairs, not that they are the lowest:
                       Davidson does not leave the sectors its seed spans, so start
                       vectors that are not the lowest roots of this space (a
                       previous SQD batch, a smaller Ext-SQD/PoolGrow space) can
                       hide a lower root while every residual reads converged.

    Notes
    -----
    Olsen, Jørgensen & Simons, Chem. Phys. Lett. 169, 463 (1990). With
    ``M = theta - D`` (the diagonal preconditioner), Ritz vector ``x`` and
    residual ``r = H x - theta x``, the plain correction ``M^-1 r`` tends to
    ``-x`` as ``D`` approaches ``H``, so Gram-Schmidt discards most of it.
    Olsen subtracts that component:

    .. math::

        \\delta = M^{-1}(r - \\epsilon x),\\qquad
        \\epsilon = \\frac{x^\\dagger M^{-1} r}{x^\\dagger M^{-1} x},

    so that :math:`x^\\dagger \\delta = 0`. It costs two inner products and
    one vector update per step, no extra matvec.

    The missed-root check (as in the warm-start study's ``lobpcg_checked``) runs a
    single-root Davidson on :math:`P H P`, :math:`P` the projector off the converged
    roots (and onto ``project_fn``'s range), from a fixed-seed random vector, which
    overlaps every symmetry sector. It stops at the first Ritz value below
    :math:`\\theta_{\\max} - 10^{-6}`: a missed root, since a Ritz value is an upper
    bound. Otherwise it runs to the solve's own ``eps``; stopping it at a looser
    residual can leave it on a higher state and clear a root that is missing. A search
    that ends undecided counts as a hit. A missed vector joins the seed, ahead of the
    roots found, and the solve is repeated, at most three times.

    Returns:
        For ``n_roots == 1`` (default / backward-compatible):
            ``(energy, eigvec, info_dict)`` — energy is a Python float,
            eigvec is a ``(N,)`` torch tensor on the same device as W.
        For ``n_roots > 1``:
            ``(energies, eigvecs, info_dict)`` — energies is a
            ``(n_roots,)`` float64 Tensor on CPU, eigvecs is a
            ``(n_roots, N)`` Tensor on the same device as W.
        In both cases ``info_dict`` matches the single-root keys plus
        an extra ``"final_residual_norms"`` list when ``n_roots > 1``.
    """
    if correction not in ("davidson", "olsen"):
        raise ValueError(f'correction must be "davidson" or "olsen", got {correction!r}')
    if restart not in ("recompute", "reuse", "thick"):
        raise ValueError(f'restart must be "recompute", "reuse" or "thick", got {restart!r}')
    if n_roots == 1:
        if W.dim() != 1:
            raise ValueError(f"W must be 1-D (N,) when n_roots == 1, got shape {tuple(W.shape)}")
        return _davidson_single(
            hii,
            W,
            matvec,
            max_davidson_iters,
            davidson_block_size,
            eps,
            de_tol=de_tol,
            project_fn=project_fn,
            verbose=verbose,
            correction=correction,
            restart=restart,
        )
    multi = dict(
        n_roots=n_roots,
        de_tol=de_tol,
        project_fn=project_fn,
        verbose=verbose,
        correction=correction,
    )
    energies, vecs, info = _davidson_multi(
        hii, W, matvec, max_davidson_iters, davidson_block_size, eps, **multi
    )
    if not missed_root_check:
        return energies, vecs, info
    n_matvecs, checks = info["n_matvecs"], []
    for attempt in range(4):
        found, v, e_v, mv = _missed_root(
            hii,
            matvec,
            vecs,
            float(energies.max()),
            eps,
            davidson_block_size,
            max_davidson_iters,
            project_fn=project_fn,
            seed=attempt,
        )
        n_matvecs += mv
        checks.append({"found": found, "energy": e_v, "n_matvecs": mv})
        if not found or attempt == 3:
            break
        # The missed vector first (it lies below the highest root), then the roots found.
        seed = torch.cat([v[None].to(vecs.dtype), vecs[: n_roots - 1]])
        energies, vecs, info = _davidson_multi(
            hii, seed, matvec, max_davidson_iters, davidson_block_size, eps, **multi
        )
        n_matvecs += info["n_matvecs"]
    info["n_matvecs"] = n_matvecs
    info["missed_root_checks"] = checks
    return energies, vecs, info


def _orthonormal_span(columns: Tensor) -> Tensor:
    """An orthonormal basis of the span of the ``(N, k)`` ``columns``.

    ``torch.linalg.qr`` everywhere except MPS, where torch's QR of a tall ``(N, k)``
    matrix aborts the process (seen at ``50000 × 10``); there a CholeskyQR2 runs
    instead, with only the ``k × k`` Gram factorization on the CPU.

    Notes
    -----
    CholeskyQR: with ``G = AᴴA = LLᴴ`` (Cholesky, float64 on the CPU),
    ``Q = A L⁻ᴴ`` has ``QᴴQ = I`` in exact arithmetic; a second pass on ``Q``
    restores orthogonality to working precision (CholeskyQR2). The deflation the
    caller builds, ``v - Q Qᴴ v``, depends only on the span, so any orthonormal
    basis of it gives the same result. Columns too close to dependent — a failed
    Cholesky, a tiny last pivot left by the narrow-precision Gram, or a ``Q`` that
    fails the final orthonormality check — fall back to a QR of a CPU copy.
    """
    if columns.device.type != "mps":
        return torch.linalg.qr(columns)[0]
    wide = torch.complex128 if columns.dtype.is_complex else torch.float64
    q = columns
    for _ in range(2):
        gram = (q.conj().T @ q).cpu().to(wide)
        factor, info = torch.linalg.cholesky_ex(0.5 * (gram + gram.conj().T))
        if int(info) != 0:
            return torch.linalg.qr(columns.cpu().to(wide))[0].to(columns)
        # A near-dependent block can still produce a successful Cholesky (the
        # narrow Gram squared the condition number); its tiny last pivot means
        # the resulting direction is noise, so take the CPU QR instead.
        pivot = factor.diagonal().abs()
        if float((pivot.min() / pivot.max().clamp_min(torch.finfo(wide).tiny)).item()) < 1e-6:
            return torch.linalg.qr(columns.cpu().to(wide))[0].to(columns)
        inverse = torch.linalg.solve_triangular(
            factor.conj().T, torch.eye(factor.shape[0], dtype=wide), upper=True
        )
        q = q @ inverse.to(device=q.device, dtype=q.dtype)
    # The Gram was formed in the narrow dtype, so a near-dependent block can
    # pass the Cholesky test and still leave Q non-orthonormal (the condition
    # number is squared before the fp64 factorization); the caller's deflation
    # is an orthogonal projector only if Q is orthonormal. One k×k check, and a
    # near-dependent block falls back to the CPU QR. 1e-3 is ~100x the error
    # seen on random blocks (2e-7 fp32 / 5e-6 complex64).
    check = (q.conj().T @ q).cpu().to(wide)
    if float((check - torch.eye(q.shape[1], dtype=wide)).abs().max()) > 1e-3:
        return torch.linalg.qr(columns.cpu().to(wide))[0].to(columns)
    return q


def _missed_root(
    hii: Tensor,
    matvec: Callable[[Tensor], Tensor],
    roots: Tensor,
    theta_max: float,
    eps: float,
    davidson_block_size: int,
    max_davidson_iters: int,
    *,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    seed: int = 0,
    margin: float = 1e-6,
) -> Tuple[bool, Tensor, float, int]:
    """Search for an eigenvalue below ``theta_max`` orthogonal to ``roots`` (see davidson_torch).

    A cold single-root solve of ``P H P``: its start is the lowest-diagonal determinant not
    already explained by the roots (the cold seed's choice), deflated, plus a fixed-seed random
    component so that every symmetry sector is reached. Started from a random vector alone,
    Davidson can settle on an interior eigenvalue of ``P H P`` above ``theta_max`` and clear a
    missing root below it (N2, projected, 3 roots: -28.1406 found, -28.3242 missed).

    Returns ``(found, vector, energy, n_matvecs)``. ``found`` is also True when the search
    neither drops below ``theta_max - margin`` nor converges (undecided: conservative).
    """
    Q = _orthonormal_span(roots.T)  # (N, k): orthonormal span of the roots

    def deflate(v: Tensor) -> Tensor:
        if project_fn is not None:
            v = project_fn(v)
        return v - Q @ (Q.conj().T @ v)

    real = roots.dtype.to_real() if roots.dtype.is_complex else roots.dtype
    gen = torch.Generator(device=hii.device).manual_seed(seed)
    noise = torch.randn(hii.shape[0], generator=gen, dtype=real, device=hii.device)
    noise = deflate(noise.to(roots.dtype))
    noise = noise / torch.linalg.norm(noise).clamp_min(torch.finfo(real).tiny)
    w = None
    for idx in torch.argsort(hii.real if hii.is_complex() else hii):
        e = torch.zeros_like(noise)
        e[int(idx)] = 1.0
        e = deflate(e)
        n_e = torch.linalg.norm(e)
        if n_e.item() > 0.5:  # mostly a new direction: the determinant, not a leftover
            w = e / n_e
            break
    w = noise if w is None else w + 0.1 * noise
    w = w / torch.linalg.norm(w).clamp_min(torch.finfo(real).tiny)
    energy, vec, info = _davidson_single(
        hii,
        w,
        lambda v: deflate(matvec(deflate(v))),
        max_davidson_iters,
        davidson_block_size,
        eps,
        project_fn=deflate,
        stop_below=theta_max - margin,
    )
    # The dense small-basis path never reads stop_below: compare the energy as well.
    found = bool(info.get("stopped_below")) or energy < theta_max - margin
    found = found or not info["converged"]
    return found, vec, float(energy), int(info["n_matvecs"])


def _olsen(next_vec: Tensor, x: Tensor, denom_safe: Tensor, tiny: float) -> Tensor:
    """Olsen's correction ``M^-1 (r - e x)`` given ``next_vec = M^-1 r`` (see davidson_torch).

    ``e = <x|M^-1 r> / <x|M^-1 x>``, so the result is orthogonal to ``x``. ``vdot``
    conjugates its first argument, which keeps this Hermitian on the complex path.
    The denominator is clamped like every other norm here (no host sync); a zero
    ``<x|M^-1 x>`` (x supported only where ``M`` is huge) leaves the plain correction.
    """
    m_inv_x = x / denom_safe
    num = torch.vdot(x, next_vec)
    den = torch.vdot(x, m_inv_x)
    safe = den.abs() > tiny
    eps_olsen = torch.where(safe, num / torch.where(safe, den, torch.ones_like(den)), 0 * num)
    return next_vec - eps_olsen * m_inv_x


# ---------------------------------------------------------------------------
# Single-root Davidson — original implementation, bit-identical to pre-n_roots
# ---------------------------------------------------------------------------


def _davidson_single(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    max_davidson_iters: int,
    davidson_block_size: int,
    eps: float,
    *,
    de_tol: Optional[float] = None,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    verbose: bool = False,
    correction: str = "davidson",
    restart: str = "recompute",
    stop_below: Optional[float] = None,
) -> Tuple[float, Tensor, dict]:
    """Original single-root Davidson — bit-identical to pre-n_roots code
    when ``de_tol`` is left at its default (``None``), which disables the
    (new) eigenvalue-stability check entirely, and ``correction``/``restart``/
    ``stop_below`` are at their defaults (see ``davidson_torch``).

    ``stop_below`` (internal, for :func:`_missed_root`): stop as soon as the Ritz value
    falls below it, with ``info["stopped_below"] = True``. A Ritz value bounds the lowest
    eigenvalue from above, so that already proves an eigenvalue lies below.
    """
    device = W.device
    dtype = W.dtype
    N = W.shape[0]
    if hii.shape != (N,):
        raise ValueError(f"hii shape {hii.shape} != ({N},)")

    # See davidson_torch's docstring for the two-condition rationale.
    # None => the de-check is disabled outright (residual-only, matching
    # sbd_ref.davidson); a value's *effective* threshold is its sqrt.
    de_tol_eff: Optional[float] = None if de_tol is None else math.sqrt(de_tol)

    eps_reg = 1.0e-12
    # Smallest normal float of ``dtype`` (~2.2e-308 for float64). Used to clamp
    # norm denominators so we never divide by exactly 0.0 without paying a
    # per-iteration host sync: mid-Davidson every norm is far above ``tiny``, so
    # ``x / norm.clamp_min(tiny) == x / norm`` bit-for-bit; the clamp only bites
    # at exact breakdown (norm == 0.0), where it yields 0 instead of a NaN —
    # matching the old ``if norm.item() > 0`` guard's intent without the sync.
    tiny = torch.finfo(dtype).tiny
    # Gram-Schmidt breakdown threshold: correction vector collapsed onto
    # the existing subspace (e.g. davidson_block_size > true dimension).
    breakdown_rtol = torch.finfo(dtype).eps ** 0.5
    # A float32/complex64 basis reorthogonalizes every correction: the error a
    # skipped second pass leaves is ~eps per column and compounds over the
    # block, which at float32 eps ends in a non-orthonormal basis and Ritz
    # values below the spectrum once the residual reaches the precision floor.
    # float64 keeps the conditional DGKS pass.
    always_reorthogonalize = torch.finfo(dtype).bits < 64

    if N <= davidson_block_size:
        # The diagonal preconditioner can stall in an invariant subspace
        # strictly smaller than N on near-degenerate spectra.
        # Since the whole space fits in one block anyway, we skip the Krylov loop
        # and diagonalize directly.
        H_dense = torch.stack([matvec(e) for e in torch.eye(N, dtype=dtype, device=device)], dim=1)
        E, U = torch.linalg.eigh(H_dense)
        W_cur = U[:, 0]
        n_matvecs = N
        if project_fn is not None:
            W_cur = project_fn(W_cur)
            W_cur = W_cur / torch.linalg.norm(W_cur).clamp_min(tiny)
        Hv = matvec(W_cur)
        n_matvecs += 1
        # vdot conjugates its first argument — identical to dot for real
        # tensors (see davidson_torch's module-level rationale), and required
        # for a well-defined Rayleigh quotient ⟨W|H|W⟩ when W is complex.
        # ``E`` is real for a Hermitian operator; ``.real`` drops any
        # roundoff-induced imaginary residue before the Python float() cast
        # (float() on a genuinely complex value would raise TypeError).
        last_E = float(torch.vdot(W_cur, Hv).real.item())
        resid_norm = float(torch.linalg.norm(Hv - last_E * W_cur).item())
        info = {
            "converged": resid_norm < eps,
            "n_outer": 1,
            "n_matvecs": n_matvecs,
            "final_residual_norm": resid_norm,
            "history": [(0, N - 1, last_E, resid_norm)],
        }
        return last_E, W_cur, info

    # Subspace storage. (davidson_block_size, N) — rows are vectors.
    try:
        C = torch.zeros((davidson_block_size, N), dtype=dtype, device=device)
        HC = torch.zeros((davidson_block_size, N), dtype=dtype, device=device)
    except (RuntimeError, MemoryError) as e:
        mem_needed_gb = (2 * davidson_block_size * N * 8) / 1e9
        raise MemoryError(
            f"Failed to allocate Davidson subspace matrices.\n"
            f"  Dimension: {N:,}\n"
            f"  Subspace size (davidson_block_size): {davidson_block_size}\n"
            f"  Memory needed: ~{mem_needed_gb:.1f} GB (for C and HC alone)\n"
            f"  Device: {device}\n"
            f"Original error: {e}"
        ) from e

    H_proj = torch.zeros((davidson_block_size, davidson_block_size), dtype=dtype, device=device)

    history: List = []
    n_matvecs = 0
    converged = False
    # Latest Ritz value / residual norm, kept as 0-dim device tensors and
    # materialized to Python floats exactly once at return — avoids two host
    # syncs per inner iteration. ``None`` until the first inner iteration runs.
    last_E_t: Optional[Tensor] = None
    last_resnorm_t: Optional[Tensor] = None
    # Previous inner iteration's Ritz value, for the |ΔE| < de_tol check.
    # Only advanced on a non-converged step (see the convergence check below)
    # so that on the converged step this still holds the pre-convergence
    # energy rather than being clobbered by the just-computed value.
    prev_E_t: Optional[Tensor] = None

    # Track current W; do not mutate caller's W
    W_cur = W.clone()
    if project_fn is not None:
        # Project initial vector onto the symmetric subspace and renormalize.
        # For the standard W=[1,0,...,0] this is a no-op (single basis element
        # on the diagonal is trivially symmetric), but doing it unconditionally
        # protects against future callers passing asymmetric starting guesses.
        W_cur = project_fn(W_cur)
        nW = torch.linalg.norm(W_cur)
        W_cur = W_cur / nW.clamp_min(tiny)

    # restart="reuse"/"thick": H applied to the current (and, for "thick", the
    # previous inner step's) Ritz vector, formed from the stored products, so a
    # restart re-enters the loop with those columns of HC already filled.
    keep_products = restart != "recompute"
    stopped_below = False
    # See _davidson_multi: a sub-float64 subspace cannot push its residual
    # below its own precision floor, so stop once it has plateaued there
    # instead of burning the full restart budget (this also bounds every
    # missed-root search, which is a single-root solve).
    low_precision = torch.finfo(dtype).bits < 64
    stagnation_window = 15
    stagnation_rtol = 0.9
    best_resnorm = float("inf")
    best_history: List[float] = []
    stagnated = False
    HW_cur: Optional[Tensor] = None
    W_prev: Optional[Tensor] = None
    HW_prev: Optional[Tensor] = None
    n_restart_matvecs_saved = 0

    for it in range(max_davidson_iters):
        C[0] = W_cur
        n_pre = 0  # leading columns of C whose HC is already known
        if it > 0 and keep_products and HW_cur is not None:
            HC[0] = HW_cur
            n_pre = 1
            if restart == "thick" and W_prev is not None and HW_prev is not None:
                # GD+1: the previous Ritz vector, orthogonalized against the current
                # one. H is linear, so its product is the same combination of the
                # stored products; a direction already in C[0] is skipped.
                olap = torch.vdot(W_cur, W_prev)
                w1 = W_prev - olap * W_cur
                hw1 = HW_prev - olap * HW_cur
                n1 = torch.linalg.norm(w1)
                if bool((n1 > breakdown_rtol).item()):
                    C[1] = w1 / n1
                    HC[1] = hw1 / n1
                    n_pre = 2
            n_restart_matvecs_saved += n_pre

        for ib in range(davidson_block_size):
            # ib < n_pre: HC[ib] was carried over from the restart (above), already
            # projected, since it is a combination of projected products.
            if ib >= n_pre:
                HC[ib] = matvec(C[ib])
                n_matvecs += 1
                # Round-off in the chunked-scatter matvec produces an HC[ib] that
                # is only approximately in the symmetric subspace; over many
                # iterations the asymmetric leakage compounds and Davidson chases
                # whichever eigenvalue is lowest in the full space (singlet or
                # triplet). Projecting HC[ib] right after matvec restores exact
                # symmetry and forces Davidson to operate entirely within the
                # symmetric (singlet) sector. No renormalization — HC[ib] doesn't
                # need unit norm; the magnitude is set by H's action on C[ib].
                if project_fn is not None:
                    HC[ib] = project_fn(HC[ib])

            # Project new column of H_proj. vdot conjugates its first
            # argument, making H_proj[jb, ib] = ⟨C[jb]|H|C[ib]⟩ (Hermitian);
            # for real tensors vdot is bit-identical to dot (see module
            # docstring), so this is unconditionally safe on the real path.
            for jb in range(ib + 1):
                v = torch.vdot(C[jb], HC[ib])
                H_proj[jb, ib] = v
                H_proj[ib, jb] = v.conj()

            # Diagonalize the (ib+1) × (ib+1) subspace matrix on-device. It is
            # tiny (≤ davidson_block_size ≤ 30) so cuSOLVER syevd is fast, and — unlike the
            # former ``.cpu()`` round-trip — introduces no host<->device sync,
            # which is what was serializing the multi-GPU worker threads. eigh
            # returns eigenvalues ascending on every backend, so E[0] (lowest)
            # and U[:, 0] (its eigenvector, up to sign) are well-defined; a
            # per-column sign flip is harmless because every consumer of the
            # returned eigenvector is sign-invariant (energy = E[0]; ⟨S²⟩ and
            # occupancies are quadratic in ψ).
            U_sub = H_proj[: ib + 1, : ib + 1]
            E, U = torch.linalg.eigh(U_sub)

            # Ritz vector for lowest eigenvalue: W_new = Σ_kb U[kb, 0] * C[kb]
            # Use matmul (cuBLAS GEMV) to avoid the (ib+1, N) broadcast temporary
            # that ``(U0 * C[:ib+1]).sum(dim=0)`` would materialize. At
            # davidson_block_size=30 and N=50M that temporary is ~12 GB — directly fatal
            # on 40GB GPUs. matmul produces W_new in-place.
            U0_flat = U[:, 0]  # (ib+1,)
            W_new = U0_flat @ C[: ib + 1]  # (N,)
            if keep_products:
                # H x from the stored products (H is linear): kept for the restart.
                HW_new = U0_flat @ HC[: ib + 1]
                R = -HW_new + E[0] * W_new
            else:
                R = -(U0_flat @ HC[: ib + 1])  # (N,)
                R = R + E[0] * W_new

            # Normalize (clamp_min avoids a host sync; see ``tiny`` above).
            norm_W = torch.linalg.norm(W_new)
            W_new = W_new / norm_W.clamp_min(tiny)
            norm_R = torch.linalg.norm(R)
            R = R / norm_R.clamp_min(tiny)

            if keep_products:
                # The previous inner step's Ritz pair, for restart="thick".
                W_prev, HW_prev = W_cur, HW_cur
                HW_cur = HW_new / norm_W.clamp_min(tiny)
            W_cur = W_new
            # Keep the latest scalars on-device; materialize to Python floats
            # only at return (Edit below), or per-iteration when ``verbose``.
            # This removes two host syncs per inner iteration.
            last_E_t = E[0]
            last_resnorm_t = norm_R
            # History keeps its (it, ib, E, ‖R‖) 4-tuple arity; the scalar slots
            # are on-device tensors here and are converted to Python floats once
            # at return, so downstream consumers see identical (int,int,float,
            # float) tuples. No consumer indexes these slots mid-solve.
            history.append((it, ib, last_E_t, last_resnorm_t))
            if verbose:
                # Debug mode only: these .item() calls re-introduce a host sync,
                # acceptable since the production (multi-GPU) path runs verbose=False.
                _e = float(last_E_t.item())
                _r = float(last_resnorm_t.item())
                print(f"  Davidson iter {it}.{ib}: E={_e:.12f}, ||R||={_r:.3e}")

            # The one intentional host sync per inner iteration: Python control
            # flow (break) needs a host bool, so we read a single combined
            # 0-dim bool. ``de_tol_eff is None`` => de-check disabled, purely
            # residual (see de_tol in the davidson_torch docstring).
            # Otherwise: ``prev_E_t is None`` on the very first inner
            # iteration forces this false (nothing to compare against yet),
            # so convergence can never trigger on iteration 0.
            if de_tol_eff is None:
                converged_now = bool((last_resnorm_t < eps).item())
            else:
                if prev_E_t is None:
                    converged_now = False
                else:
                    de_t = torch.abs(E[0] - prev_E_t)
                    converged_now = bool(((last_resnorm_t < eps) & (de_t < de_tol_eff)).item())
                if not converged_now:
                    prev_E_t = E[0].detach().clone()
            if stop_below is not None and bool((E[0] < stop_below).item()):
                stopped_below = True
                break
            if converged_now:
                if it > 0 and keep_products:
                    # Products carried across restarts accumulate round-off: confirm
                    # with a fresh one, as the multi-root path does. On failure,
                    # restart from the refreshed product.
                    fresh = matvec(W_cur)
                    n_matvecs += 1
                    if project_fn is not None:
                        fresh = project_fn(fresh)
                    HW_cur = fresh
                    last_resnorm_t = torch.linalg.norm(fresh - E[0] * W_cur)
                    W_prev = HW_prev = None
                    if not bool((last_resnorm_t < eps).item()):
                        break
                converged = True
                break

            # Preconditioned correction + Gram-Schmidt. Skipped while the next column
            # is one a restart already filled (restart="thick": C[1] and its HC[1]).
            if ib < davidson_block_size - 1 and ib + 1 >= n_pre:
                denom = E[0] - hii
                small = (torch.abs(denom) <= eps_reg).to(dtype)
                denom_safe = denom - eps_reg * small
                next_vec = R / denom_safe
                if correction == "olsen":
                    next_vec = _olsen(next_vec, W_cur, denom_safe, tiny)
                # A correction that collapses is retried as the residual, which is
                # orthogonal to the subspace (Galerkin) and so always moves the Ritz
                # pair; see the same fallback in ``_davidson_multi``. Restarting on a
                # collapsed correction alone would repeat the same step every restart.
                for candidate in (next_vec, R):
                    norm_pre_gs = torch.linalg.norm(candidate).clamp_min(tiny)
                    next_vec = candidate
                    # Gram-Schmidt against C[0..ib]. One pass leaves an
                    # orthogonality error proportional to the cancellation it
                    # performs; because each correction vector is mostly
                    # explained by the existing block, that error compounds ~10x
                    # per basis vector -- so a second pass ("twice is enough",
                    # Kahan/Parlett) is needed, but only when the DGKS criterion
                    # says the first pass actually cancelled enough to risk it.
                    for kb in range(ib + 1):
                        olap = torch.vdot(C[kb], next_vec)
                        next_vec = next_vec - olap * C[kb]
                    if always_reorthogonalize or bool(
                        (torch.linalg.norm(next_vec) < DGKS_ETA * norm_pre_gs).item()
                    ):
                        for kb in range(ib + 1):
                            olap = torch.vdot(C[kb], next_vec)
                            next_vec = next_vec - olap * C[kb]
                    # Project onto symmetric subspace (if requested) before
                    # renormalization. Because C[0..ib] are themselves symmetric,
                    # the projection is orthogonal to the Gram-Schmidt step:
                    # the antisymmetric part of next_vec was already orthogonal
                    # to all of C[0..ib], so we don't have to re-orthogonalize.
                    if project_fn is not None:
                        next_vec = project_fn(next_vec)
                    norm_C = torch.linalg.norm(next_vec)
                    # Breakdown: next_vec is already fully explained by C[0..ib]
                    # (e.g. davidson_block_size exceeds the reachable space). One
                    # 0-dim host read, next to the O(ib N) Gram-Schmidt above.
                    breakdown = bool((norm_C < breakdown_rtol * norm_pre_gs).item())
                    if not breakdown:
                        break
                # A zero column must not be diagonalized: it is an exact Ritz pair with
                # value 0 and residual 0, which a spectrum above 0 (a shifted or
                # positive Hamiltonian) returns as the "converged" ground state.
                # Restart from the current Ritz vector instead.
                if breakdown:
                    break
                C[ib + 1] = next_vec / norm_C.clamp_min(tiny)

        if converged or stopped_below:
            break
        if low_precision and last_resnorm_t is not None:
            current = float(last_resnorm_t.item())
            best_resnorm = min(best_resnorm, current)
            best_history.append(best_resnorm)
            if (
                len(best_history) > stagnation_window
                and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
            ):
                stagnated = True
                break

    # Materialize the deferred scalars exactly once (a single host sync per
    # solve, not per inner iteration). Guard the degenerate zero-iteration run.
    last_E = float(last_E_t.item()) if last_E_t is not None else 0.0
    last_resnorm = float(last_resnorm_t.item()) if last_resnorm_t is not None else float("inf")
    # Convert history's on-device scalar slots to Python floats so the returned
    # tuples are (int, int, float, float) — identical shape/dtype to before.
    history = [(it_i, ib_i, float(e_t), float(r_t)) for (it_i, ib_i, e_t, r_t) in history]
    info = {
        "converged": converged,
        "n_outer": it + 1,
        "n_matvecs": n_matvecs,
        "final_residual_norm": last_resnorm,
        "history": history,
    }
    if keep_products:
        # Restart columns whose product came from storage rather than a matvec.
        info["n_restart_products_reused"] = n_restart_matvecs_saved
    if stop_below is not None:
        info["stopped_below"] = stopped_below
    if stagnated:
        info["stagnated"] = True
    return last_E, W_cur, info


# ---------------------------------------------------------------------------
# Seed block helper (used by multi-root Davidson)
# ---------------------------------------------------------------------------


def _make_seed_block(
    hii: Tensor,
    W: Tensor,
    n_roots: int,
    project_fn: Optional[Callable[[Tensor], Tensor]],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build an (n_roots, N) orthonormal seed block for block Davidson.

    Column 0 is ``W`` (projected and normalised).  Columns 1..n_roots-1 are
    unit vectors at the indices of the ``n_roots-1`` smallest ``hii`` elements
    that are linearly independent of the vectors already accepted (tested via
    Gram-Schmidt deflation).  If the projection kills a candidate we skip it
    and try the next-lowest diagonal element.

    Each unit vector gets the same tiny fixed-seed perturbation that
    ``basis_init_vector_torch`` applies to the ``init=0`` HF guess (column 0,
    via ``W``): a pure one-hot vector can already be an exact eigenvector of
    a near-degenerate Hamiltonian block, giving a residual of exactly 0 with
    no room for Davidson to correct toward the true eigenpair. Without this,
    higher roots seeded here would be exposed to the same stall/false-result
    failure the earlier seed-stall fix addressed for the ground state.

    ``W`` may also be ``(k, N)``: ``k`` start vectors (e.g. a previous solve's
    roots), each projected and Gram-Schmidt deflated against those before it, in
    order; one with nothing left is skipped. Unit vectors then fill the rest.

    Returns an (n_roots, N) tensor; raises if projection leaves too few directions.
    """
    N = W.shape[-1]
    accepted: List[Tensor] = []

    if W.dim() == 2:
        # ---- leading columns: the caller's start vectors, deflated in order ----
        for row in W.to(device=device, dtype=dtype):
            vi = project_fn(row) if project_fn is not None else row.clone()
            ref = torch.linalg.norm(vi)
            for vj in accepted:
                vi = vi - torch.vdot(vj, vi) * vj
            n_vi = torch.linalg.norm(vi)
            if ref.item() > 0 and n_vi.item() > 1e-8 * ref.item():
                accepted.append(vi / n_vi)
            if len(accepted) >= n_roots:
                break
    else:
        # ---- first column: W projected and normalised ----
        v0 = W.clone().to(device=device, dtype=dtype)
        if project_fn is not None:
            v0 = project_fn(v0)
        nv = torch.linalg.norm(v0)
        if nv.item() > 0:
            v0 = v0 / nv
            accepted.append(v0)

    # ---- subsequent columns: unit vectors at sorted-hii positions ----
    if n_roots > 1:
        order = torch.argsort(hii)  # ascending
        for idx in order:
            if len(accepted) >= n_roots:
                break
            i = int(idx.item())
            vi = torch.zeros(N, dtype=dtype, device=device)
            vi[i] = 1.0
            # See docstring: tiny fixed-seed noise so this one-hot vector
            # cannot be an exact eigenvector of the Hamiltonian block. Seeded
            # by ``i`` so each candidate column gets a distinct, deterministic
            # perturbation.
            gen = torch.Generator(device=device)
            gen.manual_seed(i)
            noise = torch.randn(N, generator=gen, dtype=dtype, device=device)
            noise = noise / torch.linalg.norm(noise).clamp_min(torch.finfo(dtype).tiny)
            vi = vi + SEED_NOISE_SCALE * noise
            if project_fn is not None:
                vi = project_fn(vi)
            # Gram-Schmidt deflate against accepted vectors
            for vj in accepted:
                vi = vi - torch.vdot(vj, vi) * vj
            n_vi = torch.linalg.norm(vi)
            if n_vi.item() < 1e-10:
                continue  # degenerate after projection, skip
            accepted.append(vi / n_vi)

    if len(accepted) < n_roots:
        raise ValueError(
            f"n_roots ({n_roots}) exceeds the available projected CI dimension ({len(accepted)})."
        )
    return torch.stack(accepted, dim=0)  # (n_roots, N)


# ---------------------------------------------------------------------------
# Multi-root Davidson (n_roots > 1)
# ---------------------------------------------------------------------------


def _davidson_multi(
    hii: Tensor,
    W: Tensor,
    matvec: Callable[[Tensor], Tensor],
    max_davidson_iters: int,
    davidson_block_size: int,
    eps: float,
    *,
    n_roots: int,
    de_tol: Optional[float] = None,
    project_fn: Optional[Callable[[Tensor], Tensor]] = None,
    verbose: bool = False,
    correction: str = "davidson",
) -> Tuple[Tensor, Tensor, dict]:
    """Multi-root Davidson: find the ``n_roots`` lowest eigenpairs.

    Uses the same single-root Krylov expansion loop as
    ``_davidson_single``.  At each inner step, all ``n_roots`` Ritz
    pairs are extracted and the correction vector is aimed at the
    unconverged root with the largest residual norm.  This keeps the
    subspace cost identical to the single-root case — ``(davidson_block_size, N)``
    — while steering the expansion toward whatever root needs it most.(like PySCF)

    Args:
        n_roots: must satisfy ``1 < n_roots <= davidson_block_size`` and fit
            in the CI space (or the projected sector when ``project_fn`` is set).

    Returns:
        (energies, eigvecs, info) — energies: (n_roots,) on CPU,
        eigvecs: (n_roots, N) on device.
    """
    if n_roots > davidson_block_size:
        raise ValueError(
            f"n_roots ({n_roots}) cannot exceed davidson_block_size ({davidson_block_size}). "
            f"Increase davidson_block_size or reduce n_roots."
        )

    ratio = davidson_block_size / n_roots
    if ratio < 2.0:
        warnings.warn(
            f"Subspace size (davidson_block_size={davidson_block_size}) is extremely small relative to "
            f"the number of requested roots (n_roots={n_roots}). This ratio ({ratio:.1f}) "
            f"leaves very little room for Krylov expansion and will likely cause "
            f"stagnation or severely degraded convergence. "
            f"A ratio of at least 2.0 (preferably 3.0 or 4.0) is strongly recommended.",
            RuntimeWarning,
            stacklevel=2,
        )

    device = W.device
    dtype = W.dtype
    N = W.shape[-1]
    if W.dim() == 2 and W.shape[0] > n_roots:
        raise ValueError(f"W has {W.shape[0]} start vectors, more than n_roots ({n_roots}).")
    if hii.shape != (N,):
        raise ValueError(f"hii shape {hii.shape} != ({N},)")
    if n_roots > N:
        raise ValueError(f"n_roots ({n_roots}) cannot exceed the CI dimension ({N}).")

    eps_reg = 1.0e-7
    tiny = torch.finfo(dtype).tiny
    # See ``_davidson_single`` for why this guard exists.
    breakdown_rtol = torch.finfo(dtype).eps ** 0.5
    # See davidson_torch's docstring for the two-condition rationale.
    # None => the de-check is disabled outright (residual-only, matching
    # sbd_ref.davidson); a value's *effective* threshold is its sqrt.
    de_tol_eff: Optional[float] = None if de_tol is None else math.sqrt(de_tol)

    if N <= davidson_block_size:
        # Mirror of the _davidson_single escape: when the whole CI space fits
        # in one block the diagonal preconditioner can stall deterministically
        # (e.g. at N=16). Diagonalize directly and return all n_roots pairs.
        H_dense = torch.stack([matvec(e) for e in torch.eye(N, dtype=dtype, device=device)], dim=1)
        basis = None
        if project_fn is not None:
            # Solve inside the projector's range. Projecting full-space
            # eigenvectors afterwards can erase a root yet report zero residual.
            projector = torch.stack(
                [project_fn(e) for e in torch.eye(N, dtype=dtype, device=device)], dim=1
            )
            p_values, p_vectors = torch.linalg.eigh(projector)
            basis = p_vectors[:, p_values > 0.5]  # (N, projected_dimension)
            if n_roots > basis.shape[1]:
                raise ValueError(
                    f"n_roots ({n_roots}) exceeds the projected CI dimension ({basis.shape[1]})."
                )
            H_dense = basis.conj().T @ H_dense @ basis
        E_all, U_all = torch.linalg.eigh(H_dense)
        if basis is not None:
            U_all = basis @ U_all
        n_avail = n_roots
        vecs = U_all[:, :n_avail].T.contiguous()  # (n_roots, N)
        # Compute residual norms for info. E_all is already real (Hermitian
        # eigenvalues); keep it real rather than casting to a complex dtype.
        Hvecs = torch.stack([matvec(v) for v in vecs])  # (n_avail, N)
        if project_fn is not None:
            Hvecs = torch.stack([project_fn(v) for v in Hvecs])
        E_avail = E_all[:n_avail]
        resids = Hvecs - E_avail.unsqueeze(1) * vecs
        resnorms = torch.linalg.norm(resids, dim=1)  # (n_avail,)
        energies_out = E_all[:n_roots].cpu().to(dtype=torch.float64)
        resnorms_list = resnorms.cpu().tolist()
        max_resnorm = float(resnorms.max().item())
        info = {
            "converged": max_resnorm < eps,
            "n_outer": 1,
            "n_matvecs": N + n_avail,
            "final_residual_norm": max_resnorm,
            "final_residual_norms": resnorms_list,
            "history": [(0, N - 1, energies_out[:n_avail].tolist(), resnorms_list)],
        }
        return energies_out, vecs, info

    try:
        C = torch.zeros((davidson_block_size, N), dtype=dtype, device=device)
        HC = torch.zeros((davidson_block_size, N), dtype=dtype, device=device)
    except (RuntimeError, MemoryError) as e:
        mem_needed_gb = (2 * davidson_block_size * N * 8) / 1e9
        raise MemoryError(
            f"Failed to allocate Davidson subspace matrices.\n"
            f"  Dimension: {N:,}\n"
            f"  Subspace size (davidson_block_size): {davidson_block_size}\n"
            f"  Memory needed: ~{mem_needed_gb:.1f} GB (for C and HC alone)\n"
            f"  Device: {device}\n"
            f"Original error: {e}"
        ) from e

    H_proj = torch.zeros((davidson_block_size, davidson_block_size), dtype=dtype, device=device)

    history: List = []
    n_matvecs = 0
    converged = False
    # A sub-float64 subspace has a residual floor set by its own precision
    # (float32/complex64: ~sqrt(nnz)·‖H‖·eps — measured ~5e-4 at N=54k and
    # |E|≈260 Ha). Once the worst root's residual stops improving there, the
    # solve cannot reach ``eps`` and would burn every remaining restart (plus
    # the missed-root check's solves on top). Stop after a window of restarts
    # with no 10% improvement; full-precision subspaces are untouched.
    low_precision = torch.finfo(dtype).bits < 64
    stagnation_window = 15
    stagnation_rtol = 0.9
    best_resnorm = float("inf")
    best_history: List[float] = []
    stagnated = False

    # Previous inner iteration's per-root energies, for the |ΔE| < de_tol
    # check (see davidson_torch's docstring). ``has_prev`` starts False so
    # the very first inner iteration can never satisfy the de-check; both
    # are only advanced on a non-converged step, so on the converged step
    # ``prev_energies`` still holds the pre-convergence energies rather than
    # being overwritten by the just-computed converged values.
    has_prev = False
    prev_energies = torch.zeros(n_roots, dtype=torch.float64)  # CPU

    last_energies = torch.zeros(n_roots, dtype=torch.float64)  # CPU
    last_resnorms = torch.full((n_roots,), float("inf"))  # CPU

    # --- Build the initial seed block: n_roots orthonormal vectors.
    # C[0] = W (projected), C[1..n_roots-1] = unit vectors at the n_roots-1
    # lowest-diagonal elements NOT already covered by C[0].
    # This ensures the subspace starts spanning all n_roots directions from
    # the very first inner step, rather than having to grow into them.
    seed_block = _make_seed_block(hii, W, n_roots, project_fn, dtype, device)

    # Thick restart: keep one buffer Ritz vector beyond the n_roots targets. With
    # only the targets kept, a target can converge onto an excited state that a
    # seed happened to start on (the next-lowest diagonal determinant), while the
    # true lower root, reachable only through the restart-discarded rest of the
    # subspace, is lost for good; every residual is then small and the solve
    # reports converged on the wrong root (H6/STO-3G: E4 instead of E3). The
    # buffer keeps that direction alive so it can rotate into the targets. It is
    # never a convergence target and never gets a correction, so its cost is one
    # restart slot. Only added when a restart still leaves a slot for a new
    # direction; never below n_roots, which would drop a target's Ritz vector.
    n_keep = n_roots + 1 if davidson_block_size >= n_roots + 2 else n_roots

    # Will be set on the final inner step of the last outer iteration
    final_U: Optional[Tensor] = None
    final_ib: int = 0
    # Tracks all n_roots current best Ritz vectors (for multi-vector restart)
    ritz_restart: Optional[Tensor] = None  # (n_roots, N)
    ritz_Hrestart: Optional[Tensor] = None  # (n_roots, N), H applied to retained roots

    for it in range(max_davidson_iters):
        # On the first outer iteration seed from the diagonal-based block;
        # on restarts seed from the n_roots current best Ritz vectors so
        # the second (and higher) roots retain their Krylov information.
        if it == 0 or ritz_restart is None:
            seed_size = seed_block.shape[0]  # may be < n_roots for tiny bases
            C[:seed_size] = seed_block
            sub_start = seed_size  # first empty slot after seeds
        else:
            # Restart: pre-fill with the kept Ritz vectors (targets + buffer); they
            # are already mutually orthonormal (unit-norm from normalisation below).
            sub_start = ritz_restart.shape[0]
            C[:sub_start] = ritz_restart

        # H is linear: retain H*ritz alongside ritz instead of repeating the
        # expensive matvecs for every root at every restart.
        H_proj.zero_()
        if ritz_Hrestart is None:
            for r in range(sub_start):
                HC[r] = matvec(C[r])
                n_matvecs += 1
                if project_fn is not None:
                    HC[r] = project_fn(HC[r])
        else:
            HC[:sub_start] = ritz_Hrestart
        # Conjugate the bra vectors to preserve complex Hermitian projection.
        seed_projection = C[:sub_start].conj() @ HC[:sub_start].T
        H_proj[:sub_start, :sub_start] = (
            torch.triu(seed_projection) + torch.triu(seed_projection, diagonal=1).mH
        )

        for ib in range(sub_start - 1, davidson_block_size):
            # --- Diagonalize current subspace ---
            sub_size = ib + 1
            U_sub = H_proj[:sub_size, :sub_size].clone().cpu()
            try:
                E_cpu, U_cpu = torch.linalg.eigh(U_sub)
            except torch.linalg.LinAlgError:
                # A near-degenerate fp32 subspace can make LAPACK's eigh fail to
                # converge on some CPUs (seen on one A100 node, not reproducible
                # on others). The matrix is at most davidson_block_size wide, so
                # redo it in double precision; the results are cast back below.
                wide = torch.complex128 if U_sub.is_complex() else torch.float64
                E_cpu, U_cpu = torch.linalg.eigh(U_sub.to(wide))
            # E is real (Hermitian eigenvalues) even when H_proj/dtype is
            # complex; keep it real rather than casting to a complex dtype —
            # E is only ever used as a real Rayleigh-quotient/preconditioner
            # scalar below.
            # Real part of the working dtype: a no-op normally, and it casts the
            # double-precision fallback's eigenvalues back, so they cannot promote
            # the corrections below out of the subspace's dtype.
            E = E_cpu.to(device=device, dtype=dtype.to_real() if dtype.is_complex else dtype)
            U = U_cpu.to(device=device, dtype=dtype)

            # How many roots can we extract from a subspace of size sub_size?
            n_avail = min(n_roots, sub_size)
            # Ritz vectors kept for the restart: the n_roots targets plus buffer
            # roots (see ``n_keep`` above). Only the first n_avail are targets.
            n_kept = min(n_keep, sub_size)

            # --- Ritz vectors for the targets and the buffer ---
            ritz_vecs = U[:, :n_kept].T @ C[:sub_size]  # (n_kept, N)
            ritz_Hvecs = U[:, :n_kept].T @ HC[:sub_size]  # (n_kept, N)

            # Normalize Ritz vectors
            norms_v = torch.linalg.norm(ritz_vecs, dim=1, keepdim=True).clamp(min=1e-30)
            ritz_vecs = ritz_vecs / norms_v
            ritz_Hvecs = ritz_Hvecs / norms_v

            # Residuals of the targets only: R[k] = H*ritz[k] - E[k]*ritz[k]
            residuals = ritz_Hvecs[:n_avail] - E[:n_avail].unsqueeze(1) * ritz_vecs[:n_avail]

            # Residual norms
            resnorms_dev = torch.linalg.norm(residuals, dim=1)  # (n_avail,) on device

            last_energies[:n_avail] = E_cpu[:n_avail].to(dtype=torch.float64)
            last_resnorms[:n_avail] = resnorms_dev.cpu()
            final_U = U
            final_ib = ib
            # Store the targets and the buffer for restart on the next outer iteration
            ritz_restart = ritz_vecs.detach()  # (n_kept, N), independent of C storage
            ritz_Hrestart = ritz_Hvecs.detach()

            if verbose:
                for k in range(n_avail):
                    print(
                        f"  Davidson[multi] iter {it}.{ib} root {k}: "
                        f"E={float(E[k].item()):.12f}, "
                        f"||R||={float(resnorms_dev[k].item()):.3e}"
                    )
            history.append((
                it,
                ib,
                last_energies[:n_avail].tolist(),
                last_resnorms[:n_avail].tolist(),
            ))

            # Converged when all n_roots roots are available and tight on
            # residual. If de_tol_eff is not None (opt-in), additionally
            # require (pyscf-style) that each Ritz value has stopped
            # changing by more than de_tol_eff — see davidson_torch's
            # docstring. ``has_prev`` is False on the very first inner
            # iteration, so the de-check (when enabled) can never trigger
            # there.
            resid_ok = n_avail == n_roots and resnorms_dev.max().item() < eps
            if de_tol_eff is None:
                converged_now = resid_ok
            else:
                de_ok = has_prev and bool(
                    (
                        torch.abs(last_energies[:n_avail] - prev_energies[:n_avail]) < de_tol_eff
                    ).all()
                )
                converged_now = resid_ok and de_ok
                if not converged_now:
                    prev_energies[:n_avail] = last_energies[:n_avail]
                    has_prev = True
            if converged_now:
                if it > 0:
                    # Cached products accumulate roundoff across restarts. Check
                    # the actual operator before declaring convergence; if it
                    # disagrees, restart from refreshed products.
                    targets = ritz_restart[:n_avail]
                    checked_H = torch.stack([matvec(v) for v in targets])
                    n_matvecs += n_avail
                    if project_fn is not None:
                        checked_H = torch.stack([project_fn(v) for v in checked_H])
                    checked_norms = torch.linalg.norm(
                        checked_H - E[:n_avail, None] * targets, dim=1
                    )
                    # Refresh only the targets' cached products; the buffer keeps
                    # its own (it is not a convergence target).
                    ritz_Hrestart = torch.cat([checked_H, ritz_Hrestart[n_avail:]])
                    last_resnorms[:n_avail] = checked_norms.cpu()
                    if checked_norms.max().item() >= eps:
                        break
                converged = True
                break

            # --- Preconditioned correction for the most-needy unconverged root ---
            if ib < davidson_block_size - 1:
                # Pick the unconverged root with the largest residual norm.
                resnorms_cpu = resnorms_dev.cpu()
                unconverged_mask = resnorms_cpu >= eps
                if unconverged_mask.any():
                    masked_norms = resnorms_cpu.clone()
                    masked_norms[~unconverged_mask] = -1.0
                    k_target = int(masked_norms.argmax().item())
                else:
                    k_target = 0  # all available converged; keep expanding

                R_target = residuals[k_target]
                E_target = E[k_target]

                norm_R = torch.linalg.norm(R_target)
                if norm_R.item() > 0:
                    R_target = R_target / norm_R

                denom = E_target - hii
                small = (torch.abs(denom) <= eps_reg).to(dtype)
                denom_safe = denom - eps_reg * small
                next_vec = R_target / denom_safe
                if correction == "olsen":
                    next_vec = _olsen(next_vec, ritz_vecs[k_target], denom_safe, tiny)
                # Two-pass classical Gram-Schmidt: GEMVs replace one launch
                # per basis vector, and reorthogonalization retains accuracy
                # when the correction nearly lies in the existing subspace.
                # If the correction collapses, the residual is tried instead. Restarting
                # on a collapsed correction alone can cycle: the restart drops the
                # direction the previous correction added, the next correction re-adds
                # it, and the one after collapses again, with every Ritz pair unchanged
                # until max_davidson_iters (H6 2-root fp32 solve on an x86 MKL host: 57
                # identical restarts, root 0 returned 7.6 mHa high). The residual is
                # orthogonal to the subspace (Galerkin), so it always moves the pair.
                basis = C[:sub_size]
                for candidate in (next_vec, R_target):
                    norm_pre_gs = torch.linalg.norm(candidate).clamp_min(tiny)
                    next_vec = candidate - (basis.conj() @ candidate) @ basis
                    next_vec = next_vec - (basis.conj() @ next_vec) @ basis
                    if project_fn is not None:
                        next_vec = project_fn(next_vec)
                    norm_C = torch.linalg.norm(next_vec)
                    breakdown = bool((norm_C < breakdown_rtol * norm_pre_gs).item())
                    if not breakdown:
                        break
                if breakdown:
                    # Both lie in the subspace already. Do not diagonalize a zero
                    # column: it is an exact Ritz pair with value 0, which is below
                    # every root of a Hamiltonian whose spectrum lies above 0 (a
                    # shifted or positive one) and then becomes the restart vector,
                    # whose residual grows without bound. Restart from the current
                    # Ritz vectors instead, which keeps every target and the buffer.
                    break
                C[ib + 1] = next_vec / norm_C.clamp_min(tiny)
                HC[ib + 1] = matvec(C[ib + 1])
                n_matvecs += 1
                if project_fn is not None:
                    HC[ib + 1] = project_fn(HC[ib + 1])
                new_col = ib + 1
                column = C[: new_col + 1].conj() @ HC[new_col]
                H_proj[: new_col + 1, new_col] = column
                H_proj[new_col, : new_col + 1] = column.conj()

        if converged:
            break
        if low_precision:
            current = float(last_resnorms[:n_roots].max().item())
            best_resnorm = min(best_resnorm, current)
            best_history.append(best_resnorm)
            if (
                len(best_history) > stagnation_window
                and best_history[-1] > stagnation_rtol * best_history[-stagnation_window - 1]
            ):
                stagnated = True
                break

    # --- Final Ritz extraction ---
    # Use the last valid (U, ib) to extract n_roots eigenpairs.
    # If we ran out of subspace before seeing n_roots, we return what we have.
    assert final_U is not None, "Davidson loop did not execute even once"
    assert ritz_Hrestart is not None
    n_final = min(n_roots, final_ib + 1)
    final_vecs = final_U[:, :n_final].T @ C[: final_ib + 1]  # (n_final, N)
    norms_f = torch.linalg.norm(final_vecs, dim=1, keepdim=True).clamp(min=1e-30)
    final_vecs = final_vecs / norms_f

    if n_final < n_roots:
        # Subspace too small — pad with zeros and warn.  This only happens
        # when davidson_block_size < n_roots, which is blocked by the guard at the top,
        # or in a degenerate 1-element basis.
        pad = torch.zeros((n_roots - n_final, N), dtype=dtype, device=device)
        final_vecs = torch.cat([final_vecs, pad], dim=0)

    # Build final residual norms for info dict. E_final stays real (see the
    # E-dtype note earlier in this function) — real x complex broadcasts fine.
    # last_energies is CPU float64; recast to dtype's real component before
    # moving on-device (MPS has no float64 support).
    final_Hvecs = ritz_Hrestart[:n_final]
    e_final_dtype = dtype.to_real() if dtype.is_complex else dtype
    E_final = last_energies[:n_final].to(device=device, dtype=e_final_dtype)
    final_resid = final_Hvecs - E_final.unsqueeze(1) * final_vecs[:n_final]
    final_resnorms_list = torch.linalg.norm(final_resid, dim=1).cpu().tolist()

    info = {
        "converged": converged,
        "n_outer": it + 1,
        "n_matvecs": n_matvecs,
        # Keep singular key for compatibility with perf-log readers that
        # already use info.get("final_residual_norm").
        "final_residual_norm": float(last_resnorms[:n_roots].max().item()),
        # Extra plural key for multi-root callers.
        "final_residual_norms": final_resnorms_list,
        "history": history,
    }
    if stagnated:
        info["stagnated"] = True
    return last_energies[:n_roots].to(dtype=torch.float64), final_vecs, info
