"""Tests for n_roots (multi-root Davidson) support.

Verifies (numbered in file order — matches the ``# N.`` section headers below):
  1. n_roots=1 (default) is bit-identical to the pre-n_roots single-root path.
  2. n_roots=2 on a synthetic dense matrix returns both lowest eigenvalues
     matching numpy's eigvalsh.
  3. n_roots > davidson_block_size raises ValueError.
  4. davidson_block_size exceeding the true CI space dimension N (near-duplicate
     determinants collapsing the space, including a case built from real
     benzoic acid shots) does not corrupt the ground energy.
  5. de_tol is opt-in (None by default = residual-only, matching
     sbd_ref.davidson). When explicitly passed, it withholds convergence on
     the very first inner iteration even when ||R|| is already ~0 (an
     exact-eigenvector initial guess) — there is no previous energy yet to
     confirm eigenvalue stability. Regression coverage for the pyscf-style
     two-condition stopping criterion (``|dE| < sqrt(de_tol)`` and
     ``||R|| < eps``), for both ways of reaching the second data point (an
     outer restart, and in-place subspace growth).
  6. _pick_num_block caps the auto-sized subspace at N = M_a * M_b.
  7. n_roots=2 via TorchSCISolver on a small molecule (H4 full CI) returns
     ground-state energy matching PySCF FCI and a second energy >= ground
     state, via list[EigenStates] with all roots in EigenStates.states;
     n_roots=1 (default) is bit-identical to the pre-n_roots path; n_roots
     < 1 raises ValueError.
  8. Real transition-metal Hamiltonian (dataset ID 758): a single Gram-Schmidt pass
     corrupts the ground energy on a near-degenerate block; the double pass fixes it.
  9. Real benzoic acid (dataset ID 182) small space (N=16): diagonal-preconditioner
     stagnation, not orthogonality loss — restarting cannot fix it. Also
     covers the bypass (davidson_block_size >= N) vs main-loop
     (davidson_block_size < N) agreement check.
  10. Under the alpha-beta projection, multi-root Davidson returns the lowest
      singlets (H6 full CI vs the dense spectrum): a restart from the targets
      alone used to lose the second singlet and converge onto a higher one.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch, davidson_torch

# ============================================================================
# Helpers shared across tests
# ============================================================================


def _make_symmetric_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Random symmetric n×n matrix with a dominant diagonal (well-separated eigs)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    H = 0.5 * (A + A.T)
    np.fill_diagonal(H, np.diag(H) - 20.0)
    return H


def _random_chem_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic random one-/two-body integrals with chemist 8-fold symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    np.fill_diagonal(h1, np.diag(h1) - 3.0)
    A = rng.standard_normal((norb, norb, norb, norb))
    eri = A + A.transpose(1, 0, 2, 3) + A.transpose(0, 1, 3, 2) + A.transpose(1, 0, 3, 2)
    eri = (eri + eri.transpose(2, 3, 0, 1)) * 0.05
    return h1, eri


def _full_ci_strings(norb: int, n_elec: int) -> np.ndarray:
    out = []
    for occ in itertools.combinations(range(norb), n_elec):
        s = 0
        for o in occ:
            s |= 1 << o
        out.append(s)
    return np.array(sorted(out), dtype=np.int64)


# ============================================================================
# 1. Synthetic matrix: n_roots=1 matches dense eigvalsh
# ============================================================================


def test_n_roots_1_matches_single_root():
    """n_roots=1 result is identical to the standard single-root call."""
    H = _make_symmetric_matrix(60, seed=1)
    hii = torch.tensor(np.diag(H), dtype=torch.float64)
    W = torch.zeros(60, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return torch.tensor(H @ v.numpy(), dtype=torch.float64)

    e1, v1, info1 = davidson_torch(
        hii, W, matvec, max_davidson_iters=30, davidson_block_size=20, eps=1e-10
    )
    e2, v2, info2 = davidson_torch(
        hii, W, matvec, max_davidson_iters=30, davidson_block_size=20, eps=1e-10, n_roots=1
    )

    assert isinstance(e1, float)
    assert isinstance(e2, float)
    assert abs(e1 - e2) < 1e-12, f"n_roots=1 energies differ: {e1} vs {e2}"
    assert torch.allclose(v1, v2, atol=1e-10), "n_roots=1 eigvecs differ"


# ============================================================================
# 2. Synthetic matrix: n_roots=2 returns both lowest eigenvalues
# ============================================================================


def test_n_roots_2_synthetic():
    """n_roots=2 returns both lowest eigenpairs matching np.linalg.eigvalsh."""
    H = _make_symmetric_matrix(80, seed=42)
    hii = torch.tensor(np.diag(H), dtype=torch.float64)
    W = torch.zeros(80, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return torch.tensor(H @ v.numpy(), dtype=torch.float64)

    e_dense = np.linalg.eigvalsh(H)[:2]

    # davidson_block_size=40 gives the subspace enough room to converge both roots
    energies, eigvecs, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=50, davidson_block_size=40, eps=1e-9, n_roots=2
    )

    assert energies.shape == (2,), f"Expected (2,) energies, got {energies.shape}"
    assert eigvecs.shape == (2, 80), f"Expected (2, 80) eigvecs, got {eigvecs.shape}"

    for k in range(2):
        assert abs(float(energies[k].item()) - e_dense[k]) < 1e-7, (
            f"Root {k}: davidson={float(energies[k].item()):.10f}, "
            f"dense={e_dense[k]:.10f}, diff={abs(float(energies[k].item()) - e_dense[k]):.2e}"
        )

    # Eigvecs should be unit-norm
    for k in range(2):
        nrm = float(torch.linalg.norm(eigvecs[k]).item())
        assert abs(nrm - 1.0) < 1e-8, f"Root {k} eigvec not unit-norm: {nrm}"

    # Ordering: E[0] <= E[1]
    assert float(energies[0].item()) <= float(energies[1].item()) + 1e-10


# ============================================================================
# 3. n_roots > davidson_block_size raises ValueError
# ============================================================================


def test_n_roots_exceeds_num_block_raises():
    """Davidson should raise ValueError if n_roots > davidson_block_size."""
    H = _make_symmetric_matrix(20, seed=7)
    hii = torch.tensor(np.diag(H), dtype=torch.float64)
    W = torch.zeros(20, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return torch.tensor(H @ v.numpy(), dtype=torch.float64)

    with pytest.raises(ValueError, match="n_roots"):
        davidson_torch(
            hii, W, matvec, max_davidson_iters=10, davidson_block_size=3, eps=1e-8, n_roots=5
        )


# ============================================================================
# 4. davidson_block_size > N (space dimension) doesn't corrupt the ground energy
# ============================================================================


def test_num_block_exceeds_dimension_does_not_corrupt_energy():
    """davidson_block_size forced past the true space dimension N must not corrupt E[0].

    When a pool of near-duplicate determinants collapses that size to a small N,
    ``davidson_block_size`` can end up larger than N. Davidson then has no choice but
    to attempt an (N+1)-th linearly independent basis vector in an N-dim
    space once its correction step outruns convergence — Gram-Schmidt must
    return (numerically) zero for that attempt.

    ``eps`` is set far tighter than the double-precision residual floor so
    convergence can *never* trigger before ``davidson_block_size`` is exhausted —
    deterministically forcing the breakdown path on every run rather than
    relying on getting numerically unlucky.

    Without a Gram-Schmidt breakdown guard, that forced extra step
    renormalizes pure round-off noise into a spurious basis vector, which
    contaminates ``H_proj`` and corrupts ``E[0]`` — reproducing the
    "ill-conditioned / too many repeated eigenvalues" failure this guards
    against.
    """
    N = 8
    H = _make_symmetric_matrix(N, seed=3)
    e_true = float(np.linalg.eigvalsh(H)[0])

    hii = torch.tensor(np.diag(H), dtype=torch.float64)
    W = torch.zeros(N, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return torch.tensor(H @ v.numpy(), dtype=torch.float64)

    e_dav, v_dav, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=1, davidson_block_size=12, eps=1e-300
    )

    assert not info["converged"], "eps=1e-300 must be unreachable — test setup is broken otherwise"
    assert abs(e_dav - e_true) < 1e-8, (
        f"ground energy corrupted by davidson_block_size > N: davidson={e_dav!r}, true={e_true!r}"
    )
    resid = H @ v_dav.numpy() - e_dav * v_dav.numpy()
    assert np.linalg.norm(resid) < 1e-6

    e_dav_less_strict, v_dav_less_strict, info_less_strict = davidson_torch(
        hii, W, matvec, max_davidson_iters=1, davidson_block_size=12, eps=1e-9
    )
    assert info_less_strict["converged"]


def test_bitstrings_num_block_exceeds_dimension():
    """Four benzoic acid (SQD molecules dataset ID 182) SqDRIFT shots, the first four
    in the right particle sector in the bundled 10k-shot file. Their unique alpha and
    beta strings give a tiny N=16 CI space (M_a=M_b=4), reproducing the scenario where a
    memory-budget-sized ``davidson_block_size`` exceeds N.
    """
    pytest.importorskip("qiskit_addon_sqd")
    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

    shots = [
        "000000000100110111111111111111110110010101100111111110011000",
        "000000001001101110111111111111000001010100110111010111111111",
        "000000001110011111011011111111100000100110111001110111111110",
        "000000101000111111111101011111101011010100001111110110111010",
    ]
    D = np.array([[c == "1" for c in s] for s in shots], dtype=bool)

    ci_a, ci_b = bitstring_matrix_to_ci_strs(D, open_shell=True)
    norb, nelec = 30, (18, 18)
    N = len(ci_a) * len(ci_b)
    assert N == 16, "expected these bitstrings to give an N=16 CI space"

    h1, eri = _random_chem_integrals(norb, seed=0)

    truth = (
        make_torch_sci_solver(
            device="cpu",
            davidson_block_size=N,
            davidson_tolerance=1e-9,
            max_davidson_iters=200,
            apply_ab_projection=False,
        )([(ci_a, ci_b)], h1, eri, norb, nelec)[0]
        .states[0]
        .energy
    )

    solver = make_torch_sci_solver(
        device="cpu",
        davidson_block_size=20,
        davidson_tolerance=1e-3,
        max_davidson_iters=6,
        apply_ab_projection=False,
    )
    result = solver([(ci_a, ci_b)], h1, eri, norb, nelec)[0].states[0]
    assert abs(result.energy - truth) < 1e-8, (
        f"ground energy corrupted by davidson_block_size > N: davidson={result.energy!r}, true={truth!r}"
    )
    assert result.state == "converged"


# ============================================================================
# 5. de_tol withholds convergence on the very first inner iteration even
#    when ||R|| is already ~0 (an exact-eigenvector initial guess)
# ============================================================================


def test_de_tol_withholds_convergence_on_first_inner_iteration():
    """``de_tol`` is opt-in (``None`` by default = residual-only, matching
    ``sbd_ref.davidson`` exactly). When explicitly passed, a residual that
    is already ~0 on the very first inner iteration must NOT be enough to
    declare convergence — there is no previous energy yet to confirm the
    eigenvalue has also stabilized.

    Regression coverage for the pyscf-style two-condition stopping criterion
    (``linalg_helper.py``: ``abs(de[k]) < tol and dx_norm[k] < toloose``).
    With ``de_tol`` enabled, this exact case (an initial guess that happens
    to already be an eigenvector, so ``||R||`` is ~0 at the very first Ritz
    step) would otherwise report ``converged=True`` after a single matvec
    purely from the residual — with no way to distinguish "the true ground
    state" from "a wrong/excited-state root the guess happens to sit on",
    which is the general shape of the false-convergence bug. Feeding the
    *exact* ground eigenvector as the initial guess reproduces the
    residual-already-tiny half of that bug deterministically; the fix must
    withhold convergence for lack of a previous energy to compare against,
    then confirm it on the very next data point.

    The second Ritz value the de-check compares against can come from either
    of two places, and both must gate convergence the same way: a fresh outer
    restart (``davidson_block_size=1`` above, so ``ib`` never exceeds 0 and
    only ``it`` advances), or the subspace growing from 1 to 2 vectors within
    a *single* outer iteration (``ib`` advancing from 0 to 1, no restart at
    all). The ``davidson_block_size=2`` case below exercises the latter: only
    ``n_roots=1`` is in play here (we only ever read ``E[0]``/``U[:, 0]``, the
    lowest Ritz pair of whatever subspace size is current — the single-root
    path never looks at ``E[1]``), so this isn't about a second *root*, just
    the second *data point* for the same root's de-check, reached by subspace
    growth instead of a restart.

    Note: ``de_tol``'s *effective* threshold is ``sqrt(de_tol)`` — the value
    passed below is ``eps``, matching the recommended usage.
    """
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    H = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(H)
    hii = torch.tensor(np.diag(H), dtype=torch.float64)
    W = torch.tensor(evecs[:, 0], dtype=torch.float64)  # exact ground eigenvector

    def matvec(v):
        return torch.tensor(H @ v.numpy(), dtype=torch.float64)

    # Default (de_tol=None): residual-only, so a single, already-tiny
    # residual is immediately enough — no eigenvalue-stability check at all.
    e0, _v0, info0 = davidson_torch(
        hii, W, matvec, max_davidson_iters=1, davidson_block_size=1, eps=1e-8
    )
    assert info0["converged"], (
        "de_tol defaults to None (disabled) — a tiny residual alone must "
        "converge immediately, matching sbd_ref.davidson"
    )
    assert abs(e0 - evals[0]) < 1e-10

    # davidson_block_size=1 forces exactly one inner (ib=0) step per outer
    # restart, so max_davidson_iters directly controls how many Ritz-value
    # data points the (now explicitly enabled) de_tol check has seen.
    e1, _v1, info1 = davidson_torch(
        hii, W, matvec, max_davidson_iters=1, davidson_block_size=1, eps=1e-8, de_tol=1e-8
    )
    assert info1["final_residual_norm"] < 1e-8, (
        "test setup is broken: the exact eigenvector should give ~0 residual "
        "on the very first Ritz step"
    )
    assert not info1["converged"], (
        "no previous energy exists yet after a single inner iteration — "
        "convergence must be withheld even though ||R|| is already tiny"
    )

    e2, _v2, info2 = davidson_torch(
        hii, W, matvec, max_davidson_iters=2, davidson_block_size=1, eps=1e-8, de_tol=1e-8
    )
    assert info2["converged"], (
        "a second, unchanged Ritz value (via outer restart) must confirm convergence"
    )
    assert abs(e1 - evals[0]) < 1e-10
    assert abs(e2 - evals[0]) < 1e-10

    # Same second data point, reached by growing the subspace within a
    # single outer iteration instead of restarting.
    e3, _v3, info3 = davidson_torch(
        hii, W, matvec, max_davidson_iters=1, davidson_block_size=2, eps=1e-8, de_tol=1e-8
    )
    assert info3["converged"], (
        "a second, unchanged Ritz value (via subspace growth, no restart) must confirm convergence"
    )
    assert info3["n_outer"] == 1, "this must converge within the first outer iteration"
    assert info3["n_matvecs"] == 2, "convergence must be reached at ib=1, not via a restart"
    assert abs(e3 - evals[0]) < 1e-10


# ============================================================================
# 6. _pick_num_block caps the auto-sized subspace at N = M_a * M_b
# ============================================================================


def test_pick_num_block_caps_at_ci_dimension():
    """The memory-budget heuristic must not size davidson_block_size past N.

    Free memory alone can suggest davidson_block_size up to 30, but a collapsed CI
    space (e.g. N=16 from a small sampled batch) has no valid (N+1)-th basis
    vector. _pick_num_block must clamp its auto-tuned result to N.
    """
    from torch_sci_solver import make_torch_sci_solver

    solver = make_torch_sci_solver(device="cpu")
    # Small CI space: the usual cap-at-30 heuristic would exceed N=16.
    assert solver._pick_num_block(4, 4) == 16
    # Large CI space: ordinary cap-at-30 behavior is unaffected by the N-cap.
    assert solver._pick_num_block(10, 10) == 30

    # Explicit davidson_block_size bypasses the heuristic (and the N-cap) entirely.
    solver_override = make_torch_sci_solver(device="cpu", davidson_block_size=99)
    assert solver_override._pick_num_block(4, 4) == 99


# ============================================================================
# 7. TorchSCISolver with n_roots=2 on H4 full CI (needs pyscf + qiskit-addon-sqd)
# ============================================================================

pyscf = pytest.importorskip("pyscf")
pytest.importorskip("qiskit_addon_sqd")

from pyscf import ao2mo, fci, gto, scf  # noqa: E402


def _h4_integrals(spacing: float = 0.74):
    atoms = "; ".join(f"H 0 0 {i * spacing}" for i in range(4))
    mol = gto.M(atom=atoms, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    nelec = mol.nelec
    # PySCF FCI ground state
    cisolver = fci.FCI(mf)
    e_fci_gs = cisolver.kernel()[0]
    # Excited state via nroots=2
    cisolver2 = fci.FCI(mf)
    cisolver2.nroots = 2
    fci_results = cisolver2.kernel()
    e_fci_all = sorted(fci_results[0])  # ascending energies
    return h1, eri, e_core, norb, nelec, e_fci_gs, e_fci_all


def test_n_roots_2_on_h4_matches_fci():
    """n_roots=2 returns list[EigenStates]; both roots accessible via .states."""
    h1, eri, e_core, norb, nelec, e_fci_gs, e_fci_all = _h4_integrals()

    ci_a = _full_ci_strings(norb, nelec[0])
    ci_b = _full_ci_strings(norb, nelec[1])

    # davidson_block_size must be >= n_roots; use 20 so both roots have room to converge
    solver = make_torch_sci_solver(
        device="cpu", n_roots=2, davidson_block_size=20, apply_ab_projection=False
    )
    results = solver([(ci_a, ci_b)], h1, eri, norb, nelec)

    # One subspace → one EigenStates in the returned list
    assert len(results) == 1, "Should return one EigenStates per subspace"

    # Both roots must be present inside the EigenStates
    assert len(results[0].states) == 2, (
        f"Expected 2 roots in EigenStates, got {len(results[0].states)}"
    )

    # Ground-state energy must match FCI (root 0)
    e0 = results[0].states[0].energy + e_core
    assert abs(e0 - e_fci_gs) < 1e-7, (
        f"Ground state: torch={e0:.10f}, pyscf={e_fci_gs:.10f}, diff={abs(e0 - e_fci_gs):.2e}"
    )

    # Excited-state energy (root 1)
    e1 = results[0].states[1].energy + e_core

    # Root 1 must be >= root 0 (ascending order)
    assert e1 >= e0 - 1e-8, f"Root 1 energy ({e1:.8f}) below root 0 ({e0:.8f})"

    # If PySCF returned a distinct excited state, compare with a relaxed tolerance.
    # Davidson on a bounded subspace can only guarantee convergence of the Ritz
    # value to within the residual norm; 1e-4 Ha is a safe budget for davidson_block_size=20.
    if len(e_fci_all) >= 2:
        e_ex_pyscf = e_fci_all[1]
        assert abs(e1 - e_ex_pyscf) < 1e-4, (
            f"Excited state: torch={e1:.10f}, pyscf={e_ex_pyscf:.10f}, "
            f"diff={abs(e1 - e_ex_pyscf):.2e}"
        )


def test_n_roots_1_solver_unchanged():
    """n_roots=1 (default) on H4 gives bit-identical result to baseline solver."""
    h1, eri, e_core, norb, nelec, e_fci_gs, _ = _h4_integrals()
    ci_a = _full_ci_strings(norb, nelec[0])
    ci_b = _full_ci_strings(norb, nelec[1])

    solver_baseline = make_torch_sci_solver(device="cpu", apply_ab_projection=False)
    solver_n1 = make_torch_sci_solver(device="cpu", n_roots=1, apply_ab_projection=False)

    res_baseline = solver_baseline([(ci_a, ci_b)], h1, eri, norb, nelec)
    res_n1 = solver_n1([(ci_a, ci_b)], h1, eri, norb, nelec)

    # Both return list[EigenStates]; ground state is at states[0]
    assert abs(res_baseline[0].states[0].energy - res_n1[0].states[0].energy) < 1e-12, (
        f"n_roots=1 energy differs from default: "
        f"{res_baseline[0].states[0].energy} vs {res_n1[0].states[0].energy}"
    )

    # n_roots=1 produces exactly one root per subspace
    assert len(res_n1[0].states) == 1, "EigenStates should contain exactly one root for n_roots=1"


def test_n_roots_invalid_raises():
    """n_roots < 1 on TorchSCISolver constructor should raise ValueError."""
    with pytest.raises(ValueError, match="n_roots"):
        make_torch_sci_solver(device="cpu", n_roots=0)


# ============================================================================
# 8. Real transition-metal Hamiltonian: single-pass Gram-Schmidt corrupts the
#    ground energy
# ============================================================================


def test_transition_metal_hamiltonian_survives_orthogonality_loss():
    """A near-degenerate Hamiltonian block corrupts E[0] with a single
    Gram-Schmidt pass; the double pass fixes it. ``_random_chem_integrals``
    (used elsewhere in this file) is well-conditioned and does not reproduce
    this, and neither do the bundled closed-shell organic molecules.

    The block is ``fixtures/tm_complex8_758_block_n64.npz``: the 64x64 Hamiltonian of
    "Transition metal complex 8" (Fe(CN)6, SQD molecules dataset ID 758, 54 electrons in
    50 orbitals; doi:10.5281/zenodo.23286514) on the product basis of the 8 CI strings
    stored with it (HF single/double excitations near the Fermi level, HF excluded), built
    with this package's batched matvec. It is stored because the FCIDUMP (56 MB) is too
    large to bundle.

    One Gram-Schmidt pass lets the correction vector re-admit round-off from
    the near-degenerate low-lying states typical of transition-metal complexes,
    producing a near-duplicate basis row. Because H_proj assumes an
    orthonormal basis and is solved as a standard (not generalized)
    eigenproblem, that duplicate row contributes a [[h, h], [h, h]] block
    whose lowest eigenvalue is 2h -- the reported ground energy plunges to
    roughly twice the true value, comfortably below the variational bound.
    """
    fixture = np.load(Path(__file__).parent / "fixtures" / "tm_complex8_758_block_n64.npz")
    H = torch.tensor(fixture["H"], dtype=torch.float64)
    N = H.shape[0]
    e_true = float(np.linalg.eigvalsh(H.numpy())[0])

    hii = torch.diag(H)
    W = torch.zeros(N, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return H @ v

    # Tight settings: many restarts give round-off many chances to compound.
    e_dav, _, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=200, davidson_block_size=30, eps=1e-10
    )
    assert info["converged"], "eps=1e-10 should be reachable on this real block"
    assert e_dav > e_true - 1e-6, (
        f"ground energy fell below the variational bound (real transition-metal block): "
        f"davidson={e_dav!r}, true={e_true!r}"
    )

    # Production settings (davidson_block_size=20, max_davidson_iters=6, davidson_tolerance=1e-3) must
    # not corrupt the energy either, even though they converge more loosely.
    e_prod, _, info_prod = davidson_torch(
        hii, W, matvec, max_davidson_iters=6, davidson_block_size=20, eps=1e-3
    )
    assert e_prod > e_true - 1e-2, (
        f"production settings corrupted the ground energy: davidson={e_prod!r}, true={e_true!r}"
    )


def test_multiroot_seed_block_escapes_degenerate_unit_vector():
    """``_make_seed_block`` seeds root 0 with ``W`` and roots 1..n_roots-1
    with raw one-hot unit vectors at the lowest ``hii`` positions. If one of
    those unit vectors happens to be an exact eigenvector of H (e.g. its
    index is fully decoupled from the rest of the matrix), its residual is
    exactly 0 and Davidson has no correction to build from it -- the same
    failure the earlier seed-stall fix addressed for the single ``W`` guess (root 0, via
    ``basis_init_vector_torch``'s noise), but left unaddressed for the
    unit-vector columns built here.

    H below: index 0 is decoupled (eigenvalue 1.0, seeds ``W`` -- protected
    by ``basis_init_vector_torch``'s noise). Index 1 is also decoupled
    (eigenvalue -0.9) and, because its ``hii`` is the lowest in the matrix,
    is exactly the unit vector ``_make_seed_block`` picks for column 1.
    Indices {2, 3} are a coupled block whose true eigenvalues (-1.058,
    0.558) are NOT visible from either seed's own diagonal. With
    davidson_block_size=3, there is only one spare slot to grow into, so a
    frozen (zero-residual) column 1 leaves the block unable to ever discover
    the true second-lowest eigenvalue (-0.9 itself is already correct here,
    but without the fix Davidson instead reports 0.558 -- the *third*-lowest
    eigenvalue -- as root 1, having never explored past its stuck seed).
    """
    H = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -0.9, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.3],
            [0.0, 0.0, 0.3, 0.5],
        ],
        dtype=torch.float64,
    )
    hii = torch.diag(H).clone()
    e_true = torch.linalg.eigvalsh(H)  # [-1.0578, -0.9, 0.5578, 1.0]

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return H @ v

    W = basis_init_vector_torch(4, 1, init=0, device="cpu", dtype=torch.float64)

    # davidson_block_size=3 < N=4: exercises the iterative seed-block path
    # (davidson_block_size=4 would trip the N<=davidson_block_size direct-diag
    # escape and bypass _make_seed_block entirely).
    e_dav, _, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=10, davidson_block_size=3, eps=1e-9, n_roots=2
    )
    assert info["converged"]
    assert e_dav[0] == pytest.approx(e_true[0].item(), abs=1e-8)
    assert e_dav[1] == pytest.approx(e_true[1].item(), abs=1e-8), (
        f"root 1 should be the true second-lowest eigenvalue ({e_true[1].item()!r}), "
        f"not a higher eigenvalue reached by skipping past the frozen unit-vector seed: "
        f"got {e_dav[1].item()!r}"
    )


# CI strings (identical for alpha and beta) of the benzoic acid (SQD molecules dataset ID 182)
# block below: configuration
# recovery, with the Hartree-Fock prior and seed 5, on the first 200 bundled SqDRIFT shots,
# strings [5:9] in order of excitation level from Hartree-Fock. Chosen so the test fails
# when the direct-diagonalization path it guards is removed.
_BENZOIC_ACID_STRS_N16 = [4456191, 8648703, 8650623, 16973823]


def _benzoic_acid_block(strs: list[int]) -> torch.Tensor:
    """Dense benzoic acid Hamiltonian on the product basis ``strs x strs``."""
    from pyscf import ao2mo
    from pyscf.tools import fcidump
    from torch_sci_solver import _paths
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
    from torch_sci_solver.torch_impl.validate_matvec import _setup

    raw = fcidump.read(str(_paths.fcidump_path("benzoic_acid")), verbose=False)
    norb = raw["NORB"]
    eri = ao2mo.restore(1, raw["H2"].astype(np.float64), norb)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(raw["H1"], eri, raw["ECORE"])
    hii, adets, bdets, integrals, helpers = _setup(
        strs, strs, norb, I0, I1, I2, torch.device("cpu"), torch.float64, 2_000_000
    )
    matvec = build_matvec(
        "batched", hii, adets, bdets, norb, integrals, helpers, chunk_size=2_000_000
    )
    eye = torch.eye(len(strs) ** 2, dtype=torch.float64)
    return torch.stack([matvec(eye[:, k]) for k in range(eye.shape[0])], dim=1)


# ============================================================================
# 9. Real benzoic acid small space (N=16): diagonal-preconditioner
#    stagnation, not orthogonality loss -- restarting cannot fix it.
# ============================================================================


def test_real_benzoic_acid_small_space_converges():
    """
    Tests a distinct failure mode from the N=64 orthogonality-loss
    case above: here N=16 is small enough that the whole CI space fits in one
    Davidson block (davidson_block_size=20), but the diagonal-preconditioned correction
    vector collapses onto the existing 2-vector block after the first
    restart -- confirmed by tracing every outer iteration -- landing on a
    Ritz value that is not an eigenvalue of the block at all. Because the
    stall is deterministic and the restart seed carries no randomness,
    max_davidson_iters=6, 50, and 200 all reproduce the identical wrong energy;
    more iterations cannot help escape an invariant subspace that Davidson's
    own correction-vector map created.

    The fix: when N <= davidson_block_size, skip the Krylov iteration and diagonalize
    directly -- the iterative method offers no benefit when the whole space
    already fits in a single block, and it carries active stagnation risk.
    """
    H = _benzoic_acid_block(_BENZOIC_ACID_STRS_N16)
    N = H.shape[0]
    assert N == 16
    e_true = float(np.linalg.eigvalsh(H.numpy())[0])

    hii = torch.diag(H)
    W = torch.zeros(N, dtype=torch.float64)
    W[0] = 1.0

    def matvec(v):
        return H @ v

    e_dav, v_dav, info = davidson_torch(
        hii, W, matvec, max_davidson_iters=6, davidson_block_size=20, eps=1e-4
    )
    assert info["converged"], (
        f"production settings failed to converge: davidson={e_dav!r}, "
        f"true={e_true!r}, resid={info['final_residual_norm']!r}"
    )
    assert abs(e_dav - e_true) < 1e-8, f"davidson={e_dav!r}, true={e_true!r}"

    resid = H.numpy() @ v_dav.numpy() - e_dav * v_dav.numpy()
    assert np.linalg.norm(resid) < 1e-6, "returned vector is not an eigenvector of H"


def test_bypass_and_main_loop_agree_when_seed_is_non_ground_eigenvector():
    """H's row/col 0 is decoupled, so the one-hot seed (index 0) is an exact
    eigenvector of eigenvalue 1.0 — not the ground state, which lives in the
    coupled 1-2 block. The bypass (davidson_block_size >= N) must find it, and so must
    the main loop (davidson_block_size < N).
    """
    H = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.3],
            [0.0, 0.3, 0.5],
        ],
        dtype=torch.float64,
    )
    hii = torch.diag(H).clone()

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return H @ v

    W = basis_init_vector_torch(3, 1, init=0, device="cpu", dtype=torch.float64)
    e_true = torch.linalg.eigvalsh(H)[0].item()

    e_bypass, _, info_bypass = davidson_torch(
        hii, W, matvec, max_davidson_iters=30, davidson_block_size=3, eps=1e-9
    )
    e_main, _, info_main = davidson_torch(
        hii, W, matvec, max_davidson_iters=30, davidson_block_size=2, eps=1e-9
    )

    assert info_bypass["converged"]
    assert e_bypass == pytest.approx(e_true, abs=1e-8)
    assert info_main["converged"]
    assert e_main == pytest.approx(e_true, abs=1e-8)


# ============================================================================
# 10. Multi-root restart must not lose a lower root of the projected sector
# ============================================================================


@pytest.mark.parametrize("davidson_block_size", [8, 12, 20])
def test_multiroot_finds_lowest_singlets_under_ab_projection(davidson_block_size):
    """With the alpha-beta projection, root 1 must be the second-lowest singlet.

    H6/STO-3G full CI: the singlet sector's two lowest states are E0 and E3 (E1/E2
    are triplets). The second seed, the next-lowest diagonal determinant after
    projection, is ~0.97 E4, and E3 is reachable only through the rest of the
    subspace. Restarting from the n_roots target Ritz vectors alone discarded that
    direction, so root 1 converged onto E4 and the solve still reported
    convergence (block 8 and 12; block 20 happened to keep it). The buffer Ritz
    vector kept across restarts fixes it. Checked against the dense spectrum, not
    against another Davidson run.
    """
    from sci_solvers.eigen_problem import EigenProblem

    mol = gto.M(atom="; ".join(f"H 0 0 {i * 1.0}" for i in range(6)), basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    strs = _full_ci_strings(norb, mol.nelec[0])

    dense = EigenProblem(strs, strs, h1, eri, norb, mol.nelec).sparse().toarray()
    energies, vectors = np.linalg.eigh(dense)
    m = len(strs)
    singlets = [
        e
        for e, v in zip(energies, vectors.T)
        if np.linalg.norm(v.reshape(m, m) - v.reshape(m, m).T) < 1e-6  # alpha<->beta even
    ]

    solver = make_torch_sci_solver(
        device="cpu",
        n_roots=2,
        davidson_block_size=davidson_block_size,
        max_davidson_iters=60,
        davidson_tolerance=1e-6,
        apply_ab_projection=True,
    )
    states = solver([(strs, strs)], h1, eri, norb, mol.nelec)[0].states
    assert all(s.state == "converged" for s in states)
    np.testing.assert_allclose([s.energy for s in states], singlets[:2], atol=1e-8)


@pytest.mark.parametrize("n_roots", [2, 3])
def test_multiroot_block_equal_to_n_roots_keeps_every_target(n_roots):
    """With no free slot for the buffer, every target Ritz vector must still be kept.

    The buffer is only added when ``davidson_block_size >= n_roots + 2``. Capping the
    kept count at ``davidson_block_size - 1`` instead kept fewer Ritz vectors than
    targets, which crashed at ``n_roots=3`` (broadcast mismatch). With no free slot
    the solve cannot converge, as on main, but it must run and return every root
    with orthonormal vectors whose energies are their Rayleigh quotients.
    """
    h1, eri = _random_chem_integrals(6, seed=3)
    strs = _full_ci_strings(6, 3)
    solver = make_torch_sci_solver(
        device="cpu",
        n_roots=n_roots,
        davidson_block_size=n_roots,
        max_davidson_iters=5,
        apply_ab_projection=False,
    )
    with pytest.warns(RuntimeWarning, match="extremely small"):
        states = solver([(strs, strs)], h1, eri, 6, (3, 3))[0].states
    assert len(states) == n_roots
    # A dropped target's Ritz vector shows as a duplicate or zero root vector, and its
    # energy then no longer belongs to it: check orthonormality and that each energy is
    # its own vector's Rayleigh quotient.
    from sci_solvers.eigen_problem import EigenProblem

    dense = EigenProblem(strs, strs, h1, eri, 6, (3, 3)).sparse().toarray()
    vecs = np.stack([s.sci_state.amplitudes.ravel() for s in states])
    np.testing.assert_allclose(vecs @ vecs.T, np.eye(n_roots), atol=1e-8)
    rayleigh = np.einsum("ki,ij,kj->k", vecs, dense, vecs)
    np.testing.assert_allclose([s.energy for s in states], rayleigh, atol=1e-8)
