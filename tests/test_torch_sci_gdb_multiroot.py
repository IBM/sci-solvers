"""Validate real GDB multi-root solves against TPB on the same Cartesian product."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from itertools import combinations
from typing import Any

import numpy as np
import pytest
import torch
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl import mult_pairs
from torch_sci_solver.torch_impl.davidson import davidson_torch
from torch_sci_solver.torch_impl.pairs import PairSCIState
from torch_sci_solver.torch_impl.rdm_pairs import _inter_to_block_signs_pair
from torch_sci_solver.torch_impl.sbdiag import diag_torch

from sci_solvers.eigen_problem import PairCIProblem
from sci_solvers.energy_solvers.factory import create_solver

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device"),
    ),
]


def _assert_amplitudes_match(actual: np.ndarray, expected: np.ndarray, atol: float) -> None:
    """Compare (n_roots, n_pairs) coefficients up to each real eigenvector's overall sign."""
    assert actual.shape == expected.shape
    signs = np.where(np.sum(actual * expected, axis=1) < 0, -1.0, 1.0)
    aligned = actual * signs[:, None]
    np.testing.assert_allclose(aligned, expected, rtol=0, atol=atol)
    assert np.max(np.linalg.norm(aligned - expected, axis=1)) < 10 * atol


@pytest.fixture
def product_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Provide six strings per spin and real four-orbital chemical integrals."""
    strings = np.array(sorted(sum(1 << i for i in c) for c in combinations(range(4), 2)))
    rng = np.random.default_rng(73)
    h = rng.standard_normal((4, 4))
    hcore = h + h.T
    eri = np.zeros((4, 4, 4, 4))
    for _ in range(3):
        g = rng.standard_normal((4, 4))
        g = 0.1 * (g + g.T)
        eri += np.einsum("pq,rs->pqrs", g, g)
    return strings, hcore, eri


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "n_roots,block_size,project,spin_sq",
    [
        (1, 12, False, None),
        (2, 12, False, None),
        (3, 12, False, None),
        (3, 40, False, None),
        (3, None, False, None),
        (3, 12, True, None),
        (3, 40, True, None),
        (3, None, True, None),
        (3, 40, False, 0.0),
    ],
)
def test_gdb_roots_match_tpb_on_cartesian_product(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_roots: int,
    block_size: int | None,
    project: bool,
    spin_sq: float | None,
    device: str,
) -> None:
    """Compare every root, its observables and its residual in the identical CI space."""
    strings, hcore, eri = product_problem
    atol = 2e-4 if device == "mps" else 1e-7
    residual_tol = 2e-4 if device == "mps" else 1e-8
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    # Scrambling and duplicating pairs must preserve the Cartesian-product space.
    order = np.random.default_rng(5).permutation(len(pa))
    pair_batch = (np.r_[pa[order], pa[:2]], np.r_[pb[order], pb[:2]])
    options = dict(
        device=device,
        dtype=torch.float32 if device == "mps" else torch.float64,
        n_roots=n_roots,
        davidson_block_size=block_size,
        max_davidson_iters=100,
        davidson_tolerance=5e-5 if device == "mps" else 1e-9,
        apply_ab_projection=project,
        compute_full_rdm=True,
        compute_spin_squared=True,
    )
    tpb = make_torch_sci_solver(mode="tpb", **options)
    gdb = make_torch_sci_solver(mode="gdb", **options)
    reference = tpb([(strings, strings)], hcore, eri, 4, (2, 2), spin_sq=spin_sq)[0]
    actual = gdb([pair_batch], hcore, eri, 4, (2, 2), spin_sq=spin_sq)[0]

    assert len(actual.states) == len(reference.states) == n_roots
    problem = PairCIProblem(pa, pb, hcore, eri, 4, (2, 2))
    sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), 4).numpy()
    vectors = np.stack([root.sci_state.amplitudes for root in actual.states])
    np.testing.assert_allclose(vectors @ vectors.T, np.eye(n_roots), rtol=0, atol=atol)
    reference_vectors = np.stack([root.sci_state.amplitudes.ravel() for root in reference.states])
    _assert_amplitudes_match(vectors * sigma, reference_vectors, atol)
    for root_idx, (got, ref) in enumerate(zip(actual.states, reference.states)):
        assert got.state == ref.state == "converged"
        assert isinstance(got.sci_state, PairSCIState)
        assert got.sci_state.amplitudes.shape == (36,)
        np.testing.assert_array_equal(got.sci_state.ci_strs_a, pa)
        np.testing.assert_array_equal(got.sci_state.ci_strs_b, pb)
        assert got.energy == pytest.approx(ref.energy, abs=residual_tol)
        np.testing.assert_allclose(
            got.orbital_occupancies, ref.orbital_occupancies, rtol=0, atol=atol
        )
        residual = problem.matvec(vectors[root_idx]) - got.energy * vectors[root_idx]
        assert np.linalg.norm(residual) < residual_tol
        if root_idx == 0:
            np.testing.assert_allclose(got.rdm1, ref.rdm1, rtol=0, atol=atol)
            np.testing.assert_allclose(got.rdm2, ref.rdm2, rtol=0, atol=atol)
        else:
            assert got.rdm1 is got.rdm2 is None
    np.testing.assert_allclose(
        gdb.stats["spin_squared"], tpb.stats["spin_squared"], rtol=0, atol=atol
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "norb,nelec,n_roots,seed",
    [
        pytest.param(6, (3, 3), 4, 73, id="400-pairs-4-roots"),
        pytest.param(6, (3, 3), 4, 19, id="400-pairs-second-seed"),
        pytest.param(7, (3, 2), 5, 73, id="735-pairs-open-shell"),
        pytest.param(8, (4, 4), 6, 73, marks=pytest.mark.slow, id="4900-pairs-6-roots"),
    ],
)
def test_scaled_gdb_amplitudes_match_cpu_tpb(
    norb: int,
    nelec: tuple[int, int],
    n_roots: int,
    seed: int,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep GDB matvecs on the requested device and match CPU TPB coefficients after restarts."""
    strings = [
        np.array(sorted(sum(1 << i for i in c) for c in combinations(range(norb), count)))
        for count in nelec
    ]
    sa, sb = strings
    pa, pb = np.repeat(sa, len(sb)), np.tile(sb, len(sa))
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    hcore = (h + h.T) * 0.1 + np.diag(np.arange(norb) * 0.5)
    eri = np.zeros((norb,) * 4)
    for _ in range(3):
        g = rng.standard_normal((norb, norb))
        g = (g + g.T) * 0.03
        eri += np.einsum("pq,rs->pqrs", g, g)

    block_size = 24
    assert len(pa) > block_size  # These cases must exercise iterative Davidson.
    options = dict(
        n_roots=n_roots,
        apply_ab_projection=False,
        matvec_impl="batched",
        davidson_block_size=block_size,
        max_davidson_iters=100,
    )
    tpb = make_torch_sci_solver(device="cpu", davidson_tolerance=1e-9, **options)
    reference = tpb([(sa, sb)], hcore, eri, norb, nelec)[0]
    ref_energies = np.array([root.energy for root in reference.states])
    # These fixtures have isolated eigenvalues, so sign alignment suffices;
    # degenerate eigenspaces would require comparing subspaces instead.
    assert np.min(np.diff(ref_energies)) > 1e-4

    original_matvec = mult_pairs.mult_pair_torch
    matvec_calls = 0

    def checked_matvec(
        hii: torch.Tensor, v: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Verify that (N,) diagonal, input and output stay on the requested device.

        On MPS the solver evaluates each root's energy once more as a float64
        Rayleigh quotient on the CPU (MPS has no float64); that call is the only
        one allowed off the device.
        """
        nonlocal matvec_calls
        if device == "mps" and v.device.type == "cpu":
            assert hii.device.type == "cpu" and v.dtype in (torch.float64, torch.complex128)
            return original_matvec(hii, v, *args, **kwargs)
        assert hii.device.type == v.device.type == device
        result = original_matvec(hii, v, *args, **kwargs)
        assert result.device == v.device
        matvec_calls += 1
        return result

    monkeypatch.setattr(mult_pairs, "mult_pair_torch", checked_matvec)
    # The largest Hamiltonian needs a higher float32 residual threshold;
    # retain tighter solves for the smaller fixtures with nearby roots.
    mps_eps = 5e-5 if norb == 8 else 1e-5
    gdb = make_torch_sci_solver(
        mode="gdb",
        device=device,
        dtype=torch.float32 if device == "mps" else torch.float64,
        davidson_tolerance=mps_eps if device == "mps" else 1e-9,
        **options,
    )
    # Preserve the same pair set despite a different input order and duplicates.
    order = rng.permutation(len(pa))
    batch = (np.r_[pa[order], pa[:3]], np.r_[pb[order], pb[:3]])
    actual = gdb([batch], hcore, eri, norb, nelec)[0]
    monkeypatch.setattr(mult_pairs, "mult_pair_torch", original_matvec)
    assert matvec_calls > block_size  # Require a restart, including on GPU.
    assert len(actual.states) == len(reference.states) == n_roots
    assert all(root.state == "converged" for root in actual.states + reference.states)

    # Nearby spin partners amplify float32 rounding in individual eigenvectors.
    # Keep energy/residual checks tighter, and require float64 coefficient parity
    # on CPU/CUDA. MPS cannot run float64.
    atol = 5e-4 if device == "mps" else 1e-7
    energy_tol = 2e-5 if device == "mps" else 1e-8
    residual_tol = 2 * mps_eps if device == "mps" else 1e-8
    energies = np.array([root.energy for root in actual.states])
    np.testing.assert_allclose(energies, ref_energies, rtol=0, atol=energy_tol)
    vectors = np.stack([root.sci_state.amplitudes for root in actual.states])
    ref_vectors = np.stack([root.sci_state.amplitudes.ravel() for root in reference.states])
    assert vectors.shape == (n_roots, len(pa))
    sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), norb).numpy()
    # GDB amplitudes use interleaved-Wick signs; TPB exports spin-block signs.
    _assert_amplitudes_match(vectors * sigma, ref_vectors, atol)
    np.testing.assert_allclose(vectors @ vectors.T, np.eye(n_roots), rtol=0, atol=atol)
    problem = PairCIProblem(pa, pb, hcore, eri, norb, nelec)
    residuals = problem.matvec(vectors.T) - vectors.T * energies
    assert np.max(np.linalg.norm(residuals, axis=0)) < residual_tol
    for got, ref in zip(actual.states, reference.states):
        np.testing.assert_array_equal(got.sci_state.ci_strs_a, pa)
        np.testing.assert_array_equal(got.sci_state.ci_strs_b, pb)
        np.testing.assert_allclose(
            got.orbital_occupancies, ref.orbital_occupancies, rtol=0, atol=atol
        )


@pytest.mark.parametrize("n_roots", [1, 3])
@pytest.mark.parametrize("block_size", [12, 40])
def test_diag_torch_gdb_multiroot(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_roots: int,
    block_size: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise low-level forwarding, scalar compatibility and verbose multi-root output."""
    strings, hcore, eri = product_problem
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    integrals = setup_integrals_from_pyscf_h1_eri(hcore, eri, 0.3)
    options = dict(
        n_roots=n_roots,
        davidson_block_size=block_size,
        max_davidson_iters=100,
        eps=1e-9,
        verbose=True,
    )
    expected, _, ref_info = diag_torch(strings.tolist(), strings.tolist(), 4, *integrals, **options)
    actual, vectors, info = diag_torch(
        pa.tolist(), pb.tolist(), 4, *integrals, mode="gdb", **options
    )
    assert info["converged"] and ref_info["converged"]
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert vectors.shape == ((36,) if n_roots == 1 else (n_roots, 36))
    if n_roots == 1:
        assert isinstance(actual, float)
        assert "pair Energy" in capsys.readouterr().out
    else:
        assert isinstance(actual, torch.Tensor)
        assert "pair Energies" in capsys.readouterr().out


@pytest.mark.parametrize("root_key", ["nroots", "n_roots"])
def test_outer_torch_solver_preserves_gdb_roots(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray], root_key: str
) -> None:
    """Both public root-count spellings return all roots in every batch."""
    strings, hcore, eri = product_problem
    batch = (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    solver = create_solver(
        "torch_sci",
        mode="gdb",
        symmetrize_spin=False,
        device="cpu",
        apply_ab_projection=False,
        davidson_block_size=40,
        **{root_key: 3},
    )
    results = solver.sci_solver_callable_eigen_states([batch, batch], hcore, eri, 4, (2, 2))
    assert len(results) == 2
    assert all(len(result.states) == 3 for result in results)
    assert all(root.state == "converged" for result in results for root in result.states)
    energies = [[root.energy for root in result.states] for result in results]
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-12)
    assert np.all(np.diff(energies[0]) >= 0)


def test_gdb_multiroot_warm_start_keeps_ground_state_snapshot(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Warm extension preserves all roots while caching a single flat ground-state vector."""
    strings, hcore, eri = product_problem
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    solver = make_torch_sci_solver(
        mode="gdb",
        device="cpu",
        n_roots=3,
        apply_ab_projection=False,
        davidson_block_size=12,
        max_davidson_iters=100,
        davidson_tolerance=1e-9,
    )
    core, snapshots = solver([(pa[:18], pb[:18])], hcore, eri, 4, (2, 2), probe_method="warm_start")
    assert snapshots[0].eigvec_inter.shape == (18,)
    np.testing.assert_allclose(snapshots[0].eigvec_inter, core[0].states[0].sci_state.amplitudes)
    warm = solver(
        [(pa, pb)], hcore, eri, 4, (2, 2), probe_method="warm_start", probe_state=snapshots
    )
    cold = solver([(pa, pb)], hcore, eri, 4, (2, 2))
    assert len(warm[0].states) == len(cold[0].states) == 3
    np.testing.assert_allclose(
        [root.energy for root in warm[0].states],
        [root.energy for root in cold[0].states],
        atol=1e-8,
    )


def test_gdb_multiroot_rejects_too_many_roots_after_dedup(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Repeated input pairs do not count as independent eigenstates."""
    strings, hcore, eri = product_problem
    solver = make_torch_sci_solver(mode="gdb", n_roots=3, apply_ab_projection=False)
    with pytest.raises(ValueError, match="CI dimension"):
        solver([(strings[[0, 1, 0, 1]], strings[[0, 0, 0, 0]])], hcore, eri, 4, (2, 2))


@pytest.mark.parametrize("block_size", [3, 8])
def test_multiroot_projection_respects_sector_dimension(block_size: int) -> None:
    """A projector killing the initial guess must still yield valid roots or reject excess roots."""
    h = torch.diag(torch.arange(-6.0, 0.0, dtype=torch.float64))
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        """Apply the diagonal Hamiltonian to a vector of shape (6,)."""
        return h @ v

    def project(v: torch.Tensor) -> torch.Tensor:
        """Project a vector of shape (6,) into the selected two-dimensional sector."""
        return mask * v

    w = torch.eye(6, dtype=torch.float64)[0]
    energies, vectors, info = davidson_torch(
        h.diag(), w, matvec, 20, block_size, 1e-9, n_roots=2, project_fn=project
    )
    assert info["converged"]
    torch.testing.assert_close(energies, torch.tensor([-5.0, -3.0], dtype=torch.float64))
    torch.testing.assert_close(vectors @ vectors.T, torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="projected CI dimension"):
        davidson_torch(h.diag(), w, matvec, 20, block_size, 1e-9, n_roots=3, project_fn=project)


def test_complex_gdb_multiroot_is_supported() -> None:
    """Complex dtype no longer blocks GDB multi-root construction."""
    solver = make_torch_sci_solver(mode="gdb", n_roots=2, dtype=torch.complex128)
    assert solver.n_roots == 2
    assert solver.dtype == torch.complex128


def test_gdb_multiroot_reports_nonconvergence(
    product_problem: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Exhausting the iteration budget must not mark an incomplete root block converged."""
    strings, hcore, eri = product_problem
    batch = (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    solver = make_torch_sci_solver(
        mode="gdb",
        device="cpu",
        n_roots=3,
        apply_ab_projection=False,
        davidson_block_size=6,
        max_davidson_iters=1,
        davidson_tolerance=1e-12,
    )
    result = solver([batch], hcore, eri, 4, (2, 2))[0]
    assert len(result.states) == 3
    assert all(root.state == "not_converged" for root in result.states)


# ---------------------------------------------------------------------------
# Complex multi-root: the GDB diagonal must be real for Davidson's seed sort.
# ---------------------------------------------------------------------------

# These complex128 checks require CPU or CUDA (MPS has no float64 support).
COMPLEX_DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    ),
]


def _complex_hermitian_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian ``hcore`` and a physically valid complex chemist-order ERI."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    eri = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    eri = 0.5 * (eri + eri.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    # A symmetrized factor * conj(factor) fixture accidentally makes ERI real.
    assert np.max(np.abs(eri.imag)) > 0.1
    return hcore, eri


def _assert_complex_amplitudes_match(actual: np.ndarray, expected: np.ndarray, atol: float) -> None:
    """Compare (n_roots, n) complex coefficients up to one global phase per root."""
    assert actual.shape == expected.shape
    for got, ref in zip(actual, expected):
        phase = np.vdot(ref, got)
        if abs(phase) > 0:
            got = got * np.conj(phase) / abs(phase)
        np.testing.assert_allclose(got, ref, rtol=0, atol=atol)


def test_pair_diagonal_stays_real_for_complex_integrals() -> None:
    """Pair-mode ``hii`` must be real: complex breaks Davidson's ``argsort`` seed."""
    from torch_sci_solver.torch_impl.integrals import (
        setup_integrals_from_pyscf_h1_eri_vec,
        to_torch_integrals,
    )
    from torch_sci_solver.torch_impl.mult_pairs import make_diag_terms_pair_torch

    strings = np.array(sorted(sum(1 << i for i in c) for c in combinations(range(4), 2)))
    hcore, eri = _complex_hermitian_integrals(4, seed=13)
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(hcore, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=torch.device("cpu"), dtype=torch.complex128)
    basis = PairCIProblem(pa, pb, hcore, eri, 4, (2, 2)).pair_basis
    hii = make_diag_terms_pair_torch(basis, 4, integrals)
    assert hii.dtype == torch.float64


@pytest.mark.parametrize("device", COMPLEX_DEVICES)
@pytest.mark.parametrize("nonproduct", [False, True], ids=["product", "nonproduct"])
def test_complex_gdb_matches_independent_reference(device: str, nonproduct: bool) -> None:
    """Match three (36,) or (29,) roots to an independent PySCF operator."""
    from sci_solvers.complex_ci_problem import build_complex_ci_problem

    strings = np.array(sorted(sum(1 << i for i in c) for c in combinations(range(4), 2)))
    hcore, eri = _complex_hermitian_integrals(4, seed=13)
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    rows = (
        np.sort(np.random.default_rng(19).choice(36, 29, replace=False))
        if nonproduct
        else np.arange(36)
    )
    pa, pb = pa[rows], pb[rows]
    full = build_complex_ci_problem(hcore, eri, 4, (2, 2)).sparse().toarray()
    sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), 4).numpy()
    h = sigma[:, None] * full[np.ix_(rows, rows)] * sigma[None, :]
    expected_e, expected_v = np.linalg.eigh(h)
    solver = make_torch_sci_solver(
        mode="gdb",
        device=device,
        dtype=torch.complex128,
        n_roots=3,
        davidson_block_size=12,
        max_davidson_iters=100,
        davidson_tolerance=1e-9,
        apply_ab_projection=False,
    )
    result = solver([(pa, pb)], hcore, eri, 4, (2, 2))[0]
    assert all(root.state == "converged" for root in result.states)
    energies = np.array([root.energy for root in result.states])
    vectors = np.stack([root.sci_state.amplitudes for root in result.states])
    np.testing.assert_allclose(energies, expected_e[:3], atol=1e-8, rtol=0)
    _assert_complex_amplitudes_match(vectors, expected_v[:, :3].T, 1e-7)
    assert np.max(np.linalg.norm(h @ vectors.T - vectors.T * energies, axis=0)) < 1e-8


@pytest.mark.parametrize("broken_symmetry", ["conjugate", "particle-exchange"])
def test_complex_eri_rejects_each_broken_symmetry(broken_symmetry: str) -> None:
    """Reject (2,2,2,2) ERIs violating either chemist-order symmetry separately."""
    from torch_sci_solver.torch_impl.integrals import setup_integrals_from_pyscf_h1_eri_vec

    eri = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    eri[0, 1, 0, 0] = 1.0
    if broken_symmetry == "conjugate":
        eri[0, 0, 0, 1] = 1.0  # preserves particle exchange only
    else:
        eri[1, 0, 0, 0] = 1.0  # preserves conjugate symmetry only
    with pytest.raises(ValueError, match=broken_symmetry):
        setup_integrals_from_pyscf_h1_eri_vec(np.eye(2), eri, 0.0)


@pytest.mark.parametrize("device", COMPLEX_DEVICES)
@pytest.mark.parametrize(
    "n_roots,block_size,project,spin_sq",
    [
        (1, 12, False, None),
        (3, 12, False, None),
        (3, 40, False, None),
        (3, 40, True, None),
        (3, 12, False, 0.0),
    ],
)
def test_complex_gdb_roots_match_tpb_on_cartesian_product(
    n_roots: int,
    block_size: int,
    project: bool,
    spin_sq: float | None,
    device: str,
) -> None:
    """Complex GDB multi-root matches TPB on energies, phases, RDMs and residuals."""
    strings = np.array(sorted(sum(1 << i for i in c) for c in combinations(range(4), 2)))
    hcore, eri = _complex_hermitian_integrals(4, seed=13)
    pa, pb = np.repeat(strings, len(strings)), np.tile(strings, len(strings))
    # Scrambling and duplicating pairs must preserve the Cartesian-product space.
    order = np.random.default_rng(5).permutation(len(pa))
    pair_batch = (np.r_[pa[order], pa[:2]], np.r_[pb[order], pb[:2]])
    options = dict(
        device=device,
        dtype=torch.complex128,
        n_roots=n_roots,
        davidson_block_size=block_size,
        max_davidson_iters=100,
        davidson_tolerance=1e-9,
        apply_ab_projection=project,
        compute_full_rdm=True,
        compute_spin_squared=True,
    )
    tpb = make_torch_sci_solver(mode="tpb", **options)
    gdb = make_torch_sci_solver(mode="gdb", **options)
    reference = tpb([(strings, strings)], hcore, eri, 4, (2, 2), spin_sq=spin_sq)[0]
    actual = gdb([pair_batch], hcore, eri, 4, (2, 2), spin_sq=spin_sq)[0]

    atol = 1e-7
    residual_tol = 1e-8
    assert len(actual.states) == len(reference.states) == n_roots
    energies = np.array([root.energy for root in actual.states])
    ref_energies = np.array([root.energy for root in reference.states])
    np.testing.assert_allclose(energies, ref_energies, rtol=0, atol=atol)

    problem = PairCIProblem(pa, pb, hcore, eri, 4, (2, 2))
    sigma = _inter_to_block_signs_pair(torch.as_tensor(pa), torch.as_tensor(pb), 4).numpy()
    vectors = np.stack([root.sci_state.amplitudes for root in actual.states])
    reference_vectors = np.stack([root.sci_state.amplitudes.ravel() for root in reference.states])
    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(n_roots), rtol=0, atol=atol)
    # GDB amplitudes use interleaved-Wick signs; TPB exports spin-block signs.
    _assert_complex_amplitudes_match(vectors * sigma, reference_vectors, atol)
    residuals = problem.matvec(vectors.T) - vectors.T * energies
    assert np.max(np.linalg.norm(residuals, axis=0)) < residual_tol

    for root_idx, (got, ref) in enumerate(zip(actual.states, reference.states)):
        assert got.state == ref.state == "converged"
        np.testing.assert_allclose(
            got.orbital_occupancies, ref.orbital_occupancies, rtol=0, atol=atol
        )
        if root_idx == 0:
            np.testing.assert_allclose(got.rdm1, ref.rdm1, rtol=0, atol=atol)
            np.testing.assert_allclose(got.rdm2, ref.rdm2, rtol=0, atol=atol)
        else:
            assert got.rdm1 is got.rdm2 is None
    np.testing.assert_allclose(
        gdb.stats["spin_squared"], tpb.stats["spin_squared"], rtol=0, atol=atol
    )
