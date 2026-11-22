"""Tests for the factorized αβ channel (``ab_factorization="cholesky"``).

1. ``factorize_eri``: pivoted Cholesky on a positive semidefinite supermatrix
   (all signs +1, exact to the tolerance), the signed eigendecomposition on an
   indefinite one, and the real-only guard.
2. ``FactorizedAlphaBeta`` equals the direct αβ doubles kernel on random
   vectors: uneven α/β string sets, strings wider than 64 bits, and MPS.
3. Solver level: energies, 1-RDMs, multi-root, the αβ projection and the spin
   penalty agree with ``ab_factorization="off"``.
4. Every configuration the factorized pass does not serve is rejected loudly,
   and ``config`` / ``capabilities()`` report the knob.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.ab_factorized import FactorizedAlphaBeta, factorize_eri
from torch_sci_solver.torch_impl.helpers import _as_string_limbs, generate_all_torch_helpers
from torch_sci_solver.torch_impl.integrals import to_torch_integrals
from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
from torch_sci_solver.torch_impl.mult import _mult_alphabeta_torch, make_diag_terms_torch
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

_PERMS = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
_PERMS += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]


def _random_system(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """8-fold symmetric random integrals (an indefinite ERI supermatrix)."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    return h1, sum(raw.transpose(p) for p in _PERMS) / 8.0


def _psd_system(norb: int, rank: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Integrals whose supermatrix is a sum of ``rank`` symmetric outer products."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    eri = np.zeros((norb,) * 4)
    for _ in range(rank):
        g = rng.standard_normal((norb, norb))
        g = 0.5 * (g + g.T)
        eri += 0.3 * np.einsum("pq,rs->pqrs", g, g)
    return h1, eri


def _strings(norb: int, nocc: int) -> list[int]:
    """All ``nocc``-electron strings of ``norb`` orbitals, ascending."""
    return sorted(sum(1 << o for o in c) for c in itertools.combinations(range(norb), nocc))


def test_cholesky_factors_a_semidefinite_supermatrix_exactly() -> None:
    """A rank-r PSD supermatrix gives r positive factors that rebuild it."""
    _, eri = _psd_system(6, rank=5)
    factors, signs = factorize_eri(torch.tensor(eri), tol=1e-12)
    assert factors.shape[0] == 5
    assert torch.all(signs == 1)
    rebuilt = (factors.T * signs) @ factors
    torch.testing.assert_close(rebuilt, torch.tensor(eri).reshape(36, 36), atol=1e-10, rtol=0)


def test_indefinite_supermatrix_takes_signed_factors() -> None:
    """Random integrals are indefinite: mixed signs, still an exact rebuild."""
    _, eri = _random_system(5)
    factors, signs = factorize_eri(torch.tensor(eri))
    assert torch.any(signs < 0) and torch.any(signs > 0)
    rebuilt = (factors.T * signs) @ factors
    torch.testing.assert_close(rebuilt, torch.tensor(eri).reshape(25, 25), atol=1e-9, rtol=0)


def test_factorize_rejects_complex_and_asymmetric_integrals() -> None:
    """Complex ERIs and a supermatrix without (pq)<->(rs) symmetry are refused."""
    _, eri = _random_system(4)
    with pytest.raises(ValueError, match="real integrals"):
        factorize_eri(torch.tensor(eri, dtype=torch.complex128))
    asym = eri.copy()
    asym[0, 1, 2, 3] += 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        factorize_eri(torch.tensor(asym))


def _pass_vs_direct(norb, na, nb, *, device="cpu", dtype=torch.float64, wide_orbital=None):
    """Max |factorized - direct| and the direct pass's scale on a random vector."""
    h1, eri = _random_system(norb, seed=3)
    sa, sb = _strings(norb, na), _strings(norb, nb)
    sb = sb[: max(3, len(sb) - 2)]  # an uneven β set
    n_orb = norb
    if wide_orbital is not None:
        # Relabel the last orbital past bit 64 so the strings need two limbs.
        last = norb - 1
        n_orb = wide_orbital + 1
        index = np.array(list(range(last)) + [wide_orbital])
        big_h, big_eri = np.zeros((n_orb, n_orb)), np.zeros((n_orb,) * 4)
        big_h[np.ix_(index, index)] = h1
        big_eri[np.ix_(index, index, index, index)] = eri
        h1, eri = big_h, big_eri

        def move(s: int) -> int:
            return (s & ~(1 << last)) | ((1 << wide_orbital) if (s >> last) & 1 else 0)

        sa, sb = sorted(move(s) for s in sa), sorted(move(s) for s in sb)
    integrals = to_torch_integrals(
        *setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0), device=device, dtype=dtype
    )
    helpers = generate_all_torch_helpers(sa, sb, n_orb, device=device)
    if wide_orbital is not None:
        adets = torch.tensor(_as_string_limbs(sa, n_orb).view(np.int64), device=device)
        bdets = torch.tensor(_as_string_limbs(sb, n_orb).view(np.int64), device=device)
    else:
        adets, bdets = torch.tensor(sa, device=device), torch.tensor(sb, device=device)
    helper = next(h for h in helpers if h.task_type == 0)
    m_a, m_b = len(sa), len(sb)
    x = torch.tensor(
        np.random.default_rng(5).standard_normal(m_a * m_b), dtype=dtype, device=device
    )
    direct = torch.zeros_like(x)
    _mult_alphabeta_torch(direct, x, adets, bdets, integrals, helper, m_a, m_b, 1 << 20)
    factored = torch.zeros_like(x)
    ab_pass = FactorizedAlphaBeta(
        adets, bdets, n_orb, helper, *factorize_eri(torch.tensor(eri)), dtype=dtype
    )
    ab_pass(factored, x)
    return float((factored - direct).abs().max()), float(direct.abs().max())


@pytest.mark.parametrize(("norb", "na", "nb"), [(6, 3, 2), (7, 3, 3), (8, 2, 3)])
def test_factorized_pass_equals_direct_kernel(norb: int, na: int, nb: int) -> None:
    """The factorized αβ pass reproduces the direct kernel to round-off."""
    err, scale = _pass_vs_direct(norb, na, nb)
    assert scale > 1.0
    assert err < 1e-12 * scale


def test_factorized_pass_on_strings_wider_than_64_bits() -> None:
    """Limb strings take the same factorized pass."""
    err, scale = _pass_vs_direct(6, 2, 3, wide_orbital=69)
    assert err < 1e-12 * scale


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device available")
def test_factorized_pass_on_mps() -> None:
    """MPS (float32, the index_add fallback) matches the direct kernel."""
    err, scale = _pass_vs_direct(7, 3, 3, device="mps", dtype=torch.float32)
    assert err < 1e-5 * scale


def test_build_matvec_with_factors_equals_the_direct_matvec() -> None:
    """The full matvec with the factorized αβ pass equals the direct one."""
    norb = 6
    h1, eri = _psd_system(norb, rank=8)
    strings = _strings(norb, 3)
    integrals = to_torch_integrals(*setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0))
    helpers = generate_all_torch_helpers(strings, strings, norb, device="cpu")
    dets = torch.tensor(strings)
    hii = make_diag_terms_torch(dets, dets, norb, integrals)
    args = ("batched", hii, dets, dets, norb, integrals, helpers)
    direct = build_matvec(*args, chunk_size=1 << 20)
    factored = build_matvec(*args, chunk_size=1 << 20, ab_factors=factorize_eri(torch.tensor(eri)))
    x = torch.tensor(np.random.default_rng(2).standard_normal(len(strings) ** 2))
    torch.testing.assert_close(factored(x), direct(x), atol=1e-11, rtol=0)


def _solve(system, factorization: str, **kwargs):
    """Ground state (and solver) of the 6-orbital (3, 3) product basis."""
    h1, eri = system
    strings = np.array(_strings(6, 3), dtype=np.int64)
    kwargs.setdefault("apply_ab_projection", False)
    solver = make_torch_sci_solver(
        device="cpu",
        davidson_tolerance=1e-10,
        max_davidson_iters=100,
        ab_factorization=factorization,
        **kwargs,
    )
    result = solver([(strings, strings)], h1, eri, 6, (3, 3))
    return result[0].states, solver


@pytest.mark.parametrize("system", ["psd", "indefinite"])
@pytest.mark.parametrize(
    "extra",
    [
        {"compute_full_rdm": True, "compute_spin_squared": True},
        {"apply_ab_projection": True},
        {"n_roots": 3, "davidson_block_size": 12},
    ],
    ids=["rdm_s2", "projection", "multiroot"],
)
def test_solver_energies_match_the_direct_channel(system: str, extra: dict) -> None:
    """``ab_factorization="cholesky"`` reproduces the direct solve."""
    integrals = _psd_system(6, rank=10) if system == "psd" else _random_system(6)
    off, _ = _solve(integrals, "off", **extra)
    on, solver = _solve(integrals, "cholesky", **extra)
    for a, b in zip(on, off):
        assert float(a.energy) == pytest.approx(float(b.energy), abs=1e-9)
    if off[0].rdm1 is not None:
        np.testing.assert_allclose(np.asarray(on[0].rdm1), np.asarray(off[0].rdm1), atol=1e-8)
    assert solver.config["resolved"]["ab_factorization"] == "cholesky"


def test_spin_penalty_with_factorized_channel() -> None:
    """The S² penalty composes with the factorized matvec."""
    h1, eri = _psd_system(6, rank=10)
    strings = np.array(_strings(6, 3), dtype=np.int64)
    energies = []
    for factorization in ("off", "cholesky"):
        solver = make_torch_sci_solver(
            device="cpu",
            davidson_tolerance=1e-10,
            max_davidson_iters=100,
            apply_ab_projection=False,
            ab_factorization=factorization,
        )
        state = solver([(strings, strings)], h1, eri, 6, (3, 3), spin_sq=0.0)[0].states[0]
        energies.append(float(state.energy))
    assert energies[1] == pytest.approx(energies[0], abs=1e-9)


def test_factors_are_computed_once_per_hamiltonian(monkeypatch) -> None:
    """A second call with equal integrals (even a copy) reuses the factors."""
    import torch_sci_solver.torch_impl.ab_factorized as module

    calls = []
    original = module.factorize_eri

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "factorize_eri", counting)
    h1, eri = _psd_system(6, rank=6)
    strings = np.array(_strings(6, 3), dtype=np.int64)
    solver = make_torch_sci_solver(
        device="cpu", apply_ab_projection=False, ab_factorization="cholesky"
    )
    solver([(strings, strings)], h1, eri, 6, (3, 3))
    solver([(strings, strings)], h1, eri.copy(), 6, (3, 3))
    assert len(calls) == 1


def test_configurations_the_factorized_pass_does_not_serve_are_rejected() -> None:
    """GDB, complex dtypes, distributed=, spinless calls and bad values raise."""
    with pytest.raises(ValueError, match="mode='gdb'"):
        make_torch_sci_solver(mode="gdb", ab_factorization="cholesky")
    with pytest.raises(ValueError, match="real integrals"):
        make_torch_sci_solver(dtype=torch.complex128, ab_factorization="cholesky")
    with pytest.raises(ValueError, match="distributed="):
        make_torch_sci_solver(
            ab_factorization="cholesky", distributed={"world_size": 2, "backend": "gloo"}
        )
    with pytest.raises(ValueError, match="ab_factorization must be one of"):
        make_torch_sci_solver(ab_factorization="df")
    with pytest.raises(ValueError, match="ab_factorization_tol"):
        make_torch_sci_solver(ab_factorization="cholesky", ab_factorization_tol=0.0)
    h1, eri = _random_system(6)
    solver = make_torch_sci_solver(ab_factorization="cholesky")
    with pytest.raises(ValueError, match="spinless"):
        solver([np.array(_strings(6, 3))], h1, eri, 6, 3)


def test_capabilities_report_the_factorized_channel() -> None:
    """``capabilities()`` lists the factorization modes per path."""
    caps = make_torch_sci_solver(ab_factorization="cholesky").capabilities()
    assert caps["ab_factorization"] == "cholesky"
    assert caps["ab_factorization_modes"] == {
        "tpb": ["off", "cholesky"],
        "gdb": ["off"],
        "spinless": ["off"],
    }


def test_factorized_pass_takes_a_float32_input() -> None:
    """A narrow (float32) Davidson input promotes against float64 factors."""
    norb = 6
    h1, eri = _random_system(norb, seed=3)
    strings = _strings(norb, 3)
    integrals = to_torch_integrals(*setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0))
    helpers = generate_all_torch_helpers(strings, strings, norb, device="cpu")
    dets = torch.tensor(strings)
    helper = next(h for h in helpers if h.task_type == 0)
    m = len(strings)
    x = torch.tensor(np.random.default_rng(1).standard_normal(m * m))
    direct = torch.zeros_like(x)
    _mult_alphabeta_torch(direct, x, dets, dets, integrals, helper, m, m, 1 << 20)
    ab_pass = FactorizedAlphaBeta(
        dets, dets, norb, helper, *factorize_eri(torch.tensor(eri)), dtype=torch.float64
    )
    out = torch.zeros_like(x)
    ab_pass(out, x.to(torch.float32))
    torch.testing.assert_close(out, direct, atol=1e-5 * float(direct.abs().max()), rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device available")
def test_factorized_pass_on_mps_with_wide_strings() -> None:
    """MPS, strings past bit 64: the chunked fallback matches the direct kernel."""
    err, scale = _pass_vs_direct(6, 2, 3, device="mps", dtype=torch.float32, wide_orbital=69)
    assert err < 1e-5 * scale


def test_packed_eri_is_unpacked_before_factorizing() -> None:
    """A PySCF 4-fold packed ERI solves like the full tensor."""
    ao2mo = pytest.importorskip("pyscf.ao2mo")
    h1, eri = _psd_system(6, rank=8)
    packed = ao2mo.restore(4, eri, 6)
    full, _ = _solve((h1, eri), "cholesky")
    packed_states, _ = _solve((h1, packed), "cholesky")
    assert float(packed_states[0].energy) == pytest.approx(float(full[0].energy), abs=1e-10)


def test_new_eri_content_refactorizes() -> None:
    """Different integrals in a new array are factorized again, not reused."""
    h1, eri = _psd_system(6, rank=8)
    strings = np.array(_strings(6, 3), dtype=np.int64)
    solver = make_torch_sci_solver(
        device="cpu",
        davidson_tolerance=1e-10,
        apply_ab_projection=False,
        ab_factorization="cholesky",
    )
    solver([(strings, strings)], h1, eri, 6, (3, 3))
    changed_eri = 1.5 * eri
    changed = solver([(strings, strings)], h1, changed_eri, 6, (3, 3))[0].states[0]
    reference = make_torch_sci_solver(
        device="cpu", davidson_tolerance=1e-10, apply_ab_projection=False
    )([(strings, strings)], h1, changed_eri, 6, (3, 3))[0].states[0]
    assert float(changed.energy) == pytest.approx(float(reference.energy), abs=1e-9)


def test_integral_program_asymmetry_is_accepted() -> None:
    """A ~1e-12 (pq)<->(rs) asymmetry (as integral programs leave) is symmetrized."""
    _, eri = _psd_system(5, rank=6)
    noisy = eri + 4e-12 * np.random.default_rng(0).standard_normal(eri.shape)
    factors, signs = factorize_eri(torch.tensor(noisy))
    symmetric = 0.5 * (noisy.reshape(25, 25) + noisy.reshape(25, 25).T)
    rebuilt = (factors.T * signs) @ factors
    torch.testing.assert_close(rebuilt, torch.tensor(symmetric), atol=1e-9, rtol=0)


def test_near_semidefinite_input_never_needs_eigh(monkeypatch) -> None:
    """Round-off-level negative directions are absorbed by signed pivots."""
    _, eri = _psd_system(5, rank=6)
    noise = np.random.default_rng(1).standard_normal((25, 25))
    noisy = eri.reshape(25, 25) + 1e-11 * 0.5 * (noise + noise.T)

    def no_eigh(*_args, **_kwargs):
        raise AssertionError("the eigendecomposition fallback ran")

    monkeypatch.setattr(torch.linalg, "eigh", no_eigh)
    factors, signs = factorize_eri(torch.tensor(noisy.reshape(5, 5, 5, 5)), tol=1e-13)
    rebuilt = (factors.T * signs) @ factors
    torch.testing.assert_close(rebuilt, torch.tensor(noisy), atol=1e-12, rtol=0)
