"""MPS (Apple Silicon) device support.

MPS has no float64 (or complex128) support at all, unlike CUDA where every
CPU op has a straightforward device-templated counterpart. Two spots in the
engine assumed a device-agnostic ``float64``/``complex128`` scratch dtype was
always safe and broke on MPS:

  - ``rdm._inter_to_block_signs`` used ``float64`` to contract an int64
    matmul (CUDA has no native int64 matmul); fixed to ``float32``, which is
    still exact for the bounded values involved (L^2 <= 4096 for L <= 64).
  - ``davidson._davidson_multi``'s final residual-norm step moved a CPU
    ``float64`` energies tensor onto ``device`` without also recasting to
    that dtype's real component; fixed to cast to ``dtype.to_real()`` (or
    ``dtype`` itself when real) before the device move.

A third, distinct gotcha (not a dtype-portability bug, but a usability trap):
Davidson's ``‖R‖ < eps`` stopping criterion is an *absolute* residual, not
scaled to the energy. On MPS/float32 that residual cannot be driven below
roughly ``|E| * finfo(float32).eps`` regardless of ``max_davidson_iters`` --
a ``davidson_tolerance`` below ``qiskit_adapter.MPS_DAVIDSON_TOLERANCE_FLOOR``
(``1e-5``) would silently burn the full iteration budget on every call
without ever converging. ``TorchSCISolver`` clamps up to that floor (both at
construction and on a per-call ``tol=`` override) and warns when it does.

These tests are regression coverage for all three fixes, plus a correctness
sanity check at a larger (norb=60) synthetic subspace. All skip outright when MPS
isn't available (there's no software fallback to test against).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from itertools import combinations, islice

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="no MPS device available"
)


def _strings(norb: int, nocc: int, limit: int) -> np.ndarray:
    """Strings."""
    combos = islice(combinations(range(norb), nocc), limit)
    return np.array([sum(1 << i for i in c) for c in combos], dtype=np.int64)


def _random_system(seed: int, norb: int):
    """Random Hermitian hcore/eri at a given orbital count."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    x = rng.standard_normal((norb,) * 4)
    eri = np.zeros_like(x)
    for perm in [
        (0, 1, 2, 3),
        (1, 0, 2, 3),
        (0, 1, 3, 2),
        (1, 0, 3, 2),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
        (2, 3, 1, 0),
        (3, 2, 1, 0),
    ]:
        eri += x.transpose(perm)
    eri /= 8.0
    return hcore, eri


def test_inter_to_block_signs_matches_cpu_on_mps():
    """The float32-on-MPS scratch dtype in _inter_to_block_signs must give the
    same +-1 signs as the CPU float64 path (regression for the rdm.py fix)."""
    from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs

    rng = np.random.default_rng(0)
    norb = 20
    adets_np = rng.integers(0, 2**norb, size=50)
    bdets_np = rng.integers(0, 2**norb, size=50)

    adets_cpu = torch.tensor(adets_np, dtype=torch.int64, device="cpu")
    bdets_cpu = torch.tensor(bdets_np, dtype=torch.int64, device="cpu")
    sigma_cpu = _inter_to_block_signs(adets_cpu, bdets_cpu, norb)

    adets_mps = torch.tensor(adets_np, dtype=torch.int64, device="mps")
    bdets_mps = torch.tensor(bdets_np, dtype=torch.int64, device="mps")
    sigma_mps = _inter_to_block_signs(adets_mps, bdets_mps, norb)

    assert torch.equal(sigma_cpu, sigma_mps.cpu().to(sigma_cpu.dtype))


@pytest.mark.parametrize("call_kwarg", [False, True])
def test_davidson_tolerance_clamped_on_mps(call_kwarg):
    """A davidson_tolerance below MPS_DAVIDSON_TOLERANCE_FLOOR is overridden
    (not just warned about) on device='mps', so a solver constructed with
    the package default (1e-7, well below the floor) still converges rather
    than silently burning the full iteration budget every call. Checked both
    at construction time and via the per-call ``tol=`` override, since the
    override reads its own kwarg independently of the constructor value."""
    from torch_sci_solver import make_torch_sci_solver
    from torch_sci_solver.torch_impl.qiskit_adapter import MPS_DAVIDSON_TOLERANCE_FLOOR

    too_tight = MPS_DAVIDSON_TOLERANCE_FLOOR / 100

    if call_kwarg:
        solver = make_torch_sci_solver(device="mps", dtype=torch.float32)
        call_kwargs = {"tol": too_tight}
    else:
        with pytest.warns(RuntimeWarning, match="davidson_tolerance"):
            solver = make_torch_sci_solver(
                device="mps", dtype=torch.float32, davidson_tolerance=too_tight
            )
        assert solver.davidson_tolerance == MPS_DAVIDSON_TOLERANCE_FLOOR
        call_kwargs = {}

    norb, na, nb = 10, 2, 2
    hcore, eri = _random_system(seed=0, norb=norb)
    adets = _strings(norb, na, limit=10**9)
    bdets = _strings(norb, nb, limit=10**9)

    if call_kwarg:
        with pytest.warns(RuntimeWarning, match="davidson_tolerance"):
            result = solver([(adets, bdets)], hcore, eri, norb, (na, nb), **call_kwargs)[0].states[
                0
            ]
    else:
        result = solver([(adets, bdets)], hcore, eri, norb, (na, nb))[0].states[0]

    assert result.state == "converged"


def test_solve_fermion_h2_on_mps_matches_cpu_fci():
    """End-to-end solve on MPS/float32 reproduces the CPU/float64 FCI energy
    for H2 to float32 precision (regression for the davidson.py fix).

    Uses make_torch_sci_solver directly (rather than solve_fermion) so the
    convergence flag is visible: an explicit, energy-scaled
    davidson_tolerance is required here since the package default (1e-7) is
    below float32 machine epsilon and would leave Davidson never actually
    converging (see qiskit_adapter.py's MPS tolerance warning)."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    from pyscf import ao2mo, fci, gto, scf
    from torch_sci_solver import make_torch_sci_solver

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    n_a, n_b = mol.nelec
    e_fci = fci.FCI(mf).kernel()[0]

    ci_a = _strings(norb, n_a, limit=10**9)
    ci_b = _strings(norb, n_b, limit=10**9)

    solver = make_torch_sci_solver(device="mps", dtype=torch.float32, davidson_tolerance=1e-5)
    result = solver([(ci_a, ci_b)], h1, eri, norb, (n_a, n_b))[0].states[0]

    assert result.state == "converged"
    assert result.energy + e_core == pytest.approx(e_fci, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("k", [20, 50])
def test_solve_fermion_norb60_mps_matches_cpu(k):
    """Larger synthetic subspace (norb=60) sanity check: MPS/float32 tracks
    CPU/float64 within float32 rounding noise as the subspace grows.

    Both runs must actually converge for the comparison to mean anything —
    see the MPS tolerance warning in qiskit_adapter.py for why an unscaled
    davidson_tolerance can silently never converge on float32."""
    from torch_sci_solver import make_torch_sci_solver

    norb, na, nb = 60, 6, 6
    hcore, eri = _random_system(seed=0, norb=norb)
    adets = _strings(norb, na, k)
    bdets = _strings(norb, nb, k)
    nelec = (na, nb)

    cpu_solver = make_torch_sci_solver(device="cpu", dtype=torch.float64)
    r_cpu = cpu_solver([(adets, bdets)], hcore, eri, norb, nelec)[0].states[0]
    assert r_cpu.state == "converged"

    # Scale the MPS tolerance to this system's own energy magnitude, with a
    # 10x safety margin above the bare float32 floor (|E| * finfo(float32).eps)
    mps_tolerance = max(1e-6, abs(r_cpu.energy) * torch.finfo(torch.float32).eps * 10.0)
    mps_solver = make_torch_sci_solver(
        device="mps", dtype=torch.float32, davidson_tolerance=mps_tolerance
    )
    r_mps = mps_solver([(adets, bdets)], hcore, eri, norb, nelec)[0].states[0]
    assert r_mps.state == "converged"

    assert r_mps.energy == pytest.approx(r_cpu.energy, abs=5e-5)


# --------------------------------------------------------------------------- #
# Feature parity on MPS: every single-process path against CPU fp64
# --------------------------------------------------------------------------- #
_NORB, _NA, _NB = 6, 3, 3
_SPINORB, _N_SPINLESS = 8, 3
_WIDE_NORB = 66
_MPS_TOL = 1e-5


def _solver(device: str, **kwargs):
    """A solver on ``device``: float32/complex64 on MPS, float64/complex128 on CPU."""
    from torch_sci_solver import make_torch_sci_solver

    complex_values = kwargs.pop("complex_values", False)
    if device == "mps":
        dtype = torch.complex64 if complex_values else torch.float32
    else:
        dtype = torch.complex128 if complex_values else torch.float64
    kwargs.setdefault("apply_ab_projection", False)
    return make_torch_sci_solver(
        device=device,
        dtype=dtype,
        davidson_tolerance=_MPS_TOL if device == "mps" else 1e-10,
        max_davidson_iters=100,
        **kwargs,
    )


def _complex_spatial_system(norb: int, seed: int):
    """Complex Hermitian ``hcore`` and an ERI with ``(pq|rs) = (rs|pq) = (qp|sr)*``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = 0.5 * (a + a.conj().T)
    eri = np.zeros((norb,) * 4, dtype=np.complex128)
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        factor = g + g.conj().T
        eri += 0.05 * np.einsum("pq,rs->pqrs", factor, factor.conj())
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hcore, eri


def _wide_system():
    """A 4-orbital problem embedded in 66 orbitals with strings past bit 64.

    Orbitals 64/65 carry only a one-body energy, so the low-orbital ground state
    is exact while the basis spans two limbs.
    """
    h_low, eri_low = _random_system(seed=4, norb=4)
    hcore = np.zeros((_WIDE_NORB, _WIDE_NORB))
    hcore[:4, :4] = h_low
    hcore[64, 64] = hcore[65, 65] = 3.0
    eri = np.zeros((_WIDE_NORB,) * 4)
    eri[:4, :4, :4, :4] = eri_low
    strings = [int(s) for s in _strings(4, 2, limit=10**9)]
    strings += [(1 << 0) | (1 << 64), (1 << 64) | (1 << 65)]
    return hcore, eri, sorted(strings)


def _product(strings_a, strings_b):
    """The Cartesian pair list of two string sets, as object arrays (wide-safe)."""
    a = np.array(list(strings_a), dtype=object)
    b = np.array(list(strings_b), dtype=object)
    return np.repeat(a, len(b)), np.tile(b, len(a))


def _first(result):
    """The ground-state record of a (results[, states]) solver return."""
    if isinstance(result, tuple):
        result = result[0]
    return result[0].states[0]


def _tpb(device, **kw):
    """TPB full-CI solve with RDMs, S², the spin penalty and the projection."""
    hcore, eri = _random_system(seed=0, norb=_NORB)
    ci = [(_strings(_NORB, _NA, 10**9), _strings(_NORB, _NB, 10**9))]
    solver = _solver(device, compute_full_rdm=True, compute_spin_squared=True, **kw)
    state = _first(solver(ci, hcore, eri, _NORB, (_NA, _NB), spin_sq=0.0))
    return state, solver.stats["spin_squared"][-1]


def _gdb(device, **kw):
    """GDB full-product solve with RDMs and S²."""
    hcore, eri = _random_system(seed=0, norb=_NORB)
    pairs = _product(_strings(_NORB, _NA, 10**9), _strings(_NORB, _NB, 10**9))
    pairs = (pairs[0].astype(np.int64), pairs[1].astype(np.int64))
    solver = _solver(device, mode="gdb", compute_full_rdm=True, compute_spin_squared=True, **kw)
    state = _first(solver([pairs], hcore, eri, _NORB, (_NA, _NB)))
    return state, solver.stats["spin_squared"][-1]


def _spinless(device, complex_values: bool):
    """Spinless solve with the full RDM on real or complex spin-orbital integrals."""
    if complex_values:
        hcore, eri = _complex_spatial_system(_SPINORB, seed=5)
    else:
        hcore, eri = _random_system(seed=5, norb=_SPINORB)
    strings = _strings(_SPINORB, _N_SPINLESS, 10**9)
    solver = _solver(device, complex_values=complex_values, compute_full_rdm=True)
    return _first(solver([strings], hcore, eri, _SPINORB, _N_SPINLESS)), None


def _warm(device, layout: str):
    """A cold solve on a sub-basis, then a warm extension to the full basis."""
    if layout == "spinless":
        hcore, eri = _random_system(seed=5, norb=_SPINORB)
        strings = _strings(_SPINORB, _N_SPINLESS, 10**9)
        small, full, norb, nelec, kw = [strings[:20]], [strings], _SPINORB, _N_SPINLESS, {}
    else:
        hcore, eri = _random_system(seed=0, norb=_NORB)
        sa, sb = _strings(_NORB, _NA, 10**9), _strings(_NORB, _NB, 10**9)
        norb, nelec = _NORB, (_NA, _NB)
        if layout == "tpb":
            small, full, kw = [(sa[:8], sb[:8])], [(sa, sb)], {}
        else:
            pa, pb = _product(sa, sb)
            pa, pb = pa.astype(np.int64), pb.astype(np.int64)
            small, full, kw = [(pa[:120], pb[:120])], [(pa, pb)], {"mode": "gdb"}
    solver = _solver(device, **kw)
    _, states = solver(small, hcore, eri, norb, nelec, probe_method="warm_start")
    warm = solver(full, hcore, eri, norb, nelec, probe_method="warm_start", probe_state=states)
    return _first(warm), None


def _wide(device, mode: str):
    """A cold solve on strings wider than 64 bits."""
    hcore, eri, strings = _wide_system()
    ci = [(strings, strings)] if mode == "tpb" else [_product(strings, strings)]
    return _first(_solver(device, mode=mode)(ci, hcore, eri, _WIDE_NORB, (2, 2))), None


def _complex(device, mode: str):
    """A complex Hermitian solve: complex64 on MPS, complex128 on CPU."""
    norb = 5
    hcore, eri = _complex_spatial_system(norb, seed=1)
    strings = _strings(norb, 2, 10**9)
    ci = [(strings, strings)] if mode == "tpb" else [_product(strings, strings)]
    if mode == "gdb":
        ci = [(ci[0][0].astype(np.int64), ci[0][1].astype(np.int64))]
    solver = _solver(device, mode=mode, complex_values=True)
    return _first(solver(ci, hcore, eri, norb, (2, 2))), None


_FEATURES = {
    "tpb": lambda d: _tpb(d),
    "tpb_projection": lambda d: _tpb(d, apply_ab_projection=True),
    "tpb_ab_factorized": lambda d: _tpb(d, ab_factorization="cholesky"),
    "gdb_streamed": lambda d: _gdb(d, gdb_kwargs={"block_matvec": "off"}),
    "gdb_block": lambda d: _gdb(d, gdb_kwargs={"block_matvec": "on"}),
    "spinless_real": lambda d: _spinless(d, complex_values=False),
    "spinless_complex": lambda d: _spinless(d, complex_values=True),
    "warm_tpb": lambda d: _warm(d, "tpb"),
    "warm_gdb": lambda d: _warm(d, "gdb"),
    "warm_spinless": lambda d: _warm(d, "spinless"),
    "wide_tpb": lambda d: _wide(d, "tpb"),
    "wide_gdb": lambda d: _wide(d, "gdb"),
    "complex_tpb": lambda d: _complex(d, "tpb"),
    "complex_gdb": lambda d: _complex(d, "gdb"),
}


@pytest.mark.parametrize("feature", sorted(_FEATURES))
def test_feature_matches_cpu_on_mps(feature: str) -> None:
    """Each single-process path on MPS (float32/complex64) tracks CPU fp64.

    Energies agree to the float32 residual floor; the 1-RDM, when built, and
    ⟨S²⟩, when requested, agree to float32 rounding.
    """
    state_mps, s2_mps = _FEATURES[feature]("mps")
    state_cpu, s2_cpu = _FEATURES[feature]("cpu")
    assert state_mps.state == "converged"
    assert state_cpu.state == "converged"
    assert float(state_mps.energy) == pytest.approx(float(state_cpu.energy), abs=1e-4)
    if state_cpu.rdm1 is not None:
        np.testing.assert_allclose(
            np.asarray(state_mps.rdm1), np.asarray(state_cpu.rdm1), atol=1e-3, rtol=0
        )
    if s2_cpu is not None:
        assert float(s2_mps) == pytest.approx(float(s2_cpu), abs=1e-3)


# --------------------------------------------------------------------------- #
# Float64 CPU energies for float32 MPS solves
# --------------------------------------------------------------------------- #
def _shifted_system(norb: int, shift: float, seed: int = 0):
    """Random integrals with a large one-body offset, so |E| makes float32 rounding visible."""
    hcore, eri = _random_system(seed=seed, norb=norb)
    return hcore - shift * np.eye(norb), eri


def _cpu_reference(ci, hcore, eri, norb, nelec, **kwargs):
    """The tight float64 CPU energies of the same problem."""
    solver = _solver("cpu", **kwargs)
    return [float(s.energy) for s in solver(ci, hcore, eri, norb, nelec)[0].states]


_REFINED = {
    "tpb": dict(),
    "tpb_penalty": dict(call={"spin_sq": 0.0}),
    # A smaller offset keeps every excited root above the float32 residual floor.
    "tpb_multiroot": dict(n_roots=3, davidson_block_size=12, shift=60.0),
    "gdb": dict(mode="gdb"),
}


@pytest.mark.parametrize("case", sorted(_REFINED))
def test_mps_reports_float64_cpu_energies(case: str) -> None:
    """MPS energies are float64 Rayleigh quotients: float64-accurate, unlike the Ritz values."""
    kwargs = dict(_REFINED[case])
    call = kwargs.pop("call", {})
    hcore, eri = _shifted_system(_NORB, shift=kwargs.pop("shift", 300.0))
    strings = _strings(_NORB, _NA, 10**9)
    ci = [_product(strings, strings)] if kwargs.get("mode") == "gdb" else [(strings, strings)]
    if kwargs.get("mode") == "gdb":
        ci = [(ci[0][0].astype(np.int64), ci[0][1].astype(np.int64))]
    reference = _cpu_reference(ci, hcore, eri, _NORB, (_NA, _NB), **kwargs)
    errors = {}
    for mode in ("auto", "ritz"):
        solver = _solver("mps", davidson_fp32_energy=mode, **kwargs)
        states = solver(ci, hcore, eri, _NORB, (_NA, _NB), **call)[0].states
        errors[mode] = max(abs(float(s.energy) - r) for s, r in zip(states, reference))
        reported = solver.stats["per_call_log"][-1]["subspaces"][0]["davidson_fp32_energy"]
        assert reported == ("rq64" if mode == "auto" else None)
    assert errors["auto"] < 1e-6
    assert errors["ritz"] > 10 * errors["auto"]


@pytest.mark.parametrize("complex_values", [False, True])
def test_mps_spinless_reports_float64_cpu_energies(complex_values: bool) -> None:
    """The spinless path takes the same float64 CPU Rayleigh quotient."""
    if complex_values:
        hcore, eri = _complex_spatial_system(_SPINORB, seed=5)
        hcore = hcore - 300.0 * np.eye(_SPINORB)
    else:
        hcore, eri = _shifted_system(_SPINORB, shift=300.0, seed=5)
    strings = _strings(_SPINORB, _N_SPINLESS, 10**9)
    reference = _cpu_reference(
        [strings], hcore, eri, _SPINORB, _N_SPINLESS, complex_values=complex_values
    )
    solver = _solver("mps", complex_values=complex_values)
    energy = float(solver([strings], hcore, eri, _SPINORB, _N_SPINLESS)[0].states[0].energy)
    assert energy == pytest.approx(reference[0], abs=1e-6)


def test_mps_complex_tpb_reports_complex128_cpu_energies() -> None:
    """A complex64 MPS solve's energy is a complex128 CPU Rayleigh quotient."""
    norb = 5
    hcore, eri = _complex_spatial_system(norb, seed=1)
    hcore = hcore - 300.0 * np.eye(norb)
    strings = _strings(norb, 2, 10**9)
    ci = [(strings, strings)]
    reference = _cpu_reference(ci, hcore, eri, norb, (2, 2), complex_values=True)
    energy = float(
        _solver("mps", complex_values=True)(ci, hcore, eri, norb, (2, 2))[0].states[0].energy
    )
    assert energy == pytest.approx(reference[0], abs=1e-6)


# --------------------------------------------------------------------------- #
# Missed-root check on tall blocks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows, cols", [(50_000, 10), (99_856, 10), (200_000, 12)])
def test_missed_root_span_of_a_tall_block_on_mps(rows: int, cols: int, monkeypatch) -> None:
    """The missed-root check's orthonormal span works where torch's MPS QR aborts.

    torch's MPS ``linalg.qr`` kills the process on these shapes, which the
    10-root missed-root check hits from about 50k determinants up. The CPU-QR
    fallback is disabled here, so a silent fallback — the tall transfer this
    helper exists to avoid — fails the test instead of passing on the other
    path.
    """
    from torch_sci_solver.torch_impl import davidson

    block = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0))
    reference = torch.linalg.qr(block.double())[0]

    def _no_qr(*_args, **_kwargs):
        raise AssertionError("_orthonormal_span fell back to the CPU QR")

    monkeypatch.setattr(davidson.torch.linalg, "qr", _no_qr)
    q = davidson._orthonormal_span(block.to("mps")).cpu().double()
    eye = torch.eye(cols, dtype=torch.float64)
    assert float((q.T @ q - eye).abs().max()) < 1e-5
    assert float((q @ (q.T @ reference) - reference).abs().max()) < 1e-5


def test_mps_multiroot_missed_root_check_matches_cpu() -> None:
    """The missed-root check runs end to end on MPS and keeps the CPU energies.

    Five roots over 4,900 determinants exercise the check's orthonormal span
    on a tall block; at 50k+ determinants that call used to abort inside
    torch's MPS ``linalg.qr``. The float32 tolerance is a reachable one
    (1e-3), so the solve converges rather than stopping at its floor.
    """
    from torch_sci_solver import make_torch_sci_solver

    norb, na, nb = 8, 4, 4
    hcore, eri = _shifted_system(norb, shift=100.0, seed=3)
    strings = _strings(norb, na, 10**9)
    ci = [(strings, strings)]
    common = dict(
        n_roots=5,
        davidson_block_size=16,
        davidson_missed_root_check=True,
        max_davidson_iters=100,
        apply_ab_projection=False,
    )
    cpu = make_torch_sci_solver(
        device="cpu", dtype=torch.float64, davidson_tolerance=1e-10, **common
    )
    reference = [float(s.energy) for s in cpu(ci, hcore, eri, norb, (na, nb))[0].states]
    mps = make_torch_sci_solver(
        device="mps", dtype=torch.float32, davidson_tolerance=1e-3, **common
    )
    states = mps(ci, hcore, eri, norb, (na, nb))[0].states
    errors = [abs(float(s.energy) - r) for s, r in zip(states, reference)]
    assert len(errors) == 5
    # A reachable float32 residual (1e-3) leaves the roots ~‖r‖²/gap off,
    # ~2e-4 Ha on this spectrum; the point is that all five roots match.
    assert max(errors) < 5e-4
    perf = mps.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["missed_root_checks"]  # the check actually ran
