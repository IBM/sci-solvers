"""Warm-start probe states on the flat spin-orbital ("spinless") path.

Covers the ``probe_method="warm_start"`` round trip end to end:

1. cold-with-probe returns ``(results, states)`` and a warm extension
   reproduces the cold energy on the flat layout, for real and complex
   integrals;
2. the cached state pickles and its ``describe()`` is JSON-friendly;
3. a cached basis that is not a subset of the new basis raises;
4. a product-mode ``TorchSCIState`` is rejected with a type error;
5. determinants wider than 64 bits embed by limb-aware string lookup;
6. a changed Hamiltonian fingerprint raises, and a soft Davidson knob
   mismatch warns;
7. ``preflight`` and ``capabilities()`` report the warm/probe support and the
   ``spin_sq`` / ``compute_spin_squared`` physics limitation;
8. a 2-rank gloo run warms on both the contiguous and tiled owner-computes
   layouts, real and complex.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
import pickle

import numpy as np
import pytest
import torch
from pyscf.fci import cistring
from torch_sci_solver.torch_impl.probe_spinless import (
    TorchSpinlessSCIState,
    _cached_positions,
    _dedup_and_check_spinless_superset,
    _embed_eigvec_spinless,
)
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver
from torch_sci_solver.torch_impl.spinless_helpers import (
    dets_to_python_ints,
    spinless_basis_from_array,
)

SOLVER_KWARGS = dict(device="cpu", davidson_block_size=4, davidson_tolerance=1e-9)


def _real_spinorbital_integrals(nmo: int, seed: int):
    """Real symmetric hcore + a chemist-symmetric eri."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((nmo, nmo))
    hcore = 0.5 * (hcore + hcore.T)
    raw = rng.standard_normal((nmo,) * 4)
    eri = np.zeros_like(raw)
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
        eri += raw.transpose(perm)
    eri /= 8.0
    return hcore, eri


def _complex_spinorbital_integrals(nmo: int, seed: int):
    """Complex Hermitian hcore + a physically valid complex eri."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    hcore = hc + hc.conj().T
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.complex128)
    for _ in range(nmo + 2):
        g = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _integrals(kind: str, nmo: int, seed: int):
    if kind == "real":
        return (*_real_spinorbital_integrals(nmo, seed), torch.float64)
    return (*_complex_spinorbital_integrals(nmo, seed), torch.complex128)


def _strings(nmo: int, n_particles: int) -> np.ndarray:
    return np.asarray(cistring.make_strings(range(nmo), n_particles))


# --------------------------------------------------------------------------- #
# Flat layout: cold-with-probe -> warm extension                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_flat_warm_extend_reproduces_cold_energy(kind: str) -> None:
    """The cached ground state seeds the same basis to the cold eigenvalue."""
    nmo, n, seed = 5, 2, 42
    h1, eri, dtype = _integrals(kind, nmo, seed)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    cold = solver([strs], h1, eri, nmo, n)[0].states[0]

    results, states = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    assert results[0].states[0].energy == pytest.approx(cold.energy, abs=1e-10)
    state = states[0]
    assert isinstance(state, TorchSpinlessSCIState)
    assert state.n_basis == len(strs)
    assert state.n_orb == nmo
    assert state.n_particles == n
    assert state.energy == pytest.approx(cold.energy, abs=1e-10)
    assert state.n_matvecs is not None

    # The snapshot is a plain dataclass + CPU tensor: pickle-safe, and its
    # description survives json.dumps.
    clone = pickle.loads(pickle.dumps(state))
    assert clone.dets == state.dets
    assert torch.equal(clone.eigvec_inter, state.eigvec_inter)
    assert json.loads(json.dumps(state.describe()))["type"] == "TorchSpinlessSCIState"

    warm = solver([strs], h1, eri, nmo, n, probe_method="warm_start", probe_state=clone)[0].states[
        0
    ]
    assert warm.energy == pytest.approx(cold.energy, abs=1e-10)
    assert warm.state == "converged"


@pytest.mark.parametrize("kind", ["real", "complex"])
def test_flat_warm_extension_with_added_determinants(kind: str) -> None:
    """A strict-extension basis converges to its own cold ground state."""
    nmo, n, seed = 6, 3, 52
    h1, eri, dtype = _integrals(kind, nmo, seed)
    full = _strings(nmo, n)  # C(6, 3) = 20 determinants
    core = full[:-3]
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    reference = solver([full], h1, eri, nmo, n)[0].states[0]
    _, (state,) = solver([core], h1, eri, nmo, n, probe_method="warm_start")
    warm = solver([full], h1, eri, nmo, n, probe_method="warm_start", probe_state=state)[0].states[
        0
    ]
    assert warm.energy == pytest.approx(reference.energy, abs=1e-7)


def test_warm_extend_can_return_a_chained_state() -> None:
    """``build_output_state`` opts a warm extension back into ``(results, states)``."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("complex", nmo, 43)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    results, (chained,) = solver(
        [strs],
        h1,
        eri,
        nmo,
        n,
        probe_method="warm_start",
        probe_state=state,
        probe_kwargs={"build_output_state": True},
    )
    assert isinstance(chained, TorchSpinlessSCIState)
    assert chained.energy == pytest.approx(results[0].states[0].energy, abs=1e-10)


def test_warm_extend_dedups_shuffled_caller_strings() -> None:
    """Caller order and duplicate determinants do not change the warm start."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("complex", nmo, 51)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    cold = solver([strs], h1, eri, nmo, n)[0].states[0]
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    caller = np.concatenate([strs[::-1], strs[:3]])
    with pytest.warns(UserWarning, match="duplicate"):
        warm = solver([caller], h1, eri, nmo, n, probe_method="warm_start", probe_state=state)[
            0
        ].states[0]
    assert warm.energy == pytest.approx(cold.energy, abs=1e-10)


def test_warm_extend_rejects_non_superset() -> None:
    """A cached determinant the caller dropped is an error, not a silent loss."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("real", nmo, 44)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    with pytest.raises(ValueError, match="superset"):
        solver(
            [strs[:-1]],
            h1,
            eri,
            nmo,
            n,
            probe_method="warm_start",
            probe_state=state,
        )


def test_product_mode_state_is_rejected() -> None:
    """The three cached-state types are not interchangeable."""
    nmo, n = 4, 2
    h1, eri, _ = _integrals("real", nmo, 45)
    one_strings = _strings(nmo, 1)
    tpb = make_torch_sci_solver(device="cpu", dtype=torch.float64)
    _, tpb_states = tpb(
        [(one_strings, one_strings)],
        h1,
        eri,
        nmo,
        (1, 1),
        probe_method="warm_start",
    )
    spinless = make_torch_sci_solver(device="cpu", dtype=torch.float64)
    with pytest.raises(TypeError, match="TorchSpinlessSCIState"):
        spinless(
            [_strings(nmo, n)],
            h1,
            eri,
            nmo,
            n,
            probe_method="warm_start",
            probe_state=tpb_states[0],
        )


def test_zero_cached_eigenvector_falls_back_to_the_cold_seed() -> None:
    """A cache with no surviving amplitude must not hand Davidson a zero vector."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("complex", nmo, 46)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    cold = solver([strs], h1, eri, nmo, n)[0].states[0]
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    state.eigvec_inter = torch.zeros_like(state.eigvec_inter)
    warm = solver([strs], h1, eri, nmo, n, probe_method="warm_start", probe_state=state)[0].states[
        0
    ]
    # The fallback reuses the deterministic cold seed, so the solve is a cold
    # solve in every respect.
    assert warm.energy == cold.energy


# --------------------------------------------------------------------------- #
# Wide determinants (>64 spin-orbitals)                                        #
# --------------------------------------------------------------------------- #
def test_wide_embed_uses_limb_aware_string_lookup() -> None:
    """Cached amplitudes follow their determinant strings across limbs."""
    nmo = 68
    cached = [
        (1 << 0) | (1 << 1),
        (1 << 64) | (1 << 0),
        (1 << 63) | (1 << 2),
        (1 << 67) | (1 << 66),
    ]
    added = (1 << 65) | (1 << 1)
    extended = cached + [added]
    vec = torch.arange(1, len(cached) + 1, dtype=torch.float64)

    positions = _cached_positions(cached, extended, nmo)
    np.testing.assert_array_equal(positions, [0, 1, 2, 3, -1])
    embedded = _embed_eigvec_spinless(
        vec, cached, extended, nmo, device="cpu", dtype=torch.float64, normalize=False
    )
    np.testing.assert_allclose(embedded.numpy(), [1.0, 2.0, 3.0, 4.0, 0.0])
    normalized = _embed_eigvec_spinless(
        vec, cached, extended, nmo, device="cpu", dtype=torch.float64
    )
    assert float(torch.linalg.norm(normalized)) == pytest.approx(1.0)

    # The union helper uses the same limb order as the engine basis.
    union, info = _dedup_and_check_spinless_superset(cached, extended, nmo)
    basis = spinless_basis_from_array(union, nmo, 2)
    assert union == [int(det) for det in dets_to_python_ints(basis.dets)]
    assert info == {
        "dets_added": 1,
        "dets_in_input_overlapping_core": len(cached),
        "dets_dup_in_input": 0,
        "cached_N": len(cached),
        "extended_N": len(extended),
    }


def test_wide_warm_extend_end_to_end() -> None:
    """A >64 spin-orbital basis warms from a limb-string cached state."""
    nmo, n = 65, 2
    h1 = np.diag(np.arange(nmo, dtype=np.float64))
    eri = np.zeros((nmo, nmo, nmo, nmo), dtype=np.float64)
    cached = np.array(
        [
            (1 << 0) | (1 << 1),
            (1 << 0) | (1 << 64),
            (1 << 63) | (1 << 1),
            (1 << 64) | (1 << 63),
        ],
        dtype=object,
    )
    added = np.array([(1 << 2) | (1 << 3)], dtype=object)
    extended = np.concatenate([cached, added])
    solver = make_torch_sci_solver(
        device="cpu", dtype=torch.float64, davidson_block_size=2, davidson_tolerance=1e-9
    )
    _, (state,) = solver([cached], h1, eri, nmo, n, probe_method="warm_start")
    assert state.n_orb == nmo
    np.testing.assert_array_equal(np.sort(state.dets), np.sort([int(x) for x in cached]))
    cold_extended = solver([extended], h1, eri, nmo, n)[0].states[0]
    warm_extended = solver(
        [extended], h1, eri, nmo, n, probe_method="warm_start", probe_state=state
    )[0].states[0]
    assert warm_extended.energy == pytest.approx(cold_extended.energy, abs=1e-10)


# --------------------------------------------------------------------------- #
# Fingerprint and knob validation                                              #
# --------------------------------------------------------------------------- #
def test_warm_extend_rejects_changed_hamiltonian() -> None:
    """A different Hamiltonian invalidates the exact-extension claim."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("real", nmo, 47)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    h1_changed = h1.copy()
    h1_changed[0, 0] += 1e-3
    with pytest.raises(ValueError, match="fingerprint"):
        solver(
            [strs],
            h1_changed,
            eri,
            nmo,
            n,
            probe_method="warm_start",
            probe_state=state,
        )


def test_warm_extend_warns_on_soft_knob_mismatch() -> None:
    """Davidson knobs do not change the operator, so a mismatch warns."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("real", nmo, 48)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    state.solver_knobs["davidson_tolerance"] = 1e-3
    with pytest.warns(UserWarning, match="davidson_tolerance"):
        solver(
            [strs],
            h1,
            eri,
            nmo,
            n,
            probe_method="warm_start",
            probe_state=state,
            tol=1e-9,
        )


# --------------------------------------------------------------------------- #
# preflight / capabilities                                                     #
# --------------------------------------------------------------------------- #
def test_capabilities_report_warm_start_and_spin_limitation() -> None:
    """Discovery reports warm/probe support and why S^2 is unavailable."""
    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128)
    spinless = solver.capabilities()["modes"]["spinless"]
    assert spinless["warm_start"] is True
    assert spinless["probe_state_type"] == "TorchSpinlessSCIState"
    assert spinless["warm_start_hf_augmentation"] is False
    assert spinless["spin_squared"] is False
    assert spinless["spin_penalty"] is False
    assert "alpha/beta" in spinless["spin_squared_unsupported_reason"]
    assert "alpha/beta" in spinless["spin_penalty_unsupported_reason"]


def test_preflight_reports_spin_limitation_and_warm_plan() -> None:
    """The preflight plan names the cached-state type and the physics limit."""
    nmo, n = 5, 2
    h1, eri, dtype = _integrals("real", nmo, 49)
    strs = _strings(nmo, n)
    solver = make_torch_sci_solver(dtype=dtype, **SOLVER_KWARGS)
    plan = solver.preflight([strs], n, norb=nmo)
    assert plan["probe_state_type"] == "TorchSpinlessSCIState"
    assert plan["spin_sq"] is None
    assert plan["compute_spin_squared"] is False
    assert "alpha/beta" in plan["spin_sq_unsupported_reason"]
    assert "alpha/beta" in plan["compute_spin_squared_unsupported_reason"]
    assert plan["warm_start"] is False
    assert plan["build_output_states"] is False
    assert plan["subspaces"][0]["warm_source"] is None

    cold_probe_plan = solver.preflight([strs], n, norb=nmo, probe_method="warm_start")
    assert cold_probe_plan["warm_start"] is False
    assert cold_probe_plan["build_output_states"] is True

    with pytest.raises(ValueError, match="spin_sq is not supported"):
        solver.preflight([strs], n, norb=nmo, spin_sq=0.0)

    _, (state,) = solver([strs], h1, eri, nmo, n, probe_method="warm_start")
    warm_plan = solver.preflight(
        [strs],
        n,
        norb=nmo,
        probe_method="warm_start",
        probe_state=state,
        one_body_tensor=h1,
        two_body_tensor=eri,
    )
    assert warm_plan["warm_start"] is True
    assert warm_plan["build_output_states"] is False
    assert warm_plan["subspaces"][0]["start"] == "warm"
    assert warm_plan["subspaces"][0]["warm_source"] == "probe_state"

    with pytest.raises(TypeError, match="TorchSpinlessSCIState"):
        solver.preflight([strs], n, norb=nmo, probe_method="warm_start", probe_state=[object()])


def test_preflight_rejects_compute_spin_squared() -> None:
    """compute_spin_squared is a documented physics limitation, refused up front."""
    nmo, n = 4, 2
    solver = make_torch_sci_solver(device="cpu", dtype=torch.complex128, compute_spin_squared=True)
    with pytest.raises(ValueError, match="compute_spin_squared is not supported"):
        solver.preflight([_strings(nmo, n)], n, norb=nmo)


# --------------------------------------------------------------------------- #
# Distributed owner-computes layouts (2-rank gloo)                             #
# --------------------------------------------------------------------------- #
def _distributed_kwargs(layout: str) -> dict:
    kwargs = {"world_size": 2, "backend": "gloo", "persistent": False}
    if layout == "tiled":
        kwargs["tile"] = "force"
    return kwargs


@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("layout", ["contiguous", "tiled"])
def test_distributed_warm_extend_matches_cold(kind: str, layout: str) -> None:
    """Both owner-computes layouts embed a cached core into an extended basis."""
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend is required")
    nmo, n, seed = 7, 3, 50
    h1, eri, dtype = _integrals(kind, nmo, seed)
    strs = _strings(nmo, n)  # 35 determinants: above the replicated cap
    core = strs[:-3]  # 32 determinants: still an owner-computes solve
    reference_solver = make_torch_sci_solver(
        device="cpu", dtype=dtype, davidson_block_size=8, davidson_tolerance=1e-9
    )
    reference = reference_solver([strs], h1, eri, nmo, n)[0].states[0]

    solver = make_torch_sci_solver(
        device="cpu",
        dtype=dtype,
        davidson_block_size=8,
        davidson_tolerance=1e-9,
        distributed=_distributed_kwargs(layout),
    )
    try:
        plan = solver.preflight(
            [core],
            n,
            norb=nmo,
            one_body_tensor=h1,
            two_body_tensor=eri,
        )
        assert plan["subspaces"][0]["path"] == "local"
        if layout == "tiled":
            assert plan["subspaces"][0]["tile_grid"] is not None

        cold_results, states = solver([core], h1, eri, nmo, n, probe_method="warm_start")
        assert cold_results[0].states[0].state == "converged"
        state = states[0]
        assert isinstance(state, TorchSpinlessSCIState)
        assert state.n_basis == len(core)

        warm = solver([strs], h1, eri, nmo, n, probe_method="warm_start", probe_state=state)[
            0
        ].states[0]
        assert warm.energy == pytest.approx(reference.energy, abs=1e-7)
    finally:
        solver.close()
        reference_solver.close()
