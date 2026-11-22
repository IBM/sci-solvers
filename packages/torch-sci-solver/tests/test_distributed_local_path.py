"""End-to-end checks for the owner-computes distributed TPB path.

These exercise the owner-computes integration through the public solver:
the local αβ-swap projector (no replicated fallback), warm-start seeds, the
fp32 subspace and the missed-root search. They run on CPU gloo with two ranks
and mirror ``test_distributed_solver.py``'s synthetic 4-orbital system.
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
from torch_sci_solver import make_torch_sci_solver

NORB = 4
NELEC = (2, 2)
BLOCK_SIZE = 8
N_ROOTS = 2


def _synthetic_h4():
    """Deterministic 4-orbital / 4-electron integrals and full CI strings."""
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
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
    strings = np.array(
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(NORB), 2)),
        dtype=np.int64,
    )
    return h1, eri, strings


def _solver_kwargs(**overrides):
    """Deterministic settings shared by the reference and distributed runs."""
    kwargs = dict(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=BLOCK_SIZE,
        max_davidson_iters=200,
        davidson_tolerance=1e-8,
        n_roots=N_ROOTS,
        davidson_missed_root_check=False,
        verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def _distributed(**overrides):
    """The two-rank gloo local-mode launch config."""
    return {
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 300,
            "matvec": "local",
            **overrides,
        }
    }


def _skip_without_gloo():
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _energies(solver, h1, eri, strings, **call_kwargs):
    result = solver([(strings, strings)], h1, eri, NORB, NELEC, **call_kwargs)
    return [float(state.energy) for state in result[0].states]


def _last_perf(solver) -> dict:
    return solver.stats["per_call_log"][-1]["subspaces"][0]


def test_projection_stays_on_local_matvec() -> None:
    """The default symmetric basis uses the local projector, not the fallback."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**_solver_kwargs(**_distributed()))
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    assert _last_perf(distributed)["distributed_matvec"] == "local"


def test_warm_init_vector_converges_immediately() -> None:
    """A caller-supplied exact eigenvector warm-starts the local path."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(n_roots=1))
    result = reference([(strings, strings)], h1, eri, NORB, NELEC)
    ground = result[0].states[0]
    start = np.asarray(ground.sci_state.amplitudes, dtype=np.float64).reshape(-1)

    distributed = make_torch_sci_solver(**_solver_kwargs(n_roots=1, **_distributed()))
    observed = _energies(distributed, h1, eri, strings, init_vectors=[start])
    assert observed[0] == pytest.approx(float(ground.energy), abs=1e-8)
    assert _last_perf(distributed)["warm"] is True


def test_zero_init_vector_falls_back_to_cold() -> None:
    """A vector with no overlap with the basis takes the cold path on every rank."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(n_roots=1))
    expected = _energies(reference, h1, eri, strings)
    zero = np.zeros(len(strings) * len(strings), dtype=np.float64)

    distributed = make_torch_sci_solver(**_solver_kwargs(n_roots=1, **_distributed()))
    observed = _energies(distributed, h1, eri, strings, init_vectors=[zero])
    assert observed == pytest.approx(expected, abs=1e-8)
    assert _last_perf(distributed)["warm"] is False


def test_fp32_subspace_matches_fp64() -> None:
    """The fp32 subspace with its global shift reproduces the fp64 energy."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(n_roots=1))
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            n_roots=1,
            davidson_dtype=torch.float32,
            davidson_tolerance=1e-6,
            **_distributed(),
        )
    )
    observed = _energies(distributed, h1, eri, strings)
    assert observed[0] == pytest.approx(expected[0], abs=1e-5)
    perf = _last_perf(distributed)
    assert perf["davidson_dtype"] == "torch.float32"
    assert perf["davidson_fp32_energy"] in ("rq64", "ritz")


def test_missed_root_check_runs_on_local_path() -> None:
    """The multi-root missed-root search is implemented, not rejected."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(davidson_missed_root_check=True, **_distributed())
    )
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    checks = _last_perf(distributed)["missed_root_checks"]
    assert isinstance(checks, list) and len(checks) <= 4
    assert all({"found", "energy", "n_matvecs"} <= set(check) for check in checks)


def test_replicated_path_supports_fp32() -> None:
    """A call forced onto the full-space fallback still gets the fp32 subspace."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            davidson_dtype=torch.float32,
            davidson_tolerance=1e-6,
            distributed={
                "world_size": 2,
                "backend": "gloo",
                "persistent": False,
                "matvec": "replicated",
            },
        )
    )
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-5)
    perf = _last_perf(distributed)
    assert perf["davidson_dtype"] == "torch.float32"
    assert perf["distributed_matvec"] == "replicated"
    assert perf["davidson_fp32_energy"] in ("rq64", "ritz")


# --------------------------------------------------------------------------- #
# GDB (pair mode), complex and spinless on the owner-computes path             #
# --------------------------------------------------------------------------- #
def _pair_strings(strings):
    """The GDB pair list built from a product-mode string list."""
    return np.repeat(strings, len(strings)), np.tile(strings, len(strings))


def _pair_energies(solver, h1, eri, strings, **call_kwargs):
    result = solver([_pair_strings(strings)], h1, eri, NORB, NELEC, **call_kwargs)
    return [float(state.energy) for state in result[0].states]


def _spinless_system(norb: int = 6, n_particles: int = 3):
    """A deterministic real spin-orbital Hamiltonian and its flat strings."""
    rng = np.random.default_rng(11)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
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
    strings = np.array(
        sorted(
            sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), n_particles)
        ),
        dtype=np.int64,
    )
    return h1, eri, strings


def _spinless_energies(solver, h1, eri, strings, n_particles, **call_kwargs):
    result = solver([strings], h1, eri, 6, n_particles, **call_kwargs)
    return [float(state.energy) for state in result[0].states]


def test_gdb_owner_computes_matches_serial() -> None:
    """GDB (pair mode) takes the owner-computes path, not the replicated matvec."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb"))
    expected = _pair_energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**_solver_kwargs(mode="gdb", **_distributed()))
    observed = _pair_energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    perf = _last_perf(distributed)
    assert perf["distributed_matvec"] == "local"
    assert perf["pair_mode"] is True


def test_gdb_fp32_subspace_matches_fp64() -> None:
    """The GDB owner-computes path supports the fp32 subspace."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb"))
    expected = _pair_energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            mode="gdb",
            davidson_dtype=torch.float32,
            davidson_tolerance=1e-6,
            **_distributed(),
        )
    )
    observed = _pair_energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-5)
    perf = _last_perf(distributed)
    assert perf["davidson_dtype"] == "torch.float32"
    assert perf["distributed_matvec"] == "local"


def test_complex_owner_computes_matches_serial() -> None:
    """Complex TPB integrals use the owner-computes path (batched kernels are generic)."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(reference, complex_h1, complex_eri, strings)

    distributed = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128, **_distributed()))
    observed = _energies(distributed, complex_h1, complex_eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    assert _last_perf(distributed)["distributed_matvec"] == "local"


def test_complex64_subspace_matches_complex128() -> None:
    """The complex64 narrow subspace works through the complex owner-computes path."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(reference, complex_h1, complex_eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            davidson_dtype=torch.complex64,
            davidson_tolerance=1e-6,
            **_distributed(),
        )
    )
    observed = _energies(distributed, complex_h1, complex_eri, strings)
    assert observed == pytest.approx(expected, abs=1e-5)
    perf = _last_perf(distributed)
    assert perf["davidson_dtype"] == "torch.complex64"
    assert perf["distributed_matvec"] == "local"


def test_spinless_owner_computes_matches_serial() -> None:
    """A scalar nelec runs the flat owner-computes spinless path on a larger basis."""
    _skip_without_gloo()
    h1, eri, strings = _spinless_system()
    n_particles = 3
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _spinless_energies(reference, h1, eri, strings, n_particles)

    distributed = make_torch_sci_solver(**_solver_kwargs(**_distributed()))
    observed = _spinless_energies(distributed, h1, eri, strings, n_particles)
    assert observed == pytest.approx(expected, abs=1e-8)


def test_spinless_complex64_subspace_matches_complex128() -> None:
    """The spinless owner-computes path supports the complex64 narrow subspace."""
    _skip_without_gloo()
    h1, eri, strings = _spinless_system()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    n_particles = 3
    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _spinless_energies(reference, complex_h1, complex_eri, strings, n_particles)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            davidson_dtype=torch.complex64,
            davidson_tolerance=1e-6,
            **_distributed(),
        )
    )
    observed = _spinless_energies(distributed, complex_h1, complex_eri, strings, n_particles)
    assert observed == pytest.approx(expected, abs=1e-5)


def test_gdb_complex64_subspace_matches_complex128() -> None:
    """The GDB owner-computes path supports the complex64 narrow subspace."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb", dtype=torch.complex128))
    expected = _pair_energies(reference, complex_h1, complex_eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            mode="gdb",
            dtype=torch.complex128,
            davidson_dtype=torch.complex64,
            davidson_tolerance=1e-6,
            **_distributed(),
        )
    )
    observed = _pair_energies(distributed, complex_h1, complex_eri, strings)
    assert observed == pytest.approx(expected, abs=1e-5)
    perf = _last_perf(distributed)
    assert perf["davidson_dtype"] == "torch.complex64"
    assert perf["distributed_matvec"] == "local"


def test_gdb_missed_root_check_runs_locally() -> None:
    """The GDB owner-computes path implements the multi-root missed-root search."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb", davidson_missed_root_check=True))
    expected = _pair_energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(mode="gdb", davidson_missed_root_check=True, **_distributed())
    )
    observed = _pair_energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    checks = _last_perf(distributed)["missed_root_checks"]
    assert isinstance(checks, list) and len(checks) <= 4


def test_gdb_complex_owner_computes_matches_serial() -> None:
    """Complex pair-mode integrals run on the GDB owner-computes path."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb", dtype=torch.complex128))
    expected = _pair_energies(reference, complex_h1, complex_eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(mode="gdb", dtype=torch.complex128, **_distributed())
    )
    observed = _pair_energies(distributed, complex_h1, complex_eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    assert _last_perf(distributed)["distributed_matvec"] == "local"


def test_gdb_warm_init_vector_converges() -> None:
    """A caller-supplied exact eigenvector warm-starts the GDB local path."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb", n_roots=1))
    ground = reference([_pair_strings(strings)], h1, eri, NORB, NELEC)[0].states[0]
    start = np.asarray(ground.sci_state.amplitudes, dtype=np.float64).reshape(-1)

    distributed = make_torch_sci_solver(**_solver_kwargs(mode="gdb", n_roots=1, **_distributed()))
    result = distributed([_pair_strings(strings)], h1, eri, NORB, NELEC, init_vectors=[start])
    state = result[0].states[0]
    assert float(state.energy) == pytest.approx(float(ground.energy), abs=1e-8)
    assert _last_perf(distributed)["warm"] is True
