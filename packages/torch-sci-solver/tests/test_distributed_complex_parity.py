"""Complex-dtype parity for the distributed solver on CPU (gloo, 2 and 4 ranks).

Covers the complex gaps left around the owner-computes/replicated routing:

* TPB and GDB (pair) bases, 1-D and tiled, complex128: energies/roots and
  ``rdm1``/``rdm2``/``<S^2>`` match the serial complex solver, and a real
  Hamiltonian cast to complex128 matches the real distributed run.
* The narrow ``complex64`` subspace on both the owner-computes and replicated
  layouts.
* Flat and tiled complex spinless bases, including the full-RDM build.
* Warm starts from a cached probe state and ``init_vectors`` on all layouts.
* Multi-root missed-root search on the complex owner-computes path.
* The ``spin_sq`` penalty (tiled and implicit ``(P, 1)``) and the memmapped
  result hand-back.
* Complex calls on the replicated fallback use partitioned ``C``/``HC`` blocks
  and reject the missed-root search with the documented error.

No GPU is required; the only backend used is gloo.
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


# --------------------------------------------------------------------------- #
# Fixtures and helpers                                                         #
# --------------------------------------------------------------------------- #
def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _synthetic_h4() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic real 4-orbital / 4-electron integrals and full CI strings."""
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


def _complex_hermitian_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian ``hcore`` plus a Hermitian, ``(pq|rs)=(rs|pq)`` ERI.

    The same construction as ``test_complex_matvec.py``'s fixture: a Hermitian
    one-body matrix and an ERI built as a sum of ``L L†`` outer products, which
    gives a genuinely complex Hamiltonian (couplings carry phases, so a
    conjugation error in an inner product changes the spectrum).
    """
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    naux = norb + 2
    eri = np.zeros((norb, norb, norb, norb), dtype=np.complex128)
    for _ in range(naux):
        g = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
        L = g + g.conj().T
        eri += np.einsum("pq,rs->pqrs", L, L.conj())
    eri = 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))
    return hcore, eri


def _spinless_system(
    norb: int = 6, n_particles: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic real spin-orbital Hamiltonian and its flat strings."""
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


def _pair_strings(strings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The full (α, β) pair list built from a product-mode string list."""
    return np.repeat(strings, len(strings)), np.tile(strings, len(strings))


def _solver_kwargs(**overrides) -> dict:
    """Deterministic settings shared by every reference and distributed run."""
    kwargs = dict(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=BLOCK_SIZE,
        max_davidson_iters=250,
        davidson_tolerance=1e-8,
        n_roots=N_ROOTS,
        davidson_missed_root_check=False,
        # Pin the native default explicitly (the qiskit drop-in factory
        # otherwise warns and turns the projector off).
        apply_ab_projection=True,
        verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def _distributed(
    world_size: int = 2,
    *,
    tile=None,
    matvec: str = "local",
    **overrides,
) -> dict:
    """A gloo launch config with the requested world size and tile layout."""
    cfg = {
        "world_size": world_size,
        "backend": "gloo",
        "persistent": False,
        "timeout": 600,
        "matvec": matvec,
        "tile": tile,
    }
    cfg.update(overrides)
    return {"distributed": cfg}


def _basis(mode: str, strings: np.ndarray) -> list:
    """The per-subspace basis argument for a mode (``tpb``, ``gdb`` or spinless)."""
    if mode == "gdb":
        return [_pair_strings(strings)]
    if mode == "spinless":
        return [strings]
    return [(strings, strings)]


def _solve(solver, mode: str, strings: np.ndarray, h1, eri, nelec, **call_kwargs):
    """Run one subspace and return its ``EigenStates``."""
    norb = int(h1.shape[0])
    return solver(_basis(mode, strings), h1, eri, norb, nelec, **call_kwargs)[0]


def _energies(states) -> list[float]:
    """Energies of a solved subspace, in root order."""
    return [float(state.energy) for state in states.states]


def _last_subspace(solver) -> dict:
    """The per-subspace performance record of the most recent call."""
    return solver.stats["per_call_log"][-1]["subspaces"][0]


def _close(actual, expected, atol: float) -> None:
    """Assert elementwise closeness for real or complex array-likes."""
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=0.0)


# --------------------------------------------------------------------------- #
# Real vs complex parity (real Hamiltonian carried in a complex dtype)         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("world_size", "tile"),
    [(2, None), (2, (1, 2)), (4, (2, 2))],
    ids=["2ranks-1d", "2ranks-tile", "4ranks-tile"],
)
def test_tpb_real_vs_complex_energies(world_size: int, tile) -> None:
    """TPB complex128 reproduces the real distributed roots, 1-D and tiled."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)

    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(_solve(reference, "tpb", strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(dtype=torch.complex128, **_distributed(world_size, tile=tile))
    )
    try:
        observed = _energies(_solve(distributed, "tpb", strings, ch1, ceri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["distributed_matvec"] == "local"
    if tile is not None:
        assert perf["tile_grid"] == list(tile)


@pytest.mark.parametrize(
    ("world_size", "tile"),
    [(2, None), (2, (1, 2)), (4, (2, 2))],
    ids=["2ranks-1d", "2ranks-tile", "4ranks-tile"],
)
def test_gdb_real_vs_complex_energies(world_size: int, tile) -> None:
    """GDB complex128 reproduces the real distributed roots, 1-D and tiled."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    gdb = {"mode": "gdb", "gdb_kwargs": {"block_matvec": "on"}}

    reference = make_torch_sci_solver(**_solver_kwargs(**gdb))
    expected = _energies(_solve(reference, "gdb", strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(dtype=torch.complex128, **gdb, **_distributed(world_size, tile=tile))
    )
    try:
        observed = _energies(_solve(distributed, "gdb", strings, ch1, ceri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["distributed_matvec"] == "local"


@pytest.mark.parametrize(
    ("world_size", "tile"),
    [(2, None), (2, (1, 2)), (4, (2, 2))],
    ids=["2ranks-1d", "2ranks-tile", "4ranks-tile"],
)
def test_spinless_real_vs_complex_energies(world_size: int, tile) -> None:
    """Flat and tiled spinless complex128 reproduce the real distributed roots."""
    _skip_without_gloo()
    h1, eri, strings = _spinless_system()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)

    reference = make_torch_sci_solver(**_solver_kwargs(apply_ab_projection=False))
    expected = _energies(_solve(reference, "spinless", strings, h1, eri, 3))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            apply_ab_projection=False,
            **_distributed(world_size, tile=tile),
        )
    )
    try:
        observed = _energies(_solve(distributed, "spinless", strings, ch1, ceri, 3))
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)


def test_complex_spinless_small_basis_replicated() -> None:
    """A complex spinless basis below one Davidson block runs the serial fallback."""
    _skip_without_gloo()
    norb = 4
    n_particles = 2
    h1, eri, strings = _spinless_system(norb, n_particles)
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    kwargs = dict(dtype=torch.complex128, apply_ab_projection=False)

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    expected = _energies(_solve(reference, "spinless", strings, ch1, ceri, n_particles))
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2)))
    try:
        observed = _energies(_solve(distributed, "spinless", strings, ch1, ceri, n_particles))
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)


# --------------------------------------------------------------------------- #
# Genuinely complex Hamiltonians: serial-vs-distributed parity                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tile", [None, (1, 2)], ids=["1d", "tile"])
def test_complex_hermitian_tpb_parity(tile) -> None:
    """TPB on a genuinely complex H matches the serial complex energies/RDMs/S²."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=0)
    strings = _synthetic_h4()[2]
    kwargs = dict(
        dtype=torch.complex128,
        compute_full_rdm=True,
        compute_spin_squared=True,
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_states = _solve(reference, "tpb", strings, h1, eri, NELEC)
    expected = _energies(ref_states)
    expected_rdm1 = ref_states.states[0].rdm1
    expected_rdm2 = ref_states.states[0].rdm2
    expected_s2 = list(reference.stats["spin_squared"])
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=tile)))
    try:
        observed_states = _solve(distributed, "tpb", strings, h1, eri, NELEC)
        observed_s2 = list(distributed.stats["spin_squared"])
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert _energies(observed_states) == pytest.approx(expected, abs=1e-8)
    _close(observed_states.states[0].rdm1, expected_rdm1, 1e-8)
    _close(observed_states.states[0].rdm2, expected_rdm2, 1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert perf["distributed_matvec"] == ("local" if tile is not None else "replicated")


def test_complex_hermitian_gdb_tile_parity() -> None:
    """Tiled GDB on a genuinely complex H matches the serial complex RDM/S² fields."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=1)
    strings = _synthetic_h4()[2]
    kwargs = dict(
        dtype=torch.complex128,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
        compute_full_rdm=True,
        compute_spin_squared=True,
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_states = _solve(reference, "gdb", strings, h1, eri, NELEC)
    expected = _energies(ref_states)
    expected_rdm1 = ref_states.states[0].rdm1
    expected_rdm2 = ref_states.states[0].rdm2
    expected_s2 = list(reference.stats["spin_squared"])
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=(1, 2))))
    try:
        observed_states = _solve(distributed, "gdb", strings, h1, eri, NELEC)
        observed_s2 = list(distributed.stats["spin_squared"])
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert _energies(observed_states) == pytest.approx(expected, abs=1e-8)
    _close(observed_states.states[0].rdm1, expected_rdm1, 1e-8)
    _close(observed_states.states[0].rdm2, expected_rdm2, 1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert perf["distributed_matvec"] == "local"


def test_complex_hermitian_gdb_flat_spin_sq_parity() -> None:
    """A no-tile GDB solve applies the complex spin penalty and <S^2> on the 1-D block."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=8)
    strings = _synthetic_h4()[2]
    kwargs = dict(
        dtype=torch.complex128,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
        compute_spin_squared=True,
    )
    call = dict(spin_sq=2.0, spin_penalty_lambda=0.3)

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_states = _solve(reference, "gdb", strings, h1, eri, NELEC, **call)
    expected = _energies(ref_states)
    expected_s2 = reference.stats["spin_sq_final"]
    expected_diagnostic = list(reference.stats["spin_squared"])
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2)))
    try:
        observed_states = _solve(distributed, "gdb", strings, h1, eri, NELEC, **call)
        observed_s2 = distributed.stats["spin_sq_final"]
        observed_diagnostic = list(distributed.stats["spin_squared"])
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert _energies(observed_states) == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert observed_diagnostic == pytest.approx(expected_diagnostic, abs=1e-8)
    assert perf["distributed_matvec"] == "local"
    assert "tile_grid" not in perf
    assert perf["spin_penalty_layout"] == "1d"


@pytest.mark.parametrize("tile", [None, (1, 2)], ids=["flat", "tile"])
def test_complex_hermitian_spinless_parity(tile) -> None:
    """Flat and tiled complex spinless bases match the serial complex RDM fields."""
    _skip_without_gloo()
    norb = 6
    n_particles = 3
    h1, eri = _complex_hermitian_integrals(norb, seed=2)
    strings = _spinless_system(norb, n_particles)[2]
    kwargs = dict(
        dtype=torch.complex128,
        apply_ab_projection=False,
        compute_full_rdm=True,
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_states = _solve(reference, "spinless", strings, h1, eri, n_particles)
    expected = _energies(ref_states)
    expected_rdm1 = ref_states.states[0].rdm1
    expected_rdm2 = ref_states.states[0].rdm2
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=tile)))
    try:
        observed_states = _solve(distributed, "spinless", strings, h1, eri, n_particles)
    finally:
        distributed.close()
    assert _energies(observed_states) == pytest.approx(expected, abs=1e-8)
    _close(observed_states.states[0].rdm1, expected_rdm1, 1e-8)
    _close(observed_states.states[0].rdm2, expected_rdm2, 1e-8)


def test_real_vs_complex_rdm_s2_parity_on_a_tile() -> None:
    """Real and complex128 distributed runs report identical energies/RDMs/S².

    The Hamiltonian is the real fixture carried in both dtypes, so every
    reported observable must agree at ``1e-8``; this is the observable-level
    real-vs-complex parity the complex plumbing must not perturb.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = dict(compute_full_rdm=True, compute_spin_squared=True)

    real = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=(1, 2))))
    try:
        real_states = _solve(real, "tpb", strings, h1, eri, NELEC)
        real_s2 = list(real.stats["spin_squared"])
        expected = _energies(real_states)
        expected_rdm1 = real_states.states[0].rdm1
        expected_rdm2 = real_states.states[0].rdm2
    finally:
        real.close()

    complex_solver = make_torch_sci_solver(
        **_solver_kwargs(dtype=torch.complex128, **kwargs, **_distributed(2, tile=(1, 2)))
    )
    try:
        complex_states = _solve(
            complex_solver,
            "tpb",
            strings,
            h1.astype(np.complex128),
            eri.astype(np.complex128),
            NELEC,
        )
        complex_s2 = list(complex_solver.stats["spin_squared"])
    finally:
        complex_solver.close()
    assert _energies(complex_states) == pytest.approx(expected, abs=1e-8)
    _close(complex_states.states[0].rdm1, expected_rdm1, 1e-8)
    _close(complex_states.states[0].rdm2, expected_rdm2, 1e-8)
    assert complex_s2 == pytest.approx(real_s2, abs=1e-8)


# --------------------------------------------------------------------------- #
# Narrow complex64 subspace                                                    #
# --------------------------------------------------------------------------- #
def test_complex64_narrow_subspace_tpb_tile() -> None:
    """The complex64 subspace reproduces the complex128 roots on a TPB tile."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=3)
    strings = _synthetic_h4()[2]

    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(_solve(reference, "tpb", strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            davidson_dtype=torch.complex64,
            davidson_tolerance=1e-6,
            **_distributed(2, tile=(1, 2)),
        )
    )
    try:
        observed = _energies(_solve(distributed, "tpb", strings, h1, eri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-5)
    assert perf["davidson_dtype"] == "torch.complex64"
    assert perf["distributed_matvec"] == "local"


def test_complex64_narrow_subspace_replicated() -> None:
    """The complex64 subspace works on the replicated fallback too."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=3)
    strings = _synthetic_h4()[2]

    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(_solve(reference, "tpb", strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            davidson_dtype=torch.complex64,
            davidson_tolerance=1e-6,
            **_distributed(2, matvec="replicated"),
        )
    )
    try:
        observed = _energies(_solve(distributed, "tpb", strings, h1, eri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-5)
    assert perf["davidson_dtype"] == "torch.complex64"
    assert perf["distributed_matvec"] == "replicated"


# --------------------------------------------------------------------------- #
# Warm starts and caller-provided init vectors                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tile", [None, (1, 2)], ids=["1d", "tile"])
def test_complex_tpb_warm_state(tile) -> None:
    """A cached complex probe state warm-starts the TPB 1-D and tiled paths."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    core = strings[::2]
    kwargs = dict(dtype=torch.complex128, n_roots=1)

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_core = reference([(core, core)], ch1, ceri, NORB, NELEC, probe_method="warm_start")[1][0]
    expected = _energies(
        reference(
            [(strings, strings)],
            ch1,
            ceri,
            NORB,
            NELEC,
            probe_method="warm_start",
            probe_state=[ref_core],
        )[0]
    )
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=tile)))
    try:
        dist_core = distributed([(core, core)], ch1, ceri, NORB, NELEC, probe_method="warm_start")[
            1
        ][0]
        observed = _energies(
            distributed(
                [(strings, strings)],
                ch1,
                ceri,
                NORB,
                NELEC,
                probe_method="warm_start",
                probe_state=[dist_core],
            )[0]
        )
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["warm"] is True
    assert perf["distributed_matvec"] == "local"


@pytest.mark.parametrize(
    ("matvec", "tile"),
    [("local", None), ("local", (1, 2)), ("replicated", None)],
    ids=["local-1d", "local-tile", "replicated"],
)
def test_complex_warm_state_preserves_phases(matvec: str, tile) -> None:
    """The cached complex probe state keeps its phases on every layout.

    A genuinely complex ground state has complex coefficients; caching the
    interleaved-Wick vector as fp64 would discard them and silently warm-start
    from a different vector.
    """
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=5)
    strings = _synthetic_h4()[2]
    core = strings[::2]
    kwargs = dict(dtype=torch.complex128, n_roots=1)

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_core = reference([(core, core)], h1, eri, NORB, NELEC, probe_method="warm_start")[1][0]
    ref_warm = _energies(
        reference(
            [(strings, strings)],
            h1,
            eri,
            NORB,
            NELEC,
            probe_method="warm_start",
            probe_state=[ref_core],
        )[0]
    )
    reference.close()
    ref_vec = np.asarray(ref_core.eigvec_inter)
    assert np.iscomplexobj(ref_vec)
    assert np.abs(ref_vec.imag).max() > 1e-6

    distributed = make_torch_sci_solver(
        **_solver_kwargs(**kwargs, **_distributed(2, tile=tile, matvec=matvec))
    )
    try:
        dist_core = distributed([(core, core)], h1, eri, NORB, NELEC, probe_method="warm_start")[1][
            0
        ]
        observed = _energies(
            distributed(
                [(strings, strings)],
                h1,
                eri,
                NORB,
                NELEC,
                probe_method="warm_start",
                probe_state=[dist_core],
            )[0]
        )
    finally:
        distributed.close()
    dist_vec = np.asarray(dist_core.eigvec_inter)
    assert np.iscomplexobj(dist_vec)
    # Same state up to the arbitrary global phase Davidson picks.
    phase = np.vdot(ref_vec, dist_vec)
    phase /= abs(phase)
    _close(dist_vec, ref_vec * phase, 1e-8)
    assert observed == pytest.approx(ref_warm, abs=1e-8)


@pytest.mark.parametrize("tile", [None, (1, 2)], ids=["1d", "tile"])
def test_complex_gdb_warm_state(tile) -> None:
    """A cached complex probe state warm-starts the GDB 1-D and tiled paths."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    core = strings[::2]
    kwargs = dict(
        dtype=torch.complex128,
        n_roots=1,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_core = reference([_pair_strings(core)], ch1, ceri, NORB, NELEC, probe_method="warm_start")[
        1
    ][0]
    expected = _energies(
        reference(
            [_pair_strings(strings)],
            ch1,
            ceri,
            NORB,
            NELEC,
            probe_method="warm_start",
            probe_state=[ref_core],
        )[0]
    )
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=tile)))
    try:
        dist_core = distributed(
            [_pair_strings(core)], ch1, ceri, NORB, NELEC, probe_method="warm_start"
        )[1][0]
        observed = _energies(
            distributed(
                [_pair_strings(strings)],
                ch1,
                ceri,
                NORB,
                NELEC,
                probe_method="warm_start",
                probe_state=[dist_core],
            )[0]
        )
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["warm"] is True


def test_complex_gdb_warm_state_preserves_phases() -> None:
    """The GDB cached probe state keeps its complex phases (replicated branch)."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=6)
    strings = _synthetic_h4()[2]
    core = strings[::2]
    kwargs = dict(
        dtype=torch.complex128,
        n_roots=1,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_core = reference([_pair_strings(core)], h1, eri, NORB, NELEC, probe_method="warm_start")[1][
        0
    ]
    ref_warm = _energies(
        reference(
            [_pair_strings(strings)],
            h1,
            eri,
            NORB,
            NELEC,
            probe_method="warm_start",
            probe_state=[ref_core],
        )[0]
    )
    reference.close()
    ref_vec = np.asarray(ref_core.eigvec_inter)
    assert np.iscomplexobj(ref_vec)
    assert np.abs(ref_vec.imag).max() > 1e-6

    distributed = make_torch_sci_solver(
        **_solver_kwargs(**kwargs, **_distributed(2, matvec="replicated"))
    )
    try:
        dist_core = distributed(
            [_pair_strings(core)], h1, eri, NORB, NELEC, probe_method="warm_start"
        )[1][0]
        observed = _energies(
            distributed(
                [_pair_strings(strings)],
                h1,
                eri,
                NORB,
                NELEC,
                probe_method="warm_start",
                probe_state=[dist_core],
            )[0]
        )
    finally:
        distributed.close()
    dist_vec = np.asarray(dist_core.eigvec_inter)
    assert np.iscomplexobj(dist_vec)
    phase = np.vdot(ref_vec, dist_vec)
    phase /= abs(phase)
    _close(dist_vec, ref_vec * phase, 1e-8)
    assert observed == pytest.approx(ref_warm, abs=1e-8)


@pytest.mark.parametrize("mode", ["tpb", "gdb", "spinless"], ids=["tpb", "gdb", "spinless"])
def test_complex_init_vectors(mode: str) -> None:
    """A caller-supplied complex eigenvector takes the warm path on every layout."""
    _skip_without_gloo()
    if mode == "spinless":
        norb, n_particles = 6, 3
        h1, eri, strings = _spinless_system(norb, n_particles)
        nelec = n_particles
    else:
        norb, n_particles = NORB, NELEC
        h1, eri, strings = _synthetic_h4()
        nelec = NELEC
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    kwargs = dict(
        dtype=torch.complex128,
        n_roots=1,
        apply_ab_projection=mode != "spinless",
    )
    if mode == "gdb":
        kwargs["mode"] = "gdb"
        kwargs["gdb_kwargs"] = {"block_matvec": "on"}

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ground = _solve(reference, mode, strings, ch1, ceri, nelec).states[0]
    expected = float(ground.energy)
    start = np.asarray(ground.sci_state.amplitudes, dtype=np.complex128).reshape(-1)
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2)))
    try:
        observed_states = _solve(distributed, mode, strings, ch1, ceri, nelec, init_vectors=[start])
        # The spinless path does not write the TPB/GDB per-call perf log.
        perf = _last_subspace(distributed) if mode != "spinless" else None
    finally:
        distributed.close()
    assert float(observed_states.states[0].energy) == pytest.approx(expected, abs=1e-8)
    if perf is not None:
        assert perf["warm"] is True


# --------------------------------------------------------------------------- #
# Multi-root missed-root search on the owner-computes complex path             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mode", "tile"),
    [("tpb", None), ("tpb", (1, 2)), ("gdb", None), ("gdb", (1, 2))],
    ids=["tpb-1d", "tpb-tile", "gdb-1d", "gdb-tile"],
)
def test_complex_missed_root_check(mode: str, tile) -> None:
    """The multi-root missed-root search runs on the complex owner-computes path."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    kwargs = dict(dtype=torch.complex128, davidson_missed_root_check=True)
    if mode == "gdb":
        kwargs["mode"] = "gdb"
        kwargs["gdb_kwargs"] = {"block_matvec": "on"}

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    expected = _energies(_solve(reference, mode, strings, ch1, ceri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=tile)))
    try:
        observed = _energies(_solve(distributed, mode, strings, ch1, ceri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["distributed_matvec"] == "local"
    checks = perf["missed_root_checks"]
    assert isinstance(checks, list)
    assert all({"found", "energy", "n_matvecs"} <= set(check) for check in checks)


def test_complex_missed_root_rejected_on_replicated_fallback() -> None:
    """The replicated complex fallback rejects the missed-root search clearly."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    solver = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            davidson_missed_root_check=True,
            **_distributed(2, matvec="replicated"),
        )
    )
    try:
        with pytest.raises(NotImplementedError, match="owner-computes"):
            solver.preflight([(strings, strings)], NELEC, norb=NORB, two_body_tensor=ceri)
        with pytest.raises(NotImplementedError, match="owner-computes"):
            _solve(solver, "tpb", strings, ch1, ceri, NELEC)
    finally:
        solver.close()


# --------------------------------------------------------------------------- #
# spin_sq penalty                                                             #
# --------------------------------------------------------------------------- #
def test_complex_spin_sq_penalty_tpb_implicit_and_tiled() -> None:
    """The complex spin penalty runs on the implicit (P, 1) and explicit tiles."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    call = {"spin_sq": 2.0, "spin_penalty_lambda": 0.3}

    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(_solve(reference, "tpb", strings, ch1, ceri, NELEC, **call))
    expected_s2 = reference.stats["spin_sq_final"]
    reference.close()

    for tile in (None, (1, 2)):
        distributed = make_torch_sci_solver(
            **_solver_kwargs(dtype=torch.complex128, **_distributed(2, tile=tile))
        )
        try:
            observed = _energies(_solve(distributed, "tpb", strings, ch1, ceri, NELEC, **call))
            observed_s2 = distributed.stats["spin_sq_final"]
            perf = _last_subspace(distributed)
        finally:
            distributed.close()
        assert observed == pytest.approx(expected, abs=1e-7)
        assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
        assert perf["distributed_matvec"] == "local"
        assert perf["tile_grid"] == ([2, 1] if tile is None else list(tile))


def test_complex_spin_sq_penalty_gdb_tile() -> None:
    """The complex spin penalty runs on a GDB pair tile."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    call = {"spin_sq": 2.0, "spin_penalty_lambda": 0.3}
    kwargs = dict(
        dtype=torch.complex128,
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
    )

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    expected = _energies(_solve(reference, "gdb", strings, ch1, ceri, NELEC, **call))
    expected_s2 = reference.stats["spin_sq_final"]
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2, tile=(1, 2))))
    try:
        observed = _energies(_solve(distributed, "gdb", strings, ch1, ceri, NELEC, **call))
        observed_s2 = distributed.stats["spin_sq_final"]
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert perf["distributed_matvec"] == "local"


# --------------------------------------------------------------------------- #
# Replicated fallback and memmapped hand-back                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["tpb", "gdb"], ids=["tpb", "gdb"])
def test_complex_small_basis_dense_fallback(mode: str) -> None:
    """A complex basis below one Davidson block takes the replicated dense escape."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=7)
    strings = _synthetic_h4()[2]
    kwargs = dict(dtype=torch.complex128, davidson_block_size=64)
    if mode == "gdb":
        kwargs["mode"] = "gdb"
        kwargs["gdb_kwargs"] = {"block_matvec": "on"}

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    expected = _energies(_solve(reference, mode, strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(**_solver_kwargs(**kwargs, **_distributed(2)))
    try:
        observed = _energies(_solve(distributed, mode, strings, h1, eri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["distributed_matvec"] == "replicated"


def test_complex_replicated_fallback_matches_serial() -> None:
    """Complex replicated solves keep partitioned C/HC blocks and match serial."""
    _skip_without_gloo()
    h1, eri = _complex_hermitian_integrals(NORB, seed=4)
    strings = _synthetic_h4()[2]

    reference = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    expected = _energies(_solve(reference, "tpb", strings, h1, eri, NELEC))
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(dtype=torch.complex128, **_distributed(2, matvec="replicated"))
    )
    try:
        observed = _energies(_solve(distributed, "tpb", strings, h1, eri, NELEC))
        perf = _last_subspace(distributed)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert perf["distributed_matvec"] == "replicated"


@pytest.mark.parametrize(
    ("mode", "tile"),
    [("tpb", (1, 2)), ("gdb", (1, 2)), ("spinless", None), ("spinless", (1, 2))],
    ids=["tpb-tile", "gdb-tile", "spinless-flat", "spinless-tile"],
)
def test_complex_memmap_hand_back(mode: str, tile) -> None:
    """The memmapped hand-back returns complex amplitudes matching the serial run."""
    _skip_without_gloo()
    if mode == "spinless":
        norb, n_particles = 6, 3
        h1, eri, strings = _spinless_system(norb, n_particles)
        nelec = n_particles
    else:
        norb, n_particles = NORB, NELEC
        h1, eri, strings = _synthetic_h4()
        nelec = NELEC
    ch1 = h1.astype(np.complex128)
    ceri = eri.astype(np.complex128)
    kwargs = dict(
        dtype=torch.complex128,
        n_roots=1,
        apply_ab_projection=mode != "spinless",
    )
    if mode == "gdb":
        kwargs["mode"] = "gdb"
        kwargs["gdb_kwargs"] = {"block_matvec": "on"}

    reference = make_torch_sci_solver(**_solver_kwargs(**kwargs))
    ref_state = _solve(reference, mode, strings, ch1, ceri, nelec).states[0]
    expected_energy = float(ref_state.energy)
    expected_amp = np.asarray(ref_state.sci_state.amplitudes)
    reference.close()

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            **kwargs,
            **_distributed(2, tile=tile, hand_back="memmap"),
        )
    )
    try:
        states = _solve(distributed, mode, strings, ch1, ceri, nelec)
        observed_energy = float(states.states[0].energy)
        observed_amp = np.asarray(states.states[0].sci_state.amplitudes)
    finally:
        distributed.close()
    assert observed_energy == pytest.approx(expected_energy, abs=1e-8)
    assert np.iscomplexobj(observed_amp)
    assert np.asarray(observed_amp).shape == np.asarray(expected_amp).shape
    # Davidson fixes the eigenvector only up to a global phase; compare magnitudes.
    _close(np.abs(observed_amp), np.abs(expected_amp), 1e-7)
