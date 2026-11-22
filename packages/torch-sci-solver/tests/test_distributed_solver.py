"""Tests for the opt-in distributed ``TorchSCISolver`` path.

The primary check is CPU-only: a world_size=2 gloo solve must reproduce the
single-process energies and roots on a small full-CI determinant space. The
same test runs on two GPUs with NCCL when they are available (marked ``gpu``
and ``slow``; skipped otherwise).

Nothing here changes the default solver: every other test in the package
exercises the single-process path unchanged.
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
from torch_sci_solver.torch_impl.distributed_solver import parse_distributed_config

NORB = 4
NELEC = (2, 2)
N_ROOTS = 2
BLOCK_SIZE = 8


def _synthetic_h4():
    """A deterministic, Hermitian 4-orbital / 4-electron Hamiltonian.

    Small enough for full CI in the (2α, 2β) sector (M_a = M_b = 6, N = 36),
    with 8-fold-symmetric ERIs so the solver's real path is exercised exactly
    like an FCIDUMP.
    """
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


def _synthetic_system(norb: int, n_elec: int):
    """Deterministic Hermitian integral set and full CI strings for ``(norb, n_elec)``."""
    rng = np.random.default_rng(norb * 100 + n_elec)
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
        sorted(sum(1 << o for o in occ) for occ in itertools.combinations(range(norb), n_elec)),
        dtype=np.int64,
    )
    return h1, eri, strings


def _solver_kwargs(**overrides):
    """Deterministic solver settings shared by the reference and distributed runs."""
    kwargs = dict(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=BLOCK_SIZE,
        max_davidson_iters=200,
        davidson_tolerance=1e-8,
        n_roots=N_ROOTS,
        # Distributed mode does not implement the missed-root search; the cold
        # single-process reference does not enable it either ("auto" is on only
        # for warm starts), so turning it off keeps both arms identical.
        davidson_missed_root_check=False,
        verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def _energies(solver, h1, eri, strings):
    """Call the solver on one subspace and return the root energies."""
    result = solver([(strings, strings)], h1, eri, NORB, NELEC)
    return [float(state.energy) for state in result[0].states]


def _skip_without_gloo():
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


# --------------------------------------------------------------------------- #
# Config / interface
# --------------------------------------------------------------------------- #
def test_parse_distributed_config() -> None:
    """The public keyword normalizes to a launch config or a clean fallback."""
    assert parse_distributed_config(None, device="cpu", cuda_device_count=0) is None
    assert parse_distributed_config(False, device="cpu", cuda_device_count=0) is None
    # ``True`` with no GPUs and no launcher WORLD_SIZE falls back cleanly.
    assert parse_distributed_config(True, device="cpu", cuda_device_count=0) is None
    assert parse_distributed_config({"world_size": 1}, device="cpu", cuda_device_count=0) is None
    cfg = parse_distributed_config(
        {"world_size": 2, "backend": "gloo", "timeout": 30},
        device="cpu",
        cuda_device_count=0,
    )
    assert cfg is not None
    assert cfg["world_size"] == 2
    assert cfg["backend"] == "gloo"
    assert cfg["timeout"] == 30.0
    assert cfg["init_method"].startswith("tcp://")

    with pytest.raises(ValueError, match="unknown distributed keys"):
        parse_distributed_config({"world_size": 2, "bogus": 1}, device="cpu", cuda_device_count=0)
    with pytest.raises(ValueError, match="device_ids"):
        parse_distributed_config(
            {"world_size": 2, "backend": "gloo", "device_ids": [0]},
            device="cpu",
            cuda_device_count=0,
        )

    cfg = parse_distributed_config(
        {"world_size": 2, "backend": "gloo", "tile": "auto", "exchange_bytes": 4096},
        device="cpu",
        cuda_device_count=0,
    )
    assert cfg is not None
    assert cfg["tile"] == "auto"
    assert cfg["exchange_bytes"] == 4096
    assert cfg["hand_back_dir"] is None
    cfg = parse_distributed_config(
        {"world_size": 2, "backend": "gloo", "hand_back_dir": "/tmp/hand_back"},
        device="cpu",
        cuda_device_count=0,
    )
    assert cfg is not None
    assert cfg["hand_back_dir"] == "/tmp/hand_back"
    assert (
        parse_distributed_config(
            {"world_size": 2, "backend": "gloo"}, device="cpu", cuda_device_count=0
        )["exchange_bytes"]
        is None
    )
    with pytest.raises(ValueError, match="exchange_bytes"):
        parse_distributed_config(
            {"world_size": 2, "backend": "gloo", "exchange_bytes": 0},
            device="cpu",
            cuda_device_count=0,
        )


def test_solver_accepts_distributed_and_keeps_default_off() -> None:
    """``distributed`` is parsed at construction; the default stays disabled."""
    distributed = make_torch_sci_solver(
        **_solver_kwargs(distributed={"world_size": 2, "backend": "gloo"})
    )
    assert distributed._distributed_config is not None
    assert distributed._distributed_ctx is None
    assert make_torch_sci_solver(**_solver_kwargs())._distributed_config is None
    assert (
        make_torch_sci_solver(**_solver_kwargs(distributed={"world_size": 1}))._distributed_config
        is None
    )


class _RecordingSolver:
    """Stand-in solver that records the backend of one in-process rank run."""

    def __init__(self, requested: str, resolved: str = "batched", mode: str = "tpb") -> None:
        """Expose the attributes ``_run_in_process`` saves and rewrites."""
        self._matvec_impl_requested = requested
        self.matvec_impl = resolved
        self.mode = mode
        self.device = torch.device("cpu")
        self.devices = (self.device,)
        self._distributed_ctx = None
        self._inside_distributed = False
        self.seen: list[str] = []

    def __call__(self, *_args, **_kwargs):
        """Record the backend and return an empty result."""
        self.seen.append(self.matvec_impl)
        return []


def _nccl_cfg() -> dict:
    """Minimal launch config for a CUDA rank."""
    return {
        "backend": "nccl",
        "device_ids": [0, 1],
        "device_ids_explicit": True,
        "matvec": "local",
        "tile": None,
        "exchange_bytes": None,
        "hand_back_dir": None,
    }


def test_in_process_auto_reresolves_against_the_rank_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An "auto" request constructed on CPU runs Triton on an NCCL rank."""
    from torch_sci_solver.torch_impl import distributed_solver

    monkeypatch.setattr(
        distributed_solver, "_rank_device", lambda _cfg, rank: torch.device("cuda", rank)
    )
    monkeypatch.setattr(distributed_solver, "_triton_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    solver = _RecordingSolver("auto")
    payload = (None, None, None, NORB, NELEC, None, {})
    assert (
        distributed_solver._run_in_process(solver, _nccl_cfg(), payload, rank=0, world_size=2) == []
    )
    assert solver.seen == ["triton"]
    # "auto" keeps the resolved value after the call so metadata names the
    # backend that actually ran; the device and context are restored.
    assert solver.matvec_impl == "triton"
    assert solver.device == torch.device("cpu")


def test_in_process_gdb_auto_stays_batched_on_a_cuda_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GDB "auto" never selects Triton, even after device re-resolution."""
    from torch_sci_solver.torch_impl import distributed_solver

    monkeypatch.setattr(
        distributed_solver, "_rank_device", lambda _cfg, rank: torch.device("cuda", rank)
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    solver = _RecordingSolver("auto", mode="gdb")
    payload = (None, None, None, NORB, NELEC, None, {})
    distributed_solver._run_in_process(solver, _nccl_cfg(), payload, rank=0, world_size=2)
    assert solver.seen == ["batched"]


def test_in_process_keeps_an_explicit_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit request is untouched by the rank-device re-resolution."""
    from torch_sci_solver.torch_impl import distributed_solver

    monkeypatch.setattr(
        distributed_solver, "_rank_device", lambda _cfg, rank: torch.device("cuda", rank)
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    solver = _RecordingSolver("batched")
    payload = (None, None, None, NORB, NELEC, None, {})
    distributed_solver._run_in_process(solver, _nccl_cfg(), payload, rank=0, world_size=2)
    assert solver.seen == ["batched"]
    assert solver.matvec_impl == "batched"


def test_explicit_triton_is_rejected_on_gloo() -> None:
    """Triton needs CUDA; a gloo rank raises instead of silently going batched."""
    from torch_sci_solver.torch_impl import distributed_solver

    solver = _RecordingSolver("triton", resolved="triton")
    payload = (None, None, None, NORB, NELEC, None, {})
    with pytest.raises(ValueError, match="requires CUDA"):
        distributed_solver._run_in_process(
            solver, {"backend": "gloo"}, payload, rank=0, world_size=2
        )
    assert solver.seen == []


def test_run_distributed_call_rejects_triton_on_gloo_before_spawning() -> None:
    """The parent surfaces the gloo/Triton mismatch, not a worker crash."""
    from torch_sci_solver.torch_impl import distributed_solver

    class _FakeSolver:
        """Only the attributes the entry check reads."""

        _inside_distributed = False
        _distributed_config = {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": None,
        }
        devices = (torch.device("cpu"),)
        matvec_impl = "triton"
        _matvec_impl_requested = "triton"

    with pytest.raises(ValueError, match="requires CUDA"):
        distributed_solver.run_distributed_call(
            _FakeSolver(), None, None, None, NORB, NELEC, None, {}
        )


def test_gloo_resolution_leaves_the_construction_time_backend_in_place() -> None:
    """A CPU/gloo solve keeps "auto" resolved to batched and rejects Triton."""
    from torch_sci_solver.torch_impl.distributed_solver import _worker_matvec_impl

    solver = make_torch_sci_solver(
        **_solver_kwargs(matvec_impl="auto", distributed={"world_size": 2, "backend": "gloo"})
    )
    assert solver.matvec_impl == "batched"
    assert solver._matvec_impl_requested == "auto"
    cpu = torch.device("cpu")
    assert _worker_matvec_impl("auto", mode="tpb", dev=cpu, backend="gloo") == "batched"
    assert _worker_matvec_impl("batched", mode="tpb", dev=cpu, backend="gloo") == "batched"
    with pytest.raises(ValueError, match="requires CUDA"):
        _worker_matvec_impl("triton", mode="tpb", dev=cpu, backend="gloo")


def test_world_size_one_falls_back_to_single_process() -> None:
    """world_size <= 1 runs the existing path and returns the same energies."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    fallback = make_torch_sci_solver(
        **_solver_kwargs(distributed={"world_size": 1, "backend": "gloo"})
    )
    e_ref = _energies(reference, h1, eri, strings)
    e_fallback = _energies(fallback, h1, eri, strings)
    assert e_fallback == pytest.approx(e_ref, abs=1e-12)


def test_complex_and_spinless_layouts_are_supported() -> None:
    """Complex and spinless calls run distributed instead of being rejected."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    complex_h1 = h1.astype(np.complex128)
    complex_eri = eri.astype(np.complex128)
    dist_cfg = {"world_size": 2, "backend": "gloo", "timeout": 120, "persistent": False}

    # Complex: the replicated full solve on every rank (matvec="replicated"),
    # independent of the owner-computes complex path.
    ref_complex = make_torch_sci_solver(**_solver_kwargs(dtype=torch.complex128))
    dist_complex = make_torch_sci_solver(
        **_solver_kwargs(
            dtype=torch.complex128,
            distributed={**dist_cfg, "matvec": "replicated"},
        )
    )
    assert _energies(dist_complex, complex_h1, complex_eri, strings) == pytest.approx(
        _energies(ref_complex, complex_h1, complex_eri, strings), abs=1e-10
    )

    # Spinless: scalar nelec dispatches to the flat path (replicated here, the
    # 6-determinant basis fits in one block).
    ref_spinless = make_torch_sci_solver(**_solver_kwargs())
    dist_spinless = make_torch_sci_solver(**_solver_kwargs(distributed=dist_cfg))
    n_particles = int(NELEC[0])
    e_ref = [
        float(state.energy)
        for state in ref_spinless([strings], h1, eri, NORB, n_particles)[0].states
    ]
    e_dist = [
        float(state.energy)
        for state in dist_spinless([strings], h1, eri, NORB, n_particles)[0].states
    ]
    assert e_dist == pytest.approx(e_ref, abs=1e-10)


# --------------------------------------------------------------------------- #
# CPU gloo: the CI check
# --------------------------------------------------------------------------- #
def test_two_rank_gloo_matches_single_process() -> None:
    """world_size=2 gloo reproduces the single-process energies and roots."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()

    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={"world_size": 2, "backend": "gloo", "timeout": 120, "persistent": False}
        )
    )
    observed = _energies(distributed, h1, eri, strings)

    assert len(observed) == N_ROOTS
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed[0] < observed[1]
    # The parent solver's cumulative stats are copied back from rank 0.
    assert distributed.stats["n_calls"] >= 1


def test_two_rank_gloo_single_root() -> None:
    """The single-root return shape also matches the single-process solve."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(n_roots=1)

    reference = make_torch_sci_solver(**kwargs)
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {"world_size": 2, "backend": "gloo", "timeout": 120, "persistent": False},
    })
    observed = _energies(distributed, h1, eri, strings)
    assert len(observed) == 1
    assert observed == pytest.approx(expected, abs=1e-8)


def test_two_rank_gloo_matches_single_process_gdb() -> None:
    """The GDB (pair-basis) branch also matches its single-process solve."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb")

    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {"world_size": 2, "backend": "gloo", "timeout": 120, "persistent": False},
    })
    observed = [
        float(state.energy)
        for state in distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]
    assert len(observed) == N_ROOTS
    assert observed == pytest.approx(expected, abs=1e-8)


def test_two_rank_gloo_tiled_gdb_matches_single_process_on_a_recovered_list() -> None:
    """A forced tile grid on an irregular pair list solves its masked product.

    A recovered list drops pairs from the full product, so its exact block
    decomposition splits into many tiny partner-set groups. The tiled GDB path
    embeds the list in the product of its unique strings and keeps the padding
    pairs at zero amplitude; on the caller's pairs the energies must match the
    single-process solve, and the returned amplitudes and pair strings must
    describe the caller's list rather than the padded product.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(0).permutation(len(pairs_a))[:32])
    pairs_a, pairs_b = pairs_a[keep], pairs_b[keep]
    pairs = list(zip(pairs_a.tolist(), pairs_b.tolist()))
    kwargs = _solver_kwargs(mode="gdb", compute_full_rdm=True, compute_spin_squared=True)

    reference = make_torch_sci_solver(**kwargs)
    reference_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)
    expected = [float(state.energy) for state in reference_result[0].states]
    expected_s2 = reference.stats["spin_squared"][-1]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": "force",
            "timeout": 120,
            "persistent": False,
        },
    })
    result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)
    observed = [float(state.energy) for state in result[0].states]
    assert len(observed) == N_ROOTS
    assert observed == pytest.approx(expected, abs=1e-8)

    # The RDM and <S^2> post-processing also runs on the superset; the padding
    # pairs carry zero amplitude, so both must equal the caller's-basis values.
    reference_state = reference_result[0].states[0]
    for got, want in (
        (result[0].states[0].rdm1, reference_state.rdm1),
        (result[0].states[0].rdm2, reference_state.rdm2),
    ):
        assert got is not None and want is not None
        assert np.allclose(got, want, atol=1e-8)
    assert distributed.stats["spin_squared"][-1] == pytest.approx(expected_s2, abs=1e-8)

    state = result[0].states[0].sci_state
    assert len(state.amplitudes) == len(pairs)
    returned = sorted(
        zip(
            np.asarray(state.ci_strs_a).tolist(),
            np.asarray(state.ci_strs_b).tolist(),
        )
    )
    assert returned == pairs

    perf = distributed.stats["per_call_log"][-1]["subspaces"][0]
    assert "tile_grid" in perf
    assert perf["pair_superset"] is True
    assert perf["pair_basis_N"] == len(pairs)
    assert perf["N"] == len(strings) ** 2


def test_two_rank_gloo_tiled_gdb_replaces_fragmented_exact_blocks() -> None:
    """An exact ``"on"`` decomposition with tiny partner-set groups is replaced.

    At scale a recovered list's block decomposition is exact but splits into one
    tiny group per partner set. Even when the caller forces ``block_matvec="on"``,
    the tiled path must solve the masked product, whose one-rectangle layout is
    what the tile kernel amortizes, and still match the single-process energies.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_system(6, 3)
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(0).permutation(len(pairs_a))[:360])
    pairs_a, pairs_b = pairs_a[keep], pairs_b[keep]
    kwargs = _solver_kwargs(mode="gdb")

    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, 6, (3, 3))[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "gdb_kwargs": {"block_matvec": "on"},
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": "force",
            "timeout": 120,
            "persistent": False,
        },
    })
    result = distributed([(pairs_a, pairs_b)], h1, eri, 6, (3, 3))
    observed = [float(state.energy) for state in result[0].states]
    assert observed == pytest.approx(expected, abs=1e-8)
    perf = distributed.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["pair_superset"] is True
    assert perf["pair_basis_N"] == len(pairs_a)
    assert perf["N"] == len(strings) ** 2


def test_two_rank_gloo_auto_declined_tile_keeps_the_caller_basis() -> None:
    """A superset candidate never overrides a declined ``tile="auto"``.

    On CPU the auto policy always prefers the 1-D layout, so an irregular list
    must take the owner-computes path on its own (exact) basis — no masked
    product, no masking assertion — and still match the single-process solve.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_system(6, 3)
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(0).permutation(len(pairs_a))[:360])
    pairs_a, pairs_b = pairs_a[keep], pairs_b[keep]
    kwargs = _solver_kwargs(mode="gdb")

    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, 6, (3, 3))[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "gdb_kwargs": {"block_matvec": "on"},
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": "auto",
            "timeout": 120,
            "persistent": False,
        },
    })
    result = distributed([(pairs_a, pairs_b)], h1, eri, 6, (3, 3))
    observed = [float(state.energy) for state in result[0].states]
    assert observed == pytest.approx(expected, abs=1e-8)
    perf = distributed.stats["per_call_log"][-1]["subspaces"][0]
    assert "pair_superset" not in perf
    assert perf["N"] == len(pairs_a)


def test_two_rank_gloo_tiled_gdb_spin_penalty_on_a_recovered_list() -> None:
    """The spin_sq penalty and its energy strip also run on the masked product.

    The tiled penalty applies ``S^2`` through the same staged exchange as ``H``,
    so on an irregular list both terms must reduce to the caller's basis and
    the stripped energy must match the single-process augmented solve.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(1).permutation(len(pairs_a))[:32])
    pairs_a, pairs_b = pairs_a[keep], pairs_b[keep]
    kwargs = _solver_kwargs(mode="gdb")

    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=0.0)[0].states
    ]
    expected_s2 = reference.stats["spin_squared"][-1]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": "force",
            "hand_back": "memmap",
            "timeout": 120,
            "persistent": False,
        },
    })
    result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=0.0)
    observed = [float(state.energy) for state in result[0].states]
    assert observed == pytest.approx(expected, abs=1e-8)
    assert distributed.stats["spin_squared"][-1] == pytest.approx(expected_s2, abs=1e-8)

    # The penalty only runs on the masked product; assert the layout was used.
    perf = distributed.stats["per_call_log"][-1]["subspaces"][0]
    assert perf["pair_superset"] is True
    assert perf["pair_basis_N"] == len(pairs_a)

    # The forced memmap hand-back writes only the caller's pairs (the padding
    # pairs never reach the file), so the materialized state has its length.
    state = result[0].states[0].sci_state
    assert len(state.amplitudes) == len(pairs_a)
    returned = sorted(
        zip(
            np.asarray(state.ci_strs_a).tolist(),
            np.asarray(state.ci_strs_b).tolist(),
        )
    )
    assert returned == list(zip(pairs_a.tolist(), pairs_b.tolist()))


def test_two_rank_gloo_small_recovered_list_keeps_the_replicated_penalty() -> None:
    """A grid that cannot reach the owner-computes path keeps the replicated penalty.

    A pair basis below one Davidson block never takes the local path. A tile
    grid, even one granted by the post-processing fallback, must not turn on the
    tiled penalty operator for it; the solve keeps the replicated penalty.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(2).permutation(len(pairs_a))[:20])
    pairs_a, pairs_b = pairs_a[keep], pairs_b[keep]
    # The default block-size fallback (max(30, n_roots + 3)) is above this
    # basis, so it never reaches the owner-computes path.
    kwargs = _solver_kwargs(mode="gdb", davidson_block_size=None)

    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=0.0)[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "matvec": "local",
            "tile": "auto",
            "timeout": 120,
            "persistent": False,
        },
    })
    result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=0.0)
    observed = [float(state.energy) for state in result[0].states]
    assert observed == pytest.approx(expected, abs=1e-8)
    perf = distributed.stats["per_call_log"][-1]["subspaces"][0]
    assert "pair_superset" not in perf
    assert perf["N"] == len(pairs_a)


def test_distributed_rejects_missed_root_check_on_replicated_path() -> None:
    """The real replicated fallback refuses the missed-root search, not ignores it.

    The owner-computes paths implement it (TPB and GDB), so construction no
    longer rejects the knob; a call forced onto the full-space
    ``davidson_distributed`` fallback must raise instead of silently dropping it.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    solver = make_torch_sci_solver(
        **_solver_kwargs(
            mode="gdb",
            davidson_missed_root_check=True,
            distributed={
                "world_size": 2,
                "backend": "gloo",
                "persistent": False,
                "matvec": "replicated",
            },
        )
    )
    with pytest.raises(NotImplementedError, match="missed_root_check"):
        solver([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)


def test_tile_auto_with_full_rdm_is_conservative_for_the_missed_root_gate() -> None:
    """An "auto" tile spec is not treated as tiled before the basis is known.

    A declined grid may still route full-RDM post-processing to the owner-
    computes path, but the parent cannot know that before the basis is
    available; it must reject the missed-root search up front instead of
    failing in a worker that lands on the replicated fallback.
    """
    h1, eri, strings = _synthetic_h4()
    solver = make_torch_sci_solver(
        **_solver_kwargs(
            compute_full_rdm=True,
            davidson_missed_root_check=True,
            distributed={
                "world_size": 2,
                "backend": "gloo",
                "persistent": False,
                "tile": "auto",
            },
        )
    )
    with pytest.raises(NotImplementedError, match="missed_root_check"):
        solver([(strings, strings)], h1, eri, NORB, NELEC)


def _assert_perf_matvec(solver, expected: str) -> None:
    """Check which distributed matvec the last call recorded."""
    subspaces = solver.stats["per_call_log"][-1]["subspaces"]
    assert subspaces[0]["distributed_matvec"] == expected


def test_owner_computes_matvec_matches_single_process() -> None:
    """The owner-computes TPB matvec reproduces the single-process energies."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(apply_ab_projection=False)

    reference = make_torch_sci_solver(**kwargs)
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 120,
            "matvec": "local",
        },
    })
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    _assert_perf_matvec(distributed, "local")


def test_projection_uses_local_matvec() -> None:
    """The αβ-swap projector is applied on local row blocks."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(apply_ab_projection=True)

    reference = make_torch_sci_solver(**kwargs)
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 120,
            "matvec": "local",
        },
    })
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)
    _assert_perf_matvec(distributed, "local")


def test_distributed_tiny_block_size_still_converges() -> None:
    """A user block size of 2 is floored; the restart keeps the new correction."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(
        n_roots=1,
        davidson_block_size=2,
        apply_ab_projection=False,
    )

    reference = make_torch_sci_solver(**{**kwargs, "davidson_block_size": BLOCK_SIZE})
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 120,
            "matvec": "local",
        },
    })
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-8)


def test_large_result_round_trip() -> None:
    """A result larger than a 64 KiB pipe buffer comes back intact.

    norb=9, 4+4 electrons -> M=126, N=15,876; the returned amplitudes are
    ~127 KiB, which deadlocked the old queue hand-back.
    """
    _skip_without_gloo()
    norb = 9
    h1, eri, strings = _synthetic_system(norb, 4)
    nelec = (4, 4)
    kwargs = _solver_kwargs(
        n_roots=1,
        davidson_block_size=BLOCK_SIZE,
        apply_ab_projection=False,
        max_davidson_iters=100,
        davidson_tolerance=1e-6,
    )

    reference = make_torch_sci_solver(**kwargs)
    expected = float(reference([(strings, strings)], h1, eri, norb, nelec)[0].states[0].energy)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 300,
            "matvec": "local",
        },
    })
    result = distributed([(strings, strings)], h1, eri, norb, nelec)
    amplitude_bytes = result[0].states[0].sci_state.amplitudes.nbytes
    assert amplitude_bytes > 64 * 1024
    assert float(result[0].states[0].energy) == pytest.approx(expected, abs=1e-4)


def test_persistent_session_reuses_workers() -> None:
    """A persistent session serves several calls without respawning workers."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(apply_ab_projection=False)
    reference = make_torch_sci_solver(**kwargs)
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "persistent": True,
        },
    })
    try:
        first = _energies(distributed, h1, eri, strings)
        second = _energies(distributed, h1, eri, strings)
        assert first == pytest.approx(expected, abs=1e-8)
        assert second == pytest.approx(expected, abs=1e-8)
        assert distributed.stats["n_calls"] >= 2
        assert distributed._distributed_session is not None
    finally:
        distributed.close()
    assert distributed._distributed_session is None
    # close() is idempotent.
    distributed.close()


# --------------------------------------------------------------------------- #
# GPU / NCCL (skipped unless two CUDA devices and NCCL are available)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
    ),
    reason="two CUDA devices and NCCL are required",
)
def test_two_rank_nccl_matches_single_process() -> None:
    """world_size=2 NCCL reproduces the CPU single-process energies.

    Run on a two-GPU node with:

        uv run pytest packages/torch-sci-solver/tests/test_distributed_solver.py \\
            -m gpu -v
    """
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 2,
                "backend": "nccl",
                "device_ids": [0, 1],
                "timeout": 300,
            }
        )
    )
    observed = _energies(distributed, h1, eri, strings)
    assert observed == pytest.approx(expected, abs=1e-7)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
    ),
    reason="two CUDA devices and NCCL are required",
)
def test_gdb_flat_spin_sq_penalty_two_rank_nccl() -> None:
    """The 1-D penalty path reproduces the serial solve on two NCCL ranks."""
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(
        mode="gdb", gdb_kwargs={"block_matvec": "on"}, compute_spin_squared=True
    )
    call = dict(spin_sq=2.0, spin_penalty_lambda=0.3)
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, **call)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]
    expected_diagnostic = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "nccl",
            "device_ids": [0, 1],
            "timeout": 300,
            "matvec": "local",
            "persistent": False,
        },
    })
    try:
        result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, **call)[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        observed_diagnostic = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-6)
    assert observed_diagnostic == pytest.approx(expected_diagnostic, abs=1e-6)
    assert subspace["distributed_matvec"] == "local"
    assert "tile_grid" not in subspace
    assert subspace["spin_penalty_layout"] == "1d"


@pytest.mark.gpu
def test_gdb_triton_matches_batched_on_gpu() -> None:
    """GDB ``matvec_impl="triton"`` reproduces the batched energies on a GPU.

    The Triton kernel covers the same-spin doubles slices; every other channel
    stays batched, so the solve must match the fully batched reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    gdb_kwargs = {"block_matvec": "on"}
    reference = make_torch_sci_solver(**_solver_kwargs(mode="gdb", gdb_kwargs=gdb_kwargs))
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]

    triton_solver = make_torch_sci_solver(
        **_solver_kwargs(mode="gdb", gdb_kwargs=gdb_kwargs, device="cuda", matvec_impl="triton")
    )
    observed = [
        float(state.energy)
        for state in triton_solver([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]
    assert observed == pytest.approx(expected, abs=1e-7)


def test_tile_partition_matches_single_process() -> None:
    """A 2x2 tile partition reproduces the single-process energies (gloo, 4 ranks).

    Exercises the whole tile path end to end: the adapter builds the tile
    partition and the column-sliced helpers, the tile operator runs the
    staged slab/corner exchanges, the occupancies scatter back to global
    strings, and the result is gathered from per-tile rectangles.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs())
    expected = _energies(reference, h1, eri, strings)

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "tile": [2, 2],
            }
        )
    )
    try:
        observed = _energies(distributed, h1, eri, strings)
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)


def test_tile_spin_squared_matches_single_process() -> None:
    """A 2x2 tile partition reproduces the serial <S^2> diagnostic.

    With ``compute_spin_squared`` on, the tiled solve keeps the owner-computes
    path: the diagnostic is accumulated from the operator's staged exchanges
    (and one all-reduce) instead of a full-vector two-RDM build, so its value
    must match the serial solver's.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    reference = make_torch_sci_solver(**_solver_kwargs(compute_spin_squared=True))
    expected = _energies(reference, h1, eri, strings)
    expected_s2 = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            compute_spin_squared=True,
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "tile": [2, 2],
                "persistent": False,
            },
        )
    )
    try:
        observed = _energies(distributed, h1, eri, strings)
        observed_s2 = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    # The diagnostic did not push the call back to the replicated matvec.
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


def test_tile_spin_sq_penalty_matches_single_process() -> None:
    """A 2x2 tile partition reproduces the serial spin_sq penalty solve.

    With ``spin_sq`` set, the tiled solve composes ``H + lam (S^2 - target)``
    on the tile operator (the S^2 term reuses the staged exchanges), so the
    physical energies and the final ``<S^2>`` must match the serial solver's.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()

    reference = make_torch_sci_solver(**_solver_kwargs())
    ref_result = reference(
        [(strings, strings)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "tile": [2, 2],
                "persistent": False,
            }
        )
    )
    try:
        result = distributed(
            [(strings, strings)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    # The penalty did not push the call back to the replicated matvec.
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


def test_tile_warm_start_matches_single_process() -> None:
    """A warm start on a 2x2 tile reproduces the serial warm solve.

    The cached interleaved-Wick state is pasted into the extended basis one
    tile rectangle at a time, so the warm energy must match the serial
    row-embedding path.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    core = strings[::2]

    reference = make_torch_sci_solver(**_solver_kwargs())
    ref_out = reference([(core, core)], h1, eri, NORB, NELEC, probe_method="warm_start")
    core_state = ref_out[1][0]
    ref_warm = reference(
        [(strings, strings)],
        h1,
        eri,
        NORB,
        NELEC,
        probe_method="warm_start",
        probe_state=[core_state],
    )[0]
    expected = [float(state.energy) for state in ref_warm.states]

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "tile": [2, 2],
                "persistent": False,
            }
        )
    )
    try:
        dist_out = distributed([(core, core)], h1, eri, NORB, NELEC, probe_method="warm_start")
        dist_core_state = dist_out[1][0]
        dist_warm = distributed(
            [(strings, strings)],
            h1,
            eri,
            NORB,
            NELEC,
            probe_method="warm_start",
            probe_state=[dist_core_state],
        )[0]
        observed = [float(state.energy) for state in dist_warm.states]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert subspace["tile_grid"] == [2, 2]


def test_one_d_tile_spin_sq_penalty_matches_single_process() -> None:
    """A penalty run without a tile spec keeps the owner-computes path.

    ``spin_sq`` now uses the tile machinery even when no grid was requested: a
    ``(P, 1)`` partition reproduces the 1-D row split, so the penalty stays on
    the local path instead of falling back to the replicated matvec.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()

    reference = make_torch_sci_solver(**_solver_kwargs())
    ref_result = reference(
        [(strings, strings)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "persistent": False,
            }
        )
    )
    try:
        result = distributed(
            [(strings, strings)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(reference.stats["spin_sq_final"], abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [4, 1]


def test_tile_full_rdm_matches_single_process() -> None:
    """A 2x2 tile partition reproduces the serial full-RDM fields.

    With ``compute_full_rdm`` on, the tiled solve keeps the owner-computes
    path: the spin-summed ``rdm1``/``rdm2`` come from the tiled builders over
    the operator's staged exchanges, so they must match the serial solver's.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()

    reference = make_torch_sci_solver(**_solver_kwargs(compute_full_rdm=True))
    ref_result = reference([(strings, strings)], h1, eri, NORB, NELEC)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_rdm1 = ref_result.states[0].rdm1
    expected_rdm2 = ref_result.states[0].rdm2

    distributed = make_torch_sci_solver(
        **_solver_kwargs(
            compute_full_rdm=True,
            distributed={
                "world_size": 4,
                "backend": "gloo",
                "timeout": 120,
                "matvec": "local",
                "tile": [2, 2],
                "persistent": False,
            },
        )
    )
    try:
        result = distributed([(strings, strings)], h1, eri, NORB, NELEC)[0]
        observed = [float(state.energy) for state in result.states]
        observed_rdm1 = result.states[0].rdm1
        observed_rdm2 = result.states[0].rdm2
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_rdm1 == pytest.approx(expected_rdm1, abs=1e-8)
    assert observed_rdm2 == pytest.approx(expected_rdm2, abs=1e-8)
    # The RDM post-processing did not push the call back to the replicated matvec.
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


def test_gdb_tile_partition_matches_single_process() -> None:
    """A 2x2 GDB pair-tile partition reproduces the single-process energies.

    Exercises the tiled owner-computes GDB path end to end: the adapter builds
    the uid-slab tile partition over the explicit pair basis, the tile operator
    runs the staged pair exchange, the αβ-swap projector runs on the scattered
    owned pairs (the default ``apply_ab_projection``), and the result is
    gathered back from the per-tile pair lists.
    """
    _skip_without_gloo()
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
    from torch_sci_solver.torch_impl.qiskit_adapter import _pair_ab_swap_permutation

    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]
    # The full Cartesian product is closed under (α, β) → (β, α), so the
    # projector really runs on the tile path instead of being skipped.
    basis = pair_basis_from_arrays(pairs_a, pairs_b, device="cpu")
    k_swap, _ = _pair_ab_swap_permutation(basis, NORB, torch.device("cpu"), torch.float64)
    assert k_swap is not None

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": [2, 2],
            "persistent": False,
        },
    })
    try:
        observed = [
            float(state.energy)
            for state in distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
        ]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    # The resolved grid reaches the benchmark's perf record.
    assert distributed.stats["per_call_log"][-1]["subspaces"][0]["tile_grid"] == [2, 2]


def test_gdb_tile_fp32_subspace_matches_fp64() -> None:
    """The tiled GDB path supports the fp32 subspace.

    Regression: the tile operator used to allocate its per-fragment output
    buffers with the input vector's dtype, so a narrow (fp32) subspace input
    made the full-precision slab kernels fail with an ``index_add_`` dtype
    mismatch. The buffers now follow the operator's precision.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "davidson_dtype": torch.float32,
        "davidson_tolerance": 1e-6,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": [2, 1],
            "persistent": False,
        },
    })
    try:
        observed = [
            float(state.energy)
            for state in distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
        ]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-5)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["davidson_dtype"] == "torch.float32"
    assert subspace["tile_grid"] == [2, 1]


def test_gdb_tile_spin_squared_matches_single_process() -> None:
    """A 2x2 GDB pair-tile partition reproduces the serial <S^2> diagnostic."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(
        mode="gdb", gdb_kwargs={"block_matvec": "on"}, compute_spin_squared=True
    )
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": [2, 2],
            "persistent": False,
        },
    })
    try:
        result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


def test_gdb_tile_auto_spin_sq_penalty_uses_the_one_d_tile_on_cpu() -> None:
    """A declined GDB auto grid keeps the penalty on the (P, 1) pair tile.

    The auto policy declines without a device budget, but the penalty composes
    on the implicit (P, 1) uid partition instead of the replicated solve.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference(
        [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": "auto",
            "persistent": False,
        },
    })
    try:
        result = distributed(
            [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["tile_grid"] == [4, 1]


def test_gdb_tile_full_rdm_matches_single_process() -> None:
    """A 2x2 GDB pair-tile partition reproduces the serial rdm1/rdm2 fields."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"}, compute_full_rdm=True)
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_rdm1 = ref_result.states[0].rdm1
    expected_rdm2 = ref_result.states[0].rdm2

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": [2, 2],
            "persistent": False,
        },
    })
    try:
        result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
        observed = [float(state.energy) for state in result.states]
        observed_rdm1 = result.states[0].rdm1
        observed_rdm2 = result.states[0].rdm2
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_rdm1 == pytest.approx(expected_rdm1, abs=1e-8)
    assert observed_rdm2 == pytest.approx(expected_rdm2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


def test_gdb_tile_spin_sq_penalty_matches_single_process() -> None:
    """A 2x2 GDB pair tile reproduces the serial spin_sq penalty solve."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference(
        [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": [2, 2],
            "persistent": False,
        },
    })
    try:
        result = distributed(
            [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [2, 2]


@pytest.mark.parametrize("block_matvec", ["on", "off"])
def test_gdb_flat_spin_sq_penalty_matches_single_process(block_matvec: str) -> None:
    """A GDB solve with no tile applies the penalty on the 1-D owned block.

    The penalty used to force the replicated matvec when no grid was
    configured, whose O(K) spin-swap tables dwarf the basis itself.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": block_matvec})
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference(
        [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "persistent": False,
        },
    })
    try:
        result = distributed(
            [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert "tile_grid" not in subspace
    assert subspace["spin_penalty_layout"] == "1d"


def test_gdb_flat_spin_sq_penalty_fp32_subspace() -> None:
    """The 1-D penalty path supports the narrow fp32 Davidson subspace."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference(
        [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
    )[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = reference.stats["spin_sq_final"]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "davidson_dtype": torch.float32,
        "davidson_tolerance": 1e-6,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "persistent": False,
        },
    })
    try:
        result = distributed(
            [(pairs_a, pairs_b)], h1, eri, NORB, NELEC, spin_sq=2.0, spin_penalty_lambda=0.3
        )[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = distributed.stats["spin_sq_final"]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-5)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-4)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["davidson_dtype"] == "torch.float32"
    assert subspace["spin_penalty_layout"] == "1d"


@pytest.mark.parametrize("with_penalty", [False, True], ids=["diagnostic", "penalty"])
def test_gdb_flat_spin_squared_diagnostic_matches_single_process(with_penalty: bool) -> None:
    """A no-tile GDB solve computes <S^2> on the 1-D block.

    The diagnostic uses the same flat sweep as the penalty, with or without the
    penalty itself, and builds no O(K) swap table.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(
        mode="gdb", gdb_kwargs={"block_matvec": "on"}, compute_spin_squared=True
    )
    call = dict(spin_sq=2.0, spin_penalty_lambda=0.3) if with_penalty else {}
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, **call)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_s2 = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "persistent": False,
        },
    })
    try:
        result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC, **call)[0]
        observed = [float(state.energy) for state in result.states]
        observed_s2 = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-7)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert "tile_grid" not in subspace
    assert subspace["rdm_accumulation"] == "occupancies_only"
    if with_penalty:
        assert subspace["spin_penalty_layout"] == "1d"
    else:
        assert "spin_penalty_layout" not in subspace


def test_gdb_tile_auto_keeps_one_d_when_it_fits() -> None:
    """GDB ``tile="auto"`` keeps the 1-D block matvec on a small basis.

    The pair-tile layout is slower than the 1-D block matvec at equal size, so
    the auto policy only tiles when the 1-D working set would not fit; a tiny
    basis must therefore run on the flat owner-computes path.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    reference = make_torch_sci_solver(**kwargs)
    expected = [
        float(state.energy)
        for state in reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
    ]

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": "auto",
            "persistent": False,
        },
    })
    try:
        observed = [
            float(state.energy)
            for state in distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0].states
        ]
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert "tile_grid" not in subspace


def test_gdb_tile_auto_with_full_rdm_keeps_the_one_d_tile_on_cpu() -> None:
    """``tile="auto"`` declines on CPU, but a full RDM stays on a (P, 1) tile.

    The auto policy runs unchanged when a diagnostic is requested; without a
    device budget it keeps the 1-D block matvec. The full-RDM build then uses a
    (P, 1) uid grid -- the pair analogue of the 1-D row split -- instead of the
    replicated full-vector solve, and must still match serial.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    pairs_a = np.repeat(strings, len(strings))
    pairs_b = np.tile(strings, len(strings))
    kwargs = _solver_kwargs(
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
        compute_full_rdm=True,
        compute_spin_squared=True,
    )
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_rdm1 = ref_result.states[0].rdm1
    expected_rdm2 = ref_result.states[0].rdm2
    expected_s2 = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": "auto",
            "persistent": False,
        },
    })
    try:
        result = distributed([(pairs_a, pairs_b)], h1, eri, NORB, NELEC)[0]
        observed = [float(state.energy) for state in result.states]
        observed_rdm1 = result.states[0].rdm1
        observed_rdm2 = result.states[0].rdm2
        observed_s2 = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_rdm1 == pytest.approx(expected_rdm1, abs=1e-8)
    assert observed_rdm2 == pytest.approx(expected_rdm2, abs=1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["tile_grid"] == [4, 1]


def test_tpb_tile_auto_with_full_rdm_keeps_the_one_d_tile_on_cpu() -> None:
    """A declined TPB auto grid keeps the RDM and <S^2> work on a (P, 1) tile.

    Without a device budget the auto policy keeps 1-D. The post-processing then
    uses the implicit (P, 1) partition -- the same layout as the 1-D row split
    -- instead of falling back to the replicated full-vector solve.
    """
    _skip_without_gloo()
    h1, eri, strings = _synthetic_h4()
    kwargs = _solver_kwargs(compute_full_rdm=True, compute_spin_squared=True)
    reference = make_torch_sci_solver(**kwargs)
    ref_result = reference([(strings, strings)], h1, eri, NORB, NELEC)[0]
    expected = [float(state.energy) for state in ref_result.states]
    expected_rdm1 = ref_result.states[0].rdm1
    expected_rdm2 = ref_result.states[0].rdm2
    expected_s2 = list(reference.stats["spin_squared"])

    distributed = make_torch_sci_solver(**{
        **kwargs,
        "distributed": {
            "world_size": 4,
            "backend": "gloo",
            "timeout": 120,
            "matvec": "local",
            "tile": "auto",
            "persistent": False,
        },
    })
    try:
        result = distributed([(strings, strings)], h1, eri, NORB, NELEC)[0]
        observed = [float(state.energy) for state in result.states]
        observed_rdm1 = result.states[0].rdm1
        observed_rdm2 = result.states[0].rdm2
        observed_s2 = list(distributed.stats["spin_squared"])
        subspace = distributed.stats["per_call_log"][-1]["subspaces"][0]
    finally:
        distributed.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert observed_rdm1 == pytest.approx(expected_rdm1, abs=1e-8)
    assert observed_rdm2 == pytest.approx(expected_rdm2, abs=1e-8)
    assert observed_s2 == pytest.approx(expected_s2, abs=1e-8)
    assert subspace["distributed_matvec"] == "local"
    assert subspace["tile_grid"] == [4, 1]


def test_worker_result_save_uses_pickle_protocol_4(tmp_path, monkeypatch) -> None:
    """Worker results use protocol 4, so they can exceed the 4 GiB protocol-2 cap.

    A 4·10⁸-determinant fp64 eigenvector alone is 3.2 GiB, and the result dict
    around it crosses 4 GiB; protocol 2 cannot serialize a string that large.
    """
    from torch_sci_solver.torch_impl import distributed_solver

    recorded = {}

    def fake_save(_obj, _path, **kwargs):
        """Record the kwargs without touching the filesystem."""
        recorded.update(kwargs)

    monkeypatch.setattr(distributed_solver.torch, "save", fake_save)
    distributed_solver._save_worker_result(str(tmp_path / "result.pt"), {"result": None})
    assert recorded.get("pickle_protocol", 0) >= 4
