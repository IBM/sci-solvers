"""Tests for the public solver interface: config, capabilities and preflight.

These are CPU-only and deliberately small: they exercise the construction-time
validation, the JSON-friendly discovery objects, the call-time gate replication
in :meth:`TorchSCISolver.preflight`, the ``qiskit`` kwarg routing fixes and the
transport completeness of the new knobs.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import json
import warnings

import numpy as np
import pytest
import torch
from torch_sci_solver import qiskit as qiskit_shim
from torch_sci_solver.torch_impl import distributed_solver
from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver, make_torch_sci_solver

NORB = 4
NELEC = (2, 2)


def _synthetic_h4():
    """Deterministic Hermitian H4 integrals and the full (2α, 2β) strings."""
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


def _base(**kwargs):
    """A small deterministic CPU solver."""
    options = dict(
        device="cpu",
        davidson_block_size=8,
        max_davidson_iters=100,
        davidson_tolerance=1e-8,
        apply_ab_projection=False,
    )
    options.update(kwargs)
    return TorchSCISolver(**options)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_is_json_friendly_and_normalized() -> None:
    """``solver.config`` has requested/resolved halves and survives json.dumps."""
    solver = _base(mode="gdb", gdb_kwargs={"block_matvec": True}, n_roots=2)
    config = solver.config
    assert set(config) == {"requested", "resolved"}
    assert json.loads(json.dumps(config)) == config

    requested = config["requested"]
    assert requested["mode"] == "gdb"
    # gdb_kwargs defaults are filled even though only one key was supplied.
    assert requested["gdb_kwargs"]["block_matvec"] == "on"
    assert requested["gdb_kwargs"]["budget_bytes"] is None

    resolved = config["resolved"]
    assert resolved["mode"] == "gdb"
    assert resolved["matvec_impl"] == "batched"
    assert resolved["matvec_impl_requested"] == "auto"
    assert resolved["dtype"] == "torch.float64"
    assert resolved["davidson_dtype"] == "torch.float64"
    assert resolved["device"] == "cpu"
    assert resolved["devices"] == ["cpu"]
    assert resolved["distributed"] is None
    assert resolved["gdb_kwargs"]["block_matvec"] == "on"
    assert resolved["unknown_call_kwargs"] == "ignore"


def test_config_reports_distributed_and_new_knobs() -> None:
    """The resolved distributed config carries the new launch/hand-back keys."""
    solver = _base(
        distributed={
            "world_size": 2,
            "backend": "gloo",
            "tile": "auto",
            "hand_back": "memmap",
            "hand_back_bytes": 4096,
            "budget_bytes": 1 << 20,
            "worker_startup_timeout_s": 42,
        },
        memory_budget_bytes=1 << 21,
        table_dtype="int32",
        compact_eri="off",
        triton_block=256,
        diagnostics={"dump_dir": "/tmp/torch_sci_probe"},
        deterministic=True,
    )
    config = solver.config
    assert json.loads(json.dumps(config)) == config
    resolved = config["resolved"]
    cfg = resolved["distributed"]
    assert cfg["world_size"] == 2
    assert cfg["backend"] == "gloo"
    assert cfg["matvec"] == "local"
    assert cfg["tile"] == "auto"
    assert cfg["persistent"] is True
    assert cfg["init_method"].startswith("tcp://")
    assert cfg["hand_back"] == "memmap"
    assert cfg["hand_back_bytes"] == 4096
    assert cfg["budget_bytes"] == 1 << 20
    assert cfg["worker_startup_timeout_s"] == 42.0
    assert resolved["memory_budget_bytes"] == 1 << 21
    assert resolved["table_dtype"] == "int32"
    assert resolved["compact_eri"] == "off"
    assert resolved["triton_block"] == 256
    assert resolved["deterministic"] is True


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
def test_capabilities_per_mode() -> None:
    """The capability map reports each mode's layouts and feature support."""
    caps = _base(mode="gdb", gdb_kwargs={"block_matvec": "on"}).capabilities()
    assert set(caps["modes"]) == {"tpb", "gdb", "spinless"}

    tpb = caps["modes"]["tpb"]
    assert tpb["layouts"] == ["product"]
    assert tpb["gdb_block_matvec"] is False
    assert tpb["table_dtype"] == ["torch.int64", "torch.int32"]

    gdb = caps["modes"]["gdb"]
    assert gdb["layouts"] == ["pairs", "structural"]
    assert gdb["gdb_block_matvec"] is True
    assert gdb["gdb_structural"] is True
    assert gdb["gdb_move_cache"] is True

    spinless = caps["modes"]["spinless"]
    assert spinless["spin_penalty"] is False
    assert spinless["spin_squared"] is False
    assert spinless["init_vectors"] is True

    assert "tol" in caps["accepted_call_kwargs"]
    assert "max_cycle" in caps["rejected_call_kwargs"]
    assert caps["unknown_call_kwargs"] == "ignore"
    assert caps["triton_pass_backends"]["alpha_doubles"] == "tiled"
    assert "same_spin_doubles" in gdb["triton_passes"]
    assert "alphabeta_doubles" in gdb["triton_passes"]
    assert "doubles" in spinless["triton_passes"]


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def test_preflight_tpb_matches_real_call_plan() -> None:
    """Preflight reports the same dimension/path/backend the real solve uses."""
    h1, eri, strings = _synthetic_h4()
    solver = _base()
    plan = solver.preflight([(strings, strings)], NELEC, norb=NORB, tol=1e-6)
    assert plan["mode"] == "tpb"
    assert plan["basis"] == "product"
    assert plan["particle_sector"] == {"n_alpha": 2, "n_beta": 2}
    assert plan["subspaces"][0]["dimension"] == strings.size**2
    assert plan["subspaces"][0]["n_alpha"] == strings.size
    assert plan["subspaces"][0]["path"] == "single_process"
    assert plan["call_tol"] == 1e-6
    assert plan["warm_start"] is False

    solver([(strings, strings)], h1, eri, NORB, NELEC, tol=1e-6)
    logged = solver.stats["per_call_log"][-1]["subspaces"][0]
    assert plan["subspaces"][0]["dimension"] == logged["N"]
    assert plan["matvec_impl"] == solver.matvec_impl == "batched"
    assert plan["subspaces"][0]["start"] == "cold"


def test_preflight_gdb_and_structural_dimensions() -> None:
    """GDB plans report the pair count; structural plans report the block union."""
    _, _, strings = _synthetic_h4()
    gdb = _base(mode="gdb")
    plan = gdb.preflight([(strings, strings)], NELEC, norb=NORB)
    assert plan["basis"] == "pairs"
    assert plan["subspaces"][0]["layout"] == "pairs"
    assert plan["subspaces"][0]["dimension"] == strings.size  # one pair per index

    dist_gdb = _base(
        mode="gdb",
        davidson_block_size=4,
        distributed={"world_size": 2, "backend": "gloo"},
    )
    dist_plan = dist_gdb.preflight([(strings, strings)], NELEC, norb=NORB)
    assert dist_plan["subspaces"][0]["path"] == "local"
    assert dist_plan["matvec_impl"] == "batched"

    blocks = [[(strings[:3].tolist(), strings[:2].tolist())]]
    structural = _base(
        mode="gdb",
        gdb_kwargs={"structural_blocks": blocks, "block_matvec": "on"},
        distributed={"world_size": 2, "backend": "gloo", "tile": "force"},
    )
    plan = structural.preflight([(strings, strings)], NELEC, norb=NORB)
    assert plan["basis"] == "structural"
    assert plan["subspaces"][0]["layout"] == "structural"
    assert plan["subspaces"][0]["dimension"] == 3 * 2
    assert plan["subspaces"][0]["path"] == "local"
    assert plan["subspaces"][0]["tile_grid"] is not None


def test_preflight_gdb_spin_sq_without_a_tile_stays_local() -> None:
    """A GDB spin penalty/diagnostic with no tile plans the 1-D block.

    Full RDMs still require a tile grid, so that request keeps the replicated
    matvec without one.
    """
    _, _, strings = _synthetic_h4()
    cfg = {"world_size": 2, "backend": "gloo"}
    solver = _base(mode="gdb", davidson_block_size=4, distributed=cfg)
    plan = solver.preflight([(strings, strings)], NELEC, norb=NORB, spin_sq=0.0)
    subspace = plan["subspaces"][0]
    assert subspace["path"] == "local"
    assert subspace["tile_grid"] is None

    diagnostic = _base(
        mode="gdb", davidson_block_size=4, distributed=cfg, compute_spin_squared=True
    )
    plan = diagnostic.preflight([(strings, strings)], NELEC, norb=NORB)
    assert plan["subspaces"][0]["path"] == "local"

    rdm = _base(mode="gdb", davidson_block_size=4, distributed=cfg, compute_full_rdm=True)
    plan = rdm.preflight([(strings, strings)], NELEC, norb=NORB)
    assert plan["subspaces"][0]["path"] == "replicated"


def test_preflight_spinless_dimension_and_gates() -> None:
    """A scalar nelec plans the flat spin-orbital path and runs its gates."""
    _, _, strings = _synthetic_h4()
    solver = TorchSCISolver(device="cpu", dtype=torch.complex128, davidson_block_size=8)
    plan = solver.preflight([strings], 2, norb=NORB)
    assert plan["basis"] == "spinless"
    assert plan["particle_sector"]["n_particles"] == 2
    assert plan["subspaces"][0]["dimension"] == strings.size

    with pytest.raises(ValueError, match="spin_sq is not supported"):
        solver.preflight([strings], 2, norb=NORB, spin_sq=0.0)


def test_preflight_warm_start_plan() -> None:
    """An init_vectors entry marks the planned start as warm."""
    _, _, strings = _synthetic_h4()
    solver = _base()
    vec = np.ones(strings.size**2)
    plan = solver.preflight([(strings, strings)], NELEC, norb=NORB, init_vectors=[vec])
    assert plan["warm_start"] is True
    assert plan["subspaces"][0]["start"] == "warm"


def test_preflight_raises_the_call_gate_errors() -> None:
    """Preflight surfaces the same call-time errors as a real solve."""
    h1, eri, strings = _synthetic_h4()
    solver = _base()
    with pytest.raises(TypeError, match="max_cycle"):
        solver.preflight([(strings, strings)], NELEC, norb=NORB, max_cycle=10)
    with pytest.raises(TypeError, match="max_cycle"):
        solver([(strings, strings)], h1, eri, NORB, NELEC, max_cycle=10)
    with pytest.raises(ValueError, match="popcount"):
        solver([(strings + 1, strings)], h1, eri, NORB, NELEC)
    with pytest.raises(ValueError, match="init_vectors"):
        solver.preflight([(strings, strings)], NELEC, norb=NORB, init_vectors=[np.ones(3)])
    with pytest.raises(ValueError, match="popcount"):
        solver.preflight([(strings + 1, strings)], NELEC, norb=NORB)
    with pytest.raises(TypeError, match="unknown per-call keyword"):
        _base(unknown_call_kwargs="error").preflight(
            [(strings, strings)], NELEC, norb=NORB, bogus=1
        )
    replicated = _base(
        n_roots=2,
        davidson_missed_root_check=True,
        distributed={"world_size": 2, "backend": "gloo", "matvec": "replicated"},
    )
    with pytest.raises(NotImplementedError, match="missed_root_check"):
        replicated.preflight([(strings, strings)], NELEC, norb=NORB)


def test_scalar_nelec_rejects_devices_and_gdb_kwargs() -> None:
    """The spinless path no longer silently ignores devices= or GDB options."""
    h1, eri, strings = _synthetic_h4()
    solver = TorchSCISolver(device="cpu")
    solver.devices = (torch.device("cpu"), torch.device("cpu"))
    with pytest.raises(ValueError, match="devices="):
        solver([strings], h1, eri, NORB, 2)
    with pytest.raises(ValueError, match="devices="):
        solver.preflight([strings], 2, norb=NORB)

    gdb = TorchSCISolver(device="cpu", mode="gdb", gdb_kwargs={"block_matvec": "on"})
    with pytest.raises(ValueError, match="gdb_kwargs"):
        gdb([strings], h1, eri, NORB, 2)
    with pytest.raises(ValueError, match="gdb_kwargs"):
        gdb.preflight([strings], 2, norb=NORB)


# --------------------------------------------------------------------------- #
# validation tightening
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"davidson_block_size": 0}, "davidson_block_size"),
        ({"max_davidson_iters": 0}, "max_davidson_iters"),
        ({"davidson_tolerance": 0.0}, "davidson_tolerance"),
        ({"davidson_de_tolerance": -1.0}, "davidson_de_tolerance"),
        ({"budget_fraction": 0.0}, "budget_fraction"),
        ({"budget_fraction": 1.5}, "budget_fraction"),
        ({"dtype": torch.int64}, "dtype"),
        ({"seed_strategy": "bogus"}, "seed_strategy"),
        ({"n_roots": 5, "davidson_block_size": 4}, "n_roots"),
        ({"unknown_call_kwargs": "bogus"}, "unknown_call_kwargs"),
        ({"memory_budget_bytes": 0}, "memory_budget_bytes"),
        ({"table_dtype": "int8"}, "table_dtype"),
        ({"mode": "gdb", "table_dtype": "int32"}, "table_dtype"),
        ({"matvec_impl": "triton", "table_dtype": "int32"}, "table_dtype"),
        ({"compact_eri": "yes"}, "compact_eri"),
        ({"triton_block": 0}, "triton_block"),
        ({"triton_block": 17}, "triton_block"),
        ({"diagnostics": {"bogus": 1}}, "diagnostics"),
        ({"deterministic": "yes"}, "deterministic"),
    ],
)
def test_constructor_validation_errors(kwargs, match) -> None:
    """Every tightened knob fails at construction with a descriptive error."""
    with pytest.raises((ValueError, TypeError), match=match):
        _base(**kwargs)


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"world_size": 0}, "world_size"),
        ({"world_size": 2, "backend": "gloo", "timeout": 0}, "timeout"),
        ({"world_size": 2, "backend": "gloo", "master_port": 70000}, "master_port"),
        ({"world_size": 2, "backend": "gloo", "init_method": "bogus://x"}, "init_method"),
        (
            {"world_size": 2, "backend": "gloo", "device_ids": [0, 1]},
            "device_ids",
        ),
        (
            {"world_size": 2, "backend": "gloo", "matvec": "replicated", "tile": "auto"},
            "tile",
        ),
        (
            {"world_size": 2, "backend": "gloo", "exchange_bytes": 16},
            "exchange_bytes",
        ),
        ({"world_size": 2, "backend": "gloo", "hand_back": "disk"}, "hand_back"),
        ({"world_size": 2, "backend": "gloo", "hand_back_bytes": -1}, "hand_back_bytes"),
        ({"world_size": 2, "backend": "gloo", "budget_bytes": -1}, "budget_bytes"),
        (
            {"world_size": 2, "backend": "gloo", "worker_startup_timeout_s": 0},
            "worker_startup_timeout_s",
        ),
    ],
)
def test_distributed_validation_errors(raw, match) -> None:
    """Distributed launch knobs are validated at construction."""
    with pytest.raises(ValueError, match=match):
        distributed_solver.parse_distributed_config(raw, device="cpu", cuda_device_count=0)


# --------------------------------------------------------------------------- #
# qiskit shim / wrapper routing
# --------------------------------------------------------------------------- #
def test_qiskit_init_keys_include_misrouted_constructor_knobs() -> None:
    """The constructor knobs are no longer left in the per-call kwargs."""
    expected = {
        "matvec_impl",
        "distributed",
        "davidson_dtype",
        "davidson_fp32_shift",
        "davidson_fp32_energy",
        "davidson_correction",
        "davidson_restart",
        "davidson_missed_root_check",
        "davidson_de_tolerance",
    }
    assert expected <= qiskit_shim._SOLVER_INIT_KEYS
    init_kwargs, call_kwargs = qiskit_shim._split_kwargs(dict.fromkeys(expected))
    assert set(init_kwargs) == expected
    assert call_kwargs == {}


def test_root_factory_warns_about_the_projection_default() -> None:
    """The qiskit shim keeps its projection default but makes the choice visible."""
    with pytest.warns(UserWarning, match="apply_ab_projection"):
        qiskit_shim.make_torch_sci_solver(device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = qiskit_shim.make_torch_sci_solver(device="cpu", apply_ab_projection=True)
    assert solver.apply_ab_projection is True


def test_worker_cfg_and_factory_kwargs_forward_new_knobs() -> None:
    """Every new knob reaches the worker config and the spawned factory kwargs."""
    solver = _base(
        unknown_call_kwargs="warn",
        memory_budget_bytes=1 << 20,
        table_dtype="int32",
        compact_eri="off",
        triton_block=64,
        diagnostics={"dump_dir": "/tmp/torch_sci_probe"},
        deterministic=True,
    )
    cfg = solver._worker_cfg()
    for key in (
        "unknown_call_kwargs",
        "memory_budget_bytes",
        "table_dtype",
        "compact_eri",
        "triton_block",
        "diagnostics",
        "deterministic",
    ):
        assert key in cfg
    factory = solver._factory_kwargs
    for key in (
        "unknown_call_kwargs",
        "memory_budget_bytes",
        "table_dtype",
        "compact_eri",
        "triton_block",
        "diagnostics",
        "deterministic",
    ):
        assert key in factory
    assert factory["distributed"] is None


def test_int32_tables_force_batched_on_a_cuda_rank() -> None:
    """A narrowed TPB table cannot be read by the Triton kernels."""
    assert (
        distributed_solver._worker_matvec_impl(
            "auto",
            mode="tpb",
            dev=torch.device("cuda", 0),
            backend="nccl",
            table_dtype="int32",
        )
        == "batched"
    )
    with pytest.raises(ValueError, match="table_dtype"):
        distributed_solver._worker_matvec_impl(
            "triton",
            mode="tpb",
            dev=torch.device("cuda", 0),
            backend="nccl",
            table_dtype="int32",
        )


def test_distributed_runtime_config_round_trip() -> None:
    """``parse_distributed_config`` + the solver runtime hook carry the new options."""
    cfg = distributed_solver.parse_distributed_config(
        {
            "world_size": 2,
            "backend": "gloo",
            "hand_back": "memory",
            "hand_back_bytes": 123,
            "budget_bytes": 456,
        },
        device="cpu",
        cuda_device_count=0,
    )
    solver = _base()
    distributed_solver._apply_runtime_config(solver, cfg)
    assert solver._runtime_hand_back == "memory"
    assert solver._runtime_hand_back_bytes == 123
    assert solver._memory_budget(None) == min(solver.memory_budget_bytes or 456, 456)


def _fake_engine(return_value=None):
    """Minimal engine stand-in for the src/sci_solvers wrapper tests."""

    class FakeEngine:
        dtype = torch.float64
        max_davidson_iters = 20
        stats = {"per_call_log": []}

        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return [] if return_value is None else return_value

    return FakeEngine()


def test_src_wrapper_translates_compute_rdm2_and_maxiter() -> None:
    """The sci_solvers wrapper maps generic aliases onto the engine's names."""
    from sci_solvers.energy_solvers.torch_sci_solver import (
        TorchSCISolver as WrapperSolver,
    )

    wrapper = WrapperSolver(symmetrize_spin=False, device="cpu", compute_rdm2=True)
    assert wrapper._solver.compute_full_rdm is True

    _, _, strings = _synthetic_h4()
    engine = _fake_engine(return_value=[])
    wrapper._solver = engine
    wrapper.sci_solver_callable_eigen_states(
        [(strings, strings)],
        np.zeros((NORB, NORB)),
        np.zeros((NORB,) * 4),
        NORB,
        NELEC,
        maxiter=17,
        tol=1e-5,
    )
    call = engine.calls[-1]
    assert call["max_davidson_iters"] == 17
    assert call["tol"] == 1e-5


def test_src_wrapper_handles_the_warm_start_tuple() -> None:
    """A ``(results, states)`` engine return is unpacked without breaking iteration."""
    from sci_solvers.energy_solvers.torch_sci_solver import (
        TorchSCISolver as WrapperSolver,
    )

    wrapper = WrapperSolver(symmetrize_spin=False, device="cpu")
    states = [object()]
    engine = _fake_engine(return_value=([], states))
    wrapper._solver = engine
    _, _, strings = _synthetic_h4()
    result = wrapper.sci_solver_callable_eigen_states(
        [(strings, strings)], np.zeros((NORB, NORB)), np.zeros((NORB,) * 4), NORB, NELEC
    )
    assert result == []
    assert wrapper._last_probe_states is states


def test_src_wrapper_exposes_config_capabilities_and_preflight() -> None:
    """The workflow-facing wrapper delegates the new public interface."""
    from sci_solvers.energy_solvers.torch_sci_solver import (
        TorchSCISolver as WrapperSolver,
    )

    wrapper = WrapperSolver(symmetrize_spin=False, device="cpu", davidson_block_size=8)
    assert wrapper.config["resolved"]["mode"] == "tpb"
    assert "tpb" in wrapper.capabilities()["modes"]
    _, _, strings = _synthetic_h4()
    plan = wrapper.preflight([(strings, strings)], NELEC, norb=NORB)
    assert plan["subspaces"][0]["dimension"] == strings.size**2


# --------------------------------------------------------------------------- #
# MPS device options (construction only: no MPS hardware needed)
# --------------------------------------------------------------------------- #
def test_constructor_rejects_options_mps_cannot_run() -> None:
    """float64/complex128, Triton and distributed= raise at construction on MPS."""
    for dtype in (torch.float64, torch.complex128):
        with pytest.raises(ValueError, match="no float64 or complex128"):
            make_torch_sci_solver(device="mps", dtype=dtype, davidson_tolerance=1e-5)
    with pytest.raises(ValueError, match="no float64 or complex128"):
        make_torch_sci_solver(devices=["mps"], davidson_tolerance=1e-5)
    for mode in ("tpb", "gdb"):
        with pytest.raises(ValueError, match="needs a CUDA device"):
            make_torch_sci_solver(
                device="mps",
                dtype=torch.float32,
                mode=mode,
                matvec_impl="triton",
                davidson_tolerance=1e-5,
            )
    with pytest.raises(ValueError, match="MPS exposes a single device"):
        make_torch_sci_solver(
            device="mps",
            dtype=torch.float32,
            davidson_tolerance=1e-5,
            distributed={"world_size": 2, "backend": "gloo"},
        )


def test_mps_tolerance_floor_applies_through_devices() -> None:
    """``devices=["mps"]`` rebinds the primary device and clamps the tolerance."""
    from torch_sci_solver.torch_impl.qiskit_adapter import MPS_DAVIDSON_TOLERANCE_FLOOR

    with pytest.warns(RuntimeWarning, match="davidson_tolerance"):
        solver = make_torch_sci_solver(
            devices=["mps"], dtype=torch.float32, davidson_tolerance=1e-9
        )
    assert solver.davidson_tolerance == MPS_DAVIDSON_TOLERANCE_FLOOR


def test_mps_free_bytes_is_capped_by_the_metal_headroom(monkeypatch) -> None:
    """On MPS the free budget is min(host available, Metal recommended - allocated)."""
    import types

    import psutil

    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 10 * 2**30)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 4 * 2**30)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(available=32 * 2**30)
    )
    solver = make_torch_sci_solver(device="mps", dtype=torch.float32, davidson_tolerance=1e-5)
    assert solver._free_bytes(torch.device("mps")) == 6 * 2**30
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: types.SimpleNamespace(available=2 * 2**30)
    )
    assert solver._free_bytes(torch.device("mps")) == 2 * 2**30


def test_capabilities_report_mps_limits() -> None:
    """``capabilities()`` on an MPS solver lists only what MPS can run."""
    caps = make_torch_sci_solver(
        device="mps", dtype=torch.float32, davidson_tolerance=1e-5
    ).capabilities()
    assert "triton" not in caps["matvec_impls"]
    for entry in caps["modes"].values():
        assert set(entry["dtypes"]) == {"torch.float32", "torch.complex64"}
        assert entry["distributed_matvec"] == []
        assert entry["distributed_tile"] is False
        assert entry["gdb_structural"] is False
        assert entry["triton_passes"] == {}
    assert "structural" not in caps["modes"]["gdb"]["layouts"]
    assert "tiled_half_strings" not in caps["modes"]["spinless"]["layouts"]
    cpu_caps = make_torch_sci_solver(device="cpu").capabilities()
    assert {"torch.float64", "torch.float32"} <= set(cpu_caps["modes"]["spinless"]["dtypes"])
    assert cpu_caps["modes"]["tpb"]["distributed_tile"] is True


# --------------------------------------------------------------------------- #
# Capacity diagnostics (preflight)
# --------------------------------------------------------------------------- #
def _random_system(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """8-fold symmetric random integrals."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((norb,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    return h1, sum(raw.transpose(perm) for perm in perms) / 8.0


def _capacity_strings() -> np.ndarray:
    """All 3-electron strings of 8 orbitals (N = 3136 for the product basis)."""
    combos = itertools.combinations(range(8), 3)
    return np.array(sorted(sum(1 << o for o in c) for c in combos), dtype=np.int64)


@pytest.mark.parametrize(
    ("mode", "dtype", "factorized"),
    [
        ("tpb", torch.float64, False),
        ("tpb", torch.float32, False),
        ("tpb", torch.complex128, False),
        ("tpb", torch.float64, True),
        ("gdb", torch.float64, False),
    ],
    ids=["tpb-fp64", "tpb-fp32", "tpb-c128", "tpb-factorized", "gdb"],
)
def test_preflight_capacity_predicts_the_block_the_solver_picks(mode, dtype, factorized) -> None:
    """Under a byte cap that binds below 30, the predicted block is the solve's block."""
    strings = _capacity_strings()
    h1, eri = _random_system(8)
    if dtype.is_complex:
        h1 = h1.astype(np.complex128)
    if mode == "gdb":
        ci = [(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))]
    else:
        ci = [(strings, strings)]
    # About 12 blocks of C and HC plus the work vectors for N = 3136 at 8 bytes.
    cap = (2 * 12 + 9) * len(strings) ** 2 * 8
    solver = make_torch_sci_solver(
        device="cpu",
        dtype=dtype,
        mode=mode,
        apply_ab_projection=False,
        memory_budget_bytes=cap,
        ab_factorization="cholesky" if factorized else "off",
    )
    capacity = solver.preflight(ci, (3, 3), norb=8)["subspaces"][0]["capacity"]
    assert 2 < capacity["davidson_block"] < 30
    assert capacity["davidson_block_basis"] == "budget_cap"
    solver(ci, h1, eri, 8, (3, 3))
    used = solver.stats["per_call_log"][-1]["subspaces"][0]["davidson_block_size"]
    assert capacity["davidson_block"] == used
    assert capacity["fits"] is True and capacity["approximate"] is False
    assert capacity["dimension_per_rank"] == len(strings) ** 2


def test_preflight_capacity_applies_the_distributed_block_clamp() -> None:
    """A distributed solve raises a tiny explicit block to ``n_roots + 3``."""
    strings = _capacity_strings()
    solver = make_torch_sci_solver(
        device="cpu",
        apply_ab_projection=False,
        davidson_block_size=2,
        distributed={"world_size": 2, "backend": "gloo", "matvec": "local"},
    )
    capacity = solver.preflight([(strings, strings)], (3, 3), norb=8)["subspaces"][0]["capacity"]
    assert capacity["davidson_block"] == 4
    assert capacity["approximate"] is True  # the solve cuts rows by cost


def test_preflight_capacity_terms() -> None:
    """Work vectors, the distributed split and an impossible budget are reported."""
    strings = _capacity_strings()
    ci = [(strings, strings)]
    base = make_torch_sci_solver(device="cpu", apply_ab_projection=False)
    off = base.preflight(ci, (3, 3), norb=8)["subspaces"][0]["capacity"]
    factorized = make_torch_sci_solver(
        device="cpu", apply_ab_projection=False, ab_factorization="cholesky"
    ).preflight(ci, (3, 3), norb=8)["subspaces"][0]["capacity"]
    n = len(strings) ** 2
    assert (
        factorized["per_rank_bytes"]["matvec_work"]
        == off["per_rank_bytes"]["matvec_work"] + 3 * n * 8
    )
    assert off["per_rank_bytes"]["tables_upper_bound"] > 0
    tiny = make_torch_sci_solver(
        device="cpu", apply_ab_projection=False, memory_budget_bytes=1024
    ).preflight(ci, (3, 3), norb=8)["subspaces"][0]["capacity"]
    assert tiny["fits"] is False and tiny["davidson_block"] == 2
    split = make_torch_sci_solver(
        device="cpu",
        apply_ab_projection=False,
        davidson_block_size=8,
        distributed={"world_size": 2, "backend": "gloo", "matvec": "local"},
    ).preflight(ci, (3, 3), norb=8)["subspaces"][0]["capacity"]
    assert split["dimension_per_rank"] == -(-n // 2)
    assert split["device_free_bytes"] is None  # the parent stays off the rank devices
    assert split["per_rank_bytes"]["davidson"] == 2 * 8 * split["dimension_per_rank"] * 8


@pytest.mark.parametrize("mode", ["gdb", "spinless"])
def test_preflight_capacity_on_flat_layouts(mode: str) -> None:
    """GDB and spinless plans carry a capacity estimate too."""
    strings = _capacity_strings()
    if mode == "gdb":
        solver = make_torch_sci_solver(device="cpu", mode="gdb", apply_ab_projection=False)
        ci, nelec = [(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))], (3, 3)
    else:
        solver = make_torch_sci_solver(device="cpu", apply_ab_projection=False)
        ci, nelec = [strings], 3
    capacity = solver.preflight(ci, nelec, norb=8)["subspaces"][0]["capacity"]
    terms = capacity["per_rank_bytes"]
    n = capacity["dimension_per_rank"]
    assert terms["matvec_work"] == 4 * n * 8 and terms["diagonal"] == 8 * n
    assert terms["gathered_input"] == 0  # single process: nothing is gathered
    assert terms["tables_upper_bound"] is None  # the flat paths stream their tables
    if mode == "gdb":
        assert n == len(strings) ** 2
        assert terms["pair_list"] == 16 * n
        assert terms["eri"] == 16**4 * 8  # dense spin-orbital below 64 spin orbitals
    else:
        assert n == len(strings)
        assert terms["pair_list"] == 0 and terms["eri"] == 8**4 * 8  # spin-orbital ERI
    assert capacity["total_bytes"] == sum(v or 0 for v in terms.values())
    assert capacity["davidson_block"] >= 2


def test_float64_energy_refinement_applies_to_mps_only() -> None:
    """Only an MPS solver swaps its float32 Ritz values for float64 CPU quotients."""
    cpu = make_torch_sci_solver(device="cpu", dtype=torch.float32)
    assert cpu._refines_fp32_energy(1e-8) is False
    mps = make_torch_sci_solver(device="mps", dtype=torch.float32, davidson_tolerance=1e-5)
    assert mps._refines_fp32_energy(1e-5) is True
    assert mps._refines_fp32_energy(3e-3) is False  # auto keeps the Ritz value at loose tol
    ritz = make_torch_sci_solver(
        device="mps", dtype=torch.float32, davidson_tolerance=1e-5, davidson_fp32_energy="ritz"
    )
    assert ritz._refines_fp32_energy(1e-5) is False
