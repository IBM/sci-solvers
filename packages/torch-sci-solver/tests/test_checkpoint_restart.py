"""CPU (gloo) checkpoint/restart tests for the distributed solver.

Each restart test runs an uninterrupted reference and an interrupted run that
stops after one Davidson restart cycle, then resumes the same checkpoint with
the full budget. The resumed energy must match the uninterrupted solve; the
checkpoint is rank-local, so no rank ever gathers the full subspace.

The mismatch tests cover the three resume policies (``auto`` / ``off`` /
``require``) and the manifest identity fields a resumed run must match. GPU
runs reuse the same scenario with NCCL and are skipped unless two CUDA devices
are available.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.torch_impl.distributed_solver import (
    CHECKPOINT_FORMAT_VERSION,
    parse_distributed_config,
)

NORB = 6
NELEC = (3, 3)
N_ROOTS = 2
BLOCK_SIZE = 8


def _skip_without_gloo() -> None:
    """Skip when this torch build has no distributed gloo backend."""
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")


def _skip_without_nccl() -> None:
    """Skip unless two CUDA devices and the NCCL backend are available."""
    if not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
    ):
        pytest.skip("two CUDA devices and NCCL are required")


def _synthetic_system(norb: int, n_elec: int):
    """Deterministic Hermitian integrals and the full CI strings for ``(norb, n_elec)``."""
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
    """Deterministic solver settings shared by every arm of a restart test."""
    kwargs = dict(
        device="cpu",
        matvec_impl="batched",
        davidson_block_size=BLOCK_SIZE,
        max_davidson_iters=200,
        davidson_tolerance=1e-8,
        n_roots=N_ROOTS,
        davidson_missed_root_check=False,
        apply_ab_projection=False,
        verbose=False,
    )
    kwargs.update(overrides)
    return kwargs


def _scenario(name: str):
    """Return ``(h1, eri, basis, solver_kwargs, distributed_cfg)`` for one layout."""
    h1, eri, strings = _synthetic_system(NORB, NELEC[0])
    kwargs = _solver_kwargs()
    cfg = {
        "world_size": 2,
        "backend": "gloo",
        "timeout": 300,
        "persistent": False,
        "matvec": "local",
    }
    basis = (strings, strings)
    if name == "tpb_tiled":
        cfg["tile"] = [2, 1]
    elif name == "gdb_1d":
        kwargs["mode"] = "gdb"
        basis = (np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    elif name == "complex_replicated":
        h1 = h1.astype(np.complex128)
        eri = eri.astype(np.complex128)
        kwargs["dtype"] = torch.complex128
        cfg["matvec"] = "replicated"
    return h1, eri, basis, kwargs, cfg


def _energies(solver, basis, h1, eri, **call_kwargs):
    """Solve one subspace and return its root energies."""
    result = solver([basis], h1, eri, NORB, NELEC, **call_kwargs)
    return [float(state.energy) for state in result[0].states]


def _checkpoint_files(root: Path) -> list[Path]:
    """Every file written under a checkpoint root."""
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_checkpoint_config_validation_and_preflight(tmp_path: Path) -> None:
    """The new keys validate at construction and surface in config/preflight."""
    cfg = parse_distributed_config(
        {"world_size": 2, "backend": "gloo"},
        device="cpu",
        cuda_device_count=0,
    )
    assert cfg["checkpoint_dir"] is None
    assert cfg["checkpoint_every"] == 50
    assert cfg["checkpoint_resume"] == "auto"

    root = tmp_path / "checkpoints"
    explicit = parse_distributed_config(
        {
            "world_size": 2,
            "backend": "gloo",
            "checkpoint_dir": str(root),
            "checkpoint_every": 7,
            "checkpoint_resume": "require",
        },
        device="cpu",
        cuda_device_count=0,
    )
    assert explicit["checkpoint_dir"] == str(root)
    assert explicit["checkpoint_every"] == 7
    assert explicit["checkpoint_resume"] == "require"

    with pytest.raises(ValueError, match="checkpoint_every"):
        parse_distributed_config(
            {
                "world_size": 2,
                "backend": "gloo",
                "checkpoint_dir": str(root),
                "checkpoint_every": 0,
            },
            device="cpu",
            cuda_device_count=0,
        )
    with pytest.raises(ValueError, match="checkpoint_resume"):
        parse_distributed_config(
            {
                "world_size": 2,
                "backend": "gloo",
                "checkpoint_dir": str(root),
                "checkpoint_resume": "sometimes",
            },
            device="cpu",
            cuda_device_count=0,
        )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        parse_distributed_config(
            {"world_size": 2, "backend": "gloo", "checkpoint_every": 5},
            device="cpu",
            cuda_device_count=0,
        )
    with pytest.raises(ValueError, match="checkpoint_dir"):
        parse_distributed_config(
            {"world_size": 2, "backend": "gloo", "checkpoint_resume": "off"},
            device="cpu",
            cuda_device_count=0,
        )

    solver = make_torch_sci_solver(
        **_solver_kwargs(
            davidson_block_size=BLOCK_SIZE,
            distributed={
                "world_size": 2,
                "backend": "gloo",
                "checkpoint_dir": str(root),
                "checkpoint_every": 7,
            },
        )
    )
    resolved = solver.config["resolved"]["distributed"]
    assert resolved["checkpoint_dir"] == str(root)
    assert resolved["checkpoint_every"] == 7
    assert resolved["checkpoint_resume"] == "auto"
    h1, eri, strings = _synthetic_system(4, 2)
    plan = solver.preflight(
        [(strings, strings)],
        (2, 2),
        norb=4,
        one_body_tensor=h1,
        two_body_tensor=eri,
    )
    checkpoint = plan["checkpoint"]
    assert checkpoint["enabled"] is True
    assert checkpoint["dir"] == str(root)
    assert checkpoint["every"] == 7
    assert checkpoint["resume"] == "auto"
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert checkpoint["subspaces"][0]["manifest"] is False
    assert checkpoint["subspaces"][0]["would_resume"] is None


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["tpb_1d", "tpb_tiled", "gdb_1d"])
def test_interrupted_resume_matches_uninterrupted(tmp_path: Path, scenario: str) -> None:
    """A resumed multi-iteration run reproduces the uninterrupted energy.

    The interrupted arm stops after one restart cycle (``max_davidson_iters=1``
    with a tolerance it cannot reach) and leaves a checkpoint at its last
    iteration boundary; the second call resumes from that iteration with the
    full budget.
    """
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario(scenario)
    root = tmp_path / "checkpoints"

    uninterrupted = make_torch_sci_solver(**kwargs, distributed=dict(cfg))
    try:
        expected = _energies(uninterrupted, basis, h1, eri)
    finally:
        uninterrupted.close()

    interrupted = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(
            interrupted,
            basis,
            h1,
            eri,
            max_davidson_iters=1,
            tol=1e-12,
        )
        files = _checkpoint_files(root)
        assert (root / "subspace_0" / "manifest.json").is_file()
        assert (root / "subspace_0" / "rank_0.pt").is_file()
        assert (root / "subspace_0" / "rank_1.pt").is_file()

        observed = _energies(interrupted, basis, h1, eri)
        log = interrupted.stats.get("checkpoint_log", [])
        resumed = [entry for entry in log if entry["action"] == "resumed"]
        assert resumed, log
        assert resumed[-1]["iteration"] > 0
    finally:
        interrupted.close()

    assert observed == pytest.approx(expected, abs=1e-8)
    assert files


@pytest.mark.slow
def test_interrupted_resume_matches_uninterrupted_complex(tmp_path: Path) -> None:
    """The complex replicated (Krylov) path checkpoints and resumes too."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("complex_replicated")
    root = tmp_path / "checkpoints"

    uninterrupted = make_torch_sci_solver(**kwargs, distributed=dict(cfg))
    try:
        expected = _energies(uninterrupted, basis, h1, eri)
    finally:
        uninterrupted.close()

    solver = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(solver, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
        observed = _energies(solver, basis, h1, eri)
        log = solver.stats.get("checkpoint_log", [])
        assert any(entry["action"] == "resumed" for entry in log), log
    finally:
        solver.close()

    assert observed == pytest.approx(expected, abs=1e-8)


@pytest.mark.slow
def test_checkpoint_every_bounds_the_number_of_writes(tmp_path: Path) -> None:
    """A write happens only at iteration boundaries divisible by ``checkpoint_every``."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    root = tmp_path / "checkpoints"
    solver = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 3},
    )
    try:
        # 2 restart cycles * block size 8 = 16 iterations; the tolerance is
        # unreachable, so the loop runs every one of them and writes at
        # iterations 3, 6, 9, 12 and 15.
        _energies(solver, basis, h1, eri, max_davidson_iters=2, tol=1e-30)
        writes = int(solver.stats.get("checkpoint_writes", 0))
        log = solver.stats.get("checkpoint_log", [])
    finally:
        solver.close()
    assert writes == 5
    assert len(_checkpoint_files(root)) == 3  # manifest + one shard per rank
    manifest = json.loads((root / "subspace_0" / "manifest.json").read_text())
    assert manifest["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert manifest["iteration"] == 15
    assert manifest["state_layout"] == "local"
    assert log[0]["action"] == "cold"


@pytest.mark.slow
def test_checkpointing_off_writes_nothing_and_matches_reference(tmp_path: Path) -> None:
    """Without ``checkpoint_dir`` no state is written and the energy is unchanged."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    reference = make_torch_sci_solver(**kwargs)
    expected = _energies(reference, basis, h1, eri)

    solver = make_torch_sci_solver(**kwargs, distributed=dict(cfg))
    try:
        observed = _energies(solver, basis, h1, eri)
        assert "checkpoint_writes" not in solver.stats
        assert "checkpoint_log" not in solver.stats
    finally:
        solver.close()
    assert observed == pytest.approx(expected, abs=1e-8)
    assert not any(tmp_path.iterdir())


@pytest.mark.slow
def test_mismatched_basis_raises_under_auto_and_cold_starts_under_off(
    tmp_path: Path,
) -> None:
    """A different basis is rejected by ``auto`` and overwritten under ``off``."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    root = tmp_path / "checkpoints"
    strings = basis[0]
    smaller = (strings[1:], strings[1:])

    writer = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(writer, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
    finally:
        writer.close()

    mismatched = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        with pytest.raises(Exception, match="different run"):
            _energies(mismatched, smaller, h1, eri)
    finally:
        mismatched.close()

    overrider = make_torch_sci_solver(
        **kwargs,
        distributed={
            **cfg,
            "checkpoint_dir": str(root),
            "checkpoint_every": 1,
            "checkpoint_resume": "off",
        },
    )
    try:
        observed = _energies(overrider, smaller, h1, eri, max_davidson_iters=1)
        log = overrider.stats.get("checkpoint_log", [])
    finally:
        overrider.close()
    assert log[-1]["action"] == "disabled"
    assert len(observed) == N_ROOTS
    manifest = json.loads((root / "subspace_0" / "manifest.json").read_text())
    assert manifest["identity"]["dimension"] == len(strings[1:]) ** 2


@pytest.mark.slow
def test_incomplete_shard_cold_starts_under_auto_and_raises_under_require(
    tmp_path: Path,
) -> None:
    """A crash between the shard writes is a cold start under auto, an error under require."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    root = tmp_path / "checkpoints"

    writer = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(writer, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
    finally:
        writer.close()
    shard = root / "subspace_0" / "rank_1.pt"
    assert shard.is_file()
    shard.unlink()

    auto = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        observed = _energies(auto, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
        assert auto.stats["checkpoint_log"][-1]["action"] == "cold_incomplete"
    finally:
        auto.close()
    assert len(observed) == N_ROOTS
    # The cold run wrote a complete pair again; break it once more for require.
    shard.unlink()

    required = make_torch_sci_solver(
        **kwargs,
        distributed={
            **cfg,
            "checkpoint_dir": str(root),
            "checkpoint_every": 1,
            "checkpoint_resume": "require",
        },
    )
    try:
        with pytest.raises(Exception, match="incomplete"):
            _energies(required, basis, h1, eri)
    finally:
        required.close()


@pytest.mark.slow
def test_tiled_superset_checkpoint_rejects_a_changed_active_list(tmp_path: Path) -> None:
    """A checkpoint is tied to the caller's list, not to the padded product."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_system(NORB, NELEC[0])
    product_a = np.repeat(strings, len(strings))
    product_b = np.tile(strings, len(strings))
    keep = np.sort(np.random.default_rng(0).permutation(len(product_a))[:360])
    other = np.sort(np.random.default_rng(1).permutation(len(product_a))[:360])
    basis = (product_a[keep], product_b[keep])
    # Both lists embed in the same 400-pair product, so their dimension, unique
    # counts and rank partition agree; only the membership mask differs.
    changed = (product_a[other], product_b[other])
    root = tmp_path / "checkpoints"
    kwargs = _solver_kwargs(mode="gdb", gdb_kwargs={"block_matvec": "on"})
    cfg = {
        "world_size": 2,
        "backend": "gloo",
        "timeout": 300,
        "persistent": False,
        "matvec": "local",
        "tile": "force",
        "checkpoint_dir": str(root),
        "checkpoint_every": 1,
    }

    writer = make_torch_sci_solver(**kwargs, distributed=dict(cfg))
    try:
        _energies(writer, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
        assert (root / "subspace_0" / "manifest.json").is_file()
        perf = writer.stats["per_call_log"][-1]["subspaces"][0]
        assert perf["pair_superset"] is True
    finally:
        writer.close()

    resumed = make_torch_sci_solver(**kwargs, distributed=dict(cfg))
    try:
        with pytest.raises(Exception, match="different run"):
            _energies(resumed, changed, h1, eri)
    finally:
        resumed.close()


@pytest.mark.slow
def test_mismatched_integrals_fingerprint_raises_under_auto(tmp_path: Path) -> None:
    """The same basis with different integrals is a different run."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    root = tmp_path / "checkpoints"

    writer = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(writer, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
    finally:
        writer.close()

    reader = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        with pytest.raises(Exception, match="integrals_fingerprint"):
            _energies(reader, basis, h1 * 1.01, eri)
    finally:
        reader.close()


@pytest.mark.slow
def test_mismatched_world_size_raises_under_auto(tmp_path: Path) -> None:
    """A checkpoint written by two ranks is not resumed by four ranks."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    root = tmp_path / "checkpoints"

    writer = make_torch_sci_solver(
        **kwargs,
        distributed={**cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(writer, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
    finally:
        writer.close()

    reader = make_torch_sci_solver(
        **kwargs,
        distributed={
            **cfg,
            "world_size": 4,
            "checkpoint_dir": str(root),
            "checkpoint_every": 1,
        },
    )
    try:
        with pytest.raises(Exception, match="world_size"):
            _energies(reader, basis, h1, eri)
    finally:
        reader.close()


@pytest.mark.slow
def test_require_without_a_manifest_raises(tmp_path: Path) -> None:
    """``checkpoint_resume='require'`` refuses to cold-start."""
    _skip_without_gloo()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    solver = make_torch_sci_solver(
        **kwargs,
        distributed={
            **cfg,
            "checkpoint_dir": str(tmp_path / "empty"),
            "checkpoint_resume": "require",
        },
    )
    try:
        with pytest.raises(Exception, match="no checkpoint manifest"):
            _energies(solver, basis, h1, eri)
    finally:
        solver.close()


@pytest.mark.gpu
@pytest.mark.slow
def test_nccl_interrupted_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """Two-rank NCCL resume reproduces the uninterrupted TPB energy."""
    _skip_without_nccl()
    h1, eri, basis, kwargs, cfg = _scenario("tpb_1d")
    # NCCL collectives need CUDA operands; the shared scenario is CPU-based.
    kwargs = {**kwargs, "device": "cuda"}
    nccl_cfg = {
        "world_size": 2,
        "backend": "nccl",
        "device_ids": [0, 1],
        "timeout": 300,
    }
    root = tmp_path / "checkpoints"

    uninterrupted = make_torch_sci_solver(**kwargs, distributed=dict(nccl_cfg))
    try:
        expected = _energies(uninterrupted, basis, h1, eri)
    finally:
        uninterrupted.close()

    solver = make_torch_sci_solver(
        **kwargs,
        distributed={**nccl_cfg, "checkpoint_dir": str(root), "checkpoint_every": 1},
    )
    try:
        _energies(solver, basis, h1, eri, max_davidson_iters=1, tol=1e-12)
        observed = _energies(solver, basis, h1, eri)
    finally:
        solver.close()
    assert observed == pytest.approx(expected, abs=1e-7)


@pytest.mark.slow
def test_checkpoint_rejects_spinless_path(tmp_path: Path) -> None:
    """The unimplemented spinless path fails loudly instead of dropping state."""
    _skip_without_gloo()
    h1, eri, strings = _synthetic_system(4, 2)
    solver = make_torch_sci_solver(
        **_solver_kwargs(
            distributed={
                "world_size": 2,
                "backend": "gloo",
                "persistent": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
            }
        )
    )
    try:
        with pytest.raises(Exception, match="spinless path"):
            solver([strings], h1, eri, 4, 2)
    finally:
        solver.close()
