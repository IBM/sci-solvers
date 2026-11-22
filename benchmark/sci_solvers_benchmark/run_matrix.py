"""Run every (molecule × input file × solver variant) cell of a benchmark config.

A config is a YAML file with ``solver_variants`` (each an ``id``, a solver ``name``,
``init_kwargs`` and ``call_kwargs``), optionally merged over a shared file named by
``common_file``; keys in the per-solver file win. Cells run one after another, all
molecules for one variant before the next variant starts, so two GPU solvers never hold a
CUDA context at once. A failing cell is recorded with ``status="error"`` and the run goes
on. A cell whose JSON already exists is skipped, so an interrupted run can be resumed.

Inputs are discovered under ``base_dir``::

    <base_dir>/molecules/<molecule>/*.fcidump              # FCIDUMP (*.dat also accepted)
    <base_dir>/molecules/<molecule>/<family>/<n>/<input>   # one cell per input file

with ``<family>`` one of ``sqdrift``, ``lucj`` or ``hf_expansion``. Output::

    <output_dir>/config.yaml
    <output_dir>/per_run/<molecule>/<family>/<n>/<input stem>__<variant id>.json

Usage::

    python -m sci_solvers_benchmark.run_matrix --config_file configs/solvers/torch_sci_gpu.yaml \\
        --base_dir <base_dir> --output_dir <output_dir>

Notes
-----
For the synthetic family (``hf_expansion``) a nonzero subsample size selects the file
built for that ``n_alpha`` (``hf_product_<n_alpha>.npz``) instead of subsampling, since
subsampling would replace the controlled subspace with a random one.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import datetime
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml
from jsonargparse import CLI

from sci_solvers_benchmark.fcidump import read_fcidump_header
from sci_solvers_benchmark.run_single import run_single
from sci_solvers_benchmark.shots import SUPPORTED_SUFFIXES, load_shots

FAMILIES = ("sqdrift", "lucj", "hf_expansion")
SYNTHETIC_PREFIX = "hf_product_"


def _synthetic_size(stem: str) -> int | None:
    tail = stem.removeprefix(SYNTHETIC_PREFIX)
    return int(tail) if stem.startswith(SYNTHETIC_PREFIX) and tail.isdigit() else None


def _find_fcidump(mol_dir: Path) -> Path | None:
    for pattern in ("*.fcidump", "*.dat", "*.txt"):
        found = sorted(mol_dir.glob(pattern))
        if found:
            return found[0]
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware() -> dict[str, Any]:
    """CPU, RAM and GPU description of this machine, recorded once per run."""
    cpu = platform.processor() or None
    ram_gb = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_gb = round(int(line.split()[1]) / 1024**2, 1)
                break
    except OSError:
        pass
    gpus: list[str] = []
    try:
        import torch

        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return {
        "cpu_model": cpu,
        "n_cpus_available": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "total_ram_gb": ram_gb,
        "n_gpus": len(gpus),
        "gpu_names": gpus,
        "gpu_name": gpus[0] if gpus else None,
    }


def _cells(base_dir: Path, molecules: list[str] | str, families: list[str] | str) -> list:
    """One dict per (molecule, family, n, input file), with the problem's metadata."""
    root = base_dir / "molecules"
    if molecules == "auto":
        molecules = sorted(d.name for d in root.iterdir() if d.is_dir())
    cells = []
    for mol in molecules:
        mol_dir = root / mol
        fcidump = _find_fcidump(mol_dir)
        if fcidump is None:
            print(f"[skip] {mol}: no FCIDUMP in {mol_dir}", file=sys.stderr)
            continue
        h = read_fcidump_header(fcidump)
        fcidump_sha = _sha256(fcidump)
        on_disk = {d.name for d in mol_dir.iterdir() if d.is_dir()}
        wanted = FAMILIES if families == "auto" else families
        for family in [f for f in wanted if f in on_disk]:
            for num_dir in sorted(p for p in (mol_dir / family).iterdir() if p.is_dir()):
                files = sorted(
                    (p for p in num_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES),
                    key=lambda p: (_synthetic_size(p.stem) or 0, p.name),
                )
                for shots in files:
                    cells.append({
                        "molecule": mol,
                        "circuit_type": family,
                        "circuit_num": num_dir.name,
                        "fcidump": str(fcidump),
                        "fcidump_relpath": str(fcidump.relative_to(base_dir)),
                        "fcidump_sha256": fcidump_sha,
                        "shots": str(shots),
                        "shots_stem": shots.stem,
                        "shots_relpath": str(shots.relative_to(base_dir)),
                        "shots_sha256": _sha256(shots),
                        "ms2": h["ms2"],
                        "n_orbitals": h["norb"],
                        "n_electrons_alpha": h["n_alpha"],
                        "n_electrons_beta": h["n_beta"],
                        "n_qubits": 2 * h["norb"],
                        "n_unique_bitstrings": len(load_shots(shots)),
                    })
    return cells


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config, merged over the ``common_file`` it names (if any)."""
    cfg = yaml.safe_load(path.read_text())
    common_ref = cfg.pop("common_file", None)
    if common_ref is None:
        return cfg
    base = yaml.safe_load((path.parent / common_ref).read_text())
    base.update(cfg)
    return base


def run_matrix(
    config_file: str,
    output_dir: str,
    base_dir: str | None = None,
    molecules: list[str] | None = None,
    circuit_types: list[str] | None = None,
    subsample_size: int | None = None,
    num_runs: int | None = None,
    seed: int | None = None,
    warmup: bool | None = None,
    symmetrize_spin: bool | None = None,
    inject_hf: bool | None = None,
    measure_memory: bool | None = None,
    nroots: int | None = None,
    run_timeout: int | None = None,
) -> None:
    """Run the config's matrix and write one JSON per cell under ``output_dir/per_run``.

    Every argument after ``output_dir`` overrides the config value of the same name;
    ``base_dir``, ``molecules`` and ``circuit_types`` are top-level keys, the rest live
    under ``common``. ``base_dir`` must be given here or in the config.

    Raises:
        ValueError: No ``base_dir``.
    """
    cfg_path = Path(config_file)
    cfg = load_config(cfg_path)
    for key, value in (
        ("base_dir", base_dir),
        ("molecules", molecules),
        ("circuit_types", circuit_types),
    ):
        if value is not None:
            cfg[key] = value
    if not cfg.get("base_dir"):
        raise ValueError("set base_dir in the config or pass --base_dir")
    common: dict[str, Any] = dict(cfg.get("common") or {})
    overrides = {
        "subsample_size": subsample_size,
        "num_runs": num_runs,
        "seed": seed,
        "warmup": warmup,
        "symmetrize_spin": symmetrize_spin,
        "inject_hf": inject_hf,
        "measure_memory": measure_memory,
        "nroots": nroots,
        "run_timeout": run_timeout,
    }
    common.update({k: v for k, v in overrides.items() if v is not None})
    size = int(common.pop("subsample_size", 0))

    base = Path(cfg["base_dir"])
    out = Path(output_dir)
    (out / "per_run").mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out / "config.yaml")

    variants = cfg["solver_variants"]
    cells = _cells(base, cfg.get("molecules", "auto"), cfg.get("circuit_types", "auto"))
    hardware = _hardware()
    resolved = {
        "config_file": str(cfg_path),
        "molecules": cfg.get("molecules"),
        "circuit_types": cfg.get("circuit_types"),
        "common": {**common, "subsample_size": size},
        "solver_variants": variants,
    }
    total = len(cells) * len(variants)
    print(f"running {total} cells: {len(cells)} inputs × {len(variants)} variants", flush=True)

    i = done = 0
    for var in variants:
        for cell in cells:
            i += 1
            tag = f"{cell['molecule']}/{cell['circuit_type']}/{cell['shots_stem']} × {var['id']}"
            path = (
                out
                / "per_run"
                / cell["molecule"]
                / cell["circuit_type"]
                / cell["circuit_num"]
                / f"{cell['shots_stem']}__{var['id']}.json"
            )
            if path.exists():
                print(f"[{i}/{total}] [skip] {tag} (already done)", flush=True)
                continue
            cell_size = size
            if cell["circuit_type"] == "hf_expansion" and size:
                n_alpha = _synthetic_size(cell["shots_stem"])
                if n_alpha is not None and n_alpha != size:
                    continue
                cell_size = 0  # the file is already the requested size
            print(f"[{i}/{total}] {tag}", flush=True)
            payload: dict[str, Any] = {
                "cell": cell,
                "variant": var,
                "common": {**common, "subsample_size": cell_size},
                "solver_config": resolved,
                "provenance": hardware,
                "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            try:
                payload["results"] = run_single(
                    fcidump=cell["fcidump"],
                    shots=cell["shots"],
                    solver_name=var["name"],
                    solver_init_kwargs=var.get("init_kwargs"),
                    solver_call_kwargs=var.get("call_kwargs"),
                    subsample_size=cell_size,
                    **common,
                )
                payload["status"], payload["error"] = "ok", None
            except Exception as e:  # noqa: BLE001 - one failing cell must not stop the run
                payload["results"], payload["status"] = None, "error"
                payload["error"] = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
                print(f"  [error] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str))
            done += 1
    print(f"done: {done} cells run")


if __name__ == "__main__":
    CLI(run_matrix, as_positional=False)
