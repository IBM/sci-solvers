"""Aggregate per-run JSONs from ``run_matrix.py`` into CSV + markdown summaries.

Reads every ``per_run/**/*.json`` under ``output_dir`` (so a ladder directory with one
``size_<N>/per_run`` per size works too) and emits:

- ``output_dir/summary/across_molecules.csv`` — long-form: one row per cell,
  all metrics; None becomes empty CSV cell.
- ``output_dir/summary/per_molecule/[molecule].md`` — one table per molecule,
  one row per variant × circuit × num, with ``*`` flagging fastest mean time
  and lowest total energy in that molecule's block.
- ``output_dir/summary/summary.md`` — top-level cross-molecule solver comparison:
  win rates, mean ranks, failure rates, wrong-spin rates, mean ΔE from best.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from jsonargparse import CLI

CSV_COLUMNS = [
    "molecule",
    "circuit_type",
    "circuit_num",
    "shots_stem",
    "variant_id",
    "status",
    "n_orbitals",
    "n_electrons_alpha",
    "n_electrons_beta",
    "n_qubits",
    "n_unique_bitstrings",
    "num_runs",
    "subspace_dim",
    "n_alpha_dets",
    "n_beta_dets",
    "total_energy_mean",
    "total_energy_stdev",
    "electronic_energy_mean",
    "electronic_energy_stdev",
    "solver_time_mean_s",
    "solver_time_stdev_s",
    "spin_squared_mean",
    "spin_squared_stdev",
    "multiplicity_mean",
    "multiplicity_stdev",
    "ms2",
    "wrong_spin",
    "peak_ram_mb_mean",
    "peak_ram_mb_stdev",
    "peak_vram_mb_mean",
    "peak_vram_mb_stdev",
    "gpu_name",
    "error_type",
    "error_message",
]

# ⟨S²⟩ tolerance: flag as wrong spin if |computed - expected| exceeds this.
SPIN_TOLERANCE = 0.1


def _expected_spin_squared(ms2: int | None) -> float | None:
    """Return S(S+1) for a given MS2 value, or None if ms2 is unknown."""
    if ms2 is None:
        return None
    s = ms2 / 2.0
    return s * (s + 1)


def _wrong_spin(spin_sq: float | None, ms2: int | None) -> bool | None:
    """True if ⟨S²⟩ deviates from expected by more than SPIN_TOLERANCE."""
    if spin_sq is None:
        return None
    expected = _expected_spin_squared(ms2)
    if expected is None:
        return None
    return abs(spin_sq - expected) > SPIN_TOLERANCE


def _row(payload: dict) -> dict:
    cell = payload["cell"]
    var = payload["variant"]
    r = payload.get("results") or {}
    err = payload.get("error") or {}
    prov = payload.get("provenance") or {}
    ms2 = cell.get("ms2")
    spin_sq = r.get("spin_squared_mean")
    # Older JSONs (pre-status-field) fall back to error-derived status.
    status = payload.get("status") or ("error" if err else "ok")
    return {
        "molecule": cell["molecule"],
        "circuit_type": cell["circuit_type"],
        "circuit_num": cell["circuit_num"],
        "shots_stem": cell["shots_stem"],
        "variant_id": var["id"],
        "status": status,
        "n_orbitals": cell.get("n_orbitals"),
        "n_electrons_alpha": cell.get("n_electrons_alpha"),
        "n_electrons_beta": cell.get("n_electrons_beta"),
        "n_qubits": cell.get("n_qubits"),
        "n_unique_bitstrings": cell.get("n_unique_bitstrings"),
        "num_runs": r.get("num_runs"),
        "subspace_dim": r.get("subspace_dim"),
        "n_alpha_dets": r.get("n_alpha_dets"),
        "n_beta_dets": r.get("n_beta_dets"),
        "total_energy_mean": r.get("total_energy_mean"),
        "total_energy_stdev": r.get("total_energy_stdev"),
        "electronic_energy_mean": r.get("electronic_energy_mean"),
        "electronic_energy_stdev": r.get("electronic_energy_stdev"),
        "solver_time_mean_s": r.get("solver_time_mean_s"),
        "solver_time_stdev_s": r.get("solver_time_stdev_s"),
        "spin_squared_mean": spin_sq,
        "spin_squared_stdev": r.get("spin_squared_stdev"),
        "multiplicity_mean": r.get("multiplicity_mean"),
        "multiplicity_stdev": r.get("multiplicity_stdev"),
        "ms2": ms2,
        "wrong_spin": _wrong_spin(spin_sq, ms2),
        "peak_ram_mb_mean": r.get("peak_ram_mb_mean"),
        "peak_ram_mb_stdev": r.get("peak_ram_mb_stdev"),
        "peak_vram_mb_mean": r.get("peak_vram_mb_mean"),
        "peak_vram_mb_stdev": r.get("peak_vram_mb_stdev"),
        "gpu_name": prov.get("gpu_name"),
        "error_type": err.get("type"),
        "error_message": err.get("message"),
    }


def _fmt(v: Any, spec: str = ".6g") -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return format(v, spec)
    return str(v)


def _per_molecule_markdown(mol: str, rows: list[dict]) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    best_time = min(ok, key=lambda x: x["solver_time_mean_s"]) if ok else None
    best_energy = min(ok, key=lambda x: x["total_energy_mean"]) if ok else None

    lines = [f"# {mol}", ""]
    lines.append(
        "| variant | circuit | num | time mean (s) | ± stdev | E_tot mean | ± stdev | ⟨S²⟩ | wrong spin | dim | status |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark_t = " *" if r is best_time else ""
        mark_e = " *" if r is best_energy else ""
        lines.append(
            f"| {r['variant_id']} | {r['circuit_type']} | {r['circuit_num']} | "
            f"{_fmt(r['solver_time_mean_s'])}{mark_t} | {_fmt(r['solver_time_stdev_s'])} | "
            f"{_fmt(r['total_energy_mean'], '.10g')}{mark_e} | {_fmt(r['total_energy_stdev'], '.3g')} | "
            f"{_fmt(r['spin_squared_mean'])} | {_fmt(r['wrong_spin'])} | "
            f"{_fmt(r['subspace_dim'])} | {r['status']} |"
        )
    lines.append("")
    lines.append("*`*`  = fastest mean solver-time / lowest mean total-energy in this table.*")
    return "\n".join(lines) + "\n"


def _solver_comparison_markdown(rows: list[dict]) -> str:
    """Cross-molecule solver comparison table for summary.md."""
    # Group ok rows by (molecule, circuit_type, circuit_num, shots_stem) — one
    # "problem instance" per group. Within each group rank solvers by energy and time.
    variants: list[str] = []
    seen: set[str] = set()
    for r in rows:
        v = r["variant_id"]
        if v not in seen:
            variants.append(v)
            seen.add(v)

    # stats accumulators per variant
    stats: dict[str, dict] = {
        v: {
            "energy_wins": 0,
            "time_wins": 0,
            "energy_rank_sum": 0,
            "time_rank_sum": 0,
            "instances": 0,
            "failures": 0,
            "wrong_spin": 0,
            "delta_e_sum": 0.0,
            "delta_e_count": 0,
            "time_sum": 0.0,
            "time_count": 0,
        }
        for v in variants
    }

    # group by problem instance
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["molecule"], r["circuit_type"], r["circuit_num"], r["shots_stem"])
        groups[key].append(r)

    n_instances = len(groups)

    for key, group in groups.items():
        ok = [r for r in group if r["status"] == "ok"]

        # failures
        for r in group:
            if r["status"] == "error":
                stats[r["variant_id"]]["failures"] += 1
            stats[r["variant_id"]]["instances"] += 1

        if not ok:
            continue

        # energy ranking (lower = better) — dense ranking (1, 1, 2)
        ok_e = [r for r in ok if r["total_energy_mean"] is not None]
        if ok_e:
            best_e = min(r["total_energy_mean"] for r in ok_e)
            sorted_e = sorted(ok_e, key=lambda x: x["total_energy_mean"])
            rank = 0
            prev = None
            for r in sorted_e:
                if prev is None or not math.isclose(r["total_energy_mean"], prev, abs_tol=1e-8):
                    rank += 1
                    prev = r["total_energy_mean"]
                stats[r["variant_id"]]["energy_rank_sum"] += rank
                stats[r["variant_id"]]["delta_e_sum"] += r["total_energy_mean"] - best_e
                stats[r["variant_id"]]["delta_e_count"] += 1
                if math.isclose(r["total_energy_mean"], best_e, abs_tol=1e-8):
                    stats[r["variant_id"]]["energy_wins"] += 1

        # time ranking (lower = better) — dense ranking (1, 1, 2)
        ok_t = [r for r in ok if r["solver_time_mean_s"] is not None]
        if ok_t:
            best_t = min(r["solver_time_mean_s"] for r in ok_t)
            sorted_t = sorted(ok_t, key=lambda x: x["solver_time_mean_s"])
            rank = 0
            prev = None
            for r in sorted_t:
                if prev is None or not math.isclose(r["solver_time_mean_s"], prev, rel_tol=1e-3):
                    rank += 1
                    prev = r["solver_time_mean_s"]
                stats[r["variant_id"]]["time_rank_sum"] += rank
                stats[r["variant_id"]]["time_sum"] += r["solver_time_mean_s"]
                stats[r["variant_id"]]["time_count"] += 1
                if math.isclose(r["solver_time_mean_s"], best_t, rel_tol=1e-3):
                    stats[r["variant_id"]]["time_wins"] += 1

        # wrong spin
        for r in ok:
            if r.get("wrong_spin"):
                stats[r["variant_id"]]["wrong_spin"] += 1

    lines = ["## Cross-molecule solver comparison", ""]
    lines.append(
        f"*{n_instances} problem instance(s) across {len({k[0] for k in groups})} molecule(s).*"
    )
    lines.append("")
    lines.append(
        "| variant | energy win% | time win% | mean energy rank | mean time rank"
        " | mean ΔE (Ha) | mean time (s) | wrong spin% | failure% |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for v in variants:
        s = stats[v]
        n = s["instances"] or 1
        e_rank = s["energy_rank_sum"] / s["delta_e_count"] if s["delta_e_count"] else None
        t_rank = s["time_rank_sum"] / s["time_count"] if s["time_count"] else None
        mean_de = s["delta_e_sum"] / s["delta_e_count"] if s["delta_e_count"] else None
        mean_t = s["time_sum"] / s["time_count"] if s["time_count"] else None
        e_win_pct = 100.0 * s["energy_wins"] / n_instances if n_instances else None
        t_win_pct = 100.0 * s["time_wins"] / n_instances if n_instances else None
        ws_pct = 100.0 * s["wrong_spin"] / n if n else None
        fail_pct = 100.0 * s["failures"] / n if n else None
        lines.append(
            f"| {v} | {_fmt(e_win_pct, '.1f')}% | {_fmt(t_win_pct, '.1f')}% | "
            f"{_fmt(e_rank, '.2f')} | {_fmt(t_rank, '.2f')} | "
            f"{_fmt(mean_de, '.2e')} | {_fmt(mean_t, '.3g')} | "
            f"{_fmt(ws_pct, '.1f')}% | {_fmt(fail_pct, '.1f')}% |"
        )

    lines.append("")
    lines.append(
        "*ΔE = deviation from best energy in that instance. "
        "Ranks: dense (1, 1, 2) — 1 = best, ties share the same rank. "
        "Win = tied-for-best within 1e-8 Ha (energy) or 0.1% (time).*"
    )
    return "\n".join(lines) + "\n"


def summarize(output_dir: str) -> None:
    """Aggregate every per-run JSON under ``output_dir`` into CSV + markdown."""
    root = Path(output_dir)
    summary = root / "summary"
    per_mol_dir = summary / "per_molecule"
    per_mol_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for p in sorted(root.glob("**/per_run/**/*.json")):
        rows.append(_row(json.loads(p.read_text())))
    if not rows:
        print(f"no per_run results found under {root}")
        return

    # long-form CSV
    csv_path = summary / "across_molecules.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # per-molecule markdown tables
    per_mol: dict[str, list[dict]] = {}
    for r in rows:
        per_mol.setdefault(r["molecule"], []).append(r)
    for mol, mrows in sorted(per_mol.items()):
        (per_mol_dir / f"{mol}.md").write_text(_per_molecule_markdown(mol, mrows))

    # top-level summary: per-molecule winners + cross-molecule comparison
    top = ["# Benchmark summary", ""]

    top.append(_solver_comparison_markdown(rows))
    top.append("")
    top.append("---")
    top.append("")
    top.append("## Per-molecule winners")
    top.append("")
    for mol, mrows in sorted(per_mol.items()):
        ok = [r for r in mrows if r["status"] == "ok"]
        best_time = min(ok, key=lambda x: x["solver_time_mean_s"]) if ok else None
        best_energy = min(ok, key=lambda x: x["total_energy_mean"]) if ok else None
        errors = [r for r in mrows if r["status"] == "error"]
        wrong = [r for r in ok if r.get("wrong_spin")]
        top.append(f"### {mol}")
        if best_time:
            top.append(
                f"- fastest (mean): **{best_time['variant_id']}** "
                f"({_fmt(best_time['solver_time_mean_s'])} s)"
            )
        if best_energy:
            top.append(
                f"- lowest energy (mean): **{best_energy['variant_id']}** "
                f"({_fmt(best_energy['total_energy_mean'], '.10g')} Ha)"
            )
        if wrong:
            top.append(
                f"- wrong spin (⟨S²⟩ > expected+{SPIN_TOLERANCE}): {', '.join(r['variant_id'] for r in wrong)}"
            )
        if errors:
            top.append(f"- errored: {', '.join(r['variant_id'] for r in errors)}")
        top.append("")

    (summary / "summary.md").write_text("\n".join(top))

    print(f"wrote {csv_path}")
    print(f"wrote {summary / 'summary.md'}")
    print(f"wrote {len(per_mol)} per-molecule table(s) under {per_mol_dir}")


if __name__ == "__main__":
    CLI(summarize, as_positional=False)
