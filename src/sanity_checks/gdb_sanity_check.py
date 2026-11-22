"""Sanity-check the GDB (``mode="gdb"``) solvers on physical Hamiltonians.

Runs every case on every solver it names, solving each as a tensor product (TPB) and as
the same determinants given as a GDB pair list, and prints one row per case and solver.
A row passes when GDB and TPB agree, within the bounds of ``gdb_physical.failures``, in
energy, ``rdm1``, ``rdm2`` and the ground-state vector up to a global phase. A fulqrum
mode other than the default (``arpack@fulqrum:csr_fast``, ``lobpcg@fulqrum:matrix_free``)
must also agree, within the same bounds, with the same solver in ``fulqrum_mode="csr"`` on
the same seed, in both bases (the ``mode_*`` columns). Rows that need fulqrum SKIP when it
is not installed. The process exits non-zero if any row fails unexpectedly (FAIL, or
XPASS on a case marked ``expected: fail``).

Cases come from YAML (see :mod:`gdb_cases` for the schema): each path is a case file, a
suite file or a directory of either. With no path, the suite's own PySCF cases run::

    uv run python src/sanity_checks/gdb_sanity_check.py
    uv run python src/sanity_checks/gdb_sanity_check.py src/sanity_checks/suites/default.yaml
    uv run python src/sanity_checks/gdb_sanity_check.py my_cases/ --solvers arpack,davidson \
        --json results.json
    uv run python src/sanity_checks/gdb_sanity_check.py --seeds 0-9   # ARPACK spread

ARPACK starts from a random vector, so its rows run once per seed
(``gdb_physical.seeded_arpack``): one fixed seed, 0, by default, which makes the run
reproducible, or every seed of ``--seeds`` (or a case's ``seeds:``) to sample its
run-to-run spread. The other solvers are deterministic and run once.

Two kinds of column are reported, not checked. ``overlap_conj`` is ``|<tpb*|gdb>|``: it
equals ``overlap`` for a real Hamiltonian, and is below 1 for a complex one unless GDB
returned the complex-conjugated vector. The ``exact_*`` columns compare GDB with the
exact reference where one exists; they carry the solver's own convergence error.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdb_cases import SkipCase, builtin_cases, discover, load  # noqa: E402
from gdb_physical import compare_gdb_tpb, failures  # noqa: E402

COLUMNS = [
    "energy",
    "rdm1",
    "rdm2",
    "overlap",
    "overlap_conj",
    "exact_energy",
    "exact_rdm1",
    "exact_rdm2",
    "mode_energy",
    "mode_rdm1",
    "mode_rdm2",
    "mode_overlap",
]


def _status(bad: list[str], expected_fail: bool) -> str:
    """PASS / FAIL, or XFAIL / XPASS for a case marked ``expected: fail``."""
    if expected_fail:
        return "XFAIL" if bad else "XPASS"
    return "FAIL" if bad else "PASS"


def _seeds(text: str) -> list[int]:
    """``"0,3,5-7"`` -> ``[0, 3, 5, 6, 7]``."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the suite and return the process exit code (0 when no row fails unexpectedly)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, help="case/suite YAMLs or directories")
    parser.add_argument("--solvers", help="comma-separated; overrides every case's list")
    parser.add_argument("--datasets", help="comma-separated case names to run")
    parser.add_argument("--json", type=Path, help="write every row to this JSON file")
    parser.add_argument(
        "--seeds", type=_seeds, help="ARPACK start-vector seeds, e.g. 0,3 or 0-9 (default 0)"
    )
    args = parser.parse_args(argv)

    solvers = args.solvers.split(",") if args.solvers else None
    cases = discover(args.paths, solvers) if args.paths else builtin_cases(solvers)
    if args.datasets:
        keep = set(args.datasets.split(","))
        cases = [c for c in cases if c.name in keep]

    print(
        f"{'case':<28} {'solver':<28} {'nelec':<8} {'dim':>7} {'|Im eri|':>8} "
        + " ".join(f"{c:>12}" for c in COLUMNS)
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            ham, strings = load(case)
        except SkipCase as exc:
            rows.append({"case": case.name, "status": "SKIP", "reason": str(exc)})
            print(f"{case.name:<28} SKIP: {exc}")
            continue
        im = float(np.abs(np.imag(ham.eri)).max())
        seeds = args.seeds or case.spec.get("seeds") or [0]
        csr: dict[tuple[str, int], Any] = {}  # (solver name, seed) -> its fulqrum csr states

        def csr_states(name: str, seed: int) -> Any:
            if (name, seed) not in csr:
                csr[name, seed] = compare_gdb_tpb(ham, f"{name}@fulqrum", strings, seed)[1]
            return csr[name, seed]

        for solver in case.solvers:
            name, _, backend = solver.partition("@")
            backend, _, mode = backend.partition(":")
            if backend == "fulqrum" and importlib.util.find_spec("fulqrum") is None:
                rows.append({"case": case.name, "solver": solver, "status": "SKIP"})
                print(f"{case.name:<28} {solver:<28} SKIP: fulqrum not installed")
                continue
            for seed in seeds if name == "arpack" else [0]:
                ref = csr_states(name, seed) if mode not in ("", "csr") else None
                diffs, states = compare_gdb_tpb(ham, solver, strings, seed, ref)
                if backend == "fulqrum" and mode in ("", "csr"):
                    csr[name, seed] = states
                bad = failures(diffs)
                status = _status(bad, case.expected_fail)
                rows.append({
                    "case": case.name,
                    "solver": solver,
                    "seed": seed,
                    "nelec": list(ham.nelec),
                    "status": status,
                    "failed": bad,
                    **diffs,
                })
                cells = " ".join(
                    f"{diffs[c]:12.3e}" if c in diffs else f"{'-':>12}" for c in COLUMNS
                )
                detail = f" {','.join(bad)}" if bad else ""
                label = f"{solver}/{seed}" if name == "arpack" else solver
                print(
                    f"{case.name:<28} {label:<28} {str(ham.nelec):<8} {diffs['dim']:>7} "
                    f"{im:8.3f} {cells}  {status}{detail}"
                )

    counts = {
        s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "XFAIL", "XPASS", "SKIP")
    }
    print(" ".join(f"{k}={v}" for k, v in counts.items()))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2))
    return 1 if counts["FAIL"] or counts["XPASS"] else 0


if __name__ == "__main__":
    sys.exit(main())
