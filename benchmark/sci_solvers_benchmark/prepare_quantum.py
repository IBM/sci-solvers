"""Build the quantum-sampled benchmark inputs from the published molecules dataset.

The dataset (Zenodo record 23286514) ships, per molecule, an FCIDUMP and the raw
hardware shots of 1,500 SqDRIFT circuits (or one LUCJ circuit). This module merges one
molecule's shots into a single ``{bitstring: count}`` sample, keeps only the bitstrings
with the molecule's alpha and beta electron counts, and writes the result in the layout
``run_matrix`` reads::

    <out>/molecules/mol_<ID>/mol_<ID>.fcidump
    <out>/molecules/mol_<ID>/<family>/1/mol_<ID>_pn_filtered.json

Usage::

    sci-bench-prepare-quantum --dataset-dir <extracted dataset> --out-dir <base_dir> \\
        --ids 170 171 176 375 376 670 671 672 674 675

``--dataset-dir`` is the directory holding the extracted ``fcidump/`` and ``shots/``
trees (see the dataset README for reassembling the split archives).

Notes
-----
SqDRIFT circuits may measure the qubits in a randomized order. Each such circuit has a
``shots/sqdrift_permutations/<ID>/<circuit>_metadata.txt`` holding its ``permutation``
(big-endian); a circuit without one keeps the identity order. Following the dataset
README, the stored little-endian bitstring columns are reordered by
``(n - 1) - permutation[::-1]`` before shots from different circuits are merged, so every
bitstring is in orbital order. LUCJ shots are already in orbital order.

A bitstring of length ``2 * norb`` holds the beta occupations in its first ``norb``
characters and the alpha occupations in the last ``norb``, the order
``qiskit_addon_sqd.fermion.bitstring_matrix_to_ci_strs`` reads.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import ast
import json
import shutil
from pathlib import Path

import numpy as np

from sci_solvers_benchmark.fcidump import read_fcidump_header

SHOTS_FAMILIES = ("sqdrift", "lucj")


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["bitstrings"], data["count"].astype(np.int64)


def _permute(bitstrings: np.ndarray, permutation: np.ndarray) -> list[str]:
    """Reorder each bitstring's characters through a big-endian qubit permutation."""
    n = len(bitstrings[0])
    columns = (n - 1) - permutation[::-1]
    codes = bitstrings.view(np.uint32).reshape(len(bitstrings), n)
    permuted = np.ascontiguousarray(codes[:, columns])
    return permuted.view(f"<U{n}").ravel().tolist()


def load_sqdrift_counts(shots_dir: Path, permutations_dir: Path) -> dict[str, int]:
    """Merge every SqDRIFT circuit of one molecule into ``{bitstring: count}``.

    Args:
        shots_dir: ``shots/sqdrift/<ID>``.
        permutations_dir: ``shots/sqdrift_permutations/<ID>``.

    Returns:
        Counts summed over circuits after each circuit's columns are put in orbital
        order. Insertion order follows the sorted file names, then the file's rows.

    Raises:
        FileNotFoundError: ``shots_dir`` holds no ``*_shots.npz``.
    """
    files = sorted(shots_dir.glob("*_shots.npz"))
    if not files:
        raise FileNotFoundError(f"no *_shots.npz under {shots_dir}")
    counts: dict[str, int] = {}
    for path in files:
        bitstrings, shot_counts = _read_npz(path)
        stem = path.name.removesuffix("_shots.npz")
        meta = permutations_dir / f"{stem}_metadata.txt"
        if meta.exists():
            permutation = np.asarray(ast.literal_eval(meta.read_text().strip()))
        else:
            permutation = np.arange(len(bitstrings[0]))
        for s, c in zip(_permute(bitstrings, permutation), shot_counts.tolist()):
            counts[s] = counts.get(s, 0) + c
    return counts


def load_lucj_counts(shots_file: Path) -> dict[str, int]:
    """Read one LUCJ shots file into ``{bitstring: count}`` (already in orbital order)."""
    bitstrings, shot_counts = _read_npz(shots_file)
    counts: dict[str, int] = {}
    for s, c in zip(bitstrings.tolist(), shot_counts.tolist()):
        counts[s] = counts.get(s, 0) + c
    return counts


def filter_particle_number(counts: dict[str, int], norb: int, n_alpha: int, n_beta: int) -> dict:
    """Keep the bitstrings with ``n_beta`` ones in the first ``norb`` bits and ``n_alpha`` after."""
    return {
        s: c
        for s, c in counts.items()
        if s[:norb].count("1") == n_beta and s[norb:].count("1") == n_alpha
    }


def prepare_molecule(dataset_dir: Path, out_dir: Path, mol_id: str, family: str) -> dict:
    """Write one molecule's FCIDUMP and particle-number-filtered shots under ``out_dir``.

    Args:
        dataset_dir: Root of the extracted dataset (holds ``fcidump/`` and ``shots/``).
        out_dir: Benchmark ``base_dir``; files go under ``out_dir/molecules/mol_<ID>/``.
        mol_id: Three-digit molecule ID, e.g. ``"170"``.
        family: ``"sqdrift"`` or ``"lucj"``.

    Returns:
        A summary row: shot and bitstring counts before and after filtering.
    """
    fcidump = dataset_dir / "fcidump" / f"{mol_id}.fcidump"
    header = read_fcidump_header(fcidump)
    if family == "sqdrift":
        counts = load_sqdrift_counts(
            dataset_dir / "shots" / "sqdrift" / mol_id,
            dataset_dir / "shots" / "sqdrift_permutations" / mol_id,
        )
    else:
        counts = load_lucj_counts(dataset_dir / "shots" / "lucj" / f"{mol_id}_shots.npz")
    kept = filter_particle_number(counts, header["norb"], header["n_alpha"], header["n_beta"])
    if not kept:
        raise ValueError(f"mol_{mol_id}: no {family} bitstring has the right electron counts")

    name = f"mol_{mol_id}"
    mol_dir = out_dir / "molecules" / name
    shots_dir = mol_dir / family / "1"
    shots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fcidump, mol_dir / f"{name}.fcidump")
    (shots_dir / f"{name}_pn_filtered.json").write_text(
        json.dumps({"counts": kept}, separators=(",", ":"))
    )
    return {
        "molecule": name,
        "family": family,
        "shots": sum(counts.values()),
        "unique": len(counts),
        "kept_shots": sum(kept.values()),
        "kept_unique": len(kept),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ids", nargs="+", required=True, help="three-digit molecule IDs")
    p.add_argument("--family", choices=SHOTS_FAMILIES, default="sqdrift")
    args = p.parse_args(argv)

    for mol_id in args.ids:
        row = prepare_molecule(args.dataset_dir, args.out_dir, f"{int(mol_id):03d}", args.family)
        print(
            f"{row['molecule']}: {row['shots']:,} shots ({row['unique']:,} unique) -> "
            f"{row['kept_shots']:,} with the right electron counts "
            f"({row['kept_unique']:,} unique)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
