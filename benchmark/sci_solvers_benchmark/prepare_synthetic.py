"""Build the synthetic benchmark inputs: product subspaces of exactly ``n_alpha**2``.

For each molecule and each ``n_alpha`` on the ladder, the alpha strings are the
Hartree-Fock string plus its single, double, ... excitations, up to the smallest rank
whose pool holds ``n_alpha`` strings; the top rank is subsampled when the pool is larger.
Beta uses the same strings (closed shell), so a tensor-product-basis solver diagonalizes
exactly ``n_alpha**2`` determinants. Output layout, read by ``run_matrix``::

    <out>/molecules/<name>/<name>.fcidump
    <out>/molecules/<name>/hf_expansion/001/hf_product_<n_alpha>.npz
    <out>/manifest.csv

Usage::

    sci-bench-prepare-synthetic --dataset-dir <extracted dataset> --out-dir <base_dir> \\
        --ids 170 171 176 375 376 670 671 672 674 675

``--dataset-dir`` holds the extracted ``fcidump/`` tree; molecule ``<ID>`` reads
``fcidump/<ID>.fcidump`` and is written as ``mol_<ID>``. A molecule that is not in the
dataset is added with ``--fcidump NAME=PATH``.

Notes
-----
For ``norb`` spatial orbitals and ``nocc`` electrons per spin, the number of alpha
strings within ``r`` excitations of HF is

.. math:: P(r) = \\sum_{k=0}^{r} \\binom{n_\\mathrm{occ}}{k} \\binom{n_\\mathrm{orb}-n_\\mathrm{occ}}{k},

which saturates at :math:`\\binom{n_\\mathrm{orb}}{n_\\mathrm{occ}}` (Vandermonde), so
the largest product subspace is the full-CI dimension
:math:`\\binom{n_\\mathrm{orb}}{n_\\mathrm{occ}}^2`. The excitation level used for
``n_alpha`` is the smallest ``r`` with :math:`P(r) \\ge n_\\mathrm{alpha}`.

Strings are generated rank by rank. A rank-``k`` string empties ``k`` of the ``nocc``
occupied orbitals and fills ``k`` of the virtual ones; the ``(occupied, virtual)``
combination pairs are indexed in row-major order of the two
``itertools.combinations`` lists. A rank with at most ``n_alpha`` strings is taken whole,
in index order; a larger rank is sampled, ``n_alpha`` indices without replacement from a
fresh ``numpy.random.default_rng(seed)``. Strings are appended rank by rank, in that
order, until ``n_alpha`` are collected, so only the last rank used is cut short.

Each file stores ``n_alpha`` rows, not the ``n_alpha**2`` product: row ``i`` pairs alpha
string ``i`` with beta string ``i``. A TPB solver deduplicates each half and rebuilds the
product, so the rows define the full subspace. **Do not feed these files to a GDB
solver**, which would read each row as one determinant.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from sci_solvers_benchmark.fcidump import read_fcidump_header

DEFAULT_LADDER = (100, 200, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 15000, 18000)


def alpha_pool_size(norb: int, nocc: int, excitation: int) -> int:
    """Number of alpha strings within ``excitation`` excitations of HF, :math:`P(r)`."""
    nvirt = norb - nocc
    return sum(comb(nocc, k) * comb(nvirt, k) for k in range(excitation + 1))


@dataclass(frozen=True)
class PoolPlan:
    """The excitation level chosen for one ``n_alpha``, and what it implies.

    Attributes:
        excitation: Smallest rank whose pool holds ``n_alpha`` strings.
        n_alpha: Alpha (and beta) strings kept.
        subspace_dim: ``n_alpha**2``.
        alpha_pool_available: :math:`P(\\text{excitation})`.
        complete_through_rank: Highest rank kept whole: ``excitation`` if the pool is
            used in full, else ``excitation - 1``.
        max_total_rank: ``2 * excitation``, the highest total rank of a product
            determinant.
        fraction_of_full_ci: ``subspace_dim`` over the full-CI dimension.
    """

    excitation: int
    n_alpha: int
    subspace_dim: int
    alpha_pool_available: int
    complete_through_rank: int
    max_total_rank: int
    fraction_of_full_ci: float


def plan_pool(norb: int, nocc: int, n_alpha: int) -> PoolPlan:
    """Pick the smallest excitation level that supplies ``n_alpha`` alpha strings.

    Raises:
        ValueError: ``n_alpha < 1``, or more strings than :math:`\\binom{n_\\mathrm{orb}}
            {n_\\mathrm{occ}}` exist.
    """
    full = comb(norb, nocc)
    if not 1 <= n_alpha <= full:
        raise ValueError(f"n_alpha={n_alpha} must be in [1, {full}] for norb={norb}, nocc={nocc}")
    excitation = next(r for r in range(nocc + 1) if alpha_pool_size(norb, nocc, r) >= n_alpha)
    available = alpha_pool_size(norb, nocc, excitation)
    return PoolPlan(
        excitation=excitation,
        n_alpha=n_alpha,
        subspace_dim=n_alpha * n_alpha,
        alpha_pool_available=available,
        complete_through_rank=excitation if n_alpha == available else max(0, excitation - 1),
        max_total_rank=2 * excitation,
        fraction_of_full_ci=(n_alpha / full) ** 2,
    )


def _rank_strings(norb: int, nocc: int, k: int, cap: int, seed: int) -> list[frozenset]:
    """Occupied-orbital sets of the rank-``k`` strings: all of them, or ``cap`` sampled."""
    holes = list(combinations(range(nocc), k))
    particles = list(combinations(range(nocc, norb), k))
    size = len(holes) * len(particles)
    if cap >= size:
        linear = np.arange(size)
    else:
        linear = np.random.default_rng(seed).choice(size, size=cap, replace=False)
    i_hole, i_particle = np.unravel_index(linear, (len(holes), len(particles)))
    hf = frozenset(range(nocc))
    return [(hf - set(holes[a])) | set(particles[b]) for a, b in zip(i_hole, i_particle)]


def alpha_strings(norb: int, nocc: int, plan: PoolPlan, seed: int = 42) -> np.ndarray:
    """Return the plan's ``n_alpha`` alpha strings as a ``(n_alpha, norb)`` 0/1 array.

    Row 0 is HF. Orbital ``j`` is array column ``norb - 1 - j``, so HF occupies the
    ``nocc`` highest columns (the bitstring order ``bitstring_matrix_to_ci_strs`` reads).
    """
    rows = [frozenset(range(nocc))]
    for k in range(1, plan.excitation + 1):
        if len(rows) >= plan.n_alpha:
            break
        rows.extend(_rank_strings(norb, nocc, k, plan.n_alpha, seed))
    out = np.zeros((plan.n_alpha, norb), dtype=np.int8)
    for i, occupied in enumerate(rows[: plan.n_alpha]):
        out[i, [norb - 1 - j for j in occupied]] = 1
    return out


def write_molecule(name: str, fcidump: Path, out_dir: Path, ladder: list[int], seed: int) -> list:
    """Write every ladder size one molecule supports; return its manifest rows.

    Sizes beyond the molecule's full alpha space end its ladder and are skipped.

    Raises:
        ValueError: The FCIDUMP is open shell (``n_alpha != n_beta``).
    """
    h = read_fcidump_header(fcidump)
    norb, nocc = h["norb"], h["n_alpha"]
    if h["n_alpha"] != h["n_beta"]:
        raise ValueError(f"{name}: closed shell only, got MS2={h['ms2']}")
    mol_dir = out_dir / "molecules" / name
    data_dir = mol_dir / "hf_expansion" / "001"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(fcidump, mol_dir / f"{name}.fcidump")

    rows = []
    for n_alpha in ladder:
        if n_alpha > comb(norb, nocc):
            print(f"  {name}: stop at n_alpha={n_alpha} (only {comb(norb, nocc)} alpha strings)")
            break
        plan = plan_pool(norb, nocc, n_alpha)
        alpha = alpha_strings(norb, nocc, plan, seed)
        npz = data_dir / f"hf_product_{n_alpha}.npz"
        np.savez_compressed(
            npz,
            bitstrings=np.concatenate([alpha, alpha], axis=1),  # beta half, then alpha half
            probs=np.full(n_alpha, 1.0 / n_alpha),
            norb=norb,
            num_elec_a=nocc,
            num_elec_b=nocc,
            seed=seed,
            **asdict(plan),
        )
        rows.append({
            "molecule": name,
            "norb": norb,
            "nocc": nocc,
            **asdict(plan),
            "seed": seed,
            "npz": str(npz.relative_to(out_dir)),
        })
        print(f"  {name}: n_alpha={n_alpha} D={plan.subspace_dim:,} exc={plan.excitation}")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset-dir", type=Path, help="extracted dataset (holds fcidump/)")
    p.add_argument("--ids", nargs="+", default=[], help="three-digit molecule IDs")
    p.add_argument(
        "--fcidump",
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help="other molecules by FCIDUMP, e.g. my_mol=path/to/my_mol.fcidump",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--ladder",
        type=lambda s: [int(x) for x in s.split(",")],
        default=list(DEFAULT_LADDER),
        help="comma-separated n_alpha values",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.ids and args.dataset_dir is None:
        p.error("--ids needs --dataset-dir")
    if not args.ids and not args.fcidump:
        p.error("give --ids and/or --fcidump")

    molecules = [
        (f"mol_{int(i):03d}", args.dataset_dir / "fcidump" / f"{int(i):03d}.fcidump")
        for i in args.ids
    ]
    for item in args.fcidump:
        name, _, path = item.partition("=")
        molecules.append((name, Path(path)))

    manifest = []
    for name, fcidump in molecules:
        manifest += write_molecule(name, fcidump, args.out_dir, args.ladder, args.seed)
    with (args.out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"wrote {len(manifest)} files and {args.out_dir / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
