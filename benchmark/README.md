# sci-solvers benchmark

Scripts to reproduce the cross-solver benchmark of the `sci_solvers` eigensolvers: each
solver diagonalizes the same fixed subspaces, and its wall time, peak host RAM, peak GPU
memory, energy and ⟨S²⟩ are recorded. There is no SQD loop and no configuration recovery;
a cell is one solver call on one subspace, repeated three times after a warm-up.

There are two input datasets:

- **Quantum-sampled**: SqDRIFT hardware shots from the published molecules dataset,
  filtered to the right electron counts. A size on the ladder is the number of
  bitstrings drawn (weighted by count) from that pool.
- **Synthetic**: product subspaces of exactly `n_alpha**2` determinants, built from the
  Hartree-Fock string and its excitations. A size on the ladder is `n_alpha`.

## Install

From the repository root:

```console
uv sync --extra benchmark
```

The solvers come with `sci_solvers` itself; only the optional ones need their own extra,
added to the same `uv sync`: `--extra sbd-gpu` for sbd on the GPU (it also installs the
CUDA 12 PyTorch that sbd needs), `--extra sbd` for sbd on the CPU, and `--extra primme`
for PRIMME. Dice needs its external binary
([`scripts/energy_solvers/README.md`](../scripts/energy_solvers/README.md)).

`uv run` re-syncs the environment before each command. That keeps the benchmark
installed, but swaps a PyTorch picked by an extra (`sbd-gpu`, `cpu`) back to the
default build, so either repeat those extras (`uv run --extra sbd-gpu ...`) or pass
`--no-sync`.

## 1. Prepare the inputs

The inputs are built from the published molecules dataset
([Zenodo record 23286514](https://zenodo.org/records/23286514)), which includes the 10
organic molecules of the benchmark (IDs below). Download it, reassemble and extract
`fcidump.tar.bz2`, `sqdrift.tar.bz2` and `sqdrift_permutations.tar.bz2` as its README
describes, then:

```console
uv run sci-bench-prepare-quantum \
    --dataset-dir <dataset> --out-dir <quantum_base> \
    --ids 170 171 176 375 376 670 671 672 674 675

uv run sci-bench-prepare-synthetic \
    --dataset-dir <dataset> --out-dir <synthetic_base> \
    --ids 170 171 176 375 376 670 671 672 674 675
```

`prepare_quantum` merges a molecule's 1,500 circuits after reordering each circuit's
qubits by its recorded permutation (the dataset README's procedure), and keeps the
bitstrings with the FCIDUMP's alpha and beta electron counts. `prepare_synthetic` writes
one `hf_product_<n_alpha>.npz` per ladder size, plus `manifest.csv` with the excitation
level and full-CI fraction of each file. Its output depends only on the FCIDUMP header
and the seed (default 42).

Other molecules can be benchmarked the same way. Place each FCIDUMP and shots file in
this layout (a bitstring is the beta occupations followed by the alpha occupations, the
order `qiskit_addon_sqd` reads):

```
<base_dir>/molecules/<name>/<name>.fcidump
<base_dir>/molecules/<name>/<family>/1/<name>_pn_filtered.json   # {"counts": {bitstring: count}}
```

and build their synthetic inputs with `--fcidump <name>=<path>`, alongside or instead of
`--ids`.

## 2. Run

Every configuration is one file in [`configs/solvers/`](configs/solvers), merged over
[`configs/_common.yaml`](configs/_common.yaml) (3 timed runs after a warm-up, HF always
included, 6 h cap per solver call). A configuration's Davidson-type settings are matched
across solvers: residual tolerance 1e-4, subspace 10, at most 80 matrix-vector products.

[`configs/kernels/`](configs/kernels) holds the configurations of the kernel and
Hamiltonian-backend comparison: arpack, lobpcg and primme with the same solver settings but
a different matrix-vector product: the batched kernel on the GPU (`*_gpu_batched`, against
the Triton kernel in `configs/solvers/`), the batched kernel on CPU (`*_cpu`), and fulqrum's
CSR, fast CSR and matrix-free Hamiltonians on CPU (`*_fulqrum_*_cpu`, needs
`--extra fulqrum`). They run the same way.

One (molecule, configuration) pair up its size ladder, stopping at the first size that
fails (time limit, out of memory, or error):

```console
uv run benchmark/scripts/run_ladder.sh \
    mol_170 benchmark/configs/solvers/torch_sci_gpu.yaml \
    --base-dir <quantum_base> --output-dir runs/quantum/mol_170__torch_sci_gpu
```

The sizes are bitstrings drawn from the quantum-sampled pool, or `n_alpha` when
`--base-dir` holds synthetic inputs (the script picks the right default ladder; override
it with `--sizes`). The script runs everything in the current shell and does not submit
jobs, so run each pair inside its own allocation (e.g. a batch job) with the CPU, memory
and GPU limits you want measured. The benchmark used one A100 80 GB, 8 CPU
cores and 100 GB of RAM for the GPU configurations, and 16 cores and 200 GB for the CPU
configurations (`*_cpu`).

A single cell, without the ladder:

```console
uv run python -m sci_solvers_benchmark.run_matrix \
    --config_file benchmark/configs/solvers/pyscf_cpu.yaml --base_dir <quantum_base> \
    --molecules '["mol_670"]' --subsample_size 1000 --output_dir runs/one_cell
```

## 3. Summarize

```console
uv run python -m sci_solvers_benchmark.summarize --output_dir runs/quantum
```

This reads every `per_run/**/*.json` below `--output_dir` and writes
`summary/across_molecules.csv` (one row per cell: subspace size, time, energies, ⟨S²⟩,
peak memory, errors), `summary/summary.md` and `summary/per_molecule/<molecule>.md`.
Each per-run JSON also records the SHA-256 of its FCIDUMP and shots file, the full
configuration, and the CPU, RAM and GPU of the machine.

## Layout

This directory, `benchmark/`, holds the Python package `sci_solvers_benchmark` (its
import name; the distribution is `sci-solvers-benchmark`), the configurations and the
ladder script:

```
benchmark/
  pyproject.toml
  sci_solvers_benchmark/
    prepare_quantum.py     # published dataset -> quantum-sampled inputs
    prepare_synthetic.py   # FCIDUMP -> synthetic n_alpha**2 inputs
    run_single.py          # time one solver on one subspace
    run_matrix.py          # all (molecule × input × configuration) cells of a config
    summarize.py           # per-run JSONs -> CSV + markdown
    memory.py, shots.py, spin_square.py, fcidump.py
  configs/solvers/         # the main solver configurations
  configs/kernels/         # kernel and Hamiltonian-backend comparison
  scripts/run_ladder.sh    # size ladder for one (molecule, configuration)
  tests/
```
