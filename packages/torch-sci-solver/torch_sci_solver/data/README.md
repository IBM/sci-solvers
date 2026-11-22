# Bundled test data

All files are from the **SQD molecules dataset**: J. Schuhmacher, S. Barison, D. Grabocka,
A. Baiardi, F. Zipoli, P. Ruch, T. Laino and M. Manica, "A Molecular Hamiltonian and
Quantum-Hardware Shots Dataset for Sample-based Quantum Diagonalization",
[doi:10.5281/zenodo.23286514](https://doi.org/10.5281/zenodo.23286514), licensed under the
[Community Data License Agreement – Permissive 2.0](https://cdla.dev/permissive-2-0/).

| File | Dataset source | Content |
|---|---|---|
| `methane_001.fcidump` | `fcidump/001.fcidump` | methane, (10e, 10o), unmodified |
| `methane_001_sqdrift_shots_10k.npz` | `shots/sqdrift/001/` | methane SqDRIFT shots (see below) |
| `benzoic_acid_182.fcidump` | `fcidump/182.fcidump` | benzoic acid, (36e, 30o), unmodified |
| `benzoic_acid_182_sqdrift_shots_10k.npz` | `shots/sqdrift/182/` | benzoic acid SqDRIFT shots (see below) |

The test suite also stores one matrix derived from the dataset, outside this folder:
`packages/torch-sci-solver/tests/fixtures/tm_complex8_758_block_n64.npz`, a 64×64 block of
the Hamiltonian of "Transition metal complex 8" (dataset ID 758, Fe(CN)6, 54 electrons in 50
orbitals), built from `fcidump/758.fcidump` on the 8 CI strings stored with it. Its FCIDUMP
(56 MB) is not bundled.

`packages/torch-sci-solver/tests/fixtures/n2_cas10_sqd_batch.npz` is not from the dataset:
the active-space integrals (`h1`, and `eri8` in 8-fold symmetry) of N2 at 1.4 Å in 6-31G,
from PySCF's `mcscf.CASCI(scf.RHF(mol), 10, 10)`. They are stored rather than rebuilt
because N2's degenerate orbitals are rotated differently by different BLAS builds; a rebuild
gives the same CAS energy but not the same subspace energies on the test's 40 strings.

## How the shots files were made

For each molecule, all 1,500 SqDRIFT shot files (IBM Quantum `ibm_aachen`, 1.5M shots in
total) were merged with the dataset README's own loader: each file's qubit permutation is
applied to bring its columns into orbital order, and identical bitstrings are summed. From the
merged distinct bitstrings, a random sample of 10,000 was kept
(`numpy.random.default_rng(0).choice(..., size=10_000, replace=False)`), each with its measured
count unchanged. The sample is not filtered by electron number: like the raw hardware data,
most bitstrings have the wrong particle count and are repaired by configuration recovery.

| Molecule | Distinct bitstrings | Kept | With the correct electron count | Shots kept |
|---|---|---|---|---|
| methane (001) | 673,839 | 10,000 | 596 | 22,419 |
| benzoic acid (182) | 1,499,999 | 10,000 | 70 | 10,000 |

Each file is a `.npz` with `packed` (one row per measured shot, bits packed little-endian)
and `L` (bits per shot), the format `torch_sci_solver.shots.load_shots` reads.
