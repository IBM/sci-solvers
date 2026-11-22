"""Tests for the benchmark's input preparation and its run → summarize pipeline."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pytest
import yaml
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

pytest.importorskip(
    "sci_solvers_benchmark",
    reason="benchmark package not installed (`uv sync --extra benchmark`)",
)

from sci_solvers_benchmark import prepare_quantum, prepare_synthetic  # noqa: E402
from sci_solvers_benchmark.run_matrix import run_matrix  # noqa: E402
from sci_solvers_benchmark.run_single import hf_bitstring  # noqa: E402
from sci_solvers_benchmark.summarize import summarize  # noqa: E402


def _write_shots(path, rows, counts):
    np.savez(path, bitstrings=np.array(rows), count=np.array(counts))


def test_sqdrift_permutation_follows_dataset_readme(tmp_path):
    """A circuit's columns are reordered by (n-1) - perm[::-1]; no metadata means identity."""
    shots, perms = tmp_path / "shots", tmp_path / "perms"
    shots.mkdir()
    perms.mkdir()
    _write_shots(shots / "a_shots.npz", ["1100", "0011"], [2, 1])
    (perms / "a_metadata.txt").write_text("[1, 0, 3, 2]")
    _write_shots(shots / "b_shots.npz", ["1001"], [4])

    counts = prepare_quantum.load_sqdrift_counts(shots, perms)

    # perm [1,0,3,2] -> little-endian columns (3 - [2,3,0,1]) = [1,0,3,2]
    assert counts == {"1100": 2, "0011": 1, "1001": 4}
    (perms / "a_metadata.txt").write_text("[0, 2, 1, 3]")  # columns [0,2,1,3]
    assert prepare_quantum.load_sqdrift_counts(shots, perms) == {
        "1010": 2,
        "0101": 1,
        "1001": 4,
    }


def test_particle_number_filter():
    """Beta electrons are counted in the first norb characters, alpha in the rest."""
    counts = {"0011": 1, "0110": 2, "1100": 3, "0101": 4}
    assert prepare_quantum.filter_particle_number(counts, 2, 1, 1) == {"0110": 2, "0101": 4}


@pytest.mark.parametrize(("norb", "nocc"), [(6, 3), (8, 4), (9, 3)])
def test_pool_sizes_match_enumeration(norb, nocc):
    """P(r) counts exactly the strings within r excitations, and saturates at C(norb, nocc)."""
    for r in range(nocc + 1):
        expected = sum(
            1 for occ in combinations(range(norb), nocc) if len(set(occ) - set(range(nocc))) <= r
        )
        assert prepare_synthetic.alpha_pool_size(norb, nocc, r) == expected
    assert prepare_synthetic.alpha_pool_size(norb, nocc, nocc) == comb(norb, nocc)


@pytest.mark.parametrize("n_alpha", [1, 7, 15, 40, 70])
def test_synthetic_subspace_is_exactly_n_alpha_squared(n_alpha):
    """HF first, unique strings, right electron count, rank bound, and D = n_alpha**2."""
    norb, nocc = 8, 4
    plan = prepare_synthetic.plan_pool(norb, nocc, n_alpha)
    alpha = prepare_synthetic.alpha_strings(norb, nocc, plan, seed=42)

    assert alpha.shape == (n_alpha, norb)
    assert list(alpha[0]) == [0] * (norb - nocc) + [1] * nocc
    assert len({row.tobytes() for row in alpha}) == n_alpha
    assert (alpha.sum(axis=1) == nocc).all()
    ranks = (alpha[:, : norb - nocc]).sum(axis=1)  # electrons in HF-virtual orbitals
    assert ranks.max() <= plan.excitation
    bits = np.concatenate([alpha, alpha], axis=1).astype(bool)
    strs_a, strs_b = bitstring_matrix_to_ci_strs(bits, open_shell=True)
    assert len(strs_a) * len(strs_b) == plan.subspace_dim == n_alpha**2
    again = prepare_synthetic.alpha_strings(norb, nocc, plan, seed=42)
    assert np.array_equal(alpha, again)


def test_plan_rejects_more_strings_than_exist():
    """Asking for more alpha strings than C(norb, nocc) is an error."""
    with pytest.raises(ValueError, match="must be in"):
        prepare_synthetic.plan_pool(6, 3, comb(6, 3) + 1)


def test_hf_bitstring_layout():
    """HF is the beta half then the alpha half, occupied orbitals rightmost."""
    assert hf_bitstring(4, 3, 1) == "0001" + "0111"


@pytest.fixture
def h4_base_dir(tmp_path):
    """A linear H4 chain (4 orbitals, 2+2 electrons) and every valid bitstring as input."""
    from pyscf import gto, scf
    from pyscf.tools import fcidump

    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mol_dir = tmp_path / "molecules" / "h4"
    (mol_dir / "sqdrift" / "1").mkdir(parents=True)
    fcidump.from_scf(mf, str(mol_dir / "h4.fcidump"))
    pairs = [
        (b, a)
        for b in range(16)
        for a in range(16)
        if bin(a).count("1") == 2 and bin(b).count("1") == 2
    ]
    counts = {f"{b:04b}{a:04b}": 1 + (i % 3) for i, (b, a) in enumerate(pairs)}
    (mol_dir / "sqdrift" / "1" / "h4_pn_filtered.json").write_text(json.dumps({"counts": counts}))
    return tmp_path


def test_run_matrix_then_summarize(h4_base_dir, tmp_path):
    """One CPU cell end to end: the full-CI subspace reproduces the H4 FCI energy."""
    from pyscf import fci
    from pyscf.tools import fcidump

    base_dir = h4_base_dir
    cfg = {
        "molecules": "auto",
        "circuit_types": "auto",
        "common": {"num_runs": 2, "warmup": False, "measure_memory": True},
        "solver_variants": [
            {
                "id": "pyscf_test",
                "name": "pyscf",
                "init_kwargs": {"symmetrize_spin": True},
                "call_kwargs": {},
            }
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"

    run_matrix(str(cfg_path), str(out), base_dir=str(base_dir))

    [path] = list((out / "per_run").rglob("*.json"))
    payload = json.loads(path.read_text())
    assert payload["status"] == "ok", payload["error"]
    results = payload["results"]
    assert results["subspace_dim"] == 36
    assert len(results["solver_time_values_s"]) == 2
    data = fcidump.read(str(base_dir / "molecules" / "h4" / "h4.fcidump"))
    e_fci, _ = fci.direct_spin1.FCI().kernel(data["H1"], data["H2"], 4, (2, 2))
    assert results["total_energy_mean"] == pytest.approx(e_fci + data["ECORE"], abs=1e-8)
    assert results["spin_squared_mean"] == pytest.approx(0.0, abs=1e-6)

    summarize(str(out))
    assert (out / "summary" / "across_molecules.csv").read_text().count("\n") == 2


class _FakeSolver:
    """Stands in for a real solver: records each call and returns a fixed state."""

    calls: list[tuple] = []

    def __init__(self, _hamiltonian, **kwargs):
        """Keep the constructor arguments for the test to inspect."""
        self.kwargs = kwargs

    def sci_solver_callable_eigen_states(self, batches, _hcore, _eri, norb, nelec, **kwargs):
        """Return one root of energy -1 with amplitudes on the first determinant."""
        from qiskit_addon_sqd.fermion import SCIResult, SCIState

        from sci_solvers.eigen_states import EigenStates

        [(strs_a, strs_b)] = batches
        _FakeSolver.calls.append((len(strs_a), len(strs_b), kwargs))
        amplitudes = np.zeros((len(strs_a), len(strs_b)))
        amplitudes[0, 0] = 1.0
        state = SCIState(amplitudes, strs_a, strs_b, norb, nelec)
        return [EigenStates(states=[SCIResult(-1.0, state, (None, None))])]


def _write_dataset(root, mol_id, norb, n_alpha, n_beta):
    """A miniature published dataset: one FCIDUMP and two SqDRIFT circuits for one ID."""
    from pyscf import ao2mo, gto, scf
    from pyscf.tools import fcidump

    (root / "fcidump").mkdir(parents=True)
    mol = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    fcidump.from_integrals(
        str(root / "fcidump" / f"{mol_id}.fcidump"),
        h1,
        eri,
        norb,
        n_alpha + n_beta,
        mol.energy_nuc(),
        ms=n_alpha - n_beta,
    )
    shots = root / "shots" / "sqdrift" / mol_id
    perms = root / "shots" / "sqdrift_permutations" / mol_id
    shots.mkdir(parents=True)
    perms.mkdir(parents=True)
    # Stored little-endian; circuit "a" swaps qubits 0<->1 and 6<->7, so in orbital order
    # its strings are 00110011 (HF), 00110101 and 11000000 (wrong electron count).
    _write_shots(shots / "c_a_shots.npz", ["00110011", "00110110", "11000000"], [5, 2, 1])
    (perms / "c_a_metadata.txt").write_text("[1, 0, 2, 3, 4, 5, 7, 6]")
    _write_shots(shots / "c_b_shots.npz", ["01010101", "00110011"], [3, 4])  # identity order


def test_prepare_run_summarize_with_mocked_solver(tmp_path, monkeypatch):
    """Dataset -> both prepare steps -> run_matrix -> summarize, with the solver mocked."""
    import sci_solvers_benchmark.run_single as run_single_module

    dataset = tmp_path / "dataset"
    _write_dataset(dataset, "042", norb=4, n_alpha=2, n_beta=2)
    base = tmp_path / "base"
    prepare_quantum.main(["--dataset-dir", str(dataset), "--out-dir", str(base), "--ids", "42"])
    prepare_synthetic.main([
        "--dataset-dir",
        str(dataset),
        "--out-dir",
        str(base),
        "--ids",
        "42",
        "--ladder",
        "2,3",
    ])

    pool = json.loads((base / "molecules/mol_042/sqdrift/1/mol_042_pn_filtered.json").read_text())
    assert pool["counts"] == {"00110011": 9, "00110101": 2, "01010101": 3}

    monkeypatch.setattr(
        run_single_module,
        "create_solver",
        lambda name, hamiltonian, **kw: _FakeSolver(hamiltonian, **kw),
    )
    _FakeSolver.calls = []
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump({
            "common_file": str(Path(__file__).parents[1] / "configs" / "_common.yaml"),
            "solver_variants": [
                {
                    "id": "fake",
                    "name": "fake",
                    "init_kwargs": {"device": "cpu"},
                    "call_kwargs": {"spin_sq": None},
                }
            ],
        })
    )
    out = tmp_path / "out"
    run_matrix(str(cfg), str(out), base_dir=str(base), num_runs=2, warmup=False)

    results = {p.stem: json.loads(p.read_text()) for p in (out / "per_run").rglob("*.json")}
    assert set(results) == {"mol_042_pn_filtered__fake", "hf_product_2__fake", "hf_product_3__fake"}
    for payload in results.values():
        assert payload["status"] == "ok", payload["error"]
        assert payload["variant"]["init_kwargs"] == {"device": "cpu"}
        assert payload["results"]["electronic_energy_mean"] == -1.0
        assert len(payload["results"]["solver_time_values_s"]) == 2
    # symmetrize_spin pools both halves: alpha {0011, 0101}, beta {0011, 0101} -> 2 x 2.
    assert results["mol_042_pn_filtered__fake"]["results"]["subspace_dim"] == 4
    assert results["hf_product_3__fake"]["results"]["subspace_dim"] == 9
    assert all(kw == {"spin_sq": None} for *_, kw in _FakeSolver.calls)

    summarize(str(out))
    assert (out / "summary" / "across_molecules.csv").read_text().count("\n") == 4
