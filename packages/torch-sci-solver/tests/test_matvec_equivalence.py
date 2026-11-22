"""Integration equivalence: matvec_impl="triton" vs "batched" (GPU) vs
"batched" (CPU) must all agree, through the high-level ``TorchSCISolver`` API.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import time
from itertools import combinations, islice

import numpy as np
import pytest
import torch
from torch_sci_solver import _paths

from sci_solvers.energy_solvers.torch_sci_solver import TorchSCISolver
from sci_solvers.hamiltonian import HamiltonianData

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="equivalence needs a CUDA device to exercise the Triton kernel",
)

_SOLVE_KWARGS = dict(max_davidson_iters=200, davidson_block_size=30, davidson_tolerance=1e-7)


def _strings(norb: int, nocc: int, limit: int | None = None) -> np.ndarray:
    combos = combinations(range(norb), nocc)
    if limit is not None:
        combos = islice(combos, limit)
    return np.array([sum(1 << i for i in c) for c in combos], dtype=np.int64)


def _random_system(seed: int, norb: int, na: int, nb: int):
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    x = rng.standard_normal((norb,) * 4)
    eri = np.zeros_like(x)
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
        eri += x.transpose(perm)
    eri /= 8.0
    return hcore, eri, _strings(norb, na), _strings(norb, nb)


def _energy(hcore, eri, norb, nelec, adets, bdets, *, symmetrize_spin, matvec_impl, device):
    solver = TorchSCISolver(
        symmetrize_spin=symmetrize_spin,
        device=device,
        matvec_impl=matvec_impl,
        **_SOLVE_KWARGS,
    )
    states = solver.sci_solver_callable_eigen_states([(adets, bdets)], hcore, eri, norb, nelec)[
        0
    ].states
    energy = states[0].energy
    assert energy != -1.0, f"matvec_impl={matvec_impl!r} device={device!r} did not converge"
    return energy


def _assert_impls_agree(
    hcore, eri, norb, nelec, adets, bdets, *, symmetrize_spin, label, atol=1e-6
):
    kwargs = dict(symmetrize_spin=symmetrize_spin)
    e_triton = _energy(
        hcore,
        eri,
        norb,
        nelec,
        adets,
        bdets,
        matvec_impl="triton",
        device="cuda",
        **kwargs,
    )
    e_batched_gpu = _energy(
        hcore,
        eri,
        norb,
        nelec,
        adets,
        bdets,
        matvec_impl="batched",
        device="cuda",
        **kwargs,
    )
    e_batched_cpu = _energy(
        hcore,
        eri,
        norb,
        nelec,
        adets,
        bdets,
        matvec_impl="batched",
        device="cpu",
        **kwargs,
    )
    print(
        f"[{label}] triton={e_triton:.10f} batched_gpu={e_batched_gpu:.10f} batched_cpu={e_batched_cpu:.10f}"
    )
    np.testing.assert_allclose(
        e_triton,
        e_batched_cpu,
        atol=atol,
        rtol=0,
        err_msg=f"{label}: triton vs batched(cpu) mismatch",
    )
    np.testing.assert_allclose(
        e_batched_gpu,
        e_batched_cpu,
        atol=atol,
        rtol=0,
        err_msg=f"{label}: batched(gpu) vs batched(cpu) mismatch",
    )


@pytest.mark.parametrize("norb,na,nb", [(5, 3, 3), (7, 2, 2)], ids=["dim100", "dim441"])
def test_matvec_equivalence_random(norb, na, nb):
    """Small random Hamiltonians with several excitations per determinant --
    exactly the shape (``jj >= 1`` in the alphabeta kernel's inner loop) that
    hid the index-shadowing bug, without needing production-scale data."""
    hcore, eri, adets, bdets = _random_system(11, norb, na, nb)
    _assert_impls_agree(
        hcore,
        eri,
        norb,
        (na, nb),
        adets,
        bdets,
        symmetrize_spin=False,
        label=f"random[{norb},{na},{nb}]",
    )


@pytest.mark.parametrize("k", [30, 50], ids=["K30", "K50"])
def test_matvec_equivalence_benzoic_acid_small(k):
    """Real benzoic acid (SQD molecules dataset ID 182) Hamiltonian, HF + singles
    K=30/50 subsets -- the same bases used by the diag_torch anchors in
    test_torch_anchors.py, exercised here through the high-level TorchSCISolver API."""
    ham = HamiltonianData.from_fcidump(_paths.fcidump_path("benzoic_acid"))
    adets = np.array(_paths.hf_and_singles(ham.num_orbitals, ham.num_elec_a)[:k], dtype=np.int64)
    bdets = adets.copy()
    _assert_impls_agree(
        ham.hcore,
        ham.eri,
        ham.num_orbitals,
        (ham.num_elec_a, ham.num_elec_b),
        adets,
        bdets,
        symmetrize_spin=True,
        label=f"benzoic_acid_K{k}",
    )


@pytest.mark.slow
@pytest.mark.parametrize("num_dets", [10, 20, 100, 500])
def test_matvec_equivalence_benzoic_acid_with_runtime_report(num_dets):
    """Real benzoic acid Hamiltonian across different subspace sizes."""
    ham = HamiltonianData.from_fcidump(_paths.fcidump_path("benzoic_acid"))
    norb, na, nb = ham.num_orbitals, ham.num_elec_a, ham.num_elec_b

    adets = _strings(norb, na, num_dets)
    bdets = _strings(norb, nb, num_dets)
    assert len(adets) * len(bdets) == num_dets**2

    implementations = [
        ("cuda", "triton"),
        ("cuda", "batched"),
    ]
    if num_dets <= 20:
        implementations.append(("cpu", "batched"))

    report = {}

    print(f"\nSubspace dimension: {num_dets**2:,}")
    print(f"{'impl':<15}{'energy':>18}{'wall_s':>12}")

    for dev, impl in implementations:
        t0 = time.perf_counter()
        energy = _energy(
            ham.hcore,
            ham.eri,
            norb,
            (na, nb),
            adets,
            bdets,
            symmetrize_spin=True,
            matvec_impl=impl,
            device=dev,
        )
        dt = time.perf_counter() - t0

        key = f"{impl}_{dev}"
        report[key] = (energy, dt)

        print(f"{key:<15}{energy:>18.10f}{dt:>12.2f}")

    np.testing.assert_allclose(
        report["triton_cuda"][0],
        report["batched_cuda"][0],
        atol=1e-6,
        rtol=0,
    )

    if num_dets <= 20:
        np.testing.assert_allclose(
            report["batched_cpu"][0],
            report["batched_cuda"][0],
            atol=1e-6,
            rtol=0,
        )
