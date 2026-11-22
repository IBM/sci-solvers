"""Hardware-agnostic scaling regression gate for two or more CUDA GPUs.

A synthetic 9·10⁶-determinant TPB problem is solved on one GPU (in a spawned
process) and on two NCCL ranks (1-D owner-computes and a forced tile grid). The
gate uses ratios, so it holds on any CUDA hardware: the best two-rank per-matvec
time must beat one GPU by ``MIN_SPEEDUP``, rank 0's peak allocation must shrink
below ``MAX_PEAK_RATIO`` of the single-GPU peak, converged energies must agree to
``ENERGY_ATOL``, and the TPB matvec must resolve to Triton.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

# Measured on A100-SXM4-80GB: 1.20x (1-D) and 1.24x (tile) on this 9e6-det
# problem. A 10^8-determinant benchmark reaches 1.56x/1.70x at two ranks because
# its fixed per-matvec overhead is a smaller fraction; this gate deliberately
# uses a smaller, faster problem, so its floor is set just below the measured
# values.
MIN_SPEEDUP = 1.15
# At this size the replicated Triton tables and exchange buffers dominate rank
# 0's allocated peak, so the partition only removes its share of the Davidson
# subspace: measured 2.465 GiB at two ranks vs 2.528 GiB single (0.975x). The
# gate still requires a strict shrink; a 10^8-determinant benchmark measures the
# large-N ratio that the memory model predicts.
MAX_PEAK_RATIO = 1.0
ENERGY_ATOL = 1e-8
NORB, NOCC = 15, 5

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def _system():
    """8-fold symmetric random integrals and all 5-of-15 strings (3003² dets)."""
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    raw = rng.standard_normal((NORB,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = sum(raw.transpose(p) for p in perms) / 8.0
    combos = itertools.combinations(range(NORB), NOCC)
    strings = np.array(sorted(sum(1 << o for o in c) for c in combos), dtype=np.int64)
    return h1, eri, strings


def _solve(distributed):
    """Best warm per-matvec seconds, rank-0 peak GiB, converged energy, backend.

    Runs in its own process (one-GPU baseline) or as the parent of NCCL ranks,
    so the test process never holds a CUDA context on a rank's device. The
    timing calls stop after three restarts; the energy comes from a converged
    solve, since truncated single- and multi-rank runs start from different
    seed noise.
    """
    h1, eri, strings = _system()
    ci = [(strings, strings)]
    common = dict(device="cuda", apply_ab_projection=False, distributed=distributed)
    timing = make_torch_sci_solver(
        davidson_tolerance=1e-6, davidson_block_size=8, max_davidson_iters=3, **common
    )
    converged = make_torch_sci_solver(
        davidson_tolerance=1e-9, davidson_block_size=8, max_davidson_iters=200, **common
    )
    try:
        timing(ci, h1, eri, NORB, (NOCC, NOCC))  # warm-up: JIT and setup
        seconds, peak = float("inf"), None
        for _ in range(2):
            timing(ci, h1, eri, NORB, (NOCC, NOCC))
            perf = timing.stats["per_call_log"][-1]["subspaces"][0]
            seconds = min(seconds, float(perf["t_davidson"]) / max(1, int(perf["n_matvecs"])))
            peak = perf.get("peak_gpu_allocated_gib")
        state = converged(ci, h1, eri, NORB, (NOCC, NOCC))[0].states[0]
        return seconds, peak, float(state.energy), timing.matvec_impl
    finally:
        timing.close()
        converged.close()


@pytest.mark.parametrize("tile", [None, "force"], ids=["1d", "tile"])
def test_two_rank_scaling_gate(tile) -> None:
    """Two NCCL ranks beat one GPU, use less memory per rank and keep the energy."""
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    pytest.importorskip("triton")
    from torch_sci_solver.multi_gpu import run_in_spawned_process

    single = run_in_spawned_process(_solve, None)
    config = {"world_size": 2, "backend": "nccl", "matvec": "local"}
    if tile is not None:
        config["tile"] = tile
    double = _solve(config)
    assert single[3] == "triton"
    assert single[0] / double[0] >= MIN_SPEEDUP, (single[0], double[0])
    if single[1] is not None and double[1] is not None:
        assert double[1] < MAX_PEAK_RATIO * single[1], (single[1], double[1])
    assert abs(single[2] - double[2]) <= ENERGY_ATOL
