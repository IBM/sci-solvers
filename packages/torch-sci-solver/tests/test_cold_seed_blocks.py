"""Regression tests for the Davidson seed on a block-diagonal product space.

The TPB and GDB cold starts seeded Davidson on the smallest-α × smallest-β
determinant (TPB basis index 0, GDB ``_smallest_int_pair_index``). A product
space whose strings form clusters more than a double excitation apart is block
diagonal, and Davidson does not leave the block its seed lies in. When the seed
sat in a high-energy block, the solver returned that block's lowest eigenvalue,
above min(diag H). Production symptom: on a 36-orbital [4Fe-4S] space, the α-
and β-like strings of a broken-symmetry reference symmetrized into one set gave
an energy 16 Ha above the reference determinant's diagonal, while PySCF's
selected CI (seeded at the lowest diagonal) found the ground state.

The fix seeds the cold start at argmin(hii), so the Ritz value is <= min(hii),
and repeats a single-device solve once from argmin(hii), with a warning, when
it still ends above min(hii) (a ``connected`` or caller-supplied start).
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
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch
from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
from torch_sci_solver.torch_impl.validate_matvec import _setup

NORB = 8
NELEC = (3, 3)


def _window_strings(orbitals) -> list[int]:
    """All three-electron strings inside ``orbitals``."""
    return [sum(1 << p for p in occ) for occ in itertools.combinations(orbitals, NELEC[0])]


# Two clusters 6 bits apart: no single or double excitation connects them, so the
# product space splits into four blocks (low/high α × low/high β), and the
# smallest-int determinant (basis index 0) lies in the (low, low) block.
LOW = _window_strings(range(0, 4))
HIGH = _window_strings(range(4, 8))
STRS = np.array(sorted(LOW + HIGH), dtype=np.int64)

# Loose enough that the seed's 1e-7 noise can never lift the residual outside
# the seed's block above it (the regression bites on any layout); also the
# default of the sci_solvers wrapper.
TOL = 1e-4


def _integrals() -> tuple[np.ndarray, np.ndarray]:
    """Small random integrals whose high-window orbitals sit 0.1 Ha lower.

    The (high, high) block then holds the ground state, a few tenths of a
    hartree below the (low, low) block.
    """
    rng = np.random.default_rng(7)
    h1 = 0.02 * rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    h1[np.diag_indices(NORB)] += [0.0, 0.1, 0.2, 0.3, -0.1, 0.0, 0.1, 0.2]
    x = 0.02 * rng.standard_normal((NORB,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    eri = sum(x.transpose(p) for p in perms) / 8.0
    return h1, eri


def _explicit_h(h1: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Dense H on ``STRS × STRS`` from the batched matvec applied to unit vectors."""
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, 0.0)
    strs = [int(s) for s in STRS]
    hii, adets, bdets, integrals, helpers = _setup(
        strs, strs, NORB, I0, I1, I2, torch.device("cpu"), torch.float64, 2_000_000
    )
    matvec = build_matvec(
        "batched", hii, adets, bdets, NORB, integrals, helpers, chunk_size=2_000_000
    )
    eye = torch.eye(len(strs) ** 2, dtype=torch.float64)
    return torch.stack([matvec(eye[:, k]) for k in range(eye.shape[0])], dim=1).numpy()


def _reference() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Integrals, exact ground energy and min(diag H), checking the blocks they rely on."""
    h1, eri = _integrals()
    H = _explicit_h(h1, eri)
    low = np.isin(STRS, LOW)
    block = 2 * np.repeat(low, len(STRS)) + np.tile(low, len(STRS))
    assert np.all(H[block[:, None] != block[None, :]] == 0.0), "blocks must be decoupled"
    e_true = float(np.linalg.eigvalsh(H)[0])
    seed_block = block == block[0]
    e_seed_block = float(np.linalg.eigvalsh(H[np.ix_(seed_block, seed_block)])[0])
    assert e_seed_block > e_true + 0.1, "index 0's block must not hold the ground state"
    ground_block = block == block[int(np.argmin(np.diag(H)))]
    e_ground_block = float(np.linalg.eigvalsh(H[np.ix_(ground_block, ground_block)])[0])
    assert e_ground_block == pytest.approx(e_true, abs=1e-12), "argmin(hii) must share its block"
    return h1, eri, e_true, float(np.diag(H).min())


def _distributed_local() -> dict:
    """Two-rank gloo owner-computes launch config."""
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    return {
        "distributed": {
            "world_size": 2,
            "backend": "gloo",
            "persistent": False,
            "timeout": 300,
            "matvec": "local",
        }
    }


def _solve(mode: str, h1: np.ndarray, eri: np.ndarray, **kwargs) -> float:
    """Ground energy of the full ``STRS × STRS`` space (GDB: its full pair list)."""
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        mode=mode,
        apply_ab_projection=False,
        davidson_block_size=8,
        max_davidson_iters=50,
        davidson_tolerance=TOL,
        **kwargs,
    )
    if mode == "gdb":
        basis = (np.repeat(STRS, len(STRS)), np.tile(STRS, len(STRS)))
    else:
        basis = (STRS, STRS)
    return float(solver([basis], h1, eri, NORB, NELEC)[0].states[0].energy)


def test_basis_init_vector_seed_index_moves_unit_seed_and_keeps_noise() -> None:
    """``seed_index=k`` moves the unit seed to k; the noise pattern is unchanged."""
    w0 = basis_init_vector_torch(3, 4, init=0)
    w5 = basis_init_vector_torch(3, 4, init=0, seed_index=5)
    shift = torch.zeros(12, dtype=torch.float64)
    shift[5], shift[0] = 1.0, -1.0
    np.testing.assert_allclose((w5 - w0).numpy(), shift.numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="seed_index"):
        basis_init_vector_torch(3, 4, init=0, seed_index=12)


@pytest.mark.parametrize("layout", ["replicated", "distributed_local"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_cold_seed_reaches_the_ground_block(mode: str, layout: str) -> None:
    """The cold start finds the ground state outside the smallest-int determinant's block."""
    h1, eri, e_true, diag_min = _reference()
    extra = _distributed_local() if layout == "distributed_local" else {}
    energy = _solve(mode, h1, eri, **extra)
    assert energy <= diag_min + 1e-9
    assert energy == pytest.approx(e_true, abs=1e-5)


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_start_above_min_diagonal_is_repeated_from_it(mode: str) -> None:
    """A ``connected`` start in the wrong block ends above min(diag H): one warned re-solve."""
    h1, eri, e_true, diag_min = _reference()
    with pytest.warns(RuntimeWarning, match=r"above min\(diag H\)"):
        energy = _solve(mode, h1, eri, seed_strategy="connected")
    assert energy <= diag_min + 1e-9
    assert energy == pytest.approx(e_true, abs=1e-5)
