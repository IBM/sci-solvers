"""Regression tests for the Davidson start on a block-diagonal Hamiltonian.

H couples two determinants only when they differ by at most a double
excitation, so a basis whose CI strings form clusters further apart splits H
into blocks, and Davidson never leaves the block its start vector lies in. The
TPB and GDB cold starts seeded the smallest-α × smallest-β determinant, and a
solve whose seed sat in a high block returned that block's lowest eigenvalue.
Production symptom: on a 36-orbital [4Fe-4S] space, the α- and β-like strings
of a broken-symmetry reference symmetrized into one set gave an energy 16 Ha
above the reference determinant's diagonal.

Seeding at the lowest diagonal entry does not settle it: a block whose lowest
diagonal entry is higher can hold a lower eigenvalue when its couplings are
stronger. A single-root cold start therefore seeds at the lowest entry of the
diagonal Davidson sees, each other block is then solved from its own lowest
entry, and the lowest energy wins. Both test spaces use the same eight strings
(three electrons in orbitals 0-3 or in orbitals 4-7, six bits apart), so the
product space splits into four blocks:

* ``favourable``: the lowest diagonal entry lies in the ground state's block
  and the smallest-int determinant does not, so the first solve must already
  reach the ground state;
* ``counter``: the lowest diagonal entry and the smallest-int determinant lie
  in the weakly coupled low block, and the strongly coupled high block holds
  the ground state, so only the per-block solves find it.

On a connected small-molecule space the cold start must keep the exact energy.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import itertools
import warnings

import numpy as np
import pytest
import torch
from torch_sci_solver import make_torch_sci_solver
from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
from torch_sci_solver.torch_impl.davidson import basis_init_vector_torch
from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
from torch_sci_solver.torch_impl.rdm import _inter_to_block_signs
from torch_sci_solver.torch_impl.validate_matvec import _setup

NORB = 8
NELEC = (3, 3)


def _window_strings(orbitals, n_electrons: int = NELEC[0]) -> list[int]:
    """All ``n_electrons``-electron strings inside ``orbitals``."""
    return [sum(1 << p for p in occ) for occ in itertools.combinations(orbitals, n_electrons)]


# Two clusters 6 bits apart: no single or double excitation connects them, so the
# product space splits into four blocks (low/high α × low/high β), and the
# smallest-int determinant (basis index 0) lies in the (low, low) block.
LOW = _window_strings(range(0, 4))
HIGH = _window_strings(range(4, 8))
STRS = np.array(sorted(LOW + HIGH), dtype=np.int64)
PAIRS = (np.repeat(STRS, len(STRS)), np.tile(STRS, len(STRS)))

# Loose enough that the seed's 1e-7 noise can never lift the residual outside
# the seed's block above it (the regression bites on any layout); also the
# default of the sci_solvers wrapper.
TOL = 1e-4


def _symmetrized_eri(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    """Random real ERIs with the 8-fold permutational symmetry."""
    x = scale * rng.standard_normal((n,) * 4)
    perms = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    perms += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]
    return sum(x.transpose(p) for p in perms) / 8.0


def _favourable_integrals() -> tuple[np.ndarray, np.ndarray]:
    """Small random integrals whose high-window orbitals sit 0.1 Ha lower.

    The (high, high) block then holds both the lowest diagonal entry and the
    ground state, a few tenths of a hartree below the (low, low) block.
    """
    rng = np.random.default_rng(7)
    h1 = 0.02 * rng.standard_normal((NORB, NORB))
    h1 = 0.5 * (h1 + h1.T)
    h1[np.diag_indices(NORB)] += [0.0, 0.1, 0.2, 0.3, -0.1, 0.0, 0.1, 0.2]
    return h1, _symmetrized_eri(rng, NORB, 0.02)


def _counter_integrals(n: int = NORB, onsite: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Integrals whose low window is weakly coupled and whose high window is strongly coupled.

    The low-window orbitals sit at -0.1 Ha and hop by about 0.01 Ha; the
    high-window ones sit at ``onsite`` and hop by about +0.4 Ha, all to all,
    which puts all but one of their one-electron levels 0.4 Ha below
    ``onsite``. The (low, low) block then holds the lowest diagonal entry and
    the smallest-int determinant, while the (high, high) block's lowest
    eigenvalue lies about 0.6 Ha below every eigenvalue of the (low, low) block.

    The high window's diagonal entries sit further above the low block's
    energy than the high block's ground state sits below it, so Davidson's
    diagonal preconditioner damps the seed's 1e-7 noise in the high block
    instead of amplifying it: the solve from the lowest diagonal entry stays in
    the low block, as on the [4Fe-4S] space. With ``onsite=0`` the noise
    grows instead, and that solve leaks into the high block on its own.
    """
    rng = np.random.default_rng(11)
    half = n // 2
    weak = 0.01 * rng.standard_normal((half, half))
    strong = 0.4 + 0.05 * rng.standard_normal((n - half, n - half))
    h1 = np.zeros((n, n))
    h1[:half, :half] = 0.5 * (weak + weak.T)
    h1[half:, half:] = 0.5 * (strong + strong.T)
    h1[np.diag_indices(n)] = [-0.1] * half + [onsite] * (n - half)
    return h1, _symmetrized_eri(rng, n, 0.005)


def _mixed_integrals() -> tuple[np.ndarray, np.ndarray]:
    """Integrals whose ground state lies in a mixed (low, high) block.

    Opposite-spin electrons repel by 0.05 Ha inside each window and not
    across, so a mixed block saves 0.45 Ha against a same-window one. With the
    high window's +0.3 Ha all-to-all hopping and 0.25 Ha onsite energy, the
    (low, high) block and its mirror then hold the ground state 0.3 Ha below
    the (low, low) block, which still holds the lowest diagonal entry. A
    projected solve of that block must fill its mirror block too.
    """
    rng = np.random.default_rng(13)
    weak = 0.01 * rng.standard_normal((4, 4))
    strong = 0.3 + 0.05 * rng.standard_normal((4, 4))
    h1 = np.zeros((NORB, NORB))
    h1[:4, :4] = 0.5 * (weak + weak.T)
    h1[4:, 4:] = 0.5 * (strong + strong.T)
    h1[np.diag_indices(NORB)] = [-0.15, -0.1, -0.05, 0.0] + [0.25] * 4
    eri = _symmetrized_eri(rng, NORB, 0.005)
    for window in (range(0, 4), range(4, 8)):
        for p in window:
            for q in window:
                eri[p, p, q, q] += 0.05
    return h1, eri


SPACES = {
    "favourable": _favourable_integrals,
    "counter": _counter_integrals,
    "mixed": _mixed_integrals,
}


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


def _projected_h(H: np.ndarray) -> np.ndarray:
    """``H`` on the αβ-swap-symmetric sector the projection keeps, in an orthonormal basis.

    The projector is ``P v[i, j] = (v[i, j] + τ[i, j] v[j, i]) / 2`` in the
    interleaved-Wick basis of the matvec, with ``τ = σ σᵀ``.
    """
    strs = torch.as_tensor(STRS)
    sigma = _inter_to_block_signs(strs, strs, NORB).to(torch.float64).numpy()
    tau = sigma * sigma.T
    m = len(STRS)
    rows, cols = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    swap = np.zeros((m * m, m * m))
    swap[(rows * m + cols).ravel(), (cols * m + rows).ravel()] = tau.ravel()
    weights, vectors = np.linalg.eigh(0.5 * (np.eye(m * m) + swap))
    sector = vectors[:, weights > 0.5]
    return sector.T @ H @ sector


def _space(name: str, projection: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrals and exact ground energy of a test space, checking the blocks it relies on.

    With ``projection`` the energy is the lowest on the αβ-swap-symmetric sector.
    """
    h1, eri = SPACES[name]()
    H = _explicit_h(h1, eri)
    low = np.isin(STRS, LOW)
    block = 2 * np.repeat(low, len(STRS)) + np.tile(low, len(STRS))
    assert np.all(H[block[:, None] != block[None, :]] == 0.0), "blocks must be decoupled"
    e_true = float(np.linalg.eigvalsh(H)[0])

    def lowest(det: int) -> float:
        """Lowest eigenvalue of the block holding determinant ``det``."""
        keep = block == block[det]
        return float(np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0])

    assert lowest(0) > e_true + 0.1, "index 0's block must not hold the ground state"
    seed_block_lowest = lowest(int(np.argmin(np.diag(H))))
    if name == "favourable":
        assert seed_block_lowest == pytest.approx(e_true, abs=1e-12), "argmin(hii) shares it"
    else:
        assert seed_block_lowest > e_true + 0.1, "argmin(hii)'s block must not hold it"
    if projection:
        e_true = float(np.linalg.eigvalsh(_projected_h(H))[0])
    return h1, eri, e_true


def _layout(name: str) -> dict:
    """Solver kwargs of a layout: one process, or two gloo ranks."""
    if name == "replicated":
        return {}
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    distributed = {
        "world_size": 2,
        "backend": "gloo",
        "persistent": False,
        "timeout": 300,
        "matvec": "local",
    }
    if name == "distributed_replicated":
        distributed["matvec"] = "replicated"
    elif name == "tile":
        distributed["tile"] = "force"
    return {"distributed": distributed}


def _solve(
    mode: str,
    h1: np.ndarray,
    eri: np.ndarray,
    *,
    basis: tuple[np.ndarray, np.ndarray] | None = None,
    projection: bool = False,
    call: dict | None = None,
    **kwargs,
) -> tuple[list[float], dict]:
    """Energies and per-subspace perf record of one solve (GDB default: the full pair list)."""
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        mode=mode,
        apply_ab_projection=projection,
        davidson_block_size=8,
        max_davidson_iters=50,
        davidson_tolerance=TOL,
        **kwargs,
    )
    if basis is None:
        basis = PAIRS if mode == "gdb" else (STRS, STRS)
    result = solver([basis], h1, eri, NORB, NELEC, **(call or {}))
    perf = solver.stats["per_call_log"][-1]["subspaces"][0]
    return [float(state.energy) for state in result[0].states], perf


def _assert_start_missed_and_blocks_found(
    perf: dict, e_true: float, *, start_may_leak: bool = False
) -> None:
    """The start's solve ended in its own block, above ``e_true``; another block's solve found it.

    The start's solve is the one without a seed index (the caller's start: the
    lowest diagonal entry, or the ``connected`` determinant); the others
    started at each other block's lowest diagonal entry. ``start_may_leak``
    drops the first check where the start's 1e-7 seed noise can carry it into
    the ground block by itself.
    """
    solves = perf["block_solves"]
    start = [solve for solve in solves if solve["seed"] is None]
    assert len(start) == 1 and len(solves) >= 2
    if not start_may_leak:
        assert start[0]["E"] > e_true + 0.1, "the start's solve must stay in its own block"
    assert min(solve["E"] for solve in solves) == pytest.approx(e_true, abs=1e-5)
    rescued = [solve for solve in solves if solve["seed"] is not None]
    assert min(solve["E"] for solve in rescued) == pytest.approx(e_true, abs=1e-5)


def test_basis_init_vector_seed_index_moves_unit_seed_and_keeps_noise() -> None:
    """``seed_index=k`` moves the unit seed to k; the noise pattern is unchanged."""
    w0 = basis_init_vector_torch(3, 4, init=0)
    w5 = basis_init_vector_torch(3, 4, init=0, seed_index=5)
    shift = torch.zeros(12, dtype=torch.float64)
    shift[5], shift[0] = 1.0, -1.0
    np.testing.assert_allclose((w5 - w0).numpy(), shift.numpy(), atol=1e-6)
    with pytest.raises(ValueError, match="seed_index"):
        basis_init_vector_torch(3, 4, init=0, seed_index=12)


def test_block_labels_split_the_two_window_space() -> None:
    """String union-find finds the two windows; product and pair labels agree with it."""
    from torch_sci_solver.torch_impl.hamiltonian_blocks import (
        edge_forest,
        graph_block_labels,
        product_block_labels,
        string_forest,
    )

    forest = string_forest(STRS, NORB)
    assert forest.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]

    labels, n_blocks, trivial = product_block_labels(forest, forest, symmetric=False)
    grid = labels.reshape(len(STRS), len(STRS)).numpy()
    assert n_blocks == 4 and not bool(trivial.any())
    assert len(np.unique(grid)) == 4
    merged = product_block_labels(forest, forest, symmetric=True)[0]
    merged_grid = merged.reshape(len(STRS), len(STRS)).numpy()
    assert len(np.unique(merged_grid)) == 3 and merged_grid[0, 4] == merged_grid[4, 0]
    flat = torch.tensor([0, 4, 36, 63])
    picked = product_block_labels(forest, forest, symmetric=False, index=flat)[0]
    assert picked.tolist() == grid.ravel()[flat.numpy()].tolist()

    edges = [(torch.tensor([0, 2]), torch.tensor([1, 3])), (torch.tensor([1]), torch.tensor([2]))]
    chain = edge_forest(5, edges)
    assert chain.tolist() == [0, 0, 0, 0, 4]
    pair_labels, n_pair_blocks, pair_trivial = graph_block_labels(chain)
    assert pair_labels.tolist() == [0, 0, 0, 0, 1] and n_pair_blocks == 2
    assert pair_trivial.tolist() == [False, True]
    mirrored = graph_block_labels(chain, swap=torch.tensor([4, 1, 2, 3, 0]))
    assert mirrored[1] == 1 and not bool(mirrored[2].any())


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("layout", ["replicated", "distributed_local"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_cold_seed_reaches_the_ground_block(mode: str, layout: str, projection: bool) -> None:
    """The first solve, from the lowest diagonal entry, already finds the ground state.

    Index 0's block does not hold it here; the lowest diagonal entry's does. The
    solves of the other blocks must not change the answer, and the warning
    filter fails the test if anything but the seed had to rescue it.
    """
    h1, eri, e_true = _space("favourable", projection)
    energies, perf = _solve(mode, h1, eri, projection=projection, **_layout(layout))
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    assert perf["n_blocks"] >= 3
    assert perf["block_solves"][0]["seed"] is None
    assert perf["block_solves"][0]["E"] == pytest.approx(e_true, abs=1e-5)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize(
    "layout", ["replicated", "distributed_replicated", "distributed_local", "tile"]
)
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_ground_block_above_the_lowest_diagonal_is_solved(
    mode: str, layout: str, projection: bool
) -> None:
    """A strongly coupled block beats the weakly coupled one that holds the lowest diagonal.

    The first solve, from the lowest diagonal entry, ends in the weak block:
    below min(diag H), so no bound flags it, yet several tenths of a hartree
    above the ground state. The solves of the other blocks must find it.
    """
    h1, eri, e_true = _space("counter", projection)
    # A 64-pair list is below the "auto" block decomposition's size floor, and
    # the projection rules out the masked superset, so a pair tile needs "on".
    extra_kwargs: dict = {}
    if (mode, layout) == ("gdb", "tile"):
        extra_kwargs = {"gdb_kwargs": {"block_matvec": "on"}}
    energies, perf = _solve(mode, h1, eri, projection=projection, **extra_kwargs, **_layout(layout))
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    _assert_start_missed_and_blocks_found(perf, e_true)
    # One device solves each other product block on its own strings; pair
    # bases and distributed layouts solve it on the full space.
    extra = [solve for solve in perf["block_solves"] if solve["seed"] is not None]
    if mode == "tpb" and layout == "replicated":
        assert all(solve["dimension"] < len(STRS) ** 2 for solve in extra)
    else:
        assert all(solve["dimension"] == len(STRS) ** 2 for solve in extra)


def _ground_state(
    h1: np.ndarray, eri: np.ndarray, *, projection: bool, call: dict | None, **kwargs
) -> tuple[float, np.ndarray, dict]:
    """Energy, interleaved-Wick ground vector on ``STRS × STRS`` and perf of a tight TPB solve."""
    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        mode="tpb",
        apply_ab_projection=projection,
        davidson_block_size=8,
        max_davidson_iters=200,
        davidson_tolerance=1e-6,
        **kwargs,
    )
    state = solver([(STRS, STRS)], h1, eri, NORB, NELEC, **(call or {}))[0].states[0]
    perf = solver.stats["per_call_log"][-1]["subspaces"][0]
    strs = torch.as_tensor(STRS)
    sigma = _inter_to_block_signs(strs, strs, NORB).to(torch.float64).numpy()
    return float(state.energy), (np.asarray(state.sci_state.amplitudes) * sigma).ravel(), perf


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    ("space", "projection", "spin_sq"),
    [
        ("counter", False, None),
        ("counter", True, None),
        ("counter", False, 0.0),
        ("mixed", False, None),
        ("mixed", True, None),
    ],
)
def test_block_solves_on_their_own_strings_match_full_space_solves(
    space: str, projection: bool, spin_sq: float | None
) -> None:
    """A block solved on its own strings and padded back is the full-space block solve.

    One device solves each other product block on its own strings; a tile grid
    (the (P, 1) grid with the spin penalty) solves it on the full space. Both
    must give the exact energy and an eigenvector of H; the same one, up to
    sign, unless the ground state is degenerate (``mixed`` unprojected: the
    (low, high) block and its mirror). ``mixed`` projected pads the mirror
    block. Tolerance 1e-6 puts the energy error near 1e-12 Ha.
    """
    h1, eri, e_true = _space(space, projection)
    H = _explicit_h(h1, eri)
    call = None if spin_sq is None else {"spin_sq": spin_sq}
    full_layout = "tile" if spin_sq is None else "distributed_local"
    e_own, v_own, perf_own = _ground_state(h1, eri, projection=projection, call=call)
    e_full, v_full, perf_full = _ground_state(
        h1, eri, projection=projection, call=call, **_layout(full_layout)
    )
    n = len(STRS) ** 2
    own = [solve for solve in perf_own["block_solves"] if solve["seed"] is not None]
    full = [solve for solve in perf_full["block_solves"] if solve["seed"] is not None]
    assert own and full
    for solve in own:
        assert solve["dimension"] < n and solve["dimension"] in (solve["size"], solve["size"] // 2)
    assert all(solve["dimension"] == n for solve in full)
    for energy, vector in ((e_own, v_own), (e_full, v_full)):
        assert energy == pytest.approx(e_true, abs=1e-10)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-10)
        assert np.linalg.norm(H @ vector - energy * vector) < 1e-5
    if not (space == "mixed" and not projection):
        assert abs(float(v_own @ v_full)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_connected_start_in_another_block_still_reaches_the_ground_state(mode: str) -> None:
    """``seed_strategy="connected"`` keeps its start; the other blocks' solves find the ground."""
    h1, eri, e_true = _space("favourable")
    energies, perf = _solve(mode, h1, eri, seed_strategy="connected")
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    _assert_start_missed_and_blocks_found(perf, e_true)


def test_block_solves_start_at_the_lowest_diagonal_whatever_the_strategy() -> None:
    """A block solved on its own strings starts at its lowest diagonal entry, as on the full space.

    Seeding it at the block's most-connected determinant instead converged to an
    excited state on a [4Fe-4S] block (0.22 Ha above the block's ground state).
    """
    solver = make_torch_sci_solver(
        device="cpu", seed_strategy="connected", apply_ab_projection=False
    )
    diagonal = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    assert solver._lowest_diagonal_index(diagonal) is None
    assert solver._lowest_diagonal_index(diagonal, block_solve=True) == 1


def test_lowest_diagonal_is_another_name_for_smallest_int() -> None:
    """``seed_strategy="lowest_diagonal"`` runs exactly the default solves."""
    h1, eri, _ = _space("counter")
    energies, perf = _solve("tpb", h1, eri, seed_strategy="lowest_diagonal")
    reference, reference_perf = _solve("tpb", h1, eri)
    assert energies == pytest.approx(reference, abs=1e-12)
    assert perf["block_solves"] == reference_perf["block_solves"]


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_spin_penalty_tile_path_seeds_and_solves_every_block() -> None:
    """The owner-computes spin-penalty solve, on its (P, 1) tile grid, solves every block.

    The ground state is a singlet, so the penalty-stripped energy is exact. On
    this layout the start's seed noise reaches the ground block by itself (the
    seed-only solver gets it too), so the test checks that the strong block's
    own solve finds it, not that the start misses it.
    """
    h1, eri, e_true = _space("counter")
    energies, perf = _solve("tpb", h1, eri, call={"spin_sq": 0.0}, **_layout("distributed_local"))
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    _assert_start_missed_and_blocks_found(perf, e_true, start_may_leak=True)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_masked_superset_tile_path_seeds_and_solves_every_block() -> None:
    """An irregular pair list on a tile grid solves its masked product block by block.

    The cold start seeds among the caller's pairs at the lowest diagonal entry
    (the inactive product pairs carry a large filled diagonal), not at the
    caller's first pair, and the other blocks are solved after it.
    """
    h1, eri = _counter_integrals()
    H = _explicit_h(h1, eri)
    m = len(STRS)
    low = np.isin(STRS, LOW)
    rows, cols = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    # Every (low, low) and (high, high) pair plus one mixed partner per string,
    # each a different one, so no two strings share a partner set.
    mixed = np.where(low[rows], cols == rows + 4, cols == (rows - 3) % 4)
    kept = ((low[rows] == low[cols]) | mixed).ravel()
    e_true = float(np.linalg.eigvalsh(H[np.ix_(kept, kept)])[0])
    basis = (PAIRS[0][kept], PAIRS[1][kept])
    energies, perf = _solve("gdb", h1, eri, basis=basis, **_layout("tile"))
    assert perf["pair_superset"] is True
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    _assert_start_missed_and_blocks_found(perf, e_true)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("layout", ["distributed_local", "tile"])
def test_structural_pair_basis_seeds_and_solves_every_block(layout: str) -> None:
    """A structural GDB basis (four Cartesian blocks) seeds low and solves every block."""
    h1, eri, e_true = _space("counter")
    low, high = np.array(LOW, dtype=np.int64), np.array(HIGH, dtype=np.int64)
    blocks = [(low, low), (high, high), (low, high), (high, low)]
    placeholder = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    energies, perf = _solve(
        "gdb",
        h1,
        eri,
        basis=placeholder,
        gdb_kwargs={"block_matvec": "on", "structural_blocks": [blocks]},
        **_layout(layout),
    )
    assert energies[0] == pytest.approx(e_true, abs=1e-5)
    _assert_start_missed_and_blocks_found(perf, e_true)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("layout", ["replicated", "distributed_local"])
def test_spinless_ground_block_above_the_lowest_diagonal_is_solved(layout: str) -> None:
    """The spinless path, complex dtype: the strongly coupled block's ground state wins."""
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals_spinless
    from torch_sci_solver.torch_impl.mult_spinless import (
        make_diag_terms_spinless_torch,
        mult_spinless_torch,
    )
    from torch_sci_solver.torch_impl.spinless_helpers import (
        generate_spinless_helpers,
        spinless_basis_from_array,
    )

    nmo, n_particles = 10, 3
    # onsite=0: the solve from the lowest diagonal entry stays put on this basis.
    hcore, eri = _counter_integrals(nmo, onsite=0.0)
    low_strings = _window_strings(range(0, 5), n_particles)
    strs = np.array(sorted(low_strings + _window_strings(range(5, 10), n_particles)))
    basis = spinless_basis_from_array(strs, nmo, n_particles, device="cpu")
    helpers = generate_spinless_helpers(basis)
    integrals = to_torch_integrals_spinless(hcore, eri, device="cpu", dtype=torch.complex128)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    eye = torch.eye(basis.dim, dtype=torch.complex128)
    M = mult_spinless_torch(hii, eye, basis, helpers, integrals).numpy()
    H = 0.5 * (M + M.conj().T)
    e_true = float(np.linalg.eigvalsh(H)[0])
    low = np.isin(strs, low_strings)  # one-limb basis: the rows stay in sorted order
    assert np.all(H[low[:, None] != low[None, :]] == 0.0), "blocks must be decoupled"
    assert low[int(torch.argmin(hii))], "the lowest diagonal entry must be in the low block"
    assert float(np.linalg.eigvalsh(H[np.ix_(low, low)])[0]) > e_true + 0.1

    solver = make_torch_sci_solver(
        device="cpu",
        dtype=torch.complex128,
        davidson_block_size=8,
        max_davidson_iters=50,
        davidson_tolerance=TOL,
        **_layout(layout),
    )
    energy = float(solver([strs], hcore, eri, nmo, n_particles)[0].states[0].energy)
    assert energy == pytest.approx(e_true, abs=1e-5)


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_multi_root_on_a_block_diagonal_space(mode: str) -> None:
    """A multi-root solve is not split by block: it warns; the missed-root check finds the roots.

    Its seed block holds only weak-block determinants here (the smallest-int one
    and the lowest diagonal ones), so without the check it cannot reach the
    strongly coupled block.
    """
    h1, eri = _counter_integrals()
    exact = np.linalg.eigvalsh(_explicit_h(h1, eri))[:2]
    with pytest.warns(RuntimeWarning, match="multi-root"):
        _solve(mode, h1, eri, n_roots=2)
    energies, perf = _solve(mode, h1, eri, n_roots=2, davidson_missed_root_check=True)
    assert energies == pytest.approx(exact, abs=1e-5)
    assert perf["n_blocks"] >= 3


def _h2o_sto3g() -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
    """RHF/STO-3G integrals of H2O: 7 orbitals, 5 + 5 electrons (FCI dimension 441)."""
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    return h1, eri, norb, mol.nelec


@pytest.mark.parametrize("space", ["full", "hf_singles"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_cold_seed_keeps_the_exact_energy_on_a_connected_molecule_space(
    mode: str, space: str
) -> None:
    """On a connected H2O/STO-3G space the default cold start matches PySCF to 1e-10 Ha.

    ``full`` is every α × β string, so the reference is FCI. ``hf_singles`` is
    HF plus its single excitations in each spin (11 of the 21 strings), a
    selected subspace that singles and doubles connect; the reference is PySCF's
    selected CI on the same strings. Either space is one Hamiltonian block, so
    the guard against a start in the wrong block must stay silent.
    """
    from pyscf import fci
    from pyscf.fci import selected_ci

    h1, eri, norb, nelec = _h2o_sto3g()
    n_occ = nelec[0]
    hf = (1 << n_occ) - 1
    if space == "full":
        occs = itertools.combinations(range(norb), n_occ)
        strs = np.array(sorted(sum(1 << p for p in occ) for occ in occs), dtype=np.int64)
        reference = fci.direct_spin1.FCI()
        reference.conv_tol = 1e-12
        e_ref = float(reference.kernel(h1, eri, norb, nelec)[0])
    else:
        singles = [hf ^ (1 << i) ^ (1 << a) for i in range(n_occ) for a in range(n_occ, norb)]
        strs = np.array(sorted([hf, *singles]), dtype=np.int64)
        reference = selected_ci.SelectedCI()
        reference.conv_tol = 1e-12
        result = selected_ci.kernel_fixed_space(reference, h1, eri, norb, nelec, (strs, strs))
        e_ref = float(result[0])

    solver = make_torch_sci_solver(
        device="cpu",
        matvec_impl="batched",
        mode=mode,
        apply_ab_projection=False,
        davidson_block_size=8,
        max_davidson_iters=200,
        davidson_tolerance=1e-9,
    )
    if mode == "gdb":
        basis = (np.repeat(strs, len(strs)), np.tile(strs, len(strs)))
    else:
        basis = (strs, strs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        energy = float(solver([basis], h1, eri, norb, nelec)[0].states[0].energy)
    assert not [w for w in caught if "above min(diag H)" in str(w.message)]
    assert energy == pytest.approx(e_ref, abs=1e-10)
