"""One-word CI strings with bit 63 set (orbital 63 of a 64-orbital basis).

A 64-orbital string is one unsigned 64-bit word, but the engine stores it as a
signed int64 bit pattern, so a string with orbital 63 occupied is a negative
int64. These tests pin the unsigned semantics end to end:

1. Every input form — uint64 arrays, int64 arrays of signed patterns, and
   Python ints up to ``2**64 - 1`` — solves to the same state.
2. An orbital-relabeling oracle: the same Hamiltonian with orbital 63 renamed
   to orbital 6 has an identical spectrum, 1-RDM (permuted) and ``<S^2>``, for
   TPB, GDB (streamed and block), spinless, warm starts and ``init_vectors``.
3. The basis is in integer order (the HF-like string first, the
   ``smallest_int`` seed) and the returned ``SCIState`` strings are uint64.
4. The validator flags a bit-63 string when ``norb < 64``.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import functools
import itertools

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.qiskit_adapter import make_torch_sci_solver

N_ORB = 64
ACTIVE = [0, 1, 2, 3, 4, 5, 63]
RELABEL = {orbital: index for index, orbital in enumerate(ACTIVE)}
NELEC = (3, 3)
KINDS = ["uint64", "int64-signed", "python-int"]
_PERMS = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
_PERMS += [(2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]


@functools.lru_cache(maxsize=None)
def _system(top: bool):
    """Integrals on the active orbitals, placed at ``ACTIVE`` (top) or at 0..6."""
    rng = np.random.default_rng(0)
    k = len(ACTIVE)
    h_act = rng.standard_normal((k, k))
    h_act = 0.5 * (h_act + h_act.T)
    raw = rng.standard_normal((k,) * 4)
    eri_act = sum(raw.transpose(p) for p in _PERMS) / 8.0
    orbitals = np.array(ACTIVE if top else list(range(k)))
    h1 = np.zeros((N_ORB, N_ORB))
    eri = np.zeros((N_ORB,) * 4)
    h1[np.ix_(orbitals, orbitals)] = h_act
    eri[np.ix_(orbitals, orbitals, orbitals, orbitals)] = eri_act
    strings = [sum(1 << o for o in c) for c in itertools.combinations(ACTIVE, 3)]
    if not top:
        strings = [sum(1 << RELABEL[o] for o in ACTIVE if (s >> o) & 1) for s in strings]
    return h1, eri, strings


def _as_input(strings: list[int], kind: str):
    """The strings in one caller-side representation."""
    if kind == "uint64":
        return np.array(strings, dtype=np.uint64)
    if kind == "int64-signed":
        return np.array([v - (1 << 64) if v >= 1 << 63 else v for v in strings], dtype=np.int64)
    return list(strings)


def _solver(**kwargs):
    """A tight CPU solver."""
    kwargs.setdefault("apply_ab_projection", False)
    return make_torch_sci_solver(
        device="cpu", davidson_tolerance=1e-10, max_davidson_iters=100, **kwargs
    )


def _pairs(strings: list[int]) -> tuple[list[int], list[int]]:
    """The Cartesian pair list of one string set with itself."""
    return [a for a in strings for _ in strings], [b for _ in strings for b in strings]


def _ci(strings: list[int], kind: str, mode: str):
    """The ``ci_strings`` entry of one subspace."""
    if mode == "tpb":
        return [(_as_input(strings, kind), _as_input(strings, kind))]
    a, b = _pairs(strings)
    return [(_as_input(a, kind), _as_input(b, kind))]


def _rdm1_active(rdm1, orbitals) -> np.ndarray:
    """The 1-RDM restricted to ``orbitals`` (spin blocks kept)."""
    r = np.asarray(rdm1)
    ix = np.array(orbitals)
    return r[:, ix][:, :, ix] if r.ndim == 3 else r[np.ix_(ix, ix)]


def _solve(top: bool, kind: str, mode: str, **kwargs):
    """Ground state and ``<S^2>`` on the top-bit (or relabeled) basis."""
    h1, eri, strings = _system(top)
    solver = _solver(mode=mode, **kwargs)
    state = solver(_ci(strings, kind, mode), h1, eri, N_ORB, NELEC)[0].states[0]
    s2 = solver.stats["spin_squared"][-1] if solver.stats["spin_squared"] else None
    return state, s2


def _assert_oracle(top_state, top_s2, low_state, low_s2) -> None:
    """The top-bit solve equals the relabeled solve."""
    assert top_state.state == "converged"
    assert float(top_state.energy) == pytest.approx(float(low_state.energy), abs=1e-9)
    if low_state.rdm1 is not None:
        np.testing.assert_allclose(
            _rdm1_active(top_state.rdm1, ACTIVE),
            _rdm1_active(low_state.rdm1, range(len(ACTIVE))),
            atol=1e-8,
            rtol=0,
        )
    if low_s2 is not None:
        assert top_s2 == pytest.approx(low_s2, abs=1e-9)


_FEATURES = {
    "tpb_rdm_s2_projection": dict(
        mode="tpb", compute_full_rdm=True, compute_spin_squared=True, apply_ab_projection=True
    ),
    "gdb_streamed": dict(mode="gdb", gdb_kwargs={"block_matvec": "off"}),
    "gdb_block_rdm_s2": dict(
        mode="gdb",
        gdb_kwargs={"block_matvec": "on"},
        compute_full_rdm=True,
        compute_spin_squared=True,
    ),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("feature", sorted(_FEATURES))
def test_top_bit_basis_matches_relabeled_basis(feature: str, kind: str) -> None:
    """Orbital 63 behaves like any other orbital, in every input form."""
    top = _solve(True, kind, **_FEATURES[feature])
    low = _solve(False, "python-int", **_FEATURES[feature])
    _assert_oracle(*top, *low)


@pytest.mark.parametrize("kind", KINDS)
def test_top_bit_spinless_matches_relabeled_basis(kind: str) -> None:
    """The spinless path keys a bit-63 determinant like any other."""
    states = []
    for top in (True, False):
        h1, eri, strings = _system(top)
        solver = _solver(compute_full_rdm=True)
        strs = _as_input(strings, kind if top else "python-int")
        states.append(solver([strs], h1, eri, N_ORB, 3)[0].states[0])
    assert float(states[0].energy) == pytest.approx(float(states[1].energy), abs=1e-9)
    np.testing.assert_allclose(
        _rdm1_active(states[0].rdm1, ACTIVE),
        _rdm1_active(states[1].rdm1, range(len(ACTIVE))),
        atol=1e-8,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ["tpb", "gdb"])
@pytest.mark.parametrize("kind", KINDS)
def test_top_bit_warm_start(kind: str, mode: str) -> None:
    """A warm extension onto a bit-63 basis reproduces the cold energy."""
    h1, eri, strings = _system(True)
    solver = _solver(mode=mode)
    if mode == "tpb":
        small = [(_as_input(strings[:20], kind), _as_input(strings[:20], kind))]
    else:
        a, b = _pairs(strings)
        small = [(_as_input(a[:300], kind), _as_input(b[:300], kind))]
    _, cached = solver(small, h1, eri, N_ORB, NELEC, probe_method="warm_start")
    warm = solver(
        _ci(strings, kind, mode),
        h1,
        eri,
        N_ORB,
        NELEC,
        probe_method="warm_start",
        probe_state=cached,
    )[0].states[0]
    cold, _ = _solve(False, "python-int", mode)
    assert float(warm.energy) == pytest.approx(float(cold.energy), abs=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_top_bit_init_vectors_map_by_string(kind: str) -> None:
    """A converged vector re-fed in caller order converges at the first check."""
    h1, eri, strings = _system(True)
    caller = list(reversed(strings))
    solver = _solver()
    ci = [(_as_input(caller, kind), _as_input(caller, kind))]
    state = solver(ci, h1, eri, N_ORB, NELEC)[0].states[0].sci_state
    pos_a = {int(v): i for i, v in enumerate(state.ci_strs_a)}
    pos_b = {int(v): i for i, v in enumerate(state.ci_strs_b)}
    amplitudes = np.asarray(state.amplitudes)
    vec = amplitudes[np.ix_([pos_a[v] for v in caller], [pos_b[v] for v in caller])]
    again = solver(ci, h1, eri, N_ORB, NELEC, init_vectors=[vec.reshape(-1)], max_davidson_iters=1)
    assert again[0].states[0].state == "converged"
    assert solver.stats["per_call_log"][-1]["subspaces"][0]["warm"]


@pytest.mark.parametrize("kind", KINDS)
def test_top_bit_strings_are_returned_unsigned_in_integer_order(kind: str) -> None:
    """``SCIState`` strings are uint64, sorted as integers (HF-like first)."""
    _, _, strings = _system(True)
    tpb, _ = _solve(True, kind, "tpb")
    ci_a = np.asarray(tpb.sci_state.ci_strs_a)
    assert ci_a.dtype == np.uint64
    assert ci_a.tolist() == sorted(strings)
    gdb, _ = _solve(True, kind, "gdb")
    pair_a = np.asarray(gdb.sci_state.ci_strs_a)
    assert pair_a.dtype == np.uint64
    assert set(pair_a.tolist()) == set(strings)


def test_low_orbital_basis_keeps_int64_strings() -> None:
    """Without bit 63 the returned strings keep their int64 dtype."""
    state, _ = _solve(False, "python-int", "tpb")
    assert np.asarray(state.sci_state.ci_strs_a).dtype == np.int64


def test_smallest_int_pair_seed_is_unsigned_on_one_word_bases() -> None:
    """Bit-63 pairs sort first as negative patterns but are the largest integers."""
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
    from torch_sci_solver.torch_impl.qiskit_adapter import _smallest_int_pair_index

    strings = [0b111, 0b1011, (1 << 63) | 0b11]
    a, b = _pairs(strings)
    basis = pair_basis_from_arrays(
        np.array(a, dtype=np.uint64).view(np.int64), np.array(b, dtype=np.uint64).view(np.int64)
    )
    assert int(basis.alpha_of_pair[0]) < 0  # the signed sort puts bit 63 first
    index = _smallest_int_pair_index(basis)
    assert int(basis.alpha_of_pair[index]) == int(basis.beta_of_pair[index]) == 0b111


def test_validator_rejects_bit_63_below_64_orbitals() -> None:
    """A bit-63 string is out of range for ``norb < 64``, sign bit or not."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _validate_ci_basis

    good = np.array([0b11, 0b101], dtype=np.int64)
    bad = np.array([0b11, (1 << 63) | 0b1], dtype=np.uint64).view(np.int64)
    with pytest.raises(ValueError, match="bits set above orbital"):
        _validate_ci_basis(bad, good, 10, (2, 2))
    _validate_ci_basis(bad, good, 64, (2, 2))


@pytest.mark.slow
def test_top_bit_gloo_tiled_solves_match_relabeled_basis() -> None:
    """gloo, 2 ranks, TPB and GDB tile grids: the oracle holds distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    cfg = {
        "world_size": 2,
        "backend": "gloo",
        "persistent": False,
        "matvec": "local",
        "tile": "force",
        "timeout": 180,
    }
    for mode, extra in (("tpb", {}), ("gdb", {"gdb_kwargs": {"block_matvec": "on"}})):
        top = _solve(True, "uint64", mode, distributed=cfg, compute_spin_squared=True, **extra)
        low = _solve(False, "python-int", mode, compute_spin_squared=True, **extra)
        _assert_oracle(*top, *low)
        assert np.asarray(top[0].sci_state.ci_strs_a).dtype == np.uint64


def test_mixed_spins_share_one_string_dtype() -> None:
    """α using bit 63 and β not: both spins come back uint64, so they concatenate exactly."""
    h1, eri, strings = _system(True)
    low = [s for s in strings if s < 1 << 63][:20]
    solver = _solver()
    state = solver(
        [(_as_input(strings, "uint64"), _as_input(low, "uint64"))], h1, eri, N_ORB, NELEC
    )[0].states[0]
    ci_a, ci_b = np.asarray(state.sci_state.ci_strs_a), np.asarray(state.sci_state.ci_strs_b)
    assert ci_a.dtype == ci_b.dtype == np.uint64
    merged = np.concatenate([ci_a, ci_b])
    assert merged.dtype == np.uint64 and set(merged.tolist()) == set(strings) | set(low)


@pytest.mark.parametrize("kind", ["uint64", "python-int"])
def test_top_bit_gdb_init_vectors_map_by_pair(kind: str) -> None:
    """GDB init vectors resolve bit-63 pairs by value and converge at once."""
    h1, eri, strings = _system(True)
    a, b = _pairs(list(reversed(strings)))
    solver = _solver(mode="gdb")
    ci = [(_as_input(a, kind), _as_input(b, kind))]
    state = solver(ci, h1, eri, N_ORB, NELEC)[0].states[0].sci_state
    pos = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(state.ci_strs_a, state.ci_strs_b))}
    vec = np.array([np.asarray(state.amplitudes)[pos[(x, y)]] for x, y in zip(a, b)])
    again = solver(ci, h1, eri, N_ORB, NELEC, init_vectors=[vec], max_davidson_iters=1)
    assert again[0].states[0].state == "converged"


def test_gdb_accepts_its_own_signed_pair_lists() -> None:
    """The cached pair state's signed int64 patterns feed back as ``ci_strings``."""
    h1, eri, strings = _system(True)
    a, b = _pairs(strings)
    solver = _solver(mode="gdb")
    results, states = solver(
        [(_as_input(a, "uint64"), _as_input(b, "uint64"))],
        h1,
        eri,
        N_ORB,
        NELEC,
        probe_method="warm_start",
    )
    cached = states[0]
    assert any(v < 0 for v in cached.alpha_of_pair)  # signed bit patterns
    again = solver([(list(cached.alpha_of_pair), list(cached.beta_of_pair))], h1, eri, N_ORB, NELEC)
    assert float(again[0].states[0].energy) == pytest.approx(
        float(results[0].states[0].energy), abs=1e-9
    )


def test_preflight_infers_64_orbitals_from_signed_patterns() -> None:
    """``preflight`` without ``norb`` reads a negative int64 pattern as bit 63 set."""
    _, _, strings = _system(True)
    signed = _as_input(strings, "int64-signed")
    plan = _solver().preflight([(signed, signed)], NELEC)
    assert plan["subspaces"][0]["dimension"] == len(strings) ** 2


@pytest.mark.parametrize("norb", [10, 70])
def test_float_string_lists_are_rejected(norb: int) -> None:
    """A list of floats is not silently truncated to integers."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _ci_string_limbs, _signed_int64_keys

    with pytest.raises(TypeError):
        _ci_string_limbs([3.0, 5.7], norb)
    with pytest.raises(TypeError):
        _signed_int64_keys([3.0, 5.0])


@pytest.mark.parametrize("wide", [False, True])
def test_structural_seed_is_the_integer_smallest_pair(wide: bool) -> None:
    """A structural block holding top-bit strings does not take the seed."""
    from torch_sci_solver.torch_impl.pairs import structural_pair_basis_from_blocks
    from torch_sci_solver.torch_impl.qiskit_adapter import _smallest_int_pair_index

    top = (1 << 64) | (1 << 65) if wide else (1 << 63) | 0b1
    low = [0b11, 0b101, 0b110]
    blocks = [([top], [top]), (low, low)]
    basis = structural_pair_basis_from_blocks(blocks, n_orb=70 if wide else 64)
    index = _smallest_int_pair_index(basis)
    alpha, beta = basis.pair_strings(torch.tensor([index]))
    as_int = (
        (lambda t: int(t[0, 0]) % (1 << 64) | (int(t[0, 1]) % (1 << 64)) << 64)
        if wide
        else (lambda t: int(t[0]) % (1 << 64))
    )
    assert (as_int(alpha), as_int(beta)) == (0b11, 0b11)


def test_solve_fermion_takes_top_bit_string_tuples() -> None:
    """``solve_fermion`` keeps bit-63 strings exact from a tuple of ints or uint64."""
    from torch_sci_solver.qiskit import solve_fermion

    h1, eri, strings = _system(True)
    _, _, low_strings = _system(False)
    reference, _ = _solve(False, "python-int", "tpb")
    for kind in ("uint64", "python-int"):
        energy, state, _, _ = solve_fermion(
            (_as_input(strings, kind), _as_input(strings, kind)),
            h1,
            eri,
            davidson_tolerance=1e-10,
            max_davidson_iters=100,
        )
        assert energy == pytest.approx(float(reference.energy), abs=1e-9)
        assert np.asarray(state.ci_strs_a).dtype == np.uint64


def test_pair_sqd_loop_at_64_orbitals() -> None:
    """The pair SQD loop keys bit-63 pairs consistently across iterations."""
    from qiskit.primitives import BitArray
    from torch_sci_solver.qiskit import diagonalize_fermionic_hamiltonian_pairs

    h1, eri, strings = _system(True)

    def bits(value: int) -> list[int]:
        return [(value >> (N_ORB - 1 - i)) & 1 for i in range(N_ORB)]

    rows = [bits(b) + bits(a) for a in strings for b in strings]
    bit_array = BitArray.from_bool_array(np.array(rows, dtype=bool))
    result = diagonalize_fermionic_hamiltonian_pairs(
        h1,
        eri,
        bit_array,
        norb=N_ORB,
        nelec=NELEC,
        max_iterations=2,
        use_full_recovered_pool=True,
        seed=0,
        device="cpu",
        davidson_tolerance=1e-10,
        apply_ab_projection=False,
    )
    reference, _ = _solve(False, "python-int", "gdb")
    assert float(result.energy) == pytest.approx(float(reference.energy), abs=1e-8)
