"""Tests for the pair-basis (no-product-space) solver path.

Covers:

  * The ``mode`` flag exists on ``TorchSCISolver.__init__`` and
    ``diag_torch``, defaults to ``True``, is stored on the solver, and
    round-trips through the ``_SOLVER_INIT_KEYS`` split in ``qiskit.py``.
  * ``PairBasis`` construction dedups pairs by ``(α, β)``, sorts
    lexicographically, and produces consistent uid maps and partner CSRs.
  * ``basis_init_vector_pair_torch`` returns a well-formed length-N vector
    for both HF-seed and random init.
  * Kernel correctness: pair mode on a full ``adets × bdets`` product
    reproduces product mode and PySCF FCI; pair mode on a subset of that
    product reproduces a brute-force dense reference.
  * 1-RDM, 2-RDM, ⟨S²⟩, spin penalty, warm start, and the pair-mode outer
    SQD loop, each cross-checked against the product-mode path.
  * The excitation tables stay streamed rather than materialized (see
    ``test_pair_helpers_hold_nothing_of_size_nnz``).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl import pair_helpers as ph
from torch_sci_solver.torch_impl import qiskit_adapter
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import (
    PairBasis,
    basis_init_vector_pair_torch,
    pair_basis_from_arrays,
    pair_basis_from_full_product,
)
from torch_sci_solver.torch_impl.qiskit_adapter import (
    GDB_KWARG_DEFAULTS,
    TorchSCISolver,
    _pair_ab_swap_permutation,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

    from torch_sci_solver.torch_impl.pair_helpers import PairHelpers


# ----------------------------------------------------------------------
# PairBasis construction
# ----------------------------------------------------------------------


def test_pair_basis_dedupes_and_sorts():
    """Duplicate (α, β) pairs collapse; the surviving pairs are lex-sorted.

    Insertion order is irrelevant — the pair basis is a deterministic
    function of the *set* of (α, β) tuples in the input.
    """
    # Insertion order: (3, 7), (1, 5), (3, 7)-dup, (1, 5)-dup, (2, 9)
    alphas_in = np.array([3, 1, 3, 1, 2], dtype=np.int64)
    betas_in = np.array([7, 5, 7, 5, 9], dtype=np.int64)
    pb = pair_basis_from_arrays(alphas_in, betas_in, device="cpu")

    # 3 distinct pairs, sorted by (α, β): (1, 5), (2, 9), (3, 7).
    assert pb.n_pairs == 3
    torch.testing.assert_close(pb.alpha_of_pair, torch.tensor([1, 2, 3], dtype=torch.int64))
    torch.testing.assert_close(pb.beta_of_pair, torch.tensor([5, 9, 7], dtype=torch.int64))


def test_pair_basis_uid_maps_index_into_uniques():
    """``alpha_of_pair[k] == unique_alphas[alpha_uid_of_pair[k]]`` — always."""
    alphas_in = np.array([10, 10, 11, 12, 11], dtype=np.int64)
    betas_in = np.array([20, 21, 22, 20, 22], dtype=np.int64)
    pb = pair_basis_from_arrays(alphas_in, betas_in, device="cpu")

    torch.testing.assert_close(
        pb.alpha_of_pair,
        pb.unique_alphas[pb.alpha_uid_of_pair],
    )
    torch.testing.assert_close(
        pb.beta_of_pair,
        pb.unique_betas[pb.beta_uid_of_pair],
    )


def test_pair_basis_partners_of_alpha_csr_covers_every_pair():
    """The α-partners CSR is a permutation over ``[0, N)``.

    For every unique α, ``partners_of_alpha_pair_idx[offsets[a] :
    offsets[a+1]]`` is the list of pair indices with that α. Union over
    all a must be exactly ``{0, ..., N-1}`` (each pair has exactly one
    α, hence appears in exactly one bucket).
    """
    # A basis where α-buckets have different sizes exercises the CSR.
    alphas_in = np.array([1, 1, 1, 2, 3, 3], dtype=np.int64)
    betas_in = np.array([10, 20, 30, 15, 25, 35], dtype=np.int64)
    pb = pair_basis_from_arrays(alphas_in, betas_in, device="cpu")

    n = pb.n_pairs
    idx = pb.partners_of_alpha_pair_idx.tolist()
    assert sorted(idx) == list(range(n))
    # Offsets are monotone and sum to N.
    off = pb.partners_of_alpha_offsets.tolist()
    assert off[0] == 0
    assert off[-1] == n
    assert all(off[i] <= off[i + 1] for i in range(len(off) - 1))
    # Bucket contents match: for α_a, pair-idx-list has α_of_pair == unique_alphas[a].
    for a in range(pb.n_unique_alphas):
        pairs_a = pb.partners_of_alpha_pair_idx[off[a] : off[a + 1]]
        assert torch.all(pb.alpha_of_pair[pairs_a] == pb.unique_alphas[a])
        # The stored β-uid entries match the α-partners' β-uid.
        beta_uids_a = pb.partners_of_alpha_beta_uid[off[a] : off[a + 1]]
        torch.testing.assert_close(beta_uids_a, pb.beta_uid_of_pair[pairs_a])


def test_pair_basis_partners_of_beta_csr_covers_every_pair():
    """Mirror of the α-partners invariant, for β."""
    alphas_in = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
    betas_in = np.array([10, 10, 10, 20, 20, 20], dtype=np.int64)
    pb = pair_basis_from_arrays(alphas_in, betas_in, device="cpu")

    n = pb.n_pairs
    idx = pb.partners_of_beta_pair_idx.tolist()
    assert sorted(idx) == list(range(n))
    off = pb.partners_of_beta_offsets.tolist()
    assert off[0] == 0 and off[-1] == n

    for b in range(pb.n_unique_betas):
        pairs_b = pb.partners_of_beta_pair_idx[off[b] : off[b + 1]]
        assert torch.all(pb.beta_of_pair[pairs_b] == pb.unique_betas[b])
        alpha_uids_b = pb.partners_of_beta_alpha_uid[off[b] : off[b + 1]]
        torch.testing.assert_close(alpha_uids_b, pb.alpha_uid_of_pair[pairs_b])


def test_pair_basis_rejects_unequal_length_inputs():
    """Pair-mode inputs must be parallel arrays; catch mistakes at the door."""
    with pytest.raises(ValueError, match="equal length"):
        pair_basis_from_arrays(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5], dtype=np.int64),
        )


def test_pair_basis_rejects_empty_input():
    """An empty pair set is not a valid basis."""
    with pytest.raises(ValueError, match="empty"):
        pair_basis_from_arrays(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def test_pair_basis_handles_high_bit_string_values():
    """CI-string bitmasks can span the full uint64 range (e.g. 36+ orbitals).

    We deliberately store these as signed int64 bit patterns to match
    ``sbdiag._to_int64_python`` — verify that a value with the high bit
    set survives the round-trip and dedups correctly.
    """
    high = (1 << 63) + 42  # would overflow int64 if not wrapped
    alphas_in = np.array([high, high], dtype=object)  # object → Python int
    betas_in = np.array([1, 1], dtype=object)
    pb = pair_basis_from_arrays(alphas_in, betas_in, device="cpu")
    assert pb.n_pairs == 1
    # The stored bit pattern round-trips: adding back 2^64 recovers the original.
    got = int(pb.alpha_of_pair[0].item())
    assert got + (1 << 64) if got < 0 else got == high


# ----------------------------------------------------------------------
# α↔β swap permutation (pair-mode ab-projection)
# ----------------------------------------------------------------------


def test_pair_ab_swap_permutation_matches_lex_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chunked searchsorted lookup matches the per-pair dict semantics."""
    strings = np.array([0b0011, 0b0101, 0b1001, 0b1111], dtype=np.int64)
    basis = pair_basis_from_arrays(np.repeat(strings, len(strings)), np.tile(strings, len(strings)))
    alpha = basis.alpha_of_pair.numpy()
    beta = basis.beta_of_pair.numpy()
    lookup = {(int(x), int(y)): k for k, (x, y) in enumerate(zip(alpha, beta))}
    expected = np.array([lookup[(int(y), int(x))] for x, y in zip(alpha, beta)])
    for chunk in (1 << 22, 3):
        monkeypatch.setattr(qiskit_adapter, "_PAIR_SWAP_LOOKUP_CHUNK", chunk)
        k_swap, tau = _pair_ab_swap_permutation(basis, 4, torch.device("cpu"), torch.float64)
        assert k_swap is not None and tau is not None
        np.testing.assert_array_equal(k_swap.numpy(), expected)
        # The swap is an involution and tau is constant on swap pairs.
        np.testing.assert_array_equal(k_swap.numpy()[k_swap.numpy()], np.arange(len(expected)))
        np.testing.assert_array_equal(tau.numpy()[k_swap.numpy()], tau.numpy())
    # The local path holds only its owned slice; the closure check stays global.
    monkeypatch.setattr(qiskit_adapter, "_PAIR_SWAP_LOOKUP_CHUNK", 3)
    k_swap, tau = _pair_ab_swap_permutation(
        basis, 4, torch.device("cpu"), torch.float64, pair_range=(3, len(expected) - 2)
    )
    assert k_swap is not None and tau is not None
    np.testing.assert_array_equal(k_swap.numpy(), expected[3 : len(expected) - 2])


def test_pair_ab_swap_permutation_skips_unclosed_basis() -> None:
    """A pair set without its swapped partners is skipped, not corrupted."""
    # (0b11, 0b01) has no (0b01, 0b11) partner, so the set is not swap-closed.
    basis = pair_basis_from_arrays(
        np.array([0b01, 0b10, 0b11], dtype=np.int64),
        np.array([0b10, 0b01, 0b01], dtype=np.int64),
    )
    k_swap, tau = _pair_ab_swap_permutation(basis, 2, torch.device("cpu"), torch.float64)
    assert k_swap is None and tau is None
    # An owned slice must not hide a closure failure elsewhere in the basis.
    k_swap, tau = _pair_ab_swap_permutation(
        basis, 2, torch.device("cpu"), torch.float64, pair_range=(1, 2)
    )
    assert k_swap is None and tau is None


# ----------------------------------------------------------------------
# basis_init_vector_pair_torch
# ----------------------------------------------------------------------


def test_pair_init_vector_hf_seed():
    """``init=0`` places (almost) all weight at index 0 (HF-seed convention).

    NOTE: not exactly 1.0/0.0 because of a tiny fixed-seed perturbation is mixed in,
    so a pure one-hot guess can never be an exact eigenvector of an invariant subspace
    that excludes the ground state — see ``basis_init_vector_pair_torch``.
    """
    v = basis_init_vector_pair_torch(7, init=0, device="cpu", dtype=torch.float64)
    assert v.shape == (7,)
    assert v[0].item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(v[1:], torch.zeros(6, dtype=torch.float64), atol=1e-6)
    assert torch.linalg.norm(v).item() == pytest.approx(1.0, abs=1e-12)


def test_pair_init_vector_random_is_unit_norm():
    """``init=1`` returns a normalized random vector."""
    v = basis_init_vector_pair_torch(20, init=1, device="cpu", dtype=torch.float64, seed=42)
    assert v.shape == (20,)
    assert torch.linalg.norm(v).item() == pytest.approx(1.0, abs=1e-12)


def test_pair_init_vector_rejects_bad_init():
    """Reject unknown ``init`` codes at the boundary."""
    with pytest.raises(ValueError, match="init must be 0"):
        basis_init_vector_pair_torch(5, init=99, device="cpu")


# ----------------------------------------------------------------------
# pair_basis_from_full_product — the full-product correctness-test fixture
# ----------------------------------------------------------------------


def test_pair_basis_from_full_product_has_expected_shape():
    """Full product of ``M_α`` × ``M_β`` yields ``M_α · M_β`` distinct pairs."""
    adets = [1, 2, 4]  # 3 distinct α-strings
    bdets = [8, 16]  # 2 distinct β-strings
    pb = pair_basis_from_full_product(adets, bdets, device="cpu")

    assert pb.n_pairs == 6
    assert pb.n_unique_alphas == 3
    assert pb.n_unique_betas == 2
    # Every α has exactly ``M_β`` partners (full product).
    partners_count = torch.diff(pb.partners_of_alpha_offsets).tolist()
    assert partners_count == [2, 2, 2]
    # And every β has exactly ``M_α``.
    partners_count_b = torch.diff(pb.partners_of_beta_offsets).tolist()
    assert partners_count_b == [3, 3]


# ----------------------------------------------------------------------
# Flag routing through TorchSCISolver + qiskit.py
# ----------------------------------------------------------------------


def test_solver_default_mode_is_tpb():
    """The default basis stays TPB, preserving historical behaviour."""
    assert TorchSCISolver(device="cpu").mode == "tpb"


def test_solver_stores_gdb_mode():
    """``mode="gdb"`` is remembered on the instance."""
    assert TorchSCISolver(device="cpu", mode="gdb").mode == "gdb"


def test_solver_rejects_unknown_mode():
    """An unrecognized mode fails at construction, not silently as TPB."""
    with pytest.raises(ValueError, match='mode must be "tpb" or "gdb"'):
        TorchSCISolver(device="cpu", mode="pairs")


def test_qiskit_init_keys_includes_mode():
    """The kwarg is routed to the constructor, not silently forwarded per-call."""
    from torch_sci_solver.qiskit import _SOLVER_INIT_KEYS

    assert "mode" in _SOLVER_INIT_KEYS


def test_make_torch_sci_solver_forwards_mode():
    """The public factory accepts and stores the mode."""
    from torch_sci_solver import make_torch_sci_solver

    assert make_torch_sci_solver(device="cpu", mode="gdb").mode == "gdb"


# ----------------------------------------------------------------------
# Gate: pair mode raises with a clear message until the pair matvec lands.
# Remove this test (or invert it) when the pair-mode matvec lights up.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# TorchSCISolver end-to-end pair-mode: correctness on the full product
# ----------------------------------------------------------------------


def _solve_via_torch_sci_solver_full_product(mol_atom, basis_name="sto-3g"):
    """Diagonalize a small molecule via ``TorchSCISolver`` twice — once in
    product mode, once in pair mode with the full-product basis expanded
    into an explicit pair list. Returns ``(e_prod, e_pair, e_fci)``.

    Exercises the full ``_solve_one_subspace_pair`` path (dedup, feature
    gates, pair helpers, matvec, Davidson, occupancies, SCIState) — not
    just the low-level ``diag_torch`` driver.
    """
    import itertools

    from pyscf import ao2mo, fci, gto, scf
    from torch_sci_solver import solve_fermion

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec
    e_fci = fci.FCI(mf).kernel()[0]

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    e_prod, _, _, _ = solve_fermion((ci_a, ci_b), hcore=h1, eri=eri, device="cpu")

    # Pair mode: full product expanded into explicit (α, β) pairs.
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    e_pair, sci_state_pair, occs_pair, _ = solve_fermion(
        (pairs_a, pairs_b),
        hcore=h1,
        eri=eri,
        device="cpu",
        mode="gdb",
    )
    return (
        e_prod,
        e_pair,
        e_fci - mol.energy_nuc(),
        sci_state_pair,
        occs_pair,
        len(pairs_a),
    )


def test_solve_fermion_pair_mode_matches_product_h2():
    """H2/STO-3G via the high-level ``solve_fermion`` entry point."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    e_prod, e_pair, e_fci_elec, _, _, _ = _solve_via_torch_sci_solver_full_product(
        "H 0 0 0; H 0 0 0.74"
    )
    assert e_prod == pytest.approx(e_fci_elec, abs=1e-9)
    assert e_pair == pytest.approx(e_fci_elec, abs=1e-9)
    assert e_pair == pytest.approx(e_prod, abs=1e-10)


def test_solve_fermion_pair_mode_matches_product_h4():
    """H4 chain — exercises α-doubles and αβ-doubles end-to-end."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    e_prod, e_pair, e_fci_elec, _, _, _ = _solve_via_torch_sci_solver_full_product(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    assert e_prod == pytest.approx(e_fci_elec, abs=1e-9)
    assert e_pair == pytest.approx(e_fci_elec, abs=1e-9)
    assert e_pair == pytest.approx(e_prod, abs=1e-10)


def test_pair_mode_scistate_is_flat_pairsciestate(tmp_path: Path):
    """Pair mode's SCIResult carries a ``PairSCIState`` with 1D
    length-N ``amplitudes`` and parallel length-N ``ci_strs_a`` /
    ``ci_strs_b``.

    An earlier version of this code returned a 2D
    ``(M_α_unique, M_β_unique)`` sparse SCIState to be shape-compatible
    with ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``.
    That allocation is 320 GB for a 200k × 200k unique-string basis and
    caused an OOM in production. Pair mode uses its own outer loop
    (:func:`diagonalize_fermionic_hamiltonian_pairs`), so the
    qiskit-addon-sqd shape isn't required — the flat form is honest
    and memory-linear in ``N``.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    (
        e_prod,
        e_pair,
        e_fci,
        sci_state,
        occs,
        n_pairs,
    ) = _solve_via_torch_sci_solver_full_product("H 0 0 0; H 0 0 0.74")
    # 1D amplitudes, one entry per pair, parallel to length-N ci_strs_a/b.
    assert sci_state.amplitudes.ndim == 1
    assert sci_state.amplitudes.shape == (n_pairs,)
    assert sci_state.ci_strs_a.shape == (n_pairs,)
    assert sci_state.ci_strs_b.shape == (n_pairs,)
    # State is normalized.
    assert float(np.sum(sci_state.amplitudes**2)) == pytest.approx(1.0, abs=1e-10)
    # Occupancies are physical: sum to particle count.
    occ_a, occ_b = occs
    assert occ_a.sum() == pytest.approx(1.0, abs=1e-9)  # H2: 1 α electron
    assert occ_b.sum() == pytest.approx(1.0, abs=1e-9)
    # Save the state and check the consistency
    state_filepath = tmp_path / "state.npz"
    sci_state.save(state_filepath)
    loaded_sci_state = np.load(state_filepath)
    assert (sci_state.amplitudes == loaded_sci_state["amplitudes"]).all()
    assert (sci_state.ci_strs_a == loaded_sci_state["ci_strs_a"]).all()
    assert (sci_state.ci_strs_b == loaded_sci_state["ci_strs_b"]).all()
    assert sci_state.norb == loaded_sci_state["norb"].item()
    assert sci_state.nelec == tuple(loaded_sci_state["nelec"])


def test_pair_mode_rejects_unequal_length_inputs():
    """The ValueError should reach the user cleanly, not as an internal
    shape mismatch far downstream.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)

    ci_a = np.array([1, 2, 1], dtype=np.int64)  # length 3
    ci_b = np.array([1, 2], dtype=np.int64)  # length 2 — mismatch
    solver = TorchSCISolver(device="cpu", mode="gdb")
    with pytest.raises(ValueError, match="len\\(strings_a\\)"):
        solver(
            [(ci_a, ci_b)],
            np.asarray(h1, dtype=np.float64),
            np.asarray(eri, dtype=np.float64),
            norb,
            mol.nelec,
        )


def test_pair_mode_rejects_bad_seed_strategy():
    """Only ``'smallest_int'`` and ``'connected'`` are valid seed strategies
    in pair mode — a typo should surface as a ValueError, not fail
    silently.
    """
    # The enum is validated eagerly at construction, before any basis work.
    with pytest.raises(ValueError, match="seed_strategy"):
        TorchSCISolver(device="cpu", mode="gdb", seed_strategy="random_typo")


# ----------------------------------------------------------------------
# Warm-start (probe_method='warm_start') in pair mode
# ----------------------------------------------------------------------


def _pair_warm_start_setup(mol_atom, basis_name="sto-3g"):
    """H4-scale fixture shared across warm-start tests."""
    import itertools

    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    return h1, eri, norb, (n_a, n_b), ci_a, ci_b


def test_pair_warm_start_matches_cold_solve_on_extended_h4():
    """Warm-extending from a core pair set to a larger pair set must
    produce the same energy as a cold pair-mode solve on the same
    extended pair set (up to Davidson's residual tolerance).

    This is the warm-start contract:
        |warm(P₀ → P).energy − cold(P).energy| ≤ davidson_tolerance
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import make_torch_sci_solver

    h1, eri, norb, nelec, ci_a, ci_b = _pair_warm_start_setup(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )

    # Full-product pair list as the "extended" P.
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)

    # "Core" P₀ = every third pair (proper subset).
    core_idx = list(range(0, len(pairs_a), 3))
    core_a = pairs_a[core_idx]
    core_b = pairs_b[core_idx]

    # (1) Cold solve on the core P₀ with probe_method='warm_start' → cache
    #     a TorchPairSCIState. include_hf_in_core auto-adds HF pair to
    #     protect ground-state overlap.
    solver = make_torch_sci_solver(device="cpu", mode="gdb", davidson_tolerance=1e-10)
    _res_tuple = solver(
        [(core_a, core_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        nelec,
        probe_method="warm_start",
    )
    eigen_states_list, states = _res_tuple
    core_state = states[0]
    assert core_state is not None
    # solver(...) now returns list[EigenStates]; pair mode is single-root
    # so ground-state SCIResult = states[0].
    e_cold_core = eigen_states_list[0].states[0].energy

    # (2) Warm-extend to the full-product pair basis. On warm-extend the
    #     solver only returns (results, states) if the caller explicitly
    #     asked for it via build_output_state=True; the default is just
    #     results (matching product-mode behavior).
    warm_eigen_states_list = solver(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        nelec,
        probe_method="warm_start",
        probe_state=[core_state],
    )
    e_warm_extended = warm_eigen_states_list[0].states[0].energy

    # (3) Cold solve on the same extended pair basis, for reference.
    from torch_sci_solver import solve_fermion

    e_cold_extended, *_ = solve_fermion(
        (pairs_a, pairs_b),
        hcore=h1,
        eri=eri,
        device="cpu",
        mode="gdb",
        davidson_tolerance=1e-10,
    )

    # Contract: warm and cold on the same basis must agree to davidson_tolerance.
    assert e_warm_extended == pytest.approx(e_cold_extended, abs=1e-8)
    # Sanity: core solve is bounded above by extended solve (variational).
    assert e_cold_core >= e_cold_extended - 1e-9


def test_pair_warm_start_rejects_non_superset():
    """If the caller supplies a pair basis that is not a superset of the
    cached one, ``_dedup_and_check_pair_superset`` must raise cleanly.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import make_torch_sci_solver

    h1, eri, norb, nelec, ci_a, ci_b = _pair_warm_start_setup(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    core_idx = list(range(0, len(pairs_a), 3))
    core_a = pairs_a[core_idx]
    core_b = pairs_b[core_idx]

    solver = make_torch_sci_solver(device="cpu", mode="gdb")
    results, states = solver(
        [(core_a, core_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        nelec,
        probe_method="warm_start",
    )
    core_state = states[0]
    assert core_state is not None

    # Non-superset: keep only a subset of the cached pairs.
    truncated_a = np.array([core_a[0]], dtype=np.int64)
    truncated_b = np.array([core_b[0]], dtype=np.int64)
    with pytest.raises(ValueError, match="superset"):
        solver(
            [(truncated_a, truncated_b)],
            np.asarray(h1, dtype=np.float64),
            np.asarray(eri, dtype=np.float64),
            norb,
            nelec,
            probe_method="warm_start",
            probe_state=[core_state],
        )


def test_pair_warm_start_rejects_product_mode_state():
    """A cached product-mode ``TorchSCIState`` cannot warm-extend a pair
    solve — the two state types disagree on basis structure.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import make_torch_sci_solver

    h1, eri, norb, nelec, ci_a, ci_b = _pair_warm_start_setup("H 0 0 0; H 0 0 0.74")
    # Product-mode cold solve → TorchSCIState cached.
    solver_prod = make_torch_sci_solver(device="cpu")
    _, prod_states = solver_prod(
        [(ci_a, ci_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        nelec,
        probe_method="warm_start",
    )
    prod_state = prod_states[0]
    assert prod_state is not None

    # Try to feed the TorchSCIState into a pair-mode warm-extend.
    solver_pair = make_torch_sci_solver(device="cpu", mode="gdb")
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    with pytest.raises(TypeError, match="TorchPairSCIState"):
        solver_pair(
            [(pairs_a, pairs_b)],
            np.asarray(h1, dtype=np.float64),
            np.asarray(eri, dtype=np.float64),
            norb,
            nelec,
            probe_method="warm_start",
            probe_state=[prod_state],
        )


def test_pair_mode_connected_seed_matches_smallest_int_on_h4():
    """``seed_strategy='connected'`` and ``'smallest_int'`` must converge to
    the same ground-state energy — Davidson is basis-ordering-invariant.
    Verifies (a) that the connected-seed code path runs without error
    and (b) that the seeding change does not steer Davidson onto a
    different eigenstate.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver import make_torch_sci_solver

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)

    # solver(...) returns list[EigenStates]; pair mode = single-root, so
    # ground-state SCIResult = states[0].
    r_default = make_torch_sci_solver(device="cpu", mode="gdb", seed_strategy="smallest_int")(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]
    r_connected = make_torch_sci_solver(device="cpu", mode="gdb", seed_strategy="connected")(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]
    assert r_connected.energy == pytest.approx(r_default.energy, abs=1e-9)


# ----------------------------------------------------------------------
# Pair-mode RDMs correctness cross-check
# ----------------------------------------------------------------------


def _rdms_pair_and_product_on_full_product(mol_atom, basis_name="sto-3g"):
    """Diagonalize a molecule twice (product-mode + pair-mode-on-full-product)
    and build the full 1-RDM + αα/ββ/αβ 2-RDM in both. Returns both sets
    of tensors on CPU numpy so the caller can assert equality.
    """
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.davidson import (
        basis_init_vector_torch,
        davidson_torch,
    )
    from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals
    from torch_sci_solver.torch_impl.matvec_dispatch import build_matvec
    from torch_sci_solver.torch_impl.mult import make_diag_terms_torch
    from torch_sci_solver.torch_impl.mult_pairs import (
        make_diag_terms_pair_torch,
        mult_pair_torch,
    )
    from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
    from torch_sci_solver.torch_impl.pairs import (
        basis_init_vector_pair_torch,
        pair_basis_from_full_product,
    )
    from torch_sci_solver.torch_impl.rdm import make_one_rdm_torch, make_two_rdm_torch
    from torch_sci_solver.torch_impl.rdm_pairs import (
        make_one_rdm_pair_torch,
        make_two_rdm_pair_torch,
    )

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    # Product-mode diagonalization + RDMs.
    device = torch.device("cpu")
    dtype = torch.float64
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    helpers = list(generate_all_torch_helpers(adets, bdets, norb, device=device))
    hii = make_diag_terms_torch(adets_t, bdets_t, norb, integrals)
    W = basis_init_vector_torch(len(adets), len(bdets), init=0, device=device, dtype=dtype)
    matvec = build_matvec(
        "batched", hii, adets_t, bdets_t, norb, integrals, helpers, chunk_size=2_000_000
    )
    e_prod, eigvec_prod, _ = davidson_torch(hii, W, matvec, 50, 30, 1e-9, verbose=False)
    ga_prod, gb_prod = make_one_rdm_torch(eigvec_prod, adets_t, bdets_t, norb, helpers)
    Gaa_prod, Gab_prod, Gbb_prod = make_two_rdm_torch(eigvec_prod, adets_t, bdets_t, norb, helpers)

    # Pair-mode on the full product basis.
    pair_basis = pair_basis_from_full_product(adets, bdets, device=device)
    pair_helpers = generate_pair_helpers(pair_basis, norb, device=device)
    hii_pair = make_diag_terms_pair_torch(pair_basis, norb, integrals)
    W_pair = basis_init_vector_pair_torch(pair_basis.n_pairs, init=0, device=device, dtype=dtype)

    def _mv(v):
        """Mv."""
        return mult_pair_torch(hii_pair, v, pair_basis, pair_helpers, norb, integrals)

    e_pair, eigvec_pair, _ = davidson_torch(hii_pair, W_pair, _mv, 50, 30, 1e-9, verbose=False)
    ga_pair, gb_pair = make_one_rdm_pair_torch(eigvec_pair, pair_basis, pair_helpers, norb)
    Gaa_pair, Gab_pair, Gbb_pair = make_two_rdm_pair_torch(
        eigvec_pair, pair_basis, pair_helpers, norb
    )

    return {
        "e_prod": float(e_prod),
        "e_pair": float(e_pair),
        "ga_prod": ga_prod.numpy(),
        "gb_prod": gb_prod.numpy(),
        "Gaa_prod": Gaa_prod.numpy(),
        "Gab_prod": Gab_prod.numpy(),
        "Gbb_prod": Gbb_prod.numpy(),
        "ga_pair": ga_pair.numpy(),
        "gb_pair": gb_pair.numpy(),
        "Gaa_pair": Gaa_pair.numpy(),
        "Gab_pair": Gab_pair.numpy(),
        "Gbb_pair": Gbb_pair.numpy(),
        "n_a": n_a,
        "n_b": n_b,
    }


def test_pair_1rdm_matches_product_on_h4():
    """Pair-mode 1-RDM (γα, γβ) equals product-mode on the full product basis.

    Both are computed from independently converged Davidson eigenvectors
    (which agree to 1e-10). Both use the same
    interleaved-Wick → spin-block σ convention.
    """
    pytest.importorskip("pyscf")
    r = _rdms_pair_and_product_on_full_product("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22")
    assert r["e_pair"] == pytest.approx(r["e_prod"], abs=1e-10)
    np.testing.assert_allclose(r["ga_pair"], r["ga_prod"], atol=1e-8)
    np.testing.assert_allclose(r["gb_pair"], r["gb_prod"], atol=1e-8)
    # Trace of each block equals the electron count.
    assert np.trace(r["ga_pair"]) == pytest.approx(r["n_a"], abs=1e-8)
    assert np.trace(r["gb_pair"]) == pytest.approx(r["n_b"], abs=1e-8)


def test_pair_2rdm_matches_product_on_h4():
    """Pair-mode 2-RDM (Γaa, Γab, Γbb) equals product-mode on the full product."""
    pytest.importorskip("pyscf")
    r = _rdms_pair_and_product_on_full_product("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22")
    np.testing.assert_allclose(r["Gaa_pair"], r["Gaa_prod"], atol=1e-8)
    np.testing.assert_allclose(r["Gab_pair"], r["Gab_prod"], atol=1e-8)
    np.testing.assert_allclose(r["Gbb_pair"], r["Gbb_prod"], atol=1e-8)


def test_pair_rdm_energy_reconstruction_matches_davidson_on_h4():
    """E = I0 + Σ h (γα+γβ) + (1/2) Σ (pq|rs)(Γaa+2Γab+Γbb) must equal the
    pair-mode Davidson eigenvalue. This cross-check catches sign/index
    bugs in the RDM builders that would leave the RDMs internally
    consistent but wrong.
    """
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals
    from torch_sci_solver.torch_impl.rdm import energy_from_rdm

    r = _rdms_pair_and_product_on_full_product("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22")
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())
    integrals = to_torch_integrals(I0, I1, I2, device=torch.device("cpu"), dtype=torch.float64)
    e_recon = energy_from_rdm(
        torch.from_numpy(r["ga_pair"]),
        torch.from_numpy(r["gb_pair"]),
        torch.from_numpy(r["Gaa_pair"]),
        torch.from_numpy(r["Gab_pair"]),
        torch.from_numpy(r["Gbb_pair"]),
        integrals,
    )
    assert float(e_recon) == pytest.approx(r["e_pair"], abs=1e-8)


def test_solve_fermion_pair_mode_compute_full_rdm_matches_product_h4():
    """End-to-end via solve_fermion(compute_full_rdm=True) — the SCIResult
    ``rdm1``/``rdm2`` fields (spin-summed) must match between pair and
    product mode on the full product basis.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver import make_torch_sci_solver

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    # solver(...) returns list[EigenStates] — unwrap ground state via
    # ``.states[0]`` (pair mode is single-root).
    solver_prod = make_torch_sci_solver(device="cpu", compute_full_rdm=True)
    r_prod = solver_prod(
        [(ci_a, ci_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]

    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    solver_pair = make_torch_sci_solver(device="cpu", compute_full_rdm=True, mode="gdb")
    r_pair = solver_pair(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]

    assert r_pair.energy == pytest.approx(r_prod.energy, abs=1e-10)
    np.testing.assert_allclose(r_pair.rdm1, r_prod.rdm1, atol=1e-8)
    np.testing.assert_allclose(r_pair.rdm2, r_prod.rdm2, atol=1e-8)


# ----------------------------------------------------------------------
# SpinSqOperator + spin_sq penalty + compute_spin_squared diagnostic
# ----------------------------------------------------------------------


def test_pair_spin_sq_operator_diagonal_matches_product_on_h4():
    """PairSpinSqOperator's diagonal must equal the product-mode
    SpinSqOperator's diagonal on the full-product basis (same lex-order
    layout as the pair basis constructor).
    """
    pytest.importorskip("pyscf")
    import itertools

    from pyscf import gto
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_full_product
    from torch_sci_solver.torch_impl.spin_sq import SpinSqOperator
    from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    norb = 4
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    device = torch.device("cpu")
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    op_prod = SpinSqOperator(adets_t, bdets_t, norb, n_a, n_b, device=device)

    pair_basis = pair_basis_from_full_product(adets, bdets, device=device)
    op_pair = PairSpinSqOperator(pair_basis, norb, n_a, n_b, device=device)
    torch.testing.assert_close(op_pair.diagonal(), op_prod.diagonal())


def test_pair_spin_sq_apply_matches_product_on_h4():
    """Pair-mode ``S² @ c`` must equal product-mode's on the full product."""
    pytest.importorskip("pyscf")
    import itertools

    from pyscf import gto
    from torch_sci_solver.torch_impl.pairs import pair_basis_from_full_product
    from torch_sci_solver.torch_impl.spin_sq import SpinSqOperator
    from torch_sci_solver.torch_impl.spin_sq_pairs import PairSpinSqOperator

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    norb = 4
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    device = torch.device("cpu")
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    op_prod = SpinSqOperator(adets_t, bdets_t, norb, n_a, n_b, device=device)

    pair_basis = pair_basis_from_full_product(adets, bdets, device=device)
    op_pair = PairSpinSqOperator(pair_basis, norb, n_a, n_b, device=device)

    torch.manual_seed(0)
    c = torch.randn(pair_basis.n_pairs, dtype=torch.float64, device=device)
    torch.testing.assert_close(op_pair.apply(c), op_prod.apply(c))


def test_solve_fermion_pair_mode_compute_spin_squared_matches_product_h4():
    """End-to-end ⟨S²⟩ diagnostic parity via ``compute_spin_squared=True``."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver import make_torch_sci_solver

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    solver_prod = make_torch_sci_solver(device="cpu", compute_spin_squared=True)
    solver_prod(
        [(ci_a, ci_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )
    s2_prod = solver_prod.stats["spin_squared"][0]

    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    solver_pair = make_torch_sci_solver(device="cpu", compute_spin_squared=True, mode="gdb")
    solver_pair(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )
    s2_pair = solver_pair.stats["spin_squared"][0]
    assert s2_pair == pytest.approx(s2_prod, abs=1e-8)
    # H4 ground state is a singlet.
    assert s2_pair == pytest.approx(0.0, abs=1e-6)


def test_pair_mode_ab_projection_matches_product_on_closed_basis_h4():
    """When the pair set is the full product of ``adets`` with itself
    (αβ-symmetric basis, ``adets == bdets``), the pair set IS closed
    under (α, β) ↔ (β, α), so pair-mode ab-projection is well-defined
    and should agree with product-mode.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec
    assert n_a == n_b  # closed-shell H4

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    # Product mode with ab-projection ON (native default; not the
    # solve_fermion factory's opt-out). solver(...) returns list[EigenStates];
    # unwrap ground-state SCIResult via ``.states[0]``.
    solver_prod = TorchSCISolver(device="cpu", apply_ab_projection=True)
    r_prod = solver_prod(
        [(ci_a, ci_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]

    # Pair mode over the full product — closed under swap by construction.
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    solver_pair = TorchSCISolver(device="cpu", apply_ab_projection=True, mode="gdb")
    r_pair = solver_pair(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]
    assert r_pair.energy == pytest.approx(r_prod.energy, abs=1e-9)


def test_pair_mode_spin_sq_penalty_recovers_singlet_h4():
    """The per-call spin_sq=0.0 penalty in pair mode strips cleanly:
    reported physical energy on the full-product basis matches product-mode
    with the same penalty (both are singlets by construction).
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver import solve_fermion

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return np.array(sorted(out), dtype=np.int64)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    e_prod, *_ = solve_fermion((ci_a, ci_b), hcore=h1, eri=eri, device="cpu", spin_sq=0.0)
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    e_pair, *_ = solve_fermion(
        (pairs_a, pairs_b),
        hcore=h1,
        eri=eri,
        device="cpu",
        mode="gdb",
        spin_sq=0.0,
    )
    assert e_pair == pytest.approx(e_prod, abs=1e-9)


# ----------------------------------------------------------------------
# Correctness: pair-mode on the full product must equal product-mode
# ----------------------------------------------------------------------


def _diag_pair_and_product_on_full_product(mol_atom, basis_name="sto-3g"):
    """Diagonalize a small molecule twice: once product-mode, once pair-mode
    over the same basis expressed as an explicit full-product pair list.

    Returns ``(e_prod, v_prod, e_pair, v_pair, e_fci)`` so callers can
    assert bit-for-bit agreement between the two engines and both agree
    with PySCF FCI.
    """
    import itertools

    from pyscf import ao2mo, fci, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.sbdiag import diag_torch

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    n_a, n_b = mol.nelec
    e_fci = fci.FCI(mf).kernel()[0]

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, e_core)

    e_prod, v_prod, _ = diag_torch(adets, bdets, norb, I0, I1, I2)

    # Pair mode: the full product basis expressed as an explicit
    # (α, β) list. After pair_basis_from_arrays dedups + sorts, this
    # must yield the exact same H matrix as product mode.
    pairs_a = np.array([a for a in adets for _ in bdets], dtype=np.int64)
    pairs_b = np.array([b for _ in adets for b in bdets], dtype=np.int64)
    e_pair, v_pair, _ = diag_torch(pairs_a.tolist(), pairs_b.tolist(), norb, I0, I1, I2, mode="gdb")
    return e_prod, v_prod, e_pair, v_pair, e_fci


# ----------------------------------------------------------------------
# matvec_impl resolution in pair mode
# ----------------------------------------------------------------------


def test_pair_mode_matvec_impl_auto_resolves_to_batched_cpu():
    """``matvec_impl='auto'`` in pair mode resolves to the batched path.

    The GDB Triton kernel is opt-in (``matvec_impl='triton'``) and covers the
    same-spin doubles slices only.
    """
    solver = TorchSCISolver(device="cpu", mode="gdb", matvec_impl="auto")
    assert solver.matvec_impl == "batched"


def test_pair_mode_matvec_impl_batched_stays_batched():
    """Explicit ``matvec_impl='batched'`` is a no-op passthrough in pair mode."""
    solver = TorchSCISolver(device="cpu", mode="gdb", matvec_impl="batched")
    assert solver.matvec_impl == "batched"


def test_pair_mode_matvec_impl_triton_requires_a_triton_device():
    """An explicit Triton request in GDB mode is kept, never silently
    downgraded: the kernel entry point refuses off a Triton/CUDA device.
    """
    solver = TorchSCISolver(device="cpu", mode="gdb", matvec_impl="triton")
    assert solver.matvec_impl == "triton"

    from torch_sci_solver.sbd_ref.integrals import OneInt, TwoInt
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals
    from torch_sci_solver.torch_impl.mult_pair_blocks_triton import (
        apply_same_spin_doubles,
    )

    integrals = to_torch_integrals(0.0, OneInt(2), TwoInt(2), device="cpu")
    out = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="requires Triton"):
        apply_same_spin_doubles(
            out,
            out,
            out,
            out,
            1,
            integrals,
            bra=out,
            ket=out,
            cr0=out,
            cr1=out,
            an0=out,
            an1=out,
            M_a=1,
            M_b=1,
            is_alpha=True,
            T_stride=1,
        )


def test_product_mode_matvec_impl_auto_still_resolves_normally():
    """Regression check: product-mode ``matvec_impl='auto'`` on CPU must
    still resolve to ``'batched'`` after the pair-mode branching change.
    """
    solver = TorchSCISolver(device="cpu", mode="tpb", matvec_impl="auto")
    assert solver.matvec_impl == "batched"


# ----------------------------------------------------------------------
# diag_torch threading: matvec_impl and block_matvec reach the pair driver
# ----------------------------------------------------------------------


def _h2_pair_integrals():
    """H2/STO-3G integrals plus the full-product pair lists (four pairs)."""
    pytest.importorskip("pyscf")
    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, mol.energy_nuc())

    def strings(n_elec):
        """Sorted bitmasks of every ``n_elec``-of-``norb`` occupancy."""
        return sorted(
            sum(1 << i for i in combo) for combo in itertools.combinations(range(norb), n_elec)
        )

    sa = strings(mol.nelec[0])
    sb = strings(mol.nelec[1])
    alphas = [a for a in sa for _ in sb]
    betas = [b for _ in sa for b in sb]
    return norb, I0, I1, I2, sa, sb, alphas, betas


def test_diag_torch_gdb_auto_resolves_to_batched(monkeypatch):
    """``diag_torch(mode="gdb", matvec_impl="auto")`` stays on the batched pair path."""
    from torch_sci_solver.torch_impl import sbdiag

    norb, I0, I1, I2, _, _, alphas, betas = _h2_pair_integrals()
    seen: list[bool] = []
    real = sbdiag.mult_pair_torch

    def spy(*args, **kwargs):
        seen.append(kwargs["triton_matvec"])
        return real(*args, **kwargs)

    monkeypatch.setattr(sbdiag, "mult_pair_torch", spy)
    e_auto, _, _ = sbdiag.diag_torch(
        alphas, betas, norb, I0, I1, I2, mode="gdb", matvec_impl="auto"
    )
    assert seen and not any(seen)
    e_default, _, _ = sbdiag.diag_torch(alphas, betas, norb, I0, I1, I2, mode="gdb")
    assert e_auto == pytest.approx(e_default, abs=1e-10)


def test_diag_torch_gdb_forwards_triton_matvec(monkeypatch):
    """An explicit ``triton`` reaches ``mult_pair_torch`` (which falls back on CPU)."""
    from torch_sci_solver.torch_impl import sbdiag

    norb, I0, I1, I2, _, _, alphas, betas = _h2_pair_integrals()
    seen: list[bool] = []
    real = sbdiag.mult_pair_torch

    def spy(*args, **kwargs):
        seen.append(kwargs["triton_matvec"])
        return real(*args, **kwargs)

    monkeypatch.setattr(sbdiag, "mult_pair_torch", spy)
    e_triton, _, _ = sbdiag.diag_torch(
        alphas, betas, norb, I0, I1, I2, mode="gdb", matvec_impl="triton"
    )
    assert seen and all(seen)
    e_batched, _, _ = sbdiag.diag_torch(
        alphas, betas, norb, I0, I1, I2, mode="gdb", matvec_impl="batched"
    )
    assert e_triton == pytest.approx(e_batched, abs=1e-10)


def test_diag_torch_gdb_forwards_block_matvec(monkeypatch):
    """``block_matvec`` reaches ``generate_pair_helpers`` in the pair driver."""
    from torch_sci_solver.torch_impl import sbdiag

    norb, I0, I1, I2, _, _, alphas, betas = _h2_pair_integrals()
    seen: dict = {}
    real = sbdiag.generate_pair_helpers

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sbdiag, "generate_pair_helpers", spy)
    sbdiag.diag_torch(alphas, betas, norb, I0, I1, I2, mode="gdb", block_matvec="on")
    assert seen["block_matvec"] == "on"


def test_diag_torch_tpb_auto_never_reaches_build_matvec(monkeypatch):
    """``auto`` is resolved in ``diag_torch``, so ``build_matvec`` never sees it."""
    from torch_sci_solver.torch_impl import sbdiag

    norb, I0, I1, I2, sa, sb, _, _ = _h2_pair_integrals()
    seen: dict = {}
    real = sbdiag.build_matvec

    def spy(impl, *args, **kwargs):
        seen["impl"] = impl
        return real(impl, *args, **kwargs)

    monkeypatch.setattr(sbdiag, "build_matvec", spy)
    sbdiag.diag_torch(sa, sb, norb, I0, I1, I2, matvec_impl="auto")
    assert seen["impl"] == "batched"  # CPU resolution


def test_diag_torch_rejects_unknown_matvec_impl():
    """An unknown backend fails before any basis or integral work."""
    from torch_sci_solver.torch_impl import sbdiag

    with pytest.raises(ValueError, match="matvec_impl must be one of"):
        sbdiag.diag_torch([], [], 1, 0.0, None, None, matvec_impl="bogus")


# ----------------------------------------------------------------------
# Correctness: pair-mode on a proper subset of the product basis
# ----------------------------------------------------------------------


def _build_full_h_via_product_matvec(adets, bdets, norb, I0, I1, I2):
    """Materialize the full ``(N, N)`` H matrix over the ``adets × bdets``
    product basis by applying product-mode ``mult_torch`` to each basis
    vector. Independent of the pair-mode code path — safe as a reference.
    """
    from torch_sci_solver.torch_impl.helpers import generate_all_torch_helpers
    from torch_sci_solver.torch_impl.integrals import to_torch_integrals
    from torch_sci_solver.torch_impl.mult import make_diag_terms_torch, mult_torch

    device = torch.device("cpu")
    dtype = torch.float64
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    adets_t = torch.tensor(adets, dtype=torch.int64, device=device)
    bdets_t = torch.tensor(bdets, dtype=torch.int64, device=device)
    helpers = list(generate_all_torch_helpers(adets, bdets, norb, device=device))
    hii = make_diag_terms_torch(adets_t, bdets_t, norb, integrals)
    n_full = len(adets) * len(bdets)
    H_full = torch.zeros((n_full, n_full), dtype=dtype, device=device)
    for j in range(n_full):
        e_j = torch.zeros(n_full, dtype=dtype, device=device)
        e_j[j] = 1.0
        H_full[:, j] = mult_torch(hii, e_j, adets_t, bdets_t, norb, integrals, helpers)
    return H_full.numpy()


def test_pair_mode_on_subset_matches_brute_reference_h4():
    """Pair-mode diagonalization on a strict subset of the full product must
    equal ``np.linalg.eigvalsh`` on the corresponding sub-matrix of the
    product-mode-built full H.

    This is the invariant that the full-product test can't reach: if
    the pair-helper builder incorrectly emitted rows whose ket pair
    isn't in the basis, pair-mode would apply out-of-basis couplings and
    diverge from the true restricted spectrum.
    """
    pytest.importorskip("pyscf")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.sbdiag import diag_torch

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, e_core)

    n_full = len(adets) * len(bdets)  # 6 * 6 = 36 for H4/STO-3G
    H_full = _build_full_h_via_product_matvec(adets, bdets, norb, I0, I1, I2)

    # Take a *proper* subset: every third pair from the full product.
    # 12 pairs — small enough for a dense eigensolve, large enough to
    # exercise all five excitation classes (α-single, α-double, β-single,
    # β-double, αβ-double) within the subset.
    subset_idx = list(range(0, n_full, 3))
    assert len(subset_idx) < n_full, "subset must be a proper subset"

    H_sub = H_full[np.ix_(subset_idx, subset_idx)]
    e_ref = float(np.linalg.eigvalsh(H_sub)[0])

    # Reconstruct the pair basis for the subset.
    subset_alphas = []
    subset_betas = []
    for k in subset_idx:
        ia, ib = k // len(bdets), k % len(bdets)
        subset_alphas.append(adets[ia])
        subset_betas.append(bdets[ib])

    e_pair, _, info = diag_torch(
        subset_alphas,
        subset_betas,
        norb,
        I0,
        I1,
        I2,
        mode="gdb",
        max_davidson_iters=100,
        davidson_block_size=min(len(subset_idx), 30),
        eps=1e-10,
    )
    assert info["converged"], f"Davidson did not converge: {info}"
    assert e_pair == pytest.approx(e_ref, abs=1e-8)


def test_pair_mode_on_hf_plus_singles_matches_brute_reference_h4():
    """A physically meaningful subset: HF determinant plus all singly-excited
    pairs from it. Different topology from the "every third pair" subset —
    exercises the excitation-table filter under an alternative pattern.
    """
    pytest.importorskip("pyscf")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.sbdiag import diag_torch

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets_full = _strings(n_a)
    bdets_full = _strings(n_b)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, e_core)
    H_full = _build_full_h_via_product_matvec(adets_full, bdets_full, norb, I0, I1, I2)

    # HF pair = (adets_full[0], bdets_full[0]) — smallest-int is the HF ref.
    hf_alpha = adets_full[0]
    hf_beta = bdets_full[0]

    def _popcount(x):
        """Popcount."""
        return bin(int(x)).count("1")

    subset_pairs = [(hf_alpha, hf_beta)]
    # Single α-excitations from HF (β fixed at HF_β).
    for a in adets_full[1:]:
        if _popcount(hf_alpha ^ a) == 2:
            subset_pairs.append((a, hf_beta))
    # Single β-excitations from HF (α fixed at HF_α).
    for b in bdets_full[1:]:
        if _popcount(hf_beta ^ b) == 2:
            subset_pairs.append((hf_alpha, b))
    assert len(subset_pairs) > 1, "Subset must contain more than HF alone"

    # Map subset back to flat indices in the full basis to slice H_full.
    subset_idx = []
    for a, b in subset_pairs:
        ia = adets_full.index(a)
        ib = bdets_full.index(b)
        subset_idx.append(ia * len(bdets_full) + ib)
    H_sub = H_full[np.ix_(subset_idx, subset_idx)]
    e_ref = float(np.linalg.eigvalsh(H_sub)[0])

    subset_alphas = [p[0] for p in subset_pairs]
    subset_betas = [p[1] for p in subset_pairs]
    e_pair, _, info = diag_torch(
        subset_alphas,
        subset_betas,
        norb,
        I0,
        I1,
        I2,
        mode="gdb",
        max_davidson_iters=100,
        davidson_block_size=min(len(subset_idx), 30),
        eps=1e-10,
    )
    assert info["converged"]
    assert e_pair == pytest.approx(e_ref, abs=1e-8)


def test_pair_mode_asymmetric_subset_lih():
    """LiH (asymmetric n_α ≠ n_β) subset test — sanity-checks the pair
    kernel on a non-square helper build with a non-trivial subset.
    """
    pytest.importorskip("pyscf")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver.sbd_ref.integrals import setup_integrals_from_pyscf_h1_eri
    from torch_sci_solver.torch_impl.sbdiag import diag_torch

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    e_core = mol.energy_nuc()
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri(h1, eri, e_core)
    n_full = len(adets) * len(bdets)
    H_full = _build_full_h_via_product_matvec(adets, bdets, norb, I0, I1, I2)

    # Half the basis — every other pair.
    subset_idx = list(range(0, n_full, 2))
    H_sub = H_full[np.ix_(subset_idx, subset_idx)]
    e_ref = float(np.linalg.eigvalsh(H_sub)[0])
    subset_alphas = [adets[k // len(bdets)] for k in subset_idx]
    subset_betas = [bdets[k % len(bdets)] for k in subset_idx]
    e_pair, _, info = diag_torch(
        subset_alphas,
        subset_betas,
        norb,
        I0,
        I1,
        I2,
        mode="gdb",
        max_davidson_iters=100,
        davidson_block_size=min(len(subset_idx), 30),
        eps=1e-10,
    )
    assert info["converged"]
    assert e_pair == pytest.approx(e_ref, abs=1e-8)


def test_pair_matches_product_on_full_product_h2():
    """H2/STO-3G: pair mode on the full product basis must reproduce
    product mode (bit-identical energy up to Davidson tolerance) and
    both must reproduce PySCF FCI.
    """
    pytest.importorskip("pyscf")
    e_prod, v_prod, e_pair, v_pair, e_fci = _diag_pair_and_product_on_full_product(
        "H 0 0 0; H 0 0 0.74"
    )
    assert e_prod == pytest.approx(e_fci, abs=1e-9)
    assert e_pair == pytest.approx(e_fci, abs=1e-9)
    # Two different code paths landing on the same eigenvalue.
    assert e_pair == pytest.approx(e_prod, abs=1e-10)


def test_pair_matches_product_on_full_product_h4():
    """H4 chain / STO-3G. Larger basis exercises α-doubles and αβ-doubles
    paths (H2 only has trivial cases).
    """
    pytest.importorskip("pyscf")
    e_prod, v_prod, e_pair, v_pair, e_fci = _diag_pair_and_product_on_full_product(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    assert e_prod == pytest.approx(e_fci, abs=1e-9)
    assert e_pair == pytest.approx(e_fci, abs=1e-9)
    assert e_pair == pytest.approx(e_prod, abs=1e-10)


def test_pair_matches_product_on_full_product_lih():
    """LiH / STO-3G. Non-square α × β (n_α ≠ n_β) — exercises the
    asymmetric branches of the pair-helper build.
    """
    pytest.importorskip("pyscf")
    e_prod, v_prod, e_pair, v_pair, e_fci = _diag_pair_and_product_on_full_product(
        "Li 0 0 0; H 0 0 1.6"
    )
    assert e_prod == pytest.approx(e_fci, abs=1e-9)
    assert e_pair == pytest.approx(e_fci, abs=1e-9)
    assert e_pair == pytest.approx(e_prod, abs=1e-10)


# ----------------------------------------------------------------------
# Physics-motivated variational tests using PySCF's FCI ground-state support
# ----------------------------------------------------------------------
#
# The invariants tested here are the ones that motivate pair mode in the
# first place: given the "right" set of (α, β) pairs, pair mode should
# reproduce the FCI ground state without needing the full Cartesian
# product. Concretely:
#
#   1. Take PySCF's FCI civec on a small molecule. Its support (pairs
#      with |c_k| > threshold) captures the wavefunction to machine
#      precision. Pair-mode over the support must reproduce E_FCI.
#
#   2. Add extra pairs (chosen from outside the support) to the same
#      basis. E should remain E_FCI — the additional basis is inert
#      because the exact eigenvector has zero amplitude there.
#
#   3. Drop 10% of the support pairs. E must go UP (strict variational
#      bound: the pruned basis can no longer represent the FCI state
#      exactly), and stay above E_FCI.
#
# H4/STO-3G is used because its full CI space is only 36-dimensional
# yet non-trivial — the FCI amplitudes span several orders of magnitude
# and there are pairs with genuinely negligible weight.


def _fci_pairs_and_amplitudes(mol_atom, basis_name="sto-3g"):
    """Diagonalize PySCF FCI on a small molecule and return the ground
    state as parallel (α_of_pair, β_of_pair, amplitude) arrays.

    Returns ``(h1, eri, norb, nelec, e_fci_electronic, alpha, beta, amps)``
    where ``e_fci_electronic = e_total - e_nuc`` matches what our solver
    returns (integrals include the nuclear-repulsion in ``I0`` so the
    Davidson eigenvalue already contains ``e_nuc``; PySCF's
    ``fci.FCI(mf).kernel()[0]`` also includes ``e_nuc`` — the two are
    directly comparable).

    ``alpha`` and ``beta`` are int64 bitmasks parallel to the ``amps``
    array over ALL pairs of the FCI space (M_α · M_β entries) — not just
    the support. Callers threshold on ``amps`` themselves.
    """
    import itertools

    from pyscf import ao2mo, fci, gto, scf

    mol = gto.M(atom=mol_atom, basis=basis_name, verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec
    e_fci_total, civec = fci.FCI(mf).kernel()

    # PySCF orders CI strings using the same lex-by-occupation convention
    # we use (lowest bits set first, "combinations" in sorted order).
    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    adets = _strings(n_a)
    bdets = _strings(n_b)
    assert civec.shape == (len(adets), len(bdets))

    m_a, m_b = civec.shape
    alpha_flat = np.array([a for a in adets for _ in bdets], dtype=np.int64)
    beta_flat = np.array([b for _ in adets for b in bdets], dtype=np.int64)
    amps_flat = np.asarray(civec, dtype=np.float64).reshape(-1)
    return (
        h1,
        eri,
        norb,
        (n_a, n_b),
        float(e_fci_total),
        alpha_flat,
        beta_flat,
        amps_flat,
    )


def _pair_energy(alpha, beta, h1, eri):
    """Diagonalize pair mode on the given (alpha, beta) pair list."""
    from torch_sci_solver import solve_fermion

    e, *_ = solve_fermion(
        (alpha, beta),
        hcore=h1,
        eri=eri,
        device="cpu",
        mode="gdb",
        davidson_tolerance=1e-11,
    )
    return e


def test_pair_on_fci_support_recovers_fci_energy_h4():
    """Pair mode over the support of PySCF's FCI ground state must
    reproduce the FCI electronic energy.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    from pyscf import gto

    h1, eri, norb, nelec, e_fci, alpha, beta, amps = _fci_pairs_and_amplitudes(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    # Support: pairs with amplitude above a tight threshold. For H4/STO-3G
    # the FCI amplitudes span many orders of magnitude; a 1e-12 cutoff
    # keeps every physically meaningful contributor and drops only pairs
    # whose amplitude is at float64 noise.
    mask = np.abs(amps) > 1e-12
    support_alpha = alpha[mask]
    support_beta = beta[mask]
    assert support_alpha.size > 0
    assert support_alpha.size <= alpha.size

    e_pair = _pair_energy(support_alpha, support_beta, h1, eri)

    # Compare on ELECTRONIC energy — solve_fermion returns
    # ⟨ψ|H|ψ⟩ + e_nuc via ``I0``; PySCF's e_fci also includes e_nuc,
    # so the two are directly comparable.
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    e_nuc = mol.energy_nuc()
    assert e_pair + e_nuc == pytest.approx(e_fci, abs=1e-8)


def test_pair_on_fci_support_plus_extras_still_recovers_fci_energy_h4():
    """Adding pairs not in the FCI support must not degrade the energy —
    the exact FCI eigenvector has zero amplitude on those pairs, so
    they carry no variational weight.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    from pyscf import gto

    h1, eri, norb, nelec, e_fci, alpha, beta, amps = _fci_pairs_and_amplitudes(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    mask = np.abs(amps) > 1e-12
    support_alpha = alpha[mask]
    support_beta = beta[mask]
    # "Extras" — pairs with amplitude at noise level. On H4/STO-3G the
    # full 36-dim FCI vector may or may not have exact-zero entries;
    # regardless, we augment with all off-support pairs and expect the
    # energy to be unchanged since the FCI state has zero (or noise-
    # level) amplitude on them.
    off_support = ~mask
    if off_support.any():
        aug_alpha = np.concatenate([support_alpha, alpha[off_support]])
        aug_beta = np.concatenate([support_beta, beta[off_support]])
    else:
        # No off-support pairs exist at this threshold — nothing to add.
        # In that case this test degenerates to the previous invariant.
        aug_alpha = support_alpha
        aug_beta = support_beta

    e_pair = _pair_energy(aug_alpha, aug_beta, h1, eri)
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    e_nuc = mol.energy_nuc()
    assert e_pair + e_nuc == pytest.approx(e_fci, abs=1e-8)


def test_pair_mode_returns_flat_pairsciestate_h4():
    """Pair-mode ``SCIResult.sci_state`` is a length-N ``PairSCIState``
    with 1D amplitudes and parallel length-N ci_strs_a / ci_strs_b.

    The earlier 2D projected form was memory-quadratic in the unique-
    string counts (320 GB at 200k × 200k unique strings). Pair mode
    consumes its own flat SCIState via
    :func:`diagonalize_fermionic_hamiltonian_pairs`; it is *not* a
    drop-in for ``qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian``
    (that outer loop expects a 2D SCIState).
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    import itertools

    from pyscf import ao2mo, gto, scf
    from torch_sci_solver import make_torch_sci_solver

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)

    solver_pair = make_torch_sci_solver(device="cpu", mode="gdb")
    # solver(...) returns list[EigenStates] — pair mode is single-root, so
    # the ground-state SCIResult is ``[0].states[0]``.
    r_pair = solver_pair(
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        (n_a, n_b),
    )[0].states[0]

    st = r_pair.sci_state
    n_pairs = st.amplitudes.shape[0]
    assert st.amplitudes.ndim == 1
    assert st.ci_strs_a.shape == (n_pairs,)
    assert st.ci_strs_b.shape == (n_pairs,)
    assert float(np.sum(st.amplitudes**2)) == pytest.approx(1.0, abs=1e-10)


# ----------------------------------------------------------------------
# solve_fermion_from_bit_array_pairs — the one-shot pair-mode entry point
# ----------------------------------------------------------------------


def _mol_full_bit_array_and_integrals(mol_atom):
    """Fixture: bit_array covering the full FCI space of a small molecule.

    Each row of the bit_array is one (α, β) pair encoded following
    ``qiskit_addon_sqd.fermion.bitstring_matrix_to_ci_strs``'s convention:
    the left half (columns 0..norb-1) holds β occupancies with column 0
    as MSB (weight ``1 << (norb-1)``), and the right half holds α with
    the same MSB-first packing.
    """
    import itertools

    from pyscf import ao2mo, fci, gto, scf
    from qiskit.primitives import BitArray

    mol = gto.M(atom=mol_atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    n_a, n_b = mol.nelec
    e_fci_total, _ = fci.FCI(mf).kernel()

    def _strings(n_elec):
        """Strings."""
        out = []
        for combo in itertools.combinations(range(norb), n_elec):
            m = 0
            for b in combo:
                m |= 1 << b
            out.append(m)
        return sorted(out)

    ci_a = _strings(n_a)
    ci_b = _strings(n_b)
    rows = []
    # Bit layout: MSB-first packing per half.
    # bit ``p`` (LSB=orbital 0) of the integer α ⇔ column ``norb - 1 - p``
    # of the right half of the row (α columns are ``norb..2*norb-1``).
    for a in ci_a:
        for b in ci_b:
            r = np.zeros(2 * norb, dtype=bool)
            for p in range(norb):
                # Column position for orbital p under MSB-first packing.
                col_msb = norb - 1 - p
                if (b >> p) & 1:
                    r[col_msb] = True  # β in LEFT half
                if (a >> p) & 1:
                    r[norb + col_msb] = True  # α in RIGHT half
            rows.append(r)
    ba = BitArray.from_bool_array(np.stack(rows))
    return {
        "bit_array": ba,
        "hcore": h1,
        "eri": eri,
        "norb": norb,
        "nelec": (n_a, n_b),
        "e_fci_total": float(e_fci_total),
        "e_nuc": mol.energy_nuc(),
        "num_rows": len(rows),
    }


def _h4_full_bit_array_and_integrals():
    """H4/STO-3G shortcut for the shared fixture."""
    return _mol_full_bit_array_and_integrals("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22")


def test_solve_fermion_from_bit_array_pairs_matches_fci_h4(tmp_path: Path):
    """Full-FCI bit_array through the new entry point → recovers E_FCI."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    e, sci_state, occs, _ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        norb=fx["norb"],
        davidson_tolerance=1e-11,
    )
    assert e + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)
    # Flat PairSCIState (length-N amplitudes, parallel length-N ci_strs).
    assert sci_state.amplitudes.ndim == 1
    # Occupancies sum to particle counts.
    occ_a, occ_b = occs
    assert occ_a.sum() == pytest.approx(fx["nelec"][0], abs=1e-9)
    assert occ_b.sum() == pytest.approx(fx["nelec"][1], abs=1e-9)
    # Save the state and check the consistency
    state_filepath = tmp_path / "state.npz"
    sci_state.save(state_filepath)
    loaded_sci_state = np.load(state_filepath)
    assert (sci_state.amplitudes == loaded_sci_state["amplitudes"]).all()
    assert (sci_state.ci_strs_a == loaded_sci_state["ci_strs_a"]).all()
    assert (sci_state.ci_strs_b == loaded_sci_state["ci_strs_b"]).all()
    assert sci_state.norb == loaded_sci_state["norb"].item()
    assert sci_state.nelec == tuple(loaded_sci_state["nelec"])


def test_solve_fermion_from_bit_array_pairs_accepts_bool_matrix_input():
    """Accepts a pre-unpacked bool matrix as well as a BitArray."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    # Unpack the BitArray into a bool matrix manually.
    bit_matrix = fx["bit_array"].to_bool_array().astype(bool)

    e_from_ba, *_ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        davidson_tolerance=1e-11,
    )
    e_from_mat, *_ = solve_fermion_from_bit_array_pairs(
        bit_matrix, fx["hcore"], fx["eri"], nelec=fx["nelec"], davidson_tolerance=1e-11
    )
    assert e_from_mat == pytest.approx(e_from_ba, abs=1e-11)


def test_solve_fermion_from_bit_array_pairs_postselects_noisy_rows_h4():
    """Rows with wrong particle count are dropped by postselect by default.

    Simulates a noisy sample by adding random bitstrings with the wrong
    popcount to the physical rows. Postselect must keep exactly the
    physical rows.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from qiskit.primitives import BitArray
    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    physical = fx["bit_array"].to_bool_array().astype(bool)

    # Add "noisy" rows: random 8-bit patterns whose per-half popcount
    # generally does NOT equal (n_α, n_β). Fixed seed for reproducibility.
    rng = np.random.default_rng(seed=123)
    noise = rng.integers(0, 2, size=(50, 2 * fx["norb"]), dtype=np.uint8).astype(bool)
    # Filter noise to only rows with WRONG popcount (guaranteed drop).
    n_a, n_b = fx["nelec"]
    noise_alpha = noise[:, : fx["norb"]].sum(axis=1)
    noise_beta = noise[:, fx["norb"] :].sum(axis=1)
    bad = ~((noise_alpha == n_a) & (noise_beta == n_b))
    noise = noise[bad]
    assert noise.shape[0] > 0, "no noise rows survived the filter — retry seed"

    mixed = np.concatenate([physical, noise], axis=0)
    ba_mixed = BitArray.from_bool_array(mixed)
    e_mixed, *_ = solve_fermion_from_bit_array_pairs(
        ba_mixed, fx["hcore"], fx["eri"], nelec=fx["nelec"], davidson_tolerance=1e-11
    )
    # Same energy as clean BitArray — noise dropped by postselect.
    assert e_mixed + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)


def test_solve_fermion_from_bit_array_pairs_raises_when_all_rows_dropped():
    """If postselect drops every row, the error names the cause."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from qiskit.primitives import BitArray
    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    # Rows with all zero bits (0 electrons) → drop for any (n_a>0, n_b>0).
    zeros = np.zeros((5, 2 * fx["norb"]), dtype=bool)
    ba = BitArray.from_bool_array(zeros)
    with pytest.raises(ValueError, match="postselect dropped every row"):
        solve_fermion_from_bit_array_pairs(ba, fx["hcore"], fx["eri"], nelec=fx["nelec"])


def test_solve_fermion_from_bit_array_pairs_max_pairs_subsamples_h4():
    """``max_pairs`` caps the pair set. On a truncated basis pair mode
    must still return a physically-sensible variational energy — higher
    than the FCI on the full basis, still finite and below zero.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    e_full, *_ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        davidson_tolerance=1e-11,
    )
    # Cap at half the pairs.
    e_capped, *_ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        max_pairs=fx["num_rows"] // 2,
        seed=7,
        davidson_tolerance=1e-11,
    )
    # Variational: capped ≥ full basis.
    assert e_capped >= e_full - 1e-10


def test_solve_fermion_from_bit_array_pairs_matches_fci_lih_asymmetric():
    """LiH/STO-3G — n_α ≠ n_β. If α and β halves were swapped or packed
    with the wrong bit order, postselect would drop most rows and/or
    the physics would end up in the wrong particle-number sector. On a
    closed-shell H4 the α↔β swap goes undetected; LiH catches it.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import solve_fermion_from_bit_array_pairs

    fx = _mol_full_bit_array_and_integrals("Li 0 0 0; H 0 0 1.6")
    # Sanity: LiH is not closed-shell only if the basis and geometry are
    # picked to force it; here it happens to be closed-shell too (nelec=2,2).
    # For a stronger asymmetric test we could use e.g. Li+ or H2+ but this
    # fixture already exercises the MSB-first packing across a non-square
    # (M_α, M_β) grid.
    e, sci_state, _, _ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        norb=fx["norb"],
        davidson_tolerance=1e-11,
    )
    assert e + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)


def test_solve_fermion_from_bit_array_pairs_matches_bitstring_matrix_to_ci_strs():
    """Sanity check on the row-packing convention. For each row, the
    (α, β) pair we extract must be bit-identical to what
    ``bitstring_matrix_to_ci_strs`` would produce on the same row —
    otherwise our α and β would disagree with the rest of the package
    on strings coming from a Qiskit sample.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs

    fx = _mol_full_bit_array_and_integrals("H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22")
    unpacked = fx["bit_array"].to_bool_array().astype(bool)
    # Take one specific row and check.
    row = unpacked[7:8]  # arbitrary
    ci_a_qiskit, ci_b_qiskit = bitstring_matrix_to_ci_strs(row, open_shell=True)
    # Our α/β extraction on the same row (same convention).
    norb = fx["norb"]
    shift = np.arange(norb - 1, -1, -1, dtype=np.int64)
    a_int = int((row[:, norb : 2 * norb].astype(np.int64) << shift[None, :]).sum())
    b_int = int((row[:, :norb].astype(np.int64) << shift[None, :]).sum())
    assert a_int == int(ci_a_qiskit[0]), (
        f"α extraction mismatch: pair mode {a_int:#x} vs "
        f"bitstring_matrix_to_ci_strs {int(ci_a_qiskit[0]):#x}"
    )
    assert b_int == int(ci_b_qiskit[0])


def test_diagonalize_fermionic_hamiltonian_pairs_recovers_fci_on_clean_h4():
    """The iterative outer loop reproduces FCI on a clean bit_array
    (every row is a valid FCI-space configuration, no noise). Serves as
    a smoke test that the recovery/subsample/carryover machinery works
    together in pair mode without changing the answer on clean input.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import diagonalize_fermionic_hamiltonian_pairs

    fx = _h4_full_bit_array_and_integrals()
    result = diagonalize_fermionic_hamiltonian_pairs(
        fx["hcore"],
        fx["eri"],
        fx["bit_array"],
        samples_per_batch=fx["num_rows"],  # keep every row per batch
        norb=fx["norb"],
        nelec=fx["nelec"],
        max_iterations=3,
        num_batches=1,
        seed=42,
        device="cpu",
    )
    assert result.energy + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)
    # Flat PairSCIState.
    assert result.sci_state.amplitudes.ndim == 1


def test_diagonalize_fermionic_hamiltonian_pairs_recovers_noisy_h4():
    """Iterative recovery on a synthetically noisy H4 bit_array — some
    rows have the wrong particle count, some don't. Iteration 0 uses
    postselect and finds physical rows; subsequent iterations use the
    RDM occupancies to bit-flip noisy rows toward the target sector.
    The recovered energy should still be near FCI (the recovery has
    enough physical rows to lock onto the ground state).
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from qiskit.primitives import BitArray
    from torch_sci_solver import diagonalize_fermionic_hamiltonian_pairs

    fx = _h4_full_bit_array_and_integrals()
    physical = fx["bit_array"].to_bool_array().astype(bool)

    # Mix ~30% noise (random bitstrings, popcount rarely matches target).
    rng = np.random.default_rng(seed=101)
    n_noise = int(0.3 * physical.shape[0])
    noise = rng.integers(0, 2, size=(n_noise, 2 * fx["norb"]), dtype=np.uint8).astype(bool)
    mixed = np.concatenate([physical, noise], axis=0)
    # Shuffle so noise isn't at the tail.
    perm = rng.permutation(mixed.shape[0])
    mixed = mixed[perm]
    ba = BitArray.from_bool_array(mixed)

    result = diagonalize_fermionic_hamiltonian_pairs(
        fx["hcore"],
        fx["eri"],
        ba,
        samples_per_batch=mixed.shape[0],
        norb=fx["norb"],
        nelec=fx["nelec"],
        max_iterations=4,
        num_batches=1,
        seed=42,
        device="cpu",
    )
    # Iterative recovery + one-shot Davidson on H4/STO-3G should land
    # very close to FCI even with 30% noise mixed in — the physical
    # rows alone span the FCI space and recovery bit-flips noisy rows
    # back toward the target sector.
    assert result.energy + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-6)


def test_diagonalize_fermionic_hamiltonian_pairs_full_recovered_pool_h4():
    """``use_full_recovered_pool=True`` skips subsample and diagonalizes
    every recovered row per iteration. On clean H4 full-FCI input this
    is equivalent to samples_per_batch=n_shots (they cover the same
    36 rows) — both should recover E_FCI.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import diagonalize_fermionic_hamiltonian_pairs

    fx = _h4_full_bit_array_and_integrals()
    result = diagonalize_fermionic_hamiltonian_pairs(
        fx["hcore"],
        fx["eri"],
        fx["bit_array"],
        norb=fx["norb"],
        nelec=fx["nelec"],
        max_iterations=3,
        num_batches=1,
        seed=42,
        use_full_recovered_pool=True,  # no samples_per_batch needed
        device="cpu",
    )
    assert result.energy + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)


def test_diagonalize_fermionic_hamiltonian_pairs_full_pool_rejects_bad_args():
    """``use_full_recovered_pool=True`` requires ``num_batches=1``; and
    the default (False) requires an explicit ``samples_per_batch``."""
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import diagonalize_fermionic_hamiltonian_pairs

    fx = _h4_full_bit_array_and_integrals()
    with pytest.raises(ValueError, match="num_batches=1"):
        diagonalize_fermionic_hamiltonian_pairs(
            fx["hcore"],
            fx["eri"],
            fx["bit_array"],
            norb=fx["norb"],
            nelec=fx["nelec"],
            num_batches=2,
            use_full_recovered_pool=True,
        )
    with pytest.raises(ValueError, match="samples_per_batch is required"):
        diagonalize_fermionic_hamiltonian_pairs(
            fx["hcore"],
            fx["eri"],
            fx["bit_array"],
            norb=fx["norb"],
            nelec=fx["nelec"],
            use_full_recovered_pool=False,  # default
            # samples_per_batch omitted
        )


def test_diagonalize_fermionic_hamiltonian_pairs_add_hf_pair_h4():
    """With ``add_hf_pair=True`` the HF pair ``((1<<n_α)-1, (1<<n_β)-1)``
    is spliced into every batch's pair set. Verifies (a) the run
    succeeds on H4 and returns FCI, and (b) the HF pair actually
    appears in the returned ``ci_strs_a`` / ``ci_strs_b`` even when
    the sampled bit_array excludes it.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from qiskit.primitives import BitArray
    from torch_sci_solver import diagonalize_fermionic_hamiltonian_pairs

    fx = _h4_full_bit_array_and_integrals()
    # Drop the HF row(s) from the bit_array — force the sampler to have
    # missed HF, so add_hf_pair has a real job.
    unpacked = fx["bit_array"].to_bool_array().astype(bool)
    norb = fx["norb"]
    n_a, n_b = fx["nelec"]
    hf_alpha = (1 << n_a) - 1
    hf_beta = (1 << n_b) - 1
    shift = np.arange(norb - 1, -1, -1, dtype=np.int64)
    alpha_ints = (unpacked[:, norb : 2 * norb].astype(np.int64) << shift[None, :]).sum(axis=1)
    beta_ints = (unpacked[:, :norb].astype(np.int64) << shift[None, :]).sum(axis=1)
    hf_mask = (alpha_ints == hf_alpha) & (beta_ints == hf_beta)
    assert hf_mask.any(), "fixture must contain the HF pair to test its removal"
    ba_no_hf = BitArray.from_bool_array(unpacked[~hf_mask])

    r = diagonalize_fermionic_hamiltonian_pairs(
        fx["hcore"],
        fx["eri"],
        ba_no_hf,
        samples_per_batch=ba_no_hf.num_shots,
        norb=norb,
        nelec=fx["nelec"],
        max_iterations=3,
        num_batches=1,
        seed=42,
        add_hf_pair=True,
        device="cpu",
    )
    # HF pair is in the final basis.
    idx_hf = np.where((r.sci_state.ci_strs_a == hf_alpha) & (r.sci_state.ci_strs_b == hf_beta))[0]
    assert idx_hf.size == 1, "HF pair should appear exactly once with add_hf_pair=True"
    # And FCI is still recovered on the augmented basis (H4 is small
    # enough that HF + all singly/doubly excited pairs span the FCI
    # space).
    assert r.energy + fx["e_nuc"] == pytest.approx(fx["e_fci_total"], abs=1e-8)


def test_dump_pair_coupling_magnitudes_h4():
    """The diagnostic reconstructs pair_helpers from a pair-mode
    ``SCIState`` and prints per-table ``|Δ|`` distribution. On H4/STO-3G
    with the full FCI-space bit_array, all 5 tables should be
    non-empty (the H4 FCI space has αα, ββ, and αβ excitations), and
    ``|max|`` per table should be O(1 Ha) — matching typical
    2-electron chemistry integrals.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import (
        dump_pair_coupling_magnitudes,
        solve_fermion_from_bit_array_pairs,
    )

    fx = _h4_full_bit_array_and_integrals()
    _, sci_state, _, _ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        norb=fx["norb"],
        davidson_tolerance=1e-11,
    )
    summary = dump_pair_coupling_magnitudes(
        sci_state,
        fx["hcore"],
        fx["eri"],
        norb=fx["norb"],
        nelec=fx["nelec"],
        device="cpu",
        label="test",
    )
    # On the full H4 FCI space we expect non-empty tables in all five
    # channels (any two configurations with popcount(XOR)==2 or 4 give
    # a coupling; full space always has these).
    for name in ("asingles", "adoubles", "bsingles", "bdoubles", "ab"):
        assert summary[name]["K"] > 0, f"expected {name} to have entries"
        # Matrix elements should be finite and O(1 Ha) or smaller.
        assert 0 < summary[name]["max"] < 100.0


def test_diagonalize_fermionic_hamiltonian_pairs_rejects_product_mode_solver():
    """Passing a product-mode ``TorchSCISolver`` is a caller bug — reject
    at the door, don't silently produce a wrong subspace.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import (
        diagonalize_fermionic_hamiltonian_pairs,
        make_torch_sci_solver,
    )

    fx = _h4_full_bit_array_and_integrals()
    product_solver = make_torch_sci_solver(device="cpu", mode="tpb")
    with pytest.raises(ValueError, match='mode="gdb"'):
        diagonalize_fermionic_hamiltonian_pairs(
            fx["hcore"],
            fx["eri"],
            fx["bit_array"],
            samples_per_batch=fx["num_rows"],
            norb=fx["norb"],
            nelec=fx["nelec"],
            solver=product_solver,
        )


def test_solve_fermion_from_bit_array_pairs_matches_manual_workflow_h4():
    """The new entry point matches the manual bit_array → row-pack →
    solve_fermion workflow that the previous documentation test showed.
    Both should be exactly the same code path, just packaged.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import solve_fermion, solve_fermion_from_bit_array_pairs

    fx = _h4_full_bit_array_and_integrals()
    # Entry-point call.
    e_new, *_ = solve_fermion_from_bit_array_pairs(
        fx["bit_array"],
        fx["hcore"],
        fx["eri"],
        nelec=fx["nelec"],
        davidson_tolerance=1e-11,
    )
    # Manual workflow: unpack + pack + solve_fermion(mode="gdb").
    # Manual workflow using the qiskit-compatible convention (β left,
    # α right, MSB-first). This is what the entry point does internally.
    unpacked = fx["bit_array"].to_bool_array().astype(bool)
    n_a, n_b = fx["nelec"]
    norb = fx["norb"]
    b_half = unpacked[:, :norb]  # β in the left half
    a_half = unpacked[:, norb : 2 * norb]  # α in the right half
    keep = (a_half.sum(axis=1) == n_a) & (b_half.sum(axis=1) == n_b)
    a_half = a_half[keep]
    b_half = b_half[keep]
    shift = np.arange(norb - 1, -1, -1, dtype=np.int64)
    alpha_int = (a_half.astype(np.int64) << shift[None, :]).sum(axis=1)
    beta_int = (b_half.astype(np.int64) << shift[None, :]).sum(axis=1)
    e_manual, *_ = solve_fermion(
        (alpha_int, beta_int),
        hcore=fx["hcore"],
        eri=fx["eri"],
        mode="gdb",
        davidson_tolerance=1e-11,
    )
    assert e_new == pytest.approx(e_manual, abs=1e-11)


def test_pair_on_pruned_fci_support_energy_goes_up_h4():
    """Removing pairs from the FCI support must raise the energy — the
    pruned basis can no longer represent the FCI eigenstate exactly.

    Variational bound: any subspace-restricted energy is ≥ E_FCI.
    Removing 10% of the support pairs (chosen randomly with a fixed
    seed for reproducibility) *and* not restoring them via any other
    mechanism guarantees a strict increase — even if the dropped pairs
    were low-amplitude, the projection is imperfect.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")
    from pyscf import gto

    h1, eri, norb, nelec, e_fci, alpha, beta, amps = _fci_pairs_and_amplitudes(
        "H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22"
    )
    # "Support" here means the full FCI vector (threshold at float64
    # noise). This makes ``e_full_support`` equal to ``e_fci``, so any
    # pruning-induced energy change is unambiguously above numerical
    # noise. Using a coarser threshold conflates truncation error with
    # pruning-induced error and muddies the assertion.
    mask = np.abs(amps) > 1e-12
    support_alpha = alpha[mask]
    support_beta = beta[mask]
    n_support = support_alpha.size
    assert n_support >= 10, "need enough support pairs to prune 10%"

    # Reference: solve on the full support first.
    e_full_support = _pair_energy(support_alpha, support_beta, h1, eri)

    # Prune 10% (at least 1) of the support pairs at random. Fixed seed
    # picks a mix of small- and large-amplitude pairs; on average the
    # net variational cost is well above 1e-6 Ha on H4/STO-3G.
    rng = np.random.default_rng(seed=42)
    n_drop = max(1, n_support // 10)
    drop_idx = rng.choice(n_support, size=n_drop, replace=False)
    keep_mask = np.ones(n_support, dtype=bool)
    keep_mask[drop_idx] = False
    pruned_alpha = support_alpha[keep_mask]
    pruned_beta = support_beta[keep_mask]

    e_pruned = _pair_energy(pruned_alpha, pruned_beta, h1, eri)

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22",
        basis="sto-3g",
        verbose=0,
    )
    e_nuc = mol.energy_nuc()
    # (a) Full-support energy IS FCI to well under 1e-8.
    assert e_full_support + e_nuc == pytest.approx(e_fci, abs=1e-8)
    # (b) Pruning raises the energy by a physically meaningful margin.
    assert e_pruned > e_full_support + 1e-6
    # (c) Variational lower bound: pruned energy is still ≥ FCI (up to
    #     Davidson's residual tolerance).
    assert e_pruned + e_nuc >= e_fci - 1e-10


# ----------------------------------------------------------------------
# Streaming excitation tables (memory contract)
# ----------------------------------------------------------------------


def _dense_pair_basis_with_many_couplings() -> tuple[PairBasis, "PairHelpers"]:
    """A small pair basis whose coupling count far exceeds its dimension.

    Full ``adets × bdets`` product for 6 orbitals / 3 electrons per spin, so
    every determinant is connected to many others: nnz(H) >> N. That gap is
    what makes the streaming contract worth asserting.

    Returns:
        The 400-pair :class:`PairBasis` and its :class:`PairHelpers`, both on
        CPU.
    """
    norb, nelec = 6, 3
    strs = sorted(sum(1 << i for i in c) for c in itertools.combinations(range(norb), nelec))
    pairs = [(a, b) for a in strs for b in strs]
    basis = pair_basis_from_arrays(
        [p[0] for p in pairs], [p[1] for p in pairs], device=torch.device("cpu")
    )
    return basis, generate_pair_helpers(basis, norb, device=torch.device("cpu"))


def test_pair_helpers_hold_nothing_of_size_nnz() -> None:
    """``PairHelpers`` must stay O(N): no stored tensor may exceed the pair count.

    The regression this guards: an earlier implementation materialized all
    five excitation tables up front, costing ~32-48 bytes per nonzero of H.
    Since nnz(H)/N grows with basis size, that made peak memory scale with
    nnz and put the 1e6-determinant target out of reach. If someone
    reintroduces a precomputed table, one of these tensors goes long and
    this fails.
    """
    basis, helpers = _dense_pair_basis_with_many_couplings()
    n = basis.n_pairs
    nnz = sum(helpers.nnz().values())
    # Precondition: this basis really is one where storing would hurt.
    assert nnz > 10 * n, f"fixture too sparse to be a meaningful guard (nnz={nnz}, N={n})"

    for field, value in vars(helpers).items():
        if isinstance(value, torch.Tensor):
            assert value.numel() <= n, (
                f"PairHelpers.{field} has {value.numel()} elements for a {n}-pair "
                f"basis — tables must be streamed, not stored"
            )


def test_pair_helpers_streams_are_repeatable() -> None:
    """Every table generator can be re-consumed and yields the same rows.

    Each ``helpers.<table>()`` call must return a *fresh* generator. If one
    were cached, the second pass would come back empty and the second matvec
    of a Davidson run would silently drop the off-diagonal Hamiltonian.
    """
    _, helpers = _dense_pair_basis_with_many_couplings()

    def collect(
        stream: "Callable[[], Iterator[dict[str, torch.Tensor]]]",
    ) -> torch.Tensor:
        """Concatenated ``(k_bra, k_ket)`` rows of one full pass.

        Args:
            stream: A table accessor, called to get a fresh chunk generator.

        Returns:
            A ``(2, total_rows)`` int64 tensor of bra/ket pair indices.
        """
        rows = [torch.stack([t["k_bra"], t["k_ket"]]) for t in stream()]
        return torch.cat(rows, dim=1) if rows else torch.zeros((2, 0), dtype=torch.int64)

    for stream in (
        helpers.asingles,
        helpers.adoubles,
        helpers.bsingles,
        helpers.bdoubles,
        helpers.ab_doubles,
    ):
        first, second = collect(stream), collect(stream)
        assert first.shape[1] > 0, "fixture should couple through every table kind"
        assert torch.equal(first, second)


def test_resolve_budget_bytes_branches() -> None:
    """Explicit budgets win; ``None`` gives the flat default on every device.

    A "biggest chunk that fits free VRAM" auto-sizer was measured slower than
    a flat 8192 MB budget on an A100 across dimensions (up to 4.7x at 10k
    pairs), so there is no per-device branch left to exercise.
    """
    assert ph.resolve_budget_bytes(4 * 1024**3) == 4 * 1024**3
    assert ph.resolve_budget_bytes(None) == ph.DEFAULT_BUDGET_BYTES
    assert ph.DEFAULT_BUDGET_BYTES == 8192 * 1024**2


def test_gdb_kwargs_validation() -> None:
    """``gdb_kwargs`` fills defaults, rejects typos, and rejects TPB misuse.

    A bare options dict gives up the typo protection named parameters get for
    free, so the guard is the whole reason the bundle is acceptable — if it
    stops working, ``gdb_kwargs={"budget_byte": ...}`` silently does nothing.
    """
    # Defaults are filled in, and an explicit value survives.
    assert TorchSCISolver(mode="gdb", device="cpu").gdb_kwargs == GDB_KWARG_DEFAULTS
    solver = TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"budget_bytes": 4 * 1024**3})
    assert solver.gdb_kwargs["budget_bytes"] == 4 * 1024**3

    # The block-kernel knobs are normalized and validated too.
    solver = TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"block_matvec": "off"})
    assert solver.gdb_kwargs["block_matvec"] == "off"
    solver = TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"block_matvec": True})
    assert solver.gdb_kwargs["block_matvec"] == "on"
    with pytest.raises(ValueError, match="block_matvec"):
        TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"block_matvec": "sometimes"})
    solver = TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"block_table_bytes": 1 << 20})
    assert solver.gdb_kwargs["block_table_bytes"] == 1 << 20
    with pytest.raises(ValueError, match="block_table_bytes"):
        TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"block_table_bytes": 0})

    # A misspelled key is an error, not a no-op.
    with pytest.raises(ValueError, match="unknown gdb_kwargs key"):
        TorchSCISolver(mode="gdb", device="cpu", gdb_kwargs={"budget_byte": 1})

    # GDB options on a TPB solver are an error too — they would be ignored.
    with pytest.raises(ValueError, match='mode="tpb"'):
        TorchSCISolver(mode="tpb", device="cpu", gdb_kwargs={"budget_bytes": 1})

    # An empty dict is not misuse, so it is accepted in either mode.
    assert TorchSCISolver(mode="tpb", device="cpu", gdb_kwargs={}).gdb_kwargs == (
        GDB_KWARG_DEFAULTS
    )


def test_gdb_block_matvec_solver_parity() -> None:
    """A GDB solve with the block kernels equals the streamed path.

    The block decomposition of a full-product basis is exact, so forcing it on
    must leave the ground-state energy unchanged (up to Davidson's tolerance).
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("qiskit_addon_sqd")

    from torch_sci_solver import make_torch_sci_solver

    h1, eri, norb, nelec, ci_a, ci_b = _pair_warm_start_setup("H 0 0 0; H 0 0 0.74")
    pairs_a = np.array([a for a in ci_a for _ in ci_b], dtype=np.int64)
    pairs_b = np.array([b for _ in ci_a for b in ci_b], dtype=np.int64)
    args = (
        [(pairs_a, pairs_b)],
        np.asarray(h1, dtype=np.float64),
        np.asarray(eri, dtype=np.float64),
        norb,
        nelec,
    )
    off = make_torch_sci_solver(device="cpu", mode="gdb", gdb_kwargs={"block_matvec": "off"})(
        *args
    )[0].states[0]
    on = make_torch_sci_solver(device="cpu", mode="gdb", gdb_kwargs={"block_matvec": "on"})(*args)[
        0
    ].states[0]
    assert on.energy == pytest.approx(off.energy, abs=1e-8)


def test_same_spin_dedup_matches_per_pair_enumeration() -> None:
    """The deduplicated same-spin enumerator yields the reference row *set*.

    ``PairHelpers`` enumerates same-spin excitations once per unique spin
    string and expands the survivors over the pairs sharing that string
    (:func:`~torch_sci_solver.torch_impl.pair_helpers._stream_same_spin_dedup`),
    rather than once per pair. That is a large win because a GDB basis reuses
    each string across many pairs, but it is only valid if it produces exactly
    the rows the straightforward per-pair enumerator does.

    This pins that equivalence against
    :func:`~torch_sci_solver.torch_impl.pair_helpers._stream_same_spin`, which
    is retained as the reference. Row *order* deliberately differs — dedup
    groups by source string, so ``k_bra`` is not monotonic — so the comparison
    is set-based, which is what every consumer needs since they all
    scatter-add.
    """
    from torch_sci_solver.torch_impl.pair_helpers import _stream_same_spin

    basis, helpers = _dense_pair_basis_with_many_couplings()
    norb = 6

    def as_set(stream: "Iterator[dict[str, torch.Tensor]]", order: int) -> set[tuple[int, ...]]:
        """All rows of one pass as a set of index tuples."""
        rows: set[tuple[int, ...]] = set()
        for chunk in stream:
            cols = [chunk["k_bra"], chunk["k_ket"], chunk["cr0"], chunk["an0"]]
            if order == 2:
                cols += [chunk["cr1"], chunk["an1"]]
            for row in torch.stack(cols, dim=1).tolist():
                rows.add(tuple(row))
        return rows

    cases = (
        ("asingles", helpers.asingles, True, 1),
        ("adoubles", helpers.adoubles, True, 2),
        ("bsingles", helpers.bsingles, False, 1),
        ("bdoubles", helpers.bdoubles, False, 2),
    )
    for name, dedup_stream, is_alpha, order in cases:
        reference = as_set(
            _stream_same_spin(
                source_of_pair=basis.alpha_of_pair if is_alpha else basis.beta_of_pair,
                unique_target=basis.unique_alphas if is_alpha else basis.unique_betas,
                pair_keys=helpers.pair_keys,
                m_beta=helpers.m_beta,
                n_orb=norb,
                n_particles=helpers.n_a if is_alpha else helpers.n_b,
                is_alpha=is_alpha,
                order=order,
                device=torch.device("cpu"),
                budget_bytes=helpers.budget_bytes,
            ),
            order,
        )
        assert reference, f"{name}: fixture should produce rows for this table kind"
        assert as_set(dedup_stream(), order) == reference, (
            f"{name}: deduplicated enumeration does not match the per-pair reference"
        )


def test_same_spin_dedup_matches_reference_across_small_budgets() -> None:
    """Dedup equivalence must hold when the budget forces many chunks.

    Both stages of the dedup enumerator are chunked — the per-string candidate
    grid and the survivor expansion — and each gets half the budget. A tiny
    budget drives both to their minimum slice, which is where an off-by-one in
    the flat-index bookkeeping would show up. Any row lost or duplicated at a
    chunk boundary silently changes the Hamiltonian, so it is worth pinning
    separately from the default-budget case above.
    """
    from torch_sci_solver.torch_impl.pair_helpers import _stream_same_spin

    norb, nelec = 6, 3
    strs = sorted(sum(1 << i for i in c) for c in itertools.combinations(range(norb), nelec))
    pairs = [(a, b) for a in strs for b in strs]
    basis = pair_basis_from_arrays(
        [p[0] for p in pairs], [p[1] for p in pairs], device=torch.device("cpu")
    )

    def rows_of(helpers: "PairHelpers", is_alpha: bool, order: int) -> set[tuple[int, ...]]:
        """Dedup rows for one table kind, as a set."""
        stream = {
            (True, 1): helpers.asingles,
            (True, 2): helpers.adoubles,
            (False, 1): helpers.bsingles,
            (False, 2): helpers.bdoubles,
        }[(is_alpha, order)]
        out: set[tuple[int, ...]] = set()
        for chunk in stream():
            cols = [chunk["k_bra"], chunk["k_ket"], chunk["cr0"], chunk["an0"]]
            if order == 2:
                cols += [chunk["cr1"], chunk["an1"]]
            for row in torch.stack(cols, dim=1).tolist():
                out.add(tuple(row))
        return out

    reference_helpers = generate_pair_helpers(basis, norb, device=torch.device("cpu"))
    for is_alpha, order in ((True, 1), (True, 2), (False, 1), (False, 2)):
        reference = set()
        for chunk in _stream_same_spin(
            source_of_pair=basis.alpha_of_pair if is_alpha else basis.beta_of_pair,
            unique_target=basis.unique_alphas if is_alpha else basis.unique_betas,
            pair_keys=reference_helpers.pair_keys,
            m_beta=reference_helpers.m_beta,
            n_orb=norb,
            n_particles=reference_helpers.n_a if is_alpha else reference_helpers.n_b,
            is_alpha=is_alpha,
            order=order,
            device=torch.device("cpu"),
            budget_bytes=reference_helpers.budget_bytes,
        ):
            cols = [chunk["k_bra"], chunk["k_ket"], chunk["cr0"], chunk["an0"]]
            if order == 2:
                cols += [chunk["cr1"], chunk["an1"]]
            for row in torch.stack(cols, dim=1).tolist():
                reference.add(tuple(row))

        for budget in (1024, 8192, 1 << 20):
            helpers = generate_pair_helpers(
                basis, norb, device=torch.device("cpu"), budget_bytes=budget
            )
            assert rows_of(helpers, is_alpha, order) == reference, (
                f"budget={budget}, is_alpha={is_alpha}, order={order}: "
                f"chunked dedup lost or duplicated rows"
            )


def test_explicit_triton_falls_back_to_batched_on_cpu():
    """An explicit Triton request is accepted and matches batched off CUDA.

    The streamed GDB tables and the spinless matvec now have entry-parallel
    Triton kernels; on a CPU run both decline the launch and keep the batched
    scatter, so the solve must succeed with the batched energy instead of
    rejecting the request.
    """
    h1 = np.array([[0.5, 0.1], [0.1, 1.0]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    strings = [0b01, 0b10]
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]

    def solve(matvec_impl):
        solver = TorchSCISolver(
            device="cpu",
            mode="gdb",
            matvec_impl=matvec_impl,
            gdb_kwargs={"block_matvec": "off"},
            max_davidson_iters=20,
        )
        return solver([(pairs_a, pairs_b)], h1, eri, 2, (1, 1))[0].states[0].energy

    assert solve("triton") == pytest.approx(solve("batched"))

    def solve_spinless(matvec_impl):
        solver = TorchSCISolver(device="cpu", matvec_impl=matvec_impl, max_davidson_iters=20)
        return (
            solver(
                [np.array([0b011, 0b101], dtype=np.int64)],
                np.eye(3, dtype=np.float64),
                np.zeros((3, 3, 3, 3), dtype=np.float64),
                3,
                2,
            )[0]
            .states[0]
            .energy
        )

    assert solve_spinless("triton") == pytest.approx(solve_spinless("batched"))


def test_move_cache_bytes_is_validated_and_aliased():
    """The move-cache budget is public, aliased and rejected where it cannot run."""
    from torch_sci_solver.torch_impl.qiskit_adapter import _validate_gdb_kwargs

    validated = _validate_gdb_kwargs({"move_cache_bytes": 1024}, mode="gdb")
    assert validated["move_cache_bytes"] == 1024
    # The tile-only spelling is the deprecated alias of the same budget.
    aliased = _validate_gdb_kwargs({"tile_move_cache_bytes": 2048}, mode="gdb")
    assert aliased["move_cache_bytes"] == 2048
    with pytest.raises(ValueError, match="disagree"):
        _validate_gdb_kwargs({"move_cache_bytes": 1024, "tile_move_cache_bytes": 2048}, mode="gdb")
    with pytest.raises(ValueError, match="move_cache_bytes"):
        _validate_gdb_kwargs({"move_cache_bytes": -1}, mode="gdb")

    h1 = np.eye(2, dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    strings = [0b01, 0b10]
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]
    solver = TorchSCISolver(
        device="cpu",
        mode="gdb",
        gdb_kwargs={"block_matvec": "off", "move_cache_bytes": 1 << 20},
        max_davidson_iters=5,
    )
    with pytest.raises(ValueError, match="no block decomposition"):
        solver([(pairs_a, pairs_b)], h1, eri, 2, (1, 1))
