"""Tests for the entry-parallel Triton kernels of the streamed matvec passes.

Covers:

1. CPU: the layout gate accepts dense and compact ERIs, single-word and
   limb-matrix strings, and 1-D or ``(N, K)`` block inputs, while the limb
   width, one-body and ERI-store rules reject malformed tables; and the public
   matvecs/solvers accept ``triton_matvec=True`` and reproduce the batched
   results exactly through the off-CUDA fallback.
2. CUDA (``@pytest.mark.gpu``): the kernels themselves reproduce the batched
   :func:`~torch_sci_solver.torch_impl.determinants.one_excite_batch` and
   :func:`~torch_sci_solver.torch_impl.determinants.two_excite_batch` scatters
   for the spinless and GDB-streamed layouts, real and complex, in the
   ``float64``/``complex128`` (``1e-12``) and ``float32``/``complex64``
   (``1e-5``) precisions. Coverage includes the compact spatial ERI store at
   ``n_spinorb >= 64``, limb-matrix strings at 70 and 130 spin-orbitals
   (including compact+complex+limb launches), ``(N, K)`` block inputs with
   the owner-computes ``offset`` path, and the public matvec/solver wiring.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import itertools
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch_sci_solver.torch_impl.integrals import (
    CompactEri,
    TorchIntegrals,
    setup_integrals_from_pyscf_h1_eri_vec,
    to_torch_integrals,
    to_torch_integrals_spinless,
)
from torch_sci_solver.torch_impl.mult_entries_triton import (
    TRITON_ENTRY_PASS_BACKENDS,
    _entry_layout_supported,
    _eri_buffer,
    _single_entry_layout_supported,
    _table_limbs,
    apply_double_entries,
    apply_single_entries,
)
from torch_sci_solver.torch_impl.mult_pairs import (
    _scatter_double_pair,
    _scatter_single_pair,
    make_diag_terms_pair_torch,
    mult_pair_local_torch,
    mult_pair_torch,
)
from torch_sci_solver.torch_impl.mult_spinless import (
    _scatter_double_spinless,
    _scatter_single_spinless,
    make_diag_terms_spinless_torch,
    mult_spinless_torch,
)
from torch_sci_solver.torch_impl.pair_helpers import generate_pair_helpers
from torch_sci_solver.torch_impl.pairs import pair_basis_from_arrays
from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver
from torch_sci_solver.torch_impl.spinless_helpers import (
    generate_spinless_helpers,
    spinless_basis_from_array,
)

# Spinless fixture: 6 spin-orbitals, 3 particles (20 determinants).
NMO = 6
N_PARTICLES = 3
# Pair fixture: 4 spatial orbitals, 2 alpha and 2 beta electrons (36 pairs).
NORB = 4
N_A = N_B = 2

_DTYPES = [torch.float64, torch.float32, torch.complex128, torch.complex64]
_IDS = ["fp64", "fp32", "c128", "c64"]
_ATOL = {
    torch.float64: 1e-12,
    torch.float32: 1e-5,
    torch.complex128: 1e-12,
    torch.complex64: 1e-5,
}

# The eight chemist-order permutations of a real 8-fold-symmetric ERI.
_PERMUTATIONS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (0, 1, 3, 2),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 2, 0, 1),
    (2, 3, 1, 0),
    (3, 2, 1, 0),
)


def _bits(*orbitals: int) -> int:
    """Bitmask with the given orbital positions set."""
    return sum(1 << int(orbital) for orbital in orbitals)


# Spinless selected bases: one base determinant plus double-excitation
# neighbours, so every ``doubles`` table is non-empty while the basis stays
# small. The wide sets place orbitals beyond each 64-bit limb boundary so the
# parity ranges and ERI offsets cross limbs.
NMO_COMPACT = 64
_DETS64 = [
    _bits(0, 1, 2),
    _bits(0, 40, 63),
    _bits(1, 40, 63),
    _bits(2, 40, 63),
    _bits(0, 30, 40),
    _bits(2, 30, 63),
]
_DETS70 = [
    _bits(0, 1, 2),
    _bits(0, 64, 69),
    _bits(1, 64, 69),
    _bits(2, 64, 69),
    _bits(0, 60, 67),
    _bits(2, 60, 67),
]
_DETS130 = [
    _bits(0, 1, 2),
    _bits(0, 100, 129),
    _bits(1, 100, 129),
    _bits(2, 70, 128),
    _bits(0, 66, 127),
    _bits(2, 66, 127),
]

# Pair selected bases: one shared α/β string set per width, whose members are
# connected by single and double excitations so all five streamed channels are
# non-empty. Strings beyond the 64-bit boundary become limb matrices.
_N_ORB_COMPACT = 32
_STRINGS32 = [_bits(0, 1), _bits(30, 31), _bits(0, 31), _bits(2, 3)]
_STRINGS35 = [_bits(0, 1), _bits(31, 34), _bits(0, 34), _bits(2, 33)]
_STRINGS65 = [_bits(0, 1), _bits(61, 64), _bits(0, 64), _bits(2, 63)]


def _comb_strs(norb: int, nelec: int) -> list[int]:
    """Sorted bitmasks of every ``nelec``-of-``norb`` occupancy."""
    combos = itertools.combinations(range(norb), nelec)
    return sorted(sum(1 << bit for bit in combo) for combo in combos)


def _real_integrals(norb: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Real symmetric ``hcore`` and an 8-fold-symmetric real ERI."""
    rng = np.random.default_rng(seed)
    hcore = rng.standard_normal((norb, norb))
    hcore = 0.5 * (hcore + hcore.T)
    raw = rng.standard_normal((norb,) * 4)
    eri = np.zeros_like(raw)
    for perm in _PERMUTATIONS:
        eri += raw.transpose(perm)
    return hcore, eri / 8.0


def _complex_integrals(norb: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Complex Hermitian ``hcore`` and an ERI with both Hermitian symmetries."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    hcore = hc + hc.conj().T
    g = rng.standard_normal((norb,) * 4) + 1j * rng.standard_normal((norb,) * 4)
    eri = 0.5 * (g + g.conj().transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return hcore, eri


def _rand_vector(n: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
    """Deterministic vector of ``dtype`` on ``device``."""
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(n, generator=gen, dtype=torch.float64)
    if dtype.is_complex:
        imag = torch.randn(n, generator=gen, dtype=torch.float64)
        return (real + 1j * imag).to(dtype).to(device)
    return real.to(dtype).to(device)


def _spinless_fixture(dtype: torch.dtype, device: torch.device | str):
    """Full ``C(NMO, N_PARTICLES)`` spinless basis, integrals, diagonal and vector."""
    strs = _comb_strs(NMO, N_PARTICLES)
    basis = spinless_basis_from_array(strs, NMO, N_PARTICLES, device=device)
    helpers = generate_spinless_helpers(basis)
    hcore, eri = (
        _complex_integrals(NMO, seed=11) if dtype.is_complex else _real_integrals(NMO, seed=11)
    )
    integrals = to_torch_integrals_spinless(hcore, eri, device=device, dtype=dtype)
    hii = make_diag_terms_spinless_torch(basis, integrals)
    return basis, helpers, integrals, hii, _rand_vector(basis.dim, dtype, device)


def _pair_fixture(dtype: torch.dtype, device: torch.device | str):
    """Full-product pair basis (``block_matvec="off"``), integrals, diagonal, vector."""
    alphas = _comb_strs(NORB, N_A)
    betas = _comb_strs(NORB, N_B)
    pairs_a = np.array([a for a in alphas for _ in betas], dtype=np.int64)
    pairs_b = np.array([b for _ in alphas for b in betas], dtype=np.int64)
    basis = pair_basis_from_arrays(pairs_a, pairs_b, n_orb=NORB, device=device)
    helpers = generate_pair_helpers(basis, NORB, device=device, block_matvec="off")
    h1, eri = _complex_integrals(NORB, seed=13) if dtype.is_complex else _real_integrals(NORB, 13)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)
    hii = make_diag_terms_pair_torch(basis, NORB, integrals)
    return basis, helpers, integrals, hii, _rand_vector(basis.n_pairs, dtype, device)


def _assert_close(got: torch.Tensor, want: torch.Tensor, dtype: torch.dtype) -> None:
    """Assert agreement at the dtype's parity tolerance."""
    diff = float((got - want).abs().max()) if got.numel() else 0.0
    assert torch.allclose(got, want, atol=_ATOL[dtype], rtol=0.0), diff


def _skip_without_triton() -> None:
    """Skip a CUDA test when Triton cannot be imported."""
    pytest.importorskip("triton")


def _cuda() -> torch.device:
    """Return the CUDA device, skipping when none is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _hermitian_matrix(n: int, seed: int, *, is_complex: bool) -> np.ndarray:
    """Random Hermitian ``(n, n)`` matrix (real symmetric when not complex)."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((n, n))
    if is_complex:
        h = h + 1j * rng.standard_normal((n, n))
    return 0.5 * (h + h.conj().T)


def _low_rank_eri(n: int, seed: int, *, is_complex: bool) -> np.ndarray:
    """Hermitian, particle-exchange-symmetric ``(n, n, n, n)`` ERI.

    ``E[p, q, r, s] = g[p, q] conj(g[r, s])`` with a Hermitian ``g`` carries
    ``(pq|rs) == conj((qp|sr))``; averaging it with its ``(pq|rs) <-> (rs|pq)``
    transpose (which commutes with that symmetry) adds the particle-exchange
    symmetry the spatial checks require. Its rank-one structure keeps the
    wide-basis fixtures cheap.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    if is_complex:
        g = g + 1j * rng.standard_normal((n, n))
    g = g + g.conj().T
    eri = np.einsum("pq,rs->pqrs", g, g.conj())
    return 0.5 * (eri + np.transpose(eri, (2, 3, 0, 1)))


def _compact_bundle(n_orb: int, dtype: torch.dtype, device) -> TorchIntegrals:
    """Bundle from ``to_torch_integrals(..., compact_eri=True)``."""
    h1 = _hermitian_matrix(n_orb, seed=41, is_complex=dtype.is_complex)
    eri = _low_rank_eri(n_orb, seed=42, is_complex=dtype.is_complex)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
    integrals = to_torch_integrals(I0, I1, I2, device=device, dtype=dtype, compact_eri=True)
    assert integrals.eri_compact
    return integrals


def _manual_compact_bundle(spatial: np.ndarray, dtype: torch.dtype, device) -> TorchIntegrals:
    """Compact bundle around a spatial ERI without materialising ``(2L)^4``.

    ``to_torch_integrals(..., compact_eri=True)`` takes an already-built dense
    ``TwoInt``, so its producer path allocates ``(2L)^4`` elements before
    extracting the spatial block. At 130 spin-orbitals that dense tensor is
    several gigabytes, so the wide fixtures wrap the same
    :class:`CompactEri` store directly.
    """
    n_orb = int(spatial.shape[0])
    n_spinorb = 2 * n_orb
    dev = torch.device(device)
    zeros2 = torch.zeros((n_spinorb, n_spinorb), dtype=dtype, device=dev)
    return TorchIntegrals(
        I0=0.0,
        I1=zeros2,
        I2_eri=CompactEri(torch.tensor(spatial, dtype=dtype, device=dev)),
        I1_diag=torch.zeros(n_spinorb, dtype=dtype, device=dev),
        I2_direct=zeros2,
        I2_exchange=zeros2,
        n_spinorb=n_spinorb,
        device=dev,
        dtype=dtype,
        eri_compact=True,
    )


def _dense_spinless_bundle(nmo: int, dtype: torch.dtype, device) -> TorchIntegrals:
    """Dense spin-orbital bundle (``nmo^4`` ERI) for wide-basis tests."""
    hcore = _hermitian_matrix(nmo, seed=44, is_complex=dtype.is_complex)
    eri = _low_rank_eri(nmo, seed=45, is_complex=dtype.is_complex)
    return to_torch_integrals_spinless(
        hcore, eri, device=device, dtype=dtype, check_hermitian=False
    )


def _dense_pair_bundle(n_orb: int, dtype: torch.dtype, device) -> TorchIntegrals:
    """Dense spin-orbital bundle built from a spatial ``(L, L, L, L)`` ERI."""
    h1 = _hermitian_matrix(n_orb, seed=46, is_complex=dtype.is_complex)
    eri = _low_rank_eri(n_orb, seed=47, is_complex=dtype.is_complex)
    I0, I1, I2 = setup_integrals_from_pyscf_h1_eri_vec(h1, eri, 0.0)
    return to_torch_integrals(I0, I1, I2, device=device, dtype=dtype)


def _spinless_case(nmo: int, dets: list[int], device):
    """Selected spinless basis and its streaming helpers."""
    basis = spinless_basis_from_array(dets, nmo, 3, device=device)
    return basis, generate_spinless_helpers(basis)


def _pair_case(n_orb: int, strings: list[int], device):
    """Selected pair basis (one shared α/β string set) and its helpers."""
    pairs_a = [a for a in strings for _ in strings]
    pairs_b = [b for _ in strings for b in strings]
    basis = pair_basis_from_arrays(pairs_a, pairs_b, n_orb=n_orb, device=device)
    helpers = generate_pair_helpers(basis, n_orb, device=device, block_matvec="off")
    return basis, helpers


def _table_columns(table: dict) -> dict:
    """The six per-entry arrays of one doubles table."""
    return dict(
        k_bra=table["k_bra"],
        k_ket=table["k_ket"],
        cr0=table["cr0"],
        cr1=table["cr1"],
        an0=table["an0"],
        an1=table["an1"],
    )


def _pair_columns(table: dict) -> dict:
    """Creation/annihilation columns of one pair table (cross-spin keys differ)."""
    if "cr_a" in table:
        return dict(cr0=table["cr_a"], cr1=table["cr_b"], an0=table["an_a"], an1=table["an_b"])
    return dict(cr0=table["cr0"], cr1=table["cr1"], an0=table["an0"], an1=table["an1"])


def _pair_tables(helpers) -> list[dict]:
    """Every streamed doubles table of a pair basis."""
    return list(helpers.adoubles()) + list(helpers.bdoubles()) + list(helpers.ab_doubles())


def _apply_spinless_table(Wb, T, integrals, basis, table, *, offset=0) -> bool:
    """Apply one spinless-mode doubles table with the entry kernel."""
    return apply_double_entries(
        Wb,
        T,
        integrals,
        mode="spinless",
        dets=basis.dets,
        offset=offset,
        **_table_columns(table),
    )


def _scatter_spinless_tables(Wb, T, integrals, basis, tables) -> None:
    """Batched reference: scatter every spinless doubles table into ``Wb``."""
    for table in tables:
        _scatter_double_spinless(
            Wb=Wb,
            T=T,
            dets=basis.dets,
            integrals=integrals,
            chunk_size=10**9,
            **_table_columns(table),
        )


def _apply_pair_table(Wb, T, integrals, basis, table, *, offset=0) -> bool:
    """Apply one pair-mode doubles table with the entry kernel."""
    return apply_double_entries(
        Wb,
        T,
        integrals,
        k_bra=table["k_bra"],
        k_ket=table["k_ket"],
        mode="pair",
        alpha_of_pair=basis.alpha_of_pair,
        beta_of_pair=basis.beta_of_pair,
        offset=offset,
        **_pair_columns(table),
    )


def _scatter_pair_tables(Wb, T, integrals, basis, n_orb, tables) -> None:
    """Batched reference: scatter every pair doubles table into ``Wb``."""
    for table in tables:
        _scatter_double_pair(
            Wb=Wb,
            T=T,
            alpha_of_pair=basis.alpha_of_pair,
            beta_of_pair=basis.beta_of_pair,
            k_bra=table["k_bra"],
            k_ket=table["k_ket"],
            n_orb=n_orb,
            integrals=integrals,
            chunk_size=10**9,
            **_pair_columns(table),
        )


def _pair_column_reference(t_col, integrals, basis, n_orb, tables):
    """Batched pair-scatter reference for one column of a block input."""
    out = torch.zeros_like(t_col)
    _scatter_pair_tables(out, t_col, integrals, basis, n_orb, tables)
    return out


def _single_columns(table: dict) -> dict:
    """The four per-entry arrays of one singles table."""
    return dict(k_bra=table["k_bra"], k_ket=table["k_ket"], cr=table["cr0"], an=table["an0"])


def _pair_single_tables(helpers) -> list[dict]:
    """Every streamed singles table of a pair basis (α then β)."""
    return list(helpers.asingles()) + list(helpers.bsingles())


def _apply_spinless_single_table(Wb, T, integrals, basis, table, *, offset=0) -> bool:
    """Apply one spinless-mode singles table with the entry kernel."""
    return apply_single_entries(
        Wb,
        T,
        integrals,
        mode="spinless",
        dets=basis.dets,
        offset=offset,
        **_single_columns(table),
    )


def _scatter_spinless_singles(Wb, T, integrals, basis, tables) -> None:
    """Batched reference: scatter every spinless singles table into ``Wb``."""
    for table in tables:
        _scatter_single_spinless(
            Wb=Wb,
            T=T,
            dets=basis.dets,
            integrals=integrals,
            chunk_size=10**9,
            **_single_columns(table),
        )


def _apply_pair_single_table(Wb, T, integrals, basis, table, *, offset=0) -> bool:
    """Apply one pair-mode singles table with the entry kernel."""
    return apply_single_entries(
        Wb,
        T,
        integrals,
        mode="pair",
        alpha_of_pair=basis.alpha_of_pair,
        beta_of_pair=basis.beta_of_pair,
        offset=offset,
        **_single_columns(table),
    )


def _scatter_pair_singles(Wb, T, integrals, basis, n_orb, tables) -> None:
    """Batched reference: scatter every streamed pair singles table into ``Wb``."""
    for table in tables:
        _scatter_single_pair(
            Wb=Wb,
            T=T,
            alpha_of_pair=basis.alpha_of_pair,
            beta_of_pair=basis.beta_of_pair,
            n_orb=n_orb,
            integrals=integrals,
            chunk_size=10**9,
            **_single_columns(table),
        )


def _pair_single_column_reference(t_col, integrals, basis, n_orb, tables):
    """Batched pair-scatter reference for one column of a singles block input."""
    out = torch.zeros_like(t_col)
    _scatter_pair_singles(out, t_col, integrals, basis, n_orb, tables)
    return out


# ============================================================================
# CPU: the predicate falls back, never raising or changing the result
# ============================================================================


def test_apply_single_entries_returns_false_on_cpu() -> None:
    """Off CUDA the singles predicate declines, so callers keep the batched scatter."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.singles()))
    ran = apply_single_entries(
        torch.zeros(basis.dim),
        x,
        integrals,
        k_bra=table["k_bra"],
        k_ket=table["k_ket"],
        cr=table["cr0"],
        an=table["an0"],
        mode="spinless",
        dets=basis.dets,
    )
    assert ran is False

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_table = _pair_single_tables(pair_helpers)[0]
    Wb = torch.zeros(pair_basis.n_pairs)
    ran = _apply_pair_single_table(Wb, px, pair_integrals, pair_basis, pair_table)
    assert ran is False


def test_entry_pass_backends_list_singles_tiled() -> None:
    """The backend table advertises the singles passes as entry-tiled."""
    assert TRITON_ENTRY_PASS_BACKENDS["spinless_singles"] == "tiled"
    assert TRITON_ENTRY_PASS_BACKENDS["gdb_streamed_singles"] == "tiled"


def test_single_entry_layout_accepts_extended_storage() -> None:
    """Compact ERI, limb strings and block inputs pass the singles layout gate."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.singles()))
    tables = tuple(_single_columns(table).values())
    Wb = torch.zeros(basis.dim, dtype=torch.float64)
    block = torch.stack([x, -x], dim=1)
    block_wb = torch.zeros_like(block)

    # Dense ERI, single-word (M, 1) strings, vector and block inputs.
    assert _single_entry_layout_supported(Wb, x, integrals, tables, 0, basis.dets, None, None)
    assert _single_entry_layout_supported(
        block_wb, block, integrals, tables, 0, basis.dets, None, None
    )

    # A two-limb matrix covers the six-bit determinant space (the extra limb
    # is zero, which the kernel reads as no bits).
    limb = torch.zeros(basis.dim, 2, dtype=torch.int64)
    limb[:, 0] = basis.dets[:, 0]
    assert _single_entry_layout_supported(Wb, x, integrals, tables, 0, limb, None, None)

    # The compact spatial store produced by ``compact_eri=True`` (L = 3).
    compact = _compact_bundle(NMO // 2, torch.float64, "cpu")
    assert _single_entry_layout_supported(Wb, x, compact, tables, 0, basis.dets, None, None)
    assert _single_entry_layout_supported(block_wb, block, compact, tables, 0, limb, None, None)

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_tables = tuple(_single_columns(_pair_single_tables(pair_helpers)[0]).values())
    pair_wb = torch.zeros(pair_basis.n_pairs, dtype=torch.float64)
    pair_block = torch.stack([px, -px], dim=1)
    pair_compact = _compact_bundle(NORB, torch.float64, "cpu")
    for wb, t_in, ints in (
        (pair_wb, px, pair_integrals),
        (torch.zeros_like(pair_block), pair_block, pair_integrals),
        (pair_wb, px, pair_compact),
    ):
        assert _single_entry_layout_supported(
            wb,
            t_in,
            ints,
            pair_tables,
            1,
            None,
            pair_basis.alpha_of_pair,
            pair_basis.beta_of_pair,
        )


def test_single_entry_layout_rejects_malformed_layouts() -> None:
    """One-body, table, store and shape mismatches decline without raising."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    columns = _single_columns(next(iter(helpers.singles())))
    tables = tuple(columns.values())
    Wb = torch.zeros(basis.dim, dtype=torch.float64)

    # The one-body tensor must be a contiguous dense (2L, 2L) image in the
    # output's dtype.
    for bad_i1 in (None, integrals.I1.float(), integrals.I1[:-1], integrals.I1.t()):
        assert not _single_entry_layout_supported(
            Wb, x, replace(integrals, I1=bad_i1), tables, 0, basis.dets, None, None
        )

    # Table rows, string rows, T/Wb ranks and the ERI store share the doubles
    # gate. A shorter table is legitimate (the callers stream per chunk), but
    # the columns of one table must agree in length.
    mismatched = dict(columns)
    mismatched["k_ket"] = mismatched["k_ket"][:2]
    assert not _single_entry_layout_supported(
        Wb, x, integrals, tuple(mismatched.values()), 0, basis.dets, None, None
    )
    assert not _single_entry_layout_supported(
        Wb, x, integrals, tables, 0, basis.dets[:2], None, None
    )
    block = torch.stack([x, -x], dim=1)
    assert not _single_entry_layout_supported(
        Wb, block, integrals, tables, 0, basis.dets, None, None
    )
    bad_eri = replace(integrals, eri_compact=True)
    assert not _single_entry_layout_supported(Wb, x, bad_eri, tables, 0, basis.dets, None, None)

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_tables = tuple(_single_columns(_pair_single_tables(pair_helpers)[0]).values())
    pair_wb = torch.zeros(pair_basis.n_pairs, dtype=torch.float64)
    assert not _single_entry_layout_supported(
        pair_wb,
        px,
        pair_integrals,
        pair_tables,
        1,
        None,
        pair_basis.alpha_of_pair,
        pair_basis.beta_of_pair[:1],
    )


def test_apply_single_entries_falls_back_for_extended_layouts_on_cpu() -> None:
    """Layout-accepted singles calls still decline off CUDA."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.singles()))
    block = torch.stack([x, -x], dim=1)
    compact = _compact_bundle(NMO // 2, torch.float64, "cpu")
    assert (
        _apply_spinless_single_table(
            torch.zeros(basis.dim, dtype=torch.float64), x, compact, basis, table
        )
        is False
    )
    assert (
        _apply_spinless_single_table(torch.zeros_like(block), block, integrals, basis, table)
        is False
    )

    limb = torch.zeros(basis.dim, 2, dtype=torch.int64)
    limb[:, 0] = basis.dets[:, 0]
    assert (
        apply_single_entries(
            torch.zeros(basis.dim, dtype=torch.float64),
            x,
            integrals,
            mode="spinless",
            dets=limb,
            **_single_columns(table),
        )
        is False
    )

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    assert (
        _apply_pair_single_table(
            torch.zeros(pair_basis.n_pairs, dtype=torch.float64),
            px,
            pair_integrals,
            pair_basis,
            _pair_single_tables(pair_helpers)[0],
        )
        is False
    )


def test_apply_double_entries_returns_false_on_cpu() -> None:
    """Off CUDA the predicate declines, so callers keep the batched scatter."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.doubles()))
    ran = apply_double_entries(
        torch.zeros(basis.dim),
        x,
        integrals,
        k_bra=table["k_bra"],
        k_ket=table["k_ket"],
        cr0=table["cr0"],
        cr1=table["cr1"],
        an0=table["an0"],
        an1=table["an1"],
        mode="spinless",
        dets=basis.dets,
    )
    assert ran is False

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_table = next(iter(pair_helpers.adoubles()))
    Wb = torch.zeros(pair_basis.n_pairs)
    ran = _apply_pair_table(Wb, px, pair_integrals, pair_basis, pair_table)
    assert ran is False


def test_entry_layout_accepts_extended_storage() -> None:
    """Compact ERI, limb strings and block inputs pass the layout gate."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.doubles()))
    tables = tuple(_table_columns(table).values())
    Wb = torch.zeros(basis.dim, dtype=torch.float64)
    block = torch.stack([x, -x], dim=1)
    block_wb = torch.zeros_like(block)

    # Dense ERI, single-word (M, 1) strings, vector and block inputs.
    assert _entry_layout_supported(Wb, x, integrals, tables, 0, basis.dets, None, None)
    assert _entry_layout_supported(block_wb, block, integrals, tables, 0, basis.dets, None, None)

    # A two-limb matrix covers the six-bit determinant space (the extra limb
    # is zero, which the kernel reads as no bits).
    limb = torch.zeros(basis.dim, 2, dtype=torch.int64)
    limb[:, 0] = basis.dets[:, 0]
    assert _entry_layout_supported(Wb, x, integrals, tables, 0, limb, None, None)

    # The compact spatial store produced by ``compact_eri=True`` (L = 3).
    compact = _compact_bundle(NMO // 2, torch.float64, "cpu")
    assert isinstance(_eri_buffer(compact), torch.Tensor)
    assert _entry_layout_supported(Wb, x, compact, tables, 0, basis.dets, None, None)
    assert _entry_layout_supported(block_wb, block, compact, tables, 0, limb, None, None)

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_wb = torch.zeros(pair_basis.n_pairs, dtype=torch.float64)
    pair_block = torch.stack([px, -px], dim=1)
    pair_compact = _compact_bundle(NORB, torch.float64, "cpu")
    for table in (next(iter(pair_helpers.adoubles())), next(iter(pair_helpers.ab_doubles()))):
        pair_tables = tuple(_pair_columns(table).values())
        for wb, t_in, ints in (
            (pair_wb, px, pair_integrals),
            (torch.zeros_like(pair_block), pair_block, pair_integrals),
            (pair_wb, px, pair_compact),
        ):
            assert _entry_layout_supported(
                wb,
                t_in,
                ints,
                pair_tables,
                1,
                None,
                pair_basis.alpha_of_pair,
                pair_basis.beta_of_pair,
            )


def test_entry_layout_rejects_malformed_layouts() -> None:
    """Width, store and shape mismatches decline without raising."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.doubles()))
    tables = tuple(_table_columns(table).values())
    Wb = torch.zeros(basis.dim, dtype=torch.float64)

    # A single word cannot cover a wider determinant space; extra limbs can.
    assert _table_limbs(torch.zeros(basis.dim, dtype=torch.int64), 65) is None
    assert _table_limbs(torch.zeros(basis.dim, 2, dtype=torch.int64), 130) is None
    assert _table_limbs(torch.zeros(basis.dim, 3, dtype=torch.int64), 130) == 3
    assert _table_limbs(torch.zeros(basis.dim, dtype=torch.int32), 6) is None

    # String rows, T/Wb rank and column count, contiguity and dtype all gate.
    assert not _entry_layout_supported(Wb, x, integrals, tables, 0, basis.dets[:2], None, None)
    block = torch.stack([x, -x], dim=1)
    assert not _entry_layout_supported(
        torch.zeros(basis.dim, 3, dtype=torch.float64),
        block,
        integrals,
        tables,
        0,
        basis.dets,
        None,
        None,
    )
    assert not _entry_layout_supported(Wb, block, integrals, tables, 0, basis.dets, None, None)
    non_contig = block.t().contiguous().t()
    assert not _entry_layout_supported(
        torch.zeros_like(block), non_contig, integrals, tables, 0, basis.dets, None, None
    )
    assert not _entry_layout_supported(Wb, x.float(), integrals, tables, 0, basis.dets, None, None)

    # ``eri_compact`` without the wrapper's spatial buffer is not a store the
    # kernel can read (a dense tensor has no ``_spatial`` slot).
    bad_eri = replace(integrals, eri_compact=True)
    assert _eri_buffer(bad_eri) is None
    assert not _entry_layout_supported(Wb, x, bad_eri, tables, 0, basis.dets, None, None)

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_table = next(iter(pair_helpers.adoubles()))
    pair_tables = tuple(_pair_columns(pair_table).values())
    pair_wb = torch.zeros(pair_basis.n_pairs, dtype=torch.float64)
    assert not _entry_layout_supported(
        pair_wb,
        px,
        pair_integrals,
        pair_tables,
        1,
        None,
        pair_basis.alpha_of_pair,
        pair_basis.beta_of_pair[:2],
    )
    wide_alpha = torch.zeros(pair_basis.n_pairs, 2, dtype=torch.int64)
    wide_beta = torch.zeros(pair_basis.n_pairs, 1, dtype=torch.int64)
    assert not _entry_layout_supported(
        pair_wb, px, pair_integrals, pair_tables, 1, None, wide_alpha, wide_beta
    )


def test_apply_double_entries_falls_back_for_extended_layouts_on_cpu() -> None:
    """Layout-accepted calls still decline off CUDA, so callers keep batched."""
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, "cpu")
    table = next(iter(helpers.doubles()))
    block = torch.stack([x, -x], dim=1)
    compact = _compact_bundle(NMO // 2, torch.float64, "cpu")
    assert (
        _apply_spinless_table(torch.zeros(basis.dim, dtype=torch.float64), x, compact, basis, table)
        is False
    )
    assert _apply_spinless_table(torch.zeros_like(block), block, integrals, basis, table) is False

    limb = torch.zeros(basis.dim, 2, dtype=torch.int64)
    limb[:, 0] = basis.dets[:, 0]
    assert (
        apply_double_entries(
            torch.zeros(basis.dim, dtype=torch.float64),
            x,
            integrals,
            mode="spinless",
            dets=limb,
            **_table_columns(table),
        )
        is False
    )

    pair_basis, pair_helpers, pair_integrals, _phii, px = _pair_fixture(torch.float64, "cpu")
    pair_table = next(iter(pair_helpers.adoubles()))
    assert (
        _apply_pair_table(
            torch.zeros(pair_basis.n_pairs, dtype=torch.float64),
            px,
            pair_integrals,
            pair_basis,
            pair_table,
        )
        is False
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
@pytest.mark.parametrize("block", [False, True], ids=["vector", "block"])
def test_spinless_triton_flag_falls_back_bit_exactly_on_cpu(dtype, block) -> None:
    """``triton_matvec=True`` on CPU reproduces the batched matvec exactly."""
    basis, helpers, integrals, hii, x = _spinless_fixture(dtype, "cpu")
    T = torch.stack([x, 2 * x, -x], dim=1) if block else x
    want = mult_spinless_torch(hii, T, basis, helpers, integrals)
    got = mult_spinless_torch(hii, T, basis, helpers, integrals, triton_matvec=True)
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_pair_streamed_triton_flag_falls_back_bit_exactly_on_cpu(dtype) -> None:
    """``triton_matvec=True`` on CPU reproduces the batched streamed pair matvec."""
    basis, helpers, integrals, hii, x = _pair_fixture(dtype, "cpu")
    want = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)
    got = mult_pair_torch(hii, x, basis, helpers, NORB, integrals, triton_matvec=True)
    assert torch.equal(got, want)

    for row_start, row_stop in ((0, 10), (7, 23), (0, basis.n_pairs)):
        want_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
        )
        got_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
            triton_matvec=True,
        )
        assert torch.equal(got_local, want_local)


def test_solver_spinless_explicit_triton_matches_batched_on_cpu() -> None:
    """Explicit Triton is accepted on the spinless path and matches batched."""
    hcore, eri = _complex_integrals(NMO, seed=5)
    strs = np.asarray(_comb_strs(NMO, N_PARTICLES), dtype=np.int64)
    common = dict(device="cpu", dtype=torch.complex128, max_davidson_iters=100)
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)([strs], hcore, eri, NMO, N_PARTICLES)[0]
        .states[0]
        .energy
    )
    solver = TorchSCISolver(matvec_impl="triton", **common)
    assert solver.matvec_impl == "triton"
    e_triton = solver([strs], hcore, eri, NMO, N_PARTICLES)[0].states[0].energy
    assert e_triton == pytest.approx(e_batched, abs=1e-10)


def test_solver_gdb_streamed_explicit_triton_matches_batched_on_cpu() -> None:
    """Streamed GDB with explicit Triton is accepted and matches batched."""
    h1, eri = _real_integrals(NORB, seed=17)
    ci = [(_comb_strs(NORB, N_A), _comb_strs(NORB, N_B))]
    common = dict(
        device="cpu",
        mode="gdb",
        gdb_kwargs={"block_matvec": "off"},
        max_davidson_iters=100,
    )
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)(ci, h1, eri, NORB, (N_A, N_B))[0]
        .states[0]
        .energy
    )
    solver = TorchSCISolver(matvec_impl="triton", **common)
    assert solver.matvec_impl == "triton"
    e_triton = solver(ci, h1, eri, NORB, (N_A, N_B))[0].states[0].energy
    assert e_triton == pytest.approx(e_batched, abs=1e-10)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_spinless_singles_triton_flag_falls_back_bit_exactly_on_cpu(dtype) -> None:
    """The singles call site falls back bit-exactly on the spinless matvec."""
    basis, helpers, integrals, hii, x = _spinless_fixture(dtype, "cpu")
    assert list(helpers.singles())
    T = torch.stack([x, 2 * x, -x], dim=1)
    for t_in in (x, T):
        want = mult_spinless_torch(hii, t_in, basis, helpers, integrals)
        got = mult_spinless_torch(hii, t_in, basis, helpers, integrals, triton_matvec=True)
        assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_streamed_singles_triton_flag_falls_back_bit_exactly_on_cpu(dtype) -> None:
    """The singles call site falls back bit-exactly on the streamed pair matvec."""
    basis, helpers, integrals, hii, x = _pair_fixture(dtype, "cpu")
    assert _pair_single_tables(helpers)
    want = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)
    got = mult_pair_torch(hii, x, basis, helpers, NORB, integrals, triton_matvec=True)
    assert torch.equal(got, want)

    for row_start, row_stop in ((0, 10), (7, 23), (0, basis.n_pairs)):
        want_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
        )
        got_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
            triton_matvec=True,
        )
        assert torch.equal(got_local, want_local)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_spinless_triton_solver_energy_bit_exact_on_cpu(dtype) -> None:
    """The spinless solver energy is bit-exact with the singles flag on."""
    hcore, eri = (
        _complex_integrals(NMO, seed=5) if dtype.is_complex else _real_integrals(NMO, seed=5)
    )
    strs = np.asarray(_comb_strs(NMO, N_PARTICLES), dtype=np.int64)
    common = dict(device="cpu", dtype=dtype, max_davidson_iters=100)
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)([strs], hcore, eri, NMO, N_PARTICLES)[0]
        .states[0]
        .energy
    )
    e_triton = (
        TorchSCISolver(matvec_impl="triton", **common)([strs], hcore, eri, NMO, N_PARTICLES)[0]
        .states[0]
        .energy
    )
    assert e_triton == e_batched


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_gdb_streamed_triton_solver_energy_bit_exact_on_cpu(dtype) -> None:
    """The streamed-GDB solver energy is bit-exact with the singles flag on."""
    h1, eri = _complex_integrals(NORB, seed=17) if dtype.is_complex else _real_integrals(NORB, 17)
    ci = [(_comb_strs(NORB, N_A), _comb_strs(NORB, N_B))]
    common = dict(
        device="cpu",
        dtype=dtype,
        mode="gdb",
        gdb_kwargs={"block_matvec": "off"},
        max_davidson_iters=100,
    )
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)(ci, h1, eri, NORB, (N_A, N_B))[0]
        .states[0]
        .energy
    )
    e_triton = (
        TorchSCISolver(matvec_impl="triton", **common)(ci, h1, eri, NORB, (N_A, N_B))[0]
        .states[0]
        .energy
    )
    assert e_triton == e_batched


# ============================================================================
# CUDA: the kernel itself
# ============================================================================


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_spinless_double_entries_kernel_matches_batched(dtype) -> None:
    """The entry kernel reproduces the batched spinless doubles scatter."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _spinless_fixture(dtype, device)
    tables = list(helpers.doubles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert (
            apply_double_entries(
                got,
                x,
                integrals,
                k_bra=table["k_bra"],
                k_ket=table["k_ket"],
                cr0=table["cr0"],
                cr1=table["cr1"],
                an0=table["an0"],
                an1=table["an1"],
                mode="spinless",
                dets=basis.dets,
            )
            is True
        )
    want = torch.zeros_like(got)
    for table in tables:
        _scatter_double_spinless(
            Wb=want,
            T=x,
            dets=basis.dets,
            k_bra=table["k_bra"],
            k_ket=table["k_ket"],
            cr0=table["cr0"],
            cr1=table["cr1"],
            an0=table["an0"],
            an1=table["an1"],
            integrals=integrals,
            chunk_size=10**9,
        )
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_streamed_double_entries_kernel_matches_batched(dtype) -> None:
    """The entry kernel reproduces the batched pair doubles scatter."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _pair_fixture(dtype, device)
    tables = list(helpers.adoubles()) + list(helpers.bdoubles()) + list(helpers.ab_doubles())
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    for table in tables:
        if "cr_a" in table:
            cr0, cr1, an0, an1 = table["cr_a"], table["cr_b"], table["an_a"], table["an_b"]
        else:
            cr0, cr1, an0, an1 = table["cr0"], table["cr1"], table["an0"], table["an1"]
        _scatter_double_pair(
            Wb=want,
            T=x,
            alpha_of_pair=basis.alpha_of_pair,
            beta_of_pair=basis.beta_of_pair,
            k_bra=table["k_bra"],
            k_ket=table["k_ket"],
            cr0=cr0,
            cr1=cr1,
            an0=an0,
            an1=an1,
            n_orb=NORB,
            integrals=integrals,
            chunk_size=10**9,
        )
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_spinless_single_entries_kernel_matches_batched(dtype) -> None:
    """The singles entry kernel reproduces the batched spinless singles scatter."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _spinless_fixture(dtype, device)
    tables = list(helpers.singles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert _apply_spinless_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_spinless_singles(want, x, integrals, basis, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_streamed_single_entries_kernel_matches_batched(dtype) -> None:
    """The singles entry kernel reproduces the batched pair singles scatter."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _pair_fixture(dtype, device)
    tables = _pair_single_tables(helpers)
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_pair_singles(want, x, integrals, basis, NORB, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_compact_eri_spinless_singles_kernel_matches_batched(dtype) -> None:
    """The singles kernel reads the compact spatial store at ``n_spinorb = 64``."""
    _skip_without_triton()
    device = _cuda()
    integrals = _compact_bundle(NMO_COMPACT // 2, dtype, device)
    basis, helpers = _spinless_case(NMO_COMPACT, _DETS64, device)
    x = _rand_vector(basis.dim, dtype, device)
    tables = list(helpers.singles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert _apply_spinless_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_spinless_singles(want, x, integrals, basis, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_compact_eri_pair_singles_kernel_matches_batched(dtype) -> None:
    """The singles kernel reads the compact spatial store on the pair path."""
    _skip_without_triton()
    device = _cuda()
    integrals = _compact_bundle(_N_ORB_COMPACT, dtype, device)
    basis, helpers = _pair_case(_N_ORB_COMPACT, _STRINGS32, device)
    x = _rand_vector(basis.n_pairs, dtype, device)
    tables = _pair_single_tables(helpers)
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_pair_singles(want, x, integrals, basis, _N_ORB_COMPACT, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("nmo", "dets"), [(70, _DETS70), (130, _DETS130)], ids=["70spinorb", "130spinorb"]
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_spinless_singles_limb_strings_match_batched(nmo, dets, dtype) -> None:
    """Limb-matrix ket strings reproduce the batched singles scatter past 64 bits."""
    _skip_without_triton()
    device = _cuda()
    if nmo == 70:
        integrals = _dense_spinless_bundle(nmo, dtype, device)
    else:
        spatial = _low_rank_eri(nmo // 2, seed=43, is_complex=dtype.is_complex)
        integrals = _manual_compact_bundle(spatial, dtype, device)
    basis, helpers = _spinless_case(nmo, dets, device)
    assert basis.dets.shape[1] == (nmo + 63) // 64
    x = _rand_vector(basis.dim, dtype, device)
    tables = list(helpers.singles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert _apply_spinless_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_spinless_singles(want, x, integrals, basis, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("n_orb", "strings"),
    [(35, _STRINGS35), (65, _STRINGS65)],
    ids=["70spinorb", "130spinorb"],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_pair_singles_limb_strings_match_batched(n_orb, strings, dtype) -> None:
    """Pair α/β strings cross the limb boundary and singles still match."""
    _skip_without_triton()
    device = _cuda()
    if n_orb <= 64:
        integrals = _dense_pair_bundle(n_orb, dtype, device)
    else:
        spatial = _low_rank_eri(n_orb, seed=48, is_complex=dtype.is_complex)
        integrals = _manual_compact_bundle(spatial, dtype, device)
    basis, helpers = _pair_case(n_orb, strings, device)
    assert basis.alpha_of_pair.dim() == (1 if n_orb <= 64 else 2)
    x = _rand_vector(basis.n_pairs, dtype, device)
    tables = _pair_single_tables(helpers)
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_single_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_pair_singles(want, x, integrals, basis, n_orb, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_spinless_singles_block_input_matches_batched(dtype) -> None:
    """Singles ``(N, K)`` blocks scatter per column, including an offset block."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _spinless_fixture(dtype, device)
    T = torch.stack([x, 2 * x, -0.5 * x], dim=1)
    tables = list(helpers.singles())
    assert tables

    got = torch.zeros_like(T)
    for table in tables:
        assert _apply_spinless_single_table(got, T, integrals, basis, table) is True
    want = torch.zeros_like(T)
    _scatter_spinless_singles(want, T, integrals, basis, tables)
    _assert_close(got, want, dtype)

    start, stop = 3, basis.dim - 2
    local = torch.zeros((stop - start, T.shape[1]), dtype=dtype, device=device)
    for table in tables:
        keep = (table["k_bra"] >= start) & (table["k_bra"] < stop)
        filtered = {key: value[keep] for key, value in table.items()}
        assert (
            _apply_spinless_single_table(local, T, integrals, basis, filtered, offset=start) is True
        )
    _assert_close(local, want[start:stop], dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_streamed_singles_block_input_matches_batched(dtype) -> None:
    """Singles ``(N, K)`` blocks scatter per column on the pair path too."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _pair_fixture(dtype, device)
    T = torch.stack([x, 2 * x, -0.5 * x], dim=1)
    tables = _pair_single_tables(helpers)
    assert tables

    got = torch.zeros_like(T)
    for table in tables:
        assert _apply_pair_single_table(got, T, integrals, basis, table) is True
    want = torch.stack(
        [
            _pair_single_column_reference(T[:, col], integrals, basis, NORB, tables)
            for col in range(T.shape[1])
        ],
        dim=1,
    )
    _assert_close(got, want, dtype)

    start, stop = 5, basis.n_pairs - 3
    local = torch.zeros((stop - start, T.shape[1]), dtype=dtype, device=device)
    for table in tables:
        keep = (table["k_bra"] >= start) & (table["k_bra"] < stop)
        filtered = {key: value[keep] for key, value in table.items()}
        assert _apply_pair_single_table(local, T, integrals, basis, filtered, offset=start) is True
    _assert_close(local, want[start:stop], dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_compact_eri_spinless_kernel_matches_batched(dtype) -> None:
    """The entry kernel reads the compact spatial store at ``n_spinorb = 64``."""
    _skip_without_triton()
    device = _cuda()
    integrals = _compact_bundle(NMO_COMPACT // 2, dtype, device)
    basis, helpers = _spinless_case(NMO_COMPACT, _DETS64, device)
    x = _rand_vector(basis.dim, dtype, device)
    tables = list(helpers.doubles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert _apply_spinless_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_spinless_tables(want, x, integrals, basis, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_compact_eri_pair_kernel_matches_batched(dtype) -> None:
    """The entry kernel reads the compact spatial store on the pair path."""
    _skip_without_triton()
    device = _cuda()
    integrals = _compact_bundle(_N_ORB_COMPACT, dtype, device)
    basis, helpers = _pair_case(_N_ORB_COMPACT, _STRINGS32, device)
    x = _rand_vector(basis.n_pairs, dtype, device)
    tables = _pair_tables(helpers)
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_pair_tables(want, x, integrals, basis, _N_ORB_COMPACT, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("nmo", "dets"), [(70, _DETS70), (130, _DETS130)], ids=["70spinorb", "130spinorb"]
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_spinless_limb_strings_match_batched(nmo, dets, dtype) -> None:
    """Limb-matrix ket strings reproduce the batched scatter past 64 bits."""
    _skip_without_triton()
    device = _cuda()
    if nmo == 70:
        integrals = _dense_spinless_bundle(nmo, dtype, device)
    else:
        # A dense (2L)^4 store is several gigabytes at 130 spin-orbitals; the
        # compact store exercises the same limb parsing and also covers the
        # compact + complex + limb combination.
        spatial = _low_rank_eri(nmo // 2, seed=43, is_complex=dtype.is_complex)
        integrals = _manual_compact_bundle(spatial, dtype, device)
    basis, helpers = _spinless_case(nmo, dets, device)
    assert basis.dets.shape[1] == (nmo + 63) // 64
    x = _rand_vector(basis.dim, dtype, device)
    tables = list(helpers.doubles())
    assert tables

    got = torch.zeros(basis.dim, dtype=dtype, device=device)
    for table in tables:
        assert _apply_spinless_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_spinless_tables(want, x, integrals, basis, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("n_orb", "strings"),
    [(35, _STRINGS35), (65, _STRINGS65)],
    ids=["70spinorb", "130spinorb"],
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128], ids=["fp64", "c128"])
def test_pair_limb_strings_match_batched(n_orb, strings, dtype) -> None:
    """Pair α/β strings cross the one- and two-limb boundary and still match."""
    _skip_without_triton()
    device = _cuda()
    if n_orb <= 64:
        # n_spinorb = 70 with single-word α/β strings.
        integrals = _dense_pair_bundle(n_orb, dtype, device)
    else:
        spatial = _low_rank_eri(n_orb, seed=48, is_complex=dtype.is_complex)
        integrals = _manual_compact_bundle(spatial, dtype, device)
    basis, helpers = _pair_case(n_orb, strings, device)
    assert basis.alpha_of_pair.dim() == (1 if n_orb <= 64 else 2)
    x = _rand_vector(basis.n_pairs, dtype, device)
    tables = _pair_tables(helpers)
    assert tables

    got = torch.zeros(basis.n_pairs, dtype=dtype, device=device)
    for table in tables:
        assert _apply_pair_table(got, x, integrals, basis, table) is True
    want = torch.zeros_like(got)
    _scatter_pair_tables(want, x, integrals, basis, n_orb, tables)
    _assert_close(got, want, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_spinless_block_input_matches_batched(dtype) -> None:
    """``(N, K)`` blocks scatter per column, including an owner-computes block."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _spinless_fixture(dtype, device)
    T = torch.stack([x, 2 * x, -0.5 * x], dim=1)
    tables = list(helpers.doubles())
    assert tables

    got = torch.zeros_like(T)
    for table in tables:
        assert _apply_spinless_table(got, T, integrals, basis, table) is True
    want = torch.zeros_like(T)
    _scatter_spinless_tables(want, T, integrals, basis, tables)
    _assert_close(got, want, dtype)

    start, stop = 3, basis.dim - 2
    local = torch.zeros((stop - start, T.shape[1]), dtype=dtype, device=device)
    for table in tables:
        keep = (table["k_bra"] >= start) & (table["k_bra"] < stop)
        filtered = {key: value[keep] for key, value in table.items()}
        assert _apply_spinless_table(local, T, integrals, basis, filtered, offset=start) is True
    _assert_close(local, want[start:stop], dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_streamed_block_input_matches_batched(dtype) -> None:
    """``(N, K)`` blocks scatter per column on the pair path too."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _pair_fixture(dtype, device)
    T = torch.stack([x, 2 * x, -0.5 * x], dim=1)
    tables = _pair_tables(helpers)
    assert tables

    got = torch.zeros_like(T)
    for table in tables:
        assert _apply_pair_table(got, T, integrals, basis, table) is True
    # The batched pair scatter takes a 1-D vector, so the block reference is
    # one column at a time; the kernel's per-column update must equal it.
    want = torch.stack(
        [
            _pair_column_reference(T[:, col], integrals, basis, NORB, tables)
            for col in range(T.shape[1])
        ],
        dim=1,
    )
    _assert_close(got, want, dtype)

    start, stop = 5, basis.n_pairs - 3
    local = torch.zeros((stop - start, T.shape[1]), dtype=dtype, device=device)
    for table in tables:
        keep = (table["k_bra"] >= start) & (table["k_bra"] < stop)
        filtered = {key: value[keep] for key, value in table.items()}
        assert _apply_pair_table(local, T, integrals, basis, filtered, offset=start) is True
    _assert_close(local, want[start:stop], dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_spinless_matvec_triton_matches_batched(dtype) -> None:
    """The public spinless matvec with ``triton_matvec=True`` matches batched."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, hii, x = _spinless_fixture(dtype, device)
    want = mult_spinless_torch(hii, x, basis, helpers, integrals)
    got = mult_spinless_torch(hii, x, basis, helpers, integrals, triton_matvec=True)
    _assert_close(got, want, dtype)

    # A block input runs through the kernel's per-column loop.
    T_block = torch.stack([x, 2 * x], dim=1)
    want_block = mult_spinless_torch(hii, T_block, basis, helpers, integrals)
    got_block = mult_spinless_torch(hii, T_block, basis, helpers, integrals, triton_matvec=True)
    _assert_close(got_block, want_block, dtype)

    # The compact spatial store flows through the same public matvec.
    compact = _compact_bundle(NMO // 2, dtype, device)
    hii_compact = make_diag_terms_spinless_torch(basis, compact)
    for T_in in (x, T_block):
        want_compact = mult_spinless_torch(hii_compact, T_in, basis, helpers, compact)
        got_compact = mult_spinless_torch(
            hii_compact, T_in, basis, helpers, compact, triton_matvec=True
        )
        _assert_close(got_compact, want_compact, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_pair_streamed_matvec_triton_matches_batched(dtype) -> None:
    """The public streamed pair matvec and its owner-computes variant match batched."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, hii, x = _pair_fixture(dtype, device)
    want = mult_pair_torch(hii, x, basis, helpers, NORB, integrals)
    got = mult_pair_torch(hii, x, basis, helpers, NORB, integrals, triton_matvec=True)
    _assert_close(got, want, dtype)

    n_pairs = basis.n_pairs
    for row_start, row_stop in ((0, 10), (7, 23), (0, n_pairs)):
        want_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
        )
        got_local = mult_pair_local_torch(
            hii[row_start:row_stop],
            x[row_start:row_stop],
            x,
            basis,
            helpers,
            NORB,
            integrals,
            row_start,
            triton_matvec=True,
        )
        _assert_close(got_local, want_local, dtype)

    # The compact spatial store flows through the streamed pair matvec too.
    compact = _compact_bundle(NORB, dtype, device)
    hii_compact = make_diag_terms_pair_torch(basis, NORB, compact)
    want_compact = mult_pair_torch(hii_compact, x, basis, helpers, NORB, compact)
    got_compact = mult_pair_torch(hii_compact, x, basis, helpers, NORB, compact, triton_matvec=True)
    _assert_close(got_compact, want_compact, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("dtype", "tol"),
    [
        (torch.float64, 1e-8),
        (torch.complex128, 1e-8),
        (torch.float32, 1e-5),
        (torch.complex64, 1e-5),
    ],
    ids=["fp64", "c128", "fp32", "c64"],
)
def test_solver_triton_matches_batched_on_cuda(dtype, tol) -> None:
    """Spinless and streamed-GDB solves with explicit Triton match batched."""
    _skip_without_triton()
    device = _cuda()

    hcore, eri = (
        _complex_integrals(NMO, seed=5) if dtype.is_complex else _real_integrals(NMO, seed=5)
    )
    strs = np.asarray(_comb_strs(NMO, N_PARTICLES), dtype=np.int64)
    common = dict(device=device, dtype=dtype, max_davidson_iters=200)
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)([strs], hcore, eri, NMO, N_PARTICLES)[0]
        .states[0]
        .energy
    )
    e_triton = (
        TorchSCISolver(matvec_impl="triton", **common)([strs], hcore, eri, NMO, N_PARTICLES)[0]
        .states[0]
        .energy
    )
    assert e_triton == pytest.approx(e_batched, abs=tol)

    h1, eri = _complex_integrals(NORB, seed=17) if dtype.is_complex else _real_integrals(NORB, 17)
    ci = [(_comb_strs(NORB, N_A), _comb_strs(NORB, N_B))]
    common = dict(
        device=device,
        dtype=dtype,
        mode="gdb",
        gdb_kwargs={"block_matvec": "off"},
        max_davidson_iters=200,
    )
    e_batched = (
        TorchSCISolver(matvec_impl="batched", **common)(ci, h1, eri, NORB, (N_A, N_B))[0]
        .states[0]
        .energy
    )
    e_triton = (
        TorchSCISolver(matvec_impl="triton", **common)(ci, h1, eri, NORB, (N_A, N_B))[0]
        .states[0]
        .energy
    )
    assert e_triton == pytest.approx(e_batched, abs=tol)


@pytest.mark.gpu
def test_apply_double_entries_rejects_malformed_layouts_on_gpu() -> None:
    """Malformed layouts decline on CUDA instead of launching a wrong element."""
    _skip_without_triton()
    device = _cuda()
    basis, helpers, integrals, _hii, x = _spinless_fixture(torch.float64, device)
    table = next(iter(helpers.doubles()))
    Wb = torch.zeros(basis.dim, dtype=torch.float64, device=device)
    columns = _table_columns(table)

    # ``eri_compact`` set on a bundle whose ERI is a dense tensor: there is no
    # spatial buffer for the kernel to decode.
    bad_eri = replace(integrals, eri_compact=True)
    assert (
        apply_double_entries(Wb, x, bad_eri, mode="spinless", dets=basis.dets, **columns) is False
    )
    # String rows that do not match the input vector: the kernel indexes
    # ``dets[k_ket]``, so a short determinant table is malformed (a short
    # entry table is legitimate chunking and stays supported).
    assert (
        apply_double_entries(Wb, x, integrals, mode="spinless", dets=basis.dets[:2], **columns)
        is False
    )
    # A 2-D input block with a 1-D output vector.
    block = torch.stack([x, -x], dim=1)
    assert (
        apply_double_entries(Wb, block, integrals, mode="spinless", dets=basis.dets, **columns)
        is False
    )
