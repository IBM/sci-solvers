"""``hamiltonian_backend="fulqrum"`` on the general-algebra solvers.

The fulqrum backend only swaps what builds each batch's projected Hamiltonian, so
every result must match the default torch backend: energies, RDMs and amplitudes,
real and complex integrals, multi-root, spin symmetrization and spin filtering. Also
covers ``mode="gdb"`` (the pair operator must match ``PairCIProblem``'s entrywise),
the spinless path (against the brute-force ladder-operator oracle and the torch
spinless path, on a genuinely complex spin-orbital ERI), the backend's rejections
(``pyscf_davidson``, a non-CPU device, a missing ``fulqrum`` install), its
operator cache, and the three ``fulqrum_mode`` values (``csr``, ``csr_fast``,
``matrix_free``) in every basis, against the reference operator and end to end. Tests
that build a fulqrum operator are skipped unless the optional ``fulqrum`` extra is
installed.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import importlib.util

import numpy as np
import pytest
from test_complex_tpb_double_excitations import _full_strs, _genuinely_complex_integrals

from sci_solvers.energy_solvers.factory import create_solver

requires_fulqrum = pytest.mark.skipif(
    importlib.util.find_spec("fulqrum") is None, reason="fulqrum extra not installed"
)

NORB = 5
NELEC = (2, 2)
# Dependency-free, complex-capable general-algebra solvers.
SOLVERS = ["arpack", "davidson", "lobpcg"]


def _real_integrals(norb: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Real symmetric ``hcore`` and a real 8-fold-symmetric chemist ERI."""
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((norb, norb))
    eri = np.zeros((norb,) * 4)
    for _ in range(norb + 2):
        g = rng.standard_normal((norb, norb))
        eri += np.einsum("pq,rs->pqrs", g + g.T, g + g.T)
    return h + h.T, eri


def _integrals(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Real, or genuinely complex Hermitian, integrals for ``NORB`` orbitals."""
    return _real_integrals(NORB) if kind == "real" else _genuinely_complex_integrals(NORB)


def _selected_strs(*, ascending: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """A selected product subset (dim 72, above arpack's dense cutoff).

    Ascending by default, as ``bitstring_matrix_to_ci_strs`` produces and as the
    general-algebra solvers require (PySCF ``selected_ci`` needs it).
    """
    strs_a, strs_b = _full_strs(NORB, NELEC)
    rng = np.random.default_rng(0)
    strs_a, strs_b = rng.permutation(strs_a)[:8], rng.permutation(strs_b)[:9]
    return (np.sort(strs_a), np.sort(strs_b)) if ascending else (strs_a, strs_b)


def _solve(name: str, backend: str, batch, hcore, eri, **kwargs):
    """Every root of one batch, from ``name`` with the given backend."""
    solver = create_solver(name, symmetrize_spin=False, hamiltonian_backend=backend, **kwargs)
    return solver.sci_solver_callable_eigen_states([batch], hcore, eri, NORB, NELEC)[0].states


def _assert_same_states(got, ref, *, atol: float = 1e-7) -> None:
    """Same energies and RDMs root by root, and the same ground state up to a phase."""
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.energy == pytest.approx(r.energy, abs=atol)
        np.testing.assert_allclose(g.rdm1, r.rdm1, atol=1e-5)
    overlap = np.vdot(np.ravel(ref[0].sci_state.amplitudes), np.ravel(got[0].sci_state.amplitudes))
    assert abs(overlap) == pytest.approx(1.0, abs=1e-6)


@requires_fulqrum
@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("name", SOLVERS)
def test_fulqrum_backend_matches_torch(name: str, kind: str) -> None:
    """Two roots on a selected subset: energies, RDMs and amplitudes match torch."""
    hcore, eri = _integrals(kind)
    batch = _selected_strs()
    ref = _solve(name, "torch", batch, hcore, eri, nroots=2)
    got = _solve(name, "fulqrum", batch, hcore, eri, nroots=2)
    _assert_same_states(got, ref)
    np.testing.assert_array_equal(got[0].sci_state.ci_strs_a, batch[0])
    np.testing.assert_array_equal(got[0].sci_state.ci_strs_b, batch[1])


@requires_fulqrum
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_fulqrum_backend_rdm2_energy_matches_torch(kind: str) -> None:
    """With ``compute_rdm2`` the RDM-rebuilt energy and the 2-RDM match torch."""
    hcore, eri = _integrals(kind)
    batch = _selected_strs()
    ref = _solve("arpack", "torch", batch, hcore, eri, compute_rdm2=True)
    got = _solve("arpack", "fulqrum", batch, hcore, eri, compute_rdm2=True)
    _assert_same_states(got, ref)
    np.testing.assert_allclose(got[0].rdm2, ref[0].rdm2, atol=1e-5)


@requires_fulqrum
def test_fulqrum_backend_rejects_unsorted_strings() -> None:
    """The base's ascending-strings guard applies to the fulqrum backend too."""
    hcore, eri = _real_integrals(NORB)
    with pytest.raises(ValueError, match="must be strictly ascending"):
        _solve("arpack", "fulqrum", _selected_strs(ascending=False), hcore, eri)


@requires_fulqrum
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_fulqrum_builder_keeps_unsorted_input_order(kind: str) -> None:
    """The builder maps fulqrum's internally sorted basis back to the caller's order.

    Direct builder check (the solvers reject unsorted strings up front): on
    unsorted strings the fulqrum matrix equals the torch builder's entrywise.
    """
    from sci_solvers.complex_ci_problem import build_selected_complex_ci_problem
    from sci_solvers.eigen_problem import EigenProblem
    from sci_solvers.fulqrum_ci_problem import build_fulqrum_ci_problem

    hcore, eri = _integrals(kind)
    strs_a, strs_b = _selected_strs(ascending=False)
    got = build_fulqrum_ci_problem(strs_a, strs_b, hcore, eri, NORB, NELEC)
    if kind == "real":
        ref = EigenProblem(strs_a, strs_b, hcore, eri, NORB, NELEC)
    else:
        ref = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, NORB, NELEC)
    np.testing.assert_allclose(got.sparse().toarray(), ref.sparse().toarray(), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(got.strs_a, strs_a)


@requires_fulqrum
def test_fulqrum_backend_symmetrize_spin_matches_torch() -> None:
    """``symmetrize_spin`` (packed sector view) works on the fulqrum operator."""
    hcore, eri = _real_integrals(NORB, seed=1)
    strs = np.sort(np.random.default_rng(1).permutation(_full_strs(NORB, NELEC)[0])[:8])
    energies = {}
    for backend in ("torch", "fulqrum"):
        solver = create_solver(
            "arpack", symmetrize_spin=True, nroots=2, hamiltonian_backend=backend
        )
        states = solver.sci_solver_callable_eigen_states([(strs, strs)], hcore, eri, NORB, NELEC)[
            0
        ].states
        energies[backend] = [s.energy for s in states]
    np.testing.assert_allclose(energies["fulqrum"], energies["torch"], atol=1e-7)


@requires_fulqrum
def test_fulqrum_backend_spin_filter_matches_torch() -> None:
    """The ``spin_sq`` filter (SpinFilterMixin) keeps the same roots on either backend."""
    hcore, eri = _real_integrals(NORB, seed=2)
    batch = _full_strs(NORB, NELEC)
    ref = _solve("arpack", "torch", batch, hcore, eri, nroots=2, spin_sq=0.0)
    got = _solve("arpack", "fulqrum", batch, hcore, eri, nroots=2, spin_sq=0.0)
    np.testing.assert_allclose([s.energy for s in got], [s.energy for s in ref], atol=1e-7)


@requires_fulqrum
def test_fulqrum_operator_is_cached_across_batches() -> None:
    """The fulqrum operator setup is built once per Hamiltonian and reused across batches."""
    from sci_solvers import fulqrum_ci_problem

    hcore, eri = _real_integrals(NORB, seed=3)
    strs_a, strs_b = _full_strs(NORB, NELEC)
    solver = create_solver("arpack", symmetrize_spin=False, hamiltonian_backend="fulqrum")
    solver.sci_solver_callable(
        [(strs_a[:5], strs_b[:6]), (strs_a[2:], strs_b[1:])], hcore, eri, NORB, NELEC
    )
    hsub = fulqrum_ci_problem.fulqrum_subspace_hamiltonian(hcore.copy(), eri.copy())
    assert list(fulqrum_ci_problem._HSUB_CACHE.values()) == [hsub]
    other = fulqrum_ci_problem.fulqrum_subspace_hamiltonian(hcore + 1.0, eri)
    assert other is not hsub
    assert list(fulqrum_ci_problem._HSUB_CACHE.values()) == [other]


@requires_fulqrum
def test_fulqrum_builder_rejects_duplicate_strings() -> None:
    """The builder rejects duplicate CI strings (fulqrum would keep them as separate rows).

    Direct builder check: through a solver, the base's ascending-strings guard
    already rejects duplicates before the builder runs.
    """
    from sci_solvers.fulqrum_ci_problem import build_fulqrum_ci_problem

    hcore, eri = _real_integrals(NORB)
    strs_a, strs_b = _full_strs(NORB, NELEC)
    dup = np.concatenate([strs_a[:3], strs_a[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        build_fulqrum_ci_problem(dup, strs_b, hcore, eri, NORB, NELEC)


@requires_fulqrum
def test_fulqrum_backend_rejects_non_hermitian_complex_integrals() -> None:
    """Complex integrals get the same Hermiticity guard as the torch complex builder."""
    hcore, eri = _genuinely_complex_integrals(NORB)
    eri = eri.copy()
    eri[0, 1, 2, 3] += 0.5j
    with pytest.raises(ValueError, match="Hermitian|does not satisfy"):
        _solve("arpack", "fulqrum", _full_strs(NORB, NELEC), hcore, eri)


def _pairs() -> tuple[np.ndarray, np.ndarray]:
    """A GDB pair list (70 unique pairs), unsorted and with repeated pairs."""
    strs_a, strs_b = _full_strs(NORB, NELEC)
    rng = np.random.default_rng(4)
    pairs = np.array([(a, b) for a in strs_a for b in strs_b], dtype=np.int64)
    pairs = pairs[rng.choice(len(pairs), 70, replace=False)]
    pairs = np.concatenate([pairs, pairs[:5]])
    return pairs[:, 0], pairs[:, 1]


@requires_fulqrum
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_fulqrum_pair_problem_matches_pair_kernel(kind: str) -> None:
    """GDB: the fulqrum pair operator equals ``PairCIProblem``'s entrywise.

    That includes the interleaved-Wick signs, for real and complex integrals.
    """
    from sci_solvers.eigen_problem import PairCIProblem
    from sci_solvers.fulqrum_ci_problem import FulqrumPairCIProblem

    hcore, eri = _integrals(kind)
    pa, pb = _pairs()
    ref = PairCIProblem(pa, pb, hcore, eri, NORB, NELEC)
    got = FulqrumPairCIProblem(pa, pb, hcore, eri, NORB, NELEC)
    assert got.dim == ref.dim == 70
    np.testing.assert_array_equal(got.strs_a, ref.strs_a)
    np.testing.assert_array_equal(got.strs_b, ref.strs_b)
    np.testing.assert_allclose(got.sparse().toarray(), ref.sparse().toarray(), atol=1e-10, rtol=0)
    np.testing.assert_allclose(got.hdiag(), ref.hdiag(), atol=1e-10)


@requires_fulqrum
@pytest.mark.parametrize("kind", ["real", "complex"])
@pytest.mark.parametrize("name", SOLVERS)
def test_fulqrum_backend_gdb_matches_torch(name: str, kind: str) -> None:
    """GDB end to end: energies, RDMs and amplitudes match the torch pair backend."""
    hcore, eri = _integrals(kind)
    batch = _pairs()
    ref = _solve(name, "torch", batch, hcore, eri, mode="gdb", nroots=2, compute_rdm2=True)
    got = _solve(name, "fulqrum", batch, hcore, eri, mode="gdb", nroots=2, compute_rdm2=True)
    _assert_same_states(got, ref)
    np.testing.assert_allclose(got[0].rdm2, ref[0].rdm2, atol=1e-5)
    np.testing.assert_array_equal(got[0].sci_state.ci_strs_a, ref[0].sci_state.ci_strs_a)


@requires_fulqrum
def test_fulqrum_backend_gdb_rejects_non_hermitian_complex_integrals() -> None:
    """The GDB fulqrum path runs the complex Hermiticity guard (``PairCIProblem`` does not)."""
    hcore, eri = _genuinely_complex_integrals(NORB)
    eri = eri.copy()
    eri[0, 1, 2, 3] += 0.5j
    with pytest.raises(ValueError, match="Hermitian|does not satisfy"):
        _solve("arpack", "fulqrum", _pairs(), hcore, eri, mode="gdb")


@requires_fulqrum
@pytest.mark.parametrize("subset", ["full", "selected", "unsorted"])
@pytest.mark.parametrize("nmo, n, seed", [(6, 3, 1), (8, 4, 3)])
def test_fulqrum_spinless_problem_matches_brute_force(
    nmo: int, n: int, seed: int, subset: str
) -> None:
    """Spinless: the fulqrum operator equals the brute-force ladder-operator one entrywise.

    On a genuinely complex spin-orbital ERI, over the full canonical list, a selected
    subset, and the subset in a scrambled order (the matrix follows the input order).
    """
    from test_spinless_complex_ci import _all_dets, _brute_force_dense
    from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

    from sci_solvers.fulqrum_ci_problem import build_fulqrum_spinless_ci_problem

    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed)
    dets = _all_dets(nmo, n)
    rng = np.random.default_rng(seed)
    if subset != "full":
        dets = sorted(rng.choice(dets, len(dets) * 2 // 3, replace=False).tolist())
    if subset == "unsorted":
        dets = rng.permutation(dets).tolist()
    got = build_fulqrum_spinless_ci_problem(hcore, eri, n, np.array(dets, dtype=np.int64))
    np.testing.assert_allclose(
        got.sparse().toarray(), _brute_force_dense(hcore, eri, nmo, dets), atol=1e-10, rtol=0
    )
    assert got.nelec == (n, 0) and got.norb == nmo
    assert [int(x) for x in got.strs] == [int(x) for x in dets]


@requires_fulqrum
@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("selected", [False, True], ids=["full", "selected"])
def test_fulqrum_backend_spinless_matches_torch(name: str, selected: bool) -> None:
    """Spinless end to end: energies, RDMs and amplitudes match the torch spinless path."""
    from test_spinless_complex_ci import _all_dets
    from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

    nmo, n = 8, 4
    hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
    dets = _all_dets(nmo, n)
    if selected:
        dets = sorted(np.random.default_rng(9).choice(dets, 46, replace=False).tolist())
    strs = np.array(dets, dtype=np.int64)
    out = {}
    for backend in ("torch", "fulqrum"):
        solver = create_solver(
            name, symmetrize_spin=False, nroots=2, compute_rdm2=True, hamiltonian_backend=backend
        )
        out[backend] = solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, n)[0].states
    _assert_same_states(out["fulqrum"], out["torch"])
    np.testing.assert_allclose(out["fulqrum"][0].rdm2, out["torch"][0].rdm2, atol=1e-5)


@requires_fulqrum
def test_fulqrum_spinless_rejects_bad_input() -> None:
    """Spinless guards: real integrals, duplicates, a wrong particle number, non-Hermitian ERI."""
    from test_spinless_complex_ci import _all_dets
    from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

    from sci_solvers.fulqrum_ci_problem import build_fulqrum_spinless_ci_problem

    hcore, eri = _genuinely_complex_spinorbital_integrals(6, seed=1)
    dets = np.array(_all_dets(6, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="real integrals"):
        build_fulqrum_spinless_ci_problem(hcore.real, eri.real, 3, dets)
    with pytest.raises(ValueError, match="duplicate"):
        build_fulqrum_spinless_ci_problem(hcore, eri, 3, np.concatenate([dets, dets[:1]]))
    with pytest.raises(ValueError, match="exactly 2 particles"):
        build_fulqrum_spinless_ci_problem(hcore, eri, 2, dets)
    bad = eri.copy()
    bad[0, 1, 2, 3] += 0.5j
    with pytest.raises(ValueError, match="Hermitian|does not satisfy"):
        build_fulqrum_spinless_ci_problem(hcore, bad, 3, dets)


def test_pyscf_davidson_rejects_fulqrum_backend() -> None:
    """``pyscf_davidson`` never reads the problem operator, so the backend is rejected."""
    with pytest.raises(ValueError, match="does not support hamiltonian_backend='fulqrum'"):
        create_solver("pyscf_davidson", symmetrize_spin=False, hamiltonian_backend="fulqrum")


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_fulqrum_backend_rejects_non_cpu_device(device: str, mode: str) -> None:
    """With ``device_fallback="raise"``, a GPU device is rejected at construction.

    Needs no GPU and no fulqrum: the check runs before anything is built.
    """
    with pytest.raises(ValueError, match="CPU-only"):
        create_solver(
            "arpack",
            symmetrize_spin=False,
            mode=mode,
            hamiltonian_backend="fulqrum",
            device=device,
            device_fallback="raise",
        )


@requires_fulqrum
@pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps"])
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_fulqrum_backend_falls_back_to_cpu_with_a_warning(device: str, mode: str) -> None:
    """By default a GPU device warns and the solver runs on CPU.

    The device is replaced before anything reads it, so ``matvec_impl="auto"``
    resolves for CPU too (it would pick Triton on a CUDA device). Needs no GPU.
    """
    with pytest.warns(UserWarning, match="CPU-only.*device='cpu' instead"):
        solver = create_solver(
            "arpack",
            symmetrize_spin=False,
            mode=mode,
            hamiltonian_backend="fulqrum",
            device=device,
            matvec_impl="auto",
        )
    assert solver.device == "cpu"
    assert solver.matvec_impl == "batched"


@requires_fulqrum
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_fulqrum_cpu_fallback_gives_the_cpu_results(mode: str) -> None:
    """A GPU request that falls back to CPU gives exactly the ``device="cpu"`` results."""
    hcore, eri = _real_integrals(NORB)
    batch = _pairs() if mode == "gdb" else _selected_strs()
    ref = _solve("arpack", "fulqrum", batch, hcore, eri, mode=mode, device="cpu")
    with pytest.warns(UserWarning, match="CPU-only"):
        got = _solve("arpack", "fulqrum", batch, hcore, eri, mode=mode, device="cuda")
    _assert_same_states(got, ref, atol=1e-12)


def test_device_fallback_rejects_unknown_values() -> None:
    """Only ``"warn"`` and ``"raise"`` are accepted."""
    with pytest.raises(ValueError, match="device_fallback must be"):
        create_solver("arpack", symmetrize_spin=False, device_fallback="ignore")


def test_device_fallback_leaves_the_torch_backend_alone() -> None:
    """The torch backend runs on GPUs, so ``device`` is kept and nothing is warned."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver = create_solver("arpack", symmetrize_spin=False, device="cuda")
    assert solver.device == "cuda"


@requires_fulqrum
def test_fulqrum_backend_accepts_explicit_cpu_device() -> None:
    """``device="cpu"`` (and a ``torch.device``) is accepted."""
    import torch

    for device in ("cpu", torch.device("cpu")):
        create_solver("arpack", symmetrize_spin=False, hamiltonian_backend="fulqrum", device=device)


def test_unknown_hamiltonian_backend_raises() -> None:
    """Only ``"torch"`` and ``"fulqrum"`` are accepted."""
    with pytest.raises(ValueError, match="hamiltonian_backend must be"):
        create_solver("arpack", symmetrize_spin=False, hamiltonian_backend="numpy")


def test_fulqrum_backend_without_fulqrum_raises_import_error(monkeypatch) -> None:
    """Without the extra installed, construction fails with an install hint."""
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "fulqrum" else find_spec(name, *args, **kwargs)
        ),
    )
    with pytest.raises(ImportError, match="uv sync --extra fulqrum"):
        create_solver("arpack", symmetrize_spin=False, hamiltonian_backend="fulqrum")


# ---- fulqrum_mode: "csr" (default), "csr_fast", "matrix_free" -------------------------
MODES = ["csr", "csr_fast", "matrix_free"]


def _dense_reference(kind: str, which: str):
    """The torch builder's dense operator for one basis (``tpb``, ``gdb`` or ``spinless``).

    Returns ``(dense, build)``, where ``build(mode)`` makes the fulqrum problem over the
    same basis: unsorted TPB strings, an unsorted GDB pair list with repeats, and a
    scrambled spinless determinant list, so every mode's basis permutation is exercised.
    """
    from sci_solvers.fulqrum_ci_problem import (
        FulqrumPairCIProblem,
        build_fulqrum_ci_problem,
        build_fulqrum_spinless_ci_problem,
    )

    if which == "spinless":
        from test_spinless_complex_ci import _all_dets, _brute_force_dense
        from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

        nmo, n = 8, 4
        hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
        dets = np.random.default_rng(2).permutation(_all_dets(nmo, n))[:46].tolist()
        strs = np.array(dets, dtype=np.int64)
        return _brute_force_dense(
            hcore, eri, nmo, dets
        ), lambda m: build_fulqrum_spinless_ci_problem(hcore, eri, n, strs, fulqrum_mode=m)
    hcore, eri = _integrals(kind)
    if which == "gdb":
        from sci_solvers.eigen_problem import PairCIProblem

        pa, pb = _pairs()
        return PairCIProblem(
            pa, pb, hcore, eri, NORB, NELEC
        ).sparse().toarray(), lambda m: FulqrumPairCIProblem(
            pa, pb, hcore, eri, NORB, NELEC, fulqrum_mode=m
        )
    from sci_solvers.complex_ci_problem import build_selected_complex_ci_problem
    from sci_solvers.eigen_problem import EigenProblem

    strs_a, strs_b = _selected_strs(ascending=False)
    if kind == "real":
        ref = EigenProblem(strs_a, strs_b, hcore, eri, NORB, NELEC)
    else:
        ref = build_selected_complex_ci_problem(strs_a, strs_b, hcore, eri, NORB, NELEC)
    return ref.sparse().toarray(), lambda m: build_fulqrum_ci_problem(
        strs_a, strs_b, hcore, eri, NORB, NELEC, fulqrum_mode=m
    )


@requires_fulqrum
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "kind, which",
    [
        ("real", "tpb"),
        ("complex", "tpb"),
        ("real", "gdb"),
        ("complex", "gdb"),
        ("complex", "spinless"),
    ],
)
def test_fulqrum_mode_operator_matches_reference(kind: str, which: str, mode: str) -> None:
    """Every mode applies the reference operator, in the caller's basis order.

    Checked on single vectors and a block (the column-wise path) and on the diagonal,
    which the preconditioners read; the CSR modes' ``sparse()`` too (the matrix-free
    one would assemble it densely, which is what the solver guard prevents).
    """
    dense, build = _dense_reference(kind, which)
    got = build(mode)
    assert got.fulqrum_mode == mode
    rng = np.random.default_rng(6)
    block = rng.standard_normal((dense.shape[0], 3))
    if np.iscomplexobj(dense):
        block = block + 1j * rng.standard_normal(block.shape)
    np.testing.assert_allclose(got.matvec(block[:, 0]), dense @ block[:, 0], atol=1e-10)
    np.testing.assert_allclose(got.matvec(block), dense @ block, atol=1e-10)
    np.testing.assert_allclose(got.hdiag(), np.diag(dense).real, atol=1e-10)
    if mode != "matrix_free":
        np.testing.assert_allclose(got.sparse().toarray(), dense, atol=1e-10, rtol=0)


@requires_fulqrum
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which", ["tpb", "gdb", "spinless"])
def test_fulqrum_problem_is_freed_without_the_garbage_collector(
    monkeypatch, which: str, mode: str
) -> None:
    """A dropped problem frees its operator at once, not at the next cyclic collection.

    Reference cycles (the problem's cached ``sparse``, and fulqrum's ``CSRLinearOperator``
    holding its own bound ``matvec``) kept each batch's CSR matrix alive until the
    collector ran; repeated solves on one large subspace then held one matrix per solve
    and ran out of memory (the benchmark in ``benchmark/``, molecule 375 at size 2000).
    """
    import gc
    import weakref

    import fulqrum as fq

    built: list[weakref.ref] = []  # fulqrum's own CSR operators, as they are built
    for method in ("to_csr_linearoperator", "to_csr_linearoperator_fast"):
        orig = getattr(fq.SubspaceHamiltonian, method)

        def spy(self, _orig=orig):
            out = _orig(self)
            built.append(weakref.ref(out))
            return out

        monkeypatch.setattr(fq.SubspaceHamiltonian, method, spy)
    _, build = _dense_reference("complex", which)
    gc.collect()
    gc.disable()
    try:
        problem = build(mode)
        if mode != "matrix_free":
            problem.sparse()
        ref = weakref.ref(problem)
        del problem
        assert ref() is None
        assert len(built) == (mode != "matrix_free")
        assert all(r() is None for r in built)
    finally:
        gc.enable()


@requires_fulqrum
def test_fulqrum_matrix_free_operator_survives_a_later_build() -> None:
    """A matrix-free operator still applies its own subspace after a later build.

    The module caches one ``SubspaceHamiltonian`` and the next build replaces its
    subspace; the earlier operator must re-apply its own before each product.
    """
    from sci_solvers.fulqrum_ci_problem import build_fulqrum_ci_problem

    hcore, eri = _real_integrals(NORB, seed=5)
    strs_a, strs_b = _full_strs(NORB, NELEC)
    first = build_fulqrum_ci_problem(
        strs_a[:6], strs_b[:7], hcore, eri, NORB, NELEC, fulqrum_mode="matrix_free"
    )
    ref = build_fulqrum_ci_problem(strs_a[:6], strs_b[:7], hcore, eri, NORB, NELEC)
    build_fulqrum_ci_problem(
        strs_a[2:], strs_b[1:], hcore, eri, NORB, NELEC, fulqrum_mode="matrix_free"
    )
    x = np.random.default_rng(5).standard_normal(first.dim)
    np.testing.assert_allclose(first.matvec(x), ref.sparse().toarray() @ x, atol=1e-10)


@requires_fulqrum
@pytest.mark.parametrize("mode", ["csr_fast", "matrix_free"])
@pytest.mark.parametrize("name", SOLVERS)
@pytest.mark.parametrize("which", ["tpb", "gdb", "spinless"])
def test_fulqrum_mode_backend_matches_torch(which: str, name: str, mode: str) -> None:
    """End to end in every basis: two roots per mode match the torch backend.

    (``"csr"`` is the default, so the tests above already cover it end to end.)
    """
    if which == "spinless":
        from test_spinless_complex_ci import _all_dets
        from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

        nmo, n = 8, 4
        hcore, eri = _genuinely_complex_spinorbital_integrals(nmo, seed=3)
        strs = np.array(_all_dets(nmo, n), dtype=np.int64)
        out = {}
        for backend, extra in (("torch", {}), ("fulqrum", {"fulqrum_mode": mode})):
            solver = create_solver(
                name, symmetrize_spin=False, nroots=2, hamiltonian_backend=backend, **extra
            )
            out[backend] = solver.sci_solver_callable_eigen_states([strs], hcore, eri, nmo, n)[
                0
            ].states
        _assert_same_states(out["fulqrum"], out["torch"])
        return
    hcore, eri = _integrals("complex")
    batch, extra = (_pairs(), {"mode": "gdb"}) if which == "gdb" else (_selected_strs(), {})
    ref = _solve(name, "torch", batch, hcore, eri, nroots=2, **extra)
    got = _solve(name, "fulqrum", batch, hcore, eri, nroots=2, fulqrum_mode=mode, **extra)
    _assert_same_states(got, ref)


@requires_fulqrum
def test_fulqrum_mode_symmetrize_spin_matches_torch() -> None:
    """``symmetrize_spin`` (the packed sector view) works on every mode."""
    hcore, eri = _real_integrals(NORB, seed=1)
    strs = np.sort(np.random.default_rng(1).permutation(_full_strs(NORB, NELEC)[0])[:8])
    energies = {}
    for backend, extra in [("torch", {})] + [("fulqrum", {"fulqrum_mode": m}) for m in MODES]:
        solver = create_solver(
            "arpack", symmetrize_spin=True, nroots=2, hamiltonian_backend=backend, **extra
        )
        states = solver.sci_solver_callable_eigen_states([(strs, strs)], hcore, eri, NORB, NELEC)[
            0
        ].states
        energies[extra.get("fulqrum_mode", backend)] = [s.energy for s in states]
    for mode in MODES:
        np.testing.assert_allclose(energies[mode], energies["torch"], atol=1e-7)


def test_fulqrum_mode_rejects_unknown_value() -> None:
    """``fulqrum_mode`` is validated at construction."""
    with pytest.raises(ValueError, match="fulqrum_mode must be one of"):
        create_solver("arpack", symmetrize_spin=False, fulqrum_mode="csr_like")


@requires_fulqrum
@pytest.mark.parametrize(
    "name, kwargs",
    [("arpack", {"use_operator": False}), ("slepc", {"storage": "aij"})],
)
def test_fulqrum_matrix_free_rejects_assembled_solver_settings(name: str, kwargs: dict) -> None:
    """A solver configured to assemble the matrix can't take a matrix-free operator.

    It would build ``matvec(eye(dim))`` densely; the CSR modes are fine there.
    """
    with pytest.raises(ValueError, match="fulqrum_mode='matrix_free' cannot be used"):
        create_solver(
            name,
            symmetrize_spin=False,
            hamiltonian_backend="fulqrum",
            fulqrum_mode="matrix_free",
            **kwargs,
        )
    create_solver(
        name, symmetrize_spin=False, hamiltonian_backend="fulqrum", fulqrum_mode="csr", **kwargs
    )


@requires_fulqrum
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which", ["tpb", "gdb", "spinless"])
def test_fulqrum_mode_reaches_every_builder(monkeypatch, which: str, mode: str) -> None:
    """The solver's ``fulqrum_mode`` reaches the builder in every basis.

    A dropped mode still solves correctly (only memory and speed differ), so the
    end-to-end tests can't see it: record the mode each built problem carries.
    """
    import sci_solvers.fulqrum_ci_problem as fcp

    seen: list[str] = []
    real = fcp._fulqrum_operator

    def spy(hsub, m, perm, signs=None):
        seen.append(m)
        return real(hsub, m, perm, signs)

    monkeypatch.setattr(fcp, "_fulqrum_operator", spy)
    solver = create_solver(
        "arpack",
        symmetrize_spin=False,
        hamiltonian_backend="fulqrum",
        fulqrum_mode=mode,
        **({"mode": "gdb"} if which == "gdb" else {}),
    )
    if which == "spinless":
        from test_spinless_complex_ci import _all_dets
        from test_spinless_genuinely_complex import _genuinely_complex_spinorbital_integrals

        hcore, eri = _genuinely_complex_spinorbital_integrals(8, seed=3)
        solver.sci_solver_callable_eigen_states(
            [np.array(_all_dets(8, 4), dtype=np.int64)], hcore, eri, 8, 4
        )
    else:
        hcore, eri = _integrals("real")
        batch = _pairs() if which == "gdb" else _selected_strs()
        solver.sci_solver_callable_eigen_states([batch], hcore, eri, NORB, NELEC)
    assert seen and set(seen) == {mode}


@requires_fulqrum
@pytest.mark.parametrize(
    "mode, method",
    [("csr", "to_csr_linearoperator"), ("csr_fast", "to_csr_linearoperator_fast")],
)
def test_fulqrum_csr_modes_use_their_fulqrum_build(monkeypatch, mode: str, method: str) -> None:
    """``csr`` uses fulqrum's direct CSR build and ``csr_fast`` its two-stage one.

    They build the same matrix, so only which fulqrum method ran tells them apart.
    """
    import fulqrum as fq

    from sci_solvers.fulqrum_ci_problem import build_fulqrum_ci_problem

    calls: list[str] = []
    for name in ("to_csr_linearoperator", "to_csr_linearoperator_fast"):
        real = getattr(fq.SubspaceHamiltonian, name)

        def spy(self, _real=real, _name=name):
            calls.append(_name)
            return _real(self)

        monkeypatch.setattr(fq.SubspaceHamiltonian, name, spy)
    hcore, eri = _integrals("real")
    build_fulqrum_ci_problem(*_selected_strs(), hcore, eri, NORB, NELEC, fulqrum_mode=mode)
    assert calls == [method]
