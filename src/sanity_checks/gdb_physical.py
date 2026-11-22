"""Physical Hamiltonians and checks for the GDB (``mode="gdb"``) sanity suite.

Each check solves one Hamiltonian over the full CI space twice, once as a tensor
product (``mode="tpb"``) and once as the same determinants given as an explicit pair
list (``mode="gdb"``), and requires the two ground states to agree (:data:`TOLERANCES`).
The Hamiltonians come from PySCF, so the suite reads no local data file:

* :func:`rotated_radical` -- a molecular open-shell radical whose real integrals are
  rotated by a random complex unitary. The integrals are genuinely complex, and the
  real full-CI solution rotated by the same unitary is an exact reference that does
  not use any sci-solvers RDM code; its deviations are reported, not checked.
* :func:`twisted_hydrogen_chain` -- a periodic hydrogen chain at a single k-point off
  the zone centre (twisted boundary conditions), whose restricted integrals are
  complex because of the Bloch phase.
* :func:`real_molecule` -- a closed-shell molecule with real integrals, for which GDB
  and TPB must agree as they always have.

Other data (FCIDUMPs, ``.h5`` integrals) comes in through the
YAML cases of :mod:`gdb_cases`.

Shared by ``gdb_sanity_check.py`` (the command-line suite) and the CI test
``tests/test_gdb_sanity_checks.py``.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any
from unittest import mock

import numpy as np
import scipy.sparse.linalg as spla
from pyscf import ao2mo, fci, gto, scf
from scipy.linalg import expm

from sci_solvers.energy_solvers.factory import create_solver


@dataclass
class PhysicalHamiltonian:
    """Spatial-orbital integrals of one physical system, with optional exact reference.

    Attributes:
        name: Short label for reports.
        hcore: One-electron integrals ``(norb, norb)``, real or complex Hermitian.
        eri: Chemist-order two-electron integrals ``(norb,) * 4``.
        norb: Number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        exact: Optional exact ground state ``{"energy", "rdm1", "rdm2"}`` obtained
            without any sci-solvers code, in the same orbital basis as the integrals.
    """

    name: str
    hcore: np.ndarray
    eri: np.ndarray
    norb: int
    nelec: tuple[int, int]
    exact: dict[str, np.ndarray | float] | None = field(default=None, repr=False)


def _random_unitary(norb: int, seed: int) -> np.ndarray:
    """A random complex unitary ``exp(i K)`` with ``K`` Hermitian and ``O(1)`` entries."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((norb, norb)) + 1j * rng.standard_normal((norb, norb))
    return expm(0.5j * (k + k.conj().T))


def rotated_radical(seed: int = 0) -> PhysicalHamiltonian:
    """NH2 (doublet, STO-3G, ROHF orbitals) in a randomly rotated complex orbital basis.

    Notes
    -----
    New orbitals ``phi'_p = sum_a U_ap phi_a`` give ``a'^+_p = sum_a U_ap a^+_a``. Writing
    the Hamiltonian in the new operators, the integrals transform as

    .. math::

        h'_{pq} = \\sum_{ab} U^*_{ap} U_{bq} h_{ab}, \\qquad
        (pq|rs)' = \\sum_{abcd} U^*_{ap} U_{bq} U^*_{cr} U_{ds} (ab|cd),

    and the spectrum is unchanged. The RDMs ``gamma_pq = <a^+_p a_q>`` and
    ``Gamma_pqrs = <a^+_p a^+_r a_s a_q>`` are expectation values of the operators
    themselves, so they transform with ``U`` and ``U^*`` exchanged:

    .. math::

        \\gamma'_{pq} = \\sum_{ab} U_{ap} U^*_{bq} \\gamma_{ab}, \\qquad
        \\Gamma'_{pqrs} = \\sum_{abcd} U_{ap} U^*_{bq} U_{cr} U^*_{ds} \\Gamma_{abcd}.

    The real full-CI energy and the RDMs rotated this way are the exact reference. An NH2
    radical has ``(n_alpha, n_beta) = (5, 4)``, which is the open-shell case where the
    two mixed-spin 2-RDM blocks differ.
    """
    mol = gto.M(atom="N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6", basis="sto-3g", spin=1, verbose=0)
    mf = scf.ROHF(mol).run()
    c = mf.mo_coeff
    norb = c.shape[1]
    h = c.T @ mf.get_hcore() @ c
    eri = ao2mo.restore(1, ao2mo.full(mol, c), norb)
    cis = fci.direct_spin1.FCI()
    cis.conv_tol = 1e-14
    e0, civec = cis.kernel(h, eri, norb, mol.nelec)
    d1, d2 = cis.make_rdm12(civec, norb, mol.nelec)

    u = _random_unitary(norb, seed)
    uc = u.conj()
    rot2 = "ap,bq,ab->pq"
    rot4 = "ap,bq,cr,ds,abcd->pqrs"
    return PhysicalHamiltonian(
        name="NH2 rotated",
        hcore=np.einsum(rot2, uc, u, h),
        eri=np.einsum(rot4, uc, u, uc, u, eri, optimize=True),
        norb=norb,
        nelec=mol.nelec,
        exact={
            "energy": float(e0),
            "rdm1": np.einsum(rot2, u, uc, d1),
            "rdm2": np.einsum(rot4, u, uc, u, uc, d2, optimize=True),
        },
    )


def twisted_hydrogen_chain(
    nelec: tuple[int, int] = (2, 1), twist: float = 0.2
) -> PhysicalHamiltonian:
    """A periodic H4 chain (STO-3G) at the k-point ``twist`` (fraction of the zone).

    Notes
    -----
    A restricted Hartree-Fock solution at a single k-point ``k`` gives Bloch orbitals
    ``psi_p(r + R) = e^{i k R} psi_p(r)``, so for ``k`` off the zone centre (and off its
    boundary) the integrals ``h_{pq}`` and ``(pq|rs)`` are genuinely complex. They keep
    ``(pq|rs) = (qp|sr)^*`` and ``(pq|rs) = (rs|pq)``, the two symmetries the CI
    operator needs. ``nelec`` can differ from the neutral ``(2, 2)``: the orbitals stay
    those of the neutral cell, which is enough to define an open-shell problem.
    """
    from pyscf.pbc import gto as pbc_gto
    from pyscf.pbc import scf as pbc_scf

    cell = pbc_gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
        a=np.diag([8.0, 8.0, 5.6]),
        unit="bohr",
        basis="sto-3g",
        verbose=0,
    )
    kpt = cell.get_abs_kpts([0.0, 0.0, twist])
    mf = pbc_scf.RHF(cell, kpt=kpt).density_fit().run()
    c = mf.mo_coeff
    norb = c.shape[1]
    h = c.conj().T @ mf.get_hcore() @ c
    eri = mf.with_df.ao2mo([c] * 4, [kpt] * 4, compact=False).reshape((norb,) * 4)
    return PhysicalHamiltonian(name=f"H4 chain k={twist}", hcore=h, eri=eri, norb=norb, nelec=nelec)


def real_molecule() -> PhysicalHamiltonian:
    """N2 (STO-3G, RHF orbitals, two lowest orbitals frozen), real closed shell."""
    from pyscf import mcscf

    mol = gto.M(atom="N 0 0 0; N 0 0 1.1", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    cas = mcscf.CASCI(mf, 8, 10)
    h, _ = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), 8)
    return PhysicalHamiltonian(name="N2 CAS(10,8)", hcore=h, eri=eri, norb=8, nelec=(5, 5))


# Bounds on the GDB-vs-TPB discrepancies, the same for every solver. The energy bound is
# relative to |E|; the RDM and overlap bounds are absolute. They sit 2-3x above the worst
# case measured across every solver and 20 ARPACK seeds, which came from ARPACK on a large
# real FCIDUMP: energy 7.7e-15 relative, rdm1 6e-10, rdm2 1.25e-9 and 1 - overlap 2.2e-15.
# Every other case and solver is <= 1e-11.
#
# Possible future change (not needed so far): the RDM spread tracks the solver, not the
# RDM size (max|rdm1| ~ 2 and max|rdm2| ~ 4 everywhere). Two alternatives:
#   * relative to |E|, e.g. ``rdm < 1e-11 * |E|``, if the spread turns out to grow with |E|;
#   * per-solver bounds, e.g. 3e-9 for ARPACK and 1e-10 for the deterministic lobpcg and
#     davidson, if a tighter check on those becomes worth the extra table.
TOLERANCES = {"energy": 1e-12, "rdm1": 3e-9, "rdm2": 3e-9, "overlap": 1e-12}


def failures(diffs: dict[str, float]) -> list[str]:
    """Names of the GDB-vs-TPB quantities in ``diffs`` outside :data:`TOLERANCES`.

    ``energy`` is checked relative to ``|E|``, ``overlap`` as ``|1 - overlap|``. The
    ``mode_*`` quantities, when present (a fulqrum mode against ``csr``), are checked
    against the same bounds.
    ``overlap`` is the only check that sees a conjugated ground state, since conjugation
    leaves the energy and RDMs of a full product basis unchanged; ``rdm2`` is the only
    one that sees a wrong mixed-spin packing, which leaves the vector unchanged.
    """
    dev = {
        "energy": diffs["energy"] / max(1.0, diffs["energy_scale"]),
        "rdm1": diffs["rdm1"],
        "rdm2": diffs["rdm2"],
        "overlap": abs(1.0 - diffs["overlap"]),
    }
    if "mode_energy" in diffs:
        dev |= {
            "mode_energy": diffs["mode_energy"] / max(1.0, diffs["energy_scale"]),
            "mode_rdm1": diffs["mode_rdm1"],
            "mode_rdm2": diffs["mode_rdm2"],
            "mode_overlap": abs(1.0 - diffs["mode_overlap"]),
        }
    return [k for k in dev if not dev[k] < TOLERANCES[k.removeprefix("mode_")]]


@contextmanager
def seeded_arpack(seed: int | None) -> Iterator[None]:
    """Make every ARPACK (``scipy.sparse.linalg.eigsh``) call start from a seeded vector.

    ``eigsh`` starts from a random vector unless given ``v0``, and ``ArpackSolver`` passes
    none, so its results vary from run to run at the level of its tolerance. Inside this
    context each call that has no ``v0`` gets one drawn from
    ``np.random.default_rng(seed)``, so a fixed ``seed`` reproduces a run and different
    seeds sample the run-to-run spread. ``seed=None`` leaves ``eigsh`` untouched. Other
    solvers are unaffected.
    """
    if seed is None:
        yield
        return
    eigsh = spla.eigsh

    def seeded(a: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("v0") is None:
            v0 = np.random.default_rng(seed).standard_normal(a.shape[0])
            kwargs["v0"] = v0.astype(a.dtype)
        return eigsh(a, *args, **kwargs)

    with mock.patch.object(spla, "eigsh", seeded):
        yield


def _strings(norb: int, n: int) -> np.ndarray:
    """All ``n``-electron occupation bitstrings over ``norb`` orbitals, ascending."""
    return np.array(
        sorted(sum(1 << i for i in c) for c in combinations(range(norb), n)), dtype=np.int64
    )


def _inter_to_block_sign(a: int, b: int, norb: int) -> int:
    """Sign between a pair amplitude (interleaved-Wick order) and the spin-block one.

    ``(-1)^{sum_{j in alpha} popcount(beta & ((1 << j) - 1))}``: the pair engine orders
    spin orbitals interleaved, the product basis puts every alpha before every beta.
    """
    n = sum(bin(b & ((1 << j) - 1)).count("1") for j in range(norb) if a >> j & 1)
    return -1 if n % 2 else 1


def compare_gdb_tpb(
    ham: PhysicalHamiltonian,
    solver: str = "arpack",
    strings: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int | None = 0,
    reference: tuple[Any, Any] | None = None,
) -> tuple[dict[str, float], tuple[Any, Any]]:
    """Solve ``ham`` on one determinant space in both bases and return the discrepancies.

    The space is the product ``strings[0] x strings[1]`` (default: the full CI space),
    given to TPB as the two string lists and to GDB as the flattened pair list. Both
    solves run under :func:`seeded_arpack` with ``seed``. ``solver`` is a factory name,
    optionally ``<name>@<backend>`` to build the Hamiltonian with that
    ``hamiltonian_backend`` (e.g. ``arpack@fulqrum``, which runs on the CPU), and
    ``<name>@fulqrum:<mode>`` to choose its ``fulqrum_mode`` (``csr``, the default,
    ``csr_fast`` or ``matrix_free``).

    ``reference`` is the ``(tpb, gdb)`` ground-state pair of an earlier call on the same
    space, typically the same solver in ``fulqrum_mode="csr"``. When given, each basis is
    also compared with it, and the ``mode_*`` entries report the worse of the two.

    Returns:
        A dict of discrepancies, and the ``(tpb, gdb)`` ground states (``SCIState`` objects)
        to pass as a later call's ``reference``. The dict holds:
        ``dim``: the number of determinants.
        ``energy``, ``rdm1``, ``rdm2``: max-abs differences GDB vs TPB.
        ``energy_scale``: ``|E_tpb|``.
        ``overlap``: ``|<tpb|gdb>|`` after the interleaved-to-block sign; 1 when the
        two ground states are the same vector up to a global phase.
        ``overlap_conj``: ``|<tpb*|gdb>|``; 1 instead of ``overlap`` if GDB returned
        the complex-conjugated amplitudes.
        ``exact_energy``, ``exact_rdm1``, ``exact_rdm2``: max-abs differences GDB vs
        the exact reference, when ``ham.exact`` is set.
        ``mode_energy``, ``mode_rdm1``, ``mode_rdm2``, ``mode_overlap``: the same four
        against ``reference``, when it is given; ``mode_overlap`` is the smaller of the
        TPB and GDB ground-state overlaps.
    """
    norb, nelec = ham.norb, ham.nelec
    if strings is None:
        strings = (_strings(norb, nelec[0]), _strings(norb, nelec[1]))
    sa, sb = strings
    pa, pb = (x.ravel() for x in np.meshgrid(sa, sb, indexing="ij"))
    name, _, backend = solver.partition("@")
    backend, _, fulqrum_mode = backend.partition(":")
    budget = "max_cycle" if name == "davidson" else "maxiter"
    kw: dict[str, Any] = {"tol": 1e-12, budget: 2000}
    if backend:
        kw["hamiltonian_backend"] = backend
    if fulqrum_mode:
        kw["fulqrum_mode"] = fulqrum_mode

    def ground(mode: str, basis: tuple[np.ndarray, np.ndarray]):
        s = create_solver(name, symmetrize_spin=False, mode=mode, compute_rdm2=True, **kw)
        return s.sci_solver_callable_eigen_states([basis], ham.hcore, ham.eri, norb, nelec)[
            0
        ].states[0]

    with seeded_arpack(seed):
        tpb = ground("tpb", (sa, sb))
        gdb = ground("gdb", (pa, pb))

    st = gdb.sci_state
    ia = {int(s): i for i, s in enumerate(tpb.sci_state.ci_strs_a)}
    ib = {int(s): i for i, s in enumerate(tpb.sci_state.ci_strs_b)}
    c_tpb = np.array([
        _inter_to_block_sign(int(a), int(b), norb)
        * tpb.sci_state.amplitudes[ia[int(a)], ib[int(b)]]
        for a, b in zip(st.ci_strs_a, st.ci_strs_b)
    ])
    out = {
        "dim": len(pa),
        "energy": abs(gdb.energy - tpb.energy),
        "rdm1": float(np.abs(gdb.rdm1 - tpb.rdm1).max()),
        "rdm2": float(np.abs(gdb.rdm2 - tpb.rdm2).max()),
        "energy_scale": abs(tpb.energy),
        "overlap": float(abs(np.vdot(c_tpb, st.amplitudes))),
        "overlap_conj": float(abs(np.vdot(c_tpb.conj(), st.amplitudes))),
    }
    if ham.exact is not None:
        out["exact_energy"] = abs(gdb.energy - float(ham.exact["energy"]))
        out["exact_rdm1"] = float(np.abs(gdb.rdm1 - ham.exact["rdm1"]).max())
        out["exact_rdm2"] = float(np.abs(gdb.rdm2 - ham.exact["rdm2"]).max())
    if reference is not None:
        pairs = list(zip(reference, (tpb, gdb), strict=True))
        out["mode_energy"] = max(abs(x.energy - r.energy) for r, x in pairs)
        out["mode_rdm1"] = max(float(np.abs(x.rdm1 - r.rdm1).max()) for r, x in pairs)
        out["mode_rdm2"] = max(float(np.abs(x.rdm2 - r.rdm2).max()) for r, x in pairs)
        out["mode_overlap"] = min(
            float(abs(np.vdot(r.sci_state.amplitudes, x.sci_state.amplitudes))) for r, x in pairs
        )
    return out, (tpb, gdb)
