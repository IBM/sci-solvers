#!/usr/bin/env python
"""
PySCF's lib/linalg_helper.py implementation of the davidson method except that
for the energy shift in the preconditioner we propose 3 variants:

  ground shift : what PySCF does — always shift by E_0, the last estimate of
                 the ground state energy.
  active shift : what DICE does — shift by the energy of the last converged root.
  per root shift: each residual uses its own energy estimate.

Also contains simplified FCI and selected-CI fixed-space kernels:

  fci_kernel                : full CI, spin1 or spin0 path.
  sci_kernel_fixed_space    : selected CI on a fixed determinant space,
                              spin1 or spin0 path.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import sys
from typing import Any, List, Tuple, cast

import numpy as np
import scipy.linalg
from pyscf import __config__, ao2mo, lib
from pyscf.fci import cistring, direct_spin0, direct_spin1
from pyscf.fci import selected_ci as sci
from pyscf.fci import selected_ci_spin0 as scis0
from pyscf.fci.selected_ci import _as_SCIvector

from sci_solvers.eigen_states import EigenStates
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    davidson_block_size_to_max_space,
    reject_init_vectors,
    reject_spinless_nelec,
)
from sci_solvers.energy_solvers.spin_filter_mixin import SpinFilterMixin
from sci_solvers.energy_solvers.truncated_davidson_solver.solver import make_sci_result
from sci_solvers.hamiltonian import HamiltonianData


class LinearDependenceError(RuntimeError):
    """Stand-in for pyscf.lib.exceptions.LinearDependencyError."""

    pass


class _Logger:
    """Stand-in for pyscf.lib.logger.Logger -- only .debug/.debug1/.warn
    and .verbose, the only members davidson1 touches."""

    WARN = 30
    DEBUG = 40
    DEBUG1 = 50

    def __init__(self, stdout=sys.stdout, verbose=WARN):
        self.stdout = stdout
        self.verbose = verbose

    def debug(self, msg, *args):
        if self.verbose >= self.DEBUG:
            print((msg % args) if args else msg, file=self.stdout)

    def debug1(self, msg, *args):
        if self.verbose >= self.DEBUG1:
            print((msg % args) if args else msg, file=self.stdout)

    def warn(self, msg, *args):
        print("WARN: " + ((msg % args) if args else msg), file=self.stdout)


class _LoggerModule:
    WARN = _Logger.WARN
    Logger = _Logger


logger = _LoggerModule()


class _DictBackedStore(dict):
    """Fallback for pyscf.lib.misc.H5TmpFile when h5py isn't available --
    same __setitem__/__getitem__/flush interface used by _Xlist below,
    backed by an in-memory dict instead of an HDF5 scratch file. Only the
    storage medium differs; _Xlist's algorithmic behavior is unchanged."""

    def flush(self):
        pass


class _Misc:
    class StreamObject:
        stdout = sys.stdout

    @staticmethod
    def H5TmpFile():
        try:
            import tempfile

            import h5py

            tmp = tempfile.NamedTemporaryFile()
            return h5py.File(tmp.name, "w")
        except ImportError:
            return _DictBackedStore()


misc = _Misc()

DAVIDSON_LINDEP = float(
    os.environ.get("DAVIDSON_LINDEP", getattr(__config__, "DAVIDSON_LINDEP", 1e-14))
)
MAX_MEMORY = int(os.environ.get("MAX_MEMORY", getattr(__config__, "MAX_MEMORY", 4000)))
_env_follow = os.environ.get("FOLLOW_STATE", str(getattr(__config__, "FOLLOW_STATE", False)))
FOLLOW_STATE = _env_follow.lower() in ("true", "1", "yes")


def make_diag_precond(diag, level_shift=1e-3):
    """Generate the preconditioner function with the diagonal function."""

    # For diagonal matrix A, precond (Ax-x*e)/(diag(A)-e) is not able to
    # generate linearly independent basis. Use level_shift to
    # break the correlation between Ax-x*e and diag(A)-e.
    def precond(dx, e, *_args):
        diagd = diag - (e - level_shift)
        diagd[abs(diagd) < 1e-8] = 1e-8
        return dx / diagd

    return precond


def _fill_heff_hermitian(heff, _xs, ax, xt, axt, dot):
    nrow = len(axt)
    row1 = len(ax)
    row0 = row1 - nrow
    for ip, i in enumerate(range(row0, row1)):
        for jp, j in enumerate(range(row0, i)):
            heff[i, j] = dot(xt[ip].conj(), axt[jp])
            heff[j, i] = heff[i, j].conj()
        heff[i, i] = dot(xt[ip].conj(), axt[ip]).real

    for i in range(row0):
        axi = np.asarray(ax[i])
        for jp, j in enumerate(range(row0, row1)):
            heff[j, i] = dot(xt[jp].conj(), axi)
            heff[i, j] = heff[j, i].conj()
        axi = None
    return heff


def davidson(
    aop,
    x0,
    precond,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
    lindep=DAVIDSON_LINDEP,
    max_memory=MAX_MEMORY,
    dot=np.dot,
    callback=None,
    nroots=1,
    lessio=False,
    pick=None,
    verbose=logger.WARN,
    follow_state=FOLLOW_STATE,
    shift_mode="ground",
):
    r"""Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function(x) => array_like_x
            aop(x) to mimic the matrix vector multiplication.
            The argument is a 1D array.  The returned value is a 1D array.
        x0 : 1D array or a list of 1D array
            Initial guess.
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx
            Preconditioner to generate new trial vector.

    Kwargs:
        (unchanged from upstream -- see davidson1 below)
        shift_mode : {'ground', 'active', 'per_root'}
            Which Ritz value is passed to `precond` as `e` for a given
            cycle's unconverged residuals. Default 'ground' reproduces
            upstream PySCF's actual behavior exactly (always e[0]). Not
            part of upstream PySCF -- see module docstring.

    Returns:
        e : float or list of floats
        c : 1D array or list of 1D arrays
    """
    e, x = davidson1(
        lambda xs: [aop(x) for x in xs],
        x0,
        precond,
        tol,
        max_cycle,
        max_space,
        lindep,
        max_memory,
        dot,
        callback,
        nroots,
        lessio,
        pick,
        verbose,
        follow_state,
        shift_mode=shift_mode,
    )[1:]
    if nroots == 1:
        return e[0], x[0]
    else:
        return e, x


def davidson1(
    aop,
    x0,
    precond,
    tol=1e-12,
    max_cycle=50,
    max_space=12,
    lindep=DAVIDSON_LINDEP,
    max_memory=MAX_MEMORY,
    dot=np.dot,
    callback=None,
    nroots=1,
    lessio=False,
    pick=None,
    verbose=logger.WARN,
    follow_state=FOLLOW_STATE,
    tol_residual=None,
    fill_heff=_fill_heff_hermitian,
    shift_mode="ground",
):
    r"""Davidson diagonalization method to solve  a c = e c.  Ref
    [1] E.R. Davidson, J. Comput. Phys. 17 (1), 87-94 (1975).
    [2] http://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter11.pdf

    Note: This function has an overhead of memory usage ~4*x0.size*nroots

    Args:
        aop : function([x]) => [array_like_x]
        x0 : 1D array or a list of 1D arrays or a function to generate x0 array(s)
        precond : diagonal elements of the matrix or  function(dx, e, x0) => array_like_dx

    Kwargs:
        tol, max_cycle, max_space, lindep, max_memory, dot, callback,
        nroots, lessio, pick, verbose, follow_state, tol_residual, fill_heff
            -- all unchanged from upstream pyscf.lib.linalg_helper.davidson1.
        shift_mode : {'ground', 'active', 'per_root'}
            Which single Ritz value e is handed to `precond` for every
            unconverged residual this cycle (or, for 'per_root', which
            per-residual value). Not part of upstream PySCF.

            'ground' (default): always e[0] -- this exactly reproduces
                upstream's unconditional `precond(xt[k], e[0], x0[k])`.
            'active': e[k0], where k0 is the index of the lowest root
                that has not yet converged this cycle -- batched analogue
                of DICE's Davidson.cpp (ei = eigenvalues()[convergedRoot]).
            'per_root': each residual k uses its own e[k].

    Returns:
        conv : bool
        e : list of floats
        c : list of 1D arrays
    """
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    if tol_residual is None:
        toloose = np.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1("tol %g  toloose %g", tol, toloose)

    if not callable(precond):
        precond = make_diag_precond(precond)

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, np.ndarray) and x0.ndim == 1:
        x0 = [x0]
    # max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots - 1) * 4
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    _incore = max_memory * 1e6 / x0[0].nbytes > max_space * 2 + nroots * 3
    lessio = lessio and not _incore
    log.debug1(
        "max_cycle %d  max_space %d  max_memory %d  incore %s",
        max_cycle,
        max_space,
        max_memory,
        _incore,
    )
    dtype = None
    heff = None
    fresh_start = True
    e = None
    v = None
    conv = np.zeros(nroots, dtype=bool)

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
            else:
                xs = _Xlist()
                ax = _Xlist()
            space = 0
            # Orthogonalize xt space because the basis of subspace xs must be orthogonal
            # but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                log.warn("QR decomposition removed %d vectors.", x0len - len(xt))
                if callable(pick):
                    log.warn(
                        "Check to see if `pick` function %s is providing linear dependent vectors",
                        pick.__name__,
                    )
                if len(xt) == 0:
                    if icyc == 0:
                        msg = "Initial guess is empty or zero"
                    else:
                        msg = (
                            "No more linearly independent basis were found. "
                            "Unless loosen the lindep tolerance (current value "
                            f"{lindep}), the diagonalization solver is not able "
                            "to find eigenvectors."
                        )
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        axt = aop(xt)
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
        rnow = len(xt)
        head, space = space, space + rnow

        if dtype is None:
            try:
                dtype = np.result_type(axt[0], xt[0])
            except IndexError:
                raise LinearDependenceError(
                    "No linearly independent basis found by the diagonalization solver."
                )
        if heff is None:  # Lazy initialize heff to determine the dtype
            heff = np.empty((max_space + nroots, max_space + nroots), dtype=dtype)
        else:
            heff = np.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        xt = axt = None
        w, v = scipy.linalg.eigh(heff[:space, :space])
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
            if len(w) == 0:
                raise RuntimeError(f"Not enough eigenvalues found by {pick}")

        e = w[:nroots]
        v = v[:, :nroots]
        conv = np.zeros(e.size, dtype=bool)
        if not fresh_start:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug(
                "Number of roots different from the previous step (%d,%d)", e.size, elast.size
            )
            de = e
        else:
            de = e - elast

        x0 = None
        x0 = _gen_x0(v, xs)
        if lessio:
            ax0 = aop(x0)
        else:
            ax0 = _gen_x0(v, ax)

        dx_norm = np.zeros(e.size)
        xt = [None] * nroots
        for k, ek in enumerate(e):
            xt[k] = ax0[k] - ek * x0[k]
            dx_norm[k] = np.sqrt(dot(xt[k].conj(), xt[k]).real)
            conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose
            if conv[k] and not conv_last[k]:
                log.debug(
                    "root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g", k, dx_norm[k], ek, de[k]
                )
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = np.argmax(abs(de))
        if all(conv):
            log.debug(
                "converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g",
                icyc,
                space,
                max_dx_norm,
                e,
                de[ide],
            )
            break
        elif (
            follow_state
            and max_dx_norm > 1
            and max_dx_norm / max_dx_last > 3
            and space > nroots + 2
        ):
            log.debug(
                "davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g",
                icyc,
                space,
                max_dx_norm,
                e,
                de[ide],
            )
            log.debug("Large |r| detected, restore to previous x0")
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        # --- shift_mode: which Ritz value e is passed to precond() below ---
        # Not part of upstream PySCF, which always uses e[0] for every k
        # (see docstring). 'ground' reproduces that exactly.
        if shift_mode == "per_root":
            e_shift_by_k = e
        else:
            if shift_mode == "active":
                unconverged = np.where(~conv)[0]
                k0 = int(unconverged[0]) if unconverged.size else 0
                e_shift = e[k0]
            elif shift_mode == "ground":
                e_shift = e[0]
            else:
                raise ValueError(f"Unknown shift_mode: {shift_mode!r}")
            e_shift_by_k = np.full(nroots, e_shift)
        # ---------------------------------------------------------------------

        # remove subspace linear dependency
        for k, ek in enumerate(e):
            if (not conv[k]) and dx_norm[k] ** 2 > lindep:
                xt[k] = precond(xt[k], e_shift_by_k[k], x0[k])
                xt[k] *= dot(xt[k].conj(), xt[k]).real ** -0.5
            elif not conv[k]:
                # Remove linearly dependent vector
                xt[k] = None
                log.debug1("Drop eigenvector %d, norm=%4.3g", k, dx_norm[k])
            else:
                xt[k] = None

        xt, norm_min = _normalize_xt_(xt, xs, lindep, dot)
        log.debug(
            "davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g",
            icyc,
            space,
            max_dx_norm,
            e,
            de[ide],
            norm_min,
        )
        if len(xt) == 0:
            log.debug("Linear dependency in trial subspace. |r| for each state %s", dx_norm)
            conv = dx_norm < toloose
            break

        max_dx_last = max_dx_norm
        fresh_start = space + nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = list(x0)  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        log.warn(f"Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})")

    return np.asarray(conv), e, x0


def _qr(xs, dot, lindep=1e-14):
    """QR decomposition for a list of vectors (for linearly independent vectors only).
    xs = (r.T).dot(qs)
    """
    nvec = len(xs)
    xs = qs = np.array(xs, copy=True)
    rmat = np.eye(nvec, order="F", dtype=xs.dtype)

    nv = 0
    for i in range(nvec):
        xi = xs[i]
        for j in range(nv):
            prod = dot(qs[j].conj(), xi)
            xi -= qs[j] * prod
            rmat[:, nv] -= rmat[:, j] * prod
        innerprod = dot(xi.conj(), xi).real
        norm = np.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi / norm
            rmat[: nv + 1, nv] /= norm
            nv += 1
    return qs[:nv], np.linalg.inv(rmat[:nv, :nv])


def _outprod_to_subspace(v, xs):
    ndim = v.ndim
    if ndim == 1:
        v = v[:, None]
    space, nroots = v.shape
    x0 = np.einsum("c,x->cx", v[space - 1], np.asarray(xs[space - 1]))
    for i in reversed(range(space - 1)):
        xsi = np.asarray(xs[i])
        for k in range(nroots):
            x0[k] += v[i, k] * xsi
    if ndim == 1:
        x0 = x0[0]
    return x0


_gen_x0 = _outprod_to_subspace


def _sort_elast(elast, conv_last, vlast, v, log):
    """
    Eigenstates may be flipped during the Davidson iterations.  Reorder the
    eigenvalues of last iteration to make them comparable to the eigenvalues
    of the current iterations.
    """
    head, nroots = vlast.shape
    ovlp = abs(np.dot(v[:head].conj().T, vlast))
    mapping = np.argmax(ovlp, axis=1)
    found = np.any(ovlp > 0.5, axis=1)

    if log.verbose >= logger.Logger.DEBUG:
        ordering_diff = mapping != np.arange(len(mapping))
        if any(ordering_diff & found):
            log.debug("Old state -> New state")
            for i in np.where(ordering_diff)[0]:
                log.debug("  %3d     ->   %3d ", mapping[i], i)

    conv = conv_last[mapping]
    e = elast[mapping]
    conv[~found] = False
    e[~found] = 0.0
    return e, conv


def _normalize_xt_(xt, xs, threshold, dot):
    """Projects out existing basis vectors xs. Also checks whether the precond
    function is ill-conditioned"""
    xt = [x for x in xt if x is not None]
    for xsi in xs:
        xsi = np.asarray(xsi)
        for xi in xt:
            xi -= xsi * dot(xsi.conj(), xi)

    norm_min = 1
    out = []
    for xi in xt:
        norm = dot(xi.conj(), xi).real ** 0.5
        if norm**2 > threshold:
            xi *= 1 / norm
            out.append(xi)
            norm_min = min(norm_min, norm)
    return out, norm_min


class _Xlist(list):
    def __init__(self):
        self.scr_h5 = misc.H5TmpFile()
        self.index = []

    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[str(key)]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def append(self, x):
        length = len(self.index)
        key = length + 1
        index_set = set(self.index)
        if key in index_set:
            key = set(range(length)).difference(index_set).pop()
        self.index.append(key)

        self.scr_h5[str(key)] = x
        self.scr_h5.flush()

    def extend(self, x):
        for xi in x:
            self.append(xi)

    def __setitem__(self, n, x):
        key = self.index[n]
        self.scr_h5[str(key)][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del self.scr_h5[str(key)]


def _check_singlet(c, na):
    """Symmetrize a spin0 CI vector and verify it's (numerically) a singlet."""
    c = c.reshape(na, na)
    c = lib.transpose_sum(c, inplace=True) * 0.5
    norm = np.linalg.norm(c)
    if abs(norm - 1) > 1e-6:
        raise ValueError("State not singlet %g" % (norm - 1))
    return (c / norm).ravel()


def fci_kernel(
    h1e,
    eri,
    norb,
    nelec,
    spin=None,
    ci0=None,
    link_index=None,
    tol=1e-10,
    lindep=1e-14,
    max_cycle=50,
    max_space=12,
    nroots=1,
    ecore=0,
    **kwargs,
):
    """FCI kernel"""
    if isinstance(nelec, (int, np.integer)):
        nelecb = (nelec - (spin or 0)) // 2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    use_spin0 = spin == 0

    if use_spin0:
        na = cistring.num_strings(norb, neleca)
        link_index = cistring.gen_linkstr_index_trilidx(range(norb), neleca)

        h2e = direct_spin0.absorb_h1e(h1e, eri, norb, (neleca, nelecb), fac=0.5)

        def hop(c):
            hc = direct_spin0.contract_2e(
                h2e, c.reshape(na, na), norb, (neleca, nelecb), link_index
            )
            return hc.ravel()

        hdiag = direct_spin0.make_hdiag(h1e, eri, norb, (neleca, nelecb)).ravel()
        precond = make_diag_precond(hdiag)

        if ci0 is None:
            ci0 = direct_spin0.get_init_guess(norb, (neleca, nelecb), nroots, hdiag)
            ci0 = [x.ravel() for x in ci0]
        elif isinstance(ci0, np.ndarray):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

        conv, e, c = davidson1(
            lambda xs: [hop(x) for x in xs],
            ci0,
            precond,
            tol=tol,
            lindep=lindep,
            max_cycle=max_cycle,
            max_space=max_space,
            nroots=nroots,
            **kwargs,
        )

        # project each root back onto the symmetric (singlet) subspace
        if nroots == 1:
            e, c = e[0], c[0]
            c = _check_singlet(c, na)
            return e + ecore, c.reshape(na, na).view(direct_spin1.FCIvector)
        else:
            c = [_check_singlet(ci, na) for ci in c]
            return [ei + ecore for ei in e], [
                ci.reshape(na, na).view(direct_spin1.FCIvector) for ci in c
            ]

    else:
        # --- fall back to your existing direct_spin1 wrapper ---
        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
        h2e = direct_spin1.absorb_h1e(h1e, eri, norb, (neleca, nelecb), fac=0.5)

        def hop(c):
            hc = direct_spin1.contract_2e(h2e, c, norb, (neleca, nelecb), link_index)
            return hc.ravel()

        hdiag = direct_spin1.make_hdiag(h1e, eri, norb, (neleca, nelecb)).ravel()
        precond = make_diag_precond(hdiag)

        if ci0 is None:
            ci0 = []
            for addr in np.argsort(hdiag)[:nroots]:
                x = np.zeros(na * nb)
                x[addr] = 1
                ci0.append(x)
        elif isinstance(ci0, np.ndarray):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]

        conv, e, c = davidson1(
            lambda xs: [hop(x) for x in xs],
            ci0,
            precond,
            tol=tol,
            lindep=lindep,
            max_cycle=max_cycle,
            max_space=max_space,
            nroots=nroots,
            **kwargs,
        )

        if nroots == 1:
            e, c = e[0], c[0]
            return e + ecore, c.reshape(na, nb).view(direct_spin1.FCIvector)
        else:
            return [ei + ecore for ei in e], [
                ci.reshape(na, nb).view(direct_spin1.FCIvector) for ci in c
            ]


def _check_singlet_sci(c, na, ci_strs):
    """Symmetrize a spin0-selected-CI vector and verify it's (numerically)
    a singlet."""
    c = c.reshape(na, na)
    c = lib.transpose_sum(c, inplace=True) * 0.5
    norm = np.linalg.norm(c)
    if abs(norm - 1) > 1e-6:
        raise ValueError("State not singlet %g" % (norm - 1))
    return _as_SCIvector(c / norm, ci_strs)


def sci_kernel_fixed_space(
    h1e,
    eri,
    norb,
    nelec,
    ci_strs,
    spin=None,
    tol=1e-10,
    lindep=1e-14,
    max_cycle=50,
    max_space=12,
    nroots=1,
    ecore=0,
    **kwargs,
):
    """Selected-CI diagonalization on a FIXED determinant space.

    ``ci_strs`` arrays must be strictly ascending (sorted, unique): PySCF's
    ``selected_ci`` assumes it and is silently wrong otherwise.
    """
    nelec = direct_spin1._unpack_nelec(nelec, spin)
    na, nb = len(ci_strs[0]), len(ci_strs[1])

    want_spin0 = spin == 0
    use_spin0 = want_spin0 and na == nb and np.array_equal(ci_strs[0], ci_strs[1])

    if want_spin0 and not use_spin0:
        # caller asked for spin0 semantics but strings don't support it
        raise ValueError("spin0 kernel requires neleca==nelecb and ci_strs[0] == ci_strs[1]")

    # -------------------------------------------------------------------
    if use_spin0:
        # NOTE: scis0.contract_2e expects the pre-absorbed 2e Hamiltonian
        # (same convention as sci.contract_2e below), NOT raw eri. Its
        # internal h_ps correction only restores the aaaa/bbaa symmetry
        # of whatever tensor it's given -- it does not fold h1e in itself.
        h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
        h2e = ao2mo.restore(1, h2e, norb)

        link_index = sci._all_linkstr_index(ci_strs, norb, nelec)
        hdiag = scis0.make_hdiag(h1e, eri, ci_strs, norb, nelec)  # raw h1e/eri, symmetrized
        precond = make_diag_precond(hdiag)

        def hop(c):
            hc = scis0.contract_2e(h2e, _as_SCIvector(c, ci_strs), norb, nelec, link_index)
            return np.asarray(hc).ravel()

        ci0 = direct_spin1._get_init_guess(na, na, nroots, hdiag, nelec)
        x0 = []
        for x in ci0:
            x = x.reshape(na, na)
            x = lib.transpose_sum(x, inplace=True) * 0.5
            norm = np.linalg.norm(x)
            if norm > 1e-12:
                x = x / norm
            x0.append(x.ravel())

    else:
        h2e = direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
        h2e = ao2mo.restore(1, h2e, norb)

        link_index = sci._all_linkstr_index(ci_strs, norb, nelec)
        hdiag = sci.make_hdiag(h1e, eri, ci_strs, norb, nelec)
        precond = make_diag_precond(hdiag)

        def hop(c):
            hc = sci.contract_2e(h2e, _as_SCIvector(c, ci_strs), norb, nelec, link_index)
            return hc.ravel()

        ci0 = direct_spin1._get_init_guess(na, nb, nroots, hdiag, nelec)
        x0 = [c.ravel() for c in ci0]

    # --- diagonalize (shared) -----------------------------------------
    conv, e, c = davidson1(
        lambda xs: [hop(x) for x in xs],
        x0,
        precond,
        tol=tol,
        lindep=lindep,
        max_cycle=max_cycle,
        max_space=max_space,
        nroots=nroots,
        **kwargs,
    )

    if use_spin0:
        if nroots == 1:
            return e[0] + ecore, _check_singlet_sci(c[0], na, ci_strs)
        else:
            return [ei + ecore for ei in e], [_check_singlet_sci(ci, na, ci_strs) for ci in c]
    else:
        if nroots == 1:
            return e[0] + ecore, _as_SCIvector(c[0].reshape(na, nb), ci_strs)
        else:
            return [ei + ecore for ei in e], [
                _as_SCIvector(ci.reshape(na, nb), ci_strs) for ci in c
            ]


class DavidsonSolver(EnergyCalculatorBase):
    """
    sci_solver wrapping the custom shift-mode Davidson selected-CI kernel
    (davidson_solver.sci_kernel_fixed_space).

    Same interface as PySCFSolver, but diagonalizes each batch with the
    in-house Davidson implementation instead of PySCF's own eigensolver.
    Exposes ``shift_mode`` to select which Ritz value is used in the
    preconditioner's energy shift ("ground", "active", or "per_root").

    Multi-root: the fixed-space kernel accepts ``nroots`` and returns the lowest
    ``nroots`` eigenpairs over the (shared) input determinant space, so
    ``self.nroots`` is threaded straight through — restoring the multi-root
    support the eigensolver adapters previously exposed.
    """

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        verbose: bool = False,
        shift_mode: str = "ground",
        nroots: int = 1,
    ):
        """Initialize the DavidsonSolver."""
        super().__init__(
            hamiltonian=hamiltonian,
            symmetrize_spin=symmetrize_spin,
            verbose=verbose,
            nroots=nroots,
        )
        self.shift_mode = shift_mode

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        *,
        spin_sq: float | None = None,
        max_cycle: int = 200,
        davidson_block_size: int | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Diagonalize each batch via the custom shift-mode Davidson kernel.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals, shape ``(norb,)*4``.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``. A scalar ``nelec`` (the spinless
                flat spin-orbital path) is rejected with
                ``NotImplementedError`` — this solver does not implement it.
            spin_sq: Target value for total spin squared (S²). Only ``None``
                or ``0`` are meaningful here — a nonzero target would require
                a spin-penalty term this kernel doesn't implement; ``spin_sq
                == 0`` instead selects the spin0 (singlet) code path when
                ``neleca == nelecb`` and the batch's alpha/beta strings match.
            max_cycle: Flat Davidson iteration budget — total matvec cycles
                across all restarts. This kernel is PySCF's ``davidson1`` (with a
                modified preconditioner shift), so its loop is flat and a restart
                is an internal flag rather than an outer loop. Kept distinct from
                the nested solvers' ``max_davidson_iters``, for which
                ``max_cycle ~= max_davidson_iters * davidson_block_size``.
            davidson_block_size: Davidson subspace size — vectors accumulated
                before a restart, same meaning as on ``TorchSCISolver`` /
                ``SBDSolver``. ``davidson1`` widens its ``max_space`` by
                ``(nroots-1)*4``, so that widening is subtracted here and the
                effective subspace equals what you request. ``None`` (default)
                selects ``4*nroots + 8`` — this solver's historical default (it
                inherited ``sci_kernel_fixed_space``'s flat ``max_space=12``)
                restated in this unit, so the value the kernel receives is
                unchanged. Note ``PySCFSolver``'s default is larger
                (``6*nroots + 8``); each solver's own default is preserved. Pass
                this instead of ``max_space``, which is rejected.
            **kwargs: Additional kwargs forwarded to
                ``sci_kernel_fixed_space`` — e.g. ``tol``, ``lindep``.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in
            input order, each holding the ``self.nroots`` lowest
            :class:`SCIResult`\\ s this solver computes over that batch's
            determinant space, ordered by ascending energy. Energies are
            electronic only.
        """
        reject_init_vectors(type(self).__name__, kwargs)
        # Rejected rather than forwarded: max_space is the same knob in davidson1's
        # own pre-widening units, so accepting both spellings would allow two
        # different subspace sizes to be requested in one call.
        if "max_space" in kwargs:
            raise TypeError(
                "max_space is not accepted; pass davidson_block_size instead. It is "
                "the same knob expressed as the subspace actually restarted at "
                "(davidson1 widens max_space by (nroots-1)*4 internally), so it "
                "carries the same meaning here as on TorchSCISolver and SBDSolver."
            )
        # default_max_space=12 is sci_kernel_fixed_space's own default, which this
        # solver inherited by not passing max_space at all — preserved so the
        # default stays behaviour-identical (effective subspace 4*nroots + 8).
        max_space = davidson_block_size_to_max_space(
            davidson_block_size, self.nroots, default_max_space=12
        )
        if self.verbose:
            print(
                f"[Davidson] hcore.shape={hcore.shape}, eri.shape={eri.shape}, "
                f"len(batches)={len(ci_strs_batches)}, "
                f"per-batch sizes={[(len(b[0]), len(b[1])) for b in ci_strs_batches]}, "
                f"shift_mode={self.shift_mode}"
            )
        super().sci_solver_callable_eigen_states(
            ci_strs_batches, hcore, eri, norb, nelec, spin_sq=spin_sq, max_cycle=max_cycle
        )
        reject_spinless_nelec(type(self).__name__, nelec)
        # Spatial path from here on (the scalar-nelec spinless case raised above).
        nelec = cast(Tuple[int, int], nelec)
        ci_strs_batches = cast(List[Tuple[np.ndarray, np.ndarray]], ci_strs_batches)

        neleca, nelecb = nelec
        spin = 0 if (spin_sq == 0 and neleca == nelecb) else None

        results = []
        for ci_strs in ci_strs_batches:
            e, civec = sci_kernel_fixed_space(
                hcore,
                eri,
                norb,
                nelec,
                ci_strs,
                spin=spin,
                nroots=self.nroots,
                ecore=0,
                shift_mode=self.shift_mode,
                max_cycle=max_cycle,
                max_space=max_space,
                **kwargs,
            )
            # sci_kernel_fixed_space returns a bare (energy, civec) for a single
            # root and parallel ([energies], [civecs]) lists for nroots > 1
            # (already ascending); normalize so the per-root wrapping is uniform.
            energies = [e] if self.nroots == 1 else list(e)
            civecs = [civec] if self.nroots == 1 else list(civec)
            strs_a, strs_b = ci_strs
            batch_states = [
                make_sci_result(
                    energy=root_e,
                    vec=np.asarray(root_c).ravel(),
                    strs_a=strs_a,
                    strs_b=strs_b,
                    norb=norb,
                    nelec=nelec,
                    symmetrize_spin=self.symmetrize_spin,
                )
                for root_e, root_c in zip(energies, civecs)
            ]
            results.append(EigenStates(states=batch_states))

        return results


class DavidsonSolverSpinFilterMixin(SpinFilterMixin, DavidsonSolver):
    """:class:`DavidsonSolver` with composable ``<S^2>`` spin projection.

    Wraps :class:`DavidsonSolver` (the in-house shift-mode Davidson solve) in
    :class:`~sci_solvers.energy_solvers.spin_filter_mixin.SpinFilterMixin`, so a
    ``spin_sq`` target filters the returned root(s) by ``<S^2>``.
    """
