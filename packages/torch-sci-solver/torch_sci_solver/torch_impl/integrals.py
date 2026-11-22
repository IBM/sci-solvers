"""Torch tensors for the SBD integrals.

Light wrapper around ``torch_sci_solver.sbd_ref.OneInt`` / ``TwoInt``: takes
the numpy-backed scalar reference object and ships its arrays as
``float64`` tensors to a target device.

The "Torch integrals" used by ``torch_impl.mult`` are a flat tuple
``(I0, I1, I2_eri, I1_diag, I2_direct, I2_exchange)`` where:
  - ``I0``: Python float (core energy).
  - ``I1``: (2L, 2L) one-electron Hamiltonian, spin-orbital indexed.
  - ``I2_eri``: (2L, 2L, 2L, 2L) full (pq|rs) chemist tensor,
                spin-selection zeros already in place. May be a
                :class:`CompactEri` (opt-in via ``compact_eri=True``),
                which stores only the spatial L^4 tensor and reconstructs
                the dense entries on indexing — 16x less ERI memory.
  - ``I1_diag``: (2L,)  I1.diag(); convenient for zero_excite.
  - ``I2_direct``: (2L, 2L)  (ii|jj).
  - ``I2_exchange``: (2L, 2L) (ij|ji).

We pull out diag / direct / exchange because zero_excite_batch wants
them as separate tensors (vectorized over the basis).

Memory: for 36 orbitals (2L=72), the (2L)^4 ERI tensor is 72^4 × 8 bytes
≈ 215 MB. Trivial on a modern GPU, modest on CPU. If memory becomes
tight, switch to the 8-fold-packed format later — the API surface
here is small enough to absorb that change cleanly.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from torch_sci_solver.sbd_ref.integrals import OneInt, TwoInt, _to_full_eri

# Integer dtypes accepted as advanced indices (torch's own integer index set,
# minus ``bool``: boolean-mask indexing has different axis semantics).
_INDEX_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class CompactEri:
    """Spin-block-compressed stand-in for the dense (2L, 2L, 2L, 2L) ERI.

    An RHF integral set has ``eri[2i+sp, 2j+sp, 2k+sq, 2m+sq] == spatial[i, j, k, m]``
    for the four spin pairs ``(sp, sq) in {0, 1}^2`` and is zero on the other
    twelve spin quads, so the dense tensor is 16x redundant. This wrapper holds
    only the spatial ``(L, L, L, L)`` tensor and reproduces the dense tensor's
    ``__getitem__`` for a 4-tuple of Python ints and/or integer tensors, which
    is the indexing surface the batched matvec kernels use.

    ``dtype``/``device``/``shape``/``element_size()``/``numel()`` mirror the
    dense tensor; ``storage_bytes`` reports the compact footprint. Indexing
    uses torch advanced-indexing broadcasting, so all four entries are
    broadcast together exactly as they would be for the dense tensor.

    Notes:
        With spin-orbital ``p = 2i + sp`` (α at even indices, β at odd), the
        dense entry is

        .. math::

            E_{pqrs} = \\begin{cases}
                S_{\\lfloor p/2 \\rfloor \\lfloor q/2 \\rfloor
                  \\lfloor r/2 \\rfloor \\lfloor s/2 \\rfloor},
                & (p \\bmod 2) = (q \\bmod 2) \\;\\text{and}\\;
                  (r \\bmod 2) = (s \\bmod 2), \\\\
                0, & \\text{otherwise},
            \\end{cases}

        where :math:`S` is the spatial chemist tensor. Slice / ``None`` /
        Ellipsis indexing and ``reshape`` / ``contiguous`` / ``flatten`` raise
        instead of silently returning wrong data; call :meth:`to_dense` to
        materialise the dense tensor explicitly. In particular the Triton
        path, which flattens the ERI for its kernels, must keep a dense
        ``I2_eri``.

    Args:
        spatial: ``(L, L, L, L)`` spatial-orbital ``(ij|kl)`` tensor on the
            target device and dtype.
    """

    __slots__ = ("_spatial",)

    def __init__(self, spatial: Tensor) -> None:
        """Wrap the spatial ``(L, L, L, L)`` tensor holding the compact ERI."""
        if spatial.dim() != 4 or len(set(spatial.shape)) != 1:
            raise ValueError(f"spatial ERI must be (L, L, L, L); got {tuple(spatial.shape)}.")
        self._spatial = spatial

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the dense tensor (the spatial tensor's dtype)."""
        return self._spatial.dtype

    @property
    def device(self) -> torch.device:
        """Device of the dense tensor."""
        return self._spatial.device

    @property
    def shape(self) -> torch.Size:
        """Dense spin-orbital shape ``(2L, 2L, 2L, 2L)``."""
        n_spinorb = 2 * self._spatial.shape[0]
        return torch.Size((n_spinorb, n_spinorb, n_spinorb, n_spinorb))

    @property
    def storage_bytes(self) -> int:
        """Bytes actually stored: the spatial ``L^4`` tensor only."""
        return self._spatial.numel() * self._spatial.element_size()

    def element_size(self) -> int:
        """Bytes per element (same itemsize as the dense tensor)."""
        return self._spatial.element_size()

    def numel(self) -> int:
        """Number of entries of the dense tensor, ``(2L)^4``."""
        n_spinorb = 2 * self._spatial.shape[0]
        return n_spinorb**4

    def to_dense(self) -> Tensor:
        """Materialise the equivalent dense ``(2L, 2L, 2L, 2L)`` ERI tensor.

        Returns:
            A new dense tensor in the spatial tensor's dtype and device: the
            four nonzero spin quads (``(sp, sp, sq, sq)`` for
            ``sp, sq in {0, 1}``) hold the spatial block and the other twelve
            quads are zero.

        Notes:
            Allocating and zero-filling ``(2L)^4`` elements costs
            ``16 * L^4 * element_size()`` bytes — 16x the compact footprint,
            e.g. ``72^4 * 8`` ≈ 215 MB for ``L = 36`` in ``float64``. Prefer
            :meth:`__getitem__` when only a subset of the entries is needed.
        """
        dense = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        for sp in (0, 1):
            for sq in (0, 1):
                dense[sp::2, sp::2, sq::2, sq::2] = self._spatial
        return dense

    def __getitem__(self, key: object) -> Tensor:
        """Gather dense-tensor entries with advanced-indexing broadcasting.

        Only a 4-tuple of Python ints and/or integer tensors is supported;
        anything that needs the materialised dense tensor (a slice, ``None``
        or Ellipsis) raises. Call :meth:`to_dense` first for those access
        patterns.

        Args:
            key: 4-tuple whose entries are Python ints and/or integer
                tensors; the tensors broadcast against each other exactly as
                they would when indexing the dense tensor.

        Returns:
            Tensor of the broadcast index shape (0-d for four Python ints),
            in the spatial tensor's dtype and device.

        Raises:
            TypeError: ``key`` is not a 4-tuple of Python ints and integer
                tensors.
            NotImplementedError: ``key`` uses a slice, ``None`` or Ellipsis.
        """
        if not isinstance(key, tuple):
            raise TypeError(
                "CompactEri supports only a 4-tuple of Python ints and integer tensors; "
                f"got a single {type(key).__name__}."
            )
        if len(key) != 4:
            raise TypeError(f"CompactEri requires exactly 4 indices; got {len(key)}.")
        if any(isinstance(entry, slice) or entry is None or entry is Ellipsis for entry in key):
            raise NotImplementedError(
                "CompactEri does not implement slice / None / Ellipsis indexing; call "
                "to_dense() to materialise the (2L, 2L, 2L, 2L) tensor for that access "
                "pattern."
            )
        indices: list[Tensor] = []
        for entry in key:
            if isinstance(entry, Tensor):
                if entry.dtype not in _INDEX_DTYPES:
                    raise TypeError(
                        f"CompactEri index tensors must be integer dtype; got {entry.dtype}."
                    )
                indices.append(entry)
            elif isinstance(entry, int) and not isinstance(entry, bool):
                indices.append(
                    torch.as_tensor(entry, dtype=torch.int64, device=self._spatial.device)
                )
            else:
                raise TypeError(
                    "CompactEri indices must be Python ints or integer tensors; got "
                    f"{type(entry).__name__}."
                )
        p, q, r, s = indices
        values = self._spatial[p // 2, q // 2, r // 2, s // 2]
        spin_mismatch = ((p & 1) != (q & 1)) | ((r & 1) != (s & 1))
        return values.masked_fill(spin_mismatch, 0)

    def reshape(self, *shape: object) -> Tensor:
        """Reject reshaping: this wrapper has no dense buffer to reshape."""
        _ = shape
        raise NotImplementedError(
            "CompactEri cannot be reshaped into a dense tensor; build "
            "to_torch_integrals(..., compact_eri=False) for a dense I2_eri."
        )

    def contiguous(self) -> Tensor:
        """Reject ``contiguous``: this wrapper has no dense buffer to copy."""
        raise NotImplementedError(
            "CompactEri has no dense tensor to make contiguous; build "
            "to_torch_integrals(..., compact_eri=False) for a dense I2_eri."
        )

    def flatten(self, *args: object) -> Tensor:
        """Reject ``flatten``: this wrapper has no dense buffer to flatten."""
        _ = args
        raise NotImplementedError(
            "CompactEri cannot be flattened into a dense tensor; build "
            "to_torch_integrals(..., compact_eri=False) for a dense I2_eri."
        )


@dataclass
class TorchIntegrals:
    """Bundle of torch tensors needed by the batched matvec.

    All tensors live on the same device. ``I0`` stays as a Python float
    (no autograd / device transfer needed for a scalar).
    """

    I0: float
    I1: Tensor  # (2L, 2L)
    I2_eri: Tensor | CompactEri  # (2L, 2L, 2L, 2L) dense or its compact view
    I1_diag: Tensor  # (2L,)
    I2_direct: Tensor  # (2L, 2L)
    I2_exchange: Tensor  # (2L, 2L)
    n_spinorb: int  # cached for convenience
    device: torch.device
    dtype: torch.dtype
    eri_compact: bool = False  # I2_eri is a CompactEri when True


def _dense_eri_from_integrals(integrals: TorchIntegrals) -> Tensor:
    """Return the dense ``(2L, 2L, 2L, 2L)`` ERI tensor of an integral bundle.

    Args:
        integrals: bundle whose ``I2_eri`` is either a dense tensor or a
            :class:`CompactEri`.

    Returns:
        ``I2_eri`` itself when it is already dense (no copy), otherwise the
        materialised dense tensor from :meth:`CompactEri.to_dense`.

    Notes:
        The compact branch allocates ``(2L)^4`` elements — 16x the stored
        ``L^4`` spatial block, so materialising a large compact ERI can cost
        hundreds of megabytes (see :meth:`CompactEri.to_dense`).
    """
    eri = integrals.I2_eri
    return eri.to_dense() if isinstance(eri, CompactEri) else eri


def _compact_spatial_eri(eri: np.ndarray) -> np.ndarray | None:
    """Return the spatial ``(L, L, L, L)`` block iff ``eri`` has the RHF spin structure.

    The dense spin-orbital tensor of an RHF integral set is the same spatial
    chemist tensor on all four same-spin-within-bra / same-spin-within-ket
    quads and zero everywhere else::

        eri[2i + sp, 2j + sp, 2k + sq, 2m + sq] == spatial[i, j, k, m]
            for (sp, sq) in {(0, 0), (0, 1), (1, 0), (1, 1)}

    and ``eri[p, q, r, s] == 0`` whenever ``(p % 2) != (q % 2)`` or
    ``(r % 2) != (s % 2)``.

    Args:
        eri: host ``(2L, 2L, 2L, 2L)`` dense ERI array.

    Returns:
        The ``(L, L, L, L)`` spatial block (a strided view of ``eri``) when
        the structure holds exactly, else ``None``.

    Notes:
        The check is exact (no tolerance): the dense tensor is assembled by
        copying the spatial block into each quad, so a valid tensor matches
        bit-for-bit. Cost is one pass over the four nonzero ``L^4`` blocks
        (compared against the first) plus one pass over the twelve zero quads,
        all on the host before any device allocation.
    """
    if eri.ndim != 4 or eri.shape[0] % 2:
        return None
    spatial = eri[0::2, 0::2, 0::2, 0::2]
    for sp, sq in ((0, 0), (1, 1), (0, 1), (1, 0)):
        if not np.array_equal(eri[sp::2, sp::2, sq::2, sq::2], spatial):
            return None
    for sp, tp in ((0, 0), (1, 1), (0, 1), (1, 0)):
        for sq, tq in ((0, 0), (1, 1), (0, 1), (1, 0)):
            if sp == tp and sq == tq:
                continue
            if np.any(eri[sp::2, tp::2, sq::2, tq::2]):
                return None
    return spatial


def to_torch_integrals(
    I0: float,
    I1: OneInt,
    I2: TwoInt,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    compact_eri: bool = False,
) -> TorchIntegrals:
    """Convert scalar-reference integrals to a TorchIntegrals bundle.

    Args:
        I0: Python float, the core energy.
        I1: OneInt object (numpy-backed).
        I2: TwoInt object (numpy-backed). Spin-selection zeros are already
            present (caller built it via setup_integrals_from_pyscf_h1_eri).
        device: target torch device. None → CPU.
        dtype: float dtype, default torch.float64.
        compact_eri: store the ERI in the 16x smaller :class:`CompactEri`
            form instead of the dense (2L)^4 tensor (default ``False``).
            The RHF spin-block structure is verified on the host numpy array
            before any device allocation: the four nonzero spin quads are
            compared exactly against the spatial ``L^4`` block and the other
            twelve quads are checked to be zero (one pass over the four
            blocks plus the zero-quad pass). If the check fails, the dense
            conversion is used and ``eri_compact`` is ``False``. The Triton
            matvec flattens ``I2_eri`` and therefore needs a dense tensor.

    Returns:
        TorchIntegrals dataclass holding the converted tensors; when
        ``compact_eri`` succeeds, ``I2_eri`` is a :class:`CompactEri` and
        ``eri_compact`` is ``True``.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    n_spinorb = I1.n_spinorb
    assert I2.n_spinorb == n_spinorb, "I1 / I2 spin-orbital count mismatch"

    # Verify the spatial block structure on the host before materialising
    # anything on the device; fall back to dense when it does not hold.
    spatial = _compact_spatial_eri(I2._eri) if compact_eri else None
    eri_compact = spatial is not None

    I1_t = torch.tensor(I1._h, dtype=dtype, device=device)

    if spatial is not None:
        I2_t: Tensor | CompactEri = CompactEri(torch.tensor(spatial, dtype=dtype, device=device))
    else:
        I2_t = torch.tensor(I2._eri, dtype=dtype, device=device)

    # Diagonal of I1 (used by zero_excite)
    I1_diag = torch.diagonal(I1_t).contiguous()

    I2_direct = torch.tensor(I2._direct, dtype=dtype, device=device)
    I2_exchange = torch.tensor(I2._exchange, dtype=dtype, device=device)

    return TorchIntegrals(
        I0=float(I0),
        I1=I1_t,
        I2_eri=I2_t,
        I1_diag=I1_diag,
        I2_direct=I2_direct,
        I2_exchange=I2_exchange,
        n_spinorb=n_spinorb,
        device=device,
        dtype=dtype,
        eri_compact=eri_compact,
    )


def _check_hermitian_spinless(
    hcore: np.ndarray, eri: np.ndarray, *, atol: float = 1e-7, rtol: float = 1e-5
) -> None:
    """Validate the two Hermitian symmetries a spinless CI operator needs.

    Like :func:`_check_hermitian` (the spatial complex path), the spinless operator
    ``0.5 * sum (pq|rs) a_p^ a_r^ a_s a_q`` is Hermitian only when ``eri``
    carries *both* ``(pq|rs) == conj((qp|sr))`` and ``(pq|rs) == (rs|pq)`` —
    mirrors ``sci_solvers.spinless_ci_problem.build_spinless_complex_ci_problem``'s
    own check exactly, including its relaxed default tolerances: a two-
    component SCF (SOC/GHF) export is Hermitian only up to round-off that
    scales with the integral magnitude, which a pure ``atol`` cannot absorb
    (see AGENTS.md's "arpack eri Hermiticity tolerance" note).

    Raises:
        ValueError: naming which symmetry failed and the largest deviation.
    """
    h_dev = np.max(np.abs(hcore - hcore.conj().T))
    if h_dev > atol + rtol * np.max(np.abs(hcore)):
        raise ValueError(
            f"hcore is not Hermitian (hcore != hcore.conj().T): max deviation "
            f"{h_dev:.3e} > atol={atol:.1e} + rtol={rtol:.1e}*max|hcore|."
        )
    lhs = eri
    rhs_pq = np.conj(np.transpose(eri, (1, 0, 3, 2)))
    if not np.allclose(lhs, rhs_pq, atol=atol, rtol=rtol):
        dev = np.max(np.abs(lhs - rhs_pq))
        raise ValueError(
            f"eri does not satisfy (pq|rs) == conj((qp|sr)): max deviation "
            f"{dev:.3e} > atol={atol:.1e} + rtol={rtol:.1e}*|entry|."
        )
    rhs_rs = np.transpose(eri, (2, 3, 0, 1))
    if not np.allclose(lhs, rhs_rs, atol=atol, rtol=rtol):
        dev = np.max(np.abs(lhs - rhs_rs))
        raise ValueError(
            f"eri does not satisfy (pq|rs) == (rs|pq): max deviation "
            f"{dev:.3e} > atol={atol:.1e} + rtol={rtol:.1e}*|entry|."
        )


def to_torch_integrals_spinless(
    hcore: np.ndarray,
    eri: np.ndarray,
    core_energy: float = 0.0,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.complex128,
    *,
    check_hermitian: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
) -> TorchIntegrals:
    """Build a :class:`TorchIntegrals` bundle for a flat spinless operator.

    The spinless counterpart of :func:`setup_integrals_from_pyscf_h1_eri_vec`
    + :func:`to_torch_integrals`: takes ``hcore``/``eri`` *directly* as flat
    ``nmo``-spin-orbital quantities, with **no spin-doubling and no
    same-spin quadrant masking** — a genuinely spinless Hamiltonian (from a
    GHF / spin-orbit-coupled / generalized-spinor calculation) has no
    alpha/beta factorization to double from, and its off-diagonal spin
    blocks (spin-flip) are generically non-zero. Mirrors
    ``sci_solvers.spinless_ci_problem._ComplexSpinOrbitalIntegrals`` but
    returns a torch-tensor bundle instead of a ``.value``-style adapter.

    Args:
        hcore: ``(nmo, nmo)`` complex Hermitian one-electron spin-orbital
            integrals.
        eri: ``(nmo, nmo, nmo, nmo)`` complex chemist-order two-electron
            integrals, carrying both Hermitian symmetries (see
            :func:`_check_hermitian_spinless`).
        core_energy: nuclear+core scalar (``I0``).
        device: target torch device. ``None`` -> CPU.
        dtype: torch dtype, default ``torch.complex128`` (a spinless operator
            is complex Hermitian by construction).
        check_hermitian: validate ``hcore``/``eri`` Hermiticity before
            building (default ``True``).
        atol, rtol: Hermiticity tolerances forwarded to
            :func:`_check_hermitian_spinless`.

    Returns:
        A :class:`TorchIntegrals` bundle with ``n_spinorb = nmo``, usable by
        :func:`torch_sci_solver.torch_impl.mult_spinless.mult_spinless_torch`
        and :func:`torch_sci_solver.torch_impl.determinants.zero_excite_batch_spinless`.

    Raises:
        ValueError: shapes are inconsistent, or (when ``check_hermitian``)
            the integrals are not Hermitian.
    """
    hcore = np.asarray(hcore)
    eri = np.asarray(eri)
    nmo = int(hcore.shape[0])
    if hcore.shape != (nmo, nmo):
        raise ValueError(f"hcore shape {hcore.shape} != ({nmo}, {nmo}).")
    if eri.shape != (nmo, nmo, nmo, nmo):
        raise ValueError(f"eri shape {eri.shape} != ({nmo},)*4.")
    if check_hermitian:
        _check_hermitian_spinless(hcore, eri, atol=atol, rtol=rtol)

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    I1_t = torch.tensor(hcore, dtype=dtype, device=device)
    I2_t = torch.tensor(eri, dtype=dtype, device=device)
    I1_diag = torch.diagonal(I1_t).contiguous()

    p = np.arange(nmo).reshape(-1, 1)
    q = np.arange(nmo).reshape(1, -1)
    I2_direct = torch.tensor(eri[p, p, q, q], dtype=dtype, device=device)
    I2_exchange = torch.tensor(eri[p, q, q, p], dtype=dtype, device=device)

    return TorchIntegrals(
        I0=float(core_energy),
        I1=I1_t,
        I2_eri=I2_t,
        I1_diag=I1_diag,
        I2_direct=I2_direct,
        I2_exchange=I2_exchange,
        n_spinorb=nmo,
        device=device,
        dtype=dtype,
    )


def _check_hermitian(h1_spatial: np.ndarray, eri_full: np.ndarray, *, atol: float = 1e-8) -> None:
    """Validate the Hermiticity a complex integral set must satisfy.

    Only called on the complex path (see ``setup_integrals_from_pyscf_h1_eri_vec``)
    — the real path never needed this because a real symmetric ``h1_spatial``
    and a real ``eri_spatial`` with the usual chemist 8-fold symmetry are
    trivially Hermitian.

    Notes:
        Requires ``h1_spatial == h1_spatial.conj().T``, plus both chemist-order
        ERI symmetries: ``(pq|rs) == conj((qp|sr))`` (Hermitian conjugation)
        and ``(pq|rs) == (rs|pq)`` (particle exchange, without conjugation).
        Conjugating the particle-exchange symmetry would incorrectly reject
        valid imaginary ERI components.

    Raises:
        ValueError: naming which tensor and the largest violation found.
    """
    h1_dev = np.max(np.abs(h1_spatial - h1_spatial.conj().T))
    if h1_dev > atol:
        raise ValueError(
            f"h1_spatial is not Hermitian (h1 != h1.conj().T): max deviation "
            f"{h1_dev:.3e} > atol={atol:.1e}. A complex one-body Hamiltonian must "
            "be Hermitian for the operator to have a real spectrum."
        )
    eri_dev = np.max(np.abs(eri_full - eri_full.transpose(1, 0, 3, 2).conj()))
    if eri_dev > atol:
        raise ValueError(
            "eri_spatial violates Hermitian conjugate symmetry "
            "((pq|rs) != conj((qp|sr))): "
            f"max deviation {eri_dev:.3e} > atol={atol:.1e}."
        )
    eri_dev = np.max(np.abs(eri_full - eri_full.transpose(2, 3, 0, 1)))
    if eri_dev > atol:
        raise ValueError(
            "eri_spatial violates particle-exchange symmetry ((pq|rs) != (rs|pq)): "
            f"max deviation {eri_dev:.3e} > atol={atol:.1e}."
        )


def setup_integrals_from_pyscf_h1_eri_vec(
    h1_spatial: np.ndarray, eri_spatial: np.ndarray, core_energy: float
) -> Tuple[float, OneInt, TwoInt]:
    """Numpy-vectorized builder for the ``(I0, OneInt, TwoInt)`` triple.

    Same contract as :func:`torch_sci_solver.sbd_ref.integrals.
    setup_integrals_from_pyscf_h1_eri` but replaces its O(L⁴) Python
    quad-loop with four numpy slice assignments — the (2L)⁴ ERI tensor is
    populated one spin-block quadrant at a time (``αα|αα``, ``αα|ββ``,
    ``ββ|αα``, ``ββ|ββ``) via strided slicing, and the ``direct`` /
    ``exchange`` caches come out of ``_eri`` with fancy indexing.

    Bit-identical to the scalar reference; checked at ``L=10`` in
    ``tests/test_torch_helpers.py``. At L=36 this cuts the build
    time from ~1.4 s to ~60 ms — a real ``t_integrals_build`` reduction,
    since ``qiskit_adapter._solve_one_subspace`` calls this on every
    integrals-cache miss (once per unique Hamiltonian).

    Complex Hermitian ``h1_spatial``/``eri_spatial`` (dtype ``complex64`` or
    ``complex128``) is also accepted: the same slice assignments carry the
    complex values through unchanged, and ``I1``/``I2`` are allocated with a
    matching complex dtype instead of the default ``float64``. Callers must
    pass Hermitian integrals (``h1_spatial == h1_spatial.conj().T`` and the
    matching 4-index Hermitian symmetry for ``eri_spatial``); this is
    validated by :func:`_check_hermitian` and raises ``ValueError`` on
    failure rather than silently building a non-Hermitian operator.

    Args:
        h1_spatial: (L, L) one-electron h matrix in spatial-orbital basis.
        eri_spatial: 4-index chemist ``(ij|kl)`` tensor. Any of the shapes
            :func:`torch_sci_solver.sbd_ref.integrals._to_full_eri`
            accepts (full, 4-fold, or 8-fold packed) works — it is
            reshaped to full ``(L, L, L, L)`` first. Complex input must
            already be full ``(L, L, L, L)`` (see ``_to_full_eri``).
        core_energy: nuclear+core scalar (the ``I0`` in SBD).

    Returns:
        ``(I0, OneInt, TwoInt)`` — same triple ``setup_integrals_from_pyscf_h1_eri``
        returns; the objects are usable by :func:`to_torch_integrals`
        without further changes.
    """
    L = h1_spatial.shape[0]
    n_spinorb = 2 * L
    eri_full = _to_full_eri(eri_spatial, L)

    is_complex = np.iscomplexobj(h1_spatial) or np.iscomplexobj(eri_full)
    if is_complex:
        _check_hermitian(h1_spatial, eri_full)
    int_dtype = np.result_type(h1_spatial, eri_full) if is_complex else np.float64

    I1 = OneInt(n_spinorb, dtype=int_dtype)
    # RHF: α and β blocks share the spatial one-electron matrix h(i, j).
    # Interleaved layout: even indices are α, odd are β (see sbd_ref module doc).
    I1._h[0::2, 0::2] = h1_spatial
    I1._h[1::2, 1::2] = h1_spatial

    I2 = TwoInt(n_spinorb, dtype=int_dtype)
    # Populate the four same-spin-within-bra × same-spin-within-ket quadrants.
    # Mixed-spin quadrants (e.g. αβ|αα) stay 0 as initialized, which is the
    # spin-selection rule the scalar TwoInt.value enforces at read time.
    I2._eri[0::2, 0::2, 0::2, 0::2] = eri_full  # αα|αα
    I2._eri[0::2, 0::2, 1::2, 1::2] = eri_full  # αα|ββ
    I2._eri[1::2, 1::2, 0::2, 0::2] = eri_full  # ββ|αα
    I2._eri[1::2, 1::2, 1::2, 1::2] = eri_full  # ββ|ββ

    # direct[p, q] = eri[p, p, q, q]: (p, p) and (q, q) are trivially same-spin,
    # so this entry is always in a populated quadrant — no spin filter needed.
    # exchange[p, q] = eri[p, q, q, p]: 0 in _eri when p%2 != q%2 (mixed-spin
    # quadrant is 0), so the fancy-index read already carries the spin
    # selection rule; matches scalar TwoInt.value(p, q, q, p).
    p = np.arange(n_spinorb).reshape(-1, 1)
    q = np.arange(n_spinorb).reshape(1, -1)
    I2._direct = I2._eri[p, p, q, q]
    I2._exchange = I2._eri[p, q, q, p]

    return float(core_energy), I1, I2
