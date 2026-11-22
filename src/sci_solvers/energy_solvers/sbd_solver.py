"""
SBD sci_solver backed by the in-process ``sbd-eigensolver`` package.

Wraps RIKEN's open-source ``sbd`` package (``import sbd``, pybind11 in-process
API — no subprocess) in this package's :class:`EnergyCalculatorBase` contract.
Two diagonalization modes are exposed through the ``mode`` constructor argument:

- ``mode="tpb"`` (Tensor-Product Basis, the default): each batch's subspace is
  the Cartesian product of its alpha and beta CI-string lists. This delegates
  directly to the package's :func:`sbd.sbd_solver.solve_sci_batch`.

- ``mode="gdb"`` (General-Determinant Basis): each batch's subspace is an
  explicit sparse determinant list. Following ``torch_sci_solver``'s established
  ``mode="gdb"`` contract, the ``(strs_a, strs_b)`` arrays are treated as
  **parallel length-N pair lists** — index ``i`` defines determinant
  ``(alpha_i, beta_i)`` — NOT an outer product. This is deliberately *not* a
  drop-in for qiskit-addon-sqd's per-spin recovery loop: feeding it a
  product-style ``(strs_a, strs_b)`` silently diagonalizes a different subspace.

Spin handling
-------------
The ``sbd`` package has NO spin-projection mechanism (``spin_sq`` is accepted but
ignored by both config classes and both diag functions). This wrapper therefore
rejects any non-``None`` ``spin_sq`` kwarg.

``sbd`` computes in *interleaved-Wick* parity, which differs from the spin-block
convention PySCF and ``SCIState`` assume by a per-determinant-pair sign. In
``mode="tpb"``, ``sbd`` (from ``sbd-eigensolver`` 1.7.1) converts the amplitudes of the
``SCIState`` it returns itself, so downstream ``<S^2>`` and RDMs are correct without a
further conversion here; converting again would put the signs back and give a singlet a
spuriously large ``<S^2>`` even though the energy is exact. ``mode="gdb"`` instead
returns a flat :class:`~torch_sci_solver.torch_impl.pairs.PairSCIState` over the
diagonalized ``(alpha_i, beta_i)`` pairs, left in interleaved-Wick parity — the same
contract and convention as the general-algebra solvers' ``mode="gdb"``
(``torch_sci_solver``'s pair RDM builders assume it by default).

Returned energy
---------------
Each ``SCIResult.energy`` is the ELECTRONIC energy only; callers add ``e_nuc``
themselves, matching qiskit-addon-sqd's convention. When integrals are handed in
memory the package writes a regenerated FCIDUMP with ECORE=0; when
``fcidump_path`` is given, the file's ECORE is subtracted so the contract holds.

Installation
------------
``sbd`` has no PyPI wheel — it compiles at install against local MPI/BLAS. Prepare
the toolchain with ``scripts/energy_solvers/build_sbd.sh``, then
``uv sync --extra sbd``. Because it is an optional extra, a plain ``uv sync``
never builds it and the base install cannot break.

Convergence (``mode="tpb"`` only)
----------------------------------
sbd's native ``Davidson(...)`` is a ``void`` C++ function: it exits identically
whether it converged (``norm_R < eps``), exhausted ``max_it`` restarts, or hit the
``max_time`` wall-clock cap, and reports none of that back through the pybind
boundary — every exit path returns the same energy/wavefunction shape with no
convergence signal attached. ``mode="tpb"`` closes this gap itself (see
:meth:`SBDSolver._tpb_convergence_state`) by recomputing ``‖H*W - E*W‖``
independently, via the same :class:`~sci_solvers.eigen_problem.EigenProblem`
operator the general-algebra solvers already share, and comparing it to
``sbd_config["eps"]`` — mirroring
:class:`~sci_solvers.energy_solvers.general_algebra_solvers.lobpcg.LobpcgSolver`'s
identical self-check for scipy's equally silent ``lobpcg``. ``mode="gdb"`` does not
yet have this check.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import os
import warnings
from pathlib import Path
from typing import Any, List, Tuple, cast

import numpy as np
import torch
from qiskit_addon_sqd.fermion import SCIResult
from torch_sci_solver.torch_impl.pairs import PairSCIState

from sci_solvers._mpi_singleton import configure_singleton_mpi
from sci_solvers.eigen_problem import EigenProblem
from sci_solvers.eigen_states import EigenStates, SCIResultWithConvergence
from sci_solvers.energy_calculation_base import (
    EnergyCalculatorBase,
    reject_init_vectors,
    reject_spinless_nelec,
)
from sci_solvers.hamiltonian import HamiltonianData

_VALID_MODES = ("tpb", "gdb")

# sbd's own TPB default (mirrors the GDB default `_create_gdb_config` sets below,
# since the native library does not expose its TPB_SBD default to Python); used
# only as a fallback in `_tpb_convergence_state` when `sbd_config` doesn't set
# `eps` explicitly. The benchmark configs that exercise this solver always set
# `eps` themselves, so this fallback is a last resort, not the common path.
_TPB_DEFAULT_EPS = 1e-8


def _cuda_runtime_conflict(maps_path: str = "/proc/self/maps") -> str | None:
    """Describe a CUDA-runtime mix that makes sbd's GPU backend return ``Energy = 0``, else ``None``.

    sbd's Thrust backend is built by NVHPC against CUDA 12 and links ``libcudart.so.12``. When
    ``libcudart.so.13`` (a torch built for CUDA 13) is already mapped and no ``libcudart.so.12``
    is, sbd's CUDA calls land on a second runtime that initialised after torch's, and the
    diagonalization silently returns zeros. The conflict is decided from the process's loaded
    libraries, so it does not depend on which torch version is installed; it is absent when torch
    ships ``libcudart.so.12`` or when sbd's runtime was loaded first.

    Args:
        maps_path: Memory-map listing to inspect; overridable for tests.

    Returns:
        A one-line description of the offending runtime, or ``None`` when there is no conflict or
        the listing cannot be read (non-Linux).
    """
    try:
        text = Path(maps_path).read_text()
    except OSError:
        return None
    names = {
        Path(line.split(None, 5)[-1]).name for line in text.splitlines() if "libcudart" in line
    }
    if any(n.startswith("libcudart.so.13") for n in names) and not any(
        n.startswith("libcudart.so.12") for n in names
    ):
        return ", ".join(sorted(names))
    return None


class SBDSolver(EnergyCalculatorBase):
    """
    sci_solver wrapping RIKEN's in-process ``sbd-eigensolver`` package.

    ``mode="tpb"`` diagonalizes the alpha-beta product subspace (delegated to the
    package's batch helper); ``mode="gdb"`` diagonalizes an explicit sparse list
    of ``(alpha_i, beta_i)`` determinant pairs. See the module docstring for the
    GDB parallel-pairs semantics, spin handling, and installation notes.
    """

    supports_multiroot = False  # one ground state per batch

    def __init__(
        self,
        *,
        symmetrize_spin: bool,
        hamiltonian: HamiltonianData | str | os.PathLike[str] | None = None,
        mode: str = "tpb",
        device: str = "cpu",
        sbd_config: dict[str, Any] | None = None,
        fcidump_path: str | Path | None = None,
        temp_dir: str | Path | None = None,
        clean_temp_dir: bool = True,
        compute_rdm2: bool = False,
        verbose: bool = False,
        verify_convergence: bool = False,
    ):
        """Initialize the SBDSolver.

        Args:
            symmetrize_spin: Forwarded to :class:`EnergyCalculatorBase`.
            hamiltonian: Optional bound Hamiltonian (instance or FCIDUMP path).
            mode: ``"tpb"`` (alpha-beta product subspace, default) or ``"gdb"``
                (explicit parallel-pairs determinant list). See the module
                docstring for the GDB semantics.
            device: Compute device passed to ``sbd``'s ``DeviceConfig`` —
                ``"cpu"`` (default), ``"gpu"`` (Thrust), ``"gpu-omp"``, or
                ``"auto"``.
            sbd_config: Optional dict of SBD tunables (e.g. ``method``, ``eps``,
                ``max_it``) overriding the package defaults.
            fcidump_path: If set, ``sbd`` loads this FCIDUMP directly (its ECORE
                is subtracted so returned energies stay electronic). Must be
                visible to every MPI rank on multi-node runs.
            temp_dir: Directory for the package's intermediate files.
            clean_temp_dir: Delete those intermediate files on exit.
            compute_rdm2: When ``True`` set SBD's ``do_rdm = 1`` so each result
                carries the spin-summed ``rdm1``/``rdm2`` (assembled by
                ``sbd.sbd_solver.assemble_rdms``). An explicit ``do_rdm`` in
                ``sbd_config`` takes precedence.
            verbose: Print solver-specific progress output.
            verify_convergence: ``mode="tpb"`` only. When ``True`` the returned
                wavefunction is re-checked with an independent torch matvec
                (:meth:`_tpb_convergence_state`) and the result carries a ``state``
                of ``"converged"``/``"not_converged"``. That check costs about as
                much time as the sbd solve itself plus extra GPU memory, so it is
                off by default; results then carry no ``state``.

        Raises:
            ValueError: If ``mode`` is neither ``"tpb"`` nor ``"gdb"``.
        """
        # Validate before the lazy import so a bad `mode` raises a clear
        # ValueError even when the optional `sbd` backend isn't installed.
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

        # A plain (non-mpirun) `import sbd` does a singleton MPI_Init that hangs
        # on an HPC login node probing the cluster fabric; steer a detected
        # singleton onto a safe transport first. No-op under mpirun/srun, so
        # multi-node jobs keep the fabric. The transport is DEVICE-AWARE: the GPU
        # backend passes CUDA device pointers into MPI collectives, so it needs
        # the CUDA-aware (UCX) transport — the CPU shared-memory transport would
        # segfault on the device buffer. Passing `device` here upgrades the
        # transport accordingly (overriding the CPU default the sci_solvers
        # package import may have set). Must run BEFORE the import below, which is
        # what triggers MPI_Init.
        configure_singleton_mpi(device=device)

        # sbd's GPU backend links libcudart.so.12; a CUDA-13 torch imported first leaves a second
        # runtime in the process and the backend may then silently return Energy = 0. Warn, don't stop.
        conflict = _cuda_runtime_conflict() if device.lower().startswith("gpu") else None
        if conflict is not None:
            warnings.warn(
                f"sbd's GPU backend needs the CUDA 12 runtime (libcudart.so.12), but this process "
                f"already loaded a CUDA 13 one ({conflict}), typically via a CUDA-13 torch imported "
                "first. sbd may then silently return Energy = 0. Either import sbd's GPU backend before "
                "torch (e.g. `import sbd; sbd.get_backend('gpu')` first), start python with "
                "LD_PRELOAD=<NVHPC cuda/lib64>/libcudart.so.12, or install a CUDA 12 torch with "
                "`uv sync --extra sbd-gpu`.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Lazy import so the class stays importable/registrable without the
        # optional `sbd` backend installed (mirrors TorchSCISolver).
        import sbd
        from sbd.device_config import DeviceConfig

        super().__init__(hamiltonian=hamiltonian, symmetrize_spin=symmetrize_spin, verbose=verbose)

        self.mode = mode
        self._sbd = sbd
        self._device_config = DeviceConfig(device=device)
        self.sbd_config = dict(sbd_config) if sbd_config is not None else None
        self._warn_on_unknown_sbd_config(sbd, mode, device, self.sbd_config)
        self._fcidump_path = str(Path(fcidump_path).resolve()) if fcidump_path is not None else None
        self.temp_dir = str(temp_dir) if temp_dir is not None else None
        self.clean_temp_dir = bool(clean_temp_dir)
        self.compute_rdm2 = bool(compute_rdm2)
        self.verify_convergence = bool(verify_convergence)

    @staticmethod
    def _warn_on_unknown_sbd_config(
        sbd: Any, mode: str, device: str, sbd_config: dict[str, Any] | None
    ) -> None:
        """Warn about ``sbd_config`` keys the SBD config object will silently drop.

        Both this wrapper's GDB config builder and the package's own TPB builder
        apply overrides with a ``hasattr`` guard (``if hasattr(cfg, key):
        setattr(...)``), so a typo like ``max_iter`` for ``max_it`` is dropped
        without a trace and the run quietly uses the default. We build the config
        object for the selected ``mode`` and warn (once, per unknown key) about any
        key it does not expose, turning a silent misconfiguration into a visible
        one. Best-effort: if we cannot introspect the object (e.g. an unexpected
        backend build), we skip the check rather than block construction.
        """
        if not sbd_config:
            return
        try:
            factory = sbd.GDB_SBD if mode == "gdb" else sbd.TPB_SBD
            probe = factory(device=device)
            unknown = sorted(k for k in sbd_config if not hasattr(probe, k))
            known = sorted(a for a in dir(probe) if not a.startswith("_"))
        except Exception:  # pragma: no cover - defensive: never block construction
            return
        if unknown:
            warnings.warn(
                f"SBDSolver: sbd_config key(s) {unknown} are not recognized by the "
                f"{mode!r} SBD config and will be ignored (check for typos, e.g. "
                f"'max_it' not 'max_iter'). Recognized tunables: {', '.join(known)}.",
                stacklevel=2,
            )

    def _verification_device(self) -> str:
        """Torch device for :meth:`_tpb_convergence_state`'s residual recompute.

        Matches sbd's own backend (``self._device_config.device``) rather than
        :class:`~sci_solvers.eigen_problem.EigenProblem`'s own ``"cpu"`` default:
        this benchmark's subspaces run up to the hundreds-of-millions-dim regime,
        far too large for a CPU matvec to be a cheap verification step when sbd
        itself ran on GPU.
        """
        if self._device_config.device == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _tpb_convergence_state(
        self,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        result: SCIResult,
    ) -> str:
        """Recompute ``‖H*W - E*W‖`` independently and compare it to sbd's own ``eps``.

        See the module docstring's "Convergence" section for why sbd itself cannot
        report this. Builds the same :class:`~sci_solvers.eigen_problem.EigenProblem`
        operator the general-algebra solvers already share from ``(strs_a, strs_b)``
        -- the exact subspace sbd was asked to solve, not read back from ``result``,
        since ``mode="tpb"`` never selects a subset of what it was given -- and
        applies it to the returned wavefunction, which ``sbd`` has already put in
        spin-block parity.

        Args:
            strs_a: Alpha CI strings for this batch.
            strs_b: Beta CI strings for this batch.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron integrals, chemist notation.
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``.
            result: The ``SCIResult`` sbd returned for this batch, in spin-block
                parity.

        Returns:
            ``"converged"`` if the recomputed residual is below ``eps``
            (``self.sbd_config["eps"]``, falling back to ``_TPB_DEFAULT_EPS`` when
            not set), else ``"not_converged"``.
        """
        problem = EigenProblem(
            strs_a, strs_b, hcore, eri, norb, nelec, device=self._verification_device()
        )
        w = np.asarray(result.sci_state.amplitudes, dtype=np.float64).ravel()
        norm_w = np.linalg.norm(w)
        if norm_w > 0:
            w = w / norm_w
        residual = float(np.linalg.norm(problem.matvec(w) - result.energy * w))
        eps = (self.sbd_config or {}).get("eps", _TPB_DEFAULT_EPS)
        return "converged" if residual < eps else "not_converged"

    def _with_recomputed_convergence(
        self,
        result: SCIResult,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
    ) -> SCIResultWithConvergence:
        """Wrap one ``mode="tpb"`` ``SCIResult`` with an independently-checked ``state``.

        See :meth:`_tpb_convergence_state` for the recompute itself.
        """
        state = self._tpb_convergence_state(strs_a, strs_b, hcore, eri, norb, nelec, result)
        return SCIResultWithConvergence(
            energy=result.energy,
            sci_state=result.sci_state,
            orbital_occupancies=result.orbital_occupancies,
            rdm1=result.rdm1,
            rdm2=result.rdm2,
            state=state,
        )

    def sci_solver_callable_eigen_states(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]] | List[np.ndarray],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int] | int,
        spin_sq: float | None = None,
        **kwargs: Any,
    ) -> List[EigenStates]:
        """
        Diagonalize each batch via the ``sbd`` package.

        Args:
            ci_strs_batches: List of ``(strs_a, strs_b)`` int64-array tuples. In
                ``mode="gdb"`` each pair is interpreted as parallel lists (see
                the module docstring), so ``len(strs_a) == len(strs_b)`` is
                required per batch.
            hcore: One-electron integrals, shape ``(norb, norb)``.
            eri: Two-electron repulsion integrals (any pyscf-accepted packing).
            norb: Number of spatial orbitals.
            nelec: ``(n_alpha, n_beta)``. A scalar ``nelec`` (the spinless
                flat spin-orbital path) is rejected with
                ``NotImplementedError`` — this solver does not implement it.
            spin_sq: Must be ``None``; ``sbd`` has no spin-projection mechanism.

        Returns:
            One :class:`~sci_solvers.eigen_states.EigenStates` per batch, in
            input order, each wrapping the single ground-state
            :class:`SCIResult`. Energies are electronic only.

        Raises:
            NotImplementedError: If ``nelec`` is a scalar (the spinless path).
        """
        reject_init_vectors(type(self).__name__, kwargs)
        assert spin_sq is None, "SBDSolver does not support spin projection; spin_sq must be None"
        # Rejected rather than accepted-and-ignored: sbd's iteration cap lives in
        # ``sbd_config`` (``max_it``), so silently swallowing max_cycle here would let
        # a caller believe they had bounded the solve when nothing had changed.
        if "max_cycle" in kwargs:
            raise TypeError(
                "max_cycle is not accepted by SBDSolver: sbd's Davidson is configured "
                "through sbd_config (its `max_it` key) at construction. max_cycle "
                "denotes a flat total-matvec budget on the PySCF-family solvers and has "
                "no sbd equivalent."
            )
        super().sci_solver_callable_eigen_states(
            ci_strs_batches, hcore, eri, norb, nelec, spin_sq=spin_sq, **kwargs
        )
        reject_spinless_nelec(type(self).__name__, nelec)
        # Spatial path from here on (the scalar-nelec spinless case raised above).
        nelec = cast(Tuple[int, int], nelec)
        ci_strs_batches = cast(List[Tuple[np.ndarray, np.ndarray]], ci_strs_batches)

        if self.mode == "tpb":
            sbd_config = self.sbd_config
            if self.compute_rdm2 and "do_rdm" not in (sbd_config or {}):
                sbd_config = {**(sbd_config or {}), "do_rdm": 1}
            results = self._sbd.sbd_solver.solve_sci_batch(
                ci_strs_batches,
                hcore,
                eri,
                int(norb),
                (int(nelec[0]), int(nelec[1])),
                spin_sq=None,
                sbd_config=sbd_config,
                temp_dir=self.temp_dir,
                clean_temp_dir=self.clean_temp_dir,
                device_config=self._device_config,
                fcidump_path=self._fcidump_path,
            )
            if self.verify_convergence:
                results = [
                    self._with_recomputed_convergence(
                        r, strs_a, strs_b, hcore, eri, int(norb), nelec
                    )
                    for r, (strs_a, strs_b) in zip(results, ci_strs_batches)
                ]
        else:
            results = self._solve_gdb_batch(
                ci_strs_batches, hcore, eri, int(norb), (int(nelec[0]), int(nelec[1]))
            )
        return [EigenStates(states=[r]) for r in results]

    def _solve_gdb_batch(
        self,
        ci_strs_batches: List[Tuple[np.ndarray, np.ndarray]],
        hcore: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
    ) -> List[SCIResult]:
        """
        Diagonalize each batch with GDB, one ``sbd.gdb_diag`` call per batch.

        The FCIDUMP is resolved once (loaded from ``fcidump_path`` or regenerated
        with ECORE=0 from the integrals) and reused across batches, mirroring
        :func:`sbd.sbd_solver.solve_sci_batch`. Each ``(strs_a, strs_b)`` pair is
        treated as parallel length-N lists: determinant ``i`` is ``(alpha_i,
        beta_i)``, encoded into one ``2 * norb``-bit configuration with bit
        ``2k`` = alpha orbital ``k`` and bit ``2k + 1`` = beta orbital ``k`` (the
        convention shared by ``sbd.gdb_diag`` and ``torch_sci_solver``).
        """
        from sbd.sbd_solver import (
            SBD_DEFAULT_BIT_LENGTH,
            _load_or_regenerate_fcidump,
            _make_sbd_dir,
        )

        sbd = self._sbd
        backend = sbd.get_backend(self._device_config.device)
        mpi_comm = sbd.get_comm()
        mpi_rank = mpi_comm.Get_rank()

        sbd_dir, owns_sbd_dir = _make_sbd_dir(mpi_comm, mpi_rank, self.temp_dir)
        try:
            fcidump, ecore_offset = _load_or_regenerate_fcidump(
                backend,
                mpi_rank,
                mpi_comm,
                sbd_dir,
                self._fcidump_path,
                hcore,
                eri,
                norb,
                nelec,
            )
            return [
                self._solve_gdb_one(
                    strs_a,
                    strs_b,
                    norb=norb,
                    nelec=nelec,
                    backend=backend,
                    mpi_comm=mpi_comm,
                    mpi_rank=mpi_rank,
                    sbd_dir=sbd_dir,
                    fcidump=fcidump,
                    ecore_offset=ecore_offset,
                    bit_length=SBD_DEFAULT_BIT_LENGTH,
                )
                for strs_a, strs_b in ci_strs_batches
            ]
        finally:
            if self.clean_temp_dir and owns_sbd_dir and mpi_rank == 0:
                import shutil

                shutil.rmtree(sbd_dir, ignore_errors=True)

    def _solve_gdb_one(
        self,
        strs_a: np.ndarray,
        strs_b: np.ndarray,
        *,
        norb: int,
        nelec: Tuple[int, int],
        backend: Any,
        mpi_comm: Any,
        mpi_rank: int,
        sbd_dir: Path,
        fcidump: Any,
        ecore_offset: float,
        bit_length: int,
    ) -> SCIResult:
        """Run one GDB diagonalization for one parallel-pairs batch."""
        strs_a = np.asarray(strs_a, dtype=np.int64)
        strs_b = np.asarray(strs_b, dtype=np.int64)
        if len(strs_a) != len(strs_b):
            raise ValueError(
                "mode='gdb' treats (strs_a, strs_b) as parallel length-N pair lists, "
                f"so they must have equal length, got {len(strs_a)} and {len(strs_b)}. "
                "(This is deliberately NOT an alpha-beta outer product — use mode='tpb' "
                "for product subspaces.)"
            )

        sbd_data = self._create_gdb_config(backend, bit_length)
        # sbd_config may override bit_length; pack with the value the engine will read.
        bit_length = sbd_data.bit_length
        _check_gdb_bit_length(bit_length, norb)
        det = _pack_gdb_dets(strs_a, strs_b, norb, backend, bit_length)

        wf_dump_file = sbd_dir / "wavefunction"
        results = backend.gdb_diag(mpi_comm, sbd_data, fcidump, det, "", str(wf_dump_file))
        # gdb_diag is collective and every rank returns the same `results`, but only
        # rank 0 reads the on-disk wavefunction shard below. Sync here so that read
        # cannot start before the collective's disk writes are complete on all ranks.
        mpi_comm.Barrier()

        if mpi_rank != 0:
            result = SCIResult(
                0.0,
                PairSCIState(
                    amplitudes=np.empty(0, dtype=np.float64),
                    ci_strs_a=np.array([], dtype=np.int64),
                    ci_strs_b=np.array([], dtype=np.int64),
                    norb=norb,
                    nelec=nelec,
                ),
                orbital_occupancies=(
                    np.zeros(norb, dtype=np.float64),
                    np.zeros(norb, dtype=np.float64),
                ),
            )
        else:
            # --- rank 0 only ---
            energy = results["energy"] - ecore_offset
            density = np.array(results["density"])
            occupancies = (density[::2], density[1::2])

            solved_a, solved_b, amplitudes = _read_gdb_wavefunction(
                sbd_dir / "wavefunction000000.bin",
                norb=norb,
                backend=backend,
                bit_length=bit_length,
                global_dim=results.get("global_dim"),
            )
            sci_state = PairSCIState(
                amplitudes=amplitudes,
                ci_strs_a=solved_a,
                ci_strs_b=solved_b,
                norb=norb,
                nelec=nelec,
            )
            from sbd.sbd_solver import assemble_rdms

            # gdb_diag shares tpb_diag's one_p_rdm/two_p_rdm keys and flat-index layout;
            # upstream verified assemble_rdms against PySCF on the TPB path only.
            rdm1, rdm2 = assemble_rdms(results, norb)
            result = SCIResult(
                energy, sci_state, orbital_occupancies=occupancies, rdm1=rdm1, rdm2=rdm2
            )

        # Trailing barrier makes the inter-batch ordering invariant EXPLICIT rather
        # than relying on the next batch's collective gdb_diag to re-synchronize: no
        # rank leaves this batch — to enter the next gdb_diag or the final temp-dir
        # cleanup — until rank 0 has finished reading this batch's wavefunction shard
        # off disk. Cheap (one barrier per batch) next to the diagonalization itself.
        mpi_comm.Barrier()
        return result

    def _create_gdb_config(self, backend: Any, bit_length: int) -> Any:
        """Build a ``GDB_SBD`` config, mirroring the package's TPB defaults.

        Uses the same defaults as ``sbd.sbd_solver._create_sbd_config`` (including
        its ``carryover_type = 0``: the state is read from the full-basis wavefunction
        file, so no carryover selection is needed), applies any user ``sbd_config``
        overrides (guarded by ``hasattr``, as the package does), and pins
        ``b_comm_size = 1``.
        """
        sbd_data = backend.GDB_SBD()
        sbd_data.method = 0  # Davidson
        sbd_data.max_it = 100
        sbd_data.max_nb = 50
        sbd_data.eps = 1e-8
        sbd_data.max_time = 3600
        sbd_data.init = 0
        sbd_data.do_rdm = 1 if self.compute_rdm2 else 0
        sbd_data.carryover_type = 0
        sbd_data.ratio = 0.1
        sbd_data.threshold = 1e-4
        sbd_data.bit_length = bit_length
        if self.sbd_config:
            for key, value in self.sbd_config.items():
                if hasattr(sbd_data, key):
                    setattr(sbd_data, key, value)
        if hasattr(sbd_data, "b_comm_size"):
            # Every rank passes the whole basis, which gdb_diag reads as b_comm_size == 1
            # (larger values expect each rank to pass only its own shard).
            sbd_data.b_comm_size = 1
        self._device_config.apply(sbd_data)
        return sbd_data


# ─────────────────────────────── GDB determinant packing ─────────────────── #


def _check_gdb_bit_length(bit_length: int, norb: int) -> None:
    """Refuse a GDB ``bit_length`` that sbd-eigensolver 1.7.1 cannot handle.

    Measured with sbd-eigensolver 1.7.1 on the CPU backend (``bit_length`` 3 to 24,
    ``norb`` 6 and 11, 2+2 electrons, full CI): every odd ``bit_length`` with which a
    determinant spans several words (``2 * norb > bit_length``) made ``gdb_diag``
    crash, exit with an error or hang ("unexpected situation happened in bdet"); every
    even value, and every odd value that holds the whole determinant in one word, gave
    the FCI energy. The GPU backends were not measured. As I read sbd's
    ``getHalfDets``, it splits each word into alpha (even) and beta (odd) bits and joins
    the words with a shift of ``bit_length / 2``, which would fit only an even
    ``bit_length``.

    Raises:
        ValueError: If ``bit_length`` is odd and ``2 * norb > bit_length``.
    """
    if bit_length % 2 == 1 and 2 * norb > bit_length:
        raise ValueError(
            "sbd's GDB mode needs an even bit_length when a determinant spans several "
            f"words; got bit_length={bit_length} for norb={norb} ({2 * norb} bits). "
            "Use an even value (the default is 20)."
        )


def _pack_gdb_dets(
    strs_a: np.ndarray, strs_b: np.ndarray, norb: int, backend: Any, bit_length: int
) -> list:
    """Interleave parallel ``(alpha_i, beta_i)`` strings into GDB determinants.

    Each determinant is a ``2 * norb``-bit configuration where bit ``2k`` is the
    occupation of spin-alpha orbital ``k`` and bit ``2k + 1`` that of spin-beta
    orbital ``k`` — the encoding ``sbd.gdb_diag`` documents and
    ``torch_sci_solver`` shares. A determinant is the backend's word-vector form:
    a ``list[size_t]`` where word ``w`` holds interleaved bits
    ``[w * bit_length, (w + 1) * bit_length)`` little-endian — exactly what
    ``backend.from_string`` produces. Duplicate ``(alpha, beta)`` pairs collapse
    via the package's ``sort_bitarray`` (which dedups into canonical order).

    Vectorized in numpy for ANY orbital count: the work scales with the number of
    determinants N (the 10M+ GDB regime), and the vectorization removes the
    per-determinant Python loop and string round-trip there regardless of ``norb``.
    The only remaining Python loop is ``O(2 * norb)`` iterations of whole-array
    numpy ops — negligible next to N. A per-determinant fallback covers the
    unphysical ``norb > 63`` case (a per-spin CI string wider than ``uint64``).
    """
    if norb <= 63:
        dets = _pack_gdb_dets_vectorized(strs_a, strs_b, norb, bit_length)
    else:  # pragma: no cover - norb > 63 is not a physical active space here
        dets = _pack_gdb_dets_reference(strs_a, strs_b, norb, backend, bit_length)
    return backend.sort_bitarray(dets)


def _pack_gdb_dets_vectorized(
    strs_a: np.ndarray, strs_b: np.ndarray, norb: int, bit_length: int
) -> list:
    """Vectorized :func:`_pack_gdb_dets` word-packing, for any ``norb <= 63``.

    Builds each ``bit_length``-wide determinant word directly from the per-orbital
    bit planes it covers, so the ``2 * norb``-bit configuration never has to live
    in one scalar — this works for ``2 * norb > 64`` (e.g. ``norb = 36``), unlike a
    single-``uint64`` interleave. Interleaved bit ``i`` is alpha orbital ``i // 2``
    when ``i`` is even and beta orbital ``i // 2`` when odd; it lands in word
    ``i // bit_length`` at position ``i % bit_length``. Bit-exact with the
    backend's ``from_string`` (``result[i // bit_length] |= 1 << (i % bit_length)``
    over set bits ``i``), verified against a port of the C++ implementation across
    ``norb`` up to 63 and every ``bit_length``.

    The loop is over the ``2 * norb`` interleaved bit positions, each iteration a
    single whole-array numpy op over all N determinants; the per-determinant cost
    that dominated at large N is gone.
    """
    a = np.asarray(strs_a, dtype=np.uint64)
    b = np.asarray(strs_b, dtype=np.uint64)
    n_dets = a.shape[0]
    total_bits = 2 * norb
    num_words = (total_bits + bit_length - 1) // bit_length
    one = np.uint64(1)

    # words[w] accumulates the w-th bit_length-wide word for every determinant.
    words = np.zeros((num_words, n_dets), dtype=np.uint64)
    for i in range(total_bits):
        k = i // 2
        plane = ((a if i % 2 == 0 else b) >> np.uint64(k)) & one  # 0/1 per det
        words[i // bit_length] |= plane << np.uint64(i % bit_length)
    # Transpose to determinant-major: list[list[int]] of word vectors.
    return words.T.tolist()


def _pack_gdb_dets_reference(
    strs_a: np.ndarray, strs_b: np.ndarray, norb: int, backend: Any, bit_length: int
) -> list:
    """Per-determinant :func:`_pack_gdb_dets` fallback for ``norb > 63``.

    Builds each interleaved bitstring and hands it to ``backend.from_string``. Slow
    at large sizes (Python loop), but ``norb > 63`` is not a physical active space
    in this context — the vectorized path covers every real GDB run.
    """
    dets = []
    for a, b in zip(strs_a, strs_b):
        a_int = int(a)
        b_int = int(b)
        interleaved = 0
        for k in range(norb):
            if (a_int >> k) & 1:
                interleaved |= 1 << (2 * k)
            if (b_int >> k) & 1:
                interleaved |= 1 << (2 * k + 1)
        bitstr = format(interleaved, f"0{2 * norb}b")
        dets.append(backend.from_string(bitstr, bit_length, 2 * norb))
    return dets


def _unpack_gdb_dets(
    dets: list | np.ndarray, norb: int, backend: Any, bit_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split GDB determinants back into per-spin CI-string arrays.

    Inverse of :func:`_pack_gdb_dets`: bit ``2k`` -> alpha orbital ``k``, bit
    ``2k + 1`` -> beta orbital ``k``. Vectorized in numpy for any ``norb <= 63``
    (bit-exact with the ``makestring`` round-trip below), falling back to the
    per-determinant path only for the unphysical ``norb > 63`` case.
    """
    if len(dets) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if norb <= 63:
        return _unpack_gdb_dets_vectorized(dets, norb, bit_length)

    # pragma: no cover - norb > 63 is not a physical active space here
    strs_a = []
    strs_b = []
    for det in dets:
        bitstr = backend.makestring(det, bit_length, 2 * norb)
        # makestring emits MSB-first over 2*norb bits; bit index k from the LSB.
        value = int(bitstr, 2)
        a_int = 0
        b_int = 0
        for k in range(norb):
            if (value >> (2 * k)) & 1:
                a_int |= 1 << k
            if (value >> (2 * k + 1)) & 1:
                b_int |= 1 << k
        strs_a.append(a_int)
        strs_b.append(b_int)
    return np.array(strs_a, dtype=np.int64), np.array(strs_b, dtype=np.int64)


def _unpack_gdb_dets_vectorized(
    dets: list | np.ndarray, norb: int, bit_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`_unpack_gdb_dets`, for any ``norb <= 63``.

    Reads each interleaved bit ``i`` directly from its word (``word i //
    bit_length``, position ``i % bit_length``) and routes it to alpha orbital
    ``i // 2`` (even ``i``) or beta orbital ``i // 2`` (odd ``i``), so the
    ``2 * norb``-bit value never has to be reassembled into one scalar — this
    works for ``2 * norb > 64`` (e.g. ``norb = 36``). The loop is over the
    ``2 * norb`` interleaved bit positions, each iteration a whole-array numpy op
    over all N determinants. Verified bit-exact against the per-determinant path.
    """
    words = np.asarray(dets, dtype=np.uint64)  # (n_dets, num_words)
    n_dets = words.shape[0]
    total_bits = 2 * norb
    one = np.uint64(1)

    strs_a = np.zeros(n_dets, dtype=np.uint64)
    strs_b = np.zeros(n_dets, dtype=np.uint64)
    for i in range(total_bits):
        bit = (words[:, i // bit_length] >> np.uint64(i % bit_length)) & one
        k = i // 2
        if i % 2 == 0:
            strs_a |= bit << np.uint64(k)
        else:
            strs_b |= bit << np.uint64(k)
    return strs_a.astype(np.int64), strs_b.astype(np.int64)


def _read_gdb_wavefunction(
    path: Path, *, norb: int, backend: Any, bit_length: int, global_dim: int | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the diagonalized GDB basis and its amplitudes from ``gdb_diag``'s save file.

    ``gdb_diag`` writes one ``.bin`` shard per ``b_comm`` rank via SBD's
    ``SaveWavefunction`` (``caop/basic/restart.h``); with ``b_comm_size == 1`` the only
    shard is ``...000000.bin`` and it holds the WHOLE basis. Its native little-endian
    layout is a ``size_t`` header ``(basis_size, basis_length)``, then ``basis_size``
    determinants of ``basis_length`` ``size_t`` words, then ``basis_size`` ``float64``
    amplitudes (the real chemistry-GDB ``ElemT``).

    The labels come from the file's own determinants rather than the input lists:
    ``gdb_diag`` works on the sorted, deduplicated basis, so the amplitudes are in
    that order. Every inconsistency raises — returning a placeholder state here
    would be indistinguishable from a real one downstream.

    Returns:
        ``(strs_a, strs_b, amplitudes)``: parallel length-N alpha/beta CI strings and
        the flat amplitude vector over those pairs, in ``sbd``'s interleaved-Wick
        parity.
    """
    if not path.exists():
        raise RuntimeError(
            f"gdb_diag wrote no wavefunction file at {path}, so the diagonalization did "
            "not complete as expected."
        )
    with path.open("rb") as fh:
        header = np.fromfile(fh, dtype=np.uint64, count=2)
        if header.size != 2:
            raise RuntimeError(f"GDB wavefunction file {path} has a truncated header.")
        basis_size, basis_length = int(header[0]), int(header[1])
        expected_length = (2 * norb + bit_length - 1) // bit_length
        if basis_length != expected_length:
            raise RuntimeError(
                f"GDB wavefunction file {path} stores {basis_length} words per "
                f"determinant, expected {expected_length} for norb={norb}, "
                f"bit_length={bit_length}."
            )
        words = np.fromfile(fh, dtype=np.uint64, count=basis_size * basis_length)
        amplitudes = np.fromfile(fh, dtype=np.float64, count=basis_size)
    if words.size != basis_size * basis_length or amplitudes.size != basis_size:
        raise RuntimeError(
            f"GDB wavefunction file {path} is truncated: header promises {basis_size} "
            "determinants and amplitudes."
        )
    if global_dim is not None and basis_size != int(global_dim):
        raise RuntimeError(
            f"GDB wavefunction file {path} holds {basis_size} determinants, but "
            f"gdb_diag reports global_dim={global_dim}."
        )
    strs_a, strs_b = _unpack_gdb_dets(
        words.reshape(basis_size, basis_length), norb, backend, bit_length
    )
    return strs_a, strs_b, amplitudes
