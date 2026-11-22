"""Opt-in distributed execution for :class:`TorchSCISolver`.

This is the execution layer behind ``make_torch_sci_solver(distributed=...)``.
It decides whether a call runs distributed, launches one process per GPU (or
one CPU process per rank for gloo), initializes the process group, and passes
the caller's data to the workers. The numerical work lives in
``torch_sci_solver.torch_impl.distributed_davidson``.

``torch.distributed`` is imported lazily: importing this module (or the solver)
must not require a distributed build, a process group, or a GPU. With no
``distributed`` keyword — or with ``world_size <= 1`` — every call returns
:data:`NOT_DISTRIBUTED` and the solver runs its existing single-process path
byte for byte.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import dataclasses
import hashlib
import json
import multiprocessing
import os
import shutil
import socket
import tempfile
import warnings
import weakref
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

from torch_sci_solver.torch_impl.distributed_davidson import _dist

# Sentinel: "this call did not take the distributed path; run the existing one".
NOT_DISTRIBUTED = object()

# Above this many bytes a distributed solve hands its amplitudes back through
# memmapped .npy files under TMPDIR instead of one in-memory gather to rank 0.
# One gibibyte holds a 10^8-determinant fp64 vector in ~1.25
# vectors; the size-dependent split keeps small SQD solves on the in-memory
# path and keeps a 10^9 solve off every GPU.
DISTRIBUTED_HAND_BACK_SMALL_BYTES = 1 << 30


@dataclasses.dataclass
class MemmapAmplitudes:
    """Placeholder for an amplitude array left in a memmapped ``.npy`` file.

    A distributed rank writes its rows to the file and puts this marker in the
    returned state (or ``TorchSCIState.eigvec_inter``). ``torch.save``
    pickles the small marker, never the array; the parent process swaps it for
    an ``np.load(..., mmap_mode="r")`` view in :func:`materialize_result` and
    unlinks the file (POSIX keeps the mapping valid). This is what lets a
    multi-gigabyte result cross the worker -> parent boundary without a second
    full copy on rank 0 or on any non-zero rank.
    """

    path: str
    shape: tuple[int, ...]


@dataclasses.dataclass
class DeferredSCIState:
    """Worker-side stand-in for a qiskit ``SCIState`` with memmapped amplitudes.

    ``qiskit_addon_sqd.fermion.SCIState.__post_init__`` validates
    ``amplitudes.shape`` against the string lists, so it cannot carry a
    :class:`MemmapAmplitudes` marker. Rank 0 returns this small placeholder for
    a large TPB hand-back instead; :func:`materialize_result` maps the file and
    builds the real ``SCIState`` (reshaped to the Cartesian ``(M_a, M_b)``
    layout) in the parent process.
    """

    amplitudes: MemmapAmplitudes
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    norb: int
    nelec: tuple[int, int]
    shape: tuple[int, ...]


def _materialize_array(marker: MemmapAmplitudes) -> np.ndarray:
    """Open a memmapped file and unlink it once mapped (caller keeps the view)."""
    array = np.load(marker.path, mmap_mode="r")
    path = Path(marker.path)
    try:
        path.unlink()
    except OSError:
        pass
    try:
        path.parent.rmdir()
    except OSError:
        pass
    return array


def _materialize_state(state: Any) -> Any:
    """Replace marker amplitudes in one ``SCIState``-like object."""
    if isinstance(state, DeferredSCIState):
        from qiskit_addon_sqd.fermion import SCIState

        amplitudes = _materialize_array(state.amplitudes).reshape(state.shape)
        return SCIState(
            amplitudes=amplitudes,
            ci_strs_a=state.ci_strs_a,
            ci_strs_b=state.ci_strs_b,
            norb=state.norb,
            nelec=state.nelec,
        )
    amplitudes = getattr(state, "amplitudes", None)
    if isinstance(amplitudes, MemmapAmplitudes):
        return dataclasses.replace(state, amplitudes=_materialize_array(amplitudes))
    return state


def materialize_result(result: Any, _seen: Optional[set[int]] = None) -> Any:
    """Swap every :class:`MemmapAmplitudes` marker for a memmapped numpy view.

    Handles the shapes a solver call can return: ``list[EigenStates]``, the
    probe tuple (``list[EigenStates]`` plus the cached ``TorchSCIState`` /
    ``TorchPairSCIState`` objects), ``EigenStates``, ``SCIResult``, the cached
    state objects themselves (whose ``eigvec_inter`` may also be a marker) and
    a :class:`DeferredSCIState` (which becomes a real ``SCIState``). Files are
    unlinked as soon as their mapping exists; nothing is copied. Objects
    without markers pass through unchanged.
    """
    if _seen is None:
        _seen = set()
    if id(result) in _seen:
        return result
    _seen.add(id(result))

    if isinstance(result, DeferredSCIState):
        return _materialize_state(result)
    if isinstance(result, (tuple, list)):
        converted = [materialize_result(item, _seen) for item in result]
        return tuple(converted) if isinstance(result, tuple) else converted
    # TorchSCIState and TorchPairSCIState both cache the ground-state vector as
    # ``eigvec_inter``; duck-typing avoids importing the pair module here.
    eigvec_inter = getattr(result, "eigvec_inter", None)
    if isinstance(eigvec_inter, MemmapAmplitudes):
        array = _materialize_array(eigvec_inter)
        return dataclasses.replace(result, eigvec_inter=torch.from_numpy(array))
    # EigenStates: .states is a mutable list of SCIResultWithConvergence.
    states = getattr(result, "states", None)
    if isinstance(states, list):
        for index, sci_result in enumerate(states):
            sci_state = getattr(sci_result, "sci_state", None)
            new_state = _materialize_state(sci_state)
            if new_state is not sci_state:
                states[index] = dataclasses.replace(sci_result, sci_state=new_state)
        return result
    # A bare SCIResult (defensive; the solver returns EigenStates).
    sci_state = getattr(result, "sci_state", None)
    new_state = _materialize_state(sci_state)
    if new_state is not sci_state:
        return dataclasses.replace(result, sci_state=new_state)
    return result


def total_hand_back_bytes(
    n_roots: int, n_determinants: int, *, warm_state: bool, itemsize: int = 8
) -> int:
    """Bytes of the large-result hand-back (amplitudes, plus the warm eigvec)."""
    return (int(n_roots) + (1 if warm_state else 0)) * int(n_determinants) * int(itemsize)


def check_hand_back_disk(directory: str, n_bytes: int) -> None:
    """Raise a clear error if ``directory`` cannot hold the memmapped result.

    Names ``TMPDIR`` because that is where a single-node caller points at a
    large local disk; a multi-node caller sets
    ``distributed={"hand_back_dir": ...}`` to shared storage instead.
    """
    free = shutil.disk_usage(directory).free
    if free < n_bytes:
        raise RuntimeError(
            f"distributed result hand-back needs {n_bytes / 1024**3:.2f} GiB of "
            f"temporary disk under TMPDIR ({directory}) but only "
            f"{free / 1024**3:.2f} GiB are free. Set TMPDIR (or the distributed "
            "hand_back_dir) to a larger disk, or reduce the basis/n_roots."
        )


def _create_result_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> Any:
    """Create a ``.npy`` memmap that object-store mounts can also open.

    ``np.lib.format.open_memmap(..., mode="w+")`` writes the header, then
    truncates to the payload inside :class:`numpy.memmap` and maps it at once;
    on an s3fs mount a large new length is not visible to the very next
    ``fstat``, so the map raises ``ValueError: mmap length is greater than
    file size``. The fallback writes the header itself, extends the file and
    maps it in ``r+`` mode, re-checking the size before mapping.

    Args:
        path: File to create.
        dtype: Payload dtype (real or complex).
        shape: Full array shape.

    Returns:
        The writable memmap, sized ``shape``.
    """
    try:
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    except ValueError:
        array_dtype = np.dtype(dtype)
        with path.open("wb+") as handle:
            np.lib.format.write_array_header_1_0(
                handle,
                {
                    "descr": np.lib.format.dtype_to_descr(array_dtype),
                    "fortran_order": False,
                    "shape": tuple(int(n) for n in shape),
                },
            )
            total = handle.tell() + int(np.prod(shape)) * array_dtype.itemsize
            handle.truncate(total)
            handle.flush()
            if path.stat().st_size < total:
                # The mount only materialises the length once a byte lands there.
                handle.seek(total - 1)
                handle.write(b"\0")
                handle.flush()
        return np.lib.format.open_memmap(path, mode="r+")


def _hand_back_directory(context: Any) -> Path:
    """Create this hand-back's directory on rank 0.

    Uses ``context.hand_back_dir`` when set (multi-node ranks cannot see each
    other's temporary directory, so those runs must point it at shared
    storage) and the node-local temporary directory otherwise.
    """
    root = getattr(context, "hand_back_dir", None)
    if root is not None:
        Path(root).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="torch_sci_dist_amp_", dir=root))


def _write_memmaps(
    blocks: dict[str, torch.Tensor],
    shapes: dict[str, tuple[int, ...]],
    *,
    context: Any,
    write_block: Optional[Callable[[Any, torch.Tensor], None]],
) -> dict[str, MemmapAmplitudes]:
    """Create the shared files, then let every rank write its own block.

    Shared by the row, tile and scattered pair-tile hand-backs. Rank 0 creates
    one ``.npy`` file per block under :func:`_hand_back_directory` and checks
    the disk budget, broadcasts the result, then every rank applies
    ``write_block`` to its own writable memmap. The collective sequence matches
    the three former copies: one broadcast, a barrier before the write pass and
    one after it.

    Args:
        blocks: Name -> this rank's block.
        shapes: Name -> the full on-disk shape of that block's file.
        context: :class:`DistributedContext` of the initialized group.
        write_block: Called as ``write_block(handle, block)`` for each block
            this rank writes; ``None`` when this rank owns nothing and must skip
            the write pass.

    Returns:
        Name -> :class:`MemmapAmplitudes` marker with the flat shape.

    Raises:
        RuntimeError: rank 0 cannot fit the files under TMPDIR.
    """
    dist = _dist()
    rank = int(context.rank)
    status: list[Any] = [None]
    if rank == 0:
        directory = _hand_back_directory(context)
        total = sum(
            int(np.prod(shapes[name])) * (16 if blocks[name].is_complex() else 8) for name in blocks
        )
        try:
            check_hand_back_disk(str(directory), total + (16 << 20))
        except RuntimeError as exc:
            status[0] = (False, str(exc))
        else:
            for name in blocks:
                handle = _create_result_memmap(
                    directory / f"{name}.npy",
                    np.complex128 if blocks[name].is_complex() else np.float64,
                    shapes[name],
                )
                handle.flush()
                del handle
            status[0] = (True, str(directory))
    dist.broadcast_object_list(status, src=0, group=context.group)
    ok, message = status[0]
    if not ok:
        raise RuntimeError(message)
    directory = Path(message)

    # The files exist on every rank's view of the shared filesystem now.
    dist.barrier(group=context.group)
    if write_block is not None:
        for name, block in blocks.items():
            handle = np.load(directory / f"{name}.npy", mmap_mode="r+")
            write_block(handle, block)
            handle.flush()
            del handle
    dist.barrier(group=context.group)
    return {
        name: MemmapAmplitudes(
            path=str(directory / f"{name}.npy"), shape=(int(np.prod(shapes[name])),)
        )
        for name in blocks
    }


def write_row_memmaps(
    blocks: dict[str, torch.Tensor],
    shapes: dict[str, tuple[int, ...]],
    *,
    row_start: int,
    row_stop: int,
    context: Any,
) -> dict[str, MemmapAmplitudes]:
    """Write rank-local row slices into shared ``.npy`` files under TMPDIR.

    Every rank calls this. Rank 0 creates the files and a directory under
    ``context.hand_back_dir`` when set and ``tempfile.gettempdir()`` (which
    honors ``TMPDIR``) otherwise, broadcasts the path, then each rank opens its
    files in ``r+`` mode and writes ``[row_start, row_stop)`` of the flat array.
    The returned markers are small and picklable; :func:`materialize_result`
    turns them into memmapped views in the parent process.

    Args:
        blocks: Name -> this rank's 1-D slice ``(row_stop - row_start,)``.
        shapes: Name -> the full flat shape, e.g. ``(M_a * M_b,)``.
        row_start, row_stop: This rank's flat row range in every file.
        context: :class:`DistributedContext` of the initialized group.

    Returns:
        Name -> :class:`MemmapAmplitudes` marker, on every rank.

    Raises:
        RuntimeError: rank 0 cannot fit the files under TMPDIR.
    """

    def write_block(handle: Any, block: torch.Tensor) -> None:
        handle[row_start:row_stop] = block.detach().cpu().numpy().reshape(-1)

    return _write_memmaps(
        blocks,
        shapes,
        context=context,
        write_block=write_block if row_stop > row_start else None,
    )


def write_tile_memmaps(
    blocks: dict[str, torch.Tensor],
    shapes: dict[str, tuple[int, int]],
    *,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    context: Any,
) -> dict[str, MemmapAmplitudes]:
    """Write rank-local tile slices into shared 2-D ``.npy`` files under TMPDIR.

    The 2-D counterpart of :func:`write_row_memmaps`: rank 0 creates one
    ``(M_a, M_b)`` file per block, every rank writes its
    ``[row_start, row_stop) x [col_start, col_stop)`` rectangle, and the
    returned markers carry the flat ``M_a * M_b`` shape the parent reshapes.

    Args:
        blocks: Name -> this rank's tile block, ``(local_rows, local_cols)``
            flattened or 2-D.
        shapes: Name -> the full 2-D shape ``(M_a, M_b)``.
        row_start, row_stop: This rank's row range in every file.
        col_start, col_stop: This rank's column range in every file.
        context: :class:`DistributedContext` of the initialized group.

    Returns:
        Name -> :class:`MemmapAmplitudes` marker, on every rank.

    Raises:
        RuntimeError: rank 0 cannot fit the files under TMPDIR.
    """
    height = row_stop - row_start
    width = col_stop - col_start

    def write_block(handle: Any, block: torch.Tensor) -> None:
        handle[row_start:row_stop, col_start:col_stop] = (
            block.detach().cpu().numpy().reshape(height, width)
        )

    return _write_memmaps(
        blocks,
        shapes,
        context=context,
        write_block=write_block if height > 0 and width > 0 else None,
    )


def write_pair_tile_memmaps(
    blocks: dict[str, torch.Tensor],
    shapes: dict[str, tuple[int, ...]],
    *,
    pair_index: torch.Tensor,
    context: Any,
) -> dict[str, MemmapAmplitudes]:
    """Write scattered GDB pair-tile values into shared 1-D ``.npy`` files.

    A GDB tile owns a sorted, scattered subset ``pair_index`` of the flat pair
    vector (one rectangle per basis block), so unlike
    :func:`write_row_memmaps` the written positions are not contiguous. Rank 0
    creates one flat file per block and broadcasts the directory; every rank
    then writes its own disjoint positions, so no two ranks touch the same
    entry. The returned markers carry the flat ``(n_pairs,)`` shape and are
    materialized by :func:`materialize_result` in the parent process.

    Args:
        blocks: Name -> this rank's 1-D block, ``(pair_index.numel(),)``.
        shapes: Name -> the full flat shape, e.g. ``(n_pairs,)``.
        pair_index: This rank's sorted global pair indices.
        context: :class:`DistributedContext` of the initialized group.

    Returns:
        Name -> :class:`MemmapAmplitudes` marker, on every rank.

    Raises:
        RuntimeError: rank 0 cannot fit the files under TMPDIR, or a block
            length does not match ``pair_index``.
    """
    index = pair_index.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    for name, block in blocks.items():
        if int(block.numel()) != int(index.numel()):
            raise ValueError(
                f"block {name!r} has {int(block.numel())} entries but pair_index has "
                f"{int(index.numel())}"
            )
    positions = index.numpy()

    def write_block(handle: Any, block: torch.Tensor) -> None:
        handle[positions] = block.detach().cpu().numpy().reshape(-1)

    return _write_memmaps(
        blocks,
        shapes,
        context=context,
        write_block=write_block if index.numel() else None,
    )


_ALLOWED_KEYS = {
    "world_size",
    "backend",
    "device_ids",
    "master_addr",
    "master_port",
    "init_method",
    "timeout",
    "matvec",
    "persistent",
    "tile",
    "exchange_bytes",
    "hand_back",
    "hand_back_bytes",
    "hand_back_dir",
    "budget_bytes",
    "worker_startup_timeout_s",
    "checkpoint_dir",
    "checkpoint_every",
    "checkpoint_resume",
}

#: Values accepted by ``distributed={"hand_back": ...}``.
HAND_BACK_MODES = ("auto", "memory", "memmap")

#: Values accepted by ``distributed={"checkpoint_resume": ...}``.
CHECKPOINT_RESUME_MODES = ("auto", "off", "require")

#: Bumped when the on-disk shard/manifest layout changes incompatibly.
CHECKPOINT_FORMAT_VERSION = 1

#: Default Davidson iterations between checkpoint writes.
DEFAULT_CHECKPOINT_EVERY = 50


def _env_world_size() -> int:
    """Read ``WORLD_SIZE`` from the launcher environment (torchrun/srun)."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1") or 1)
    except ValueError:
        return 1


def _free_port() -> int:
    """Pick a free loopback TCP port for a single-node rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ============================================================================
# Checkpoint / restart
# ============================================================================


def checkpoint_subspace_dir(root: "str | os.PathLike[str]", subspace: int) -> Path:
    """Directory holding one CI subspace's checkpoint files.

    Args:
        root: The configured ``checkpoint_dir``.
        subspace: CI subspace index within the solver call.

    Returns:
        ``<root>/subspace_<k>``.
    """
    return Path(root) / f"subspace_{int(subspace)}"


def checkpoint_manifest_path(root: "str | os.PathLike[str]", subspace: int) -> Path:
    """Path of the rank-0 checkpoint manifest for one CI subspace."""
    return checkpoint_subspace_dir(root, subspace) / "manifest.json"


def checkpoint_shard_path(root: "str | os.PathLike[str]", subspace: int, rank: int) -> Path:
    """Path of ``rank``'s durable Davidson state for one CI subspace."""
    return checkpoint_subspace_dir(root, subspace) / f"rank_{int(rank)}.pt"


def checkpoint_basis_hash(*arrays: Any) -> str:
    """Stable SHA-256 over a CI basis' canonical arrays.

    Every array is hashed with its shape and dtype, so two bases collide only
    if their entries agree elementwise. The basis is replicated on every rank,
    so every rank computes the same value without communication.

    Args:
        *arrays: Integer basis arrays (numpy arrays, lists, or int64 tensors).

    Returns:
        Hex digest of the concatenated shape/dtype/payload bytes.
    """
    digest = hashlib.sha256()
    for array in arrays:
        tensor = torch.as_tensor(array).detach().to(device="cpu")
        digest.update(str(tuple(int(n) for n in tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(tensor.numpy()).tobytes())
    return digest.hexdigest()


def checkpoint_partition_identity(partition: Any, world_size: int) -> dict[str, Any]:
    """JSON-friendly identity of the rank partition a solve runs on.

    The identity names the partition kind and every rank's owned range, so a
    resumed run cannot silently pair a saved local block with a different
    layout (for example a 1-D row split read as a tile).

    Args:
        partition: A :class:`RowPartition`, :class:`TilePartition`,
            :class:`VectorPartition`, any object exposing ``bounds``, or
            ``None`` for the canonical contiguous split.
        world_size: Number of ranks.

    Returns:
        A canonical dict suitable for JSON and hashing.
    """
    if partition is None:
        return {"kind": "canonical", "world_size": int(world_size)}
    from torch_sci_solver.torch_impl.distributed_davidson import (
        RowPartition,
        TilePartition,
        VectorPartition,
    )

    if isinstance(partition, TilePartition):
        return {
            "kind": "tile",
            "world_size": int(world_size),
            "n_rows": int(partition.n_rows),
            "n_cols": int(partition.n_cols),
            "grid": [int(partition.grid.p_alpha), int(partition.grid.p_beta)],
            "bounds": [
                [int(b.row_start), int(b.row_stop), int(b.col_start), int(b.col_stop)]
                for b in partition.bounds
            ],
        }
    if isinstance(partition, RowPartition):
        return {
            "kind": "rows",
            "world_size": int(world_size),
            "n_rows": int(partition.n_rows),
            "row_size": int(partition.row_size),
            "bounds": [[int(b.start), int(b.stop)] for b in partition.bounds],
        }
    if isinstance(partition, VectorPartition):
        return {
            "kind": "vector",
            "world_size": int(world_size),
            "n": int(partition.n),
            "bounds": [[int(b.start), int(b.stop)] for b in partition.bounds],
        }
    bounds = getattr(partition, "bounds", None)
    identity: dict[str, Any] = {
        "kind": type(partition).__name__,
        "world_size": int(world_size),
    }
    if bounds is not None:
        identity["bounds"] = [[int(b.start), int(b.stop)] for b in bounds]
    return identity


def _checkpoint_jsonable(value: Any) -> Any:
    """Recursively convert a metadata value into canonical JSON types."""
    if isinstance(value, dict):
        return {str(key): _checkpoint_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_jsonable(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _checkpoint_identity_hash(identity: dict[str, Any]) -> str:
    """Hash the canonical (sorted-key) form of one checkpoint identity."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_replace(path: Path, writer: Callable[[Path], None]) -> None:
    """Write ``path`` through a same-directory temp file and a rename.

    Args:
        path: destination file.
        writer: called with the temporary path; it must write the payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        writer(temp)
        Path(temp).replace(path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _atomic_torch_save(payload: Any, path: Path) -> None:
    """Write ``payload`` to ``path`` through a same-directory temp + rename."""
    _atomic_replace(path, lambda temp: torch.save(payload, temp, pickle_protocol=4))


def _atomic_write_json(payload: Any, path: Path) -> None:
    """Write ``payload`` as JSON through a same-directory temp + rename."""
    _atomic_replace(
        path,
        lambda temp: temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"),
    )


def _read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    """Read and type-check one checkpoint manifest."""
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest is not a JSON object")
    return manifest


def _checkpoint_mismatches(manifest: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    """Return the identity field names that disagree with ``manifest``."""
    mismatches: list[str] = []
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        mismatches.append("format_version")
    saved = manifest.get("identity")
    if not isinstance(saved, dict):
        mismatches.append("identity")
        return mismatches
    for key in sorted(set(identity) | set(saved)):
        if key not in saved or key not in identity or saved[key] != identity[key]:
            mismatches.append(key)
    return mismatches


def _checkpoint_short_repr(value: Any, limit: int = 120) -> str:
    """A truncated ``repr`` for one mismatch field (partition dicts are wide)."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _checkpoint_mismatch_detail(
    manifest: dict[str, Any], identity: dict[str, Any], fields: list[str]
) -> str:
    """One-line saved-vs-current summary for a mismatch error message."""
    saved_identity = manifest.get("identity")
    saved: dict[str, Any] = saved_identity if isinstance(saved_identity, dict) else {}
    parts = []
    for field in fields[:6]:
        parts.append(
            f"{field}: saved={_checkpoint_short_repr(saved.get(field, '<missing>'))} "
            f"current={_checkpoint_short_repr(identity.get(field, '<missing>'))}"
        )
    if len(fields) > 6:
        parts.append(f"... and {len(fields) - 6} more")
    return "; ".join(parts)


def make_checkpoint_hook(
    settings: dict,
    *,
    subspace: int,
    metadata: dict,
    context: Any,
    stats: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Any:
    """Build the Davidson checkpoint hook for one CI subspace.

    Every rank writes its own ``rank_<r>.pt`` shard; rank 0 additionally
    writes the ``manifest.json`` that names the identity all shards belong to.
    Both files are written through a temp file and a rename, so a reader never
    sees a partially written file.

    Resume semantics:

    - ``"auto"`` (default): resume a matching checkpoint; start cold when none
      exists; raise when a manifest exists but names a different run (basis,
      Hamiltonian, world size, layout, dtype, block size or solver knobs).
    - ``"off"``: ignore any existing checkpoint and start cold; the next write
      overwrites it.
    - ``"require"``: like ``"auto"``, but a missing or incomplete checkpoint is
      an error instead of a cold start.

    Args:
        settings: ``{"dir": ..., "every": ..., "resume": ...}`` from the
            resolved distributed config.
        subspace: CI subspace index (the ``subspace_<k>`` directory).
        metadata: JSON-friendly identity of this run. Must be identical on
            every rank.
        context: :class:`DistributedContext` of the initialized group.
        stats: Optional solver stats dict; rank 0 records the resume decision
            and the number of manifest writes.
        extra: Optional informational fields (for example the full solver-knob
            snapshot) stored in the manifest but excluded from matching.

    Returns:
        A :class:`DavidsonCheckpoint` with ``load``/``save`` closures.

    Raises:
        ValueError: invalid ``every`` or ``resume``.
    """
    from torch_sci_solver.torch_impl.distributed_davidson import DavidsonCheckpoint

    root = str(settings["dir"])
    every = int(settings.get("every", DEFAULT_CHECKPOINT_EVERY))
    resume_mode = str(settings.get("resume", "auto")).lower()
    if every <= 0:
        raise ValueError(f"checkpoint_every must be positive, got {every}")
    if resume_mode not in CHECKPOINT_RESUME_MODES:
        raise ValueError(
            f"checkpoint_resume must be one of {CHECKPOINT_RESUME_MODES}, got {resume_mode!r}"
        )

    identity = _checkpoint_jsonable(metadata)
    identity_hash = _checkpoint_identity_hash(identity)
    rank = int(context.rank)
    world_size = int(context.world_size)
    manifest_path = checkpoint_manifest_path(root, subspace)
    shard_path = checkpoint_shard_path(root, subspace, rank)

    def _broadcast(payload: list) -> None:
        dist = _dist()
        if world_size > 1:
            dist.broadcast_object_list(payload, src=0, group=context.group)
        elif payload[0] is None:
            raise RuntimeError("checkpoint manifest could not be read on rank 0")

    def _record(entry: dict) -> None:
        if stats is None or rank != 0:
            return
        stats.setdefault("checkpoint_log", []).append(entry)

    def _raise_mismatch(manifest: dict[str, Any], fields: list[str]) -> None:
        detail = _checkpoint_mismatch_detail(manifest, identity, fields)
        message = (
            f"distributed checkpoint at {manifest_path} belongs to a different run "
            f"({', '.join(fields)}): {detail}. Restart with "
            "distributed={'checkpoint_resume': 'off'} to start cold and overwrite it, "
            f"or clear {checkpoint_subspace_dir(root, subspace)}."
        )
        raise RuntimeError(message)

    def load() -> Optional[dict]:
        if resume_mode == "off":
            _record({
                "subspace": int(subspace),
                "action": "disabled",
                "iteration": None,
                "checkpoint_dir": root,
                "resume": resume_mode,
                "world_size": world_size,
            })
            return None

        payload: list = [None]
        if rank == 0:
            try:
                manifest: Optional[dict] = _read_checkpoint_manifest(manifest_path)
                payload[0] = {"status": "ok", "manifest": manifest}
            except FileNotFoundError:
                payload[0] = {"status": "missing"}
            except Exception as exc:  # noqa: BLE001 - report to every rank
                payload[0] = {"status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
        _broadcast(payload)
        status = payload[0]["status"]

        if status == "missing":
            if resume_mode == "require":
                raise RuntimeError(
                    f"checkpoint_resume='require' but no checkpoint manifest exists at "
                    f"{manifest_path}. Run once with checkpoint_resume='auto' (or 'off') "
                    "to create it, or drop checkpoint_resume='require'."
                )
            _record({
                "subspace": int(subspace),
                "action": "cold",
                "iteration": None,
                "checkpoint_dir": root,
                "resume": resume_mode,
                "world_size": world_size,
            })
            return None
        if status == "invalid":
            raise RuntimeError(
                f"distributed checkpoint manifest {manifest_path} is unreadable "
                f"({payload[0]['error']}). Restart with "
                "distributed={'checkpoint_resume': 'off'} to start cold and overwrite "
                f"it, or clear {checkpoint_subspace_dir(root, subspace)}."
            )

        manifest = payload[0]["manifest"]
        mismatches = _checkpoint_mismatches(manifest, identity)
        if mismatches:
            _raise_mismatch(manifest, mismatches)

        local_ok = True
        reason = ""
        shard: Optional[dict] = None
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            expected_iteration = int(manifest.get("iteration", -1))
            checks = {
                "format_version": shard.get("format_version") == CHECKPOINT_FORMAT_VERSION,
                "identity_hash": shard.get("identity_hash") == identity_hash,
                "subspace": int(shard.get("subspace", -1)) == int(subspace),
                "rank": int(shard.get("rank", -1)) == rank,
                "world_size": int(shard.get("world_size", -1)) == world_size,
                "iteration": int(shard.get("iteration", -2)) == expected_iteration,
                "state_layout": shard.get("state_layout")
                == (shard.get("state") or {}).get("layout"),
                "state": isinstance(shard.get("state"), dict),
            }
            failed = sorted(name for name, ok in checks.items() if not ok)
            if failed:
                local_ok = False
                reason = f"{shard_path} does not match the manifest ({', '.join(failed)})"
        except Exception as exc:  # noqa: BLE001 - every rank must reach the reduce
            local_ok = False
            reason = f"{shard_path} could not be loaded ({type(exc).__name__}: {exc})"

        # The MIN-reduce runs on the rank's own device: NCCL has no CPU
        # backend, so a host flag would raise "No backend type associated
        # with device type cpu" on a GPU run.
        flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64, device=context.device)
        dist = _dist()
        if world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=context.group)
        if int(flag.item()) != 1:
            reasons: list = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(reasons, reason, group=context.group)
            else:
                reasons = [reason]
            detail = "; ".join(str(item) for item in reasons if item)
            message = f"distributed checkpoint is incomplete: {detail or 'unknown rank shard'}"
            if resume_mode == "require":
                raise RuntimeError(
                    f"{message}. checkpoint_resume='require' cannot continue; run with "
                    "checkpoint_resume='auto' or 'off' to start cold."
                )
            if rank == 0:
                warnings.warn(f"{message}; starting cold and overwriting it.", RuntimeWarning)
            _record({
                "subspace": int(subspace),
                "action": "cold_incomplete",
                "iteration": int(manifest.get("iteration", -1)),
                "checkpoint_dir": root,
                "resume": resume_mode,
                "world_size": world_size,
                "reason": detail,
            })
            return None

        iteration = int(manifest.get("iteration", -1))
        _record({
            "subspace": int(subspace),
            "action": "resumed",
            "iteration": iteration,
            "checkpoint_dir": root,
            "resume": resume_mode,
            "world_size": world_size,
        })
        assert shard is not None
        return {"iteration": iteration, "state": shard["state"]}

    def save(iteration: int, state: dict) -> None:
        state_layout = state.get("layout")
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "identity_hash": identity_hash,
            "subspace": int(subspace),
            "rank": rank,
            "world_size": world_size,
            "iteration": int(iteration),
            "state_layout": state_layout,
            "state": state,
        }
        _atomic_torch_save(payload, shard_path)
        if rank != 0:
            return
        manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "identity": identity,
            "identity_hash": identity_hash,
            "subspace": int(subspace),
            "iteration": int(iteration),
            "state_layout": state_layout,
            "world_size": world_size,
            "extra": _checkpoint_jsonable(extra or {}),
        }
        _atomic_write_json(manifest, manifest_path)
        if stats is not None:
            stats["checkpoint_writes"] = int(stats.get("checkpoint_writes", 0)) + 1

    return DavidsonCheckpoint(every=every, load=load, save=save)


def inspect_checkpoint(
    root: "str | os.PathLike[str]",
    subspace: int,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """Read one checkpoint manifest without a process group or a solve.

    Used by :meth:`TorchSCISolver.preflight` to report the resume decision it
    can know before the basis and rank partition exist. Only the fields present
    in both ``metadata`` and the manifest identity are compared; the solve
    itself always checks the full identity.

    Args:
        root: The configured ``checkpoint_dir``.
        subspace: CI subspace index.
        metadata: Optional partial identity to compare.

    Returns:
        ``{"manifest": bool, "iteration": int | None, "matches": bool | None,
        "mismatched_fields": list[str], "checked_fields": list[str],
        "error": str | None}``. ``matches`` is ``None`` when no manifest exists.
    """
    path = checkpoint_manifest_path(root, subspace)
    result: dict[str, Any] = {
        "manifest": False,
        "iteration": None,
        "matches": None,
        "mismatched_fields": [],
        "checked_fields": [],
        "error": None,
    }
    try:
        manifest = _read_checkpoint_manifest(path)
    except FileNotFoundError:
        return result
    except Exception as exc:  # noqa: BLE001 - preflight reports, never raises
        result["manifest"] = True
        result["matches"] = False
        result["mismatched_fields"] = ["manifest"]
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["manifest"] = True
    try:
        result["iteration"] = int(manifest.get("iteration", -1))
    except (TypeError, ValueError):
        result["iteration"] = None
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        result["mismatched_fields"].append("format_version")
    if metadata is not None:
        current = _checkpoint_jsonable(metadata)
        saved = manifest.get("identity")
        saved = saved if isinstance(saved, dict) else {}
        checked = sorted(set(current) & set(saved))
        result["checked_fields"] = checked
        result["mismatched_fields"].extend(key for key in checked if current[key] != saved[key])
    result["matches"] = not result["mismatched_fields"]
    return result


def parse_distributed_config(
    distributed: "bool | dict | None",
    *,
    device: torch.device | str,
    cuda_device_count: int,
) -> Optional[dict]:
    """Normalize the public ``distributed`` keyword into a launch config.

    Args:
        distributed: ``None``/``False`` (disabled), ``True`` (auto: all visible
            CUDA devices, else the launcher's ``WORLD_SIZE``), or a dict with
            any of ``world_size``, ``backend`` (``"gloo"``/``"nccl"``),
            ``device_ids``, ``master_addr``, ``master_port``, ``init_method``,
            ``timeout``, the matvec/tile/hand-back options, and the checkpoint
            options ``checkpoint_dir`` (shared-filesystem directory or
            ``None``), ``checkpoint_every`` (positive Davidson iterations
            between writes, default 50) and ``checkpoint_resume``
            (``"auto"``/``"off"``/``"require"``, default ``"auto"``).
        device: The solver's configured device, used to choose a backend when
            one is not given.
        cuda_device_count: Number of visible CUDA devices.

    Returns:
        ``None`` when the call should run single-process (disabled, or
        ``world_size <= 1``), else the resolved config dict.

    Raises:
        TypeError: ``distributed`` is neither bool nor dict.
        ValueError: unknown keys, a bad backend, inconsistent device ids, or an
            invalid checkpoint option (a non-positive ``checkpoint_every``, an
            unknown ``checkpoint_resume`` policy, or either without
            ``checkpoint_dir``).
        RuntimeError: NCCL requested without CUDA.
    """
    if distributed is None or distributed is False:
        return None
    if distributed is True:
        raw: dict = {}
    elif isinstance(distributed, dict):
        raw = dict(distributed)
    else:
        raise TypeError(f"distributed must be bool or dict, got {type(distributed).__name__}")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(
            f"unknown distributed keys {sorted(unknown)}; allowed: {sorted(_ALLOWED_KEYS)}"
        )

    world_size = raw.get("world_size")
    auto_world = world_size is None
    if world_size is None:
        world_size = cuda_device_count if cuda_device_count >= 2 else _env_world_size()
    world_size = int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size <= 1:
        return None

    device_ids = raw.get("device_ids")
    if device_ids is not None:
        device_ids = [int(i) for i in device_ids]
        if len(device_ids) != world_size:
            raise ValueError(
                f"device_ids has {len(device_ids)} entries for world_size={world_size}"
            )
        if any(index < 0 for index in device_ids):
            raise ValueError(f"device_ids must be non-negative, got {device_ids}")

    local_rank = os.environ.get("LOCAL_RANK")
    external_launch = _env_world_size() > 1
    backend = raw.get("backend")
    if backend is None:
        # ``distributed=True`` auto-selects all visible GPUs, so it prefers
        # NCCL even when the solver's default device is "cpu"; an explicit
        # dict is read from devices/device_ids. An external launcher
        # (torchrun/srun) keys on CUDA availability, because each rank's
        # CUDA_VISIBLE_DEVICES view may show only its own GPU.
        wants_cuda = (external_launch and torch.cuda.is_available()) or (
            cuda_device_count >= 2
            and (
                (auto_world and device_ids is None)
                or device_ids is not None
                or torch.device(device).type == "cuda"
            )
        )
        backend = "nccl" if wants_cuda else "gloo"
    backend = str(backend).lower()
    if backend not in ("gloo", "nccl"):
        raise ValueError(f"backend must be 'gloo' or 'nccl', got {backend!r}")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='nccl' requires CUDA")
        # A launch that already has a process group (torchrun/srun) may give
        # each rank a one-GPU CUDA_VISIBLE_DEVICES view; a self-spawned job
        # needs world_size distinct devices in this process.
        dist = _dist()
        external = (dist.is_available() and dist.is_initialized()) or _env_world_size() > 1
        if not external and cuda_device_count < world_size:
            raise RuntimeError(
                f"backend='nccl' with world_size={world_size} needs at least "
                f"{world_size} CUDA devices; found {cuda_device_count}"
            )
        if device_ids is None:
            if external and cuda_device_count < world_size:
                device_ids = [0] * world_size
            else:
                device_ids = list(range(world_size))
        if local_rank is None and any(index >= cuda_device_count for index in device_ids):
            raise RuntimeError(
                f"device_ids {device_ids} exceed the visible CUDA devices "
                f"({cuda_device_count}) on this rank"
            )
    else:
        if device_ids is not None:
            raise ValueError(
                "device_ids selects CUDA devices for the NCCL backend; the gloo "
                "backend runs its ranks on CPU. Drop device_ids or pass "
                "backend='nccl'."
            )

    init_method = raw.get("init_method")
    master_addr = str(raw.get("master_addr", "127.0.0.1"))
    master_port = raw.get("master_port")
    auto_port = init_method is None and master_port is None
    if init_method is not None:
        init_method = str(init_method).strip()
        if not init_method:
            raise ValueError("init_method must be a non-empty string")
        scheme = init_method.split("://", 1)[0].lower()
        if scheme not in ("tcp", "env", "file"):
            raise ValueError(
                f"init_method must use a tcp://, env:// or file:// scheme, got {init_method!r}"
            )
    if master_port is not None:
        master_port = int(master_port)
        if not 0 < master_port <= 65535:
            raise ValueError(f"master_port must be in [1, 65535], got {master_port}")
    if init_method is None:
        if master_port is None:
            master_port = _free_port()
        init_method = f"tcp://{master_addr}:{int(master_port)}"

    matvec = str(raw.get("matvec", "local")).lower()
    if matvec not in ("local", "replicated"):
        raise ValueError(f"matvec must be 'local' or 'replicated', got {matvec!r}")
    persistent = raw.get("persistent", True)
    if not isinstance(persistent, bool):
        raise ValueError(f"persistent must be a bool, got {persistent!r}")

    tile = raw.get("tile", False)
    if tile is None or tile is False:
        tile = None
    elif tile is True or (isinstance(tile, str) and tile == "force"):
        tile = "force"
    elif isinstance(tile, str) and tile == "auto":
        tile = "auto"
    elif isinstance(tile, (list, tuple)):
        if len(tile) != 2:
            raise ValueError(f"tile must be [p_alpha, p_beta], got {tile!r}")
        p_alpha, p_beta = int(tile[0]), int(tile[1])
        if p_alpha <= 0 or p_beta <= 0 or p_alpha * p_beta != world_size:
            raise ValueError(
                f"tile grid ({p_alpha}, {p_beta}) must have positive entries "
                f"multiplying to world_size={world_size}"
            )
        tile = (p_alpha, p_beta)
    else:
        raise ValueError(
            f"tile must be False, True, 'auto', 'force' or [p_alpha, p_beta], got {tile!r}"
        )

    if tile is not None and matvec == "replicated":
        raise ValueError(
            "distributed={'tile': ...} selects the owner-computes tile layout and "
            "cannot be combined with matvec='replicated'. Drop the tile spec or "
            "pass matvec='local'."
        )

    exchange_bytes = raw.get("exchange_bytes")
    if exchange_bytes is not None:
        exchange_bytes = int(exchange_bytes)
        if exchange_bytes <= 0:
            raise ValueError(f"exchange_bytes must be > 0, got {exchange_bytes}")
        if tile is None:
            raise ValueError(
                "exchange_bytes caps one staged tile exchange; pass "
                "distributed={'tile': ...} alongside it (the 1-D row path has no "
                "tile exchange)."
            )

    hand_back = str(raw.get("hand_back", "auto")).lower()
    if hand_back not in HAND_BACK_MODES:
        raise ValueError(f"hand_back must be one of {HAND_BACK_MODES}, got {hand_back!r}")
    hand_back_bytes = raw.get("hand_back_bytes")
    if hand_back_bytes is not None:
        hand_back_bytes = int(hand_back_bytes)
        if hand_back_bytes <= 0:
            raise ValueError(f"hand_back_bytes must be positive or None, got {hand_back_bytes}")
    budget_bytes = raw.get("budget_bytes")
    if budget_bytes is not None:
        budget_bytes = int(budget_bytes)
        if budget_bytes <= 0:
            raise ValueError(f"budget_bytes must be positive or None, got {budget_bytes}")
    startup_timeout = raw.get("worker_startup_timeout_s")
    if startup_timeout is None:
        startup_timeout = float(os.environ.get("TORCH_SCI_STARTUP_TIMEOUT_S", "600"))
    else:
        startup_timeout = float(startup_timeout)
        if startup_timeout <= 0:
            raise ValueError(f"worker_startup_timeout_s must be positive, got {startup_timeout}")
    timeout = float(raw.get("timeout", 1800.0))
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    hand_back_dir = raw.get("hand_back_dir")
    if hand_back_dir is not None:
        hand_back_dir = str(hand_back_dir)

    checkpoint_dir = raw.get("checkpoint_dir")
    if checkpoint_dir is not None:
        checkpoint_dir = str(checkpoint_dir)
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir must be a non-empty path or None")
    checkpoint_every = int(raw.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY))
    if checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be positive, got {checkpoint_every}")
    checkpoint_resume = str(raw.get("checkpoint_resume", "auto")).lower()
    if checkpoint_resume not in CHECKPOINT_RESUME_MODES:
        raise ValueError(
            f"checkpoint_resume must be one of {CHECKPOINT_RESUME_MODES}, got {checkpoint_resume!r}"
        )
    if checkpoint_dir is None:
        if "checkpoint_every" in raw:
            raise ValueError(
                "checkpoint_every only applies when checkpoint_dir is set; "
                "pass checkpoint_dir (or drop checkpoint_every)."
            )
        if "checkpoint_resume" in raw:
            raise ValueError(
                "checkpoint_resume only applies when checkpoint_dir is set; "
                "pass checkpoint_dir (or drop checkpoint_resume)."
            )

    return {
        "world_size": world_size,
        "backend": backend,
        "device_ids": device_ids,
        "device_ids_explicit": raw.get("device_ids") is not None,
        "init_method": str(init_method),
        "master_addr": master_addr,
        "master_port": int(master_port) if master_port is not None else None,
        "timeout": timeout,
        "matvec": matvec,
        "persistent": persistent,
        "tile": tile,
        "exchange_bytes": exchange_bytes,
        "hand_back": hand_back,
        "hand_back_bytes": hand_back_bytes,
        "hand_back_dir": hand_back_dir,
        "budget_bytes": budget_bytes,
        "worker_startup_timeout_s": startup_timeout,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_every": checkpoint_every,
        "checkpoint_resume": checkpoint_resume,
        "auto_port": auto_port,
    }


def _rank_device(cfg: dict, rank: int) -> torch.device:
    """Device a worker rank uses: its CUDA id for NCCL, CPU for gloo."""
    if cfg["backend"] == "nccl":
        local_rank = os.environ.get("LOCAL_RANK")
        if cfg["device_ids_explicit"] or local_rank is None:
            index = cfg["device_ids"][rank]
        else:
            index = 0 if torch.cuda.device_count() == 1 else int(local_rank)
        return torch.device("cuda", index)
    return torch.device("cpu")


def _make_context(cfg: dict, *, rank: int, world_size: int, dev: torch.device):
    """Build the rank's :class:`DistributedContext` from the resolved config."""
    from torch_sci_solver.torch_impl.distributed_davidson import DistributedContext

    return DistributedContext(
        rank=rank,
        world_size=world_size,
        backend=cfg["backend"],
        device=dev,
        group=None,
        matvec=cfg["matvec"],
        tile=cfg.get("tile"),
        exchange_bytes=cfg.get("exchange_bytes"),
        hand_back_dir=cfg.get("hand_back_dir"),
    )


def _apply_runtime_config(solver: Any, cfg: dict) -> None:
    """Attach the resolved distributed execution options to ``solver``.

    ``DistributedContext`` is frozen and shared with the numerical layers, so
    the hand-back policy, the per-rank byte budget and the checkpoint options
    live on the solver's runtime attributes instead. A spawned worker rebuilds
    its solver with ``distributed=None``, so these attributes are how it learns
    the parent's checkpoint settings.
    """
    solver._runtime_hand_back = cfg.get("hand_back", "auto")
    solver._runtime_hand_back_bytes = cfg.get("hand_back_bytes")
    solver._runtime_budget_bytes = cfg.get("budget_bytes")
    checkpoint_dir = cfg.get("checkpoint_dir")
    solver._runtime_checkpoint = (
        {
            "dir": str(checkpoint_dir),
            "every": int(cfg.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY)),
            "resume": str(cfg.get("checkpoint_resume", "auto")),
        }
        if checkpoint_dir is not None
        else None
    )


def _apply_triton_block(solver: Any) -> "Optional[int]":
    """Point the distributed tile kernels at the solver's ``triton_block``.

    ``distributed_matvec`` reads its module-level launch block per call, so
    this restores the previous value after the solve. Returns the previous
    value for :func:`_restore_triton_block` (``None`` when nothing changed).
    """
    block = getattr(solver, "triton_block", None)
    if block is None:
        return None
    from torch_sci_solver.torch_impl import distributed_matvec

    previous = distributed_matvec._TRITON_BLOCK
    distributed_matvec._TRITON_BLOCK = int(block)
    return previous


def _restore_triton_block(previous: "Optional[int]") -> None:
    """Restore the launch block saved by :func:`_apply_triton_block`."""
    if previous is None:
        return
    from torch_sci_solver.torch_impl import distributed_matvec

    distributed_matvec._TRITON_BLOCK = int(previous)


def _triton_available() -> bool:
    """Whether the Triton matvec kernels import in this process."""
    try:
        from torch_sci_solver.torch_impl.mult_triton import _HAVE_TRITON
    except Exception:  # noqa: BLE001 - a broken install behaves as unavailable
        return False
    return bool(_HAVE_TRITON)


def _reject_triton_on_gloo(requested: str, backend: str) -> None:
    """Reject an explicit Triton request on the CPU-only gloo backend.

    Args:
        requested: The caller's ``matvec_impl`` before resolution.
        backend: Process-group backend.

    Raises:
        ValueError: ``backend`` is gloo and ``requested`` is ``"triton"``.
    """
    if backend == "gloo" and requested == "triton":
        raise ValueError(
            "matvec_impl='triton' requires CUDA, but the gloo distributed backend "
            "runs its ranks on CPU. Pass matvec_impl='batched' (or 'auto') for a "
            "CPU/gloo run."
        )


def _worker_matvec_impl(
    requested: str,
    *,
    mode: str,
    dev: torch.device,
    backend: str,
    table_dtype: str = "auto",
) -> str:
    """Resolve one worker rank's matvec backend against the device it will use.

    The solver resolves ``"auto"`` once at construction time, which under a
    launcher (torchrun/srun) may have seen only the host's CPU device: an NCCL
    rank actually runs on CUDA and must re-resolve to the Triton kernels
    before the solve. An explicit request is returned unchanged; one the
    backend cannot honour raises here instead of being silently rewritten.

    Args:
        requested: The caller's ``matvec_impl`` (``"auto"``, ``"batched"`` or
            ``"triton"``) before construction-time resolution.
        mode: Solver mode; GDB keeps ``"auto"`` on the batched pair path.
        dev: Device this rank will run on.
        backend: Process-group backend (``"gloo"`` or ``"nccl"``).
        table_dtype: The solver's ``table_dtype`` policy. Narrowed (int32)
            tables cannot be read by the Triton kernels, so they force the
            batched path.

    Returns:
        The resolved backend for this rank.

    Raises:
        ValueError: an explicit ``"triton"`` request on the CPU-only gloo
            backend, or one combined with ``table_dtype="int32"``.
        RuntimeError: ``"auto"`` selected Triton on a CUDA rank but the
            kernels do not import.
    """
    _reject_triton_on_gloo(requested, backend)
    if table_dtype == "int32":
        if requested == "triton":
            raise ValueError(
                "matvec_impl='triton' reads int64 table payloads; pass "
                "table_dtype='int64' or matvec_impl='batched'."
            )
        return "batched"
    if requested != "auto":
        return requested
    if mode == "gdb" or dev.type != "cuda":
        return "batched"
    if not _triton_available():
        raise RuntimeError(
            "matvec_impl='auto' selects the Triton backend on a CUDA rank, but "
            "Triton is not available (failed to import triton). Install a working "
            "Triton, or pass matvec_impl='batched' explicitly."
        )
    return "triton"


def _run_in_process(
    solver: Any,
    cfg: dict,
    payload: tuple,
    *,
    rank: int,
    world_size: int,
) -> Any:
    """Run the solver body on this already-initialized rank."""
    dev = _rank_device(cfg, rank)
    requested = getattr(solver, "_matvec_impl_requested", solver.matvec_impl)
    # Resolve before mutating the solver, so a rejected request leaves it intact.
    resolved_matvec_impl = _worker_matvec_impl(
        requested,
        mode=getattr(solver, "mode", "tpb"),
        dev=dev,
        backend=cfg["backend"],
        table_dtype=getattr(solver, "table_dtype", "auto"),
    )
    if dev.type == "cuda":
        torch.cuda.set_device(dev)

    saved = (
        solver.device,
        solver.devices,
        solver.matvec_impl,
        solver._distributed_ctx,
        solver._inside_distributed,
        getattr(solver, "_runtime_hand_back", "auto"),
        getattr(solver, "_runtime_hand_back_bytes", None),
        getattr(solver, "_runtime_budget_bytes", None),
        getattr(solver, "_runtime_checkpoint", None),
    )
    solver.device = dev
    solver.devices = (dev,)
    solver.matvec_impl = resolved_matvec_impl
    solver._distributed_ctx = _make_context(cfg, rank=rank, world_size=world_size, dev=dev)
    solver._inside_distributed = True
    _apply_runtime_config(solver, cfg)
    previous_block = _apply_triton_block(solver)
    ci_strings, one_body, two_body, norb, nelec, spin_sq, call_kwargs = payload
    try:
        result = solver(
            ci_strings,
            one_body,
            two_body,
            norb,
            nelec,
            spin_sq=spin_sq,
            **call_kwargs,
        )
        # In-process (torchrun/srun) callers get rank 0's result directly; the
        # memmap markers of a large hand-back must become live arrays here,
        # since no parent process will do it.
        if rank == 0:
            result = materialize_result(result)
        return result
    finally:
        (
            solver.device,
            solver.devices,
            saved_matvec_impl,
            solver._distributed_ctx,
            solver._inside_distributed,
            solver._runtime_hand_back,
            solver._runtime_hand_back_bytes,
            solver._runtime_budget_bytes,
            solver._runtime_checkpoint,
        ) = saved
        # An "auto" request keeps the rank-resolved backend, mirroring the
        # constructor's one-time resolution, so post-run metadata names the
        # backend that actually ran. An explicit request is restored untouched.
        if requested != "auto":
            solver.matvec_impl = saved_matvec_impl
        _restore_triton_block(previous_block)


def _build_rank_solver(rank: int, cfg: dict, factory_kwargs: dict) -> tuple[Any, Any, Any]:
    """Initialize one spawned rank and build its rank-local solver.

    Shared by the one-shot and persistent workers: initialize the process
    group, resolve the rank's device, rewrite the factory kwargs for a spawned
    rank (no nested distributed launch, one device), construct the solver,
    attach the distributed context, apply the runtime config and set this
    rank's Triton block.

    Args:
        rank: this process's rank.
        cfg: the resolved distributed launch config.
        factory_kwargs: the parent's solver-construction kwargs.

    Returns:
        ``(solver, restore_triton, dist)``: ``restore_triton`` is the previous
        Triton block for :func:`_restore_triton_block`, and ``dist`` is the
        lazily imported ``torch.distributed`` module for the group teardown.
    """
    dist = _dist()
    dist.init_process_group(
        backend=cfg["backend"],
        init_method=cfg["init_method"],
        rank=rank,
        world_size=cfg["world_size"],
        timeout=timedelta(seconds=cfg["timeout"]),
    )
    try:
        dev = _rank_device(cfg, rank)
        if factory_kwargs.get("deterministic"):
            # cuBLAS reads its workspace setting when the rank creates its first
            # handle, so it must be in this process's environment beforehand.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        kwargs = dict(factory_kwargs)
        kwargs["distributed"] = None
        kwargs["devices"] = None
        kwargs["device"] = dev
        # Re-resolve "auto" against the device this rank actually runs on; an
        # explicit request the backend cannot honour raises before any solve.
        kwargs["matvec_impl"] = _worker_matvec_impl(
            kwargs.get("matvec_impl", "auto"),
            mode=kwargs.get("mode", "tpb"),
            dev=dev,
            backend=cfg["backend"],
            table_dtype=kwargs.get("table_dtype", "auto"),
        )

        from torch_sci_solver.torch_impl.qiskit_adapter import TorchSCISolver

        worker_solver = TorchSCISolver(**kwargs)
        worker_solver._distributed_ctx = _make_context(
            cfg, rank=rank, world_size=cfg["world_size"], dev=dev
        )
        _apply_runtime_config(worker_solver, cfg)
        previous_block = _apply_triton_block(worker_solver)
    except BaseException:
        dist.destroy_process_group()
        raise
    return worker_solver, previous_block, dist


def _distributed_worker(
    rank: int,
    cfg: dict,
    factory_kwargs: dict,
    payload: tuple,
    result_path: str,
) -> None:
    """One spawned rank: init the process group, solve, write rank 0's result.

    The result is written with ``torch.save`` rather than put on a
    ``multiprocessing.Queue``: a queue write blocks once the pipe buffer
    (~64 KiB) fills, and the parent would be waiting in ``join`` instead of
    reading. A file has no size limit and cannot deadlock.
    """
    worker_solver, previous_block, dist = _build_rank_solver(rank, cfg, factory_kwargs)
    try:
        ci_strings, one_body, two_body, norb, nelec, spin_sq, call_kwargs = payload
        result = worker_solver(
            ci_strings,
            one_body,
            two_body,
            norb,
            nelec,
            spin_sq=spin_sq,
            **call_kwargs,
        )
        if rank == 0:
            _save_worker_result(
                result_path,
                {
                    "result": result,
                    "stats": dict(worker_solver.stats),
                    "matvec_impl": worker_solver.matvec_impl,
                },
            )
    finally:
        _restore_triton_block(previous_block)
        dist.destroy_process_group()


def _save_worker_result(path: str, payload: dict) -> None:
    """Write a worker result with a pickle protocol that has no 4 GiB cap.

    ``torch.save`` defaults to protocol 2, which cannot serialize an object
    larger than 4 GiB; a single 4·10⁸-determinant fp64 eigenvector is already
    3.2 GiB, so the surrounding result crosses that limit.

    Args:
        path: destination file for the serialized result.
        payload: result and solver stats to write.
    """
    torch.save(payload, path, pickle_protocol=4)


def _session_worker(
    rank: int,
    cfg: dict,
    factory_kwargs: dict,
    conn: Any,
) -> None:
    """Persistent worker: initialize once, then solve one payload per message."""
    worker_solver, previous_block, dist = _build_rank_solver(rank, cfg, factory_kwargs)
    # Handshake: the parent bounds worker startup by
    # ``worker_startup_timeout_s`` instead of blocking forever on the first
    # result. Construction above (CUDA context, Triton import) is the slow
    # part this covers.
    try:
        conn.send("ready")
        while True:
            try:
                command = conn.recv()
            except EOFError:
                break
            if command is None:
                break
            payload, result_path = command
            try:
                ci_strings, one_body, two_body, norb, nelec, spin_sq, call_kwargs = payload
                result = worker_solver(
                    ci_strings,
                    one_body,
                    two_body,
                    norb,
                    nelec,
                    spin_sq=spin_sq,
                    **call_kwargs,
                )
                if rank == 0:
                    _save_worker_result(
                        result_path,
                        {
                            "result": result,
                            "stats": dict(worker_solver.stats),
                            "matvec_impl": worker_solver.matvec_impl,
                        },
                    )
                conn.send("ok")
            except BaseException as exc:  # noqa: BLE001 - report to the parent
                try:
                    conn.send(f"err: {type(exc).__name__}: {exc}")
                except Exception:  # noqa: BLE001 - parent is gone; exit
                    pass
                break
        _restore_triton_block(previous_block)
    finally:
        dist.destroy_process_group()


class DistributedSession:
    """Persistent spawned rank pool reused across solver calls.

    Worker startup (CUDA context, Triton JIT, helper caches) is paid once per
    solver, not once per SQD iteration. The parent sends one payload to each
    rank per call; rank 0 writes the result to a temp file and each rank acks.
    ``close()`` (also wired through the solver's ``close``) sends a sentinel to
    every worker and joins it.
    """

    def __init__(self, factory_kwargs: dict, cfg: dict, stats: dict) -> None:
        """Spawn and initialize one worker per rank."""
        self.stats = stats
        self.world_size = cfg["world_size"]
        # Backend the rank-0 worker resolved against its device; the parent
        # copies it after each call so benchmark metadata sees the truth.
        self.matvec_impl: Optional[str] = None
        self._startup_timeout_s = float(cfg.get("worker_startup_timeout_s", 600.0))
        self._closed = False
        spawn_cfg = dict(cfg)
        if spawn_cfg.get("auto_port"):
            spawn_cfg["init_method"] = f"tcp://{spawn_cfg['master_addr']}:{_free_port()}"
        self._ctx = multiprocessing.get_context("spawn")
        self._conns = []
        self._procs = []
        for rank in range(self.world_size):
            parent_conn, child_conn = self._ctx.Pipe()
            process = self._ctx.Process(
                target=_session_worker,
                args=(rank, spawn_cfg, factory_kwargs, child_conn),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(process)
        # Wait for every rank to finish initializing (CUDA context, Triton
        # import, process group) before the first payload. Without this a
        # startup hang would show up as an unexplained blocking ``recv``.
        for rank, conn in enumerate(self._conns):
            if not conn.poll(self._startup_timeout_s):
                self.close()
                raise RuntimeError(
                    f"distributed rank {rank} did not report ready within "
                    f"{self._startup_timeout_s:g}s (worker_startup_timeout_s). "
                    "Increase it for slow shared-filesystem/CUDA startup, or check "
                    "the rank's log for an initialization failure."
                )
            try:
                status = conn.recv()
            except EOFError as exc:
                self.close()
                raise RuntimeError(f"distributed rank {rank} died during startup") from exc
            if status != "ready":
                self.close()
                raise RuntimeError(f"distributed rank {rank} failed to start: {status}")

    def run(self, payload: tuple) -> Any:
        """Run one solver call on the persistent pool."""
        if self._closed:
            raise RuntimeError("distributed session is closed")
        with tempfile.TemporaryDirectory(prefix="torch_sci_dist_session_") as workdir:
            result_path = str(Path(workdir) / "result.pt")
            for conn in self._conns:
                try:
                    conn.send((payload, result_path))
                except (BrokenPipeError, EOFError) as exc:
                    self.close()
                    raise RuntimeError("a distributed worker died before the call") from exc
            for rank, conn in enumerate(self._conns):
                try:
                    status = conn.recv()
                except EOFError as exc:
                    self.close()
                    raise RuntimeError(f"distributed rank {rank} died during the call") from exc
                if status != "ok":
                    self.close()
                    raise RuntimeError(f"distributed rank {rank} failed: {status}")
            if not Path(result_path).exists():
                self.close()
                raise RuntimeError("rank 0 did not write a result")
            saved = torch.load(result_path, weights_only=False)
        # The workers' stats are cumulative across session calls; replace the
        # parent's distributed-call stats rather than double counting.
        self.stats.clear()
        self.stats.update(saved["stats"])
        if saved.get("matvec_impl"):
            self.matvec_impl = str(saved["matvec_impl"])
        return materialize_result(saved["result"])

    def close(self) -> None:
        """Send a sentinel to every rank and join the workers."""
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, EOFError):
                pass
        for process in self._procs:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for conn in self._conns:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 - already closed
                pass

    def __enter__(self) -> "DistributedSession":
        """Return the session for ``with`` use."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the session on context exit."""
        self.close()


def _close_session(session: DistributedSession) -> None:
    """Finalizer target that holds the session, not the solver."""
    session.close()


# Stats that hold a value rather than a running total: a later subspace's
# result replaces the previous one instead of adding to it.
_REPLACED_STATS = frozenset({"spin_sq_final"})


def _merge_stats(target: dict, delta: dict) -> None:
    """Merge a per-call worker stats dict into the parent's cumulative stats."""
    for key, value in delta.items():
        if key in _REPLACED_STATS:
            if value is not None:
                target[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, list):
            target.setdefault(key, []).extend(value)
        elif value is not None:
            target[key] = value


def _spawn(solver: Any, cfg: dict, payload: tuple) -> Any:
    """Launch ``world_size`` spawned workers and return rank 0's result.

    Rank 0 writes the result to a temp file before exiting; ``join=True`` can
    therefore never deadlock on a full pipe, and an oversized result (large
    ``SCIState.amplitudes``, RDMs) is fine.
    """
    spawn_cfg = dict(cfg)
    if spawn_cfg.get("auto_port"):
        spawn_cfg["init_method"] = f"tcp://{spawn_cfg['master_addr']}:{_free_port()}"
    with tempfile.TemporaryDirectory(prefix="torch_sci_dist_") as workdir:
        result_path = str(Path(workdir) / "result.pt")
        torch.multiprocessing.start_processes(
            _distributed_worker,
            args=(spawn_cfg, solver._factory_kwargs, payload, result_path),
            nprocs=cfg["world_size"],
            start_method="spawn",
            join=True,
        )
        if not Path(result_path).exists():
            raise RuntimeError(
                "distributed workers exited without writing a result (rank 0 failed)"
            )
        saved = torch.load(result_path, weights_only=False)
    if saved.get("stats"):
        _merge_stats(solver.stats, saved["stats"])
    if saved.get("matvec_impl"):
        # Rank 0 resolved "auto" against its real device; keep that value so
        # metadata recorded after the call names the backend that ran.
        solver.matvec_impl = str(saved["matvec_impl"])
    return materialize_result(saved["result"])


def run_distributed_call(
    solver: Any,
    ci_strings: Any,
    one_body_tensor: Any,
    two_body_tensor: Any,
    norb: int,
    nelec: Any,
    spin_sq: Optional[float],
    call_kwargs: dict,
) -> Any:
    """Run one solver call through the distributed path when it is enabled.

    Returns :data:`NOT_DISTRIBUTED` when the call should use the existing
    single-process path (disabled, ``world_size <= 1``, or a group of one).

    Raises:
        NotImplementedError: the call needs a feature the replicated fallback
            does not implement (a multi-root missed-root search).
        ValueError: ``devices=`` and ``distributed=`` were both requested, an
            initialized group's world size disagrees with the config, or an
            explicit ``matvec_impl="triton"`` was combined with gloo.
    """
    if getattr(solver, "_inside_distributed", False):
        return NOT_DISTRIBUTED
    if getattr(solver, "_distributed_config", None) is None:
        return NOT_DISTRIBUTED
    cfg = solver._distributed_config

    if solver.devices is not None and len(solver.devices) > 1:
        raise ValueError(
            "devices= (batch-parallel multi-GPU) and distributed= are mutually exclusive"
        )

    # Fail an explicit Triton request on the CPU-only backend here, where the
    # caller sees the ValueError, instead of surfacing it as a worker crash.
    _reject_triton_on_gloo(
        getattr(solver, "_matvec_impl_requested", solver.matvec_impl), cfg["backend"]
    )

    # The replicated fallback (``davidson_distributed``) has no missed-root
    # search. Feature gates that force that fallback are known here, so reject
    # the knob before spawning instead of surfacing a ProcessRaisedException;
    # the workers keep the same guard as a backstop. The fallback carries
    # complex coefficients (the real Hamiltonian diagonal is unchanged), so the
    # guard applies to every dtype; the spinless path runs ``_run_davidson`` on
    # every rank in its replicated branch and never takes the fallback, and a
    # small GDB basis is only known in the worker, which raises there.
    # A tiled solve computes its full RDMs and the <S^2> diagnostic from its
    # own staged exchanges, and a local solve computes the spin penalty on a
    # tile or on the 1-D owner-computes block, so those knobs do not force the
    # replicated fallback there; without a grid the RDM knob still does, and the
    # <S^2> knob does outside an explicit GDB pair list (the pre-spawn guard is
    # conservative: the layout and basis size are resolved in the worker). An
    # "auto" spec is only a policy until the basis is known, and a
    # decline may land on the replicated path, so it is treated conservatively
    # here instead of failing late in a worker.
    tile_postprocess = cfg.get("tile") not in (None, "auto")
    local_penalty = spin_sq is not None and cfg["matvec"] == "local"
    replicated = not isinstance(nelec, (int, np.integer)) and (
        cfg["matvec"] == "replicated"
        or (spin_sq is not None and not local_penalty)
        or (solver.compute_full_rdm and not tile_postprocess)
        or (solver.compute_spin_squared and not tile_postprocess)
    )
    if replicated:
        missed_root = solver.n_roots > 1 and (
            solver.davidson_missed_root_check is True
            or (
                solver.davidson_missed_root_check == "auto"
                and call_kwargs.get("probe_state") is not None
            )
        )
        if missed_root:
            raise NotImplementedError(
                "davidson_missed_root_check on a multi-root distributed solve requires "
                "the owner-computes path; this call uses the replicated matvec"
            )

    payload = (
        ci_strings,
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
        spin_sq,
        call_kwargs,
    )

    dist = _dist()
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size <= 1:
            return NOT_DISTRIBUTED
        if world_size != cfg["world_size"]:
            raise ValueError(
                f"the initialized process group has world_size={world_size}, "
                f"but distributed requested world_size={cfg['world_size']}"
            )
        return _run_in_process(solver, cfg, payload, rank=dist.get_rank(), world_size=world_size)

    env_world = _env_world_size()
    if env_world > 1:
        if env_world != cfg["world_size"]:
            raise ValueError(
                f"launcher WORLD_SIZE={env_world} disagrees with "
                f"distributed world_size={cfg['world_size']}"
            )
        dist.init_process_group(
            backend=cfg["backend"],
            init_method="env://",
            timeout=timedelta(seconds=cfg["timeout"]),
        )
        try:
            return _run_in_process(solver, cfg, payload, rank=dist.get_rank(), world_size=env_world)
        finally:
            dist.destroy_process_group()

    if cfg.get("persistent", True):
        session = getattr(solver, "_distributed_session", None)
        if session is None:
            session = DistributedSession(solver._factory_kwargs, cfg, solver.stats)
            solver._distributed_session = session
            solver._distributed_session_finalizer = weakref.finalize(
                solver, _close_session, session
            )
        try:
            result = session.run(payload)
        except BaseException:
            session.close()
            solver._distributed_session = None
            raise
        if session.matvec_impl is not None:
            solver.matvec_impl = session.matvec_impl
        return result
    return _spawn(solver, cfg, payload)
