"""Tests for :class:`FlatGather`, the flat operators' asynchronous input gather.

1. Single process: ``world_size == 1`` returns a copy of the block, and
   ``start``/``wait`` must alternate.
2. gloo, 2 and 3 ranks: even and uneven contiguous partitions, real and complex
   blocks, gathered to exactly the global vector, repeatedly (the block sizes
   are exchanged once, on the first start).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import pathlib

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_sci_solver.torch_impl.distributed_davidson import FlatGather, RankBounds


def test_single_rank_returns_a_copy_and_pairs_start_with_wait() -> None:
    """One rank: ``wait`` returns a copy; a second ``start`` before ``wait`` raises."""
    x = torch.arange(5, dtype=torch.float64)
    gather = FlatGather(RankBounds(n=5, world_size=1, rank=0, start=0, stop=5))
    pending = gather.start(x)
    assert pending.done()
    with pytest.raises(RuntimeError, match="twice"):
        gather.start(x)
    full = pending.wait()
    assert torch.equal(full, x) and full.data_ptr() != x.data_ptr()
    with pytest.raises(RuntimeError, match="without start"):
        gather.wait()
    with pytest.raises(ValueError, match="shape"):
        gather.start(x[:3])


def _worker(rank: int, world_size: int, starts: list[int], init_file: str, out_file: str) -> None:
    """Gather real and complex blocks of a known vector on every rank, twice."""
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        n = starts[-1]
        bounds = RankBounds(
            n=n, world_size=world_size, rank=rank, start=starts[rank], stop=starts[rank + 1]
        )
        reference = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
        complex_reference = torch.complex(reference, reference.flip(0))
        gather = FlatGather(bounds)
        for scale in (1.0, -3.0):
            real = gather.start(scale * reference[bounds.start : bounds.stop]).wait()
            assert torch.equal(real, scale * reference), (rank, real)
        cplx = gather.start(complex_reference[bounds.start : bounds.stop]).wait()
        assert cplx.dtype == torch.complex128
        assert torch.equal(cplx, complex_reference)
    finally:
        dist.destroy_process_group()
    pathlib.Path(out_file).write_text("ok")


@pytest.mark.parametrize(
    "starts",
    [[0, 6, 12], [0, 7, 11], [0, 4, 8, 12], [0, 5, 6, 13], [0, 0, 9]],
    ids=["2-even", "2-uneven", "3-even", "3-uneven", "2-empty-rank"],
)
def test_gloo_gather_reassembles_the_global_vector(tmp_path, starts: list[int]) -> None:
    """Every rank receives the exact global vector for even and uneven blocks."""
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("torch.distributed with the gloo backend is required")
    world_size = len(starts) - 1
    init_file = tmp_path / "init"
    out_files = [tmp_path / f"out{r}" for r in range(world_size)]
    mp.start_processes(
        _spawn_entry,
        args=(world_size, starts, str(init_file), [str(f) for f in out_files]),
        nprocs=world_size,
        start_method="spawn",
        join=True,
    )
    assert all(f.read_text() == "ok" for f in out_files)


def _spawn_entry(
    rank: int, world_size: int, starts: list[int], init_file: str, out_files: list[str]
) -> None:
    """Route each rank to its own marker file."""
    _worker(rank, world_size, starts, init_file, out_files[rank])


def test_prefetch_draws_one_chunk_while_the_gather_is_pending() -> None:
    """``prefetch_while_pending`` pulls exactly one item ahead of a pending gather."""
    from torch_sci_solver.torch_impl.mult_pairs import prefetch_while_pending

    pulled: list[int] = []

    def stream():
        for i in range(4):
            pulled.append(i)
            yield i

    class _Pending:
        def __init__(self, done: bool) -> None:
            self._done = done

        def done(self) -> bool:
            return self._done

    chained = prefetch_while_pending(stream(), _Pending(done=False))
    assert pulled == [0]
    assert list(chained) == [0, 1, 2, 3]
    pulled.clear()
    chained = prefetch_while_pending(stream(), _Pending(done=True))
    assert pulled == []
    assert list(chained) == [0, 1, 2, 3]
