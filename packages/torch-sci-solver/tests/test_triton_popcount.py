"""Check TPB population counts and multi-limb ranges against Python integers."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import random

import pytest
import torch
from torch_sci_solver.torch_impl import mult_triton as mt


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tpb_popcount_and_limb_ranges(seed: int) -> None:
    """Include signed words, bit 63, empty ranges, and ranges crossing bit 64."""
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TPB Triton kernels")
    import triton.language as tl

    @triton.jit
    def check_bits(
        words_ptr,
        lo_ptr,
        hi_ptr,
        counts_ptr,
        ranges_ptr,
        N: tl.constexpr,
    ):
        rows = tl.arange(0, 128)
        active = rows < N
        first = tl.load(words_ptr + rows * 2, mask=active, other=0)
        lo = tl.load(lo_ptr + rows, mask=active, other=0)
        hi = tl.load(hi_ptr + rows, mask=active, other=0)
        tl.store(counts_ptr + rows, mt._popc64(first), mask=active)
        counts = mt._string_range_popcount(
            first,
            words_ptr,
            rows,
            lo,
            hi,
            active,
            N_LIMBS=2,
        )
        tl.store(ranges_ptr + rows, counts, mask=active)

    rng = random.Random(seed)
    mask = (1 << 64) - 1
    patterns = [0, 1, 1 << 63, mask, mask >> 1, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555]
    patterns.extend(rng.getrandbits(64) for _ in range(64))
    pairs = [(value, patterns[-index - 1]) for index, value in enumerate(patterns)]
    windows = [(0, 128), (0, 64), (63, 65), (64, 128), (65, 127), (-5, 129), (64, 64)]
    windows = [windows[index % len(windows)] for index in range(len(pairs))]
    signed = [[value if value < 1 << 63 else value - (1 << 64) for value in pair] for pair in pairs]
    words = torch.tensor(signed, dtype=torch.int64, device="cuda")
    lo, hi = torch.tensor(windows, dtype=torch.int64, device="cuda").T.contiguous()
    counts = torch.empty(len(pairs), dtype=torch.int64, device="cuda")
    ranges = torch.empty_like(counts)
    check_bits[(1,)](words, lo, hi, counts, ranges, len(pairs), num_warps=4)

    expected_ranges = []
    for (first, second), (start, end) in zip(pairs, windows, strict=True):
        start, end = max(0, start), min(128, end)
        window_mask = ((1 << end) - 1) ^ ((1 << start) - 1)
        expected_ranges.append(((first | (second << 64)) & window_mask).bit_count())
    assert counts.cpu().tolist() == [value.bit_count() for value in patterns]
    assert ranges.cpu().tolist() == expected_ranges


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["native", "swar"])
def test_popc64_variants_match_python(variant: str) -> None:
    """Both popcount implementations count all 64 bits, including bit 63."""
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the TPB Triton kernels")
    if variant == "native" and torch.version.hip is not None:
        pytest.skip("native popcount is PTX and exists only on the NVIDIA backend")
    import triton.language as tl

    @triton.jit
    def popcount(words_ptr, counts_ptr, N: tl.constexpr, POPC: tl.constexpr):
        rows = tl.arange(0, 128)
        active = rows < N
        words = tl.load(words_ptr + rows, mask=active, other=0)
        tl.store(counts_ptr + rows, POPC(words), mask=active)

    rng = random.Random(0)
    mask = (1 << 64) - 1
    values = [0, 1, 1 << 63, mask, mask >> 1, 0xAAAAAAAAAAAAAAAA, 0x5555555555555555]
    values.extend(rng.getrandbits(64) for _ in range(64))
    signed = [value if value < 1 << 63 else value - (1 << 64) for value in values]
    words = torch.tensor(signed, dtype=torch.int64, device="cuda")
    counts = torch.empty_like(words)
    popc = mt._popc64_native if variant == "native" else mt._popc64_swar
    popcount[(1,)](words, counts, len(values), popc, num_warps=4)

    assert counts.cpu().tolist() == [value.bit_count() for value in values]
