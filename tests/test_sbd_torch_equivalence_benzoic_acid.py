"""SBD/TorchSCI equivalence checks on the bundled benzoic acid (SQD molecules dataset ID 182)."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import pytest
from sbd_equivalence_utils import (
    BENZOIC_ACID_DEEP_SYSTEM,
    BENZOIC_ACID_SMALL_SYSTEM,
    DEVICES,
    assert_gdb_full_product_agrees,
    assert_gdb_nonproduct_agrees,
    assert_torch_matches_dense,
    assert_tpb_agrees,
)


@pytest.mark.parametrize("device", DEVICES)
def test_sbd_tpb_energy_matches_torch(device: str) -> None:
    """Test whether SBD and TorchSCI agree in TPB mode.

    Args:
        device: Device
    """
    assert_tpb_agrees(
        "sbd",
        BENZOIC_ACID_SMALL_SYSTEM,
        device,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_sbd_gdb_energy_matches_torch_on_full_product(device: str) -> None:
    """Test whether SBD and TorchSCI agree in GDB mode.

    Args:
        device: Device
    """
    assert_gdb_full_product_agrees(
        BENZOIC_ACID_SMALL_SYSTEM,
        device,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_sbd_gdb_energy_matches_torch_on_nonproduct_subset(device: str) -> None:
    """Test whether SBD and TorchSCI agree in GDB mode.

    Args:
        device: Device
    """
    assert_gdb_nonproduct_agrees(
        BENZOIC_ACID_SMALL_SYSTEM,
        device,
    )


@pytest.mark.parametrize("comparison", ["tpb", "gdb_full", "gdb_nonproduct"])
def test_benzoic_acid_deep_energy_matches_torch_on_gpu(comparison: str) -> None:
    """Test whether SBD and TorchSCI agree.

    Args:
        comparison: comparison
    """
    if comparison == "tpb":
        assert_tpb_agrees(
            "sbd",
            BENZOIC_ACID_DEEP_SYSTEM,
            "gpu",
        )
    elif comparison == "gdb_full":
        assert_gdb_full_product_agrees(
            BENZOIC_ACID_DEEP_SYSTEM,
            "gpu",
        )
    else:
        assert_gdb_nonproduct_agrees(
            BENZOIC_ACID_DEEP_SYSTEM,
            "gpu",
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_torch_matches_dense(mode: str, device: str) -> None:
    """Test whether SBD and TorchSCI agree.
    Args:
        device: Device
    """
    assert_torch_matches_dense(
        BENZOIC_ACID_SMALL_SYSTEM,
        device,
        mode,
    )
