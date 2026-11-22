"""SBD/TorchSCI equivalence checks on the small H4 full-CI system."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import pytest
from sbd_equivalence_utils import (
    DEVICES,
    H4_SYSTEM,
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
    assert_tpb_agrees("sbd", H4_SYSTEM, device)


@pytest.mark.parametrize("device", DEVICES)
def test_sbd_gdb_energy_matches_torch_on_full_product(device: str) -> None:
    """Test whether SBD and TorchSCI agree in GDB mode.
    Args:
        device: Device
    """
    assert_gdb_full_product_agrees(H4_SYSTEM, device)


@pytest.mark.parametrize("device", DEVICES)
def test_sbd_gdb_energy_matches_torch_on_nonproduct_subset(device: str) -> None:
    """Test whether SBD and TorchSCI agree in GDB mode.
    Args:
        device: Device
    """
    assert_gdb_nonproduct_agrees(H4_SYSTEM, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["tpb", "gdb"])
def test_torch_matches_dense(mode: str, device: str) -> None:
    """Test whether SBD and TorchSCI agree.
    Args:
        mode: mode
        device: Device
    """
    assert_torch_matches_dense(H4_SYSTEM, device, mode)
