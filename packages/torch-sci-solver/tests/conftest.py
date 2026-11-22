"""Shared pytest fixtures for the torch-sci-solver tests."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _gloo_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin gloo to loopback so spawned ranks can reach each other on 127.0.0.1.

    By default gloo binds to the address the hostname resolves to; when that is a public
    IPv6 address (common on macOS) the ranks never connect and every collective times out.
    A value already set by the user is kept; monkeypatch restores the environment afterwards.
    """
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
