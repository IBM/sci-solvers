"""Solvers advertised by other packages through the ``sci_solvers.solvers`` entry-point group."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import importlib
from importlib.metadata import EntryPoint
from typing import Iterator

import pytest

from sci_solvers.energy_solvers import factory


class FakePluginSolver:
    """Stands in for a solver class that another package registers."""


def _entry_point(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=factory.SOLVER_ENTRY_POINT_GROUP)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap in a scratch copy of the registry, so a test cannot leak entries into the real one."""
    monkeypatch.setattr(factory, "_SOLVER_LOADERS", dict(factory._SOLVER_LOADERS))


def _advertise(monkeypatch: pytest.MonkeyPatch, *entry_points: EntryPoint) -> None:
    monkeypatch.setattr(factory, "entry_points", lambda group: entry_points)


@pytest.mark.usefixtures("registry")
def test_a_plugin_solver_is_resolved_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry point's class is returned by ``solver_class``, matched case-insensitively."""
    _advertise(monkeypatch, _entry_point("Fake_Plugin", f"{__name__}:FakePluginSolver"))
    factory._register_plugin_solvers()
    assert factory.solver_class("fake_plugin") is FakePluginSolver
    assert factory.solver_class("FAKE_PLUGIN") is FakePluginSolver


@pytest.mark.usefixtures("registry")
def test_a_plugin_cannot_replace_a_builtin_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin that reuses a built-in name is ignored."""
    builtin = factory.solver_class("pyscf")
    _advertise(monkeypatch, _entry_point("pyscf", f"{__name__}:FakePluginSolver"))
    factory._register_plugin_solvers()
    assert factory.solver_class("pyscf") is builtin


@pytest.mark.usefixtures("registry")
def test_a_plugin_that_fails_to_import_only_fails_when_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registering is lazy: a broken plugin does not break the factory or the other solvers."""
    _advertise(monkeypatch, _entry_point("broken_plugin", "no_such_package.module:Missing"))
    factory._register_plugin_solvers()
    assert factory.solver_class("pyscf") is not None
    with pytest.raises(ImportError):
        factory.solver_class("broken_plugin")


@pytest.fixture
def reloaded_factory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reload the factory with a plugin advertised, then restore it."""
    plugin = _entry_point("reload_plugin", f"{__name__}:FakePluginSolver")
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None, **kw: (plugin,))
    importlib.reload(factory)
    yield
    monkeypatch.undo()
    importlib.reload(factory)


@pytest.mark.usefixtures("reloaded_factory")
def test_a_plugin_is_listed_with_the_built_in_solvers() -> None:
    """``AVAILABLE_SOLVERS`` is derived after the plugins are added, and stays sorted."""
    assert "reload_plugin" in factory.AVAILABLE_SOLVERS
    assert {"pyscf", "sbd", "torch_sci"} <= set(factory.AVAILABLE_SOLVERS)
    assert factory.AVAILABLE_SOLVERS == tuple(sorted(factory.AVAILABLE_SOLVERS))
