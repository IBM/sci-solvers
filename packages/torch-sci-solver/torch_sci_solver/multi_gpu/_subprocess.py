"""Run a callable in a fresh spawned subprocess and return its result.

Shared by the multi-GPU benchmark and the multi-GPU tests. The reason both need
it: on clusters where GPUs are in ``Exclusive_Process`` compute mode, only one
process may hold a CUDA context on a given GPU. A single-GPU solve initializes a
context on cuda:0; if that ran in the same process that later spawns the
multi-GPU worker pool, the worker for cuda:0 could not acquire the device. By
running each GPU phase in a child that fully exits, its context is released
before the next phase starts, and the orchestrating (parent) process stays
CUDA-free.

``run_in_spawned_process(fn, *args)`` runs ``fn(*args)`` in a spawned child and
returns its value; if the child raised, a same-type exception is re-raised in
the parent (so ``pytest.raises(match=...)`` still matches), with the child
traceback appended. ``fn`` and ``args`` must be picklable (spawn), and ``fn``
must be importable (module-level, not a closure/lambda).
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


import multiprocessing
import traceback
from typing import Any, Callable

# One spawn context for the whole package — fork + an initialized CUDA context
# deadlocks, so every subprocess here must use spawn.
_MP_CTX = multiprocessing.get_context("spawn")

# Exception types we can faithfully reconstruct in the parent (builtins whose
# constructor takes the message string). Anything else is surfaced as a
# RuntimeError carrying the original type name + message.
_REBUILDABLE = {"ValueError": ValueError, "TypeError": TypeError, "RuntimeError": RuntimeError}


def _entry(fn: Callable, args: tuple, q) -> None:
    """Child entry: run ``fn(*args)``, ship back the result or the exception."""
    try:
        q.put(("ok", fn(*args)))
    except BaseException as e:  # noqa: BLE001 — marshal everything back to the parent
        q.put(("err", type(e).__name__, str(e), traceback.format_exc()))


def run_in_spawned_process(fn: Callable, *args: Any) -> Any:
    """Run ``fn(*args)`` in a fresh spawned process and return its result.

    Re-raises a matching exception in the parent if the child raised.
    """
    q = _MP_CTX.Queue()
    p = _MP_CTX.Process(target=_entry, args=(fn, args, q))
    p.start()
    payload = q.get()
    p.join()
    if payload[0] == "ok":
        return payload[1]
    _, exc_type, msg, tb = payload
    exc_cls = _REBUILDABLE.get(exc_type, RuntimeError)
    raise exc_cls(f"{msg}\n[subprocess traceback]\n{tb}")
