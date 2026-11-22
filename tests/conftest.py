"""Shared pytest setup for sci-solvers' tests.

numba starts one worker thread per CPU it sees by default. On large nodes that is
well over a hundred threads for the small problems these tests solve, and the
truncated-Davidson tests then run far slower than with a few threads. Cap it at 8
(or the CPU count, if lower) before numba is imported, unless the caller has set
NUMBA_NUM_THREADS.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os

# numba's own default (numba.core.config): the CPUs this process may run on.
if hasattr(os, "sched_getaffinity"):
    _cpus = len(os.sched_getaffinity(0))
else:
    _cpus = os.cpu_count() or 1
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(8, max(1, _cpus))))
