"""Multi-GPU support utilities for torch_sci_solver.

Currently exposes ``run_in_spawned_process`` — the spawn-and-return helper used
by the multi-GPU benchmark and tests to isolate each GPU phase in its own
process (required under Exclusive_Process compute mode; see _subprocess.py).
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from torch_sci_solver.multi_gpu._subprocess import run_in_spawned_process

__all__ = ["run_in_spawned_process"]
