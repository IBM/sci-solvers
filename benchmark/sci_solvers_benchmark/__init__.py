"""Cross-solver benchmark harness for the ``sci_solvers`` eigensolvers.

Three stages: prepare the inputs (``prepare_quantum``, ``prepare_synthetic``), time the
solvers on them (``run_single``, ``run_matrix``, ``scripts/run_ladder.sh``), then
aggregate the per-run JSONs (``summarize``). See ``benchmark/README.md``.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""
