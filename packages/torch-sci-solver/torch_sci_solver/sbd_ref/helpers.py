"""Helper construction for the SBD matvec algorithm.

Faithful port of sbd/include/sbd/chemistry/tpb/helper.h::GenerateExcitation
(lines 185-323), simplified for single-task / no-MPI use.

The "helpers" precompute, for each bra α-string (and bra β-string),
which ket strings are reachable by single or double excitation, and
what creation/annihilation orbital indices change. SBD splits this work
into "tasks" indexed by ``taskType``:

    taskType == 0  →  αβ-doubles  (needs single-α and single-β tables)
    taskType == 1  →  β only      (needs single-β and double-β tables)
    taskType == 2  →  α only      (needs single-α and double-α tables)

We expose:
    TaskHelper          — dataclass holding the per-task tables
    generate_excitation(adets, bdets, n_orb, task_type) → TaskHelper

The dataclass stores parallel arrays (one entry per bra string):
    singles_from_alpha[ia]:    list of ket α-indices reachable
    singles_alpha_cran[ia]:    list of (cr_spinorb, an_spinorb) pairs, one
                               per ket in singles_from_alpha[ia], in the
                               same order
    doubles_from_alpha[ia]:    list of ket α-indices reachable by double
    doubles_alpha_cran[ia]:    list of (cr0_so, cr1_so, an0_so, an1_so) tuples

α-spin uses spin-orbital index = ``2 * spatial``;
β-spin uses spin-orbital index = ``2 * spatial + 1``.
This matches SBD's helper code (lines 230-232, 287-288, etc.) exactly.

Storage format note: SBD packs cr/an pairs into a flat array
``SinglesAlphaCrAn[ia] = [cr0_for_ja0, an0_for_ja0, cr0_for_ja1, an0_for_ja1, ...]``.
We use tuples per ket entry for clarity; the content is identical and
we'll flatten later when vectorizing for GPU. Doubles use a flat 4-tuple
``(cr0, cr1, an0, an1)`` per entry; SBD stores
``[cr0, cr1, an0, an1, cr0, cr1, an0, an1, ...]`` flat.
"""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

Derived from SBD (https://github.com/r-ccs-cms/sbd), licensed under the Apache
License 2.0: Copyright 2025 RIKEN and its contributors, and the other SBD
copyright holders listed in NOTICE. This file is a modified Python version of
the original C++ code.

SPDX-License-Identifier: Apache-2.0
"""


from dataclasses import dataclass, field
from typing import List, Tuple

from torch_sci_solver.sbd_ref.determinants import difference, orbital_difference

# ============================================================================
# Data structure
# ============================================================================


@dataclass
class TaskHelper:
    """Per-task excitation tables. See module docstring.

    Empty lists for excitation types not relevant to this task_type.
    """

    task_type: int  # 0, 1, or 2
    n_alpha: int  # number of α-strings
    n_beta: int  # number of β-strings

    # α tables — populated for task_type in {0, 2}
    singles_from_alpha: List[List[int]] = field(default_factory=list)
    singles_alpha_cran: List[List[Tuple[int, int]]] = field(default_factory=list)
    doubles_from_alpha: List[List[int]] = field(default_factory=list)
    doubles_alpha_cran: List[List[Tuple[int, int, int, int]]] = field(default_factory=list)

    # β tables — populated for task_type in {0, 1}
    singles_from_beta: List[List[int]] = field(default_factory=list)
    singles_beta_cran: List[List[Tuple[int, int]]] = field(default_factory=list)
    doubles_from_beta: List[List[int]] = field(default_factory=list)
    doubles_beta_cran: List[List[Tuple[int, int, int, int]]] = field(default_factory=list)


# ============================================================================
# Excitation table construction  (SBD: helper.h GenerateExcitation lines 185-323)
# ============================================================================


def generate_excitation(
    adets: List[int], bdets: List[int], n_orb: int, task_type: int
) -> TaskHelper:
    """Build excitation tables for one task type.

    Args:
        adets:     sorted α-bitmasks, each over n_orb bits.
        bdets:     sorted β-bitmasks, each over n_orb bits.
        n_orb:     number of spatial orbitals L.
        task_type: 0 (αβ-doubles), 1 (β-only), or 2 (α-only).
                   See module docstring.

    Returns:
        Fully populated TaskHelper. Tables not used by `task_type` are
        left empty.
    """
    if task_type not in (0, 1, 2):
        raise ValueError(f"task_type must be 0, 1, or 2, got {task_type}")

    h = TaskHelper(task_type=task_type, n_alpha=len(adets), n_beta=len(bdets))

    n_a = len(adets)
    n_b = len(bdets)

    # --- α singles + α doubles  (taskType != 1 → 0 or 2) ---
    if task_type != 1:
        h.singles_from_alpha = [[] for _ in range(n_a)]
        h.singles_alpha_cran = [[] for _ in range(n_a)]
        # doubles only for task 2 (α-only)
        if task_type == 2:
            h.doubles_from_alpha = [[] for _ in range(n_a)]
            h.doubles_alpha_cran = [[] for _ in range(n_a)]

        for ia in range(n_a):
            for ja in range(n_a):
                d = difference(adets[ia], adets[ja], n_orb)
                if d == 2:
                    cr, an = orbital_difference(adets[ia], adets[ja], n_orb)
                    # SBD: SinglesAlphaCrAn.push_back(2*cr[0]); push_back(2*an[0])
                    h.singles_from_alpha[ia].append(ja)
                    h.singles_alpha_cran[ia].append((2 * cr[0], 2 * an[0]))
                elif d == 4 and task_type == 2:
                    cr, an = orbital_difference(adets[ia], adets[ja], n_orb)
                    # SBD: DoublesAlphaCrAn.push_back(2*cr[0]); push_back(2*cr[1]);
                    #      push_back(2*an[0]); push_back(2*an[1])
                    h.doubles_from_alpha[ia].append(ja)
                    h.doubles_alpha_cran[ia].append((2 * cr[0], 2 * cr[1], 2 * an[0], 2 * an[1]))

    # --- β singles + β doubles  (taskType != 2 → 0 or 1) ---
    if task_type != 2:
        h.singles_from_beta = [[] for _ in range(n_b)]
        h.singles_beta_cran = [[] for _ in range(n_b)]
        if task_type == 1:
            h.doubles_from_beta = [[] for _ in range(n_b)]
            h.doubles_beta_cran = [[] for _ in range(n_b)]

        for ib in range(n_b):
            for jb in range(n_b):
                d = difference(bdets[ib], bdets[jb], n_orb)
                if d == 2:
                    cr, an = orbital_difference(bdets[ib], bdets[jb], n_orb)
                    # SBD: SinglesBetaCrAn.push_back(2*cr[0]+1); push_back(2*an[0]+1)
                    h.singles_from_beta[ib].append(jb)
                    h.singles_beta_cran[ib].append((2 * cr[0] + 1, 2 * an[0] + 1))
                elif d == 4 and task_type == 1:
                    cr, an = orbital_difference(bdets[ib], bdets[jb], n_orb)
                    h.doubles_from_beta[ib].append(jb)
                    h.doubles_beta_cran[ib].append((
                        2 * cr[0] + 1,
                        2 * cr[1] + 1,
                        2 * an[0] + 1,
                        2 * an[1] + 1,
                    ))

    return h


def generate_all_helpers(
    adets: List[int], bdets: List[int], n_orb: int
) -> Tuple[TaskHelper, TaskHelper, TaskHelper]:
    """Convenience: build all three task helpers for a serial (no-MPI) run.

    SBD: MakeHelpers in helper.h:706+ schedules these three tasks when
    adet_comm_size == bdet_comm_size == 1.

    Returns:
        (helper_task_2, helper_task_1, helper_task_0)
        Order matches SBD's schedule (alpha-only first, beta-only second,
        αβ third) — preserved so any future debugging against SBD output
        matches step-by-step.
    """
    return (
        generate_excitation(adets, bdets, n_orb, task_type=2),
        generate_excitation(adets, bdets, n_orb, task_type=1),
        generate_excitation(adets, bdets, n_orb, task_type=0),
    )
