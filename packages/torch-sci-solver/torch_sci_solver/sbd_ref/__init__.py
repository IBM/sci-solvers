"""SBD-aligned scalar reference implementation.

Pure Python, no torch. Slow by design; used as a regression target for
the eventual vectorized torch port.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

Derived from SBD (https://github.com/r-ccs-cms/sbd), licensed under the Apache
License 2.0: Copyright 2025 RIKEN and its contributors, and the other SBD
copyright holders listed in NOTICE. This file is a modified Python version of
the original C++ code.

SPDX-License-Identifier: Apache-2.0
"""


from torch_sci_solver.sbd_ref.davidson import (
    basis_init_vector,
    davidson,
)
from torch_sci_solver.sbd_ref.determinants import (
    bitcount,
    det_from_alpha_beta,
    difference,
    getocc,
    hij,
    one_excite,
    orbital_difference,
    parity,
    setocc,
    two_excite,
    zero_excite,
)
from torch_sci_solver.sbd_ref.helpers import (
    TaskHelper,
    generate_all_helpers,
    generate_excitation,
)
from torch_sci_solver.sbd_ref.integrals import (
    OneInt,
    TwoInt,
    setup_integrals_from_pyscf_h1_eri,
)
from torch_sci_solver.sbd_ref.mult import (
    make_diag_terms,
    mult,
)
from torch_sci_solver.sbd_ref.sbdiag import diag

__all__ = [
    "getocc",
    "setocc",
    "bitcount",
    "det_from_alpha_beta",
    "difference",
    "orbital_difference",
    "parity",
    "zero_excite",
    "one_excite",
    "two_excite",
    "hij",
    "OneInt",
    "TwoInt",
    "setup_integrals_from_pyscf_h1_eri",
    "TaskHelper",
    "generate_excitation",
    "generate_all_helpers",
    "make_diag_terms",
    "mult",
    "basis_init_vector",
    "davidson",
    "diag",
]
