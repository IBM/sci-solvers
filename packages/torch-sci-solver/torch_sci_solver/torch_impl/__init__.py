"""Vectorized PyTorch port of torch_sci_solver.sbd_ref."""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


from torch_sci_solver.torch_impl.davidson import (
    basis_init_vector_torch,
    davidson_torch,
)
from torch_sci_solver.torch_impl.determinants import (
    det_from_alpha_beta_batch,
    getocc_batch,
    one_excite_batch,
    parity_batch,
    two_excite_batch,
    zero_excite_batch,
)
from torch_sci_solver.torch_impl.helpers import (
    TorchHelper,
    generate_all_torch_helpers,
    torch_helper_from_scalar,
)
from torch_sci_solver.torch_impl.integrals import (
    TorchIntegrals,
    to_torch_integrals,
)
from torch_sci_solver.torch_impl.mult import (
    make_diag_terms_torch,
    mult_torch,
)
from torch_sci_solver.torch_impl.rdm import (
    energy_from_rdm,
    make_one_rdm_diag_torch,
    make_one_rdm_torch,
    make_two_rdm_aa_torch,
    make_two_rdm_ab_torch,
    make_two_rdm_bb_torch,
    make_two_rdm_torch,
    s_squared_from_rdm,
)
from torch_sci_solver.torch_impl.sbdiag import (
    diag_torch,
)
from torch_sci_solver.torch_impl.sqd_loop import (
    sqd_loop,
)

__all__ = [
    "getocc_batch",
    "parity_batch",
    "det_from_alpha_beta_batch",
    "zero_excite_batch",
    "one_excite_batch",
    "two_excite_batch",
    "TorchIntegrals",
    "to_torch_integrals",
    "TorchHelper",
    "torch_helper_from_scalar",
    "generate_all_torch_helpers",
    "make_diag_terms_torch",
    "mult_torch",
    "basis_init_vector_torch",
    "davidson_torch",
    "diag_torch",
    "make_one_rdm_diag_torch",
    "make_one_rdm_torch",
    "sqd_loop",
    "energy_from_rdm",
    "make_two_rdm_aa_torch",
    "make_two_rdm_ab_torch",
    "make_two_rdm_bb_torch",
    "make_two_rdm_torch",
    "s_squared_from_rdm",
]
