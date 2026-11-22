"""Selectable matvec backends for Davidson."""

from __future__ import annotations

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""


MATVEC_IMPLS = ("batched", "triton")


def build_matvec(
    impl, hii, adets_t, bdets_t, n_orb, integrals, helpers, *, chunk_size, ab_factors=None
):
    """Return a ``matvec(v) -> H @ v`` closure for the requested backend.

    The per-subspace, ``T``-independent setup (the β-pass terms, and for
    Triton the same-spin ``eij``) is built once here and closed over, so it is
    not recomputed on every matvec.

    ``ab_factors`` is an optional ``(factors, signs)`` pair from
    :func:`~torch_sci_solver.torch_impl.ab_factorized.factorize_eri`: the αβ
    doubles pass (the ``task_type == 0`` helper) is then evaluated in factorized
    form by :class:`~torch_sci_solver.torch_impl.ab_factorized.FactorizedAlphaBeta`
    and the backend runs the remaining passes, with the same result.
    """
    helpers = tuple(helpers)
    if ab_factors is not None:
        from torch_sci_solver.torch_impl.ab_factorized import FactorizedAlphaBeta

        ab_helpers = [h for h in helpers if h.task_type == 0]
        ab_passes = [
            FactorizedAlphaBeta(
                adets_t,
                bdets_t,
                n_orb,
                h,
                *ab_factors,
                dtype=integrals.I2_eri.dtype,
                chunk_size=chunk_size,
            )
            for h in ab_helpers
        ]
        base = build_matvec(
            impl,
            hii,
            adets_t,
            bdets_t,
            n_orb,
            integrals,
            [h for h in helpers if h.task_type != 0],
            chunk_size=chunk_size,
        )

        def _factorized_matvec(v):
            """Backend passes plus the factorized αβ doubles."""
            out = base(v)
            for ab_pass in ab_passes:
                ab_pass(out, v)
            return out

        return _factorized_matvec
    if impl == "batched":
        from torch_sci_solver.torch_impl.mult import build_beta_pass_cache, mult_torch

        beta_cache = build_beta_pass_cache(adets_t, bdets_t, n_orb, integrals, helpers)

        def _matvec(v):
            """Matvec."""
            return mult_torch(
                hii,
                v,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                helpers,
                chunk_size=chunk_size,
                cache=beta_cache,
            )

        return _matvec

    if impl == "triton":
        from torch_sci_solver.torch_impl.mult_triton import build_samespin_cache, mult_triton

        cache = build_samespin_cache(adets_t, bdets_t, n_orb, integrals, helpers)

        def _matvec(v):
            """Matvec."""
            return mult_triton(
                hii,
                v,
                adets_t,
                bdets_t,
                n_orb,
                integrals,
                helpers,
                chunk_size=chunk_size,
                cache=cache,
            )

        return _matvec

    raise ValueError(f"unknown matvec_impl {impl!r}; expected one of {MATVEC_IMPLS}")
