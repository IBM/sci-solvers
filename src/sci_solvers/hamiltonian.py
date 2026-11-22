__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import os
from dataclasses import dataclass

import numpy as np
from pyscf import ao2mo
from pyscf.tools import fcidump


@dataclass
class HamiltonianData:
    """
    Hamiltonian data for energy calculations.

    Note: Chemists' notation is assumed, as expected by qiskit-addon-sqd and pyscf.
    If you route integrals through ffsim instead, ffsim uses physicists' notation apply `eri.transpose(0, 2, 3, 1)` (or its inverse) at that boundary.

    Attributes:
        hcore: One-electron integrals, shape (n_orbitals, n_orbitals).
        eri: Two-electron repulsion integrals, shape (n_orbitals, n_orbitals, n_orbitals, n_orbitals).
        e_nuc: Nuclear repulsion energy.
        num_orbitals: Number of spatial orbitals.
        num_elec_a: Number of alpha (spin-up) electrons.
        num_elec_b: Number of beta (spin-down) electrons.
        fcidump_path: Optional path to the source FCIDUMP file
    """

    hcore: np.ndarray
    eri: np.ndarray
    e_nuc: float
    num_orbitals: int
    num_elec_a: int
    num_elec_b: int
    fcidump_path: str | os.PathLike[str] | None = None

    def __post_init__(self):
        """Validate and normalize Hamiltonian data.

        Real integrals keep their historical normalization exactly: the ERI is
        run through :func:`pyscf.ao2mo.restore` (accepting full-4D, 4-fold, or
        8-fold packed input and expanding to full 4-index). Complex integrals
        (e.g. a single non-Gamma k-point or a spin-orbit-coupled Hamiltonian for
        a periodic system) are kept complex and are *not* passed through
        ``ao2mo.restore`` — that routine only supports real 8-fold-symmetric
        integrals and raises on complex input. A complex ERI carries only the
        Hermitian (4-fold) symmetry, so it must already be supplied as the full
        dense ``(n, n, n, n)`` tensor (which is what the periodic-Hamiltonian
        generator emits).
        """

        n = self.num_orbitals

        self.hcore = np.asarray(self.hcore)
        if self.hcore.shape != (n, n):
            raise ValueError(f"hcore shape {self.hcore.shape} doesn't match num_orbitals={n}")

        eri = np.asarray(self.eri)
        if np.iscomplexobj(self.hcore) or np.iscomplexobj(eri):
            # Complex Hermitian path: no ao2mo.restore (it rejects complex).
            # Require the full dense 4-index tensor, since packed storage assumes
            # real permutational symmetry that a complex ERI does not have.
            if eri.shape != (n, n, n, n):
                raise ValueError(
                    "complex ERI must be supplied as the full dense "
                    f"(n, n, n, n) tensor, got shape {eri.shape} for num_orbitals={n}. "
                    "Complex integrals only carry 4-fold (Hermitian) symmetry, so "
                    "packed storage is not supported."
                )
            self.eri = eri
        else:
            # Real path unchanged: accepts full 4D, 4-fold, or 8-fold packed ERI
            # and normalizes to full 4D (also raises if shape is inconsistent).
            self.eri = ao2mo.restore(1, eri, n)

        if self.num_elec_a < 0 or self.num_elec_b < 0:
            raise ValueError("Number of electrons must be non-negative")

    @property
    def is_complex(self) -> bool:
        """True if the integrals are complex (a complex Hermitian Hamiltonian)."""
        return np.iscomplexobj(self.hcore) or np.iscomplexobj(self.eri)

    @classmethod
    def from_fcidump(cls, filename: str | os.PathLike[str]) -> "HamiltonianData":
        """Load Hamiltonian from an FCIDUMP file."""

        data = fcidump.read(os.fspath(filename))
        nelec, ms2 = data["NELEC"], data["MS2"]
        return cls(
            hcore=data["H1"],
            eri=data["H2"],
            e_nuc=data["ECORE"],
            num_orbitals=data["NORB"],
            num_elec_a=(nelec + ms2) // 2,
            num_elec_b=(nelec - ms2) // 2,
            fcidump_path=filename,
        )

    def to_fcidump(self, filename: str | os.PathLike[str]) -> None:
        """Save Hamiltonian to an FCIDUMP file.

        Raises:
            NotImplementedError: If the integrals are complex. The FCIDUMP format
                stores only real integrals; a complex Hermitian Hamiltonian (e.g.
                a non-Gamma k-point) cannot be represented and would be silently
                truncated to its real part, so this is refused.
        """
        if self.is_complex:
            raise NotImplementedError(
                "Cannot write a complex Hamiltonian to FCIDUMP: the format stores "
                "only real integrals. Persist complex integrals as a complex128 "
                "array/HDF5 sidecar instead."
            )
        fcidump.from_integrals(
            os.fspath(filename),
            self.hcore,
            self.eri,
            self.num_orbitals,
            (self.num_elec_a, self.num_elec_b),
            nuc=self.e_nuc,
            ms=self.ms2,
        )

    @property
    def num_elec(self) -> int:
        """Total number of electrons."""
        return self.num_elec_a + self.num_elec_b

    @property
    def ms2(self) -> int:
        """2 * M_s (spin projection quantum number)."""
        return self.num_elec_a - self.num_elec_b

    @property
    def is_closed_shell(self) -> bool:
        """True if alpha and beta electron counts are equal."""
        return self.num_elec_a == self.num_elec_b
