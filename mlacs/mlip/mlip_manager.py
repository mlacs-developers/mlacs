"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ..utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class MlipManager:
    """
    Parent Class for the management of Machine-Learning Interatomic Potential
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 nthrow=10,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=0.0,
                 kargs_mbar=None,
                 no_zstress=False):

        self.elements, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(atoms)
        self.rcut = rcut

        self.energy_coefficient = energy_coefficient
        self.forces_coefficient = forces_coefficient
        self.stress_coefficient = stress_coefficient

        self.mbar = None
        if kargs_mbar is not None:
            from . import MbarManager
            self.mbar = MbarManager.__init__(**kargs_mbar)

        self.no_zstress = no_zstress

        self.nthrow = nthrow
        self.nconfs = 0

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        natoms = len(atoms)
        descriptor, data = self.compute_fit_matrix(atoms)
        amat_e = descriptor[0] / natoms
        amat_f = descriptor[1:1+3*natoms]
        amat_s = descriptor[1+3*natoms:]
        ymat_e = data[0] / natoms
        ymat_f = data[1:1+3*natoms]
        ymat_s = data[1+3*natoms:]
        if self.nconfs == 0:
            self.amat_e = amat_e
            self.amat_f = amat_f
            self.amat_s = amat_s
            self.ymat_e = ymat_e
            self.ymat_f = ymat_f
            self.ymat_s = ymat_s
            self.natoms = np.array([natoms])
        else:
            self.amat_e = np.vstack((self.amat_e, amat_e))
            self.amat_f = np.r_[self.amat_f, amat_f]
            self.amat_s = np.r_[self.amat_s, amat_s]
            self.ymat_e = np.r_[self.ymat_e, ymat_e]
            self.ymat_f = np.r_[self.ymat_f, ymat_f]
            self.ymat_s = np.r_[self.ymat_s, ymat_s]
            self.natoms = np.append(self.natoms, natoms)

        self.nconfs += 1

# ========================================================================== #
    def train_mlip(self):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    def get_mlip_dict(self):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    def _get_idx_fit(self):
        """
        """
        if self.nconfs < self.nthrow:
            idx_e = idx_f = idx_s = 0
        elif self.nconfs >= self.nthrow and self.nconfs < 2 * self.nthrow:
            idx_e = self.nconfs - self.nthrow
        else:
            idx_e = self.nthrow
        idx_f = 3 * self.natoms[:idx_e].sum()
        if self.no_zstress:
            idx_s = idx_e * 3
        else:
            idx_s = idx_e * 6
        return idx_e, idx_f, idx_s
