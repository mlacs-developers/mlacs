"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np
from ase.atoms import Atoms


# ========================================================================== #
# ========================================================================== #
class MlipManager:
    """
    Parent Class for the management of Machine-Learning Interatomic Potential
    """
    def __init__(self,
                 descriptor,
                 nthrow=10,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0,
                 mbar=None,
                 no_zstress=False):

        self.descriptor = descriptor
        self.mbar = mbar

        self.ecoef = energy_coefficient
        self.fcoef = forces_coefficient
        self.scoef = stress_coefficient

        self.amat_e = None
        self.amat_f = None
        self.amat_s = None

        self.ymat_e = None
        self.ymat_f = None
        self.ymat_s = None

        self.no_zstress = no_zstress

        self.nthrow = nthrow
        if self.mbar is not None:
            self.nthrow = 0
        self.nconfs = 0

        # Some initialization for sampling interface
        self.pair_style = None
        self.pair_coeff = None
        self.model_post = None
        self.atom_style = "atomic"
        self.bonds = None
        self.angles = None
        self.bond_style = None
        self.bond_coeff = None
        self.angle_style = None
        self.angle_coeff = None

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if self.mbar is not None:
            self.mbar.update_database(atoms)
        amat_all = self.descriptor.calculate(atoms)
        energy = np.array([at.get_potential_energy() for at in atoms])
        forces = np.array([at.get_forces() for at in atoms]).flatten()
        stress = np.array([at.get_stress() for at in atoms]).flatten()
        nat = np.array([len(at) for at in atoms])

        for amat in amat_all:
            if self.amat_e is None:
                self.amat_e = amat["desc_e"]
                self.amat_f = amat["desc_f"]
                self.amat_s = amat["desc_s"]

            else:
                self.amat_e = np.r_[self.amat_e, amat["desc_e"]]
                self.amat_f = np.r_[self.amat_f, amat["desc_f"]]
                self.amat_s = np.r_[self.amat_s, amat["desc_s"]]

        if self.ymat_e is None:
            self.ymat_e = energy
            self.ymat_f = forces
            self.ymat_s = stress
            self.natoms = nat
        else:
            self.ymat_e = np.r_[self.ymat_e, energy]
            self.ymat_f = np.r_[self.ymat_f, forces]
            self.ymat_s = np.r_[self.ymat_s, stress]
            self.natoms = np.append(self.natoms, [nat])
        self.nconfs += len(atoms)

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
        idx_s = idx_e * 6
        return idx_e, idx_f, idx_s
