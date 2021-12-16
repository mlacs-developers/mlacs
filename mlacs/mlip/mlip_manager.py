"""
"""
import numpy as np

from ase.units import GPa

from mlacs.utilities import get_elements_Z_and_masses


#===================================================================================================================================================#
#===================================================================================================================================================#
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
                ):

        self.elements, self.Z, self.masses = get_elements_Z_and_masses(atoms)
        self.natoms = len(atoms)
        self.rcut   = rcut

        self.energy_coefficient = energy_coefficient
        self.forces_coefficient = forces_coefficient
        self.stress_coefficient = stress_coefficient

        self.amatrix_energy = np.array([])
        self.amatrix_forces = np.array([])
        self.amatrix_stress = np.array([])
        self.ymatrix_energy = np.array([])
        self.ymatrix_forces = np.array([])
        self.ymatrix_stress = np.array([])

        self.nthrow = nthrow
        self.nconfs = 0


#===================================================================================================================================================#
    def update_matrices(self, atoms):
        """
        """
        natoms = len(atoms)
        descriptor, data = self.compute_fit_matrix(atoms)
        if len(self.amatrix_energy) == 0:
            self.amatrix_energy = descriptor[0] / natoms
            self.amatrix_forces = descriptor[1:1+3*natoms]
            self.amatrix_stress = descriptor[1+3*natoms:]
            self.ymatrix_energy = data[0] / natoms
            self.ymatrix_forces = data[1:1+3*natoms]
            self.ymatrix_stress = data[1+3*natoms:]
        else:
            self.amatrix_energy = np.vstack((self.amatrix_energy, descriptor[0] / natoms))
            self.amatrix_forces = np.vstack((self.amatrix_forces, descriptor[1:1+3*natoms]))
            self.amatrix_stress = np.vstack((self.amatrix_stress, descriptor[1+3*natoms:]))
            self.ymatrix_energy = np.hstack((self.ymatrix_energy, data[0] / natoms))
            self.ymatrix_forces = np.hstack((self.ymatrix_forces, data[1:1+3*natoms]))
            self.ymatrix_stress = np.hstack((self.ymatrix_stress, data[1+3*natoms:]))

        self.nconfs += 1


#===================================================================================================================================================#
    def train_mlip(self):
        """
        """
        import NotImplementedError


#===================================================================================================================================================#
    def get_mlip_dict(self):
        """
        """
        import NotImplementedError


#===================================================================================================================================================#
    def _get_idx_fit(self):
        if self.nconfs < self.nthrow:
            idx = 0
        elif self.nconfs >= self.nthrow and self.nconfs < 2 * self.nthrow:
            idx = self.nconfs - self.nthrow
        else:
            idx = self.nthrow
        return idx
