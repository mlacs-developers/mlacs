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
        if self.nconfs < self.nthrow:
            idx = 0
        elif self.nconfs >= self.nthrow and self.nconfs < 2 * self.nthrow:
            idx = self.nconfs - self.nthrow
        else:
            idx = self.nthrow

        amatrix        = np.vstack((self.energy_coefficient * self.amatrix_energy[idx:], \
                                    self.forces_coefficient * self.amatrix_forces[idx*3*self.natoms:], \
                                    self.stress_coefficient * self.amatrix_stress[idx*6:]))
        ymatrix        = np.hstack((self.energy_coefficient * self.ymatrix_energy[idx:], \
                                    self.forces_coefficient * self.ymatrix_forces[idx*3*self.natoms:], \
                                    self.stress_coefficient * self.ymatrix_stress[idx*6:]))

        # Good ol' Ordinary Linear Least-Square fit
        self.coefficients = np.linalg.lstsq(amatrix, ymatrix, rcond=None)[0]

        self.write_mlip()
        self.init_calc()

        # Prepare some data to check accuracy of the fit
        e_true = self.ymatrix_energy[idx:]
        e_mlip = np.einsum('i,ki->k', self.coefficients, self.amatrix_energy[idx:])
        f_true = self.ymatrix_forces[idx*3*self.natoms:]
        f_mlip = np.einsum('i,ki->k', self.coefficients, self.amatrix_forces[idx*3*self.natoms:])
        s_true = self.ymatrix_stress[idx*6:] / GPa
        s_mlip = np.einsum('i,ki->k', self.coefficients, self.amatrix_stress[idx*6:]) / GPa

        # Compute RMSE and MAE
        rmse_energy = np.sqrt(np.mean((e_true - e_mlip)**2))
        mae_energy  = np.mean(np.abs(e_true - e_mlip))

        rmse_forces = np.sqrt(np.mean((f_true - f_mlip)**2))
        mae_forces  = np.mean(np.abs(f_true - f_mlip))

        rmse_stress = np.sqrt(np.mean((s_true - s_mlip)**2))
        mae_stress  = np.mean(np.abs(s_true - s_mlip))

        # Prepare message to the log
        msg  = "number of configurations for training:  {:}\n".format(len(self.amatrix_energy[idx:]))
        msg += "RMSE Energy    {:.4f} eV/at\n".format(rmse_energy)
        msg += "MAE Energy     {:.4f} eV/at\n".format(mae_energy)
        msg += "RMSE Forces    {:.4f} eV/angs\n".format(rmse_forces)
        msg += "MAE Forces     {:.4f} eV/angs\n".format(mae_forces)
        msg += "RMSE Stress    {:.4f} GPa\n".format(rmse_stress)
        msg += "MAE Stress     {:.4f} GPa\n".format(mae_stress)
        msg += "\n"

        header = "rmse: {:.5f} eV/at,    mae: {:.5f} eV/at\n".format(rmse_energy, mae_energy) + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat", np.vstack((e_true, e_mlip)).T, header=header)
        header = "rmse: {:.5f} eV/angs   mae: {:.5f} eV/angs\n".format(rmse_forces, mae_forces) + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat", np.vstack((f_true.flatten(), f_mlip.flatten())).T, header=header)
        header = "rmse: {:.5f} GPa       mae: {:.5f} GPa\n".format(rmse_stress, mae_stress) + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat", np.vstack((s_true.flatten(), s_mlip.flatten())).T, header=header)

        return msg


#===================================================================================================================================================#
    def get_mlip_dict(self):
        mlip_dict = self.lammps_interface.get_mlip_dict()
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        return mlip_dict
