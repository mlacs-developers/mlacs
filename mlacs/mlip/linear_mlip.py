'''
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
'''
import numpy as np
from ase.units import GPa

from mlacs.mlip import MlipManager


# ========================================================================== #
# ========================================================================== #
class LinearMlip(MlipManager):
    """
    Parent Class for linear MLIP
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 nthrow=10,
                 regularization=None,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0,
                 rescale_energy=True,
                 rescale_forces=True,
                 rescale_stress=True):
        MlipManager.__init__(self,
                             atoms,
                             rcut,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient)

        self.regularization = regularization
        self.rescale_energy = rescale_energy
        self.rescale_forces = rescale_forces
        self.rescale_stress = rescale_stress

# ========================================================================== #
    def train_mlip(self):
        """
        """
        idx = self._get_idx_fit()

        sigma_e = 1.0
        if self.rescale_energy:
            sigma_e = np.std(self.amatrix_energy[idx:])
        sigma_f = 1.0
        if self.rescale_forces:
            sigma_e = np.std(self.amatrix_forces[idx*3*self.natoms:])
        sigma_s = 1.0
        if self.rescale_stress:
            sigma_e = np.std(self.amatrix_stress[idx*6:])

        ecoef = self.energy_coefficient / sigma_e / \
            len(self.amatrix_energy[idx:])
        fcoef = self.forces_coefficient / sigma_f / \
            len(self.amatrix_forces[idx*3*self.natoms:])
        scoef = self.stress_coefficient / sigma_s / \
            len(self.amatrix_stress[idx*6:])

        amatrix = np.vstack((ecoef * self.amatrix_energy[idx:],
                             fcoef * self.amatrix_forces[idx*3*self.natoms:],
                             scoef * self.amatrix_stress[idx*6:]))
        ymatrix = np.hstack((ecoef * self.ymatrix_energy[idx:],
                             fcoef * self.ymatrix_forces[idx*3*self.natoms:],
                             scoef * self.ymatrix_stress[idx*6:]))

        # TODO function to test several lambda values of regularization
        if self.regularization is not None:
            lamb = self.regularization
            regul = self.get_regularization_vector(lamb)
            regul = self.regularization * np.diag(regul)

            ymatrix = amatrix.T.dot(ymatrix)
            amatrix = amatrix.T.dot(amatrix) + regul

        # Good ol' Ordinary Linear Least-Square fit
        self.coefficients = np.linalg.lstsq(amatrix, ymatrix, rcond=None)[0]

        msg = self.compute_tests(idx)
        self.write_mlip()
        self.init_calc()
        return msg

# ========================================================================== #
    def get_mlip_dict(self):
        mlip_dict = {}
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        return mlip_dict

# ========================================================================== #
    def compute_tests(self, idx):
        # Prepare some data to check accuracy of the fit
        e_true = self.ymatrix_energy[idx:]
        e_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amatrix_energy[idx:])
        f_true = self.ymatrix_forces[idx*3*self.natoms:]
        f_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amatrix_forces[idx*3*self.natoms:])
        s_true = self.ymatrix_stress[idx*6:] / GPa
        s_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amatrix_stress[idx*6:]) / GPa

        # Compute RMSE and MAE
        rmse_energy = np.sqrt(np.mean((e_true - e_mlip)**2))
        mae_energy = np.mean(np.abs(e_true - e_mlip))

        rmse_forces = np.sqrt(np.mean((f_true - f_mlip)**2))
        mae_forces = np.mean(np.abs(f_true - f_mlip))

        rmse_stress = np.sqrt(np.mean((s_true - s_mlip)**2))
        mae_stress = np.mean(np.abs(s_true - s_mlip))

        # Prepare message to the log
        msg = "number of configurations for training:  " + \
              f"{len(self.amatrix_energy[idx:])}\n"
        msg += "RMSE Energy    {:.4f} eV/at\n".format(rmse_energy)
        msg += "MAE Energy     {:.4f} eV/at\n".format(mae_energy)
        msg += "RMSE Forces    {:.4f} eV/angs\n".format(rmse_forces)
        msg += "MAE Forces     {:.4f} eV/angs\n".format(mae_forces)
        msg += "RMSE Stress    {:.4f} GPa\n".format(rmse_stress)
        msg += "MAE Stress     {:.4f} GPa\n".format(mae_stress)
        msg += "\n"

        header = f"rmse: {rmse_energy:.5f} eV/at,    " + \
                 f"mae: {mae_energy:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.vstack((e_true, e_mlip)).T,
                   header=header)
        header = f"rmse: {rmse_forces:.5f} eV/angs   " + \
                 f"mae: {mae_forces:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header)
        header = f"rmse: {rmse_stress:.5f} GPa       " + \
                 f"mae: {mae_stress:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.vstack((s_true.flatten(), s_mlip.flatten())).T,
                   header=header)
        return msg
