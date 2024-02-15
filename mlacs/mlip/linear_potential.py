'''
'''
from pathlib import Path
import numpy as np
from ase.units import GPa

from . import MlipManager
from ..utilities import compute_correlation


default_parameters = {"method": "ols",
                      "lambda_ridge": 1e-8,
                      "hyperparameters": {},
                      "gridcv": {}}


# ========================================================================== #
# ========================================================================== #
class LinearPotential(MlipManager):
    """
    Potential that assume a linear relation between the descriptor and the
    energy.

    Parameters
    ----------
    descriptor: :class:`Descriptor`
        The descriptor used in the model.

    nthrow: :class: int
        Number of first configurations to ignore when doing the fit

    energy_coefficient: :class:`float`
        Weight of the energy in the fit
        Default 1.0

    forces_coefficient: :class:`float`
        Weight of the forces in the fit
        Default 1.0

    stress_coefficient: :class:`float`
        Weight of the stress in the fit
        Default 1.0

    weight: :class:`WeightingPolicy`
        Weight used for the fitting and calculation of properties.
        Default :class:`None`
    """
    def __init__(self,
                 descriptor,
                 nthrow=0,
                 parameters={},
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0,
                 weight=None,
                 folder=Path("MLIP")):
        MlipManager.__init__(self,
                             descriptor,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient,
                             weight,
                             folder)

        self.parameters = default_parameters
        self.parameters.update(parameters)

        self.coefficients = None

        if self.parameters["method"] != "ols":
            if self.parameters["hyperparameters"] is None:
                hyperparam = {}
            else:
                hyperparam = self.parameters["hyperparameters"]
            hyperparam["fit_intercept"] = False
            self.parameters["hyperparameters"] = hyperparam

# ========================================================================== #
    def train_mlip(self, mlip_subfolder):
        """
        """
        if mlip_subfolder is None:
            mlip_subfolder = self.folder
        else:
            mlip_subfolder = self.folder / mlip_subfolder

        msg = ''
        idx_e, idx_f, idx_s = self._get_idx_fit()
        amat_e = self.amat_e[idx_e:] / self.natoms[idx_e:, None]
        amat_f = self.amat_f[idx_f:]
        amat_s = self.amat_s[idx_s:]
        ymat_e = self.ymat_e[idx_e:] / self.natoms[idx_e:]
        ymat_f = self.ymat_f[idx_f:]
        ymat_s = self.ymat_s[idx_s:]

        ecoef = self.ecoef / amat_e.std() / len(amat_e)
        fcoef = self.fcoef / amat_f.std() / len(amat_f)
        scoef = self.scoef / amat_s.std() / len(amat_s)

        amat = np.r_[amat_e * ecoef,
                     amat_f * fcoef,
                     amat_s * scoef]
        ymat = np.r_[ymat_e * ecoef,
                     ymat_f * fcoef,
                     ymat_s * scoef]

        if self.weight.train_mlip:
            W = self.weight.get_weights()
            amat = amat * W[:, np.newaxis]
            ymat = ymat * W

        if self.parameters["method"] == "ols":
            self.coefficients = np.linalg.lstsq(amat,
                                                ymat,
                                                None)[0]
        elif self.parameters["method"] == "ridge":
            lamb = self.parameters["lambda_ridge"]
            gamma = self.descriptor._regularization_matrix()
            ymat = amat.T @ ymat
            amat = amat.T @ amat + gamma * lamb
            self.coefficients = np.linalg.lstsq(amat,
                                                ymat,
                                                None)[0]

        msg += "\nNumber of configurations for training: " + \
               f"{len(self.natoms[idx_e:]):}\n"
        msg += "Number of atomic environments for training: " + \
               f"{self.natoms[idx_e:].sum():}\n\n"

        msg += self.weight.compute_weight(amat_e,
                                          self.coefficients,
                                          self.get_mlip_energy,
                                          subfolder=self.folder)

        msg += self.compute_tests(amat_e, amat_f, amat_s,
                                  ymat_e, ymat_f, ymat_s)

        self.descriptor.write_mlip(self.coefficients, subfolder=mlip_subfolder)
        return msg

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s):
        """
        Computed the weighted RMSE and MAE.
        """
        e_mlip = np.einsum('ij,j->i', amat_e, self.coefficients)
        f_mlip = np.einsum('ij,j->i', amat_f, self.coefficients)
        s_mlip = np.einsum('ij,j->i', amat_s, self.coefficients)

        w = self.weight.init_weight()
        we, wf, ws = self.weight.build_W_efs(w)

        rmse_e = np.sqrt(np.mean(we * (ymat_e - e_mlip)**2))
        mae_e = np.mean(we * np.abs(ymat_e - e_mlip))

        rmse_f = np.sqrt(np.mean(wf * (ymat_f - f_mlip)**2))
        mae_f = np.mean(wf * np.abs(ymat_f - f_mlip))

        rmse_s = np.sqrt(np.mean(ws * ((ymat_s - s_mlip) / GPa)**2))
        mae_s = np.mean(ws * np.abs((ymat_s - s_mlip) / GPa))

        # Prepare message to the log
        msg = f"Weighted RMSE Energy    {rmse_e:.4f} eV/at\n"
        msg += f"Weighted MAE Energy     {mae_e:.4f} eV/at\n"
        msg += f"Weighted RMSE Forces    {rmse_f:.4f} eV/angs\n"
        msg += f"Weighted MAE Forces     {mae_f:.4f} eV/angs\n"
        msg += f"Weighted RMSE Stress    {rmse_s:.4f} GPa\n"
        msg += f"Weighted MAE Stress     {mae_s:.4f} GPa\n"
        msg += "\n"

        header = f"Weighted rmse: {rmse_e:.5f} eV/at,    " + \
                 f"Weighted mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.c_[ymat_e, e_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {rmse_f:.5f} eV/angs   " + \
                 f"Weighted mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.c_[ymat_f, f_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {rmse_s:.5f} GPa       " + \
                 f"Weighted mae: {mae_s:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.c_[ymat_s, s_mlip] / GPa,
                   header=header, fmt="%25.20f  %25.20f")
        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        from .calculator import MlipCalculator
        calc = MlipCalculator(self)
        return calc

# ========================================================================== #
    def predict(self, atoms):
        """
        """
        assert self.coefficients is not None, 'The model has not been trained'

        res = self.descriptor.calculate(atoms, subfolder=self.folder)[0]

        # We use the latest value coefficients to get the properties
        energy = np.einsum('ij,j->', res['desc_e'], self.coefficients)
        forces = np.einsum('ij,j->i', res['desc_f'], self.coefficients)
        stress = np.einsum('ij,j->i', res['desc_s'], self.coefficients)

        forces = forces.reshape(len(atoms), 3)

        return energy, forces, stress

# ========================================================================== #
    def get_mlip_energy(self, coef, desc):
        """
        """
        return np.einsum('ij,j->i', desc, coef)

# ========================================================================== #
    def set_coefficients(self, coefficients):
        """
        """
        if coefficients is not None:
            assert len(coefficients) == self.descriptor.ncolumns
        self.coefficients = coefficients

# ========================================================================== #
    def __str__(self):
        txt = f"Linear potential with {str(self.descriptor)}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Linear potential\n"
        txt += "Parameters:\n"
        txt += "-----------\n"
        txt += f"energy coefficient :    {self.ecoef}\n"
        txt += f"forces coefficient :    {self.fcoef}\n"
        txt += f"stress coefficient :    {self.scoef}\n"
        txt += f"Fit method :            {self.parameters['method']}\n"
        txt += "\n"
        txt += "Descriptor used in the potential:\n"
        txt += repr(self.descriptor)
        return txt
