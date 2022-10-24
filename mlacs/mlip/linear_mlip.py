'''
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
'''
import numpy as np
from ase.units import GPa

from . import MlipManager
try:
    import sklearn.linear_model as lin_mod
    from sklearn.model_selection import GridSearchCV
except ImportError:
    lin_mod = None


default_parameters = {"method": "ols",
                      "hyperparameters": None,
                      "gridcv": None}


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
                 parameters=None,
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

        self._initialize_parameters(parameters)
        self.rescale_energy = rescale_energy
        self.rescale_forces = rescale_forces
        self.rescale_stress = rescale_stress

# ========================================================================== #
    def train_mlip(self):
        """
        """
        idx_e, idx_f, idx_s = self._get_idx_fit()

        sigma_e = 1.0
        if self.rescale_energy:
            sigma_e = np.std(self.amat_e[idx_e:])
        sigma_f = 1.0
        if self.rescale_forces:
            sigma_f = np.std(self.amat_f[idx_f:])
        sigma_s = 1.0
        if self.rescale_stress:
            sigma_s = np.std(self.amat_s[idx_s:])

        ecoef = self.energy_coefficient / sigma_e / \
            len(self.ymat_e[idx_e:])
        fcoef = self.forces_coefficient / sigma_f / \
            len(self.ymat_f[idx_f:])
        scoef = self.stress_coefficient / sigma_s / \
            len(self.ymat_s[idx_s:])

        amat = np.r_[self.amat_e[idx_e:] * ecoef,
                     self.amat_f[idx_f:] * fcoef,
                     self.amat_s[idx_s:] * scoef]
        ymat = np.r_[self.ymat_e[idx_e:] * ecoef,
                     self.ymat_f[idx_f:] * fcoef,
                     self.ymat_s[idx_s:] * scoef]

        msg = "number of configurations for training: " + \
              f"{len(self.natoms[idx_e:]):}\n"
        msg += "number of atomic environments for training: " + \
               f"{self.natoms[idx_e:].sum():}\n"

        if self.parameters["method"] == "ols":
            self.coefficients = np.linalg.lstsq(amat,
                                                ymat,
                                                rcond=None)[0]
        else:
            if lin_mod is None:
                msg = "You need sklearn installed to use other method " + \
                      "than Ordinary Least Squares"
                raise ModuleNotFoundError(msg)

            nelem = self._get_nelem()
            intercept_col = self.amat_e[idx_e:, :nelem].mean(axis=0)

            fitmethod = getattr(lin_mod, self.parameters["method"])
            fitlin = fitmethod(**self.parameters["hyperparameters"])
            if self.parameters["gridcv"] is None:
                fitlin.fit(amat, ymat)
            else:
                fitgcv = GridSearchCV(fitlin,
                                      self.parameters["gridcv"], verbose=0)
                fitgcv.fit(amat, ymat)
                fitlin = fitgcv.best_estimator_

                msg += "Hyperparameters found by Grid Seach Cross Validation\n"
                for key in fitgcv.best_params_.keys():
                    msg += f"    {key} :    {fitgcv.best_params_[key]}\n"

            # With sklearn, we need to update the intercept as it could
            # have been modified with the regularization method
            # If LinearRegression is used, this method recovers the
            # usual Ordinary Least Square solution as with np.linalg.lstsq
            mean_e = self.ymat_e[idx_e:].mean()
            intercept = np.einsum("ij,j->i",
                                  self.amat_e[idx_e:, nelem:],
                                  fitlin.coef_[nelem:]).mean()
            intercept_col /= intercept_col.sum()
            intercept = intercept_col * (mean_e - intercept)

            self.coefficients = fitlin.coef_
            self.coefficients[:nelem] = intercept

        msg = self.compute_tests(idx_e, idx_f, idx_s, msg)
        self.write_mlip()
        self.init_calc()
        return msg

# ========================================================================== #
    def _initialize_parameters(self, parameters):
        """
        """
        self.parameters = default_parameters
        if parameters is not None:
            self.parameters.update(parameters)

        if self.parameters["method"] != "ols":
            if self.parameters["hyperparameters"] is None:
                hyperparam = {}
            else:
                hyperparam = self.parameters["hyperparameters"]
            hyperparam["fit_intercept"] = False
            self.parameters["hyperparameters"] = hyperparam

# ========================================================================== #
    def get_mlip_dict(self):
        mlip_dict = {}
        mlip_dict['energy_coefficient'] = self.energy_coefficient
        mlip_dict['forces_coefficient'] = self.forces_coefficient
        mlip_dict['stress_coefficient'] = self.stress_coefficient
        return mlip_dict

# ========================================================================== #
    def compute_tests(self, idx_e, idx_f, idx_s, msg):
        # Prepare some data to check accuracy of the fit
        e_true = self.ymat_e[idx_e:]
        e_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amat_e[idx_e:])
        f_true = self.ymat_f[idx_f:]
        f_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amat_f[idx_f:])
        s_true = self.ymat_s[idx_s:] / GPa
        s_mlip = np.einsum('i,ki->k', self.coefficients,
                           self.amat_s[idx_s:]) / GPa

        # Compute RMSE and MAE
        rmse_energy = np.sqrt(np.mean((e_true - e_mlip)**2))
        mae_energy = np.mean(np.abs(e_true - e_mlip))

        rmse_forces = np.sqrt(np.mean((f_true - f_mlip)**2))
        mae_forces = np.mean(np.abs(f_true - f_mlip))

        rmse_stress = np.sqrt(np.mean((s_true - s_mlip)**2))
        mae_stress = np.mean(np.abs(s_true - s_mlip))

        # Prepare message to the log
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
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_forces:.5f} eV/angs   " + \
                 f"mae: {mae_forces:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.vstack((f_true.flatten(), f_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_stress:.5f} GPa       " + \
                 f"mae: {mae_stress:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.vstack((s_true.flatten(), s_mlip.flatten())).T,
                   header=header, fmt="%25.20f  %25.20f")
        return msg
