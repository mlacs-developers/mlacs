'''
'''
import numpy as np
from ase.units import GPa

from . import MlipManager


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
    """
    def __init__(self,
                 descriptor,
                 nthrow=0,
                 parameters={},
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0,
                 mbar=None):
        MlipManager.__init__(self,
                             descriptor,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient,
                             mbar)

        self.parameters = default_parameters
        self.parameters.update(parameters)

        pair_style, pair_coeff = self.descriptor.get_pair_style_coeff()
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff

        self.coefficients = None

        if self.parameters["method"] != "ols":
            if self.parameters["hyperparameters"] is None:
                hyperparam = {}
            else:
                hyperparam = self.parameters["hyperparameters"]
            hyperparam["fit_intercept"] = False
            self.parameters["hyperparameters"] = hyperparam

# ========================================================================== #
    def train_mlip(self):
        """
        """
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

        if self.mbar is not None and self.mbar.train_mlip:
            amat, ymat = self.mbar.reweight_mlip(amat, ymat)

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

        _msg = "Number of configurations for training: " + \
               f"{len(self.natoms[idx_e:]):}\n"
        _msg += "Number of atomic environments for training: " + \
                f"{self.natoms[idx_e:].sum():}\n"

        msg = self.compute_tests(amat_e, amat_f, amat_s,
                                 ymat_e, ymat_f, ymat_s,
                                 _msg)

        if self.mbar is not None:
            if self.mbar.train_mlip:
                msg = self.mbar.compute_tests(amat_e, amat_f, amat_s,
                                              ymat_e, ymat_f, ymat_s,
                                              self.coefficients, _msg)
            msg += self.mbar.run_weight(amat_e, self.coefficients)

        self.descriptor.write_mlip(self.coefficients)
        return msg

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s, msg):
        e_mlip = np.einsum('ij,j->i', amat_e, self.coefficients)
        f_mlip = np.einsum('ij,j->i', amat_f, self.coefficients)
        s_mlip = np.einsum('ij,j->i', amat_s, self.coefficients)

        rmse_e = np.sqrt(np.mean((ymat_e - e_mlip)**2))
        mae_e = np.mean(np.abs(ymat_e - e_mlip))

        rmse_f = np.sqrt(np.mean((ymat_f - f_mlip)**2))
        mae_f = np.mean(np.abs(ymat_f - f_mlip))

        rmse_s = np.sqrt(np.mean((((ymat_s - s_mlip) / GPa)**2)))
        mae_s = np.mean(np.abs((ymat_s - s_mlip) / GPa))

        # Prepare message to the log
        msg += f"RMSE Energy    {rmse_e:.4f} eV/at\n"
        msg += f"MAE Energy     {mae_e:.4f} eV/at\n"
        msg += f"RMSE Forces    {rmse_f:.4f} eV/angs\n"
        msg += f"MAE Forces     {mae_f:.4f} eV/angs\n"
        msg += f"RMSE Stress    {rmse_s:.4f} GPa\n"
        msg += f"MAE Stress     {mae_s:.4f} GPa\n"
        msg += "\n"

        header = f"rmse: {rmse_e:.5f} eV/at,    " + \
                 f"mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.c_[ymat_e, e_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_f:.5f} eV/angs   " + \
                 f"mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("MLIP-Forces_comparison.dat",
                   np.c_[ymat_f, f_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_s:.5f} GPa       " + \
                 f"mae: {mae_s:.5f} GPa\n" + \
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

        res = self.descriptor.calculate(atoms)[0]
        energy = np.einsum('ij,j->', res['desc_e'], self.coefficients)
        forces = np.einsum('ij,j->i', res['desc_f'], self.coefficients)
        stress = np.einsum('ij,j->i', res['desc_s'], self.coefficients)

        forces = forces.reshape(len(atoms), 3)

        return energy, forces, stress

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
