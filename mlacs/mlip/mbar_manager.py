"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ase.units import kB, GPa

from pymbar.mbar import MBAR


# ========================================================================== #
# ========================================================================== #
class MbarManager:
    """
    M

    Parameters
    ----------
    """
    def __init__(self,
                 database,
                 weight=None,
                 every=1,
                 nthrow=10,
                 solver='L-BFGS-B'):
        """
        Initialisation
        """
        self.every = every
        self.nthrow = nthrow
        self.weight = weight
        self.W = None
        self.weight_mat = []
        if self.weight is not None:
            self.weight_mat.append(self.weight)
        self.solver = solver
        self.mlip_amat = []
        self.mlip_coef = []
        self.database = None

# ========================================================================== #
    def update_weight(self):
        """
        """
        shape = (len(self.mlip_coef), len(self.mlip_amat[-1]))
        ukn = np.zeros(shape)
        for istep, coeff in enumerate(self.mlip_coef):
            ukn[istep] = self._get_ukn(self.mlip_amat[-1], coeff)
        self.weight = self._compute_weight(ukn)

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s, coeff, msg):
        e_mlip = np.einsum('ij,j->i', amat_e, coeff)
        f_mlip = np.einsum('ij,j->i', amat_f, coeff)
        s_mlip = np.einsum('ij,j->i', amat_s, coeff)

        reweight = self._reweight_conf()

        rmse_e = np.sqrt(np.mean(reweight[0] * (ymat_e - e_mlip)**2))
        mae_e = np.mean(reweight[0] * np.abs(ymat_e - e_mlip))

        rmse_f = np.sqrt(np.mean(reweight[1] * (ymat_f - f_mlip)**2))
        mae_f = np.mean(reweight[1] * np.abs(ymat_f - f_mlip))

        rmse_s = np.sqrt(np.mean(reweight[2] * ((ymat_s - s_mlip) / GPa)**2))
        mae_s = np.mean(reweight[2] * np.abs((ymat_s - s_mlip) / GPa))

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
    def _compute_weight(self, ukn):
        """
        """
        mbar = MBAR(ukn, len(self.database),
                    solver_protocol={'method': self.solver})
        self.weight_mat.append(mbar.getWeights()[:, -1])
        return self.weight_mat[-1]

# ========================================================================== #
    def get_mlip_energy(self, amat_e, coefficient):
        """
        """
        return np.einsum('ij,j->i', amat_e, coefficient)

# ========================================================================== #
    def _init_weight(self):
        """
        Initialize the weight matrice.
        """
        nconf = len(self.database)
        weight = np.ones(nconf) / nconf
        if nconf <= self.nthrow:
            return weight
        nef = np.sum(self.weight_mat[-1])**2 / np.sum(self.weight_mat[-1]**2)
        if nef > 1.5 * self.every:
            weight = 0.0 * weight
            weight[self.every:] = self.weight_mat[-1]
            return weight
        weight = 0.1 * weight
        weight[self.every:] += self.weight_mat[-1]
        return weight / np.sum(weight)

# ========================================================================== #
    def _get_ukn(self, a, c):
        """
        """
        P = np.zeros(len(self.database))
        T = np.array([_.get_temperature() for _ in self.database])
        V = np.array([_.get_volume() for _ in self.database])
        if np.abs(np.diff(V)).sum() != 0.0:
            P = np.array([-np.sum(_.get_stress()[:3]) / 3
                          for _ in self.database])
        ekn = self.get_mlip_energy(a, c)
        assert len(ekn) == len(self.database)
        ukn = (ekn + P * V * GPa) / (kB * T)
        return ukn

# ========================================================================== #
    def _reweight_conf(self):
        """
        """
