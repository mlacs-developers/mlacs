"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ase.units import kB, GPa

from pymbar.mbar import MBAR


default_parameters = {"every": 1,
                      "mode": "compute",
                      "solver": "L-BFGS-B",
                      "nthrow": 10}


# ========================================================================== #
# ========================================================================== #
class MbarManager:
    """
    M

    Parameters
    ----------
    """
    def __init__(self,
                 database=[],
                 parameters={},
                 weight=None):
        """
        Initialisation
        """
        self.parameters = default_parameters
        self.parameters.update(parameters)

        self.every = self.parameters['every']
        self.nthrow = self.parameters['nthrow']
        self.database = database
        self.W = None
        self.weight = []
        if weight is not None:
            if isinstance(weight, str):
                weight = np.loadtxt(weight)
            self.weight.append(weight)
            we, wf, ws = self._build_W_efs(weight)
            self.W = np.r_[we, wf, ws]
        self.train_mlip = False
        if self.parameters['mode'] == 'train':
            self.train_mlip = True
        self.mlip_amat = []
        self.mlip_coef = []

# ========================================================================== #
    def run_weight(self):
        """
        """
        shape = (len(self.mlip_coef), len(self.mlip_amat[-1]))
        ukn = np.zeros(shape)
        for istep, coeff in enumerate(self.mlip_coef):
            ukn[istep] = self._get_ukn(self.mlip_amat[-1], coeff)
        weight = self._compute_weight(ukn)
        self.weight.append(weight)
        neff = self.get_effective_conf()
        header = f"Effective number of configurations: {neff}\n"
        np.savetxt("MLIP.weight", self.weight[-1],
                   header=header, fmt="%25.20f")
        msg = "Computing new weights with MBAR\n"
        msg += header
        return msg

# ========================================================================== #
    def reweight_mlip(self, a, y):
        """
        """
        weight = self._init_weight()
        we, wf, ws = self._build_W_efs(weight)
        self.W = np.r_[we, wf, ws]
        return a * self.W[:, np.newaxis], y * self.W

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s, coeff, msg):
        e_mlip = np.einsum('ij,j->i', amat_e, coeff)
        f_mlip = np.einsum('ij,j->i', amat_f, coeff)
        s_mlip = np.einsum('ij,j->i', amat_s, coeff)

        weight = self._init_weight()
        we, wf, ws = self._build_W_efs(weight)

        rmse_e = np.sqrt(np.mean(we * (ymat_e - e_mlip)**2))
        mae_e = np.mean(we * np.abs(ymat_e - e_mlip))

        rmse_f = np.sqrt(np.mean(wf * (ymat_f - f_mlip)**2))
        mae_f = np.mean(wf * np.abs(ymat_f - f_mlip))

        rmse_s = np.sqrt(np.mean(ws * ((ymat_s - s_mlip) / GPa)**2))
        mae_s = np.mean(ws * np.abs((ymat_s - s_mlip) / GPa))

        # Prepare message to the log
        msg += f"Weighted RMSE Energy    {rmse_e:.4f} eV/at\n"
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
#    def _compute_weight(self, ukn):
#        """
#        """
#        n_newconf = len(self.database[-1] - len(self.database[-2]
#        mbar = MBAR(ukn, n_newconf,
#                    solver_protocol=[{'method': self.parameters['solver']}])
#        weight = mbar.getWeights()[:, -1]
#        return weight

# ========================================================================== #
    def get_mlip_energy(self, amat_e, coefficient):
        """
        """
        return np.einsum('ij,j->i', amat_e, coefficient)

# ========================================================================== #
    def get_effective_conf(self):
        """
        """
        neff = np.sum(self.weight[-1])**2 / np.sum(self.weight[-1]**2)
        return neff

# ========================================================================== #
    def _init_weight(self):
        """
        Initialize the weight matrice.
        """
        nconf = len(self.database[-1])
        weight = np.ones(nconf) / nconf
        if nconf <= self.nthrow:
            return weight
        nef = np.sum(self.weight[-1])**2 / np.sum(self.weight[-1]**2)
        if nef > 1.5 * self.every:
            weight = 0.0 * weight
            weight[:-self.every] = self.weight[-1]
            return weight
        weight = 0.1 * weight
        weight[:-self.every] += self.weight[-1]
        return weight / np.sum(weight)

# ========================================================================== #
    def _get_ukn(self, a, c):
        """
        """
        P = np.zeros(len(self.database[-1]))
        T = np.array([_.get_temperature() for _ in self.database[-1]])
        V = np.array([_.get_volume() for _ in self.database[-1]])
        if np.abs(np.diff(V)).sum() != 0.0:
            P = np.array([-np.sum(_.get_stress()[:3]) / 3
                          for _ in self.database[-1]])
        ekn = self.get_mlip_energy(a, c)
        assert len(ekn) == len(self.database[-1])
        ukn = (ekn + P * V * GPa) / (kB * T)
        return ukn

# ========================================================================== #
    def _build_W_efs(self, weight):
        """
        """
        w_e = self.weight[-1] / np.sum(self.weight[-1])
        w_f = []
        w_s = []
        for a in self.database[-1]:
            w_f.append(self.weight[-1] * np.ones(3 * len(a)) / (3 * len(a)))
            w_s.append(self.weight[-1] * np.ones(6) / 6)
        w_f = np.r_w_f / np.sum(np.r_w_f)
        w_s = np.r_w_s / np.sum(np.r_w_s)
        return w_e, w_f, w_s
