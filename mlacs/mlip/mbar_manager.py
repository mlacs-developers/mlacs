"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path

import numpy as np

from pymbar import MBAR

from ase.atoms import Atoms
from ase.units import kB, GPa


default_parameters = {"mode": "compute",
                      "step_start": 2,
                      "solver": "L-BFGS-B",
                      }


# ========================================================================== #
# ========================================================================== #
class MbarManager:
    """
    M

    Parameters
    ----------
    """
    def __init__(self, database=None, parameters=dict(),
                 weight=None, folder=""):
        self.parameters = default_parameters
        self.parameters.update(parameters)

        self.database = database
        self._newddb = []
        self.Nk = []
        self.W = None
        self.folder = Path(folder).absolute() 
        self.weight = []
        if weight is not None:
            if isinstance(weight, str):
                weight = np.loadtxt(self.folder + weight)
            self.weight.append(weight)
            we, wf, ws = self._build_W_efs(weight)
            self.W = np.r_[we, wf, ws]
        self.train_mlip = False
        if self.parameters['mode'] == 'train':
            self.train_mlip = True
        self.mlip_amat = []
        self.mlip_coef = []

# ========================================================================== #
    def run_weight(self, a, c):
        """
        Get Ae matrices and SNAP coefficients.
        Compute the matrice Ukn of partition fonctions.
        """

        self.mlip_amat.append(a)
        self.mlip_coef.append(c)

        if self.database is None:
            self.database = []
        self.database.extend(self._newddb)
        self.nconfs = len(self.database)

        if 0 == len(self.mlip_coef):
            self.Nk = np.r_[self.nconfs]
        else:
            self.Nk = np.append(self.Nk, [len(self._newddb)])
        self._newddb = []

        if not 0 == len(self.mlip_coef):
            shape = (len(self.mlip_coef), len(self.mlip_amat[-1]))
            ukn = np.zeros(shape)
            for istep, coeff in enumerate(self.mlip_coef):
                ukn[istep] = self._get_ukn(self.mlip_amat[-1], coeff)

#            print(ukn)
#            print(self.Nk)
            weight = self._compute_weight(ukn)
            self.weight.append(weight)
            neff = self.get_effective_conf()

            header = f"Effective number of configurations: {neff:10.5f}\n"
            np.savetxt(self.folder / "MLIP.weight", self.weight[-1],
                       header=header, fmt="%25.20f")
        return header

# ========================================================================== #
    def reweight_mlip(self, a, y):
        """
        Return weigthted A and Y matrices.
        """
        weight = self._init_weight()
        we, wf, ws = self._build_W_efs(weight)
        self.W = np.r_[we, wf, ws]
        return a * self.W[:, np.newaxis], y * self.W

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s, coeff, msg):
        """
        Computed the weighted RMSE and MAE.
        """
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
    def get_mlip_energy(self, amat_e, coefficient):
        """
        Return Uo from A.D.
        """
        return np.einsum('ij,j->i', amat_e, coefficient)

# ========================================================================== #
    def get_effective_conf(self):
        """
        Compute the number of effective configurations.
        Gives an idea on MLACS convergence.
        """
        neff = np.sum(self.weight[-1])**2 / np.sum(self.weight[-1]**2)
        return neff

# ========================================================================== #
    def update_database(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self._newddb.extend(atoms)

# ========================================================================== #
    def _init_weight(self):
        """
        Initialize the weight matrice.
        """
        n_new = len(self._newddb)
        weight = np.ones(self.nconfs) / self.nconfs
        nef = np.sum(self.weight[-1])**2 / np.sum(self.weight[-1]**2)
        if nef > 1.5 * n_new:
            weight = 0.0 * weight
            weight[:-n_new] = self.weight[-1]
            return weight
        weight = 0.1 * weight
        weight[:-n_new] += self.weight[-1]
        return weight / np.sum(weight)

# ========================================================================== #
    def _get_ukn(self, a, c):
        """
        """
        ddb = self.database
        P = np.zeros(self.nconfs)
        T = np.array([_.get_temperature() for _ in ddb])
        V = np.array([_.get_volume() for _ in ddb])
        if np.abs(np.diff(V)).sum() != 0.0:
            P = np.array([-np.sum(_.get_stress()[:3]) / 3 for _ in ddb])
        ekn = self.get_mlip_energy(a, c)
        assert len(ekn) == self.nconfs
        ukn = (ekn + P * V * GPa) / (kB * T)
        return ukn

# ========================================================================== #
    def _compute_weight(self, ukn):
        """
        """
        mbar = MBAR(ukn, self.Nk,
                    solver_protocol=[{'method': self.parameters['solver']}])
        weight = mbar.weights()[:, -1]
        return weight

# ========================================================================== #
    def _build_W_efs(self, weight):
        """
        """
        w_e = self.weight[-1] / np.sum(self.weight[-1])
        w_f = []
        w_s = []
        for a in self.database:
            w_f.append(self.weight[-1] * np.ones(3 * len(a)) / (3 * len(a)))
            w_s.append(self.weight[-1] * np.ones(6) / 6)
        w_f = np.r_w_f / np.sum(np.r_w_f)
        w_s = np.r_w_s / np.sum(np.r_w_s)
        return w_e, w_f, w_s
