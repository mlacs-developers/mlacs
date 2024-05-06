"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path
import numpy as np
from ..utilities import subfolder
from ase.atoms import Atoms


# ========================================================================== #
# ========================================================================== #
class WeightingPolicy:
    """
    Parent class to manage weight in MLACS. This class define the standard to
    be used by the MLIP.

    Parameters
    ----------
    energy_coefficient: :class:`float`
        Weight of the energy in the fit
        Default 1.0

    forces_coefficient: :class:`float`
        Weight of the forces in the fit
        Default 1.0

    stress_coefficient: :class:`float`
        Weight of the stress in the fit
        Default 1.0

    database: :class:`ase.Trajectory`
        Initial database (optional)
        Default :class:`None`

    weight: :class:`list` or :class:`str`
        If you use an initial database, it needs weight.
        Can a list or an np.array of values or a file.
        Default :class:`None`

    """

    def __init__(self, energy_coefficient=1.0, forces_coefficient=1.0,
                 stress_coefficient=1.0, database=None, weight=None):
        self.database = database
        self.matsize = []

        sum_efs = energy_coefficient + forces_coefficient + stress_coefficient
        self.energy_coefficient = energy_coefficient / sum_efs
        self.forces_coefficient = forces_coefficient / sum_efs
        self.stress_coefficient = stress_coefficient / sum_efs

        if database is not None:
            self.matsize = [len(a) for a in database]

        self.weight = np.array([])
        if weight is not None:
            if isinstance(weight, str):
                weight = np.loadtxt(weight)
            self.weight = weight
        elif Path("MLIP.weight").exists():
            weight = np.loadtxt("MLIP.weight")
            self.weight = weight
        else:
            self.weight = np.array([])

# ========================================================================== #
    def get_effective_conf(self):
        """
        Compute the number of effective configurations.
        """
        if len(self.weight) == 0:
            return 0
        neff = np.sum(self.weight)**2 / np.sum(self.weight**2)
        return neff

# ========================================================================== #
    def get_weights(self):
        """
        Return weighting matrices
        """
        w = self.init_weight()
        we, wf, ws = self.build_W_efs(w)
        we = we * self.energy_coefficient
        wf = wf * self.forces_coefficient
        ws = ws * self.stress_coefficient
        return np.r_[we, wf, ws]

# ========================================================================== #
    def init_weight(self, scale=1):
        """
        Scale the weight matrice to include the new configurations.
        Those have weight 1/Neff.
        """
        n_tot = len(self.matsize)
        neff = self.get_effective_conf()
        nnew = n_tot - len(self.weight)
        weight = (np.ones(n_tot)*scale) / (neff + nnew)
        ratio = neff / (neff + nnew)
        weight[:len(self.weight)] = self.weight * ratio
        return weight / np.sum(weight)

# ========================================================================== #
    def build_W_efs(self, w):
        """
        Transform W to W_efs.
        """
        w_e = w / np.sum(w)
        w_f = []
        w_s = []
        for i, n in enumerate(self.matsize):
            w_f.extend(w[i] * np.ones(3 * n) / (3 * n))
            w_s.extend(w[i] * np.ones(6) / 6)
        w_f = np.r_[w_f] / np.sum(np.r_[w_f])
        w_s = np.r_[w_s] / np.sum(np.r_[w_s])
        return w_e, w_f, w_s

# ========================================================================== #
    def update_database(self, atoms):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    @subfolder
    def compute_weight(self, coef=None, predict=None):
        """
        """
        raise NotImplementedError

