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
                 weight=None,
                 solver='L-BFGS-B'):
        """
        Initialisation
        """
        self.weight = weight
        self.weight_mat = []
        if self.weight is not None:
            self.weight_mat.append(self.weight)
        self.solver = solver
        self.mlip_amat = []
        self.mlip_coef = []
        self.database = None

# ========================================================================== #
    def _compute_weight(self):
        """
        """ 
        mbar = MBAR(u_kn, len(self.database), 
                    solver_protocol={'method':self.solver}) 
        self.weight_mat.append(mbar.getWeights()[:,-1])
        return self.weight_mat[-1]

# ========================================================================== #
    def get_mlip_energy(self, amat_e, coefficient):
        """
        """ 
        return np.einsum('ij,j->i', amat_e, coefficient)

# ========================================================================== #
    def _init_weight(self, conf):
        """
        Initialize the weight matrice.
        """
        nconf = len(conf)
        weight = np.ones(nconf) / nconf 
        if nconf <= 2:  
            return weight
        nef = np.sum(self.weight[-1])**2 / np.sum(self.weight[-1]**2)
        if nef > 1.5 * self.every:
            weight = 0.0 * weight
            weight[self.every:] = self.weight[-1]
            return weight
        weight = 0.1 * weight
        weight[self.every:] += self.weight[-1] 
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
        ukn = (self.get_mlip_energy(a, c) + P * V * GPa) / kB * T 
        return ukn 
