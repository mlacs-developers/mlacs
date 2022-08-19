"""
"""
import numpy as np


def compute_weight_multiensemble(confs, t_range, p_range=None):
    """
    Function to compute the
    """
class MultiEnsembleAnalysis:
    """
    """
    def __init__(self,
                 confs,
                 t_range,
                 p_range=None,
                 nbasis=10,
                 verbose=True):
        self.confs = confs

        self.tmin, self.tmax = t_range
        if p_range is not None:
            self.pmin, self.pmax = p_range


        self._minimize_bias(nbasis)

    def _minimize_bias(self, nbasis):
        """
        """
        eff_temp = (self.tmax - self.tmin) / 2.0
        energy = np.array([at.get_potential_energy() for at in self.confs])
