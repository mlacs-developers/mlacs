'''
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
'''
import numpy as np
from ase.units import GPa

from mlacs.mlip import MlipManager
try:
    from pytorch import nn
except ImportError:
    msg = "You need PyTorch installed to use Neural Networks model"
    raise ImportError(msg)


default_parameters = {"hidenlayers": [3, 3],
                      "activation": "tanh",
                      "alpha": 1e-6,
                      "epoch": 300}


# ========================================================================== #
# ========================================================================== #
class NeuralNetworkMlip(MlipManager):
    """
    Parent Class for Neural Network MLIP
    """
    def __init__(self,
                 atoms,
                 rcut=5.0,
                 nthrow=10,
                 parameters=None,
                 energy_coefficient=1.0,
                 stress_coefficient=1.0,
                 forces_coefficient=1.0):
        MlipManager.__init__(self,
                             atoms,
                             rcut,
                             nthrow,
                             energy_coefficient,
                             forces_coefficient,
                             stress_coefficient)

        self._initialize_parameters(parameters)

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

        msg = "number of configurations for training:  " + \
              f"{len(self.amatrix_energy[idx:])}\n"

        return msg

# ========================================================================== #
    def _initialize_parameters(self, parameters):
        """
        """
        self.parameters = default_parameters
        if parameters is not None:
            self.parameters.update(parameters)

        self.istrain = False
