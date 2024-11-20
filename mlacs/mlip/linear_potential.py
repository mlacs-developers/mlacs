"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.units import GPa
from ase import Atoms

from . import MlipManager
from ..utilities import compute_correlation, create_link

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

    parameters: :class:`dict`
        The parameters for the fit.
        By default, the fit is a simple ordinary least squares.
        Ridge regression can be use by setting a dictionnary as
        ``dict(method=ridge, lambda_ridge=alpha)``, with alpha the ridge
        coefficient.

    weight: :class:`WeightingPolicy`
        Weight used for the fitting and calculation of properties.
        Default :class:`None`, which results in the use of uniform weights.

    Examples
    --------

    >>> from ase.io import read
    >>> confs = read('Trajectory.traj', index=':')
    >>>
    >>> from mlacs.mlip import SnapDescriptor, LinearPotential
    >>> desc = SnapDescriptor(confs[0], rcut=6.2, parameters=dict(twojmax=6))
    >>> mlip = LinearPotential(desc)
    >>>
    >>> mlip.update_matrices(confs)
    >>> mlip.train_mlip()
    """

    def __init__(self,
                 descriptor,
                 parameters={},
                 weight=None,
                 **kwargs):

        MlipManager.__init__(self,
                             descriptor,
                             weight,
                             **kwargs)

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
        self.can_use_weight = True

# ========================================================================== #
    def train_mlip(self):
        """
        Compute the coefficients of the MLIP, then write MLIP.

        Notes
        -----

        Local variables:
            - `amat`: ndarray of shape ((7+3*Nat)*N, K)
                Feature matrix
            - `ymat`: ndarray of shape ((7+3*Nat)*N,)
                Label vector
            - `W`: ndarray of shape ((7+3*Nat)*N,)
                Weighting matrix

        Where:
            - `K` is the number of descriptor components
            - `N` is the number of configurations
            - `Nat` is the number of atoms in each cell

        var = `amat`, `ymat`, `W` have a stacked structure
            +-------+
            |   e   |  <-- energy block
            +-------+
            |   f   |  <-- forces block
            +-------+
            |   s   |  <-- stresses block
            +-------+
        where each block has the shape of var_i, with i=e,f,s
        """
        self.weight.workdir = self.workdir
        self.weight.folder = self.folder
        self.weight.subfolder = self.subfolder
        self.descriptor.workdir = self.workdir
        self.descriptor.folder = self.folder
        self.descriptor.subfolder = self.subfolder

        msg = ''
        amat_e = self.amat_e / self.natoms[:, None]
        amat_f = self.amat_f
        amat_s = self.amat_s
        ymat_e = np.copy(self.ymat_e) / self.natoms
        ymat_f = self.ymat_f
        ymat_s = self.ymat_s

        # Division by amat.std : If de=1e2 and ds=1e5, we dont want to fit
        # 1000x more on the stress than on the energy. Careful ymat/AMAT.std
        amat = np.r_[amat_e / amat_e.std(),
                     amat_f / amat_f.std(),
                     amat_s / amat_s.std()]
        ymat = np.r_[ymat_e / amat_e.std(),
                     ymat_f / amat_f.std(),
                     ymat_s / amat_s.std()]

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

        else:
            msg = f"Fitting method {self.parameters['method']} " + \
                  "unknown"
            raise ValueError(msg)

        msg += "\nNumber of configurations for training: " + \
               f"{len(self.natoms):}\n"
        msg += "Number of atomic environments for training: " + \
               f"{self.natoms.sum():}\n\n"

        tmp_msg, weight_fn = self.weight.compute_weight(self.coefficients,
                                                        self.predict)

        msg += tmp_msg
        msg += self.compute_tests(amat_e, amat_f, amat_s,
                                  ymat_e, ymat_f, ymat_s)

        mlip_fn = self.descriptor.write_mlip(self.coefficients)
        create_link(self.subsubdir / weight_fn, self.subdir / weight_fn)
        create_link(self.subsubdir / mlip_fn, self.subdir / mlip_fn)

        if self.log:
            self.log.write(msg)

# ========================================================================== #
    def compute_tests(self, amat_e, amat_f, amat_s,
                      ymat_e, ymat_f, ymat_s):
        """
        Computed the weighted RMSE and MAE.
        """
        e_mlip = np.einsum('ij,j->i', amat_e, self.coefficients)
        f_mlip = np.einsum('ij,j->i', amat_f, self.coefficients)
        s_mlip = np.einsum('ij,j->i', amat_s, self.coefficients)

        w = None
        if len(self.weight.weight) > 0:
            w = self.weight.weight

        wf = np.array([])
        for i in range(len(w)):
            wf = np.append(wf, np.ones(self.natoms[i]*3)*(w[i]/3))

        res_E = compute_correlation(np.c_[ymat_e, e_mlip], weight=w)
        res_F = compute_correlation(np.c_[ymat_f, f_mlip], weight=wf)
        res_S = compute_correlation(np.c_[ymat_s, s_mlip]/GPa, weight=w)
        self.fit_res = np.c_[res_E, res_F, res_S]

        # Information to MLIP-Energy_comparison.dat
        header = f"Weighted rmse: {self.fit_res[0, 0]:.6f} eV/at,    " + \
                 f"Weighted mae: {self.fit_res[1, 0]:.6f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.c_[ymat_e, e_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {self.fit_res[0, 1]:.6f} eV/angs   " + \
                 f"Weighted mae: {self.fit_res[1, 1]:.6f} eV/angs\n" + \
                 " True Forces           Predicted Forces"

        np.savetxt("MLIP-Forces_comparison.dat",
                   np.c_[ymat_f, f_mlip],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {self.fit_res[0, 2]:.6f} GPa       " + \
                 f"Weighted mae: {self.fit_res[1, 2]:.6f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.c_[ymat_s, s_mlip] / GPa,
                   header=header, fmt="%25.20f  %25.20f")

        # Message to Mlacs.log
        msg = f"Weighted RMSE Energy    {self.fit_res[0, 0]:.4f} eV/at\n"
        msg += f"Weighted MAE Energy     {self.fit_res[1, 0]:.4f} eV/at\n"
        msg += f"Weighted RMSE Forces    {self.fit_res[0, 1]:.4f} eV/angs\n"
        msg += f"Weighted MAE Forces     {self.fit_res[1, 1]:.4f} eV/angs\n"
        msg += f"Weighted RMSE Stress    {self.fit_res[0, 2]:.4f} GPa\n"
        msg += f"Weighted MAE Stress     {self.fit_res[1, 2]:.4f} GPa\n"
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
    def predict(self, atoms, coef=None):
        """
        Predict energy (eV), forces (eV/ang) and stress (eV/ang**3) given
        desc which can be of type ase.Atoms or list of ase.Atoms.
        Can choose the coefficients to calculate with, or use the latest one
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if coef is None:
            coef = self.coefficients
        assert coef is not None, 'The model has not been trained'

        if not isinstance(atoms[0], Atoms):
            raise NotImplementedError

        desc = self.descriptor.compute_descriptors(atoms)

        amat_e = [d['desc_e'] for d in desc]
        amat_f = [d['desc_f'] for d in desc]
        amat_s = [d['desc_s'] for d in desc]

        # We use the latest value coefficients to get the properties
        energy = np.einsum('nij,j->n',  amat_e, coef)
        forces = [np.einsum('ij,j->i', conf_f, coef) for conf_f in amat_f]
        stress = np.einsum('nij,j->ni', amat_s, coef)

        if len(energy) == 1:
            return energy[0], forces[0], stress[0]
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
        txt += f"Fit method :            {self.parameters['method']}\n"
        txt += "\n"
        txt += "Descriptor used in the potential:\n"
        txt += repr(self.descriptor)
        return txt
