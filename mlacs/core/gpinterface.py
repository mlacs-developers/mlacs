"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF,
                                                  ConstantKernel as C,
                                                  WhiteKernel)
except ImportError:
    msg = "You need sklearn to use the calphagpy modules"
    raise ModuleNotFoundError(msg)


default_gp_parameters = {"n_restarts_optimizer": 100,
                         "normalize_y": True,
                         "alpha": 1e-10}


# ========================================================================== #
# ========================================================================== #
class GaussianProcessInterface:
    """
    Base class to interface gaussian process from scikit-learn

    Parameters
    ----------

    ndim: :class:`int`
        The number of dimensions for the feature
    kernel: :class:`Kernel`
        The kernel of the gaussian process.
        Default are :
        RBF() * C() + WhiteKernel()
    gp_parameters: :class:`dict`
        Parameters for the GaussianProcessRegressor object.
        The default parameters are :
        {"n_restart_optimizer": 100, "normalyze_y": True, "alpha": 1e-10}
    """
    def __init__(self,
                 ndim=1,
                 kernel=RBF()*C()+WhiteKernel,
                 gp_parameters={}):

        default_gp_parameters.update(gp_parameters)
        self.gp = GaussianProcessRegressor(kernel=kernel,
                                           **default_gp_parameters)

        self.trained = False

        self.ndim = ndim
        self.x = None
        self.y = None

# ========================================================================== #
    def add_new_data(self, x, y):
        """
        Add new data to train the gaussian process

        Parameters
        ----------

        x: :class:`np.ndarray`
            The features, with dimension (nsamples, nfeatures)
        y: :class:`np.ndarray`
            The target values, with dimension (nsamples, ndim)
        """
        self._add_new_data(x, y)

# ========================================================================== #
    def _add_new_data(self, x, y):
        """
        """
        # We need to ensure that x and y has the right dimensions
        if len(x.shape) == 1 and self.ndim == 1:
            x = x.reshape(-1, 1)
        elif len(x.shape) == 1 and self.ndim > 1:
            x = x.reshape(1, -1)

        if self.x is None:
            self.x = x
            self.y = y
        else:
            self.x = np.r_[self.x, x]
            self.y = np.r_[self.y, y]

# ========================================================================== #
    def train(self):
        """
        Train the gaussian process.
        """
        if self.x is None or self.y is None:
            msg = "You need to add data with the add_new_data() function " + \
                  "to train the gaussian process"
            raise RuntimeError(msg)

        # Do the fit
        self.gp.fit(self.x, self.y)
        self.trained = True

        # Get some lower bounds for the predictions
        self.lb = self.x.min(axis=0)
        self.ub = self.x.max(axis=0)

# ========================================================================== #
    def predict(self, x, return_cov=False):
        """
        Predict values for a given x

        Parameters
        ----------

        x: :class:`np.ndarray`
            The features, with dimension (nsamples, nfeatures)
        """
        if not self.trained:
            msg = "You need to train the gaussian process before doing " + \
                  "predictions"
            raise RuntimeError(msg)

        # We need to have everything as array so that sklearn is happy
        if isinstance(x, (float, int)):
            x = np.array([x])
        elif isinstance(x, list):
            x = np.array(x)

        # We need to ensure that x has the right dimensions
        if len(x.shape) == 1 and self.ndim == 1:
            x = x.reshape(-1, 1)
        elif len(x.shape) == 1 and self.ndim > 1:
            x = x.reshape(1, -1)
        if return_cov:
            y, y_cov = self.gp.predict(x, return_cov=True)
            return y, y_cov
        else:
            y = self.gp.predict(x)
            return y
