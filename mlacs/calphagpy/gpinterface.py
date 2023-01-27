import numpy as np
from scipy.stats.qmc import Sobol, LatinHypercube
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (RBF,
                                                  ConstantKernel as C)
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
    """
    def __init__(self,
                 name,
                 ndim=1,
                 kernel=RBF()*C(),
                 gp_parameters={}):
        self.name = name

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
        """
        # We need to ensure that x and y has the right dimensions
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        self.x = x
        self.y = y

# ========================================================================== #
    def train(self):
        """
        """
        if self.x is None or self.y is None:
            msg = "You need to add data with the add_new_data() function " + \
                  "to train the gaussian process"
            raise RuntimeError(msg)

        self.gp.fit(self.x, self.y)
        self.trained = True

# ========================================================================== #
    def predict(self, x, return_cov=False):
        """
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
        elif len(x.shape) == 1 and self.ndim == 2:
            x = x.reshape(1, -1)
        if return_cov:
            y, y_cov = self.gp.predict(x, return_cov=True)
            return y, y_cov
        else:
            y = self.gp.predict(x)
            return y
