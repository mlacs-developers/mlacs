import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


# ========================================================================== #
# ========================================================================== #
class Basis:
    """
    """
    def __init__(self):
        pass


# ========================================================================== #
# ========================================================================== #
class ChebyshevDistance(Basis):
    def __init__(self, n, rcut=5.0):
        self.rcut = rcut
        domain = [0, rcut]
        self.b = Chebyshev.basis(n, domain)
        self.db = self.b.deriv()

    def __call__(self, x):
        return fcutoff(x, self.rcut) * self.b(x)

    def deriv(self, x):
        der1 = self.db(x) * fcutoff(x, self.rcut)
        der2 = self.b(x) * dfcutoff(x, self.rcut)
        return der1 + der2


# ========================================================================== #
# ========================================================================== #
class ChebyshevAngle(Basis):
    def __init__(self, n):
        domain = [-1, 1]
        self.b = Chebyshev.basis(n, domain=domain)

    def __call__(self, costheta):
        return self.b(costheta)

    def deriv(self, theta):
        return self.b(theta)


# ========================================================================== #
def fcutoff(x, rcut):
    """
    Cutoff function for the descriptors.
    f(x) = 1 for x < rcut - 1
    f(x) = cos(pi * (x - rcut + 1.0)) / 2 + 1/2 for rcut - 1 < x < rcut
    f(x) = 0 for x > rcut
    """
    res = np.ones(x.shape)
    theta = np.pi * (x - rcut + 1.0)
    idx = x >= rcut - 1.0
    res[idx] = 0.5 * np.cos(theta[idx]) + 0.5
    res[x >= rcut] = 0.0
    return res


# ========================================================================== #
def dfcutoff(x, rcut):
    """
    Derivative of the cutoff function
    """
    res = np.zeros(x.shape)
    theta = np.pi * (x - rcut + 1.0)
    idx = x >= rcut - 1.0
    res[idx] = -0.5 * np.pi * np.sin(theta[idx]) / (rcut - 1.0)
    res[x >= rcut] = 0.0
    return res
