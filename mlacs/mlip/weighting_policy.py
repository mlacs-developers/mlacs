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
        self.matsize = None
        self.matsize = []

        self.energy_coefficient = energy_coefficient
        self.forces_coefficient = forces_coefficient
        self.stress_coefficient = stress_coefficient

        if database is not None:
            self.matsize = [len(a) for a in database]

        self.weight = []
        if weight is not None:
            if isinstance(weight, str):
                weight = np.loadtxt(weight)
            self.weight = weight
        elif Path("MLIP.weight").exists():
            weight = np.loadtxt("MLIP.weight")
            self.weight = weight
        else:
            self.weight = []

# ========================================================================== #
    @subfolder
    def compute_weight(self, coef, f_mlipE):
        """
        """
        raise NotImplementedError

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
    def update_database(self, atoms):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    def init_weight(self):
        """
        Initialize the weight matrice with W = scale * 1/N.
        """
        n_tot = len(self.matsize)
        weight = np.ones(n_tot) / n_tot
        weight[:len(self.weight)] = self.weight
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
# ========================================================================== #
class UniformWeight(WeightingPolicy):
    """
    Class that gives uniform weight in MLACS.

    Parameters
    ----------
    nthrow: :class:`int`
        Number of configurations to ignore when doing the fit.
        Three cases :

        1. If nconf > 2*nthrow, remove the nthrow first configuration
        2. If nthrow < nconf < 2*nthrow, remove the nconf-nthrow first conf
        3. If nconf < nthrow, keep all conf

    """

    def __init__(self, nthrow=0, energy_coefficient=1.0,
                 forces_coefficient=1.0, stress_coefficient=1.0,
                 database=None, weight=None):
        self.nthrow = nthrow
        WeightingPolicy.__init__(
                self,
                energy_coefficient=energy_coefficient,
                forces_coefficient=forces_coefficient,
                stress_coefficient=stress_coefficient,
                database=database, weight=weight)

# ========================================================================== #
    @subfolder
    def compute_weight(self, coef, f_mlipE):
        """
        Compute Uniform Weight taking into account nthrow :
        """
        if Path("MLIP.weight").exists():
            Path("MLIP.weight").unlink()

        nconf = len(self.matsize)
        to_remove = 0
        if nconf > 2*self.nthrow:
            to_remove = self.nthrow
        elif nconf > self.nthrow:
            to_remove = nconf-self.nthrow

        w = np.ones(nconf-to_remove) / (nconf-to_remove)
        w = np.r_[np.zeros(to_remove), w]
        self.weight = w

        header = "Using Uniform weighting\n"
        np.savetxt("MLIP.weight", self.weight, header=header, fmt="%25.20f")
        return header, "MLIP.weight"

# ========================================================================== #
    def update_database(self, atoms):
        """
        Update the database.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.matsize.extend([len(a) for a in atoms])
