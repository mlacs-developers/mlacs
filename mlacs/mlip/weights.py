"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path
import numpy as np
from ase.atoms import Atoms

from ..utilities import subfolder
from .weighting_policy import WeightingPolicy


# ========================================================================== #
# ========================================================================== #
class DrautzWeight(WeightingPolicy):
    """
    Class that gives weight according to w_n = C/[E_n - E_min + delta]**2
    where C is a normalization constant.

    Parameters
    ----------
    delta: :class:`float`
        Shift to avoid overweighting of the ground state (eV/at)
        Default 1.0
    """
    def __init__(self, delta=1.0, energy_coefficient=1.0,
                 forces_coefficient=1.0, stress_coefficient=1.0,
                 database=None, weight=None):
        self.delta = delta
        self.energies = []
        WeightingPolicy.__init__(
                self,
                energy_coefficient=energy_coefficient,
                forces_coefficient=forces_coefficient,
                stress_coefficient=stress_coefficient,
                database=database, weight=weight)

# ========================================================================== #
    @subfolder
    def compute_weight(self, coef=None, f_mlipE=None):
        """
        Compute Uniform Weight taking into account nthrow :
        """
        if Path("MLIP.weight").exists():
            Path("MLIP.weight").unlink()
        emin = min(self.energies)
        w = np.array([1/(en - emin + self.delta)**2 for en in self.energies])
        self.weight = w / np.sum(w)

        header = "Using Drautz weighting\n"
        np.savetxt("MLIP.weight", self.weight, header=header, fmt="%25.20f")
        return header, "MLIP.weight"

# ========================================================================== #
    def update_database(self, atoms):
        """
        Update the database.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.energies.extend([a.get_potential_energy()/len(a) for a in atoms])
        self.matsize.extend([len(a) for a in atoms])


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
    def compute_weight(self, coef=None, f_mlipE=None):
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
