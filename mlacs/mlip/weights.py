"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
import numpy as np
from ase.atoms import Atoms
from ..core.manager import Manager

from .weighting_policy import WeightingPolicy


# ========================================================================== #
# ========================================================================== #
class FixedWeight(WeightingPolicy):
    """
    Class that gives a static weight to the first few configurations and then
    use the given WeightingPolicy for the other configurations.
    Can be used to give a fixed weight to the training configurations

    Parameters
    ----------
    subweight: :class:`WeightingPolicy`
        Weight of the configurations after
        Default UniformWeight

    static_weight: :class:`np.array`
        Weights of the first configuration. The number of configuration is
        given by the length of the array. The weight must be normalized
        Default [0]
    """
    def __init__(self, static_weight=np.array([0]), subweight=None,
                 energy_coefficient=1.0, forces_coefficient=1.0,
                 stress_coefficient=1.0):

        assert np.sum(static_weight) <= 1.0
        self.subweight = subweight
        self.static_weight = static_weight
        self.nstatic = len(static_weight)
        self.remaining = 1-np.sum(static_weight)
        self.matsize = []
        self.weight = np.array([])

        if subweight is None:
            self.subweight = UniformWeight(
                energy_coefficient=energy_coefficient,
                forces_coefficient=forces_coefficient,
                stress_coefficient=stress_coefficient)
        self.energy_coefficient = self.subweight.energy_coefficient
        self.forces_coefficient = self.subweight.forces_coefficient
        self.stress_coefficient = self.subweight.stress_coefficient

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def compute_weight(self, coef=None, predict=None, **kwargs):
        """
        Compute Uniform Weight taking into account nthrow :
        """
        self.subweight.workdir = self.workdir
        self.subweight.folder = self.folder
        self.subweight.subfolder = self.subfolder

        subweight_name = type(self.subweight).__name__
        header2 = ""

        if len(self.matsize) == (self.nstatic+1):  # Exactly 1 conf not static
            self.weight = np.append(self.static_weight, self.remaining)
        elif len(self.matsize) > self.nstatic:
            tmp, fn = self.subweight.compute_weight(coef=coef,
                                                    predict=predict,
                                                    **kwargs)
            header2 += tmp
            dynamic_w = self.remaining * self.subweight.weight
            self.weight = np.append(self.static_weight, dynamic_w)
        else:
            curr_weight = self.static_weight[:len(self.matsize)]
            if np.sum(curr_weight) == 0:  # A niche bug
                curr_weight = np.ones(len(self.matsize))
            self.weight = curr_weight/np.sum(curr_weight)

        header = f"Using Fixed weighting and {subweight_name}\n"
        header += f"{header2}\n"
        if Path("MLIP.weight").exists():
            Path("MLIP.weight").unlink()
        np.savetxt("MLIP.weight", self.weight, header=header, fmt="%25.20f")
        return header, "MLIP.weight"

# ========================================================================== #
    def update_database(self, atoms):
        """
        Update the database.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        if len(self.matsize) > self.nstatic:
            self.subweight.update_database(atoms)
        elif len(self.matsize) + len(atoms) > self.nstatic:
            idx = len(self.matsize) + len(atoms) - self.nstatic
            self.subweight.update_database(atoms[-idx:])
        else:  # Static weights
            pass

        self.matsize.extend([len(a) for a in atoms])


# ========================================================================== #
# ========================================================================== #
class EnergyBasedWeight(WeightingPolicy):
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
    @Manager.exec_from_subsubdir
    def compute_weight(self, coef=None, predict=None, **kwargs):
        """
        Compute Uniform Weight taking into account nthrow :
        """
        if Path("MLIP.weight").exists():
            Path("MLIP.weight").unlink()
        emin = min(self.energies)
        w = np.array([1/(en - emin + self.delta)**2 for en in self.energies])
        self.weight = w / np.sum(w)

        header = "Using EnergyBased weighting\n"
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
        Three cases:

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
    @Manager.exec_from_subsubdir
    def compute_weight(self, coef=None, predict=None, **kwargs):
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


# ========================================================================== #
# ========================================================================== #
class IncreasingWeight(WeightingPolicy):
    """
    Class that gives increasing weight with the index of a configuration
    in MLACS.
    This weighting policy has been though for structural optimization.

    Parameters
    ----------

    nthrow: :class:`int`
        Number of configurations to ignore when doing the fit.
        Three cases :

         1. If nconf > 2*nthrow, remove the nthrow first configuration
         2. If nthrow < nconf < 2*nthrow, remove the nconf-nthrow first conf
         3. If nconf < nthrow, keep all conf
    """

    def __init__(self, nthrow=0, power=1, energy_coefficient=1.0,
                 forces_coefficient=1.0, stress_coefficient=1.0,
                 database=None, weight=None):
        self.nthrow = nthrow
        self.power = power
        WeightingPolicy.__init__(
                self,
                energy_coefficient=energy_coefficient,
                forces_coefficient=forces_coefficient,
                stress_coefficient=stress_coefficient,
                database=database, weight=weight)

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def compute_weight(self, coef, f_mlipE, **kwargs):
        """
        Compute Increasing Weight taking into account nthrow :
        """
        fname = "MLIP.weight"
        if (filepath := Path(fname)).exists():
            filepath.unlink()

        nconf = len(self.matsize)
        to_remove = 0
        if nconf > 2*self.nthrow:
            to_remove = self.nthrow
        elif nconf > self.nthrow:
            to_remove = nconf-self.nthrow

        w = (np.arange(nconf-to_remove, dtype=float) + 1)**self.power
        w /= w.sum()
        w = np.r_[np.zeros(to_remove), w]
        self.weight = w

        header = "Using Increasing weighting\n"
        np.savetxt(fname, self.weight, header=header, fmt="%25.20f")
        return header, fname

# ========================================================================== #
    def update_database(self, atoms):
        """
        Update the database.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.matsize.extend([len(a) for a in atoms])
