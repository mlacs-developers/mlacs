"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from abc import ABC, abstractmethod

from ..core import Manager

import numpy as np


# ========================================================================== #
# ========================================================================== #
class StateManager(ABC, Manager):
    """
    Parent Class managing the state being simulated
    """
    def __init__(self,
                 nsteps=1000,
                 nsteps_eq=100,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 folder='MolecularDynamics',
                 **kwargs):

        Manager.__init__(self, folder=folder, **kwargs)

        self._nsteps = nsteps
        self._nsteps_eq = nsteps_eq
        self._logfile = logfile
        self._trajfile = trajfile
        self._loginterval = loginterval

# ========================================================================== #
    @abstractmethod
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post,
                     atom_style="atomic",
                     eq=False,
                     elements=None):
        """
        Run the dynamics for the state, during nsteps
        then return the last atoms of the simulation
        """
        pass

# ========================================================================== #
    def initialize_momenta(self, atoms):
        """
        Setup the momenta during initialization of the simulation
        """
        if not atoms.has("momenta"):
            momenta = np.zeros((len(atoms), 3))
            atoms.set_momenta(momenta)

# ========================================================================== #
    @abstractmethod
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        return ""

# ========================================================================== #
    def __repr__(self):
        return self.log_recap_state()

# ========================================================================== #
    @property
    def nsteps(self):
        return self._nsteps

# ========================================================================== #
    @nsteps.setter
    def nsteps(self, nsteps):
        self._nsteps = nsteps

# ========================================================================== #
    @property
    def nsteps_eq(self):
        return self._nsteps_eq

# ========================================================================== #
    @nsteps_eq.setter
    def nsteps_eq(self, nsteps_eq):
        self._nsteps_eq = nsteps_eq

# ========================================================================== #
    @property
    def logfile(self):
        return self._logfile

# ========================================================================== #
    @logfile.setter
    def logfile(self, logfile):
        self._logfile = logfile

# ========================================================================== #
    @property
    def trajfile(self):
        return self._trajfile

# ========================================================================== #
    @trajfile.setter
    def trajfile(self, trajfile):
        self._trajfile = trajfile

# ========================================================================== #
    @property
    def loginterval(self):
        return self._loginterval

# ========================================================================== #
    @loginterval.setter
    def loginterval(self, loginterval):
        self._loginterval = loginterval
