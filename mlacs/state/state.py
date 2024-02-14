"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path

import numpy as np


# ========================================================================== #
# ========================================================================== #
class StateManager:
    """
    Parent Class managing the state being simulated
    """
    def __init__(self,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 workdir=None):

        self.dt = dt
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.fixcm = fixcm
        self.logfile = logfile
        self.trajfile = trajfile
        self.loginterval = loginterval

        if workdir is None:
            workdir = "MolecularDynamics"
        self.workdir = Path(workdir).absolute()

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post,
                     atom_style="atomic",
                     eq=False):
        """
        Run the dynamics for the state, during nsteps
        then return the last atoms of the simulation
        """
        raise NotImplementedError

# ========================================================================== #
    def initialize_momenta(self, atoms):
        """
        Setup the momenta during initialization of the simulation
        """
        if not atoms.has("momenta"):
            momenta = np.zeros((len(atoms), 3))
            atoms.set_momenta(momenta)

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        return ""

# ========================================================================== #
    def set_workdir(self, workdir):
        """
        """
        self.workdir = Path(workdir).absolute()
