"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin



#========================================================================================================================#
#========================================================================================================================#
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
                 interval=50,
                 loginterval=50,
                 trajinterval=50
                ):

        self.dt             = dt
        self.nsteps         = nsteps
        self.nsteps_eq      = nsteps_eq
        self.fixcm          = fixcm
        self.logfile        = logfile
        self.trajfile       = trajfile
        self.interval       = interval
        self.loginterval    = loginterval
        self.trajinterval   = trajinterval
        if self.interval is not None:
            self.loginterval    = interval
            self.trajinterval   = interval

        self.isnpt          = False
        self.islammps       = False


#========================================================================================================================#
    def run_dynamics(self, supercell, calc, eq=False, logfile=None, trajfile=None):
        """
        Run the dynamics for the state, during nsteps then return the last atoms of the simulation
        """
        raise NotImplementedError(msg)


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        Setup the momenta during initialization of the simulation
        """
        if not atoms.has("momenta"):
            momenta = np.zeros((len(atoms), 3))
            atoms.set_momenta(momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        return ""
