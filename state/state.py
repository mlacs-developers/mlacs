"""
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
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=250,
                 dyn_parameters=None,
                 logfile=None,
                 trajfname=None
                ):

        self.dt             = dt
        self.nsteps         = nsteps
        self.nsteps_eq      = nsteps_eq
        self.dyn_parameters = dyn_parameters
        self.logfile        = logfile
        self.trajfname      = trajfname


#========================================================================================================================#
    def run_dynamics(self, supercell, calc, eq=False, logfile=None, trajfname=None):
        """
        Run the dynamics for the state, during nsteps then return the last atoms of the simulation
        """
        raise NotImplementedError(msg)


#========================================================================================================================#
    def get_initial_momenta(self, atoms):
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
