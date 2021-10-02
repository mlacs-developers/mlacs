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
    Class managing a state being simulated
    """
    def __init__(self,
                 temperature,
                 pressure=None,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=250,
                 dyn_parameters=None,
                 logfile=None,
                 trajfname=None
                ):

        self.temperature    = temperature
        self.pressure       = pressure
        self.dt             = dt
        self.nsteps         = nsteps
        self.nsteps_eq      = nsteps_eq
        self.dyn_parameters = dyn_parameters
        self.logfile        = logfile
        self.trajfname      = trajfname


#========================================================================================================================#
    def run_dynamics(self, supercell, calc, eq=False, logfile=None, trajfname=None):
        """
        Run the dynamics for the state, during nsteps
        """
        atoms      = supercell.copy()
        atoms.calc = calc

        if nsteps is None:
            nsteps = self.nsteps
        else:
            nsteps = nsteps

        trajectory = None
        if trajfname is not None:
            trajectory = Trajectory(trajfname, mode="r", atoms=atoms)
        
        if self.dyn_parameters is None:
            dyn = self.dynamics(atoms=atoms, timestep=self.dt, temperature_K=self.temperature, trajectory=trajectory, logfile=logfile)
        else:
            dyn = self.dynamics(atoms=atoms, timestep=self.dt, temperature_K=self.temperature, trajectory=trajectory, logfile=logfile, **self.dyn_parameters)
        dyn.run(nsteps)

        return dyn.atoms


#========================================================================================================================#
    def get_initial_momenta(self, atoms):
        """
        """
        momenta = np.zeros((len(atoms), 3))
        atoms.set_momenta(momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        return ""
