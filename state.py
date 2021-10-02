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
                 dynamics=None,
                 pressure=None,
                 dt=1.5*fs,
                 nsteps=5000,
                 dyn_parameters=None,
                 nthrow=2500,
                 verbose=True
                ):

        self.temperature = temperature
        self.pressure    = pressure
        self.dt          = dt
        self.nsteps      = nsteps

        if dynamics is None:
            self.dynamics = Langevin
            if dyn_parameters is None:
                dyn_parameters = {'friction': 0.01}
        else:
            self.dynamics = dynamics
        self.dyn_parameters = dyn_parameters


#========================================================================================================================#
    def run_dynamics(self, supercell, calc, nsteps=None, logfile=None, trajfname=None):
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
