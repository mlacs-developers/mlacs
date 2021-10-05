import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.verlet import VelocityVerlet
from ase.md import MDLogger

from mlacs.state import StateManager


#========================================================================================================================#
#========================================================================================================================#
class VerletState(StateManager):
    """
    """
    def __init__(self,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 init_momenta=None
                ):

        StateManager.__init__(self,
                              dt,
                              nsteps,
                              nsteps_eq,
                              fixcm,
                              logfile,
                              trajfile,
                              loginterval,
                              trajinterval
                             )
        self.init_momenta = init_momenta



#========================================================================================================================#
    def run_dynamics(self, supercell, calc, eq=False):
        """
        """
        atoms      = supercell.copy()
        atoms.calc = calc

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        dyn = VelocityVerlet(atoms, self.dt)

        if self.trajfile is not None:
            trajectory = Trajectory(self.trajfile, mode="a", atoms=atoms)
            dyn.attach(trajectory.write)

        if self.logfile is not None:
            dyn.attach(MDLogger(dyn, atoms, self.logfile, stress=True))

        dyn.run(nsteps)
        return dyn.atoms


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is not None:
            atoms.set_momenta(self.init_momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        msg  = "Simulated state:\n"
        msg += "NVE ensemble\n"
        msg += "parameters:\n"
        msg += "timestep     {:} fs\n".format(self.dt / fs)
        msg += "\n"
        return msg
