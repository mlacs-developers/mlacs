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
    State class for running a NVE simulation using a velocity verlet integrator, as implemented in ASE

    Parameters
    ----------
    dt : float (optional)
        Timestep, in fs. Default 1.5 fs.
    nsteps : int (optional)
        Number of MLMD steps for production runs. Default 1000 steps.
    nsteps_eq : int (optional)
        Number of MLMD steps for equilibration runs. Default 100 steps.
    fixcm : bool (optional)
        Fix position and momentum center of mass. Default True.
    logfile : str (optional)
        Name of the file for logging the MLMD trajectory.
        If none, no log file is created. Default None.
    trajfile : str (optional)
        Name of the file for saving the MLMD trajectory.
        If none, no traj file is created. Default None.
    interval : int (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default 50
    loginterval : int (optional)
        Number of steps between MLMD logging. Default 50.
    trajinterval : int (optional)
        Number of steps between MLMD traj writing. Default 50.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to numpy.random.default_rng()
    init_momenta : array (optional)
        If None, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
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

        dyn = VelocityVerlet(atoms, self.dt * fs)

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
        msg  = "Simulated state :\n"
        msg += "NVE ensemble with the Velocity-Verlet integrator as implemented in ASE\n"
        msg += "Number of MLMD equilibration steps :     {0}\n".format(self.nsteps_eq)
        msg += "Number of MLMD production steps :        {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt)
        msg += "\n"
        return msg
