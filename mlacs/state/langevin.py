import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import StateManager

#========================================================================================================================#
#========================================================================================================================#
class LangevinState(StateManager):
    """
    State Class for running a Langevin simulation as implemented in ASE

    Parameters
    ----------

    temperature : float
        Temperature of the simulation, in Kelvin
    friction : float (optional)
        Friction coefficient of the thermostat. Default 0.01.
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
    workdir : str (optional)
        Working directory for the LAMMPS MLMD simulations. If none, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 temperature,
                 friction=0.01,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
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

        self.temperature  = temperature
        self.friction     = friction
        self.init_momenta = init_momenta
        self.rng          = rng
        if self.rng is None:
            self.rng = np.random.default_rng()


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

        dyn = Langevin(atoms, self.dt, temperature_K=self.temperature, friction=self.friction, fixcm=self.fixcm, rng=self.rng)

        if self.trajfile is not None:
            trajectory = Trajectory(self.trajfile, mode="a", atoms=atoms)
            dyn.attach(trajectory.write, interval=self.trajinterval)

        if self.logfile is not None:
            dyn.attach(MDLogger(dyn, atoms, self.logfile, stress=True), interval=self.loginterval)

        dyn.run(nsteps)
        return dyn.atoms


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature, rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        msg  = "Simulated state :\n"
        msg += "NVT Langevin dynamics as implemented in ASE\n"
        msg += "Temperature (in Kelvin)                  {0}\n".format(self.temperature)
        msg += "Number of MLMD equilibration steps :     {0}\n".format(self.nsteps_eq)
        msg += "Number of MLMD production steps :        {0}\n".format(self.nsteps)
        msg += "Timestep (in fs) :                       {0}\n".format(self.dt / fs)
        msg += "Friction (in fs) :                       {:}\n".format(self.friction / fs)
        msg += "\n"
        return msg
