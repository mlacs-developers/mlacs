import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import StateManager

default_langevin = {'friction': 0.01}

#========================================================================================================================#
#========================================================================================================================#
class LangevinState(StateManager):
    """
    State Class for running a Langevin simulation as implemented in ASE

    Parameters
    ----------

    temperature : float
        Temperature of the simulation, in Kelvin
    friction : float
        Friction coefficient of the thermostat
    dt : float
        Timestep, in fs
    nsteps : int
        Number of MLMD steps for production runs
    nsteps_eq : int
        Number of MLMD steps for equilibration runs
    fixcm : bool
        Fix position and momentum center of mass
    logfile : str
        Name of the file for logging the MLMD trajectory
    trajfile : str
        Name of the file for saving the MLMD trajectory
    interval : int
        Number of steps between log and traj writing. Override
        loginterval and trajinterval
    loginterval : int
        Number of steps between MLMD logging
    trajinterval : int
        Number of steps between MLMD traj writing
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to numpy.random.default_rng()
    init_momenta : array (optional)
        If None, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
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

        self.temperature = temperature
        self.friction    = friction
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
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

        dyn = Langevin(atoms, self.dt, temperature_K=self.temperature, friction=self.friction, fixcm=self.fixcm)

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
        msg  = "Simulated state:\n"
        msg += "NVT ensemble witht the Langevin thermostat\n"
        msg += "parameters:\n"
        msg += "timestep     {:} fs\n".format(self.dt / fs)
        msg += "friction     {:}\n".format(self.friction)
        msg += "\n"
        return msg
