import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from otf_mlacs.state import StateManager

default_langevin = {'friction': 0.01}

#========================================================================================================================#
#========================================================================================================================#
class LangevinState(StateManager):
    """
    """
    def __init__(self,
                 temperature,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=250,
                 dyn_parameters=default_langevin,
                 logfile=None,
                 trajfname=None,
                 rng=None,
                 init_momenta=True
                ):
        
        StateManager.__init__(self,
                              temperature,
                              None,
                              dt,
                              nsteps,
                              nsteps_eq,
                              dyn_parameters,
                              logfile,
                              trajfname
                             )
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

        if self.dyn_parameters is not None:
            dyn = Langevin(atoms, self.dt, temperature_K=self.temperature, rng=self.rng, **self.dyn_parameters)
        else:
            dyn = Langevin(atoms, self.dt, temperature_K=self.temperature, rng=self.rng)

        if self.trajfname is not None:
            trajectory = Trajectory(trajfname, mode="r", atoms=atoms)
            dyn.attach(trajectory.write)

        if self.logfile is not None:
            dyn.attach(MDLogger(dyn, atoms, self.logfile, stress=True))

        dyn.run(nsteps)
        return dyn.atoms


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature, rng=self.rng)
        else:
            momenta = np.zeros((len(atoms), 3))
            atoms.set_momenta(momenta)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        msg  = "Simulated state:\n"
        msg += "NVT ensemble witht the Langevin thermostat\n"
        msg += "parameters:\n"
        msg += "timestep     {:} fs\n".format(self.dt / fs)
        for key in self.dyn_parameters.keys():
            msg += key + "     {:}\n".format(self.dyn_parameters[key])
        msg += "\n"
        return msg
