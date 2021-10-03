import numpy as np

from otf_mlacs.path_integral.pimd import PathIntegralDynamics
from ase import units
from ase.parallel import world, DummyMPI


class PathIntegralLangevin(PathIntegralMolecularDynamics):
    """
    Langevin thermostat for path integral molecular dynamics

    from A simple and accurate algorithm for path integral molecular dynamics with the Langevin thermostat
         J.Liu, D.Li and Liu,X  J. Chem. Phys 145 024103 (2016)
    """
    def __init__(self, qpolymer, timestep, temperature=None, friction=None,
                 fixcm=True, *, trajectory=None, logfile=None, loginterval=1, rng=None,
                 append_trajectory=False):
        """
        """
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.kBT = units.kB * temperature
        self.fix_com = fixcm
        if rng is None:
            self.rng = np.random.default_rng()
        PathIntegralDynamics.__init__(self, qpolymer, timestep, trajectory,
                                      logfile, loginterval,
                                      append_trajectory=append_trajectory)
        self.updatevars()


    def set_temperature(self, temperature):
        """
        """
        self.kBT = units.kB * temperature
        self.updatevars()
        self.qpolymer.set_temperature(temperature)
        

    def set_friction(self, friction):
        """
        """
        self.fr = friction
        self.updatevars()


    def set_timestep(self, timestep):
        """
        """
        self.dt = timestep
        self.updatevars()


    def updatevars(self):
        """
        """
        dt     = self.dt
        kBT    = self.kBT
        fr     = self.fr
        masses = self.masses


    def step(self, forces=None):
        """
        """
        qpolymer = self.qpolymer
        natoms   = qpolymer.natoms
        nbeads   = qpolymer.nbeads

        if forces is None:
            forces = qpolymer.get_forces(md=True)

        v = qpolymer.get_velocities()
