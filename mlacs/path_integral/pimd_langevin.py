import numpy as np

from mlacs.path_integral.pimd import PathIntegralMolecularDynamics
from ase import units
from ase.parallel import world, DummyMPI


class PathIntegralLangevin(PathIntegralMolecularDynamics):
    """
    Langevin thermostat for path integral molecular dynamics

    The thermostat used are from 'A simple and accurate algorithm 
    for path integral molecular dynamics with the Langevin thermostat'
    
    J.Liu, D.Li and X,Liu  J. Chem. Phys **145** 024103 (2016)


    Parameters
    ----------

    qpolymer : :class: `mlacs.path_integral.qpolymer`
        The qPolymer object
    timestep : float
        The time step
    temperature : float
        The temperature in Kelvin
    fixcm : bool
        if True, the positions and momentum of the center of mass is kept unperturbed. Default True
    integrator : str
        Can be 'BAOAB', 'OBABO', 'BAOAB-num' or 'OBABO-num'. Default 'BAOAB-num'
    trajectory : PathIntegralTrajectory object or str
        Attach a PathIntegralTrajectory object. Default None (no trajectory)
    logfile : str
        Attach a logger for the trajectory
    rng : RNG object
        Random number generator. By default np.random.default_rng()
    loginterval : int
    """
    def __init__(self, qpolymer, timestep, temperature=None, 
                 fixcm=True, integrator="BAOAB-num", *, trajectory=None, logfile=None, loginterval=1, rng=None,
                 append_trajectory=False):
        """
        """
        self.kBT = units.kB * temperature
        self.fixcm = fixcm
        if rng is None:
            self.rng = np.random.default_rng()
        PathIntegralMolecularDynamics.__init__(self, qpolymer, timestep, trajectory,
                                               logfile, loginterval,
                                               append_trajectory=append_trajectory)
        #self.hfreq   = 1e-3
        self.hfreq   = qpolymer.hfreq
        self.tmasses = self.qpolymer.get_staging_masses()
        self.tmasses_half = self.tmasses.copy()
        self.tmasses_half[0] = 0.0
        self.integrator = integrator
        self.updatevars()


    def set_temperature(self, temperature):
        """
        """
        self.kBT = units.kB * temperature
        self.updatevars()
        self.qpolymer.set_temperature(temperature)


    def set_timestep(self, timestep):
        """
        """
        self.dt = timestep
        self.updatevars()


    def updatevars(self):
        """
        """
        self.c1    = np.exp(-self.hfreq * self.dt)
        self.c2    = np.sqrt(1 - self.c1**2)
        self.cos_a = np.cos(0.5 * self.hfreq * self.dt)
        self.sin_a = np.sin(0.5 * self.hfreq * self.dt)


    def step(self, tforces=None):
        """
        """
        if self.integrator == "BAOAB-num":
            self._BAOAB_num_integrator(tforces)
        elif self.integrator == "BAOAB":
            self._BAOAB_integrator(tforces)
        elif self.integrator == "OBABO-num":
            self._OBABO_num_integrator(tforces)
        elif self.integrator == "OBABO":
            self._OBABO_integrator(tforces)
        else:
            raise NotImplementedError("Only 'BAOAB-num', 'BAOAB', 'OBABO-num' and 'OBABO' integrators are implemented")
    

    def _OBABO_integrator(self, tforces):
        """
        """
        qpolymer = self.qpolymer
        tmasses  = self.tmasses

        eta = self.rng.normal(size=(qpolymer.nbeads, qpolymer.natoms, 3))

        if tforces is None:
            tforces = qpolymer.get_staging_forces()

        tx = qpolymer.get_staging_positions()
        p  = qpolymer.get_momenta()

        p  = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta
        p += 0.5 * self.dt * tforces

        tx[0] += self.dt * p[0] / tmasses[0,:,None]

        p_tmp  = p[1:]
        tx_tmp = tx[1:]

        tx[1:] = self.cos_a * tx_tmp + self.sin_a * p_tmp / (self.hfreq * tmasses[1:,:,None])
        p[1:]  = self.cos_a * p_tmp  - self.hfreq * self.sin_a * tx_tmp * tmasses[1:,:,None]

        if self.fixcm:
            old_cm = qpolymer.get_center_of_mass()
        qpolymer.set_positions_from_staging_transform(tx)
        if self.fixcm:
            qpolymer.set_center_of_mass(old_cm)

        tforces = qpolymer.get_staging_forces()

        p += 0.5 * self.dt * tforces
        p  = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta

        if self.fixcm:
            p -= self._get_com_velocity(p / tmasses[:,:,None])[:,None,:] * tmasses[:,:,None]

        qpolymer.set_positions_from_staging_transform(tx)
        qpolymer.set_momenta(p)


    def _BAOAB_integrator(self, tforces):
        """
        """
        qpolymer = self.qpolymer
        tmasses  = self.tmasses

        eta = self.rng.normal(size=(qpolymer.nbeads, qpolymer.natoms, 3))

        if tforces is None:
            tforces = qpolymer.get_staging_forces()

        tx = qpolymer.get_staging_positions()
        p  = qpolymer.get_momenta()

        # first half
        p     += 0.5 * tforces * self.dt
        tx[0] += 0.5 * self.dt * p[0] / tmasses[0,:,None]

        p_tmp  = p[1:]
        tx_tmp = tx[1:]

        tx[1:] = self.cos_a * tx_tmp + self.sin_a * p_tmp / (self.hfreq * tmasses[1:,:,None])
        p[1:]  = self.cos_a * p_tmp  - self.hfreq * self.sin_a * tx_tmp * tmasses[1:,:,None]

        p      = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta
        tx[0] += 0.5 * self.dt * p[0] / tmasses[0,:,None]

        p_tmp = p[1:]
        tx_tmp = tx[1:]

        tx[1:] = self.cos_a * tx_tmp + self.sin_a * p_tmp / (self.hfreq * tmasses[1:,:,None])
        p[1:]  = self.cos_a * p_tmp  - self.hfreq * self.sin_a * tx_tmp * tmasses[1:,:,None]

        if self.fixcm:
            old_cm = qpolymer.get_center_of_mass()
        qpolymer.set_positions_from_staging_transform(tx)
        if self.fixcm:
            qpolymer.set_center_of_mass(old_cm)

        tforces = qpolymer.get_staging_forces()

        p += 0.5 * tforces * self.dt

        if self.fixcm:
            p -= self._get_com_velocity(p / tmasses[:,:,None])[:,None,:] * tmasses[:,:,None]

        qpolymer.set_positions_from_staging_transform(tx)
        qpolymer.set_momenta(p)


    def _BAOAB_num_integrator(self, tforces):
        """
        """
        qpolymer     = self.qpolymer
        tmasses      = self.tmasses
        tmasses_half = self.tmasses_half

        eta = self.rng.normal(size=(qpolymer.nbeads, qpolymer.natoms, 3))

        if tforces is None:
            tforces = qpolymer.get_staging_forces()

        tx = qpolymer.get_staging_positions()
        p  = qpolymer.get_momenta()

        p  += 0.5 * self.dt * (tforces - self.hfreq**2 * tmasses_half[:,:,None] * tx)
        tx += 0.5 * self.dt * p / tmasses[:,:,None]

        p   = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta
        tx += 0.5 * self.dt * p / tmasses[:,:,None]

        if self.fixcm:
            old_cm = qpolymer.get_center_of_mass()
        qpolymer.set_positions_from_staging_transform(tx)
        if self.fixcm:
            qpolymer.set_center_of_mass(old_cm)

        tforces = qpolymer.get_staging_forces()

        p  += 0.5 * self.dt * (tforces - self.hfreq**2 * tmasses_half[:,:,None] * tx)

        if self.fixcm:
            p -= self._get_com_velocity(p / tmasses[:,:,None])[:,None,:] * tmasses[:,:,None]

        qpolymer.set_positions_from_staging_transform(tx)
        qpolymer.set_momenta(p)



    def _OBABO_num_integrator(self, tforces):
        """
        """
        qpolymer     = self.qpolymer
        tmasses      = self.tmasses
        tmasses_half = self.tmasses_half

        eta = self.rng.normal(size=(qpolymer.nbeads, qpolymer.natoms, 3))

        if tforces is None:
            tforces = qpolymer.get_staging_forces()

        tx = qpolymer.get_staging_positions()
        p  = qpolymer.get_momenta()

        p  = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta
        p += 0.5 * self.dt * (tforces - self.hfreq**2 * tmasses_half[:,:,None] * tx)

        tx += self.dt * p / tmasses[:,:,None]
        if self.fixcm:
            old_cm = qpolymer.get_center_of_mass()
        qpolymer.set_positions_from_staging_transform(tx)
        if self.fixcm:
            qpolymer.set_center_of_mass(old_cm)

        tforces = qpolymer.get_staging_forces()

        p += 0.5 * self.dt * (tforces - self.hfreq**2 * tmasses_half[:,:,None] * tx)
        p  = self.c1 * p + self.c2 * np.sqrt(tmasses[:,:,None] * self.kBT) * eta

        if self.fixcm:
            p -= self._get_com_velocity(p / tmasses[:,:,None])[:,None,:] * tmasses[:,:,None]

        qpolymer.set_positions_from_staging_transform(tx)
        qpolymer.set_momenta(p)
