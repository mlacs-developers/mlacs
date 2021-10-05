
import warnings
import numpy as np

from mlacs.path_integral.pid import PathIntegralDynamics
from mlacs.path_integral.logger import PIMDLogger


class PathIntegralMolecularDynamics(PathIntegralDynamics):
    """
    """
    def __init__(self, qpolymer, timestep, trajectory, logfile=None, loginterval=1, append_trajectory=False):
        """
        """
        self.dt = timestep

        Dynamics.__init__(self, qpolymer, logfile=None, trajectory=None)

        self.masses = qpolymer.get_masses()
        self.max_steps = None

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (1, -1, 1)

        if not self.qpolymer.has('momenta'):
            self.qpolymer.set_momenta(np.zeros([len(self.atoms), 3]))

        if logfile:
            logger = self.closelater(MDLogger(dyn=self, qpolymer=qpolymer, logfile=logfile))
            self.attach(logger, loginterval)

    def irun(self, steps=50):
        """ Call Dynamics.irun and adjust max_steps """
        self.max_steps = steps + self.nsteps
        return PathIntegralDynamics.irun(self)


    def run(self, steps=50):
        """
        """
        self.max_steps = steps + self.nsteps
        return PathIntegralDynamics.run(self)


    def get_time(self):
        return self.nsteps * self.dt


    def converged(self):
        """
        """
        return self.nsteps >= self.max_steps


    def _get_com_velocity(self, velocity):
        """
        """
        return np.dot(self.masses.ravel(), velocity) / self.masses.sum()
