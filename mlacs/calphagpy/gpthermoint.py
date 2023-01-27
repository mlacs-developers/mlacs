import numpy as np
from scipy.integrate import quad
try:
    from sklearn.gaussian_process.kernels import (RBF,
                                                  ConstantKernel as C)
except ImportError:
    msg = "You need sklearn to use the calphagpy modules"
    raise ModuleNotFoundError(msg)

from .gpinterface import GaussianProcessInterface


# ========================================================================== #
# ========================================================================== #
class GpThermoInt(GaussianProcessInterface):
    """
    """
    def __init__(self,
                 name,
                 ref_free_energy,
                 ref_state,
                 mode="nvt",
                 kernel=RBF()*C(),
                 gp_parameters={}):
        self.mode = mode
        if self.mode == "nvt":
            ndim = 1
            self.temp0 = ref_state
        elif self.mode == "npt":
            ndim = 2
            print(ref_state)
            self.temp0 = ref_state[0]
            self.vol0 = ref_state[1]
            msg = "The NPT mode is not implemented yet"
            #raise NotImplementedError(msg)

        GaussianProcessInterface.__init__(self,
                                          name=name,
                                          ndim=ndim,
                                          kernel=kernel,
                                          gp_parameters=gp_parameters)
        self.f0 = ref_free_energy

# ========================================================================== #
    def get_free_energy(self, state):
        """
        """
        if self.mode == "nvt":
            if isinstance(state, (float, int)):
                state = np.array([state])
            else:
                state = np.array(state)
        elif self.mode == "npt":
            if isinstance(state, (list)):
                state = np.array(state)
            if len(state.shape) == 1:
                temp = np.array([state[0]])
                vol = np.array([state[1]])
                state = np.zeros((1, 2))
                state[0] = [temp, vol]
            else:
                temp = state[:, 0]
                vol = state[:, 1]

        nstate = state.shape[0]
        fe = np.zeros(nstate)
        for i, s in enumerate(state):
            fe[i] = self._get_free_energy_onestate(s)
        return fe

# ========================================================================== #
    def _get_free_energy_onestate(self, state):
        """
        """
        if self.mode == "nvt":
            temp = state
        elif self.mode == "npt":
            temp = state[0]
            vol = state[1]

        # We start from the reference free energy
        fe = self.f0

        # First temperature integration
        delta_fe = self._temperature_integration(temp)
        fe = (self.f0 / self.temp0 - delta_fe) * temp

        # If needed, we continue with volume integration
        if self.mode == "npt":
            delta_fe = self._volume_integration(vol, temp)
            fe -= delta_fe

        return fe

# ========================================================================== #
    def _temperature_integration(self, temp):
        """
        """
        delta_fe = quad(self._integrand_temp, self.temp0, temp)[0]
        return delta_fe

# ========================================================================== #
    def _integrand_temp(self, temp):
        """
        """
        if self.mode == "nvt":
            temp = np.atleast_2d(temp).T
            return self.predict(temp) / temp**2
        elif self.mode == "npt":
            state = np.array([temp, self.vol0])
            state = state.reshape(1, -1)
            return self.gp.predict(state)[0][0] / temp**2

# ========================================================================== #
    def _volume_integration(self, vol, temp):
        """
        """
        delta_fe = quad(self._integrand_vol, self.vol0,
                        vol, args=(temp))[0]
        return delta_fe

# ========================================================================== #
    def _integrand_vol(self, vol, temp):
        """
        """
        x = np.array([temp, vol])
        x = x.reshape(1, -1)
        return self.predict(x)[0][1]
