"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from ase.units import GPa
try:
    from sklearn.gaussian_process.kernels import (RBF,
                                                  ConstantKernel as C,
                                                  WhiteKernel)
except ImportError:
    msg = "You need sklearn to use the calphagpy modules"
    raise ModuleNotFoundError(msg)

from .gpinterface import GaussianProcessInterface


available_modes = ["t", "vt"]


# ========================================================================== #
# ========================================================================== #
class GpThermoIntT(GaussianProcessInterface):
    """
    Class to do thermodynamic integration with gaussian process on the
    temperature dimension.


    Parameters
    ----------

    ref_free_energy: :class:`float`
        The free energy at the reference point, in eV/at
    ref_temperature: :class:`float`
        The temperature at the reference point, in K
    kernel: :class:`Kernel`
        The kernel of the gaussian process.
        Default are :
        RBF() * C() + WhiteKernel()
    gp_parameters: :class:`dict`
        Parameters for the GaussianProcessRegressor object.
        The default parameters are :
        {"n_restart_optimizer": 100, "normalyze_y": True, "alpha": 1e-10}
    """
    def __init__(self,
                 ref_free_energy,
                 ref_temperature,
                 kernel=RBF()*C()+WhiteKernel(),
                 gp_parameters={}):

        GaussianProcessInterface.__init__(self,
                                          ndim=1,
                                          kernel=kernel,
                                          gp_parameters=gp_parameters)
        self.f0 = ref_free_energy
        self.t0 = ref_temperature

# ========================================================================== #
    def add_new_data(self, temperature, energy):
        """
        Add data point to train the gaussian process

        Parameters
        ----------

        temperature: :class:`np.ndarray`
            The temperature points, in K
        energy: :class:`float` or :class:`np.ndarray`
            The energy, in eV/at
        """
        if len(temperature.shape) == 1:
            temperature = temperature.reshape(-1, 1)
        self._add_new_data(temperature, energy)

# ========================================================================== #
    def get_helmholtz_free_energy(self, temperature):
        """
        Function to get the Helmholtz free energy at a given temperature.

        Parameters
        ----------

        temperature: :class:`float` or :class:`np.ndarray`
            The temperature at which to compute the free energy, in K

        Return
        ------

        fe: :class:`float` or :class:`np.ndarray`
            The free energy at the desired temperature, in eV/at
        """
        if isinstance(temperature, (float, int)):
            temperature = np.array([temperature])

        # Due to the integration, we have to launch one state after the other
        fe = np.zeros(temperature.shape[0])
        for i, t in enumerate(temperature):
            fe[i] = self._get_helmholtz_onestate(t)
        return fe

# ========================================================================== #
    def _get_helmholtz_onestate(self, temperature):
        """
        """
        # We start with temperature integration
        delta_fe = self._temperature_integration(temperature)
        fe = (self.f0 / self.t0 - delta_fe) * temperature
        return fe

# ========================================================================== #
    def _temperature_integration(self, temp):
        """
        """
        delta_fe = quad(self._integrand_temp, self.t0, temp)[0]
        return delta_fe

# ========================================================================== #
    def _integrand_temp(self, temp):
        """
        """
        temp = np.atleast_2d(temp).T
        return self.predict(temp) / temp**2


# ========================================================================== #
# ========================================================================== #
class GpThermoIntVT(GaussianProcessInterface):
    """
    Class to do thermodynamic integration with gaussian process on the
    volume/temperature dimensions.
    Can compute the Helmholtz and Gibbs free energies.


    Parameters
    ----------

    ref_free_energy: :class:`float`
        The free energy at the reference point
    ref_volume: :class:`float`
        The volume at the reference point
    ref_temperature: :class:`float`
        The temperature at the reference point
    kernel: :class:`Kernel`
        The kernel of the gaussian process.
        Default are :
        RBF() * C() + WhiteKernel()
    gp_parameters: :class:`dict`
        Parameters for the GaussianProcessRegressor object.
        The default parameters are :
        {"n_restart_optimizer": 100, "normalyze_y": True, "alpha": 1e-10}
    """
    def __init__(self,
                 ref_free_energy,
                 ref_volume,
                 ref_temperature,
                 kernel=RBF([1.0, 1.0])*C()+WhiteKernel(),
                 gp_parameters={}):

        GaussianProcessInterface.__init__(self,
                                          ndim=2,
                                          kernel=kernel,
                                          gp_parameters=gp_parameters)

        # Now get the reference state
        self.f0 = ref_free_energy
        self.v0 = ref_volume
        self.t0 = ref_temperature

        self.lb = 15
        self.ub = 23

# ========================================================================== #
    def add_new_data(self, volume, temperature, energy, pressure):
        """
        Add data point to train the gaussian process

        Parameters
        ----------

        volume: :class:`float` or :class:`np.ndarray`
            The volume points, in angstrom**3
        temperature: :class:`float` or :class:`np.ndarray`
            The temperature points, in K
        energy: :class:`float` or :class:`np.ndarray`
            The energy, in eV/at
        pressure: :class:`float` or :class:`np.ndarray`
            The pressure, in GPa
        """
        x = np.c_[volume, temperature]
        y = np.c_[energy, pressure * GPa]
        self._add_new_data(x, y)

# ========================================================================== #
    def get_helmholtz_free_energy(self, volume, temperature):
        """
        Function to get the Helmholtz free energy at a given volume
        and temperature.

        Parameters
        ----------

        volume: :class:`float` or :class:`np.ndarray`
            The volume at which to compute the free energy, in angstrom**3
        temperature: :class:`float` or :class:`np.ndarray`
            The temperature at which to compute the free energy, in K

        Return
        ------

        fe: :class:`float` or :class:`np.ndarray`
            The free energy at the desired temperature, in eV/at
        """
        state = np.array([volume, temperature])
        state = state.reshape(1, -1)

        # Due to the integration, we have to launch one state after the other
        fe = np.zeros(state.shape[0])
        for i, s in enumerate(state):
            fe[i] = self._get_helmholtz_onestate(s)
        return fe

# ========================================================================== #
    def get_gibbs_free_energy(self, pressure, temperature, lb=None, ub=None):
        """
        Function to get the Gibbs free energy at a given pressure
        and temperature.

        Parameters
        ----------

        pressure: :class:`float` or :class:`np.ndarray`
            The pressure at which to compute the free energy, in GPa
        temperature: :class:`float` or :class:`np.ndarray`
            The temperature at which to compute the free energy, in K

        Return
        ------

        fe: :class:`float` or :class:`np.ndarray`
            The free energy at the desired temperature, in eV/at
        """
        state = np.array([pressure * GPa, temperature])
        state = state.reshape(1, -1)

        # Due to the integration, we have to launch one state after the other
        gb = np.zeros(state.shape[0])
        for i, s in enumerate(state):
            gb[i] = self._get_gibbs_onestate(s, lb, ub)
        return gb

# ========================================================================== #
    def get_volume_from_press_temp(self, pressure, temperature,
                                   lb=None, ub=None):
        """
        Function to get the volume corresponding to given temperature
        and pressure points

        Parameters
        ----------

        pressure: :class:`float` or :class:`np.ndarray`
            The pressure at which to compute the free energy, in GPa
        temperature: :class:`float` or :class:`np.ndarray`
            The temperature at which to compute the free energy, in K

        Return
        ------

        vol: :class:`float` or :class:`np.ndarray`
            The volume at the desired pressure/temperature points,
            in angstrom**3
        """
        state = np.array([pressure * GPa, temperature])
        state = state.reshape(1, -1)

        vol = np.zeros(state.shape[0])
        for i, s in enumerate(state):
            vol[i] = self._get_volume_from_one_pt(s[0], s[1], lb, ub)
        return vol

# ========================================================================== #
    def get_thermal_expansion(self, pressure, temperature,
                              step=1e-8, lb=None, ub=None):
        """
        Function to get the thermal expansion coefficient at given temperature
        and pressure points

        Parameters
        ----------

        pressure: :class:`float` or :class:`np.ndarray`
            The pressure at which to compute the free energy, in GPa
        temperature: :class:`float` or :class:`np.ndarray`
            The temperature at which to compute the free energy, in K

        Return
        ------

        alpha: :class:`float` or :class:`np.ndarray`
            The thermal expansion coefficient  at the desired
            pressure/temperature points, in 1/K
        """
        state = np.array([pressure * GPa, temperature])
        state = state.reshape(1, -1)

        alpha = np.zeros(state.shape[0])
        for i, s in enumerate(state):
            alpha[i] = self._get_thermal_expansion_onestate(s, step, lb, ub)
        return alpha

# ========================================================================== #
    def _get_helmholtz_onestate(self, state):
        """
        """
        # First check integration mode
        vol = state[0]
        temp = state[1]

        # We start with temperature integration
        delta_fe = self._temperature_integration(temp)
        fe = (self.f0 / self.t0 - delta_fe) * temp

        # Now we do the volume integration
        delta_fe = self._volume_integration(vol, temp)
        fe = fe - delta_fe

        return fe

# ========================================================================== #
    def _temperature_integration(self, temp):
        """
        """
        delta_fe = quad(self._integrand_temp, self.t0, temp)[0]
        return delta_fe

# ========================================================================== #
    def _integrand_temp(self, temp):
        """
        """
        state = np.array([self.v0, temp])
        state = state.reshape(1, -1)
        return self.gp.predict(state)[0, 0] / temp**2

# ========================================================================== #
    def _volume_integration(self, vol, temp):
        """
        """
        delta_fe = quad(self._integrand_vol, self.v0,
                        vol, args=(temp))[0]
        return delta_fe

# ========================================================================== #
    def _integrand_vol(self, vol, temp):
        """
        """
        x = np.array([vol, temp])
        x = x.reshape(1, -1)
        return self.predict(x)[0, 1]

# ========================================================================== #
    def _get_gibbs_onestate(self, state, lb, ub):
        """
        """
        # First check integration mode
        pres = state[0]
        temp = state[1]

        # First we need to get the volume for this pressure/temperature point
        vol = self._get_volume_from_one_pt(pres, temp, lb, ub)

        fe = self.get_helmholtz_free_energy(vol, temp)

        gb = fe + pres * vol
        return gb

# ========================================================================== #
    def _get_volume_from_one_pt(self, pres, temp, lb=None, ub=None):
        """
        """
        if lb is None:
            lb = self.lb[0]
        if ub is None:
            ub = self.ub[0]
        vol = brentq(self._p_minus_ptarget,
                     lb, ub, args=(pres, temp))
        return vol

# ========================================================================== #
    def _p_minus_ptarget(self, vol, ptarget, temp):
        """
        """
        state = np.array([vol, temp])
        state = state.reshape(1, -1)
        res = self.gp.predict(state)
        return ptarget - res[0, 1]

# ========================================================================== #
    def _get_thermal_expansion_onestate(self, state,
                                        step=1e-8, lb=None, ub=None):
        """
        """
        vol = self._get_volume_from_one_pt(state[0], state[1], lb, ub)
        volplus = self._get_volume_from_one_pt(state[0], state[1]+step/2,
                                               lb, ub)
        volmins = self._get_volume_from_one_pt(state[0], state[1]-step/2,
                                               lb, ub)
        dvdt = (volplus - volmins) / step
        return dvdt / vol
