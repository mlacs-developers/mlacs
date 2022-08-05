"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np
from ase.units import kB

from mlacs.calc import CalcManager


# ========================================================================== #
# ========================================================================== #
class McSpinCalcManager(CalcManager):
    """
    Class for Monte-Carlo spin

    With this Calculator manager, for each configuration,
    two computation will be launched.
    One of the configuration has a spin flip. The spin flip is accepted
    using a Monte-Carlo Metropolis acceptance probability

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object.
    magmoms: :class:`np.ndarray`
        An array for the initial magnetic moments
    temperature: :class:`int`
        The temperature used for the Monte-Carlo acceptance step
    """
    def __init__(self, calc, magmoms, temperature):
        CalcManager.__init__(self, calc, magmoms)

        self.temperature = temperature
        self.rng = np.random.default_rng()

# ========================================================================== #
    def compute_true_potential(self, atoms):
        """
        """
        atoms_init = atoms.copy()
        atoms_try = atoms.copy()

        # First compute energy with the current magnetization
        atoms_init.set_initial_magnetic_moments(self.magmoms)
        atoms_init.calc = self.calc
        eini = atoms_init.get_potential_energy()

        # Now, flip a random spin
        magmoms_try = self.magmoms.copy()
        idx_spin = np.nonzero(magmoms_try)[0]
        idx_choice = self.rng.choice(idx_spin, 1)
        magmoms_try[idx_choice] *= -1

        # Compute energy with the spin flip
        atoms_try.set_initial_magnetic_moments(magmoms_try)
        atoms_try.calc = self.calc
        etry = atoms_try.get_potential_energy()

        # Compute acceptance
        beta = 1.0 / (self.temperature * kB)
        de = etry - eini
        p = self.rng.random()
        acc = np.exp(-beta * de)
        if p > acc:
            self.magmoms = magmoms_try
            return atoms_try
        else:
            return atoms_init

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        name = self.calc.name

        msg = "True potential parameters:\n"
        msg += "Calculator : {0}\n".format(name)
        if hasattr(self.calc, "todict"):
            dct = self.calc.todict()
            msg += "parameters :\n"
            for key in dct.keys():
                msg += "   " + key + "  {0}\n".format(dct[key])
        msg += "Spin flip Monte-Carlo\n"
        msg += f"   Spin temperature : {self.temperature}\n"
        return msg
