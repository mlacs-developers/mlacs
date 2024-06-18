"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

from ..core.manager import Manager

from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import CalculatorError


# ========================================================================== #
# ========================================================================== #
class CalcManager(Manager):
    """
    Parent Class managing the true potential being simulated.
    This Calc class can support any :class:`ase.calculator` calculators from
    the ASE python package.
    Create, execute and read the output of any ASE supported calculators.

    Parameters
    ----------
    calc: :class:`ase.calculator`
        An ASE calculator object

    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.
    """

    def __init__(self,
                 calc,
                 magmoms=None,
                 folder='Calc',
                 **kwargs):

        Manager.__init__(self, folder=folder, **kwargs)

        self.calc = calc
        self.magmoms = magmoms

# ========================================================================== #
    def compute_true_potential(self, confs, subfolder=None, step=None):
        """
        Compute the energy of given configurations with an ASE calculator.
        """
        confs = [at.copy() for at in confs]
        result_confs = []
        for at in confs:
            at.set_initial_magnetic_moments(self.magmoms)
            at.calc = self.calc
            try:
                energy = at.get_potential_energy()
                forces = at.get_forces()
                stress = at.get_stress()
                sp_calc = SinglePointCalculator(at,
                                                energy=energy,
                                                forces=forces,
                                                stress=stress)
                at.calc = sp_calc
                result_confs.append(at)
            except CalculatorError:
                result_confs.append(None)
        return result_confs

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        try:
            name = self.calc.name
        except AttributeError:
            name = None

        msg = "True potential parameters:\n"
        if name is not None:
            msg += "Calculator : {0}\n".format(name)
        if hasattr(self.calc, "todict"):
            dct = self.calc.todict()
            msg += "parameters :\n"
            for key in dct.keys():
                msg += "   " + key + "  {0}\n".format(dct[key])
        msg += "\n"
        return msg
