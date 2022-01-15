"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

#===================================================================================================================================================#
#===================================================================================================================================================#
class CalcManager:
    """
    Parent Class managing the true potential being simulated

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object
    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.
    """
    def __init__(self,
                 calc,
                 magmoms=None
                ):

        self.calc    = calc
        self.magmoms = magmoms

#===================================================================================================================================================#
    def compute_true_potential(self, atoms):
        """
        """
        atoms.set_initial_magnetic_moments(self.magmoms)
        atoms.calc = self.calc
        try:
            atoms.get_potential_energy()
        except:
            atoms = None
        return atoms


#===================================================================================================================================================#
    def log_recap_state(self):
        """
        """
        name = self.calc.name

        msg  = "True potential parameters:\n"
        msg += "Calculator : {0}\n".format(name)
        if hasattr(self.calc, "todict"):
            dct = self.calc.todict()
            msg += "parameters :\n"
            for key in dct.keys():
                msg += "   " + key + "  {0}\n".format(dct[key])
        msg += "\n"
        return msg
