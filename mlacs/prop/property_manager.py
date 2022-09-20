"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

from mlacs.prop import CalcProperty

# ========================================================================== #
# ========================================================================== #
class PropertyManager:
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
                 prop,
                 nstate=None):
        self.ncalc = nstate
        if prop is None:
            self.check = [False]
        elif isinstance(prop, CalcProperty):
            self.manager = [calcprop for _ in range(self.ncalc)]
            self.check = [False for _ in range(self.ncalc)]
        else:
            self.manager = prop
            self.check = [False for _ in range(self.ncalc)]

# ========================================================================== #
    @property
    def check_criterion(self):
        """
        """
        for _ in self.check:
            if not _:
                return False
        return True

# ========================================================================== #
    @property
    def run(self):
        """
        """
        msg = ""
        for prop in self.manager:
            prop.run()
            msg += prop.log_recap()
        msg += '\n' 
        return msg
