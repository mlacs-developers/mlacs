"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

from mlacs.prop import CalcProperty

# ========================================================================== #
# ========================================================================== #
class PropertyManager:
    """
    Parent Class managing the calculation of differents properties

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
                 prop):
        if prop is None:
            self.check = [False]
        elif isinstance(prop, list):
            self.manager = prop
            self.check = [False for _ in range(len(prop))]
        else:
            self.manager = [prop]
            self.check = [False]

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
    def run(self, wdir, step):
        """
        """
        msg = ""
        self.check = []
        for prop in self.manager:
            if prop.freq:
                results = prop._exec(wdir)
                check = prop._check(results)
                msg += prop.log_recap()
                self.check.append(check)
        msg += '\n' 
        return msg
