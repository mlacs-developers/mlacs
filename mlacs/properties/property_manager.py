"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import os


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

        self.workdir = os.getcwd() + "/Properties/"
        if prop is None:
            self.check = [False]
        elif isinstance(prop, list):
            self.manager = prop
            self.check = [False for _ in range(len(prop))]
            if not os.path.exists(self.workdir):
                os.makedirs(self.workdir)
        else:
            self.manager = [prop]
            self.check = [False]
            if not os.path.exists(self.workdir):
                os.makedirs(self.workdir)

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
    def run(self, wdir):
        """
        """
        msg = ""
        self.check = []
        if not os.path.exists(wdir):
            os.makedirs(wdir)
        for prop in self.manager:
            if prop.freq:
                check = prop._exec(wdir)
                msg += prop.log_recap()
                self.check.append(check)
        msg += '\n'
        return msg
