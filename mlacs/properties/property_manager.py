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
    """
    def __init__(self,
                 prop):

        self.workdir = os.getcwd() + "/Properties/"
        if prop is None:
            self.check = [False]
            self.manager = None
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
        Check all criterions. They have to converged at the same time.
        """
        for _ in self.check:
            if not _:
                return False
        return True

# ========================================================================== #
    def run(self, step, wdir):
        """
        Run property calculation.
        """
        dircheck = False
        for prop in self.manager:
            if step % prop.freq == 0:
                dircheck = True
        if not os.path.exists(wdir) and dircheck:
            os.makedirs(wdir)
        msg = ""
        for i, prop in enumerate(self.manager):
            if step % prop.freq == 0:
                self.check[i] = prop._exec(wdir)
                msg += repr(prop)
        return msg

# ========================================================================== #
    def calc_initialize(self, **kwargs):
        """
        Add on the fly arguments for calculation of properties.
        """
        for prop in self.manager:
            if prop.useatoms:
                prop.get_atoms(kwargs['atoms'])
