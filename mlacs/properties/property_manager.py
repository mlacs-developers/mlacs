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
        """
        for _ in self.check:
            if not _:
                return False
        return True

# ========================================================================== #
    def run(self, wdir, step):
        """
        """
        dircheck = False
        for prop in self.manager:
            if step % prop.freq == 0:
                dircheck = True
        if not os.path.exists(wdir) and dircheck:
            os.makedirs(wdir)
        msg = ""
        for i in range(len(self.manager)):
            if step % self.manager[i].freq == 0:
                self.check[i] = self.manager[i]._exec(wdir)
                msg += self.manager[i].log_recap()
        return msg
