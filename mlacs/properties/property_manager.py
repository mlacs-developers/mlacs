"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

from pathlib import Path


# ========================================================================== #
# ========================================================================== #
class PropertyManager:
    """
    Parent Class managing the calculation of differents properties
    """
    def __init__(self,
                 prop,
                 workdir=None):

        if workdir is None:
            workdir = "Properties"
        self.workdir = Path(workdir).absolute()
        if prop is None:
            self.check = [False]
            self.manager = None
        elif isinstance(prop, list):
            self.manager = prop
            self.check = [False for _ in range(len(prop))]
            self.workdir.mkdir(exist_ok=True, parents=True)
        else:
            self.manager = [prop]
            self.check = [False]
            self.workdir.mkdir(exist_ok=True, parents=True)

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
        if dircheck:
            wdir.mkdir(exist_ok=True, parents=True)
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
