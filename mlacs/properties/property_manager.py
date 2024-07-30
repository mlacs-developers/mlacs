"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ..core.manager import Manager


# ========================================================================== #
# ========================================================================== #
class PropertyManager(Manager):
    """
    Parent Class managing the calculation of differents properties
    """
    def __init__(self,
                 prop,
                 folder='Properties',
                 **kwargs):

        Manager.__init__(self, folder=folder, **kwargs)

        if prop is None:
            self.check = [False]
            self.manager = None

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
        Check all criterions. They have to converged at the same time.
        """
        for _ in self.check:
            if not _:
                return False
        return True

# ========================================================================== #
    @Manager.exec_from_workdir
    def run(self, step):
        """
        Run property calculation.
        """
        dircheck = False
        for prop in self.manager:
            if step % prop.freq == 0:
                dircheck = True
        if dircheck:
            self.path.mkdir(exist_ok=True, parents=True)
        msg = ""
        for i, prop in enumerate(self.manager):
            if step % prop.freq == 0:
                self.check[i] = prop._exec()
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
