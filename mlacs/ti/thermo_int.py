"""
"""
import os

import numpy as np

from ase.atoms import Atoms
from ase.io import read


#========================================================================================================================#
#========================================================================================================================#
class ThermodynamicIntegration:
    """
    """
    def __init__(self,
                 thermostate,
                 mlip=None,
                 workdir=None,
                )

    # Construct the working directory to run the thermodynamic integrations
    self.workdir = workdir
    if self.workdir is None:
        self.workdir = os.getcwd() + "/ThermodynamicIntegration/"
    if self.workdir[-1] != "/"
        self.wordir += "/"
    if not os.path.exists(self.workdir):
        os.makedirs(self.wordir)


#========================================================================================================================#
    def run(self):
        """
        """
        self.create_links()
