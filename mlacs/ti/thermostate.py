"""
"""
import os
from subprocess import call

import numpy as np

from ase.io import Trajectory, read
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import StateManager
from mlacs.utilities import get_elements_Z_and_masses


#========================================================================================================================#
#========================================================================================================================#
class ThermoState:
    """
    Parent class for the thermodynamic state used in thermodynamic integration
    """
    def __init__(self,
                 atoms,
                 dt=1.5,
                 nsteps=10000,
                 nsteps_eq=5000,
                 rng=None,
                ):
        
        self.atoms     = atoms
        self.dt        = dt
        self.nsteps    = nsteps
        self.nsteps_eq = nsteps_eq

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self._get_lammps_command()


#========================================================================================================================#
    def run_dynamics(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        """
        atomsfname     = wdir + "atoms.in"
        lammpsfname    = wdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input(wdir, pair_style, pair_coeff, mlip_style)
        call(lammps_command, shell=True, cwd=wdir)


#========================================================================================================================#
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd    = os.environ.get(envvar)
        if cmd is None:
            cmd    = "lmp"
        self.cmd = cmd


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        raise NotImplementedError


#========================================================================================================================#
    def get_workdir(self):
        """
        """
        return self.suffixdir


#========================================================================================================================#
    def post_process(self):
        """
        """
        pass
