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
                 fixcm=True,
                 rng=None,
                 workdir=None
                )
        
        self.atoms     = atoms
        self.dt        = dt
        self.nsteps    = nsteps
        self.nsteps_eq = nsteps_eq
        self.fixcm     = fixcm

        self.rng = rng


#========================================================================================================================#
    def run_dynamics(self, pair_style, pair_coeff):
        """
        """
        atomsfname     = self.workdir + "atoms.in"
        lammpsfname    = self.workdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + self.lammpsfname + "> log"


        write_lammps_data(atomsfname, atoms, velocities=True)

        self.write_lammps_input(atoms, pair_style, pair_coeff, nsteps)
        call(lammps_command, shell=True, cwd=self.wordkir)


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
