import os
from subprocess import call

import numpy as np

from ase.io import Trajectory, read
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsdata import write_lammps_data
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from otf_mlacs.state import StateManager
from otf_mlacs.utilities import get_elements_Z_and_masses


default_lammps = {}

#========================================================================================================================#
#========================================================================================================================#
class LammpsState(StateManager):
    """
    """
    def __init__(self,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfname=None,
                 rng=None,
                 init_momenta=None,
                 workdir=None
                ):
        
        StateManager.__init__(self,
                              dt,
                              nsteps,
                              nsteps_eq,
                              fixcm,
                              logfile,
                              trajfname
                             )

        # Construct the working directory to run the LAMMPS MLDMD simulations
        self.workdir = None
        if self.workdir is None:
            self.workdir = os.getcwd() + "/LammpsMLMD/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.init_momenta = init_momenta

        self.atomsfname = self.workdir + "atoms.in"
        self.lammpsfname = self.workdir + "lammps_input.in"

        self._get_lammps_command()
        self.islammps = True

#========================================================================================================================#
    def run_dynamics(self, supercell, pair_style, pair_coeff, eq=False):
        """
        """
        atoms = supercell.copy()

        write_lammps_data(self.atomsfname, supercell, velocities=True)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        self.write_lammps_input(atoms, pair_style, pair_coeff, nsteps)
        lammps_command = self.cmd + "< " + self.lammpsfname + "> log"
        call(lammps_command, shell=True, cwd=self.workdir)

        #atoms = read_lammps_data(self.workdir + "configurations.out")
        atoms = read(self.workdir + "configurations.out")
        return atoms.copy()


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
