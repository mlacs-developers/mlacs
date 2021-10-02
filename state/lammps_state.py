import os

import numpy as np

from ase.io import Trajectory
from ase.io.lammpsdata import write_lammps_data
from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from otf_mlacs.state import StateManager


#========================================================================================================================#
#========================================================================================================================#
class LammpsState(StateManager):
    """
    """
    def __init__(self,
                 dt=1.5*fs,
                 nsteps=1000,
                 nsteps_eq=250,
                 dyn_parameters=default_lammps,
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
                              dyn_parameters,
                              logfile,
                              trajfname
                             )

        # Construct the working directory to run the LAMMPS MLDMD simulations
        self.workdir = None
        if self.workdir is None:
            workdir = os.getcwd() + "/LammpsMLMD/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        self.atomsfname = self.workdir + "atoms.in"
        self.lammpsfname = self.workdir + "lammps_input.in"

#========================================================================================================================#
    def run_dynamics(self, supercell, calc, eq=False):
        """
        """
        atoms = supercell.copy()

        write_lammps_data(self.atomsfname)

        self.write_lammps_input(atoms)
        run_lammps()

        atoms = read_lammps_data(self.workdir + "LastConfig")
        return atoms


#========================================================================================================================#
    def run_lammps(self):
        """
        """
        lammps_command = self.cmd
        call(lammps_command, shell=True)


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
    def write_lammps_input(self, atoms):
        """
        Write the LAMMPS input for the MD simulation
        """
        parameters = self.dyn_parameters


        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "\n"

        input_string += "# Interactions\n"
        input_string += "pair_style     \n"
        input_string += "pair_coeff     \n"
        input_string += "\n"

        if "fix" in parameters:
            input_string += join(["fix     {0}".format(p) for p in np.atleast_1d(parameters["fix"])])
            input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt * 1000 / fs)
        input_string += "\n"


        with open(self.lammpsfname, "w") as f:
            f.write(input_string)
