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
                 temperature,
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

        self.temperature = temperature

        # Construct the working directory to run the LAMMPS MLDMD simulations
        self.workdir = None
        if self.workdir is None:
            self.workdir = os.getcwd() + "/LammpsMLMD/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        self.atomsfname = self.workdir + "atoms.in"
        self.lammpsfname = self.workdir + "lammps_input.in"

        self._get_lammps_command()
        self.islammps = True

#========================================================================================================================#
    def run_dynamics(self, supercell, pair_style, pair_coeff, eq=False):
        """
        """
        atoms = supercell.copy()

        write_lammps_data(self.atomsfname, supercell)

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
        parameters = self.dyn_parameters
        elem, Z, masses = get_elements_Z_and_masses(atoms)


        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        input_string += "mass   "
        for i, mass in enumerate(masses):
            input_string += str(i + 1) + "  " + str(mass)
        input_string += "\n"

        input_string += "velocity  all create {0}  {1}\n".format(self.temperature, np.random.randint(999999))


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ (fs * 1000))
        input_string += "\n"

        if "fix" in parameters:
            input_string += " ".join(["fix     {0}\n".format(p) for p in np.atleast_1d(parameters["fix"])])
            input_string += "\n"

        if self.logfile is not None:
            input_string += "# Logging\n"
            input_string += "variable    t equal step\n"
            input_string += "variable    mytemp equal temp\n"
            input_string += 'fix mythermofile all print 1 "t  ${mytemp}" file {0}'.format(self.logfile)
            input_string += "\n"

        if self.trajfname is not None:
            input_string += "# Dumping\n"
            input_string += "dump dum1 all custom 5 " + self.workdir + "dump_output.dat id type xu yu zu vx vy vz fx fy fz element \n" 
            input_string += "dump_modify dum1 element " # Add element type


        input_string += "# Dump last step\n"
        input_string += "dump last all custom {0} ".format(nsteps) + self.workdir + "configurations.out  id type xu yu zu vx vy vz fx fy fz element\n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
        input_string += "\n"

        input_string += "run  {0}".format(nsteps)


        with open(self.lammpsfname, "w") as f:
            f.write(input_string)
