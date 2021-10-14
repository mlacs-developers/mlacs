import os
from subprocess import call

import numpy as np

from ase.io import Trajectory, read
from ase.io.lammpsdata import read_lammps_data
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import StateManager
from mlacs.utilities import get_elements_Z_and_masses


default_lammps = {}

#========================================================================================================================#
#========================================================================================================================#
class LammpsState(StateManager):
    """
    Parent Class for the Lammps States
    """
    def __init__(self,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
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
                              trajfile,
                              interval,
                              loginterval,
                              trajinterval
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


#========================================================================================================================#
    def get_log_in(self):
        """
        """
        input_string  = "# Logging\n"
        input_string += "variable    t equal step\n"
        input_string += "variable    mytemp equal temp\n"
        input_string += "variable    mype equal pe\n"
        input_string += "variable    myke equal ke\n"
        input_string += "variable    myetot equal etotal\n"
        input_string += "variable    mypress equal press/10000\n"
        input_string += "variable    mylx  equal lx\n"
        input_string += "variable    myly  equal ly\n"
        input_string += "variable    mylz  equal lz\n"
        input_string += "variable    vol   equal (lx*ly*lz)\n"
        input_string += "variable    mypxx equal pxx/(vol*10000)\n"
        input_string += "variable    mypyy equal pyy/(vol*10000)\n"
        input_string += "variable    mypzz equal pzz/(vol*10000)\n"
        input_string += "variable    mypxy equal pxy/(vol*10000)\n"
        input_string += "variable    mypxz equal pxz/(vol*10000)\n"
        input_string += "variable    mypyz equal pyz/(vol*10000)\n"

        input_string += 'fix mythermofile all print {0} "$t ${{myetot}}  ${{mype}} ${{myke}} ${{mytemp}}  ${{mypress}} ${{mypxx}} ${{mypyy}} ${{mypzz}} ${{mypxy}} ${{mypxz}} ${{mypyz}}" append {1} title "# Step  Etot  Epot  Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'.format(self.loginterval, self.logfile)
        input_string += "\n"
        return input_string


#========================================================================================================================#
    def get_traj_in(self, elem):
        """
        """
        input_string  = "# Dumping\n"
        input_string += "dump dum1 all custom {0} ".format(self.trajinterval) + self.workdir + "{0} id type xu yu zu vx vy vz fx fy fz element \n".format(self.trajfile)
        input_string += "dump_modify dum1 append yes\n" # Don't overwrite file
        input_string += "dump_modify dum1 element " # Add element type
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        return input_string
