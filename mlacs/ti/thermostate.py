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
                 pair_style,
                 pair_coeff,
                 dt=1.5,
                 nsteps=10000,
                 nsteps_eq=5000,
                 rng=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 trajinterval=50,
                ):
        
        self.atoms      = atoms
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.dt         = dt
        self.nsteps     = nsteps
        self.nsteps_eq  = nsteps_eq

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self._get_lammps_command()

        self.logfile      = logfile
        self.trajfile     = trajfile
        self.loginterval  = loginterval
        self.trajinterval = trajinterval
        if interval is not None:
            self.loginterval    = interval
            self.trajinterval   = interval

        self.elem, self.Z, self.masses = get_elements_Z_and_masses(self.atoms)

#========================================================================================================================#
    def run_dynamics(self, wdir):
        """
        """
        atomsfname     = wdir + "atoms.in"
        lammpsfname    = wdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input(wdir)
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
    def write_lammps_input(self, atoms):
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


#========================================================================================================================#
    def get_log_input(self, suffix=None):
        """
        """
        input_string  = "#####################################\n"
        input_string += "#          Logging\n"
        input_string += "#####################################\n"
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

        if suffix is None:
            input_string += 'fix mythermofile all print {0} "$t ${{myetot}}  ${{mype}} ${{myke}} ${{mytemp}}  ${{mypress}} ${{mypxx}} ${{mypyy}} ${{mypzz}} ${{mypxy}} ${{mypxz}} ${{mypyz}}" append mlmd.log title "# Step  Etot  Epot  Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'.format(self.loginterval)
        else:
            input_string += 'fix mythermofile all print {0} "$t ${{myetot}}  ${{mype}} ${{myke}} ${{mytemp}}  ${{mypress}} ${{mypxx}} ${{mypyy}} ${{mypzz}} ${{mypxy}} ${{mypxz}} ${{mypyz}}" append mlmd_{1}.log title "# Step  Etot  Epot  Ekin  Press  Pxx  Pyy  Pzz  Pxy  Pxz  Pyz"\n'.format(self.loginterval, suffix)
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string


#========================================================================================================================#
    def get_traj_input(self, suffix=None):
        """
        """
        input_string  = "#####################################\n"
        input_string += "#           Dumping\n"
        input_string += "#####################################\n"
        if suffix is None:
            input_string += "dump dum1 all custom {0} ".format(self.trajinterval) + "mlmd.traj id type xu yu zu vx vy vz fx fy fz element \n".format(self.trajfile)
        else:
            input_string += "dump dum1 all custom {0} ".format(self.trajinterval) + "mlmd_{0}.traj id type xu yu zu vx vy vz fx fy fz element \n".format(suffix)
        input_string += "dump_modify dum1 append yes\n"
        input_string += "dump_modify dum1 element " # Add element type
<<<<<<< HEAD
        input_string += " ".join([p for p in self.elem])
        input_string += "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string


#========================================================================================================================#
    def get_general_input(self):
        """
        """
        input_string  = "# LAMMPS input file to run a MLMD simulation for thermodynamic integration\n"
        input_string += "#####################################\n"
        input_string += "#           General parameters\n"
        input_string += "#####################################\n"
        input_string += "units        metal\n"

        pbc = self.atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    atoms.in\n"
        for i, mass in enumerate(self.masses):
            input_string += "mass         " + str(i + 1) + "  " + str(mass) + "\n"
        for iel, el in enumerate(self.elem):
            input_string += "group        {0} type {1}\n".format(el, iel+1)
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string


#========================================================================================================================#
    def get_interaction_input(self):
        """
        """
        input_string  = "#####################################\n"
        input_string += "#           Interactions\n"
        input_string += "#####################################\n"
        input_string += "pair_style    " + self.pair_style + "\n"
        input_string += "pair_coeff    " + self.pair_coeff + "\n"
=======
        input_string += " ".join([p for p in elem])
        input_string += "\n"
>>>>>>> ti
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string
