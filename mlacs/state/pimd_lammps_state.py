"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np

from ase.io import read
from ase.io.lammpsdata import read_lammps_data, write_lammps_data

from mlacs.state import LammpsState

#========================================================================================================================#
#========================================================================================================================#
class PIMDLammpsState(LammpsState):
    """
    """
    def __init__(self,
                 nbeads,
                 dt=1,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 neighbourlist=100,
                 nprocperbead=1,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 workdir=None
                ):

        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                             rng,
                             None,
                             workdir
                            )

        self.nbeads        = nbeads
        self.neighbourlist = neighbourlist
        self.nprocperbead  = nprocperbead


#========================================================================================================================#
    def get_nbeads(self):
        """
        """
        return self.nbeads

#========================================================================================================================#
    def run_dynamics(self, atoms, pair_style, pair_coeff, eq=False):
        """
        """
        for ibead, at in enumerate(atoms):
            atomsfname = self.workdir + "atoms_{0}.in".format(ibead+1)
            write_lammps_data(atomsfname, at, velocities=True)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        self.write_lammps_input(atoms, pair_style, pair_coeff, nsteps)
        lammps_command  = self.cmd + " -partition {0}x{1} ".format(self.nbeads, self.nprocperbead)
        lammps_command += "-in " + self.lammpsfname  + "> log"
        call(lammps_command, shell=True, cwd=self.workdir)

        #atoms = read_lammps_data(self.workdir + "configurations.out")
        atoms = []
        for ibead in range(self.nbeads):
            atoms.append(read(self.workdir + "configurations_{0}.out".format(ibead+1)))
        return atoms


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        raise NotImplementedError


#========================================================================================================================#
    def get_general_input(self, pbc, masses):
        """
        """
        input_string  = "# LAMMPS input file to run a MLPIMD simulation for MLACS\n"
        input_string += "#####################################\n"
        input_string += "#           General parameters\n"
        input_string += "#####################################\n"
        input_string += "units             metal\n"
        input_string += "atom_modify       map array\n"
        input_string += "boundary          {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "variable          ibead uloop {0}\n".format(self.nbeads)
        input_string += "neigh_modify      delay 0 every {0} check no\n".format(self.neighbourlist)
        input_string += "atom_style        atomic\n"
        input_string += "read_data         atoms_${ibead}.in\n"
        for i, mass in enumerate(masses):
            input_string += "mass              " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string


#========================================================================================================================#
    def get_last_dump_input(self, elem, nsteps):
        """
        """
        input_string  = "#####################################\n"
        input_string += "#         Dump last step\n"
        input_string += "#####################################\n"
        input_string += "dump last all custom {0} ".format(nsteps) + self.workdir + "configurations_${ibead}.out id type xu yu zu vx vy vz fx fy fz element\n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string
