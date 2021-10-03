import os

import numpy as np

from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from otf_mlacs.state import LammpsState
from otf_mlacs.utilities import get_elements_Z_and_masses

#========================================================================================================================#
#========================================================================================================================#
class CustomLammpsState(LammpsState):
    """
    """
    def __init__(self,
                 custom_input,
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

        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfname,
                             rng,
                             init_momenta,
                             workdir
                            )
                     
        self.custom_input = custom_input


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)

        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ (fs * 1000))
        input_string += "\n"

        input_string += self.custom_input

        if self.logfile is not None:
            input_string += "# Logging\n"
            input_string += "variable    t equal step\n"
            input_string += "variable    mytemp equal temp\n"
            input_string += 'fix mythermofile all print 1 "$t  ${{mytemp}}" file {0}\n'.format(self.logfile)
            input_string += "\n"

        if self.trajfname is not None:
            input_string += "# Dumping\n"
            input_string += "dump dum1 all custom 5 " + self.workdir + "{0} id type xu yu zu vx vy vz fx fy fz element \n".format(self.trajfname)
            input_string += "dump_modify dum1 element " # Add element type
            input_string += " ".join([p for p in elem])
            input_string += "\n"


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


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        pass
