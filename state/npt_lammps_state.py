import os

import numpy as np

from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from otf_mlacs.state import LammpsState
from otf_mlacs.utilities import get_elements_Z_and_masses



#========================================================================================================================#
#========================================================================================================================#
class NPTLammpsState(LammpsState):
    """
    """
    def __init__(self,
                 temperature,
                 pressure,
                 ptype="iso",
                 dtemp=None,
                 dpres=None,
                 dt=1.5*fs,
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
                             init_momenta,
                             workdir
                            )

        self.temperature = temperature
        self.pressure    = pressure
        self.ptype       = ptype
        self.dtemp       = dtemp
        self.dpres       = dpres


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)

        dtemp = self.dtemp
        if self.dtemp is None:
            dtemp = "$(100*dt)"

        dpres = self.dpres
        if self.dpres is None:
            dpres = "$(1000*dt)"

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

        #input_string += "velocity  all create {0}  {1}\n".format(self.temperature, self.rng.integers(999999))


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ (fs * 1000))
        input_string += "\n"

        input_string += "fix    1  all npt temp {0} {0}  {1} {2} {3} {3} {4}\n".format(self.temperature, dtemp, self.ptype, self.pressure, dpres)
        if self.fixcm:
            input_string += "fix    2  all recenter INIT INIT INIT"

        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)


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
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature, rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)
