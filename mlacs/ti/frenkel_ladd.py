"""
"""
import os
from subprocess import call

import numpy as np

from ase.units import kB
from ase.io.lammpsdata import write_lammps_data

from mlacs.utilities.miscellanous import get_elements_Z_and_masses
from mlacs.ti.thermostate import ThermoState




#========================================================================================================================#
#========================================================================================================================#
class FrenkelLaddState(ThermoState):
    """
    """
    def __init__(self,
                 atoms,
                 temperature,
                 dt=1,
                 damp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_msd=5000,
                 gjf=True,
                 rng=None
                ):

        self.atoms       = atoms
        self.temperature = temperature
        self.damp        = damp
        self.gjf         = gjf
        self.nsteps_msd  = nsteps_msd

        ThermoState.__init__(self,
                             atoms,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng
                            )

        self.suffixdir = "FrenkelLadd_T{0}K/".format(self.temperature)


#========================================================================================================================#
    def run(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

        # First get optimal spring constant
        msd = self.compute_msd(wdir, pair_style, pair_coeff, mlip_style)

        self.run_dynamics(wdir, pair_style, pair_coeff, mlip_style, msd)

        with open(wdir + "MLMD.done", "w") as f:
            f.write("Done")


#========================================================================================================================#
    def run_dynamics(self, wdir, pair_style, pair_coeff, mlip_style, msd):
        """
        """
        atomsfname     = wdir + "atoms.in"
        lammpsfname    = wdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input(wdir, pair_style, pair_coeff, mlip_style, msd)
        call(lammps_command, shell=True, cwd=wdir)


#========================================================================================================================#
    def compute_msd(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        """
        atomsfname     = wdir + "atoms.in"
        lammpsfname    = wdir + "lammps_msd_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input_msd(wdir, pair_style, pair_coeff, mlip_style)
        call(lammps_command, shell=True, cwd=wdir)

        elem, Z, masses = get_elements_Z_and_masses(self.atoms)
        kall     = []
        prt_data = []
        with open(wdir + "msd.dat", "w") as f:
            for e in elem:
                data = np.loadtxt(wdir + "msd{0}.dat".format(e))
                nat  = np.count_nonzero([a==e for a in self.atoms.get_chemical_symbols()])
                k    = 3 * kB * self.temperature / data.mean()
                kall.append(k)
                f.write(e + " {0}   {1:10.5f}".format(nat, k))
        return kall


#========================================================================================================================#
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """


#========================================================================================================================#
    def write_lammps_input(self, wdir, pair_style, pair_coeff, mlip_style, msd):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(self.atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"


        input_string  = "# LAMMPS input file to run a MLMD simulation for thermodynamic integration\n"
        input_string += "units        metal\n"

        pbc = self.atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + wdir + "atoms.in"
        input_string += "\n"
        for iel, el in enumerate(elem):
            input_string += "group       {0} type {1}\n".format(el, iel+1)

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"

        input_string += "variable      nsteps equal {0}\n".format(self.nsteps)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)

        input_string += "\n\n"

        input_string += "# Interactions\n"
        input_string += pair_style + "\n"
        input_string += pair_coeff
        if mlip_style == "snap" or mlip_style == "mliap":
            input_string += " " + " ".join(elem)
        input_string += "\n\n\n"

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"
        input_string += "velocity      all create {0} {1} dist gaussian\n".format(self.temperature, self.rng.integers(99999))

        input_string += "\n\n"


        input_string += "# Integrators\n"
        if self.gjf:
            input_string += "fix    f1  all langevin {0} {0}  {1} {2}  zero yes gjf vhalf\n".format(self.temperature, damp, self.rng.integers(99999))
        else:
            input_string += "fix    f1  all langevin {0} {0}  {0}  {1} zero yes\n".format(self.temperature, damp, self.rng.integers(99999))
        input_string += "fix    f2  all nve\n\n"
        for iel, el in enumerate(elem):
            input_string += "fix    ff{0} {0} ti/spring {1} ${{nsteps}} ${{nstepseq}} function 2\n".format(el, msd[iel])

        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"

        input_string += "\n"

        """
        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input()
        """

        input_string += "\n"

        input_string += "variable     dE equal pe-" + "-".join(["f_ff{0}".format(e) for e in el]) + "\n"
        input_string += "variable     lambda equal f_ff{0}[1]\n".format(elem[0])

        input_string += "#######################\n"
        input_string += "# Forward integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" screen no " + \
                        "append forward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f4\n"
        
        input_string += "\n\n"


        input_string += "#######################\n"
        input_string += "# Backward integration\n"
        input_string += "#######################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" screen no " + \
                        "append backward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def write_lammps_input_msd(self, wdir, pair_style, pair_coeff, mlip_style):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        elem, Z, masses = get_elements_Z_and_masses(self.atoms)

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"
        

        input_string  = "# LAMMPS input file to run a MLMD simulation for thermodynamic integration\n"
        input_string += "units        metal\n"

        pbc = self.atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + wdir + "atoms.in"
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"

        input_string += "variable      nsteps equal {0}\n".format(self.nsteps_msd)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)

        input_string += "\n\n"

        input_string += "# Interactions\n"
        input_string += pair_style + "\n"
        input_string += pair_coeff
        if mlip_style == "snap" or mlip_style == "mliap":
            input_string += " " + " ".join(elem)
        input_string += "\n\n\n"

        input_string += "timestep      {0}\n".format(self.dt/1000)
        input_string += "\n"
        input_string += "velocity      all create {0} {1} dist gaussian\n".format(self.temperature, self.rng.integers(99999))

        input_string += "\n\n"


        input_string += "# Integrators\n"
        if self.gjf:
            input_string += "fix    f1  all langevin {0} {0} {1} {2}  zero yes gjf vhalf\n".format(self.temperature, damp, self.rng.integers(99999))
        else:
            input_string += "fix    f1  all langevin {0} {0} {1}  {2} zero yes\n".format(self.temperature, damp, self.rng.integers(99999))
        input_string += "fix    f2  all nve\n"
        input_string += "\n"

        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"

        input_string += "\n"

        """
        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input()
        """

        for iel, el in enumerate(elem):
            input_string += "group       {0} type {1}\n".format(el, iel+1)
            input_string += "compute     c{0} {1} msd com yes\n".format(i,  el)
            input_string += "variable    msd{0} equal c_c{1}[4]\n".format(el, iel)

        input_string += "run         ${nstepseq}\n"

        for iel, el in enumerate(elem):
            input_string += "fix         f{0} {1} print 1 \"${{msd{1}}}\" screen no append msd{1}.dat\n".format(iel+3, el)

        input_string += "run         ${nsteps}"

        with open(wdir + "lammps_msd_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        npt = False
        if self.damp is None:
            damp = 100 * self.dt
        
        msg  = "Thermodynamic Integration using Frenkel-Ladd path for a solid\n"
        msg += "Temperature :                   {0}\n".format(self.temperature)
        msg += "Langevin damping :              {0} fs\n".format(damp)
        msg += "Timestep :                      {0} fs\n".format(self.dt)
        return msg
