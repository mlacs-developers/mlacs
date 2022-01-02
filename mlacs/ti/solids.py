"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np

from ase.units import kB
from ase.io.lammpsdata import write_lammps_data

from mlacs.utilities.miscellanous import get_elements_Z_and_masses
from mlacs.utilities.thermo import free_energy_harmonic_oscillator, free_energy_com_harmonic_oscillator
from mlacs.ti.thermostate import ThermoState


eV   = 1.602176634e-19  # eV
hbar = 6.582119514e-16  # hbar
amu  = 1.6605390666e-27 # atomic mass constant


#========================================================================================================================#
#========================================================================================================================#
class EinsteinSolidState(ThermoState):
    """
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 temperature,
                 fcorr1=None,
                 fcorr2=None,
                 k=None,
                 dt=1,
                 damp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_msd=5000,
                 rng=None,
                 suffixdir=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 trajinterval=50
                ):

        self.atoms       = atoms
        self.temperature = temperature
        self.damp        = damp
        self.nsteps_msd  = nsteps_msd

        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2

        ThermoState.__init__(self,
                             atoms,
                             pair_style,
                             pair_coeff,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                            )

        self.suffixdir = "FrenkelLadd_T{0}K/".format(self.temperature)
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"

        self.k           = k
        if self.k is not None:
            if isinstance(self.k, list):
                if not len(self.k) == len(self.elem):
                    msg = "The spring constant paramater has to be a float or a list of length n=number of different species in the system"
                    raise ValueError(msg)
            elif isinstance(self.k, (float, int)):
                self.k = [self.k] * len(self.elem)
            else:
                msg = "The spring constant parameter k has to be a float or a list of length n=\'number of different species in the system\'"
                raise ValueError(msg)


#========================================================================================================================#
    def run(self, wdir):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

        if self.k is None:
            # First get optimal spring constant
            self.compute_msd(wdir)

        self.run_dynamics(wdir)

        with open(wdir + "MLMD.done", "w") as f:
            f.write("Done")


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
    def compute_msd(self, wdir):
        """
        """
        atomsfname     = wdir + "atoms.in"
        lammpsfname    = wdir + "lammps_msd_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input_msd(wdir)
        call(lammps_command, shell=True, cwd=wdir)

        kall     = []
        prt_data = []
        with open(wdir + "msd.dat", "w") as f:
            for e in self.elem:
                data = np.loadtxt(wdir + "msd{0}.dat".format(e))
                nat  = np.count_nonzero([a==e for a in self.atoms.get_chemical_symbols()])
                k    = 3 * kB * self.temperature / data.mean()
                kall.append(k)
                f.write(e + " {0}   {1:10.5f}\n".format(nat, k))
        self.k = kall


#========================================================================================================================#
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """
        # Get needed value/constants
        vol             = self.atoms.get_volume()
        k               = self.k
        kBT             = kB * self.temperature
        nat_tot         = len(self.atoms)

        # Compute some oscillator frequencies and number of atoms for each species
        omega  = []
        nat    = []
        for iel, e in enumerate(self.elem):
            omega.append(np.sqrt(self.k[iel] / (self.masses[iel])))
            nat.append(np.count_nonzero([a==e for a in self.atoms.get_chemical_symbols()]))

        # Compute free energy of the Einstein crystal
        f_harm = free_energy_harmonic_oscillator(omega, self.temperature, nat) # eV/at

        # Compute the center of mass correction
        #f_cm    = free_energy_com_harmonic_oscillator(omega, self.temperature, nat, vol, self.masses) # eV/at
        f_cm    = free_energy_com_harmonic_oscillator(self.k, self.temperature, nat, vol, self.masses) # eV/at


        # Compute the work between einstein crystal and the MLIP
        dE_f, lambda_f = np.loadtxt(wdir+"forward.dat", unpack=True)
        dE_b, lambda_b = np.loadtxt(wdir+"backward.dat", unpack=True)
        int_f = np.trapz(dE_f, lambda_f)
        int_b = np.trapz(dE_b, lambda_b)

        work  = (int_f - int_b) / 2.0
        #work /= nat_tot # eV/at

        free_energy = f_harm + f_cm + work
        free_energy_corrected  = free_energy
        if self.fcorr1 is not None:
            free_energy_corrected += self.fcorr1
        if self.fcorr2 is not None:
            free_energy_corrected += self.fcorr2

        with open(wdir+"free_energy.dat", "w") as f:
            header  = "#   T [K]     Fe tot [eV/at]     Fe harm [eV/at]      Work [eV/at]      Fe com [eV/at]"
            results = "{0:10.3f}     {1:10.6f}         {2:10.6f}          {3:10.6f}         {4:10.6f}".format(self.temperature, free_energy, f_harm, work, f_cm)
            if self.fcorr1 is not None:
                header  += "    Delta F1 [eV/at]"
                results += "         {0:10.6f}".format(self.fcorr1)
            if self.fcorr2 is not None:
                header  += "    Delta F2 [eV/at]"
                results += "          {0:10.6f}".format(self.fcorr2)
            if self.fcorr1 is not None or self.fcorr2 is not None:
                header  += "    Fe corrected [eV/at]"
                results += "           {0:10.6f}".format(free_energy_corrected)
            header += "\n"
            f.write(header)
            f.write(results)
        
        msg  = "Summary of results for this state\n"
        msg += '===============================================================\n' 
        msg += "Frenkel-Ladd path integration, with an Einstein crystal reference\n"
        msg += "Reference Einstein crystal spring :\n"
        for iel, e in enumerate(self.elem):
            msg += "    For {0} :                   {1:10.6f} eV/angs^2\n".format(e, self.k[iel])
        msg += "Temperature :                   {0:10.3f} K\n".format(self.temperature)
        msg += "Volume :                        {0:10.3f} angs^3\n".format(vol/nat_tot)
        msg += "Free energy :                   {0:10.6f} eV/at\n".format(free_energy)
        msg += "Excess work :                   {0:10.6f} eV/at\n".format(work)
        msg += "Einstein crystal free energy :  {0:10.6f} eV/at\n".format(f_harm)
        msg += "Center of mass free energy :    {0:10.6f} eV/at\n".format(f_cm)
        if self.fcorr1 is not None:
            msg += "1st order true pot correction : {0:10.6f} eV/at\n".format(self.fcorr1)
        if self.fcorr2 is not None:
            msg += "2nd order true pot correction : {0:10.6f} eV/at\n".format(self.fcorr2)
        if self.fcorr1 is not None or self.fcorr2 is not None:
            msg += "Free energy corrected :         {0:10.6f} eV/at\n".format(free_energy_corrected)
        return msg


#========================================================================================================================#
    def write_lammps_input(self, wdir):
        """
        Write the LAMMPS input for the MLMD simulation
        """

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"


        input_string  = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += "variable      nsteps equal {0}\n".format(self.nsteps)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)
        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "#####################################\n"

        input_string += "\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "# Integrators\n"
        input_string += "#####################################\n"
        input_string += "velocity      all create {0} {1} dist gaussian\n".format(self.temperature, self.rng.integers(99999))
        input_string += "fix           f2  all nve\n"
        for iel, el in enumerate(self.elem):
            input_string += "fix           ff{0} {0} ti/spring {1} ${{nsteps}} ${{nstepseq}} function 2\n".format(el, self.k[iel])
        input_string += "fix           f1  all langevin {0} {0}  {1}  {2} zero yes\n\n".format(self.temperature, damp, self.rng.integers(99999))
        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"
        input_string += "#####################################\n"

        input_string += "\n\n"

        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input()

        input_string += "\n"

        input_string += "#####################################\n"
        input_string += "#         Integration\n"
        input_string += "#####################################\n"
        input_string += "variable     dE equal (pe-" + "-".join(["f_ff" + e for e in self.elem]) + ")/atoms \n"
        input_string += "variable     lambda equal f_ff{0}[1]\n".format(self.elem[0])
        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "#       Forward integration\n"
        input_string += "#####################################\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" screen no " + \
                        "append forward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f4\n"
        input_string += "#####################################\n"
        
        input_string += "\n\n"


        input_string += "#####################################\n"
        input_string += "#       Backward integration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" screen no " + \
                        "append backward.dat title \"# pe    lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "#####################################\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def write_lammps_input_msd(self, wdir):
        """
        Write the LAMMPS input for the MLMD simulation
        """

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"
        

        input_string  = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += "variable      nsteps equal {0}\n".format(self.nsteps_msd)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)
        input_string += "timestep      {0}\n".format(self.dt/1000)
        for iel, el in enumerate(self.elem):
            #input_string += "group         {0} type {1}\n".format(el, iel+1)
            input_string += "compute       c{0} {1} msd com yes\n".format(10+iel,  el)
            input_string += "variable      msd{0} equal c_c{1}[4]\n".format(el, 10+iel)
        input_string += "#####################################\n"
        input_string += "\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "#          Integrators\n"
        input_string += "#####################################\n"
        input_string += "velocity      all create {0} {1} dist gaussian\n".format(self.temperature, self.rng.integers(99999))
        input_string += "fix    f2  all nve\n"
        input_string += "fix    f1  all langevin {0} {0} {1}  {2} zero yes\n".format(self.temperature, damp, self.rng.integers(99999))
        input_string += "# Fix center of mass\n"
        input_string += "compute       c1 all temp/com\n"
        input_string += "fix_modify    f1 temp c1\n"
        input_string += "#####################################\n"
        input_string += "\n\n"

        if self.logfile is not None:
            input_string += self.get_log_input("msd")
        if self.trajfile is not None:
            input_string += self.get_traj_input("msd")

        input_string += "#####################################\n"
        input_string += "#           Compute MSD\n"
        input_string += "#####################################\n"
        input_string += "run         ${nstepseq}\n"
        for iel, el in enumerate(self.elem):
            input_string += "fix         f{0} {1} print 1 \"${{msd{1}}}\" screen no append msd{1}.dat\n".format(iel+3, el)
        input_string += "run         ${nsteps}\n"
        input_string += "#####################################\n"
        with open(wdir + "lammps_msd_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        npt = False
        if self.damp is None:
            damp = 100 * self.dt
        
        msg  = "Thermodynamic Integration using Frenkel-Ladd path and an Einstein crystal\n"
        msg += "Temperature :                   {0}\n".format(self.temperature)
        msg += "Langevin damping :              {0} fs\n".format(damp)
        msg += "Timestep :                      {0} fs\n".format(self.dt)
        msg += "Number of steps :               {0}\n".format(self.nsteps)
        msg += "Number of equilibration steps : {0}\n".format(self.nsteps_eq)
        if self.k is None:
            msg += "Reference einstein crystal to be computed\n"
        else:
            msg += "Reference einstein crystal spring :\n"
            for iel, e in enumerate(self.elem):
                msg += "    For {0} :                   k = {1} eV/angs^2\n".format(e, self.k[iel])
        return msg
