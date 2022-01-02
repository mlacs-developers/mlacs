"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np

from ase.units import kB
from ase.io.lammpsdata import write_lammps_data

from mlacs.utilities.thermo import free_energy_uhlenbeck_ford, free_energy_ideal_gas
from mlacs.utilities.miscellanous import get_elements_Z_and_masses
from mlacs.ti.thermostate import ThermoState


p_tabled = [1, 25, 50, 75, 100]

#========================================================================================================================#
#========================================================================================================================#
class UFLiquidState(ThermoState):
    """
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 temperature,
                 fcorr1=None,
                 fcorr2=None,
                 p=50,
                 sigma=2.0,
                 dt=1,
                 damp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
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

        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2

        self.p     = p
        self.sigma = sigma

        if self.p not in p_tabled:
            msg = "The p value of the UF potential has to be one for which the free energy of the Uhlenbeck-Ford potential is tabulated\n" + \
                  "Those value are : 1, 25, 50, 75 and 100"
            raise ValueError(msg)

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

        self.suffixdir = "LiquidUF_T{0}K/".format(self.temperature)
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"


#========================================================================================================================#
    def run(self, wdir):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

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
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """
        pass
        # Get needed value/constants
        vol             = self.atoms.get_volume() # angs**3
        kBT             = kB * self.temperature   # eV
        nat_tot         = len(self.atoms)

        nat = []
        for iel, e in enumerate(self.elem):
            nat.append(np.count_nonzero([a==e for a in self.atoms.get_chemical_symbols()]))

        # Compute the ideal gas free energy
        f_ig = free_energy_ideal_gas(vol, nat, self.masses, self.temperature) # eV/at

        # Compute Uhlenbeck-Ford excess free energy
        f_uf = free_energy_uhlenbeck_ford(nat_tot/vol, self.p, self.sigma, self.temperature) # eV/at


        # Compute the work between Uhlenbeck-Ford potential and the MLIP
        u_f, lambda_f = np.loadtxt(wdir+"forward.dat", unpack=True)
        u_b, lambda_b = np.loadtxt(wdir+"backward.dat", unpack=True)
        int_f = np.trapz(u_f, lambda_f)
        int_b = np.trapz(u_b, lambda_b)
        work  = (int_f - int_b) / 2.0 # eV/at

        # Add everything together
        free_energy = f_ig + f_uf  + work
        free_energy_corrected  = free_energy
        if self.fcorr1 is not None:
            free_energy_corrected += self.fcorr1
        if self.fcorr2 is not None:
            free_energy_corrected += self.fcorr2

        # write the results
        with open(wdir+"free_energy.dat", "w") as f:
            header  = "#   T [K]     Fe tot [eV/at]     Fe harm [eV/at]      Work [eV/at]      Fe com [eV/at]"
            results = "{0:10.3f}     {1:10.6f}         {2:10.6f}          {3:10.6f}".format(self.temperature, free_energy, f_uf, work)
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
        msg += "Frenkel-Ladd path integration, with an Uhlenbeck-Ford potential reference\n"
        msg += "Reference potential parameters :\n"
        msg += "      sigma                     {0}\n".format(self.sigma)
        msg += "      p                         {0}\n".format(self.p)
        msg += "Temperature :                   {0:10.3f} K\n".format(self.temperature)
        msg += "Volume :                        {0:10.3f} angs^3/at\n".format(vol/nat_tot)
        msg += "Free energy :                   {0:10.6f} eV/at\n".format(free_energy)
        msg += "Excess work :                   {0:10.6f} eV/at\n".format(work)
        msg += "Ideal gas free energy :         {0:10.6f} eV/at\n".format(f_ig)
        msg += "UF excess free energy :         {0:10.6f} eV/at\n".format(f_uf)
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

        pair_coeff = self.pair_coeff.split()
        pair_style = self.pair_style.split()
        hybrid_pair_coeff = " ".join([*pair_coeff[:2], pair_style[0], *pair_coeff[2:]])

        input_string  = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += "variable      nsteps equal {0}\n".format(self.nsteps)
        input_string += "variable      nstepseq equal {0}\n".format(self.nsteps_eq)
        input_string += "variable      T equal {0}\n".format(self.temperature)
        input_string += "timestep      {0}\n".format(self.dt/ 1000)

        input_string += "\n\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "#       UF potential parameters\n"
        input_string += "#####################################\n"
        input_string += "variable      p    equal  {0}\n".format(self.p)
        input_string += "variable      kB   equal  8.6173303e-5\n"
        input_string += "variable      eps  equal  ${T}*${p}*${kB}\n"
        input_string += "variable      sig  equal  {0}\n".format(self.sigma)
        input_string += "variable      rc   equal  5.0*${sig}\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"



        input_string += "#####################################\n"
        input_string += "# Integrators\n"
        input_string += "#####################################\n"
        input_string += "velocity      all create {0} {1} dist gaussian\n".format(self.temperature, self.rng.integers(99999))
        input_string += "fix           f2  all nve\n"
        input_string += "fix           f1  all langevin ${{T}} ${{T}}  {0}  {1} zero yes\n\n".format(damp, self.rng.integers(99999))
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
        input_string += "#         Equilibration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration without UF potential\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "\n\n"
        

        input_string += "#####################################\n"
        input_string += "#       Forward integration\n"
        input_string += "#####################################\n"
        input_string += "variable     tau equal ramp(1,0)\n"
        input_string += "variable     lambda_true equal v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)\n"
        input_string += "variable     lambda_ufm equal 1-v_lambda_true\n"
        input_string += "\n"
        input_string += "pair_style   hybrid/scaled v_lambda_true {0} v_lambda_ufm ufm ${{rc}}\n".format(pair_style[0])
        input_string += "pair_coeff   " + hybrid_pair_coeff + "\n"
        input_string += "pair_coeff   * * ufm ${eps} ${sig}\n"
        input_string += "\n"
        input_string += "compute      c2 all pair {0}\n".format(pair_style[0])
        input_string += "compute      c3 all pair ufm\n"
        input_string += "\n"
        input_string += "variable     dU equal (c_c2-c_c3)/atoms\n"
        input_string += "variable     lamb equal 1-v_lambda_true\n"
        input_string += "\n"
        input_string += "fix          f3 all print 1 \"${dU}  ${lamb}\" " + \
                                   "title \"# dU lambda\" screen no append forward.dat\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f3\n"
        input_string += "#####################################\n"

        
        input_string += "\n\n"


        input_string += "#####################################\n"
        input_string += "#       Backward integration\n"
        input_string += "#####################################\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "\n"
        input_string += "variable     tau equal ramp(0,1)\n"
        input_string += "variable     lambda_true equal v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)\n"
        input_string += "variable     lambda_ufm equal 1-v_lambda_true\n"
        input_string += "\n"
        input_string += "pair_style   hybrid/scaled v_lambda_true {0} v_lambda_ufm ufm ${{rc}}\n".format(pair_style[0])
        input_string += "pair_coeff   " + hybrid_pair_coeff + "\n"
        input_string += "pair_coeff   * * ufm ${eps} ${sig}\n"
        input_string += "\n"
        input_string += "fix          f3 all print 1 \"${dU}  ${lamb}\" " + \
                                   "title \"# dU lambda\" screen no append backward.dat\n"
        input_string += "run          ${nsteps}\n"
        input_string += "#####################################\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def log_recap_state(self):
        """
        """
        npt = False
        if self.damp is None:
            damp = 100 * self.dt
        
        msg  = "Thermodynamic Integration using Frenkel-Ladd path and Uhlenberg-Ford potential for the liquid state\n"
        msg += "Temperature :                   {0}\n".format(self.temperature)
        msg += "Langevin damping :              {0} fs\n".format(damp)
        msg += "Timestep :                      {0} fs\n".format(self.dt)
        msg += "Number of steps :               {0}\n".format(self.nsteps)
        msg += "Number of equilibration steps : {0}\n".format(self.nsteps_eq)
        msg += "Parameters for UF potential :\n"
        msg += "      sigma                     {0}\n".format(self.sigma)
        msg += "      p                         {0}\n".format(self.p)
        return msg
