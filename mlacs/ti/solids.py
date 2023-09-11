"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np
from ase.units import kB
from ase.io.lammpsdata import write_lammps_data

from ..utilities.thermo import (free_energy_harmonic_oscillator,
                                free_energy_com_harmonic_oscillator)
from .thermostate import ThermoState


eV = 1.602176634e-19  # eV
hbar = 6.582119514e-16  # hbar
amu = 1.6605390666e-27  # atomic mass constant


# ========================================================================== #
# ========================================================================== #
class EinsteinSolidState(ThermoState):
    """
    Class for performing thermodynamic integration from
    an Einstein crystal reference

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE atoms object on which the simulation will be performed
    pair_style: :class:`str`
        pair_style for the LAMMPS input
    pair_coeff: :class:`str` or :class:`list` of :class:`str`
        pair_coeff for the LAMMPS input
    temperature: :class:`float`
        Temperature of the simulation
    pressure: :class:`float`
        Pressure. None default value
    fcorr1: :class:`float` or ``None``
        First order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.
    fcorr2: :class:`float` or ``None``
        Second order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.
    k: :class:`float` or :class:`list` of :class:float` or ``None``
        Spring constant for the Einstein crystal reference.
        If a float, all atoms type have the same spring constant.
        If a list, a value for each atoms type should be provided.
        If ``None``, a short simulation is run to determine the optimal value.
        Default ``None``
    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``
    damp : :class:`float` (optional)
        Damping parameter.
        If ``None``, a damping parameter of  1000 x dt is used.
    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.
    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.
    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    suffixdir: :class:`str`
        Suffix for the directory in which the computation will be run.
        If ``None``, a directory ``\"Solid_TXK\"`` is created,
        where X is the temperature. Default ``None``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    interval : :class:`int` (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default ``50``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    trajinterval : :class:`int` (optional)
        Number of steps between MLMD traj writing. Default ``50``.
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 temperature,
                 pressure=None,
                 fcorr1=None,
                 fcorr2=None,
                 k=None,
                 dt=1,
                 damp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_msd=25000,
                 rng=None,
                 suffixdir=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 trajinterval=50):

        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure
        self.damp = damp
        self.nsteps_msd = nsteps_msd

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
                             trajinterval)

        self.suffixdir = f"Solid_T{self.temperature}K/"
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"

        self.k = k
        if self.k is not None:
            if isinstance(self.k, list):
                if not len(self.k) == len(self.elem):
                    msg = "The spring constant paramater has to be a " + \
                          "float or a list of length n=number of " + \
                          "different species in the system"
                    raise ValueError(msg)
            elif isinstance(self.k, (float, int)):
                self.k = [self.k] * len(self.elem)
            else:
                msg = "The spring constant parameter k has to be a " + \
                      "float or a list of length n=\'number of " + \
                      "different species in the system\'"
                raise ValueError(msg)

# ========================================================================== #
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

# ========================================================================== #
    def run_dynamics(self, wdir):
        """
        """
        atomsfname = wdir + "atoms.in"
        lammpsfname = wdir + "lammps_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input(wdir)
        call(lammps_command, shell=True, cwd=wdir)

# ========================================================================== #
    def compute_msd(self, wdir):
        """
        """
        atomsfname = wdir + "atoms.in"
        lammpsfname = wdir + "lammps_msd_input.in"
        lammps_command = self.cmd + "< " + lammpsfname + "> log"

        write_lammps_data(atomsfname, self.atoms)

        self.write_lammps_input_msd(wdir)
        call(lammps_command, shell=True, cwd=wdir)

        kall = []
        with open(wdir + "msd.dat", "w") as f:
            for e in self.elem:
                data = np.loadtxt(wdir + f"msd{e}.dat")
                nat = np.count_nonzero([a == e for a in
                                        self.atoms.get_chemical_symbols()])
                k = 3 * kB * self.temperature / data.mean()
                kall.append(k)
                f.write(e + " {0}   {1:10.5f}\n".format(nat, k))
        self.k = kall

# ========================================================================== #
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """
        # Get needed value/constants
        vol = self.atoms.get_volume()
        nat_tot = len(self.atoms)

        # Compute some oscillator frequencies and number
        # of atoms for each species
        omega = []
        nat = []
        for iel, e in enumerate(self.elem):
            omega.append(np.sqrt(self.k[iel] / (self.masses[iel])))
            nat.append(np.count_nonzero([a == e for a in
                                         self.atoms.get_chemical_symbols()]))

        # Compute free energy of the Einstein crystal
        f_harm = free_energy_harmonic_oscillator(omega,
                                                 self.temperature,
                                                 nat)  # eV/at

        # Compute the center of mass correction
        f_cm = free_energy_com_harmonic_oscillator(self.k,
                                                   self.temperature,
                                                   nat,
                                                   vol,
                                                   self.masses)  # eV/at

        # Compute the work between einstein crystal and the MLIP
        dE_f, lambda_f = np.loadtxt(wdir+"forward.dat", unpack=True)
        dE_b, lambda_b = np.loadtxt(wdir+"backward.dat", unpack=True)
        int_f = np.trapz(dE_f, lambda_f)
        int_b = np.trapz(dE_b, lambda_b)

        work = (int_f - int_b) / 2.0

        free_energy = f_harm + f_cm + work
        free_energy_corrected = free_energy
        if self.fcorr1 is not None:
            free_energy_corrected += self.fcorr1
        if self.fcorr2 is not None:
            free_energy_corrected += self.fcorr2

        if self.pressure is not None:
            pv = self.pressure/(160.21766208)*vol/nat_tot
        else:
            pv = 0.0
        with open(wdir+"free_energy.dat", "w") as f:
            header = "#   T [K]     Fe tot [eV/at]     " + \
                     "Fe harm [eV/at]      Work [eV/at]      Fe com [eV/at]      PV [eV/at]"
            results = f"{self.temperature:10.3f}     " + \
                      f"{free_energy:10.6f}         " + \
                      f"{f_harm:10.6f}          " + \
                      f"{work:10.6f}         " + \
                      f"{f_cm:10.6f}         " + \
                      f"{pv:10.6f}"
            if self.fcorr1 is not None:
                header += "    Delta F1 [eV/at]"
                results += f"         {self.fcorr1:10.6f}"
            if self.fcorr2 is not None:
                header += "    Delta F2 [eV/at]"
                results += f"          {self.fcorr2:10.6f}"
            if self.fcorr1 is not None or self.fcorr2 is not None:
                header += "    Fe corrected [eV/at]"
                results += f"           {free_energy_corrected:10.6f}"
            header += "\n"
            f.write(header)
            f.write(results)

        msg = "Summary of results for this state\n"
        msg += '============================================================\n'
        msg += "Frenkel-Ladd path integration, with an " + \
               "Einstein crystal reference\n"
        msg += "Reference Einstein crystal spring :\n"
        for iel, e in enumerate(self.elem):
            msg += f"    For {e} :                   " + \
                   f"{self.k[iel]:10.6f} eV/angs^2\n"
        msg += f"Temperature :                   {self.temperature:10.3f} K\n"
        msg += f"Volume :                        {vol/nat_tot:10.3f} angs^3\n"
        msg += f"Free energy :                   {free_energy:10.6f} eV/at\n"
        msg += f"Excess work :                   {work:10.6f} eV/at\n"
        msg += f"Einstein crystal free energy :  {f_harm:10.6f} eV/at\n"
        msg += f"Center of mass free energy :    {f_cm:10.6f} eV/at\n"
        if self.fcorr1 is not None:
            msg += "1st order true pot correction : " + \
                   f"{self.fcorr1:10.6f} eV/at\n"
        if self.fcorr2 is not None:
            msg += "2nd order true pot correction : " + \
                   f"{self.fcorr2:10.6f} eV/at\n"
        if self.fcorr1 is not None or self.fcorr2 is not None:
            msg += "Free energy corrected :         " + \
                   f"{free_energy_corrected:10.6f} eV/at\n"
        # add Fe or Fe_corrected to return to be read for cv purpose
        if self.fcorr1 is not None or self.fcorr2 is not None:
            return msg, free_energy_corrected
        else:
            if self.pressure is None:
                return msg, free_energy
            else:
                return msg, free_energy + pv    

# ========================================================================== #
    def write_lammps_input(self, wdir):
        """
        Write the LAMMPS input for the MLMD simulation
        """

        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        input_string = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += f"variable      nsteps equal {self.nsteps}\n"
        input_string += f"variable      nstepseq equal {self.nsteps_eq}\n"
        input_string += f"timestep      {self.dt / 1000}\n"
        input_string += "#####################################\n"

        input_string += "\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "# Integrators\n"
        input_string += "#####################################\n"
        input_string += f"velocity      all create {self.temperature} " + \
                        f"{self.rng.integers(99999)} dist gaussian\n"
        input_string += "fix           f2  all nve\n"
        for iel, el in enumerate(self.elem):
            input_string += f"fix           ff{el} {el} ti/spring " + \
                            f"{self.k[iel]} ${{nsteps}} ${{nstepseq}} " + \
                            "function 2\n"
        input_string += "fix           f1  all langevin " + \
                        f"{self.temperature} {self.temperature}  " + \
                        f"{damp}  {self.rng.integers(99999)} zero yes\n\n"
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
        input_string += "variable     dE equal (pe-" + \
                        "-".join(["f_ff" + e for e in self.elem]) + ")/atoms\n"
        input_string += f"variable     lambda equal f_ff{self.elem[0]}[1]\n"
        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "#       Forward integration\n"
        input_string += "#####################################\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" " + \
                        "screen no append forward.dat title \"# pe  lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "unfix        f4\n"
        input_string += "#####################################\n"

        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "#       Backward integration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "fix          f4 all print 1 \"${dE} ${lambda}\" " + \
                        "screen no append backward.dat title \"# pe lambda\"\n"
        input_string += "run          ${nsteps}\n"
        input_string += "#####################################\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)

# ========================================================================== #
    def write_lammps_input_msd(self, wdir):
        """
        Write the LAMMPS input for the MLMD simulation
        """
        damp = self.damp
        if damp is None:
            damp = "$(100*dt)"

        input_string = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += f"variable      nsteps equal {self.nsteps_msd}\n"
        input_string += f"variable      nstepseq equal {self.nsteps_eq}\n"
        input_string += f"timestep      {self.dt/1000}\n"
        for iel, el in enumerate(self.elem):
            # input_string += "group         {0} type {1}\n".format(el, iel+1)
            input_string += f"compute       c{10+iel} {el} msd com yes\n"
            input_string += f"variable      msd{el} equal c_c{10+iel}[4]\n"
        input_string += "#####################################\n"
        input_string += "\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "#          Integrators\n"
        input_string += "#####################################\n"
        input_string += f"velocity      all create {self.temperature} " + \
                        f"{self.rng.integers(99999)} dist gaussian\n"
        input_string += "fix    f2  all nve\n"
        input_string += f"fix    f1  all langevin {self.temperature} " + \
                        f"{self.temperature} {damp}  " + \
                        f"{self.rng.integers(99999)} zero yes\n"
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
            input_string += f"fix         f{iel+3} {el} print 1 " + \
                            f"\"${{msd{el}}}\" screen no append msd{el}.dat\n"
        input_string += "run         ${nsteps}\n"
        input_string += "#####################################\n"
        with open(wdir + "lammps_msd_input.in", "w") as f:
            f.write(input_string)

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        if self.damp is None:
            damp = 100 * self.dt

        msg = "Thermodynamic Integration using Frenkel-Ladd " + \
              "path and an Einstein crystal\n"
        msg += f"Temperature :                   {self.temperature}\n"
        msg += f"Langevin damping :              {damp} fs\n"
        msg += f"Timestep :                      {self.dt} fs\n"
        msg += f"Number of steps :               {self.nsteps}\n"
        msg += f"Number of equilibration steps : {self.nsteps_eq}\n"
        if self.k is None:
            msg += "Reference einstein crystal to be computed\n"
        else:
            msg += "Reference einstein crystal spring :\n"
            for iel, e in enumerate(self.elem):
                msg += f"    For {e} :                   " + \
                       f"k = {self.k[iel]} eV/angs^2\n"
        return msg
