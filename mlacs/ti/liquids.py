"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import call

import numpy as np
from ase.io.lammpsdata import write_lammps_data

from .thermostate import ThermoState
from ..utilities.thermo import (free_energy_uhlenbeck_ford,
                                free_energy_ideal_gas)


p_tabled = [1, 25, 50, 75, 100]


# ========================================================================== #
# ========================================================================== #
class UFLiquidState(ThermoState):
    """
    Class for performing thermodynamic integration
    from a Uhlenbeck-Ford potential reference

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
    pressure: :class:float
        Pressure. None default value
    fcorr1: :class:`float` or ``None``
        First order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.
    fcorr2: :class:`float` or ``None``
        Second order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.
    p: :class:`int`
        p parameter of the Uhlenbeck-Ford potential.
        Should be ``1``, ``25``, ``50``, ``75`` or ``100``. Default ``50``
    p: :class:`float`
        sigma parameter of the Uhlenbeck-Ford potential. Default ``2.0``.
    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``
    damp : :class:`float` (optional)
        Damping parameter.
        If ``None``, a damping parameter of 100 x dt is used.
    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.
    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.
    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    suffixdir: :class:`str`
        Suffix for the directory in which the computation will be run.
        If ``None``, a directory ``\"Liquid_TXK\"`` is created,
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
                 trajinterval=50):

        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure
        self.damp = damp

        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2

        self.p = p
        self.sigma = sigma

        if self.p not in p_tabled:
            msg = "The p value of the UF potential has to be one for " + \
                  "which the free energy of the Uhlenbeck-Ford potential " + \
                  "is tabulated\n" + \
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
                             trajinterval)

        self.suffixdir = "LiquidUF_T{0}K/".format(self.temperature)
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"

# ========================================================================== #
    def run(self, wdir):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

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
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """
        pass
        # Get needed value/constants
        vol = self.atoms.get_volume()  # angs**3
        nat_tot = len(self.atoms)

        nat = []
        for iel, e in enumerate(self.elem):
            nat.append(np.count_nonzero([a == e for a in
                                         self.atoms.get_chemical_symbols()]))

        # Compute the ideal gas free energy
        f_ig = free_energy_ideal_gas(vol,
                                     nat,
                                     self.masses,
                                     self.temperature)  # eV/at

        # Compute Uhlenbeck-Ford excess free energy
        f_uf = free_energy_uhlenbeck_ford(nat_tot/vol,
                                          self.p,
                                          self.sigma,
                                          self.temperature)  # eV/at

        # Compute the work between Uhlenbeck-Ford potential and the MLIP
        u_f, lambda_f = np.loadtxt(wdir+"forward.dat", unpack=True)
        u_b, lambda_b = np.loadtxt(wdir+"backward.dat", unpack=True)
        int_f = np.trapz(u_f, lambda_f)
        int_b = np.trapz(u_b, lambda_b)
        work = (int_f - int_b) / 2.0  # eV/at

        # Add everything together
        free_energy = f_ig + f_uf + work
        free_energy_corrected = free_energy
        if self.fcorr1 is not None:
            free_energy_corrected += self.fcorr1
        if self.fcorr2 is not None:
            free_energy_corrected += self.fcorr2

        if self.pressure is not None:
            pv = self.pressure / (160.21766208) * vol / nat_tot
        else:
            pv = 0.0

        # write the results
        with open(wdir+"free_energy.dat", "w") as f:
            header = "#   T [K]     Fe tot [eV/at]     " + \
                      "Fe harm [eV/at]      Work [eV/at]      PV [eV/at]"
            results = f"{self.temperature:10.3f}     " + \
                      f"{free_energy:10.6f}         " + \
                      f"{f_uf:10.6f}          " + \
                      f"{work:10.6f}          " + \
                      f"{pv:10.6}"
            if self.fcorr1 is not None:
                header += "    Delta F1 [eV/at]"
                results += "         {0:10.6f}".format(self.fcorr1)
            if self.fcorr2 is not None:
                header += "    Delta F2 [eV/at]"
                results += "          {0:10.6f}".format(self.fcorr2)
            if self.fcorr1 is not None or self.fcorr2 is not None:
                header += "    Fe corrected [eV/at]"
                results += "           {0:10.6f}".format(free_energy_corrected)
            header += "\n"
            f.write(header)
            f.write(results)

        msg = "Summary of results for this state\n"
        msg += '===========================================================\n'
        msg += "Frenkel-Ladd path integration, " \
               "with an Uhlenbeck-Ford potential reference\n"
        msg += "Reference potential parameters :\n"
        msg += f"      sigma                     {self.sigma}\n"
        msg += f"      p                         {self.p}\n"
        msg += f"Temperature :                   {self.temperature:10.3f} K\n"
        msg += "Volume :                        " + \
               f"{vol/nat_tot:10.3f} angs^3/at\n"
        msg += f"Free energy :                   {free_energy:10.6f} eV/at\n"
        msg += f"Excess work :                   {work:10.6f} eV/at\n"
        msg += f"Ideal gas free energy :         {f_ig:10.6f} eV/at\n"
        msg += f"UF excess free energy :         {f_uf:10.6f} eV/at\n"
        if self.fcorr1 is not None:
            msg += "1st order true pot correction : " + \
                   f"{self.fcorr1:10.6f} eV/at\n"
        if self.fcorr2 is not None:
            msg += "2nd order true pot correction : " + \
                   f"{self.fcorr2:10.6f} eV/at\n"
        if self.fcorr1 is not None or self.fcorr2 is not None:
            msg += "Free energy corrected :         " + \
                   f"{free_energy_corrected:10.6f} eV/at\n"

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

        pair_style = self.pair_style.split()
        if len(self.pair_coeff) == 1:
            pair_coeff = self.pair_coeff.split()
            hybrid_pair_coeff = " ".join([*pair_coeff[:2],
                                          pair_style[0],
                                          *pair_coeff[2:]])

        else:
            hybrid_pair_coeff = []
            for pc in self.pair_coeff:
                pc_ = pc.split()
                hpc_=" ".join([*pc_[:2], *pc_[2:]])
                hybrid_pair_coeff.append(hpc_)

        input_string = self.get_general_input()

        input_string += "#####################################\n"
        input_string += "#        Initialize variables\n"
        input_string += "#####################################\n"
        input_string += f"variable      nsteps equal {self.nsteps}\n"
        input_string += f"variable      nstepseq equal {self.nsteps_eq}\n"
        input_string += f"variable      T equal {self.temperature}\n"
        input_string += f"timestep      {self.dt / 1000}\n"
        input_string += "#####################################\n"

        input_string += "\n\n\n"

        input_string += self.get_interaction_input()

        input_string += "#####################################\n"
        input_string += "#       UF potential parameters\n"
        input_string += "#####################################\n"
        input_string += f"variable      p    equal  {self.p}\n"
        input_string += "variable      kB   equal  8.6173303e-5\n"
        input_string += "variable      eps  equal  ${T}*${p}*${kB}\n"
        input_string += f"variable      sig  equal  {self.sigma}\n"
        input_string += "variable      rc   equal  5.0*${sig}\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"

        input_string += "#####################################\n"
        input_string += "# Integrators\n"
        input_string += "#####################################\n"
        input_string += f"velocity      all create {self.temperature} " + \
                        f"{self.rng.integers(99999)} dist gaussian\n"
        input_string += "fix           f2  all nve\n"
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
        input_string += "#         Equilibration\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration without UF potential\n"
        input_string += "run          ${nstepseq}\n"
        input_string += "\n\n"

        input_string += "#####################################\n"
        input_string += "#       Forward integration\n"
        input_string += "#####################################\n"
        input_string += "variable     tau equal ramp(1,0)\n"
        input_string += "variable     lambda_true equal " + \
            "v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)\n"
        input_string += "variable     lambda_ufm equal 1-v_lambda_true\n"
        input_string += "\n"
        # pair_style comd compatible only with one zbl, To be fixed
        input_string += "pair_style   hybrid/scaled v_lambda_true " + \
            f"{pair_style[1]} {pair_style[2]} {pair_style[3]} v_lambda_true " + \
            f"{pair_style[4]} v_lambda_ufm ufm ${{rc}}\n"
        input_string += "pair_coeff   " + hybrid_pair_coeff[0] + "\n"
        input_string += "pair_coeff   " + hybrid_pair_coeff[1] + "\n"
        input_string += "pair_coeff   * * ufm ${eps} ${sig}\n"
        input_string += "\n"
        if len(self.pair_coeff)==1:
            input_string += f"compute      c2 all pair {pair_style[0]}\n"
            input_string += "compute      c3 all pair ufm\n"
            input_string += "variable     dU equal (c_c2-c_c3)/atoms\n"
        else:
            input_string += f"compute      c2 all pair {pair_style[1]}\n"
            input_string += f"compute      c4 all pair {pair_style[4]}\n"
            input_string += "compute      c3 all pair ufm\n"
            input_string += "variable     dU equal ((c_c2+c_c4)-c_c3)/atoms\n"
        input_string += "\n"
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
        input_string += "variable     lambda_true equal " + \
            "v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)\n"
        input_string += "variable     lambda_ufm equal 1-v_lambda_true\n"
        input_string += "\n"
        input_string += "pair_style   hybrid/scaled v_lambda_true " + \
            f"{pair_style[1]} {pair_style[2]} {pair_style[3]} v_lambda_true " + \
            f"{pair_style[4]} v_lambda_ufm ufm ${{rc}}\n"
        input_string += "pair_coeff   " + hybrid_pair_coeff[0] + "\n"
        input_string += "pair_coeff   " + hybrid_pair_coeff[1] + "\n"
        input_string += "pair_coeff   * * ufm ${eps} ${sig}\n"
        input_string += "\n"
        if len(self.pair_coeff)==1:
            input_string += f"compute      c2 all pair {pair_style[0]}\n"
            input_string += "compute      c3 all pair ufm\n"
            input_string += "variable     dU equal (c_c2-c_c3)/atoms\n"
        else:
            input_string += f"compute      c2 all pair {pair_style[1]}\n"
            input_string += f"compute      c4 all pair {pair_style[4]}\n"
            input_string += "compute      c3 all pair ufm\n"
            input_string += "variable     dU equal ((c_c2+c_c4)-c_c3)/atoms\n"
        input_string += "\n"
        input_string += "variable     lamb equal 1-v_lambda_true\n"
        input_string += "\n"
        input_string += "fix          f3 all print 1 \"${dU}  ${lamb}\" " + \
            "title \"# dU lambda\" screen no append backward.dat\n"
        input_string += "run          ${nsteps}\n"
        input_string += "#####################################\n"

        with open(wdir + "lammps_input.in", "w") as f:
            f.write(input_string)

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        if self.damp is None:
            damp = 100 * self.dt

        msg = "Thermodynamic Integration using " + \
              "Frenkel-Ladd path and Uhlenberg-Ford " + \
              "potential for the liquid state\n"
        msg += f"Temperature :                   {self.temperature}\n"
        msg += f"Langevin damping :              {damp} fs\n"
        msg += f"Timestep :                      {self.dt} fs\n"
        msg += f"Number of steps :               {self.nsteps}\n"
        msg += f"Number of equilibration steps : {self.nsteps_eq}\n"
        msg += "Parameters for UF potential :\n"
        msg += f"      sigma                     {self.sigma}\n"
        msg += f"      p                         {self.p}\n"
        return msg
