"""
// Copyright (C) 2022-2024 MLACS group (PR, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
import numpy as np

from .thermostate import ThermoState
from ..core.manager import Manager
from ..utilities.thermo import (free_energy_uhlenbeck_ford,
                                free_energy_ideal_gas)

from ..utilities.io_lammps import LammpsBlockInput

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

    sigma: :class:`float`
        sigma parameter of the Uhlenbeck-Ford potential. Default ``2.0``.

    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1``

    damp : :class:`float` (optional)
        Damping parameter.
        If ``None``, a damping parameter of 100 x dt is used.

    pdamp: :class:`float` (optional)
        Pressure damping parameter, used is the pressure is not `None`
        By default, this correspond to 1000 times the timestep.

    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.

    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.

    nsteps_averaging: :class:`int` (optional)
        Number of step for equilibrate ideal structure
        at zero or finite pressure. Default ``10000``.

    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    langevin: :class:`Bool`
        Settle or not a langevin thermostat to equilibrate
        an ideal structure at zero or finite pressure. Default ``True``

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    interval : :class:`int` (optional)
        Number of steps between log and traj writing. Override
        loginterval. Default ``50``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
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
                 pdamp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_averaging=10000,
                 rng=None,
                 langevin=True,
                 logfile=None,
                 trajfile=None,
                 interval=500,
                 loginterval=50,
                 **kwargs):

        kwargs.setdefault('folder', f"LiquidUF_T{temperature}K")

        super().__init__(atoms,
                         pair_style,
                         pair_coeff,
                         dt=dt,
                         nsteps=nsteps,
                         nsteps_eq=nsteps_eq,
                         nsteps_averaging=nsteps_averaging,
                         rng=rng,
                         langevin=langevin,
                         logfile=logfile,
                         trajfile=trajfile,
                         interval=interval,
                         loginterval=loginterval,
                         **kwargs)

        self.atoms = atoms
        self.temperature = temperature
        self.pressure = pressure
        self.damp = damp
        self.pdamp = pdamp

        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2

        self.p = p
        self.sigma = sigma
        self.dt = dt
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.langevin = langevin

        if self.pressure is not None:
            self.equilibrate = True
        else:
            self.equilibrate = False

        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq

        if self.p not in p_tabled:
            msg = "The p value of the UF potential has to be one for " + \
                  "which the free energy of the Uhlenbeck-Ford potential " + \
                  "is tabulated\n" + \
                  "Those value are : 1, 25, 50, 75 and 100"
            raise ValueError(msg)

# ========================================================================== #
    @Manager.exec_from_path
    def run(self):
        """
        """

        if self.equilibrate:
            self.run_averaging()

        self.run_dynamics(self.atoms, self.pair_style, self.pair_coeff)

        with open("MLMD.done", "w") as f:
            f.write("Done")

# ========================================================================== #
    @Manager.exec_from_path
    def postprocess(self):
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
        u_f, lambda_f = np.loadtxt("forward.dat", unpack=True)
        u_b, lambda_b = np.loadtxt("backward.dat", unpack=True)
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
        with open("free_energy.dat", "w") as f:
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
    def _get_block_thermostat(self, eq):
        """
        """
        if self.damp is None:
            self.damp = "$(100*dt)"

        temp = self.temperature
        self.info_dynamics["temperature"] = temp
        if self.langevin:
            langevinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Integrators")
        block("timestep", f"timestep {self.dt / 1000}")
        block("momenta", "velocity all create " +
              f"{temp} {langevinseed} dist gaussian")
        # If we are using Langevin, we want to remove the random part
        # of the forces
        if self.langevin:
            block("rmv_langevin", "fix ff all store/force")
            txt = f"fix f1 all langevin {temp} {temp} {self.damp} " + \
                  f"{langevinseed} zero yes"
            block("langevin", txt)
            block("nve", "fix f2 all nve")
        else:
            block("nvt", f"fix f1 all nvt temp {temp} {temp} {self.damp}")
        block("compute temp without cm", "compute c1 all temp/com")
        block("fix cm", "fix_modify f1 temp c1")
        return block

# ========================================================================== #
    def _get_block_traj(self, el):
        """
        """
        if self.trajfile:
            block = LammpsBlockInput("dump", "Dumping")
            txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} "
            txt += "id type xu yu zu vx vy vz fx fy fz "
            txt += "element"
            block("dump", txt)
            block("dump_modify1", "dump_modify dum1 append yes")
            txt = "dump_modify dum1 element " + " ".join([p for p in el])
            block("dump_modify2", txt)
            return block
        else:
            pass

# ========================================================================== #
    def _get_neti(self):
        """
        """
        blocks = []
        pair_style = self.pair_style.split()
        if len(self.pair_coeff) == 1:
            pair_coeff = self.pair_coeff[0].split()
            hybrid_pair_coeff = " ".join([*pair_coeff[:2],
                                          pair_style[0],
                                          *pair_coeff[2:]])
        else:
            hybrid_pair_coeff = []
            for pc in self.pair_coeff:
                pc_ = pc.split()
                hpc_ = " ".join([*pc_[:2], *pc_[2:]])
                hybrid_pair_coeff.append(hpc_)

        block0 = LammpsBlockInput("ufm params", "UF potential parameters")
        block0("p", f"variable p equal {self.p}")
        block0("temp", f"variable T equal {self.temperature}")
        block0("kB", "variable kB equal 8.6173303e-5")
        block0("eps", "variable eps equal ${T}*${p}*${kB}")
        block0("sigma", f"variable sig equal {self.sigma}")
        block0("rc", "variable rc equal 5.0*${sig}")
        blocks.append(block0)

        block1 = LammpsBlockInput("eq fwd",
                                  "Equilibration without UF potential")
        block1("run eq fwd", f"run {self.nsteps_eq}")
        blocks.append(block1)

        block2 = LammpsBlockInput("fwd", "Forward Integration")
        block2("tau", "variable tau equal ramp(1,0)")
        txt = "variable lambda_true equal " + \
              "v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)"
        block2("lambda_true", txt)
        block2("lambda_ufm", "variable lambda_ufm equal 1-v_lambda_true")
        if len(self.pair_coeff) == 1:
            txt = "pair_style hybrid/scaled " + \
                 f"v_lambda_true {self.pair_style} " + \
                  "v_lambda_ufm ufm ${rc}\n"
            block2("scaling pair_style", txt)
            txt = "pair_coeff " + hybrid_pair_coeff
            block2("true_pair_coeff", txt)
            block2("ufm_pair_coeff", "pair_coeff * * ufm ${eps} ${sig}")
        else:
            # pair_style comd compatible only with one zbl, To be fixed
            txt = "pair_style hybrid/scaled " + \
                  f"{pair_style[1]} {pair_style[2]} " + \
                  f"{pair_style[3]} v_lambda_true " + \
                  f"{pair_style[4]} v_lambda_ufm ufm ${{rc}}\n"
            block2("scaling pair_style", txt)
            txt = "pair_coeff" + hybrid_pair_coeff[0]
            block2("true_pair_coeff_1", txt)
            txt = "pair_coeff" + hybrid_pair_coeff[1]
            block2("true_pair_coeff_1", txt)
            block2("ufm_pair_coeff", "pair_coeff * * ufm ${eps} ${sig}")

        if len(self.pair_coeff) == 1:
            block2("compute pair true", f"compute c2 all pair {pair_style[0]}")
            block2("compute pair ufm", "compute c3 all pair ufm")
            block2("dU", "variable dU equal (c_c2-c_c3)/atoms")
        else:
            # pair_style comd compatible only with one zbl, To be fixed
            block2("compute pair 1", f"compute c2 all pair {pair_style[1]}")
            block2("compute pair 2", f"compute c4 all pair {pair_style[4]}")
            block2("compute pair ufm", "compute c3 all pair ufm")
            block2("dU", "variable dU equal ((c_c2+c_c4)-c_c3)/atoms")

        block2("lamb", "variable lamb equal 1-v_lambda_true")
        block2("write fwd", "fix  f3 all print 1 \"${dU}  ${lamb}\" title " +
                            "\"# dU lambda\" screen no append forward.dat")
        block2("run", f"run {self.nsteps}")
        block2("unfix write fwd", "unfix f3")
        blocks.append(block2)

        block3 = LammpsBlockInput("eq bwd",
                                  "Equilibration with only UF potential")
        block3("run eq bwd", f"run {self.nsteps_eq}")
        blocks.append(block3)

        block4 = LammpsBlockInput("bwd", "Backrward Integration")
        block4("tau", "variable tau equal ramp(0,1)")
        txt = "variable lambda_true equal " + \
              "v_tau^5*(70*v_tau^4-315*v_tau^3+540*v_tau^2-420*v_tau+126)"
        block4("lambda_true", txt)
        block4("lambda_ufm", "variable lambda_ufm equal 1-v_lambda_true")
        if len(self.pair_coeff) == 1:
            txt = "pair_style hybrid/scaled " + \
                  f"v_lambda_true {self.pair_style} " + \
                  "v_lambda_ufm ufm ${rc}\n"
            block4("scaling pair_style", txt)
            txt = "pair_coeff " + hybrid_pair_coeff
            block4("true_pair_coeff", txt)
            block4("ufm_pair_coeff", "pair_coeff * * ufm ${eps} ${sig}")
        else:
            # pair_style comd compatible only with one zbl, To be fixed
            txt = "pair_style hybrid/scaled " + \
                  f"{pair_style[1]} {pair_style[2]} " + \
                  f"{pair_style[3]} v_lambda_true " + \
                  f"{pair_style[4]} v_lambda_ufm ufm ${{rc}}\n"
            block4("scaling pair_style", txt)
            txt = "pair_coeff " + hybrid_pair_coeff[0]
            block4("true_pair_coeff_1", txt)
            txt = "pair_coeff " + hybrid_pair_coeff[1]
            block4("true_pair_coeff_1", txt)
            block4("ufm_pair_coeff", "pair_coeff * * ufm ${eps} ${sig}")
        block4("write bwd", "fix  f3 all print 1 \"${dU}  ${lamb}\" title " +
                            "\"# dU lambda\" screen no append backward.dat")
        blocks.append(block4)

        return blocks

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
