"""
// Copyright (C) 2022-2024 MLACS group (PR, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.units import kB, GPa

from ..core.manager import Manager
from ..utilities.thermo import (free_energy_harmonic_oscillator,
                                free_energy_com_harmonic_oscillator)
from ..utilities.io_lammps import get_msd_input
from ..utilities.io_lammps import LammpsBlockInput

from ..state.lammps_state import LammpsState
from .thermostate import ThermoState


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
        Timestep for the simulations, in fs. Default ``1``

    damp : :class:`float` (optional)
        Damping parameter.
        If ``None``, a damping parameter of  1000 x dt is used.

    pdamp: :class:`float` (optional)
        Pressure damping parameter, used is the pressure is not `None`
        By default, this correspond to 1000 times the timestep.

    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.

    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.

    nsteps_md: :class:`int` (optional)
        Number of steps used to compute the spring constants.
        Default `25000`

    nsteps_averaging: :class:`int` (optional)
        Number of step for equilibrate ideal structure
        at zero or finite pressure.
        Default ``10000``.

    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    langevin: :class:`bool`
        Whether to use a langevin thermostat. Default `True`

    logfile : :class:`Bool` (optional)
        Activate file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Activate Name of the file for saving the MLMD trajectory dump.
        If ``None``, no dump file is created. Default ``None``.

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
                 k=None,
                 dt=1,
                 damp=None,
                 pdamp=None,
                 nsteps=50000,
                 nsteps_eq=5000,
                 nsteps_msd=25000,
                 nsteps_averaging=10000,
                 rng=None,
                 langevin=True,
                 logfile=None,
                 trajfile=None,
                 interval=500,
                 loginterval=50,
                 **kwargs):

        kwargs.setdefault('folder', f"Solid_T{temperature}K")

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
        self.nsteps_msd = nsteps_msd
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.langevin = langevin

        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2

        if self.pressure is not None:
            self.equilibrate = True
        else:
            self.equilibrate = False

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

        self.dt = dt

# ========================================================================== #
    @Manager.exec_from_path
    def run(self):
        """
        """
        if self.equilibrate:
            self.run_averaging()

        if self.k is None:
            # First get optimal spring constant
            self.compute_msd()

        self.run_dynamics(self.atoms, self.pair_style, self.pair_coeff)

# ========================================================================== #
    @Manager.exec_from_path
    def compute_msd(self):
        """
        """
        msd_state = LammpsState(self.temperature,
                                pressure=self.pressure,
                                nsteps=self.nsteps_msd,
                                nsteps_eq=self.nsteps_eq,
                                lammpsfname='lammps_input_msd.in',
                                blocks=get_msd_input(self, 'msd.dat'),
                                workdir=self.workdir,
                                folder=self.folder,
                                subfolder='MSD',
                                rng=self.rng)

        msd_state.run_dynamics(self.atoms,
                               self.pair_style,
                               self.pair_coeff)

        kall = []
        with open("msd.dat", "w") as f:
            for e in self.elem:
                data = np.loadtxt(str(msd_state.path / f"msd{e}.dat"))
                nat = np.count_nonzero([a == e for a in
                                        self.atoms.get_chemical_symbols()])
                k = 3 * kB * self.temperature / data.mean()
                kall.append(k)
                f.write(e + " {0}   {1:10.5f}\n".format(nat, k))
        self.k = kall

# ========================================================================== #
    @Manager.exec_from_path
    def postprocess(self):
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
        dE_f, lambda_f = np.loadtxt("forward.dat", unpack=True)
        dE_b, lambda_b = np.loadtxt("backward.dat", unpack=True)
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
            pv = self.pressure*GPa*vol/nat_tot
        else:
            pv = 0.0
        with open("free_energy.dat", "w") as f:
            header = "#   T [K]     Fe tot [eV/at]     " + \
                     "Fe harm [eV/at]      Work [eV/at]     " + \
                     "Fe com [eV/at]      PV [eV/at]"
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
        # add Fe or Fe_corrected to return to be read for cv purpose and RS
        if self.fcorr1 is not None or self.fcorr2 is not None:
            free_energy = free_energy_corrected
        if self.pressure:
            free_energy += pv
        return msg, free_energy

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
        Write the LAMMPS input neti in solids
        """
        blocks = []
        block0 = LammpsBlockInput("einstein params",
                                  "Integrators and variables for solid neti")
        for iel, el in enumerate(self.elem):
            txt = f"fix ff{el} {el} ti/spring " + \
                  f"{self.k[iel]} {self.nsteps} {self.nsteps_eq} " + \
                  "function 2"
            block0(f"ti_spring_{el}", txt)
            block0("dE", f"variable dE equal (pe-f_ff{el})/atoms")
        block0("lambda", f"variable lambda equal f_ff{self.elem[0]}[1]")
        blocks.append(block0)

        block1 = LammpsBlockInput("eq fwd",
                                  "Equilibration without Einstein potential")
        block1("run eq fwd", f"run {self.nsteps_eq}")
        blocks.append(block1)

        block2 = LammpsBlockInput("fwd", "Forward Integration")
        block2("write fwd", "fix f4 all print 1 \"${dE} ${lambda}\" " +
                            "screen no append forward.dat title " +
                            "\"# pe  lambda\"")
        block2("run fwd", f"run {self.nsteps}")
        block2("unifx write fwd", "unfix f4")
        blocks.append(block2)

        block3 = LammpsBlockInput("eq bwd",
                                  "Equilibration with only Einstein potential")
        block3("run eq bwd", f"run {self.nsteps_eq}")
        blocks.append(block3)

        block4 = LammpsBlockInput("bwd", "Backward Integration")
        block4("write bwd", "fix f4 all print 1 \"${dE} ${lambda}\" " +
                            "screen no append backward.dat title " +
                            "\"# pe  lambda\"")
        blocks.append(block4)

        return blocks

        block4 = LammpsBlockInput("bwd", "Backward Integration")
        block4("write bwd", "fix f4 all print 1 \"${dE} ${lambda}\" " +
               "screen no append backward.dat title \"# pe  lambda\"")
        blocks.append(block4)

        return blocks

        block4 = LammpsBlockInput("bwd", "Backward Integration")
        block4("write bwd", "fix f4 all print 1 \"${dE} ${lambda}\" " +
               "screen no append backward.dat title \"# pe  lambda\"")
        blocks.append(block4)

        return blocks

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
