"""
// Copyright (C) 2022-2024 MLACS group (PR)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
try:  # Scipy >= 1.6.0
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:  # Scipy < 1.6.0
    from scipy.integrate import cumtrapz
from ase.units import GPa

from ..core.manager import Manager
from ..utilities.io_lammps import LammpsBlockInput

from .thermostate import ThermoState
from .solids import EinsteinSolidState
from .liquids import UFLiquidState
from .thermoint import ThermodynamicIntegration


# ========================================================================== #
# ========================================================================== #
class PressureScalingState(ThermoState):
    """
    Class for performing thermodynamic integration for a
    range of pressure using pressure scaling (in NPT).

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE atoms object on which the simulation will be performed

    pair_style: :class:`str`
        pair_style for the LAMMPS input

    pair_coeff: :class:`str` or :class:`list` of :class:`str`
        pair_coeff for the LAMMPS input

    fcorr1: :class:`float` or ``None``
        First order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.

    fcorr2: :class:`float` or ``None``
        Second order cumulant correction to the free energy, in eV/at,
        to be added to the results.
        If ``None``, no value is added. Default ``None``.

    p_start: :class:`float` (optional)
        Initial pressure of the simulation, in GPa. Default ``0``.

    p_end: :class:`float` (optional)
        Final pressure of the simulation, in GPa. Default ``10``.

    g_init: :class:`float` (optional)
        Free energy of the initial temperature, in eV/at. Default ``None``.

    phase: :class:`str`
        The phase of the system for which the free energy is computed.
        This input is used to compute the reference free energy at the
        starting pressure using a EinsteinSolidState object for solids
        and a UFLiquidState object for liquids. Can be either 'solid'
        or 'liquid'

    ninstance: :class:`int` (optional)
        If a reference Free energy calculation has to be done before
        the temperature sweep
        Settles the number of forward and backward runs. Default ``1``.

    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``

    damp : :class:`float` (optional)
        Damping parameter. If ``None``, a damping parameter of a
        hundred time the timestep is used.

    temperature: :class:`float` or ``None``
        Temperature of the simulation.
        Default ``300``.

    pdamp : :class:`float` (optional)
        Damping parameter for the barostat. Default 1000 times ``dt`` is used.
        Default ``None``.

    nsteps: :class:`int` (optional)
        Number of production steps. Default ``10000``.

    nsteps_eq: :class:`int` (optional)
        Number of equilibration steps. Default ``5000``.

    nsteps_averaging: :class:`int` (optional)
        Number of steps done to equilibrate the system at the start pressure.
        Default `10000`

    gjf: :class:`bool`
        Whether to use the GJF integrator, if the Langevin thermostat is used.
        Default `True`

    rng: :class:`RNG object`
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    langevin: :class:`bool`
        Whether to use a langevin thermostat. Default `True`

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
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 fcorr1=None,
                 fcorr2=None,
                 p_start=0,
                 p_end=10,
                 g_init=None,
                 phase=None,
                 ninstance=1,
                 dt=1,
                 damp=None,
                 temperature=300,
                 pdamp=None,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_averaging=10000,
                 gjf=True,
                 rng=None,
                 langevin=True,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50,
                 **kwargs):

        fname = f"PressureScaling_P{p_start}_P{p_end}GPa_T{temperature}K"
        kwargs.setdefault('folder', fname)

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
                         **kwargs)

        self.atoms = atoms
        self.fcorr1 = fcorr1
        self.fcorr2 = fcorr2
        self.p_start = p_start
        self.p_end = p_end
        self.g_init = g_init
        self.phase = phase
        self.ninstance = ninstance
        self.damp = damp
        self.temperature = temperature
        self.pdamp = pdamp
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.gjf = gjf
        self.dt = dt
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

# ========================================================================== #
    @Manager.exec_from_path
    def run(self):
        """
        """
        if self.g_init is None:
            self.run_single_ti()

        self.run_dynamics(self.atoms, self.pair_style, self.pair_coeff)

# ========================================================================== #
    def run_single_ti(self):
        """
        Free energy calculation before sweep
        """
        folder = "Fe_ref"
        if self.phase == 'solid':
            self.state = EinsteinSolidState(self.atoms,
                                            self.pair_style,
                                            self.pair_coeff,
                                            self.temperature,
                                            self.p_start,
                                            self.fcorr1,
                                            self.fcorr2,
                                            k=None,
                                            dt=self.dt,
                                            folder=folder)
        elif self.phase == 'liquid':
            self.state = UFLiquidState(self.atoms,
                                       self.pair_style,
                                       self.pair_coeff,
                                       self.temperature,
                                       self.p_start,
                                       self.fcorr1,
                                       self.fcorr2,
                                       dt=self.dt,
                                       folder=folder)

        self.ti = ThermodynamicIntegration(self.state,
                                           ninstance=self.ninstance,
                                           logfile='FreeEnergy.log')
        self.g_init = self.ti.run().mean(axis=1)[0]
        return self.g_init

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """
        """
        if self.damp is None:
            self.damp = "$(100*dt)"

        if self.pdamp is None:
            self.pdamp = "$(1000*dt)"

        temp = self.temperature
        self.info_dynamics["temperature"] = temp
        langevinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Integrators")
        block("timestep", f"timestep {self.dt / 1000}")
        block("momenta", "velocity all create " +
              f"{temp} {langevinseed} dist gaussian")
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
        li = 1
        lf = self.p_end
        temp = self.temperature

        blocks = []

        block0 = LammpsBlockInput("eq fwd", "Equilibration before fwd rs")
        line = f"fix f1 all npt temp {temp} {temp} {self.damp} "
        line += f"iso {self.p_start*10000} {self.p_start*10000} {self.pdamp}"
        block0("eq fwd npt", line)
        block0("run eq fwd", f"run {self.nsteps_eq}")
        block0("unfix eq fwd", "unfix f1")
        blocks.append(block0)

        block1 = LammpsBlockInput("fwd", "Forward Integration")
        block1("lambda fwd", "variable lambda equal " +
               f"ramp({li*10000},{lf*10000})")
        block1("pp", "variable pp equal " +
               f"ramp({self.p_start*10000},{self.p_end*10000})")
        block1("fwd npt", f"fix f2 all npt temp {temp} {temp} {self.damp} " +
               f"iso {self.p_start*10000} {self.p_end*10000} {self.pdamp}")
        block1("write fwd", "fix f3 all print 1 " +
               "\"$(pe/atoms) ${pp} ${vol} ${lambda}\" screen no " +
               "append forward.dat title \"# de                  " +
               " pressure  vol         lambda\"\n")
        block1("run fwd", f"run {self.nsteps}")
        block1("unfix fwd npt", "unfix f2")
        block1("unfix f3", "unfix f3 ")
        blocks.append(block1)

        block2 = LammpsBlockInput("eq bwd", "Equilibration before bwd rs")
        block2("eqbwd npt", f"fix f1 all npt temp {temp} {temp} {self.damp} " +
               f"iso {self.p_end*10000} {self.p_end*10000} {self.pdamp}")
        block2("run eq bwd", f"run {self.nsteps_eq}")
        block2("unfix eq bwd", "unfix f1")
        blocks.append(block2)

        block3 = LammpsBlockInput("bwd", "Backward Integration")
        block3("lambda bwd", "variable lambda equal " +
               f"ramp({lf*10000},{li*10000})")
        block3("pp", "variable pp equal " +
               f"ramp({self.p_end*10000},{self.p_start*10000})")
        block3("bwd npt", f"fix f2 all npt temp {temp} {temp} {self.damp} " +
               f"iso {self.p_end*10000} {self.p_start*10000} {self.pdamp}")
        block3("write bwd", "fix f3 all print 1 " +
               "\"$(pe/atoms) ${pp} ${vol} ${lambda}\" screen no " +
               "append backward.dat title \"# de                 " +
               " pressure  vol         lambda\"\n")
        blocks.append(block3)
        return blocks

# ========================================================================== #
    @Manager.exec_from_path
    def postprocess(self):
        """
        Compute the free energy from the simulation
        """
        natoms = len(self.atoms)

        # Get data
        _, fp, fvol, _ = np.loadtxt("forward.dat", unpack=True)
        _, bp, bvol, _ = np.loadtxt("backward.dat", unpack=True)

        # pressure contribution
        fvol = fvol / natoms
        bvol = bvol / natoms

        fp = fp / 10000 * GPa
        bp = bp / 10000 * GPa

        # Integrate the forward and backward data
        wf = cumtrapz(fvol, fp, initial=0)
        wb = cumtrapz(bvol[::-1], bp[::-1], initial=0)
        # Compute the total work
        work = (wf + wb) / 2

        pressure = np.linspace(self.p_start, self.p_end, len(work))

        free_energy = self.g_init + work

        results = np.array([pressure, free_energy]).T
        header = "p [GPa]    G [eV/at]"
        fmt = "%10.6f    %10.6f"
        np.savetxt("free_energy.dat", results, header=header, fmt=fmt)

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        msg = "Thermodynamic Integration using Pressure Scaling\n"
        msg += f"Starting pressure :          {self.p_start} GPa\n"
        msg += f"Stopping pressure :          {self.p_end} GPa\n"
        msg += f"Pressure damping :              {self.pdamp} fs\n"
        msg += f"Temperature damping :              {self.damp} fs\n"
        msg += f"Timestep :                      {self.dt} fs\n"
        msg += f"Number of steps :               {self.nsteps}\n"
        msg += f"Number of equilibration steps : {self.nsteps_eq}\n"
        return msg
