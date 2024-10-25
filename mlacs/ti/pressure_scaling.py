"""
// Copyright (C) 2022-2024 MLACS group (PR)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import numpy as np
from scipy.integrate import cumtrapz
from ase.units import kB

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

    ninstance: :class:`int` (optional)
        If Free energy calculation has to be done before temperature sweep
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
                 gjf=True,
                 rng=None,
                 suffixdir=None,
                 logfile=True,
                 trajfile=True,
                 interval=500,
                 loginterval=50):

        self.atoms = atoms
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
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
        
        # reversible scaling
        ThermoState.__init__(self,
                             atoms,
                             pair_style,
                             pair_coeff,
                             dt,
                             nsteps,
                             nsteps_eq,
                             rng=rng,
                             logfile=logfile,
                             trajfile=trajfile,
                             interval=interval,
                             loginterval=loginterval)

        self.suffixdir = f"PressureScaling_P{self.p_start}GPa_P{self.p_end}GPa"
        self.suffixdir += "_{0}K".format(self.temperature)
        self.suffixdir += "/"
        if suffixdir is not None:
            self.suffixdir = suffixdir
        if self.suffixdir[-1] != "/":
            self.suffixdir += "/"

# ========================================================================== #
    @Manager.exec_from_path
    def run(self, wdir):
        """
        """
        if not os.path.exists(wdir):
            os.makedirs(wdir)

        self.workdir = wdir #to be transmitted by the BaseLammpsState class
        
        if self.g_init is None:
            self.run_single_ti()

        self.run_dynamics(self.atoms, self.pair_style, self.pair_coeff)

        with open(wdir + "MLMD.done", "w") as f:
            f.write("Done")

# ========================================================================== #
    def run_single_ti(self):
        """
        Free energy calculation before sweep
        """
        if self.phase == 'solid':
            self.state = EinsteinSolidState(self.atoms,
                                            self.pair_style,
                                            self.pair_coeff,
                                            self.temperature,
                                            self.p_start,
                                            self.fcorr1,
                                            self.fcorr2,
                                            k=None,
                                            dt=self.dt)
        elif self.phase == 'liquid':
            self.state = UFLiquidState(self.atoms,
                                       self.pair_style,
                                       self.pair_coeff,
                                       self.temperature,
                                       self.p_start,
                                       self.fcorr1,
                                       self.fcorr2,
                                       dt=self.dt)
            
        self.ti = ThermodynamicIntegration(self.state,
                                           ninstance=self.ninstance,
                                           logfile='FreeEnergy.log')
        self.ti.run()
        
        # Get G
        if self.ninstance == 1:
            _, self.g_init = self.state.postprocess(self.ti.get_fedir())
        elif self.ninstance > 1:
            tmp = []
            for i in range(self.ninstance):
                _, tmp_g_init = self.state.postprocess(
                    self.ti.get_fedir() + f"for_back_{i+1}/")
                tmp.append(tmp_g_init)
            self.g_init = np.mean(tmp)

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
        block("momenta", f"velocity all create " +\
              f"{temp} {langevinseed} dist gaussian")
        return block

# ========================================================================== #
    def _get_block_traj(self, atoms):
        """
        """
        if self.trajfile:
            el, Z, masses, charges = get_elements_Z_and_masses(atoms)
            block = LammpsBlockInput("dump", "Dumping")
            txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} " + \
                  "id type xu yu zu vx vy vz fx fy fz "
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

        block0 = LammpsBlockInput("eq fwd", "Equilibration before fwd rs")
        block0("eq fwd npt", f"fix f1 all npt temp {temp} {temp} {self.damp} " + \
              f"iso {self.p_start*10000} {self.p_start*10000} {self.pdamp}")
        block0("run eq fwd", f"run {self.nsteps_eq}")
        block0("unfix eq fwd", f"unfix f1")
        blocks.append(block0)

        block1 = LammpsBlockInput("fwd", "Forward Integration")
        block1("lambda fwd", f"variable lambda equal ramp({li*10000},{lf*10000})")
        block1("pp", f"variable pp equal ramp({self.p_start*10000},{self.p_end*10000})")
        block1("fwd npt", f"fix f2 all npt temp {temp} {temp} {self.damp} " + \
              f"iso {self.p_start*10000} {self.p_end*10000} {self.pdamp}")
        block1("write fwd", "fix f3 all print 1 " + \
               "\"$(pe/atoms) ${pp} ${vol} ${lambda}\" screen no " + \
               "append forward.dat title \"# de                   pressure  vol         lambda\"\n")
        block1("run fwd", f"run {self.nsteps}")
        block1("unfix fwd npt", f"unfix f2")
        block1("unfix f3", "unfix f3 ")
        blocks.append(block1)

        block2 = LammpsBlockInput("eq bwd", "Equilibration before bwd rs")
        block2("eq bwd npt", f"fix f1 all npt temp {temp} {temp} {self.damp} " + \
              f"iso {self.p_end*10000} {self.p_end*10000} {self.pdamp}")
        block2("run eq bwd", f"run {self.nsteps_eq}")
        block2("unfix eq bwd", f"unfix f1")
        blocks.append(block2)

        block3 = LammpsBlockInput("bwd", "Backward Integration")
        block3("lambda bwd", f"variable lambda equal ramp({lf*10000},{li*10000})")
        block3("pp", f"variable pp equal ramp({self.p_end*10000},{self.p_start*10000})")
        block3("bwd npt", f"fix f2 all npt temp {temp} {temp} {self.damp} " + \
              f"iso {self.p_end*10000} {self.p_start*10000} {self.pdamp}")
        block3("write bwd", "fix f3 all print 1 " + \
               "\"$(pe/atoms) ${pp} ${vol} ${lambda}\" screen no " + \
               "append backward.dat title \"# de                  pressure  vol         lambda\"\n")
        blocks.append(block3)
        return blocks

# ========================================================================== #
    @Manager.exec_from_path
    def postprocess(self, wdir):
        """
        Compute the free energy from the simulation
        """
        natoms = len(self.atoms)
        
        # Get data
        _, fp, fvol, _ = np.loadtxt(wdir+"forward.dat", unpack=True)
        _, bp, bvol, _ = np.loadtxt(wdir+"backward.dat", unpack=True)

        # pressure contribution
        fvol = fvol / natoms
        bvol = bvol / natoms

        fp = fp / (10000*160.21766208)
        bp = bp / (10000*160.21766208)

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
        np.savetxt(wdir + "free_energy.dat", results, header=header, fmt=fmt)
        return 

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
