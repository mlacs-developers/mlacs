"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

from .lammps_state import BaseLammpsState
from ..core.manager import Manager
from ..utilities.io_lammps import (LammpsBlockInput,
                                   write_atoms_lammps_spin_style)


_default_params = dict(temperature=300,
                       t_stop=None,
                       damp=None,
                       dt=1.5,
                       fixcm=True)


# ========================================================================== #
# ========================================================================== #
class SpinLammpsState(BaseLammpsState):
    """
    Class to manage magnetic spin simulations with LAMMPS and the SPIN package.

    Tranchida, et al., Journal of Computational Physics, 372, 406-425, (2018).

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.

    t_stop: :class:`float` or ``None`` (optional)
        When this input is not ``None``, the temperature of
        the molecular dynamics simulations is randomly chosen
        in a range between `temperature` and `t_stop`.
        Default ``None``

    damp: :class:`float` or ``None`` (optional)

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.

    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    init_momenta : :class:`numpy.ndarray` (optional)
        Gives the (Nat, 3) shaped momenta array that will be used
        to initialize momenta when using
        the `initialize_momenta` function.
        If the default ``None`` is set, momenta are initialized with a
        Maxwell Boltzmann distribution.
    """
    def __init__(self, temperature, t_stop=None, damp=None, dt=1.5, fixcm=True,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, blocks=None, **kwargs):

        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         blocks, **kwargs)

        self.temperature = temperature
        self.t_stop = t_stop
        self.damp = damp
        self.dt = dt
        self.fixcm = fixcm

        if self.rng is None:
            self.rng = np.random.default_rng()
        if self.damp is None:
            self.damp = "$(100*dt)"

# ========================================================================== #
    def _get_block_init(self, atom_style, pbc, el, masses):
        """

        """
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))

        block = LammpsBlockInput("init", "Initialization")
        block("map", "atom_modify map array")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", "atom_style spin")
        block("read_data", "read_data atoms.in")
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
        return block

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """
        Function to write the thermostat of the mlmd run
        """
        if self.t_stop is None:
            temp = self.temperature
        else:
            tmp_temp = np.sort([self.temperature, self.t_stop])
            if eq:
                temp = np.max(tmp_temp)
            else:
                temp = self.rng.uniform(*tmp_temp)
        langevinseed = self.rng.integers(1, 9999999)
        spinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Thermostat")
        block("timestep", f"timestep {self.dt / 1000}")
        block("rmv_langevin", "fix ff all store/force")

        txt = f"fix f1 all langevin {temp} {temp} {self.damp} " + \
              f"{langevinseed} gjf {self.gjf} zero yes"
        block("langevin", txt)
        txt = f"fix fspin all langevin/spin {temp} {self.damp} {spinseed}"
        block("nvt_spin", txt)
        block("nve_spin", "fix fspin2 all nve/spin lattice moving")
        if self.fixcm:
            block("cm", "fix fcm all recenter INIT INIT INIT")
        return block

# ========================================================================== #
    def _get_block_traj(self, el):
        """

        """
        block = LammpsBlockInput("traj", "Dumping trajectory")
        txt = "compute spin all property/atom sp spx spy spz fmx fmy fmz"
        block("compute", txt)
        txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} " + \
              "id type xu yu zu vx vy vz fx fy fz " + \
              "f_ff[1] f_ff[2] f_ff[3] c_spin[1] " + \
              "c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]" + \
              " element"
        block("dump", txt)
        block("dump_modify1", "dump_modify dum1 append yes")
        txt = "dump_modify dum1 element " + " ".join([p for p in el])
        block("dump_modify2", txt)
        return block

# ========================================================================== #
    def _get_block_log(self):
        """

        """
        block = LammpsBlockInput("log", "Logging")
        # First we compute the magnetic quantities
        block("compute", "compute out_mag all spin")
        block("magx", "variable magx equal c_out_mag[1]")
        block("magy", "variable magy equal c_out_mag[2]")
        block("magz", "variable magz equal c_out_mag[3]")
        block("magn", "variable magn equal c_out_mag[4]")
        block("mage", "variable mage equal c_out_mag[5]")
        block("magt", "variable magt equal c_out_mag[6]")

        variables = ["t equal step", "mytemp equal temp",
                     "mype equal pe", "myke equal ke", "myetot equal etotal",
                     "mypress equal press/10000", "vol equal (lx*ly*lz)"]
        for i, var in enumerate(variables):
            block(f"variable{i}", f"variable {var}")
        txt = f"fix mylog all print {self.loginterval} \""
        variables = ["$t", "${mytemp}", "$(v_magt)", "${vol}", "${myetot}",
                     "${myke}", "$(v_mage)", "${mypress}",
                     "$(v_magx)", "$(v_magy)", "$(v_magz)", "$(v_magn)"]
        txt += " ".join(variables) + "\" "
        txt += f"append {self.logfile} title "
        titles = ["Step", "T_at", "T_spin", "Vol", "Etot", "Epot", "Ekin",
                  "Espin", "Press", "Magn_x", "Magn_y", "Magn_z", "Magn_norm"]
        txt += "\"# " + " ".join(titles) + "\""
        block("fix", txt)
        return block

# ========================================================================== #
    def _get_block_lastdump(self, el, eq):
        block = LammpsBlockInput("lastdump", "Dump last configuration")
        txt = "dump last all custom 1 configurations.out " + \
              "id type xu yu zu vx vy vz fx fy fz c_spin[1] " + \
              "c_spin[2] c_spin[3] c_spin[4] c_spin[5] c_spin[6] c_spin[7]" + \
              " element"
        block("dump", txt)
        txt = "dump_modify last element " + " ".join([p for p in el])
        block("dump_modify1", txt)
        block("run_dump", "run 0")
        return block

# ========================================================================== #
    @Manager.exec_from_path
    def _write_lammps_atoms(self, atoms, atom_style, elements=None):
        """

        """
        spins = atoms.get_array("spins")
        with open("atoms.in", "w") as fd:
            write_atoms_lammps_spin_style(fd, atoms, spins)
