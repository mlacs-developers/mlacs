"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.lammpsrun import LAMMPS

from ..core.manager import Manager
from .state import StateManager


# ========================================================================== #
# ========================================================================== #
class LangevinState(StateManager):
    """
    State Class for running a Langevin simulation as implemented in ASE

    Parameters
    ----------
    temperature : :class:`float`
        Temperature of the simulation, in Kelvin

    friction : :class:`float` (optional)
        Friction coefficient of the thermostat. Default 0.01.

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default True.

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
        Default correspond to ``numpy.random.default_rng()``.

    init_momenta : :class:`numpy.ndarray` (optional)
        If ``None``, velocities are initialized with a
        Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    """
    def __init__(self,
                 temperature,
                 friction=0.01,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 rng=None,
                 init_momenta=None,
                 **kwargs):

        StateManager.__init__(self,
                              nsteps=nsteps,
                              nsteps_eq=nsteps_eq,
                              logfile=logfile,
                              trajfile=trajfile,
                              loginterval=loginterval,
                              **kwargs)
        self.pressure = None
        self.dt = dt
        self.fixcm = fixcm
        self.temperature = temperature
        self.dt = dt
        self.fixcm = fixcm
        self.friction = friction
        self.init_momenta = init_momenta
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.pressure = None

        self.info_dynamics = dict(ensemble="NVT",
                                  temperature=self.temperature)

        self.ispimd = False
        self.isrestart = False
        self.isneb = False

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     nbeads=1,
                     elements=None):
        """
        """
        atoms = supercell.copy()
        calc = LAMMPS(pair_style=pair_style, pair_coeff=pair_coeff,
                      atom_style=atom_style,
                      # keep_tmp_files=True, # DEBUG
                      # tmp_dir=str(self.workdir/'tmp_lammps'),  # DEBUG
                      )

        if model_post is not None:
            calc.set(model_post=model_post)

        atoms.calc = calc

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        dyn = Langevin(atoms,
                       self.dt*fs,
                       temperature_K=self.temperature,
                       friction=self.friction,
                       fixcm=self.fixcm,
                       rng=self.rng)
        if self.trajfile is not None:
            trajectory = Trajectory(self.trajfile, mode="a", atoms=atoms)
            dyn.attach(trajectory.write, interval=self.loginterval)

        if self.logfile is not None:
            dyn.attach(MDLogger(dyn, atoms, self.logfile, stress=True),
                       interval=self.loginterval)
        dyn.run(nsteps)
        dyn.atoms.info['info_state'] = self.info_dynamics
        return dyn.atoms

# ========================================================================== #
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms,
                                         temperature_K=self.temperature,
                                         rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        msg = "NVT Langevin dynamics as implemented in ASE\n"
        msg += f"Temperature (in Kelvin)                {self.temperature}\n"
        msg += f"Number of MLMD equilibration steps :   {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :      {self.nsteps}\n"
        msg += f"Timestep (in fs) :                     {self.dt}\n"
        msg += f"Friction (in fs) :                     {self.friction / fs}\n"
        msg += "\n"
        return msg
