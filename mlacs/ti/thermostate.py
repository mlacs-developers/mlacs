"""
// Copyright (C) 2022-2024 MLACS group (PR, AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ..core.manager import Manager
from ..utilities import get_elements_Z_and_masses

from ..state.lammps_state import (BaseLammpsState,
                                  LammpsState)


# ========================================================================== #
# ========================================================================== #
class ThermoState(BaseLammpsState):
    """
    Parent class for the thermodynamic state used in thermodynamic integration

    Parameters
    ----------
    atoms: :class:`ase.Atoms`
        ASE atoms object on which the simulation will be performed

    pair_style: :class:`str`
        pair_style for the LAMMPS input

    pair_coeff: :class:`str` or :class:`list` of :class:`str`
        pair_coeff for the LAMMPS input

    dt: :class:`int` (optional)
        Timestep for the simulations, in fs. Default ``1.5``

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

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created

    blocks : :class:`LammpsBlockInput` or :class:`list` (optional)
        Custom block input class. Can be a list of blocks.
        If ``None``, nothing is added in the input. Default ``None``.
    """
    def __init__(self,
                 atoms,
                 pair_style,
                 pair_coeff,
                 dt=1.5,
                 nsteps=10000,
                 nsteps_eq=5000,
                 nsteps_averaging=10000,
                 rng=None,
                 langevin=False,
                 logfile=None,
                 trajfile=None,
                 interval=500,
                 loginterval=50,
                 blocks=None,
                 neti=True,
                 **kwargs):

        super().__init__(nsteps,
                         nsteps_eq,
                         logfile,
                         trajfile,
                         loginterval=loginterval,
                         blocks=blocks,
                         neti=neti,
                         **kwargs)

        self.atoms = atoms
        self.pair_style = pair_style
        self.pair_coeff = pair_coeff
        self.nsteps = nsteps_averaging
        self.elem, self.Z, self.masses, self.charges = (
            get_elements_Z_and_masses(self.atoms))
        self.dt = dt

# ========================================================================== #
    @Manager.exec_from_path
    def run_averaging(self):
        """
        Get the right volume structure at finite pressure
        """
        eq_state = LammpsState(self.temperature,
                               self.pressure,
                               nsteps=self.nsteps,
                               nsteps_eq=self.nsteps_eq,
                               workdir=self.workdir,
                               folder=self.folder,
                               subfolder='Equilibration')

        self.atoms = eq_state.run_dynamics(self.atoms,
                                           self.pair_style,
                                           self.pair_coeff)

# ========================================================================== #
    def post_process(self):
        """
        """
        pass

# ========================================================================== #
    def _get_block_custom(self):
        """
        """
        return self._get_neti()
