"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from .lammps_state import BaseLammpsState
from ..utilities.io_lammps import (LammpsBlockInput,
                                   EmptyLammpsBlockInput)


# ========================================================================== #
# ========================================================================== #
class OptimizeLammpsState(BaseLammpsState):
    """
    Class to manage geometry optimizations with LAMMPS.

    Parameters
    ----------
    min_style: :class:`str`
        Choose a minimization algorithm to use when a minimize command is
        performed.
        The options are the one available with the ``min_style`` command
        of LAMMPS.

        - `cg`
        - `hftn`
        - `sd`
        - `quickmin`
        - `fire`

        Default `cg`.

    etol: :class:`float`
        Stopping tolerance for energy
        Default ``0.0``

    ftol: :class:`float`
        Stopping tolerance for forces
        Default ``1.0e-6``

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``0.5`` fs.

    pressure: :class:`float` or ``None`` (optional)
        Target pressure for the optimization, in GPa.
        Only available if min_style is 'cg'.
        If ``None``, no cell relaxation is applied.
        Default ``None``

    ptype: ``iso`` or ``aniso`` (optional)
        Only available if min_style is 'cg'.
        Handle the type of pressure applied. Default ``iso``

    vmax: ``iso`` or ``aniso`` (optional)
        The vmax keyword can be used to limit the fractional change in the
        volume of the simulation box that can occur in one iteration of
        the minimizer.
        Default ``1.0e-3``

    nsteps : :class:`int` (optional)
        Maximum number of minimizer iterations during production phase.
        Also sets up the max number of force/energy evaluations.
        Default ``10000`` steps.

    nsteps_eq : :class:`int` (optional)
        Maximum number of minimizer iterations during equilibration phase.
        Also sets up the max number of force/energy evaluations.
        Default ``1000`` steps.

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.

    blocks : :class:`LammpsBlockInput` or :class:`list` (optional)
        Custom block input class. Can be a list of blocks.
        If ``None``, nothing is added in the input. Default ``None``.

    Examples
    --------

    >>> from ase.io import read
    >>> initial = read('A.traj')
    >>>
    >>> from mlacs.state import OptimizeLammpsState
    >>> neb = OptimizeLammpsState(initial, pressure=0, ptype='iso')
    >>> state.run_dynamics(initial, mlip.pair_style, mlip.pair_coeff)
    """

    def __init__(self, min_style="cg", etol=0.0, ftol=1e-6, dt=0.5,
                 pressure=None, ptype="iso", vmax=1e-3,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, blocks=None, **kwargs):
        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         blocks, **kwargs)

        self.min_style = min_style
        self.dt = dt
        self.pressure = pressure
        self.ptype = ptype
        self.vmax = vmax

        self.criterions = (etol, ftol)

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        return EmptyLammpsBlockInput("empty_thermostat")

# ========================================================================== #
    def _get_block_run(self, eq):
        etol, ftol = self.criterions
        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        block = LammpsBlockInput("optimization", "Geometry optimization")

        if self.pressure is not None:
            txt = f"fix box all box/relax {self.ptype} " + \
                  f"{self.pressure*10000} vmax {self.vmax}"
            block("press", txt)
        block("thermo", "thermo 1")
        block("min_style", f"min_style {self.min_style}")
        block("minimize", f"minimize {etol} {ftol} {nsteps} {2*nsteps}")
        return block

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Geometry optimization as implemented in LAMMPS\n"
        if self.pressure is not None:
            msg += f"   target pressure: {self.pressure}\n"
        msg += f"   min_style: {self.min_style}\n"
        msg += f"   energy tolerance: {self.criterions[0]}\n"
        msg += f"   forces tolerance: {self.criterions[1]}\n"
        msg += "\n"
        return msg
