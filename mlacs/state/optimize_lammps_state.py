"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from .lammps_state import LammpsState
from ..utilities.io_lammps import (LammpsBlockInput,
                                   EmptyLammpsBlockInput)


# ========================================================================== #
# ========================================================================== #
class OptimizeLammpsState(LammpsState):
    """
    Class to manage geometry optimizations with LAMMPS.

    Parameters
    ----------
    min_style: :class:`str`
        Choose a minimization algorithm to use when a minimize command is
        performed.
        Default `cg`.

    etol: :class:`float`
        Stopping tolerance for energy
        Default ``0.0``

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``1.0e-6``

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

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``0.5`` fs.

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

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created

    Examples
    --------

    >>> from ase.io import read
    >>> initial = read('A.traj')
    >>>
    >>> from mlacs.state import OptimizeLammpsState
    >>> neb = OptimizeLammpsState(initial, pressure=0, ptype='iso')
    >>> state.run_dynamics(initial, mlip.pair_style, mlip.pair_coeff)
    """
    def __init__(self,
                 min_style='cg',
                 etol=0.0,
                 ftol=1.0e-6,
                 dt=0.5,
                 pressure=None,
                 ptype="iso",
                 vmax=1.0e-3,
                 nsteps=10000,
                 nsteps_eq=1000,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 workdir=None):
        LammpsState.__init__(self,
                             temperature=0.0,
                             pressure=pressure,
                             ptype="iso",
                             dt=dt,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             logfile=logfile,
                             trajfile=trajfile,
                             loginterval=loginterval,
                             workdir=workdir)

        self.style = min_style
        self.criterions = (etol, ftol)
        self.vmax = vmax
        self.langevin = False

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
        block("min_style", f"min_style {self.style}")
        block("minimize", f"minimize {etol} {ftol} {nsteps} {2*nsteps}")
        return block
