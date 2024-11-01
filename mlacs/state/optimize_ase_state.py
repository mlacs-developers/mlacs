"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ase.calculators.lammpsrun import LAMMPS
from ase.units import GPa

from .state import StateManager
from ..core.manager import Manager

default_parameters = {}


# ========================================================================== #
# ========================================================================== #
class OptimizeAseState(StateManager):
    """
    Class to manage Structure optimization with ASE Optimizers.

    Parameters
    ----------
    optimizer: :class:`ase.optimize`
        Optimizer from ase.optimize.
        Default :class:`BFGS`

    opt_parameters: :class:`dict`
        Dictionnary with the parameters for the Optimizer.
        Default: {}

    constraints: :class:`ase.constraints` or :class:`list`
        Constraints to apply to the system during the minimization or list
        of constraints.
        By default there is no constraints.

    cstr_parameters: :class:`dict` or :class:`list`
        Dictionnary with the parameter for the constraints or list
        of constraints parameters.
        Default: {}

    filters: :class:`ase.filters`
        Filters are constraints to apply on the cell during the minimization.
        By default there is no filters.

    fltr_parameters: :class:`dict`
        Dictionnary with the parameter for the constraints.
        Default: {}

    fmax: :class:`float`
        The maximum value for the forces to be considered converged.
        Default: 1e-5

    Examples
    --------

    >>> from ase.io import read
    >>> initial = read('A.traj')
    >>>
    >>> from mlacs.state import OptimizeAseState
    >>> opt = OptimizeAseState()
    >>> opt.run_dynamics(initial, mlip.pair_style, mlip.pair_coeff)

    To perform volume optimization, import the UnitCellFilter filter

    >>> from ase.filters import UnitCellFilter
    >>> opt = OptimizeAseState(filters=UnitCellFilter,
                               fltr_parameters=dict(cell_factor=10))
    >>> opt.run_dynamics(initial, mlip.pair_style, mlip.pair_coeff)
    """
    def __init__(self, optimizer=None, opt_parameters={},
                 constraints=None, cstr_parameters={},
                 filters=None, fltr_parameters={},
                 fmax=1e-5, nsteps=1000, nsteps_eq=100, **kwargs):

        super().__init__(nsteps=nsteps, nsteps_eq=nsteps_eq, **kwargs)

        self._opt = optimizer
        self.criterions = fmax
        self._opt_parameters = default_parameters
        self._opt_parameters.update(opt_parameters)
        if optimizer is None:
            from ase.optimize import BFGS
            self._opt = BFGS

        # RB constraints on Atoms.
        self._cstr = constraints
        self._cstr_params = cstr_parameters

        # RB constraints on Cell.
        self._fltr = filters
        self._fltr_params = fltr_parameters

        self.ispimd = False
        self.isrestart = False

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post,
                     atom_style="atomic",
                     eq=False,
                     elements=None):
        """
        Run state function.
        """
        atoms = supercell.copy()
        calc = LAMMPS(pair_style=pair_style, pair_coeff=pair_coeff,
                      atom_style=atom_style)
        if model_post is not None:
            calc.set(model_post=model_post)
        atoms.calc = calc
        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        atoms = self.run_optimize(atoms, nsteps)
        return atoms.copy()

# ========================================================================== #
    def run_optimize(self, atoms, steps):
        """
        Run state function.
        """

        opt_at = atoms
        if self._cstr is not None:
            if isinstance(self._cstr, list):
                cstr = [c(**p) for c, p in zip(self._cstr, self._cstr_params)]
                opt_at.set_constraint(cstr)
            else:
                opt_at.set_constraint(self._cstr(**self._cstr_params))
        if self._fltr is not None:
            opt_at = self._fltr(atoms, **self._fltr_params)

        opt = self._opt(opt_at, **self._opt_parameters)
        opt.run(steps=steps, fmax=self.criterions)

        if self._fltr is not None:
            atoms = opt.atoms.atoms
        else:
            atoms = opt.atoms

        return atoms.copy()

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Geometry optimization as implemented in ASE\n"
        # RB not implemented yet.
        # AC now it's implemented, but not easily accessible
        if self._fltr is not None:
            if self._fltr.__name__ == "UnitCellFilter":
                if "scalar_pressure" in self._fltr_params.keys():
                    press = self._fltr_params["scalar_pressure"] / GPa
                else:
                    press = 0.0 / GPa
                msg += f"   target pressure: {press} GPa\n"
        # if self.pressure is not None:
        #    msg += f"   target pressure: {self.pressure}\n"
        msg += f"   min_style: {self._opt.__name__}\n"
        msg += f"   forces tolerance: {self.criterions}\n"
        msg += "\n"
        return msg
