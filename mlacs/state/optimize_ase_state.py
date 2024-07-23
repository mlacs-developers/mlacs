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

    constraints: :class:`ase.constraints`
        Constraints to apply to the system during the minimization.
        By default there is no constraints.

    cstr_parameters: :class:`dict`
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

    To perform volume optimization, import the UnitCellFilter constraint

    >>> from ase.constraints import UnitCellFilter
    >>> opt = OptimizeAseState(constraints=UnitCellFilter,
                               cstr_parameters=dict(cell_factor=10))
    >>> opt.run_dynamics(initial, mlip.pair_style, mlip.pair_coeff)
    """
    def __init__(self, optimizer=None, opt_parameters={},
                 constraints=None, cstr_parameters={}, fmax=1e-5,
                 nsteps=1000, nsteps_eq=100, **kwargs):

        super().__init__(nsteps=nsteps, nsteps_eq=nsteps_eq, **kwargs)

        self._opt = optimizer
        self.criterions = fmax
        self._opt_parameters = default_parameters
        self._opt_parameters.update(opt_parameters)
        if optimizer is None:
            from ase.optimize import BFGS
            self._opt = BFGS

        self._cstr = constraints
        self._cstr_params = cstr_parameters

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
                     eq=False):
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
            opt_at = self._cstr(atoms, **self._cstr_params)

        opt = self._opt(opt_at, **self._opt_parameters)
        opt.run(steps=steps, fmax=self.criterions)

        if self._cstr is not None:
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
        if self._cstr is not None:
            if self._cstr.__name__ == "UnitCellFilter":
                if "scalar_pressure" in self._cstr_params.keys():
                    press = self._cstr_params["scalar_pressure"] / GPa
                else:
                    press = 0.0 / GPa
                msg += f"   target pressure: {press} GPa\n"
        # if self.pressure is not None:
        #    msg += f"   target pressure: {self.pressure}\n"
        msg += f"   min_style: {self._opt.__name__}\n"
        msg += f"   forces tolerance: {self.criterions}\n"
        msg += "\n"
        return msg
