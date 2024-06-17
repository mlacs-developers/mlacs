from .state import StateManager
from ..core.manager import Manager
from ..mlip.calculator import MlipCalculator

default_parameters = {}


# ========================================================================== #
# ========================================================================== #
class OptimizeAseState(StateManager):
    """
    Class to manage Structure optimization with ASE Optimizers.

    Parameters
    ----------
    model: :class:`LinearPotential` or :class:`DeltaLearningPotential`
        mlacs.mlip linear object.
        Default ``None``

    optimizer: :class:`ase.optimize`
        Optimizer from ase.optimize.
        Default :class:`BFGS`

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``5.0e-2``

    parameters: :class:`dict` (optional)
        Stoping criterion for the optimization run.

    """
    def __init__(self, model=None, optimizer=None, ftol=5.0e-2,
                 parameters={}, **kwargs):

        super().__init__(self, **kwargs)

        self.model = model
        self.opt = optimizer
        self.criterions = ftol
        self.parameters = default_parameters
        self.parameters.update(parameters)
        if optimizer is None:
            from ase.optimize import BFGS
            self.opt = BFGS

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
        atoms = self.run_optimize(atoms)
        return atoms.copy()

# ========================================================================== #
    def run_optimize(self,
                     atoms):
        """
        Run state function.
        """

        if self.model is not None:
            atoms.calc = MlipCalculator(self.model)
        if atoms.calc is None:
            raise TypeError('No Calculator defined !')

        opt = self.opt(atoms)
        opt.run(fmax=self.criterions)

        return atoms.copy()

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Geometry optimization as implemented in LAMMPS\n"
        # RB not implemented yet.
        # if self.pressure is not None:
        #    msg += f"   target pressure: {self.pressure}\n"
        msg += f"   min_style: {self.opt.__name__}\n"
        msg += f"   forces tolerance: {self.criterions}\n"
        msg += "\n"
        return msg
