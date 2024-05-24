from ..core.manager import Manager
from ..mlip.calculator import MlipCalculator

default_parameters = {}


# ========================================================================== #
# ========================================================================== #
class OptimizeAseState(Manager):
    """
    Class to manage Structure optimization with ASE Optimizers.
    (not in production)

    Parameters
    ----------
    model: :class:`LinearPotential` or :class:`DeltaLearningPotential`
        mlacs.mlip linear object.
        Default ``None``

    optimizer: :class:`ase.optimize`
        Optimizer from ase.optimize.
        Default :class:`BFGS`

    parameters: :class:`dict` (optional)
        Stoping criterion for the optimization run.

    """
    def __init__(self,
                 model=None,
                 optimizer=None,
                 parameters={},
                 **kwargs):

        Manager.__init__(self, **kwargs)

        self.model = model
        self.opt = optimizer
        self.parameters = default_parameters
        self.parameters.update(parameters)
        if optimizer is None:
            from ase.optimize import BFGS
            self.opt = BFGS

# ========================================================================== #
    def run_optimize(self,
                     atoms):
        """
        Run state function.
        """

        if self.model is not None:
            atoms.calc = MlipCalculator(self.model)
        if self.atoms.calc is None:
            raise TypeError('No Calculator defined !')

        self.opt(atoms)
        self.opt.run()
        return atoms.copy()

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "NEB calculation as implemented in LAMMPS\n"
        msg += f"Number of replicas :                     {self.nreplica}\n"
        msg += f"String constant :                        {self.Kspring}\n"
        msg += f"Sampling mode :                          {self.mode}\n"
        msg += "\n"
        return msg
