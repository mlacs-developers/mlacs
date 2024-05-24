from ase.io import write

from .state import StateManager
from ..core import PathAtoms
from ..mlip.calculator import MlipCalculator

default_parameters = {}


# ========================================================================== #
# ========================================================================== #
class BaseMepState(StateManager):
    """
    Class to manage Minimum Energy Path sampling with ASE Optimizers.

    Parameters
    ----------
    images: :class:`list` or `PathAtoms`
        mlacs.PathAtoms or list of ase.Atoms object.
        The list contain initial and final configurations of the reaction path.

    xi: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        Default ``None``

    nimages : :class:`int` (optional)
        Number of images used along the reaction coordinate. Default ``1``.
        which is suposed the saddle point.

    mode: :class:`float` or :class:`string`
        Value of the reaction coordinate or sampling mode:
        - ``float`` sampling at a precise coordinate.
        - ``rdm_true`` randomly return the coordinate of an images.
        - ``rdm_spl`` randomly return the coordinate of a splined images.
        - ``rdm_memory`` homogeneously sample the splined reaction coordinate.
        - ``None`` return the saddle point.
        Default ``saddle``

    model: :class:`LinearPotential` or :class:`DeltaLearningPotential`
        mlacs.mlip linear object.
        Default ``None``

    optimizer: :class:`ase.optimize`
        Optimizer from ase.optimize.
        Default :class:`BFGS`

    etol: :class:`float`
        Stopping tolerance for energy
        Default ``0.0``

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``1.0e-3``

    interpolate: :class:`str`
        Method for position interpolation,
        linear or idpp (Image dependent pair potential).
        Default ``linear``

    parameters: :class:`dict` (optional)
        Parameters for ase.neb.NEB class.

    """
    def __init__(self, images, xi=None, nimages=4, Kspring=0.1, mode=None,
                 model=None, optimizer=None, etol=0.0, ftol=1.0e-3,
                 interpolate='linear', parameters={}, **kwargs):

        super().__init__(self, **kwargs)

        self.model = model
        self.interpolate = interpolate
        if self.model is None:
            raise TypeError('No Calculator defined !')
        self.opt = optimizer
        self.criterions = (etol, ftol)
        self.parameters = default_parameters
        self.parameters.update(parameters)
        self.patoms = images
        self.nreplica = nimages
        self.Kspring = Kspring
        if not isinstance(self.patoms, PathAtoms):
            img = [self.patoms.initial]
            img += [self.patoms.initial.copy() for i in range(self.nreplica)]
            img += [self.patoms.final]
            self.patoms = PathAtoms(img)
        if xi is not None:
            self.patoms.xi = xi
        if mode is not None:
            self.patoms.mode = mode
        if optimizer is None:
            from ase.optimize import BFGS
            self.opt = BFGS

# ========================================================================== #
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
        initial_charges = atoms.get_initial_charges()

        images = self.patoms.images

        images = self._run_optimize(images)

        self.patoms.images = images
        atoms = self._get_atoms_results(initial_charges)
        return atoms.copy()

# ========================================================================== #
    def _run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """
        pass

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """
        """
        self.patoms.update
        atoms = self.patoms.splined
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        if self.print:
            write(str(self.subsubdir / 'pos_neb_images.xyz'),
                  self.patoms.images, format='extxyz')
            write(str(self.subsubdir / 'pos_neb_splined.xyz'),
                  self.patoms.splined, format='extxyz')
        return atoms

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        pass


# ========================================================================== #
# ========================================================================== #
class LinearInterpolation(BaseMepState):
    """
    Class to do a simple Linear interpolation of positions with ASE.
    Can be used with the Image dependent pair potential method.
    """
    def __init__(self, images, xi=None, nimages=4, Kspring=0.1, mode='saddle',
                 model=None, optimizer=None, etol=0.0, ftol=1.0e-3,
                 interpolate=None, parameters={}, **kwargs):

        super().__init__(self, images, xi, nimages, mode, model, optimizer,
                         etol, ftol, parameters, **kwargs)

# ========================================================================== #
    def _run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """

        from ase.mep import NEB
        neb = NEB(images, **self.parameters)

        if self.interpolate == 'idpp':
            neb.interpolate(method='idpp')
        else:
            neb.interpolate()

        for img in images:
            img.calc = MlipCalculator(self.model)

        return images

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Linear interpolation\n"
        msg += f"Number of replicas:                     {self.nreplica}\n"
        msg += f"Interpolation method:                   {self.interpolate}\n"
        msg += f"Sampling mode:                          {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                     {self.patoms.xi}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class NebAseState(BaseMepState):
    """
    Class to run the Nudged Elastic Band method with ASE Optimizers.
    """
    def __init__(self, images, xi=None, nimages=4, Kspring=0.1, mode='saddle',
                 model=None, optimizer=None, etol=0.0, ftol=1.0e-3,
                 interpolate=None, parameters={}, **kwargs):

        super().__init__(self, images, xi, nimages, mode, model, optimizer,
                         etol, ftol, parameters, **kwargs)

        if self.optimizer is None:
            from ase.optimize import BFGS
            self.optimizer = BFGS

# ========================================================================== #
    def _run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """

        for img in images[1:-2]:
            img.calc = MlipCalculator(self.model)

        from ase.mep import NEB
        neb = NEB(images, **self.parameters)

        if self.interpolate == 'idpp':
            neb.interpolate(method='idpp')
        else:
            neb.interpolate()

        self.opt(neb)
        self.opt.run()

        images[0].calc = MlipCalculator(self.model)
        images[-1].calc = MlipCalculator(self.model)

        return images

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "NEB calculation as implemented in ASE\n"
        msg += f"Number of replicas:                     {self.nreplica}\n"
        msg += f"Interpolation method:                   {self.interpolate}\n"
        msg += f"Sampling mode:                          {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                     {self.patoms.xi}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class CiNebAseState(NebAseState):
    """
    Class to run the Climbing Image Nudged Elastic Band method
    with ASE Optimizers.
    """
    def __init__(self, images, xi=None, nimages=4, Kspring=0.1, mode='saddle',
                 model=None, optimizer=None, etol=0.0, ftol=1.0e-3,
                 interpolate=None, parameters={}, **kwargs):

        super().__init__(self, images, xi, nimages, mode, model, optimizer,
                         etol, ftol, parameters, **kwargs)

        self.parameters.update(dict(climb=True))

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Ci-NEB calculation as implemented in ASE\n"
        msg += f"Number of replicas:                     {self.nreplica}\n"
        msg += f"Interpolation method:                   {self.interpolate}\n"
        msg += f"Sampling mode:                          {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                     {self.patoms.xi}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class StringMethodAseState(NebAseState):
    """
    Class to run the String Method with ASE Optimizers.
    """
    def __init__(self, images, xi=None, nimages=4, Kspring=0.1, mode=None,
                 model=None, optimizer=None, etol=0.0, ftol=1.0e-3,
                 interpolate=None, parameters={}, **kwargs):

        super().__init__(self, images, xi, nimages, mode, model, optimizer,
                         etol, ftol, parameters, **kwargs)

        self.parameters.update(dict(method='string'))

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "String method calculation as implemented in ASE\n"
        msg += f"Number of replicas:                     {self.nreplica}\n"
        msg += f"Interpolation method:                   {self.interpolate}\n"
        msg += f"Sampling mode:                          {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                     {self.patoms.xi}\n"
        msg += "\n"
        return msg
