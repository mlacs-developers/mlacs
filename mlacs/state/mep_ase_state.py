"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ase.io import write

from .state import StateManager
from ..core import PathAtoms
from ..core.manager import Manager
from ..mlip.calculator import MlipCalculator
from ..utilities import get_elements_Z_and_masses

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

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``5.0e-2``

    interpolate: :class:`str`
        Method for position interpolation,
        linear or idpp (Image dependent pair potential).
        Default ``linear``

    parameters: :class:`dict` (optional)
        Parameters for ase.neb.NEB class.

    """
    def __init__(self, images, xi=None, nimages=4, mode=None, model=None,
                 interpolate='linear', parameters={}, print=True, **kwargs):

        super().__init__(**kwargs)

        self.model = model
        self.interpolate = interpolate
        self.parameters = default_parameters
        self.parameters.update(parameters)
        self.print = print
        self.patoms = images
        self.nreplica = nimages
        if not isinstance(self.patoms, PathAtoms):
            self.patoms = PathAtoms(images)
            img = [self.patoms.initial]
            img += [self.patoms.initial.copy() for i in range(self.nreplica)]
            img += [self.patoms.final]
            self.patoms.images = img
        if self.model is None:
            raise TypeError('No Calculator defined !')
        if xi is not None:
            self.patoms.xi = xi
        if mode is not None:
            self.patoms.mode = mode

        self.ispimd = False
        self.isrestart = False

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     elements=None):
        """
        Run state function.
        """
        atoms = supercell.copy()
        el, _, _, _ = get_elements_Z_and_masses(atoms)
        if elements is not None and el != elements:
            # ON : Look at LammpsState run_dynamics to implement
            # It is a question of having an MLIP fitted on different elements
            raise NotImplementedError("Need to implement vartypat here")
        initial_charges = atoms.get_initial_charges()

        images = self.patoms.images

        images = self.run_optimize(images)

        self.patoms.images = images
        atoms = self._get_atoms_results(initial_charges)
        return atoms.copy()

# ========================================================================== #
    def run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """
        pass

# ========================================================================== #
    def _set_calculator(self, images):
        """
        Set Calculator for Forces evaluation.
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
            write(str(self.folder / 'pos_neb_images.xyz'),
                  self.patoms.images, format='extxyz')
            write(str(self.folder / 'pos_neb_splined.xyz'),
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
    def __init__(self, images, xi=None, nimages=4, mode=None,
                 model=None, interpolate='linear',
                 parameters={}, print=False, **kwargs):

        super().__init__(images, xi, nimages, mode, model, interpolate,
                         parameters, print, **kwargs)

# ========================================================================== #
    def run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """

        # RB in future ASE version NEB should be imported from ase.mep
        from ase.neb import NEB
        neb = NEB(images, **self.parameters)

        if self.interpolate == 'idpp':
            neb.interpolate(method='idpp')
        else:
            neb.interpolate()

        images = self._set_calculator(images)

        return images

# ========================================================================== #
    def _set_calculator(self, images):
        """
        Interpolate images and run the optimization.
        """
        for img in images:
            img.calc = MlipCalculator(self.model)
        return images

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "Linear interpolation\n"
        msg += f"Number of replicas:                 {self.patoms.nreplica}\n"
        msg += f"Interpolation method:               {self.interpolate}\n"
        msg += f"Sampling mode:                      {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                 {self.patoms.xi}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class NebAseState(BaseMepState):
    """
    Class to run the Nudged Elastic Band method with ASE Optimizers.
    """
    def __init__(self, images, xi=None, nimages=4, mode=None,
                 model=None, interpolate='linear',
                 Kspring=0.1, optimizer=None, ftol=5.0e-2,
                 parameters={}, print=False, **kwargs):

        super().__init__(images, xi, nimages, mode, model, interpolate,
                         parameters, print, **kwargs)

        self.opt = optimizer
        self.criterions = ftol
        self.Kspring = Kspring
        if self.opt is None:
            from ase.optimize import MDMin
            self.opt = MDMin

# ========================================================================== #
    def run_optimize(self, images):
        """
        Interpolate images and run the optimization.
        """

        images = self._set_calculator(images)

        # RB in future ASE version NEB should be imported from ase.mep
        from ase.neb import NEB
        neb = NEB(images, k=self.Kspring, **self.parameters)

        if self.interpolate == 'idpp':
            neb.interpolate(method='idpp')
        else:
            neb.interpolate()

        opt = self.opt(neb)
        opt.run(fmax=self.criterions, steps=self.nsteps)

        images[0].calc = MlipCalculator(self.model)
        images[-1].calc = MlipCalculator(self.model)

        return images

# ========================================================================== #
    def _set_calculator(self, images):
        """
        Interpolate images and run the optimization.
        """
        for img in images[1:-1]:
            img.calc = MlipCalculator(self.model)
        return images

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "NEB calculation as implemented in ASE\n"
        msg += f"Number of replicas:                 {self.patoms.nreplica}\n"
        msg += f"Interpolation method:               {self.interpolate}\n"
        msg += f"Sampling mode:                      {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                 {self.patoms.xi}\n"
        msg += "\n"
        return msg


# ========================================================================== #
# ========================================================================== #
class CiNebAseState(NebAseState):
    """
    Class to run the Climbing Image Nudged Elastic Band method
    with ASE Optimizers.
    """
    def __init__(self, images, xi=None, nimages=3, mode=None,
                 model=None, interpolate='linear',
                 Kspring=0.1, optimizer=None, ftol=5.0e-2,
                 parameters={}, print=False, **kwargs):

        super().__init__(images, xi, nimages, mode, model, interpolate,
                         Kspring, optimizer, ftol,
                         parameters, print, **kwargs)

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
    def __init__(self, images, xi=None, nimages=4, mode=None,
                 model=None, interpolate='linear',
                 Kspring=0.1, optimizer=None, ftol=5.0e-2,
                 parameters={}, print=False, **kwargs):

        super().__init__(images, xi, nimages, mode, model, interpolate,
                         Kspring, optimizer, ftol,
                         parameters, print, **kwargs)

        self.parameters.update(dict(method='string'))

# ========================================================================== #
    def _set_calculator(self, images):
        """
        Interpolate images and run the optimization.
        """
        for img in images:
            img.calc = MlipCalculator(self.model)
        return images

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "String method calculation as implemented in ASE\n"
        msg += f"Number of replicas:                 {self.patoms.nreplica}\n"
        msg += f"Interpolation method:               {self.interpolate}\n"
        msg += f"Sampling mode:                      {self.patoms.mode}\n"
        msg += f"Sampled coordinate:                 {self.patoms.xi}\n"
        msg += "\n"
        return msg
