'''
'''
import numpy as np

from . import MlipManager
from ..utilities import compute_correlation, create_link

from ase import Atoms
from pathlib import Path


# ========================================================================== #
# ========================================================================== #
class TensorpotPotential(MlipManager):
    """
    Potential that use Tensorpotential to minimize a cost function.
    For now, only works with AceDescriptor

    Parameters
    ----------
    descriptor: :class:`Descriptor`
        The descriptor used in the model.
    weight: :class:`WeightingPolicy`
        Weight used for the fitting and calculation of properties.
        Default :class:`None`
    """
    def __init__(self,
                 descriptor,
                 folder="Tensorpot",
                 weight=None):
        MlipManager.__init__(self,
                             descriptor=descriptor,
                             folder=folder,
                             weight=weight)
        self.natoms = []
        self.nconfs = 0
        db_fn = Path(self.folder / self.descriptor.db_fn).absolute()
        self.descriptor.db_fn = db_fn
        ps, pc = self.descriptor.get_pair_style_coeff(self.folder)
        self.pair_style = ps
        self.pair_coeff = pc

        if self.weight.stress_coefficient != 0:
            raise ValueError("Tensorpotential can't fit on stress")

        self.new_ymat_e = []
        self.new_ymat_f = []
        self.new_atomic_env = []
        self.atoms = []
        self.name = []
        self.coef = []

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.weight.update_database(atoms)

        for at in atoms:
            free_e = self.descriptor.calc_free_e(at)
            self.new_ymat_e.append(at.get_potential_energy() - free_e)
            self.new_ymat_f.append(at.get_forces())

        for i in range(self.nconfs, self.nconfs+len(atoms)):
            self.name.append(f"config{i}")
        self.nconfs += len(atoms)
        self.natoms.extend([len(_) for _ in atoms])
        self.atoms.extend(atoms)

# ========================================================================== #
    def train_mlip(self, mlip_subfolder):
        """
        """
        if mlip_subfolder is None:
            mlip_subfolder = self.folder
        else:
            mlip_subfolder = self.folder / mlip_subfolder

        W = self.weight.get_weights()
        md_fn, coef_fn = self.descriptor.fit(
            weights=W, atoms=self.atoms, name=self.name, natoms=self.natoms,
            energy=self.new_ymat_e, forces=self.new_ymat_f, 
            subfolder=mlip_subfolder)
        self.coef.append(mlip_subfolder/coef_fn)

        msg = "Number of configurations for training: " + \
              f"{self.nconfs}\n"
        msg += "Number of atomic environments for training: " + \
               f"{sum(self.natoms)}\n"

        f_mlipE = self.get_mlip_energy  # (desc, coef)
        tmp_msg, weight_fn = self.weight.compute_weight(
           desc=self.atoms,
           coef=coef_fn,
           f_mlipE=f_mlipE,
           subfolder=mlip_subfolder)

        create_link(mlip_subfolder/weight_fn, self.folder/weight_fn)
        create_link(mlip_subfolder/md_fn, self.folder/md_fn)
        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        from .calculator import MlipCalculator
        calc = MlipCalculator(self)
        return calc

# ========================================================================== #
    def predict(self, atoms):
        """
        Give the energy forces stress of atoms according to the potential.
        """
        return self.get_mlip_energy(atoms)

# ========================================================================== #
    def get_mlip_energy(self, atoms, coef=None):
        """
        Calculate the energy predicted by the potential on a configuration

        Parameters
        ----------
        coef: :class:`pathlib.Path` or :class:`str`
              Path to the potential file

        atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
              Configuration to evaluate
        """
        if coef is None:
            coef = self.coef[-1]
        if isinstance(coef, str):
            coef = Path(coef)

        return self.descriptor.get_mlip_energy(atoms, coef)

# ========================================================================== #
    def __str__(self):
        dname = type(self.descriptor).__name__
        txt = f"Tensorpotential potential with {dname}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Tensorpotential\n"
        txt += "Parameters:\n"
        txt += "-----------\n"
        txt += f"Weight : {type(self.weight).__name__}"
        txt += "\n"
        txt += "Descriptor used in the potential:\n"
        txt += repr(self.descriptor)
        return txt
