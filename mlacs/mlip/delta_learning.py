"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ase.atoms import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator

from .mlip_manager import MlipManager


# ========================================================================== #
# ========================================================================== #
class DeltaLearningPotential(MlipManager):
    """
    Parameters
    ----------
    model: :class:`MlipManager`
        The MLIP model to train on the difference between the true energy
        and the energy of a LAMMPS reference model.

    pair_style: :class:`str` or :class:`list` of :class:`str`
        The pair_style of the LAMMPS reference potential.
        If only one pair style is used, can be set as a :class:`str`.
        If an overlay of pair style is used, this input as to be a
        :class:`list` of :class:`str` of the pair_style.
        For example :

    pair_coeff: :class:`list` of :class:`str`
        The pair_coeff of the LAMMPS reference potential.

    folder: :class:`str` (optional)
        The root for the directory in which the MLIP are stored.
        Default 'MLIP'

    Examples
    --------

    >>> from ase.io import read
    >>> confs = read('Trajectory.traj', index=':')
    >>>
    >>> from mlacs.mlip import SnapDescriptor, LinearPotential
    >>> desc = SnapDescriptor(confs[0], rcut=6.2, parameters=dict(twojmax=6))
    >>> mlip = LinearPotential(desc)
    >>>
    >>> from mlacs.mlip import DeltaLearningPotential
    >>> ps = ['sw', 'zbl 3.0 4.0']
    >>> pc = [['* * Si.sw Si'], ['* * 14 14']]
    >>> dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=pc)
    >>> dmlip.update_matrices(confs)
    >>> dmlip.train_mlip()
    """
    def __init__(self,
                 model,
                 pair_style,
                 pair_coeff,
                 model_post=None,
                 atom_style="atomic",
                 folder=None,
                 **kwargs):

        if folder != model.folder:
            if folder is not None:
                model.folder = folder
            else:
                folder = model.folder

        self.model = model
        weight = self.model.weight

        if not isinstance(pair_style, list):
            pair_style = [pair_style]

        self.ref_pair_style = pair_style
        self.ref_pair_coeff = pair_coeff

        self.ref_model_post = model_post
        self.model_post = model_post
        self.ref_atom_style = atom_style
        self.atom_style = atom_style

        MlipManager.__init__(self, self.model.descriptor, weight,
                             folder=folder, **kwargs)

        self._ref_e = None
        self._ref_f = None
        self._ref_s = None

# ========================================================================== #

    def _set_directories(self):
        self.model.workdir = self.workdir
        self.model.folder = self.folder
        self.model.subfolder = self.subfolder
        self.model.descriptor.workdir = self.workdir
        self.model.descriptor.folder = self.folder
        self.model.descriptor.subfolder = self.subfolder
        self.model.weight.workdir = self.workdir
        self.model.weight.folder = self.folder
        self.model.weight.subfolder = self.subfolder

# ========================================================================== #
    def get_ref_pair_style(self, lmp=False):
        """
        Return self.ref_pair_style which is an array.
        If lmp=True, it returns it formatted as a lammps input.
        """
        self._set_directories()
        if not lmp:
            return self.ref_pair_style

        if len(self.ref_pair_style) == 1:
            return self.ref_pair_style[0]
        else:  # Here the tricky part. I need to create hybrid overlay ...
            full_pair_style = "hybrid/overlay "
            for ps in self.ref_pair_style:
                full_pair_style += f"{ps} "
            return full_pair_style

# ========================================================================== #
    def get_ref_pair_coeff(self):
        """
        Return the pair_coeff for the reference calculations
        """
        self._set_directories()
        if len(self.ref_pair_style) == 1:
            return self.ref_pair_coeff
        else:
            ref_pair_coeff = []
            for ps, pc in zip(self.ref_pair_style, self.ref_pair_coeff):
                refpssplit = ps.split()
                for ppc in pc:
                    refpcsplit = ppc.split()
                    refpc = " ".join([*refpcsplit[:2],
                                      refpssplit[0],
                                      *refpcsplit[2:]])
                    ref_pair_coeff.append(refpc)
            return ref_pair_coeff

# ========================================================================== #
    def _get_pair_style(self):
        self._set_directories()
        # We need to create the hybrid/overlay format of LAMMPS
        if not isinstance(self.ref_pair_style, list):
            self.ref_pair_style = [self.ref_pair_style]

        if len(self.ref_pair_style) == 1:
            full_pair_style = f"hybrid/overlay {self.ref_pair_style[0]} " + \
                              f"{self.model.pair_style}"
        else:
            full_pair_style = "hybrid/overlay "
            for ps in self.ref_pair_style:
                full_pair_style += f"{ps} "
            full_pair_style += f"{self.model.pair_style}"

        return full_pair_style

# ========================================================================== #
    def _get_pair_coeff(self):
        self._set_directories()
        if not isinstance(self.ref_pair_style, list):
            self.ref_pair_style = [self.ref_pair_style]

        # First let's take care of only one reference potential
        if len(self.ref_pair_style) == 1:
            refpssplit = self.ref_pair_style[0].split()
            full_pair_coeff = []
            for refpc in self.ref_pair_coeff:
                refpcsplit = refpc.split()
                full_pair_coeff.append(" ".join([*refpcsplit[:2],
                                       refpssplit[0],
                                       *refpcsplit[2:]]))
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            full_pair_coeff.append(mlpc)

        # And now with an overlay reference potential
        else:
            full_pair_coeff = []
            for ps, pc in zip(self.ref_pair_style, self.ref_pair_coeff):
                refpssplit = ps.split()
                for ppc in pc:
                    refpcsplit = ppc.split()
                    refpc = " ".join([*refpcsplit[:2],
                                      refpssplit[0],
                                      *refpcsplit[2:]])
                    full_pair_coeff.append(refpc)
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            full_pair_coeff.append(mlpc)
        return full_pair_coeff

# ========================================================================== #
    def predict(self, atoms, coef=None):
        """
        Function that gives the mlip_energy
        """
        return self.model.predict(atoms, coef)

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """

        # First compute reference energy/forces/stress
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        calc = LAMMPS(pair_style=self.get_ref_pair_style(lmp=True),
                      pair_coeff=self.get_ref_pair_coeff(),
                      atom_style=self.ref_atom_style)

        if self.model_post is not None:
            calc.set(model_post=self.ref_model_post)

        dummy_at = []
        for at in atoms:
            at0 = at.copy()
            at0.calc = calc
            refe = at0.get_potential_energy()
            reff = at0.get_forces()
            refs = at0.get_stress()

            dumdum = at.copy()

            e = at.get_potential_energy() - refe
            f = at.get_forces() - reff
            s = at.get_stress() - refs
            spcalc = SinglePointCalculator(dumdum,
                                           energy=e,
                                           forces=f,
                                           stress=s)
            dumdum.calc = spcalc
            dummy_at.append(dumdum)

        # Now get descriptor features
        self.model.update_matrices(dummy_at)
        self.nconfs = self.model.nconfs

# ========================================================================== #
    def next_coefs(self, mlip_coef, *args, **kwargs):
        """
        """
        msg = self.model.next_coefs(mlip_coef, *args, **kwargs)
        return msg

# ========================================================================== #
    def train_mlip(self):
        """
        """
        self._set_directories()
        self.model.train_mlip()

    # GA: Need to overwrite this abstract methods, but I'm not sure
    #     if it is used at all.
    def get_mlip_energy(coef, desc):
        """
        Function that gives the mlip_energy
        """
        raise NotImplementedError

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff,
                      atom_style=self.atom_style,
                      keep_alive=False)
        if self.model_post is not None:
            calc.set(model_post=self.model_post)
        return calc

# ========================================================================== #
    def __str__(self):
        txt = "Delta Learning potential\n"
        txt += f"Reference pair_style: {self.ref_pair_style}\n"
        txt += f"Reference pair_coeff: {self.ref_pair_coeff}\n"
        txt += str(self.model)
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Delta learning potential\n"
        txt += "------------------------\n"
        txt += "Reference potential :\n"
        txt += f"pair_style {self.ref_pair_style}\n"
        for pc in self.ref_pair_coeff:
            txt += f"pair_coeff {pc}\n\n"
        txt += "MLIP potential :\n"
        txt += repr(self.model)
        return txt
