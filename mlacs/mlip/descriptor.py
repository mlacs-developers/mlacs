"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

from abc import ABC, abstractmethod

from ase.neighborlist import neighbor_list

from ..core import Manager
from ..utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class Descriptor(Manager, ABC):
    """
    Base class for descriptors

    Parameters
    ----------
    atoms: :class:`ase.atoms`
        Reference structure, with the elements for the descriptor

    rcut: :class:`float`
        The cutoff for the descriptor
    """
# ========================================================================== #
    def __init__(self, atoms, rcut=5.0, alpha=1.0, prefix='MLIP', **kwargs):

        Manager.__init__(self, prefix=prefix, **kwargs)

        if isinstance(atoms, list):
            self.elements, self.Z, self.masses, self.charges = \
                    [np.array([]) for _ in range(4)]
            for at in atoms:
                el, Z, masses, charges = get_elements_Z_and_masses(at)
                for i in range(len(el)):
                    if el[i] not in self.elements:
                        self.elements = np.append(self.elements, el[i])
                        self.Z = np.append(self.Z, Z[i])
                        self.masses = np.append(self.masses, masses[i])
                        if charges is None:
                            self.charges = np.append(self.charges, 0.)
                        else:
                            self.charges = np.append(self.charges, charges[i])
            if np.allclose(self.charges, 0.0, atol=1e-8):
                self.charges = None
        else:
            self.elements, self.Z, self.masses, self.charges = \
              get_elements_Z_and_masses(atoms)
        self.nel = len(self.elements)
        self.rcut = rcut
        self.welems = np.array(self.Z) / np.sum(self.Z)
        self.alpha = alpha
        self.need_neigh = False

# ========================================================================== #
    def compute_descriptors(self, atoms, forces=True, stress=True):
        desc = []
        for at in atoms:
            # AC : apparently, the at.info for descriptor does not work
            # RB : Tested this, the fix works fined for Trajectory confs but
            #      not for Training. I don't know why ?!
            if "descriptor" in at.info:
                desc.append(at.info['descriptor'])
            else:
                desc.append(self.compute_descriptor(atoms=at,
                                                    forces=forces,
                                                    stress=stress))
        return desc

# ========================================================================== #
    @abstractmethod
    def compute_descriptor(self, atoms, forces=True, stress=True):
        pass

# ========================================================================== #
    def _compute_rij(self, atoms):
        """
        """
        iat, jat, vdist = neighbor_list("ijD", atoms, self.rcut)
        iel = np.array(atoms.get_chemical_symbols())
        iel = np.array([np.where(self.elements == el)[0][0]
                        for el in iel])
        return iat, jat, vdist, iel

# ========================================================================== #
    def _regularization_matrix(self):
        """
        """
        return np.eye(self.ncolumns) * self.alpha


# ========================================================================== #
# ========================================================================== #
class SumDescriptor(Descriptor):
    """
    A class to mix several descriptors together.

    Parameters
    ----------
    args: :class:`list` of :class:`Descriptor`
        A list of all the descriptors to mix.
    """
    def __init__(self, *args):
        # This is wrong for write_mlip. I don't know if there is still
        # a use case for this now that we have DeltaLearningPotential
        raise NotImplementedError("SumDescriptor are not functional")
        self.desc = args
        self.elements = self.desc[0].elements.copy()
        self.rcut = np.max([d.rcut for d in self.desc])
        self.need_neigh = np.any([d.need_neigh for d in self.desc])
        self.ncolumns = np.sum([d.ncolumns for d in self.desc])

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def write_mlip(self, coefficients):
        icol = 0
        for d in self.desc:
            fcol = icol + d.ncolumns
            d.write_mlip(coefficients[icol:fcol])
            icol = fcol

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=True):
        pass

# ========================================================================== #
    def _regularization_matrix(self):
        """
        """
        reg = []
        for d in self.desc:
            reg.append(d._regularization_matrix())
        reg = combine_reg(reg)
        return reg

# ========================================================================== #
    def get_pair_style(self):
        pair_style = "hybrid/overlay "
        for d in self.desc:
            pair_style_d = d.get_pair_style()
            pair_style += f"{pair_style_d} "
        return pair_style

# ========================================================================== #
    def get_pair_coeff(self):
        pair_coeff = []
        for d in self.desc:
            pair_style_d, pair_coeff_d = d.get_pair_style_coeff()
            for coeff in pair_coeff_d:
                style = pair_style_d.split()[0]
                co = coeff.split()
                co.insert(2, style)
                pair_coeff.append(" ".join(co))
        return pair_coeff

# ========================================================================== #
    def get_pair_style_coeff(self):
        return self.get_pair_style(), self.get_pair_coeff()

# ========================================================================== #
    def to_dict(self):
        des = []
        for d in self.desc:
            des.append(d.to_dict())
        dct = dict(name="SumDescriptor",
                   descriptor=des)
        return dct

# ========================================================================== #
    @staticmethod
    def from_dict(dct):
        dct.pop("name", None)
        alldesc = []
        for d in dct["descriptor"]:
            name = d.pop("name")
            import mlacs.mlip as tmp
            descclass = getattr(tmp, name)
            desc = descclass.from_dict(d)
            alldesc.append(desc)
        return SumDescriptor(*alldesc)

# ========================================================================== #
    def __str__(self):
        """
        """
        txt = f"Sum descriptor composed of {len(self.desc)} descriptor"
        return txt

# ========================================================================== #
    def __repr__(self):
        """
        """
        txt = "Sum descriptor\n"
        txt += "--------------\n"
        txt += f"Number of descriptor : {len(self.desc)}\n"
        txt += f"Max rcut :             {self.rcut}\n"
        txt += f"Dimension total :      {self.ncolumns}\n"
        for i, d in enumerate(self.desc):
            txt += f"Descriptors {i+1}:\n"
            txt += repr(d)
            txt += "\n"
        return txt


# ========================================================================== #
# ========================================================================== #
class BlankDescriptor(Descriptor):
    """
    A blank descriptor to serve as Dummy for model that compute the
    descriptor AND do the regression
    """
    def __init__(self, atoms):
        Descriptor.__init__(self, atoms)

# ========================================================================== #
    def compute_descriptor(self, atoms, forces=True, stress=True):
        pass

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def write_mlip(self, mlip_coef):
        pass


# ========================================================================== #
def combine_reg(matrices):
    """
    Combine regularization matrices. Adapted from UF3 code
    available on
    https://github.com/uf3/uf3/blob/master/uf3/regression/regularize.py
    """
    sizes = np.array([len(m) for m in matrices])
    nfeat = int(sizes.sum())
    fullmat = np.zeros((nfeat, nfeat))
    origins = np.insert(np.cumsum(sizes), 0, 0)
    for i, mat in enumerate(matrices):
        start = origins[i]
        end = origins[i + 1]
        fullmat[start:end, start:end] = mat
    return fullmat
