import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from ..utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class Descriptor:
    """
    Base class for descriptors

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor

    rcut : :class:`float`
        The cutoff for the descriptor
    """
# ========================================================================== #
    def __init__(self, atoms, rcut=5.0, alpha=1.0):
        self.elements, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(atoms)
        self.nel = len(self.elements)
        self.rcut = rcut
        self.welems = np.array(self.Z) / np.sum(self.Z)
        self.alpha = alpha
        self.need_neigh = False

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
    def calculate(self, atoms, forces=True, stress=True):
        """
        """
        if stress and not forces:
            raise ValueError("You need the forces to compute the stress")

        if isinstance(atoms, Atoms):
            atoms = [atoms]
        res = []
        for at in atoms:
            if self.need_neigh:
                iat, jat, vdist, iel = self._compute_rij(at)
                res_iat = self._compute_descriptor(at, iat, jat, vdist, iel)
            else:
                res_iat = self._compute_descriptor(at, forces, stress)
            res.append(res_iat)
        return res

# ========================================================================== #
    def _regularization_matrix(self):
        """
        """
        return np.eye(self.ncolumns) * self.alpha


# ========================================================================== #
# ========================================================================== #
class SumDescriptor(Descriptor):
    """
    """
    def __init__(self, *args):
        self.desc = args
        self.elements = self.desc[0].elements.copy()
        self.rcut = np.max([d.rcut for d in self.desc])
        self.need_neigh = np.any([d.need_neigh for d in self.desc])
        self.ncolumns = np.sum([d.ncolumns for d in self.desc])

# ========================================================================== #
    def write_mlip(self, coefficients):
        icol = 0
        for d in self.desc:
            fcol = icol + d.ncolumns
            d.write_mlip(coefficients[icol:fcol])
            icol = fcol

# ========================================================================== #
    def calculate(self, atoms, forces=True, stress=True):
        """
        """
        if stress and not forces:
            raise ValueError("You need the forces to compute the stress")

        if isinstance(atoms, Atoms):
            atoms = [atoms]
        res = []
        for at in atoms:
            if self.need_neigh:
                iat, jat, vdist, iel = self._compute_rij(at)
            desc_e = np.empty((1, 0))
            desc_f = np.empty((len(at) * 3, 0))
            desc_s = np.empty((6, 0))
            for desc in self.desc:
                if desc.need_neigh:
                    res_iat_d = desc._compute_descriptor(at, iat, jat,
                                                         vdist, iel)
                else:
                    res_iat_d = desc._compute_descriptor(at, forces, stress)
                desc_e = np.c_[desc_e, res_iat_d["desc_e"]]
                desc_f = np.c_[desc_f, res_iat_d["desc_f"]]
                desc_s = np.c_[desc_s, res_iat_d["desc_s"]]
            res.append(dict(desc_e=desc_e,
                            desc_f=desc_f,
                            desc_s=desc_s))
        return res

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
    def get_pair_style_coeff(self):
        pair_style = "hybrid/overlay "
        pair_coeff = []
        for d in self.desc:
            pair_style_d, pair_coeff_d = d.get_pair_style_coeff()
            pair_style += f"{pair_style_d} "
            for coeff in pair_coeff_d:
                style = pair_style_d.split()[0]
                co = coeff.split()
                co.insert(2, style)
                pair_coeff.append(" ".join(co))
        return pair_style, pair_coeff

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