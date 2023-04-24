import datetime
from pathlib import Path

import numpy as np
from ase.atoms import Atoms

from ..utilities import get_elements_Z_and_masses
from .basis import ChebyshevDistance


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
    chemflag: :class:`bool`
        If True, the descriptor is constructed in an explicitely multi-
        element way.
        This increase the number of coefficient to fit, since the number
        of different interaction is given by
        ```
        nint = nel * (nel - 1) / 2 + nel
        ```
        Default False.
    """
# ========================================================================== #
    def __init__(self, atoms, rcut=5.0, chemflag=False):
        self.elements, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(atoms)
        self.nel = len(self.elements)
        self.rcut = rcut
        self.chemflag = chemflag
        self.weights = np.array(self.Z) / np.sum(self.Z)
        self.need_neigh = False
        if self.chemflag:
            self.nint = int(self.nel * (self.nel - 1) / 2) + self.nel
            R, C = np.triu_indices(self.nel)
            self.idx_int = np.zeros((self.nel, self.nel), dtype=int)
            self.idx_int[R, C] = np.arange(self.nint)
            self.idx_int[C, R] = np.arange(self.nint)
        else:
            self.nint = self.nel

# ========================================================================== #
    def _compute_rij(self, atoms):
        """
        """
        dist = atoms.get_all_distances(mic=True)
        vdist = atoms.get_all_distances(mic=True, vector=True)
        iel = np.array(atoms.get_chemical_symbols())
        iel = np.array([np.where(self.elements == el)[0][0]
                        for el in iel])
        return iel, dist, vdist

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
                iel, dist, vdist = self._compute_rij(at)
                res_iat = self._compute_descriptor(at, forces, stress,
                                                   iel, dist, vdist)
            else:
                res_iat = self._compute_descriptor(at, forces, stress)
            res.append(res_iat)
        return res

# ========================================================================== #
    def _regularization_matrix(self):
        """
        """
        return np.eye(self.ncolumns)


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
                iel, dist, vdist = self._compute_rij(at)
            desc_e = np.empty((1, 0))
            desc_f = np.empty((len(at) * 3, 0))
            desc_s = np.empty((6, 0))
            for desc in self.desc:
                if desc.need_neigh:
                    res_iat_d = desc._compute_descriptor(at, forces, stress,
                                                         iel, dist, vdist)
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
class OneBody(Descriptor):
    """
    Simple descriptor counting the number of a given elements in the
    configurations.
    This descriptor allows to fit an intercept.

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor
    """
    def __init__(self, atoms):
        Descriptor.__init__(self, atoms, rcut=0.0)
        self.ncolumns = self.nel

    def _compute_descriptor(self, atoms, forces, stress):
        """
        """
        amat_e = np.zeros((1, self.nel))
        amat_f = np.zeros((len(atoms) * 3, self.nel))
        amat_s = np.zeros((6, self.nel))
        chemsymb = np.array(atoms.get_chemical_symbols())
        for iel in range(self.nel):
            nel = chemsymb == self.elements[iel]
            amat_e[0, iel] = np.asarray(nel, dtype=np.float).sum()
        res = dict(desc_e=amat_e,
                   desc_f=amat_f,
                   desc_s=amat_s)
        return res

# ========================================================================== #
    def _regularization_matrix(self):
        """
        """
        return np.zeros((self.ncolumns, self.ncolumns))

# ========================================================================== #
    def write_mlip(self, coefficients):
        """
        """
        print(coefficients)

# ========================================================================== #
    def __str__(self):
        """
        """
        txt = "One body descriptor"
        return txt

# ========================================================================== #
    def __repr__(self):
        """
        """
        txt = "One body descriptor\n"
        txt += "-------------------\n"
        txt += "Elements :\n"
        txt += " ".join(self.elements) + "\n"
        return txt


# ========================================================================== #
class ChebyPair(Descriptor):
    """
    Pair descriptor constructed on Chebyshev polynomial of the first kind.

    Parameters
    ----------
    atoms : :class:`ase.atoms`
        Reference structure, with the elements for the descriptor
    rcut : :class:`float`
        The cutoff for the descriptor, in anstrom.
        Default 5.0.
    order: :class:`int`
        The order of the polynomial. Gives the number of basis function.
        Default 10
    chemflag: :class:`bool`
        If True, the descriptor is constructed in an explicitely multi-
        element way.
        This increase the number of coefficient to fit, since the number
        of different interaction is given by
        ```
        nint = nel * (nel - 1) / 2 + nel
        ```
        With chemflag as True, the number of component for this descriptor
        is ````nint * order```
        Default True.
    """
    def __init__(self, atoms, rcut=5.0, order=10, chemflag=True,
                 folder=".", fname="PairPot.table"):
        Descriptor.__init__(self, atoms, rcut, chemflag)
        self.need_neigh = True
        self.order = order
        self.dmin = self.rcut
        self.folder = Path(folder).absolute()
        self.fname = fname

        self.basis = []
        for n in range(self.order):
            self.basis.append(ChebyshevDistance(n+1, self.rcut))

        self.ncolumns = self.nint * self.order

# ========================================================================== #
    def _compute_descriptor(self, atoms, forces, stress, iel, dist, vdist):
        """
        """
        nat = len(atoms)

        # The first part consist in creating a flat array with the distances
        # To get everything right, we also need to keep some index
        iat, jat = np.nonzero(np.logical_and(dist <= self.rcut,
                                             dist != 0))
        iat_el = iel[iat]  # index of atom i related to the interaction
        jat_el = iel[jat]  # index of atom j related to the interaction
        dd = dist[iat, jat]  # flat array of distance
        vd = vdist[iat, jat]  # flat array of vector distance
        self.dmin = np.min([self.dmin, dd.min()])  # To keep track

        # now we compute the cutoff, as well as the derivative and so on
        if forces:
            dr = vd / dd[:, None]

        # Let's initialize the matrix where we will keep everything
        amat_e = np.zeros((nat, self.nint, self.order))
        amat_f = np.zeros((nat, 3, self.nint, self.order))
        amat_s = np.zeros((6, self.nint, self.order))

        # We will also need a virial matrix with the stress per atom
        if stress:
            virial = np.zeros((nat, 3, 3, self.nint, self.order))

        # And the basis function
        basis = np.array([b(dd) for b in self.basis]).T
        dbasis = np.array([b.deriv(dd) for b in self.basis]).T

        # some derivatives
        if forces:
            der = dbasis[:, None] * dr[:, :, None]

        # And let's go for the huge mess
        for iint in range(self.nint):
            if self.chemflag:
                iel_int, jel_int = np.nonzero(self.idx_int == iint)
                iel_int = iel_int[0]
                jel_int = jel_int[0]
                # Then, get index for interactions
                idx_el = np.logical_and(iel_int == iat_el,
                                        jel_int == jat_el)
            else:
                idx_el = np.logical_or(iat_el == iint,
                                       jat_el == iint)

            np.add.at(amat_e[:, iint], iat[idx_el],
                      0.5 * basis[idx_el])
            np.add.at(amat_f[:, :, iint], iat[idx_el],
                      0.5 * der[idx_el])
            np.add.at(amat_f[:, :, iint], jat[idx_el],
                      -0.5 * der[idx_el])
            if stress:
                temp = vd[:, :, None, None] * der[:, None]
                np.add.at(virial[:, :, :, iint], iat[idx_el],
                          0.5 * temp[idx_el])

        if stress:
            virial = virial.sum(axis=0)
            amat_s[0] = virial[0, 0]  # xx
            amat_s[1] = virial[1, 1]  # yy
            amat_s[2] = virial[2, 2]  # zz
            amat_s[3] = virial[1, 2]  # yz
            amat_s[4] = virial[0, 2]  # xz
            amat_s[5] = virial[0, 1]  # xy
            if np.all(atoms.get_pbc()):
                amat_s /= atoms.get_volume()

        amat_e = amat_e.sum(axis=0)

        amat_e = amat_e.reshape(1, self.nint * self.order)
        amat_f = amat_f.reshape(len(atoms) * 3, self.nint * self.order)
        amat_s = amat_s.reshape(6, self.nint * self.order)

        res = dict(desc_e=amat_e,
                   desc_f=amat_f,
                   desc_s=amat_s)
        return res

# ========================================================================== #
    def write_mlip(self, coefficients, npoints=5000, folder=".",
                   fname="PairPot.table"):
        """
        """
        folder = Path(folder)
        extra = 0.1 * (self.rcut - self.dmin)
        xmin = np.min([self.dmin - extra, 1.0])
        x = np.linspace(xmin, self.rcut + extra, npoints)
        i = np.arange(1, npoints+1)

        b = np.array([b(x) for b in self.basis])
        db = np.array([b.deriv(x) for b in self.basis])

        with open(folder / fname, "w") as fd:
            date = datetime.datetime.now().strftime('%Y-%m-%d')
            fd.write(f"# DATE: {date} UNITS: metal\n")
            fd.write("# Chebyshev Pair potential, created using MLACS\n")
            fd.write("# Elements : ")
            fd.write(" ".join(self.elements))
            fd.write("\n\n")

            for iint in range(self.nint):
                iel, jel = np.nonzero(self.idx_int == iint)
                el1 = self.elements[iel[0]]
                el2 = self.elements[jel[0]]
                fd.write("".join([el1, el2]) + "\n")
                fd.write(f"N {npoints}\n")
                fd.write("\n")
                iidx = iint * self.order
                fidx = (iint + 1) * self.order
                coefs = coefficients[iidx:fidx]
                if el1 != el2:
                    pref = 0.5
                else:
                    pref = 1.0
                energy = pref * (b * coefs[:, None]).sum(axis=0)
                forces = -pref * (db * coefs[:, None]).sum(axis=0)
                data = np.c_[i, x, energy, forces]
                fmt = "%-6d %20.15f %20.15f %20.15f"
                np.savetxt(fd, data, fmt=fmt)
                fd.write("\n\n")

# ========================================================================== #
    def get_pair_style_coeff(self):
        """
        """
        pair_style = "table linear 5000"
        pair_coeff = []
        tablefile = self.folder / self.fname
        for iint in range(self.nint):
            iel, jel = np.nonzero(self.idx_int == iint)
            iel = iel[0]
            jel = jel[0]
            el1 = self.elements[iel]
            el2 = self.elements[jel]
            entry = "".join([el1, el2])
            tmp_coeff = f"{iel+1}  {jel+1}  {tablefile}  {entry}"
            pair_coeff.append(tmp_coeff)
        return pair_style, pair_coeff

# ========================================================================== #
    def __str__(self):
        """
        """
        txt = " ".join(self.elements)
        txt += " Chebyshev pair descriptor,"
        txt += f"rcut = {self.rcut}, "
        txt += f"order = {self.order}"
        return txt

# ========================================================================== #
    def __repr__(self):
        """
        """
        txt = "Chebyshev Pair descriptor\n"
        txt += "-------------------------\n"
        txt += "Elements :\n"
        txt += " ".join(self.elements) + "\n"
        txt += "Parameters :\n"
        txt += f"rcut                {self.rcut}\n"
        txt += f"order               {self.order}\n"
        txt += f"chemflag            {self.chemflag}\n"
        txt += f"dimension           {self.ncolumns}\n"
        return txt


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
