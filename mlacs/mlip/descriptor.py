import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from ..utilities import get_elements_Z_and_masses

class Descriptor:
    """
    """
    def __init__(self, atoms, rcut=5.0, chemflag=False):
        self.elements, self.Z, self.masses, self.charges = \
            get_elements_Z_and_masses(atoms)
        self.nel = len(self.elements)
        self.rcut = rcut
        self.chemflag = chemflag
        if self.chemflag:
            self.nint = int(nelements * (nelements - 1) / 2) + nelements
            R, C = np.triu_indices(nelements)
            self.idx_int = np.zeros((nelements, nelements), dtype=int)
            self.idx_int[R, C] = np.arange(nint)
            self.idx_int[C, R] = np.arange(nint)
        else:
            self.nint = self.nel
            self.weights = np.zeros(self.nel)
            for i, zel in enumerate(self.Z):
                self.weights[i] = zel / (self.Z.sum())

    def calculate(self, atoms, forces=True, stress=True):
        """
        """
        
        return self._compute_descriptor(atoms, forces, stress)


class OneBody(Descriptor):
    def __init__(self, atoms, rcut=5.0):
        Descriptor.__init__(self, atoms, rcut)

    def _compute_descriptor(self, atoms, forces=True, stress=True):
        amat_e = np.zeros((1, self.nel))
        amat_f = np.zeros((len(atoms) * 3, self.nel))
        amat_s = np.zeros((6, self.nel))
        chemsymb = np.array(atoms.get_chemical_symbols())
        for iel in range(self.nel):
            nel = np.nonzero(chemsymb == self.elements[iel])
            amat_e[iel] = nel
        res = dict(desc_e=amat_e,
                   desc_f=amat_f,
                   desc_s=amat_s)
        return res


class ChebyPair(Descriptor):
    """
    """
    def __init__(self, atoms, rcut, order=10):
        Descriptor.__init__(self, atoms, rcut)
        self.order = 10
        
    def _compute_descriptor(self, atoms, forces=True, stress=True):
