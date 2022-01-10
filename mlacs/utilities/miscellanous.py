"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ase.atoms import Atoms
from ase.io import read
from ase.units import Hartree, Bohr


#========================================================================================================================#
def get_elements_Z_and_masses(supercell):
    '''
    Get the unique chemical symbols and atomic numbers of a supercell.
    The list are returned according to the alphabetical order of the elements.

    Parameters
    ----------
    supercell: :class:`ase.Atoms`
        ASE atoms object

    Return
    ------
    elements: :class:`list` of :class:`str`
        list of unique elements in the supercell
    Z: :class:`list` of :class:`int`
        list of unique Z in the supercell
    masses: :class:`list` of :class:`float`
        list of unique masses in the supercell
    '''
    elements = supercell.get_chemical_symbols()
    Z        = supercell.get_atomic_numbers()
    masses   = supercell.get_masses()

    un_elements = sorted(set(elements))
    un_Z        = []
    un_masses   = []
    for iel in range(len(un_elements)):
        idx = elements.index(un_elements[iel])
        un_Z.append(Z[idx])
        un_masses.append(masses[idx])
    return un_elements, un_Z, un_masses


#========================================================================================================================#
def create_random_structures(atoms, std, nconfs):
    """
    Create n random structures by displacing atoms around position

    Parameters
    ----------
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled
    std: :class:`float`
        Standard deviation of the gaussian used to generate the random displacements. In angstrom.
    nconfs: :class:`int`
        Number of configurations to generate

    Return
    ------
    confs: :class:`list` of :class:`ase.Atoms`
        Configurations with random displacements
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    rng = np.random.default_rng()
    confs = []
    for iat, at in enumerate(atoms):
        for i in range(nconfs):
            iatoms = at.copy()
            iatoms.rattle(stdev=std)
            confs.append(iatoms)
    return confs
