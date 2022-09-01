"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np
from ase.atoms import Atoms


# ========================================================================== #
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
    Z = supercell.get_atomic_numbers()
    masses = supercell.get_masses()
    charges = supercell.get_initial_charges()

    un_elements = sorted(set(elements))
    un_Z = []
    un_masses = []
    un_charges = []
    for iel in range(len(un_elements)):
        idx = elements.index(un_elements[iel])
        un_Z.append(Z[idx])
        un_masses.append(masses[idx])
        un_charges.append(charges[idx])

    if np.allclose(un_charges, 0.0, atol=1e-8):
        un_charges = None
    return un_elements, un_Z, un_masses, un_charges


# ========================================================================== #
def create_random_structures(atoms, std, nconfs):
    """
    Create n random structures by displacing atoms around position

    Parameters
    ----------
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled
    std: :class:`float`
        Standard deviation of the gaussian used to generate
        the random displacements. In angstrom.
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
            iatoms.rattle(stdev=std, rng=rng)
            confs.append(iatoms)
    return confs


# ========================================================================== #
def compute_correlation(data):
    """
    Function to compute the RMSE and MAE

    Parameters
    ----------

    data: :class:`numpy.ndarray` of shape (ndata, 2)
        The data for which to compute the correlation.
        The first column should be the gound truth and the second column
        should be the prediction of the model
    datatype: :class:`str`
        The type of data to which the correlation are to be computed.
        Can be either energy, forces or stress
    """
    datatrue = data[:, 0]
    datatest = data[:, 1]
    rmse = np.sqrt(np.mean((datatrue - datatest)**2))
    mae = np.mean(np.abs(datatrue - datatest))
    sse = ((datatrue - datatest)**2).sum()
    sst = ((datatrue - datatrue.mean())**2).sum()
    rsquared = 1 - sse / sst
    return rmse, mae, rsquared
