"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from ase.io import read
from ase.units import Hartree, Bohr


#========================================================================================================================#
def extract_data_for_tdep(fname, nthrow=10):
    """
    Little function to extract ASCII data to be used by aTDEP
    The forces and energies are converted to atomic units
    """
    trajectory = read(fname, index=":")
    nconfs     = len(trajectory)

    xred  = []
    fcart = []
    etot  = []

    for iat in range(nthrow, nconfs):
        atoms = trajectory[iat]

        energy_tmp = atoms.get_potential_energy() / Hartree
        forces_tmp = atoms.get_forces() / Hartree * Bohr
        xred_tmp   = atoms.get_scaled_positions(wrap=False) # No wrapping of xred or it doesn't work

        etot.append(energy_tmp)
        fcart.append(forces_tmp)
        xred.append(xred_tmp)
    etot  = np.array(etot)
    fcart = np.array(fcart).reshape(-1, 3)
    xred  = np.array(xred).reshape(-1, 3)
    return etot, fcart, xred



#========================================================================================================================#
def get_elements_Z_and_masses(supercell):
    '''
    Get the unique chemical symbols and atomic numbers, for LAMMPS compatibility
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


def create_random_structures(atoms, std, nconfs):
    rng = np.random.default_rng()
    confs = []
    for i in range(nconfs):
        iatoms = atoms.copy()
        iatoms.rattle(stdev=std)
        confs.append(iatoms)
    return confs
