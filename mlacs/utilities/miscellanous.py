"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np
import subprocess 

from ase.atoms import Atoms
from ase.io import read
from ase.units import Hartree, Bohr
from ase.io.lammpsdata import write_lammps_data 

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
    charges  = supercell.get_initial_charges()

    un_elements = sorted(set(elements))
    un_Z        = []
    un_masses   = []
    un_charges  = []
    for iel in range(len(un_elements)):
        idx = elements.index(un_elements[iel])
        un_Z.append(Z[idx])
        un_masses.append(masses[idx])
        un_charges.append(charges[idx])

    if np.allclose(un_charges, 0.0, atol=1e-8):
        un_charges = None
    return un_elements, un_Z, un_masses, un_charges


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
            iatoms.rattle(stdev=std, rng=rng)
            confs.append(iatoms)
    return confs

def write_lammps_data_full(name, atoms, bonds=[], angles=[], velocities=False) : 
    """
    Write lammps data file with bonds and angles 

    Parameters
    ----------
    name : :class:`str` 
        name of the output file 
    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        ASE atoms objects to be rattled
    bonds: :class:`numpy.array`
        array of bonds list
    nconfs: :class:`numpy.array`
        array of angles list 
    Return
    ------
    """
    write_lammps_data('coord_tmp.lmp', atoms, atom_style = "full", velocities=velocities)
    with open('coord_tmp.lmp', 'r') as file : 
        lines = file.readlines()

    ind = [i for i, element in enumerate(lines) if "atoms" in element][0] 
    lines.insert(ind+1, str(len(bonds))+' bonds \n')
    lines.insert(ind+2, str(len(angles))+' angles \n')
    

    ind = [i for i, element in enumerate(lines) if "atom types" in element][0] 
    lines.insert(ind+1, str(len(np.unique(bonds[:,1])))+' bond types \n')
    lines.insert(ind+2, str(len(np.unique(angles[:,1])))+' angle types \n')

    f = open(name, 'w')
    for line in lines : 
        f.write(line)
    f.write("\n")
    f.write(" Bonds \n \n")
    np.savetxt(f, bonds, fmt='%s') 
    f.write("\n")
    f.write(" Angles \n \n")
    np.savetxt(f, angles, fmt='%s')
    f.close()
    subprocess.run('rm coord_tmp.lmp', shell=True)
