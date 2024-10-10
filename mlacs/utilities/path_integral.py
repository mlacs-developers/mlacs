"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kB, fs

hbar = 6.582119514e-16 * 1e15 * fs  # from eV.s to eV.(ASE time units)


# ========================================================================== #
def compute_centroid_atoms(confs, temperature):
    """
    Function to compute the centroid

    Parameters
    ----------

    confs: :class:`list` of :class:`ase.Atoms`
        The configurations of the quantum polymer
    temperature: :class:`float`

    Returns
    -------

    atoms: :class:`ase.Atoms`
        The centroid of the quantum polymer
    """
    nbead = len(confs)
    atoms = confs[0].copy()
    natoms = len(atoms)
    masses = confs[0].get_masses()
    momenta = np.zeros((natoms, 3))

    kBT = kB * temperature

    pos = np.zeros((nbead, natoms, 3))
    forces = np.zeros((nbead, natoms, 3))
    epot = np.zeros(nbead)
    stress = np.zeros((nbead, 6))
    cell = np.zeros((nbead, 3, 3))
    for ibead, at in enumerate(confs):
        pos[ibead] = at.get_positions()
        forces[ibead] = at.get_forces()
        epot[ibead] = at.get_potential_energy()
        stress[ibead] = at.get_stress()
        cell[ibead] = at.get_cell()

    cpos = pos.mean(axis=0)
    cforces = forces.mean(axis=0)
    cekin = 1.5 * natoms * kBT - 0.5 * np.sum((pos - cpos) * forces) / nbead
    cenergy = epot.mean()  # + cekin
    cstress = stress.mean(axis=0)
    ccell = cell.mean(axis=0)
    momenta[0, 0] = np.sqrt(2*cekin*masses[0])

    atoms.set_positions(cpos)
    atoms.set_cell(ccell, True)
    atoms.set_momenta(momenta)

    calc = SinglePointCalculator(atoms,
                                 energy=cenergy,
                                 forces=cforces,
                                 stress=cstress)
    atoms.calc = calc
    return atoms
