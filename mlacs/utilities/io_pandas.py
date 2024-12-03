"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np


try:
    import pandas as pd
except ImportError:
    pass

def make_dataframe(df, name, atoms, atomic_env,
                   energy=None, forces=None, we=None, wf=None):
    """
    Append atoms information to the dataframe.
    Return the dataframe WITHOUT writing in a file

    2 modes : 1. We don't yet know energy, forces, we, wf
              2. We also add energy, forces, we, wf

    name: :class:`list` :shape:`[nconfs]`
        Name of each configuration

    atoms: :class:`list` :shape:`[nconfs]`
        Ase.Atoms object for each configuration

    atomic_env: :class:`list` :shape:`[nconfs]`
        pyace.catomicenvironment.ACEAtomicEnvironment for each confs

    energy: :class:`np.array` :shape:`[nconfs]`
        Energy of each configuration

    forces: :class:`list` :shape:`[nconfs][natoms:3]`
        list of np.array containing the forces on each atoms

    we: :class:`np.array` :shape:`[nconfs]`
        Weight of each configuration

    wf: :class:`list` :shape:`[nconfs][natoms]`
        list of np.array containing the weight for the forces on each atom
        The sum must be equal to 1.
        The relative weight between e and f is given by alpha
    """
    add_ef = all(_ is not None for _ in (energy, forces, we, wf))
    nat = np.array([])
    for at in atoms:
        nat = np.append(nat, len(at))

    # Creating the dict and adding it to the df
    if add_ef:
        to_add = dict(name=name, ase_atoms=atoms,
                      energy_corrected=energy, forces=forces,
                      NUMBER_OF_ATOMS=nat, atomic_env=atomic_env,
                      w_energy=we, w_forces=wf)
    else:
        to_add = dict(name=name, ase_atoms=atoms,
                      NUMBER_OF_ATOMS=nat, atomic_env=atomic_env)

    return pd.DataFrame(to_add)
