import pandas as pd
import numpy as np 

# ========================================================================== #
def create_dataframe():
    """Create a new empty dataframe"""
    columns = ["name", "ase_atoms", "energy_corrected", "forces",
               "atomic_env", "NUMBER_OF_ATOMS","w_energy","w_forces"]
    return pd.DataFrame(columns=columns)

# ========================================================================== #
def update_dataframe(df, name, atoms, atomic_env, 
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
        pyace.catomicenvironment.ACEAtomicEnvironment for each configuration

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

    if not isinstance(df,pd.DataFrame):
        if isinstance(df, str):
            df = pd.read_pickle(df, compression="gzip")
        else:
            raise ValueError("Unrecognized type for the dataframe")

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

    new_data = pd.DataFrame(to_add)
    return pd.concat([df, new_data], ignore_index=True)

