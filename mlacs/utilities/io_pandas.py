try:
    import pandas as pd
except ImportError:
    pass

# ========================================================================== #
def create_dataframe():
    """Create a new empty dataframe"""
    columns = ["name", "ase_atoms", "energy_corrected", "forces",
               "atomic_env", "NUMBER_OF_ATOMS","w_energy","w_forces"]
    return pd.DataFrame(columns=columns)

# ========================================================================== #
def update_dataframe(atoms, descriptor, add_result, df_fn=None, df=None, mbar=None):
    """
    Add the new configuration atoms to the dataframe. 
    add_result = True : Compute e,f,s and recalculate weights
    add_result = False : Only put the atomic position
    It doesn't write the file  
    """
    if df is None:
        if df_fn is None:
            df_fn = descriptor.df_fn
        df = pd.read_pickle(df_fn, compression="gzip")

    # The 8 columns
    name, ase_atoms, NUMBER_OF_ATOMS, = [],[],[]
    if add_result:
        e_corr, forces, w_energy, w_forces = [],[],[],[]
    atomic_env = descriptor.get_atomic_env(atoms)

    for at in atoms:
        name.append(f"conf{len(df.index)}")
        ase_atoms.append(at)
        NUMBER_OF_ATOMS.append(len(at))
        if add_result:
            free_e = descriptor.calc_free_e(at)
            e_corr.append(at.get_potential_energy() - free_e)
            forces.append(at.get_forces())
    if add_result:
        w_energy, w_forces = descriptor.calc_weights(df, mbar, atoms)
        to_add = dict(name=name, ase_atoms=ase_atoms,
                      energy_corrected=e_corr, forces=forces,
                      NUMBER_OF_ATOMS=NUMBER_OF_ATOMS, atomic_env=atomic_env,
                      w_energy=w_energy[-len(name):], w_forces=w_forces[-len(name):])
        df['w_energy'] = w_energy[:-len(name)]
        df['w_forces'] = w_forces[:-len(name)]


    else:
        to_add = dict(name=name, ase_atoms=ase_atoms,
                      NUMBER_OF_ATOMS=NUMBER_OF_ATOMS, atomic_env=atomic_env)

    new_data = pd.DataFrame(to_add)
    return pd.concat([df, new_data], ignore_index=True)
