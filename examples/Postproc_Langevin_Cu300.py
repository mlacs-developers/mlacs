"""
Example of postprocessing with *HIST.nc file.
Requires prior execution of Langevin_Cu300K.py
"""

from mlacs.utilities.io_abinit import HistFile

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 300


# Requires prior execution of Langevin_Cu300K.py
workdir = 'run_Langevin_Cu300K'
path = Path().absolute()
# script_name = 'Langevin_Cu300K'
script_name = 'Langevin_Cu300K'
ncname = script_name + '_HIST.nc'
ncpath = str(path / workdir / ncname)

if os.path.isfile(ncpath):
    ncfile = HistFile(ncpath=ncpath)
    print('HIST.nc file format: ', ncfile.ncformat)

    var_names = ncfile.get_var_names()
    dict_var_units = ncfile.get_units()

    obs_name = 'etotal'
    observable = ncfile.read_obs(obs_name)

    res_all = ncfile.read_all()

    obs_meta = ncfile.read_obs(obs_name + '_meta')

    # Mlacs iteration index
    mlacs_idx = obs_meta[:, 0]
    # Index of state
    state_idx = obs_meta[:, 1]
    # Index of configuration in database
    confs_idx = np.array([i+1 for i in range(len(observable))])

    w_obs_data, w_obs_idx = ncfile.read_weighted_obs('weighted_' + obs_name)

    weights = ncfile.read_obs('weights')
    weights_meta = ncfile.read_obs('weights_meta')

    fig, ax = plt.subplots()
    ax.plot(confs_idx, observable, marker='+', label='raw data', alpha=0.7)
    ax.plot(w_obs_idx, w_obs_data, c='g', marker='.', label='uniform weights')
    ax.set_xlabel('Index of configuration in database')
    ylabel = obs_name
    try:
        obs_unit = dict_var_units[obs_name]
        ylabel += ' [' + obs_unit + ']'
    except KeyError:
        msg = 'No unit found for ' + str(obs_name)
        raise KeyError(msg)
    ax.set_ylabel(ylabel)

else:
    msg = '*HIST.nc file not found.\n'
    msg += 'This example requires prior execution of Langevin_Cu300K.py'
    raise FileNotFoundError(msg)
