"""
Example of postprocessing with *HIST.nc file.
Requires prior execution of mlacs_32Ag_EMT_300K10GPa_Snap.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from mlacs.utilities.io_abinit import HistFile

plt.rcdefaults()
plt.rcParams["font.size"] = 10
plt.rcParams['figure.dpi'] = 300

# Requires prior execution of mlacs_32Ag_EMT_300K10GPa_Snap.py
workdir = os.path.basename(__file__).split('.')[0].split('post_')[-1]
path = Path().absolute()
prefix = f'mlacs_{workdir}'
ncname = glob.glob(f'{prefix}/*_HIST.nc')[0]
ncpath = str(path / ncname)

if os.path.isfile(ncpath):
    ncfile = HistFile(ncpath=ncpath)
    # print('HIST.nc file format: ', ncfile.ncformat)

    weights_ncpath = ncpath
    if 'NETCDF3' in ncfile.ncformat:
        weights_ncpath = ncpath.replace('HIST', 'WEIGHTS')
    weights_ncfile = HistFile(ncpath=weights_ncpath)

    var_names = ncfile.get_var_names()
    dict_var_units = ncfile.get_units()
    var_dim_dict = ncfile.var_dim_dict
    dict_name_label = {x[0]: label for label, x in var_dim_dict.items()}

    obs_name = 'etotal'
    observable = ncfile.read_obs(obs_name)
    obs_label = obs_name
    if obs_name in dict_name_label:
        obs_label = dict_name_label[obs_name].replace("_", " ")

    res_all = ncfile.read_all()

    obs_meta = ncfile.read_obs(obs_name + '_meta')

    # Mlacs iteration index
    mlacs_idx = obs_meta[:, 0]
    # Index of state
    state_idx = obs_meta[:, 1]
    # Index of configuration in database
    confs_idx = np.array([i+1 for i in range(len(observable))])

    w_obs_data, w_obs_idx = ncfile.read_weighted_obs('weighted_' + obs_name)

    weights = weights_ncfile.read_obs('weights')
    weights_meta = weights_ncfile.read_obs('weights_meta')

    fig, ax = plt.subplots()
    ax.plot(confs_idx, observable, marker='+', label='raw data', alpha=0.7)
    ax.plot(w_obs_idx, w_obs_data, c='g', marker='.', label='uniform weights')

    ax.set_xlabel(
        'Configuration index in database \n[training confs. excluded]')
    ylabel = obs_label
    try:
        obs_unit = dict_var_units[obs_name]
        ylabel += ' [' + obs_unit + ']'
    except KeyError:
        msg = 'No unit found for ' + str(obs_name)
        raise KeyError(msg)
    ax.set_ylabel(ylabel)

    legend1 = ax.legend(frameon=False, loc='best')
    legend1.get_frame().set_facecolor('none')
    plt.savefig(str(path / prefix / f'{workdir}_plot.pdf'))

else:
    msg = '*HIST.nc file not found.\n'
    msg += 'This example requires prior execution of '
    msg += 'mlacs_32Ag_EMT_300K10GPa_Snap.py'
    raise FileNotFoundError(msg)
