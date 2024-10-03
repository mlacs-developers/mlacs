"""
Example of postprocessing with *HIST.nc file.
Requires prior execution of tuto_mbar.py
"""

from mlacs.utilities.io_abinit import HistFile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
plt.rcParams["font.size"] = 10
plt.rcParams['figure.dpi'] = 300

# Requires prior execution of tuto_mbar.py
workdir = 'run_tuto_mbar'
path = Path().absolute()
script_name = 'tuto_mbar'
ncname = script_name + '_HIST.nc'
ncpath = str(path / workdir / ncname)

if os.path.isfile(ncpath):
    ncfile = HistFile(ncpath=ncpath)
    # print('HIST.nc file format: ', ncfile.ncformat)

    var_names = ncfile.get_var_names()
    dict_var_units = ncfile.get_units()
    var_dim_dict = ncfile.nc_routine_conv()[0]
    dict_name_label = {x[0]: label for label, x in var_dim_dict.items()}
    # print('Variables names: ', var_names)

    obs_name = 'temper'
    # obs_name = 'press'
    # obs_name = 'vol'
    # obs_name = 'etotal'
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

    uniform_obs = np.array([np.mean(observable[:i]) for i in w_obs_idx])

    fig1, ax1 = plt.subplots()
    ax1.plot(confs_idx, observable, label='raw data', alpha=0.7)
    ax1.plot(w_obs_idx, uniform_obs, c='g', label='uniform weights')
    ax1.plot(w_obs_idx, w_obs_data, c='r',  label='mbar')
    ax1.set_xlabel(
        'Configuration index in database \n[training confs. excluded]')
    ylabel = obs_label
    try:
        obs_unit = dict_var_units[obs_name]
        ylabel += ' [' + obs_unit + ']'
    except KeyError:
        msg = 'No unit found for ' + str(obs_name)
        raise KeyError(msg)
    ax1.set_ylabel(ylabel)

    legend1 = ax1.legend(frameon=False, loc='best')
    legend1.get_frame().set_facecolor('none')

    weights = ncfile.read_obs('weights')
    weights_meta = ncfile.read_obs('weights_meta')
    weights_idx = weights_meta[:, 0]
    nb_effective_conf = weights_meta[:, 1][weights_idx == 1.0]
    nb_conf = weights_meta[:, 2][weights_idx == 1.0]

    fig3, ax3 = plt.subplots(1, 2, figsize=(7, 3))
    ax3[0].plot(nb_conf, nb_effective_conf)
    ax3[0].plot(nb_conf, nb_conf, c='k', ls=':', label=r'$y=x$')
    ax3[0].set_xlabel('Number of configurations in database')
    ax3[0].set_ylabel('Number of effective configurations')
    ax3[0].set_xscale('log')
    ax3[0].set_yscale('log')
    legend3_0 = ax3[0].legend(frameon=False, loc=4)
    legend3_0.get_frame().set_facecolor('none')

    # dict_weights maps an Mlacs iteration index to its Mbar data [idx,weights]
    dict_weights = {}
    idx_bounds = np.argwhere(weights_idx == 1.0)[:, 0]
    for ii in range(len(idx_bounds)-1):
        iter_mlacs = ii+1
        i1, i2 = idx_bounds[ii], idx_bounds[ii+1]
        dict_weights[iter_mlacs] = [weights_idx[i1:i2], weights[i1:i2]]

    def _plot_distribution(iter_loc):
        loc_weights_idx = dict_weights[iter_loc][0]
        normalized_x = (loc_weights_idx-1)/(loc_weights_idx[-1]-1)
        loc_weights = dict_weights[iter_loc][1]
        normalized_y = loc_weights/np.mean(loc_weights)
        Nconfs_loc = np.round(nb_effective_conf[iter_loc-1], 1)
        lab_str = r'$N_{\text{eff}} \simeq$'+'{}'.format(Nconfs_loc)
        ax3[1].step(normalized_x, normalized_y, where='mid', label=lab_str)

    if len(idx_bounds)-1 > 5:
        mlacs_iter_arr = np.geomspace(3, len(idx_bounds)-1, 4, dtype=int)
    else:
        mlacs_iter_arr = [1, len(idx_bounds)-1]

    for iter_mlacs in mlacs_iter_arr:
        _plot_distribution(iter_mlacs)
        ax3[0].scatter(nb_conf[iter_mlacs-1],
                       nb_effective_conf[iter_mlacs-1],
                       marker='s',
                       s=20)

    ax3[1].set_xlabel(r"Normalized config. index")
    ax3[1].set_ylabel(r"Weights / $ \langle $Weights$ \rangle $")
    ax3[1].set_title('Evolution of the distribution of weights',
                     fontsize=plt.rcParams["font.size"])

    legend3_1 = ax3[1].legend(frameon=False, loc='best', ncol=2)
    legend3_1.get_frame().set_facecolor('none')
    fig3.tight_layout()


else:
    msg = '*HIST.nc file not found.\n'
    msg += 'This example requires prior execution of tuto_mbar.py'
    raise FileNotFoundError(msg)
