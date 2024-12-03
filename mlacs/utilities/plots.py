"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import compute_correlation
from mlacs.utilities.io_abinit import MlacsHist
from mlacs.utilities.units import unit_converter

cyan = "#17becf"
blue = "#1f77b4"
red = "#d62728"
orange = "#ff7f0e"
green = "#2ca02c"
violet = "#9467bd"
grey = "#7f7f7f"

colors = [blue, red, orange, green, violet, cyan, grey]


def plot_correlation(ax,
                     data,
                     color=blue,
                     marker="o",
                     datatype=None,
                     density=False,
                     weight=None,
                     cmap="inferno",
                     showrmse=True,
                     showmae=True,
                     showrsquared=True,
                     size=5,
                     axcbar=None):
    """
    Function to plot the correlation between true and model data on an axes

    Parameters:
    -----------
    ax: Axes.axes
        The axes on which to plot the data
    data: `np.ndarray`
        The data to plot. Has to be of shape (n, 2)
        with n the number of datapoint.
    color:
        The color of the marker in the scatter plot.
        Ignored if density is True.
    marker:
        Marker type in the scatter plot.
    datatype: `None` or `str`
        The type of data. Can be either "energy", "forces" or "stress"
    density: `Bool`
        If True, each datapoint is colored according to the density
        of data
    cmap: `str`
        The colormap used if density is True.
        Ignored if density is False
    showrmse: `Bool`
        Whether to show the RMSE on the plot
    showmae: `Bool`
        Whether to show the MAE on the plot
    showrsquared: `Bool`
        Whether to show the R^2 on the plot

    Returns:
    --------
    ax
    """

    if datatype == "energy":
        data[:, 1] -= data[:, 0].min()
        data[:, 0] -= data[:, 0].min()

    cancbar = np.any([weight is not None, density])
    if axcbar is not None and not cancbar:
        msg = "You need weight or density to use plot a color bar"
        raise ValueError(msg)

    datatrue = data[:, 0]
    datatest = data[:, 1]

    mindata = data.min()
    maxdata = data.max()
    minmax = [mindata, maxdata]

    rmse, mae, rsquared = compute_correlation(data, weight)

    if density:
        xy = np.vstack([datatrue, datatest])
        z = gaussian_kde(xy)(xy)
        norm = mpl.colors.LogNorm(z.min(), z.max())
        idx = z.argsort()
        plot = ax.scatter(datatrue[idx], datatest[idx], c=z[idx],
                          linewidths=5, norm=norm, s=size, cmap=cmap)
        if axcbar is not None:
            mpl.colorbar.Colorbar(axcbar, plot, cmap=cmap, norm=norm)
            axcbar.set_ylabel("Density")

    elif weight is not None:
        if datatype != "energy":
            w = []
            if len(datatrue) % len(weight) == 0:
                n = int(len(datatrue)/len(weight))
                for i, _w in enumerate(weight):
                    w.extend(_w * np.ones(n) / n)
            else:
                msg = "Number of weight not consistent with the Database"
                raise ValueError(msg)
            weight = np.r_[w] / np.sum(np.r_[w])
        # We add a small number to the min to avoid a possible 0 with the log
        norm = mpl.colors.LogNorm(weight.min() + 1e-8, weight.max())
        plot = ax.scatter(datatrue, datatest, c=weight,
                          linewidths=5, norm=norm, s=size, cmap=cmap)
        if axcbar is not None:
            mpl.colorbar.Colorbar(axcbar, plot, cmap=cmap, norm=norm)
            axcbar.set_ylabel("Weight")
    else:
        ax.plot(datatrue, datatest, ls="", marker=marker,
                c=color, rasterized=True, markersize=size,
                markeredgewidth=size/5)
    ax.plot(minmax, minmax, ls="--", alpha=0.75, c=red)

    if datatype is not None:
        if datatype == "energy":
            labelx = "True energy [eV/at]"
            labely = "Model energy [eV/at]"
            unit = "[eV/at]"
        elif datatype == "forces":
            labelx = "True forces [eV/angs]"
            labely = "Model forces [eV/angs]"
            unit = "[eV/angs]"
        elif datatype == "stress":
            labelx = "True stress [GPa]"
            labely = "Model stress [GPa]"
            unit = "[GPa]"
        else:
            msg = "datVatype should be energy, forces or stress"
            raise ValueError(msg)
    else:
        labelx = None
        labely = None
        unit = ""

    if showrmse:
        ax.text(0.01, 0.9, f"RMSE = {rmse:5.4f} {unit}",
                fontsize=30,
                transform=ax.transAxes)
    if showmae:
        ax.text(0.01, 0.8, f"MAE = {mae:5.4f} {unit}",
                fontsize=30,
                transform=ax.transAxes)
    if showrsquared:
        ax.text(0.01, 0.7, f"R$^2$ = {rsquared:5.4f}",
                fontsize=30,
                transform=ax.transAxes,)

    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    ax.set_xlim(minmax)
    ax.set_ylim(minmax)
    return ax


def plot_error(ax,
               data,
               color=blue,
               datatype=None,
               showrmse=True,
               showmae=True,
               showrsquared=True):
    """
    Function to plot the error distribution between true and model data

    Parameters:
    -----------
    ax: Axes.axes
        The axes on which to plot the data
    data: `np.ndarray`
        The data to plot. Has to be of shape (n, 2)
        with n the number of datapoint.
    color:
        The color of the marker in the scatter plot.
        Ignored if density is True.
    datatype: `None` or `str`
        The type of data. Can be either "energy", "forces" or "stress"
    showrmse: `Bool`
        Whether to show the RMSE on the plot
    showmae: `Bool`
        Whether to show the MAE on the plot
    showrsquared: `Bool`
        Whether to show the R^2 on the plot

    Returns:
    --------
    ax
    """
    dataerror = data[:, 0] - data[:, 1]

    if datatype is not None:
        if datatype == "energy":
            dataerror *= 1000
            labelx = "Energy error [meV/at]"
            unit = "[meV/at]"
        elif datatype == "forces":
            dataerror *= 1000
            labelx = "Forces error [meV/angs]"
            unit = "[meV/angs]"
        elif datatype == "stress":
            labelx = "Stress error [GPa]"
            unit = "[GPa]"
        else:
            msg = "datatype should be energy, forces or stress"
            raise ValueError(msg)
    else:
        labelx = None
        unit = ""

    errmin = -3 * dataerror.std()
    errmax = 3 * dataerror.std()
    errmean = dataerror.mean()
    minmax = [errmin - errmean, errmean + errmax]

    rmse, mae, rsquared = compute_correlation(data)

    kdeerror = gaussian_kde(dataerror)

    x = np.linspace(errmin, errmax, 1000)
    kde_pred = kdeerror(x)
    kde_pred *= 100 / (kde_pred).sum()
    ax.axvline(errmean, c=grey, ls="--")
    ax.plot(x, kde_pred, c="k")
    ax.fill_between(x, kde_pred, alpha=0.75)

    if showrmse:
        ax.text(0.01, 0.9, f"RMSE = {rmse:5.4f} {unit}",
                fontsize=30,
                transform=ax.transAxes)
    if showmae:
        ax.text(0.01, 0.8, f"MAE = {mae:5.4f} {unit}",
                fontsize=30,
                transform=ax.transAxes)
    if showrsquared:
        ax.text(0.01, 0.7, f"R$^2$ = {rsquared:5.4f}",
                fontsize=30,
                transform=ax.transAxes,)

    ax.set_xlabel(labelx)
    ax.set_ylabel("Density [%]")
    ax.set_xlim(minmax)
    ax.set_ylim(0)
    return ax


def plot_weights(ax, weights, color=blue, fontsize=30):
    """
    Function to plot the error distribution between true and model data

    Parameters:
    -----------
    ax: Axes.axes
        The axes on which to plot the data
    weights: `np.ndarray`
        The weights to plot.
    color:
        The color of the marker in the scatter plot.
        Ignored if density is True.

    Returns:
    --------
    ax
    """
    xrange = np.arange(len(weights))
    neff = np.sum(weights)**2 / np.sum(weights**2)

    ax.bar(xrange, weights)
    ax.text(0.01, 0.9, f"Eff. N. conf = {neff:5.4f}",
            transform=ax.transAxes)
    ax.set_ylim(0)
    ax.set_xlabel("Configuration index")
    ax.set_ylabel("Weight")
    return ax


def init_rcParams():
    """
    """
    mpl.rcParams["lines.linewidth"] = 5
    mpl.rcParams["lines.markeredgecolor"] = "k"
    mpl.rcParams["lines.markersize"] = 25
    mpl.rcParams["lines.markeredgewidth"] = 5

    mpl.rcParams['figure.dpi'] = 300

    mpl.rcParams["font.size"] = 30

    mpl.rcParams["axes.linewidth"] = 5

    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.major.size"] = 12
    mpl.rcParams["xtick.major.width"] = 5
    mpl.rcParams["xtick.direction"] = "in"

    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.major.size"] = 12
    mpl.rcParams["ytick.major.width"] = 5
    mpl.rcParams["ytick.direction"] = "in"


# ========================================================================== #
# ========================================================================== #
class HistPlot:
    """
    Class to handle the plots of Abinit-like *HIST.nc file.

    Parameters
    ----------

    ncpath: :class:`str` or :class:`Path` of `pathlib` module (optional)
        Absolute path to *HIST.nc file, i.e. `path_to_ncfile/ncfilename`.
        Default ''.
    """

    def __init__(self,
                 ncpath=''):

        mpl.rcdefaults()
        mpl.rcParams["font.size"] = 10
        mpl.rcParams['figure.dpi'] = 300

        if os.path.isfile(ncpath):
            ncfile = MlacsHist(ncpath=ncpath)
            dict_var_units = ncfile.get_units()
            var_dim_dict = ncfile.var_dim_dict
            dict_name_label = {x[0]: lab for lab, x in var_dim_dict.items()}
            dict_name_label['press'] = 'Pressure'
            self.ncfile = ncfile
            self.dict_name_label = dict_name_label
            self.dict_var_units = dict_var_units
            self.basic_obs = ['temper', 'etotal', 'press', 'vol']
            self.energy_obs = ['ekin', 'epot']

            weights_ncpath = ncpath
            if 'NETCDF3' in ncfile.ncformat:
                weights_ncpath = ncpath.replace('HIST', 'WEIGHTS')
            self.weights_ncfile = MlacsHist(ncpath=weights_ncpath)
        else:
            msg = '*HIST.nc file not found.'
            raise FileNotFoundError(msg)

# ========================================================================== #
    def _initialize_weights(self, get_dict_weights=False):
        """
        Initialize attributes that are relevant when weights are needed.
        """
        weights_ncfile = self.weights_ncfile
        weights = weights_ncfile.read_obs('weights')
        weights_meta = weights_ncfile.read_obs('weights_meta')
        weights_idx = weights_meta[:, 0]
        nb_effective_conf = weights_meta[:, 1][weights_idx == 1.0]
        nb_conf = weights_meta[:, 2][weights_idx == 1.0]

        self.weights = weights
        self.weights_idx = weights_idx
        self.nb_effective_conf = nb_effective_conf
        self.nb_conf = nb_conf
        self.number_of_states = weights_meta[0][-1]

        # Check if weights are uniform within tolerance
        self.uniform_weight = np.allclose(nb_conf, nb_effective_conf,
                                          atol=1e-10)

# ========================================================================== #
    def _get_dict_weights(self):
        """
        Compute a dictionary that maps MLACS iterations to weights data.
        """
        weights_idx = self.weights_idx
        weights = self.weights

        dict_weights = {}
        idx_bounds = np.argwhere(weights_idx == 1.0)[:, 0]

        for ii in range(len(idx_bounds)-1):
            iter_mlacs = ii+1
            i1, i2 = idx_bounds[ii], idx_bounds[ii+1]
            dict_weights[iter_mlacs] = [weights_idx[i1:i2], weights[i1:i2]]
        dict_weights[iter_mlacs+1] = [weights_idx[i2:], weights[i2:]]

        self.dict_weights = dict_weights
        self.available_iter = list(dict_weights.keys())

# ========================================================================== #
    def _plot_weight_distribution(self, iter_loc, ax_loc, normalize=True):
        """
        Plot the weight distribution for a specified MLACS iteration.
        """
        dict_weights = self.dict_weights
        nb_effective_conf = self.nb_effective_conf

        weights_idx = dict_weights[iter_loc][0]-1
        weights = dict_weights[iter_loc][1]

        xlabel = r"Number of configurations in database"
        ylabel = r"Weights"
        if normalize:
            weights /= np.mean(weights)
            ylabel += r"/ $ \langle $Weights$ \rangle $"

        Nconfs_loc = np.round(nb_effective_conf[iter_loc-1], 1)
        lab_str = r'$N_{\text{eff}} \simeq$'+'{}'.format(Nconfs_loc)

        ax_loc.step(weights_idx, weights, where='mid', label=lab_str,
                    zorder=10-iter_loc)
        ax_loc.set_xlabel(xlabel)
        ax_loc.set_ylabel(ylabel)

# ========================================================================== #
    def _core_plot(self, obs_name, fig=None, ax=None):
        """Plot the observable named `obs_name` in the ncfile."""
        if None in (fig, ax):
            fig, ax = plt.subplots()

        ncfile = self.ncfile
        dict_name_label = self.dict_name_label
        dict_var_units = self.dict_var_units

        try:
            nc_unit = dict_var_units[obs_name]
        except KeyError:
            msg = 'No unit found for ' + str(obs_name)
            nc_unit = ''
            raise KeyError(msg)

        # Read atomic unit observable from *HIST.nc file
        obs_au = ncfile.read_obs(obs_name)
        # Convert to custom visualization units
        observable, new_unit = unit_converter(obs_au, nc_unit, target='custom')

        obs_meta = ncfile.read_obs(obs_name + '_meta')

        state_idx = obs_meta[:, 1]
        confs_idx = np.arange(1, len(observable)+1)

        w_obs_au, w_obs_idx = ncfile.read_weighted_obs('weighted_'+obs_name)
        w_obs_data, new_unit = unit_converter(w_obs_au,
                                              nc_unit,
                                              target='custom')

        uniform_obs = np.array([np.mean(observable[:i]) for i in w_obs_idx])

        ax.plot(confs_idx, observable, label='raw data', alpha=0.7)
        ax.plot(w_obs_idx, uniform_obs, c='g', label='uniform weights',
                marker='.')

        self._initialize_weights()
        if not self.uniform_weight:
            ax.plot(w_obs_idx, w_obs_data, c='r', ls='-', label='mbar',
                    marker='.')

        ax.set_xlabel('Configuration index in database')
        obs_label = obs_name
        if obs_name in dict_name_label:
            obs_label = dict_name_label[obs_name].replace("_", " ")
        ylabel = obs_label
        if new_unit != '':
            ylabel += ' [' + new_unit + ']'
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc='best')
        par_title = (int(len(confs_idx)), int(max(state_idx)),)
        str_title = '# configurations: {}, # states: {}'.format(*par_title)
        fig.suptitle(str_title)

# ========================================================================== #
    def plot_thermo_basic(self, show=True, savename=''):
        """Plot the set of observables in self.basic_obs"""
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        for idx, ax_loc in enumerate(ax.reshape(-1)):
            obs_name = self.basic_obs[idx]
            self._core_plot(obs_name, fig, ax_loc)
        fig.tight_layout()

        dynamic_savename = f"{savename or 'plot_thermo'}.pdf"
        fig.savefig(dynamic_savename, bbox_inches='tight')

        if show:
            os.system('xdg-open '+dynamic_savename)

# ========================================================================== #
    def plot_neff(self, show=True, savename='', selected_iter=None):
        """Plot Neff against Nconfs, along with some weight distributions."""
        self._initialize_weights()
        self._get_dict_weights()
        nb_conf = self.nb_conf
        nb_effective_conf = self.nb_effective_conf
        available_iter = self.available_iter
        number_of_states = self.number_of_states

        fig, ax = plt.subplots(1, 2, figsize=(7, 3))

        ax[0].plot(nb_conf, nb_effective_conf)
        ax[0].plot(nb_conf, nb_conf, c='k', ls=':', label=r'$y=x$')
        ax[0].set_xlabel('Number of configurations in database')
        ax[0].set_ylabel('Number of effective configurations')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')

        # Secondary x-axis for MLACS iterations
        def f(x): return x/number_of_states
        def g(x): return x*number_of_states
        second_x_axis_0 = ax[0].secondary_xaxis("top", functions=(f, g))
        second_x_axis_0.set_xlabel("MLACS iteration")
        ax[0].legend(frameon=False, loc=4)

        # Identify MLACS iterations to plot
        mlacs_iter_arr = [available_iter[-1]]
        if selected_iter is None:
            if len(available_iter) > 5:
                mlacs_iter_arr = np.geomspace(3, available_iter[-1], 4,
                                              dtype=int)
        else:
            if set(selected_iter) <= set(available_iter):
                selected_iter.sort()
                mlacs_iter_arr = np.array(selected_iter)
            else:
                s = sorted(set(selected_iter))
                missing_iter = [x for x in s if x not in available_iter]
                if len(missing_iter) > 0:
                    raise ValueError(
                        "The following iteration(s) are not available: "
                        f"{missing_iter}. \n"
                        f"Available iteration(s): {available_iter}")

        # Plot weight distributions and pinpoint iterations on left panel
        for iter_mlacs in mlacs_iter_arr:
            self._plot_weight_distribution(iter_mlacs, ax[1])
            ax[0].scatter(nb_conf[iter_mlacs-1],
                          nb_effective_conf[iter_mlacs-1],
                          marker='s',
                          s=20)

        ax[1].legend(frameon=False, loc='best', ncol=2)
        fig.tight_layout()

        dynamic_savename = f"{savename or 'plot_neff'}.pdf"
        fig.savefig(dynamic_savename, bbox_inches='tight')

        if show:
            os.system('xdg-open '+dynamic_savename)

# ========================================================================== #
    def plot_ith_weights(self, show=True, savename='', selected_iter=None):
        """Plot MLACS weights distribution for iteration i."""
        self._initialize_weights()
        self._get_dict_weights()
        available_iter = self.available_iter

        # Identify MLACS iterations to plot
        if selected_iter is None:
            mlacs_iter_arr = [available_iter[-1]]
        else:
            if set(selected_iter) <= set(available_iter):
                selected_iter.sort()
                mlacs_iter_arr = np.array(selected_iter)
            else:
                s = sorted(set(selected_iter))
                missing_iter = [x for x in s if x not in available_iter]
                if len(missing_iter) > 0:
                    raise ValueError(
                        "The following iteration(s) are not available: "
                        f"{missing_iter}. \n"
                        f"Available iteration(s): {available_iter}")

        # Plot weight distributions
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for iter_mlacs in mlacs_iter_arr:
            self._plot_weight_distribution(iter_mlacs, ax, normalize=False)
        ax.legend(frameon=False, loc='best')
        fig.tight_layout()

        dynamic_savename = f"{savename or 'plot_weights'}.pdf"
        fig.savefig(dynamic_savename, bbox_inches='tight')

        if show:
            os.system('xdg-open '+dynamic_savename)
