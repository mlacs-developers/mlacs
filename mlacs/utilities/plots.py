import os

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
 
from . import compute_correlation
from mlacs.utilities.io_abinit import HistFile

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
    """
    def __init__(self,
                 ncpath=''):
        
        mpl.rcParams["font.size"] = 10
        mpl.rcParams['figure.dpi'] = 300
        
        if os.path.isfile(ncpath):
            ncfile = HistFile(ncpath=ncpath)        
            # var_names = ncfile.get_var_names()
            dict_var_units = ncfile.get_units()
            var_dim_dict = ncfile.nc_routine_conv()[0]
            dict_name_label = {x[0]: lab for lab, x in var_dim_dict.items()}
            dict_name_label['press'] = 'Pressure'
            self.ncfile = ncfile
            self.dict_name_label = dict_name_label
            self.dict_var_units = dict_var_units
            self.basic_obs = ['temper', 'etotal', 'press', 'vol']
            self.energy_obs = ['ekin', 'epot']
        else:
            msg = '*HIST.nc file not found.' 
            raise FileNotFoundError(msg)

# ========================================================================== #
    def _core_plot(self, obs_name, fig=None, ax=None):
        """  """
        if None in (fig, ax):
            fig, ax = plt.subplots()

        ncfile = self.ncfile
        dict_name_label = self.dict_name_label
        dict_var_units = self.dict_var_units

        observable = ncfile.read_obs(obs_name)
        obs_label = obs_name
        if obs_name in dict_name_label:
            obs_label = dict_name_label[obs_name].replace("_", " ")

        obs_meta = ncfile.read_obs(obs_name + '_meta')
        
        weights_meta = ncfile.read_obs('weights_meta')
        weights_idx = weights_meta[:, 0]
        nb_effective_conf = weights_meta[:, 1][weights_idx == 1.0]
        nb_conf = weights_meta[:, 2][weights_idx == 1.0]
        if np.max(np.abs(nb_conf-nb_effective_conf))>10**-10:
            uniform_weight = False
        else:
            uniform_weight = True
        
        # Mlacs iteration index
        mlacs_idx = obs_meta[:, 0]
        # Index of state
        state_idx = obs_meta[:, 1]
        # Index of configuration in database
        confs_idx = np.array([i+1 for i in range(len(observable))])

        w_obs_data, w_obs_idx = ncfile.read_weighted_obs('weighted_'+obs_name)
        uniform_w_obs_data = np.array([np.mean(observable[:i+1]) 
                                       for i in range(len(w_obs_idx))])

        ax.plot(confs_idx, observable, label='raw data', alpha=0.7)
        ax.plot(w_obs_idx, uniform_w_obs_data, c='g', label='uniform weights')

        if uniform_weight == False:
            ax.plot(w_obs_idx, w_obs_data, c='r', ls='-', label='mbar')
        xlabel_str = 'Configuration index in database \n'
        xlabel_str += '[training confs. excluded]'
        ax.set_xlabel(xlabel_str)
        par_title = (int(len(confs_idx)), int(max(state_idx)),)
        str_title = '# configurations: {}, # states: {}'.format(*par_title)
        fig.suptitle(str_title)
        ylabel = obs_label
        try:
            obs_unit = dict_var_units[obs_name]
            ylabel += ' [' + obs_unit + ']'
        except:
            print('No unit found for ', obs_name)
        ax.set_ylabel(ylabel)

        legend1 = ax.legend(frameon=False, loc='best')
        legend1.get_frame().set_facecolor('none')

# ========================================================================== #
    def plot_thermo_basic(self, show=True, savename=''):
        fig, ax = plt.subplots(2, 2, figsize=(9, 7))
        for idx,ax_loc in enumerate(ax.reshape(-1)):
            obs_name = self.basic_obs[idx]
            self._core_plot(obs_name, fig, ax_loc)
        fig.tight_layout()
        if savename == '':
            savename = 'plot_thermo'
        savename += '.jpeg'
        fig.savefig(savename, bbox_inches='tight')
        if show == True:
            os.system('xdg-open '+savename)

# ========================================================================== #
    def plot_neff(self, show=True, savename=''):
        ncfile = self.ncfile
        weights = ncfile.read_obs('weights')
        weights_meta = ncfile.read_obs('weights_meta')
        weights_idx = weights_meta[:, 0]
        nb_effective_conf = weights_meta[:, 1][weights_idx == 1.0]
        nb_conf = weights_meta[:, 2][weights_idx == 1.0]

        fig, ax = plt.subplots(1, 2, figsize=(7, 3))
        ax[0].plot(nb_conf, nb_effective_conf)
        ax[0].plot(nb_conf, nb_conf, c='k', ls=':', label=r'$y=x$')
        ax[0].set_xlabel('Number of configurations in database')
        ax[0].set_ylabel('Number of effective configurations')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        legend_0 = ax[0].legend(frameon=False, loc=4)
        legend_0.get_frame().set_facecolor('none')
        
        # dict_weights maps an Mlacs iteration index to its Mbar data
        dict_weights = {}
        idx_bounds = np.argwhere(weights_idx == 1.0)[:,0]
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
            ax[1].step(normalized_x, normalized_y, where='mid',label=lab_str)
        
        if len(idx_bounds)-1 > 5:
            mlacs_iter_arr = np.geomspace(3, len(idx_bounds)-1, 4, dtype=int)
        else:
            mlacs_iter_arr = [2]
        for iter_mlacs in mlacs_iter_arr:
            _plot_distribution(iter_mlacs)
            ax[0].scatter(nb_conf[iter_mlacs-1],
                           nb_effective_conf[iter_mlacs-1],
                           marker='s',
                           s=20)
            
        ax[1].set_xlabel(r"Normalized config. index")
        ax[1].set_ylabel(r"Weights / $ \langle $Weights$ \rangle $")
        ax[1].set_title('Evolution of the distribution of weights',
                         fontsize=plt.rcParams["font.size"])
        
        legend_1 = ax[1].legend(frameon=False, loc='best', ncol=2)
        legend_1.get_frame().set_facecolor('none')
        fig.tight_layout()

        if savename == '':
            savename = 'plot_neff'
        savename += '.jpeg'
        fig.savefig(savename, bbox_inches='tight')

        if show == True:
            os.system('xdg-open '+savename)
        


