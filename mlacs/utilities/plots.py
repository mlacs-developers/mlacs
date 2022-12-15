import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl

from . import compute_correlation

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
                     datatype=None,
                     density=False,
                     cmap="inferno",
                     showrmse=True,
                     showmae=True,
                     showrsquared=True):
    """
    Function to plot the correlation between true and model data on an axes

    Parameters:
    -----------
    ax: Axes.axes
        The axes on which to plot the data
    data: `np.ndarray`
        The data to plot. Has to be of shape (n, 2)
        with n the number of datapoint
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
        data[:, 0] -= data[:, 0].min()
        data[:, 1] -= data[:, 1].min()

    datatrue = data[:, 0]
    datatest = data[:, 1]

    mindata = data.min()
    maxdata = data.max()
    minmax = [mindata, maxdata]

    rmse, mae, rsquared = compute_correlation(data)

    if density:
        xy = np.vstack([datatrue, datatest])
        z = gaussian_kde(xy)(xy)
        norm = mpl.colors.LogNorm(z.min(), z.max())
        idx = z.argsort()
        ax.scatter(datatrue[idx], datatest[idx], c=z[idx],
                   linewidths=5, norm=norm, s=50, cmap=cmap)
    else:
        ax.plot(datatrue, datatest, ls="", marker="o")
    ax.plot(minmax, minmax, ls="--", alpha=0.75, c=grey)

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
            msg = "datatype should be energy, forces or stress"
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
