"""
"""

import numpy as np
import matplotlib.pyplot as plt
from ..utilities.plots import plot_correlation, init_rcParams


def main(args, parser):

    data = np.loadtxt(args.file)
    if args.datatype not in ["energy", "forces", "stress", None]:
        raise ValueError("The type argument has to be "
                         "energy, forces or stress")
    rmse = True
    if args.normse:
        rmse = False
    mae = True
    if args.nomae:
        mae = False
    rsquared = True
    if args.nor2:
        rsquared = False
    density = False
    if args.density:
        density = True
    if args.weight is not None:
        weight = np.loadtxt(args.weight)
    else:
        weight = None
    cmap = args.cmap
    size = float(args.size)
    figsize = (float(args.figsize), float(args.figsize))
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    init_rcParams()
    ax = fig.add_subplot()
    plot_correlation(ax,
                     data,
                     datatype=args.datatype,
                     density=density,
                     weight=weight,
                     showrmse=rmse,
                     showmae=mae,
                     showrsquared=rsquared,
                     cmap=cmap,
                     size=size)
    ax.set_aspect("equal")
    if args.save is not None:
        plt.savefig(args.save)
    if not args.noshow:
        plt.show()


class CLICommand:
    """Plot correlation of DFT and MLIP data

    Read data points in the file and plot the correlation along with
    the RMSE, MAE and M^2

    Example:

        $ mlacs correlation MLIP-Energy_comparison.dat --datatype energy
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="file with the data")
        parser.add_argument("-s", "--save", default=None,
                            help="Name of the file to save the plot. "
                                 "Default None")
        parser.add_argument("--noshow", action="store_true",
                            help="To disable the visualisation of the plot")
        parser.add_argument('--datatype', default=None,
                            help="Type of data in the file. Can be "
                            "energy, forces or stress")
        parser.add_argument('--density', action="store_true",
                            help="Color points according to the density")
        parser.add_argument('--weight', default=None,
                            help="Color points according to MBAR weights")
        parser.add_argument('--nomae', action="store_true",
                            help="To remove the MAE from the plot")
        parser.add_argument('--normse', action="store_true",
                            help="To remove rmse from the plot")
        parser.add_argument('--nor2', action="store_true",
                            help="to remove the r^2 from the plot")
        parser.add_argument("--figsize", default="10",
                            help="Size of the figure for matplotlib")
        parser.add_argument("--cmap", default="inferno",
                            help="Colormap for the density plot")
        parser.add_argument("--size", default=10,
                            help="Size of the marker")

    @staticmethod
    def run(args, parser):
        main(args, parser)
