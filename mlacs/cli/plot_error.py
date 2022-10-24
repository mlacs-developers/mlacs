"""
"""
import numpy as np
import matplotlib.pyplot as plt
from ..utilities.plots import plot_error, init_rcParams


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
    figsize = (float(args.figsize), float(args.figsize))
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    init_rcParams()
    ax = fig.add_subplot()
    plot_error(ax,
               data,
               datatype=args.datatype,
               showrmse=rmse,
               showmae=mae,
               showrsquared=rsquared)
    if args.save is not None:
        plt.savefig(args.save)
    if not args.noshow:
        plt.show()


class CLICommand:
    """Plot the distribution of error for the MLIP


    Example:

        $ mlacs error MLIP-Energy_comparison.dat -datatype energy
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
        parser.add_argument('--nomae', action="store_true",
                            help="To remove the MAE from the plot")
        parser.add_argument('--normse', action="store_true",
                            help="To remove rmse from the plot")
        parser.add_argument('--nor2', action="store_true",
                            help="to remove the r^2 from the plot")
        parser.add_argument("--figsize", default="10",
                            help="Size of the figure for matplotlib")

    @staticmethod
    def run(args, parser):
        main(args, parser)
