"""
"""


def main(args, parser):
    import numpy as np
    import matplotlib.pyplot as plt
    from mlacs.utilities.plots import (plot_correlation,
                                       init_rcParams)
    data = np.loadtxt(args.file)
    if args.datatype not in ["energy", "forces", "stress", None]:
        raise ValueError("The type argument has to be "
                         "energy, forces or stress")
    figsize = (float(args.figsize), float(args.figsize))
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    init_rcParams()
    ax = fig.add_subplot()
    plot_correlation(ax,
                     data,
                     datatype=args.datatype,
                     rmse=args.rmse,
                     mae=args.mae)
    ax.set_aspect("equal")
    if args.save is not None:
        plt.savefig(args.save)
    if not args.noshow:
        plt.show()


class CLICommand:
    """
    Plot correlation function

    Example:

        $ mlacs correlation MLIP-Energy_comparison.dat -datatype energy
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
        parser.add_argument('--mae', default=True,
                            help="Set to true to plot the mean absolute error")
        parser.add_argument('--rmse', default=True,
                            help="Set to true to plot the root "
                                 "mean squared error")
        parser.add_argument("--figsize", default="10")

    @staticmethod
    def run(args, parser):
        main(args, parser)
