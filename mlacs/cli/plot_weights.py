"""
// Copyright (C) 2022-2024 MLACS group (AC, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ..utilities.plots import plot_weights, init_rcParams, HistPlot


def main(args, parser):
    if '_HIST.nc' in args.file:
        ncname = args.file
        path = Path().absolute()
        ncpath = str(path / ncname)
        ncplot = HistPlot(ncpath=ncpath)
        boolean_no_show = args.noshow
        boolean_show = not boolean_no_show
        if args.iteration is not None:
            args.iteration = [int(x) for x in args.iteration]

        ncplot.plot_ith_weights(show=boolean_show, savename=args.save,
                                selected_iter=args.iteration)

    else:
        weights = np.loadtxt(args.file)

        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        init_rcParams()
        ax = fig.add_subplot()

        plot_weights(ax, weights)
        if args.save is not None:
            plt.savefig(args.save)
        if not args.noshow:
            plt.show()


class CLICommand:
    """
    Plot barplot of the weights. The file from which the data is loaded can be
    either
        MLIP.weights (only last MLACS iteration),
        *HIST.nc file (any MLACS iteration can be given, or a list of MLACS
                       iteration; if no iteration is given, then the last one
                       if shown by default).

    Examples:

        $ mlacs plot_weights MLIP.weights
        $ mlacs plot_weights example_HIST.nc
        $ mlacs plot_weights example_HIST.nc -i 2 5 10
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="file with the data")
        parser.add_argument("-s", "--save", default=None,
                            help="Name of the file to save the plot. "
                                 "Default None")
        parser.add_argument("--noshow", action="store_true",
                            help="To disable the visualisation of the plot")
        parser.add_argument('-i', '--iteration', nargs='+', default=None,
                            help="List of MLACS iteration index ")

    @staticmethod
    def run(args, parser):
        main(args, parser)
