"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
import matplotlib.pyplot as plt
from ..utilities.plots import plot_weights, init_rcParams


def main(args, parser):
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
    """Plot barplot of the weights

    Example:

        $ mlacs plot_weights MLIP.weights
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="file with the data")
        parser.add_argument("-s", "--save", default=None,
                            help="Name of the file to save the plot. "
                                 "Default None")
        parser.add_argument("--noshow", action="store_true",
                            help="To disable the visualisation of the plot")

    @staticmethod
    def run(args, parser):
        main(args, parser)
