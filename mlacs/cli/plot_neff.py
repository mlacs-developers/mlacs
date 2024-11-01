"""
// Copyright (C) 2022-2024 MLACS group (CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path

from ..utilities.plots import HistPlot


def main(args, parser):
    ncname = args.file
    path = Path().absolute()
    ncpath = str(path / ncname)
    ncplot = HistPlot(ncpath=ncpath)
    boolean_no_show = args.noshow
    boolean_show = not boolean_no_show

    ncplot.plot_neff(show=boolean_show, savename=args.save)


class CLICommand:
    """Plot evolution of number of effective configurations, from HIST file.

    Example:

        $ mlacs plot_neff *HIST.nc file
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="Full name of *HIST.nc file")
        parser.add_argument("-s", "--save", default='',
                            help="Name of the file to save the plot. "
                                 "Default None")
        parser.add_argument("--noshow", action="store_true",
                            help="To disable the visualisation of the plot")

    @staticmethod
    def run(args, parser):
        main(args, parser)
