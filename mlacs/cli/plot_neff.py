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
    if args.iteration is not None:
        args.iteration = [int(x) for x in args.iteration]

    ncplot.plot_neff(show=boolean_show, savename=args.save,
                     selected_iter=args.iteration)


class CLICommand:
    """
    Plot evolution of number of effective configurations, from HIST file.

    Examples:

        $ mlacs plot_neff *HIST.nc
        $ mlacs plot_neff *HIST.nc  -i 3 4 5     [Plot distribution weights
                                                  of MLACS iterations i=3,4,5]
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="Full name of *HIST.nc file")
        parser.add_argument("-s", "--save", default='',
                            help="Name of the file to save the plot. "
                                 "Default None")
        parser.add_argument("--noshow", action="store_true",
                            help="To disable the visualisation of the plot")
        parser.add_argument('-i', '--iteration', nargs='+', default=None,
                            help="List of MLACS iteration index ")

    @staticmethod
    def run(args, parser):
        main(args, parser)
