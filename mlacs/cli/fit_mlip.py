"""
"""
from ase.io import read

from ..mlip import FitLammpsMlip
from ..mlip.linearfit_factory import (default_fit,
                                      default_snap,
                                      default_so3)

def main(args, parser):

    confs = read(args.file, index=':')
    atoms = confs[0]
    if args.id_reference is not None: 
        atoms = confs[args.id_reference]

    

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
    """
    Fit a MLIP from a Trajectory file.

    Example:

        $ mlacs fit Trajectory.traj --rcut 4.2 
    """
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('file', help="file with the data")
        parser.add_argument("--style", default="snap",
                            help="The descriptor style used."
                                 "Default snap")
        parser.add_argument("--model", default="linear",
                            help="The model for the MLIP.")
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
