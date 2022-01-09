"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from mlacs.state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses

#========================================================================================================================#
#========================================================================================================================#
class CustomLammpsState(LammpsState):
    """
    State Class for running a user-designed simulation using the LAMMPS code

    Parameters
    ----------
    custom_input : :class:`str`
        input included in the LAMMPS input file to generate the MLMD dynamic.
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    interval : :class:`int` (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default ``50``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    trajinterval : :class:`int` (optional)
        Number of steps between MLMD traj writing. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to :class:`numpy.random.default_rng()`
    init_momenta : :class:`numpy.ndarray` (optional)
        If ``None``, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations. If ``None``, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 custom_input,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 workdir=None
                ):

        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                             rng,
                             init_momenta,
                             workdir
                            )
                     
        self.custom_input = custom_input


#========================================================================================================================#
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        """
        elem, Z, masses = get_elements_Z_and_masses(atoms)
        pbc             = atoms.get_pbc()

        input_string  = ""
        input_string += self.get_general_input(pbc, masses)

        input_string += self.get_interaction_input(pair_style, pair_coeff)

        input_string += "timestep      {0}\n".format(self.dt/ 1000)
        input_string += "\n"

        input_string += self.custom_input

        input_string += "\n\n"

        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)

        input_string += self.get_last_dump_input(elem, nsteps)
        input_string += "run  {0}".format(nsteps)

        with open(self.lammpsfname, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        pass
