import numpy as np

from ase.units import fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlacs.state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses

#========================================================================================================================#
#========================================================================================================================#
class CustomLammpsState(LammpsState):
    """
    State Class for running a user-designed simulation using the LAMMPS code

    Parameters
    ----------
    custom_input : str
        input included in the LAMMPS input file to generate the MLMD dynamic.
    dt : float (optional)
        Timestep, in fs. Default 1.5 fs.
    nsteps : int (optional)
        Number of MLMD steps for production runs. Default 1000 steps.
    nsteps_eq : int (optional)
        Number of MLMD steps for equilibration runs. Default 100 steps.
    fixcm : bool (optional)
        Fix position and momentum center of mass. Default True.
    logfile : str (optional)
        Name of the file for logging the MLMD trajectory.
        If none, no log file is created. Default None.
    trajfile : str (optional)
        Name of the file for saving the MLMD trajectory.
        If none, no traj file is created. Default None.
    interval : int (optional)
        Number of steps between log and traj writing. Override
        loginterval and trajinterval. Default 50
    loginterval : int (optional)
        Number of steps between MLMD logging. Default 50.
    trajinterval : int (optional)
        Number of steps between MLMD traj writing. Default 50.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat. 
        Default correspond to numpy.random.default_rng()
    init_momenta : array (optional)
        If None, velocities are initialized with a Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    workdir : str (optional)
        Working directory for the LAMMPS MLMD simulations. If none, a LammpsMLMD
        directory is created
    """
    def __init__(self,
                 custom_input,
                 dt=1.5*fs,
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

        input_string  = "# LAMMPS input file to run a MLMD simulation\n"
        input_string += "units      metal\n"

        pbc = atoms.get_pbc()
        input_string += "boundary     {0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += "atom_style   atomic\n"
        input_string += "read_data    " + self.atomsfname
        input_string += "\n"

        for i, mass in enumerate(masses):
            input_string += "mass  " + str(i + 1) + "  " + str(mass) + "\n"
        input_string += "\n"


        input_string += "# Interactions\n"
        input_string += "pair_style    {0}\n".format(pair_style)
        input_string += "pair_coeff    {0}\n".format(pair_coeff)
        input_string += "\n"

        input_string += "timestep      {0}\n".format(self.dt/ (fs * 1000))
        input_string += "\n"

        input_string += self.custom_input

        if self.logfile is not None:
            input_string += self.get_log_in()
        if self.trajfile is not None:
            input_string += self.get_traj_in(elem)


        input_string += "# Dump last step\n"
        input_string += "dump last all custom {0} ".format(nsteps) + self.workdir + "configurations.out  id type xu yu zu vx vy vz fx fy fz element\n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
        input_string += "\n"

        input_string += "run  {0}".format(nsteps)


        with open(self.lammpsfname, "w") as f:
            f.write(input_string)


#========================================================================================================================#
    def initialize_momenta(self, atoms):
        """
        """
        pass
