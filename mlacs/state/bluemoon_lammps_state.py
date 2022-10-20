import os
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ase.units import fs, kB
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data

from mlacs.state import CustomLammpsState
from mlacs.utilities import (get_elements_Z_and_masses,
                             write_lammps_NEB_ASCIIfile,
                             _create_ASE_object)
from mlacs.utilities.io_lammps import (get_general_input,
                                       get_log_input,
                                       get_traj_input,
                                       get_interaction_input,
                                       get_last_dump_input)


# ========================================================================== #
# ========================================================================== #
class BlueMoonState(CustomLammpsState):
    """
    Class to manage PafiStates with LAMMPS

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        Default ``0.5``
    Kspring: :class:`float`
        Spring constante for the NEB calculation.
        Default ``1.0``
    maxjump: :class:`float`
        Maximum atomic jump authorized for the free energy calculations.
        Configurations with an high `maxjump` will be removed.
        Default ``0.4``
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    damp: :class:`float` or ``None``
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    BROWNian: :class:`Bool`
        If ``True``, a Brownian thermostat is used for the thermostat.
        Else, a Langevin thermostat is used
        Default ``True``
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    init_momenta : :class:`numpy.ndarray` (optional)
        If ``None``, velocities are initialized with a
        Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    prt : :class:`Bool` (optional)
        Printing options. Default ``True``
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=0.5,
                 pressure=None,
                 langevin=True,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 colvarsfile=None,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 prt=True,
                 workdir=None):

        self.temperature = temperature
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.dt = dt
        self.damp = damp
        if colvarsfile is None:
            self.colvarsfile = 'bluemoon.colvars'

        self.ispimd = False
        self.isrestart = False
        self.isappend = False


# ========================================================================== #
    def run(self):
        """
        Write the LAMMPS input for the constrained MD simulation
        """
        self.write_lammps_input_colvars()
        self.write_reference_colvars()
        colvarsfix = f'fix colvars {self.colvarsfile}'
        CustomLammpsState.__init__(self,
                                   colvarsfix, 
                                   **self.param)

# ========================================================================== #
    def write_lammps_input_colvars(self):
        """
        Write the LAMMPS input for the constrained MD simulation
        """
        with open(fname, "w") as f:
            f.write(input_string)

