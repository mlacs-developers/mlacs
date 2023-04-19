import os

import numpy as np
from ase.io import read
from ase.units import Hartree, kB
from ase.build import bulk

from mlacs.ti import ThermodynamicIntegration
from mlacs.ti import EinsteinSolidState

# Creation of the system of interest --------------------------------------------
atoms=bulk('Au', cubic=True).repeat(8) #2048 atoms
pair_style = "eam/alloy"
pair_coeff = "* * /home/richard/docs/test_calphy/potentials/Au.eam.alloy Au"

# Parameters --------------------------------------------------------------------
temp = 300 # K
# number of steps to go from the initial state to the final state
nsteps = 100000
# equilibration of the system before going through the path
nsteps_eq = 25000
# steps performed to get the spring constant, which is then used to compute the
# hamiltonien at each step
nsteps_msd = 50000
# number of forward and backward orocess to be performed per state
nrepeat = 5

# Link LAMMPS executable -------------------------------------------------------
lmp_exe = 'lmp_mpi'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 4 {lmp_exe}'

# Creation of the state manager ------------------------------------------------
state = EinsteinSolidState(atoms,
                           pair_style,
                           pair_coeff,
                           temp,
                           nsteps=nsteps,
                           nsteps_eq=nsteps_eq,
                           nsteps_msd=nsteps_msd,
                           suffixdir=f"T_{temp}/")

# Creation of ti object --------------------------------------------------------
ti = ThermodynamicIntegration(state,
                              ninstance = nrepeat,
                              logfile= "state.log")

# Run the simu -----------------------------------------------------------------
ti.run()
