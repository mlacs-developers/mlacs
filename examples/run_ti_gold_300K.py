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
pair_coeff = "* * /home/richard/Documents/test_calphy/potentials/Au.eam.alloy Au"

# Parameters --------------------------------------------------------------------
temp = 300 # K
nsteps = 100000 # number of steps to go from the initial state to the final state
nsteps_eq = 25000 # equilibration of the system before going through the path
nsteps_msd = 25000 # steps performed to get the spring constant, which is then used to computer the hamiltonien at each step

##need output from mlacs to compute fcorr1 and fcorr2 
#etot_dft = np.loadtxt("")
#etot_mlip = np.loadtxt("")
#weights = np.loadtxt("")
#v_diff = etot_dft - etot_mlip
#fcorr1 = np.sum(weights*v_diff)
#beta = 1 / (kB*temp)
#fcorr2 = -0.5 * beta * (np.sum(wheights*v_diff**2) - np.sum(weights*v_diff)**2)
#fcorr1 = fcorr1 / len(atoms)
#fcorr2 = fcorr2 / len(atoms)

# Link LAMMPS executable ------------------------------------------------------
lmp_exe = 'lmp_mpi'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 2 {lmp_exe}'

# Creation of the state manager -----------------------------------------------
state = EinsteinSolidState(atoms, pair_style, pair_coeff, temp, nsteps=nsteps, nsteps_eq=nsteps_eq, nsteps_msd=nsteps_msd, suffixdir="108atoms_4x4x4")

# Creation of ti object -------------------------------------------------------
ti = ThermodynamicIntegration(state)

# Run the simu ----------------------------------------------------------------
ti.run()
