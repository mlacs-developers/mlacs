"""
Example of a MLACS simulation of Cu at 300 K.
The true potential is the EMT as implemented in ASE.
"""

import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import AceDescriptor, TensorpotPotential
from mlacs.state.lammps_state import LammpsState
from mlacs import OtfMlacs

workdir = os.path.basename(__file__).split('.')[0]

# MLACS Parameters ------------------------------------------------------------
nconfs = 10        # Numbers of final configurations, also set the end of the
                   # simulation
nsteps = 5         # Numbers of MD steps in the production phase.
nsteps_eq = 5      # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations.
# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 0.1           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.
# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lmp_serial'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP
ace_descriptor = AceDescriptor(atoms=atoms,
                               free_at_e={'Cu': 0},
                               rcut=rcut)
mlip = TensorpotPotential(descriptor=ace_descriptor, folder="ACE")

# Creation of the State Manager
state = LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir=workdir)

# Run the simulation
sampling.run(nconfs)
