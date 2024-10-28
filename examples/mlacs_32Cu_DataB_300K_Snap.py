"""
Training MLIP from a database.
The system is Cu is at 300 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""

import numpy as np
import os
from ase.build import bulk
from ase.calculators.emt import EMT
from mlacs.calc import DatabaseCalc

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs

workdir = os.path.basename(__file__).split('.')[0]
os.mkdir(workdir)
os.chdir(workdir)

# MLACS Parameters ------------------------------------------------------------
nconfs = 20         # Numbers of mlacs loop
neq = 5             # Numbers of mlacs equilibration iterations.
nsteps = 100        # Numbers of MD steps in the production phase.
nsteps_eq = 50      # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 0.5           # Integration time in fs.
damp = 100 * dt    # Damping parameter

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)

# Potential -------------------------------------------------------------------
# Note : We need different mlip/Mbar object to reset coefficients
desc1 = MliapDescriptor(atoms=atoms,
                        rcut=rcut,
                        parameters=mlip_params,
                        model="linear",
                        style="snap")

desc2 = MliapDescriptor(atoms=atoms,
                        rcut=rcut,
                        parameters=mlip_params,
                        model="linear",
                        style="snap")

mlip1 = LinearPotential(descriptor=desc1)

mlip2 = LinearPotential(descriptor=desc2)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the State Manager
state = []
nstate = 1
for i in range(nstate):
    state.append(LammpsState(temperature, nsteps=nsteps,
                             nsteps_eq=nsteps_eq, dt=dt, damp=damp,
                             folder='Database'))

# Create the database ---------------------------------------------------------
calc_emt = EMT()
creator = OtfMlacs(atoms, state, calc_emt, mlip1, neq=neq,
                   workdir='Creator', keep_tmp_mlip=False)
creator.run(nconfs)

# Train the MLIP again using the database -------------------------------------
calc_db = DatabaseCalc(trajfile="Creator/Database.traj",
                       trainfile="Creator/Training_configurations.traj")

# The calculator is replaced with the database
reader = OtfMlacs(atoms, state, calc_db, mlip2, neq=neq,
                  workdir='Database', keep_tmp_mlip=True)
reader.run(nconfs)

# Compare the coefficients trained on-the-faly
# against those trained from the database
assert np.allclose(creator.mlip.coefficients, reader.mlip.coefficients)
