"""
Performing MLACS Langevin dynamics.
The system is Cu is at 300 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state.langevin import LangevinState
from mlacs import OtfMlacs


# MLACS Parameters ------------------------------------------------------------
nconfs = 50        # Numbers of final configurations.
neq = 5            # Numbers of mlacs equilibration iterations. 
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 1.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP
descriptor = MliapDescriptor(atoms=atoms, 
                             rcut=rcut, 
                             parameters=mlip_params, 
                             model="linear", 
                             style="snap", 
                             alpha="1.0")

mlip = LinearPotential(descriptor=descriptor)

# Creation of the State Manager
state = LangevinState(temperature,
                      nsteps=nsteps,
                      nsteps_eq=nsteps_eq,
                      dt=dt,
                      friction=friction)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir='run_Langevin_Cu300K')

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
