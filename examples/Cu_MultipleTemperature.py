"""
Performing MLACS Langevin dynamics at multiple temperatures.
The system is Cu.
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LangevinState
from mlacs import OtfMlacs


# MLACS Parameters ------------------------------------------------------------
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 5

# MD Parameters ---------------------------------------------------------------
temperature = [300, 1200, 2500]  # K
dt = 1.5  # fs
friction = 0.01

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(2)
calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms,
                             rcut=rcut,
                             parameters=mlip_params,
                             model="linear",
                             style="snap",
                             alpha="1.0")

mlip = LinearPotential(descriptor)

# Creation of the State Manager
state = []
prefix = []
for t in temperature:
    state.append(LangevinState(t, nsteps=nsteps, nsteps_eq=nsteps_eq,
                               dt=dt, friction=friction))
    prefix.append(f"{t}K")

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, prefix_output=prefix)

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
