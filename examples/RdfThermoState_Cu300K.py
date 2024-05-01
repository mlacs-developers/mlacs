"""
Performing MLACS Langevin dynamics
with computation of the radial distribution function.
The system is Cu is at 300 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import RdfLammpsState
from mlacs import OtfMlacs


# MLACS Parameters ------------------------------------------------------------
nconfs = 20        # Numbers of final configurations.
neq = 5            # Numbers of mlacs equilibration iterations. 
nsteps = 500      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 1.5           # Integration time in fs.
damp = 100 * dt    # Damping parameter

# MLIP Parameters -------------------------------------------------------------
rcut = 6.5
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(3)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Calc
calc = EMT()

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms,
                                   rcut=rcut,
                                   parameters=mlip_params,
                                   model='linear',
                                   style='snap',
                                   alpha="1.0")

mlip = LinearPotential(descriptor=descriptor)

# Creation of the State Manager
state = RdfLammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq, rdffile='rdf.dat', dt=dt, damp=damp)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir='run_rdf_Cu')

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)

