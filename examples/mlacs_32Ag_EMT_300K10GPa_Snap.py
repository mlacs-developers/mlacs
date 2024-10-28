"""
Performing MLACS Nosé-Hoover thermostat and barostat.
The system is Al is at 300 K and 10 GPa.
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs

workdir = os.path.basename(__file__).split('.')[0]

# MLACS Parameters ------------------------------------------------------------
nconfs = 20        # Numbers of final configurations.
neq = 5            # Numbers of mlacs equilibration iterations.
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature in K.
pressure = 10      # Pressure in GPa, if pressure=None switch to NVT.
dt = 1.5           # Integration time in fs.
damp = 100 * dt    # Damping parameter
langevin = False   # Nosé–Hoover thermostat and barostat, if langevin = True
                   # switch to a langevin thermostat and barostat.
# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Ag', cubic=True).repeat(cell_size)

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
state = LammpsState(temperature,
                    pressure,
                    dt=dt,
                    damp=damp,
                    langevin=langevin,
                    nsteps=nsteps,
                    nsteps_eq=nsteps_eq)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir=workdir)

# Run the simulation
sampling.run(nconfs)
