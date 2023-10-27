import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state.langevin import LangevinState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K.
The true potential is the EMT as implemented in ASE.
"""

# MLACS Parameters ------------------------------------------------------------
nconfs = 50        # Numbers of final configurations, also set the end of the 
                   # simulation
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations. 
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

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lmp_serial'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP
snap_descriptor = MliapDescriptor(atoms=atoms, 
                                  rcut=rcut, 
                                  parameters=mlip_params, 
                                  model="linear", 
                                  style="snap", 
                                  alpha="1.0")

mlip = LinearPotential(descriptor=snap_descriptor, 
                       nthrow=0,
                       parameters={},
                       energy_coefficient=1.0,
                       forces_coefficient=1.0,
                       stress_coefficient=1.0,
                       mbar=None)

# Creation of the State Manager
state = LangevinState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation

sampling.run(nconfs)
