"""
Tutorial on mbar that serves at prerequisite for postprocessing example.
"""

from mlacs.state import LammpsState
from mlacs.mlip import MbarManager

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs import OtfMlacs

workdir = 'run_tuto_mbar'

# MLACS Parameters ------------------------------------------------------------
nconfs = 200        # Numbers of final configurations.
neq = 5           # Numbers of mlacs equilibration iterations.
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 400  # Temperature of the simulation in K.
pressure = 100  # GPa

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 6}

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

parameters = {"solver": "L-BFGS-B"}
mbar = MbarManager(parameters=parameters)

mlip = LinearPotential(descriptor=descriptor, weight=mbar)

# Creation of the State Manager
state = list(LammpsState(temperature, nsteps=nsteps) for i in range(2))

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir=workdir)

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
