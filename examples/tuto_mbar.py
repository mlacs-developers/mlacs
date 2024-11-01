"""
Tutorial on mbar that serves at prerequisite for postprocessing example.
"""

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import LinearPotential, SnapDescriptor
from mlacs import OtfMlacs
from mlacs.state import LammpsState
from mlacs.mlip import MbarManager

workdir = 'run_tuto_mbar'

# MLACS Parameters ------------------------------------------------------------
nconfs = 20        # Numbers of final configurations.
neq = 0           # Numbers of mlacs equilibration iterations.
nsteps = 100      # Numbers of MD steps in the production phase.

# MD Parameters ---------------------------------------------------------------
temperature = 400  # Temperature of the simulation in K.
pressure = 50  # GPa

# Supercell creation ----------------------------------------------------------
cell_size = 5      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)

parameters = {"twojmax": 6}
descriptor = SnapDescriptor(atoms,
                            parameters=parameters)

parameters = {"solver": "L-BFGS-B"}
mbar = MbarManager(parameters=parameters)

mlip = LinearPotential(descriptor=descriptor, weight=mbar)

# Creation of the State Manager
state = list(LammpsState(temperature, pressure, nsteps=nsteps)
             for i in range(5))

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
# ncformat='NETCDF3_CLASSIC'
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir=workdir)

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
