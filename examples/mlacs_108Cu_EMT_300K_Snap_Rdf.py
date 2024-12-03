"""
Example of a MLACS simulation of Cu at 300 K.
The true potential is the EMT as implemented in ASE.
"""
import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs
from mlacs.properties import CalcRdf

workdir = os.path.basename(__file__).split('.')[0]

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# MLACS Parameters ------------------------------------------------------------
nconfs = 10        # Numbers of final configurations, also set the end of the
                   # simulation
nsteps = 5000      # Numbers of MD steps in the production phase.
nsteps_eq = 500    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations.

# MD Parameters ---------------------------------------------------------------
temperature = 300
dt = 1.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.

# MLIP Parameters -------------------------------------------------------------
rcut = 6.5
parameters = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(3)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation
calc = EMT()

# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut=rcut, parameters=parameters)
mlip = LinearPotential(descriptor)

# Creation of the State Manager
state = LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
kwargs = {'mlip': mlip}
properties = CalcRdf(state=state, args=kwargs)
sampling = OtfMlacs(atoms, state, calc, mlip, properties,
                    neq=neq, workdir=workdir)

# Run the simulation
sampling.run(nconfs)
