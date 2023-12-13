import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K.
The true potential is the EMT as implemented in ASE.
"""
# System
atoms = bulk('Cu', cubic=True).repeat(3)

# Calc
calc = EMT()

# State
temperature = 300
pressure = 0.0
nsteps = 500

# Lammps Exe 
lmp_exe = 'lmp_mpi'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# MLIP Parameters
rcut = 6.5
mlip_params = {"twojmax": 4}

# MLACS Parameters 
nconfs = 20        # Numbers of final configurations, also set the end of the 
                   # simulation
nsteps = 5000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations. 

# MD Parameters
dt = 1.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation

# Creation of the MLIP Manager
snap_descriptor = MliapDescriptor(atoms=atoms,
                                   rcut=rcut,
                                   parameters=mlip_params,
                                   model='linear',
                                   style='snap')
mlip = LinearPotential(descriptor=snap_descriptor,
                       energy_coefficient=1.0,
                       forces_coefficient=1.0,
                       stress_coefficient=1.0,
                       mbar=None)
# Creation of the State Manager
state = LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq, rdffile='rdf.dat')
# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip)

# Run the simulation
sampling.run(nconfs)
