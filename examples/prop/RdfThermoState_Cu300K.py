import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState, RdfLammpsState
from mlacs import OtfMlacs
from mlacs.properties import CalcRdf

"""
Example of a MLACS simulation of Cu at 300 K to converge on RDF.
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
parameters = {"twojmax": 4}

# MLACS Parameters 
nconfs = 20        # Numbers of final configurations, also set the end of the 
                   # simulation
nsteps = 5000      # Numbers of MD steps in the production phase.
nsteps_eq = 500    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations. 

# MD Parameters
dt = 1.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation

# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut=rcut, parameters=parameters)
mlip = LinearPotential(descriptor)

rdf_params = {'temperature': 300,
              'dt': 1.5,
              'nsteps':20000,
              'nsteps_eq':10000,
              'langevin': False,
              'logfile': 'mlmd.log',
              'rdffile': 'rdf_.dat',
              'pair_style': mlip.pair_style,
              'pair_coeff': mlip.pair_coeff}

properties = [CalcRdf(rdf_params, atoms)]

# Creation of the State Manager
state = LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, properties, neq=neq)

# Run the simulation
sampling.run(nconfs)
