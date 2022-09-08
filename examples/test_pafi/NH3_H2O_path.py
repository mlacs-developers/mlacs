from ase import Atoms
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import PafiLammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 300  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 5
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------



calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, mlip_parameters=mlip_params)

# Creation of the State Manager
state = PafiLammpsState(temperature, neb, reaction_coordinate=0.5, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
