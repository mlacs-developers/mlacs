import os

from ase.build import bulk
from ase.units import GPa, bar
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 300  # K
pressure = 1 # GPa
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 5
cell_size = 2
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------
atoms = bulk('Al', cubic=True).repeat(cell_size)
calc = EMT()

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, stress_coefficient=1.0, descriptor_parameters=mlip_params)

# Creation of the State Manager
# Langevin
state = LammpsState(temperature, pressure, langevin=True, nsteps=nsteps, nsteps_eq=nsteps_eq)
# Nosé-Hoover
#state = LammpsState(temperature, pressure, langevin=False, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
