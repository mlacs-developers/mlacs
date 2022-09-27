import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import LangevinState, LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at several temperature
The true potential is the EMT as implemented in ASE
"""

# Parameters ------------------------------------------------------------------
temperature = 300 # K
nconfs = 20
nsteps = 1000
nsteps_eq = 100
nsteps_ref = 5000
neq = 5
cell_size = 2
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}

# Link LAMMPS executable ------------------------------------------------------
lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(cell_size)
calc = EMT()


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, descriptor_parameters=mlip_params)

# Creation of the State Manager
state = []
prefix = []
state.append(LangevinState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq))
prefix.append("full_langevin")
state.append(LangevinState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq))
prefix.append("restart_langevin")
state[1].restart = True
state.append(LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq))
prefix.append("full_lmpstate")
state.append(LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq))
prefix.append("restart_lmpstate")
state[3].restart = True

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, prefix_output=prefix)
# Run the simulation
sampling.run(nconfs)
