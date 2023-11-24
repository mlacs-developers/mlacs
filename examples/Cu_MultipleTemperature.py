import os

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import LangevinState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at several temperature
The true potential is the EMT as implemented in ASE
"""

# Parameters ------------------------------------------------------------------
temperature = [300, 1200, 2500]  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
nsteps_ref = 5000
neq = 5
cell_size = 2
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(cell_size)
calc = EMT()

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut, mlip_params)
mlip = LinearPotential(descriptor, stress_coefficient=1.0)

# Creation of the State Manager
state = []
prefix = []
for t in temperature:
    state.append(LangevinState(t, nsteps=nsteps, nsteps_eq=nsteps_eq))
    prefix.append(f"{t}K")

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, prefix_output=prefix)
# Run the simulation
sampling.run(nconfs)
