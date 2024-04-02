"""
Performing MLACS Nudge Elastic Band (NEB) calculation
for vacancy diffusion.
The system is Al is at 10 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""
# FIXME: Lammps crashes with the error: "Cannot use NEB with a single replica."

import os
import numpy as np

from ase.build import bulk
from ase.io import write as asewrite
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS

from mlacs import OtfMlacs
from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import NebLammpsState

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = f'mpirun -n 7 {lmp_exe} -partition 7x1'

# MLACS Parameters ------------------------------------------------------------
nconfs = 10
nsteps = 1000
nsteps_eq = 100
neq = 30

# MD Parameters ---------------------------------------------------------------
temperature = 10  # K
dt = 1  # fs

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk("Ag", cubic=True).repeat(3)
#atoms.set_pbc([1, 1, 1])

neb = [atoms.copy(), atoms.copy()]  # Initial and final configuration
neb[0].pop(0)  # Remove an atom from initial configuration
neb[1].pop(1)  # Remove a different atom from final configuration

asewrite('pos.xyz', neb, format='extxyz')

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
calc = EMT()

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms, 
                                  rcut=rcut, 
                                  parameters=mlip_params, 
                                  model="linear", 
                                  style="snap", 
                                  alpha="1.0")

mlip = LinearPotential(descriptor=descriptor)
              
# Creation of the State Manager
mode = 'rdm_spl'  # Sampling method along the reaction path:
                  #  - <float>: reaction coordinate
                  #  - col: search the position of the energy maximum
                  #  - rdm_spl: random, splined reaction path 
                  #  - rdm_true: random, true reaction path 

state = NebLammpsState(neb, 
                       mode=mode,
                       dt=dt,
                       )

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, neq=neq)

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
