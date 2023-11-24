import os
import numpy as np

from ase.build import bulk
from ase.io import write as asewrite
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS

from mlacs import OtfMlacs
from mlacs.mlip import LammpsMlip
from mlacs.state import NebLammpsState


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 10  # K
nconfs = 10
nsteps = 1000
nsteps_eq = 100
neq = 30
rcut = 4.2
dt = 1  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------
atoms = bulk("Ag", cubic=True).repeat(3)
atoms.set_pbc([1, 1, 1])

neb = [atoms.copy(),
       atoms.copy()]

neb[0].pop(0)
neb[1].pop(1)

asewrite('pos.xyz', neb, format='extxyz')

lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = f'mpirun -n 7 {lmp_exe} -partition 7x1'

calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut, mlip_params)
mlip = LinearPotential(descriptor, stress_coefficient=1.0)
              
# Creation of the State Manager
mode = 'rdm_spl'  # Sampling method along the reaction path:
                  #  - <float>: reaction coordinate
                  #  - col: search the position of the energy maximum
                  #  - rdm_spl: random, splined reaction path 
                  #  - rdm_true: random, true reaction path 

state = NebLammpsState(neb, 
                       mode=mode)

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
