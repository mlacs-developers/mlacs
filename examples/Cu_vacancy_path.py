import os
import numpy as np

from ase.build import bulk
from ase.io import write as asewrite
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS

from mlacs.mlip import LammpsMlip
from mlacs.state import PafiLammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 10  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 30
rcut = 4.2
dt = 1  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------
atoms = bulk("Cu", cubic=True).repeat(3)
#potentiel = "/home/bejaudr/software/Mlacs/otf_mlacs/examples/test_pafi/Si_vacancy/Si.tersoff.mod" 
#calc = LAMMPS(pair_style="tersoff/mod",
#              pair_coeff=[f"* * {potentiel} Si"])

neb = [atoms.copy(),
       atoms.copy()]

neb[0].pop(0)
neb[1].pop(1)

asewrite('pos.xyz', neb, format='extxyz')

os.environ["ASE_LAMMPSRUN_COMMAND"] = 'lammps '
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = 'mpirun -n 7 lammps -partition 7x1 '

calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
mlip = LammpsMlip(neb[0], rcut=rcut, descriptor_parameters=mlip_params)

# Creation of the State Manager
state = PafiLammpsState(temperature, 
                        neb, 
                        reaction_coordinate=0.5, 
                        dt=dt, 
                        nsteps=nsteps, 
                        nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
