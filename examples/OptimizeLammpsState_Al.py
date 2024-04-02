"""
Performing MLACS geometry optimization.
The system is bulk Al.
The descriptor is ZBL+SNAP.
The true potential is from EMT as implemented in ASE.
"""
# Note: This example takes a long time to run.

import os

from ase.build import bulk
from ase.units import GPa, bar
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.mlip.mbar_manager import MbarManager
from mlacs.state import OptimizeLammpsState
from mlacs import OtfMlacs
from mlacs.properties import CalcExecFunction


# Parameters-------------------------------------------------------------------
nconfs = 100
nsteps = 10000
nsteps_eq = 1000
neq = 5
cell_size = 4
rcut = 4.2
dt = 0.1
mlip_params = {"twojmax": 4}
pstyle = 'zbl 1.0 2.0'
cstyle = ['* * 13 13'] 


# Supercell creation ----------------------------------------------------------
atoms = bulk('Al', cubic=True).repeat(cell_size)
atoms.pop(0)
calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms,
                             rcut=rcut,
                             parameters=mlip_params,
                             model="linear",
                             style="snap",
                             alpha="1.0")
mlip = LinearPotential(descriptor)
dmlip = DeltaLearningPotential(mlip, pair_style=pstyle, pair_coeff=cstyle) 

# Creation of the State Manager
# Langevin
state_cg = OptimizeLammpsState(nsteps=nsteps, 
                               nsteps_eq=nsteps_eq,
                               dt=dt)

state_fire = OptimizeLammpsState(min_style='fire', 
                                 nsteps=nsteps, 
                                 nsteps_eq=nsteps_eq,
                                 dt=dt)

state_boxiso = OptimizeLammpsState(min_style='cg', 
                                   pressure=0.0, 
                                   nsteps=nsteps, 
                                   nsteps_eq=nsteps_eq,
                                   dt=dt)

state_boxaniso = OptimizeLammpsState(min_style='cg', 
                                     pressure=0.0, 
                                     ptype='aniso',
                                     nsteps=nsteps, 
                                     nsteps_eq=nsteps_eq,
                                     dt=dt)

states = [state_cg, state_fire, state_boxiso, state_boxaniso]
properties = [CalcExecFunction('get_forces',
                               criterion=0.001,
                               frequence=1)]

# Run the simulations ---------------------------------------------------------
for state in states:
    samp = OtfMlacs(atoms, state, calc, dmlip, properties, neq=neq)
    samp.run(nconfs)
    os.remove('Trajectory.traj')
