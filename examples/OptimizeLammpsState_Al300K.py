import os

from ase.build import bulk
from ase.units import GPa, bar
from ase.calculators.emt import EMT

#from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.mlip import SnapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.mlip.mbar_manager import MbarManager
from mlacs.state import OptimizeLammpsState
from mlacs import OtfMlacs
from mlacs.properties import CalcExecFunction


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
nconfs = 100
nsteps = 10000
nsteps_eq = 1000
neq = 5
cell_size = 4
rcut = 4.2
dt = 0.1
friction = 0.01
mlip_params = {"twojmax": 4}
pstyle = 'zbl 1.0 2.0'
cstyle = ['* * 13 13'] 


# Supercell creation ----------------------------------------------------------
atoms = bulk('Al', cubic=True).repeat(cell_size)
atoms.pop(0)
calc = EMT()

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut, mlip_params)
mlip = LinearPotential(descriptor, stress_coefficient=1.0)
dmlip = DeltaLearningPotential(mlip, pair_style=pstyle, pair_coeff=cstyle) 

# Creation of the State Manager
# Langevin
state_cg = OptimizeLammpsState(nsteps=nsteps, 
                               nsteps_eq=nsteps_eq)
state_fire = OptimizeLammpsState(min_style='fire', 
                                 nsteps=nsteps, 
                                 nsteps_eq=nsteps_eq)
state_boxiso = OptimizeLammpsState(min_style='cg', 
                                   pressure=0.0, 
                                   nsteps=nsteps, 
                                   nsteps_eq=nsteps_eq)
state_boxaniso = OptimizeLammpsState(min_style='cg', 
                                     pressure=0.0, 
                                     ptype='aniso',
                                     nsteps=nsteps, 
                                     nsteps_eq=nsteps_eq)
states = [state_cg, state_fire, state_boxiso, state_boxaniso]
properties = [CalcExecFunction('get_forces',
                               criterion=0.001,
                               frequence=1)]

# Run the simulations
for state in states:
    samp = OtfMlacs(atoms, state, calc, dmlip, properties, neq=neq)
    samp.run(nconfs)
    os.remove('Trajectory.traj')
