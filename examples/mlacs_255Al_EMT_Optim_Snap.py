"""
Performing MLACS geometry optimization.
The system is bulk Al.
The descriptor is ZBL+SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os
from copy import copy

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.state import OptimizeLammpsState
from mlacs import OtfMlacs
from mlacs.properties import CalcExecFunction

workdir = os.path.basename(__file__).split('.')[0]
os.mkdir(workdir)
os.chdir(workdir)

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# MLACS Parameters ------------------------------------------------------------
nconfs = 10
nsteps = 10000
nsteps_eq = 1000
neq = 5

# MD Parameters ---------------------------------------------------------------
dt = 0.1

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}
pstyle = 'zbl 1.0 2.0'
cstyle = ['* * 13 13']

# Supercell creation ----------------------------------------------------------
cell_size = 4
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
workdirs = ['CG', 'FIRE', 'CG_0GPa_Iso', 'CG_0GPa_Aniso']
properties = [CalcExecFunction('get_forces',
                               criterion=0.01,
                               frequence=1)]

# Run the simulations ---------------------------------------------------------
for state, wdir in zip(states, workdirs):
    samp = OtfMlacs(atoms, state, calc, copy(dmlip), properties,
                    neq=neq, workdir=wdir)
    samp.run(nconfs)
