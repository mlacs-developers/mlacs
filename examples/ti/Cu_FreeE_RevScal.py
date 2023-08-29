"""
Example tu run a temerature sweep calculation if initial free energy is not known
NETI is performed before sweep
"""
import os

import numpy as np
from ase.io import read
from ase.build import make_supercell, bulk
from mlacs.ti import ReversibleScalingState, EinsteinSolidState, ThermodynamicIntegration

# Create the rootdir (which is the previous directory)
rootdir = os.getcwd()
#rootdir = "/".join(rootdir.split("/")[:-1])


# Link LAMMPS executable -------------------------------------------------------
lmp_exe = 'lmp_mpi'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 4 {lmp_exe}'

# System
atoms = bulk('Cu', cubic=True).repeat(3)

pair_style = "eam/alloy"
pair_coeff = [f"* * {rootdir}/Cu01.eam.alloy Cu"]


# Some parameters
nsteps = 500
nsteps_eq = 250
ninstance = 1
t_start = 1200
t_end = 1400
fe_init = -4.070765361869089
pressure = 0 
# Create a list with all the state to simulate
state = ReversibleScalingState(atoms,
                               pair_style,
                               pair_coeff,
                               t_start=t_start,
                               t_end=t_end,
                               #fe_init=None,
                               phase='solid',
                               ninstance=5,
                               nsteps=nsteps
                               )

ti = ThermodynamicIntegration(state, ninstance=ninstance)
ti.run()
