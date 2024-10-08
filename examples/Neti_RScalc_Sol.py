from pathlib import Path

import numpy as np
from ase.io import read
from ase.build import make_supercell, bulk
from mlacs.ti import ReversibleScalingState, EinsteinSolidState, ThermodynamicIntegration

rootdir = Path.cwd()

# Link LAMMPS executable
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

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
                               fe_init=None,
                               phase='solid',
                               nsteps=nsteps,
                               nsteps_eq=nsteps_eq)

ti = ThermodynamicIntegration(state, ninstance=ninstance, workdir='run_Neti_RScalc_Sol')
ti.run()
#ti.state[0].ti._run_one_state(0,0)
#for st in ti.state:
#    print(st)
#    print(st.postprocess())
