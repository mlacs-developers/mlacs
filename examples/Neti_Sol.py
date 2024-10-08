from pathlib import Path

import numpy as np
from ase.io import read
from ase.build import make_supercell
from mlacs.ti import EinsteinSolidState, ThermodynamicIntegration

rootdir = Path.cwd()

# Link LAMMPS executable 
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Load the atoms and the calculator
atoms = read(f"{rootdir}/Cu.json")

# Make the supercell
mult = [[-2, 2, 2],
        [2, -2, 2],
        [2, 2, -2]]
atoms = make_supercell(atoms, mult)
pair_style = "eam/alloy"
pair_coeff = [f"* * {rootdir}/Cu01.eam.alloy Cu"]


# Some parameters
temperature = 1000
nsteps = 50000

# Create a list with all the state to simulate
state = EinsteinSolidState(atoms,
                           pair_style,
                           pair_coeff,
                           temperature,
                           pressure=None,
                           k=2,
                           nsteps=nsteps,
                           logfile='neti.log')

ti = ThermodynamicIntegration(state, ninstance=1)
ti.run()
