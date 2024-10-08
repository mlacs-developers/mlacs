from pathlib import Path

import numpy as np
from ase.io import read
from ase.build import make_supercell
from mlacs.ti import ThermodynamicIntegration, UFLiquidState

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
temperatures = [800, 1000]
nsteps = 30000
ntemp = len(temperatures)

# Create a list with all the state to simulate
states = []
for t in temperatures:
    state = UFLiquidState(atoms,
                              pair_style,
                              pair_coeff,
                              t,
                              pressure=None,
                              nsteps=nsteps,
                              logfile='neti.log')
    states.append(state)

ti = ThermodynamicIntegration(states, workdir='run_Neti_Liq')
ti.run()
