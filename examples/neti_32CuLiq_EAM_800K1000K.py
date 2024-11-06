"""
Performing a Non Equilibrium Thermodynamic Integration
for Cu liquid
A simple EAM force field is used.
"""

import os

from pathlib import Path

from ase.io import read
from ase.build import make_supercell

from mlacs.ti import ThermodynamicIntegration, UFLiquidState

root = Path().absolute().parents[0]
root = root / 'examples' / 'filesforexamples'
workdir = os.path.basename(__file__).split('.')[0]

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Supercell creation ----------------------------------------------------------
atoms = read(f"{root}/Cu.json")
mult = [[-2, 2, 2],
        [2, -2, 2],
        [2, 2, -2]]
atoms = make_supercell(atoms, mult)

# MLIP Parameters -------------------------------------------------------------
pair_style = "eam/alloy"
pair_coeff = [f"* * {root}/Cu01.eam.alloy Cu"]

# MD Parameters ---------------------------------------------------------------
temperatures = [800, 1000]
ntemp = len(temperatures)
nsteps = 30000

# States ----------------------------------------------------------------------
states = [UFLiquidState(atoms, pair_style, pair_coeff, t, nsteps=nsteps,
          logfile='neti.log') for t in temperatures]

# Creation of the TI object
ti = ThermodynamicIntegration(states, workdir=workdir)

# Run the simulation
ti.run()
