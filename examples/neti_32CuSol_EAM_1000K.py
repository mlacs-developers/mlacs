"""
Performing a Non Equilibrium Thermodynamic Integration
for Cu liquid
A simple EAM force field is used.
"""

import os

from pathlib import Path

from ase.io import read
from ase.build import make_supercell

from mlacs.ti import EinsteinSolidState, ThermodynamicIntegration


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
temperature = 1000
nsteps = 50000
kcst = 2

# States ----------------------------------------------------------------------
state = EinsteinSolidState(atoms, pair_style, pair_coeff, temperature,
                           k=kcst, nsteps=nsteps, logfile='neti.log')

# Creation of the TI object
ti = ThermodynamicIntegration(state)

# Run the simulation
ti.run()
