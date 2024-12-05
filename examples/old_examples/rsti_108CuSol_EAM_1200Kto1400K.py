"""
Performing a Reversible Scaling Thermodynamic Integration
for fcc Cu
A simple EAM force field is used.
"""
# FIXME: Seems like directories are not working properly.

import os

from pathlib import Path

from ase.build import bulk
from mlacs.ti import ReversibleScalingState, ThermodynamicIntegration

root = Path().absolute().parents[0]
root = root / 'examples' / 'filesforexamples'
workdir = os.path.basename(__file__).split('.')[0]

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Supercell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(3)

# MLIP Parameters -------------------------------------------------------------
pair_style = "eam/alloy"
pair_coeff = [f"* * {root}/Cu01.eam.alloy Cu"]


# MD Parameters ---------------------------------------------------------------
nsteps = 500
nsteps_eq = 250
ninstance = 1
t_start = 1200
t_end = 1400
fe_init = -4.070765361869089

# States ----------------------------------------------------------------------
state = ReversibleScalingState(atoms, pair_style, pair_coeff, t_start=t_start,
                               t_end=t_end, fe_init=None, phase='solid',
                               nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the TI object
ti = ThermodynamicIntegration(state, ninstance=ninstance, workdir=workdir)

# Run the simulation
ti.run()
