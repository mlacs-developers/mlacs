"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS

from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs


atoms = read("../Si.xyz")
calc = LAMMPS(pair_style="tersoff/mod",
              pair_coeff=["* * ../Si.tersoff.mod Si"])

sw = "/Users/alois/Documents/Physique/Work/Develop/mlacs/Test/Silicium/Si.sw"
ref_pot = {"pair_style": ["sw"],
           "pair_coeff": [f"* * {sw} Si"]}
ref_pot = None


# Parameters-------------------------------------------------------------------
t_start = 200  # K
t_stop = 1200
pressure = None
nconfs = 1000
nsteps = 1000
nsteps_eq = 50
neq = 5
rcut = 5.5
twojmax = 8
dt = 1.0  # fs
scoef = 1.0
desc_params = dict(twojmax=8)

train_conf = read("../MlacsIpi/Trajectory_1.traj", index=":")


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation---------
# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut, desc_params)
mlip = LinearPotential(descriptor, stress_coefficient=scoef)

mlip.update_matrices(train_conf)

# Creation of the State Manager
state = LammpsState(t_start,
                    pressure,
                    t_stop,
                    dt=dt,
                    nsteps=nsteps,
                    nsteps_eq=nsteps_eq,
                    logfile="mlmd.log",
                    trajfile="mlmd.traj")

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
