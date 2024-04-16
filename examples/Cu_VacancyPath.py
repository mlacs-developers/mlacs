"""
Performing MLACS pafi calculation
for vacancy diffusion.
The system is Cu is at 10 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""
# Warning: You to compile the MPI LAMMPS version with the REPLICA package to
#          run the example.

import os
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import PafiLammpsState, NebLammpsState
from mlacs import OtfMlacs


# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# MLACS Parameters ------------------------------------------------------------
nconfs = 10
nsteps = 1000
nsteps_eq = 100
neq = 30

# MD Parameters ---------------------------------------------------------------
temperature = 10  # K
dt = 1  # fs
damp = 100 * dt

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = bulk("Cu", cubic=True).repeat(3)
atoms.set_pbc([1, 1, 1])
neb = [atoms.copy(), atoms.copy()]
neb[0].pop(0)
neb[1].pop(1)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
calc = EMT()

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms,
                             rcut=rcut,
                             parameters=mlip_params,
                             model="linear",
                             style="snap",
                             alpha="1.0")

mlip = LinearPotential(descriptor)


# Creation of the State Manager
xi = np.arange(0, 1.1, 0.1)
path = NebLammpsState(neb, xi_coordinate=xi, nimages=4)
state = PafiLammpsState(temperature,
                        path=path,
                        dt=dt,
                        damp=damp,
                        nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)

# Run the MFEP calculation
state.run_pafipath_dynamics(neb[0], mlip.pair_style, mlip.pair_coeff,
                            ncpus=4, xi=xi)
