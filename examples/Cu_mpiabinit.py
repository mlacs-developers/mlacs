"""
Example of a MLACS simulation of a 2x2x2 supercell of Cu
The true potential is calculed using Abinit with 4 processors
The weight of a configuration is given by Multistate Bennett Acceptance Ratio

This simulation will do 5 Abinit simulation in between each MD
Having 20 available processors will start the 5 simulation in parallel
Each simulation use nproc=4 processor to do the calculation

The PBE pseudopotential of Cu to run this example can be found on Pseudo-Dojo :
http://www.pseudo-dojo.org

To run this example, you need to have :
    Additional software : Abinit
    Additional python module : netCDF4, mpi4py, pymbar

OpenMP thread can be used by setting the variable OMP_NUM_THREADS
in your environment before calling this python script.
"""
# FIXME: No longer working with mbar

import os
from ase.build import bulk

from mlacs.calc import AbinitManager
from mlacs.mlip import MliapDescriptor, LinearPotential, MbarManager
from mlacs.state import LammpsState
from mlacs import OtfMlacs


# The 2x2x2 supercell of Cu
cell_size = 2
atoms = bulk('Cu').repeat(cell_size)

# MLACS Parameters -----------------------------------------------------------
temp = 300  # K
nconfs = 10
nsteps = 150
nsteps_eq = 15
neq = 5
cell_size = 2
rcut = 4.2  # ang
dt = 0.25  # fs
damp = 100 * dt
mlip_params = {"twojmax": 4}

# Abinit Manager  ----------------------------------------------------------
ha2ev = 27.2114  # ase takes eV as the unit of energy

nproc = 4  # Each Abinit will run on 4 processors

# Dictionnary of Abinit Input
variables = dict(
    ixc=11,  # Important to explicitly state ixc
    ecut=20*ha2ev,
    tsmear=0.01*ha2ev,
    occopt=3,
    nband=82,
    ngkpt=[2, 2, 2],
    shiftk=[0, 0, 0],
    toldfe=1e-7,
    autoparal=1,
    nsym=1)

pseudos = {"Cu": "Cu.psp8"}

# Creation of the Abinit Calc Manager
calc = AbinitManager(parameters=variables,
                     pseudos=pseudos,
                     abinit_cmd="abinit",
                     mpi_runner="ccc_mprun",
                     logfile="abinit.log",
                     errfile="abinit.err",
                     nproc=nproc)

calc.ncfile = None  # Small hack to fix an error

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
snap_descriptor = MliapDescriptor(atoms=atoms,
                                  rcut=rcut,
                                  parameters=mlip_params,
                                  model="linear",
                                  style="snap",
                                  alpha="1.0")

mbar_params = dict(mode="train", solver="L-BFGS-B")
mbar_manager = MbarManager(parameters=mbar_params)

mlip = LinearPotential(descriptor=snap_descriptor,
                       parameters={},
                       mbar=mbar_manager)

# Creation of the State Manager
nsim = 5
state = []
prefix = []
for i in range(nsim):
    prefix.append("Traj{j}".format(j=len(prefix)+1))
    state.append(LammpsState(temperature=temp,
                             dt=dt,
                             damp=damp,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq))

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, prefix_output=prefix)

# Run the simulation
sampling.run(nconfs)
