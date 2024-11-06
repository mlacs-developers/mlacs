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
import os
from ase.build import bulk

from ase.units import Hartree as Ha2eV
from mlacs.calc import AbinitManager
from mlacs.mlip import MliapDescriptor, LinearPotential, MbarManager
from mlacs.state import LammpsState
from mlacs import OtfMlacs

MPI_RUNNER = "mpirun"
nproc = 4  # Each Abinit will run on 4 processors
workdir = os.path.basename(__file__).split('.')[0]

# MLACS Parameters -----------------------------------------------------------
temp = 300  # K
nconfs = 5
neq = 2
nsteps = 50
nsteps_eq = 15

# MD Parameters ---------------------------------------------------------------
dt = 0.025  # fs
damp = 100 * dt

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2  # ang
mlip_params = {"twojmax": 4}

# Cell creation ----------------------------------------------------------
atoms = bulk('Cu', cubic=True)

# Abinit Manager  ----------------------------------------------------------
# Dictionnary of Abinit Input
variables = dict(
    ixc=11,               # Important to explicitly state ixc
    ecut=8*Ha2eV,         # Testing only
    pawecutdg=16*Ha2eV,
    tsmear=0.01*Ha2eV,
    occopt=3,
    nband=44,
    ngkpt=[1, 1, 1],      # Testing only
    shiftk=[0, 0, 0],
    istwfk=1,
    toldfe=1e-4,          # Testing only
    autoparal=1,
    nsym=1)

pseudos = {"Cu": f"{os.getcwd()}/filesforexamples/Cu.xml"}

# Creation of the Abinit Calc Manager
calc = AbinitManager(parameters=variables,
                     pseudos=pseudos,
                     abinit_cmd="abinit",
                     mpi_runner=MPI_RUNNER,
                     logfile="abinit.log",
                     errfile="abinit.err",
                     nproc=nproc)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
snap_descriptor = MliapDescriptor(atoms=atoms,
                                  rcut=rcut,
                                  parameters=mlip_params,
                                  model="linear",
                                  style="snap",
                                  alpha="1.0")

mbar = MbarManager()

mlip = LinearPotential(descriptor=snap_descriptor,
                       weight=mbar)

# Creation of the State Manager
nsim = 1
state = []
for i in range(nsim):
    state.append(LammpsState(temperature=temp,
                             dt=dt,
                             damp=damp,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             folder='Traj'))

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq, workdir=workdir)

# Run the simulation
sampling.run(nconfs)
