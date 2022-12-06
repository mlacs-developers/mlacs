import os
from ase.build import bulk

from mlacs.calc import AbinitManager
from mlacs.mlip import LammpsMlip
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of a 2x2x2 supercell of Cu
The true potential is calculed using Abinit with 4 processors

The PBE pseudopotential of Cu to run this example can be found on Pseudo-Dojo :
http://www.pseudo-dojo.org
"""

# The 2x2x2 supercell of Cu
cell_size = 2
atoms = bulk('Cu').repeat(cell_size)

# MLACS Parameters -----------------------------------------------------------
temp = 300  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
nsteps_ref = 5000
neq = 5
cell_size = 2
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}

# Abinit Manager  ----------------------------------------------------------
ha2ev = 27.2114  # ase takes eV as the unit of energy

nproc = 4  # Abinit will run on 4 processors
nomp_thread = 1  # Using multiple threads is still in development

# Dictionnary of Abinit Input
variables = dict(
    ecut=20*ha2ev,
    tsmear=0.01*ha2ev,
    occopt=3,
    nband=168,
    ngkpt=[2, 2, 2],
    shiftk=[0, 0, 0],
    toldfe=1e-7,
    autoparal=1)

pseudos = {"Cu": os.getcwd() + "/Cu.psp8"}

# Creation of the Abinit Calc Manager
calc = AbinitManager(parameters=variables,
                     pseudos=pseudos,
                     abinit_cmd="abinit",
                     mpi_runner="mpirun",
                     logfile="abinit.log",
                     errfile="abinit.err",
                     nproc=nproc,
                     nomp_thread=nomp_thread)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, descriptor_parameters=mlip_params)

# Creation of the State Manager
state = LammpsState(temperature=temp, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMlacs object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
