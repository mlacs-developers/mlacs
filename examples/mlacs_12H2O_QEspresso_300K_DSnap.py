"""
Performing MLACS NVT sampling liquid water at room temperature.
The descriptor is SNAP.
The true potential is optain from Quantum Espresso via the ASE interface.
A Lennard-Jones potential is used as reference for Hydrogen interactions.
"""
# FIXME: Need to be tested with quantum espresso

import os
import numpy as np

from ase.build import molecule
from ase.calculators.espresso import Espresso

from mlacs import OtfMlacs
from mlacs.mlip import MliapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.state import LammpsState

# Environment -----------------------------------------------------------------
lmp_exe = 'lmp'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# MLACS Parameters ------------------------------------------------------------
nconfs = 300
nsteps = 100
nsteps_eq = 10
neq = 50

# MD Parameters ---------------------------------------------------------------
temperature = 300  # K
pressure = None  # GPa
dt = 0.5  # fs
logfile = "mlmd.log"
trajfile = "mlmd.traj"

# MLIP Parameters -------------------------------------------------------------
rcut = 3.0
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
atoms = molecule('H2O', vacuum=1.0).repeat(4)
q = np.array(32*[-0.84, 0.42, 0.42])
atoms.set_initial_charges(q)

# Potential -------------------------------------------------------------------
ps = "lj/cut/coul/long 3.5"
pc = ["2 2 0.00674 3.165492"]
mp = ["special_bonds lj 0.  0.  0.  coul 0.  0.  0.0\n",
      "kspace_style ewald 1e-6\n"]

# DFT Parameters --------------------------------------------------------------
input_data = {'prefix': 'mlacs_12h2o',
              'tprnfor': True,
              'tstress': True,
              'etot_conv_thr': 5e-5,
              'forc_conv_thr': 5e-4,
              'ecutwfc': 80.0,
              'vdw_corr': 'grimme-d2',
              'electron_maxstep': 100,
              'conv_thr': 1e-8}
pseudopotentials = {'O': 'O.blyp.UPF',
                    'H': 'H.blyp2.UPF'}

# MLIP Potential
descriptor = MliapDescriptor(atoms=atoms,
                             rcut=rcut,
                             parameters=mlip_params,
                             model="linear",
                             style="snap")

mlip = LinearPotential(descriptor=descriptor)

# L-J + MLIP
dmlip = DeltaLearningPotential(model=mlip,
                               pair_style=ps,
                               pair_coeff=pc,
                               model_post=mp,
                               atom_style="full")

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the State Manager
state = LammpsState(temperature,
                    pressure=pressure,
                    dt=dt,
                    langevin=False,
                    nsteps=nsteps,
                    nsteps_eq=nsteps_eq,
                    logfile=logfile,
                    trajfile=trajfile)

# Creation of the Calculator Manager
calc = Espresso(input_data=input_data,
                pseudopotentials=pseudopotentials,
                kpts=None)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, dmlip, neq=neq)

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
