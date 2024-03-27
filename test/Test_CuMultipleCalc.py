import os
import numpy as np

from ase.build import bulk
from ase.io import write as asewrite
from ase.calculators.emt import EMT

from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import PafiLammpsState
from mlacs.properties import CalcMfep, CalcNeb
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 10  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 30
rcut = 4.2
dt = 1  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------
atoms = bulk("Cu", cubic=True).repeat(3)
atoms.set_pbc([1, 1, 1])

neb = [atoms.copy(),
       atoms.copy()]

neb[0].pop(0)
neb[1].pop(1)

asewrite('pos.xyz', neb, format='extxyz')

lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'
cmd_replica = f'mpirun -n 7 {lmp_exe} -partition 7x1'
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = cmd_replica

calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = SnapDescriptor(neb[0], rcut, mlip_params)
mlip = LinearPotential(descriptor)

# Creation of the Property Manager
xi = np.arange(0, 1.1, 0.1)
mfep_param = {'temperature': temperature,
              'configurations': neb,
              'dt': dt,
              'pair_style': mlip.pair_style,
              'pair_coeff': mlip.pair_coeff,
              'ncpus': 8,
              'xi': xi,
              'nsteps': 2000,
              'interval': 5,
              'nthrow': 500}
cneb_param = {'temperature': temperature,
              'configurations': neb,
              'pair_style': mlip.pair_style,
              'pair_coeff': mlip.pair_coeff}

properties = [CalcMfep(mfep_param), CalcNeb(cneb_param)]

# Creation of the State Manager
state = PafiLammpsState(temperature,
                        neb,
                        reaction_coordinate=0.5,
                        dt=dt,
                        nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, properties, neq=neq)

# Run the simulation
sampling.run(nconfs)

# Run the MFEP calculation
# xi = np.arange(0, 1.1, 0.1)
# state.run_MFEP(mlip.pair_style, mlip.pair_coeff, ncpus=8, xi=xi)
