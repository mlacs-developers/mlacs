import os
import numpy as np

from ase import Atoms
from ase.io import write as asewrite
from ase.io import read as aseread
from ase.units import Hartree, Bohr
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import PafiLammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at 300 K
The true potential is the EMT as implemented in ASE
"""

# Parameters-------------------------------------------------------------------
temperature = 300  # K
nconfs = 50
nsteps = 1000
nsteps_eq = 100
neq = 5
rcut = 4.2
dt = 1.5  # fs
friction = 0.01
mlip_params = {"twojmax": 4}


# Supercell creation ----------------------------------------------------------

# neb = aseread('pos.xyz')
initial = Atoms(numbers=np.array([8, 1, 1, 7, 1, 1, 1, 1]),
                cell=np.array([19, 9.45, 9.45])*Bohr,
                pbc=[1, 1, 1],
                positions=np.array([[ 0.00000000000000e+00,   0.00000000000000e+00,   0.00000000000000e+00],
                                    [-7.10348053351713e-01,  -5.40272701392005e-01,   1.64595146174340e+00], 
                                    [-7.26410725481241e-01,   1.63952639289158e+00,  -5.39327838325563e-01], 
                                    [ 7.55890453154257e+00,   0.00000000000000e+00,   0.00000000000000e+00], 
                                    [ 8.21274977352101e+00,  -1.88575770800658e-01,  -1.80374359383935e+00], 
                                    [ 8.16172716793309e+00,  -1.48664754874114e+00,   1.07147471734616e+00], 
                                    [ 8.20330114285658e+00,   1.65086474968890e+00,   7.60236823259894e-01], 
                                    [ 1.94263846460644e+00,   4.28967832165041e-02,   2.96309057636469e-02]])*Bohr+np.array([4, 2, 2])) 
final = Atoms(numbers=np.array([8, 1, 1, 7, 1, 1, 1, 1]),
              cell=np.array([19, 9.45, 9.45])*Bohr,
              pbc=[1, 1, 1],
              positions=np.array([[ 0.00000000000000e+00,   0.00000000000000e+00,   0.00000000000000e+00],
                                  [-5.74665717010524e-01,  -3.59803855701426e-01,   1.71719413695318e+00], 
                                  [-6.09436677855620e-01,   1.70604475276916e+00,  -3.55835430822367e-01], 
                                  [ 7.55890453154257e+00,   0.00000000000000e+00,   0.00000000000000e+00], 
                                  [ 8.47920115825788e+00,  -2.81947139026538e-01,  -1.69489536858513e+00], 
                                  [ 7.96330592398010e+00,  -1.48683652135442e+00,   1.19222821723755e+00], 
                                  [ 8.21841895191966e+00,   1.64576248913011e+00,   8.07291003968747e-01], 
                                  [ 5.58603044880996e+00,   1.05635690828307e-01,  -2.56435836232582e-01]])*Bohr+np.array([4, 2, 2])) 
neb = [initial, final]
asewrite('pos.xyz', neb, format='extxyz')
os.environ["ASE_LAMMPSRUN_COMMAND"] = 'lammps '
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = 'mpirun -n 7 lammps -partition 7x1 '

calc = EMT()

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
mlip = LammpsMlip(neb[0], rcut=rcut, descriptor_parameters=mlip_params)

# Creation of the State Manager
state = PafiLammpsState(temperature, neb, reaction_coordinate=0.5, nsteps=nsteps, nsteps_eq=nsteps_eq)

# Creation of the OtfMLACS object
sampling = OtfMlacs(neb[0], state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
