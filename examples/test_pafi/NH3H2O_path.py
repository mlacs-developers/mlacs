import os
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.io import write as asewrite

from mlacs.mlip import LammpsMlip
from mlacs.state import PafiLammpsState
from mlacs.calc.abinit_manager import AbinitManager
from mlacs import OtfMlacs

initial = Atoms(numbers=np.array([8, 1, 1, 7, 1, 1, 1, 1]),
                cell=np.array([19, 9.45, 9.45])*Bohr,
                positions=np.array([[ 0.00000000000000e+00,   0.00000000000000e+00,   0.00000000000000e+00],
                                    [-7.10348053351713e-01,  -5.40272701392005e-01,   1.64595146174340e+00], 
                                    [-7.26410725481241e-01,   1.63952639289158e+00,  -5.39327838325563e-01], 
                                    [ 7.55890453154257e+00,   0.00000000000000e+00,   0.00000000000000e+00], 
                                    [ 8.21274977352101e+00,  -1.88575770800658e-01,  -1.80374359383935e+00], 
                                    [ 8.16172716793309e+00,  -1.48664754874114e+00,   1.07147471734616e+00], 
                                    [ 8.20330114285658e+00,   1.65086474968890e+00,   7.60236823259894e-01], 
                                    [ 1.94263846460644e+00,   4.28967832165041e-02,   2.96309057636469e-02]])*Bohr) 
final = Atoms(numbers=np.array([8, 1, 1, 7, 1, 1, 1, 1]),
              cell=np.array([19, 9.45, 9.45])*Bohr,
              positions=np.array([[ 0.00000000000000e+00,   0.00000000000000e+00,   0.00000000000000e+00],
                                  [-5.74665717010524e-01,  -3.59803855701426e-01,   1.71719413695318e+00], 
                                  [-6.09436677855620e-01,   1.70604475276916e+00,  -3.55835430822367e-01], 
                                  [ 7.55890453154257e+00,   0.00000000000000e+00,   0.00000000000000e+00], 
                                  [ 8.47920115825788e+00,  -2.81947139026538e-01,  -1.69489536858513e+00], 
                                  [ 7.96330592398010e+00,  -1.48683652135442e+00,   1.19222821723755e+00], 
                                  [ 8.21841895191966e+00,   1.64576248913011e+00,   8.07291003968747e-01], 
                                  [ 5.58603044880996e+00,   1.05635690828307e-01,  -2.56435836232582e-01]])*Bohr) 
neb = [initial, final]
asewrite('pos.xyz', neb, format='extxyz')

param = {"nband": 10,
         "kptopt": 0,
         "cellcharge": 1,
         "ecut": 20,
         "pawecutdg": 40,
         "toldff": "5.0d-7",
         "nstep": 50}
cwd = "/ccc/scratch/cont002/dam/bejaurom/test_otmlacs_0.0.10d3/test_example/"
pseudos = {"O": cwd + "8o_hard.paw",
           "N": cwd + "7n.paw",
           "H": cwd + "1h.paw"}


lmpcwd = '/ccc/work/cont002/dam/bejaurom/software/lammps/v03Aug22/lammps_03Aug22/src/'
os.environ["ASE_LAMMPSRUN_COMMAND"] = lmpcwd + ' lmp_mpi '
os.environ["ASE_LAMMPSREPLICA_COMMAND"] = 'mpirun -n 1 ' + lmpcwd + ' -partition 1x1 '

state = []
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=0.001))
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=0.2))
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=0.4))
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=0.6))
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=0.8))
state.append(PafiLammpsState(100, neb, dt=0.5, reaction_coordinate=1.001))

#parameters = {"method": "Ridge",
#              "gridcv": {"alpha": np.geomspace(1e-2, 1e-15, 20)}}

atom = [ initial, initial, initial, final, final, final]

calc = AbinitManager(param, pseudos, "mpirun -n 1 abinit", ninstance=2)

mlip = LammpsMlip(initial, descriptor_parameters={"twojmax": 5})

dyn = OtfMlacs(atom, state, calc, mlip=mlip, confs_init=5)
dyn.run()
