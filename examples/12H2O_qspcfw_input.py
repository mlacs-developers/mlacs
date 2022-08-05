import numpy as np

from ase.io import read
from ase.visualize import view
from ase.calculators.espresso import Espresso

from mlacs.mlip import LammpsSnap, LammpsMlip
from mlacs.state import LammpsState
from mlacs import OtfMlacs

# Configuration initiale :
atom = read('startcoord_withcharges.lmp', format='lammps-data')
z = np.array(12*[8,1,1])
atom.set_atomic_numbers(z)
q = np.array(12*[-0.84, 0.42, 0.42]) 
atom.set_initial_charges(q)
#view(atom)

# Set calculator :
input_data = {
                'prefix'            : 'mlacs_12h2o',
                'tprnfor'           : True,
                'tstress'           : True, # print the stress tensor
                'etot_conv_thr'     : 5e-5,
                'forc_conv_thr'     : 5e-4,
                'ecutwfc'           : 80.0,
                'vdw_corr'          :'grimme-d2',
                'electron_maxstep'  : 100,
                'conv_thr'          : 1e-8,
                }
pseudopotentials = { 'O': 'O.blyp.UPF',
                     'H': 'H.blyp2.UPF'}
calc = Espresso(input_data=input_data, pseudopotentials=pseudopotentials, kpts=None)

# MLACS Parametres :
temperature    = 300 # K
pressure       = None # GPa
nconfs         = 300
nsteps         = 100
nsteps_eq      = 10
neq            = 50
rcut           = 3
dt             = 1.0 # fs
#friction       = 0.01
ecoeff      = 1.0
fcoeff      = 1.0
scoeff      = 0.0  # 1.0
welems      = [0.5 , 0.5]  # weight of the different elemetns in the fit; default: w_i=Z_i / Sum_i Z_i.
style          = "snap"
mlip_params = {"twojmax": 4}
fit_dielectric = False

bonds = np.array(
[[ 1,  1,  1,  3],
 [ 2,  1,  1,  2],
 [ 3,  1,  4,  6],
 [ 4,  1,  4,  5],
 [ 5,  1,  7,  8],
 [ 6,  1,  7,  9],
 [ 7,  1, 10, 11],
 [ 8,  1, 10, 12],
 [ 9,  1, 13, 14],
 [10,  1, 13, 15],
 [11,  1, 16, 17],
 [12,  1, 16, 18],
 [13,  1, 19, 21],
 [14,  1, 19, 20],
 [15,  1, 22, 24],
 [16,  1, 22, 23],
 [17,  1, 25, 27],
 [18,  1, 25, 26],
 [19,  1, 28, 30],
 [20,  1, 28, 29],
 [21,  1, 31, 33],
 [22,  1, 31, 32],
 [23,  1, 34, 35],
 [24,  1, 34, 36]]).astype(str) 

angles = np.array(
[[ 1,  1,  2,  1,  3],
 [ 2,  1,  5,  4,  6],
 [ 3,  1,  8,  7,  9],
 [ 4,  1, 11, 10, 12],
 [ 5,  1, 14, 13, 15],
 [ 6,  1, 17, 16, 18],
 [ 7,  1, 20, 19, 21],
 [ 8,  1, 23, 22, 24],
 [ 9,  1, 26, 25, 27],
 [10,  1, 29, 28, 30],
 [11,  1, 32, 31, 33],
 [12,  1, 35, 34, 36]]).astype(str)


ref_pot = {
	"atom_style": "full",
	"bond_style": ["harmonic"],
	"bond_coeff": [["1 harmonic 22.965 1.00"]],
	"angle_style": ["harmonic"],
	"angle_coeff": [["1 harmonic 1.6456800 112.000"]],
	"pair_style" : ["lj/cut/coul/long 3.5"],
	"pair_coeff" : [["2 2 0.00674 3.165492"]],
	"model_post" : ["special_bonds lj 0.  0.  0.  coul 0.  0.  0.0\n", "kspace_style ewald 1e-6\n"]
    "bonds": bonds,
    "angles": angles
	}

mlip = LammpsMlip(atom, rcut=rcut, mlip_parameters=mlip_params, stress_coefficient=scoeff, reference_potential=ref_pot, style=style, fit_dielectric=fit_dielectric)
state = LammpsState(temperature, pressure=pressure,langevin=False, nsteps=nsteps, nsteps_eq=nsteps_eq, logfile="mlmd.log", trajfile="mlmd.traj")
sampling = OtfMlacs(atom, state, calc, mlip, neq=neq)
sampling.run(nconfs)
