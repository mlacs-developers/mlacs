from ase.io import read
from ase.calculators.lammpsrun import LAMMPS

from mlacs.mlip import LammpsMlip
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
nmax = 4
lmax = 4
alpha = 2.0
dt = 1.0  # fs
friction = 0.01
scoef = 1.0
style = "snap"

parameters = None

train_conf = read("../MlacsIpi/Trajectory_1.traj", index=":")


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation---------
# Creation of the MLIP Manager
mlip = LammpsMlip(atoms,
                  rcut=rcut,
                  stress_coefficient=scoef,
                  style=style,
                  reference_potential=ref_pot,
                  fit_parameters=parameters)

for at in train_conf:
    mlip.update_matrices(at)

# Creation of the State Manager
state = LammpsState(t_start,
                    pressure,
                    t_stop,
                    nsteps=nsteps,
                    nsteps_eq=nsteps_eq,
                    logfile="mlmd.log",
                    trajfile="mlmd.traj")

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
