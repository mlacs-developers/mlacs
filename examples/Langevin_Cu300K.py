from ase.io import write, Trajectory
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

from mlacs.mlip import LammpsMlip
from mlacs.state import LangevinState
from mlacs import OtfMlacs


# Parameters---------------------------------------------------------------------------
temperature = 300 # K
nconfs      = 50
nsteps      = 1000
nsteps_eq   = 100
nsteps_ref  = 5000
neq         = 5
cell_size   = 2
rcut        = 4.2
twojmax     = 4
dt          = 1.5 # fs
friction    = 0.01


# Supercell creation-------------------------------------------------------------------
atoms = bulk('Cu', cubic=True).repeat(cell_size)
calc  = EMT()

# Create reference trajectory----------------------------------------------------------
atoms_ref = atoms.copy()
atoms_ref.calc = calc

# Create the reference Langevin dynamics
dyn = Langevin(atoms_ref, timestep=dt, temperature_K=temperature, friction=friction)

# Create the reference Trajectory file
reftrajfile = "Reference.traj"
ref_traj = Trajectory(reftrajfile, mode="w", atoms=atoms_ref)
dyn.attach(ref_traj.write)

dyn.run(nsteps_ref)


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation-----------------
# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, twojmax=twojmax)
# Creation of the State Manager
state = LangevinState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq)
# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)
# Run the simulation
sampling.run(nconfs)
