from ase.io import write, Trajectory
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin

from mlacs.mlip import LammpsMlip
from mlacs.state import LangevinState
from mlacs import OtfMLACS


# Parameters---------------------------------------------------------------------------
temperature = [300, 1200, 2500] # K
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

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation-----------------
# Creation of the MLIP Manager
mlip = LammpsMlip(atoms, rcut=rcut, twojmax=twojmax)
# Creation of the State Manager
state  = []
prefix = []
for t in temperature:
    state.append(LangevinState(t, nsteps=nsteps, nsteps_eq=nsteps_eq))
    prefix.append("{0}K".format(t))
# Creation of the OtfMLACS object
sampling = OtfMLACS(atoms, state, calc, mlip, neq=neq, prefix_output=prefix)
# Run the simulation
sampling.run(nconfs)
