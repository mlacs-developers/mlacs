import os

from ase.build import bulk
from ase.units import GPa, bar
from ase.calculators.emt import EMT

from mlacs.mlip import LammpsMlip
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Al at 300 K and 10 GPa. 
The true potential is the EMT as implemented in ASE.
A Nosé-Hoover thermostat and barostat is used.
"""

# MLACS Parameters ------------------------------------------------------------
nconfs = 50        # Numbers of final configurations, also set the end of the 
                   # simulation
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations. 
# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature in K.
pressure = 10      # Pressure in GPa, if pressure=None switch to NVT.
dt = 1.5           # Integration time in fs.
langevin = False   # Nosé–Hoover thermostat and barostat, if langevin = True 
                   # switch to a langevin thermostat and barostat.
# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}
ecoeff = 1.0       # Weight of energy for the MLIP fitting.
fcoeff = 1.0       # Weight of forces for the MLIP fitting.
scoeff = 1.0       # Weight of stresses for the MLIP fitting, need a value
                   # if you are running a NPT simulation (default=0).

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Al', cubic=True).repeat(cell_size)

# Lammps Exe ------------------------------------------------------------------
lmp_exe = 'lammps'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = SnapDescriptor(atoms, rcut, mlip_params)
mlip = LinearPotential(descriptor, 
                       energy_coefficient=scoeff, 
                       forces_coefficient=scoeff, 
                       stress_coefficient=scoeff)

# Creation of the State Manager
state = LammpsState(temperature, 
                    pressure, 
                    langevin=langevin, 
                    nsteps=nsteps, 
                    nsteps_eq=nsteps_eq)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)

# Run the simulation
sampling.run(nconfs)
