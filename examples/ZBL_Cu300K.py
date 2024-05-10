"""
Performing MLACS Langevin dynamics.
The system is Cu is at 300 K
The descriptor is ZBL potential + SNAP.
The true potential is from EMT as implemented in ASE.
"""

import os

from ase.data import atomic_numbers
from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.state.langevin import LangevinState
from mlacs import OtfMlacs



# MLACS Parameters ------------------------------------------------------------
nconfs = 50        # Numbers of final configurations, also set the end of the 
                   # simulation
nsteps = 1000      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.
neq = 5            # Numbers of mlacs equilibration iterations. 

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 1.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.

# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)

# Potential -------------------------------------------------------------------

def generate_pair_coeff(atoms):
    """Generate the variable pair_coeff."""
    symbols = sorted(set(atoms.get_chemical_symbols()))
    z_list = [atomic_numbers[x] for x in symbols]
    pc_list = [] # i.e. [[a1,a2],[b1,b2]]
    for t1 in range(len(z_list)):
        for t2 in range(t1+1):
            pc_list.append(f"{t2+1} {t1+1} {z_list[t2]} {z_list[t1]}")
    return pc_list

zbl_style = ["zbl 0.9 2.0"] # Inner cutoff 0.9 ang, Outer cutoff 2.0 ang 
zbl_coeff = generate_pair_coeff(atoms) # zbl_coeff = ['1 1 29 29'] 
# Element 1 with Element 1 interacts as ZBL between 29 protons and 29 protons

# MLIP Potential
snap_descriptor = MliapDescriptor(atoms=atoms,
                                  rcut=rcut,
                                  parameters=mlip_params,
                                  model="linear",
                                  style="snap")

mlip = LinearPotential(descriptor=snap_descriptor)

# ZBL + MLIP
dlpot = DeltaLearningPotential(model=mlip,
                               pair_style=zbl_style,
                               pair_coeff=zbl_coeff,
                               atom_style="atomic")

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the State Manager
state = LangevinState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq,
                      dt=dt, friction=friction)

# Creation of the Calculator Manager
calc = EMT()

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, dlpot, neq=neq, 
                    workdir='run_ZBL_Cu300K')

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
