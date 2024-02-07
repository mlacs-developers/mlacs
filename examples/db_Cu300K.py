import numpy as np
import os
from ase.data import atomic_numbers
from ase.build import bulk
from ase.calculators.emt import EMT
from mlacs.calc import DatabaseCalc

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs

"""
Example of a MLACS simulation using ZBL potential + SNAP potential
Cu is at 300 K and the true potential is from EMT as implemented in ASE.
"""


def generate_pair_coeff(atoms):
    """
    Generating the variable pair_coeff. Works for multiple species
    """
    symbols = sorted(set(atoms.get_chemical_symbols()))
    z_list = [atomic_numbers[x] for x in symbols]
    pc_list = []  # i.e. [[a1,a2],[b1,b2]]
    for t1 in range(len(z_list)):
        for t2 in range(t1+1):
            pc_list.append(f"{t2+1} {t1+1} {z_list[t2]} {z_list[t1]}")
    return pc_list


# MLACS Parameters ------------------------------------------------------------
nconfs = 20         # Numbers of mlacs loop
nsteps = 100        # Numbers of MD steps in the production phase.
nsteps_eq = 50      # Numbers of MD steps in the equilibration phase.
neq = 5             # Numbers of mlacs equilibration iterations.
# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
dt = 0.5           # Integration time in fs.
friction = 0.01    # Friction coefficient for the Langevin thermostat.
# MLIP Parameters -------------------------------------------------------------
rcut = 4.2
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 2      # Multiplicity of the supercell, here 2x2x2.
atoms = bulk('Cu', cubic=True).repeat(cell_size)

# Potential -------------------------------------------------------------------
# Element 1 with Element 1 interacts as ZBL between 29 protons and 29 protons
lmp_exe = 'lmp_serial'
os.environ["ASE_LAMMPSRUN_COMMAND"] = f'{lmp_exe}'

# Note : We need different mlip/Mbar object to reset coefficients
desc1 = MliapDescriptor(atoms=atoms,
                        rcut=rcut,
                        parameters=mlip_params,
                        model="linear",
                        style="snap")

desc2 = MliapDescriptor(atoms=atoms,
                        rcut=rcut,
                        parameters=mlip_params,
                        model="linear",
                        style="snap")

mlip1 = LinearPotential(descriptor=desc1,
                        nthrow=0,
                        energy_coefficient=1.0,
                        forces_coefficient=0.0,
                        stress_coefficient=0.0,
                        mbar=None)

mlip2 = LinearPotential(descriptor=desc2,
                        nthrow=0,
                        energy_coefficient=1.0,
                        forces_coefficient=0.0,
                        stress_coefficient=0.0,
                        mbar=None)

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------
# Creation of the State Manager
state = []
nstate = 1
for i in range(nstate):
    state.append(LammpsState(temperature, nsteps=nsteps,
                             nsteps_eq=nsteps_eq, dt=dt))

# ACT 1 : Creation of a database ####
os.mkdir("Creator")
os.chdir("Creator")
calc_emt = EMT()
creator = OtfMlacs(atoms, state, calc_emt, mlip1, neq=neq,
                   keep_tmp_mlip=False, prefix_output="Database")
creator.run(nconfs)

# ACT 2 : Reading of the database ####
os.mkdir("../Database")
os.chdir("../Database")
calc_db = DatabaseCalc(trajfile="../Creator/Database.traj",
                       trainfile="../Creator/Training_configurations.traj")
reader = OtfMlacs(atoms, state, calc_db, mlip2, neq=neq, keep_tmp_mlip=True)
reader.run(nconfs)

# Testing that we indeed get the same values ####
assert np.allclose(creator.mlip.coefficients, reader.mlip.coefficients)
