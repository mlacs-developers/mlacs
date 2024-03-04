"""
Performing MLACS computation of free energy by thermodynamical integration.
The system is Cu is at 300 K
The descriptor is SNAP.
The true potential is from EMT as implemented in ASE.
"""
# FIXME: Example is broken. Some file doesnt gets written.

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LammpsState

from mlacs.ti import ThermodynamicIntegration
from mlacs.ti import EinsteinSolidState

from mlacs import OtfMlacs
from mlacs.properties import CalcTi

# MLACS Parameters ------------------------------------------------------------
nconfs = 20        # Numbers of final configurations.
neq = 5            # Numbers of mlacs equilibration iterations. 
nsteps = 500      # Numbers of MD steps in the production phase.
nsteps_eq = 100    # Numbers of MD steps in the equilibration phase.

# MD Parameters ---------------------------------------------------------------
temperature = 300  # Temperature of the simulation in K.
pressure = 0.0
dt = 1.5           # Integration time in fs.
damp = 100 * dt    # Damping parameter

# MLIP Parameters -------------------------------------------------------------
rcut = 6.5
mlip_params = {"twojmax": 4}

# Supercell creation ----------------------------------------------------------
cell_size = 3      # Multiplicity of the supercell, here 3x3x3.
atoms = bulk('Cu', cubic=True).repeat(cell_size)


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Calc
calc = EMT()

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms, 
                             rcut=rcut, 
                             parameters=mlip_params, 
                             model="linear", 
                             style="snap", 
                             alpha="1.0")

mlip = LinearPotential(descriptor)

neti_params = {'atoms': atoms,
               'temperature': temperature,
               'pressure': pressure,
               'dt': dt,
               'nsteps':200,
               'nsteps_eq':100,
               'nsteps_msd': 100,
               'pair_style': mlip.pair_style,
               'pair_coeff': mlip.pair_coeff}

properties = [CalcTi(neti_params, 'solid', ninstance=4, frequence=2)]

# Creation of the State Manager
state = LammpsState(temperature, nsteps=nsteps, nsteps_eq=nsteps_eq, dt=dt, damp=damp)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms, state, calc, mlip, properties, neq=neq)

# Run the simulation
sampling.run(nconfs)
