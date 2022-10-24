from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk

from mlacs.mlip import LammpsMlip
from mlacs import OtfMlacs
from mlacs.state import IpiState

"""
Example of a MLACS-PIMD simulation of silicium at 20 K
The LAMMPS potential to run the example can be found
on the nist potential repository
https://www.ctcms.nist.gov/potentials/entry/2007--Kumagai-T-Izumi-S-Hara-S-Sakai-S--Si/ # noqa

The simulations uses a SNAP potential with 2Jmax of 6 and cutoff of 5.5 angs
Only 2 beads are computed by the true potential, while the pimd is run
with 4 beads
"""


atoms = bulk("Si", cubic=True).repeat(3)
calc = LAMMPS(pair_style="tersoff/mod",
              pair_coeff=["* * ../Si.tersoff.mod Si"])

# Parameters ------------------------------------------------------------------
temperature = 20  # K
pressure = None  # gives 0 GPa
ensemble = "npt"
nconfs = 100
nsteps = 1000
nsteps_eq = 10
neq = 5
dt = 1.0  # fs
# Parameters MLIP -------------------------------------------------------------
style = "snap"
rcut = 5.5
twojmax = 6
friction = 0.01
scoef = 1.0
# Parameters PIMD -------------------------------------------------------------
paralbeads = 4
nbeads = 4
nbeads_sim = 2

mlip_params = {"twojmax": twojmax}

# Prepare the On The Fly Machine-Learning Assisted Sampling simulation---------

# Creation of the MLIP Manager
mlip = LammpsMlip(atoms,
                  rcut=rcut,
                  stress_coefficient=scoef,
                  style=style,
                  descriptor_parameters=mlip_params)

# Creation of the State Manager
state = IpiState(temperature,
                 pressure,
                 nsteps_eq=nsteps_eq,
                 paralbeads=paralbeads,
                 nbeads=nbeads,
                 ensemble=ensemble,
                 nsteps=nsteps)

# Creation of the OtfMLACS object
sampling = OtfMlacs(atoms,
                    state,
                    calc,
                    mlip,
                    neq=neq,
                    nbeads=nbeads_sim)

# Run the simulation
sampling.run(nconfs)
