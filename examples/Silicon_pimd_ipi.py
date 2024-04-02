"""
Example of a MLACS-PIMD simulation of silicon at 20 K
The LAMMPS potential to run the example can be found
on the nist potential repository
https://www.ctcms.nist.gov/potentials/entry/2007--Kumagai-T-Izumi-S-Hara-S-Sakai-S--Si/ # noqa

The simulations uses a SNAP potential with 2Jmax of 6 and cutoff of 5.5 angs
Only 2 beads are computed by the true potential, while the pimd is run
with 4 beads
"""
# FIXME: This example seems to get stuck at the first step.

import os

from ase.calculators.lammpsrun import LAMMPS
from ase.build import bulk

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs import OtfMlacs
from mlacs.state import IpiState


atoms = bulk("Si", cubic=True).repeat(3)
calc = LAMMPS(pair_style="tersoff/mod",
              pair_coeff=["* * ../Si.tersoff.mod Si"],
              keep_tmp_files=False)

# Parameters ------------------------------------------------------------------
temperature = 20  # K
pressure = None  # gives 0 GPa
ensemble = "npt"
nconfs = 100
nsteps = 1000
nsteps_eq = 10
neq = 5
dt = 1.0  # fs
damp = 100 * dt

# Parameters MLIP -------------------------------------------------------------
rcut = 5.5
mlip_params = {"twojmax": 6}

# Parameters PIMD -------------------------------------------------------------
paralbeads = 4
nbeads = 4
nbeads_sim = 2


# Prepare the On The Fly Machine-Learning Assisted Sampling simulation --------

# Creation of the MLIP Manager
descriptor = MliapDescriptor(atoms=atoms,
                             rcut=rcut,
                             parameters=mlip_params,
                             model="linear",
                             style="snap",
                             alpha="1.0")

mlip = LinearPotential(descriptor)

# Creation of the State Manager
state = IpiState(temperature,
                 pressure,
                 dt=dt,
                 damp=damp,
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

# Run the simulation ----------------------------------------------------------
sampling.run(nconfs)
