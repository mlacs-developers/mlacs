import pytest

import os

from ase.build import bulk
from ase.calculators.emt import EMT
from mlacs.mlip import DeltaLearningPotential, LinearPotential
from mlacs.mlip import MliapDescriptor

from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at several temperature
The true potential is the EMT as implemented in ASE
"""


@pytest.fixture
def expected_folder(expected_folder_base):
    return expected_folder_base


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


def test_zbl_restart(root, treelink):
    # Parameters --------------------------------------------------------------
    temperature = 300  # K
    nconfs = 2
    nsteps = 10
    nsteps_eq = 10
    neq = 1
    cell_size = 2
    rcut = 4.2
    dt = 1.5  # fs
    mlip_params = {"twojmax": 4}
    # Link LAMMPS executable --------------------------------------------------
    lmp_exe = 'lammps'
    os.environ["ASE_LAMMPSRUN_COMMAND"] = f'mpirun -n 1 {lmp_exe}'

    # Supercell creation ------------------------------------------------------
    atoms = bulk('Cu', cubic=True).repeat(cell_size)
    calc = EMT()

    # Running Initial Lammps State
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

    zbl_style = ["zbl 0.9 2.0"]
    zbl_coeff = ['1 1 29 29']
    desc = MliapDescriptor(atoms, rcut, mlip_params, style="snap")
    mlip = LinearPotential(desc, folder="Snap")
    dlpot = DeltaLearningPotential(model=mlip,
                                   pair_style=zbl_style,
                                   pair_coeff=zbl_coeff,
                                   atom_style="atomic")

    sampling = OtfMlacs(atoms, state, calc, dlpot, neq=neq)
    sampling.run(nconfs)

    # Running Restart Lammps State
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)
    desc = MliapDescriptor(atoms, rcut, mlip_params, style="snap")
    mlip = LinearPotential(desc, folder="Snap")
    dlpot = DeltaLearningPotential(model=mlip,
                                   pair_style=zbl_style,
                                   pair_coeff=zbl_coeff,
                                   atom_style="atomic")

    sampling = OtfMlacs(atoms, state, calc, dlpot, neq=neq)
    sampling.run(nconfs)
    assert sum(sampling.nconfs) == 4

# def test_mtp_restart(root, treelink):
#    pass