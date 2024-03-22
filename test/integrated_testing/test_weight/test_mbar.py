import pytest

import os
import numpy as np

from ase.build import bulk

from ... import context  # noqa

from mlacs.calc import DatabaseCalc
from mlacs.mlip import MliapDescriptor, LinearPotential, MbarManager
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Test of DatabaseCalc and of Non-Regression for Mbar
"""


@pytest.fixture
def expected_folder(expected_folder_base):
    return ["Mliap", "MolecularDynamics"]


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base

@pytest.mark.full
def test_mbar_databasecalc(root, treelink):
    ref = root.absolute() / "reference_files"
    # Parameters --------------------------------------------------------------
    temperature = 300  # K
    nconfs = 5
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

    # Weight
    mbar_params = dict(mode="train", solver="L-BFGS-B")
    mbar = MbarManager(parameters=mbar_params)

    # Running DatabaseCalc
    db_calc = DatabaseCalc(trajfile = ref / "Database.traj",
                           trainfile = ref / "Training_Database.traj")
    state = LammpsState(temperature, dt=dt, nsteps=nsteps, nsteps_eq=nsteps_eq)
    desc = MliapDescriptor(atoms, rcut, mlip_params, style="snap")
    mlip = LinearPotential(desc, folder="Mliap", weight=mbar)
    sampling = OtfMlacs(atoms, state, db_calc, mlip, neq=neq)
    sampling.run(nconfs)

    # Make sure the weight computation did not change
    precomputed_weight = [0.2456848,  0.13181529, 0.15127496,
                          0.20821541, 0.26300954]
    w = sampling.mlip.weight.weight
    assert np.allclose(precomputed_weight, w, atol=1e-5), \
        "It seems mbar evaluate weight differently than before."
