"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import LangevinState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at several temperature
The true potential is the EMT as implemented in ASE
"""


@pytest.fixture
def expected_folder(expected_folder_base):
    return ["Snap"]


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


def test_langevin_mliap_restart(root, treelink):
    # Parameters --------------------------------------------------------------
    temperature = 300  # K
    nconfs = 2
    nsteps = 10
    nsteps_eq = 10
    neq = 1
    cell_size = 2
    rcut = 4.2
    dt = 1.5  # fs
    friction = 0.01
    mlip_params = {"twojmax": 4}

    # Supercell creation  -----------------------------------------------------
    atoms = bulk('Cu', cubic=True).repeat(cell_size)
    calc = EMT()

    # Prepare the On The Fly Machine-Learning Assisted Sampling simulation ----
    # Creation of the MLIP Manager
    mlip_params = dict(twojmax=4)

    # Running Initial Langevin State
    tmp_state = LangevinState(temperature, dt=dt, nsteps=nsteps,
                              nsteps_eq=nsteps_eq, friction=friction)
    tmp_desc = MliapDescriptor(atoms, rcut, mlip_params, style="snap")
    tmp_mlip = LinearPotential(tmp_desc, folder="Snap")
    sampling = OtfMlacs(atoms, tmp_state, calc, tmp_mlip, neq=neq)
    sampling.run(nconfs)

    # Running Restart Langevin State
    tmp_state = LangevinState(temperature, dt=dt, nsteps=nsteps,
                              nsteps_eq=nsteps_eq, friction=friction)
    tmp_desc = MliapDescriptor(atoms, rcut, mlip_params, style="snap")
    tmp_mlip = LinearPotential(tmp_desc, folder="Snap")
    sampling = OtfMlacs(atoms, tmp_state, calc, tmp_mlip, neq=neq)
    sampling.run(nconfs)
    assert sum(sampling.nconfs) == 4
