"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

from ase.build import bulk

from ... import context  # noqa
from mlacs.ti import PressureScalingState


@pytest.fixture
def expected_folder(expected_folder_base):
    # return ["ThermoInt"]
    return ["ReversibleScaling"]


@pytest.fixture
def expected_files(expected_files_base):
    return []


def test_pressure_scaling(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    pair_style = "lj/cut 5.0"
    pair_coeff = ["* * 0.52031 2.29726"]
    temperature = 300
    folder = "ReversibleScaling"

    nsteps = 5

    state = PressureScalingState(atoms,
                                 pair_style,
                                 pair_coeff,
                                 temperature,
                                 phase="solid",
                                 nsteps=nsteps,
                                 nsteps_eq=nsteps,
                                 nsteps_msd=nsteps,
                                 nsteps_averagin=nsteps,
                                 folder=folder)
    state.run()
