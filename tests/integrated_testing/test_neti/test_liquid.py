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
from mlacs.ti import UFLiquidState, ThermodynamicIntegration


@pytest.fixture
def expected_folder(expected_folder_base):
    return ["ThermoInt"]


@pytest.fixture
def expected_files(expected_files_base):
    return []


def test_einstein_onestate(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    pair_style = "eam/alloy"
    potfold = (root / "reference_files").as_posix()
    pair_coeff = [f"* * {potfold}/Cu01.eam.alloy Cu"]
    temperature = 300

    nsteps = 5

    state = UFLiquidState(atoms,
                          pair_style,
                          pair_coeff,
                          temperature,
                          nsteps=nsteps,
                          nsteps_eq=nsteps,
                          nsteps_msd=nsteps,
                          nsteps_averagin=nsteps)
    ti = ThermodynamicIntegration(state)
    ti.run()


def test_einstein_twostate(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    pair_style = "eam/alloy"
    potfold = (root / "reference_files").as_posix()
    pair_coeff = [f"* * {potfold}/Cu01.eam.alloy Cu"]

    nsteps = 5

    states = []
    for t in [300, 500]:
        states.append(UFLiquidState(atoms,
                      pair_style,
                      pair_coeff,
                      t,
                      0.0,
                      nsteps=nsteps,
                      nsteps_eq=nsteps,
                      nsteps_msd=nsteps,
                      nsteps_averagin=nsteps))
    ti = ThermodynamicIntegration(states)
    ti.run()


def test_einstein_twoinstance(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    pair_style = "eam/alloy"
    potfold = (root / "reference_files").as_posix()
    pair_coeff = [f"* * {potfold}/Cu01.eam.alloy Cu"]
    temperature = 300

    nsteps = 5

    state = UFLiquidState(atoms,
                          pair_style,
                          pair_coeff,
                          temperature,
                          nsteps=nsteps,
                          nsteps_eq=nsteps,
                          nsteps_msd=nsteps,
                          nsteps_averagin=nsteps)
    ti = ThermodynamicIntegration(state, 2)
    ti.run()