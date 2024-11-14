"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs


@pytest.fixture
def expected_folder(expected_folder_base):
    return expected_folder_base


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


def test_mlacs_vanilla(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    natoms = len(atoms)
    nstep = 5
    nconfs = 4
    nconfs_init = 1
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    state = LammpsState(300, nsteps_eq=2, nsteps=100)

    sampling = OtfMlacs(atoms, state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")

    assert len(traj) == nstep
    # Check that the first atom is the one we started with
    assert traj[0] == atoms
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms

    ml_energy = np.loadtxt(root / "MLIP-Energy_comparison.dat")
    ml_forces = np.loadtxt(root / "MLIP-Forces_comparison.dat")
    ml_stress = np.loadtxt(root / "MLIP-Stress_comparison.dat")

    assert ml_energy.shape == (nconfs + nconfs_init, 2)
    assert ml_forces.shape == ((nconfs + nconfs_init) * natoms * 3, 2)
    assert ml_stress.shape == ((nconfs + nconfs_init) * 6, 2)


def test_mlacs_several_training(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    natoms = len(atoms)
    nsteps = 2
    nconfs = 1
    nconfs_init = 5
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    state = LammpsState(300, nsteps_eq=10, nsteps=100)

    sampling = OtfMlacs(atoms, state, calc, mlip, neq=5,
                        confs_init=nconfs_init)
    sampling.run(nsteps)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    assert len(traj) == nsteps
    # Check that the first atom is the one we started with
    assert traj[0] == atoms
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms

    traintraj = read(root / "Training_configurations.traj", ":")
    assert len(traintraj) == nconfs_init

    ml_energy = np.loadtxt(root / "MLIP-Energy_comparison.dat")
    ml_forces = np.loadtxt(root / "MLIP-Forces_comparison.dat")
    ml_stress = np.loadtxt(root / "MLIP-Stress_comparison.dat")

    assert ml_energy.shape == (nconfs + nconfs_init, 2)
    assert ml_forces.shape == ((nconfs + nconfs_init) * natoms * 3, 2)
    assert ml_stress.shape == ((nconfs + nconfs_init) * 6, 2)
