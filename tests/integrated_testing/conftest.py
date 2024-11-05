"""
// Copyright (C) 2022-2024 MLACS group (AC, CD, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import shutil
import pytest
from itertools import chain
from pathlib import Path


@pytest.fixture(autouse=True)
def root():
    return Path().absolute()


@pytest.fixture(autouse=True)
def expected_folder_base():
    folder = ["MolecularDynamics", "Snap"]
    return folder


@pytest.fixture(autouse=True)
def expected_files_base():
    files = ["MLACS.log", "Training_configurations.traj", "Trajectory.traj",
             "MLIP-Energy_comparison.dat", "MLIP-Forces_comparison.dat",
             "MLIP-Stress_comparison.dat", "Trajectory_potential.dat"]
    return files


@pytest.fixture(autouse=True)
def treelink(root, expected_folder, expected_files):

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()

    folder, files = expected_folder, expected_files
    yield dict(folder=folder, files=files)

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for f in expected_files:
        (root / f).unlink()


@pytest.fixture
def langevin_treelink(root, expected_folder, expected_files):
    if "MolecularDynamics" in expected_folder:
        expected_folder.remove("MolecularDynamics")

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()

    folder, files = expected_folder, expected_files
    yield dict(folder=folder, files=files)

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for f in expected_files:
        (root / f).unlink()


@pytest.fixture(autouse=True)
def clean_up_nc(root):
    def _clean_up():
        patterns = [root.rglob("*_WEIGHTS.nc"), root.rglob("*_HIST.nc")]
        for filename in chain(*patterns):
            if 'reference_files' in str(filename):
                continue
            filename.unlink()
    _clean_up()
    yield
    _clean_up()
