import shutil
import pytest

from pathlib import Path


@pytest.fixture(autouse=True)
def root():
    return Path()

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
