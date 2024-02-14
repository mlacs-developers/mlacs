from pathlib import Path
import shutil

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs


def test_mlacs_vanilla():
    root = Path()
    expected_folder = ["MolecularDynamics",
                       "Snap"]

    expected_files = ["MLACS.log",
                      "Training_configurations.traj",
                      "MLIP-Energy_comparison.dat",
                      "MLIP-Forces_comparison.dat",
                      "MLIP-Stress_comparison.dat",
                      "Trajectory.traj",
                      "Trajectory_potential.dat"]

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()


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

    for folder in expected_folder:
        assert (root / folder).exists()

    for file in expected_files:
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

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for file in expected_files:
        (root / file).unlink()

def test_mlacs_several_training():
    root = Path()

    expected_folder = ["MolecularDynamics",
                       "Snap"]

    expected_files = ["MLACS.log",
                      "Training_configurations.traj",
                      "MLIP-Energy_comparison.dat",
                      "MLIP-Forces_comparison.dat",
                      "MLIP-Stress_comparison.dat",
                      "Trajectory.traj",
                      "Trajectory_potential.dat"]

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()

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

    for folder in expected_folder:
        assert (root / folder).exists()

    for file in expected_files:
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

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for file in expected_files:
        (root / file).unlink()
