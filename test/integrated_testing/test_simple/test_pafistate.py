import pytest

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import NebLammpsState, PafiLammpsState
from mlacs import OtfMlacs


@pytest.fixture
def expected_folder(expected_folder_base):
    return expected_folder_base


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


@pytest.mark.skipif(context.has_lammps_nompi(),
                    reason="Lammps needs mpi to run PAFI")
def test_mlacs_pafi_vanilla(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 5
    nconfs = 4
    nconfs_init = 1
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    nimages = 6
    neb = NebLammpsState(nebat, nimages=nimages)
    state = PafiLammpsState(300, neb, nsteps_eq=2, nsteps=100)

    sampling = OtfMlacs(nebat[0], state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    assert len(traj) == nstep
    # Check that the first atom is the one we started with
    assert traj[0] == nebat[0]
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms

    ml_energy = np.loadtxt(root / "MLIP-Energy_comparison.dat")
    ml_forces = np.loadtxt(root / "MLIP-Forces_comparison.dat")
    ml_stress = np.loadtxt(root / "MLIP-Stress_comparison.dat")

    nconfs = nconfs + nconfs_init
    assert ml_energy.shape == (nconfs, 2)
    assert ml_forces.shape == (nconfs * natoms * 3, 2)
    assert ml_stress.shape == (nconfs * 6, 2)

    # Check that spline is working well
    assert state.path.spline_coordinates[0].shape == (natoms, 15)


@pytest.mark.skipif(context.has_lammps_nompi(),
                    reason="Lammps needs mpi to run PAFI")
def test_mlacs_pafi_linear(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 5
    nconfs = 4
    nconfs_init = 1
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    nimages = 6
    # This is the setup to do BlueMoon Sampling
    neb = NebLammpsState(nebat, nimages=nimages, linear=True)
    state = PafiLammpsState(300, neb, nsteps_eq=2, nsteps=100)

    sampling = OtfMlacs(nebat[0], state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    assert len(traj) == nstep
    # Check that the first atom is the one we started with

    for at in traj:
        assert len(at) == natoms

    ml_energy = np.loadtxt(root / "MLIP-Energy_comparison.dat")
    ml_forces = np.loadtxt(root / "MLIP-Forces_comparison.dat")
    ml_stress = np.loadtxt(root / "MLIP-Stress_comparison.dat")

    nconfs = nconfs + nconfs_init
    assert ml_energy.shape == (nconfs, 2)
    assert ml_forces.shape == (nconfs * natoms * 3, 2)
    assert ml_stress.shape == (nconfs * 6, 2)

    # Check that spline is working well
    assert state.path.spline_coordinates[0].shape == (natoms, 15)
    assert not np.any(state.path.spline_coordinates[0, :, -3:])