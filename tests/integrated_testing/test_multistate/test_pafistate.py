"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

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
def expected_files():
    files = ["MLACS.log", "Training_configurations.traj",
             "MLIP-Energy_comparison.dat", "MLIP-Forces_comparison.dat",
             "MLIP-Stress_comparison.dat", "Trajectory_1.traj",
             "Trajectory_2.traj", "Trajectory_3.traj",
             "Trajectory_1_potential.dat", "Trajectory_2_potential.dat",
             "Trajectory_3_potential.dat"]
    return files


@pytest.mark.slow
@pytest.mark.skipif(context.has_lammps_nompi(),
                    reason="Lammps needs mpi to run PAFI")
def test_mlacs_pafi_multi(root, treelink):
    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 3
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    atoms = []
    state = []
    nimages = 6
    xi = [0.01, 0.6, 1.1]
    temp = [100, 200, 300]
    for x, t in zip(xi, temp):
        neb = NebLammpsState(nebat, nimages=nimages, xi=x)
        state.append(PafiLammpsState(t, mep=neb, nsteps_eq=2, nsteps=100))
        atoms.append(nebat[0])

    sampling = OtfMlacs(atoms, state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    for i in range(1, 4):
        traj = read(root / f"Trajectory_{i}.traj", ":")
        assert len(traj) == nstep
        # Check that the first atom is the one we started with
        assert traj[0] == atoms[i-1]
        # Check that the system didn't change in the process
        for at in traj:
            assert len(at) == natoms
