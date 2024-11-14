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
from mlacs import OtfMlacs
from mlacs.mlip import MliapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs.state import (NebLammpsState,
                         NebAseState,
                         LinearInterpolation,
                         StringMethodAseState)
# from mlacs.properties import CalcNeb


@pytest.fixture
def expected_folder(expected_folder_base):
    folder = expected_folder_base
    folder.append("Mliap")
    folder.pop(folder.index("Snap"))
    return expected_folder_base


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


def issame(a, b):
    return np.all(np.abs(a - b) < 1e-8)


@pytest.mark.slow
@pytest.mark.skipif(context.has_lammps_nompi(),
                    reason="Lammps needs mpi to run PAFI")
def test_mlacs_nebstate_lammps(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 3
    calc = EMT()

    mlip_params = dict(nmax=4, lmax=4)
    desc = MliapDescriptor(atoms, 4.2, mlip_params, style="so3")
    mlip = LinearPotential(desc, folder="Mliap")
    ps, cs = 'zbl 1.0 2.0', ['* * 47 47']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    mode = "rdm_memory"
    nimages = 6
    state = NebLammpsState(nebat, nimages=nimages, mode=mode)

    # RB I will add this later.
    # pair_mlip = dict(pair_style=mlip.pair_style, pair_coeff=mlip.pair_coeff)
    # func = CalcNeb(state=state, args=pair_mlip)
    # sampling = OtfMlacs(nebat[0], state, calc, mlip, func, neq=5)

    # Check that the same Atoms are used
    assert nebat[0] == state.patoms.initial
    assert nebat[-1] == state.patoms.final

    sampling = OtfMlacs(nebat[0], state, calc, dmlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms
    # Check the size of splined objects
    assert nimages == state.patoms.nreplica
    assert len(state.patoms.splE) == 1

    xi = np.linspace(0, 1, 1001)
    state.patoms.xi = xi
    assert len(state.patoms.splE) == len(xi)
    # Check the effective masses for vacancies only one atom should move
    w_m = state.patoms.masses
    assert len(w_m) == natoms
    assert np.sum(w_m) >= 1 and np.sum(w_m) < natoms


def test_mlacs_linear_ase(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 3
    calc = EMT()

    mlip_params = dict(nmax=4, lmax=4)
    desc = MliapDescriptor(atoms, 4.2, mlip_params, style="so3")
    mlip = LinearPotential(desc, folder="Mliap")
    ps, cs = 'zbl 1.0 2.0', ['* * 47 47']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    mode = "rdm_memory"
    nimages = 6
    state = LinearInterpolation(nebat, nimages=6, mode=mode, model=dmlip)

    # Check that the same Atoms are used
    assert nebat[0] == state.patoms.initial
    assert nebat[-1] == state.patoms.final

    sampling = OtfMlacs(nebat[0], state, calc, dmlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms
    # Check the size of splined objects
    assert nimages+2 == state.patoms.nreplica
    assert len(state.patoms.splE) == 1

    xi = np.linspace(0, 1, 1001)
    state.patoms.xi = xi
    assert len(state.patoms.splE) == len(xi)


@pytest.mark.slow
def test_mlacs_neb_ase(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 3
    calc = EMT()

    mlip_params = dict(nmax=4, lmax=4)
    desc = MliapDescriptor(atoms, 4.2, mlip_params, style="so3")
    mlip = LinearPotential(desc, folder="Mliap")
    ps, cs = 'zbl 1.0 2.0', ['* * 47 47']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    nimages = 6
    state = NebAseState(nebat, nimages=6, model=dmlip)

    # Check that the same Atoms are used
    assert nebat[0] == state.patoms.initial
    assert nebat[-1] == state.patoms.final

    sampling = OtfMlacs(nebat[0], state, calc, dmlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms
    # Check the size of splined objects
    assert nimages+2 == state.patoms.nreplica
    assert len(state.patoms.splE) == 1

    xi = np.linspace(0, 1, 1001)
    state.patoms.xi = xi
    assert len(state.patoms.splE) == len(xi)


@pytest.mark.slow
def test_mlacs_string_ase(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 3
    calc = EMT()

    mlip_params = dict(nmax=4, lmax=4)
    desc = MliapDescriptor(atoms, 4.2, mlip_params, style="so3")
    mlip = LinearPotential(desc, folder="Mliap")
    ps, cs = 'zbl 1.0 2.0', ['* * 47 47']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    mode = "rdm_memory"
    nimages = 6
    state = StringMethodAseState(nebat, nimages=6, mode=mode, model=dmlip,
                                 ftol=1e-1)

    # Check that the same Atoms are used
    assert nebat[0] == state.patoms.initial
    assert nebat[-1] == state.patoms.final

    sampling = OtfMlacs(nebat[0], state, calc, dmlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms
    # Check the size of splined objects
    assert nimages+2 == state.patoms.nreplica
    assert len(state.patoms.splE) == 1

    xi = np.linspace(0, 1, 1001)
    state.patoms.xi = xi
    assert len(state.patoms.splE) == len(xi)
