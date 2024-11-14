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
from mlacs.state import OptimizeLammpsState
from mlacs.mlip import SnapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs import MlMinimizer


@pytest.fixture
def files_with_prefix():
    files = []
    ptype = ['iso', 'iso', 'iso', 'aniso', 'aniso']
    press = [None, None, 0.0, 0.0, 5.0]
    algo = ['cg', 'fire', 'cg', 'cg', 'cg']
    for t, p, a in zip(ptype, press, algo):
        files.append(f'{a}_{p}_{t}.traj')
        files.append(f'{a}_{p}_{t}_potential.dat')
    return files


@pytest.fixture
def expected_folder():
    folder = ["MolecularDynamics", "Snap"]
    return folder


@pytest.fixture
def expected_files(files_with_prefix):
    files = ["MLACS.log", "MLACS.log0001", "MLACS.log0002", "MLACS.log0003",
             "MLACS.log0004", "Training_configurations.traj",
             "MLIP-Energy_comparison.dat", "MLIP-Forces_comparison.dat",
             "MLIP-Stress_comparison.dat"]
    files.extend(files_with_prefix)
    return files


@pytest.mark.slow
def test_mlacs_optimize(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    atoms.pop(0)
    natoms = len(atoms)
    nstep = 10
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")
    ps, cs = 'zbl 1.0 2.0', ['* * 29 29']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    etol = 0.01
    ftol = 0.01
    stol = 0.01

    prefix = []
    ptype = ['iso', 'iso', 'iso', 'aniso', 'aniso']
    press = [None, None, 0.0, 0.0, 5.0]
    algo = ['cg', 'fire', 'cg', 'cg', 'cg']
    for t, p, a in zip(ptype, press, algo):
        prefix.append(f'{a}_{p}_{t}')
        state = OptimizeLammpsState(min_style=a, pressure=p, ptype=t,
                                    prefix=prefix[-1])
        sampling = MlMinimizer(atoms, state, calc, dmlip, etol, ftol, stol)
        sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    for p in prefix:
        traj = read(root / f"{p}.traj", ":")
        # Check that the first atom is the one we started with
        assert traj[0] == atoms
        # Check that the system didn't change in the process
        for at in traj:
            assert len(at) == natoms
        # Check that the criterion on forces is achieved
        assert np.max(traj[-1].get_forces()) <= ftol
        # Check that volume is constant
        if 'None' in p:
            assert traj[-1].get_volume() == atoms.get_volume()
