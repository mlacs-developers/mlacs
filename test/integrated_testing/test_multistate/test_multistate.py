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


def test_mlacs_multistate():
    root = Path()
    expected_folder = ["MolecularDynamics",
                       "Snap"]

    expected_files = ["MLACS.log",
                      "Training_configurations.traj",
                      "MLIP-Energy_comparison.dat",
                      "MLIP-Forces_comparison.dat",
                      "MLIP-Stress_comparison.dat",
                      "Trajectory_1.traj",
                      "Trajectory_2.traj",
                      "Trajectory_3.traj",
                      "Trajectory_1_potential.dat",
                      "Trajectory_2_potential.dat",
                      "Trajectory_3_potential.dat"]

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()


    atoms = bulk("Cu", cubic=True).repeat(2)
    natoms = len(atoms)
    nstep = 4
    nconfs = 3
    nconfs_init = 1
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")

    atoms = []
    state = []
    acell = [3.300, 3.200, 3.100]
    temp = [100, 200, 300]
    press = [-1, 0, 1]
    for t, p, a in zip(temp, press, acell):
        state.append(LammpsState(t, p, nsteps_eq=2, nsteps=100))
        atoms.append(bulk("Cu", cubic=True, a=a).repeat(2))
    nstate = len(state)

    sampling = OtfMlacs(atoms, state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in expected_folder:
        assert (root / folder).exists()

    for file in expected_files:
        assert (root / file).exists()

    for i in range(1, 4):
        traj = read(root / f"Trajectory_{i}.traj", ":")
        assert len(traj) == nstep
        # Check that the first atom is the one we started with
        assert traj[0] == atoms[i-1]
        # Check that the system didn't change in the process
        for at in traj:
            assert len(at) == natoms

    ml_energy = np.loadtxt(root / "MLIP-Energy_comparison.dat")
    ml_forces = np.loadtxt(root / "MLIP-Forces_comparison.dat")
    ml_stress = np.loadtxt(root / "MLIP-Stress_comparison.dat")

    nconfs = nstate * (nconfs + nconfs_init)
    assert ml_energy.shape == (nconfs, 2)
    assert ml_forces.shape == (nconfs * natoms * 3, 2)
    assert ml_stress.shape == (nconfs * 6, 2)

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for f in expected_files:
        (root / f).unlink()