import re
import pytest

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT
from ase.optimize import BFGS, BFGSLineSearch
from ase.filters import UnitCellFilter
from ase.units import GPa

from ... import context  # noqa
from mlacs.state import OptimizeAseState
from mlacs.mlip import SnapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs import MlMinimizer


@pytest.fixture
def files_with_prefix():
    files = []
    fltr = [None, UnitCellFilter, UnitCellFilter]
    press = [None, 0.0, 5.0]
    algo = [None, BFGS, BFGSLineSearch]
    for f, p, a in zip(fltr, press, algo):
        if a is None:
            name = "None"
        else:
            name = a.__name__
        if f is None:
            fname = "None"
        else:
            fname = f.__name__
        files.append(f'{name}_{p}_{fname}.traj')
        files.append(f'{name}_{p}_{fname}_potential.dat')
    return files


@pytest.fixture
def expected_folder():
    folder = ["MolecularDynamics", "Snap"]
    return folder


@pytest.fixture
def expected_files(files_with_prefix):
    files = ["MLACS.log", "MLACS.log0001", "MLACS.log0002",
             "Training_configurations.traj",
             "MLIP-Energy_comparison.dat", "MLIP-Forces_comparison.dat",
             "MLIP-Stress_comparison.dat"]
    files.extend(files_with_prefix)
    return files


def test_mlacs_optimize(root, treelink):

    atoms = bulk("Cu", cubic=True).repeat(2)
    atoms.pop(0)
    natoms = len(atoms)
    nstep = 20
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
    fltr = [None, UnitCellFilter, UnitCellFilter]
    press = [None, 0.0, 5.0]
    algo = [None, BFGS, BFGSLineSearch]
    for f, p, a in zip(fltr, press, algo):
        if a is None:
            name = "None"
        else:
            name = a.__name__
        if f is None:
            fname = "None"
        else:
            fname = f.__name__
        prefix.append(f'{name}_{p}_{fname}')
        if p is not None:
            press = p * GPa
        state = OptimizeAseState(optimizer=a, filters=f,
                                 fltr_parameters=dict(scalar_pressure=press,
                                                      cell_factor=10),
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
        # Check that the pressure is consistent with the target
        else:
            pres = -traj[-1].get_stress()[:3].mean() / GPa
            ptarg = float(re.findall("[0-9]", p)[0])
            assert np.isclose(pres, ptarg, atol=1e-1)
