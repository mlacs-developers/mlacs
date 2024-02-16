from pathlib import Path
import shutil

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.state import OptimizeLammpsState
from mlacs.properties import CalcExecFunction
from mlacs.mlip import SnapDescriptor, LinearPotential, DeltaLearningPotential
from mlacs import OtfMlacs


def test_mlacs_optimize():
    root = Path()
    expected_folder = ["MolecularDynamics",
                       "Properties",
                       "Snap"]

    expected_files = ["MLACS.log",
                      "MLACS.log0001",
                      "MLACS.log0002",
                      "MLACS.log0003",
                      "MLACS.log0004",
                      "Training_configurations.traj",
                      "MLIP-Energy_comparison.dat",
                      "MLIP-Forces_comparison.dat",
                      "MLIP-Stress_comparison.dat"]

    for folder in expected_folder:
        if (root/folder).exists():
            shutil.rmtree(root / folder)

    for f in expected_files:
        if (root/f).exists():
            (root / f).unlink()


    atoms = bulk("Cu", cubic=True).repeat(2)
    atoms.pop(0)
    natoms = len(atoms)
    nstep = 4
    nconfs = 3
    nconfs_init = 1
    calc = EMT()

    mlip_params = dict(twojmax=4)
    desc = SnapDescriptor(atoms, 4.2, mlip_params)
    mlip = LinearPotential(desc, folder="Snap")
    ps, cs = 'zbl 1.0 2.0', ['* * 29 29']
    dmlip = DeltaLearningPotential(mlip, pair_style=ps, pair_coeff=cs)

    ftol = 0.005
    func = CalcExecFunction('get_forces', criterion=ftol, frequence=1)

    prefix = []
    ptype = ['iso', 'iso', 'iso', 'aniso', 'aniso']
    press = [None, None, 0.0, 0.0, 10.0]
    algo = ['cg', 'fire', 'cg', 'cg', 'cg']
    for t, p, a in zip(ptype, press, algo):
        prefix.append(f'{a}_{p}_{t}')
        state = OptimizeLammpsState(min_style=a, pressure=p, ptype=t) 
        sampling = OtfMlacs(atoms, state, calc, mlip, func, neq=5,
                            prefix_output=prefix[-1])
        expected_files.extend([f'{prefix[-1]}.traj',
                               f'{prefix[-1]}_potential.dat'])
        sampling.run(nstep)

    for folder in expected_folder:
        assert (root / folder).exists()

    for file in expected_files:
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

    for folder in expected_folder:
        shutil.rmtree(root / folder)

    for f in expected_files:
        (root / f).unlink()
