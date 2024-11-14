"""
// Copyright (C) 2022-2024 MLACS group (AC, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import shutil
import numpy as np
from ase import Atoms
from . import TensorpotPotential, MomentTensorPotential, MbarManager
from . import LinearPotential, DeltaLearningPotential
from .weighting_policy import WeightingPolicy


def split_dataset(confs, train_ratio=0.5, rng=None):
    """
    Split the dataset into a train set and a test set.
    """
    if rng is None:
        rng = np.random.default_rng()
    nconfs = len(confs)

    ntrain = int(np.ceil(nconfs * train_ratio))

    allidx = np.arange(0, nconfs)
    trainset_idx = rng.choice(allidx, ntrain, replace=False)
    testset_idx = list(set(allidx) - set(trainset_idx))

    trainset = []
    for i in trainset_idx:
        trainset.append(confs[i])
    testset = []
    for i in testset_idx:
        testset.append(confs[i])

    return trainset, testset


def linfit_traj(traj, mlip):
    """
    Fit an MLIP according to the trajectory
    """
    if not isinstance(traj[0], Atoms):
        raise ValueError("Traj must be an Ase.Trajectory")
    if not (isinstance(mlip, LinearPotential)
            or isinstance(mlip, DeltaLearningPotential)):
        msg = "Only LinearPotential or DeltaLearningPotential " + \
              "are allowed for linfit_traj"
        raise NotImplementedError(msg)
    atoms = [at for at in traj]
    if mlip.subfolder:
        if mlip.subsubdir.exists():
            shutil.rmtree(mlip.subsubdir)
    mlip.subsubdir.mkdir(parents=True, exist_ok=True)

    mlip.update_matrices(atoms)
    mlip.train_mlip()


def mtpfit_traj(traj, mlip):
    """
    Fit an MLIP according to the trajectory
    """
    if not isinstance(traj[0], Atoms):
        raise ValueError("Traj must be an Ase.Trajectory")
    if not isinstance(mlip, MomentTensorPotential):
        msg = "Only MomentTensorPotential are allowed for mtpfit_traj"
        raise NotImplementedError(msg)
    atoms = [at for at in traj]
    if mlip.subfolder:
        if mlip.subsubdir.exists():
            shutil.rmtree(mlip.subsubdir)
    mlip.subsubdir.mkdir(parents=True, exist_ok=True)

    mlip.update_matrices(atoms)
    mlip._write_configurations()
    mlip._write_input()
    mlip._write_mtpfile()
    mlip._run_mlp()


def acefit_traj(traj, mlip, weights=None, initial_potential=None):
    """
    Fit an MLIP according to the trajectory
    initial_potential :
        Potential to start the fitting from.
        Useful to reconverge with a better precision on a parameter.
        Can be a filename (str) or a BBasisConfiguration
    """
    if isinstance(weights, list):
        weights = np.array(weights)
    if not isinstance(traj[0], Atoms):
        raise ValueError("Traj must be an Ase.Trajectory")
    if not isinstance(mlip, TensorpotPotential):
        msg = "Only Tensorpotential are allowed for acefit_traj"
        raise NotImplementedError(msg)

    # Prepare the data
    atoms = [at for at in traj]
    mlip.update_matrices(atoms)

    if weights is None:
        if isinstance(mlip.weight, MbarManager):
            msg = "Use another WeightingPolicy in the mlip or give weight."
            raise ValueError(msg)
        mlip.weight.compute_weight()
        weights = mlip.weight.get_weights()
    else:
        if len(atoms) == len(weights):
            we, wf, ws = WeightingPolicy(database=atoms).build_W_efs(weights)
            weights = np.append(np.append(we, wf), ws)
    if initial_potential is not None:
        mlip.descriptor.bconf.load(initial_potential)
        curr_fc = mlip.descriptor.bconf.metadata['_fit_cycles']
        mlip.descriptor.fitting['fit_cycles'] = int(curr_fc) + 1
    mlip.descriptor.redirect_logger()
    mlip.descriptor.fit(weights=weights, atoms=atoms)
