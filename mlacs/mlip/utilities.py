from pathlib import Path
import numpy as np

from ase import Atoms
from . import TensorpotPotential, MbarManager
from .weighting_policy import WeightingPolicy

def split_dataset(confs, train_ratio=0.5, rng=None):
    """

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

def acefit_traj(traj, mlip, weights=None, initial_potential=None):
    """
    Fit an MLIP according to the trajectory

    initial_potential can be a filename (str) or a BBasisConfiguration
    """
    from pyace.basis import BBasisConfiguration
    if isinstance(weights, list):
        weights = np.array(weights)
    if not isinstance(traj[0], Atoms):
        raise ValueError("Traj must be an Ase.Trajectory")
    if not isinstance(mlip, TensorpotPotential):
        raise NotImplementedError("Only Tensorpotential are allowed for now")

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
        if isinstance(initial_potential, str):
            initial_potential = BBasisConfiguration(initial_potential)
        mlip.descriptor.bconf.set_all_coeffs(initial_potential.get_all_coeffs())

    mlip.descriptor.fit(weights=weights, atoms=atoms, subfolder=mlip.folder)
