import numpy as np


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

def fit_traj(traj, mlip, weight=None):
    """
    Fit an MLIP according to the trajectory
    """
    if not isinstance(traj, ase.Trajectory):
        raise ValueError("Traj must be an Ase.Trajectory")
    if not isinstance(mlip, Tensorpotential):
        raise NotImplementedError("Only Tensorpotential are allowed for now")
    
    # Prepare the data
    atoms = [at for at in traj]
    mlip.update_matrices(atoms)
    if weight is None:
        if isinstance(mlip.weight, MbarManager):
            msg = "Use another WeightingPolicy in the mlip or give weight."
            raise ValueError(msg)
        mlip.weight.update_database(atoms)
        mlip.weight.compute_weight()
        weight = mlip.weight.get_weights()




    

