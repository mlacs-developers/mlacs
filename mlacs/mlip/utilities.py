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
