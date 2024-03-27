import numpy as np
from ase.build import bulk

from ... import context  # noqa
from mlacs.mlip.utilities import split_dataset


def test_split_dataset():
    """

    """
    rng = np.random.default_rng()
    # First we create fake dataset
    allconfs = []
    for i in range(10):
        at = bulk("Si", cubic=True)
        at.rattle(rng=rng)
        at.info = dict(number=i)
        allconfs.append(at)

    trainset, testset = split_dataset(allconfs, rng=rng)
    # First let's check that we have the right number of configurations
    assert len(trainset) == 5
    assert len(testset) == 5

    #  Now let's check that configurations are differents
    train_idx = np.array([at.info["number"] for at in trainset])
    test_idx = np.array([at.info["number"] for at in testset])
    for i in train_idx:
        assert i not in test_idx

    # Let's set up a different train ratio
    trainset, testset = split_dataset(allconfs, 0.3, rng=rng)
    # First let's check that we have the right number of configurations
    assert len(trainset) == 3
    assert len(testset) == 7

    #  Now let's check that configurations are differents
    train_idx = np.array([at.info["number"] for at in trainset])
    test_idx = np.array([at.info["number"] for at in testset])
    for i in train_idx:
        assert i not in test_idx
