import numpy as np

from ase.io import read

from ... import context  # noqa
from mlacs.core import PathAtoms


def test_path_atoms_img(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')
    neb = PathAtoms([at[0], at[1]])
    assert len(neb.images) == 2
    assert neb.nreplica == 2
    neb.images = at
    assert len(neb.images) == 8
    assert neb.nreplica == 8
    assert np.all(np.abs(neb.imgxi - np.linspace(0, 1, 8)) < 1e-8)
    assert len(neb.imgE) == 8
    assert neb.imgC.shape == (8, 3, 3)
    assert neb.imgR.shape == (8, 107, 3)
    assert neb.imgE[0] == neb.imgE[-1]
    assert neb.masses[0] == 1.0


def test_path_atoms_xi(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')

    obj = [PathAtoms(at), PathAtoms(at, xi=0.5), PathAtoms(at, mode=0.5),
           PathAtoms(at, xi=0.8), PathAtoms(at, mode='rdm')]
    obj[-2].xi = 0.5
    obj[-1].xi = 0.5

    for o in obj:
        assert o.xi == 0.5
        assert o.splined == obj[0].splined
        assert o.splE == obj[0].splE
        assert o.splC == obj[0].splC
        assert o.splR == obj[0].splR
        assert o.splDR == obj[0].splDR
        assert o.splD2R == obj[0].splD2R

    for o in obj:
        assert o.xi == o.splined.info['reaction_coordinate']


def test_path_atoms_mode(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')

    neb = PathAtoms(at, mode='rdm')
    at = neb.splined
