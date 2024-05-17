import pytest

from ase.io import read
from mlacs.core import PathAtoms

from ... import context  # noqa


@pytest.fixture
def test_path_atoms_img(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')
    neb = PathAtoms([at[0], at[1]])
    assert len(neb.images) == 2
    neb.images = at
    assert len(neb.images) == 8
    assert len(neb.imgE) == 8
    assert neb.imgC.shape == (8, 3, 3)
    assert neb.imgR.shape == (107, 3, 8)
    assert neb.imgE[0] == neb.imgE[-1]
    assert neb.masses[0] == 1.0
