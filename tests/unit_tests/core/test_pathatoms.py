"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest
import numpy as np

from ase.io import read

from ... import context  # noqa
from mlacs.core import PathAtoms
from mlacs.utilities import interpolate_points as intpts


def issame(a, b):
    return np.all(np.abs(a - b) < 1e-8)


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
    assert issame(neb.imgxi, np.linspace(0, 1, 8))
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
        assert issame(o.splC, obj[0].splC)
        assert issame(o.splR, obj[0].splR)
        assert issame(o.splDR, obj[0].splDR)
        assert issame(o.splD2R, obj[0].splD2R)

    for o in obj:
        assert o.xi == o.splined.info['reaction_coordinate']


@pytest.mark.slow
def test_path_atoms_xi_array(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')

    xi = np.linspace(0, 1, 5)
    obj = [PathAtoms(at, xi=xi), PathAtoms(at, xi=xi.tolist()),
           PathAtoms(at, mode=xi), PathAtoms(at, mode=xi.tolist()),
           PathAtoms(at, xi=0.8), PathAtoms(at, mode='rdm')]
    obj[-2].xi = xi
    obj[-1].xi = xi

    for o in obj:
        assert issame(o.xi, xi)
        assert len(o.splined) == len(obj[0].splined)
        assert issame(o.splE, obj[0].splE)
        assert issame(o.splC, obj[0].splC)
        assert issame(o.splR, obj[0].splR)
        assert issame(o.splDR, obj[0].splDR)
        assert issame(o.splD2R, obj[0].splD2R)

    for o in obj:
        at_xi = np.r_[[a.info['reaction_coordinate'] for a in o.splined]]
        assert issame(o.xi, at_xi)


def test_path_atoms_mode(root):
    """
    """
    f = root / 'reference_files'
    at = read(f / 'Database_NEB_path.xyz', index=':')

    mode = ['saddle', 'rdm', 'rdm_memory', 'rdm_true', 0.8,
            np.linspace(0, 1, 20), [0.0, 0.4, 0.8, 1.0]]
    for m in mode:
        neb = PathAtoms(at, mode=m)
        spl = neb.splined

        # Test on the only atom moving.
        ref = np.r_[[intpts(neb.imgxi, neb.imgR[:, 0, i],
                     neb.xi, 0) for i in range(3)]]
        assert issame(ref, neb.splR[0])
        if not isinstance(m, (list, np.ndarray)):
            assert issame(ref, spl.positions[0])
