"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.atoms import Atoms
from ase.build import bulk

from ... import context  # noqa
from mlacs.utilities.miscellanous import (get_elements_Z_and_masses,
                                          create_random_structures)


def test_get_elements():
    """
    Here we test that the function get_elements_Z_and_masses() returns
    the unique elements, Z, masses and charges, ordered alphabetically
    with the elements
    """
    # First try with some charges
    elem = np.array(["Si", "Fe", "Rh", "U", "H", "H", "Si", "Fe"])
    charges = [1, 2, 0, 0, -1, -1, 1, 2]
    pos = np.zeros((len(elem), 3))
    cell = np.eye(3)

    idx = np.array([1, 4, 2, 0, 3])

    at = Atoms(elem, pos, cell=cell, charges=charges)
    ref_masses = at.get_masses()[idx]
    ref_charges = at.get_initial_charges()[idx]
    ref_elem = elem[idx]
    ref_Z = at.get_atomic_numbers()[idx]

    pred_el, pred_Z, pred_masses, pred_charges = get_elements_Z_and_masses(at)
    for ref, pred in zip(ref_elem, pred_el):
        assert ref == pred
    for ref, pred in zip(ref_Z, pred_Z):
        assert ref == pred
    for ref, pred in zip(ref_masses, pred_masses):
        assert ref == pred
    for ref, pred in zip(ref_charges, pred_charges):
        assert ref == pred

    # The same without charges
    elem = np.array(["Si", "Fe", "Rh", "U", "H", "H", "Si", "Fe"])
    pos = np.zeros((len(elem), 3))
    cell = np.eye(3)

    idx = np.array([1, 4, 2, 0, 3])

    at = Atoms(elem, pos, cell=cell)
    ref_masses = at.get_masses()[idx]
    ref_charges = at.get_initial_charges()[idx]
    ref_elem = elem[idx]
    ref_Z = at.get_atomic_numbers()[idx]

    pred_el, pred_Z, pred_masses, pred_charges = get_elements_Z_and_masses(at)
    for ref, pred in zip(ref_elem, pred_el):
        assert ref == pred
    for ref, pred in zip(ref_Z, pred_Z):
        assert ref == pred
    for ref, pred in zip(ref_masses, pred_masses):
        assert ref == pred
    assert pred_charges is None


def test_create_random_structures():
    """
    Here, we check that create_random_structures() return the right number
    of structures, and that the structures are different
    (a problem that can happend when atoms.rattle() is not used rightly)
    """
    at = bulk("Si").repeat(2)
    nconfs = 1
    newconfs = create_random_structures(at, std=0.01, nconfs=nconfs)
    assert len(newconfs) == nconfs
    nconfs = 4
    newconfs = create_random_structures(at, std=0.01, nconfs=nconfs)
    assert len(newconfs) == nconfs

    for at in newconfs[1:]:
        assert at != newconfs[0]
