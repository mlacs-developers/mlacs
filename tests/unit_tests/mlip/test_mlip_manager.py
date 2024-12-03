"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from unittest.mock import patch

import numpy as np
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.mlip.mlip_manager import MlipManager, SelfMlipManager
from mlacs.mlip.delta_learning import DeltaLearningPotential


def create_dum_data(atoms):
    at = atoms.copy()
    energy = np.random.random()
    forces = np.random.random((len(atoms), 3))
    stress = np.random.random(6)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                 stress=stress)
    at.calc = calc
    return at


@patch.multiple(MlipManager, __abstractmethods__=set())
def test_update_matrices():
    at = bulk("Si")
    desc = SnapDescriptor(at, 2.4)
    manager = MlipManager(desc)

    nconfs = 0
    nparams = desc.ncolumns
    assert manager.nconfs == nconfs

    # Let's try adding one atom
    fakeat = create_dum_data(at)
    manager.update_matrices(fakeat)
    nconfs += 1
    assert manager.nconfs == nconfs

    nf = 3 * len(at) * nconfs
    ns = nconfs * 6
    assert manager.amat_e.shape == (nconfs, nparams)
    assert manager.amat_f.shape == (nf, nparams)
    assert manager.amat_s.shape == (ns, nparams)
    assert manager.ymat_e.shape == (nconfs,)
    assert manager.ymat_f.shape == (nf,)
    assert manager.ymat_s.shape == (ns,)

    # Let's try adding several atoms
    fakeat = []
    for _ in range(5):
        fakeat.append(create_dum_data(at))
        nconfs += 1
    manager.update_matrices(fakeat)
    assert manager.nconfs == nconfs

    nf = 3 * len(at) * nconfs
    ns = nconfs * 6
    assert manager.amat_e.shape == (nconfs, nparams)
    assert manager.amat_f.shape == (nf, nparams)
    assert manager.amat_s.shape == (ns, nparams)
    assert manager.ymat_e.shape == (nconfs,)
    assert manager.ymat_f.shape == (nf,)
    assert manager.ymat_s.shape == (ns,)

    # if manager.folder.exists():
    #    shutil.rmtree(manager.folder)


@patch.multiple(SelfMlipManager, __abstractmethods__=set())
def test_update_matrices_self():
    at = bulk("Si")
    desc = SnapDescriptor(at, 2.3)
    manager = SelfMlipManager(desc)

    nconfs = 0
    assert len(manager.natoms) == nconfs

    # Let's try adding one atom
    fakeat = create_dum_data(at)
    manager.update_matrices(fakeat)
    nconfs += 1
    assert manager.nconfs == nconfs
    assert len(manager.natoms) == nconfs
    assert len(manager.configurations) == nconfs

    # Let's try adding several atoms
    fakeat = []
    for _ in range(5):
        fakeat.append(create_dum_data(at))
        nconfs += 1
    manager.update_matrices(fakeat)
    assert manager.nconfs == nconfs
    assert len(manager.natoms) == nconfs
    assert len(manager.configurations) == nconfs

    # Now check that the atoms are the same we added
    manager = SelfMlipManager(desc)
    manager.update_matrices(fakeat)
    for refat, predat in zip(fakeat, manager.configurations):
        assert refat == predat


@patch.multiple(SelfMlipManager, __abstractmethods__=set())
def test_update_matrices_delta():
    at = bulk("Si")
    desc = SnapDescriptor(at, 2.3)
    nparams = desc.ncolumns
    model = LinearPotential(desc, parameters=dict(twojmax=4))
    ref_pair_style = "zbl 3.0 4.0"
    ref_pair_coeff = ["* * 14 14"]
    manager = DeltaLearningPotential(model, ref_pair_style,
                                     ref_pair_coeff)

    nconfs = 0
    assert len(manager.model.natoms) == nconfs

    # Let's try adding one atom
    fakeat = create_dum_data(at)
    manager.update_matrices(fakeat)
    nconfs += 1
    assert manager.model.nconfs == nconfs
    assert len(manager.model.natoms) == nconfs

    nf = 3 * len(at) * nconfs
    ns = nconfs * 6
    assert manager.model.amat_e.shape == (nconfs, nparams)
    assert manager.model.amat_f.shape == (nf, nparams)
    assert manager.model.amat_s.shape == (ns, nparams)
    assert manager.model.ymat_e.shape == (nconfs,)
    assert manager.model.ymat_f.shape == (nf,)
    assert manager.model.ymat_s.shape == (ns,)

    # Let's try adding several atoms
    fakeat = []
    for _ in range(5):
        fakeat.append(create_dum_data(at))
        nconfs += 1
    manager.update_matrices(fakeat)
    assert manager.nconfs == nconfs
    assert len(manager.model.natoms) == nconfs

    nf = 3 * len(at) * nconfs
    ns = nconfs * 6
    assert manager.model.amat_e.shape == (nconfs, nparams)
    assert manager.model.amat_f.shape == (nf, nparams)
    assert manager.model.amat_s.shape == (ns, nparams)
    assert manager.model.ymat_e.shape == (nconfs,)
    assert manager.model.ymat_f.shape == (nf,)
    assert manager.model.ymat_s.shape == (ns,)

    # if manager.folder.exists():
    #    shutil.rmtree(manager.folder)
