"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
import shutil

import numpy as np
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor
from mlacs.mlip import LinearPotential


def create_dum_data(atoms):
    at = atoms.copy()
    at.rattle(0.01, seed=np.random.randint(1, 99999))
    energy = np.random.random()
    forces = np.random.random((len(atoms), 3))
    stress = np.random.random(6)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces,
                                 stress=stress)
    calc = EMT()
    at.calc = calc
    return at


def test_linear_potential():
    """
    TODO find other things to test
    Here, we just test if everything runs alright
    """
    root = Path()
    folder = 'Snap'
    snapfold = root / folder
    if snapfold.exists():
        shutil.rmtree(snapfold)

    at = bulk("Cu").repeat(2)

    params = dict(twojmax=4)
    desc = SnapDescriptor(at, 3.0, params)

    # First we test if OLS functions
    print(snapfold.absolute())
    mlip = LinearPotential(
        desc, workdir=root, folder=folder, subfolder='LeastSquare')
    fakeat = []
    for _ in range(5):
        fakeat.append(create_dum_data(at))

    mlip.update_matrices(fakeat)
    mlip.train_mlip()

    shutil.rmtree(snapfold)

    # And with ridge regression
    mlip_params = dict(method="ridge")
    mlip = LinearPotential(desc, parameters=mlip_params,
                           workdir=root, folder=folder, subfolder='Ridge')

    mlip.update_matrices(fakeat)
    mlip.train_mlip()

    # Let's check that what we compute with the matrix is also what
    # we get with LAMMPS

    calc = LAMMPS(pair_style=mlip.pair_style,
                  pair_coeff=mlip.pair_coeff)
    mlip_fakeat = []
    emat = []
    fmat = []
    smat = []
    for at in fakeat:
        at0 = at.copy()
        at0.calc = calc
        at0.get_potential_energy()
        mlip_fakeat.append(at0)

        etmp, ftmp, stmp = mlip.predict(at)
        emat.append(etmp)
        fmat.append(ftmp)
        smat.append(stmp)

    emat = np.array(emat)
    fmat = np.array(fmat)
    smat = np.array(smat)

    elammps = np.array([at.get_potential_energy() for at in mlip_fakeat])
    flammps = np.array([at.get_forces() for at in mlip_fakeat])
    flammps = np.reshape(flammps, np.shape(fmat))
    slammps = np.array([at.get_stress() for at in mlip_fakeat])

    # TODO check if it's a writing problem or if we have a slight difference in
    # How we compute stuff wrt LAMMPS
    assert np.allclose(elammps, emat, atol=1e-0)
    assert np.allclose(flammps, fmat, atol=1e-0)
    assert np.allclose(slammps, smat, atol=1e-0)

    shutil.rmtree(snapfold)
    for file in ["Energy", "Forces", "Stress"]:
        (root / f"MLIP-{file}_comparison.dat").unlink()
