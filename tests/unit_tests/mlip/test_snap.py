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

from ... import context  # noqa
from mlacs.mlip import SnapDescriptor, LinearPotential


def test_parameters():
    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5,
                      radelems=0.2,
                      welems=0.3)
    snap = SnapDescriptor(at, rcut, parameters=parameters)

    assert snap.ndesc == 20
    assert snap.rcut == rcut
    assert snap.chemflag == 0
    assert np.all(snap.radelems == 0.2)
    assert np.all(snap.welems == 0.3)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems)
    snap = SnapDescriptor(at, rcut, parameters=parameters)

    assert snap.ndesc == 20
    assert snap.rcut == rcut
    assert snap.chemflag == 0
    assert np.all(snap.radelems == radelems)
    assert np.all(snap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems,
                      chemflag=1)
    snap = SnapDescriptor(at, rcut, parameters=parameters)

    assert snap.ndesc == 160
    assert snap.rcut == rcut
    assert snap.chemflag == 1
    assert np.all(snap.radelems == radelems)
    assert np.all(snap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems)
    snap = SnapDescriptor(at, rcut, parameters=parameters, model="quadratic")

    assert snap.ndesc == 230
    assert snap.rcut == rcut
    assert np.all(snap.radelems == radelems)
    assert np.all(snap.welems == welems)


model_ref = """# Si MLIP parameters
# Descriptor   SNAP

1 21
Si 0.5 1.0
   0.000000000000000000000000000000
   1.000000000000000000000000000000
   2.000000000000000000000000000000
   3.000000000000000000000000000000
   4.000000000000000000000000000000
   5.000000000000000000000000000000
   6.000000000000000000000000000000
   7.000000000000000000000000000000
   8.000000000000000000000000000000
   9.000000000000000000000000000000
  10.000000000000000000000000000000
  11.000000000000000000000000000000
  12.000000000000000000000000000000
  13.000000000000000000000000000000
  14.000000000000000000000000000000
  15.000000000000000000000000000000
  16.000000000000000000000000000000
  17.000000000000000000000000000000
  18.000000000000000000000000000000
  19.000000000000000000000000000000
  20.000000000000000000000000000000
"""


def test_writing_model():
    root = Path()
    folder = 'Snap'
    snapfold = root / folder
    if snapfold.exists():
        shutil.rmtree(snapfold)
    snapfold.mkdir()

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)
    snap = SnapDescriptor(at, rcut, parameters=parameters,
                          workdir=root, folder=folder)
    coeff = np.arange(0, 21)

    snap.write_mlip(coeff)

    snapfile = snapfold / "SNAP.model"
    assert snapfile.exists()

    allref = model_ref.split("\n")
    allref = [line.rstrip() for line in allref]
    with open(snapfile, "r") as fd:
        allpred = [line.rstrip() for line in fd]
    assert allref[0] == allpred[0]
    assert allref[1] == allpred[1]

    ref_info = [int(a) for a in allref[3].split()]
    pred_info = [int(a) for a in allpred[3].split()]
    assert np.all(ref_info == pred_info)

    assert allref[4].split()[0] == allpred[4].split()[0]
    ref_info = [float(a) for a in allref[4].split()[1:]]
    pred_info = [float(a) for a in allpred[4].split()[1:]]
    assert np.all(ref_info == pred_info)

    for cref, cpred in zip(allref[5:], allpred[5:]):
        assert np.isclose(float(cref), float(cpred))

    shutil.rmtree(snapfold)


desc_ref = """# Si MLIP parameters
# Descriptor:  SNAP
# Model:       linear

rcutfac         3.5
twojmax         5
rfac0           0.99363
rmin0           0.0
switchflag      1
bzeroflag       1
wselfallflag    0
bnormflag       1
"""


def test_writing_descriptor():
    root = Path()
    folder = 'Snap'
    snapfold = root / folder

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)
    snap = SnapDescriptor(at, rcut, parameters=parameters,
                          workdir=root, folder=folder)
    snapfile = snapfold / "SNAP.descriptor"

    if snapfold.exists():
        shutil.rmtree(snapfold)
    snapfold.mkdir()

    snap._write_mlip_params()

    allref = desc_ref.split("\n")
    allref = [line.rstrip() for line in allref]

    with open(snapfile, "r") as fd:
        allpred = [line.rstrip() for line in fd]

    assert allpred[0] == allref[0]
    assert allpred[1] == allref[1]
    assert allpred[2] == allref[2]

    for i in range(4, 12):
        ref = allref[i].split()
        pred = allpred[i].split()

        assert ref[0] == pred[0]
        assert np.isclose(float(ref[1]), float(pred[1]), atol=1e-8)
    shutil.rmtree(snapfold)


def test_get_pair_style_coeff():
    root = Path().absolute()
    folder = "Snap"

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)

    snap = SnapDescriptor(at, rcut, parameters=parameters)
    mlip = LinearPotential(descriptor=snap, workdir=root, folder=folder)

    pred_st, pred_co = mlip.pair_style, mlip.pair_coeff

    model_file = (root / folder / "SNAP.model").as_posix()
    desc_file = (root / folder / "SNAP.descriptor").as_posix()
    assert pred_st == "snap"
    ref_co = [f"* * {model_file}  {desc_file} Si"]
    assert pred_co[0] == ref_co[0]
