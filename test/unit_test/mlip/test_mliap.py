from pathlib import Path
import shutil

import numpy as np
from ase.build import bulk

from ... import context  # noqa
from mlacs.mlip import MliapDescriptor


def test_parameters_snap():
    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5,
                      radelems=0.2,
                      welems=0.3)
    mliap = MliapDescriptor(at, rcut, parameters=parameters)

    assert mliap.ndesc == 20
    assert mliap.rcut == rcut
    assert mliap.chemflag == 0
    assert np.all(mliap.radelems == 0.2)
    assert np.all(mliap.welems == 0.3)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems)
    mliap = MliapDescriptor(at, rcut, parameters=parameters)

    assert mliap.ndesc == 20
    assert mliap.rcut == rcut
    assert mliap.chemflag == 0
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems,
                      chemflag=1)
    mliap = MliapDescriptor(at, rcut, parameters=parameters)

    assert mliap.ndesc == 160
    assert mliap.rcut == rcut
    assert mliap.chemflag == 1
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(twojmax=5,
                      radelems=radelems,
                      welems=welems)
    mliap = MliapDescriptor(at, rcut, parameters=parameters, model="quadratic")

    assert mliap.ndesc == 230
    assert mliap.rcut == rcut
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)


def test_parameters_so3():
    at = bulk("Si")

    rcut = 3.5
    radelems = np.array([0.2])
    welems = np.array([0.5])
    parameters = dict(nmax=2,
                      lmax=2,
                      radelems=radelems,
                      welems=welems)
    mliap = MliapDescriptor(at, rcut, parameters=parameters, style="so3")

    assert mliap.ndesc == 9
    assert mliap.rcut == rcut
    assert mliap.chemflag == 0
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(nmax=2,
                      lmax=2,
                      radelems=radelems,
                      welems=welems)
    mliap = MliapDescriptor(at, rcut, parameters=parameters, style="so3")

    assert mliap.ndesc == 9
    assert mliap.rcut == rcut
    assert mliap.chemflag == 0
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)

    at = bulk("FeRh", "cesiumchloride", 3.02)
    rcut = 3.5
    radelems = np.array([0.2, 0.3])
    welems = np.array([0.5, 0.3])
    parameters = dict(nmax=2,
                      lmax=2,
                      radelems=radelems,
                      welems=welems)
    mliap = MliapDescriptor(at, rcut, parameters=parameters, model="quadratic",
                            style="so3")

    assert mliap.ndesc == 54
    assert mliap.rcut == rcut
    assert np.all(mliap.radelems == radelems)
    assert np.all(mliap.welems == welems)


model_ref = """# Si MLIP parameters
# Descriptor   snap

# nelems   ncoefs
1 21
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
    mliapfold = root / "Mliap"

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)
    mliap = MliapDescriptor(at, rcut, parameters=parameters)
    coeff = np.arange(0, 21)

    if mliapfold.exists():
        shutil.rmtree(mliapfold)
    mliapfold.mkdir()
    mliap.write_mlip(coeff, subfolder=mliapfold)

    mliapfile = mliapfold / "MLIP.model"
    assert mliapfile.exists()

    allref = model_ref.split("\n")
    allref = [line.rstrip() for line in allref]
    with open(mliapfile, "r") as fd:
        allpred = [line.rstrip() for line in fd]
    assert allref[0] == allpred[0]
    assert allref[1] == allpred[1]

    ref_info = [int(a) for a in allref[4].split()]
    pred_info = [int(a) for a in allpred[4].split()]
    assert np.all(ref_info == pred_info)

    for cref, cpred in zip(allref[5:], allpred[5:]):
        assert np.isclose(float(cref), float(cpred))

    shutil.rmtree(mliapfold)


desc_ref = """# Si MLIP parameters
# Descriptor:  snap
# Model:       linear

rcutfac         3.5
twojmax         5
rfac0           0.99363
rmin0           0.0
switchflag      1
bzeroflag       1
wselfallflag    0
bnormflag       1



nelems      1
elems       Si
radelems    0.5
welems      1.0
"""


def test_writing_descriptor():
    root = Path()
    mliapfold = root / "Mliap"

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)
    mliapfile = mliapfold / "MLIP.descriptor"
    mliap = MliapDescriptor(at, rcut, parameters=parameters)

    if mliapfold.exists():
        shutil.rmtree(mliapfold)
    mliapfold.mkdir()

    mliap._write_mlip_params(subfolder=mliapfold)

    allref = desc_ref.split("\n")
    allref = [line.rstrip() for line in allref]

    with open(mliapfile, "r") as fd:
        allpred = [line.rstrip() for line in fd]

    assert allpred[0] == allref[0]
    assert allpred[1] == allref[1]
    assert allpred[2] == allref[2]

    for i in range(4, 12):
        ref = allref[i].split()
        pred = allref[i].split()

        assert ref[0] == pred[0]
        assert np.isclose(float(ref[1]), float(pred[1]), atol=1e-8)
    shutil.rmtree(mliapfold)


desc_ref_so3 = """# Si MLIP parameters
# Descriptor:  so3
# Model:       linear

rcutfac         3.5
nmax            3
lmax            2
alpha           1.0



nelems      1
elems       Si
radelems    0.5
welems      1.0
"""


def test_writing_descriptor_so3():
    root = Path()
    mliapfold = root / "Mliap"

    at = bulk("Si")

    rcut = 3.5
    parameters = dict(nmax=3, lmax=2)
    mliap = MliapDescriptor(at, rcut, parameters=parameters, style="so3")
    mliapfile = mliapfold / "MLIP.descriptor"

    if mliapfold.exists():
        shutil.rmtree(mliapfold)
    mliapfold.mkdir()

    mliap._write_mlip_params(subfolder=mliapfold)

    allref = desc_ref_so3.split("\n")
    allref = [line.rstrip() for line in allref]

    with open(mliapfile, "r") as fd:
        allpred = [line.rstrip() for line in fd]

    assert allpred[0] == allref[0]
    assert allpred[1] == allref[1]
    assert allpred[2] == allref[2]

    for i in range(4, 8):
        ref = allref[i].split()
        pred = allpred[i].split()

        assert ref[0] == pred[0]
        assert np.isclose(float(ref[1]), float(pred[1]), atol=1e-8)

    for i in range(11, 15):
        ref = allref[i].split()
        pred = allpred[i].split()
        if ref[0] == "elems":
            assert ref[0] == pred[0]
            assert ref[1] == pred[1]
        else:
            assert ref[0] == pred[0]
            assert np.isclose(float(ref[1]), float(pred[1]), atol=1e-8)
    shutil.rmtree(mliapfold)


def test_get_pair_style_coeff():
    root = Path().absolute()
    at = bulk("Si")

    rcut = 3.5
    parameters = dict(twojmax=5)
    mliap = MliapDescriptor(at, rcut, parameters=parameters)
    mliap.mlip_model = (root / "Mliap")
    mliap.mlip_desc = (root / "Mliap")

    pred_st, pred_co = mliap.get_pair_style_coeff()

    model_file = (root / "Mliap/MLIP.model").as_posix()
    desc_file = (root / "Mliap/MLIP.descriptor").as_posix()

    ref_st = f"mliap model linear {model_file} descriptor sna {desc_file}"
    assert pred_st == ref_st
    ref_co = ["* * Si"]
    assert pred_co[0] == ref_co[0]

    # Now with so3
    mliap = MliapDescriptor(at, rcut, parameters=parameters, style="so3")
    mliap.mlip_model = (root / "Mliap")
    mliap.mlip_desc = (root / "Mliap")

    pred_st, pred_co = mliap.get_pair_style_coeff()

    model_file = (root / "Mliap/MLIP.model").as_posix()
    desc_file = (root / "Mliap/MLIP.descriptor").as_posix()
    ref_st = f"mliap model linear {model_file} descriptor so3 {desc_file}"
    assert pred_st == ref_st
    ref_co = ["* * Si"]
    assert pred_co[0] == ref_co[0]