"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

from pathlib import Path

from ase.io.abinit import read_abinit_out
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from mlacs.utilities import AbinitNC

from ... import context  # noqa


@pytest.fixture()
def build_ncobj():
    root = Path()
    files = []
    for f in root.rglob('*nc'):
        _nc = AbinitNC()
        _nc.ncfile = f
    yield files


@pytest.fixture
@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_atoms_from_ncfiles(root, build_ncobj):
    """
    """
    f = root / 'reference_files'
    with open(f / 'abinit.abo', 'r') as fd:
        results = read_abinit_out(fd)

    atoms = results.pop("atoms")
    energy = results.pop("energy")
    forces = results.pop("forces")
    stress = results.pop("stress")

    calc = SPCalc(atoms,
                  energy=energy,
                  forces=forces,
                  stress=stress)
    calc.version = results.pop("version")
    atoms.calc = calc

    for ncobj in build_ncobj:
        ncobj.read()
        ncatoms = ncobj.convert_to_atoms()
        assert atoms == ncatoms


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_dict_from_ncfile(root, build_ncobj):
    """
    """

    fulldict = dict()
    _nc = AbinitNC()
    for f in root.rglob('*nc'):
        fulldict.update(_nc.read(f))

    adddict = dict()
    for ncobj in build_ncobj:
        adddict.update(ncobj.read())

    for key, val in adddict.items():
        assert (fulldict[key] == val).all()
