import pytest
import os

from pathlib import Path

import numpy as np
from ase.io.abinit import read_abinit_out
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from mlacs.utilities import AbinitNC, set_aseAtoms

from ... import context  # noqa


@pytest.fixture()
def build_ncobj():
    root = Path()
    files = []
    for f in root.rglob('*nc'):
        _nc = AbinitNC()
        _nc.ncfile = f
    yield files


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
        ncatoms = set_aseAtoms(ncobj.read())
        assert atoms == ncatoms


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
