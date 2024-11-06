"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest
from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.mlip import MomentTensorPotential
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation of Cu at several temperature
The true potential is the EMT as implemented in ASE
"""


@pytest.fixture
def expected_folder(expected_folder_base):
    return ["MTP"]


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


@pytest.mark.skipif(context.has_mlp(), reason="Need MLIP-2 to use MTP")
def test_mtp_restart(root, treelink):
    # Parameters --------------------------------------------------------------
    temperature = 300  # K
    nconfs = 2
    nsteps = 10
    nsteps_eq = 10
    neq = 1
    cell_size = 2
    dt = 1.5  # fs
    mtp_params = dict(level=8,
                      radial_basis_type="RBChebyshev",
                      min_dist=1.0,
                      max_dist=5.0,
                      radial_basis_size=8)

    mtp_fit = dict(scale_by_forces=0,
                   max_iter=100,
                   bfgs_conv_tol=1e-3,
                   weighting="vibrations",
                   init_params="random",
                   update_mindist=False)

    # Supercell creation ------------------------------------------------------
    atoms = bulk('Cu', cubic=True).repeat(cell_size)
    calc = EMT()

    # Running Initial Lammps State
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

    mtp = MomentTensorPotential(atoms,
                                mlpbin="mlp",
                                folder=Path("MTP").absolute(),
                                mtp_parameters=mtp_params,
                                fit_parameters=mtp_fit)

    sampling = OtfMlacs(atoms, state, calc, mtp, neq=neq)
    sampling.run(nconfs)

    # Running Restart Lammps State
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

    mtp = MomentTensorPotential(atoms,
                                mlpbin="mlp",
                                folder=Path("MTP").absolute(),
                                mtp_parameters={},
                                fit_parameters={})

    sampling = OtfMlacs(atoms, state, calc, mtp, neq=neq)
    sampling.run(nconfs)
    assert sum(sampling.nconfs) == 4
