"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest

from ase.build import bulk
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs.mlip import TensorpotPotential, AceDescriptor
from mlacs.mlip import EnergyBasedWeight
from mlacs.state import LammpsState
from mlacs import OtfMlacs


"""
Example of a MLACS simulation using ACE
This test is slow.
Maybe put it outside the testsuite
"""


@pytest.fixture
def expected_folder(expected_folder_base):
    return ["MLIP", "MolecularDynamics"]


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


@pytest.mark.slow
@pytest.mark.skipif(context.has_pyace(), reason="Need Python-Ace to use ACE")
def test_ace_restart(root, treelink):
    import pandas as pd

    # Parameters --------------------------------------------------------------
    temperature = 300  # K
    nconfs = 2
    nsteps = 10
    nsteps_eq = 10
    neq = 1
    cell_size = 2
    dt = 1.5  # fs
    rcut = 6

    # Supercell creation ------------------------------------------------------
    atoms = bulk('Cu', cubic=True).repeat(cell_size)
    calc = EMT()
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)

    weighter = EnergyBasedWeight(delta=0.025, stress_coefficient=0)
    fitting_dict = {'maxiter': 200, 'fit_cycles': 1, 'repulsion': 'auto',
                    'optimizer': 'BFGS',
                    'optimizer_options': {'disp': True, 'gtol': 0, 'xrtol': 0}}
    ace_descriptor = AceDescriptor(atoms=atoms,
                                   free_at_e={'Cu': 0},
                                   tol_e=10.0,  # meV/at
                                   tol_f=500.0,  # meV/ang
                                   nworkers=4,
                                   fitting_dict=fitting_dict,
                                   rcut=rcut)
    mlip = TensorpotPotential(descriptor=ace_descriptor,
                              weight=weighter)
    sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)
    sampling.run(nconfs)

    # Running Restart Lammps State
    state = LammpsState(temperature, dt=dt, nsteps=nsteps,
                        nsteps_eq=nsteps_eq)
    weighter = EnergyBasedWeight(delta=0.025, stress_coefficient=0)
    ace_descriptor = AceDescriptor(atoms=atoms,
                                   free_at_e={'Cu': 0},
                                   tol_e=10.0,  # meV/at
                                   tol_f=500.0,  # meV/ang
                                   nworkers=4,
                                   fitting_dict=fitting_dict,
                                   rcut=rcut)
    mlip = TensorpotPotential(descriptor=ace_descriptor,
                              weight=weighter)
    sampling = OtfMlacs(atoms, state, calc, mlip, neq=neq)
    sampling.run(nconfs)
    df = pd.read_pickle("MLIP/ACE.pckl.gzip", compression="gzip")
    # traj = read("Trajectory.traj", index=":")
    assert sum(sampling.nconfs) == 4
    assert len(df) == 4
    # assert len(traj) == 4  # BUG : To fix
