import pytest

import numpy as np
from ase.build import bulk
from ase.io import read
from ase.calculators.emt import EMT

from ... import context  # noqa
from mlacs import OtfMlacs
from mlacs.mlip import MliapDescriptor, LinearPotential
from mlacs.state import NebLammpsState
# from mlacs.properties import CalcNeb


@pytest.fixture
def expected_folder(expected_folder_base):
    folder = expected_folder_base
    folder.append("Mliap")
    folder.pop(folder.index("Snap"))
    return expected_folder_base


@pytest.fixture
def expected_files(expected_files_base):
    return expected_files_base


def test_mlacs_nebstate_vanilla(root, treelink):

    atoms = bulk("Ag", cubic=True).repeat(3)
    nebat = [atoms.copy(), atoms.copy()]
    nebat[0].pop(0)
    nebat[1].pop(1)
    # Check that the first atom is the one we started with
    assert len(nebat[0]) == len(nebat[1])
    natoms = len(nebat[-1])
    nstep = 2
    calc = EMT()

    mlip_params = dict(nmax=4, lmax=4)
    desc = MliapDescriptor(atoms, 4.2, mlip_params,
                           model="quadratic", style="so3")
    mlip = LinearPotential(desc, folder="Mliap")

    mode = "rdm_memory"
    nimages = 6
    state = NebLammpsState(nebat, nimages=nimages, mode=mode)

# RB I will add this later.
#    pair_mlip = dict(pair_style=mlip.pair_style, pair_coeff=mlip.pair_coeff)
#    func = CalcNeb(state=state, args=pair_mlip)

#    sampling = OtfMlacs(nebat[0], state, calc, mlip, func, neq=5)
    sampling = OtfMlacs(nebat[0], state, calc, mlip, neq=5)
    sampling.run(nstep)

    for folder in treelink["folder"]:
        assert (root / folder).exists()

    for file in treelink["files"]:
        assert (root / file).exists()

    traj = read(root / "Trajectory.traj", ":")
    # Check that the same Atoms are used
    assert nebat == state.atoms
    # Check that the system didn't change in the process
    for at in traj:
        assert len(at) == natoms
    # Check the size of splined objects
    assert len(state.true_energies) == nimages
    assert isinstance(state.spline_energies, float)

    xi = np.linspace(0, 1, 1001)
    state.compute_spline(xi)

    assert len(state.spline_energies) == len(xi)
    # Check the effective masses for vacancies only one atom should move
    w_m = state._compute_weight_masses()
    assert len(w_m) == natoms
    assert np.sum(w_m) >= 1 and np.sum(w_m) < natoms
