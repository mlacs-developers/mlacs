"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
import shutil
from pathlib import Path

from ase import Atoms
import pytest

from ... import context  # noqa

from mlacs.mlip import SnapDescriptor, MliapDescriptor, LinearPotential
from mlacs.mlip import MomentTensorPotential, AceDescriptor, TensorpotPotential
from mlacs.mlip import UniformWeight
from mlacs.state import LammpsState


def generate_atoms():
    # Create_H2O
    positions = [[0.0, 0.0, 0.0],          # O
                 [0.9584, 0.0, 0.0],       # H
                 [-0.2396, 0.9273, 0.0]]   # H
    cell = [10.0, 10.0, 10.0]
    h2o = Atoms(symbols="OH2", positions=positions, cell=cell, pbc=True)

    # Create_H2O2
    positions = [[0.0, 0.0, 0.0],          # O
                 [1.45, 0.0, 0.0],         # O
                 [-0.78, 0.76, 0.0],       # H
                 [2.23, 0.76, 0.0]]        # H
    cell = [10.0, 10.0, 10.0]
    h2o2 = Atoms(symbols="H2O2", positions=positions, cell=cell, pbc=True)

    # Create_CH4
    positions = [[0.0, 0.0, 0.0],          # C
                 [0.63, 0.63, 0.63],       # H
                 [-0.63, -0.63, 0.63],     # H
                 [0.63, -0.63, -0.63],     # H
                 [-0.63, 0.63, -0.63]]      # H
    cell = [10.0, 10.0, 10.0]
    ch4 = Atoms(symbols="CH4", positions=positions, cell=cell, pbc=True)
    return [h2o, h2o2, ch4]


def generate_mlip(atoms, mlip_name):
    rcut = 5
    uni = UniformWeight(nthrow=10,
                        energy_coefficient=1.0,
                        forces_coefficient=1.0,
                        stress_coefficient=0.0)

    # SNAP
    if mlip_name == "SNAP_HOC.model":
        snap_params = {"twojmax": 4}
        snap_desc = SnapDescriptor(atoms=atoms,
                                   rcut=rcut,
                                   parameters=snap_params,
                                   model="linear")
        snap = LinearPotential(descriptor=snap_desc,
                               weight=uni, folder="SNAP")
        return snap

    # MLIAP
    if mlip_name == "MLIAP_HOC.model":
        snap_params = {"twojmax": 4}
        mliap_desc = MliapDescriptor(atoms=atoms,
                                     rcut=rcut,
                                     parameters=snap_params,
                                     model="linear",
                                     style="snap")
        mliap = LinearPotential(descriptor=mliap_desc,
                                weight=uni, folder="MLIAP")
        return mliap

    # MTP
    if mlip_name == "MTP_HOC.mtp":
        mtp_params = {"level": 6}
        mtp = MomentTensorPotential(atoms=atoms, mtp_parameters=mtp_params)
        return mtp
    # ACE
    if mlip_name == "ACE_HOC.yace":
        ace_desc = AceDescriptor(atoms, free_at_e={'C': 0, 'H': 0, 'O': 0},
                                 tol_e=100, tol_f=250, rcut=rcut)
        ace = TensorpotPotential(ace_desc, weight=uni, folder="ACE")
        return ace


@pytest.mark.parametrize("mlip_fn", ["SNAP_HOC.model", "MLIAP_HOC.model",
                                     "MTP_HOC.mtp", "ACE_HOC.yace"])
def test_lammpsstate_varntypat(root, mlip_fn):
    if not (Path(root) / "reference_files").exists():
        root = root.parents[0]  # Go up 1 folders to find reference_files
    if not (Path(root) / "reference_files").exists():
        root = root.parents[0]  # Go up another folder to find reference_files

    atoms = generate_atoms()
    mlip = generate_mlip(atoms, mlip_fn)
    fn = Path(root) / "reference_files" / "Pretrained-Potential" / mlip_fn
    state = LammpsState(temperature=500, dt=0.01, nsteps=5)
    folder = "MLIP"

    # The loading of MLIP to run_dynamics should be cleaned up
    if fn.suffix == ".model":  # SNAP or MLIAP
        coef = mlip.descriptor.get_coef(fn)
        if "SNAP" in str(fn):
            folder = "SNAP"
        else:
            folder = "MLIAP"
    elif fn.suffix == ".mtp":  # MTP
        folder = Path("MTP")
        folder.mkdir(exist_ok=True)
        mtp_fn = folder / fn.name
        shutil.copy(fn, mtp_fn)
        coef = fn
        with open(folder / "mlip.ini", "w") as f:
            f.write(f"mtp-filename    {mtp_fn.absolute()}\n")
            f.write("select          FALSE")
    elif fn.suffix == ".yace":  # ACE
        coef = fn
        folder = Path("ACE")
        folder.mkdir(exist_ok=True)
        shutil.copy(fn, folder / "ACE.yace")
    mlip.next_coefs(coef)
    for molecule in atoms:
        state.run_dynamics(supercell=molecule,
                           pair_style=mlip.pair_style,
                           pair_coeff=mlip.pair_coeff,
                           elements=mlip.get_elements())
    shutil.rmtree(folder)
    shutil.rmtree("Trajectory")


@pytest.mark.parametrize("mlip_fn", ["SNAP_HOC.model", "MLIAP_HOC.model",
                                     "ACE_HOC.yace"])
def test_predict_varntypat(root, mlip_fn):
    """
    Test that a MLIAP/SNAP/MTP/ACE can use the predict function
    for a variable natom/ntypat
    """
    if not (Path(root) / "reference_files").exists():
        root = root.parents[0]  # Go up 1 folders to find reference_files
    if not (Path(root) / "reference_files").exists():
        root = root.parents[0]  # Go up another folder to find reference_files

    atoms = generate_atoms()
    mlip = generate_mlip(atoms, mlip_fn)
    fn = Path(root) / "reference_files" / "Pretrained-Potential" / mlip_fn
    folder = "MLIP"

    # The loading of MLIP to run_dynamics should be cleaned up
    if fn.suffix == ".model":  # SNAP or MLIAP
        coef = mlip.descriptor.get_coef(fn)
        if "SNAP" in str(fn):
            folder = "SNAP"
        else:
            folder = "MLIAP"
    elif fn.suffix == ".mtp":  # MTP
        folder = Path("MTP")
        folder.mkdir(exist_ok=True)
        mtp_fn = folder / fn.name
        shutil.copy(fn, mtp_fn)
        coef = fn
        with open(folder / "mlip.ini", "w") as f:
            f.write(f"mtp-filename    {mtp_fn.absolute()}\n")
            f.write("select          FALSE")
    elif fn.suffix == ".yace":  # ACE
        coef = fn
        folder = Path("ACE")
        folder.mkdir(exist_ok=True)
        shutil.copy(fn, folder / "ACE.yace")
    mlip.next_coefs(coef)

    mlip.predict(atoms)
    shutil.rmtree(folder)
