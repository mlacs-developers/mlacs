"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path
import shutil
from ase.build import bulk
from ... import context  # noqa

from mlacs.state import LammpsState


def create_mlip():
    mlip_model = """# Cu MLIP parameters
# Descriptor   snap

# nelems   ncoefs
1 15
  -0.209698103962831089441820608954
  -0.002981838238410259092625587130
   0.030446044924109540857903866140
  -0.087136152276158285179263884856
   0.026362053083803682618224684120
  -0.100902848724669771951489849471
   0.038311487921084796948179018727
  -0.064435647545731972174465340686
   0.001572805371360150132609145857
  -0.032502969114244058268514692145
  -0.115870519655493775523780186631
  -0.049361552677329688376683947126
   0.000011351286669566577674714281
  -0.023482718193700437281368209597
   0.004556622739442980560276019730"""
    mlip_descriptor = """# Cu MLIP parameters
# Descriptor:  snap
# Model:       linear

rcutfac         4.2
twojmax         4
rfac0           0.99363
rmin0           0.0
switchflag      1
bzeroflag       1
wselfallflag    0



nelems      1
elems       Cu
radelems    0.5
welems      1.0"""
    root = Path()
    path = root / "MLIP"
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / 'MLIP.model'
    descriptor_path = path / 'MLIP.descriptor'
    model_path.write_text(mlip_model)
    descriptor_path.write_text(mlip_descriptor)
    return model_path.absolute(), descriptor_path.absolute()


def cleanup():
    root = Path()
    paths = [Path("MLIP"), Path("MolecularDynamics"), Path("Trajectory")]
    for path in paths:
        if (root/path).exists():
            shutil.rmtree(root/path)


def test_NVT():
    """
    Make sure NVT correctly add atoms.info['info_state']
    """
    cleanup()
    mpath, dpath = create_mlip()

    ls = LammpsState(273.15, nsteps=100, nsteps_eq=20, dt=0.5)
    at = bulk("Cu").repeat(2)
    ps = f"mliap model linear {mpath} descriptor sna {dpath}"
    pc = ['* * Cu']
    model_post = None
    atom_style = "atomic"
    eq = False

    md_at = ls.run_dynamics(at, ps, pc, model_post, atom_style, eq)

    # Test
    assert ('info_state' in md_at.info)
    info = md_at.info['info_state']
    assert (info['ensemble'] == "NVT")
    assert (info['temperature'] == 273.15)
    assert (info['pressure'] is None)
    assert ('volume' not in md_at.info)
    cleanup()


def test_NPT():
    """
    Make sure NPT correctly add atoms.info['info_state']
    """
    cleanup()
    mpath, dpath = create_mlip()

    ls = LammpsState(3200, nsteps=100, nsteps_eq=20, dt=0.5, pressure=1)
    at = bulk("Cu").repeat(2)
    ps = f"mliap model linear {mpath} descriptor sna {dpath}"
    pc = ['* * Cu']
    model_post = None
    atom_style = "atomic"
    eq = False

    md_at = ls.run_dynamics(at, ps, pc, model_post, atom_style, eq)

    # Test
    assert ('info_state' in md_at.info)
    info = md_at.info['info_state']
    assert (info['ensemble'] == "NPT")
    assert (info['temperature'] == 3200)
    assert (info['pressure'] == 1)
    assert ('volume' not in md_at.info)
    cleanup()


def test_tstop():
    """
    Make sure NPT correctly add atoms.info['info_state']
    """
    cleanup()
    mpath, dpath = create_mlip()

    ls = LammpsState(300, nsteps=100, nsteps_eq=20,
                     dt=0.5, pressure=1, t_stop=500)
    at = bulk("Cu").repeat(2)
    ps = f"mliap model linear {mpath} descriptor sna {dpath}"
    pc = ['* * Cu']
    model_post = None
    atom_style = "atomic"
    eq = False

    md_at = ls.run_dynamics(at, ps, pc, model_post, atom_style, eq)

    # Test
    assert ('info_state' in md_at.info)
    info = md_at.info['info_state']
    assert (info['ensemble'] == "NPT")
    assert (info['temperature'] is not None)
    assert (info['pressure'] == 1)
    assert ('volume' not in md_at.info)
    cleanup()


def test_pstop():
    """
    Make sure NPT correctly add atoms.info['info_state']
    """
    cleanup()
    mpath, dpath = create_mlip()

    ls = LammpsState(3200, nsteps=10, nsteps_eq=10,
                     dt=0.5, pressure=1, p_stop=10)
    at = bulk("Cu").repeat(2)
    ps = f"mliap model linear {mpath} descriptor sna {dpath}"
    pc = ['* * Cu']
    model_post = None
    atom_style = "atomic"
    eq = False

    md_at = ls.run_dynamics(at, ps, pc, model_post, atom_style, eq)

    # Test
    assert ('info_state' in md_at.info)
    info = md_at.info['info_state']
    assert (info['ensemble'] == "NPT")
    assert (info['temperature'] == 3200)
    assert (info['pressure'] is not None)
    assert ('volume' not in md_at.info)
    cleanup()
