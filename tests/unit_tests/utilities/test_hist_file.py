"""
// Copyright (C) 2022-2024 MLACS group (CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import pytest
from pathlib import Path
import os
import netCDF4 as nc
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.units import Bohr, Hartree

from mlacs.mlip import SnapDescriptor
from mlacs.mlip import LinearPotential
from mlacs.mlip import MbarManager
from mlacs.state import LammpsState
from mlacs.properties import CalcExecFunction
from mlacs import OtfMlacs
from mlacs.utilities.io_abinit import MlacsHist, AbinitNC


from ... import context  # noqa


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_basic_hist():
    """
    Check that a '*HIST_nc' file is created, with required format.
    Check that it stores some basic thermodynamic variables.
    Check that it can be read with the hist_read() method.
    """
    pytest_path = os.getenv('PYTEST_CURRENT_TEST')
    root = Path(pytest_path).parents[0].absolute()
    test_wkdir = root / 'tmp_hist_dir1'

    nb_mlacs_iter1 = 5
    at = bulk("Cu", cubic=True).repeat(2)
    calc = EMT()
    T = 400
    P = 0
    nsteps = 10
    parameters = {"twojmax": 6}
    descriptor = SnapDescriptor(at, parameters=parameters)
    mbar = MbarManager(parameters={"solver": "L-BFGS-B"})
    mlip_mbar = LinearPotential(descriptor, weight=mbar)
    target_nc_format = 'NETCDF4'
    nb_states = 2
    states = list(LammpsState(T, P, nsteps=nsteps) for i in range(nb_states))
    properties = [CalcExecFunction('get_kinetic_energy')]
    dyn1 = OtfMlacs(at,
                    states,
                    calc,
                    mlip_mbar,
                    properties,
                    neq=0,
                    workdir=test_wkdir,
                    ncprefix='obj1',
                    ncformat=target_nc_format)
    dyn1.run(nb_mlacs_iter1)

    # Check that file exists
    script_name = 'obj1_' + 'test_hist_file'
    script_name += '_HIST.nc'
    path_name = test_wkdir / script_name
    assert path_name.is_file()

    # Check that some basic properties are stored, and have proper features
    with nc.Dataset(str(path_name), 'r') as ncfile:
        assert ncfile.file_format == target_nc_format
        vol_var = ncfile['vol']
        vol_data = vol_var[:].data
        vol_unit = vol_var.unit

        assert isinstance(vol_var[:], np.ma.MaskedArray)
        assert isinstance(vol_data, np.ndarray)
        assert vol_data.shape == (nb_states*(nb_mlacs_iter1-1),)
        assert vol_unit == 'Bohr^3'
        etotal_var = ncfile['etotal']
        etotal_unit = etotal_var.unit
        assert etotal_unit == 'Ha'

    # Check that the MlacsHist class is running properly
    ncfile = MlacsHist(ncpath=path_name)
    var_names = ncfile.get_var_names()
    assert isinstance(var_names, list)
    dict_var_units = ncfile.get_units()
    assert isinstance(dict_var_units, dict)
    var_dim_dict = ncfile.var_dim_dict
    dict_name_label = {x[0]: label for label, x in var_dim_dict.items()}
    assert 'temper' in dict_name_label

    # Check that weights from HIST and from MLIP folder are the same
    weights = ncfile.read_obs('weights')
    weights_meta = ncfile.read_obs('weights_meta')
    weights_idx = weights_meta[:, 0]
    dict_weights = {}
    idx_bounds = np.argwhere(weights_idx == 1.0)[:, 0]
    for ii in range(len(idx_bounds)-1):
        iter_mlacs = ii+1
        i1, i2 = idx_bounds[ii], idx_bounds[ii+1]
        dict_weights[iter_mlacs] = [weights_idx[i1:i2], weights[i1:i2]]
    last_weights = dict_weights[iter_mlacs][1]
    N_eff = np.sum(last_weights)**2 / np.sum(last_weights**2)
    if nb_mlacs_iter1 >= 5:
        subfold = "Coef" + str(nb_mlacs_iter1-2)
        p = test_wkdir / 'MLIP' / subfold
        test_weights = np.loadtxt(p / "MLIP.weight")
        # The first two training confs are excluded from the HIST file
        test_w = test_weights[2:]
        test_w /= np.sum(test_w)
        nb_eff_test = np.sum(test_w)**2 / np.sum(test_w**2)
        assert np.round(N_eff, 5) == np.round(nb_eff_test, 5)


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_distinct_hist():
    """
    Check that a distinct OtfMlacs object with different features
    is saved in a separate *HIST.nc file.
    In particular, for NETCDF3_CLASSIC format, a *WEIGHTS.nc file is created.
    """
    pytest_path = os.getenv('PYTEST_CURRENT_TEST')
    root = Path(pytest_path).parents[0].absolute()
    test_wkdir2 = root / 'tmp_hist_dir2'

    nb_mlacs_iter2 = 2
    at = bulk("Cu", cubic=True).repeat(2)
    calc = EMT()
    T = 400
    P = 0
    nsteps = 10
    parameters = {"twojmax": 6}
    descriptor = SnapDescriptor(at, parameters=parameters)
    parameters = {"solver": "L-BFGS-B"}
    mlip_mbar = LinearPotential(descriptor)
    target_nc_format2 = 'NETCDF3_CLASSIC'
    nc_name2 = 'obj2_test_hist_file_HIST.nc'
    path_name2 = test_wkdir2 / nc_name2
    nc_weights = 'obj2_test_hist_file_WEIGHTS.nc'
    path_weight = test_wkdir2 / nc_weights
    nb_states2 = 2
    states2 = list(LammpsState(T, P, nsteps=nsteps) for i in range(nb_states2))
    dyn2 = OtfMlacs(at,
                    states2,
                    calc,
                    mlip_mbar,
                    neq=0,
                    workdir=test_wkdir2,
                    ncprefix='obj2',
                    ncformat=target_nc_format2)
    dyn2.run(nb_mlacs_iter2)
    assert path_name2.is_file()
    assert path_weight.is_file()
    with nc.Dataset(str(path_name2), 'r') as ncfile:
        assert ncfile.file_format == target_nc_format2
        vol_var = ncfile['vol']
        vol_data = vol_var[:].data
        assert vol_data.shape == (nb_states2*(nb_mlacs_iter2-1),)


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_restart_hist():
    """
    Check restart of *HIST.nc file, i.e. proper concatenation of data.
    """
    pytest_path = os.getenv('PYTEST_CURRENT_TEST')
    root = Path(pytest_path).parents[0].absolute()
    test_wkdir = root / 'tmp_hist_dir3'
    target_nc_format = 'NETCDF4'
    nb_states = 1

    nb_mlacs_iter1 = 3
    nb_mlacs_iter3 = 1
    at = bulk("Cu", cubic=True).repeat(2)
    calc = EMT()
    T = 400
    P = 0
    nsteps = 10
    parameters = {"twojmax": 6}
    descriptor = SnapDescriptor(at, parameters=parameters)
    parameters = {"solver": "L-BFGS-B"}
    mlip_mbar = LinearPotential(descriptor)
    states = list(LammpsState(T, P, nsteps=nsteps) for i in range(nb_states))
    properties = [CalcExecFunction('get_kinetic_energy')]
    dyn1 = OtfMlacs(at,
                    states,
                    calc,
                    mlip_mbar,
                    properties,
                    neq=0,
                    workdir=test_wkdir,
                    ncprefix='obj3',
                    ncformat=target_nc_format)
    dyn1.run(nb_mlacs_iter1)

    states = list(LammpsState(T, P, nsteps=nsteps) for i in range(nb_states))
    properties = [CalcExecFunction('get_kinetic_energy')]
    dyn2 = OtfMlacs(at,
                    states,
                    calc,
                    mlip_mbar,
                    properties,
                    neq=0,
                    workdir=test_wkdir,
                    ncprefix='obj3',
                    ncformat=target_nc_format)
    dyn2.run(nb_mlacs_iter3)

    script_name = 'obj3_' + 'test_hist_file'
    script_name += '_HIST.nc'
    path_name = test_wkdir / script_name
    with nc.Dataset(str(path_name), 'r') as ncfile:
        vol_var = ncfile['vol']
        vol_data = vol_var[:].data
        assert vol_data.shape == (nb_states*(nb_mlacs_iter1-1+nb_mlacs_iter3),)


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_hist_converter():
    """
    Check that
        (1) converter from Abinit HIST.nc to list of ase.Atoms
            [namely AbinitNC.convert_to_atoms()]
        (2) converter from list of ase.Atoms to Abinit HIST.nc
            [namely AtomsToHist]
    are inverses of each other.
    """
    # Load a reference HIST.nc file from an Abinit calculation
    workdir = str(Path() / 'reference_files/')
    prefix = 'sqa'
    suffix = 'HIST'
    hist = AbinitNC(workdir=workdir, prefix=prefix, suffix=suffix)
    hist_results = hist.read()
    hist_cell = hist_results['rprimd']
    hist_positions = hist_results['xcart']

    # Convert to ASE list of Atoms
    hist_atoms = hist.convert_to_atoms()

    # Check the conversion of units (must be ASE units)
    assert len(hist_atoms) == len(hist_positions)
    ase_energy_first_conf = hist_atoms[0].get_total_energy()
    abinit_energy_first_conf = hist_results['etotal'][0]*Hartree
    assert ase_energy_first_conf == abinit_energy_first_conf
    ase_cell_first_conf = np.array(hist_atoms[0].get_cell())
    abinit_cell_first_conf = hist_cell[0]*Bohr
    assert np.array_equal(ase_cell_first_conf, abinit_cell_first_conf)

    # Convert ASE list of Atoms back into HIST.nc format
    ncprefix = 'myconverted'
    workdir = Path('')
    obj = MlacsHist(ncprefix=ncprefix,
                    workdir=workdir,
                    atoms=hist_atoms)
    obj.convert_to_hist()

    # Check that created new HIST.nc is same as the reference one.
    hist_new = AbinitNC()
    nc_name = ncprefix + '_HIST.nc'
    nc_path = Path('') / nc_name
    hist_new.ncfile = str(nc_path)
    hist_new_results = hist_new.read()
    assert np.allclose(hist_new_results['znucl'],
                       hist_results['znucl'],
                       rtol=10**-8)
    assert np.allclose(hist_new_results['etotal'],
                       hist_results['etotal'],
                       rtol=10**-8)
    assert np.allclose(hist_new_results['xcart'],
                       hist_results['xcart'],
                       rtol=10**-8)
    assert np.allclose(hist_new_results['fcart'],
                       hist_results['fcart'],
                       rtol=10**-8)
    assert np.allclose(hist_new_results['strten'],
                       hist_results['strten'],
                       rtol=10**-8)

    nc_path.unlink()


@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_gsr_out():
    """
    Check that reference GSR.nc/OUT.nc files can be read by AbinitNC class,
    and converted to list of ase.Atoms in ASE units.
    """

    workdir = str(Path() / 'reference_files/')
    prefix = 'abinit'

    gsr = AbinitNC(workdir=workdir, prefix=prefix, suffix='GSR')
    gsr_atoms = gsr.convert_to_atoms()

    out = AbinitNC(workdir=workdir, prefix=prefix, suffix='OUT')

    out_atoms = out.convert_to_atoms()

    assert len(gsr_atoms) == 1
    assert len(out_atoms) == 1

    gsr_cell = np.array(gsr_atoms[0].get_cell())
    gsr_results = gsr.read()
    nc_gsr_cell = gsr_results['primitive_vectors']
    assert np.allclose(gsr_cell, nc_gsr_cell*Bohr, rtol=10**-8)

    out_cell = np.array(out_atoms[0].get_cell())
    out_results = out.read()
    out_rprim = np.reshape(out_results['rprim'], (3, 3))
    nc_out_cell = out_results['acell'] * out_rprim
    assert np.allclose(out_cell, nc_out_cell*Bohr, rtol=10**-8)
