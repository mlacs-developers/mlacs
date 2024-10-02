import pytest
from pathlib import Path
import os
import netCDF4 as nc
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import SnapDescriptor
from mlacs.mlip import LinearPotential
from mlacs.state import LammpsState
from mlacs.properties import CalcExecFunction
from mlacs import OtfMlacs

from ... import context  # noqa

nb_mlacs_iter1 = 3
nb_mlacs_iter2 = 2
nb_mlacs_iter3 = 1


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

    at = bulk("Cu", cubic=True).repeat(4)
    calc = EMT()
    T = 400
    P = 0
    nsteps = 10
    parameters = {"twojmax": 6}
    descriptor = SnapDescriptor(at, parameters=parameters)
    parameters = {"solver": "L-BFGS-B"}
    mlip_mbar = LinearPotential(descriptor)
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
        assert vol_unit == 'Ang^3'
        etotal_var = ncfile['etotal']
        etotal_unit = etotal_var.unit
        assert etotal_unit == 'eV'


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

    at = bulk("Cu", cubic=True).repeat(4)
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

    at = bulk("Cu", cubic=True).repeat(4)
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
