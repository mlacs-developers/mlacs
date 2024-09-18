import pytest
from pathlib import Path
import shutil, os
import netCDF4 as nc
import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT

from mlacs.mlip import SnapDescriptor, LinearPotential
from mlacs.state import LammpsState
from mlacs.properties import CalcExecFunction
from mlacs import OtfMlacs

from ... import context  # noqa

@pytest.mark.skipif(context.has_netcdf(),
                    reason="You need the netCDF4 package to run the test.")
def test_hist():
    """
    Check that a '*HIST_nc' file is created, and that it stores some basic 
    thermodynamic variables.
    """
    at = bulk("Cu", cubic=True).repeat(4)
    calc = EMT()
    T = 400
    P = 0
    nsteps = 10
    nb_states = 3
    states = list(LammpsState(T, P, nsteps=nsteps) for i in range(nb_states))
    parameters = {"twojmax": 6}
    descriptor = SnapDescriptor(at, parameters=parameters)
    parameters = {"solver": "L-BFGS-B"}
    mlip_mbar = LinearPotential(descriptor)

    properties = [CalcExecFunction('get_kinetic_energy')]

    
    pytest_path = os.getenv('PYTEST_CURRENT_TEST')
    test_wkdir = Path(pytest_path).parents[0].absolute() / 'tmp_data'    

    dyn = OtfMlacs(at,
                   states,
                   calc,
                   mlip_mbar,
                   properties,
                   neq=0,
                   workdir=test_wkdir,
                   ncprefix='obj1',
                   ncformat='NETCDF4')
    nb_mlacs_iter = 3
    dyn.run(nb_mlacs_iter)
    
    # Check that file exists
    script_name = 'obj1_' + 'test_hist_file'
    script_name += '_HIST.nc'
    path_name = test_wkdir / script_name
    assert path_name.is_file()
    
    # Check that some basic properties are stored, and have proper features
    with nc.Dataset(str(path_name), 'r') as ncfile:
        vol_var = ncfile['vol']
        vol_data = vol_var[:].data
        vol_unit = vol_var.unit
        assert isinstance(vol_var[:], np.ma.MaskedArray)
        assert isinstance(vol_data, np.ndarray)
        assert vol_data.shape == (nb_states*(nb_mlacs_iter-1),)
        assert vol_unit == 'Ang^3'

    # Clean-up
    shutil.rmtree(test_wkdir)

