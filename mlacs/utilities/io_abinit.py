"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import sys
import numpy as np
from pathlib import Path

from ase import Atoms
from ase.units import Bohr, Hartree
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc

try:
    import netCDF4 as nc
except ImportError:
    nc = None


def set_aseAtoms(results=dict()) -> Atoms:
    """Read results dictionary to construct Ase Atoms object"""
    if 'reduced_atom_positions' in results.keys():
        return _set_from_gsrNC(results)
    elif 'xred' in results.keys():
        return _set_from_outNC(results)
    else:
        msg = 'No atomic positions found in dictionary'
        raise AttributeError(msg)


def _set_from_outNC(results=dict()) -> Atoms:
    """Read results from OUT.nc"""
    Z = np.r_[[results['znucl'][i - 1] for i in results['typat']]]
    nat = len(results['typat'])
    cell = results['acell'] * np.reshape(results['rprim'], (3, 3)) * Bohr
    atoms = Atoms(numbers=Z,
                  cell=cell,
                  positions=np.reshape(results['xangst'], (nat, 3)),
                  pbc=True)
    calc = SPCalc(atoms,
                  energy=results['etotal'] * Hartree,
                  forces=results['fcart'] * Hartree / Bohr,
                  stress=results['strten'] * Hartree / Bohr**3)
    atoms.calc = calc
    return atoms


def _set_from_gsrNC(results=dict()) -> Atoms:
    """Read results from GSR.nc"""
    Z = np.r_[[results['atomic_numbers'][i - 1]
               for i in results['atom_species']]]
    cell = results['primitive_vectors'] * Bohr
    positions = np.matmul(results['reduced_atom_positions'], cell)
    atoms = Atoms(numbers=Z,
                  cell=cell,
                  positions=positions,
                  pbc=True)
    calc = SPCalc(atoms,
                  energy=results['etotal'] * Hartree,
                  forces=results['cartesian_forces'] * Hartree/Bohr,
                  stress=results['cartesian_stress_tensor'] * Hartree/Bohr**3,
                  free_energy=results['entropy'] * Hartree)
    atoms.calc = calc
    return atoms


def get_nc_path(ncprefix, workdir, launched):
    """Return netcdf path of Abinit-style HIST file"""
    if ncprefix != '' and (not ncprefix.endswith('_')):
        ncprefix += '_'
    script_name = ncprefix
    # Obtain script name, including in case of pytest execution
    pytest_path = os.getenv('PYTEST_CURRENT_TEST')
    if pytest_path is not None:
        if str(workdir) == '':
            workdir = Path(pytest_path).parents[0].absolute()
        name1 = os.path.basename(pytest_path)
        script_name += name1.partition('.py')[0]
    else:
        script_name += os.path.basename(sys.argv[0])

    if script_name.endswith('.py'):
        script_name = script_name[:-3]
    ncname = script_name + "_HIST.nc"
    ncpath = str(Path(workdir) / ncname)

    ncfile_exists = os.path.isfile(ncpath)
    if ncfile_exists:
        # if it is the first MLAS launch
        if not launched:
            S = 1
            while ncfile_exists:
                ncname = script_name + '_{:04d}'.format(S) + "_HIST.nc"
                ncpath = str(Path(workdir) / ncname)
                ncfile_exists = os.path.isfile(ncpath)
                S += 1
    return ncpath


def add_dim(ncpath, dict_dim, ncformat, mode='r+'):
    with nc.Dataset(ncpath, mode, format=ncformat) as new:
        for dim_name, dim_value in dict_dim.items():
            new.createDimension(dim_name, (dim_value))


def add_var(ncpath, dict_var, ncformat, mode='r+', datatype='float64'):
    with nc.Dataset(ncpath, mode, format=ncformat) as new:
        for var_name, var_dim in dict_var.items():
            new.createVariable(var_name, datatype, var_dim)


def create_nc_file(ncpath, ncformat, atoms):
    """
    Create netcdf file(s).

    Create Abinit-style dimensions
    Create Abinit-style variables that are not 'RoutineProperties'
    """

    # Assume ntypat and natom are the same for all items in list
    if isinstance(atoms, list):
        atoms = atoms[0]
    ntypat = len(set(atoms.get_atomic_numbers()))
    natom = len(atoms)

    dict_dim = {'time': None,
                'two': 2,
                'xyz': 3,
                'npsp': 3,
                'six': 6,
                'ntypat': ntypat,
                'natom': natom,
                }
    dict_var = {'typat': ('natom',),
                'znucl': ('npsp',),
                'amu': ('ntypat',),
                'dtion': (),
                'mdtemp': ('two',),
                'mdtime': ('time',),
                }

    add_dim(ncpath, dict_dim, ncformat)
    add_var(ncpath, dict_var, ncformat)

    dict_w_dim = {'weights_dim': None}
    dict_w_var = {'weights': ('weights_dim',),
                  'weights_meta': ('weights_dim',),
                  }

    weights_ncpath = ncpath
    if 'NETCDF3' in ncformat:
        weights_ncpath = ncpath.replace('HIST', 'WEIGHTS')

    add_dim(weights_ncpath, dict_w_dim, ncformat)
    add_var(weights_ncpath, dict_w_var, ncformat)


def create_nc_var(ncpath, prop_list):
    """Create Abinit-style variables in netcdf file"""
    datatype = 'float64'
    if prop_list is not None:
        for obs in prop_list:
            nc_name = obs.nc_name
            nc_dim = obs.nc_dim
            # Observables need these two attributes to get saved in the HIST.nc
            if None not in (nc_name, nc_dim):
                with nc.Dataset(ncpath, 'a') as ncfile:
                    var = ncfile.createVariable(nc_name, datatype, nc_dim)
                    var.setncattr('unit', obs.nc_unit)
                    meta_dim = ('time', 'two',)
                    meta_name = nc_name + '_meta'
                    ncfile.createVariable(meta_name, datatype, meta_dim)
                    w_name = 'weighted_' + nc_name
                    wvar = ncfile.createVariable(w_name, datatype, nc_dim)
                    wvar.setncattr('unit', obs.nc_unit)


def nc_conv():
    """Define several conventions related to *HIST.nc file"""

    # Variable names and dimensions are those produced by Abinit
    var_dim_dict = {'Total_Energy': ['etotal', ('time',)],
                    'Kinetic_Energy': ['ekin', ('time',)],
                    'Potential_Energy': ['epot', ('time',)],
                    'Velocities': ['vel', ('time', 'natom', 'xyz')],
                    'Forces': ['fcart', ('time', 'natom', 'xyz')],
                    'Positions': ['xcart', ('time', 'natom', 'xyz')],
                    'Scaled_Positions': ['xred', ('time', 'natom', 'xyz')],
                    'Temperature': ['temper', ('time',)],
                    'Volume': ['vol', ('time',)],
                    'Stress': ['strten', ('time', 'six')],
                    'Cell': ['rprimd', ('time', 'xyz', 'xyz')]
                    }

    # Units correspond to those used in ASE
    units_dict = {'Total_Energy': 'eV',
                  'Kinetic_Energy': 'eV',
                  'Potential_Energy': 'eV',
                  'Velocities': '',
                  'Forces': 'eV/Ang',
                  'Positions': 'Ang',
                  'Scaled_Positions': 'dimensionless',
                  'Temperature': 'K',
                  'Volume': 'Ang^3',
                  'Stress': 'eV/Ang^3',
                  'Cell': 'Ang',
                  }

    return var_dim_dict, units_dict


# ========================================================================== #
# ========================================================================== #
class AbinitNC:
    """
    Class to handle all Abinit NetCDF files.

    Parameters
    ----------
    workdir: :class:`str` (optional)
        The root for the directory.
        Default 'DFT'
    """
    def __init__(self, workdir=None, prefix='abinit'):

        if nc is None:
            raise ImportError("You need Netcdf4 to use the AbinitNC class")

        self.workdir = workdir
        if self.workdir is None:
            self.workdir = os.getcwd() + "/DFT/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            self.workdir = ''

        self.ncfile = self.workdir + f'{prefix}o_GSR.nc'
        self.results = {}

# ========================================================================== #
    def read(self, filename=None):
        """Read NetCDF output of Abinit"""

        if filename is not None:
            self.dataset = nc.Dataset(filename)
        elif filename is None and hasattr(self, 'ncfile'):
            self.dataset = nc.Dataset(self.ncfile)
        else:
            raise FileNotFoundError('No NetCDF file defined')

        self._keyvar = [_ for _ in self.dataset.variables]
        self._defattr()
        if not hasattr(self, 'results'):
            self.results = {}
        self.results.update(vars(self))
        return self.results

# ========================================================================== #
    def ncdump(self, filename=None) -> str:
        """Read NetCDF output of Abinit"""
        from subprocess import check_output
        return check_output(['ncdump', filename])

# ========================================================================== #
    def _defattr(self):
        for attr in self._keyvar:
            setattr(self, attr, self._extractattr(attr))
        for attr in self._keyvar:
            value = getattr(self, attr)
            if isinstance(value, (int, float, str)):
                continue
            elif isinstance(value, np.ndarray):
                setattr(self, attr, self._decodearray(value))
            elif isinstance(value, memoryview):
                if () == value.shape:
                    setattr(self, attr, value.tolist())
                else:
                    setattr(self, attr, self._decodearray(value.obj))
            else:
                delattr(self, attr)
                msg = f'Unknown object type: {type(value)}\n'
                msg += '-> deleted attribute from AbiAtoms object.\n'
                msg += 'Should be added in the class AbiAtoms, if needed !'
                raise Warning(msg)

# ========================================================================== #
    def _extractattr(self, value):
        return self.dataset[value][:].data

# ========================================================================== #
    def _decodeattr(self, value) -> str:
        # Weird thing to check the end of the file, don't ask ...
        _chk = ''.join([' ' for i in range(80)])
        _str = ''
        for s in value.tolist():
            if _chk == _str[-80:]:
                break
            _str += bytes.decode(s)
        return _str.strip()

# ========================================================================== #
    def _decodearray(self, value):
        if 'S1' != value.dtype:
            return value
        elif 1 == len(value.shape):
            return self._decodeattr(value)
        else:
            return np.r_[[self._decodeattr(v) for v in value]]
