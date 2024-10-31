"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
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
    if 'spinat' in results.keys():
        atoms.set_array('spinat', results['spinat'].reshape((nat, 3)))
    else:
        atoms.set_array('spinat', np.zeros((nat, 3)))
    atoms.calc = calc
    return atoms


def _set_from_gsrNC(results=dict()) -> Atoms:
    """Read results from GSR.nc"""
    Z = np.r_[[results['atomic_numbers'][i - 1]
               for i in results['atom_species']]]
    nat = len(results['atom_species'])
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
    if 'spinat' in results.keys():
        atoms.set_array('spinat', results['spinat'])
    else:
        atoms.set_array('spinat', np.zeros((nat, 3)))
    atoms.calc = calc
    return atoms


# ========================================================================== #
# ========================================================================== #
class HistFile:
    """
    Class to handle Abinit-like *HIST.nc file.
    The script name format is: `ncprefix + scriptname + '_HIST.nc'.`

    Parameters
    ----------

    ncprefix: :class:`str` (optional)
        The prefix to prepend the name of the *HIST.nc file.
        Default `''`.

    workdir: :class:`str` (optional)
        The directory in which to run the calculation.

    ncformat: :class:`str` (optional)
        The format of the *HIST.nc file. One of the five flavors of netCDF
        files format available in netCDF4 python package: 'NETCDF3_CLASSIC',
        'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA','NETCDF4_CLASSIC',
        'NETCDF4'.
        Default ``NETCDF3_CLASSIC``.

    launched: :class:`Bool` (optional)
        If True then is not the first MLACS start of the related Mlas instance,
        i.e. it is a restart situation for which a *HIST.nc already exists.
        Default ``True``.

    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms` (optional)
        the atom object on which the simulation is run.
        Default ``None``.

    ncpath: :class:`str` or :class:`Path` of `pathlib` module (optional)
        Absolute path to *HIST.nc file, i.e. `path_to_ncfile/ncfilename`.
        Must be given in a postprocessing use, otherwise defined by the
        _get_nc_path method.
        Default ``None``.
    """

    def __init__(self,
                 ncprefix='',
                 workdir='',
                 ncformat='NETCDF3_CLASSIC',
                 launched=True,
                 atoms=None,
                 ncpath=None):

        if nc is None:
            raise ImportError("You need Netcdf4 to use the AbinitNC class")

        self.ncprefix = ncprefix
        self.launched = launched
        self._set_name_conventions()
        self._set_unit_conventions()

        if ncpath is not None:
            # Initialize path in a post-processing usage, with given ncpath
            self.ncpath = ncpath
            self.workdir = Path(ncpath).parents[0]
            with nc.Dataset(str(ncpath), 'r') as ncfile:
                self.ncformat = ncfile.file_format
        else:
            # Initialize path during MLACS execution:
            # Compute the path itself, then create file if it doesn't exist
            self.ncformat = ncformat
            self.workdir = workdir
            self.ncpath = self._get_nc_path()
            if not os.path.isfile(self.ncpath):
                self._create_nc_file(atoms)

# ========================================================================== #
    def _get_nc_path(self):
        """Return netcdf path of Abinit-style HIST file"""
        ncpref = self.ncprefix
        wdir = self.workdir
        if ncpref and not ncpref.endswith('_'):
            ncpref += '_'

        script_name = ncpref
        pytest_path = os.getenv('PYTEST_CURRENT_TEST')
        if pytest_path:
            wdir = wdir or Path(pytest_path).parents[0].absolute()
            script_name += Path(pytest_path).stem
        else:
            script_name += Path(sys.argv[0]).stem
        ncname = script_name + "_HIST.nc"

        # Deal with potential duplicates, i.e. existing files with ncname
        ncpath = Path(wdir).absolute() / ncname
        if ncpath.is_file():
            if not self.launched:   # if first MLAS launch
                suffix = 1
                while ncpath.is_file():
                    ncname = f"{script_name}_{suffix:04d}_HIST.nc"
                    ncpath = Path(wdir) / ncname
                    suffix += 1

        return str(ncpath)

# ========================================================================== #
    def _create_nc_file(self, atoms):
        """
        Create netcdf file(s).
        Create Abinit-style dimensions.
        Create Abinit-style variables that are not 'CalcProperty' objects:
        The latter are typically static structural data obeying Abinit naming
        conventions. Some of these variables (namely: znucl, typat, amu, dtion)
        are initialized here.
        """

        # Assume ntypat, natom, etc., are the same for all items in list
        if isinstance(atoms, list):
            atoms = atoms[0]
        atomic_numbers = list(atoms.get_atomic_numbers())
        atomic_masses = list(atoms.get_masses())
        natom = len(atoms)
        znucl = sorted(set(atomic_numbers), key=atomic_numbers.index)
        amu = sorted(set(atomic_masses), key=atomic_masses.index)
        ntypat = len(znucl)
        typat = [1+znucl.index(x) for x in atomic_numbers]
        # dtion is not well-defined in MLACS. Set to one below by convention.
        dtion = 1.0

        dict_dim = {'time': None,
                    'one': 1,
                    'two': 2,
                    'xyz': 3,
                    'npsp': 3,
                    'six': 6,
                    'ntypat': ntypat,
                    'natom': natom,
                    }
        dict_var = {'typat': ('natom',),
                    'znucl': ('ntypat',),
                    'amu': ('ntypat',),
                    'dtion': ('one',),
                    'mdtemp': ('two',),
                    'mdtime': ('time',),
                    }
        dict_initialize_var = {'typat': typat,
                               'znucl': znucl,
                               'amu': amu,
                               'dtion': dtion,
                               }

        self._add_dim(self.ncpath, dict_dim)
        self._add_var(self.ncpath, dict_var)
        self._initialize_var(self.ncpath, dict_initialize_var)

        dict_w_dim = {'weights_dim': None}
        dict_w_var = {'weights': ('weights_dim',),
                      'weights_meta': ('weights_dim', 'xyz',),
                      }

        self.weights_ncpath = self.ncpath
        if 'NETCDF3' in self.ncformat:
            dict_w_dim['xyz'] = 3
            self.weights_ncpath = self.ncpath.replace('HIST', 'WEIGHTS')

        self._add_dim(self.weights_ncpath, dict_w_dim)
        self._add_var(self.weights_ncpath, dict_w_var)

# ========================================================================== #
    def _add_dim(self, ncfilepath, dict_dim, mode='r+'):
        with nc.Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for dim_name, dim_value in dict_dim.items():
                new.createDimension(dim_name, (dim_value))

# ========================================================================== #
    def _add_var(self, ncfilepath, dict_var, mode='r+', datatype='float64'):
        with nc.Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for var_name, var_dim in dict_var.items():
                new.createVariable(var_name, datatype, var_dim)

# ========================================================================== #
    def _initialize_var(self, ncfilepath, dict_initialize_var, mode='r+'):
        with nc.Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for var_name, var_value in dict_initialize_var.items():
                new[var_name][:] = var_value

# ========================================================================== #
    def create_nc_var(self, prop_list):
        """Create Abinit-style variables in netcdf file"""
        datatype = 'float64'
        if prop_list is not None:
            for obs in prop_list:
                nc_name = obs.nc_name
                nc_dim = obs.nc_dim
                # Observables need these attributes to get saved in the HIST.nc
                if None not in (nc_name, nc_dim):
                    with nc.Dataset(self.ncpath, 'a') as ncfile:
                        var = ncfile.createVariable(nc_name, datatype, nc_dim)
                        var.setncattr('unit', obs.nc_unit)
                        meta_dim = ('time', 'two',)
                        meta_name = nc_name + '_meta'
                        ncfile.createVariable(meta_name, datatype, meta_dim)
                        w_name = 'weighted_' + nc_name
                        wvar = ncfile.createVariable(w_name, datatype, nc_dim)
                        wvar.setncattr('unit', obs.nc_unit)

# ========================================================================== #
    def read_obs(self, obs_name):
        """Read specific observable from netcdf file"""
        with nc.Dataset(self.ncpath, 'r') as ncfile:
            observable_values = ncfile[obs_name][:].data
        return observable_values

# ========================================================================== #
    def read_weighted_obs(self, obs_name):
        """
        Read specific weighted observable from netcdf file.
        Return values, idx in database
        """
        with nc.Dataset(self.ncpath, 'r') as ncfile:
            wobs_values = ncfile[obs_name][:]
            weighted_obs_data = wobs_values[wobs_values.mask == False].data  # noqa
            weighted_obs_idx = 1 + np.where(~wobs_values.mask)[0]
        return weighted_obs_data, weighted_obs_idx

# ========================================================================== #
    def read_all(self):
        """Read all observables from netcdf file"""
        res = {}
        with nc.Dataset(self.ncpath, 'r') as ncfile:
            for name, variable in ncfile.variables.items():
                res[name] = variable[:]
        return res

# ========================================================================== #
    def get_var_names(self):
        """Return list of all observable names"""
        res = []
        with nc.Dataset(self.ncpath, 'r') as ncfile:
            for name, variable in ncfile.variables.items():
                res.append(name)
        return res

# ========================================================================== #
    def get_units(self):
        """
        Return dict where keys are obs. names and values are units.
        Variables without units do not appear.
        """
        res = {}
        with nc.Dataset(self.ncpath, 'r') as ncfile:
            for name, variable in ncfile.variables.items():
                if hasattr(variable, 'unit'):
                    res[name] = variable.unit
        return res

# ========================================================================== #
    def _set_name_conventions(self):
        """Define naming conventions related to routine properties"""
        # Variable names and dimensions as defined in Abinit
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
        self.var_dim_dict = var_dim_dict

# ========================================================================== #
    def _set_unit_conventions(self):
        """Define unit conventions related to routine properties"""
        # Dict whose keys are 'var names' and values are LAMMPS's 'metal' units
        lammps_units_dict = {'Total_Energy': 'eV',
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
        self.lammps_units_dict = lammps_units_dict

        # Dict whose keys are 'var names' and values are Abinit units
        abinit_units_dict = {'Total_Energy': 'Ha',
                             'Kinetic_Energy': 'Ha',
                             'Potential_Energy': 'Ha',
                             'Velocities': '',
                             'Forces': 'Ha/Bohr',
                             'Positions': 'Bohr',
                             'Scaled_Positions': 'dimensionless',
                             'Temperature': 'K',
                             'Volume': 'Bohr^3',
                             'Stress': 'Ha/Bohr^3',
                             'Cell': 'Bohr',
                             }
        self.abinit_units_dict = abinit_units_dict


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
    def _check_end_of_dataset(self, _str, consecutive_empty_spaces):
        """
        Return True if end of dataset has been reached.
        In particular, this function handles the dataset named 'input_string',
        which corresponds to the Abinit input file, but also contains unwanted
        (i.e., not encoded in UTF-8) information at the bottom.
        """
        last_Lammps_line = 'chkexit 1 # abinit.exit file in the running directory'  # noqa
        last_Lammps_line += ' terminates after the current SCF'
        if last_Lammps_line in _str[-len(last_Lammps_line):]:
            return True
        return False
        large_blank = ' '*80
        if large_blank == _str[-len(large_blank):]:
            return True
        if consecutive_empty_spaces == 80:
            return True

# ========================================================================== #
    def _decodeattr(self, value) -> str:
        _str = ''
        consec_empty_spaces = 0
        for idx, s in enumerate(value.tolist()):
            if self._check_end_of_dataset(_str, consec_empty_spaces) is True:
                break
            try:
                _str += bytes.decode(s)
                if len(bytes.decode(s)) == 0:
                    consec_empty_spaces += 1
                else:
                    consec_empty_spaces = 0
            except UnicodeDecodeError:
                # Just to be on the safe side.
                break
        return _str.strip()

# ========================================================================== #
    def _decodearray(self, value):
        if 'S1' != value.dtype:
            return value
        elif 1 == len(value.shape):
            return self._decodeattr(value)
        else:
            return np.r_[[self._decodeattr(v) for v in value]]
