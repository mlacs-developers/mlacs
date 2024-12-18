"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import os
import sys
from pathlib import Path
from subprocess import check_output

import numpy as np

from ase.atoms import Atoms

from mlacs.utilities.miscellanous import create_ASE_object
from mlacs.utilities.units import unit_converter

try:
    from netCDF4 import Dataset
except ImportError as exc:
    raise ImportError("You need netCDF4 to use the AbinitNC class") from exc


# ========================================================================== #
# ========================================================================== #
class MlacsHist:
    """
    Parent class to handle the HIST.nc file created by MLACS.

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
        Absolute path to HIST.nc file, i.e. `path_to_ncfile/ncfilename`.
        Compulsory in case of post-processing usage.
        Default ``None``.

    Examples
    -------
    Post-processing usage:
    ::

        ncfile = MlacsHist(ncpath='/path/to/HIST.nc')
        var_names = ncfile.get_var_names()
        dict_var_units = ncfile.get_units()
        var_dim_dict = ncfile.var_dim_dict
        energy_array = ncfile.read_obs('etotal')

    Conversion of a list of ASE Atoms into HIST.nc:
    ::

        ncprefix = 'my_ncprefix'
        workdir = '/path/to/workdir/'
        obj = MlacsHist(ncprefix=ncprefix,
                        workdir=workdir,
                        atoms=list_of_ase_atoms)
        obj.convert_to_hist()

    The HIST.nc is saved in workdir with name f'{ncprefix}_HIST.nc'.
    """

    def __init__(self,
                 ncprefix='',
                 workdir='',
                 ncformat='NETCDF3_CLASSIC',
                 launched=True,
                 atoms=None,
                 ncpath=None):

        self.ncprefix = ncprefix
        self.workdir = workdir
        self.ncformat = ncformat
        self.launched = launched
        self.atoms = atoms
        self._set_name_conventions()
        self._set_unit_conventions()

        if ncpath:
            self.ncpath = ncpath
            self.workdir = Path(ncpath).parents[0]

            # Specific to post-processing usage
            if Path(ncpath).is_file():
                with Dataset(str(ncpath), 'r') as ncfile:
                    self.ncformat = ncfile.file_format

# ========================================================================== #
    def _initialize_nc_file(self, atoms):
        """
        Initialize netCDF file:
            - Create Abinit-style dimensions.
            - Create Abinit-style variables that are not 'CalcProperty'. The
            latter are typically static structural data obeying Abinit naming
            conventions. Some of these variables (namely: znucl, typat, amu,
            dtion) are initialized here.

        Parameters
        ----------

        atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
            An example of the atom object produced by the simulation. It is
            assumed the all ase.Atoms` have the same  `typat`, `znucl`, etc.
            This excludes, e.g., grand-canonical computations.
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

# ========================================================================== #
    def _add_dim(self, ncfilepath, dict_dim, mode='r+'):
        with Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for dim_name, dim_value in dict_dim.items():
                new.createDimension(dim_name, (dim_value))

# ========================================================================== #
    def _add_var(self, ncfilepath, dict_var, mode='r+', datatype='float64'):
        with Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for var_name, var_dim in dict_var.items():
                new.createVariable(var_name, datatype, var_dim)

# ========================================================================== #
    def _initialize_var(self, ncfilepath, dict_initialize_var, mode='r+'):
        with Dataset(ncfilepath, mode, format=self.ncformat) as new:
            for var_name, var_value in dict_initialize_var.items():
                new[var_name][:] = var_value

# ========================================================================== #
    def create_nc_var(self, prop_list):
        """
        Create Abinit-style variables in netCDF file from `prop_list`.

        Parameters
        -------

        prop_list: :class:`list` of :class: `CalcProperty`
            Cf. mlacs.properties.calc_property.py
        """
        datatype = 'float64'
        if prop_list is not None:
            for obs in prop_list:
                nc_name = obs.nc_name
                nc_dim = obs.nc_dim
                # Observables need these attributes to get saved in the HIST.nc
                if None not in (nc_name, nc_dim):
                    with Dataset(self.ncpath, 'a') as ncfile:
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
        with Dataset(self.ncpath, 'r') as ncfile:
            observable_values = ncfile[obs_name][:].data
        return observable_values

# ========================================================================== #
    def read_weighted_obs(self, obs_name):
        """
        Read specific weighted observable from netcdf file.
        Return values, idx in database
        """
        with Dataset(self.ncpath, 'r') as ncfile:
            wobs_values = ncfile[obs_name][:]
            weighted_obs_data = wobs_values[wobs_values.mask == False].data  # noqa
            weighted_obs_idx = 1 + np.where(~wobs_values.mask)[0]
        return weighted_obs_data, weighted_obs_idx

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
        # XXX : This should be moved to utilities.units.py
        # Dict whose keys are 'var names' and values are ASE units
        ase_units_dict = {'Total_Energy': 'eV',
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
        self.ase_units_dict = ase_units_dict

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
    def read_all(self):
        """Read all observables from netcdf file"""
        res = {}
        with Dataset(self.ncpath, 'r') as ncfile:
            for name, variable in ncfile.variables.items():
                res[name] = variable[:]
        return res

# ========================================================================== #
    def get_var_names(self):
        """Return list of all observable names"""
        return list(self.read_all().keys())

# ========================================================================== #
    def get_units(self):
        """
        Return dict where keys are obs. names and values are units.
        Variables without units do not appear.
        """
        res = {}
        with Dataset(self.ncpath, 'r') as ncfile:
            for name, variable in ncfile.variables.items():
                if hasattr(variable, 'unit'):
                    res[name] = variable.unit
        return res

# ========================================================================== #
    def convert_to_hist(self):
        """
        Convert list of Ase's Atoms into HIST.nc respecting Abinit conventions.
        """

        # Ensure `self.atoms` is a list
        if isinstance(self.atoms, Atoms):
            atoms_list = [self.atoms]
        atoms_list = self.atoms

        # Initialize target HIST.nc file
        ncname = self.ncprefix + "_HIST.nc"
        ncpath = Path(self.workdir) / ncname
        self.ncpath = ncpath
        self._initialize_nc_file(atoms_list[0])

        # Initialize list of properties
        # TODO: Fix circular dependency of RoutinePropertyManager import
        from mlacs.properties import RoutinePropertyManager
        rout_prop_obj = RoutinePropertyManager(self, False)

        # Emulate Mlas run
        for idx, at in enumerate(atoms_list):
            rout_prop_obj.calc_initialize(atoms=[at])
            step = idx + 1
            rout_prop_obj.run(step)
            rout_prop_obj.save_prop(step)


# ========================================================================== #
# ========================================================================== #
class OtfMlacsHist(MlacsHist):
    """
    Handle HIST.nc file during the on-the-fly execution of MLACS.

    This class is thought as a low-level, internal object used only during the
    MLACS cycle, through the `mlas` object. It is tailored to save 'Properties'
    'RoutineProperties', weights and metadata, in a common HIST.nc file.

    Notes
    ----------

    The Hist name format is: `ncprefix + scriptname + '_HIST.nc'.`

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
    """

    def __init__(self,
                 ncprefix='',
                 workdir='',
                 ncformat='NETCDF3_CLASSIC',
                 launched=True,
                 atoms=None):
        MlacsHist.__init__(self,
                           ncprefix=ncprefix,
                           workdir=workdir,
                           ncformat=ncformat,
                           launched=launched,
                           atoms=atoms,
                           ncpath=None)
        self.ncformat = ncformat
        self.workdir = workdir
        self.ncpath = self._get_nc_path()
        # Check if there is only one type of ase.Atoms objects
        all_chem_symb = [at.get_chemical_formula() for at in atoms]
        self.unique_atoms_type = len(set(all_chem_symb)) == 1
        if not os.path.isfile(self.ncpath) and self.unique_atoms_type:
            self._initialize_nc_file(atoms)
            self._create_weights_var_dim()

# ========================================================================== #
    def _get_nc_path(self):
        """Return netcdf path of Abinit-style HIST file."""
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
    def _create_weights_var_dim(self):
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
# ========================================================================== #
class AbinitNC:
    """
    Class to read all netCDF files created by Abinit.

    Can also convert a netCDF file into a list of ASE Atoms objects, cf.
    convert_to_atoms().

    Parameters
    ----------

    workdir: :class:`str` (optional)
        The root for the directory.
        Default 'DFT'

    prefix: :class:`str` (optional)
        The prefix of Abinit nc file, i.e. str before o_{suffix}.nc.

    suffix: :class:`str` (optional)
        'HIST' or 'GSR' or 'OUT'.
        If no suffix is given, it is set automatically by _update_suffix().

    Examples
    -------
    netCDF files can be loaded by defining the `ncfile` attribute, here for
    instance with a HIST.nc:
    ::

       hist = AbinitNC()
       hist.ncfile = '/path/to/HIST.nc'
       atoms_list = hist.convert_to_atoms()

    Alternatively, one can directly pass workdir, prefix, suffix as parameters
    of the AbinitNC class. The path of the ncfile must follow the syntax
    f'{workdir}{prefix}o_{suffix}.nc'. For instance with a GSR.nc:
    ::

        gsr = AbinitNC(workdir=workdir, prefix=prefix, suffix='GSR')
        atoms_list = gsr.convert_to_atoms()

    Once the netCDF files are loaded, the data can be extracted as a dictionary
    where keys are Abinit variable names and values are arrays in atomic units:
    ::

        hist_results = hist.read()
        energy = hist_results['etotal']  # Atomic units

    Alternatively, conversion to a list of ASE atoms can also be achieved in
    the following way:
    ::

        hist_atoms_list = hist.convert_to_atoms()  # ASE units
        gsr_atoms_list = gsr.convert_to_atoms()  # ASE units
    """

    def __init__(self, workdir=None, prefix='abinit', suffix=None):

        self.workdir = workdir
        if self.workdir is None:
            self.workdir = os.getcwd() + "/DFT/"
        if self.workdir[-1] != "/":
            self.workdir += "/"
        if not os.path.exists(self.workdir):
            self.workdir = ''

        self.ncfile = f'{self.workdir}{prefix}o_{suffix}.nc'
        self.results = {}
        self.atoms_list = []
        self.suffix = suffix

# ========================================================================== #
    def read(self, filename=None):
        """
        Read NetCDF output of Abinit from netCDF4 library.

        Parameters
        ----------

        filename: :class:`str` (optional)
            The name of netCDF file.

        Returns
        -------

        self.results: :class:`dict`
            The dictionary mapping netCDF variables names to their values.
        """

        if filename is not None:
            self.ncfile = filename
            self.dataset = Dataset(filename)
        elif filename is None and hasattr(self, 'ncfile'):
            self.dataset = Dataset(self.ncfile)
        else:
            raise FileNotFoundError('No NetCDF file defined')

        self._keyvar = list(self.dataset.variables)
        self._defattr()
        if not hasattr(self, 'results'):
            self.results = {}
        self.results.update(vars(self))
        return self.results

# ========================================================================== #
    def ncdump(self, filename=None) -> str:
        """
        Read netCDF output of Abinit from command-line ncdump tool.

        Parameters
        ----------

        filename: :class:`str` (optional)
            The name of netCDF file.

        Returns
        -------

        :class:`str` (optional)
            The output of the `ncdump` command.
        """
        return check_output(['ncdump', filename])

# ========================================================================== #
    def _defattr(self):
        for attr in self._keyvar:
            setattr(self, attr, self._extractattr(attr))
        for attr in self._keyvar:
            value = getattr(self, attr)
            if isinstance(value, (int, float, str)):
                continue
            if isinstance(value, np.ndarray):
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
        last_lammps_line = 'chkexit 1 # abinit.exit file in the running directory'  # noqa
        last_lammps_line += ' terminates after the current SCF'
        if last_lammps_line in _str[-len(last_lammps_line):]:
            return True
        large_blank = ' '*80
        if large_blank == _str[-len(large_blank):]:
            return True
        if consecutive_empty_spaces == 80:
            return True
        return False

# ========================================================================== #
    def _decodeattr(self, value) -> str:
        _str = ''
        consec_empty_spaces = 0
        for s in value.tolist():
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
        if 1 == len(value.shape):
            return self._decodeattr(value)
        return np.r_[[self._decodeattr(v) for v in value]]

# ========================================================================== #
    def _update_suffix(self):
        """Initialize suffix when it has not been passed explicitely."""
        if 'HIST.nc' in self.ncfile:
            self.suffix = 'HIST'
        elif 'OUT.nc' in self.ncfile:
            self.suffix = 'OUT'
        elif 'GSR.nc' in self.ncfile:
            self.suffix = 'GSR'

# ========================================================================== #
    def _extract_out_data(self, results):
        """
        Extract data from Abinit's OUT.nc.
        """
        typat = results['typat'].astype(int)
        znucl = results['znucl'].astype(int)
        atomic_numbers = znucl[typat-1]
        nat = len(typat)
        rprim = np.reshape(results['rprim'], (3, 3))
        cell = results['acell'] * rprim
        positions = np.reshape(results['xcart'], (nat, 3))
        structural_data = [atomic_numbers, positions, cell]

        energy = results['etotal']
        forces = results['fcart']
        stress = results['strten']
        main_properties = [energy, forces, stress]

        free_energy = None
        if 'spinat' in results:
            spinat = results['spinat'].reshape((nat, 3))
        else:
            spinat = None
        others = [free_energy, spinat]

        return structural_data, main_properties, others

# ========================================================================== #
    def _extract_gsr_data(self, results):
        """
        Extract data from Abinit's GSR.nc.
        """
        typat = results['atom_species'].astype(int)
        znucl = results['atomic_numbers'].astype(int)
        atomic_numbers = znucl[typat-1]
        cell = results['primitive_vectors']
        positions = results['reduced_atom_positions'] @ cell
        structural_data = [atomic_numbers, positions, cell]

        energy = results['etotal']
        forces = results['cartesian_forces']
        stress = results['cartesian_stress_tensor']
        main_properties = [energy, forces, stress]

        free_energy = results['entropy']
        if 'spinat' in results:
            spinat = results['spinat']
        else:
            spinat = None
        others = [free_energy, spinat]

        return structural_data, main_properties, others

# ========================================================================== #
    def _extract_hist_data(self, results):
        """
        Extract data from Abinit's HIST.nc.
        """
        typat = results['typat'].astype(int)
        znucl = results['znucl'].astype(int)
        atomic_numbers = znucl[typat-1]
        cell = results['rprimd']
        positions = results['xcart']
        structural_data = [atomic_numbers, positions, cell]

        energy = results['etotal']
        forces = results['fcart']
        stress = results['strten']
        main_properties = [energy, forces, stress]

        free_energy = None
        spinat = None
        others = [free_energy, spinat]

        return structural_data, main_properties, others

# ========================================================================== #
    def convert_to_atoms(self):
        """
        Convert Abinit *.nc file to a list of ASE Atoms objects.

        Notes
        ----------

        In the process, atomic units (Abinit) are converted to ASE units.

        Returns
        -------

        atoms_list: :class:`list` of :class:`ase.Atoms`
            The list of configurations in ASE format.
        """

        # Ensure Abinit *.nc file has been read
        if not self.results:
            self.read()
        res_dict = self.results

        self._update_suffix()
        if self.suffix == 'HIST':
            struct_data, main_prop, others = self._extract_hist_data(res_dict)
        elif self.suffix == 'OUT':
            struct_data, main_prop, others = self._extract_out_data(res_dict)
        elif self.suffix == 'GSR':
            struct_data, main_prop, others = self._extract_gsr_data(res_dict)

        atomic_numbers, positions, cell = struct_data
        energy, forces, stress = main_prop
        free_energy, spinat = others
        if spinat is None:
            spinat = np.zeros((len(atomic_numbers), 3))

        # Add new axis to arrays to uniformize shapes with 'HIST'
        # such that array.shape becomes (nb_confs, array.shape)
        if self.suffix in ['OUT', 'GSR']:
            positions = np.expand_dims(positions, axis=0)
            cell = np.expand_dims(cell, axis=0)
            energy = np.expand_dims(energy, axis=0)
            forces = np.expand_dims(forces, axis=0)
            stress = np.expand_dims(stress, axis=0)

        # If Abinit nimage > 1, only the result of the first image is kept
        # XXX CD: in PIMD∕NEB calculations, this will need to be adapted
        if cell[0].shape != (3, 3):
            positions = positions[:, 0, ...]
            cell = cell[:, 0, ...]
            energy = energy[:, 0, ...]
            forces = forces[:, 0, ...]
            stress = stress[:, 0, ...]

        nb_confs = len(energy)
        atoms_list = []
        for i in range(nb_confs):
            # Convert atomic units to ASE units
            conv_p = unit_converter(positions[i], 'Bohr', target='ASE')[0]
            conv_c = unit_converter(cell[i], 'Bohr', target='ASE')[0]
            conv_e = unit_converter(energy[i], 'Ha', target='ASE')[0]
            conv_f = unit_converter(forces[i], 'Ha/Bohr', target='ASE')[0]
            conv_s = unit_converter(stress[i], 'Ha/Bohr^3', target='ASE')[0]

            atoms = create_ASE_object(atomic_numbers,
                                      positions=conv_p,
                                      cell=conv_c,
                                      energy=conv_e,
                                      forces=conv_f,
                                      stresses=conv_s)
            atoms.set_array('spinat', spinat)
            atoms_list.append(atoms)

        self.atoms_list = atoms_list
        return atoms_list
