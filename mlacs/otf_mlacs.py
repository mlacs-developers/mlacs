"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import os
import sys
import netCDF4 as nc4

from .mlas import Mlas
from .core import Manager
from .properties import (PropertyManager,
                         CalcRoutineFunction,
                         CalcPressure,
                         CalcAcell,
                         CalcAngles)


# ========================================================================== #
# ========================================================================== #
class OtfMlacs(Mlas, Manager):
    """
    A Learn on-the-fly simulation constructed in order to sample approximate
    distribution

    Parameters
    ----------

    atoms: :class:`ase.Atoms` or :class:`list` of :class:`ase.Atoms`
        the atom object on which the simulation is run.

    state: :class:`StateManager` or :class:`list` of :class:`StateManager`
        Object determining the state to be sampled

    calc: :class:`ase.calculators` or :class:`CalcManager`
        Class controlling the potential energy of the system
        to be approximated.
        If a :class:`ase.calculators` is attached, the :class:`CalcManager`
        is automatically created.

    mlip: :class:`MlipManager` (optional)
        Object managing the MLIP to approximate the real distribution
        Default is a LammpsMlip object with a snap descriptor,
        ``5.0`` angstrom rcut with ``8`` twojmax.

    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.

    workdir: :class:`str` (optional)
        The directory in which to run the calculation.

    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms` (optional)
        If :class:`int`, Number of configurations used to train a preliminary
        MLIP. The configurations are created by rattling the first structure.
        If :class:`list` of :class:`ase.Atoms`, The atoms that are to be
        computed in order to create the initial training configurations.
        Default ``1``.

    std_init: :class:`float` (optional)
        Variance (in :math:`\mathring{a}^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 \mathring{a}^2`

    keep_tmp_mlip: :class:`bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``False``.

    ntrymax: :class:`int` (optional)
        The maximum number of tentative to retry a step if
        the reference potential raises an error or didn't converge.
        Default ``0``.
    """

    def __init__(self,
                 atoms,
                 state,
                 calc,
                 mlip=None,
                 prop=None,
                 neq=10,
                 confs_init=None,
                 std_init=0.05,
                 keep_tmp_mlip=True,
                 ntrymax=0,
                 workdir='',
                 ncprefix='',
                 ncformat='NETCDF3_CLASSIC'):

        Mlas.__init__(self, atoms, state, calc, mlip=mlip, prop=None, neq=neq,
                      confs_init=confs_init, std_init=std_init,
                      ntrymax=ntrymax, keep_tmp_mlip=keep_tmp_mlip,
                      workdir=workdir)

        self.ncprefix = ncprefix
        self.ncformat = ncformat

        # Check if trajectory files already exist
        self.launched = self._check_if_launched()

        ncpath = self._get_nc_path()
        if not os.path.isfile(ncpath):
            self._create_nc_file(ncpath, atoms)

        self._initialize_properties(prop, ncpath)
        self._initialize_routine_properties(ncpath)

# ========================================================================== #
    def _get_nc_path(self):
        """Return netcdf path"""
        if self.ncprefix != '' and (not self.ncprefix.endswith('_')):
            self.ncprefix += '_'
        script_name = self.ncprefix
        script_name += os.path.basename(sys.argv[0])
        if script_name.endswith('.py'):
            script_name = script_name[:-3]
        ncname = script_name + "_HIST.nc"
        ncpath = str(self.workdir / ncname)

        ncfile_exists = os.path.isfile(ncpath)
        if ncfile_exists:
            # if it is the first MLAS launch
            if not self.launched:
                S = 1
                while ncfile_exists:
                    ncname = script_name + '_{:04d}'.format(S) + "_HIST.nc"
                    ncpath = str(self.workdir / ncname)
                    ncfile_exists = os.path.isfile(ncpath)
                    S += 1

        return ncpath

# ========================================================================== #
    def _create_nc_file(self, ncpath, atoms):
        """
        Create netcdf file(s).

        Create Abinit-style dimensions
        Create Abinit-style variables that are not 'RoutineProperties'
        """
        
        def _core(ncpath, mode, f, dict_dim, dict_var):
            with nc4.Dataset(ncpath, mode, format=f) as new:
                for dim_name, dim_value in dict_dim.items():
                    new.createDimension(dim_name, (dim_value))
                for var_name, var_dim in dict_var.items():
                    new.createVariable(var_name, datatype, var_dim)

        datatype = 'float64'
        
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

        _core(ncpath, 'r+', self.ncformat, dict_dim, dict_var)

        dict_w_dim = {'weights_dim': None}
        dict_w_var = {'weights': ('weights_dim',),
                      'weights_meta': ('weights_dim',),
                      }

        weights_ncpath = ncpath
        if 'NETCDF3' in self.ncformat:
            weights_ncpath = ncpath.replace('HIST', 'WEIGHTS')

        _core(weights_ncpath, 'r+', self.ncformat, dict_w_dim, dict_w_var)


# ========================================================================== #
    def _initialize_properties(self, prop, ncpath):
        """Create property object"""
        if prop is None:
            self.prop = PropertyManager(None)
        elif isinstance(prop, PropertyManager):
            self.prop = prop
        else:
            self.prop = PropertyManager(prop)

        self.prop.workdir = self.workdir
        if not self.prop.folder:
            self.prop.folder = 'Properties'

        self.prop.isfirstlaunched = not self.launched
        self.prop.ncpath = ncpath

# ========================================================================== #
    def _initialize_routine_properties(self, ncpath):
        """Create routine property object"""

        label_dict = {'Total_Energy': ['etotal', ('time',)],
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

        routine_prop_list = []
        for x in label_dict:
            var_name, var_dim = label_dict[x]
            lammps_func = 'get_' + x.lower()
            observable = CalcRoutineFunction(lammps_func,
                                             label=x,
                                             nc_name=var_name,
                                             nc_dim=var_dim,
                                             frequence=1)
            routine_prop_list.append(observable)

        other_observables = [CalcPressure(label='Pressure',
                                          nc_name='press',
                                          nc_dim=('time',),
                                          frequence=1),
                             CalcAcell(label='Acell',
                                       nc_name='acell',
                                       nc_dim=('time', 'xyz'),
                                       frequence=1),
                             CalcAngles(label='Angles',
                                        nc_name='angl',
                                        nc_dim=('time', 'xyz'),
                                        frequence=1),
                             ]
        routine_prop_list += other_observables

        if not self.launched:
            # Create Abinit-style variables in netcdf file
            datatype = 'float64'
            for obs in routine_prop_list:
                with nc4.Dataset(ncpath, 'a') as new:
                    new.createVariable(obs.nc_name, datatype, obs.nc_dim)
                    meta_dim = ('time', 'two',)
                    meta_name = obs.nc_name + '_meta'
                    new.createVariable(meta_name, datatype, meta_dim)
                    w_name = 'weighted_' + obs.nc_name
                    new.createVariable(w_name, datatype, obs.nc_dim)

        self.routine_prop = PropertyManager(routine_prop_list)
        self.routine_prop.workdir = self.workdir
        self.routine_prop.folder = 'Properties/RoutineProperties'

        self.routine_prop.isfirstlaunched = not self.launched
        self.routine_prop.ncpath = ncpath

# ========================================================================== #
    def _compute_properties(self):
        """

        """
        if self.prop.manager is not None:
            self.prop.calc_initialize(atoms=self.atoms)
            msg = self.prop.run(self.step)
            self.log.logger_log.info(msg)
            if self.prop.check_criterion:
                msg = "All property calculations are converged, " + \
                      "stopping MLACS ...\n"
                self.log.logger_log.info(msg)

        # Compute routine properties
        self.routine_prop.calc_initialize(atoms=self.atoms)
        msg = self.routine_prop.run(self.step)
        self.log.logger_log.info(msg)
        self.routine_prop.save_prop(self.step)
        self.routine_prop.save_weighted_prop(self.step, self.mlip.weight)
        w_first2 = self.routine_prop.save_weights(self.step,
                                                  self.mlip.weight,
                                                  self.ncformat)

        # Monitor the weight of first two confs, that should be (close to) zero
        msg = "Weight of first two configurations:" + f"{w_first2:20.15f}"
        self.log.logger_log.info(msg)
        if w_first2 > 10**-3 and w_first2 != 1.0:
            w_msg = "\t"*2 + " WARNING: This value may be abnormally high.\n"
            self.log.logger_log.warning(w_msg)
        else:
            self.log.logger_log.info("")
        
        
