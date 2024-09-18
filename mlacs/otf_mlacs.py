"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""

import os

from .mlas import Mlas
from .core import Manager
from .utilities.io_abinit import (get_nc_path,
                                  create_nc_file,
                                  nc_conv,
                                  create_nc_var)
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
                 ncformat='NETCDF4'):

        Mlas.__init__(self, atoms, state, calc, mlip=mlip, prop=None, neq=neq,
                      confs_init=confs_init, std_init=std_init,
                      ntrymax=ntrymax, keep_tmp_mlip=keep_tmp_mlip,
                      workdir=workdir)

        self.ncprefix = ncprefix
        self.ncformat = ncformat

        # Check if trajectory files already exist
        self.launched = self._check_if_launched()

        # Create Abinit-style netcdf file
        ncpath = get_nc_path(ncprefix, workdir, self.launched)
        if not os.path.isfile(ncpath):
            create_nc_file(ncpath, ncformat, atoms)

        self._initialize_properties(prop, ncpath)
        self._initialize_routine_properties(ncpath)

# ========================================================================== #
    def _initialize_properties(self, prop, ncpath):
        """Create property object"""
        self.prop = PropertyManager(prop)

        if not self.launched:
            create_nc_var(ncpath, prop)

        self.prop.workdir = self.workdir
        if not self.prop.folder:
            self.prop.folder = 'Properties'

        self.prop.isfirstlaunched = not self.launched
        self.prop.ncpath = ncpath

# ========================================================================== #
    def _initialize_routine_properties(self, ncpath):
        """Create routine property object"""

        # Get variables names, dimensions, and units
        var_dim_dict, units_dict = nc_conv()

        # Build a PropertyManager made of "routine" observables
        routine_prop_list = []
        for x in var_dim_dict:
            var_name, var_dim = var_dim_dict[x]
            var_unit = units_dict[x]
            lammps_func = 'get_' + x.lower()
            observable = CalcRoutineFunction(lammps_func,
                                             label=x,
                                             nc_name=var_name,
                                             nc_dim=var_dim,
                                             nc_unit=var_unit,
                                             frequence=1)
            routine_prop_list.append(observable)
        other_observables = [CalcPressure(), CalcAcell(), CalcAngles()]
        routine_prop_list += other_observables
        self.routine_prop = PropertyManager(routine_prop_list)

        if not self.launched:
            create_nc_var(ncpath, routine_prop_list)

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
            self.prop.save_prop(self.step)
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
        self.routine_prop.save_weights(self.step,
                                       self.mlip.weight,
                                       self.ncformat)
