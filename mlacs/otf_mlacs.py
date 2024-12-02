"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from .mlas import Mlas
from .core import Manager
from .properties import PropertyManager


# ========================================================================== #
# ========================================================================== #
class OtfMlacs(Mlas, Manager):
    r"""
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
        Default ``None``.

    std_init: :class:`float` (optional)
        Variance (in :math:`\mathring{a}^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 \mathring{a}^2`

    keep_tmp_mlip: :class:`Bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``True``.

    prefix: :class:`str` (optional)
        The prefix to prepend the name of the States files.

    ncprefix: :class:`str` (optional)
        The prefix to prepend the name of the *HIST.nc file.
        Script name format: ncprefix + scriptname + '_HIST.nc'.
        Default `''`.

    ncformat: :class:`str` (optional)
        The format of the *HIST.nc file. One of the five flavors of netCDF
        files format available in netCDF4 python package: 'NETCDF3_CLASSIC',
        'NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT_DATA','NETCDF4_CLASSIC',
        'NETCDF4'.
        Default ``NETCDF3_CLASSIC``.
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
                 workdir='',
                 prefix='Trajectory',
                 ncprefix='',
                 ncformat='NETCDF3_CLASSIC'):
        Mlas.__init__(self, atoms, state, calc, mlip=mlip, prop=None, neq=neq,
                      confs_init=confs_init, std_init=std_init,
                      keep_tmp_mlip=keep_tmp_mlip, workdir=workdir,
                      prefix=prefix, ncprefix=ncprefix, ncformat=ncformat)

        # RB: Move the initialization of properties out of Mlas.
        self._initialize_properties(prop)

# ========================================================================== #
    def _initialize_properties(self, prop):
        """Create property object"""
        self.prop = PropertyManager(prop)

        if self.ncfile is not None:
            if not self.launched:
                self.ncfile.create_nc_var(self.prop.manager)

            self.prop.workdir = self.workdir
            # RB: I think this is not necessary anymore
            # if not self.prop.folder:
            #     self.prop.folder = 'Properties'

            self.prop.isfirstlaunched = not self.launched
            self.prop.ncfile = self.ncfile

# ========================================================================== #
    def _compute_properties(self):
        """
        Main method to compute/save properties of OtfMlacs objects.
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
