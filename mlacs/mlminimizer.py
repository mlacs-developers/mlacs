"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
from ase.units import GPa

from .mlas import Mlas
from .properties import PropertyManager, CalcExecFunction


# ========================================================================== #
# ========================================================================== #
class MlMinimizer(Mlas):
    r"""
    Class to perform structure minimization assisted with machine-learning
    potential

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

    etol: :class:`float` (optional)
        The tolerance for the energy, in eV/at.
        If the difference of energy of the true potential between two
        consecutive steps is lower than ``etol``, the algorithm stops.

    ftol: :class:`float` (optional)
        The tolerance for the forces, in eV/angstrom.
        If the maximum absolute force of the true potential between two
        consecutive steps is lower than ``etol``, the algorithm stops.

    stol: :class:`float` (optional)
        The tolerance for the stress, in GPa.
        If the difference in the stress tensor of the true potential
        between two consecutive steps is lower than ``etol``,
        the algorithm stops.

    neq: :class:`int` (optional)
        The number of equilibration iteration. Default ``10``.

    confs_init: :class:`int` or :class:`list` of :class:`ase.Atoms` (optional)
        if :class:`int`: Number of configuirations used
        to train a preliminary MLIP
        The configurations are created by rattling the first structure
        if :class:`list` of :class:`ase.Atoms`: The atoms that are to be
        computed in order to create the initial training configurations
        Default ``None``.

    std_init: :class:`float` (optional)
        Variance (in :math:`\mathring{a}^2`) of the displacement
        when creating initial configurations.
        Default :math:`0.05 \mathring{a}^2`

    keep_tmp_mlip: :class:`Bool` (optional)
        Keep every generated MLIP. If True and using MBAR, a restart will
        recalculate every previous MLIP.weight using the old coefficients.
        Default ``False``.

    workdir: :class:`str` (optional)
        The directory in which to run the calculation.

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

    def __init__(self, atoms, state, calc, mlip=None, etol=1e-2, ftol=1e-4,
                 stol=1e-3, neq=10, confs_init=None, std_init=0.05,
                 keep_tmp_mlip=False, workdir='', prefix='Trajectory',
                 ncprefix='', ncformat='NETCDF3_CLASSIC'):
        Mlas.__init__(self, atoms, state, calc, mlip=mlip, prop=None, neq=neq,
                      confs_init=confs_init, std_init=std_init,
                      keep_tmp_mlip=keep_tmp_mlip, workdir=workdir,
                      prefix=prefix, ncprefix=ncprefix, ncformat=ncformat)

        if set((etol, ftol, stol)) == {None}:
            msg = "You need to set at least one of etol, ftol or stol"
            raise ValueError(msg)

        prop = []
        if etol is None:
            # We set to stupidly high number so that it's still computed
            etol = 1e8
        prop.append(CalcExecFunction("get_potential_energy",
                                     criterion=etol, frequence=1,
                                     gradient=True))
        if ftol is None:
            # We set to stupidly high number so that it's still computed
            ftol = 1e8
        prop.append(CalcExecFunction("get_forces", criterion=ftol,
                                     frequence=1))
        if stol is None:
            # We set to stupidly high number so that it's still computed
            stol = 1e8
        prop.append(CalcExecFunction("get_stress", criterion=stol * GPa,
                                     frequence=1, gradient=True))

        self.prop = PropertyManager(prop)
        self._etol = etol
        self._ftol = ftol

        msg = "\n\nStructure optimization assisted by machine-learning\n"
        msg += f"   Energy tolerance: {etol} eV/at\n"
        msg += f"   Forces tolerance: {ftol} eV/angs\n"
        msg += f"   Stress tolerance: {stol} GPa\n"
        self.log.logger_log.info(msg)

        self.step = 0
        self.log.logger_log.info("")

# ========================================================================== #
    def _compute_properties(self):
        """
        Main method to compute/save properties of OtfMlacs objects.
        """
        self.prop.calc_initialize(atoms=self.atoms)
        msg = self.prop.run(self.step)
        msg = ""
        ediff = self.prop.manager[0].maxf
        msg += f"Energy difference : {ediff:6.5} eV/at\n"
        maxf = self.prop.manager[1].maxf
        msg += f"Maximum absolute force : {maxf:6.5} eV/angs\n"
        maxs = self.prop.manager[2].maxf / GPa
        msg += f"Maximum stress tensor difference : {maxs:6.5} GPa\n"
        msg == "\n"
        self.log.logger_log.info(msg)

# ========================================================================== #
    def _check_early_stop(self):
        return self.prop.check_criterion
