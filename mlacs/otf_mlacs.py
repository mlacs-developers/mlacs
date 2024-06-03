"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np

from .mlas import Mlas
from .core import Manager
from .properties import PropertyManager
from .utilities.log import MlacsLog


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
                 workdir=''):

        Manager.__init__(self, workdir=workdir)

        # Initialize working directory
        self.workdir.mkdir(exist_ok=True, parents=True)

        ##############
        # Check inputs
        ##############
        self.keep_tmp_mlip = keep_tmp_mlip
        Mlas.__init__(self, atoms, state, calc, mlip=mlip, prop=None, neq=neq,
                      confs_init=confs_init, std_init=std_init,
                      ntrymax=ntrymax, keep_tmp_mlip=keep_tmp_mlip)

        # Miscellanous initialization
        self.rng = np.random.default_rng()
        self.ntrymax = ntrymax

        # Check if trajectory files already exists
        self.launched = self._check_if_launched()

        self.log = MlacsLog(str(self.workdir / "MLACS.log"), self.launched)
        self.logger = self.log.logger_log
        msg = ""
        for i in range(self.nstate):
            msg += f"State {i+1}/{self.nstate} :\n"
            msg += repr(self.state[i])
        self.logger.info(msg)
        msg = self.calc.log_recap_state()
        self.logger.info(msg)
        self.logger.info(repr(self.mlip))

# ========================================================================== #
    def _initialize_properties(self, prop):
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
