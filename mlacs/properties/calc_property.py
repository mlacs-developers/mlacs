"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, PR, CD)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import copy
import importlib
import numpy as np
from operator import attrgetter

from ase.atoms import Atoms
from ase.units import Hartree, Bohr

from ..core.manager import Manager
from ..utilities.io_lammps import LammpsBlockInput
from ..utilities.miscellanous import read_distribution_files as read_df

ti_args = ['atoms',
           'pair_style',
           'pair_coeff',
           'temperature',
           'pressure',
           'nsteps',
           'nsteps_eq']


# ========================================================================== #
# ========================================================================== #
class CalcProperty(Manager):
    """
    Parent Class for on the fly property calculations.

    Parameters
    ----------
    method: :class:`str`
    Type of criterion.
        - max, maximum difference between to consecutive step < criterion
        - ave, average difference between to consecutive step < criterion
    Default ``max``

    criterion: :class:`float`
        Stopping criterion value (eV). Default ``0.001``
        Can be ``None``, in this case there is no criterion.

    frequence : :class:`int`
        Interval of Mlacs step to compute the property. Default ``1``
    """

    def __init__(self,
                 args={},
                 state=None,
                 method='max',
                 criterion=0.001,
                 frequence=1,
                 **kwargs):

        self.freq = frequence
        self.stop = criterion
        self.method = method
        self.kwargs = args
        self.isfirst = True
        self.isgradient = True
        self.useatoms = True
        self.needdir = True
        self.label = 'Observable_Label'
        self.shape = None
        self.nc_name = None
        self.nc_dim = None
        self.nc_unit = ''
        if state is not None:
            self.state = copy.deepcopy(state)

# ========================================================================== #
    def _exec(self):
        """
        Dummy execution function.
        """
        raise RuntimeError("Execution not implemented.")

# ========================================================================== #
    @property
    def isconverged(self):
        """
        Check if the property is converged.
        """
        if not isinstance(self.new, np.ndarray):
            self.new = np.r_[self.new]
        if self.isfirst:
            self.old = np.zeros(self.new.shape)
            self.isfirst = False
        check = self._check
        self.old = self.new
        return check

# ========================================================================== #
    @property
    def _check(self):
        """
        Check criterions.
        """
        self.maxf = np.max(np.abs(self.new - self.old))
        if not self.isgradient:
            self.maxf = np.max(np.abs(self.new))
        self.avef = np.average(np.abs(self.new - self.old))
        if self.stop is None:
            return False
        elif self.method == 'max' and self.maxf < self.stop:
            return True
        elif self.method == 'ave' and self.avef < self.stop:
            return True
        else:
            return False

# ========================================================================== #
    def get_atoms(self, atoms):
        """
        If reference configuration needed.
        """
        if isinstance(atoms, Atoms):
            self.atoms = [atoms.copy()]
        else:
            self.atoms = atoms.copy()

# ========================================================================== #
    def __repr__(self):
        """
        Dummy function for the real logger.
        """
        return ""


# ========================================================================== #
# ========================================================================== #
class CalcPafi(CalcProperty):
    """
    Class to set a minimum free energy calculation.
    See :func:`PafiLammpsState.run_dynamics` parameters.
    """

    def __init__(self,
                 args,
                 state=None,
                 method='max',
                 criterion=0.001,
                 frequence=1,
                 **kwargs):
        CalcProperty.__init__(self, args, state, method, criterion, frequence,
                              **kwargs)
        self.state.folder = 'PafiPath_Calculation'

# ========================================================================== #
    @Manager.exec_from_workdir
    def _exec(self):
        """
        Exec a MFEP calculation with lammps. Use replicas.
        """
        self.state.workdir = self.folder
        self.state.subfolder = self.subfolder
        atoms = self.state.path.atoms[0]
        mlip = self.kwargs['mlip']
        self.new = self.state.run_pafipath_dynamics(
                atoms, mlip.pair_style, mlip.pair_coeff)[1]
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = 'Computing the minimum free energy path:\n'
        msg += self.state.log_recap_state()
        msg += 'Free energy difference along the path with previous step:\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcNeb(CalcProperty):
    """
    Class to set a NEB calculation.
    See :func:`NebLammpsState.run_dynamics` parameters.
    """

    def __init__(self,
                 args,
                 state=None,
                 method='max',
                 criterion=0.001,
                 frequence=1):
        CalcProperty.__init__(self, args, state, method, criterion, frequence)
        self.state.folder = 'Neb_Calculation'

# ========================================================================== #
    @Manager.exec_from_workdir
    def _exec(self):
        """
        Exec a NEB calculation with lammps. Use replicas.
        """
        self.state.workdir = self.folder
        self.state.subfolder = self.subfolder
        atoms = self.state.atoms[0]
        mlip = self.kwargs['mlip']
        self.state.run_dynamics(atoms, mlip.pair_style, mlip.pair_coeff)
        self.state.extract_NEB_configurations()
        self.new = self.state.spline_energies
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = 'Computing the minimum energy path from a NEB calculation:\n'
        msg += self.state.log_recap_state()
        msg += 'Energy difference along the reaction '
        msg += 'path with previous step:\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcRdf(CalcProperty):
    """
    Class to set a radial distribution function calculation.
    """

    def __init__(self,
                 args,
                 state=None,
                 method='max',
                 criterion=0.05,
                 frequence=2):
        CalcProperty.__init__(self, args, state, method, criterion, frequence)

        self.useatoms = True
        self.step = self.state.nsteps_eq
        if 'nsteps' in self.kwargs.keys():
            self.step = self.kwargs['nsteps'] / 10
            self.state.nsteps = self.kwargs['nsteps']
            self.kwargs.pop('nsteps')
        self.filename = 'spce-rdf.dat'
        if 'filename' in self.kwargs.keys():
            self.filename = self.kwargs['filename']
            self.kwargs.pop('filename')
        self.state.folder = 'Rdf_Calculation'

# ========================================================================== #
    @Manager.exec_from_workdir
    def _exec(self):
        """
        Exec a Rdf calculation with lammps.
        """

        from ..utilities.io_lammps import get_block_rdf

        self.state.workdir = self.folder
        self.state.subfolder = self.subfolder
        if self.state._myblock is None:
            block = LammpsBlockInput("Calc RDF", "Calculation of the RDF")
            block("equilibrationrun", f"run {self.step}")
            block("reset_timestep", "reset_timestep 0")
            block.extend(get_block_rdf(self.step, self.filename))
            self.state._myblock = block
        mlip = self.kwargs['mlip']
        self.state.run_dynamics(self.atoms[-1], mlip.pair_style,
                                mlip.pair_coeff)
        self.new = read_df(self.state.subsubdir / self.filename)[0]
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = 'For the radial distribution function g(r):\n'
        msg += self.state.log_recap_state()
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcAdf(CalcProperty):
    """
    Class to set the angle distribution function calculation.
    """

    def __init__(self,
                 args,
                 state=None,
                 method='max',
                 criterion=0.05,
                 frequence=5):
        CalcProperty.__init__(self, args, state, method, criterion, frequence)

        self.useatoms = True
        self.step = self.state.nsteps_eq
        if 'nsteps' in self.kwargs.keys():
            self.step = self.kwargs['nsteps'] / 10
            self.state.nsteps = self.kwargs['nsteps']
            self.kwargs.pop('nsteps')
        self.filename = 'spce-adf.dat'
        if 'filename' in self.kwargs.keys():
            self.filename = self.kwargs['filename']
            self.kwargs.pop('filename')
        self.state.folder = 'Adf_Calculation'

# ========================================================================== #
    @Manager.exec_from_workdir
    def _exec(self):
        """
        Exec an Adf calculation with lammps.
        """

        from ..utilities.io_lammps import get_block_adf

        self.state.workdir = self.folder
        self.state.subfolder = self.subfolder
        if self.state._myblock is None:
            block = LammpsBlockInput("Calc ADF", "Calculation of the ADF")
            block("equilibrationrun", f"run {self.step}")
            block("reset_timestep", "reset_timestep 0")
            block.extend(get_block_adf(self.step, self.filename))
            self.state._myblock = block
        mlip = self.kwargs['mlip']
        self.state.run_dynamics(self.atoms[-1], mlip.pair_style,
                                mlip.pair_coeff)
        self.new = read_df(self.state.subsubdir / self.filename)[0]
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = 'For the angle distribution function g(theta):\n'
        msg += self.state.log_recap_state()
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcTi(CalcProperty):
    """
    Class to set a nonequilibrium thermodynamic integration calculation.
    See the :class:`ThermodynamicIntegration` classe.

    Parameters
    ----------
    phase: :class:`str`
        Structure of the system: solild or liquid.
        Set either the Einstein crystal as a reference system or the UF liquid.
    """

    def __init__(self,
                 args,
                 phase,
                 state=None,
                 ninstance=None,
                 method='max',
                 criterion=0.001,
                 frequence=10):
        CalcProperty.__init__(self, args, state, method, criterion, frequence)

        self.ninstance = ninstance
        self.phase = phase
        if self.phase == 'solid':
            from mlacs.ti import EinsteinSolidState
        elif self.phase == 'liquid':
            from mlacs.ti import UFLiquidState
#        else:
#            print('abort_unkown_phase')
#            exit(1)
        self.ti_state = {}
        self.kwargs = {}
        for keys, values in args.items():
            if keys in ti_args:
                self.ti_state[keys] = values
            else:
                self.kwargs[keys] = values
        if self.phase == 'solid':
            self.state = EinsteinSolidState(**self.ti_state)
        elif self.phase == 'liquid':
            self.state = UFLiquidState(**self.ti_state)

# ========================================================================== #
    @Manager.exec_from_workdir
    def _exec(self):
        """
        Exec a NETI calculation with lammps.
        """
        from mlacs.ti import ThermodynamicIntegration

        # Creation of ti object ---------------------------------------------
        self.ti = ThermodynamicIntegration(self.state,
                                           self.ninstance,
                                           logfile="TiCheckFe.log")
        self.ti.workdir = self.folder
        self.ti.folder = 'Neti_Calculation'
        self.ti.subfolder = self.subfolder

        # Run the simu ------------------------------------------------------
        self.ti.run()
        # Get Fe ------------------------------------------------------------
        if self.ninstance == 1:
            _, self.new = self.state.postprocess(self.ti.get_fedir())
        elif self.ninstance > 1:
            tmp = []
            for i in range(self.ninstance):
                _, tmp_new = self.state.postprocess(self.ti.get_fedir()
                                                    + f"for_back_{i+1}/")
                tmp.append(tmp_new)
            self.new = np.r_[np.mean(tmp)]
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = 'For the free energy convergence check:\n'
        msg += 'Free energy at this step is: '
        for _ in self.new:
            msg += f' {_:10.6f}'
        msg += '\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        msg += f'        - Averaged : {self.avef}\n\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcExecFunction(CalcProperty):
    """
    Class to execute on the fly a python function and converge on the result.

    Parameters
    ----------
    function: :class:`str` or `function`
        Function to call. If the function is a `str`, you to define the
        module to load the function.

    args: :class:`dict`
        Arguments of the function.

    module: :class:`str`
        Module to load the function.

    useatoms: :class:`bool`
        True if the function is called from an ase.Atoms object.

    nc_unit: :class:`str` (optional)
        Unit of the observable saved in *HIST.nc file.
        These units are derived from the atomic unit system: Bohr, Ha, etc.
        Cf. mlacs.utilities.io_abinit.MlacsHist._set_unit_conventions().
        Default ''.

    ase_unit: :class:`str` (optional)
        Unit of the observable (ASE convention).
        Cf. mlacs.utilities.io_abinit.MlacsHist._set_unit_conventions().
        These units are expected to be the `metal` units, cf. ase.units.py.
        Default ''.
    """

    def __init__(self,
                 function,
                 args={},
                 module=None,
                 use_atoms=True,
                 gradient=False,
                 criterion=0.001,
                 frequence=1,
                 nc_unit='',
                 ase_unit=''):
        CalcProperty.__init__(self, args, None, 'max', criterion, frequence)

        self._func = function
        if module is not None:
            importlib.import_module(module)
            self._function = getattr(module, function)
        self.isfirst = True
        self.needdir = False
        self.use_atoms = use_atoms
        self.isgradient = gradient
        self.label = function
        self.shape = None
        self.nc_unit = nc_unit
        self.ase_unit = ase_unit

# ========================================================================== #
    def _unit_converter(self):
        """
        Convert units from ASE unit convention to Abinit's, i.e.,
        the result self.new of the _exec() routine is expressed in atomic unit
        """
        # TODO: Move this routine to utilities.units.py
        eV2Ha = 1/Hartree
        Ang2Bohr = 1/Bohr
        # Dictionary that maps ASE units to the corresponding multiplication
        # factors that convert them to Abinit units system.
        # Example: if a = 100 eV then a*unit_convert_dict['eV'] is 3.67 Hartree
        unit_convert_dict = {'eV': eV2Ha,
                             'Ang': Ang2Bohr,
                             'eV/Ang': eV2Ha/Ang2Bohr,
                             'Ang^3': Ang2Bohr**3,
                             'eV/Ang^3': eV2Ha/Ang2Bohr**3,
                             }

        # Proceed with the conversion itself
        if hasattr(self, 'new') and self.ase_unit in unit_convert_dict:
            self.new *= unit_convert_dict[self.ase_unit]

# ========================================================================== #
    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            self._function = [getattr(_, self._func) for _ in self.atoms]
            self.new = np.r_[[_f(**self.kwargs) for _f in self._function]]
        else:
            self.new = self._function(**self.kwargs)
        self._unit_converter()
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        property.
        """
        msg = f'Converging on the result of {self._func} function\n'
        if self.isgradient:
            msg += 'Computed with the previous step:\n'
        msg += f'        - Maximum  : {self.maxf}\n'
        return msg


# ========================================================================== #
# ========================================================================== #
class CalcRoutineFunction(CalcExecFunction):
    """
    Class to routinely compute basic thermodynamic observables.

    Parameters
    ----------

    function: :class:`str`
        Name of Lammps function, e.g. `get_kinetic_energy'.

    label: :class:`str`
        Label of the function to be executed, e.g. `Kinetic_Energy`.
        Cf. mlacs.utilities.io_abinit.MlacsHist.nc_routine_conv().

    nc_name: :class:`str` (optional)
        Name of the observable in *HIST.nc file, e.g. `ekin`.
        Cf. mlacs.utilities.io_abinit.MlacsHist.nc_routine_conv().
        This name should follow Abinit conventions as much as possible.
        Default ``None``.

    nc_dim: :class:`str` (optional)
        Name of the dimension of the observable in *HIST.nc file.
        Cf. mlacs.utilities.io_abinit.MlacsHist._set_name_conventions().
        Default ``None``.

    nc_unit: :class:`str` (optional)
        Unit of the observable saved in *HIST.nc file.
        These units are derived from the atomic unit system: Bohr, Ha, etc.
        Cf. mlacs.utilities.io_abinit.MlacsHist._set_unit_conventions().
        Default ''.

    ase_unit: :class:`str` (optional)
        Unit of the observable (ASE convention).
        Cf. mlacs.utilities.io_abinit.MlacsHist._set_unit_conventions().
        These units are expected to be the `metal` units, cf. ase.units.py.
        Default ''.

    weight: :class:`WeightingPolicy` (optional)
        WeightingPolicy class, Default: `None`.
    """

    def __init__(self,
                 function,
                 label,
                 nc_name=None,
                 nc_dim=None,
                 nc_unit='',
                 ase_unit='',
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):
        CalcExecFunction.__init__(self, function, dict(), None, True, gradient,
                                  criterion, frequence, nc_unit, ase_unit)
        self.weight = weight
        self.label = label
        self.needdir = False
        self.nc_name = nc_name
        self.nc_dim = nc_dim

# ========================================================================== #
    def __repr__(self):
        """
        Return a string for the log with informations of the calculated
        routine property.
        """
        name_observable = self.label.lower().replace("_", " ")
        unit = self.nc_unit
        msg = f'Routine computation of the {name_observable}\n'
        if len(self.shape) == 0:
            if len(self.new > 0):
                for idx_state, val in enumerate(self.new):
                    msg += f'        - Value for state {idx_state+1} : '
                    msg += '{:.5e}'.format(val) + ' ' + unit + ' \n'
            else:
                msg += f'        - Value for state 1  : {self.new}\n'
        else:
            # Too big to print, cf. *_HIST.nc file
            msg = ''

        return msg


# ========================================================================== #
# ========================================================================== #
class CalcPressure(CalcRoutineFunction):
    """
    Class to compute the hydrostatic pressure.

    Parameters
    ----------

    weight: :class:`WeightingPolicy`
        WeightingPolicy class, Default: `None`.
    """

    def __init__(self,
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):

        label = 'Pressure'
        nc_name = 'press'
        nc_dim = ('time',)
        nc_unit = 'Ha/Bohr^3'
        ase_unit = 'eV/Ang^3'
        CalcRoutineFunction.__init__(self,
                                     'get_stress',
                                     label,
                                     nc_name,
                                     nc_dim,
                                     nc_unit,
                                     ase_unit)

    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            self._function = [getattr(_, self._func) for _ in self.atoms]
            self.new = np.r_[[-np.mean(_f(**self.kwargs)[:3])
                              for _f in self._function]]
        else:
            self.new = self._function(**self.kwargs)
        self._unit_converter()
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged


# ========================================================================== #
# ========================================================================== #
class CalcAcell(CalcRoutineFunction):
    """
    Class to compute the cell lengths.

    Parameters
    ----------
    weight: :class:`WeightingPolicy`
        WeightingPolicy class, Default: `None`.
    """

    def __init__(self,
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):
        label = 'Acell'
        nc_name = 'acell'
        nc_dim = ('time', 'xyz')
        nc_unit = 'Bohr'
        ase_unit = 'Ang'
        CalcRoutineFunction.__init__(self,
                                     'get_cell_lengths_and_angles',
                                     label,
                                     nc_name,
                                     nc_dim,
                                     nc_unit,
                                     ase_unit)

    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            attr = 'cell.cellpar'
            self._function = [attrgetter(attr)(_) for _ in self.atoms]
            self.new = np.r_[[_f(**self.kwargs)[:3] for _f in self._function]]
        else:
            self.new = self._function(**self.kwargs)
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged


# ========================================================================== #
# ========================================================================== #
class CalcAngles(CalcRoutineFunction):
    """
    Class to compute the cell angles.

    Parameters
    ----------
    weight: :class:`WeightingPolicy`
        WeightingPolicy class, Default: `None`.
    """

    def __init__(self,
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):
        label = 'Angles'
        nc_name = 'angl'
        nc_dim = ('time', 'xyz')
        nc_unit = 'deg'
        ase_unit = 'deg'
        CalcRoutineFunction.__init__(self,
                                     'get_cell_lengths_and_angles',
                                     label,
                                     nc_name,
                                     nc_dim,
                                     nc_unit,
                                     ase_unit)

    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            attr = 'cell.cellpar'
            self._function = [attrgetter(attr)(_) for _ in self.atoms]
            self.new = np.r_[[_f(**self.kwargs)[3:] for _f in self._function]]
        else:
            self.new = self._function(**self.kwargs)
        self._unit_converter()
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged


# ========================================================================== #
# ========================================================================== #
class CalcSpinAt(CalcRoutineFunction):
    """
    Class to obtain the electronic spin-magnetization (as computed by Abinit)
    from ASE's Atoms object.
    See also AbinitNC class in mlacs.utilities.io_abinit.py
    """

    def __init__(self,
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):

        label = 'Electronic_Spin_Magnetization'
        nc_name = 'spinat'
        nc_dim = ('time', 'natom', 'xyz',)
        nc_unit = 'hbar/2'
        ase_unit = 'hbar/2'
        CalcRoutineFunction.__init__(self,
                                     '',
                                     label,
                                     nc_name,
                                     nc_dim,
                                     nc_unit,
                                     ase_unit)

    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            try:
                self.new = np.r_[[_.get_array('spinat') for _ in self.atoms]]
            except KeyError:
                self.new = np.r_[[np.zeros((len(_), 3)) for _ in self.atoms]]
        else:
            self.new = self._function(**self.kwargs)
        self._unit_converter()
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged


# ========================================================================== #
# ========================================================================== #
class CalcElectronicEntropy(CalcRoutineFunction):
    """
    Class to obtain the electronic entropy (as computed by Abinit)
    from ASE's Atoms object.
    """

    def __init__(self,
                 weight=None,
                 gradient=False,
                 criterion=None,
                 frequence=1):

        label = 'Electronic_Entropy'
        nc_name = 'entropy'
        nc_dim = ('time',)
        nc_unit = ''
        ase_unit = ''
        CalcRoutineFunction.__init__(self,
                                     '',
                                     label,
                                     nc_name,
                                     nc_dim,
                                     nc_unit,
                                     ase_unit)

    def _exec(self):
        """
        Execute function
        """
        if self.use_atoms:
            try:
                self.new = np.r_[[_.get_properties('')['free_energy']
                                  for _ in self.atoms]]
            except KeyError:
                self.new = np.r_[[0.0 for _ in self.atoms]]
        else:
            self.new = self._function(**self.kwargs)
        self._unit_converter()
        if self.isfirst:
            self.shape = self.new[0].shape
        return self.isconverged
