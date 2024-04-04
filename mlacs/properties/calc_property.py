"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import copy
import importlib
import numpy as np

from ase.atoms import Atoms

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
class CalcProperty:
    """
    Parent Class for on the fly property calculations.

    Parameters
    ----------
    method: :class:`str` type of criterion.
        - max, maximum difference between to consecutive step < criterion
        - ave, average difference between to consecutive step < criterion

        Default ``max``

    criterion: :class:`float`
        Stopping criterion value (eV). Default ``0.001``

    frequence : :class:`int`
        Interval of Mlacs step to compute the property. Default ``1``
    """

    def __init__(self,
                 args={},
                 state=None,
                 method='max',
                 criterion=0.001,
                 frequence=1):

        self.freq = frequence
        self.stop = criterion
        self.method = method
        self.kwargs = args
        self.isfirst = True
        self.isgradient = True
        self.useatoms = True
        if state is not None:
            self.state = copy.deepcopy(state)

# ========================================================================== #
    def _exec(self, wdir=None):
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
        if self.isfirst:
            if isinstance(self.new, np.ndarray):
                self.old = np.zeros(self.new.shape)
            else:
                self.new = np.r_[self.new]
                self.old = np.zeros(self.new.shape)
            check = self._check
            self.old = self.new
            self.isfirst = False
        else:
            check = self._check
            if not isinstance(self.new, np.ndarray):
                self.new = np.r_[self.new]
            self.old = self.new
        return check

# ========================================================================== #
    @property
    def _check(self):
        """
        Check criterions.
        """
        self.maxf = np.max(np.abs(self.new-self.old))
        if not self.isgradient:
            self.maxf = np.max(np.abs(self.new))
        self.avef = np.average(np.abs(self.new-self.old))
        if self.method == 'max' and self.maxf < self.stop:
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
                 frequence=1):
        CalcProperty.__init__(self, args, state, method, criterion, frequence)

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a MFEP calculation with lammps. Use replicas.
        """
        self.state.workdir = wdir / 'PafiPath_Calculation'
        atoms = self.state.path.atoms[0]
        self.new = self.state.run_pafipath_dynamics(atoms, **self.kwargs)[1]
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

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a NEB calculation with lammps. Use replicas.
        """
        self.state.workdir = wdir / 'NEB_Calculation'
        atoms = self.state.atoms[0]
        self.state.run_dynamics(atoms, **self.kwargs)
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
                 frequence=5):
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

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec a Rdf calculation with lammps.
        """

        from ..utilities.io_lammps import get_block_rdf

        self.state.workdir = wdir / 'Rdf_Calculation'
        if self.state._myblock is None:
            block = LammpsBlockInput("Calc RDF", "Calculation of the RDF")
            block("equilibrationrun", f"run {self.step}")
            block("reset_timestep", "reset_timestep 0")
            block.extend(get_block_rdf(self.step, self.filename))
            self.state._myblock = block
        self.state.run_dynamics(self.atoms[-1], **self.kwargs)
        self.new = read_df(self.state.workdir / self.filename)[0]
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

# ========================================================================== #
    def _exec(self, wdir):
        """
        Exec an Adf calculation with lammps.
        """

        from ..utilities.io_lammps import get_block_adf

        self.state.workdir = wdir / 'Adf_Calculation'
        if self.state._myblock is None:
            block = LammpsBlockInput("Calc ADF", "Calculation of the ADF")
            block("equilibrationrun", f"run {self.step}")
            block("reset_timestep", "reset_timestep 0")
            block.extend(get_block_adf(self.step, self.filename))
            self.state._myblock = block
        self.state.run_dynamics(self.atoms[-1], **self.kwargs)
        self.new = read_df(self.state.workdir / self.filename)[0]
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
    def _exec(self, wdir):
        """
        Exec a NETI calculation with lammps.
        """
        from mlacs.ti import ThermodynamicIntegration
        # Creation of ti object ---------------------------------------------
        self.ti = ThermodynamicIntegration(self.state,
                                           self.ninstance,
                                           wdir,
                                           logfile=os.path.join(wdir, "TiCheckFe.log"))

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
    """

    def __init__(self,
                 function,
                 args={},
                 module=None,
                 aseatoms=True,
                 gradient=False,
                 criterion=0.001,
                 frequence=1):
        CalcProperty.__init__(self, args, None, 'max', criterion, frequence)

        self._func = function
        if module is not None:
            importlib.import_module(module)
            self._function = getattr(module, function)
        self.isfirst = True
        self.useatoms = aseatoms
        self.isgradient = gradient

# ========================================================================== #
    def _exec(self, wdir=None):
        """
        Execute function
        """
        if self.useatoms:
            self._function = [getattr(_, self._func) for _ in self.atoms]
            self.new = np.r_[[_f(**self.kwargs) for _f in self._function]]
        else:
            self.new = self._function(**self.kwargs)
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
