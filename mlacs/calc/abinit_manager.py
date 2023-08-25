"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import shlex
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.symbols import symbols2numbers
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from ase.io.abinit import (write_abinit_in,
                           read_abinit_out)

from .calc_manager import CalcManager


# ========================================================================== #
# ========================================================================== #
class AbinitManager(CalcManager):
    """
    Class to handle abinit calculators

    Parameters
    ----------
    parameters: :class:`dict`
        Dictionnary of abinit input

    pseudos: :class:`dict`
        Dictionnary for the pseudopotentials
        {'O': /path/to/pseudo}

    abinit_cmd: :class:`str`
        The command to execute the abinit binary.

    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.

    workdir: :class:`str` (optional)
        The root for the directory in which the computation are to be done
        Default 'DFT'

    ninstance: :class:`int` (optional)
        Number of instance of abinit to run in parallel.
        Default 1
    """
    def __init__(self,
                 parameters,
                 pseudos,
                 abinit_cmd="abinit",
                 magmoms=None,
                 workdir=None,
                 ninstance=1):

        CalcManager.__init__(self, "dummy", magmoms)
        self.parameters = parameters
        self._organize_pseudos(pseudos)
        self.cmd = shlex.split(abinit_cmd + " abinit.abi",
                               posix=(os.name == "posix"))
        self.ninstance = ninstance

        try:
            AbinitNC.__init__()
            self._read_output_abi = self._read_output_nc
        except:
            self._read_output_abi = self._read_output

        self.workdir = workdir
        if self.workdir is None:
            self.workdir = os.getcwd() + "/DFT/"
        if self.workdir[-1] != "/":
            self.workdir[-1] += "/"
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

# ========================================================================== #
    def compute_true_potential(self,
                               confs,
                               state,
                               step):
        """
        """
        # First we need to prepare every calculation
        confs = [at.copy() for at in confs]
        confdir = []
        for i, at in enumerate(confs):
            at.set_initial_magnetic_moments(self.magmoms)
            cdir = self.workdir + state[i] + f"/Step{step[i]}/"
            confdir.append(cdir)
            self._write_input(at, cdir)

        # Now we can execute everything, ninstance at a time
        # Yeah for threading
        with ThreadPoolExecutor(max_workers=self.ninstance) as executor:
            for cdir in confdir:
                executor.submit(run,
                                self.cmd,
                                cwd=cdir,
                                stdout=PIPE,
                                stderr=PIPE)

        # Now we can read everything
        results_confs = []
        for (cdir, at) in zip(confdir, confs):
            results_confs.append(self._read_output_abi(cdir, at))
        # Tada !
        return results_confs

# ========================================================================== #
    def _write_input(self, atoms, confdir):
        """
        Write the input for the current atoms
        """
        if os.path.exists(confdir):
            self._remove_previous_run(confdir)
        else:
            os.makedirs(confdir)
        # First we need to prepare some stuff
        species = sorted(set(atoms.numbers))
        with open(confdir + "abinit.abi", "w") as fd:
            write_abinit_in(fd,
                            atoms,
                            self.parameters,
                            species,
                            self.pseudos)

# ========================================================================== #
    def _read_output(self, cdir, at):
        """
        """
        results = {}
        with open(cdir + "abinit.abo") as fd:
            dct = read_abinit_out(fd)
            results.update(dct)
        atoms = results.pop("atoms")
        energy = results.pop("energy")
        forces = results.pop("forces")
        stress = results.pop("stress")

        atoms.set_velocities(at.get_velocities())
        calc = SPCalc(atoms,
                      energy=energy,
                      forces=forces,
                      stress=stress)
        calc.version = results.pop("version")
        atoms.calc = calc
        return atoms

# ========================================================================== #
    def _read_output_nc(self, cdir, at):
        """
        """
        ncfiles = AbinitNC()
        results = ncfiles.read(cdir + "abinito_GSR.nc")

        atoms.set_velocities(at.get_velocities())
        calc = SPCalc(atoms,
                      energy=energy,
                      forces=forces,
                      stress=stress)
        calc.version = results.pop("version")
        atoms.calc = calc
        return atoms

# ========================================================================== #
    def _organize_pseudos(self, pseudos):
        """
        To have the pseudo well organized, we need to sort the pseudos
        """
        typat = []
        pseudolist = []
        for ityp in pseudos.keys():
            typat.append(ityp)
            pseudolist.append(pseudos[ityp])
        pseudolist = np.array(pseudolist)

        znucl = symbols2numbers(typat)
        idx = np.argsort(znucl)
        pseudolist = pseudolist[idx]
        self.pseudos = pseudolist

# ========================================================================== #
    def _remove_previous_run(self, confdir):
        """
        Little function to remove any trace of previous calculation
        """
        if os.path.exists(confdir + "abinit.abi"):
            os.remove(confdir + "abinit.abi")
        if os.path.exists(confdir + "abinit.abo"):
            os.remove(confdir + "abinit.abo")
        if os.path.exists(confdir + "abinit.log"):
            os.remove(confdir + "abinit.log")
        if os.path.exists(confdir + "abinito_GSR.nc"):
            os.remove(confdir + "abinito_GSR.nc")
        if os.path.exists(confdir + "abinito_OUT.nc"):
            os.remove(confdir + "abinito_OUT.nc")
        if os.path.exists(confdir + "abinito_DEN"):
            os.remove(confdir + "abinito_DEN")
        if os.path.exists(confdir + "abinito_WF"):
            os.remove(confdir + "abinito_WF")
        if os.path.exists(confdir + "abinito_DDB"):
            os.remove(confdir + "abinito_DDB")
        if os.path.exists(confdir + "abinito_EIG"):
            os.remove(confdir + "abinito_EIG")
        if os.path.exists(confdir + "abinito_EBANDS.agr"):
            os.remove(confdir + "abinito_EBANDS.agr")

# ========================================================================== #
# ========================================================================== #
class AbinitNC:
    """
    Class to handle all Abinit NetCDF files.

    Parameters
    ----------
    workdir: :class:`str` (optional)
        The root for the directory in which the computation are to be done
        Default 'DFT'
    """
    def __init__(self, workdir=None, prefix='abinit'):

        import netCDF4 as nc

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

        import netCDF4 as nc

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
