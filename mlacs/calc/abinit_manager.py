"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import shutil
import numpy as np
import shlex

# IMPORTANT : subprocess->Popen doesnt work if we import run, PIPE
from subprocess import Popen
import logging
from concurrent.futures import ThreadPoolExecutor

from ase.symbols import symbols2numbers
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from ase.io.abinit import (write_abinit_in,
                           read_abinit_out)
from .calc_manager import CalcManager
from ..utilities.io_abinit import (AbinitNC,
                                   set_aseAtoms)


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

    mpi_runner : :class:`str`
        The command to call MPI.
        I assume the number of processor is specified using -n argument
        Default ``mpirun``

    magmoms: :class:`np.ndarray` (optional)
        An array for the initial magnetic moments for each computation
        If ``None``, no initial magnetization. (Non magnetic calculation)
        Default ``None``.

    workdir: :class:`str` (optional)
        The root for the directory in which the computation are to be done
        Default 'DFT'

    logfile: :class:`str` (optional)
        The name of the Abinit log file inside the workdir folder
        Default 'abinit.log'

    errfile: :class:`str` (optional)
        The name of the Abinit error file inside the workdir folder
        Default 'abinit.err'

    nproc: :class:`int` (optional)
        Number of processor available for Abinit.
        Distributed equally over all submitted calculations
        And start MPI abinit calculation if more than 1 processor

    ninstance: :class:`int` (optional)
        Number of instance of abinit to run in parallel.
        Default 1

    """
    def __init__(self,
                 parameters,
                 pseudos,
                 abinit_cmd="abinit",
                 mpi_runner="mpirun",
                 magmoms=None,
                 workdir=None,
                 logfile="abinit.log",
                 errfile="abinit.err",
                 nproc=1):

        CalcManager.__init__(self, "dummy", magmoms)
        self.parameters = parameters
        if 'IXC' in self.parameters.keys():
            self.parameters['ixc'] = self.parameters['IXC']
            del self.parameters['IXC']
        if 'ixc' not in self.parameters.keys():
            msg = 'WARNING AbinitManager:\n'
            msg += 'You should specify an ixc value or ASE will set 7 (LDA) !'
            logging.warning(msg)

        self._organize_pseudos(pseudos)
        self.abinit_cmd = abinit_cmd
        self.mpi_runner = mpi_runner
        self.nproc = nproc

        self.logfile = logfile
        self.errfile = errfile
        self.ncfile = AbinitNC()
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
        Create, execute and read the output of an Abinit calculation
        """
        # First we need to prepare every calculation
        confs = [at.copy() for at in confs]
        prefix = []
        for i, at in enumerate(confs):
            at.set_initial_magnetic_moments(self.magmoms)
            stateprefix = self.workdir \
                + state[i] \
                + f"/Step{step[i]}/{state[i]}_"

            prefix.append(stateprefix)
            self._write_input(at, stateprefix)

        # Define the function to be called by every process.
        def submit_abinit_calc(cmd, logfile, errfile, cdir):
            with open(logfile, 'w') as lfile, \
                 open(errfile, 'w') as efile:
                process = Popen(cmd,
                                cwd=cdir,
                                stderr=efile,
                                stdout=lfile,
                                shell=False)
                process.wait()

        # Calculate the number of processor assigned to each task
        # Divide them equally between each calculation.
        ntask = len(prefix)
        nproc = self.nproc

        # Yeah for threading
        with ThreadPoolExecutor(max_workers=ntask) as executor:
            for stateprefix in prefix:
                command = self._make_command(stateprefix, nproc)
                # Get the directory but remove the prefix to abifile
                cdir = '/'.join(stateprefix.split('/')[:-1])
                executor.submit(submit_abinit_calc,
                                command,
                                stateprefix+self.logfile,
                                stateprefix+self.errfile,
                                cdir=cdir)
            executor.shutdown(wait=True, cancel_futures=False)

        # Now we can read everything
        results_confs = []
        for (stateprefix, at) in zip(prefix, confs):
            results_confs.append(self._read_output(stateprefix, at))
        # Tada !
        return results_confs

# ========================================================================== #
    def _make_command(self, stateprefix, nproc):
        """
        Make the command to call Abinit including MPI :
        """
        abinit_cmd = self.abinit_cmd + " " + stateprefix + "abinit.abi"

        if nproc > 1:
            mpi_cmd = "{} -n {}".format(self.mpi_runner, nproc)
        else:
            mpi_cmd = ""

        full_cmd = "{} {}".format(mpi_cmd, abinit_cmd)
        full_cmd = shlex.split(full_cmd, posix=(os.name == "posix"))

        return full_cmd

# ========================================================================== #
    def _write_input(self, atoms, stateprefix):
        """
        Write the input for the current atoms
        """
        if os.path.exists(stateprefix):
            self._remove_previous_run(stateprefix)
        else:  # Get the directory but remove the prefix to abifile
            cdir = '/'.join(stateprefix.split('/')[:-1])
            os.makedirs(cdir)

        # First we need to prepare some stuff
        original_pseudos = self.pseudos.copy()
        species = sorted(set(atoms.numbers))
        self._create_unique_file(stateprefix)
        with open(stateprefix + "abinit.abi", "w") as fd:
            write_abinit_in(fd,
                            atoms,
                            self.parameters,
                            species,
                            self.pseudos)
        self.pseudos = original_pseudos

# ========================================================================== #
    def _create_unique_file(self, stateprefix):
        """
        Create a unique file for each read/write operation
        to prevent netCDF4 error.
        The path and the filename must be unique.
        """
        def _create_copy(source, dest):
            if os.path.exists(dest):
                return
            else:
                shutil.copy(source, dest)
        new_psp = []
        pp_dirpath = self.parameters.get("pp_dirpath")
        pseudos = self.pseudos

        # Create an unique psp file in the DFT/State/Step folder
        if pp_dirpath is None:
            pp_dirpath = ""
        if isinstance(pseudos, str):
            pseudos = [pseudos]
        for psp in pseudos:
            fn = psp.split('/')[-1]
            source = pp_dirpath+psp
            dest = stateprefix+fn
            _create_copy(source, dest)
            new_psp.append(dest)
        self.pseudos = new_psp
        self.parameters.pop('pspdir', None)

# ========================================================================== #
    def _read_output(self, stateprefix, at):
        """
        """
        results = {}
        if self.ncfile is not None:
            dct = self.ncfile.read(stateprefix + "abinito_GSR.nc")
            results.update(dct)
            atoms = set_aseAtoms(results)
            atoms.set_velocities(at.get_velocities())
            return atoms

        with open(stateprefix + "abinit.abo") as fd:
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
    def _remove_previous_run(self, stateprefix):
        """
        Little function to remove any trace of previous calculation
        """
        if os.path.exists(stateprefix + "abinit.abi"):
            os.remove(stateprefix + "abinit.abi")
        if os.path.exists(stateprefix + "abinit.abo"):
            os.remove(stateprefix + "abinit.abo")
        if os.path.exists(stateprefix + "abinit.log"):
            os.remove(stateprefix + "abinit.log")
        if os.path.exists(stateprefix + "abinito_GSR.nc"):
            os.remove(stateprefix + "abinito_GSR.nc")
        if os.path.exists(stateprefix + "abinito_OUT.nc"):
            os.remove(stateprefix + "abinito_OUT.nc")
        if os.path.exists(stateprefix + "abinito_DEN"):
            os.remove(stateprefix + "abinito_DEN")
        if os.path.exists(stateprefix + "abinito_WF"):
            os.remove(stateprefix + "abinito_WF")
        if os.path.exists(stateprefix + "abinito_DDB"):
            os.remove(stateprefix + "abinito_DDB")
        if os.path.exists(stateprefix + "abinito_EIG"):
            os.remove(stateprefix + "abinito_EIG")
        if os.path.exists(stateprefix + "abinito_EBANDS.agr"):
            os.remove(stateprefix + "abinito_EBANDS.agr")
