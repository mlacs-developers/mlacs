"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import shlex
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor
#DEBUG
from subprocess import Popen
from concurrent.futures import wait, ALL_COMPLETED
import time

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
    logfile: :class:`str` (optional)
        The name of the Abinit log file inside the workdir folder
        Default 'abinit.log'
    errfile: :class:`str` (optional)
        The name of the Abinit error file inside the workdir folder
        Default 'abinit.err'
    nproc: :class:`int` (optional)
        Number of processor available by abinit.
        Used to start multiple calculation in parallel and use mpi if some proc are still available
        Default 1
    nomp_thread :class: `int` (optional)
        Number of OpenMP thread to use per processor.
        Default 1

    """
    def __init__(self,
                 parameters,
                 pseudos,
                 abinit_cmd="abinit",
                 magmoms=None,
                 workdir=None,
                 logfile="abinit.log",
                 errfile="abinit.err",
                 nproc=1,
                 nomp_thread=1):

        CalcManager.__init__(self, "dummy", magmoms)
        self.parameters = parameters
        self._organize_pseudos(pseudos)
        self.cmd = shlex.split(abinit_cmd + " abinit.abi",
                               posix=(os.name == "posix"))
        self.nproc = nproc
        self.nomp_thread = nomp_thread

        self.logfile = logfile
        self.errfile = errfile
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
        # TESTING
        confdir.append("/Users/oliviernadeau/Projects/MLACS/03-test_MLACS/ntask_2/")
        for i, at in enumerate(confs):
            at.set_initial_magnetic_moments(self.magmoms)
            cdir = self.workdir + state[i] + f"/Step{step[i]}/"
            confdir.append(cdir)
            self._write_input(at, cdir)

        # I assume there is a possibility to have multiple Abinit Calculation at the same time
        # I simply distribute the number of processor equally between the task
        # nproc = Number of processors to use
        # len(confdir) = Number of different simulation to start

        # Define the function to call by every process.
        # Context : We need this function so the "with open logfile" doesn't close the logfile right after the calculation is submitted
        def submit_abinit_calc(cmd, cdir, logfile, errfile, nproc):
            cmd = cmd[0] + " " + cdir + cmd[1]
            mpi_cmd= "mpirun -np {nproc} {cmd} -j {nomp_thread}".format(nproc=proc_per_task, cmd=cmd, nomp_thread=self.nomp_thread)
            with open(cdir + logfile, 'w') as lfile, open(cdir + errfile, 'w') as efile:
                proc = Popen(mpi_cmd, cwd=cdir, stderr=efile, stdout=lfile, shell=True)
                proc.wait()

        ntask = len(confdir)
        nproc = self.nproc
        proc_per_task = 1 if nproc <= ntask else nproc//ntask # Divide the number of processor equally between the task.

        # Yeah for threading
        with ThreadPoolExecutor(max_workers=ntask) as executor:
            futures = [executor.submit(submit_abinit_calc, self.cmd, cdir, self.logfile, self.errfile, proc_per_task) for cdir in confdir]
        
        # Now we can read everything
        results_confs = []
        for cdir in confdir:
            self._read_output(cdir, results_confs)
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
    def _read_output(self, cdir, results_confs):
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
        calc = SPCalc(atoms,
                      energy=energy,
                      forces=forces,
                      stress=stress)
        calc.version = results.pop("version")
        atoms.calc = calc
        results_confs.append(atoms)

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
