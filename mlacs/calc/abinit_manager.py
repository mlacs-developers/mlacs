"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
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
    
    nomp_thread :class: `int` (optional)
        Number of OpenMP thread to use per processor.

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
                 nproc=1,
                 nomp_thread=1):

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

        if nomp_thread != 1:
            print("Warning : OpenMP is still in development.")
        self.nproc = nproc
        self.nomp_thread = nomp_thread

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
        confdir = []
        for i, at in enumerate(confs):
            at.set_initial_magnetic_moments(self.magmoms)
            cdir = self.workdir + state[i] + f"/Step{step[i]}/"
            confdir.append(cdir)
            self._write_input(at, cdir)

        # Define the function to be called by every process.
        def submit_abinit_calc(cmd, logfile, errfile):
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
        ntask = len(confdir)
        nproc = self.nproc
        proc_per_task = 1 if nproc <= ntask else nproc//ntask

        # Yeah for threading
        with ThreadPoolExecutor(max_workers=ntask) as executor:
            for cdir in confdir:
                command = self._make_command(cdir, proc_per_task)
                executor.submit(submit_abinit_calc,
                                command,
                                cdir+self.logfile,
                                cdir+self.errfile)

        # Now we can read everything
        results_confs = []
        for (cdir, at) in zip(confdir, confs):
            results_confs.append(self._read_output(cdir, at))
        # Tada !
        return results_confs

# ========================================================================== #
    def _make_command(self, cdir, nproc):
        """
        Make the command to call Abinit including MPI and OpenMP :
        env OMP_NUM_THREADS=1 mpirun -n 2 abinit /path/abinit.abi
        """
        abinit_cmd = self.abinit_cmd + " " + cdir + "abinit.abi"
        omp_cmd = "env OMP_NUM_THREADS={n}".format(n=self.nomp_thread)

        if nproc > 1:
            mpi_cmd = "{} -n {}".format(self.mpi_runner, nproc)
        else:
            mpi_cmd = ""

        full_cmd = "{} {} {}".format(omp_cmd, mpi_cmd, abinit_cmd)
        full_cmd = shlex.split(full_cmd, posix=(os.name == "posix"))

        return full_cmd

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
        if self.ncfile is not None:
            dct = self.ncfile.read(cdir + "abinito_GSR.nc")
            results.update(dct)
            atoms = set_aseAtoms(results)
            atoms.set_velocities(at.get_velocities())
            return atoms

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
