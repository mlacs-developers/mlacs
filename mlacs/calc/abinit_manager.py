"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""
import os
import shutil
import numpy as np
import shlex

# IMPORTANT : subprocess->Popen doesnt work if we import run, PIPE
from subprocess import Popen
from subprocess import check_output
import logging
from concurrent.futures import ThreadPoolExecutor

from ase import Atom
from ase.symbols import symbols2numbers
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from ase.io.abinit import (write_abinit_in,
                           read_abinit_out)

from ..core.manager import Manager
from .calc_manager import CalcManager
from ..utilities import save_cwd
from ..utilities.io_abinit import AbinitNC


# ========================================================================== #
# ========================================================================== #
class AbinitManager(CalcManager):
    """
    This Calc class is an extended object for Abinit calculators.
    The AbinitManager can handdle netCDF files, MPI processes and a better
    pseudopotentials files management.

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

    folder: :class:`str` (optional)
        The root for the directory in which the computation are to be done
        Default 'DFT'

    logfile: :class:`str` (optional)
        The name of the Abinit log file inside the workdir folder
        Default 'abinit.log'

    errfile: :class:`str` (optional)
        The name of the Abinit error file inside the workdir folder
        Default 'abinit.err'

    nproc: :class:`int` (optional)
        Number of processor available for all Abinit.

    nproc_per_task: :class:`int` (optional)
        Number of processor available for all Abinit.
        Default nproc


    Examples
    --------

    >>> from mlacs.calc import AbinitManager
    >>> variables = dict(ixc=-1012, ecut=12, tsmear=0.001, occopt=3, nband=82,
    >>>                  ngkpt=[2, 2, 2], shiftk=[0.5, 0.5, 0.5],
    >>>                  autoparall=1, nsym=1) # Cu, 8 atoms.
    >>> pseudos = {'Cu': "/path/to/pseudo/Cu.LDA_PW-JTH.xml"}
    >>> calc = AbinitManager(parameters=variables, pseudos=pseudos)
    """

    def __init__(self,
                 parameters,
                 pseudos,
                 abinit_cmd="abinit",
                 mpi_runner="mpirun",
                 magmoms=None,
                 folder='DFT',
                 logfile="abinit.log",
                 errfile="abinit.err",
                 nproc=1,
                 nproc_per_task=None,
                 **kwargs):

        CalcManager.__init__(self, "dummy", magmoms,
                             folder=folder, **kwargs)

        self.parameters = parameters
        if 'IXC' in self.parameters.keys():
            self.parameters['ixc'] = self.parameters['IXC']
            del self.parameters['IXC']
        if 'ixc' not in self.parameters.keys():
            msg = 'WARNING AbinitManager:\n'
            msg += 'You should specify an ixc value or ASE will set 7 (LDA) !'
            msg += '\n(Does not apply if using a PAW xml format)'
            logging.warning(msg)

        self._organize_pseudos(pseudos)
        self.abinit_cmd = abinit_cmd
        self.mpi_runner = mpi_runner
        self.nproc = nproc
        if nproc_per_task is None:
            nproc_per_task = self.nproc
        self.nproc_per_task = nproc_per_task

        self.log = logfile
        self.err = errfile
        self.ncfile = AbinitNC()

# ========================================================================== #
    @staticmethod
    def submit_abinit_calc(cmd, logfile, errfile, cdir):
        with open(logfile, 'w') as lfile, \
                open(errfile, 'w') as efile:
            try:
                process = Popen(cmd,
                                cwd=cdir,
                                stderr=efile,
                                stdout=lfile,
                                shell=False)
                process.wait()
            except Exception as e:
                msg = f"This command {' '.join(cmd)}\n"
                msg += f"raised this exception {e}"
                efile.write(msg)

# ========================================================================== #
    @Manager.exec_from_subdir
    def compute_true_potential(self, confs: [Atom],
                               subfolder: [str],
                               step: [int]):
        """
        Compute the energy/forces/stress of given configurations with Abinit.

        Parameters
        ----------
        confs: :class:`list` of :class:`ase.Atoms`
            The input list of atom objects.

        subfolder: :class:`list` of :class:`str` (optional)
            Subfolder in which the properties are saved.

        step: :class:`list` of :class:`int`  (optional)
            The list of configuration indices.

        Returns
        -------
        result_confs: :class:`list` of :class:`ase.Atoms`
            The output list of atom objects, with corresponding
            SinglePointCalculator resulting from true potential calculation.
            See also _read_output() where SinglePointCalculator() is called.
        """
        assert len(confs) == len(subfolder) == len(step)
        nparal = self.nproc // self.nproc_per_task

        # Prepare all calculations
        confs = [at.copy() for at in confs]
        path_prefix_l = []
        for at, sf, istep in zip(confs, subfolder, step):
            # First set the prefix
            self.subfolder = sf
            self.prefix = str(self.subfolder) + '_'
            # Then append a level to subsubdir
            self.subsubdir = self.subsubdir / f"Step{istep}"
            # Save this list for parallel execution
            path_prefix_l.append((self.subsubdir, self.prefix))
            # Initialize objects
            at.set_initial_magnetic_moments(self.magmoms)
            self._write_input(at)

        # Yeah for threading
        # GA: I would move the threading outside of this function
        # because the files naming depends on external objects.
        with save_cwd(), ThreadPoolExecutor(max_workers=nparal) as executor:
            for (path, pref) in path_prefix_l:
                self.subsubdir = path
                self.prefix = pref
                command = self._make_command()
                executor.submit(self.submit_abinit_calc,
                                command,
                                self.get_filepath(self.log),
                                self.get_filepath(self.err),
                                cdir=str(self.subsubdir))
            executor.shutdown(wait=True)

        # Now we can read everything
        results_confs = []
        for ((path, pref), at) in zip(path_prefix_l, confs):
            self.subsubdir = path
            self.prefix = pref
            results_confs.append(self._read_output(at))

        for i in range(len(results_confs)):
            results_confs[i].info = confs[i].info

        # Reset values for good measure...
        self.prefix = ''
        self.subfolder = ''

        # Tada !
        return results_confs

# ========================================================================== #
    def _make_command(self):
        """
        Make the command to call Abinit including MPI :
        """
        abinit_cmd = self.abinit_cmd + " " + self.get_filepath("abinit.abi")
        nproc = self.nproc_per_task

        if nproc > 1:
            mpi_cmd = "{} -n {}".format(self.mpi_runner, nproc)
        else:
            mpi_cmd = ""

        full_cmd = "{} {}".format(mpi_cmd, abinit_cmd)
        full_cmd = shlex.split(full_cmd, posix=(os.name == "posix"))
        return full_cmd

# ========================================================================== #
    @Manager.exec_from_subdir
    def _write_input(self, atoms):
        """
        Write the input for the current atoms
        """
        if self.subsubdir.exists():
            self._remove_previous_run(str(self.subsubdir))
        else:
            self.subsubdir.mkdir(parents=True)

        # First we need to prepare some stuff
        original_pseudos = self.pseudos.copy()
        species = sorted(set(atoms.numbers))
        self._copy_pseudos()

        unique_elements = set(atoms.get_chemical_symbols())
        pseudos = [pseudo for pseudo, el in zip(self.pseudos, self.typat) if
                   el in unique_elements]

        with open(self.get_filepath("abinit.abi"), "w") as fd:
            write_abinit_in(fd,
                            atoms,
                            self.parameters,
                            species,
                            pseudos)
        self.pseudos = original_pseudos

# ========================================================================== #
    def _copy_pseudos(self):
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
            dest = self.get_filepath(fn)
            _create_copy(source, dest)
            new_psp.append(dest)
        self.pseudos = new_psp
        self.parameters.pop('pspdir', None)

# ========================================================================== #
    def _read_output(self, at):
        """
        """
        results = {}
        if 'None' not in str(self.ncfile):
            ncpath = self.get_filepath("abinito_GSR.nc")
            self.ncfile.read(filename=ncpath)
            atoms = self.ncfile.convert_to_atoms()[0]
            atoms.set_velocities(at.get_velocities())
            return atoms

        with open(self.get_filepath("abinit.abo")) as fd:
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
        typat = np.array(typat)
        pseudolist = np.array(pseudolist)

        self.typat = typat
        znucl = symbols2numbers(typat)
        idx = np.argsort(znucl)

        # Reorder in increasing Z
        self.typat = typat[idx]
        self.pseudos = pseudolist[idx]

# ========================================================================== #
    def _remove_previous_run(self, stateprefix):
        """
        Little function to remove any trace of previous calculation
        """
        if os.path.exists(stateprefix + "abinit.abi"):
            os.remove(stateprefix + "abinit.abi")
        if os.path.exists(stateprefix + "abinit.abo"):
            os.remove(stateprefix + "abinit.abo")
        if os.path.exists(stateprefix + self.log):
            os.remove(stateprefix + self.log)
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

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        cmd = self.abinit_cmd
        cmd += ' --version'
        version = check_output(cmd, shell=True).decode('utf-8')
        msg = "True potential parameters:\n"
        msg += f"Abinit : {version}\n"
        dct = self.parameters
        msg += "parameters :\n"
        for key in dct.keys():
            msg += "   " + key + "  {0}\n".format(dct[key])
        msg += "\n"
        return msg
