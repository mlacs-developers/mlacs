"""
// (c) 2022 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import shlex
from subprocess import call, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ase.symbols import symbols2numbers
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc
from ase.io.abinit import (write_abinit_in,
                           read_abinit_out)

from mlacs.calc.calc_manager import CalcManager


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
                executor.submit(call,
                                self.cmd,
                                cwd=cdir,
                                stdout=PIPE,
                                stderr=PIPE)

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
