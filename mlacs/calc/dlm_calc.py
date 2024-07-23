"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import CalculatorError
try:
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells
    isicet = True
except ImportError:
    isicet = False

from . import CalcManager


# ========================================================================== #
# ========================================================================== #
class DlmCalcManager(CalcManager):
    """
    Class for Disorder Local Moment simulation.

    Disorder Local Moment is a method to simulate an antiferromagnetic
    material by imposing periodically a random spin configuration
    by means of Special Quasirandom Structures.

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object.

    unitcell: :class:`ase.atoms`
        The Atoms object for the unitcell, for which the symmetry will
        be used to create the magnetic sqs.

    supercell: :class:`ase.atoms`
        The supercell with ideal positions

    magnetic_sites: :class:`list`
        List of integers describing the unitcell sites where
        the magnetic atoms are located

    mu_b: :class:`float`
        The initial spin amplitude, imposed before the calculation,
        in Bohr magneton. Default ``1.0``.

    cutoff: :class:`list` of :class:`float`
        The cutoffs for the SQS generation.
        See icet documentation for more information. Default ``[6.0, 4.0]``.

    n_steps: :class:`int` (optional)
        Number of Monte-Carlo steps for the generation of the magnetic SQS.
        Default ``3000``.
    """
    def __init__(self,
                 calc,
                 unitcell,
                 supercell,
                 magnetic_sites,
                 mu_b=1.0,
                 cutoffs=[6.0, 4.0],
                 n_steps=3000,
                 **kwargs):
        CalcManager.__init__(self, calc, **kwargs)

        if not isicet:
            msg = "You need the icet package installed to use a DLM calculator"
            raise ModuleNotFoundError(msg)

        chemsymb = [["N"]] * len(unitcell)
        for i in magnetic_sites:
            chemsymb[i] = ["H", "B"]  # H -> haut et B -> bas
        self.cutoffs = cutoffs
        self.mu_b = mu_b
        self.supercell = supercell.copy()
        self.cs = ClusterSpace(unitcell, cutoffs, chemsymb)
        self.n_steps = n_steps
        self.target_concentrations = {"H": 0.5, "B": 0.5}

# ========================================================================== #
    def compute_true_potential(self,
                               confs,
                               state=None,
                               step=None):
        """
        Compute the energy of given configurations with an ASE calculator,
        using a random spin configuration from the SQS.
        """
        confs = [at.copy() for at in confs]
        result_confs = []
        for at in confs:
            sqs = generate_sqs_from_supercells(self.cs,
                                               [self.supercell],
                                               self.target_concentrations,
                                               n_steps=self.n_steps)
            magmoms = np.zeros(len(self.supercell))
            for i, c in enumerate(sqs.get_chemical_symbols()):
                if c == "H":
                    magmoms[i] = self.mu_b
                if c == "B":
                    magmoms[i] = -self.mu_b
            at.set_initial_magnetic_moments(magmoms)
            at.calc = self.calc
            try:
                at.get_potential_energy()
                energy = at.get_potential_energy()
                forces = at.get_forces()
                stress = at.get_stress()
                sp_calc = SinglePointCalculator(at,
                                                energy=energy,
                                                forces=forces,
                                                stress=stress)
                at.calc = sp_calc
                result_confs.append(at)
            except CalculatorError:
                result_confs.append(None)
        return result_confs

# ========================================================================== #
    def log_recap_state(self):
        """
        """
        name = self.calc.name

        msg = "True potential parameters:\n"
        msg += "Calculator : {0}\n".format(name)
        if hasattr(self.calc, "todict"):
            dct = self.calc.todict()
            msg += "parameters :\n"
            for key in dct.keys():
                msg += "   " + key + f"  {dct[key]}\n"
        msg += "Disorder Local Moment method for antifferomagnetic spin\n"
        msg += f"Initial absolute magnetic moment : {self.mu_b}\n"
        msg += "Cutoffs : " + " ".join([str(c) for c in self.cutoffs]) + "\n"
        msg += "Number of Monte-Carlo steps for the sqs generation : " + \
               f"{self.n_steps}\n"
        msg += "Cluster Space:\n"
        msg += str(self.cs)
        msg += "\n"
        return msg
