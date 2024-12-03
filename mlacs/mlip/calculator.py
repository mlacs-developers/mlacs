"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from ase.calculators.calculator import Calculator


system_changes = ['positions', 'numbers', 'cell', 'pbc']


# ========================================================================== #
# ========================================================================== #
class MlipCalculator(Calculator):
    """
    Ase Caculator object for MLACS generated MLIP that can take atomic
    configurations from an Atoms object and calculate the energy, forces and
    also stresses.
    """
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, model, **kwargs):
        """
        """
        Calculator.__init__(self, **kwargs)
        self.model = model

# ========================================================================== #
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces, stress = self.model.predict(atoms)

        self.results['energy'] = energy
        self.results['forces'] = forces.reshape(len(atoms), 3)
        self.results['stress'] = stress
