from ase.calculators.calculator import Calculator


system_changes = ['positions', 'numbers', 'cell', 'pbc']


# ========================================================================== #
# ========================================================================== #
class MlipCalculator(Calculator):
    """
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
