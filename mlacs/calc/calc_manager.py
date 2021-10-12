"""
"""

class CalcManager:
    """
    """
    def __init__(self,
                 calc,
                 magmoms=None
                ):

        self.calc    = calc
        self.magmoms = magmoms

    def compute_true_potential(self, atoms):
        """
        """
        atoms.set_initial_magnetic_moments(self.magmoms)
        atoms.calc = self.calc
        atoms.get_potential_energy()

        return atoms
