"""
"""
from mlacs.calc import CalcManager

from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells


class DLMCalcManager(CalcManager):
    """
    """
    def __init__(self, calc, atoms_ideal, mu_b=1.0, cutoff=[6.0, 4.0], n_steps=3000):
        CalcManager.__init__(self, calc,)

        self.mu_b      = mu_b
        self.supercell = atoms_ideal.copy()
        self.cs        = ClusterSpace(self.supercell, cutoff, ["H", "B"]) # H -> haut et B -> bas
        self.n_steps   = n_steps
        self.target_concentrations = {"H": 0.5, "B": 0.5}
        print(self.cs)


    def compute_true_potential(self, atoms):
        """
        """
        sqs = generate_sqs_from_supercells(self.cs,
                                           [self.supercell],
                                           self.target_concentrations,
                                           n_steps=self.n_steps
                                          )
        magmoms = []
        for i in sqs.get_chemical_symbols():
            if i == "H":
                magmoms.append( self.mu_b)
            if i == "B":
                magmoms.append(-self.mu_b)
        atoms.set_initial_magnetic_moments(magmoms)
        atoms.calc = self.calc
        atoms.get_potential_energy()
        return atoms
