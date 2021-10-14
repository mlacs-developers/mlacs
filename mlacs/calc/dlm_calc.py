"""
"""
from mlacs.calc import CalcManager

try:
    from icet import ClusterSpace
    from icet.tools.structure_generation import generate_sqs_from_supercells
except:
    raise ImportError



#===================================================================================================================================================#
#===================================================================================================================================================#
class DLMCalcManager(CalcManager):
    """
    Class for Disorder Local Moment simulation.

    Disorder Local Moment is a method to simulate an antiferromagnetic material by
    imposing periodically a random spin configuration by means of Special Quasirandom Structures.

    Parameters
    ----------
    calc: :class:`ase.calculator`
        A ASE calculator object.
    atoms_ideal: :class:`ase.atoms`
        The Atoms object with ideal positions, for which the disorder spin structure will be computed.
    mu_b: float
        The initial spin amplitude, imposed before the calculation, in Bohr magneton.
    cutoff: list of float
        The cutoffs for the SQS generation. See icet documentation for more information.
    n_steps: int
        Number of Monte-Carlo steps for the generation of the magnetic SQS.
    """
    def __init__(self, calc, atoms_ideal, mu_b=1.0, cutoffs=[6.0, 4.0], n_steps=3000):
        CalcManager.__init__(self, calc,)

        self.cutoffs   = cutoffs
        self.mu_b      = mu_b
        self.supercell = atoms_ideal.copy()
        self.cs        = ClusterSpace(self.supercell, cutoffs, ["H", "B"]) # H -> haut et B -> bas
        self.n_steps   = n_steps
        self.target_concentrations = {"H": 0.5, "B": 0.5}


#===================================================================================================================================================#
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


#===================================================================================================================================================#
    def log_recap_state(self):
        """
        """
        name = self.calc.name

        msg  = "True potential parameters:\n"
        msg += "Calculator : {0}\n".format(name)
        if hasattr(self.calc, "todict"):
            dct = self.calc.todict()
            msg += "parameters :\n"
            for key in dct.keys():
                msg += "   " + key + "  {0}\n".format(dct[key])
        msg += "Disorder Local Moment method for antifferomagnetic spin\n"
        msg += "Initial absolute magnetic moment : {0}\n".format(self.mu_b)
        msg += "Cutoffs : " + " ".join([str(c) for c in self.cutoffs]) + "\n"
        msg += "Number of Monte-Carlo steps for the sqs generation : {0}\n".format(self.n_steps)
        msg += "Cluster Space:\n"
        msg += str(self.cs)
        msg += "\n"
        return msg
