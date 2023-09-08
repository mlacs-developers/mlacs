import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator

from .mlip_manager import MlipManager
from .delta_learning import DeltaLearningPotential


# ========================================================================== #
# ========================================================================== #
class SpinLatticePotential(DeltaLearningPotential):
    """
    Class to learn a MLIP on top of a spin-lattice potential as implemented
    in LAMMPS

    Parameters
    ----------
    model: :class:`MlipManager`
        The MLIP model to train on the difference between the true energy
        and the energy of a LAMMPS reference model.

    exchange: :class:`list`
        The parameters of the exchange/spin pair_style
    anisotropy: :class:`list`
        The value of the anisotropy. Uses the precession/spin anisotropy
        fix of the SPIN package in LAMMPS
    """
    def __init__(self,
                 model,
                 exchange,
                 exchange_rcut=4.0,
                 anisotropy=None
                 ):

        self.model = model

        ecoef = self.model.ecoef
        fcoef = self.model.fcoef
        scoef = self.model.scoef
        nthrow = self.model.nthrow

        MlipManager.__init__(self, self.model.descriptor,
                             nthrow, ecoef, fcoef, scoef)

        self.ref_pair_style = pair_style
        self.ref_pair_coeff = pair_coeff
        self.ref_model_post = model_post

        self._ref_e = None
        self._ref_f = None
        self._ref_s = None

        # First let's care of the exchange
        pair_style = [f"exchange/spin {exchange_rcut}"]
        pair_coeff = [[str(param) for param in exchange]]

        # Then the anisotropy
        self.anisotropy = anisotropy

        self._create_pair_styles_coeff(pair_style, pair_coeff)

# ========================================================================== #
    def update_matrices(self, atoms, spins):
        """
        """
        # First compute reference energy/forces/stress
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        # Make sure that the number of spins correspond to the atoms
        # given in the dataset
        assert spins.shape[0] == len(atoms)
        assert spins.shape[1] == len(atoms[0])
        assert spins.shape[2] == 3

        calc = LAMMPS(pair_style=self.ref_pair_style,
                      pair_coeff=self.ref_pair_coeff)

        energy = []
        forces = []
        stress = []
        dummy_at = []
        for at, spin in zip(atoms, spins):
            at0 = at.copy()

            at0 = self._compute_spin_properties(at, spin)
            refe = at0.get_potential_energy()
            reff = at0.get_forces()
            refs = at0.get_stress()

            dumdum = at.copy()
            e = at.get_potential_energy() - refe
            f = at.get_forces() - reff
            s = at.get_stress() - refs
            spcalc = SinglePointCalculator(dumdum,
                                           energy=e,
                                           forces=f,
                                           stress=s)
            dumdum.calc = spcalc
            dummy_at.append(dumdum)

        # Now get descriptor features
        self.model.update_matrices(dummy_at)
        self.nconfs = self.model.nconfs

# ========================================================================== #
    def train_mlip(self):
        """
        """
        msg = self.model.train_mlip()
        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff,
                      keep_alive=False)
        return calc

# ========================================================================== #
    def _compute_spin_properties(self, atoms, spin):
        write_atoms_lammps_spin_style(atoms, spin)
        self._write_lammps_input()
        self._run_lammps()
        at, energy, forces = self._read_out()
        calc = SinglePointCalculator(at, energy=energy, forces=forces,
                                     stress=stress)
        at.calc = calc
        return at
