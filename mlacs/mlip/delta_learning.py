import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator

from .mlip_manager import MlipManager


# ========================================================================== #
# ========================================================================== #
class DeltaLearningPotential(MlipManager):
    """
    Parameters
    ----------
    model: :class:`MlipManager`
        The MLIP model to train on the difference between the true energy
        and the energy of a LAMMPS reference model.

    pair_style: :class:`str` or :class:`list` of :class:`str`
        The pair_style of the LAMMPS reference potential.
        If only one pair style is used, can be set as a :class:`str`.
        If an overlay of pair style is used, this input as to be a
        :class:`list` of :class:`str` of the pair_style.

    pair_coeff: :class:`list` of :class:`str`
        The pair_coeff of the LAMMPS reference potential.
    """
    def __init__(self,
                 model,
                 pair_style,
                 pair_coeff,
                 model_post=None):
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
            
        # For the rest of the
        # We need to create the hybrid/overlay format of LAMMPS
        """
        pair_style  hybrid/overlay pair_style1 ... pair_styleN
        pair_coeff1 * * pair_style1[0] pair_coeff1
                            .
                            .
                            .
        pair_coeffN * * pair_styleN[0] pair_coeffN
        """

        if not isinstance(pair_style, list):
            pair_style = [pair_style]

        # First let's take care of only one reference potential
        if len(pair_style) == 1:
            self.ref_pair_style = pair_style[0]
            self.pair_style = f"hybrid/overlay {pair_style[0]} " + \
                              f"{self.model.pair_style}"
            self.ref_pair_coeff = pair_coeff
            refpcsplit = pair_coeff[0].split()
            refpssplit = pair_style[0].split()
            refpc = " ".join([*refpcsplit[:2],
                              refpssplit[0],
                              *refpcsplit[2:]])
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            self.pair_coeff = [refpc, mlpc]

        # And now with an overlay reference potential
        else:
            self.ref_pair_style = "hybrid/overlay "
            self.pair_coeff = "hybrid/overlay "
            for ps, pc in zip(pair_style, pair_coeff):
                self.pair_style += f"{ps} "
                self.ref_pair_style += f"{ps} "

                refpcsplit = pc.split()
                refpssplit = ps.split()
                refpc = " ".join([*refpcsplit[:2],
                                  refpssplit[0],
                                  *refpcsplit[2:]])
                self.pair_coeff.append([refpc])
                self.ref_pair_coeff.append([refpc])
            mlpcsplit = self.model.pair_coeff.split()
            mlpssplit = self.pair_style[0].split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            self.pair_style += f"{self.model.pair_style}"
            self.pair_coeff.append(mlpc)

        print(self.pair_style)
        print(self.pair_coeff)
        print()
        print(self.ref_pair_style)
        print(self.ref_pair_coeff)

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        # First compute reference energy/forces/stress
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        calc = LAMMPS(pair_style=self.ref_pair_style,
                      pair_coeff=self.ref_pair_coeff)
        energy = []
        forces = []
        stress = []
        dummy_at = []
        for at in atoms:
            at0 = at.copy()
            at0.calc = calc
            refe = at0.get_potential_energy()
            reff = at0.get_forces()
            refs = at0.get_stress()

            """
            energy.append(at0.get_potential_energy())
            forces.extend(at0.get_forces().flatten())
            stress.extend(at0.get_stress())
            """

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

        """
        if self._ref_e is None:
            self._ref_e = energy
            self._ref_f = forces
            self._ref_s = stress
        else:
            self._ref_e = np.r_[self._ref_e, energy]
            self._ref_f = np.r_[self._ref_f, forces]
            self._ref_s = np.r_[self._ref_s, stress]

        dummy_at = []
        for i, at in enumerate(atoms):
            dumdum = at.copy()
            e = at.get_potential_energy() - energy[i]
            f = at.get_forces() - forces[i]
            s = at.get_stress() - stress[i]
            spcalc = SinglePointCalculator(dumdum,
                                           energy=e,
                                           forces=f,
                                           stress=s)
            dumdum.calc = spcalc
            dummy_at.append(dumdum)

        for at in dummy_at:
            print(at.get_potential_energy())
        """

        # Now get descriptor features
        self.model.update_matrices(dummy_at)
        self.nconfs = self.model.nconfs

# ========================================================================== #
    def train_mlip(self):
        """
        """
        self.model.train_mlip()

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff,
                      keep_alive=False)
        return calc
