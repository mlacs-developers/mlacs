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
        For example :
        pair_style = ['sw', 'zbl 3.0 4.0']
        pair_coeff = [['* * Si.sw Si'],
                      [* * 14 14]]

    pair_coeff: :class:`list` of :class:`str`
        The pair_coeff of the LAMMPS reference potential.
    """
    def __init__(self,
                 model,
                 pair_style,
                 pair_coeff,
                 model_post=None):
        self.model = model

        mbar = self.model.mbar
        ecoef = self.model.ecoef
        fcoef = self.model.fcoef
        scoef = self.model.scoef
        nthrow = self.model.nthrow

        MlipManager.__init__(self, self.model.descriptor,
                             nthrow, ecoef, fcoef, scoef, mbar)

        self.ref_pair_style = pair_style
        self.ref_pair_coeff = pair_coeff
        self.ref_model_post = model_post

        self._ref_e = None
        self._ref_f = None
        self._ref_s = None

        # For the rest of the
        # We need to create the hybrid/overlay format of LAMMPS
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
            self.pair_style = "hybrid/overlay "
            self.pair_coeff = []
            self.ref_pair_coeff = []
            for ps, pc in zip(pair_style, pair_coeff):
                self.pair_style += f"{ps} "
                self.ref_pair_style += f"{ps} "

                refpssplit = ps.split()
                for ppc in pc:
                    refpcsplit = ppc.split()
                    refpc = " ".join([*refpcsplit[:2],
                                      refpssplit[0],
                                      *refpcsplit[2:]])
                    self.pair_coeff.append(refpc)
                    self.ref_pair_coeff.append(refpc)
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            self.pair_style += f"{self.model.pair_style}"
            self.pair_coeff.append(mlpc)

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        # First compute reference energy/forces/stress
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        calc = LAMMPS(pair_style=self.ref_pair_style,
                      pair_coeff=self.ref_pair_coeff)
        dummy_at = []
        for at in atoms:
            at0 = at.copy()
            at0.calc = calc
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
    def __str__(self):
        txt = " ".join(self.elements)
        txt += "Delta Learning potential,"
        txt += str(self.model)

# ========================================================================== #
    def __repr__(self):
        txt = "Delta learning potential\n"
        txt += "------------------------\n"
        txt += "Reference potential :\n"
        txt += f"pair_style {self.ref_pair_style}\n"
        for pc in self.ref_pair_coeff:
            txt += f"pair_coeff {pc}\n\n"
        txt += "MLIP potential :\n"
        txt += repr(self.model)
        return txt
