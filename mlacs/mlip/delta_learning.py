from pathlib import Path

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
                 model_post=None,
                 atom_style="atomic",
                 folder=Path("MLIP")):
        self.model = model

        weight = self.model.weight
        ecoef = self.model.ecoef
        fcoef = self.model.fcoef
        scoef = self.model.scoef
        nthrow = self.model.nthrow

        MlipManager.__init__(self, self.model.descriptor, nthrow,
                             ecoef, fcoef, scoef, weight, folder)

        if not isinstance(pair_style, list):
            pair_style = [pair_style]

        self.ref_pair_style = pair_style
        self.ref_pair_coeff = pair_coeff
        self.ref_model_post = model_post
        self.model_post = model_post
        self.ref_atom_style = atom_style
        self.atom_style = atom_style

        self._ref_e = None
        self._ref_f = None
        self._ref_s = None

# ========================================================================== #
    def get_ref_pair_style(self, lmp=False):
        """
        Return self.ref_pair_style which is an array.
        If lmp=True, it returns it formatted as a lammps input.
        """
        if not lmp:
            return self.ref_pair_style

        if len(self.ref_pair_style) == 1:
            return self.ref_pair_style[0]
        else:  # Here the tricky part. I need to create hybrid overlay ...
            full_pair_style = "hybrid/overlay "
            for ps in self.ref_pair_style:
                full_pair_style += f"{ps} "
            return full_pair_style

# ========================================================================== #
    @property
    def pair_style(self):
        # We need to create the hybrid/overlay format of LAMMPS
        if not isinstance(self.ref_pair_style, list):
            self.ref_pair_style = [self.ref_pair_style]

        if len(self.ref_pair_style) == 1:
            full_pair_style = f"hybrid/overlay {self.ref_pair_style[0]} " + \
                              f"{self.model.pair_style}"
        else:
            full_pair_style = "hybrid/overlay "
            for ps in self.ref_pair_style:
                full_pair_style += f"{ps} "
            full_pair_style += f"{self.model.pair_style}"

        return full_pair_style

# ========================================================================== #
    @property
    def pair_coeff(self):
        if not isinstance(self.ref_pair_style, list):
            self.ref_pair_style = [self.ref_pair_style]

        # First let's take care of only one reference potential
        if len(self.ref_pair_style) == 1:
            refpcsplit = self.ref_pair_coeff[0].split()
            refpssplit = self.ref_pair_style[0].split()
            refpc = " ".join([*refpcsplit[:2],
                              refpssplit[0],
                              *refpcsplit[2:]])
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            full_pair_coeff = [refpc, mlpc]

        # And now with an overlay reference potential
        else:
            full_pair_coeff = []
            for ps, pc in zip(self.ref_pair_style, self.ref_pair_coeff):
                refpssplit = ps.split()
                for ppc in pc:
                    refpcsplit = ppc.split()
                    refpc = " ".join([*refpcsplit[:2],
                                      refpssplit[0],
                                      *refpcsplit[2:]])
                    full_pair_coeff.append(refpc)
            mlpcsplit = self.model.pair_coeff[0].split()
            mlpssplit = self.model.pair_style.split()
            mlpc = " ".join([*mlpcsplit[:2],
                             mlpssplit[0],
                             *mlpcsplit[2:]])
            full_pair_coeff.append(mlpc)

        return full_pair_coeff

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        # First compute reference energy/forces/stress
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        calc = LAMMPS(pair_style=self.get_ref_pair_style(lmp=True),
                      pair_coeff=self.ref_pair_coeff,
                      atom_style=self.ref_atom_style)

        if self.model_post is not None:
            calc.set(model_post=self.ref_model_post)

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
    def next_coefs(self, mlip_coef, mlip_subfolder):
        """
        """
        msg = self.model.next_coefs(mlip_coef, mlip_subfolder)
        return msg

# ========================================================================== #
    def train_mlip(self, mlip_subfolder):
        """
        """
        msg = self.model.train_mlip(mlip_subfolder=mlip_subfolder)
        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff,
                      atom_style=self.atom_style,
                      keep_alive=False)
        if self.model_post is not None:
            calc.set(model_post=self.model_post)
        return calc

# ========================================================================== #
    def __str__(self):
        txt = "Delta Learning potential\n"
        txt += f"Reference pair_style: {self.ref_pair_style}\n"
        txt += f"Reference pair_coeff: {self.ref_pair_coeff}\n"
        txt += str(self.model)
        return txt

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
