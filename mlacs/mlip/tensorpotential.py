'''
'''
import sys
import numpy as np

from . import MlipManager
from .weights import UniformWeight
from ..utilities import compute_correlation, create_link

from ase.units import GPa
from ase import Atoms
from pathlib import Path


# ========================================================================== #
# ========================================================================== #
class TensorpotPotential(MlipManager):
    """
    Potential that use Tensorpotential to minimize a cost function.
    For now, only works with AceDescriptor

    Parameters
    ----------
    descriptor: :class:`Descriptor`
        The descriptor used in the model.
    weight: :class:`WeightingPolicy`
        Weight used for the fitting and calculation of properties.
        Default :class:`None`
    """
    def __init__(self,
                 descriptor,
                 weight=None,
                 **kwargs):
        if weight is None:
            weight = UniformWeight(energy_coefficient=1.0,
                                   forces_coefficient=1.0,
                                   stress_coefficient=0.0)

        MlipManager.__init__(self,
                             descriptor=descriptor,
                             weight=weight,
                             **kwargs)
        self.natoms = []
        self.nconfs = 0

        if self.weight.stress_coefficient != 0:
            raise ValueError("Tensorpotential can't fit on stress")

        self.atoms = []
        self.coefficients = []

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.weight.update_database(atoms)
        self.nconfs += len(atoms)
        self.natoms.extend([len(_) for _ in atoms])
        self.atoms.extend(atoms)

# ========================================================================== #
    def train_mlip(self):
        """
        """
        if self.descriptor.log is None:
            self.descriptor.redirect_logger()

        self.weight.workdir = self.workdir
        self.weight.folder = self.folder
        self.weight.subfolder = self.subfolder
        self.descriptor.workdir = self.workdir
        self.descriptor.folder = self.folder
        self.descriptor.subfolder = self.subfolder

        W = self.weight.get_weights()

        fitting_log = self.descriptor.log.handlers[0].baseFilename
        with open(fitting_log, "a") as f:
            sys.stdout = f
            coef_fn = self.descriptor.fit(weights=W, atoms=self.atoms)
            full_fn = self.workdir / self.folder / self.subfolder / coef_fn
            self.coefficients = full_fn
            sys.stdout = sys.__stdout__

        msg = "Number of configurations for training: " + \
              f"{self.nconfs}\n"
        msg += "Number of atomic environments for training: " + \
               f"{sum(self.natoms)}\n"

        tmp_msg, weight_fn = self.weight.compute_weight(
            self.coefficients,
            self.predict)

        msg += tmp_msg
        msg += self.compute_tests()

        create_link(self.subsubdir / weight_fn, self.subdir/weight_fn)
        create_link(self.subsubdir / coef_fn, self.subdir/coef_fn)
        return msg

# ========================================================================== #
    def read_parent_mlip(self, traj):
        """
        Get a list of all the mlip that have generated a conf in traj
        and get the coefficients of all these mlip
        """
        parent_mlip = []
        mlip_coef = []
        directory = self.descriptor.subdir

        # Make the MBAR variable Nk and mlip_coef
        for state in traj:
            for conf in state:
                if "parent_mlip" not in conf.info:  # Initial or training
                    continue
                else:  # A traj
                    model = conf.info['parent_mlip']
                    directory = Path(model)
                    if not directory.exists:
                        err = "Some parent MLIP are missing. "
                        err += "Rerun MLACS with DatabaseCalculator and "
                        err += "OtfMlacs.keep_tmp_files=True on your traj"
                        raise FileNotFoundError(err)
                    if model not in parent_mlip:  # New state
                        parent_mlip.append(model)
                        coef = self.descriptor.get_mlip_file(folder=model)
                        mlip_coef.append(coef)
        return parent_mlip, np.array(mlip_coef)

# ========================================================================== #
    def next_coefs(self, mlip_coef):
        """
        Update MLACS just like train_mlip, but without actually computing
        the coefficients
        """
        self.weight.subfolder = self.subfolder
        self.descriptor.subfolder = self.subfolder

        self.coefficients = mlip_coef

        self.descriptor.set_restart_coefficient(mlip_coef)
        _, weight_fn = self.weight.compute_weight(mlip_coef, self.predict)

        create_link(mlip_coef + "/" + weight_fn, self.subdir/weight_fn)
        create_link(mlip_coef + "/" + "ACE.yace", self.subdir/"ACE.yace")

# ========================================================================== #
    def compute_tests(self):
        """
        Compute the weighted RMSE and MAE
        """
        mlip_e, mlip_f, mlip_s = self.predict(desc=self.atoms)
        true_e = np.array([at.get_potential_energy()/len(at)
                          for at in self.atoms])
        true_f = [at.get_forces() for at in self.atoms]
        true_s = [at.get_stress() for at in self.atoms]

        mlip_f = np.reshape(mlip_f, [-1])
        mlip_s = np.reshape(mlip_s, [-1])
        true_f = np.reshape(true_f, [-1])
        true_s = np.reshape(true_s, [-1])

        w = None
        if len(self.weight.weight) > 0:
            w = self.weight.weight

        res_E = compute_correlation(np.c_[true_e, mlip_e], weight=w)
        res_F = compute_correlation(np.c_[true_f, mlip_f], weight=w)
        res_S = compute_correlation(np.c_[true_s, mlip_s]/GPa, weight=w)
        r = np.c_[res_E, res_F, res_S]
        self.fit_res = r

        # Information to MLIP-Energy_comparison.dat
        header = f"Weighted rmse: {self.fit_res[0, 0]:.6f} eV/at,    " + \
                 f"Weighted mae: {self.fit_res[1, 0]:.6f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("MLIP-Energy_comparison.dat",
                   np.c_[true_e, mlip_e],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {self.fit_res[0, 1]:.6f} eV/angs   " + \
                 f"Weighted mae: {self.fit_res[1, 1]:.6f} eV/angs\n" + \
                 " True Forces           Predicted Forces"

        np.savetxt("MLIP-Forces_comparison.dat",
                   np.c_[true_f, mlip_f],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"Weighted rmse: {self.fit_res[0, 2]:.6f} GPa       " + \
                 f"Weighted mae: {self.fit_res[1, 2]:.6f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("MLIP-Stress_comparison.dat",
                   np.c_[true_s, mlip_s] / GPa,
                   header=header, fmt="%25.20f  %25.20f")

        # Message to Mlacs.log
        msg = f"Weighted RMSE Energy    {self.fit_res[0, 0]:.4f} eV/at\n"
        msg += f"Weighted MAE Energy     {self.fit_res[1, 0]:.4f} eV/at\n"
        msg += f"Weighted RMSE Forces    {self.fit_res[0, 1]:.4f} eV/angs\n"
        msg += f"Weighted MAE Forces     {self.fit_res[1, 1]:.4f} eV/angs\n"
        msg += f"Weighted RMSE Stress    {self.fit_res[0, 2]:.4f} GPa\n"
        msg += f"Weighted MAE Stress     {self.fit_res[1, 2]:.4f} GPa\n"
        return msg

# ========================================================================== #
    def get_calculator(self):
        """
        Initialize a ASE calculator from the model
        """
        from .calculator import MlipCalculator
        calc = MlipCalculator(self)
        return calc

# ========================================================================== #
    def predict(self, desc, coef=None):
        """
        Give the energy forces stress of atoms according to the potential.
        """
        if coef is None:
            coef = self.coefficients
        if isinstance(coef, str):
            coef = Path(coef)
        return self.descriptor.predict(desc, coef, folder=self.subdir)

# ========================================================================== #
    def __str__(self):
        dname = type(self.descriptor).__name__
        txt = f"Tensorpotential potential with {dname}"
        return txt

# ========================================================================== #
    def __repr__(self):
        txt = "Tensorpotential\n"
        txt += "Parameters:\n"
        txt += "-----------\n"
        txt += f"Weight : {type(self.weight).__name__}"
        txt += "\n"
        txt += "Descriptor used in the potential:\n"
        txt += repr(self.descriptor)
        return txt
