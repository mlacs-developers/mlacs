"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

from ase.atoms import Atoms
from ase.units import GPa

from ..core import Manager
from ..utilities import compute_correlation, create_link
from .weighting_policy import UniformWeight


# ========================================================================== #
# ========================================================================== #
class MlipManager(Manager):
    """
    Parent Class for the management of Machine-Learning Interatomic Potential
    """
    def __init__(self,
                 descriptor,
                 weight=None,
                 folder='MLIP',
                 **kwargs):

        Manager.__init__(self, folder=folder, **kwargs)

        self.descriptor = descriptor
        self.descriptor.workdir = self.workdir
        self.descriptor.folder = self.folder

        self.amat_e = None
        self.amat_f = None
        self.amat_s = None

        self.ymat_e = None
        self.ymat_f = None
        self.ymat_s = None

        self.natoms = []

        self.fit_res = None

        self.weight = weight
        if self.weight is None:
            self.weight = UniformWeight()

        self.weight.workdir = self.workdir
        self.weight.folder = self.folder
        self.weight.subfolder = self.subfolder

        self.nconfs = 0

        # Some initialization for sampling interface
        self.model_post = None
        self.atom_style = "atomic"

        self.can_use_weight = False

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        self.weight.update_database(atoms)

        self.descriptor.workdir = self.workdir
        self.descriptor.folder = self.folder
        amat_all = self.descriptor.calculate(atoms)

        energy = np.array([at.get_potential_energy() for at in atoms])
        forces = []
        for at in atoms:
            forces.extend(at.get_forces().flatten())
        forces = np.array(forces)
        stress = np.array([at.get_stress() for at in atoms]).flatten()
        nat = np.array([len(at) for at in atoms])

        for amat in amat_all:
            if self.amat_e is None:
                self.amat_e = amat["desc_e"]
                self.amat_f = amat["desc_f"]
                self.amat_s = amat["desc_s"]
            else:
                self.amat_e = np.r_[self.amat_e, amat["desc_e"]]
                self.amat_f = np.r_[self.amat_f, amat["desc_f"]]
                self.amat_s = np.r_[self.amat_s, amat["desc_s"]]

        if self.ymat_e is None:
            self.ymat_e = energy
            self.ymat_f = forces
            self.ymat_s = stress
            self.natoms = nat
        else:
            self.ymat_e = np.r_[self.ymat_e, energy]
            self.ymat_f = np.r_[self.ymat_f, forces]
            self.ymat_s = np.r_[self.ymat_s, stress]
            self.natoms = np.append(self.natoms, [nat])
        self.nconfs += len(atoms)

# ========================================================================== #
    @abstractmethod
    def train_mlip(self):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    @abstractmethod
    def predict(self, atoms, coef=None):
        """
        Function that gives the e, f, s
        """
        raise NotImplementedError

# ========================================================================== #
    def read_parent_mlip(self, traj):
        """
        Get a list of all the mlip that have generated a conf in traj
        and get the coefficients of all these mlip
        """
        parent_mlip = []
        mlip_coef = []
        prefix = self.descriptor.prefix

        # Check that this simulation and the previous one use the same mlip
        fn_descriptor = self.subdir / f"{prefix}.descriptor"
        with open(fn_descriptor, "r") as f:
            lines = f.read()

        if not lines == self.descriptor.get_mlip_params():
            err = "The MLIP.descriptor from {fn_descriptor} seems different "
            err += "to the one you have in this simulation. If you want a "
            err += "new mlip: Rerun MLACS with DatabaseCalculator and "
            err += "OtfMlacs.keep_tmp_files=True on your traj"
            raise ValueError(err)

        # Make the MBAR variable Nk and mlip_coef
        for state in traj:
            for conf in state:
                if "parent_mlip" not in conf.info:  # Initial or training
                    continue
                else:  # A traj
                    model = conf.info['parent_mlip']  # GA: Not sure if this is absolute or relative
                    if not Path(model).exists:
                        err = "Some parent MLIP are missing. "
                        err += "Rerun MLACS with DatabaseCalculator and "
                        err += "OtfMlacs.keep_tmp_files=True on your traj"
                        raise FileNotFoundError(err)
                    if model not in parent_mlip:  # New state
                        parent_mlip.append(model)
                        self.descriptor.subfolder = model
                        coef = self.descriptor.read_mlip()
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
        idx_e, idx_f, idx_s = self._get_idx_fit()

        # TODO GA: Passing names like this is a bit shady. Should clean up.
        mlip_fn = self.descriptor.write_mlip(mlip_coef)
        _, weight_fn = self.weight.compute_weight(mlip_coef, self.predict)
        prefix = self.descriptor.prefix
        create_link(self.subsubdir/weight_fn, self.subdir/"MLIP.weight")
        create_link(self.subsubdir/mlip_fn, self.subdir/f"{prefix}.model")

# ========================================================================== #
    @Manager.exec_from_workdir
    def test_mlip(self, testset):
        """
        """
        calc = self.get_calculator()

        ml_e = []
        ml_f = []
        ml_s = []
        dft_e = []
        dft_f = []
        dft_s = []
        for at in testset:
            mlat = at.copy()
            mlat.calc = calc
            e = mlat.get_potential_energy() / len(mlat)
            f = mlat.get_forces().flatten()
            s = mlat.get_stress()

            ml_e.append(e)
            ml_f.extend(f)
            ml_s.extend(s)

            e = at.get_potential_energy() / len(at)
            f = at.get_forces().flatten()
            s = at.get_stress()

            dft_e.append(e)
            dft_f.extend(f)
            dft_s.extend(s)

        dft_e = np.array(dft_e)
        dft_f = np.array(dft_f)
        dft_s = np.array(dft_s)
        ml_e = np.array(ml_e)
        ml_f = np.array(ml_f)
        ml_s = np.array(ml_s)

        rmse_e, mae_e, rsq_e = compute_correlation(np.c_[dft_e, ml_e])
        rmse_f, mae_f, rsq_f = compute_correlation(np.c_[dft_f, ml_f])
        rmse_s, mae_s, rsq_s = compute_correlation(np.c_[dft_s, ml_s] / GPa)

        nat = np.array([len(at) for at in testset]).sum()
        msg = "number of configurations for training: " + \
              f"{len(testset)}\n"
        msg += "number of atomic environments for training: " + \
               f"{nat}\n"

        # Prepare message to the log
        msg += f"RMSE Energy    {rmse_e:.4f} eV/at\n"
        msg += f"MAE Energy     {mae_e:.4f} eV/at\n"
        msg += f"RMSE Forces    {rmse_f:.4f} eV/angs\n"
        msg += f"MAE Forces     {mae_f:.4f} eV/angs\n"
        msg += f"RMSE Stress    {rmse_s:.4f} GPa\n"
        msg += f"MAE Stress     {mae_s:.4f} GPa\n"
        msg += "\n"

        header = f"rmse: {rmse_e:.5f} eV/at,    " + \
                 f"mae: {mae_e:.5f} eV/at\n" + \
                 " True Energy           Predicted Energy"
        np.savetxt("TestSet-Energy_comparison.dat",
                   np.c_[dft_e, ml_e],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_f:.5f} eV/angs   " + \
                 f"mae: {mae_f:.5f} eV/angs\n" + \
                 " True Forces           Predicted Forces"
        np.savetxt("TestSet-Forces_comparison.dat",
                   np.c_[dft_f, ml_f],
                   header=header, fmt="%25.20f  %25.20f")
        header = f"rmse: {rmse_s:.5f} GPa       " + \
                 f"mae: {mae_s:.5f} GPa\n" + \
                 " True Stress           Predicted Stress"
        np.savetxt("TestSet-Stress_comparison.dat",
                   np.c_[dft_s, ml_s] / GPa,
                   header=header, fmt="%25.20f  %25.20f")
        return msg

# ========================================================================== #
    def _get_idx_fit(self):
        """
        """
        idx_e = 0
        idx_f = 3 * self.natoms[:idx_e].sum()
        idx_s = idx_e * 6
        return idx_e, idx_f, idx_s

    @property
    def pair_style(self):
        return self._get_pair_style()

    @property
    def pair_coeff(self):
        return self._get_pair_coeff()

# ========================================================================== #
    def _get_pair_style(self):
        self.descriptor.folder = self.folder
        self.descriptor.subfolder = ''
        return self.descriptor.get_pair_style()

# ========================================================================== #
    def _get_pair_coeff(self):
        self.descriptor.folder = self.folder
        self.descriptor.subfolder = ''
        return self.descriptor.get_pair_coeff()


# ========================================================================== #
# ========================================================================== #
class SelfMlipManager(MlipManager):
    """
    Mlip manager for model that both compute the descriptor and takes
    care of the regression (ex. MTP, POD)
    """
    def __init__(self,
                 descriptor,
                 weight=None,
                 **kwargs):
        MlipManager.__init__(self, descriptor, weight=weight, **kwargs)
        self.configurations = []
        self.natoms = []

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        nat = np.array([len(at) for at in atoms], dtype=int)

        self.configurations.extend(atoms)
        self.natoms = np.append(self.natoms, nat)
        self.natoms = np.array(self.natoms, dtype=int)
        self.nconfs += len(atoms)
