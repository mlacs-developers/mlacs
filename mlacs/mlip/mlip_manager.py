"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammpsrun import LAMMPS
from ase.units import GPa


# ========================================================================== #
# ========================================================================== #
class MlipManager:
    """
    Parent Class for the management of Machine-Learning Interatomic Potential
    """
    def __init__(self,
                 descriptor,
                 nthrow=0,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=1.0,
                 mbar=None,
                 no_zstress=False):

        self.descriptor = descriptor
        self.mbar = mbar

        self.ecoef = energy_coefficient
        self.fcoef = forces_coefficient
        self.scoef = stress_coefficient

        self.amat_e = None
        self.amat_f = None
        self.amat_s = None

        self.ymat_e = None
        self.ymat_f = None
        self.ymat_s = None

        self.no_zstress = no_zstress

        self.nthrow = nthrow
        if self.mbar is not None:
            self.nthrow = 0
        self.nconfs = 0

        # Some initialization for sampling interface
        self.pair_style = None
        self.pair_coeff = None
        self.model_post = None
        self.atom_style = "atomic"

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if self.mbar is not None:
            self.mbar.update_database(atoms)
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
    def train_mlip(self):
        """
        """
        raise NotImplementedError

# ========================================================================== #
    def test_mlip(self, testset):
        """
        """
        calc = LAMMPS(pair_style=self.pair_style,
                      pair_coeff=self.pair_coeff)
        if self.model_post is not None:
            calc.set(model_post=self.model_post)

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

        rmse_e = np.sqrt(np.mean((dft_e - ml_e)**2))
        mae_e = np.mean(np.abs(dft_e - ml_e))

        rmse_f = np.sqrt(np.mean((dft_f - ml_f)**2))
        mae_f = np.mean(np.abs(dft_f - ml_f))

        rmse_s = np.sqrt(np.mean((((dft_s - ml_s) / GPa)**2)))
        mae_s = np.mean(np.abs((dft_s - ml_s) / GPa))

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
        if self.nconfs < self.nthrow:
            idx_e = idx_f = idx_s = 0
        elif self.nconfs >= self.nthrow and self.nconfs < 2 * self.nthrow:
            idx_e = self.nconfs - self.nthrow
        else:
            idx_e = self.nthrow
        idx_f = 3 * self.natoms[:idx_e].sum()
        idx_s = idx_e * 6
        return idx_e, idx_f, idx_s


# ========================================================================== #
# ========================================================================== #
class SelfMlipManager(MlipManager):
    """
    Mlip manager for model that both compute the descriptor and takes
    care of the regression (ex. MTP, POD)
    """
    def __init__(self,
                 descriptor,
                 nthrow=10,
                 energy_coefficient=1.0,
                 forces_coefficient=1.0,
                 stress_coefficient=0.0):
        MlipManager.__init__(self, descriptor, nthrow,
                             energy_coefficient, forces_coefficient,
                             stress_coefficient)
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
