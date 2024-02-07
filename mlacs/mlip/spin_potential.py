import os
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import GPa

from .delta_learning import DeltaLearningPotential
from ..utilities import (get_elements_Z_and_masses,
                         compute_correlation,
                         subfolder)
from ..utilities.io_lammps import (write_atoms_lammps_spin_style,
                                   get_interaction_input,
                                   get_last_dump_input,
                                   get_log_input)


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
                 pair_style,
                 pair_coeff,
                 model_post=None):

        # We need to add a zero pair_style for cases with non-magnetic atoms
        if isinstance(pair_style, str):
            pair_style = [pair_style,
                          "zero 0.0"]
            pair_coeff = [pair_coeff,
                          ["* *"]]
        else:
            pair_style.append("zero 0.0")
            pair_coeff.append(["* *"])

        DeltaLearningPotential.__init__(self, model, pair_style,
                                        pair_coeff, model_post)

        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

# ========================================================================== #
    def update_matrices(self, atoms):
        """
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        # Compute reference energy/forces/stress
        dummy_at = []
        for at in atoms:
            at0 = at.copy()
            spin = at.get_array("spins")
            assert spin.shape[0] == len(at0)
            assert spin.shape[1] == 3

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
                      keep_alive=False)
        return calc

# ========================================================================== #
    def test_mlip(self, testset, mlip_subfolder=""):
        """
        """
        # TODO this is way too copypasta from mlip_manager.py
        # Needs to be cut-down a bit over there to fix this.
        calc = LAMMPS(pair_style=self.model.pair_style,
                      pair_coeff=self.model.pair_coeff)

        ml_e = []
        ml_f = []
        ml_s = []
        dft_e = []
        dft_f = []
        dft_s = []
        for at in testset:
            # First the Spin part
            at0sp = at.copy()
            spin = at.get_array("spins")

            at0sp = self._compute_spin_properties(at0sp,
                                                  spin,
                                                  subfolder=mlip_subfolder)
            spe = at0sp.get_potential_energy()
            spf = at0sp.get_forces()
            sps = at0sp.get_stress()

            at0ml = at.copy()
            at0ml.calc = calc
            mle = at0ml.get_potential_energy()
            mlf = at0ml.get_forces()
            mls = at0ml.get_stress()

            ml_e.append((mle + spe) / len(at))
            ml_f.extend((mlf + spf).flatten())
            ml_s.extend(mls + sps)

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
    @subfolder
    def _compute_spin_properties(self, atoms, spin):
        # First we delete old stuff to be sure everything is right
        filenames = ["atoms.in", "logfile.out", "configurations.out",
                     "lmp.out"]
        for fname in filenames:
            if Path(fname).exists():
                Path(fname).unlink()

        with open("atoms.in", "w") as fd:
            write_atoms_lammps_spin_style(fd, atoms, spin)
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        self._write_lammps_input(elem, masses, atoms.get_pbc())
        self._run_lammps()
        at, energy, forces, stress = self._read_out_log()
        calc = SinglePointCalculator(at, energy=energy, forces=forces,
                                     stress=stress)
        at.calc = calc
        return at

# ========================================================================== #
    def _read_out_log(self):
        at = read("configurations.out")
        forces = at.get_forces()
        data = np.loadtxt("logfile.out")
        energy = data[3]
        xx, yy, zz, xy, xz, yz = data[-6:]
        stress = [xx, yy, xy, yz, xz, xy]
        return at, energy, forces, stress

# ========================================================================== #
    def _write_lammps_input(self, elem, masses, pbc):
        input_string = "# LAMMPS input file for compute spin potential\n"
        input_string += "clear\n"
        input_string += "boundary "
        for ppp in pbc:
            if ppp:
                input_string += "p "
            else:
                input_string += "f "
        input_string += "\n"
        input_string += "atom_modify map array\n"
        input_string += "atom_style       spin\n"
        input_string += "units            metal\n"
        input_string += "read_data        atoms.in\n"
        for n1 in range(len(masses)):
            input_string += f"mass             {n1+1} {masses[n1]}\n"

        input_string += get_interaction_input(self.ref_pair_style,
                                              self.ref_pair_coeff,
                                              self.model_post)

        input_string += "fix fix_nve       all nve/spin lattice moving\n"

        input_string += get_log_input(1, "logfile.out")

        input_string += get_last_dump_input(elem, 1,
                                            with_delay=False)

        input_string += "thermo         1\n"
        input_string += "timestep       0.005\n"
        # input_string += "neighbor       1.0 bin\n"
        # input_string += "neigh_modify   once no every 1 delay 0 check yes\n"
        input_string += "run            0\n"

        with open("lammps_input.in", "w") as fd:
            fd.write(input_string)

# ========================================================================== #
    def _run_lammps(self):
        lammps_cmd = self.cmd + ' -in lammps_input.in -log none -sc lmp.out'
        lammps_cmd = lammps_cmd.split()
        lmp_handle = run(lammps_cmd,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def __str__(self):
        txt = " ".join(self.elements)
        txt += "Spin-Lattice potential,"
        txt += str(self.model)

# ========================================================================== #
    def __repr__(self):
        txt = "Spin Lattice potential\n"
        txt += "------------------------\n"
        txt += "Spin potential :\n"
        txt += f"pair_style {self.ref_pair_style}\n"
        for pc in self.ref_pair_coeff:
            txt += f"pair_coeff {pc}\n\n"
        txt += "MLIP potential :\n"
        txt += repr(self.model)
        return txt
