import os
from pathlib import Path
from subprocess import run, PIPE

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.singlepoint import SinglePointCalculator

from .delta_learning import DeltaLearningPotential
from ..utilities import get_elements_Z_and_masses
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
                 model_post=None,
                 folder=Path().absolute() / "Spin"
                 ):

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

        self.folder = folder
        self.folder.mkdir(exist_ok=True, parents=True)

        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp"
        self.cmd = cmd

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
        assert spins.shape[2] == 4

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
        # First we delete old stuff to be sure everything is right
        filenames = ["atoms.in", "logfile.out", "configurations.out",
                     "lmp.out"]
        for fname in filenames:
            if (self.folder / fname).exists():
                (self.folder / fname).unlink()

        with open(self.folder / "atoms.in", "w") as fd:
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
        at = read(self.folder / "configurations.out")
        forces = at.get_forces()
        data = np.loadtxt(self.folder / "logfile.out")
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

        input_string += get_last_dump_input(self.folder, elem, 1,
                                            with_delay=False)

        input_string += "thermo         1\n"
        input_string += "timestep       0.005\n"
        # input_string += "neighbor       1.0 bin\n"
        # input_string += "neigh_modify   once no every 1 delay 0 check yes\n"
        input_string += "run            0\n"

        with open(self.folder / "lammps_input.in", "w") as fd:
            fd.write(input_string)

# ========================================================================== #
    def _run_lammps(self):
        lammps_cmd = self.cmd + ' -in lammps_input.in -log none -sc lmp.out'
        lammps_cmd = lammps_cmd.split()
        lmp_handle = run(lammps_cmd,
                         stderr=PIPE,
                         cwd=self.folder)

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
