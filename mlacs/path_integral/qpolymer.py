import numpy as np

from ase.atoms import Atoms
from ase.units import kB, _hbar
from ase.cell import Cell
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import wrap_positions, find_mic, get_angles, get_distances
from ase.symbols import Symbols, symbols2numbers

hbar = 6.582119570e-16



class qPolymer:
    """
    """
    def __init__(self,
                 atoms,
                 temperature,
                 nbeads=None,
                ):

        self.kBT = temperature * kB

        if nbeads is not None:
            self.nbeads = nbeads
        else:
            self.nbeads = 1

        calc = atoms.calc

        self.beads = []
        for ibead in range(self.nbeads):
            self.beads.append(atoms.copy())
            self.beads[ibead].calc = calc

        self.harm_freq_squared = self.nbeads * self.kBT**2 / hbar**2

        self.natoms = len(self.beads[0])
        self.masses = self.beads[0].get_masses()



    def get_masses(self):
        """
        """
        return self.masses


    def get_positions(self, wrap=False, **wrap_kw):
        """
        """
        positions = np.zeros((self.nbeads, len(self.beads[0]), 3))
        for ibead in range(self.nbeads):
            positions[ibead] = self.beads[ibead].get_positions(wrap, **wrap_kw)
        return positions

    @property
    def symbols(self):
        """
        """
        return Symbols(self.beads[0].numbers)


    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        """
        """
        for ibead in range(self.nbeads):
            self.beads[ibead].set_cell(cell, scale_atoms, apply_constraint)


    def get_potential_energy(self, force_consistent=False, apply_constraint=True):
        """
        """
        energy = np.zeros((self.nbeads))
        for ibead in range(self.nbeads):
            energy[ibead] = self.beads[ibead].get_potential_energy(force_consistent, apply_constraint)
        return energy


    def get_forces(self, apply_constraint=True, md=False):
        """
        """
        forces = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            forces[ibead] = self.beads[ibead].get_forces(apply_constraint, md)
        return forces


    def get_stress(self, voigt=True, apply_constraint=True, include_ideal_gas=False):
        """
        """
        if voigt is False:
            stress = np.zeros((self.nbeads, 3, 3))
            for ibead in range(self.nbeads):
                stress[ibead] = self.beads[ibead].get_stress(voigt, apply_constraint, include_ideal_gas)
        else:
            stress = np.zeros((self.nbeads, 6))
            for ibead in range(self.nbeads):
                stress[ibead] = self.beads[ibead].get_stress(voigt, apply_constraint, include_ideal_gas)
        return stress


    def get_center_of_mass(self, scaled=False):
        """
        """
        com = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            com = self.beads[ibead].get_center_of_mass(scaled)
        return com


    def rattle(self, stdev=0.001, seed=None, rng=None):
        """
        """
        if seed is not None and rng is not None:
            raise ValueError('Please do not provide both seed and rng.')

        if rng is None:
            if seed is not None:
                rng = np.random.RandomState(seed)
            else:
                rng = np.random.default_rng()

        for ibead in range(self.nbeads):
            self.beads[ibead].rattle(stdev, None, rng)


    def get_scaled_positions(self, wrap):
        """
        """
        scaled_positions = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            scaled_positions[ibead] = self.beads[ibead].get_scaled_positions(wrap)
        return scaled_positions


    def get_volume(self):
        """
        """
        volume = np.zeros((self.nbeads))
        for ibead in range(self.nbeads):
            volume[ibead] = self.beads[ibead].get_volume()
        return volume


    def set_positions(self, newpositions, apply_constraint=True):
        """
        """
        if len(np.shape(newpositions)) == 2:
            for ibead in range(self.nbeads):
                self.beads[ibead].set_positions(newpositions, apply_constraint)
        elif len(np.shape(newpositions)) == 3:
            for ibead in range(self.nbeads):
                self.beads[ibead].set_positions(newpositions[ibead], apply_constraint)
        else:
            raise ValueError("Positions can only be a natoms X 3 or a nbeads X natoms X 3 array")


    def set_scaled_positions(self, newpositions):
        """
        """
        if len(np.shape(newpositions)) == 2:
            for ibead in range(self.nbeads):
                self.beads[ibead].set_scaled_positions(newpositions, apply_constraint)
        elif len(np.shape(newpositions)) == 3:
            for ibead in range(self.nbeads):
                self.beads[ibead].set_scaled_positions(newpositions[ibead], apply_constraint)
        else:
            raise ValueError("Positions can only be a natoms X 3 or a nbeads X natoms X 3 array")


    def set_calculator(self, calc):
        """
        """
        for ibead in range(self.nbeads):
            self.beads[ibead].calc = calc


    def get_spring_potential(self):
        """
        """
        pos_tmp1 = self.get_positions()
        pos_tmp2 = pos_tmp1.copy()
        pos_tmp2 = np.delete(pos_tmp2, 0, 0)
        pos_tmp2 = np.vstack((pos_tmp2, pos_tmp1[None,0]))

        spring   = 0.5 * (self.masses[None, :, None] * self.harm_freq_squared * (pos_tmp1 - pos_tmp2)**2).sum()
        return spring


    def get_centroid(self):
        """
        """
        positions = self.get_positions(wrap=False)
        return positions.mean(axis=0)


    def get_potential_estimator(self):
        """
        """
        energy = self.get_potential_energy()
        potential_estimator = np.mean(energy)
        return potential_estimator


    def get_kinetic_estimator(self, virial=False):
        """
        """
        classic_kin = 1.5 * self.natoms * self.kBT
        if virial:
            kinetic_estimator = classic_kin
            centroid = self.get_centroid()
            pos_tmp  = self.get_positions(wrap=False)
            forces   = self.get_forces()
            kinetic_estimator -= 0.5 * np.sum((pos_tmp - centroid[None,:,:]) * forces) / self.nbeads

        else:
            kinetic_estimator = classic_kin * self.nbeads - self.get_spring_potential()
        return kinetic_estimator
