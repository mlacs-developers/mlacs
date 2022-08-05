import numpy as np

from ase.units import kB, fs
from ase.symbols import Symbols

hbar = 6.582119514e-16 * 1e15*fs  # from eV.s to eV.(ASE time units)


# ========================================================================== #
# ========================================================================== #
class QPolymer:
    """
    """
    def __init__(self,
                 atoms,
                 temperature,
                 nbeads,
                 calculator=None):

        self.kBT = kB * temperature
        self.nbeads = nbeads

        if calculator is None:
            calculator = atoms.calc

        self._beads = []
        for ibead in range(self.nbeads):
            self._beads.append(atoms.copy())

        self.hfreq = np.sqrt(self.nbeads) * self.kBT / hbar

        self.natoms = len(self[0])
        self.numbers = atoms.arrays["numbers"]
        self.masses = atoms.get_masses()

        momenta = atoms.get_momenta()
        self.set_momenta(momenta, apply_constraint=False)

        self.calc = calculator

# ========================================================================== #
    def set_temperature(self, temperature):
        """
        """
        self.kBT = temperature * kB
        self.hfreq = np.sqrt(self.nbeads) * self.kBT / hbar

# ========================================================================== #
    def set_momenta(self, newmomenta, apply_constraint=True):
        """
        """
        if len(np.shape(newmomenta)) == 2:
            for ibead in range(self.nbeads):
                self[ibead].set_momenta(newmomenta, apply_constraint)
        elif len(np.shape(newmomenta)) == 3:
            for ibead in range(self.nbeads):
                self[ibead].set_momenta(newmomenta[ibead], apply_constraint)
        else:
            raise ValueError("Positions can only be a natoms X 3 " +
                             "or a nbeads X natoms X 3 array")

# ========================================================================== #
    def get_momenta(self):
        """
        """
        momenta = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            momenta[ibead] = self[ibead].get_momenta()
        return momenta

# ========================================================================== #
    def set_velocities(self, velocities, apply_constraint=True):
        """
        """
        if len(np.shape(velocities)) == 2:
            self.set_momenta(self.get_masses()[None, :, None] * velocities)
        if len(np.shape(velocities)) == 3:
            self.set_momenta(self.get_masses()[:, None] * velocities)

# ========================================================================== #
    def get_velocities(self):
        """
        """
        velocities = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            velocities[ibead] = self[ibead].get_velocities()
        return velocities

# ========================================================================== #
    def get_masses(self):
        """
        """
        return self.masses.copy()

# ========================================================================== #
    def get_positions(self, wrap=False, **wrap_kw):
        """
        """
        positions = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            positions[ibead] = self[ibead].get_positions(wrap, **wrap_kw)
        return positions

# ========================================================================== #
    @property
    def symbols(self):
        """
        """
        return Symbols(self.numbers)

# ========================================================================== #
    @symbols.setter
    def symbols(self, obj):
        new_symbols = Symbols.fromsymbols(obj)
        self.numbers[:] = new_symbols.numbers

# ========================================================================== #
    @property
    def calc(self):
        """
        """
        return self._calc

# ========================================================================== #
    @calc.setter
    def calc(self, calc):
        self._calc = calc
        for ibead in range(self.nbeads):
            self[ibead].calc = calc

# ========================================================================== #
    def set_cell(self, cell, scale_atoms=False, apply_constraint=True):
        """
        """
        for ibead in range(self.nbeads):
            self[ibead].set_cell(cell, scale_atoms, apply_constraint)

# ========================================================================== #
    def get_potential_energy(self,
                             force_consistent=False,
                             apply_constraint=True):
        """
        """
        energy = np.zeros((self.nbeads))
        for ibead in range(self.nbeads):
            energy[ibead] = self[ibead].get_potential_energy(force_consistent,
                                                             apply_constraint)
        return energy

# ========================================================================== #
    def get_forces(self, apply_constraint=True, md=False):
        """
        """
        forces = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            forces[ibead] = self[ibead].get_forces(apply_constraint, md)
        return forces

# ========================================================================== #
    def get_stress(self,
                   voigt=True,
                   apply_constraint=True,
                   include_ideal_gas=False):
        """
        """
        if voigt is False:
            stress = np.zeros((self.nbeads, 3, 3))
            for ibead in range(self.nbeads):
                stress[ibead] = self[ibead].get_stress(voigt,
                                                       apply_constraint,
                                                       include_ideal_gas)
        else:
            stress = np.zeros((self.nbeads, 6))
            for ibead in range(self.nbeads):
                stress[ibead] = self[ibead].get_stress(voigt,
                                                       apply_constraint,
                                                       include_ideal_gas)
        return stress

# ========================================================================== #
    def get_center_of_mass(self, scaled=False):
        """
        """
        com = np.zeros((self.nbeads, 3))
        for ibead in range(self.nbeads):
            com[ibead] = self[ibead].get_center_of_mass(scaled)
        return com

# ========================================================================== #
    def set_center_of_mass(self, com, scaled=False):
        """
        """
        old_com = self.get_center_of_mass(scaled=scaled)
        diff = old_com - com
        if scaled:
            self.set_scaled_positions(self.get_scaled_positions() +
                                      diff[:, None, :])
        else:
            self.set_positions(self.get_positions() + diff[:, None, :])

# ========================================================================== #
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
            self[ibead].rattle(stdev, None, rng)

# ========================================================================== #
    def get_scaled_positions(self, wrap=True):
        """
        """
        scaled_positions = np.zeros((self.nbeads, self.natoms, 3))
        for ibead in range(self.nbeads):
            scaled_positions[ibead] = self[ibead].get_scaled_positions(wrap)
        return scaled_positions

# ========================================================================== #
    def get_volume(self):
        """
        """
        volume = np.zeros((self.nbeads))
        for ibead in range(self.nbeads):
            volume[ibead] = self[ibead].get_volume()
        return volume

# ========================================================================== #
    def set_positions(self, newpositions, apply_constraint=True):
        """
        """
        if len(np.shape(newpositions)) == 2:
            for ibead in range(self.nbeads):
                self[ibead].set_positions(newpositions, apply_constraint)
        elif len(np.shape(newpositions)) == 3:
            for ibead in range(self.nbeads):
                self[ibead].set_positions(newpositions[ibead],
                                          apply_constraint)
        else:
            raise ValueError("Positions can only be a natoms X 3 or " +
                             "a nbeads X natoms X 3 array")

# ========================================================================== #
    def set_scaled_positions(self, newpositions):
        """
        """
        if len(np.shape(newpositions)) == 2:
            for ibead in range(self.nbeads):
                self[ibead].set_scaled_positions(newpositions)
        elif len(np.shape(newpositions)) == 3:
            for ibead in range(self.nbeads):
                self[ibead].set_scaled_positions(newpositions[ibead])
        else:
            raise ValueError("Positions can only be a natoms X 3 or " +
                             "a nbeads X natoms X 3 array")

# ========================================================================== #
    def set_calculator(self, calc):
        """
        """
        for ibead in range(self.nbeads):
            self[ibead].calc = calc

# ========================================================================== #
    def get_spring_potential(self):
        """
        """
        pos_tmp1 = self.get_positions()
        pos_tmp2 = pos_tmp1.copy()
        pos_tmp2 = np.delete(pos_tmp2, 0, 0)
        pos_tmp2 = np.vstack((pos_tmp2, pos_tmp1[None, 0]))

        spring = 0.5 * self.hfreq**2 * (self.get_masses()[None, :, None] *
                                        (pos_tmp1 - pos_tmp2)**2).sum()
        return spring

# ========================================================================== #
    def get_centroid(self):
        """
        """
        positions = self.get_positions(wrap=False)
        return positions.mean(axis=0)

# ========================================================================== #
    def get_potential_estimator(self):
        """
        """
        energy = self.get_potential_energy()
        potential_estimator = np.mean(energy)
        return potential_estimator

# ========================================================================== #
    def get_kinetic_estimator(self, virial=False):
        """
        """
        classic_kin = 1.5 * self.natoms * self.kBT
        if virial:
            kinetic_estimator = classic_kin
            centroid = self.get_centroid()
            pos_tmp = self.get_positions(wrap=False)
            forces = self.get_forces()
            kinetic_estimator -= 0.5 * np.sum((pos_tmp - centroid[None, :, :])
                                              * forces) / self.nbeads
        else:
            print(self.get_spring_potential())
            kinetic_estimator = classic_kin * self.nbeads - \
                self.get_spring_potential()
        return kinetic_estimator

# ========================================================================== #
    def get_staging_positions(self):
        """
        """
        pos = self.get_positions()
        tpos = np.zeros(pos.shape)
        tpos[0] = pos[0]
        for j in range(1, self.nbeads-1):
            tpos[j] = pos[j] - (j * pos[j+1] + pos[0]) / (j+1)
        tpos[self.nbeads-1] = pos[self.nbeads-1] - \
            ((self.nbeads - 1) * pos[0] + pos[0]) / (self.nbeads)
        return tpos

# ========================================================================== #
    def inverse_staging_transform(self, tpos):
        """
        """
        pos = np.zeros(tpos.shape)
        pos[:] = tpos[0]
        for j in range(1, self.nbeads):
            for k in range(j, self.nbeads):
                pos[j] += tpos[k] * j / k
        return pos

# ========================================================================== #
    def set_positions_from_staging_transform(self, tpos):
        """
        """
        pos = self.inverse_transform(tpos)
        self.set_positions(pos)

# ========================================================================== #
    def get_staging_masses(self):
        """
        """
        tmasses = np.zeros((self.nbeads, self.natoms))
        tmasses[0] = self.get_masses()
        for j in range(1, self.nbeads):
            tmasses[j] = (j+1)/j * self.masses
        return tmasses

# ========================================================================== #
    def get_staging_forces(self):
        """
        """
        forces = self.get_forces()
        tforces = np.zeros(forces.shape)

        tforces[0] = forces.mean(axis=0)
        for j in range(1, self.nbeads):
            tforces[j] = (forces[j] + (j-1)/j * tforces[j-1]) / self.nbeads
        return tforces

# ========================================================================== #
    def __len__(self):
        """
        """
        return self.nbeads

# ========================================================================== #
    def __getitem__(self, i):
        """
        """
        return self._beads[i]

# ========================================================================== #
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
