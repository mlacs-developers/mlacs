"""
// Copyright (C) 2022-2024 MLACS group (AC)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

from ase.atoms import Atoms


def write_cfg(filename, atoms, chemmap):
    if isinstance(atoms, Atoms):
        atoms = [atoms]

    with open(filename, "w") as fd:
        for at in atoms:
            nat = len(at)
            pos = at.get_positions()
            cell = at.get_cell()
            chemsymb = np.array(at.get_chemical_symbols())
            forces = at.get_forces()
            energy = at.get_potential_energy()
            stress = at.get_stress() * at.get_volume()
            idx = np.arange(1, nat+1)
            types = np.zeros((nat), dtype=int)
            for i, el in enumerate(chemmap):
                elidx = np.nonzero(chemsymb == el)[0]
                types[elidx] = i

            fd.write("BEGIN_CFG\n")
            fd.write("Size\n")
            fd.write(f"{nat}\n")
            fd.write("Supercell\n")
            for a, b, c in cell:
                fd.write(f"{a:25.20f}  {b:25.20f}  {c:25.20f}\n")
            fd.write("AtomData: id type cartes_x cartes_y cartes_z ")
            fd.write("fx fy fz\n")
            for i, xpos, f in zip(idx, pos, forces):
                fd.write(f"{i} {types[i-1]} {xpos[0]:25.20f} ")
                fd.write(f"{xpos[1]:25.20f} {xpos[2]:25.20f} ")
                fd.write(f"{f[0]:25.20f} {f[1]:25.20f} {f[2]:25.20f}\n")
            fd.write("Energy\n")
            fd.write(f"{energy:25.20f}\n")
            fd.write("PlusStress: xx yy zz yz xz xy\n")
            fd.write(f"{-stress[0]:25.20f} {-stress[1]:25.20f} ")
            fd.write(f"{-stress[2]:25.20f} {-stress[3]:25.20f} ")
            fd.write(f"{-stress[4]:25.20f} {-stress[5]:25.20f}\n")
            fd.write("END_CFG\n")
            fd.write("\n")


def read_cfg_data(filename):
    energy = []
    forces = []
    stress = []
    with open(filename, "r") as fd:
        for line in fd:
            if line.startswith("BEGIN_CFG"):
                fd.readline()  # Size
                nat = int(fd.readline())
                fd.readline()  # Supercell
                cell = np.zeros((3, 3))
                for i in range(3):
                    line = fd.readline()
                    cell[i] = [float(a) for a in line.split()]
                # We need a dummy Atoms to get the volume for the stress
                atoms = Atoms(["Si"], np.zeros((1, 3)), cell=cell)
                fd.readline()  # AtomData
                pos = np.empty((nat, 3))
                fconf = np.empty((nat, 3))
                for i in range(nat):
                    line = fd.readline().split()
                    pos[i] = [float(a) for a in line[2:5]]
                    fconf[i] = [float(a) for a in line[5:8]]
                fd.readline()  # Energy
                econf = float(fd.readline()) / nat
                fd.readline()  # PlusStress
                line = fd.readline().split()
                sconf = np.array([-float(a) for a in line])
                sconf /= atoms.get_volume()

                energy.append(econf)
                forces.extend(fconf.flatten())
                stress.extend(sconf)
    energy = np.array(energy)
    forces = np.array(forces)
    stress = np.array(stress)
    return energy, forces, stress
