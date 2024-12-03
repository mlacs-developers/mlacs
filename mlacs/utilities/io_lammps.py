"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, PR)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from pathlib import Path

import numpy as np
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lammps import Prism, convert

from .path_integral import hbar


class LammpsInput:
    """

    """
    def __init__(self, preambule=None):
        if preambule is not None:
            self.preambule = f"# {preambule}\n\n"
        else:
            self.preambule = ""
        self.nvar = 0
        self.vardict = dict()

    def add_block(self, name, block, order=-1, before=None, after=None):
        """

        """
        if before is not None and after is not None:
            msg = "before and after can't be both set"
            raise ValueError(msg)

        if before is not None:
            order = self.vardict[before]["order"]
        elif after is not None:
            order = self.vardict[after]["order"] + 1

        if order < 0:
            order = self.nvar + 1
        else:
            keys = []
            values = []
            for key, val in self.vardict.items():
                keys.append(key)
                values.append(val["order"])
            keys = np.array(keys)
            values = np.array(values)
            argsort = np.argsort(values)
            values = values[argsort]
            keys = keys[argsort]
            if order > np.max(values) or order not in values:
                self.vardict["name"]["order"] = order
            elif order in values:
                values[values >= order] += 1
                for i, (key, val) in enumerate(zip(keys, values)):
                    self.vardict[key]["order"] = values[i]
        self.vardict[name] = dict(order=order, block=block)
        self.nvar += 1

    def to_string(self):
        """

        """
        keys = []
        orders = []
        blocks = []
        for key, val in self.vardict.items():
            keys.append(key)
            orders.append(val["order"])
            blocks.append(val["block"])

        keys = np.array(keys)
        orders = np.array(orders)
        blocks = np.array(blocks)

        argsort = np.argsort(orders)
        blocks = blocks[argsort]

        txt = self.preambule
        txt += "\n\n".join(str(block) for block in blocks)
        return txt

    def pop(self, name):
        return self.vardict.pop(name)

    def __str__(self):
        return self.to_string()

    def __call__(self, name, block, order=-1):
        self.add_block(name, block, order)


class LammpsBlockInput:
    """

    """
    def __init__(self, name, title=None):
        self.name = name
        self.vardict = dict()
        self.nvar = 0
        if title is not None:
            title = title.strip()
            nchar = len(title)
            self.title = "#" * (12 + nchar) + "\n"
            self.title += title.center(nchar + 10, " ").center(nchar + 12, "#")
            self.title += "\n"
            self.title += "#" * (12 + nchar) + "\n"
        else:
            self.title = "\n"

    def add_variable(self, name, line, order=-1, before=None, after=None):
        """

        """
        if before is not None and after is not None:
            msg = "before and after can't be both set"
            raise ValueError(msg)

        if before is not None:
            order = self.vardict[before]["order"]
        elif after is not None:
            order = self.vardict[after]["order"] + 1

        if order < 0:
            order = self.nvar + 1
        else:
            keys = []
            values = []
            for key, val in self.vardict.items():
                keys.append(key)
                values.append(val["order"])
            keys = np.array(keys)
            values = np.array(values)
            argsort = np.argsort(values)
            values = values[argsort]
            keys = keys[argsort]
            if order in values:
                values[values >= order] += 1
                for i, (key, val) in enumerate(zip(keys, values)):
                    self.vardict[key]["order"] = values[i]
        self.vardict[name] = dict(order=order, line=line)
        self.nvar += 1

    def to_string(self):
        """

        """
        keys = []
        orders = []
        lines = []
        for key, val in self.vardict.items():
            keys.append(key)
            orders.append(val["order"])
            lines.append(val["line"])

        keys = np.array(keys)
        orders = np.array(orders)
        lines = np.array(lines)

        argsort = np.argsort(orders)
        line = lines[argsort]

        txt = self.title
        txt += "\n".join(line)
        return txt

    def pop(self, name):
        '''Remove block line'''
        return self.vardict.pop(name)

    def extend(self, block):
        '''Concatenate Blocks'''
        for key, val in block.vardict.items():
            self.__call__(key, val['line'])

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return f"LammbsBlockInput({self.name})"

    def __call__(self, name, line, order=-1):
        self.add_variable(name, line, order)


class EmptyLammpsBlockInput(LammpsBlockInput):
    """

    """
    def __init__(self, name):
        self.name = name

    def to_string(self):
        return ""


# ========================================================================== #
def get_block_rdf(nsteps, filename='spce-rdf.dat', rmax=None):
    """
    Function to compute and output the radial distribution function
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("RDF", "Compute RDF")
    block("v_rep_rdf", f"variable rep_rdf equal {nsteps}/2")
    txt = "compute rdf all rdf ${rep_rdf} 1 1"
    if rmax is not None:
        txt += ' cutoff {rmax}'
    block("c_rdf", txt)
    txt = "fix rdf all ave/time 1 ${rep_rdf}" + \
          f" {nsteps} c_rdf[*] file {filename} mode vector\n"
    block("rdf", txt)
    return block


# ========================================================================== #
def get_block_adf(nsteps, filename='spce-adf.dat'):
    """
    Function to compute and output the angle distribution function
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("ADF", "Compute ADF")
    block("v_rep_adf", "variable rep_adf equal 1")
    block("c_adf", "compute adf all adf 360")
    txt = "fix adf all ave/time 100 ${rep_adf}" + \
          f" {nsteps} c_adf[*] file {filename} mode vector\n"
    block("adf", txt)
    return block


# RB: This could be done in a better way.
# ========================================================================== #
def get_block_diffusion(nsteps, filename='diffusion.dat'):
    """
    Function to compute and output the diffusion coefficient
    """
    # freq = int(nsteps/5)
    block = LammpsBlockInput("MSD", "Compute MSD and diffusion coefficient")
    block("v_t", "variable t equal step")
    block("c_msd", "compute msd all msd")
    block("v_msd", "variable msd equal c_msd[4]")
    block("v_twopts", "variable twopoint equal c_msd[4]/6/(step*dt+1.0e-6)")
    block("f_msd", "fix msd all vector 1000 c_msd[4]")
    block("v_slope", "variable fitslope equal slope(f_msd)/6/(10000*dt)")
    txt = 'fix dcoeff all print 100 "${t} ${msd} ${twopoint} ${fitslope}"' + \
          f' append {filename} title "# Step MSD D(start) D(slope)"'
    block("diffusion", txt)
    return block


# ========================================================================== #
def write_atoms_lammps_spin_style(fd, atoms, spin, velocities=True):
    """
    Function to write atoms in the LAMMPS spin style
    Loosely adapted from ASE write_lammpsdata function
    """
    fd.write("# Atoms in spin style, Written by MLACS\n\n")

    nat = len(atoms)
    fd.write(f"{nat} atoms\n")

    symbols = atoms.get_chemical_symbols()
    species = sorted(set(symbols))
    n_atom_type = len(species)
    fd.write(f"{n_atom_type} atom types\n\n")

    prismobj = Prism(atoms.get_cell())
    xhi, yhi, zhi, xy, xz, yz = convert(prismobj.get_lammps_prism(),
                                        'distance',
                                        'ASE',
                                        'metal')

    fd.write(f'0.0 {xhi:23.17g} xlo xhi\n')
    fd.write(f'0.0 {yhi:23.17g} ylo yhi\n')
    fd.write(f'0.0 {zhi:23.17g} zlo zhi\n')
    fd.write("\n\n")

    fd.write("Atoms # spin\n\n")

    pos = prismobj.vector_to_lammps(atoms.get_positions(), wrap=False)
    for i, r in enumerate(pos):
        r = convert(r, "distance", "ASE", "metal")
        s = species.index(symbols[i]) + 1
        line = f"{i+1:>6} {s:>3} "  # Index and species
        line += f"{r[0]:23.17f} {r[1]:23.17f} {r[2]:23.17f} "  # Positions
        norm = np.linalg.norm(spin[i])
        if np.isclose(norm, 0, 1e-5):
            norm = 0.0
            sp = np.zeros(3)
        else:
            sp = spin[i] / norm
        line += f"{sp[0]:23.17f} {sp[1]:23.17f} {sp[2]:23.17f} "
        line += f"{norm} "
        line += "\n"
        fd.write(line)

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities \n\n")
        vel = prismobj.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            v = convert(v, "velocity", "ASE", "metal")
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    fd.flush()


# ========================================================================== #
def reconstruct_mlmd_trajectory(trajfile, logfile):
    """
    Function to reconstruct a trajectory from LAMMPS
    Note that this function is made specifically for
    trajectory launched with MLACS, and won't work
    for general LAMMPS trajectories.

    Parameters
    ----------
    trajfile: Path or str
        The file for the trajectory
    logfile: Path or str
        The file for the log
    return
    ------
        Traj: list of ase.Atoms object
    """
    trajfile = Path(trajfile)
    logfile = Path(logfile)

    isspin = False

    # First we get the index of the quantities we want
    with open(logfile, "r") as fd:
        line = np.array(fd.readline().split())
        assert line[0] == "#"

        idx_epot = np.where(line == "Epot")[0][0] - 1

        if "Magn_x" in line:
            isspin = True
            idx_emagn = np.where(line == "Espin")

    trajconf = read(trajfile, ":")
    log = np.loadtxt(logfile)[:-1]  # the -1 is due to the last dump
    nconf = len(trajconf)
    assert nconf == log.shape[0]

    traj = []
    for at, logat in zip(trajconf, log):
        newat = at.copy()
        epot = logat[idx_epot]
        if isspin:
            espin = logat[idx_emagn]

            sp_n = newat.arrays.pop("c_spin[1]")
            sp_x = newat.arrays.pop("c_spin[2]")
            sp_y = newat.arrays.pop("c_spin[3]")
            sp_z = newat.arrays.pop("c_spin[4]")
            sp = np.c_[sp_x, sp_y, sp_z] * sp_n

            fsp_x = newat.arrays.pop("c_spin[5]")
            fsp_y = newat.arrays.pop("c_spin[6]")
            fsp_z = newat.arrays.pop("c_spin[7]")
            fsp = np.c_[fsp_x, fsp_y, fsp_z] * hbar / (2 * np.pi)

            epot = epot + espin

            newat.set_array("spins", sp)
            newat.set_array("spin_forces", fsp)

        try:
            f_x = newat.arrays.pop("f_ff[1]")
            f_y = newat.arrays.pop("f_ff[2]")
            f_z = newat.arrays.pop("f_ff[3]")
            forces = np.c_[f_x, f_y, f_z]
        except KeyError:
            forces = at.get_forces()

        calc = SinglePointCalculator(newat, energy=epot, forces=forces,
                                     stress=np.zeros(6))
        newat.calc = calc
        traj.append(newat)
    return traj

# RB: msdfile variable probably not needed.
# ========================================================================== #
def get_msd_input(self, msdfile):
    """
    Function to compute msd for neti in solid
    """
    block = LammpsBlockInput("msd", "Compute MSD")
    block("eq", f"run {self.nsteps_eq}")
    for iel, el in enumerate(self.elem):
        block("compute", f"compute c{10+iel} {el} msd com yes")
        block("variable", f"variable msd{el} equal c_c{10+iel}[4]")
        block("msd el", f"fix f{iel+3} {el} print 1 " +
              f"\"${{msd{el}}}\" screen no append msd{el}.dat")
    return block


# ========================================================================== #
def get_lammps_command():
    '''
    Function to load the bash command to run LAMMPS
    '''
    # Since some ASE update, there is a new way to get commands
    envvar = "ASE_LAMMPSRUN_COMMAND"
    try:
        from ase.config import cfg
        if "lammps" in cfg.parser:
            section = cfg.parser["lammps"]
            cmd = section["command"]
        else:
            cmd = cfg.get(envvar)
    except ModuleNotFoundError:
        # The goal is to have this deprecated in the long run
        # when the update of file-io calculators in ASE is completely done
        import os
        cmd = os.environ.get(envvar)

    # And we try the default one afterward
    if cmd is None:
        cmd = "lammps"

    return cmd
