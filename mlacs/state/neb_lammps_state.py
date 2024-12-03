"""
// Copyright (C) 2022-2024 MLACS group (AC, RB)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

import numpy as np

from ase.atoms import Atoms
from ase.io import write
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from .lammps_state import BaseLammpsState
from ..core import PathAtoms
from ..core.manager import Manager
from ..utilities.io_lammps import (LammpsBlockInput,
                                   EmptyLammpsBlockInput,
                                   get_lammps_command)


# ========================================================================== #
# ========================================================================== #
class NebLammpsState(BaseLammpsState):
    """
    Class to manage Nudged Elastic Band (NEB) calculation with LAMMPS.
    This class is a part of TransPath objects, meaning that it produces
    positions interpolation according to a reaction coordinate.

    Parameters
    ----------
    images: :class:`list` or `PathAtoms`
        mlacs.PathAtoms or list of ase.Atoms object.
        The list contain initial and final configurations of the reaction path.

    xi: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        Default ``None``

    min_style: :class:`str`
        Choose a minimization algorithm to use when a minimize command is
        performed. Default `quickmin`.

    Kspring: :class:`float`
        Spring constante for the NEB calculation.
        Default ``1.0``

    etol: :class:`float`
        Stopping tolerance for energy
        Default ``0.0``

    ftol: :class:`float`
        Stopping tolerance for energy
        Default ``1.0e-3``

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    nimages : :class:`int` (optional)
        Number of images used along the reaction coordinate. Default ``1``.
        which is suposed the saddle point.

    nprocs : :class:`int` (optional)
        Total number of process used to run LAMMPS.
        Have to be a multiple of the number of images.
        If nprocs > than nimages, each image will be parallelized using the
        partition scheme of LAMMPS.
        Per default it assumes that nprocs = nimages

    mode: :class:`float` or :class:`string`
        Value of the reaction coordinate or sampling mode:
        - ``float`` sampling at a precise coordinate.
        - ``rdm_true`` randomly return the coordinate of an images.
        - ``rdm_spl`` randomly return the coordinate of a splined images.
        - ``rdm_memory`` homogeneously sample the splined reaction coordinate.
        - ``None`` return the saddle point.
        Default ``saddle``

    linear : :class:`Bool` (optional)
        If true, the reaction coordinate is a linear interpolation.
        Default ``False``

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.

    prt : :class:`Bool` (optional)
        Printing options. Default ``True``

    Examples
    --------

    >>> from ase.io import read
    >>> initial = read('A.traj')
    >>> final = read('B.traj')
    >>>
    >>> from mlacs.state import NebLammpsState
    >>> neb = NebLammpsState([initial, final])
    >>> state.run_dynamics(None, mlip.pair_style, mlip.pair_coeff)
    """

    def __init__(self, images, xi=None,
                 min_style="quickmin", Kspring=1.0, etol=0.0, ftol=1.0e-3,
                 dt=1.5, nimages=4, nprocs=None, mode=None, interval=None,
                 linear=False, print=False,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, blocks=None, **kwargs):

        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         blocks, **kwargs)

        self.dt = dt
        self.pressure = None

        self._step = 1
        self.style = min_style
        self.criterions = (etol, ftol)
        self.nprocs = nprocs
        self.nreplica = nimages
        self.atomsfname = "atoms-0.data"
        self.print = print
        self.Kspring = Kspring
        self.patoms = images
        if not isinstance(self.patoms, PathAtoms):
            self.patoms = PathAtoms(self.patoms, interval=interval)
        if xi is not None:
            self.patoms.xi = xi
        if mode is not None:
            self.patoms.mode = mode

        self.linear = linear

# ========================================================================== #
    @Manager.exec_from_path
    def _write_lammps_atoms(self, atoms, atom_style, elements=None):
        """

        """
        write_lammps_data(self.atomsfname,
                          self.patoms.initial,
                          velocities=False,
                          atom_style=atom_style)
        instr = '# Final coordinates of the NEB calculation.\n'
        instr += '{0}\n'.format(len(self.patoms.final))
        for atoms in self.patoms.final:
            instr += '{} {} {} {}\n'.format(atoms.index+1, *atoms.position)
        with open("atoms-1.data", "w") as w:
            w.write(instr)

# ========================================================================== #
    def _get_block_init(self, atom_style, pbc, el, masses):
        """

        """
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))

        block = LammpsBlockInput("init", "Initialization")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", f"atom_style {atom_style}")
        block("atom_modify", "atom_modify  map array sort 0 0.0")
        txt = "neigh_modify every 2 delay 10" + \
              " check yes page 1000000 one 100000"
        block("neigh_modify", txt)
        block("read_data", "read_data atoms-0.data")
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
        return block

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        return EmptyLammpsBlockInput("empty_thermostat")

# ========================================================================== #
    def _get_block_lastdump(self, el, eq):
        return EmptyLammpsBlockInput("empty_lastdump")

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """

        """
        self.patoms.images = self.extract_NEB_configurations()
        self.patoms.update
        atoms = self.patoms.splined
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        # RB: Usefull for tests, we should move these in the Mlminimizer.
        i = self._step
        if self.print:
            if self.patoms._gpi is not None:
                if 4 < len(self.patoms._gpi.y):
                    import numpy as np
                    x = np.linspace(0, 1, 1001)
                    m, s = self.patoms._gpi.predict(x, True)
                    err = self.patoms._gp_error
                    np.savetxt(str(self.subsubdir / f'mean_{i:02d}.dat'), m)
                    np.savetxt(str(self.subsubdir / f'err_{i:02d}.dat'), err)
                    np.savetxt(str(self.subsubdir / f'max_{i:02d}.dat'),
                               np.max(s, axis=0))
                    np.savetxt(str(self.subsubdir / f'min_{i:02d}.dat'),
                               np.min(s, axis=0))
            write(str(self.subsubdir / f'pos_neb_images_{i:02d}.xyz'),
                  self.patoms.images, format='extxyz')
            write(str(self.subsubdir / f'pos_neb_splined_{i:02d}.xyz'),
                  self.patoms.splined, format='extxyz')
        self._step += 1
        return atoms

# ========================================================================== #
    def _get_block_run(self, eq):
        etol, ftol = self.criterions

        block = LammpsBlockInput("transpath", "Transition Path")
        block("thermo", "thermo 1")
        block("timestep", f"timestep {self.dt / 1000}")
        block("fix_neb", f"fix neb all neb {self.Kspring} parallel ideal")
        block("run", "run 100")
        block("reset", "reset_timestep 0")
        block("image", "variable i equal part")
        block("min_style", f"min_style {self.style}")
        if self.linear:
            block("neb", f"neb {etol} {ftol} 1 1 1 final atoms-1.data")
        else:
            block("neb", f"neb {etol} {ftol} 200 100 1 final atoms-1.data")
        block("write_data", "write_data neb.$i")
        return block

# ========================================================================== #
    def extract_NEB_configurations(self):
        """
        Extract the positions and energies of a NEB calculation for all
        replicas.
        """
        img_at = []

        def set_atoms(C, R):
            Z = self.patoms.initial.get_atomic_numbers()
            at = Atoms(numbers=Z, positions=R, cell=C)
            at.set_pbc(True)
            return at

        for rep in range(int(self.nreplica)):
            nebfile = str(self.path / f'neb.{rep}')
            positions, cell = self._read_lammpsdata(nebfile)
            check = False
            with open(self.path / f'log.lammps.{rep}') as r:
                for _ in r:
                    if check:
                        etotal = _.split()[2]
                        break
                    if 'initial, next-to-last, final =' in _:
                        check = True
            atoms = set_atoms(cell, positions)
            calc = SPC(atoms=atoms, energy=etotal)
            atoms.calc = calc
            img_at.append(atoms)
        return img_at

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS with replica.
        '''
        cmd = get_lammps_command()
        exe = cmd.split()[-1]

        if "-partition" in cmd:
            _ = cmd.split().index('-n')+1
            self.nprocs = int(cmd.split()[_])
            self.nreplica = int(cmd.split('x')[0][-1])
            return f"{cmd} -in {self.lammpsfname} -sc out.lmp"

        if self.nreplica is not None and self.nprocs is not None:
            pass
        elif self.nreplica is not None and self.nprocs is None:
            if '-n' in cmd:
                _ = cmd.split().index('-n')+1
                self.nprocs = int(cmd.split()[_])
            else:
                self.nprocs = self.nreplica
        elif self.nreplica is None and self.nprocs is not None:
            self.nreplica = self.nprocs
        else:
            if '-n' in cmd:
                _ = cmd.split().index('-n')+1
                self.nprocs = int(cmd.split()[_])
                self.nreplica = self.nprocs
            else:
                self.nreplica, self.nprocs = 1, 1

        n1, n2 = self.nreplica, self.nprocs // self.nreplica
        if n2 == 0:
            n2 = 1
        cmd = f"mpirun -n {int(n1*n2)} {exe} -partition {n1}x{n2}"
        return f"{cmd} -in {self.lammpsfname} -sc out.lmp"

# ========================================================================== #
    def _read_lammpsdata(self, filename, wrap=True):
        """
        Extract positions from lammpsdata files with memory of periodicity.
        Inspired from ASE.
        """
        (xy, xz, yz) = None, None, None
        (section, style) = None, None
        pos_in = {}
        travel_in = {}

        with open(filename, 'r') as r:
            for _ in r:
                if 'atoms' in _:
                    N = int(_.split()[0])
                if 'Atoms' in _:
                    (section, _, style) = _.split()
                    continue
                if 'Velocities' in _:
                    (section) = _.split()
                    continue
                if 'xlo xhi' in _:
                    (xlo, xhi) = [float(x) for x in _.split()[0:2]]
                if 'ylo yhi' in _:
                    (ylo, yhi) = [float(x) for x in _.split()[0:2]]
                if 'zlo zhi' in _:
                    (zlo, zhi) = [float(x) for x in _.split()[0:2]]
                if 'xy xz yz' in _:
                    (xy, xz, yz) = [float(x) for x in _.split()[0:3]]
                if section == 'Atoms':
                    fields = _.split()
                    lenght = len(fields)
                    if lenght == 0:
                        continue
                    id = int(fields[0])
                    if style == "atomic" and (lenght == 5 or lenght == 8):
                        # id type x y z [tx ty tz]
                        pos_in[id] = (
                            int(fields[1]),
                            float(fields[2]),
                            float(fields[3]),
                            float(fields[4]),
                        )
                        if lenght == 8:
                            travel_in[id] = (
                                int(fields[5]),
                                int(fields[6]),
                                int(fields[7]),
                            )
                    else:
                        msg = f"Style '{style}' not supported or" + \
                              f"invalid number of fields {lenght}"
                        raise RuntimeError(msg)

        # set cell
        cell = np.zeros((3, 3))
        cell[0, 0] = xhi - xlo
        cell[1, 1] = yhi - ylo
        cell[2, 2] = zhi - zlo
        if xy is not None:
            cell[1, 0] = xy
        if xz is not None:
            cell[2, 0] = xz
        if yz is not None:
            cell[2, 1] = yz
        positions = np.zeros((N, 3))
        for id in pos_in.keys():
            ind = id - 1
            positions[ind, :] = [pos_in[id][1]+cell[0, 0]*travel_in[id][0],
                                 pos_in[id][2]+cell[1, 1]*travel_in[id][1],
                                 pos_in[id][3]+cell[2, 2]*travel_in[id][2]]
        return positions, cell

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        msg = "NEB calculation as implemented in LAMMPS\n"
        msg += f"Number of replicas :                     {self.nreplica}\n"
        msg += f"String constant :                        {self.Kspring}\n"
        msg += f"Sampling mode :                          {self.patoms.mode}\n"
        msg += "\n"
        return msg
