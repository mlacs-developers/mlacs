import os

import numpy as np

from scipy.spatial import distance

from ase.io import write
from ase.io.lammpsdata import write_lammps_data

from .lammps_state import BaseLammpsState
from ..utilities.io_lammps import (LammpsBlockInput,
                                   EmptyLammpsBlockInput)

from ..utilities import (get_elements_Z_and_masses,
                         _create_ASE_object)

from ..utilities.io_lammps import write_lammps_NEB_ASCIIfile

from ..utilities import interpolate_points as intpts


# ========================================================================== #
# ========================================================================== #
class NebLammpsState(BaseLammpsState):
    """
    Class to manage Nudged Elastic Band (NEB) calculation with LAMMPS.
    This class is a part of TransPath objects, meaning that it produces
    positions interpolation according to a reaction coordinate.

    Parameters
    ----------
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.

    xi_coordinate: :class:`numpy.array` or `float`
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
        Default ``rdm_memory``

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

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created

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

    def __init__(self, configurations, xi_coordinate=None,
                 min_style="quickmin", Kspring=1.0, etol=0.0, ftol=1.0e-3,
                 dt=1.5, nimages=None, nprocs=None, mode="rdm_memory",
                 linear=False, prt=False,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, workdir=None, blocks=None):
        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         workdir, blocks)

        self.dt = dt
        self.pressure = None

        self.xi = xi_coordinate
        self.style = min_style
        self.criterions = (etol, ftol)
        self.finder = None
        self.nprocs = nprocs
        self.nreplica = nimages
        self.atomsfname = "atoms-0.data"
        self.mode = mode
        if self.xi is None:
            self.splprec = 1001
            self.finder = [0.0, 1.0]
        self.print = prt
        self.Kspring = Kspring
        self.atoms = configurations
        if len(self.atoms) != 2:
            raise TypeError('First and last configurations are not defined')
        self.masses = configurations[0].get_masses()

        self.linear = linear

# ========================================================================== #
    def _write_lammps_atoms(self, atoms, atom_style):
        """

        """
        write_lammps_data(self.workdir / self.atomsfname,
                          self.atoms[0],
                          velocities=False,
                          atom_style=atom_style)
        write_lammps_NEB_ASCIIfile(self.workdir / "atoms-1.data",
                                   self.atoms[1])

# ========================================================================== #
    def _get_block_init(self, atoms, atom_style):
        """

        """
        pbc = atoms.get_pbc()
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

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
    def _get_block_lastdump(self, atoms, eq):
        return EmptyLammpsBlockInput("empty_lastdump")

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """

        """
        self.extract_NEB_configurations()
        xi = self.compute_spline(self._xifinder(self.mode))
        atoms = self.spline_atoms[-1].copy()
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        if self.print:
            with open('Coordinate_sampling.dat', 'a') as a:
                a.write(f'{xi}\n')
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
        true_atoms = []
        true_coordinates = []
        Z = self.atoms[0].get_atomic_numbers()
        for rep in range(int(self.nreplica)):
            nebfile = self.workdir / f'neb.{rep}'
            positions, cell = self._read_lammpsdata(nebfile)
            true_coordinates.append(positions)
            check = False
            with open(self.workdir / f'log.lammps.{rep}') as r:
                for _ in r:
                    if check:
                        etotal = _.split()[2]
                        break
                    if 'initial, next-to-last, final =' in _:
                        check = True
            atoms = _create_ASE_object(Z, positions, cell, etotal)
            true_atoms.append(atoms)
        self.true_atoms = true_atoms
        self.path_coordinates = np.arange(self.nreplica)/(self.nreplica-1)
        self.true_coordinates = np.array(true_coordinates)
        self.eff_masses = np.sum(self._compute_weight_masses() * self.masses)
        self.true_energies = np.array([true_atoms[i].get_potential_energy()
                                       for i in range(self.nreplica)])
        self.true_energies = self.true_energies.astype(float)
        if self.print:
            write(self.workdir / 'pos_neb_path.xyz',
                  true_atoms, format='extxyz')

# ========================================================================== #
    def compute_spline(self, xi=None):
        """
        Compute a 1D CubicSpline interpolation from a NEB calculation.
        The function also set up the lammps data file for a constrained MD.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at
                reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at
                reaction coordinate xi.
        """
        Z = self.atoms[0].get_atomic_numbers()
        N = len(self.atoms[0])

        if xi is None:
            if self.xi is not None:
                xi = self.xi
            else:
                x = np.linspace(0, 1, self.splprec)
                y = intpts(self.path_coordinates, self.true_energies,
                           x, 0, border=1)
                y = np.array(y)
                xi = x[y.argmax()]

            if self.finder is not None:
                self.finder.append(xi)

        self.spline_energies = intpts(self.path_coordinates,
                                      self.true_energies,
                                      xi, 0, border=1)
        spline_coordinates = []

        # Spline interpolation of the referent path and calculation of
        # the path tangent and path tangent derivate
        for i in range(N):
            coord = [intpts(self.path_coordinates,
                            self.true_coordinates[:, i, j],
                            xi, 0) for j in range(3)]
            coord.extend([intpts(self.path_coordinates,
                                 self.true_coordinates[:, i, j],
                                 xi, 1) for j in range(3)])
            coord.extend([intpts(self.path_coordinates,
                                 self.true_coordinates[:, i, j],
                                 xi, 2) for j in range(3)])
            spline_coordinates.append(coord)
        spline_coordinates = np.array(spline_coordinates)

        self.spline_atoms = []
        self.spline_coordinates = []
        if isinstance(xi, float):
            self.spline_coordinates.append(np.array(self._COM_corrections(
                spline_coordinates.tolist())).round(8))
            self.spline_coordinates = np.array(self.spline_coordinates)
            self.spline_atoms.append(_create_ASE_object(
                Z, np.hsplit(self.spline_coordinates[0], 5)[0],
                self.atoms[0].get_cell(), self.spline_energies))
        else:
            for rep in range(len(xi)):
                self.spline_coordinates.append(self._COM_corrections(
                    spline_coordinates[:, :, rep].tolist()))
            self.spline_coordinates = np.array(
                    self.spline_coordinates).round(8)
            for rep in range(len(xi)):
                self.spline_atoms.append(_create_ASE_object(
                    Z, np.hsplit(self.spline_coordinates[rep, :, :], 5)[0],
                    self.atoms[0].get_cell(), self.spline_energies[rep]))
        if self.print:
            write(self.workdir / 'pos_neb_spline.xyz',
                  self.spline_atoms, format='extxyz')
        return xi

# ========================================================================== #
    def _xifinder(self, mode):
        """
        Return the reaction coordinate xi(R).
        """
        def find_dist(_l):
            m = []
            _l.sort()
            for i, val in enumerate(_l[1:]):
                m.append(np.abs(_l[i+1] - _l[i]))
            i = np.array(m).argmax()
            return _l[i+1], _l[i]
        if self.xi is not None:
            return self.xi
        if isinstance(mode, float):
            return mode
        elif mode == 'rdm_spl':
            return np.random.uniform(0, 1)
        elif mode == 'rdm_memory':
            x, y = find_dist(self.finder)
            return np.random.uniform(x, y)
        elif mode == 'rdm_true':
            r = np.random.default_rng()
            x = r.integers(self.nreplica) / self.nreplica
            return x
        else:
            return None

# ========================================================================== #
    def _compute_weight_masses(self):
        """
        Return weights for effective masse.
        """
        coordinates = np.transpose(self.true_coordinates, (1, 0, 2))
        weight = np.array([np.max(distance.cdist(d, d, "euclidean"))
                           for d in coordinates])
        weight = weight / np.max(weight)
        return weight

# ========================================================================== #
    def _COM_corrections(self, spline):
        """
        Correction of the path tangent to have zero center of mass
        """
        N = len(self.atoms[0])
        pos, der, der2 = np.hsplit(np.array(spline), 3)
        com = np.array([np.average(der[:, i]) for i in range(3)])
        norm = 0
        for i in range(N):
            norm += np.sum([(der[i, j] - com[j])**2 for j in range(3)])
        norm = np.sqrt(norm)
        for i in range(N):
            spline[i].extend([(der[i, j] - com[j]) / norm for j in range(3)])
            spline[i].extend([der2[i, j] / norm / norm for j in range(3)])
        return spline

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS with replica.
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp_mpi"
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
        msg += f"Sampling mode :                          {self.mode}\n"
        msg += "\n"
        return msg
