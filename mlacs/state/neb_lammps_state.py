import os
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ase.units import fs, kB
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data

from .lammps_state import LammpsState

from ..utilities import (get_elements_Z_and_masses,
                             write_lammps_NEB_ASCIIfile)

from ..utilities import (get_elements_Z_and_masses,
                         write_lammps_NEB_ASCIIfile,
                         _create_ASE_object)

from ..utilities.io_lammps import (get_general_input,
                                   get_log_input,
                                   get_traj_input,
                                   get_interaction_input,
                                   get_last_dump_input)

from ..utilities import integrate_points as IntP
from ..utilities import interpolate_points as IP


# ========================================================================== #
# ========================================================================== #
class NebLammpsState(LammpsState):
    """
    Class to manage NEB with LAMMPS

    Parameters
    ----------
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        Default ``None``
    Kspring: :class:`float`
        Spring constante for the NEB calculation.
        Default ``1.0``
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
    """
    def __init__(self,
                 configurations,
                 reaction_coordinate=None,
                 neb_configurations=1,
                 Kspring=1.0,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 prt=True,
                 workdir=None):

        self.NEBcoord = reaction_coordinate
        if self.NEBcoord is None:
            self.splprec = 1001
        self.include_neb = neb_configurations
        self.print = prt
        self.Kspring = Kspring
        self.confNEB = configurations
        if len(self.confNEB) != 2:
            raise TypeError('First and last configurations are not defined')
        self._get_lammps_command_replica()

        self.ispimd = False
        self.isrestart = False
        self.isappend = False

# ========================================================================== #
    def run_NEB(self,
                pair_style,
                pair_coeff,
                model_post=None,
                atom_style="atomic",
                bonds=None,
                angles=None,
                bond_style=None,
                bond_coeff=None,
                angle_style=None,
                angle_coeff=None,
                workdir=None):
        """
        Run a NEB calculation with lammps. Use replicas.
        """
        if workdir is not None:
            self.workdir = workdir
        self.NEBworkdir = self.workdir + "NEB/"
        if not os.path.exists(self.NEBworkdir):
            os.makedirs(self.NEBworkdir)
        write_lammps_data(self.NEBworkdir+'atoms-0.data',
                          self.confNEB[0])
        write_lammps_NEB_ASCIIfile(self.NEBworkdir+'atoms-1.data',
                                   self.confNEB[1])

        fname = self.NEBworkdir + "lammps_input.in"
        self.write_lammps_input_NEB(self.confNEB[0],
                                    atom_style,
                                    bond_style,
                                    bond_coeff,
                                    angle_style,
                                    angle_coeff,
                                    pair_style,
                                    pair_coeff,
                                    model_post,
                                    fname)
        lammps_command = self.cmdreplica + " -in " + fname + \
            " -sc out.lmp"
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=self.NEBworkdir,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

# ========================================================================== #
    def write_lammps_input_NEB(self,
                               atoms,
                               atom_style,
                               bond_style,
                               bond_coeff,
                               angle_style,
                               angle_coeff,
                               pair_style,
                               pair_coeff,
                               model_post,
                               fname):
        """
        Write the LAMMPS input for NEB simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        pbc = atoms.get_pbc()

        custom = "atom_modify  map array sort 0 0.0\n"
        custom += "neigh_modify every 2 delay 10" + \
                  " check yes page 1000000 one 100000\n\n"
        filename = "atoms-0.data"
        input_string = ""
        input_string += get_general_input(pbc,
                                          masses,
                                          charges,
                                          atom_style,
                                          filename,
                                          custom)
        input_string += get_interaction_input(bond_style,
                                              bond_coeff,
                                              angle_style,
                                              angle_coeff,
                                              pair_style,
                                              pair_coeff,
                                              model_post)
        input_string += self.get_neb_input()

        with open(fname, "w") as f:
            f.write(input_string)

# ========================================================================== #
    def get_neb_input(self):
        """
        Function to write the general parameters for NEB
        """
        input_string = "#####################################\n"
        input_string += "# Compute relevant field for NEB simulation\n"
        input_string += "#####################################\n"
        input_string += "timestep    {0}\n".format(self.dt / (fs * 1000))
        input_string += "thermo      1\n"
        input_string += f"fix         neb all neb {self.Kspring} " + \
                        "parallel ideal\n"
        input_string += "run 100\n"
        input_string += "reset_timestep  0\n\n"
        input_string += "variable    i equal part\n"
        input_string += "min_style   quickmin\n"
        input_string += "neb         0.0 0.001 200 100 10 final atoms-1.data\n"
        input_string += "write_data  neb.$i\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def extract_NEB_configurations(self):
        """
        Step 1:
        Extract the positions and energies of a NEB calculation for all
        replicas.
        """
        true_atoms = []
        true_coordinates = []
        Z = self.confNEB[0].get_atomic_numbers()
        for rep in range(int(self.nreplica)):
            nebfile = self.NEBworkdir + f'neb.{rep}'
            positions, cell = self._read_lammpsdata(nebfile)
            true_coordinates.append(positions)
            check = False
            with open(self.NEBworkdir + f'log.lammps.{rep}') as r:
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
        # RB check float
        self.true_energies = np.array([true_atoms[i].get_potential_energy()
                                       for i in range(self.nreplica)])
        self.true_energies = self.true_energies.astype(float)
        if self.print:
            write(self.NEBworkdir + 'pos_neb_path.xyz',
                  true_atoms, format='extxyz')

# ========================================================================== #
    def compute_spline(self, xi=None):
        """
        Step 2:
        Compute a 1D CubicSpline interpolation from a NEB calculation.
        The function also set up the lammps data file for a constrained MD.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at
                reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at
                reaction coordinate xi.
        """
        Z = self.confNEB[0].get_atomic_numbers()
        N = len(self.confNEB[0])

        if xi is None:
            if self.NEBcoord is not None:
                xi = self.NEBcoord
            else:
                x = np.linspace(0, 1, self.splprec)
                y = IP(self.path_coordinates,
                       self.true_energies,
                       x, 0, border=1)
                y = np.array(y)
                xi = x[y.argmax()]
                print(xi)

        self.spline_energies = IP(self.path_coordinates,
                                  self.true_energies,
                                  xi, 0, border=1)
        spline_coordinates = []

        # Spline interpolation of the referent path and calculation of
        # the path tangent and path tangent derivate
        for i in range(N):
            coord = [IP(self.path_coordinates,
                        self.true_coordinates[:, i, j],
                        xi, 0) for j in range(3)]
            coord.extend([IP(self.path_coordinates,
                             self.true_coordinates[:, i, j],
                             xi, 1) for j in range(3)])
            coord.extend([IP(self.path_coordinates,
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
                self.confNEB[0].get_cell(), self.spline_energies))
        else:
            for rep in range(len(xi)):
                self.spline_coordinates.append(self._COM_corrections(
                    spline_coordinates[:, :, rep].tolist()))
            self.spline_coordinates = np.array(
                    self.spline_coordinates).round(8)
            for rep in range(len(xi)):
                self.spline_atoms.append(_create_ASE_object(
                    Z, np.hsplit(self.spline_coordinates[rep, :, :], 5)[0],
                    self.confNEB[0].get_cell(), self.spline_energies[rep]))
        if self.print:
            write(self.NEBworkdir + 'pos_neb_spline.xyz',
                  self.spline_atoms, format='extxyz')

# ========================================================================== #
    def _get_lammps_command_replica(self):
        '''
        Function to load the batch command to run LAMMPS with replica
        '''
        envvar = "ASE_LAMMPSREPLICA_COMMAND"
        cmdreplica = os.environ.get(envvar)
        self.cmdreplica = cmdreplica
        self.nreplica = None
        if cmdreplica is None:
            if '-n' in self.cmd:
                index = self.cmd.split().index('-n')+1
                self.nreplica = int(self.cmd.split()[index])
                self.cmdreplica = self.cmd + f' -partition {self.nreplica}x1 '
            else:
                msg = "ASE_LAMMPSREPLICA_COMMAND variable not defined"
                raise TypeError(msg)
        if self.nreplica is None:
            index = self.cmdreplica.split().index('-partition')+1
            self.nreplica = int(self.cmdreplica.split()[index].split('x')[0])

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
    @property
    def get_images(self):
        """
        Function to return NEB true configurations
        """
        atoms = []
        if self.include_neb is None:
            return atoms

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = self.damp
        if damp is None:
            damp = 100 * self.dt
        coord = self.NEBcoord
        if coord is None:
            coord = 0.5

        msg = "NEB calculation as implemented in LAMMPS\n"
        msg += f"Number of replicas :                     {self.nreplica}\n"
        msg += f"String constant :                        {self.Kspring}\n"
        msg += "\n"
        if isinstance(coord, float):
            msg += f"Reaction coordinate :                    {coord}\n"
        else:
            step = coord[1]-coord[0]
            i, f = (coord[0], coord[-1])
            msg += f"Reaction interval :                      [{i} : {f}]\n"
            msg += f"Reaction step interval :                 {step}\n"
        msg += "\n"
        return msg
