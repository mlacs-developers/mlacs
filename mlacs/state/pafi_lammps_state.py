import os
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ase import Atoms
from ase.units import fs, kB
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPC

from mlacs.state import LammpsState
from mlacs.utilities import (get_elements_Z_and_masses,
                             write_lammps_NEB_ASCIIfile,
                             _create_ASE_object)
from mlacs.utilities.io_lammps import (get_general_input,
                                       get_log_input,
                                       get_traj_input,
                                       get_interaction_input,
                                       get_last_dump_input)


# ========================================================================== #
# ========================================================================== #
class PafiLammpsState(LammpsState):
    """
    Class to manage PafiStates with LAMMPS

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    configurations: :class:`list`
        List of ase.Atoms object, the list contain initial and final
        configurations of the reaction path.
    reaction_coordinate: :class:`numpy.array` or `float`
        Value of the reaction coordinate for the constrained MD.
        Default ``0.5``
    Kspring: :class:`float`
        Spring constante for the NEB calculation.
        Default ``1.0``
    maxjump: :class:`float`
        Maximum atomic jump authorized for the free energy calculations.
        Configurations with an high `maxjump` will be removed.
        Default ``0.4``
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    damp: :class:`float` or ``None``
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.
    brownian: :class:`Bool`
        If ``True``, a Brownian thermostat is used for the thermostat.
        Else, a Langevin thermostat is used
        Default ``True``
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.
    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.
    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.
    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`
    init_momenta : :class:`numpy.ndarray` (optional)
        If ``None``, velocities are initialized with a
        Maxwell Boltzmann distribution
        N * 3 velocities for the initial configuration
    prt : :class:`Bool` (optional)
        Printing options. Default ``True``
    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 configurations,
                 reaction_coordinate=0.5,
                 Kspring=1.0,
                 maxjump=0.4,
                 dt=1.5,
                 damp=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 brownian=True,
                 fixcm=True,
                 logfile=None,
                 trajfile=None,
                 interval=50,
                 loginterval=50,
                 trajinterval=50,
                 rng=None,
                 init_momenta=None,
                 prt=True,
                 workdir=None):
        LammpsState.__init__(self,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             interval,
                             loginterval,
                             trajinterval,
                             rng,
                             init_momenta,
                             workdir)

        self.temperature = temperature
        self.nsteps = nsteps
        self.nsteps_eq = nsteps_eq
        self.NEBcoord = reaction_coordinate
        self.print = prt
        self.Kspring = Kspring
        self.maxjump = maxjump
        self.dt = dt
        self.damp = damp
        self.brownian = brownian
        self.confNEB = configurations
        if len(self.confNEB) != 2:
            raise TypeError('First and last configurations are not defined')
        self._get_lammps_command_replica()

        self.ispimd = False
        self.isrestart = False
        self.isappend = False

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
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
                     eq=False,
                     rep=None):
        """
        Function to run the PAFI dynamics
        """
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        if atom_style is None:
            atom_style = "atomic"

        atoms = supercell.copy()

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        if rep is None:
            fname = self.workdir + self.lammpsfname
            atfname = self.workdir + self.atomsfname
            spatoms = self.spline_atoms[0]
            spcoord = self.spline_coordinates[0]
        else:
            fname = self.MFEPworkdir + self.lammpsfname + f'.{rep}'
            atfname = self.MFEPworkdir + self.atomsfname + f'.{rep}'
            spatoms = atoms
            spcoord = self.spline_coordinates[rep]

        self._write_PafiPath_atoms(atfname,
                                   spatoms,
                                   spcoord)
        self.write_lammps_input_pafi(atoms,
                                     atom_style,
                                     bond_style,
                                     bond_coeff,
                                     angle_style,
                                     angle_coeff,
                                     pair_style,
                                     pair_coeff,
                                     model_post,
                                     nsteps,
                                     fname,
                                     atfname,
                                     rep)

        lammps_command = self.cmd + " -in " + fname + \
            " -sc out.lmp"
        cwd = self.workdir
        if rep is not None:
            lammps_command += f'.{rep}'
            cwd = self.MFEPworkdir
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=cwd,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

        if charges is not None:
            init_charges = atoms.get_initial_charges()
        atoms = read(self.workdir + "configurations.out")
        if charges is not None:
            atoms.set_initial_charges(init_charges)

        return atoms.copy()

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
                angle_coeff=None):
        """
        Run a NEB calculation with lammps. Use replicas.
        """
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
    def run_MFEP(self,
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
                 workdir=None,
                 ncpus=1,
                 restart=0,
                 fstop=0.001,
                 xi=None,
                 nsteps=10000,
                 interval=10,
                 nthrow=2000):
        """
        Run a MFEP calculation with lammps. Use replicas.
        """
        self.nsteps = nsteps
        self.MFEPworkdir = self.workdir + "MFEP/"
        if not os.path.exists(self.MFEPworkdir):
            os.makedirs(self.MFEPworkdir)
        if xi is None:
            xi = np.arange(0, 1.01, 0.01)
        if not hasattr(self, 'true_atoms'):
            self.run_NEB(pair_style,
                         pair_coeff,
                         model_post,
                         atom_style,
                         bonds,
                         angles,
                         bond_style,
                         bond_coeff,
                         angle_style,
                         angle_coeff)
            self.extract_NEB_configurations()
        self.compute_spline(xi)
        nrep = len(self.spline_atoms)
        with ThreadPoolExecutor(max_workers=ncpus) as executor:
            for rep in range(restart, nrep):
                atoms = self.spline_atoms[rep].copy()
                atoms.set_pbc([1, 1, 1])
                executor.submit(self.run_dynamics,
                                *(atoms,
                                  pair_style,
                                  pair_coeff,
                                  model_post,
                                  atom_style,
                                  bonds,
                                  angles,
                                  bond_style,
                                  bond_coeff,
                                  angle_style,
                                  angle_coeff,
                                  False,
                                  rep))
        self.pafi = []
        for rep in range(nrep):
            logfile = self.MFEPworkdir + f'pafi.log.{rep}'
            data = np.loadtxt(logfile).T[:, nthrow:].tolist()
            self.pafi.append(data)
        self.pafi = np.array(self.pafi)
        F = self.log_free_energy(xi)
        return F

# ========================================================================== #
    def write_lammps_input_pafi(self,
                                atoms,
                                atom_style,
                                bond_style,
                                bond_coeff,
                                angle_style,
                                angle_coeff,
                                pair_style,
                                pair_coeff,
                                model_post,
                                nsteps,
                                fname,
                                atfname,
                                rep=None):
        """
        Write the LAMMPS input for the constrained MD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        pbc = atoms.get_pbc()

        custom = "atom_modify  map array sort 0 0.0\n"
        custom += "neigh_modify every 2 delay 10" + \
                  " check yes page 1000000 one 100000\n\n"
        custom += "fix 1 all property/atom d_nx d_ny d_nz" + \
                  " d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
        filename = atfname + " fix 1 NULL PafiPath"
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
        input_string += self.get_pafi_input()
        if self.logfile is not None:
            input_string += get_log_input(self.loginterval, self.logfile)
        if self.trajfile is not None:
            input_string += get_traj_input(self.loginterval,
                                           self.trajfile,
                                           elem)
        if rep is None:
            rep = 0
        input_string += self.get_pafilogging_input(rep)
        input_string += get_last_dump_input(self.workdir,
                                            elem,
                                            nsteps)
        input_string += f"run  {nsteps}"

        with open(fname, "w") as f:
            f.write(input_string)

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
    def get_pafi_input(self):
        """
        Function to write the general parameters for PAFI dynamics
        """
        input_string = "#####################################\n"
        input_string += "# Compute relevant field for PAFI simulation\n"
        input_string += "#####################################\n"
        input_string += "timestep  {0}\n".format(self.dt / 1000)
        input_string += "thermo    1\n"
        input_string += "min_style fire\n"
        input_string += "compute   1 all property/atom d_nx d_ny d_nz "
        input_string += "d_dnx d_dny d_dnz d_ddnx d_ddny d_ddnz\n"
        input_string += "run 0\n"
        input_string += "\n"

        input_string += "# Set up PAFI Langevin/Brownian integration\n"
        if self.damp is None:
            damp = "$(10*dt)"
        else:
            damp = self.damp
        seed = self.rng.integers(99999)
        if self.brownian:
            input_string += "fix       pafihp all pafi 1 " + \
                            f"{self.temperature} {damp} {seed} " + \
                            "overdamped yes com yes\n"
        else:
            input_string += "fix       pafihp all pafi 1 " + \
                            f"{self.temperature} {damp} {seed} " + \
                            "overdamped no com yes\n"
        input_string += "\n"
        input_string += "run 0\n"
        input_string += "\n"
        input_string += "minimize 0 0 250 250\n"
        input_string += "reset_timestep  0\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_pafilogging_input(self, rep=0):
        """
        Function to write several PAFI outputs
        """
        input_string = "#####################################\n"
        input_string += "#          Logging\n"
        input_string += "#####################################\n"
        input_string += "variable    dU    equal f_pafihp[1]\n"
        input_string += "variable    dUerr equal f_pafihp[2]\n"
        input_string += "variable    psi   equal f_pafihp[3]\n"
        input_string += "variable    err   equal f_pafihp[4]\n"
        input_string += "compute     disp    all displace/atom\n"
        input_string += "compute     maxdisp all reduce max c_disp[4]\n"
        input_string += "variable    maxjump equal sqrt(c_maxdisp)\n"

        if self.isappend:
            input_string += 'fix logpafi all print 1 ' + \
                            '"${dU}  ${dUerr} ${psi} ${err} ${maxjump}" ' + \
                            f'append pafi.log.{rep} title ' + \
                            '"# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'
        else:
            input_string += 'fix logpafi all print 1 ' + \
                            '"${dU}  ${dUerr} ${psi} ${err} ${maxjump}" ' + \
                            f'file pafi.log.{rep} title ' + \
                            '"# dU/dxi  (dU/dxi)^2  psi  err  maxjump"\n'
        input_string += "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

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

        from mlacs.utilities import interpolate_points as IP

        if xi is None:
            xi = self.NEBcoord

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
    def _COM_corrections(self, spline):
        """
        Correction of the path tangent to have zero center of mass
        """
        N = len(self.confNEB[0])
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
    def _write_PafiPath_atoms(self, filename, atoms, spline):
        """
        Write the lammps data file for a constrained MD, from an Atoms object.
            - Three first columns: atomic positons at reaction coordinate xi.
            - Three next columns:  normalized atomic first derivatives at
                reaction coordinate xi, with the corrections of the COM.
            - Three last columns:  normalized atomic second derivatives at
                reaction coordinate xi.
        """
        symbol = atoms.get_chemical_symbols()
        species = sorted(set(symbol))
        N = len(symbol)
        cell = atoms.get_cell()
        instr = '#{0} (written by MLACS)\n\n'.format(filename)
        instr += '{0} atoms\n'.format(N)
        instr += '{0} atom types\n'.format(len(species))
        instr += '0 {0} xlo xhi\n'.format(cell[0, 0])
        instr += '0 {0} ylo yhi\n'.format(cell[1, 1])
        instr += '0 {0} zlo zhi\n'.format(cell[2, 2])
        instr += '\nAtoms\n\n'
        for i in range(N):
            strformat = '{:>6} ' + '{:>3} ' + ('{:12.8f} ' * 3) + '\n'
            instr += strformat.format(i+1, species.index(symbol[i]) + 1,
                                      *spline[i, :3])
        instr += '\nPafiPath\n\n'
        for i in range(N):
            strformat = '{:>6} ' + ('{:12.8f} ' * 9) + '\n'
            instr += strformat.format(i+1, *spline[i, :3], *spline[i, 9:])
        with open(filename, 'w') as w:
            w.write(instr)

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
    def log_free_energy(self, xi):
        """
        Extract the MFEP gradient from log files.
        Integrate the MFEP and compute the Free energy barier.
        """

        from mlacs.utilities.miscellanous import integrate_points as IntP

        dF = []
        psi = []
        cor = []
        maxjump = []
        for rep in range(len(xi)):
            dF.append(np.average(self.pafi[rep, 0]))
            psi.append(np.average(self.pafi[rep, 2]))
            cor.append(np.average(
                np.log(np.abs(self.pafi[rep, 2] / self.pafi[0, 2]))))
            maxjump.append(
                [x for x in self.pafi[rep, 4].tolist() if x >= self.maxjump])
        dF = np.array(dF)
        cor = np.array(cor)
        psi = np.array(psi)
        maxjump = np.array(maxjump)
        F = -np.array(IntP(xi, dF, xi))
        Fcor = -np.array(IntP(xi, dF + kB * self.temperature * cor, xi))
        # Ipsi = np.array(IntP(xi, psi, xi))
        if self.print:
            with open('free_energy.dat', 'w') as w:
                dFM = max(F) - min(F)
                w.write(f'##  Free energy barier: {dFM} eV  ' +
                        '##  xi  <dF/dxi>  <F(xi)>  <psi>  ' +
                        'cor  Fcor(xi)  Nmaxjump  ##\n')
                strformat = ('{:12.8f} ' * 7) + '\n'
                for i in range(len(xi)):
                    w.write(strformat.format(xi[i],
                                             dF[i],
                                             F[i],
                                             psi[i],
                                             kB * self.temperature * cor[i],
                                             Fcor[i],
                                             len(maxjump[i])))
        return Fcor

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
        msg += "Constrain dynamics as implemented in LAMMPS with fix PAFI\n"
        msg += f"Temperature (in Kelvin) :                {self.temperature}\n"
        msg += f"Number of MLMD equilibration steps :     {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :        {self.nsteps}\n"
        msg += f"Timestep (in fs) :                       {self.dt}\n"
        msg += f"Themostat damping parameter (in fs) :    {damp}\n"
        msg += f"Reaction coordinate :                    {coord}\n"
        msg += "\n"
        return msg
