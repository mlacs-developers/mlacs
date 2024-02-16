"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
import sys
import shlex
import time
import numpy as np
import xml.etree.cElementTree as ET
from subprocess import Popen, PIPE

from ase import Atoms
from ase.units import Bohr, fs
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
from ase.calculators.singlepoint import SinglePointCalculator as SPCalc

from . import LammpsState
from ..utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class IpiState(LammpsState):
    """
    State Class for running a Path Integral MD simulation as implemented
    in I-Pi and using sockets to compute properties with LAMMPS.

    Parameters
    ----------

    temperature: :class:`float`
        Temperature of the simulation, in Kelvin

    pressure: :class:`float` (optional)
        Pressure for the simulation, in GPa
        Default ``0`` GPa.

    stress : (3x3) :class:`np.ndarray` (optional)
        Stress for the simulation, in GPa
        Default ``0`` GPa for the nine coefficients.
        Pressure matrice if pressure is not None.

    ensemble : 'nve', 'nvt', 'npt' or 'nst' (optional)
        Define the ensemble that will be sampled.
        Default ``'nvt'``

    nbeads : :class:`int` (optional)
        Number of breads.
        Default ``1``, to do classical MD.

    paralbeads : :class:`int` (optional)
        Reduce parallelisation over breads.
        Default ``None``, means full parallelisation.

    socketname : :class:`str` (optional)
        Name of sockets.

    mode : :class:`str` (optional)
        Specifies whether the driver interface will listen onto a
        internet 'inet' or a unix 'unix' socket.
        Default ``'unix'``

    prefix : :class:`str` (optional)
        Prefix for output names.
        Default simulation but should be OtfMLACS.prefix

    thermostyle : 'langevin', 'svr', 'pile_l' or 'pile_g' (optional)
        Define the style for the thermostat.
        Default 'pile_l', white noise langevin thermostat
        to the normal mode representation.

    barostyle : 'isotropic' or 'anisotropic' (optional)
        Define the style for the barostat.
        Default 'isotropic' for NPT, 'anisotropic' for NST.

    damp : :class:`float` (optional)
        Damping parameter. If None a damping parameter of 100 timestep is used.
        Default ``None``.

    pdamp : :class:`float` (optional)
        Damping parameter for the barostat. Default 1000 timestep is used.
        Default ``None``.

    pilelambda : :class:`float` (optional)
        Scaling for the PILE damping relative to the critical damping.
        gamma_k = 2*pilelambda*omega_k
        Default ``0.5``, ``0.2`` is another typical value.

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

    fixcm : :class:`bool` (optional)
        Fix position and momentum center of mass. Default ``True``.

    loginterval : :class:`int` (optional)
        Number of steps between log and traj writing. Default ``50``

    printcentroid: :class:`Bool` (optional)
        If ``True``, the centroid of the trajectory is written

    rng : :class:`int` (optional)
        Default correspond to numpy.random.default_rng()

    init_momenta : :class:`numpy.ndarray` (optional)
        Gives the (Nat, 3) shaped momenta array that will be used
        to initialize momenta when using
        the `initialize_momenta` function.
        If the default ``None`` is set, momenta are initialized with a
        Maxwell Boltzmann distribution.

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If none, a LammpsMLMD directory is created
    """
    def __init__(self,
                 temperature,
                 pressure=None,
                 stress=None,
                 ensemble='nvt',
                 nbeads=1,
                 paralbeads=None,
                 socketname='mysocket',
                 mode='unix',
                 prefix='simulation',
                 thermostyle='pile_l',
                 barostyle='isotropic',
                 diagonal=True,
                 damp=None,
                 pdamp=None,
                 pilelambda=0.5,
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 loginterval=50,
                 printcentroid=True,
                 rng=None,
                 init_momenta=None,
                 workdir=None):

        LammpsState.__init__(self,
                             temperature,
                             pressure,
                             dt=dt,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             fixcm=fixcm,
                             loginterval=loginterval,
                             rng=rng,
                             init_momenta=init_momenta,
                             workdir=workdir)

        self.socketname = socketname
        self.hostname = None
        self.socketmode = mode
        if self.socketmode == 'inet':
            self.hostname = os.environ.get('HOSTNAME')
        self.ensemble = ensemble
        self.thermostyle = thermostyle
        self.barostyle = barostyle
        self.pilelambda = pilelambda
        self.damp = damp
        self.pdamp = pdamp
        if self.pressure is None and ensemble == 'npt':
            self.pressure = 0
        if stress is None and ensemble == 'nst':
            self.barostyle = 'anisotropic'
            self.stress = np.zeros((3, 3))
            if pressure is not None:
                self.stress = -np.identity(3)*pressure/3
        self.diagonal = diagonal

        self.printcentroid = printcentroid
        self.nbeads = nbeads  # Default value to do classical MD
        if self.nbeads > 1:
            self.ispimd = True
        else:
            self.printcentroid = False
        self.paralbeads = paralbeads
        if self.paralbeads is None:
            self.paralbeads = 1
        self.prefix = prefix

        self.ipiatomsfname = "ipi_atoms.xyz"
        self.ipifname = "ipi_input.xml"

        self._get_ipi_cmd()

        try:
            self.rngint = self.rng.integers
        except AttributeError:
            self.rngint = self.rng.randint
        # Port number should be taken in the range 1025-65535.
        # Port number also serve as seed for velocity distribution,
        # it works ...
        self.rngnum = self.rngint(1025, 65535)

# ========================================================================== #
    def _get_ipi_cmd(self):
        '''
        Function to load the batch command to run i-Pi
        '''
        envvar = "ASE_I-PI_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "i-pi "
        self.cmdipi = cmd

# ========================================================================== #
    def _build_lammps_command(self, bead=''):
        """
        """
        lammps_command = self.cmd + ' -in ' + \
            self.lammpsfname + " -screen log." + bead + ' -log none'
        return lammps_command

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     nbeads=1):
        """
        """
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        atoms = supercell.copy()

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        write_lammps_data(self.workdir + self.atomsfname,
                          atoms,
                          velocities=True,
                          atom_style=atom_style)
        pbc = atoms.pbc  # We need it when reading the simulation

        self.initialize_momenta(atoms)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        atomswrite = atoms.copy()
        atomswrite.positions = atoms.get_positions() / Bohr
        write(self.workdir + self.ipiatomsfname, atomswrite, format='xyz')
        self.write_lammps_input(atoms,
                                atom_style,
                                pair_style,
                                pair_coeff,
                                model_post,
                                1000000000,
                                self.temperature,
                                self.pressure)
        self.write_ipi_input(atoms, nsteps)
        ipi_command = f"{self.cmdipi} {self.ipifname} > ipi.log"
        # We start by running ipi alone
        ipi_handle = Popen(ipi_command, shell=True, cwd=self.workdir,
                           stderr=PIPE)
        time.sleep(5)  # We need to wait a bit for i-pi ready the socket
        # We get all LAMMPS run in an array
        alllammps = []
        for i in range(self.paralbeads):
            alllammps.append(self._build_lammps_command(str(i)))
        # And we run all LAMMPS instance
        [Popen(shlex.split(i, posix=(os.name == "posix")),
               cwd=self.workdir) for i in alllammps]
        ipi_handle.wait()
        if ipi_handle.returncode != 0:
            msg = "i-pi stopped prematurely"
            raise RuntimeError(msg)
        atoms = self.create_ase_atom(pbc, nbeads)

        # Set the simulation T and P for weighting purpose
        if self.temperature is not None:
            atoms.info['simulation_temperature'] = self.temperature
        if self.pressure is not None:
            atoms.info['simulation_pressure'] = self.pressure

        return atoms

# ========================================================================== #
    def write_ipi_input(self, atoms, nsteps):
        '''
        '''
        def _add_textxml(element, text):
            element.text = text
            return element

        def _add_tailxml(element, text):
            element.tail = text
            return element

        def _add_Subelements(element, subelements):
            if type(subelements) is list:
                for _ in subelements:
                    element.append(_)
            else:
                element.append(subelements)
            return element

        # Simulation parameters:
        #  - Output
        #  - Total Steps
        #  - FFsocket
        #  - System

        # Output parameters
        propstr = ' [ step, ' + \
                  'time{picosecond}, ' + \
                  'conserved, ' + \
                  'temperature{kelvin}, ' + \
                  'kinetic_cv, ' + \
                  'potential, ' + \
                  'pressure_cv{gigapascal}, ' + \
                  'volume, ' + \
                  'cell_h] '
        attrib_tmp = {'stride': str(self.loginterval),
                      'filename': 'out'}
        properties = _add_textxml(ET.Element('properties',
                                             attrib=attrib_tmp),
                                  propstr)
        trajarr = [properties]
        if self.loginterval is not None:
            attrib_tmp = {'stride': str(self.loginterval),
                          'filename': 'pos',
                          'format': 'xyz',
                          'cell_units': 'angstrom'}
            trajectory0 = _add_textxml(ET.Element('trajectory',
                                                  attrib=attrib_tmp),
                                       'positions{angstrom}')
            trajarr.append(trajectory0)

            attrib_tmp = {'stride': str(self.loginterval),
                          'filename': 'for',
                          'format': 'xyz',
                          'cell_units': 'angstrom'}
            trajectory1 = _add_textxml(ET.Element('trajectory',
                                                  attrib=attrib_tmp),
                                       'forces{ev/ang}')
            trajarr.append(trajectory1)

            attrib_tmp = {'stride': str(self.loginterval),
                          'filename': 'vel',
                          'format': 'xyz',
                          'cell_units': 'angstrom'}
            trajectory2 = _add_textxml(ET.Element('trajectory',
                                                  attrib=attrib_tmp),
                                       'velocities{m/s}')
            trajarr.append(trajectory2)

            if self.printcentroid:
                attrib_tmp = {'stride': str(self.loginterval),
                              'filename': 'pos_c',
                              'format': 'xyz',
                              'cell_units': 'angstrom'}
                trajectory3 = _add_textxml(ET.Element('trajectory',
                                                      attrib=attrib_tmp),
                                           'x_centroid{angstrom}')
                trajarr.append(trajectory3)

                attrib_tmp = {'stride': str(self.loginterval),
                              'filename': 'for_c',
                              'format': 'xyz',
                              'cell_units': 'angstrom'}
                trajectory4 = _add_textxml(ET.Element('trajectory',
                                                      attrib=attrib_tmp),
                                           'f_centroid{ev/ang}')
                trajarr.append(trajectory4)

                attrib_tmp = {'stride': str(self.loginterval),
                              'filename': 'vel_c',
                              'format': 'xyz',
                              'cell_units': 'angstrom'}
                trajectory5 = _add_textxml(ET.Element('trajectory',
                                                      attrib=attrib_tmp),
                                           'v_centroid{m/s}')
                trajarr.append(trajectory5)

        # Adding trajectory for outputs
        attrib_tmp = {'stride': str(nsteps),
                      'filename': 'outpos',
                      'format': 'xyz',
                      'cell_units': 'angstrom'}
        outtrajpos = _add_textxml(ET.Element('trajectory',
                                             attrib=attrib_tmp),
                                  'positions{angstrom}')
        trajarr.append(outtrajpos)

        attrib_tmp = {'stride': str(nsteps),
                      'filename': 'outfor',
                      'format': 'xyz',
                      'cell_units': 'angstrom'}
        outtrajfor = _add_textxml(ET.Element('trajectory',
                                             attrib=attrib_tmp),
                                  'forces{ev/ang}')
        trajarr.append(outtrajfor)

        attrib_tmp = {'stride': str(nsteps),
                      'filename': 'outvel',
                      'format': 'xyz',
                      'cell_units': 'angstrom'}
        outtrajvel = _add_textxml(ET.Element('trajectory',
                                             attrib=attrib_tmp),
                                  'velocities{m/s}')
        trajarr.append(outtrajvel)

        output = ET.Element('output', attrib={'prefix': self.prefix})
        output = _add_Subelements(output, trajarr)

        # Total Steps
        total_steps = _add_textxml(ET.Element('total_steps'), str(nsteps))

        # Prng parameters
        seed = _add_textxml(ET.Element('seed'),
                            str(self.rng.integers(0, 999999)))
        prng = ET.Element('prng')
        prng.append(seed)

        # FFsocket parameters
        address = _add_textxml(ET.Element('address'), self.socketname)
        ffsocket = ET.Element('ffsocket',
                              attrib={'name': 'lammps',
                                      'mode': self.socketmode})
        # TODO Test latency: Number of seconds the thread will wait.
        if self.socketmode == 'inet':
            address = _add_textxml(ET.Element('address'), 'localhost')
            latency = _add_textxml(ET.Element('latency'), str(0))
            port = _add_textxml(ET.Element('port'), str(self.rngnum))
            ffsocket = _add_Subelements(ffsocket, [latency, port])
        ffsocket.append(address)

        # System parameters
        forces = ET.Element('forces')
        # TODO Possibility to use Abinit socket (force.attrib = 'abinit').
        force = ET.Element('force',
                           attrib={'name': 'lammps',
                                   'forcefield': 'lammps'})
        forces.append(force)

        initialize = ET.Element('initialize',
                                attrib={'nbeads': str(self.nbeads)})
        fileatom = _add_textxml(ET.Element('file',
                                attrib={'mode': 'xyz'}), self.ipiatomsfname)
        cell = _add_textxml(ET.Element('cell', attrib={'units': 'angstrom'}),
                            np.array2string(atoms.get_cell().reshape(9),
                                            separator=', '))
        velocities = _add_textxml(ET.Element('velocities',
                                  attrib={'mode': 'thermal',
                                          'units': 'kelvin'}),
                                  str(self.temperature))
        initialize = _add_Subelements(initialize, [fileatom, cell, velocities])

        ensemble = ET.Element('ensemble')
        temperature = _add_textxml(ET.Element('temperature',
                                   attrib={'units': 'kelvin'}),
                                   str(self.temperature))
        ensemble.append(temperature)
        if self.ensemble == 'npt':
            pressure = _add_textxml(ET.Element('pressure',
                                    attrib={'units': 'gigapascal'}),
                                    str(self.pressure))
            ensemble.append(pressure)
        if self.ensemble == 'nst':
            stress = _add_textxml(ET.Element('stress',
                                  attrib={'units': 'gigapascal'}),
                                  np.array2string(self.stress.reshape(9),
                                                  separator=', '))
            ensemble.append(stress)

        motion = ET.Element('motion', attrib={'mode': 'dynamics'})

        # Dynamics parameters
        # TODO Possibility to use other motion mode.
        # TODO Possibility to use other dynamics mode.
        # Currently implemented : nve, nvt, npt, nst
        dynamics = ET.Element('dynamics', attrib={'mode': self.ensemble})

        if self.damp is None:
            damp = 100*self.dt
        tdamp = _add_textxml(ET.Element('tau',
                                        attrib={'units': 'femtosecond'}),
                             str(damp))

        # Setup Barostats
        if self.ensemble == 'npt' or self.ensemble == 'nst':
            barostat = ET.Element('barostat',
                                  attrib={'mode': self.barostyle})
            thermostatb = ET.Element('thermostat',
                                     attrib={'mode': 'langevin'})
            thermostatb.append(tdamp)
            if self.pdamp is None:
                pdamp = damp*2
            h0 = _add_textxml(ET.Element('h0'),
                              np.array2string(atoms.get_cell().reshape(9),
                                              separator=', '))
            pdamp = _add_textxml(ET.Element('tau',
                                            attrib={'units': 'femtosecond'}),
                                 str(pdamp))
            if self.diagonal:
                diagonal = _add_textxml(ET.Element('hfix'),
                                        "[offdiagonal]")
                barostat = _add_Subelements(barostat,
                                            [thermostatb, pdamp, h0, diagonal])
            else:
                barostat = _add_Subelements(barostat,
                                            [thermostatb, pdamp, h0])
            dynamics.append(barostat)

        # Setup Thermostats
        thermostat = ET.Element('thermostat',
                                attrib={'mode': self.thermostyle})
        thermostat.append(tdamp)
        if 'pile' in self.thermostyle:
            pile_lambda = _add_textxml(ET.Element('pile_lambda'),
                                       str(self.pilelambda))
            thermostat.append(pile_lambda)
        timestep = _add_textxml(ET.Element('timestep',
                                           attrib={'units': 'femtosecond'}),
                                str(self.dt))
        dynamics = _add_Subelements(dynamics, [thermostat, timestep])
        motion.append(dynamics)

        # Fix COM
        if not self.fixcm:
            fixcom = _add_textxml(ET.Element('fixcom'), str(self.fixcm))
            motion.append(fixcom)

        system = ET.Element('system')
        system = _add_Subelements(system,
                                  [initialize, forces, ensemble, motion])

        # Add Simulation parameters
        simulation = ET.Element('simulation', attrib={'verbosity': 'high'})
        simulation = _add_Subelements(simulation,
                                      [output,
                                       total_steps,
                                       prng,
                                       ffsocket,
                                       system])
        tree = ET.ElementTree(simulation)
        if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
            ET.indent(tree)
        tree.write(self.workdir + self.ipifname, encoding='unicode',
                   xml_declaration=True)

# ========================================================================== #
    def create_ase_atom(self, pbc, nbeads_return):
        """
        """
        pref = self.workdir + self.prefix
        nmax = len(str(self.nbeads))
        if self.nbeads == 1:
            image = 0
            atoms = self._get_one_atoms(image, pref, pbc, nmax)
        elif nbeads_return == 1:
            image = self.rngint(0, self.nbeads-1)
            atoms = self._get_one_atoms(image, pref, pbc, nmax)
        else:
            allidx = np.linspace(0, self.nbeads,
                                 nbeads_return, dtype=int,
                                 endpoint=False)
            atoms = []
            for idx in allidx:
                atoms.append(self._get_one_atoms(idx, pref, pbc, nmax))
        return atoms

# ========================================================================== #
    def _get_one_atoms(self, idx, pref, pbc, nmax):
        """
        """
        file = pref + '.outpos_{i:0{j}d}.xyz'.format(i=idx, j=nmax)
        Z = read(file, index=-1).get_atomic_numbers()
        cell = self._read_cells(file)
        positions = read(file, index=-1).positions
        file = pref + '.outfor_{i:0{j}d}.xyz'.format(i=idx, j=nmax)
        forces = read(file, index=-1).positions
        file = pref + '.outvel_{i:0{j}d}.xyz'.format(i=idx, j=nmax)
        velocities = read(file, index=-1).positions
        velocities *= 1e-5 / fs  # To get back to ASE units

        atoms = Atoms(numbers=Z, cell=cell, positions=positions, pbc=pbc)
        atoms.set_velocities(velocities)
        calc = SPCalc(atoms, forces=forces)
        atoms.set_calculator(calc)
        return atoms.copy()

# ========================================================================== #
    def _read_cells(self, filename):
        """
        """
        with open(filename, 'r') as fd:
            for line in fd:
                if '# CELL(abcABC):' in line:
                    pass
                if '# CELL(abcABC):' in line:
                    cell = np.array(line.split()[2:8])
        return cell

# ========================================================================== #
    def get_temperature(self):
        """
        Return the temperature of the state
        """
        return self.temperature

# ========================================================================== #
    def get_thermostat_input(self, temp=None, press=None):
        """
        """
        if self.socketmode == 'inet':
            input_string = f"fix    1  all ipi {self.hostname} {self.rngnum}\n"
        else:
            input_string = "fix    1  all " + \
                           f"ipi {self.socketname} {self.rngnum} unix\n"
        return input_string

# ========================================================================== #
    def log_recap_state(self):
        """
        Function to return a string describing the state for the log
        """
        damp = self.damp
        if damp is None:
            damp = 100 * self.dt
        pdamp = self.pdamp
        if pdamp is None:
            pdamp = 2*damp

        if self.ispimd:
            msg = "Running Path-Integral Molecular Dynamics, " + \
                  "using Lammps with I-Pi\n"
        else:
            msg = "Running Molecular Dynamics, " + \
                  "using Lammps with I-Pi\n"
        msg += f"Langevin dynamics in the {self.ensemble} " + \
            "ensemble as implemented in IPI\n"
        if self.ispimd:
            msg += f"Number of beads :                     {self.nbeads}\n"
        msg += f"Temperature (in Kelvin) :             {self.temperature}\n"
        if self.ensemble != 'nvt':
            msg += f"Pressure (GPa) :                      {self.pressure}\n"
        msg += f"Number of MLMD equilibration steps :  {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :     {self.nsteps}\n"
        msg += f"Timestep (in fs) :                    {self.dt}\n"
        msg += f"Themostat damping parameter (in fs) : {damp}\n"
        if self.ensemble != 'nvt':
            msg += f"Barostat damping parameter (in fs) :     {pdamp}\n"
        msg += "\n"
        return msg

# ========================================================================== #
    def set_workdir(self, workdir):
        """
        """
        self.workdir = workdir
        self.ipiatomsfname = "ipi_atoms.xyz"
        self.ipifname = "ipi_input.xml"


if __name__ == '__main__':
    help(IpiState)
