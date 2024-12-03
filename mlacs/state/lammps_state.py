"""
// Copyright (C) 2022-2024 MLACS group (AC, RB, ON, PR)
// This file is distributed under the terms of the
// GNU General Public License, see LICENSE.md
// or http://www.gnu.org/copyleft/gpl.txt .
// For the initials of contributors, see CONTRIBUTORS.md
"""

from subprocess import run, PIPE
from abc import abstractmethod

import numpy as np

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.data import chemical_symbols, atomic_masses

from .state import StateManager
from ..core.manager import Manager
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import (LammpsInput,
                                   EmptyLammpsBlockInput,
                                   LammpsBlockInput,
                                   get_lammps_command)


class BaseLammpsState(StateManager):
    """
    Base class to perform simulations with LAMMPS.
    """
    def __init__(self, nsteps, nsteps_eq, logfile, trajfile, loginterval=50,
                 lammpsfname=None, blocks=None, neti=False, eq_mass_md=False,
                 **kwargs):

        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         **kwargs)

        self.eq_mass_md = eq_mass_md
        self.ispimd = False
        self.isrestart = False
        self.nbeads = 1  # Dummy nbeads to help

        self.atomsfname = "atoms.in"
        self.lammpsfname = lammpsfname
        if self.lammpsfname is None:
            self.lammpsfname = "lammps_input.in"
        self._myblock = blocks
        # key word to adapt functions from BaseLammsState for NETI
        self.neti = neti

        self.info_dynamics = dict()
        if isinstance(blocks, list):
            self._myblock = blocks[0]
            if len(blocks) != 1:
                for block in blocks[1:]:
                    self._myblock.extend(block)

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     elements=None):
        """
        Function to run the dynamics
        """
        atoms = supercell.copy()

        if elements is None:
            el, Z, masses, charges = get_elements_Z_and_masses(atoms)
            elements = el

        initial_charges = atoms.get_initial_charges()

        blocks = self._get_block_inputs(atoms, pair_style, pair_coeff,
                                        model_post, atom_style, eq, elements)
        if self.neti is False:
            txt = "Lammps input to run MlMD created by MLACS"
            lmp_input = LammpsInput(txt)
        else:
            txt = "Lammps input to run a NETI created by MLACS"
            lmp_input = LammpsInput(txt)
        for block in blocks:
            lmp_input(block.name, block)

        with open(self.subsubdir / self.lammpsfname, "w") as fd:
            fd.write(str(lmp_input))

        self._write_lammps_atoms(atoms, atom_style, elements)

        lmp_cmd = self._get_lammps_command()
        lmp_handle = run(lmp_cmd,
                         shell=True,
                         cwd=str(self.subsubdir),
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)
        if self.neti is False:
            atoms = self._get_atoms_results(initial_charges)

        # Set the info of atoms
        atoms.info['info_state'] = self.info_dynamics

        return atoms.copy()

# ========================================================================== #
    @Manager.exec_from_subsubdir
    def _write_lammps_atoms(self, atoms, atom_style, elements):
        """

        """
        write_lammps_data(str(self.subsubdir / self.atomsfname),
                          atoms,
                          velocities=True,
                          atom_style=atom_style,
                          specorder=elements.tolist())


# ========================================================================== #
    def _get_block_inputs(self, atoms, pair_style, pair_coeff, model_post,
                          atom_style, eq, elements):
        """

        """
        masses = [atomic_masses[chemical_symbols.index(element)]
                  for element in elements]

        pbc = atoms.get_pbc()
        if self.eq_mass_md:
            masses = np.ones(np.shape(masses))

        blocks = []
        blocks.append(self._get_block_init(atom_style, pbc, elements, masses))
        blocks.append(self._get_block_interactions(pair_style, pair_coeff,
                                                   model_post, atom_style,
                                                   atoms))
        blocks.append(self._get_block_thermostat(eq))
        if self.logfile is not None:
            blocks.append(self._get_block_log())
        if self.trajfile is not None:
            blocks.append(self._get_block_traj(elements))
        if isinstance(self._get_block_custom(), list):
            blocks.extend(self._get_block_custom())
        else:
            blocks.append(self._get_block_custom())
        blocks.append(self._get_block_run(eq))
        if self.neti is False:
            blocks.append(self._get_block_lastdump(elements, eq))
        return blocks

# ========================================================================== #
    def _get_block_init(self, atom_style, pbc, el, masses):
        """

        """
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))

        block = LammpsBlockInput("init", "Initialization")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", f"atom_style {atom_style}")
        block("read_data", f"read_data {self.atomsfname}")
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
        for iel, e in enumerate(el):
            block("group", f"group {e} type {iel+1}")
        return block

# ========================================================================== #
    def _get_block_run(self, eq):
        """

        """
        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps
        block = LammpsBlockInput("run")
        block("run", f"run {nsteps}")
        return block

# ========================================================================== #
    def _get_block_interactions(self, pair_style, pair_coeff, model_post,
                                atom_style, atoms):
        """

        """
        # Write lammps input
        block = LammpsBlockInput("interaction", "Interaction")
        block("pair_style", f"pair_style {pair_style}")
        for i, pair in enumerate(pair_coeff):
            block(f"pair_coeff{i}", f"pair_coeff {pair}")
        if model_post is not None:
            for i, model in enumerate(model_post):
                block(f"model{i}", f"{model}")
        return block

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        return None

# ========================================================================== #
    def _get_block_log(self):
        """

        """
        block = LammpsBlockInput("log", "Logging")
        variables = ["t equal step", "mytemp equal temp",
                     "mype equal pe", "myke equal ke", "myetot equal etotal",
                     "mypress equal press/10000", "vol equal (lx*ly*lz)"]
        for i, var in enumerate(variables):
            block(f"variable{i}", f"variable {var}")
        txt = f"fix mylog all print {self.loginterval} " + \
              '"$t ${mytemp} ${vol} ${myetot} ${mype} ${myke} ${mypress}" ' + \
              f"append {self.logfile} title " + \
              '"# Step Temp Vol Etot Epot Ekin Press"'
        block("fix", txt)
        return block

# ========================================================================== #
    def _get_block_lastdump(self, el, eq):
        """

        """
        block = LammpsBlockInput("lastdump", "Dump last configuration")
        txt = "dump last all custom 1 configurations.out " + \
              "id type xu yu zu vx vy vz fx fy fz element"
        block("dump", txt)
        txt = "dump_modify last element " + " ".join([p for p in el])
        block("dump_modify1", txt)
        block("run_dump", "run 0")
        return block

# ========================================================================== #
    def _get_block_traj(self, el):
        """

        """
        block = LammpsBlockInput("traj", "Dumping trajectory")
        txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} " + \
              "id type xu yu zu vx vy vz fx fy fz "
        txt += "element"
        block("dump", txt)
        block("dump_modify1", "dump_modify dum1 append yes")
        txt = "dump_modify dum1 element " + " ".join([p for p in el])
        block("dump_modify2", txt)
        return block

# ========================================================================== #
    def _get_block_custom(self):
        """

        """
        if isinstance(self._myblock, LammpsBlockInput):
            return self._myblock
        else:
            return EmptyLammpsBlockInput("empty_custom")

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """

        """
        atoms = read(self.subsubdir / "configurations.out")
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        return atoms

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the bash command to run LAMMPS
        '''
        cmd = get_lammps_command()
        return f"{cmd} -in {self.lammpsfname} -sc out.lmp"

# ========================================================================== #
    def initialize_momenta(self, atoms):
        """

        """
        pass

# ========================================================================== #
    @abstractmethod
    def log_recap_state(self):
        pass


# ========================================================================== #
# ========================================================================== #
class LammpsState(BaseLammpsState):
    """
    Class to perform NVT or NPT simulations with LAMMPS.

    Parameters
    ----------
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.

    pressure: :class:`float` or ``None`` (optional)
        Pressure of the simulation, in GPa.
        If ``None``, no barostat is applied and
        the simulation is in the NVT ensemble. Default ``None``

    t_stop: :class:`float` or ``None`` (optional)
        When this input is not ``None``, the temperature of
        the molecular dynamics simulations is randomly chosen
        in a range between `temperature` and `t_stop`.
        Default ``None``

    p_stop: :class:`float` or ``None`` (optional)
        When this input is not ``None``, the pressure of
        the molecular dynamics simulations is randomly chosen
        in a range between `pressure` and `p_stop`.
        Naturally, the `pressure` input has to be set.
        Default ``None``

    damp: :class:`float` or ``None`` (optional)
        The damping value for the thermostat.
        The default gives a sensible value of a hundred times the
        timestep.

    langevin: :class:`Bool` (optional)
        If ``True``, a Langevin thermostat is used for the thermostat.
        Default ``True``

    gjf: ``no`` or ``vfull`` or ``vhalf`` (optional)
        Whether to use the Gronbech-Jensen/Farago integrator
        for the Langevin dynamics. Only apply if langevin is ``True``.
        Default ``vhalf``.

    qtb: :clas::`Bool` (optional)
        Whether to use a quantum thermal bath to approximate quantum effects.
        If True, it override the langevin and gjf inputs.
        Default False

    fd: :class:`float` (optional)
        The frequency cutoff for the qtb thermostat. Should be around
        2~3 times the Debye frequency. In THz.
        Default 200 THz.

    n_f: :class:`int` (optional)
        Frequency grid size for the qtb thermostat.
        Default 100.

    pdamp: :class:`float` or ``None`` (optional)
        Damping parameter for the barostat.
        If ``None``, apply a damping parameter of
        1000 times the timestep of the simulation. Default ``None``

    ptype: ``iso`` or ``aniso`` (optional)
        Handle the type of pressure applied. Default ``iso``

    twodimensional: :class:`bool` (optional)
        If set to ``True`` and pressure is not ``None``, set the pressure
        only on the x and y axis. Pressure along `x` and `y` axis can
        still be coupled by setting ``ptype`` to `iso`.
        default ``False``

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.

    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    init_momenta : :class:`numpy.ndarray` (optional)
        Gives the (Nat, 3) shaped momenta array that will be used
        to initialize momenta when using
        the `initialize_momenta` function.
        If the default ``None`` is set, momenta are initialized with a
        Maxwell Boltzmann distribution.

    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

    eq_mass_md : :class:`Bool` (optional)
        If all atoms have the same mass for the MD. Default ``False``
        If True, make sure your MLIP is also correctly parametrized

    logfile : :class:`str` (optional)
        Name of the file for logging the MLMD trajectory.
        If ``None``, no log file is created. Default ``None``.

    trajfile : :class:`str` (optional)
        Name of the file for saving the MLMD trajectory.
        If ``None``, no traj file is created. Default ``None``.

    loginterval : :class:`int` (optional)
        Number of steps between MLMD logging. Default ``50``.

    blocks : :class:`LammpsBlockInput` or :class:`list` (optional)
        Custom block input class. Can be a list of blocks.
        If ``None``, nothing is added in the input. Default ``None``.

    Examples
    --------

    >>> from mlacs.state import LammpsState
    >>>
    >>> state = LammpsState(temperature=300, pressure=None) #NVT
    >>> state = LammpsState(temperature=300, pressure=0)    #NPT
    >>> state.run_dynamics(atoms, mlip.pair_style, mlip.pair_coeff)
    """
    def __init__(self,
                 temperature,
                 pressure=None,
                 t_stop=None,
                 p_stop=None,
                 damp=None,
                 langevin=True,
                 gjf="vhalf",
                 qtb=False,
                 fd=200,
                 n_f=100,
                 pdamp=None,
                 ptype="iso",
                 twodimensional=False,
                 dt=1.5,
                 fixcm=True,
                 rng=None,
                 init_momenta=None,
                 nsteps=1000,
                 nsteps_eq=100,
                 eq_mass_md=False,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 blocks=None,
                 folder='Trajectory',
                 **kwargs):

        kwargs.setdefault('prefix', folder)  # To keep previous behaviour

        super().__init__(nsteps, nsteps_eq, logfile, trajfile,
                         loginterval=loginterval, blocks=blocks,
                         eq_mass_md=eq_mass_md, folder=folder, **kwargs)

        self.temperature = temperature
        self.pressure = pressure
        self.t_stop = t_stop
        self.p_stop = p_stop
        self.damp = damp
        self.langevin = langevin
        self.gjf = gjf
        self.qtb = qtb
        self.fd = fd
        self.n_f = n_f
        self.pdamp = pdamp
        self.ptype = ptype
        self.dt = dt
        self.fixcm = fixcm
        self.rng = rng
        self.init_momenta = init_momenta
        self.twodimensional = twodimensional

        if self.rng is None:
            self.rng = np.random.default_rng()
        if self.damp is None:
            self.damp = "$(100*dt)"
        if self.pdamp is None:
            self.pdamp = "$(1000*dt)"

        if self.p_stop is not None:
            if self.pressure is None:
                msg = "You need to put a pressure with p_stop"
                raise ValueError(msg)

        self._make_info_dynamics()

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """

        """
        if self.t_stop is None:
            temp = self.temperature
        else:
            tmp_temp = np.sort([self.temperature, self.t_stop])
            if eq:
                temp = np.max(tmp_temp)
            else:
                temp = self.rng.uniform(*tmp_temp)
            self.info_dynamics["temperature"] = temp
        if self.p_stop is None:
            press = self.pressure
        else:
            press = self.rng.uniform(self.pressure, self.p_stop)
            self.info_dynamics["pressure"] = press
        if self.qtb:
            qtbseed = self.rng.integers(1, 99999999)
        if self.langevin:
            langevinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Thermostat")
        block("timestep", f"timestep {self.dt / 1000}")

        # If we are using Langevin, we want to remove the random part
        # of the forces
        if self.langevin:
            block("rmv_langevin", "fix ff all store/force")

        if self.pressure is None:
            if self.qtb:
                block("nve", "fix f1 all nve")
                txt = f"fix f2 all qtb temp {temp} damp {self.damp} " + \
                      f"f_max {self.fd} N_f {self.n_f} seed {qtbseed}"
                block("qtb", txt)
            elif self.langevin:
                txt = f"fix f1 all langevin {temp} {temp} {self.damp} " + \
                      f"{langevinseed} gjf {self.gjf} zero yes"
                block("langevin", txt)
                block("nve", "fix f2 all nve")
            else:
                block("nvt", f"fix f1 all nvt temp {temp} {temp} {self.damp}")
        else:
            if self.qtb:
                txt = f"fix f1 all nph {self.ptype} " + \
                      f"{press*10000} {press*10000} {self.pdamp}"
                block("nph", txt)
                txt = f"fix f1 all qtb temp {temp} damp {self.damp}" + \
                      f"f_max {self.fd} N_f {self.n_f} seed {qtbseed}"
                block("qtb", txt)
            elif self.langevin:
                txt = f"fix f1 all langevin {temp} {temp} {self.damp} " + \
                      f"{langevinseed} gjf {self.gjf} zero yes"
                block("langevin", txt)
                ptxt = f"{press*10000} {press*10000} {self.pdamp}"
                txt = "fix f2 all nph "
                if self.twodimensional:
                    txt += f"x {ptxt} y {ptxt} "
                    if self.ptype == "iso":
                        txt += "couple xy "
                else:
                    txt += f"{self.ptype} {ptxt}"
                block("nph", txt)
            else:
                txt = f"fix f1 all npt temp {temp} {temp} {self.damp} " + \
                      f"{self.ptype} {press*10000} {press*10000} {self.pdamp}"
                block("npt", txt)
        if self.fixcm:
            block("cm", "fix fcm all recenter INIT INIT INIT")
        return block

# ========================================================================== #
    def _make_info_dynamics(self):

        # NVT, NPT, no (uVT, uPT, NVE) yet
        ensemble = ["X", "X", "X"]

        # N or mu
        ensemble[0] = "N"

        # V or P
        ensemble[1] = "V"
        pressure = None
        if self.pressure is not None:
            ensemble[1] = "P"
            if self.p_stop is None:
                pressure = self.pressure

        # T or E
        if self.temperature is None:  # NVE
            raise NotImplementedError
        ensemble[2] = "T"
        temperature = self.temperature

        # NVT, NPT, no (uVT, uPT, NVE) yet
        self.ensemble = ''.join(ensemble)
        self.info_dynamics = dict(ensemble=self.ensemble,
                                  temperature=temperature,
                                  pressure=pressure)

# ========================================================================== #
    def _get_block_traj(self, el):
        """

        """
        block = LammpsBlockInput("traj", "Dumping trajectory")
        txt = f"dump dum1 all custom {self.loginterval} {self.trajfile} " + \
              "id type xu yu zu vx vy vz fx fy fz "
        if self.langevin:
            txt += "f_ff[1] f_ff[2] f_ff[3] "
        txt += "element"
        block("dump", txt)
        block("dump_modify1", "dump_modify dum1 append yes")
        txt = "dump_modify dum1 element " + " ".join([p for p in el])
        block("dump_modify2", txt)
        return block

# ========================================================================== #
    def initialize_momenta(self, atoms):
        """
        """
        if self.init_momenta is None:
            MaxwellBoltzmannDistribution(atoms,
                                         temperature_K=self.temperature,
                                         rng=self.rng)
        else:
            atoms.set_momenta(self.init_momenta)

        atoms.info['info_state'] = self.info_dynamics

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
            pdamp = 1000 * self.dt

        if self.temperature is None and self.pressure is None:
            msg = "Geometry optimization as implemented in LAMMPS\n"
        elif self.pressure is None:
            msg = "NVT dynamics as implemented in LAMMPS\n"
        else:
            msg = "NPT dynamics as implemented in LAMMPS\n"
        msg += f"Temperature (in Kelvin)                 {self.temperature}\n"
        if self.langevin:
            msg += "A Langevin thermostat is used\n"
        if self.pressure is not None:
            msg += f"Pressure (GPa)                          {self.pressure}\n"
        msg += f"Number of MLMD equilibration steps :    {self.nsteps_eq}\n"
        msg += f"Number of MLMD production steps :       {self.nsteps}\n"
        msg += f"Timestep (in fs) :                      {self.dt}\n"
        if self.temperature is not None:
            msg += f"Themostat damping parameter (in fs) :   {damp}\n"
            if self.pressure is not None:
                msg += f"Barostat damping parameter (in fs) :    {pdamp}\n"
        msg += "\n"
        return msg
