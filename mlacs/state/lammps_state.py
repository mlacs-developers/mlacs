"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from pathlib import Path
from subprocess import run, PIPE

import numpy as np

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .state import StateManager
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import (LammpsInput,
                                   EmptyLammpsBlockInput,
                                   LammpsBlockInput)


# ========================================================================== #
# ========================================================================== #
class LammpsState(StateManager):
    """
    Class to manage States with LAMMPS

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

    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.

    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs. Default ``1000`` steps.

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.

    blocks : :class:`LammpsBlockInput` or :class:`list` (optional)
        Custom block input class. Can be a list of blocks.
        If ``None``, nothing is added in the input. Default ``None``.

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
        Gives the (Nat, 3) shaped momenta array that will be used
        to initialize momenta when using
        the `initialize_momenta` function.
        If the default ``None`` is set, momenta are initialized with a
        Maxwell Boltzmann distribution.

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created
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
                 dt=1.5,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 blocks=None,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 rng=None,
                 init_momenta=None,
                 workdir=None):
        StateManager.__init__(self,
                              dt,
                              nsteps,
                              nsteps_eq,
                              fixcm,
                              logfile,
                              trajfile,
                              loginterval,
                              workdir)

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.init_momenta = init_momenta

        self.atomsfname = "atoms.in"
        self.lammpsfname = "lammps_input.in"

        self.ispimd = False
        self.isrestart = False

        self.temperature = temperature
        self.langevin = langevin
        self.pressure = pressure
        self.gjf = gjf
        self.ptype = ptype
        self.qtb = qtb
        self.fd = fd
        self.n_f = n_f
        self.nbeads = 1  # Dummy nbeads to help
        self.damp = damp
        self.pdamp = pdamp
        if self.damp is None:
            self.damp = "$(100*dt)"
        if self.pdamp is None:
            self.pdamp = "$(1000*dt)"

        self.t_stop = t_stop
        self.p_stop = p_stop
        if self.p_stop is not None:
            if self.pressure is None:
                msg = "You need to put a pressure with p_stop"
                raise ValueError(msg)

        self.myblock = blocks
        if isinstance(blocks, list):
            self.myblock = blocks[0]
            if len(blocks) != 1:
                for block in blocks[1:]:
                    self.myblock.extend(block)

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False):
        """
        Function to run the dynamics
        """
        atoms = supercell.copy()
        initial_charges = atoms.get_initial_charges()
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        self.workdir.mkdir(exist_ok=True, parents=True)

        blocks = self._get_block_inputs(atoms, pair_style, pair_coeff,
                                        model_post, atom_style, eq)
        lmp_input = LammpsInput("Lammps input to run MlMD created by MLACS")
        for block in blocks:
            lmp_input(block.name, block)

        with open(self.workdir / self.lammpsfname, "w") as fd:
            fd.write(str(lmp_input))

        self._write_lammps_atoms(atoms, atom_style)

        lmp_cmd = self._get_lammps_command()
        lmp_handle = run(lmp_cmd,
                         shell=True,
                         cwd=self.workdir,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

        atoms = self._get_atoms_results(initial_charges)
        return atoms.copy()

# ========================================================================== #
    def _write_lammps_atoms(self, atoms, atom_style):
        """

        """
        write_lammps_data(self.workdir / self.atomsfname,
                          atoms,
                          velocities=True,
                          atom_style=atom_style)

# ========================================================================== #
    def _get_block_inputs(self, atoms, pair_style, pair_coeff, model_post,
                          atom_style, eq):
        """

        """
        blocks = []
        blocks.append(self._get_block_init(atoms, atom_style))
        blocks.append(self._get_block_interactions(pair_style, pair_coeff,
                                                   model_post, atom_style))
        blocks.append(self._get_block_thermostat(eq))
        if self.logfile is not None:
            blocks.append(self._get_block_log())
        if self.trajfile is not None:
            blocks.append(self._get_block_traj(atoms))
        blocks.append(self._get_block_custom())
        blocks.append(self._get_block_run(eq))
        blocks.append(self._get_block_lastdump(atoms, eq))
        return blocks

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
        block("read_data", f"read_data {self.atomsfname}")
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
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
                                atom_style):
        """

        """
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
        """

        """

        if self.t_stop is None:
            temp = self.temperature
        else:
            if eq:
                temp = self.t_stop
            else:
                tmp_temp = np.sort([self.temperature, self.t_stop])
                temp = self.rng.uniform(*tmp_temp)
        if self.p_stop is None:
            press = self.pressure
        else:
            press = self.rng.uniform(self.pressure, self.p_stop)
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
                txt = f"fix f2 all nph {self.ptype} " + \
                      f"{press*10000} {press*10000} {self.pdamp}"
                block("nph", txt)
            else:
                txt = f"fix f1 all npt temp {temp} {temp} {self.damp} " + \
                      f"{self.ptype} {press*10000} {press*10000} {self.pdamp}"
                block("npt", txt)
        if self.fixcm:
            block("cm", "fix fcm all recenter INIT INIT INIT")
        return block

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
    def _get_block_lastdump(self, atoms, eq):
        """

        """
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)
        block = LammpsBlockInput("lastdump", "Dump last configuration")
        txt = "dump last all custom 1 configurations.out " + \
              "id type xu yu zu vx vy vz fx fy fz element"
        block("dump", txt)
        txt = "dump_modify last element " + " ".join([p for p in el])
        block("dump_modify1", txt)
        block("run_dump", "run 0")
        return block

# ========================================================================== #
    def _get_block_traj(self, atoms):
        """

        """
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)
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
    def _get_block_custom(self):
        """

        """
        if isinstance(self.myblock, LammpsBlockInput):
            return self.myblock
        else:
            return EmptyLammpsBlockInput("empty_custom")

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """

        """
        atoms = read(self.workdir / "configurations.out")
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        return atoms

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

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp_serial"
        return f"{cmd} -in {self.lammpsfname} -sc out.lmp"

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

# ========================================================================== #
    def set_workdir(self, workdir):
        """
        """
        self.workdir = Path(workdir).absolute()
