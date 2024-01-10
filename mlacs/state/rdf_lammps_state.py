"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import run, PIPE

import numpy as np

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .state import StateManager
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import (get_general_input,
                                   get_log_input,
                                   get_traj_input,
                                   get_rdf_input,
                                   get_interaction_input,
                                   get_last_dump_input)


# ========================================================================== #
# ========================================================================== #
class RdfLammpsState(StateManager):
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
        as it is to test the mlip and compute physical properties, it is
        recommended to run a MLMD on dozen of ps

    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs. Default ``100`` steps.

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

    workdir : :class:`str` (optional)
        Working directory for the LAMMPS MLMD simulations.
        If ``None``, a LammpsMLMD directory is created

    rdffile : :class:`str` (optional)
        Name of the file for radial distribution function calculation.
        If ``None``, no file is created. Default ``None``.

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
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 msdfile=None,
                 rdffile=None,
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
                              msdfile,
                              rdffile,
                              workdir)

        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        self.atomsfname = "atoms.in"
        self.lammpsfname = "lammps_input.in"

        self._get_lammps_command()
        self.ispimd = False
        self.isrestart = False

        self.temperature = temperature
        self.langevin = langevin
        self.pressure = pressure
        self.damp = damp
        self.gjf = gjf
        self.pdamp = pdamp
        self.ptype = ptype
        self.qtb = qtb
        self.fd = fd
        self.n_f = n_f

        self.t_stop = t_stop
        self.p_stop = p_stop
        if self.p_stop is not None:
            if self.pressure is None:
                msg = "You need to put a pressure with p_stop"
                raise ValueError(msg)

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     replicate="1 1 1",
                     eq=False,
                     workdir=None):
        """
        Function to run the dynamics
        """
        if workdir is not None:
            self.workdir = workdir

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

        atoms = supercell.copy()

        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        if self.t_stop is None:
            temp = self.temperature
        else:
            if eq:
                temp = self.t_stop
            else:
                temp = self.rng.uniform(self.temperature, self.t_stop)

        if self.p_stop is None:
            press = self.pressure
        else:
            if eq:
                press = self.pressure
            else:
                press = self.rng.uniform(self.pressure, self.p_stop)

        if self.t_stop is not None:
            MaxwellBoltzmannDistribution(atoms,
                                         temperature_K=temp,
                                         rng=self.rng)
        write_lammps_data(self.workdir + self.atomsfname,
                          atoms,
                          velocities=True,
                          atom_style=atom_style)

        nsteps_eq = self.nsteps_eq
        nsteps = self.nsteps
        self.write_lammps_rdf(atoms,
                              atom_style,
                              replicate,
                              pair_style,
                              pair_coeff,
                              model_post,
                              nsteps_eq,
                              nsteps,
                              temp,
                              press)

        lammps_command = self.cmd + " -in " + self.lammpsfname + \
            " -sc out.lmp"
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=self.workdir,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            print(lmp_handle.returncode)
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
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp_serial"
        self.cmd = cmd

# ========================================================================== #
    def write_lammps_rdf(self,
                         atoms,
                         atom_style,
                         replicate,
                         pair_style,
                         pair_coeff,
                         model_post,
                         nsteps_eq,
                         nsteps,
                         temp,
                         press):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms)
        pbc = atoms.get_pbc()

        input_string = ""
        input_string += get_general_input(pbc,
                                          masses,
                                          charges,
                                          atom_style,
                                          replicate)
        input_string += get_interaction_input(pair_style,
                                              pair_coeff,
                                              model_post)
        if self.logfile is not None:
            input_string += get_log_input(self.loginterval, self.logfile)
        if self.trajfile is not None:
            input_string += get_traj_input(self.loginterval,
                                           self.trajfile,
                                           elem)
        input_string += self.get_equilibration_input(temp,
                                                     press)
        input_string += self.get_thermostat_input(temp, press)
        input_string += get_rdf_input(self.rdffile, nsteps)

        input_string += get_last_dump_input(self.workdir,
                                            elem,
                                            nsteps)
        input_string += f"run  {nsteps}"

        with open(self.workdir + "lammps_input.in", "w") as f:
            f.write(input_string)

# ========================================================================== #
    def get_equilibration_input(self, temp, press):
        """
        Function to equilibrate the system before computing physical qties
        Note that equilibration is also in NVT at the volume of the
        i-ème mlacs step corresponding structure.
        To be modified to the avereage vol
        """
        damp = self.damp
        nsteps_eq = self.nsteps_eq
        if self.damp is None:
            damp = "$(100*dt)"

        input_string = "#####################################\n"
        input_string += "#      Thermostat/Integrator\n"
        input_string += "#####################################\n"
        input_string += "# Equilibration run\n"
#        input_string += "variable seed_factory
#                         equal round(random(1,25000000,666)) \n"
#        input_string += f"velocity all create {temp}
#                          $((v_seed_factory)) dist gaussian \n"
        input_string += f"velocity all create {temp} "
        input_string += f"{self.rng.integers(999999)} dist gaussian \n"
        input_string += "timestep      {0}\n".format(self.dt / 1000)
        input_string += f"fix f1 all nvt temp {temp} {temp} {damp}\n"
        input_string += f"run   {nsteps_eq}\n"
        input_string += "unfix f1\n"
        return input_string

# ========================================================================== #
    def get_thermostat_input(self, temp, press):
        """
        Function to write the thermostat of the mlmd run
        """
        damp = self.damp
        if self.damp is None:
            damp = "$(100*dt)"

        pdamp = self.pdamp
        if self.pdamp is None:
            pdamp = "$(1000*dt)"

        input_string = "# Average Run\n"
        if self.pressure is None:
            if self.langevin:
                # Langevin part
                input_string += f"fix  f1 all langevin {temp} " + \
                                f"{temp}  {damp} " + \
                                f"{self.rng.integers(999999)} " + \
                                f"gjf {self.gjf} zero yes\n"
                # Integration part
                input_string += "fix   f2 all nve\n"
            else:
                input_string += f"fix  f1 all nvt temp {temp} " + \
                                f"{temp}  {damp}\n"
        else:
            if self.langevin:
                # Langevin part
                input_string += f"fix  f1 all langevin {temp} " + \
                                f"{temp}  {damp} " + \
                                f"{self.rng.integers(999999)} " + \
                                f"gjf {self.gjf} zero yes\n"
                # Barostat part
                input_string += f"fix    f2 all nph  {self.ptype} " + \
                                f"{press*10000} " + \
                                f"{press*10000} {pdamp}\n"
            else:
                input_string += f"fix  f1 all npt temp {temp} " + \
                                f"{temp}  {damp} {self.ptype} " + \
                                f"{press*10000} " + \
                                f"{press*10000} {pdamp}\n"
        if self.fixcm:
            input_string += "fix    fcm all recenter INIT INIT INIT\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
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
            pdamp = 1000 * self.dt

        if self.pressure is None:
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
        msg += f"Themostat damping parameter (in fs) :   {self.dt}\n"
        if self.pressure is not None:
            msg += f"Barostat damping parameter (in fs) :    {pdamp}\n"
        msg += "\n"
        return msg

# ========================================================================== #
    def set_workdir(self, workdir):
        """
        """
        self.workdir = workdir
