"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os

import numpy as np
from ase.io import read

from .lammps_state import BaseLammpsState
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import LammpsBlockInput


# ========================================================================== #
# ========================================================================== #
class PimdLammpsState(BaseLammpsState):
    """
    Class to manage PIMD simulations with LAMMPS


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

    nbeads : :class:`int` (optional)
        Number of beads used in the PIMD quantum polymer. Default ``1``,
        which revert to classical sampling.

    nprocs : :class:`int` (optional)
        Total number of process used to run LAMMPS.
        Have to be a multiple of the number of beads.
        If nprocs > than nbeads, each replica will be parallelized using the
        partition scheme of LAMMPS.
        Per default it assumes that nprocs = nbeads

    integrator : :class:`str` (optional)
        Type of integrator to use. Can be baoab or obabo. Default ``'baoab'``

    fmmode : :class:`str` (optional)
        Type of fictitious mass preconditioning. Default ``'physical'``

    fmass: :class:`float` or None (optional)
        Scaling factor for the fictitious masses of the beads. Default ``None``
        which set it to the number of beads.

    scale: :class:`int` (optional)
        scaling factor for the damping for non-centroid modes. Default 1.0

    barostat: :class:`int` (optional)
        Type of barostat used. Default ``'BZP'``

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

    msdfile : :class:`str` (optional)
        Name of the file for diffusion coefficient calculation.
        If ``None``, no file is created. Default ``None``.

    rdffile : :class:`str` (optional)
        Name of the file for radial distribution function calculation.
        If ``None``, no file is created. Default ``None``.

    rng : RNG object (optional)
        Rng object to be used with the Langevin thermostat.
        Default correspond to :class:`numpy.random.default_rng()`

    init_momenta : :class:`numpy.ndarray` (optional)
        Gives the (Nat, 3) shaped momenta array that will be used
        to initialize momenta when using
        the `initialize_momenta` function.
        If the default ``None`` is set, momenta are initialized with a
        Maxwell Boltzmann distribution.

    """
    def __init__(self, temperature, pressure=None, t_stop=None, p_stop=None,
                 damp=None, pdamp=None, ptype="iso", dt=1.0, fixcm=True,
                 rng=None, init_momenta=None, integrator="baoab",
                 fmmode="physical", fmass=None, scale=1.0, barostat="BZP",
                 nprocs=None, nbeads=1,
                 nsteps=1000, nsteps_eq=100, logfile=None, trajfile=None,
                 loginterval=50, blocks=None, **kwargs):

        super().__init__(nsteps, nsteps_eq, logfile, trajfile, loginterval,
                         blocks, **kwargs)

        self.temperature = temperature
        self.pressure = pressure
        self.t_stop = t_stop
        self.p_stop = p_stop
        self.damp = damp
        self.pdamp = pdamp
        self.ptype = ptype
        self.dt = dt
        self.fixcm = fixcm
        self.rng = rng
        self.init_momenta = init_momenta
        self.integrator = integrator
        self.fmmode = fmmode
        self.fmass = fmass
        self.scale = scale
        self.barostat = barostat
        self.nprocs = nprocs
        self.nbeads = nbeads
        self.ispimd = True

        if self.rng is None:
            self.rng = np.random.default_rng()
        if self.trajfile is not None and self.nbeads > 1:
            self.trajfile = f"{self.trajfile}" + "_${ibead}"
        if self.fmass is None:
            self.fmass = self.nbeads
        if self.damp is None:
            self.damp = "$(100*dt)"
        if self.pdamp is None:
            self.pdamp = "$(1000*dt)"
        if self.p_stop is not None:
            if self.pressure is None:
                msg = "You need to put a pressure with p_stop"
                raise ValueError(msg)
        if self.nprocs is not None:
            if self.nprocs % self.nbeads != 0:
                msg = "The number of processor needs to be a multiple " + \
                    "of the number of beads"
                raise ValueError(msg)

# ========================================================================== #
    def _get_block_init(self, atoms, atom_style):
        """

        """
        pbc = atoms.get_pbc()
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        block = LammpsBlockInput("init", "Initialization")
        block("map", "atom_modify map yes")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", f"atom_style {atom_style}")
        block("read_data", "read_data atoms.in")
        block("variable", f"variable ibead uloop {self.nbeads} pad")
        for i, mass in enumerate(masses):
            block(f"mass{i}", f"mass {i+1}  {mass}")
        return block

# ========================================================================== #
    def _get_block_thermostat(self, eq):
        """
        Function to write the thermostat of the mlmd run
        """
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
            press = self.rng.uniform(self.pressure, self.p_stop)
        seed = self.rng.integers(1, 999999)

        block = LammpsBlockInput("thermostat", "Thermostat")
        fix = "fix f1 all pimd/langevin "
        # The temperature
        fix += f"temp {temp} "
        # The integrator
        fix += f"integrator {self.integrator} "
        # Then the thermostat
        fix += f"thermostat PILE_L {seed} "
        # The damping parameter
        fix += f"tau  {self.damp} "
        # Scaling factor of non-centroid mode
        fix += f"scale {self.scale} "
        # the fmmode
        fix += f"fmmode {self.fmmode} "
        # Scaling factor of the mass
        fix += f"fmass {self.fmass} "
        if self.fixcm:
            fix += "fixcom yes "
        if self.pressure is not None:
            # Tell that it's NPT
            fix += "ensemble npt "
            # value of pressure
            if self.ptype == "iso":
                fix += f"iso {press} "
            elif self.ptype == "aniso":
                fix += f"aniso {press} "
            fix += f"barostat {self.barostat} "
            fix += f"taup {self.pdamp} "
        else:
            fix += "ensemble nvt "
        block("fix", fix)
        return block

# ========================================================================== #
    def _get_atoms_results(self, initial_charges):
        """

        """
        fname = "configurations.out"
        ndigit = len(str(self.nbeads))
        fname = f"{fname}_{1:0{ndigit}d}"
        atoms = read(self.path / fname)
        if initial_charges is not None:
            atoms.set_initial_charges(initial_charges)
        return atoms

# ========================================================================== #
    def _get_block_lastdump(self, atoms, eq):
        """

        """
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)
        block = LammpsBlockInput("lastdump", "Dump last configuration")
        txt = "dump last all custom 1 configurations.out_${ibead} " + \
              "id type xu yu zu vx vy vz fx fy fz element"
        block("dump", txt)
        txt = "dump_modify last element " + " ".join([p for p in el])
        block("dump_modify1", txt)
        block("run_dump", "run 0")
        return block

# ========================================================================== #
    def _get_lammps_command(self):
        '''
        Function to load the batch command to run LAMMPS
        '''
        envvar = "ASE_LAMMPSRUN_COMMAND"
        cmd = os.environ.get(envvar)
        if cmd is None:
            cmd = "lmp_serial"
        n1 = self.nbeads
        if self.nprocs is not None:
            n2 = self.nprocs // self.nbeads
        else:
            n2 = 1
        return f"{cmd} -partition {n1}x{n2} -in {self.lammpsfname} -sc out.lmp"

# ========================================================================== #
    def log_recap_state(self):
        """

        """
        damp = self.damp
        if damp is None:
            damp = 100 * self.dt
        pdamp = self.pdamp
        if pdamp is None:
            pdamp = 1000 * self.dt

        if self.pressure is None:
            msg = "NVT Path-Integral dynamics as implemented in LAMMPS\n"
        else:
            msg = "NPT Path-Integral dynamics as implemented in LAMMPS\n"
        msg += f"Temperature (in Kelvin)                 {self.temperature}\n"
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
