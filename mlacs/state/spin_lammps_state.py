"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from .lammps_state import LammpsState
from ..utilities import get_elements_Z_and_masses
from ..utilities.io_lammps import (LammpsBlockInput,
                                   write_atoms_lammps_spin_style)


# ========================================================================== #
# ========================================================================== #
class SpinLammpsState(LammpsState):
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
        LammpsState.__init__(self,
                             temperature=temperature,
                             pressure=pressure,
                             t_stop=t_stop,
                             p_stop=p_stop,
                             damp=damp,
                             pdamp=pdamp,
                             ptype=ptype,
                             dt=dt,
                             nsteps=nsteps,
                             nsteps_eq=nsteps_eq,
                             fixcm=fixcm,
                             logfile=logfile,
                             trajfile=trajfile,
                             loginterval=loginterval,
                             rng=rng,
                             init_momenta=init_momenta,
                             workdir=workdir)

        self.ispimd = False
        if self.trajfile is not None and self.nbeads > 1:
            self.trajfile += "_${ibead}"

# ========================================================================== #
    def _get_block_init(self, atoms, atom_style):
        """

        """
        pbc = atoms.get_pbc()
        pbc = "{0} {1} {2}".format(*tuple("sp"[int(x)] for x in pbc))
        el, Z, masses, charges = get_elements_Z_and_masses(atoms)

        block = LammpsBlockInput("init", "Initialization")
        block("map", "atom_modify map array")
        block("units", "units metal")
        block("boundary", f"boundary {pbc}")
        block("atom_style", "atom_style spin")
        block("read_data", "read_data atoms.in")
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
        if self.qtb:
            qtbseed = self.rng.integers(1, 99999999)
        if self.langevin:
            langevinseed = self.rng.integers(1, 9999999)
        spinseed = self.rng.integers(1, 9999999)

        block = LammpsBlockInput("thermostat", "Thermostat")
        block("timestep", f"timestep {self.dt / 1000}")
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
        txt = f"fix fspin all langevin/spin {temp} {self.damp} {spinseed}"
        block("nvt_spin", txt)
        block("nve_spin", "fix fspin2 all nve/spin lattice moving")
        if self.fixcm:
            block("cm", "fix fcm all recenter INIT INIT INIT")
        return block

# ========================================================================== #
    def _write_lammps_atoms(self, atoms, atom_style):
        """

        """
        spins = atoms.get_array("spins")
        with open(self.workdir / "atoms.in", "w") as fd:
            write_atoms_lammps_spin_style(fd, atoms, spins)
