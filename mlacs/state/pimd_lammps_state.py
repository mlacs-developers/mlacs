"""
// (c) 2023 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
import os
from subprocess import run, PIPE

from ase.io import read
from ase.io.lammpsdata import write_lammps_data
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .lammps_state import LammpsState
from ..utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class PimdLammpsState(LammpsState):
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
                 nbeads=1,
                 integrator="baoab",
                 fmmode="physical",
                 fmass=None,
                 scale=1.0,
                 barostat="BZP",
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

        self.integrator = integrator
        self.fmmode = fmmode
        self.scale = scale
        self.barostat = barostat
        self.nbeads = nbeads
        if fmass is None:
            self.fmass = nbeads
        else:
            self.fmass = fmass
        self.ispimd = True
        if self.trajfile is not None and self.nbeads > 1:
            self.trajfile += "_${ibead}"

# ========================================================================== #
    def run_dynamics(self,
                     supercell,
                     pair_style,
                     pair_coeff,
                     model_post=None,
                     atom_style="atomic",
                     eq=False,
                     nbeads_return=1):
        """
        Function to run the dynamics
        """
        if nbeads_return != 1:
            msg = "The possibility to return several beads have not been " + \
                "implemented (yet !)"
            raise NotImplementedError(msg)


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

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        self.write_lammps_input(atoms,
                                atom_style,
                                pair_style,
                                pair_coeff,
                                model_post,
                                nsteps,
                                temp,
                                press)

        lammps_command = f"{self.cmd} -partition {self.nbeads}x1 -in " + \
            f"{self.lammpsfname} -sc out.lmp"
        lmp_handle = run(lammps_command,
                         shell=True,
                         cwd=self.workdir,
                         stderr=PIPE)

        if lmp_handle.returncode != 0:
            msg = "LAMMPS stopped with the exit code \n" + \
                  f"{lmp_handle.stderr.decode()}"
            raise RuntimeError(msg)

        if charges is not None:
            init_charges = atoms.get_initial_charges()
        fname = "configurations.out"
        if self.nbeads > 1:
            ndigit = len(str(self.nbeads))
            # Will be changed for the full PIMD simulations
            fname += f"_{1:0{ndigit}d}"
        atoms = read(f"{self.workdir}{fname}")
        if charges is not None:
            atoms.set_initial_charges(init_charges)

        return atoms.copy()

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

        input_string = "#####################################\n"
        input_string += "#      Thermostat/Integrator\n"
        input_string += "#####################################\n"
        input_string += "timestep      {0}\n".format(self.dt / 1000)

        fix = "fix f1 all pimd/langevin "
        # The temperature
        fix += f"temp {temp} "
        # The integrator
        fix += f"integrator {self.integrator} "
        # Then the thermostat
        fix += f"thermostat PILE_L {self.rng.integers(99999999)} "
        # The damping parameter
        fix += f"tau  {damp} "
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
            fix += f"taup {pdamp} "

        fix += "\n"
        input_string += fix
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_temperature(self):
        """
        Return the temperature of the state
        """
        return self.temperature
