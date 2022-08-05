"""
// (c) 2021 Aloïs Castellano
// This code is licensed under MIT license (see LICENSE.txt for details)
"""
from subprocess import call

from ase.io import read
from ase.io.lammpsdata import write_lammps_data

from mlacs.state.lammps_state import LammpsState
from mlacs.utilities import get_elements_Z_and_masses


# ========================================================================== #
# ========================================================================== #
class PimdLammpsState(LammpsState):
    """
    Parent class for the PIMD states using LAMMPS

    Parameters
    ----------
    nbeads: :class:`int`
        Number of replica of the system
    temperature: :class:`float`
        Temperature of the simulation, in Kelvin.
    pressure: :class:`float` or ``None``
        Pressure of the simulation, in GPa.
        If ``None``, no barostat is applied and
        the simulation is in the NVT ensemble. Default ``None``.
    damp: :class:`float` or ``None``
        Damping parameter for the thermostat.
        If ``None``, apply a damping parameter of
        100 times the timestep of the simulation. Default ``None``.
    langevin: :class:`Bool`
        If ``True``, a Langevin thermostat is used for the thermostat.
        Default ``True``.
    gjf: ``no`` or ``vfull`` or ``vhalf``
        Whether to use the Gronbech-Jensen/Farago integrator
        for the Langevin dynamics. Only apply if langevin is ``True``.
        Default ``vhalf``.
    pdamp: :class:`float` or ``None``
        Damping parameter for the barostat.
        If ``None``, apply a damping parameter of 1000 times
        the timestep of the simulation. Default ``None``.
    ptype: ``iso`` or ``aniso``
        Handle the type of pressure applied. Default ``iso``.
    dt : :class:`float` (optional)
        Timestep, in fs. Default ``1.5`` fs.
    nsteps : :class:`int` (optional)
        Number of MLMD steps for production runs.
        Default ``1000`` steps.
    nsteps_eq : :class:`int` (optional)
        Number of MLMD steps for equilibration runs.
        Default ``100`` steps.
    fixcm : :class:`Bool` (optional)
        Fix position and momentum center of mass. Default ``True``.
    neighbourlist: :class:`int`
        Frequency (in timesteps) at which the neighbour list is updated
        during the PIMD runs. Default ``100``.
    nprocperbead: :class:`int`
        Number of process per replica. Should be a number such that
        nbeds*nprocperbead=N, where N is the max number of process
        (put with the -n or -np variable of mpirun). Default ``1``.
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
    """
    def __init__(self,
                 nbeads,
                 temperature,
                 pressure=None,
                 damp=None,
                 langevin=True,
                 gjf="vhalf",
                 pdamp=None,
                 ptype="iso",
                 dt=1,
                 nsteps=1000,
                 nsteps_eq=100,
                 fixcm=True,
                 neighbourlist=100,
                 nprocperbead=1,
                 logfile=None,
                 trajfile=None,
                 loginterval=50,
                 rng=None,
                 workdir=None):
        LammpsState.__init__(self,
                             temperature,
                             pressure,
                             damp,
                             langevin,
                             gjf,
                             pdamp,
                             ptype,
                             dt,
                             nsteps,
                             nsteps_eq,
                             fixcm,
                             logfile,
                             trajfile,
                             loginterval,
                             rng,
                             None,
                             workdir)

        self.ispimd = True

        self.nbeads = nbeads
        self.neighbourlist = neighbourlist
        self.nprocperbead = nprocperbead

# ========================================================================== #
    def get_nbeads(self):
        """
        Return the number of beads of the state
        """
        return self.nbeads

# ========================================================================== #
    def get_temperature(self):
        """
        Return the temperature of the state
        """
        return self.temperature

# ========================================================================== #
    def run_dynamics(self, atoms, pair_style, pair_coeff, eq=False):
        """
        Function to run the dynamics
        """
        for ibead, at in enumerate(atoms):
            atomsfname = self.workdir + "atoms_{0}.in".format(ibead+1)
            write_lammps_data(atomsfname, at, velocities=True)

        if eq:
            nsteps = self.nsteps_eq
        else:
            nsteps = self.nsteps

        self.write_lammps_input(atoms, pair_style, pair_coeff, nsteps)
        lammps_command = self.cmd + f" -partition {self.nbeads}x" + \
            "{self.nprocperbead} "
        lammps_command += "-in " + self.lammpsfname + "> log"
        call(lammps_command, shell=True, cwd=self.workdir)

        atoms = []
        for ibead in range(self.nbeads):
            atoms.append(read(self.workdir + f"configurations_{ibead+1}.out"))
        return atoms

# ========================================================================== #
    def write_lammps_input(self, atoms, pair_style, pair_coeff, nsteps):
        """
        Write the LAMMPS input for the MD simulation
        """
        elem, Z, masses, charges = get_elements_Z_and_masses(atoms[0])
        pbc = atoms[0].get_pbc()

        input_string = ""
        input_string += self.get_general_input(pbc, masses)
        input_string += self.get_interaction_input(pair_style, pair_coeff)
        input_string += self.get_thermostat_input()
        if self.logfile is not None:
            input_string += self.get_log_input()
        if self.trajfile is not None:
            input_string += self.get_traj_input(elem)

        input_string += self.get_last_dump_input(elem, nsteps)
        input_string += "run  {0}".format(nsteps)

        with open(self.lammpsfname, "w") as f:
            f.write(input_string)

# ========================================================================== #
    def get_thermostat_input(self):
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
        input_string += "fix    f1  all rpmd {0}\n".format(self.temperature)
        if self.pressure is None:
            if self.langevin:
                input_string += "fix    f2  all langevin " + \
                                f"{self.temperature} {self.temperature} " + \
                                f"{damp} {self.rng.integers(999999)}" + \
                                f"${{ibead}}  gjf {self.gfj} zero yes\n"
                input_string += "fix    f3  all nve\n"
            else:
                input_string += "fix    f2  all nvt temp " + \
                                f"{self.temperature} {self.temperature} " + \
                                f"{damp}\n"
        else:
            if self.langevin:
                input_string += "fix    f2  all langevin " + \
                                f"{self.temperature} {self.temperature} " + \
                                f"{damp} {self.rng.integers(999999)}" + \
                                f"${{ibead}}  gjf {self.gfj} zero yes\n"
                input_string += f"fix    f3  all nph  {self.ptype} " + \
                                f"{self.pressure*10000} " + \
                                f"{self.pressure*10000} {pdamp}\n"
            else:
                input_string += "fix    f2  all npt temp " + \
                                f"{self.temperature} {self.temperature}  " + \
                                f"{damp} {self.ptype} " + \
                                f"{self.pressure*10000} " + \
                                f"{self.pressure*10000} {pdamp}\n"
        if self.fixcm:
            input_string += "fix    fcm all recenter INIT INIT INIT\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_log_input(self):
        """
        Function to write the log of the mlmd run
        """
        input_string = "#####################################\n"
        input_string += "#          Logging\n"
        input_string += "#####################################\n"
        input_string += "variable    t equal step\n"
        input_string += "variable    mytemp equal temp\n"
        input_string += "variable    mype equal pe\n"
        input_string += "variable    myke equal ke\n"
        input_string += "variable    myetot equal etotal\n"
        input_string += "variable    mypress equal press/10000\n"
        input_string += "variable    mylx  equal lx\n"
        input_string += "variable    myly  equal ly\n"
        input_string += "variable    mylz  equal lz\n"
        input_string += "variable    vol   equal (lx*ly*lz)\n"
        input_string += "variable    mypxx equal pxx/(vol*10000)\n"
        input_string += "variable    mypyy equal pyy/(vol*10000)\n"
        input_string += "variable    mypzz equal pzz/(vol*10000)\n"
        input_string += "variable    mypxy equal pxy/(vol*10000)\n"
        input_string += "variable    mypxz equal pxz/(vol*10000)\n"
        input_string += "variable    mypyz equal pyz/(vol*10000)\n"

        input_string += 'fix mythermofile all print ' + \
                        f'{self.loginterval} "$t ${{myetot}} ' + \
                        '${mype} ${myke} ${mytemp}  ${mypress} ' + \
                        '${mypxx} ${mypyy} ${mypzz} ${mypxy} ' + \
                        '${mypxz} ${mypyz}" append ' + \
                        f'{self.logfile}_${{ibead}} title "# Step  ' + \
                        'Etot  Epot  Ekin  Press  Pxx  Pyy  Pzz  ' + \
                        'Pxy  Pxz  Pyz"\n'
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_traj_input(self, elem):
        """
        Function to write the dump of the mlmd run
        """
        input_string = "#####################################\n"
        input_string += "#           Dumping\n"
        input_string += "#####################################\n"
        input_string += f"dump dum1 all custom {self.loginterval} " + \
                        f"{self.trajfile}_${{ibead}} id type xu yu zu " + \
                        "vx vy vz fx fy fz element \n"
        input_string += "dump_modify dum1 append yes\n"
        input_string += "dump_modify dum1 element "  # Add element type
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_general_input(self, pbc, masses):
        """
        Function to write the general parameters in the input
        """
        input_string = "# LAMMPS input file " + \
                       "to run a MLPIMD simulation for MLACS\n"
        input_string += "#####################################\n"
        input_string += "#           General parameters\n"
        input_string += "#####################################\n"
        input_string += "units          metal\n"
        input_string += "atom_modify    map array\n"
        input_string += "boundary       " + \
            "{0} {1} {2}\n".format(*tuple("sp"[int(x)] for x in pbc))
        input_string += f"variable       ibead uloop {self.nbeads}\n"
        input_string += "neigh_modify   delay 0 every " + \
            f"{self.neighbourlist} check no\n"
        input_string += "atom_style     atomic\n"
        input_string += "read_data      atoms_${ibead}.in\n"
        for i, mass in enumerate(masses):
            input_string += "mass              " + str(i + 1) + \
                "  " + str(mass) + "\n"
        input_string += "#####################################\n"
        input_string += "\n\n\n"
        return input_string

# ========================================================================== #
    def get_last_dump_input(self, elem, nsteps):
        """
        Function to write the dump of the last configuration of the mlmd
        """
        input_string = "#####################################\n"
        input_string += "#         Dump last step\n"
        input_string += "#####################################\n"
        input_string += f"dump last all custom {nsteps} " + \
                        "configurations_${{ibead}} id type xu yu zu " + \
                        "vx vy vz fx fy fz element \n"
        input_string += "dump_modify last element "
        input_string += " ".join([p for p in elem])
        input_string += "\n"
        input_string += "dump_modify last delay {0}\n".format(nsteps)
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

        msg = "Path Integral Molecular Dynamics\n"
        if self.pressure is None:
            msg += "NVT dynamics as implemented in LAMMPS\n"
        else:
            msg += "NPT dynamics as implemented in LAMMPS\n"
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
